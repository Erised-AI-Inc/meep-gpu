"""Laptop contracts for Phase B — the fold under complex storage and under beta.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware
— bit-identity against the array path — lives in
``parity/meep_gpu/gate_triton_folded_complex.py``, because a byte comparison is
only meaningful on the device that runs it.

What is pinned here is everything that decides whether the four kernels are ever
ALLOWED to run, the constants they hard-code, and the ONE arithmetic claim the
whole tranche turns on:

* the four boundary codes, ``MIRROR_SOURCE_INDEX`` and ``TARGET_IYEE`` are
  RESTATED in ``folded_complex.py`` and pinned equal to BOTH ``symmetry.py``'s
  and ``kernels.py``' spellings — read off SOURCE, so the pin holds on a machine
  with no Triton;
* the ``MIRROR_METALLIC`` / ``MIRROR_PERIODIC`` classification and the storage
  rule it rests on (``stored == owned + 1`` on a folded periodic axis at BOTH
  count parities, ``stored == owned`` on a folded metallic one);
* the reflect row — ``stored - 2`` at an even full count, ``stored - 3`` at an
  odd one — read from ``stepping._far_reflect_rows``, never recomputed;
* THE HEADLINE: on complex64 the array path's ``phase * plane`` is a FULL complex
  product, and a plane-wise ``±1 * word`` diverges from it in real, engineered
  words — measured here, on NumPy, including at the EVEN mirror every corpus row
  uses;
* the beta insert's SLOT (after the dtdx curl, before BOTH ownership masks),
  asserted against the kernel source, because under a fold that slot is
  load-bearing rather than conventional;
* ``PH*`` must be 0 on every folded axis, which the predicate refuses and the
  kernel does not check;
* the coupling to ``symmetry.mirror_ghost_fill_coverage``, which this tranche
  reuses for the REAL family's fill and does not own.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import pathlib
import sys

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import IYEE_SHIFTS, Fields, mirror_parity
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import symmetry
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

MODULE_NAME = "meep_gpu.triton_kernels.folded_complex"
PACKAGE_DIR = pathlib.Path(symmetry.__file__).parent
PARITY_DIR = pathlib.Path(__file__).resolve().parents[1] / "parity" / "meep_gpu"
GATE_PATH = PARITY_DIR / "gate_triton_folded_complex.py"
COMPOSITION_PATH = PARITY_DIR / "probe_triton_folded_complex_composition.py"
SLURM_PATH = PARITY_DIR / "run_triton_folded_complex.slurm"
VALIDATOR_PATH = PARITY_DIR / "validate_folded_complex_reference_vs_stepping.py"

#: The corpus anchors' own numbers, so the tests exercise the values the unlock
#: claim is about (all three are MIRROR_PERIODIC at an EVEN count, phase +1).
BETA_EIGSRC = 0.2                        # test_special_kz.test_eigsrc_kz complex
BETA_GRATING_13_2 = -0.6850526103319672  # test_binary_grating_special_kz 13.2
BETA_GRATING_17_7 = -0.9120991827764708  # ... and 17.7
KX_GRATING_13_2 = 2.9207367086
KX_GRATING_17_7 = 2.8579844438


@pytest.fixture(scope="module")
def fc():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


def _probe_record(value: str = "FMA_V1", backend: str = "cupy",
                  patterns: str = "parity"):
    module = importlib.import_module(MODULE_NAME)
    names = {"base": module.PROBE_PATTERNS,
             "parity": module.PARITY_PROBE_PATTERNS,
             "beta": module.FOLDED_BETA_PROBE_PATTERNS}[patterns]
    return stamp_probe_record(
        {"backend": backend, "patterns": {name: value for name in names}})


def _all_patterns_record(value: str = "FMA_V1"):
    """One artifact carrying every pattern set this tranche can ask for."""
    module = importlib.import_module(MODULE_NAME)
    names = set(module.PARITY_PROBE_PATTERNS) | set(module.FOLDED_BETA_PROBE_PATTERNS)
    return stamp_probe_record(
        {"backend": "cupy",
         "patterns": {name: value for name in sorted(names)}})


def _fold_grid(axis: str = "Y", extent: float = 2.0, phase: int = 1,
               boundaries: str = "periodic", complex_storage: bool = True,
               beta: float = 0.0, k_point=(0.0, 0.0, 0.0), courant: float = 0.35,
               dimensions: int = 2, other: float = 1.6, pml=None, **kwargs):
    """A folded Grid/Fields/PML triple on NumPy, complex or real storage.

    Default: the ``eigsrc_0_complex`` class — a folded PERIODIC Y at an even full
    count with an even plane, complex64 storage, an active PML on the folded
    axis's HIGH face only (the low face is refused by
    ``stepping._require_consistent_pml``).
    """
    size = [other, other, 0.0] if dimensions == 2 else [other, other, other]
    size["XYZ".index(axis)] = extent
    # A 2-D grid's invariant z axis carries NO boundary condition at all (MEEP
    # does not loop over it), so a blanket "metallic" is refused at construction:
    # the declaration is applied per axis, to the folded one.
    declared = ({axis.lower(): boundaries} if dimensions == 2 and boundaries != "periodic"
                else boundaries)
    grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=dimensions,
                courant=courant, boundaries=declared,
                symmetry=(Mirror(axis, phase),), k_point=k_point, beta=beta,
                xp=np, **kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    if pml is None:
        # Per-axis thickness: no layer on the invariant axis, and no LOW-face
        # layer on the folded axis (the mirror plane sits there).
        thickness = []
        for index in range(3):
            if grid.shape[index] < 6:
                thickness.append((0, 0))
            elif index == "XYZ".index(axis):
                thickness.append((0, 2))
            else:
                thickness.append((2, 2))
        pml = tuple(thickness)
    return fields, PML(grid=grid, thickness=pml)


def _reasons(verdict):
    return [r for r in verdict.reasons if "array module" not in r]


def code_of(path: pathlib.Path) -> str:
    """A module's source with comments and docstrings removed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def kernel_source(name: str) -> str:
    """One kernel's source MINUS its docstring, read from the FILE.

    Read from the file rather than through ``inspect`` because with Triton absent
    the kernels degrade to :class:`_UnavailableKernel` instances carrying no
    ``fn`` — and these transcription assertions must hold on exactly the machine
    that has no Triton. The docstring is dropped because it QUOTES the spellings
    these tests forbid ("widened from ``== METALLIC``", "do not write
    ``PHASE * word``"), and a prose mention is not a transcription.
    """
    path = PACKAGE_DIR / "folded_complex.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body[1:] if ast.get_docstring(node) else node.body
            segments = [ast.get_source_segment(text, statement) for statement in body]
            signature = text[:node.body[0].col_offset]  # unused; kept explicit
            del signature
            header = "\n".join(
                text.splitlines()[node.lineno - 1:node.body[0].lineno - 1])
            return header + "\n" + "\n".join(s for s in segments if s)
    raise AssertionError(f"{name} is not defined at module scope in folded_complex.py")


def kernel_parameters(name: str):
    """One kernel's parameter names, from the file, for the same reason."""
    path = PACKAGE_DIR / "folded_complex.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = node.args
            return [a.arg for a in list(args.args) + list(args.kwonlyargs)]
    raise AssertionError(f"{name} is not defined at module scope in folded_complex.py")


class FieldsShim:
    """A ``Fields`` proxy with ONE attribute overridden.

    ``Fields`` exposes its media flags as read-only properties, so a refusal test
    cannot monkeypatch them onto the instance. A proxy is not a weaker test: the
    predicates read every one of these through ``getattr``, which is exactly what
    the proxy answers.
    """

    def __init__(self, fields, **overrides):
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_overrides", dict(overrides))

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_fields"), name)


class GridShim:
    """The same device for ``Grid``, used where the Grid REFUSES the pairing at
    construction — a clause that exists so the refusal is NAMED rather than left
    implied by another object's guard."""

    def __init__(self, grid, **overrides):
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_overrides", dict(overrides))

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_grid"), name)


# ---------------------------------------------------------------------------
# The optional dependency must stay optional, and dispatch must stay untouched
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_this_module_eagerly():
    for name in list(sys.modules):
        if name == MODULE_NAME:
            del sys.modules[name]
    importlib.import_module("meep_gpu.triton_kernels")
    eagerly_imported = MODULE_NAME in sys.modules
    importlib.import_module(MODULE_NAME)  # restore for the other tests
    assert not eagerly_imported, (
        "the package __init__ must not pull this unwired module in")


def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    """A missing optional dependency must not break the engine — or this module."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in (MODULE_NAME, "meep_gpu.triton_kernels.special_kz",
                 "meep_gpu.triton_kernels.complex_fields",
                 "meep_gpu.triton_kernels.kernels"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module(MODULE_NAME)
    fields, pml = _fold_grid()
    assert module.explain_folded_complex(fields, pml).reasons  # answers, not raises
    assert module.plan_folded_complex_pml_curl(fields, pml, "step_B") is None
    assert module.plan_folded_mirror_ghost_fill_complex(fields, "B") is None
    assert module.plan_folded_beta_pml_curl(fields, pml, "step_B") is None
    assert module.plan_folded_beta_bloch_pml_curl(fields, pml, "step_B") is None


def test_the_kernels_fail_clearly_at_the_launch_expression_without_triton(fc):
    if not isinstance(fc.folded_bloch_pml_curl_step, fc._UnavailableKernel):
        pytest.skip("triton is installed here; the absence arm is exercised on "
                    "the laptop box without it")
    for kernel in (fc.folded_bloch_pml_curl_step,
                   fc.folded_mirror_ghost_fill_complex,
                   fc.folded_beta_pml_curl_step,
                   fc.folded_beta_bloch_pml_curl_step):
        with pytest.raises(ImportError, match="triton"):
            kernel[(1,)]


def test_this_module_touches_neither_dispatch_nor_the_other_tracks_files():
    """File ownership, enforced rather than agreed. Dispatch stays deferred."""
    code = code_of(PACKAGE_DIR / "folded_complex.py")
    assert "fastpath" not in code
    assert "cuda_kernels" not in code
    assert "fingerprints" not in code
    assert "plan_step" not in code
    # symmetry.py is RESTATED, never imported: importing it would make this
    # file's constants a function of another tranche's phrasing.
    assert "from .symmetry" not in code and "import symmetry" not in code


def test_the_contraction_guard_is_the_packages_one_spelling():
    """Every launch site passes the shared constant, never a literal.

    A launch site that omits ``enable_fp_fusion`` compiles a kernel that is
    bit-wrong but numerically plausible — one that passes a carelessly written
    gate outright at dtdx = 0.5.
    """
    spelling = "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)"
    code = code_of(PACKAGE_DIR / "folded_complex.py")
    assert code.count("enable_fp_fusion=") == 4
    assert code.count(spelling) == 4


def test_the_gate_its_composition_probe_and_the_reference_validator_exist():
    for path in (GATE_PATH, COMPOSITION_PATH, VALIDATOR_PATH):
        assert path.exists(), f"{path} is part of this tranche's deliverable"


# ---------------------------------------------------------------------------
# The restated constants, against BOTH originators
# ---------------------------------------------------------------------------

def test_the_boundary_codes_agree_with_symmetry_and_with_the_shipped_kernels(fc):
    """Read off SOURCE rather than imported, so this holds with no Triton.

    A plan built here and a plan built in ``symmetry.py`` index the same table;
    a disagreement is a silently wrong ghost rule on every axis.
    """
    kernels_source = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    assert f"PERIODIC = tl.constexpr({fc.CODE_PERIODIC})" in kernels_source
    assert f"METALLIC = tl.constexpr({fc.CODE_METALLIC})" in kernels_source
    assert symmetry.CODE_PERIODIC == fc.CODE_PERIODIC
    assert symmetry.CODE_METALLIC == fc.CODE_METALLIC
    assert symmetry.CODE_MIRROR_METALLIC == fc.CODE_MIRROR_METALLIC
    assert symmetry.CODE_MIRROR_PERIODIC == fc.CODE_MIRROR_PERIODIC
    assert len({fc.CODE_PERIODIC, fc.CODE_METALLIC, fc.CODE_MIRROR_METALLIC,
                fc.CODE_MIRROR_PERIODIC}) == 4


def test_the_mirror_source_index_is_steppings(fc):
    assert fc.MIRROR_SOURCE_INDEX == stepping.MIRROR_SOURCE_INDEX
    assert fc.MIRROR_SOURCE_INDEX == symmetry.MIRROR_SOURCE_INDEX


def test_the_yee_shift_table_is_the_engines(fc):
    """``TARGET_IYEE`` decides which plane each mask and each fill touches."""
    for name, shifts in fc.TARGET_IYEE.items():
        assert tuple(IYEE_SHIFTS[name]) == tuple(shifts), name
    assert fc.TARGET_IYEE == symmetry.TARGET_IYEE
    assert set(fc.TARGET_IYEE) == {"Bx", "By", "Bz", "Dx", "Dy", "Dz"}


def test_the_ghost_fill_families_are_symmetrys(fc):
    assert {family: tuple(spec["targets"])
            for family, spec in fc.GHOST_FILL_FAMILIES.items()} == {
        family: tuple(spec["targets"])
        for family, spec in symmetry.GHOST_FILL_FAMILIES.items()}


def test_the_boundary_kind_names_are_steppings_own_strings(fc):
    """``_boundary_kinds`` may report exactly these three for a Cartesian grid."""
    fields, pml = _fold_grid()
    kinds = stepping._boundary_kinds(fields.grid, pml)
    assert set(kinds) <= set(fc.BOUNDARY_KIND_NAMES)


# ---------------------------------------------------------------------------
# The fold classification and the storage rule it rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extent,expected_reflect_offset", [
    (2.0, 2),   # n_full 20 EVEN -> reflect = stored - 2
    (2.1, 3),   # n_full 21 ODD  -> reflect = stored - 3
    (2.2, 2),
    (2.3, 3),
])
def test_a_folded_periodic_axis_stores_one_past_owned_at_both_parities(
        fc, extent, expected_reflect_offset):
    fields, pml = _fold_grid(extent=extent)
    grid = fields.grid
    assert grid.stored_cells(1) == grid.owned_cells(1) + 1
    codes, reasons = fc.folded_axis_kinds(grid, pml)
    assert reasons == ()
    assert codes[1] == fc.CODE_MIRROR_PERIODIC
    row = stepping._far_reflect_rows(grid)[1]
    assert row == grid.stored_cells(1) - expected_reflect_offset


@pytest.mark.parametrize("extent", [2.0, 2.1])
def test_a_folded_metallic_axis_stores_exactly_owned_and_has_no_reflect_row(
        fc, extent):
    fields, pml = _fold_grid(extent=extent, boundaries="metallic")
    grid = fields.grid
    assert grid.stored_cells(1) == grid.owned_cells(1)
    codes, reasons = fc.folded_axis_kinds(grid, pml)
    assert reasons == ()
    assert codes[1] == fc.CODE_MIRROR_METALLIC
    assert stepping._far_reflect_rows(grid)[1] is None


def test_one_grid_can_carry_both_reflect_row_arms_at_once(fc):
    """Mirror(X)+Mirror(Y) at n_full (20, 21): stored-2 on x, stored-3 on y."""
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), boundaries="periodic",
                symmetry=(Mirror("X", 1), Mirror("Y", 1)), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((0, 2), (0, 2), (2, 2)))
    codes, reasons = fc.folded_axis_kinds(grid, pml)
    assert reasons == ()
    assert codes[0] == codes[1] == fc.CODE_MIRROR_PERIODIC
    rows = stepping._far_reflect_rows(grid)
    assert rows[0] == grid.stored_cells(0) - 2
    assert rows[1] == grid.stored_cells(1) - 3
    del fields


def test_the_classification_refuses_when_the_two_routes_disagree(fc, monkeypatch):
    """``_stored_past_owned`` and ``is_metallic`` are independent routes to the
    same fact; a disagreement means one has drifted and NEITHER may be trusted."""
    fields, pml = _fold_grid()
    monkeypatch.setattr(type(fields.grid), "is_metallic",
                        lambda self, axis: axis == 1, raising=False)
    codes, reasons = fc.folded_axis_kinds(fields.grid, pml)
    assert codes is None
    assert any("disagree about the fold's termination" in r for r in reasons)


def test_the_classification_refuses_a_fold_that_cannot_reflect(fc, monkeypatch):
    fields, pml = _fold_grid()
    monkeypatch.setattr(type(fields.grid), "stored_cells",
                        lambda self, axis: 2 if axis == 1 else 8, raising=False)
    codes, reasons = fc.folded_axis_kinds(fields.grid, pml)
    assert codes is None
    assert any("stored cells" in r for r in reasons)


def test_the_classification_refuses_an_illegal_mirror_phase(fc, monkeypatch):
    fields, pml = _fold_grid()
    monkeypatch.setattr(type(fields.grid), "mirror_phase",
                        lambda self, axis: 0 if axis == 1 else None, raising=False)
    codes, reasons = fc.folded_axis_kinds(fields.grid, pml)
    assert codes is None
    assert any("not +1 or -1" in r for r in reasons)


# ---------------------------------------------------------------------------
# THE HEADLINE: the parity multiply is a full complex product, measured
# ---------------------------------------------------------------------------

ENGINEERED_WORDS = (0.0, -0.0, 1e-45, -1e-45, 7e-45, 1.5, -1.5, 3.4e38)


def _engineered_complex_plane():
    """64 engineered word PAIRS = 128 words, over the classes the fill can hit.

    Signed zeros and subnormals are the whole point: a random-data sweep is
    PROVABLY blind to the class this tranche's needle lives in.

    PLANE ASSIGNMENT, never ``re + 1j*im`` — the same rule
    ``gate_triton_complex._interleave_c8`` exists to enforce, and for the same
    reason: the ADDITION destroys signed zeros (``-0.0 + (+0.0)`` is ``+0.0``)
    and the signed-zero rows ARE the discriminating vectors. Measured on this
    laptop: the ``re + 1j*im`` build of this exact word set carries 3 negative-zero
    words and ZERO negative-zero IMAGINARY words, against 16 and 8 for the
    sign-preserving build — and with the imaginary signs gone the headline needle
    collapsed to 2/128 at phase +1 and 13/128 at phase -1, contradicting the
    shipped module's own 8/128-at-BOTH-parities record (folded_complex.py:64-66)
    and the device gate's needle prediction.
    """
    real, imag = np.meshgrid(np.float32(ENGINEERED_WORDS),
                             np.float32(ENGINEERED_WORDS), indexing="ij")
    plane = np.empty(real.shape, dtype=np.complex64)
    plane.real = real.astype(np.float32)
    plane.imag = imag.astype(np.float32)
    return plane


#: LAPTOP-MEASURED (NumPy 2.x, this engineered set): words in which a plane-wise
#: ``±1 * word`` fill diverges from the array path's complex product. The
#: even-mirror row is the load-bearing one — it is the mirror EVERY corpus row
#: uses, and plane-wise looks like the identity there. This is the module's own
#: headline number (folded_complex.py:64-66) and the device gate's needle
#: prediction (gate_triton_folded_complex.py, plane_wise_parity_fill), so the
#: three agree by construction rather than by coincidence.
PLANE_WISE_DIVERGENCE = {1: 8, -1: 8}


@pytest.mark.parametrize("phase", [1, -1])
def test_the_parity_multiply_is_a_full_complex_product_not_a_sign_flip(phase):
    """THE measurement the whole phase turns on, on the reference NumPy.

    ``stepping._write_mirror_ghost`` spells ``phase * plane`` with a PYTHON INT,
    and NumPy carries only 'FF->F' complex loops — so on complex64 that is the
    FULL multiply by ``(±1.0, +0.0)`` including its zero cross terms. The
    plane-wise emulation below is ``symmetry.mirror_ghost_fill``'s certified REAL
    spelling ported word-for-word to a complex volume, which is exactly the
    ``plane_wise_parity_fill`` gate mutation, and it is byte-different at BOTH
    parities — INCLUDING at the even mirror, where it looks like the identity and
    the array path is not.

    The counts are the LAPTOP's, on this word set, and are pinned as a regression
    rather than quoted from elsewhere: the CuPy answer is the arbiter and the
    gate re-cuts it on device under the stripped IEEE-keep policy.
    """
    plane = _engineered_complex_plane()
    array_path = np.ascontiguousarray(phase * plane)      # what stepping.py does
    words = plane.view(np.float32).copy()                 # the word-pair view
    plane_wise = (np.float32(phase) * words).astype(np.float32)
    a = array_path.view(np.uint32).ravel()
    b = np.ascontiguousarray(plane_wise).view(np.uint32).ravel()
    differing = int((a != b).sum())
    assert a.size == 128
    assert differing == PLANE_WISE_DIVERGENCE[phase], (
        f"the plane-wise fill must diverge from the array path in "
        f"{PLANE_WISE_DIVERGENCE[phase]}/128 engineered words at phase "
        f"{phase:+d}; measured {differing}")
    assert differing > 0


def test_the_python_int_float_and_complex64_spellings_of_the_phase_agree():
    """The coefficient may be host-rounded through numpy.complex64 without
    changing the array path's bytes — which is what licenses passing WORDS."""
    plane = _engineered_complex_plane()
    for phase in (1, -1):
        reference = np.ascontiguousarray(phase * plane).view(np.uint32)
        for spelling in (float(phase), np.complex64(phase)):
            got = np.ascontiguousarray(spelling * plane).view(np.uint32)
            assert np.array_equal(reference, got), (phase, spelling)


def test_the_even_mirror_is_not_the_identity_under_complex_storage():
    """An ``if phase == +1: copy`` shortcut is wrong on the mirror every corpus
    row uses — this is the ``identity_shortcut_on_even_mirror`` gate mutation's
    host-side justification."""
    plane = _engineered_complex_plane()
    a = np.ascontiguousarray(1 * plane).view(np.uint32).ravel()
    b = np.ascontiguousarray(plane).view(np.uint32).ravel()
    assert int((a != b).sum()) == 8, (
        "an `if phase == +1: copy` shortcut must be byte-different from the "
        "array path on the engineered signed-zero set")


def test_the_parity_coefficient_words_are_the_hosts_rounded_bits(fc):
    """near = complex64(+phase), far = complex64(-phase) — the shift-1 parity."""
    for phase in (1, -1):
        near, far = fc.mirror_parity_coefficients(phase)
        assert near == (float(phase), 0.0)
        assert far == (float(-phase), 0.0)
        # ``mirror_parity(c, axis, phase) == phase * (1 - 2*iyee[c][axis])``:
        # near touches shift-0 components only, far shift-1 only.
        for component, shifts in fc.TARGET_IYEE.items():
            for axis in range(3):
                expected = near[0] if shifts[axis] == 0 else far[0]
                assert mirror_parity(component, axis, phase) == expected


def test_an_illegal_declared_phase_is_refused_by_the_coefficient_builder(fc):
    with pytest.raises(ValueError, match=r"\+1 or -1"):
        fc.mirror_parity_coefficients(0)


def test_the_parity_coefficients_imaginary_word_is_bitwise_zero(fc):
    """THE PREMISE the ``synthesize_zero_imag`` RETIREMENT rests on, asserted
    directly rather than inferred from one device run's divergence count.

    ``mirror_parity_coefficients`` (folded_complex.py:2100) rounds ``+phase``
    and ``-phase`` through ``numpy.complex64`` on the host and passes both WORDS
    to the kernel. The gate's ``synthesize_zero_imag`` mutant rewrites the
    kernel's ``near_im`` / ``far_im`` argument to a literal ``0.0``; it was
    MEASURED vacuous on run direct_20260813T084108Z (hits 6, launched 192, PTX
    differs, 0 diverging words over 96 fill rows) for exactly one structural
    reason: the word the host already passes is bitwise ``0x00000000``, so the
    mutant and the shipped kernel receive BIT-IDENTICAL operands and no grid,
    state family or needle can separate them.

    That vacuity is a property of THIS function — not of the mutation, and not
    of the gate's grid list — so this is where it gets pinned. Without this
    test the retirement is a one-time observation and the dead mutation sits in
    the table claiming coverage it cannot deliver; with it, the premise breaking
    fails on a laptop with no GPU, and the gate's null must be revisited.

    ``== 0.0`` is NOT the assertion, and that is the whole point: ``-0.0``
    compares equal to ``0.0`` while carrying the word ``0x80000000``, which the
    substituted literal ``+0.0`` does not reproduce. Under that spelling the
    mutation would be LIVE again and its retirement wrong. The check is on the
    32-bit pattern, at both mirror phases, for both the near and the far
    coefficient, and on the words that actually reach the kernel.
    """
    def word(value) -> int:
        return int(np.float32(value).view(np.uint32))

    substituted = word(0.0)          # the literal the mutant puts in its place
    assert substituted == 0x00000000

    for phase in (1, -1):
        near, far = fc.mirror_parity_coefficients(phase)
        for label, pair in (("near", near), ("far", far)):
            assert word(pair[1]) == 0x00000000, (
                f"the {label} coefficient's imaginary word at phase {phase} is "
                f"0x{word(pair[1]):08x}, not bitwise zero: the "
                f"synthesize_zero_imag mutant no longer receives an operand "
                f"identical to the shipped kernel's, so the gate's recorded "
                f"structural null is FALSE and must be re-armed")
            # ...and the real word is NOT zero, or the mutation would be vacuous
            # for the uninteresting reason that nothing is passed at all.
            assert word(pair[0]) != 0x00000000, (label, phase, pair)
            # The vacuity, spelled as the identity it actually is.
            assert word(pair[1]) == substituted, (label, phase, pair)

    # The same claim on the words the PLAN hands the kernel, at both mirror
    # phases on one grid, since that is the object the mutation meets.
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), boundaries="periodic",
                symmetry=(Mirror("X", 1), Mirror("Y", -1)), xp=np)
    for family in ("B", "D"):
        for entry in fc.ghost_fill_axis_entries(grid, family):
            assert word(entry["near_words"][1]) == substituted, entry
            assert word(entry["far_words"][1]) == substituted, entry


# ---------------------------------------------------------------------------
# The fill entries: order, reflect row, shifts, coefficients
# ---------------------------------------------------------------------------

def test_the_fill_entries_are_in_x_y_z_order_and_carry_both_coefficients(fc):
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), boundaries="periodic",
                symmetry=(Mirror("X", 1), Mirror("Y", -1)), xp=np)
    for family, targets in (("B", ("Bx", "By", "Bz")), ("D", ("Dx", "Dy", "Dz"))):
        entries = fc.ghost_fill_axis_entries(grid, family)
        assert [e["axis"] for e in entries] == [0, 1], (
            "the array path applies the axes in X, Y, Z order; under complex "
            "storage that order is load-bearing, not cosmetic")
        for entry in entries:
            axis = entry["axis"]
            assert entry["far"] is True
            assert entry["reflect_row"] == stepping._far_reflect_rows(grid)[axis]
            assert entry["shifts"] == tuple(fc.TARGET_IYEE[n][axis] for n in targets)
            near, far = fc.mirror_parity_coefficients(entry["phase"])
            assert entry["near_words"] == near
            assert entry["far_words"] == far


def test_a_folded_metallic_axis_gets_no_far_fill(fc):
    fields, _pml = _fold_grid(boundaries="metallic")
    entries = fc.ghost_fill_axis_entries(fields.grid, "D")
    assert len(entries) == 1
    assert entries[0]["far"] is False
    assert entries[0]["reflect_row"] == -1


def test_the_fill_entries_refuse_a_grid_whose_fold_cannot_be_classified(
        fc, monkeypatch):
    fields, _pml = _fold_grid()
    monkeypatch.setattr(type(fields.grid), "is_metallic",
                        lambda self, axis: axis == 1, raising=False)
    with pytest.raises(ValueError, match="cannot be classified"):
        fc.ghost_fill_axis_entries(fields.grid, "B")


# ---------------------------------------------------------------------------
# The transcription, asserted against the kernel source
# ---------------------------------------------------------------------------



@pytest.mark.parametrize("name", ["folded_bloch_pml_curl_step",
                                  "folded_beta_pml_curl_step",
                                  "folded_beta_bloch_pml_curl_step"])
def test_the_cell_zero_mask_is_widened_to_non_periodic_on_every_curl(fc, name):
    """``_mask_non_owned_cells`` asks ``is_mirrored or is_metallic or is_axis``
    (S:1898-1902), which ``_boundary_kinds`` resolves to a non-periodic kind. A
    kernel narrowing this back to ``== METALLIC`` leaves a live curl on the mirror
    plane — the ``mask_only_metallic`` gate mutation."""
    source = kernel_source(name)
    assert "!= PERIODIC" in source
    assert "== METALLIC" not in source, (
        "the folded kernels must not carry the unwidened cell-0 mask")


@pytest.mark.parametrize("name", ["folded_bloch_pml_curl_step",
                                  "folded_beta_pml_curl_step",
                                  "folded_beta_bloch_pml_curl_step"])
def test_the_top_plane_mask_exists_and_is_gated_on_mirror_periodic(fc, name):
    """Nine (side, target, axis) clauses on the complex kernels, nine on the real
    one: the B side masks two axes per target, the D side one."""
    source = kernel_source(name)
    assert source.count("== MIRROR_PERIODIC") == 9
    assert "last_x, last_y, last_z" in source


@pytest.mark.parametrize("name", ["folded_bloch_pml_curl_step",
                                  "folded_beta_pml_curl_step",
                                  "folded_beta_bloch_pml_curl_step"])
def test_only_the_periodic_branch_wraps_the_ghost_index(fc, name):
    """Both mirror codes take the METALLIC ghost branch — mask and serve 0.0.

    A mirror code taking the PERIODIC index-wrap branch is the ``fold_ghost_wraps``
    gate mutation, caught on MIRROR_METALLIC only.
    """
    source = kernel_source(name)
    assert source.count("if BCX == PERIODIC:") == 1
    assert source.count("if BCY == PERIODIC:") == 1
    assert source.count("if BCZ == PERIODIC:") == 1


@pytest.mark.parametrize("name,partner_first,partner_second", [
    ("folded_beta_pml_curl_step", "beta_plus * b", "beta_minus * a"),
    ("folded_beta_bloch_pml_curl_step", "bp_re, bp_im, b_re, b_im",
     "bm_re, bm_im, a_re, a_im"),
])
def test_the_beta_insert_sits_after_the_curl_and_before_both_masks(
        fc, name, partner_first, partner_second):
    """THE fold-order needle. The array path adds the increment after
    ``_curl_from_operands`` and before ``_mask_non_owned_cells`` (S:356-363 then
    :369; S:438-445 then :450), and a fold WIDENS that mask — so an insert below
    the masks leaves a live increment on the mirror plane and on the far ghost
    plane, invisible in the interior."""
    source = kernel_source(name)
    beta_at = source.index("if HAS_BETA:")
    curl_at = source.index("curl0" if "beta_plus" in partner_first else "curl0_re")
    cell_zero_mask_at = source.index("at_x, at_y, at_z")
    top_mask_at = source.index("last_x, last_y, last_z")
    assert curl_at < beta_at < cell_zero_mask_at < top_mask_at
    assert partner_first in source and partner_second in source


def test_the_complex_beta_partners_are_the_unrotated_centre_words(fc):
    """The Bloch rotation applies to SHIFTED operands only; the beta partner is
    the CENTER pair loaded before the phase section and never rotated."""
    source = kernel_source("folded_beta_bloch_pml_curl_step")
    rotate_at = source.index("if PHX:")
    beta_at = source.index("if HAS_BETA:")
    assert rotate_at < beta_at
    for rotated in ("b_x_re", "c_y_re", "a_z_re"):
        assert f"{rotated}, {rotated.replace('_re', '_im')}, b_re" not in source
    assert "_mul_imag_coefficient_left(bp_re, bp_im, b_re, b_im" in source
    assert "_mul_imag_coefficient_left(bm_re, bm_im, a_re, a_im" in source


def test_the_fill_uses_the_passed_coefficient_words_not_a_bare_multiply(fc):
    """``PHASE * word`` is the headline gate needle; the shipped kernel must
    reach the parity through the full complex product helper, and the coefficient
    words must be PASSED, never synthesised as a literal ``+0.0`` in-kernel."""
    source = kernel_source("folded_mirror_ghost_fill_complex")
    assert source.count("_mul_imag_coefficient_left(near_re, near_im") == 3
    assert source.count("_mul_imag_coefficient_left(far_re, far_im") == 3
    assert "PHASE" not in source
    assert "0.0, z_re" not in source and "0.0, z_im" not in source


def test_the_fill_keeps_the_reflect_row_a_runtime_scalar(fc):
    """Baking ``n - 2`` reflects about the window top instead of about the second
    mirror — a whole cell wrong on every ODD-count run."""
    parameters = kernel_parameters("folded_mirror_ghost_fill_complex")
    assert "reflect_row" in parameters
    # A constexpr would be baked into the specialization; it must be a RUNTIME
    # scalar, so it sits among the plain arguments, before the constexpr block.
    assert parameters.index("reflect_row") < parameters.index("AXIS")
    source = kernel_source("folded_mirror_ghost_fill_complex")
    assert "reflect_row * stride" in source
    assert "last - 2" not in source and "nx - 2" not in source


def test_the_fill_is_two_passes_and_the_plan_exposes_them_separately(fc):
    """MEASURED CORRECTION, not a preference. The array path runs
    ``fill_symmetry_bc_*`` and ``fill_folded_far_ghosts_*`` as two separate
    whole-grid passes with ``zero_metal_*`` between them (driver.py:3209-3211 /
    :3222-3224). Fusing them per axis reorders a component unowned on two axes
    with different Yee shifts, and complex multiplication is not associative.
    """
    source = kernel_source("folded_mirror_ghost_fill_complex")
    parameters = kernel_parameters("folded_mirror_ghost_fill_complex")
    assert "DO_NEAR" in parameters and "DO_FAR" in parameters
    assert source.count("if DO_NEAR and S") == 3
    assert "if DO_FAR:" in source
    plan_type = fc.FoldedMirrorGhostFillComplexPlan
    for method in ("run_near", "run_far", "run"):
        assert callable(getattr(plan_type, method))


def test_fusing_the_two_fill_passes_diverges_on_a_two_folded_axis_grid():
    """The measurement behind the two-pass plan, on NumPy.

    ``By`` is FAR on a folded x (Yee shift 1) and NEAR on a folded y (shift 0),
    so the doubly-unowned corner sees near-then-far in the array path and
    far-then-near under a fused per-axis launch. Under REAL storage the two agree
    (±1 * float32 is exact), which is why ``symmetry.MirrorGhostFillPlan`` may
    fuse them; under complex64 they do not.
    """
    fold = importlib.import_module(MODULE_NAME)
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), courant=0.35,
                boundaries="periodic",
                symmetry=(Mirror("X", 1), Mirror("Y", 1)), xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((0, 2), (0, 2), (2, 2)))
    rng = np.random.default_rng(20260804)
    rows = stepping._far_reflect_rows(grid)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"):
        volume = np.empty(grid.shape, dtype=np.complex64)
        volume.real = rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
        volume.imag = rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
        # The needle lives in the signed-zero / subnormal class, and random data
        # is provably blind to it: plant on the two rows the fill READS.
        for axis in (0, 1):
            for row in (2, rows[axis]):
                if row is None:
                    continue
                _plant_words(volume, axis, int(row))
        getattr(fields, name)[...] = volume
    # The plants carry 3.4e38, so the step and the fills legitimately overflow;
    # the BYTES are the measurement and an inf is a perfectly good byte here.
    with np.errstate(over="ignore", invalid="ignore"):
        stepping.step_B(fields, pml)

        names = fold.GHOST_FILL_FAMILIES["B"]["targets"]
        entries = fold.ghost_fill_axis_entries(grid, "B")
        split = {n: np.array(getattr(fields, n), copy=True) for n in names}
        fused = {n: np.array(getattr(fields, n), copy=True) for n in names}
        _fill_two_pass(split, names, entries)
        _fill_fused_per_axis(fused, names, entries)
    diverging = {n: int((np.ascontiguousarray(split[n]).view(np.uint32).ravel()
                         != np.ascontiguousarray(fused[n]).view(np.uint32).ravel()
                         ).sum()) for n in names}
    assert sum(diverging.values()) > 0, (
        "fusing the two fill passes must be byte-visible on a two-folded-axis "
        f"complex grid; measured {diverging}")
    assert diverging["By"] > 0, diverging


def _slab(axis: int, index: int):
    return tuple(index if a == axis else slice(None) for a in range(3))


#: Signed zeros, subnormals and an overflow-adjacent magnitude — the classes a
#: full complex product and a plane-wise sign flip disagree on.
NEEDLE_WORDS = np.float32([0.0, -0.0, 1e-45, -1e-45, 7e-45, 1.5, -1.5, 3.4e38])


def _plant_words(volume: np.ndarray, axis: int, row: int) -> None:
    for plane, offset in ((volume.real[_slab(axis, row)], 0),
                          (volume.imag[_slab(axis, row)], 3)):
        flat = plane.reshape(-1)
        for index in range(flat.size):
            flat[index] = NEEDLE_WORDS[(index + offset) % NEEDLE_WORDS.size]
        plane[...] = flat.reshape(plane.shape)


def _fill_two_pass(arrays, names, entries) -> None:
    """The array path: every NEAR plane, then every FAR plane."""
    fold = importlib.import_module(MODULE_NAME)
    for name in names:
        for entry in entries:
            axis = int(entry["axis"])
            if fold.TARGET_IYEE[name][axis] == 0:
                arrays[name][_slab(axis, 0)] = (
                    mirror_parity(name, axis, int(entry["phase"]))
                    * arrays[name][_slab(axis, 2)])
    for name in names:
        for entry in entries:
            axis = int(entry["axis"])
            if entry["far"] and fold.TARGET_IYEE[name][axis] == 1:
                arrays[name][_slab(axis, -1)] = (
                    mirror_parity(name, axis, int(entry["phase"]))
                    * arrays[name][_slab(axis, int(entry["reflect_row"]))])


def _fill_fused_per_axis(arrays, names, entries) -> None:
    """The fused form ``symmetry.MirrorGhostFillPlan`` uses — correct for REAL
    storage, and the thing this test shows is NOT correct for complex."""
    fold = importlib.import_module(MODULE_NAME)
    for entry in entries:
        axis = int(entry["axis"])
        for name in names:
            shift = fold.TARGET_IYEE[name][axis]
            if shift == 0:
                arrays[name][_slab(axis, 0)] = (
                    mirror_parity(name, axis, int(entry["phase"]))
                    * arrays[name][_slab(axis, 2)])
            elif entry["far"]:
                arrays[name][_slab(axis, -1)] = (
                    mirror_parity(name, axis, int(entry["phase"]))
                    * arrays[name][_slab(axis, int(entry["reflect_row"]))])


def test_the_fill_addresses_word_pairs_and_the_near_source_is_cell_two(fc):
    source = kernel_source("folded_mirror_ghost_fill_complex")
    assert "base + 2 * stride" in source          # MIRROR_SOURCE_INDEX = 2, a CELL
    assert "2 * base" in source and "2 * base + 1" in source
    assert "2 * src" in source and "2 * dst" in source


@pytest.mark.parametrize("name", ["folded_bloch_pml_curl_step",
                                  "folded_beta_bloch_pml_curl_step"])
def test_the_complex_curls_never_spell_a_negated_addend_with_unary_minus(fc, name):
    """Triton 3.1.0 lowers ``-x`` as ``0.0 - x``, which canonicalizes ±0 addends
    to +0 — MEASURED REACHABLE at the driver level (jobs 2343/2345). The helpers
    carry the ``(a*b) * -1.0`` spelling; the kernels must not reintroduce one."""
    source = kernel_source(name)
    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("-"), stripped
        assert " -(" not in stripped, stripped


def test_no_division_appears_in_any_kernel_body(fc):
    """Triton's f32 ``/`` lowers to ``div.full.f32``, APPROXIMATE (~2 ulp); any
    division would have to be ``tl.math.div_rn``. These kernels have none."""
    for name in ("folded_bloch_pml_curl_step", "folded_mirror_ghost_fill_complex",
                 "folded_beta_pml_curl_step", "folded_beta_bloch_pml_curl_step"):
        source = kernel_source(name)
        for line in source.splitlines():
            stripped = line.split("#")[0]
            if "//" in stripped:          # integer plane arithmetic is fine
                stripped = stripped.replace("//", "")
            assert "/" not in stripped, (name, line)


# ---------------------------------------------------------------------------
# Coverage — positive admission, then every named refusal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_folded_complex_configuration_is_refused_only_for_the_backend(
        fc, sub_step):
    """K1's target class: complex64 storage, a live PML, a folded PERIODIC Y at
    k = 0, beta = 0. Only the CuPy clause may fire on NumPy."""
    fields, pml = _fold_grid()
    verdict = fc.folded_complex_pml_curl_coverage(fields, pml, sub_step,
                                                  probe=_all_patterns_record())
    assert _reasons(verdict) == []


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
@pytest.mark.parametrize("boundaries", ["periodic", "metallic"])
@pytest.mark.parametrize("phase", [1, -1])
@pytest.mark.parametrize("extent", [2.0, 2.1])
def test_every_cell_of_the_termination_parity_phase_matrix_is_admitted(
        fc, sub_step, boundaries, phase, extent):
    """The corpus supplies ONE cell (MIRROR_PERIODIC, even count, phase +1); the
    other seven exist only synthetically and must be admitted here, or the gate
    can never measure them."""
    fields, pml = _fold_grid(boundaries=boundaries, phase=phase, extent=extent)
    verdict = fc.folded_complex_pml_curl_coverage(fields, pml, sub_step,
                                                  probe=_all_patterns_record())
    assert _reasons(verdict) == []


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_a_fold_on_each_axis_is_admitted(fc, axis):
    """A folded Z must classify exactly as a folded X does."""
    fields, pml = _fold_grid(axis=axis, dimensions=3, other=1.2)
    verdict = fc.folded_complex_pml_curl_coverage(fields, pml, "step_B",
                                                  probe=_all_patterns_record())
    assert _reasons(verdict) == []


def test_zero_folded_axes_is_admitted_by_the_standalone_verdict_and_refused_by_composition(
        fc):
    """The reduction leg is an EQUIVALENCE product: with nothing folded K1 must be
    byte-identical to the certified unfolded kernel, which the gate MEASURES.
    Routing is a different question, and the composition verdict answers it."""
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                courant=0.35, boundaries="periodic", xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))
    record = _all_patterns_record()
    assert _reasons(fc.folded_complex_pml_curl_coverage(
        fields, pml, "step_B", probe=record)) == []
    assert any("equivalence product" in r for r in _reasons(
        fc.folded_complex_composition_curl_coverage(
            fields, pml, "step_B", probe=record)))


def test_real_storage_is_refused_by_the_complex_curl(fc):
    fields, pml = _fold_grid(complex_storage=False)
    assert any("storage is real float32" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   fields, pml, "step_B", probe=_all_patterns_record())))


def test_a_nonzero_beta_is_refused_by_the_plain_folded_complex_curl(fc):
    fields, pml = _fold_grid(beta=BETA_EIGSRC)
    assert any("belongs to the folded beta kernels" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   fields, pml, "step_B", probe=_all_patterns_record())))


def test_a_zero_beta_is_refused_by_both_beta_curls(fc):
    """Selecting between two valid products by branch order would make the
    numerical method depend on composer order."""
    fields, pml = _fold_grid(beta=0.0)
    assert any("grid.beta is zero" in r
               for r in _reasons(fc.folded_beta_bloch_pml_curl_coverage(
                   fields, pml, "step_B", probe=_all_patterns_record())))
    real_fields, real_pml = _fold_grid(beta=0.0, complex_storage=False)
    assert any("grid.beta is zero" in r
               for r in _reasons(fc.folded_beta_pml_curl_coverage(
                   real_fields, real_pml, "step_B")))


def test_no_active_pml_is_refused(fc):
    fields, pml = _fold_grid(pml=0)
    assert any("no active PML layer" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   fields, pml, "step_B", probe=_all_patterns_record())))


def test_a_pml_on_the_low_face_of_a_folded_axis_never_reaches_the_predicate():
    """A PML on the mirror plane's own face is refused BEFORE a plan can exist.

    ``PML._resolve_mirror_faces`` raises at construction, so the configuration
    the spec names ("a PML on the LOW face of a folded axis") cannot be handed to
    a predicate at all — a refusal, not a crash, and one this tranche inherits
    rather than restates. Pinned here so a future PML change that silently admits
    it is caught by a test rather than by a plane of wrong values.
    """
    with pytest.raises(ValueError, match="mirror-symmetric"):
        _fold_grid(pml=((2, 2), (2, 2), (0, 0)))


@pytest.mark.parametrize("k", [0.3, 0.5])
def test_a_bloch_k_on_the_folded_axis_is_refused_zone_edge_included(fc, k):
    """``driver._require_bloch_is_representable`` refuses the zone INTERIOR and
    the zone EDGE alike; a kernel cannot lift what the array path will not run.

    ``Grid`` itself refuses the pairing at construction, so the configuration is
    reachable only through a shim — which is precisely why the clause is written
    out here rather than inferred from another object's guard: the refusal must
    be NAMED, not implied.
    """
    fields, pml = _fold_grid()
    with pytest.raises(ValueError, match="requires k_point"):
        _fold_grid(k_point=(0.0, k, 0.0))
    shim = FieldsShim(fields, grid=GridShim(fields.grid, k_point=(0.0, k, 0.0)))
    reasons = _reasons(fc.folded_complex_pml_curl_coverage(
        shim, pml, "step_B", probe=_all_patterns_record()))
    assert any("folded with k component" in r for r in reasons), reasons


def test_a_bloch_k_on_an_UNFOLDED_periodic_axis_is_admitted(fc):
    """The two binary-grating rows carry an in-plane kx on the PML'd X axis
    beside a folded Y — the phases compose across DIFFERENT axes."""
    fields, pml = _fold_grid(k_point=(0.21, 0.0, 0.0))
    assert _reasons(fc.folded_complex_pml_curl_coverage(
        fields, pml, "step_B", probe=_all_patterns_record())) == []


def test_real_storage_with_a_nonzero_k_point_is_refused_by_the_real_beta_curl(fc):
    fields, pml = _fold_grid(complex_storage=False, beta=BETA_EIGSRC,
                             k_point=(0.21, 0.0, 0.0))
    reasons = _reasons(fc.folded_beta_pml_curl_coverage(fields, pml, "step_B"))
    assert any("k_point" in r for r in reasons)


def test_a_k_component_on_the_invariant_axis_is_refused_by_the_beta_curls(fc):
    """beta IS the analytic z dependence; a planted k_point[2] would ride the
    axis beta already carries."""
    fields, pml = _fold_grid(beta=BETA_EIGSRC)
    shim = FieldsShim(fields,
                      grid=GridShim(fields.grid, k_point=(0.0, 0.0, 0.37)))
    reasons = _reasons(fc.folded_beta_bloch_pml_curl_coverage(
        shim, pml, "step_B", probe=_all_patterns_record()))
    assert any("invariant axis" in r for r in reasons), reasons


@pytest.mark.parametrize("flag,needle", [
    ("has_nonlinearity", "chi2/chi3"),
    ("has_offdiagonal_epsilon", "off-diagonal chi1inv"),
    ("has_magnetic_conductivity", "magnetic (B) conductivity"),
])
def test_the_media_refusals_fire_by_name(fc, flag, needle):
    fields, pml = _fold_grid()
    reasons = _reasons(fc.folded_complex_pml_curl_coverage(
        FieldsShim(fields, **{flag: True}), pml, "step_B",
        probe=_all_patterns_record()))
    assert any(needle in r for r in reasons), reasons


def test_a_conductivity_on_a_curl_target_is_refused(fc):
    fields, pml = _fold_grid()
    shim = FieldsShim(fields,
                      condfac_for=lambda name: object() if name == "Dx" else None)
    assert any("conductivity is installed on Dx" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   shim, pml, "step_B", probe=_all_patterns_record())))


def test_an_unreadable_conductivity_table_is_refused_outright(fc):
    """Inferring "no conductivity" from the ABSENCE of ``condfac_for`` is
    admission by attribute absence — the exact reasoning coverage exists to
    refuse."""
    fields, pml = _fold_grid()
    assert any("does not expose condfac_for" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   FieldsShim(fields, condfac_for=None), pml, "step_B",
                   probe=_all_patterns_record())))


def test_bfast_is_refused(fc):
    fields, pml = _fold_grid()
    shim = FieldsShim(fields, grid=GridShim(fields.grid, bfast_active=True))
    assert any("BFAST is active" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   shim, pml, "step_B", probe=_all_patterns_record())))


def test_a_registered_susceptibility_is_refused_under_complex_storage(fc):
    fields, pml = _fold_grid()
    assert any("susceptibility is registered" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   FieldsShim(fields, has_polarizations=True), pml, "step_B",
                   probe=_all_patterns_record())))


def test_recomputed_E_is_refused(fc):
    fields, pml = _fold_grid()
    assert any("recomputed from D" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   FieldsShim(fields, stores_E=False), pml, "step_B",
                   probe=_all_patterns_record())))


def test_a_cylindrical_grid_is_refused_by_name(fc):
    """The r = 0 ghost is ``r_to_minus_r`` and the axis row belongs to the per-m
    rules; ``_boundary_kinds`` would report ``"axis"``, which this file's
    classification does not carry."""
    grid = Grid(resolution=10.0, cell_size=(1.6, 0.0, 1.6), cylindrical=True, xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((0, 2), (0, 0), (2, 2)))
    reasons = _reasons(fc.folded_complex_pml_curl_coverage(
        fields, pml, "step_B", probe=_all_patterns_record()))
    assert any("cylindrical" in r for r in reasons), reasons
    assert any("r = 0 axis" in r for r in reasons), reasons


def test_an_unknown_sub_step_raises_rather_than_refusing(fc):
    fields, pml = _fold_grid()
    for predicate in (fc.folded_complex_pml_curl_coverage,
                      fc.folded_beta_pml_curl_coverage):
        with pytest.raises(ValueError, match="sub_step"):
            predicate(fields, pml, "step_Q")
    with pytest.raises(ValueError, match="family"):
        fc.folded_mirror_ghost_fill_complex_coverage(fields, "Q")
    with pytest.raises(ValueError, match="side"):
        fc.folded_complex_constitutive_coverage(fields, pml, "Q")


# ---------------------------------------------------------------------------
# The fill predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["B", "D"])
@pytest.mark.parametrize("boundaries", ["periodic", "metallic"])
def test_the_fill_is_admitted_on_a_folded_complex_grid(fc, family, boundaries):
    fields, _pml = _fold_grid(boundaries=boundaries)
    assert _reasons(fc.folded_mirror_ghost_fill_complex_coverage(
        fields, family, probe=_all_patterns_record())) == []


def test_the_fill_is_refused_with_no_mirror_plane(fc):
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                boundaries="periodic", xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    assert any("no mirror plane is active" in r
               for r in _reasons(fc.folded_mirror_ghost_fill_complex_coverage(
                   fields, "B", probe=_all_patterns_record())))


def test_the_fill_is_refused_on_real_storage_and_says_why(fc):
    """The real fold's fill stays ``symmetry.mirror_ghost_fill``, which is EXACT
    there — this predicate must not silently claim that territory."""
    fields, _pml = _fold_grid(complex_storage=False)
    assert any("exact sign flip" in r or "EXACT" in r
               for r in _reasons(fc.folded_mirror_ghost_fill_complex_coverage(
                   fields, "B", probe=_all_patterns_record())))


def test_the_fill_checks_the_reflect_row_bounds_rather_than_assuming_them(
        fc, monkeypatch):
    """The far ghost must not image the plane it writes, or read out of the
    allocation. Disjointness is CHECKED, not assumed."""
    fields, _pml = _fold_grid()
    grid = fields.grid
    monkeypatch.setattr(stepping, "_far_reflect_rows",
                        lambda g: (None, int(grid.shape[1]) - 1, None))
    reasons = _reasons(fc.folded_mirror_ghost_fill_complex_coverage(
        fields, "B", probe=_all_patterns_record()))
    assert any("reflect row" in r and "outside" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# The probe contract
# ---------------------------------------------------------------------------

def test_the_parity_pattern_is_a_new_required_pattern(fc):
    """Inheriting the base four patterns' verdict would admit an ambiguous
    platform silently — ``special_kz`` set the precedent that a new operand
    ORIENTATION earns its own pattern."""
    assert fc.PARITY_PROBE_PATTERN not in fc.PROBE_PATTERNS
    assert set(fc.PROBE_PATTERNS) < set(fc.PARITY_PROBE_PATTERNS)
    assert fc.parity_expansion_from_probe(_probe_record(patterns="base")) is None
    assert fc.parity_expansion_from_probe(_probe_record(patterns="parity")) == 1


# ---------------------------------------------------------------------------
# The non-discriminating clause, over the EXTENDED pattern set
# ---------------------------------------------------------------------------
#
# THE CLAUSE, and the two situations it must never conflate. A pattern that
# CANNOT tell the licensable arms apart constrains nothing and is EXCLUDED from
# the agreement test; a pattern that CAN tell them apart and names a different
# arm is evidence against a licence and must VETO it. Until 2026-08-15 this
# tranche implemented its own loop, which accepted AMBIGUOUS_BOTH for
# PARITY_PROBE_PATTERN alone and let any BASE pattern gone blind veto — the
# pre-clause behaviour the base family had already retired. It now calls
# ``complex_fields.expansion_license`` over PARITY_PROBE_PATTERNS.

def test_a_pattern_that_cannot_discriminate_is_excluded_from_the_agreement_test(fc):
    """The parity pattern is the expected blind one: with ``c_re = ±1`` exactly
    and ``c_im = +0.0`` the fused arm's extra product is exact, so both arms
    produce identical bytes (measured on the reference NumPy: 0 mismatch words
    on both arms, arms 0 words apart over 4608 vectors). It is excluded and the
    base four name the arm — and the SAME treatment now reaches a base pattern
    gone blind, which used to veto."""
    record = set_pattern(_probe_record(patterns="parity"),
                         fc.PARITY_PROBE_PATTERN, AMBIGUOUS, apart=0)
    verdict = fc.parity_expansion_license(record)
    assert verdict["expansion"] == 1
    assert verdict["basis"] == "measured"
    assert verdict["non_discriminating"] == [fc.PARITY_PROBE_PATTERN]
    assert verdict["refusals"] == []

    both_blind = set_pattern(record, "c8_mul_c8", AMBIGUOUS, apart=0)
    verdict = fc.parity_expansion_license(both_blind)
    assert verdict["expansion"] == 1, verdict["refusals"]
    assert verdict["basis"] == "measured"
    assert verdict["non_discriminating"] == ["c8_mul_c8",
                                             fc.PARITY_PROBE_PATTERN]


def test_a_pattern_that_discriminates_but_disagrees_still_refuses(fc):
    """THE OPPOSITE SITUATION, and the one a loosened clause would swallow — the
    parity pattern naming the other arm, naming no arm at all, and claiming
    ambiguity its own detail block measures 128 words of disagreement against."""
    disagreeing = set_pattern(_probe_record(patterns="parity"),
                              fc.PARITY_PROBE_PATTERN, "NAIVE")
    verdict = fc.parity_expansion_license(disagreeing)
    assert verdict["expansion"] is None
    assert any("disagree" in reason for reason in verdict["refusals"]), verdict
    assert fc.PARITY_PROBE_PATTERN not in verdict["non_discriminating"]

    contradictory = set_contradictory_ambiguity(_probe_record(patterns="parity"),
                                                fc.PARITY_PROBE_PATTERN)
    verdict = fc.parity_expansion_license(contradictory)
    assert verdict["expansion"] is None
    assert any("words APART" in reason for reason in verdict["refusals"]), verdict
    assert verdict["non_discriminating"] == [], (
        "a pattern whose ambiguity claim its own detail contradicts must never "
        "be recorded as excluded: excluded means MEASURABLY blind")
    assert verdict["ambiguity_refused"] == [fc.PARITY_PROBE_PATTERN]

    only_bad_one_left = set_contradictory_ambiguity(
        _probe_record(patterns="parity"), fc.PARITY_PROBE_PATTERN)
    for name in fc.PROBE_PATTERNS:
        set_pattern(only_bad_one_left, name, AMBIGUOUS, apart=0)
    assert fc.parity_expansion_from_probe(only_bad_one_left) is None


def test_a_parity_ambiguity_the_plane_wise_arm_also_passes_refuses(fc):
    """PLANE-WISE is a LIVE alternative on this pattern, not a theoretical one:
    the gate records ``parity_is_planewise`` as a platform finding precisely
    because a unit-real scalar multiply can reproduce it. A comparison the
    known-wrong transcription passes discriminates nothing and licenses
    nothing — it is a refusal, never an exclusion."""
    record = set_pattern(_probe_record(patterns="parity"),
                         fc.PARITY_PROBE_PATTERN, AMBIGUOUS, apart=0,
                         diagnostic_matched=True)
    verdict = fc.parity_expansion_license(record)
    assert verdict["expansion"] is None
    assert any("known-wrong transcription" in reason
               for reason in verdict["refusals"]), verdict["refusals"]


def test_a_record_on_which_nothing_discriminates_is_not_a_measurement(fc):
    record = _probe_record(patterns="parity")
    for name in fc.PARITY_PROBE_PATTERNS:
        set_pattern(record, name, AMBIGUOUS, apart=0)
    verdict = fc.parity_expansion_license(record)
    assert verdict["expansion"] is None
    assert verdict["arms_coincide_on_every_pattern"] is True
    assert any("matches no measured row" in reason
               for reason in verdict["refusals"]), verdict["refusals"]


def test_the_licence_names_the_pattern_set_and_the_patterns_it_excluded(fc):
    record = set_pattern(_probe_record(patterns="parity"),
                         fc.PARITY_PROBE_PATTERN, AMBIGUOUS, apart=0)
    verdict = fc.parity_expansion_license(record)
    assert verdict["probe_patterns"] == list(fc.PARITY_PROBE_PATTERNS)
    assert verdict["basis"] == "measured"
    assert verdict["non_discriminating"] == [fc.PARITY_PROBE_PATTERN]
    assert fc.parity_expansion_from_probe(record) == verdict["expansion"]


def test_the_folded_beta_licence_carries_the_same_clause(fc):
    """K3b delegates to ``special_kz``; the delegation is what is pinned here, so
    the two entry points cannot drift into different rules."""
    record = set_pattern(_probe_record(patterns="beta"),
                         "c8_mul_c8_imaginary_coefficient_left", AMBIGUOUS,
                         apart=0)
    verdict = fc.folded_beta_expansion_license(record)
    assert verdict["expansion"] == 1
    assert verdict["basis"] == "measured"
    assert verdict["non_discriminating"] == [
        "c8_mul_c8_imaginary_coefficient_left"]
    assert fc.folded_beta_expansion_from_probe(record) == 1

    contradictory = set_contradictory_ambiguity(
        _probe_record(patterns="beta"), "c8_mul_c8_imaginary_coefficient_left")
    assert fc.folded_beta_expansion_from_probe(contradictory) is None


def test_a_neither_verdict_on_the_parity_pattern_refuses(fc):
    record = _probe_record(patterns="parity")
    record["patterns"][fc.PARITY_PROBE_PATTERN] = "NEITHER"
    assert fc.parity_expansion_from_probe(record) is None


@pytest.mark.parametrize("record", [None, {"backend": "numpy", "patterns": {}},
                                    {"backend": "cupy"}, "not a record"])
def test_a_missing_or_wrong_backend_probe_refuses(fc, record):
    assert fc.parity_expansion_from_probe(record) is None


def test_disagreeing_base_patterns_refuse(fc):
    record = _probe_record(patterns="parity")
    record["patterns"]["python_float_left"] = "NAIVE"
    assert fc.parity_expansion_from_probe(record) is None


def test_the_fill_refuses_without_the_parity_pattern_in_the_artifact(fc):
    fields, _pml = _fold_grid()
    reasons = _reasons(fc.folded_mirror_ghost_fill_complex_coverage(
        fields, "B", probe=_probe_record(patterns="base")))
    assert any(fc.PARITY_PROBE_PATTERN in r for r in reasons), reasons


def test_the_complex_beta_curl_requires_the_extended_beta_pattern_set(fc):
    fields, pml = _fold_grid(beta=BETA_EIGSRC)
    reasons = _reasons(fc.folded_beta_bloch_pml_curl_coverage(
        fields, pml, "step_B", probe=_probe_record(patterns="base")))
    assert any("c8_mul_c8_imaginary_coefficient_left" in r for r in reasons), reasons


def test_the_plain_folded_complex_curl_binds_the_BASE_pattern_set(fc):
    """K1 launches no new product orientation, so the base contract is exactly
    right — asking for more would refuse a licensing artifact for no reason."""
    fields, pml = _fold_grid()
    assert _reasons(fc.folded_complex_pml_curl_coverage(
        fields, pml, "step_B", probe=_probe_record(patterns="base"))) == []


def test_the_complex_constitutive_delegate_binds_the_BASE_pattern_set(fc):
    """The kernel launched there is the certified ``bloch_constitutive_step``,
    whose contract is the base set — reached by RESTATING the element-wise
    contract, never by subtracting reasons from another predicate's output."""
    fields, pml = _fold_grid()
    for side in ("H", "E"):
        assert _reasons(fc.folded_complex_constitutive_coverage(
            fields, pml, side, probe=_probe_record(patterns="base"))) == []


def test_a_missing_probe_refuses_every_complex_predicate(fc, monkeypatch):
    monkeypatch.delenv("MEEP_GPU_COMPLEX_EXPANSION_PROBE", raising=False)
    fields, pml = _fold_grid()
    for verdict in (fc.folded_complex_pml_curl_coverage(fields, pml, "step_B"),
                    fc.folded_mirror_ghost_fill_complex_coverage(fields, "B"),
                    fc.folded_complex_constitutive_coverage(fields, pml, "E")):
        assert any("probe artifact" in r for r in verdict.reasons)


# ---------------------------------------------------------------------------
# The real + beta family, and the coupling to symmetry.py this tranche reuses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_folded_real_beta_configuration_is_refused_only_for_the_backend(
        fc, sub_step):
    fields, pml = _fold_grid(complex_storage=False, beta=BETA_GRATING_13_2)
    assert _reasons(fc.folded_beta_pml_curl_coverage(fields, pml, sub_step)) == []


def test_the_real_beta_family_needs_no_probe_artifact(fc, monkeypatch):
    """Real storage carries no complex-multiply expansion; demanding a probe
    there would be a refusal with no measurement behind it."""
    monkeypatch.delenv("MEEP_GPU_COMPLEX_EXPANSION_PROBE", raising=False)
    fields, pml = _fold_grid(complex_storage=False, beta=BETA_GRATING_13_2)
    assert _reasons(fc.folded_beta_pml_curl_coverage(fields, pml, "step_B")) == []


def test_symmetrys_own_fill_predicate_still_admits_a_folded_REAL_beta_grid():
    """THE COUPLING THIS TRANCHE DOES NOT OWN. K3a reuses
    ``symmetry.mirror_ghost_fill`` through ``symmetry.mirror_ghost_fill_coverage``,
    which carries no beta clause today. If the symmetry tranche's owner adds one,
    K3a's fill path silently stops being admitted — so the coupling is pinned
    here rather than assumed.
    """
    fields, _pml = _fold_grid(complex_storage=False, beta=BETA_GRATING_13_2)
    verdict = symmetry.mirror_ghost_fill_coverage(fields, "B")
    assert [r for r in verdict.reasons if "array module" not in r] == [], (
        "symmetry.mirror_ghost_fill_coverage has gained a clause that refuses a "
        "folded REAL beta grid; the folded beta family's fill path needs a "
        "decision, not a silent refusal")


@pytest.mark.parametrize("side", ["H", "E"])
def test_the_real_beta_constitutive_delegate_drops_only_the_beta_clause(fc, side):
    fields, pml = _fold_grid(complex_storage=False, beta=BETA_GRATING_17_7)
    assert _reasons(fc.folded_beta_run_constitutive_coverage(fields, pml, side)) == []


def test_the_real_beta_constitutive_still_refuses_a_registered_polarization(fc):
    fields, pml = _fold_grid(complex_storage=False, beta=BETA_GRATING_17_7)
    assert any("(D - sum P)" in r for r in _reasons(
        fc.folded_beta_run_constitutive_coverage(
            FieldsShim(fields, has_polarizations=True), pml, "E")))


def test_the_beta_coefficients_come_from_the_beta_tranche_unchanged(fc):
    """No second implementation of ``sign * 2*pi*beta*dt`` lives here."""
    from meep_gpu.triton_kernels import special_kz

    assert (fc.beta_curl_coefficients.__module__
            == special_kz.beta_curl_coefficients.__module__
            == "meep_gpu.triton_kernels.special_kz")
    plus, minus = fc.beta_curl_coefficients(BETA_EIGSRC, 0.05, magnetic=True,
                                            complex_storage=False)
    assert plus == -minus and plus != 0.0
    words = fc.beta_curl_coefficients(BETA_EIGSRC, 0.05, magnetic=True,
                                      complex_storage=True)
    assert len(words) == 2 and all(len(pair) == 2 for pair in words)


# ---------------------------------------------------------------------------
# Plan builders — refusal is None, and the launch bindings are the engine's
# ---------------------------------------------------------------------------

def test_every_plan_builder_refuses_to_None_and_never_raises(fc):
    fields, pml = _fold_grid()                    # NumPy: never covered
    assert fc.plan_folded_complex_pml_curl(fields, pml, "step_B",
                                           probe=_all_patterns_record()) is None
    assert fc.plan_folded_mirror_ghost_fill_complex(
        fields, "B", probe=_all_patterns_record()) is None
    assert fc.plan_folded_complex_constitutive(
        fields, pml, "E", probe=_all_patterns_record()) is None
    assert fc.plan_folded_beta_bloch_pml_curl(
        fields, pml, "step_B", probe=_all_patterns_record()) is None
    real_fields, real_pml = _fold_grid(complex_storage=False, beta=BETA_EIGSRC)
    assert fc.plan_folded_beta_pml_curl(real_fields, real_pml, "step_B") is None
    assert fc.plan_folded_beta_run_constitutive(real_fields, real_pml, "E") is None


def test_the_from_arrays_builders_bind_the_engines_own_argument_order(fc):
    """The gate builds from bare arrays; the two routes must agree on which
    volume lands in which pointer slot, or the gate certifies a different kernel
    than the engine would launch."""
    shape = (6, 5, 1)
    arrays = {name: np.zeros(shape, dtype=np.complex64)
              for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                           "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                           "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")}
    flat = {f"{stem}_{axis}": np.ones(shape[index], dtype=np.float32)
            for index, axis in enumerate("xyz") for stem in ("kms", "sinv")}
    plan = fc.plan_folded_complex_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, fc.CODE_MIRROR_PERIODIC, 0),
        (None, None, None), 0.35, 1)
    assert plan.sub_step == "step_B" and plan.backward == 0
    assert plan.bc == (0, fc.CODE_MIRROR_PERIODIC, 0)
    assert plan.phased == (0, 0, 0)
    assert plan.n_elem == 30
    assert repr(plan).startswith("FoldedComplexPmlCurlPlan")

    entries = [{"axis": 1, "phase": 1, "far": True, "reflect_row": 3,
                "near_words": (1.0, 0.0), "far_words": (-1.0, 0.0),
                "shifts": (1, 0, 1)}]
    fill = fc.plan_folded_mirror_ghost_fill_complex_from_arrays(
        "B", arrays, entries, 1)
    assert fill.family == "B" and [e["axis"] for e in fill.axes] == [1]
    assert fill.expansion == 1


def test_the_word_view_binding_refuses_a_real_volume(fc):
    """Real storage read as complex word pairs is the wrong-stride wrong answer
    in the other direction."""
    shape = (4, 4, 1)
    arrays = {name: np.zeros(shape, dtype=np.float32)
              for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez",
                           "fu_Bx", "fu_By", "fu_Bz")}
    flat = {f"{stem}_{axis}": np.ones(shape[index], dtype=np.float32)
            for index, axis in enumerate("xyz") for stem in ("kms", "sinv")}
    with pytest.raises(ValueError, match="complex64"):
        fc.plan_folded_complex_pml_curl_from_arrays(
            "step_B", arrays, flat, (0, fc.CODE_MIRROR_PERIODIC, 0),
            (None, None, None), 0.35, 1)


def test_the_real_beta_plan_carries_two_host_rounded_scalars_and_has_beta(fc):
    shape = (6, 5, 1)
    arrays = {name: np.zeros(shape, dtype=np.float32)
              for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez",
                           "fu_Bx", "fu_By", "fu_Bz")}
    flat = {f"{stem}_{axis}": np.ones(shape[index], dtype=np.float32)
            for index, axis in enumerate("xyz") for stem in ("kms", "sinv")}
    plus, minus = fc.beta_curl_coefficients(BETA_GRATING_13_2, 0.05,
                                            magnetic=True, complex_storage=False)
    plan = fc.plan_folded_beta_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, fc.CODE_MIRROR_METALLIC, 0), 0.35,
        plus, minus)
    assert plan.beta_plus == plus and plan.beta_minus == minus
    assert plan.has_beta == 1
    identity = fc.plan_folded_beta_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, fc.CODE_MIRROR_METALLIC, 0), 0.35,
        plus, minus, has_beta=0)
    assert identity.has_beta == 0


# ---------------------------------------------------------------------------
# The three corpus anchors, resolved on real Grid objects
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,cell,resolution,beta,kx,thickness", [
    ("eigsrc_0_complex", (14.0, 14.0, 0.0), 30.0, BETA_EIGSRC, 0.0,
     ((2, 2), (0, 2), (0, 0))),
    ("special_kz_0_13_2", (4.5, 6.0, 0.0), 30.0, BETA_GRATING_13_2,
     KX_GRATING_13_2, ((1, 1), (0, 0), (0, 0))),
    ("special_kz_1_17_7", (4.5, 6.0, 0.0), 30.0, BETA_GRATING_17_7,
     KX_GRATING_17_7, ((1, 1), (0, 0), (0, 0))),
])
def test_the_three_corpus_anchors_are_admitted_and_sit_in_one_matrix_cell(
        fc, name, cell, resolution, beta, kx, thickness):
    """All three are MIRROR_PERIODIC at an EVEN full count with phase +1, so they
    are ANCHORS, not coverage: the odd-count arm, the metallic arm and the
    phase = -1 arm exist in no corpus row and are synthesised by the gate."""
    grid = Grid(resolution=resolution, cell_size=cell, dimensions=2,
                boundaries="periodic", symmetry=(Mirror("Y", 1),),
                k_point=(kx, 0.0, 0.0), beta=beta, xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=thickness)

    assert grid.shape_full[1] % 2 == 0, "every anchor has an EVEN full count"
    codes, reasons = fc.folded_axis_kinds(grid, pml)
    assert reasons == () and codes[1] == fc.CODE_MIRROR_PERIODIC
    assert grid.stored_cells(1) == grid.owned_cells(1) + 1
    assert stepping._far_reflect_rows(grid)[1] == grid.stored_cells(1) - 2
    assert grid.mirror_phase(1) == 1

    verdict = fc.folded_beta_bloch_pml_curl_coverage(
        fields, pml, "step_B", probe=_all_patterns_record())
    assert _reasons(verdict) == [], (name, verdict.reasons)
    assert _reasons(fc.folded_mirror_ghost_fill_complex_coverage(
        fields, "B", probe=_all_patterns_record())) == []


def test_the_eigsrc_anchor_puts_a_live_absorber_on_the_folded_axis():
    """That row's far ghost plane sits in DEEP PML — the tiny-magnitude regime
    where the subnormal policy bites, and the most likely place for an FTZ
    discrepancy to surface on device."""
    grid = Grid(resolution=30.0, cell_size=(14.0, 14.0, 0.0), dimensions=2,
                boundaries="periodic", symmetry=(Mirror("Y", 1),),
                beta=BETA_EIGSRC, xp=np)
    pml = PML(grid=grid, thickness=((2, 2), (0, 2), (0, 0)))
    assert pml.is_active
    kms_y = np.asarray(pml.kms_y).ravel()
    assert float(kms_y.min()) < 1.0, (
        "the folded axis must carry a live absorber on its HIGH face")


# ---------------------------------------------------------------------------
# The gate's armed mutations, checked WITHOUT a GPU
# ---------------------------------------------------------------------------
#
# A mutation whose transform matches nothing is DISARMED: the leg compiles the
# SHIPPED kernel, launches it, sees zero divergence and reports a hollow pass —
# which is worse than no leg at all. The gate detects that at run time; this
# detects it at the merge bar, where a rename or a re-indent actually happens.


def _mutation_sources():
    """The shipped bodies the gate's transforms operate on, from the FILES."""
    package = PACKAGE_DIR
    text = (package / "folded_complex.py").read_text(encoding="utf-8")
    helpers = ((package / "complex_fields.py").read_text(encoding="utf-8")
               + (package / "special_kz.py").read_text(encoding="utf-8"))
    bodies = {node.name: ast.get_source_segment(text, node)
              for node in ast.parse(text).body
              if isinstance(node, ast.FunctionDef)}
    helper_bodies = {node.name: ast.get_source_segment(helpers, node)
                     for node in ast.parse(helpers).body
                     if isinstance(node, ast.FunctionDef)}
    shared = [helper_bodies[name] for name in
              ("_rotate_field_left", "_mul_field_left", "_mul_coefficient_left",
               "_mul_imag_coefficient_left")]
    return {
        "curl": "\n\n".join(shared + [bodies["folded_bloch_pml_curl_step"]]),
        "fill": "\n\n".join(shared
                             + [bodies["folded_mirror_ghost_fill_complex"]]),
        "beta_real": "\n\n".join(shared
                                  + [bodies["folded_beta_pml_curl_step"]]),
        "beta_complex": "\n\n".join(
            shared + [bodies["folded_beta_bloch_pml_curl_step"]]),
    }


def _gate_module():
    """The gate, imported as itself — with NO ``sys.modules['cupy']`` stub.

    Both loaders used to do ``sys.modules.setdefault("cupy", np)``. The gate has
    imported ``cupy`` defensively for a while, so the stub bought nothing, and it
    is process-global and never undone: every module imported AFTER this file in
    the same interpreter then bound NumPy as ``cp``, took ``cp is not None`` to
    mean a device, and reached for a ``numpy.asnumpy`` that does not exist.
    Measured (2026-08-13): with the stub, a collection order that put this file
    before ``test_triton_cylindrical_complex.py`` and ``test_triton_conductivity.py``
    turned the cylindrical gate's device-absent refusal into an -ftz cache-policy
    RuntimeError and broke all six conductivity bit-identity cases — seven tests
    whose greenness depended on which files were collected alongside them.
    ``meep_gpu/conftest.py`` fails any test that reinstates such a stub.
    """
    sys.path.insert(0, str(PARITY_DIR))
    return importlib.import_module("gate_triton_folded_complex")


def _validator_module():
    sys.path.insert(0, str(PARITY_DIR))
    return importlib.import_module(VALIDATOR_PATH.stem)


def test_every_gate_source_mutation_is_ARMED_and_still_compiles():
    """Each transform must MATCH, must CHANGE the source, and the result must
    parse with its entry point intact — the three ways a mutation silently
    disarms (a rename, a re-indent, a syntax break)."""
    gate = _gate_module()
    sources = _mutation_sources()
    disarmed = []
    for name, (transform, body, entry, _note) in gate.SOURCE_MUTATIONS.items():
        mutated, hits = transform(sources[body])
        if hits == 0 or mutated == sources[body]:
            disarmed.append((name, "matched nothing"))
            continue
        try:
            tree = ast.parse(mutated)
        except SyntaxError as exc:
            disarmed.append((name, f"does not parse: {exc}"))
            continue
        names = {node.name for node in tree.body
                 if isinstance(node, ast.FunctionDef)}
        if entry not in names:
            disarmed.append((name, f"entry point {entry} is gone"))
    assert disarmed == [], disarmed


def test_the_headline_mutation_targets_the_parity_product_on_both_arms():
    """``plane_wise_parity_fill`` must remove EVERY complex product from the
    fill — three near and three far — or it leaves a half-mutated kernel whose
    verdict means nothing."""
    gate = _gate_module()
    transform = gate.SOURCE_MUTATIONS["plane_wise_parity_fill"][0]
    mutated, hits = transform(_mutation_sources()["fill"])
    assert hits == 6, hits
    body = mutated.split("def folded_mirror_ghost_fill_complex")[-1]
    assert "_mul_imag_coefficient_left" not in body


def test_the_gate_names_every_mutation_the_spec_requires():
    """The EXACT armed set, and the count the docstring and the slurm banner
    advertise. A subset assertion cannot see drift: the gate said "fourteen" for
    sixteen mutations and no test could tell."""
    gate = _gate_module()
    named = set(gate.SOURCE_MUTATIONS) | set(gate.HOST_MUTATIONS)
    required = {
        "plane_wise_parity_fill", "identity_shortcut_on_even_mirror",
        "synthesize_zero_imag", "fold_ghost_wraps", "drop_top_plane_mask",
        "mask_only_metallic", "reflect_row_n_minus_two", "beta_after_mask",
        "rotate_beta_partner", "phase_on_folded_axis", "unary_minus_addend",
        "fold_zero_cross_terms", "reverse_axis_order",
        "far_fill_uses_plus_phase",
    }
    assert required <= named, sorted(required - named)
    total = len(gate.SOURCE_MUTATIONS) + len(gate.HOST_MUTATIONS)
    assert total == 17, total
    # The advertised split, not just the advertised total: one SOURCE mutation is
    # retired as a recorded structural null, and the denominator still says 17.
    spelled = ("ten armed SOURCE mutations, one RECORDED STRUCTURAL NULL and\n"
               "  six HOST ones (seventeen")
    assert spelled in (gate.__doc__ or ""), "the gate docstring's count drifted"
    armed_source = len(gate.SOURCE_MUTATIONS) - len(gate.SOURCE_MUTATION_NULLS)
    assert (armed_source, len(gate.SOURCE_MUTATION_NULLS),
            len(gate.HOST_MUTATIONS)) == (10, 1, 6)
    # The two host legs that were provably no-ops, or absent, until 2026-08-12.
    assert "imaginary_words_are_read_not_synthesised" in gate.HOST_MUTATIONS
    assert "synthesize_zero_imag_host" not in gate.HOST_MUTATIONS
    assert "fuse_fill_passes" in gate.HOST_MUTATIONS


def test_reverse_axis_order_is_a_measurement_not_a_predicted_null():
    """``symmetry.py``'s commute null is a REAL-storage measurement and does not
    transfer: complex float multiplication is not associative.

    The LABEL is not the arming. A grid whose folded axes carry the same declared
    phase makes the composition commute bit-exactly, so the leg needs a
    MIXED-PHASE two-folded-axis row to be a measurement at all — and the entry
    mutation needs an armed-row count so "nothing was varied" cannot read as
    "diverged: 0".
    """
    gate = _gate_module()
    assert gate.HOST_MUTATIONS["reverse_axis_order"] is None
    mixed = [spec for spec in gate.FILL_GRIDS
             if len(spec["axis"]) > 1
             and isinstance(spec["phase"], (tuple, list))
             and len(set(spec["phase"])) > 1]
    assert mixed, "no mixed-phase two-folded-axis fill grid: the leg is a null"
    quick = [gate.FILL_GRIDS[i] for i in gate.QUICK_FILL_GRID_INDICES]
    assert any(spec in mixed for spec in quick), (
        "--quick builds no mixed-phase grid, so reverse_axis_order is a literal "
        "no-op on every row it runs there")


def test_the_order_mutations_needle_diverges_on_the_grid_the_gate_builds():
    """The ARMING, executed — not the label, and not the grid list.

    MEASURED here on NumPy through the gate's own ``build_folded_fields`` /
    ``seed_fill_state`` / reference fills: reversing the entry list on the
    MIXED-phase two-folded-axis grid diverges from the array path in 5 words of
    ``Bz`` and 5 of ``Dz`` on ``engineered_rows``, while the SAME-phase grid — a
    row the leg counts as armed — diverges in 0 words. ``Bz`` is far on both
    folded axes and ``Dz`` near on both, so those are the doubly-written corners.
    """
    gate = _gate_module()
    fold = importlib.import_module(MODULE_NAME)
    same = [s for s in gate.FILL_GRIDS
            if s["axis"] == "XY" and not isinstance(s["phase"], tuple)][0]
    mixed = [s for s in gate.FILL_GRIDS
             if s["axis"] == "XY" and isinstance(s["phase"], tuple)][0]

    def diverging(spec, family):
        grid, fields, pml = gate.build_folded_fields(spec, np, thin_pml=True)
        names = fold.GHOST_FILL_FAMILIES[family]["targets"]
        with np.errstate(over="ignore", invalid="ignore"):
            gate.seed_fill_state(fields, grid, names, "engineered_rows", pml)
        entries = [dict(e) for e in fold.ghost_fill_axis_entries(grid, family)]
        base = {n: np.array(getattr(fields, n), copy=True) for n in names}
        out = {}
        for label, order in (("forward", entries),
                             ("reversed", list(reversed(entries)))):
            arrays = {n: np.array(v, copy=True) for n, v in base.items()}
            with np.errstate(over="ignore", invalid="ignore"):
                gate.reference_ghost_fill_near(np, arrays, family, order)
                gate.reference_ghost_fill_far(np, arrays, family, order)
            out[label] = arrays
        return {n: int((np.ascontiguousarray(out["forward"][n])
                        .view(np.uint32).ravel()
                        != np.ascontiguousarray(out["reversed"][n])
                        .view(np.uint32).ravel()).sum()) for n in names}

    assert sum(diverging(same, "B").values()) == 0, "the same-phase grid is a null"
    assert sum(diverging(same, "D").values()) == 0, "the same-phase grid is a null"
    assert diverging(mixed, "B")["Bz"] > 0, diverging(mixed, "B")
    assert diverging(mixed, "D")["Dz"] > 0, diverging(mixed, "D")


def test_a_host_mutation_that_is_armed_but_cannot_differ_is_DISARMED():
    """``armed_rows > 0`` was the only gate, and it admits a provable no-op.

    A ``predicted is None`` leg reached ``verdict: MEASURED`` before any
    divergence or vacuity check, so a run in which every armed row was a no-op
    recorded MEASURED with a coverage note behind it. Both halves are pinned:
    armed-with-no-discriminating-row DISARMS, and discriminating-rows-with-zero-
    divergence is NEEDLE-MISSED even for a leg whose predicted verdict is
    "record the measurement".
    """
    gate = _gate_module()

    def verdict(armed, discriminating, diverged, predicted=None,
                name="reverse_axis_order"):
        summary = {"ran": 96, "identical": 96 - diverged}
        if armed is not None:
            summary["armed_rows"] = armed
            summary["discriminating_rows"] = discriminating
        return gate.host_mutation_verdict(name, predicted, summary, diverged)

    # nothing varied at all
    assert verdict(0, 0, 0)["verdict"] == "DISARMED"
    # armed on 16 rows, every one of them a provable no-op — the case that used
    # to record MEASURED with a coverage note behind it
    record = verdict(16, 0, 0)
    assert record["verdict"] == "DISARMED"
    assert "provable no-op" in record["failure"]
    # a reachable class and zero diverging words REFUTES the claim
    record = verdict(16, 8, 0)
    assert record["verdict"] == "NEEDLE-MISSED"
    assert "BLIND to the defect" in record["failure"]
    # and only a reachable class WITH divergence is the measurement
    record = verdict(16, 8, 4)
    assert record["verdict"] == "MEASURED"
    assert "discriminating_rows" in record["note"]
    assert record["discriminating_rows"] == 8

    # a leg with no entry-list counts at all (phase_on_folded_axis) is untouched
    record = verdict(None, None, 2, name="phase_on_folded_axis")
    assert record["verdict"] == "MEASURED"
    assert record["armed_rows"] is None
    # ...and a PREDICTED-caught value mutation still fails on zero divergence
    assert verdict(22, 22, 0, predicted=True,
                   name="far_fill_uses_plus_phase")["verdict"] == "NEEDLE-MISSED"
    assert verdict(22, 22, 9, predicted=True,
                   name="far_fill_uses_plus_phase")["verdict"] == "CAUGHT"


def test_the_retired_source_mutation_is_a_recorded_null_not_a_deletion():
    """``synthesize_zero_imag`` is VACUOUS BY CONSTRUCTION and is retired as a
    RECORDED STRUCTURAL NULL — kept in the run, judged, and carrying its reason,
    its live twin and the numbers behind it.

    The three ways this retirement could have been dishonest, each refused here:
    DELETING it (the table would show sixteen and a disappearance), flipping it
    to "predicted not-caught" without saying why (an assertion in place of a
    measurement), and stating a reason that is about the GATE's states rather
    than the shipped code (that is a blind configuration — NEEDLE-MISSED — and
    the repair is a better needle, not a retirement).

    The evidence is also checked for STALENESS, not just for presence: the
    recorded ``hits`` must still be what the transform produces on today's
    shipped source, so editing the fill body invalidates the record instead of
    leaving a number from a run that measured different code.
    """
    gate = _gate_module()
    name = "synthesize_zero_imag"
    record = gate.SOURCE_MUTATION_NULLS[name]

    # kept, not deleted — still transformed, still compiled, still launched
    assert name in gate.SOURCE_MUTATIONS
    transform, body, entry, note = gate.SOURCE_MUTATIONS[name]
    assert entry == "folded_mirror_ghost_fill_complex"
    assert "RECORDED STRUCTURAL NULL" in note

    # the reason is about the SHIPPED code and names where it lives
    assert "folded_complex.py:2100" in record["reason"]
    assert "BITWISE" in record["reason"] and "0x00000000" in record["reason"]
    assert "BIT-IDENTICAL operands" in record["reason"]

    # the live claim moved rather than vanished, and the twin is a REAL leg
    twin = record["twin"]
    assert twin == "imaginary_words_are_read_not_synthesised"
    assert gate.HOST_MUTATIONS[twin] is True
    for number in ("96", "82"):
        assert number in record["twin_claim"], record["twin_claim"]

    # the numbers the retirement rests on, off a named run
    measured = record["measured"]
    assert measured["run"] == "direct_20260813T084108Z"
    assert measured["ptx_differs"] is True
    assert measured["launches"] == 192
    assert measured["diverged_cases"] == 0
    assert measured["fill_ran"] == measured["fill_identical"] == 96
    assert measured["overlapping_specializations"] == 0

    # ...and they are not stale: the recorded hit count is re-derived here.
    _mutated, hits = transform(_mutation_sources()[body])
    assert hits == measured["hits"], (
        "the shipped fill body changed under the recorded null: its evidence "
        "was measured against different code and the retirement must be re-run")

    # THE SELF-INVALIDATION HOOK: the named premise test must exist in this file.
    path, _sep, test_name = record["premise_test"].partition("::")
    assert path.endswith(pathlib.Path(__file__).name), path
    tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
    assert test_name in {node.name for node in tree.body
                         if isinstance(node, ast.FunctionDef)}, (
        "the gate's null points at a premise test that does not exist; the "
        "retirement no longer re-opens itself")


def test_a_predicted_null_is_judged_not_skipped_and_self_invalidates():
    """The SOURCE-mutation verdict's clause order, executed on the laptop.

    A null is not a skip. All three DISARMED paths run BEFORE the null clause,
    so a declared null whose mutant never matched, never launched, or compiled to
    a shipped specialization is DISARMED — being unmeasured must never launder
    itself into being provably unmeasurable.

    And the null is self-invalidating at the gate layer too: a predicted null
    that DIVERGES is NULL-INVALIDATED and FAILS, because divergence falsifies the
    recorded reason. Relabelling it CAUGHT would leave a false reason standing in
    the coverage table, which is the failure mode the retirement exists to avoid.
    """
    gate = _gate_module()
    null = "synthesize_zero_imag"
    live = "plane_wise_parity_fill"

    # the three DISARMED paths, on the NULL itself
    assert gate.source_mutation_verdict(null, 0, None, None, None)["verdict"] \
        == "DISARMED"
    assert gate.source_mutation_verdict(null, 6, 0, True, 0)["verdict"] \
        == "DISARMED"
    stale = gate.source_mutation_verdict(null, 6, 192, False, 0)
    assert stale["verdict"] == "DISARMED"
    assert "stale binary" in stale["failure"]

    # armed, launched, PTX-distinct and unable to move a byte — the null
    recorded = gate.source_mutation_verdict(null, 6, 192, True, 0)
    assert recorded["verdict"] == "PREDICTED-NULL"
    assert "failure" not in recorded
    assert "0x00000000" in recorded["null_reason"]
    assert recorded["live_claim_carried_by"] == \
        "imaginary_words_are_read_not_synthesised"
    assert recorded["retirement_evidence"]["launches"] == 192
    assert recorded["premise_test"].endswith(
        "test_the_parity_coefficients_imaginary_word_is_bitwise_zero")

    # the premise broke: FAIL, do not quietly upgrade to CAUGHT
    broken = gate.source_mutation_verdict(null, 6, 192, True, 7)
    assert broken["verdict"] == "NULL-INVALIDATED"
    assert "is now FALSE" in broken["failure"]
    assert "do not relabel this caught" in broken["failure"].lower()

    # a mutation that is NOT declared a null keeps the old, stricter reading
    assert gate.source_mutation_verdict(live, 6, 192, True, 0)["verdict"] \
        == "NEEDLE-MISSED"
    assert gate.source_mutation_verdict(live, 6, 192, True, 64)["verdict"] \
        == "CAUGHT"


def test_the_summary_reports_seventeen_mutations_with_an_honest_split():
    """The artifact must show 17 mutations and how each answered — never 16 and
    a disappearance.

    Four refusals are pinned: a PREDICTED-NULL does not fail the gate but does
    carry its reason into the summary; a NULL-INVALIDATED does fail; a declared
    null the mutation leg never ran is a failure (the third way a null could go
    quiet); and a null whose live twin did not itself come back established is a
    failure, because the retirement's whole argument is that the coverage MOVED.
    """
    gate = _gate_module()
    licence = {"licensed": True, "reasons": []}

    def mutations(null_verdict="PREDICTED-NULL", twin_verdict="CAUGHT",
                  drop_null=False):
        records = {}
        for name in list(gate.SOURCE_MUTATIONS) + list(gate.HOST_MUTATIONS):
            records[name] = {"verdict": "CAUGHT"}
        for name in ("reverse_axis_order", "phase_on_folded_axis"):
            records[name] = {"verdict": "MEASURED"}
        records["imaginary_words_are_read_not_synthesised"] = {
            "verdict": twin_verdict, "failure": "…"}
        if drop_null:
            records.pop("synthesize_zero_imag")
        else:
            declared = gate.SOURCE_MUTATION_NULLS["synthesize_zero_imag"]
            records["synthesize_zero_imag"] = {
                "verdict": null_verdict,
                "failure": "…",
                "null_reason": declared["reason"],
                "live_claim_carried_by": declared["twin"],
                "twin_claim": declared["twin_claim"],
                "retirement_evidence": declared["measured"],
                "premise_test": declared["premise_test"]}
        return {"mutations": records}

    built = gate.build_summary(mutations(), licence)
    counts = built["mutation_counts"]
    assert counts["asked"] == counts["ran"] == 17
    assert counts["PREDICTED-NULL"] == 1
    assert counts["MEASURED"] == 2
    assert counts["CAUGHT"] == 14
    assert sum(value for key, value in counts.items()
               if key not in ("asked", "ran")) == 17
    null = built["mutation_nulls"]["synthesize_zero_imag"]
    assert "0x00000000" in null["reason"]
    assert null["twin_verdict"] == "CAUGHT"
    assert null["retirement_evidence"]["diverged_cases"] == 0
    assert not [f for f in built["failures"] if "synthesize_zero_imag" in f]

    # the premise broke on the device
    broken = gate.build_summary(mutations(null_verdict="NULL-INVALIDATED"),
                                licence)
    assert [f for f in broken["failures"]
            if "synthesize_zero_imag" in f and "NULL-INVALIDATED" in f]

    # declared here, never executed there
    missing = gate.build_summary(mutations(drop_null=True), licence)
    assert missing["mutation_counts"]["ran"] == 16
    assert [f for f in missing["failures"] if "a null is judged, not skipped" in f]

    # the twin stopped carrying the claim
    orphaned = gate.build_summary(mutations(twin_verdict="NEEDLE-MISSED"),
                                  licence)
    assert [f for f in orphaned["failures"]
            if "established by nothing" in f]


def test_the_gate_carries_the_planewise_diagnostic_arm_explicitly():
    """THE headline risk: if CuPy's complex64 unit-real scalar multiply takes a
    real-scalar fast path, the parity product COLLAPSES to plane-wise and K2's
    delta evaporates. The probe must be able to say so out loud."""
    gate = _gate_module()
    rng = np.random.default_rng(11)
    operands = gate._parity_operands(rng)
    candidates = gate._parity_candidates(*operands)
    assert "PLANEWISE_diagnostic" in candidates
    # And the two licensed arms must AGREE here — c_re = +-1, c_im = +0.0 makes
    # the fused arm's extra product exact — which is why this pattern is the one
    # EXPECTED to go blind and be excluded from the agreement test. It is not
    # excluded because it is this pattern: the exclusion clause reads the
    # measured arms-apart count of whichever pattern claims ambiguity.
    fma_re, fma_im = candidates["FMA_V1"]
    naive_re, naive_im = candidates["NAIVE"]
    assert np.array_equal(fma_re.view(np.uint32), naive_re.view(np.uint32))
    assert np.array_equal(fma_im.view(np.uint32), naive_im.view(np.uint32))
    # ...and the plane-wise arm must NOT agree, or there is no needle at all.
    plane_re, plane_im = candidates["PLANEWISE_diagnostic"]
    differing = (int(np.count_nonzero(fma_re.view(np.uint32)
                                      != plane_re.view(np.uint32)))
                 + int(np.count_nonzero(fma_im.view(np.uint32)
                                        != plane_im.view(np.uint32))))
    assert differing > 0, "the parity probe's vectors cannot see the needle"


def test_the_record_the_gate_emits_can_satisfy_the_licence_rule_on_every_pattern(fc):
    """THE TWO HALVES MUST FIT, and the exclusion clause is what makes the second
    half load-bearing here.

    Since both licences became ``complex_fields.expansion_license`` over a longer
    pattern list, an ``AMBIGUOUS_BOTH`` claim is believed only when the record's
    OWN ``vectors[name]`` and ``detail[name]`` back it. This tranche's earlier
    loop took the word on trust, so nothing forced the gate to WRITE that
    evidence for the two patterns it adds — and an honest record that refuses on
    a bookkeeping gap is as much a defect as a wrong licence. The base gate had
    exactly that gap on ``python_float_left`` (detail a LIST, no ``vectors``
    entry at all), and the beta pattern this record folds in carries the same
    list shape, one entry per corpus coefficient.

    Measured on this host rather than asserted: the loop below rewrites each
    pattern's detail to a CONSISTENT ambiguity and checks the evidence carries
    it, which is host-independent; the verdict assertions read what the record
    says rather than pinning an arm, because which patterns discriminate is a
    platform fact and this suite runs wherever the merge bar does.
    """
    complex_fields = importlib.import_module(
        "meep_gpu.triton_kernels.complex_fields")
    gate = _gate_module()
    record = gate.measure_folded_expansion_record(np, "numpy")
    required = tuple(dict.fromkeys(fc.PARITY_PROBE_PATTERNS
                                   + fc.FOLDED_BETA_PROBE_PATTERNS))
    for name in required:
        assert record["vectors"].get(name), f"no vector count for {name!r}"
        assert record["detail"].get(name), f"no detail block for {name!r}"

    for name in required:
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
    for label, verdict in (("parity", fc.parity_expansion_license(measured)),
                           ("beta", fc.folded_beta_expansion_license(measured))):
        assert verdict["ambiguity_refused"] == [], (label, verdict["refusals"])
        blind = [name for name in verdict["probe_patterns"]
                 if measured["patterns"][name] == complex_fields.AMBIGUOUS_BOTH]
        assert verdict["non_discriminating"] == blind, (label, verdict)
        unlicensable = sorted(
            name for name in verdict["probe_patterns"]
            if measured["patterns"][name] not in complex_fields.EXPANSIONS
            and measured["patterns"][name] != complex_fields.AMBIGUOUS_BOTH)
        if unlicensable:
            assert verdict["expansion"] is None, (label, unlicensable, verdict)
            for name in unlicensable:
                assert any(repr(name) in reason
                           for reason in verdict["refusals"]), (
                    label, name, verdict["refusals"])
        else:
            assert verdict["basis"] == "measured", (label, verdict)
            assert verdict["expansion"] is not None, (label, verdict["refusals"])


def test_both_artifact_cutters_record_the_basis_and_the_exclusions_by_name():
    """An arm and a word is not a readable licence.

    'measured' against 'environment_default' is the difference between this run
    having DISCRIMINATED the arm and having inherited it from a prior one, and
    the patterns a verdict EXCLUDED are what a reader needs to weigh either — an
    arm carried by one live pattern with four excluded is a weaker claim than one
    carried by six, and an artifact printing only ``licensed: true`` cannot be
    told apart from the weaker case. BOTH licences are recorded, K2's parity and
    K3b's beta, because the two bind different pattern sets. Pinned by source
    text because the write happens on a CuPy host and this suite has none; the
    verdict's own shape is pinned by behaviour above.
    """
    for path in (GATE_PATH, COMPOSITION_PATH):
        text = path.read_text(encoding="utf-8")
        assert "parity_expansion_license(" in text, path
        assert "folded_beta_expansion_license(" in text, path
        assert "non_discriminating" in text, path
        assert "basis" in text, path


# ---------------------------------------------------------------------------
# The harness's own defects — every one of these pins a leg that was measured
# to be measuring nothing, or measuring the wrong thing
# ---------------------------------------------------------------------------

def test_the_engineered_plane_keeps_its_signed_zeros():
    """``re + 1j*im`` destroys the discriminating class; plane assignment does not.

    This is the same rule ``gate_triton_complex._interleave_c8`` was written for.
    The counts below are the MEASUREMENT: the addition build carries 3 negative
    zeros and no negative-zero IMAGINARY word at all, which is why the headline
    needle collapsed from 8/128-at-both-parities to 2/128 and 13/128.
    """
    plane = _engineered_complex_plane()
    words = np.ascontiguousarray(plane).view(np.uint32).ravel()
    assert int((words == np.uint32(0x80000000)).sum()) == 16
    imag = np.ascontiguousarray(plane.imag.copy()).view(np.uint32).ravel()
    assert int((imag == np.uint32(0x80000000)).sum()) == 8

    real, imaginary = np.meshgrid(np.float32(ENGINEERED_WORDS),
                                  np.float32(ENGINEERED_WORDS), indexing="ij")
    added = (real.astype(np.float32)
             + 1j * imaginary.astype(np.float32)).astype(np.complex64)
    added_words = np.ascontiguousarray(added).view(np.uint32).ravel()
    assert int((added_words == np.uint32(0x80000000)).sum()) == 3
    assert not np.array_equal(words, added_words)


def test_the_headline_count_agrees_with_the_module_and_the_gate():
    """One number, three homes. The module states it, the test pins it and the
    gate predicts it; when the test's word set was degraded they disagreed and
    nothing said so."""
    assert PLANE_WISE_DIVERGENCE == {1: 8, -1: 8}
    text = (PACKAGE_DIR / "folded_complex.py").read_text(encoding="utf-8")
    assert "diverges in 8/128 words at BOTH parities" in text
    gate_text = GATE_PATH.read_text(encoding="utf-8")
    assert "8/128 engineered words" in gate_text


def test_the_gates_coefficient_tables_are_broadcast_shaped_and_the_reference_runs():
    """The gate's PRIMARY byte leg, executed on NumPy for every shape it sweeps.

    ``_coefficients`` returned FLAT 1-D vectors while ``folded_reference_step``
    forwards them to ``probe.reference_recurrence``, which needs the
    axis-broadcastable tables. Every ``SHAPES`` entry raised ``ValueError`` at
    ``fu *= kms`` on the FIRST term of the FIRST case, and ``main()`` runs
    ``run_synthetic`` before every other device leg with no ``try``/``except`` —
    so the gate could never produce a fill, zero_init, identity, mutation, engine
    or corpus result at all. This runs the host half of ``one_curl_case``.
    """
    gate = _gate_module()
    sys.path.insert(0, str(PARITY_DIR))
    probe = importlib.import_module("probe_fused_kernel_bit_identity")
    for shape in gate.SHAPES:
        rng = np.random.default_rng(gate.SEED + 401)
        coefficients = gate._coefficients(shape, rng)
        for index, axis in enumerate("xyz"):
            for stem in ("kms", "sinv"):
                assert (coefficients[f"{stem}_{axis}"].shape
                        == probe.broadcast_shape(index, int(shape[index])))
        for config in gate.CONFIGS:
            if gate.viable(shape, config, (None, None, None)):
                continue
            for sub_step in gate.CURL_SUB_STEPS:
                spec = gate.folded_complex.SUB_STEPS[sub_step]
                targets = tuple(spec["targets"])
                names = (targets + tuple("fu_" + t for t in targets)
                         + tuple(spec["sources"]))
                rows = gate.reflect_rows_for(shape, config["boundaries"],
                                             (2, 2, 2))
                for complex_storage in (True, False):
                    state = {
                        n: (gate.gate._seed_host_complex(shape, rng)
                            if complex_storage
                            else gate._seed_host_real(shape, rng))
                        for n in names}
                    with np.errstate(over="ignore", invalid="ignore"):
                        gate.folded_reference_step(
                            np, state, coefficients, np.float32(0.35), sub_step,
                            config["boundaries"], (None, None, None), (1, 1, 1),
                            rows, beta=gate.BETA_GRATING_13_2,
                            dt=gate.BETA_DT, complex_storage=complex_storage)


def test_the_gate_asserts_the_fusion_OFF_rows_not_the_fusion_ON_ones():
    """``guard`` is the FUSION setting: ``plan.run(guard=True)`` means fusion ON.

    The shipped configuration is ``kernels.ENABLE_FP_FUSION`` = False, which is
    what ``guard=False`` reproduces, so the fusion-OFF rows are the ones a pass
    must assert. The gate asserted the fusion-ON rows instead — the inverse of
    both certified sibling gates — so every row in the configuration production
    launches could have diverged with ``passed`` still true.
    """
    gate = _gate_module()
    cases = [{"guard": False, "verdict": {"bit_identical": False}},
             {"guard": True, "verdict": {"bit_identical": True}}]
    summary = gate._summarize(cases)
    assert summary["fusion_off_asserted_total"] == 1
    assert summary["fusion_off_asserted_identical"] == 0
    assert summary["fusion_on_measured_total"] == 1
    built = gate.build_summary({"synthetic": {"summary": summary}},
                               {"licensed": True, "reasons": []})
    assert not built["passed"]
    assert any("fusion-OFF" in f for f in built["failures"]), built["failures"]
    # ...and an EMPTY asserted set is a failure too, not a vacuous pass.
    empty = gate._summarize([{"guard": True,
                              "verdict": {"bit_identical": True}}])
    built = gate.build_summary({"synthetic": {"summary": empty}},
                               {"licensed": True, "reasons": []})
    assert not built["passed"]


def test_the_gate_READS_the_shipped_fusion_default_rather_than_stating_it():
    """The asserted set is spelled ``guard is False``, so the claim "that IS the
    shipped configuration" only holds while ``kernels.ENABLE_FP_FUSION`` is
    False. The gate reads that constant (``kernels.py`` needs Triton to import,
    so it is parsed from source on a laptop) and refuses if it ever stops
    matching — otherwise the same defect returns in the other direction, with the
    gate certifying a configuration production no longer launches.
    """
    gate = _gate_module()
    source = (PACKAGE_DIR / "kernels.py").read_text()
    tree = ast.parse(source)
    literal = [ast.literal_eval(node.value) for node in tree.body
               if isinstance(node, ast.Assign)
               for target in node.targets
               if isinstance(target, ast.Name) and target.id == "ENABLE_FP_FUSION"]
    assert literal == [False], literal
    assert gate.shipped_fusion_default() is False

    # ARMED: with the package default flipped, a clean all-identical synthetic
    # summary must FAIL, because the rows it asserts are no longer the shipped
    # ones. Without this the mutation is unobservable.
    clean = gate._summarize([{"guard": False, "verdict": {"bit_identical": True}},
                             {"guard": True, "verdict": {"bit_identical": True}}])
    passing = gate.build_summary({"synthetic": {"summary": clean}},
                                 {"licensed": True, "reasons": []})
    assert passing["passed"], passing["failures"]
    original = gate.shipped_fusion_default
    gate.shipped_fusion_default = lambda: True
    try:
        flipped = gate.build_summary({"synthetic": {"summary": clean}},
                                     {"licensed": True, "reasons": []})
    finally:
        gate.shipped_fusion_default = original
    assert not flipped["passed"]
    assert any("ENABLE_FP_FUSION is True" in f for f in flipped["failures"]), \
        flipped["failures"]


def test_a_leg_that_RAISES_is_recorded_and_the_remaining_legs_still_run(
        tmp_path, monkeypatch):
    """One leg's defect must not erase every other leg's evidence.

    ``main`` called each leg bare in a fixed order with ``synthetic`` first, so a
    raise there took the whole gate down: no fill, zero_init, identity, mutation,
    engine or corpus result, and a traceback on stdout with no verdict attached.
    ARMED both ways — the raising leg is recorded AND named as a failure, and the
    leg after it still produces its result.
    """
    gate = _gate_module()
    out = tmp_path / "gate.json"
    ran: list = []

    def exploding(*_args, **_kwargs):
        raise RuntimeError("armed leg failure")

    def reference(results, out_path, _xp):
        ran.append("reference")
        results["reference"] = {"ran": 3, "identical": 3}
        gate.save(results, out_path)
        return results["reference"]

    # The laptop configuration this models: no CuPy anywhere. Pinned rather than
    # assumed, so the same assertions hold on a device host — every module that
    # reads ``cp is not None`` would otherwise take a device path, the -ftz
    # cache-policy refusal and ``probe.device_info``, both real behaviours that
    # belong in a device run and not in this test's path.
    base = importlib.import_module("gate_triton_complex")
    probe = importlib.import_module("probe_fused_kernel_bit_identity")
    for module in (gate, base, probe):
        monkeypatch.setattr(module, "cp", None)
    monkeypatch.setattr(gate, "run_expansion", exploding)
    monkeypatch.setattr(gate, "run_reference_validation", reference)
    code = gate.main(["--out", str(out), "--legs", "expansion,reference"])

    import json as _json
    record = _json.loads(out.read_text())
    # The leg AFTER the raising one still ran and still wrote its result.
    assert ran == ["reference"]
    assert record["reference"] == {"ran": 3, "identical": 3}
    # The raise is recorded with its traceback, not swallowed...
    assert record["leg_errors"]["expansion"]["type"] == "RuntimeError"
    assert record["leg_errors"]["expansion"]["message"] == "armed leg failure"
    assert "RuntimeError" in record["leg_errors"]["expansion"]["traceback"]
    # ...and it is a NAMED failure, not a skip.
    assert not record["summary"]["passed"]
    assert any("leg expansion RAISED" in f
               for f in record["summary"]["failures"]), record["summary"]
    assert code == 1


def test_a_device_leg_that_vanishes_without_a_result_or_an_error_FAILS():
    """A leg that was requested, was not cleanly skipped, and left neither a
    result key nor a recorded error has not run — and a gate that reports nothing
    about a leg it was asked to run must not call that a pass. This is the shape
    a leg takes when it returns before its first ``save``, which is exactly what
    ``run_leg``'s recording would otherwise be unable to distinguish from a leg
    the caller never requested.
    """
    gate = _gate_module()
    clean = gate._summarize([{"guard": False, "verdict": {"bit_identical": True}}])
    requested = {"legs": ["expansion", "synthetic", "fill"],
                 "synthetic": {"summary": clean}}
    built = gate.build_summary(requested, {"licensed": True, "reasons": []})
    assert not built["passed"]
    assert any("leg fill was requested" in f for f in built["failures"]), \
        built["failures"]
    # A cleanly skipped device leg is NOT this failure.
    skipped = dict(requested, device_legs_skipped="no cupy on this host")
    assert gate.build_summary(skipped,
                              {"licensed": True, "reasons": []})["passed"]


def test_the_planted_fill_words_cover_every_needle_class():
    """The three classes the fill's mutations live in, each MEASURED reachable.

    The diagonal pairing this replaced (``words[i % 8]`` / ``words[(i + 3) % 8]``)
    hit 0 of the 8 plane-wise pairs at mirror phase +1, 1 of 8 at phase -1 and 0
    of the 8 commute pairs — so ``plane_wise_parity_fill`` and
    ``identity_shortcut_on_even_mirror`` reported CAUGHT entirely off the phase
    = -1 sign flip, and ``reverse_axis_order`` could not diverge at all.
    """
    gate = _gate_module()
    pairs = [(np.float32(r), np.float32(i)) for r, i in gate.FILL_NEEDLE_PAIRS]

    def word(re_, im_):
        out = np.empty(1, np.complex64)
        out.real = re_
        out.imag = im_
        return out

    def bytes_of(array):
        return np.ascontiguousarray(array).view(np.uint32)

    planewise = {1: 0, -1: 0}
    identity = 0
    commute = 0
    for re_, im_ in pairs:
        z = word(re_, im_)
        for phase in (1, -1):
            flipped = word(np.float32(phase) * re_, np.float32(phase) * im_)
            if not np.array_equal(bytes_of(phase * z), bytes_of(flipped)):
                planewise[phase] += 1
        if not np.array_equal(bytes_of(1 * z), bytes_of(z)):
            identity += 1
        if not np.array_equal(bytes_of(1 * (-1 * z)), bytes_of(-1 * (1 * z))):
            commute += 1
    assert planewise[1] > 0 and planewise[-1] > 0, planewise
    assert identity > 0
    assert commute > 0, "reverse_axis_order has no reachable needle"


# ---------------------------------------------------------------------------
# The needle-placement policy and the comparator policy it forced
# ---------------------------------------------------------------------------

def test_the_extreme_fill_needle_stays_FINITE_through_the_whole_path():
    """THE NEEDLE-PLACEMENT POLICY, pinned as an inequality instead of a comment.

    A needle may sit anywhere in the float32 range whose downstream products stay
    finite, and nowhere past it. The shipped ``3.4e38`` pair broke that: it
    overflowed the curl, and the byte comparison then rested on a NaN's sign and
    payload — unspecified by IEEE 754-2019 (§6.2.1, §6.2.3, §7.2) and MEASURED to
    vary by numpy version on hash-identical sources (17/17 on 2.4.3, 14/17 on
    2.2.6, every differing word a NaN against a NaN).

    The bound is recomputed HERE from the constants the run actually uses — the
    PML extremes off the validator's own case list, its ``dt/dx``, its beta and
    its step count — so raising ``STEPS``, thickening the absorber or moving the
    Courant number fails this test rather than silently reintroducing the NaNs.
    """
    gate = _gate_module()
    validator = _validator_module()
    flt_max = float(np.finfo(np.float32).max)

    # (1) every shipped needle word is finite. That is the policy in one line.
    for real, imaginary in gate.FILL_NEEDLE_PAIRS:
        assert np.isfinite(np.float32(real)), (real, imaginary)
        assert np.isfinite(np.float32(imaginary)), (real, imaginary)

    # (2) ...and the MAGNITUDE class was fixed, not deleted. The ablation that
    # localised the defect used 1.5; taking that would drop the class entirely.
    magnitude = max(max(abs(float(real)), abs(float(imaginary)))
                    for real, imaginary in gate.FILL_NEEDLE_PAIRS)
    assert magnitude == float(gate.FILL_NEEDLE_MAGNITUDE)
    assert magnitude > 1e20, "the extreme-magnitude needle class is gone"

    # (3) the signed-zero and subnormal classes are untouched — job 2343 proved
    # the signed-zero class reachable at driver level, so they are load-bearing.
    words = [np.float32(value) for pair in gate.FILL_NEEDLE_PAIRS
             for value in pair]
    bits = np.array([w.view(np.uint32) for w in words], dtype=np.uint32)
    assert int((bits == np.uint32(0x80000000)).sum()) > 0, "no -0.0 needle left"
    exponent = (bits >> np.uint32(23)) & np.uint32(0xFF)
    significand = bits & np.uint32(0x7FFFFF)
    assert int(((exponent == 0) & (significand != 0)).sum()) > 0, \
        "no subnormal needle left"

    # (4) THE BOUND. Coefficient extremes measured off the validator's own grids.
    extremes = {"kms": 0.0, "sinv": 0.0, "kps": 0.0}
    dtdx = beta_coefficient = 0.0
    for _label, axes, declaration, cell, phase, storage, beta, kx in \
            validator.CASES:
        grid, _fields, pml = validator._build(axes, declaration, cell, phase,
                                              storage, beta, kx, np)
        dtdx = max(dtdx, float(np.float32(grid.dt / grid.dx)))
        beta_coefficient = max(beta_coefficient,
                               abs(2.0 * np.pi * float(beta) * float(grid.dt)))
        for suffix in ("", "_h"):
            for axis in "xyz":
                for stem in extremes:
                    extremes[stem] = max(extremes[stem], float(np.abs(
                        np.asarray(getattr(pml, f"{stem}_{axis}{suffix}"))).max()))
    kms, sinv, kps = extremes["kms"], extremes["sinv"], extremes["kps"]

    # Per CURL sub-step, bounding every state word by S:
    #   shift  (unit-modulus Bloch multiply)          <= 2 S
    #   curl   dtdx * ((s1 - f1) + (f2 - s2))         <= 6 dtdx S
    #   beta   curl - (c (x) partner)                 <= + 2 |c| S
    #   fu'    (fu kms - curl) sinv                   <= (kms + 6 dtdx + 2|c|) S
    #   field' (field kms_u + fu' - fu) sinv_u        <= (kms + fu'/S + 1) S
    curl_gain = (2 * kms + 6 * dtdx + 2 * beta_coefficient + 1) * sinv
    # Per CONSTITUTIVE sub-step: field += kps * fw - kms * fw_previous.
    constitutive_gain = 1 + kps + kms
    # A full step is step_B, update_H, step_D, update_E.
    gain = (curl_gain ** 2 * constitutive_gain ** 2) ** validator.STEPS
    assert magnitude * gain <= flt_max, (
        f"needle {magnitude:g} times the worst-case path gain {gain:g} exceeds "
        f"FLT_MAX; the largest admissible needle is {flt_max / gain:g}")


def test_the_validator_compares_NaN_by_IS_NAN_and_everything_else_by_BITS():
    """The comparator policy, word by word.

    Byte-exact stays byte-exact for every class the standard DOES specify —
    finite, signed zero, subnormal, infinity. The single exemption is NaN against
    NaN, because IEEE 754-2019 leaves a NaN's sign bit (§6.2.1) and payload
    (§6.2.3 is a *should*; §7.2 does not fix the payload an invalid operation
    delivers) unspecified, and §5.11 makes NaN unordered even with itself. A NaN
    opposite anything that is not a NaN must still FAIL.
    """
    validator = _validator_module()

    def words(*values):
        return np.array(values, dtype=np.uint32)

    finite = np.uint32(np.float32(1.5).view(np.uint32))
    quiet_positive = np.uint32(0x7FC00000)      # the GPU-host divergence, side A
    quiet_negative = np.uint32(0xFFC00000)      # ... and side B
    other_payload = np.uint32(0x7FDEADBE)
    signalling = np.uint32(0x7F800001)
    positive_infinity = np.uint32(0x7F800000)
    negative_infinity = np.uint32(0xFF800000)
    negative_zero = np.uint32(0x80000000)
    subnormal = np.uint32(0x00000001)

    # A NaN where the other side has a NUMBER is a real transcription defect.
    assert validator.compare_words(words(quiet_positive), words(finite)) == (1, 1, 0)
    assert validator.compare_words(words(finite), words(quiet_positive)) == (1, 1, 0)
    # ...including against zero and against infinity, which are not NaNs.
    assert validator.compare_words(words(quiet_positive),
                                   words(np.uint32(0)))[0] == 1
    assert validator.compare_words(words(quiet_positive),
                                   words(positive_infinity))[0] == 1

    # NaN against NaN passes however the sign and payload differ, and is COUNTED.
    for other in (quiet_negative, other_payload, signalling):
        mismatched, nan_words, exempt = validator.compare_words(
            words(quiet_positive), words(other))
        assert mismatched == 0, hex(int(other))
        assert nan_words == 1 and exempt == 1, (hex(int(other)), nan_words, exempt)
    # An identical NaN pair is still counted by the census but forgives nothing.
    assert validator.compare_words(words(quiet_positive),
                                   words(quiet_positive)) == (0, 1, 0)

    # Everything the standard DOES specify stays raw-bit, unchanged.
    assert validator.compare_words(words(negative_zero),
                                   words(np.uint32(0))) == (1, 0, 0)
    assert validator.compare_words(words(subnormal),
                                   words(np.uint32(0))) == (1, 0, 0)
    assert validator.compare_words(words(positive_infinity),
                                   words(negative_infinity)) == (1, 0, 0)
    assert validator.compare_words(words(positive_infinity),
                                   words(positive_infinity)) == (0, 0, 0)
    assert validator.compare_words(words(finite, negative_zero, subnormal),
                                   words(finite, negative_zero, subnormal)) \
        == (0, 0, 0)


def test_the_validator_REPORTS_its_NaN_census_and_the_shipped_needle_makes_it_zero(
        monkeypatch):
    """The exemption may not be a place a NaN bloom can hide.

    Two runs of the SAME case: with the shipped finite needle the census is zero
    — the run makes no use of the exemption at all, so this validator's verdict
    and a raw-bit verdict are the same measurement — and with the overflow needle
    restored the census is non-zero and says so out loud. Case 13 is the one that
    first diverged on the GPU host.
    """
    gate = _gate_module()
    validator = _validator_module()
    case = next(c for c in validator.CASES if c[0] == "Y/periodic/+1/even+beta")
    monkeypatch.setattr(validator, "CASES", (case,))

    clean = validator.run(np)
    assert clean["identical"] == clean["ran"] == 1
    assert clean["nan_words"] == 0 and clean["nan_exempt"] == 0
    assert clean["nan_checks"] == 0
    assert clean["needle_magnitude"] == float(gate.FILL_NEEDLE_MAGNITUDE)
    assert clean["cases"][0]["nan_words"] == 0

    overflowing = tuple((3.4e38 if real == gate.FILL_NEEDLE_MAGNITUDE else real,
                         imaginary)
                        for real, imaginary in gate.FILL_NEEDLE_PAIRS)
    assert overflowing != gate.FILL_NEEDLE_PAIRS
    monkeypatch.setattr(gate, "FILL_NEEDLE_PAIRS", overflowing)
    with np.errstate(over="ignore", invalid="ignore"):
        blooming = validator.run(np)
    assert blooming["nan_words"] > 0, \
        "the census cannot detect a bloom it does not count"
    assert blooming["nan_checks"] > 0
    assert blooming["cases"][0]["nan_words"] == blooming["nan_words"]


def test_the_artifact_STAMPS_THE_POLICY_COUNTERS_AFTER_the_work_not_before(
        tmp_path, monkeypatch):
    """The strip counters in the artifact must be the ones the RUN produced.

    ``policy_stamp`` returns a COPY of the strip's counters
    (``int(state["calls"])``, gate_triton_complex.py:404), so a stamp taken while
    the results dict is being built freezes them at zero no matter what the run
    then compiles. That is how a certified artifact came to advertise
    ``ieee_keep_ftz_stripped`` beside ``nvrtc_calls: 0`` on a run whose expansion
    sub-record — stamped after its own work — read 6 compiles with -ftz=true
    removed from all 6.

    The stub counts stamps, so the assertion is about WHICH call reached the
    artifact, not about any device: on this host there is no CuPy and the real
    counters would be zero either way.
    """
    gate = _gate_module()
    stamps: list = []

    def counting_stamp(backend_name):
        stamps.append(backend_name)
        return {"policy": "stub", "stamp_ordinal": len(stamps)}

    monkeypatch.setattr(gate.gate, "policy_stamp", counting_stamp)
    # `cupy` is stubbed with NumPy at import, so `cp is not None` and main() tries
    # to install the real strip; there is no NVRTC seam behind the stub. The
    # installation is not what is under test here.
    monkeypatch.setattr(gate.gate, "install_ftz_strip",
                        lambda: {"installed": False, "reason": "stubbed in test"})
    # Same reason: provenance asks the stub for device properties it cannot have.
    monkeypatch.setattr(gate, "write_provenance",
                        lambda directory: {"device": "stubbed in test"})
    out = tmp_path / "gate.json"
    gate.main(["--legs", "reference", "--out", str(out)])

    written = json.loads(out.read_text())
    assert len(stamps) >= 2, \
        f"the policy was stamped {len(stamps)} time(s); the artifact keeps a " \
        f"pre-run snapshot"
    assert written["subnormal_policy"]["stamp_ordinal"] == len(stamps), \
        "the artifact kept an EARLIER stamp than the last one taken"
    # The run really did happen between the two stamps, or this proves nothing.
    assert written["reference"]["ran"] == written["reference"]["identical"] == 17


def test_the_curl_mutation_leg_BUILDS_NO_ROW_the_array_path_cannot_build(
        monkeypatch):
    """The mutation leg's (shape, config) product is not rectangular.

    ``SHAPES[:3]`` carries the 2-D sheet ``(9, 7, 1)`` and the config list
    carries ``fold_z_*``, so the raw product asks for a fold of a ONE-CELL axis.
    The array path cannot build one — the near ghost reads cell 2 — and the
    reference shift raises ``IndexError`` rather than returning a wrong answer,
    which took the WHOLE mutation leg down with it and left fourteen of the
    seventeen mutants unrun. ``run_synthetic`` has always screened those rows
    through :func:`viable`; this pins that the mutation leg does too.

    ``one_curl_case`` is replaced by a recorder, so the test needs no Triton and
    no device: what is under test is which rows the leg ASKS FOR.
    """
    gate = _gate_module()
    asked: list = []

    def recorder(shape, config, phase_set, dtdx, mirror_phase, offsets, sub_step,
                 guard, expansion, beta, storage, kernel=None):
        asked.append((tuple(shape), config["name"], phase_set["phases"]))
        return {"config": config["name"],
                "verdict": {"bit_identical": True, "differing_floats": 0}}

    monkeypatch.setattr(gate, "one_curl_case", recorder)
    result = gate.run_mutation_curl(1, object(), "curl", False)

    assert asked, "the leg asked for no row at all"
    for shape, config_name, phases in asked:
        config = next(c for c in gate.CONFIGS if c["name"] == config_name)
        reason = gate.viable(shape, config, phases)
        assert reason is None, \
            f"the leg asked for {config_name} on {shape}, which cannot be built: {reason}"

    # The guard must be LOAD-BEARING, not decorative: the raw product really does
    # contain unbuildable rows, and they are recorded rather than dropped in
    # silence — a mutant left with no viable row must reach DISARMED visibly.
    skipped = result["summary"]["skipped_unbuildable"]
    assert skipped, "no row was screened; this guard would be untested scaffolding"
    assert any(entry["shape"] == [9, 7, 1] and entry["config"].startswith("fold_z")
               for entry in skipped), skipped
    for entry in skipped:
        assert "cells" in entry["reason"], entry


def test_the_validator_CROSSES_the_device_boundary_EXPLICITLY_never_implicitly():
    """``run(cp)`` reads device fields; the reference stays on NumPy.

    The gate calls ``run(cp)`` when CuPy is present, so every field read and every
    PML table crosses a backend boundary, while the reference half is stepped with
    NumPy unconditionally. CuPy REFUSES the implicit crossing — ``np.asarray`` on
    a ``cupy.ndarray`` raises TypeError rather than transferring — so a validator
    that spells the crossing implicitly passes on the laptop, passes standalone on
    the cluster (line 64 stubs cupy with NumPy), and raises only inside the gate,
    on the device, which is exactly where it did.

    The stub reproduces CuPy's two-part contract rather than importing it: no
    implicit ``__array__``, an explicit ``.get()``. A test that used a real NumPy
    array could not fail, because the implicit spelling works on NumPy.
    """
    validator = _validator_module()

    class RefusesImplicitConversion:
        """CuPy's contract: ``.get()`` transfers, ``np.asarray`` raises."""

        def __init__(self, host):
            self._host = host

        def __array__(self, *args, **kwargs):
            raise TypeError("Implicit conversion to a NumPy array is not "
                            "allowed. Please use `.get()` to construct a NumPy "
                            "array explicitly.")

        def get(self):
            return self._host

    # The stub really does refuse, or nothing below is a measurement.
    payload = np.array([1.5, -0.0, np.float32(1e-45)], dtype=np.float32)
    with pytest.raises(TypeError):
        np.asarray(RefusesImplicitConversion(payload))

    # The crossing is bit-preserving: the uint32 words this module compares are
    # the same whichever side of the bus they were read from.
    device = RefusesImplicitConversion(payload)
    assert np.array_equal(validator._words(device), validator._words(payload))
    # Signed zero and the subnormal survive it — the classes the whole gate is
    # about, and the ones a lossy "conversion" would quietly normalise.
    words = validator._words(device)
    assert int(words[1]) == 0x80000000
    assert int(words[2]) == 0x00000001

    # A host array is passed through untouched, not copied through `.get`.
    assert validator._host(payload) is payload

    # The PML tables take the same route. A table left on the device would be
    # mixed into NumPy expressions against host operands, which either raises or
    # — the dangerous outcome — hands the whole reference step to the device.
    class Tables:
        pass

    tables = Tables()
    for axis in "xyz":
        for stem in ("kms", "sinv"):
            for suffix in ("", "_h"):
                setattr(tables, f"{stem}_{axis}{suffix}",
                        RefusesImplicitConversion(payload))
    for half_integer in (False, True):
        coefficients = validator._coefficients_of(tables, half_integer)
        assert set(coefficients) == {f"{stem}_{axis}"
                                     for axis in "xyz"
                                     for stem in ("kms", "sinv")}
        for name, table in coefficients.items():
            assert isinstance(table, np.ndarray), \
                f"{name} reached the NumPy reference still on the device"


def test_the_real_storage_seed_carries_signed_zeros_and_subnormals():
    """K3a is the REAL arm and was byte-compared only against uniform random
    states: measured 0 signed zeros and 0 subnormals in 11583 words, so it was
    provably blind to the class the ieee_keep_ftz_stripped policy exists for and
    to the module's own single-subtract claim."""
    gate = _gate_module()
    shape = gate.SHAPES[0]
    rng = np.random.default_rng(gate.SEED + 401)
    plain = np.ascontiguousarray(
        rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))
    plain_words = plain.view(np.uint32).ravel()
    assert int((plain_words == np.uint32(0x80000000)).sum()) == 0

    seeded = gate._seed_host_real(shape, np.random.default_rng(gate.SEED + 401))
    words = np.ascontiguousarray(seeded).view(np.uint32).ravel()
    assert int((words == np.uint32(0x80000000)).sum()) > 0
    exponent = (words >> 23) & np.uint32(0xFF)
    mantissa = words & np.uint32(0x7FFFFF)
    subnormal = int(((exponent == 0) & (mantissa != 0)).sum())
    assert subnormal > 0, "no subnormal survived the real seed"
    # No overflow-adjacent magnitude: a host reference and a device kernel need
    # not agree on a NaN PAYLOAD, and `inf - inf` in the recurrence would compare
    # a platform choice rather than the transcription.
    assert float(np.max(np.abs(seeded[np.isfinite(seeded)]))) < 1e30


def test_the_entry_mutations_report_whether_they_armed_anything():
    """The host-side twin of a source mutation's ``hits == 0``.

    ``reverse_axis_order`` on a ONE-element entry list is the identity, and
    without the count the record reads ``verdict: MEASURED, diverged: 0`` —
    indistinguishable from "the harness never varied anything".
    """
    gate = _gate_module()
    grid = Grid(resolution=10.0, cell_size=(1.6, 2.0, 0.0), dimensions=2,
                courant=0.35, boundaries="periodic",
                symmetry=(Mirror("Y", 1),), xp=np)
    single = [dict(e) for e in
              gate.folded_complex.ghost_fill_axis_entries(grid, "B")]
    assert len(single) == 1
    _entries, armed, _disc = gate.apply_entry_mutation(
        single, "reverse_axis_order", grid)
    assert armed == 0

    two = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), courant=0.35,
               boundaries="periodic", symmetry=(Mirror("X", 1), Mirror("Y", -1)),
               xp=np)
    pair = [dict(e) for e in
            gate.folded_complex.ghost_fill_axis_entries(two, "B")]
    assert len(pair) == 2
    _entries, armed, _disc = gate.apply_entry_mutation(
        pair, "reverse_axis_order", two)
    assert armed == 1


def test_armed_is_not_arming_for_the_order_mutation():
    """The count that makes the other count mean something.

    Reversing the two entries of a SAME-PHASE two-folded-axis grid CHANGES the
    list — ``armed == 1`` — and cannot change a stored byte: the doubly-written
    corner's two coefficients are equal and ``c (x) (c (x) z)`` commutes
    bit-exactly. Measured over the gate's own fill matrix, that grid
    (``FILL_GRIDS[10]``) diverges in 0 words on all 8 of its (family, state)
    cells while the MIXED-phase one diverges in 5. A leg judged on ``armed_rows``
    alone therefore records ``MEASURED, diverged: 0`` for a no-op and passes.
    """
    gate = _gate_module()
    same = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), courant=0.35,
                boundaries="periodic", symmetry=(Mirror("X", 1), Mirror("Y", 1)),
                xp=np)
    mixed = Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), courant=0.35,
                 boundaries="periodic", symmetry=(Mirror("X", 1), Mirror("Y", -1)),
                 xp=np)
    for family in ("B", "D"):
        entries = [dict(e) for e in
                   gate.folded_complex.ghost_fill_axis_entries(same, family)]
        _e, armed, discriminating = gate.apply_entry_mutation(
            entries, "reverse_axis_order", same)
        assert (armed, discriminating) == (1, 0), (family, armed, discriminating)

        entries = [dict(e) for e in
                   gate.folded_complex.ghost_fill_axis_entries(mixed, family)]
        _e, armed, discriminating = gate.apply_entry_mutation(
            entries, "reverse_axis_order", mixed)
        assert (armed, discriminating) == (1, 1), (family, armed, discriminating)

    # A VALUE mutation that rewrites a word the plan READS has both counts at 1.
    entries = [dict(e) for e in
               gate.folded_complex.ghost_fill_axis_entries(same, "B")]
    _e, armed, discriminating = gate.apply_entry_mutation(
        entries, "nonzero_imaginary_words", same)
    assert (armed, discriminating) == (1, 1)

    # ...and one that rewrites a DEAD word does not. A folded METALLIC axis runs
    # no far pass, so `far_words` is compiled away and `far_fill_uses_plus_phase`
    # is armed and empty there — measured 96 armed rows against 64 discriminating
    # over the gate's own fill matrix.
    metallic = Grid(resolution=10.0, cell_size=(1.6, 2.0, 0.0), dimensions=2,
                    courant=0.35, boundaries={"y": "metallic"},
                    symmetry=(Mirror("Y", 1),), xp=np)
    entries = [dict(e) for e in
               gate.folded_complex.ghost_fill_axis_entries(metallic, "B")]
    assert not any(entry["far"] for entry in entries)
    _e, armed, discriminating = gate.apply_entry_mutation(
        entries, "far_fill_uses_plus_phase", metallic)
    assert (armed, discriminating) == (1, 0)

    periodic = [dict(e) for e in
                gate.folded_complex.ghost_fill_axis_entries(same, "B")]
    _e, armed, discriminating = gate.apply_entry_mutation(
        periodic, "far_fill_uses_plus_phase", same)
    assert (armed, discriminating) == (1, 1)


def test_fusing_the_passes_is_counted_on_the_rows_where_it_can_differ():
    """``len(entries) > 1`` is not the arming test for ``fuse_fill_passes``.

    Only a target NEAR on one folded axis and FAR on another sees a different
    order under ``_launch_pass(1, 1)``. A second fold that is METALLIC runs no far
    pass at all, so a two-entry list can still be a provable no-op.
    """
    gate = _gate_module()
    periodic = gate.folded_complex.ghost_fill_axis_entries(
        Grid(resolution=10.0, cell_size=(2.0, 2.1, 1.2), courant=0.35,
             boundaries="periodic", symmetry=(Mirror("X", 1), Mirror("Y", 1)),
             xp=np), "B")
    assert len(periodic) == 2
    assert gate.fusion_is_discriminating(periodic) == 1
    # The same list with the far pass switched off on BOTH axes — what a folded
    # METALLIC pair produces — carries no near/far pair on any target.
    metallic = [dict(entry, far=False) for entry in periodic]
    assert gate.fusion_is_discriminating(metallic) == 0
    assert gate.fusion_is_discriminating(list(periodic[:1])) == 0


def test_a_fill_cell_blind_in_every_state_family_is_a_FAILURE():
    """The granularity matters in both directions.

    A SINGLE family reading census 0 is a predicted null with a reason: on a
    folded METALLIC axis at mirror phase +1 the near coefficient is ``+1`` and
    there is no far fill, so a quiet grid stores no signed zero at all (measured
    0 at 1, 2, 3 and 4 driven steps, against 16..650 on every other row). A CELL
    blind in EVERY family cannot see any of the fill's needles, and recording
    that under ``vacuous_fill_states`` while passing is the silence the
    discipline forbids.
    """
    gate = _gate_module()
    spec = {"axis": "Y", "boundaries": "metallic", "phase": 1,
            "cell": (1.6, 2.0, 0.0)}
    predicted_null = [
        {"spec": spec, "family": "D", "state": state, "census": census,
         "verdict": {"bit_identical": True}}
        for state, census in (("zero_init_absorber", 0), ("engineered_rows", 24),
                              ("engineered_rows_post_step", 0),
                              ("live_post_step", 0))]
    assert gate.blind_fill_cells(predicted_null) == []
    built = gate.build_summary(
        {"fill": {"gate": {"ran": 4, "identical": 4,
                           "cases": predicted_null}}},
        {"licensed": True, "reasons": []})
    assert built["passed"], built["failures"]

    blind = [dict(case, census=0) for case in predicted_null]
    assert len(gate.blind_fill_cells(blind)) == 1
    built = gate.build_summary(
        {"fill": {"gate": {"ran": 4, "identical": 4, "cases": blind}}},
        {"licensed": True, "reasons": []})
    assert not built["passed"]
    assert any("blind in every state family" in f
               for f in built["failures"]), built["failures"]


def test_the_recorded_reason_for_a_zero_census_is_the_measured_one():
    """A predicted null is only a predicted null if its REASON is true.

    The recorded reason said a census of 0 belongs to the folded METALLIC phase
    = +1 rows, "while the periodic and phase = -1 rows read 16..650". MEASURED
    over the whole 12 x 2 x 4 matrix on this laptop, that is false for one of the
    four families: ``live_post_step`` reads 0 on EVERY row, periodic ones
    included, because four driven steps from a random seed leave generic nonzero
    floats. It is the realistic-state arm, not a signed-zero arm, and the two
    reasons are now recorded separately. ``engineered_rows`` is what actually
    carries the metallic +1 cells — it is the only family that never reads 0.
    """
    gate = _gate_module()
    fold = importlib.import_module(MODULE_NAME)
    metallic = {"axis": "Y", "boundaries": "metallic", "phase": 1,
                "cell": (1.6, 2.0, 0.0)}
    periodic = {"axis": "Y", "boundaries": "periodic", "phase": 1,
                "cell": (1.6, 2.0, 0.0)}

    def census(spec, family, state):
        grid, fields, pml = gate.build_folded_fields(spec, np, thin_pml=True)
        names = fold.GHOST_FILL_FAMILIES[family]["targets"]
        with np.errstate(over="ignore", invalid="ignore"):
            gate.seed_fill_state(fields, grid, names, state, pml)
            reference = Fields(grid=grid, force_complex_fields=True)
            reference.enable_pml_storage()
            for name in names:
                getattr(reference, name)[...] = getattr(fields, name)
            getattr(stepping, f"fill_symmetry_bc_{family}")(reference)
            getattr(stepping, f"fill_folded_far_ghosts_{family}")(reference)
        return gate.signed_zero_census(
            np, {name: getattr(reference, name) for name in names})

    for family in ("B", "D"):
        # live_post_step is 0 on BOTH terminations — not a metallic-only null.
        assert census(metallic, family, "live_post_step") == 0
        assert census(periodic, family, "live_post_step") == 0
        # the metallic +1 cell rests entirely on engineered_rows
        assert census(metallic, family, "zero_init_absorber") == 0
        assert census(metallic, family, "engineered_rows_post_step") == 0
        assert census(metallic, family, "engineered_rows") > 0
        # ...and the periodic row reaches the class from the quiet absorber too
        assert census(periodic, family, "zero_init_absorber") > 0
    text = GATE_PATH.read_text(encoding="utf-8")
    assert "reads 0 on EVERY row of the matrix" in text
    assert "16..650" not in text, "the refuted census range is still recorded"


def test_the_zero_init_leg_byte_compares_a_kernel():
    """The leg named for the ZERO-INIT NEGATIVE-COEFFICIENT-ABSORBER case ran no
    kernel: it stepped the array path and counted signed zeros, which establishes
    REACHABILITY, not agreement. Its verdict now carries a bit-identity half and
    a refusal to substitute fails it."""
    gate = _gate_module()
    source = GATE_PATH.read_text(encoding="utf-8")
    body = source.split("def _zero_init_bit_identity")[1].split("\ndef ")[0]
    assert "bit_compare" in body
    assert "plan_folded_complex_pml_curl_from_arrays" in body
    assert "plan_folded_mirror_ghost_fill_complex_from_arrays" in body
    verdict = {"census": [1, 1], "passed": False,
               "bit_identity": {"ran": 0, "identical": 0, "passed": False},
               "failure": "no licensed EXPANSION; the leg could launch nothing"}
    built = gate.build_summary({"zero_init": verdict},
                               {"licensed": True, "reasons": []})
    assert not built["passed"]


def test_the_zero_init_census_expectation_is_the_one_this_laptop_produces():
    """``ZERO_INIT_LAPTOP_CENSUS`` is the artifact's ``laptop_expectation``; a
    stale one misleads every reviewer comparing the device census against it."""
    gate = _gate_module()
    grid, fields, pml = gate.build_folded_fields(gate.ZERO_INIT_SPEC, np,
                                                 thin_pml=True)
    for name in gate.ALL_STATE:
        volume = getattr(fields, name, None)
        if volume is not None:
            volume[...] = 0
    kms = min(float(np.asarray(getattr(pml, f"kms_{a}")).min()) for a in "xyz")
    kms_h = min(float(np.asarray(getattr(pml, f"kms_{a}_h")).min()) for a in "xyz")
    assert min(kms, kms_h) < 0.0, (kms, kms_h)
    tracked = {n: getattr(fields, n) for n in gate.ALL_STATE
               if getattr(fields, n, None) is not None}
    census = []
    for _ in range(gate.ZERO_INIT_STEPS):
        gate._drive_array_path(fields, pml, 1)
        census.append(gate.signed_zero_census(np, tracked))
    assert tuple(census) == tuple(gate.ZERO_INIT_LAPTOP_CENSUS), census


def test_unary_minus_addend_arms_both_expansion_arms():
    """The claim under audit was that ``fold_zero_cross_terms`` and
    ``unary_minus_addend`` rewrite BOTH arms. It was true for the first and FALSE
    for the second: all three sites sat inside ``if EXPANSION == FMA_V1``, so on
    a NAIVE-binding platform the mutant's source differed (no DISARMED) while its
    compiled body did not, ``_ptx_differs`` reported overlap, and the leg failed
    for a non-defect. The NAIVE arm spells the term as a SUBTRACT, so arming it
    means ``a - b`` -> ``a + (-b)`` — the same lowering by another route.
    """
    gate = _gate_module()
    sources = _mutation_sources()
    for name in ("unary_minus_addend", "fold_zero_cross_terms"):
        transform, body, _entry, _note = gate.SOURCE_MUTATIONS[name]
        shipped = sources[body]
        fma_only = "\n".join(_expansion_arm(shipped, "FMA_V1"))
        naive_only = "\n".join(_expansion_arm(shipped, "NAIVE"))
        _text, fma_hits = transform(fma_only)
        _text, naive_hits = transform(naive_only)
        assert fma_hits > 0, (name, "no FMA_V1 site")
        assert naive_hits > 0, (name, "no NAIVE site — DISARMED on a NAIVE "
                                      "platform, and silently so")


def _expansion_arm(source: str, arm: str):
    """The lines of ``source`` that belong to one EXPANSION arm.

    ``if EXPANSION == FMA_V1:`` opens the fused arm and its ``else:`` the naive
    one; both are indented one level deeper than the ``if``.
    """
    out, state, depth = [], None, 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("if EXPANSION == FMA_V1:"):
            state, depth = "FMA_V1", len(line) - len(line.lstrip())
            continue
        if state is not None and stripped == "else:" and (
                len(line) - len(line.lstrip())) == depth:
            state = "NAIVE"
            continue
        if state is not None and stripped and (
                len(line) - len(line.lstrip())) <= depth:
            state = None
        if state == arm and stripped:
            out.append(stripped)
    return out


def test_an_unfolded_beta_grid_is_refused_by_both_beta_curls(fc):
    """OVER-COVERAGE. ``special_kz``'s beta predicates refuse a fold but do not
    REQUIRE one, so they stay live on an unfolded beta grid — and both of this
    file's beta predicates admitted exactly that grid, in both storages. The
    module's own wiring note argues no shipped predicate can admit a
    configuration these admit; this is what makes the note true."""
    for complex_storage in (False, True):
        grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                    courant=0.35, boundaries="periodic", beta=BETA_EIGSRC, xp=np)
        fields = Fields(grid=grid, force_complex_fields=complex_storage)
        fields.enable_pml_storage()
        pml = PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))
        if complex_storage:
            reasons = _reasons(fc.folded_beta_bloch_pml_curl_coverage(
                fields, pml, "step_B", probe=_all_patterns_record()))
        else:
            reasons = _reasons(fc.folded_beta_pml_curl_coverage(
                fields, pml, "step_B"))
        assert any("no mirror plane is active" in r for r in reasons), reasons
    # ...and a FOLDED beta grid is still admitted, in both storages.
    fields, pml = _fold_grid(beta=BETA_EIGSRC)
    assert _reasons(fc.folded_beta_bloch_pml_curl_coverage(
        fields, pml, "step_B", probe=_all_patterns_record())) == []
    real_fields, real_pml = _fold_grid(beta=BETA_EIGSRC, complex_storage=False)
    assert _reasons(fc.folded_beta_pml_curl_coverage(
        real_fields, real_pml, "step_B")) == []


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
@pytest.mark.parametrize("target", ["Bx", "Dx"])
def test_the_real_beta_curl_checks_all_six_curl_targets_for_conductivity(
        fc, sub_step, target):
    """K3a scoped the conductivity clause to the NAMED sub-step's three targets
    while every sibling predicate in the file scopes it to all six. An ELECTRIC
    conductivity was therefore ADMITTED by K3a for ``step_B`` and REFUSED by
    every other predicate for the same run, so a composed run could only ever be
    half-substituted."""
    fields, pml = _fold_grid(beta=BETA_EIGSRC, complex_storage=False)
    shim = FieldsShim(
        fields, condfac_for=lambda name: object() if name == target else None)
    reasons = _reasons(fc.folded_beta_pml_curl_coverage(shim, pml, sub_step))
    assert any(f"conductivity is installed on {target}" in r
               for r in reasons), reasons
    # The complex twin has always checked all six; they must agree.
    complex_fields_, complex_pml = _fold_grid(beta=BETA_EIGSRC)
    complex_shim = FieldsShim(
        complex_fields_,
        condfac_for=lambda name: object() if name == target else None)
    twin = _reasons(fc.folded_beta_bloch_pml_curl_coverage(
        complex_shim, complex_pml, sub_step, probe=_all_patterns_record()))
    assert any(f"conductivity is installed on {target}" in r for r in twin), twin


def test_the_parity_probe_measures_the_scalar_operand_shape():
    """The array path's fill is a SCALAR times an array (``phase * plane`` with a
    PYTHON INT, S:1450-1451), and the headline risk is a real-scalar fast path on
    exactly that multiply. The probe classified an ARRAY times an array, which
    forces the ``'FF->F'`` loop by construction and so cannot see it."""
    gate = _gate_module()
    rng = np.random.default_rng(gate.SEED + 913)
    c_re, c_im, z_re, z_im = gate._parity_operands(rng)
    assert c_re.size == 2 * gate.PARITY_BLOCK
    z = gate.gate._interleave_c8(z_re, z_im)
    c = gate.gate._interleave_c8(c_re, c_im)
    spellings = gate._parity_spellings(np, z, c)
    assert set(spellings) == {"python_int_scalar_left", "numpy_scalar_left",
                              "array_left"}
    detail = gate.measure_parity_pattern(np)
    assert "python_int scalar" in detail["operand_shape"]
    assert detail["spelling_deltas"]["python_int_scalar_left"] == 0
    # On NumPy the three agree; a platform where they do not is a named finding,
    # and the licence refuses rather than picking one silently.
    assert detail["spellings_agree"] is True
    gate_text = GATE_PATH.read_text(encoding="utf-8")
    assert "parity_spellings_agree" in gate_text
    assert "three spellings of the parity" in gate_text


def test_the_fill_state_families_are_four_and_the_quiet_one_steps():
    """``zero_init_absorber`` zeroed and never stepped, so on every phase = +1
    grid the full product and the plane-wise form both returned ``(+0.0, +0.0)``
    and 12 of the 66 fill cases were provably blind to four mutations at once.
    ``engineered_rows_post_step`` exists because the two-pass split's own
    justification needs a plant AND a step, and no other family does both."""
    gate = _gate_module()
    assert gate.STATE_FAMILIES == ("zero_init_absorber", "engineered_rows",
                                   "engineered_rows_post_step", "live_post_step")
    source = GATE_PATH.read_text(encoding="utf-8")
    body = source.split("def seed_fill_state")[1].split("\ndef ")[0]
    assert "_drive_array_path(fields, pml, ZERO_INIT_STEPS)" in body


def test_the_two_pass_fill_split_has_a_needle_and_the_needle_is_reachable():
    """The module states the two-pass split is load-bearing and cites a measured
    ``By`` divergence under a fused per-axis launch. Nothing in the gate exercised
    that claim, and no state family could reach the class: ``seed_fill_state``
    planted without stepping or stepped without planting, and the plant's diagonal
    word pairing carried none of the association-divergent pairs.

    MEASURED here on NumPy, on the gate's own two-folded-axis grid and its own
    seeding: fused-per-axis vs two-pass differs in 5 words of ``By`` (B family) /
    5 of ``Dx`` (D family) on ``engineered_rows``, and 3 / 4 on
    ``engineered_rows_post_step``.
    """
    gate = _gate_module()
    assert gate.HOST_MUTATIONS["fuse_fill_passes"] is True
    source = GATE_PATH.read_text(encoding="utf-8")
    assert "plan._launch_pass(1, 1, None)" in source

    fold = importlib.import_module(MODULE_NAME)
    spec = [s for s in gate.FILL_GRIDS
            if s["axis"] == "XY" and not isinstance(s["phase"], tuple)][0]
    caught = {}
    for family in ("B", "D"):
        for state in ("engineered_rows", "engineered_rows_post_step"):
            grid, fields, pml = gate.build_folded_fields(spec, np, thin_pml=True)
            names = fold.GHOST_FILL_FAMILIES[family]["targets"]
            with np.errstate(over="ignore", invalid="ignore"):
                gate.seed_fill_state(fields, grid, names, state, pml)
            entries = list(fold.ghost_fill_axis_entries(grid, family))
            base = {n: np.array(getattr(fields, n), copy=True) for n in names}
            split = {n: np.array(v, copy=True) for n, v in base.items()}
            fused = {n: np.array(v, copy=True) for n, v in base.items()}
            with np.errstate(over="ignore", invalid="ignore"):
                _fill_two_pass(split, names, entries)
                _fill_fused_per_axis(fused, names, entries)
            caught[(family, state)] = sum(
                int((np.ascontiguousarray(split[n]).view(np.uint32).ravel()
                     != np.ascontiguousarray(fused[n]).view(np.uint32).ravel()
                     ).sum()) for n in names)
    assert all(count > 0 for count in caught.values()), caught


def test_the_nonzero_imaginary_word_entry_mutation_is_actually_armed():
    """``synthesize_zero_imag_host`` set the entry's imaginary word to ``+0.0``,
    which is what ``mirror_parity_coefficients`` already returns — a PROVABLE
    no-op recorded as a predicted CAUGHT, so the leg could only ever report
    diverged = 0. Passing a NONZERO word is the armed form of the same claim."""
    gate = _gate_module()
    fold = importlib.import_module(MODULE_NAME)
    grid = Grid(resolution=10.0, cell_size=(1.6, 2.0, 0.0), dimensions=2,
                courant=0.35, boundaries="periodic",
                symmetry=(Mirror("Y", 1),), xp=np)
    entries = [dict(e) for e in fold.ghost_fill_axis_entries(grid, "B")]
    assert all(entry["near_words"][1] == 0.0 and entry["far_words"][1] == 0.0
               for entry in entries), "the shipped imaginary word is not +0.0"
    _same, armed, discriminating = gate.apply_entry_mutation(
        [dict(e) for e in entries], "nonzero_imaginary_words", grid)
    assert (armed, discriminating) == (1, 1)
    assert all(entry["near_words"][1] != 0.0 for entry in _same)
    with pytest.raises(ValueError, match="unknown entry mutation"):
        gate.apply_entry_mutation([dict(e) for e in entries],
                                  "synthesize_zero_imag", grid)


# ---------------------------------------------------------------------------
# OFF-DIAGONAL chi1inv on the curls and update_H — admission only, no new kernel
# ---------------------------------------------------------------------------
#
# Closes residual group (D)'s nine identity slots: step_B, step_D and update_H
# on TestArrayMetadata.test_array_metadata, TestHoleyWvgBands.test_fields_at_kx
# and solve-cw.py — the largest single block in the residual set and none of the
# three a TestLoadDump case. update_E on the same three rows STAYS REFUSED and is
# the best-motivated unbuilt kernel in the queue.

#: The device leg's verdicts (results/residual_closure_2026-08-15/device/
#: bodies/bodies.json), 8 complete cycles each, uint32 over the whole stored
#: inventory, with the +-0 lattice held live through every cycle.
_GROUP_D_MEASURED = {
    "D_complex_fold_offdiag_step_B": (0, 110592),
    "D_complex_fold_offdiag_step_D": (0, 110592),
    "D_complex_fold_offdiag_update_H": (0, 110592),
    "D_complex_fold_offdiag_update_E": (25041, 110592),
}

_OFFDIAG_ALL = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                "Hx", "Hy", "Hz",
                "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
                "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")


def _offdiag_seed(fields, seed=13):
    rng = np.random.default_rng(seed)
    for name in _OFFDIAG_ALL:
        array = getattr(fields, name, None)
        if array is None:
            continue
        real = rng.uniform(-0.4, 0.4, size=array.shape)
        if array.dtype.kind == "c":
            array[...] = (real + 1j * rng.uniform(-0.4, 0.4, size=array.shape)
                          ).astype(array.dtype)
        else:
            array[...] = real.astype(array.dtype)
    return fields


def _offdiag_fixture(rows=True, seed=13, **kwargs):
    """A folded complex triple carrying (or not) a surviving off-diagonal row."""
    fields, pml = _fold_grid(**kwargs)
    shape = fields.grid.shape
    eps = np.full(shape, 2.25, dtype=np.float32)
    inverse = (1.0 / eps).astype(np.float32)
    row = np.full(shape, 0.03 if rows else 0.0, dtype=np.float32)
    fields.set_epsilon_volumes(
        {c: eps for c in ("Ex", "Ey", "Ez")},
        {c: inverse for c in ("Ex", "Ey", "Ez")},
        {"Ex": {"Ey": row}, "Ey": {"Ex": row}})
    assert fields.has_offdiagonal_epsilon is bool(rows)
    return _offdiag_seed(fields, seed=seed), pml


def _uint32_differing(left, right):
    a = np.frombuffer(left.tobytes(), dtype=np.uint32)
    b = np.frombuffer(right.tobytes(), dtype=np.uint32)
    return int(np.count_nonzero(a != b))


def _offdiag_compare(sub_step, written, cycles=4):
    with_rows, pml_a = _offdiag_fixture(rows=True)
    without, pml_b = _offdiag_fixture(rows=False)
    before = {name: getattr(with_rows, name).copy() for name in written}
    differing = 0
    for _ in range(cycles):
        sub_step(with_rows, pml_a)
        sub_step(without, pml_b)
        for name in written:
            differing += _uint32_differing(getattr(with_rows, name),
                                           getattr(without, name))
    moved = sum(_uint32_differing(before[name], getattr(with_rows, name))
                for name in written)
    return differing, moved


_D_CURL_WRITTEN = {
    "step_B": ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz"),
    "step_D": ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"),
}


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_an_offdiagonal_row_does_not_enter_either_folded_complex_curl(sub_step):
    """The reference test for group (D)'s curl half.

    A folded complex run with a surviving off-diagonal row and its row-free
    twin, from identical bytes, four complete cycles. The curls difference the
    STORED E and H arrays and read no chi1inv, so the array path must write the
    same words — and the CERTIFIED K1 kernel, which binds no chi1inv pointer at
    all, cannot see the difference either. The device leg measured 0/110592.
    """
    differing, moved = _offdiag_compare(
        getattr(stepping, sub_step), _D_CURL_WRITTEN[sub_step])
    assert moved > 0, f"{sub_step} moved no words: a no-op agreeing with a no-op"
    assert differing == 0, (
        f"{sub_step} differs in {differing} words with an off-diagonal row "
        f"installed; the device leg measured "
        f"{_GROUP_D_MEASURED['D_complex_fold_offdiag_' + sub_step]}")


def test_an_offdiagonal_row_does_not_enter_folded_complex_update_H():
    differing, moved = _offdiag_compare(
        stepping.update_H, ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz"))
    assert moved > 0
    assert differing == 0, f"update_H differs in {differing} words"


def test_the_CONTROL_update_E_does_differ_with_an_offdiagonal_row():
    """25041 of 110592 words on the device; here it need only be nonzero.

    Without this leg the three identity legs above prove nothing: they would be
    equally happy if the fixture never installed a row at all.
    """
    differing, moved = _offdiag_compare(
        stepping.update_E, ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"))
    assert moved > 0
    assert differing > 0, (
        "update_E agreed with and without an off-diagonal row: the fixture is "
        "not installing one and every identity leg above is vacuous")


def test_the_new_predicates_admit_what_the_incumbents_refuse():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True)
    probe = _probe_record(patterns="base")
    for sub_step in ("step_B", "step_D"):
        assert _reasons(module.folded_complex_offdiag_pml_curl_coverage(
            fields, pml, sub_step, probe=probe)) == [], sub_step
        assert any("off-diagonal" in r for r in
                   module.folded_complex_pml_curl_coverage(
                       fields, pml, sub_step, probe=probe).reasons)
    assert _reasons(module.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "H", probe=probe)) == []
    assert any("off-diagonal" in r for r in
               module.folded_complex_constitutive_coverage(
                   fields, pml, "H", probe=probe).reasons)


def test_the_two_families_are_disjoint_with_no_row_installed():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=False)
    probe = _probe_record(patterns="base")
    reasons = _reasons(module.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B", probe=probe))
    assert any("no off-diagonal chi1inv row is installed" in r for r in reasons), reasons
    reasons = _reasons(module.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "H", probe=probe))
    assert any("no off-diagonal chi1inv row is installed" in r for r in reasons), reasons
    # And the incumbents take it.
    assert _reasons(module.folded_complex_pml_curl_coverage(
        fields, pml, "step_B", probe=probe)) == []
    assert _reasons(module.folded_complex_constitutive_coverage(
        fields, pml, "H", probe=probe)) == []


def test_side_E_is_refused_by_name_with_its_measurement():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True)
    probe = _probe_record(patterns="base")
    joined = " ".join(module.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "E", probe=probe).reasons)
    assert "side='E'" in joined
    assert "25041/110592" in joined, "the refusal must carry its measurement"
    assert "_offdiagonal_terms" in joined
    assert module.plan_folded_complex_offdiag_constitutive(
        fields, pml, "E", probe=probe) is None


def test_an_unfolded_grid_is_refused_by_both_new_predicates():
    module = importlib.import_module(MODULE_NAME)
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                courant=0.35, boundaries="periodic", xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    shape = grid.shape
    eps = np.full(shape, 2.25, dtype=np.float32)
    row = np.full(shape, 0.03, dtype=np.float32)
    fields.set_epsilon_volumes({c: eps for c in ("Ex", "Ey", "Ez")},
                               {c: (1.0 / eps).astype(np.float32)
                                for c in ("Ex", "Ey", "Ez")},
                               {"Ex": {"Ey": row}})
    pml = PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))
    probe = _probe_record(patterns="base")
    assert any("no mirror plane is active" in r for r in _reasons(
        module.folded_complex_offdiag_pml_curl_coverage(fields, pml, "step_B",
                                                        probe=probe)))
    assert any("no mirror plane is active" in r for r in _reasons(
        module.folded_complex_offdiag_constitutive_coverage(fields, pml, "H",
                                                            probe=probe)))


def test_real_storage_is_refused_toward_the_real_folded_families():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True, complex_storage=False)
    joined = " ".join(module.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B", probe=_probe_record(patterns="base")).reasons)
    assert "folded_offdiag_update_e.py" in joined


def test_beta_with_an_offdiagonal_row_is_refused_in_both_directions():
    """This IS the pairing MEEP itself refuses, and the refusal must say so."""
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True, beta=0.25)
    probe = _probe_record(patterns="base")
    for verdict in (module.folded_complex_offdiag_pml_curl_coverage(
                        fields, pml, "step_B", probe=probe),
                    module.folded_complex_offdiag_constitutive_coverage(
                        fields, pml, "H", probe=probe)):
        assert any("beta" in r for r in verdict.reasons), verdict.reasons
    joined = " ".join(module.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B", probe=probe).reasons)
    assert "fields.cpp:548-549" in joined


def test_chi2_chi3_and_a_susceptibility_and_bfast_stay_refused():
    module = importlib.import_module(MODULE_NAME)
    probe = _probe_record(patterns="base")

    fields, pml = _offdiag_fixture(rows=True)
    fields.set_nonlinear_volumes({"Ez": 0.0}, {"Ez": 0.05})
    assert any("chi2/chi3 is installed" in r for r in _reasons(
        module.folded_complex_offdiag_pml_curl_coverage(fields, pml, "step_B",
                                                        probe=probe)))

    from types import SimpleNamespace as _NS
    fields, pml = _offdiag_fixture(rows=True)
    fields.polarizations.append(_NS(
        susceptibility=_NS(kind="lorentzian"),
        driven=lambda: ("Ez",), drives=lambda name: name == "Ez"))
    assert any("complex-storage ADE" in r for r in _reasons(
        module.folded_complex_offdiag_constitutive_coverage(fields, pml, "H",
                                                            probe=probe)))


def test_a_missing_probe_artifact_is_refused():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True)
    assert not module.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B", probe={"backend": "cupy", "patterns": {}}).covered
    assert not module.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "H", probe={"backend": "cupy", "patterns": {}}).covered


def test_a_missing_volume_is_refused_by_name():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True)
    fields.fu_Bz = None
    assert any("fu_Bz is not allocated" in r for r in
               module.folded_complex_offdiag_pml_curl_coverage(
                   fields, pml, "step_B", probe=_probe_record(patterns="base")).reasons)


def test_unknown_sub_step_and_side_raise():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True)
    with pytest.raises(ValueError):
        module.folded_complex_offdiag_pml_curl_coverage(fields, pml, "update_E")
    with pytest.raises(ValueError):
        module.folded_complex_offdiag_constitutive_coverage(fields, pml, "B")


def test_mutation_dropping_the_inverted_offdiag_clause_breaks_disjointness(monkeypatch):
    module = importlib.import_module(MODULE_NAME)
    original = module._offdiag_media_reasons

    def mutated(fields, grid, targets):
        return [r for r in original(fields, grid, targets)
                if "no off-diagonal chi1inv row is installed" not in r]

    monkeypatch.setattr(module, "_offdiag_media_reasons", mutated)
    fields, pml = _offdiag_fixture(rows=False)
    probe = _probe_record(patterns="base")
    assert _reasons(module.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B", probe=probe)) == [], (
        "the mutation is DISARMED: the inverted clause was not what kept the "
        "row-free run out")
    assert _reasons(module.folded_complex_pml_curl_coverage(
        fields, pml, "step_B", probe=probe)) == [], (
        "and the incumbent admits it too, which is the overlap the clause exists "
        "to prevent")


def test_the_module_records_the_measurement_that_licensed_the_admission():
    text = (PACKAGE_DIR / "folded_complex.py").read_text(encoding="utf-8")
    for case, (differing, compared) in _GROUP_D_MEASURED.items():
        assert case in text, f"{case} is not recorded in the module"
    assert "25041" in text and "110592" in text
    assert "awaiting wiring" not in text.lower(), (
        "the arms are wired now; the deferral header would be a false statement "
        "about launch.py")
    # And the SCOPE the wiring bought has to be stated where the clause lives:
    # the family refuses an off-diagonal row at update_E alone, not across all
    # four sub-steps, and that is the sentence a reader of _media_reasons needs.
    assert "update_E`` ALONE" in text or "update_E`` alone" in text, (
        "the module must say the off-diagonal refusal now scopes to update_E")


def test_the_admission_itself_was_measured_not_only_the_kernel_body():
    """The module must record a leg through its OWN builders, predicate unpatched.

    The nine triage legs patch the INCUMBENT predicate in front of the INCUMBENT
    builder, which licenses ``FoldedComplexPmlCurlPlan`` and
    ``ComplexConstitutivePlan`` and says nothing about
    ``folded_complex_offdiag_pml_curl_coverage`` or
    ``plan_folded_complex_offdiag_pml_curl``.
    """
    text = (PACKAGE_DIR / "folded_complex.py").read_text(encoding="utf-8")
    assert "new_predicates.json" in text
    assert "unpatched" in text
    for builder in ("plan_folded_complex_offdiag_pml_curl",
                    "plan_folded_complex_offdiag_constitutive"):
        assert builder in text


def test_the_fixture_is_not_claimed_to_be_a_corpus_row_and_rowshapes_were_run():
    """The 3-D fixture is synthetic; the rows are 2-D with more folds and a phase.

    Claiming a synthetic ``[9,16,16]`` box IS "the corpus configuration" is the
    kind of overstatement that survives review because the verdict it carries is
    true. The remedy is not softer wording — it is the row-shaped measurement,
    and the module must carry both.
    """
    text = (PACKAGE_DIR / "folded_complex.py").read_text(encoding="utf-8")
    assert "D_ROWSHAPE_TWOFOLD" in text and "D_ROWSHAPE_BLOCH" in text, (
        "the row-shaped legs are not recorded")
    assert "[9,16,16]" in text or "9,16,16" in text, (
        "the module must name the synthetic fixture's own shape")
    assert "3.5" in text, "the Bloch row's k_point is not named"
    # Comment prose wraps, so the disclaimer is matched on flattened text.
    flat = " ".join(text.replace("#", " ").split())
    assert "is NOT the corpus configuration" in flat, (
        "the module must say in so many words that its fixture is not a corpus row")


def test_the_stale_reachability_analogy_is_named_as_stale():
    """The refusal at update_E is right; the reason it CITED was not.

    ``_media_reasons`` cites ``fields.cpp:548-549`` and
    ``_special_kz_beta_term``. Both are about BETA plus an off-diagonal epsilon
    in REAL storage, not about a FOLD plus one — and the corpus census lifted
    three rows carrying fold + complex + off-diagonal simultaneously. The module
    must say so, because the next reader will otherwise re-derive the wrong
    conclusion the brief carried in.
    """
    text = (PACKAGE_DIR / "folded_complex.py").read_text(encoding="utf-8")
    assert "REACHABILITY ARGUMENT DOES NOT SURVIVE" in text
    for row in ("TestArrayMetadata", "TestHoleyWvgBands", "solve-cw.py"):
        assert row in text
