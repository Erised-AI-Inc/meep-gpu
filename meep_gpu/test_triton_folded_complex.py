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
# THE HEADLINE: the parity multiply is per part, measured
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


#: LAPTOP-MEASURED (NumPy 2.x, this engineered set): words in which the previous
#: whole-complex fill (``phase * plane`` with the coefficient promoted to ``(+-1,
#: +0)``) diverges from the array path's per-part product. This is the module's
#: own headline number and the device gate's inverted needle
#: (gate_triton_folded_complex.py, full_complex_parity_fill), so the three agree by
#: construction rather than by coincidence.
PLANE_WISE_DIVERGENCE = {1: 8, -1: 8}


@pytest.mark.parametrize("phase", [1, -1])
def test_the_parity_multiply_is_per_part_not_a_full_complex_product(phase):
    """THE measurement the fill turns on, on the reference NumPy.

    ``stepping._write_mirror_ghost`` writes ``parity * plane`` through
    ``host_writes.copy_face``, which forms it with ``part_products.scale`` — each
    stored part times the rounded Python int, MEEP's per-part connection. The
    previous whole-complex product (NumPy's ``phase * plane``, the multiply by
    ``(+-1.0, +0.0)`` with its zero cross terms) is byte-different from it at BOTH
    parities on the engineered signed-zero set, which is what the gate's inverted
    headline mutation needs to be visible.
    """
    from meep_gpu import part_products, stepping

    plane = _engineered_complex_plane()
    array_path = np.ascontiguousarray(part_products.scale(phase, plane))
    shifted = np.zeros((3,) + plane.shape, dtype=np.complex64)
    shifted[stepping.MIRROR_SOURCE_INDEX] = plane
    stepping._write_mirror_ghost(shifted, 0, phase)
    assert np.array_equal(np.ascontiguousarray(shifted[0]).view(np.uint32),
                          array_path.view(np.uint32)), "the fill is not part_products.scale"
    full = np.ascontiguousarray(phase * plane)            # the previous form
    a = array_path.view(np.uint32).ravel()
    b = full.view(np.uint32).ravel()
    differing = int((a != b).sum())
    assert a.size == 128
    assert differing == PLANE_WISE_DIVERGENCE[phase], (
        f"the whole-complex fill must diverge from the per-part one in "
        f"{PLANE_WISE_DIVERGENCE[phase]}/128 engineered words at phase "
        f"{phase:+d}; measured {differing}")


def test_the_python_int_float_and_complex64_spellings_of_the_phase_agree():
    """The coefficient may be host-rounded through numpy.complex64 without
    changing the array path's bytes — which is what licenses passing WORDS."""
    plane = _engineered_complex_plane()
    for phase in (1, -1):
        reference = np.ascontiguousarray(phase * plane).view(np.uint32)
        for spelling in (float(phase), np.complex64(phase)):
            got = np.ascontiguousarray(spelling * plane).view(np.uint32)
            assert np.array_equal(reference, got), (phase, spelling)


def test_the_even_mirror_is_the_identity_per_part():
    """Per part, ``x * (+1.0)`` IS ``x`` on every engineered word: the
    ``identity_shortcut_on_even_mirror`` gate mutation (the multiply dropped) is a
    structural null on the +1 rows and is caught on the -1 rows only, where the
    copy keeps the sign the parity flips."""
    from meep_gpu import part_products

    plane = _engineered_complex_plane()
    even = np.ascontiguousarray(part_products.scale(1, plane)).view(np.uint32).ravel()
    odd = np.ascontiguousarray(part_products.scale(-1, plane)).view(np.uint32).ravel()
    words = np.ascontiguousarray(plane).view(np.uint32).ravel()
    assert np.array_equal(even, words)
    assert int((odd != words).sum()) == 128


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
    """The entries' imaginary coefficient word is bitwise ``+0``: what lets the
    per-part fill launch the real word alone. (It was also the premise the retired
    ``synthesize_zero_imag`` null rested on, asserted directly.)

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


@pytest.mark.parametrize("name,clauses", [("folded_bloch_pml_curl_step", 9),
                                          ("folded_beta_pml_curl_step", 15),
                                          ("folded_beta_bloch_pml_curl_step", 15)])
def test_the_top_plane_mask_exists_and_is_gated_on_mirror_periodic(fc, name, clauses):
    """Nine (side, target, axis) clauses on every curl: the B side masks two axes per
    target, the D side one. The beta bodies carry six more on the pass's own
    increment, which reaches targets 0 and 1 only (four on the B side, two on D)."""
    source = kernel_source(name)
    assert source.count("== MIRROR_PERIODIC") == clauses
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


@pytest.mark.parametrize("name,increments", [
    ("folded_beta_pml_curl_step",
     ("q0 = (beta_plus * b) * -1.0", "q1 = (beta_minus * a) * -1.0")),
    ("folded_beta_bloch_pml_curl_step",
     ("q0_re = beta_plus * b_im", "q0_im = (beta_plus * b_re) * -1.0",
      "q1_re = beta_minus * a_im", "q1_im = (beta_minus * a_re) * -1.0")),
])
def test_the_beta_pass_follows_the_forms_and_masks_its_own_increment(
        fc, name, increments):
    """THE fold-order needle. MEEP runs the beta pass after every component's curl
    (step_db.cpp:161-175): after the curl, its cell-zero and top-plane masks and the
    curl forms, and before the stores. A fold WIDENS the curl's ownership mask, and the
    pass's increment carries the same clauses (an exact +0 on every cell the curl
    drops), so no live increment reaches the mirror plane or the far ghost plane; it
    enters targets 0 and 1 through the ladder."""
    source = kernel_source(name)
    top_mask_at = source.index("last_x, last_y, last_z")
    forms_at = source.index("curl_form(")
    beta_at = source.index("if HAS_BETA:")
    ladder_at = source.index("fold_ladder(")
    store_at = source.index("tl.store(")
    assert top_mask_at < forms_at < beta_at < ladder_at < store_at
    pass_text = source[beta_at:store_at]
    for increment in increments:
        assert increment in pass_text, increment
    assert "tl.where(at_" in pass_text and "tl.where(last_" in pass_text
    assert pass_text.count("fold_ladder(") == len(increments)
    assert "f2 " not in pass_text and "v2" not in pass_text


def test_the_complex_beta_partners_are_the_unrotated_centre_words(fc):
    """The Bloch connection applies to SHIFTED operands only; the beta partner is the
    CENTER pair loaded before the connection section, never reassigned, and turned a
    quarter by one real word per part (no complex product, no multiply helper)."""
    import re  # noqa: PLC0415

    source = kernel_source("folded_beta_bloch_pml_curl_step")
    connect_at = source.index("_connect(")
    beta_at = source.index("if HAS_BETA:")
    assert connect_at < beta_at
    for word in ("a_re", "a_im", "b_re", "b_im"):
        bound = re.findall(rf"^\s*(?:\w+, )*{word}\b(?:, \w+)* = ", source, re.M)
        assert len(bound) == 1, (word, bound)
    pass_text = source[beta_at:source.index("tl.store(", beta_at)]
    for connected in ("b0_re", "b2_re", "a2_re", "r_re"):
        assert connected not in pass_text, connected
    assert "_mul_imag_coefficient_left" not in source


def test_the_fill_multiplies_each_part_by_the_passed_parity_word(fc):
    """The parity reaches the fill as one host-rounded float32 word per pass and
    multiplies each stored part (``part_products.scale``): six products per pass,
    no complex product helper, no ``PHASE`` constexpr and no imaginary coefficient
    word."""
    source = kernel_source("folded_mirror_ghost_fill_complex")
    assert source.count("near_c * tl.load(") == 6
    assert source.count("far_c * tl.load(") == 6
    assert "_mul_imag_coefficient_left" not in source
    assert "PHASE" not in source
    parameters = kernel_parameters("folded_mirror_ghost_fill_complex")
    assert "near_c" in parameters and "far_c" in parameters
    assert not any(name.endswith("_im") for name in parameters)


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
    to +0 — MEASURED REACHABLE at the driver level. The helpers
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
    verdict = fc.folded_complex_pml_curl_coverage(fields, pml, sub_step)
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
    verdict = fc.folded_complex_pml_curl_coverage(fields, pml, sub_step)
    assert _reasons(verdict) == []


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_a_fold_on_each_axis_is_admitted(fc, axis):
    """A folded Z must classify exactly as a folded X does."""
    fields, pml = _fold_grid(axis=axis, dimensions=3, other=1.2)
    verdict = fc.folded_complex_pml_curl_coverage(fields, pml, "step_B")
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
    assert _reasons(fc.folded_complex_pml_curl_coverage(
        fields, pml, "step_B")) == []
    assert any("equivalence product" in r for r in _reasons(
        fc.folded_complex_composition_curl_coverage(
            fields, pml, "step_B")))


def test_real_storage_is_refused_by_the_complex_curl(fc):
    fields, pml = _fold_grid(complex_storage=False)
    assert any("storage is real float32" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   fields, pml, "step_B")))


def test_a_nonzero_beta_is_refused_by_the_plain_folded_complex_curl(fc):
    fields, pml = _fold_grid(beta=BETA_EIGSRC)
    assert any("belongs to the folded beta kernels" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   fields, pml, "step_B")))


def test_a_zero_beta_is_refused_by_both_beta_curls(fc):
    """Selecting between two valid products by branch order would make the
    numerical method depend on composer order."""
    fields, pml = _fold_grid(beta=0.0)
    assert any("grid.beta is zero" in r
               for r in _reasons(fc.folded_beta_bloch_pml_curl_coverage(
                   fields, pml, "step_B")))
    real_fields, real_pml = _fold_grid(beta=0.0, complex_storage=False)
    assert any("grid.beta is zero" in r
               for r in _reasons(fc.folded_beta_pml_curl_coverage(
                   real_fields, real_pml, "step_B")))


def test_no_active_pml_is_refused(fc):
    fields, pml = _fold_grid(pml=0)
    assert any("no active PML layer" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   fields, pml, "step_B")))


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
        shim, pml, "step_B"))
    assert any("folded with k component" in r for r in reasons), reasons


def test_a_bloch_k_on_an_UNFOLDED_periodic_axis_is_admitted(fc):
    """The two binary-grating rows carry an in-plane kx on the PML'd X axis
    beside a folded Y — the phases compose across DIFFERENT axes."""
    fields, pml = _fold_grid(k_point=(0.21, 0.0, 0.0))
    assert _reasons(fc.folded_complex_pml_curl_coverage(
        fields, pml, "step_B")) == []


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
        shim, pml, "step_B"))
    assert any("invariant axis" in r for r in reasons), reasons


@pytest.mark.parametrize("flag,needle", [
    ("has_nonlinearity", "chi2/chi3"),
    ("has_offdiagonal_epsilon", "off-diagonal chi1inv"),
    ("has_magnetic_conductivity", "magnetic (B) conductivity"),
])
def test_the_media_refusals_fire_by_name(fc, flag, needle):
    fields, pml = _fold_grid()
    reasons = _reasons(fc.folded_complex_pml_curl_coverage(
        FieldsShim(fields, **{flag: True}), pml, "step_B"))
    assert any(needle in r for r in reasons), reasons


def test_a_conductivity_on_a_curl_target_is_refused(fc):
    fields, pml = _fold_grid()
    shim = FieldsShim(fields,
                      condfac_for=lambda name: object() if name == "Dx" else None)
    assert any("conductivity is installed on Dx" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   shim, pml, "step_B")))


def test_an_unreadable_conductivity_table_is_refused_outright(fc):
    """Inferring "no conductivity" from the ABSENCE of ``condfac_for`` is
    admission by attribute absence — the exact reasoning coverage exists to
    refuse."""
    fields, pml = _fold_grid()
    assert any("does not expose condfac_for" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   FieldsShim(fields, condfac_for=None), pml, "step_B")))


def test_bfast_is_refused(fc):
    fields, pml = _fold_grid()
    shim = FieldsShim(fields, grid=GridShim(fields.grid, bfast_active=True))
    assert any("BFAST is active" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   shim, pml, "step_B")))


def test_a_registered_susceptibility_is_refused_under_complex_storage(fc):
    fields, pml = _fold_grid()
    assert any("susceptibility is registered" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   FieldsShim(fields, has_polarizations=True), pml, "step_B")))


def test_recomputed_E_is_refused(fc):
    fields, pml = _fold_grid()
    assert any("recomputed from D" in r
               for r in _reasons(fc.folded_complex_pml_curl_coverage(
                   FieldsShim(fields, stores_E=False), pml, "step_B")))


def test_a_cylindrical_grid_is_refused_by_name(fc):
    """The r = 0 ghost is ``r_to_minus_r`` and the axis row belongs to the per-m
    rules; ``_boundary_kinds`` would report ``"axis"``, which this file's
    classification does not carry."""
    grid = Grid(resolution=10.0, cell_size=(1.6, 0.0, 1.6), cylindrical=True, xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((0, 2), (0, 0), (2, 2)))
    reasons = _reasons(fc.folded_complex_pml_curl_coverage(
        fields, pml, "step_B"))
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
        fields, family)) == []


def test_the_fill_is_refused_with_no_mirror_plane(fc):
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                boundaries="periodic", xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    assert any("no mirror plane is active" in r
               for r in _reasons(fc.folded_mirror_ghost_fill_complex_coverage(
                   fields, "B")))


def test_the_fill_is_refused_on_real_storage_and_says_why(fc):
    """The real fold's fill stays ``symmetry.mirror_ghost_fill``, which is EXACT
    there — this predicate must not silently claim that territory."""
    fields, _pml = _fold_grid(complex_storage=False)
    assert any("exact sign flip" in r or "EXACT" in r
               for r in _reasons(fc.folded_mirror_ghost_fill_complex_coverage(
                   fields, "B")))


def test_the_fill_checks_the_reflect_row_bounds_rather_than_assuming_them(
        fc, monkeypatch):
    """The far ghost must not image the plane it writes, or read out of the
    allocation. Disjointness is CHECKED, not assumed."""
    fields, _pml = _fold_grid()
    grid = fields.grid
    monkeypatch.setattr(stepping, "_far_reflect_rows",
                        lambda g: (None, int(grid.shape[1]) - 1, None))
    reasons = _reasons(fc.folded_mirror_ghost_fill_complex_coverage(
        fields, "B"))
    assert any("reflect row" in r and "outside" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# The two folded complex predicates admit a folded complex grid
# ---------------------------------------------------------------------------


def test_the_plain_folded_complex_curl_admits_a_folded_complex_grid(fc):
    """K1 admits the folded complex grid; no reason is left but the backend's."""
    fields, pml = _fold_grid()
    assert _reasons(fc.folded_complex_pml_curl_coverage(
        fields, pml, "step_B")) == []


def test_the_complex_constitutive_delegate_admits_a_folded_complex_grid(fc):
    """The kernel launched there is the certified ``bloch_constitutive_step``,
    whose admission is reached by RESTATING the element-wise
    contract, never by subtracting reasons from another predicate's output."""
    fields, pml = _fold_grid()
    for side in ("H", "E"):
        assert _reasons(fc.folded_complex_constitutive_coverage(
            fields, pml, side)) == []


# ---------------------------------------------------------------------------
# The real + beta family, and the coupling to symmetry.py this tranche reuses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_folded_real_beta_configuration_is_refused_only_for_the_backend(
        fc, sub_step):
    fields, pml = _fold_grid(complex_storage=False, beta=BETA_GRATING_13_2)
    assert _reasons(fc.folded_beta_pml_curl_coverage(fields, pml, sub_step)) == []


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
    assert len(words) == 2 and all(isinstance(word, float) for word in words)
    assert words[0] == -words[1] and words[0] != 0.0


# ---------------------------------------------------------------------------
# Plan builders — refusal is None, and the launch bindings are the engine's
# ---------------------------------------------------------------------------

def test_every_plan_builder_refuses_to_None_and_never_raises(fc):
    fields, pml = _fold_grid()                    # NumPy: never covered
    assert fc.plan_folded_complex_pml_curl(fields, pml, "step_B") is None
    assert fc.plan_folded_mirror_ghost_fill_complex(
        fields, "B") is None
    assert fc.plan_folded_complex_constitutive(
        fields, pml, "E") is None
    assert fc.plan_folded_beta_bloch_pml_curl(
        fields, pml, "step_B") is None
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
        (None, None, None), 0.35, chunk_bounds=_every_cell(shape))
    assert plan.sub_step == "step_B" and plan.backward == 0
    assert plan.bc == (0, fc.CODE_MIRROR_PERIODIC, 0)
    assert plan.connections == (0, 0, 0)
    assert plan.chunk_bounds.tolist() == _every_cell(shape).tolist()
    assert plan.n_elem == 30
    assert repr(plan).startswith("FoldedComplexPmlCurlPlan")

    entries = [{"axis": 1, "phase": 1, "far": True, "reflect_row": 3,
                "near_words": (1.0, 0.0), "far_words": (-1.0, 0.0),
                "shifts": (1, 0, 1)}]
    fill = fc.plan_folded_mirror_ghost_fill_complex_from_arrays(
        "B", arrays, entries)
    assert fill.family == "B" and [e["axis"] for e in fill.axes] == [1]


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
            (None, None, None), 0.35, chunk_bounds=_every_cell(shape))


def test_the_folded_curl_plan_refuses_a_connection_on_a_folded_axis(fc):
    """The connection exists only on a wrapping axis; on a folded one plane 0 is a
    stored, stepped plane the array path never connects."""
    shape = (6, 5, 1)
    arrays = {name: np.zeros(shape, dtype=np.complex64)
              for name in ("Bx", "By", "Bz", "Ex", "Ey", "Ez",
                           "fu_Bx", "fu_By", "fu_Bz")}
    flat = {f"{stem}_{axis}": np.ones(shape[index], dtype=np.float32)
            for index, axis in enumerate("xyz") for stem in ("kms", "sinv")}
    with pytest.raises(ValueError, match="only a periodic wrap"):
        fc.plan_folded_complex_pml_curl_from_arrays(
            "step_B", arrays, flat, (0, fc.CODE_MIRROR_PERIODIC, 0),
            (None, complex(0.6, 0.8), None), 0.35, chunk_bounds=_every_cell(shape))


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
        plus, minus, chunk_bounds=_every_cell(shape))
    assert plan.beta_plus == plus and plan.beta_minus == minus
    assert plan.has_beta == 1
    identity = fc.plan_folded_beta_pml_curl_from_arrays(
        "step_B", arrays, flat, (0, fc.CODE_MIRROR_METALLIC, 0), 0.35,
        plus, minus, has_beta=0, chunk_bounds=_every_cell(shape))
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
        fields, pml, "step_B")
    assert _reasons(verdict) == [], (name, verdict.reasons)
    assert _reasons(fc.folded_mirror_ghost_fill_complex_coverage(
        fields, "B")) == []


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


def _every_cell(shape):
    """Every cell of every resolved axis inside a layer chunk, both lattices."""
    from meep_gpu.chunk_map import RECORD_PRESENCE_WORD, RECORD_WORDS, record_index

    record = np.zeros(RECORD_WORDS, dtype=np.int32)
    for axis, n in enumerate(shape):
        if n > 1:
            record[RECORD_PRESENCE_WORD] |= 1 << axis
            for half in (False, True):
                record[record_index(half, axis, 1)] = n
    return record


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
              ("_connect", "_rotate_field_left", "_mul_field_left",
               "_mul_coefficient_left", "_mul_imag_coefficient_left")]
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


def test_the_headline_inversion_rewrites_every_parity_product():
    """``full_complex_parity_fill`` must replace EVERY per-part product of the fill
    — six near and six far — or it leaves a half-mutated kernel whose verdict
    means nothing."""
    gate = _gate_module()
    transform = gate.SOURCE_MUTATIONS["full_complex_parity_fill"][0]
    mutated, hits = transform(_mutation_sources()["fill"])
    assert hits == 12, hits
    body = mutated.split("def folded_mirror_ghost_fill_complex")[-1]
    assert body.count("(0.0 * tl.load(") == 12


def test_the_gate_names_every_mutation_the_spec_requires():
    """The EXACT armed set, and the count the docstring advertises. A subset
    assertion cannot see drift: the gate once said "fourteen" for sixteen
    mutations and no test could tell."""
    gate = _gate_module()
    named = set(gate.SOURCE_MUTATIONS) | set(gate.HOST_MUTATIONS)
    required = {
        "full_complex_parity_fill", "identity_shortcut_on_even_mirror",
        "fold_ghost_wraps", "drop_top_plane_mask", "mask_only_metallic",
        "pairwise_stencil", "whole_grid_recurrence", "membership_one_row_off",
        "fu_store_unmasked", "presence_ignored", "connection_dropped",
        "reflect_row_n_minus_two", "far_fill_uses_plus_phase", "fuse_fill_passes",
        "reverse_axis_order", "phase_on_folded_axis",
    }
    # Each beta body's rows: the beta pass's shared mutations and the stencil inversion.
    shared = ("b_pass_dropped", "b_partners_swapped", "b_sign_flipped",
              "b_scaled_by_dtdx", "b_pass_inside_the_curl", "b_increment_unmasked",
              "b_ladder_without_dsigu", "b_ladder_axes_swapped",
              "b_auxiliary_not_advanced", "b_top_plane_mask_dropped",
              "a1_pairwise_stencil")
    required |= {f"beta_real:{name}" for name in shared}
    required |= {f"beta_complex:{name}" for name in shared + ("b_quarter_turn_reversed",)}
    assert named == required, sorted(named ^ required)
    assert (len(gate.SOURCE_MUTATIONS), len(gate.HOST_MUTATIONS)) == (34, 5)
    spelled = ("thirty-four armed SOURCE mutations (two on the fill, nine on\n"
               "  K1's forms and connection, eleven on K3a and twelve on K3b: the beta "
               "pass's shared\n  mutations and the stencil inversion, "
               "``fold_pass_mutations``) and five HOST ones\n  (thirty-nine;")
    assert spelled in (gate.__doc__ or ""), "the gate docstring's count drifted"
    # the per-part fill commutes: two RECORDED host nulls, no source null
    assert set(gate.HOST_MUTATION_NULLS) == {"fuse_fill_passes", "reverse_axis_order"}
    assert gate.SOURCE_MUTATION_NULLS == {}
    # the imaginary coefficient word is gone, and so are the two legs about it
    assert "synthesize_zero_imag" not in gate.SOURCE_MUTATIONS
    assert "imaginary_words_are_read_not_synthesised" not in gate.HOST_MUTATIONS


def test_reverse_axis_order_is_a_recorded_null_and_the_mixed_phase_grid_stays():
    """Per part, two planes' parity words commute bitwise, so reversing the axis
    order is a STRUCTURAL null, recorded with its reason — and the MIXED-PHASE
    two-folded-axis row that made it a measurement under the whole-complex fill
    stays in the grid list, so the null is judged on the row that used to diverge."""
    gate = _gate_module()
    assert "commute bitwise" in gate.HOST_MUTATION_NULLS["reverse_axis_order"]
    mixed = [spec for spec in gate.FILL_GRIDS
             if len(spec["axis"]) > 1
             and isinstance(spec["phase"], (tuple, list))
             and len(set(spec["phase"])) > 1]
    assert mixed, "no mixed-phase two-folded-axis fill grid"
    quick = [gate.FILL_GRIDS[i] for i in gate.QUICK_FILL_GRID_INDICES]
    assert any(spec in mixed for spec in quick)


def test_reversing_the_axis_order_moves_no_word_per_part():
    """THE PREMISE of the ``reverse_axis_order`` null, executed on NumPy through
    the gate's own grids, state and reference fills: on BOTH the same-phase and the
    MIXED-phase two-folded-axis grids the reversed order stores the same words
    (under the whole-complex fill the mixed one diverged in 5 words of Bz / Dz)."""
    gate = _gate_module()
    fold = importlib.import_module(MODULE_NAME)
    grids = [s for s in gate.FILL_GRIDS if s["axis"] == "XY"]
    assert len(grids) == 2

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
            gate.reference_ghost_fill_near(np, arrays, family, order)
            gate.reference_ghost_fill_far(np, arrays, family, order)
            out[label] = arrays
        return sum(int((np.ascontiguousarray(out["forward"][n]).view(np.uint32).ravel()
                        != np.ascontiguousarray(out["reversed"][n]).view(np.uint32)
                        .ravel()).sum()) for n in names)

    for spec in grids:
        for family in ("B", "D"):
            assert diverging(spec, family) == 0, (spec["phase"], family)


def test_a_host_mutation_that_is_armed_but_cannot_differ_is_DISARMED():
    """``armed_rows > 0`` was the only gate, and it admits a provable no-op.

    Pinned: armed-with-no-discriminating-row DISARMS, discriminating-rows-with-
    zero-divergence is NEEDLE-MISSED even for a leg whose predicted verdict is
    "record the measurement", a RECORDED host null is judged as a null (zero
    divergence is its verdict, any divergence invalidates it), and the folded-axis
    connection is a refusal the plan must make.
    """
    gate = _gate_module()

    def verdict(armed, discriminating, diverged, predicted=None,
                name="a_measured_leg", **extra):
        summary = {"ran": 96, "identical": 96 - diverged, **extra}
        if armed is not None:
            summary["armed_rows"] = armed
            summary["discriminating_rows"] = discriminating
        return gate.host_mutation_verdict(name, predicted, summary, diverged)

    # nothing varied at all
    assert verdict(0, 0, 0)["verdict"] == "DISARMED"
    # armed on 16 rows, every one of them a provable no-op
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
    assert record["discriminating_rows"] == 8

    # a recorded null: judged, never skipped, and self-invalidating
    record = verdict(16, 0, 0, name="reverse_axis_order")
    assert record["verdict"] == "PREDICTED-NULL"
    assert "commute bitwise" in record["null_reason"]
    record = verdict(16, 8, 3, name="fuse_fill_passes")
    assert record["verdict"] == "NULL-INVALIDATED"
    assert verdict(0, 0, 0, name="fuse_fill_passes")["verdict"] == "DISARMED"

    # the folded-axis connection: refused by the plan on every case, or DISARMED
    assert verdict(None, None, 0, name="phase_on_folded_axis",
                   refused=96)["verdict"] == "REFUSED-BY-PLAN"
    assert verdict(None, None, 0, name="phase_on_folded_axis",
                   refused=3)["verdict"] == "DISARMED"
    # ...and a PREDICTED-caught value mutation still fails on zero divergence
    assert verdict(22, 22, 0, predicted=True,
                   name="far_fill_uses_plus_phase")["verdict"] == "NEEDLE-MISSED"
    assert verdict(22, 22, 9, predicted=True,
                   name="far_fill_uses_plus_phase")["verdict"] == "CAUGHT"


def test_the_retired_imaginary_word_legs_have_no_subject_left():
    """``synthesize_zero_imag`` and its host twin were about the fill's imaginary
    coefficient word; the per-part fill takes one real word per pass, so both are
    RETIRED outright (there is nothing to synthesise or to read). Pinned as the
    structural fact itself: the kernel's signature carries no imaginary word, the
    plan launches the real one, and the gate no longer names either leg."""
    gate = _gate_module()
    fold = importlib.import_module(MODULE_NAME)
    parameters = kernel_parameters("folded_mirror_ghost_fill_complex")
    assert [name for name in parameters if name in ("near_c", "far_c")] == [
        "near_c", "far_c"]
    assert not any("_im" in name for name in parameters)
    source = (PACKAGE_DIR / "folded_complex.py").read_text(encoding="utf-8")
    assert 'entry["near_words"][0]' in source and 'entry["far_words"][0]' in source
    assert "synthesize_zero_imag" not in gate.SOURCE_MUTATIONS
    assert "imaginary_words_are_read_not_synthesised" not in gate.HOST_MUTATIONS
    with pytest.raises(ValueError, match="unknown entry mutation"):
        gate.apply_entry_mutation([], "nonzero_imaginary_words", None)


def test_a_predicted_null_is_judged_not_skipped_and_self_invalidates(monkeypatch):
    """The SOURCE-mutation verdict's clause order, executed on the laptop with a
    declared null installed for the duration (the table carries none today).

    All three DISARMED paths run BEFORE the null clause, so a declared null whose
    mutant never matched, never launched, or compiled to a shipped specialization
    is DISARMED; a null that DIVERGES is NULL-INVALIDATED and FAILS.
    """
    gate = _gate_module()
    null = "a_declared_null"
    live = "full_complex_parity_fill"
    monkeypatch.setitem(gate.SOURCE_MUTATION_NULLS, null, {
        "reason": "a stated structural reason", "twin": live,
        "twin_claim": "carried by the live twin", "measured": {"launches": 192},
        "premise_test": "meep_gpu/test_triton_folded_complex.py::"
                        "test_the_even_mirror_is_the_identity_per_part"})

    assert gate.source_mutation_verdict(null, 0, None, None, None)["verdict"] \
        == "DISARMED"
    assert gate.source_mutation_verdict(null, 6, 0, True, 0)["verdict"] \
        == "DISARMED"
    stale = gate.source_mutation_verdict(null, 6, 192, False, 0)
    assert stale["verdict"] == "DISARMED"
    assert "stale binary" in stale["failure"]

    recorded = gate.source_mutation_verdict(null, 6, 192, True, 0)
    assert recorded["verdict"] == "PREDICTED-NULL"
    assert recorded["live_claim_carried_by"] == live

    broken = gate.source_mutation_verdict(null, 6, 192, True, 7)
    assert broken["verdict"] == "NULL-INVALIDATED"
    assert "is now FALSE" in broken["failure"]

    assert gate.source_mutation_verdict(live, 12, 192, True, 0)["verdict"] \
        == "NEEDLE-MISSED"
    assert gate.source_mutation_verdict(live, 12, 192, True, 64)["verdict"] \
        == "CAUGHT"


def test_the_summary_reports_every_mutation_with_an_honest_split():
    """The artifact must show every mutation asked and how each answered.

    Pinned: a PREDICTED-NULL (the two recorded host nulls) does not fail the gate
    but carries its reason into the summary; a NULL-INVALIDATED does fail; a
    declared host null the mutation leg never ran is a failure; and the folded-axis
    connection's REFUSED-BY-PLAN is a passing verdict.
    """
    gate = _gate_module()

    def mutations(null_verdict="PREDICTED-NULL", drop_null=False):
        records = {}
        for name in list(gate.SOURCE_MUTATIONS) + list(gate.HOST_MUTATIONS):
            records[name] = {"verdict": "CAUGHT"}
        records["phase_on_folded_axis"] = {"verdict": "REFUSED-BY-PLAN"}
        for name, reason in gate.HOST_MUTATION_NULLS.items():
            records[name] = {"verdict": null_verdict, "failure": "...",
                             "null_reason": reason}
        if drop_null:
            records.pop("fuse_fill_passes")
        return {"mutations": records}

    built = gate.build_summary(mutations())
    counts = built["mutation_counts"]
    assert counts["asked"] == counts["ran"] == 39
    assert counts["PREDICTED-NULL"] == 2
    assert counts["REFUSED-BY-PLAN"] == 1
    assert counts["CAUGHT"] == 36
    null = built["mutation_nulls"]["reverse_axis_order"]
    assert "commute bitwise" in null["reason"]
    assert not [f for f in built["failures"] if "reverse_axis_order" in f]

    broken = gate.build_summary(mutations(null_verdict="NULL-INVALIDATED"))
    assert [f for f in broken["failures"]
            if "fuse_fill_passes" in f and "NULL-INVALIDATED" in f]

    missing = gate.build_summary(mutations(drop_null=True))
    assert [f for f in missing["failures"]
            if "fuse_fill_passes" in f and "judged, not skipped" in f]


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
    assert "in 8/128 engineered words at BOTH parities" in text
    gate_text = GATE_PATH.read_text(encoding="utf-8")
    assert "the 8/128 words at both parities" in gate_text


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
                            dt=gate.BETA_DT, complex_storage=complex_storage,
                            record=gate.two_run_record(shape))


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
    built = gate.build_summary({"synthetic": {"summary": summary}})
    assert not built["passed"]
    assert any("fusion-OFF" in f for f in built["failures"]), built["failures"]
    # ...and an EMPTY asserted set is a failure too, not a vacuous pass.
    empty = gate._summarize([{"guard": True,
                              "verdict": {"bit_identical": True}}])
    built = gate.build_summary({"synthetic": {"summary": empty}})
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
    passing = gate.build_summary({"synthetic": {"summary": clean}})
    assert passing["passed"], passing["failures"]
    original = gate.shipped_fusion_default
    gate.shipped_fusion_default = lambda: True
    try:
        flipped = gate.build_summary({"synthetic": {"summary": clean}})
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

    def synthetic(results, out_path, quick=False):
        ran.append("synthetic")
        results["synthetic"] = {"summary": gate._summarize(
            [{"guard": False, "verdict": {"bit_identical": True}}])}
        gate.save(results, out_path)
        return results["synthetic"]

    # The configuration this models has no device: every module that reads
    # ``cp is not None`` would otherwise take a device path (the -ftz cache-policy
    # refusal and ``probe.device_info``), real behaviours that belong in a device
    # run and not in this test's path. The gate's own ``cp`` is a sentinel only so
    # that its device legs are scheduled; the strip and the policy stamp, the two
    # things ``main`` does with it, are stubbed, and the leg it runs is a recorder.
    base = importlib.import_module("gate_triton_complex")
    probe = importlib.import_module("probe_fused_kernel_bit_identity")
    for module in (base, probe):
        monkeypatch.setattr(module, "cp", None)
    monkeypatch.setattr(gate, "cp", object())
    monkeypatch.setattr(gate, "_TRITON_AVAILABLE", True)
    monkeypatch.setattr(base, "install_ftz_strip", lambda *_a, **_k: None)
    monkeypatch.setattr(base, "policy_stamp", lambda _backend: {"stubbed": True})
    monkeypatch.setattr(gate, "run_reference_validation", exploding)
    monkeypatch.setattr(gate, "run_synthetic", synthetic)
    code = gate.main(["--out", str(out), "--legs", "reference,synthetic"])

    import json as _json
    record = _json.loads(out.read_text())
    # The leg AFTER the raising one still ran and still wrote its result.
    assert ran == ["synthetic"]
    assert record["synthetic"]["summary"]["fusion_off_asserted_identical"] == 1
    # The raise is recorded with its traceback, not swallowed...
    assert record["leg_errors"]["reference"]["type"] == "RuntimeError"
    assert record["leg_errors"]["reference"]["message"] == "armed leg failure"
    assert "RuntimeError" in record["leg_errors"]["reference"]["traceback"]
    # ...and it is a NAMED failure, not a skip.
    assert not record["summary"]["passed"]
    assert any("leg reference RAISED" in f
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
    requested = {"legs": ["synthetic", "fill"],
                 "synthetic": {"summary": clean}}
    built = gate.build_summary(requested)
    assert not built["passed"]
    assert any("leg fill was requested" in f for f in built["failures"]), \
        built["failures"]
    # A cleanly skipped device leg is NOT this failure.
    skipped = dict(requested, device_legs_skipped="no cupy on this host")
    assert gate.build_summary(skipped)["passed"]


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

    # (3) the signed-zero and subnormal classes are untouched — a device run proved
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
                 guard, beta, storage, kernel=None):
        asked.append((tuple(shape), config["name"], phase_set["phases"]))
        return {"config": config["name"],
                "verdict": {"bit_identical": True, "differing_floats": 0}}

    monkeypatch.setattr(gate, "one_curl_case", recorder)
    result = gate.run_mutation_curl(object(), "curl", False)

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
    periodic = Grid(resolution=10.0, cell_size=(1.6, 2.0, 0.0), dimensions=2,
                    courant=0.35, boundaries="periodic",
                    symmetry=(Mirror("Y", 1),), xp=np)
    entries = [dict(e) for e in
               gate.folded_complex.ghost_fill_axis_entries(periodic, "B")]
    _e, armed, discriminating = gate.apply_entry_mutation(
        entries, "far_fill_uses_plus_phase", periodic)
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
                           "cases": predicted_null}}})
    assert built["passed"], built["failures"]

    blind = [dict(case, census=0) for case in predicted_null]
    assert len(gate.blind_fill_cells(blind)) == 1
    built = gate.build_summary(
        {"fill": {"gate": {"ran": 4, "identical": 4, "cases": blind}}})
    assert not built["passed"]
    assert any("blind in every state family" in f
               for f in built["failures"]), built["failures"]


def test_the_recorded_reason_for_a_zero_census_is_the_measured_one():
    """A predicted null is only a predicted null if its REASON is true.

    MEASURED over the whole 12 x 2 x 4 matrix on this laptop (2026-10-08, the
    per-part array path): ``live_post_step`` reads 0 on every row but one,
    periodic ones included, because four driven steps from a random seed leave
    generic nonzero floats — the realistic-state arm, not a signed-zero arm.
    ``zero_init_absorber`` reads 0 on the folded METALLIC phase = +1 rows only
    (near coefficient +1, no far fill). The two engineered families never read 0
    and carry the metallic +1 cells.
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
        # the metallic +1 cell rests on the two engineered families (the per-part
        # products keep the planted zeros' signs through the step)
        assert census(metallic, family, "zero_init_absorber") == 0
        assert census(metallic, family, "engineered_rows_post_step") > 0
        assert census(metallic, family, "engineered_rows") > 0
        # ...and the periodic row reaches the class from the quiet absorber too
        assert census(periodic, family, "zero_init_absorber") > 0
    text = GATE_PATH.read_text(encoding="utf-8")
    assert "reads 0 on every row but one" in text
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
               "failure": "the leg could launch nothing"}
    built = gate.build_summary({"zero_init": verdict})
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
                fields, pml, "step_B"))
        else:
            reasons = _reasons(fc.folded_beta_pml_curl_coverage(
                fields, pml, "step_B"))
        assert any("no mirror plane is active" in r for r in reasons), reasons
    # ...and a FOLDED beta grid is still admitted, in both storages.
    fields, pml = _fold_grid(beta=BETA_EIGSRC)
    assert _reasons(fc.folded_beta_bloch_pml_curl_coverage(
        fields, pml, "step_B")) == []
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
        complex_shim, complex_pml, sub_step))
    assert any(f"conductivity is installed on {target}" in r for r in twin), twin


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
    for sub_step in ("step_B", "step_D"):
        assert _reasons(module.folded_complex_offdiag_pml_curl_coverage(
            fields, pml, sub_step)) == [], sub_step
        assert any("off-diagonal" in r for r in
                   module.folded_complex_pml_curl_coverage(
                       fields, pml, sub_step).reasons)
    assert _reasons(module.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "H")) == []
    assert any("off-diagonal" in r for r in
               module.folded_complex_constitutive_coverage(
                   fields, pml, "H").reasons)


def test_the_two_families_are_disjoint_with_no_row_installed():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=False)
    reasons = _reasons(module.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B"))
    assert any("no off-diagonal chi1inv row is installed" in r for r in reasons), reasons
    reasons = _reasons(module.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "H"))
    assert any("no off-diagonal chi1inv row is installed" in r for r in reasons), reasons
    # And the incumbents take it.
    assert _reasons(module.folded_complex_pml_curl_coverage(
        fields, pml, "step_B")) == []
    assert _reasons(module.folded_complex_constitutive_coverage(
        fields, pml, "H")) == []


def test_side_E_is_refused_by_name_with_its_measurement():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True)
    joined = " ".join(module.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "E").reasons)
    assert "side='E'" in joined
    assert "25041/110592" in joined, "the refusal must carry its measurement"
    assert "_offdiagonal_terms" in joined
    assert module.plan_folded_complex_offdiag_constitutive(
        fields, pml, "E") is None


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
    assert any("no mirror plane is active" in r for r in _reasons(
        module.folded_complex_offdiag_pml_curl_coverage(fields, pml, "step_B")))
    assert any("no mirror plane is active" in r for r in _reasons(
        module.folded_complex_offdiag_constitutive_coverage(fields, pml, "H")))


def test_real_storage_is_refused_toward_the_real_folded_families():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True, complex_storage=False)
    joined = " ".join(module.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B").reasons)
    assert "folded_offdiag_update_e.py" in joined


def test_beta_with_an_offdiagonal_row_is_refused_in_both_directions():
    """This IS the pairing MEEP itself refuses, and the refusal must say so."""
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True, beta=0.25)
    for verdict in (module.folded_complex_offdiag_pml_curl_coverage(
                        fields, pml, "step_B"),
                    module.folded_complex_offdiag_constitutive_coverage(
                        fields, pml, "H")):
        assert any("beta" in r for r in verdict.reasons), verdict.reasons
    joined = " ".join(module.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B").reasons)
    assert "fields.cpp:548-549" in joined


def test_chi2_chi3_and_a_susceptibility_and_bfast_stay_refused():
    module = importlib.import_module(MODULE_NAME)

    fields, pml = _offdiag_fixture(rows=True)
    fields.set_nonlinear_volumes({"Ez": 0.0}, {"Ez": 0.05})
    assert any("chi2/chi3 is installed" in r for r in _reasons(
        module.folded_complex_offdiag_pml_curl_coverage(fields, pml, "step_B")))

    from types import SimpleNamespace as _NS
    fields, pml = _offdiag_fixture(rows=True)
    fields.polarizations.append(_NS(
        susceptibility=_NS(kind="lorentzian"),
        driven=lambda: ("Ez",), drives=lambda name: name == "Ez"))
    assert any("complex-storage ADE" in r for r in _reasons(
        module.folded_complex_offdiag_constitutive_coverage(fields, pml, "H")))


def test_a_missing_probe_artifact_is_refused():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True)
    assert not module.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B").covered
    assert not module.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "H").covered


def test_a_missing_volume_is_refused_by_name():
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _offdiag_fixture(rows=True)
    fields.fu_Bz = None
    assert any("fu_Bz is not allocated" in r for r in
               module.folded_complex_offdiag_pml_curl_coverage(
                   fields, pml, "step_B").reasons)


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
    assert _reasons(module.folded_complex_offdiag_pml_curl_coverage(
        fields, pml, "step_B")) == [], (
        "the mutation is DISARMED: the inverted clause was not what kept the "
        "row-free run out")
    assert _reasons(module.folded_complex_pml_curl_coverage(
        fields, pml, "step_B")) == [], (
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
