"""Tests for the folded-grid (mirror symmetry) Triton kernels.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware
— bit-identity against the array path — lives in
``parity/meep_gpu/gate_triton_symmetry.py``, because a byte comparison is only
meaningful on the device that runs it.

What is pinned here is everything that decides whether the folded kernel is ever
ALLOWED to run, and the four facts it hard-codes as compile-time constants:

* the ``MIRROR_METALLIC`` / ``MIRROR_PERIODIC`` classification, which is the
  single point of failure in the file — backwards on one axis is a plane of wrong
  values, not a crash;
* the STORAGE RULE that classification rests on (``stored == owned + 1`` on a
  folded periodic axis at BOTH count parities, ``stored == owned`` on a folded
  metallic one), so a future grid change breaks a test rather than a kernel;
* the reflect row, which is ``stored - 2`` at an even full count and
  ``stored - 3`` at an odd one — the only place the count parity still shows;
* the parity collapse ``mirror_parity(c, a, phase) == phase * (1 - 2*iyee)``,
  which is what lets one signed constexpr per axis stand in for the whole
  parity table.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import pathlib
import sys

import pytest

from meep_gpu import stepping
from meep_gpu.fields import FIELD_COMPONENTS, IYEE_SHIFTS, Fields, mirror_parity
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import symmetry

PACKAGE_DIR = pathlib.Path(symmetry.__file__).parent


def build(cell_size=(0.8, 0.8, 0.8), thickness=2, **grid_kwargs):
    """A real Grid/Fields/PML triple on NumPy, with the PML storage allocated."""
    grid = Grid(resolution=10.0, cell_size=cell_size, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=thickness)


def folded(boundaries="periodic", extent=2.0, axis="Y", **kwargs):
    """A folded grid on one axis, over the named outer declaration."""
    size = [0.8, 0.8, 0.8]
    size["XYZ".index(axis)] = extent
    return build(cell_size=tuple(size), boundaries=boundaries,
                 symmetry=(axis,), **kwargs)


def reasons_without_backend(verdict):
    return [r for r in verdict.reasons if "cupy" not in r]


# ---------------------------------------------------------------------------
# The optional dependency must stay optional
# ---------------------------------------------------------------------------

def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    """A missing optional dependency must not break the engine — or this module.

    The kernels live behind an import guard so the predicate, the classification
    and the plan builders' refusal path all work on a machine that has never heard
    of Triton. Only BUILDING a plan for a covered configuration may fail, and a
    NumPy configuration is never covered, so it refuses first.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "meep_gpu.triton_kernels.symmetry", raising=False)
    monkeypatch.delitem(sys.modules, "meep_gpu.triton_kernels.kernels", raising=False)

    module = importlib.import_module("meep_gpu.triton_kernels.symmetry")
    assert module.triton is None
    fields, pml = folded()
    assert module.explain_folded(fields, pml).reasons  # answers, rather than raising
    assert module.plan_folded_pml_curl(fields, pml, "step_B") is None
    assert module.plan_mirror_ghost_fill(fields, "B") is None


def code_of(path: pathlib.Path) -> str:
    """A module's source with comments and docstrings removed."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_this_module_touches_neither_dispatch_nor_the_other_track():
    """File ownership, enforced rather than agreed. Dispatch stays deferred."""
    code = code_of(PACKAGE_DIR / "symmetry.py")
    assert "fastpath" not in code
    assert "cuda_kernels" not in code
    assert "options=" not in code


def test_the_contraction_guard_is_the_packages_one_spelling():
    """Both launch sites here pass the shared constant, never a literal.

    A launch site that omits ``enable_fp_fusion`` compiles a kernel that is
    bit-wrong but numerically plausible — one that passes a carelessly written
    gate outright at dtdx = 0.5.
    """
    spelling = "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)"
    code = code_of(PACKAGE_DIR / "symmetry.py")
    assert code.count("enable_fp_fusion=") == 2
    assert code.count(spelling) == 2


# ---------------------------------------------------------------------------
# The constants the kernel hard-codes, against the engine's own
# ---------------------------------------------------------------------------

def test_the_boundary_codes_agree_with_the_shipped_kernels():
    """``PERIODIC``/``METALLIC`` must be the same integers in both kernel modules.

    Read off ``kernels.py``'s SOURCE rather than imported, so this holds on a
    machine with no Triton — where importing that module is impossible. A plan
    built here and a plan built there index the same table; a disagreement is a
    silently wrong ghost rule on every axis.
    """
    source = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    assert f"\nPERIODIC = tl.constexpr({symmetry.CODE_PERIODIC})\n" in source
    assert f"\nMETALLIC = tl.constexpr({symmetry.CODE_METALLIC})\n" in source
    assert len({symmetry.CODE_PERIODIC, symmetry.CODE_METALLIC,
                symmetry.CODE_MIRROR_METALLIC, symmetry.CODE_MIRROR_PERIODIC}) == 4


def test_the_mirror_source_index_is_steppings_own():
    assert symmetry.MIRROR_SOURCE_INDEX == stepping.MIRROR_SOURCE_INDEX


def test_the_target_yee_shifts_are_the_engines_own():
    """A swapped triple here is a mask on the wrong plane, not a crash."""
    assert symmetry.TARGET_IYEE == {name: IYEE_SHIFTS[name]
                                    for name in symmetry.TARGET_IYEE}


def test_the_top_plane_mask_covers_exactly_the_shift_one_planes():
    """The new mask's (target, axis) set is the COMPLEMENT of the cell-0 mask's.

    ``_mask_non_owned_cells`` drops cell 0 where the target's Yee shift is 0 and
    the LAST cell where it is 1 (on a folded periodic axis). The kernel writes
    both out as compile-time constants, so the two sets are checked against the
    Yee table here rather than read out of the kernel's comments.
    """
    b_top = tuple(tuple(a for a in range(3) if IYEE_SHIFTS[c][a] == 1)
                  for c in ("Bx", "By", "Bz"))
    d_top = tuple(tuple(a for a in range(3) if IYEE_SHIFTS[c][a] == 1)
                  for c in ("Dx", "Dy", "Dz"))
    # The B family masks the two axes that are NOT its own; the D family its own.
    assert b_top == ((1, 2), (0, 2), (0, 1))
    assert d_top == ((0,), (1,), (2,))
    # And they are disjoint from the cell-0 sets, which is what "complement" means.
    for family, top in (("BBB", b_top), ("DDD", d_top)):
        names = {"B": ("Bx", "By", "Bz"), "D": ("Dx", "Dy", "Dz")}[family[0]]
        zero = tuple(tuple(a for a in range(3) if IYEE_SHIFTS[c][a] == 0)
                     for c in names)
        for one_set, zero_set in zip(top, zero):
            assert not set(one_set) & set(zero_set)


def test_the_parity_collapses_to_plus_or_minus_the_plane_phase():
    """F2, exhaustively: ``mirror_parity(c, a, ph) == ph * (1 - 2*iyee[c][a])``.

    This is what lets ONE signed constexpr per folded axis stand in for the whole
    parity table: the near fill only ever touches shift-0 components (so +phase)
    and the far fill only shift-1 ones (so -phase).
    """
    for component in FIELD_COMPONENTS:
        for axis in range(3):
            for phase in (1, -1):
                assert mirror_parity(component, axis, phase) == (
                    phase * (1 - 2 * IYEE_SHIFTS[component][axis]))


# ---------------------------------------------------------------------------
# The storage rule the classification rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("extent", (2.0, 2.1))
def test_a_folded_periodic_axis_stores_one_cell_past_owned_at_both_parities(extent):
    """The whole basis of the top-plane mask, pinned so a grid change breaks a test.

    ``Grid.stored_cells`` carries MEEP's ``big_corner`` plane AND the shift-1 slot
    half a cell past it on a folded PERIODIC axis; that extra slot is the far
    ghost the fill pass writes and the curl must not step.
    """
    grid = Grid(resolution=10.0, cell_size=(0.8, extent, 0.8),
                boundaries="periodic", symmetry=("Y",))
    assert grid.stored_cells(1) == grid.owned_cells(1) + 1
    assert stepping._stored_past_owned(grid, 1) is True
    assert grid.shape_full[1] % 2 == (0 if extent == 2.0 else 1)


@pytest.mark.parametrize("extent", (2.0, 2.1))
def test_a_folded_metallic_axis_stores_exactly_its_owned_cells(extent):
    grid = Grid(resolution=10.0, cell_size=(0.8, extent, 0.8),
                boundaries="metallic", symmetry=("Y",))
    assert grid.stored_cells(1) == grid.owned_cells(1)
    assert stepping._stored_past_owned(grid, 1) is False


def test_the_reflect_row_is_two_below_the_top_at_even_and_three_at_odd():
    """THE ONLY PLACE THE COUNT PARITY STILL SHOWS. Baking ``n - 2`` is a cell wrong.

    ``_far_reflect_rows`` is ``n_full - stored + 2``: the ghost images about the
    SECOND MIRROR, which is ``big_corner`` itself at an even full count and half a
    cell below it at an odd one.
    """
    for extent, offset in ((2.0, 2), (2.1, 3)):
        grid = Grid(resolution=10.0, cell_size=(0.8, extent, 0.8),
                    boundaries="periodic", symmetry=("Y",))
        row = stepping._far_reflect_rows(grid)[1]
        assert row == grid.stored_cells(1) - offset
        assert row == grid.shape_full[1] - grid.stored_cells(1) + 2


# ---------------------------------------------------------------------------
# The classification
# ---------------------------------------------------------------------------

def test_a_folded_periodic_axis_classifies_as_mirror_periodic():
    fields, pml = folded("periodic")
    codes, reasons = symmetry.folded_axis_kinds(fields.grid, pml)
    assert reasons == ()
    assert codes == (symmetry.CODE_PERIODIC, symmetry.CODE_MIRROR_PERIODIC,
                     symmetry.CODE_PERIODIC)


def test_a_folded_metallic_axis_classifies_as_mirror_metallic():
    fields, pml = folded("metallic")
    codes, reasons = symmetry.folded_axis_kinds(fields.grid, pml)
    assert reasons == ()
    assert codes == (symmetry.CODE_METALLIC, symmetry.CODE_MIRROR_METALLIC,
                     symmetry.CODE_METALLIC)


def test_two_planes_and_a_folded_x_classify_per_axis():
    """A per-axis bug cannot hide behind a uniform declaration, so neither may a test."""
    fields, pml = build(cell_size=(2.0, 2.0, 0.8), boundaries="periodic",
                        symmetry=("X", "Y"))
    codes, reasons = symmetry.folded_axis_kinds(fields.grid, pml)
    assert reasons == ()
    assert codes == (symmetry.CODE_MIRROR_PERIODIC, symmetry.CODE_MIRROR_PERIODIC,
                     symmetry.CODE_PERIODIC)


def test_an_odd_mirror_plane_classifies_the_same_way():
    """The curl is phase-independent by construction; the fill is not."""
    fields, pml = folded("periodic", axis="Y")
    odd_grid = Grid(resolution=10.0, cell_size=(0.8, 2.0, 0.8),
                    boundaries="periodic", symmetry=(Mirror("Y", -1),))
    assert odd_grid.mirror_phase(1) == -1
    codes, reasons = symmetry.folded_axis_kinds(odd_grid, None)
    assert reasons == ()
    assert codes == symmetry.folded_axis_kinds(fields.grid, None)[0]


def test_the_classification_refuses_a_grid_that_cannot_answer_owned_cells(monkeypatch):
    """MUTATION: ``_stored_past_owned`` answers False for a grid it cannot read.

    A folded PERIODIC axis would then compile as ``MIRROR_METALLIC``, which drops
    the top-plane mask — the exact defect that measured as a ``Bx``/``fu_Bx``
    divergence. It must be a refusal, never a default.
    """
    fields, _ = folded("periodic")
    monkeypatch.delattr(type(fields.grid), "owned_cells", raising=True)
    codes, reasons = symmetry.folded_axis_kinds(fields.grid, None)
    assert codes is None
    assert any("owned_cells" in r for r in reasons), reasons


def test_the_classification_refuses_when_its_two_routes_disagree(monkeypatch):
    """MUTATION: force ``_stored_past_owned`` to lie, and the cross-check must fire.

    ``Grid.stored_cells`` adds its extra slot exactly when the axis is mirrored
    and not metallic, so ``_stored_past_owned`` and ``is_metallic`` are two routes
    to one fact. Getting the split backwards on one axis is a plane of wrong
    values; a disagreement between the routes must therefore refuse rather than
    pick one.
    """
    fields, _ = folded("periodic")
    monkeypatch.setattr(symmetry, "_stored_past_owned_reader",
                        lambda: (lambda grid, axis: False))
    codes, reasons = symmetry.folded_axis_kinds(fields.grid, None)
    assert codes is None
    assert any("disagree" in r for r in reasons), reasons


def test_a_folded_axis_with_too_few_stored_cells_is_refused(monkeypatch):
    """The near ghost images stored cell 2; an axis of three cells or fewer has none."""
    fields, _ = folded("periodic")
    monkeypatch.setattr(type(fields.grid), "stored_cells",
                        lambda self, axis: 2 if axis == 1 else 8)
    codes, reasons = symmetry.folded_axis_kinds(fields.grid, None)
    assert codes is None
    assert any("stored cells" in r for r in reasons), reasons


def test_a_folded_axis_with_no_readable_phase_is_refused(monkeypatch):
    fields, _ = folded("periodic")
    monkeypatch.setattr(type(fields.grid), "mirror_phase",
                        lambda self, axis: None)
    codes, reasons = symmetry.folded_axis_kinds(fields.grid, None)
    assert codes is None
    assert any("mirror phase" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# Coverage: the positive verdict, then the refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("boundaries", ("periodic", "metallic"))
def test_the_folded_target_configuration_is_refused_only_for_the_backend(boundaries):
    """The positive half: on the configuration this kernel is FOR, only CuPy is missing.

    A predicate no configuration satisfies is as useless as one everything
    satisfies, and this is the clause that stops the file from being the latter.
    """
    fields, pml = folded(boundaries)
    verdict = symmetry.folded_pml_curl_coverage(fields, pml)
    assert verdict.covered is False
    assert len(verdict.reasons) == 1, verdict.reasons
    assert "cupy" in verdict.reasons[0]


def test_an_unfolded_run_is_admitted_too_and_reduces_to_the_shipped_kernel():
    fields, pml = build()
    assert reasons_without_backend(
        symmetry.folded_pml_curl_coverage(fields, pml)) == []
    codes, _ = symmetry.folded_axis_kinds(fields.grid, pml)
    assert codes == (symmetry.CODE_PERIODIC,) * 3


def test_two_mirror_planes_are_admitted():
    fields, pml = build(cell_size=(2.0, 2.0, 0.8), boundaries="periodic",
                        symmetry=("X", "Y"))
    assert reasons_without_backend(
        symmetry.folded_pml_curl_coverage(fields, pml)) == []


def test_a_nonzero_k_point_is_refused_on_a_folded_run_too():
    """``parallel-wvgs-force`` in the corpus. The engine refuses a nonzero k on the
    folded axis outright, and the zone-edge pairing MEEP supports as well — a
    kernel cannot lift what the array path will not run."""
    fields, pml = folded("periodic", k_point=(0.0, 0.0, 0.3))
    reasons = reasons_without_backend(symmetry.folded_pml_curl_coverage(fields, pml))
    assert any("k_point" in r for r in reasons), reasons


def test_special_kz_beta_is_refused():
    """MEEP's out-of-plane phase adds couplings to every curl. Silent, not an error."""
    grid = Grid(resolution=10.0, cell_size=(0.8, 2.0, 0.0), dimensions=2,
                boundaries="periodic", symmetry=("Y",), beta=0.4)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    reasons = reasons_without_backend(
        symmetry.folded_pml_curl_coverage(fields, None))
    assert any("beta" in r for r in reasons), reasons


def test_complex_storage_is_refused():
    grid = Grid(resolution=10.0, cell_size=(0.8, 2.0, 0.8), symmetry=("Y",))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    reasons = reasons_without_backend(symmetry.folded_pml_curl_coverage(
        fields, PML(grid=grid, thickness=2)))
    assert any("complex" in r for r in reasons), reasons


def test_no_active_layer_is_refused():
    fields, _ = folded("periodic")
    reasons = reasons_without_backend(
        symmetry.folded_pml_curl_coverage(fields, None))
    assert any("PML" in r for r in reasons), reasons


def cylindrical_fields():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.0, 0.8), dimensions=2,
                cylindrical=True, m=0)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields


def test_a_cylindrical_grid_is_refused():
    reasons = reasons_without_backend(
        symmetry.folded_pml_curl_coverage(cylindrical_fields(), None))
    assert any("cylindrical" in r or "r = 0" in r for r in reasons), reasons


def test_a_conductivity_on_a_curl_target_is_refused():
    """mp.Absorber's path: the three-history conductive recurrence, not this one."""
    fields, pml = folded("periodic")
    import numpy as np

    fields.set_d_conductivity(
        {"Dz": np.full(fields.grid.shape, 0.5, dtype=np.float32)})
    reasons = reasons_without_backend(symmetry.folded_pml_curl_coverage(fields, pml))
    assert any("conductivity" in r for r in reasons), reasons


def test_a_missing_auxiliary_is_refused():
    fields, pml = folded("periodic")
    fields.fu_Bz = None
    reasons = reasons_without_backend(symmetry.folded_pml_curl_coverage(fields, pml))
    assert any("fu_Bz" in r for r in reasons), reasons


def test_a_non_contiguous_field_volume_is_refused():
    fields, pml = folded("periodic")
    fields.Ez = fields.Ez[::1, ::2, ::1]
    reasons = reasons_without_backend(symmetry.folded_pml_curl_coverage(fields, pml))
    assert any("Ez" in r for r in reasons), reasons


def test_mutation_admitting_a_cylindrical_axis_is_caught(monkeypatch):
    """The predicate is only load-bearing if removing a clause changes the verdict.

    Drop the Cartesian clause and a cylindrical grid — whose r axis has a
    different ghost rule AND a different ownership rule — becomes covered but for
    the backend, which is what a silently wrong answer looks like from here.
    """
    fields = cylindrical_fields()
    pml = None
    assert reasons_without_backend(
        symmetry.folded_pml_curl_coverage(fields, pml)) != []

    original = symmetry._shared_grid_reasons

    def without_the_cartesian_clause(f, p, g):
        return [r for r in original(f, p, g)
                if "cylindrical" not in r and "r = 0" not in r]

    monkeypatch.setattr(symmetry, "_shared_grid_reasons",
                        without_the_cartesian_clause)
    widened = reasons_without_backend(symmetry.folded_pml_curl_coverage(fields, pml))
    assert any("axis" in r for r in widened), (
        "with the Cartesian clause gone the CLASSIFICATION must still refuse the "
        "cylindrical r axis; if it does not, one clause is carrying the whole file")


# ---------------------------------------------------------------------------
# The ghost-fill plan's compile-time inputs
# ---------------------------------------------------------------------------

def test_the_fill_entries_are_built_in_x_y_z_order():
    """A corner unowned on two planes must carry the PRODUCT of both parities.

    That is only true if the axes are applied in X, Y, Z order — the order
    ``_fill_symmetry_ghost_cells`` applies them in — which one launch cannot
    promise and a list of launches can.
    """
    fields, _ = build(cell_size=(2.0, 2.0, 0.8), boundaries="periodic",
                      symmetry=("Y", "X"))
    entries = symmetry.ghost_fill_axis_entries(fields.grid, "B")
    assert [e["axis"] for e in entries] == [0, 1]


@pytest.mark.parametrize("family,expected", (
    ("B", {0: (0, 1, 1), 1: (1, 0, 1), 2: (1, 1, 0)}),
    ("D", {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)}),
))
def test_the_fill_entries_carry_each_targets_yee_shift_on_that_axis(family, expected):
    """Which of the three targets gets a near fill, and which a far one."""
    for axis, name in enumerate("XYZ"):
        size = [0.8, 0.8, 0.8]
        size[axis] = 2.0
        fields, _ = build(cell_size=tuple(size), boundaries="periodic",
                          symmetry=(name,))
        entry, = symmetry.ghost_fill_axis_entries(fields.grid, family)
        assert entry["axis"] == axis
        assert entry["shifts"] == expected[axis]


@pytest.mark.parametrize("extent,offset", ((2.0, 2), (2.1, 3)))
def test_the_fill_entry_carries_the_engines_reflect_row_not_n_minus_two(extent, offset):
    fields, _ = folded("periodic", extent=extent)
    entry, = symmetry.ghost_fill_axis_entries(fields.grid, "D")
    assert entry["far"] is True
    assert entry["reflect_row"] == fields.grid.stored_cells(1) - offset


def test_a_folded_metallic_axis_gets_no_far_fill():
    """MEEP holds that plane at zero; imaging it would be a fill where none belongs."""
    fields, _ = folded("metallic")
    entry, = symmetry.ghost_fill_axis_entries(fields.grid, "B")
    assert entry["far"] is False
    assert entry["reflect_row"] == -1


def test_the_fill_entry_carries_the_declared_plane_phase():
    grid = Grid(resolution=10.0, cell_size=(0.8, 2.0, 0.8), boundaries="periodic",
                symmetry=(Mirror("Y", -1),))
    entry, = symmetry.ghost_fill_axis_entries(grid, "B")
    assert entry["phase"] == -1


def test_the_fill_predicate_refuses_a_run_with_no_fold():
    fields, _ = build()
    reasons = reasons_without_backend(
        symmetry.mirror_ghost_fill_coverage(fields, "B"))
    assert any("no mirror plane" in r for r in reasons), reasons


def test_the_fill_predicate_refuses_an_unnamed_family():
    fields, _ = folded("periodic")
    with pytest.raises(ValueError, match="family"):
        symmetry.mirror_ghost_fill_coverage(fields, "E")


@pytest.mark.parametrize("family", ("B", "D"))
def test_the_fill_target_configuration_is_refused_only_for_the_backend(family):
    fields, _ = folded("periodic")
    verdict = symmetry.mirror_ghost_fill_coverage(fields, family)
    assert verdict.covered is False
    assert len(verdict.reasons) == 1, verdict.reasons
    assert "cupy" in verdict.reasons[0]


def test_the_fill_predicate_refuses_a_reflect_row_that_images_its_own_plane(monkeypatch):
    """A row at the top slot would image the plane it writes — a silent no-op fill."""
    fields, _ = folded("periodic")
    top = fields.grid.stored_cells(1) - 1
    monkeypatch.setattr(symmetry, "_far_reflect_rows",
                        lambda grid: (None, top, None))
    reasons = reasons_without_backend(
        symmetry.mirror_ghost_fill_coverage(fields, "B"))
    assert any("reflect row" in r for r in reasons), reasons
