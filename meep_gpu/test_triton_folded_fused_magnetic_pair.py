"""Tests for the folded fused magnetic pair — ``step_B`` welded into ``update_H``.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware
— byte identity against the array path and against the three separately certified
products this launch replaces — lives in
``parity/meep_gpu/probe_triton_folded_fused_magnetic_pair.py``, WHICH HAS NEVER
RUN. That is stated in the module under test and re-stated here, because a test
suite that is green while the device gate is unrun must not be mistaken for a
certification.

What IS pinned here:

* the geometry that makes this product different from the electric one — the near
  fill touches component ``m`` on axis ``m`` for B and on the two OTHER axes for
  D, which moves the constitutive coefficient index and removes the parity
  products;
* every clause of the seam predicate, in both directions;
* the transcription, read off the shipped source: the curl half's parenthesisation
  and the constitutive half's two-accumulation grouping;
* the deferral — nothing in ``launch.py`` or ``fastpath.py`` names this module;
* the design itself, through the gate's own no-device legs, which run here.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import pathlib
import sys

import pytest

from meep_gpu.fields import Fields, IYEE_SHIFTS, mirror_parity
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import folded_fused_magnetic_pair as product
from meep_gpu.triton_kernels import symmetry
from meep_gpu.triton_kernels.coverage import zero_metal_axes

PACKAGE_DIR = pathlib.Path(product.__file__).parent
API_ROOT = PACKAGE_DIR.parents[1]
GATE = API_ROOT / "parity" / "meep_gpu" / "probe_triton_folded_fused_magnetic_pair.py"
RUNNER = (API_ROOT / "parity" / "meep_gpu"
          / "run_triton_folded_fused_magnetic_pair_direct.sh")


def build(cell=(1.6, 3.0, 1.0), boundaries="metallic", mirrors=(("Y", 1),),
          thickness=0.2):
    """A real Grid/Fields/PML triple on NumPy, with the PML storage allocated."""
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=thickness)


def residual(verdict):
    """The reasons that are not the NumPy-host backend clause."""
    return [reason for reason in verdict.reasons if "cupy" not in reason]



def magnetic_deposit(fields):
    """A REAL source that publishes the index the injection writes.

    :class:`Source` below carries a ``field_type`` and nothing else: enough to PLACE
    a source in a seam, and deliberately not enough to CARRY one -- the repair
    refuses it by name for publishing no deposit index. A case about the carry needs
    the engine's own source, and one that deposits nothing would make the admitting
    assertion pass by measuring an empty scatter, so both are checked here rather
    than assumed at each call site.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    source = VolumeSource(grid=fields.grid, component="Hy",
                          center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    assert source._n_source_points, "the case deposits nothing and measures nothing"
    assert deposit_repair._deposit_index(source) is not None, (
        "the case cannot exercise the repair: this source publishes no deposit index")
    return source


class Source:
    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def load_gate():
    spec = importlib.util.spec_from_file_location("probe_folded_fused_B", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The geometry: this is the B family and not the D family
# ---------------------------------------------------------------------------

def test_the_near_fill_touches_each_B_component_on_its_own_axis_alone():
    """The one structural fact the whole carry rests on, read off IYEE_SHIFTS.

    ``_fill_symmetry_ghost_cells`` (stepping.py:1453) fills component ``m`` on axis
    ``a`` exactly when ``iyee[m][a] == 0``. For B that is ``a == m`` — the
    component's OWN axis, and only it — where for D it is ``a != m``. Everything
    that makes this product simpler than the electric one (no parity products, at
    most one destination per component) and everything that makes it harder (the
    coefficient index moves) follows from this line.
    """
    magnetic = {axis: tuple(index for index, name in enumerate(("Bx", "By", "Bz"))
                            if IYEE_SHIFTS[name][axis] == 0) for axis in range(3)}
    electric = {axis: tuple(index for index, name in enumerate(("Dx", "Dy", "Dz"))
                            if IYEE_SHIFTS[name][axis] == 0) for axis in range(3)}
    assert magnetic == {0: (0,), 1: (1,), 2: (2,)}, magnetic
    assert electric == {0: (1, 2), 1: (0, 2), 2: (0, 1)}, electric
    assert product.NEAR_FILL_COMPONENTS == ((0,), (1,), (2,))


def test_a_B_component_is_a_fill_destination_on_at_most_one_axis():
    """Why no parity product can arise here, and the D side's does.

    ``_fill_symmetry_ghost_cells`` applies the axes in X, Y, Z order so a cell
    unowned on two planes carries the PRODUCT of both parities. On the B family a
    component has exactly one unowned plane, so two folded axes never meet on one
    component and the application order is unobservable — which is what lets one
    launch carry the fill without reproducing an ordering.
    """
    for axis, components in enumerate(product.NEAR_FILL_COMPONENTS):
        assert len(components) == 1, (axis, components)
    per_component = [sum(1 for axis in range(3)
                         if index in product.NEAR_FILL_COMPONENTS[axis])
                     for index in range(3)]
    assert per_component == [1, 1, 1], per_component
    # The contrast, so the claim is a comparison and not an assertion about one
    # side: a D component is unowned on TWO axes.
    electric = [sum(1 for axis in range(3) if IYEE_SHIFTS[name][axis] == 0)
                for name in ("Dx", "Dy", "Dz")]
    assert electric == [2, 2, 2], electric


def test_the_constitutive_coefficient_index_moves_on_the_magnetic_half():
    """The fill images ALONG the axis ``update_H`` indexes its coefficient on.

    ``H_CONSTITUTIVE_TERMS`` (stepping.py:227) pairs Hx with axis x, Hy with y, Hz
    with z — the component's own axis, MEEP's ``dsigw``. The near fill images
    component ``m`` along axis ``m``. So the fill's source (stored 2) and its
    destination (stored 0) differ in exactly the coordinate the coefficient is
    indexed on, and the destination's pair is NOT the source's. On the electric
    half the fill runs on ``a != m``, so the pair is shared — which is the
    property the Metal folded D/E pair relies on and this one cannot.
    """
    from meep_gpu.stepping import E_CONSTITUTIVE_TERMS, H_CONSTITUTIVE_TERMS

    for index, (component, source, axis_name) in enumerate(H_CONSTITUTIVE_TERMS):
        axis = "xyz".index(axis_name)
        assert axis == index, (component, source, axis_name)
        # The fill's axis for this component is its own axis: the indices move.
        assert index in product.NEAR_FILL_COMPONENTS[axis]
    for index, entry in enumerate(E_CONSTITUTIVE_TERMS):
        axis = "xyz".index(entry[-1])
        assert axis == index
        electric_fill_axes = [a for a in range(3)
                              if IYEE_SHIFTS[("Dx", "Dy", "Dz")[index]][a] == 0]
        assert axis not in electric_fill_axes, (index, electric_fill_axes)


def test_the_near_fill_parity_is_plus_the_planes_phase_for_every_B_component():
    """One signed constexpr per axis is the entire parity input, exhaustively.

    ``mirror_parity(c, axis, phase) == phase * (1 - 2 * iyee[c][axis])``
    (fields.py:117), and the near fill only ever touches shift-0 components, so
    the factor is ``+phase``. Checked over all 12 components x 3 axes x 2 phases
    so the collapse is a measurement and not a remembered identity.
    """
    for component, shifts in IYEE_SHIFTS.items():
        for axis in range(3):
            for phase in (1, -1):
                assert mirror_parity(component, axis, phase) == (
                    phase * (1 - 2 * shifts[axis])), (component, axis, phase)
    for axis, name in enumerate(("Bx", "By", "Bz")):
        for phase in (1, -1):
            assert mirror_parity(name, axis, phase) == phase


def test_the_wall_clear_and_the_near_fill_are_disjoint_on_every_folded_axis():
    """``_zero_metal`` skips a folded axis, so the two passes cannot meet.

    Both act on component ``m`` at stored cell 0 of axis ``m``; the wall clear
    additionally requires ``is_metallic(axis) and not is_mirrored(axis)``
    (stepping.py:2284-2286). The D side has to decide an ORDER between them
    because its two cell sets intersect at a corner; here the question is vacuous,
    which is checked rather than asserted.
    """
    for mirrors in ((("Y", 1),), (("X", 1),), (("X", 1), ("Y", -1))):
        fields, _pml = build(cell=(3.0, 3.0, 1.0), mirrors=mirrors)
        walls = zero_metal_axes(fields.grid)
        for axis in range(3):
            if fields.grid.is_mirrored(axis):
                assert not walls[axis], (mirrors, axis, walls)


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_a_folded_metallic_grid_is_admitted_modulo_the_numpy_host():
    fields, pml = build()
    verdict = product.folded_fused_magnetic_pair_coverage(fields, pml, ())
    assert residual(verdict) == [], residual(verdict)
    assert not verdict.covered  # the NumPy host is refused, by the backend clause
    assert any("cupy" in reason for reason in verdict.reasons)


def test_an_electric_source_is_admitted_and_a_magnetic_one_is_refused_by_name():
    """The seam clause, in BOTH directions.

    The driver injects magnetic currents between ``step_B`` and ``update_H``
    (driver.py:3283) and electric ones in the other seam (driver.py:3296-3301). A
    predicate that refused both would be safe and useless: 157 of the corpus's 186
    rows carry an electric source.
    """
    fields, pml = build()
    admitted = product.folded_fused_magnetic_pair_coverage(
        fields, pml, (Source("D"), Source("D")))
    assert residual(admitted) == []

    # (1) CARRIED since 2026-08-30: a real magnetic deposit in this pair's own seam.
    carried = product.folded_fused_magnetic_pair_coverage(
        fields, pml, (Source("D"), magnetic_deposit(fields)))
    assert residual(carried) == [], residual(carried)

    # (2) STILL REFUSED BY NAME: a magnetic source whose deposit cannot be saved.
    refused = product.folded_fused_magnetic_pair_coverage(
        fields, pml, (Source("D"), Source("B")))
    assert any("does not publish the index it writes" in reason
               and "source 1" in reason
               for reason in refused.reasons), refused.reasons


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    """Ignorance is never an empty set: ``Fields`` does not hold the sources."""
    fields, pml = build()
    verdict = product.folded_fused_magnetic_pair_coverage(fields, pml, None)
    assert any("was not declared" in reason for reason in verdict.reasons)
    assert product.plan_folded_fused_magnetic_pair(fields, pml, None) is None


def test_holding_the_carry_flag_False_puts_the_magnetic_seam_refusal_straight_back(
        monkeypatch):
    """(3) of the source clause: the admission above is the FLAG's doing.

    Split into its own case rather than folded into the one above, because it is
    the only assertion in this file that is about the retired behaviour, and a
    reader deleting it should have to delete something named for what it does.
    """
    fields, pml = build()
    monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
    held = product.folded_fused_magnetic_pair_coverage(
        fields, pml, (magnetic_deposit(fields),))
    assert any("is magnetic" in reason for reason in held.reasons), held.reasons


def test_a_folded_periodic_axis_is_CARRIED_not_refused():
    """THE FLIP OF 2026-08-20, at the predicate.

    This asserted the OPPOSITE until the far carry landed: a folded PERIODIC axis
    was refused BY NAME on ``fill_folded_far_ghosts_B``, which runs inside this
    seam (driver.py:3287). The kernel images that plane now, so the only residual
    reason on a laptop is the backend one.
    """
    fields, pml = build(boundaries="periodic")
    verdict = product.folded_fused_magnetic_pair_coverage(fields, pml, ())
    residual = [reason for reason in verdict.reasons if "cupy" not in reason]
    assert residual == [], residual
    assert not any("fill_folded_far_ghosts_B" in reason
                   for reason in verdict.reasons)
    assert "fill_folded_far_ghosts_B" in product.REPLACES


def test_the_two_fills_reach_complementary_axis_sets():
    """NEAR on the component's own axis, FAR on the two that are not.

    ``fields.IYEE_SHIFTS`` says a B component's shift is 0 on its own axis and 1 on
    the others, and the two fills select on exactly that
    (``stepping._fill_symmetry_ghost_cells``:1441 vs
    ``_fill_folded_far_ghosts``:1516). Both tables in the module are DERIVED from
    that one; this is the pin that the derivation is the right way up, and that the
    far carry did not simply reuse the near fill's axis set.
    """
    for axis in range(3):
        near = set(product.NEAR_FILL_COMPONENTS[axis])
        far = set(product.FAR_FILL_COMPONENTS[axis])
        assert near == {index for index, name in enumerate(("Bx", "By", "Bz"))
                        if IYEE_SHIFTS[name][axis] == 0}
        assert far == {index for index, name in enumerate(("Bx", "By", "Bz"))
                       if IYEE_SHIFTS[name][axis] == 1}
        assert near | far == {0, 1, 2} and not (near & far)
        assert near == {axis}


def test_the_far_carrys_parity_is_minus_the_plane_phase():
    """``mirror_parity`` is ``phase * (1 - 2 * iyee)``, so a shift-1 component takes
    ``-phase`` — the opposite of the near fill's ``+phase``, and the sign a carry
    that reused the near rule would get backwards on every far ghost.

    Checked against ``fields.mirror_parity`` itself over both phases and all three
    axes, and against the kernel text, which spells the far weights with a leading
    minus."""
    source = (PACKAGE_DIR / "folded_fused_magnetic_pair.py").read_text(
        encoding="utf-8")
    for axis, name in enumerate(("Bx", "By", "Bz")):
        for phase in (1, -1):
            assert mirror_parity(name, axis, phase) == phase        # its own axis
            for other in range(3):
                if other == axis:
                    continue
                assert mirror_parity(name, other, phase) == -phase   # a far axis
    for weight, register in (("-PHY", "v0"), ("-PHZ", "v0"), ("-PHX", "v1"),
                             ("-PHZ", "v1"), ("-PHX", "v2"), ("-PHY", "v2")):
        assert f"{weight} * {register}" in source, (weight, register)


def test_the_plan_refuses_a_reflect_row_it_cannot_own():
    """The far ghost's destination is written by the lane AT the reflect row, so a
    row outside the allocation is an out-of-range store from a lane that owns
    neither cell — refused at build time rather than launched."""
    from meep_gpu.triton_kernels.symmetry import (
        CODE_MIRROR_PERIODIC, CODE_PERIODIC,
    )

    codes = (CODE_PERIODIC, CODE_MIRROR_PERIODIC, CODE_PERIODIC)
    common = dict(shape=(8, 8, 1), dtdx=0.35, bc=codes,
                  zero_metal=(False, False, False), phases=(0, 1, 0), block=64,
                  targets=(), auxiliaries=(), sources=(), curl_coefficients=(),
                  h_targets=(), h_aux=(), h_coefficients=())
    with pytest.raises(ValueError, match="outside"):
        product.FoldedFusedMagneticPairPlan(reflect=(None, 7, None), **common)
    with pytest.raises(ValueError, match="not folded PERIODIC"):
        product.FoldedFusedMagneticPairPlan(reflect=(3, 6, None), **common)


def test_an_unfolded_grid_is_refused_because_this_is_a_composition_product():
    fields, pml = build(cell=(1.6, 1.6, 1.0), mirrors=())
    verdict = product.folded_fused_magnetic_pair_coverage(fields, pml, ())
    assert any("no mirror plane is active" in reason for reason in verdict.reasons)


def test_no_pml_is_refused_because_update_H_is_a_no_op_without_one():
    """``update_H`` returns immediately with no active PML (stepping.py:915-916)."""
    fields, _pml = build()
    verdict = product.folded_fused_magnetic_pair_coverage(fields, None, ())
    assert any("no active PML" in reason for reason in verdict.reasons), \
        verdict.reasons


def test_the_predicate_reports_both_halves_reasons_with_their_side_named():
    fields, pml = build(cell=(1.6, 1.6, 1.0), mirrors=())
    reasons = product.folded_fused_magnetic_pair_coverage(fields, pml, ()).reasons
    assert any(reason.startswith("folded curl half: ") for reason in reasons)
    assert any(reason.startswith("folded constitutive half: ") for reason in reasons)


class _ShrunkGrid:
    """A real grid that reports a two-cell stored extent on its folded axis.

    A proxy rather than a stub: every other question — the boundary kinds, the
    mirror phase, the ownership split — is answered by the engine's own object, so
    the clause under test is the only thing that changed. A hand-written stub
    would have to reproduce ``_boundary_kinds``' whole contract and would fail for
    a reason that has nothing to do with the fill's source plane.
    """

    def __init__(self, grid, axis: int, cells: int,
                 shrink_stored: bool = True) -> None:
        self.__dict__["_grid"] = grid
        self.__dict__["_axis"] = axis
        self.__dict__["_cells"] = cells
        self.__dict__["_shrink_stored"] = shrink_stored
        shape = list(grid.shape)
        shape[axis] = cells
        self.__dict__["shape"] = tuple(shape)

    def __getattr__(self, name):
        return getattr(self._grid, name)

    def stored_cells(self, axis):
        if self._shrink_stored and axis == self._axis:
            return self._cells
        return self._grid.stored_cells(axis)


class _FieldsOn:
    """``fields`` with one attribute swapped: the grid."""

    def __init__(self, fields, grid) -> None:
        self.__dict__["_fields"] = fields
        self.__dict__["grid"] = grid

    def __getattr__(self, name):
        return getattr(self._fields, name)


def test_a_folded_axis_with_two_or_fewer_stored_cells_is_refused():
    """The near fill images stored cell 2 FROM THAT CELL'S OWN LANE.

    ``folded_axis_kinds`` already refuses such an axis; the clause is restated in
    the shipped predicate because a missing source plane here is an out-of-range
    WRITE from a different lane rather than a soft error, and this test asks the
    shipped predicate rather than the shared one.
    """
    fields, pml = build()
    axis = next(a for a in range(3) if fields.grid.is_mirrored(a))

    # (a) the INHERITED clause: `folded_axis_kinds` reads `grid.stored_cells`.
    shrunk = _FieldsOn(fields, _ShrunkGrid(fields.grid, axis, 2))
    verdict = product.folded_fused_magnetic_pair_coverage(shrunk, pml, ())
    assert not verdict.covered
    assert any("near ghost images stored cell" in reason
               for reason in verdict.reasons), verdict.reasons
    assert product.plan_folded_fused_magnetic_pair(shrunk, pml, ()) is None

    # (b) the RESTATED clause, and this is why it is restated rather than
    # inherited: it reads ``grid.shape``, which is what the kernel's flat index
    # arithmetic uses, where `folded_axis_kinds` reads ``grid.stored_cells``. A
    # grid whose two readers disagree passes the inherited clause and would hand
    # the carry a source plane that is not there — an out-of-range write from a
    # lane that is not the destination's.
    drifted = _FieldsOn(fields, _ShrunkGrid(fields.grid, axis, 2,
                                            shrink_stored=False))
    reasons = product.folded_fused_magnetic_pair_coverage(drifted, pml, ()).reasons
    assert any("images it from that cell's own lane" in reason
               for reason in reasons), reasons


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_backward_constexpr_is_step_Bs_and_only_step_Bs():
    """The carry is transcribed for the B family's Yee shifts and no other."""
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    assert product.BACKWARD == SUB_STEPS["step_B"]["backward"] == 0
    assert product.CURL_SUB_STEP == "step_B"
    assert product.CONSTITUTIVE_SIDE == "H"
    assert product.REPLACES == ("step_B", "fill_B", "zero_metal_B",
                               "fill_folded_far_ghosts_B", "update_H")


def test_the_kernels_literal_source_index_is_the_named_constant():
    """The jit body spells MEEP's ``io = -2`` as a literal, and this is the pin.

    ``symmetry.mirror_ghost_fill`` does the same (``base + 2 * stride``) for the
    same reason — a jit body that closes over a module-level Python int is a
    Triton-version question no laptop can settle — so the literal is checked
    against ``MIRROR_SOURCE_INDEX`` here rather than trusted. Baking ``n - 1``
    would image the wrong plane on every folded run.
    """
    source = (PACKAGE_DIR / "folded_fused_magnetic_pair.py").read_text(encoding="utf-8")
    assert symmetry.MIRROR_SOURCE_INDEX == 2
    index = symmetry.MIRROR_SOURCE_INDEX
    for coordinate, stride in (("i", " * nyz"), ("j", " * nz"), ("k", "")):
        assert f"live & ({coordinate} == {index})" in source, coordinate
        assert f"= -{index}{stride}" in source, coordinate
    # ...and the name is still the engine's, so a change there fails here.
    from meep_gpu.stepping import MIRROR_SOURCE_INDEX as engine_index

    assert engine_index == symmetry.MIRROR_SOURCE_INDEX


def test_the_plan_refuses_a_fold_that_also_carries_a_wall_or_a_bad_phase():
    """The plan re-checks what the predicate checked, at the point of binding.

    A predicate can be bypassed — ``plan_..._from_arrays`` is the gate's route and
    runs none of it — so the invariants the kernel's constexprs encode are
    asserted where the constexprs are chosen.
    """
    kwargs = dict(shape=(4, 8, 4), dtdx=0.5, block=256,
                  targets=[], auxiliaries=[], sources=[], curl_coefficients=[],
                  h_targets=[], h_aux=[], h_coefficients=[])
    with pytest.raises(ValueError, match="not \\+1 or -1"):
        product.FoldedFusedMagneticPairPlan(
            bc=(1, symmetry.CODE_MIRROR_METALLIC, 1), zero_metal=(True, False, True),
            phases=(0, 0, 0), **kwargs)
    with pytest.raises(ValueError, match="fold and a wall clear"):
        product.FoldedFusedMagneticPairPlan(
            bc=(1, symmetry.CODE_MIRROR_METALLIC, 1), zero_metal=(True, True, True),
            phases=(0, 1, 0), **kwargs)
    with pytest.raises(ValueError, match="folded PERIODIC"):
        product.FoldedFusedMagneticPairPlan(
            bc=(1, symmetry.CODE_MIRROR_PERIODIC, 1), zero_metal=(True, False, True),
            phases=(0, 1, 0), **kwargs)


def test_mirror_phases_answers_zero_off_a_folded_axis_and_the_plane_on_one():
    fields, _pml = build(mirrors=(("Y", -1),))
    assert product.mirror_phases(fields.grid) == (0, -1, 0)
    fields, _pml = build(cell=(3.0, 3.0, 1.0), mirrors=(("X", 1), ("Y", -1)))
    assert product.mirror_phases(fields.grid) == (1, -1, 0)


def test_the_builder_refuses_every_configuration_the_predicate_refuses():
    for kwargs in ({"boundaries": "periodic"},
                   {"cell": (1.6, 1.6, 1.0), "mirrors": ()}):
        fields, pml = build(**kwargs)
        assert product.plan_folded_fused_magnetic_pair(fields, pml, ()) is None
    fields, pml = build()
    # Admitted modulo the host, so the builder still refuses — on NumPy.
    assert product.plan_folded_fused_magnetic_pair(fields, pml, ()) is None


# ---------------------------------------------------------------------------
# The optional dependency stays optional
# ---------------------------------------------------------------------------

def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules,
                        "meep_gpu.triton_kernels.folded_fused_magnetic_pair",
                        raising=False)
    module = importlib.import_module(
        "meep_gpu.triton_kernels.folded_fused_magnetic_pair")
    assert module.triton is None
    assert module.folded_fused_curl_constitutive_B is None
    fields, pml = build()
    assert module.folded_fused_magnetic_pair_coverage(fields, pml, ()).reasons
    assert module.plan_folded_fused_magnetic_pair(fields, pml, ()) is None
    with pytest.raises(ImportError, match="triton"):
        module.folded_fused_curl_constitutive_B_kernel()


# ---------------------------------------------------------------------------
# File ownership and the deferral
# ---------------------------------------------------------------------------

def code_of(path: pathlib.Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def test_this_module_touches_neither_dispatch_nor_the_other_track():
    code = code_of(PACKAGE_DIR / "folded_fused_magnetic_pair.py")
    assert "fastpath" not in code
    assert "cuda_kernels" not in code
    assert "options=" not in code


def test_the_single_launch_site_passes_the_packages_shared_guard_constant():
    """A site that omits the guard compiles a kernel that is bit-wrong and plausible."""
    source = (PACKAGE_DIR / "folded_fused_magnetic_pair.py").read_text(encoding="utf-8")
    spelling = "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)"
    assert source.count("enable_fp_fusion=") == 1
    assert spelling in source


def test_the_planner_names_this_module_only_through_the_folded_fusion_route():
    """THE DEFERRAL WAS LIFTED 2026-08-27, and what replaced it is narrower.

    This used to assert that NOTHING in ``launch.py`` names this module — the seam
    that kept a deferred product deferred, on the ground that ``plan_step`` assigns
    at most one plan per ``STEP_ORDER`` slot while this product spans four driver
    passes. The two-slot protocol (``launch._install_fused_pair``) is that missing
    composition rule and ``launch.FOLDED_FUSED_PAIR_ARMS`` is the mapping it needs,
    so ``launch.py`` now names this module in exactly two places: the declared lazy
    seam (``SUPPORT_MODULES``) and the import inside
    ``_folded_fused_pair_entries``.

    THE OTHER TWO FILES ARE UNCHANGED, and that is the half of the old claim that
    still holds: ``fastpath.plan_fast_path`` is untouched, so no default run reaches
    this plan, and the package's ``__init__`` exports nothing from here. The KERNEL
    symbol stays private to this module in all three — the planner reaches the
    product through its builder, never through its kernel.

    ``fastpath.py`` NOW NAMES THE FAMILY, AND THE CLAIM IS NARROWED AGAIN RATHER
    THAN DROPPED (2026-08-29). The fusion opt-in gave every fused label an
    ``ARM_CERTIFICATION`` row, so the family name appears there — as a STRING in a
    lookup table read only when an artifact is written, which is the opposite of an
    entry point: it exists so a record that names a dispatched arm also names the
    gate that cut it. "The string is absent" was a proxy for "the planner cannot
    reach this module from fastpath", and the proxy stopped fitting. What replaces
    it is the property itself, in BOTH directions: fastpath must not import the
    module and must not name its kernel, every occurrence of the family name must
    fall inside the ``ARM_CERTIFICATION`` assignment, and the row must BE there
    mapping this product's arm label to this product's weld. A row that went
    missing, or a name that appeared anywhere else, now fails.

    OWED, AND NOT DOABLE IN THIS ROUND: the module's own docstring still says
    "Nothing in ``launch.py`` names this module". Its bytes are digest-pinned by
    ``fingerprints.json``'s ``triton_folded_fused_magnetic_pair_device_gate``
    (asserted by ``test_the_module_claims_identity_ONLY_through_the_weld_that_measured_it``),
    which this round may not edit, so the correction has to land in the same edit
    as the re-weld.
    """
    launch = (PACKAGE_DIR / "launch.py").read_text(encoding="utf-8")
    fastpath = (PACKAGE_DIR.parent / "fastpath.py").read_text(encoding="utf-8")
    exported = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")

    assert "folded_fused_magnetic_pair" in launch
    assert "from .folded_fused_magnetic_pair import (" in launch
    assert "folded_fused_curl_constitutive_B" not in launch, (
        "the planner reaches this product through plan_folded_fused_magnetic_pair; "
        "naming the kernel would be a second, unmeasured entry point")

    # The package's own namespace is unchanged: it exports nothing from here.
    assert "folded_fused_magnetic_pair" not in exported, "triton_kernels/__init__.py"
    assert "folded_fused_curl_constitutive_B" not in exported, "triton_kernels/__init__.py"

    # THE KERNEL stays private to this module in fastpath too — that half of the
    # old claim is untouched, and it is the half that would make a second entry
    # point.
    assert "folded_fused_curl_constitutive_B" not in fastpath, "fastpath.py"

    _fastpath_names_the_family_only_in_arm_certification(
        fastpath, "folded_fused_magnetic_pair")

    from meep_gpu import fastpath as fastpath_module  # noqa: PLC0415

    assert fastpath_module.ARM_CERTIFICATION["fused pair B (folded)"] == (
        "folded_fused_magnetic_pair",
        "triton_folded_fused_magnetic_pair_device_gate"), (
        "the arm label this product wins must map to THIS family and to the weld "
        "that measured it; an unmapped fused arm writes an artifact naming no gate")


def _fastpath_names_the_family_only_in_arm_certification(source: str,
                                                         family: str) -> None:
    """Every mention of ``family`` in ``fastpath.py`` falls inside ARM_CERTIFICATION.

    Read structurally rather than by counting occurrences: the assignment's line
    span comes from the parse tree, so a row added, removed or re-indented inside
    the table does not move the boundary, and a mention that escapes the table —
    an import, a predicate, a refusal string — lands outside it and fails with the
    line number.
    """
    import ast  # noqa: PLC0415

    tree = ast.parse(source)
    span = None
    for node in ast.walk(tree):
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else getattr(node, "targets", []))
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "ARM_CERTIFICATION":
                span = (node.lineno, node.end_lineno)
    assert span is not None, "fastpath.py no longer defines ARM_CERTIFICATION"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] + [
                getattr(node, "module", None) or ""]
            assert not any(family in name for name in names), (
                f"fastpath.py imports {family} at line {node.lineno}: that is an "
                "entry point, not a certification row")

    escaped = [index for index, line in enumerate(source.splitlines(), 1)
               if family in line and not span[0] <= index <= span[1]]
    assert not escaped, (
        f"fastpath.py names {family} outside ARM_CERTIFICATION (lines {escaped}); "
        f"the table spans {span[0]}-{span[1]}")


def test_the_planner_can_actually_reach_this_products_builder():
    """The route, exercised rather than read off the text above.

    ``_folded_fused_pair_entries`` is the one seam through which ``plan_step``
    obtains this family's predicate and builder. A test that only grepped
    ``launch.py`` would pass on a mention in a comment.
    """
    from meep_gpu.triton_kernels import launch as launch_module

    entries = launch_module._folded_fused_pair_entries()
    coverage, builder = entries["B"]
    assert coverage is product.folded_fused_magnetic_pair_coverage
    assert builder is product.plan_folded_fused_magnetic_pair
    assert launch_module.FOLDED_FUSED_PAIRS["B"]["curl"] == "step_B"
    assert launch_module.FOLDED_FUSED_PAIRS["B"]["update"] == "update_H"


def test_the_gate_exists_names_the_product_and_reports_the_run_it_took():
    source = GATE.read_text(encoding="utf-8")
    assert "plan_folded_fused_magnetic_pair" in source
    assert "folded_fused_curl_constitutive_B" in source
    assert "RELEASED 2026-08-20" in source, (
        "the gate ran and released on 2026-08-20; its header said UNRUN until "
        "then and must now say what it measured")


def test_the_module_claims_identity_ONLY_through_the_weld_that_measured_it():
    """The claim moved, so what pins it moved with it.

    Until 2026-08-20 this asserted the module contained no identity claim at all,
    because none had been measured. A gate has now released, so the module MAY
    claim identity — and the thing worth pinning is no longer the absence of a
    claim but its PROVENANCE: an identity claim in this file has to be the one a
    weld records, and the weld has to be the one whose digests match this file.
    """
    import hashlib
    import json

    source = (PACKAGE_DIR / "folded_fused_magnetic_pair.py").read_text(encoding="utf-8")
    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    weld = record["triton_folded_fused_magnetic_pair_device_gate"]
    assert weld["status"] == "PASS"
    assert "RELEASED 2026-08-20" in source

    # The weld names THIS file, at the bytes on disk.
    named = {name: digest for name, digest in weld["source_sha256"].items()
             if name.endswith("folded_fused_magnetic_pair.py")}
    assert named, "the weld does not name the module it certifies"
    for name, digest in named.items():
        live = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
        assert live == digest, (
            f"{name} has changed since the gate ran; the identity claim in its "
            f"docstring no longer describes the bytes that were measured")

    # AND THE BOUND THAT DID NOT MOVE: a weld licenses a claim, not a dispatch.
    # THE VOCABULARY MOVED, THE CLAIM DID NOT. Until 2026-08-27 these modules
    # disclaimed dispatch with the words "NOT WIRED". The routing round then gave
    # them an arm -- launch.py:1168 "THE TWO FOLDED PAIRS, routed 2026-08-27 by
    # _install_folded_fused_pairs" -- so "not wired" became false while the thing
    # this test is defending stayed exactly true, and the modules now say
    # "ROUTED BUT STILL NOT DISPATCHED" instead. This assertion tracks that
    # wording and enforces the SAME bound the comment above states: a weld
    # licenses a claim, not a dispatch. It is not weaker -- a module that
    # silently began claiming dispatch still fails here.
    assert "NOT DISPATCHED" in source, (
        "a released gate is not a wiring; the module must still say so")


# ---------------------------------------------------------------------------
# The gate's own no-device legs, run here
# ---------------------------------------------------------------------------

def test_the_gates_transcription_leg_traces_every_arithmetic_line_to_its_source():
    gate = load_gate()
    row = gate.transcription_leg()
    assert row["passed"], row["findings"]
    assert row["curl_lines_checked"] == 6
    assert row["constitutive_lines_checked"] == 6
    # TWO, not six: the seven carry statements moved into the `_carry_ghost`
    # device function when the far carry landed, because one source lane can now
    # own up to seven ghost cells and seven inlined copies would be seven places
    # for one statement to drift. What replaces the other four checks is
    # `CARRY_CALLS`, which requires the fused body to make the call for EVERY
    # destination the ownership rule derives.
    assert row["carry_lines_checked"] == 2


def test_the_gates_design_sweep_agrees_with_the_array_path_and_catches_its_flips():
    """The design, measured: which value, at which index, in which order.

    The faithful emulation must equal the array-path composition of the four
    passes; dropping the fill and reordering it past the constitutive must both be
    caught. This is not the kernel — a laptop cannot see a memory hazard — but it
    is the design decision that the kernel then has to implement.
    """
    gate = load_gate()
    row = gate.design_sweep_leg()
    assert row["passed"], row["findings"]
    by_knob = {}
    for entry in row["rows"]:
        by_knob.setdefault(entry["knob"], []).append(entry["identical"])
    assert all(by_knob["faithful"]), by_knob
    assert not any(by_knob["drop_the_near_fill"]), by_knob
    assert not any(by_knob["constitutive_before_the_fill"]), by_knob


def test_the_moved_coefficient_index_is_a_construction_fact_not_an_observable_one():
    """MEASURED, and the reason the destination load must not be 'simplified'.

    The mirror plane is the folded axis's LOW face and no absorber reaches it, so
    ``kps[0] == kps[2]`` on that axis in every real configuration probed — 0 of 64
    separate — while the unfolded control separates. Reusing the source lane's
    coefficient pair would therefore be byte-identical today and wrong by
    construction, which is exactly the kind of coincidence a later reader would
    'clean up'. This test is the note that stops that.
    """
    gate = load_gate()
    reach = gate.coefficient_reach_leg()
    assert reach["passed"], reach
    assert reach["cases"] == 64
    assert reach["separated"] == 0, reach["rows"][:4]
    assert reach["unfolded_control_separated"]

    sweep = gate.design_sweep_leg()
    flips = [entry["identical"] for entry in sweep["rows"]
             if entry["knob"] == "destination_coefficient_is_the_sources"]
    assert flips and all(flips), flips


def test_the_gates_predicate_leg_exercises_every_clause_in_both_directions():
    gate = load_gate()
    row = gate.predicate_leg()
    assert row["passed"], row["findings"]
    assert {entry["case"] for entry in row["rows"]} == {
        "folded_metallic_no_sources", "folded_metallic_electric_source",
        "folded_metallic_magnetic_deposit_is_carried",
        "folded_metallic_magnetic_source_without_a_deposit_index",
        "folded_metallic_magnetic_deposit_refused_when_the_flag_is_held_False",
        "undeclared_sources", "no_pml",
        "folded_periodic_axis_is_admitted", "unfolded_grid"}
    directions = {entry["case"]: entry["expect_admitted"] for entry in row["rows"]}
    # THE THREE ROWS THE 2026-08-30 CARRY ADDED, asserted by NAME and by direction.
    # A set equality alone would let a future round rename the carried row to the
    # refused one and stay green, so the directions are pinned individually: the
    # deposit is ADMITTED, the source that cannot publish its index is REFUSED, and
    # the deposit is REFUSED AGAIN once the flag is held down.
    assert directions["folded_metallic_magnetic_deposit_is_carried"] is True
    assert directions[
        "folded_metallic_magnetic_source_without_a_deposit_index"] is False
    assert directions[
        "folded_metallic_magnetic_deposit_refused_when_the_flag_is_held_False"] is False


def test_the_gate_arms_every_mutation_it_declares():
    """A rewrite that matches nothing reports a real defect as uncaught.

    That failure mode is not hypothetical in this tree: the fused-electric round
    lost its ``kernel=`` argument on the way to the plan and reported 4/4 real
    defects as 120/120 identical. Every rewrite here is applied to the SHIPPED
    text on the laptop, and a zero-hit rewrite fails at the merge bar rather than
    on the device.
    """
    gate = load_gate()
    source = gate._shipped_text("folded_fused_curl_constitutive_B")
    unarmed = []
    for name, _why, _expectation, rewrite in gate.mutation_table():
        mutated, hits = rewrite(source)
        if hits == 0 or mutated == source:
            unarmed.append(name)
    assert not unarmed, unarmed


def changed_lines(before, after):
    """Indices into ``before`` that a rewrite really replaced or deleted.

    A POSITIONAL zip IS WRONG HERE and was, on 2026-08-20: several rewrites
    change the line COUNT — ``m1_wall_clear_dropped`` turns three lines into two
    — so every line after the edit shifts by one and a zip comparison reports
    almost the whole kernel as changed. The guard analysis below then finds some
    top-level line among them, concludes the mutation is live, and passes. It
    passed on the very defect it was written to catch.

    SequenceMatcher reports the actual replaced and deleted ranges, which is what
    "the lines this mutation touches" means.
    """
    import difflib

    out = []
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            out.extend(range(i1, i2))
        elif tag == "insert":
            out.append(min(i1, len(before) - 1))   # the line the insert lands before
    return sorted(set(out))


def test_every_mutation_is_scored_on_a_grid_that_ENTERS_the_branch_it_rewrites():
    """The layer under "a rewrite that matches nothing": a rewrite that matches DEAD LINES.

    MEASURED 2026-08-20, on this product's first ever device run. Every mutation
    was scored on ``CASES[1]`` (``odd_y_fold_metallic``), and
    ``m5_source_index_off_by_one`` rewrites the fill guarded by
    ``if BCX == MIRROR_METALLIC:`` — which needs X to be the FOLDED axis, not
    merely a metallic wall. On a Y-folded grid that branch is never entered, so
    the leg launched a kernel whose mutated lines could not execute and reported
    UNCAUGHT. It was read for four rounds as a defect the comparator could not
    see; it was a statement about the case table.

    The sibling test above catches a rewrite that hits ZERO lines. This one
    catches a rewrite that hits real lines the scored grid never reaches, which
    is indistinguishable from the first in the artifact and much easier to miss.

    THE CHECK IS STRUCTURAL, not a list: the enclosing ``if`` is read off the
    device text for whichever lines each rewrite actually changed, so a mutation
    added later is covered without anyone remembering to add it here.
    """
    gate = load_gate()
    source = gate._shipped_text("folded_fused_curl_constitutive_B")
    lines = source.splitlines()

    def enclosing_guard(index: int) -> str:
        """The guard a changed line sits under — or IS.

        A LINE THAT IS ITSELF AN ``if`` COUNTS AS ITS OWN GUARD, and that is not a
        nicety. ``m1_wall_clear_dropped`` replaces three lines with two, DELETING
        the ``if ZM_X:`` header; asking for that line's ENCLOSING guard walks out
        to the surrounding scope, which is unguarded, so the mutation reads as
        live on any grid at all. Deleting a guard is only meaningful where the
        guard would have been entered, so the condition on the deleted line is
        exactly the requirement.
        """
        own = lines[index].strip()
        if own.startswith("if "):
            return own
        body_indent = len(lines[index]) - len(lines[index].lstrip())
        for above in range(index - 1, -1, -1):
            text = lines[above]
            if not text.strip():
                continue
            indent = len(text) - len(text.lstrip())
            if indent < body_indent and text.strip().startswith("if "):
                return text.strip()
            if indent < body_indent:
                body_indent = indent
        return ""

    #: THE GUARD IS EVALUATED, NOT PATTERN-MATCHED, as of 2026-08-21. It used to be
    #: a three-entry map of ``BC? == MIRROR_METALLIC`` fragments to the axis that
    #: had to be folded, and everything else — every ``NEAR_a``, every ``FAR_a``,
    #: every composite — fell through to "unclassified: assume live". That default
    #: is what let the three ``if NEAR_a and (FAR_b and FAR_c):`` blocks ship with
    #: no case entering them: a hand-maintained classifier cannot be short of the
    #: kernel's vocabulary if it does not have one. The condition is now EXECUTED
    #: against the constexprs the scored case really compiles, read off a real grid
    #: by ``gate.case_constexprs``.
    def enterable(guard: str, environment) -> bool:
        if not guard:
            return True                      # unguarded: live wherever the case is
        condition = guard.strip()
        assert condition.startswith("if ") and condition.endswith(":"), condition
        return bool(eval(condition[3:-1].strip(), {"__builtins__": {}},  # noqa: S307
                         dict(environment)))

    #: AT LEAST ONE, NOT ALL. Most rewrites here are a plain ``source.replace``
    #: applied to all three components, so their changed lines sit under all three
    #: axis guards at once; a grid folding one axis enters exactly one of them and
    #: the mutation is perfectly live. Demanding EVERY guard be enterable flagged
    #: seven live mutations on 2026-08-20 and would have forced a case table that
    #: folds all three axes simultaneously — which the product refuses anyway. The
    #: property that actually matters is that the mutated lines can execute AT ALL.
    offenders = []
    environments = {}
    for name, _why, _expectation, rewrite in gate.mutation_table():
        mutated, hits = rewrite(source)
        assert hits, f"{name} is unarmed; the sibling test should have caught this"
        changed = changed_lines(lines, mutated.splitlines())
        if not changed:
            continue
        guards = {enclosing_guard(n) for n in changed}
        index, case = gate.mutation_case_for(name)
        if index not in environments:
            environments[index] = gate.case_constexprs(case)
        environment = environments[index]
        if not any(enterable(guard, environment) for guard in guards):
            offenders.append(
                f"{name} rewrites lines under {sorted(guards)}, and it is scored "
                f"on {case[0]!r}, which compiles NEAR="
                f"{(environment['NEAR_X'], environment['NEAR_Y'], environment['NEAR_Z'])} "
                f"FAR={(environment['FAR_X'], environment['FAR_Y'], environment['FAR_Z'])} "
                f"ZM={(environment['ZM_X'], environment['ZM_Y'], environment['ZM_Z'])} "
                f"— NOT ONE of those branches is entered, so the leg launches a "
                f"kernel whose mutated lines cannot execute and reports a verdict "
                f"about the case table rather than about the kernel")
    assert not offenders, offenders


def test_every_constexpr_branch_the_kernel_SHIPS_is_entered_by_some_case():
    """SHIPPED CODE NO DEVICE LEG EXECUTES — the layer under both tests above.

    An adversarial verifier refuted the 2026-08-20 round on exactly this: the
    kernel shipped the three-axis folded-PERIODIC blocks — the three
    ``if NEAR_a and (FAR_b and FAR_c):`` triples — and every ``mirrors`` value in
    the released artifact carried at most TWO folded axes, so no leg ever compiled
    them. Byte identity over ten steps on ten cases is silent about a branch none
    of the ten contains.

    THE GENERAL FORM WAS WORSE THAN THE INSTANCE, and only counting showed it: 20
    of the kernel's 45 live constexpr guards were unreachable on that table, not 3
    — the whole z half of the carry, the z wall clear, and the ghost-rule wrap on
    an unfolded periodic Y. Three cases close all twenty.

    The one deliberate exception is ``BACKWARD``, whose ``if`` arm is dead in every
    admitted configuration by design: the kernel carries it so its curl half stays
    a VERBATIM copy of ``symmetry.pml_curl_step_folded``. The gate names it in
    ``DELIBERATELY_UNREACHABLE`` and this test reads that name rather than
    re-deciding it.
    """
    gate = load_gate()
    row = gate.branch_reachability_leg()
    assert row["passed"], row["findings"]
    assert row["guards"] >= 45, (
        f"only {row['guards']} live constexpr guards were found in the kernel; the "
        f"scan shape changed and this test would now enforce almost nothing")
    assert row["deliberately_unreachable"] == ["BACKWARD"], row


def test_the_case_table_reaches_the_deepest_composition_the_kernel_can_emit():
    """Seven ghost cells from one source lane, which needs three folded axes.

    ``(1 + near_m) * 2 ** (far axes other than m) - 1`` is 1 under a single
    metallic fold, 3 under two folds, and 7 under three folded PERIODIC ones. The
    kernel emits a ``_carry_ghost`` call per destination, so a table topping out at
    3 leaves the deepest three calls uncompiled — which is what the whole
    2026-08-20 table did.
    """
    gate = load_gate()
    row = gate.branch_reachability_leg()
    assert row["deepest_ghost_destinations"] == 7, row["cases"]
    deepest = [entry["case"] for entry in row["cases"]
               if max(entry["ghost_destinations"]) == 7]
    assert deepest, row["cases"]
    for name in deepest:
        case = next(entry for entry in gate.CASES if entry[0] == name)
        assert case[5].get("expect_ghost_destinations") == 7, (
            f"{name} reaches depth 7 but does not DECLARE it, so verdict_of "
            f"applies no floor and a later edit could silently shallow it")


def test_the_case_table_carries_a_leg_at_a_claimed_corpus_rows_own_shape():
    """The row the census counts, at its own grid shape rather than a lookalike.

    ``tests:TestModeDecomposition.test_grating_3d`` is cell (1.1, 0.8, 8.5) at
    resolution 25 with two even mirror planes, and it lifts to grid shape
    [16, 12, 213] (``results/predicate_coverage_2026-08-16_wired/
    tests_param.jsonl``). The Triton census counts it among this family's admitted
    rows, and until 2026-08-21 every leg of this gate was 2-D — the Metal twin
    covered 3-D two-folds and this board covered none.
    """
    gate = load_gate()
    row = gate.branch_reachability_leg()
    shapes = {entry["case"]: entry["shape"] for entry in row["cases"]}
    assert shapes.get("grating_3d_two_folds") == [16, 12, 213], shapes
    three_d = [name for name, shape in shapes.items() if min(shape) > 1]
    assert len(three_d) >= 3, three_d


def test_the_ownership_fix_the_device_run_found_is_re_planted_by_a_mutation():
    """A defect a gate once found and no mutation re-plants is a fix nothing guards.

    The 2026-08-20 device run found the carry masks were not ANDed with the
    component's ownership mask — one word of ``Hx`` and one of ``Hy``, 1-2 ULP, on
    the two-folded-axis rows alone. The fix landed; nothing re-planted it, and
    ``m4``, the only mutation near it, is scored on a fold with no far axes at all.
    """
    gate = load_gate()
    source = gate._shipped_text("folded_fused_curl_constitutive_B")
    table = {name: rewrite for name, _why, _expectation, rewrite
             in gate.mutation_table()}
    name = "m17_carry_masks_not_anded_with_ownership"
    assert name in table, sorted(table)
    mutated, hits = table[name](source)
    # TWENTY-ONE: every destination one source lane can own, across three
    # components. A rewrite that reached fewer would leave part of the fix
    # unguarded while reporting a catch.
    assert hits == 21, hits
    for mask in ("own0 & near_i & far_j & far_k)", "own1 & near_j & far_i & far_k)",
                 "own2 & near_k & far_i & far_j)"):
        assert mask in source, mask
        assert mask not in mutated, mask
    # AND IT MUST BE SCORED WHERE THE FAR MASKS EXIST. On a fold with no far axis
    # the `own?` narrowing is empty and the rewrite is inert.
    _index, case = gate.mutation_case_for(name)
    environment = gate.case_constexprs(case)
    assert any((environment["FAR_X"], environment["FAR_Y"], environment["FAR_Z"])), (
        f"{name} is scored on {case[0]!r}, which has no folded PERIODIC axis: the "
        f"carry masks it strips are not emitted there")


def test_the_gate_binds_the_seam_passes_the_driver_actually_calls():
    """The five call sites, read off ``driver.py`` rather than remembered."""
    gate = load_gate()
    driver_source = (PACKAGE_DIR.parent / "driver.py").read_text(encoding="utf-8")
    for name in gate.SEAM_PASSES:
        assert f"{name}(self.fields" in driver_source, name
    assert gate.SEAM_PASSES == ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                                "fill_folded_far_ghosts_B", "update_H")
