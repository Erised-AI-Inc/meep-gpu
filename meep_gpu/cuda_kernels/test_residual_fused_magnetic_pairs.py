"""The five RESIDUAL magnetic-seam welds and the shared cf fill carry -- the laptop half.

WHAT THIS FILE PINS. The device verdicts live in
``parity/meep_gpu/gate_cuda_fused_complex_pairs.py``'s five residual families; what a
laptop can pin is the half whose failure mode is a SILENT WRONG ANSWER before any
kernel runs:

* the shared cf carry's CLOSED FORMS -- the ownership rule, the parity chain's
  driver order (near innermost, far ascending; the order MOVES BYTES under complex
  storage, measured), and the destination-coefficient rule;
* the EMITTED TEXT's structure -- 21 ghost blocks on each fill-carrying weld (7
  destinations per component, the Triton sibling's own count), B bound exactly
  once, the sub-lattice rename applied, the ownership guard between the ``fu``
  store and the field load, the wall clear guarded on ownership;
* the SPLICE ANCHORS as welds -- a mutated certified line fails the emit rather
  than emitting a kernel that is quietly missing a mask or a seam;
* the five PREDICATES on real engine objects behind the NumPy-wearing-CuPy's-name
  stand-in -- each product admits its own cell's shape and refuses, BY NAME, the
  configurations its launch cannot serve (the fold on the three no-fill welds;
  an undeclared source set everywhere; the sibling cells that belong to other
  products).
"""

import numpy
import pytest

from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import (bfast_fused_magnetic_pair, complex_beta_fused_magnetic_pair,
               complex_emitter, complex_fill_carry,
               complex_folded_fused_magnetic_pair,
               cylindrical_real_fused_magnetic_pair, fused_pairs,
               special_kz_fused_magnetic_pair)
from .registry import LICENSE_COMPLEX
from .test_fused_pairs import _NumpyWearingCupysName

#: The licence the two complex welds require; the same stub the complex electric
#: pair tests ask with, and never defaulted.
LICENCE = {"arm": "FMA_V1", "expansion": 1, "basis": "measured",
           "refusals": [], "policy_resolved": "keep"}
POLICY = "keep"


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def build_folded_complex(xp, *, symmetry=(Mirror("Y", 1),),
                         boundaries=("metallic", "periodic", "periodic"),
                         cell=(8.0, 9.0, 10.0), complex_storage=True, beta=0.0,
                         dimensions=3):
    """A frozen triple the folded-complex weld admits (or a variant it refuses)."""
    kwargs = {}
    if beta:
        kwargs["beta"] = beta
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=tuple(symmetry), xp=xp, dimensions=dimensions, **kwargs)
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2)
                      for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


# ---------------------------------------------------------------------------
# 1. THE SHARED CARRY'S CLOSED FORMS
# ---------------------------------------------------------------------------

def test_the_parity_chain_is_the_drivers_order_and_never_a_product():
    """Near INNERMOST (its whole pass precedes the far one), then far ascending.
    The order moves bytes under complex storage -- measured on the GPU host, 4 of
    8216 words apart between (-1)x((-1)xz) and (+1)xz -- so this is arithmetic,
    not bookkeeping."""
    chain = complex_fill_carry.parity_chain((0,), (1, 2), True)
    assert chain == ((0, "near"), (1, "far"), (2, "far"))
    chain = complex_fill_carry.parity_chain((1,), (0, 2), False)
    assert chain == ((0, "far"), (2, "far"))
    with pytest.raises(AssertionError, match="ascending"):
        complex_fill_carry.parity_chain((0,), (2, 1), False)


def test_each_component_owns_seven_destinations_and_the_yee_tables_partition():
    """Subsets of the two far axes (4) times near-or-not (2), minus the thread's
    own cell = 7 per component -- 21 blocks per weld, the Triton sibling's own
    count. Near and far sets are exact complements on the B family."""
    for target in range(3):
        near = complex_fill_carry.near_fill_axes(target)
        far = complex_fill_carry.far_fill_axes(target)
        assert near == (target,)
        assert len(far) == 2 and target not in far
        destinations = complex_fill_carry.carried_destinations(near, far)
        assert len(destinations) == 7


def test_the_ownership_flags_test_exactly_the_imaged_planes():
    text = complex_fill_carry.ownership_declarations()
    assert "int own_0 = !((near_x && i == 0) || (reflect_y >= 0 && j == ny - 1) "\
           "|| (reflect_z >= 0 && k == nz - 1));" in text
    assert "int own_1 = !((near_y && j == 0) || (reflect_x >= 0 && i == nx - 1) "\
           "|| (reflect_z >= 0 && k == nz - 1));" in text
    assert "int own_2 = !((near_z && k == 0) || (reflect_x >= 0 && i == nx - 1) "\
           "|| (reflect_y >= 0 && j == ny - 1));" in text


def test_the_ghost_value_is_a_chained_certified_multiply_and_never_a_scale():
    """Every parity application is ONE mul_coefficient_left -- the coefficient is
    np.multiply's left operand -- and the far weight is spelled (phase * -1.0f),
    this directory's complex-family negation convention. A plain two-word scale
    was measured byte-WRONG on signed zeros."""
    for target in range(3):
        blocks = "\n".join(complex_fill_carry.fill_carry_blocks(target))
        assert "mul_coefficient_left(" in blocks
        assert blocks.count("* -1.0f)") >= 3   # every far application
        # the corner block chains near FIRST, then the ascending far axes
        assert blocks.count("_v = mul_coefficient_left(") >= 7


def test_the_near_ghost_takes_the_destinations_coefficient_entry():
    blocks = "\n".join(complex_fill_carry.fill_carry_blocks(0))
    assert "kps_x[0], kms_int_x[0]);" in blocks     # near: stored index 0
    assert "kps_x[i], kms_int_x[i]);" in blocks     # far: this thread's own


def test_fills_plan_raises_on_a_far_row_with_no_declared_phase():
    """A launcher handed a grid the predicate would refuse must not quietly build
    a plan for it: a folded PERIODIC axis whose plane declared no parity has a
    reflect row and no weight, and the failure mode is a plane of wrong values
    rather than an exception."""
    class _PhaselessGrid:
        def has_symmetry(self): return True
        def is_mirrored(self, axis): return axis == 0
        def is_metallic(self, axis): return False
        has_metallic = False
        def mirror_phase(self, axis): return None
        def stored_cells(self, axis): return 8
        def owned_cells(self, axis): return 7 if axis == 0 else 8
        shape_full = (13, 8, 8)
    with pytest.raises(ValueError, match="no mirror phase"):
        complex_fill_carry.fills_plan(_PhaselessGrid())


# ---------------------------------------------------------------------------
# 2. THE EMITTED TEXT
# ---------------------------------------------------------------------------

EMITTERS = {
    "folded": lambda: complex_folded_fused_magnetic_pair
        .complex_folded_fused_magnetic_pair_source("FMA_V1"),
    "beta": lambda: complex_beta_fused_magnetic_pair
        .complex_beta_fused_magnetic_pair_source("FMA_V1"),
}


@pytest.mark.parametrize("name", sorted(EMITTERS))
def test_the_fill_carrying_welds_emit_21_ghost_blocks_and_bind_b_once(name):
    source = EMITTERS[name]()
    # 21 ghost blocks: each stores B at a computed ghost index.
    assert source.count("cf_store(f0, g0_") == 7
    assert source.count("cf_store(f1, g1_") == 7
    assert source.count("cf_store(f2, g2_") == 7
    # B bound exactly once: one writable f0 in the signature, no const g-side
    # binding of it, and the constitutive half reads registers.
    assert source.count("float* __restrict__ f0,") == 1
    assert "cf s0 = b0;" in source
    # the sub-lattice rename reached all three certified statements and the
    # ghost blocks beside them.
    assert source.count("kms_int_x[") >= 2
    assert "constitutive_apply(f0" not in source
    # the ownership guard sits between the fu store and the field load.
    prelude = source.split('extern "C" __global__', 1)[0]
    guard = prelude.index("if (!owned) return cf_zero();")
    assert prelude.index("cf_store(fu, idx, fu_new);") < guard
    assert guard < prelude.index("cf a = mul_field_left(cf_load(f, idx), kms_u);")
    # the wall clear is guarded on ownership and the register is cleared beside
    # the store.
    assert "if (own_0 && wall_x && i == 0) { b0 = cf_zero(); cf_store(f0, idx, b0); }" \
        in source


def test_the_beta_weld_carries_the_beta_statements_and_the_folded_one_does_not():
    folded = EMITTERS["folded"]()
    beta = EMITTERS["beta"]()
    assert beta.count("mul_imag_coefficient_left(") >= 2
    assert "mul_imag_coefficient_left(" not in folded
    assert "float bp_re, float bp_im, float bm_re, float bm_im" in beta
    assert "bp_re" not in folded


def test_the_corner_block_chains_near_before_far():
    """The two-parity corner: the near multiply must be emitted BEFORE the far
    one, because the near pass runs to completion first and the far pass reads
    the plane it wrote."""
    source = EMITTERS["folded"]()
    corner = source.split("int g0_yn_i", 1)[1].split("}", 1)[0]
    near = corner.index("mul_coefficient_left(phase_x, b0)")
    far = corner.index("mul_coefficient_left((phase_y * -1.0f), g0_yn_v)")
    assert near < far


def test_a_moved_certified_anchor_fails_the_emit_rather_than_emitting(monkeypatch):
    """The splice is a weld: a certified line that stops matching raises, never
    emits a kernel quietly missing the guard."""
    monkeypatch.setattr(
        complex_emitter, "_TAIL",
        complex_emitter._TAIL.replace(
            "cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), fprev), "
            "sinv_u));", "cf_store(f, idx, fu_new);"))
    with pytest.raises(AssertionError):
        EMITTERS["folded"]()
    with pytest.raises(AssertionError):
        EMITTERS["beta"]()


def test_the_real_welds_refuse_the_emit_by_name_on_a_cupy_free_host():
    """Their certified halves import CuPy at module scope, so on this host the
    emitters refuse with the named reason -- the same stance the shipped real
    pair takes -- while the predicates below still answer."""
    if cylindrical_real_fused_magnetic_pair.cylindrical_kernels is not None:
        pytest.skip("CuPy present: the emitters run and the device gate owns them")
    for module, emit in (
            (cylindrical_real_fused_magnetic_pair,
             cylindrical_real_fused_magnetic_pair
             .cylindrical_real_fused_magnetic_pair_source),
            (special_kz_fused_magnetic_pair,
             special_kz_fused_magnetic_pair.special_kz_fused_magnetic_pair_source),
            (bfast_fused_magnetic_pair,
             bfast_fused_magnetic_pair.bfast_fused_magnetic_pair_source)):
        with pytest.raises(RuntimeError, match="not importable"):
            emit()


# ---------------------------------------------------------------------------
# 3. THE PREDICATES, ON REAL ENGINE OBJECTS
# ---------------------------------------------------------------------------

def test_the_folded_weld_admits_its_cell_and_names_what_it_refuses(xp):
    fields, layer, grid = build_folded_complex(xp)
    covered, reason = (complex_folded_fused_magnetic_pair
                       .covers_complex_folded_fused_magnetic_pair(
                           fields, layer, grid, (), LICENCE, POLICY))
    assert covered, reason

    # an UNFOLDED complex run belongs to the plain complex pair.
    fields2, layer2, grid2 = build_folded_complex(xp, symmetry=())
    covered, reason = (complex_folded_fused_magnetic_pair
                       .covers_complex_folded_fused_magnetic_pair(
                           fields2, layer2, grid2, (), LICENCE, POLICY))
    assert not covered
    assert "no mirror fold is active" in reason

    # an undeclared source set is a refusal about the caller, stated as one.
    covered, reason = (complex_folded_fused_magnetic_pair
                       .covers_complex_folded_fused_magnetic_pair(
                           fields, layer, grid, None, LICENCE, POLICY))
    assert not covered and "source set was not declared" in reason

    # real storage is the real welds' and the real pair's, never this one's.
    fields3, layer3, grid3 = build_folded_complex(xp, complex_storage=False)
    covered, reason = (complex_folded_fused_magnetic_pair
                       .covers_complex_folded_fused_magnetic_pair(
                           fields3, layer3, grid3, (), LICENCE, POLICY))
    assert not covered, "a real-storage run must be refused"


def test_the_folded_welds_fills_plan_names_the_fold(xp):
    _fields, _layer, grid = build_folded_complex(xp)
    plan = complex_folded_fused_magnetic_pair \
        .complex_folded_fused_magnetic_pair_fills(grid)
    axis = next(a for a in range(3) if grid.is_mirrored(a))
    assert plan["near"][axis] == 1
    assert plan["phase"][axis] in (1.0, -1.0)
    assert sum(plan["near"]) == 1


def test_the_beta_weld_requires_beta_and_serves_both_fold_sides(xp):
    # UNFOLDED beta (the test_special_kz shape: all-periodic, in-plane k).
    fields, layer, grid = build_folded_complex(
        xp, symmetry=(), boundaries=("periodic",) * 3, cell=(12.0, 8.0, 0.0),
        beta=-0.39, dimensions=2)
    covered, reason = (complex_beta_fused_magnetic_pair
                       .covers_complex_beta_fused_magnetic_pair(
                           fields, layer, grid, (), LICENCE, POLICY))
    assert covered, reason

    # FOLDED beta (the TestEigCoeffs shape).
    fields2, layer2, grid2 = build_folded_complex(
        xp, symmetry=(Mirror("Y", 1),),
        boundaries=("metallic", "periodic", "periodic"),
        cell=(8.0, 9.0, 0.0), beta=-0.685, dimensions=2)
    covered, reason = (complex_beta_fused_magnetic_pair
                       .covers_complex_beta_fused_magnetic_pair(
                           fields2, layer2, grid2, (), LICENCE, POLICY))
    assert covered, reason

    # beta = 0 is the folded pair's cell, refused by the curl half by name.
    fields3, layer3, grid3 = build_folded_complex(
        xp, symmetry=(Mirror("Y", 1),), cell=(8.0, 9.0, 0.0), dimensions=2)
    covered, reason = (complex_beta_fused_magnetic_pair
                       .covers_complex_beta_fused_magnetic_pair(
                           fields3, layer3, grid3, (), LICENCE, POLICY))
    assert not covered
    assert "curl half" in reason


def test_the_cylindrical_real_weld_admits_m0_and_refuses_complex_storage(xp):
    grid = Grid(resolution=1.0, cell_size=(9.0, 0.0, 11.0), cylindrical=True,
                m=0, boundaries={"z": "metallic"}, courant=0.5, xp=xp)
    layer = PML(grid=grid, thickness={"x": (0, 2), "z": 2})
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    covered, reason = (cylindrical_real_fused_magnetic_pair
                       .covers_cylindrical_real_fused_magnetic_pair(
                           fields, layer, grid, ()))
    assert covered, reason

    fields2 = Fields(grid=grid, force_complex_fields=True)
    fields2.enable_field_storage()
    fields2.enable_pml_storage()
    covered, reason = (cylindrical_real_fused_magnetic_pair
                       .covers_cylindrical_real_fused_magnetic_pair(
                           fields2, layer, grid, ()))
    assert not covered, "complex storage is the complex Dcyl weld's cell"

    covered, reason = (cylindrical_real_fused_magnetic_pair
                       .covers_cylindrical_real_fused_magnetic_pair(
                           fields, layer, grid, None))
    assert not covered and "source set was not declared" in reason


def test_the_special_kz_weld_requires_real_beta_and_serves_both_fold_sides(xp):
    """The eigsrc_kz_1 shape (folded real beta) is ADMITTED since 2026-09-02:
    the fold refusal that stood here rested on the deposit premise the residue
    audit overturned, and the real pair's fill carry is now spliced in."""
    grid = Grid(resolution=1.0, cell_size=(12.0, 8.0, 0.0),
                boundaries=("periodic",) * 3, xp=xp, dimensions=2, beta=0.332)
    layer = PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    covered, reason = (special_kz_fused_magnetic_pair
                       .covers_special_kz_fused_magnetic_pair(
                           fields, layer, grid, ()))
    assert covered, reason

    # THE FOLDED SIDE (the eigsrc_kz_1_real_imag shape), now served through the
    # ported carry -- and it must be served by special_kz ALONE: the real pair
    # refuses beta, every complex product refuses real storage.
    grid2 = Grid(resolution=1.0, cell_size=(8.0, 9.0, 0.0),
                 boundaries=("metallic", "periodic", "periodic"),
                 symmetry=(Mirror("Y", 1),), xp=xp, dimensions=2, beta=0.2)
    layer2 = PML(grid=grid2, thickness=tuple(
        (0, 2) if grid2.is_mirrored(axis) else
        ((2, 2) if grid2.shape[axis] >= 6 else (0, 0)) for axis in range(3)))
    fields2 = Fields(grid=grid2)
    fields2.enable_field_storage()
    fields2.enable_pml_storage()
    curl_ok, curl_reason = special_kz_fused_magnetic_pair._special_kz \
        .covers_special_kz_curl(fields2, layer2, grid2, "step_B")
    assert curl_ok, (f"the fixture must be one the CERTIFIED curl admits, or the "
                     f"carry below is unreachable: {curl_reason}")
    covered, reason = (special_kz_fused_magnetic_pair
                       .covers_special_kz_fused_magnetic_pair(
                           fields2, layer2, grid2, ()))
    assert covered, reason
    from . import fused_magnetic_pair as real_pair
    covered, _reason = real_pair.covers_fused_magnetic_pair(
        fields2, layer2, grid2, ())
    assert not covered, "the real pair must refuse a beta run"

    # an undeclared source set is a refusal about the caller, stated as one.
    covered, reason = (special_kz_fused_magnetic_pair
                       .covers_special_kz_fused_magnetic_pair(
                           fields2, layer2, grid2, None))
    assert not covered and "source set was not declared" in reason


@pytest.mark.requires_resource("cupy")
def test_the_special_kz_weld_emits_the_real_pairs_carry_with_the_beta_lines():
    """The emitted five-pass source is the real pair's carry over the beta curl:
    21 ghost blocks, B bound once, the ownership guard between the fu store and
    the field load, the wall clear guarded on own_*, and BOTH beta statements
    riding through the capture. Emitters need the CuPy-bound halves; on a host
    without them the refusal is by name and the device gate owns the text."""
    if special_kz_fused_magnetic_pair.constitutive_kernels is None:
        pytest.skip("CuPy absent: the emitter refuses by name; the device gate "
                    "owns the emitted text")
    source = special_kz_fused_magnetic_pair.special_kz_fused_magnetic_pair_source()
    assert source.count("beta_plus") >= 2 and source.count("beta_minus") >= 2
    for axis in "xyz":
        assert source.count(f"B{axis}[g{axis}_") >= 1
    assert source.count("float* __restrict__ Bx,") == 1
    assert "if (!owned) return 0.0f;" in source
    assert "if (own_x && wall_x && i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }" \
        in source
    assert "near_x, " in source or "int near_x" in source


def test_the_bfast_weld_requires_bfast_and_refuses_the_fold_by_name(xp):
    grid = Grid(resolution=1.0, cell_size=(10.0, 8.0, 0.0),
                boundaries=("periodic",) * 3, xp=xp, dimensions=2,
                courant=0.5, bfast_scaled_k=(0.35, 0.0, 0.0))
    assert grid.bfast_active
    layer = PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    covered, reason = (bfast_fused_magnetic_pair
                       .covers_bfast_fused_magnetic_pair(fields, layer, grid, ()))
    assert covered, reason

    grid2 = Grid(resolution=1.0, cell_size=(8.0, 9.0, 0.0),
                 boundaries=("metallic", "periodic", "periodic"),
                 symmetry=(Mirror("Y", 1),), xp=xp, dimensions=2,
                 courant=0.5, bfast_scaled_k=(0.35, 0.0, 0.0))
    layer2 = PML(grid=grid2, thickness=tuple(
        (0, 2) if grid2.is_mirrored(axis) else
        ((2, 2) if grid2.shape[axis] >= 6 else (0, 0)) for axis in range(3)))
    fields2 = Fields(grid=grid2)
    fields2.enable_field_storage()
    fields2.enable_pml_storage()
    # THE CERTIFIED BFAST CURL REFUSES A FOLD ITSELF ("the 'folded ghost
    # values are dead' half of that argument does not transfer and has not been
    # measured"), so on this family the weld's own fold clause is UNREACHABLE
    # today -- and it is still written, for the complex pair's stated reason:
    # the guarantee that the two fills do nothing inside this seam may not rest
    # on a clause in another module that a future device verdict could licence
    # away. What a laptop can pin is that a folded BFAST run is refused, with
    # the curl half saying so first.
    covered, reason = (bfast_fused_magnetic_pair
                       .covers_bfast_fused_magnetic_pair(fields2, layer2, grid2, ()))
    assert not covered
    assert "curl half" in reason and "mirror" in reason.lower()
    assert "mirror plane is active" in \
        __import__("inspect").getsource(
            bfast_fused_magnetic_pair.covers_bfast_fused_magnetic_pair), (
        "the weld's own fold clause has been dropped; it is the stated backstop "
        "should the certified curl ever admit a fold")


# ---------------------------------------------------------------------------
# 4. THE WIRING
# ---------------------------------------------------------------------------

RESIDUAL = {
    "cuda_complex_folded_fused_magnetic_pair": complex_folded_fused_magnetic_pair,
    "cuda_complex_beta_fused_magnetic_pair": complex_beta_fused_magnetic_pair,
    "cuda_cylindrical_real_fused_magnetic_pair": cylindrical_real_fused_magnetic_pair,
    "cuda_special_kz_fused_magnetic_pair": special_kz_fused_magnetic_pair,
    "cuda_bfast_fused_magnetic_pair": bfast_fused_magnetic_pair,
}


@pytest.mark.parametrize("family", sorted(RESIDUAL))
def test_each_product_row_names_its_own_module_and_seam(family):
    module = RESIDUAL[family]
    row = fused_pairs.FUSED_PRODUCTS[family]
    assert row["curl_slot"] == "step_B" == module.SLOT
    assert module.FAMILY == family
    assert family in fused_pairs.FUSED_PAIR_ARMS
    assert module.REPLACES[0] == "step_B" and module.REPLACES[-1] == "update_H"
    # the three fill-carrying welds declare all five passes; the two no-fill
    # welds declare three and REFUSE the fold in their predicates instead.
    # special_kz joined the carrying set on 2026-09-02 (residue round): its
    # blocked row is folded, and the fold refusal's premise fell with the
    # deposit flag.
    if family in ("cuda_complex_folded_fused_magnetic_pair",
                  "cuda_complex_beta_fused_magnetic_pair",
                  "cuda_special_kz_fused_magnetic_pair"):
        assert set(module.REPLACES) == {
            "step_B", "fill_symmetry_bc_B", "zero_metal_B",
            "fill_folded_far_ghosts_B", "update_H"}
    else:
        assert set(module.REPLACES) == {"step_B", "zero_metal_B", "update_H"}


def test_the_repair_flag_split_is_the_re_measured_census_fact():
    """The flag split is a CENSUS FACT, not a preference -- RE-MEASURED in the
    2026-09-02 residue round. The folded cell's fifth row deposits a plain
    point B Source; the beta and special_kz cells' blocked rows deposit their
    lifted eigenmode SHEETS, which DO publish point indices
    (``_lift_eigenmode_source`` synthesises ``VolumeSource`` equivalent-current
    sheets, and every source class publishes ``_point_ix/_point_iy/_point_iz``
    at setup) -- the "publishes no point index" premise the old False rested on
    was measured false on this tree. The Dcyl-real and BFAST cells' rows are
    electric-only sourced: their magnetic seam is empty on every one, so True
    there would buy zero rows and cost an unmeasured claim."""
    for module in (complex_folded_fused_magnetic_pair,
                   complex_beta_fused_magnetic_pair,
                   special_kz_fused_magnetic_pair):
        assert module.CARRIES_DEPOSIT_REPAIR is True, module.FAMILY
    for module in (cylindrical_real_fused_magnetic_pair,
                   bfast_fused_magnetic_pair):
        assert module.CARRIES_DEPOSIT_REPAIR is False, module.FAMILY


def test_a_sheet_source_publishes_its_deposit_index_and_the_seam_carries_it():
    """The re-measured premise ITSELF, pinned so a regression re-fires here
    rather than resurfacing as a wrong flag: a sheet magnetic ``VolumeSource``
    with an ``amp_func`` (the eigenmode lift's equivalent-current shape)
    returns index ARRAYS from ``deposit_repair._deposit_index``, and
    ``seam_source_reasons(carries_repair=True)`` fires no publish-refusal while
    ``carries_repair=False`` refuses it."""
    from .. import deposit_repair as dr
    from .. import sources as S

    grid = Grid(resolution=20.0, cell_size=(1.6, 1.6, 0.0), dimensions=2,
                courant=0.5)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    env = S.GaussianEnvelope(frequency=1.0, fwidth=1.0)
    sheet = S.VolumeSource(
        grid=grid, component="Bz", center=(0.4, 0.0, 0.0), size=(0.0, 1.5, 0.0),
        envelope=env, amp_func=lambda x, y, z: numpy.exp(1j * 2.0 * y))
    index = dr._deposit_index(sheet)
    assert index is not None
    assert all(numpy.atleast_1d(column).size > 1 for column in index), (
        "the sheet must publish a MULTI-cell footprint; a single point would "
        "not measure the premise")
    carried = dr.seam_source_reasons(
        fields, [sheet], 'B', undeclared="undeclared",
        refusal=lambda i, s: f"in-seam {i}", carries_repair=True)
    assert carried == ()
    refused = dr.seam_source_reasons(
        fields, [sheet], 'B', undeclared="undeclared",
        refusal=lambda i, s: f"in-seam {i}", carries_repair=False)
    assert refused and "in-seam 0" in refused[0]


def test_the_eight_step_b_candidates_partition_on_the_folded_cell(xp):
    """One admitted product per configuration: on the folded-complex fixture,
    every OTHER step_B product must refuse -- two admitters leave the seam
    unfused naming both, so an overlap here costs the very slots this round
    exists to serve."""
    fields, layer, grid = build_folded_complex(xp)
    admitted = []
    from . import (complex_fused_magnetic_pair, cylindrical_fused_magnetic_pair,
                   fused_magnetic_pair)
    candidates = {
        "real": lambda: fused_magnetic_pair.covers_fused_magnetic_pair(
            fields, layer, grid, ()),
        "complex": lambda: complex_fused_magnetic_pair
            .covers_complex_fused_magnetic_pair(
                fields, layer, grid, (), LICENCE, POLICY),
        "cyl_complex": lambda: cylindrical_fused_magnetic_pair
            .covers_cylindrical_fused_magnetic_pair(
                fields, layer, grid, (), LICENCE, POLICY),
        "folded": lambda: complex_folded_fused_magnetic_pair
            .covers_complex_folded_fused_magnetic_pair(
                fields, layer, grid, (), LICENCE, POLICY),
        "beta": lambda: complex_beta_fused_magnetic_pair
            .covers_complex_beta_fused_magnetic_pair(
                fields, layer, grid, (), LICENCE, POLICY),
        "cyl_real": lambda: cylindrical_real_fused_magnetic_pair
            .covers_cylindrical_real_fused_magnetic_pair(fields, layer, grid, ()),
        "special_kz": lambda: special_kz_fused_magnetic_pair
            .covers_special_kz_fused_magnetic_pair(fields, layer, grid, ()),
        "bfast": lambda: bfast_fused_magnetic_pair
            .covers_bfast_fused_magnetic_pair(fields, layer, grid, ()),
    }
    for name, ask in candidates.items():
        covered, _reason = ask()
        if covered:
            admitted.append(name)
    assert admitted == ["folded"], (
        f"the folded fixture must be admitted by the folded weld ALONE, "
        f"got {admitted}")
