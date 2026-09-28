"""The five ELECTRIC-TWIN welds, the D-side complex carry, and the complex E->P pair -- the laptop half.

WHAT THIS FILE PINS. The device verdicts live in
``parity/meep_gpu/gate_cuda_fused_complex_pairs.py``'s electric families and the
polarization gate; what a laptop can pin is the half whose failure mode is a
SILENT WRONG ANSWER before any kernel runs:

* the shared D-side cf carry's CLOSED FORMS -- the D-family near/far complement
  (near = the TWO shift-0 axes, far = the component's own), the pre-clear/clear/
  far composition (the real electric pair's measured ordering), and the
  near-ascending chain order (stepping's own "filled in X, Y, Z order", which
  MOVES BYTES under complex storage);
* the EMITTED TEXT's structure -- 21 ghost blocks on each fill-carrying complex
  twin, D bound exactly once, the named off-diagonal wall flags, the ownership
  guard between the ``fu`` store and the field load, the pre-clear registers;
* the SPLICE ANCHORS as welds -- a mutated certified line fails the emit;
* the predicates on real engine objects: each twin admits its own cell's shape
  WITH ITS SOURCES DECLARED and refuses, BY NAME, the configurations its launch
  cannot serve -- and ``step_D``'s nine candidates admit AT MOST ONE product per
  configuration, because two admitters leave the seam unfused naming both;
* the complex no-absorber E->P pair's seam register and its storage partition
  against the two real E->P products.
"""

import numpy
import pytest

from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import (bfast_fused_electric_pair, complex_beta_fused_electric_pair,
               complex_electric_fill_carry, complex_emitter,
               complex_folded_fused_electric_pair,
               complex_no_pml_fused_polarization_pair,
               cylindrical_real_fused_electric_pair, fused_pairs,
               special_kz_fused_electric_pair)
from .test_fused_pairs import _NumpyWearingCupysName
from .test_residual_fused_magnetic_pairs import (LICENCE, POLICY,
                                                 build_folded_complex)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


# ---------------------------------------------------------------------------
# 1. THE SHARED D-SIDE CARRY'S CLOSED FORMS
# ---------------------------------------------------------------------------

def test_the_d_family_near_and_far_sets_are_the_b_familys_complement():
    """near = the TWO shift-0 axes, far = the component's own axis -- the exact
    inversion of the B family's, read off IYEE_SHIFTS rather than hand-typed."""
    for target in range(3):
        near = complex_electric_fill_carry.near_fill_axes(target)
        far = complex_electric_fill_carry.far_fill_axes(target)
        assert far == (target,)
        assert len(near) == 2 and target not in near
        destinations = complex_electric_fill_carry.carried_destinations(near, far)
        assert len(destinations) == 7


def test_the_ownership_flags_test_exactly_the_imaged_planes():
    text = complex_electric_fill_carry.ownership_declarations()
    assert "int own_0 = !((near_y && j == 0) || (near_z && k == 0) || "\
           "(reflect_x >= 0 && i == nx - 1));" in text
    assert "int own_1 = !((near_x && i == 0) || (near_z && k == 0) || "\
           "(reflect_y >= 0 && j == ny - 1));" in text
    assert "int own_2 = !((near_x && i == 0) || (near_y && j == 0) || "\
           "(reflect_z >= 0 && k == nz - 1));" in text


def test_the_corner_chain_is_near_ascending_then_clear_then_far():
    """The order IS the arithmetic under complex storage: near multiplies
    ascending over the PRE-clear pair (stepping's own X, Y, Z order), the wall
    clear between the two products (the real electric pair's measured
    ordering), the far multiply last -- pinned on the FULL corner (far x, near
    y AND z)."""
    blocks = "\n".join(complex_electric_fill_carry.fill_carry_blocks(0))
    corner = blocks.split("int g0_xnyz_i", 1)[1].split("cf_store", 1)[0]
    first = corner.index("mul_coefficient_left(phase_y, pre0)")
    second = corner.index("mul_coefficient_left(phase_z, g0_xnyz_v)")
    clear = corner.index("if (clr_0) g0_xnyz_v = cf_zero();")
    far = corner.index("mul_coefficient_left((phase_x * -1.0f), g0_xnyz_v)")
    assert first < second < clear < far


def test_a_pure_far_ghost_reads_the_post_clear_register():
    blocks = "\n".join(complex_electric_fill_carry.fill_carry_blocks(1))
    far_only = blocks.split("int g1_y_i", 1)[1].split("}", 1)[0]
    assert "cf g1_y_v = d1;" in far_only
    assert "pre1" not in far_only
    assert "mul_coefficient_left((phase_y * -1.0f), g1_y_v)" in far_only


def test_the_far_ghost_takes_the_destinations_coefficient_entry():
    blocks = "\n".join(complex_electric_fill_carry.fill_carry_blocks(0))
    assert "kps_x[nx - 1], kms_half_x[nx - 1]);" in blocks   # far: stored n-1
    assert "kps_x[i], kms_half_x[i]);" in blocks             # near: own index


def test_the_wall_clear_is_the_off_diagonal_and_named():
    text = complex_electric_fill_carry.zero_metal_carry()
    assert "if (wall_x && i == 0) { clr_1 = 1; clr_2 = 1; }" in text
    assert "if (wall_y && j == 0) { clr_0 = 1; clr_2 = 1; }" in text
    assert "if (wall_z && k == 0) { clr_0 = 1; clr_1 = 1; }" in text
    for target in range(3):
        assert (f"if (own_{target} && clr_{target}) "
                f"{{ d{target} = cf_zero(); "
                f"cf_store(f{target}, idx, d{target}); }}") in text


def test_fills_plan_raises_on_a_far_row_with_no_declared_phase():
    """A launcher handed a grid the predicate would refuse must not quietly
    build a plan for it -- the B-side carry's own refusal, restated for the D
    plan resolver."""
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
        complex_electric_fill_carry.fills_plan(_PhaselessGrid())


# ---------------------------------------------------------------------------
# 2. THE EMITTED TEXT
# ---------------------------------------------------------------------------

EMITTERS = {
    "folded": lambda: complex_folded_fused_electric_pair
        .complex_folded_fused_electric_pair_source("FMA_V1"),
    "beta": lambda: complex_beta_fused_electric_pair
        .complex_beta_fused_electric_pair_source("FMA_V1"),
}


@pytest.mark.parametrize("name", sorted(EMITTERS))
def test_the_fill_carrying_twins_emit_21_ghost_blocks_and_bind_d_once(name):
    source = EMITTERS[name]()
    for target in range(3):
        assert source.count(f"cf_store(f{target}, g{target}_") == 7
    # D bound exactly once: one writable f0 in the signature, and the
    # constitutive half reads registers.
    assert source.count("float* __restrict__ f0,") == 1
    assert "cf s0 = d0;" in source
    # the sub-lattice rename reached all three certified statements.
    assert source.count("kms_half_x[") >= 2
    assert "constitutive_apply(f0" not in source
    # the ownership guard sits between the fu store and the field load.
    prelude = source.split('extern "C" __global__', 1)[0]
    guard = prelude.index("if (!owned) return cf_zero();")
    assert prelude.index("cf_store(fu, idx, fu_new);") < guard
    assert guard < prelude.index("cf a = mul_field_left(cf_load(f, idx), kms_u);")
    # the pre-clear registers exist and precede the wall carry.
    assert source.index("cf pre0 = d0;") < source.index("int clr_0 = 0;")


def test_the_beta_twin_carries_the_beta_statements_and_the_folded_one_does_not():
    folded = EMITTERS["folded"]()
    beta = EMITTERS["beta"]()
    assert beta.count("mul_imag_coefficient_left(") >= 2
    assert "mul_imag_coefficient_left(" not in folded
    assert "float bp_re, float bp_im, float bm_re, float bm_im" in beta
    assert "bp_re" not in folded


def test_a_moved_certified_anchor_fails_the_emit_rather_than_emitting(monkeypatch):
    monkeypatch.setattr(
        complex_emitter, "_TAIL",
        complex_emitter._TAIL.replace(
            "cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), fprev), "
            "sinv_u));", "cf_store(f, idx, fu_new);"))
    with pytest.raises(AssertionError):
        EMITTERS["folded"]()
    with pytest.raises(AssertionError):
        EMITTERS["beta"]()


def test_the_real_twins_refuse_the_emit_by_name_on_a_cupy_free_host():
    if cylindrical_real_fused_electric_pair.cylindrical_kernels is not None:
        pytest.skip("CuPy present: the emitters run and the device gate owns them")
    for emit in (cylindrical_real_fused_electric_pair
                 .cylindrical_real_fused_electric_pair_source,
                 special_kz_fused_electric_pair
                 .special_kz_fused_electric_pair_source,
                 bfast_fused_electric_pair.bfast_fused_electric_pair_source):
        with pytest.raises(RuntimeError, match="not importable"):
            emit()


def test_the_polarization_pairs_seam_register_replaces_the_drive():
    source = (complex_no_pml_fused_polarization_pair
              .fused_polarization_pair_no_pml_complex_source(
                  "FMA_V1", 0, 2, (True, False)))
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("//"))
    assert code.count("cf ev = mul_field_left(") == 1
    assert "cf_store(f0, idx, ev);" in code
    assert "cf_load(drive" not in code
    # the pole bank is bound once and BOTH halves read through it.
    assert code.count("const float* __restrict__ a0,") == 1
    assert "cf_load(a0, idx)" in code       # the recurrence's P^n load
    assert "minus_poles(g0, a0," in code    # the E half's chain
    # per-pole sigma forms: pole 0 volume, pole 1 uniform.
    assert "sigma_0[idx]" in code and "        sigma_1," in code


def test_the_polarization_pairs_zero_arity_body_is_the_e_half_alone():
    source = (complex_no_pml_fused_polarization_pair
              .fused_polarization_pair_no_pml_complex_source("FMA_V1", 2, 0, ()))
    assert "p_out" not in source
    assert "ade_step" in source  # the prelude still defines it; nothing calls it
    body = source.split('extern "C" __global__', 1)[1]
    assert "ade_step(" not in body.split(") {", 1)[1]


def test_a_drifted_certified_recurrence_fails_the_polarization_splice(monkeypatch):
    from . import complex_no_pml_kernels
    real = complex_no_pml_kernels.kernel_source

    def drifted(key, expansion):
        text = real(key, expansion)
        if key == "update_P":
            text = text.replace("cf_load(p_now, idx), cf_load(p_prev, idx)",
                                "cf_load(p_prev, idx), cf_load(p_now, idx)")
        return text

    monkeypatch.setattr(complex_no_pml_kernels, "kernel_source", drifted)
    with pytest.raises(AssertionError):
        (complex_no_pml_fused_polarization_pair
         .fused_polarization_pair_no_pml_complex_source("FMA_V1", 0, 1, (True,)))


# ---------------------------------------------------------------------------
# 3. THE PREDICATES, ON REAL ENGINE OBJECTS
# ---------------------------------------------------------------------------

def _real_grid(xp, **kwargs):
    grid = Grid(resolution=1.0, xp=xp, **kwargs)
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2) for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


ELECTRIC_CANDIDATES = {
    "real": lambda f, l, g: __import__(
        "meep_gpu.cuda_kernels.fused_electric_pair", fromlist=["x"]
    ).covers_fused_electric_pair(f, l, g, ()),
    "complex": lambda f, l, g: __import__(
        "meep_gpu.cuda_kernels.complex_fused_electric_pair", fromlist=["x"]
    ).covers_complex_fused_electric_pair(f, l, g, (), LICENCE, POLICY),
    "cyl_complex": lambda f, l, g: __import__(
        "meep_gpu.cuda_kernels.cylindrical_fused_electric_pair", fromlist=["x"]
    ).covers_cylindrical_fused_electric_pair(f, l, g, (), LICENCE, POLICY),
    "no_pml_complex": lambda f, l, g: __import__(
        "meep_gpu.cuda_kernels.no_pml_complex_fused_electric_pair", fromlist=["x"]
    ).covers_no_pml_complex_fused_electric_pair(f, l, g, (), LICENCE, POLICY),
    "folded": lambda f, l, g: complex_folded_fused_electric_pair
        .covers_complex_folded_fused_electric_pair(f, l, g, (), LICENCE, POLICY),
    "beta": lambda f, l, g: complex_beta_fused_electric_pair
        .covers_complex_beta_fused_electric_pair(f, l, g, (), LICENCE, POLICY),
    "cyl_real": lambda f, l, g: cylindrical_real_fused_electric_pair
        .covers_cylindrical_real_fused_electric_pair(f, l, g, ()),
    "special_kz": lambda f, l, g: special_kz_fused_electric_pair
        .covers_special_kz_fused_electric_pair(f, l, g, ()),
    "bfast": lambda f, l, g: bfast_fused_electric_pair
        .covers_bfast_fused_electric_pair(f, l, g, ()),
}


def _admitters(fields, layer, grid):
    admitted = []
    for name, ask in ELECTRIC_CANDIDATES.items():
        covered, _reason = ask(fields, layer, grid)
        if covered:
            admitted.append(name)
    return admitted


def test_the_folded_twin_admits_its_cell_alone(xp):
    fields, layer, grid = build_folded_complex(xp)
    assert _admitters(fields, layer, grid) == ["folded"]


def test_the_beta_twin_serves_both_fold_sides_alone(xp):
    # FOLDED beta (the TestEigCoeffs shape).
    fields, layer, grid = build_folded_complex(
        xp, symmetry=(Mirror("Y", 1),),
        boundaries=("metallic", "periodic", "periodic"),
        cell=(8.0, 9.0, 0.0), beta=-0.685, dimensions=2)
    assert _admitters(fields, layer, grid) == ["beta"]
    # UNFOLDED beta (the test_special_kz shape) -- the fills are inert.
    fields2, layer2, grid2 = build_folded_complex(
        xp, symmetry=(), boundaries=("periodic",) * 3, cell=(12.0, 8.0, 0.0),
        beta=-0.39, dimensions=2)
    assert _admitters(fields2, layer2, grid2) == ["beta"]


def test_the_cylindrical_real_twin_admits_m0_alone(xp):
    grid = Grid(resolution=1.0, cell_size=(9.0, 0.0, 11.0), cylindrical=True,
                m=0, boundaries={"z": "metallic"}, courant=0.5, xp=xp)
    layer = PML(grid=grid, thickness={"x": (0, 2), "z": 2})
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    assert _admitters(fields, layer, grid) == ["cyl_real"]
    # complex storage is the complex Dcyl twin's cell.
    fields2 = Fields(grid=grid, force_complex_fields=True)
    fields2.enable_field_storage()
    fields2.enable_pml_storage()
    covered, _reason = (cylindrical_real_fused_electric_pair
                        .covers_cylindrical_real_fused_electric_pair(
                            fields2, layer, grid, ()))
    assert not covered


def test_the_special_kz_twin_serves_both_fold_sides_alone(xp):
    fields, layer, grid = _real_grid(
        xp, cell_size=(12.0, 8.0, 0.0), boundaries=("periodic",) * 3,
        dimensions=2, beta=0.332)
    assert _admitters(fields, layer, grid) == ["special_kz"]
    # THE FOLDED SIDE (the eigsrc_kz_1_real_imag shape), served through the
    # real electric pair's carry.
    fields2, layer2, grid2 = _real_grid(
        xp, cell_size=(8.0, 9.0, 0.0),
        boundaries=("metallic", "periodic", "periodic"),
        symmetry=(Mirror("Y", 1),), dimensions=2, beta=0.2)
    assert _admitters(fields2, layer2, grid2) == ["special_kz"]


def test_the_bfast_twin_admits_its_cell_alone(xp):
    fields, layer, grid = _real_grid(
        xp, cell_size=(10.0, 8.0, 0.0), boundaries=("periodic",) * 3,
        dimensions=2, courant=0.5, bfast_scaled_k=(0.35, 0.0, 0.0))
    assert grid.bfast_active
    assert _admitters(fields, layer, grid) == ["bfast"]


def test_an_undeclared_source_set_is_refused_everywhere(xp):
    fields, layer, grid = build_folded_complex(xp)
    covered, reason = (complex_folded_fused_electric_pair
                       .covers_complex_folded_fused_electric_pair(
                           fields, layer, grid, None, LICENCE, POLICY))
    assert not covered and "source set was not declared" in reason
    fields2, layer2, grid2 = _real_grid(
        xp, cell_size=(12.0, 8.0, 0.0), boundaries=("periodic",) * 3,
        dimensions=2, beta=0.332)
    covered, reason = (special_kz_fused_electric_pair
                       .covers_special_kz_fused_electric_pair(
                           fields2, layer2, grid2, None))
    assert not covered and "source set was not declared" in reason


def test_a_conductive_engine_is_refused_on_the_injection_route(xp):
    """The driver routes electric deposits through the condinv rescale on a
    conductive engine, and the repair has no verdict on that route -- the real
    electric pair's clause, restated per twin and pinned here on one.
    ``has_conductivity`` is a Fields property, so the conductive view is a
    delegating proxy rather than an attribute assignment."""
    fields, layer, grid = _real_grid(
        xp, cell_size=(12.0, 8.0, 0.0), boundaries=("periodic",) * 3,
        dimensions=2, beta=0.332)

    class _ConductiveView:
        def __init__(self, inner):
            self._inner = inner
        has_conductivity = True

        def __getattr__(self, name):
            return getattr(self._inner, name)

    covered, reason = (special_kz_fused_electric_pair
                       .covers_special_kz_fused_electric_pair(
                           _ConductiveView(fields), layer, grid, ()))
    assert not covered
    assert "_inject_electric_through_conductivity" in reason


def test_the_polarization_pair_partitions_on_storage(xp):
    """The complex no-absorber E->P pair and the two REAL E->P products may
    never co-admit: theirs refuse complex64 by name, this one requires it."""
    from .test_dispersive import build
    from . import fused_polarization_pair as real_family

    fields, layer, grid = build(poles=2, pml_on=False)
    covered, reason = (complex_no_pml_fused_polarization_pair
                       .covers_complex_no_pml_fused_polarization_pair(
                           fields, layer, grid, (), LICENCE, POLICY))
    assert not covered, "a real-storage run must be refused"
    real_covered, real_reason = real_family.covers_no_pml_fused_polarization_pair(
        fields, layer, grid)
    assert real_covered, real_reason


# ---------------------------------------------------------------------------
# 4. THE WIRING
# ---------------------------------------------------------------------------

TWINS = {
    "cuda_complex_folded_fused_electric_pair": complex_folded_fused_electric_pair,
    "cuda_complex_beta_fused_electric_pair": complex_beta_fused_electric_pair,
    "cuda_cylindrical_real_fused_electric_pair":
        cylindrical_real_fused_electric_pair,
    "cuda_special_kz_fused_electric_pair": special_kz_fused_electric_pair,
    "cuda_bfast_fused_electric_pair": bfast_fused_electric_pair,
}


@pytest.mark.parametrize("family", sorted(TWINS))
def test_each_twin_row_names_its_own_module_and_seam(family):
    module = TWINS[family]
    row = fused_pairs.FUSED_PRODUCTS[family]
    assert row["curl_slot"] == "step_D" == module.SLOT
    assert module.FAMILY == family
    assert family in fused_pairs.FUSED_PAIR_ARMS
    assert module.REPLACES[0] == "step_D" and module.REPLACES[-1] == "update_E"
    # the three fill-carrying twins declare all five passes; the two no-fill
    # twins declare three and REFUSE the fold in their predicates instead.
    if family in ("cuda_complex_folded_fused_electric_pair",
                  "cuda_complex_beta_fused_electric_pair",
                  "cuda_special_kz_fused_electric_pair"):
        assert set(module.REPLACES) == {
            "step_D", "fill_symmetry_bc_D", "zero_metal_D",
            "fill_folded_far_ghosts_D", "update_E"}
    else:
        assert set(module.REPLACES) == {"step_D", "zero_metal_D", "update_E"}
    # THE FLAG IS THE PRODUCT on every twin: every row of every cell declares
    # an electric deposit inside this seam (measured off the residualwelds
    # census), so False would serve nothing.
    assert module.CARRIES_DEPOSIT_REPAIR is True, module.FAMILY


def test_the_polarization_pair_row_names_its_own_module_and_seam():
    module = complex_no_pml_fused_polarization_pair
    family = "cuda_complex_no_pml_fused_polarization_pair"
    row = fused_pairs.FUSED_PRODUCTS[family]
    assert row["curl_slot"] == "update_E" == module.SLOT
    assert module.FAMILY == family
    assert module.REPLACES == ("update_E", "update_P")
    assert module.CARRIES_DEPOSIT_REPAIR is False
    assert fused_pairs.FUSED_PAIR_SEAMS["update_E"] == ("update_P", None)
