"""The complex beta family runs MEEP's beta pass after the curl, read off its text.

WHAT THIS FILE SETTLES, ON A HOST WITH NO DEVICE. The complex beta curl pair
(``complex_beta_kernels``: the folded complex curl with MEEP's beta pass after the curl of
targets 0 and 1, ``-quarter_turn(c, g)`` part by part through the folded ladder) and its two
fused pairs (``complex_beta_fused_{magnetic,electric}_pair``: all five passes of the seam,
the two mirror fills carried by the shared complex carries) are executed by the complex
lane's interpreter (``test_complex_chunk_forms.ComplexInterpreter``) through their own
launchers, on NumPy wearing CuPy's name, over a seeded full-volume complex state, and
every array of the ``Fields`` is compared word for word (sign of zero included, both parts)
with the array path's own passes on a copy of the same state: unfolded and folded beta
grids, a phase on the unfolded in-plane axis, a lifted division, planted signed zeros,
the run's first ``update_H`` and a mid-run state. Planted defects must change a word, and
the one structural null (the partner read from the connected centre register rather than
the stored word) is measured as a null.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, Tuple

import numpy
import pytest

from .. import part_products, stepping
from ..chunk_map import (ChunkLayout, RegionPlan, chunk_bounds_record, grid_frames,
                         region_backend)
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import complex_beta_kernels, complex_emitter, complex_folded_kernels
from .test_complex_chunk_forms import (InterpretedKernel, bind_launcher, bits, differing,
                                       fused_oracle)
from .test_complex_chunk_forms import launcher  # noqa: F401 - the shared fixture
from .test_root_chunk_forms import F32

STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
         "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _split_y(grid: Grid, pml: PML) -> ChunkLayout:
    """The one-rank boxes with the low-y effort volume cut two cells short of its edge."""
    one_rank = ChunkLayout.one_rank(grid, pml)
    window = one_rank.frames[1].window
    low_edge = min(hi[1] for lo, hi in one_rank.boxes if lo[1] == window[0])
    cut = low_edge - 2
    boxes, flags = [], []
    for (lo, hi), flag in zip(one_rank.boxes, one_rank.pml_flags):
        if lo[1] == window[0]:
            boxes.append((lo, (hi[0], cut, hi[2])))
            flags.append(flag)
            boxes.append(((lo[0], cut, lo[2]), hi))
            flags.append((flag[0], False, flag[2]))
        else:
            boxes.append((lo, hi))
            flags.append(flag)
    return ChunkLayout(boxes, flags, grid_frames(grid), faces=pml.thickness_by_face,
                       source="meep")


#: ``name -> (cell, boundaries, symmetry, k_point, thickness, divide, beta)``. Every beta
#: grid is two-dimensional (z invariant, no phase on z: ``Grid._resolve_beta``).
CASES: Dict[str, Tuple] = {
    # TestSpecialKz.test_special_kz's class: unfolded, an in-plane kx, negative beta.
    "plain_kx": ((12.0, 8.0, 0.0), ("periodic", "metallic", "periodic"), (),
                 (0.27, 0.0, 0.0), ((2, 2), (2, 2), (0, 0)), None, -0.39073112848927377),
    "plain_walls": ((10.0, 9.0, 0.0), ("metallic", "metallic", "periodic"), (),
                    (0.0, 0.0, 0.0), ((2, 2), (2, 2), (0, 0)), None, 0.3321611318837033),
    "split_walls": ((9.0, 12.0, 0.0), ("metallic", "metallic", "periodic"), (),
                    (0.0, 0.0, 0.0), ((2, 2), (2, 2), (0, 0)), "split_y", 0.2),
    "fold_y_periodic": ((8.0, 14.0, 0.0), ("periodic", "periodic", "periodic"),
                        (("Y", 1),), (0.0, 0.0, 0.0), ((2, 2), (0, 2), (0, 0)), None, 0.2),
    # The binary-grating class: Mirror(Y) with an in-plane kx on the layered x axis.
    "fold_y_periodic_kx": ((10.0, 14.0, 0.0), ("periodic", "periodic", "periodic"),
                           (("Y", 1),), (0.27, 0.0, 0.0), ((2, 2), (0, 2), (0, 0)), None,
                           -0.6850526103319672),
    "fold_y_metallic": ((8.0, 14.0, 0.0), ("periodic", "metallic", "periodic"),
                        (("Y", -1),), (0.0, 0.0, 0.0), ((2, 2), (0, 2), (0, 0)), None, 0.2),
    "fold_x_periodic_odd": ((15.0, 8.0, 0.0), ("periodic", "metallic", "periodic"),
                            (("X", -1),), (0.0, 0.0, 0.0), ((0, 2), (2, 2), (0, 0)), None,
                            -0.9120991827764708),
}

FOLDED = tuple(c for c in CASES if CASES[c][2])


def build(case: str, plant_zeros: bool, h_separated: bool, seed: int = 21,
          beta: Any = None):
    """``(fields, grid, pml)``: a seeded complex full-volume state on ``case``'s grid."""
    cell, boundaries, symmetry, k_point, thickness, divide, case_beta = CASES[case]
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=2, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in symmetry),
                k_point=tuple(float(k) for k in k_point), xp=numpy, courant=0.5,
                beta=float(case_beta if beta is None else beta))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon = rng.uniform(1.0, 4.0, size=shape).astype(F32)
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(F32))
    pml = PML(grid=grid, thickness=thickness)
    if divide == "split_y":
        fields.install_region_plan(RegionPlan(_split_y(grid, pml), grid, pml,
                                              backend=region_backend(numpy)))
    for name in STATE:
        parts = []
        for _ in range(2):
            values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
            if plant_zeros:
                draw = rng.uniform(size=shape)
                values[draw < 0.15] = F32(-0.0)
                values[(draw >= 0.15) & (draw < 0.3)] = F32(0.0)
            parts.append(values)
        getattr(fields, name)[...] = part_products.from_parts(parts[0], parts[1],
                                                              numpy.complex64)
    fields.h_separated = bool(h_separated)
    return fields, grid, pml


def _state(fields) -> Dict[str, numpy.ndarray]:
    return {name: numpy.array(getattr(fields, name), copy=True) for name in STATE}


def _restore(fields, state) -> None:
    for name, values in state.items():
        getattr(fields, name)[...] = values


def _record(fields) -> numpy.ndarray:
    return numpy.ascontiguousarray(chunk_bounds_record(fields.region_plan.layout),
                                   dtype=numpy.int32)


# =============================================================================
# 1. The curl pair
# =============================================================================

@pytest.fixture()
def beta_launcher(launcher, monkeypatch):  # noqa: F811 - the fixture above
    """``complex_beta_kernels.step_complex_beta`` with its CuPy imports answered by the
    NumPy-loaded ``complex_pml_kernels`` and its compiled kernel by the interpreter."""
    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.complex_pml_kernels", launcher)
    monkeypatch.setattr(complex_beta_kernels, "_get_kernel",
                        lambda sub_step: InterpretedKernel(
                            complex_beta_kernels.kernel_source(sub_step)))
    complex_beta_kernels.reset_kernel_sources()
    yield complex_beta_kernels
    complex_beta_kernels.reset_kernel_sources()


def run_single(family, sub_step: str, case: str, plant: bool, text: str = None,
               beta_coefficients=None):
    """``(start, oracle, kernel)`` for one beta curl sub-step on one case."""
    fields, grid, pml = build(case, plant, True)
    start = _state(fields)
    getattr(stepping, sub_step)(fields, pml)
    want = _state(fields)
    record = _record(fields)
    _restore(fields, start)
    if text is not None:
        family.set_kernel_source(sub_step, text)
    try:
        family.step_complex_beta(sub_step, fields, grid=grid, pml=pml,
                                 chunk_bounds=record, beta_coefficients=beta_coefficients)
    finally:
        family.reset_kernel_sources()
    return start, want, _state(fields)


SINGLE_PARAMETERS = [(sub_step, case, plant) for sub_step in ("step_B", "step_D")
                     for case in sorted(CASES) for plant in (False, True)]


@pytest.mark.parametrize("sub_step,case,plant", SINGLE_PARAMETERS,
                         ids=[f"{s}-{c}-{'zeros' if p else 'random'}"
                              for s, c, p in SINGLE_PARAMETERS])
def test_the_complex_beta_curl_is_the_array_path_word_for_word(sub_step, case, plant,
                                                               beta_launcher):
    start, want, got = run_single(beta_launcher, sub_step, case, plant)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (sub_step, case, plant,
                                                {k: v for k, v in diff.items() if v})
    targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
    for target in targets:
        assert int(numpy.count_nonzero(bits(want[target]) != bits(start[target]))) > 0


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_the_beta_pass_reaches_the_output_on_every_case(sub_step):
    """No case is the folded complex curl under another name: the same state stepped
    with beta 0 differs from the beta run."""
    for case in CASES:
        fields, _grid, pml = build(case, False, True)
        getattr(stepping, sub_step)(fields, pml)
        plain, _plain_grid, plain_pml = build(case, False, True, beta=0.0)
        getattr(stepping, sub_step)(plain, plain_pml)
        assert sum(differing(_state(fields), _state(plain)).values()) > 0, (case, sub_step)


def test_every_beta_grid_windows_the_partner_axis_without_a_phase(launcher):  # noqa: F811
    """The beta pass reads the partner's stored word at the cell; the kernel reads it
    from the curl's centre register, which is the window along z. On every beta grid z
    is the invariant axis with no phase, so the window is a plain load."""
    for case in CASES:
        _fields, grid, _pml = build(case, False, True)
        for backward in (False, True):
            kinds, _values = launcher.bloch_phase_arguments(grid, backward)
            assert int(kinds[2]) == complex_emitter.CONNECT_NONE, (case, backward, kinds)


#: ``name -> (sub_step, find, replace, case, planted, beta override)``. Every one must
#: change a word, except the declared nulls in :data:`SINGLE_NULLS`.
SINGLE_DEFECTS = {
    # The 0.9.2 folding: the increment added to the curl before the masks, riding the
    # curl's own recurrence.
    "beta_folded_into_the_curl": (
        "step_B",
        "        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();\n"
        "        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = cf_zero();\n"
        "        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = cf_zero();\n"
        "        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = cf_zero();\n"
        "        pml_apply(f0,",
        "        curl = cf_sub(curl, beta_increment(bp, f_2));\n"
        "        if (bc_x == BC_METALLIC && i == 0) curl = cf_zero();\n"
        "        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = cf_zero();\n"
        "        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = cf_zero();\n"
        "        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = cf_zero();\n"
        "        pml_apply(f0,", "plain_walls", False, None),
    # The quarter turn's parts swapped: the coupling's sign of i.
    "quarter_turn_parts_swapped": (
        "step_D", "    o.re = c * g.im;\n    o.im = -(c * g.re);",
        "    o.re = -(c * g.re);\n    o.im = c * g.im;", "fold_y_periodic", False, None),
    # The masked increment added: a -0 kept by MEEP's unvisited cell becomes +0.
    "beta_masked_increment_added": (
        "step_B", "cf inc_beta = owned_cell ? beta_increment(bp, f_2) : cf_zero();",
        "cf inc_beta = owned_cell ? beta_increment(bp, f_2) : cf_neg(cf_zero());",
        "plain_walls", True, None),
    # The partner read from the shifted operand instead of the cell's own word.
    "partner_shifted": (
        "step_B", "beta_increment(bp, f_2)", "beta_increment(bp, ss)", "plain_kx", False,
        None),
    # The ladder without its recurrence stage on the dsigu cells.
    "ladder_without_the_recurrence": (
        "step_D", "    if (dsigu) cf_store(fu, idx, cf_sub(cf_load(fu, idx), df));\n", "",
        "plain_walls", False, None),
    # The partner read from the stored word rather than the centre register: the same
    # word on every beta grid (z has no phase).
    "partner_from_the_stored_word": (
        "step_B", "beta_increment(bp, f_2)", "beta_increment(bp, cf_load(g1, idx))",
        "fold_y_periodic_kx", False, None),
}

#: The defects that must change NO word: ``partner_from_the_stored_word`` (the centre
#: register is the stored word wherever beta is legal) and ``partner_shifted`` (every
#: beta partner's shifted companion is its z neighbour, and nz == 1 on every beta grid).
SINGLE_NULLS = ("partner_from_the_stored_word", "partner_shifted")


@pytest.mark.parametrize("defect", sorted(SINGLE_DEFECTS))
def test_each_planted_beta_defect_changes_a_word_and_each_null_does_not(defect,
                                                                        beta_launcher):
    sub_step, find, replace, case, plant, override = SINGLE_DEFECTS[defect]
    text = beta_launcher.beta_source(sub_step)
    assert text.count(find) == 1, (defect, text.count(find))
    mutated = text.replace(find, replace, 1)
    if "cf_neg(" in mutated and "cf_neg(" not in text:
        mutated = mutated.replace(
            'extern "C" __global__ void ',
            "__device__ __forceinline__ cf cf_neg(cf z) { cf o; o.re = -z.re; "
            "o.im = -z.im; return o; }\n" + 'extern "C" __global__ void ', 1)
    _start, want, got = run_single(beta_launcher, sub_step, case, plant, text=mutated,
                                   beta_coefficients=override)
    moved = sum(differing(want, got).values())
    if defect in SINGLE_NULLS:
        assert moved == 0, (defect, moved)
    else:
        assert moved > 0, defect


@pytest.mark.parametrize("swap", ("signs_swapped", "added_not_subtracted", "dropped"))
def test_each_host_coefficient_defect_changes_a_word(swap, beta_launcher):
    """The two host words, mutated at the launcher's door, on every case."""
    caught = 0
    for case in CASES:
        for sub_step in ("step_B", "step_D"):
            fields, grid, _pml = build(case, False, True)
            plus, minus = complex_beta_kernels.beta_curl_coefficients(
                grid.beta, grid.dt, complex_beta_kernels.MAGNETIC[sub_step])
            words = {"signs_swapped": (minus, plus),
                     "added_not_subtracted": (-plus, -minus),
                     "dropped": (0.0, 0.0)}[swap]
            _start, want, got = run_single(beta_launcher, sub_step, case, False,
                                           beta_coefficients=words)
            caught += int(sum(differing(want, got).values()) > 0)
    assert caught == 2 * len(CASES), (swap, caught)


def test_the_coefficient_is_the_array_paths_float32_word():
    """``beta_curl_coefficients`` is ``part_products._realnum`` of stepping's double
    coefficient, the D side negated before the one rounding, bit for bit."""
    for beta in (0.2, -0.6850526103319672, -0.9120991827764708, -0.39073112848927377,
                 0.3321611318837033):
        dt = 0.5
        for magnetic in (True, False):
            words = complex_beta_kernels.beta_curl_coefficients(beta, dt, magnetic)
            for word, sign in zip(words, (1.0, -1.0)):
                coefficient = sign * 2.0 * numpy.pi * beta * dt
                want = numpy.float32(coefficient if magnetic else -coefficient)
                assert numpy.float32(word).view(numpy.uint32) == want.view(numpy.uint32)


# =============================================================================
# 2. The fused pairs
# =============================================================================

FUSED = {
    "B": ("complex_beta_fused_magnetic_pair",
          ("step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
           "update_H"), False),
    "D": ("complex_beta_fused_electric_pair",
          ("step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
           "update_E"), True),
}


def run_fused(side: str, case: str, plant: bool, separated: bool, launcher,  # noqa: F811
              monkeypatch, text: str = None):
    stem, passes, backward = FUSED[side]
    family = importlib.import_module(f"meep_gpu.cuda_kernels.{stem}")
    bind_launcher(family, launcher, monkeypatch)
    fields, grid, pml = build(case, plant, separated)
    start = _state(fields)
    fused_oracle(passes, fields, pml)
    want = _state(fields)
    record = _record(fields)
    _restore(fields, start)
    fields.h_separated = bool(separated)
    codes, refusal = complex_folded_kernels.folded_complex_boundary_codes(grid)
    assert refusal is None, refusal
    flags, values = launcher.bloch_phase_arguments(grid, backward)
    kernel = InterpretedKernel(text or getattr(family, f"{stem}_source")())
    extra = {"layer": pml} if side == "B" else {}
    from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415

    getattr(family, f"launch_{stem}")(
        fields, getattr(family, f"{stem}_tables")(pml), codes, flags, values,
        family.beta_coefficients(grid), zero_metal_axes(grid),
        getattr(family, f"{stem}_fills")(grid), float(grid.dt / grid.dx),
        kernel=kernel, chunk_bounds=record, **extra)
    assert kernel.launches == 1
    return start, want, _state(fields)


FUSED_PARAMETERS = [(side, case, plant, separated)
                    for side in ("B", "D") for case in sorted(CASES)
                    for plant in (False, True)
                    for separated in ((True, False) if side == "B" else (True,))]


@pytest.mark.parametrize(
    "side,case,plant,separated", FUSED_PARAMETERS,
    ids=[f"{s}-{c}-{'zeros' if p else 'random'}-{'midrun' if h else 'first'}"
         for s, c, p, h in FUSED_PARAMETERS])
def test_the_complex_beta_fused_text_is_the_array_path_word_for_word(
        side, case, plant, separated, launcher, monkeypatch):  # noqa: F811
    start, want, got = run_fused(side, case, plant, separated, launcher, monkeypatch)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (side, case, plant, separated,
                                                {k: v for k, v in diff.items() if v})
    curl = ("Bx", "By", "Bz") if side == "B" else ("Dx", "Dy", "Dz")
    for target in curl:
        assert int(numpy.count_nonzero(bits(want[target]) != bits(start[target]))) > 0


FUSED_DEFECTS = {
    # The register captured off the curl alone: the fills and update_H read B before
    # the beta pass.
    "magnetic_register_before_the_pass": (
        "B", "        b1 = cfold_apply_reg(f1,", "        cfold_apply_reg(f1,",
        "fold_y_periodic"),
    "electric_register_before_the_pass": (
        "D", "        d0 = cfold_apply_reg(f0,", "        cfold_apply_reg(f0,",
        "fold_x_periodic_odd"),
}


@pytest.mark.parametrize("defect", sorted(FUSED_DEFECTS))
def test_each_planted_fused_beta_defect_changes_a_word(defect, launcher,  # noqa: F811
                                                       monkeypatch):
    side, find, replace, case = FUSED_DEFECTS[defect]
    stem = FUSED[side][0]
    family = importlib.import_module(f"meep_gpu.cuda_kernels.{stem}")
    text = getattr(family, f"{stem}_source")()
    assert text.count(find) == 1, (defect, text.count(find))
    _start, want, got = run_fused(side, case, False, True, launcher, monkeypatch,
                                  text=text.replace(find, replace, 1))
    assert sum(differing(want, got).values()) > 0, defect


@pytest.mark.parametrize("side", ("B", "D"))
def test_the_fused_text_decodes_the_record_once_and_keeps_no_plain_helper(side):
    stem = FUSED[side][0]
    family = importlib.import_module(f"meep_gpu.cuda_kernels.{stem}")
    text = getattr(family, f"{stem}_source")()
    prelude, body = text.split('extern "C" __global__ void ', 1)
    block = family.union_bounds_block()
    assert body.count("cb[") == block.count("cb["), side
    assert "void pml_apply(" not in prelude and "void cfold_apply(" not in prelude
    assert body.count("= cfold_apply_reg(") == 2, side
    assert "cfold_apply(" not in body.replace("cfold_apply_reg(", ""), side


# =============================================================================
# 3. The complex-beta H->D weld: update_H welded into step_D, scratch output
# =============================================================================

def run_hd(case: str, plant: bool, separated: bool, launcher, monkeypatch,  # noqa: F811
           text: str = None):
    from . import complex_beta_fused_hd_pair as family  # noqa: PLC0415
    from .test_root_chunk_forms import _NumpyWearingCupysName  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "cupy", _NumpyWearingCupysName())
    monkeypatch.setattr(family, "complex_pml_kernels", launcher)
    fields, grid, pml = build(case, plant, separated)
    start = _state(fields)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    want = _state(fields)
    record = _record(fields)
    _restore(fields, start)
    fields.h_separated = bool(separated)
    # The tables are handed in: resolve's identity check compares device base addresses,
    # which NumPy arrays do not carry; the two INTEGER groups are the same vectors.
    state = family.resolve(fields, grid, pml, chunk_bounds=record,
                           tables=launcher.complex_curl_tables(pml, False),
                           constitutive=launcher.complex_constitutive_tables(pml, False))
    kernel = InterpretedKernel(text or family.kernel_source())
    family.launch_complex_beta_fused_hd_pair(fields, state, kernel=kernel)
    state["scratch"] = family.rotate_into_fields(fields, state["scratch"])
    assert kernel.launches == 1
    return start, want, _state(fields)


HD_PARAMETERS = [(case, plant, separated) for case in sorted(CASES)
                 for plant in (False, True) for separated in (True, False)]


@pytest.mark.parametrize("case,plant,separated", HD_PARAMETERS,
                         ids=[f"{c}-{'zeros' if p else 'random'}-{'midrun' if h else 'first'}"
                              for c, p, h in HD_PARAMETERS])
def test_the_complex_beta_hd_weld_is_the_array_path_word_for_word(
        case, plant, separated, launcher, monkeypatch):  # noqa: F811
    start, want, got = run_hd(case, plant, separated, launcher, monkeypatch)
    diff = differing(want, got)
    assert all(v == 0 for v in diff.values()), (case, plant, separated,
                                                {k: v for k, v in diff.items() if v})
    for target in ("Dx", "Dy", "Dz", "Hx", "Hy", "Hz"):
        assert int(numpy.count_nonzero(bits(want[target]) != bits(start[target]))) > 0


def test_a_complex_beta_partner_on_the_stale_field_changes_a_word(launcher,  # noqa: F811
                                                                  monkeypatch):
    from . import complex_beta_fused_hd_pair as family  # noqa: PLC0415

    text = family.kernel_source()
    find = "cf inc_beta = owned_cell ? beta_increment(bp, f_2) : cf_zero();"
    assert text.count(find) == 1
    mutated = text.replace(find, "cf inc_beta = owned_cell ? beta_increment(bp, "
                                 "cf_load(g1, idx)) : cf_zero();", 1)
    _start, want, got = run_hd("plain_walls", False, True, launcher, monkeypatch,
                               text=mutated)
    assert sum(differing(want, got).values()) > 0
