"""The CUDA fold families run MEEP's folded increments as passes after the curl, read off
their text.

WHAT THIS FILE SETTLES, ON A HOST WITH NO DEVICE. The BFAST curl pair (``bfast_curl``) and
the real special_kz curl pair (``special_kz_curl``) add their terms as MEEP does
(step_db.cpp:124-175): per component the certified curl form, then the BFAST pass, and the
beta pass on the x and y components, each through its own additive ladder keyed on the
chunk record (``stepping._apply_folded_increment``, ``_bfast_term``,
``_special_kz_beta_term``; bit_parity_0_10 KERNELS.md section 6, step C7). Each kernel's
device text is EXECUTED in float32 NumPy by the root lane's interpreter
(``test_root_chunk_forms.Interpreter``) over a seeded full-volume state -- the BFAST state
included, nonzero everywhere -- and every array of the ``Fields`` is compared word for
word (sign of zero included) with the array path's own sub-step on a copy of the same
state.

The cases cover a 3-D grid with both partners on every component, a 2-D grid with a
nonzero k on the invariant axis (the class whose partner MEEP does not read), a one-axis
grid (one-partner components in MEEP's plain branch and a component with neither
partner), a lifted division whose record differs from the one-rank one, and planted
signed zeros. Each case asserts nonzero cell counts on and off the dsigu run and that
the sub-step moved its field and state. A battery of planted defects -- the 0.9.2 folding
of the increment into the curl among them -- must change a word.
"""

from __future__ import annotations

import importlib
import pathlib
from typing import Any, Dict, List, Tuple

import numpy
import pytest

from .. import stepping
from ..chunk_map import ChunkLayout, chunk_bounds_record, grid_frames
from ..driver import FdtdDriver
from .coverage import real_curl_boundary_codes
from .test_root_chunk_forms import F32, Interpreter

HERE = pathlib.Path(__file__).parent


# =============================================================================
# The modules
# =============================================================================

@pytest.fixture(scope="module")
def families() -> Dict[str, Any]:
    """The fold families. Both are stdlib-only at import (they read the root texts off
    their sibling's source), so no CuPy stand-in is needed to load them."""
    return {stem: importlib.import_module(f"meep_gpu.cuda_kernels.{stem}")
            for stem in ("bfast_curl", "special_kz_curl")}


def _text(families, kernel: str) -> str:
    if "bfast" in kernel:
        return families["bfast_curl"].kernel_source(kernel)
    return families["special_kz_curl"].kernel_source(kernel)


# =============================================================================
# The cases
# =============================================================================

def _split_y_layout(driver: FdtdDriver) -> ChunkLayout:
    """The one-rank boxes with the low-y effort volume cut short of its edge (the root
    lane's lifted division): the record differs from the one-rank one on the y line."""
    one_rank = ChunkLayout.one_rank(driver.grid, driver.pml)
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
    return ChunkLayout(boxes, flags, grid_frames(driver.grid),
                       faces=driver.pml.thickness_by_face, source="meep")


#: name -> (cell, resolution, dimensions, boundaries, pml cells, divide, extra driver
#: keywords). The BFAST cases carry bfast_scaled_k, the beta cases beta (2-D only).
CASES = {
    "bfast_box3d": ((3.0, 3.5, 4.0), 8.0, 3,
                    {"x": "periodic", "y": "metallic", "z": "periodic"},
                    {"x": 6, "y": 6, "z": 6}, None,
                    {"bfast_scaled_k": (0.81, -0.37, 0.52)}),
    "bfast_invariant_k2d": ((6.0, 5.0, 0.0), 10.0, 2,
                            {"x": "periodic", "y": "metallic"},
                            {"x": 8, "y": 8}, None,
                            {"bfast_scaled_k": (0.2, 0.0, 0.15)}),
    "bfast_line1d": ((0.0, 0.0, 9.0), 10.0, 1,
                     {"x": "periodic", "y": "periodic", "z": "periodic"},
                     {"z": 12}, None,
                     {"bfast_scaled_k": (0.8169576958985646, 0.0, 0.0)}),
    "bfast_split_y2d": ((6.0, 5.0, 0.0), 10.0, 2,
                        {"x": "metallic", "y": "metallic"},
                        {"x": 8, "y": 8}, _split_y_layout,
                        {"bfast_scaled_k": (0.6, -0.45, 0.0)}),
    "beta_slab2d": ((6.0, 5.0, 0.0), 10.0, 2,
                    {"x": "metallic", "y": "metallic"},
                    {"x": 8, "y": 8}, None, {"beta": 0.3321611318837033}),
    "beta_periodic2d": ((5.0, 6.0, 0.0), 10.0, 2,
                        {"x": "periodic", "y": "metallic"},
                        {"x": 8, "y": 8}, None, {"beta": -0.39}),
    "beta_split_y2d": ((6.0, 5.0, 0.0), 10.0, 2,
                       {"x": "metallic", "y": "metallic"},
                       {"x": 8, "y": 8}, _split_y_layout, {"beta": 0.2}),
}

BFAST_CASES = tuple(c for c in CASES if c.startswith("bfast_"))
BETA_CASES = tuple(c for c in CASES if c.startswith("beta_"))


def _driver(case: str) -> FdtdDriver:
    cell, resolution, dimensions, boundaries, layer, divide, extra = CASES[case]
    driver = FdtdDriver(cell_size=cell, resolution=resolution, dimensions=dimensions,
                        boundaries=boundaries, force_complex_fields=False,
                        prefer_gpu=False, **extra)
    driver.setup_pml(layer)
    if divide is not None:
        driver.set_chunk_layout(divide(driver))
    rng = numpy.random.default_rng(20261009)
    shape = tuple(int(n) for n in driver.grid.shape)
    driver.set_epsilon((1.0 + 11.0 * rng.uniform(size=shape)).astype(F32))
    return driver


_FIELDS = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
           "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
           "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")
_BFAST_STATE = ("f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
                "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz")


def _state_names(fields) -> Tuple[str, ...]:
    return _FIELDS + tuple(n for n in _BFAST_STATE if getattr(fields, n, None) is not None)


def _seed(driver: FdtdDriver, plant_negative_zero: bool) -> None:
    """Every stepped array nonzero everywhere (the BFAST state included), so every
    chunk-boundary plane takes its form at the first launch; optionally ``-0`` planted on
    a quarter of the cells."""
    rng = numpy.random.default_rng(11)
    fields = driver.fields
    for name in _state_names(fields):
        array = getattr(fields, name)
        values = rng.uniform(-1.0, 1.0, size=array.shape).astype(F32)
        if plant_negative_zero:
            values[rng.uniform(size=array.shape) < 0.25] = F32(-0.0)
        array[...] = values
    fields.h_separated = True  # a seeded state is a mid-run one


def _bits(array: numpy.ndarray) -> numpy.ndarray:
    return numpy.ascontiguousarray(array).view(numpy.uint32)


def _snapshot(fields) -> Dict[str, numpy.ndarray]:
    return {name: numpy.array(getattr(fields, name), copy=True)
            for name in _state_names(fields)}


def _restore(fields, state: Dict[str, numpy.ndarray]) -> None:
    for name, values in state.items():
        getattr(fields, name)[...] = values


def _flat(fields, name: str) -> numpy.ndarray:
    array = getattr(fields, name)
    assert array.flags.c_contiguous and array.dtype == F32
    return array.reshape(-1)


def _tables(pml, half_integer: bool) -> Dict[str, numpy.ndarray]:
    suffix = "_h" if half_integer else ""
    return {f"{kind}_{axis}": numpy.ascontiguousarray(
                getattr(pml, f"{kind}_{axis}{suffix}").reshape(-1))
            for axis in "xyz" for kind in ("kms", "sinv")}


def _single_arguments(families, kernel: str, driver: FdtdDriver,
                      record: numpy.ndarray) -> Dict[str, Any]:
    fields, grid, pml = driver.fields, driver.grid, driver.pml
    nx, ny, nz = (int(n) for n in grid.shape)
    sub_step = "step_B" if kernel.startswith("step_B") else "step_D"
    side, source = ("B", "E") if sub_step == "step_B" else ("D", "H")
    out: Dict[str, Any] = {"nx": nx, "ny": ny, "nz": nz, "cb": record,
                           "dtdx": F32(grid.dt / grid.dx)}
    for axis in "xyz":
        out[f"{side}{axis}"] = _flat(fields, f"{side}{axis}")
        out[f"fu_{side}{axis}"] = _flat(fields, f"fu_{side}{axis}")
        out[f"{source}{axis}"] = _flat(fields, f"{source}{axis}")
    out.update(_tables(pml, sub_step == "step_B"))
    codes, refusal = real_curl_boundary_codes(grid)
    assert refusal is None, refusal
    out.update({"bc_x": codes[0], "bc_y": codes[1], "bc_z": codes[2]})
    if "bfast" in kernel:
        module = families["bfast_curl"]
        for axis in "xyz":
            out[f"fb_{side}{axis}"] = _flat(fields, f"f_bfast_{side}{axis}")
        for (k1, k2), names in zip(module.bfast_curl_coefficients(grid, sub_step),
                                   module.K_NAMES):
            out[names[0]], out[names[1]] = F32(k1), F32(k2)
    else:
        plus, minus = families["special_kz_curl"].beta_curl_coefficients(grid.beta, grid.dt)
        out["beta_plus"], out["beta_minus"] = F32(plus), F32(minus)
    return out


ARRAY_PATH = {"step_B": stepping.step_B, "step_D": stepping.step_D}

SINGLES = ("step_B_bfast_real", "step_D_bfast_real",
           "step_B_special_kz_real", "step_D_special_kz_real")


def _cases_for(kernel: str) -> Tuple[str, ...]:
    return BFAST_CASES if "bfast" in kernel else BETA_CASES


def _run(text: str, families, kernel: str, case: str, plant: bool,
         record_of=None) -> Tuple[Dict, Dict, Dict, Any]:
    """``(start, oracle, kernel state, driver facts)`` for one kernel text on one case."""
    driver = _driver(case)
    try:
        _seed(driver, plant)
        fields = driver.fields
        start = _snapshot(fields)
        sub_step = "step_B" if kernel.startswith("step_B") else "step_D"
        ARRAY_PATH[sub_step](fields, driver.pml)
        oracle = _snapshot(fields)
        _restore(fields, start)
        record = numpy.ascontiguousarray(
            driver.chunk_bounds_record() if record_of is None else record_of(driver),
            dtype=numpy.int32)
        Interpreter(text).launch(_single_arguments(families, kernel, driver, record),
                                 int(numpy.prod(driver.grid.shape)))
        kernel_state = _snapshot(fields)
        side = sub_step[-1]
        plan = fields.region_plan
        shape = tuple(int(n) for n in driver.grid.shape)
        counts = {}
        for axis in "xyz":
            inside = numpy.broadcast_to(numpy.asarray(plan.mask("curl", f"{side}{axis}", numpy)),
                                        shape)
            counts[f"{side}{axis}"] = (int(inside.sum()), int((~inside).sum()))
        terms = stepping.B_CURL_TERMS if side == "B" else stepping.D_CURL_TERMS
        partnered = {term.target: any(stepping.curl_presence(driver.grid, term))
                     for term in terms}
        facts = {"counts": counts, "partnered": partnered,
                 "invariant": tuple(bool(driver.grid.is_invariant(a)) for a in range(3))}
    finally:
        driver.close()
    return start, oracle, kernel_state, facts


def _differing(a: Dict[str, numpy.ndarray], b: Dict[str, numpy.ndarray]) -> Dict[str, int]:
    return {name: int(numpy.count_nonzero(_bits(a[name]) != _bits(b[name]))) for name in a}


# =============================================================================
# The text computes the array path
# =============================================================================

def _single_matrix() -> List[Tuple[str, str, bool]]:
    return [(kernel, case, plant) for kernel in SINGLES for case in _cases_for(kernel)
            for plant in (False, True)]


@pytest.mark.parametrize("kernel,case,plant", _single_matrix(),
                         ids=[f"{k}-{c}-{'planted' if p else 'random'}"
                              for k, c, p in _single_matrix()])
def test_the_fold_single_is_the_array_path_word_for_word(kernel, case, plant, families):
    start, oracle, state, facts = _run(_text(families, kernel), families, kernel, case, plant)
    differing = _differing(oracle, state)
    assert all(v == 0 for v in differing.values()), (kernel, case, plant,
                                                     {k: v for k, v in differing.items() if v})
    side = "B" if kernel.startswith("step_B") else "D"
    for axis in "xyz":
        component = f"{side}{axis}"
        moved = int(numpy.count_nonzero(_bits(oracle[component]) != _bits(start[component])))
        if facts["partnered"][component]:
            assert moved > 0, (kernel, case, component)
            continue
        # Neither partner: MEEP steps no such component, and none of these cases puts a
        # layer on its dsig or dsigu axis, so its field and its BFAST state keep every word.
        assert moved == 0, (kernel, case, component, moved)
        if "bfast" in kernel:
            state_name = f"f_bfast_{component}"
            assert numpy.array_equal(_bits(oracle[state_name]), _bits(start[state_name])), (
                kernel, case, state_name)
    assert any(facts["partnered"].values()), facts
    if "bfast" in kernel:
        moved_state = sum(int(numpy.count_nonzero(
            _bits(oracle[f"f_bfast_{side}{axis}"]) != _bits(start[f"f_bfast_{side}{axis}"])))
            for axis in "xyz")
        assert moved_state > 0, (kernel, case)
    # Off and on the dsigu run, on the axes that carry a layer.
    assert any(inside > 0 for inside, _outside in facts["counts"].values()), facts
    assert all(outside > 0 for _inside, outside in facts["counts"].values()), facts


def test_the_lifted_division_moves_the_record():
    for case in ("bfast_split_y2d", "beta_split_y2d"):
        driver = _driver(case)
        try:
            lifted = numpy.asarray(driver.chunk_bounds_record())
            default = chunk_bounds_record(ChunkLayout.one_rank(driver.grid, driver.pml))
        finally:
            driver.close()
        assert not numpy.array_equal(lifted, default), case


@pytest.mark.parametrize("kernel", ("step_B_bfast_real", "step_D_special_kz_real"))
def test_a_fold_single_handed_the_one_rank_record_on_the_lifted_division_is_caught(
        kernel, families):
    case = "bfast_split_y2d" if "bfast" in kernel else "beta_split_y2d"
    _start, oracle, state, _facts = _run(
        _text(families, kernel), families, kernel, case, False,
        record_of=lambda d: chunk_bounds_record(ChunkLayout.one_rank(d.grid, d.pml)))
    assert any(_differing(oracle, state).values())


#: Defects the equivalence must see, each a rewrite of one emitted text:
#: ``(kernel, find, replace, case, planted)``. Every one must change a word.
DEFECTS = {
    # The 0.9.2 folding: the increment added to the curl before the masks, riding the
    # curl's own recurrence, and no ladder of its own.
    "beta_folded_into_the_curl": (
        "step_B_special_kz_real",
        "        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;\n"
        "        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;\n"
        "        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;\n"
        "        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;\n"
        "        pml_apply(Bx,",
        "        curl = curl - (beta_plus * f2);\n"
        "        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;\n"
        "        if (bc_x == BC_MIRROR_PERIODIC && i == 0) curl = 0.0f;\n"
        "        if (bc_y == BC_MIRROR_PERIODIC && j == ny - 1) curl = 0.0f;\n"
        "        if (bc_z == BC_MIRROR_PERIODIC && k == nz - 1) curl = 0.0f;\n"
        "        pml_apply(Bx,", "beta_slab2d", False),
    "bfast_one_rounding_advance": (
        "step_B_bfast_real",
        "    float updated = (pr1 && pr2) ? (((k1 * (sf + f1)) - (k2 * (ss + f2))) - fprev)",
        "    float updated = (pr1 && pr2) ? (fprev + (((k1 * (sf + f1)) - (k2 * (ss + f2)))"
        " - (2.0f * fprev)))", "bfast_box3d", False),
    "bfast_state_on_an_unowned_cell": (
        "step_D_bfast_real", "    if (!owned) return 0.0f;\n    F[idx] = updated;",
        "    F[idx] = updated;\n    if (!owned) return 0.0f;", "bfast_split_y2d", False),
    "bfast_absent_partner_read": (
        "step_B_bfast_real",
        "    float updated = (pr1 && pr2) ?",
        "    float updated = (pr1 || pr2) ?", "bfast_invariant_k2d", False),
    "bfast_plain_branch_dropped": (
        "step_B_bfast_real", ": (plain ? single : (single - fprev));",
        ": (single - fprev);", "bfast_line1d", False),
    "bfast_masked_increment_added": (
        "step_D_bfast_real", "    if (!owned) return 0.0f;\n    F[idx]",
        "    if (!owned) return -0.0f;\n    F[idx]", "bfast_split_y2d", True),
    "fold_ladder_without_the_recurrence": (
        "step_D_special_kz_real", "    if (dsigu) fu[idx] = fu[idx] - df;\n",
        "", "beta_slab2d", False),
    "fold_ladder_sinv_twice": (
        "step_B_bfast_real", ": (fcur - df);", ": (fcur - (sinv * df));",
        "bfast_box3d", False),
    "beta_masked_increment_added": (
        "step_B_special_kz_real", "(-(beta_plus * f2)) : 0.0f;",
        "(-(beta_plus * f2)) : -0.0f;", "beta_slab2d", True),
}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_each_planted_fold_defect_changes_a_word(defect, families):
    kernel, find, replace, case, plant = DEFECTS[defect]
    text = _text(families, kernel)
    assert text.count(find) >= 1, (defect, find)
    mutated = text.replace(find, replace)
    assert mutated != text
    _start, oracle, state, _facts = _run(mutated, families, kernel, case, plant)
    assert any(_differing(oracle, state).values()), defect
