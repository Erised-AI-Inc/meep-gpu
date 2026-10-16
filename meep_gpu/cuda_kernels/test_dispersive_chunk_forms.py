"""The CUDA dispersive and ADE kernels transcribe the array path's forms, read off their text.

WHAT THIS FILE SETTLES, ON A HOST WITH NO DEVICE. The hand-CUDA ``update_P`` kernels
(``ade_kernels``), the PML dispersive ``update_E`` (``dispersive_kernels``' arm 1) and the
two products that lift them -- the E->P pair (``fused_polarization_pair``) and the
three-slot weld's recurrence group (``three_slot_dispersive_weld``) -- step the array
path's forms (bit_parity_0_10 DESIGN.md D1, D3, D27; KERNELS.md section 6, F_DISPERSIVE):

* ``update_P``: MEEP's isotropic recurrence in its four float32 words and its grouping,
  ``gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s * w)))``
  (``dispersion.PolarizationState.update``), with MEEP's drive PER CHUNK, ``w = f_w ? f_w
  : f`` (update_pols.cpp:44, ``Fields.drive_field``), selected in the kernel from the run's
  chunk record;
* ``update_E`` under a layer: ``E = (D - sum P) * inv_eps`` on the cells whose chunk does
  not carry the component's own layer, MEEP's accumulation on those that do
  (``stepping._apply_constitutive_regions``), ``f_w`` written on the latter only.

Each kernel's device text is EXECUTED in float32 NumPy by the interpreter of
``test_root_chunk_forms`` over a seeded full-volume state, and every array the sub-steps
read or write -- the fields, the auxiliaries, and every polarization buffer BY SLOT, so a
wrong rotation shows -- is compared word for word (sign of zero included) with the array
path on a copy of the same state. The cases are the root file's: a 3-D box, a 2-D slab, a
periodic axis carrying a layer and a lifted division whose record differs from the
one-rank record, plus a mirror fold for ``update_P`` (which runs over the whole stored
array). Each asserts that the drive select is live: the component whose own axis carries a
layer has cells on both sides of its membership run, and ``f_w`` and the stored E differ
there by construction of the seed.

A planted-defect table then shows the equivalence is not vacuous: each row rewrites one
text and must change a word.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Dict, List, Tuple

import numpy
import pytest

from .. import stepping
from ..dispersion import Susceptibility
from ..driver import FdtdDriver
from ..grid import Mirror
from . import dispersive_kernels, fused_polarization_pair, three_slot_dispersive_weld
from .test_root_chunk_forms import (F32, Interpreter, _NumpyWearingCupysName,
                                    _split_y_layout)

HERE = pathlib.Path(__file__).parent
COMPONENTS = ("Ex", "Ey", "Ez")


@pytest.fixture(scope="module")
def ade():
    """``ade_kernels`` loaded by path under a private name, ``cupy`` stubbed for the load."""
    previous = sys.modules.get("cupy")
    sys.modules["cupy"] = _NumpyWearingCupysName()
    try:
        spec = importlib.util.spec_from_file_location(
            "_dispersive_chunk_forms_ade_kernels", HERE / "ade_kernels.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = previous
    return module


# =============================================================================
# The cases
# =============================================================================

#: name -> (cell, resolution, dimensions, boundaries, pml cells, divide, symmetry)
CASES = {
    "box3d": ((3.0, 3.5, 4.0), 8.0, 3, {"x": "metallic", "y": "metallic", "z": "metallic"},
              {"x": 6, "y": 6, "z": 6}, None, ()),
    "slab2d": ((6.0, 5.0, 0.0), 10.0, 2, {"x": "metallic", "y": "metallic"},
               {"x": 8, "y": 8}, None, ()),
    "periodic3d": ((3.0, 3.0, 3.5), 8.0, 3, {"x": "periodic", "y": "metallic", "z": "metallic"},
                   {"x": 6, "y": 6, "z": 6}, None, ()),
    "split_y2d": ((6.0, 5.0, 0.0), 10.0, 2, {"x": "metallic", "y": "metallic"},
                  {"x": 8, "y": 8}, _split_y_layout, ()),
}

#: The mirror fold, for the sub-step that runs over the whole stored array.
FOLD_CASES = {
    "fold_y3d": ((3.0, 3.5, 3.0), 8.0, 3, {"x": "metallic", "y": "metallic", "z": "metallic"},
                 {"x": 6, "y": (0, 6), "z": 6}, None, (("Y", -1),)),
}


def _driver(case: str, plant_negative_zero: bool = False) -> FdtdDriver:
    cell, resolution, dimensions, boundaries, layer, divide, symmetry = (
        {**CASES, **FOLD_CASES}[case])
    driver = FdtdDriver(cell_size=cell, resolution=resolution, dimensions=dimensions,
                        boundaries=boundaries, force_complex_fields=False, prefer_gpu=False,
                        symmetry=tuple(Mirror(axis, phase) for axis, phase in symmetry))
    driver.setup_pml(layer)
    if divide is not None:
        driver.set_chunk_layout(divide(driver))
    rng = numpy.random.default_rng(20261008)
    shape = tuple(int(n) for n in driver.grid.shape)
    driver.set_epsilon((1.0 + 3.0 * rng.uniform(size=shape)).astype(F32))
    # Two terms, one of each kind and one of each sigma form per component: a volume on
    # Ex and Ez of the Lorentzian, a scalar on its Ey and on every Drude component.
    driver.add_susceptibility(
        Susceptibility(frequency=1.1, gamma=0.08, kind="lorentzian"),
        {"Ex": rng.uniform(0.2, 0.9, size=shape).astype(F32), "Ey": 0.45,
         "Ez": rng.uniform(0.2, 0.9, size=shape).astype(F32)})
    driver.add_susceptibility(Susceptibility(frequency=0.8, gamma=0.11, kind="drude"), 0.3)
    _seed(driver, plant_negative_zero)
    return driver


def _state_names(fields) -> List[str]:
    return [*(f"{side}{axis}" for side in "DE" for axis in "xyz"),
            *(f"f_w_E{axis}" for axis in "xyz")]


def _seed(driver: FdtdDriver, plant_negative_zero: bool) -> None:
    """Every array nonzero everywhere (each boundary plane takes its form at the first
    launch); ``f_w`` and the stored E seeded apart so the drive select is observable."""
    rng = numpy.random.default_rng(7)
    fields = driver.fields
    shape = tuple(int(n) for n in driver.grid.shape)
    # The plant: D = -0 and every pole word +0 on a quarter of the cells, so the chain
    # (-0 - 0) - 0 keeps -0 and src = -0 * inv_eps = -0 reaches the constitutive form.
    planted = (rng.uniform(size=shape) < 0.25) if plant_negative_zero else None
    for name in _state_names(fields):
        array = getattr(fields, name)
        low, high = ((0.6, 0.95) if name.startswith("E") else (-1.0, 1.0))
        values = rng.uniform(low, high, size=array.shape).astype(F32)
        if planted is not None and name.startswith("D"):
            values[planted] = F32(-0.0)
        array[...] = values
    for state in fields.polarizations:
        for component in state.driven():
            values = rng.uniform(-1, 1, state.P[component].shape).astype(F32)
            if planted is not None:
                values[planted] = F32(0.0)
            state.P[component][...] = values
            state.P_prev[component][...] = rng.uniform(
                -1, 1, state.P_prev[component].shape).astype(F32)
        state._scratch[...] = rng.uniform(-1, 1, state._scratch.shape).astype(F32)
    fields.h_separated = True


def _bits(array: numpy.ndarray) -> numpy.ndarray:
    return numpy.ascontiguousarray(array).view(numpy.uint32)


def _snapshot(fields) -> Dict[str, numpy.ndarray]:
    """Every array by NAME, and every polarization buffer by SLOT."""
    out = {name: numpy.array(getattr(fields, name), copy=True) for name in _state_names(fields)}
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            out[f"P{index}[{component}]"] = numpy.array(state.P[component], copy=True)
            out[f"Pprev{index}[{component}]"] = numpy.array(state.P_prev[component], copy=True)
        out[f"scratch{index}"] = numpy.array(state._scratch, copy=True)
    return out


def _restore(fields, snapshot: Dict[str, numpy.ndarray]) -> None:
    for name in _state_names(fields):
        getattr(fields, name)[...] = snapshot[name]
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            state.P[component][...] = snapshot[f"P{index}[{component}]"]
            state.P_prev[component][...] = snapshot[f"Pprev{index}[{component}]"]
        state._scratch[...] = snapshot[f"scratch{index}"]


def _differing(left: Dict[str, numpy.ndarray], right: Dict[str, numpy.ndarray]) -> Dict[str, int]:
    return {name: int(numpy.count_nonzero(_bits(left[name]) != _bits(right[name])))
            for name in left}


def _flat(array: numpy.ndarray) -> numpy.ndarray:
    assert array.flags.c_contiguous and array.dtype == F32
    return array.reshape(-1)


def _record(driver: FdtdDriver) -> numpy.ndarray:
    return numpy.ascontiguousarray(driver.chunk_bounds_record(), dtype=numpy.int32)


def _rotate(state, component: str) -> None:
    """``PolarizationState.update``'s rotation, once the launch returned."""
    p, p_prev, scratch = state.P[component], state.P_prev[component], state._scratch
    state.P[component] = scratch
    state.P_prev[component] = p
    state._scratch = p_prev


def _constants(state) -> Dict[str, Any]:
    names = ("two_minus_od", "gamma1", "omega0dtsqr", "gamma1inv")
    return {name: F32(value) for name, value in zip(names, state.constants)}


def _lanes(driver: FdtdDriver) -> int:
    return int(numpy.prod(driver.grid.shape))


# =============================================================================
# The kernel legs, transcribing each launcher's loop
# =============================================================================

def _run_update_p(ade, driver: FdtdDriver, texts: Dict[bool, str] = None) -> int:
    """``ade_kernels.update_P_fused_pml_real``'s loop, each launch executed from its text."""
    texts = texts or {True: ade._update_P_pml_real_kernel_code,
                      False: ade._update_P_pml_real_uniform_kernel_code}
    fields = driver.fields
    nx, ny, nz = (int(n) for n in driver.grid.shape)
    record = _record(driver)
    launches = 0
    for state in fields.polarizations:
        for component in state.driven():
            volume = bool(ade.ade_sigma_is_volume(state, component))
            drive_w, drive_e = ade.drive_pair(fields, driver.pml, component)
            arguments = {"p_out": _flat(state._scratch), "p_now": _flat(state.P[component]),
                         "p_prev": _flat(state.P_prev[component]),
                         "sigma": (_flat(state.sigma[component]) if volume
                                   else F32(state.sigma[component])),
                         "drive_w": _flat(drive_w), "drive_e": _flat(drive_e),
                         "nx": nx, "ny": ny, "nz": nz,
                         "own_axis": COMPONENTS.index(component), "cb": record,
                         **_constants(state)}
            Interpreter(texts[volume]).launch(arguments, _lanes(driver))
            launches += 1
            _rotate(state, component)
    return launches


def _e_arguments(driver: FdtdDriver, record: numpy.ndarray) -> Dict[str, Any]:
    fields, pml = driver.fields, driver.pml
    nx, ny, nz = (int(n) for n in driver.grid.shape)
    out: Dict[str, Any] = {"nx": nx, "ny": ny, "nz": nz, "cb": record}
    for axis in "xyz":
        out[f"E{axis}"] = _flat(getattr(fields, f"E{axis}"))
        out[f"f_w_E{axis}"] = _flat(getattr(fields, f"f_w_E{axis}"))
        out[f"D{axis}"] = _flat(getattr(fields, f"D{axis}"))
        out[f"inv_eps_E{axis}"] = numpy.ascontiguousarray(
            fields.inverse_epsilon_for(f"E{axis}")).reshape(-1)
        for kind in ("kps", "kms"):
            out[f"{kind}_{axis}"] = numpy.ascontiguousarray(
                getattr(pml, f"{kind}_{axis}_h").reshape(-1))
    plan = dispersive_kernels.resolve_pole_plan(fields)
    for target in COMPONENTS:
        for index, array in enumerate(plan[target]):
            out[f"P_{target}_{index}"] = _flat(array)
    return out


def _run_update_e(driver: FdtdDriver, text: str = None) -> None:
    counts = dispersive_kernels.pole_counts_of(dispersive_kernels.resolve_pole_plan(driver.fields))
    text = text or dispersive_kernels.dispersive_source("pml", counts)
    Interpreter(text).launch(_e_arguments(driver, _record(driver)), _lanes(driver))


def _run_polarization_pair(driver: FdtdDriver, transform=None) -> int:
    """``fused_polarization_pair.run_fused_polarization_pair``'s loop, PML arm."""
    fields, pml = driver.fields, driver.pml
    nx, ny, nz = (int(n) for n in driver.grid.shape)
    record = _record(driver)
    launches = 0
    for index, (target, source, axis) in enumerate(fused_polarization_pair.ELECTRIC_TERMS):
        spec = fused_polarization_pair.component_specs(fields)[target]
        states, kinds = spec["states"], spec["kinds"]
        text = fused_polarization_pair.fused_polarization_pair_source(
            "pml", index, len(states), kinds)
        if transform is not None:
            text = transform(text)
        arguments: Dict[str, Any] = {
            target: _flat(getattr(fields, target)),
            f"f_w_{target}": _flat(getattr(fields, f"f_w_{target}")),
            source: _flat(getattr(fields, source)),
            f"inv_eps_{target}": numpy.ascontiguousarray(
                fields.inverse_epsilon_for(target)).reshape(-1),
            "nx": nx, "ny": ny, "nz": nz, "cb": record,
            f"kps_{axis}": numpy.ascontiguousarray(getattr(pml, f"kps_{axis}_h").reshape(-1)),
            f"kms_{axis}": numpy.ascontiguousarray(getattr(pml, f"kms_{axis}_h").reshape(-1)),
        }
        for i, state in enumerate(states):
            arguments[f"P_{target}_{i}"] = _flat(state.P[target])
            arguments[f"p_out_{i}"] = _flat(state._scratch)
            arguments[f"p_prev_{i}"] = _flat(state.P_prev[target])
            arguments[f"sigma_{i}"] = (_flat(state.sigma[target]) if kinds[i]
                                       else F32(state.sigma[target]))
            arguments.update({f"{name}_{i}": value
                              for name, value in _constants(state).items()})
        Interpreter(text).launch(arguments, _lanes(driver))
        launches += 1
        for state in states:
            _rotate(state, target)
    return launches


def _run_three_slot_trailing(driver: FdtdDriver, transform=None) -> int:
    """``three_slot_dispersive_weld.run_three_slot_polarization``'s loop, PML arm."""
    fields = driver.fields
    nx, ny, nz = (int(n) for n in driver.grid.shape)
    record = _record(driver)
    launches = 0
    for index, target in enumerate(COMPONENTS):
        spec = three_slot_dispersive_weld.component_specs(fields)[target]
        states, kinds = spec["states"], spec["kinds"]
        if not states:
            continue
        text = three_slot_dispersive_weld.three_slot_polarization_source(index, len(states), kinds)
        if transform is not None:
            text = transform(text)
        drive_w, drive_e = three_slot_dispersive_weld._drive_arrays(fields, "pml", target)
        arguments: Dict[str, Any] = {"drive_w": _flat(drive_w), "drive_e": _flat(drive_e),
                                     "nx": nx, "ny": ny, "nz": nz, "own_axis": index,
                                     "cb": record}
        for i, state in enumerate(states):
            arguments[f"P_{target}_{i}"] = _flat(state.P[target])
            arguments[f"p_out_{i}"] = _flat(state._scratch)
            arguments[f"p_prev_{i}"] = _flat(state.P_prev[target])
            arguments[f"sigma_{i}"] = (_flat(state.sigma[target]) if kinds[i]
                                       else F32(state.sigma[target]))
            arguments.update({f"{name}_{i}": value
                              for name, value in _constants(state).items()})
        Interpreter(text).launch(arguments, _lanes(driver))
        launches += 1
        for state in states:
            _rotate(state, target)
    return launches


def _array_update_p(driver: FdtdDriver) -> None:
    stepping.update_P(driver.fields, driver.pml)


def _array_update_e(driver: FdtdDriver) -> None:
    stepping.update_E(driver.fields, driver.pml)


def _array_e_then_p(driver: FdtdDriver) -> None:
    stepping.update_E(driver.fields, driver.pml)
    stepping.update_P(driver.fields, driver.pml)


def _compare(case: str, plant: bool, oracle_leg, kernel_leg) -> Tuple[Dict, Dict, Dict, Any]:
    driver = _driver(case, plant)
    try:
        fields = driver.fields
        start = _snapshot(fields)
        oracle_leg(driver)
        oracle = _snapshot(fields)
        _restore(fields, start)
        kernel_leg(driver)
        got = _snapshot(fields)
        plan = fields.region_plan
        shape = tuple(int(n) for n in driver.grid.shape)
        inside = {c: numpy.broadcast_to(numpy.asarray(plan.mask("constitutive", c, numpy)),
                                        shape).copy() for c in COMPONENTS}
    finally:
        driver.close()
    return start, oracle, got, inside


def _assert_live(case: str, start, oracle, inside) -> None:
    """The comparison can see the forms: a split component and a moved state."""
    split = [c for c in COMPONENTS if inside[c].any() and (~inside[c]).any()]
    assert split, (case, "no component has cells on both sides of its membership run")
    for c in split:
        differ = _bits(start[f"f_w_{c}"]) != _bits(start[c])
        assert differ[inside[c]].any() and differ[~inside[c]].any(), (case, c)
    moved = [name for name in oracle if name.startswith("scratch") or name.startswith("P")
             if numpy.count_nonzero(_bits(oracle[name]) != _bits(start[name]))]
    assert moved, case


# =============================================================================
# 1. update_P: MEEP's words, grouping and per-chunk drive
# =============================================================================

@pytest.mark.parametrize("case", sorted({**CASES, **FOLD_CASES}))
def test_the_update_p_text_is_the_array_path_word_for_word(case, ade):
    start, oracle, got, inside = _compare(case, False, _array_update_p,
                                          lambda d: _run_update_p(ade, d))
    differing = _differing(oracle, got)
    assert all(v == 0 for v in differing.values()), (case, differing)
    _assert_live(case, start, oracle, inside)


def test_the_drive_flag_is_the_update_e_membership_of_the_components_own_axis(ade):
    """The kernel's ``dsigw`` at every cell equals the region plan's constitutive mask."""
    for case in sorted({**CASES, **FOLD_CASES}):
        driver = _driver(case)
        try:
            fields = driver.fields
            shape = tuple(int(n) for n in driver.grid.shape)
            record = _record(driver)
            for index, component in enumerate(COMPONENTS):
                ones = numpy.ones(_lanes(driver), dtype=F32)
                out = numpy.zeros(_lanes(driver), dtype=F32)
                # drive_w = 1, drive_e = 0, unit constants and history zeroed: the stored
                # word is the flag (gamma1inv * ((0 - 0) + (1 * (1 * w)))).
                arguments = {"p_out": out, "p_now": numpy.zeros_like(out),
                             "p_prev": numpy.zeros_like(out), "sigma": F32(1.0),
                             "drive_w": ones, "drive_e": numpy.zeros_like(out),
                             "two_minus_od": F32(1.0), "gamma1": F32(1.0),
                             "omega0dtsqr": F32(1.0), "gamma1inv": F32(1.0),
                             "nx": shape[0], "ny": shape[1], "nz": shape[2],
                             "own_axis": index, "cb": record}
                Interpreter(ade._update_P_pml_real_uniform_kernel_code).launch(
                    arguments, _lanes(driver))
                mask = numpy.broadcast_to(numpy.asarray(
                    fields.region_plan.mask("constitutive", component, numpy)), shape)
                assert numpy.array_equal(out.reshape(shape) == 1.0, mask), (case, component)
        finally:
            driver.close()


def test_the_drive_pair_is_f_w_and_e_under_a_layer_and_e_twice_without(ade):
    driver = _driver("box3d")
    try:
        fields = driver.fields
        w, e = ade.drive_pair(fields, driver.pml, "Ey")
        assert w is fields.f_w_Ey and e is fields.Ey
        w, e = ade.drive_pair(fields, None, "Ey")
        assert w is fields.Ey and e is fields.Ey
    finally:
        driver.close()


# =============================================================================
# 2. The PML dispersive update_E: the per-chunk constitutive forms
# =============================================================================

@pytest.mark.parametrize("plant", (False, True), ids=("random", "planted_negative_zero"))
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_dispersive_update_e_text_is_the_array_path_word_for_word(case, plant):
    start, oracle, got, inside = _compare(case, plant, _array_update_e, _run_update_e)
    differing = _differing(oracle, got)
    assert all(v == 0 for v in differing.values()), (case, plant, differing)
    for c in COMPONENTS:
        # The auxiliary contract: f_w off the membership run is never written.
        off = ~inside[c]
        assert numpy.array_equal(_bits(oracle[f"f_w_{c}"])[off], _bits(start[f"f_w_{c}"])[off])
        assert numpy.count_nonzero(_bits(oracle[c]) != _bits(start[c])), (case, c)


def test_the_dispersive_pml_arm_decodes_update_es_record_entries():
    text = dispersive_kernels.dispersive_source("pml", (1, 2, 0))
    from . import bounds_block
    assert (bounds_block.DECODE_END + "\n"
            + bounds_block.bounds_block(dispersive_kernels.PML_BOUNDS_ENTRIES)) in text
    assert "const int* __restrict__ cb\n) {" in text
    for axis, coordinate in zip("xyz", "ijk"):
        assert (f"kps_{axis}[{coordinate}], kms_{axis}[{coordinate}], in_h_{axis});") in text


def test_the_dispersive_prelude_is_the_root_constitutive_helper_character_for_character():
    """The family keeps its own copy (a probe rewrites one family's string at a time),
    and the copy is the root helper's text."""
    import re
    root = (HERE / "constitutive_kernels.py").read_text(encoding="utf-8")
    helper = re.search(r"__device__ __forceinline__ void constitutive_apply\(.*?\n}\n", root,
                       re.S).group(0)
    assert dispersive_kernels._PML_ACCUMULATION_NOTE.count(helper) == 1


# =============================================================================
# 3. The products that lift them
# =============================================================================

@pytest.mark.parametrize("plant", (False, True), ids=("random", "planted_negative_zero"))
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_polarization_pair_is_update_e_then_update_p_word_for_word(case, plant):
    start, oracle, got, inside = _compare(case, plant, _array_e_then_p, _run_polarization_pair)
    differing = _differing(oracle, got)
    assert all(v == 0 for v in differing.values()), (case, plant, differing)
    _assert_live(case, start, oracle, inside)


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_three_slot_recurrence_group_is_update_p_word_for_word(case):
    start, oracle, got, inside = _compare(case, False, _array_update_p, _run_three_slot_trailing)
    differing = _differing(oracle, got)
    assert all(v == 0 for v in differing.values()), (case, differing)
    _assert_live(case, start, oracle, inside)


# =============================================================================
# 4. Planted defects the equivalence must see
# =============================================================================

_SELECT = "dsigw ? drive_w[idx] : drive_e[idx]"
_RECURRENCE = "gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s * w)))"

#: ``name -> (leg, find, replace, case, plant)``. ``leg`` names the kernel the rewrite
#: lands in. Every row must change a word of the comparison above.
DEFECTS = {
    # update_P (the single, both sigma forms rewritten alike)
    "drive_f_w_on_every_cell": ("update_P", _SELECT, "drive_w[idx]", "box3d", False),
    "drive_stored_e_on_every_cell": ("update_P", _SELECT, "drive_e[idx]", "box3d", False),
    "drive_flag_on_another_axis": ("update_P", "own_axis == 0 ? in_h_x", "own_axis == 0 ? in_h_y",
                                   "box3d", False),
    "drive_bound_off_by_one": ("update_P", "(j >= cbh_y_a0 && j < cbh_y_a1)",
                               "(j >= cbh_y_a0 && j < cbh_y_a1 + 1)", "box3d", False),
    "drive_on_the_integer_record": ("update_P", "cbh_z_a0 = cb[20], cbh_z_a1 = cb[21]",
                                    "cbh_z_a0 = cb[8], cbh_z_a1 = cb[9]", "box3d", False),
    "lifted_record_ignored": ("update_P", "(j >= cbh_y_a0 && j < cbh_y_a1)",
                              "(j >= cbh_y_a0 && j < cbh_y_a1 - 1)", "split_y2d", False),
    "folded_three_term_form": (
        "update_P", _RECURRENCE,
        "((p * (gamma1inv * two_minus_od)) - ((gamma1inv * gamma1) * q)) + "
        "((gamma1inv * omega0dtsqr) * (s * w))", "box3d", False),
    "drive_constant_associated_with_sigma": (
        "update_P", "(omega0dtsqr * (s * w))", "((omega0dtsqr * s) * w)", "box3d", False),
    "drive_term_added_before_the_history": (
        "update_P", _RECURRENCE,
        "gamma1inv * (((p * two_minus_od) + (omega0dtsqr * (s * w))) - (gamma1 * q))",
        "box3d", False),
    # update_E, the PML dispersive arm
    "old_constitutive_grouping": ("update_E", "(f[idx] + ((kps * src) - (kms * prev)))",
                                  "((f[idx] + (kps * src)) - (kms * prev))", "box3d", False),
    "accumulation_on_every_cell": ("update_E", "int dsigw\n) {",
                                   "int dsigw_chunk\n) {\n    const int dsigw = 1;",
                                   "box3d", False),
    "plain_constitutive_as_unit_coefficients": (
        "update_E", ": src;\n    f[idx] = value;",
        ": (0.0f + ((1.0f * src) - (0.0f * prev)));\n    f[idx] = value;", "box3d", True),
    "pole_chain_reassociated": ("update_E",
                                "    s_z = s_z - P_Ez_0[idx];\n    s_z = s_z - P_Ez_1[idx];",
                                "    s_z = s_z - (P_Ez_0[idx] + P_Ez_1[idx]);", "box3d", False),
    # the E->P pair and the three-slot recurrence group
    "pair_hands_d_across_the_seam": ("pair", "float w = src_x;", "float w = s_x;", "box3d", False),
    "three_slot_drive_f_w_on_every_cell": ("three_slot", _SELECT, "drive_w[idx]", "box3d", False),
}

#: The row a random state cannot catch: the plain constitutive written as the
#: accumulation with ``kps = 1, kms = 0`` and zeroed inputs differs only on ``D = -0``.
PLANTED_ONLY = ("plain_constitutive_as_unit_coefficients",)


def _defect_legs(ade, name: str):
    leg, find, replace, _case, _plant = DEFECTS[name]

    def rewrite(text: str) -> str:
        assert text.count(find) >= 1, (name, find)
        return text.replace(find, replace)

    if leg == "update_P":
        texts = {True: rewrite(ade._update_P_pml_real_kernel_code),
                 False: rewrite(ade._update_P_pml_real_uniform_kernel_code)}
        return _array_update_p, lambda d: _run_update_p(ade, d, texts)
    if leg == "update_E":
        def run(d):
            counts = dispersive_kernels.pole_counts_of(
                dispersive_kernels.resolve_pole_plan(d.fields))
            _run_update_e(d, rewrite(dispersive_kernels.dispersive_source("pml", counts)))
        return _array_update_e, run
    if leg == "pair":
        return _array_e_then_p, lambda d: _run_polarization_pair(
            d, lambda t: t.replace(find, replace) if find in t else t)
    return _array_update_p, lambda d: _run_three_slot_trailing(d, rewrite)


@pytest.mark.parametrize("name", sorted(DEFECTS))
def test_each_planted_defect_is_caught(name, ade):
    _leg, _find, _replace, case, plant = DEFECTS[name]
    oracle_leg, kernel_leg = _defect_legs(ade, name)
    _start, oracle, got, _inside = _compare(case, plant, oracle_leg, kernel_leg)
    assert any(_differing(oracle, got).values()), f"{name} changed no word on {case}"


@pytest.mark.parametrize("name", PLANTED_ONLY)
def test_the_planted_only_rows_are_invisible_to_a_random_state(name, ade):
    """Why the -0 plant is load-bearing: on a random state the row is a null."""
    _leg, _find, _replace, case, _plant = DEFECTS[name]
    oracle_leg, kernel_leg = _defect_legs(ade, name)
    _start, oracle, got, _inside = _compare(case, False, oracle_leg, kernel_leg)
    assert not any(_differing(oracle, got).values()), name
