"""The four real-storage fold fused pairs run MEEP's folded passes, read off their text.

WHAT THIS FILE SETTLES, ON A HOST WITH NO DEVICE. The BFAST fused pairs
(``bfast_fused_{magnetic,electric}_pair``: one launch for ``step_B`` -> ``zero_metal_B``
-> ``update_H`` and its electric twin) and the special_kz fused pairs
(``special_kz_fused_{magnetic,electric}_pair``: all five passes of the seam, the two mirror
fills carried by the real pairs' ownership inversion) are spliced from the fold singles,
which run MEEP's BFAST and beta passes after each component's curl through their own
additive ladder (bit_parity_0_10 KERNELS.md section 6, step C7). Each emitted device text
is EXECUTED in float32 NumPy by the real fused pairs' interpreter
(``test_fused_pair_chunk_forms.ScatterInterpreter``) over a seeded full-volume state -- the
BFAST state included -- and every array of the ``Fields`` is compared word for word (sign
of zero included) with the array path's own passes, in driver order, on a copy of the same
state: unfolded and folded grids, a lifted division, the run's first ``update_H`` and a
mid-run state, planted ``-0`` words. A set of planted defects, each a rewrite of one
emitted text, must change a word.

The kernel leg is the launcher's: the text, then -- on the run's first ``update_H``, on
the magnetic side -- the launcher's own start of H on the array path
(``fused_magnetic_pair.start_of_h``).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy
import pytest

from .. import stepping
from ..chunk_map import (ChunkLayout, RegionPlan, chunk_bounds_record, grid_frames,
                         region_backend)
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import (bfast_curl, bfast_fused_electric_pair, bfast_fused_magnetic_pair,
               fused_electric_pair, fused_magnetic_pair, in_seam_coverage, special_kz_curl,
               special_kz_fused_electric_pair, special_kz_fused_magnetic_pair)
from .test_fused_pair_chunk_forms import ScatterInterpreter
from .test_fused_pair_chunk_forms import certified  # noqa: F401 - the shared fixture
from .test_root_chunk_forms import F32

#: The four families: ``name -> (module, source emitter, side, fold kind)``.
FAMILIES = {
    "bfast_B": (bfast_fused_magnetic_pair, "bfast_fused_magnetic_pair_source", "B",
                "bfast"),
    "bfast_D": (bfast_fused_electric_pair, "bfast_fused_electric_pair_source", "D",
                "bfast"),
    "kz_B": (special_kz_fused_magnetic_pair, "special_kz_fused_magnetic_pair_source", "B",
             "beta"),
    "kz_D": (special_kz_fused_electric_pair, "special_kz_fused_electric_pair_source", "D",
             "beta"),
}


class _Installed:
    """The certified CuPy-bound modules installed, for the duration of a block, on every
    module whose emitter reads them (the families and the two real pairs they lift)."""

    def __init__(self, certified: Dict[str, Any]):
        self.certified = certified
        self.held = []

    def __enter__(self):
        modules = [entry[0] for entry in FAMILIES.values()]
        modules += [fused_magnetic_pair, fused_electric_pair]
        for module in modules:
            for name, value in self.certified.items():
                if hasattr(module, name):
                    self.held.append((module, name, getattr(module, name)))
                    setattr(module, name, value)
        return self

    def __exit__(self, *exc):
        for module, name, value in reversed(self.held):
            setattr(module, name, value)


@pytest.fixture(scope="module")
def texts(certified) -> Dict[str, str]:  # noqa: F811 - the fixture above
    with _Installed(certified):
        return {name: getattr(module, emitter)()
                for name, (module, emitter, _side, _kind) in FAMILIES.items()}


# =============================================================================
# The cases
# =============================================================================

def _split_y(grid: Grid, pml: PML) -> ChunkLayout:
    """The one-rank boxes with the low-y effort volume cut two cells short of its edge;
    the sliver carries no y layer, so its rows leave the run (the record moves)."""
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


#: ``name -> (kind, cell, dimensions, boundaries, symmetry, layer, divide, fold value)``.
#: ``fold value`` is ``bfast_scaled_k`` for a BFAST case and ``beta`` for a beta one.
CASES: Dict[str, Tuple] = {
    "bfast_walls3d": ("bfast", (9.0, 10.0, 11.0), 3, ("metallic",) * 3, (), 2, None,
                      (0.3, 0.2, 0.15)),
    "bfast_periodic3d": ("bfast", (9.0, 10.0, 11.0), 3,
                         ("periodic", "metallic", "periodic"), (), 2, None,
                         (0.31, -0.27, 0.22)),
    # z invariant: one-partner components in MEEP's plain branch on the interior.
    "bfast_invariant2d": ("bfast", (10.0, 9.0, 0.0), 2, ("periodic", "metallic", "periodic"),
                          (), {"x": 2, "y": 2}, None, (0.35, 0.21, 0.0)),
    "bfast_split_y3d": ("bfast", (9.0, 12.0, 8.0), 3, ("metallic",) * 3, (), 2, "split_y",
                        (0.25, 0.3, -0.2)),
    "kz_slab2d": ("beta", (9.0, 10.0, 0.0), 2, ("metallic", "metallic", "periodic"), (),
                  {"x": 2, "y": 2}, None, 0.3321611318837033),
    "kz_periodic2d": ("beta", (10.0, 9.0, 0.0), 2, ("periodic", "metallic", "periodic"), (),
                      {"x": 2, "y": 2}, None, -0.39),
    "kz_split_y2d": ("beta", (9.0, 12.0, 0.0), 2, ("metallic", "metallic", "periodic"), (),
                     {"x": 2, "y": 2}, "split_y", 0.2),
    "kz_fold_y_periodic": ("beta", (9.0, 10.0, 0.0), 2, ("periodic",) * 3, (("Y", 1),),
                           ((2, 2), (0, 2), (0, 0)), None, 0.2),
    "kz_fold_y_metallic": ("beta", (9.0, 10.0, 0.0), 2, ("periodic", "metallic", "periodic"),
                           (("Y", -1),), ((2, 2), (0, 2), (0, 0)), None, -0.6850526103319672),
    "kz_fold_x_periodic": ("beta", (10.0, 9.0, 0.0), 2, ("periodic", "metallic", "periodic"),
                           (("X", -1),), ((0, 2), (2, 2), (0, 0)), None, 0.45),
}

BASE_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
              "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
              "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")
BFAST_STATE = ("f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
               "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz")


def _state(fields) -> Tuple[str, ...]:
    return BASE_STATE + tuple(n for n in BFAST_STATE if getattr(fields, n, None) is not None)


def build(case: str, plant_negative_zero: bool, h_separated: bool, seed: int = 9,
          without_fold_term: bool = False):
    """``(fields, grid, pml)``: a seeded full-volume state on ``case``'s grid
    (``without_fold_term``: the same grid and state with beta or BFAST off)."""
    kind, cell, dimensions, boundaries, symmetry, layer, divide, value = CASES[case]
    extra = {"bfast_scaled_k": tuple(value)} if kind == "bfast" else {"beta": float(value)}
    if without_fold_term:
        extra = {}
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=dimensions,
                boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in symmetry),
                xp=numpy, courant=0.5, **extra)
    fields = Fields(grid=grid)
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon = rng.uniform(1.0, 4.0, size=shape).astype(F32)
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(F32))
    pml = PML(grid=grid, thickness=layer)
    if divide == "split_y":
        fields.install_region_plan(RegionPlan(_split_y(grid, pml), grid, pml,
                                              backend=region_backend(numpy)))
    for name in _state(fields):
        values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
        if plant_negative_zero:
            values[rng.uniform(size=shape) < 0.25] = F32(-0.0)
        getattr(fields, name)[...] = values
    fields.h_separated = bool(h_separated)
    return fields, grid, pml


def _snapshot(fields) -> Dict[str, numpy.ndarray]:
    return {name: numpy.array(getattr(fields, name), copy=True) for name in _state(fields)}


def _restore(fields, state) -> None:
    for name, values in state.items():
        getattr(fields, name)[...] = values


def _bits(array) -> numpy.ndarray:
    return numpy.ascontiguousarray(array).view(numpy.uint32)


def _differing(want, got) -> Dict[str, int]:
    return {name: int(numpy.count_nonzero(_bits(want[name]) != _bits(got[name])))
            for name in want}


def _flat(fields, name):
    array = getattr(fields, name)
    assert array.flags.c_contiguous and array.dtype == F32
    return array.reshape(-1)


PASSES = {
    "B": ("step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
          "update_H"),
    "D": ("step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
          "update_E"),
}


def oracle(side: str, fields, pml) -> None:
    for name in PASSES[side]:
        function = getattr(stepping, name)
        if name in ("step_B", "step_D", "update_H", "update_E"):
            function(fields, pml)
        else:
            function(fields)


def arguments(family: str, certified, fields, grid, pml) -> Dict[str, Any]:  # noqa: F811
    """The launch's named arguments, resolved the way the family's own launcher does."""
    module, _emitter, side, kind = FAMILIES[family]
    curl_tables = certified["step_curl_kernels"].real_pml_curl_tables(pml, side == "B")
    constitutive_tables = certified["constitutive_kernels"].real_constitutive_tables(
        pml, side == "D")
    codes = certified["step_curl_kernels"].real_curl_boundary_codes(grid)
    walls = in_seam_coverage.zero_metal_axes(grid)
    nx, ny, nz = (int(n) for n in grid.shape)
    out: Dict[str, Any] = {"nx": nx, "ny": ny, "nz": nz, "dtdx": F32(grid.dt / grid.dx)}
    for name in module._FIELD_BINDINGS:  # noqa: SLF001 - the launcher's own order
        key = name.replace("f_bfast_", "fb_")
        out[key] = _flat(fields, name)
    if side == "D":
        for axis, array in zip("xyz", module.inverse_epsilon_bindings(fields)):
            out[f"inv_eps_E{axis}"] = numpy.ascontiguousarray(array).reshape(-1)
    for key in module._CURL_TABLE_KEYS:  # noqa: SLF001
        out[key] = numpy.ascontiguousarray(curl_tables[key]).reshape(-1)
    rename = "kms_int_" if side == "B" else "kms_half_"
    for key in module._CONSTITUTIVE_TABLE_KEYS:  # noqa: SLF001
        name = key.replace("kms_", rename) if key.startswith("kms_") else key
        out[name] = numpy.ascontiguousarray(constitutive_tables[key]).reshape(-1)
    for axis, letter in enumerate("xyz"):
        out[f"bc_{letter}"] = int(codes[axis])
        out[f"wall_{letter}"] = int(bool(walls[axis]))
    if kind == "bfast":
        sub_step = "step_B" if side == "B" else "step_D"
        for (k1, k2), names in zip(bfast_curl.bfast_curl_coefficients(grid, sub_step),
                                   bfast_curl.K_NAMES):
            out[names[0]], out[names[1]] = F32(k1), F32(k2)
    else:
        plus, minus = special_kz_curl.beta_curl_coefficients(grid.beta, grid.dt)
        out["beta_plus"], out["beta_minus"] = F32(plus), F32(minus)
        real = fused_magnetic_pair if side == "B" else fused_electric_pair
        with _Installed(certified):
            fills = (real.fused_magnetic_pair_fills(grid) if side == "B"
                     else real.fused_electric_pair_fills(grid))
        for axis, letter in enumerate("xyz"):
            out[f"near_{letter}"] = int(fills["near"][axis])
            out[f"reflect_{letter}"] = int(fills["reflect"][axis])
            out[f"phase_{letter}"] = F32(fills["phase"][axis])
    out["cb"] = numpy.ascontiguousarray(chunk_bounds_record(fields.region_plan.layout),
                                        dtype=numpy.int32)
    return out


def run(family: str, text: str, case: str, plant: bool, h_separated: bool,
        certified):  # noqa: F811
    """``(start, oracle, kernel)`` for one text on one case."""
    side = FAMILIES[family][2]
    fields, grid, pml = build(case, plant, h_separated)
    start = _snapshot(fields)
    oracle(side, fields, pml)
    want = _snapshot(fields)
    _restore(fields, start)
    fields.h_separated = bool(h_separated)
    ScatterInterpreter(text).launch(arguments(family, certified, fields, grid, pml),
                                    int(numpy.prod(grid.shape)))
    if side == "B" and not h_separated:
        fused_magnetic_pair.start_of_h(fields, pml)
    return start, want, _snapshot(fields)


def _cases_for(family: str) -> Tuple[str, ...]:
    kind = FAMILIES[family][3]
    return tuple(case for case in CASES if CASES[case][0] == kind)


# =============================================================================
# 1. The texts compute the array path, word for word
# =============================================================================

PARAMETERS = [(family, case, plant, separated)
              for family in FAMILIES for case in _cases_for(family)
              for plant in (False, True)
              for separated in ((True, False) if FAMILIES[family][2] == "B" else (True,))]


@pytest.mark.parametrize(
    "family,case,plant,separated", PARAMETERS,
    ids=[f"{f}-{c}-{'planted' if p else 'random'}-{'midrun' if h else 'first'}"
         for f, c, p, h in PARAMETERS])
def test_the_fold_fused_text_is_the_array_path_word_for_word(family, case, plant, separated,
                                                             texts, certified):  # noqa: F811
    start, want, got = run(family, texts[family], case, plant, separated, certified)
    differing = _differing(want, got)
    assert all(v == 0 for v in differing.values()), (
        family, case, plant, separated, {k: v for k, v in differing.items() if v})
    side = FAMILIES[family][2]
    curl = ("Bx", "By", "Bz") if side == "B" else ("Dx", "Dy", "Dz")
    constitutive = ("Hx", "Hy", "Hz") if side == "B" else ("Ex", "Ey", "Ez")
    for component in curl + constitutive:
        moved = int(numpy.count_nonzero(_bits(want[component]) != _bits(start[component])))
        assert moved > 0, (family, case, component)
    if FAMILIES[family][3] == "bfast":
        moved = sum(int(numpy.count_nonzero(_bits(want[f"f_bfast_{c}"])
                                            != _bits(start[f"f_bfast_{c}"])))
                    for c in curl)
        assert moved > 0, (family, case)


def test_the_folded_cases_run_the_fills_and_the_split_moves_the_record():
    for case in ("kz_fold_y_periodic", "kz_fold_y_metallic", "kz_fold_x_periodic"):
        _fields, grid, _pml = build(case, False, True)
        assert grid.has_symmetry(), case
    for case in ("bfast_split_y3d", "kz_split_y2d"):
        fields, grid, pml = build(case, False, True)
        assert not numpy.array_equal(
            chunk_bounds_record(fields.region_plan.layout),
            chunk_bounds_record(ChunkLayout.one_rank(grid, pml))), case


def test_the_beta_pass_reaches_the_output_on_every_beta_case(texts, certified):  # noqa: F811
    """The beta cases are not the plain real pair under another name: zeroing the two
    coefficients changes the words the array path leaves."""
    for case in _cases_for("kz_B"):
        for side in ("B", "D"):
            fields, grid, pml = build(case, False, True)
            oracle(side, fields, pml)
            with_beta = _snapshot(fields)
            plain, _grid, plain_pml = build(case, False, True, without_fold_term=True)
            oracle(side, plain, plain_pml)
            assert sum(_differing(with_beta, _snapshot(plain)).values()) > 0, (case, side)


# =============================================================================
# 2. Planted defects change a word
# =============================================================================

#: ``name -> (family, find, replace, case, h_separated)``; each must change a word.
DEFECTS = {
    # The register captured off the curl alone: update_H reads B before the BFAST pass.
    "bfast_magnetic_register_before_the_pass": (
        "bfast_B", "        b_x = fold_apply_reg(Bx,", "        fold_apply_reg(Bx,",
        "bfast_walls3d", True),
    "bfast_electric_register_before_the_pass": (
        "bfast_D", "        d_y = fold_apply_reg(Dy,", "        fold_apply_reg(Dy,",
        "bfast_periodic3d", True),
    # The register captured off the curl alone on the beta side: the fills and update_H
    # read B (D) before the beta pass.
    "beta_magnetic_register_before_the_pass": (
        "kz_B", "        b_y = fold_apply_reg(By,", "        fold_apply_reg(By,",
        "kz_fold_y_periodic", True),
    "beta_electric_register_before_the_pass": (
        "kz_D", "        d_x = fold_apply_reg(Dx,", "        fold_apply_reg(Dx,",
        "kz_fold_x_periodic", True),
    # The fold ladder's fu stage dropped in the fused twin.
    "fused_ladder_without_the_recurrence": (
        "kz_B", "    if (dsigu) fu[idx] = fu[idx] - df;\n", "", "kz_slab2d", True),
    # The beta increment on an unowned cell left live (the mask dropped). Where the far
    # fill images the cell its field word is overwritten either way, but the ladder's fu
    # stage runs above the ownership guard: Dy's dsigu axis (x) carries a layer on the
    # far fill's plane, so the unmasked increment reaches fu there.
    "beta_increment_unmasked_on_a_fold": (
        "kz_D", "float inc_beta = owned_cell ? (-(beta_minus * f1)) : 0.0f;",
        "float inc_beta = (-(beta_minus * f1));", "kz_fold_y_periodic", True),
    # The BFAST state stored on an unowned cell.
    "bfast_state_on_an_unowned_cell": (
        "bfast_B", "    if (!owned) return 0.0f;\n    F[idx] = updated;",
        "    F[idx] = updated;\n    if (!owned) return 0.0f;", "bfast_walls3d", True),
}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_each_planted_fused_defect_changes_a_word(defect, texts, certified):  # noqa: F811
    family, find, replace, case, separated = DEFECTS[defect]
    text = texts[family]
    assert text.count(find) == 1, (defect, text.count(find))
    mutated = text.replace(find, replace, 1)
    counts = []
    for plant in (False, True):
        _start, want, got = run(family, mutated, case, plant, separated, certified)
        counts.append(sum(_differing(want, got).values()))
    assert any(counts), (defect, counts)


# =============================================================================
# 3. The text's shape
# =============================================================================

@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_the_record_is_decoded_once_and_no_plain_helper_survives(family, texts):
    from . import bounds_block

    module = FAMILIES[family][0]
    text = texts[family]
    prelude, body = text.split('extern "C" __global__ void ', 1)
    side = FAMILIES[family][2]
    real = fused_magnetic_pair if side == "B" else fused_electric_pair
    block = (module.union_bounds_block() if hasattr(module, "union_bounds_block")
             else real.union_bounds_block())
    assert body.count("cb[") == block.count("cb[") == 4 * 6 + 1, family
    assert body.count(bounds_block.DECODE_END + "\n" + block) == 1, family
    # The plain helpers leave the prelude once every call is captured.
    assert "void pml_apply(" not in prelude and "void fold_apply(" not in prelude, family
    assert "pml_apply(" not in body.replace("pml_apply_reg(", ""), family
    assert "fold_apply(" not in body.replace("fold_apply_reg(", ""), family
    folds = 3 if FAMILIES[family][3] == "bfast" else 2
    assert body.count("= fold_apply_reg(") == folds, family
    assert body.count("bfast_pass(") == (3 if FAMILIES[family][3] == "bfast" else 0), family
    assert "bfast_pass(" not in prelude or FAMILIES[family][3] == "bfast", family


@pytest.mark.parametrize("family", ("kz_B", "kz_D"))
def test_the_carrying_pairs_pass_the_ownership_flag_to_both_helpers(family, texts):
    body = texts[family].split('extern "C" __global__ void ', 1)[1]
    register = "b" if FAMILIES[family][2] == "B" else "d"
    for axis in "xyz":
        calls = [line.strip() for line in body.splitlines()
                 if line.strip().startswith(f"{register}_{axis} = ")
                 and ("pml_apply_reg(" in line or "fold_apply_reg(" in line)]
        assert calls and all(call.endswith(f", own_{axis});") for call in calls), (
            family, axis, calls)


# =============================================================================
# 4. The real-beta H->D weld: update_H welded into step_D, scratch output
# =============================================================================

def _real_weld_interpreter():
    """The complex lane's interpreter (it models the weld's argument struct, its
    out-parameters and ``&array[k]``) with the real weld's ``float name[n];`` locals."""
    import re  # noqa: PLC0415

    from .test_complex_chunk_forms import CFArray, ComplexInterpreter  # noqa: PLC0415

    class RealArray(CFArray):
        def __init__(self, size: int):  # noqa: D401 - the items are float32 words
            self.items = [F32(numpy.nan) for _ in range(size)]

    class RealWeldInterpreter(ComplexInterpreter):
        def statement(self, statement: str, frame: Dict[str, Any]) -> None:
            match = re.fullmatch(r"float (\w+)\[(\d+)\];", statement)
            if match:
                frame[match.group(1)] = RealArray(int(match.group(2)))
                return
            if statement == "return;":
                # A void helper's return (the conductive tail's lossless branch): the
                # lanes that reach it run nothing further in the helper.
                if "__done__" not in frame:
                    raise AssertionError("a bare return outside a helper")
                frame["__done__"] = numpy.logical_or(frame["__done__"],
                                                     self._active(frame))
                return
            super().statement(statement, frame)

    return RealWeldInterpreter


def run_hd(case: str, plant: bool, separated: bool, certified, monkeypatch,  # noqa: F811
           text: str = None):
    """``(start, oracle, kernel)`` for the special_kz H->D weld on one case: the array
    path's ``update_H`` then ``step_D`` against one interpreted launch and the rotation."""
    import sys  # noqa: PLC0415

    from . import fused_hd_pair, special_kz_fused_hd_pair as family  # noqa: PLC0415
    from .test_complex_chunk_forms import InterpretedKernel  # noqa: PLC0415

    for name in ("step_curl_kernels", "constitutive_kernels"):
        monkeypatch.setattr(fused_hd_pair, name, certified[name])
    # The launcher's ``from . import step_curl_kernels`` answered by the certified module
    # loaded on NumPy, whether or not this host imports CuPy (the package attribute is
    # what the import statement reads first, sys.modules second).
    import meep_gpu.cuda_kernels as package  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "meep_gpu.cuda_kernels.step_curl_kernels",
                        certified["step_curl_kernels"])
    monkeypatch.setattr(package, "step_curl_kernels", certified["step_curl_kernels"],
                        raising=False)
    fields, grid, pml = build(case, plant, separated)
    start = _snapshot(fields)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    want = _snapshot(fields)
    _restore(fields, start)
    fields.h_separated = bool(separated)
    record = numpy.ascontiguousarray(chunk_bounds_record(fields.region_plan.layout),
                                     dtype=numpy.int32)
    curl = certified["step_curl_kernels"].real_pml_curl_tables(pml, False)
    constitutive = certified["constitutive_kernels"].real_constitutive_tables(pml, False)
    tables = {"kms": {a: curl[f"kms_{a}"] for a in "xyz"},
              "sinv": {a: curl[f"sinv_{a}"] for a in "xyz"},
              "kps": {a: constitutive[f"kps_{a}"] for a in "xyz"}}
    for axis in "xyz":
        assert numpy.array_equal(curl[f"kms_{axis}"], constitutive[f"kms_{axis}"])
    scratch = {name: numpy.empty_like(getattr(fields, name))
               for name in family.SCRATCH_VOLUMES}
    kernel = InterpretedKernel(text or family.kernel_source(),
                               interpreter=_real_weld_interpreter())
    family.launch_special_kz_fused_hd_pair(
        fields, scratch, tables, family.special_kz_fused_hd_pair_codes(grid),
        family.beta_scalars(grid), float(grid.dt / grid.dx), kernel=kernel,
        chunk_bounds=record, layer=pml)
    family.rotate_into_fields(fields, scratch)
    assert kernel.launches == 1
    return start, want, _snapshot(fields)


HD_PARAMETERS = [(case, plant, separated) for case in _cases_for("kz_D")
                 for plant in (False, True) for separated in (True, False)]


@pytest.mark.parametrize("case,plant,separated", HD_PARAMETERS,
                         ids=[f"{c}-{'planted' if p else 'random'}-"
                              f"{'midrun' if h else 'first'}" for c, p, h in HD_PARAMETERS])
def test_the_beta_hd_weld_is_the_array_path_word_for_word(case, plant, separated,
                                                          certified, monkeypatch):  # noqa: F811
    start, want, got = run_hd(case, plant, separated, certified, monkeypatch)
    differing = _differing(want, got)
    assert all(v == 0 for v in differing.values()), (
        case, plant, separated, {k: v for k, v in differing.items() if v})
    for component in ("Dx", "Dy", "Dz", "Hx", "Hy", "Hz"):
        assert int(numpy.count_nonzero(_bits(want[component]) != _bits(start[component])))


def test_a_beta_partner_left_on_the_stale_field_changes_a_word(certified,  # noqa: F811
                                                               monkeypatch):
    """The beta pass consumes the weld's own post-update_H register; reading the
    pre-launch H instead (the gate's ``beta_partner_reads_stale_h``) is caught."""
    from . import fused_hd_pair, special_kz_fused_hd_pair as family  # noqa: PLC0415

    for name in ("step_curl_kernels", "constitutive_kernels"):
        monkeypatch.setattr(fused_hd_pair, name, certified[name])
    text = family.kernel_source()
    find = "float inc_beta = owned_cell ? (-(beta_plus * f2)) : 0.0f;"
    assert text.count(find) == 1
    mutated = text.replace(find, "float inc_beta = owned_cell ? (-(beta_plus * Hy[idx])) "
                                 ": 0.0f;", 1)
    _start, want, got = run_hd("kz_slab2d", False, True, certified, monkeypatch,
                               text=mutated)
    assert sum(_differing(want, got).values()) > 0


# =============================================================================
# 5. The conductive/BFAST H->D weld: both variants
# =============================================================================

#: ``name -> (sigma placement, per-component mask, keep expected all-True)``: a uniform
#: D conductivity on every chunk, on two of three components, and on the high half of x
#: only (the PML chunks of the low half keep none: the ``ck`` mask is bound).
CONDUCTIVE_CASES = {
    "cond_uniform": ("uniform", (True, True, True)),
    "cond_two_of_three": ("uniform", (True, False, True)),
    "cond_half_space": ("half_x", (True, True, True)),
}


def build_conductive(case: str, plant: bool, separated: bool, seed: int = 13):
    """A walled 3-D grid with a PML and a D conductivity, seeded full-volume."""
    placement, mask = CONDUCTIVE_CASES[case]
    grid = Grid(resolution=1.0, cell_size=(10.0, 9.0, 8.0), dimensions=3,
                boundaries=("metallic", "periodic", "metallic"), xp=numpy, courant=0.5)
    fields = Fields(grid=grid)
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon = rng.uniform(1.0, 4.0, size=shape).astype(F32)
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(F32))
    sigma = {}
    for component, live in zip(("Dx", "Dy", "Dz"), mask):
        if not live:
            continue
        values = rng.uniform(0.05, 0.9, size=shape).astype(F32)
        if placement == "half_x":
            values[: shape[0] // 2] = F32(0.0)
        sigma[component] = values
    pml = PML(grid=grid, thickness=2)
    fields.set_d_conductivity(sigma)
    names = BASE_STATE + tuple(f"f_cond_{c}" for c, live in zip(("Dx", "Dy", "Dz"), mask)
                               if live)
    for name in names:
        values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
        if plant:
            values[rng.uniform(size=shape) < 0.25] = F32(-0.0)
        getattr(fields, name)[...] = values
    fields.h_separated = bool(separated)
    return fields, grid, pml, names


def _preprocess(text: str) -> str:
    """The conductive template's preprocessor, resolved before the interpreter reads it:
    ``#if``/``#else``/``#endif`` over the integer ``#define``s, and object-like macros
    with an expression body (``#define KEPT0 (ck0[idx] != 0)``) substituted into the
    lines that follow. Integer ``#define``s stay for the interpreter, which reads them."""
    import re  # noqa: PLC0415

    macros: Dict[str, str] = {}
    stack = []
    active = True
    out = []

    def evaluate(expression: str) -> bool:
        for name, value in macros.items():
            expression = re.sub(rf"\b{name}\b", f"({value})", expression)
        expression = (expression.replace("||", " or ").replace("&&", " and ")
                      .replace("!", " not ").replace(" not =", "!="))
        return bool(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - test text

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#if "):
            taken = evaluate(stripped[len("#if "):]) if active else False
            stack.append((active, taken))
            active = active and taken
            continue
        if stripped == "#else":
            parent, taken = stack[-1]
            active = parent and not taken
            continue
        if stripped == "#endif":
            active, _taken = stack.pop()
            continue
        if not active:
            continue
        match = re.fullmatch(r"#define (\w+) (.+)", stripped)
        if match:
            macros[match.group(1)] = match.group(2).strip()
            if match.group(2).strip().isdigit():
                out.append(line)
            continue
        for name, value in macros.items():
            if not value.isdigit():
                line = re.sub(rf"\b{name}\b", value, line)
        out.append(line)
    assert not stack, "an #if without its #endif"
    return "\n".join(out) + "\n"


def run_weld(variant: str, case: str, plant: bool, separated: bool, certified,  # noqa: F811
             monkeypatch, text: str = None):
    """``(start, oracle, kernel, state)`` for the conductive/BFAST weld on one case."""
    import sys  # noqa: PLC0415

    import meep_gpu.cuda_kernels as package  # noqa: PLC0415

    from . import conductive_bfast_fused_hd_pair as family, fused_hd_pair  # noqa: PLC0415
    from .test_complex_chunk_forms import InterpretedKernel  # noqa: PLC0415

    for name in ("step_curl_kernels", "constitutive_kernels"):
        monkeypatch.setattr(fused_hd_pair, name, certified[name])
        monkeypatch.setitem(sys.modules, f"meep_gpu.cuda_kernels.{name}", certified[name])
        monkeypatch.setattr(package, name, certified[name], raising=False)
    if variant == "bfast":
        fields, grid, pml = build(case, plant, separated)
        names = tuple(_state(fields))
    else:
        fields, grid, pml, names = build_conductive(case, plant, separated)

    def snap():
        return {name: numpy.array(getattr(fields, name), copy=True) for name in names}

    start = snap()
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    want = snap()
    for name, values in start.items():
        getattr(fields, name)[...] = values
    fields.h_separated = bool(separated)
    record = numpy.ascontiguousarray(chunk_bounds_record(fields.region_plan.layout),
                                     dtype=numpy.int32)
    curl = certified["step_curl_kernels"].real_pml_curl_tables(pml, False)
    constitutive = certified["constitutive_kernels"].real_constitutive_tables(pml, False)
    tables = {"kms": {a: curl[f"kms_{a}"] for a in "xyz"},
              "sinv": {a: curl[f"sinv_{a}"] for a in "xyz"},
              "kps": {a: constitutive[f"kps_{a}"] for a in "xyz"}}
    scratch = {name: numpy.empty_like(getattr(fields, name))
               for name in family.SCRATCH_VOLUMES}
    state = family.resolve(fields, grid, pml, variant=variant, tables=tables,
                           scratch=scratch, chunk_bounds=record)
    source = text or family.kernel_source(variant, state["cond"], keep=state["keep"])
    kernel = InterpretedKernel(_preprocess(source), interpreter=_real_weld_interpreter())
    family.launch_conductive_bfast_fused_hd_pair(fields, state, kernel=kernel)
    family.rotate_into_fields(fields, scratch)
    assert kernel.launches == 1
    return start, want, snap(), state


WELD_PARAMETERS = ([("bfast", case, plant, separated) for case in _cases_for("bfast_D")
                    for plant in (False, True) for separated in (True, False)]
                   + [("conductive", case, plant, separated)
                      for case in sorted(CONDUCTIVE_CASES)
                      for plant in (False, True) for separated in (True, False)])


@pytest.mark.parametrize("variant,case,plant,separated", WELD_PARAMETERS,
                         ids=[f"{v}-{c}-{'planted' if p else 'random'}-"
                              f"{'midrun' if h else 'first'}"
                              for v, c, p, h in WELD_PARAMETERS])
def test_the_conductive_bfast_weld_is_the_array_path_word_for_word(
        variant, case, plant, separated, certified, monkeypatch):  # noqa: F811
    start, want, got, state = run_weld(variant, case, plant, separated, certified,
                                       monkeypatch)
    differing = _differing(want, got)
    assert all(v == 0 for v in differing.values()), (
        variant, case, plant, separated, {k: v for k, v in differing.items() if v})
    for component in ("Dx", "Dy", "Dz", "Hx", "Hy", "Hz"):
        assert int(numpy.count_nonzero(_bits(want[component]) != _bits(start[component])))
    if case == "cond_half_space":
        assert not all(state["keep"]), state["keep"]


def test_a_conductive_weld_compiled_as_kept_everywhere_is_caught_on_the_half_space(
        certified, monkeypatch):  # noqa: F811
    """The KEEP triple and the bound mask are load-bearing: the half-space case
    compiled with every target kept (the mask never read) changes a word."""
    from . import conductive_bfast_fused_hd_pair as family  # noqa: PLC0415

    fields, _grid, _pml, _names = build_conductive("cond_half_space", False, True)
    cond = family.conductive_mask(fields)
    text = family.kernel_source("conductive", cond, keep=(True, True, True))
    _start, want, got, state = run_weld("conductive", "cond_half_space", False, True,
                                        certified, monkeypatch, text=text)
    assert not all(state["keep"])
    assert sum(_differing(want, got).values()) > 0
