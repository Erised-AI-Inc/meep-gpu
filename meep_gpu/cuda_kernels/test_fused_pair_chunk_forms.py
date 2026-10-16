"""The CUDA real fused pairs transcribe the array path's per-chunk PML forms, read off their text.

WHAT THIS FILE SETTLES, ON A HOST WITH NO DEVICE. The fused magnetic pair
(``fused_magnetic_pair``: ``step_B`` -> the two B fills -> ``zero_metal_B`` -> ``update_H``)
and the fused electric pair (``fused_electric_pair``: ``step_D`` -> the D fills ->
``zero_metal_D`` -> ``update_E``) are spliced from the root kernels, which choose MEEP's PML
form per structure chunk from the run's chunk record (bit_parity_0_10 DESIGN.md D1, D5,
D10, D11). Each emitted device text is EXECUTED in float32 NumPy by the root kernels' own
interpreter (``test_root_chunk_forms.Interpreter``, extended here to stores at an index
other than the lane's own, which the fill carries make) over a seeded full-volume state,
and every array of the ``Fields`` is compared word for word (sign of zero included) with the
array path's own five passes, in driver order, on a copy of the same state:

* the one-rank division on walled, periodic and two-dimensional grids;
* a lifted division whose record differs from the one-rank one;
* folded grids -- one, two and three mirrored axes, both terminations, both parities, a
  fold beside a wall, and a layer deep enough that the near ghost's own coefficient and
  membership row are observable -- where the fill carries run;
* the run's first ``update_H`` (``Fields.h_separated`` False: MEEP's start of H, which the
  magnetic launch runs on the array path right after its kernel,
  ``fused_magnetic_pair.start_of_h``) and a mid-run state;
* planted ``-0`` words.

Every case asserts a nonzero cell count on and off the membership run for each component it
should, and that the sub-steps moved their fields (a vacuity floor). A set of planted
defects, each a rewrite of one emitted text, must change a word; the declared nulls must
not.

THE INTERPRETER IS THE ROOT LANE'S, with one extension: a store's index is evaluated rather
than required to be the lane's own (``Bx[gx_n_i] = gx_n_v;``, and the certified helpers
called at a ghost index). Lanes execute in lockstep, so a cross-lane read-after-write that
a device would race on is not modelled here: that property is the write enumeration's
(``test_fused_magnetic_pair_fill_carry.py``).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Dict, Tuple

import numpy
import pytest

from .. import stepping
from ..chunk_map import (ChunkLayout, RegionPlan, chunk_bounds_record, grid_frames,
                         region_backend)
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import fused_electric_pair, fused_magnetic_pair, in_seam_coverage
from .test_root_chunk_forms import (F32, Interpreter, Unmodelled, _NumpyWearingCupysName,
                                    _Pointer, _Register)

HERE = pathlib.Path(__file__).parent


# =============================================================================
# The emitted texts, on a host with no CuPy
# =============================================================================

@pytest.fixture(scope="module")
def certified() -> Dict[str, Any]:
    """The certified root modules, loaded by path under private names, ``cupy`` stubbed
    only for the load (this directory's standing pattern)."""
    previous = sys.modules.get("cupy")
    sys.modules["cupy"] = _NumpyWearingCupysName()
    loaded = {}
    try:
        for stem in ("step_curl_kernels", "constitutive_kernels", "in_seam_passes"):
            spec = importlib.util.spec_from_file_location(
                f"_fused_chunk_forms_{stem}", HERE / f"{stem}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[stem] = module
    finally:
        if previous is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = previous
    return loaded


class _Installed:
    """Install the certified modules on both families for the duration of a block."""

    def __init__(self, certified: Dict[str, Any]):
        self.certified = certified
        self.held = []

    def __enter__(self):
        for family in (fused_magnetic_pair, fused_electric_pair):
            for name, module in self.certified.items():
                self.held.append((family, name, getattr(family, name)))
                setattr(family, name, module)
        return self

    def __exit__(self, *exc):
        for family, name, module in reversed(self.held):
            setattr(family, name, module)


@pytest.fixture(scope="module")
def texts(certified) -> Dict[str, str]:
    with _Installed(certified):
        return {"B": fused_magnetic_pair.fused_magnetic_pair_source(),
                "D": fused_electric_pair.fused_electric_pair_source()}


# =============================================================================
# The interpreter, with stores at any index
# =============================================================================

class ScatterInterpreter(Interpreter):
    """The root interpreter, storing at the EVALUATED index of each statement.

    A fill carry stores at a ghost (``Bx[gx_n_i] = gx_n_v;``) and calls the certified
    constitutive helper there, where the helper's ``idx`` is the ghost's. Each active lane
    writes the word its index names; distinct lanes name distinct words in a race-free
    launch, which the write enumeration asserts separately.
    """

    def _store(self, frame: Dict[str, Any], target: str, index: str, value: Any) -> None:
        pointer = self._lookup(target, frame)
        active = numpy.asarray(self._active(frame))
        if isinstance(pointer, _Register):
            pointer.frame[pointer.name] = numpy.where(active, value,
                                                      pointer.frame[pointer.name])
            return
        if not isinstance(pointer, _Pointer):
            raise Unmodelled(f"store to a non-pointer {target}")
        stored = numpy.asarray(value)
        assert stored.dtype == F32, (target, stored.dtype)
        where = numpy.asarray(self.value(index, frame))
        lanes = numpy.broadcast_shapes(active.shape, where.shape, stored.shape)
        active = numpy.broadcast_to(active, lanes)
        where = numpy.broadcast_to(where, lanes)[active]
        stored = numpy.broadcast_to(stored, lanes)[active]
        if where.size and (where.min() < 0 or where.max() >= pointer.array.size):
            raise AssertionError(f"{target}[{index}] stores outside the volume")
        pointer.array[where] = stored


# =============================================================================
# The cases
# =============================================================================

#: name -> (cell, dimensions, boundaries, symmetry, layer thickness, divide)
CASES: Dict[str, Tuple] = {
    "walls3d": ((9.0, 10.0, 11.0), 3, ("metallic", "metallic", "metallic"), (), 2, None),
    "periodic3d": ((9.0, 10.0, 11.0), 3, ("periodic", "metallic", "periodic"), (), 2, None),
    "slab2d": ((9.0, 10.0, 0.0), 2, ("metallic", "metallic", "periodic"), (),
               {"x": 2, "y": 2}, None),
    "split_y3d": ((9.0, 12.0, 8.0), 3, ("metallic", "metallic", "metallic"), (), 2,
                  "split_y"),
    "fold_y_periodic": ((9.0, 10.0, 11.0), 3, ("periodic", "periodic", "periodic"),
                        (("Y", 1),), 2, None),
    "fold_y_metallic": ((9.0, 10.0, 11.0), 3, ("periodic", "metallic", "periodic"),
                        (("Y", 1),), 2, None),
    "fold_y_wall_z": ((9.0, 10.0, 11.0), 3, ("periodic", "periodic", "metallic"),
                      (("Y", 1),), 2, None),
    "fold_xy_mixed": ((9.0, 10.0, 11.0), 3, ("periodic", "periodic", "periodic"),
                      (("X", 1), ("Y", -1)), 2, None),
    "fold_xyz": ((9.0, 10.0, 11.0), 3, ("periodic", "periodic", "periodic"),
                 (("X", 1), ("Y", -1), ("Z", 1)), 2, None),
    "fold_x_deep": ((9.0, 10.0, 11.0), 3, ("periodic", "periodic", "periodic"),
                    (("X", 1),), ((0, 5), (2, 2), (2, 2)), None),
    # The same grid on a lifted division (the one-rank boxes, read the way a lift's
    # are): its membership line copies the near ghost's image row onto the ghost row
    # (chunk_map, D5), where the one-rank line (chunk_map.one_rank_line) does not.
    "fold_x_deep_lifted": ((9.0, 10.0, 11.0), 3, ("periodic", "periodic", "periodic"),
                           (("X", 1),), ((0, 5), (2, 2), (2, 2)), "lifted"),
}

#: The state each fused pair reads and writes, and every other array a step carries:
#: all of them are compared, so a pair that wrote outside its own set is seen.
STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
         "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _split_y(grid: Grid, pml: PML) -> ChunkLayout:
    """The one-rank boxes with the low-y effort volume cut one cell short of its edge;
    the sliver carries no y layer (past the sigma extent), so its rows leave the run."""
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


def _lifted(grid: Grid, pml: PML) -> ChunkLayout:
    """The one-rank boxes and flags as a lift reads them: membership from the boxes, with
    each ghost row taking its image row's flag."""
    one_rank = ChunkLayout.one_rank(grid, pml)
    return ChunkLayout(one_rank.boxes, one_rank.pml_flags, grid_frames(grid),
                       faces=pml.thickness_by_face, source="meep")


def build(case: str, plant_negative_zero: bool, h_separated: bool, seed: int = 7):
    """``(fields, grid, pml)``: a seeded full-volume state on ``case``'s grid."""
    cell, dimensions, boundaries, symmetry, layer, divide = CASES[case]
    grid = Grid(resolution=1.0, cell_size=cell, dimensions=dimensions,
                boundaries=boundaries,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in symmetry),
                xp=numpy, courant=0.5)
    fields = Fields(grid=grid)
    fields.enable_pml_storage()
    rng = numpy.random.default_rng(seed)
    shape = tuple(int(n) for n in grid.shape)
    epsilon = rng.uniform(1.0, 4.0, size=shape).astype(F32)
    fields.set_isotropic_epsilon_volume(epsilon, (1.0 / epsilon).astype(F32))
    folded = {"XYZ".index(axis) for axis, _phase in symmetry}
    if isinstance(layer, int) and folded:
        layer = tuple((0, layer) if axis in folded else (layer, layer) for axis in range(3))
    pml = PML(grid=grid, thickness=layer)
    if divide == "split_y":
        fields.install_region_plan(RegionPlan(_split_y(grid, pml), grid, pml,
                                              backend=region_backend(numpy)))
    elif divide == "lifted":
        fields.install_region_plan(RegionPlan(_lifted(grid, pml), grid, pml,
                                              backend=region_backend(numpy)))
    for name in STATE:
        values = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
        if plant_negative_zero:
            values[rng.uniform(size=shape) < 0.25] = F32(-0.0)
        getattr(fields, name)[...] = values
    fields.h_separated = bool(h_separated)
    return fields, grid, pml


def _snapshot(fields) -> Dict[str, numpy.ndarray]:
    return {name: numpy.array(getattr(fields, name), copy=True) for name in STATE}


def _restore(fields, state) -> None:
    for name, values in state.items():
        getattr(fields, name)[...] = values


def _bits(array) -> numpy.ndarray:
    return numpy.ascontiguousarray(array).view(numpy.uint32)


def _flat(fields, name):
    array = getattr(fields, name)
    assert array.flags.c_contiguous and array.dtype == F32
    return array.reshape(-1)


#: Each side: (oracle passes in driver order, curl targets, constitutive targets).
SIDES = {
    "B": (("step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
           "update_H"), ("Bx", "By", "Bz"), ("Hx", "Hy", "Hz")),
    "D": (("step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
           "update_E"), ("Dx", "Dy", "Dz"), ("Ex", "Ey", "Ez")),
}


def oracle(side: str, fields, pml) -> None:
    for name in SIDES[side][0]:
        function = getattr(stepping, name)
        if name in ("step_B", "step_D", "update_H", "update_E"):
            function(fields, pml)
        else:
            function(fields)


def arguments(side: str, certified, fields, grid, pml) -> Dict[str, Any]:
    """The launch's named arguments, resolved the way the family's own launcher does."""
    family = fused_magnetic_pair if side == "B" else fused_electric_pair
    with _Installed(certified):
        tables = (family.fused_magnetic_pair_tables(pml) if side == "B"
                  else family.fused_electric_pair_tables(pml))
        codes = certified["step_curl_kernels"].real_curl_boundary_codes(grid)
        fills = (family.fused_magnetic_pair_fills(grid) if side == "B"
                 else family.fused_electric_pair_fills(grid))
    walls = in_seam_coverage.zero_metal_axes(grid)
    nx, ny, nz = (int(n) for n in grid.shape)
    out: Dict[str, Any] = {"nx": nx, "ny": ny, "nz": nz, "dtdx": F32(grid.dt / grid.dx)}
    for name in family._FIELD_BINDINGS:  # noqa: SLF001 - the launcher's own order
        out[name] = _flat(fields, name)
    if side == "D":
        for axis, array in zip("xyz", family.inverse_epsilon_bindings(fields)):
            out[f"inv_eps_E{axis}"] = numpy.ascontiguousarray(array).reshape(-1)
    for key in family._CURL_TABLE_KEYS:  # noqa: SLF001
        out[key] = numpy.ascontiguousarray(tables["curl"][key]).reshape(-1)
    rename = "kms_int_" if side == "B" else "kms_half_"
    for key in family._CONSTITUTIVE_TABLE_KEYS:  # noqa: SLF001
        name = key.replace("kms_", rename) if key.startswith("kms_") else key
        out[name] = numpy.ascontiguousarray(tables["constitutive"][key]).reshape(-1)
    for axis, letter in enumerate("xyz"):
        out[f"bc_{letter}"] = int(codes[axis])
        out[f"wall_{letter}"] = int(bool(walls[axis]))
        out[f"near_{letter}"] = int(fills["near"][axis])
        out[f"reflect_{letter}"] = int(fills["reflect"][axis])
        out[f"phase_{letter}"] = F32(fills["phase"][axis])
    out["cb"] = numpy.ascontiguousarray(chunk_bounds_record(fields.region_plan.layout),
                                        dtype=numpy.int32)
    return out


def run(side: str, text: str, case: str, plant: bool, h_separated: bool, certified,
        host_start: bool = True):
    """``(start, oracle, kernel, form counts)`` for one text on one case.

    The kernel leg is the launcher's: the text, then -- on the run's first update_H,
    on the magnetic side -- the launcher's own start of H on the array path
    (``fused_magnetic_pair.start_of_h``). ``host_start=False`` drops that start, the
    control that shows it is needed.
    """
    fields, grid, pml = build(case, plant, h_separated)
    start = _snapshot(fields)
    oracle(side, fields, pml)
    want = _snapshot(fields)
    plan = fields.region_plan
    _restore(fields, start)
    fields.h_separated = bool(h_separated)
    ScatterInterpreter(text).launch(arguments(side, certified, fields, grid, pml),
                                    int(numpy.prod(grid.shape)))
    if side == "B" and not h_separated and host_start:
        fused_magnetic_pair.start_of_h(fields, pml)
    got = _snapshot(fields)
    shape = tuple(int(n) for n in grid.shape)
    counts = {}
    _passes, curl_targets, constitutive_targets = SIDES[side]
    for sub_step, components in (("curl", curl_targets),
                                 ("constitutive", constitutive_targets)):
        for component in components:
            inside = numpy.broadcast_to(plan.mask(sub_step, component, numpy), shape)
            counts[component] = (int(inside.sum()), int((~inside).sum()))
    return start, want, got, counts


def _differing(want, got) -> Dict[str, int]:
    return {name: int(numpy.count_nonzero(_bits(want[name]) != _bits(got[name])))
            for name in STATE}


def _selecting_axis(component: str, sub_step: str) -> str:
    """The axis whose membership chooses this component's form: dsigu for a curl, the
    component's own axis for a constitutive."""
    if sub_step == "curl":
        term = next(t for t in (*stepping.B_CURL_TERMS, *stepping.D_CURL_TERMS)
                    if t.target == component)
        return term.dsigu
    return component[1]


#: (case, axis) pairs whose layer covers the whole stored axis, so every component keyed
#: on it takes one form there: the lifted deep fold, where the x layer's chunk spans the
#: halved axis (its line reads the near ghost from its image, and the mirror row 1 lies in
#: the layer's box).
_WHOLE_AXIS_IN_THE_RUN = {("fold_x_deep_lifted", "x")}


def _invariant(case: str, component: str, sub_step: str) -> bool:
    """Is the axis that selects this component's form invariant on this case's grid?"""
    if CASES[case][1] == 3:
        return False
    if sub_step == "curl":
        term = next(t for t in (*stepping.B_CURL_TERMS, *stepping.D_CURL_TERMS)
                    if t.target == component)
        return term.dsigu == "z"
    return component[1] == "z"


# =============================================================================
# 1. The texts compute the array path, word for word
# =============================================================================

PARAMETERS = [(side, case, plant, separated)
              for side in ("B", "D") for case in sorted(CASES)
              for plant in (False, True)
              for separated in ((True, False) if side == "B" else (True,))]


@pytest.mark.parametrize(
    "side,case,plant,separated", PARAMETERS,
    ids=[f"{s}-{c}-{'planted' if p else 'random'}-{'midrun' if h else 'first'}"
         for s, c, p, h in PARAMETERS])
def test_the_fused_text_is_the_array_path_word_for_word(side, case, plant, separated,
                                                        texts, certified):
    start, want, got, counts = run(side, texts[side], case, plant, separated, certified)
    differing = _differing(want, got)
    assert all(v == 0 for v in differing.values()), (side, case, plant, separated, differing)
    _passes, curl_targets, constitutive_targets = SIDES[side]
    for sub_step, components in (("curl", curl_targets),
                                 ("constitutive", constitutive_targets)):
        for component in components:
            inside, outside = counts[component]
            if _invariant(case, component, sub_step):
                assert inside == 0 and outside > 0, (case, component, counts)
            elif (case, _selecting_axis(component, sub_step)) in _WHOLE_AXIS_IN_THE_RUN:
                assert inside > 0 and outside == 0, (case, component, counts)
            else:
                assert inside > 0 and outside > 0, (case, component, counts)
            moved = int(numpy.count_nonzero(_bits(want[component]) != _bits(start[component])))
            assert moved > 0, (side, case, component)


def test_the_start_of_h_is_observable_on_these_states(texts, certified):
    """The first-update cases are not the mid-run cases under another name: the array
    path's start of H changes the words, so a kernel that skipped it would be caught."""
    fields, grid, pml = build("walls3d", False, h_separated=False)
    start = _snapshot(fields)
    oracle("B", fields, pml)
    first = _snapshot(fields)
    _restore(fields, start)
    fields.h_separated = True
    oracle("B", fields, pml)
    later = _snapshot(fields)
    assert sum(_differing(first, later).values()) > 0


def test_the_lifted_division_moves_the_record_off_the_one_rank_record():
    fields, grid, pml = build("split_y3d", False, True)
    lifted = chunk_bounds_record(fields.region_plan.layout)
    default = chunk_bounds_record(ChunkLayout.one_rank(grid, pml))
    assert not numpy.array_equal(lifted, default)


def test_the_first_update_needs_the_launchers_start_of_h(texts, certified):
    """The text alone does not start H: without the launcher's start on the array path
    the run's first update differs from the array path, so the start is load-bearing."""
    start, want, got, _counts = run("B", texts["B"], "walls3d", False, False, certified,
                                    host_start=False)
    assert sum(_differing(want, got).values()) > 0


def test_the_two_deep_fold_divisions_disagree_on_the_near_ghost_row():
    """The pair of cases the near-ghost keying is judged on is not one case twice: on the
    one-rank line the ghost row (stored 0 on x, integer lattice) reads its own flag, which
    differs from its image row's (stored 2); on the lifted division it reads the image's."""
    from ..chunk_map import membership_from_record

    one_rank_fields, grid, pml = build("fold_x_deep", False, True)
    stepping.update_H(one_rank_fields, pml)
    one_rank = numpy.asarray(one_rank_fields.region_plan.layout.membership(0, False))
    lifted_fields, _grid, _pml = build("fold_x_deep_lifted", False, True)
    lifted = numpy.asarray(lifted_fields.region_plan.layout.membership(0, False))
    assert one_rank[0] != one_rank[2], one_rank
    assert lifted[0] == lifted[2], lifted
    assert callable(membership_from_record)


# =============================================================================
# 2. Planted defects change a word; declared nulls do not
# =============================================================================

#: ``name -> (side, find, replace, case, h_separated, must_change)``. Each is a rewrite of
#: one emitted text; ``find`` must match it.
DEFECTS = {
    # The union block's lattices swapped for the constitutive half: update_H reads the
    # half-integer record along x.
    "update_H_reads_the_half_integer_record": (
        "B", "constitutive_apply_pre(Hx, f_w_Hx, idx, b_x, pre_w_x, pre_h_x, kps_x[i], "
             "kms_int_x[i], in_i_x);",
        "constitutive_apply_pre(Hx, f_w_Hx, idx, b_x, pre_w_x, pre_h_x, kps_x[i], "
        "kms_int_x[i], in_h_x);", "walls3d", True, True),
    # The near ghost's membership negated: the ghost takes the other form.
    "near_ghost_membership_negated": (
        "B", "int gx_n_w = (0 >= cbi_x_a0 && 0 < cbi_x_a1) || (0 >= cbi_x_b0 && 0 < cbi_x_b1);",
        "int gx_n_w = !((0 >= cbi_x_a0 && 0 < cbi_x_a1) || (0 >= cbi_x_b0 && 0 < cbi_x_b1));",
        "fold_x_deep", True, True),
    # The carried select keyed at the SOURCE row (stored 2) instead of the ghost's own row
    # (stored 0): on the one-rank line the two differ here (the ghost row reads out, its
    # image in), so the ghost takes the other form.
    "near_ghost_keyed_at_the_source": (
        "B", "int gx_n_w = (0 >= cbi_x_a0 && 0 < cbi_x_a1) || (0 >= cbi_x_b0 && 0 < cbi_x_b1);",
        "int gx_n_w = in_i_x;", "fold_x_deep", True, True),
    # The auxiliary stored on a plain cell at the own cell: C2's contract breaks.
    "electric_auxiliary_stored_on_a_plain_cell": (
        "D", "    if (dsigw) fw[idx] = src;\n", "    fw[idx] = src;\n",
        "walls3d", True, True),
    # The dsigu flag of the D curl taken on the wrong axis.
    "electric_recurrence_flag_on_the_dsig_axis": (
        "D", "kms_z[k], sinv_z[k], in_i_z, own_x);", "kms_z[k], sinv_z[k], in_i_y, own_x);",
        "walls3d", True, True),
    # The far ghost of E with its membership negated: the ghost takes the other form.
    "electric_far_ghost_membership_negated": (
        "D", "int gy_y_w = (ny - 1 >= cbh_y_a0 && ny - 1 < cbh_y_a1) || "
             "(ny - 1 >= cbh_y_b0 && ny - 1 < cbh_y_b1);",
        "int gy_y_w = !((ny - 1 >= cbh_y_a0 && ny - 1 < cbh_y_a1) || "
        "(ny - 1 >= cbh_y_b0 && ny - 1 < cbh_y_b1));", "fold_y_periodic", True, True),
    # --- declared nulls --------------------------------------------------------------
    # The same source keying on the LIFTED division of the same grid: its line copies the
    # image row onto the ghost row (D5), so the two rows read one flag.
    "near_ghost_keyed_at_the_source_on_a_lifted_division": (
        "B", "int gx_n_w = (0 >= cbi_x_a0 && 0 < cbi_x_a1) || (0 >= cbi_x_b0 && 0 < cbi_x_b1);",
        "int gx_n_w = in_i_x;", "fold_x_deep_lifted", True, False),
    # The far ghost keyed at its reflect row: both rows lie in the high run on this
    # fixture (a span of 3 covers stored ny - 3 .. ny - 1), so they read one flag.
    "far_ghost_keyed_at_the_source": (
        "D", "int gy_y_w = (ny - 1 >= cbh_y_a0 && ny - 1 < cbh_y_a1) || "
             "(ny - 1 >= cbh_y_b0 && ny - 1 < cbh_y_b1);",
        "int gy_y_w = in_h_y;", "fold_y_periodic", True, False),
    # The hoisted fu load issued unconditionally: the load is not observable in a word.
    "hoisted_fu_load_unguarded": (
        "B", "float pre_fu_x = in_h_z ? fu_Bx[idx] : 0.0f;", "float pre_fu_x = fu_Bx[idx];",
        "walls3d", True, False),
}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_each_planted_defect_changes_a_word_and_each_null_does_not(defect, texts, certified):
    side, find, replace, case, separated, must_change = DEFECTS[defect]
    text = texts[side]
    assert text.count(find) == 1, (defect, text.count(find))
    mutated = text.replace(find, replace, 1)
    counts = []
    for plant in (False, True):
        _start, want, got, _counts = run(side, mutated, case, plant, separated, certified)
        counts.append(sum(_differing(want, got).values()))
    if must_change:
        assert any(counts), (defect, counts)
    else:
        assert not any(counts), (defect, counts)


# =============================================================================
# 3. The text's shape: one record decode, flags on every carried statement
# =============================================================================

@pytest.mark.parametrize("side", ("B", "D"))
def test_the_record_is_decoded_once_over_both_halves(side, texts):
    from . import bounds_block

    family = fused_magnetic_pair if side == "B" else fused_electric_pair
    text = texts[side]
    body = text.split('extern "C" __global__ void ', 1)[1]
    block = family.union_bounds_block()
    assert body.count("cb[") == block.count("cb["), side
    assert body.count(bounds_block.DECODE_END + "\n" + block) == 1, side
    assert block.count("cb[") == 4 * 6 + 1


@pytest.mark.parametrize("side", ("B", "D"))
def test_every_constitutive_statement_passes_a_membership_flag(side, texts):
    """Own cell and every ghost: seven arguments, the last a flag."""
    body = texts[side].split('extern "C" __global__ void ', 1)[1]
    calls = [line.strip() for line in body.splitlines()
             if "constitutive_apply(" in line or "constitutive_apply_pre(" in line]
    assert len(calls) == 3 + 21, len(calls)
    for call in calls:
        last = call[:-2].rsplit(",", 1)[1].strip()
        assert last.startswith(("in_", "g")) and (last.startswith("in_") or last.endswith("_w")), call


def test_the_hoisted_loads_take_the_flags_the_certified_calls_pass(certified):
    with _Installed(certified):
        loads = fused_magnetic_pair.hoisted_loads()
        flags = fused_magnetic_pair.curl_flags()
    for axis, flag in zip("xyz", flags):
        assert f"float pre_fu_{axis} = {flag} ? fu_B{axis}[idx] : 0.0f;" in loads
    for axis, flag in zip("xyz", fused_magnetic_pair.constitutive_flags()):
        assert (f"float pre_w_{axis} = own_{axis} ? ({flag} ? f_w_H{axis}[idx] : 0.0f) "
                f": 0.0f;") in loads
        assert (f"float pre_h_{axis} = own_{axis} ? ({flag} ? H{axis}[idx] : 0.0f) "
                f": 0.0f;") in loads


def test_a_hoisted_helper_that_drops_the_guard_is_refused_by_name(certified):
    """The derivation compares statements under the GUARDED load map, so a certified helper
    whose load lost its flag no longer derives."""
    with _Installed(certified):
        prelude = certified["step_curl_kernels"]._REAL_PML_PRELUDE
        broken = prelude.replace("float fprev = dsigu ? fu[idx] : 0.0f;",
                                 "float fprev = fu[idx];", 1)
        assert broken != prelude
        certified["step_curl_kernels"]._REAL_PML_PRELUDE = broken
        try:
            with pytest.raises(AssertionError):
                fused_magnetic_pair.fused_magnetic_pair_prelude()
        finally:
            certified["step_curl_kernels"]._REAL_PML_PRELUDE = prelude
