"""The hand-CUDA table and the run's chunk division: the clause and the binding.

MEEP selects the PML form of every curl and constitutive update per structure chunk
(step_db.cpp:56-59, update_eh.cpp:162-163), and a CUDA kernel reads the division as
one ``int32[25]`` record (``chunk_map.chunk_bounds_record``). What is pinned here:

1. **The clause** (``chunk_coverage``): every arm refuses a division no record
   describes, by the record builder's own reason; the off-diagonal and cylindrical
   families also refuse one whose record does not name their chunks along the axes
   they index; a configuration a family refuses keeps the family's reason; the clause
   module imports nothing, as ``coverage.py`` does; and each clause is load-bearing
   (deleting it moves a verdict).
2. **The default**: a composition handed no division reads the one-rank division of
   its grid and PML, which is the driver's own default record, and says so.
3. **The binding**: the record is uploaded once per composition, and every plan of
   that composition, slot or fused, binds the same vector; a refused division binds
   none.
4. **The record**: the dispatch record's CUDA row carries the division the table read.

NOTHING HERE COMPILES OR LAUNCHES: the grid is NumPy wearing CuPy's ``__name__``, this
directory's standing stand-in.
"""

from __future__ import annotations

import itertools
import json
import pathlib

import numpy
import pytest

from .. import fastpath_cuda
from ..chunk_map import (RECORD_WORDS, ChunkLayout, chunk_bounds_record,
                         record_index)
from ..driver import FdtdDriver
from ..grid import Mirror
from . import arms, chunk_coverage, fused_pairs, registry
from .test_arms import _NumpyWearingCupysName, build
from .test_cylindrical_pml_real import build as build_cylindrical

METALLIC = ("metallic", "metallic", "metallic")
PERIODIC = ("periodic", "periodic", "periodic")


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def _layout(fields, layer, grid, chunk_layout=None):
    """The division a composition over this triple reads."""
    context = arms.StepContext(fields, layer, grid, chunk_layout=chunk_layout)
    return context._layout()  # noqa: SLF001


def _intervals(layout, axis):
    """``[(lo, hi, flag), ...]`` along one axis of a product division."""
    frame = layout.frames[axis]
    if frame.invariant:
        return [(frame.origin, frame.origin, False)]
    cuts = layout.boundaries(axis)
    assert cuts is not None, f"axis {axis} has local cuts"
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        flags = {box_flags[axis] for (box_lo, box_hi), box_flags
                 in zip(layout.boxes, layout.pml_flags)
                 if (box_lo[axis], box_hi[axis]) == (lo, hi)}
        assert len(flags) == 1, (axis, lo, hi, flags)
        out.append((lo, hi, flags.pop()))
    return out


def _product(layout, extra_cuts=None, flip=None, source="meep"):
    """The division's {per-axis interval} product, rebuilt as a lifted one would be.

    ``extra_cuts`` maps an axis to doubled coordinates at which to cut an interval in
    two (both pieces keep its flag); ``flip`` is ``(box_index, axis)``: that one box's
    flag on that axis is inverted, which makes the division non-separable there.
    """
    per_axis = []
    for axis in range(3):
        pieces = []
        for lo, hi, flag in _intervals(layout, axis):
            cuts = sorted(c for c in (extra_cuts or {}).get(axis, ()) if lo < c < hi)
            edges = [lo, *cuts, hi]
            pieces.extend((a, b, flag) for a, b in zip(edges[:-1], edges[1:]))
        per_axis.append(pieces)
    boxes, flags = [], []
    for px, py, pz in itertools.product(*per_axis):
        boxes.append(((px[0], py[0], pz[0]), (px[1], py[1], pz[1])))
        flags.append([px[2], py[2], pz[2]])
    if flip is not None:
        index, axis = flip
        flags[index][axis] = not flags[index][axis]
    return ChunkLayout(boxes, flags, layout.frames, faces=layout.faces, source=source)


def _declaring(plan):
    return getattr(plan, "absorbed_by", None) or plan


# --------------------------------------------------------------------------
# 1. THE CLAUSE MODULE
# --------------------------------------------------------------------------

def test_the_clause_module_pulls_in_no_device_dependency():
    """``chunk_coverage.py`` is held to ``coverage.py``'s import rule, by the same check."""
    from .test_constitutive_pml_real import STDLIB_ONLY, device_free_import_report

    source = (pathlib.Path(__file__).parent / "chunk_coverage.py").read_text(encoding="utf-8")
    report = device_free_import_report(source)
    assert report["absolute"] <= STDLIB_ONLY, report
    assert report["relative"] == [] and report["dynamic"] == set(), report


def test_the_record_layout_is_chunk_maps():
    """The transcribed word count and index arithmetic equal ``chunk_map``'s."""
    assert chunk_coverage.CHUNK_RECORD_WORDS == RECORD_WORDS
    record = numpy.arange(100, 100 + RECORD_WORDS, dtype=numpy.int32)
    for half, axis in itertools.product((False, True), range(3)):
        assert chunk_coverage.chunk_record_words(record, half, axis) == tuple(
            int(record[record_index(half, axis, word)]) for word in range(4))


def test_the_owner_label_is_the_run_formula():
    words = (0, 4, 12, 16)
    labels = [chunk_coverage.chunk_owner_label(words, row) for row in range(16)]
    assert labels == [0] * 4 + [1] * 8 + [2] * 4
    assert [chunk_coverage.chunk_owner_label((0, 0, 0, 0), row) for row in range(3)] == [1, 1, 1]
    # One to one: three intervals, three labels.
    owners = [0] * 4 + [1] * 8 + [2] * 4
    assert chunk_coverage.chunk_product_problem(words, owners) is None
    # A row whose interval the record labels as another chunk's.
    assert "chunk interval 2 reads more than one record label" in \
        chunk_coverage.chunk_product_problem(words, [2] + owners[1:])
    # Two intervals the record cannot tell apart.
    assert "both read record label 1" in chunk_coverage.chunk_product_problem(
        words, [0] * 4 + [1] * 4 + [3] * 4 + [2] * 4)
    # Rows with no requirement (walls) are not judged.
    assert chunk_coverage.chunk_product_problem(words, [None] + owners[1:]) is None


# --------------------------------------------------------------------------
# 2. THE DEFAULT
# --------------------------------------------------------------------------

def test_the_one_rank_default_is_read_recorded_and_admitted(xp):
    """No division handed in: the one-rank one of this grid and PML, named, admitted."""
    fields, layer, grid = build(xp, boundaries=METALLIC)
    plan = arms.plan_step(fields, layer, grid)
    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    block = plan.chunk_division
    assert block["source"] == arms.ONE_RANK_DEFAULT_SOURCE
    assert block["refused_because"] is None
    assert block["not_a_chunk_product_along"] == []
    assert block["record"] == chunk_bounds_record(ChunkLayout.one_rank(grid, layer)).tolist()
    json.dumps(block)


def test_the_default_is_the_drivers_own_record():
    """``FdtdDriver.chunk_bounds_record`` without a lift is the record the composer reads."""
    driver = FdtdDriver(cell_size=(8.0, 8.0, 0.0), resolution=2.0, dimensions=2,
                        boundaries={"x": "metallic", "y": "metallic"})
    try:
        driver.setup_pml({"x": 3, "y": 3})
        context = arms.StepContext(driver.fields, driver.pml, driver.grid)
        assert context.chunk_facts()["record"] == driver.chunk_bounds_record().tolist()
        assert context.chunk_facts()["source"] == arms.ONE_RANK_DEFAULT_SOURCE
    finally:
        driver.close()


def test_a_handed_in_division_is_the_one_read(xp):
    fields, layer, grid = build(xp, boundaries=METALLIC)
    lifted = _product(_layout(fields, layer, grid))
    plan = arms.plan_step(fields, layer, grid, chunk_layout=lifted)
    assert plan.chunk_division["source"] == "meep"
    assert plan.chunk_division["record"] == chunk_bounds_record(lifted).tolist()
    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")


def test_a_division_for_another_grid_is_refused_by_name(xp):
    fields, layer, grid = build(xp, boundaries=METALLIC)
    other_fields, other_layer, other_grid = build(xp, boundaries=METALLIC,
                                                  cell=(10.0, 8.0, 8.0))
    foreign = ChunkLayout.one_rank(other_grid, other_layer)
    plan = arms.plan_step(fields, layer, grid, chunk_layout=foreign)
    assert plan.replaces == ()
    assert "built for another grid" in plan.chunk_division["refused_because"]
    assert any("built for another grid" in reason for reason in plan.reasons["step_B"])


# --------------------------------------------------------------------------
# 3. THE REFUSALS
# --------------------------------------------------------------------------

def test_a_non_separable_division_is_refused_by_every_arm_and_every_product(xp):
    """R12: no record describes it, so no CUDA kernel steps any of it."""
    fields, layer, grid = build(xp, boundaries=METALLIC)
    one_rank = _layout(fields, layer, grid)
    admitted = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    assert admitted.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert admitted.selected["step_B"] == "fused magnetic pair"

    mixed = _product(one_rank, flip=(0, 0))
    assert sorted(mixed.non_separable()) == [(0, False), (0, True)]
    plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True, chunk_layout=mixed)
    assert plan.replaces == ()
    assert "not separable along x" in plan.chunk_division["refused_because"]
    assert plan.chunk_division["record"] is None
    for slot in ("step_B", "update_H", "step_D", "update_E"):
        assert any("not separable along x" in reason for reason in plan.reasons[slot]), (
            slot, plan.reasons[slot])
    for family in ("cuda_fused_magnetic_pair", "cuda_fused_electric_pair"):
        reason, = plan.reasons[f"fused_pair_{family}"]
        assert "not separable along x" in reason, reason
    # A configuration the family refuses keeps the family's own reason.
    assert any("no-PML null: an active PML layer is installed" in reason
               for reason in plan.reasons["update_H"]), plan.reasons["update_H"]


def test_an_interior_cut_refuses_only_the_chunk_indexed_families(xp):
    """Separable, same record, but four chunk intervals along x: presence can't be read."""
    from .test_offdiag_constitutive_pml_real import install_rows

    fields, layer, grid = build(xp, boundaries=METALLIC)
    install_rows(fields, grid, (1, 0, 0, 1, 0, 0), seed=20260927)
    one_rank = _layout(fields, layer, grid)
    assert arms.plan_step(fields, layer, grid).selected["update_E"] == "off-diagonal"

    cut = _product(one_rank, extra_cuts={0: (0,)})
    assert cut.is_separable()
    assert chunk_bounds_record(cut).tolist() == chunk_bounds_record(one_rank).tolist()
    plan = arms.plan_step(fields, layer, grid, chunk_layout=cut)
    assert plan.selected["step_B"] == plan.selected["step_D"] == "PML"
    assert plan.selected["update_H"] == "ordinary"
    assert "update_E" not in plan.selected
    assert any(reason.startswith("off-diagonal: the chunk division is not a {low, "
                                 "interior, high} product along x")
               for reason in plan.reasons["update_E"]), plan.reasons["update_E"]
    assert plan.chunk_division["refused_because"] is None
    assert plan.chunk_division["not_a_chunk_product_along"] == ["x"]


def test_the_cylindrical_curl_refuses_a_cut_along_r_and_admits_one_along_z(xp):
    fields, layer, grid = build_cylindrical(xp)
    one_rank = _layout(fields, layer, grid)
    assert arms.plan_step(fields, layer, grid).selected["step_B"] == "cylindrical"

    along_r = _product(one_rank, extra_cuts={0: (10,)})
    plan = arms.plan_step(fields, layer, grid, chunk_layout=along_r)
    assert "step_B" not in plan.selected and "step_D" not in plan.selected
    assert any(reason.startswith("cylindrical: the chunk division is not a {low, "
                                 "interior, high} product along x")
               for reason in plan.reasons["step_B"]), plan.reasons["step_B"]
    assert plan.selected["update_H"] == "ordinary"

    along_z = _product(one_rank, extra_cuts={2: (0,)})
    plan = arms.plan_step(fields, layer, grid, chunk_layout=along_z)
    assert plan.selected["step_B"] == plan.selected["step_D"] == "cylindrical"


def test_a_layer_on_a_wrapping_axis_refuses_the_off_diagonal_arm(xp):
    """The wrapped cell 0 joins the low run while its chunk is the high one.

    KERNELS §2's refusal, inherited: the one-rank record merges the stored cell 0 (the
    periodic image of the high wall plane, owned by the high chunk) into the low run,
    so a presence mask indexed off the record would read the low chunk's presence
    there. Curl and constitutive arms read membership only, which the record holds.
    """
    from .test_offdiag_constitutive_pml_real import install_rows

    fields, layer, grid = build(xp, boundaries=PERIODIC)
    install_rows(fields, grid, (1, 0, 0, 1, 0, 0), seed=20260927)
    plan = arms.plan_step(fields, layer, grid)
    assert plan.selected["step_B"] == plan.selected["step_D"] == "PML"
    assert "update_E" not in plan.selected
    reason = next(r for r in plan.reasons["update_E"] if r.startswith("off-diagonal: "))
    assert "chunk interval 2 reads more than one record label (row 0 reads 0" in reason
    assert plan.chunk_division["not_a_chunk_product_along"] == ["x", "y", "z"]


def _shipped_problem_without(check):
    """``chunk_product_problem`` with one of its two checks deleted, otherwise verbatim."""
    def problem(words, owners):
        labels = {}
        for row, owner in enumerate(owners):
            if owner is not None:
                labels.setdefault(int(owner), set()).add(
                    chunk_coverage.chunk_owner_label(words, row))
        if check != "one label per interval":
            if any(len(values) > 1 for values in labels.values()):
                return "an interval reads more than one label"
        if check != "distinct labels":
            seen = set()
            for owner in sorted(labels):
                label = next(iter(labels[owner]))
                if label in seen:
                    return "two intervals share a label"
                seen.add(label)
        return None

    return problem


@pytest.mark.parametrize("mutation", [
    "record clause dropped", "product clause dropped",
    "one label per interval", "distinct labels"])
def test_every_clause_is_load_bearing(xp, monkeypatch, mutation):
    """Each clause, deleted, moves at least one verdict on a fixed battery.

    The battery: the one-rank default (admitted), a non-separable division (the record
    clause), an interior cut along x (distinct labels), a layer on both faces of a
    wrapping axis (either check), and a layer on the HIGH face only of a wrapping axis
    (one label per interval: the wrapped cell 0 reads the low run's label 0 while its
    chunk is the high one, and no other interval reads 0).
    """
    from ..pml import PML
    from .test_offdiag_constitutive_pml_real import install_rows

    def battery():
        verdicts = []
        fields, layer, grid = build(xp, boundaries=METALLIC)
        install_rows(fields, grid, (1, 0, 0, 1, 0, 0), seed=20260927)
        one_rank = _layout(fields, layer, grid)
        for layout in (None, _product(one_rank, flip=(0, 0)),
                       _product(one_rank, extra_cuts={0: (0,)})):
            verdicts.append(tuple(sorted(
                arms.plan_step(fields, layer, grid, chunk_layout=layout).selected.items())))
        fields, layer, grid = build(xp, boundaries=PERIODIC)
        install_rows(fields, grid, (1, 0, 0, 1, 0, 0), seed=20260927)
        verdicts.append(tuple(sorted(arms.plan_step(fields, layer, grid).selected.items())))
        high_only = PML(grid=grid, thickness=((0, 2), (0, 2), (0, 2)))
        verdicts.append(tuple(sorted(
            arms.plan_step(fields, high_only, grid).selected.items())))
        return verdicts

    shipped = battery()
    assert shipped[0] != shipped[1] and "update_E" not in dict(shipped[4]), shipped
    original_refusal = chunk_coverage.chunk_division_refusal
    if mutation == "record clause dropped":
        monkeypatch.setattr(chunk_coverage, "chunk_division_refusal",
                            lambda facts, axes=(): original_refusal(
                                {**facts, "record_refusal": None}, axes))
    elif mutation == "product clause dropped":
        monkeypatch.setattr(chunk_coverage, "chunk_division_refusal",
                            lambda facts, axes=(): original_refusal(facts, ()))
    else:
        monkeypatch.setattr(chunk_coverage, "chunk_product_problem",
                            _shipped_problem_without(mutation))
    assert battery() != shipped, f"{mutation} deleted: no verdict moved"


def test_the_two_checks_reproduce_the_shipped_function():
    """The mutants above are deletions from THIS function, so with nothing deleted the
    hand copy answers as the shipped one on every line the battery asks."""
    both = _shipped_problem_without(None)
    cases = [((0, 4, 12, 16), [0] * 4 + [1] * 8 + [2] * 4),
             ((0, 4, 12, 16), [2] + [0] * 3 + [1] * 8 + [2] * 4),
             ((0, 1, 13, 16), [2] + [1] * 12 + [2] * 3),
             ((0, 4, 12, 16), [0] * 4 + [1] * 4 + [3] * 4 + [2] * 4)]
    for words, owners in cases:
        assert (both(words, owners) is None) == (
            chunk_coverage.chunk_product_problem(words, owners) is None), (words, owners)


# --------------------------------------------------------------------------
# 4. THE DERIVATION AGREES WITH THE LAYOUT'S OWN
# --------------------------------------------------------------------------

@pytest.mark.parametrize("boundaries,symmetry", [
    (METALLIC, ()), (PERIODIC, ()), (METALLIC, (Mirror("Y", 1),)),
    (PERIODIC, (Mirror("X", 1),)), (("periodic", "metallic", "periodic"), ())])
def test_the_interval_owners_agree_with_the_membership_line(xp, boundaries, symmetry):
    """Every owned (and ghost) row's interval carries exactly the row's membership.

    Asked of the division rebuilt from the one-rank boxes (``source="meep"``), whose
    lines ``chunk_map`` derives from the boxes by MEEP's ``owns``. The one-rank lines
    themselves are the package's rule, which departs from those boxes on folded
    periodic axes by design (``test_chunk_map.py`` pins where).
    """
    fields, layer, grid = build(xp, boundaries=boundaries, symmetry=symmetry)
    layout = _product(_layout(fields, layer, grid))
    compared = 0
    for axis, half in itertools.product(range(3), (False, True)):
        owners = arms._chunk_interval_owners(layout, axis, half)  # noqa: SLF001
        line = layout.membership(axis, half)
        flags = [flag for _lo, _hi, flag in _intervals(layout, axis)]
        for row, owner in enumerate(owners):
            if owner is None:
                continue
            assert flags[owner] == bool(line[row]), (axis, half, row, owner)
            compared += 1
    assert compared > 0


# --------------------------------------------------------------------------
# 5. THE BINDING
# --------------------------------------------------------------------------

def test_the_record_is_uploaded_once_and_bound_by_every_plan(xp):
    fields, layer, grid = build(xp, boundaries=METALLIC)
    plan = arms.plan_step(fields, layer, grid, sources=(), fuse=True)
    context = plan._context  # noqa: SLF001
    assert context.chunk_uploads == 0, "nothing uploads at plan time"
    bound = {id(_declaring(slot_plan).chunk_bounds) for slot_plan in plan.plans.values()}
    assert len(bound) == 1 and context.chunk_uploads == 1
    device = context.chunk_bounds_device()
    assert device.dtype == numpy.int32 and device.shape == (RECORD_WORDS,)
    assert device.tolist() == plan.chunk_division["record"]
    owners = {type(_declaring(p)).__name__ for p in plan.plans.values()}
    assert owners == {"CudaFusedPairPlan"}
    unfused = arms.plan_step(fields, layer, grid)
    assert {type(p).__name__ for p in unfused.plans.values()} == {"CudaSlotPlan"}
    again = {id(p.chunk_bounds) for p in unfused.plans.values()}
    assert len(again) == 1 and again != bound, "a new composition uploads its own"


def test_a_refused_division_or_a_missing_context_binds_no_record(xp):
    fields, layer, grid = build(xp, boundaries=METALLIC)
    mixed = _product(_layout(fields, layer, grid), flip=(0, 0))
    context = arms.StepContext(fields, layer, grid, chunk_layout=mixed)
    with pytest.raises(RuntimeError, match="no chunk record for this composition: "
                                           "the chunk division is not separable along x"):
        context.chunk_bounds_device()
    assert context.chunk_uploads == 0
    with pytest.raises(RuntimeError, match="carries no composition context"):
        arms.CudaSlotPlan("probe", "step_B", "PML", "x").chunk_bounds
    with pytest.raises(RuntimeError, match="carries no composition context"):
        fused_pairs.CudaFusedPairPlan("probe", "p", "k", (), ("step_B", "update_H"),
                                      None).chunk_bounds


# --------------------------------------------------------------------------
# 6. THE DISPATCH RECORD AND THE TABLE'S DECLARATIONS
# --------------------------------------------------------------------------

def test_the_dispatch_record_carries_the_division_the_table_read(xp):
    fields, layer, grid = build(xp, boundaries=METALLIC)
    plan = arms.plan_step(fields, layer, grid)
    record = {"tables": {"cuda": {"candidate": True, "refused_because": None}}}
    fastpath_cuda.merge_tables([("cuda", plan)], pending_of={"cuda": {}}, record=record)
    assert record["tables"]["cuda"]["chunk_division"] == plan.chunk_division
    json.dumps(record["tables"]["cuda"])
    bare = {"tables": {"cuda": {"candidate": True, "refused_because": None}}}
    fastpath_cuda.merge_tables([("cuda", arms.CudaStepPlan({}, {}, {}))],
                               pending_of={"cuda": {}}, record=bare)
    assert "chunk_division" not in bare["tables"]["cuda"]


def test_every_chunk_indexed_family_declares_the_axes_it_indexes():
    """Off-diagonal rows index every axis; cylindrical curls index r; nothing else does."""
    def expected(name):
        if "offdiag" in name:
            return chunk_coverage.CHUNK_AXES_OFFDIAG
        if "cyl" in name:
            return chunk_coverage.CHUNK_AXES_CYLINDRICAL
        return ()

    declared = 0
    for row in registry._TABLE:  # noqa: SLF001
        if row["family"] == "cuda_cylindrical" or row["family"] == "cuda_cyl_complex":
            assert set(row["slots"]) <= {"step_B", "step_D"}, row
        assert tuple(row.get("chunk_axes", ())) == expected(row["family"]), row["family"]
        declared += bool(row.get("chunk_axes"))
    for family, product in fused_pairs.FUSED_PRODUCTS.items():
        assert tuple(product.get("chunk_axes", ())) == expected(product["module"]), family
        declared += bool(product.get("chunk_axes"))
    assert declared == 7 + 11, declared
