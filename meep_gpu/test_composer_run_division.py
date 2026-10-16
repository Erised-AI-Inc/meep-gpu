"""A composer handed no chunk division reads the division the run steps with.

THE ROW. ``TestHoleyWvgBands.test_fields_at_kx`` of MEEP's test suite: 2-D, complex
storage, a Bloch phase on x (k = 3.5), an odd mirror on y with a layer on the folded
axis's high face, and an off-diagonal ``chi1inv`` row (Ex/Ey) that is nonzero only
near the fold plane. MEEP's own division of that run (the lift installs it,
``Fields.chunk_layout``) puts the layer's chunk one row lower than the package's
one-rank rule does (``chunk_map.one_rank_line`` documents the departure on a folded
axis with a high-face layer), so the two records differ by one row on each lattice.

THE DISAGREEMENT IT CAUSED. The arms' off-diagonal presence clauses read the array
path's presence through ``stepping._chunk_layout``, which is the run's own division,
while a composer handed no division read the one-rank one. On this row the presence
then matched no word of the one-rank record (the Metal arm: "2020 of 2040" cells of
one chunk code keep the array) and the CUDA arm refused the division as no product the
record names, so the census, which hands no division in, recorded ``update_E``
unselected while the fused D/E products' predicates admitted the row, and a driver,
which hands its own division in, composed and served it. Each composer now resolves
``None`` to the division the fields carry, else the one-rank default, so the census
composes what a driver composes.

What is pinned here, on the row's configuration built without MEEP (the boxes and
flags the lift recorded):

1. the run's record is not the one-rank record (the row's mechanism);
2. Metal: ``launch.chunk_division`` handed none reads the run's division, and the
   presence words build against its record and refuse against the one-rank record
   with the measured text (the planted control: the clause is alive); on the Apple GPU
   the composer handed none selects ``update_E`` and, offered the product, installs the
   fused pair over both slots;
3. CUDA: the composer handed none selects the folded complex D curl and the complex
   off-diagonal ``update_E``, the two halves the folded complex off-diagonal fused
   product conjoins, and that product's predicate admits the row; handed the one-rank
   division explicitly it refuses ``update_E`` by name;
4. Triton: the composer handed none judges the run's division;
5. the same class under the run's own division: the CUDA E->P weld's predicate asks the
   composer's chunk-division clause, so it no longer admits a run whose ``update_E`` arm
   the composer refuses (``examples/absorbed_power_density.py`` on the 1.0 census).

NOTHING HERE LAUNCHES, and only the Metal composer's test compiles (it is the one test
that needs the Apple GPU). The CUDA composer is asked through a grid whose array module
answers ``cupy``, as the census asks it.
"""

from __future__ import annotations

from typing import Any

import numpy
import pytest

from .chunk_map import ChunkLayout, chunk_bounds_record, grid_frames
from .driver import FdtdDriver
from .grid import Mirror

#: The lifted row's own division (``ChunkLayout.from_meep`` on the row, read back with
#: ``to_record``): the layer's chunk on the folded y axis from doubled 198, the
#: interior chunk below it. The one-rank rule's lines start the layer's run at stored
#: row 102 (integer lattice) and 101 (half-integer); MEEP's at 101 and 100.
ROW_BOXES = (((-20, 198, 0), (20, 240, 0)), ((-20, -2, 0), (20, 198, 0)))
ROW_FLAGS = ((False, True, False), (False, False, False))
ROW_FACES = ((0.0, 0.0), (0.0, 20.0), (0.0, 0.0))

#: The Metal presence words the row's arms bind on its own record, measured on the
#: lifted row: the Ex/Ey and Ey/Ex slots keep chunk code 13 (the interior chunk,
#: ``1 + 3*1 + 9*1``) only; every other slot is dead and reads all kept.
ROW_PRESENCE = (1 << 13, (1 << 27) - 1, (1 << 27) - 1, 1 << 13, (1 << 27) - 1,
                (1 << 27) - 1)

#: The CUDA arms the folded complex off-diagonal fused D/E product conjoins.
CUDA_CURL_ARM = "folded complex"
CUDA_UPDATE_E_ARM = "complex off-diagonal PML"


class _CupyNamed:
    """The grid's array module, reporting ``cupy`` as its name and nothing else."""

    def __init__(self, real: Any) -> None:
        object.__setattr__(self, "_real", real)

    __name__ = "cupy"

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)


class _CupyNamedGrid:
    """The grid with ``xp`` reporting ``cupy``; every other read forwarded."""

    def __init__(self, grid: Any) -> None:
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "xp", _CupyNamed(grid.xp))

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_grid"), name)


def _row_driver() -> FdtdDriver:
    """The row's configuration, with its own division installed as a lift installs it."""
    driver = FdtdDriver(cell_size=(1.0, 12.0, 0.0), resolution=20.0, dimensions=2,
                        symmetry=(Mirror("Y", -1),), k_point=(3.5, 0.0, 0.0),
                        force_complex_fields=True)
    shape = tuple(int(n) for n in driver.grid.shape)
    epsilon = {name: numpy.full(shape, 13.0, numpy.float32) for name in ("Ex", "Ey", "Ez")}
    row = numpy.zeros(shape, numpy.float32)
    row[3:9, 0:8, :] = numpy.float32(0.0125)  # near the fold plane only, as the lift's
    driver.set_epsilon_components(
        epsilon, chi1inv_offdiagonal={"Ex": {"Ey": row}, "Ey": {"Ex": row.copy()}})
    driver.set_chunk_layout(ChunkLayout(ROW_BOXES, ROW_FLAGS, grid_frames(driver.grid),
                                        faces=ROW_FACES, source="meep"))
    driver.setup_pml({"x": (0.0, 0.0), "y": (0.0, 20.0)})
    return driver


@pytest.fixture
def row():
    driver = _row_driver()
    try:
        yield driver
    finally:
        driver.close()


def test_the_rows_own_record_is_not_the_one_rank_record(row):
    own = row.fields.chunk_layout
    assert own is not None and own.source == "meep"
    one_rank = ChunkLayout.one_rank(row.grid, row.pml)
    own_words = [int(v) for v in chunk_bounds_record(own)]
    one_words = [int(v) for v in chunk_bounds_record(one_rank)]
    assert own_words != one_words
    # The layer's run on the folded y axis, integer then half-integer lattice.
    assert (own_words[4], own_words[16]) == (101, 100)
    assert (one_words[4], one_words[16]) == (102, 101)


def test_the_metal_composer_handed_none_reads_the_runs_division(row):
    from .metal_kernels import chunk_bounds, folded_complex, launch  # noqa: PLC0415
    from .metal_kernels import offdiag_update_e  # noqa: PLC0415

    fields, pml, grid = row.fields, row.pml, row.grid
    shape = tuple(int(n) for n in grid.shape)
    layout, words, division = launch.chunk_division(fields, pml, None)
    assert division == ()
    assert layout is fields.chunk_layout
    assert numpy.array_equal(words, chunk_bounds.for_layout(fields.chunk_layout, shape))

    codes, _reasons = folded_complex.folded_axis_kinds(grid, pml)
    assert offdiag_update_e.presence_for(fields, pml, codes, words) == ROW_PRESENCE

    # The planted control: the one-rank record, handed over explicitly, is refused by
    # the clause, with the measured count, so the clause is alive and only the
    # default moved.
    one_rank = ChunkLayout.one_rank(grid, pml)
    _layout, one_words, _division = launch.chunk_division(fields, pml, one_rank)
    with pytest.raises(offdiag_update_e.PresenceRefusal,
                       match=r"chi1inv_offdiag\[Ex\]\[Ey\].*\(2020 of 2040 do\)"):
        offdiag_update_e.presence_for(fields, pml, codes, one_words)
    reasons = offdiag_update_e.presence_reasons(fields, pml, codes, one_words)
    assert reasons and reasons[0].startswith("the off-diagonal presence is refused: ")


#: The Metal arms the folded complex off-diagonal fused D/E product conjoins, and its label.
METAL_UPDATE_E_ARM = "complex folded off-diagonal PML E"
METAL_PRODUCT = "folded complex off-diagonal fused electric D/E pair"


@pytest.mark.requires_resource("mps_device")
def test_the_metal_composer_selects_the_rows_update_e_and_its_fused_pair(row, monkeypatch):
    """On the device: the Metal composer handed none selects ``update_E`` on the row, and
    with the product offered it installs the fused pair over both slots; handed the
    one-rank division explicitly it refuses ``update_E`` on the presence clause.

    Under ``flush``, the one float32 subnormal policy the MPS executor attains (every Metal
    arm refuses ``keep`` by name)."""
    from conftest import requires_resource_skip  # noqa: PLC0415

    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")

    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        requires_resource_skip("mps_device", "PyTorch is not importable")
    if not getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        requires_resource_skip("mps_device", "this PyTorch reports no usable Apple GPU (MPS)")
    from .metal_kernels import device, launch  # noqa: PLC0415

    fields, pml = row.fields, row.pml
    plan = launch.plan_step(fields, pml, residency=device.Residency(), sources=())
    assert plan.selected.get("update_E") == METAL_UPDATE_E_ARM, plan.reasons.get("update_E")
    fused = launch.plan_step(fields, pml, residency=device.Residency(), sources=(),
                             fuse=True, fuse_labels=(METAL_PRODUCT,))
    assert (fused.selected.get("step_D"), fused.selected.get("update_E")) == (
        METAL_PRODUCT, METAL_PRODUCT), (fused.selected, fused.reasons)

    refused = launch.plan_step(fields, pml, residency=device.Residency(), sources=(),
                               chunk_layout=ChunkLayout.one_rank(row.grid, pml))
    assert "update_E" not in refused.selected
    assert any(reason.startswith(f"{METAL_UPDATE_E_ARM}: the off-diagonal presence is "
                                 "refused: ") and "(2020 of 2040 do)" in reason
               for reason in refused.reasons.get("update_E", ())), refused.reasons.get("update_E")


def test_the_cuda_composer_and_the_fused_product_agree_on_the_row(row):
    from .cuda_kernels import arms  # noqa: PLC0415
    from .cuda_kernels import folded_complex_offdiag_fused_electric_pair as product  # noqa: PLC0415

    fields, pml = row.fields, row.pml
    grid = _CupyNamedGrid(row.grid)
    plan = arms.plan_step(fields, pml, grid, sources=())
    assert plan.chunk_division["source"] == "meep", plan.chunk_division
    assert plan.chunk_division["record"] == chunk_bounds_record(fields.chunk_layout).tolist()
    assert plan.selected.get("step_D") == CUDA_CURL_ARM, plan.reasons.get("step_D")
    assert plan.selected.get("update_E") == CUDA_UPDATE_E_ARM, plan.reasons.get("update_E")
    covered, reason = product.covers_folded_complex_offdiag_fused_electric_pair(
        fields, pml, grid, ())
    assert covered is True, reason

    # The planted control: handed the one-rank division, the composer refuses the arm
    # by name on the record it was handed.
    refused = arms.plan_step(fields, pml, grid, sources=(),
                             chunk_layout=ChunkLayout.one_rank(row.grid, pml))
    assert "update_E" not in refused.selected
    assert any(reason.startswith(f"{CUDA_UPDATE_E_ARM}: the chunk division is not a "
                                 "{low, interior, high} product along y")
               for reason in refused.reasons.get("update_E", ())), refused.reasons.get("update_E")


def test_the_triton_composer_handed_none_judges_the_runs_division(row, monkeypatch):
    from .triton_kernels import launch  # noqa: PLC0415

    seen = []
    judge = launch.chunk_layout_reasons

    def recording(fields: Any, layout: Any, **kwargs: Any):
        seen.append(layout)
        return judge(fields, layout, **kwargs)

    monkeypatch.setattr(launch, "chunk_layout_reasons", recording)
    launch.plan_step(row.fields, row.pml, sources=())
    assert seen == [row.fields.chunk_layout]


def test_a_driver_without_a_lift_still_reads_the_one_rank_default():
    """Fields that carry no division: every composer's default is unchanged."""
    from .cuda_kernels import arms  # noqa: PLC0415
    from .metal_kernels import launch  # noqa: PLC0415

    driver = FdtdDriver(cell_size=(1.0, 12.0, 0.0), resolution=20.0, dimensions=2,
                        symmetry=(Mirror("Y", -1),), k_point=(3.5, 0.0, 0.0),
                        force_complex_fields=True)
    try:
        driver.setup_pml({"x": (0.0, 0.0), "y": (0.0, 20.0)})
        assert driver.fields.chunk_layout is None
        layout, words, division = launch.chunk_division(driver.fields, driver.pml, None)
        assert division == () and layout.source == "one_rank"
        assert [int(v) for v in chunk_bounds_record(layout)] == [
            int(v) for v in chunk_bounds_record(ChunkLayout.one_rank(driver.grid, driver.pml))]
        plan = arms.plan_step(driver.fields, driver.pml, _CupyNamedGrid(driver.grid),
                              sources=())
        assert plan.chunk_division["source"] == arms.ONE_RANK_DEFAULT_SOURCE
    finally:
        driver.close()


# ---------------------------------------------------------------------------
# The same class under the run's own division: the CUDA E->P product
# ---------------------------------------------------------------------------

def test_the_dispersive_offdiag_polarization_pair_asks_the_composers_chunk_clause():
    """The CUDA E->P weld admitted ``examples/absorbed_power_density.py`` on the 1.0
    census while the composer refused the ``update_E`` arm it implements there, on the
    run's own division ("not a {low, interior, high} product along x"), so the board
    credited a weld the composer never installs. The predicate now asks the arm's
    clause: on the weld's own corpus-shaped fixture, walled so the one-rank record names
    its chunks, it admits; with a division whose record cannot name its chunks along x
    installed (an interior cut, as a lift installs one), it refuses with the composer's
    own text."""
    from .cuda_kernels import arms  # noqa: PLC0415
    from .cuda_kernels import dispersive_offdiag_fused_polarization_pair as weld  # noqa: PLC0415
    from .cuda_kernels.chunk_coverage import CHUNK_AXES_OFFDIAG  # noqa: PLC0415
    from .cuda_kernels.test_chunk_binding import _layout, _product  # noqa: PLC0415
    from .cuda_kernels.test_dispersive_offdiag_polarization_pair import (  # noqa: PLC0415
        WALLED, _build)
    from .cuda_kernels.test_fused_pairs import _NumpyWearingCupysName  # noqa: PLC0415

    fields, layer, grid = _build(_NumpyWearingCupysName(), boundaries=WALLED)
    covered, why = weld.covers_dispersive_offdiag_fused_polarization_pair(
        fields, layer, grid, ())
    assert covered, why
    split = _product(_layout(fields, layer, grid), extra_cuts={0: (0,)})
    fields.install_chunk_layout(split)
    covered, why = weld.covers_dispersive_offdiag_fused_polarization_pair(
        fields, layer, grid, ())
    assert covered is False
    clause = arms.StepContext(fields, layer, grid, chunk_layout=split
                              ).chunk_division_refusal(CHUNK_AXES_OFFDIAG)
    assert clause is not None and why == clause, (why, clause)
    assert "not a {low, interior, high} product along x" in why
    # The composer, handed nothing, refuses the arm the weld implements by that text.
    plan = arms.plan_step(fields, layer, grid, sources=())
    assert "update_E" not in plan.selected
    assert f"dispersive off-diagonal: {why}" in plan.reasons.get("update_E", ()), (
        plan.reasons.get("update_E"))
