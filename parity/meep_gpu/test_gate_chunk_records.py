"""The harness's two chunk records: each is a record the Triton plans accept, and says what it names.

``gate_chunk_records`` is how a gate or probe states which division its reference
steps with when it builds a Triton plan from bare arrays (the harness route has no
default). Host-only; no device, no Triton.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE.parents[1])):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate_chunk_records as records  # noqa: E402
from meep_gpu.chunk_map import (  # noqa: E402
    RECORD_PRESENCE_WORD,
    ChunkLayout,
    chunk_bounds_record,
    membership_from_record,
)
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.triton_kernels.launch import bind_chunk_bounds  # noqa: E402


@pytest.mark.parametrize("shape", [(6, 5, 4), (64, 64, 1), (1, 1, 32), (9, 1, 7)])
def test_the_every_cell_record_puts_every_resolved_cell_in_a_layer_chunk(shape):
    record = records.every_cell_record(shape)
    host, _pointer = bind_chunk_bounds(record, shape, np.zeros(1, np.float32), "the test")
    assert np.array_equal(host, record)
    for axis, n in enumerate(shape):
        assert bool(int(record[RECORD_PRESENCE_WORD]) >> axis & 1) is (n > 1)
        for half in (False, True):
            line = membership_from_record(record, half, axis, n)
            assert line.tolist() == [n > 1] * n


def test_the_every_cell_record_refuses_a_shape_that_is_not_three_axes():
    with pytest.raises(ValueError, match="three axes"):
        records.every_cell_record((8, 8))


def test_the_layer_record_is_the_one_rank_record_of_that_layer():
    grid = Grid(resolution=20.0, cell_size=(10.0, 10.0, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic"})
    layer = PML(grid, thickness={"x": 20, "y": 20})
    record = records.layer_record(grid, layer)
    assert np.array_equal(record, chunk_bounds_record(ChunkLayout.one_rank(grid, layer)))
    bind_chunk_bounds(record, grid.shape, np.zeros(1, np.float32), "the test")
    # No absorber: no cell in a layer chunk, partners present on x and y only.
    bare = records.layer_record(grid, None)
    assert int(bare[RECORD_PRESENCE_WORD]) == 0b011
    assert not bare[:RECORD_PRESENCE_WORD].any()


def test_the_region_plan_record_is_the_division_the_oracle_stepped_with():
    """A gate whose oracle is ``stepping`` hands its kernel leg the division of the plan
    that oracle installed; with no plan installed it is refused by name."""
    import types

    from meep_gpu import stepping
    from meep_gpu.driver import FdtdDriver

    with pytest.raises(RuntimeError, match="no region plan"):
        records.region_plan_record(types.SimpleNamespace(region_plan=None))
    driver = FdtdDriver(cell_size=(6.0, 5.0, 0.0), resolution=10.0, dimensions=2,
                        boundaries={"x": "metallic", "y": "metallic"},
                        force_complex_fields=False, prefer_gpu=False)
    try:
        driver.setup_pml({"x": 8, "y": 8})
        stepping.update_E(driver.fields, driver.pml)
        record = records.region_plan_record(driver.fields)
        assert record.dtype == np.int32 and record.flags.c_contiguous
        assert np.array_equal(record, driver.chunk_bounds_record())
        assert np.array_equal(record, records.layer_record(driver.grid, driver.pml))
    finally:
        driver.close()
