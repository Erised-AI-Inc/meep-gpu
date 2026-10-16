"""The dispatch planner hands the run's chunk division to both NVIDIA kernel tables.

``FdtdDriver.step`` hands ``plan_fast_path`` its own division
(``FdtdDriver.chunk_layout``), and rung 5 of ``fastpath._decide`` hands that same
object to the Triton composer (``triton_kernels.plan_step``) and to the hand-written
CUDA composer (``cuda_kernels.arms.plan_step``). Each composer resolves ``None`` to
the division the fields carry, else the one-rank division of its grid and PML
(``test_composer_run_division``); handing the division over states which one both
tables bind. What is pinned here:

1. **The hand-over.** Both composers receive the caller's object by identity: from a
   direct ``plan_fast_path`` call, and from a driver's freeze, lifted or not. A direct
   caller that names no division hands both ``None``, the documented default.
2. **The default is not read on the dispatch route.** A driver's dispatched CUDA
   composition names the driver's division as ``tables.cuda.chunk_division.source``
   (``"one_rank"`` for a driver built without MEEP, ``"meep"`` once lifted), never
   ``arms.ONE_RANK_DEFAULT_SOURCE``, and binds the driver's own record. For the lifted
   division used here that record differs from the one-rank record, so the default
   would have bound other words.
3. **Both tables judge the division handed over.** A lifted division that no record
   describes (not separable along y) is refused by name by the Triton table on every
   slot and by every CUDA arm, and the step runs on the array path. The same driver
   without the lift is not refused on its division.

The Metal half of the hand-over (driver to ``plan_fast_path`` to rung 4bM) is pinned in
``test_metal_chunk_bounds``.

NOTHING HERE COMPILES OR LAUNCHES. The NVIDIA rungs are reached on a host whose engine
is NumPy under the name ``cupy`` with a stub Triton (``test_dispatch_contract``'s
doubles). The composers are either recorders or the real planners, which build plans
without a device; a plan the warm pass cannot compile leaves its slot on the array
path, which is what every driver below steps.
"""

from __future__ import annotations

import pytest

import meep_gpu.fastpath as fastpath
from meep_gpu.chunk_map import ChunkLayout, chunk_bounds_record, grid_frames
from meep_gpu.cuda_kernels import arms as cuda_arms
from meep_gpu.driver import FdtdDriver
from meep_gpu.test_chunk_map import _case_a_division
from meep_gpu.test_dispatch_contract import (
    cupy_like_grid,
    install_composer,
    install_cuda_composer,
    numpy_backed_cupy_alias,
    step_plan,
    stub_triton,
)
from meep_gpu.triton_kernels.coverage import NON_SEPARABLE_DIVISION

#: The four slots a PML run's curl and constitutive sub-steps occupy.
PML_SLOTS = ("step_B", "update_H", "step_D", "update_E")

#: ``chunk_map.chunk_bounds_record``'s refusal of the non-separable division below, as
#: the CUDA arms quote it.
NOT_SEPARABLE_ALONG_Y = "not separable along y"


@pytest.fixture(autouse=True)
def _dispatch_reachable(monkeypatch, tmp_path):
    """Reach the composers, with a cold subnormal policy around every test.

    ``meep_gpu/conftest.py`` pins ``MEEP_GPU_DISPATCH=0``; these tests exercise
    dispatch, so they enable it, as ``test_dispatch_contract`` does. Rung 8b installs
    a process-wide subnormal policy, which is reset on both sides of each test, and
    the CuPy cache directory is private to the test.
    """
    from meep_gpu import subnormal_policy

    subnormal_policy._reset_for_tests()  # noqa: SLF001
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None  # noqa: SLF001
    fastpath.reset_dispatch_announcements()
    monkeypatch.delenv(fastpath.SUBNORMAL_INSTALL_SWITCH, raising=False)
    monkeypatch.delenv(subnormal_policy.POLICY_ENV, raising=False)
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy-cache"))
    monkeypatch.delenv(fastpath.FUSED_KILL_SWITCH, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.delenv(fastpath.DISPATCH_LOG, raising=False)
    monkeypatch.delenv(fastpath.WARM_SWITCH, raising=False)
    monkeypatch.delenv(fastpath.KERNEL_TABLE_SWITCH, raising=False)
    monkeypatch.delenv(fastpath.BACKEND_PREFERENCE_SWITCH, raising=False)
    yield
    subnormal_policy._reset_for_tests()  # noqa: SLF001
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None  # noqa: SLF001
    fastpath.reset_dispatch_announcements()


def _lifted(driver, **cut):
    """Case (a)'s division, cut as ``test_chunk_map._case_a_division`` cuts it, as a lift."""
    boxes, flags = _case_a_division(**cut)
    return ChunkLayout(boxes, flags, grid_frames(driver.grid),
                       faces=driver.pml.thickness_by_face, source="meep")


def _nvidia_driver(monkeypatch, **cut):
    """Case (a) (2-D, metallic walls, PML 20 cells on x and y, real storage), planning
    on the NVIDIA ladder: its engine answers ``cupy`` and steps NumPy's arithmetic.

    With ``cut`` the driver steps a lifted division (:func:`_lifted`), installed
    before the first step as a lift installs it.
    """
    driver = FdtdDriver(cell_size=(10.0, 10.0, 0.0), resolution=20.0, dimensions=2,
                        boundaries={"x": "metallic", "y": "metallic"},
                        force_complex_fields=False)
    driver.setup_pml({"x": 20, "y": 20})
    if cut:
        driver.set_chunk_layout(_lifted(driver, **cut))
    monkeypatch.setattr(driver.grid, "xp", numpy_backed_cupy_alias())
    driver.gpu = "cuda"  # The engine answers cupy: a CUDA driver plans.
    return driver


#: A separable lift whose record is not the one-rank record: a y cut at doubled -160
#: leaves the sliver [-160, -158] outside the low-y layer's chunk.
SPLIT_Y = {"split_y": -160}

#: The same cut made inside one x interval only: no per-axis record describes it.
NON_SEPARABLE = {"split_y": -160, "split_x_range": (-158, 158)}


# --------------------------------------------------------------------------
# 1. THE HAND-OVER
# --------------------------------------------------------------------------

def test_a_direct_call_hands_its_division_to_both_nvidia_composers(monkeypatch):
    stub_triton(monkeypatch)
    triton_calls = install_composer(monkeypatch, step_plan())
    cuda_calls = install_cuda_composer(monkeypatch, cuda_arms.CudaStepPlan({}, {}, {}))
    marker = object()
    fastpath.plan_fast_path(object(), object(), cupy_like_grid(), (), chunk_layout=marker)
    assert len(triton_calls) == 1 and len(cuda_calls) == 1, (triton_calls, cuda_calls)
    assert triton_calls[0]["chunk_layout"] is marker
    assert cuda_calls[0]["chunk_layout"] is marker
    # A caller that names no division hands both composers None, which each resolves
    # to the division the fields carry, else the one-rank division of the grid and PML.
    fastpath.plan_fast_path(object(), object(), cupy_like_grid(), ())
    assert triton_calls[-1]["chunk_layout"] is None
    assert cuda_calls[-1]["chunk_layout"] is None


def test_a_drivers_freeze_hands_its_own_division_to_both_nvidia_composers(monkeypatch):
    stub_triton(monkeypatch)
    triton_calls = install_composer(monkeypatch, step_plan())
    cuda_calls = install_cuda_composer(monkeypatch, cuda_arms.CudaStepPlan({}, {}, {}))

    driver = _nvidia_driver(monkeypatch)
    try:
        driver.step()
        default = driver.chunk_layout
        assert default is not None and default.source == "one_rank"
        assert len(triton_calls) == len(cuda_calls) == 1
        assert triton_calls[0]["chunk_layout"] is default
        assert cuda_calls[0]["chunk_layout"] is default
    finally:
        driver.close()

    lifted = _nvidia_driver(monkeypatch, **SPLIT_Y)
    try:
        layout = lifted.chunk_layout
        assert layout.source == "meep"
        lifted.step()
        assert triton_calls[-1]["chunk_layout"] is layout
        assert cuda_calls[-1]["chunk_layout"] is layout
        assert len(triton_calls) == len(cuda_calls) == 2
    finally:
        lifted.close()


# --------------------------------------------------------------------------
# 2. THE DEFAULT IS NOT READ ON THE DISPATCH ROUTE
# --------------------------------------------------------------------------

def test_a_dispatched_cuda_composition_reads_the_drivers_division(monkeypatch):
    """The real CUDA composer, reached through a driver's freeze, names what it read.

    ``tables.cuda.chunk_division`` is written while the tables merge, before any
    later rung refuses, so it is on the record whether or not the plan dispatches.
    """
    stub_triton(monkeypatch)
    install_composer(monkeypatch, step_plan())  # Triton holds no slot.
    blocks = {}
    for name, cut in (("one_rank", {}), ("meep", SPLIT_Y)):
        driver = _nvidia_driver(monkeypatch, **cut)
        try:
            driver.step()
            block = driver.fast_path_report()["tables"]["cuda"]["chunk_division"]
            assert block["source"] == name, block
            assert block["source"] != cuda_arms.ONE_RANK_DEFAULT_SOURCE
            assert block["refused_because"] is None, block
            assert block["record"] == driver.chunk_bounds_record().tolist()
            assert block["record"] == chunk_bounds_record(driver.chunk_layout).tolist()
            blocks[name] = block
        finally:
            driver.close()
    # The lifted record is not the one-rank one, so the default would have bound
    # different words for the lifted run.
    assert blocks["meep"]["record"] != blocks["one_rank"]["record"]


# --------------------------------------------------------------------------
# 3. BOTH TABLES JUDGE THE DIVISION HANDED OVER
# --------------------------------------------------------------------------

def test_a_lifted_division_no_record_describes_is_refused_by_both_nvidia_tables(monkeypatch):
    stub_triton(monkeypatch)  # The real composers, both of them.

    driver = _nvidia_driver(monkeypatch, **NON_SEPARABLE)
    try:
        assert driver.chunk_layout.non_separable()
        driver.step()
        record = driver.fast_path_report()
        assert record["step_path"] == "array" and record["decision"] == "refused"
        # Triton composes first and refuses the whole plan on the division.
        for slot in PML_SLOTS:
            entry = record["slots"][slot]
            assert entry["arm"] is None, (slot, entry)
            assert NON_SEPARABLE_DIVISION in entry["reason"], (slot, entry)
        # Every CUDA arm refuses it by the record builder's reason.
        block = record["tables"]["cuda"]["chunk_division"]
        assert block["source"] == "meep" and block["record"] is None, block
        assert NOT_SEPARABLE_ALONG_Y in block["refused_because"], block
        cuda_reasons = record["arbitration"]["secondary_reasons"]["cuda"]
        for slot in PML_SLOTS:
            assert any(NOT_SEPARABLE_ALONG_Y in reason for reason in cuda_reasons[slot]), (
                slot, cuda_reasons[slot])
    finally:
        driver.close()

    # The control: the same driver stepping its one-rank division is not refused on it.
    control = _nvidia_driver(monkeypatch)
    try:
        control.step()
        record = control.fast_path_report()
        block = record["tables"]["cuda"]["chunk_division"]
        assert block["source"] == "one_rank" and block["refused_because"] is None, block
        for slot in PML_SLOTS:
            reason = record["slots"][slot].get("reason") or ""
            assert NON_SEPARABLE_DIVISION not in reason, (slot, reason)
            assert NOT_SEPARABLE_ALONG_Y not in reason, (slot, reason)
    finally:
        control.close()
