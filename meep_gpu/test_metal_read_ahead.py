"""The held residency's host reads and small writes stop waiting on the queue one by one.

Three changes, each pinned here word for word against the 1.0 statement it replaces:

* READ-AHEAD (``device.READ_AHEAD``). A run of device gathers the host issues back to back
  -- a flux monitor's six components, the deposit repair's field and ``f_w`` -- is
  remembered, and the next run that starts with the same read comes off the device in ONE
  transfer. Pinned: the values equal the per-gather reads, the transfer count drops
  (``Residency.gather_reads`` / ``gathers_ahead``), a host-current mirror is never read off
  the device, and the FENCE: a door write, a bracket or a torch op landing between two reads
  of a remembered run voids what was read ahead. The fence's control disables both of its
  guards and shows the stale word it exists to refuse.
* SCATTER ARGUMENTS (``device.SCATTER_ARGUMENT_WORDS``). ``Residency.scatter``'s payload
  crosses as kernel arguments (the float32 words packed two to an int64) instead of a
  blocking upload, up to 1,024 words: every float32 pattern, signalling NaNs included, lands
  as given; past the bound, or on a launcher that refuses list arguments, the upload stays
  and counts in ``Residency.scatter_uploads``.
* Shape probes past the barrier (``deposit_repair._shape_probe``). The repair's seven
  shape and allocation probes no longer copy device-owned volumes out whole: off a hold the
  probe is the object ``getattr`` returns; over a held ``pml_2d`` and ``magnetic_seam_2d``
  run the bracket copies no volume, reads its cells in two transfers a step, uploads
  nothing, and the state equals the array path's -- with run-level controls on the
  read-ahead and on the scatter payload that must, and do, diverge.

Progress reporting: each case prints one flushed line with what it measured.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict

import numpy as np
import pytest

_PARITY = pathlib.Path(__file__).resolve().parent.parent / "parity" / "meep_gpu"
if str(_PARITY) not in sys.path:
    sys.path.insert(0, str(_PARITY))

from conftest import requires_resource_skip  # noqa: E402

from meep_gpu import backends, host_writes, subnormal_policy  # noqa: E402

RES = 10
WARM_STEPS = 10
WINDOW_STEPS = 30


def say(line: str) -> None:
    print(line, flush=True)


def require_mps() -> Any:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        requires_resource_skip("torch", "PyTorch is not installed")
    if not torch.backends.mps.is_available():
        requires_resource_skip("mps_device", "the held doors launch on an Apple GPU")
    return torch


@pytest.fixture(autouse=True, scope="module")
def _leave_the_process_as_this_file_found_it():
    """Put the FPU and the policy bookkeeping back: the run-level cases freeze held plans."""
    flushing_at_entry = backends.subnormals_flushed()
    try:
        yield
    finally:
        subnormal_policy._reset_for_tests()  # noqa: SLF001
        if backends.subnormals_flushed() != flushing_at_entry:
            subnormal_policy._drive_host_fpu(flushing_at_entry)  # noqa: SLF001


@pytest.fixture(autouse=True)
def _no_hold_outlives_a_test():
    yield
    from meep_gpu import metal_dispatch  # noqa: PLC0415

    for residency in host_writes._live():  # noqa: SLF001
        metal_dispatch.release_residency_hold(residency, None)


def _words(rng, count: int) -> np.ndarray:
    """Random float32 words with both zeros, infinities, subnormals and quiet and signalling NaNs."""
    words = rng.integers(0, 2**32, size=count, dtype=np.uint64).astype(np.uint32)
    special = np.array([0x80000000, 0x7F800000, 0xFF800000, 0x00000001, 0x807FFFFF,
                        0x7FC00001, 0x7F800001, 0xFFBFFFFF, 0x00000000, 0x3F800000],
                       dtype=np.uint32)
    k = min(count, special.size)
    words[:k] = special[:k]
    return words


def _held(dtypes: Dict[str, Any], cells: int, seed: int):
    """A bare residency holding one device-owned mirror per name, filled with random words."""
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    residency = device.Residency(device="mps")
    hosts = {}
    for name, dtype in dtypes.items():
        width = 2 if dtype == np.complex64 else 1
        hosts[name] = _words(rng, cells * width).view(np.float32).view(dtype).copy()
        residency.mirror(name, hosts[name], dtype=dtype)
    residency.arm_hold(list(dtypes))
    residency.sync_out(list(dtypes))
    for name in dtypes:
        assert residency.state_of(name) == device.DEVICE_OWNED
    return residency, hosts, rng


def _device_words(residency, name: str) -> np.ndarray:
    return residency.tensor(name).cpu().numpy().reshape(-1).view(np.uint32).copy()


def _counts(residency) -> Dict[str, int]:
    return {"gathers": residency.sparse_gathers, "reads": residency.gather_reads,
            "ahead": residency.gathers_ahead}


def _delta(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
    return {key: after[key] - before[key] for key in after}


# ---------------------------------------------------------------------------
# Read-ahead
# ---------------------------------------------------------------------------

def test_a_repeated_run_of_gathers_comes_off_the_device_in_one_transfer_with_the_same_words():
    require_mps()
    residency, hosts, rng = _held({"a": np.float32, "b": np.complex64, "c": np.float32}, 5000, 7)
    tables = [np.ascontiguousarray(rng.permutation(5000)[:n], dtype=np.int64) for n in (300, 41, 1)]
    other = np.arange(3, dtype=np.int64)

    def run():
        return [residency.gather(hosts[name], table)
                for name, table in zip(("a", "b", "c"), tables, strict=True)]

    direct = {name: _device_words(residency, name) for name in hosts}
    first = run()
    learned = _counts(residency)
    residency.zero_index(hosts["c"], other)          # any other Residency call ends a run
    expect_c = _device_words(residency, "c")
    before = _counts(residency)
    second = run()
    batched = _delta(_counts(residency), before)
    for values, name, table in zip(second, ("a", "b", "c"), tables, strict=True):
        want = (direct[name] if name != "c" else expect_c)
        width = 2 if hosts[name].dtype == np.complex64 else 1
        want_rows = want.reshape(-1, width)[table].reshape(-1)
        got = np.ascontiguousarray(values).reshape(-1).view(np.uint32)
        assert values.dtype == hosts[name].dtype and values.shape == (table.size,)
        assert np.array_equal(got, want_rows), name
    for a, b in zip(first[:2], second[:2], strict=True):
        assert np.array_equal(np.ascontiguousarray(a).view(np.uint32), np.ascontiguousarray(b).view(np.uint32))
    say(f"read-ahead: first run {learned['reads']} transfers for {learned['gathers']} gathers; the "
        f"repeat {batched['reads']} transfer for {batched['gathers']} gathers, {batched['ahead']} "
        f"answered ahead; words equal to the per-gather reads (float32 + complex64 + float32)")
    assert learned == {"gathers": 3, "reads": 3, "ahead": 0}
    assert batched == {"gathers": 3, "reads": 1, "ahead": 2}


def _remembered_pair(seed: int):
    residency, hosts, rng = _held({"a": np.float32, "b": np.float32}, 2000, seed)
    ia = np.ascontiguousarray(rng.permutation(2000)[:16], dtype=np.int64)
    ib = np.ascontiguousarray(rng.permutation(2000)[:16], dtype=np.int64)
    residency.gather(hosts["a"], ia)
    residency.gather(hosts["b"], ib)
    residency.zero_index(hosts["a"], np.array([1999], dtype=np.int64))  # ends the learning run
    return residency, hosts, ia, ib


def _hazard(monkeypatch, disable: str):
    """Start a remembered run, write ``b`` between its two reads, return what the second read saw.

    ``disable``: which of the fence's guards to take out -- "none", "boundary" (the run is not
    ended by the write), "fence" (the counter snapshot never changes), or "both".
    """
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    residency, hosts, ia, ib = _remembered_pair(11)
    if disable in ("boundary", "both"):
        real_end = device.Residency._end_read_run  # noqa: SLF001
        calls = {"in_gather": False}

        def end_only_from_gather(self):
            # The write's own boundary is disabled; gather's fence-triggered one stays,
            # so the counter snapshot alone decides.
            if calls["in_gather"]:
                return real_end(self)
            return None

        real_gather = device.Residency.gather

        def gather(self, array, linear):
            calls["in_gather"] = True
            try:
                return real_gather(self, array, linear)
            finally:
                calls["in_gather"] = False

        monkeypatch.setattr(device.Residency, "_end_read_run", end_only_from_gather)
        monkeypatch.setattr(device.Residency, "gather", gather)
    if disable in ("fence", "both"):
        monkeypatch.setattr(device.Residency, "_read_fence", lambda self: ())
    handle = residency._mirrors["b"].tensor  # noqa: SLF001 - read without a Residency call
    before = _counts(residency)
    residency.gather(hosts["a"], ia)                 # reads a AND b ahead
    stale = handle.cpu().numpy()[ib].copy()          # what was read ahead for b
    fresh = np.full(ib.size, 3.5, dtype=np.float32)
    residency.scatter(hosts["b"], ib, fresh)         # a door write between the two reads
    seen = residency.gather(hosts["b"], ib)
    return {"seen_fresh": bool(np.array_equal(seen.view(np.uint32), fresh.view(np.uint32))),
            "seen_stale": bool(np.array_equal(seen.view(np.uint32), stale.view(np.uint32))),
            "counts": _delta(_counts(residency), before)}


@pytest.mark.parametrize("disable", ("none", "boundary", "fence"))
def test_a_door_write_between_two_reads_of_a_remembered_run_voids_the_read_ahead(monkeypatch, disable):
    require_mps()
    result = _hazard(monkeypatch, disable)
    say(f"fence, guard disabled: {disable}: the second read saw the write {result['seen_fresh']}, "
        f"the stale batch {result['seen_stale']}; counters {result['counts']}")
    assert result["seen_fresh"] and not result["seen_stale"], result
    assert result["counts"]["ahead"] == 0, "the batch must not answer after the write"


def test_the_fence_control_with_both_guards_disabled_serves_the_stale_word(monkeypatch):
    """The control: the hazard is real, and the fence is what refuses it."""
    require_mps()
    result = _hazard(monkeypatch, "both")
    say(f"fence control (both guards disabled): the second read saw the stale batch "
        f"{result['seen_stale']}, the write {result['seen_fresh']}; counters {result['counts']}")
    assert result["seen_stale"] and not result["seen_fresh"], result
    assert result["counts"]["ahead"] == 1


def test_a_torch_op_on_a_mirror_tensor_between_two_reads_voids_the_read_ahead():
    """No Residency call at all: the tensor's ``_version`` is the guard."""
    require_mps()
    residency, hosts, ia, ib = _remembered_pair(13)
    before = _counts(residency)
    tensor = residency._mirrors["b"].tensor  # noqa: SLF001 - a handle taken outside the class
    residency.gather(hosts["a"], ia)                 # reads a AND b ahead
    tensor.add_(1.0)                                 # no Residency call between the reads
    seen = residency.gather(hosts["b"], ib)
    want = tensor.cpu().numpy()[ib]
    counts = _delta(_counts(residency), before)
    say(f"torch add_ between two reads: the second read equals the device words "
        f"{np.array_equal(seen.view(np.uint32), want.view(np.uint32))}; counters {counts}")
    assert np.array_equal(seen.view(np.uint32), want.view(np.uint32))
    assert counts["ahead"] == 0


def test_a_mirror_the_host_holds_current_is_never_read_off_the_device():
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    require_mps()
    residency, hosts, ia, ib = _remembered_pair(17)
    residency.acquire_read(hosts["b"])               # b is now CLEAN_BOTH
    assert residency.state_of("b") == device.CLEAN_BOTH
    residency.tensor("b").zero_()                    # make the device side visibly different
    before = _counts(residency)
    a = residency.gather(hosts["a"], ia)
    b = residency.gather(hosts["b"], ib)
    counts = _delta(_counts(residency), before)
    say(f"host-current mirror in a remembered run: a from the device ({counts}), b from the host "
        f"{np.array_equal(b.view(np.uint32), hosts['b'].reshape(-1)[ib].view(np.uint32))}")
    assert np.array_equal(b.view(np.uint32), hosts["b"].reshape(-1)[ib].view(np.uint32))
    assert counts == {"gathers": 1, "reads": 1, "ahead": 0}
    assert np.array_equal(a.view(np.uint32), residency.tensor("a").cpu().numpy()[ia].view(np.uint32))


def test_a_run_that_changes_is_remembered_as_it_ran_and_read_ahead_off_reads_one_at_a_time(monkeypatch):
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    require_mps()
    residency, hosts, ia, ib = _remembered_pair(19)
    ic = np.array([5, 6, 7], dtype=np.int64)
    residency.gather(hosts["a"], ia)                 # batch (a, b) read; b is not asked
    residency.gather(hosts["a"], ic)
    residency.zero_index(hosts["b"], np.array([0], dtype=np.int64))
    before = _counts(residency)
    residency.gather(hosts["a"], ia)
    residency.gather(hosts["a"], ic)
    replaced = _delta(_counts(residency), before)
    residency.zero_index(hosts["b"], np.array([0], dtype=np.int64))
    monkeypatch.setattr(device, "READ_AHEAD", False)
    before = _counts(residency)
    residency.gather(hosts["a"], ia)
    residency.gather(hosts["a"], ic)
    off = _delta(_counts(residency), before)
    say(f"changed run read ahead as it last ran: {replaced}; READ_AHEAD off: {off}")
    assert replaced == {"gathers": 2, "reads": 1, "ahead": 1}
    assert off == {"gathers": 2, "reads": 2, "ahead": 0}


# ---------------------------------------------------------------------------
# Scatter arguments
# ---------------------------------------------------------------------------

def _scatter(dtype, rows: int, cells: int, seed: int, argument_words: int, state: str = "device"):
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    residency, hosts, rng = _held({"f": dtype}, cells, seed)
    if state == "host":
        residency.acquire_read(hosts["f"])
    width = 2 if dtype == np.complex64 else 1
    table = np.ascontiguousarray(rng.permutation(cells)[:rows], dtype=np.int64)
    payload = _words(rng, rows * width).view(np.float32).view(dtype).copy()
    saved = device.SCATTER_ARGUMENT_WORDS
    device.SCATTER_ARGUMENT_WORDS = argument_words
    try:
        before = residency.scatter_uploads
        residency.scatter(hosts["f"], table, payload)
        uploads = residency.scatter_uploads - before
    finally:
        device.SCATTER_ARGUMENT_WORDS = saved
    got = _device_words(residency, "f").reshape(-1, width)[table].reshape(-1)
    want = payload.view(np.uint32)
    host_side = np.ascontiguousarray(hosts["f"]).reshape(-1).view(np.uint32).reshape(-1, width)[table].reshape(-1)
    return {"differing": int(np.count_nonzero(got != want)), "uploads": uploads,
            "host_written": bool(np.array_equal(host_side, want)), "words": int(want.size)}


@pytest.mark.parametrize("dtype,rows", [(np.float32, n) for n in (1, 2, 3, 512, 1024)]
                         + [(np.complex64, n) for n in (1, 2, 3, 512)],
                         ids=lambda v: getattr(v, "__name__", str(v)))
def test_a_scatter_up_to_the_bound_carries_its_words_as_arguments_exactly(dtype, rows):
    require_mps()
    arguments = _scatter(dtype, rows, 4 * rows + 9, 100 + rows, 1024)
    uploaded = _scatter(dtype, rows, 4 * rows + 9, 100 + rows, 0)
    say(f"scatter {rows} rows {np.dtype(dtype).name}: arguments {arguments['differing']} / "
        f"{arguments['words']} words differ, {arguments['uploads']} uploads; forced upload "
        f"{uploaded['differing']} differ, {uploaded['uploads']} uploads")
    assert (arguments["differing"], arguments["uploads"]) == (0, 0), arguments
    assert (uploaded["differing"], uploaded["uploads"]) == (0, 1), uploaded


@pytest.mark.parametrize("dtype,rows", [(np.float32, 1025), (np.complex64, 513)],
                         ids=("real-1025", "complex-513"))
def test_past_1024_words_the_scatter_uploads(dtype, rows):
    require_mps()
    past = _scatter(dtype, rows, 3 * rows, 300, 1024)
    say(f"scatter {rows} rows {np.dtype(dtype).name} past the bound: {past['uploads']} upload, "
        f"{past['differing']} words differ")
    assert (past["differing"], past["uploads"]) == (0, 1)


def test_a_scatter_to_a_host_current_mirror_writes_both_sides_with_the_same_words():
    require_mps()
    result = _scatter(np.complex64, 8, 64, 400, 1024, state="host")
    say(f"host-current scatter: device {result['differing']} differ, host written {result['host_written']}")
    assert result["differing"] == 0 and result["host_written"] and result["uploads"] == 0


def test_a_launcher_that_refuses_list_arguments_scatters_by_upload_with_the_same_words(monkeypatch):
    require_mps()
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    real_library = device._door_kernels()  # noqa: SLF001

    class ListRefusingLibrary:
        def __getattr__(self, name: str) -> Any:
            kernel = getattr(real_library, name)

            def launch(*args: Any, **kwargs: Any) -> Any:
                if any(isinstance(arg, (list, tuple)) for arg in args):
                    raise RuntimeError("Unsupported argument type")
                return kernel(*args, **kwargs)
            return launch

    monkeypatch.setattr(device, "_ARGUMENT_PAYLOADS", {})
    monkeypatch.setattr(device, "_door_kernels", lambda: ListRefusingLibrary())
    result = _scatter(np.float32, 8, 64, 500, 1024)
    say(f"refusing launcher: {result['uploads']} upload, {result['differing']} words differ")
    assert device._ARGUMENT_PAYLOADS == {"mps": False}  # noqa: SLF001
    assert (result["differing"], result["uploads"]) == (0, 1)


# ---------------------------------------------------------------------------
# The deposit repair's shape probes, and held runs
# ---------------------------------------------------------------------------

#: Every name a shape probe can be asked for: the seam targets, their sources and ``f_w``;
#: plus one no ``Fields`` carries, so the probe's not-allocated branch (``getattr``'s default)
#: is exercised whatever the case allocates.
PROBED = tuple(f"{p}{a}" for p in ("E", "D", "H", "B", "f_w_E", "f_w_H") for a in "xyz") + ("f_w_Dz",)


def _lift(case: str, res: int, prefer_gpu: bool):
    try:
        import meep as mp  # noqa: PLC0415
    except ImportError:
        requires_resource_skip("meep", "the case is lifted from an mp.Simulation declaration")
    import gate_dispatch_end_to_end as e2e  # noqa: PLC0415
    import gate_dispatch_metal_route as route  # noqa: PLC0415
    from meep_gpu import lift_simulation  # noqa: PLC0415

    mp.verbosity(0)
    builder = e2e.CASES.get(case) or route.CASES[case]
    return lift_simulation(builder(mp, res)[0], prefer_gpu=prefer_gpu)


@pytest.mark.parametrize("case", ("pml_2d", "magnetic_seam_2d", "no_pml_dispersive_2d"))
def test_off_a_hold_the_shape_probe_is_the_object_getattr_returns(case):
    from meep_gpu import deposit_repair  # noqa: PLC0415

    driver = _lift(case, RES, prefer_gpu=False)
    driver.run(num_steps=3)
    fields = driver.fields
    same = {name: deposit_repair._shape_probe(fields, name) is getattr(fields, name, None)  # noqa: SLF001
            for name in PROBED}
    present = sorted(name for name in PROBED if getattr(fields, name, None) is not None)
    say(f"{case} (NumPy reference): shape probe is getattr's object for {sum(same.values())} / "
        f"{len(PROBED)} names ({len(present)} allocated, {len(PROBED) - len(present)} not)")
    assert all(same.values()), {k: v for k, v in same.items() if not v}
    assert present and len(present) < len(PROBED), "both branches must be exercised"


def _held_run(monkeypatch, case: str, control: str = "") -> Dict[str, Any]:
    torch = require_mps()
    import gate_dispatch_end_to_end as e2e  # noqa: PLC0415
    from meep_gpu import fastpath, metal_dispatch  # noqa: PLC0415
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    if control == "read_ahead":
        real = device.Residency._read_run_ahead  # noqa: SLF001

        def flipped(self, plan):
            # Bit 0 (one float32 ULP) of the middle word of EVERY read the batch answers:
            # the one returned and each one queued ahead.
            values = real(self, plan).copy()
            for words in [values] + [entry[4] for queue in self._read_ahead.values()  # noqa: SLF001
                                     for entry in queue]:
                flat = words.reshape(-1).view(np.uint32)
                flat[flat.size // 2] ^= np.uint32(1)
            return values

        monkeypatch.setattr(device.Residency, "_read_run_ahead", flipped)
    if control == "scatter":
        real_rows = device._scatter_rows  # noqa: SLF001

        def flipped_rows(view, index, values_dev):
            if isinstance(values_dev, list) and values_dev:
                values_dev = [values_dev[0] ^ 1] + values_dev[1:]
            return real_rows(view, index, values_dev)

        monkeypatch.setattr(device, "_scatter_rows", flipped_rows)
    held = _lift(case, RES, prefer_gpu=True)
    ref = _lift(case, RES, prefer_gpu=True)
    assert held.gpu == "metal", held.gpu
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.delenv(metal_dispatch.RESIDENCY_ENV, raising=False)
    held.run(num_steps=WARM_STEPS)
    block = dict((held.fast_path_report() or {}).get("residency") or {})
    assert held.active_step_path == "fused" and block.get("mode") == metal_dispatch.RESIDENCY_HELD
    residency = held._fast_path.residency  # noqa: SLF001
    names = ("copies_out", "sparse_gathers", "gather_reads", "gathers_ahead", "sparse_scatters",
             "scatter_uploads", "double_subtracts", "deposit_uploads")
    uploads = {"cpu_to_mps": 0}
    original_to = torch.Tensor.to

    def counted_to(self, *args: Any, **kwargs: Any) -> Any:
        out = original_to(self, *args, **kwargs)
        if self.device.type == "cpu" and getattr(out, "device", None) is not None \
                and out.device.type == "mps":
            uploads["cpu_to_mps"] += 1
        return out

    def detached(driver: Any, steps: int) -> None:
        # The window counts the bracket alone, so the monitors are detached for it -- on
        # both drivers, so they skip the same updates.
        lists = ("_dft_monitors", "_flux_monitors")
        kept = {name: list(getattr(driver, name, []) or []) for name in lists}
        for name in lists:
            setattr(driver, name, [])
        try:
            driver.run(num_steps=steps)
        finally:
            for name, value in kept.items():
                setattr(driver, name, value)

    before = {name: getattr(residency, name) for name in names}
    monkeypatch.setattr(torch.Tensor, "to", counted_to)
    detached(held, WINDOW_STEPS)
    monkeypatch.setattr(torch.Tensor, "to", original_to)
    after = {name: getattr(residency, name) for name in names}
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "0")
    ref.run(num_steps=WARM_STEPS)
    detached(ref, WINDOW_STEPS)
    verdict = e2e.compare_state(e2e.collect_state(held), e2e.collect_state(ref))
    per_step = {name: (after[name] - before[name]) / WINDOW_STEPS for name in names}
    return {"per_step": per_step, "tensor_to_per_step": uploads["cpu_to_mps"] / WINDOW_STEPS,
            "identical": verdict["identical"], "arrays": verdict["arrays"],
            "words": verdict["words_compared"],
            "words_differing": sum(d.get("words_mismatched", 0) for d in verdict["differences"])}


@pytest.mark.parametrize("case", ("pml_2d", "magnetic_seam_2d"))
def test_a_held_bracketed_run_copies_no_volume_reads_twice_a_step_uploads_nothing_and_keeps_the_words(
        monkeypatch, case):
    run = _held_run(monkeypatch, case)
    p = run["per_step"]
    say(f"{case} res {RES}, {WINDOW_STEPS} held steps (monitors detached): copies out "
        f"{p['copies_out']:.2f}, gathers {p['sparse_gathers']:.2f} in {p['gather_reads']:.2f} "
        f"transfers ({p['gathers_ahead']:.2f} ahead), scatters {p['sparse_scatters']:.2f} with "
        f"{p['scatter_uploads']:.2f} uploads, CPU->MPS Tensor.to {run['tensor_to_per_step']:.2f}; "
        f"identical to the array path over {run['arrays']} arrays / {run['words']} words: "
        f"{run['identical']}")
    assert p["copies_out"] == 0, p
    assert p["sparse_gathers"] == 3 and p["gather_reads"] == 2 and p["gathers_ahead"] == 1, p
    assert p["scatter_uploads"] == 0 and run["tensor_to_per_step"] == 0, run
    assert run["identical"], run


@pytest.mark.parametrize("control,case", (("read_ahead", "dispersive_2d"), ("scatter", "pml_2d")))
def test_the_run_level_controls_on_the_read_ahead_and_the_scatter_payload_diverge(monkeypatch, control,
                                                                               case):
    """The run-level controls. The read-ahead's runs on ``dispersive_2d``: on ``pml_2d`` with
    the monitors detached the only batch is the save's ``field``/``f_w`` pair, and its source
    cell lies outside the absorber's chunks, where the repair takes the plain form and reads
    neither -- a control there could not fire. ``dispersive_2d``'s repair batches D with P
    (``deposit_repair._fresh_at``), whose difference every repaired word takes."""
    run = _held_run(monkeypatch, case, control=control)
    say(f"control {control} on {case} (bit 0 of the middle word flipped): {run['words_differing']} / "
        f"{run['words']} words differ from the array path; batched reads/step "
        f"{run['per_step']['gathers_ahead']:.2f}")
    assert not run["identical"] and run["words_differing"] > 0, run
