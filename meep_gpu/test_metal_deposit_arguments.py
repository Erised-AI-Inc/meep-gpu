"""The held source deposit carries its payload as kernel arguments: no upload, no wait.

``Residency.subtract_double_index`` hands the deposit's binary64 payload to
``door_scatter_sub``. Up to ``device.DEPOSIT_ARGUMENT_WORDS`` points the payload's int64
bit patterns go as ``int`` lists the launcher binds when the launch is encoded; a larger
table uploads them as a tensor, a host-to-device copy that waits for every launch queued
ahead of it (352-442 us of a held ``pml_3d`` step at 110,592 cells while the source was
on). The kernel and its source are the same in both forms; only the transport differs.

What this file pins, each against the array path's own statement
(``host_writes._rounded_difference`` on a host copy) AND against the uploaded form on the
same data, word for word, sign of zero included:

* a deposit of 1 to 512 points, both storage widths, under keep and flush, forms the
  array path's words and counts no upload (``Residency.deposit_uploads``) and no read
  back; the same deposits with the upload forced count one upload each, so the counter
  can move;
* 512 points is the last argument deposit and 513 uploads;
* a launcher that refuses list arguments is detected once, on a scratch word, and the
  deposit falls back to the upload with the same words;
* over a held ``pml_3d`` run with the source on, the package's counter reads no payload
  upload and no CPU-to-MPS ``Tensor.to`` happens at all, while one device deposit a step
  lands; the run's state equals the array path's;
* over a held ``pml_2d`` run, the same with the upload forced counts one upload per
  deposit, the argument form removes exactly those copies (the composition's
  deposit-repair bracket writes back through ``Residency.scatter``, whose payload crosses
  as arguments too, ``test_metal_read_ahead.py``), and both forms equal the array path,
  so the counter and the comparison could each have failed.

Progress reporting: each case prints one flushed line with what it measured.
"""

from __future__ import annotations

import pathlib
import platform
import sys
from typing import Any, Dict

import numpy as np
import pytest

_PARITY = pathlib.Path(__file__).resolve().parent.parent / "parity" / "meep_gpu"
if str(_PARITY) not in sys.path:
    sys.path.insert(0, str(_PARITY))

from conftest import requires_resource_skip  # noqa: E402

from meep_gpu import backends, host_writes, subnormal_policy  # noqa: E402

POLICIES = ("keep", "flush")
#: Point counts for the argument form: one, a few, an interpolated 3-D point's eight, a
#: line source, and the two counts at the bound.
ARGUMENT_POINTS = (1, 2, 8, 64, 511, 512)
#: Steps of the run-level cases: enough to freeze, warm and then count a window, all
#: well inside ``pml_2d``'s source (on for its first ~600 steps at resolution 10).
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
        requires_resource_skip("mps_device", "the held door launches on an Apple GPU")
    return torch


class host_mode:
    """``with host_mode("flush"):`` drives this thread's FPU and restores it on exit."""

    def __init__(self, policy: str) -> None:
        self.flush = policy == "flush"

    def __enter__(self) -> bool:
        if platform.machine().lower() not in ("arm64", "aarch64"):
            requires_resource_skip("arm64_host", "the Metal door is specified by an arm64 "
                                   "host's float64 unit")
        self.before = backends.subnormals_flushed()
        if self.before != self.flush:
            subnormal_policy._drive_host_fpu(self.flush)  # noqa: SLF001
        if backends.subnormals_flushed() != self.flush:
            requires_resource_skip("host_fpu_mode", "this host's FPU mode cannot be driven")
        return self.flush

    def __exit__(self, *exc: Any) -> None:
        if backends.subnormals_flushed() != self.before:
            subnormal_policy._drive_host_fpu(self.before)  # noqa: SLF001
        assert backends.subnormals_flushed() == self.before, "the host FPU mode was not restored"


@pytest.fixture(autouse=True, scope="module")
def _leave_the_process_as_this_file_found_it():
    """Put the FPU and the policy bookkeeping back when this file is done.

    The run-level cases freeze held plans, which install ``flush`` and drive the host FPU
    there (``test_metal_held_fallback`` carries the same guard for the same leak).
    """
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


def _stored_words(rng, cells: int, dtype) -> np.ndarray:
    """Float32 words with both zeros, normals of both signs, a NaN, infinities and subnormals."""
    width = 2 if dtype == np.complex64 else 1
    words = rng.integers(0, 2**32, size=cells * width, dtype=np.uint64).astype(np.uint32)
    special = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e-40, -1e-40,
                        3.0e-30, -7.5e5], dtype=np.float32).view(np.uint32)
    words[: special.size] = special
    return words.view(np.float32).view(dtype).copy()


def _payload(rng, points: int) -> np.ndarray:
    """Binary64 subtrahends: random bit patterns and ordinary magnitudes, plus both zeros,
    the smallest subnormal and values a float32 rounding splits, where the table is long
    enough to carry them."""
    bits = rng.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=points,
                        dtype=np.int64, endpoint=True)
    values = bits.view(np.float64).copy()
    half = points // 2
    values[points - half:] = rng.standard_normal(half)
    fixed = np.array([0.0, -0.0, backends._smallest_subnormal(), 1.0 + 2.0**-24, -(1.0 + 3 * 2.0**-24), 0.3],
                     dtype=np.float64)
    if points >= 2 * fixed.size:
        values[: fixed.size] = fixed
    return values


def _held_deposit(dtype, policy: str, points: int, cells: int, seed: int,
                  argument_words: int) -> Dict[str, Any]:
    """Deposit four disjoint tables into a held mirror; return its words, the reference, counters."""
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    width = 2 if dtype == np.complex64 else 1
    host = _stored_words(rng, cells, dtype)
    reference = host.copy()
    order = rng.permutation(cells)
    tables = [np.ascontiguousarray(order[k * points:(k + 1) * points], dtype=np.int64)
              for k in range(4)]
    payloads = [(_payload(rng, points), _payload(rng, points) if width == 2 else None)
                for _ in tables]
    residency = device.Residency(device="mps")
    residency.mirror("field", host, dtype=dtype)
    residency.arm_hold(["field"])
    residency.sync_out(["field"])
    assert residency.state_of("field") == device.DEVICE_OWNED
    host_writes.register(residency)
    saved = device.DEPOSIT_ARGUMENT_WORDS
    device.DEPOSIT_ARGUMENT_WORDS = argument_words
    try:
        before = (residency.double_subtracts, residency.deposit_uploads,
                  residency.sparse_gathers, residency.copies_out)
        with host_mode(policy), np.errstate(all="ignore"):
            for linear, (real64, imag64) in zip(tables, payloads):
                host_writes.subtract_double(host, linear, real64, imag64)
                flat = reference.reshape(-1)
                flat[linear] = host_writes._rounded_difference(flat[linear], real64, imag64)  # noqa: SLF001
        after = (residency.double_subtracts, residency.deposit_uploads,
                 residency.sparse_gathers, residency.copies_out)
        assert residency.state_of("field") == device.DEVICE_OWNED
        words = residency.tensor("field").cpu().numpy().reshape(-1).view(np.uint32).copy()
    finally:
        device.DEPOSIT_ARGUMENT_WORDS = saved
        host_writes.unregister(residency)
    want = np.ascontiguousarray(reference).reshape(-1).view(np.uint32)
    return {"words": words, "want": want,
            "differing": int(np.count_nonzero(words != want)),
            "moved": int(np.count_nonzero(words != np.ascontiguousarray(
                _stored_words(np.random.default_rng(seed), cells, dtype)).reshape(-1).view(np.uint32))),
            "deposits": after[0] - before[0], "uploads": after[1] - before[1],
            "gathers": after[2] - before[2], "copies_out": after[3] - before[3]}


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("dtype", (np.float32, np.complex64), ids=("real", "complex"))
@pytest.mark.parametrize("points", ARGUMENT_POINTS)
def test_a_held_deposit_up_to_the_bound_carries_its_payload_as_arguments_and_uploads_nothing(
        points, dtype, policy):
    require_mps()
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    assert device.DEPOSIT_ARGUMENT_WORDS == 512
    cells = 4 * points + 7
    seed = 20261016 + points * 4 + (2 if dtype == np.complex64 else 0) + (policy == "flush")
    arguments = _held_deposit(dtype, policy, points, cells, seed, argument_words=512)
    uploaded = _held_deposit(dtype, policy, points, cells, seed, argument_words=0)
    across = int(np.count_nonzero(arguments["words"] != uploaded["words"]))
    say(f"deposit {points} points {np.dtype(dtype).name} {policy}: arguments "
        f"{arguments['differing']} / {arguments['want'].size} words vs the array path, "
        f"{across} vs the upload; {arguments['moved']} words moved; uploads "
        f"{arguments['uploads']} (forced: {uploaded['uploads']}) of {arguments['deposits']} deposits")
    assert arguments["differing"] == 0 and uploaded["differing"] == 0 and across == 0
    assert arguments["moved"] > 0, "a comparison over unchanged words proves nothing"
    assert (arguments["deposits"], arguments["uploads"], arguments["gathers"],
            arguments["copies_out"]) == (4, 0, 0, 0), arguments
    assert (uploaded["deposits"], uploaded["uploads"]) == (4, 4), uploaded


@pytest.mark.parametrize("dtype", (np.float32, np.complex64), ids=("real", "complex"))
def test_512_points_is_the_last_argument_deposit_and_513_uploads(dtype):
    require_mps()
    at_bound = _held_deposit(dtype, "flush", 512, 4 * 513 + 3, 512, argument_words=512)
    past = _held_deposit(dtype, "flush", 513, 4 * 513 + 3, 513, argument_words=512)
    say(f"bound {np.dtype(dtype).name}: 512 points -> {at_bound['uploads']} uploads, "
        f"513 points -> {past['uploads']} uploads; differing {at_bound['differing']} / "
        f"{past['differing']}")
    assert (at_bound["uploads"], past["uploads"]) == (0, 4)
    assert at_bound["differing"] == 0 and past["differing"] == 0


def test_a_launcher_that_refuses_list_arguments_falls_back_to_the_upload_with_the_same_words(
        monkeypatch):
    torch = require_mps()
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    real_library = device._door_kernels()  # noqa: SLF001
    seen = {"refused": 0, "launched": 0}

    class ListRefusingLibrary:
        """The door library as a launcher without list arguments would present it."""

        def __getattr__(self, name: str) -> Any:
            kernel = getattr(real_library, name)

            def launch(*args: Any, **kwargs: Any) -> Any:
                if any(isinstance(arg, (list, tuple)) for arg in args):
                    seen["refused"] += 1
                    raise RuntimeError("Unsupported argument type")
                seen["launched"] += 1
                return kernel(*args, **kwargs)
            return launch

    monkeypatch.setattr(device, "_ARGUMENT_PAYLOADS", {})
    monkeypatch.setattr(device, "_door_kernels", lambda: ListRefusingLibrary())
    result = _held_deposit(np.complex64, "flush", 8, 64, 77, argument_words=512)
    say(f"refusing launcher: probe refused {seen['refused']}, door launched {seen['launched']}, "
        f"uploads {result['uploads']} of {result['deposits']}, differing {result['differing']}")
    assert device._ARGUMENT_PAYLOADS == {"mps": False}  # noqa: SLF001
    assert seen["refused"] == 1, "the capability is asked once, on scratch, then remembered"
    assert (result["deposits"], result["uploads"], result["differing"]) == (4, 4, 0), result
    assert result["moved"] > 0
    del torch


# ---------------------------------------------------------------------------
# A held run with the source on
# ---------------------------------------------------------------------------

def _held_run(monkeypatch, case: str, res: int, argument_words: int) -> Dict[str, Any]:
    require_mps()
    try:
        import meep as mp  # noqa: PLC0415
    except ImportError:
        requires_resource_skip("meep", "the case is lifted from an mp.Simulation declaration")
    import torch  # noqa: PLC0415

    import gate_dispatch_end_to_end as e2e  # noqa: PLC0415
    from meep_gpu import fastpath, lift_simulation, metal_dispatch  # noqa: PLC0415
    from meep_gpu.metal_kernels import device  # noqa: PLC0415

    monkeypatch.setattr(device, "DEPOSIT_ARGUMENT_WORDS", argument_words)
    mp.verbosity(0)
    held = lift_simulation(e2e.CASES[case](mp, res)[0], prefer_gpu=True)
    ref = lift_simulation(e2e.CASES[case](mp, res)[0], prefer_gpu=True)
    assert held.gpu == "metal", held.gpu
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.delenv(metal_dispatch.RESIDENCY_ENV, raising=False)
    held.run(num_steps=WARM_STEPS)
    block = dict((held.fast_path_report() or {}).get("residency") or {})
    assert held.active_step_path == "fused" and block.get("mode") == metal_dispatch.RESIDENCY_HELD
    residency = held._fast_path.residency  # noqa: SLF001
    uploads = {"cpu_to_mps": 0}
    original_to = torch.Tensor.to

    def counted_to(self, *args: Any, **kwargs: Any) -> Any:
        out = original_to(self, *args, **kwargs)
        if self.device.type == "cpu" and getattr(out, "device", None) is not None \
                and out.device.type == "mps":
            uploads["cpu_to_mps"] += 1
        return out

    before = (residency.double_subtracts, residency.deposit_uploads, residency.sparse_gathers)
    monkeypatch.setattr(torch.Tensor, "to", counted_to)
    held.run(num_steps=WINDOW_STEPS)
    monkeypatch.setattr(torch.Tensor, "to", original_to)
    after = (residency.double_subtracts, residency.deposit_uploads, residency.sparse_gathers)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "0")
    ref.run(num_steps=WARM_STEPS + WINDOW_STEPS)
    verdict = e2e.compare_state(e2e.collect_state(held), e2e.collect_state(ref))
    return {"deposits": after[0] - before[0], "uploads": after[1] - before[1],
            "gathers": after[2] - before[2], "tensor_to_cpu_to_mps": uploads["cpu_to_mps"],
            "identical": verdict["identical"], "arrays": verdict["arrays"],
            "words": verdict["words_compared"], "differences": verdict["differences"][:3]}


def _line(case: str, res: int, leg: str, run: Dict[str, Any]) -> str:
    return (f"{case} res {res} {leg}, {WINDOW_STEPS} held steps, source on: {run['deposits']} "
            f"device deposits, {run['uploads']} payload uploads, {run['tensor_to_cpu_to_mps']} "
            f"CPU->MPS Tensor.to, {run['gathers']} gathers; identical to the array path over "
            f"{run['arrays']} arrays / {run['words']} words: {run['identical']}")


def test_a_held_pml_3d_run_with_the_source_on_uploads_nothing_and_keeps_the_array_paths_words(
        monkeypatch):
    """``pml_3d``: the composition MLXLAUNCH timed, where the deposit was the step's one upload."""
    run = _held_run(monkeypatch, "pml_3d", 6, argument_words=512)
    say(_line("pml_3d", 6, "arguments", run))
    assert run["deposits"] >= WINDOW_STEPS, "the window must run with the source on"
    assert run["uploads"] == 0 and run["tensor_to_cpu_to_mps"] == 0, run
    assert run["identical"], run["differences"]


def test_the_upload_forced_counts_one_upload_per_deposit_and_the_argument_form_removes_exactly_those(
        monkeypatch):
    """``pml_2d`` both ways: the counter moves when the upload is forced, and only the
    deposit's copy leaves -- this composition's deposit-repair bracket writes back with
    one scatter a step, whose transport ``device.SCATTER_ARGUMENT_WORDS`` governs, not
    ``DEPOSIT_ARGUMENT_WORDS``."""
    arguments = _held_run(monkeypatch, "pml_2d", RES, argument_words=512)
    forced = _held_run(monkeypatch, "pml_2d", RES, argument_words=0)
    say(_line("pml_2d", RES, "arguments", arguments))
    say(_line("pml_2d", RES, "upload forced", forced))
    assert arguments["deposits"] >= WINDOW_STEPS and forced["deposits"] == arguments["deposits"]
    assert arguments["uploads"] == 0 and forced["uploads"] == forced["deposits"], (arguments, forced)
    assert arguments["tensor_to_cpu_to_mps"] == forced["tensor_to_cpu_to_mps"] - forced["deposits"]
    assert arguments["identical"] and forced["identical"], (arguments["differences"],
                                                            forced["differences"])
