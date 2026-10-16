"""The silent source deposit on a held Metal residency: skipped where it is an identity.

``sources._deposit`` returns early when the current is exactly ``(+0, +0)`` and its own
statements at that current subtract ``+0`` at every point (``_silent_deposit_is_identity``).
Under a held residency the deposit otherwise reaches the device door
(``Residency.subtract_double_index``: software binary64 on the device's words). Each case
here deposits into a DEVICE_OWNED mirror through the engine's ``_deposit`` and compares
every device word with the ALWAYS-DEPOSIT form (the same ``_deposit`` with the early
return switched off) applied to a host copy with no residency, sign of zero included,
under the host FPU's keep and flush; and counts the door's launches: none for a skipped
table, one per table otherwise, and no word read back to the host either way.
"""

from __future__ import annotations

import platform
from typing import Any

import numpy as np
import pytest

from conftest import requires_resource_skip  # noqa: E402

from meep_gpu import backends, host_writes, subnormal_policy
from meep_gpu import sources as src


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


#: Stored words at each table's cells: both zeros, normals of both signs, a NaN, and under
#: keep both subnormal signs (under flush a stored subnormal never exists: the section
#: note of ``test_source_deposit.py``).
STORED = (0.0, -0.0, 1.0, -1.0, 3.0e-30, -7.5e5, float("nan"))
STORED_SUBNORMAL = (1.0e-40, -1.0e-40)
#: Amplitudes: unflagged (+1, 0.5+0.25j), flagged in the real plane (-1, -0.5+0.25j), and
#: flagged only in the imaginary plane (-1-1j, which real storage does not take).
AMPLITUDES = (1.0 + 0.0j, 0.5 + 0.25j, -1.0 + 0.0j, -0.5 + 0.25j, -1.0 - 1.0j)
#: The current: the exact silent one, another zero, and a live one.
CURRENTS = (complex(0.0, 0.0), complex(-0.0, 0.0), complex(0.3, -0.2))
#: None (no conductivity), an ordinary condinv, and a negative one (sigma <= -2/dt), which
#: inverts which amplitudes are flagged.
CONDINV = (None, 0.875, -0.5)


def _tables(complex_storage: bool, policy: str):
    stored = STORED + (STORED_SUBNORMAL if policy == "keep" else ())
    n = len(stored)
    dtype = np.complex64 if complex_storage else np.float32
    host = np.zeros((len(AMPLITUDES) * n, 1, 1), dtype=dtype)
    tables = []
    for t, amp in enumerate(AMPLITUDES):
        cells = np.arange(t * n, (t + 1) * n, dtype=np.int64)
        for k, cell in enumerate(cells.tolist()):
            host[cell, 0, 0] = complex(stored[k], stored[(k + 2) % n]) if complex_storage else stored[k]
        zeros = np.zeros_like(cells)
        tables.append((cells, zeros, zeros.copy(), np.full(n, amp, dtype=np.complex128)))
    return host, tables


def _always(array, tables, current, cndinv):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(src, "_silent_deposit_is_identity", lambda *args: False)
        with np.errstate(all="ignore"):
            for ix, iy, iz, amps in tables:
                src._deposit(array, ix, iy, iz, amps, current, 0.05, "Ez", cndinv)  # noqa: SLF001


@pytest.mark.parametrize("policy", ("keep", "flush"))
@pytest.mark.parametrize("storage", ("real", "complex"))
def test_a_held_silent_deposit_launches_no_door_where_it_is_an_identity_and_stores_the_always_deposit_words(
        storage, policy):
    require_mps()
    from meep_gpu.metal_kernels.device import DEVICE_OWNED, Residency  # noqa: PLC0415

    complex_storage = storage == "complex"
    skipped = launched = compared = 0
    with host_mode(policy):
        for current in CURRENTS:
            for cndinv_value in CONDINV:
                host, tables = _tables(complex_storage, policy)
                reference = host.copy()
                cndinv = (None if cndinv_value is None
                          else np.full(host.shape, cndinv_value, dtype=np.float32))
                residency = Residency(device="mps")
                residency.mirror("field", host, dtype=host.dtype)
                residency.arm_hold(["field"])
                residency.sync_out(["field"])
                assert residency.state_of("field") == DEVICE_OWNED
                host_writes.register(residency)
                try:
                    for ix, iy, iz, amps in tables:
                        before = (residency.double_subtracts, residency.sparse_gathers,
                                  residency.copies_out)
                        with np.errstate(all="ignore"):
                            src._deposit(host, ix, iy, iz, amps, current, 0.05, "Ez", cndinv)  # noqa: SLF001
                        doors = residency.double_subtracts - before[0]
                        assert (residency.sparse_gathers - before[1],
                                residency.copies_out - before[2]) == (0, 0)
                        keep_real = not complex_storage
                        identity = (src._is_plus_zero(current.real)  # noqa: SLF001
                                    and src._is_plus_zero(current.imag)  # noqa: SLF001
                                    and src._silent_deposit_is_identity(  # noqa: SLF001
                                        amps, ix, iy, iz, 0.05, cndinv, keep_real))
                        assert doors == (0 if identity else 1), (storage, policy, current,
                                                                  cndinv_value, complex(amps[0]))
                        skipped += identity
                        launched += doors
                    assert residency.state_of("field") == DEVICE_OWNED
                    device = residency.tensor("field").cpu().numpy().reshape(-1).view(np.uint32)
                finally:
                    host_writes.unregister(residency)
                _always(reference, tables, current, cndinv)
                want = np.ascontiguousarray(reference).reshape(-1).view(np.uint32)
                assert int(np.count_nonzero(device != want)) == 0, (storage, policy, current,
                                                                    cndinv_value)
                compared += int(want.size)
    # Both sides of the rule ran on the device: skipped tables and door launches at (+0, +0).
    assert skipped > 0 and launched > 0 and compared > 0, (skipped, launched)


def test_the_held_comparison_sees_a_flagged_table_that_is_skipped():
    """The must-catch control on the device: an amplitude -1 table skipped as if unflagged
    keeps the device's -0 words, which the always-deposit form turns into +0."""
    require_mps()
    from meep_gpu.metal_kernels.device import Residency  # noqa: PLC0415

    with host_mode("flush"):
        host, tables = _tables(False, "flush")
        reference = host.copy()
        residency = Residency(device="mps")
        residency.mirror("field", host, dtype=host.dtype)
        residency.arm_hold(["field"])
        residency.sync_out(["field"])
        host_writes.register(residency)
        try:
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(src, "_silent_deposit_is_identity", lambda *args: True)
                for ix, iy, iz, amps in tables:
                    src._deposit(host, ix, iy, iz, amps, complex(0.0, 0.0), 0.05, "Ez", None)  # noqa: SLF001
            device = residency.tensor("field").cpu().numpy().reshape(-1).view(np.uint32)
        finally:
            host_writes.unregister(residency)
        _always(reference, tables, complex(0.0, 0.0), None)
        want = np.ascontiguousarray(reference).reshape(-1).view(np.uint32)
        differing = np.flatnonzero(device != want)
        assert differing.size > 0
        assert set(device[differing].tolist()) == {0x80000000}
        assert set(want[differing].tolist()) == {0x00000000}
