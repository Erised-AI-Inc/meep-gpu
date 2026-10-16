"""The radial sum's source pull under a held residency (cylindrical complex Metal plans).

Three plans form the cylindrical radial difference on the HOST and bind it as a mirror:
:class:`~meep_gpu.metal_kernels.cylindrical_complex.CylindricalComplexCurlPlan` (both curl
sub-steps) and the two cylindrical complex fused pairs. Each ``refresh_prefix`` reads the
source volume -- ``Ey`` on B, ``Hy`` on D -- from its host array, so the host copy must be
current first. Under a hold, ``Residency.sync_out`` of a held mirror copies nothing: it
marks the mirror device-owned and seals it. The pull is
:func:`~meep_gpu.metal_kernels.cylindrical_complex.pull_radial_source`, which reads through
the hold (``acquire_read``) and keeps the unheld bracket's copy unchanged.

MEASURED before the pull existed, on the Metal engine route with held residency and no
monitor: m = 1 and m = 2 runs diverged from the array path from step 4 (111,415 / 1,376,256
and 118,204 / 1,146,880 words of the whole state by step 64); the shipped bracket read 0.

Two layers, each driving every plan's own ``refresh_prefix`` on a bare plan that carries
only the attributes the refresh reads (no kernel is compiled or launched):

* every host: a residency stand-in with ``device.Residency``'s held-mode contract for the
  two doors (``sync_out`` of a held name marks and copies nothing; ``acquire_read`` copies
  the device's words while the device owns them), so the contract is pinned without torch;
* an Apple GPU: a real ``device.Residency`` holding the source, its device words moved away
  from the host's the way a launch moves them, and the refresh must sum the DEVICE words.
"""

from __future__ import annotations

import numpy
import pytest

from conftest import requires_resource_skip

from .metal_kernels import cylindrical_complex as cyl
from .metal_kernels import cylindrical_fused_electric_pair as cfe
from .metal_kernels import cylindrical_fused_magnetic_pair as cfm

#: A small Dcyl volume, (r, phi, z): enough rows for a non-trivial radial sum.
SHAPE = (7, 1, 5)

#: Every plan that forms the radial difference on the host:
#: (plan class, curl sub-step, source mirror, radial mirror).
PLANS = (
    (cyl.CylindricalComplexCurlPlan, "step_B", "Ey", "cyl:radial:step_B"),
    (cyl.CylindricalComplexCurlPlan, "step_D", "Hy", "cyl:radial:step_D"),
    (cfm.CylindricalFusedMagneticPairPlan, cfm.CURL_SUB_STEP, "Ey", "cylfused:radial:step_B"),
    (cfe.CylindricalFusedElectricPairPlan, cfe.CURL_SUB_STEP, "Hy", "cylfusedE:radial:step_D"),
)
IDS = ("curl_B", "curl_D", "fused_magnetic_pair", "fused_electric_pair")


def say(line: str) -> None:
    print(line, flush=True)


def _volume(seed: int) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    parts = rng.standard_normal((2,) + SHAPE).astype(numpy.float32)
    return (parts[0] + 1j * parts[1]).astype(numpy.complex64)


def _words(array: numpy.ndarray) -> numpy.ndarray:
    return numpy.ascontiguousarray(array).reshape(-1).view(numpy.uint32)


def _bare(cls, sub_step: str, residency, source_name: str, source: numpy.ndarray,
          radial_name: str):
    """A plan carrying only what ``refresh_prefix`` reads: no kernel, no binding."""
    plan = object.__new__(cls)
    plan.residency = residency
    if cls is cyl.CylindricalComplexCurlPlan:
        plan.sub_step = sub_step
    plan.starts = None
    plan.scratch = None
    plan.prefix_syncs = 0
    plan._radial_host = numpy.zeros(SHAPE, dtype=numpy.complex64)
    plan._radial_name = radial_name
    plan._source_name = source_name
    plan._source_map = {source_name: source}
    plan._xp = numpy
    return plan


def _expected(sub_step: str, source_name: str, words: numpy.ndarray) -> numpy.ndarray:
    return numpy.asarray(cyl.radial_difference(numpy, sub_step, {source_name: words}, None))


def test_the_plans_are_the_ones_that_sum_on_the_host():
    """The table above is the set of plans whose refresh reads a host source."""
    assert cfm.CURL_SUB_STEP == "step_B" and cfe.CURL_SUB_STEP == "step_D"
    for cls, _sub_step, _source, _radial in PLANS:
        assert callable(getattr(cls, "refresh_prefix", None)), cls


class _HeldModeDoors:
    """``device.Residency``'s two doors, on host arrays, with its held-mode contract.

    ``device_words`` stands for what the last launch left on the device. ``sync_out`` of a
    HELD name copies nothing (it marks the mirror device-owned); of an unheld name it
    copies. ``acquire_read`` copies the device's words while the device owns the mirror.
    """

    def __init__(self, name: str, host: numpy.ndarray, device_words: numpy.ndarray,
                 held: bool) -> None:
        self.held = frozenset({name}) if held else frozenset()
        self.name = name
        self.host = host
        self.device_words = device_words
        self.device_owned = True
        self.calls = []

    def sync_out(self, names=None) -> None:
        for name in tuple(names):
            self.calls.append(("sync_out", name))
            if name == self.name:
                if name in self.held:
                    self.device_owned = True
                    continue
                self.host[...] = self.device_words

    def acquire_read(self, name):
        self.calls.append(("acquire_read", name))
        if name == self.name and self.device_owned:
            self.host[...] = self.device_words
            self.device_owned = False
        return self.host

    def sync_in(self, names=None) -> None:
        for name in tuple(names):
            self.calls.append(("sync_in", name))


@pytest.mark.parametrize("held", (True, False), ids=("held", "unheld"))
@pytest.mark.parametrize("cls,sub_step,source_name,radial_name", PLANS, ids=IDS)
def test_the_refresh_sums_the_words_the_device_holds(cls, sub_step, source_name,
                                                     radial_name, held):
    stale, current = _volume(1), _volume(2)
    host = stale.copy()
    doors = _HeldModeDoors(source_name, host, current, held)
    plan = _bare(cls, sub_step, doors, source_name, host, radial_name)
    plan.refresh_prefix()
    expected = _expected(sub_step, source_name, current)
    differing = int(numpy.count_nonzero(_words(plan._radial_host) != _words(expected)))
    say(f"{cls.__name__} {sub_step} held={held}: {differing} / {expected.size * 2} radial "
        f"words differ from the sum of the device's words; calls {doors.calls}")
    assert differing == 0, (cls.__name__, sub_step, held, differing)
    assert numpy.count_nonzero(_words(_expected(sub_step, source_name, stale))
                               != _words(expected)) > 0, "the two volumes must sum apart"
    pull = ("acquire_read", source_name) if held else ("sync_out", source_name)
    assert doors.calls == [pull, ("sync_in", radial_name)], doors.calls
    assert plan.prefix_syncs == 1


@pytest.fixture(scope="module")
def _host_fpu_as_found():
    """Leave the host FPU's subnormal mode as this file found it."""
    from meep_gpu import backends, subnormal_policy  # noqa: PLC0415

    flushing_at_entry = backends.subnormals_flushed()
    try:
        yield
    finally:
        if backends.subnormals_flushed() != flushing_at_entry:
            subnormal_policy._drive_host_fpu(flushing_at_entry)  # noqa: SLF001


def _mps():
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # pragma: no cover - platform gate
        requires_resource_skip("mps_device", "PyTorch is not importable")
    if not getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        requires_resource_skip("mps_device", "this PyTorch reports no usable Apple GPU (MPS)")
    return torch


@pytest.mark.parametrize("cls,sub_step,source_name,radial_name", PLANS, ids=IDS)
def test_under_a_real_hold_the_refresh_reads_through_to_the_device(
        _host_fpu_as_found, cls, sub_step, source_name, radial_name):
    torch = _mps()
    from .metal_kernels import device  # noqa: PLC0415

    stale, current = _volume(3), _volume(4)
    host = stale.copy()
    residency = device.Residency()
    tensor = residency.mirror(source_name, host, dtype=numpy.complex64)
    plan = _bare(cls, sub_step, residency, source_name, host, radial_name)
    residency.mirror(radial_name, plan._radial_host, dtype=numpy.complex64)
    assert residency.arm_hold((source_name,)) == ()
    try:
        # A launch's output: the device's words move and the host's do not; the held
        # sync_out after a launch marks the mirror device-owned and copies nothing.
        tensor.copy_(torch.from_numpy(current.reshape(-1).copy()).to(tensor.device))
        residency.sync_out((source_name,))
        assert residency.state_of(source_name) == device.DEVICE_OWNED
        assert numpy.array_equal(_words(host), _words(stale)), "the host is stale by design"

        plan.refresh_prefix()

        expected = _expected(sub_step, source_name, current)
        differing = int(numpy.count_nonzero(_words(plan._radial_host) != _words(expected)))
        on_device = residency.tensor(radial_name).cpu().numpy().reshape(SHAPE)
        device_differing = int(numpy.count_nonzero(_words(on_device) != _words(expected)))
        say(f"{cls.__name__} {sub_step}: radial {differing} / {expected.size * 2} words "
            f"differ on the host, {device_differing} on the device; source state "
            f"{residency.state_of(source_name)}, copies_out {residency.copies_out}")
        assert differing == 0 and device_differing == 0, (differing, device_differing)
        assert numpy.array_equal(_words(host), _words(current))
        assert residency.state_of(source_name) == device.CLEAN_BOTH
        assert residency.copies_out == 1
        assert plan.prefix_syncs == 1
    finally:
        residency.release_hold()
