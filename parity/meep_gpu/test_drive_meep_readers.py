"""The driven pass's ``get_array`` / ``get_dft_array`` follow the MEEP release in hand.

MEEP 1.35.0 validates both readers' arguments (#3320): ``component`` is checked by
``_check_component``, a ``Volume`` first argument is a ``TypeError``, ``frequency``
is for material components only, ``vol`` excludes ``center``/``size``, a supplied
``arr`` must match the slice's dtype and shape and be C-contiguous and writeable,
the options after ``size`` are keyword-only, ``dft_obj`` is required and ``num_freq``
must index the monitor's frequencies. MEEP's own tests assert those exceptions, so
the emulations ``bind_engine_readers`` installs raise the same exception type as
MEEP's own reader for each bad call on that release, and keep the earlier signature
on 1.33.0. Each case below is put to MEEP's own reader on an initialized simulation
and to the emulation, and the two exception types are compared; the emulation reads
a stand-in result, since only its argument handling is under test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    MEEP_MISSING, reason="CPU MEEP is not installed in this environment"
)

_DRIVER = Path(__file__).with_name("drive_meep_test_assertions.py")
#: The skip on a MEEP before 1.35.0, which has none of the checks these cases compare.
_PREDATES = ("[requires_resource][meep_get_array_checks] MEEP {version} predates MEEP "
             "1.35.0's reader checks (#3320); the emulation keeps that release's signature")


def _driver_module():
    spec = importlib.util.spec_from_file_location("drive_meep_test_assertions_readers", _DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Result:
    """Stands in for GpuRunResult: arrays of MEEP's own shapes, and the package's refusal."""

    def __init__(self, shapes):
        self.shapes = shapes
        self.driver = None

    def get_array(self, name):
        return np.zeros(self.shapes[name], dtype=np.complex128)

    def get_dft_array(self, dft_obj, name, num_freq):
        kind = type(dft_obj).__name__
        if kind != "DftFields":
            raise TypeError(f"get_dft_array reads DftFields monitors (sim.add_dft_fields) only; "
                            f"a {kind} is refused: stand-in")
        return np.zeros(3, dtype=np.complex128)


class _Simulation:
    """The attributes bind_engine_readers reads and rebinds, and nothing that steps."""

    def __init__(self, meep_sim):
        self.get_array = meep_sim.get_array
        self.dft_objects = []
        self.fields = None


@pytest.fixture(scope="module")
def readers():
    import meep as mp

    module = _driver_module()
    # A source, so MEEP allocates the field components its DFT objects need.
    meep_sim = mp.Simulation(cell_size=mp.Vector3(1, 1, 0), resolution=10, sources=[
        mp.Source(mp.GaussianSource(1.0, fwidth=0.5), mp.Ez, mp.Vector3())])
    meep_sim.init_sim()
    monitor = meep_sim.add_dft_fields([mp.Ez], 1.0, 0, 1)
    energy = meep_sim.add_energy(1.0, 0, 1, mp.EnergyRegion(
        center=mp.Vector3(), size=mp.Vector3(0.5, 0)))
    shapes = {"Ez": np.asarray(meep_sim.get_array(component=mp.Ez)).shape}
    emulated = _Simulation(meep_sim)
    module.bind_engine_readers(mp, emulated, _Result(shapes), [])
    strict = module.meep_release(mp.__version__) >= module.GET_ARRAY_CHECKS_FROM
    return mp, module, meep_sim, emulated, monitor, energy, strict


def _raised(call):
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the type is what is compared
        return type(exc)
    return None


def _get_array_cases(mp, shape):
    whole = mp.Volume(center=mp.Vector3(), size=mp.Vector3(1, 1, 0))
    single = np.float32 if mp.is_single_precision() else np.float64
    return {
        "volume_first": lambda get: get(whole),
        "component_not_a_constant": lambda get: get("Ez"),
        "component_out_of_range": lambda get: get(-3),
        "frequency_on_a_field": lambda get: get(mp.Ez, frequency=1.0),
        "vol_and_center": lambda get: get(mp.Ez, vol=whole, center=mp.Vector3()),
        "cmplx_positional": lambda get: get(mp.Ez, None, None, None, True),
        "arr_not_an_array": lambda get: get(component=mp.Ez, arr=[0.0]),
        "arr_wrong_dtype": lambda get: get(component=mp.Ez, arr=np.zeros(shape, np.int32)),
        "arr_wrong_shape": lambda get: get(component=mp.Ez,
                                           arr=np.zeros((shape[0] + 1,) + shape[1:], single)),
        "arr_not_contiguous": lambda get: get(component=mp.Ez,
                                              arr=np.zeros(shape, single, order="F")),
        "arr_read_only": lambda get: get(component=mp.Ez, arr=_read_only(np.zeros(shape, single))),
    }


def _read_only(array):
    array.flags.writeable = False
    return array


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", [
    "volume_first", "component_not_a_constant", "component_out_of_range",
    "frequency_on_a_field", "vol_and_center", "cmplx_positional", "arr_not_an_array",
    "arr_wrong_dtype", "arr_wrong_shape", "arr_not_contiguous", "arr_read_only",
])
def test_a_bad_get_array_call_raises_what_this_meeps_own_reader_raises(readers, case):
    mp, _module, meep_sim, emulated, _monitor, _energy, strict = readers
    if not strict:
        pytest.skip(_PREDATES.format(version=mp.__version__))
    shape = np.asarray(meep_sim.get_array(component=mp.Ez)).shape
    call = _get_array_cases(mp, shape)[case]
    theirs = _raised(lambda: call(meep_sim.get_array))
    mine = _raised(lambda: call(emulated.get_array))
    assert theirs is not None, f"{case}: MEEP {mp.__version__} accepted the call"
    assert mine is theirs, f"{case}: MEEP raises {theirs.__name__}, the emulation {mine}"


@requires_meep
@skip_without_meep
def test_a_good_get_array_call_still_reads_the_engine(readers):
    mp, _module, meep_sim, emulated, _monitor, _energy, strict = readers
    shape = np.asarray(meep_sim.get_array(component=mp.Ez)).shape
    single = np.float32 if mp.is_single_precision() else np.float64
    values = emulated.get_array(component=mp.Ez)
    assert values.shape == shape and values.dtype == single
    supplied = np.ones(shape, single)
    assert emulated.get_array(component=mp.Ez, arr=supplied) is supplied
    assert not supplied.any(), "the stand-in's zeros were not written into the array in place"
    if not strict:
        # MEEP 1.33.0's options are positional: cmplx is the fifth argument.
        assert emulated.get_array(mp.Ez, None, None, None, True).dtype.kind == "c"


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("case", ["dft_obj_none", "component_out_of_range",
                                  "num_freq_past_the_end", "num_freq_negative",
                                  "num_freq_not_an_integer"])
def test_a_bad_get_dft_array_call_raises_what_this_meeps_own_reader_raises(readers, case):
    mp, _module, meep_sim, emulated, monitor, _energy, strict = readers
    call = {
        "dft_obj_none": lambda get: get(None, mp.Ez, 0),
        "component_out_of_range": lambda get: get(monitor, -3, 0),
        "num_freq_past_the_end": lambda get: get(monitor, mp.Ez, 1),
        "num_freq_negative": lambda get: get(monitor, mp.Ez, -1),
        "num_freq_not_an_integer": lambda get: get(monitor, mp.Ez, 0.5),
    }[case]
    if not strict:
        pytest.skip(_PREDATES.format(version=mp.__version__))
    theirs = _raised(lambda: call(meep_sim.get_dft_array))
    mine = _raised(lambda: call(emulated.get_dft_array))
    assert theirs is not None, f"{case}: MEEP {mp.__version__} accepted the call"
    assert mine is theirs, f"{case}: MEEP raises {theirs.__name__}, the emulation {mine}"


@requires_meep
@skip_without_meep
def test_a_monitor_the_package_refuses_is_a_reader_this_driver_does_not_serve(readers):
    """An energy monitor: readable through MEEP 1.35.0's get_dft_array, refused by the
    package by name, so the driven pass records the row as not driven."""
    mp, module, _meep_sim, emulated, monitor, energy, _strict = readers
    with pytest.raises(module.UnsupportedDrive, match="a DftEnergy is refused"):
        emulated.get_dft_array(energy, mp.Ez, 0)
    assert emulated.get_dft_array(monitor, mp.Ez, 0).shape == (3,)


def test_the_release_is_read_from_meeps_version_string():
    module = _driver_module()
    assert module.meep_release("1.35.0") == (1, 35, 0)
    assert module.meep_release("1.33.0") < module.GET_ARRAY_CHECKS_FROM
    assert module.meep_release("1.36.0rc1") >= module.GET_ARRAY_CHECKS_FROM
    assert module.meep_release("unknown") == (0, 0, 0)
