"""Lifted-state capsules (``capsule.py``): a driver saved and loaded back word for word.

Every driver here is hand-built under ``recording_construction`` with NumPy alone, as a
lift would build it: the constructor, then the configuration calls. The claim each test
pins is a word compare, never a tolerance: the loaded driver holds the same bytes in
every array the capsule pins, steps to the same bytes, and refuses by name what the
schema does not carry. The one MEEP test lifts a small simulation and loads its capsule
in a process that cannot import MEEP.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from . import capsule
from .capsule import CapsuleRefusal
from .dispersion import LORENTZIAN, Susceptibility
from .driver import FdtdDriver
from .grid import Mirror

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(MEEP_MISSING, reason="[requires_resource][meep] MEEP is not installed")

STEPS = 20


# ---------------------------------------------------------------------------------
# Hand-built drivers, one per family the lift produces
# ---------------------------------------------------------------------------------

def _graded(shape, low, high, axis=0):
    ramp = np.linspace(low, high, shape[axis], dtype=np.float64)
    view = [1, 1, 1]
    view[axis] = shape[axis]
    return np.broadcast_to(ramp.reshape(view), shape).astype(np.float32)


def _folded_pml_3d():
    driver = FdtdDriver(cell_size=(2.0, 1.6, 1.6), resolution=10, force_complex_fields=False,
                        symmetry=(Mirror("Y", -1), Mirror("Z", -1)), boundaries="metallic")
    shape = driver.shape
    driver.set_epsilon_components({name: _graded(shape, 1.0, 4.0 + index, axis=0)
                                   for index, name in enumerate(("Ex", "Ey", "Ez"))})
    driver.setup_pml(3)
    driver.add_source({"component": "Hx", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.6, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
    return driver


def _conductive_2d():
    driver = FdtdDriver(cell_size=(4.0, 4.0, 0.0), dimensions=2, resolution=10,
                        force_complex_fields=False)
    driver.set_epsilon(np.full(driver.shape, 2.25, dtype=np.float32))
    driver.set_conductivity(0.3)
    driver.set_b_conductivity({"Bx": 0.1, "By": 0.2, "Bz": 0.1})
    driver.setup_pml({"x": (5, 5), "y": (5, 5)})
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.5, "center": (0.1, -0.2, 0.0), "size": (0.0, 0.0, 0.0)})
    driver.add_source({"component": "Ez", "frequency": 0.8, "is_integrated": True,
                       "center": (-0.3, 0.2, 0.0), "size": (0.0, 0.0, 0.0)})
    return driver


def _bloch_complex_2d():
    driver = FdtdDriver(cell_size=(3.0, 3.0, 0.0), dimensions=2, resolution=10,
                        force_complex_fields=True, k_point=(0.3, 0.0, 0.0))
    driver.set_epsilon(_graded(driver.shape, 1.0, 3.0, axis=1))
    driver.setup_pml({"y": (5, 5)})
    driver.add_source({"component": "Ez", "frequency": 1.1, "center": (0.2, 0.1, 0.0),
                       "size": (0.0, 0.0, 0.0), "amplitude": 0.5 + 0.25j})
    driver.add_source({"component": "Ex", "frequency": 0.9, "center": (0.0, -0.4, 0.0),
                       "size": (1.0, 0.0, 0.0),
                       "amp_func": lambda x, y, z: complex(1.0 + x, 0.5 * x * x)})
    return driver


def _cylindrical_m1():
    driver = FdtdDriver(cell_size=(2.0, 0.0, 3.0), resolution=10, cylindrical=True, m=1,
                        force_complex_fields=True, boundaries={"z": "metallic"})
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    driver.setup_pml({"x": (0, 5), "z": (5, 5)})
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.6, "center": (0.5, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
    return driver


def _dispersive_2d():
    driver = FdtdDriver(cell_size=(3.0, 3.0, 0.0), dimensions=2, resolution=10,
                        force_complex_fields=False)
    driver.set_epsilon(np.full(driver.shape, 2.0, dtype=np.float32))
    driver.add_susceptibility(Susceptibility(1.1, 0.1, LORENTZIAN), _graded(driver.shape, 0.0, 0.8))
    driver.setup_pml({"x": (5, 5), "y": (5, 5)})
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.5, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
    return driver


def _custom_waveform(time):
    return complex(math.sin(3.0 * time) * math.exp(-0.2 * time), 0.0)


def _custom_source_2d():
    driver = FdtdDriver(cell_size=(3.0, 3.0, 0.0), dimensions=2, resolution=10,
                        force_complex_fields=False)
    driver.set_epsilon(np.full(driver.shape, 1.5, dtype=np.float32))
    driver.setup_pml({"x": (5, 5), "y": (5, 5)})
    driver.add_source({"component": "Hz", "source_type": "custom", "src_func": _custom_waveform,
                       "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
    return driver


CASES = {
    "folded_pml_3d": _folded_pml_3d,
    "conductive_2d": _conductive_2d,
    "bloch_complex_2d": _bloch_complex_2d,
    "cylindrical_m1": _cylindrical_m1,
    "dispersive_2d": _dispersive_2d,
}


def _built(builder):
    with capsule.recording_construction() as recorder:
        driver = builder()
    return driver, recorder


def _word_differences(a, b):
    """Differing bytes per array path, over the union of paths (a missing path counts)."""
    out = {}
    for path in sorted(set(a) | set(b)):
        if path not in a or path not in b:
            out[path] = -1
            continue
        x, y = a[path], b[path]
        if x.dtype != y.dtype or x.shape != y.shape:
            out[path] = -1
            continue
        count = int(np.count_nonzero(x.reshape(-1).view(np.uint8) != y.reshape(-1).view(np.uint8)))
        if count:
            out[path] = count
    return out


def _round_trip(tmp_path, builder, name="driver.capsule"):
    driver, recorder = _built(builder)
    info = capsule.write_capsule(tmp_path / name, driver, recorder)
    stored = capsule.read_capsule(info.path)
    loaded = stored.build_driver()
    return driver, loaded, stored, info


# ---------------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(CASES))
def test_a_loaded_capsule_holds_every_word_and_steps_to_the_same_words(tmp_path, case):
    driver, loaded, stored, _ = _round_trip(tmp_path, CASES[case])
    before, after = capsule.state_arrays(driver), capsule.state_arrays(loaded)
    assert len(before) > 10, sorted(before)
    assert _word_differences(before, after) == {}
    assert stored.load_report["arrays_rebuilt_differently"] == 0, stored.load_report
    assert stored.load_report["scalars_rebuilt_differently"] == 0, stored.load_report
    for step in range(1, STEPS + 1):
        driver.step()
        loaded.step()
        if step in (1, 2, 5, 10, STEPS):
            assert capsule.state_digest(driver) == capsule.state_digest(loaded), (case, step)
    assert _word_differences(capsule.state_arrays(driver), capsule.state_arrays(loaded)) == {}
    driver.close()
    loaded.close()


def test_the_capsule_pins_more_than_the_fields(tmp_path):
    """PML coefficient tables, source deposition tables and grid coordinates are state too.

    Each array is keyed by the first path that reaches it, and public attributes are
    walked before private ones, so the PML tables are named under ``driver.pml``.
    """
    driver, loaded, _, _ = _round_trip(tmp_path, _folded_pml_3d)
    paths = set(capsule.state_arrays(loaded))
    for expected in ("driver.pml.sinv_x", "driver.pml.kms_y_h", "driver.fields.grid.x",
                     "driver._sources[0]._point_amps", "driver.fields.fu_Dx"):
        assert expected in paths, (expected, sorted(paths))


def test_an_array_reached_under_two_paths_is_one_array_after_loading(tmp_path):
    """The isotropic conductivity is ONE array under three keys; the loaded driver keeps that."""
    driver, loaded, _, _ = _round_trip(tmp_path, _conductive_2d)
    original = driver.fields._conductivity  # noqa: SLF001
    rebuilt = loaded.fields._conductivity  # noqa: SLF001
    assert original["Dx"] is original["Dy"] is original["Dz"], "the premise: one shared array"
    assert rebuilt["Dx"] is rebuilt["Dy"] is rebuilt["Dz"]
    assert loaded.fields.grid is loaded.grid
    assert loaded.pml.grid is loaded.grid
    assert all(source.grid is loaded.grid for source in loaded._sources)  # noqa: SLF001


def test_a_capsule_written_mid_run_resumes_to_the_same_words(tmp_path):
    driver, recorder = _built(_conductive_2d)
    for _ in range(7):
        driver.step()
    info = capsule.write_capsule(tmp_path / "mid.capsule", driver, recorder)
    stored = capsule.read_capsule(info.path)
    assert stored.step_count == 7
    loaded = stored.build_driver()
    assert loaded.step_count == 7
    assert _word_differences(capsule.state_arrays(driver), capsule.state_arrays(loaded)) == {}
    for _ in range(13):
        driver.step()
        loaded.step()
    assert capsule.state_digest(driver) == capsule.state_digest(loaded)


def test_a_capsule_written_after_a_read_leaves_the_readback_memos_out(tmp_path):
    """A read builds the readback's memos; they are rebuilt on demand, never carried.

    ``get_field_point`` caches its stencil and a B read with H stored caches the run's
    B/H alias rule (``_point_stencils``, ``_h_alias_cache``). Carried, the second is a
    tuple the loader's replay rebuilds as ``None``, and the writer refuses the capsule;
    left out, the loaded driver reads the same words.
    """
    driver, recorder = _built(_conductive_2d)
    for _ in range(7):
        driver.step()
    point = complex(driver.get_field_point("Ez", (0.13, -0.27, 0.0)))
    bx = np.asarray(driver.get_field("Bx", cell_centered=True))
    assert driver._point_stencils and driver._h_alias_cache is not None  # noqa: SLF001
    info = capsule.write_capsule(tmp_path / "read.capsule", driver, recorder)
    loaded = capsule.read_capsule(info.path).build_driver()
    paths = capsule.state_arrays(loaded)
    assert not [path for path in paths if "_point_stencils" in path or "_h_alias_cache" in path]
    assert capsule.state_digest(driver) == capsule.state_digest(loaded)
    assert complex(loaded.get_field_point("Ez", (0.13, -0.27, 0.0))) == point
    assert np.array_equal(np.asarray(loaded.get_field("Bx", cell_centered=True)).view(np.uint32),
                          bx.view(np.uint32))
    driver.close()
    loaded.close()


def test_the_stored_state_is_pasted_over_what_the_replay_rebuilt(tmp_path):
    """The loaded state is the stored state; the replay's own values are reported, not kept."""
    driver, recorder = _built(_folded_pml_3d)
    sinv = driver.pml.sinv_x
    sinv[0] = np.nextafter(sinv[0], np.float32(2.0))
    source = driver._sources[0]  # noqa: SLF001
    envelope = source.envelope
    object.__setattr__(envelope, "_omega", math.nextafter(envelope._omega, 10.0))  # noqa: SLF001
    info = capsule.write_capsule(tmp_path / "pasted.capsule", driver, recorder)
    stored = capsule.read_capsule(info.path)
    loaded = stored.build_driver()
    assert stored.load_report["arrays_rebuilt_differently_paths"] == ["driver.pml.sinv_x"], stored.load_report
    assert stored.load_report["scalars_rebuilt_differently_paths"] == [
        "driver._sources[0].envelope._omega"], stored.load_report
    assert np.array_equal(loaded.pml.sinv_x.view(np.uint32), driver.pml.sinv_x.view(np.uint32))
    assert loaded._sources[0].envelope._omega == envelope._omega  # noqa: SLF001
    for _ in range(10):
        driver.step()
        loaded.step()
    assert capsule.state_digest(driver) == capsule.state_digest(loaded)


def test_lift_attributes_come_back_through_their_own_constructors(tmp_path):
    """``lift_record`` and an empty ``MigratedMonitors`` are set by a lift, not by a call."""
    from .from_meep import MigratedMonitors  # noqa: PLC0415

    driver, recorder = _built(_conductive_2d)
    driver.lift_record = {"meep_version": "1.33.0", "meep_precision": "single"}
    driver.migrated_monitors = MigratedMonitors()
    info = capsule.write_capsule(tmp_path / "lifted.capsule", driver, recorder)
    loaded = capsule.load_capsule(info.path)
    assert loaded.lift_record == driver.lift_record
    assert type(loaded.migrated_monitors) is MigratedMonitors and len(loaded.migrated_monitors) == 0
    assert loaded.migrated_monitors.pairs() == ()


class _MeepTime:
    """A stand-in for a lifted source's MEEP ``src_time`` (not a package object)."""

    is_integrated = False

    def dipole(self, time):
        return complex(0.25 * time, -0.125)

    def current(self, time, dt):
        return complex(0.5 * time, dt)


def test_a_source_whose_currents_came_from_meep_loads_only_with_its_currents(tmp_path):
    """The bound ``src_time`` is not carried; the capsule names the sources that held one."""
    driver, recorder = _built(_conductive_2d)
    driver._sources[0].bind_meep_src_time(_MeepTime())  # noqa: SLF001
    info = capsule.write_capsule(tmp_path / "bound.capsule", driver, recorder)
    stored = capsule.read_capsule(info.path)
    assert stored.provenance["source_times_from_meep"] == [0]
    with pytest.raises(CapsuleRefusal, match="MEEP src_time"):
        stored.build_driver()
    with capsule.recording_currents(driver) as recording:
        for _ in range(6):
            driver.step()
    loaded = stored.build_driver(currents=recording.table())
    for _ in range(6):
        loaded.step()
    assert capsule.state_digest(loaded) == capsule.state_digest(driver)


def test_writing_the_same_driver_twice_gives_the_same_bytes(tmp_path):
    driver, recorder = _built(_bloch_complex_2d)
    first = capsule.write_capsule(tmp_path / "one.capsule", driver, recorder)
    second = capsule.write_capsule(tmp_path / "two.capsule", driver, recorder)
    assert first.file_sha256 == second.file_sha256
    assert first.document_sha256 == second.document_sha256


def test_recording_restores_the_driver_class_on_exit():
    originals = {name: FdtdDriver.__dict__[name] for name in capsule.CONSTRUCTION_CALLS + ("__init__",)}
    with capsule.recording_construction():
        assert FdtdDriver.__dict__["add_source"] is not originals["add_source"]
    assert {name: FdtdDriver.__dict__[name] for name in originals} == originals
    with pytest.raises(ValueError, match="resolution must be finite"):
        with capsule.recording_construction():
            FdtdDriver(cell_size=(1.0, 1.0, 1.0), resolution=-1.0)
    assert {name: FdtdDriver.__dict__[name] for name in originals} == originals
    with pytest.raises(CapsuleRefusal, match="already open"):
        with capsule.recording_construction():
            with capsule.recording_construction():
                pass


def test_a_callable_argument_is_answered_from_its_recorded_calls(tmp_path):
    """``amp_func`` is replayed from the calls the lift made; any other call is refused."""
    driver, loaded, stored, _ = _round_trip(tmp_path, _bloch_complex_2d)
    replayed = loaded._sources[1].amp_func  # noqa: SLF001
    assert type(replayed).__name__ == "_ReplayedCallable"
    assert len(replayed.calls) > 0
    with pytest.raises(CapsuleRefusal, match="did not record"):
        replayed(12.5, 0.0, 0.0)


# ---------------------------------------------------------------------------------
# Floats as bit patterns
# ---------------------------------------------------------------------------------

def _double_from_bits(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def test_every_float_crosses_as_its_bit_pattern():
    smallest_subnormal = _double_from_bits(1)
    largest_subnormal = _double_from_bits(0x000FFFFFFFFFFFFF)
    quiet_nan_payload = _double_from_bits(0x7FF8000000000123)
    float32_subnormal = np.frombuffer(struct.pack("<I", 0x00000005), dtype=np.float32)[0]
    values = [0.0, -0.0, smallest_subnormal, -largest_subnormal, quiet_nan_payload,
              math.inf, -math.inf, 1.0 / 3.0,
              complex(-0.0, smallest_subnormal), float32_subnormal, np.float64(-0.0), True, 7]
    for value in values:
        back = capsule._scalar_from_node(json.loads(json.dumps(capsule._scalar_node(value))))  # noqa: SLF001
        assert type(back) is type(value)
        if isinstance(value, complex):
            assert struct.pack("<dd", back.real, back.imag) == struct.pack("<dd", value.real, value.imag)
        elif isinstance(value, np.generic):
            assert back.tobytes() == value.tobytes()
        elif isinstance(value, float):
            assert struct.pack("<d", back) == struct.pack("<d", value)
        else:
            assert back == value


def test_a_capsule_document_holds_no_decimal_float(tmp_path):
    driver, recorder = _built(_dispersive_2d)
    info = capsule.write_capsule(tmp_path / "bits.capsule", driver, recorder)
    with zipfile.ZipFile(info.path) as bundle:
        text = bundle.read("capsule.json").decode("utf-8")

    def refuse(literal):
        raise AssertionError(f"a decimal float {literal!r} is in the capsule document")

    json.loads(text, parse_float=refuse)


# ---------------------------------------------------------------------------------
# Injected currents
# ---------------------------------------------------------------------------------

def test_injected_currents_replay_the_recorded_values_exactly(tmp_path):
    driver, recorder = _built(_conductive_2d)
    info = capsule.write_capsule(tmp_path / "currents.capsule", driver, recorder)
    with capsule.recording_currents(driver) as recording:
        for _ in range(STEPS):
            driver.step()
    table = recording.table()
    assert table.steps_recorded == STEPS
    assert [len(entry["calls"]["current"]) for entry in table.sources] == [STEPS, 0]
    assert [len(entry["calls"]["dipole"]) for entry in table.sources] == [0, STEPS]
    assert all("current" not in vars(source) and "dipole" not in vars(source)
               for source in driver._sources)  # noqa: SLF001
    written = tmp_path / "currents.json"
    table.write(written)
    read_back = capsule.CurrentsTable.read(written)
    loaded = capsule.load_capsule(info.path, currents=read_back)
    for _ in range(STEPS):
        loaded.step()
    assert capsule.state_digest(loaded) == capsule.state_digest(driver)
    with pytest.raises(CapsuleRefusal, match="does not hold"):
        loaded.step()


def test_a_float32_sized_change_in_one_injected_current_changes_the_words(tmp_path):
    """The control: the injected value reaches the field, so the seam is load-bearing.

    The change is a relative 2**-20 (about eight float32 ulps) of the largest recorded
    current: a one-ulp change of a double is below float32 resolution, and the deposit
    rounds to float32, so it could vanish without saying anything about the seam.
    """
    driver, recorder = _built(_folded_pml_3d)
    info = capsule.write_capsule(tmp_path / "planted.capsule", driver, recorder)
    with capsule.recording_currents(driver) as recording:
        for _ in range(STEPS):
            driver.step()
    table = recording.table()
    planted = copy.deepcopy(table)
    calls = planted.sources[0]["calls"]["current"]
    key = max(calls, key=lambda name: abs(calls[name]))
    assert abs(calls[key]) > 0.0
    calls[key] = calls[key] * (1.0 + 2.0 ** -20)
    clean = capsule.load_capsule(info.path, currents=table)
    dirty = capsule.load_capsule(info.path, currents=planted)
    for _ in range(STEPS):
        clean.step()
        dirty.step()
    assert capsule.state_digest(clean) == capsule.state_digest(driver)
    assert capsule.state_digest(dirty) != capsule.state_digest(driver)


def test_a_custom_waveform_is_refused_without_injected_currents(tmp_path):
    driver, recorder = _built(_custom_source_2d)
    info = capsule.write_capsule(tmp_path / "custom.capsule", driver, recorder)
    with pytest.raises(CapsuleRefusal, match="custom waveform"):
        capsule.load_capsule(info.path)
    with capsule.recording_currents(driver) as recording:
        for _ in range(STEPS):
            driver.step()
    loaded = capsule.load_capsule(info.path, currents=recording.table())
    for _ in range(STEPS):
        loaded.step()
    assert capsule.state_digest(loaded) == capsule.state_digest(driver)


def test_a_currents_table_for_other_sources_is_refused(tmp_path):
    driver, recorder = _built(_conductive_2d)
    other, other_recorder = _built(_folded_pml_3d)
    with capsule.recording_currents(other) as recording:
        other.step()
    with pytest.raises(CapsuleRefusal, match="describes 1 source"):
        capsule.install_currents(driver, recording.table())


# ---------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------

def _rewrite(path, mutate_document=None, mutate_member=None):
    with zipfile.ZipFile(path) as bundle:
        members = {name: bundle.read(name) for name in bundle.namelist()}
    document = json.loads(members["capsule.json"])
    if mutate_document is not None:
        mutate_document(document)
    members["capsule.json"] = json.dumps(document).encode("utf-8")
    if mutate_member is not None:
        mutate_member(members)
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)


def test_a_driver_not_built_under_recording_is_refused(tmp_path):
    driver = _conductive_2d()
    with capsule.recording_construction() as recorder:
        pass
    with pytest.raises(CapsuleRefusal, match="not constructed under recording_construction"):
        capsule.write_capsule(tmp_path / "x.capsule", driver, recorder)


def test_a_driver_with_a_monitor_is_refused(tmp_path):
    with capsule.recording_construction() as recorder:
        driver = _conductive_2d()
        driver.add_dft_monitor([1.0], ("Ez",), center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 0.0))
    with pytest.raises(CapsuleRefusal, match="monitors"):
        capsule.write_capsule(tmp_path / "x.capsule", driver, recorder)


def test_an_unrecorded_callable_in_the_state_is_refused(tmp_path):
    driver, recorder = _built(_bloch_complex_2d)
    object.__setattr__(driver._sources[1], "amp_func", lambda x, y, z: 1.0)  # noqa: SLF001
    with pytest.raises(CapsuleRefusal, match="was not recorded"):
        capsule.write_capsule(tmp_path / "x.capsule", driver, recorder)


def test_another_schema_version_is_refused_by_name(tmp_path):
    driver, recorder = _built(_conductive_2d)
    info = capsule.write_capsule(tmp_path / "v.capsule", driver, recorder)
    _rewrite(info.path, mutate_document=lambda document: document.__setitem__("version", 2))
    with pytest.raises(CapsuleRefusal, match="schema version 2"):
        capsule.read_capsule(info.path)


def test_a_member_that_fails_its_digest_is_refused(tmp_path):
    driver, recorder = _built(_conductive_2d)
    info = capsule.write_capsule(tmp_path / "t.capsule", driver, recorder)
    stored = capsule.read_capsule(info.path)
    entry = stored.document["arrays"][-1]

    def flip_last_byte(members):
        data = bytearray(members[entry["member"]])
        data[-1] ^= 0x01
        members[entry["member"]] = bytes(data)

    _rewrite(info.path, mutate_member=flip_last_byte)
    with pytest.raises(CapsuleRefusal, match="fails its sha256"):
        capsule.load_capsule(info.path)


def test_an_unlisted_member_is_refused(tmp_path):
    driver, recorder = _built(_conductive_2d)
    info = capsule.write_capsule(tmp_path / "m.capsule", driver, recorder)
    _rewrite(info.path, mutate_member=lambda members: members.__setitem__("extra.npy", b"0"))
    with pytest.raises(CapsuleRefusal, match="differ from the document"):
        capsule.read_capsule(info.path)


def test_a_rebuilt_graph_of_another_shape_is_refused_by_name(tmp_path):
    """A capsule from a tree whose driver held an attribute this tree's does not."""
    driver, recorder = _built(_conductive_2d)
    info = capsule.write_capsule(tmp_path / "shape.capsule", driver, recorder)
    _rewrite(info.path, mutate_document=lambda document: document["state"]["s"].__setitem__(
        "_attribute_from_another_tree", 3))
    with pytest.raises(CapsuleRefusal, match="_attribute_from_another_tree"):
        capsule.load_capsule(info.path)


def test_an_array_of_another_shape_is_refused_by_name(tmp_path):
    driver, recorder = _built(_conductive_2d)
    info = capsule.write_capsule(tmp_path / "array.capsule", driver, recorder)
    stored = capsule.read_capsule(info.path)
    index = next(i for i, entry in enumerate(stored.document["arrays"])
                 if entry["path"] == "driver.fields.Ez")
    replacement = np.zeros((3, 3, 1), dtype=np.float32)

    def swap(members):
        buffer = io.BytesIO()
        np.save(buffer, replacement, allow_pickle=False)
        members[stored.document["arrays"][index]["member"]] = buffer.getvalue()

    def fix_entry(document):
        import hashlib  # noqa: PLC0415

        entry = document["arrays"][index]
        entry["shape"] = [3, 3, 1]
        entry["sha256"] = hashlib.sha256(replacement.tobytes()).hexdigest()

    _rewrite(info.path, mutate_document=fix_entry, mutate_member=swap)
    with pytest.raises(CapsuleRefusal, match=r"driver\.fields\.Ez"):
        capsule.load_capsule(info.path)


# ---------------------------------------------------------------------------------
# A lifted simulation, loaded where MEEP cannot be imported
# ---------------------------------------------------------------------------------

_LOAD_WITHOUT_MEEP = r"""
import sys
class _NoMeep:
    def find_spec(self, name, path=None, target=None):
        if name == "meep" or name.startswith("meep."):
            raise ImportError("MEEP is blocked in this process")
        return None
sys.meta_path.insert(0, _NoMeep())
from meep_gpu import capsule
driver = capsule.load_capsule(sys.argv[1], currents=capsule.CurrentsTable.read(sys.argv[2]))
digests = [capsule.state_digest(driver)]
for _ in range(int(sys.argv[3])):
    driver.step()
digests.append(capsule.state_digest(driver))
assert "meep" not in sys.modules
print(" ".join(digests))
"""


@requires_meep
@skip_without_meep
def test_a_lifted_capsule_loads_and_steps_identically_where_meep_cannot_be_imported(tmp_path):
    import meep as mp  # noqa: PLC0415

    from .from_meep import lift_simulation  # noqa: PLC0415

    sim = mp.Simulation(resolution=10, cell_size=mp.Vector3(4, 4), boundary_layers=[mp.PML(0.5)],
                        geometry=[mp.Block(size=mp.Vector3(1, 1, mp.inf), material=mp.Medium(epsilon=4))],
                        sources=[mp.Source(src=mp.GaussianSource(1.0, fwidth=0.5), center=mp.Vector3(0.6, 0.3),
                                           component=mp.Ez)])
    sim.num_chunks = 4  # A division other than the default, so the layout is the lift's own.
    with capsule.recording_construction() as recorder:
        driver = lift_simulation(sim, prefer_gpu=False)
    info = capsule.write_capsule(tmp_path / "lifted.capsule", driver, recorder, sim=sim)
    stored = capsule.read_capsule(info.path)
    division = stored.chunk_division
    assert division is not None and len(division["boxes"]) == len(division["owners"]) >= 1
    assert "set_chunk_layout" in [name for name, _ in recorder.log_for(driver).calls]
    assert stored.provenance["meep"]["version"] == mp.__version__
    # The lift formed the currents through MEEP's own src_time, which no capsule carries.
    assert stored.provenance["source_times_from_meep"] == [0]
    with pytest.raises(CapsuleRefusal, match="MEEP src_time"):
        stored.build_driver()
    expected = [capsule.state_digest(driver)]
    with capsule.recording_currents(driver) as recording:
        for _ in range(STEPS):
            driver.step()
    expected.append(capsule.state_digest(driver))
    table_path = tmp_path / "currents.json"
    recording.table().write(table_path)
    loaded_here = stored.build_driver(currents=recording.table())
    assert loaded_here.chunk_layout.source == driver.chunk_layout.source == "meep"
    assert loaded_here.chunk_layout.to_record() == driver.chunk_layout.to_record()
    assert np.array_equal(np.asarray(loaded_here.chunk_bounds_record()),
                          np.asarray(driver.chunk_bounds_record()))
    loaded_here.close()
    root = Path(__file__).resolve().parent.parent
    environment = {**os.environ, "PYTHONPATH": str(root), "MEEP_GPU_DISPATCH": "0"}
    result = subprocess.run([sys.executable, "-c", _LOAD_WITHOUT_MEEP, str(info.path), str(table_path),
                             str(STEPS)], capture_output=True, text=True, env=environment, timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.split() == expected


def test_a_callable_the_table_cannot_hold_answers_the_run_and_refuses_the_capsule():
    """Recording never changes what the caller's function returns; the writer refuses instead."""
    answers = iter([1.0, 2.0])
    for function, arguments, reason in (
            (lambda *args: np.ones(2), (0.5,), "cannot be keyed or stored"),
            (lambda x: next(answers), (0.5,), "two different values"),
            (lambda x: 1.0, (np.zeros(3),), "cannot be keyed or stored")):
        proxy = capsule._RecordingCallable(function, "f")  # noqa: SLF001
        first = proxy(*arguments)
        proxy(*arguments)
        assert proxy.unrecordable is not None and reason in proxy.unrecordable, proxy.unrecordable
        assert first is not None
        with pytest.raises(CapsuleRefusal, match="cannot be replayed"):
            capsule._CallableTable().add(proxy)  # noqa: SLF001
