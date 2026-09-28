"""The MEEP build's precision, read at lift time: recorded, announced, never refused.

The engine steps single precision whatever precision MEEP was built with. A lift
reads the build's precision, records it on the driver as ``driver.lift_record``
(surfaced as ``GpuRunResult.lift_record``), and when the build is double precision
prints one line per process saying what the engine steps and what agreement to
expect. The lift proceeds either way.

The first half needs no MEEP: it hands the reader stand-ins for the module. The
second half lifts a real ``mp.Simulation`` and is skipped by a declared resource
where MEEP is not installed. There the build's answer is substituted at the
package's reader and never on the ``meep`` module: MEEP's own Python layer asks the
same function how wide its arrays are, so changing it under MEEP corrupts what MEEP
hands back.
"""
# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import types

import numpy as np
import pytest

from conftest import requires_resource_skip

from meep_gpu import from_meep, lift_simulation, run_on_gpu

NOTICE_HEAD = "meep_gpu: this MEEP build is double precision"


@pytest.fixture(autouse=True)
def _nothing_announced():
    from_meep._PRECISION_ANNOUNCED.clear()  # noqa: SLF001
    yield
    from_meep._PRECISION_ANNOUNCED.clear()  # noqa: SLF001


def _notices(err: str) -> list:
    return [line for line in err.splitlines() if line.startswith(NOTICE_HEAD)]


def _raises():
    raise RuntimeError("this build does not say")


# --- the reader, with no MEEP -------------------------------------------------------------


@pytest.mark.parametrize("module,precision", [
    (types.SimpleNamespace(__version__="1.33.0", is_single_precision=lambda: True),
     "single"),
    (types.SimpleNamespace(__version__="1.33.0", is_single_precision=lambda: False),
     "double"),
    (types.SimpleNamespace(__version__="1.20.0"), None),
    (types.SimpleNamespace(__version__="1.33.0", is_single_precision=_raises), None),
])
def test_the_build_is_read_and_an_unreadable_one_is_recorded_as_unread(module, precision):
    assert from_meep._meep_build(module) == {  # noqa: SLF001
        "meep_version": module.__version__,
        "meep_precision": precision,
        "engine_precision": "single",
    }


def test_a_module_with_no_version_is_recorded_without_one():
    build = from_meep._meep_build(types.SimpleNamespace())  # noqa: SLF001
    assert build == {"meep_version": None, "meep_precision": None,
                     "engine_precision": "single"}


def test_a_double_precision_build_is_announced_once_per_process(capsys):
    build = {"meep_version": "1.33.0", "meep_precision": "double",
             "engine_precision": "single"}
    for _ in range(3):
        from_meep._announce_precision(build)  # noqa: SLF001
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = _notices(captured.err)
    assert lines == [from_meep.DOUBLE_PRECISION_NOTICE], captured.err
    assert "the engine steps single precision" in lines[0]
    assert "The lift proceeds" in lines[0]
    assert "agreement" in lines[0] and "single-precision rounding" in lines[0]


@pytest.mark.parametrize("precision", ["single", None])
def test_a_single_or_unread_build_prints_nothing(capsys, precision):
    from_meep._announce_precision(  # noqa: SLF001
        {"meep_version": "1.33.0", "meep_precision": precision,
         "engine_precision": "single"})
    captured = capsys.readouterr()
    assert captured.out == "" and _notices(captured.err) == []


# --- the lift, with MEEP ------------------------------------------------------------------


def _import_meep_or_skip():
    try:
        import meep as mp  # noqa: PLC0415
    except ImportError:
        requires_resource_skip("meep", "the lift reads an mp.Simulation")
    mp.verbosity(0)
    return mp


def _substitute_precision(monkeypatch, single: bool) -> None:
    """Have the lift read a build of the given precision; the real reader still runs."""
    reader = from_meep._meep_build  # noqa: SLF001

    def substituted(module):
        return reader(types.SimpleNamespace(
            __version__=module.__version__, is_single_precision=lambda: single))

    monkeypatch.setattr(from_meep, "_meep_build", substituted)


def _simulation(mp):
    return mp.Simulation(
        cell_size=mp.Vector3(6, 6, 0), resolution=10,
        boundary_layers=[mp.PML(1.0)],
        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5), component=mp.Ez,
                           center=mp.Vector3())])


@pytest.mark.parametrize("single,precision", [(True, "single"), (False, "double")])
def test_the_lift_records_the_precision_and_refuses_nothing(
        monkeypatch, capsys, single, precision):
    mp = _import_meep_or_skip()
    _substitute_precision(monkeypatch, single)
    for _ in range(2):
        driver = lift_simulation(_simulation(mp), prefer_gpu=False)
        try:
            assert driver.lift_record == {
                "meep_version": mp.__version__,
                "meep_precision": precision,
                "engine_precision": "single",
            }
            driver.step()
            assert np.asarray(driver.get_field("Ez")).dtype == np.float32
        finally:
            driver.close()
    lines = _notices(capsys.readouterr().err)
    assert len(lines) == (0 if single else 1), lines


def test_the_record_says_what_this_build_is(capsys):
    """Nothing substituted: the record is this MEEP's own answer."""
    mp = _import_meep_or_skip()
    single = bool(mp.is_single_precision())
    driver = lift_simulation(_simulation(mp), prefer_gpu=False)
    try:
        assert driver.lift_record["meep_precision"] == ("single" if single else "double")
        assert driver.lift_record["meep_version"] == mp.__version__
    finally:
        driver.close()
    assert len(_notices(capsys.readouterr().err)) == (0 if single else 1)


def test_the_run_result_carries_the_record(monkeypatch, capsys):
    mp = _import_meep_or_skip()
    _substitute_precision(monkeypatch, False)
    result = run_on_gpu(_simulation(mp), until=1.0, prefer_gpu=False)
    try:
        assert result.lift_record is result.driver.lift_record
        assert result.lift_record["meep_precision"] == "double"
        assert result.lift_record["engine_precision"] == "single"
        assert result.steps > 0
    finally:
        result.close()
    assert len(_notices(capsys.readouterr().err)) == 1
