"""A dispatched update_E and the components MEEP never allocated: the seam's clear.

MEEP allocates no array for a component nothing requires (a 2-D run's other
polarization; ``fields::_require_component``, fields.cpp:552-572), never steps it
(``if (f[ec][cmp])``, update_eh.cpp:150) and reads it as 0.0 (monitor.cpp:160-164).
The array path leaves those components at +0 (``Fields.set_absent_components``). Every
kernel table's ``update_E`` steps all three components, so on a perfect-conductor cell
(``chi1inv = -1e-20``) it writes ``-1e-20 * (+0) = -0`` into E, or into ``f_w`` inside
the absorber, where the array path holds +0. On a folded grid the absent component's D
holds -0 on the planes its fills write with an odd parity (``fill_symmetry_bc_D``,
``fill_folded_far_ghosts_D``), and the stepped product is -0 there too, with no conductor
anywhere. ``fastpath._finish`` therefore builds a clear of exactly those words
(``fastpath.AbsentComponentClear``: the sign-bit cells and every fill plane) and
``FastPathPlan.dispatch`` applies it after every ``update_E`` it answers.

These tests need no GPU. They install a plan whose ``update_E`` steps every component
-- the arithmetic each table is certified against, from before the array path learnt
the absent set -- at the driver's seam, and compare every stored word against the
array path, with a control that withholds the clear and must show the -0 words.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from . import dft, fastpath, host_writes, stepping
from .driver import PEC_EPSILON_SENTINEL, FdtdDriver

TE = ("Ex", "Ey", "Dx", "Dy", "Hz", "Bz")
TM = ("Ez", "Dz", "Hx", "Hy", "Bx", "By")
STEPS = 12


def _driver(*, source: str = "Ez", pml: bool = True, complex_fields: bool = False,
            wall_into_pml: bool = False, conductor: bool = True) -> FdtdDriver:
    """A 2-D cell with a perfect-conductor shell around a dipole; the other polarization absent.

    ``wall_into_pml`` adds a conductor strip that crosses the absorber, where the -0
    of the constitutive product lands in ``f_w`` rather than in E.
    """
    driver = FdtdDriver(cell_size=(3.0, 3.0, 0.0), dimensions=2, resolution=10,
                        force_complex_fields=complex_fields)
    epsilon = np.ones(driver.shape, dtype=np.float64)
    if conductor:
        epsilon[8:22, 8:22, :] = PEC_EPSILON_SENTINEL
        epsilon[10:20, 10:20, :] = 1.0
        if wall_into_pml:
            epsilon[:, 1:3, :] = PEC_EPSILON_SENTINEL
    else:
        epsilon[8:22, 8:22, :] = 4.0
    driver.set_epsilon_components({name: epsilon for name in ("Ex", "Ey", "Ez")})
    if pml:
        driver.setup_pml({"x": (5, 5), "y": (5, 5)})
    driver.add_source({"component": source, "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.5, "center": (0.05, -0.05, 0.0), "size": (0.0, 0.0, 0.0)})
    driver.set_absent_components(TE if source in TM else TM)
    return driver


def _folded_driver(*, complex_fields: bool = False) -> FdtdDriver:
    """A 2-D cell folded on a periodic Y mirror, no conductor, the other polarization absent.

    The far ghost slot of the folded periodic Y axis is filled with Dy's parity about an
    even plane, which is -1 (Dy is normal to the plane), so Dy holds -0 on that plane from
    the first step; the absorber on X puts the stepped product into ``f_w_Ey`` there.
    Every inverse permittivity is positive, so the sign-bit set is empty and only the
    fill planes need the clear.
    """
    from .grid import Mirror  # noqa: PLC0415

    driver = FdtdDriver(cell_size=(3.0, 3.0, 0.0), dimensions=2, resolution=10,
                        force_complex_fields=complex_fields,
                        boundaries=("periodic", "periodic", "periodic"),
                        symmetry=(Mirror("Y", 1),))
    epsilon = np.ones(driver.shape, dtype=np.float64)
    epsilon[8:22, :, :] = 4.0
    driver.set_epsilon_components({name: epsilon for name in ("Ex", "Ey", "Ez")})
    driver.setup_pml({"x": (5, 5)})
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.5, "center": (0.05, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
    driver.set_absent_components(TE)
    return driver


class _EveryComponentUpdateE:
    """An ``update_E`` plan that steps every component, as each kernel table does."""

    def __init__(self, driver: FdtdDriver) -> None:
        self.driver = driver
        self.launches = 0

    def run(self) -> None:
        fields = self.driver.fields
        absent = fields.absent_components
        fields.set_absent_components(())
        try:
            stepping.update_E(fields, self.driver.pml)
        finally:
            fields.set_absent_components(absent)
        self.launches += 1


def _install(driver: FdtdDriver, monkeypatch: pytest.MonkeyPatch, *, withhold: bool = False):
    """Freeze a plan through ``fastpath._finish`` whose one slot is the stub, at the seam."""
    monkeypatch.setattr(fastpath, "_emit", lambda record: None)
    stub = _EveryComponentUpdateE(driver)
    step_plan = SimpleNamespace(plans={"update_E": stub}, selected={"update_E": "stub"},
                                reasons={}, backends={})
    record = fastpath._base_record()  # noqa: SLF001
    plan = fastpath._finish(record=record, step_plan=step_plan,  # noqa: SLF001
                            dispatchable={"update_E"}, dropped_null={}, unwarmed={},
                            selected={"update_E": "stub"}, table="triton",
                            fields=driver.fields)
    assert plan is not None, record.get("refused_because")
    if withhold:
        plan = dataclasses.replace(plan, absent_clears=())
    driver._fast_path = plan  # noqa: SLF001 - the seam's own slot, as the probes install it
    driver._fast_path_stale = False  # noqa: SLF001
    return plan, stub, record


def _state(driver: FdtdDriver) -> dict:
    """Every array on Fields and the PML, as uint32 words."""
    out = {}
    for owner, prefix in ((driver.fields, "fields"), (driver.pml, "pml")):
        if owner is None:
            continue
        for name, value in sorted(vars(owner).items()):
            if isinstance(value, np.ndarray) and value.dtype.kind in "fc":
                out[f"{prefix}.{name}"] = np.ascontiguousarray(value).reshape(-1).view(np.uint32)
    return out


def _differing(left: dict, right: dict) -> dict:
    assert set(left) == set(right)
    return {name: int(np.count_nonzero(left[name] != right[name]))
            for name in left if np.count_nonzero(left[name] != right[name])}


def _run_pair(monkeypatch, *, withhold: bool = False, build=None, **shape):
    build = _driver if build is None else build
    ours, reference = build(**shape), build(**shape)
    plan, stub, record = _install(ours, monkeypatch, withhold=withhold)
    try:
        for _ in range(STEPS):
            ours.step()
            reference.step()
        return _differing(_state(ours), _state(reference)), plan, stub, record
    finally:
        ours.close()
        reference.close()


@pytest.mark.parametrize("shape", [
    {}, {"wall_into_pml": True}, {"source": "Hz"}, {"complex_fields": True},
], ids=["tm", "tm_wall_into_pml", "te", "tm_complex"])
def test_a_dispatched_update_e_leaves_every_word_the_array_paths(monkeypatch, shape):
    """With the clear: 0 differing words. Withheld: -0 on the absent E or f_w words, nowhere else."""
    differing, plan, stub, record = _run_pair(monkeypatch, **shape)
    assert stub.launches == STEPS, "the stub must serve every update_E, or this compares nothing"
    assert plan.report()["absent_components"]["clears_run"] == STEPS
    assert differing == {}, f"words differing from the array path with the clear: {differing}"
    cleared = record["absent_components"]["cleared"]
    assert any(entry["cells"] for entry in cleared.values()), cleared

    control, _, control_stub, _ = _run_pair(monkeypatch, withhold=True, **shape)
    assert control_stub.launches == STEPS
    absent_e = {"fields." + name for entry in cleared.values() for name in entry["arrays"]}
    assert control, "withheld, the stepped absent component must hold -0 on the conductor"
    assert set(control) <= absent_e, (
        f"withheld, only the absent components' E and f_w words may differ: {sorted(control)}")
    if shape.get("wall_into_pml"):
        assert any(name.startswith("fields.f_w_") for name in control), (
            "a conductor inside the absorber puts the -0 in f_w; the control must show it")


@pytest.mark.parametrize("complex_fields", [False, True], ids=["real", "complex"])
def test_a_folded_run_clears_the_fill_planes_with_no_conductor(monkeypatch, complex_fields):
    """No sign bit anywhere, yet the stepped absent component reads -0 on the fill planes.

    With the clear: 0 differing words. Withheld: the -0 words, on the absent components'
    E and f_w only, every one of them on a plane the D fills write.
    """
    driver = _folded_driver(complex_fields=complex_fields)
    try:
        clears, block = fastpath._absent_component_clears(driver.fields)  # noqa: SLF001
        planes = {c.component: fastpath._absent_fill_planes(  # noqa: SLF001
            driver.fields, c.component, tuple(c.arrays[0].shape)) for c in clears}
    finally:
        driver.close()
    assert clears, block
    for entry in block["cleared"].values():
        assert entry["sign_bit_cells"] == 0, block
    assert any(entry["fill_plane_cells"] for entry in block["cleared"].values()), block

    differing, plan, stub, _ = _run_pair(monkeypatch, build=_folded_driver,
                                         complex_fields=complex_fields)
    assert stub.launches == STEPS
    assert plan.report()["absent_components"]["clears_run"] == STEPS
    assert differing == {}, f"words differing from the array path with the clear: {differing}"

    ours, reference = _folded_driver(complex_fields=complex_fields), \
        _folded_driver(complex_fields=complex_fields)
    try:
        _plan, control_stub, _record = _install(ours, monkeypatch, withhold=True)
        for _ in range(STEPS):
            ours.step()
            reference.step()
        control = _differing(_state(ours), _state(reference))
        assert control_stub.launches == STEPS
        assert control, "withheld, the stepped absent component must hold -0 on a fill plane"
        absent_e = {"fields." + name for c in clears for name in c.names}
        assert set(control) <= absent_e, sorted(control)
        for name in control:
            short = name.split(".", 1)[1]
            component = short[len("f_w_"):] if short.startswith("f_w_") else short
            ours_words = _state(ours)[name]
            theirs = _state(reference)[name]
            where = np.flatnonzero(ours_words != theirs)
            # A complex array's flat words are (re, im) pairs: two words per cell.
            cells = where // (2 if complex_fields else 1)
            assert np.isin(cells, planes[component]).all(), (
                f"{name}: a differing word off the fill planes: "
                f"{sorted(set(cells.tolist()) - set(planes[component].tolist()))[:8]}")
    finally:
        ours.close()
        reference.close()


def test_an_ordinary_run_is_given_no_clear(monkeypatch):
    """A positive permittivity everywhere: nothing to clear, so nothing runs per step."""
    driver = _driver(conductor=False)
    try:
        clears, block = fastpath._absent_component_clears(driver.fields)  # noqa: SLF001
        assert clears == ()
        assert set(block["absent"]) == set(TE)
        assert all(entry["cells"] == 0 for entry in block["cleared"].values()), block
        differing, plan, stub, _ = _run_pair(monkeypatch, conductor=False)
        assert differing == {} and plan.absent_clears == () and stub.launches == STEPS
        assert plan.report()["absent_components"]["clears_run"] == 0
    finally:
        driver.close()


def test_derived_e_and_a_full_set_need_no_clear():
    """No stored E (no absorber): update_E writes nothing. No absent component: nothing to clear."""
    derived = _driver(pml=False)
    every = FdtdDriver(cell_size=(3.0, 3.0, 0.0), dimensions=2, resolution=10)
    try:
        clears, block = fastpath._absent_component_clears(derived.fields)  # noqa: SLF001
        assert clears == () and "not_stored" in block, block
        clears, block = fastpath._absent_component_clears(every.fields)  # noqa: SLF001
        assert clears == () and block["absent"] == [] and block["cleared"] == {}
    finally:
        derived.close()
        every.close()


def test_a_clear_that_cannot_be_written_refuses_the_plan_by_name(monkeypatch):
    driver = _driver()
    try:
        real = driver.fields.inverse_epsilon_for
        monkeypatch.setattr(driver.fields, "inverse_epsilon_for",
                            lambda name: np.full((2, 2, 2), -1.0, dtype=np.float32)
                            if name in ("Ex", "Ey") else real(name))
        with pytest.raises(ValueError, match="does not broadcast"):
            fastpath._absent_component_clears(driver.fields)  # noqa: SLF001
        monkeypatch.setattr(fastpath, "_emit", lambda record: None)
        step_plan = SimpleNamespace(plans={"update_E": _EveryComponentUpdateE(driver)},
                                    selected={"update_E": "stub"}, reasons={}, backends={})
        record = fastpath._base_record()  # noqa: SLF001
        plan = fastpath._finish(record=record, step_plan=step_plan,  # noqa: SLF001
                                dispatchable={"update_E"}, dropped_null={}, unwarmed={},
                                selected={"update_E": "stub"}, table="triton",
                                fields=driver.fields)
        assert plan is None
        assert "absent components' words cannot be cleared" in record["refused_because"]
    finally:
        driver.close()


def test_a_plan_without_update_e_carries_no_clear(monkeypatch):
    driver = _driver()
    try:
        monkeypatch.setattr(fastpath, "_emit", lambda record: None)

        class _Curl:
            def run(self) -> None:
                stepping.step_D(driver.fields, driver.pml)

        step_plan = SimpleNamespace(plans={"step_D": _Curl()}, selected={"step_D": "stub"},
                                    reasons={}, backends={})
        record = fastpath._base_record()  # noqa: SLF001
        plan = fastpath._finish(record=record, step_plan=step_plan,  # noqa: SLF001
                                dispatchable={"step_D"}, dropped_null={}, unwarmed={},
                                selected={"step_D": "stub"}, table="triton",
                                fields=driver.fields)
        assert plan is not None and plan.absent_clears == ()
        assert "absent_components" not in record
    finally:
        driver.close()


def test_the_held_reader_door_serves_an_absent_component_as_zeros():
    """``dft._raw_factors``: the factors a monitor reads under a held residency, raw.

    Without an absorber E is derived, and the raw factors are ``(D, inv_eps)``; for an
    absent component their product is -0 on the conductor where MEEP reads 0.0. The
    door serves the zeros ``Fields.component_factors`` serves.
    """
    driver = _driver(pml=False)
    try:
        for _ in range(STEPS):
            driver.step()
        for name in ("Ex", "Ey"):
            factors = dft._raw_factors(driver.fields, name)  # noqa: SLF001
            block = factors[0]
            for factor in factors[1:]:
                block = block * factor
            words = np.ascontiguousarray(block).view(np.uint32)
            assert not np.any(words), f"{name}: {np.count_nonzero(words)} words are not +0"
            reference = driver.fields.component_factors(name)
            assert len(factors) == len(reference) == 1
        product = driver.fields.Dx * driver.fields.inverse_epsilon_for("Ex")
        assert np.count_nonzero(np.ascontiguousarray(product).view(np.uint32)), (
            "the control: D * inv_eps must show -0 on the conductor, or this test sees nothing")
    finally:
        driver.close()


def test_the_cell_door_writes_in_place_with_nothing_held():
    array = np.full((3, 4, 1), -0.0, dtype=np.float32)
    linear = np.array([0, 5, 11])
    assert not host_writes.holding()
    host_writes.zero_cells([(array, linear)])
    words = array.reshape(-1).view(np.uint32)
    assert not np.any(words[linear]), words
    assert np.count_nonzero(words) == array.size - linear.size
