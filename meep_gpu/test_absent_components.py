"""Field components MEEP allocated no array for: never stepped, read as +0, as in MEEP.

MEEP allocates a field component only when something requires it
(``fields::_require_component``, fields.cpp:552-572, over every component ``is_like``
the required one, fields.cpp:471-474). In a 2-D cell that is one polarization, so an
Ez source leaves Ex, Ey, Dx, Dy, Hz and Bz with ``f[c] == NULL``: ``update_eh``
skips them (``if (f[ec][cmp])``, update_eh.cpp:150) and ``fields_chunk::get_field``
reads them as 0.0 (monitor.cpp:160-164).

This engine steps every component. Those components only ever hold zeros, so the
difference is the sign of a zero, and the constitutive product makes it: on a
perfect-conductor cell (chi1inv = -1e-20, ``driver.PEC_CHI1INV_SENTINEL``)
``-1e-20 * (+0)`` is -0 where MEEP holds nothing and reads +0. MEASURED on
metal-cavity-ldos.py (bit_parity_0_10 lane REF): 535 Ex and 535 Ey words of the
metal shell read -0 against MEEP's +0 from step 1 (31,950 stored words over the run,
value-differing 0). The lift now records the components ``fields::have_component``
answers False for (:meth:`~.driver.FdtdDriver.set_absent_components`) and
``update_E`` leaves them untouched.

Host-only tests first (no MEEP), then the lift against MEEP, then the capsule.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from . import capsule
from .driver import PEC_EPSILON_SENTINEL, FdtdDriver
from .fields import FIELD_COMPONENTS

TE = ("Ex", "Ey", "Dx", "Dy", "Hz", "Bz")
TM = ("Ez", "Dz", "Hx", "Hy", "Bx", "By")
STEPS = 12
MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(MEEP_MISSING, reason="CPU MEEP is not installed in this environment")


def _pec_shell_tm(absent: bool, pml: bool = True) -> FdtdDriver:
    """A 2-D TM cell with a perfect-conductor shell around the source: MEEP's metal cavity, small."""
    driver = FdtdDriver(cell_size=(3.0, 3.0, 0.0), dimensions=2, resolution=10,
                        force_complex_fields=False)
    epsilon = np.ones(driver.shape, dtype=np.float64)
    epsilon[8:22, 8:22, :] = PEC_EPSILON_SENTINEL
    epsilon[10:20, 10:20, :] = 1.0
    driver.set_epsilon_components({name: epsilon for name in ("Ex", "Ey", "Ez")})
    if pml:
        driver.setup_pml({"x": (5, 5), "y": (5, 5)})
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.5, "center": (0.05, -0.05, 0.0), "size": (0.0, 0.0, 0.0)})
    if absent:
        driver.set_absent_components(TE)
    return driver


def _words(array):
    return np.ascontiguousarray(array).view(np.uint32)


def _negative_zeros(array) -> int:
    return int(np.count_nonzero(_words(array) == np.uint32(0x80000000)))


@pytest.mark.parametrize("pml", [True, False], ids=["stored_e", "derived_e"])
def test_an_absent_component_reads_plus_zero_where_its_chi1inv_is_negative(pml):
    """The absent E words stay +0 on the conductor; stepping it as before makes them -0 there.

    Two storage modes: with a PML ``update_E`` writes a stored E, without one
    ``Fields.get_E`` forms ``D * inv_eps`` on demand. The control is the same cell
    with nothing declared absent, which is how every component was stepped before.
    """
    ours, control = _pec_shell_tm(absent=True, pml=pml), _pec_shell_tm(absent=False, pml=pml)
    try:
        for _ in range(STEPS):
            ours.step()
            control.step()
        for name in ("Ex", "Ey"):
            mine = np.asarray(ours.get_field(name, cell_centered=False))
            theirs = np.asarray(control.get_field(name, cell_centered=False))
            assert not np.any(_words(mine)), (
                f"{name} is absent, so every word must be +0; {np.count_nonzero(_words(mine))} are not "
                f"({_negative_zeros(mine)} of them -0)")
            assert _negative_zeros(theirs) > 0, (
                f"the control steps {name} and must hold -0 on the conductor, or this test "
                f"cannot see the defect it pins")
            assert not np.any(theirs), f"the control's {name} must be zeros in value"
        for name in TM:
            assert np.array_equal(_words(ours.get_field(name, cell_centered=False)),
                                  _words(control.get_field(name, cell_centered=False))), (
                f"declaring the other polarization absent moved a word of {name}")
    finally:
        ours.close()
        control.close()


def test_a_source_on_an_absent_component_takes_its_polarization_back():
    """MEEP's require_component allocates the whole like class: a source puts TE back, TM stays absent."""
    driver = FdtdDriver(cell_size=(3.0, 3.0, 0.0), dimensions=2, resolution=10,
                        force_complex_fields=False)
    try:
        driver.set_absent_components(TE + TM)
        assert driver.fields.absent_components == FIELD_COMPONENTS, "kept in the twelve's order"
        driver.add_source({"component": "Hz", "source_type": "gaussian", "frequency": 1.0,
                           "fwidth": 0.5, "center": (0.0, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
        assert set(driver.fields.absent_components) == set(TM), driver.fields.absent_components
        driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                           "fwidth": 0.5, "center": (0.1, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
        assert driver.fields.absent_components == ()
        assert "absent_components" not in vars(driver.fields), "an empty set is the class default"
    finally:
        driver.close()


def test_a_sourced_component_and_a_late_declaration_are_refused_by_name():
    driver = _pec_shell_tm(absent=False)
    try:
        with pytest.raises(ValueError, match="carry a source"):
            driver.set_absent_components(("Ez",))
        with pytest.raises(ValueError, match="not field components"):
            driver.set_absent_components(("Er",))
        driver.step()
        with pytest.raises(RuntimeError, match="before stepping"):
            driver.set_absent_components(TE)
    finally:
        driver.close()


def test_a_capsule_carries_the_absent_set_and_steps_to_the_same_words(tmp_path):
    with capsule.recording_construction() as recorder:
        driver = _pec_shell_tm(absent=True)
    info = capsule.write_capsule(tmp_path / "absent.capsule", driver, recorder)
    loaded = capsule.read_capsule(info.path).build_driver()
    try:
        assert loaded.fields.absent_components == TE
        for _ in range(STEPS):
            driver.step()
            loaded.step()
        assert capsule.state_digest(driver) == capsule.state_digest(loaded)
    finally:
        driver.close()
        loaded.close()


# --- the lift, against MEEP ----------------------------------------------------------


def _metal_cavity(mp):
    """metal-cavity-ldos.py's cavity at resolution 20: an mp.metal shell, an air hole, an Ez dipole."""
    return mp.Simulation(
        cell_size=mp.Vector3(3, 3), resolution=20, boundary_layers=[mp.PML(0.5)],
        geometry=[mp.Block(mp.Vector3(1.2, 1.2, mp.inf), material=mp.metal),
                  mp.Block(mp.Vector3(1.0, 1.0, mp.inf), material=mp.air)],
        sources=[mp.Source(mp.GaussianSource(0.7, fwidth=0.2), component=mp.Ez,
                           center=mp.Vector3())])


@requires_meep
@skip_without_meep
def test_the_lift_records_what_meep_did_not_allocate_and_steps_it_as_meep_does():
    import meep as mp

    from .from_meep import lift_simulation

    mp.verbosity(0)
    sim = _metal_cavity(mp)
    driver = lift_simulation(sim, prefer_gpu=False)
    try:
        meep_absent = {name for name in TE + TM if not sim.fields.have_component(getattr(mp, name))}
        assert meep_absent == set(TE), meep_absent
        assert set(driver.fields.absent_components) == meep_absent
        for _ in range(STEPS):
            driver.step()
        sim.run(until=(STEPS - 0.5) * sim.fields.dt)
        assert int(sim.fields.t) == STEPS == driver.step_count
        for name in ("Ex", "Ey"):
            words = _words(np.asarray(driver.get_field(name, cell_centered=False)))
            assert not np.any(words), f"{name}: {np.count_nonzero(words)} words are not +0"
        assert not sim.fields.have_component(mp.Ex), "MEEP's stepping does not allocate it either"
    finally:
        driver.close()
