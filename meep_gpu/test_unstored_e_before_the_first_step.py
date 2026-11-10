"""An E the run does not store, read before the run's first step: D's words, as MEEP's.

MEEP's E is D's storage until its first ``update_eh(E_stuff)`` allocates E as a copy of D
(update_eh.cpp:165-170), so before the first step a read of E returns D's words. With no
PML and no susceptibility this engine stores no E at all: ``Fields.get_E`` forms
``D * inv_eps`` on every read. On a fresh run D is +0 everywhere, and on a perfect-conductor
cell (chi1inv = -1e-20, ``driver.PEC_CHI1INV_SENTINEL``) ``-1e-20 * (+0)`` is -0 where MEEP
reads +0. Measured on ``test_visualization``'s ``test_plot2D_pec`` (a metal block, no PML):
1,640 stored ``Ez`` words of the block at step 0, all sign of zero, and 1,560 words of
MEEP's ``get_array``; 0 from step 1, where MEEP's first update writes the same -0.

``Fields.e_reads_d`` carries the rule: the driver sets it at construction and at ``reset``
and clears it when its first step begins and when a field is seeded. Only the read surfaces
consult it; the sub-steps read E through ``get_E``, unchanged.

Host-only tests first (no MEEP), then the lift against MEEP.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from . import capsule
from .dft import _raw_factors
from .dispersion import LORENTZIAN, Susceptibility
from .driver import PEC_EPSILON_SENTINEL, FdtdDriver
from .fields import FIELD_COMPONENTS

STEPS = 6
MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(MEEP_MISSING, reason="CPU MEEP is not installed in this environment")


def _metal_block(pml: bool = False, susceptible: bool = False) -> FdtdDriver:
    """A 2-D TM cell with a perfect-conductor block beside an Ez source: the visualization test's shape."""
    driver = FdtdDriver(cell_size=(3.0, 3.0, 0.0), dimensions=2, resolution=10,
                        force_complex_fields=False)
    epsilon = np.ones(driver.shape, dtype=np.float64)
    epsilon[18:26, 8:22, :] = PEC_EPSILON_SENTINEL
    epsilon[4:10, 8:22, :] = 12.0
    driver.set_epsilon_components({name: epsilon for name in ("Ex", "Ey", "Ez")})
    if pml:
        driver.setup_pml({"x": (5, 5), "y": (5, 5)})
    if susceptible:
        driver.add_susceptibility(Susceptibility(1.1, 0.1, LORENTZIAN),
                                  np.full(driver.shape, 0.5, dtype=np.float32))
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.5, "center": (-0.05, 0.05, 0.0), "size": (0.0, 0.0, 0.0)})
    return driver


def _words(array):
    return np.ascontiguousarray(array).view(np.uint32)


def _negative_zeros(array) -> int:
    return int(np.count_nonzero(_words(array) == np.uint32(0x80000000)))


def test_before_the_first_step_an_unstored_e_reads_ds_words():
    """Every read surface serves D's words at step 0; the sub-steps' own read still forms -0 there."""
    driver = _metal_block()
    try:
        fields = driver.fields
        assert not fields.stores_E and fields.e_reads_d
        displacement = np.asarray(fields.Dz)
        raw = np.asarray(driver.get_field("Ez", cell_centered=False))
        assert np.array_equal(_words(raw), _words(displacement)) and not np.any(_words(raw)), (
            f"step 0: {np.count_nonzero(_words(raw))} raw Ez words are not D's +0 "
            f"({_negative_zeros(raw)} of them -0)")
        centred = np.asarray(driver.get_field("Ez"))
        assert not np.any(_words(centred)), f"step 0: {_negative_zeros(centred)} centred Ez words are -0"
        meep_view = np.asarray(fields.to_meep_array("Ez", driver.h_alias_rule()))
        assert not np.any(_words(meep_view)), f"step 0: {_negative_zeros(meep_view)} get_array words are -0"
        for factors in (fields.component_factors("Ez"), _raw_factors(fields, "Ez")):
            assert len(factors) == 1 and factors[0] is fields.Dz, "E is D's storage before the first step"
        assert not np.any(_words(fields.sliced_component("Ez", (slice(18, 26), slice(8, 22), 0))))
        # The control: the sub-steps' read is the product, -0 on the conductor.
        assert _negative_zeros(fields.get_E("Ez")) == 8 * 14, (
            "the constitutive product must hold -0 on every conductor cell, or this test cannot see "
            f"the defect it pins ({_negative_zeros(fields.get_E('Ez'))} -0 words)")
        driver.step()
        assert not fields.e_reads_d
        for _ in range(STEPS):
            assert np.array_equal(_words(driver.get_field("Ez", cell_centered=False)),
                                  _words(fields.get_E("Ez")))
            driver.step()
        assert _negative_zeros(driver.get_field("Ez", cell_centered=False)) > 0, (
            "from the first step E is D * inv_eps, -0 on the conductor's untouched cells")
    finally:
        driver.close()


def test_the_step_reads_the_product_even_before_its_first_constitutive_update():
    """A D written in place at step 0 is curled as ``D * inv_eps``, and the step's monitors read the product.

    ``ours`` writes D in place, so its flag stays set until its first step; the control seeds
    the same words through ``set_field``, which clears the flag. The two runs must agree word
    for word in all twelve components and in a DFT monitor's ``Ez`` accumulator at every
    step. The seed is nonzero where ``inv_eps != 1`` (the permittivity-12 block and the
    conductor), so a first accumulation that read E with the flag still set would take D's
    words there, not the product's: the monitor comparison pins that the flag is cleared
    before the first step's monitors read E.
    """
    ours, control = _metal_block(), _metal_block()
    try:
        monitors = [driver.add_dft_monitor(frequencies=[0.9, 1.0], components=("Ez",),
                                           center=(0.0, 0.0, 0.0), size=(3.0, 3.0, 0.0))
                    for driver in (ours, control)]
        seed = np.random.default_rng(7).standard_normal(ours.shape).astype(np.float32)
        ours.fields.Dz[...] = seed
        control.set_field("Dz", seed)
        assert ours.fields.e_reads_d and not control.fields.e_reads_d
        for step in range(1, STEPS + 1):
            ours.step()
            control.step()
            for name in FIELD_COMPONENTS:
                assert np.array_equal(_words(ours.get_field(name, cell_centered=False)),
                                      _words(control.get_field(name, cell_centered=False))), (
                    f"step {step}: {name} moved with the read rule")
            spectra = [np.asarray(monitor.get_dft_spectrum("Ez")) for monitor in monitors]
            differing = int(np.count_nonzero(_words(spectra[0]) != _words(spectra[1])))
            assert differing == 0, (
                f"step {step}: {differing} of {_words(spectra[1]).size} DFT Ez words moved with the read rule")
    finally:
        ours.close()
        control.close()


def test_a_seeded_d_is_read_through_inv_eps():
    driver = _metal_block()
    try:
        seed = np.random.default_rng(3).standard_normal(driver.shape).astype(np.float32)
        driver.set_field("Dz", seed)
        assert not driver.fields.e_reads_d
        expected = (driver.fields.inverse_epsilon_for("Ez") * driver.fields.Dz).astype(np.float32)
        assert np.array_equal(_words(driver.get_field("Ez", cell_centered=False)), _words(expected))
    finally:
        driver.close()


def test_reset_reads_ds_words_again():
    """MEEP's zero_fields leaves E +0 (fields.cpp:625-646); the zeroed D is what is served."""
    driver = _metal_block()
    try:
        for _ in range(STEPS):
            driver.step()
        driver.reset()
        assert driver.fields.e_reads_d
        raw = np.asarray(driver.get_field("Ez", cell_centered=False))
        assert not np.any(_words(raw)), f"after reset {_negative_zeros(raw)} Ez words are -0"
    finally:
        driver.close()


@pytest.mark.parametrize("storage", ["pml", "susceptibility"])
def test_a_stored_e_is_served_as_stored(storage):
    """A run that stores E allocates it as zeros, +0 before the first step, and never reads the flag."""
    driver = _metal_block(pml=storage == "pml", susceptible=storage == "susceptibility")
    try:
        assert driver.fields.stores_E and driver.fields.e_reads_d
        # A D written in place before the first step is not what a stored E reads.
        driver.fields.Dz[...] = np.float32(1.0)
        assert not np.any(_words(driver.get_field("Ez", cell_centered=False)))
        for _ in range(STEPS):
            assert np.array_equal(_words(driver.get_field("Ez", cell_centered=False)),
                                  _words(driver.fields.Ez))
            driver.step()
    finally:
        driver.close()


def test_an_integrated_source_added_before_the_first_step_starts_the_stored_e_from_d():
    """MEEP's allocation copies D (update_eh.cpp:168-169): +0 at step 0, not D * inv_eps's -0."""
    driver = _metal_block()
    try:
        driver.add_source({"component": "Ez", "frequency": 0.8, "is_integrated": True,
                           "center": (0.05, -0.05, 0.0), "size": (0.0, 0.0, 0.0)})
        assert driver.fields.stores_E
        stored = np.asarray(driver.fields.Ez)
        assert not np.any(_words(stored)), f"{_negative_zeros(stored)} stored Ez words start at -0"
    finally:
        driver.close()


def test_a_capsule_built_driver_reads_ds_words_at_step_zero(tmp_path):
    with capsule.recording_construction() as recorder:
        driver = _metal_block()
    info = capsule.write_capsule(tmp_path / "metal_block.capsule", driver, recorder)
    loaded = capsule.read_capsule(info.path).build_driver()
    try:
        assert loaded.fields.e_reads_d
        assert not np.any(_words(loaded.get_field("Ez", cell_centered=False)))
        for _ in range(STEPS):
            driver.step()
            loaded.step()
        assert capsule.state_digest(driver) == capsule.state_digest(loaded)
    finally:
        driver.close()
        loaded.close()


def test_a_capsule_of_a_stepped_driver_reads_the_product(tmp_path):
    """The flag is not carried (old capsules hold no such attribute); the loader derives it."""
    with capsule.recording_construction() as recorder:
        driver = _metal_block()
    try:
        for _ in range(STEPS):
            driver.step()
        info = capsule.write_capsule(tmp_path / "stepped.capsule", driver, recorder)
        loaded = capsule.read_capsule(info.path).build_driver()
        try:
            assert loaded.step_count == STEPS and not loaded.fields.e_reads_d
            assert np.array_equal(_words(loaded.get_field("Ez", cell_centered=False)),
                                  _words(driver.get_field("Ez", cell_centered=False)))
            assert _negative_zeros(loaded.get_field("Ez", cell_centered=False)) > 0
        finally:
            loaded.close()
    finally:
        driver.close()


# --- the lift, against MEEP ----------------------------------------------------------


def _pec_no_pml(mp):
    """test_visualization.py's test_plot2D_pec simulation: an eps-12 block, an mp.metal block, no PML."""
    return mp.Simulation(
        cell_size=mp.Vector3(6, 6), resolution=20, boundary_layers=[],
        geometry=[mp.Block(size=mp.Vector3(2, 2), center=mp.Vector3(-1.2), material=mp.Medium(epsilon=12)),
                  mp.Block(size=mp.Vector3(2, 2), center=mp.Vector3(1.2), material=mp.metal)],
        sources=[mp.Source(src=mp.ContinuousSource(frequency=0.15), component=mp.Ez,
                           center=mp.Vector3(0, -2))])


@requires_meep
@skip_without_meep
def test_the_lift_reads_meeps_step_zero_words_of_a_metal_block_with_no_pml():
    """MEEP's get_array of Ez against the lifted driver's, word for word, at steps 0 to 2."""
    import meep as mp

    from .from_meep import lift_simulation

    mp.verbosity(0)
    sim = _pec_no_pml(mp)
    driver = lift_simulation(sim, prefer_gpu=False)
    try:
        assert not driver.fields.stores_E and driver.fields.e_reads_d
        for step in range(3):
            while driver.step_count < step:
                driver.step()
            if step:
                sim.run(until=(step - int(sim.fields.t) - 0.5) * sim.fields.dt)
            assert int(sim.fields.t) == step == driver.step_count
            theirs = np.asarray(sim.get_array(component=mp.Ez))
            ours = np.asarray(driver.fields.to_meep_array("Ez", driver.h_alias_rule()))
            assert ours.shape == theirs.shape
            differing = int(np.count_nonzero(_words(ours) != _words(theirs)))
            assert differing == 0, f"step {step}: {differing} of {theirs.size} Ez get_array words differ"
    finally:
        driver.close()
