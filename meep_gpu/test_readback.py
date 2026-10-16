"""Behaviour pins for :mod:`readback`: MEEP's field readback, word for word.

What is pinned, and against what (NumPy only, no MEEP: the word compare against MEEP
itself is the bit-parity harness's, ``parity/meep_gpu`` of the 0.10 plan):

* the centred sum is MEEP's left-to-right association, ``((a + b) + c) + d`` and
  ``((a + b) + a) + b`` for one averaged axis (array_slice.cpp:413-415), in the
  direction order Z then R on a cylindrical grid;
* the neighbours are MEEP's chunk words: the package's ghost storage is never read
  (poisoned ghosts change no word of any reader);
* in a chunk where MEEP holds H as B's alias, a not-owned B word is the owner's H, on
  exactly the face plane of the low layer, and an integrated magnetic source in the
  chunk separates them (update_eh.cpp:162-172, :70-101);
* the point reader follows ``grid_volume::interpolate`` (the equal-weight reset, the
  1e-13 cut) and reflects a cylindrical leg below the axis through ``r -> -r`` with the
  ``m``-parity (fields.cpp:42, vec.cpp:1141-1153);
* the Bloch phase is MEEP's ``eikna`` formed in MEEP's arithmetic (two words measured off
  MEEP's own memory on this host family pin it);
* a ``center``/``size`` volume equals the whole-cell words it covers.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from . import readback
from .driver import FdtdDriver
from .fields import Fields
from .from_meep import GpuRunResult
from .grid import Grid


def _words(array):
    a = np.ascontiguousarray(np.asarray(array))
    if a.dtype == np.complex64:
        return a.view(np.float32).view(np.uint32)
    if a.dtype == np.float32:
        return a.view(np.uint32)
    raise TypeError(a.dtype)


def test_the_centred_sum_is_meeps_left_to_right_association():
    """``0.25 * (((a + b) + c) + d)`` in float32, the 0.25 in double: not two halvings."""
    grid = Grid(resolution=1.0, cell_size=(4.0, 4.0, 3.0))
    fields = Fields(grid=grid)
    dz = np.ones(grid.shape, dtype=np.float32)
    # Ez averages x then y (vec.cpp yee2cent_offsets: X before Y). At cell (1, 1, k) the
    # four words are Dz[1,1], Dz[2,1], Dz[1,2], Dz[2,2].
    dz[1, 1, :], dz[2, 1, :], dz[1, 2, :], dz[2, 2, :] = 1e8, 1.0, -1e8, 1.0
    fields.Dz[:] = dz
    centred = fields.to_cell_center("Ez")
    a, b, c, d = (np.float32(v) for v in (1e8, 1.0, -1e8, 1.0))
    expected = np.float32(np.float64(((a + b) + c) + d) * 0.25)
    assert expected == np.float32(0.25)
    assert centred[1, 1, 0] == expected
    pairwise = np.float32(0.5) * (np.float32(0.5) * (a + b) + np.float32(0.5) * (c + d))
    assert pairwise != expected, "the two associations must differ here or this pins nothing"


def test_one_averaged_axis_sums_a_b_a_b():
    """``offset2 = 0``: ``((a + b) + a) + b``, which is not ``2 * (a + b)`` in float32."""
    grid = Grid(resolution=1.0, cell_size=(3.0, 3.0, 4.0))
    fields = Fields(grid=grid)
    line = np.array([1e8, 5.0, 3.0, 7.0], dtype=np.float32)  # 1e8 + 5 rounds in float32
    fields.Bz[:] = np.broadcast_to(line.reshape(1, 1, 4), grid.shape)
    centred = fields.to_cell_center("Bz")  # Bz averages z only
    a, b = line[0], line[1]
    expected = np.float32(np.float64(((a + b) + a) + b) * 0.25)
    assert centred[0, 0, 0] == expected
    assert expected != np.float32(np.float64(np.float32(2.0) * (a + b)) * 0.25)


def _folded_pml_driver():
    driver = FdtdDriver(cell_size=(2.4, 2.0, 1.6), resolution=10, symmetry=("X", "Y"),
                        force_complex_fields=False)
    driver.setup_pml(3)
    driver.add_source({"source_type": "gaussian", "component": "Ez", "frequency": 0.8, "fwidth": 0.6,
                       "center": (0.31, 0.27, 0.0), "size": (0.0, 0.0, 0.0), "amplitude": 1.0})
    driver.run(num_steps=12)
    return driver


def _all_reads(driver):
    result = GpuRunResult(driver=driver, steps=driver.step_count, meep_time=driver.meep_time(), wall_time_s=0.0)
    names = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Bx", "By", "Bz")
    out = {}
    for name in names:
        out[("full", name)] = _words(result.get_array(name))
        out[("centres", name)] = _words(driver.get_field(name, cell_centered=True))
        out[("sub", name)] = _words(result.get_array(name, center=(-0.43, 0.21, 0.1), size=(0.9, 0.0, 0.7)))
        out[("point", name)] = np.array(complex(driver.get_field_point(name, (-0.37, 0.18, 0.05))))
    return out


def test_the_reader_never_reads_the_packages_ghost_storage():
    """Poison every mirror ghost row; no word of any reader moves.

    MEEP's not-owned words are rebuilt from owned words by its connection rules
    (step.cpp:171-221): the package's ghost rows carry its own fill arithmetic, which is
    not MEEP state, and a reader that took them would differ in exactly those words.
    """
    driver = _folded_pml_driver()
    before = _all_reads(driver)
    fields = driver.fields
    poisoned = 0
    for name in ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Bx", "By", "Bz"):
        array = getattr(fields, name)
        shifts = readback.IYEE_SHIFTS[name]
        for axis in (0, 1):
            if shifts[axis] == 0:  # the mirror ghost: stored row 0 of a shift-0 component
                index = [slice(None)] * 3
                index[axis] = 0
                array[tuple(index)] = np.float32(np.nan)
                poisoned += 1
    assert poisoned > 0
    after = _all_reads(driver)
    for key, words in before.items():
        assert np.array_equal(words, after[key]), f"{key} read the package's ghost storage"
    driver.close()


def _aliasing_driver(integrated_hz: bool):
    driver = FdtdDriver(cell_size=(2.0, 1.6, 0.0), resolution=10, dimensions=2, force_complex_fields=False)
    driver.setup_pml({"x": 3})
    if integrated_hz:
        driver.add_source({"source_type": "gaussian", "component": "Hz", "frequency": 0.8, "fwidth": 0.6,
                           "center": (0.05, 0.05, 0.0), "size": (0.0, 0.0, 0.0), "amplitude": 1.0,
                           "is_integrated": True})
    rng = np.random.default_rng(20261006)
    fields = driver.fields
    fields.Bx[:] = rng.standard_normal(fields.Bx.shape).astype(np.float32)
    fields.Hx[:] = rng.standard_normal(fields.Hx.shape).astype(np.float32)  # H != B everywhere
    return driver


@pytest.mark.parametrize("first_step", [False, True], ids=["later_step", "first_step"])
@pytest.mark.parametrize("integrated_hz", [False, True], ids=["aliasing", "integrated_source"])
def test_b_reads_the_owners_h_where_h_aliases_b(integrated_hz, first_step):
    """The interior chunk aliases H to B; its not-owned low-face Bx word is the x-PML owner's Hx.

    One-rank division with a 3-cell x layer: the low x-PML chunk owns doubled
    ``(lo, lo + 2*span]``, so the interior chunk's first centred point
    ``lo + 2*span + 1`` takes its low word ``lo + 2*span`` from the PML chunk. Bx
    averages x only: that one plane reads ``((H + B) + H) + B``, every other plane
    ``((B + B') + B) + B'``. An integrated magnetic source in the interior chunk gives
    it its own H (``have_f_minus_p``), and the plane reads B again.

    After the run's first step a chunk with H of its own still holds the zero it was
    allocated with at its not-owned B points (MEASURED on MEEP 1.33.0's own chunk words,
    Dcyl PML corner: +0 where the owner holds -0 at step 1, the owner's word from step 2):
    the low x-PML chunk's word at the periodic low corner, the high x-PML chunk's word at
    its face with the interior, and with the integrated source the interior chunk's face
    word too.
    """
    driver = _aliasing_driver(integrated_hz)
    if first_step:
        driver.step_count = 1  # the state after the first step (no step is taken here)
    fields = driver.fields
    layout = driver.chunk_layout
    rule = driver.h_alias_rule()
    centred = driver.get_field("Bx", cell_centered=True)
    b = np.asarray(fields.Bx)[:, :, 0]
    h = np.asarray(fields.Hx)[:, :, 0]
    origin = driver.grid.origin_doubled(0)
    low_box = min(layout.boxes, key=lambda box: box[0][0])
    span = (low_box[1][0] - origin) // 2
    expected = np.empty_like(centred[:, :, 0])
    zero = np.zeros_like(b[0])
    for i in range(b.shape[0]):
        low = b[i]
        up = b[(i + 1) % b.shape[0]]  # the default axes are periodic at k = 0: a copy
        if i == span:  # the interior chunk's first centre; its low word is the PML chunk's
            low = (zero if first_step else b[i]) if integrated_hz else h[i]
        if first_step and i in (0, b.shape[0] - span):  # the PML chunks' not-owned low words:
            low = zero  # the periodic low corner, and the high chunk's face with the interior
        expected[i] = (np.float64(((low + up) + low) + up) * 0.25).astype(np.float32)
    assert np.array_equal(_words(centred[:, :, 0]), _words(expected))
    interior = next(k for k, flags in enumerate(layout.pml_flags) if not flags[0])
    assert bool(rule.aliases["Bx"][interior]) is (not integrated_hz)
    assert bool(rule.separate[interior]) is integrated_hz
    driver.close()


def test_a_b_read_with_h_stored_needs_the_alias_rule():
    driver = _aliasing_driver(False)
    with pytest.raises(ValueError, match="needs the run's chunk layout"):
        driver.fields.to_cell_center("Bx")
    driver.fields.to_cell_center("Ex")  # E, D and H never alias
    driver.close()


def test_the_point_reader_takes_meeps_interpolation_weights():
    grid = Grid(resolution=10.0, cell_size=(2.0, 2.0, 0.0), dimensions=2)
    dx = grid.dx
    # Ez sits on the integer x and y lattice: a point on a cell centre is equidistant
    # from four samples, and the equal-weight reset makes them exactly 1/4.
    locs, weights = readback.interpolate(grid, "Ez", (3.5 * dx, -2.5 * dx, 0.0))
    assert weights[:4] == [0.25] * 4 and weights[4:] == [0.0] * 4
    # A lattice point plus 3e-15 is below the 1e-13 cut: one leg, weight 1.
    locs, weights = readback.interpolate(grid, "Ez", (3 * dx + 3e-15, -4 * dx + 3e-15, 0.0))
    assert weights[0] == 1.0 and weights[1:] == [0.0] * 7
    assert locs[0] == {0: 6, 1: -8}


def test_a_cylindrical_leg_below_the_axis_reads_its_r_image_with_the_m_parity():
    """``S = S + r_to_minus_r_symmetry(m)`` (fields.cpp:42): ph = -1 for odd m, flips R and P.

    Er at r = 0 straddles the axis: its legs are r = -dr/2 and +dr/2. MEEP owns no
    r < 0 point and reads the r -> -r image with ``phase_shift(Er) = -ph``: +1 at
    m = 1, so the read is the r = +dr/2 word itself; Ez's parity is ``ph``.
    """
    for m in (0, 1):
        grid = Grid(resolution=10.0, cell_size=(1.0, 0.0, 1.2), cylindrical=True, m=m)
        _, phase_r = readback._locate(grid, "Ex", {2: 2, 0: -1})
        _, phase_z = readback._locate(grid, "Ez", {2: 1, 0: -2})
        ph = -1.0 if m & 1 else 1.0
        assert phase_r[0] == -ph and phase_z[0] == ph, m


def test_the_cylindrical_centred_sum_takes_z_before_r():
    """LOOP_OVER_DIRECTIONS(Dcyl) is Z then R (vec.hpp:98-102): Ep's offset1 is z."""
    grid = Grid(resolution=1.0, cell_size=(4.0, 0.0, 4.0), cylindrical=True, m=1)
    fields = Fields(grid=grid, force_complex_fields=True)
    dp = np.ones(grid.shape, dtype=np.complex64)
    # Ep (package Ey) averages r and z. Words at cell (1, 0, 1): f, f_z, f_r, f_rz.
    dp[1, 0, 1], dp[1, 0, 2], dp[2, 0, 1], dp[2, 0, 2] = 1e8, -1e8, 1.0, 1.0
    fields.Dy[:] = dp
    centred = fields.to_cell_center("Ey")
    f, fz, fr, frz = (np.float32(v) for v in (1e8, -1e8, 1.0, 1.0))
    z_first = np.float32(np.float64(((f + fz) + fr) + frz) * 0.25)
    r_first = np.float32(np.float64(((f + fr) + fz) + frz) * 0.25)
    assert z_first != r_first
    assert centred[1, 0, 1].real == z_first


# A platform resource, declared: the words were read from MEEP on Darwin, whose libm
# rounds these transcendentals as pinned; another libm need not.
@pytest.mark.requires_resource("darwin_libm")
@pytest.mark.skipif(sys.platform != "darwin",
                    reason="[requires_resource][darwin_libm] the pinned words are MEEP's "
                           "on Darwin's libm")
def test_the_bloch_phase_is_meeps_eikna_word_for_word():
    """MEEP's ``eikna = exp(I * kk * ((2*pi/a) * n))``, read from MEEP 1.33.0's own memory.

    Two axes of the 408 measured (bit_parity_0_10 WP AP2): one where a separate ``sin``
    is a bit off MEEP's fused ``__sincos_stret``, one where the 0.9.2 association
    ``exp(2j*pi*k*L)`` is.
    """
    grid = Grid(resolution=16.0, cell_size=(7 / 16.0, 0.25, 0.0), dimensions=2,
                k_point=(-0.15597275326158105, 0.0, 0.0))
    phase = complex(grid.bloch_phase(0))
    assert (phase.real.hex(), phase.imag.hex()) == ("0x1.d1a807936d770p-1", "-0x1.a9b6d7174de1cp-2")
    k, n, res = 0.17, 12, 10.0
    grid = Grid(resolution=res, cell_size=(n / res, 0.4, 0.0), dimensions=2, k_point=(k, 0.0, 0.0))
    phase = complex(grid.bloch_phase(0))
    angle = k * ((2 * math.pi / res) * n)
    sin_a, cos_a = readback.sincos(angle)
    assert (phase.real, phase.imag) == (cos_a, sin_a)


def test_a_volume_equals_the_whole_cell_words_it_covers():
    """A ``center``/``size`` box with extent on every axis takes weight 1: MEEP's words, cut out."""
    driver = _folded_pml_driver()
    result = GpuRunResult(driver=driver, steps=driver.step_count, meep_time=driver.meep_time(), wall_time_s=0.0)
    grid = driver.grid
    for name in ("Ez", "Hx", "By"):
        whole = result.get_array(name)
        lattice = readback.CentredLattice(driver.fields, name, driver.h_alias_rule())
        box = result.get_array(name, center=(0.23, -0.31, 0.12), size=(0.84, 0.66, 0.52))
        picks = []
        for axis in range(3):
            lo = (0.23, -0.31, 0.12)[axis] - 0.5 * (0.84, 0.66, 0.52)[axis]
            hi = (0.23, -0.31, 0.12)[axis] + 0.5 * (0.84, 0.66, 0.52)[axis]
            a = grid.resolution
            i_s = 1 + 2 * int(math.floor(lo * a - 0.5))
            i_e = 1 + 2 * int(math.ceil(hi * a - 0.5))
            points = lattice.ivec[axis]
            picks.append(np.nonzero((points >= i_s) & (points <= i_e))[0])
        assert np.array_equal(_words(box), _words(whole[np.ix_(*picks)])), name
    with pytest.raises(ValueError, match="both center and size"):
        result.get_array("Ez", center=(0.0, 0.0, 0.0))
    driver.close()
