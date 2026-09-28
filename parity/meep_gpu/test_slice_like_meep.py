"""``slice_like_meep`` must cut a sub-volume the way MEEP's own ``get_array`` cuts it.

The assertion driver serves ``sim.get_array(component, vol)`` out of a whole-cell
array the engine produced. MEEP does not simply index there: an axis whose REQUESTED
extent is zero keeps the interpolation weights of the two cells that bracket the
requested coordinate and is then summed away (loop_in_chunks.cpp
``compute_boundary_weights``, the ``where.in_direction_min(d) ==
where.in_direction_max(d)`` branch, feeding ``IVEC_LOOP_WEIGHT(s0i, s1i, e0i, e1i,
1.0)`` in array_slice.cpp ``get_array_slice_chunkloop`` and then array_slice.cpp
``collapse_array``). Reproducing that is what lets a test whose oracle is a flat slice
be driven at all.

MEEP is the oracle here, not a transcription of it: every case fetches the whole cell
through MEEP, cuts it with :func:`slice_like_meep`, and compares against MEEP's own
answer for the same request. The weights are therefore measured, never asserted from
the formula that produced them.

Two cases exist to keep the measurement from being vacuous:

* An **asymmetric** offset (0.3 and 0.7 of a cell). A request centred exactly between
  two cells weights them 0.5/0.5, where swapping the two weights is invisible — the
  cavity-arrayslice tests that motivated this work are all of that kind. The
  asymmetric cases are the ones that can tell ``[1 - f, f]`` from ``[f, 1 - f]``.
* ``get_field_point`` stays STRICT. MEEP answers that one through
  ``fields::get_field``, which interpolates the component's own Yee lattice rather
  than the centred array this path holds, so an off-grid point is still refused
  instead of being served a centred-grid interpolation that would differ.
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

# MEEP's single-precision build carries the slice through float32; the exact-index
# path is bit-identical and the weighted path lands a couple of ULP away.
_BAR = 1e-6


def _driver_module():
    spec = importlib.util.spec_from_file_location("drive_meep_test_assertions", _DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stepped_sim():
    """One small 2D run, reused by every case; only the READS are under test."""
    import meep as mp

    sim = mp.Simulation(
        cell_size=mp.Vector3(4, 3, 0),
        resolution=10,
        sources=[mp.Source(mp.GaussianSource(0.3, fwidth=0.2), mp.Hz, mp.Vector3())],
        boundary_layers=[mp.PML(0.5)],
    )
    sim.run(until=3)
    return mp, sim


def _relative(mine, theirs) -> float:
    mine = np.asarray(mine)
    theirs = np.asarray(theirs)
    assert mine.shape == theirs.shape, f"shape {mine.shape} against MEEP's {theirs.shape}"
    denominator = float(np.linalg.norm(theirs))
    assert denominator > 0.0, "MEEP's own slice is all zeros; the comparison would be vacuous"
    return float(np.linalg.norm(mine - theirs) / denominator)


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.3, 0.7])
def test_flat_axis_slice_matches_meeps_own_get_array(stepped_sim, fraction):
    """A zero-thickness y at a fraction of a cell off the grid, against MEEP."""
    mp, sim = stepped_sim
    driver = _driver_module()
    tics = np.asarray(sim.get_array_metadata()[1], dtype=float)
    dy = float(tics[1] - tics[0])
    centre = float(tics[7] + fraction * dy)

    vol = mp.Volume(center=mp.Vector3(0, centre, 0), size=mp.Vector3(2.0, 0, 0))
    whole = np.asarray(sim.get_array(mp.Hz))
    mine = driver.slice_like_meep(mp, sim, whole, vol, None, None)
    theirs = np.asarray(sim.get_array(mp.Hz, vol))

    error = _relative(mine, theirs)
    assert error < _BAR, f"flat-axis slice at fraction {fraction}: {error:.3e} against MEEP"


@requires_meep
@skip_without_meep
def test_the_two_bracketing_weights_are_not_interchangeable(stepped_sim):
    """The positive control for the case above: a swapped weight must be visible.

    Without this, a battery made only of the symmetric (0.5/0.5) requests MEEP's own
    tests happen to use would pass with the weights reversed.
    """
    mp, sim = stepped_sim
    tics = np.asarray(sim.get_array_metadata()[1], dtype=float)
    dy = float(tics[1] - tics[0])
    whole = np.asarray(sim.get_array(mp.Hz))
    xt = np.asarray(sim.get_array_metadata()[0], dtype=float)

    worst = 0.0
    for fraction in (0.3, 0.7):
        centre = float(tics[7] + fraction * dy)
        vol = mp.Volume(center=mp.Vector3(0, centre, 0), size=mp.Vector3(2.0, 0, 0))
        theirs = np.asarray(sim.get_array(mp.Hz, vol))
        part_x = np.asarray(sim.get_array_metadata(vol=vol)[0], dtype=float)
        rows = np.asarray([int(np.argmin(np.abs(xt - value))) for value in part_x])
        block = whole[rows, :]
        swapped = fraction * block[:, 7] + (1.0 - fraction) * block[:, 8]
        worst = max(worst, _relative(swapped, theirs))
    assert worst > 1e-3, (
        f"swapping the two bracketing weights moved the slice by only {worst:.3e}; this "
        f"case cannot tell the weights apart and the test above proves nothing"
    )


@requires_meep
@skip_without_meep
def test_on_grid_and_bulk_slices_are_still_exact_index_cuts(stepped_sim):
    """The non-degenerate path is untouched: it must stay bit-identical to MEEP's."""
    mp, sim = stepped_sim
    driver = _driver_module()
    whole = np.asarray(sim.get_array(mp.Hz))

    vol = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(2.0, 1.0, 0))
    mine = driver.slice_like_meep(mp, sim, whole, vol, None, None)
    theirs = np.asarray(sim.get_array(mp.Hz, vol))
    assert np.array_equal(mine, theirs), "a two-dimensional sub-volume is a plain index cut"


@requires_meep
@skip_without_meep
@pytest.mark.parametrize("offsets", [(0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (0.3, 0.7)])
def test_every_axis_degenerate_at_once_still_matches(stepped_sim, offsets):
    """Two collapsed axes in one request — the rank-2 to rank-0 reduction.

    Worth measuring rather than assuming: ``get_dft_array``'s rank-0 collapse is
    the one MEEP path where this reduction is NOT the interpolated value
    (``test_driver_integration.py``'s registration sweep excludes it, measured 7.0e+00
    against a manual interpolation of MEEP's own atlas), so the read side had to be
    checked for the same quirk instead of inheriting the conclusion. It does not have
    it — ``do_get_array_slice`` recomputes the rank AFTER ``collapse_array`` rather
    than returning early — and both axes reduce exactly.
    """
    mp, sim = stepped_sim
    driver = _driver_module()
    whole = np.asarray(sim.get_array(mp.Hz))
    tics = [np.asarray(sim.get_array_metadata()[axis], dtype=float) for axis in range(2)]
    start = (12, 7)
    centre = [float(tics[a][start[a]] + offsets[a] * (tics[a][1] - tics[a][0])) for a in range(2)]

    vol = mp.Volume(center=mp.Vector3(centre[0], centre[1], 0), size=mp.Vector3())
    mine = np.asarray(driver.slice_like_meep(mp, sim, whole, vol, None, None))
    theirs = np.asarray(sim.get_array(mp.Hz, vol))
    assert mine.shape == theirs.shape == (), f"MEEP reduces to {theirs.shape}, this to {mine.shape}"
    error = abs(float(mine) - float(theirs)) / max(abs(float(theirs)), 1e-30)
    assert error < _BAR, f"point request at {offsets}: {error:.3e} against MEEP"


@requires_meep
@skip_without_meep
def test_three_degenerate_axes_in_3d_still_match():
    """The same reduction one rank up, where MEEP's DFT reader is known to differ.

    Kept separate because it needs its own 3-D run; without it the case above could
    only ever say that TWO axes reduce, and the quirk it is checking for is a
    rank-3 one.
    """
    import meep as mp

    driver = _driver_module()
    sim = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 2),
        resolution=8,
        sources=[mp.Source(mp.GaussianSource(0.5, fwidth=0.4), mp.Ez, mp.Vector3())],
        boundary_layers=[mp.PML(0.4)],
    )
    sim.run(until=2)
    whole = np.asarray(sim.get_array(mp.Ez))
    tics = [np.asarray(sim.get_array_metadata()[axis], dtype=float) for axis in range(3)]
    offsets = (0.3, 0.7, 0.4)
    centre = [float(tics[a][5] + offsets[a] * (tics[a][1] - tics[a][0])) for a in range(3)]

    vol = mp.Volume(center=mp.Vector3(*centre), size=mp.Vector3())
    mine = np.asarray(driver.slice_like_meep(mp, sim, whole, vol, None, None))
    theirs = np.asarray(sim.get_array(mp.Ez, vol))
    assert mine.shape == theirs.shape == (), f"MEEP reduces to {theirs.shape}, this to {mine.shape}"
    error = abs(float(mine) - float(theirs)) / max(abs(float(theirs)), 1e-30)
    assert error < _BAR, f"three collapsed axes: {error:.3e} against MEEP"


@requires_meep
@skip_without_meep
def test_get_field_point_still_refuses_an_off_grid_point(stepped_sim):
    """The deliberate asymmetry, stated as a test rather than left to a comment.

    ``get_array`` collapses because MEEP collapses. ``get_field_point`` does not,
    because MEEP answers it from ``fields::get_field`` on the component's own Yee
    lattice — a different interpolation from the one applied here to the centred
    array, and one this path has no oracle for.
    """
    mp, sim = stepped_sim
    driver = _driver_module()
    tics = np.asarray(sim.get_array_metadata()[1], dtype=float)
    off_grid = float(tics[7] + 0.3 * (tics[1] - tics[0]))
    whole = np.asarray(sim.get_array(mp.Hz))
    vol = mp.Volume(center=mp.Vector3(0.05, off_grid, 0), size=mp.Vector3())

    with pytest.raises(driver.UnsupportedDrive):
        driver.slice_like_meep(mp, sim, whole, vol, None, None, collapse_empty_axes=False)


@requires_meep
@skip_without_meep
def test_a_non_degenerate_off_grid_axis_is_still_refused(stepped_sim):
    """Collapsing is licensed by a ZERO requested extent, not by an off-grid coordinate.

    A hand-built metadata pair stands in for MEEP here on purpose: MEEP's own
    ``get_array_metadata`` snaps a non-degenerate axis onto grid tics, so the case
    cannot be provoked through it — and the guard would then be untested until some
    future MEEP stopped snapping.
    """
    mp, sim = stepped_sim
    driver = _driver_module()
    whole = np.asarray(sim.get_array(mp.Hz))
    full = sim.get_array_metadata()
    xt = np.asarray(full[0], dtype=float)

    class Lying:
        """``sim`` with one axis of the REQUEST reported off-grid at full extent."""

        def __init__(self, real):
            self._real = real

        def get_array_metadata(self, vol=None, **kwargs):
            meta = self._real.get_array_metadata(vol=vol, **kwargs)
            if vol is None:
                return meta
            shifted = np.asarray(meta[0], dtype=float) + 0.3 * (xt[1] - xt[0])
            return (shifted,) + tuple(meta[1:])

    vol = mp.Volume(center=mp.Vector3(0, 0, 0), size=mp.Vector3(2.0, 1.0, 0))
    with pytest.raises(driver.UnsupportedDrive):
        driver.slice_like_meep(mp, Lying(sim), whole, vol, None, None)
