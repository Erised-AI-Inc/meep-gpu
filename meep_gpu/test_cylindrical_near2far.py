"""Cylindrical near-to-far (MEEP ``greencyl``) — the ring Green's function and its parity.

A Dcyl near-field sample is not a point current: it is a RING of current at its own
radius carrying the run's ``exp(i*m*phi)`` dependence, and its far field is the phi
integral of the free-space dyadic as the source rotates around the axis
(``greencyl``, src/near2far.cpp:275-349). Radiating the same near field through the
Cartesian dyadic instead returns a smooth, confidently wrong pattern — measured 5.9
relative on the cell below, so the difference is the whole answer, not a correction.

The suite is built in two tiers, because the failure modes split:

* **The transform, exactly.** The vectorized quadrature is compared against a literal
  scalar transcription of MEEP's own C++ loop (:func:`_scalar_greencyl` below, written
  from near2far.cpp line by line and per stored component, as ``farfield_lowlevel``
  calls it) and against the closed-form far field of a single ring of current, which is
  a Bessel function of order ``m``. Both land at the float64 floor / at the Fresnel
  term, so a defect in the rotation, the ``exp(i*m*phi)`` phase, the ``1/2pi``
  normalization, the point re-use or the stopping rule cannot hide in them.

* **The whole path, end to end.** A cylindrical run's far field against CPU MEEP's
  ``get_farfield`` on the identical cell, at six far points, two resolutions and two
  azimuthal numbers, with the residual required to FALL as the grid is refined — it is
  the same cell-centre-versus-Yee sampling difference the Cartesian transformation
  documents, and its second-order convergence is what says so.

Every number quoted was measured on the run that landed this, with the MEEP version
named: the two MEEP releases in this tree do not run the same quadrature (MEEP #3047
changed the convergence criterion after 1.29).
"""

from __future__ import annotations

import cmath
import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from .driver import FdtdDriver
from .dft import Near2FarMonitor, Near2FarRegion, _cylinder_regions, to_numpy
from .grid import Grid

_HAS_MEEP = importlib.util.find_spec("meep") is not None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    not _HAS_MEEP, reason="CPU MEEP is not installed in this environment")

# One cell, shared by the CPU-MEEP oracle and every engine run below: vacuum, a PML on
# the r wall and both z faces, an OFF-AXIS Ez source (so m=1 is excited at all), and a
# closed surface of revolution well clear of the absorber.
_CASE = {
    "r": 2.0, "z": 4.0, "until": 15.0, "fcen": 1.0, "fwidth": 0.6, "pml": 0.5,
    "source": (0.4, 0.0, 0.0), "radius": 1.0, "height": 1.2,
    "points": [[3.0, 0.0, 4.0], [5.0, 0.0, 0.0], [1.0, 0.0, 6.0],
               [0.001, 0.0, 7.0], [2.0, 0.0, -5.0], [8.0, 0.0, -3.0]],
}

_ORACLE = r'''
"""CPU-MEEP oracle: cylindrical near2far far fields at a set of (r, 0, z) points."""
import json, sys
import numpy as np
import meep as mp

case = json.loads(sys.argv[1])
points = np.asarray(case["points"], dtype=float)
out = {"meep_version": np.array(mp.__version__)}
for m in (0, 1, 2):
    for resolution in (20, 40):
        sim = mp.Simulation(
            cell_size=mp.Vector3(case["r"], 0, case["z"]), resolution=resolution,
            dimensions=mp.CYLINDRICAL, m=m, force_complex_fields=True,
            boundary_layers=[mp.PML(case["pml"], direction=mp.R, side=mp.High),
                             mp.PML(case["pml"], direction=mp.Z)],
            sources=[mp.Source(mp.GaussianSource(case["fcen"], fwidth=case["fwidth"]),
                               component=mp.Ez, center=mp.Vector3(*case["source"]))])
        radius, half = case["radius"], 0.5 * case["height"]
        regions = [
            mp.Near2FarRegion(center=mp.Vector3(0.5 * radius, 0, -half),
                              size=mp.Vector3(radius, 0, 0), direction=mp.Z, weight=-1.0),
            mp.Near2FarRegion(center=mp.Vector3(0.5 * radius, 0, +half),
                              size=mp.Vector3(radius, 0, 0), direction=mp.Z, weight=+1.0),
            mp.Near2FarRegion(center=mp.Vector3(radius, 0, 0),
                              size=mp.Vector3(0, 0, 2 * half), direction=mp.R, weight=+1.0),
        ]
        near2far = sim.add_near2far(case["fcen"], 0, 1, *regions, decimation_factor=1)
        sim.run(until=case["until"])
        out[f"m{m}_r{resolution}"] = np.array(
            [sim.get_farfield(near2far, mp.Vector3(point[0], 0, point[2])) for point in points],
            dtype=np.complex128)
        print(f"oracle m={m} res={resolution} done", flush=True)
        sim.reset_meep()
np.savez(sys.argv[2], **out)
'''


@pytest.fixture(scope="module")
def cylindrical_near2far_oracle(tmp_path_factory):
    """CPU-MEEP far fields for every (m, resolution) case, from one subprocess."""
    if not _HAS_MEEP:
        pytest.skip("CPU MEEP is not installed")
    directory = tmp_path_factory.mktemp("cylindrical_near2far_oracle")
    script = directory / "oracle.py"
    script.write_text(_ORACLE, encoding="utf-8")
    output = directory / "farfields.npz"
    completed = subprocess.run(
        [sys.executable, str(script), json.dumps(_CASE), str(output)],
        capture_output=True, text=True, timeout=1800,
        env=dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE"),
    )
    assert completed.returncode == 0 and output.exists(), (
        f"cylindrical near2far oracle failed (exit {completed.returncode}).\n"
        f"stderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _engine_run(m, resolution, greencyl_tol=1e-8, silent=False):
    """The oracle's cell on this engine, with the same closed surface of revolution."""
    driver = FdtdDriver(cell_size=(_CASE["r"], 0.0, _CASE["z"]), resolution=resolution,
                        cylindrical=True, m=m, force_complex_fields=True,
                        boundaries={"z": "metallic"})
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    cells = int(round(_CASE["pml"] * resolution))
    driver.setup_pml({"x": (0, cells), "z": (cells, cells)})
    if not silent:
        driver.add_source({"component": "Ez", "frequency": _CASE["fcen"],
                           "source_type": "gaussian", "fwidth": _CASE["fwidth"],
                           "center": _CASE["source"], "size": (0.0, 0.0, 0.0)})
    monitor = Near2FarMonitor.on_driver(
        driver, frequencies=[_CASE["fcen"]],
        regions=_cylinder_regions(_CASE["radius"], _CASE["height"], 0.0),
        closed=True, decimation_factor=1, greencyl_tol=greencyl_tol,
    )
    driver.run(until=_CASE["until"])
    return driver, monitor


def _relative(produced, reference):  # Relative L2 over every component and point.
    return float(np.linalg.norm(produced - reference) / np.linalg.norm(reference))


# --------------------------------------------------------------------------------
# A literal scalar transcription of MEEP's own C++, used as the tight-tier reference.
# --------------------------------------------------------------------------------

def _scalar_green3d(x, freq, eps, mu, x0, direction, magnetic, f0):
    """src/near2far.cpp:133-187, written out with MEEP's own names and grouping."""
    rhat = np.asarray(x, dtype=float) - np.asarray(x0, dtype=float)
    r = float(np.linalg.norm(rhat))
    rhat = rhat / r
    n = math.sqrt(eps * mu)
    k = 2 * math.pi * freq * n
    ikr = 1j * k * r
    ikr2 = -((k * r) ** 2)
    expfac = f0 * cmath.rect(k * n / (4 * math.pi * r), k * r + math.pi * 0.5)
    impedance = math.sqrt(mu / eps)
    p = np.zeros(3)
    p[direction] = 1.0
    pdotrhat = float(np.dot(p, rhat))
    rhatcrossp = np.cross(rhat, p)
    term1 = 1.0 - 1.0 / ikr + 1.0 / ikr2
    term2 = (-1.0 + 3.0 / ikr - 3.0 / ikr2) * pdotrhat
    term3 = 1.0 - 1.0 / ikr
    fields = np.zeros(6, dtype=complex)
    if not magnetic:
        scale = expfac / eps
        fields[0:3] = scale * (term1 * p + term2 * rhat)
        fields[3:6] = scale * term3 * rhatcrossp / impedance
    else:
        scale = expfac / mu
        fields[0:3] = -scale * term3 * rhatcrossp * impedance
        fields[3:6] = scale * (term1 * p + term2 * rhat)
    return fields


def _scalar_greencyl(x, freq, eps, mu, x0, direction, magnetic, f0, m, tol):
    """src/near2far.cpp:275-349, one stored component at a time as MEEP calls it.

    ``direction`` is 0 (R), 1 (P) or 2 (Z); ``x`` is the ``(r, 0, z)`` far point and
    ``x0`` the ``(r0, z0)`` source ring.
    """
    x_3d = np.array([x[0], 0.0, x[2]], dtype=float)
    fields = np.zeros(6, dtype=complex)
    first_count = 16 + int(4 * abs(m))
    spacing = 2.0 / first_count
    integrand_norm = 0.0
    count = first_count
    while count <= 65536:
        spacing *= 0.5
        angle_step = spacing * 2 * math.pi
        partial = fields * 0.5
        integrand_norm *= 0.5
        start = 1 if count > first_count else 0
        stride = 2 if count > first_count else 1
        for index in range(start, count, stride):
            angle = index * angle_step
            cosine, sine = math.cos(angle), math.sin(angle)
            rotated = np.array([x0[0] * cosine, x0[0] * sine, x0[1]])
            amplitude = f0 * cmath.rect(1.0, m * angle) * spacing
            if direction == 2:
                calls = ((2, amplitude),)
            elif direction == 0:  # r_hat = c x_hat + s y_hat
                calls = ((0, amplitude * cosine), (1, amplitude * sine))
            else:  # phi_hat = c y_hat - s x_hat
                calls = ((0, amplitude * (-sine)), (1, amplitude * cosine))
            for cartesian, weight in calls:
                contribution = _scalar_green3d(x_3d, freq, eps, mu, rotated,
                                               cartesian, magnetic, weight)
                partial = partial + contribution
                integrand_norm += float(np.sum(np.abs(contribution)))
        change = float(np.sum(np.abs(fields - partial)))
        fields = partial
        if change <= integrand_norm * tol:
            break
        count *= 2
    return fields


def _scalar_farfield(monitor, positions, currents, magnetic_currents, points, m, tol):
    """``farfield_lowlevel``'s loop: one greencyl call per (ring, component, family)."""
    result = np.zeros((len(points), len(monitor.frequencies), 6), dtype=complex)
    for point_index, point in enumerate(points):
        for freq_index, frequency in enumerate(monitor.frequencies):
            for ring in range(positions.shape[0]):
                source = (float(positions[ring, 0]), float(positions[ring, 2]))
                for direction in range(3):
                    for magnetic, stack in ((False, currents), (True, magnetic_currents)):
                        amplitude = complex(stack[freq_index, ring, direction])
                        result[point_index, freq_index] += _scalar_greencyl(
                            point, frequency, monitor.eps, monitor.mu, source,
                            direction, magnetic, amplitude, m, tol)
    return result


def test_greencyl_matches_a_scalar_transcription_of_meeps_ring_quadrature():
    """The vectorized ring integral IS MEEP's loop — at the float64 floor.

    The tight tier. :func:`_scalar_greencyl` is near2far.cpp written out line by line
    and called the way ``farfield_lowlevel`` calls it: once per stored (ring, component,
    family), each with its OWN doubling schedule. This engine instead evaluates the
    whole ``(J, M)`` pair of a ring as one vector source, which is the same integral by
    linearity but lets the convergence test see the L1 norm of the summed integrand
    rather than of each component separately — so the two can in principle stop at
    different ``N``. Measured, they do not, and the answers agree to 1.8e-15..2.7e-15
    for m in (0, 1, -1, 2, 5) at BOTH tol=1e-3 and tol=1e-10.

    The second case is the one that exercises the doubling at all: a ring of radius 20
    at unit frequency has ``k*a = 126``, and the quadrature runs 28 -> 56 -> 112 -> 224
    points before it converges (measured from the transcription's own schedule). There
    the two agree to 1.3e-14. The small-ring cases converge at N=32 for every tolerance
    from 1e-2 to 1e-10, because a trapezoid on a smooth periodic integrand falls from
    O(1) error to below the float64 floor in a single doubling — which is also why
    ``greencyl_tol`` changes nothing measurable on the cell below, and is reported here
    rather than dressed up as a convergence study.
    """
    rng = np.random.default_rng(7)
    grid = Grid(resolution=20, cell_size=(2.0, 0.0, 4.0), cylindrical=True, m=1)
    monitor = Near2FarMonitor(grid, frequencies=[0.8, 1.3],
                              regions=_cylinder_regions(1.0, 1.2, 0.0), closed=True)
    rings = 9
    positions = np.stack([rng.uniform(0.05, 1.2, rings), np.zeros(rings),
                          rng.uniform(-0.6, 0.6, rings)], axis=1)
    shape = (2, rings, 3)
    currents = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    magnetic = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    points = np.array([[3.0, 0.0, 4.0], [0.0, 0.0, 7.0], [6.0, 0.0, -2.0]])
    for m in (0, 1, -1, 2, 5):
        monitor.m = float(m)
        for tol in (1e-3, 1e-10):
            monitor.greencyl_tol = tol
            produced = monitor._greencyl(positions, currents, magnetic, points)
            reference = _scalar_farfield(monitor, positions, currents, magnetic,
                                         points, m, tol)
            error = _relative(produced, reference)
            print(f"greencyl vs transcription: m={m} tol={tol:.0e} rel={error:.3e}", flush=True)
            assert error < 1e-12, (
                f"m={m}, tol={tol:.0e}: the vectorized ring quadrature is {error:.3e} from a "
                f"literal transcription of near2far.cpp")

    # k*a = 126: the doubling runs four levels, so the point re-use (only odd indices
    # after the first pass) and the stopping rule are both exercised, not just entered.
    big = Grid(resolution=4, cell_size=(60.0, 0.0, 20.0), cylindrical=True, m=3)
    wide = Near2FarMonitor(big, frequencies=[1.0],
                           regions=_cylinder_regions(20.0, 4.0, 0.0), closed=True)
    ring = np.array([[20.0, 0.0, 0.5]])
    ring_current = np.array([[[0.3 + 0.2j, -0.4 + 0.1j, 1.0 - 0.5j]]])
    quiet = np.zeros_like(ring_current)
    far = np.array([[30.0, 0.0, 40.0], [0.0, 0.0, 60.0]])
    for m in (0, 3, 10):
        wide.m = float(m)
        produced = wide._greencyl(ring, ring_current, quiet, far)
        reference = _scalar_farfield(wide, ring, ring_current, quiet, far, m,
                                     wide.greencyl_tol)
        error = _relative(produced, reference)
        print(f"greencyl vs transcription (k*a=126): m={m} rel={error:.3e}", flush=True)
        assert error < 1e-12, f"large-ring m={m}: {error:.3e} from the transcription"


def test_greencyl_reaches_the_analytic_ring_far_field():
    """One ring of z-directed current, against its closed form: a Bessel function of order m.

    The independent known value. In the far zone the ring integral collapses onto

        (1/2pi) Integral[ e^(-i k a sin(theta) cos(phi)) e^(i m phi) ] dphi
            = (-i)^m J_m(k a sin(theta))

    so a ring at radius ``a``, height ``z0``, carrying ``J_z e^(i m phi)``, radiates

        E = i omega mu g (I - rhat rhat) . zhat (-i)^m J_m(k a sin theta) e^(-i k z0 cos theta)
        H = g (i k) (rhat x zhat) (...)

    with ``g = e^(i k R)/(4 pi R)``. Nothing in that expression comes from this engine:
    it fixes the ``1/2pi`` normalization, the sign and order of the ``exp(i*m*phi)``
    phase, and the whole theta dependence at once, and it is wrong by O(1) if the
    quadrature drops the phase (a J_0 pattern in place of a J_m one).

    Measured at R = 2e5 wavelengths: 1.7e-05 (m=0) to 3.1e-05 (m=3). That residual is
    the FRESNEL term the closed form drops, not a defect — the asymptote linearizes
    ``|x - x0|`` and its next term is ``k (a sin theta)^2 / (2R)``. Ten times the
    radius divides it by exactly ten (1.674e-05 -> 1.674e-06 at m=0, 2.018e-05 ->
    2.018e-06 at m=2), which is what makes the comparison a measurement of the
    transform rather than of the asymptote.

    ``greencyl_tol`` is 1e-6 here, not tighter: at R = 2e6 the phase ``k*r`` is 1.3e7
    radians, so ``float64`` resolves the integrand itself to only ~1e-9 relative and a
    tolerance below that floor never converges — the quadrature then runs to MEEP's
    65536-point cap and the answer does not improve. That is MEEP's behaviour too (the
    cap is its, near2far.cpp:294); it is a property of huge observation radii, and the
    residual measured above is unaffected by it.
    """
    bessel = pytest.importorskip("scipy.special").jv
    grid = Grid(resolution=20, cell_size=(4.0, 0.0, 4.0), cylindrical=True, m=0)
    radius, height, frequency = 0.7, 0.3, 1.0
    wavenumber = 2 * math.pi * frequency
    ring = np.array([[radius, 0.0, height]])
    current = np.array([[[0.0, 0.0, 1.0 + 0j]]])
    quiet = np.zeros_like(current)
    axis = np.array([0.0, 0.0, 1.0])
    measured = {}
    for span in (2.0e5, 2.0e6):
        for m in (0, 1, 2, -2, 3):
            monitor = Near2FarMonitor(grid, frequencies=[frequency],
                                      regions=_cylinder_regions(1.0, 1.0, 0.0),
                                      closed=True, greencyl_tol=1e-6)
            monitor.m = float(m)
            worst = 0.0
            for theta in (0.3, 0.9, 1.4, 2.2):
                point = np.array([[span * math.sin(theta), 0.0, span * math.cos(theta)]])
                produced = monitor._greencyl(ring, current, quiet, point)[0, 0]
                unit = np.array([math.sin(theta), 0.0, math.cos(theta)])
                green = np.exp(1j * wavenumber * span) / (4 * math.pi * span)
                ring_factor = ((-1j) ** m * bessel(m, wavenumber * radius * math.sin(theta))
                               * np.exp(-1j * wavenumber * height * math.cos(theta)))
                expected = np.concatenate([
                    1j * (2 * math.pi * frequency) * green * ring_factor
                    * (axis - unit * float(np.dot(unit, axis))),
                    green * 1j * wavenumber * ring_factor * np.cross(unit, axis),
                ])
                worst = max(worst, _relative(produced, expected))
            measured[(span, m)] = worst
            print(f"ring vs Bessel: R={span:.0e} m={m} worst={worst:.3e}", flush=True)
            assert worst < 1e-4, (
                f"a ring of z current at m={m}, R={span:.0e} is {worst:.3e} from "
                f"(-i)^m J_m(k a sin theta)")
    for m in (0, 1, 2, -2, 3):
        near, far = measured[(2.0e5, m)], measured[(2.0e6, m)]
        assert abs(far * 10 / near - 1.0) < 0.05, (
            f"m={m}: the residual must be the 1/R Fresnel term the asymptote drops "
            f"({near:.3e} at R=2e5 against {far:.3e} at R=2e6, expected a factor of ten)")


def test_cylindrical_ring_weights_integrate_the_surface_of_revolution():
    """Every near-field sample carries 2*pi*r at ITS OWN radius — no MEEP in the loop.

    MEEP stores each Dcyl near2far sample already multiplied by ``2*pi*r``
    (loop_in_chunks.cpp:505-512) and ``greencyl`` divides that back out
    (near2far.cpp:277-278), so this engine has to carry the same factor for the
    quadrature to be MEEP's verbatim. The statement that pins it without a far field:
    the weights of a closed surface of revolution must integrate its AREA — ``pi*R^2``
    for each cap and ``2*pi*R*h`` for the wall — and exactly, because the ladder
    integrates linear functions exactly and ``2*pi*r`` is linear in r.

    Measured 4.4e-12..4.4e-08 relative across resolutions 10, 20 and 37 (the odd one
    is deliberate — a half-cell registration slip is invisible on even counts). The
    floor is the float32 weight storage, not the quadrature.

    Using the PATCH's radius on both bracketing cells of the r wall instead of each
    cell's own would leave this same total — it is the sum that is preserved, only the
    ring positions move — which is why the far-field control below is asserted too.
    """
    for resolution in (10, 20, 37):
        grid = Grid(resolution=resolution, cell_size=(3.0, 0.0, 4.0), cylindrical=True, m=1)
        monitor = Near2FarMonitor(grid, frequencies=[1.0],
                                  regions=_cylinder_regions(1.1, 1.3, 0.2), closed=True)
        for face in monitor._faces:
            weights = to_numpy(monitor._face_weights(face)).astype(np.float64)
            _, size = face.monitor._volume
            measure = grid.dx ** sum(1 for extent in size if extent > 0)
            integrated = float(weights.sum() * measure)
            exact = math.pi * 1.1 ** 2 if face.axis == 2 else 2 * math.pi * 1.1 * 1.3
            error = abs(integrated - exact) / exact
            print(f"ring weights: res={resolution} normal={face.axis} "
                  f"area={integrated:.9f} exact={exact:.9f} rel={error:.2e}", flush=True)
            assert error < 1e-6, (
                f"resolution {resolution}, normal axis {face.axis}: the ring weights "
                f"integrate {integrated:.9f}, not the surface's {exact:.9f}")


@requires_meep
@skip_without_meep
def test_cylindrical_near2far_matches_cpu_meep(cylindrical_near2far_oracle):
    """Far fields against ``sim.get_farfield`` on the identical cylindrical cell.

    Six far points spanning forward, sideways, backward and one essentially on the
    axis, at m in (0, 1, 2), resolution 20 and 40. Measured against MEEP 1.29.0 and
    1.33.0 — which produce BIT-IDENTICAL far fields on this cell, so #3047's change to
    the convergence criterion does not separate them here:

        m=0  res 20  2.09e-02      res 40  5.33e-03   (ratio 3.93)
        m=1  res 20  2.33e-02      res 40  5.88e-03   (ratio 3.96)
        m=2  res 20  2.40e-02      res 40  6.03e-03   (ratio 3.98)

    The residual is the same one the Cartesian transformation carries and for the same
    reason: this engine samples every component at cell centres, so ``J`` and ``M``
    describe the same point, while MEEP evaluates each at its own Yee position half a
    cell away. Both are consistent discretizations of the same integral; the difference
    is second order in the grid spacing, and the assertion that says so is the RATIO —
    refining the grid must divide the residual by four. A defect in the ring integral
    itself does not converge away, so the ratio is the load-bearing half of this test
    and the 5% bar is only the guard rail.

    The near2far surface at m=1 also has an axis row that MEEP weights with exactly
    zero (``2*pi*r`` at ``r = 0``) and this engine does not have at all (its innermost
    ring sits at ``r = dx/2``); that is part of the same second-order difference.

    **Per point, two ways.** A point's OWN relative error is not the right localization
    bar here: at m=2 the pattern vanishes on the axis, so the near-axis point carries
    1.3e-05 of the pattern's scale and reads 2.23e-01 relative at resolution 20 while
    contributing 2.9e-06 of the whole. It is a small denominator, and its convergence
    proves it (2.23e-01 -> 4.71e-02 at resolution 40, a factor of 4.7). So the
    localization bar is each point's residual against the PATTERN's scale, and EVERY
    point is separately required to converge — a localized defect passes neither.
    """
    measured = {}
    point_errors = {}
    for m in (0, 1, 2):
        for resolution in (20, 40):
            driver, monitor = _engine_run(m, resolution)
            produced = monitor.farfields(np.asarray(_CASE["points"]), freq_index=0)
            reference = cylindrical_near2far_oracle[f"m{m}_r{resolution}"]
            measured[(m, resolution)] = _relative(produced, reference)
            scale = float(np.abs(reference).max())
            per_point = [_relative(produced[i], reference[i]) for i in range(len(produced))]
            against_pattern = [float(np.linalg.norm(produced[i] - reference[i]) / scale)
                               for i in range(len(produced))]
            point_errors[(m, resolution)] = per_point
            print(f"cyl near2far vs MEEP {cylindrical_near2far_oracle['meep_version']}: "
                  f"m={m} res={resolution} rel={measured[(m, resolution)]:.4e} "
                  f"per-point={['%.2e' % value for value in per_point]} "
                  f"per-point/scale={['%.2e' % value for value in against_pattern]}", flush=True)
            assert measured[(m, resolution)] < 0.05, (
                f"m={m}, resolution {resolution}: {measured[(m, resolution)]:.4e} from CPU MEEP")
            assert max(against_pattern) < 0.05, (
                f"m={m}, resolution {resolution}: one point is {max(against_pattern):.4e} of the "
                f"pattern's own scale away from MEEP — a localized disagreement must not average "
                f"into a passing L2")
            # Non-triviality: a transformation that returned a smoothly scaled pattern,
            # or zeros, would pass a relative-L2 bar on a small enough field.
            ratio = float(np.abs(produced).max() / scale)
            assert 0.5 < ratio < 2.0, f"max|EH| is {ratio:.3f} of MEEP's, not the same field"
            driver.close()
    for m in (0, 1, 2):
        ratio = measured[(m, 20)] / measured[(m, 40)]
        assert ratio > 3.0, (
            f"m={m}: refining the grid must divide the residual by about four — it is the "
            f"cell-centre-versus-Yee sampling difference, not the ring integral. Measured "
            f"{measured[(m, 20)]:.4e} at resolution 20 against {measured[(m, 40)]:.4e} at 40 "
            f"(ratio {ratio:.2f})")
        for index, (coarse, fine) in enumerate(zip(point_errors[(m, 20)], point_errors[(m, 40)])):
            assert coarse / fine > 3.0, (
                f"m={m}, far point {index}: {coarse:.3e} at resolution 20 against {fine:.3e} at "
                f"40 (ratio {coarse / fine:.2f}) — every point's residual must be the sampling "
                f"difference, so every point's must converge")


@requires_meep
@skip_without_meep
def test_cylindrical_near2far_controls_break_the_comparison(cylindrical_near2far_oracle):
    """The controls: every plausible defect must be visible where the baseline is not.

    Measured on the m=1, resolution-20 cell against MEEP 1.29.0, baseline 2.33e-02:

        Cartesian dyadic on the same near data ....... 5.94e+00
        one patch's sign flipped ..................... 9.89e-01
        m sign conjugated ............................ 8.27e-01
        the ring weight 2*pi*r dropped ............... 8.60e-01
        2*pi*r_site -> 2*pi*<r> ...................... 1.31e+00
        exp(i*m*phi) dropped (quadrature run at m=0) . 4.71e-01

    The first is the one this file exists for: the near data is IDENTICAL, only the
    Green's function differs, and the answer is six times the field. The last two are
    the quiet ones — both leave a smooth pattern that falls off as 1/r and integrates
    to a plausible power.
    """
    reference = cylindrical_near2far_oracle["m1_r20"]
    points = np.asarray(_CASE["points"])
    driver, monitor = _engine_run(1, 20)
    positions, currents, magnetic = monitor._surface_currents()
    baseline = _relative(monitor._greencyl(positions, currents, magnetic, points)[:, 0, :],
                         reference)
    assert baseline < 0.05, f"the control harness must start from the passing case, got {baseline:.3e}"

    controls = {}
    controls["cartesian dyadic on the same data"] = _relative(
        monitor._radiate(positions, currents, magnetic, points)[:, 0, :], reference)
    flipped_j, flipped_m = currents.copy(), magnetic.copy()
    wall = to_numpy(positions[:, 0]) >= 0.99 * _CASE["radius"]
    flipped_j[:, wall] *= -1
    flipped_m[:, wall] *= -1
    controls["one patch's sign flipped"] = _relative(
        monitor._greencyl(positions, flipped_j, flipped_m, points)[:, 0, :], reference)
    radius = to_numpy(positions[:, 0])
    scale = (radius.mean() / radius)[None, :, None]
    controls["2*pi*r_site -> 2*pi*<r>"] = _relative(
        monitor._greencyl(positions, currents * scale, magnetic * scale, points)[:, 0, :],
        reference)
    drop = (1.0 / (2 * math.pi * radius))[None, :, None]
    controls["the ring weight dropped"] = _relative(
        monitor._greencyl(positions, currents * drop, magnetic * drop, points)[:, 0, :],
        reference)
    monitor.m = -1.0
    controls["m sign conjugated"] = _relative(
        monitor._greencyl(positions, currents, magnetic, points)[:, 0, :], reference)
    monitor.m = 0.0
    controls["exp(i*m*phi) dropped"] = _relative(
        monitor._greencyl(positions, currents, magnetic, points)[:, 0, :], reference)
    monitor.m = 1.0
    driver.close()

    for name, error in controls.items():
        print(f"control {name}: {error:.3e} (baseline {baseline:.3e})", flush=True)
        assert error > 0.3, (
            f"the control '{name}' scores {error:.3e} against a baseline of {baseline:.3e}: "
            f"this comparison cannot see that defect")


@requires_meep
@skip_without_meep
def test_cylindrical_near2far_with_no_source_is_exactly_zero():
    """A run with nothing in the surface radiates exactly zero, and the same cell with a
    source does not.

    The degenerate result every tolerance-based check passes: an accumulator that never
    saw a field reduces to a perfectly smooth pattern of zeros at every angle, and
    nothing about it says the run recorded nothing.
    """
    driver, monitor = _engine_run(1, 20, silent=True)
    silent = monitor.farfields(np.asarray(_CASE["points"]), freq_index=0)
    assert np.count_nonzero(silent) == 0, (
        f"a run with no source must radiate exactly zero, got max |field| "
        f"{np.max(np.abs(silent)):.3e}")
    driver.close()
    driver, monitor = _engine_run(1, 20)
    loud = monitor.farfields(np.asarray(_CASE["points"]), freq_index=0)
    assert np.min(np.abs(loud)) > 0.0, "the positive control must radiate at every point"
    driver.close()


def test_cylindrical_near2far_radiation_pattern_is_swept_on_the_axis():
    """The polar sweep centres on the axis and stays on the phi = 0 half-plane.

    A Cartesian sphere centre is the midpoint of the surface's extent on every axis,
    which in Dcyl puts it at ``r = R/2`` — off the axis, tilting every angle by a
    smooth amount that no shape of the pattern would reveal. The cylindrical default is
    ``r = 0``, and an off-axis centre or a nonzero azimuth is refused rather than
    reinterpreted.
    """
    grid = Grid(resolution=20, cell_size=(2.0, 0.0, 4.0), cylindrical=True, m=1)
    monitor = Near2FarMonitor(grid, frequencies=[1.0],
                              regions=_cylinder_regions(1.0, 1.2, 0.4), closed=True)
    low, high = monitor.enclosure_bounds()
    assert (low[0], high[0]) == (0.0, 1.0)
    assert low[2] == pytest.approx(-0.2) and high[2] == pytest.approx(1.0)
    theta = np.linspace(0.05, np.pi - 0.05, 5)
    with pytest.raises(ValueError, match="phi = 0 half-plane"):
        monitor.radiation_pattern(theta, 0.3, 20.0)
    with pytest.raises(ValueError, match="theta in"):
        monitor.radiation_pattern(np.array([-0.1]), 0.0, 20.0)
    with pytest.raises(ValueError, match="centred on the axis"):
        monitor.radiation_pattern(theta, 0.0, 20.0, center=(0.5, 0.0, 0.0))
    # With no field accumulated the pattern is zero, but the geometry it asks for is
    # what this pins: the sphere is centred on the axis at the surface's mid-height.
    pattern = monitor.radiation_pattern(theta, 0.0, 20.0)
    assert pattern.shape == theta.shape and np.count_nonzero(pattern) == 0


def test_cylindrical_near2far_refusals_are_by_name():
    """Each shape this transformation does not serve raises naming the constraint."""
    grid = Grid(resolution=20, cell_size=(2.0, 0.0, 4.0), cylindrical=True, m=1)
    regions = _cylinder_regions(1.0, 1.2, 0.0)
    with pytest.raises(NotImplementedError, match="phi-normal"):
        Near2FarMonitor(grid, 1.0, (Near2FarRegion(center=(0.5, 0.0, 0.0),
                                                   size=(1.0, 0.0, 1.2), direction=1),),
                        closed=False)
    with pytest.raises(NotImplementedError, match="cylinder"):
        Near2FarMonitor.box(grid, 1.0, center=(0, 0, 0), size=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="r wall plus the two z caps"):
        Near2FarMonitor(grid, 1.0, regions)  # closed left to be derived
    with pytest.raises(ValueError, match="greencyl_tol"):
        Near2FarMonitor(grid, 1.0, regions, closed=True, greencyl_tol=0.0)
    monitor = Near2FarMonitor(grid, 1.0, regions, closed=True)
    assert monitor.greencyl_tol == 1e-3, "MEEP's own default (src/meep.hpp:1363)"
    with pytest.raises(ValueError, match="phi. coordinate is not zero"):
        monitor.farfield((3.0, 1.0, 4.0))
    with pytest.raises(ValueError, match="needs r >= 0"):
        monitor.farfield((-3.0, 0.0, 4.0))
    # A far point ON one of the cap's own rings: the ring integral is singular there,
    # and the returned number would be enormous, finite and smooth.
    with pytest.raises(ValueError, match="nearest source ring"):
        monitor.farfield((0.475, 0.0, 0.575))
