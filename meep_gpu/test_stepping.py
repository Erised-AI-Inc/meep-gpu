"""
Value tests for the FDTD leapfrog stepping kernels.

They pin the conventions the rest of the engine is built on: MEEP's curl formula
and its stride negation for D, the periodic / Bloch-periodic / metallic / mirror
ghost rules, the split-field PML recurrence together with the integer vs
half-integer coefficient pairing, mirror-plane ownership masking (including that
the mask is applied before the PML auxiliary integrates), and the sequencing
contract that leaves BOTH ``fill_symmetry_bc_B`` and ``fill_symmetry_bc_D`` to the
driver, which calls each one after its own family's currents land. The physics check is a
vacuum plane wave that must reproduce the closed-form Yee dispersion relation and
converge toward c = 1 as O(dx^2).

The Bloch cases are held to the same standard. The wrap factor is checked against
MEEP's ``eikna`` expression term for term (including the 2*pi and the exactly-
pinned Brillouin-zone edge), and the physics check is the exact discrete
eigenmode of the Yee scheme at that k: it must come back after hundreds of steps
as itself times a unit-magnitude phase. Three controls stand behind it, because a
travelling wave that merely *looks* smooth is not evidence — the same seed run at
k = 0, at the conjugate wavevector, and against a magnitude-only comparison all
have to fail.

Mirror symmetry is held to a harder standard than either: a folded run must
reproduce the unfolded full-domain run BIT FOR BIT, on every axis (X, Y and Z),
for both declared phases (even and odd), and in combination with odd cell counts,
an absorber, a Bloch phase and a susceptibility. Every kernel here is +, - and *
on float32/complex64 with real coefficients, so exact equality is the right bar
and any tolerance would hide a wrong parity, a ghost read from the wrong cell or
a mask on the wrong axis — each of which moves the answer a few percent and
leaves a perfectly smooth field. The even X/Y case additionally carries a byte
digest, because it is the configuration the recorded CPU-MEEP floors were
measured on.

NumPy path only — no CuPy, no CUDA, no meep.
"""

from __future__ import annotations

import cmath
import math

import numpy
import pytest

from . import stepping
from .dispersion import LORENTZIAN, PolarizationState, Susceptibility
from .fields import IYEE_SHIFTS, Fields, StepScratch, get_symmetry_phase, mirror_parity
from .grid import Grid, Mirror
from .pml import PML


def make_fields(cell_size=(0.6, 0.6, 0.6), resolution=10.0, symmetry=(),
                complex_fields=True, boundaries=None) -> Fields:  # Small default system; overrides keep call sites terse.
    grid = Grid(resolution=resolution, cell_size=cell_size, symmetry=symmetry,
                boundaries=boundaries)
    return Fields(grid=grid, force_complex_fields=complex_fields)


def seed(fields: Fields, names, seed_value: int) -> None:  # Fill components with reproducible non-degenerate values.
    generator = numpy.random.default_rng(seed_value)
    for name in names:
        array = getattr(fields, name)
        values = generator.standard_normal(array.shape)
        if numpy.iscomplexobj(array):
            values = values + 1j * generator.standard_normal(array.shape)
        array[...] = values


def snapshot(fields: Fields, names) -> dict:  # Copy named component arrays for before/after comparison.
    return {name: getattr(fields, name).copy() for name in names}


def neighbour(cell, axis: int, stride: int, shape) -> tuple:  # Periodic neighbour one stride along one axis.
    index = list(cell)
    index[axis] = (index[axis] + stride) % shape[axis]
    return tuple(index)


def reference_curl(first, second, first_axis: int, second_axis: int, stride: int):
    """Explicit-index restatement of MEEP's curl, independent of the ported slicing.

    (g1[i+s] - g1[i]) + (g2[i] - g2[i+s]) with periodic wrap; s = +1 reproduces
    the forward differences of step_B and s = -1 the negated strides of step_D.
    """
    shape = first.shape
    out = numpy.zeros(shape, dtype=first.dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                cell = (i, j, k)
                out[cell] = ((first[neighbour(cell, first_axis, stride, shape)] - first[cell])
                             + (second[cell] - second[neighbour(cell, second_axis, stride, shape)]))
    return out


def test_step_B_applies_the_forward_difference_curl():
    fields = make_fields()
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 11)
    electric = {name: fields.get_E(name).copy() for name in ("Ex", "Ey", "Ez")}
    before = snapshot(fields, ("Bx", "By", "Bz"))
    dtdx = fields.grid.dt / fields.grid.dx

    stepping.step_B(fields)

    expected = {
        "Bx": before["Bx"] - dtdx * reference_curl(electric["Ez"], electric["Ey"], 1, 2, +1),
        "By": before["By"] - dtdx * reference_curl(electric["Ex"], electric["Ez"], 2, 0, +1),
        "Bz": before["Bz"] - dtdx * reference_curl(electric["Ey"], electric["Ex"], 0, 1, +1),
    }
    for name, value in expected.items():
        assert numpy.allclose(getattr(fields, name), value, rtol=1e-6, atol=1e-7)

    # Spot check with literal indices so the stride convention reads directly.
    i, j, k = 2, 3, 4
    assert fields.Bx[i, j, k] == pytest.approx(
        before["Bx"][i, j, k]
        - dtdx * ((electric["Ez"][i, j + 1, k] - electric["Ez"][i, j, k])
                  + (electric["Ey"][i, j, k] - electric["Ey"][i, j, k + 1])),
        rel=1e-6,
    )


def test_step_D_applies_the_backward_difference_curl_via_negated_strides():
    fields = make_fields()
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 12)
    magnetic = {name: fields.get_H(name).copy() for name in ("Hx", "Hy", "Hz")}
    before = snapshot(fields, ("Dx", "Dy", "Dz"))
    dtdx = fields.grid.dt / fields.grid.dx

    stepping.step_D(fields)

    expected = {
        "Dx": before["Dx"] - dtdx * reference_curl(magnetic["Hz"], magnetic["Hy"], 1, 2, -1),
        "Dy": before["Dy"] - dtdx * reference_curl(magnetic["Hx"], magnetic["Hz"], 2, 0, -1),
        "Dz": before["Dz"] - dtdx * reference_curl(magnetic["Hy"], magnetic["Hx"], 0, 1, -1),
    }
    for name, value in expected.items():
        assert numpy.allclose(getattr(fields, name), value, rtol=1e-6, atol=1e-7)

    i, j, k = 2, 3, 4
    assert fields.Dx[i, j, k] == pytest.approx(
        before["Dx"][i, j, k]
        - dtdx * ((magnetic["Hz"][i, j - 1, k] - magnetic["Hz"][i, j, k])
                  + (magnetic["Hy"][i, j, k] - magnetic["Hy"][i, j, k - 1])),
        rel=1e-6,
    )

    # The stride sign is load-bearing: the forward-difference variant must not match.
    forward = before["Dx"] - dtdx * reference_curl(magnetic["Hz"], magnetic["Hy"], 1, 2, +1)
    assert not numpy.allclose(fields.Dx, forward)


def test_plain_path_wraps_periodically_on_every_axis():
    fields = make_fields()
    fields.Dy[0, :, :] = 1.0  # inv_eps == 1, so Ey == Dy.
    dtdx = fields.grid.dt / fields.grid.dx

    stepping.step_B(fields)

    # Bz differences Ey along X: the far face reads Ey[0] by periodic wrap.
    assert numpy.allclose(fields.Bz[-1, :, :], -dtdx)
    assert numpy.allclose(fields.Bz[0, :, :], dtdx)
    assert numpy.all(fields.Bz[1:-1, :, :] == 0)


def test_vacuum_plane_wave_follows_the_discrete_dispersion_relation():
    # One period of a +z plane wave across a 1.6-long cell, resolved twice as
    # finely the second time so the phase-velocity error can be scaled.
    steps = 20
    wavelength = 1.6
    wavenumber = 2.0 * math.pi / wavelength
    velocity_errors = []

    for resolution, transverse in ((20.0, 0.2), (40.0, 0.1)):
        grid = Grid(resolution=resolution, cell_size=(transverse, transverse, wavelength))
        fields = Fields(grid=grid, force_complex_fields=True)
        z = numpy.arange(grid.nz) * grid.dx
        profile = numpy.exp(1j * wavenumber * z).astype(numpy.complex64)
        fields.Dx[...] = profile[None, None, :]

        for _ in range(steps):
            stepping.step_B(fields)
            stepping.update_H(fields)
            stepping.step_D(fields)
            stepping.update_E(fields)

        courant = grid.dt / grid.dx
        # Yee 1-D dispersion relation: sin(w*dt/2) = (dt/dx) * sin(k*dx/2).
        omega = 2.0 / grid.dt * math.asin(courant * math.sin(wavenumber * grid.dx / 2))
        theta = omega * grid.dt
        # The run starts from rest (B at t = -dt/2 is zero), so the k-mode
        # amplitude is the standing wave cos((n + 1/2)*theta) / cos(theta/2).
        amplitude = math.cos((steps + 0.5) * theta) / math.cos(theta / 2)
        assert abs(amplitude) > 0.4  # Guard against a degenerate all-zero comparison.
        assert numpy.allclose(fields.Dx, amplitude * profile[None, None, :], atol=3e-5)

        # Only the Dx/By pair may be excited by this polarization.
        for name in ("Dy", "Dz", "Bx", "Bz"):
            assert numpy.all(getattr(fields, name) == 0)

        velocity_errors.append(abs(omega / wavenumber - 1.0))

    # Numerical phase velocity: 1 - (1 - S^2)*(k*dx)^2/24, i.e. second order in dx.
    assert velocity_errors[0] < 2e-3
    assert velocity_errors[0] / velocity_errors[1] == pytest.approx(4.0, rel=0.05)


def pml_case(grid, pml, component: str, index) -> Fields:  # One-spike PML system, stepped once.
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    getattr(fields, component)[index] = 1.0
    return fields


def test_an_absorbing_axis_still_wraps_on_every_face():
    # A PML is a material, not a boundary condition: MEEP sets the condition in
    # fields::use_bloch (which every k_point, zero or not, applies to all six faces)
    # and structure_chunk::use_pml only grades a conductivity underneath it. So a
    # uniform layer must leave EVERY face wrapping.
    #
    # The auxiliaries start at zero, so fu = -curl*sinv after one step: a zero
    # auxiliary means the stencil read a zero ghost, a nonzero one means it wrapped.
    # Each case spikes a single plane so only one ghost can contribute. Before this
    # rule changed, the three x/y assertions below all read exactly 0 — that metallic
    # termination was the whole residual behind the uniform layer's CPU-MEEP floor
    # (6.31e-04 complex Ez against 4.00e-07 with the wrap restored).
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    pml = PML(grid=grid, thickness=2)

    forward_x = pml_case(grid, pml, "Ey", (0, slice(None), slice(None)))
    stepping.step_B(forward_x, pml)  # Bz differences Ey along X.
    assert numpy.all(forward_x.fu_Bz[-1, :, :] != 0), "the far x face must read the wrapped plane"
    assert numpy.all(forward_x.fu_Bz[0, :, :] != 0)

    forward_y = pml_case(grid, pml, "Ex", (slice(None), 0, slice(None)))
    stepping.step_B(forward_y, pml)  # Bz differences Ex along Y.
    assert numpy.all(forward_y.fu_Bz[:, -1, :] != 0), "the far y face must read the wrapped plane"
    assert numpy.all(forward_y.fu_Bz[:, 0, :] != 0)

    forward_z = pml_case(grid, pml, "Ey", (slice(None), slice(None), 0))
    stepping.step_B(forward_z, pml)  # Bx differences Ey along Z.
    assert numpy.all(forward_z.fu_Bx[:, :, -1] != 0)

    backward_x = pml_case(grid, pml, "Hy", (-1, slice(None), slice(None)))
    stepping.step_D(backward_x, pml)  # Dz differences Hy along X.
    assert numpy.all(backward_x.fu_Dz[0, :, :] != 0), "the near x face must read the wrapped plane"
    assert numpy.all(backward_x.fu_Dz[-1, :, :] != 0)

    backward_y = pml_case(grid, pml, "Hx", (slice(None), -1, slice(None)))
    stepping.step_D(backward_y, pml)  # Dz differences Hx along Y.
    assert numpy.all(backward_y.fu_Dz[:, 0, :] != 0), "the near y face must read the wrapped plane"
    assert numpy.all(backward_y.fu_Dz[:, -1, :] != 0)

    backward_z = pml_case(grid, pml, "Hy", (slice(None), slice(None), -1))
    stepping.step_D(backward_z, pml)  # Dx differences Hy along Z.
    assert numpy.all(backward_z.fu_Dx[:, :, 0] != 0)

    # Negative control on the same machinery: a mirror plane DOES terminate, so the
    # folded axis's far face still reads zero. Without this the test above would also
    # pass on an engine that had lost the ghost rules altogether.
    folded_grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), symmetry=("X",))
    folded_pml = PML(grid=folded_grid, thickness=2)
    folded = pml_case(folded_grid, folded_pml, "Ey", (0, slice(None), slice(None)))
    stepping.step_B(folded, folded_pml)
    assert numpy.all(folded.fu_Bz[-1, :, :] == 0), "a mirrored axis must still terminate"

    plain = Fields(grid=grid, force_complex_fields=True)
    plain.Dy[0, :, :] = 1.0  # inv_eps == 1, so Ey == Dy.
    stepping.step_B(plain)
    assert numpy.all(plain.Bz[-1, :, :] != 0)  # Same cell as forward_x, but wrapped.


def test_pml_B_curl_update_matches_the_hand_computed_recurrence():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=2)
    seed(fields, ("Ex", "Ey", "Ez", "Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz"), 5)
    electric = {name: getattr(fields, name).copy() for name in ("Ex", "Ey", "Ez")}
    before = snapshot(fields, ("Bx", "fu_Bx"))
    dtdx = grid.dt / grid.dx

    stepping.step_B(fields, pml)

    # Two cells, each clear of every ghost: the first has the dsig (Y) grading
    # active and dsigu (Z) flat, the second the other way round, so neither
    # coefficient slot can be swapped for the other unnoticed.
    for i, j, k in ((3, 1, 2), (3, 4, 1)):
        curl = dtdx * ((electric["Ez"][i, j + 1, k] - electric["Ez"][i, j, k])
                       + (electric["Ey"][i, j, k] - electric["Ey"][i, j, k + 1]))
        kms, sinv = pml.kms_y_h[0, j, 0], pml.sinv_y_h[0, j, 0]  # Bx: dsig = Y.
        kms_u, sinv_u = pml.kms_z_h[0, 0, k], pml.sinv_z_h[0, 0, k]  # Bx: dsigu = Z.
        assert kms != kms_u  # The two slots are distinguishable at this cell.
        expected_fu = (kms * before["fu_Bx"][i, j, k] - curl) * sinv
        expected_b = sinv_u * (kms_u * before["Bx"][i, j, k]
                               + expected_fu - before["fu_Bx"][i, j, k])
        assert fields.fu_Bx[i, j, k] == pytest.approx(expected_fu, rel=1e-5)
        assert fields.Bx[i, j, k] == pytest.approx(expected_b, rel=1e-5)

    # The B curl must read half-integer positions; the two sets differ here.
    assert pml.kms_y_h[0, 1, 0] != pml.kms_y[0, 1, 0]
    assert pml.sinv_z_h[0, 0, 1] != pml.sinv_z[0, 0, 1]


def test_pml_D_curl_update_uses_the_integer_coefficient_set():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=2)
    seed(fields, ("Hx", "Hy", "Hz", "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"), 6)
    magnetic = {name: getattr(fields, name).copy() for name in ("Hx", "Hy", "Hz")}
    before = snapshot(fields, ("Dx", "fu_Dx"))
    dtdx = grid.dt / grid.dx

    stepping.step_D(fields, pml)

    for i, j, k in ((3, 1, 2), (3, 4, 1)):
        curl = dtdx * ((magnetic["Hz"][i, j - 1, k] - magnetic["Hz"][i, j, k])
                       + (magnetic["Hy"][i, j, k] - magnetic["Hy"][i, j, k - 1]))
        kms, sinv = pml.kms_y[0, j, 0], pml.sinv_y[0, j, 0]  # Dx: dsig = Y.
        kms_u, sinv_u = pml.kms_z[0, 0, k], pml.sinv_z[0, 0, k]  # Dx: dsigu = Z.
        assert kms != kms_u
        expected_fu = (kms * before["fu_Dx"][i, j, k] - curl) * sinv
        expected_d = sinv_u * (kms_u * before["Dx"][i, j, k]
                               + expected_fu - before["fu_Dx"][i, j, k])
        assert fields.fu_Dx[i, j, k] == pytest.approx(expected_fu, rel=1e-5)
        assert fields.Dx[i, j, k] == pytest.approx(expected_d, rel=1e-5)

    assert pml.kms_y[0, 1, 0] != pml.kms_y_h[0, 1, 0]
    assert pml.sinv_z[0, 0, 1] != pml.sinv_z_h[0, 0, 1]


def test_constitutive_pml_updates_match_the_hand_computed_accumulation():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    fields.set_background_eps(2.25)  # Non-unit eps makes the inv_eps factor observable.
    pml = PML(grid=grid, thickness=2)
    seed(fields, ("Bx", "Dx", "Hx", "Ex", "f_w_Hx", "f_w_Ex"), 9)
    before = snapshot(fields, ("Bx", "Dx", "Hx", "Ex", "f_w_Hx", "f_w_Ex"))

    stepping.update_H(fields, pml)
    stepping.update_E(fields, pml)

    i, j, k = 1, 3, 3  # Inside the lower X absorber, where dsigw = X actually bites.
    kps, kms = pml.kps_x[i, 0, 0], pml.kms_x[i, 0, 0]  # H reads integer positions.
    assert fields.f_w_Hx[i, j, k] == pytest.approx(before["Bx"][i, j, k])
    assert fields.Hx[i, j, k] == pytest.approx(
        before["Hx"][i, j, k] + kps * before["Bx"][i, j, k] - kms * before["f_w_Hx"][i, j, k],
        rel=1e-5,
    )

    kps_h, kms_h = pml.kps_x_h[i, 0, 0], pml.kms_x_h[i, 0, 0]  # E reads half-integer positions.
    electric_source = before["Dx"][i, j, k] * fields.inv_eps[i, j, k]
    assert fields.f_w_Ex[i, j, k] == pytest.approx(electric_source, rel=1e-6)
    assert fields.Ex[i, j, k] == pytest.approx(
        before["Ex"][i, j, k] + kps_h * electric_source - kms_h * before["f_w_Ex"][i, j, k],
        rel=1e-5,
    )
    assert kps_h != kps


def test_constitutive_updates_are_no_ops_without_pml():
    fields = make_fields()
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 4)
    before = snapshot(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"))

    stepping.update_H(fields)
    stepping.update_E(fields, None)

    for name, value in before.items():
        assert numpy.array_equal(getattr(fields, name), value)
    assert fields.Ex is None and fields.Hx is None  # No storage switched on behind the driver's back.


def test_zero_thickness_pml_is_treated_as_no_pml():
    grid = Grid(resolution=10.0, cell_size=(0.6, 0.6, 0.6))
    empty_layer = PML(grid=grid, thickness=0)
    with_layer = Fields(grid=grid, force_complex_fields=True)
    without_layer = Fields(grid=grid, force_complex_fields=True)
    for fields in (with_layer, without_layer):
        seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 17)

    # A layer of no cells absorbs nothing, so the plain path must run — including
    # not demanding the PML storage the driver never allocated.
    stepping.step_B(with_layer, empty_layer)
    stepping.update_H(with_layer, empty_layer)
    stepping.step_D(with_layer, empty_layer)
    stepping.update_E(with_layer, empty_layer)
    stepping.step_B(without_layer)
    stepping.step_D(without_layer)

    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        assert numpy.array_equal(getattr(with_layer, name), getattr(without_layer, name))
    assert with_layer.Ex is None


def test_step_D_leaves_non_owned_cells_untouched_under_symmetry():
    fields = make_fields(cell_size=(0.8, 0.8, 0.8), symmetry=("X", "Y"))
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 3)
    before = snapshot(fields, ("Dx", "Dy", "Dz"))

    stepping.step_D(fields)

    # Yee shifts Dx=(1,0,0), Dy=(0,1,0), Dz=(0,0,1): cell 0 is unowned wherever
    # the shift on the mirrored axis is 0, and on a folded PERIODIC axis at an
    # EVEN count the LAST slot of a shift-1 component is the far ghost past
    # MEEP's big_corner (the fill pass writes it, not the curl).
    assert numpy.array_equal(fields.Dx[:, 0, :], before["Dx"][:, 0, :])
    assert numpy.array_equal(fields.Dx[-1, :, :], before["Dx"][-1, :, :])
    assert numpy.array_equal(fields.Dy[0, :, :], before["Dy"][0, :, :])
    assert numpy.array_equal(fields.Dy[:, -1, :], before["Dy"][:, -1, :])
    assert numpy.array_equal(fields.Dz[0, :, :], before["Dz"][0, :, :])
    assert numpy.array_equal(fields.Dz[:, 0, :], before["Dz"][:, 0, :])
    # Every owned cell moved.
    assert numpy.all(fields.Dx[:-1, 1:, :] != before["Dx"][:-1, 1:, :])
    assert numpy.all(fields.Dy[1:, :-1, :] != before["Dy"][1:, :-1, :])
    assert numpy.all(fields.Dz[1:, 1:, :] != before["Dz"][1:, 1:, :])


def test_symmetry_masking_precedes_the_pml_auxiliary_update():
    # Defect 5.11: without masking the curl first, the auxiliaries at unowned
    # cells integrate a curl built from a ghost the cell does not own.
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), symmetry=("X", "Y"))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=2)
    seed(fields, ("Ex", "Ey", "Ez", "Bx", "By", "Bz"), 7)

    stepping.step_B(fields, pml)

    assert numpy.all(fields.fu_Bx[0, :, :] == 0)  # Bx = (0,1,1): unowned in X.
    assert numpy.all(fields.fu_By[:, 0, :] == 0)  # By = (1,0,1): unowned in Y.
    # Shift-1 slots past the second mirror of a folded periodic even axis are
    # the far ghosts: their curl is masked, so their auxiliaries stay zero too.
    assert numpy.all(fields.fu_Bx[:, -1, :] == 0)  # Bx = (0,1,1): far ghost in Y.
    assert numpy.all(fields.fu_By[-1, :, :] == 0)  # By = (1,0,1): far ghost in X.
    assert numpy.all(fields.fu_Bx[1:, :-1, :] != 0)
    assert numpy.all(fields.fu_Bz[:-1, :-1, :] != 0)  # Bz = (1,1,0): owns the rest.
    assert numpy.all(fields.fu_Bz[-1, :, :] == 0) and numpy.all(fields.fu_Bz[:, -1, :] == 0)

    seed(fields, ("Hx", "Hy", "Hz", "Dx", "Dy", "Dz"), 8)
    stepping.step_D(fields, pml)

    assert numpy.all(fields.fu_Dx[:, 0, :] == 0)
    assert numpy.all(fields.fu_Dy[0, :, :] == 0)
    assert numpy.all(fields.fu_Dz[0, :, :] == 0)
    assert numpy.all(fields.fu_Dz[:, 0, :] == 0)
    assert numpy.all(fields.fu_Dx[-1, :, :] == 0)  # Dx = (1,0,0): far ghost in X.
    assert numpy.all(fields.fu_Dy[:, -1, :] == 0)  # Dy = (0,1,0): far ghost in Y.
    assert numpy.all(fields.fu_Dx[:-1, 1:, :] != 0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_ownership_mask_follows_the_folded_axis_including_z(axis):
    """Cell 0 is unowned on whichever axis is folded, Z included.

    The mask is invisible in the FIELD — the boundary pass overwrites exactly the
    cells it protects — so it has to be read off the PML auxiliary, which nothing
    overwrites. That is also why it is worth pinning per axis: a mask that quietly
    skipped Z would leave every folded-Z field right and every folded-Z absorber
    integrating a curl assembled from a ghost the cell does not own.

    The absorber is put on X only, so the coefficients are the identity on the
    folded axis and the auxiliary's contents are the masked curl and nothing else.
    """
    cell_size = [0.8, 0.8, 0.8]
    grid = Grid(resolution=10.0, cell_size=tuple(cell_size),
                symmetry=(Mirror("XYZ"[axis], -1 if axis == 2 else +1),))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness={"x": {"high": 2}} if axis == 0 else {"x": 2})
    seed(fields, ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Bx", "By", "Bz", "Dx", "Dy", "Dz"), 41)

    stepping.step_B(fields, pml)
    stepping.step_D(fields, pml)

    face = (slice(None),) * axis + (0,)
    for target in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        auxiliary = numpy.asarray(getattr(fields, "fu_" + target))
        unowned = IYEE_SHIFTS[target][axis] == 0
        if unowned:
            assert numpy.all(auxiliary[face] == 0), (
                f"{target} has Yee shift 0 on the folded {'xyz'[axis]} axis, so cell 0 is "
                f"unowned and its auxiliary must have integrated a masked (zero) curl")
        else:
            assert numpy.any(auxiliary[face] != 0), (
                f"{target} owns cell 0 on the folded {'xyz'[axis]} axis and must be stepped there")
        assert numpy.any(auxiliary[(slice(None),) * axis + (1,)] != 0), "the interior must step"


def test_an_absorber_that_disagrees_with_the_grid_about_a_fold_is_refused():
    """A PML that has not learned an axis can fold must not step it anyway.

    Both halves of the fold reach the absorber: it skips the face the mirror plane
    occupies, and it grades cell 0 from the near wall instead of treating it as the
    periodic image of the far one. A table that resolves the fold from a narrower
    set of axes than ``Grid`` does gets BOTH wrong on the axis it missed — an
    absorber written across the mirror plane, and the opposite wall's full
    conductivity on the plane itself — and neither shows as anything but a
    slightly damped, entirely plausible field.

    THE TRIPWIRE HAS FIRED AND BEEN CLEARED. This test used to assert that a real
    ``PML`` on a folded Z believed z wrapped — "the case being guarded" — because
    ``PML.axis_wraps`` and ``_resolve_mirror_faces`` enumerated X and Y while
    ``Grid`` folded all three. Its own note said it would "start passing for the
    right reason, and stop needing the tripwire, the moment the absorber folds
    every axis the grid does". Both now read ``Grid.is_mirrored``, so that moment
    has arrived: the real layer AGREES on every axis and steps, and the guard is
    exercised against a table that lies about the fold, which is the only way it
    can now be reached.
    """
    folded_z = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), symmetry=("Z",))
    fields = Fields(grid=folded_z, force_complex_fields=True)
    fields.enable_pml_storage()

    # The real layer folds z: it reports no wrap, and both curls accept it.
    absorbing_z = PML(grid=folded_z, thickness={"z": {"high": 2}})
    assert not absorbing_z.axis_wraps(2), "a folded axis has no lattice vector"
    assert absorbing_z.axis_faces(2) == (0, 2), "the mirror plane carries no absorber"
    stepping.step_B(fields, absorbing_z)
    stepping.step_D(fields, absorbing_z)

    # A table that resolves the fold from a narrower set of axes than Grid does is
    # what the guard exists for; it can no longer be built by asking PML for one, so
    # it is built by overriding the answer.
    class _WrapsTheFoldedAxis:  # A PML that has not learned z can fold.
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def axis_wraps(self, axis):
            return True

    lying = _WrapsTheFoldedAxis(absorbing_z)
    with pytest.raises(ValueError, match="treats the z axis as wrapping"):
        stepping.step_B(fields, lying)
    with pytest.raises(ValueError, match="treats the z axis as wrapping"):
        stepping.step_D(fields, lying)

    # An absorber on the folded axis's LOWER face is the other half of the same
    # disagreement, and it is the one that eats the reconstructed half outright.
    # Naming that face per side is now refused where it is written, so the run
    # cannot reach a curl with it at all.
    with pytest.raises(ValueError, match="mirror plane, which is a boundary condition"):
        PML(grid=folded_z, thickness={"z": {"low": 2}})

    class _AbsorbsOnThePlane:  # A table that kept the low face of a folded axis.
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def axis_faces(self, axis):
            return (2, 2) if axis == 2 else self._real.axis_faces(axis)

    with pytest.raises(ValueError, match="low z face, which is the mirror plane"):
        stepping.step_B(fields, _AbsorbsOnThePlane(absorbing_z))

    # An X fold is not refused: that absorber already folds X, and this run must
    # keep stepping exactly as it always has.
    folded_x = Grid(resolution=10.0, cell_size=(1.2, 0.8, 0.8), symmetry=("X",))
    x_fields = Fields(grid=folded_x, force_complex_fields=True)
    x_fields.enable_pml_storage()
    stepping.step_B(x_fields, PML(grid=folded_x, thickness=2))

    # Nor is a folded axis that carries no layer at all: sigma is zero on every one
    # of its cells whatever the table believes about its wrap.
    quiet = PML(grid=folded_z, thickness={"x": 2})
    assert not quiet.axis_has_pml(2)
    stepping.step_B(fields, quiet)


def test_neither_curl_sub_step_fills_its_mirror_cells_both_defer_to_the_driver():
    """Both symmetry fills are the driver's, in the same post-injection slot.

    MEEP runs the same three calls in the same order on each side —
    ``step_db(B_stuff)``, ``step_source(B_stuff)``, ``step_boundaries(B_stuff)``
    (step.cpp:67-72), then the D half verbatim at :95-103 — so a curl sub-step that
    repaired its own mirror cells would image every owned cell one injection early.
    ``step_B`` used to do exactly that, on the claim that MEEP's B boundary pass had
    no source injection in between; the pinned contract here is that it does not.
    """
    fields = make_fields(cell_size=(0.8, 0.8, 0.8), symmetry=("X", "Y"))
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 13)

    before_B = fields.Bx.copy()
    stepping.step_B(fields)
    assert numpy.array_equal(fields.Bx[0, :, :], before_B[0, :, :])  # Still unrepaired.

    stepping.fill_symmetry_bc_B(fields)
    assert numpy.array_equal(fields.Bx[0, :, :],
                             get_symmetry_phase("Bx", "x") * fields.Bx[2, :, :])

    before_D = fields.Dy.copy()
    stepping.step_D(fields)
    assert numpy.array_equal(fields.Dy[0, :, :], before_D[0, :, :])  # Still unrepaired.

    stepping.fill_symmetry_bc_D(fields)
    assert numpy.array_equal(fields.Dy[0, :, :],
                             get_symmetry_phase("Dy", "x") * fields.Dy[2, :, :])


def test_fill_symmetry_bc_B_writes_the_documented_mirror_ghosts():
    fields = make_fields(cell_size=(0.8, 0.8, 0.8), symmetry=("X", "Y"))
    seed(fields, ("Bx", "By", "Bz"), 21)
    before = snapshot(fields, ("Bx", "By", "Bz"))

    stepping.fill_symmetry_bc_B(fields)

    assert numpy.array_equal(fields.Bx[0, :, :],
                             get_symmetry_phase("Bx", "x") * before["Bx"][2, :, :])
    assert numpy.array_equal(fields.Bx[1:, :, :], before["Bx"][1:, :, :])
    assert numpy.array_equal(fields.By[:, 0, :],
                             get_symmetry_phase("By", "y") * before["By"][:, 2, :])
    assert numpy.array_equal(fields.By[:, 1:, :], before["By"][:, 1:, :])
    assert numpy.array_equal(fields.Bz, before["Bz"])  # Bz = (1,1,0): owns every cell.


def test_fill_symmetry_bc_D_writes_the_documented_mirror_ghosts():
    fields = make_fields(cell_size=(0.8, 0.8, 0.8), symmetry=("X", "Y"))
    seed(fields, ("Dx", "Dy", "Dz"), 22)
    before = snapshot(fields, ("Dx", "Dy", "Dz"))
    phase_x = get_symmetry_phase("Dz", "x")
    phase_y = get_symmetry_phase("Dz", "y")

    stepping.fill_symmetry_bc_D(fields)

    assert numpy.array_equal(fields.Dx[:, 0, :],
                             get_symmetry_phase("Dx", "y") * before["Dx"][:, 2, :])
    assert numpy.array_equal(fields.Dx[:, 1:, :], before["Dx"][:, 1:, :])
    assert numpy.array_equal(fields.Dy[0, :, :],
                             get_symmetry_phase("Dy", "x") * before["Dy"][2, :, :])
    assert numpy.array_equal(fields.Dy[1:, :, :], before["Dy"][1:, :, :])
    # Dz is unowned on both axes; X is filled first, so the shared corner ends up
    # carrying the doubly mirrored value.
    assert numpy.array_equal(fields.Dz[0, 1:, :], phase_x * before["Dz"][2, 1:, :])
    assert numpy.array_equal(fields.Dz[1:, 0, :], phase_y * before["Dz"][1:, 2, :])
    assert numpy.array_equal(fields.Dz[0, 0, :], phase_x * phase_y * before["Dz"][2, 2, :])
    assert numpy.array_equal(fields.Dz[1:, 1:, :], before["Dz"][1:, 1:, :])


@pytest.mark.parametrize(
    "cell_y, n_full, stored, image_row",
    [(0.8, 8, 6, 4),    # EVEN: big_corner IS the second mirror; ghost at n_full+1 images n_full-1.
     (0.9, 9, 7, 4)],   # ODD: big_corner is half a cell ABOVE it; ghost at n_full+2 images n_full-2.
)
def test_the_far_ghost_images_the_second_mirror_at_both_count_parities(
        cell_y, n_full, stored, image_row):
    """``fill_folded_far_ghosts_*`` reflects about doubled ``n_full``, not about the window top.

    A folded PERIODIC axis stores MEEP's ``num + 1`` slots (vec.cpp:293-296) at
    either parity: the top shift-0 sample sits AT ``big_corner`` and is owned
    (vec.cpp:445-462), while the shift-1 slot half a cell above it is the ghost
    ``connect_the_chunks`` fills by translation + reflection about the SECOND
    MIRROR at doubled ``n_full``.

    Which stored row that image is depends on the parity, and only on it: the
    image of stored row ``j`` at Yee shift ``s`` is ``n_full - j - s + 2``, so the
    ghost's ``(stored - 1, 1)`` lands two rows below the top at an even count and
    THREE at an odd one, where ``big_corner`` overshoots the mirror by half a
    cell. A fixed ``-2`` reflects about the window top instead — right at even,
    a whole cell wrong at odd, and smooth either way.
    """
    fields = make_fields(cell_size=(0.8, cell_y, 0.8), symmetry=("Y",))
    grid = fields.grid
    assert grid.ny_full == n_full and grid.ny == stored
    assert grid.owned_cells(1) == stored - 1  # The extra slot is MEEP's num + 1.
    assert image_row == n_full - stored + 2  # stepping._far_reflect_rows.

    seed(fields, ("Bx", "By", "Bz", "Dx", "Dy", "Dz"), 47)
    before = snapshot(fields, ("Bx", "By", "Bz", "Dx", "Dy", "Dz"))
    stepping.fill_folded_far_ghosts_B(fields)
    stepping.fill_folded_far_ghosts_D(fields)

    for name in ("Bx", "Bz", "Dy"):  # Yee shift 1 on Y: these carry the ghost.
        assert IYEE_SHIFTS[name][1] == 1
        assert numpy.array_equal(
            getattr(fields, name)[:, -1, :],
            get_symmetry_phase(name, "y") * before[name][:, image_row, :]), (
            f"{name}: the far ghost must image stored row {image_row} about the second "
            f"mirror at doubled {n_full}")
        assert numpy.array_equal(getattr(fields, name)[:, :-1, :], before[name][:, :-1, :])
    for name in ("By", "Dx", "Dz"):  # Yee shift 0 on Y: their top slot IS big_corner, owned.
        assert IYEE_SHIFTS[name][1] == 0
        assert numpy.array_equal(getattr(fields, name), before[name])

    # A folded METALLIC axis stores no such slot — the window-top plane is held at
    # zero, so there is nothing to image and the pass must leave every row alone.
    walled = make_fields(cell_size=(0.8, cell_y, 0.8), symmetry=("Y",), boundaries="metallic")
    assert walled.grid.ny == walled.grid.owned_cells(1) == stored - 1
    seed(walled, ("Bx", "Dz"), 48)
    untouched = snapshot(walled, ("Bx", "Dz"))
    stepping.fill_folded_far_ghosts_B(walled)
    stepping.fill_folded_far_ghosts_D(walled)
    assert numpy.array_equal(walled.Bx, untouched["Bx"])
    assert numpy.array_equal(walled.Dz, untouched["Dz"])


def test_shift_stencils_apply_the_documented_ghost_values():
    field = numpy.arange(5, dtype=numpy.float32).reshape(5, 1, 1)

    shifted_down = stepping._shift_down(numpy, field, stepping.AXIS_X, stepping.MIRROR, "Hz",
                                        mirror_phase=+1)
    assert get_symmetry_phase("Hz", "x") == -1  # The parity genuinely flips the sign.
    assert shifted_down[0, 0, 0] == get_symmetry_phase("Hz", "x") * field[2, 0, 0]
    assert numpy.array_equal(shifted_down[1:, 0, 0], field[:-1, 0, 0])

    # The declared phase reaches the ghost: an odd plane inverts the same reflection.
    odd = stepping._shift_down(numpy, field, stepping.AXIS_X, stepping.MIRROR, "Hz",
                               mirror_phase=-1)
    assert odd[0, 0, 0] == -shifted_down[0, 0, 0] != 0
    assert numpy.array_equal(odd[1:, 0, 0], shifted_down[1:, 0, 0])

    shifted_up = stepping._shift_up(numpy, field, stepping.AXIS_X, stepping.MIRROR)
    assert shifted_up[-1, 0, 0] == 0  # Forward differences see a zero past a folded axis.
    assert numpy.array_equal(shifted_up[:-1, 0, 0], field[1:, 0, 0])

    periodic = stepping._shift_down(numpy, field, stepping.AXIS_X, stepping.PERIODIC, "Hz")
    assert periodic[0, 0, 0] == field[-1, 0, 0]

    periodic_up = stepping._shift_up(numpy, field, stepping.AXIS_X, stepping.PERIODIC)
    assert periodic_up[-1, 0, 0] == field[0, 0, 0]

    # A mirror ghost with no declared phase is the two halves of the boundary
    # resolution out of step; it must not quietly fall back on the even mirror.
    with pytest.raises(ValueError, match="carries no mirror plane"):
        stepping._shift_down(numpy, field, stepping.AXIS_X, stepping.MIRROR, "Hz")

    # A perfect conductor has no field beyond it, so BOTH ghosts are zero — no
    # reflection, no wrap, and no dependence on the component's parity.
    #
    # THE FIELD IS OFFSET so its cell 0 is nonzero, and that is the whole point of this
    # block. ``arange(5)`` has ``field[0] == 0``, which makes the metallic far ghost and
    # the PERIODIC wrap agree by coincidence — a mutation that deleted the metallic
    # branch of ``_shift_up`` entirely then survived not only this assertion but the
    # whole CPU-MEEP metallic cross-validation, because in a real run stored cell 0 IS
    # the low wall and is held at exactly zero (``zero_metal_D``), so the wrap does
    # deliver the right value. The branch is what keeps the stencil correct on its own
    # terms rather than through an invariant maintained three functions away, and only
    # a field with a live cell 0 can tell the two apart.
    offset = field + 1.0
    assert offset[0, 0, 0] != 0 and offset[-1, 0, 0] != 0
    metallic_down = stepping._shift_down(numpy, offset, stepping.AXIS_X, stepping.METALLIC, "Hz")
    assert metallic_down[0, 0, 0] == 0
    assert numpy.array_equal(metallic_down[1:, 0, 0], offset[:-1, 0, 0])
    metallic_up = stepping._shift_up(numpy, offset, stepping.AXIS_X, stepping.METALLIC)
    assert metallic_up[-1, 0, 0] == 0
    assert numpy.array_equal(metallic_up[:-1, 0, 0], offset[1:, 0, 0])
    # The controls: on the same field the two periodic ghosts are BOTH nonzero, so
    # neither metallic assertion above can be satisfied by a wrap.
    assert stepping._shift_up(numpy, offset, stepping.AXIS_X, stepping.PERIODIC)[-1, 0, 0] != 0
    assert stepping._shift_down(
        numpy, offset, stepping.AXIS_X, stepping.PERIODIC, "Hz")[0, 0, 0] != 0

    # There is no absorbing ghost rule to reach for — a PML never produces one — and
    # an unknown name must raise rather than fall through to some default.
    for shift in (lambda: stepping._shift_up(numpy, field, stepping.AXIS_X, "absorbing"),
                  lambda: stepping._shift_down(numpy, field, stepping.AXIS_X, "absorbing", "Hz")):
        with pytest.raises(ValueError, match="unknown boundary condition"):
            shift()


@pytest.mark.parametrize("complex_fields", [True, False])
def test_stepping_preserves_the_storage_dtype(complex_fields):
    grid = Grid(resolution=10.0, cell_size=(0.6, 0.6, 0.6))
    fields = Fields(grid=grid, force_complex_fields=complex_fields)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=1)
    seed(fields, ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Dx", "Dy", "Dz", "Bx", "By", "Bz"), 31)
    expected = numpy.complex64 if complex_fields else numpy.float32

    stepping.step_B(fields, pml)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    stepping.update_E(fields, pml)

    for name in ("Bx", "Dx", "Ex", "Hx", "fu_Bx", "fu_Dx", "f_w_Ex", "f_w_Hx"):
        assert getattr(fields, name).dtype == expected
    assert numpy.all(numpy.isfinite(numpy.asarray(fields.Ex).view(numpy.float32)))


def test_pml_stepping_without_storage_fails_loudly():
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    pml = PML(grid=grid, thickness=2)

    for sub_step in (stepping.step_B, stepping.step_D, stepping.update_H, stepping.update_E):
        with pytest.raises(RuntimeError, match="enable_pml_storage"):
            sub_step(fields, pml)


def test_unknown_component_axis_and_boundary_names_raise():
    fields = make_fields()

    with pytest.raises(ValueError, match="E and H"):
        stepping._read_component(fields, "Dx")
    with pytest.raises(ValueError, match="PML axis"):
        stepping._coefficient_suffix("w", half_integer=False)
    # Z is a real mirror axis now, so the parity resolves; what must not be silent is
    # a MIRROR boundary with no plane behind it.
    assert stepping._symmetry_phase("Hz", stepping.AXIS_Z, +1) == +1
    with pytest.raises(ValueError, match="carries no mirror plane"):
        stepping._symmetry_phase("Hz", stepping.AXIS_Z, None)
    with pytest.raises(ValueError, match="boundary condition"):
        stepping._shift_up(numpy, fields.Dx, stepping.AXIS_X, "sideways")
    with pytest.raises(ValueError, match="boundary condition"):
        stepping._shift_down(numpy, fields.Dx, stepping.AXIS_X, "sideways", "Hz")


# --- Bloch-periodic boundaries ---------------------------------------------------

BLOCH_CELL = (0.15, 0.15, 1.0)  # Thin in x/y, one length unit of wrap in z.
BLOCH_RESOLUTION = 20.0


def bloch_grid(k_z: float, cell_size=BLOCH_CELL, resolution=BLOCH_RESOLUTION) -> Grid:
    return Grid(resolution=resolution, cell_size=cell_size, k_point=(0.0, 0.0, k_z))


def yee_plane_wave(grid: Grid, k_z: float):
    """Exact discrete +z eigenmode of the Yee scheme on a Bloch cell at ``k_z``.

    The continuum wavenumber is ``beta = 2*pi*k_z`` — MEEP's k is in units of
    2*pi/distance — and the mode is an exact solution of the *discrete* update, not
    a sampled continuum one, provided omega satisfies the 1-D Yee dispersion
    relation ``sin(omega*dt/2) = (dt/dz) * sin(beta*dz/2)``.

    Substituting ``Dx[k] = A exp(i(beta*k*dz - omega*n*dt))`` and
    ``By[k] = C exp(i(beta*(k+1/2)*dz - omega*(n+1/2)*dt))`` into this engine's two
    curl sub-steps gives ``C*sin(omega*dt/2) = (dt/dz)*A*sin(beta*dz/2)`` from
    step_B and the same relation back from step_D; with the dispersion relation
    holding, both collapse to C = A. That is the unit vacuum impedance, and it is
    why the seed below carries no amplitude ratio.

    B is stored half a step behind D, so the seeded By is the n = -1/2 sample:
    ``exp(+i*omega*dt/2)`` relative to the D at n = 0.

    Returns:
        (omega, Dx profile along z, By profile along z), both complex64.
    """
    beta = 2.0 * math.pi * k_z
    courant = grid.dt / grid.dx
    omega = (2.0 / grid.dt) * math.asin(courant * math.sin(beta * grid.dx / 2.0))
    index = numpy.arange(grid.nz)
    electric = numpy.exp(1j * beta * index * grid.dx)
    magnetic = numpy.exp(1j * beta * (index + 0.5) * grid.dx) * cmath.exp(0.5j * omega * grid.dt)
    return omega, electric.astype(numpy.complex64), magnetic.astype(numpy.complex64)


def seed_plane_wave(fields: Fields, electric, magnetic) -> None:  # Broadcast the z profiles over x and y.
    fields.Dx[...] = electric[None, None, :]
    fields.By[...] = magnetic[None, None, :]


def advance(fields: Fields, steps: int) -> None:  # Vacuum leapfrog, no PML and no sources.
    for _ in range(steps):
        stepping.step_B(fields)
        stepping.update_H(fields)
        stepping.step_D(fields)
        stepping.update_E(fields)


def relative_l2(candidate, reference) -> float:  # Complex relative L2; phase errors count.
    candidate = numpy.asarray(candidate, dtype=numpy.complex128).ravel()
    reference = numpy.asarray(reference, dtype=numpy.complex128).ravel()
    return float(numpy.linalg.norm(candidate - reference) / numpy.linalg.norm(reference))


def test_bloch_phase_reproduces_meeps_eikna_expression():
    # MEEP boundaries.cpp: eikna[d] = exp(I * kk * ((2*pi/a) * gv.num_direction(d))),
    # i.e. exp(i*2*pi*k*L). Dropping the 2*pi is the classic error and leaves a
    # perfectly smooth field, so the factor is pinned against the bare exponential.
    grid = Grid(resolution=20.0, cell_size=(1.0, 2.0, 0.5), k_point=(0.3, -0.15, 0.0))
    assert grid.bloch_phase(0) == pytest.approx(cmath.exp(2j * math.pi * 0.3 * 1.0))
    assert grid.bloch_phase(1) == pytest.approx(cmath.exp(2j * math.pi * -0.15 * 2.0))
    assert grid.bloch_phase(2) is None  # k_z = 0: no phase object at all, so no multiply.
    assert grid.has_bloch is True
    assert abs(grid.bloch_phase(0) - cmath.exp(1j * 0.3 * 1.0)) > 1.0  # The 2*pi is load-bearing.

    plain = Grid(resolution=20.0, cell_size=(1.0, 2.0, 0.5))
    assert plain.has_bloch is False
    assert plain.bloch_phases == (None, None, None)

    # MEEP pins the Brillouin-zone edge (k*num == 0.5*a) to exactly -1 rather than
    # letting exp() leave a 1.2e-16 imaginary part; the same test is reproduced here.
    edge = Grid(resolution=20.0, cell_size=(2.0, 1.0, 1.0), k_point=(0.25, 0.0, 0.0))
    assert edge.bloch_phase(0) == complex(-1.0, 0.0)
    assert edge.bloch_phase(0).imag == 0.0
    assert cmath.exp(1j * math.pi).imag != 0.0  # The exactness genuinely had to be pinned.

    with pytest.raises(ValueError, match="finite"):
        Grid(resolution=20.0, cell_size=(1.0, 1.0, 1.0), k_point=(float("nan"), 0.0, 0.0))
    with pytest.raises(ValueError, match="must be real"):  # MEEP's evanescent complex k.
        Grid(resolution=20.0, cell_size=(1.0, 1.0, 1.0), k_point=(0.3 + 0.1j, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"k_point must be"):
        Grid(resolution=20.0, cell_size=(1.0, 1.0, 1.0), k_point=(0.1, 0.2))
    with pytest.raises(ValueError, match="axis must be"):
        plain.bloch_phase(3)


def test_bloch_shift_stencils_phase_the_wrapped_plane_in_opposite_directions():
    # MEEP locate_point_in_user_volume translates a point UP by a lattice vector and
    # multiplies by conj(eikna), so the up-shift carries eikna and the down-shift its
    # conjugate. Taking one factor in both directions is invisible to any magnitude.
    field = (numpy.arange(1, 6, dtype=numpy.complex64) * (1 + 0.5j)).reshape(5, 1, 1)
    phase = cmath.exp(2j * math.pi * 0.3)

    up = stepping._shift_up(numpy, field, stepping.AXIS_X, stepping.PERIODIC, phase)
    assert up[-1, 0, 0] == pytest.approx(complex(field[0, 0, 0]) * phase)
    assert numpy.array_equal(up[:-1, 0, 0], field[1:, 0, 0])

    down = stepping._shift_down(numpy, field, stepping.AXIS_X, stepping.PERIODIC, "Hz", phase)
    assert down[0, 0, 0] == pytest.approx(complex(field[-1, 0, 0]) * phase.conjugate())
    assert numpy.array_equal(down[1:, 0, 0], field[:-1, 0, 0])
    assert abs(down[0, 0, 0] - complex(field[-1, 0, 0]) * phase) > 1.0  # Conjugation matters.

    # The wrap phase is applied to a copy; the caller's array is never touched.
    assert field[0, 0, 0] == 1 + 0.5j

    real_field = numpy.arange(5, dtype=numpy.float32).reshape(5, 1, 1)
    with pytest.raises(ValueError, match="complex fields"):
        stepping._shift_up(numpy, real_field, stepping.AXIS_X, stepping.PERIODIC, phase)


def test_bloch_plane_wave_propagates_undistorted_and_k_zero_does_not():
    # A plane wave whose wavevector matches the Bloch boundary is an exact eigenmode
    # of the discrete update: after N steps it must be itself times exp(-i*omega*N*dt),
    # a pure unit-magnitude phase. beta*Lz = 2*pi*0.3 is not a multiple of 2*pi, so the
    # same seed is NOT periodic and cannot survive a plain periodic wrap.
    k_z = 0.3
    steps = 200
    grid = bloch_grid(k_z)
    omega, electric, magnetic = yee_plane_wave(grid, k_z)
    fields = Fields(grid=grid, force_complex_fields=True)
    seed_plane_wave(fields, electric, magnetic)
    initial = fields.Dx.copy()

    advance(fields, steps)

    expected = initial * cmath.exp(-1j * omega * steps * grid.dt)
    error = relative_l2(fields.Dx, expected)
    # Measured 3.2e-07 — complex64 round-off over 200 steps, not a physical deviation.
    assert error < 1e-5, f"Bloch plane wave distorted by relative L2 {error:.3e} over {steps} steps."

    # The phase actually advanced — otherwise "unchanged" would pass as "undistorted".
    assert abs(cmath.exp(-1j * omega * steps * grid.dt) - 1.0) > 1.0
    # Undistorted means the envelope stayed flat, not merely that the L2 is small.
    magnitudes = numpy.abs(fields.Dx)
    assert float(magnitudes.max() - magnitudes.min()) < 1e-5  # Measured 7.2e-07.
    # Only the Dx/By pair may be excited by this polarization.
    for name in ("Dy", "Dz", "Bx", "Bz"):
        assert numpy.all(getattr(fields, name) == 0)

    # Control 1: plain periodicity. The wrap injects a step discontinuity of
    # |exp(i*2*pi*0.3) - 1| per crossing, and the mode is no longer an eigenmode.
    plain = Fields(grid=Grid(resolution=BLOCH_RESOLUTION, cell_size=BLOCH_CELL),
                   force_complex_fields=True)
    seed_plane_wave(plain, electric, magnetic)
    advance(plain, steps)
    plain_error = relative_l2(plain.Dx, expected)
    assert plain_error > 0.1, (  # Measured 1.98.
        f"A plain periodic wrap reproduced the Bloch result to {plain_error:.3e}; the boundary "
        f"phase is not being applied and the test is measuring nothing."
    )

    # Control 2: the conjugate wavevector. Magnitudes stay plausible under a sign
    # error, so this is the control that a magnitude-only comparison cannot supply.
    flipped = Fields(grid=bloch_grid(-k_z), force_complex_fields=True)
    seed_plane_wave(flipped, electric, magnetic)
    advance(flipped, steps)
    flipped_error = relative_l2(flipped.Dx, expected)
    assert flipped_error > 0.1, (  # Measured 4.3e-01.
        f"Reversing the sign of k_point changed the result by only {flipped_error:.3e}; the wrap "
        f"phase is being applied in the same direction on both faces."
    )

    # Control 3: why the comparisons above are on the complex field. The same two
    # broken runs, compared on |Dx| alone, sit at 1.6e-01 and 3.6e-01 — the first
    # would slip under a loose magnitude bound, and neither separates a sign error
    # from an amplitude one. Phase is the whole content of a Bloch boundary.
    assert relative_l2(numpy.abs(plain.Dx), numpy.abs(expected)) < plain_error / 5.0


def test_bloch_plane_wave_survives_a_half_brillouin_zone_edge_and_a_negative_k():
    # k*L = 1/2 takes MEEP's exactly-pinned -1 branch and k < 0 takes the conjugate
    # of the general one; both must still carry an eigenmode without distortion.
    steps = 120
    for k_z in (0.5, -0.35):  # Lz = 1, so k_z = 0.5 is the zone edge.
        grid = bloch_grid(k_z)
        omega, electric, magnetic = yee_plane_wave(grid, k_z)
        fields = Fields(grid=grid, force_complex_fields=True)
        seed_plane_wave(fields, electric, magnetic)
        initial = fields.Dx.copy()

        advance(fields, steps)

        expected = initial * cmath.exp(-1j * omega * steps * grid.dt)
        error = relative_l2(fields.Dx, expected)
        # Measured 3.0e-07 at the zone edge and 2.8e-07 at k_z = -0.35.
        assert error < 1e-5, f"k_z={k_z}: plane wave distorted by relative L2 {error:.3e}."


def test_zero_k_point_steps_bit_identically_to_a_plain_periodic_grid():
    # The Bloch path must be *skipped*, not exercised with a unit phase: a run at
    # k = 0 has to reproduce the pre-Bloch engine bit for bit, which is what keeps
    # the recorded 1.7e-7 no-PML parity floor a statement about the same code.
    explicit = Fields(grid=Grid(resolution=10.0, cell_size=(0.6, 0.6, 0.6), k_point=(0.0, 0.0, 0.0)),
                      force_complex_fields=True)
    implicit = Fields(grid=Grid(resolution=10.0, cell_size=(0.6, 0.6, 0.6)), force_complex_fields=True)
    for fields in (explicit, implicit):
        seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 41)
        advance(fields, 25)

    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        left = numpy.asarray(getattr(explicit, name))
        right = numpy.asarray(getattr(implicit, name))
        assert left.tobytes() == right.tobytes(), f"{name} is not bit-identical at k_point = 0."
        assert float(numpy.abs(left).max()) > 0.0  # Not vacuously equal to an empty run.


def test_bloch_refuses_the_boundary_that_cannot_carry_a_phase():
    # A mirror plane REFLECTS rather than repeating, so a k component on it has
    # nothing to describe and must not be silently dropped. That is now the only
    # boundary refused: an absorbing axis still wraps (MEEP's use_bloch keeps it
    # Periodic and grades the layer underneath), and stepping it reproduces CPU MEEP
    # at 2.82e-07 — see test_a_bloch_phase_is_applied_on_an_absorbing_axis_rather
    # _than_refused, which replaced the assertions that used to live here.
    grid = bloch_grid(0.3, cell_size=(0.8, 0.8, 0.8), resolution=10.0)

    # Grid refuses symmetry + k_point outright, so the guard is exercised on the
    # resolver directly — it is the last line of defence if that check is relaxed.
    with pytest.raises(ValueError, match="mirror"):  # grid carries k on z, so mirror z.
        stepping._bloch_phases(grid, (stepping.PERIODIC, stepping.PERIODIC, stepping.MIRROR),
                               numpy.zeros((2, 2, 2), dtype=numpy.complex64))

    # And a Bloch phase still needs complex storage, whatever the boundaries are.
    real_fields = Fields(grid=Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8),
                                   k_point=(0.3, 0.0, 0.0)), force_complex_fields=False)
    with pytest.raises(ValueError, match="complex"):
        stepping.step_B(real_fields, None)

    real_fields = Fields(grid=grid, force_complex_fields=False)
    with pytest.raises(ValueError, match="complex fields"):
        stepping.step_B(real_fields)

    with pytest.raises(ValueError, match="mirror symmetry"):
        Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), symmetry=("X",), k_point=(0.3, 0.0, 0.0))
    with pytest.raises(ValueError, match="mirror symmetry"):
        Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), symmetry=("Y",), k_point=(0.0, -0.3, 0.0))
    # A mirror on one axis and a Bloch phase on another is physically consistent and
    # the curl kernels handle it; only the driver's readback refuses it.
    Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), symmetry=("X",), k_point=(0.0, 0.0, 0.3))


def test_mirror_plane_needs_a_cell_to_reflect_from():
    # A mirrored axis holding only the plane cell and one neighbour has no cell 2
    # to reflect from; that is a broken configuration, not a silent no-op. The
    # metallic fold keeps MEEP's halved two cells (a periodic fold at N = 2 now
    # stores a third — the second-mirror plane — and reflects fine).
    fields = make_fields(cell_size=(0.2, 0.6, 0.6), symmetry=("X",),
                         boundaries="metallic")
    assert fields.grid.nx == 2
    # Asked of the passes that actually reflect. `step_B` alone no longer reaches the
    # reflection at all: its differences are FORWARD, so it never reads the near-face
    # ghost, and its own fill moved to the driver's post-injection slot.
    with pytest.raises(ValueError, match="reflect"):
        stepping.fill_symmetry_bc_B(fields)
    with pytest.raises(ValueError, match="reflect"):
        stepping.step_D(fields)


# --- Per-axis PML: an axis with no absorber must stay genuinely periodic ----------


def advance_with_pml(fields: Fields, pml, steps: int) -> None:  # Leapfrog through the PML kernels.
    for _ in range(steps):
        stepping.step_B(fields, pml)
        stepping.update_H(fields, pml)
        stepping.step_D(fields, pml)
        stepping.update_E(fields, pml)


def boundary_kinds(cell_size, thickness, symmetry=()) -> tuple:  # Resolved ghost rule per axis.
    grid = Grid(resolution=10.0, cell_size=cell_size, symmetry=symmetry)
    layer = None if thickness is None else PML(grid=grid, thickness=thickness)
    return stepping._boundary_kinds(grid, layer if stepping._pml_is_active(layer) else None)


def test_boundary_kinds_are_the_symmetry_and_nothing_else():
    # The rule, in one table: mirror where the axis is folded, periodic everywhere
    # else. No PML spelling changes any entry, because MEEP's boundary comes from
    # fields::use_bloch and its absorber from structure_chunk::use_pml, and the two
    # do not talk to each other. The uniform layer (thickness 4) is the entry that
    # moved: it used to read (METALLIC, METALLIC, PERIODIC).
    cell = (2.0, 2.0, 2.0)
    P, R = stepping.PERIODIC, stepping.MIRROR
    for thickness in (None, 0, 4, {"z": 4}, {"x": 4}, {"x": 4, "y": 4},
                      {"x": (4, 0)}, {"z": (4, 0)}, {"x": 4, "y": 4, "z": 4}, {"z": (4, 2)}):
        assert boundary_kinds(cell, thickness) == (P, P, P), thickness
    for thickness in (None, 4, {"z": 4}, {"x": {"high": 4}}):
        assert boundary_kinds(cell, thickness, symmetry=("X",)) == (R, P, P), thickness
        assert boundary_kinds(cell, thickness, symmetry=("X", "Y")) == (R, R, P), thickness
    # And the resolver genuinely ignores its layer argument rather than happening to
    # agree with it: the same grid resolves identically with the PML passed or not.
    grid = Grid(resolution=10.0, cell_size=cell, symmetry=("Y",))
    assert (stepping._boundary_kinds(grid, PML(grid=grid, thickness={"x": 4, "z": (4, 0)}))
            == stepping._boundary_kinds(grid, None) == (P, R, P))


def test_an_axis_without_a_layer_keeps_the_periodic_wrap_while_its_neighbour_absorbs():
    # The value-level statement of the crux, with no reference run needed: on a truly
    # periodic axis with no absorber the update commutes with a cyclic shift along it,
    # so rolling the initial state one cell in x and rolling the answer back must
    # reproduce the unrolled run. Putting a layer on x breaks that immediately — the
    # graded conductivity lives at fixed indices, not at a fixed offset — which is
    # asserted as the positive control below.
    components = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")

    def shifted_run(thickness, shift):
        grid = Grid(resolution=10.0, cell_size=(1.2, 1.2, 1.2))
        fields = Fields(grid=grid, force_complex_fields=True)
        fields.enable_pml_storage()
        seed(fields, components, 2027)
        for name in components:  # Roll the whole initial state, then step.
            array = getattr(fields, name)
            array[...] = numpy.roll(array, shift, axis=0)
        advance_with_pml(fields, PML(grid=grid, thickness=thickness), 12)
        return {name: numpy.roll(numpy.asarray(getattr(fields, name)), -shift, axis=0)
                for name in components}

    absorbing_z_only = (shifted_run({"z": 3}, 0), shifted_run({"z": 3}, 1))
    scale = max(float(numpy.abs(absorbing_z_only[0][name]).max()) for name in components)
    assert scale > 0.0, "an all-zero run would make the invariance vacuous"
    for name in components:
        numpy.testing.assert_allclose(
            absorbing_z_only[1][name], absorbing_z_only[0][name], rtol=0.0, atol=1e-6 * scale,
            err_msg=f"{name}: x is not translation invariant, so it is not periodic")

    # Positive control: a uniform layer grades x, and the same comparison must then
    # fail by a wide margin. Without this the test would also pass on an engine that
    # had quietly frozen x altogether.
    absorbing_x = (shifted_run(3, 0), shifted_run(3, 1))
    worst = max(float(numpy.abs(absorbing_x[1][name] - absorbing_x[0][name]).max())
                for name in components)
    assert worst > 1e-2 * scale, (
        f"an absorber on x must break translation invariance, but the shift moved it by "
        f"only {worst:.2e}")


def test_a_bloch_phase_survives_on_the_axis_the_absorber_leaves_alone():
    # Bloch + PML in one run, checked against an exact unfolding rather than a
    # tolerance: at k*L = 1/2 the wrap factor is exactly -1, so a cell of length L
    # with k = 0.5/L is identical to a cell of length 2L at k = 0 seeded with the
    # antiperiodic extension [P, -P]. Both runs carry the same z absorber, so this
    # says the phased x wrap is applied, and applied correctly, while z absorbs.
    components = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")
    resolution, nx, nz_cells = 10.0, 12, 12
    generator = numpy.random.default_rng(619)
    profile = {
        name: (generator.standard_normal((nx, 1, nz_cells))
               + 1j * generator.standard_normal((nx, 1, nz_cells))).astype(numpy.complex64)
        for name in components
    }

    def run(cell_x, k_x, seeds):
        grid = Grid(resolution=resolution, cell_size=(cell_x, 0.1, nz_cells / resolution),
                    k_point=(k_x, 0.0, 0.0))
        fields = Fields(grid=grid, force_complex_fields=True)
        fields.enable_pml_storage()
        for name in components:
            getattr(fields, name)[...] = seeds[name]
        advance_with_pml(fields, PML(grid=grid, thickness={"z": 3}), 20)
        return {name: numpy.asarray(getattr(fields, name)) for name in components}

    folded = run(nx / resolution, 0.5 / (nx / resolution), profile)
    doubled = run(2 * nx / resolution, 0.0,
                  {name: numpy.concatenate([values, -values], axis=0) for name, values in profile.items()})
    scale = max(float(numpy.abs(folded[name]).max()) for name in components)
    assert scale > 0.0, "an all-zero run would make the unfolding vacuous"
    for name in components:
        numpy.testing.assert_allclose(
            folded[name], doubled[name][:nx], rtol=0.0, atol=1e-5 * scale,
            err_msg=f"{name}: the Bloch-phased x wrap does not unfold onto the doubled cell")
    # Control: the same folded run at k = 0 — an engine that accepted the k_point and
    # dropped it once a PML existed — must NOT unfold onto the antiperiodic cell.
    unphased = run(nx / resolution, 0.0, profile)
    worst = max(float(numpy.abs(unphased[name] - doubled[name][:nx]).max()) for name in components)
    assert worst > 1e-2 * scale, f"dropping the Bloch phase must be visible, moved only {worst:.2e}"


def test_a_layer_that_absorbs_nowhere_steps_bit_identically_to_no_layer():
    # Degenerate case: zero on every face is not "a PML with sigma 0" — the split-field
    # recurrence would still run and cost round-off — it is no PML at all.
    components = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")
    for thickness in (0, {"z": 0}, {"x": (0, 0), "y": 0}):
        grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
        empty = Fields(grid=grid, force_complex_fields=True)
        empty.enable_pml_storage()
        seed(empty, components, 88)
        plain = Fields(grid=grid, force_complex_fields=True)
        plain.enable_pml_storage()
        seed(plain, components, 88)
        advance_with_pml(empty, PML(grid=grid, thickness=thickness), 10)
        advance_with_pml(plain, None, 10)
        for name in components:
            left = numpy.asarray(getattr(empty, name))
            right = numpy.asarray(getattr(plain, name))
            assert left.tobytes() == right.tobytes(), f"{name} differs for thickness={thickness!r}"
            assert float(numpy.abs(left).max()) > 0.0  # Not vacuously equal to an empty run.


def test_a_bloch_phase_is_applied_on_an_absorbing_axis_rather_than_refused():
    """An axis that wraps carries its phase whether or not a layer absorbs on it.

    This test used to assert the opposite. The stance was that an absorber terminates
    the axis, so the field a lattice vector up is whatever the layer left rather than
    exp(i*2*pi*k*L) times the field here, and stepping the pairing raised. MEEP takes
    the other view: `use_bloch` makes every direction Periodic as soon as any k_point
    is given, and a PML is graded material UNDERNEATH that wrap. Measuring settled it —
    the refused configuration reproduces CPU MEEP's complex Ez at 2.82e-07, against
    2.63e-07 with the absorber removed, and the refusal was costing seven scripts of
    MEEP's own corpus (pw-source.py among them).

    What is asserted here is that the phase is really APPLIED and not quietly dropped:
    a dropped phase is the failure the old refusal was written to prevent, and it still
    would be, it just needs catching by a value rather than by an exception.
    """
    for k_point, spec, axis in (
        ((0.3, 0.0, 0.0), {"x": 2}, 0),
        ((0.3, 0.0, 0.0), {"x": (2, 0)}, 0),
        ((0.0, 0.0, 0.3), {"z": 2}, 2),
        ((0.0, 0.0, 0.3), {"z": (0, 2)}, 2),
    ):
        grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8), k_point=k_point)
        fields = Fields(grid=grid, force_complex_fields=True)
        fields.enable_pml_storage()
        pml = PML(grid=grid, thickness=spec)
        stepping.step_B(fields, pml)      # steps rather than raising
        stepping.step_D(fields, pml)
        phases = stepping._bloch_phases(
            grid, stepping._boundary_kinds(grid, pml),
            fields.get_component("Bx"), pml,
        )
        assert phases[axis] is not None, (
            f"k={k_point} with PML {spec}: the phase on axis {axis} was dropped. A "
            f"dropped Bloch phase runs a plain absorbing cell while the caller believes "
            f"they set a band-structure boundary — the exact outcome the old refusal "
            f"existed to prevent."
        )
        assert abs(abs(complex(phases[axis])) - 1.0) < 1e-6, (
            f"a Bloch wrap factor must be a unit phasor; got {phases[axis]}"
        )


# ---------------------------------------------------------------------------
# Dispersion: update_E over D - sum P, and the update_P sub-step
# ---------------------------------------------------------------------------


def _dispersive(fields, susceptibility, sigma):  # Attach one susceptibility to a Fields.
    state = PolarizationState(susceptibility, sigma, fields.grid, numpy.complex64)
    if state.driven():
        fields.enable_field_storage()
    fields.polarizations.append(state)
    return state


def test_update_E_stores_the_constitutive_product_minus_the_polarization():
    fields = make_fields()
    state = _dispersive(fields, Susceptibility(1.0, 0.1, LORENTZIAN), 0.5)
    fields.eps.fill(4.0)
    fields.inv_eps.fill(0.25)
    fields.Dz.fill(8.0)
    state.P["Ez"].fill(2.0)
    stepping.update_E(fields, None)
    numpy.testing.assert_allclose(fields.Ez, numpy.full(fields.grid.shape, 1.5, dtype=numpy.complex64))
    # And the components nothing drives still get the plain product.
    fields.Dx.fill(8.0)
    stepping.update_E(fields, None)
    numpy.testing.assert_allclose(fields.Ex, numpy.full(fields.grid.shape, 2.0, dtype=numpy.complex64))


def test_update_E_is_still_a_no_op_with_neither_pml_nor_a_susceptibility():
    fields = make_fields()
    stepping.update_E(fields, None)
    assert fields.Ex is None and not fields.stores_E


def test_update_E_refuses_to_run_without_the_storage_it_writes_into():
    fields = make_fields()
    state = PolarizationState(Susceptibility(1.0, 0.1, LORENTZIAN), 0.5, fields.grid, numpy.complex64)
    fields.polarizations.append(state)
    fields._stored_E = True  # Claim the mode without allocating: stepping must not paper over it.
    with pytest.raises(RuntimeError, match="enable_field_storage"):
        stepping.update_E(fields, None)


def test_update_P_advances_the_polarization_from_the_drive_field():
    fields = make_fields()
    state = _dispersive(fields, Susceptibility(1.1, 0.05, LORENTZIAN), 0.6)
    fields.Ez.fill(1.0)
    c_now, c_prev, c_drive = state.susceptibility.coefficients(fields.grid.dt)
    stepping.update_P(fields, None)
    expected = c_drive * 0.6 * 1.0
    numpy.testing.assert_allclose(state.P["Ez"], expected, rtol=1e-6)
    stepping.update_P(fields, None)
    numpy.testing.assert_allclose(
        state.P["Ez"], c_now * expected + c_prev * 0.0 + c_drive * 0.6, rtol=1e-6
    )


def test_update_P_reads_the_pml_auxiliary_and_not_the_absorbed_field():
    """Inside an absorber the stored E has already taken the PML accumulation.

    This is the one distinction no interior measurement can see, so it is pinned
    directly: with PML storage live, ``update_P`` must consume ``f_w`` — the
    un-absorbed constitutive product — and ignore whatever the stored E holds.
    """
    fields = make_fields()
    state = _dispersive(fields, Susceptibility(1.1, 0.05, LORENTZIAN), 1.0)
    fields.enable_pml_storage()
    fields.Ez.fill(100.0)   # The absorbed field: must NOT drive the polarization.
    fields.f_w_Ez.fill(1.0)  # MEEP's w.
    stepping.update_P(fields, None)
    _, _, c_drive = state.susceptibility.coefficients(fields.grid.dt)
    numpy.testing.assert_allclose(state.P["Ez"], c_drive * 1.0, rtol=1e-6)


def test_update_P_does_nothing_without_a_susceptibility():
    fields = make_fields()
    stepping.update_P(fields, None)  # Must not raise, and must not allocate.
    assert fields.polarizations == []


def test_update_P_needs_no_boundary_pass_on_a_folded_grid():
    """P is element-wise, so a mirror plane reaches it only through its drive field.

    Cell 0 of a folded axis is not owned by the curl loop, but ``update_P`` runs over
    the whole stored array with no mask: its drive field there was already repaired
    by ``fill_symmetry_bc_D`` before ``update_E``, so the polarization follows by
    induction. If that were wrong, the folded-vs-full equivalence would stop being
    exact — which is what the driver-level test asserts.
    """
    fields = make_fields(cell_size=(0.6, 0.6, 0.6), symmetry=("X",))
    state = _dispersive(fields, Susceptibility(1.0, 0.1, LORENTZIAN), 1.0)
    fields.Ez[0, :, :] = 5.0
    fields.Ez[1:, :, :] = 0.0
    stepping.update_P(fields, None)
    _, _, c_drive = state.susceptibility.coefficients(fields.grid.dt)
    assert complex(state.P["Ez"][0, 0, 0]) == pytest.approx(c_drive * 5.0, rel=1e-6), (
        "cell 0 of a mirrored axis must be updated, not masked out"
    )
    assert complex(state.P["Ez"][1, 0, 0]) == 0.0


# ---------------------------------------------------------------------------
# Conductivity in step_D
# ---------------------------------------------------------------------------


def test_step_D_applies_meeps_conductive_curl_update():
    """MEEP step_generic.cpp:87-97, ``f = ((1 - dt/2 cnd) f - dtdx curl) cndinv``.

    The loss acts on D, not on E: MEEP's sigma_D appears as ``sigma_D * D`` in the
    equations, so it differs from the textbook electric conductivity by a factor of
    epsilon. Checked against a hand-evaluated single step with the curl frozen at
    zero, so only the conductive part is under test.
    """
    fields = make_fields()
    fields.set_d_conductivity(numpy.full(fields.grid.shape, 0.6, dtype=numpy.float32))
    fields.Dz.fill(2.0)
    stepping.step_D(fields, None)  # H is identically zero, so the curl contributes nothing.
    half_dt = fields.grid.dt / 2.0
    expected = 2.0 * (1.0 - 0.6 * half_dt) / (1.0 + 0.6 * half_dt)
    numpy.testing.assert_allclose(fields.Dz, expected, rtol=1e-6)


def test_step_D_applies_meeps_three_stage_conductivity_plus_pml_recurrence():
    """Hand-check f_cond -> f_u -> D from step_generic.cpp's general case."""
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    fields.set_d_conductivity(numpy.full(grid.shape, 0.6, dtype=numpy.float32))
    pml = PML(grid=grid, thickness=2)
    seed(
        fields,
        ("Hx", "Hy", "Hz", "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz",
         "f_cond_Dx", "f_cond_Dy", "f_cond_Dz"),
        41,
    )
    magnetic = {name: getattr(fields, name).copy() for name in ("Hx", "Hy", "Hz")}
    before = snapshot(fields, ("Dx", "fu_Dx", "f_cond_Dx"))
    dtdx = grid.dt / grid.dx

    stepping.step_D(fields, pml)

    i, j, k = 3, 1, 1
    curl = dtdx * (
        (magnetic["Hz"][i, j - 1, k] - magnetic["Hz"][i, j, k])
        + (magnetic["Hy"][i, j, k] - magnetic["Hy"][i, j, k - 1])
    )
    condfac = fields.condfac_for("Dx")[i, j, k]
    condinv = fields.condinv_for("Dx")[i, j, k]
    kms, sinv = pml.kms_y[0, j, 0], pml.sinv_y[0, j, 0]  # Dx: dsig = Y.
    kms_u, sinv_u = pml.kms_z[0, 0, k], pml.sinv_z[0, 0, k]  # Dx: dsigu = Z.
    expected_f_cond = (
        condfac * before["f_cond_Dx"][i, j, k] - curl
    ) * condinv
    expected_fu = (
        kms * before["fu_Dx"][i, j, k]
        + expected_f_cond
        - before["f_cond_Dx"][i, j, k]
    ) * sinv
    expected_d = (
        kms_u * before["Dx"][i, j, k]
        + expected_fu
        - before["fu_Dx"][i, j, k]
    ) * sinv_u
    assert fields.f_cond_Dx[i, j, k] == pytest.approx(expected_f_cond, rel=1e-5)
    assert fields.fu_Dx[i, j, k] == pytest.approx(expected_fu, rel=1e-5)
    assert fields.Dx[i, j, k] == pytest.approx(expected_d, rel=1e-5)


def test_a_zero_conductivity_leaves_step_D_bit_identical():
    """Not installing the volume at all is what keeps the lossless path exact.

    Storing zeros and multiplying by 1.0 would be equal only to within round-off,
    and this engine's whole regression story is that "equal to within round-off" is
    where silent divergences hide.
    """
    plain = make_fields()
    lossy = make_fields()
    for container in (plain, lossy):
        seed(container, ("Bx", "By", "Bz"), 11)
        container.Dz.fill(1.25)
    lossy.set_d_conductivity(None)
    assert not lossy.has_conductivity
    stepping.step_D(plain, None)
    stepping.step_D(lossy, None)
    numpy.testing.assert_array_equal(lossy.Dz, plain.Dz)


def test_step_B_applies_the_same_conductive_curl_update_as_step_D():
    """The magnetic half of an ``mp.Absorber``: step_generic.cpp's step_curl is ONE function.

    Same dt, same ``condinv = 1/(1 + sigma*dt/2)`` (structure.cpp:697-699), no half-step
    offset between the two field types — MEEP's ``FOR_D_AND_B`` loop installs the same
    coefficients on both. Dropping this half of an absorber measured 4.08e-03 relative
    Linf in 1-D and 6.46e-01 in 2-D against MEEP.
    """
    fields = make_fields()
    assert not fields.has_magnetic_conductivity
    fields.set_b_conductivity(numpy.full(fields.grid.shape, 0.6, dtype=numpy.float32))
    assert fields.has_magnetic_conductivity and not fields.has_conductivity
    fields.Bz.fill(2.0)
    stepping.step_B(fields, None)  # E is identically zero, so the curl contributes nothing.
    half_dt = fields.grid.dt / 2.0
    expected = 2.0 * (1.0 - 0.6 * half_dt) / (1.0 + 0.6 * half_dt)
    numpy.testing.assert_allclose(fields.Bz, expected, rtol=1e-6)


def test_step_B_applies_meeps_three_stage_conductivity_plus_pml_recurrence():
    """The B-side f_cond -> f_u -> B ladder, hand-checked, with the HALF-INTEGER PML slice.

    B sits at half-integer transverse positions, so its PML coefficients come from the
    ``_h`` sigma sub-lattice; reading the integer set here — the copy-paste defect this
    branch invites — is a half-cell registration error that still absorbs plausibly.
    """
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    fields.set_b_conductivity(numpy.full(grid.shape, 0.6, dtype=numpy.float32))
    pml = PML(grid=grid, thickness=2)
    seed(
        fields,
        ("Ex", "Ey", "Ez", "Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz",
         "f_cond_Bx", "f_cond_By", "f_cond_Bz"),
        47,
    )
    electric = {name: getattr(fields, name).copy() for name in ("Ex", "Ey", "Ez")}
    before = snapshot(fields, ("Bx", "fu_Bx", "f_cond_Bx"))
    dtdx = grid.dt / grid.dx

    stepping.step_B(fields, pml)

    i, j, k = 3, 1, 1
    # Bx: g1 = Ez along y, g2 = Ey along z, FORWARD strides — MEEP negates the strides
    # only for D_stuff, which is the whole of the sign difference between the two halves.
    curl = dtdx * (
        (electric["Ez"][i, j + 1, k] - electric["Ez"][i, j, k])
        + (electric["Ey"][i, j, k] - electric["Ey"][i, j, k + 1])
    )
    condfac = fields.condfac_for("Bx")[i, j, k]
    condinv = fields.condinv_for("Bx")[i, j, k]
    kms, sinv = pml.kms_y_h[0, j, 0], pml.sinv_y_h[0, j, 0]  # Bx: dsig = Y, half-integer.
    kms_u, sinv_u = pml.kms_z_h[0, 0, k], pml.sinv_z_h[0, 0, k]  # Bx: dsigu = Z, half-integer.
    expected_f_cond = (condfac * before["f_cond_Bx"][i, j, k] - curl) * condinv
    expected_fu = (
        kms * before["fu_Bx"][i, j, k] + expected_f_cond - before["f_cond_Bx"][i, j, k]
    ) * sinv
    expected_b = (
        kms_u * before["Bx"][i, j, k] + expected_fu - before["fu_Bx"][i, j, k]
    ) * sinv_u
    assert fields.f_cond_Bx[i, j, k] == pytest.approx(expected_f_cond, rel=1e-5)
    assert fields.fu_Bx[i, j, k] == pytest.approx(expected_fu, rel=1e-5)
    assert fields.Bx[i, j, k] == pytest.approx(expected_b, rel=1e-5)


def test_a_conductivity_is_read_per_component_not_per_field_type():
    """One conductive component leaves its neighbours on the untouched arithmetic path.

    MEEP allocates ``s->conductivity[c][d]`` per component, so a run whose sigma is
    nonzero only where Dz lives steps Dx and Dy exactly as a lossless run does. A
    global "is anything conductive" flag would multiply them by 1.0 instead — equal to
    within round-off, which is where this package's silent divergences hide.
    """
    plain = make_fields()
    mixed = make_fields()
    for container in (plain, mixed):
        seed(container, ("Bx", "By", "Bz"), 13)
        container.Dx.fill(1.25)
        container.Dz.fill(1.25)
    mixed.set_d_conductivity({"Dz": numpy.full(mixed.grid.shape, 0.6, dtype=numpy.float32)})
    assert mixed.conductive_components == ("Dz",)
    stepping.step_D(plain, None)
    stepping.step_D(mixed, None)
    numpy.testing.assert_array_equal(mixed.Dx, plain.Dx)  # Untouched: bit-identical.
    assert float(numpy.abs(mixed.Dz - plain.Dz).max()) > 0.0


def test_a_conductivity_damps_a_seeded_displacement_monotonically():
    # Passivity, as a property rather than a recorded number: sigma_D >= 0 can only
    # remove energy from D, at a rate that grows with sigma_D.
    magnitudes = []
    for sigma_d in (0.0, 0.5, 2.0):
        fields = make_fields()
        if sigma_d:
            fields.set_d_conductivity(numpy.full(fields.grid.shape, sigma_d, dtype=numpy.float32))
        fields.Dz.fill(1.0)
        for _ in range(20):
            stepping.step_D(fields, None)
        magnitudes.append(float(numpy.abs(fields.Dz).max()))
    assert magnitudes[0] == pytest.approx(1.0)
    assert magnitudes[0] > magnitudes[1] > magnitudes[2] > 0.0


# ---------------------------------------------------------------------------
# Real-valued (float32) field mode
# ---------------------------------------------------------------------------

# One deterministic seed pattern for the whole section, built from small integers so
# every value is exactly representable in float32 and the "bit for bit" claims below
# are about the kernels rather than about the seed's own rounding.
_REAL_SEED_COMPONENTS = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")


def _integer_seed(fields: Fields, offset: int = 0) -> None:  # Exactly representable, distinct per cell.
    for index, name in enumerate(_REAL_SEED_COMPONENTS):
        array = getattr(fields, name)
        ramp = numpy.arange(array.size, dtype=numpy.float32).reshape(array.shape)
        real_part = ((ramp + 7 * index + offset) % 17) - 8
        if numpy.iscomplexobj(array):
            # A DIFFERENT imaginary pattern, so a kernel that mixed the two planes
            # (a complex coefficient, say) would move the real part and be caught.
            array[...] = real_part + 1j * (((ramp + 3 * index) % 11) - 5)
        else:
            array[...] = real_part


def _step_once(fields: Fields, pml) -> None:  # The driver's per-timestep order, sources aside.
    stepping.step_B(fields, pml)
    stepping.fill_symmetry_bc_B(fields)
    stepping.update_H(fields, pml)
    stepping.step_D(fields, pml)
    stepping.fill_symmetry_bc_D(fields)
    stepping.update_E(fields, pml)
    stepping.update_P(fields, pml)


def _build_pair(cell_size, symmetry, pml_thickness, dispersive, resolution=10.0):
    """A real and a complex container on the same grid, seeded identically in the real part."""
    pairs = []
    for complex_fields in (False, True):
        grid = Grid(resolution=resolution, cell_size=cell_size, symmetry=symmetry)
        fields = Fields(grid=grid, force_complex_fields=complex_fields)
        if dispersive:
            fields.eps.fill(2.25)
            fields.inv_eps.fill(1.0 / 2.25)
            fields.polarizations.append(
                PolarizationState(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6, grid,
                                  fields._field_dtype())
            )
            fields.enable_field_storage()
        pml = PML(grid=grid, thickness=pml_thickness) if pml_thickness else None
        if pml is not None:
            fields.enable_pml_storage()
        _integer_seed(fields)
        pairs.append((fields, pml))
    return pairs[0], pairs[1]


@pytest.mark.parametrize(
    "label, cell_size, symmetry, pml_thickness, dispersive",
    [
        # Even AND odd cell counts on every axis, because a half-cell registration
        # error has historically been invisible at even counts alone.
        ("plain_even", (0.6, 0.6, 0.8), (), 0, False),
        ("plain_odd", (0.5, 0.7, 0.9), (), 0, False),
        ("pml_even", (0.8, 0.8, 1.0), (), 2, False),
        ("pml_odd", (0.7, 0.9, 1.1), (), 2, False),
        ("dispersive_even", (0.6, 0.6, 0.8), (), 0, True),
        ("dispersive_odd", (0.5, 0.7, 0.9), (), 0, True),
        ("dispersive_pml", (0.8, 0.8, 1.0), (), 2, True),
        ("mirror_x", (0.8, 0.6, 0.8), ("X",), 0, False),
        ("mirror_xy", (0.8, 0.8, 0.6), ("X", "Y"), 0, False),
        ("mirror_xy_pml", (1.2, 1.2, 0.8), ("X", "Y"), 2, False),
    ],
)
def test_real_stepping_is_the_real_part_of_complex_stepping(label, cell_size, symmetry,
                                                            pml_thickness, dispersive):
    """Every kernel carries real coefficients, so the two modes cannot disagree at all.

    This is the load-bearing equivalence behind real mode: it is not a cheaper
    approximation, it is the same arithmetic on half the data. Asserted with
    ``assert_array_equal`` — zero tolerance — over ten configurations sweeping even and
    odd cell counts, the absorber, dispersion and both mirror planes, because a single
    complex coefficient anywhere in the chain (a PML table built at the field dtype, a
    polarization constant cast to complex64) would move the real part by a round-off
    that no tolerance-bearing check would ever fail on.

    The complex seed carries a DIFFERENT imaginary pattern from its real one, so a
    kernel that leaked the imaginary plane into the real one is a detectable
    divergence rather than a coincidence.
    """
    (real_fields, real_pml), (complex_fields, complex_pml) = _build_pair(
        cell_size, symmetry, pml_thickness, dispersive
    )
    for _ in range(12):
        _step_once(real_fields, real_pml)
        _step_once(complex_fields, complex_pml)

    checked = 0
    for name in _REAL_SEED_COMPONENTS + ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        real_array = getattr(real_fields, name, None)
        complex_array = getattr(complex_fields, name, None)
        if real_array is None or complex_array is None:
            continue
        assert real_array.dtype == numpy.float32, f"{label}/{name} left real mode"
        numpy.testing.assert_array_equal(
            real_array, complex_array.real,
            err_msg=f"{label}: {name} is not the real part of the complex run",
        )
        checked += 1
    assert checked >= 6, "the comparison covered nothing"
    peak = max(float(numpy.abs(getattr(complex_fields, name).real).max())
               for name in _REAL_SEED_COMPONENTS)
    assert peak > 0.0, "both runs decayed to nothing; the equivalence would be vacuous"

    if dispersive:
        for real_state, complex_state in zip(real_fields.polarizations,
                                             complex_fields.polarizations):
            for component in real_state.driven():
                assert real_state.P[component].dtype == numpy.float32
                numpy.testing.assert_array_equal(
                    real_state.P[component], complex_state.P[component].real,
                    err_msg=f"{label}: P[{component}] is not the real part of the complex P",
                )


def test_real_mode_never_promotes_a_field_array_to_complex():
    """NumPy would promote silently on any complex intermediate; the dtypes are the alarm.

    An in-place ``-=`` refuses the cast, but a rebinding assignment (``self.P[c] =
    scratch`` in the polarization rotation, for instance) does not — it would just
    start storing complex64, double the memory the mode exists to save, and keep
    producing correct numbers.
    """
    (fields, pml), _ = _build_pair((0.7, 0.9, 1.1), (), 2, True)
    for _ in range(6):
        _step_once(fields, pml)
    promoted = {
        name: array.dtype
        for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz", "Ex", "Ey", "Ez",
                     "Hx", "Hy", "Hz", "fu_Dz", "f_w_Ez")
        for array in [getattr(fields, name)]
        if array is not None and array.dtype != numpy.float32
    }
    for index, state in enumerate(fields.polarizations):
        for component, array in list(state.P.items()) + list(state.P_prev.items()):
            if array.dtype != numpy.float32:
                promoted[f"P{index}[{component}]"] = array.dtype
    assert not promoted, f"real-mode stepping promoted storage to complex: {promoted}"


# Byte pin for complex mode. Recorded from the engine as it stood BEFORE real-field
# support existed, over a configuration whose every operation is float32 +, - and *
# (no libm call, no source waveform, no PML grading), so it is reproducible on any
# IEEE-754 platform rather than only on the machine that recorded it. If this digest
# ever has to move, the move must be justified by a MEASURED improvement against CPU
# MEEP — every accuracy floor in the parity matrix was recorded in complex mode.
_COMPLEX_CORE_DIGEST = "44b25b4191cd6e9c60b26ea8ef4f0a789bd5f08c6a51144fec679f60a64d4a95"

# The one float64 input to that run that is not a bare +, - or *: the polarization
# recurrence constants, which reach the arrays as scalars. Pinned separately so a
# digest failure says WHICH half moved — a platform's libm rounding these differently
# is a false alarm to be fixed by re-deriving the digest, whereas the same digest
# failure with these intact is a real change in the kernels.
_LORENTZ_COEFFICIENTS_AT_DT_0P05 = (
    1.8659228628515696, -0.9844144453917001, 0.11849158254013052,
)


def _complex_core_digest() -> str:  # Hash the raw bytes of a fully deterministic complex run.
    import hashlib

    grid = Grid(resolution=10.0, cell_size=(0.5, 0.7, 0.9))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.polarizations.append(
        PolarizationState(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6, grid, numpy.complex64)
    )
    fields.enable_field_storage()
    _integer_seed(fields)
    for _ in range(24):
        _step_once(fields, None)
    digest = hashlib.sha256()
    for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz", "Ex", "Ey", "Ez"):
        digest.update(getattr(fields, name).tobytes())
    for state in fields.polarizations:
        for component in sorted(state.P):
            digest.update(state.P[component].tobytes())
            digest.update(state.P_prev[component].tobytes())
    return digest.hexdigest()


def test_complex_mode_is_byte_identical_to_its_recorded_bytes():
    """The regression guard for every accuracy floor: complex mode must not move a bit.

    A digest rather than a tolerance, because the failure this guards against is not
    a wrong answer but a slightly different one — a coefficient promoted to complex64,
    an expression reassociated, a real-mode branch taken by accident — which no
    parity bound would notice and which would silently invalidate the recorded floors
    (2.41e-07 no-PML core, 4.00e-07 uniform PML, 3.2e-06 homogeneous Lorentz, ...).

    Deterministic by construction: integer-valued float32 seeds, and only +, - and *
    on float32/complex64 arrays with real coefficients, all of which are exactly
    specified by IEEE-754.
    """
    grid = Grid(resolution=10.0, cell_size=(0.5, 0.7, 0.9))
    assert grid.dt == 0.05
    assert (Susceptibility(1.1, 0.05, LORENTZIAN).coefficients(grid.dt)
            == _LORENTZ_COEFFICIENTS_AT_DT_0P05), (
        "the polarization constants moved; a digest failure below is that, not the kernels"
    )
    assert _complex_core_digest() == _COMPLEX_CORE_DIGEST, (
        "complex-mode stepping changed its bytes; re-derive the floors before moving this"
    )


def test_the_byte_pin_can_actually_see_a_change():
    """A digest that a real change cannot move is not a guard.

    The mutation is the smallest one that matters: a single cell of the seed nudged
    by one float32 ULP, twenty-four steps back. It must change the digest.
    """
    import hashlib

    grid = Grid(resolution=10.0, cell_size=(0.5, 0.7, 0.9))
    fields = Fields(grid=grid, force_complex_fields=True)
    _integer_seed(fields)
    fields.Dz[0, 0, 0] = numpy.nextafter(fields.Dz[0, 0, 0].real, numpy.float32(1e9))
    for _ in range(24):
        _step_once(fields, None)
    mutated = hashlib.sha256(fields.Dz.tobytes()).hexdigest()

    reference = Fields(grid=Grid(resolution=10.0, cell_size=(0.5, 0.7, 0.9)),
                       force_complex_fields=True)
    _integer_seed(reference)
    for _ in range(24):
        _step_once(reference, None)
    assert mutated != hashlib.sha256(reference.Dz.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Mirror symmetry: folded quadrant vs full domain, on every axis and both phases
# ---------------------------------------------------------------------------
#
# THE DECISIVE TEST. A folded run is a claim that half (or a quarter, or an
# eighth) of the domain can be reconstructed rather than stepped. The only way to
# check that claim without a second implementation is to step BOTH and demand they
# agree — and, because every kernel here is +, - and * on float32/complex64 with
# real coefficients, they must agree BIT FOR BIT rather than to a tolerance. Any
# tolerance would hide exactly the failures worth catching: a parity applied to the
# wrong component, a ghost read from the wrong cell, a mask on the wrong axis, all
# of which move the answer by a few percent and leave a perfectly smooth field.
#
# The construction:
#   * an initial condition on the FULL domain that satisfies the mirror relation
#     exactly (`_symmetrize`, which is exact in floating point — see its docstring);
#   * compactly supported near the plane, so the far face of the folded axis stays
#     identically zero for the whole run. That matters because the two runs
#     genuinely differ there: the full domain WRAPS its far face while a folded
#     axis terminates it with a zero ghost. They are the same system only while
#     that face is dark, which every assertion below checks rather than assumes.


def _mirror_partner(n_full: int, shift: int):  # Index permutation one mirror performs on one axis.
    """Cell indices swapped by a mirror about the plane, for a component with Yee shift ``shift``.

    Doubled coordinates (MEEP vec.hpp): cell ``i`` of a centred axis sits at
    ``-n_full + 2i + s``, and the mirror sends that to its negative. Solving
    ``-n_full + 2i' + s = n_full - 2i - s`` gives ``i' = n_full - i - s``, taken
    modulo ``n_full`` because the full domain is periodic — which is also the
    statement that a periodic cell with a mirror at 0 has a SECOND mirror at
    +/-L/2, and why the far face has to stay dark for the comparison to mean
    anything.
    """
    index = numpy.arange(n_full)
    return (n_full - index - shift) % n_full


def _symmetrize(array, component: str, planes) -> numpy.ndarray:
    """Project a field onto the subspace one or more mirror planes allow.

    ``planes`` maps axis -> declared mirror phase. The projection
    ``0.5 * (f + parity * f_mirrored)`` is EXACT in IEEE arithmetic, which is what
    lets the comparison below be bit-for-bit rather than approximate: ``0.5 * x``
    and multiplication by +/-1 are exact, float addition is commutative, and
    ``fl(b - a)`` is exactly ``-fl(a - b)``. So the result satisfies
    ``f[mirror(i)] == parity * f[i]`` to the last bit, on the self-mapped cells too
    (an odd component lands on exactly 0 there, which is the physics).
    """
    result = array
    for axis, mirror_phase in planes.items():
        parity = mirror_parity(component, axis, mirror_phase)
        partner = _mirror_partner(result.shape[axis], IYEE_SHIFTS[component][axis])
        mirrored = numpy.take(result, partner, axis=axis)
        result = 0.5 * (result + parity * mirrored)
    return result.astype(array.dtype)


def _compact_blob(shape, planes, radius: int = 2, seed_value: int = 5):
    """Random field confined to ``radius`` cells either side of every mirror plane.

    Confined so the far face of each folded axis is identically zero and stays that
    way: the stencil reaches one cell per sub-step, so ``radius + steps`` cells is
    the most the disturbance can cover.
    """
    generator = numpy.random.default_rng(seed_value)
    values = generator.integers(-8, 9, size=shape).astype(numpy.float32)  # Exact in float32.
    values = values + 1j * generator.integers(-5, 6, size=shape).astype(numpy.float32)
    for axis in planes:
        n_full = shape[axis]
        window = numpy.zeros(n_full, dtype=numpy.float32)
        centre = n_full // 2
        window[centre - radius: centre + radius] = 1.0
        broadcast = [1, 1, 1]
        broadcast[axis] = n_full
        values = values * window.reshape(broadcast)
    return values.astype(numpy.complex64)


def _folded_slice(axis: int, n_full: int):  # Quadrant q is full-domain cell q + n_full//2 - 1.
    return _slab(axis, slice(n_full // 2 - 1, None))


def _slab(axis: int, index):  # Index tuple selecting along one axis only.
    return (slice(None),) * axis + (index,)


def _fold(array, planes, shape_full):  # The stored quadrant of a full-domain array.
    """Fold a full-domain array into the stored layout of a PERIODIC mirror grid.

    Quadrant row q is full-domain cell ``q + n_full//2 - 1``; at an EVEN full
    count the stored array carries one row more (`Grid.stored_cells` vs
    `owned_cells`): the second-mirror plane at doubled ``n_full`` plus the ghost
    slot past it, which on the full periodic domain are exactly full row 0's own
    samples one lattice vector up (doubled ``n_full`` = ``-n_full`` wrapped, at
    k = 0 with no phase). So the fold appends full row 0 there — for shift-0 and
    shift-1 components alike.
    """
    for axis in planes:
        n_full = shape_full[axis]
        folded = array[_folded_slice(axis, n_full)]
        if n_full % 2 == 0:
            folded = numpy.concatenate((folded, array[_slab(axis, slice(0, 1))]),
                                       axis=axis)
        array = folded
    return array


PRIMARY_COMPONENTS = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")

# Every mirror axis, both phases, and the combinations — including the eightfold
# fold, which is the case that exercises a corner unowned on three planes at once.
FOLD_CASES = {
    "x_even": {0: +1},
    "x_odd": {0: -1},
    "y_even": {1: +1},
    "y_odd": {1: -1},
    "z_even": {2: +1},
    "z_odd": {2: -1},
    "xy_even": {0: +1, 1: +1},
    "xy_mixed": {0: -1, 1: +1},
    "xz_odd": {0: -1, 2: -1},
    "yz_mixed": {1: +1, 2: -1},
    "xyz_even": {0: +1, 1: +1, 2: +1},
    "xyz_odd": {0: -1, 1: -1, 2: -1},
}


def _symmetry_argument(planes):  # Grid `symmetry=` request for a plane map.
    return tuple(Mirror("XYZ"[axis], phase) for axis, phase in sorted(planes.items()))


@pytest.mark.parametrize("case", sorted(FOLD_CASES))
@pytest.mark.parametrize("complex_fields", [True, False])
def test_a_folded_run_reproduces_the_full_domain_run_bit_for_bit(case, complex_fields):
    """The equivalence that makes folding legitimate, for every axis and both phases.

    An odd mirror is not a new algorithm, and neither is a Z one: the parity comes
    from ``fields.mirror_parity``, the fold from ``Grid``, and the ghost cells and
    ownership masks from the same per-axis rules the even X mirror has always used.
    If any of those three had a leftover X/Y or phase = +1 assumption, this
    comparison moves — by ~1e0 for a wrong parity, by ~1e-1 for a mask on the wrong
    axis — and it is checked at zero tolerance so it cannot be absorbed.
    """
    planes = FOLD_CASES[case]
    cell_size = (2.0, 2.0, 2.0)
    steps = 6  # radius 2 + 6 steps = 8 cells < the 10 that reach the far face.
    full_grid = Grid(resolution=10.0, cell_size=cell_size)
    folded_grid = Grid(resolution=10.0, cell_size=cell_size,
                       symmetry=_symmetry_argument(planes))
    assert folded_grid.shape != full_grid.shape, "the folded grid must actually be smaller"

    full = Fields(grid=full_grid, force_complex_fields=complex_fields)
    folded = Fields(grid=folded_grid, force_complex_fields=complex_fields)
    blob = _compact_blob(full_grid.shape, planes)
    for component in PRIMARY_COMPONENTS:
        seeded = _symmetrize(blob, component, planes)
        if not complex_fields:
            seeded = seeded.real.copy()
        getattr(full, component)[...] = seeded
        getattr(folded, component)[...] = _fold(seeded, planes, full_grid.shape_full)

    for _ in range(steps):
        _step_once(full, None)
        _step_once(folded, None)

    for component in PRIMARY_COMPONENTS:
        reference = _fold(numpy.asarray(getattr(full, component)), planes,
                          full_grid.shape_full)
        candidate = numpy.asarray(getattr(folded, component))
        assert candidate.shape == reference.shape
        numpy.testing.assert_array_equal(
            candidate, reference,
            err_msg=f"{case}: folded {component} differs from the full-domain quadrant")

    # The comparison is only meaningful while the boundary the two runs treat
    # DIFFERENTLY is dark, so that is checked rather than assumed: a folded axis
    # ends in a zero ghost where the full domain wraps.
    for axis in planes:
        far_face = numpy.asarray(folded.Dz)[_slab(axis, -1)]
        assert numpy.count_nonzero(far_face) == 0, (
            f"{case}: the far {'xyz'[axis]} face is live, so the two runs no longer "
            f"terminate the same system and this equivalence proves nothing")
    assert numpy.abs(numpy.asarray(folded.Dz)).max() > 0.0, "the folded run carries no field"


def _folded_pair(planes, cell_size, resolution, complex_fields, k_point, pml_request,
                 dispersive):
    """A full-domain and a folded container on matching grids, seeded symmetrically."""
    built = []
    for symmetry in ((), _symmetry_argument(planes)):
        grid = Grid(resolution=resolution, cell_size=cell_size, symmetry=symmetry,
                    k_point=k_point)
        fields = Fields(grid=grid, force_complex_fields=complex_fields)
        if dispersive:
            fields.eps.fill(2.25)  # Uniform, so the material itself is trivially symmetric.
            fields.inv_eps.fill(1.0 / 2.25)
            fields.polarizations.append(
                PolarizationState(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6, grid,
                                  fields._field_dtype())
            )
            fields.enable_field_storage()
        pml = PML(grid=grid, thickness=pml_request) if pml_request else None
        if pml is not None:
            fields.enable_pml_storage()
        built.append((fields, pml))
    return built[0], built[1]


# Combinations, because every one of them reaches the fold by a different route: an
# odd cell count moves the origin of the UNFOLDED axes, an absorber adds the
# split-field auxiliaries under the ownership mask, a Bloch phase adds a wrap ghost
# on an axis the fold does not touch, and a susceptibility adds a per-cell history
# that the boundary pass has to leave alone.
COMBINATION_CASES = {
    # label: (planes, cell_size, k_point, pml, dispersive)
    "odd_unfolded_counts": ({0: -1}, (2.0, 0.9, 1.1), (0.0, 0.0, 0.0), None, False),
    "odd_counts_z_fold": ({2: -1}, (0.9, 1.1, 2.0), (0.0, 0.0, 0.0), None, False),
    "pml_on_the_unfolded_axis": ({0: -1}, (2.0, 0.8, 2.0), (0.0, 0.0, 0.0), {"z": 3}, False),
    "z_fold_with_pml_on_x": ({2: +1}, (2.0, 0.8, 2.0), (0.0, 0.0, 0.0), {"x": 3}, False),
    "bloch_on_the_unfolded_axis": ({0: -1}, (2.0, 0.8, 2.0), (0.0, 0.0, 0.35), None, False),
    "bloch_with_a_z_fold": ({2: -1}, (0.8, 0.8, 2.0), (0.3, 0.0, 0.0), None, False),
    "dispersive_odd_fold": ({0: -1}, (2.0, 0.9, 1.1), (0.0, 0.0, 0.0), None, True),
    "dispersive_z_fold_with_pml": ({2: -1}, (0.9, 0.8, 2.0), (0.0, 0.0, 0.0), {"y": 3}, True),
    "two_folds_pml_and_dispersion": ({0: +1, 2: -1}, (2.0, 0.8, 2.0), (0.0, 0.0, 0.0),
                                     {"y": 3}, True),
}


@pytest.mark.parametrize("case", sorted(COMBINATION_CASES))
def test_the_fold_equivalence_survives_pml_bloch_dispersion_and_odd_counts(case):
    """The same bit-for-bit equivalence, with everything else switched on around it.

    Sweeping combinations rather than the bare fold is not thoroughness for its own
    sake — it is where this engine's silent wrong answers have actually lived. Odd
    cell counts hid a half-cell registration error on every axis that had one; a
    folded axis and an absorber disagreeing about a wrap put full conductivity on
    a boundary plane. Each case here puts the fold next to one of those.
    """
    planes, cell_size, k_point, pml_request, dispersive = COMBINATION_CASES[case]
    complex_fields = True  # Bloch needs it, and the real path is pinned by the case above.
    (full, full_pml), (folded, folded_pml) = _folded_pair(
        planes, cell_size, 10.0, complex_fields, k_point, pml_request, dispersive)

    blob = _compact_blob(full.grid.shape, planes)
    for component in PRIMARY_COMPONENTS:
        seeded = _symmetrize(blob, component, planes)
        getattr(full, component)[...] = seeded
        getattr(folded, component)[...] = _fold(seeded, planes, full.grid.shape_full)

    for _ in range(6):
        _step_once(full, full_pml)
        _step_once(folded, folded_pml)

    for component in PRIMARY_COMPONENTS:
        reference = _fold(numpy.asarray(getattr(full, component)), planes,
                          full.grid.shape_full)
        numpy.testing.assert_array_equal(
            numpy.asarray(getattr(folded, component)), reference,
            err_msg=f"{case}: folded {component} differs from the full-domain quadrant")
    for axis in planes:
        assert numpy.count_nonzero(numpy.asarray(folded.Dz)[_slab(axis, -1)]) == 0, (
            f"{case}: the far {'xyz'[axis]} face is live; the two runs terminate it "
            f"differently, so the comparison would prove nothing")
    assert numpy.abs(numpy.asarray(folded.Dz)).max() > 0.0, "the folded run carries no field"


def test_the_fold_equivalence_can_actually_see_a_wrong_parity():
    """A test that passes on the wrong parity is not evidence.

    The mutation is the one the feature is about: resolve the parity from the
    even-mirror default instead of the declared phase. On an odd fold every ghost
    and every reconstruction flips sign, and the folded run must diverge from the
    full-domain one by order unity.
    """
    planes = {0: -1}
    cell_size = (2.0, 2.0, 2.0)
    full_grid = Grid(resolution=10.0, cell_size=cell_size)
    folded_grid = Grid(resolution=10.0, cell_size=cell_size, symmetry=(Mirror("X", -1),))
    full = Fields(grid=full_grid, force_complex_fields=True)
    folded = Fields(grid=folded_grid, force_complex_fields=True)
    blob = _compact_blob(full_grid.shape, planes)
    for component in PRIMARY_COMPONENTS:
        seeded = _symmetrize(blob, component, planes)
        getattr(full, component)[...] = seeded
        getattr(folded, component)[...] = _fold(seeded, planes, full_grid.shape_full)

    original = stepping._mirror_phases
    stepping._mirror_phases = lambda grid: tuple(  # The even-mirror default, everywhere.
        1 if grid.is_mirrored(axis) else None for axis in range(3))
    try:
        for _ in range(6):
            _step_once(full, None)
            _step_once(folded, None)
    finally:
        stepping._mirror_phases = original

    reference = _fold(numpy.asarray(full.Dz), planes, full_grid.shape_full)
    deviation = (float(numpy.linalg.norm(numpy.asarray(folded.Dz) - reference))
                 / float(numpy.linalg.norm(reference)))
    assert deviation > 0.1, (
        f"forcing the even-mirror parity on an odd fold moved the answer by only "
        f"{deviation:.2e}; the equivalence test above is not sensitive to the parity")


# Recorded from the even X/Y + PML fold below, and verified against the engine as it
# stood BEFORE mirror phases and Z folds existed (the same script run against the
# pre-change grid.py / fields.py / stepping.py returned these exact digests).
# Regenerate ONLY with a measurement showing the CPU-MEEP error went down.
# Re-blessed 2026-08-05 with the folded-periodic far-plane storage (the second
# mirror at +L/2 stored and stepped, ghost slot filled, readback reflecting):
# re-measured against CPU MEEP with the far face LIVE at 13-24% of peak, worst
# whole-volume complex rel L2 4.5e-07 over even/odd counts x even/odd phases
# (2-D 4x2 cell, res 16, PML X only, Y periodic at Gamma, Gaussian, 256 steps).
_EVEN_XY_FOLD_DIGEST = "ce4c7e524b415e80f17ef7bbf9cc1bac135fdfc2f5e9c612b90c5b8060036706"
_EVEN_XY_READBACK_DIGEST = "ac61d591712835a5121364dea72f78c6c8bac1bfb1ac5ed1ae9b98af57813887"


def test_an_even_xy_fold_is_byte_identical_to_the_recorded_pre_phase_bytes():
    """The regression guard for the exactly-0.0 floor: even X/Y must not move a bit.

    Generalizing the parity to a declared phase and the fold to any axis is only
    safe if the case that was already cross-validated against CPU MEEP comes out
    unchanged — bit for bit, not to a tolerance. The failure mode being guarded is
    not a wrong answer but a slightly different one (a parity recomputed in a
    different order, a reconstruction reassociated, a coefficient promoted), which
    no parity bound would notice and which would silently invalidate every recorded
    symmetry floor.

    Two digests, because the fold has two halves and they fail independently: the
    STEPPING bytes cover the mirror ghosts, the ownership masks and the PML
    auxiliaries under them, and the READBACK digest covers cell-centering with a
    metallic far edge, quadrant unfolding with the mirror parities, and MEEP's
    (N, N, N+1) get_array geometry.
    """
    import hashlib

    grid = Grid(resolution=10.0, cell_size=(1.2, 1.2, 0.9), symmetry=("X", "Y"))
    fields = Fields(grid=grid, force_complex_fields=True)
    pml = PML(grid=grid, thickness=2)
    fields.enable_pml_storage()
    _integer_seed(fields)
    for _ in range(16):
        _step_once(fields, pml)

    stepped = hashlib.sha256()
    for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        stepped.update(numpy.asarray(getattr(fields, name)).tobytes())
    assert stepped.hexdigest() == _EVEN_XY_FOLD_DIGEST, (
        "even X/Y mirror stepping changed its bytes; the folded CPU-MEEP floors were "
        "measured on the old ones and must be re-measured before this moves"
    )

    readback = hashlib.sha256()
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        array = numpy.ascontiguousarray(numpy.asarray(fields.to_meep_array(name)))
        readback.update(array.tobytes())
    assert array.shape == (grid.nx_full, grid.ny_full, grid.nz + 1), (
        "a folded axis returns its full cell count and no periodic duplicate"
    )
    assert readback.hexdigest() == _EVEN_XY_READBACK_DIGEST, (
        "even X/Y quadrant unfolding changed its bytes; the CPU-MEEP comparisons run "
        "through to_meep_array and were measured on the old ones"
    )


# --------------------------------------------------------------------------------------
# Instantaneous chi2 / chi3 (MEEP's mp.Medium(chi2=..., chi3=...)).
#
# The kernel is small, so these are value tests against MEEP's own expressions rather
# than against a re-run of the engine: calc_nonlinear_u term for term, the four-point
# Yee sum cell index by cell index, and the two properties that carry the physics —
# zero nonlinearity byte-identical to the linear path, and real/imaginary parts
# nonlinearized independently the way MEEP's DOCMP loop does it.
# --------------------------------------------------------------------------------------


def _nonlinear_fields(chi2=0.0, chi3=0.0, cell_size=(0.6, 0.6, 0.6), complex_fields=True,
                      epsilon=1.0, symmetry=()) -> Fields:
    fields = make_fields(cell_size=cell_size, complex_fields=complex_fields, symmetry=symmetry)
    if epsilon != 1.0:
        fields.set_background_eps(epsilon)
    fields.set_nonlinear_volumes({name: chi2 for name in ("Ex", "Ey", "Ez")},
                                 {name: chi3 for name in ("Ex", "Ey", "Ez")})
    if fields.has_nonlinearity:
        fields.enable_field_storage()
    return fields


def test_calc_nonlinear_u_is_meeps_pade_expression_term_for_term():
    """step_generic.cpp:542-548, restated independently and evaluated at awkward values.

    Restating rather than re-running is the point: a transcription error in the
    powers of chi1inv or in the 2/3 weights is invisible at chi1inv = 1, which is
    what every vacuum test runs at, so the reference here carries a chi1inv that is
    neither 1 nor its own square.
    """
    for dsqr, di, chi1inv, chi2, chi3 in [
        (4.0, 2.0, 0.25, 0.3, 0.1),
        (9.0, -3.0, 0.4, -0.2, 0.05),
        (0.0, 0.0, 1.0, 5.0, 5.0),
        (1e-3, 1e-2, 2.0, 1.0, -1.0),
    ]:
        c2 = di * chi2 * chi1inv ** 2
        c3 = dsqr * chi3 * chi1inv ** 3
        expected = (1.0 + c2 + 2.0 * c3) / (1.0 + 2.0 * c2 + 3.0 * c3)
        assert stepping.calc_nonlinear_u(dsqr, di, chi1inv, chi2, chi3) == pytest.approx(expected, rel=1e-12)

    # Zero nonlinearity is EXACTLY one, not one to round-off: the linear path's
    # byte-identity rests on the factor never being applied, and on it being the
    # identity if it ever were.
    assert stepping.calc_nonlinear_u(3.7, -1.4, 0.6, 0.0, 0.0) == 1.0

    # First order in the coefficients the material model is itself defined to:
    # u = 1 - c2 - c3 + O(2). A Pade with the numerator and denominator weights
    # swapped reproduces neither sign.
    small = stepping.calc_nonlinear_u(1e-6, 1e-6, 1.0, 1e-3, 1e-3)
    assert small == pytest.approx(1.0 - 1e-9 - 1e-9, rel=1e-6)


def test_the_pade_approximant_tracks_the_cubic_it_stands_in_for():
    """MEEP substitutes the Pade for solving D = eps*E + chi3*E^3; check it against that root.

    With eps = 1 the cubic is ``u + c3*u^3 = 1`` for ``E = u*D``. The approximant is
    accurate to several orders where the expansion parameter is small and degrades
    exactly where MEEP says the underlying power series stops being physical, so the
    bound tightens as c3 shrinks — which is the shape of the claim, not a tolerance
    picked to pass.
    """
    for c3, bound in [(0.01, 2e-5), (0.1, 2e-3), (1.0 / 3.0, 2e-2)]:
        approximated = stepping.calc_nonlinear_u(1.0, 0.0, 1.0, 0.0, c3)
        exact = numpy.roots([c3, 0.0, 1.0, -1.0])
        exact = float(min(root.real for root in exact if abs(root.imag) < 1e-12 and root.real > 0))
        assert abs(approximated - exact) / exact < bound, (
            f"Pade at c3={c3} deviates from the cubic root by more than {bound}"
        )

    # And the pole is real: the denominator vanishes at 1 + 3*c3 = 0, past which the
    # recovered field is finite, smooth and of the wrong sign. This is what
    # nonlinear_margin exists to measure.
    assert stepping.calc_nonlinear_u(1.0, 0.0, 1.0, 0.0, -0.34) < 0.0
    assert stepping.calc_nonlinear_u(1.0, 0.0, 1.0, 0.0, -0.30) > 0.0


def test_zero_chi2_and_chi3_step_the_linear_kernel_byte_for_byte():
    """The regression guard behind every recorded accuracy floor.

    A chi2/chi3 pair that is identically zero must not merely agree with the linear
    engine to round-off — it must produce the same bytes, because a multiply by a
    Pade factor that evaluates to 1.0 is NOT the identity in float32 once it is
    reassociated with the constitutive product. Held over a PML + dispersive run, so
    the auxiliaries and the polarization history are compared too.
    """
    import hashlib

    digests = []
    for chi in (None, 0.0):
        grid = Grid(resolution=10.0, cell_size=(0.7, 0.5, 0.9))
        fields = Fields(grid=grid, force_complex_fields=True)
        fields.set_background_eps(2.25)
        fields.polarizations.append(
            PolarizationState(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6, grid,
                              fields._field_dtype())
        )
        pml = PML(grid=grid, thickness=2)
        fields.enable_pml_storage()
        if chi is not None:
            fields.set_nonlinear_volumes({name: chi for name in ("Ex", "Ey", "Ez")},
                                         {name: chi for name in ("Ex", "Ey", "Ez")})
        assert not fields.has_nonlinearity, "an identically zero pair must leave no nonlinearity"
        _integer_seed(fields)
        for _ in range(8):
            _step_once(fields, pml)
        digest = hashlib.sha256()
        for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                     "f_w_Ex", "f_w_Ey", "f_w_Ez"):
            digest.update(numpy.asarray(getattr(fields, name)).tobytes())
        digests.append(digest.hexdigest())
    assert digests[0] == digests[1], (
        "set_nonlinear_volumes with an identically zero pair changed the stepped bytes; "
        "every recorded CPU-MEEP floor was measured without it"
    )

    # Positive control: the same run with a live chi3 must NOT match, or the digest
    # comparison above is comparing a kernel that never runs against itself.
    grid = Grid(resolution=10.0, cell_size=(0.7, 0.5, 0.9))
    live = Fields(grid=grid, force_complex_fields=True)
    live.set_background_eps(2.25)
    live.polarizations.append(
        PolarizationState(Susceptibility(1.1, 0.05, LORENTZIAN), 0.6, grid, live._field_dtype())
    )
    live_pml = PML(grid=grid, thickness=2)
    live.enable_pml_storage()
    live.set_nonlinear_volumes({name: 0.0 for name in ("Ex", "Ey", "Ez")},
                               {name: 1e-3 for name in ("Ex", "Ey", "Ez")})
    assert live.has_nonlinearity
    _integer_seed(live)
    for _ in range(8):
        _step_once(live, live_pml)
    control = hashlib.sha256()
    for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        control.update(numpy.asarray(getattr(live, name)).tobytes())
    assert control.hexdigest() != digests[0], (
        "a live chi3 produced the same bytes as no nonlinearity at all; the kernel is not running"
    )


def test_the_transverse_yee_sum_reads_half_a_cell_down_and_half_a_cell_up():
    """MEEP's ``g1[i] + g1[i+s] + g1[i-s1] + g1[i+(s-s1)]``, cell index by cell index.

    Restated with explicit periodic indices instead of the ported rolls, on a field
    of distinct integers so that ANY of the four corners landing on the wrong cell
    changes the sum. The two shifts genuinely go in opposite directions — DOWN the
    partner's own axis, UP the updated component's — and taking them the same way
    round is a half-cell registration error that costs 2.2e-02 against CPU MEEP
    while every scalar-polarization case stays on its floor.
    """
    fields = _nonlinear_fields(chi3=1e-3, cell_size=(0.5, 0.7, 0.4), complex_fields=False)
    generator = numpy.random.default_rng(20260801)
    for name in ("Dx", "Dy", "Dz"):
        # Small integers: exact in float32, so a four-term sum is association-order
        # independent and the comparison can be made at zero tolerance.
        getattr(fields, name)[...] = generator.integers(
            -8, 9, size=fields.grid.shape).astype(numpy.float32)
    displacement = stepping._nonlinear_displacement(fields, None)

    for component, own_axis in (("Ex", 0), ("Ey", 1), ("Ez", 2)):
        first, second = stepping._nonlinear_transverse_sums(fields, component, displacement)
        for offset, produced in ((1, first), (2, second)):
            partner_axis = (own_axis + offset) % 3
            values = numpy.asarray(getattr(fields, "D" + "xyz"[partner_axis]))
            shape = values.shape
            expected = numpy.zeros(shape, dtype=values.dtype)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        cell = (i, j, k)
                        up = neighbour(cell, own_axis, +1, shape)
                        down = neighbour(cell, partner_axis, -1, shape)
                        corner = neighbour(down, own_axis, +1, shape)
                        expected[cell] = (values[cell] + values[up]
                                          + values[down] + values[corner])
            numpy.testing.assert_allclose(numpy.asarray(produced), expected, rtol=0, atol=0)

    # And the transverse partners follow cycle_direction (vec.hpp:586), X -> Y -> Z:
    # Ez's first partner is Dx and its second is Dy. Documentation rather than a
    # numerical guard, and deliberately so — Dsqr adds the two SQUARED sums, so
    # exchanging them changes nothing, and the only thing that could go wrong is a
    # later reader pairing g1s with the wrong axis. Pinned so the pairing is written
    # down somewhere rather than only implied by the loop.
    fields.Dx[...] = 0.0
    fields.Dy[...] = 1.0
    fields.Dz[...] = 0.0
    displacement = stepping._nonlinear_displacement(fields, None)
    first, second = stepping._nonlinear_transverse_sums(fields, "Ez", displacement)
    assert float(numpy.max(numpy.abs(numpy.asarray(first)))) == 0.0, "Ez's g1 partner must be Dx"
    assert float(numpy.min(numpy.asarray(second))) == 4.0, "Ez's g2 partner must be Dy"


def test_dsqr_is_the_component_plus_the_quarter_weighted_transverse_means():
    """``gs*gs + 0.0625*(g1s*g1s + g2s*g2s)`` — the 1/16 that turns two sums into means.

    With every D component set to the same constant c, each four-point sum is 4c and
    Dsqr must be c^2 + 0.0625*(16c^2 + 16c^2) = 3c^2 — i.e. |D|^2 for an isotropic
    field, which is what the weight is FOR. A weight of 0.25 would give 9c^2 and pass
    any test that only asked for "some transverse contribution".
    """
    fields = _nonlinear_fields(chi3=1e-3, cell_size=(0.5, 0.5, 0.5), complex_fields=False)
    for name in ("Dx", "Dy", "Dz"):
        getattr(fields, name)[...] = 3.0
    displacement = stepping._nonlinear_displacement(fields, None)
    transverse = stepping._nonlinear_transverse_sums(fields, "Ez", displacement)
    dsqr = stepping._nonlinear_dsqr(fields.Dz, transverse, stepping._WHOLE)
    numpy.testing.assert_allclose(numpy.asarray(dsqr), 27.0, rtol=1e-6)


def test_real_and_imaginary_parts_are_nonlinearized_independently():
    """MEEP's DOCMP loop runs the whole nonlinear update once per Cartesian part.

    update_eh.cpp:149 passes ``dmp[dc][cmp]``, so Re E is recovered from Re D with a
    ``u`` built only from real parts. Forming one complex Dsqr instead is the
    natural-looking implementation and is wrong by 47% against CPU MEEP on the
    chi3 point-source case, with every magnitude still looking sensible. Checked here
    by giving the two parts different magnitudes and comparing each against a scalar
    evaluation of MEEP's expression.
    """
    fields = _nonlinear_fields(chi2=0.02, chi3=0.05, cell_size=(0.5, 0.5, 0.5),
                               complex_fields=True, epsilon=4.0)
    fields.Dx[...] = 0.0
    fields.Dy[...] = 0.0
    fields.Dz[...] = 8.0 + 24.0j
    displacement = stepping._nonlinear_displacement(fields, None)
    produced = numpy.asarray(stepping._nonlinear_constitutive(fields, "Ez", displacement))

    chi1inv = 0.25
    # Dx = Dy = 0, so each transverse sum is zero and Dsqr is the component alone.
    expected_real = 8.0 * chi1inv * stepping.calc_nonlinear_u(64.0, 8.0, chi1inv, 0.02, 0.05)
    expected_imaginary = 24.0 * chi1inv * stepping.calc_nonlinear_u(576.0, 24.0, chi1inv, 0.02, 0.05)
    numpy.testing.assert_allclose(produced.real, expected_real, rtol=1e-5)
    numpy.testing.assert_allclose(produced.imag, expected_imaginary, rtol=1e-5)
    # The mixed-part alternative differs by 40% here, so this is not a coincidence of
    # the chosen numbers; on the CPU-MEEP chi3 point-source case it is 47% off.
    mixed = (8.0 + 24.0j) * chi1inv * stepping.calc_nonlinear_u(
        (8.0 + 24.0j) ** 2, 8.0 + 24.0j, chi1inv, 0.02, 0.05)
    assert abs(mixed - complex(produced.flat[0])) / abs(produced.flat[0]) > 0.3


def test_a_real_run_reproduces_the_real_part_of_the_complex_one_under_a_nonlinearity():
    """float32 storage is the real part of complex64 storage, nonlinearity included.

    The property every real-mode floor in this engine rests on. It survives chi2/chi3
    only BECAUSE MEEP nonlinearizes the two parts separately — a complex Dsqr would
    couple them and break it, which makes this an independent check on the same rule
    the test above pins directly.
    """
    digests = []
    for complex_fields in (False, True):
        grid = Grid(resolution=10.0, cell_size=(0.7, 0.5, 0.9))
        fields = Fields(grid=grid, force_complex_fields=complex_fields)
        fields.set_background_eps(2.25)
        fields.set_nonlinear_volumes({name: 2e-3 for name in ("Ex", "Ey", "Ez")},
                                     {name: 5e-3 for name in ("Ex", "Ey", "Ez")})
        fields.enable_field_storage()
        pml = PML(grid=grid, thickness=2)
        fields.enable_pml_storage()
        for index, name in enumerate(("Dx", "Dy", "Dz", "Bx", "By", "Bz")):
            array = getattr(fields, name)
            ramp = numpy.arange(array.size, dtype=numpy.float32).reshape(array.shape)
            array[...] = ((ramp + 7 * index) % 17) - 8  # Real part only; imag stays zero.
        for _ in range(8):
            _step_once(fields, pml)
        digests.append({name: numpy.asarray(getattr(fields, name)).real.copy()
                        for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez")})
    for name, values in digests[0].items():
        numpy.testing.assert_array_equal(
            values, digests[1][name],
            err_msg=f"real-mode {name} is not bit-for-bit the real part of the complex run",
        )


def test_the_nonlinear_factor_scales_the_pml_auxiliary_not_the_field_after_it():
    """MEEP applies the Pade factor to ``fw``, upstream of the absorbing accumulation.

    step_generic.cpp:653-655 writes ``fw[i] = (gs*us) * calc_nonlinear_u(...)`` and
    only then accumulates ``f[i] += (kap+sig)*fw - (kap-sig)*fwprev``. Applying it to
    ``f`` afterwards instead would leave the polarization drive field ``w`` linear —
    exactly zero error outside the layer and smoothly wrong inside it, which is the
    defect class this engine has already shipped once.
    """
    grid = Grid(resolution=10.0, cell_size=(0.6, 0.6, 0.6))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.set_background_eps(4.0)
    fields.set_nonlinear_volumes({name: 0.0 for name in ("Ex", "Ey", "Ez")},
                                 {name: 0.05 for name in ("Ex", "Ey", "Ez")})
    fields.enable_field_storage()
    pml = PML(grid=grid, thickness=2)
    fields.enable_pml_storage()
    fields.Dx[...] = 0.0
    fields.Dy[...] = 0.0
    fields.Dz[...] = 2.0
    stepping.update_E(fields, pml)

    expected = 2.0 * 0.25 * stepping.calc_nonlinear_u(4.0, 2.0, 0.25, 0.0, 0.05)
    numpy.testing.assert_allclose(numpy.asarray(fields.f_w_Ez), expected, rtol=1e-6)
    # drive_field is f_w under PML, so the polarization sees the NONLINEAR field.
    numpy.testing.assert_allclose(numpy.asarray(fields.drive_field("Ez")), expected, rtol=1e-6)
    assert expected != pytest.approx(0.5, rel=1e-6), "the linear value must be distinguishable"


def test_a_partly_nonlinear_material_leaves_the_other_components_on_the_linear_branch():
    """MEEP's ``else if (u)`` branch: a component with no chi2/chi3 takes ``gs*us`` unscaled."""
    fields = make_fields(cell_size=(0.5, 0.5, 0.5), complex_fields=False)
    fields.set_background_eps(4.0)
    fields.set_nonlinear_volumes({"Ez": 0.0}, {"Ez": 0.05})
    fields.enable_field_storage()
    assert fields.nonlinear_components == ("Ez",)
    fields.Dx[...] = 2.0
    fields.Dy[...] = 0.0
    fields.Dz[...] = 2.0
    stepping.update_E(fields, None)
    numpy.testing.assert_allclose(numpy.asarray(fields.Ex), 0.5, rtol=1e-6)
    # Ez's Dsqr picks up Dx through the transverse average: 4 + 0.0625*(8^2) = 8.
    expected = 2.0 * 0.25 * stepping.calc_nonlinear_u(8.0, 2.0, 0.25, 0.0, 0.05)
    numpy.testing.assert_allclose(numpy.asarray(fields.Ez), expected, rtol=1e-6)


def test_nonlinear_margin_reports_the_pade_denominator_and_the_expansion_parameter():
    """The measured quantity the driver's refusal is built on, checked against its closed form."""
    assert stepping.nonlinear_margin(make_fields()) is None, "a linear run has no margin to report"

    # One nonlinear component, so the reduction has exactly one closed form to match.
    fields = make_fields(cell_size=(0.5, 0.5, 0.5), complex_fields=False)
    fields.set_background_eps(4.0)
    fields.set_nonlinear_volumes({"Ez": 0.02}, {"Ez": 0.05})
    fields.enable_field_storage()
    fields.Dx[...] = 0.0
    fields.Dy[...] = 0.0
    fields.Dz[...] = 3.0
    margin = stepping.nonlinear_margin(fields)
    chi1inv = 0.25
    c2 = 3.0 * 0.02 * chi1inv ** 2
    c3 = 9.0 * 0.05 * chi1inv ** 3
    assert margin.component == "Ez"
    assert margin.expansion == pytest.approx(abs(c2) + abs(c3), rel=1e-5)
    assert margin.denominator == pytest.approx(1.0 + 2.0 * c2 + 3.0 * c3, rel=1e-5)

    # The inequality the driver's 1/3 limit is derived from, checked rather than asserted.
    assert margin.denominator >= 1.0 - 3.0 * margin.expansion - 1e-6

    # And the scan covers every nonlinear component, transverse coupling included: with
    # all three live, Ex sees Dz only through the Yee average (Dsqr = 0.0625*(4*3)^2 = 9)
    # and its c2 is zero, which makes it the smallest denominator on the grid.
    fields.set_nonlinear_volumes({name: 0.02 for name in ("Ex", "Ey", "Ez")},
                                 {name: 0.05 for name in ("Ex", "Ey", "Ez")})
    transverse = stepping.nonlinear_margin(fields)
    assert transverse.denominator == pytest.approx(1.0 + 3.0 * 9.0 * 0.05 * chi1inv ** 3, rel=1e-5)
    assert transverse.denominator < margin.denominator


@pytest.mark.parametrize("case", ["x_even", "y_odd", "z_even", "xy_mixed", "xyz_odd"])
def test_a_folded_chi3_run_reproduces_the_full_domain_run_bit_for_bit(case):
    """The fold equivalence, with a live chi3 — the same zero-tolerance bar.

    The nonlinearity is the first part of the constitutive update that is NOT
    pointwise: MEEP's ``Dsqr`` averages the two transverse D components onto the
    updated component's position, so it reads a cell across the mirror plane and a
    cell past the far face. Every other kernel in this file was proved fold-safe
    while ``update_E`` read nothing at all, so that proof does not carry over and the
    equivalence has to be re-established here.

    chi3 alone, because chi3 alone is symmetry-compatible: ``c3`` carries ``Dsqr``
    and is even whatever parity the plane gives D, so ``u`` is even and the recovered
    E keeps D's parity exactly. chi2 does not, and is refused at installation rather
    than folded — see
    :func:`test_a_chi2_medium_is_refused_on_a_plane_that_makes_its_component_odd`.

    Same harness as the linear case above: a compact blob projected onto the mirror
    subspace, few enough steps that the far face stays dark, and equality at zero
    tolerance because every operation involved is +, - and * on real coefficients.
    """
    planes = FOLD_CASES[case]
    # A wider cell and fewer steps than the linear case: the transverse Yee average
    # reaches one cell further per step in each direction, so the disturbance from a
    # radius-2 blob covers ~2 cells per step rather than 1 and the far face goes live
    # sooner. That is checked below rather than assumed.
    cell_size = (2.4, 2.4, 2.4)
    steps = 4
    full_grid = Grid(resolution=10.0, cell_size=cell_size)
    folded_grid = Grid(resolution=10.0, cell_size=cell_size, symmetry=_symmetry_argument(planes))
    full = Fields(grid=full_grid, force_complex_fields=True)
    folded = Fields(grid=folded_grid, force_complex_fields=True)
    for container in (full, folded):
        container.set_background_eps(2.25)
        container.set_nonlinear_volumes({name: 0.0 for name in ("Ex", "Ey", "Ez")},
                                        {name: 5e-3 for name in ("Ex", "Ey", "Ez")})
        container.enable_field_storage()
        assert container.has_nonlinearity

    blob = _compact_blob(full_grid.shape, planes)
    for component in PRIMARY_COMPONENTS:
        seeded = _symmetrize(blob, component, planes)
        getattr(full, component)[...] = seeded
        getattr(folded, component)[...] = _fold(seeded, planes, full_grid.shape_full)

    for _ in range(steps):
        _step_once(full, None)
        _step_once(folded, None)

    # Checked FIRST, and over every primary component: this is the precondition the
    # comparison rests on, and a stencil that reached further would otherwise surface
    # as an unexplained mismatch instead of as the run outgrowing its cell.
    for axis in planes:
        for component in PRIMARY_COMPONENTS:
            far_face = numpy.asarray(getattr(folded, component))[_slab(axis, -1)]
            assert numpy.count_nonzero(far_face) == 0, (
                f"{case}: the far {'xyz'[axis]} face of {component} is live, so the two runs no "
                f"longer terminate the same system and this equivalence proves nothing")

    for component in PRIMARY_COMPONENTS + ("Ex", "Ey", "Ez"):
        reference = _fold(numpy.asarray(getattr(full, component)), planes, full_grid.shape_full)
        numpy.testing.assert_array_equal(
            numpy.asarray(getattr(folded, component)), reference,
            err_msg=f"{case}: folded nonlinear {component} differs from the full-domain quadrant")
    # Positive control: the nonlinearity must actually have moved the field, or this
    # is the linear fold test wearing a different name.
    linear = Fields(grid=full_grid, force_complex_fields=True)
    linear.set_background_eps(2.25)
    for component in PRIMARY_COMPONENTS:
        getattr(linear, component)[...] = _symmetrize(blob, component, planes)
    for _ in range(steps):
        _step_once(linear, None)
    moved = float(numpy.max(numpy.abs(numpy.asarray(full.Dz) - numpy.asarray(linear.Dz))))
    assert moved > 0.0, f"{case}: the chi2/chi3 pair changed nothing, so the fold proves nothing"


@pytest.mark.parametrize("phase, refused, allowed", [(+1, "Ex", "Ey"), (-1, "Ey", "Ex")])
def test_a_chi2_medium_is_refused_on_a_plane_that_makes_its_component_odd(phase, refused, allowed):
    """A chi2 medium is not centrosymmetric, so a mirror plane is not a symmetry of it.

    ``c2 = D_c * chi2 * chi1inv^2`` is LINEAR in D and so carries D's parity, while
    everything else in the Pade factor is even. A plane that makes ``D_c`` odd
    therefore makes ``u`` neither even nor odd, and the folded half is reconstructed
    from a symmetry the material does not have — quietly, at the 4e-02 level after a
    single step.

    The rule is the parity and not the axis, which is why both phases are checked and
    why they swap which component is refused: an EVEN x plane makes Ex odd and leaves
    Ey/Ez even, an ODD one does the reverse. chi3 is accepted under both, because
    ``Dsqr`` is even whatever D does.
    """
    grid = Grid(resolution=10.0, cell_size=(1.2, 0.6, 0.6), symmetry=(Mirror("X", phase),))
    fields = Fields(grid=grid, force_complex_fields=True)
    zero = {name: 0.0 for name in ("Ex", "Ey", "Ez")}

    with pytest.raises(ValueError, match="centrosymmetric"):
        fields.set_nonlinear_volumes({refused: 1e-3}, zero)
    assert not fields.has_nonlinearity, "a refused install must leave the material linear"

    # The component the same plane leaves EVEN takes a chi2 without complaint, and
    # chi3 is accepted on every component: this is a parity rule, not a blanket ban.
    fields.set_nonlinear_volumes({allowed: 1e-3}, zero)
    assert fields.nonlinear_components == (allowed,)
    fields.set_nonlinear_volumes(zero, {name: 5e-3 for name in ("Ex", "Ey", "Ez")})
    assert fields.nonlinear_components == ("Ex", "Ey", "Ez")

    # An identically zero chi2 is not a chi2 and must not be refused, or a caller who
    # spells out the pair explicitly could not fold at all.
    fields.set_nonlinear_volumes({refused: 0.0}, {refused: 5e-3})
    assert fields.nonlinear_components == (refused,)


# --- Metallic (PEC) outer boundaries ------------------------------------------------------


def make_metallic_fields(cell_size=(0.6, 0.7, 0.6), boundaries="metallic", symmetry=(),
                         resolution=10.0, complex_fields=True) -> Fields:
    grid = Grid(resolution=resolution, cell_size=cell_size, symmetry=symmetry,
                boundaries=boundaries)
    return Fields(grid=grid, force_complex_fields=complex_fields)


def test_boundary_kinds_resolve_per_axis_with_the_fold_outranking_the_declaration():
    plain = Grid(resolution=10.0, cell_size=(0.6, 0.7, 0.6))
    assert stepping._boundary_kinds(plain, None) == (stepping.PERIODIC,) * 3
    walled = Grid(resolution=10.0, cell_size=(0.6, 0.7, 0.6), boundaries="metallic")
    assert stepping._boundary_kinds(walled, None) == (stepping.METALLIC,) * 3
    # Per axis, and coexisting: a run may wall one axis and repeat the others.
    mixed = Grid(resolution=10.0, cell_size=(0.6, 0.7, 0.6), boundaries={"y": "metallic"})
    assert stepping._boundary_kinds(mixed, None) == (
        stepping.PERIODIC, stepping.METALLIC, stepping.PERIODIC)
    # A folded axis resolves to MIRROR whatever it declares, because the stored
    # quadrant's near face is the plane and its far face already wears the zero ghost
    # a PEC wall has. The declaration still shows through Grid.is_metallic.
    folded = Grid(resolution=10.0, cell_size=(0.6, 0.7, 0.6), symmetry=("X",),
                  boundaries="metallic")
    assert stepping._boundary_kinds(folded, None) == (
        stepping.MIRROR, stepping.METALLIC, stepping.METALLIC)
    assert folded.is_metallic(0) is True


def expected_after_wall_clear(values, grid, component, walled_axes):
    """``values`` with cell 0 of every walled axis the component sits ON set to zero."""
    expected = values.copy()
    for axis in walled_axes:
        if IYEE_SHIFTS[component][axis] == 0:
            expected[stepping._face(axis, 0)] = 0
    return expected


def test_the_curl_mask_drops_cell_zero_of_a_metallic_axis_only_for_the_components_on_the_wall():
    grid = Grid(resolution=10.0, cell_size=(0.6, 0.7, 0.6), boundaries={"y": "metallic"})
    for component, shifts in IYEE_SHIFTS.items():
        curl = numpy.ones((6, 7, 6), dtype=numpy.complex64)
        stepping._mask_non_owned_cells(curl, grid, shifts)
        # Tangential E/D and normal H/B are the components a y wall shorts out, and
        # they are exactly the ones whose y Yee shift is 0.
        expected = expected_after_wall_clear(
            numpy.ones((6, 7, 6), dtype=numpy.complex64), grid, component, (1,))
        assert numpy.array_equal(curl, expected), component
        # The periodic sibling axes of the same grid mask nothing at all, so a
        # component off the y wall keeps every cell.
        assert bool(numpy.all(curl == 1)) is (shifts[1] == 1), component


def test_zero_metal_clears_exactly_the_wall_planes_and_nothing_else():
    fields = make_metallic_fields(boundaries={"x": "metallic", "z": "metallic"})
    names = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")
    seed(fields, names, 11)
    before = snapshot(fields, names)
    stepping.zero_metal_D(fields)
    for name in names:
        # Only D is cleared here; B's walls are cleared inside step_B, where MEEP's
        # step_boundaries(B_stuff) runs, with no source injection in between.
        walled = (0, 2) if name.startswith("D") else ()
        expected = expected_after_wall_clear(before[name], fields.grid, name, walled)
        assert getattr(fields, name).tobytes() == expected.tobytes(), name
    # And the y axis, which is periodic in this run, keeps its cell 0 on every component.
    for name in names:
        assert numpy.array_equal(getattr(fields, name)[:, 0, 1:][1:],
                                 before[name][:, 0, 1:][1:]), name


def test_zero_metal_is_a_no_op_byte_for_byte_without_a_metallic_axis():
    # The periodic path must not pay for, or be perturbed by, a feature it does not use.
    for boundaries in (None, "periodic", {"x": "periodic"}):
        fields = make_metallic_fields(boundaries=boundaries)
        names = ("Dx", "Dy", "Dz", "Bx", "By", "Bz")
        seed(fields, names, 12)
        before = snapshot(fields, names)
        stepping.zero_metal_D(fields)
        for name in names:
            assert getattr(fields, name).tobytes() == before[name].tobytes(), name


def test_zero_metal_leaves_a_folded_metallic_axis_alone_because_its_cell_zero_is_the_mirror_ghost():
    """The regression that measured 1.28e+00 against CPU MEEP before it was caught.

    ``grid_volume::halve`` puts a folded axis's origin at ``io = -2`` (vec.cpp), so its
    stored cell 0 is one full cell BELOW the mirror plane and carries the
    parity-weighted ghost ``fill_symmetry_bc_D`` writes there — not a wall. The axis's
    real walls are the plane (not a wall at all) and the far face at L/2, which the
    fold does not store and ``_shift_up`` supplies as zero. Clearing cell 0 there
    destroys the fold, and the run still completes and still looks like a field.
    """
    fields = make_metallic_fields(cell_size=(0.6, 0.7, 0.6), symmetry=("X",),
                                  boundaries="metallic")
    names = ("Dx", "Dy", "Dz")
    seed(fields, names, 13)
    before = snapshot(fields, names)
    stepping.zero_metal_D(fields)
    for name in names:
        # Only the two UNFOLDED metallic axes are cleared; the folded x axis is skipped
        # entirely, so its cell 0 keeps whatever the fold put there.
        expected = expected_after_wall_clear(before[name], fields.grid, name, (1, 2))
        assert getattr(fields, name).tobytes() == expected.tobytes(), name
    # The statement that matters, spelled out: Dy and Dz have x shift 0 and would have
    # been cleared on x had the fold not been checked — and their cell 0 is untouched.
    for name in ("Dy", "Dz"):
        assert IYEE_SHIFTS[name][0] == 0
        assert float(numpy.abs(getattr(fields, name)[0]).max()) > 0.0, name


def test_a_metallic_wall_stays_exactly_zero_through_a_run_that_drives_it():
    """A driven run leaves every wall sample at exactly zero, and a seeded one is wiped.

    THE WIPE IS REACHED THROUGH ``set_field``, not through a source, and that is
    measured rather than assumed. A source ASKED FOR on the wall never lands there:
    ``sources._build_source_points`` clips a non-wrapping axis to MEEP's own owned
    range (``little_owned_corner0(c) = little_corner + 2 - iyee_shift(c)``, i.e. the
    shift-0 components start one cell in), so the deposit is dropped before injection.
    MEEP reaches the same answer by the other route — it owns the HIGH wall, deposits
    there, and ``zero_metal`` wipes it — and the engine's clip and MEEP's wipe agree.
    A test that puts a source on the wall therefore passes with ``zero_metal_B``
    deleted, which is exactly what a mutation run found.

    What does reach the wall is a seeded primary field: ``set_field`` writes wherever
    it is told, and the curl mask then FREEZES that value (an unowned cell integrates a
    zero curl), so a stale wall sample would persist for the whole run and radiate —
    ``_curl`` reads the unshifted partner array, so cell 0 of a shift-0 component feeds
    the target next to it every step. The two halves below seed D and B respectively
    and check the run comes back clean.
    """
    from .driver import FdtdDriver

    def build(seed_component=None):
        driver = FdtdDriver(cell_size=(0.6, 0.7, 1.0), resolution=10, force_complex_fields=True,
                            boundaries="metallic")
        if seed_component is not None:
            hot = numpy.zeros(driver.shape, dtype=numpy.complex64)
            hot[0] = 1.0 + 0.5j  # The whole low x wall plane, which a PEC shorts out.
            driver.set_field(seed_component, hot)
            assert float(numpy.abs(numpy.asarray(
                driver.get_field(seed_component, cell_centered=False))[0]).max()) > 0.0, (
                "the seed did not reach the wall plane, so this measures nothing")
        return driver

    def finish(driver):
        stored = {name: numpy.asarray(driver.get_field(name, cell_centered=False))
                  for name in ("Dy", "Dz", "Bx", "Ey", "Ez", "Hx")}
        driver.close()
        return stored

    # 1. A seeded D wall plane is wiped by zero_metal_D: exactly zero after the run,
    #    where the curl mask alone would FREEZE the seed there and re-radiate it every
    #    step. (One step of its influence does escape before the first wipe, in this
    #    engine and in MEEP alike — step_db runs before step_boundaries — so the
    #    statement is about the wall plane, not about the volume being untouched.)
    seeded_d = build("Dz")
    seeded_d.run(num_steps=40)
    result = finish(seeded_d)
    assert float(numpy.abs(result["Dz"][0]).max()) == 0.0, (
        "a D value seeded onto a perfect conductor survived the run on the wall plane, "
        "where the curl mask freezes it and it re-radiates every step")

    # 2. The same for a seeded B wall plane, which is the OTHER pass — zero_metal_B,
    #    after the magnetic sources rather than after the electric ones.
    seeded_b = build("Bx")
    seeded_b.run(num_steps=40)
    result = finish(seeded_b)
    assert float(numpy.abs(result["Bx"][0]).max()) == 0.0, (
        "a B value seeded onto a perfect conductor survived the run on the wall plane")

    # 3. The control: seeding one cell IN instead of on the wall does drive the run, so
    #    the two assertions above are the wall and not a seed that never worked.
    live = FdtdDriver(cell_size=(0.6, 0.7, 1.0), resolution=10, force_complex_fields=True,
                      boundaries="metallic")
    hot = numpy.zeros(live.shape, dtype=numpy.complex64)
    hot[1] = 1.0 + 0.5j
    live.set_field("Dz", hot)
    live.run(num_steps=40)
    assert float(numpy.abs(numpy.asarray(live.get_field("Dz", cell_centered=False))).max()) > 0.0
    live.close()

    # 4. A driven run: every sample that lies ON a wall is exactly zero and every sample
    #    half a cell off one is free, on all three axes at once.
    driven = build()
    driven.add_source({"component": "Ez", "frequency": 1.0, "center": (0.05, 0.05, -0.15),
                       "size": (0.0, 0.0, 0.0)})
    driven.add_source({"component": "Hx", "frequency": 1.0, "center": (-0.25, 0.05, -0.15),
                       "size": (0.0, 0.0, 0.0)})
    driven.run(num_steps=40)
    result = finish(driven)
    assert float(numpy.abs(result["Ez"]).max()) > 0.0
    for name, stored in result.items():
        for axis in range(3):
            on_wall = IYEE_SHIFTS[name][axis] == 0  # Tangential E/D, normal H/B.
            plane = stored[stepping._face(axis, 0)]
            assert bool(numpy.all(plane == 0)) is on_wall, (
                f"{name} on the low {'xyz'[axis]} wall: shift {IYEE_SHIFTS[name][axis]}")


def test_a_source_asked_for_on_a_metallic_wall_deposits_nothing():
    """The clip, stated separately from the wipe, because they are different mechanisms.

    ``sources._build_source_points`` clips a NON-WRAPPING axis to MEEP's owned range,
    and ``Grid.axis_wraps`` now reports a metallic axis as non-wrapping — so a current
    asked for on the low wall is dropped before it is ever injected, and never reaches
    ``zero_metal_D`` at all. MEEP arrives at the same field by the other route: it owns
    the HIGH wall of a shift-0 component, deposits there, and wipes it in
    ``step_boundaries``. Both give a run driven by nothing.

    Worth its own test because the two mechanisms are separately deletable and the
    engine would look identical with either one gone — until a seeded field or a MEEP
    comparison at the far wall says otherwise.
    """
    from .driver import FdtdDriver

    def peak(component, centre):
        driver = FdtdDriver(cell_size=(0.6, 0.7, 1.0), resolution=10,
                            force_complex_fields=True, boundaries="metallic")
        driver.add_source({"component": component, "frequency": 1.0, "center": centre,
                           "size": (0.0, 0.0, 0.0)})
        driver.run(num_steps=30)
        value = float(numpy.abs(numpy.asarray(
            driver.get_field("Ez", cell_centered=False))).max())
        driver.close()
        return value

    wall = -0.3  # x = -L/2 for a 0.6-wide cell: the low x wall, an Ez/Hx sample plane.
    assert peak("Ez", (wall, 0.05, -0.15)) == 0.0
    assert peak("Hx", (wall, 0.05, -0.15)) == 0.0
    # One cell in, the identical request drives the run.
    assert peak("Ez", (wall + 0.1, 0.05, -0.15)) > 0.0
    assert peak("Hx", (wall + 0.05, 0.05, -0.15)) > 0.0


def test_a_bloch_phase_on_a_metallic_axis_is_refused_by_the_stencils_too():
    # Grid refuses the pair at construction; the kernels refuse it again on their own
    # inputs, because a stub grid or a hand-built boundary tuple can reach them
    # directly, and the message must name the wall rather than a mirror plane.
    grid = Grid(resolution=10.0, cell_size=(0.6, 0.7, 0.6), boundaries={"x": "metallic"},
                k_point=(0.0, 0.25, 0.0))
    sample = numpy.zeros((6, 7, 6), dtype=numpy.complex64)
    boundaries = (stepping.METALLIC, stepping.METALLIC, stepping.PERIODIC)
    with pytest.raises(ValueError, match="perfect-electric-conductor wall terminates") as raised:
        stepping._bloch_phases(grid, boundaries, sample, None)
    assert "mirror plane" not in str(raised.value)


@pytest.mark.parametrize("cell_size", [(0.6, 0.7, 0.6), (0.5, 0.7, 0.9)])  # Even and odd counts.
def test_a_metallic_run_is_byte_identical_to_the_same_run_spelled_with_pairs(cell_size):
    # Five spellings of one wall must step to the same bytes, or the spelling is a
    # numerical choice rather than a notation.
    from .driver import FdtdDriver

    digests = []
    for spelling in ("metallic", ("metallic",) * 3, {"x": "metallic", "y": "metallic",
                                                     "z": "metallic"},
                     [("metallic", "metallic")] * 3, (("metallic", "metallic"),) * 3):
        driver = FdtdDriver(cell_size=cell_size, resolution=10, force_complex_fields=True,
                            boundaries=spelling)
        driver.add_source({"component": "Ez", "frequency": 1.0, "center": (0.03, -0.05, 0.07),
                           "size": (0.0, 0.0, 0.0)})
        driver.run(num_steps=30)
        digests.append(b"".join(
            numpy.asarray(driver.get_field(name, cell_centered=False)).tobytes()
            for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz")))
        driver.close()
    assert len(set(digests)) == 1


# Raw-byte pins for the periodic and Bloch paths. These are the configurations every
# recorded CPU-MEEP floor was measured on, and the metallic boundary was added
# underneath them: an A/B against a copy of this package with the feature mechanically
# reverted found all ten cases below byte-identical, and these digests are that result
# frozen. A change here is a change to the periodic arithmetic, which invalidates the
# floors in `driver.py`'s docstrings — not something to re-bless without re-measuring.
_PERIODIC_BYTE_PINS = {
    "plain_periodic": "584a248b233696b5",
    "plain_periodic_real": "2323a3382c1274f9",
    "bloch": "bb1a4354ea455b75",
    "bloch_bz_edge": "c8795539e853c359",
    "pml_uniform": "00fad1285c52ce41",
    "pml_z_bloch": "a3f82ce820ed28f9",
    "symmetry_X": "5ac1e340782e43c5",
    "symmetry_XY_pml": "c8e36362cd00a535",
    "dispersive": "785680b2d022d8fb",
    "odd_counts": "7e83d523eb295e28",
}

_PERIODIC_BYTE_CASES = {
    "plain_periodic": dict(cell=(1.0, 1.1, 2.0), src=(0.03, -0.05, -0.35), steps=40, complex=True),
    "plain_periodic_real": dict(cell=(1.0, 1.1, 2.0), src=(0.03, -0.05, -0.35), steps=40,
                                complex=False),
    "bloch": dict(cell=(1.0, 1.0, 2.0), src=(0.05, 0.05, -0.35), steps=40, complex=True,
                  k_point=(0.1234567, -0.3, 0.0)),
    "bloch_bz_edge": dict(cell=(1.0, 1.0, 2.0), src=(0.05, 0.05, -0.35), steps=40, complex=True,
                          k_point=(0.5, 0.0, 0.0)),
    "pml_uniform": dict(cell=(1.4, 1.4, 2.4), src=(0.05, -0.05, -0.35), steps=40, complex=True,
                        pml=4),
    "pml_z_bloch": dict(cell=(1.0, 1.0, 2.4), src=(0.05, 0.05, -0.35), steps=40, complex=True,
                        k_point=(0.3, 0.0, 0.0), pml={"z": 4}),
    "symmetry_X": dict(cell=(4.0, 1.0, 2.0), src=(0.0, -0.05, -0.35), steps=12, complex=True,
                       symmetry=("X",)),
    "symmetry_XY_pml": dict(cell=(2.0, 2.0, 2.4), src=(0.0, 0.0, -0.35), steps=20, complex=True,
                            symmetry=("X", "Y"), pml={"x": (0, 4), "y": (0, 4), "z": 4}),
    "dispersive": dict(cell=(1.0, 1.0, 2.0), src=(0.05, 0.05, -0.35), steps=40, complex=True,
                       eps=2.0, susceptibility=True),
    "odd_counts": dict(cell=(1.1, 0.9, 1.5), src=(0.03, -0.05, -0.25), steps=40, complex=True),
}


@pytest.mark.parametrize("name", sorted(_PERIODIC_BYTE_CASES))
def test_the_periodic_and_bloch_paths_are_pinned_on_raw_bytes(name):
    import hashlib

    from .driver import FdtdDriver

    case = _PERIODIC_BYTE_CASES[name]
    kwargs = dict(cell_size=case["cell"], resolution=10, force_complex_fields=case["complex"])
    if case.get("k_point"):
        kwargs["k_point"] = case["k_point"]
    if case.get("symmetry"):
        kwargs["symmetry"] = case["symmetry"]
    driver = FdtdDriver(**kwargs)
    if case.get("pml"):
        driver.setup_pml(case["pml"])
    if case.get("eps"):
        driver.set_epsilon(numpy.full(driver.shape, case["eps"], dtype=numpy.float32))
    if case.get("susceptibility"):
        driver.add_susceptibility(
            Susceptibility(frequency=1.1, gamma=0.05, kind=LORENTZIAN), 0.4)
    driver.add_source({"component": "Ez", "frequency": 1.0, "center": case["src"],
                       "size": (0.0, 0.0, 0.0)})
    driver.run(num_steps=case["steps"])
    payload = b"".join(
        numpy.asarray(driver.get_field(component, cell_centered=False)).tobytes()
        for component in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"))
    driver.close()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    assert digest == _PERIODIC_BYTE_PINS[name], (
        f"The {name} run's raw field bytes changed (sha256[:16] {digest}, expected "
        f"{_PERIODIC_BYTE_PINS[name]}). Every CPU-MEEP floor this package records was "
        f"measured on these configurations; a change here means the periodic / Bloch "
        f"arithmetic moved and the floors have to be re-measured before this is re-blessed."
    )


# --------------------------------------------------------------------------------------
# BFAST — MEEP's broadband fixed-angle source technique (``bfast_scaled_k``).
#
# The whole feature is one extra additive pass inside step_db (step_db.cpp:129-142 ->
# step_generic.cpp:335-471). It touches nothing else: grep for "bfast" over MEEP's src/
# finds it in the curl sub-step, in the flux backup/restore pair, in the dump/load state
# list and in the CW solver, and NOWHERE in sources.cpp, structure.cpp, boundaries.cpp,
# step.cpp, update_eh, update_pols or dft.cpp.
#
# These tests pin the four things that can be got wrong silently — which k multiplies
# which operand, the D-side sign flip, the SUM (not the curl's difference), and the IIR
# recursion's shape — plus the two properties that are easy to "fix" into something
# wrong: the filter is a bilinear derivative, and it is marginally stable.
# --------------------------------------------------------------------------------------


def _bfast_fields(k, cell_size=(0.6, 0.7, 0.8), resolution=10.0, complex_fields=True,
                  dimensions=3) -> Fields:
    grid = Grid(resolution=resolution, cell_size=cell_size, bfast_scaled_k=k,
                dimensions=dimensions)
    return Fields(grid=grid, force_complex_fields=complex_fields)


def _bfast_reference_sum(fields: Fields, term, k1: float, k2: float, backward: bool):
    """S = k1*(g1[i+s] + g1[i]) - k2*(g2[i+s] + g2[i]), by explicit index.

    Written against :func:`reference_curl`'s own loop so it shares nothing with the
    slicing under test: same wrap, same stride, SUMS where the curl takes differences.
    """
    snapshot_arrays = {name: stepping._read_component(fields, name)
                       for name in (term.first, term.second)}
    first = snapshot_arrays[term.first]
    second = snapshot_arrays[term.second]
    stride = -1 if backward else 1
    out = numpy.zeros(first.shape, dtype=first.dtype)
    for i in range(first.shape[0]):
        for j in range(first.shape[1]):
            for k in range(first.shape[2]):
                cell = (i, j, k)
                up_first = neighbour(cell, term.first_axis, stride, first.shape)
                up_second = neighbour(cell, term.second_axis, stride, second.shape)
                out[cell] = (k1 * (first[up_first] + first[cell])
                             - k2 * (second[up_second] + second[cell]))
    return out


# The k index each operand carries, read off step_db.cpp:129-136 with
# component_index (vec.hpp:445) = the component's OWN direction. `first` is MEEP's
# g1 = f_p and `second` its g2 = f_m, so k1 (on g1) is indexed by the SECOND
# component and k2 (on g2) by the FIRST. Written out per term rather than derived,
# so a rule change has to disagree with a table someone can check against the source.
_BFAST_K_INDICES = {
    # target: (index of k1 = the k multiplying `first`, index of k2 = the k on `second`)
    "Bx": (1, 2),  # first Ez, second Ey -> k1 = k_y on Ez, k2 = k_z on Ey: (k x E)_x
    "By": (2, 0),  # first Ex, second Ez -> k1 = k_z on Ex, k2 = k_x on Ez: (k x E)_y
    "Bz": (0, 1),  # first Ey, second Ex -> k1 = k_x on Ey, k2 = k_y on Ex: (k x E)_z
    "Dx": (1, 2),  # first Hz, second Hy
    "Dy": (2, 0),  # first Hx, second Hz
    "Dz": (0, 1),  # first Hy, second Hx
}


def test_bfast_k_is_indexed_by_the_components_own_direction_not_the_derivative():
    # THE subtlest choice in the pass, and the one MEEP annotated in place ("puts k1
    # in direction of g2", step_db.cpp:131/:133). Indexing by the DERIVATIVE direction
    # instead — the intuitive reading, since every other per-axis quantity in the curl
    # is indexed that way — swaps k1 with k2 and turns the cross product into its
    # transpose. It stays smooth, finite and complete.
    k = (0.31, -0.47, 0.19)
    fields = _bfast_fields(k)
    for terms, magnetic in ((stepping.B_CURL_TERMS, True), (stepping.D_CURL_TERMS, False)):
        for term in terms:
            first_index, second_index = _BFAST_K_INDICES[term.target]
            # `first` is g1 and carries k1, which step_db indexes by c_m = `second`.
            assert first_index == stepping._bfast_axis(term.second), term.target
            assert second_index == stepping._bfast_axis(term.first), term.target
            # The two k's are DIFFERENT components on every term, so interchanging
            # them — the plausible slip, and the reason MEEP annotated the pair in
            # place — is always observable rather than a no-op somewhere.
            assert first_index != second_index, term.target
            # And an identity worth stating rather than relying on: the three
            # directions of a curl term are a permutation of x, y, z, so the OTHER
            # partner's own direction IS this partner's derivative direction. Reading
            # `bfast_scaled_k[component_index(c_m)]` and reading it at the plus
            # partner's derivative axis are therefore the same index — which is why
            # the only distinguishable mistake here is the k1/k2 interchange above.
            assert first_index == term.first_axis and second_index == term.second_axis

    # Component by component, the six S values ARE the two cross products.
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 7)
    electric = numpy.stack([stepping._read_component(fields, name)
                            for name in ("Ex", "Ey", "Ez")])
    # (k x E)_x = k_y E_z - k_z E_y, evaluated on the un-averaged field: the engine's
    # S is twice the two-point average of exactly this, so with a CONSTANT field the
    # average is the field and S is 2*(k x E).
    constant = _bfast_fields(k)
    for name, value in zip(("Ex", "Ey", "Ez"), (0.5, -1.25, 2.0)):
        # Seed B's partners through D so get_E returns them (vacuum, inv_eps = 1).
        getattr(constant, "D" + name[-1])[...] = value
    cross_e = numpy.cross(numpy.array(k), numpy.array([0.5, -1.25, 2.0]))
    for term, expected in zip(stepping.B_CURL_TERMS, cross_e):
        k1 = k[stepping._bfast_axis(term.second)]
        k2 = k[stepping._bfast_axis(term.first)]
        operands = stepping._curl_operands(
            numpy, stepping._component_snapshot(constant, ("Ex", "Ey", "Ez")), term,
            (stepping.PERIODIC,) * 3, (None, None, None), (None, None, None),
            backward=False)
        total = (k1 * (operands.shifted_first + operands.first)
                 - k2 * (operands.shifted_second + operands.second))
        assert numpy.allclose(total, 2.0 * expected), (term.target, expected)
    assert electric.shape[0] == 3  # the seeded read above actually produced three arrays


def test_bfast_d_side_negates_both_k_and_the_b_side_does_not():
    # step_db.cpp:137-140, `if (ft == D_stuff) { k1 = -k1; k2 = -k2; }`. This is the
    # sign that turns +d/dt(k x E) on the B side into -d/dt(k x H) on the D side; drop
    # it and the shear runs the wrong way for half the fields, which is a complete,
    # smooth run at the wrong incidence angle rather than a blow-up.
    k = (0.4, 0.0, 0.0)
    fields = _bfast_fields(k)
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 11)
    snapshots = {name: stepping._read_component(fields, name)
                 for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}

    magnetic_terms = {term.target: term for term in stepping.B_CURL_TERMS}
    electric_terms = {term.target: term for term in stepping.D_CURL_TERMS}
    # By <- Ez and Dz <- Hy are the two terms a k along x makes live, and they are the
    # two the corpus row (test_refl_angular, P polarization in the XZ plane) exercises.
    b_term = magnetic_terms["By"]
    d_term = electric_terms["Dz"]

    b_operands = stepping._curl_operands(numpy, snapshots, b_term, (stepping.PERIODIC,) * 3,
                                         (None, None, None), (None, None, None), backward=False)
    b_increment = stepping._bfast_term(fields, b_term, b_operands, magnetic=True)
    # First application from a zeroed state: advance = S - 0, returned negated.
    expected_b = -_bfast_reference_sum(fields, b_term, 0.0, k[0], backward=False)
    assert numpy.allclose(b_increment, expected_b, atol=1e-6)

    d_operands = stepping._curl_operands(numpy, snapshots, d_term, (stepping.PERIODIC,) * 3,
                                         (None, None, None), (None, None, None), backward=True)
    d_increment = stepping._bfast_term(fields, d_term, d_operands, magnetic=False)
    expected_d = -_bfast_reference_sum(fields, d_term, -k[0], 0.0, backward=True)
    assert numpy.allclose(d_increment, expected_d, atol=1e-6)
    # The two are not the same expression with the same sign: negating one k pair and
    # not the other is exactly what the flip does.
    assert not numpy.allclose(
        d_increment, -_bfast_reference_sum(fields, d_term, k[0], 0.0, backward=True),
        atol=1e-6)


def test_bfast_state_advances_by_the_tustin_recursion_not_a_backward_difference():
    # F_n = S_n - F_{n-1} with output F_n - F_{n-1}. Dropping the `- F_{n-1}` from the
    # state update (the shape of MEEP's own inconsistency at step_generic.cpp:376) turns
    # the bilinear derivative into a backward difference at twice the amplitude — a
    # smooth, stable, wrong run.
    k = (0.35, 0.0, 0.0)
    fields = _bfast_fields(k)
    term = {t.target: t for t in stepping.B_CURL_TERMS}["By"]
    state = fields.f_bfast_By
    assert state is not None and numpy.all(state == 0)

    history = []
    for step in range(4):
        seed(fields, ("Dz",), 100 + step)
        snapshots = stepping._component_snapshot(fields, ("Ex", "Ey", "Ez"))
        operands = stepping._curl_operands(numpy, snapshots, term, (stepping.PERIODIC,) * 3,
                                           (None, None, None), (None, None, None),
                                           backward=False)
        previous = state.copy()
        total = _bfast_reference_sum(fields, term, 0.0, k[0], backward=False)
        increment = stepping._bfast_term(fields, term, operands, magnetic=True)
        # State: F_new = S - F_prev, exactly.
        assert numpy.allclose(state, total - previous, atol=1e-6)
        # Increment (in curl sign convention): -(F_new - F_prev) = -(S - 2*F_prev).
        assert numpy.allclose(increment, -(total - 2.0 * previous), atol=1e-6)
        history.append(float(numpy.abs(state).max()))
    assert min(history) > 0.0  # the state is genuinely live, not a zero that agrees


def test_bfast_iir_is_a_bilinear_derivative_and_its_homogeneous_mode_never_decays():
    # Two properties of F_n = S_n - F_{n-1}, on the scalar recursion alone, so nothing
    # about the grid can mask them.
    #
    # (1) With S = 2*A, the output F_n - F_{n-1} is dt * dA/dt. That is the whole
    #     reason the pass carries no explicit dt and no dtdx: the bilinear operator
    #     (1 - z^-1)/(1 + z^-1) supplies the dt/2 itself.
    #     The drive turns on from a genuinely flat start (both A and A' are ~1e-11 at
    #     t = 0) so the alternating homogeneous mode of (2) is not excited; excite it
    #     and the increments straddle the derivative instead of tracking it, which is
    #     a property of the initial condition, not of the identity.
    dt = 0.005
    times = numpy.arange(800) * dt
    amplitude = (numpy.sin(2.0 * numpy.pi * 0.7 * times)
                 * numpy.exp(-((times - 2.0) / 0.4) ** 2))
    state = 0.0
    increments = []
    for value in amplitude:
        previous = state
        state = 2.0 * value - previous
        increments.append(state - previous)
    increments = numpy.array(increments)
    reference = dt * numpy.gradient(amplitude, dt)
    interior = slice(5, -5)
    error = float(numpy.linalg.norm(increments[interior] - reference[interior])
                  / numpy.linalg.norm(reference[interior]))
    # Measured 3.5e-04 relative — np.gradient's own second-order floor on this signal.
    assert error < 3e-3, f"bilinear derivative identity off by relative {error:.3e}"

    #     Control: a BACKWARD DIFFERENCE of the same S — which is what dropping the
    #     `- F_prev` from the state update produces — is twice the derivative, so it
    #     is not accidentally the same quantity.
    backward = 2.0 * numpy.diff(amplitude, prepend=0.0)
    backward_error = float(numpy.linalg.norm(backward[interior] - reference[interior])
                           / numpy.linalg.norm(reference[interior]))
    assert backward_error > 0.9, f"the backward-difference control is only {backward_error:.3e} off"

    # (2) MARGINAL STABILITY. A unit perturbation with no drive alternates forever.
    #     Documented as an assertion so nobody later "stabilizes" the filter: MEEP has
    #     this property, and reproducing it is the requirement.
    undriven = 1.0
    trace = []
    for _ in range(64):
        undriven = 0.0 - undriven
        trace.append(undriven)
    assert trace[:4] == [-1.0, 1.0, -1.0, 1.0]
    assert abs(trace[-1]) == 1.0  # undamped after 64 steps, not merely at the start


def test_bfast_replaces_nothing_when_k_is_zero_and_allocates_nothing_either():
    # `use_bfast` is the disjunction of the three components (step_db.cpp:65), so an
    # all-zero k is not a cheap pass, it is NO pass: no allocation, no state, and a run
    # byte-identical to one built without the argument at all.
    plain = make_fields()
    zero_k = _bfast_fields((0.0, 0.0, 0.0))
    assert plain.grid.bfast_active is False and zero_k.grid.bfast_active is False
    for component in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        assert getattr(zero_k, "f_bfast_" + component) is None

    live = _bfast_fields((0.0, 0.0, 1e-9))  # ANY nonzero component turns it on.
    assert live.grid.bfast_active is True
    for component in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        array = getattr(live, "f_bfast_" + component)
        assert array is not None and array.shape == live.grid.shape
        assert numpy.all(array == 0)


def test_bfast_storage_exists_without_a_pml_and_without_a_conductivity():
    # MEEP's allocation test is `use_bfast && !f_bfast[cc][cmp]` and names neither PML
    # nor conductivity (step_db.cpp:76-79) — it sits BESIDE those two allocations, not
    # inside them. Gating ours on `enable_pml_storage` would silently drop the whole
    # term from every boundary-free BFAST run, which is a complete, smooth, wrong field.
    fields = _bfast_fields((0.5, 0.0, 0.0))
    assert fields.fu_Bx is None and fields.f_cond_Bx is None  # no PML, no conductivity
    assert fields.f_bfast_By is not None

    seed(fields, ("Dz",), 3)
    stepping.step_B(fields)
    assert float(numpy.abs(fields.f_bfast_By).max()) > 0.0
    # reset() must take the IIR state with it, or a re-run starts inside a history it
    # never lived through — and this recursion never forgets one.
    fields.reset()
    assert numpy.all(fields.f_bfast_By == 0)


def test_bfast_partner_on_an_invariant_axis_contributes_nothing():
    # MEEP's have_p / have_m (fields.cpp:428-455) are false when the derivative
    # direction is not one the grid resolves, and step_db.cpp:131-134 then forces that
    # k to zero. This is NOT the same thing the curl gets for free: an invariant axis
    # makes the curl's DIFFERENCE an exact zero on its own, but the BFAST SUM of the
    # same two samples is 2*g, so omitting the guard adds a term MEEP does not have.
    k = (0.4, 0.3, 0.2)
    two_d = _bfast_fields(k, cell_size=(0.6, 0.7, 0.0), dimensions=2)
    assert two_d.grid.is_invariant(2)
    seed(two_d, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 21)
    snapshots = stepping._component_snapshot(two_d, ("Ex", "Ey", "Ez"))
    # Bx: first = Ez on axis Y, second = Ey on axis Z. Z is invariant, so have_m is
    # false and k1 (the k on Ez) is zero; only the k2 term survives.
    term = {t.target: t for t in stepping.B_CURL_TERMS}["Bx"]
    operands = stepping._curl_operands(numpy, snapshots, term, (stepping.PERIODIC,) * 3,
                                       (None, None, None), (None, None, None), backward=False)
    increment = stepping._bfast_term(two_d, term, operands, magnetic=True)
    expected = -_bfast_reference_sum(two_d, term, 0.0, k[stepping._bfast_axis(term.first)],
                                     backward=False)
    assert numpy.allclose(increment, expected, atol=1e-6)
    # The unguarded reading is a different, nonzero field — the guard is load-bearing.
    unguarded = -_bfast_reference_sum(two_d, term, k[stepping._bfast_axis(term.second)],
                                      k[stepping._bfast_axis(term.first)], backward=False)
    assert not numpy.allclose(increment, unguarded, atol=1e-6)


def test_bfast_advance_is_masked_where_meeps_owned_loop_never_writes():
    # MEEP writes F only inside `sub_gv.little_owned_corner0(cc) .. big_corner()`, so
    # cell 0 of a shift-0 component on a folded axis keeps whatever it had. F has no
    # spatial stencil, so this cannot contaminate an owned cell — it is state fidelity,
    # and it is what makes a diagnostic read of f_bfast comparable with MEEP's.
    grid = Grid(resolution=10.0, cell_size=(1.2, 0.7, 0.8), symmetry=("X",),
                bfast_scaled_k=(0.0, 0.3, 0.0))
    fields = Fields(grid=grid, force_complex_fields=True)
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 31)
    # Bx has Yee shift 0 on X (IYEE_SHIFTS), so cell 0 of the folded axis is unowned;
    # k along y makes its k1 (the k on Ez) the live one.
    assert IYEE_SHIFTS["Bx"][0] == 0
    term = {t.target: t for t in stepping.B_CURL_TERMS}["Bx"]
    snapshots = stepping._component_snapshot(fields, ("Ex", "Ey", "Ez"))
    operands = stepping._curl_operands(numpy, snapshots, term, (stepping.PERIODIC,) * 3,
                                       (None, None, None), stepping._mirror_phases(grid),
                                       backward=False)
    stepping._bfast_term(fields, term, operands, magnetic=True)
    assert numpy.all(fields.f_bfast_Bx[0, :, :] == 0)
    assert float(numpy.abs(fields.f_bfast_Bx[1:, :, :]).max()) > 0.0


# --- the step-loop scratch pool ---------------------------------------------------
#
# `fields.StepScratch` reuses the step's full-volume temporaries instead of allocating
# them, which is worth 1.4x-1.9x on this path (see the class docstring). It buys that
# by handing the SAME buffer back for the same tag, so the whole correctness argument
# is that no two live temporaries ever share one — an aliasing mistake there does not
# crash, it differences a field against itself and leaves a smooth wrong answer. These
# tests hold the pooled path to BYTE equality with the allocating one, which is the
# only bar that catches it.

_SCRATCH_ARRAYS = (
    "Dx", "Dy", "Dz", "Bx", "By", "Bz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)


def _run_scratch_case(pooled: bool, *, grid_kwargs, pml_thickness, steps,
                      complex_fields=True, conductivity=None):
    """Step one configuration `steps` times, with the pool on or off, and return the bytes."""
    grid = Grid(resolution=10.0, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_fields)
    pml = PML(grid=grid, thickness=pml_thickness) if pml_thickness else None
    if pml is not None:
        fields.enable_pml_storage()
    if conductivity is not None:
        fields.set_d_conductivity(numpy.full(grid.shape, conductivity, dtype=numpy.float32))
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 17)
    if not pooled:
        fields.scratch = None  # The allocating path, unchanged from before the pool.
    for _ in range(steps):
        stepping.step_B(fields, pml)
        stepping.update_H(fields, pml)
        stepping.step_D(fields, pml)
        stepping.update_E(fields, pml)
    return {name: getattr(fields, name).tobytes()
            for name in _SCRATCH_ARRAYS if getattr(fields, name, None) is not None}


@pytest.mark.parametrize(
    "label,case",
    [
        ("pml_3d", dict(grid_kwargs=dict(cell_size=(0.8, 0.8, 0.8)), pml_thickness=2, steps=8)),
        ("no_pml", dict(grid_kwargs=dict(cell_size=(0.8, 0.8, 0.8)), pml_thickness=0, steps=8)),
        ("pml_2d", dict(grid_kwargs=dict(cell_size=(0.8, 0.8, 0.0), dimensions=2),
                        pml_thickness={"x": 2, "y": 2}, steps=8)),
        ("real_fields", dict(grid_kwargs=dict(cell_size=(0.8, 0.8, 0.8)), pml_thickness=2,
                             steps=8, complex_fields=False)),
        ("folded", dict(grid_kwargs=dict(cell_size=(0.8, 0.8, 0.8), symmetry=("X",)),
                        pml_thickness=2, steps=8)),
        ("metallic", dict(grid_kwargs=dict(cell_size=(0.8, 0.8, 0.8), boundaries="metallic"),
                          pml_thickness=2, steps=8)),
        ("conductive_pml", dict(grid_kwargs=dict(cell_size=(0.8, 0.8, 0.8)), pml_thickness=2,
                                steps=8, conductivity=0.4)),
        # Cylindrical is the RISKIEST pooling site and was the last one uncovered: it is
        # the only place a pooled buffer (`cyl_prefix`, returned as `prefix_ext`) is held
        # live across an entire curl term loop, and it stacks four pooled temporaries
        # (`cyl_extended`, `cyl_weighted`, `cyl_increment`, `cyl_prefix`) on two memoized
        # constants. m=0 and m=1 are both taken because m != 0 adds the i*m/r coupling,
        # whose factor is the memoized `cyl_imr` array — a key that forgot one of its
        # inputs would serve a stale factor, which is smooth, plausible and wrong.
        # No PML: every cylindrical pooled site is in the curl sub-step and is reached
        # without one, while the r axis's low side is the symmetry axis rather than a
        # face a layer can sit on.
        ("cylindrical_m0", dict(grid_kwargs=dict(cell_size=(0.8, 0.0, 0.8), cylindrical=True,
                                                 m=0), pml_thickness=0, steps=8)),
        ("cylindrical_m1", dict(grid_kwargs=dict(cell_size=(0.8, 0.0, 0.8), cylindrical=True,
                                                 m=1), pml_thickness=0, steps=8)),
    ],
)
def test_step_scratch_is_bit_identical(label, case):
    del label
    assert _run_scratch_case(True, **case) == _run_scratch_case(False, **case)


@pytest.mark.parametrize("shape", [(5,), (1, 7), (4, 3, 2), (2, 2, 2), (9, 1, 1), (1, 1, 6)])
@pytest.mark.parametrize("dtype", [numpy.float32, numpy.complex64])
def test_rolled_moves_exactly_the_bytes_numpy_roll_moves(shape, dtype):
    # `_rolled` replaces xp.roll on the stencil's hot path by writing roll's two slice
    # copies into a pooled buffer. That substitution is what the whole 1.4x rests on, so
    # it is pinned DIRECTLY here rather than only through the configurations
    # test_step_scratch_is_bit_identical happens to step: the pool-off path of that test
    # calls xp.roll, so it does compare the two, but only on the shapes and axes those
    # seven cases reach. A one-cell wrap face is exactly the kind of thing a later
    # rewrite gets subtly wrong on a length-1 axis.
    rng = numpy.random.default_rng(41)
    field = rng.standard_normal(shape).astype(numpy.float32)
    if dtype is numpy.complex64:
        field = (field + 1j * rng.standard_normal(shape)).astype(dtype)
    scratch = StepScratch(numpy)
    for axis in range(len(shape)):
        for shift in (-1, 1):
            expected = numpy.roll(field, shift, axis=axis)
            got = stepping._rolled(numpy, field, shift, axis, scratch, "probe")
            assert got.tobytes() == expected.tobytes(), (
                f"shape={shape} dtype={numpy.dtype(dtype).name} axis={axis} shift={shift}")


def test_rolled_refuses_a_shift_it_does_not_implement():
    # It carries the one-cell stencil shifts only; a silent wrong answer for shift=2
    # would be a smooth field, so the helper raises instead.
    scratch = StepScratch(numpy)
    with pytest.raises(ValueError, match="one-cell stencil shifts"):
        stepping._rolled(numpy, numpy.zeros((4, 4, 4), dtype=numpy.float32), 2, 0, scratch, "probe")


def test_step_scratch_never_hands_one_buffer_to_two_live_stencil_operands():
    # The tag rule stated as an assertion. `_curl_operands` holds both shifted partners
    # at once (the curl differences them, BFAST sums them), so they must not be the same
    # object — the failure this guards is silent: g1 - g1 is zero, everywhere, smoothly.
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 23)
    snapshots = stepping._component_snapshot(fields, ("Ex", "Ey", "Ez"))
    for term in stepping.B_CURL_TERMS:
        operands = stepping._curl_operands(
            numpy, snapshots, term, (stepping.PERIODIC,) * 3, (None, None, None),
            stepping._mirror_phases(grid), backward=False, scratch=fields.scratch)
        curl = stepping._curl_from_operands(operands, grid.dt / grid.dx,
                                            scratch=fields.scratch)
        live = (operands.first, operands.shifted_first,
                operands.second, operands.shifted_second, curl)
        assert len({id(array) for array in live}) == len(live)


def test_step_scratch_buffers_are_reused_rather_than_reallocated():
    # The point of the pool, stated so a future rewrite that quietly stops using it
    # fails here rather than only in a benchmark nobody runs.
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=2)
    seed(fields, ("Dx", "Dy", "Dz", "Bx", "By", "Bz"), 29)
    stepping.step_B(fields, pml)
    stepping.update_H(fields, pml)
    held = fields.scratch.bytes_held()
    identities = {id(slot) for slot in fields.scratch._slots.values()}
    assert held > 0
    for _ in range(4):
        stepping.step_B(fields, pml)
        stepping.update_H(fields, pml)
    assert fields.scratch.bytes_held() == held  # No new slot, no growth.
    assert {id(slot) for slot in fields.scratch._slots.values()} == identities
