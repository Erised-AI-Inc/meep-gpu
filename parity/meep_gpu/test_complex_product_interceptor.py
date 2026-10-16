"""The complex-product probe catches what it must, and the array path gives it nothing to catch.

Two halves (``complex_product_interceptor``, DESIGN D16):

* the probe itself: a planted ``real * complex64`` product is counted with its call site,
  additive complex arithmetic is traffic and not a violation, real arithmetic is neither,
  and every attribute it viewed is restored;
* the array path: NumPy-reference drivers on complex storage -- Bloch phases on one and on
  three axes, a PML on a phased axis, a mirror fold beside a phased axis, a uniform
  susceptibility, a D conductivity with a PML, a uniform chi3, BFAST, special-kz beta and
  cylindrical m = 1 -- step under the probe with 0 violations, and the probe saw complex
  traffic in each (so "0" is a reading, not an empty probe).

Host-only NumPy; no MEEP. Each run is a few steps of a small cell.
"""

from __future__ import annotations

import numpy
import pytest

from meep_gpu.dispersion import LORENTZIAN, Susceptibility
from meep_gpu.driver import FdtdDriver

from .complex_product_interceptor import ProbeArray, intercept, planted_complex_product


def _source(driver: FdtdDriver, component: str, center) -> None:
    driver.add_source({"component": component, "frequency": 1.0, "center": center,
                       "size": (0.0, 0.0, 0.0)})


def _bloch_2d() -> FdtdDriver:
    driver = FdtdDriver(cell_size=(2.0, 1.6, 0.0), resolution=10, k_point=(0.3, 0.2, 0.0),
                        dimensions=2)
    _source(driver, "Ez", (-0.6, -0.5, 0.0))
    _source(driver, "Hz", (0.35, 0.25, 0.0))
    return driver


def _bloch_3d() -> FdtdDriver:
    driver = FdtdDriver(cell_size=(1.2, 1.0, 0.8), resolution=10, k_point=(0.17, 0.29, 0.11))
    _source(driver, "Hx", (-0.31, -0.22, 0.13))
    _source(driver, "Ez", (0.24, 0.11, -0.17))
    return driver


def _bloch_pml_same_axis() -> FdtdDriver:
    driver = FdtdDriver(cell_size=(3.0, 2.0, 0.0), resolution=10, k_point=(0.3, 0.1, 0.0),
                        dimensions=2)
    driver.setup_pml({"x": 5})
    _source(driver, "Ez", (-0.6, -0.5, 0.0))
    return driver


def _fold_beside_bloch() -> FdtdDriver:
    driver = FdtdDriver(cell_size=(2.0, 1.6, 1.0), resolution=10, symmetry=("X",),
                        k_point=(0.0, 0.27, 0.0))
    _source(driver, "Ez", (0.0, -0.3, 0.05))
    return driver


def _dispersive_pml() -> FdtdDriver:
    driver = FdtdDriver(cell_size=(3.6, 3.6, 0.0), resolution=10, dimensions=2)
    driver.setup_pml({"x": 10, "y": 10})
    driver.add_susceptibility(Susceptibility(frequency=1.1, gamma=0.1, kind=LORENTZIAN), 0.5)
    _source(driver, "Ez", (-0.6, -0.5, 0.0))
    return driver


def _conductive_pml() -> FdtdDriver:
    driver = FdtdDriver(cell_size=(3.6, 3.6, 0.0), resolution=10, dimensions=2)
    driver.setup_pml({"x": 10, "y": 10})
    driver.set_conductivity(numpy.full(driver.shape, 0.4, dtype=numpy.float32))
    _source(driver, "Ez", (-0.6, -0.5, 0.0))
    return driver


def _bfast() -> FdtdDriver:
    driver = FdtdDriver(cell_size=(2.0, 3.6, 0.0), resolution=10, dimensions=2,
                        bfast_scaled_k=(0.2, 0.0, 0.0))
    driver.setup_pml({"y": 10})
    _source(driver, "Ez", (-0.6, -0.5, 0.0))
    _source(driver, "Hz", (0.35, 0.25, 0.0))
    return driver


def _beta() -> FdtdDriver:
    driver = FdtdDriver(cell_size=(2.0, 1.6, 0.0), resolution=10, dimensions=2,
                        k_point=(0.3, 0.0, 0.0), beta=0.4)
    _source(driver, "Ez", (-0.6, -0.5, 0.0))
    _source(driver, "Hz", (0.35, 0.25, 0.0))
    return driver


def _cylindrical_m1() -> FdtdDriver:
    driver = FdtdDriver(cell_size=(2.0, 0.0, 2.0), resolution=10, cylindrical=True, m=1)
    driver.setup_pml({"x": (0, 5), "z": 5})
    _source(driver, "Ex", (0.5, 0.0, 0.1))
    return driver


CONFIGURATIONS = {
    "bloch_2d": _bloch_2d,
    "bloch_3d_three_axes": _bloch_3d,
    "bloch_and_pml_on_one_axis": _bloch_pml_same_axis,
    "mirror_beside_bloch": _fold_beside_bloch,
    "lorentzian_with_pml": _dispersive_pml,
    "d_conductivity_with_pml": _conductive_pml,
    "bfast": _bfast,
    "special_kz_beta": _beta,
    "cylindrical_m1": _cylindrical_m1,
}


def test_the_probe_counts_a_planted_product_and_nothing_additive():
    driver = _bloch_2d()
    try:
        with intercept(driver) as probe:
            target = driver.fields.Dz
            assert isinstance(target, ProbeArray)
            target + target  # additive: traffic, not a violation
            numpy.float32(2.0) * target.real  # a real product of a part: neither
            assert probe.violations == 0  # judging is off until switched on
            with probe.judging():
                target + target
                target.real * numpy.float32(2.0)
                planted_complex_product(target)
            report = probe.report()
        assert report["violations"] == 1
        assert report["products"] == {"multiply": 1}
        assert report["complex_traffic"].get("add") == 1
        assert list(report["sites"]) == [
            next(site for site in report["sites"] if "test_the_probe_counts_a_planted_product" in site)]
        # Restored: the driver's arrays are plain arrays again and the scratch pool is too.
        assert type(driver.fields.Dz) is numpy.ndarray
        assert "take" not in vars(driver.fields.scratch)
    finally:
        driver.close()


def test_the_probe_refuses_a_device_driver():
    class _Grid:
        xp = object()

    class _Fields:
        grid = _Grid()

    class _Driver:
        fields = _Fields()

    with pytest.raises(ValueError, match="NumPy reference only"):
        with intercept(_Driver()):
            pass


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_the_array_path_forms_no_complex_product(name):
    driver = CONFIGURATIONS[name]()
    try:
        assert driver.fields.Dz.dtype == numpy.complex64
        with intercept(driver) as probe:
            with probe.judging():
                for _ in range(4):
                    driver.step()
            report = probe.report()
            with probe.judging():
                planted_complex_product(driver.fields.Dz)
            control = probe.violations - report["violations"]
    finally:
        driver.close()
    assert report["violations"] == 0, (
        f"{name}: the array path formed {report['violations']} complex product(s), which "
        f"contract on FMA hosts and flip zero signs; sites {report['sites']}")
    assert report["complex_traffic_total"] > 0, f"{name}: the probe saw no complex arithmetic at all"
    assert control == 1, f"{name}: the planted product on this run's own storage was not caught"
