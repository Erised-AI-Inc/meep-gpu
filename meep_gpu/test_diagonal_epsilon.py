"""Diagonal-permittivity tests for the standalone core and CPU-MEEP oracle."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

from .driver import FdtdDriver


_HAS_MEEP = importlib.util.find_spec("meep") is not None

_CASE = {
    "cell_size": (1.1, 1.3, 1.5),
    "resolution": 10,
    "epsilon": {"Ex": 2.0, "Ey": 3.0, "Ez": 4.0},
    "frequency": 1.0,
    "until": 1.5,
    "sources": (
        ("Ex", (0.15, -0.25, -0.35), 0.7),
        ("Ey", (-0.15, 0.25, -0.15), 0.5),
        ("Ez", (0.05, 0.05, -0.25), 1.0),
    ),
}

_ORACLE_SCRIPT = r'''
"""CPU-MEEP oracle for a uniform diagonal dielectric."""
import json
import sys

import meep as mp
import numpy as np

case = json.loads(sys.argv[1])
output_path = sys.argv[2]
epsilon = case["epsilon"]
simulation = mp.Simulation(
    cell_size=mp.Vector3(*case["cell_size"]),
    resolution=case["resolution"],
    default_material=mp.Medium(
        epsilon_diag=mp.Vector3(epsilon["Ex"], epsilon["Ey"], epsilon["Ez"])
    ),
    sources=[
        mp.Source(
            mp.ContinuousSource(frequency=case["frequency"]),
            component=getattr(mp, component),
            center=mp.Vector3(*center),
            amplitude=amplitude,
        )
        for component, center, amplitude in case["sources"]
    ],
    force_complex_fields=True,
    k_point=mp.Vector3(),
)
simulation.run(until=case["until"])
np.savez(
    output_path,
    **{
        component.lower(): np.asarray(
            simulation.get_array(component=getattr(mp, component))
        )
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    },
)
'''


@pytest.fixture(scope="module")
def diagonal_meep_oracle(tmp_path_factory):
    if not _HAS_MEEP:
        pytest.skip("CPU MEEP is not installed")
    import json

    directory = tmp_path_factory.mktemp("diagonal_epsilon_oracle")
    script_path = directory / "oracle.py"
    script_path.write_text(_ORACLE_SCRIPT, encoding="utf-8")
    output_path = directory / "fields.npz"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    completed = subprocess.run(
        [sys.executable, str(script_path), json.dumps(_CASE), str(output_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    assert completed.returncode == 0 and output_path.exists(), (
        f"CPU-MEEP diagonal-epsilon oracle failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-2000:]}\n"
        f"stderr tail:\n{completed.stderr[-2000:]}"
    )
    with np.load(output_path) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _run_diagonal_engine():
    driver = FdtdDriver(
        cell_size=_CASE["cell_size"],
        resolution=_CASE["resolution"],
        force_complex_fields=True,
    )
    driver.set_epsilon_components(
        {
            component: np.full(driver.shape, value, dtype=np.float32)
            for component, value in _CASE["epsilon"].items()
        }
    )
    for component, center, amplitude in _CASE["sources"]:
        driver.add_source(
            {
                "component": component,
                "frequency": _CASE["frequency"],
                "center": center,
                "size": (0.0, 0.0, 0.0),
                "amplitude": amplitude,
            }
        )
    driver.run(until=_CASE["until"])
    return {
        component.lower(): driver.get_field(component)
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
    }


def _relative_l2(candidate, reference):
    denominator = np.linalg.norm(reference.ravel())
    assert denominator > 0.0, "CPU-MEEP reference field is degenerate"
    return float(np.linalg.norm((candidate - reference).ravel()) / denominator)


@pytest.mark.skipif(not _HAS_MEEP, reason="CPU MEEP is not installed")
def test_uniform_diagonal_epsilon_matches_cpu_meep(diagonal_meep_oracle):
    """All six complex fields match MEEP in a genuinely anisotropic 3-D run."""
    result = _run_diagonal_engine()
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        # MEEP's periodic full-volume get_array includes the duplicate low
        # boundary plane; the driver owns the N cells after it.
        reference = diagonal_meep_oracle[component][1:, 1:, 1:]
        candidate = result[component]
        assert candidate.shape == reference.shape
        error = _relative_l2(candidate, reference)
        assert error < 5e-5, f"{component} complex relative L2 {error:.3e}"


def test_anisotropy_changes_the_physics_not_only_the_material_report():
    anisotropic = _run_diagonal_engine()

    isotropic = FdtdDriver(
        cell_size=_CASE["cell_size"],
        resolution=_CASE["resolution"],
        force_complex_fields=True,
    )
    isotropic.set_epsilon(
        np.full(isotropic.shape, _CASE["epsilon"]["Ez"], dtype=np.float32)
    )
    for component, center, amplitude in _CASE["sources"]:
        isotropic.add_source(
            {
                "component": component,
                "frequency": _CASE["frequency"],
                "center": center,
                "size": (0.0, 0.0, 0.0),
                "amplitude": amplitude,
            }
        )
    isotropic.run(until=_CASE["until"])

    difference = _relative_l2(
        anisotropic["ex"],
        isotropic.get_field("Ex"),
    )
    assert difference > 0.1, (
        f"Replacing diagonal epsilon with scalar epsilon changed Ex by only {difference:.3e}; "
        "the test would not detect a backend that ignored the component mapping."
    )
