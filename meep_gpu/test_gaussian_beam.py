"""Cross-validation for the Gaussian-beam field evaluator.

The oracle here is MEEP's OWN compiled ``gaussianbeam::get_fields``, reached through
ctypes (``meep_gpu.gaussian_beam.compiled_beam_fields``), so what is measured is the
transcription rather than anyone's reading of the C++. Both sides run the same
double-precision arithmetic, so the bar is 1e-13 relative, not a physics tolerance.

The end-to-end half — a lifted beam stepped against CPU MEEP — lives in
``test_from_meep.py`` (``beam_two_d_tm`` / ``beam_two_d_te`` / ``beam_three_d``), with
the ``hx_is_ey`` control that this file's pinning tests explain.

Everything that needs a live ``mp.Simulation`` runs in a SUBPROCESS, for the same
reason the rest of this package does it: meep and torch each ship their own OpenMP
runtime and abort when loaded together, and keeping ``import meep`` out of the module
keeps it collectable on a machine that has never had MEEP installed.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MEEP_MISSING = importlib.util.find_spec("meep") is None
requires_meep = pytest.mark.requires_resource("meep")
skip_without_meep = pytest.mark.skipif(
    MEEP_MISSING, reason="CPU MEEP is not installed in this environment"
)
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)


_CHILD_SCRIPT = '''"""Compare the transcription against MEEP's compiled get_fields, and pin MEEP's own facts."""
import json
import math
import sys

import numpy as np
import meep as mp

from meep_gpu.gaussian_beam import beam_fields, compiled_beam_fields

RNG = np.random.default_rng(20260804)

# (name, dimensions, x0, kdir, w0, freq, eps, mu, E0). The w0 values are chosen to put
# points in each of get_fields' three numeric branches: kz0 = k*k*w0*w0/2 above 30
# selects the rescaled cos/sin (sources.cpp:647-653), and the generic branch otherwise.
# The Taylor branch (|kR| <= 1e-4, sources.cpp:664-669) is measured separately below,
# because reaching it needs a waist so small that MEEP's own Eorig normalization is
# destroyed by cancellation.
CONFIGS = [
    ("three_d_real_x", 3, (0, 0, 0), (0, 0, 1), 0.8, 1.0, 1.0, 1.0, (1, 0, 0)),
    ("three_d_off_axis", 3, (0.3, -0.2, 3.0), (0.2, 0.4, 1.0), 0.8, 1.0, 2.25, 1.0, (1, 0, 0)),
    ("three_d_imaginary_E0", 3, (0, 0, 1.0), (0, 0, 1), 1.3, 0.7, 1.0, 1.0, (0, 1j, 0)),
    ("three_d_mixed_E0", 3, (0.1, 0.2, 0.3), (1, 1, 1), 0.5, 1.1, 1.44, 1.0,
     (1 + 0.5j, 0.2j, 0.3)),
    ("three_d_rescaled_fg", 3, (0, 0, 0), (0, 0, 1), 4.0, 1.0, 1.0, 1.0, (1, 0, 0)),
    ("two_d_corpus", 2, (0, 3, 0), (0, 1, 0), 0.8, 1.0, 1.0, 1.0, (0, 0, 1)),
    ("two_d_z_components", 2, (0.0, 3.0, 0.7), (0.0, 1.0, 0.5), 0.8, 1.0, 1.0, 1.0, (0, 0, 1)),
    ("two_d_te", 2, (0, 2, 0), (0.3, 1.0, 0), 1.2, 0.9, 2.0, 1.0, (1, 0.4, 0)),
    ("two_d_mixed_E0", 2, (0.2, 1.0, -0.4), (1.0, 0.6, 0.0), 0.6, 1.4, 1.0, 1.0, (0.5, 0, 1j)),
    ("two_d_rescaled_fg", 2, (0, 0, 0), (0, 1, 0), 3.0, 1.0, 1.0, 1.0, (0, 0, 1)),
]


def sample_points(count):
    return np.concatenate([
        RNG.uniform(-8.0, 8.0, size=(count, 3)),
        RNG.uniform(-40.0, 40.0, size=(count // 4, 3)),
        np.zeros((1, 3)),
    ])


def branch_counts(points, x0, kdir, w0, freq, eps, mu, dimensions):
    """How many points land in each of get_fields' three numeric branches."""
    k = 2 * math.pi * freq * math.sqrt(eps * mu)
    z0 = k * w0 * w0 / 2
    x0 = np.asarray(x0, float)
    kdir = np.asarray(kdir, float)
    looped = dimensions
    xrel = np.broadcast_to(x0, points.shape).copy()
    xrel[:, :looped] = points[:, :looped] - x0[:looped]
    zhat = kdir.copy()
    zhat[:looped] /= math.sqrt(float(np.dot(kdir[:looped], kdir[:looped])))
    rho = np.linalg.norm(np.cross(np.broadcast_to(zhat, xrel.shape), xrel), axis=-1)
    zc = xrel[:, :looped] @ zhat[:looped] - 1j * z0
    kR = k * np.sqrt((rho * rho + zc * zc).astype(complex))
    big = np.abs(kR) > 1e-4
    rescaled = big & (np.abs(kR.imag) > 30.0)
    return {"taylor": int((~big).sum()), "rescaled": int(rescaled.sum()),
            "generic": int((big & ~rescaled).sum())}


def transcription():
    out = {}
    for name, dims, x0, kdir, w0, freq, eps, mu, E0 in CONFIGS:
        points = sample_points(1200)
        mine = beam_fields(points, x0, kdir, w0, freq, eps, mu, E0, dims)
        theirs = compiled_beam_fields(mp, points, x0, kdir, w0, freq, eps, mu, E0, dims)
        scale = float(np.abs(theirs).max())
        out[name] = {
            "relative_max": float(np.abs(mine - theirs).max() / scale),
            "points": int(len(points)),
            "branches": branch_counts(points, x0, kdir, w0, freq, eps, mu, dims),
        }
    return out


def taylor_branch():
    """The small-kR series, compared as a SHAPE.

    Reaching |kR| <= 1e-4 needs w0 ~ 1e-6, and there kz0 ~ 2e-11 makes MEEP's own
    Eorig (``exp(kz0)*kz0*(kz0-1) + sinh(kz0)``, sources.cpp:739) a difference of two
    quantities that agree to 11 digits — its double-precision value is rounding noise
    on both sides. Dividing it out isolates the branch actually under test.
    """
    points = np.zeros((7, 3))
    points[:, 2] = np.linspace(-1e-6, 1e-6, 7)
    out = {}
    for label, E0 in (("real", (1, 0, 0)), ("imaginary", (0, 1j, 0)),
                      ("mixed", (1 + 0.5j, 0.2j, 0.3))):
        mine = beam_fields(points, (0, 0, 0), (0, 0, 1), 1e-6, 1.0, 1.0, 1.0, E0, 3)
        theirs = compiled_beam_fields(mp, points, (0, 0, 0), (0, 0, 1), 1e-6, 1.0, 1.0,
                                      1.0, E0, 3)
        out[label] = {
            "shape_relative_max": float(np.abs(mine / np.abs(mine).max()
                                               - theirs / np.abs(theirs).max()).max()),
            "branches": branch_counts(points, (0, 0, 0), (0, 0, 1), 1e-6, 1.0, 1.0, 1.0, 3),
        }
    return out


def eorig_cancellation():
    """Where the Eorig cancellation starts to dominate, as a function of the waist."""
    points = np.random.default_rng(11).uniform(-4, 4, size=(600, 3))
    out = {}
    for w0 in (0.8, 0.1, 0.05, 0.02, 0.01):
        mine = beam_fields(points, (0, 0, 0), (0, 0, 1), w0, 1.0, 1.0, 1.0, (1, 0, 0), 3)
        theirs = compiled_beam_fields(mp, points, (0, 0, 0), (0, 0, 1), w0, 1.0, 1.0, 1.0,
                                      (1, 0, 0), 3)
        out[f"{w0:g}"] = float(np.abs(mine - theirs).max() / np.abs(theirs).max())
    return out


def beam_simulation(**overrides):
    source = dict(center=mp.Vector3(0, 0, -1), size=mp.Vector3(3, 3, 0),
                  beam_x0=mp.Vector3(0, 0, 1), beam_kdir=mp.Vector3(0, 0, 1),
                  beam_w0=0.8, beam_E0=mp.Vector3(1, 0, 0))
    source.update(overrides)
    return mp.Simulation(
        cell_size=mp.Vector3(4, 4, 4), resolution=10, boundary_layers=[mp.PML(0.5)],
        force_complex_fields=True,
        sources=[mp.GaussianBeamSource(mp.ContinuousSource(frequency=1.0), **source)])


def meep_facts():
    """What MEEP ITSELF does, measured — so a change upstream turns this suite red."""
    sim = beam_simulation()
    sim.init_sim()
    deposited = {name: float(np.abs(np.asarray(sim.get_source(c))).max())
                 for name, c in (("Ex", mp.Ex), ("Ey", mp.Ey),
                                 ("Hx", mp.Hx), ("Hy", mp.Hy))}

    def peak(**overrides):
        probe = beam_simulation(**overrides)
        probe.init_sim()
        return float(np.abs(np.asarray(probe.get_source(mp.Ex))).max())

    reference = peak()
    # A 2-D beam polarized both in and out of plane: MEEP's own D2 parity checks
    # (sources.cpp:501-504) reject all four sheets, so MEEP steps a SOURCELESS run.
    both = mp.Simulation(
        cell_size=mp.Vector3(6, 6, 0), resolution=10, boundary_layers=[mp.PML(1.0)],
        force_complex_fields=True, sources=[mp.GaussianBeamSource(
            mp.ContinuousSource(frequency=1.0), center=mp.Vector3(0, -1),
            size=mp.Vector3(3, 0), beam_x0=mp.Vector3(0, 1), beam_kdir=mp.Vector3(0, 1),
            beam_w0=0.8, beam_E0=mp.Vector3(1, 0, 1))])
    both.run(until=4.0)
    return {
        "deposited": deposited,
        "amplitude_ratio": peak(amplitude=5.0) / reference,
        "amp_func_ratio": peak(amp_func=lambda p: 3.0) / reference,
        "two_d_both_polarizations_peak": sum(
            float(np.abs(np.asarray(both.get_array(component=c))).max())
            for c in (mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz)),
    }


MODES = {"transcription": transcription, "taylor": taylor_branch,
         "eorig": eorig_cancellation, "meep_facts": meep_facts}
mode, output_path = sys.argv[1], sys.argv[2]
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(MODES[mode](), handle)
'''


def _child(tmp_path, mode: str) -> dict:
    script_path = tmp_path / "gaussian_beam_case.py"
    script_path.write_text(_CHILD_SCRIPT, encoding="utf-8")
    output_path = tmp_path / f"{mode}.json"
    environment = dict(os.environ)
    environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    environment["PYTHONPATH"] = os.pathsep.join(
        [_PACKAGE_PARENT] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    completed = subprocess.run(
        [sys.executable, str(script_path), mode, str(output_path)],
        capture_output=True, text=True, env=environment, timeout=900,
    )
    if (completed.returncode != 0 and sys.platform != "darwin"
            and "the C++ oracle is unavailable on this build" in completed.stderr):
        # The oracle resolves MEEP's compiled get_fields by its libc++ symbol name; a
        # libstdc++ build (conda-forge MEEP on Linux) spells it differently. The
        # transcription under test does not depend on the oracle. On macOS MEEP is
        # built against libc++, so an unresolvable oracle there is a failure.
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip(
            "meep_libcxx_symbols",
            "MEEP's compiled gaussianbeam::get_fields is not resolvable by its libc++ "
            "name in this MEEP build, so the C++ oracle is unavailable")
    assert completed.returncode == 0 and output_path.exists(), (
        f"gaussian_beam child '{mode}' failed (exit {completed.returncode}).\n"
        f"stdout tail:\n{completed.stdout[-3000:]}\nstderr tail:\n{completed.stderr[-3000:]}"
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


@requires_meep
@skip_without_meep
def test_transcription_matches_meeps_compiled_get_fields(tmp_path):
    """The NumPy transcription against MEEP's own compiled evaluator, point for point.

    Measured relative max error, 1501 points per configuration, with the branch
    each configuration exercises::

        three_d_real_x         3.9e-15   generic
        three_d_off_axis       7.0e-15   generic
        three_d_imaginary_E0   4.2e-15   generic
        three_d_mixed_E0       3.3e-15   generic
        three_d_rescaled_fg    6.7e-14   rescaled cos/sin (|Im kR| > 30)
        two_d_corpus           4.5e-15   generic
        two_d_z_components     2.8e-15   generic
        two_d_te               1.3e-14   generic
        two_d_mixed_E0         7.4e-15   generic
        two_d_rescaled_fg      1.7e-14   rescaled cos/sin

    ``two_d_z_components`` is the one that pins the dimension-dependent algebra: it
    gives ``beam_x0`` and ``beam_kdir`` z components in a 2-D cell, which
    ``py_v3_to_vec`` keeps (simulation.py:130) and ``get_fields`` reads asymmetrically
    — see the module docstring of :mod:`.gaussian_beam`. Reducing them to the cell's
    dimensionality changes the DEPOSITED CURRENT by 8.4e+00 end to end
    (``beam_two_d_tm``'s ``reduce_beam_vectors`` control in test_from_meep.py).
    """
    data = _child(tmp_path, "transcription")
    for name, payload in sorted(data.items()):
        print(f"[gaussian_beam] {name:22} relative_max {payload['relative_max']:.3e} "
              f"branches {payload['branches']}", flush=True)
        assert payload["relative_max"] < 1e-13, (
            f"{name}: the transcription is {payload['relative_max']:.3e} from MEEP's own "
            f"compiled get_fields; both sides run the same double arithmetic, so anything "
            f"above 1e-13 is a transcription error and not rounding."
        )
    # The rescaled cos/sin branch is a DIFFERENT code path, not a tolerance: assert it
    # was reached, or the two configurations that exist for it prove nothing.
    assert any(payload["branches"]["rescaled"] > 0 for payload in data.values()), (
        "no configuration reached the |Im kR| > 30 rescaled branch (sources.cpp:647-653)"
    )
    assert any(payload["branches"]["generic"] > 0 for payload in data.values())


@requires_meep
@skip_without_meep
def test_the_small_kR_taylor_branch_matches(tmp_path):
    """The third branch, isolated from a normalization MEEP itself cannot evaluate.

    Measured shape agreement 1.1e-16 / 1.1e-16 / 1.2e-16 for real, imaginary and mixed
    E0, with every point in the Taylor branch. The absolute comparison is 3.8e-01 in
    this regime and that is NOT a transcription error: ``Eorig`` at kz0 = 2.0e-11 is
    ``exp(kz0)*kz0*(kz0-1) + sinh(kz0)``, whose two terms cancel to eleven digits, so
    its double-precision value is rounding noise on either side. The measured scale
    ratio between the two evaluations is 0.6229 — a pure scalar, identical for every
    point and every polarization, which is what makes the diagnosis a measurement
    rather than an excuse.
    """
    data = _child(tmp_path, "taylor")
    for label, payload in sorted(data.items()):
        print(f"[gaussian_beam] taylor/{label:10} shape {payload['shape_relative_max']:.3e} "
              f"branches {payload['branches']}", flush=True)
        assert payload["branches"]["taylor"] == sum(payload["branches"].values()), (
            f"taylor/{label}: not every point landed in the small-kR branch"
        )
        assert payload["shape_relative_max"] < 1e-14


@requires_meep
@skip_without_meep
def test_the_eorig_cancellation_regime_is_where_it_is_measured_to_be(tmp_path):
    """How small a waist the evaluator holds full precision down to.

    ``Eorig``'s cancellation, not the beam physics, sets the floor. Measured relative
    max against compiled MEEP::

        w0 = 0.80  (kz0 = 12.6  )   3.8e-15
        w0 = 0.10  (kz0 = 0.197 )   1.3e-15
        w0 = 0.05  (kz0 = 0.049 )   2.3e-14
        w0 = 0.02  (kz0 = 0.0079)   1.2e-12
        w0 = 0.01  (kz0 = 0.0020)   3.0e-11

    Recorded rather than merely bounded because it is the one place the transcription
    and MEEP can disagree while both are "right": below w0 ~ 0.02 wavelengths MEEP's
    own normalization has lost most of its significant digits.
    """
    data = _child(tmp_path, "eorig")
    for waist, value in sorted(data.items(), key=lambda item: -float(item[0])):
        print(f"[gaussian_beam] eorig w0={waist:6} relative_max {value:.3e}", flush=True)
    assert data["0.8"] < 1e-13 and data["0.1"] < 1e-13
    assert data["0.01"] < 1e-9, (
        "the Eorig cancellation should still leave 9 digits at w0 = 0.01; a bigger gap "
        "means the transcription diverges for a reason other than the cancellation."
    )


@requires_meep
@skip_without_meep
def test_meeps_own_gaussian_beam_facts_are_what_the_lift_assumes(tmp_path):
    """Pin the four MEEP behaviours the lift reproduces, so a change upstream goes red.

    Measured on stock MEEP 1.29.0, a 3-D x-polarized beam (normal z, beam_E0 = xhat)
    at resolution 10::

        get_source(Ex)  9.068233     the driven Ex sheet
        get_source(Hy)  4.573371     its magnetic partner
        get_source(Hx)  0.067138     samples gb_Ey
        get_source(Ey)  0.000000     samples gb_Hx -- EXACTLY zero

    ``get_source(Ey)`` is the whole point. The Ey sheet samples ``EH[3]``, which for
    this polarization is ``gb_Hx`` alone, and ``gb_Hx`` is the uninitialized read at
    sources.cpp:691/:716. MEEP deposits exactly nothing there. The value the intended
    ``gb_Ey`` would have deposited is sitting right next to it in ``get_source(Hx)``:
    0.067138, or 0.74% of the Ex sheet's peak — four orders above this package's
    parity band, and exactly what the ``hx_is_ey`` end-to-end control measures
    (2.0e-03 against a 2.8e-07 baseline). If MEEP ever repairs that line, this
    assertion fails instead of the parity number quietly drifting.

    ``amplitude`` and ``amp_func`` are ignored by MEEP — ratio exactly 1.000000 for
    ``amplitude=5.0`` and for an ``amp_func`` returning 3.0 — because
    ``GaussianBeam3DSource.add_source`` never passes either (source.py:751-773). The
    lift must drop them too; routing through ``_lift_source``'s ``sign * amplitude``,
    which is correct for an EigenModeSource, would silently rescale the run.

    And a 2-D beam polarized both in and out of plane leaves MEEP with a peak field of
    exactly 0.0 across all six components: its own parity checks (sources.cpp:501-504)
    reject every sheet, so it steps a sourceless run that returns a clean, smooth,
    complete, empty field. ``_check_gaussian_beam_source`` refuses that rather than
    reproducing it.
    """
    data = _child(tmp_path, "meep_facts")
    print(f"[gaussian_beam] meep facts {json.dumps(data)}", flush=True)
    assert data["deposited"]["Ey"] == 0.0, (
        f"MEEP's Ey sheet deposits {data['deposited']['Ey']!r}, not 0. The uninitialized "
        f"read at sources.cpp:691/:716 no longer yields zero on this build, so "
        f"meep_gpu.gaussian_beam._gb_hx (which transcribes that zero) is now wrong."
    )
    assert data["deposited"]["Ex"] > 1.0 and data["deposited"]["Hy"] > 1.0, (
        "the driven sheets are empty; the pinning test would pass vacuously"
    )
    assert 0.05 < data["deposited"]["Hx"] < 0.1, (
        f"the Hx sheet reads {data['deposited']['Hx']:.6f}; this is the value the "
        f"'correct' gb_Ex/gb_Hx reading would have put on the Ey sheet, and the control "
        f"that measures the defect is calibrated against it."
    )
    assert data["amplitude_ratio"] == 1.0, (
        f"MEEP now honours `amplitude` on a Gaussian beam (ratio "
        f"{data['amplitude_ratio']!r}); _lift_gaussian_beam_source drops it."
    )
    assert data["amp_func_ratio"] == 1.0, (
        f"MEEP now honours `amp_func` on a Gaussian beam (ratio "
        f"{data['amp_func_ratio']!r}); _lift_gaussian_beam_source drops it."
    )
    assert data["two_d_both_polarizations_peak"] == 0.0, (
        "MEEP no longer renders a both-polarization 2-D beam sourceless; the refusal in "
        "_check_gaussian_beam_source is now over-broad."
    )
