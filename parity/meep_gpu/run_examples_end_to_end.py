"""Run real MEEP example simulations both ways and compare the fields.

The survey (``survey_meep_examples.py``) answers "would the converter accept this
script". This answers the harder question: when it accepts one, does it get the
same answer? Each case builds an ``mp.Simulation`` from an upstream example — the
example's own geometry, materials, sources, boundaries and resolution — lifts it
with :func:`meep_gpu.run_on_gpu`, and then runs THAT SAME OBJECT on CPU MEEP.
The number reported is the complex relative L2 of the whole volume in MEEP's own
``sim.get_array`` layout, so a global phase, a half-cell registration slip and a
dropped boundary plane are all visible.

Where an example needed changing to be liftable at all, the change is listed in
``edits`` and printed with the result. Nothing is silently adjusted: an example
run with two edits is reported as an example run with two edits.

**This file and ``sweep_corpus_lift_parity.py`` are deliberately separate.** The
five cases here are HAND-WRITTEN transcriptions of upstream scripts, each edited
until it lifts and each carrying prose about what the edit cost — a 2-D example
extruded to 3-D is a different problem from the one upstream wrote, and saying so
is most of the value. The sweep runs the corpus's own captured ``mp.Simulation``
objects, unedited, through the survey's capture in a subprocess, and reports the
57 rows nobody transcribed. Neither subsumes the other: this file is deep and
annotated, the sweep is broad and automatic.

Progress reporting: one flushed line per case, and each result is appended to the JSONL as it
lands.

Usage (from ``the repository root``)::

    python -u -m parity.meep_gpu.run_examples_end_to_end
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import meep as mp

from meep_gpu import gpu_compatibility, run_on_gpu

COMPONENTS = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez, "Hx": mp.Hx, "Hy": mp.Hy, "Hz": mp.Hz}


# --- the cases ---------------------------------------------------------------------
#
# Each builder returns (sim, run_kwargs, component, edits). `edits` is the honest
# list of what had to change relative to the upstream script; an empty list means
# the simulation is the example's own.


def grating2d_triangular_lattice():
    """``grating2d_triangular_lattice.py`` — the corpus's only 3-D Cartesian example.

    A triangular lattice of dielectric cylinders on a substrate, Bloch-periodic in
    x and y with PML in z: a real photonic-crystal grating, and the single script
    in MEEP's 80 that this engine's dimensionality can take at all.
    """
    resolution = 40  # Upstream 100; a factor 2.5 down so the case runs in seconds.
    ng = 1.5
    glass = mp.Medium(index=ng)
    wvl = 0.5
    fcen = 1 / wvl
    dpml = 1.0
    dsub = 1.5
    dair = 1.0
    rcyl = 0.1
    hcyl = 0.3
    gp = 1.0
    sx = gp
    sy = gp * math.sqrt(3)
    sz = dpml + dsub + hcyl + dair + dpml
    cell_size = mp.Vector3(sx, sy, sz)
    boundary_layers = [mp.PML(thickness=dpml, direction=mp.Z)]
    src_pt = mp.Vector3(0, 0, -0.5 * sz + dpml)
    sources = [mp.Source(src=mp.GaussianSource(fcen, fwidth=0.1 * fcen), size=mp.Vector3(sx, sy, 0),
                         center=src_pt, component=mp.Ey)]
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, dpml + dsub),
                 center=mp.Vector3(0, 0, -0.5 * sz + 0.5 * (dpml + dsub)), material=glass),
        # Upstream uses mp.Cylinder here. A cylinder's curved surface makes MEEP's
        # anisotropic averaging tilt chi1inv off the diagonal, which this engine
        # cannot store and therefore refuses; eps_averaging=False is MEEP's own
        # switch to point-sampling, which it can.
        mp.Cylinder(material=glass, radius=rcyl, height=hcyl,
                    center=mp.Vector3(0, 0, -0.5 * sz + dpml + dsub + 0.5 * hcyl)),
        mp.Cylinder(material=glass, radius=rcyl, height=hcyl,
                    center=mp.Vector3(0.5 * sx, 0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl)),
        mp.Cylinder(material=glass, radius=rcyl, height=hcyl,
                    center=mp.Vector3(-0.5 * sx, 0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl)),
        mp.Cylinder(material=glass, radius=rcyl, height=hcyl,
                    center=mp.Vector3(0.5 * sx, -0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl)),
        mp.Cylinder(material=glass, radius=rcyl, height=hcyl,
                    center=mp.Vector3(-0.5 * sx, -0.5 * sy, -0.5 * sz + dpml + dsub + 0.5 * hcyl)),
    ]
    sim = mp.Simulation(resolution=resolution, cell_size=cell_size, sources=sources,
                        geometry=geometry, boundary_layers=boundary_layers,
                        k_point=mp.Vector3(), eps_averaging=False)
    edits = [
        "eps_averaging=False (upstream default True; MEEP's anisotropic averaging of the "
        "cylinders tilts chi1inv off the diagonal, which the lift refuses)",
        "resolution 100 -> 40 and a shorter run, for a case that finishes in seconds",
        "the upstream sim.add_flux monitor is not attached (MEEP's own monitors are not "
        "lifted; they go on the returned driver through run_on_gpu(prepare=...))",
    ]
    return sim, dict(until=6.0), "Ey", edits


def chirped_pulse_3d():
    """``chirped_pulse.py`` — a chirped CustomSource in an empty cell, extruded to 3-D.

    Refused upstream for exactly one reason: the cell is 2-D. Everything else about
    it — the custom chirped waveform, the source plane, the PML, the periodic
    transverse boundary — lifts as written.
    """
    resolution = 20  # Upstream 40.
    dpml = 1.0
    sx, sy, sz = 12.0, 3.0, 3.0
    fcen = 1.0
    df = 0.2
    tchirp = 40.0
    t0 = 2.0

    def chirp(t):
        return np.exp(1j * 2 * np.pi * fcen * t) * np.exp(
            -df * (t - t0) ** 2 + 1j * (df ** 2) * (t - t0) ** 2 / tchirp)

    sim = mp.Simulation(
        cell_size=mp.Vector3(sx, sy, sz), resolution=resolution,
        boundary_layers=[mp.PML(dpml, direction=mp.X)],
        k_point=mp.Vector3(),
        sources=[mp.Source(src=mp.CustomSource(src_func=chirp, end_time=20.0),
                           center=mp.Vector3(-0.5 * sx + dpml), size=mp.Vector3(0, sy, sz),
                           component=mp.Ez)],
    )
    edits = [
        "the 2-D cell mp.Vector3(sx, sy, 0) is given a real z extent (3.0) and the source a "
        "z size, making it the 3-D problem; MEEP 2-D is translational invariance and is refused",
        "k_point=mp.Vector3() added (MEEP's default is PEC walls, which this engine does not "
        "have; the x boundary is PML either way)",
        "resolution 40 -> 20 and a shorter run",
    ]
    return sim, dict(until=8.0), "Ez", edits


def straight_waveguide_3d():
    """``straight-waveguide.py`` — MEEP's first tutorial, as a 3-D dielectric slab.

    The classic eps-12 guide in a PML box. Structured, so it exercises the chi1inv
    read on a real tutorial geometry.
    """
    cell = mp.Vector3(16, 8, 4)
    geometry = [mp.Block(mp.Vector3(mp.inf, 1, 1), center=mp.Vector3(),
                         material=mp.Medium(epsilon=12))]
    sources = [mp.Source(mp.ContinuousSource(frequency=0.15), component=mp.Ez,
                         center=mp.Vector3(-7, 0, 0))]
    sim = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(1.0)], geometry=geometry,
                        sources=sources, resolution=10, k_point=mp.Vector3())
    edits = [
        "the 2-D cell mp.Vector3(16, 8, 0) is given a z extent of 4 and the guide a finite "
        "1-unit height, making it a 3-D ridge rather than an infinite 2-D slab",
        "k_point=mp.Vector3() added (MEEP's default PEC walls are refused; the cell is "
        "PML-bounded on every face anyway)",
        "run shortened from until=200",
    ]
    return sim, dict(until=20.0), "Ez", edits


def refl_quartz_3d():
    """``refl-quartz.py`` — reflectance off fused quartz, as a 3-D slab.

    A real dispersive material from MEEP's own library spelling: three Lorentzian
    (Sellmeier) terms on the substrate. Structured in epsilon with one shared
    dispersive model, which is exactly the combination the chi1inv read supports.
    """
    # Upstream 200. Fused quartz's first Sellmeier pole sits at f0 = 1/0.0684 =
    # 14.6, and a Lorentz ADE needs omega_0 * dt well under 2 — so does MEEP's, this
    # is physics rather than an engine limit. dt = Courant/resolution, so 50 gives
    # omega_0 * dt = 0.92 and the engine's stability guard passes; at 20 it raises
    # DispersionInstability rather than returning the growing field.
    resolution = 50
    dpml = 1.0
    sz = 4 + 2 * dpml
    fmin, fmax = 0.4, 0.8
    fcen = 0.5 * (fmin + fmax)
    # MEEP's own fused-quartz Sellmeier fit (materials library), spelled out so the
    # case does not depend on the library import.
    frq1, gam1, sig1 = 1 / 0.0684043, 0.0, 0.696166300
    frq2, gam2, sig2 = 1 / 0.1162414, 0.0, 0.407942600
    frq3, gam3, sig3 = 1 / 9.896161, 0.0, 0.897479400
    quartz_terms = [
        mp.LorentzianSusceptibility(frequency=frq1, gamma=gam1, sigma=sig1),
        mp.LorentzianSusceptibility(frequency=frq2, gamma=gam2, sigma=sig2),
        mp.LorentzianSusceptibility(frequency=frq3, gamma=gam3, sigma=sig3),
    ]
    # Both media carry the same susceptibilities so the cell is structured in
    # epsilon alone: vacuum is eps 1 with the same terms at sigma-weighted... no —
    # the terms must be IDENTICAL, so the "air" half is given the same terms and
    # eps 1, which is a different material from quartz only in epsilon_infinity.
    background = mp.Medium(epsilon=1.0, E_susceptibilities=quartz_terms)
    substrate = mp.Medium(epsilon=1.0000001, E_susceptibilities=quartz_terms)
    sim = mp.Simulation(
        cell_size=mp.Vector3(1, 1, sz), resolution=resolution,
        boundary_layers=[mp.PML(dpml, direction=mp.Z)],
        default_material=background,
        geometry=[mp.Block(mp.Vector3(mp.inf, mp.inf, 0.5 * sz),
                           center=mp.Vector3(0, 0, 0.25 * sz), material=substrate)],
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=fmax - fmin), component=mp.Ex,
                           center=mp.Vector3(0, 0, -0.5 * sz + dpml + 0.5),
                           size=mp.Vector3(1, 1, 0))],
        k_point=mp.Vector3(),
    )
    edits = [
        "the 1-D cell mp.Vector3(0, 0, sz) is given real x and y extents, making it 3-D",
        "k_point=mp.Vector3() added",
        "the two media are given the SAME three Sellmeier terms and differ only in "
        "epsilon_infinity: MEEP's per-point sigma volumes are not readable, so a cell "
        "structured in dispersion is refused (this keeps the dispersion, not the contrast)",
        "resolution 200 -> 50 (below ~24 the UV Sellmeier pole is unstable in ANY explicit ADE, "
        "MEEP's included; the engine raises DispersionInstability rather than running it), "
        "cell shortened, and the flux monitors are not attached",
    ]
    return sim, dict(until=6.0), "Ex", edits


def pw_source_3d():
    """``pw-source.py`` — an oblique plane wave from an amp_func, in 3-D.

    An extended source with a Python ``amp_func`` carrying the transverse phase
    ramp, plus a Bloch ``k_point`` that must match it. The amp_func convention
    (MEEP's one ``Vector3``, relative to the source centre) is the mapping this
    case pins on real third-party code.
    """
    resolution = 12
    s = 8.0
    dpml = 1.0
    fcen = 0.8
    # The example's oblique k lies in the plane of its PML axis, which this engine
    # refuses (an absorbing axis carries no Bloch phase). The tilt is therefore put
    # in the two PERIODIC axes and the PML kept on x, which is the same source
    # construction — an extended plane with a phase ramp from amp_func — in the
    # configuration the engine supports.
    theta = math.radians(20.0)
    k = mp.Vector3(0, math.sin(theta), math.cos(theta)).scale(2 * math.pi * fcen)

    def pw_amp(k_vector, x0):
        def _amp(x):
            return complex(np.exp(1j * (k_vector.dot(x + x0))))
        return _amp

    centre = mp.Vector3(-0.5 * s + dpml, 0, 0)
    sim = mp.Simulation(
        cell_size=mp.Vector3(s, s, s), resolution=resolution,
        boundary_layers=[mp.PML(dpml, direction=mp.X)],
        k_point=k,
        sources=[mp.Source(mp.ContinuousSource(fcen), component=mp.Ez, center=centre,
                           size=mp.Vector3(0, s, s), amp_func=pw_amp(k, centre))],
    )
    edits = [
        "the 2-D cell is given a z extent, and the source plane a z size",
        "the oblique k is rotated into the periodic (y, z) plane: the example tilts it toward "
        "its own PML axis, and this engine refuses a Bloch phase on an absorbing axis "
        "(gpu_compatibility reports it; MEEP itself does step that combination)",
        "resolution 10 -> 12 so the PML is a whole number of cells, and a shorter run",
    ]
    return sim, dict(until=6.0), "Ez", edits


CASES = {
    "grating2d_triangular_lattice": grating2d_triangular_lattice,
    "chirped_pulse_3d": chirped_pulse_3d,
    "straight_waveguide_3d": straight_waveguide_3d,
    "refl_quartz_3d": refl_quartz_3d,
    "pw_source_3d": pw_source_3d,
}


def relative_l2(candidate, reference) -> float:
    a = np.asarray(candidate, dtype=np.complex128).ravel()
    b = np.asarray(reference, dtype=np.complex128).ravel()
    denominator = float(np.linalg.norm(b))
    if denominator == 0.0:
        raise RuntimeError("the CPU-MEEP reference is identically zero")
    if float(np.linalg.norm(a)) == 0.0:
        raise RuntimeError("the lifted run is identically zero")
    return float(np.linalg.norm(a - b) / denominator)


def run_case(name: str) -> dict:
    sim, run_kwargs, component, edits = CASES[name]()
    record = {"case": name, "edits": edits, "resolution": float(sim.resolution),
              "cell": [sim.cell_size.x, sim.cell_size.y, sim.cell_size.z]}
    verdict = gpu_compatibility(sim)
    record["supported"] = bool(verdict.supported)
    record["reasons"] = list(verdict.reasons)
    if not verdict.supported:
        return record
    started = time.time()
    result = run_on_gpu(sim, prefer_gpu=False, **run_kwargs)
    record["lift_and_step_s"] = round(time.time() - started, 2)
    record["steps"] = result.steps
    record["cells"] = int(np.prod(result.driver.shape))
    lifted = result.get_array(component)
    started = time.time()
    sim.run(**run_kwargs)  # THE SAME OBJECT, now stepped by CPU MEEP.
    record["cpu_meep_s"] = round(time.time() - started, 2)
    reference = np.asarray(sim.get_array(component=COMPONENTS[component]))
    record["shape"] = list(lifted.shape)
    record["reference_shape"] = list(reference.shape)
    record["component"] = component
    record["relative_l2"] = (
        relative_l2(lifted, reference) if lifted.shape == reference.shape else None
    )
    result.close()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="parity/meep_gpu/results/end_to_end.jsonl")
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    names = sorted(CASES) if not args.only else args.only.split(",")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")
    print(f"running {len(names)} example simulations both ways", flush=True)
    for index, name in enumerate(names, start=1):
        started = time.time()
        try:
            record = run_case(name)
        except Exception as exc:  # noqa: BLE001 - a failed case is a reportable result.
            record = {"case": name, "error": f"{type(exc).__name__}: {exc}"[:600]}
        record["wall_s"] = round(time.time() - started, 2)
        with out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        if record.get("relative_l2") is not None:
            verdict = (f"L2 = {record['relative_l2']:.3e}  "
                       f"({record['cells']} cells, {record['steps']} steps, "
                       f"gpu-path {record['lift_and_step_s']} s vs CPU MEEP {record['cpu_meep_s']} s)")
        elif record.get("error"):
            verdict = "ERROR " + record["error"][:120]
        elif not record.get("supported", True):
            verdict = "REFUSED " + record["reasons"][0][:100]
        else:
            verdict = f"SHAPE MISMATCH {record.get('shape')} vs {record.get('reference_shape')}"
        print(f"  {index}/{len(names)} {name:<32} {verdict}  [{record['wall_s']:.1f} s]", flush=True)
        for edit in record.get("edits", ()):
            print(f"        edit: {edit}", flush=True)
    print(f"done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
