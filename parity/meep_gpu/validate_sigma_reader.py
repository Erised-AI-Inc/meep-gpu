"""Validate the MEEP susceptibility-sigma reader patch against a patched build.

Runs the five checks of the design notes (meep-sigma-reader-plan) §4 against the
`fields.get_susceptibility_sigma` family the patch adds (meep-sigma-reader.patch,
applied by build_meep_133_macos.sh under MEEP_SIGMA_PATCH=1). Refuses to run on
an unpatched MEEP rather than reporting vacuous passes.

Usage (the patched env's own python)::

    cd <repo> && \
      python -u \
      parity/meep_gpu/validate_sigma_reader.py

Progress reporting: one flushed line per check, exit nonzero on any failure.
"""

from __future__ import annotations

import sys

import numpy as np
import meep as mp

RESOLUTION = 10
CELL = mp.Vector3(4.0, 4.0, 0.0)
LORENTZ = dict(frequency=1.0, gamma=0.1, sigma=0.4)
DRUDE = dict(frequency=2.0, gamma=0.05, sigma=0.7)

failures: list[str] = []


def check(name: str, condition: bool, detail: str) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}: {detail}", flush=True)
    if not condition:
        failures.append(name)


def build(symmetries=()) -> mp.Simulation:
    # Two blocks in vacuum; eps_averaging deliberately ON so check 3 can show
    # sigma is point-sampled while chi1inv at the same points is smoothed.
    geometry = [
        # Size 1.46, not 1.5: a 1.5 block centred on the lattice puts its edges
        # exactly ON the dx/2 lattice, where every cell support integrates pure
        # material and even chi1inv shows no blend — the interface check would
        # then measure nothing. 1.46 puts the edge at x=-0.27, inside the
        # support of the Ez column at x=-0.3.
        mp.Block(center=mp.Vector3(-1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=4.0, E_susceptibilities=[
                     mp.LorentzianSusceptibility(**LORENTZ)])),
        mp.Block(center=mp.Vector3(1.0, 0, 0), size=mp.Vector3(1.46, 1.46, mp.inf),
                 material=mp.Medium(epsilon=2.0, E_susceptibilities=[
                     mp.DrudeSusceptibility(**DRUDE)])),
    ]
    sim = mp.Simulation(cell_size=CELL, resolution=RESOLUTION, geometry=geometry,
                        symmetries=list(symmetries), eps_averaging=True,
                        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.4),
                                           component=mp.Ez, center=mp.Vector3())])
    sim.init_sim()
    return sim


def main() -> int:
    sim = build()
    fields = sim.fields
    if not hasattr(fields, "get_susceptibility_sigma"):
        print("FATAL: this MEEP has no get_susceptibility_sigma — not the patched "
              "build. Nothing was validated.", flush=True)
        return 2

    # --- 1. enumeration and identification -------------------------------------
    n_e = fields.get_num_susceptibilities(mp.E_stuff)
    n_h = fields.get_num_susceptibilities(mp.H_stuff)
    check("enumeration", n_e == 2 and n_h == 0,
          f"E chain length {n_e} (want 2), H chain length {n_h} (want 0)")

    params = {n: list(fields.get_susceptibility_params(mp.E_stuff, n)) for n in range(n_e)}
    print(f"       params by entry: {params}", flush=True)
    # Lorentzian payload mirrors dump_params: {4, id, omega_0, gamma, drude_flag}.
    by_freq = {}
    for n, row in params.items():
        check(f"params[{n}] shape", len(row) == 5 and row[0] == 4.0,
              f"{row} (want 5 values led by kind tag 4)")
        by_freq[round(row[2], 9)] = (n, row)
    lor = by_freq.get(LORENTZ["frequency"])
    dru = by_freq.get(DRUDE["frequency"])
    check("identification", lor is not None and dru is not None
          and abs(lor[1][3] - LORENTZ["gamma"]) < 1e-6 and lor[1][4] == 0.0
          and abs(dru[1][3] - DRUDE["gamma"]) < 1e-6 and dru[1][4] == 1.0,
          f"lorentz entry {lor}, drude entry {dru} "
          f"(gamma + the no_omega_0_denominator drude flag must match declarations)")
    if failures:
        return 1
    n_lor, n_dru = lor[0], dru[0]

    # --- 2. bulk values at each component's own Yee site ------------------------
    # Integer lattice: unit dx/2; component c's first site at little_corner + iyee.
    gv = fields.gv
    lattice = 2.0 * RESOLUTION

    def ivec_at(comp, x, y):
        shift = gv.iyee_shift(comp)
        ix = int(round(x * lattice)); iy = int(round(y * lattice))
        # snap onto this component's parity, moving less than one cell
        if (ix - shift.x()) % 2: ix += 1
        if (iy - shift.y()) % 2: iy += 1
        # THREE args always: mp.ivec(a, b) resolves to the (ndim, val)
        # constructor through SWIG, yielding a garbage-dimension ivec that reads
        # vacuum at real points and segfaults on chain entry 1 (measured). The
        # D2 grid_volume ignores the z entry of a D3 ivec.
        return mp.ivec(ix, iy, 0)

    def sigma_at(comp, d, x, y, n):
        return fields.get_susceptibility_sigma(comp, d, ivec_at(comp, x, y), n)

    dirs = {mp.Ex: mp.X, mp.Ey: mp.Y, mp.Ez: mp.Z}
    bulk_ok = True
    detail = []
    for comp, own in dirs.items():
        a = sigma_at(comp, own, -1.0, 0.0, n_lor)   # block 1 interior
        b = sigma_at(comp, own, -1.0, 0.0, n_dru)
        c = sigma_at(comp, own, +1.0, 0.0, n_lor)   # block 2 interior
        d = sigma_at(comp, own, +1.0, 0.0, n_dru)
        v = sigma_at(comp, own, 0.0, 1.8, n_lor)    # vacuum corner
        w = sigma_at(comp, own, 0.0, 1.8, n_dru)
        ok = (abs(a - LORENTZ["sigma"]) < 1e-6 and b == 0.0
              and c == 0.0 and abs(d - DRUDE["sigma"]) < 1e-6
              and v == 0.0 and w == 0.0)
        bulk_ok = bulk_ok and ok
        detail.append(f"{mp.component_name(comp)}: L-block ({a:.3g},{b:.3g}) "
                      f"D-block ({c:.3g},{d:.3g}) vacuum ({v:.3g},{w:.3g})")
    check("bulk", bulk_ok, "; ".join(detail))

    # --- 3. interface: sigma point-sampled while chi1inv is smoothed ------------
    # March Ez sites across block 1's x = -0.25 edge; every sigma must be exactly
    # one of {0, declared}, and chi1inv must take at least one intermediate value.
    xs = np.arange(-0.6, 0.11, 1.0 / RESOLUTION)
    sig_values = [sigma_at(mp.Ez, mp.Z, x, 0.0, n_lor) for x in xs]
    eps_values = [1.0 / fields.get_chi1inv(mp.Ez, mp.Z, ivec_at(mp.Ez, x, 0.0), 0).real
                  for x in xs]
    pure = all(abs(s) < 1e-9 or abs(s - LORENTZ["sigma"]) < 1e-6 for s in sig_values)
    both = (any(abs(s) < 1e-9 for s in sig_values)
            and any(abs(s - LORENTZ["sigma"]) < 1e-6 for s in sig_values))
    blended = any(1.0 + 1e-3 < e < 4.0 - 1e-3 for e in eps_values)
    check("interface point-sampling", pure and both and blended,
          f"sigma across the edge {['%.3g' % s for s in sig_values]} (never a blend); "
          f"eps at the same sites {['%.3g' % e for e in eps_values]} (IS smoothed)")

    # --- 4. consistency identity against the audited chi1inv(freq) path ---------
    # At a bulk point: 1/chi1inv(c,c,f) == eps_inf + chi1(f, sigma_read), with
    # chi1 per susceptibility.cpp (Lorentz: sigma*w0^2/(w0^2 - f^2 - i g f)).
    f_probe = 0.7
    loc = ivec_at(mp.Ez, -1.0, 0.0)
    eps_meep = 1.0 / complex(fields.get_chi1inv(mp.Ez, mp.Z, loc, f_probe))
    sigma_read = fields.get_susceptibility_sigma(mp.Ez, mp.Z, loc, n_lor)
    w0, g = LORENTZ["frequency"], LORENTZ["gamma"]
    chi1 = sigma_read * w0 * w0 / (w0 * w0 - f_probe * f_probe - 1j * g * f_probe)
    eps_inf = 1.0 / fields.get_chi1inv(mp.Ez, mp.Z, loc, 0).real
    eps_model = eps_inf + chi1
    rel = abs(eps_meep - eps_model) / abs(eps_meep)
    check("consistency identity", rel < 1e-5,
          f"1/chi1inv({f_probe}) = {eps_meep:.6f} vs eps_inf + chi1(sigma_read) = "
          f"{eps_model:.6f} (rel {rel:.2e})")

    # --- 5. symmetry: fields-level reads through S.transform unchanged ----------
    sim_sym = build(symmetries=[mp.Mirror(mp.Y)])
    fsym = sim_sym.fields
    sym_ok = True
    sdetail = []
    for y in (0.6, -0.6):  # the negative-y point exists only through the transform
        a = fields.get_susceptibility_sigma(mp.Ez, mp.Z, ivec_at(mp.Ez, -1.0, y), n_lor)
        b = fsym.get_susceptibility_sigma(mp.Ez, mp.Z, ivec_at(mp.Ez, -1.0, y), n_lor)
        sym_ok = sym_ok and abs(a - b) < 1e-9
        sdetail.append(f"y={y:+.1f}: full {a:.3g} folded {b:.3g}")
    check("symmetry", sym_ok, "; ".join(sdetail))

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
