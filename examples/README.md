# Examples

MEEP scripts stepped through `run_on_gpu`, each with its reference runs beside it
and its recorded output. Each script builds an ordinary `mp.Simulation` and is
short enough to read in one screen.

| Script | What it is | Grid at the default resolution |
|---|---|---|
| `quickstart.py` | the README's first run: one call, one flux spectrum | 48 × 48 × 48, 110,592 cells |
| `sphere_flux_3d.py` | a dielectric sphere under PML with one flux monitor; the problem class the package is aimed at, at its smallest measured size | 48 × 48 × 48, 110,592 cells |
| `slab_flux_2d.py` | a dielectric slab waveguide with two flux planes | 200 × 120, 24,000 cells |
| `dispersive_slab_2d.py` | a Lorentzian slab: a dispersive material, so every step also advances a polarization | 160 × 120, 19,200 cells |

The two-dimensional examples show the calls. They are not a speed-up: problems of
that size are faster on MEEP itself.

## Run one

Install MEEP and this package first ([INSTALL.md](../INSTALL.md)). Then, from
the repository root:

    python examples/quickstart.py
    python examples/sphere_flux_3d.py                   # this host's GPU
    python examples/sphere_flux_3d.py --leg array       # the array path, dispatch off
    python examples/sphere_flux_3d.py --leg reference   # the NumPy reference
    python examples/sphere_flux_3d.py --leg meep        # MEEP itself
    python examples/sphere_flux_3d.py --resolution 32   # 2,097,152 cells

The four legs are the same simulation stepped four ways:

| Leg | Call | What steps it |
|---|---|---|
| `gpu` | `run_on_gpu(sim, until=...)` | this host's GPU: compiled kernels where a certified one covers the configuration, the array path elsewhere |
| `array` | the same call with `MEEP_GPU_DISPATCH=0` | the array path of the same GPU route: CuPy on the device on NVIDIA hardware, NumPy on the host on Apple hardware |
| `reference` | `run_on_gpu(sim, until=..., prefer_gpu=False)` | the NumPy reference, on any host |
| `meep` | `sim.run(until=...)` | MEEP |

On a host with no GPU the `gpu` and `array` legs raise, and the message names
`prefer_gpu=False`. The `reference` and `meep` legs run on any host.

## Run all of them, and check

    python examples/run_examples.py --check

runs every leg of every example in its own process, compares the legs, and
compares the `[result]` lines with the recorded output under `expected/`. On a
host with no GPU it runs and checks the `reference` and `meep` legs only. The exit
status is 0 when every judged comparison holds.

| Comparison | Judged how |
|---|---|
| `gpu` against `array` | every byte of the final field and of each flux spectrum must be identical |
| `gpu` against `reference` | reported, not judged |
| `reference` against `meep` | the relative difference must be within the example's band: 1e-4 for `sphere_flux_3d` and `slab_flux_2d`, 2e-3 for `dispersive_slab_2d`. Judged only when MEEP is a single-precision build; otherwise reported |
| `[result]` lines against `expected/` | text identical, numbers within 1e-4 of the largest number on the line |

One leg runs per process because a run that dispatches compiled kernels installs
its kernel table's float32 subnormal policy for the whole process. The `array` leg
is run under the policy the `gpu` leg reported, which is the condition the
package's identity holds under. The `reference` leg is run as a user would run it,
under the host's own default policy.

## The recorded output

`expected/` holds the output of `run_examples.py --record` on one Apple M1 Max
with MEEP 1.33.0 built in single precision, PyTorch 2.10.0 and Metal frontend
32023.850.10, on 2026-09-28. It has not been recorded on NVIDIA hardware. On one
NVIDIA RTX A6000 (conda-forge MEEP 1.33.0, the Triton table), the 3 examples that
compare with MEEP matched the recorded output with 0 mismatches on 2026-09-28, the
`gpu` leg identical to the `array` leg in 3 of 3; their comparison with MEEP (a
double-precision build) was reported, not judged. On the CuPy route the `array` leg points
`CUPY_CACHE_DIR` at the package's keep-policy cache before it installs the policy
by hand, as dispatch does for the `gpu` leg; without that, CuPy refuses the `keep`
policy. What the recorded output shows, for 3 of 3 examples:

| Example | `gpu` against `array` | `gpu` against `reference`, final field | `reference` against MEEP, final field | `reference` against MEEP, flux |
|---|---|---|---|---|
| `sphere_flux_3d` | identical bytes | 1.4e-5 | 6.7e-6 | 3.9e-9 |
| `slab_flux_2d` | identical bytes | 1.0e-5 | 7.9e-6 | 3.8e-7 |
| `dispersive_slab_2d` | identical bytes | 8.4e-6 | 8.1e-4 | 3.3e-5 |

Each figure is the L2 norm of the difference over the L2 norm of the second run's
result.

- **The `gpu` and `reference` legs differ, and that is the subnormal policy.** On
  Apple hardware the Metal kernel table is certified under the `flush` policy and
  the host's default resolves to `keep`. The `array` leg under `flush` reproduces
  the `gpu` leg byte for byte; the `reference` leg under `keep` differs from both
  by about one part in 10^5 of the final field.
- **The dispersive example's band is wider because its run ends late.** The engine
  and the MEEP build compared against both step single precision and differ by a
  few parts in a million of the field's peak. That example ends after the pulse
  has left the cell, when the field that remains is 0.3 % of its peak, so the same
  absolute difference is a larger fraction of what is left.
- **The wall times are not timings of record.** They were taken on a loaded
  machine, and a run of a few hundred steps includes the one-time planning and
  kernel compilation at its first step. The throughput rates in
  [Will it help?](../docs/guides/will-it-help.md) were timed over steady-state
  windows, with the host conditions stated beside them;
  `parity/meep_gpu/bench_fused_products.py` is the instrument for that.

`[host]` lines state which path served each run. `step path: fused` means
compiled kernels served the step; `step path: array` means the array path did. A
host outside the certified identities is refused by name and steps on the array
path, which is a correct run.
