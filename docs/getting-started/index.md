# Getting Started

One path, in order. Each step links the page that finishes it.

1. **Check that your problem fits.** The package is aimed at large
   three-dimensional problems on one GPU. For two-dimensional problems, small
   grids and short runs, expect MEEP itself to be as fast or faster (most of
   these have not been timed against MEEP): read
   [Will it help?](../guides/will-it-help.md) first.
2. **Install MEEP, then this package, then run the check.** MEEP goes first; it
   is not on the Python package index.
   [Installation](installation.md) gives the route for your machine and sends you
   to INSTALL.md, where each route is written out in full. The check must end
   with a line starting `OK:`.
3. **Run a first simulation.** [First lifted simulation](first-lift.md) runs
   `examples/quickstart.py`, the same script as the README: an ordinary MEEP
   simulation, `gpu_compatibility(sim)`, then `run_on_gpu(sim, until=...)`
   instead of `sim.run(until=...)`.
4. **Read what ran.** A run prints a line on standard error that says which
   kernels served it, and `result.driver.fast_path_report()` says why.
   [Reading what ran](reading-what-ran.md) explains both.
5. **Move your own script over.** [MEEP user entry points](meep-entry-points.md)
   covers the three calls: a preflight, a one-call run, and a lift for runs in
   segments. [Compatibility and refusals](../guides/compatibility.md) says what
   is accepted and what is refused by name.
6. **Measure your own case** before relying on a speed-up:
   [Time your own simulation](../guides/time-your-simulation.md).

Two defaults to know from the start:

- `run_on_gpu` and `lift_simulation` run on this host's GPU and raise on a host
  that has none. Pass `prefer_gpu=False` for the NumPy reference: it runs on the
  host CPU on every host and never runs a compiled kernel. On the one Mac
  measured it was faster than the GPU for a small cell.
- Compiled kernels serve the sub-steps they cover, on a certified GPU; the rest
  of each step runs on the array path, the same equations in NumPy or CuPy.
  `MEEP_GPU_DISPATCH=0` keeps a whole run on the array path.

When something fails, [Troubleshooting](../guides/troubleshooting.md) is
organized by the message you see. Terms such as array path, kernel table and
sub-step are defined in the [Glossary](../reference/glossary.md).
