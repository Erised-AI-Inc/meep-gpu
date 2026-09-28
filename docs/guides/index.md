# Using meep-gpu

The guides in this section describe the contract around a lifted run:

- [Will it help?](will-it-help.md) states where the GPU has been measured
  faster than MEEP and where to expect no gain, with the one comparison on an
  identical problem and the status of every number.
- [Time your own simulation](time-your-simulation.md) measures your own problem
  on this host's GPU, on the NumPy reference and on MEEP.
- [Compatibility and refusals](compatibility.md) explains what lifts,
  what is refused by name, and why a preflight is mandatory for a batch or
  service workflow.
- [Monitors and results](monitors-and-results.md) explains which
  monitor specifications migrate and which result readback helpers preserve
  MEEP conventions.
- [Compiled-kernel dispatch](kernel-dispatch.md) covers the compiled
  kernels that run inside the step loop by default. It explains how to turn them
  off, which of the three kernel tables your hardware can use, how to read what a
  run actually dispatched, and what the coverage and timing numbers support.
- [Troubleshooting](troubleshooting.md) is organized by the message you see.
- [Validation and performance](validation-and-performance.md) explains
  how to treat the NumPy path, CPU MEEP, the CuPy array path, and the compiled
  kernel tables as distinct numerical and performance comparisons.
- [Capability coverage checklist](capability-coverage.md) records the
  lift surface area by area, conditional combinations, and deferred work.

The [public API reference](../reference/public-api.md) provides the exact
callable surface, and the [Glossary](../reference/glossary.md) defines the terms.
The guides define how to use it without weakening the package's correctness
boundary.
