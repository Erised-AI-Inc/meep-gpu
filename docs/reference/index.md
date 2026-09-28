# Reference

The public surface is intentionally small — 35 names in <code>__all__</code>:

| Area | Public entry points |
|---|---|
| Capability | <code>is_available</code>, <code>available_gpu</code>, <code>missing_dependencies</code>, <code>cupy_available</code>, <code>resolve_backend</code>, <code>to_numpy</code> |
| MEEP lift | <code>gpu_compatibility</code>, <code>lift_simulation</code>, <code>run_on_gpu</code> |
| Result types | <code>GpuCompatibility</code>, <code>GpuRunResult</code>, <code>MigratedMonitors</code>, <code>StructuredSigmaLift</code> |
| Refusals to catch | <code>MeepSimulationNotLiftable</code>, <code>StepFunctionNotHosted</code>, <code>FdtdCancelled</code>, <code>FdtdDivergence</code>, <code>FdtdNonlinearityOutOfRange</code> |
| Direct solver | <code>FdtdDriver</code>, <code>Mirror</code>, <code>AbsorberLayer</code> |
| Subnormal policy | <code>FLUSH</code>, <code>KEEP</code>, <code>MATCH_MEEP</code>, <code>default_policy</code>, <code>resolve_match_meep</code>, <code>install_subnormal_policy</code>, <code>get_subnormal_policy</code>, <code>set_subnormal_policy</code>, <code>policy_stamp</code>, <code>SubnormalPolicyLocked</code>, <code>SubnormalPolicyUnattainable</code> |
| Harmonic inversion | <code>Harminv</code>, <code>Mode</code>, <code>do_harminv</code> |

Read [Public Python API](public-api.md) for the contracts users should
rely on. Internal modules, test helpers, parity harnesses, and kernel sources are
deliberately not public workflow APIs. One exception a caller can meet is not in
<code>__all__</code>: <code>meep_gpu.dispersion.DispersionInstability</code>, raised
by a lift for an unstable dispersive pole
([Refusals a caller catches](public-api.md#refusals-a-caller-catches)).

The [Glossary](glossary.md) defines the terms the manual and the package's
messages use: array path, kernel table, sub-step, dispatch, certified identity.

If the starting point is an existing <code>mp.Simulation</code>, read
[MEEP user entry points](../getting-started/meep-entry-points.md)
first. It explains when each lift entry point owns initialization, execution,
result readback, and device-memory cleanup.
