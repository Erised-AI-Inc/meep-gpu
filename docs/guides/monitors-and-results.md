# Monitors and results

## Migrated monitor specifications

The lift rebuilds supported monitor specifications that were already attached
to the MEEP simulation. Their frequency lists, regions, and decimation choices
are read from the MEEP object rather than reconstructed from a separate user
description.

The result keeps the original MEEP monitor object as the lookup key:

    flux = sim.add_flux(frequency, 0, 1, region)
    result = run_on_gpu(sim, until=200)
    try:
        spectrum = result.get_flux_spectrum(flux)
    finally:
        result.close()

Use <code>result.monitor_for(meep_monitor)</code> when the driver-side monitor
object itself is needed. It raises for an object that was not migrated instead
of returning an empty measurement.

Lookup is by object identity, and a result keeps its MEEP monitor objects alive
for as long as it is readable. That matters when a script holds two runs at once,
as the two-run normalization idiom does: without the reference, a first run's
monitors could be collected and a second run's allocated at the same addresses,
and a lookup keyed on the address alone would return the first run's spectrum for
the second run's monitor. A monitor belonging to another run raises here.

Each migrated kind has a reader named after MEEP's own:

| MEEP | here |
| --- | --- |
| <code>mp.get_fluxes(flux)</code> | <code>result.get_flux_spectrum(flux)</code> |
| <code>mp.get_forces(force)</code> | <code>result.get_forces(force)</code> |
| <code>mp.get_electric_energy(energy)</code> | <code>result.get_electric_energy(energy)</code> |
| <code>mp.get_magnetic_energy(energy)</code> | <code>result.get_magnetic_energy(energy)</code> |
| <code>mp.get_total_energy(energy)</code> | <code>result.get_total_energy(energy)</code> |
| <code>sim.get_force_data(force)</code> | <code>result.get_force_data(force)</code> |
| <code>sim.load_force_data(force, d)</code> | <code>result.load_force_data(force, d)</code> |
| <code>sim.get_farfields(n2f, ...)</code> | <code>result.load_near2far(sim, n2f)</code>, then MEEP's own call |

A <code>ForceRegion</code> is limited to MEEP's diagonal stress-tensor branch, where
the region's declared force direction is also its surface normal. A region whose
declared direction differs from its normal, a force monitor on a run declaring a
symmetry, and a force monitor on a cylindrical cell are each refused by name at
preflight rather than approximated. An <code>EnergyRegion</code> has no such
restriction and is served under a mirror fold, because MEEP's
<code>add_dft_energy</code> does not reduce its region list the way
<code>add_dft_force</code> does. MEEP ignores an <code>EnergyRegion</code>'s
<code>weight</code>, and so does the lift.

## Two-run normalization

MEEP's normalization idiom — run once without the structure, save the flux
transform, run again with it and subtract — is spelled with the same two pieces
here. <code>result.get_flux_data(flux)</code> is MEEP's
<code>sim.get_flux_data(flux)</code>: an opaque transform, not a spectrum.

    empty = run_on_gpu(sim_without_structure, until=200)
    reference = empty.get_flux_data(flux)
    empty_spectrum = empty.get_flux_spectrum(flux)
    empty.close()

    # `flux2` is the second simulation's OWN add_flux object.
    result = run_on_gpu(sim_with_structure, until=200,
                        minus_flux_data=[(flux2, reference)])
    reflectance = -result.get_flux_spectrum(flux2) / empty_spectrum

<code>minus_flux_data</code> is MEEP's <code>sim.load_minus_flux_data</code>:
the saved transform is loaded and negated, so the second run ends at
<code>E2 - E1</code> and <code>H2 - H1</code> and the reported flux is the
reflected wave's own power. It is **not** a difference of two powers.
<code>flux_data</code> is the non-negated form, MEEP's
<code>sim.load_flux_data</code>.

Declare the load to <code>run_on_gpu</code> rather than calling MEEP's method on
the second simulation. MEEP's method reaches the flux object's lazy SWIG handle,
which calls <code>init_sim()</code>, and an already-initialized simulation is a
documented refusal of the lift.

## Field arrays

<code>result.get_array("Ez")</code> returns a host NumPy copy in MEEP's field
array layout, including the package's documented handling of boundary planes and
mirror unfolding. This is the direct comparison form:

    gpu_ez = result.get_array("Ez")
    cpu_ez = sim.get_array(component=mp.Ez)

<code>result.get_field("Ez")</code> returns the driver's cell-centred field
layout instead. Do not compare these two representations by broadcasting or
casual slicing: they serve different layout contracts.

## DFT and near-to-far results

For a migrated DFT-field monitor,
<code>result.get_dft_region(meep_fields, component, freq_index)</code> returns
the region accumulated by the driver. It intentionally does not apply MEEP's
read-time array collapse. Keep the requested
region and its expected dimensionality alongside any comparison.

<code>result.get_dft_array(meep_fields, component, freq_index)</code> is the
counterpart of MEEP's own reader: it applies MEEP's read-time reduction on top of that
region — the interpolation weights of a zero-extent axis, the collapse of that
axis, and the drop of an axis one cell wide — so it returns what
<code>sim.get_dft_array</code> returns, shape included. Use it when
transliterating a script line by line; use <code>get_dft_region</code> when you
want the samples the accumulator actually holds.

For a migrated flux monitor,
<code>result.pack_flux_data(mp, meep_flux)</code> returns this run's transform in
MEEP's own <code>FluxData</code> layout, so
<code>sim.load_flux_data(flux, packed)</code> writes it into MEEP's accumulator
and MEEP's unchanged <code>sim.get_eigenmode_coefficients</code> then reads this
engine's fields. The simulation must be initialized (<code>sim.init_sim()</code>)
before the load, which is why it is done after the run rather than before it.
Both <code>sim.add_flux</code> and <code>sim.add_mode_monitor</code> are served,
including on a cell with mirror symmetry, where <code>add_flux</code> registers
only half the plane: the packer takes each chunk's own integration weight from
MEEP, so the halved region's edge weights are MEEP's rather than a model of them.
MEEP still applies its own restriction — a folded monitor cannot be decomposed
against an unpolarised mode, so pass an <code>eig_parity</code>.

For a migrated near-to-far monitor,
<code>result.load_near2far(sim, meep_near2far)</code> places the accumulated
near-surface DFT into MEEP's existing near-to-far object. Continue the
far-field evaluation through MEEP. This keeps the package focused on the
grid-wide accumulation that benefits from acceleration and avoids a duplicate
implementation of MEEP's host-side Green-function evaluation.

## Direct driver monitors

The direct driver exposes methods to add DFT and flux monitors before running.
Its documented monitor methods accept explicit frequency lists or MEEP-style
centre-bandwidth-count inputs. For a custom driver run, specify a monitor region
and then keep the returned monitor object with the result artifact.

## Compiled-kernel dispatch and these results

Compiled-kernel dispatch is on by default on a driver built with
<code>prefer_gpu=True</code>, and <code>MEEP_GPU_DISPATCH=0</code> turns it off; a
<code>prefer_gpu=False</code> driver never dispatches. The choice changes which
implementation performs the field updates. Against the array path run under
the same subnormal policy, a monitor reads the same values, bit for bit;
<code>MEEP_GPU_DISPATCH=0</code> alone does not install that policy, so a run
whose values pass through the float32 subnormal range can differ in the last
bits ([Floating point](../design/floating-point.md)). The route
gates drive each fused product through the driver's own consults and byte-compare
whole runs against the array path, with per-step launch counts proving the
substitution actually happened; every released arm additionally carries a
per-family device gate certifying its arithmetic. Monitor accumulation reads the
same field state either way.

That is a correctness statement and not a throughput one. See
[Compiled-kernel dispatch](kernel-dispatch.md) for what the timing
measurements do and do not support.

> **Warning: Do not use a zero result as a monitor-health check**
>
> A physically valid flux can be zero. Validate monitor placement and
> registration against a known, nondegenerate case before using a spectrum as
> a regression or performance signal.
