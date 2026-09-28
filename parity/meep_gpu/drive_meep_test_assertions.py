"""Run MEEP's own test assertions against this engine's output.

Field-vs-field parity answers "do two codes stepping the same object agree". This
answers something stronger and different: **does this engine satisfy the numerical
oracles MEEP's developers wrote** — the hard-coded constants, tolerances and physics
invariants inside ``python/tests``, authored years before this package existed and
never adjusted for it.

The mechanism, and its limits, stated plainly:

* The test method is executed twice in the same process — once **unmodified** (the
  control: does this test pass on stock MEEP in this environment at all?) and once
  **driven**, with ``mp.Simulation.run`` replaced by a call to
  :func:`meep_gpu.run_on_gpu` on the very object the test built.
* In the driven pass the simulation's result readers are rebound to the driver's
  accumulators — ``get_fluxes`` to the migrated ``DftFlux`` spectra, ``DftNear2Far``
  loaded back into MEEP's own object so MEEP's unchanged far-field evaluator runs on
  our near fields, ``get_array``/``get_array_slice`` to the driver's field arrays.
  Everything downstream of that — every ``self.assertAlmostEqual(...)`` in MEEP's
  file — is MEEP's code comparing MEEP's published constant against our number.
* The **two-run normalization idiom** — run once without the structure, save the
  flux transform, run again with it, subtract — is driven end to end.
  ``sim.get_flux_data`` is served from the first run's migrated monitor and
  ``sim.load_minus_flux_data`` is recorded and handed to
  ``run_on_gpu(minus_flux_data=...)`` rather than executed against MEEP, whose own
  methods would both initialize the simulation (refusing the lift) and hand back an
  empty accumulator. See ``install_driven_run``.
* Each ``assert*`` call is wrapped so its outcome is recorded individually and a
  failure does **not** abort the rest, because the quantity of interest is how many
  of the file's assertions hold, not whether the first one does. Re-entrant calls
  (``assertClose`` calls ``assertLessEqual``) are counted once, at the outer call.
* **Nothing under the MEEP checkout is read-write.** The interception is on
  ``mp.Simulation`` and on the ``TestCase`` instance this harness constructs. No
  test file is edited, and no test is rewritten to suit the engine. The one input a
  driven pass alters is an ``amp_func_file`` source, whose HDF5 dataset is read here
  and handed over as the ``amp_data`` array MEEP's own file route builds from it —
  an I/O step the package deliberately has no dependency for. It is recorded per row;
  see :func:`substitute_amp_func_file`.

* **MEEP's step functions run here.** ``mp.after_sources(mp.Harminv(...))``,
  ``mp.at_every``, ``mp.at_beginning``, ``mp.at_end``, ``mp.after_time`` and
  ``mp.combine_step_funcs`` are forwarded to ``run_on_gpu(step_functions=...)``, which
  calls them on this engine's stepper through a facade answering the ``mp.Simulation``
  surface they were written against. A wrapper whose only leaves are MEEP's HDF5
  output writers is DROPPED with a recorded note — it would write MEEP's own,
  un-stepped arrays — and the rest are refused by name. See
  :func:`triage_step_functions`.

What cannot be driven, and is recorded as such rather than silently skipped:
``mp.dft_ldos``, ``mp.PadeDFT``, ``mp.synchronized_magnetic``, ``mp.with_prefix``,
``mp.in_volume`` and the HDF5 writers (they need machinery the driver has no
counterpart for); any test that calls ``run()`` more than once (``run_on_gpu`` lifts
afresh and cannot resume); and any reader the driver has no counterpart for. Those
rows carry ``not_driven_reason`` and stay parity-only.

Used through ``survey_meep_tests.py --stage assertions``; ``child_assertions`` is its
per-case entry point.
"""

from __future__ import annotations

import json
import math
import time

# Assertion helpers that are context managers or take a callable; wrapping them would
# record "passed" at construction time rather than at the check.
_NOT_VALUE_ASSERTS = {
    "assertRaises", "assertRaisesRegex", "assertRaisesRegexp", "assertWarns",
    "assertWarnsRegex", "assertLogs", "assertNoLogs",
}


class UnsupportedDrive(Exception):
    """This test's run cannot be served by the driver; the row stays parity-only."""


# --- assertion instrumentation --------------------------------------------------------


def instrument_assertions(case) -> list[dict]:
    """Wrap every value assertion on one TestCase instance; return the growing log.

    A failure is recorded and swallowed so the remaining assertions in the method
    still execute. That is the whole point of the measurement — "42 of 47 MEEP-authored
    assertions hold" is the number, and stopping at the first failure would only ever
    report 1 or 0.
    """
    records: list[dict] = []
    depth = {"n": 0}

    def wrap(name: str, original):
        def wrapper(*args, **kwargs):
            if depth["n"]:  # assertClose -> assertLessEqual: count the outer call only.
                return original(*args, **kwargs)
            depth["n"] += 1
            entry = {"assertion": name, "args": _summarize(args), "kwargs": _summarize_kwargs(kwargs)}
            try:
                original(*args, **kwargs)
                entry["passed"] = True
            except AssertionError as exc:
                entry["passed"] = False
                entry["detail"] = str(exc)[:400]
            except BaseException as exc:  # noqa: BLE001 - an assertion that could not evaluate.
                entry["passed"] = None
                entry["detail"] = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                depth["n"] -= 1
                records.append(entry)
            return None
        return wrapper

    for name in dir(type(case)):
        if not name.startswith("assert") or name in _NOT_VALUE_ASSERTS:
            continue
        original = getattr(case, name, None)
        if not callable(original):
            continue
        setattr(case, name, wrap(name, original))
    return records


def _summarize(values) -> list[str]:
    out = []
    for value in values:
        try:
            if isinstance(value, (complex, float, int)):
                out.append(repr(value))
            elif hasattr(value, "shape"):
                out.append(f"array{tuple(value.shape)}")
            else:
                out.append(repr(value)[:120])
        except BaseException:  # noqa: BLE001
            out.append("<unrepresentable>")
    return out[:6]


def _summarize_kwargs(kwargs) -> dict:
    return {key: repr(value)[:80] for key, value in kwargs.items() if key != "msg"}


# --- driving the run through the engine ------------------------------------------------


def bind_engine_readers(mp, sim, result, notes: list[str], loaders: dict | None = None) -> None:
    """Point the simulation's result readers at the driver's accumulators.

    Only the readers the package publishes counterparts for
    (``from_meep.MIGRATABLE_MONITORS``) are rebound. Anything else keeps MEEP's own
    method, which will read an un-stepped MEEP grid — so those readers are replaced
    by a raiser instead, and a test that reaches one is recorded as not driven rather
    than quietly compared against zeros. Silently returning MEEP's empty answer is the
    exact failure mode that would make this measurement worthless.
    """
    import numpy

    # DftFlux -> the migrated flux spectrum. MEEP's own get_fluxes returns a list of
    # floats per frequency; so does the driver's spectrum.
    def get_fluxes(flux_object):
        return [float(numpy.real(value)) for value in result.get_flux_spectrum(flux_object)]

    sim.get_fluxes = get_fluxes
    notes.append("get_fluxes -> result.get_flux_spectrum")

    # DftForce / DftEnergy -> the migrated spectra. Same shape of claim as get_fluxes:
    # MEEP's own mp.get_forces / mp.get_electric_energy read the C++ accumulator, which
    # on a lifted run was never stepped and is all zeros — a full spectrum of zeros that
    # a test would compare against its published constant and fail for the wrong reason.
    def get_forces(force_object):
        return [float(numpy.real(value)) for value in result.get_forces(force_object)]

    sim.get_forces = get_forces

    # sim.get_force_data / load_force_data on OUR accumulators, for the same reason the
    # flux-data pair is intercepted: MEEP's methods reach `force.diag`, which is empty
    # here, so a test's store/load round trip would save zeros, load zeros, and still
    # read the right answer from the migrated monitor — a no-op dressed as a round trip.
    # test_force.py:38-41 does exactly that round trip.
    def get_force_data(force_object):
        return result.get_force_data(force_object)

    def load_force_data(force_object, fdata):
        result.load_force_data(force_object, fdata)

    sim.get_force_data = get_force_data
    sim.load_force_data = load_force_data

    def display_forces(*force_objects):
        # MEEP's own display_forces calls mp.get_forces on each object and prints a CSV
        # row; the module-level override below serves those, so this only has to keep
        # the frequency column off MEEP's lazy swigobj.
        for force_object in force_objects:
            get_forces(force_object)

    sim.display_forces = display_forces
    notes.append("get_forces / get_force_data / load_force_data / display_forces -> "
                 "the migrated DftForce monitor")

    for reader_name, engine_reader in (
        ("get_electric_energy", "get_electric_energy"),
        ("get_magnetic_energy", "get_magnetic_energy"),
        ("get_total_energy", "get_total_energy"),
    ):
        def energy_reader(energy_object, _engine_reader=engine_reader):
            return [float(numpy.real(value))
                    for value in getattr(result, _engine_reader)(energy_object)]

        setattr(sim, reader_name, energy_reader)
    notes.append("get_electric_energy / get_magnetic_energy / get_total_energy -> "
                 "the migrated DftEnergy monitor")

    # DftNear2Far -> write our near-surface DFT into MEEP's own object, then leave
    # every far-field call as MEEP's unchanged code. This is the strongest form of
    # the claim available: MEEP's evaluator, our fields.
    loaded = 0
    for obj in (getattr(sim, "dft_objects", ()) or ()):
        if type(obj).__name__ == "DftNear2Far":
            result.load_near2far(sim, obj)
            loaded += 1
    if loaded:
        notes.append(f"load_near2far into {loaded} MEEP DftNear2Far object(s)")

    # STRUCTURE READS STAY MEEP'S. `mp.Dielectric` and `mp.Permeability` are not
    # fields: they are read out of the structure, which this engine never steps and
    # never modifies, so MEEP's own answer is the right one whichever code did the
    # stepping. Measured on test_array_metadata's own ring geometry (resolution 25,
    # the 2-D cell below): `sim.get_array(component=mp.Dielectric)` and
    # `sim.get_array_metadata()` are BIT-IDENTICAL between an `init_sim()`-only
    # simulation and the same simulation run to t=20 — 0 differing elements of
    # 235 225, max |difference| 0.0. The comment below already declared this intent
    # for `get_epsilon_point`; `sim.get_epsilon` routes through `get_array`, so
    # without this branch the intent was not implemented and the read was refused.
    original_get_array = sim.get_array
    original_get_field_point = getattr(sim, "get_field_point", None)
    structure_components = tuple(
        value for value in (getattr(mp, "Dielectric", None), getattr(mp, "Permeability", None))
        if value is not None)

    def _is_structure(component):
        try:
            return component in structure_components
        except BaseException:  # noqa: BLE001
            return False

    # Field arrays. MEEP's get_array over the whole cell has exactly the layout the
    # driver produces; a sub-volume request is cut out of it by MATCHING COORDINATES
    # from MEEP's own get_array_metadata on both the whole cell and the request, so
    # the index window comes from MEEP rather than from an assumption here.
    def get_array(component=None, vol=None, center=None, size=None, cmplx=None,
                  arr=None, frequency=0, snap=False):
        if _is_structure(component):
            return original_get_array(component=component, vol=vol, center=center, size=size,
                                      cmplx=cmplx, arr=arr, frequency=frequency, snap=snap)
        name = _component_name(mp, component)
        if name is None:
            raise UnsupportedDrive(
                f"get_array asked for component {component!r}, which this driver does not "
                f"publish as a Cartesian field array")
        whole = numpy.asarray(result.get_array(name))
        if vol is not None or center is not None or size is not None:
            whole = slice_like_meep(mp, sim, whole, vol, center, size)
        values = whole if cmplx else whole.real
        single = bool(getattr(mp, "is_single_precision", lambda: False)())
        if cmplx:
            values = values.astype(numpy.complex64 if single else numpy.complex128)
        else:
            values = values.astype(numpy.float32 if single else numpy.float64)
        if arr is not None:
            # MEEP raises ValueError on a shape or dtype mismatch, and three of its
            # tests assert exactly that; reproducing the check keeps those oracles live.
            if tuple(arr.shape) != tuple(values.shape) or arr.dtype != values.dtype:
                raise ValueError(
                    f"supplied array of shape {arr.shape} dtype {arr.dtype} does not match "
                    f"the slice's {values.shape} {values.dtype}")
            arr[...] = values
            return arr
        return values

    sim.get_array = get_array

    def get_field_point(component, pt):
        # THE ONE READER THAT DOES NOT GO THROUGH get_array, and the reason is that
        # MEEP does not either. get_array reaches its answer through get_array_slice,
        # which averages each component's Yee values onto the CELL CENTRES first and
        # only then interpolates between those centres. get_field_point reaches its
        # answer through fields::get_field (monitor.cpp:127-142), a multilinear
        # interpolation of the RAW Yee values of that one component, and
        # averaging-then-interpolating is not the same operator as interpolating
        # directly unless the field is locally linear.
        #
        # So this calls the driver's OWN FdtdDriver.get_field_point, which is a
        # transcription of monitor.cpp — same stencil, same weights, same symmetry
        # phase and Bloch translation — rather than slicing the centred array. Cutting
        # the centred array was the previous route and it could only answer points
        # that happened to land on a cell centre: it refused both of MEEP's own
        # absorber oracles, test_absorber (Ex at z = 0, an integer-lattice point of
        # Ex's own sub-lattice, half a cell off every centre) and test_absorber_2d
        # (Hz at (4.13, 3.75), on no lattice at all). Interpolation of the raw Yee
        # values is what those tests measure, and it is what the driver implements.
        #
        # The array slice remains the FALLBACK for anything the driver's reader
        # refuses — a point outside the cell, say — so nothing that used to be driven
        # stops being driven.
        if _is_structure(component) and original_get_field_point is not None:
            return original_get_field_point(component, pt)
        name = _component_name(mp, component)
        if name is None:
            raise UnsupportedDrive(
                f"get_field_point asked for component {component!r}, which this driver does "
                f"not publish as a Cartesian field array")
        driver = getattr(result, "driver", None)
        if driver is not None:
            try:
                return complex(driver.get_field_point(
                    name, (float(pt.x), float(pt.y), float(pt.z))))
            except BaseException:  # noqa: BLE001 — fall back to the centred-array slice.
                pass
        whole = numpy.asarray(result.get_array(name))
        value = slice_like_meep(mp, sim, whole, mp.Volume(center=pt, size=mp.Vector3()),
                                None, None, collapse_empty_axes=False)
        return complex(numpy.asarray(value).reshape(-1)[0])

    sim.get_field_point = get_field_point
    notes.append("get_field_point -> FdtdDriver.get_field_point (monitor.cpp stencil)")

    # sim.fields.get_field(component, mp.vec(...)) — the LOWER of MEEP's two spellings,
    # and the one test_cyl_ellipsoid's only assertion uses (test_cyl_ellipsoid.py:69).
    # Simulation.get_field_point is a thin wrapper over it (simulation.py:2675-2681), so
    # without this the higher spelling reads the driver and the lower one reads MEEP's
    # own initialized-but-never-stepped grid — a zero, in exactly the shape of a field.
    fields = getattr(sim, "fields", None)
    if fields is not None:
        cylindrical = bool(getattr(sim, "is_cylindrical", False))

        def fields_get_field(component, location, *rest):
            if rest:  # fields::get_field(c, vec, parallel) — the flag changes nothing here.
                pass
            if cylindrical:
                point = mp.Vector3(float(location.r()), 0.0, float(location.z()))
            else:
                point = mp.Vector3(float(location.x()), float(location.y()),
                                   float(location.z()))
            return get_field_point(component, point)

        try:
            fields.get_field = fields_get_field
            notes.append("fields.get_field -> the same FdtdDriver.get_field_point reader")
        except BaseException:  # noqa: BLE001 — a SWIG proxy that refuses new attributes.
            notes.append("fields.get_field could not be rebound; the lower spelling still "
                         "reads MEEP's un-stepped grid")

    # DftFields -> the migrated region, reduced the way MEEP reduces it. The whole
    # convention lives in the package (`from_meep.collapse_dft_region`), not here, so
    # a script transliterating `sim.get_dft_array` reaches the same code this does.
    def get_dft_array(dft_obj, component, num_freq):
        name = _component_name(mp, component)
        if name is None:
            raise UnsupportedDrive(
                f"get_dft_array asked for component {component!r}, which this driver does not "
                f"accumulate as a field component")
        return result.get_dft_array(dft_obj, name, int(num_freq))

    sim.get_dft_array = get_dft_array
    notes.append("get_dft_array -> result.get_dft_array (region + MEEP's read-time collapse)")

    # DftFlux -> MEEP's OWN object, so MEEP's unchanged MPB reader runs on our fields.
    # `get_eigenmode_coefficients` splits cleanly (src/mpb.cpp:925-1004): the mode comes
    # from MPB and depends only on epsilon and frequency, and the only place the stepped
    # fields enter is `get_mode_flux_overlap` against the dft_flux's accumulated chunks.
    # So the same move as `load_near2far` applies — write our transform into MEEP's
    # accumulator and leave the reader alone.
    #
    # LAZY, and that is the whole design of it. Loading needs `sim.init_sim()`, because
    # the driven pass deliberately never let MEEP initialize (an initialized simulation
    # is a documented refusal of the lift), and building MEEP's structures for every
    # driven flux row would put a full CPU-MEEP setup on the majority of this corpus for
    # nothing. Doing it on the FIRST eigenmode call costs it only where it buys
    # something, and it happens strictly after the lift, so it cannot contaminate it.
    eigenmode_originals = {name: getattr(sim, name, None)
                           for name in ("get_eigenmode_coefficients", "get_eigenmode")}
    flux_loaded = {"done": False}

    def _load_flux_into_meep(reader_name):
        if flux_loaded["done"]:
            return
        objects = [obj for obj in (getattr(sim, "dft_objects", ()) or ())
                   if type(obj).__name__ == "DftFlux"]
        if not objects:
            raise UnsupportedDrive(
                f"sim.{reader_name} was called on a simulation carrying no DftFlux this driver "
                f"migrated, so there is no accumulator to write our fields into")
        sim.init_sim()
        try:
            packed = [(obj, result.pack_flux_data(mp, obj)) for obj in objects]
        except BaseException as exc:  # noqa: BLE001 - a refusal, recorded as one.
            raise UnsupportedDrive(
                f"sim.{reader_name} needs this run's flux transform in MEEP's own layout and "
                f"pack_flux_data refused: {exc}") from exc
        loader = (loaders or {}).get("load_flux_data") or type(sim).load_flux_data
        for obj, data in packed:
            loader(sim, obj, data)
        flux_loaded["done"] = True
        notes.append(f"pack_flux_data -> MEEP's own load_flux_data on {len(packed)} DftFlux "
                     f"object(s); sim.{reader_name} is then MEEP's unchanged MPB reader "
                     f"running on this engine's accumulated fields")

    def eigenmode_reader(reader_name):
        original = eigenmode_originals[reader_name]

        def reader(*args, **kwargs):
            _load_flux_into_meep(reader_name)
            return original(*args, **kwargs)
        return reader

    for reader_name in ("get_eigenmode_coefficients", "get_eigenmode"):
        if eigenmode_originals[reader_name] is not None:
            setattr(sim, reader_name, eigenmode_reader(reader_name))

    # --- field-function reducers ---------------------------------------------------
    #
    # MEEP's instantaneous read-time measures: `fields::integrate` / `max_abs` and the
    # five `*_in_box` quantities built on them (src/integrate.cpp,
    # src/energy_and_flux.cpp). Nothing accumulates, so they read whatever the field
    # arrays hold — which on the driven pass is the DRIVER's, not MEEP's un-stepped grid.
    #
    # BOTH SPELLINGS are rebound, and that is not a nicety. `test_modal_volume_in_box`
    # asserts `sim.fields.modal_volume_in_box(vol)` against `sim.modal_volume_in_box()`,
    # i.e. the raw fields object against the Simulation wrapper. Rebinding only the
    # wrapper leaves the other half reading an `init_sim()`-only MEEP grid, which answers
    # 0/0 = nan for a modal volume — and `assertAlmostEqual(<anything>, nan)` FAILS. The
    # row would flip from not-driven to a spurious failure. Measured on an unstepped
    # simulation: electric_energy_in_box 0.0, electric_energy_max_in_box 0.0,
    # modal_volume_in_box nan, flux_in_box 0.0.
    reducer_driver = getattr(result, "driver", None)

    def _volume_bounds(where):
        """(center, size) for anything MEEP's reducers take, or None for the whole cell."""
        if where is None:
            return None, None
        center = getattr(where, "center", None)
        size = getattr(where, "size", None)
        if center is not None and size is not None and not callable(center):
            return ((float(center.x), float(center.y), float(center.z)),
                    (abs(float(size.x)), abs(float(size.y)), abs(float(size.z))))
        swig = getattr(where, "swigobj", where)  # A SWIG meep::volume.
        try:
            low, high = swig.get_min_corner(), swig.get_max_corner()
            dim = swig.dim
        except BaseException as exc:  # noqa: BLE001 — anything else is not a volume.
            raise UnsupportedDrive(
                f"a reducer was handed {where!r}, which is neither an mp.Volume nor a "
                f"SWIG meep::volume this harness can read bounds off") from exc
        if dim == getattr(mp, "Dcyl", object()):
            raise UnsupportedDrive(
                "the driver's field-function reducers are transcribed for Cartesian grids "
                "only; a Dcyl volume needs loop_in_chunks' ring measure")
        if dim == getattr(mp, "D1", object()):
            pairs = [(0.0, 0.0), (0.0, 0.0), (low.z(), high.z())]
        elif dim == getattr(mp, "D2", object()):
            pairs = [(low.x(), high.x()), (low.y(), high.y()), (0.0, 0.0)]
        else:
            pairs = [(low.x(), high.x()), (low.y(), high.y()), (low.z(), high.z())]
        return (tuple(0.5 * (a + b) for a, b in pairs), tuple(b - a for a, b in pairs))

    def _reducer_components(components):
        names = []
        for component in components:
            if _is_structure(component):
                names.append("Dielectric" if component == getattr(mp, "Dielectric", None)
                             else "Permeability")
                continue
            name = _component_name(mp, component)
            if name is None:
                raise UnsupportedDrive(
                    f"a field function asked for component {component!r}, which this driver "
                    f"does not publish")
            names.append(name)
        return names

    def _field_func(func):
        # MEEP's SWIG bridge calls the user function as func(Vector3(loc), *complex values)
        # (python/meep.i:158-188 py_field_func_wrap); the driver hands the location over as
        # a plain (x, y, z) tuple, so the Vector3 is built here.
        return lambda location, *values: func(mp.Vector3(*location), *values)

    def _require_driver(reader_name):
        if reducer_driver is None:
            raise UnsupportedDrive(
                f"sim.{reader_name} needs the stepped driver and this run produced none")
        return reducer_driver

    def integrate_field_function(cs, func, where=None, center=None, size=None):
        driver = _require_driver("integrate_field_function")
        if where is None and center is not None and size is not None:
            where = mp.Volume(center=mp.Vector3(*center), size=mp.Vector3(*size))
        box, extent = _volume_bounds(where)
        return driver.integrate_field_function(
            _reducer_components(cs), _field_func(func), center=box, size=extent)

    def max_abs_field_function(cs, func, where=None, center=None, size=None):
        driver = _require_driver("max_abs_field_function")
        if where is None and center is not None and size is not None:
            where = mp.Volume(center=mp.Vector3(*center), size=mp.Vector3(*size))
        box, extent = _volume_bounds(where)
        return driver.max_abs_field_function(
            _reducer_components(cs), _field_func(func), center=box, size=extent)

    def _box_reader(reader_name):
        def reader(box=None, center=None, size=None):
            driver = _require_driver(reader_name)
            if box is None and center is not None and size is not None:
                box = mp.Volume(center=mp.Vector3(*center), size=mp.Vector3(*size))
            where, extent = _volume_bounds(box)
            return getattr(driver, reader_name)(where, extent)
        return reader

    def flux_in_box(d, box=None, center=None, size=None):
        driver = _require_driver("flux_in_box")
        if box is None and center is not None and size is not None:
            box = mp.Volume(center=mp.Vector3(*center), size=mp.Vector3(*size))
        where, extent = _volume_bounds(box)
        axis = {getattr(mp, "X", None): 0, getattr(mp, "Y", None): 1,
                getattr(mp, "Z", None): 2}.get(d)
        if axis is None:
            raise UnsupportedDrive(
                f"flux_in_box asked for direction {d!r}, which is not one of this driver's "
                f"Cartesian axes")
        return driver.flux_in_box(axis, where, extent)

    _REDUCERS = {
        "integrate_field_function": integrate_field_function,
        "max_abs_field_function": max_abs_field_function,
        "flux_in_box": flux_in_box,
        "electric_energy_in_box": _box_reader("electric_energy_in_box"),
        "magnetic_energy_in_box": _box_reader("magnetic_energy_in_box"),
        "field_energy_in_box": _box_reader("field_energy_in_box"),
        "modal_volume_in_box": _box_reader("modal_volume_in_box"),
    }
    for reader_name, reader in _REDUCERS.items():
        if hasattr(sim, reader_name):
            setattr(sim, reader_name, reader)
    notes.append("integrate_field_function / max_abs_field_function / flux_in_box / "
                 "electric_ / magnetic_ / field_energy_in_box / modal_volume_in_box -> the "
                 "driver's transcription of src/integrate.cpp and src/energy_and_flux.cpp, "
                 "read off the stepped fields")

    if fields is not None:
        bound = []
        for reader_name, reader in _REDUCERS.items():
            try:
                setattr(fields, reader_name, reader)
                bound.append(reader_name)
            except BaseException:  # noqa: BLE001 — a SWIG proxy that refuses new attributes.
                pass
        # `fields.total_volume()` is LEFT ALONE, deliberately. It is `user_volume
        # .surroundings()` — pure geometry, no field read — so MEEP's answer is correct
        # whichever code did the stepping, exactly like `sim.get_epsilon`. Rebinding it to
        # return an `mp.Volume` broke two OTHER rows that were reaching further than this
        # one: `Simulation.output_component` and `get_array_metadata` hand it straight to
        # SWIG as a `meep::volume const &`, and a Python object there is a TypeError.
        # `_volume_bounds` reads the SWIG object instead.
        if bound:
            notes.append("sim.fields." + " / sim.fields.".join(sorted(bound))
                         + " -> the same driver reducers, so a test comparing the raw fields "
                           "object against the Simulation wrapper compares two stepped answers")
        else:
            notes.append("sim.fields reducers could not be rebound; the lower spelling still "
                         "reads MEEP's un-stepped grid")

    # `sim.output_field_function` evaluates the same integrand and writes it to HDF5
    # through fields.output_hdf5 — MEEP's own un-stepped arrays, and machinery this path
    # has no counterpart for. Dropped with a note, exactly as the HDF5 output STEP
    # functions are: test_field_functions calls it AFTER its two assertions, so letting it
    # raise would record the row as an error and lose both.
    if hasattr(sim, "output_field_function"):
        def output_field_function(name, cs, func, real_only=False, h5file=None):
            return None

        sim.output_field_function = output_field_function
        notes.append("output_field_function dropped (a no-op) — it writes MEEP's own "
                     "un-stepped arrays to HDF5, which this path has no counterpart for")

    def unsupported(reader_name):
        def raiser(*args, **kwargs):
            raise UnsupportedDrive(
                f"the test reads its result through sim.{reader_name}, which this driver has "
                f"no counterpart for; the row stays parity-only rather than being compared "
                f"against an un-stepped MEEP grid")
        return raiser

    # `integrate2_field_function` stays refused: it needs fields::integrate2
    # (src/integrate2.cpp) over TWO simulations' fields at once, which is the two-run
    # family's problem and not this one's — the driver has no second stepped grid to
    # integrate against.
    for reader in ("get_efield", "get_hfield", "get_dfield", "get_bfield",
                   "get_energy_in_box", "get_flux_in_box", "get_force",
                   "get_ldos", "integrate2_field_function"):
        if hasattr(sim, reader):
            setattr(sim, reader, unsupported(reader))
    # get_epsilon_point / get_epsilon_grid read the STRUCTURE, not the fields. MEEP's
    # own answer is correct there whichever code did the stepping, so those are left
    # alone deliberately rather than by omission.


def _flat_axis_weights(full_tics, coordinate, axis):
    """MEEP's read-time collapse of ONE zero-thickness axis: which cells, and how weighted.

    MEEP source. ``loop_in_chunks.cpp compute_boundary_weights`` (v1.33.0 :275-287), the branch
    ``where.in_direction_min(d) == where.in_direction_max(d)``, sets
    ``s0 = w0``, ``e0 = w1`` with ``w0 = 1 - min*a + is/2`` and
    ``w1 = 1 + max*a - ie/2``. With ``ie = is + 2`` — the two cells that bracket an
    off-grid coordinate — those are ``1 - f`` and ``f`` for ``f`` the coordinate's
    fractional position between the two cell centres, i.e. plain linear
    interpolation ONTO the requested coordinate. ``array_slice.cpp
    get_array_slice_chunkloop`` (:418) keeps exactly those weights (it rebuilds
    ``s0i/s1i/e0i/e1i`` with every NON-empty direction set to 1) and
    ``array_slice.cpp collapse_array`` (:554-587) then sums the axis away; ``dft.cpp
    process_dft_component`` does the same through ``retain_interp_weights``, which
    defaults to true (meep.hpp:2139).

    The order matters and is the one thing a symmetric request cannot check: the
    weight paired with the LOWER cell is ``1 - f``. MEEP's own slice tests all sit
    halfway between two cells, where the two weights are both 0.5 and the pairing is
    invisible; ``test_slice_like_meep.py`` carries asymmetric offsets for that reason.

    Returns:
        ``(indices, weights)`` — one or two whole-cell indices and their weights.
    """
    import numpy

    spacing = float(full_tics[1] - full_tics[0])
    if not spacing > 0.0:
        raise UnsupportedDrive(f"axis {axis} has non-increasing metadata tics")
    offset = (float(coordinate) - float(full_tics[0])) / spacing
    lower = int(numpy.floor(offset))
    if lower < 0 or lower + 1 >= full_tics.size:
        raise UnsupportedDrive(
            f"the requested axis-{axis} coordinate {coordinate!r} does not lie between two "
            f"cell centres of this cell ({full_tics[0]!r} .. {full_tics[-1]!r}); MEEP would "
            f"bracket it with cells this path cannot address")
    fraction = offset - lower
    return (numpy.asarray([lower, lower + 1], dtype=int),
            numpy.asarray([1.0 - fraction, fraction], dtype=float))


def slice_like_meep(mp, sim, whole, vol, center, size, collapse_empty_axes=True):
    """Cut MEEP's requested sub-volume out of a whole-cell array, by coordinate.

    Both index windows come from MEEP's own ``get_array_metadata`` — the whole cell's
    tics and the request's tics. An axis whose requested coordinates land on grid
    points is an exact index cut, with no arithmetic at all.

    An axis whose requested EXTENT IS ZERO is collapsed the way MEEP collapses it:
    the two bracketing cells are combined with the linear weights of
    :func:`_flat_axis_weights` and the axis disappears. That is not an approximation
    of MEEP's answer, it is MEEP's answer — ``sim.get_array`` on a flat volume never
    returns cell values, only this reduction of them — and it is what makes a test
    whose oracle is a flat slice driveable at all. ``control_slice_check`` compares
    the result against MEEP's own ``get_array`` on every case that uses it, in the
    control pass, so a wrong cut is caught as a slicing defect rather than blamed on
    the engine.

    Any OTHER off-grid axis is still refused. A request with a non-zero extent whose
    edges fall between cells is not interpolated by MEEP either (array_slice.cpp
    resets the weights of every non-empty direction to 1), so an off-grid coordinate
    on such an axis means the metadata is not the ladder this path assumes, and
    guessing there would put an unvalidated numerical step between the engine and the
    test's oracle.

    Args:
        collapse_empty_axes: False keeps the pre-collapse behaviour of refusing every
            off-grid coordinate. ``get_field_point`` passes False deliberately: MEEP
            answers that reader through ``fields::get_field``, which interpolates the
            component's own Yee lattice, not the centred array held here.
    """
    import numpy

    if vol is None:
        vol = mp.Volume(center=center if center is not None else mp.Vector3(),
                        size=size if size is not None else mp.Vector3())
    requested = [float(getattr(vol.size, name)) for name in ("x", "y", "z")]
    full = sim.get_array_metadata()
    part = sim.get_array_metadata(vol=vol)
    # (kind, payload) per axis of `whole`, in axis order; applied back to front so
    # that removing one axis does not renumber the ones still to be applied.
    plans = []
    for axis in range(3):
        full_tics = numpy.atleast_1d(numpy.asarray(full[axis], dtype=float))
        part_tics = numpy.atleast_1d(numpy.asarray(part[axis], dtype=float))
        if full_tics.size <= 1:
            continue
        indices, stray = [], None
        for value in part_tics:
            distance = numpy.abs(full_tics - value)
            best = int(numpy.argmin(distance))
            if distance[best] > 1e-6:
                stray = (value, full_tics[best])  # The value that failed, not the first one.
                break
            indices.append(best)
        if stray is None:
            indices = numpy.asarray(indices, dtype=int)
            if (indices.size > 1
                    and not numpy.array_equal(indices,
                                              numpy.arange(indices[0], indices[-1] + 1))):
                raise UnsupportedDrive(f"the requested slice is not contiguous on axis {axis}")
            plans.append(("take", indices))
            continue
        if not collapse_empty_axes or requested[axis] != 0.0 or part_tics.size != 1:
            raise UnsupportedDrive(
                f"the requested slice's axis-{axis} coordinate {stray[0]!r} is not a grid point "
                f"(nearest is {stray[1]!r}), and the axis is not a zero-extent one MEEP would "
                f"collapse ({part_tics.size} requested tics, extent {requested[axis]!r}); this "
                f"path does not reproduce that")
        plans.append(("collapse", _flat_axis_weights(full_tics, part_tics[0], axis)))
    if len(plans) != whole.ndim:
        raise UnsupportedDrive(
            f"metadata gave {len(plans)} sliceable axes for an array of rank {whole.ndim}")
    out = whole
    for axis in range(len(plans) - 1, -1, -1):
        kind, payload = plans[axis]
        if kind == "take":
            out = numpy.take(out, payload, axis=axis)
            if payload.size == 1:
                out = numpy.squeeze(out, axis=axis)  # MEEP drops an axis one cell wide.
            continue
        indices, weights = payload
        # MEEP forms the product in double and stores the sum in realnum
        # (array_slice.cpp:418, collapse_array:587); doing the whole reduction in double
        # here differs from that by at most an ULP and never the other way.
        promoted = numpy.result_type(out.dtype, numpy.float64)
        out = numpy.tensordot(numpy.take(out, indices, axis=axis).astype(promoted),
                              weights.astype(promoted), axes=([axis], [0]))
    return out


def control_slice_check(mp, sim) -> list[dict]:
    """Validate :func:`slice_like_meep` against MEEP's own slicing, in the control pass.

    The driven pass reads sub-volumes out of the driver's whole-cell array. If that
    cut were wrong, a driven assertion failure would look exactly like an engine
    defect. This wraps MEEP's real ``get_array`` during the CONTROL run: for every
    sub-volume the test asks for, it also fetches the whole cell, applies the same
    cut, and records the relative difference against MEEP's own answer. A row whose
    ``slice_check`` is at machine precision has its slicing exonerated; anything else
    disqualifies the row rather than indicting the engine.
    """
    import numpy

    checks: list[dict] = []
    original = sim.get_array

    def checked(component=None, vol=None, center=None, size=None, **kwargs):
        theirs = original(component=component, vol=vol, center=center, size=size, **kwargs)
        if vol is None and center is None and size is None:
            return theirs
        entry = {"component": str(component)}
        try:
            whole = numpy.asarray(original(component=component, cmplx=True))
            mine = slice_like_meep(mp, sim, whole, vol, center, size)
            reference = numpy.asarray(theirs)
            mine = mine if kwargs.get("cmplx") else mine.real
            if mine.shape != reference.shape:
                entry["result"] = f"SHAPE {mine.shape} vs {reference.shape}"
            else:
                denominator = float(numpy.linalg.norm(reference))
                entry["rel"] = (float(numpy.linalg.norm(mine - reference) / denominator)
                                if denominator else None)
                entry["result"] = "compared"
        except BaseException as exc:  # noqa: BLE001
            entry["result"] = f"{type(exc).__name__}: {exc}"[:200]
        checks.append(entry)
        return theirs

    sim.get_array = checked

    # There is deliberately NO get_field_point arm here. It used to compare MEEP's
    # get_field_point against a slice of MEEP's own centred get_array, because that
    # slice was the route the driven pass took. It is not any more: the driven pass
    # calls FdtdDriver.get_field_point, a transcription of monitor.cpp's raw-Yee
    # stencil, so a centred-array comparison would pin an assumption nothing makes and
    # would record a "not a grid point" refusal for every off-centre oracle that now
    # drives fine. get_array itself is still checked above, where it is still used.
    return checks


def _component_name(mp, component):
    table = {mp.Ex: "Ex", mp.Ey: "Ey", mp.Ez: "Ez", mp.Hx: "Hx", mp.Hy: "Hy", mp.Hz: "Hz"}
    try:
        return table.get(component)
    except BaseException:  # noqa: BLE001
        return None


# --- step functions: host, drop, or refuse ---------------------------------------------

# MEEP's own step-function wrappers, by the name of the closure each factory returns
# (``python/simulation.py``). Classification is by NAME because that is what MEEP
# gives us: every one of these is a nested ``def`` closing over its ``step_funcs``.
#
# HOSTED — pure timing and dispatch. `run_on_gpu` hands them a `_SimulationFacade`
# over the stepped driver, which answers `round_time()`, `meep_time()`,
# `fields.dt`, `fields.t`, `fields.last_source_time()`, `sources`, `run_index` and
# `get_field_point()` — the entire surface these six closures and `mp.Harminv` touch.
_HOSTED_STEP_FUNCTIONS = {
    "_beg": "mp.at_beginning (simulation.py:5088)",
    "_end": "mp.at_end (simulation.py:5102)",
    "_every": "mp.at_every (simulation.py:5119)",
    "_true": "mp.when_true / after_time / at_time / before_time (simulation.py:5028)",
    "_after_sources": "mp.after_sources (simulation.py:5045)",
    "_after_s_and_t": "mp.after_sources_and_time (simulation.py:5060)",
    "_combine": "mp.combine_step_funcs (simulation.py:5007)",
    "Harminv": "mp.Harminv (simulation.py:1149-1180)",
}

# REFUSED — each needs machinery the driver does not have, and every one of them
# would otherwise fail deep inside MEEP with an attribute error rather than as a
# recorded refusal.
_REFUSED_STEP_FUNCTIONS = {
    "_ldos": "mp.dft_ldos needs an mp.Ldos accumulator updated from sim.fields (simulation.py:5999)",
    "PadeDFT": "mp.PadeDFT accumulates inside MEEP's own stepper",
    "_sync": "mp.synchronized_magnetic needs fields.synchronize_magnetic_fields (simulation.py:5437)",
    "_with_prefix": "mp.with_prefix needs the HDF5 filename machinery (simulation.py:5468)",
    "_in_volume": "mp.in_volume/in_point rewrite sim.output_volume for the HDF5 writers (simulation.py:5185)",
    "_to_appended": "mp.to_appended needs the HDF5 writers (simulation.py:5222)",
    "_output_png": "mp.output_png needs the HDF5 writers and h5topng (simulation.py:5569)",
}

# Wrappers to DESCEND THROUGH when looking for what a step function actually does.
# `_at_time` never appears at top level — `at_time` returns `after_time(t, _at_time)` —
# but a leaf hunt that stops at it cannot see the writer inside. A REFUSED wrapper is
# deliberately NOT descended through: it has to survive as a leaf so the refusal fires
# on it, which is what keeps `mp.at_end(mp.in_point(pt, mp.output_efield_z))` a named
# refusal rather than a dropped writer.
_STEP_FUNCTION_WRAPPERS = (set(_HOSTED_STEP_FUNCTIONS) | {"_at_time"}) - {"Harminv"}

# A test module whose source inspects the filesystem cannot have its output writers
# dropped: the file the writer would have produced IS its oracle. Scanned over the
# whole MODULE rather than the one method, because MEEP's tests routinely put the
# `run()` in a helper and the `assertTrue(os.path.exists(...))` in the method (or the
# other way round), and over-refusing costs a row while under-refusing manufactures a
# failure that has nothing to do with the engine.
_FILESYSTEM_ORACLE_TOKENS = (
    "os.path.exists", "os.path.isfile", "os.listdir", "os.stat", "h5py", ".h5", "glob(",
)


def module_reads_files(module) -> bool:
    """True when a test module's own source inspects files; see the token table above.

    Unknown source counts as True. The conservative direction is the one that refuses
    a row, never the one that drives it against an oracle the driven pass did not
    produce.
    """
    import inspect

    try:
        return any(token in inspect.getsource(module) for token in _FILESYSTEM_ORACLE_TOKENS)
    except BaseException:  # noqa: BLE001
        return True


def _step_function_leaves(function, depth: int = 0) -> list:
    """Every non-wrapper callable a MEEP step function closes over, depth first.

    MEEP's wrappers are nested ``def``s that close over their ``step_funcs`` tuple, so
    the leaves are reachable through ``__closure__`` — ``mp.at_end(mp.output_efield_z)``
    is a ``_end`` whose one cell holds ``(output_efield_z,)``. Anything that is not a
    known wrapper is itself a leaf, which keeps the walk conservative: an unrecognised
    closure blocks a DROP rather than being searched through and dismissed.
    """
    name = getattr(function, "__name__", type(function).__name__)
    closure = getattr(function, "__closure__", None) or ()
    if depth > 8 or name not in _STEP_FUNCTION_WRAPPERS or not closure:
        return [function]
    leaves: list = []
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:  # An empty cell in a recursive closure.
            continue
        for item in (value if isinstance(value, (tuple, list)) else [value]):
            if callable(item):
                leaves.extend(_step_function_leaves(item, depth + 1))
    return leaves or [function]


def _is_meep_output_writer(mp, leaf) -> bool:
    """True for ``mp.output_epsilon`` / ``output_efield_z`` / any ``mp.output_*``.

    Those call ``Simulation.output_component`` (``simulation.py:3823-3853``), which
    writes MEEP's OWN structure and field arrays to an HDF5 file. In the driven pass
    MEEP's fields are initialized and never stepped, so letting one run would write a
    file of zeros — and hosting it is impossible anyway, since the facade has no HDF5
    writer. Whether such a writer may be dropped is :func:`module_reads_files`'
    decision, not this one's, and every drop is RECORDED per row so nobody has to take
    it on trust.
    """
    name = getattr(leaf, "__name__", "")
    if not name.startswith("output_"):
        return False
    return getattr(mp, name, None) is leaf


def triage_step_functions(mp, step_functions, reads_files: bool = True) -> tuple[tuple, list[str]]:
    """Split a ``run()``'s step functions into the ones to host and the ones to drop.

    Returns ``(hosted, dropped_descriptions)``; raises :class:`UnsupportedDrive`
    naming the first entry that is neither. Three-way rather than the blanket refusal
    this replaced, because the refusal was a HARNESS policy and not an engine limit:
    ``run_on_gpu`` hosts MEEP's step-function protocol (``FdtdDriver.run``'s
    ``step_functions``, MEEP's ``_run_until``) and ``_SimulationFacade`` answers the
    calls MEEP's wrappers make.

    A wrapper is DROPPED only when every leaf is one of MEEP's HDF5 output writers AND
    ``reads_files`` is False — i.e. nothing in the test module inspects the filesystem.
    That second condition is what separates ``test_pw_source``, whose
    ``mp.at_end(mp.output_efield_z)`` no assertion ever observes, from
    ``test_use_output_directory_default``, whose identical call IS the whole oracle:
    dropping there would manufacture a failure that says nothing about the stepper.

    A MIXED wrapper — a writer alongside something else, so the drop cannot be made
    wholesale — is refused rather than partially honoured. There is no way to remove a
    leaf from inside a MEEP closure, and running the writer against the facade would
    fail deep in ``Simulation.output_component`` instead of here.
    """
    hosted, dropped = [], []
    for function in step_functions:
        name = getattr(function, "__name__", type(function).__name__)
        if name in _REFUSED_STEP_FUNCTIONS:
            raise UnsupportedDrive(
                f"run() carries {name}: {_REFUSED_STEP_FUNCTIONS[name]}")
        if _is_meep_output_writer(mp, function) or name in _HOSTED_STEP_FUNCTIONS:
            leaves = _step_function_leaves(function)
            refused = [leaf for leaf in leaves
                       if getattr(leaf, "__name__", type(leaf).__name__)
                       in _REFUSED_STEP_FUNCTIONS]
            if refused:
                inner = getattr(refused[0], "__name__", type(refused[0]).__name__)
                raise UnsupportedDrive(
                    f"run()'s {name} wraps {inner}: {_REFUSED_STEP_FUNCTIONS[inner]}")
            writers = [leaf for leaf in leaves if _is_meep_output_writer(mp, leaf)]
            if writers and reads_files:
                raise UnsupportedDrive(
                    f"run()'s {name} writes "
                    f"{', '.join(sorted(getattr(w, '__name__', '?') for w in writers))} "
                    f"to HDF5 and this test module reads files back, so the written file is "
                    f"part of its oracle; the driver has no HDF5 writer and MEEP's own would "
                    f"write its un-stepped arrays")
            if writers and len(writers) < len(leaves):
                raise UnsupportedDrive(
                    f"run()'s {name} mixes the HDF5 writer(s) "
                    f"{', '.join(sorted(getattr(w, '__name__', '?') for w in writers))} with "
                    f"{len(leaves) - len(writers)} other step function(s); a leaf cannot be "
                    f"removed from inside a MEEP closure, so the whole wrapper is refused")
            if writers:
                dropped.append(
                    f"{name} wrapping "
                    f"{', '.join(sorted(getattr(w, '__name__', '?') for w in writers))}")
                continue
            hosted.append(function)
            continue
        # A test's OWN closure (test_cyl_ellipsoid's print_stuff, test_simulation's
        # `lambda sim: None`) is hosted too — it is ordinary Python, and whatever it
        # reaches for on the facade either exists or raises StepFunctionNotHosted by
        # name, which `driven_run` records as a refusal rather than an error.
        if callable(function):
            hosted.append(function)
            continue
        raise UnsupportedDrive(
            f"run() carries a non-callable step function {function!r}")
    return tuple(hosted), dropped


def install_driven_run(mp, state: dict):
    """Replace ``Simulation.run`` with the engine-driven equivalent. Returns restore()."""
    import meep_gpu

    original_run = mp.Simulation.run
    original_flux_data = {
        name: getattr(mp.Simulation, name)
        for name in ("get_flux_data", "load_flux_data", "load_minus_flux_data")
        if hasattr(mp.Simulation, name)
    }

    # THE TWO-RUN NORMALIZATION IDIOM. `sim.get_flux_data` / `load_minus_flux_data` are
    # intercepted for two independent reasons, and dropping either one loses the row:
    #
    #   1. MEEP's own methods reach `flux.E`, whose lazy `swigobj` property calls
    #      `init_sim()`. An initialized simulation is a documented refusal of the lift
    #      (a caller can rewrite fields.boundaries per face through a pointer no reader
    #      can see), so the load alone made every test using this idiom undriveable —
    #      16 rows of this corpus, and the ones carrying MEEP's published normalized
    #      transmission and reflection constants.
    #   2. The saved transform has to be OURS. MEEP's own `get_flux_data` on a
    #      simulation this engine stepped returns MEEP's untouched accumulator, which
    #      is all zeros — so the second run would subtract nothing and report the raw
    #      spectrum as a reflectance, with every assertion comparing a plausible wrong
    #      number against MEEP's constant.
    #
    # The load is recorded here and handed to `run_on_gpu(minus_flux_data=...)`, which
    # applies it to the migrated monitor after the lift and before the first step —
    # MEEP's own ordering (python/simulation.py:3629-3649).
    def keep(sim):
        # `state` keys simulations by id(), and MEEP's tests drop the first simulation
        # (`self.sim = None`) before building the second — so without a strong reference
        # the first can be collected and the second allocated at the same address. The
        # second run would then be refused as "run() twice on the SAME simulation", or
        # worse, hand it the first one's pending loads. One list of references costs
        # nothing and makes the ids unique for the case's lifetime.
        state.setdefault("sims", []).append(sim)

    def get_flux_data(sim, flux):
        result = state["results"].get(id(sim))
        if result is None:
            raise UnsupportedDrive(
                "the test reads sim.get_flux_data off a simulation this driver has not "
                "stepped; MEEP's own accumulator is empty there and the normalization run "
                "would subtract zeros")
        return result.get_flux_data(flux)

    def record_load(sign_is_minus):
        def loader(sim, flux, fdata):
            if not isinstance(fdata, _engine_flux_data_type()):
                raise UnsupportedDrive(
                    f"the test loads flux data of type {type(fdata).__name__}, which did not "
                    f"come from this engine's get_flux_data; loading MEEP's own chunk arrays "
                    f"into the driver's accumulators is not a conversion this path performs")
            keep(sim)
            key = "minus_flux_data" if sign_is_minus else "flux_data"
            state.setdefault("pending_loads", {}).setdefault(id(sim), {}).setdefault(
                key, []).append((flux, fdata))
            return None
        return loader

    if original_flux_data:
        mp.Simulation.get_flux_data = get_flux_data
        mp.Simulation.load_flux_data = record_load(False)
        mp.Simulation.load_minus_flux_data = record_load(True)
        # MEEP aliases the mode spellings onto the same functions; a test using them
        # must follow the same path rather than silently reaching MEEP's.
        for alias, target in (("get_mode_data", "get_flux_data"),
                              ("load_mode_data", "load_flux_data"),
                              ("load_minus_mode_data", "load_minus_flux_data")):
            if hasattr(mp.Simulation, alias):
                original_flux_data[alias] = getattr(mp.Simulation, alias)
                setattr(mp.Simulation, alias, getattr(mp.Simulation, target))

    def driven_run(sim, *step_functions, **kwargs):
        # Keyed by the SIMULATION, not by the process. MEEP's tests routinely build a
        # normalization run and a measurement run as two separate mp.Simulation objects
        # in one method; refusing the second because "a run already happened" would drop
        # every one of those on a limitation that does not apply to them.
        if id(sim) in state["results"]:
            raise UnsupportedDrive(
                "the test calls run() twice on the SAME simulation; run_on_gpu lifts afresh "
                "and cannot resume a partially stepped one")
        keep(sim)
        # Step functions are HOSTED where the facade can answer them, DROPPED where the
        # wrapper's only leaves are MEEP's HDF5 output writers, and refused by name
        # otherwise. See triage_step_functions.
        hosted, dropped = triage_step_functions(
            mp, step_functions, reads_files=bool(state.get("reads_files", True)))
        if "until" in kwargs:
            run_kwargs = {"until": kwargs["until"]}
        elif "until_after_sources" in kwargs:
            run_kwargs = {"until_after_sources": kwargs["until_after_sources"]}
        else:
            raise UnsupportedDrive(f"run() given neither until nor until_after_sources: {sorted(kwargs)}")
        (key, value), = run_kwargs.items()
        index = state["call_index"]
        state["call_index"] += 1
        if hosted:
            state.setdefault("notes", []).append(
                f"run() call {index}: hosted MEEP step function(s) "
                + ", ".join(sorted(getattr(f, "__name__", type(f).__name__) for f in hosted))
                + " on the driver through _SimulationFacade")
            state["step_functions_hosted"] = True
        if dropped:
            # RECORDED, never silent: these write MEEP's own un-stepped grid to HDF5 and
            # no assertion in the rows this unlocks reads the files back, but a reader of
            # the record has to be able to see that the test did not run as written.
            state.setdefault("notes", []).append(
                f"run() call {index}: dropped MEEP output step function(s) "
                + "; ".join(dropped)
                + " — they write MEEP's own (un-stepped) arrays to HDF5 through "
                  "Simulation.output_component, which this path has no counterpart for")
            state["step_functions_dropped"] = True
        if not isinstance(value, (int, float)):
            # mp.stop_when_fields_decayed and friends are MEEP step functions polled
            # inside MEEP's own loop; this path takes a numeric stopping time. Rather
            # than substitute a criterion of this engine's own — which would silently
            # change how long the run went, and with it the DFT the test asserts on —
            # the CONTROL pass is run first, UNMODIFIED, and the simulation time MEEP's
            # own criterion stopped at is reused verbatim here. Both codes then step the
            # same object for exactly the same time, and the control's assertions are
            # stock MEEP's, decided by stock MEEP's criterion.
            substitute = (state.get("stop_times") or {}).get(index)
            if substitute is None:
                raise UnsupportedDrive(
                    f"run({key}=...) was given a stopping CONDITION "
                    f"({getattr(value, '__name__', type(value).__name__)}) rather than a time, "
                    f"and no control run has recorded when MEEP's criterion stopped")
            run_kwargs = {"until": float(substitute)}
            key, value = "until", float(substitute)
            state.setdefault("notes", []).append(
                f"run() call {index}: MEEP's stopping condition replaced by the time the "
                f"control run's identical condition stopped at, t={float(substitute):.6f}")
            state["stopping_substituted"] = True
        state.setdefault("run_kwargs", []).append({key: float(value)})
        loads = (state.get("pending_loads") or {}).pop(id(sim), {})
        if loads:
            state.setdefault("notes", []).append(
                f"run() call {index}: "
                + ", ".join(f"{len(pairs)} {name}" for name, pairs in sorted(loads.items()))
                + " applied to the migrated flux monitor(s) before the first step")
        # The HDF5 read the package delegates, done here and recorded. See
        # `substitute_amp_func_file` for why it is the harness's and not the engine's.
        substitute_amp_func_file(mp, sim, state.setdefault("notes", []))
        started = time.time()
        try:
            result = meep_gpu.run_on_gpu(sim, prefer_gpu=False, step_functions=hosted,
                                         **loads, **run_kwargs)
        except meep_gpu.StepFunctionNotHosted as refusal:
            # A hosted step function reached past the facade — sim.set_materials,
            # sim.output_component, a derived component. That is a REFUSAL, recorded as
            # one, and not an error: nothing was measured wrongly, the row simply
            # cannot be driven. Raised from run_on_gpu after the driver was closed.
            raise UnsupportedDrive(
                f"a hosted step function reached past the simulation facade: {refusal}"
            ) from refusal
        state["results"][id(sim)] = result
        state["engine_steps"] = state.get("engine_steps", 0) + int(result.steps)
        state["engine_time"] = float(result.meep_time)
        state["engine_wall_s"] = round(state.get("engine_wall_s", 0.0) + time.time() - started, 2)
        bind_engine_readers(mp, sim, result, state.setdefault("notes", []),
                            loaders=original_flux_data)
        return None

    mp.Simulation.run = driven_run
    # THE MODULE-LEVEL SPECTRUM READERS. MEEP's tests reach these as free functions —
    # `mp.get_forces(self.myforce)`, `mp.get_electric_energy(energy)` — never through
    # the simulation, so binding the readers onto `sim` alone leaves them reading MEEP's
    # own un-stepped accumulator: a correctly shaped spectrum of ZEROS compared against
    # a published constant. Each override searches the driven results for one that knows
    # the object and falls through to MEEP's original otherwise, so a monitor on a
    # simulation this harness did not step still answers MEEP's way.
    module_readers = {
        "get_fluxes": "get_flux_spectrum",
        "get_forces": "get_forces",
        "get_electric_energy": "get_electric_energy",
        "get_magnetic_energy": "get_magnetic_energy",
        "get_total_energy": "get_total_energy",
    }
    original_module_readers = {
        name: getattr(mp, name) for name in module_readers if hasattr(mp, name)
    }

    def make_module_reader(name, engine_reader):
        original = original_module_readers[name]

        def reader(dft_object):
            import numpy
            for result in state["results"].values():
                try:
                    values = getattr(result, engine_reader)(dft_object)
                except KeyError:
                    continue
                return [float(numpy.real(value)) for value in values]
            return original(dft_object)
        return reader

    for name, engine_reader in module_readers.items():
        if name in original_module_readers:
            setattr(mp, name, make_module_reader(name, engine_reader))

    # THE ADJOINT GRADIENT, refused by name — because it is the one reader in this
    # corpus that would answer with a plausible number instead of an empty one.
    # `meep.adjoint` does not read its design region through any Python array API: it
    # hands the C++ `dft_fields` chunk lists of six `add_dft_fields(..., yee_grid=True,
    # persist=True)` monitors straight to `mp._get_gradient` (python/adjoint/utils.py:63-77,
    # :146-160), which walks `chunks` itself. This harness lifts a run's monitors into
    # ITS OWN accumulators and writes back only what it has a packer for, so those C++
    # chunks are still all zeros on a driven pass — and a gradient computed from zeros
    # is finite, correctly shaped and WRONG, so `assertClose` fails on a wrong number
    # rather than the row being recorded as not driven. That is the degenerate-returns
    # class this survey exists to surface, so it is refused explicitly.
    #
    # The guard lifts the moment there is a `pack_dft_fields_data` counterpart of
    # `pack_flux_data` writing through `mp._load_dft_data(monitor.swigobj.chunks, ...)`
    # (python/meep.i:496-513) into both the forward and adjoint monitor sets. Until
    # then it is a refusal, not a limitation of the STEPPING: the five
    # `test_adjoint_solver` rows it covers all measure field parity 2.179e-06 with
    # `time_matched` true (results/meep_python_tests_2026-08-07/parity.jsonl), so the
    # fields are right and it is the reader that has no route.
    original_get_gradient = getattr(mp, "_get_gradient", None)

    def refuse_gradient(*args, **kwargs):
        if not state.get("results"):
            return original_get_gradient(*args, **kwargs)
        raise UnsupportedDrive(
            "the test reads a gradient through mp._get_gradient, which walks the C++ "
            "chunk lists of meep.adjoint's yee_grid design-region dft_fields monitors "
            "directly (python/adjoint/utils.py:63-77); this harness has no packer for "
            "those, so they hold zeros on a driven run and the gradient would be a "
            "finite WRONG number rather than a refusal")

    if original_get_gradient is not None:
        mp._get_gradient = refuse_gradient

    def restore():
        mp.Simulation.run = original_run
        for name, function in original_module_readers.items():
            setattr(mp, name, function)
        for name, method in original_flux_data.items():
            setattr(mp.Simulation, name, method)
        if original_get_gradient is not None:
            mp._get_gradient = original_get_gradient

    return restore


def substitute_amp_func_file(mp, sim, notes: list[str]) -> None:
    """Read an ``amp_func_file`` source's HDF5 dataset here and hand it over as ``amp_data``.

    THE ONLY SUBSTITUTION THIS HARNESS MAKES TO A TEST'S DECLARED INPUT, and it is an
    I/O one rather than a numerical one. MEEP's own two routes are the same route:
    ``fields::add_volume_source(..., filename, dataset, amp)`` (src/sources.cpp:421-460)
    reads ``<dataset>.re`` and ``<dataset>.im`` into one complex array and calls the
    ARRAY overload, which copies it into two static buffers and calls the ordinary
    amp_func overload with ``amp_file_func``. ``meep_gpu`` lifts the array spelling
    (``amp_interpolation``, MEEP's interpolator transcribed) and refuses the file
    spelling, because the package carries no HDF5 dependency at all
    (``test_package_boundary`` enforces it). So the read is done here and nothing else
    changes: same array, same interpolator, same deposition.

    MEASURED on ``test_source.py``'s own fixture (pristine MEEP 1.33.0, single
    precision, Ez at the origin of a 1x1 cell at resolution 60, t=200): the file route
    reads 7.16554350219667e-05 and the array route 7.165555143728852e-05 — 1.164e-10
    apart, which is the fixture's float32 storage rather than the arithmetic.

    RECORDED per row, always, because a reader of the record has to be able to see
    that the driven pass did not open the file the control pass opened. Silent would
    be indefensible; visible is a caveat with a number attached.
    """
    import numpy

    for index, source in enumerate(getattr(sim, "sources", ()) or ()):
        spec = getattr(source, "amp_func_file", "")
        if not spec:
            continue
        if type(source).__name__ != "Source":
            raise UnsupportedDrive(
                f"sources[{index}] is a {type(source).__name__} carrying amp_func_file; only a "
                f"plain mp.Source's file profile is read here")
        try:
            import h5py
        except ImportError as exc:  # noqa: PERF203 — one source, one message.
            raise UnsupportedDrive(
                f"sources[{index}] sets amp_func_file and h5py is not installed in this "
                f"environment, so the array meep_gpu lifts cannot be read") from exc
        # python/source.py:146-155, verbatim: split on the LAST colon, append '.h5' when
        # the filename does not already carry it.
        filename, _, dataset = spec.rpartition(":")
        if not filename or not dataset:
            raise UnsupportedDrive(
                f"sources[{index}] sets amp_func_file={spec!r}, which is not 'file.h5:dataset'")
        if not filename.endswith(".h5"):
            filename += ".h5"
        with h5py.File(filename, "r") as handle:
            real = numpy.asarray(handle[f"{dataset}.re"][()], dtype=numpy.float64)
            imaginary = numpy.asarray(handle[f"{dataset}.im"][()], dtype=numpy.float64)
        if real.shape != imaginary.shape:
            raise UnsupportedDrive(
                f"sources[{index}]: '{dataset}.re' is {real.shape} and '{dataset}.im' is "
                f"{imaginary.shape}; MEEP aborts on that too (sources.cpp:445-449)")
        source.amp_func_file = ""
        source.amp_data = real + 1j * imaginary
        notes.append(
            f"sources[{index}]: amp_func_file {spec!r} read here (h5py) and handed to the lift "
            f"as amp_data{tuple(real.shape)} — MEEP's own file route does the same read and "
            f"then takes the array path (sources.cpp:421-460); the package carries no HDF5 "
            f"dependency, so this read is the harness's")


def _engine_flux_data_type():
    """The opaque container this engine's ``get_flux_data`` returns."""
    from meep_gpu.dft import FluxDftData

    return FluxDftData


# --- one case, both passes --------------------------------------------------------------


def install_control_hook(mp, state: dict):
    """Control pass: MEEP's real ``run``, observed but not altered.

    Two things are recorded and nothing is changed: the slicing self-check is bound to
    the simulation once it has been stepped, and the simulation time each ``run()``
    call finished at is kept, so a driven pass can reuse the stopping time MEEP's own
    criterion chose instead of inventing one.
    """
    original_run = mp.Simulation.run

    def hooked_run(sim, *step_functions, **kwargs):
        index = state["call_index"]
        state["call_index"] += 1
        outcome = original_run(sim, *step_functions, **kwargs)
        try:
            state.setdefault("stop_times", {})[index] = float(sim.round_time())
            state.setdefault("stop_steps", {})[index] = int(sim.fields.t)
        except BaseException:  # noqa: BLE001
            pass
        if not state.get("slice_checks_installed"):
            state["slice_checks"] = control_slice_check(mp, sim)
            state["slice_checks_installed"] = True
        return outcome

    mp.Simulation.run = hooked_run

    def restore():
        mp.Simulation.run = original_run

    return restore


def _execute(module, class_name, method_name, timeout_s, namespace, driven: bool,
             stop_times: dict | None = None) -> dict:
    """Run one test method to completion, instrumented. Never raises."""
    import meep as mp

    out: dict = {"driven": driven}
    state: dict = {"results": {}, "notes": [], "call_index": 0, "stop_times": stop_times or {},
                   # Whether an HDF5 output step function may be dropped for this module;
                   # see triage_step_functions.
                   "reads_files": module_reads_files(module)}
    restore = install_driven_run(mp, state) if driven else install_control_hook(mp, state)
    started = time.time()
    namespace["_alarm"](timeout_s)
    try:
        cls = getattr(module, class_name)
        cls.setUpClass()
        case = cls(method_name)
        records = instrument_assertions(case)
        out["assertions"] = records
        case.setUp()
        getattr(case, method_name)()
        out["outcome"] = "ran_to_end"
    except UnsupportedDrive as exc:
        out["outcome"] = "not_driveable"
        out["reason"] = str(exc)[:400]
    except BaseException as exc:  # noqa: BLE001
        out["outcome"] = "error"
        out["reason"] = f"{type(exc).__name__}: {exc}"[:400]
    finally:
        namespace["_cancel_alarm"]()
        restore()
        for result in state.get("results", {}).values():
            try:
                result.close()
            except BaseException:  # noqa: BLE001
                pass
    out.setdefault("assertions", [])
    out["wall_s"] = round(time.time() - started, 2)
    out["engine_steps"] = state.get("engine_steps")
    out["engine_time"] = state.get("engine_time")
    out["run_kwargs"] = state.get("run_kwargs")
    out["reader_notes"] = state.get("notes")
    out["slice_checks"] = state.get("slice_checks")
    out["stop_times"] = state.get("stop_times")
    out["stop_steps"] = state.get("stop_steps")
    out["stopping_substituted"] = bool(state.get("stopping_substituted"))
    out["n_passed"] = sum(1 for a in out["assertions"] if a.get("passed") is True)
    out["n_failed"] = sum(1 for a in out["assertions"] if a.get("passed") is False)
    out["n_errored"] = sum(1 for a in out["assertions"] if a.get("passed") is None)
    out["n_total"] = len(out["assertions"])
    return out


def child_assertions(namespace, module, module_path, case, args, progress) -> dict:
    """Driven pass then control pass for one test method; the row compares them."""
    import os

    class_name, method_name = case
    record = {
        "module": os.path.basename(module_path),
        "case": f"{class_name}.{method_name}",
        "stage": "assertions",
        "python": __import__("sys").executable,
    }
    timeout_s = float(getattr(args, "assert_timeout", 300.0))

    # DRIVEN FIRST, deliberately. A test whose run() carries step functions is refused
    # before a single time step, so putting the driven pass first costs nothing and
    # saves a full-resolution control run of MEEP's own test for every case that was
    # never driveable — the majority of this corpus.
    progress(f"{record['case']}: driven pass (this engine)")
    driven = _execute(module, class_name, method_name, timeout_s, namespace, driven=True)
    record["driven"] = driven
    progress(f"{record['case']}: driven {driven['n_passed']}/{driven['n_total']} "
             f"({driven['outcome']}, {driven['wall_s']} s)")

    if _blocked_only_by_stopping_condition(driven):
        # The one refusal worth spending a control run on: MEEP's own criterion decides
        # the stopping time, so run the control UNMODIFIED to learn it, then drive again
        # with that exact time. Both codes then step for identically long.
        progress(f"{record['case']}: blocked by a stopping condition — control pass first")
        control = _execute(module, class_name, method_name, timeout_s, namespace, driven=False)
        record["control"] = control
        progress(f"{record['case']}: control {control['n_passed']}/{control['n_total']} "
                 f"({control['outcome']}, {control['wall_s']} s), stop times "
                 f"{control.get('stop_times')}")
        if control["outcome"] == "ran_to_end" and control.get("stop_times"):
            driven = _execute(module, class_name, method_name, timeout_s, namespace,
                              driven=True, stop_times=control["stop_times"])
            record["driven"] = driven
            record["stopping_substituted"] = True
            progress(f"{record['case']}: driven (stopping time reused) "
                     f"{driven['n_passed']}/{driven['n_total']} ({driven['outcome']})")
            return _finish(record, control, driven)
        record["assertions_driven"] = False
        record["not_driven_reason"] = driven.get("reason") or driven["outcome"]
        record["assertions_total"] = 0
        record["assertions_passed"] = 0
        record["control_total"] = control["n_total"]
        record["control_passed"] = control["n_passed"]
        record["assert_verdict"] = "not driveable"
        return record

    if driven["outcome"] != "ran_to_end" or driven["n_total"] == 0:
        # No comparison is possible, so the control run would measure only whether
        # MEEP passes its own test — which is not what this stage is for, and on this
        # corpus costs minutes per case at the test's own resolution.
        record["assertions_driven"] = False
        record["not_driven_reason"] = driven.get("reason") or driven["outcome"]
        record["engine_steps_before_refusal"] = driven.get("engine_steps")
        record["control"] = {"skipped": "the driven pass did not reach the test's assertions, so "
                                        "there is nothing for a control run to be compared with"}
        record["assertions_total"] = 0
        record["assertions_passed"] = 0
        record["control_total"] = None
        record["control_passed"] = None
        record["assert_verdict"] = "not driveable"
        return record

    progress(f"{record['case']}: control pass (stock MEEP)")
    control = _execute(module, class_name, method_name, timeout_s, namespace, driven=False)
    record["control"] = control
    progress(f"{record['case']}: control {control['n_passed']}/{control['n_total']} "
             f"({control['outcome']}, {control['wall_s']} s)")
    return _finish(record, control, driven)


def _blocked_only_by_stopping_condition(driven: dict) -> bool:
    return (driven.get("outcome") == "not_driveable"
            and "stopping CONDITION" in (driven.get("reason") or ""))


def _finish(record: dict, control: dict, driven: dict) -> dict:
    record["assertions_driven"] = driven["outcome"] == "ran_to_end" and driven["n_total"] > 0
    record["assertions_total"] = driven["n_total"]
    record["assertions_passed"] = driven["n_passed"]
    record["control_total"] = control["n_total"]
    record["control_passed"] = control["n_passed"]
    if not record["assertions_driven"]:
        record["not_driven_reason"] = driven.get("reason") or driven["outcome"]
    # The claim is only about assertions that hold for stock MEEP here: an oracle the
    # reference build itself misses (single precision, a platform difference) is not
    # evidence about this engine either way.
    paired = list(zip(control["assertions"], driven["assertions"]))
    record["aligned"] = len(control["assertions"]) == len(driven["assertions"])
    if record["aligned"]:
        record["control_pass_engine_pass"] = sum(
            1 for c, d in paired if c.get("passed") is True and d.get("passed") is True)
        record["control_pass_engine_fail"] = sum(
            1 for c, d in paired if c.get("passed") is True and d.get("passed") is not True)
        record["failures"] = [
            {"assertion": d["assertion"], "args": d.get("args"), "kwargs": d.get("kwargs"),
             "detail": d.get("detail"), "control_args": c.get("args")}
            for c, d in paired if c.get("passed") is True and d.get("passed") is not True
        ][:12]
    verdict = (f"{record.get('control_pass_engine_pass', '?')}/"
               f"{record.get('control_passed')} MEEP-passing assertions reproduced")
    record["assert_verdict"] = verdict
    return record


_ = (json, math)  # kept for the JSONL contract's numeric helpers
