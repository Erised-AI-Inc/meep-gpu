"""Can a fused pair carry a source deposit across the PLAIN constitutive branch?

THE QUESTION, AND WHY IT IS A SECOND QUESTION. ``probe_deposit_repair.py`` established
that a fused curl/constitutive launch may consume a pre-injection field and be REPAIRED
at the deposit points, and it did so for the SPLIT-FIELD recurrence -- the one an active
absorber runs. That repair is stateful: ``_apply_constitutive_pml`` (``stepping.py
:2083-2087``) reads ``fw`` before overwriting it and then accumulates into the field, so
the repair captures both arrays before the launch and recomputes them afterwards.

``deposit_repair`` refused the PLAIN branch by name, and was right to: the arithmetic
there is not that arithmetic. With ``stepping._pml_is_active`` False, ``update_E`` writes

    E[...] = (D - sum P) * inv_eps                              (stepping.py:1019-1022)

straight to storage -- a PURE OVERWRITE with no ``f_w``, no ``kps``/``kms`` and no read
of the previous value. This probe is the measurement that licensed adding
``deposit_repair.PLAIN_PATH``, the second repair, which inverts THAT.

WHY IT IS EXACT, and every clause is a property of those lines rather than an
approximation of them:

  1. NOTHING TO SAVE. The branch neither reads ``field`` nor accumulates into it, so the
     repair's ``save`` captures the CELL SET and an empty state beside it.
  2. NO ``f_w``. It is not allocated on this path, and ``Fields.drive_field`` hands the
     polarizations the STORED E instead (``fields.py:1160-1162``) -- so ``update_P``,
     which runs after ``update_E`` at ``driver.py:3314``, reads exactly the array the
     repair rewrote. Nothing else consumes the constitutive product.
  3. EVERY OPERAND IS AT THE CELL'S OWN INDEX. ``displacement_minus_polarization`` reads
     ``D_c`` and each ``P_n[c]``; ``inverse_epsilon_for`` is a per-cell volume. So the
     repair GATHERS both operands at the deposit cells and then multiplies
     (``deposit_repair._apply_plain``: ``volume[cells] * inverse[cells]`` through
     ``_constitutive_at``, or the same product through the plan's cached linear index),
     and per cell that is the same correctly rounded float product of the same two
     operands the array path writes there -- no neighbour is read. ``_constitutive_at``
     keeps the whole-array product only when the two operands do not share a shape. Until
     2026-09-19 this clause described the whole-array form, which the repair then used.

WHO THIS BUYS. Three seam-instances the Triton board records as buildable gaps with no
product: ``(conductive no-PML, no-PML stored E)`` at ceiling 2 and ``(no-PML, no-PML
stored E)`` at ceiling 1 -- ``examples:absorber-1d.py``,
``tests:TestAbsorber.test_absorber``, ``examples:material-dispersion.py``. All three
declare an ELECTRIC source in this seam, which is what made the repair the whole
product rather than a widening. ``triton_kernels/no_pml_fused_electric_pair.py`` is the
product; this probe is about the REPAIR it brackets its launch with.

THE LEGS.

  A  THE REFERENCE, the driver's own order (``driver.py:3299-3314``)::

        step_D ; inject ; fill_symmetry_bc_D ; zero_metal_D ;
        fill_folded_far_ghosts_D ; update_E ; update_P

  B  FUSED PLUS REPAIR. The product's launch runs at the ``step_D`` consult and does
     both halves against an UNINJECTED D, carrying the wall clear inline the way the
     shipped kernel does. The driver then injects and runs its own passes, and the
     ``update_E`` consult -- which the fused arm also owns -- spends itself on the
     repair::

        save the deposit cells ; step_D ; zero_metal_D ; update_E ;
        inject ; fill_symmetry_bc_D ; zero_metal_D ; fill_folded_far_ghosts_D ;
        repair(deposit cells) ; update_P

     THIS DRIVES THE SHIPPED OBJECTS: ``deposit_repair.LeadingRepairPlan`` and
     ``TrailingRepairPlan``, with ``repair_paths=(PLAIN_PATH,)`` -- the exact pair
     ``launch._install_fused_pair`` puts in a fused pair's two slots.

  NULL  Leg B with the repair withheld. It MUST diverge on every case. A case where
     both legs agree cannot tell a working repair from a missing one, and its pass is
     worth nothing.

  FOLD CONTROL  The same two legs on a MIRRORED grid, with NO SOURCE AT ALL. It is
     expected to DIVERGE, and that divergence is what says the fold refusal in
     ``no_pml_fused_electric_pair_coverage`` is about the two post-injection FILLS and
     not about the deposit: the fills rebuild rows the fused launch never wrote, which
     is a different problem from carrying a deposit and one no repair solves.

  DEVICE (``--backend triton``)  Leg B builds the real product through
     ``triton_kernels.no_pml_fused_electric_pair.plan_no_pml_fused_electric_pair`` and
     launches it, in place of the array emulation. Everything else is unchanged.
     THE PREDICATE IS NOT BYPASSED HERE, unlike in ``probe_deposit_repair``: this
     product's seam clause ADMITS an in-seam electric source (that is what
     ``CARRIES_DEPOSIT_REPAIR`` means), so the plan is obtained with the real source
     list and a refusal is a real result.

WHAT A PASS DOES NOT LICENSE. Nothing about a folded grid (refused, and measured to be
rightly refused), nothing about an active absorber (that is the other repair), nothing
about the magnetic seam (``update_H`` writes nothing on this branch, so
``PLAIN_PATH_SEAMS`` names ``D`` alone) and nothing about throughput.

Progress: one flushed line per case, and the artifact is written whether the verdict
releases or not (the progress-reporting rule).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = HERE.parent.parent
for entry in (str(API), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy  # noqa: E402

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.dispersion import PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.sources import GaussianPulsedSource  # noqa: E402

PLAIN = (deposit_repair.PLAIN_PATH,)

#: Every stored volume the comparison reads, plus the polarization buffers. THE POLE
#: STATE IS PART OF IT: without it a repair that got E right and corrupted P would read
#: as byte-identical, and P is driven by exactly the array this repair rewrites.
PRIMARY = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Bx", "By", "Bz",
           "f_w_Ex", "f_w_Ey", "f_w_Ez", "fu_Dx", "fu_Dy", "fu_Dz",
           "fu_Bx", "fu_By", "fu_Bz", "f_cond_Dx", "f_cond_Dy", "f_cond_Dz")


def _words(array) -> numpy.ndarray:
    """Raw bits, so a comparison is bytes and not a tolerance."""
    if hasattr(array, "get"):          # a CuPy array: compare on the host
        array = array.get()
    flat = numpy.ascontiguousarray(array).ravel()
    if numpy.iscomplexobj(flat):
        flat = flat.view(numpy.float32 if flat.dtype == numpy.complex64 else numpy.float64)
    if flat.dtype == numpy.float32:
        return flat.view(numpy.uint32)
    if flat.dtype == numpy.float64:
        return flat.view(numpy.uint64)
    return flat


def _snapshot(fields) -> dict:
    out = {}
    for name in PRIMARY:
        array = getattr(fields, name, None)
        if array is None:
            continue
        out[name] = numpy.array(array.get() if hasattr(array, "get") else array,
                                copy=True)
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for attribute in ("P", "P_prev"):
            table = getattr(state, attribute, None)
            if not isinstance(table, dict):
                continue
            for component, array in sorted(table.items()):
                if array is None:
                    continue
                out[f"pol{index}.{attribute}.{component}"] = numpy.array(
                    array.get() if hasattr(array, "get") else array, copy=True)
    return out


def _differing(left: dict, right: dict) -> dict:
    out = {}
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        if a is None or b is None:
            out[name] = "present on one side only"
            continue
        wa, wb = _words(a), _words(b)
        if wa.shape != wb.shape:
            out[name] = f"shape {wa.shape} vs {wb.shape}"
            continue
        n = int((wa != wb).sum())
        if n:
            out[name] = n
    return out


def _array_module(backend: str):
    """The array module the fields live on.

    A Triton plan binds DEVICE pointers, so a NumPy-backed ``Fields`` yields no plan at
    all -- which the first device run of the sibling probe reported as "dispatched none"
    rather than passing on the emulation's result.
    """
    if backend != "triton":
        return None
    import cupy  # noqa: PLC0415
    return cupy


def _build(cell=(1.2, 1.2, 0.0), resolution=12.0, poles=2, conductivity=None,
           boundaries="metallic", symmetry=(), seed=20260831, backend="numpy"):
    """One PLAIN configuration on real engine objects: ``(grid, fields, pml)``.

    ``PML(thickness=0)`` is a layer that absorbs on no face, so
    ``stepping._pml_is_active`` is False and both constitutive halves take the plain
    branch. ``enable_field_storage`` allocates E WITHOUT ``f_w``, which is the storage
    the plain branch writes into and the absence the split-field repair refuses on.
    """
    dimensions = sum(1 for extent in cell if extent > 0.0)
    xp = _array_module(backend)
    # PER AXIS, because an axis the run does not resolve carries no boundary condition
    # at all: `Grid` refuses `metallic` on a translationally invariant axis by name
    # (grid.py:863) -- a PEC there is a polarization filter, not a wall. So the wall
    # goes on the resolved axes and the invariant one stays periodic, which is what a
    # one-pixel direction is in MEEP too (fields.cpp nosize_direction).
    per_axis = {name: (boundaries if extent > 0.0 else "periodic")
                for name, extent in zip("xyz", cell)}
    kwargs = dict(resolution=resolution, cell_size=cell, dimensions=dimensions,
                  boundaries=per_axis, symmetry=symmetry)
    grid = Grid(xp=xp, **kwargs) if xp is not None else Grid(**kwargs)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    for index in range(poles):
        # A DIFFERENT sigma per component, so the three see different pole SUBSETS in
        # `poles_per_component` order -- the property the product's NP0/NP1/NP2
        # constexprs specialise on and a repair that used one chain for all three
        # would get wrong.
        sigma = {"Ex": 0.30 + 0.05 * index,
                 "Ey": 0.0 if index else 0.20,
                 "Ez": 0.25 + 0.05 * index}
        fields.polarizations.append(PolarizationState(
            Susceptibility(1.0 + 0.3 * index, 0.1, "lorentzian"), sigma, grid,
            numpy.float32))
    if conductivity is not None:
        # A VOLUME, not a scalar: `set_d_conductivity` takes grid.shape, and a
        # spatially varying sigma is what an `mp.Absorber` actually installs.
        volume = numpy.full(tuple(grid.shape), float(conductivity), dtype=numpy.float32)
        volume *= numpy.linspace(0.5, 1.5, grid.shape[0], dtype=numpy.float32)[:, None, None]
        fields.set_d_conductivity(xp.asarray(volume) if xp is not None else volume)
    fields.enable_field_storage()
    pml = PML(grid=grid, thickness=0)
    generator = numpy.random.default_rng(seed)
    # BEFORE anything is stepped, and NOT zeros: zero-init is a fixed point of both the
    # constitutive relation and the ADE, so a zeroed case passes whether or not the
    # repair does anything.
    for name in PRIMARY:
        array = getattr(fields, name, None)
        if array is None:
            continue
        filled = generator.standard_normal(array.shape).astype(numpy.float32)
        array[...] = xp.asarray(filled) if xp is not None else filled
    for state in fields.polarizations:
        for attribute in ("P", "P_prev"):
            for array in (getattr(state, attribute, None) or {}).values():
                if array is None:
                    continue
                filled = generator.standard_normal(array.shape).astype(numpy.float32)
                array[...] = xp.asarray(filled) if xp is not None else filled
    return grid, fields, pml


def _make_source(grid, component="Ez", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)):
    return GaussianPulsedSource(grid=grid, component=component, center=center,
                                size=size, frequency=1.0, fwidth=0.2, amplitude=1.0)


def _leg_a(fields, pml, sources, when, dt):
    """The driver's own order, verbatim."""
    stepping.step_D(fields, pml)
    for source in sources:
        source.inject(fields, when + 0.5 * dt)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)
    stepping.update_P(fields, pml)


class _HostPair:
    """The fused product's three call sites, for a host leg with no Triton.

    Stands in for the kernel at the FIRST consult only. The repair under test is the
    shipped component either way, so what this changes is whether the arithmetic it
    repairs came from one launch or from three array calls -- which is exactly the
    difference the device leg exists to close.
    """

    def __init__(self, fields, pml):
        self._fields, self._pml = fields, pml
        self.launches = 0

    def run(self, *_args, **_kwargs):
        stepping.step_D(self._fields, self._pml)
        stepping.zero_metal_D(self._fields)   # the kernel's inline ZM_*
        stepping.update_E(self._fields, self._pml)
        self.launches += 1


def _device_pair(fields, pml, sources):
    """The real product, or ``None`` where it cannot be built on this host.

    THE PREDICATE IS ASKED WITH THE REAL SOURCES. This family declares
    ``CARRIES_DEPOSIT_REPAIR``, so an in-seam electric deposit is something it admits
    rather than something a probe has to step around -- and a refusal here is a real
    result about the configuration, not a harness limitation.
    """
    try:
        from meep_gpu.triton_kernels import no_pml_fused_electric_pair as product
    except Exception:      # noqa: BLE001 - no Triton on this host
        return None
    try:
        return product.plan_no_pml_fused_electric_pair(fields, pml, sources)
    except Exception:      # noqa: BLE001
        return None


def _leg_b(fields, pml, sources, when, dt, repair=True, plan=None):
    """Fused pair at the ``step_D`` consult, repair at the ``update_E`` consult."""
    inner = plan if plan is not None else _HostPair(fields, pml)

    if not repair:
        # THE NULL CONTROL. The fused halves run, the driver's own passes run, and
        # nothing is saved or restored. If this leg also matches the reference then the
        # case cannot tell a working repair from a missing one, and its PASS is worth
        # nothing.
        inner.run()
        for source in sources:
            source.inject(fields, when + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        stepping.update_P(fields, pml)
        return 0

    leading = deposit_repair.LeadingRepairPlan(inner, fields, pml, sources, "D", PLAIN)
    trailing = deposit_repair.TrailingRepairPlan("update_E", leading, fields, pml)

    leading.run()          # saves the deposit cells, then ONE launch: curl + ZM + E

    # --- the driver's own unconditional passes, untouched ---
    for source in sources:
        source.inject(fields, when + 0.5 * dt)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)

    trailing.run()         # the repair, at the deposit cells only
    stepping.update_P(fields, pml)
    return leading.repairs


def run_case(name: str, *, steps: int, component: str, center, size,
             backend: str, **config) -> dict:
    record = {"case": name, "steps": steps, "component": component,
              "center": list(center), "size": list(size), "backend": backend,
              "config": {key: list(value) if isinstance(value, tuple) else value
                         for key, value in config.items()}}
    started = time.time()
    try:
        grid_a, fields_a, pml_a = _build(backend=backend, **config)
        grid_b, fields_b, pml_b = _build(backend=backend, **config)
        grid_c, fields_c, pml_c = _build(backend=backend, **config)
        record["plain_branch"] = not stepping._pml_is_active(pml_a)
        record["stores_E"] = bool(fields_a.stores_E)
        record["f_w_allocated"] = getattr(fields_a, "f_w_Ex", None) is not None
        pre = _differing(_snapshot(fields_a), _snapshot(fields_b))
        record["precondition_identical"] = not pre
        if pre:
            record["precondition_differs"] = pre
            return record
        if not record["plain_branch"]:
            record["note"] = "this build is not on the plain branch"
            return record

        source_a = [_make_source(grid_a, component, center, size)]
        source_b = [_make_source(grid_b, component, center, size)]
        source_c = [_make_source(grid_c, component, center, size)]
        index = getattr(source_a[0], "_point_ix", None)
        record["deposit_points"] = 0 if index is None else int(
            numpy.atleast_1d(index.get() if hasattr(index, "get") else index).size)
        record["grid_cells"] = int(numpy.prod(tuple(grid_a.shape)))
        if not record["deposit_points"]:
            record["note"] = "no deposit points: this case would not discriminate"
            return record

        record["repairable"] = list(deposit_repair.repairable(fields_b, "D", pml_b,
                                                              paths=PLAIN))
        plan = _device_pair(fields_b, pml_b, source_b) if backend == "triton" else None
        record["real_kernel_dispatched"] = plan is not None
        if backend == "triton" and plan is None:
            record["note"] = ("the fused pair could not be built on this host; a "
                              "device leg that silently fell back to the array "
                              "emulation would report the architecture's result as "
                              "the kernel's")
            return record
        null_plan = (_device_pair(fields_c, pml_c, source_c)
                     if backend == "triton" else None)

        dt = grid_a.dt
        repaired = 0
        for step in range(steps):
            when = step * dt
            _leg_a(fields_a, pml_a, source_a, when, dt)
            repaired += _leg_b(fields_b, pml_b, source_b, when, dt, plan=plan)
            _leg_b(fields_c, pml_c, source_c, when, dt, repair=False, plan=null_plan)
        record["repaired_cells"] = repaired
        record["differing"] = _differing(_snapshot(fields_a), _snapshot(fields_b))
        record["identical"] = not record["differing"]
        record["null_differing"] = _differing(_snapshot(fields_a), _snapshot(fields_c))
        record["null_diverges"] = bool(record["null_differing"])
        record["passed"] = bool(record["identical"] and record["null_diverges"]
                                and repaired > 0)
    except Exception as error:  # noqa: BLE001 - a raised case is a recorded case
        record["error"] = f"{type(error).__name__}: {error}"
        record["passed"] = False
    record["seconds"] = round(time.time() - started, 3)
    return record


def fold_control(backend: str) -> dict:
    """Is the folded divergence the FOLD or the REPAIR? Measured, with NO SOURCE.

    The product refuses a mirrored axis by name, on the ground that
    ``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` run inside its seam and it
    carries neither. This is the measurement behind that clause: with NO source at all
    -- so no deposit and no repair -- the fused ordering still diverges from the
    driver's on a mirrored grid and does not on an unmirrored one. A refusal whose
    reason is measured is a different thing from one whose reason is asserted.
    """
    out = {"legs": []}
    for label, symmetry in (("mirrored_Y", ("Y",)), ("unmirrored", ())):
        grid_a, fields_a, pml_a = _build(cell=(1.6, 1.6, 1.6), symmetry=symmetry,
                                         boundaries="periodic", backend=backend)
        _grid_b, fields_b, pml_b = _build(cell=(1.6, 1.6, 1.6), symmetry=symmetry,
                                          boundaries="periodic", backend=backend)
        dt = grid_a.dt
        for step in range(6):
            _leg_a(fields_a, pml_a, [], step * dt, dt)
            _leg_b(fields_b, pml_b, [], step * dt, dt)
        differing = _differing(_snapshot(fields_a), _snapshot(fields_b))
        out["legs"].append({"label": label,
                            "has_symmetry": bool(grid_a.has_symmetry()),
                            "sources": 0,
                            "differing": differing,
                            "diverges": bool(differing)})
    mirrored = next(leg for leg in out["legs"] if leg["label"] == "mirrored_Y")
    unmirrored = next(leg for leg in out["legs"] if leg["label"] == "unmirrored")
    out["the_fold_is_what_diverges"] = bool(mirrored["diverges"]
                                            and not unmirrored["diverges"])
    return out


def refusal_leg() -> dict:
    """The named refusals, in both directions, on real engine objects.

    Every row is a configuration and the verdict the two paths give it. What this pins
    is that the FIRST repair's refusals are unchanged -- the plain configuration is
    still refused with the two texts the campaign cites -- while the second path
    admits it and refuses what it in turn cannot invert.
    """
    rows = []
    _grid, plain_fields, plain_pml = _build()
    _g2, active_fields, active_pml = _build()
    active_fields.enable_pml_storage()
    active_pml = PML(grid=active_fields.grid, thickness=(2, 2, 0))

    def row(label, fields, pml, pair, paths, expected):
        covered, reasons = deposit_repair.repairable(fields, pair, pml, paths=paths)
        rows.append({"case": label, "pair": pair, "paths": list(paths),
                     "covered": covered, "expected": expected,
                     "agrees": covered == expected, "reasons": list(reasons)})

    row("plain, split-field path (the standing refusal)", plain_fields, plain_pml, "D",
        (deposit_repair.SPLIT_FIELD_PATH,), False)
    row("plain, no layer at all, split-field path", plain_fields, None, "D",
        (deposit_repair.SPLIT_FIELD_PATH,), False)
    row("plain, plain path", plain_fields, plain_pml, "D", PLAIN, True)
    row("plain, plain path, no layer at all", plain_fields, None, "D", PLAIN, True)
    row("plain, plain path, MAGNETIC seam", plain_fields, plain_pml, "B", PLAIN, False)
    row("active layer, plain path", active_fields, active_pml, "D", PLAIN, False)
    row("active layer, split-field path", active_fields, active_pml, "D",
        (deposit_repair.SPLIT_FIELD_PATH,), True)
    return {"rows": rows, "all_agree": all(row["agrees"] for row in rows)}


#: name -> (config, component, center, size, steps).
CASES = (
    ("point_two_poles", dict(poles=2), "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("point_two_poles_24_steps", dict(poles=2), "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 24),
    ("line_two_poles", dict(poles=2), "Ez", (0.0, 0.0, 0.0), (0.6, 0.0, 0.0), 6),
    ("line_on_the_one_pole_component", dict(poles=2), "Ey", (0.0, 0.0, 0.0), (0.0, 0.6, 0.0), 6),
    ("point_no_poles", dict(poles=0), "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("point_five_poles", dict(poles=5), "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("point_conductive", dict(poles=2, conductivity=0.20), "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("line_conductive", dict(poles=2, conductivity=0.20), "Ex", (0.0, 0.0, 0.0), (0.6, 0.0, 0.0), 6),
    ("point_conductive_five_poles", dict(poles=5, conductivity=0.20), "Ez",
     (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("point_periodic", dict(poles=2, boundaries="periodic"), "Ez", (0.0, 0.0, 0.0),
     (0.0, 0.0, 0.0), 6),
    ("point_3d_conductive", dict(poles=2, conductivity=0.20, cell=(1.0, 1.0, 1.0)),
     "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "triton"), default="numpy")
    parser.add_argument("--out", default=None,
                        help="artifact path; written whether or not it releases")
    args = parser.parse_args(argv)

    started = time.time()
    print(f"[plain-deposit-repair] backend={args.backend} cases={len(CASES)}",
          flush=True)
    records = []
    for index, (name, config, component, center, size, steps) in enumerate(CASES, 1):
        record = run_case(name, steps=steps, component=component, center=center,
                          size=size, backend=args.backend, **config)
        records.append(record)
        print(f"  case {index}/{len(CASES)} {name:<32} "
              f"{'PASS' if record.get('passed') else 'FAIL'} "
              f"identical={record.get('identical')} "
              f"null_diverges={record.get('null_diverges')} "
              f"repaired={record.get('repaired_cells')} "
              f"({record.get('seconds')} s)", flush=True)

    control = fold_control(args.backend)
    print(f"  fold control: the_fold_is_what_diverges="
          f"{control['the_fold_is_what_diverges']}", flush=True)
    refusals = refusal_leg()
    print(f"  refusals: all_agree={refusals['all_agree']}", flush=True)

    reasons = []
    if not all(record.get("passed") for record in records):
        reasons.append("a case did not pass")
    if not control["the_fold_is_what_diverges"]:
        reasons.append("the fold control did not isolate the fold")
    if not refusals["all_agree"]:
        reasons.append("a refusal disagreed with its expectation")
    payload = {
        "probe": "plain_deposit_repair",
        "backend": args.backend,
        "cases": records,
        "cases_passed": sum(1 for record in records if record.get("passed")),
        "cases_total": len(records),
        "fold_control": control,
        "refusals": refusals,
        "seconds": round(time.time() - started, 3),
        "released": not reasons,
        "reasons": reasons,
    }
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
            _stamp_provenance(payload)
        except Exception as error:  # noqa: BLE001 - a stamp that cannot run is recorded
            payload["provenance_error"] = repr(error)
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"  artifact {path}", flush=True)
    print(f"VERDICT {'PASS' if not reasons else 'FAIL'} "
          f"{payload['cases_passed']}/{payload['cases_total']} cases "
          f"in {payload['seconds']}s", flush=True)
    return 0 if not reasons else 1


if __name__ == "__main__":
    raise SystemExit(main())
