"""Behaviour pins for the PML forms chosen per structure chunk (the region dispatch).

MEEP selects every PML branch per chunk (step_db.cpp:56-59, update_eh.cpp:162-163), and
the array path steps each form on its own cells from the run's ``chunk_map.RegionPlan``.
What is pinned here, and against what:

* the forms themselves, cell for cell, against a scalar per-cell reference that reads
  each cell's owning chunk (``ChunkLayout.membership_mask``) — on a separable division and
  on a non-separable one, for the curl (with and without a conductivity, kept by every
  chunk or by some) and the constitutive;
* the three routes (views, a select on the line, a select on the full mask) store the
  same word in every array of a stepped driver, across 2-D, 3-D, folded, one-sided,
  cylindrical, conductive (in every chunk and in some), dispersive and nonlinear
  configurations;
* the auxiliary contract: a NaN sentinel planted in every auxiliary word outside its
  form's cells survives stepping on every route, and the fields equal the unplanted run;
* the unit-coefficient precondition is refused by name through the driver and through
  ``Fields`` stepped directly, per line and per cell;
* MEEP's start of H (a copy of B on the dsigw cells at the first ``update_H``), the
  drive field (``f_w`` in the layer's chunks, the stored E elsewhere), and the deposit
  repair's plain form outside the layer's chunks.

Host NumPy only; no MEEP.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Dict

import numpy
import pytest

from . import deposit_repair, stepping
from .chunk_map import (ChunkLayout, ChunkLayoutRefusal, RegionPlan, chunk_bounds_record,
                        grid_frames)
from .dispersion import LORENTZIAN, Susceptibility
from .driver import FdtdDriver
from .fields import IYEE_SHIFTS
from .grid import Grid, Mirror
from .pml import PML

SENTINEL = numpy.uint32(0x7FC5A5A5)  # A quiet NaN with a payload: a leak would spread.
AUX_PREFIXES = ("fu_", "f_w_", "f_cond_")
COMPONENTS = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Ex", "Ey", "Ez")


def _words(array: Any) -> numpy.ndarray:
    return numpy.ascontiguousarray(array).reshape(-1).view(numpy.uint32)


def _cell_words(array: Any) -> numpy.ndarray:
    """One row of words per cell (two for complex storage); a view of a contiguous array."""
    return _words(array).reshape(array.size, -1)


def _state(driver: FdtdDriver) -> Dict[str, numpy.ndarray]:
    """Every array the step writes: the Fields dataclass's arrays and the polarizations'."""
    fields = driver.fields
    out: Dict[str, numpy.ndarray] = {}
    for field in dataclasses.fields(fields):
        value = getattr(fields, field.name, None)
        if isinstance(value, numpy.ndarray):
            out[field.name] = value.copy()
    for index, state in enumerate(fields.polarizations):
        for name in ("P", "P_prev"):
            for component, array in getattr(state, name).items():
                out[f"{name}[{index}].{component}"] = numpy.array(array, copy=True)
    return out


def _aux_regions(fields: Any) -> Dict[str, numpy.ndarray]:
    """Auxiliary name -> its form's cells (bool, the array's shape), from the installed plan."""
    plan = fields.region_plan
    regions: Dict[str, numpy.ndarray] = {}
    terms = {term.target: term for term in (*stepping.B_CURL_TERMS, *stepping.D_CURL_TERMS)}
    for prefix, component in itertools.product(AUX_PREFIXES, COMPONENTS):
        array = getattr(fields, prefix + component, None)
        if array is None:
            continue
        if prefix == "f_w_":
            mask = plan.mask("constitutive", component)
        elif prefix == "fu_":
            mask = plan.mask("curl", component)
        else:
            # f_cond: the dsig cells of the chunks that keep the component's conductivity.
            mask = plan.axis_mask("xyz".index(terms[component].dsig), component)
            mask = mask & plan.conductive_mask(component, fields.conductivity_for(component))
        regions[prefix + component] = numpy.broadcast_to(numpy.asarray(mask), array.shape)
    return regions


# --- configurations ----------------------------------------------------------------------


def _pml_2d():
    driver = FdtdDriver(cell_size=(4.0, 3.0, 0.0), resolution=10.0, dimensions=2)
    driver.setup_pml({"x": 6, "y": 5})
    driver.add_source({"component": "Ez", "center": (-1.6, 0.2, 0.0), "size": (0, 0, 0),
                       "frequency": 0.8})
    driver.add_source({"component": "Hz", "center": (0.3, -1.2, 0.0), "size": (0, 0, 0),
                       "frequency": 0.8})
    return driver


def _pml_3d():
    driver = FdtdDriver(cell_size=(2.0, 1.6, 1.8), resolution=10.0)
    driver.setup_pml(4)
    driver.add_source({"component": "Ey", "center": (-0.75, 0.1, 0.0), "size": (0, 0, 0),
                       "frequency": 0.8})
    driver.add_source({"component": "Hx", "center": (0.1, 0.0, 0.7), "size": (0, 0, 0),
                       "frequency": 0.8})
    return driver


def _folded_3d():
    driver = FdtdDriver(cell_size=(2.0, 1.6, 1.8), resolution=10.0,
                        symmetry=(Mirror("Y", -1), Mirror("Z", 1)))
    driver.setup_pml(4)
    driver.add_source({"component": "Ey", "center": (0.3, 0.0, 0.0), "size": (0, 0, 0),
                       "frequency": 0.8})  # Even under both planes.
    return driver


def _one_sided():
    driver = FdtdDriver(cell_size=(1.2, 1.2, 2.4), resolution=10.0, force_complex_fields=True)
    driver.setup_pml({"z": (0, 5)})
    driver.add_source({"component": "Ex", "center": (0.0, 0.0, 0.9), "size": (0, 0, 0),
                       "frequency": 0.8})
    return driver


def _cylindrical():
    driver = FdtdDriver(cell_size=(2.0, 0.0, 3.0), resolution=10.0, cylindrical=True, m=1,
                        dimensions=2, boundaries={"z": "metallic"})
    driver.setup_pml({"x": (0, 5), "z": (5, 5)})
    driver.add_source({"component": "Ez", "center": (1.7, 0.0, 0.2), "size": (0, 0, 0),
                       "frequency": 0.8})
    return driver


def _conductive():
    driver = FdtdDriver(cell_size=(4.0, 3.0, 0.0), resolution=10.0, dimensions=2)
    rng = numpy.random.default_rng(7)
    shape = driver.grid.shape
    driver.set_conductivity({c: (rng.random(shape) * 0.5).astype(numpy.float32)
                             for c in ("Dx", "Dy", "Dz")})
    driver.setup_pml({"x": 6, "y": 5})
    driver.add_source({"component": "Ez", "center": (0.0, 0.0, 0.0), "size": (0, 0, 0),
                       "frequency": 0.8})
    return driver


def _conductive_chunks():
    """D and B conductivity confined to one block that reaches into the x-high layer:
    most chunks keep none and step the lossless forms, two keep it (MEEP's per-chunk
    conductivity, structure.cpp:883-900)."""
    driver = FdtdDriver(cell_size=(4.0, 3.0, 0.0), resolution=10.0, dimensions=2)
    shape = driver.grid.shape
    block = numpy.zeros(shape, dtype=numpy.float32)
    block[24:38, 10:20] = 0.5
    driver.set_conductivity({c: block for c in ("Dx", "Dy", "Dz")})
    driver.set_b_conductivity({c: block * numpy.float32(0.6) for c in ("Bx", "By", "Bz")})
    driver.setup_pml({"x": 6, "y": 5})
    driver.add_source({"component": "Ez", "center": (0.0, 0.0, 0.0), "size": (0, 0, 0),
                       "frequency": 0.8})
    driver.add_source({"component": "Hz", "center": (0.35, -0.25, 0.0), "size": (0, 0, 0),
                       "frequency": 0.8})
    return driver


def _dispersive():
    driver = FdtdDriver(cell_size=(4.0, 3.0, 0.0), resolution=10.0, dimensions=2)
    term = Susceptibility(frequency=1.1, gamma=0.05, kind=LORENTZIAN)
    driver.add_susceptibility(term, {c: numpy.full(driver.grid.shape, 0.4, dtype=numpy.float32)
                                     for c in ("Ex", "Ey", "Ez")})
    driver.setup_pml({"x": 6, "y": 5})
    driver.add_source({"component": "Ey", "center": (-1.6, 0.1, 0.0), "size": (0, 0, 0),
                       "frequency": 0.8})
    return driver


def _nonlinear():
    driver = FdtdDriver(cell_size=(4.0, 3.0, 0.0), resolution=10.0, dimensions=2)
    driver.set_chi3({c: numpy.full(driver.grid.shape, 0.02, dtype=numpy.float32)
                     for c in ("Ex", "Ey", "Ez")})
    driver.setup_pml({"x": 6, "y": 5})
    driver.add_source({"component": "Ez", "center": (-1.6, 0.2, 0.0), "size": (0, 0, 0),
                       "frequency": 0.8, "amplitude": 5.0})
    return driver


CONFIGURATIONS = {
    "pml_2d": _pml_2d, "pml_3d": _pml_3d, "folded_3d": _folded_3d, "one_sided": _one_sided,
    "cylindrical": _cylindrical, "conductive": _conductive,
    "conductive_chunks": _conductive_chunks, "dispersive": _dispersive,
    "nonlinear": _nonlinear,
}
STEPS = 12


def _force(driver: FdtdDriver, route):
    if route is not None:
        driver.fields.install_region_plan(
            RegionPlan(driver.chunk_layout, driver.grid, driver.pml, route=route))
    return driver


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_the_three_routes_store_the_same_word_in_every_array(name):
    """Views, a select on the line and a select on the full mask: one word per array.

    Every form is pointwise, so the route is a cost choice; this is the measurement that
    it is only that. The default plan mixes routes per component (NumPy selects on the
    innermost axis), and each forced route is compared with it.
    """
    states = {}
    for route in (None, "split", "select", "mask"):
        driver = _force(CONFIGURATIONS[name](), route)
        try:
            for _ in range(STEPS):
                driver.step()
            plan = driver.fields.region_plan
            assert plan.route == route
            states[route] = _state(driver)
        finally:
            driver.close()
    reference = states[None]
    moved = [key for key in reference if numpy.any(reference[key] != 0)]
    assert any(key.startswith(("D", "B")) for key in moved), "the case stepped nothing"
    for route in ("split", "select", "mask"):
        assert sorted(states[route]) == sorted(reference)
        differing = {key: int(numpy.count_nonzero(_words(reference[key]) != _words(states[route][key])))
                     for key in reference}
        assert not any(differing.values()), (name, route, {k: v for k, v in differing.items() if v})


@pytest.mark.parametrize("route", [None, "split", "select", "mask"])
@pytest.mark.parametrize("name", ["pml_2d", "pml_3d", "folded_3d", "conductive", "conductive_chunks",
                                  "dispersive"])
def test_an_auxiliary_sentinel_outside_its_cells_survives_and_moves_nothing(name, route):
    """The auxiliary contract: no path reads or writes an auxiliary word outside its cells.

    A NaN sentinel (0x7fc5a5a5) in every such word before the first step is still there
    after twelve, and every field equals the unplanted run's word for word; a sentinel
    read by any form would spread a NaN into the field.
    """
    plain = _force(CONFIGURATIONS[name](), route)
    planted = _force(CONFIGURATIONS[name](), route)
    try:
        regions = _aux_regions(planted.fields)
        assert regions
        count = 0
        for key, inside in regions.items():
            rows = _cell_words(getattr(planted.fields, key))
            outside = ~inside.reshape(-1)
            rows[outside] = SENTINEL
            count += int(numpy.count_nonzero(outside))
        assert count > 0
        for _ in range(STEPS):
            plain.step()
            planted.step()
        for key, inside in regions.items():
            rows = _cell_words(getattr(planted.fields, key))
            lost = int(numpy.count_nonzero(rows[~inside.reshape(-1)] != SENTINEL))
            assert lost == 0, (name, route, key, lost)
        a, b = _state(plain), _state(planted)
        for key in a:
            if any(key.startswith(prefix) for prefix in AUX_PREFIXES):
                inside = regions[key].reshape(-1)
                assert numpy.array_equal(_cell_words(a[key])[inside], _cell_words(b[key])[inside]), key
            else:
                assert numpy.array_equal(_words(a[key]), _words(b[key])), key
    finally:
        plain.close()
        planted.close()


# --- the forms, cell by cell -------------------------------------------------------------


def _small_grid():
    grid = Grid(resolution=10.0, cell_size=(4.0, 4.0, 0.0), dimensions=2,
                boundaries={"x": "metallic", "y": "metallic"})
    return grid, PML(grid, thickness={"x": 4, "y": 4})


def _local_split_layout(grid, layer):
    """The one-rank boxes with the low-y PML chunk over the middle x range cut LOCALLY.

    The cut sits at the sigma edge, so the sliver ``(lo + 8, lo + 10]`` reaches no sigma
    and carries no y layer (``structure_chunk::use_pml``), the mechanism behind a forced
    ``num_chunks`` division that is not separable (phase 0's P0-a).
    """
    one = ChunkLayout.one_rank(grid, layer)
    frames = grid_frames(grid)
    lo_y = frames[1].window[0]
    middle_x = sorted({(lo[0], hi[0]) for lo, hi in one.boxes})[1]
    boxes, flags = [], []
    for (lo, hi), flag in zip(one.boxes, one.pml_flags):
        if (lo[0], hi[0]) == middle_x and (lo[1], hi[1]) == (lo_y, lo_y + 10):
            boxes += [(lo, (hi[0], lo_y + 8, hi[2])), ((lo[0], lo_y + 8, lo[2]), hi)]
            flags += [flag, (flag[0], False, flag[2])]
        else:
            boxes.append((lo, hi))
            flags.append(flag)
    return ChunkLayout(boxes, flags, frames, faces=layer.thickness_by_face, source="meep")


def _f32(value):
    return numpy.float32(value)


def _vector(layer, kind, axis, half):
    return numpy.asarray(getattr(layer, f"{kind}_{'xyz'[axis]}{'_h' if half else ''}")).reshape(-1)


def _reference_curl(layout, layer, term, field, curl, fu, condfac=None, condinv=None, f_cond=None,
                    conductive=None):
    """MEEP's curl forms, one cell at a time, from each cell's owning chunk.

    With a conductivity (``f_cond`` given) ``conductive`` is the per-cell volume of the
    chunks that keep it, or ``None`` when every chunk does; a cell of any other chunk
    takes the lossless forms (step_generic.cpp:84-246). The increment ``(fcnd' - fcnd)``
    is formed before it is added (:167, :209), and ``f_cond`` changes only on the dsig
    cells of the chunks that keep a conductivity.
    """
    iyee = IYEE_SHIFTS[term.target]
    a_sig, a_sigu = "xyz".index(term.dsig), "xyz".index(term.dsigu)
    dsig = layout.membership_mask(a_sig, iyee)
    dsigu = layout.membership_mask(a_sigu, iyee)
    kms, sinv = _vector(layer, "kms", a_sig, bool(iyee[a_sig])), _vector(layer, "sinv", a_sig, bool(iyee[a_sig]))
    kmu, sinu = _vector(layer, "kms", a_sigu, bool(iyee[a_sigu])), _vector(layer, "sinv", a_sigu, bool(iyee[a_sigu]))
    f, u = field.copy(), fu.copy()
    c_new = None if f_cond is None else f_cond.copy()
    for cell in numpy.ndindex(field.shape):
        k, ku = cell[a_sig], cell[a_sigu]
        x, prev = f[cell], u[cell]
        lossy = f_cond is not None and (conductive is None or bool(conductive[cell]))
        if not lossy:
            if dsigu[cell]:
                n = (_f32(kms[k] * prev) - curl[cell]) * sinv[k]
                f[cell] = ((x * kmu[ku] + n) - prev) * sinu[ku]
                u[cell] = n
            else:
                f[cell] = (x * kms[k] - curl[cell]) * sinv[k]
            continue
        cprev = f_cond[cell]
        increment = None
        if dsig[cell]:
            c_new[cell] = (cprev * condfac[cell] - curl[cell]) * condinv[cell]
            increment = c_new[cell] - cprev
        if dsigu[cell]:
            if dsig[cell]:
                n = (prev * kms[k] + increment) * sinv[k]
            else:
                n = (prev * condfac[cell] - curl[cell]) * condinv[cell]
            f[cell] = ((x * kmu[ku] + n) - prev) * sinu[ku]
            u[cell] = n
        elif dsig[cell]:
            f[cell] = (x * kms[k] + increment) * sinv[k]
        else:
            f[cell] = (x * condfac[cell] - curl[cell]) * condinv[cell]
    return f, u, c_new


def _meep_conductive_coefficients(sigma, dt):
    """condfac and condinv as MEEP forms them (step_generic.cpp:88, structure.cpp:699)."""
    dt2 = numpy.float32(numpy.float32(dt) * numpy.float32(0.5))
    condfac = (numpy.float32(1.0) - sigma * dt2).astype(numpy.float32)
    condinv = (1.0 / (1.0 + (sigma.astype(numpy.float64) * float(dt)) * 0.5)).astype(numpy.float32)
    return condfac, condinv


def _random(shape, seed):
    rng = numpy.random.default_rng(seed)
    return (rng.standard_normal(shape) * 0.5).astype(numpy.float32)


@pytest.mark.parametrize("division", ["one_rank", "local_split"])
@pytest.mark.parametrize("route", [None, "split", "select", "mask"])
@pytest.mark.parametrize("lossy", ["every_chunk", "low_x_chunks"])
def test_each_curl_cell_steps_its_own_chunks_form(division, route, lossy):
    """The curl, with and without a conductivity, against a per-cell reference.

    The reference reads each cell's owning chunk from the full membership volume and
    applies MEEP's form for it — the recurrence where the chunk carries dsigu, the
    dsig-only form (``kms``/``sinv`` exactly 1 outside dsig make it the plain form)
    elsewhere, and, with a conductivity, MEEP's four conductive forms on the dsig and
    dsigu memberships in the chunks that keep it. ``low_x_chunks`` confines the
    conductivity to x < 0, so the high-x chunks keep none and step the lossless forms
    although the component is lossy. The local split is not separable along y, so every
    route there is the mask route.
    """
    grid, layer = _small_grid()
    layout = ChunkLayout.one_rank(grid, layer) if division == "one_rank" else _local_split_layout(grid, layer)
    assert layout.is_separable() == (division == "one_rank")
    plan = RegionPlan(layout, grid, layer, route=route)
    shape = grid.shape
    rng = numpy.random.default_rng(20261007)
    sigma = (0.1 + rng.random(shape) * 0.6).astype(numpy.float32)
    if lossy == "low_x_chunks":
        sigma[shape[0] // 2:] = 0.0
    condfac, condinv = _meep_conductive_coefficients(sigma, grid.dt)
    for index, term in enumerate((*stepping.B_CURL_TERMS, *stepping.D_CURL_TERMS)):
        half = term.target[0] == "B"
        kms, sinv = stepping._curl_coefficients(layer, term.dsig, half)
        kmu, sinu = stepping._curl_coefficients(layer, term.dsigu, half)
        field, curl, fu = _random(shape, 3 * index), _random(shape, 3 * index + 1), _random(shape, 3 * index + 2)
        want_f, want_u, _ = _reference_curl(layout, layer, term, field, curl, fu)
        got_f, got_u = field.copy(), fu.copy()
        stepping._apply_curl_regions(plan, term.target, got_f, curl, kms, sinv, kmu, sinu, got_u,
                                     numpy)
        assert numpy.array_equal(_words(got_f), _words(want_f)), (term.target, division, route)
        assert numpy.array_equal(_words(got_u), _words(want_u)), (term.target, division, route)
        chunks = plan.conductive_chunks(term.target, sigma)
        if lossy == "every_chunk":
            assert all(chunks)
        else:
            assert any(chunks) and not all(chunks), chunks
        conductive = None if all(chunks) else plan.conductive_mask(term.target, sigma)
        f_cond = _random(shape, 100 + index)
        want_f, want_u, want_c = _reference_curl(layout, layer, term, field, curl, fu, condfac,
                                                 condinv, f_cond, conductive)
        got_f, got_u, got_c = field.copy(), fu.copy(), f_cond.copy()
        stepping._apply_conductive_pml_update(
            numpy, got_f, curl, condfac, condinv, kms, sinv, kmu, sinu, got_u, got_c,
            dsig_active=plan.axis_mask("xyz".index(term.dsig), term.target),
            dsigu_active=plan.mask("curl", term.target), conductive=conductive)
        for got, want, what in ((got_f, want_f, "f"), (got_u, want_u, "fu"), (got_c, want_c, "f_cond")):
            assert numpy.array_equal(_words(got), _words(want)), (term.target, what, division, route)


@pytest.mark.parametrize("division", ["one_rank", "local_split"])
@pytest.mark.parametrize("route", [None, "split", "select", "mask"])
def test_each_constitutive_cell_steps_its_own_chunks_form(division, route):
    """The constitutive: the accumulation in the dsigw chunks, ``f = src`` elsewhere."""
    grid, layer = _small_grid()
    layout = ChunkLayout.one_rank(grid, layer) if division == "one_rank" else _local_split_layout(grid, layer)
    plan = RegionPlan(layout, grid, layer, route=route)
    shape = grid.shape
    for index, (component, _source, own) in enumerate((*stepping.H_CONSTITUTIVE_TERMS,
                                                         *stepping.E_CONSTITUTIVE_TERMS)):
        iyee = IYEE_SHIFTS[component]
        axis = "xyz".index(own)
        half = bool(iyee[axis])
        kps, kms = stepping._constitutive_coefficients(layer, own, half)
        kpv, kmv = _vector(layer, "kps", axis, half), _vector(layer, "kms", axis, half)
        inside = layout.membership_mask(axis, iyee)
        field, source, fw = (_random(shape, 7 * index + n) for n in range(3))
        want_f, want_w = field.copy(), fw.copy()
        for cell in numpy.ndindex(shape):
            if inside[cell]:
                prev = want_w[cell]
                want_w[cell] = source[cell]
                want_f[cell] = want_f[cell] + (kpv[cell[axis]] * source[cell] - kmv[cell[axis]] * prev)
            else:
                want_f[cell] = source[cell]
        got_f, got_w = field.copy(), fw.copy()
        stepping._apply_constitutive_regions(plan, component, got_f, source, kps, kms, got_w, numpy)
        assert numpy.array_equal(_words(got_f), _words(want_f)), (component, division, route)
        assert numpy.array_equal(_words(got_w), _words(want_w)), (component, division, route)


def test_a_non_separable_division_steps_through_the_driver_on_the_mask_route():
    """A lifted division the kernel record cannot hold still steps, on the array path."""
    grid, layer = _small_grid()
    layout = _local_split_layout(grid, layer)
    driver = FdtdDriver(cell_size=(4.0, 4.0, 0.0), resolution=10.0, dimensions=2,
                        boundaries={"x": "metallic", "y": "metallic"})
    try:
        driver.set_chunk_layout(ChunkLayout(layout.boxes, layout.pml_flags, grid_frames(driver.grid),
                                            faces=layout.faces, source="meep"))
        driver.setup_pml({"x": 4, "y": 4})
        driver.add_source({"component": "Ez", "center": (0.0, -1.55, 0.0), "size": (0, 0, 0),
                           "frequency": 0.8})
        plan = driver.fields.region_plan
        assert {plan.spec("curl", "Dz").mode, plan.spec("constitutive", "Ey").mode} == {"mask"}
        with pytest.raises(ChunkLayoutRefusal, match="not separable"):
            chunk_bounds_record(driver.chunk_layout)
        for _ in range(STEPS):
            driver.step()
        assert numpy.isfinite(driver.fields.Dz).all() and numpy.any(driver.fields.Dz != 0)
    finally:
        driver.close()


# --- the precondition ---------------------------------------------------------------------


def test_a_coefficient_off_one_outside_its_run_is_refused_by_name_before_anything_steps():
    """D2's one check, on every route into a step: the driver and ``Fields`` stepped directly.

    A division that leaves a graded row outside every chunk carrying its layer would step
    that row with the plain form where MEEP's own chunk carries sigma; it is refused,
    naming the coefficient and the row, when the plan is built.
    """
    grid, layer = _small_grid()
    one = ChunkLayout.one_rank(grid, layer)
    # Every chunk declared free of the x layer: the graded x rows are outside the run.
    stripped = ChunkLayout(one.boxes, [(False, flags[1], flags[2]) for flags in one.pml_flags],
                           one.frames, faces=one.faces, source="meep")
    driver = FdtdDriver(cell_size=(4.0, 4.0, 0.0), resolution=10.0, dimensions=2,
                        boundaries={"x": "metallic", "y": "metallic"})
    try:
        driver.set_chunk_layout(stripped)
        with pytest.raises(ChunkLayoutRefusal, match=r"(kms|sinv|kps)_x(_h)?\[\d+\]"):
            driver.setup_pml({"x": 4, "y": 4})
    finally:
        driver.close()
    # Fields stepped directly build the one-rank plan; a coefficient planted off 1.0
    # outside its run is refused before the field moves.
    driver = FdtdDriver(cell_size=(4.0, 4.0, 0.0), resolution=10.0, dimensions=2,
                        boundaries={"x": "metallic", "y": "metallic"})
    try:
        driver.setup_pml({"x": 4, "y": 4})
        layer = PML(driver.grid, thickness={"x": 4, "y": 4})
        layer.sinv_x = layer.sinv_x.copy()
        layer.sinv_x[20, 0, 0] = numpy.float32(0.5)
        before = {name: getattr(driver.fields, name).copy() for name in ("Dx", "Dy", "Dz")}
        with pytest.raises(ChunkLayoutRefusal, match=r"sinv_x\[20\]"):
            stepping.step_D(driver.fields, layer)
        for name, values in before.items():
            assert numpy.array_equal(values, getattr(driver.fields, name)), name
    finally:
        driver.close()


def test_the_precondition_is_checked_per_cell_on_a_non_separable_division():
    """Where no line describes the division, the mask route checks the coefficient per cell."""
    grid, layer = _small_grid()
    layout = _local_split_layout(grid, layer)
    RegionPlan(layout, grid, layer)  # MEEP's own sliver reaches no sigma: accepted.
    # Move the cut one doubled cell into the graded rows: the sliver now owns a row whose
    # sigma is not zero, under a chunk that declares no y layer.
    frames = grid_frames(grid)
    lo_y = frames[1].window[0]
    boxes = [(lo, (hi[0], lo_y + 6, hi[2])) if hi[1] == lo_y + 8 else
             (((lo[0], lo_y + 6, lo[2]), hi) if lo[1] == lo_y + 8 else (lo, hi))
             for lo, hi in layout.boxes]
    moved = ChunkLayout(boxes, layout.pml_flags, frames, faces=layout.faces, source="meep")
    assert not moved.is_separable()
    with pytest.raises(ChunkLayoutRefusal, match=r"_y(_h)?\[\d+\] = .* on a cell of a row"):
        RegionPlan(moved, grid, layer)


# --- MEEP's start of H ----------------------------------------------------------------------


def test_the_first_update_H_starts_H_and_its_f_w_from_B_on_the_dsigw_cells():
    """MEEP's H is B's storage until the first update_eh(H): then a COPY, and f_w a copy of it.

    A magnetic current deposited inside the x layer before that first update leaves B
    nonzero on Hx's dsigw cells with ``kms != 1``; MEEP's step-1 H word there is
    ``B + ((kps*B) - (kms*B))`` (phase 0's P0-f: 4 of 4 cells, where the zero start
    ``0 + ((kps*B) - (kms*0))`` reproduces 0 of 4). This pins the package's own words to
    the copy start; the MEEP comparison itself is the first MEEP check's.
    """
    driver = FdtdDriver(cell_size=(4.0, 4.0, 0.0), resolution=10.0, dimensions=2)
    try:
        driver.setup_pml({"x": 10, "y": 10})
        source = driver.add_source({"component": "Hx", "center": (-1.65, 0.05, 0.0),
                                    "size": (0, 0, 0), "frequency": 0.8})
        assert driver.fields.h_separated is False
        b_after_injection = {}
        original = source.inject

        def inject(fields, time):  # Capture B as update_H will see it.
            original(fields, time)
            b_after_injection["Bx"] = fields.Bx.copy()

        source.inject = inject
        driver.step()
        assert driver.fields.h_separated is True
        b1 = b_after_injection["Bx"]
        kps, kms = stepping._constitutive_coefficients(driver.pml, "x", False)
        kps_full = numpy.broadcast_to(kps, b1.shape)
        kms_full = numpy.broadcast_to(kms, b1.shape)
        cells = numpy.nonzero(b1)
        assert cells[0].size >= 2
        inside = numpy.broadcast_to(driver.fields.region_plan.mask("constitutive", "Hx"), b1.shape)[cells]
        assert inside.all()
        copy_start = b1[cells] + ((kps_full[cells] * b1[cells]) - (kms_full[cells] * b1[cells]))
        zero_start = numpy.float32(0) + ((kps_full[cells] * b1[cells]) - (kms_full[cells] * numpy.float32(0)))
        assert numpy.array_equal(_words(driver.fields.Hx[cells]), _words(copy_start))
        assert numpy.array_equal(_words(driver.fields.f_w_Hx[cells]), _words(b1[cells]))
        assert not numpy.array_equal(_words(copy_start), _words(zero_start)), (
            "the deposit cells' kms is 1, so this case cannot tell the two starts apart")
    finally:
        driver.close()


def test_the_start_runs_once_and_a_state_seeded_mid_run_keeps_its_H():
    """``h_separated`` is state: the start never runs again, and a seeded state opts out."""
    grid, layer = _small_grid()
    driver = FdtdDriver(cell_size=(4.0, 4.0, 0.0), resolution=10.0, dimensions=2,
                        boundaries={"x": "metallic", "y": "metallic"})
    try:
        driver.setup_pml({"x": 4, "y": 4})
        fields = driver.fields
        for name in ("Bx", "Hx", "f_w_Hx"):
            getattr(fields, name)[...] = _random(fields.Bx.shape, hash(name) % 1000)
        fields.h_separated = True  # A mid-run state.
        hx, fw = fields.Hx.copy(), fields.f_w_Hx.copy()
        stepping.separate_h_from_b(fields, driver.pml)
        assert numpy.array_equal(hx, fields.Hx) and numpy.array_equal(fw, fields.f_w_Hx)
        fields.h_separated = False
        stepping.separate_h_from_b(fields, driver.pml)
        inside = numpy.broadcast_to(fields.region_plan.mask("constitutive", "Hx"), hx.shape)
        assert numpy.array_equal(fields.Hx[inside], fields.Bx[inside])
        assert numpy.array_equal(fields.f_w_Hx[inside], fields.Bx[inside])
        assert numpy.array_equal(fields.Hx[~inside], hx[~inside])
        assert numpy.array_equal(fields.f_w_Hx[~inside], fw[~inside])
        assert fields.h_separated is True
        driver.reset()
        assert fields.h_separated is False
    finally:
        driver.close()


# --- the drive field and the deposit repair --------------------------------------------------


def test_the_drive_field_is_f_w_in_the_layer_chunks_and_the_stored_E_elsewhere():
    """MEEP's ``w = f_w ? f_w : f`` per chunk (update_pols.cpp:44)."""
    driver = _dispersive()
    try:
        for _ in range(STEPS):
            driver.step()
        fields = driver.fields
        for component in ("Ex", "Ey", "Ez"):
            mode = fields.region_plan.spec("constitutive", component).mode
            drive = fields.drive_field(component)
            if mode == "outside_only":
                assert drive is getattr(fields, component)
            inside = numpy.broadcast_to(fields.region_plan.mask("constitutive", component), drive.shape)
            stored, aux = getattr(fields, component), getattr(fields, "f_w_" + component)
            assert numpy.array_equal(_words(drive[inside]), _words(aux[inside])), component
            assert numpy.array_equal(_words(drive[~inside]), _words(stored[~inside])), component
        fields.install_region_plan(None)
        with pytest.raises(RuntimeError, match="region plan"):
            fields.drive_field("Ex")
    finally:
        driver.close()


@pytest.mark.parametrize("center,inside_expected", [((0.1, 0.0, 0.0), False), ((-0.9, 0.0, 0.1), True)])
def test_the_deposit_repair_takes_the_plain_form_outside_the_layer_chunks(center, inside_expected):
    """The repair recomputes each deposit cell with its chunk's form, as ``update_E`` does.

    A fused launch steps the constitutive against an uninjected D and the repair rewrites
    the deposit cells afterwards: the accumulation where the cell's chunk carries the
    target's own layer, ``E = fresh`` elsewhere with ``f_w`` left alone. Compared word for
    word with the driver's own order on a 3-D grid, for an Ex deposit in the interior and
    one in the x layer.
    """
    def build():
        driver = FdtdDriver(cell_size=(2.0, 1.6, 1.8), resolution=10.0)
        driver.setup_pml(3)
        driver.add_source({"component": "Ex", "center": center, "size": (0, 0, 0),
                           "frequency": 0.8})
        return driver

    reference, bracketed = build(), build()
    try:
        for _ in range(3):
            reference.step()
            bracketed.step()
        fields, pml, sources = bracketed.fields, bracketed.pml, bracketed._sources
        plan = fields.region_plan
        index = deposit_repair._deposit_index(sources[0])
        cells_inside = numpy.broadcast_to(plan.mask("constitutive", "Ex"), fields.Ex.shape)[index]
        assert bool(cells_inside.all()) is inside_expected and bool(cells_inside.any()) is inside_expected
        fw_before = fields.f_w_Ex.copy()
        # One step in the bracket order: save, the constitutive against the uninjected D,
        # the injection and fills, then the repair.
        reference.step()
        stepping.step_B(fields, pml)
        stepping.fill_symmetry_bc_B(fields)
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        stepping.update_H(fields, pml)
        saved = deposit_repair.save(fields, sources, "D", pml)
        stepping.step_D(fields, pml)
        stepping.zero_metal_D(fields)
        stepping.update_E(fields, pml)
        for source in sources:
            source.inject(fields, bracketed.time + 0.5 * bracketed.dt)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        deposit_repair.apply(fields, pml, sources, "D", saved)
        stepping.update_P(fields, pml)
        for name in ("Dx", "Ex", "f_w_Ex", "Ey", "Ez"):
            assert numpy.array_equal(_words(getattr(fields, name)), _words(getattr(reference.fields, name))), name
        if not inside_expected:
            outside = ~numpy.broadcast_to(plan.mask("constitutive", "Ex"), fw_before.shape)
            assert numpy.array_equal(_words(fields.f_w_Ex[outside]), _words(fw_before[outside]))
    finally:
        reference.close()
        bracketed.close()
