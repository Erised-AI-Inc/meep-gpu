"""The deposit repair, against the driver's own order, on both seams.

``parity/meep_gpu/probe_deposit_repair.py`` proved this shape on the electric seam, on the
host and then with the real Triton kernel. This is the package-level test of the component
that shape became, and it adds the seam the probe never covered: the MAGNETIC one, worth
58 of the 215 excluded instances.

Weighted the way the risk is. A repair that is subtly wrong produces fields nobody
notices, so every case pairs the comparison with a NULL CONTROL -- the same protocol with
the repair withheld, which must diverge. A case where both legs agree is a case that
cannot tell a working repair from a missing one.
"""

from __future__ import annotations

import numpy
import pytest

from . import deposit_repair, stepping
from .dispersion import PolarizationState, Susceptibility
from .fields import Fields
from .grid import Grid
from .pml import PML
from .sources import GaussianEnvelope, GaussianPulsedSource, VolumeSource

SEEDED = ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Bx", "By", "Bz",
          "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
          "fu_Dx", "fu_Dy", "fu_Dz", "fu_Bx", "fu_By", "fu_Bz")


def _seed(array, generator):
    """Fill with noise, real or complex.

    NOT zeros: zero-init is a FIXED POINT of the constitutive recurrence, so a zeroed
    case passes whether or not the repair does anything.
    """
    real = generator.standard_normal(array.shape)
    if numpy.iscomplexobj(array):
        array[...] = (real + 1j * generator.standard_normal(array.shape)).astype(
            array.dtype)
    else:
        array[...] = real.astype(array.dtype)


def _build(cell=(1.2, 1.2, 0.0), resolution=12.0, thickness=2, seed=20260823,
           complex_storage=False, dispersive=False):
    """The triple the comparisons run on.

    ``complex_storage`` and ``dispersive`` are the two configurations
    ``metal_kernels.complex_fused_magnetic_pair`` and
    ``metal_kernels.fused_dispersive_pair`` serve; both declared
    ``CARRIES_DEPOSIT_REPAIR`` on 2026-08-28 and neither constitutive half had ever
    been driven through this repair. See :data:`COMPLEX_CASES` and
    :data:`DISPERSIVE_CASES`.
    """
    dimensions = sum(1 for extent in cell if extent > 0.0)
    grid = Grid(resolution=resolution, cell_size=cell, dimensions=dimensions)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    if dispersive:
        # A Lorentzian on Ez, built the way `metal_composition_matrix._polarization`
        # builds one, so `displacement_minus_polarization` really is D - sum P and the
        # dispersive branch of `update_E` is the one under test.
        sigma = {name: (0.25 if name == "Ez" else 0.0)
                 for name in ("Ex", "Ey", "Ez")}
        fields.polarizations.append(PolarizationState(
            Susceptibility(1.0, 0.1, "lorentzian"), sigma, grid,
            numpy.complex64 if complex_storage else numpy.float32))
    pml = PML(grid=grid, thickness=tuple(thickness if e > 0.0 else 0 for e in cell))
    fields.enable_pml_storage()
    generator = numpy.random.default_rng(seed)
    for name in SEEDED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        _seed(array, generator)
    # The pole's own recurrence state. Left at zero it would be a fixed point of the
    # ADE just as a zeroed field is of the constitutive recurrence, and D - sum P
    # would equal D -- which is the ORDINARY case wearing a dispersive label.
    for state in fields.polarizations:
        for attribute in ("P", "P_prev"):
            for array in (getattr(state, attribute, None) or {}).values():
                if array is not None:
                    _seed(array, generator)
    return grid, fields, pml


def _words(fields):
    out = {}
    for name in SEEDED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        out[name] = numpy.ascontiguousarray(array).ravel().view(numpy.uint8)
    # THE POLE'S STATE IS PART OF THE COMPARISON. Without it a repair that corrupted
    # P while getting E right would read as byte-identical.
    for index, state in enumerate(fields.polarizations):
        for attribute in ("P", "P_prev"):
            for component, array in (getattr(state, attribute, None) or {}).items():
                if array is None:
                    continue
                out[f"pol{index}.{attribute}.{component}"] = (
                    numpy.ascontiguousarray(array).ravel().view(numpy.uint8))
    return out


def _differing(a, b):
    return {n: int((a[n] != b[n]).sum()) for n in a if int((a[n] != b[n]).sum())}


def _reference(fields, pml, sources, when, dt, pair):
    """The driver's order, verbatim (driver.py:3281-3304)."""
    if pair == "B":
        stepping.step_B(fields, pml)
        for s in sources:
            s.inject(fields, when)
        stepping.fill_symmetry_bc_B(fields)
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
        stepping.update_H(fields, pml)
    else:
        stepping.step_D(fields, pml)
        for s in sources:
            s.inject(fields, when + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
        stepping.update_E(fields, pml)


def _fused_then_repair(fields, pml, sources, when, dt, pair, repair=True):
    """The fused product's two halves against an uninjected field, then the repair."""
    saved = deposit_repair.save(fields, sources, pair) if repair else None
    if pair == "B":
        stepping.step_B(fields, pml)
        stepping.zero_metal_B(fields)      # the kernel's inline ZM_*
        stepping.update_H(fields, pml)
        for s in sources:
            s.inject(fields, when)
        stepping.fill_symmetry_bc_B(fields)
        stepping.zero_metal_B(fields)
        stepping.fill_folded_far_ghosts_B(fields)
    else:
        stepping.step_D(fields, pml)
        stepping.zero_metal_D(fields)
        stepping.update_E(fields, pml)
        for s in sources:
            s.inject(fields, when + 0.5 * dt)
        stepping.fill_symmetry_bc_D(fields)
        stepping.zero_metal_D(fields)
        stepping.fill_folded_far_ghosts_D(fields)
    if repair:
        deposit_repair.apply(fields, pml, sources, pair, saved)


def _make_source(grid, pair, component, center, size):
    """MAGNETIC SOURCES GO THROUGH VolumeSource. `GaussianPulsedSource` refuses an H
    component by name ("Magnetic components need VolumeSource", sources.py:205) -- which
    is why the magnetic half of this seam had never been exercised: the obvious
    constructor cannot express it."""
    if pair == "B":
        return VolumeSource(grid=grid, component=component, center=center, size=size,
                            envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                            amplitude=1.0)
    return GaussianPulsedSource(grid=grid, component=component, center=center, size=size,
                                frequency=1.0, fwidth=0.2, amplitude=1.0)


CASES = [
    ("electric point", "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("electric point, 24 steps", "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 24),
    ("electric line", "D", "Ez", (0.0, 0.0, 0.0), (0.6, 0.0, 0.0), 6),
    ("electric in the PML shell", "D", "Ez", (0.5, 0.5, 0.0), (0.0, 0.0, 0.0), 6),
    # THE SEAM THE PROBE NEVER COVERED -- 58 of the 215 excluded instances.
    ("magnetic point", "B", "Hz", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("magnetic point, 24 steps", "B", "Hz", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 24),
    ("magnetic line", "B", "Hx", (0.0, 0.0, 0.0), (0.0, 0.6, 0.0), 6),
]


@pytest.mark.parametrize("name,pair,component,center,size,steps", CASES,
                         ids=[c[0] for c in CASES])
def test_fuse_then_repair_is_byte_identical_to_the_driver_order(
        name, pair, component, center, size, steps):
    grid_a, fields_a, pml_a = _build()
    grid_b, fields_b, pml_b = _build()
    assert not _differing(_words(fields_a), _words(fields_b)), "preconditions differ"
    src_a = [_make_source(grid_a, pair, component, center, size)]
    src_b = [_make_source(grid_b, pair, component, center, size)]
    assert src_a[0]._n_source_points, f"{name} deposits nothing and cannot discriminate"
    dt = grid_a.dt
    for step in range(steps):
        _reference(fields_a, pml_a, src_a, step * dt, dt, pair)
        _fused_then_repair(fields_b, pml_b, src_b, step * dt, dt, pair)
    assert not _differing(_words(fields_a), _words(fields_b)), name


@pytest.mark.parametrize("name,pair,component,center,size,steps", CASES,
                         ids=[c[0] for c in CASES])
def test_withholding_the_repair_diverges(name, pair, component, center, size, steps):
    """The null control. Without this, a passing case proves only that both legs agree."""
    grid_a, fields_a, pml_a = _build()
    grid_b, fields_b, pml_b = _build()
    src_a = [_make_source(grid_a, pair, component, center, size)]
    src_b = [_make_source(grid_b, pair, component, center, size)]
    dt = grid_a.dt
    for step in range(steps):
        _reference(fields_a, pml_a, src_a, step * dt, dt, pair)
        _fused_then_repair(fields_b, pml_b, src_b, step * dt, dt, pair, repair=False)
    assert _differing(_words(fields_a), _words(fields_b)), (
        f"{name}: the unrepaired leg matched the reference, so this case cannot tell a "
        f"working repair from a missing one")


# ---------------------------------------------------------------------------
# The two constitutive halves the repair had never been driven against
# ---------------------------------------------------------------------------
#
# Everything above runs on REAL storage with no susceptibility, which is the one
# configuration `metal_kernels.fused_magnetic_pair` serves. Two more Metal families
# declared `CARRIES_DEPOSIT_REPAIR` on 2026-08-28, and each brings a constitutive half
# this module had never reconstructed:
#
#   * complex_fused_magnetic_pair -- complex64 storage on the B seam. `save`/`apply`
#     index and accumulate element-wise (deposit_repair.py:118-129, :146-168), so one
#     complex array per component needs no code change; that is a reason to expect the
#     measurement to pass, not a substitute for making it.
#   * fused_dispersive_pair -- a live pole on the D seam. `apply` recomputes
#     `displacement_minus_polarization(c) * inverse_epsilon_for(c)` (:150-152), which
#     IS the dispersive drive; `repairable` refuses only an off-diagonal chi1inv and an
#     instantaneous chi2/chi3 (:98-108), so a pole was never refused -- merely never
#     measured.
#
# Same protocol as above, null control included: a case whose unrepaired leg agrees
# with the reference cannot tell a working repair from a missing one.

COMPLEX_CASES = [
    ("complex magnetic point", "B", "Hz", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("complex magnetic point, 24 steps", "B", "Hz", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 24),
    ("complex magnetic line", "B", "Hx", (0.0, 0.0, 0.0), (0.0, 0.6, 0.0), 6),
    ("complex magnetic in the PML shell", "B", "Hz", (0.5, 0.5, 0.0), (0.0, 0.0, 0.0), 6),
]

DISPERSIVE_CASES = [
    ("dispersive electric point", "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("dispersive electric point, 24 steps", "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 24),
    ("dispersive electric line", "D", "Ez", (0.0, 0.0, 0.0), (0.6, 0.0, 0.0), 6),
    ("dispersive electric in the PML shell", "D", "Ez", (0.5, 0.5, 0.0), (0.0, 0.0, 0.0), 6),
]

_EXTRA = ([(case, {"complex_storage": True}) for case in COMPLEX_CASES]
          + [(case, {"dispersive": True}) for case in DISPERSIVE_CASES])
_EXTRA_IDS = [case[0] for case, _build_kwargs in _EXTRA]


def test_the_dispersive_cases_are_not_the_ordinary_case_wearing_a_label():
    """THE PRECONDITION, without which every dispersive case below is vacuous.

    `displacement_minus_polarization` hands back the D ARRAY ITSELF when no
    susceptibility contributes (fields.py:1079-1082). If the pole were not live, or not
    driving Ez, or its P left at zero, the repair's electric branch would be recomputing
    `D * inv_eps` -- the ordinary constitutive -- and the four cases would be re-running
    a configuration already covered above while reading as new coverage.
    """
    _grid, fields, _pml = _build(dispersive=True)
    assert fields.has_polarizations
    state = fields.polarizations[0]
    assert state.drives("Ez"), state.driven()
    drive = fields.displacement_minus_polarization("Ez")
    assert drive is not fields.Dz, "D - sum P aliased D; no polarization contributes"
    assert float(numpy.abs(drive - fields.Dz).max()) > 0.0, (
        "D - sum P equals D bit for bit, so the dispersive branch is not under test")


@pytest.mark.parametrize("case,build", _EXTRA, ids=_EXTRA_IDS)
def test_the_new_constitutive_halves_are_byte_identical_to_the_driver_order(case, build):
    name, pair, component, center, size, steps = case
    grid_a, fields_a, pml_a = _build(**build)
    grid_b, fields_b, pml_b = _build(**build)
    assert not _differing(_words(fields_a), _words(fields_b)), "preconditions differ"
    ok, why = deposit_repair.repairable(fields_a, pair)
    assert ok, f"{name}: repairable refuses this configuration: {why}"
    src_a = [_make_source(grid_a, pair, component, center, size)]
    src_b = [_make_source(grid_b, pair, component, center, size)]
    assert src_a[0]._n_source_points, f"{name} deposits nothing and cannot discriminate"
    dt = grid_a.dt
    for step in range(steps):
        _reference(fields_a, pml_a, src_a, step * dt, dt, pair)
        _fused_then_repair(fields_b, pml_b, src_b, step * dt, dt, pair)
    assert not _differing(_words(fields_a), _words(fields_b)), name


@pytest.mark.parametrize("case,build", _EXTRA, ids=_EXTRA_IDS)
def test_withholding_the_repair_diverges_on_the_new_constitutive_halves(case, build):
    """The null control for the two configurations above."""
    name, pair, component, center, size, steps = case
    grid_a, fields_a, pml_a = _build(**build)
    grid_b, fields_b, pml_b = _build(**build)
    src_a = [_make_source(grid_a, pair, component, center, size)]
    src_b = [_make_source(grid_b, pair, component, center, size)]
    dt = grid_a.dt
    for step in range(steps):
        _reference(fields_a, pml_a, src_a, step * dt, dt, pair)
        _fused_then_repair(fields_b, pml_b, src_b, step * dt, dt, pair, repair=False)
    assert _differing(_words(fields_a), _words(fields_b)), (
        f"{name}: the unrepaired leg matched the reference, so this case cannot tell a "
        f"working repair from a missing one")


# ---------------------------------------------------------------------------
# THE FOLD BOUNDARY, MEASURED IN BOTH DIRECTIONS
# ---------------------------------------------------------------------------
#
# Four fused families (two Metal, two Triton) were ROUTED on 2026-08-27 holding
# CARRIES_DEPOSIT_REPAIR at False, on the reasoning that `apply` wrote only the
# deposit INDEX while the driver runs `fill_symmetry_bc_D` and
# `fill_folded_far_ghosts_D` AFTER the injection (driver.py:3308-3311), and the near
# fill writes cell 0 from cell 2 (stepping.py:1450-1451, MIRROR_SOURCE_INDEX = 2).
#
# THE ARGUMENT WAS RIGHT AND THE BOUNDARY IS NOW RETIRED FOR THE METAL PAIR.
# `deposit_repair.repair_cells` extends the saved and restored set to the CLOSURE of
# the cells those two fills image each deposit point into, and the two Metal folded
# families flipped the flag in the same edit as that closure. What decides whether
# this file still measures anything is that BOTH directions run here:
#
#   * with the shipped closure, a deposit ON the imaged row is carried byte-exactly;
#   * with the closure REMOVED -- `repair_cells` cut back to the deposit index, which
#     is exactly the pre-2026-08-28 shape -- the same case DIVERGES, and diverges at
#     the image cell rather than somewhere unrelated.
#
# Without the second, the first would be consistent with a fold the harness never
# built; without the first, the flip is unmeasured. The mid-domain control stays for
# the third reading: folding alone never defeated the repair, only the image did.


def _folded_build(cell=(1.6, 2.0, 0.0), resolution=10.0, seed=20260828):
    """A grid folded on Y, sized so `_mirror_row_y` lands on MIRROR_SOURCE_INDEX."""
    from .grid import Mirror

    grid = Grid(resolution=resolution, cell_size=cell, dimensions=2, courant=0.35,
                symmetry=(Mirror("Y", 1),))
    fields = Fields(grid=grid, force_complex_fields=False)
    pml = PML(grid=grid, thickness=((2, 2), (0, 2), (0, 0)))
    fields.enable_pml_storage()
    generator = numpy.random.default_rng(seed)
    for name in SEEDED:
        array = getattr(fields, name, None)
        if array is not None:
            _seed(array, generator)
    return grid, fields, pml


def _folded_source(grid, y, component="Ez"):
    return GaussianPulsedSource(grid=grid, component=component, center=(0.0, y, 0.0),
                                size=(0.0, 0.0, 0.0), frequency=1.0, fwidth=0.2,
                                amplitude=1.0)


#: y that puts the deposit on the row the near fill READS FROM, and one that does not.
#: Asserted below rather than trusted — a geometry change that moved either would make
#: the whole case vacuous in the direction that reads as success.
MIRROR_ROW_Y = 0.10
MID_DOMAIN_Y = 0.50

#: The SAME grid's other fill. Y is folded PERIODIC here, so `fill_folded_far_ghosts_D`
#: is live too and images the LAST stored slot from `stepping._far_reflect_rows`, for
#: the shift-1 components only. `Ey` at this y lands on that reflect row; `Ez` at
#: :data:`FAR_ROW_SHIFT0_Y` lands on the SAME row with the other Yee shift and is
#: therefore imaged by nothing — which is what says the closure is keyed on the
#: component's sub-lattice rather than on the row number.
FAR_ROW_Y = 0.95
FAR_ROW_SHIFT0_Y = 0.90


def _folded_fused_then_repair(fields, pml, sources, when, dt, repair=True):
    """A FOLDED product's launch, then the repair.

    NOT ``_fused_then_repair``: that one models an UNFOLDED product, whose REPLACES
    carries ``step_D``, ``zero_metal_D`` and ``update_E`` only. A folded product also
    carries both fills inside the launch — ``folded_fused_pair.REPLACES`` is
    ``('step_D', 'fill_D', 'zero_metal_D', 'fill_folded_far_ghosts_D', 'update_E')`` —
    and leaving them out would diverge everywhere for a reason that has nothing to do
    with the deposit, which is exactly what the mid-domain control is here to catch.

    THE FILLS RUN BEFORE THE INJECTION, and that is not a modelling choice: there is no
    injection inside a launch. It is the whole shape of the problem.
    """
    saved = deposit_repair.save(fields, sources, "D") if repair else None
    stepping.step_D(fields, pml)
    stepping.zero_metal_D(fields)
    stepping.fill_symmetry_bc_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)
    for source in sources:
        source.inject(fields, when + 0.5 * dt)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    if repair:
        deposit_repair.apply(fields, pml, sources, "D", saved)


def _run_folded(y, steps=6, carry_images=True, monkeypatch=None, component="Ez"):
    """The folded seam, repaired, against the driver's own order.

    ``carry_images=False`` cuts `deposit_repair.repair_cells` back to the deposit
    index -- the shape this module shipped before 2026-08-28 -- so the same case can
    be run with and without the one thing the flip rests on. It is not a second
    implementation of the repair: `save` and `apply` are the shipped functions and
    only the SET they are handed changes, which is where the whole defect lived.
    """
    if not carry_images:
        assert monkeypatch is not None, "the point-only leg needs a patch fixture"
        monkeypatch.setattr(deposit_repair, "repair_cells",
                            lambda fields, target, index: index)
    grid_a, fields_a, pml_a = _folded_build()
    grid_b, fields_b, pml_b = _folded_build()
    src_a = [_folded_source(grid_a, y, component)]
    src_b = [_folded_source(grid_b, y, component)]
    rows = sorted(set(numpy.atleast_1d(src_a[0]._point_iy).tolist()))
    dt = grid_a.dt
    for step in range(steps):
        _reference(fields_a, pml_a, src_a, step * dt, dt, "D")
        _folded_fused_then_repair(fields_b, pml_b, src_b, step * dt, dt)
    return rows, _differing(_words(fields_a), _words(fields_b))


def _folded_cells(y, component="Ez", steps=1):
    """Which CELLS of one component the two legs disagree on, as index triples."""
    grid_a, fields_a, pml_a = _folded_build()
    grid_b, fields_b, pml_b = _folded_build()
    src_a = [_folded_source(grid_a, y, component)]
    src_b = [_folded_source(grid_b, y, component)]
    dt = grid_a.dt
    for step in range(steps):
        _reference(fields_a, pml_a, src_a, step * dt, dt, "D")
        _folded_fused_then_repair(fields_b, pml_b, src_b, step * dt, dt)
    left = numpy.ascontiguousarray(getattr(fields_a, component))
    right = numpy.ascontiguousarray(getattr(fields_b, component))
    shape = left.shape
    mask = (left.reshape(-1).view(numpy.uint8).reshape(left.size, left.itemsize)
            != right.reshape(-1).view(numpy.uint8).reshape(right.size,
                                                           right.itemsize)).any(axis=1)
    return [tuple(int(v) for v in cell)
            for cell in numpy.argwhere(mask.reshape(shape))]


def test_the_fold_is_really_folded_and_the_two_deposits_land_where_claimed():
    """The preconditions. Without them the case below cannot fail for its own reason."""
    from .triton_kernels.symmetry import MIRROR_SOURCE_INDEX

    grid, fields, _pml = _folded_build()
    assert grid.is_mirrored(1), "the grid is not folded; the case measures nothing"
    assert deposit_repair.repairable(fields, "D")[0], (
        "the repair refuses this configuration for some OTHER reason, so a divergence "
        "below would not be the mirror image")
    on_row = sorted(set(numpy.atleast_1d(
        _folded_source(grid, MIRROR_ROW_Y)._point_iy).tolist()))
    off_row = sorted(set(numpy.atleast_1d(
        _folded_source(grid, MID_DOMAIN_Y)._point_iy).tolist()))
    assert on_row == [MIRROR_SOURCE_INDEX], on_row
    assert MIRROR_SOURCE_INDEX not in off_row, off_row


def test_a_deposit_on_the_mirror_row_IS_carried_once_the_images_are_repaired():
    """THE MEASUREMENT BEHIND THE TWO METAL FOLDED FAMILIES' ``= True``.

    The repair restores the deposit index AND every cell the driver's post-injection
    fills image it into, so cell 0 -- rebuilt from cell 2 after the injection -- gets
    its constitutive result recomputed from the final D rather than keeping the one
    the fused launch computed against the pre-injection field.
    """
    rows, differing = _run_folded(MIRROR_ROW_Y)
    assert not differing, (rows, differing)


def test_the_image_closure_is_what_carries_it_and_a_point_repair_still_does_not(
        monkeypatch):
    """THE OTHER DIRECTION, on the same case, with only the SET changed.

    This is the test that used to assert the divergence outright. It still asserts it
    -- of the point-only repair -- which is what keeps the case above from reading as
    "a folded grid nothing disturbs". The divergence must also be AT THE IMAGE: a
    point repair that broke the run somewhere else would satisfy a bare `differing`.
    """
    rows, differing = _run_folded(MIRROR_ROW_Y, carry_images=False,
                                  monkeypatch=monkeypatch)
    assert differing, (
        f"a deposit on row {rows} was carried exactly by a POINT repair. The closure "
        f"in `repair_cells` would then be carrying nothing, and the two Metal folded "
        f"families' CARRIES_DEPOSIT_REPAIR would be resting on it for no reason")
    monkeypatch.setattr(deposit_repair, "repair_cells",
                        lambda fields, target, index: index)
    cells = _folded_cells(MIRROR_ROW_Y)
    assert cells, "the point repair diverged on some other array, not on Ez"
    assert all(cell[1] == 0 for cell in cells), (
        f"the point repair's Ez divergence is at {cells}, not on the near fill's "
        f"destination row 0; this case is measuring something else")


def test_a_deposit_on_the_FAR_ghosts_reflect_row_is_carried_too():
    """THE FOLD'S OTHER FILL, on the same grid. Y is folded PERIODIC here, so
    `fill_folded_far_ghosts_D` images the LAST stored slot from the reflect row
    `stepping._far_reflect_rows` derives — a different row, a different destination
    and the opposite Yee shift from the near fill, and a closure that only inverted
    the near one would leave this case wrong."""
    rows, differing = _run_folded(FAR_ROW_Y, component="Ey")
    assert not differing, (rows, differing)


def test_a_point_repair_still_misses_the_far_ghost(monkeypatch):
    """The far half's own other direction, and its divergence must be at the ghost."""
    from .grid import Mirror  # noqa: F401 - the fold is built by `_folded_build`

    grid, _fields, _pml = _folded_build()
    ghost = int(grid.stored_cells(1)) - 1
    rows, differing = _run_folded(FAR_ROW_Y, carry_images=False,
                                  monkeypatch=monkeypatch, component="Ey")
    assert differing, (
        f"a deposit on the reflect row {rows} was carried exactly by a POINT repair")
    monkeypatch.setattr(deposit_repair, "repair_cells",
                        lambda fields, target, index: index)
    cells = _folded_cells(FAR_ROW_Y, component="Ey")
    assert cells and all(cell[1] == ghost for cell in cells), (cells, ghost)


def test_the_closure_is_keyed_on_the_YEE_SHIFT_and_not_on_the_row(monkeypatch):
    """THE CONTROL THAT SAYS THE RULE IS NOT "row 10 is special".

    `Ez` at :data:`FAR_ROW_SHIFT0_Y` deposits on the SAME stored row the far ghost
    reflects from, but `Dz`'s Yee shift on the folded axis is 0, so
    `_fill_folded_far_ghosts` skips it and no cell images that deposit. A POINT repair
    must therefore already be exact there — and if it were not, the passing far case
    above would be consistent with a closure that images every deposit it sees.
    """
    grid, _fields, _pml = _folded_build()
    reflect = stepping._far_reflect_rows(grid)[1]
    rows = sorted(set(numpy.atleast_1d(
        _folded_source(grid, FAR_ROW_SHIFT0_Y, "Ez")._point_iy).tolist()))
    assert rows == [reflect], (rows, reflect)
    _rows, differing = _run_folded(FAR_ROW_SHIFT0_Y, carry_images=False,
                                   monkeypatch=monkeypatch, component="Ez")
    assert not differing, differing


def test_a_deposit_off_the_mirror_row_on_the_same_folded_grid_is_carried_exactly():
    """THE CONTROL. Folding by itself does not defeat the repair — only the image does.

    Without this, the divergence above would be consistent with the harness simply
    being unable to step a folded grid, which would say nothing about mirror images.
    """
    rows, differing = _run_folded(MID_DOMAIN_Y)
    assert not differing, (rows, differing)


def test_an_offdiagonal_row_is_refused_by_name(monkeypatch):
    """The stencil case must be refused, not approximated."""
    _grid, fields, _pml = _build()
    monkeypatch.setattr(type(fields), "has_offdiagonal_epsilon",
                        property(lambda self: True))
    ok, reasons = deposit_repair.repairable(fields, "D")
    assert not ok and any("off-diagonal" in r for r in reasons), reasons


def test_a_folded_cylindrical_axis_is_refused_by_name(monkeypatch):
    """THE FOLD THE CLOSURE DOES NOT MODEL. `_mirror_phases` gives the cylindrical r
    axis a slot carrying (-1)^m rather than a mirror phase (stepping.py:2363) and its
    below-axis ghost is the r_to_minus_r image the shift helpers apply with
    per-component direction signs (:170-176), not the near fill's cell 0 <- cell 2. A
    closure derived from the near/far fills would answer for a map it never inverted,
    so the seam is refused rather than repaired."""
    _grid, fields, _pml = _folded_build()
    monkeypatch.setattr(type(fields.grid), "is_axis", lambda self, axis: axis == 0)
    ok, reasons = deposit_repair.repairable(fields, "D")
    assert not ok and any("cylindrical r = 0 axis" in r for r in reasons), reasons


def test_a_folded_axis_too_short_for_the_near_fills_source_row_is_refused(monkeypatch):
    """`stepping._mirror_source` raises below three stored cells, so the row the near
    fill images does not exist. A closure that compared against it would find no image
    and repair nothing exactly where the fill does the most, which is the silent wrong
    answer this module refuses instead."""
    _grid, fields, _pml = _folded_build()
    monkeypatch.setattr(type(fields.grid), "stored_cells",
                        lambda self, axis: 2 if axis == 1 else 4)
    ok, reasons = deposit_repair.repairable(fields, "D")
    assert not ok and any("does not exist" in r for r in reasons), reasons


def test_an_unfolded_grid_gains_no_fold_refusal():
    """The clauses above must be about the FOLD. On an unfolded grid neither fill runs
    at all, and a refusal there would be a regression on every product already
    carrying the repair."""
    _grid, fields, _pml = _build()
    assert deposit_repair.repairable(fields, "D") == (True, ())
    assert deposit_repair.repairable(fields, "B") == (True, ())


# ---------------------------------------------------------------------------
# The absorber the repair inverts, and the fail-open it used to have
# ---------------------------------------------------------------------------
#
# `apply` recomputes `field + kps*fresh - kms*fw_prev`, which is the SPLIT-FIELD
# constitutive recurrence and nothing else. `stepping._pml_is_active` sends a layer
# absorbing on none of its six faces down the PLAIN path instead -- `update_H` returns
# without touching H (stepping.py:944-945) and `update_E` writes E = constitutive with no
# f_w at all (:1019-1022) -- so on such a layer the repair inverts a recurrence that never
# ran. `repairable` used to take no layer and therefore answered (True, ()) there: a
# FAIL-CLOSED guard failing OPEN. Measured below rather than argued: with the clause
# bypassed the protocol leaves 23 words differing from the driver's own order.
#
# The shape of the fix matters as much as the case. An INACTIVE layer carries exactly the
# same coefficient SHAPES as an active one -- (14,1,1)/(1,14,1)/(1,1,1) either way, only
# the values differ -- so a check that looked only at shapes would not have caught this,
# and a check that named only `is_active` would admit the next unrecognised layer. Both
# directions are pinned: the named absorber clauses, and a layout clause that refuses any
# coefficient `apply` cannot broadcast the way the sub-step read it.


def _inactive_build(**kwargs):
    """The same triple with an absorber on no face. `PML.is_active` is False here."""
    return _build(thickness=0, **kwargs)


@pytest.mark.parametrize("pair", ["B", "D"])
def test_an_inactive_absorber_is_refused_by_name(pair):
    """THE FAIL-OPEN. Before the layer reached `repairable` this returned (True, ())."""
    _grid, fields, pml = _inactive_build()
    assert not pml.is_active, "this case needs a layer that absorbs nowhere"
    ok, reasons = deposit_repair.repairable(fields, pair, pml)
    assert not ok, (pair, "an inactive absorber was admitted")
    assert any("is_active False" in reason for reason in reasons), reasons


@pytest.mark.parametrize("pair", ["B", "D"])
def test_no_absorber_at_all_is_refused_by_name(pair):
    """``None`` is an ANSWER -- the plain path -- and is refused, not treated as unknown."""
    _grid, fields, _pml = _build()
    ok, reasons = deposit_repair.repairable(fields, pair, None)
    assert not ok and any("no PML layer" in reason for reason in reasons), reasons


@pytest.mark.parametrize("pair", ["B", "D"])
def test_the_inactive_absorber_the_guard_refuses_really_does_diverge(monkeypatch, pair):
    """THE NULL CONTROL FOR THE REFUSAL, and the mutation this test exists to catch.

    A refusal nothing measures is a refusal nobody can tell from taste. Bypassing the
    absorber clauses -- which is exactly what a regression to fail-open looks like --
    must produce a field the driver's own order did not, or this clause is protecting
    nothing. With the clauses live the same protocol refuses instead.
    """
    monkeypatch.setattr(deposit_repair, "_absorber_reasons", lambda *a, **k: ())
    grid_a, fields_a, pml_a = _inactive_build()
    _grid_b, fields_b, pml_b = _inactive_build()
    component = "Hz" if pair == "B" else "Ez"
    src_a = [_make_source(grid_a, pair, component, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))]
    src_b = [_make_source(fields_b.grid, pair, component, (0.0, 0.0, 0.0),
                          (0.0, 0.0, 0.0))]
    dt = grid_a.dt
    _reference(fields_a, pml_a, src_a, 0.0, dt, pair)
    _fused_then_repair(fields_b, pml_b, src_b, 0.0, dt, pair)
    differing = _differing(_words(fields_a), _words(fields_b))
    assert differing, (
        f"seam {pair}: the bypassed repair matched the driver on an inactive absorber, "
        f"so this case cannot tell the guard from its absence")

    monkeypatch.undo()
    _grid_c, fields_c, pml_c = _inactive_build()
    src_c = [_make_source(fields_c.grid, pair, component, (0.0, 0.0, 0.0),
                          (0.0, 0.0, 0.0))]
    with pytest.raises(deposit_repair.DepositNotRepairable):
        _fused_then_repair(fields_c, pml_c, src_c, 0.0, dt, pair)
    assert deposit_repair.repairable(fields_c, pair, pml_c)[0] is False


def test_apply_refuses_an_inactive_absorber_whatever_save_was_told():
    """THE BACKSTOP THAT NEEDS NO CALLER CHANGE.

    `apply` has always been handed the layer, and it is the only place the arithmetic
    lands, so it asserts the absorber clauses itself. A caller that reached `save`
    without naming the layer is refused here rather than writing a field nothing can
    put back.
    """
    grid, fields, pml = _inactive_build()
    source = _make_source(grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    saved = deposit_repair.save(fields, [source], "D")  # no layer named
    before = _words(fields)
    with pytest.raises(deposit_repair.DepositNotRepairable, match="is_active False"):
        deposit_repair.apply(fields, pml, [source], "D", saved)
    assert not _differing(before, _words(fields)), (
        "apply wrote before it refused; a partial repair is worse than none")


class _WrongLayout:
    """An ACTIVE layer whose coefficients all carry the y layout, on every axis.

    Not a hypothetical: `apply` calls ``xp.broadcast_to(kps, field.shape)``, and a
    (1,n,1) profile broadcasts into a (n,n,1) field without complaint -- so the repair
    would apply y-grading to an x-graded component and report success.
    """

    is_active = True

    def __init__(self, extent):
        self._value = numpy.ones((1, extent, 1), dtype=numpy.float32)

    def __getattr__(self, name):
        if name.startswith(("kps_", "kms_")):
            return self._value
        raise AttributeError(name)


class _NoCoefficients:
    is_active = True


@pytest.mark.parametrize("pair", ["B", "D"])
def test_an_unrecognised_coefficient_layout_is_refused_by_name(pair):
    """The clause is about the LAYOUT, not about `is_active`: this layer is active."""
    _grid, fields, _pml = _build()
    ok, reasons = deposit_repair.repairable(
        fields, pair, _WrongLayout(fields.Ez.shape[1]))
    assert not ok, "a coefficient apply cannot broadcast correctly was admitted"
    assert any("is not a coefficient layout this repair recognises" in reason
               for reason in reasons), reasons
    # NAMED, not merely counted: the refusal says which coefficient and which component.
    assert any("kps_x" in reason for reason in reasons), reasons


@pytest.mark.parametrize("pair", ["B", "D"])
def test_a_layer_whose_coefficients_cannot_be_read_is_refused_by_name(pair):
    """Ignorance is a refusal. An active layer that cannot produce kps/kms at all is
    named with the error it raised rather than admitted for lack of evidence."""
    ok, reasons = deposit_repair.repairable(_build()[1], pair, _NoCoefficients())
    assert not ok and any("could not be read from this layer" in reason
                          for reason in reasons), reasons


@pytest.mark.parametrize("pair", ["B", "D"])
@pytest.mark.parametrize("complex_storage,dispersive",
                         [(False, False), (True, False), (False, True)],
                         ids=["real", "complex", "dispersive"])
def test_the_layer_the_shipped_products_run_on_is_still_admitted(pair, complex_storage,
                                                                 dispersive):
    """THE REGRESSION CONTROL, and it is the reason the layout clause is per-axis.

    Every configuration the five Metal families declaring CARRIES_DEPOSIT_REPAIR serve
    carries an ACTIVE layer whose coefficients are the per-axis broadcast layout, and a
    clause that refused any of them would take rows off the board. The z coefficients of
    a 2-D grid are (1,1,1) -- legitimately ungraded, not unrecognised -- which is the
    case a blanket "must vary along its axis" rule would have broken.
    """
    _grid, fields, pml = _build(complex_storage=complex_storage, dispersive=dispersive)
    assert pml.is_active
    assert deposit_repair.repairable(fields, pair, pml) == (True, ())


def test_the_leading_plan_hands_save_the_layer_it_holds():
    """The wiring, pinned. `LeadingRepairPlan` is the last moment a configuration can be
    refused BEFORE the fused launch overwrites the field the repair would put back, and
    it can only refuse if it forwards the layer it already holds."""
    _grid, fields, pml = _inactive_build()
    source = _make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    class _Inner:
        ran = False

        def run(self, *args, **kwargs):
            type(self).ran = True

    inner = _Inner()
    plan = deposit_repair.LeadingRepairPlan(inner, fields, pml, [source], "D")
    with pytest.raises(deposit_repair.DepositNotRepairable, match="is_active False"):
        plan.run()
    assert not _Inner.ran, "the fused launch ran before the refusal"


def test_save_and_apply_are_handed_ONE_cell_set(monkeypatch):
    """The two halves may not compute the closure separately.

    `apply` needs ``field_old`` and ``fw_old`` at every image it restores, so a set
    recomputed at apply time is a set that can disagree with the one saved. This pins
    that `apply` reads the saved cells: a `repair_cells` that answers differently on
    its SECOND call must not change what `apply` writes.
    """
    _grid, fields, pml = _folded_build()
    source = _folded_source(fields.grid, MIRROR_ROW_Y)
    saved = deposit_repair.save(fields, [source], "D")
    cells = saved[("Ez", 0, "cells")]
    assert int(cells[0].size) > int(numpy.atleast_1d(source._point_ix).size), (
        "the saved set carries no image, so this case cannot discriminate")
    calls = []
    monkeypatch.setattr(deposit_repair, "repair_cells",
                        lambda *a: calls.append(a) or (_ for _ in ()).throw(
                            AssertionError("apply recomputed the cell set")))
    deposit_repair.apply(fields, pml, [source], "D", saved)
    assert not calls


def test_a_nonlinear_row_is_refused_by_name(monkeypatch):
    _grid, fields, _pml = _build()
    monkeypatch.setattr(type(fields), "has_nonlinearity", property(lambda self: True))
    ok, reasons = deposit_repair.repairable(fields, "D")
    assert not ok and any("chi2/chi3" in r for r in reasons), reasons


def test_save_refuses_rather_than_returning_an_empty_repair(monkeypatch):
    """A refusal that returned {} would be applied as a no-op and read as success."""
    _grid, fields, _pml = _build()
    monkeypatch.setattr(type(fields), "has_nonlinearity", property(lambda self: True))
    with pytest.raises(deposit_repair.DepositNotRepairable):
        deposit_repair.save(fields, [], "D")


def test_the_seam_table_mirrors_steppings_own_constitutive_tables():
    """`SEAMS` restates what `stepping` already knows; a restatement can go stale.

    The repair recomputes the constitutive result, so it must read the SAME targets,
    from the SAME source volumes, on the SAME Yee sub-lattice as the sub-step it is
    repairing. All three are stated in `stepping` and copied here. A component added or
    a sub-lattice flipped there would leave this table describing the old engine, and the
    byte comparisons would keep passing on every configuration the stale entry happens to
    match.
    """
    assert deposit_repair.SEAMS["B"]["targets"] == stepping.H_CONSTITUTIVE_TERMS
    assert deposit_repair.SEAMS["D"]["targets"] == stepping.E_CONSTITUTIVE_TERMS
    # The sub-lattice each half's coefficients come from, read off the call `update_H`
    # and `update_E` actually make rather than restated.
    import inspect
    for pair, function in (("B", stepping.update_H), ("D", stepping.update_E)):
        body = inspect.getsource(function)
        expected = deposit_repair.SEAMS[pair]["half_integer"]
        assert f"half_integer={expected}" in body, (
            f"update_{'H' if pair == 'B' else 'E'} no longer takes "
            f"half_integer={expected}; SEAMS[{pair!r}] describes the old engine")


def test_the_magnetic_branch_repairs_what_update_H_actually_computes():
    """`apply` uses B itself as the magnetic constitutive input, with no inverse
    permeability. That is correct only because this engine fixes mu = 1, which
    `update_H` states and relies on. If a permeability is ever introduced there, the
    repair must gain the same factor — and this is what makes that a failing test rather
    than a silent divergence on magnetic media."""
    import inspect
    body = inspect.getsource(stepping.update_H)
    assert "mu = 1" in body, (
        "update_H no longer documents mu = 1; deposit_repair.apply feeds B straight into "
        "the constitutive recurrence and would need the same inverse permeability")
    assert "inverse_mu" not in body and "inverse_permeability" not in body


# ---------------------------------------------------------------------------
# THE SECOND REPAIR: deposit_repair.PLAIN_PATH
# ---------------------------------------------------------------------------
#
# Everything above drives the SPLIT-FIELD repair, the inverse of
# `_apply_constitutive_pml`'s stateful recurrence. This half drives the other one:
# `update_E`'s plain branch with an inactive layer, which writes
# `E[...] = (D - sum P) * inv_eps` straight to storage (stepping.py:1019-1022).
#
# WHY IT NEEDED A SECOND REPAIR RATHER THAN A RELAXED FIRST ONE. The split-field
# repair recomputes `field_old + kps*fresh - kms*fw_prev` from state captured before
# the launch. On the plain branch none of those three exists: `f_w_E*` is not
# allocated at all, there are no kps/kms tables, and the previous field value is not
# read. `test_an_inactive_absorber_is_refused_by_name` and
# `test_no_absorber_at_all_is_refused_by_name` above are the measurements that say so
# and they are UNCHANGED -- the default `paths` is still `(SPLIT_FIELD_PATH,)`, so
# both refusals still fire, unedited, for every caller that has not declared the
# second path. What is new is a caller that CAN declare it.
#
# The cases below are the same shape as the ones above and are weighted the same way:
# every comparison is paired with a NULL CONTROL that withholds the repair and must
# diverge, because a case where both legs agree cannot tell a working repair from a
# missing one.


def _plain_build(cell=(1.2, 1.2, 0.0), resolution=12.0, seed=20260831,
                 poles=2, conductivity=None):
    """A NO-ABSORBER triple with stored E: the configuration PLAIN_PATH inverts.

    `thickness=0` gives a `PML` object whose `is_active` is False -- the layer the
    first repair refuses by name -- and `enable_field_storage` allocates E without
    `f_w`, which is what `update_E`'s plain branch writes into. Poles are registered
    so `displacement_minus_polarization` really forms `D - sum P` and the degenerate
    `D * inv_eps` case is not the only one measured.
    """
    dimensions = sum(1 for extent in cell if extent > 0.0)
    grid = Grid(resolution=resolution, cell_size=cell, dimensions=dimensions)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    for index in range(poles):
        # A DIFFERENT sigma per component, so the three components see different pole
        # SUBSETS in `poles_per_component` order -- the property the fused product's
        # NP0/NP1/NP2 constexprs specialise on.
        sigma = {"Ex": 0.3 + 0.05 * index, "Ey": 0.0 if index else 0.2,
                 "Ez": 0.25 + 0.05 * index}
        fields.polarizations.append(PolarizationState(
            Susceptibility(1.0 + 0.3 * index, 0.1, "lorentzian"), sigma, grid,
            numpy.float32))
    if conductivity is not None:
        volume = numpy.full(grid.shape, float(conductivity), dtype=numpy.float32)
        volume *= numpy.linspace(0.5, 1.5, grid.shape[0], dtype=numpy.float32)[:, None, None]
        fields.set_d_conductivity(volume)
    fields.enable_field_storage()
    pml = PML(grid=grid, thickness=0)
    generator = numpy.random.default_rng(seed)
    for name in SEEDED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        _seed(array, generator)
    for state in fields.polarizations:
        for attribute in ("P", "P_prev"):
            for array in (getattr(state, attribute, None) or {}).values():
                if array is not None:
                    _seed(array, generator)
    return grid, fields, pml


PLAIN = (deposit_repair.PLAIN_PATH,)


def _plain_reference(fields, pml, sources, when, dt):
    """The driver's order on the plain path (driver.py:3299-3314), verbatim."""
    stepping.step_D(fields, pml)
    for s in sources:
        s.inject(fields, when + 0.5 * dt)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    stepping.update_E(fields, pml)
    stepping.update_P(fields, pml)


class _PlainInner:
    """The fused product's launch, emulated with the shipped array functions.

    Stands in for the kernel at the FIRST consult only. What is under test here is the
    REPAIR, which is the shipped component either way; the device gate is what closes
    the difference between three array calls and one launch.
    """

    def __init__(self, fields, pml):
        self._fields, self._pml = fields, pml
        self.ran = 0

    def run(self, *_args, **_kwargs):
        stepping.step_D(self._fields, self._pml)
        stepping.zero_metal_D(self._fields)   # the kernel's inline ZM_*
        stepping.update_E(self._fields, self._pml)
        self.ran += 1


def _plain_fused_then_repair(fields, pml, sources, when, dt, repair=True):
    """Fused launch at the step_D consult, repair at the update_E consult.

    DRIVES THE SHIPPED PLAN OBJECTS, not a copy of their logic: these are the exact
    pair `launch._install_fused_pair` puts in a fused pair's two slots, with the
    `repair_paths` a product declaring `PLAIN_PATH` hands it.
    """
    inner = _PlainInner(fields, pml)
    leading = trailing = None
    if repair:
        leading = deposit_repair.LeadingRepairPlan(inner, fields, pml, sources, "D",
                                                   PLAIN)
        trailing = deposit_repair.TrailingRepairPlan("update_E", leading, fields, pml)
        leading.run()
    else:
        inner.run()
    for s in sources:
        s.inject(fields, when + 0.5 * dt)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    if repair:
        trailing.run()
    stepping.update_P(fields, pml)
    return 0 if leading is None else leading.repairs


#: name, poles, conductivity, component, center, size, steps.
PLAIN_CASES = [
    ("point, two poles", 2, None, "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("point, two poles, 24 steps", 2, None, "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 24),
    ("line, two poles", 2, None, "Ez", (0.0, 0.0, 0.0), (0.6, 0.0, 0.0), 6),
    # The component whose pole SUBSET is smaller than the others': `_plain_build` gives
    # Ey one pole where Ex and Ez get two, so a repair that used one component's chain
    # for all three is caught here rather than nowhere.
    ("line on the one-pole component", 2, None, "Ey", (0.0, 0.0, 0.0), (0.0, 0.6, 0.0), 6),
    # THE DEGENERATE CASE, admitted on purpose: with no pole driving anything
    # `displacement_minus_polarization` returns the D array ITSELF (fields.py:1097-1098)
    # and the repair stores `D * inv_eps`. Nothing else covers a stored-E run with no
    # poles, and the aliasing is exactly the shape a repair that mutated its input
    # would corrupt.
    ("point, no poles at all", 0, None, "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    # A MATERIAL CONDUCTIVITY, which is what `mp.Absorber` installs and what two of the
    # three corpus rows carry. It changes the CURL (stepping._apply_conductive_update)
    # and the INJECTION (driver._inject_electric_through_conductivity, :3305) and
    # leaves `update_E` alone -- so this case is about the repair reading a D the
    # driver rescaled, not about a different constitutive.
    ("point, conductive", 2, 0.20, "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 6),
    ("line, conductive", 2, 0.20, "Ex", (0.0, 0.0, 0.0), (0.6, 0.0, 0.0), 6),
]


@pytest.mark.parametrize("name,poles,conductivity,component,center,size,steps",
                         PLAIN_CASES,
                         ids=[case[0] for case in PLAIN_CASES])
def test_the_plain_repair_is_byte_identical_to_the_driver_order(
        name, poles, conductivity, component, center, size, steps):
    """The measurement the second repair stands on, per COMPLETE D->E step."""
    grid_a, fields_a, pml_a = _plain_build(poles=poles, conductivity=conductivity)
    _grid_b, fields_b, pml_b = _plain_build(poles=poles, conductivity=conductivity)
    assert not stepping._pml_is_active(pml_a), "this case needs the PLAIN branch"
    assert fields_a.stores_E, "this case needs a STORED E for update_E to write"
    assert getattr(fields_a, "f_w_Ex", None) is None, (
        "f_w is allocated, so this is not the plain branch and the split-field repair "
        "would have been the right one")
    assert not _differing(_words(fields_a), _words(fields_b)), "seeds disagree"

    src_a = [_make_source(grid_a, "D", component, center, size)]
    src_b = [_make_source(fields_b.grid, "D", component, center, size)]
    dt = grid_a.dt
    repaired = 0
    for step in range(steps):
        _plain_reference(fields_a, pml_a, src_a, step * dt, dt)
        repaired += _plain_fused_then_repair(fields_b, pml_b, src_b, step * dt, dt)
    assert repaired, ("the repair touched no cell: a leg that repairs nothing and "
                      "still matches is measuring the reference against itself")
    assert not _differing(_words(fields_a), _words(fields_b))


@pytest.mark.parametrize("name,poles,conductivity,component,center,size,steps",
                         PLAIN_CASES,
                         ids=[case[0] for case in PLAIN_CASES])
def test_withholding_the_plain_repair_diverges(
        name, poles, conductivity, component, center, size, steps):
    """THE NULL CONTROL. Every case above must be able to tell the repair from its
    absence, or its pass is worth nothing."""
    grid_a, fields_a, pml_a = _plain_build(poles=poles, conductivity=conductivity)
    _grid_b, fields_b, pml_b = _plain_build(poles=poles, conductivity=conductivity)
    src_a = [_make_source(grid_a, "D", component, center, size)]
    src_b = [_make_source(fields_b.grid, "D", component, center, size)]
    dt = grid_a.dt
    for step in range(steps):
        _plain_reference(fields_a, pml_a, src_a, step * dt, dt)
        _plain_fused_then_repair(fields_b, pml_b, src_b, step * dt, dt, repair=False)
    assert _differing(_words(fields_a), _words(fields_b))


def test_the_split_field_repair_is_the_wrong_one_here_and_says_so():
    """The first repair, offered this configuration, refuses BY NAME and does not run.

    This is the fact the second path exists for, kept as a measurement rather than as
    a memory: with the default `paths` the plain configuration is still refused with
    the two texts the campaign has been citing, and `save` raises before writing.
    """
    _grid, fields, pml = _plain_build()
    ok, reasons = deposit_repair.repairable(fields, "D", pml)
    assert not ok
    assert any("is_active False" in reason for reason in reasons), reasons
    assert any("f_w_Ex is not allocated" in reason for reason in reasons), reasons
    ok, reasons = deposit_repair.repairable(fields, "D", None)
    assert not ok and any("no PML layer" in reason for reason in reasons), reasons


def test_the_plain_path_admits_it_and_names_what_it_refuses():
    """The mirror clauses, both directions."""
    _grid, fields, pml = _plain_build()
    assert deposit_repair.repairable(fields, "D", pml, paths=PLAIN) == (True, ())
    assert deposit_repair.repairable(fields, "D", None, paths=PLAIN) == (True, ())

    # THE MAGNETIC SEAM IS REFUSED, and this is the clause that keeps a vacuous repair
    # from reporting success: with an inactive layer `update_H` returns without touching
    # H, so there is no stored value to put back.
    ok, reasons = deposit_repair.repairable(fields, "B", pml, paths=PLAIN)
    assert not ok and any("update_H returns without touching H" in r for r in reasons)

    # AN ACTIVE LAYER, offered to the plain path: refused, naming the split-field
    # recurrence it would have dropped.
    _g, active_fields, active_pml = _build()
    assert active_pml.is_active
    ok, reasons = deposit_repair.repairable(active_fields, "D", active_pml, paths=PLAIN)
    assert not ok and any("SPLIT-FIELD" in reason for reason in reasons), reasons

    # NO LAYER AT ALL, offered to a plain-only caller: refused, because which
    # recurrence ran cannot be established without one. This is the clause that stops
    # the second path from admitting on ignorance the way the first one may not.
    ok, reasons = deposit_repair.repairable(fields, "D", paths=PLAIN)
    assert not ok and any("no layer was supplied" in reason for reason in reasons)


def test_a_stored_E_that_update_E_would_not_write_is_refused_by_name():
    """`stores_E` False means `update_E` returns at stepping.py:983-984 without writing.

    A repair there would store a value the sub-step never computed into an array
    `Fields.get_E` does not serve, which is a silent wrong answer rather than a loud
    one -- so it is a named refusal.
    """
    _grid, fields, pml = _plain_build(poles=0)
    object.__setattr__(fields, "_stored_E", False)
    assert not fields.stores_E
    ok, reasons = deposit_repair.repairable(fields, "D", pml, paths=PLAIN)
    assert not ok and any("stores_E is False" in reason for reason in reasons), reasons


def test_the_plain_path_declaration_is_checked_rather_than_trusted():
    """A path this module cannot perform is a refusal, not a silent no-op."""
    _grid, fields, pml = _plain_build()
    ok, reasons = deposit_repair.repairable(fields, "D", pml, paths=("no_such_path",))
    assert not ok and any("unknown repair path" in reason for reason in reasons)
    ok, reasons = deposit_repair.repairable(fields, "D", pml, paths=())
    assert not ok and any("declared no repair path" in reason for reason in reasons)


def test_the_plain_save_carries_the_cell_set_and_no_state():
    """The plain branch reads no previous value, so an empty state is the honest record.

    Pinned in BOTH directions: the cells must be there (the two halves may not compute
    the closure separately) and the field/fw snapshots must NOT be, because capturing
    state a repair never reads would say the recurrence is stateful when it is not.
    """
    _grid, fields, pml = _plain_build()
    source = _make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    saved = deposit_repair.save(fields, [source], "D", pml, paths=PLAIN)
    assert saved[("Ez", 0, "cells")][0].size
    assert ("Ez", 0, "field") not in saved and ("Ez", 0, "fw") not in saved
    # The split-field save on the SAME shape of call does carry both, which is what
    # makes the absence above a property of the path rather than of this fixture.
    _g, active_fields, active_pml = _build()
    active_source = _make_source(active_fields.grid, "D", "Ez", (0.0, 0.0, 0.0),
                                 (0.0, 0.0, 0.0))
    split = deposit_repair.save(active_fields, [active_source], "D", active_pml)
    assert ("Ez", 0, "field") in split and ("Ez", 0, "fw") in split


def test_apply_reads_the_path_from_the_state_rather_than_being_told_twice():
    """`save` stamps the path; `apply` routes on the stamp.

    The two halves cannot be told different things about which recurrence the saved
    state describes -- the same discipline that keeps the cell set travelling with it.
    A dict with no stamp is the split-field default, which is what every caller
    predating the second repair produces.
    """
    _grid, fields, pml = _plain_build()
    source = _make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    saved = deposit_repair.save(fields, [source], "D", pml, paths=PLAIN)
    assert saved[deposit_repair._PATH_KEY] == deposit_repair.PLAIN_PATH
    assert deposit_repair.apply(fields, pml, [source], "D", saved) > 0

    # ...and the same state with the stamp removed routes to the SPLIT-FIELD repair,
    # which refuses this layer by name rather than writing.
    stripped = {k: v for k, v in saved.items() if k != deposit_repair._PATH_KEY}
    before = _words(fields)
    with pytest.raises(deposit_repair.DepositNotRepairable, match="is_active False"):
        deposit_repair.apply(fields, pml, [source], "D", stripped)
    assert not _differing(before, _words(fields)), (
        "apply wrote before it refused; a partial repair is worse than none")


def test_the_plain_apply_refuses_an_active_layer_whatever_save_was_told():
    """THE BACKSTOP, on the second path as on the first.

    `_apply_plain` is the only place the plain arithmetic lands, so it asserts its own
    clauses whatever produced the dict it was handed. A saved set built against an
    inactive layer and applied against an active one is refused, not written.
    """
    _grid, fields, pml = _plain_build()
    source = _make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    saved = deposit_repair.save(fields, [source], "D", pml, paths=PLAIN)
    _g, active_fields, active_pml = _build()
    before = _words(active_fields)
    with pytest.raises(deposit_repair.DepositNotRepairable, match="SPLIT-FIELD"):
        deposit_repair.apply(active_fields, active_pml, [source], "D", saved)
    assert not _differing(before, _words(active_fields))


def test_the_leading_plan_refuses_the_plain_configuration_it_cannot_carry():
    """The wiring, pinned on the second path: a product that declared the WRONG path
    is refused BEFORE its fused launch overwrites the field the repair would put back.
    """
    _grid, fields, pml = _plain_build()
    source = _make_source(fields.grid, "D", "Ez", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    inner = _PlainInner(fields, pml)
    # The DEFAULT declaration on a plain configuration: the split-field repair, which
    # this layer is not.
    plan = deposit_repair.LeadingRepairPlan(inner, fields, pml, [source], "D")
    with pytest.raises(deposit_repair.DepositNotRepairable, match="is_active False"):
        plan.run()
    assert inner.ran == 0, "the fused launch ran before the refusal"
    # ...and the declaration that matches the layer runs.
    plan = deposit_repair.LeadingRepairPlan(inner, fields, pml, [source], "D", PLAIN)
    plan.run()
    assert inner.ran == 1


def test_the_shipped_installer_threads_the_declared_paths():
    """`launch._install_fused_pair` is the ONE constructor of the two repair plans, so
    a product declaring PLAIN_PATH has to be able to say so THROUGH it. Read off the
    installer's own signature and its call, rather than from intent."""
    import ast
    import inspect as _inspect
    import pathlib as _pathlib

    from .triton_kernels import launch as triton_launch

    signature = _inspect.signature(triton_launch._install_fused_pair)
    assert signature.parameters["repair_paths"].default == (
        deposit_repair.SPLIT_FIELD_PATH,), (
        "the default must be the repair every caller written before PLAIN_PATH ships")
    text = _pathlib.Path(triton_launch.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and getattr(node.func, "attr", None) == "LeadingRepairPlan"]
    assert len(calls) == 1, calls
    assert len(calls[0].args) == 6, (
        "the installer must hand LeadingRepairPlan the declared paths as its sixth "
        "argument; without it a plain-path product silently gets the split-field one")


def test_the_plain_path_serves_the_seam_the_product_that_needs_it_declares():
    """The product and the repair agree about which path, read from both sides."""
    from .triton_kernels import no_pml_fused_electric_pair as product

    assert product.CARRIES_DEPOSIT_REPAIR is True
    assert product.REPAIR_PATHS == (deposit_repair.PLAIN_PATH,)
    assert deposit_repair.PLAIN_PATH_SEAMS == ("D",)
