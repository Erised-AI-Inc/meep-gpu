"""The bulk chi1inv read is taken, reads exactly MEEP's own words, and falls back exactly.

``_lift_epsilon_structured`` copies the stored ``chi1inv`` words out of MEEP's chunks and
reads the rest one point at a time through ``fields::get_chi1inv``. It falls back to the
per-point read whenever its bulk read refuses or its sampled self-check disagrees, and
the fallback is exact by construction. That makes a defect in the bulk read SILENT: the
lift still installs the right permittivity, only slower, and an installed ``1/x`` can
hide a one-ulp read error, so no physics test can see it. These tests pin it directly:

* the READ words of every component (the diagonal and both off-diagonal entries, as
  uint64) equal ``fields::get_chi1inv`` at every point of the lattice the lift reads,
  and the installed arrays equal the per-point path's bit for bit;
* the bulk path is the one taken (no refusal, no restart, points read in bulk on every
  component except the named per-point-only shapes, whose reason names the shape);
* the self-check runs: a one-ulp word planted at a copied block's corner restarts the
  read with a warning, the restart installs exactly the per-point read, and the progress
  sequence the caller sees never goes back;
* a MEEP build outside ``BULK_CHI1INV_BUILDS`` reads every point one at a time and says
  which build it is;
* a lift prints its flushed progress lines on stderr by default and none at
  ``mp.verbosity(0)``.

One small cell per ownership situation the bulk read must reproduce: a 3-D PML cell with
an off-diagonal tensor over several chunks, metallic walls behind a non-vacuum
background (unowned wall planes reading MEEP's 1.0 default), a mirror fold on a periodic
axis (the far ghost slot), a cylindrical cell (the r = 0 ownership case), a dispersive
background, and a cylindrical cell with no z extent (the named per-point-only shape).
The bulk-path tests run on a build ``BULK_CHI1INV_BUILDS`` lists and skip by a declared
resource elsewhere, where the lift never takes that path.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import numpy
import pytest

from conftest import requires_resource_skip

from meep_gpu import from_meep
from meep_gpu.driver import FdtdDriver

#: The shape the bulk path owns no point of, by component: MEEP builds a cylindrical cell
#: with no z extent one pixel thick, and Er and Ep sit on that one z plane.
FLAT_CYLINDER_SHAPE = "a cylindrical cell one pixel thick in z"


def _meep():
    try:
        import meep as mp  # noqa: PLC0415
    except ImportError:
        requires_resource_skip("meep", "the lift reads an mp.Simulation")
    mp.verbosity(0)
    return mp


def _certified_meep():
    """MEEP, on a build the lift takes the bulk read on; a declared skip elsewhere."""
    mp = _meep()
    stamp = from_meep._chi1inv_build_stamp(from_meep._meep_build(mp))  # noqa: SLF001
    if stamp not in from_meep.BULK_CHI1INV_BUILDS:
        requires_resource_skip(
            "bulk_chi1inv_build",
            f"MEEP {stamp} is not in BULK_CHI1INV_BUILDS, so its lifts read every point one at "
            f"a time (test_an_unlisted_build_reads_every_point_one_at_a_time covers that path)")
    return mp


def _source(mp, component, center):
    return [mp.Source(mp.GaussianSource(frequency=0.5, fwidth=0.4), component=component, center=center)]


def _pml_offdiag_3d(mp):
    geometry = [mp.Sphere(radius=0.45, center=mp.Vector3(0.2, -0.1, 0.05), material=mp.Medium(epsilon=4.0)),
                mp.Block(size=mp.Vector3(0.9, 0.35, 0.5), center=mp.Vector3(-0.3, 0.3, -0.2),
                         e1=mp.Vector3(1, 1, 0), e2=mp.Vector3(-1, 1, 0), e3=mp.Vector3(0, 0, 1),
                         material=mp.Medium(epsilon=2.25))]
    return mp.Simulation(cell_size=mp.Vector3(2.2, 2.0, 1.8), resolution=10, geometry=geometry,
                         boundary_layers=[mp.PML(0.3)], num_chunks=4,
                         sources=_source(mp, mp.Ez, mp.Vector3(0.11, 0.07, 0.03)))


def _walls_background_3d(mp):
    geometry = [mp.Block(size=mp.Vector3(0.7, 0.5, 0.6), center=mp.Vector3(0.1, 0, 0),
                         material=mp.Medium(epsilon=12.0))]
    return mp.Simulation(cell_size=mp.Vector3(1.6, 1.4, 1.2), resolution=10, geometry=geometry,
                         default_material=mp.Medium(epsilon=2.1), num_chunks=3,
                         sources=_source(mp, mp.Ez, mp.Vector3(0.11, 0.07, 0.03)))


def _folded_periodic_2d(mp):
    geometry = [mp.Block(size=mp.Vector3(0.9, 0.7, mp.inf), center=mp.Vector3(0.13, 0),
                         material=mp.Medium(epsilon=4.0))]
    return mp.Simulation(cell_size=mp.Vector3(3.0, 2.0, 0), resolution=10, geometry=geometry,
                         default_material=mp.Medium(epsilon=1.7),
                         boundary_layers=[mp.PML(0.5, direction=mp.X)], k_point=mp.Vector3(),
                         symmetries=[mp.Mirror(mp.Y)], sources=_source(mp, mp.Ez, mp.Vector3(-0.2, 0)))


def _cylindrical(mp):
    geometry = [mp.Block(size=mp.Vector3(0.4, mp.inf, 0.8), center=mp.Vector3(0.9, 0, 0.1),
                         material=mp.Medium(epsilon=4.0)),
                mp.Sphere(radius=0.3, center=mp.Vector3(0.0, 0, -0.5), material=mp.Medium(epsilon=2.0))]
    return mp.Simulation(cell_size=mp.Vector3(2.0, 0, 2.4), resolution=12, geometry=geometry,
                         dimensions=mp.CYLINDRICAL, m=1, boundary_layers=[mp.PML(0.3)], num_chunks=3,
                         sources=_source(mp, mp.Er, mp.Vector3(0.5, 0, 0.1)))


def _dispersive_background_2d(mp):
    lorentz = mp.Medium(epsilon=2.25, E_susceptibilities=[
        mp.LorentzianSusceptibility(frequency=0.4, gamma=0.02, sigma=1.1)])
    return mp.Simulation(cell_size=mp.Vector3(4, 3, 0), boundary_layers=[mp.PML(0.6)],
                         geometry=[mp.Block(size=mp.Vector3(1.1, 0.7, mp.inf), center=mp.Vector3(0.12, -0.07),
                                            material=mp.Medium(epsilon=6.0))],
                         default_material=lorentz, resolution=12, num_chunks=3,
                         sources=_source(mp, mp.Ez, mp.Vector3(0.11, 0.07)))


def _flat_cylinder(mp):
    """A cylindrical cell with no z extent (a ring resonator in r): the per-point-only shape."""
    return mp.Simulation(cell_size=mp.Vector3(8, 0, 0), resolution=10, dimensions=mp.CYLINDRICAL, m=3,
                         geometry=[mp.Block(center=mp.Vector3(1.5), size=mp.Vector3(1, mp.inf, mp.inf),
                                            material=mp.Medium(index=3.4))],
                         boundary_layers=[mp.PML(2)], sources=_source(mp, mp.Ez, mp.Vector3(1.1)))


CELLS = {
    "pml_offdiag_3d": _pml_offdiag_3d,
    "walls_background_3d": _walls_background_3d,
    "folded_periodic_2d": _folded_periodic_2d,
    "cylindrical": _cylindrical,
    "dispersive_background_2d": _dispersive_background_2d,
    "flat_cylinder": _flat_cylinder,
}
#: Components, by this engine's name, that each cell reads with no point in bulk.
PER_POINT_ONLY = {"flat_cylinder": {"Ex": FLAT_CYLINDER_SHAPE, "Ey": FLAT_CYLINDER_SHAPE}}


def _bits(array):
    return numpy.ascontiguousarray(array, dtype=numpy.float64).view(numpy.uint64)


class _Lift:
    """One lift with its read words, its installs and its record, captured as it ran.

    The read words are held at the self-check, which the bulk fill calls for every
    component on the bulk path: there ``values`` and ``coupling`` are the arrays the read
    loop then finishes in place, so holding them gives the final read words, the points
    read one at a time included.
    """

    def __init__(self, monkeypatch, mp, sim, *, bulk_read=None, plant=None, progress=None):
        self.reads = {}       # MEEP component -> (indices, own direction, values, coupling)
        self.installs = []    # (volumes, coupling rows) per set_epsilon_components call
        real_set = FdtdDriver.set_epsilon_components
        real_check = from_meep._chi1inv_bulk_self_check  # noqa: SLF001
        real_lift = from_meep._lift_epsilon_structured  # noqa: SLF001
        lift = self

        def set_spy(driver, volumes, chi1inv_offdiagonal=None, *args, **kwargs):
            lift.installs.append((
                {name: numpy.array(volume) for name, volume in volumes.items()},
                {(row, partner): numpy.array(values)
                 for row, partners in (chi1inv_offdiagonal or {}).items()
                 for partner, values in partners.items()}))
            return real_set(driver, volumes, chi1inv_offdiagonal, *args, **kwargs)

        def check_spy(table, component, own, others, indices, ivec, get_chi1inv, values, coupling,
                      blocks, pending, *args, **kwargs):
            lift.reads[int(component)] = (
                [numpy.asarray(list(map(int, axis)), dtype=numpy.int64) for axis in indices],
                int(own), values, coupling)
            if plant is not None and plant(int(component)) and blocks:
                # A one-ulp word at a copied block's low corner, where the self-check always reads.
                _chunk, select, _flat = blocks[0]
                corner = tuple(int(axis[0]) for axis in select)
                values[corner] = numpy.nextafter(values[corner], numpy.inf)
            return real_check(table, component, own, others, indices, ivec, get_chi1inv, values,
                              coupling, blocks, pending, *args, **kwargs)

        def lift_spy(*args, **kwargs):
            if bulk_read is not None and not kwargs.get("_restarted_by"):
                kwargs["_bulk_read"] = bulk_read
            return real_lift(*args, **kwargs)

        self.cylindrical = sim.dimensions == mp.CYLINDRICAL or bool(getattr(sim, "is_cylindrical", False))
        with monkeypatch.context() as patch:
            patch.setattr(FdtdDriver, "set_epsilon_components", set_spy)
            patch.setattr(from_meep, "_chi1inv_bulk_self_check", check_spy)
            patch.setattr(from_meep, "_lift_epsilon_structured", lift_spy)
            driver = from_meep.lift_simulation(sim, prefer_gpu=False, progress_cb=progress)
        try:
            self.record = driver.chi1inv_read
        finally:
            driver.close()
        self.sim = sim


def _reread_differences(mp, lift):
    """Per read array: (words that differ from ``fields::get_chi1inv`` as uint64, words)."""
    get = lift.sim.fields.get_chi1inv
    if lift.cylindrical:
        ivec = lambda ix, jy, kz: mp.iveccyl(ix, kz)  # noqa: E731
        directions = (mp.R, mp.P, mp.Z)
    else:
        ivec, directions = mp.ivec, (mp.X, mp.Y, mp.Z)
    differ = {}
    for component, (indices, own, values, coupling) in sorted(lift.reads.items()):
        reference = {own: numpy.empty(values.shape)}
        reference.update({int(d): numpy.empty(values.shape) for d in directions if int(d) != own})
        for i, ix in enumerate(indices[0]):
            for j, jy in enumerate(indices[1]):
                for k, kz in enumerate(indices[2]):
                    location = ivec(int(ix), int(jy), int(kz))
                    for direction, words in reference.items():
                        words[i, j, k] = complex(get(component, direction, location, 0)).real
        read = {own: values, **{int(d): coupling[d] for d in coupling}}
        assert sorted(read) == sorted(reference), (component, sorted(read), sorted(reference))
        for direction, words in reference.items():
            differ[f"c{component}/d{direction}"] = (
                int((_bits(read[direction]) != _bits(words)).sum()), int(words.size))
    return differ


def _install_differences(lift, reference):
    assert len(lift.installs) == len(reference.installs) == 1
    (volumes, rows), (volumes_ref, rows_ref) = lift.installs[0], reference.installs[0]
    assert sorted(volumes) == sorted(volumes_ref) and sorted(rows) == sorted(rows_ref)
    differ = {f"volume/{name}": int((_bits(volumes[name]) != _bits(volumes_ref[name])).sum())
              for name in volumes}
    differ.update({f"row/{key}": int((_bits(rows[key]) != _bits(rows_ref[key])).sum()) for key in rows})
    return {key: count for key, count in differ.items() if count}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_the_bulk_read_reads_meeps_own_words_and_installs_the_per_point_read(monkeypatch, cell):
    mp = _certified_meep()
    lift = _Lift(monkeypatch, mp, CELLS[cell](mp))
    record = lift.record
    assert record["path"] == "bulk" and record["refusal"] is None and not record["restarted_by"], record
    assert sorted(record["components"]) == ["Ex", "Ey", "Ez"], record
    expected_flat = PER_POINT_ONLY.get(cell, {})
    for name, report in record["components"].items():
        assert report["path"] == "bulk", (name, report)
        # No chunk owns a point the user volume does not, so the user-volume mask of the
        # selection never removes a point (the mutation battery declares that reading
        # equivalent on this measurement).
        assert report["chunk_owned_outside_user_volume"] == 0, (name, report)
        if name in expected_flat:
            assert report["bulk_points"] == 0, (name, report)
            assert report["per_point_only"].startswith(expected_flat[name]), (name, report)
        else:
            assert report["bulk_points"] > 0 and "per_point_only" not in report, (name, report)
    # THE READ WORDS, against MEEP's own reader at every point: 9 x N words, uint64.
    assert len(lift.reads) == 3, sorted(lift.reads)
    differ = _reread_differences(mp, lift)
    assert sum(count for count, _words in differ.values()) == 0, (
        f"{cell}: read words that differ from fields::get_chi1inv: {differ}")
    # THE INSTALLS, against the per-point path on an identical simulation.
    reference = _Lift(monkeypatch, mp, CELLS[cell](mp), bulk_read=False)
    assert reference.record["path"] == "per-point", reference.record
    assert _install_differences(lift, reference) == {}, cell


def test_the_self_check_runs_and_a_restart_installs_the_per_point_read(monkeypatch):
    """A one-ulp word at a copied block's corner: caught, warned, re-read, and exact.

    Planted on the SECOND component, so the restart comes after the first component's
    planes were reported: the progress the caller sees must carry on, never go back. The
    cell has one dispersion everywhere, so the permittivity is the lift's only pass.
    """
    mp = _certified_meep()
    calls = []
    second = int(mp.Ey)
    with pytest.warns(RuntimeWarning, match="bulk chi1inv read refused or disagreed"):
        planted = _Lift(monkeypatch, mp, _walls_background_3d(mp),
                        plant=lambda component: component == second,
                        progress=lambda done, total: calls.append((done, total)))
    record = planted.record
    assert record["path"] == "per-point", record
    assert "Ey" in record["restarted_by"] and "differs from fields::get_chi1inv" in record["restarted_by"], record
    for name in ("Ex", "Ey", "Ez"):
        assert record["components"][name]["path"] == "per-point", record
    reference = _Lift(monkeypatch, mp, _walls_background_3d(mp), bulk_read=False)
    assert _install_differences(planted, reference) == {}
    # Ex's planes were reported before the restart; the restarted pass continues from there.
    done = [d for d, _t in calls]
    assert done == sorted(done), calls
    assert done[0] == 1 and len({t for _d, t in calls}) == 2, calls
    first_total, last_total = calls[0][1], calls[-1][1]
    assert done[-1] == last_total and last_total > first_total, calls
    # And the clean lift's self-check read words at every bulk component.
    clean = _Lift(monkeypatch, mp, _walls_background_3d(mp))
    for name, report in clean.record["components"].items():
        assert report["self_checked_points"] >= min(256, report["bulk_points"]) > 0, (name, report)


def test_an_unlisted_build_reads_every_point_one_at_a_time(monkeypatch):
    """Off ``BULK_CHI1INV_BUILDS`` the lift never copies a word, and says which build it is."""
    mp = _meep()
    monkeypatch.setattr(from_meep, "BULK_CHI1INV_BUILDS", frozenset())
    entered = []
    real_table = from_meep._chi1inv_chunk_table  # noqa: SLF001

    def table_spy(*args, **kwargs):
        entered.append(True)
        return real_table(*args, **kwargs)

    monkeypatch.setattr(from_meep, "_chi1inv_chunk_table", table_spy)
    lift = _Lift(monkeypatch, mp, _walls_background_3d(mp))
    record = lift.record
    stamp = from_meep._chi1inv_build_stamp(from_meep._meep_build(mp))  # noqa: SLF001
    assert record["path"] == "per-point" and tuple(record["stamp"]) == stamp, record
    assert f"MEEP {stamp[0]} {stamp[1]} precision on {stamp[2]} {stamp[3]}" in record["refusal"], record
    assert not entered and not lift.reads, "an unlisted build located MEEP's chunks"
    assert all(report["path"] == "per-point" for report in record["components"].values()), record


def test_the_lift_prints_flushed_progress_unless_meep_is_quiet(capsys):
    """Rule 7: by default a lift says on stderr where it is; at ``mp.verbosity(0)`` it says nothing.

    Read on either path (the per-point read prints the same lines), so this runs on every build.
    """
    mp = _meep()

    def progress_lines():
        driver = from_meep.lift_simulation(_walls_background_3d(mp), prefer_gpu=False)
        driver.close()
        return [line[len(from_meep.LIFT_PROGRESS_HEAD):] for line in capsys.readouterr().err.splitlines()
                if line.startswith(from_meep.LIFT_PROGRESS_HEAD)]

    try:
        mp.verbosity(1)
        loud = progress_lines()
        mp.verbosity(0)
        quiet = progress_lines()
    finally:
        mp.verbosity(0)
    assert loud[0].startswith("init_sim: ") and loud[1].startswith("init_sim done in "), loud
    assert any(line.startswith("chi1inv read: ") for line in loud), loud
    for name in ("Ex", "Ey", "Ez"):
        assert any(line.startswith(f"chi1inv {name}: ") for line in loud), (name, loud)
    assert loud[-1].startswith("lift done in ") and loud[-1].endswith(" s"), loud
    assert quiet == [], quiet


def test_the_listed_builds_are_keyed_as_the_lift_reads_them():
    """Every listed stamp is (version, precision, system, machine), as _chi1inv_build_stamp keys it."""
    for stamp in from_meep.BULK_CHI1INV_BUILDS:
        assert len(stamp) == 4 and stamp[1] in ("single", "double"), stamp
        assert all(isinstance(part, str) and part for part in stamp), stamp
    assert ("1.33.0", "single", "Linux", "x86_64") in from_meep.BULK_CHI1INV_BUILDS
    assert ("1.33.0", "single", "Darwin", "arm64") in from_meep.BULK_CHI1INV_BUILDS
