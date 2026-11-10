"""The bulk chi1inv read runs on every build, reads MEEP's own words, says whether it is certified.

``_lift_epsilon_structured`` copies the stored ``chi1inv`` words out of MEEP's chunks and
reads the rest one point at a time through ``fields::get_chi1inv``. It falls back to the
per-point read whenever its bulk read refuses or its sampled self-check disagrees, and
the fallback is exact by construction. That makes a defect in the bulk read SILENT: the
lift still installs the right permittivity, only slower, and an installed ``1/x`` can
hide a one-ulp read error, so no physics test can see it. These tests pin it directly:

* the READ words of every component (the diagonal and both off-diagonal entries, as
  uint64) equal ``fields::get_chi1inv`` at every point of the lattice the lift reads,
  and the installed arrays equal the per-point path's bit for bit, on a certified and on
  an uncertified build;
* the bulk path is the one taken (no refusal, no restart, points read in bulk on every
  component except the named per-point-only shapes, whose reason names the shape);
* the self-check runs: a one-ulp word planted at a copied block's corner restarts the
  read with a warning, the restart installs exactly the per-point read, and the progress
  sequence the caller sees never goes back;
* Supported is not certified: a build outside ``CERTIFIED_CHI1INV_BUILDS`` takes the bulk
  read, records ``certified_build: False``, reads every block's corners, points along its
  edges and a sample of its own (a word planted at a point only that denser check reads
  restarts the read), and prints one NOTE line per process; ``MEEP_GPU_HARDWARE=certified`` refuses
  the bulk read on it by name, and a value that switch does not accept refuses it on
  every build; a certified build reads exactly 1.0's sample and prints nothing;
* a lift prints its flushed progress lines on stderr by default and none at
  ``mp.verbosity(0)``.

One small cell per ownership situation the bulk read must reproduce: a 3-D PML cell with
an off-diagonal tensor over several chunks, metallic walls behind a non-vacuum
background (unowned wall planes reading MEEP's 1.0 default), a mirror fold on a periodic
axis (the far ghost slot), a cylindrical cell (the r = 0 ownership case), a dispersive
background, and a cylindrical cell with no z extent (the named per-point-only shape).
Every test runs on every MEEP build: whether the build is certified is set per test, by
replacing ``CERTIFIED_CHI1INV_BUILDS`` with the build's own stamp or with nothing.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import dataclasses

import numpy
import pytest

from conftest import requires_resource_skip

from meep_gpu import fastpath, from_meep
from meep_gpu.driver import FdtdDriver

#: The shape the bulk path owns no point of, by component: MEEP builds a cylindrical cell
#: with no z extent one pixel thick, and Er and Ep sit on that one z plane.
FLAT_CYLINDER_SHAPE = "a cylindrical cell one pixel thick in z"
#: The certified and the uncertified build, as each test asks for one.
CERTIFIED, UNCERTIFIED = True, False


def _meep():
    try:
        import meep as mp  # noqa: PLC0415
    except ImportError:
        requires_resource_skip("meep", "the lift reads an mp.Simulation")
    mp.verbosity(0)
    return mp


def _stamp(mp):
    return from_meep._chi1inv_build_stamp(from_meep._meep_build(mp))  # noqa: SLF001


def _certify(monkeypatch, mp, certified):
    """This build, certified (its own stamp the whole list) or not (an empty list)."""
    stamp = _stamp(mp)
    monkeypatch.setattr(from_meep, "CERTIFIED_CHI1INV_BUILDS",
                        frozenset({stamp}) if certified else frozenset())
    return stamp


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """No hardware scope from the caller's environment, and no NOTE line already printed."""
    monkeypatch.delenv(fastpath.HARDWARE_SWITCH, raising=False)
    monkeypatch.setattr(from_meep, "_CHI1INV_NOTES_ANNOUNCED", set())


def _notes(text):
    return [line for line in text.splitlines() if line.startswith(from_meep.CHI1INV_UNCERTIFIED_NOTE_HEAD)]


def _certified_count(report):
    """1.0's self-check: every block's two corners and up to 256 drawn bulk points."""
    return 2 * report["blocks"] + min(256, report["bulk_points"])


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

    ``plant(component, table, blocks, pending)`` may return ``True`` (a one-ulp word at the
    first copied block's low corner, where every self-check reads) or a position to plant
    that word at, before the self-check runs. ``table_wrap(table)`` replaces the located
    chunk table, for a test that has MEEP's ownership answer differently.
    """

    def __init__(self, monkeypatch, mp, sim, *, bulk_read=None, plant=None, progress=None,
                 table_wrap=None):
        self.reads = {}       # MEEP component -> (indices, own direction, values, coupling)
        self.checks = {}      # MEEP component -> (table, blocks, pending, certified) at the self-check
        self.installs = []    # (volumes, coupling rows) per set_epsilon_components call
        self.planted = []     # (component, position) of each planted word
        real_set = FdtdDriver.set_epsilon_components
        real_check = from_meep._chi1inv_bulk_self_check  # noqa: SLF001
        real_lift = from_meep._lift_epsilon_structured  # noqa: SLF001
        real_table = from_meep._chi1inv_chunk_table  # noqa: SLF001
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
            lift.checks[int(component)] = (table, blocks, pending.copy(), kwargs.get("certified"))
            where = plant(int(component), table, blocks, pending) if plant is not None and blocks else None
            if where is True:
                _chunk, select, _flat = blocks[0]
                where = tuple(int(axis[0]) for axis in select)
            if where:
                values[where] = numpy.nextafter(values[where], numpy.inf)
                lift.planted.append((int(component), where))
            return real_check(table, component, own, others, indices, ivec, get_chi1inv, values,
                              coupling, blocks, pending, *args, **kwargs)

        def lift_spy(*args, **kwargs):
            if bulk_read is not None and not kwargs.get("_restarted_by"):
                kwargs["_bulk_read"] = bulk_read
            return real_lift(*args, **kwargs)

        def table_spy(*args, **kwargs):
            table, refusal = real_table(*args, **kwargs)
            return (table if table is None or table_wrap is None else table_wrap(table)), refusal

        self.cylindrical = sim.dimensions == mp.CYLINDRICAL or bool(getattr(sim, "is_cylindrical", False))
        with monkeypatch.context() as patch:
            patch.setattr(FdtdDriver, "set_epsilon_components", set_spy)
            patch.setattr(from_meep, "_chi1inv_bulk_self_check", check_spy)
            patch.setattr(from_meep, "_lift_epsilon_structured", lift_spy)
            patch.setattr(from_meep, "_chi1inv_chunk_table", table_spy)
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


@pytest.mark.parametrize("certified", [CERTIFIED, UNCERTIFIED], ids=["certified", "uncertified"])
@pytest.mark.parametrize("cell", sorted(CELLS))
def test_the_bulk_read_reads_meeps_own_words_and_installs_the_per_point_read(monkeypatch, cell, certified):
    mp = _meep()
    _certify(monkeypatch, mp, certified)
    lift = _Lift(monkeypatch, mp, CELLS[cell](mp))
    record = lift.record
    assert record["path"] == "bulk" and record["refusal"] is None and not record["restarted_by"], record
    assert record["certified_build"] is certified, record
    assert sorted(record["components"]) == ["Ex", "Ey", "Ez"], record
    assert record["self_checked_points"] == sum(
        report["self_checked_points"] for report in record["components"].values()), record
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
    # The read words, against MEEP's own reader at every point: 9 x N words, uint64.
    assert len(lift.reads) == 3, sorted(lift.reads)
    assert all(check[3] is certified for check in lift.checks.values()), lift.checks
    differ = _reread_differences(mp, lift)
    assert sum(count for count, _words in differ.values()) == 0, (
        f"{cell}: read words that differ from fields::get_chi1inv: {differ}")
    # THE INSTALLS, against the per-point path on an identical simulation.
    reference = _Lift(monkeypatch, mp, CELLS[cell](mp), bulk_read=False)
    assert reference.record["path"] == "per-point", reference.record
    assert reference.record["self_checked_points"] == 0, reference.record
    assert _install_differences(lift, reference) == {}, cell


@pytest.mark.parametrize("certified", [CERTIFIED, UNCERTIFIED], ids=["certified", "uncertified"])
def test_the_self_check_runs_and_a_restart_installs_the_per_point_read(monkeypatch, capsys, certified):
    """A one-ulp word at a copied block's corner: caught, warned, re-read, and exact, on either build.

    Planted on the SECOND component, so the restart comes after the first component's
    planes were reported: the progress the caller sees must carry on, never go back. The
    cell has one dispersion everywhere, so the permittivity is the lift's only pass.
    """
    mp = _meep()
    _certify(monkeypatch, mp, certified)
    calls = []
    second = int(mp.Ey)
    with pytest.warns(RuntimeWarning, match="bulk chi1inv read refused or disagreed"):
        planted = _Lift(monkeypatch, mp, _walls_background_3d(mp),
                        plant=lambda component, *_: component == second,
                        progress=lambda done, total: calls.append((done, total)))
    record = planted.record
    assert record["path"] == "per-point" and record["certified_build"] is certified, record
    assert "Ey" in record["restarted_by"] and "differs from fields::get_chi1inv" in record["restarted_by"], record
    for name in ("Ex", "Ey", "Ez"):
        assert record["components"][name]["path"] == "per-point", record
    # A restarted read installs MEEP's own words: nothing for the NOTE line to say.
    assert _notes(capsys.readouterr().err) == []
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


def test_an_uncertified_build_takes_the_bulk_read_and_says_so(monkeypatch, capsys):
    """Off ``CERTIFIED_CHI1INV_BUILDS``: the bulk read, ``certified_build: False``, a denser check, one NOTE.

    The NOTE line is printed whatever ``mp.verbosity`` is (the lift runs at 0 here), once
    per process: a second lift on the same build prints nothing more.
    """
    mp = _meep()
    stamp = _certify(monkeypatch, mp, UNCERTIFIED)
    first = _Lift(monkeypatch, mp, _walls_background_3d(mp))
    record = first.record
    assert record["path"] == "bulk" and record["refusal"] is None and not record["restarted_by"], record
    assert record["certified_build"] is False and tuple(record["stamp"]) == stamp, record
    total = 0
    for name, report in record["components"].items():
        assert report["path"] == "bulk" and report["bulk_points"] > 0, (name, report)
        assert report["self_checked_points"] > _certified_count(report), (name, report)
        total += report["self_checked_points"]
    assert record["self_checked_points"] == total, record
    notes = _notes(capsys.readouterr().err)
    assert len(notes) == 1, notes
    assert from_meep._chi1inv_build_text(stamp) in notes[0], notes  # noqa: SLF001
    assert "MEEP_GPU_HARDWARE=certified" in notes[0] and "(certified: none)" in notes[0], notes
    second = _Lift(monkeypatch, mp, _walls_background_3d(mp))
    assert second.record["path"] == "bulk" and second.record["certified_build"] is False
    assert _notes(capsys.readouterr().err) == []
    # Under the `any` scope the same; under `supported` (the default, here set) the same.
    for scope in (fastpath.HARDWARE_ANY, fastpath.HARDWARE_SUPPORTED):
        monkeypatch.setenv(fastpath.HARDWARE_SWITCH, scope)
        third = _Lift(monkeypatch, mp, _walls_background_3d(mp))
        assert third.record["path"] == "bulk" and third.record["refusal"] is None, (scope, third.record)


def test_a_certified_stamp_is_unchanged(monkeypatch, capsys):
    """On ``CERTIFIED_CHI1INV_BUILDS``: 1.0's read and 1.0's sample, no NOTE, under every scope."""
    mp = _meep()
    stamp = _certify(monkeypatch, mp, CERTIFIED)
    for scope in (None, fastpath.HARDWARE_CERTIFIED, fastpath.HARDWARE_SUPPORTED, fastpath.HARDWARE_ANY):
        if scope is None:
            monkeypatch.delenv(fastpath.HARDWARE_SWITCH, raising=False)
        else:
            monkeypatch.setenv(fastpath.HARDWARE_SWITCH, scope)
        lift = _Lift(monkeypatch, mp, _walls_background_3d(mp))
        record = lift.record
        assert record["path"] == "bulk" and record["refusal"] is None, (scope, record)
        assert record["certified_build"] is True and tuple(record["stamp"]) == stamp, (scope, record)
        for name, report in record["components"].items():
            assert report["self_checked_points"] == _certified_count(report), (scope, name, report)
        assert record["self_checked_points"] == sum(
            _certified_count(report) for report in record["components"].values()), (scope, record)
        assert all(check[3] is True for check in lift.checks.values()), (scope, lift.checks)
    assert _notes(capsys.readouterr().err) == []


def test_certified_mode_refuses_an_uncertified_build_by_name(monkeypatch, capsys):
    """``MEEP_GPU_HARDWARE=certified`` on an uncertified build: per point, said by name, no NOTE."""
    mp = _meep()
    stamp = _certify(monkeypatch, mp, UNCERTIFIED)
    monkeypatch.setenv(fastpath.HARDWARE_SWITCH, fastpath.HARDWARE_CERTIFIED)
    entered = []
    real_table = from_meep._chi1inv_chunk_table  # noqa: SLF001

    def table_spy(*args, **kwargs):
        entered.append(True)
        return real_table(*args, **kwargs)

    monkeypatch.setattr(from_meep, "_chi1inv_chunk_table", table_spy)
    lift = _Lift(monkeypatch, mp, _walls_background_3d(mp))
    record = lift.record
    assert record["path"] == "per-point" and record["certified_build"] is False, record
    assert record["refusal"].startswith(from_meep._chi1inv_build_text(stamp)), record  # noqa: SLF001
    assert "is not a certified build of the bulk read" in record["refusal"], record
    assert "MEEP_GPU_HARDWARE=certified restricts the bulk read" in record["refusal"], record
    assert record["self_checked_points"] == 0, record
    assert all(report["path"] == "per-point" for report in record["components"].values()), record
    assert not entered and not lift.reads, "a refused build located MEEP's chunks"
    assert _notes(capsys.readouterr().err) == []
    # A comparison may still ask for the bulk read on it (the build-comparison harness does).
    forced = _Lift(monkeypatch, mp, _walls_background_3d(mp), bulk_read=True)
    assert forced.record["path"] == "bulk" and forced.record["certified_build"] is False, forced.record


def test_an_unaccepted_hardware_value_refuses_the_bulk_read_on_every_build(monkeypatch):
    """A ``MEEP_GPU_HARDWARE`` value it does not accept is refused by name, on a certified build too."""
    mp = _meep()
    for certified in (CERTIFIED, UNCERTIFIED):
        _certify(monkeypatch, mp, certified)
        monkeypatch.setenv(fastpath.HARDWARE_SWITCH, "Certified ")
        record = _Lift(monkeypatch, mp, _walls_background_3d(mp)).record
        assert record["path"] == "per-point" and record["certified_build"] is certified, record
        assert record["refusal"].startswith("MEEP_GPU_HARDWARE='Certified ' is not an accepted value"), record


def test_the_dense_plan_reads_every_corner_its_edges_and_a_block_sample():
    """The self-check's points, by group, on a synthetic block layout (no MEEP needed)."""
    shape = (40, 30, 2)
    pending = numpy.ones(shape, dtype=bool)
    selects = [[numpy.arange(1, 21), numpy.arange(0, 30), numpy.arange(0, 2)],
               [numpy.arange(21, 40), numpy.arange(0, 30), numpy.arange(0, 1)],
               [numpy.arange(21, 22), numpy.arange(5, 6), numpy.arange(1, 2)]]
    blocks = []
    for select in selects:
        pending[numpy.ix_(*select)] = False
        blocks.append((None, select, None))
    sparse = from_meep._chi1inv_self_check_plan(blocks, pending, certified=True)  # noqa: SLF001
    dense = from_meep._chi1inv_self_check_plan(blocks, pending, certified=False)  # noqa: SLF001
    bulk = int((~pending).sum())
    # Certified: 1.0's points exactly (two corners per block, the one-point block twice).
    assert [local for _index, local in sparse["structural"]] == [(0, 0, 0), (-1, -1, -1)] * 3
    assert len(sparse["corners"]) == 6 and len(sparse["samples"]) == min(256, bulk)
    assert sparse["edges"] == sparse["block_samples"] == []
    flat = [point for group in from_meep._SELF_CHECK_GROUPS for point in dense[group]]  # noqa: SLF001
    assert len(flat) == len(set(flat)), "a point read twice by the dense plan"
    assert set(sparse["corners"]) | set(sparse["samples"]) <= set(flat)
    assert set(flat) <= {tuple(map(int, point)) for point in numpy.argwhere(~pending)}
    for index, (_chunk, select, _flat) in enumerate(blocks):
        lo, hi = [int(a[0]) for a in select], [int(a[-1]) for a in select]
        corners = {(x, y, z) for x in {lo[0], hi[0]} for y in {lo[1], hi[1]} for z in {lo[2], hi[2]}}
        assert corners <= set(dense["corners"]), (index, corners)
        assert {local for i, local in dense["structural"] if i == index} == set(
            from_meep._chi1inv_block_corners(select)), index  # noqa: SLF001
        # Every edge: at least min(interior, _DENSE_EDGE_POINTS) of its interior points, the
        # two next to its corners among them (every point of a shorter edge).
        for along in range(3):
            first, second = [axis for axis in range(3) if axis != along]
            interior = [int(t) for t in select[along][1:-1]]
            for a in {lo[first], hi[first]}:
                for b in {lo[second], hi[second]}:
                    line = []
                    for t in interior:
                        point = [0, 0, 0]
                        point[along], point[first], point[second] = t, a, b
                        line.append(tuple(point))
                    read = [point for point in line if point in set(flat)]
                    assert len(read) >= min(len(line), from_meep._DENSE_EDGE_POINTS), (  # noqa: SLF001
                        index, along, len(read), len(line))
                    if line:
                        assert line[0] in read and line[-1] in read, (index, along, a, b)
        # The block's own sample: max(64, points // 4096) of its points, all of a smaller one.
        size = int(numpy.prod([len(a) for a in select]))
        inside = [p for p in flat if all(int(select[a][0]) <= p[a] <= int(select[a][-1]) for a in range(3))]
        assert len(inside) >= min(size, from_meep._DENSE_BLOCK_SAMPLES), (index, len(inside), size)  # noqa: SLF001
    assert dense["block_samples"], dense
    assert from_meep._DENSE_EDGE_POINTS == 16, "the stride the record states"  # noqa: SLF001


def _dense_without(monkeypatch, blocks, pending, group):
    """The uncertified plan's word points with one group's own source switched off."""
    with monkeypatch.context() as patch:
        if group == "corners":
            patch.setattr(from_meep, "_chi1inv_block_corners", lambda select: [(0, 0, 0), (-1, -1, -1)])
        elif group == "edges":
            patch.setattr(from_meep, "_chi1inv_block_edges", lambda select: [])
        else:
            patch.setattr(from_meep, "_DENSE_BLOCK_SAMPLES", 0)
            patch.setattr(from_meep, "_DENSE_BLOCK_FRACTION", 10 ** 12)
        plan = from_meep._chi1inv_self_check_plan(blocks, pending, certified=False)  # noqa: SLF001
    return {point for name in from_meep._SELF_CHECK_GROUPS for point in plan[name]}  # noqa: SLF001


@pytest.mark.parametrize("group", ["corners", "edges", "block_samples"])
def test_a_word_only_the_dense_check_reads_restarts_an_uncertified_read(monkeypatch, group):
    """One ulp at a point ONLY its group of the uncertified plan reads: caught, and exact after.

    The point is chosen so that nothing else reads it: neither the uncertified plan with that
    group's source switched off (two corners per block, no edges, no block samples) nor 1.0's
    plan. So the same plant on a certified build is not read, and the certified lift keeps
    the word: the gap the denser check closes.
    """
    mp = _meep()
    second = int(mp.Ey)

    def choose(component, _table, blocks, pending):
        if component != second:
            return None
        plan = from_meep._chi1inv_self_check_plan(blocks, pending, certified=False)  # noqa: SLF001
        sparse = from_meep._chi1inv_self_check_plan(blocks, pending, certified=True)  # noqa: SLF001
        rest = _dense_without(monkeypatch, blocks, pending, group)
        read_by_sparse = set(sparse["corners"]) | set(sparse["samples"])
        only = [point for point in plan[group] if point not in rest and point not in read_by_sparse]
        assert only, f"no {group} point that only the {group} group reads"
        return only[len(only) // 2]

    _certify(monkeypatch, mp, UNCERTIFIED)
    with pytest.warns(RuntimeWarning, match="bulk chi1inv read refused or disagreed"):
        planted = _Lift(monkeypatch, mp, _walls_background_3d(mp), plant=choose)
    assert planted.planted and planted.checks[second][3] is False, (planted.planted, planted.checks)
    record = planted.record
    assert record["path"] == "per-point" and "Ey" in record["restarted_by"], record
    assert "differs from fields::get_chi1inv" in record["restarted_by"], record
    reference = _Lift(monkeypatch, mp, _walls_background_3d(mp), bulk_read=False)
    assert _install_differences(planted, reference) == {}
    # The same plant on a certified build: 1.0's sample does not read it, so the word stays
    # (read words, not installs: an installed 1/x can hide one ulp).
    _certify(monkeypatch, mp, CERTIFIED)
    kept = _Lift(monkeypatch, mp, _walls_background_3d(mp), plant=choose)
    assert kept.record["path"] == "bulk" and not kept.record["restarted_by"], kept.record
    differ = _reread_differences(mp, kept)
    assert sum(count for count, _words in differ.values()) == 1, (differ, kept.planted)


class _LyingVolume:
    """A MEEP grid_volume whose ``owns`` answers False at one lattice point."""

    def __init__(self, volume, directions, lattice):
        self._volume, self._directions, self._lattice = volume, directions, lattice

    def owns(self, location):
        point = tuple(int(location.in_direction(d)) for d in self._directions if d is not None)
        return False if point == self._lattice else self._volume.owns(location)

    def __getattr__(self, name):
        return getattr(self._volume, name)


@pytest.mark.parametrize("certified", [CERTIFIED, UNCERTIFIED], ids=["certified", "uncertified"])
def test_ownership_is_asked_at_every_corner_of_an_uncertified_block(monkeypatch, certified):
    """``grid_volume::owns`` answering False at a corner off the block's diagonal.

    The uncertified check asks MEEP at all of a block's corners and restarts; the
    certified check asks at the two diagonal corners only, as 1.0 did, and does not.
    """
    mp = _meep()
    _certify(monkeypatch, mp, certified)
    first = int(mp.Ex)
    probe = _Lift(monkeypatch, mp, _walls_background_3d(mp))
    table, blocks, _pending, _certified = probe.checks[first]
    chunk, select, _flat = blocks[0]
    chunk_index = next(i for i, c in enumerate(table.chunks) if c is chunk)
    indices = probe.reads[first][0]
    local = next(corner for corner in from_meep._chi1inv_block_corners(select)  # noqa: SLF001
                 if corner not in ((0, 0, 0), (len(select[0]) - 1, len(select[1]) - 1, len(select[2]) - 1)))
    lattice = tuple(int(indices[axis][int(select[axis][local[axis]])])
                    for axis in range(3) if table.directions[axis] is not None)

    def lie(located):
        chunks = list(located.chunks)
        chunks[chunk_index] = dataclasses.replace(
            chunks[chunk_index], volume=_LyingVolume(chunks[chunk_index].volume, located.directions, lattice))
        return dataclasses.replace(located, chunks=tuple(chunks))

    if certified:
        lied = _Lift(monkeypatch, mp, _walls_background_3d(mp), table_wrap=lie)
        assert lied.record["path"] == "bulk" and not lied.record["restarted_by"], lied.record
        return
    with pytest.warns(RuntimeWarning, match="bulk chi1inv read refused or disagreed"):
        lied = _Lift(monkeypatch, mp, _walls_background_3d(mp), table_wrap=lie)
    assert lied.record["path"] == "per-point", lied.record
    assert "Ex: grid_volume::owns disagrees at a copied block's corner" in lied.record["restarted_by"], lied.record


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


def test_the_certified_builds_are_keyed_as_the_lift_reads_them():
    """Every certified stamp is (version, precision, system, machine), as _chi1inv_build_stamp keys it.

    The list names the certified reference builds, one per certified platform: MEEP 1.35.0 in
    single precision on Linux x86_64 and Darwin arm64, and nothing else.
    """
    for stamp in from_meep.CERTIFIED_CHI1INV_BUILDS:
        assert len(stamp) == 4 and stamp[1] in ("single", "double"), stamp
        assert all(isinstance(part, str) and part for part in stamp), stamp
    assert from_meep.CERTIFIED_CHI1INV_BUILDS == {("1.35.0", "single", "Linux", "x86_64"),
                                                  ("1.35.0", "single", "Darwin", "arm64")}
