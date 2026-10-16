"""The corpus cases file of Tier C and the runner's lattice, pinned without MEEP.

``meep_identity_corpus.py`` turns Table 2's legs into Tier C cases; these tests build a
manifest from synthetic legs and hold its labels, its refusals and its plan to what the
corpus harnesses did. The runner's ``lattice`` is held on stub grids: a Bloch-periodic
axis stores MEEP's owned top point as cell 0 of a Yee-shift-0 component, and a Dcyl grid
reads (r, z).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from parity.meep_gpu import meep_identity_corpus as corpus
from parity.meep_gpu import meep_identity_tier_c as tier_c


def _row(**extra):
    record = {"parity": "MEASURED", "parity_rel_l2": 1e-6, "grid_cells": 40, "grid_shape": [40, 1, 1],
              "driver_steps": 200, "cpu_meep_steps": 200, "time_matched": True,
              "capped": {"resolution_original": 20.0, "resolution_used": 10.0, "resolution_capped": True,
                         "steps_planned": 200},
              "facts": {"k_point": None, "src_time_kinds": ["GaussianSource"]}}
    record.update(extra)
    return record


def _legs(tmp_path: Path, rows_by_leg):
    for leg, directory in corpus.LEGS.items():
        path = tmp_path / directory
        path.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(row) for row in rows_by_leg.get(leg, [])]
        (path / "parity.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
    return tmp_path


def test_the_manifest_keeps_measured_rows_and_names_the_rest(tmp_path):
    results = _legs(tmp_path, {
        "examples/main": [_row(script="a-b.py"),
                          _row(script="c.py", parity="LIFT-OR-STEP-FAILED", parity_error="refused")],
        "tests": [_row(module="test_x.py", case="TestX.test_one")],
    })
    document = corpus.manifest_rows(results)
    assert [row["label"] for row in document["rows"]] == ["ex.a-b", "t.test_x.TestX.test_one"]
    assert document["not_cases"] == [{"leg": "examples/main", "case": "c.py",
                                      "parity": "LIFT-OR-STEP-FAILED", "error": "refused"}]
    assert set(document["inputs_sha256"]) == {f"{d}/parity.jsonl" for d in corpus.LEGS.values()}
    assert document["rows"][0]["table2"]["driver_steps"] == 200


def test_two_rows_with_one_label_refuse_the_manifest(tmp_path):
    results = _legs(tmp_path, {"examples/main": [_row(script="a.py")],
                               "examples/blocked": [_row(script="a.py")]})
    with pytest.raises(ValueError, match="not unique"):
        corpus.manifest_rows(results)


def test_a_capped_row_sets_the_resolution_and_refuses_an_initialized_simulation():
    row = {"label": "ex.a", "plan": {"resolution_capped": True, "resolution_used": 12.0}}
    sim = SimpleNamespace(resolution=40.0, fields=None)
    corpus._apply_plan(sim, row)
    assert sim.resolution == 12.0
    with pytest.raises(RuntimeError, match="already initialized"):
        corpus._apply_plan(SimpleNamespace(resolution=40.0, fields=object()), row)
    untouched = SimpleNamespace(resolution=40.0, fields=object())
    corpus._apply_plan(untouched, {"label": "ex.b", "plan": {"resolution_capped": False}})
    assert untouched.resolution == 40.0


def test_the_row_fixes_n_and_the_lifted_grid():
    corpus._CURRENT.clear()
    corpus._CURRENT.update({"label": "ex.a", "table2": {"driver_steps": 321, "grid_cells": 40,
                                                        "rel_l2": 1e-6}})
    assert corpus.steps_planned(None, None) == 321
    sim = SimpleNamespace(resolution=10.0)
    assert corpus.after_lift(None, sim, SimpleNamespace(shape=(40, 1, 1)))["grid_cells"] == 40
    with pytest.raises(RuntimeError, match="not the row Table 2 measured"):
        corpus.after_lift(None, sim, SimpleNamespace(shape=(41, 1, 1)))


def test_a_test_row_puts_the_parameterized_stand_in_after_the_environment(tmp_path, monkeypatch):
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    monkeypatch.setenv(corpus.SOURCE_ENVIRONMENT, str(tmp_path))
    shims = os.path.join(corpus.HERE, "test_shims")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != shims])
    before = list(sys.path)
    with pytest.raises(FileNotFoundError, match="no such test module"):
        corpus._build_test(None, {"module": "test_x.py", "case": "TestX.test_one", "label": "t.x"})
    assert sys.path == before + [shims]  # appended after the environment's entries, once
    # The stand-in names an expanded method as the rows were recorded: index, then the
    # first argument made safe (a float's point included), so a row resolves by name.
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "_shim_parameterized", os.path.join(shims, "parameterized", "__init__.py"))
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    assert shim._expanded_name("test_reflectance_angular", 1, (20.6, False)) == "test_reflectance_angular_1_20_6"
    assert shim._expanded_name("test_reflectance_angular", 0, (0, False)) == "test_reflectance_angular_0_0"


class _Vec:
    def __init__(self, x=0, y=0, z=0, r=0):
        self._v = (x, y, z, r)

    def x(self):
        return self._v[0]

    def y(self):
        return self._v[1]

    def z(self):
        return self._v[2]

    def r(self):
        return self._v[3]


class _Grid:
    def __init__(self, counts, invariant, window_low, cylindrical=False):
        self.nx, self.ny, self.nz = counts
        self._invariant, self._low, self.cylindrical = invariant, window_low, cylindrical

    def is_invariant(self, axis):
        return self._invariant[axis]

    def stored_window_low(self, axis, shift):
        return self._low[axis] if shift == 0 else 0


def _sim(corner, shift):
    gv = SimpleNamespace(little_corner=lambda: corner, iyee_shift=lambda component: shift)
    return SimpleNamespace(fields=SimpleNamespace(gv=gv), resolution=10.0)


def test_a_bloch_axis_reads_the_owned_top_point_into_stored_cell_zero():
    mp = SimpleNamespace(Hx="Hx")
    grid = _Grid((4, 3, 1), (False, False, True), (1, 0, 0))
    driver = SimpleNamespace(grid=grid)
    axes = tier_c.lattice(mp, _sim(_Vec(x=-4, y=-3), _Vec(x=0, y=1)), driver, "Hx")
    assert axes == ([4, -2, 0, 2], [-2, 0, 2], [0])
    plain = _Grid((4, 3, 1), (False, False, True), (0, 0, 0))
    assert tier_c.lattice(mp, _sim(_Vec(x=-4, y=-3), _Vec(x=0, y=1)), SimpleNamespace(grid=plain),
                          "Hx") == ([-4, -2, 0, 2], [-2, 0, 2], [0])


def test_a_dcyl_grid_reads_r_and_z_under_meep_cylindrical_names():
    mp = SimpleNamespace(Hr="Hr", Hx="wrong")
    grid = _Grid((3, 1, 2), (False, True, False), (0, 0, 0), cylindrical=True)
    driver = SimpleNamespace(grid=grid)
    assert tier_c.meep_component(mp, driver, "Hx") == "Hr"
    axes = tier_c.lattice(mp, _sim(_Vec(r=0, z=-2), _Vec(r=1, z=0)), driver, "Hx")
    assert axes == ([1, 3, 5], [0], [-2, 0])


class _FoldedGrid(_Grid):
    """Axis 0 mirror-folded and periodic, N = 8 even: 6 stored cells, 5 owned."""

    nx_full, ny_full, nz_full = 8, 1, 1

    def __init__(self):
        super().__init__((6, 1, 1), (False, True, True), (0, 0, 0))

    def is_mirrored(self, axis):
        return axis == 0

    def stored_cells(self, axis):
        return (6, 1, 1)[axis]

    def owned_cells(self, axis):
        return (5, 1, 1)[axis]


def _folded_sim(shift):
    gv = SimpleNamespace(little_corner=lambda: _Vec(x=-2), big_corner=lambda: _Vec(x=8),
                         iyee_shift=lambda component: shift)
    return SimpleNamespace(fields=SimpleNamespace(gv=gv), resolution=10.0)


def test_the_far_ghost_slot_of_a_folded_periodic_axis_reads_its_image():
    mp = SimpleNamespace(Hy="Hy", Hx="Hx")
    driver = SimpleNamespace(grid=_FoldedGrid())
    # Shift 1: -1 .. 9, and 9 is one past big_corner 8; the second mirror sits at
    # (-2) + 2 + 8 = 8, so the slot reads its image 7 (a ghost cell, never judged).
    assert tier_c.lattice(mp, _folded_sim(_Vec(x=1)), driver, "Hy")[0] == [-1, 1, 3, 5, 7, 7]
    # Shift 0 ends on big_corner itself: no far slot, nothing moves.
    assert tier_c.lattice(mp, _folded_sim(_Vec(x=0)), driver, "Hx")[0] == [-2, 0, 2, 4, 6, 8]


def test_a_refusal_before_the_case_directory_exists_is_still_written(tmp_path, monkeypatch):
    from meep_gpu import array_path_oracle  # noqa: PLC0415

    monkeypatch.setattr(array_path_oracle, "live",
                        lambda: SimpleNamespace(fixture_set="stub", fingerprint_sha256="0" * 64))
    monkeypatch.setattr(tier_c, "run_child", lambda spec: {"case": spec["case"], "error": "refused by name"})
    arguments = SimpleNamespace(out=str(tmp_path / "out"), cases_file=None, cases="a", mode="keep",
                                steps=None, rows=None)
    assert tier_c.command_compare(arguments) == 3
    written = json.loads((tmp_path / "out" / "keep" / "a.error.json").read_text())
    assert written["error"] == "refused by name"


def test_meep_point_read_drops_exactly_the_signed_zeros_meeps_complex_read_drops():
    import numpy  # noqa: PLC0415

    def word(re, im):
        out = numpy.empty((), dtype=numpy.complex64)
        out.real, out.imag = numpy.float32(re), numpy.float32(im)
        return out

    def bits(value):
        return tuple(int(v) for v in numpy.asarray(value).reshape(1).view(numpy.uint32))

    cases = [((0.0, -0.0), (0.0, 0.0)),        # imaginary -0 beside +0 reads +0
             ((3e-45, -0.0), (3e-45, 0.0)),    # ... beside a positive real part too
             ((-0.0, -0.0), (0.0, -0.0)),      # real -0 beside imaginary -0 reads +0
             ((-0.0, -1.5), (0.0, -1.5)),      # ... beside a negative imaginary part
             ((-0.0, 1.5), (-0.0, 1.5)),       # kept beside a positive one
             ((-2.0, -0.0), (-2.0, -0.0)),     # imaginary -0 beside a negative real part kept
             ((1.25, -3.5), (1.25, -3.5))]     # nonzero words unchanged
    for (re, im), (want_re, want_im) in cases:
        assert bits(tier_c.meep_point_read(word(re, im))) == bits(word(want_re, want_im)), (re, im)
    real = numpy.array([-0.0, 0.0, 1e-45], dtype=numpy.float32)
    assert tier_c.meep_point_read(real) is not None
    assert numpy.array_equal(tier_c.meep_point_read(real).view(numpy.uint32), real.view(numpy.uint32))


def test_subnormal_words_are_counted_from_bit_patterns():
    import numpy  # noqa: PLC0415

    words = numpy.array([0x00000001, 0x80400000, 0x00800000, 0x0, 0x80000000], dtype=numpy.uint32)
    assert tier_c.subnormal_words(words.view(numpy.float32)) == 2
