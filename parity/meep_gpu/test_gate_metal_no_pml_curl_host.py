"""Host checks of the two Metal no-absorber gates: their oracle, their records, their needles.

The gates' device legs need an Apple GPU; what they carry besides the kernel does not,
and it is what keeps the device legs meaningful between device runs:

* **the curl gate's oracle is the array path** on every grid its cases model: its
  bare-array transcription (``oracle_once``, with the partner presence taken from the
  case's invariance) steps exactly the words ``stepping.step_B`` / ``step_D`` store on
  an engine grid of the same shape, at 3-D, 2-D and 1-D, on periodic and metallic
  faces, with E stored and derived;
* **the record each case binds is the composer's**: ``case_record`` equals
  ``chunk_map.chunk_bounds_record(ChunkLayout.one_rank(grid, None))``;
* **every source mutation lands**: each needle is present exactly once in the source
  it is armed on, the grouping mutation plants the package's previous pairwise bracket
  (KERNELS.md §7.1), and each absent-partner row has a declared-null twin on a grid
  where the planted arm never runs. The same for the fused pair's plain family.

Each check runs the gate in a child process with torch made unimportable: the host
legs are torch-free, and the gates' own environment defaults stay out of this session.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

#: Put first in the child: torch cannot be imported, so a host leg that reached for it
#: fails here by name rather than passing on a host that happens to have it.
_NO_TORCH = textwrap.dedent('''
    import importlib.abc, sys
    class _NoTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is not importable in this host check")
            return None
    sys.meta_path.insert(0, _NoTorch())
    sys.path[:0] = [{here!r}, {root!r}]
''').format(here=HERE, root=API_ROOT)


def _child(body: str) -> dict:
    """Run ``body`` after the torch block; it prints one JSON object last."""
    environment = dict(os.environ)
    environment.pop("MEEP_GPU_SUBNORMAL_POLICY", None)
    result = subprocess.run([sys.executable, "-c", _NO_TORCH + textwrap.dedent(body)],
                            cwd=API_ROOT, env=environment, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_oracle_is_the_array_path_on_every_grid_the_cases_model():
    summary = _child('''
        import json
        import numpy as np
        import gate_metal_no_pml_curl as gate
        from meep_gpu import stepping
        from meep_gpu.fields import Fields
        from meep_gpu.grid import Grid
        from meep_gpu.pml import PML

        grids = (("3d", (1.25, 1.0, 0.75), 3), ("2d", (1.25, 1.0, 0.0), 2),
                 ("1d", (0.0, 0.0, 2.0), 1))
        rows = []
        for label, cell, dimensions in grids:
            for metallic in (False, True):
                kinds = {}
                if metallic:
                    kinds = {"x": "metallic", "z": "metallic"} if dimensions == 3 else (
                        {"x": "metallic", "y": "metallic"} if dimensions == 2
                        else {"z": "metallic"})
                for stored in (False, True):
                    for sub_step in ("step_B", "step_D"):
                        grid = Grid(resolution=8.0, cell_size=cell, courant=0.35,
                                    xp=np, dimensions=dimensions,
                                    **({"boundaries": kinds} if kinds else {}))
                        fields = Fields(grid=grid)
                        rng = np.random.default_rng(7 + len(rows))
                        inverse = {name: (0.25 + 0.6 * rng.random(grid.shape)).astype(
                            np.float32) for name in ("Ex", "Ey", "Ez")}
                        fields.set_epsilon_volumes(
                            {name: (1.0 / value).astype(np.float32)
                             for name, value in inverse.items()}, inverse)
                        if stored:
                            fields.enable_field_storage()
                        for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                                     "Ex", "Ey", "Ez"):
                            array = getattr(fields, name, None)
                            if array is not None:
                                array[...] = rng.uniform(-0.45, 0.45,
                                                         grid.shape).astype(np.float32)
                        arrays = {name: np.array(getattr(fields, name), copy=True)
                                  for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz")}
                        for name in ("Ex", "Ey", "Ez"):
                            arrays["inv_eps_" + name] = np.array(
                                fields.inverse_epsilon_for(name), copy=True)
                            stored_e = getattr(fields, name, None)
                            if stored_e is not None:
                                arrays[name] = np.array(stored_e, copy=True)
                        pml = PML(grid=grid, thickness=0)
                        kinds_resolved = stepping._boundary_kinds(grid, None)
                        codes = tuple(1 if kind == "metallic" else 0
                                      for kind in kinds_resolved)
                        invariant = tuple(grid.is_invariant(axis) for axis in range(3))
                        derive = sub_step == "step_B" and not fields.stores_E
                        before = {name: np.array(arrays[name], copy=True)
                                  for name in gate.family.SUB_STEPS[sub_step]["targets"]}
                        gate.oracle_once(arrays, sub_step, codes, grid.dt / grid.dx,
                                         derive, invariant)
                        getattr(stepping, sub_step)(fields, pml)
                        targets = gate.family.SUB_STEPS[sub_step]["targets"]
                        differing = sum(gate.differing(arrays[name], getattr(fields, name))
                                        for name in targets)
                        moved = sum(gate.differing(before[name], arrays[name])
                                    for name in targets)
                        rows.append({"grid": label, "metallic": metallic,
                                     "stored": stored, "sub_step": sub_step,
                                     "codes": codes, "invariant": invariant,
                                     "differing": differing, "moved": moved})
        print(json.dumps(rows))
    ''')
    assert len(summary) == 24, summary
    for row in summary:
        assert row["differing"] == 0, row
        assert row["moved"] > 0, row
    assert {tuple(row["invariant"]) for row in summary} == {
        (False, False, False), (False, False, True), (True, True, False)}


def test_each_case_binds_the_composers_record_for_its_grid():
    summary = _child('''
        import json
        import numpy as np
        import gate_metal_no_pml_curl as gate
        from meep_gpu.chunk_map import ChunkLayout, chunk_bounds_record
        from meep_gpu.grid import Grid

        rows = []
        for cell, dimensions in (((1.25, 1.0, 0.75), 3), ((1.25, 1.0, 0.0), 2),
                                 ((0.0, 0.0, 2.0), 1)):
            grid = Grid(resolution=8.0, cell_size=cell, courant=0.35, xp=np,
                        dimensions=dimensions)
            invariant = tuple(grid.is_invariant(axis) for axis in range(3))
            layout = chunk_bounds_record(ChunkLayout.one_rank(grid, None))
            rows.append({"dimensions": dimensions,
                         "equal": bool(np.array_equal(layout, gate.case_record(invariant)))})
        kinds = sorted({tuple(invariant) for _shape, invariant, _codes in gate.CASE_GRIDS})
        planted = gate.build_arrays((3, 4, 5), 1, planted=True)
        zeros_only = all(not np.any(np.abs(planted[name]))
                         for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"))
        signs = {int(np.count_nonzero(np.signbit(planted["Dx"]))) > 0,
                 int(np.count_nonzero(~np.signbit(planted["Dx"]))) > 0}
        print(json.dumps({"rows": rows, "kinds": kinds,
                          "products": len(gate.product_cases()),
                          "planted": len(gate.planted_cases()),
                          "planted_zeros_only": bool(zeros_only),
                          "planted_both_signs": signs == {True}}))
    ''')
    assert all(row["equal"] for row in summary["rows"]), summary
    assert summary["kinds"] == [[False, False, False], [False, False, True],
                                [True, True, False]]
    assert summary["products"] == 60
    assert summary["planted"] == 8
    assert summary["planted_zeros_only"] and summary["planted_both_signs"]


def test_every_curl_mutation_lands_and_each_absent_arm_has_a_null_twin():
    summary = _child('''
        import json
        import gate_metal_no_pml_curl as gate

        sources = {name: (source, derive) for name, source, derive
                   in gate.mutation_sources()}
        base = gate.family.plain_curl_source((1, 0, 1), False, True)
        previous = sources["previous_curl_grouping"][0]
        rows = [{"label": label, "mutation": mutation, "grid": grid,
                 "must_catch": must_catch, "record": record}
                for label, mutation, grid, must_catch, record in gate.MUTATION_ROWS]
        print(json.dumps({
            "names": sorted(sources),
            "rows": rows,
            "changed": {name: source != base for name, (source, _d) in sources.items()},
            "previous_has_old": all(old in previous for _new, old in gate._BRACKETS),
            "previous_has_new": any(new in previous for new, _old in gate._BRACKETS),
            "base_has_new": all(new in base for new, _old in gate._BRACKETS),
        }))
    ''')
    assert summary["names"] == sorted([
        "add_instead_of_subtract", "previous_curl_grouping", "drop_derived_epsilon",
        "full_bracket_over_an_absent_partner",
        "absent_first_partner_keeps_the_sign_of_dtdx",
        "absent_first_partner_as_the_reversed_difference"])
    assert all(summary["changed"].values()), summary["changed"]
    assert summary["previous_has_old"] and not summary["previous_has_new"]
    assert summary["base_has_new"]
    rows = summary["rows"]
    known = set(summary["names"])
    assert all(row["mutation"] in known or row["record"] is not None for row in rows)
    for name in ("full_bracket_over_an_absent_partner",
                 "absent_first_partner_keeps_the_sign_of_dtdx"):
        armed = [row for row in rows if row["mutation"] == name]
        assert {(row["grid"], row["must_catch"]) for row in armed} == {
            ("2d", True), ("3d", False)}, armed
    armed = [row for row in rows
             if row["mutation"] == "absent_first_partner_as_the_reversed_difference"]
    assert {(row["grid"], row["must_catch"]) for row in armed} == {
        ("2d_planted", True)}, armed
    record_rows = [row for row in rows if row["record"] is not None]
    assert [row["label"] for row in record_rows] == ["presence_word_ignored"]
    assert record_rows[0]["must_catch"] and record_rows[0]["grid"] == "2d"


def test_every_plain_pair_mutation_lands_on_the_case_it_runs_on():
    summary = _child('''
        import json
        import gate_metal_no_pml_fused_electric_pairs as gate
        from meep_gpu import stepping
        from meep_gpu.metal_kernels.coverage import zero_metal_axes

        spec = gate.FAMILIES[0]
        assert spec["name"] == "no_pml_fused_electric_pair"
        rows = []
        for label, axis, old, new, *on in (tuple(spec["mutations"])
                                          + tuple(spec["equivalences"])):
            case = on[0] if on else spec["mutation_case"]
            fields, pml = gate.build_case(spec, case)
            codes = tuple(1 if kind == "metallic" else 0
                          for kind in stepping._boundary_kinds(fields.grid, None))
            walls = zero_metal_axes(fields.grid)
            source = gate.plain_family.no_pml_fused_electric_pair_source(
                codes, axis, 2, walls)
            mutated = gate.needle(source, old, new)
            rows.append({"label": label, "case": case, "changed": mutated != source,
                         "invariant": [fields.grid.is_invariant(a) for a in range(3)]})
        print(json.dumps(rows))
    ''')
    by_label = {row["label"]: row for row in summary}
    assert all(row["changed"] for row in summary), summary
    assert "curl_parens_flattened" not in by_label
    assert by_label["curl_previous_grouping"]["case"] == "wall_xyz_2poles_all"
    for label in ("full_bracket_over_an_absent_partner",
                  "absent_first_partner_keeps_the_sign_of_dtdx"):
        assert by_label[label]["invariant"] == [False, False, True], by_label[label]
