"""Host checks of the fold-carry pairs' gates: every mutation lands on the text it edits.

The three gates (``gate_metal_folded_fused_pair.py``,
``gate_metal_folded_fused_magnetic_pair.py`` and the ``folded_fused_dispersive_pair``
row of ``gate_metal_tranche7_fused_pairs.py``) need an Apple GPU to run; their mutation
TABLES do not, and a table whose needle stopped matching is a disarmed leg. So, on the
host:

* every shader edit applies to the source its mutation case emits (each gate's own
  ``needle`` refuses an absent anchor, and the magnetic gate's refuses an ambiguous
  one), and changes it;
* the two INVERTED grouping rows (KERNELS.md §7.1) plant the package's previous
  groupings over MEEP's, which the shipped source carries;
* the two record host mutations (the B battery's bound off by one and lattice swap)
  can arm on their mutation case's record.

The gates set their subnormal policy in the environment when imported, so every check
runs in a child process with torch made unimportable: the host checks are torch-free,
and this session's environment is untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

#: Put first in the child: torch cannot be imported, so a host check that reached for
#: it fails here by name rather than passing on a host that happens to have it.
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


#: The record mutations' check, shared by the two three-component gates. Indented as
#: the test bodies it is spliced into are, so the child's dedent sees one margin.
_RECORD_CHECK = '''
        import numpy as np
        from gate_chunk_records import layer_record
        fields, pml = gate.build(keywords, 1, *extra)
        record = layer_record(fields.grid, pml)
        widened = gate.widened_record(record, fields.grid.shape)
        swapped = gate.swapped_record(record)
        out["record_widened_words"] = int(np.count_nonzero(widened != record))
        out["record_swapped_words"] = int(np.count_nonzero(swapped != record))
'''


def test_the_d_pair_gates_mutations_all_land_and_the_groupings_are_inverted():
    summary = _child('''
        import json
        import gate_metal_folded_fused_pair as gate
        cases = dict(gate.CASES)
        keywords, extra = cases[gate.MUTATION_CASE], ()
        fields, pml = gate.build(keywords, 1)
        codes, phases, walls = gate._specialisation(fields, pml)
        base = gate.family.folded_fused_pair_source(codes, phases, walls)
        edits = gate.shader_edits(codes, phases, walls)
        out = {"names": sorted(edits),
               "unchanged": sorted(n for n, s in edits.items() if s == base),
               "base_has_meep_curl": "dtdx * (((c_y - c) + b) - b_z)" in base,
               "old_curl_planted": "dtdx * ((c_y - c) + (b - b_z))"
                                   in edits["curl_previous_grouping"],
               "base_has_meep_constitutive":
                   "oacc = e0[ii] + ((kp_0 * osrc) - (km_0 * oprev));" in base,
               "old_constitutive_planted":
                   "oacc = (e0[ii] + (kp_0 * osrc)) - (km_0 * oprev);"
                   in edits["constitutive_previous_grouping"],
               "host": sorted(gate.HOST_MUTATIONS)}
    ''' + _RECORD_CHECK + '''
        print(json.dumps(out))
    ''')
    assert not summary["unchanged"], summary["unchanged"]
    assert len(summary["names"]) == 21, summary["names"]
    assert summary["base_has_meep_curl"] and summary["old_curl_planted"]
    assert summary["base_has_meep_constitutive"] and summary["old_constitutive_planted"]
    assert {name for name in summary["names"] if name.startswith("b")} >= {
        "b1_curl_select_removed", "b2_curl_select_on_the_half_integer_lattice",
        "b3_dsig_and_dsigu_swapped", "b4_fu_stored_on_a_plain_cell",
        "b5_constitutive_select_removed",
        "b6_constitutive_select_on_the_integer_lattice",
        "b7_fw_stored_on_a_plain_cell"}
    assert "curl_parens_flattened" not in summary["names"]
    assert summary["host"] == sorted([
        "e_half_takes_the_integer_lattice", "curl_takes_the_half_integer_lattice",
        "b8_record_bound_off_by_one", "b9_record_lattices_swapped"])
    assert summary["record_widened_words"] == 1
    assert summary["record_swapped_words"] > 0


def test_the_d_pair_gates_far_ghost_keying_is_observable_where_the_record_says():
    """The keyed-at-source edit lands once, and the first fixture is the observable one:
    the far ghost and its reflect row take different forms there, the second agrees."""
    summary = _child('''
        import json
        import gate_metal_folded_fused_pair as gate
        out = {}
        for name, keywords, faces in gate.KEYED_CASES:
            fields, pml = gate.build(keywords, 1, pml_faces=faces)
            codes, phases, walls = gate._specialisation(fields, pml)
            base = gate.family.folded_fused_pair_source(codes, phases, walls)
            membership = gate.far_ghost_in_run(fields, pml)
            out[name] = {"anchor": base.count(gate.FAR_KEYED_EDIT[0]),
                         "observable": membership["destination"] != membership["source"],
                         "covered": [r for r in gate.family.metal_folded_fused_pair_coverage(
                             fields, pml, ()).reasons
                             if "subnormal" not in r and "residency" not in r
                             and "torch is not importable" not in r]}
        print(json.dumps(out))
    ''')
    names = [name for name in summary]
    assert summary[names[0]]["observable"] is True, summary
    assert summary[names[1]]["observable"] is False, summary
    for name, row in summary.items():
        assert row["anchor"] == 1, (name, row)
        assert row["covered"] == [], (name, row)


def test_the_b_pair_gates_mutations_all_land_on_all_three_mutation_cases():
    summary = _child('''
        import json
        import gate_metal_folded_fused_magnetic_pair as gate
        cases = gate.cases_table()
        out = {"cases": {}}
        for label, periodic in ((gate.MUTATION_CASE, True),
                                (gate.METALLIC_MUTATION_CASE, False),
                                (gate.TRIPLE_MUTATION_CASE, True)):
            keywords, faces = cases[label]
            fields, pml = gate.build(keywords, 1, faces)
            codes, phases, walls = gate._specialisation(fields, pml)
            base = gate.family.folded_fused_magnetic_pair_source(codes, phases, walls)
            if label == gate.TRIPLE_MUTATION_CASE:
                edits = {**gate.carried_shader_edits(base), **gate.triple_shader_edits(base)}
                absent = {name: base.count(gate.shader_edit_anchors(True)[name][0])
                          for name in gate.TRIPLE_STRUCTURALLY_ABSENT}
            else:
                edits = gate.shader_edits(base, periodic)
                absent = {}
            out["cases"][label] = {
                "count": len(edits),
                "unchanged": sorted(n for n, s in edits.items() if s == base),
                "absent_anchor_counts": absent,
                "old_curl_planted": "dtdx * ((c_y - c) + (b - b_z))"
                                    in edits["curl_previous_grouping"],
                "base_has_meep_curl": "dtdx * (((c_y - c) + b) - b_z)" in base}
        keywords, faces = cases[gate.MUTATION_CASE]
        fields, pml = gate.build(keywords, 1, faces)
        codes, phases, walls = gate._specialisation(fields, pml)
        base = gate.family.folded_fused_magnetic_pair_source(codes, phases, walls)
        out["far_coefficient_anchor"] = base.count(gate.FAR_COEFFICIENT_EDIT[0])
        out["reflect_row_anchor"] = base.count(gate.REFLECT_ROW_EDIT[0])
        out["moved_coefficient_anchors"] = {
            name: base.count(edit[0][0])
            for name, edit in {**gate.MOVED_COEFFICIENT_EDITS,
                               **gate.KEYED_AT_SOURCE_EDITS}.items()}
        deep_keywords, deep_faces = cases[gate.DEEP_CASE]
        deep_fields, deep_pml = gate.build(deep_keywords, 1, deep_faces)
        out["deep_near_ghost"] = gate.near_ghost_in_run(deep_fields, deep_pml)
        out["predicted"] = {name: list(labels)
                            for name, labels in gate.PREDICTED_MUTATIONS.items()}
        extra = (faces,)
    ''' + _RECORD_CHECK + '''
        print(json.dumps(out))
    ''')
    for label, row in summary["cases"].items():
        assert not row["unchanged"], (label, row["unchanged"])
        assert row["old_curl_planted"] and row["base_has_meep_curl"], label
        assert all(count == 0 for count in row["absent_anchor_counts"].values()), row
    assert summary["far_coefficient_anchor"] == 1
    assert summary["reflect_row_anchor"] == 1
    assert set(summary["moved_coefficient_anchors"].values()) == {1}
    # The keyed-at-source edit is OBSERVABLE on the deep fixture: the near ghost is
    # outside the run at stored 0 and its source inside it at stored 2.
    assert summary["deep_near_ghost"] == {"destination": False, "source": True}
    assert set(summary["predicted"]) == {
        "near_ghost_fw_store_dropped", "triple_composite_fw_store_dropped",
        "destination_thread_keeps_its_displacement"}
    assert summary["record_widened_words"] == 1
    assert summary["record_swapped_words"] > 0


def test_the_tranche7_dispersive_rows_land_on_their_targets():
    summary = _child('''
        import json
        import gate_metal_tranche7_fused_pairs as gate
        from meep_gpu.metal_kernels.dispersive_update_e import E_TERMS, _poles
        from meep_gpu.triton_kernels.coverage import zero_metal_axes
        from meep_gpu.triton_kernels.symmetry import folded_axis_kinds
        spec = [s for s in gate.FAMILIES if s["name"] == "folded_fused_dispersive_pair"][0]
        build = dict(spec["cases"])[spec["mutation_case"]]
        fields, pml = build()
        codes, _ = folded_axis_kinds(fields.grid, pml)
        phases = tuple(int(fields.grid.mirror_phase(axis) or 0) for axis in range(3))
        walls = zero_metal_axes(fields.grid)
        driven = _poles(fields)
        poles = {axis: len(driven[component]) for component, _d, axis, _n in E_TERMS}
        rows = {}
        for label, target, old, new in spec["mutations"] + spec["equivalences"]:
            source = spec["module"].folded_fused_dispersive_pair_source(
                codes, int(target), poles[int(target)], phases, walls)
            rows[label] = {"count": source.count(old), "changes": old != new}
        print(json.dumps({"rows": rows}))
    ''')
    rows = summary["rows"]
    for label in ("curl_previous_grouping", "constitutive_previous_grouping",
                  "b1_curl_select_removed", "b5_constitutive_select_removed",
                  "drop_the_pole_subtraction", "near_ghost_reads_the_owner_poles",
                  "far_ghost_loses_its_parity"):
        assert label in rows, sorted(rows)
    for label, row in rows.items():
        assert row["count"] == 1 and row["changes"], (label, row)
