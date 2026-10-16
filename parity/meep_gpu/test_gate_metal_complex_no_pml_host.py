"""Host checks of the two Metal complex no-absorber curl gates: cases, records, needles.

The gates' device legs need an Apple GPU; what they carry besides the kernel does not,
and it is what keeps the device legs meaningful between device runs:

* **every case's record is the composer's**, and its presence word is the grid's
  invariance (3-D, 2-D, 1-D, and a 3-D grid storing one cell on a live axis);
* **the planted cases can score**: the array path leaves negative zeros in the
  reference on every planted case, which is what the planted rows require;
* **every source mutation lands**: each needle is present exactly once in the source
  it is armed on (both gates), the grouping row plants the package's previous
  pairwise bracket (KERNELS.md §7.1), and each absent-partner row has a declared-null
  twin on a grid where the planted arm never runs.

Each check runs in a child process with torch made unimportable: the host legs are
torch-free, and the gates' own environment defaults stay out of this session.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

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


def _child(body: str):
    environment = dict(os.environ)
    environment.pop("MEEP_GPU_SUBNORMAL_POLICY", None)
    result = subprocess.run([sys.executable, "-c", _NO_TORCH + textwrap.dedent(body)],
                            cwd=API_ROOT, env=environment, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_each_case_binds_the_composers_record_and_presence_is_invariance():
    rows = _child('''
        import json
        import numpy as np
        import complex_no_pml_gate_common as common
        from meep_gpu.chunk_map import RECORD_PRESENCE_WORD, ChunkLayout, chunk_bounds_record
        from meep_gpu.pml import PML

        rows = []
        for case in common.CASES:
            fields, pml = common.build(case, seed=3)
            record = common.case_record(fields, pml)
            composer = chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))
            invariant = [bool(fields.grid.is_invariant(axis)) for axis in range(3)]
            presence = int(record[RECORD_PRESENCE_WORD])
            rows.append({"case": case, "equal": bool(np.array_equal(record, composer)),
                         "presence": presence,
                         "expected": sum(1 << a for a in range(3) if not invariant[a]),
                         "runs": int(np.count_nonzero(record[:RECORD_PRESENCE_WORD])),
                         "shape": list(fields.grid.shape),
                         "all_present": int(common.all_axes_present(record)[-1])})
        print(json.dumps(rows))
    ''')
    assert len(rows) == 10
    for row in rows:
        assert row["equal"], row
        assert row["presence"] == row["expected"], row
        assert row["runs"] == 0, row
        assert row["all_present"] == 7, row
    presences = {row["case"]: row["presence"] for row in rows}
    assert presences["collapsed_y"] == 7 and presences["planar_2d_bloch"] == 3
    assert presences["line_1d_z"] == 4


def test_the_array_path_leaves_negative_zeros_on_every_planted_case():
    rows = _child('''
        import json
        import complex_no_pml_gate_common as common
        from meep_gpu import stepping

        rows = []
        for label, case in common.PLANTED_CASES:
            fields, pml = common.build(case, seed=5, planted=True)
            stepping.step_B(fields, pml)
            stepping.step_D(fields, pml)
            rows.append({"label": label, "negative_zeros": sum(
                common.negative_zeros(getattr(fields, name))
                for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"))})
        print(json.dumps(rows))
    ''')
    assert len(rows) == 6
    for row in rows:
        assert row["negative_zeros"] > 0, row


def test_every_mutation_lands_once_and_each_absent_arm_has_a_null_twin():
    summary = _child('''
        import json
        import gate_metal_complex_no_pml_curl as lossless
        import gate_metal_complex_no_pml_conductive as conductive

        out = {}
        for name, gate, shipped in (
                ("lossless", lossless, lossless._shipped_b_source),
                ("conductive", conductive, conductive._shipped_b_source)):
            rows = []
            for label, case, source, expectation, planted, present in gate.mutation_rows():
                rows.append({"label": label, "case": case, "expectation": expectation,
                             "planted": planted, "present": present,
                             "changed": source != shipped(case)})
            out[name] = rows
        print(json.dumps(out))
    ''')
    for name, rows in summary.items():
        labels = {row["label"]: row for row in rows}
        assert "previous_curl_grouping" in labels, name
        assert "curl_grouping_flattened" not in labels, name
        for row in rows:
            if row["present"]:
                assert not row["changed"], row  # the shipped kernel, a wrong record
            else:
                assert row["changed"], row
        for label in ("full_bracket_over_an_absent_partner",
                      "absent_first_partner_keeps_the_sign_of_dtdx"):
            assert labels[label]["expectation"] == "caught"
            twin = labels[label + "_on_a_resolved_grid"]
            assert twin["expectation"] == "null"
            assert twin["case"] in ("two_axis_bloch",)
        assert labels["absent_first_partner_as_the_reversed_difference"]["planted"]
        assert labels["presence_word_ignored"]["present"]
    assert {row["label"] for row in summary["conductive"]} >= {
        "condfac_condinv_swapped", "target_store_dropped"}
    assert "curl_subtraction_reversed" in {row["label"] for row in summary["lossless"]}


def test_the_grouping_row_plants_the_previous_pairwise_bracket():
    summary = _child('''
        import json
        import gate_metal_complex_no_pml_curl as gate

        source = dict((row[0], row[2]) for row in gate.mutation_rows())[
            "previous_curl_grouping"]
        print(json.dumps({
            "pairwise": "c_mul_coefficient_left(dtdx, (c_y - c_dy) + (b_dz - b_z))" in source,
            "meep": "c_mul_coefficient_left(dtdx, ((c_y - c_dy) + b_dz) - b_z)" in source}))
    ''')
    assert summary == {"pairwise": True, "meep": False}
