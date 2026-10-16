"""Host checks of the Metal PML gate: its hand reference, its form census, its needles.

The gate's device legs need an Apple GPU; three things it carries do not, and they are
what keeps the device legs meaningful between device runs:

* **the in-gate transcription is ``stepping.py``** (leg 0): the hand reference the
  device legs compare against steps the per-chunk forms and MEEP's groupings word for
  word, on the record each case binds;
* **every form is covered** (leg 9): each (component x form) class is nonzero, and
  each axis and lattice carries in-run and out-of-run rows;
* **every source mutation lands**: each needle is present in the source it is armed
  on, so no leg can be disarmed by a re-spelled kernel. The two grouping mutations are
  the INVERTED ones (KERNELS.md §7.1): the shipped form is MEEP's grouping, so each
  plants the package's previous grouping and must be caught.

The gate sets its subnormal policy in the environment when it is imported, so every
check runs it in a child process with torch made unimportable: the host legs are
torch-free, and this session's environment is untouched.
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


def test_the_hand_reference_is_stepping_and_every_form_is_covered(tmp_path):
    out = tmp_path / "gate_metal_pml_host.json"
    summary = _child(f'''
        import json
        import gate_metal_pml as gate
        payload = {{"legs": {{}}}}
        gate.leg_transcription(payload, {str(out)!r})
        gate.leg_forms(payload, {str(out)!r})
        print(json.dumps(payload["legs"], default=str))
    ''')
    rows = summary["transcription"]
    assert len(rows) == 6, rows
    for row in rows:
        assert row["moved"] > 0, row
        assert not row["differing"], row
    forms = summary["forms"]
    assert len(forms["classes"]) == 24, forms["classes"]
    assert all(count > 0 for count in forms["classes"].values()), forms["classes"]
    assert all(entry["in"] and entry["out"] for entry in forms["lines"].values())
    assert forms["wrapping_axes"] > 0


def test_every_source_mutation_lands_and_the_grouping_ones_plant_the_old_form():
    summary = _child('''
        import json
        import gate_metal_pml as gate
        from meep_gpu.metal_kernels import shaders
        report = {"curl": {}, "constitutive": {}}
        for name, entry in gate.CURL_MUTATIONS.items():
            codes = entry.get("codes", (gate.PERIODIC,) * 3)
            for sub_step in entry.get("sub_steps", ("step_B", "step_D")):
                backward = gate.SUB_STEPS[sub_step]["backward"]
                source = shaders.curl_source(codes, backward)
                mutated = entry["apply"](source)
                report["curl"][f"{name}/{sub_step}"] = {
                    "changed": mutated != source,
                    "old_grouping": "dtdx * ((c_y - c) + (b - b_z))" in mutated,
                    "must_catch": entry["must_catch"]}
        for name, entry in gate.CONSTITUTIVE_MUTATIONS.items():
            for side in ("H", "E"):
                source = shaders.constitutive_source(side)
                mutated = entry["apply"](source)
                report["constitutive"][f"{name}/{side}"] = {
                    "changed": mutated != source,
                    "old_grouping": "(f0[ii] + (kp_0 * src0)) - (km_0 * prev0)" in mutated,
                    "must_catch": entry["must_catch"]}
        print(json.dumps(report))
    ''')
    for family in ("curl", "constitutive"):
        assert summary[family], family
        for label, row in summary[family].items():
            assert row["changed"], f"{label}: the needle matched nothing in the shipped source"
    assert summary["curl"]["m1_previous_curl_grouping/step_B"]["old_grouping"]
    assert summary["curl"]["m1_previous_curl_grouping/step_D"]["old_grouping"]
    assert summary["curl"]["m1_previous_curl_grouping/step_B"]["must_catch"] is True
    for side in ("H", "E"):
        row = summary["constitutive"][f"m11_previous_accumulation_grouping/{side}"]
        assert row["old_grouping"] and row["must_catch"] is True
    # The two declared nulls of the form battery stay declared nulls.
    assert summary["curl"]["b7_auxiliary_load_unconditional/step_B"]["must_catch"] is False
    assert summary["constitutive"]["b12_constitutive_load_unconditional/H"][
        "must_catch"] is False


def test_the_record_mutations_change_the_record_on_every_case():
    summary = _child('''
        import json
        import numpy as np
        import gate_metal_pml as gate
        rows = {}
        for name, cell, boundaries in gate.CONFIGS:
            grid, _fields, pml = gate.build(cell, boundaries, 0.35, 20260814)
            truth = gate.layer_record(grid, pml)
            for label, (mutate, _why) in gate.RECORD_MUTATIONS.items():
                rows[f"{name}/{label}"] = bool(np.any(mutate(truth) != truth))
        print(json.dumps(rows))
    ''')
    assert summary and all(summary.values()), summary
