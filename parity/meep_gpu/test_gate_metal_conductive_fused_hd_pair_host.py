"""Host checks of the conductive Metal H->D weld's gate: its lift and its needles.

The gate's device legs need an Apple GPU; two things it carries do not:

* **the transcription leg**: the emitted weld differs from the two certified emitters
  (the conductive curl and ``fused_hd_pair.h_cell``) in exactly the declared edits, on
  the gate's boundary and lossy specialisations;
* **every armed mutation lands**: each needle (the shader, conductive, periodic-wrap
  and mixed tables, the predicted null and the byte-neutral control) is present in the
  source it is armed on and changes it. The grouping mutations are the INVERTED ones
  (KERNELS.md §7.1): each plants the package's previous grouping.

The gate sets its subnormal policy in the environment when it is imported, so every
check runs it in a child process with torch made unimportable.
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


def _child(body: str) -> dict:
    """Run ``body`` after the torch block; it prints one JSON object last."""
    environment = dict(os.environ)
    environment.pop("MEEP_GPU_SUBNORMAL_POLICY", None)
    result = subprocess.run([sys.executable, "-c", _NO_TORCH + textwrap.dedent(body)],
                            cwd=API_ROOT, env=environment, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_transcription_leg_passes_on_every_lossy_shape():
    summary = _child('''
        import json
        import gate_metal_conductive_fused_hd_pair as gate
        rows = {}
        for codes, lossy in (((1, 1, 1), (True, True, True)),
                             ((0, 0, 0), (True, True, True)),
                             (gate.MIXED_CODES, gate.conductive_flags(gate.MIXED_MUTATION_CASE))):
            row = gate.leg_transcription(codes, lossy)
            rows[f"{codes}/{lossy}"] = {
                "passed": row["passed"],
                "needles": row["every_mutation_needle_resolves"],
                "lift": row["constitutive_lift_edits_present"],
                "only_magnetic": row["every_change_is_a_redirected_magnetic_read"]}
        print(json.dumps(rows))
    ''')
    assert len(summary) == 3, summary
    for label, row in summary.items():
        assert row["passed"], (label, row)
        assert row["needles"] and all(row["needles"].values()), (label, row["needles"])


def test_every_armed_edit_lands_and_the_groupings_plant_the_old_forms():
    summary = _child('''
        import json
        import gate_metal_conductive_fused_hd_pair as gate
        from meep_gpu.metal_kernels import conductive_fused_hd_pair as family
        codes, lossy = (1, 1, 1), (True, True, True)
        base = family.conductive_fused_hd_pair_source(codes, lossy)
        changed = {}
        for name, (source, _why) in gate.shader_mutations(codes, lossy).items():
            changed[name] = source != base
        for name, (source, _why) in gate.conductive_mutations(codes, lossy).items():
            changed[name] = source != base
        for name, (source, _why, control) in gate.predicted_null_mutations(codes, lossy).items():
            changed[name] = source != base
        periodic = family.conductive_fused_hd_pair_source((0, 0, 0), lossy)
        for name, (source, _why) in gate.periodic_wrap_mutations((0, 0, 0), lossy).items():
            changed[name] = source != periodic
        mixed_args = (gate.MIXED_CODES, gate.conductive_flags(gate.MIXED_MUTATION_CASE))
        mixed = family.conductive_fused_hd_pair_source(*mixed_args)
        for name, (source, _why) in gate.mixed_mutations(*mixed_args).items():
            changed[name] = source != mixed
        changed["byte_neutral"] = gate.byte_neutral_source(codes, lossy) != base
        shader = gate.shader_mutations(codes, lossy)
        conductive = gate.conductive_mutations(codes, lossy)
        print(json.dumps({
            "changed": changed,
            "old_stencil": "dtdx * ((c_y - c) + (b - b_z))" in shader["curl_previous_grouping"][0],
            "old_accumulation": "(hi0[ii] + (kp_0 * src0)) - (km_0 * prev0)"
                in shader["constitutive_previous_accumulation_grouping"][0],
            "old_increment": "+ c0_new) - c0_previous) * si_y"
                in conductive["conductive_previous_increment_grouping"][0]}))
    ''')
    assert all(summary["changed"].values()), {
        name: hit for name, hit in summary["changed"].items() if not hit}
    for row in ("b1_select_removed", "b2_predicate_on_coefficient_identity",
                "b5_dsig_dsigu_swapped", "b6_f_cond_stored_off_the_dsig_cells",
                "b6_fu_stored_off_the_dsigu_cells"):
        assert row in summary["changed"], row
    assert summary["old_stencil"] and summary["old_accumulation"]
    assert summary["old_increment"]
