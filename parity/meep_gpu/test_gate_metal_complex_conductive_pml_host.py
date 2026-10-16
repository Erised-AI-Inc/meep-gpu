"""Host checks of the Metal complex conductive PML gate: census, needles, refusal fixture.

The gate's device legs need an Apple GPU; what it carries that does not is what keeps
those legs meaningful between device runs:

* **every form is reached**: over the gate's five products, each of MEEP's four
  conductive forms and both lossless forms covers at least one cell on the record
  each product binds, so no mutation of a form is vacuous;
* **every source mutation lands**: each needle is present in the source it is armed
  on and changes it, so no leg can be disarmed by a re-spelled kernel. The two
  grouping mutations are the INVERTED ones (KERNELS.md §7.1): the shipped stencil and
  ``f_cond`` increment are MEEP's groupings, so each plants the package's previous
  grouping and must be caught; the one declared null is the hoisted ``f_cond`` load;
* **the record mutations change the record** the gate's mutation case binds;
* **a chunk that keeps no conductivity is refused by name** on the refusal leg's
  fixtures, which really are partial presences, on both sides.

The gate sets its subnormal policy in the environment when it is imported, so every
check runs it in a child process with torch made unimportable: the host checks are
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


def test_every_form_is_reached_over_the_products():
    summary = _child('''
        import json
        import gate_metal_complex_conductive_pml as gate
        products = (
            (("B", "D"), False, (0.2, -0.125, 0.0), ("periodic",) * 3),
            (("D",), False, (0.2, -0.125, 0.0), ("periodic",) * 3),
            (("B",), False, (0.2, 0.0, 0.0), ("periodic",) * 3),
            (("B", "D"), True, (0.2, -0.125, 0.0), ("periodic",) * 3),
            (("B", "D"), False, (0.0, 0.125, 0.0), ("metallic", "periodic", "metallic")),
        )
        reached = {}
        for index, (sides, mixed, phase, boundaries) in enumerate(products, 1):
            fields, pml = gate.build(sides=sides, mixed=mixed, seed=18000 + index,
                                     phase=phase, boundaries=boundaries)
            for slot in ("step_B", "step_D"):
                for key, count in gate.form_counts(fields, pml, slot).items():
                    form = key.split("/", 1)[1]
                    reached[form] = reached.get(form, 0) + count
        print(json.dumps({"reached": reached, "forms": list(gate.FORMS)}))
    ''')
    assert len(summary["forms"]) == 6, summary
    for form in summary["forms"]:
        assert summary["reached"].get(form, 0) > 0, (form, summary)


def test_every_source_mutation_lands_and_the_groupings_plant_the_old_forms():
    summary = _child('''
        import json
        import gate_metal_complex_conductive_pml as gate
        from meep_gpu.metal_kernels import complex_conductive_pml as family
        kinds = gate._mutation_kinds()
        full = family.complex_conductive_pml_curl_source(
            (0, 0, 0), False, kinds, (True, True, True))
        mixed = family.complex_conductive_pml_curl_source(
            (0, 0, 0), False, kinds, (True, False, False))
        report = {}
        for name, entry in gate.mutation_table().items():
            shipped = mixed if entry["mixed"] else full
            report[name] = {
                "changed": entry["source"] != shipped,
                "must_catch": entry["must_catch"],
                "old_stencil": "(c_y - c_dy) + (b_dz - b_z)" in entry["source"],
                "old_increment": entry["source"].count("+ hn0) - hp0, si_y)"),
                "shipped_increment": shipped.count("+ hs0, si_y)")}
        print(json.dumps(report))
    ''')
    assert len(summary) >= 16, sorted(summary)
    for label, row in summary.items():
        assert row["changed"], f"{label}: the needle matched nothing in the shipped source"
    assert summary["curl_previous_grouping"]["old_stencil"]
    assert summary["curl_previous_grouping"]["must_catch"] is True
    increment = summary["increment_previous_grouping"]
    assert increment["old_increment"] == 2 and increment["shipped_increment"] == 2
    assert increment["must_catch"] is True
    nulls = [name for name, row in summary.items() if row["must_catch"] is False]
    assert nulls == ["b7_history_load_unconditional"], nulls


def test_the_record_mutations_change_the_record():
    summary = _child('''
        import json
        import numpy as np
        import gate_metal_complex_conductive_pml as gate
        fields, pml = gate.build(sides=("B", "D"), mixed=False, seed=19301)
        truth = gate.layer_record(fields.grid, pml)
        print(json.dumps({label: int(np.count_nonzero(mutate(truth) != truth))
                          for label, (mutate, _why) in gate.RECORD_MUTATIONS.items()}))
    ''')
    assert sorted(summary) == ["b3_bound_off_by_one", "b4_lattices_swapped"]
    assert all(count > 0 for count in summary.values()), summary


def test_the_refusal_legs_fixtures_are_partial_and_refused_by_name():
    summary = _child('''
        import json
        import gate_metal_complex_conductive_pml as gate
        from meep_gpu.metal_kernels.conductive_pml import conductive_presence_reasons
        out = {}
        for side, slot, target in gate.PRESENCE_CASES:
            fields, pml, kept = gate.partial_presence(side, slot, target)
            out[target] = {"kept": list(kept), "reasons": conductive_presence_reasons(
                fields, pml, slot)}
        print(json.dumps(out))
    ''')
    assert sorted(summary) == ["Bx", "Dz"], summary
    for target, row in summary.items():
        assert any(row["kept"]) and not all(row["kept"]), (target, row)
        assert any(f"structure chunks keep the conductivity of {target}" in reason
                   for reason in row["reasons"]), (target, row)
