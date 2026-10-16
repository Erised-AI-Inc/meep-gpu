"""Host checks of the Metal conductive PML gate: its form census, its needles, its refusal.

The gate's device legs need an Apple GPU; what it carries that does not is what keeps
those legs meaningful between device runs:

* **every form is reached**: over the gate's six products, each of MEEP's four
  conductive forms, both lossless forms, and both lossless forms on a LOSSY target
  whose chunk keeps no conductivity (the two partial-presence products) covers at
  least one cell on the record each product binds, so no mutation of a form is vacuous;
* **every source mutation lands**: each needle is present in the source it is armed
  on and changes it, so no leg can be disarmed by a re-spelled kernel. The two
  grouping mutations are the INVERTED ones (KERNELS.md §7.1): the shipped stencil and
  ``f_cond`` increment are MEEP's groupings, so each plants the package's previous
  grouping and must be caught;
* **the record mutations change the record** the gate's mutation case binds;
* **a chunk that keeps no conductivity is served** through the presence words on the
  refusal leg's fixture, which really is a partial presence, and **a record that
  cannot carry that presence is refused by name**; the presence mutation changes the
  words the plan would bind.

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


def test_every_form_is_reached_over_the_products():
    summary = _child('''
        import json
        import gate_metal_conductive_pml as gate
        products = (
            (("B", "D"), False, ("periodic", "periodic", "periodic"), False),
            (("B",), False, ("metallic", "periodic", "metallic"), False),
            (("D",), False, ("periodic", "metallic", "periodic"), False),
            (("B", "D"), True, ("metallic", "periodic", "metallic"), False),
            (("B", "D"), False, ("metallic", "periodic", "metallic"), True),
            (("B", "D"), True, ("periodic", "metallic", "periodic"), True),
        )
        reached = {}
        for index, (sides, mixed, boundaries, partial) in enumerate(products, 1):
            fields, pml = gate.build(sides=sides, mixed=mixed, boundaries=boundaries,
                                     seed=30100 + index, partial=partial)
            for slot in gate._sub_steps(sides):
                for key, count in gate._form_counts(fields, pml, slot).items():
                    form = key.split("/", 1)[1]
                    reached[form] = reached.get(form, 0) + count
        print(json.dumps(reached))
    ''')
    for form in ("both", "dsigu_only", "dsig_only", "neither", "lossless_recurrence",
                 "lossless_dsig_only", "unkept_recurrence", "unkept_dsig_only"):
        assert summary.get(form, 0) > 0, (form, summary)


def test_every_source_mutation_lands_and_the_grouping_ones_plant_the_old_form():
    summary = _child('''
        import json
        import gate_metal_conductive_pml as gate
        from meep_gpu.metal_kernels import conductive_pml
        full = conductive_pml.conductive_pml_curl_source((1, 0, 1), False,
                                                         (True, True, True))
        mixed = conductive_pml.conductive_pml_curl_source((1, 0, 1), False,
                                                          (True, False, False))
        report = {}
        for name, entry in gate._mutation_table().items():
            shipped = mixed if entry["mixed"] else full
            report[name] = {
                "changed": entry["source"] != shipped,
                "must_catch": entry["must_catch"],
                "old_stencil": "dtdx * ((c_y - c) + (b - b_z))" in entry["source"],
                "old_increment": "+ c0_new) - c0_previous) * si_y" in entry["source"]}
        print(json.dumps(report))
    ''')
    assert len(summary) >= 14, sorted(summary)
    assert summary["b9_presence_flag_inverted"]["must_catch"] is True
    for label, row in summary.items():
        assert row["changed"], f"{label}: the needle matched nothing in the shipped source"
    assert summary["curl_previous_grouping"]["old_stencil"]
    assert summary["curl_previous_grouping"]["must_catch"] is True
    assert summary["increment_previous_grouping"]["old_increment"]
    assert summary["increment_previous_grouping"]["must_catch"] is True
    assert summary["b7_history_load_unconditional"]["must_catch"] is False
    nulls = [name for name, row in summary.items() if row["must_catch"] is False]
    assert nulls == ["b7_history_load_unconditional"], nulls


def test_the_record_mutations_change_the_record():
    summary = _child('''
        import json
        import numpy as np
        import gate_metal_conductive_pml as gate
        fields, pml = gate.build(sides=("B",), mixed=False,
                                 boundaries=("metallic", "periodic", "metallic"),
                                 seed=30301)
        truth = gate.layer_record(fields.grid, pml)
        print(json.dumps({label: bool(np.any(mutate(truth) != truth))
                          for label, (mutate, _why) in gate.RECORD_MUTATIONS.items()}))
    ''')
    assert sorted(summary) == ["b3_bound_off_by_one", "b4_lattices_swapped"]
    assert all(summary.values()), summary


def test_the_refusal_legs_partial_presence_is_served_and_an_uncarried_record_refused():
    summary = _child('''
        import json
        import numpy as np
        import gate_metal_conductive_pml as gate
        from meep_gpu.chunk_map import ChunkLayout, record_index
        from meep_gpu.metal_kernels import conductive_pml as family
        fields, pml = gate.build(sides=(), mixed=False,
                                 boundaries=("periodic", "periodic", "periodic"),
                                 seed=30302)
        sigma = np.zeros(fields.grid.shape, dtype=np.float32)
        sigma[4:6] = np.float32(0.15)
        fields.set_b_conductivity({"Bx": sigma, "By": None, "Bz": None})
        kept = ChunkLayout.one_rank(fields.grid, pml).conductive_chunks(sigma, (0, 1, 1))
        record = gate.layer_record(fields.grid, pml)
        wrong = np.array(record, copy=True)
        wrong[record_index(False, 0, 1)] += 2
        print(json.dumps({
            "kept": list(kept),
            "served": family.presence_reasons(fields, pml, "step_B", chunk_bounds=record),
            "words": list(family.presence_for(fields, pml, "step_B", record)),
            "uncarried": family.presence_reasons(fields, pml, "step_B",
                                                 chunk_bounds=wrong)}))
    ''')
    assert any(summary["kept"]) and not all(summary["kept"]), summary
    assert summary["served"] == [], summary
    assert 0 < summary["words"][0] < (1 << 27) - 1, summary
    assert summary["uncarried"] and "conductivity presence is refused" in \
        summary["uncarried"][0], summary


def test_the_presence_mutation_changes_the_words_the_plan_binds():
    summary = _child('''
        import json
        import gate_metal_conductive_pml as gate
        from meep_gpu.metal_kernels import conductive_pml as family
        fields, pml = gate.build(sides=("B",), mixed=False,
                                 boundaries=("metallic", "periodic", "metallic"),
                                 seed=30401, partial=True)
        truth = family.presence_for(fields, pml, "step_B",
                                    gate.layer_record(fields.grid, pml))
        print(json.dumps({label: list(words) != list(truth)
                          for label, (words, _why) in gate.PRESENCE_MUTATIONS.items()}))
    ''')
    assert sorted(summary) == ["b10_presence_words_all_kept"]
    assert all(summary.values()), summary
