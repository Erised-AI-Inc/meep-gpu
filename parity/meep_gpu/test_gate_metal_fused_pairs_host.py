"""Host checks of the Metal same-cell fused-pair gates: their needles and their tables.

The five gates (``gate_metal_fused_magnetic_pair``, ``gate_metal_fused_electric_pair``,
``gate_metal_fused_dispersive_pair``, ``gate_metal_fused_ade_chain`` and the nonlinear
product of ``gate_metal_below_the_cut_fused_pairs``) need an Apple GPU for their device
legs. What they ARM does not, and it is what keeps those legs meaningful between device
runs:

* **every source mutation and every battery row lands**: each needle is present in the
  shipped source it is armed on and changes it, so no leg is disarmed by a re-spelled
  kernel. The grouping mutations are the INVERTED ones (KERNELS.md §7.1): the shipped
  forms are MEEP's groupings, so each plants the package's previous grouping;
* **the battery keeps the base gate's declarations**: two declared nulls, one row on the
  +-0 lattice, the rest must-catch;
* **the text-only legs pass**: the transcription legs and the E->P chain's single-guard
  leg read emitted sources only.

Each check runs in a child process, because a gate sets its subnormal policy in the
environment when it is imported. Kernel compilation is replaced in the child by a stub
that hands back the source, so nothing reaches a device.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))

#: Put first in the child: the paths, and a compile stub in every module that compiles.
_PRELUDE = textwrap.dedent('''
    import sys
    sys.path[:0] = [{here!r}, {root!r}]

    class _Compiled:
        """What ``compile_source`` hands back here: every entry point is the source."""

        def __init__(self, source):
            self.source = source

        def __getattr__(self, name):
            return self.source

    def _stub(*modules):
        for module in modules:
            module.compile_source = lambda source: _Compiled(source)
''').format(here=HERE, root=API_ROOT)


def _child(body: str) -> dict:
    """Run ``body`` after the prelude; it prints one JSON object last."""
    environment = dict(os.environ)
    environment.pop("MEEP_GPU_SUBNORMAL_POLICY", None)
    result = subprocess.run([sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
                            cwd=API_ROOT, env=environment, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_lifting_pairs_arm_every_mutation_and_every_battery_row():
    summary = _child('''
        import json
        import gate_metal_fused_magnetic_pair as magnetic
        import gate_metal_fused_electric_pair as electric
        from meep_gpu.metal_kernels import fused_electric_pair, fused_magnetic_pair
        _stub(magnetic, electric)
        report = {}
        for name, gate, family, source_of in (
                ("magnetic", magnetic, fused_magnetic_pair,
                 fused_magnetic_pair.fused_magnetic_pair_source),
                ("electric", electric, fused_electric_pair,
                 fused_electric_pair.fused_electric_pair_source)):
            codes, walls = (1, 1, 1), (True, True, True)
            shipped = source_of(codes, walls)
            mutants = gate.shader_mutations(codes, walls)
            rows = gate.battery_rows(codes, walls)
            report[name] = {
                "mutations": {label: table["off"] != shipped
                              for label, table in mutants.items()},
                "previous_curl": "dtdx * ((c_y - c) + (b - b_z))"
                                 in mutants["curl_previous_grouping"]["off"],
                "previous_accumulation": ("(" + ("h0" if name == "magnetic" else "e0")
                                          + "[ii] + (kp_0 * src0)) - (km_0 * prev0)")
                                         in mutants["constitutive_previous_grouping"]["off"],
                "battery": {label: {"changed": next(iter(row["functions"].values())) != shipped,
                                    "must_catch": row["must_catch"],
                                    "value_class": row["value_class"]}
                            for label, row in rows.items()},
                "transcription": gate.leg_transcription(codes, walls)["passed"],
            }
        print(json.dumps(report))
    ''')
    for name in ("magnetic", "electric"):
        report = summary[name]
        assert report["mutations"] and all(report["mutations"].values()), report["mutations"]
        assert report["previous_curl"] and report["previous_accumulation"], report
        battery = report["battery"]
        assert len(battery) == 11 and all(row["changed"] for row in battery.values())
        nulls = sorted(label for label, row in battery.items() if row["must_catch"] is False)
        assert nulls == ["b12_constitutive_load_unconditional",
                         "b7_auxiliary_load_unconditional"], nulls
        lattice = sorted(label for label, row in battery.items()
                         if row["value_class"] == "pm_zero_lattice")
        assert lattice == ["b13_plain_constitutive_through_the_accumulation"], lattice
        assert report["transcription"] is True


def test_the_dispersive_pair_arms_every_mutation_and_every_battery_row():
    summary = _child('''
        import json
        import gate_metal_fused_dispersive_pair as gate
        from meep_gpu.metal_kernels import fused_dispersive_pair as family
        _stub(gate, family)
        codes, poles, walls = (1, 1, 1), 3, (True, True, True)
        shipped = family.fused_dispersive_pair_source(codes, 0, poles, walls)
        mutants = gate.shader_mutations(codes, poles, walls)
        rows = gate.battery_rows(codes, poles, walls)
        key = ("off", 0)
        print(json.dumps({
            "mutations": {label: table[key] != shipped for label, table in mutants.items()},
            "previous_curl": "dtdx * ((c_y - c) + (b - b_z))"
                             in mutants["curl_previous_grouping"][key],
            "previous_accumulation": "(e_out[idx] + (kps[i] * src)) - (kms[i] * prev)"
                                     in mutants["pml_previous_accumulation_grouping"][key],
            "battery": {label: {"changed": row["functions"][key] != shipped,
                                "must_catch": row["must_catch"],
                                "value_class": row["value_class"]}
                        for label, row in rows.items()},
        }))
    ''')
    assert summary["mutations"] and all(summary["mutations"].values()), summary["mutations"]
    assert summary["previous_curl"] and summary["previous_accumulation"]
    battery = summary["battery"]
    assert len(battery) == 11 and all(row["changed"] for row in battery.values())
    assert sorted(label for label, row in battery.items() if row["must_catch"] is False) == [
        "b12_constitutive_load_unconditional", "b7_auxiliary_load_unconditional"]


def test_the_dispersive_gate_floors_movement_on_the_auxiliary_contract():
    """An auxiliary whose form region is empty under a case's division must NOT move;
    the narrow-z case has no z layer, so fu_Dx/fu_Bx and f_w_Ez/f_w_Hz stay put."""
    summary = _child('''
        import json
        import gate_metal_fused_dispersive_pair as gate
        name, cell, boundaries, poles, kind = gate._case("narrow_z_two_pole")
        _grid, fields, pml = gate.build(cell, boundaries, poles, kind, 1)
        census = gate.battery.form_census(fields.grid, pml)
        print(json.dumps({name: forms for name, forms in census.items()}))
    ''')
    assert summary["Dx"]["recurrence"] == 0 and summary["Ez"]["accumulate"] == 0
    assert summary["Dy"]["recurrence"] and summary["Ey"]["accumulate"]


def test_the_ade_chain_arms_every_row_and_its_text_legs_pass():
    summary = _child('''
        import json
        import gate_metal_fused_ade_chain as gate
        _stub(gate, gate.family)
        case = gate._case(gate.MUTATION_CASE)
        # The compiled shape of the mutation case, without planning (planning binds
        # mirrors on a device): every component drives every pole, each with a
        # volume sigma.
        shape = tuple((axis, case.poles, (case.sigma_volume,) * case.poles)
                      for axis in range(3))
        axis, poles, flags = next(row for row in shape if row[0] == 0)
        shipped = gate.family.fused_ade_chain_source(case.arm, axis, poles, flags)
        key = ("off", 0)
        mutants = gate.shader_mutations(case, shape)
        rows = gate.battery_rows(case, shape)
        print(json.dumps({
            "mutations": {label: table[key] != shipped for label, table in mutants.items()},
            "battery": {label: {"changed": row["functions"][key] != shipped,
                                "must_catch": row["must_catch"],
                                "lattice": row["lattice"]}
                        for label, row in rows.items()},
            "single_guard": gate.leg_single_guard()["passed"],
            "transcription": gate.leg_transcription()["passed"],
        }))
    ''')
    assert summary["mutations"] and all(summary["mutations"].values()), summary["mutations"]
    assert "pml_previous_accumulation_grouping" in summary["mutations"]
    assert "ade_previous_folded_three_term_form" in summary["mutations"]
    battery = summary["battery"]
    assert len(battery) == 5 and all(row["changed"] for row in battery.values())
    assert [label for label, row in battery.items() if row["must_catch"] is False] == [
        "b12_constitutive_load_unconditional"]
    assert [label for label, row in battery.items() if row["lattice"]] == [
        "b13_plain_constitutive_through_the_accumulation"]
    assert summary["single_guard"] is True and summary["transcription"] is True


def test_the_nonlinear_product_arms_the_shared_set_on_the_aligned_body():
    summary = _child('''
        import json
        import gate_metal_below_the_cut_fused_pairs as gate
        product = next(p for p in gate.PRODUCTS if p.key == "nonlinear")
        fields, pml = gate.build(product, gate.case_keywords(product, product.mutation_case),
                                 1)
        codes, walls = gate._specialisation(fields, pml)
        shipped = product.source(codes, walls)
        edits = gate.shared_mutations(product, codes, walls)
        print(json.dumps({
            "mutations": {label: text != shipped for label, text in edits.items()},
            "transcription": gate.leg_transcription(product, codes, walls)["passed"],
        }))
    ''')
    assert summary["mutations"] and all(summary["mutations"].values()), summary["mutations"]
    assert "curl_previous_grouping" in summary["mutations"]
    assert "constitutive_previous_grouping" in summary["mutations"]
    assert summary["transcription"] is True
