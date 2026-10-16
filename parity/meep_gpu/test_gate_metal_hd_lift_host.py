"""Host checks of the Metal H->D gates' lift accounting and of their census rebinding.

Two things every Metal H->D gate carries need no Apple GPU:

* **the lift's refusal accounting on the 0.10 source path**
  (``gate_metal_fused_hd_pair.lift_refusal_accounting``): a lifted row carrying a
  magnetic source is refused by ``launch.h_start_reasons`` and every other row is
  admitted, the seam record's ``withdraw_in_seam`` rows included. The rows are built
  here from the product's own refusal text, so each disposition the accounting must
  fail is shown failing;
* **a directly run gate does not stack its census**: a gate run as a script executes
  its module twice (``__main__``, then the runner's copy), and a sibling census that
  subclassed the first copy leaked one wrapper per use on the BFAST gate until its last
  leg raised ``RecursionError``. Each sibling now extends
  ``gate_metal_fused_hd_pair.BaseIntermediateCensus``.

The gates set their subnormal policy in the environment when imported, so every check
runs in a child process with torch made unimportable.
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


def test_the_lift_accounting_refuses_exactly_the_magnetic_rows_by_name():
    summary = _child('''
        import json, types
        import numpy
        import gate_metal_fused_hd_pair as shared
        from meep_gpu.metal_kernels import launch

        unseparated = types.SimpleNamespace(h_separated=False)
        magnetic = ["constitutive half: " + reason for reason in launch.h_start_reasons(
            unseparated, (types.SimpleNamespace(field_type="B"),))]
        holds_b = ["constitutive half: " + reason for reason in launch.h_start_reasons(
            types.SimpleNamespace(h_separated=False,
                                  Bx=numpy.ones(4, dtype=numpy.float32)),
            (types.SimpleNamespace(field_type="D"),))]
        chunk = ("curl half: 1 of 9 structure chunks keep the conductivity of Dx: MEEP "
                 "steps the lossless forms in the other chunks")
        withdraw = "fused half: a standing integrated electric withdraw"
        D = [{"field_type": "D", "is_integrated": False}]
        DI = [{"field_type": "D", "is_integrated": True}]
        DB = [{"field_type": "D", "is_integrated": False},
              {"field_type": "B", "is_integrated": False}]

        def row(admits, sources, reasons=()):
            block = {"predicate_admits": admits, "predicate_reasons": list(reasons)}
            if sources is not None:
                block["sources"] = sources
            return block

        cell = {"m": row(False, DB, magnetic), "d": row(True, D), "w": row(True, DI)}
        cases = {
            "the_0_10_cell": (cell, ["w"], ()),
            "a_magnetic_row_admitted": ({**cell, "m": row(True, DB)}, ["w"], ()),
            "an_electric_row_refused_for_live_B": (
                {**cell, "d": row(False, D, holds_b)}, ["w"], ()),
            "the_withdraw_row_refused_by_its_withdraw": (
                {**cell, "w": row(False, DI, [withdraw])}, ["w"], ()),
            "a_row_with_no_source_list": ({**cell, "d": row(True, None)}, ["w"], ()),
            "an_unrecognised_reason_beside_the_magnetic_one": (
                {**cell, "m": row(False, DB, magnetic + [chunk])}, ["w"], ()),
            "the_chunk_reason_declared_beside_the_magnetic_one": (
                {**cell, "m": row(False, DB, magnetic + [chunk])}, ["w"], ("structure chunks keep the conductivity of",)),
            "the_chunk_reason_alone_on_an_electric_row": (
                {**cell, "d": row(False, D, [chunk])}, ["w"], ("structure chunks keep the conductivity of",)),
        }
        out = {"lead": shared.h_start_lead(), "magnetic_reason": magnetic}
        for name, (blocks, withdraw_rows, further) in cases.items():
            accounting = shared.lift_refusal_accounting(blocks, withdraw_rows, further)
            out[name] = {key: accounting[key] for key in (
                "passed", "expected_refused_from_the_row_sources", "refused",
                "refused_by_h_start_reasons", "refused_by_a_standing_withdraw",
                "refusal_reasons_unrecognised", "rows_without_a_source_list",
                "of_those_refused_without_a_magnetic_source", "reason_tally")}
        print(json.dumps(out))
    ''')
    assert summary["lead"] and summary["lead"] in summary["magnetic_reason"][0]
    cell = summary["the_0_10_cell"]
    assert cell["passed"], cell
    assert cell["expected_refused_from_the_row_sources"] == ["m"] == cell["refused"]
    assert cell["refused_by_h_start_reasons"] == ["m"]
    assert cell["reason_tally"] == {"h_start_reasons: magnetic source": 1}
    for name in ("a_magnetic_row_admitted", "an_electric_row_refused_for_live_B",
                 "the_withdraw_row_refused_by_its_withdraw", "a_row_with_no_source_list",
                 "an_unrecognised_reason_beside_the_magnetic_one",
                 "the_chunk_reason_alone_on_an_electric_row"):
        assert summary[name]["passed"] is False, (name, summary[name])
    assert summary["the_withdraw_row_refused_by_its_withdraw"][
        "refused_by_a_standing_withdraw"] == ["w"]
    assert summary["the_withdraw_row_refused_by_its_withdraw"][
        "of_those_refused_without_a_magnetic_source"] == ["w"]
    assert summary["a_row_with_no_source_list"]["rows_without_a_source_list"] == ["d"]
    assert list(summary["an_unrecognised_reason_beside_the_magnetic_one"][
        "refusal_reasons_unrecognised"]) == ["m"]
    assert summary["the_chunk_reason_declared_beside_the_magnetic_one"]["passed"], (
        summary["the_chunk_reason_declared_beside_the_magnetic_one"])


def test_a_directly_run_gate_does_not_stack_or_leak_its_census():
    summary = _child('''
        import importlib.util, json, os, sys
        here = sys.path[0]
        out = {}
        for stem, helper, wrapper in (
                ("gate_metal_bfast_fused_hd_pair", "_bfast_term", "bfast"),
                ("gate_metal_conductive_fused_hd_pair", "_apply_conductive_pml_update",
                 "conductive")):
            for name in ("__main_copy_of_" + stem, "_metal_gate_runtime_" + stem + "_1"):
                spec = importlib.util.spec_from_file_location(
                    name, os.path.join(here, stem + ".py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            import gate_metal_fused_hd_pair as shared
            from meep_gpu import stepping

            def depth():
                function, levels = getattr(stepping, helper), 0
                while (getattr(function, "__name__", "") == wrapper
                       and function.__closure__):
                    cells = dict(zip(function.__code__.co_freevars, function.__closure__))
                    function = cells["original"].cell_contents
                    levels += 1
                return levels

            uses = []
            for _ in range(3):
                with shared.IntermediateCensus():
                    inside = depth()
                uses.append([inside, depth()])
            out[stem] = {
                "mro": [c.__module__ + "." + c.__name__
                        for c in shared.IntermediateCensus.__mro__],
                "base_is_the_plain_class": shared.IntermediateCensus.__mro__[1]
                is shared.BaseIntermediateCensus,
                "uses": uses}
        print(json.dumps(out))
    ''')
    for stem, row in summary.items():
        assert len(row["mro"]) == 3 and row["base_is_the_plain_class"], (stem, row)
        assert row["uses"] == [[1, 0]] * 3, (stem, row)
