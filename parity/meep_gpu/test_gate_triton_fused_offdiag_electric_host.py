"""Host checks of the D->E off-diagonal refutation gate's assembly (no device, no Triton).

The gate splices the shipped ``kernels.fused_curl_constitutive_D`` curl half and the
shipped ``offdiag_update_e.offdiag_constitutive_step`` body into one subject kernel,
and rebuilds the standalone off-diagonal kernel with its neighbour indices made
pointwise. Each module is compiled on the device, where a name the spliced text reads
and its header does not define fails every leg at launch. The splice, its header and
every rewrite are text, so they are checked here against the shipped sources.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import re
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def _load_gate():
    path = HERE / "gate_triton_fused_offdiag_electric.py"
    spec = importlib.util.spec_from_file_location(
        "gate_triton_fused_offdiag_electric_host", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_gate()


def _free_names(function: ast.FunctionDef) -> set:
    bound = {a.arg for a in function.args.args + function.args.kwonlyargs}
    loads = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            else:
                loads.add(node.id)
    return loads - bound - set(dir(builtins))


def _module_names(tree: ast.Module) -> set:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _unresolved(source: str, function_name: str) -> list:
    tree = ast.parse(source)
    function = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == function_name)
    return sorted(_free_names(function) - _module_names(tree))


def test_the_splice_is_verbatim_and_the_subject_resolves_every_name(gate):
    source, record = gate.build_subject_source()
    assert record["splice_is_verbatim_after_reversing_both_edits"] is True
    assert not _unresolved(source, "fused_curl_offdiag_constitutive_D")


def test_the_subject_binds_the_record_and_the_presence_the_bodies_read(gate):
    """Both shipped bodies read ``cb``; the off-diagonal body reads ``op`` and ``P*``."""
    source, _record = gate.build_subject_source()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    params = {a.arg for a in function.args.args}
    for name in ("cb", "op", "PRESENCE", "P01", "P02", "P11", "P12", "P21", "P22"):
        assert name in params, name


@pytest.mark.parametrize("mutation", ["flatten_curl_parens", "mispair_row_coefficients",
                                      "reload_own_D_from_memory"])
def test_every_source_mutation_rewrites_the_subject_and_still_resolves(gate, mutation):
    base, _ = gate.build_subject_source()
    mutated, record = gate.build_subject_source(mutation=mutation)
    assert mutated != base and record["mutation"] == mutation
    assert not _unresolved(mutated, "fused_curl_offdiag_constitutive_D")


def test_the_curl_mutant_is_the_pre_0_10_pairwise_grouping(gate):
    """KERNELS.md section 7.1: the shipped stencil is MEEP's, so the mutant is the old one."""
    mutated, _ = gate.build_subject_source(mutation="flatten_curl_parens")
    assert "curl0 = dtdx * ((c_y - c) + (b - b_z))" in mutated
    assert "curl0 = curl_stencil(" not in mutated


def test_the_pointwise_plant_rewrites_both_routes_the_same_way(gate):
    _source, record = gate.build_subject_source(pointwise_variant=True)
    planted = record["planted"]["lines_rewritten"]
    body = textwrap.dedent(gate._function_source(
        gate.kernel_dir() / "offdiag_update_e.py", "offdiag_constitutive_step"))
    standalone = re.sub(r"^(\s+)\w+ \* nyz \+ \w+ \* nz \+ \w+,$", r"\1idx,", body,
                        flags=re.M)
    rewritten = sum(1 for a, b in zip(body.splitlines(), standalone.splitlines()) if a != b)
    assert planted == rewritten > 0
    assert not _unresolved(gate.STANDALONE_HEADER + standalone, "offdiag_constitutive_step")
