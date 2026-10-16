"""Host checks of the Triton fused D->E gate (no device, no Triton).

The gate's mutation legs rewrite the shipped ``kernels.fused_curl_constitutive_D``
text and compile the copy on the device. Two harness failures read as results there:
a needle that matches nothing aborts the battery (or, for a count other than three,
arms a partial mutant), and a copy whose module lacks a name the kernel reads fails
to compile, which the legs score as a difference, so every must-differ mutation would
read caught while measuring nothing. Both are decided by text, so both are checked
here, against the shipped source read off ``kernels.py``.

The gate also installs, measures and stamps the campaign's subnormal policy
(``composition_policy.py``); its ``main`` is driven here with the device work stubbed
out and the policy module simulated.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import json
import sys
import textwrap
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(HERE), str(API_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import composition_policy as CP  # noqa: E402

KERNELS = API_ROOT / "meep_gpu" / "triton_kernels" / "kernels.py"
KERNEL_NAME = "fused_curl_constitutive_D"


def _load_gate():
    path = HERE / "gate_triton_fused_electric.py"
    spec = importlib.util.spec_from_file_location("gate_triton_fused_electric_host", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_gate()


def _shipped_source() -> str:
    text = KERNELS.read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name == KERNEL_NAME)
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return textwrap.dedent("\n".join(text.splitlines()[first - 1:node.end_lineno]))


def _free_names(function: ast.FunctionDef) -> set:
    """Names the function body reads that it neither binds nor takes as a parameter."""
    bound = {a.arg for a in function.args.args + function.args.kwonlyargs}
    loads = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            else:
                loads.add(node.id)
    return loads - bound - set(dir(builtins))


def _module_names(header: str) -> set:
    names = set()
    for node in ast.parse(header).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


@pytest.mark.parametrize("name", ["reload_D_from_memory", "reload_D_from_next_cell",
                                  "inverse_epsilon_on_left", "drop_fw_store",
                                  "regroup_constitutive"])
def test_every_source_mutation_arms_three_sites_and_still_parses(gate, name):
    assert name in gate.SOURCE_MUTATIONS
    shipped = _shipped_source()
    mutated, hits = gate.mutated_source(name, shipped)
    assert hits == 3, (name, hits)
    assert mutated != shipped, name
    ast.parse(mutated)


def test_the_mutation_list_is_the_one_main_scores(gate):
    text = (HERE / "gate_triton_fused_electric.py").read_text(encoding="utf-8")
    for name in gate.SOURCE_MUTATIONS:
        assert f'("{name}", ' in text and f', "{name}"),' in text, name


def test_the_inverted_constitutive_mutant_is_the_pre_0_10_grouping(gate):
    """KERNELS.md section 7.1: the shipped form is MEEP's, so the mutant is the old one."""
    mutated, _hits = gate.mutated_source("regroup_constitutive", _shipped_source())
    assert "constitutive_form(" not in mutated.split('"""')[-1]
    for t in range(3):
        assert (f"a{t} = tl.where(dw{t}, (tl.load(e{t} + idx, mask=live & dw{t}, "
                f"other=0.0) + kp_{t} * src{t}) - km_{t} * prev{t}, src{t})") in mutated


def test_the_mutated_copy_resolves_every_name_the_kernel_reads(gate, monkeypatch):
    """Every global the shipped kernel reads is defined by the copy's module header."""
    monkeypatch.setattr(gate, "_constant", lambda name: 0)
    header = gate.mutation_header()
    function = next(n for n in ast.walk(ast.parse(_shipped_source()))
                    if isinstance(n, ast.FunctionDef))
    missing = sorted(_free_names(function) - _module_names(header))
    assert not missing, f"the mutated copy would not compile: {missing} undefined"


# ---------------------------------------------------------------------------
# The campaign's subnormal policy: installed before anything is built, refused by name
# when it is not attained, stamped where every Triton reader looks, confirmed at the end
# ---------------------------------------------------------------------------
#
# The gate's main runs with its device work stubbed out (no CuPy, no Triton, no kernel)
# and ``meep_gpu.subnormal_policy`` simulated, as ``test_composition_policy.py`` does for
# the five composition gates. Until the 1.0 certification round this gate installed
# nothing, its ledger entry declared ``_mixed_policy``, and the two arms that cite it
# (``fused pair B`` and ``fused pair D``) could never carry flush evidence.

LEDGER = API_ROOT / "meep_gpu" / "triton_kernels" / "fingerprints.json"
WELD = "triton_fused_electric_device_gate"
STAMP_NAMES = {"keep": "ieee_keep_ftz_stripped", "flush": "meep_x86_flush"}
#: The two source mutations the gate scores as measured nulls (must stay identical).
NULL_MUTATIONS = {"reload_D_from_memory", "inverse_epsilon_on_left"}


def _stamp(policy: str, *, installed: bool = True,
           attained=("host", "cupy", "triton")) -> Dict[str, Any]:
    token = "ftz_stripped" if policy == "keep" else "flush"
    return {
        "policy": STAMP_NAMES[policy] if installed else "none_installed",
        "requested": policy, "resolved": policy, "installed": installed,
        "resolved_from": "environment", "match_meep": None,
        "cache_dir": f"/caches/cupy_cache_{token}_fused_electric",
        "executors": {name: {"executor": name, "requested": policy,
                             "attained": name in attained,
                             "reasons": [] if name in attained else [f"{name} refused"]}
                      for name in CP.EXECUTORS},
        "unattained": sorted(set(CP.EXECUTORS) - set(attained)),
    }


class _Policy:
    """Stands in for ``meep_gpu.subnormal_policy`` and the host FPU probe."""

    def __init__(self, refuse: Optional[str] = None, host_after: Optional[bool] = None,
                 unattained: tuple = ()):
        self.installed: Optional[str] = None
        self.refuse = refuse
        self.host_after = host_after
        self.unattained = unattained
        self.calls: List[str] = []
        self.stamps = 0

    def install(self, policy, cupy=None, strict=True, executors=()):
        from meep_gpu import subnormal_policy

        self.calls.append(policy)
        assert strict is True and tuple(executors) == CP.EXECUTORS
        if self.refuse:
            raise subnormal_policy.SubnormalPolicyUnattainable(self.refuse)
        self.installed = policy
        return self.stamp()

    def stamp(self):
        self.stamps += 1
        if self.installed is None:
            stamp = _stamp("keep", installed=False, attained=())
        else:
            stamp = _stamp(self.installed, attained=tuple(
                name for name in CP.EXECUTORS if name not in self.unattained))
        stamp["nvrtc_calls"] = self.stamps   # a counter that moves between stamps
        return stamp

    def resolved(self):
        return self.installed or "keep"

    def flushed(self):
        if self.host_after is not None:
            return self.host_after
        return self.installed == "flush"


@pytest.fixture
def stubbed(gate, monkeypatch, tmp_path):
    """The gate's main with its device work stubbed; returns a runner."""
    from meep_gpu import backends, subnormal_policy

    measured: List[str] = []

    def one_case(shape, boundaries, dtdx, source, repo_root, *, kernel=None,
                 mutation=None, num_warps=1):
        measured.append(mutation or kernel or source)
        identical = (mutation is None and kernel is None) or kernel in NULL_MUTATIONS
        verdict = {"bit_identical": identical, "differing_floats": 0 if identical else 7}
        return {"shape": list(shape), "vs_array": dict(verdict),
                "vs_two_kernels": dict(verdict), "two_kernels_vs_array": dict(verdict)}

    monkeypatch.setattr(gate, "cp", types.ModuleType("cupy"))
    monkeypatch.setattr(gate, "provenance", lambda num_warps: {"num_warps": num_warps})
    monkeypatch.setattr(gate, "one_case", one_case)
    monkeypatch.setattr(gate, "mutate_kernel",
                        lambda name: (name, str(tmp_path / f"{name}.py"), 3))

    def run(policy_env: Optional[str], **kwargs) -> Dict[str, Any]:
        state = _Policy(**kwargs)
        monkeypatch.setattr(subnormal_policy, "install_subnormal_policy", state.install)
        monkeypatch.setattr(subnormal_policy, "policy_stamp", state.stamp)
        monkeypatch.setattr(subnormal_policy, "get_subnormal_policy", state.resolved)
        monkeypatch.setattr(backends, "subnormals_flushed", state.flushed)
        if policy_env is None:
            monkeypatch.delenv("MEEP_GPU_SUBNORMAL_POLICY", raising=False)
        else:
            monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", policy_env)
        measured.clear()
        out = tmp_path / f"run_{len(list(tmp_path.glob('run_*')))}" / "gate.json"
        rc = gate.main(["--out", str(out)])
        return {"rc": rc, "artifact": json.loads(out.read_text(encoding="utf-8")),
                "measured": list(measured), "state": state}

    return run


def _shipped_entry() -> Dict[str, Any]:
    return json.loads(LEDGER.read_text(encoding="utf-8"))[WELD]


def test_the_gate_installs_before_it_builds_and_confirms_before_it_passes():
    source = (HERE / "gate_triton_fused_electric.py").read_text(encoding="utf-8")
    main = source[source.index("def main("):]
    assert main.count("stamp_campaign_policy(") == 1
    assert main.count("confirm_campaign_policy(") == 1
    install = main.index("stamp_campaign_policy(")
    assert install < main.index("provenance(args.num_warps)")
    assert install < main.index("one_case(")
    assert install < main.index("mutate_kernel(")
    assert main.index("confirm_campaign_policy(") < main.index('results["summary"] = {')


@pytest.mark.parametrize("policy", ["keep", "flush"])
def test_the_verdict_carries_the_attained_policy(gate, stubbed, policy):
    import gate_provenance
    import rebind_triton_welds
    from seed_triton_welds import policy_line

    run = stubbed(policy)
    artifact = run["artifact"]
    assert run["rc"] == 0, artifact.get("summary")
    assert run["state"].calls == [policy]
    assert run["measured"], "the gate measured nothing"
    assert artifact["summary"]["pass"] is True
    assert artifact["subnormal_policy"]["resolved"] == policy
    assert artifact["subnormal_policy"]["policy"] == STAMP_NAMES[policy]
    # Re-stamped at the end of the run, so the counters cover the run, not the install.
    assert artifact["subnormal_policy"]["nvrtc_calls"] == run["state"].stamps >= 2
    assert gate_provenance.read_verdict(artifact)["released"] is True
    assert policy_line(artifact).startswith(f"{policy} (installed by the gate")
    # Through the SHIPPED entry, as the rebind reads it: the run's policy is attained.
    assert rebind_triton_welds.attained_policy(artifact, _shipped_entry()) == policy


def test_an_unset_campaign_policy_installs_keep(stubbed):
    import rebind_triton_welds

    run = stubbed(None)
    assert run["rc"] == 0 and run["state"].calls == ["keep"]
    assert rebind_triton_welds.attained_policy(run["artifact"], _shipped_entry()) == "keep"


@pytest.mark.parametrize("planted", ["raised", "executor"])
def test_a_planted_policy_that_is_not_attained_refuses_before_anything_is_measured(
        stubbed, planted):
    import gate_provenance
    import rebind_triton_welds

    if planted == "raised":
        run = stubbed("flush", refuse="the 'flush' subnormal policy is UNATTAINABLE for "
                                      "the 'host' executor: MEEP is not imported in this "
                                      "process")
        needle = "UNATTAINABLE for the 'host' executor"
    else:
        run = stubbed("flush", unattained=("triton",))
        needle = "'triton' executor did not attain 'flush'"
    artifact = run["artifact"]
    assert run["rc"] == 2
    assert run["measured"] == []
    assert artifact["base_cases"] == [] and artifact["mutations"] == []
    assert artifact["summary"]["status"] == "refused"
    assert needle in artifact["summary"]["reasons"][0]
    assert gate_provenance.read_verdict(artifact)["released"] is False
    # What the rebind reads first: a run that is not released binds nothing, whatever
    # its stamp says (an executor that did not attain is in a stamp that installed).
    assert rebind_triton_welds._verdict(artifact)[0]["released"] is False
    if planted == "raised":
        assert rebind_triton_welds.attained_policy(artifact, _shipped_entry()) is None


def test_a_policy_a_record_cannot_claim_is_refused_by_name(stubbed):
    run = stubbed("match_meep")
    assert run["rc"] == 2 and run["measured"] == [] and run["state"].calls == []
    assert run["artifact"]["summary"]["status"] == "refused"
    assert "'match_meep'" in run["artifact"]["summary"]["reasons"][0]


def test_a_policy_that_did_not_hold_to_the_end_refuses(stubbed):
    import gate_provenance

    run = stubbed("keep", host_after=True)
    artifact = run["artifact"]
    assert run["rc"] == 2
    assert run["measured"], "the run should have measured before the end check"
    assert artifact["summary"]["status"] == "refused"
    assert gate_provenance.read_verdict(artifact)["released"] is False
    assert "host FPU is flushing" in json.dumps(artifact["summary"])


def test_the_weld_no_longer_declares_the_mixed_policy_exception():
    """The gate stamps its policy, so the entry may not declare that it installs none:
    ``seed_triton_welds.policy_line`` refuses a stamped run for an entry that does, and
    the rebind could then never record either policy for ``fused pair B`` / ``D``."""
    from seed_triton_welds import MIXED_POLICY_FIELD

    assert MIXED_POLICY_FIELD not in _shipped_entry()
