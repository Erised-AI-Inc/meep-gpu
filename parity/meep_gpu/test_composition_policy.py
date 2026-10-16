"""The five Triton composition gates install, measure and stamp the campaign's policy.

Owner ruling R16: the composition gates the 11 routing arms cite install the subnormal
policy their campaign names before they build anything, refuse by name when it is not
attained, and stamp it where every Triton writer reads a policy, so a flush composition
run carries per-arm flush evidence like every other gate (``composition_policy.py``).

Three halves, all host-only (no GPU, no CuPy, no Triton, no MEEP run, no ledger):

* the helper's rules, on stamps shaped like ``subnormal_policy.policy_stamp()``, and
  once on the real install with nothing but the host present;
* each gate's ``main`` with its device work stubbed out and the policy module
  simulated: the verdict carries the stamp under keep and under flush, a policy that is
  not attained (at the install or at the end of the run) refuses, and nothing is
  measured after a refusal;
* the re-cut tool on those artifacts: ``recut_composition_records.second_policy_leg``
  takes flush evidence from a stamped flush verdict and refuses the unstamped shape
  every composition gate wrote before R16.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

HERE = Path(__file__).resolve().parent
API = HERE.parents[1]
for _path in (str(HERE), str(API)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import composition_policy as CP  # noqa: E402

#: ledger key -> (probe file, leg directory the fleet writes, verdict field)
GATES = {
    "cylindrical_composition_gate": ("probe_triton_cylindrical_composition.py",
                                     "cylindrical_composition", "summary"),
    "symmetry_composition_gate": ("probe_triton_symmetry_composition.py",
                                  "symmetry_composition", "summary"),
    "conductivity_composition_gate": ("probe_triton_conductivity_composition.py",
                                      "conductivity_composition", "summary"),
    "dispersive_composition_gate": ("probe_triton_engine_route.py", "engine_route",
                                    "summary"),
    "no_pml_composition_gate": ("gate_triton_no_pml.py", "no_pml", "validation"),
}

STAMP_NAMES = {"keep": "ieee_keep_ftz_stripped", "flush": "meep_x86_flush"}


# ---------------------------------------------------------------------------
# Which gates, and that each one calls the helper
# ---------------------------------------------------------------------------


def test_the_five_gates_are_the_composition_records_the_arms_cite():
    """Every composition record an arm cites is one of the five, and the re-cut agrees."""
    from meep_gpu import fastpath

    import recut_composition_records as R

    cited = {gate for _family, gate in fastpath.ARM_CERTIFICATION.values()
             if gate.endswith("_composition_gate")}
    assert cited == set(GATES)
    arms = sorted(arm for arm, (_family, gate) in fastpath.ARM_CERTIFICATION.items()
                  if gate in GATES)
    assert len(arms) == 11, arms
    for key, (probe, leg, _verdict) in GATES.items():
        legs, probe_rel = R.LEGS[key]
        assert probe_rel == f"parity/meep_gpu/{probe}"
        assert leg in legs


@pytest.mark.parametrize("key", sorted(GATES))
def test_each_gate_installs_before_it_builds_and_confirms_before_it_passes(key):
    probe, _leg, verdict = GATES[key]
    source = (HERE / probe).read_text(encoding="utf-8")
    assert source.count("stamp_campaign_policy(") == 1, probe
    assert ("confirm_campaign_policy(" in source) or ("end_of_run_reasons(" in source)
    main = source[source.index("def main("):]
    # The install comes before the first case and before the first passing verdict.
    first_work = min(i for i in (main.find("run_case("), main.find("one_case_with_whole_step("),
                                 main.find("leg_product("))
                     if i >= 0)
    assert main.index("stamp_campaign_policy(") < first_work, probe
    if verdict == "validation":
        assert 'verdict="validation"' in main


# ---------------------------------------------------------------------------
# The helper's rules
# ---------------------------------------------------------------------------


def _stamp(policy: str, *, installed: bool = True, attained=("host", "cupy", "triton"),
           resolved: str | None = None) -> Dict[str, Any]:
    token = "ftz_stripped" if policy == "keep" else "flush"
    return {
        "policy": STAMP_NAMES[policy] if installed else "none_installed",
        "requested": policy, "resolved": resolved or policy, "installed": installed,
        "resolved_from": "argument", "match_meep": None,
        "cache_dir": f"/caches/cupy_cache_{token}_gate",
        "executors": {name: {"executor": name, "requested": policy,
                             "attained": name in attained,
                             "reasons": [] if name in attained else [f"{name} refused"]}
                      for name in CP.EXECUTORS},
        "unattained": sorted(set(CP.EXECUTORS) - set(attained)),
    }


@pytest.mark.parametrize("value, expected", [
    (None, "keep"), ("", "keep"), ("keep", "keep"), ("flush", "flush"), (" FLUSH ", "flush")])
def test_the_requested_policy_is_the_campaigns_or_keep(value, expected):
    environ = {} if value is None else {"MEEP_GPU_SUBNORMAL_POLICY": value}
    assert CP.requested_policy(environ) == expected


@pytest.mark.parametrize("value", ["match_meep", "ieee", "0"])
def test_a_policy_a_record_cannot_claim_is_refused_by_name(value):
    with pytest.raises(CP.CampaignPolicyRefused, match=repr(value)):
        CP.requested_policy({"MEEP_GPU_SUBNORMAL_POLICY": value})


def test_an_attained_stamp_has_no_reasons():
    assert CP.attainment_reasons(_stamp("keep"), "keep") == []
    assert CP.attainment_reasons(_stamp("flush"), "flush") == []


@pytest.mark.parametrize("stamp, policy, needle", [
    (_stamp("keep", installed=False), "keep", "no subnormal policy is installed"),
    (_stamp("keep", resolved="flush"), "keep", "resolved to 'flush'"),
    (_stamp("flush", attained=("cupy", "triton")), "flush", "'host' executor did not attain"),
])
def test_an_unattained_stamp_is_refused_by_name(stamp, policy, needle):
    reasons = CP.attainment_reasons(stamp, policy)
    assert reasons and any(needle in reason for reason in reasons), reasons


def test_an_executor_that_was_never_driven_is_a_reason():
    stamp = _stamp("keep")
    del stamp["executors"]["triton"]
    assert any("'triton' executor was never driven" in r
               for r in CP.attainment_reasons(stamp, "keep"))


def test_the_real_install_attains_keep_on_a_host_with_nothing_else(monkeypatch):
    """The stamp shape the helper checks is the one the policy module writes."""
    if importlib.util.find_spec("cupy") or importlib.util.find_spec("triton"):
        pytest.skip("a device stack is importable here; this leg is the host-only one")
    from meep_gpu import backends, subnormal_policy
    from seed_triton_welds import policy_line

    if backends.subnormals_flushed():
        pytest.skip("this host is flushing before any install")
    subnormal_policy._reset_for_tests()
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "keep")
    try:
        payload: Dict[str, Any] = {}
        assert CP.stamp_campaign_policy(payload, None) is None
        assert payload["subnormal_policy"]["installed"] is True
        assert payload["subnormal_policy"]["resolved"] == "keep"
        assert policy_line(payload).startswith("keep (installed by the gate")
        assert CP.confirm_campaign_policy(payload) is None
    finally:
        subnormal_policy._reset_for_tests()


# ---------------------------------------------------------------------------
# Each gate's main, with its device work stubbed out
# ---------------------------------------------------------------------------


#: What happened, in order: "install" for each policy install, "meep" for the first
#: import of MEEP (a gate that lifts MEEP simulations must import it BEFORE the install,
#: because importing MEEP flushes the host).
EVENTS: List[str] = []


class _MeepFinder:
    """Serves an empty ``meep`` module and records when it was first imported."""

    def find_spec(self, name, path=None, target=None):
        if name != "meep":
            return None
        return importlib.util.spec_from_loader(name, self)

    def create_module(self, spec):
        EVENTS.append("meep")
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        return None


class _Policy:
    """Stands in for ``meep_gpu.subnormal_policy`` and the host FPU probe."""

    def __init__(self, refuse: str | None = None, host_after: bool | None = None):
        self.installed: str | None = None
        self.refuse = refuse
        self.host_after = host_after
        self.calls: List[str] = []
        self.stamps = 0

    def install(self, policy, cupy=None, strict=True, executors=()):
        from meep_gpu import subnormal_policy

        self.calls.append(policy)
        EVENTS.append("install")
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
            stamp = _stamp(self.installed)
        stamp["nvrtc_calls"] = self.stamps   # a counter that moves between stamps
        return stamp

    def resolved(self):
        return self.installed or "keep"

    def flushed(self):
        if self.host_after is not None:
            return self.host_after
        return self.installed == "flush"


@pytest.fixture
def device_free(monkeypatch):
    """A fake CuPy/Triton/MEEP and a simulated policy module; returns a configurer."""
    from meep_gpu import backends, subnormal_policy

    cupy = types.ModuleType("cupy")
    cupy.__version__ = "0-test"
    runtime = types.SimpleNamespace(
        getDevice=lambda: 0,
        getDeviceProperties=lambda index: {"name": b"test device", "major": 8, "minor": 6})
    cupy.cuda = types.SimpleNamespace(runtime=runtime)
    triton = types.ModuleType("triton")
    triton.__version__ = "0-test"
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "triton", triton)
    EVENTS.clear()
    # Set aside whatever ``meep`` this process holds, and take back out at teardown
    # the empty module the finder serves during the test. ``delitem`` alone records
    # nothing when the key is absent, so the stand-in stayed in ``sys.modules`` for
    # the rest of the session: every later ``import meep`` got a module with no
    # ``Vector3``, with or without MEEP installed. ``setitem`` first records the
    # entry as it was; teardown undoes the two in reverse, ending at that state.
    monkeypatch.setitem(sys.modules, "meep", None)
    monkeypatch.delitem(sys.modules, "meep")
    monkeypatch.setattr(sys, "meta_path", [_MeepFinder(), *sys.meta_path])
    monkeypatch.setattr(backends, "guard_kernel_compilation", lambda cp: True)

    def configure(policy_env: str | None, **kwargs) -> _Policy:
        state = _Policy(**kwargs)
        monkeypatch.setattr(subnormal_policy, "install_subnormal_policy", state.install)
        monkeypatch.setattr(subnormal_policy, "policy_stamp", state.stamp)
        monkeypatch.setattr(subnormal_policy, "get_subnormal_policy", state.resolved)
        monkeypatch.setattr(backends, "subnormals_flushed", state.flushed)
        if policy_env is None:
            monkeypatch.delenv("MEEP_GPU_SUBNORMAL_POLICY", raising=False)
        else:
            monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", policy_env)
        return state

    return configure


def _load(probe: str):
    spec = importlib.util.spec_from_file_location(f"cpa_{probe[:-3]}", HERE / probe)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _engine_case(module, name, kwargs, covered):
    case = {"case": name, "kwargs": kwargs, "covered": covered, "refusals": {},
            "builder_matches_its_predicate": True, "replaces": []}
    if covered:
        case.update(bit_identical=True, per_step=[{}] * module.STEPS,
                    whole_step={"bit_identical": True,
                                "per_step": [{}] * module.WHOLE_STEPS})
    else:
        case.update(plan_is_none=True)
    return case


def _run_gate(key: str, tmp_path: Path, monkeypatch) -> Dict[str, Any]:
    """Run one gate's main with its measurements stubbed; returns rc, artifact, measured."""
    probe, _leg, _verdict = GATES[key]
    module = _load(probe)
    measured: List[str] = []
    out = tmp_path / key / "gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if key == "cylindrical_composition_gate":
        def run_case(cp, name, cell, resolution, component, steps):
            measured.append(name)
            return {"case": name, "steps": steps}
        monkeypatch.setattr(module, "run_case", run_case)
        monkeypatch.setattr(sys, "argv", [probe, "--out", str(out)])
        rc = module.main()
    elif key in ("symmetry_composition_gate", "conductivity_composition_gate"):
        def run_case(cp, name, *rest):
            measured.append(name)
            return {"case": name, "steps": rest[-1] if key.startswith("symmetry")
                    else rest[1]}
        monkeypatch.setattr(module, "run_case", run_case)
        rc = module.main(["--out", str(out)])
    elif key == "dispersive_composition_gate":
        covered = {name for name, _kwargs in module.COVERED_CASES}

        def one_case(name, kwargs):
            measured.append(name)
            return _engine_case(module, name, kwargs, name in covered)
        monkeypatch.setattr(module, "one_case_with_whole_step", one_case)
        rc = module.main(["--out", str(out)])
    else:
        planted = {"cases": [{"bit_identical": True}], "identical": "1/1",
                   "negative_zero_words": 1, "positive_zero_words": 1}
        engine = {"covered": [{"bit_identical": True, "first_divergent_step": None,
                               "steps": 12,
                               "replaces": ["step_B", "update_H", "step_D", "update_E"],
                               "selected": {"step_B": "PlainCurlPlan",
                                            "step_D": "PlainCurlPlan"}}],
                  "refusals": [{"plan_is_none": {"step_B": True, "step_D": True},
                                "central_selected": {"step_B": "PmlCurlPlan"}}]}

        def leg_planted(cp):
            measured.append("planted")
            return planted

        def leg_engine(cp, steps, jsonl=None):
            measured.append("engine")
            return engine
        monkeypatch.setattr(module, "environment", lambda: {})
        monkeypatch.setattr(module, "leg_planted", leg_planted)
        monkeypatch.setattr(module, "leg_engine", leg_engine)
        out = tmp_path / key / "gate.json"
        rc = module.main(["--out-dir", str(out.parent), "--leg", "planted",
                          "--leg", "engine"])
    return {"rc": rc, "artifact": json.loads(out.read_text(encoding="utf-8")),
            "measured": measured, "path": out}


@pytest.mark.parametrize("policy", ["keep", "flush"])
@pytest.mark.parametrize("key", sorted(GATES))
def test_the_verdict_carries_the_attained_policy(key, policy, device_free, tmp_path,
                                                 monkeypatch):
    import recut_composition_records as R
    from seed_triton_welds import policy_line

    state = device_free(policy)
    run = _run_gate(key, tmp_path, monkeypatch)
    artifact = run["artifact"]
    assert run["rc"] == 0, artifact.get(GATES[key][2])
    assert state.calls == [policy]
    assert run["measured"], "the gate measured nothing"
    assert artifact["subnormal_policy"]["resolved"] == policy
    assert artifact["subnormal_policy"]["policy"] == STAMP_NAMES[policy]
    # Re-stamped at the end of the run, so the counters cover the run, not the install.
    assert artifact["subnormal_policy"]["nvrtc_calls"] == state.stamps >= 2
    assert R.released(artifact)[0] is True
    assert R.attained_policy(artifact) == policy
    assert policy_line(artifact).startswith(f"{policy} (installed by the gate")


#: The gates whose cases lift a MEEP simulation (``cases.BUILDERS``).
LIFTS_MEEP = {"conductivity_composition_gate", "dispersive_composition_gate",
              "no_pml_composition_gate"}


@pytest.mark.parametrize("key", sorted(GATES))
def test_a_gate_that_lifts_meep_imports_it_before_the_install(key, device_free, tmp_path,
                                                              monkeypatch):
    """Importing MEEP flushes the host, so after a keep install it would undo it."""
    device_free("keep")
    run = _run_gate(key, tmp_path, monkeypatch)
    assert run["rc"] == 0
    if key in LIFTS_MEEP:
        assert EVENTS[:2] == ["meep", "install"], EVENTS
    else:
        assert EVENTS == ["install"], EVENTS


@pytest.mark.parametrize("key", sorted(GATES))
def test_an_unset_campaign_policy_installs_keep(key, device_free, tmp_path, monkeypatch):
    import recut_composition_records as R

    state = device_free(None)
    run = _run_gate(key, tmp_path, monkeypatch)
    assert run["rc"] == 0 and state.calls == ["keep"]
    assert R.attained_policy(run["artifact"]) == "keep"


@pytest.mark.parametrize("key", sorted(GATES))
def test_an_unattainable_policy_refuses_before_anything_is_measured(
        key, device_free, tmp_path, monkeypatch):
    import recut_composition_records as R

    reason = ("the 'flush' subnormal policy is UNATTAINABLE for the 'host' executor: "
              "MEEP is not imported in this process")
    device_free("flush", refuse=reason)
    run = _run_gate(key, tmp_path, monkeypatch)
    artifact = run["artifact"]
    verdict = artifact[GATES[key][2]]
    assert run["rc"] == 2
    assert run["measured"] == []
    assert verdict["status"] == "refused"
    assert "UNATTAINABLE for the 'host' executor" in verdict["reasons"][0]
    assert R.released(artifact)[0] is False
    assert R.attained_policy(artifact) is None


@pytest.mark.parametrize("key", sorted(GATES))
def test_a_policy_that_did_not_hold_to_the_end_refuses(key, device_free, tmp_path,
                                                       monkeypatch):
    import recut_composition_records as R

    device_free("keep", host_after=True)
    run = _run_gate(key, tmp_path, monkeypatch)
    artifact = run["artifact"]
    assert run["rc"] != 0
    assert run["measured"], "the run should have measured before the end check"
    assert R.released(artifact)[0] is False
    reasons = json.dumps(artifact[GATES[key][2]])
    assert "host FPU is flushing" in reasons


# ---------------------------------------------------------------------------
# The re-cut tool reads the stamp
# ---------------------------------------------------------------------------


def _flush_run(tmp_path: Path, artifact: Dict[str, Any], leg: str) -> Path:
    """A flush composition root holding one leg, with the staged manifest the round writes."""
    import recut_composition_records as R

    run = tmp_path / "triton_composition_flush"
    (run / leg).mkdir(parents=True)
    (run / leg / "gate.json").write_text(json.dumps(artifact), encoding="utf-8")
    (run / "staged_source_sha256.txt").write_text(
        f"{'1' * 64}  {R.pinned_path('launch.py')}\n", encoding="utf-8")
    return run


@pytest.mark.parametrize("key", sorted(GATES))
def test_the_recut_takes_flush_evidence_from_a_stamped_flush_verdict(
        key, device_free, tmp_path, monkeypatch):
    import recut_composition_records as R

    device_free("flush")
    artifact = _run_gate(key, tmp_path / "gate", monkeypatch)["artifact"]
    leg = GATES[key][1]
    run = _flush_run(tmp_path, artifact, leg)
    monkeypatch.setattr(R, "leg_capability", lambda payload, dirs: ("8.6", {}))
    evidence, why = R.second_policy_leg(run, "results/flush", R.LEGS[key][0], "8.6",
                                        ["launch.py"], {"launch.py": "1" * 64}, "flush")
    assert why == "" and evidence is not None
    assert evidence["verdict_read_from"].startswith(f"results/flush/{leg}/gate.json:")


@pytest.mark.parametrize("key", sorted(GATES))
def test_the_recut_refuses_the_unstamped_verdict_every_gate_wrote_before(
        key, device_free, tmp_path, monkeypatch):
    import recut_composition_records as R

    device_free("flush")
    artifact = _run_gate(key, tmp_path / "gate", monkeypatch)["artifact"]
    del artifact["subnormal_policy"]
    leg = GATES[key][1]
    run = _flush_run(tmp_path, artifact, leg)
    monkeypatch.setattr(R, "leg_capability", lambda payload, dirs: ("8.6", {}))
    evidence, why = R.second_policy_leg(run, "results/flush", R.LEGS[key][0], "8.6",
                                        ["launch.py"], {"launch.py": "1" * 64}, "flush")
    assert evidence is None and "attained None" in why


def test_the_recut_refuses_a_keep_verdict_offered_as_flush(device_free, tmp_path,
                                                           monkeypatch):
    import recut_composition_records as R

    device_free("keep")
    key = "cylindrical_composition_gate"
    artifact = _run_gate(key, tmp_path / "gate", monkeypatch)["artifact"]
    run = _flush_run(tmp_path, artifact, GATES[key][1])
    monkeypatch.setattr(R, "leg_capability", lambda payload, dirs: ("8.6", {}))
    evidence, why = R.second_policy_leg(run, "results/flush", R.LEGS[key][0], "8.6",
                                        ["launch.py"], {"launch.py": "1" * 64}, "flush")
    assert evidence is None and "attained 'keep'" in why
