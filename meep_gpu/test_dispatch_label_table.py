"""Which ledger answers for an arm label: the table rung 8b and the dispatch record read.

A namespaced label (``cuda:``) answers for itself. A bare label in a plan composed on
the CUDA table is a Triton arm, because every CUDA label crosses into ``fastpath``
namespaced (``fastpath_cuda.merge_tables`` prefixes each CUDA unit, the primary's
included), so it is answered from the Triton ledger. A bare label on any other table is
that table's own: Triton on a CuPy host, Metal on a NumPy host with an MPS device.

WHAT IS PINNED HERE:

* the table itself (:func:`fastpath._table_for_label`) and the lookup it routes
  (:func:`fastpath._certification_for`);
* rung 8b under a caller-installed ``flush`` on a CUDA-led merged plan
  (``MEEP_GPU_BACKEND_PREFERENCE=cuda``): bare Triton singles and Triton fused pairs are
  judged on the Triton ledger's run, so they are admitted where that run released under
  flush and refused, naming the Triton gate, where it did not. That includes the five
  bare labels the CUDA composer also spells (``PML``, ``ordinary``, ``off-diagonal``,
  ``folded off-diagonal``, ``mirror fill``): their same-named CUDA entry never answers
  for the Triton arm;
* the keep path. Rung 8b asks no per-arm question under the install default, and the
  record's per-arm certification is read off the merged plan's ``backends``, so the
  admission and every record field are the same under this table and under the lookup it
  replaced (:func:`_previous_lookup`, kept here as the reference those tests compare
  against).

Nothing here launches a kernel.
"""

from __future__ import annotations

import copy
import json

import pytest

import meep_gpu.fastpath as fastpath
import meep_gpu.subnormal_policy as policy_module
from meep_gpu import backends, fastpath_cuda, metal_dispatch
from meep_gpu.test_dispatch_contract import (
    CountingPlan,
    certified_envelope_fields,
    certified_envelope_pml,
    cuda_step_plan,
    plan_on_a_two_table_host,
    step_plan,
)
from meep_gpu.test_dispatch_flush_licence import install_caller_flush
from meep_gpu.test_uncertified_opt_in import _envelope_grid_with_device
from meep_gpu.triton_kernels import launch as launch_module

CAPABILITY = "8.6"

#: The Triton singles a CUDA-led plan planned on the route gate's ``cuda_flush`` leg
#: (``dispersive_2d``'s veto leg, the ``nonlinear_3d`` envelope control,
#: ``complex_no_pml_3d`` and ``absorber_1d``). Each is a bare label of
#: ``fastpath.ARM_CERTIFICATION`` with no ``cuda:`` twin.
TRITON_SINGLES = ("nonlinear", "nonlinear run PML", "ADE update_P", "dispersive",
                  "conductive no-PML", "complex conductive no-PML curl")

#: The Triton fused pairs the same leg's arbitration control plans.
TRITON_PAIRS = ("fused pair B", "fused pair D")

#: The bare Triton labels the CUDA composer also spells: ``fastpath_cuda.certification_for``
#: namespaces whatever it is handed, so a bare one reaching it reads a CUDA entry.
SHARED_SPELLINGS = ("PML", "ordinary", "off-diagonal", "folded off-diagonal", "mirror fill")


def _previous_lookup(arm, table):
    """The lookup before the table moved: a bare label took the composing table. Kept
    as the keep path's reference only; nothing in the package reads it."""
    return fastpath._table_of(arm) or table


def _evidence(policies):
    return {policy: {"artifact_sha256": "0" * 64, "leg": f"a {policy} leg"}
            for policy in policies}


def _ledger(table, *, flush_except=()):
    """The shipped ledger with ``certified_policies`` on every cited gate's 8.6 run:
    ``keep`` and ``flush`` on each, except the gates in ``flush_except`` (``keep`` only)."""
    ledger = copy.deepcopy(dict(fastpath._fingerprints(table)))  # noqa: SLF001
    cited = {gate for _family, gate in fastpath._arm_certification_map(table).values()}
    for gate in sorted(cited):
        run = ledger[gate][fastpath.RUNS][CAPABILITY]
        run[fastpath.CERTIFIED_POLICIES] = _evidence(
            ("keep",) if gate in flush_except else ("keep", "flush"))
    return ledger


def _install_ledgers(monkeypatch, triton, cuda):
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {"triton_kernels": triton,
                                                    "cuda_kernels": cuda})


def _triton_gate(label):
    return fastpath.ARM_CERTIFICATION[label][1]


FUSED_ELECTRIC_GATE = _triton_gate("fused pair B")


def _fresh(flushing):
    """No policy installed, no announcement made, and the host FPU in ``flushing``."""
    policy_module._reset_for_tests()
    fastpath.reset_dispatch_announcements()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None
    if backends.subnormals_flushed() != flushing:
        policy_module._drive_host_fpu(flushing)


@pytest.fixture(autouse=True)
def _clean_process(monkeypatch, tmp_path):
    """A fresh policy state per test, and the host FPU put back as it was found: a
    caller ``flush`` leaves this thread flushing, which no uninstall undoes."""
    policy_module._reset_for_tests()
    fastpath.reset_dispatch_announcements()
    fastpath._POLICY_INSTALLED_BY_DISPATCH = None
    flushing_at_entry = backends.subnormals_flushed()
    for name in (fastpath.FUSED_KILL_SWITCH, fastpath.SUBNORMAL_INSTALL_SWITCH,
                 fastpath.DISPATCH_LOG, fastpath.WARM_SWITCH, fastpath.FUSE_ARMS_SWITCH,
                 fastpath.BACKEND_PREFERENCE_SWITCH, fastpath.KERNEL_TABLE_SWITCH,
                 policy_module.POLICY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(fastpath.DISPATCH_ENABLE, "1")
    monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy-cache"))
    try:
        yield
    finally:
        _fresh(flushing_at_entry)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,composing,answers", [
    # a CUDA-led merged plan: its bare labels are Triton singles and Triton fused pairs
    *[(label, fastpath.CUDA_TABLE, "triton") for label in TRITON_SINGLES + TRITON_PAIRS],
    *[(label, fastpath.CUDA_TABLE, "triton") for label in SHARED_SPELLINGS],
    # a Triton-led plan
    ("nonlinear", "triton", "triton"),
    ("fused pair B", "triton", "triton"),
    ("PML", "triton", "triton"),
    # a Metal host (a NumPy engine with an MPS device)
    ("PML", fastpath.METAL_TABLE, fastpath.METAL_TABLE),
    ("ADE update_P", fastpath.METAL_TABLE, fastpath.METAL_TABLE),
    # a namespaced label answers for itself, whatever composed the plan
    ("cuda:fused magnetic pair", fastpath.CUDA_TABLE, fastpath.CUDA_TABLE),
    ("cuda:fused magnetic pair", "triton", fastpath.CUDA_TABLE),
    ("cuda:PML", fastpath.CUDA_TABLE, fastpath.CUDA_TABLE),
    ("cuda:PML", fastpath.METAL_TABLE, fastpath.CUDA_TABLE),
])
def test_the_label_table(label, composing, answers):
    assert fastpath._table_for_label(label, composing) == answers


def test_a_bare_label_still_names_no_table_of_its_own():
    """``_table_of`` keeps answering None for a bare label: on a Metal host that label is
    the Metal table's, so only the composing table can say which ledger it is."""
    for label in TRITON_SINGLES + TRITON_PAIRS + SHARED_SPELLINGS:
        assert fastpath._table_of(label) is None
    assert fastpath._table_of("cuda:fused magnetic pair") == fastpath.CUDA_TABLE


def test_a_bare_label_on_the_cuda_table_quotes_the_triton_ledger(monkeypatch):
    """And never reaches the CUDA lookup, which would namespace it and read a CUDA
    entry: a family beside a dead key, or for a shared spelling the wrong table's run."""
    reached = []
    real = fastpath_cuda.certification_for

    def cuda_lookup(arm, *, capability=None):
        reached.append(arm)
        return real(arm, capability=capability)

    monkeypatch.setattr(fastpath_cuda, "certification_for", cuda_lookup)
    for label in TRITON_SINGLES + TRITON_PAIRS + SHARED_SPELLINGS:
        entry = fastpath._certification_for(label, fastpath.CUDA_TABLE,
                                            capability=CAPABILITY)
        assert (entry["family"], entry["gate"]) == fastpath.ARM_CERTIFICATION[label], label
        assert entry.get("capability") == CAPABILITY, (label, entry.get("run_record"))
    assert reached == []
    entry = fastpath._certification_for("cuda:fused magnetic pair", "triton",
                                        capability=CAPABILITY)
    assert reached == ["cuda:fused magnetic pair"]
    assert (entry["family"], entry["gate"]) == fastpath_cuda.CUDA_ARM_CERTIFICATION[
        "cuda:fused magnetic pair"]


def test_a_bare_label_on_a_metal_plan_quotes_the_metal_map():
    for label in ("PML", "ADE update_P"):
        entry = fastpath._certification_for(label, fastpath.METAL_TABLE)
        assert (entry["family"], entry["gate"]) == metal_dispatch.ARM_CERTIFICATION[label]


# ---------------------------------------------------------------------------
# Rung 8b's per-arm evidence, against ledgers that carry certified_policies
# ---------------------------------------------------------------------------


def test_under_flush_the_triton_singles_of_a_cuda_led_plan_are_admitted(monkeypatch):
    """Every cited gate released under flush, the fused D->E gate apart: the six singles
    and the CUDA products are admitted, and the two fused pairs are refused by name for
    their own gate's missing evidence, not as unmapped."""
    _install_ledgers(monkeypatch, _ledger("triton", flush_except=(FUSED_ELECTRIC_GATE,)),
                     _ledger(fastpath.CUDA_TABLE))
    cuda_labels = ["cuda:fused magnetic pair", "cuda:three-slot dispersive weld"]
    assert fastpath.policy_evidence_reasons(
        list(TRITON_SINGLES) + cuda_labels, fastpath.CUDA_TABLE, CAPABILITY, "flush") == []
    reasons = fastpath.policy_evidence_reasons(
        list(TRITON_SINGLES + TRITON_PAIRS) + cuda_labels, fastpath.CUDA_TABLE,
        CAPABILITY, "flush")
    assert [reason.split(":", 1)[0] for reason in reasons] == list(TRITON_PAIRS)
    for reason in reasons:
        assert f"{FUSED_ELECTRIC_GATE!r}" in reason, reason
        assert "certifies ['keep'], not 'flush'" in reason, reason
        assert "unmapped" not in reason, reason


def test_under_flush_the_fused_pairs_are_admitted_once_their_evidence_is_bound(
        monkeypatch):
    _install_ledgers(monkeypatch, _ledger("triton"), _ledger(fastpath.CUDA_TABLE))
    assert fastpath.policy_evidence_reasons(
        list(TRITON_SINGLES + TRITON_PAIRS) + ["cuda:fused magnetic pair"],
        fastpath.CUDA_TABLE, CAPABILITY, "flush") == []


@pytest.mark.parametrize("label", SHARED_SPELLINGS)
def test_a_shared_spelling_is_never_admitted_on_the_cuda_entry(monkeypatch, label):
    """The CUDA entry of the same spelling released under flush and the Triton gate did
    not: the Triton arm is refused, naming the Triton gate."""
    gate = _triton_gate(label)
    _install_ledgers(monkeypatch, _ledger("triton", flush_except=(gate,)),
                     _ledger(fastpath.CUDA_TABLE))
    cuda_entry = fastpath._certification_for(f"cuda:{label}", fastpath.CUDA_TABLE,
                                             capability=CAPABILITY)
    assert "flush" in cuda_entry[fastpath.CERTIFIED_POLICIES]
    (reason,) = fastpath.policy_evidence_reasons([label], fastpath.CUDA_TABLE,
                                                 CAPABILITY, "flush")
    assert reason.startswith(f"{label}:") and f"{gate!r}" in reason, reason
    assert "certifies ['keep'], not 'flush'" in reason, reason


def test_keep_asks_no_arm_anything(monkeypatch):
    """The install default needs no per-arm evidence: the lookup is never consulted."""
    def refuse(*args, **kwargs):
        raise AssertionError("keep consulted the per-arm certification")

    monkeypatch.setattr(fastpath, "_certification_for", refuse)
    for table in ("triton", fastpath.CUDA_TABLE):
        assert fastpath.policy_evidence_reasons(
            list(TRITON_SINGLES + TRITON_PAIRS + SHARED_SPELLINGS)
            + ["cuda:fused magnetic pair"], table, CAPABILITY, "keep") == []


# ---------------------------------------------------------------------------
# End to end through plan_fast_path: a CUDA-led merged plan on a governable host
# ---------------------------------------------------------------------------


def _triton_singles_plan():
    return step_plan({"step_B": CountingPlan("b"), "update_H": CountingPlan("h")},
                     {"step_B": "nonlinear run PML", "update_H": "nonlinear"})


def _triton_pair_plan():
    pair = CountingPlan("fused pair B")
    return step_plan({"step_B": pair, "update_H": launch_module.NoopPlan("update_H", pair)},
                     {"step_B": "fused pair B", "update_H": "fused pair B"})


def _plan(monkeypatch, triton_plan, preference="cuda"):
    plan, _calls = plan_on_a_two_table_host(
        monkeypatch, triton_plan, cuda_step_plan(), preference=preference,
        grid=_envelope_grid_with_device((8, 6)), fields=certified_envelope_fields(),
        pml=certified_envelope_pml())
    return plan, fastpath.last_dispatch_report()


def _gate(record):
    return record["subnormal"]["gate"]


def test_rung_8b_admits_a_cuda_led_plan_with_triton_singles_under_flush(monkeypatch):
    _install_ledgers(monkeypatch, _ledger("triton", flush_except=(FUSED_ELECTRIC_GATE,)),
                     _ledger(fastpath.CUDA_TABLE))
    install_caller_flush()
    plan, record = _plan(monkeypatch, _triton_singles_plan())
    assert plan is not None, record.get("refused_because")
    assert record["arbitration"]["effective_order"] == ["cuda", "triton"]
    assert plan.backends["step_B"] == "triton" and plan.backends["step_D"] == "cuda"
    gate = _gate(record)
    assert gate["policy_evidence"]["missing"] == []
    assert sorted(gate["policy_evidence"]["arms"]) == sorted(
        ["nonlinear run PML", "nonlinear", "cuda:fused electric pair"])
    assert gate["required_policy"] == "flush" and gate["policy_in_force"] == "flush"
    for arm in ("nonlinear run PML", "nonlinear"):
        assert record["families"][arm]["table"] == "triton"
        assert record["families"][arm]["gate"] == _triton_gate(arm)
    assert record["families"]["cuda:fused electric pair"]["table"] == fastpath.CUDA_TABLE


def test_rung_8b_judges_a_triton_fused_pair_in_a_cuda_led_plan_on_its_own_gate(
        monkeypatch):
    """Refused by name while the fused D->E gate carries no flush run; admitted once it
    does. Either way the clause cites the Triton gate."""
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, "fused pair B")
    _install_ledgers(monkeypatch, _ledger("triton", flush_except=(FUSED_ELECTRIC_GATE,)),
                     _ledger(fastpath.CUDA_TABLE))
    install_caller_flush()
    plan, record = _plan(monkeypatch, _triton_pair_plan())
    assert plan is None
    missing = _gate(record)["policy_evidence"]["missing"]
    assert [reason.split(":", 1)[0] for reason in missing] == ["fused pair B"], missing
    assert f"{FUSED_ELECTRIC_GATE!r}" in missing[0] and "unmapped" not in missing[0]

    _fresh(False)
    _install_ledgers(monkeypatch, _ledger("triton"), _ledger(fastpath.CUDA_TABLE))
    install_caller_flush()
    plan, record = _plan(monkeypatch, _triton_pair_plan())
    assert plan is not None, record.get("refused_because")
    assert plan.arms["step_B"] == "fused pair B" and plan.backends["step_B"] == "triton"
    assert _gate(record)["policy_evidence"]["missing"] == []
    assert record["families"]["fused pair B"]["gate"] == FUSED_ELECTRIC_GATE


def test_rung_8b_refuses_a_shared_spelling_the_triton_gate_did_not_release(monkeypatch):
    """``PML`` in a CUDA-led plan, whose ``cuda:PML`` twin released under flush: refused,
    naming the Triton gate. The CUDA entry's evidence is about a different kernel."""
    gate = _triton_gate("PML")
    _install_ledgers(monkeypatch, _ledger("triton", flush_except=(gate,)),
                     _ledger(fastpath.CUDA_TABLE))
    install_caller_flush()
    plan, record = _plan(monkeypatch, step_plan({"step_B": CountingPlan("b")},
                                                {"step_B": "PML"}))
    assert plan is None
    (reason,) = _gate(record)["policy_evidence"]["missing"]
    assert reason.startswith("PML:") and f"{gate!r}" in reason, reason


# ---------------------------------------------------------------------------
# The keep path: the admission and the record are the previous lookup's
# ---------------------------------------------------------------------------




def _comparable(record):
    """The record as JSON, without the freeze's own clock readings."""
    text = json.dumps(record, sort_keys=True, default=repr)
    value = json.loads(text)
    for key in ("planned_utc", "elapsed_s", "plan_seconds"):
        value.pop(key, None)
    return value


@pytest.mark.parametrize("triton_plan", [_triton_singles_plan, _triton_pair_plan],
                         ids=["triton_singles", "triton_fused_pair"])
@pytest.mark.parametrize("preference", ["cuda", None], ids=["cuda_led", "triton_led"])
def test_keep_admission_and_record_are_unchanged(monkeypatch, tmp_path, triton_plan,
                                                preference):
    """Nothing installed, so dispatch installs keep: the plan, the gate and every record
    field are the same under the label table and under the lookup it replaced.

    Each freeze starts from the same process state: the keep install points
    ``CUPY_CACHE_DIR`` at the policy's private directory, so it is put back first."""
    monkeypatch.setenv(fastpath.FUSE_ARMS_SWITCH, "fused pair B")
    flushing = backends.subnormals_flushed()
    records = {}
    for name, lookup in (("table", fastpath._table_for_label),
                         ("previous", _previous_lookup)):
        _fresh(flushing)
        monkeypatch.setenv("CUPY_CACHE_DIR", str(tmp_path / "cupy-cache"))
        monkeypatch.setattr(fastpath, "_table_for_label", lookup)
        plan, record = _plan(monkeypatch, triton_plan(), preference=preference)
        assert plan is not None, (name, record.get("refused_because"))
        gate = _gate(record)
        assert gate["required_policy"] == "keep"
        assert "policy_evidence" not in gate
        records[name] = (sorted(plan.slots), dict(plan.arms), dict(plan.backends),
                         _comparable(record))
    assert records["table"][:3] == records["previous"][:3]
    table, previous = _leaves(records["table"][3]), _leaves(records["previous"][3])
    # ``policy_epoch`` counts installs in this process, so it is one higher on the
    # second freeze whatever composed it; it is the only leaf excluded.
    differing = sorted(key for key in set(table) | set(previous)
                       if table.get(key) != previous.get(key)
                       and not key.endswith("/policy_epoch"))
    assert differing == [], [(key, table.get(key), previous.get(key))
                             for key in differing[:8]]


def _leaves(value, path=""):
    """Every leaf of a JSON value, keyed by its path."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_leaves(item, f"{path}/{key}"))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(_leaves(item, f"{path}[{index}]"))
        return out or {path: []}
    return {path: value}
