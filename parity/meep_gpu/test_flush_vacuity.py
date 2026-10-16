"""The flush not-applicable rule, and the gates that consult it (host-only).

``flush_vacuity.subnormal_class_not_applicable`` may waive a SUBNORMAL-operand floor only
when three measurements agree: the installed policy is flush, the host FPU flushes when
the class is seeded, and the class's own census counted 0 subnormal words. These tests pin
that each measurement is load-bearing, that the floors it does not cover (signed zero,
zero, nonzero) stand under both policies, and that the gates which carry a subnormal
floor consult the rule at the floor and record the measurement beside their census.
Nothing here imports MEEP or needs a device.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE.parents[1])):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import flush_vacuity  # noqa: E402

FLUSHING = {"policy_installed": True, "policy": "flush", "host_flushing": True}


def test_the_three_measurements_together_declare_the_floor_not_applicable():
    reason = flush_vacuity.subnormal_class_not_applicable(0, FLUSHING)
    assert reason and reason.startswith(flush_vacuity.NOT_APPLICABLE)


@pytest.mark.parametrize("conditions", [
    dict(FLUSHING, policy="keep"),            # the keep campaign: never
    dict(FLUSHING, policy_installed=False),   # no install: the name is a resolution only
    dict(FLUSHING, host_flushing=False),      # the host kept subnormals at seed time
    dict(FLUSHING, host_flushing=None),       # not measured
    None, {}, "flush",
])
def test_any_one_measurement_missing_and_the_floor_applies(conditions):
    assert flush_vacuity.subnormal_class_not_applicable(0, conditions) is None


@pytest.mark.parametrize("count", [1, 7, "3"])
def test_a_counted_subnormal_keeps_the_floor_under_flush(count):
    """A bit-planted subnormal that survived the seeding is a measured class: scored."""
    assert flush_vacuity.subnormal_class_not_applicable(count, FLUSHING) is None


@pytest.mark.parametrize("count", [None, "not a count", object()])
def test_an_unreadable_census_keeps_the_floor(count):
    assert flush_vacuity.subnormal_class_not_applicable(count, FLUSHING) is None


def test_seed_conditions_reads_the_install_and_measures_the_host(monkeypatch):
    from meep_gpu import backends, subnormal_policy

    monkeypatch.setattr(subnormal_policy, "policy_is_installed", lambda: True)
    monkeypatch.setattr(subnormal_policy, "get_subnormal_policy", lambda: "flush")
    monkeypatch.setattr(backends, "subnormals_flushed", lambda: True)
    assert flush_vacuity.seed_conditions() == FLUSHING
    monkeypatch.setattr(subnormal_policy, "policy_is_installed", lambda: False)
    monkeypatch.setattr(backends, "subnormals_flushed", lambda: False)
    assert flush_vacuity.seed_conditions() == {
        "policy_installed": False, "policy": None, "host_flushing": False}


#: Every gate carrying a subnormal floor that failed the flush fleet's first run (FPC),
#: plus the cylindrical probe whose flush cut became real with it. Each must consult the
#: rule and take the seed-time measurement.
GATES = (
    "probe_triton_folded_beta_complex_fused_pair.py",
    "probe_triton_folded_beta_complex_fused_magnetic_pair.py",
    "probe_triton_folded_complex_fused_pair.py",
    "probe_triton_folded_complex_fused_magnetic_pair.py",
    "gate_triton_nonlinear.py",
    "gate_triton_offdiag.py",
    "gate_triton_no_pml_constitutive.py",
    "gate_triton_folded_offdiag.py",
    "probe_triton_cylindrical_fused_magnetic_pair.py",
)


@pytest.mark.parametrize("name", GATES)
def test_each_gate_consults_the_rule_and_records_the_seed_measurement(name):
    text = (HERE / name).read_text(encoding="utf-8")
    assert "flush_vacuity.subnormal_class_not_applicable(" in text
    assert "flush_vacuity.seed_conditions()" in text


FOLDED = (
    "probe_triton_folded_beta_complex_fused_pair",
    "probe_triton_folded_beta_complex_fused_magnetic_pair",
    "probe_triton_folded_complex_fused_pair",
    "probe_triton_folded_complex_fused_magnetic_pair",
)


def _row(value_class, census, conditions):
    return {"value_class": value_class, "operand_census": census,
            "seed_conditions": conditions, "first_divergence": None,
            "control_divergence": None}


def _vacuous(failures, key):
    return [f for f in failures if "VACUOUS OPERAND CLASS" in f and key in f]


@pytest.mark.parametrize("module", FOLDED)
def test_folded_probe_band_floor_holds_under_keep(module):
    probe = importlib.import_module(module)
    row = _row("subnormal_band", {"words": 8, "subnormals": 0, "negative_zeros": 0,
                                  "zeros": 0}, dict(FLUSHING, policy="keep"))
    _passed, failures = probe.verdict_of(row, require_moved=False)
    assert _vacuous(failures, "subnormals")
    assert "not_applicable" not in row


@pytest.mark.parametrize("module", FOLDED)
def test_folded_probe_band_floor_is_not_applicable_only_under_measured_flush(module):
    probe = importlib.import_module(module)
    row = _row("subnormal_band", {"words": 8, "subnormals": 0, "negative_zeros": 0,
                                  "zeros": 0}, FLUSHING)
    _passed, failures = probe.verdict_of(row, require_moved=False)
    assert not _vacuous(failures, "subnormals")
    assert row["not_applicable"].startswith(flush_vacuity.NOT_APPLICABLE)
    # The same row with a host that did NOT flush at seed time keeps the floor.
    row = _row("subnormal_band", {"words": 8, "subnormals": 0, "negative_zeros": 0,
                                  "zeros": 0}, dict(FLUSHING, host_flushing=False))
    _passed, failures = probe.verdict_of(row, require_moved=False)
    assert _vacuous(failures, "subnormals")


@pytest.mark.parametrize("module", FOLDED)
def test_folded_probe_signed_zero_floor_stands_under_flush(module):
    probe = importlib.import_module(module)
    row = _row("zero_lattice", {"words": 8, "subnormals": 0, "negative_zeros": 0,
                                "zeros": 0}, FLUSHING)
    _passed, failures = probe.verdict_of(row, require_moved=False)
    assert _vacuous(failures, "negative_zeros")


def _identity_case(census, conditions):
    return {"name": "plain_3d_np2", "comparison": {"bit_identical": True,
                                                   "arrays_compared": 4,
                                                   "total_words": 64},
            "census_before": census, "seed_conditions": conditions, "cycles": 2,
            "plan_runs": {"update_H": 2, "update_E": 2}, "ok": True}


def test_no_pml_constitutive_contract_waives_only_the_subnormal_clause_under_flush():
    gate = importlib.import_module("gate_triton_no_pml_constitutive")
    census = {"nonzero_words": 10, "signed_zero_words": 2, "subnormal_words": 0}
    vacuous = lambda failures: [f for f in failures if "VACUOUS" in f]  # noqa: E731
    for conditions, expect in ((FLUSHING, []),
                               (dict(FLUSHING, policy="keep"),
                                ["identity case plain_3d_np2 is VACUOUS: "
                                 "subnormal_words == 0"])):
        failures = []
        block = {"ran": 1, "ok": 1, "cases": [_identity_case(census, conditions)]}
        gate._check_identity_shaped_leg("identity", block, ["plain_3d_np2"], failures)
        assert vacuous(failures) == expect
    failures = []
    block = {"ran": 1, "ok": 1, "cases": [_identity_case(
        dict(census, signed_zero_words=0), FLUSHING)]}
    gate._check_identity_shaped_leg("identity", block, ["plain_3d_np2"], failures)
    assert vacuous(failures) == ["identity case plain_3d_np2 is VACUOUS: "
                                 "signed_zero_words == 0"]


def test_cylindrical_probe_counts_subnormal_words_off_the_bits():
    import numpy as np

    probe = importlib.import_module("probe_triton_cylindrical_fused_magnetic_pair")
    words = np.array([0x00000001, 0x80000002, 0x00000000, 0x80000000, 0x00800000,
                      0x3F800000], dtype=np.uint32)
    assert probe.subnormal_census({"Bx": words}, {"Ex": words[:2]}) == 4
