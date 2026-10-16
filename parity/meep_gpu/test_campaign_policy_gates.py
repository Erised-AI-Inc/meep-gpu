"""Three Triton gates cut the CAMPAIGN's subnormal policy, one per process (host-only).

``gate_triton_complex_ade.py``, ``gate_triton_complex_no_pml_stored_e.py`` and
``probe_triton_cylindrical_fused_magnetic_pair.py`` installed keep whatever the campaign
asked, so the flush fleet refused all three on its own flush cache token (pre-certification
pass FPC). They now read ``MEEP_GPU_SUBNORMAL_POLICY`` through
``composition_policy.requested_policy``. The cylindrical probe's old two-cut loop recorded a
"flush" stamp that resolved keep on every released run (the install is memoised); the
board's verdict-prose reading now refuses such an artifact and admits a one-policy run.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE.parents[1])):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import composition_policy  # noqa: E402

ATTAINED = {name: {"attained": True} for name in ("host", "cupy", "triton")}


def _stamp(resolved, executors=ATTAINED):
    return {"install": {"resolved": resolved, "requested": resolved,
                        "executors": dict(executors)}}


def _artifact(stamps, own):
    return {"environment": {"subnormal_policy_stamps": stamps},
            "subnormal_policy": {"resolved": own}}


@pytest.fixture(scope="module")
def board():
    return importlib.import_module("build_triton_fusion_matrix")


def test_a_one_policy_run_is_read_as_a_device_release_under_either_policy(board):
    assert board.verdict_prose_policy_reasons(_artifact({"keep": _stamp("keep")}, "keep")) == []
    assert board.verdict_prose_policy_reasons(
        _artifact({"flush": _stamp("flush")}, "flush")) == []


def test_the_two_cut_artifact_whose_flush_stamp_resolved_keep_is_refused(board):
    """The shape of every released run of 2026-09-04 .. 09-07: two labels, one install."""
    historical = _artifact({"keep": _stamp("keep"), "flush": _stamp("keep")}, "keep")
    reasons = board.verdict_prose_policy_reasons(historical)
    assert any("resolved to another policy" in r for r in reasons)


@pytest.mark.parametrize("artifact", [
    _artifact({"keep": _stamp("keep", {"host": {"attained": True}})}, "keep"),   # host only
    _artifact({"keep": _stamp("keep")}, None),                                   # no own stamp
    _artifact({"keep": _stamp("keep")}, "flush"),                                # another policy
    _artifact({}, "keep"),                                                       # no stamps
])
def test_a_run_that_did_not_attain_its_own_policy_on_device_is_refused(board, artifact):
    assert board.verdict_prose_policy_reasons(artifact)


def test_the_cylindrical_probe_reads_the_campaign_policy(monkeypatch):
    probe = importlib.import_module("probe_triton_cylindrical_fused_magnetic_pair")
    monkeypatch.delenv("MEEP_GPU_SUBNORMAL_POLICY", raising=False)
    assert probe.campaign_policy() == "keep"
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "flush")
    assert probe.campaign_policy() == "flush"
    monkeypatch.setenv("MEEP_GPU_SUBNORMAL_POLICY", "match_meep")
    with pytest.raises(composition_policy.CampaignPolicyRefused):
        probe.campaign_policy()


def _main_source(name: str) -> str:
    text = (HERE / name).read_text(encoding="utf-8")
    tree = ast.parse(text)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    return ast.get_source_segment(text, main)


def test_the_cylindrical_probe_names_no_policy_literal_in_main():
    """Every leg runs under, and is labelled with, the one policy the campaign named."""
    body = _main_source("probe_triton_cylindrical_fused_magnetic_pair.py")
    assert "campaign_policy()" in body
    assert '"keep"' not in body and '"flush"' not in body
    assert "for policy in POLICIES" not in body


@pytest.mark.parametrize("name", ["gate_triton_complex_ade.py",
                                  "gate_triton_complex_no_pml_stored_e.py"])
def test_the_two_ade_gates_install_the_requested_policy(name):
    body = _main_source(name)
    assert "policy = composition_policy.requested_policy()" in body
    assert "G.install_ftz_strip(policy)" in body
    assert 'policy = fastpath.TABLE_SUBNORMAL_POLICY["triton"]' not in body
    # the default may not drift from the install default
    assert 'composition_policy.DEFAULT_POLICY != fastpath.TABLE_SUBNORMAL_POLICY["triton"]' in body


def test_the_default_is_the_triton_install_default():
    from meep_gpu import fastpath

    assert composition_policy.DEFAULT_POLICY == fastpath.TABLE_SUBNORMAL_POLICY["triton"]
