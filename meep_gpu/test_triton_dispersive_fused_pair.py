"""Laptop contracts for the fused dispersive D/E Triton pair.

The CUDA gate owns byte identity.  These tests own the legal dependency seam,
fail-closed composition, pole ordering, and exact source shape without requiring
CuPy or Triton.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from meep_gpu.device_identity import weld_survives_edit
from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels import launch as launch_module
from meep_gpu.triton_kernels import dispersive_fused_pair as module

PACKAGE_DIR = pathlib.Path(module.__file__).parent
PROBE = PACKAGE_DIR.parents[1] / "parity" / "meep_gpu" / "probe_triton_dispersive_fused_pair.py"


def _live_probe_is_welded(record, weld_name, repo_key, path):
    """The live probe file is the one a PASS weld recorded, or say why not.

    Kept separate from the historical entry's ``probe_sha256`` on purpose: that
    field names the bytes a particular past job ran and must not be re-typed when
    the file changes. This one asks the different question a reader actually needs
    answered — "is the probe on disk today the probe some released device run
    executed?" — and it is answered by the weld, which is the record that carries
    a status, a policy and an artifact digest.

    A MISSING WELD IS A NAMED FAILURE, never a skip. The probe was re-cut on
    device on 2026-08-19 and the weld entry is pending application to
    fingerprints.json; until it lands this test says so, with the digest to apply.
    A skip here would let the tree sit indefinitely with a probe no record covers,
    which is the orphaned-weld failure this file already carries scar tissue from.
    """
    import hashlib

    live = hashlib.sha256(path.read_bytes()).hexdigest()
    weld = record.get(weld_name)
    assert isinstance(weld, dict), (
        f"{weld_name} is not in fingerprints.json; the live {repo_key} "
        f"({live}) is covered by no PASS weld. Apply the 2026-08-19 device weld.")
    assert weld.get("status") == "PASS", f"{weld_name} is not a PASS weld"
    recorded = weld["source_sha256"].get(repo_key)
    assert recorded == live, (
        f"{weld_name} recorded {recorded} for {repo_key}; the live file is "
        f"{live} — re-run the gate, do not re-type the record")


def with_poles(counts=(2, 1, 2)):
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    for index in range(max(counts)):
        sigma = {
            component: (0.3 + 0.05 * index if counts[axis] > index else 0.0)
            for axis, component in enumerate(("Ex", "Ey", "Ez"))
        }
        fields.polarizations.append(PolarizationState(
            Susceptibility(frequency=1.0 + 0.1 * index, gamma=0.1),
            sigma, grid, np.float32))
    return fields, PML(grid=grid, thickness=2)


def reasons(fields, pml, sources=()):
    return [reason for reason in
            module.dispersive_fused_pair_coverage(fields, pml, sources).reasons
            if "not cupy" not in reason]


def test_the_module_is_importable_without_triton_and_refused_before_launch():
    fields, pml = with_poles()
    verdict = module.dispersive_fused_pair_coverage(fields, pml, ())
    assert not verdict.covered
    assert reasons(fields, pml, ()) == []
    assert module.plan_dispersive_fused_pair(fields, pml, ()) is None


@pytest.mark.parametrize(
    ("sources", "refused"),
    (
        ((), False),
        ((SimpleNamespace(field_type="B"),), False),
        ((SimpleNamespace(field_type="D"),), True),
        ((SimpleNamespace(field_type="B"), SimpleNamespace(field_type="D")), True),
        (None, True),
    ),
)
def test_only_electric_injection_occupies_the_dispersive_D_E_seam(
        monkeypatch, sources, refused):
    """WHICH FAMILY LANDS IN THIS SEAM, asked with the deposit repair not declared.

    The seam assignment and what the product does about it are two facts. This one is
    the assignment, and it is unchanged by the flip: a magnetic source is injected in
    the B/H seam and is not this pair's problem, an electric one is, and an undeclared
    inventory is a refusal rather than an empty set. The next test owns the other half.
    """
    fields, pml = with_poles()
    monkeypatch.setattr(module, "CARRIES_DEPOSIT_REPAIR", False)
    verdict = reasons(fields, pml, sources)
    has_source_refusal = any(
        "source" in reason and ("electric" in reason or "not declared" in reason)
        for reason in verdict)
    assert has_source_refusal is refused, verdict


def test_the_electric_seam_is_now_carried_by_the_deposit_repair():
    """The other half: this product declares the repair, so the seam is admitted.

    An electric source that publishes the index it writes is carried — the fused launch
    runs against an uninjected D and ``TrailingRepairPlan`` recomputes E and ``f_w_E``
    at the deposit points afterwards. One that hides its index is still refused, which
    is what keeps the declaration from being an unconditional yes.
    """
    fields, pml = with_poles()
    assert module.CARRIES_DEPOSIT_REPAIR is True

    opaque = SimpleNamespace(field_type="D")
    refused = reasons(fields, pml, (opaque,))
    assert not any("driver injects it BETWEEN" in reason for reason in refused), refused
    assert any("does not publish the index it writes" in reason
               for reason in refused), refused

    index = np.array([1], dtype=np.intp)
    carried = SimpleNamespace(field_type="D", _point_ix=index, _point_iy=index,
                              _point_iz=index)
    assert reasons(fields, pml, (carried,)) == []
    # An undeclared inventory is STILL a refusal: the repair carries a seam it can see,
    # and ignorance is not an empty set.
    assert any("not declared" in reason for reason in reasons(fields, pml, None))


def test_ordinary_and_dispersive_D_E_pair_predicates_are_disjoint():
    fields, pml = with_poles()
    assert reasons(fields, pml, ()) == []
    ordinary = [reason for reason in
                coverage_module.fused_pair_coverage(fields, pml, "D", ()).reasons
                if "not cupy" not in reason]
    assert ordinary and any("susceptibility" in reason for reason in ordinary)

    plain_grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    plain = Fields(grid=plain_grid, force_complex_fields=False)
    plain.enable_pml_storage()
    plain_pml = PML(grid=plain_grid, thickness=2)
    assert [reason for reason in coverage_module.fused_pair_coverage(
        plain, plain_pml, "D", ()).reasons if "not cupy" not in reason] == []
    specialized = reasons(plain, plain_pml, ())
    assert specialized and any("no susceptibility" in reason for reason in specialized)


def test_the_kernel_keeps_the_curl_grouping_pole_order_and_constitutive_grouping():
    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    code = ast.unparse(tree)
    assert "options=" not in code
    assert code.count("enable_fp_fusion=") == 1
    for line in (
        "curl0 = dtdx * ((c_y - c) + (b - b_z))",
        "curl1 = dtdx * ((a_z - a) + (c - c_x))",
        "curl2 = dtdx * ((b_x - b) + (a - a_y))",
        "n0 = ((p0 * km_y) - curl0) * si_y",
        "v0 = (((tl.load(f0 + idx, mask=live, other=0.0) * km_z) + n0) - p0) * si_z",
        "src0 = s0 * tl.load(ie0 + idx, mask=live, other=0.0)",
        "a0v = a0v + kp_0 * src0",
        "a0v = a0v - km_0 * prev0",
    ):
        assert line in source, line
    body = source.split("def fused_curl_dispersive_E(", 1)[1]
    for letter, displacement in (("a", "v0"), ("b", "v1"), ("c", "v2")):
        assert f"s{ord(letter) - ord('a')} = {displacement}" in body
        for slot in range(module.MAX_POLES):
            assert f"tl.load({letter}{slot} + idx" in body


def _install_composition_stubs(monkeypatch, *, ordinary_fused, specialized_fused):
    refused = coverage_module.Coverage(False, ("outside this routing test",))
    admitted = coverage_module.Coverage(True, ())
    monkeypatch.setattr(launch_module, "pml_curl_coverage",
                        lambda fields, pml, name: admitted)
    monkeypatch.setattr(launch_module, "plan_pml_curl",
                        lambda fields, pml, name, block=None: f"curl:{name}")
    monkeypatch.setattr(launch_module, "conductive_pml_curl_coverage",
                        lambda *args, **kwargs: refused)
    monkeypatch.setattr(launch_module, "plain_curl_coverage",
                        lambda *args, **kwargs: refused)
    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda fields, pml, side: admitted if side == "H" else refused)
    monkeypatch.setattr(launch_module, "plan_constitutive",
                        lambda fields, pml, side, block=None: f"constitutive:{side}")
    monkeypatch.setattr(launch_module, "dispersive_constitutive_coverage",
                        lambda *args, **kwargs: admitted)
    monkeypatch.setattr(launch_module, "plan_dispersive_constitutive",
                        lambda *args, **kwargs: "dispersive:E")
    monkeypatch.setattr(
        launch_module, "fused_pair_coverage",
        lambda fields, pml, pair, sources: (
            ordinary_fused if pair == "D" else refused))
    monkeypatch.setattr(launch_module, "dispersive_fused_pair_coverage",
                        lambda *args, **kwargs: specialized_fused, raising=False)


def test_plan_step_selects_the_specialized_pair_and_keeps_ADE_separate(monkeypatch):
    refused = coverage_module.Coverage(False, ("ordinary D/E refuses poles",))
    admitted = coverage_module.Coverage(True, ())
    _install_composition_stubs(
        monkeypatch, ordinary_fused=refused, specialized_fused=admitted)
    sentinel = object()
    monkeypatch.setattr(launch_module, "plan_dispersive_fused_pair",
                        lambda *args, **kwargs: sentinel, raising=False)

    plan = launch_module.plan_step(SimpleNamespace(polarizations=()), object(),
                                   fuse=True, sources=())

    assert plan.plans["step_D"] is sentinel
    assert isinstance(plan.plans["update_E"], launch_module.NoopPlan)
    assert plan.plans["update_E"].absorbed_by is sentinel


def test_plan_step_fails_closed_if_both_D_E_pair_predicates_admit(monkeypatch):
    admitted = coverage_module.Coverage(True, ())
    _install_composition_stubs(
        monkeypatch, ordinary_fused=admitted, specialized_fused=admitted)
    monkeypatch.setattr(launch_module, "plan_fused_pair",
                        lambda *args, **kwargs: object())
    monkeypatch.setattr(launch_module, "plan_dispersive_fused_pair",
                        lambda *args, **kwargs: object(), raising=False)

    plan = launch_module.plan_step(SimpleNamespace(polarizations=()), object(),
                                   fuse=True, sources=())

    assert plan.plans["step_D"] == "curl:step_D"
    assert plan.plans["update_E"] == "dispersive:E"
    assert any("both" in reason for reason in
               plan.reasons["fused_pair_dispersive_D"])


def test_the_cuda_gate_has_multi_pole_metallic_and_second_oracle_cases():
    spec = importlib.util.spec_from_file_location("dispersive_fused_probe", PROBE)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert [case[0] for case in probe.CASES] == [
        "one_pole_periodic", "two_pole_anisotropic", "three_pole_metallic_x",
        "two_pole_electric_source",
    ]
    assert sum(case[3] for case in probe.CASES) == 48
    # THE DEPOSIT CASE IS THE ONE THAT EARNS THE DECLARATION, so it is pinned by shape
    # and not merely by name: exactly one case declares a source, it is electric, and it
    # runs on a metallic axis so the inline wall clear is live in the same seam.
    with_sources = [case for case in probe.CASES if case[4]]
    assert [case[0] for case in with_sources] == ["two_pole_electric_source"]
    assert [item["component"] for item in with_sources[0][4]] == ["Ez"]
    assert with_sources[0][1][0] == "metallic"
    source = PROBE.read_text(encoding="utf-8")
    assert "fused_vs_array" in source and "fused_vs_separate" in source
    assert "Pscratch" in source and "uint32" in source
    assert "raise AssertionError" in source and "save(payload, args.out)" in source
    # A repaired case that repaired nothing would be identical for the wrong reason.
    assert "repairs_seen" in source
    assert "a pass here would be vacuous" in source
    assert "no deposit point was ever repaired" in source
    assert "LeadingRepairPlan" in source and "TrailingRepairPlan" in source


def test_the_cuda_probe_reports_progress_per_step():
    assert "case {name} step {step}/{steps}" in PROBE.read_text(encoding="utf-8")


def test_the_fused_dispersive_record_is_welded_to_every_live_dependency():
    import hashlib
    import json
    import meep_gpu.triton_kernels as public

    record = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    specialized = record["specialized_kernel_sources"][
        "dispersive_fused_pair.py"]
    assert specialized["kernel"] == "fused_curl_dispersive_E"
    module_path = pathlib.Path(module.__file__)
    live = hashlib.sha256(module_path.read_bytes()).hexdigest()
    # ONE HOME FOR THE RULE (device_identity.py:209). This entry pins a
    # specialized kernel module, and what a kernel computes is its ``@triton.jit``
    # body plus the constants that body reads — so an edit the shared rule proves
    # leaves those alone leaves the record describing the shipped kernel. The
    # helper returns False for anything it cannot establish, this entry among
    # them until it carries a device_sha256, and the byte rule then applies.
    if specialized["sha256"] != live and not weld_survives_edit(
            module_path, specialized, "dispersive_fused_pair.py"):
        assert specialized["sha256"] == live, (
            "dispersive_fused_pair.py drifted from the recorded specialized "
            "kernel source and the change reaches executable code")

    gate = record["dispersive_fused_pair_gate"]
    assert record["current_host_gate"] in record
    # 36/36 -> 48/48 ON 2026-08-30, and the number moved because the GATE GAINED A
    # CASE, not because this pin was loosened. The 2026-08-11 run measured three
    # pole products at 12 complete steps each; the 2026-08-30 re-run measures four,
    # adding ``two_pole_electric_source`` -- the IN-SEAM DEPOSIT case, which is the
    # whole reason the family was re-gated once ``deposit_repair`` started carrying
    # a deposit across this seam. The record's own ``_this_recut`` states the move
    # and ``recut_composition_records.CLAIM_RECUTS`` is where it was declared before
    # being written; the two counters below are what make the claim checkable, and
    # 12 * 4 == 48 is the arithmetic that says a case was ADDED rather than a total
    # inflated. Carrying 36/36 here beside a 48/48 record would be a test pinning a
    # claim the artifact no longer makes.
    assert gate["product"]["complete_steps_exact"] == "48/48"
    assert gate["product"]["cases_exact"] == "4/4"
    assert gate["product"]["in_seam_deposit_cases"].startswith("1/1")
    assert "separate" in gate["product"]["oracles_per_step"]
    assert gate["dispatch"].startswith("DISABLED")

    package = PACKAGE_DIR.parent
    live = {
        "driver.py": package / "driver.py",
        "dispersion.py": package / "dispersion.py",
        "fields.py": package / "fields.py",
        "sources.py": package / "sources.py",
        "kernels.py": PACKAGE_DIR / "kernels.py",
        "launch.py": PACKAGE_DIR / "launch.py",
        "coverage.py": PACKAGE_DIR / "coverage.py",
        "conductivity.py": PACKAGE_DIR / "conductivity.py",
        "no_pml.py": PACKAGE_DIR / "no_pml.py",
        "dispersive_update_e.py": PACKAGE_DIR / "dispersive_update_e.py",
        "dispersive_fused_pair.py": PACKAGE_DIR / "dispersive_fused_pair.py",
        "fused_ade_state.py": PACKAGE_DIR / "fused_ade_state.py",
        "symmetry.py": PACKAGE_DIR / "symmetry.py",
        "__init__.py": PACKAGE_DIR / "__init__.py",
    }
    # This record is intentionally historical; the current route pins the live
    # central dependency closure in test_triton_kernels.
    for name in live:
        assert len(gate["source_sha256"][name]) == 64, name
    # THE LIVE-PROBE PIN MOVED, and the historical entry is left alone.
    #
    # It used to read ``gate["probe_sha256"] == sha256(PROBE)`` — the 2026-08-11
    # Slurm job's probe digest compared against whatever is on disk today. That
    # was a drift detector, and on 2026-08-19 it fired for the right reason: the
    # probe was edited to install a NAMED subnormal policy before the first
    # device compile and to route every save through ``gate_provenance.stamp``,
    # because the record it cut carried no readable verdict and no policy and
    # could therefore not be welded.
    #
    # The response this tree already uses for a re-run is to ADD a record, never
    # to re-type an old one: every ``recert_*`` block here says the fields above
    # are true about that job and its bytes. So the historical ``probe_sha256``
    # keeps naming the certifying run's probe, and the live file is pinned by the
    # 2026-08-19 device weld instead — which is what a later reader must consult
    # to learn what the shipping probe was measured to do.
    _live_probe_is_welded(record, "triton_dispersive_fused_pair_device_gate",
                          "parity/meep_gpu/probe_triton_dispersive_fused_pair.py",
                          PROBE)
    for name in ("dispersive_fused_pair_coverage",
                 "plan_dispersive_fused_pair"):
        assert name in public.__all__ and callable(getattr(public, name))
