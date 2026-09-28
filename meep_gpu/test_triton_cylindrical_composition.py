"""Laptop-safe contracts for the m=0 cylindrical Triton composition route.

``cylindrical_triton`` owns the radial-prefix curl kernel and its standalone device
gate.  This module owns the central-composer seam: the two curl plans and the two
already-measured elementwise constitutive updates must be selected together for the
strict real Dcyl slice, without weakening any Cartesian predicate.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import pathlib
from types import SimpleNamespace

from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels import launch as launch_module


ADMITTED = coverage_module.Coverage(True, ())
REFUSED = coverage_module.Coverage(False, ("outside this routing test",))
PARITY_DIR = pathlib.Path(__file__).parents[1] / "parity" / "meep_gpu"
PROBE = PARITY_DIR / "probe_triton_cylindrical_composition.py"
SLURM = PARITY_DIR / "run_triton_cylindrical_composition.slurm"


def cylindrical_fields():
    return SimpleNamespace(
        grid=SimpleNamespace(cylindrical=True, has_symmetry=lambda: False),
        polarizations=(),
    )


def install_cylindrical_routing_stubs(monkeypatch, *, ordinary=REFUSED,
                                      cylindrical=ADMITTED):
    """Replace every device builder while preserving the plan-selection topology."""
    monkeypatch.setattr(launch_module, "folded_grid_active", lambda fields: False)
    monkeypatch.setattr(launch_module, "pml_curl_coverage", lambda *args: ordinary)
    monkeypatch.setattr(launch_module, "conductive_pml_curl_coverage",
                        lambda *args: REFUSED)
    monkeypatch.setattr(launch_module, "plain_curl_coverage", lambda *args: REFUSED)
    monkeypatch.setattr(launch_module, "cylindrical_curl_coverage",
                        lambda *args: cylindrical, raising=False)
    monkeypatch.setattr(launch_module, "plan_cylindrical_curl",
                        lambda fields, pml, sub_step, block=None: f"cyl:{sub_step}",
                        raising=False)

    monkeypatch.setattr(launch_module, "constitutive_coverage",
                        lambda *args: ordinary)
    monkeypatch.setattr(launch_module, "dispersive_constitutive_coverage",
                        lambda *args: REFUSED)
    monkeypatch.setattr(launch_module, "cylindrical_constitutive_coverage",
                        lambda *args: cylindrical, raising=False)
    monkeypatch.setattr(launch_module, "plan_cylindrical_constitutive",
                        lambda fields, pml, side, block=None: f"cyl:{side}",
                        raising=False)


def test_plan_step_selects_all_four_real_m0_cylindrical_substeps(monkeypatch):
    install_cylindrical_routing_stubs(monkeypatch)

    plan = launch_module.plan_step(cylindrical_fields(), object())

    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert plan.plans == {
        "step_B": "cyl:step_B",
        "update_H": "cyl:H",
        "step_D": "cyl:step_D",
        "update_E": "cyl:E",
    }


def test_cylindrical_overlap_fails_closed_instead_of_selecting_by_order(monkeypatch):
    install_cylindrical_routing_stubs(monkeypatch, ordinary=ADMITTED)
    monkeypatch.setattr(launch_module, "plan_pml_curl",
                        lambda *args, **kwargs: "ordinary")
    monkeypatch.setattr(launch_module, "plan_constitutive",
                        lambda fields, pml, side, block=None: f"ordinary:{side}")

    plan = launch_module.plan_step(cylindrical_fields(), object())

    for name in ("step_B", "step_D", "update_H", "update_E"):
        assert name not in plan.plans
        assert any("ambiguous" in reason or "both" in reason
                   for reason in plan.reasons[name])


def test_opt_in_pair_fusion_cannot_replace_the_m0_cylindrical_plans(monkeypatch):
    install_cylindrical_routing_stubs(monkeypatch)
    monkeypatch.setattr(launch_module, "fused_pair_coverage",
                        lambda *args: REFUSED)
    monkeypatch.setattr(launch_module, "dispersive_fused_pair_coverage",
                        lambda *args: REFUSED)

    plan = launch_module.plan_step(cylindrical_fields(), object(), fuse=True,
                                   sources=())

    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    assert plan.plans["step_B"] == "cyl:step_B"
    assert plan.plans["update_E"] == "cyl:E"
    assert "fused_pair_B" in plan.reasons
    assert "fused_pair_D" in plan.reasons


def test_public_package_exposes_the_cylindrical_entry_points_lazily():
    import meep_gpu.triton_kernels as public

    for name in (
        "cylindrical_curl_coverage", "plan_cylindrical_curl",
        "cylindrical_constitutive_coverage", "plan_cylindrical_constitutive",
    ):
        assert name in public.__all__
        assert callable(getattr(public, name))


def test_cuda_gate_exercises_the_composed_m0_slice_and_an_active_axis_source():
    spec = importlib.util.spec_from_file_location("cylindrical_composition_probe", PROBE)
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    assert [case[0] for case in probe.CASES] == [
        "r_high_z_pml", "r_high_z_pml_tall", "r_high_z_pml_ez_axis",
    ]
    assert sum(case[-1] for case in probe.CASES) == 26

    # PER CASE, NOT ONE SHARED TABLE. Since the 2026-09-02 dispatch expansion the
    # composer selects the cylindrical real FUSED pair, so the probe pins each case
    # to the exact composition it produces. That is twelve slot assertions where the
    # single `EXPECTED` table this replaced made four, and it is the only form in
    # which the axis case's difference is visible: an ACTIVE axis source routes the
    # electric half through the deposit-repair bracket (Leading/TrailingRepairPlan)
    # rather than the fused pair, and a shared table averaged that away.
    assert probe.EXPECTED_BY_CASE == {
        "r_high_z_pml": {
            "step_B": "CylindricalRealFusedMagneticPairPlan",
            "update_H": "NoopPlan",
            "step_D": "CylindricalRealFusedElectricPairPlan",
            "update_E": "NoopPlan",
        },
        "r_high_z_pml_tall": {
            "step_B": "CylindricalRealFusedMagneticPairPlan",
            "update_H": "NoopPlan",
            "step_D": "CylindricalRealFusedElectricPairPlan",
            "update_E": "NoopPlan",
        },
        "r_high_z_pml_ez_axis": {
            "step_B": "CylindricalRealFusedMagneticPairPlan",
            "update_H": "NoopPlan",
            "step_D": "LeadingRepairPlan",
            "update_E": "TrailingRepairPlan",
        },
    }
    # every case carries its own pin, so a case added without one cannot ride in
    # on another's expectations
    assert {case[0] for case in probe.CASES} == set(probe.EXPECTED_BY_CASE)
    # the generic cylindrical fusion routes must still refuse BY NAME
    assert probe.CYLINDRICAL_REFUSALS == (
        "fused_pair_B", "fused_pair_D", "fused_pair_dispersive_D")
    source = PROBE.read_text(encoding="utf-8")
    assert "source_effect_seen" in source and "uint32" in source
    assert "case {name} step {step}/{steps}" in source


def test_cylindrical_composition_provenance_welds_the_live_central_seam():
    """The current device verdict is inseparable from the exact routed bytes."""
    package = pathlib.Path(launch_module.__file__).parent
    record = json.loads((package / "fingerprints.json").read_text(encoding="utf-8"))
    gate = record["cylindrical_composition_gate"]

    assert record["current_host_gate"] == "cylindrical_composition_gate"
    # The run identity, not a Slurm job id: Slurm cannot place this gate on a free
    # device on the certifying box, so it was run directly against a pinned one.
    #
    # This constant MOVES WITH THE RECORD, and moving it is not a way of making a
    # red test green: the digest assertions below are what carry the weld, and this
    # line only says which run took them. It last moved when the five-family
    # integration recert replaced direct_213162; it moves here because wiring the
    # FOUR remaining certified families into plan_step changed launch.py and
    # __init__.py again, so the gate was re-run against those bytes (2026-08-14,
    # pinned GPU 0, same probe, same product 3/3 and 26/26, certified subnormal
    # policy pinned explicitly). Both prior runs are preserved verbatim under
    # gate["superseded_runs"].
    assert gate["run_id"] == "recert_cylindrical_20260814T095103Z"
    assert gate["run_mechanism"].startswith("DIRECT pinned run, not a Slurm job")
    assert "slurm_job_id" not in gate and "slurm_state" not in gate
    assert any(prior["run_id"] == "direct_213162"
               for prior in gate["superseded_runs"])
    assert gate["product"]["cases_exact"] == "3/3"
    assert gate["product"]["complete_steps_exact"] == "26/26"
    assert gate["product"]["source_cases_nonvacuous"] == "1/1: on-axis Ez"
    assert gate["dispatch"].startswith("DISABLED")

    api = package.parent
    sources = {
        "driver.py": api / "driver.py",
        "fields.py": api / "fields.py",
        "grid.py": api / "grid.py",
        "pml.py": api / "pml.py",
        "sources.py": api / "sources.py",
        "stepping.py": api / "stepping.py",
        "kernels.py": package / "kernels.py",
        "coverage.py": package / "coverage.py",
        "launch.py": package / "launch.py",
        "cylindrical_triton.py": package / "cylindrical_triton.py",
        "__init__.py": package / "__init__.py",
    }
    # A gate's ``source_sha256`` is a FACT about the bytes that gate executed, and
    # it is never rewritten to match a later tree. So a file that changed after
    # the gate ran must be DECLARED rather than smoothed away, and the live bytes
    # must still be pinned by something — otherwise the check goes vacuous at
    # exactly the change it exists to catch. ``driver_dispatch`` is that
    # declaration: it carries the live digest of the wiring it describes and names
    # the driver-route gate whose run clears the drift (and re-runs this one).
    declared = json.loads(
        (package / "fingerprints.json").read_text(encoding="utf-8"))["driver_dispatch"]
    drift = declared["source_sha256"]
    for name, path in sources.items():
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in drift:
            assert drift[name] == live, (
                f"{name} drifted again after the dispatch shape was recorded; "
                "re-cut fingerprints.json['driver_dispatch']['source_sha256']")
            # READ THE CLEARING STATUS, not a proxy for it. This used to assert
            # what_would_certify_this.status == "NOT RUN", which was a stand-in for
            # "the drift has not been cleared yet" and stopped meaning that the
            # moment the driver-route gate ran (2026-08-15) WITHOUT re-running this
            # one. The two are different claims: that gate compares two driver
            # routes and runs no mutation leg, no source-case non-vacuity check and
            # no per-sub-step probe, so it cannot license re-typing THIS gate's
            # digest. The declaration now says so in its own field.
            assert declared["declared_drift_clearing"]["status"] == "NOT CLEARED", (
                f"{name}'s drift is declared but recorded as cleared; re-cut this "
                "gate's source_sha256 from the run that cleared it instead")
            continue
        # THE RAW BYTES, WITH NO ALLOWANCE — the same rule as the Metal weld test,
        # and restored here for the same reason. A ``weld_survives_edit`` fall-through
        # stood between these two lines and was removed on 2026-08-29: this gate's
        # record carries neither ``device_sha256`` nor ``code_sha256`` (its digest keys
        # are artifact/log/probe/recert/slurm/source), and that helper returns False
        # whenever it has neither — so the branch could admit nothing it was asked
        # about, while standing ready to start admitting the moment either block was
        # added to this record. Measured before removing it: all four paths pass the
        # strict comparison, so nothing was resting on it.
        #
        # A DECLARED drift is already handled above, by ``driver_dispatch``. That is the
        # one allowance, it is per-file and recorded, and it is where a real one belongs;
        # a drift with no declaration is fixed by re-running the gate, never by relaxing
        # this comparison or by typing a digest in.
        assert gate["source_sha256"][name] == live, (
            f"{name} drifted from the bytes this gate executed; declare it under "
            f"driver_dispatch or re-run the gate")
    assert gate["probe_sha256"] == hashlib.sha256(PROBE.read_bytes()).hexdigest()
    assert gate["slurm_launcher_sha256"] == hashlib.sha256(SLURM.read_bytes()).hexdigest()
