"""Merge-bar tests for :mod:`meep_gpu.triton_kernels.no_pml` — the no-absorber curl.

Everything here runs on a laptop with NO GPU and NO Triton, which is the point.
The kernel's bit-identity is measured on a CUDA host by
``parity/meep_gpu/gate_triton_no_pml.py``; what a laptop can hold is (a) that the
measured verdict and the source it was measured on cannot drift apart, (b) that
the optional dependency stays optional, and (c) the predicate itself, which is
pure Python and is the half that decides whether a wrong answer is possible.

THE SHIPPED MODULE OWNS THE OPTIONAL-DEPENDENCY BOUNDARY. It installs an
unlaunchable sentinel when Triton is absent, leaving the pure-Python predicate and
refusal path usable on every host. The kernel body is never executed here and
could not be — there is no device. Separate tests block Triton at import time,
exercise the real predicate, require a clear launch failure, and pin the real
``@triton.jit`` source that the CUDA gate compiled.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib
import pathlib
import sys
from types import SimpleNamespace

import numpy
import pytest

from meep_gpu.device_identity import weld_survives_edit
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import GUARD_SPELLING, PACKAGE_DIR, code_of
from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels import launch as launch_module

MODULE_PATH = PACKAGE_DIR / "no_pml.py"
MODULE_NAME = "meep_gpu.triton_kernels.no_pml"
GATE_PATH = PACKAGE_DIR.parents[1] / "parity" / "meep_gpu" / "gate_triton_no_pml.py"


# ---------------------------------------------------------------------------
# Importing the module: the optional dependency stays optional
# ---------------------------------------------------------------------------

def _block_triton(monkeypatch):
    """Make ``import triton`` raise, the way a machine without it would."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in list(sys.modules):
        if name == "triton" or name.startswith("triton."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in (MODULE_NAME, "meep_gpu.triton_kernels.kernels",
                 "meep_gpu.triton_kernels"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_the_package_still_imports_with_triton_absent_and_does_not_pull_in_no_pml(
        monkeypatch):
    """Adding a third kernel module must not make the package need Triton.

    ``meep_gpu.triton_kernels`` is imported by the engine's own capability checks,
    so it has to answer on a machine that has never heard of Triton. This module is
    NOT exported from ``__init__`` (dispatch is deferred — see the WIRING note), and
    that is checked here as behaviour rather than left as an intention: if a later
    hand wires it in with a top-level import, this test is what fails.
    """
    _block_triton(monkeypatch)
    package = importlib.import_module("meep_gpu.triton_kernels")
    assert package.triton_available() is False
    assert MODULE_NAME not in sys.modules, (
        "importing the package pulled in no_pml, which needs Triton")
    for name in ("plain_curl_coverage", "plan_plain_curl"):
        assert name in package.__all__
        assert callable(getattr(package, name))
    # The package still answers a coverage question, which is its contract.
    fields, pml = _build(pml_thickness=2, storage=True)
    assert package.explain(fields, pml).reasons


def test_the_module_answers_coverage_but_its_kernel_fails_clearly_without_triton(
        monkeypatch):
    """The predicate is optional-dependency-safe; launching the kernel is not."""
    _block_triton(monkeypatch)
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _build()
    verdict = module.plain_curl_coverage(fields, pml, "step_B")
    assert verdict.covered is False
    assert len(verdict.reasons) == 1 and "cupy" in verdict.reasons[0]
    with pytest.raises(ImportError, match="optional.*triton|triton.*optional"):
        module.plain_curl_step[(1,)]


def test_the_shipped_source_declares_a_module_level_jit_kernel():
    """The real decorator remains, with an explicit unavailable-launch sentinel."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "    import triton\n" in source
    assert "    import triton.language as tl\n" in source
    assert "except ImportError" in source
    assert "class _UnavailableKernel" in source
    assert "\n@triton.jit\ndef plain_curl_step(" in source


# ---------------------------------------------------------------------------
# The kernel is welded to the verdict measured on it
# ---------------------------------------------------------------------------

def _module_constant(name: str):
    """Read a module-level constant out of the SOURCE, without importing it."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level constant of no_pml.py")


def _kernel_source_segment() -> str:
    """``plain_curl_step``'s signature through its final store, decorator excluded."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "plain_curl_step":
            return ast.get_source_segment(source, node)
    raise AssertionError("no_pml.py no longer defines plain_curl_step")


def test_the_recorded_kernel_hash_matches_the_shipped_kernel():
    """The source-segment pin complements the full-file package fingerprint.

    ``fingerprints.json`` now enumerates this specialized module and its central
    composition gate. This narrower constant still pins the kernel body separately,
    so prose and optional-import work cannot be mistaken for a numerical recut.

    These kernels are bit-identical to ``stepping.py`` because their expression
    grouping survives Triton's MLIR pipeline, which is an EMPIRICAL fact about one
    Triton version and one source text — hence a hash and not a review.
    """
    live = hashlib.sha256(_kernel_source_segment().encode("utf-8")).hexdigest()
    assert _module_constant("KERNEL_SOURCE_SHA256") == live, (
        "plain_curl_step changed since the bit-identity gate certified it. Re-run "
        "parity/meep_gpu/gate_triton_no_pml.py on a CUDA host and re-cut both "
        "KERNEL_SOURCE_SHA256 and GATE_VERDICT.")


def test_the_no_pml_composition_fingerprint_names_the_exact_gated_bytes():
    """The public route, specialized kernel and clean Slurm verdict are one record."""
    import json

    record = json.loads((PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    specialized = record["specialized_kernel_sources"]["no_pml.py"]
    assert specialized["kernel"] == "plain_curl_step"
    live = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
    # ONE HOME FOR THE RULE (device_identity.py:209). The kernel this entry pins
    # is a ``@triton.jit`` body plus the constants it reads, and the test above
    # already holds that segment against KERNEL_SOURCE_SHA256 -- so an edit the
    # shared rule proves reaches neither leaves the record true. False for
    # anything it cannot establish, and the byte rule below then applies.
    if specialized["sha256"] != live and not weld_survives_edit(
            MODULE_PATH, specialized, "no_pml.py"):
        assert specialized["sha256"] == live, (
            "no_pml.py drifted from the recorded specialized kernel source and "
            "the change reaches executable code")

    gate = record["no_pml_composition_gate"]
    assert gate["product"]["single_launch_guarded"].startswith("108/108")
    assert gate["product"]["multistep_guarded"].startswith("12/12")
    assert "12/12 complete FdtdDriver.step calls" in gate["real_engine_route"]["covered"]
    assert gate["real_engine_route"]["refusals"].startswith("5/5")
    assert gate["dispatch"].startswith("DISABLED")


def test_the_recorded_verdict_is_pass_shaped_and_states_its_step_budget():
    """A record that degenerated into "whatever we measured" is not a record.

    Note what is NOT asserted: that the unguarded control is ``0/N``. On this kernel
    it is 38/108 identical — 70 caught — because contracting the multiply into the
    subtraction lands on the same float32 in the remaining 38. That is a weaker
    control than the PML curl's 0/120 and the record says so; what is required here
    is that it CAUGHT something, and that the sweep was not confined to a Courant of
    0.5, where the discrepancy largely vanishes (12/36 differ, against 29/36 at each
    of the two non-power-of-two Courants).
    """
    verdict = _module_constant("GATE_VERDICT")
    guarded = verdict["single_launch_guarded"].split()[0]
    ran, total = guarded.split("/")
    assert ran == total and int(total) > 0, guarded
    caught = verdict["single_launch_unguarded"]
    assert "CAUGHT" in caught and not caught.startswith("0/0"), caught
    assert set(verdict["courants"]) - {0.5}, "the sweep must leave the power of two"
    assert any(c not in (0.5, 0.25, 0.125) for c in verdict["courants"])
    # The budget is a claim, not a footnote: it must be present, finite, and the
    # step after it must be the one that differs.
    assert verdict["first_divergent_step"] == verdict["step_budget"] + 1
    assert verdict["step_budget"] > 0
    # At least one mutation caught, and the deliberately-uncaught control labelled.
    assert verdict["mutations_caught"]["add_instead_of_subtract"] == "54/54"
    assert verdict["mutations_caught"]["swap_derive_operand_order"] == "0/54"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("gate_triton_no_pml_contract", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _passing_gate_payload():
    return {
        "legs": ["product", "multistep", "mutations", "engine"],
        "product_guarded": {
            "n_identical": 2, "n_live": 2, "worst_differing_floats": 0,
            "all_targets_changed": "2/2",
        },
        "product_unguarded": {"n_identical": 1, "n_live": 2},
        "multistep": {"cases": [{"bit_identical": True}]},
        "mutations": {"mutations": [
            {"mutation": "add_instead_of_subtract", "n_caught": 2, "n_live": 2},
            {"mutation": "flatten_grouping", "n_caught": 1, "n_live": 2},
            {"mutation": "drop_inverse_epsilon", "n_caught": 1, "n_live": 2},
            {"mutation": "derive_centre_only", "n_caught": 1, "n_live": 2},
            {"mutation": "drop_ownership_mask", "n_caught": 1, "n_live": 2},
            {"mutation": "periodic_ghost_on_metallic", "n_caught": 1, "n_live": 2},
            {"mutation": "swap_derive_operand_order", "n_caught": 0, "n_live": 2},
        ]},
        "engine": {
            "covered": [{
                "bit_identical": True,
                "first_divergent_step": None,
                "steps": 12,
                # The certified post-null-family shape (planner recert
                # 2026-08-14): the curls plus a null plan on each constitutive
                # sub-step, so ``replaces`` carries all four names while
                # ``selected`` records the monkeypatched curl slots only.
                "replaces": ["step_B", "update_H", "step_D", "update_E"],
                "selected": {"step_B": "PlainCurlPlan", "step_D": "PlainCurlPlan"},
            }],
            "refusals": [{
                "plan_is_none": {"step_B": True, "step_D": True},
                "central_selected": {"step_B": "PmlCurlPlan"},
            }],
        },
    }


def test_the_cuda_gate_is_self_enforcing_instead_of_only_recording_bad_results():
    gate = _load_gate_module()
    payload = _passing_gate_payload()
    summary = gate.validate_payload(payload)
    assert summary["status"] == "passed"
    payload["product_guarded"]["n_identical"] = 1
    with pytest.raises(AssertionError, match="guarded product"):
        gate.validate_payload(payload)


def test_the_cuda_gate_fails_if_the_central_route_selects_the_wrong_product():
    gate = _load_gate_module()
    payload = _passing_gate_payload()
    payload["engine"]["covered"][0]["selected"]["step_D"] = "PmlCurlPlan"
    with pytest.raises(AssertionError, match="central no-PML route"):
        gate.validate_payload(payload)


def test_the_guard_is_the_packages_one_spelling_and_appears_once():
    """One launch site, spelled the shared way. Duplicated on purpose.

    ``test_every_launch_site_passes_the_shared_guard_constant`` globs the package
    and would catch a literal here too — but it asserts a total, so a NEW module
    that both adds a site and misspells it shows up there as an arithmetic mismatch
    with no name attached. This says which file.
    """
    code = code_of(MODULE_PATH)
    assert code.count("enable_fp_fusion=") == 1, "no_pml has one launch site"
    assert code.count(GUARD_SPELLING) == 1
    assert "options=" not in code


# ---------------------------------------------------------------------------
# The predicate — real Grid/Fields/PML, stub Triton
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def no_pml_module():
    """The shipped module itself must be importable without a test-only stub."""
    return importlib.import_module(MODULE_NAME)


def _build(cell_size=(0.8, 0.8, 0.8), pml_thickness=0, storage=False, **grid_kwargs):
    """A real Grid/Fields/PML triple on NumPy. Default: NO absorber, NO storage."""
    grid = Grid(resolution=10.0, cell_size=cell_size, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=False)
    if storage:
        fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=pml_thickness)


def _reasons(module, fields, pml, sub_step="step_B"):
    """Refusal reasons with the backend clause dropped — a laptop has no CuPy."""
    return [r for r in module.plain_curl_coverage(fields, pml, sub_step).reasons
            if "cupy" not in r]


# --- the positive verdict --------------------------------------------------

@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_target_configuration_is_refused_only_for_the_backend(no_pml_module, sub_step):
    """A predicate no configuration satisfies is as useless as one everything does.

    The configuration this kernel is FOR is a plain periodic run with no absorber
    and no PML storage — ``2d_plain``. Every clause but CuPy must pass, on BOTH
    sub-steps, because coverage here is a set per sub-step and not per run.
    """
    fields, pml = _build()
    verdict = no_pml_module.plain_curl_coverage(fields, pml, sub_step)
    assert verdict.covered is False
    assert len(verdict.reasons) == 1, verdict.reasons
    assert "cupy" in verdict.reasons[0]


def test_metallic_boundaries_are_covered_too(no_pml_module):
    fields, pml = _build(boundaries="metallic")
    assert _reasons(no_pml_module, fields, pml) == []


def test_the_two_curl_kernels_coverage_sets_are_a_partition(no_pml_module):
    """Clause 3 INVERTED is the module's whole reason to exist, so it is pinned.

    ``stepping._pml_is_active`` is the test the array path itself branches on, so an
    active layer belongs to ``pml_curl_step`` and an absent or all-zero-face one
    belongs here. Exactly one of the two predicates may admit any configuration —
    never both, and (for a configuration both would otherwise take) never neither.
    """
    from meep_gpu.triton_kernels.coverage import pml_curl_coverage

    plain_fields, no_layer = _build()
    pml_fields, layer = _build(pml_thickness=2, storage=True)

    plain_here = _reasons(no_pml_module, plain_fields, no_layer) == []
    plain_there = [r for r in pml_curl_coverage(plain_fields, no_layer).reasons
                   if "cupy" not in r] == []
    pml_here = _reasons(no_pml_module, pml_fields, layer) == []
    pml_there = [r for r in pml_curl_coverage(pml_fields, layer).reasons
                 if "cupy" not in r] == []

    assert (plain_here, plain_there) == (True, False)
    assert (pml_here, pml_there) == (False, True)


# --- the refusals, one per silent-wrong-answer surface ---------------------

def test_an_active_layer_is_refused_and_names_the_other_kernel(no_pml_module):
    fields, pml = _build(pml_thickness=2, storage=True)
    reasons = _reasons(no_pml_module, fields, pml)
    assert any("active PML layer" in r for r in reasons), reasons


def test_pml_storage_without_an_active_layer_is_refused(no_pml_module):
    """Clause 3b: the configuration an "absence of a blocker" reading would admit.

    ``Fields.enable_pml_storage`` is a ONE-WAY switch that also changes what
    ``get_E``/``get_H`` return. With storage on and the layer inert, ``get_H``
    serves a STORED H that ``update_H`` declines to write, so the curl would
    difference a frozen field — no exception, no NaN, a smooth wrong answer.
    """
    fields, pml = _build(pml_thickness=0, storage=True)
    reasons = _reasons(no_pml_module, fields, pml)
    assert any("inert" in r for r in reasons), reasons


def test_the_inert_storage_reason_does_not_fire_where_the_layer_is_active(no_pml_module):
    """A right verdict with a false reason is still a defect in a reasons list.

    The recon build tested ``fields._pml_active`` alone, so every active-PML refusal
    carried "while the layer is inert" as a second reason — a sentence that is false
    for that case. The verdict was never wrong; the explanation was, and a refusal
    list nobody can trust is a refusal list nobody reads.
    """
    fields, pml = _build(pml_thickness=2, storage=True)
    reasons = _reasons(no_pml_module, fields, pml)
    assert any("active PML layer" in r for r in reasons), reasons
    assert not any("inert" in r for r in reasons), reasons


def test_a_conductivity_is_refused_by_name(no_pml_module):
    """Clause 8: without PML a conductivity is a DIFFERENT update, not this one.

    ``mp.Absorber`` routes to ``stepping._apply_conductive_update`` — a three-factor
    form, not ``f -= curl``. Naming it keeps this kernel's coverage set and the
    conductivity kernel's disjoint by construction rather than by coincidence.
    """
    fields, pml = _build()
    fields.set_d_conductivity(numpy.full(fields.grid.shape, 0.5, dtype=numpy.float32))
    reasons = _reasons(no_pml_module, fields, pml)
    assert any("conductivity" in r for r in reasons), reasons


def test_complex_storage_is_refused(no_pml_module):
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    reasons = _reasons(no_pml_module, fields, PML(grid=grid, thickness=0))
    assert any("complex" in r for r in reasons), reasons


def test_a_mirror_plane_is_refused(no_pml_module):
    fields, pml = _build(symmetry=("X",))
    assert any("mirror" in r for r in _reasons(no_pml_module, fields, pml))


def test_a_nonzero_k_point_is_refused(no_pml_module):
    """And this is why the no-PML corpus value is zero, not merely small.

    ``material-dispersion.py`` is one of the only two example scripts that declare
    no boundary layer, and ``sim.run_k_points`` forces complex fields — so the one
    script the coverage set might have claimed is refused right here.
    """
    fields, pml = _build(k_point=(0.3, 0.0, 0.0))
    assert any("k_point" in r for r in _reasons(no_pml_module, fields, pml))


def test_bfast_and_special_kz_are_refused(no_pml_module):
    fields, pml = _build()
    fields.grid.beta = 0.25
    assert any("beta" in r for r in _reasons(no_pml_module, fields, pml))
    fields.grid.beta = 0.0
    fields.grid.bfast_scaled_k = (0.0, 0.0, 0.5)
    if fields.grid.bfast_active:
        assert any("BFAST" in r for r in _reasons(no_pml_module, fields, pml))


def test_an_unknown_sub_step_raises_rather_than_refusing(no_pml_module):
    """A typo must not read as "not covered" — that is how a kernel goes unused."""
    fields, pml = _build()
    with pytest.raises(ValueError, match="sub_step"):
        no_pml_module.plain_curl_coverage(fields, pml, "step_H")


# --- the layout clauses, including the one the recon build dropped ---------

def test_a_non_contiguous_target_is_refused(no_pml_module):
    """The gap this file closed on landing, and it is a silent-wrong-answer gap.

    A reversed view has the right shape and the right dtype and is read in the
    WRONG ORDER by a kernel that indexes a flat cell index. The recon build
    re-stated coverage.py's volume check by hand and dropped the contiguity clause
    on the way; the landed module calls ``coverage._layout_reasons`` so the two
    paths cannot drift.
    """
    fields, pml = _build()
    fields.Bx = fields.Bx[::-1]
    assert any("C-contiguous" in r for r in _reasons(no_pml_module, fields, pml))


def test_a_non_contiguous_inverse_epsilon_is_refused_under_derive(no_pml_module):
    """Same hazard on the derived source, which the array path would have hidden.

    ``xp.multiply(D, inv_eps, out=...)`` broadcasts and strides correctly whatever
    the view; the kernel's ``load(g + idx) * load(e + idx)`` does not.
    """
    fields, pml = _build()
    reversed_inverse = numpy.ascontiguousarray(
        numpy.full(fields.grid.shape, 0.5, dtype=numpy.float32))[::-1]
    fields.inverse_epsilon_for = lambda name: reversed_inverse
    reasons = _reasons(no_pml_module, fields, pml)
    assert any("C-contiguous" in r for r in reasons), reasons


def test_a_scalar_inverse_epsilon_is_refused_under_derive(no_pml_module):
    fields, pml = _build()
    fields.inverse_epsilon_for = lambda name: numpy.float32(0.5)
    reasons = _reasons(no_pml_module, fields, pml)
    assert any("shape" in r for r in reasons), reasons


def test_the_derive_clauses_are_asked_only_when_the_source_is_derived(no_pml_module):
    """``step_D`` differences B directly, so no inverse epsilon can disqualify it."""
    fields, pml = _build()
    fields.inverse_epsilon_for = lambda name: numpy.float32(0.5)
    assert _reasons(no_pml_module, fields, pml, "step_B") != []
    assert _reasons(no_pml_module, fields, pml, "step_D") == []


def test_off_diagonal_epsilon_is_refused_for_derive_and_only_for_derive(no_pml_module):
    """Clause 15 — the subtle one, and the one a later edit is most likely to lose.

    The PML curl ADMITS off-diagonal chi1inv because its entire effect is inside
    ``update_E``. ``DERIVE`` moves part of that product INTO this kernel, and the
    row-coupling term reads the OTHER components — which this kernel never loads.
    So the same feature is admitted on one branch and refused on the other, and
    that asymmetry is deliberate.
    """
    fields, pml = _build()
    # The property reads the installed rows, so the STATE is set rather than the
    # property stubbed: a test that patched the property would still pass if the
    # engine changed what makes a row off-diagonal.
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": numpy.full(
        fields.grid.shape, 0.1, dtype=numpy.float32)}}
    assert fields.has_offdiagonal_epsilon is True
    assert any("off-diagonal" in r for r in _reasons(no_pml_module, fields, pml)), (
        "DERIVE must refuse off-diagonal chi1inv")
    # step_D never derives, so the same field is not a refusal there.
    assert _reasons(no_pml_module, fields, pml, "step_D") == []
    # And the PML curl still admits it, which is the asymmetry being pinned.
    from meep_gpu.triton_kernels.coverage import pml_curl_coverage

    pml_fields, layer = _build(pml_thickness=2, storage=True)
    pml_fields._chi1inv_offdiagonal = {"Ex": {"Ey": numpy.full(
        pml_fields.grid.shape, 0.1, dtype=numpy.float32)}}
    assert [r for r in pml_curl_coverage(pml_fields, layer).reasons
            if "cupy" not in r] == []


# --- the plan ---------------------------------------------------------------

def test_derive_tracks_stores_E_rather_than_being_chosen(no_pml_module):
    """``stores_E`` is what ``_read_component`` branches on, so it is what DERIVE is.

    Getting it backwards is not a crash. With ``stores_E`` True and ``DERIVE`` 1 the
    kernel would difference ``D * inv_eps`` while the engine's monitors read a
    stored E one polarization out of date — smooth, plausible, wrong.
    """
    fields, _ = _build()
    assert fields.stores_E is False
    assert no_pml_module.SUB_STEPS["step_B"]["displacement"] == ("Dx", "Dy", "Dz")
    assert no_pml_module.SUB_STEPS["step_D"]["displacement"] is None
    # step_D's sources ARE the B arrays: without PML, get_H returns B itself.
    assert no_pml_module.SUB_STEPS["step_D"]["sources"] == ("Bx", "By", "Bz")
    assert no_pml_module.SUB_STEPS["step_B"]["backward"] == 0
    assert no_pml_module.SUB_STEPS["step_D"]["backward"] == 1


def test_the_step_plan_replaces_nothing_on_a_numpy_host_and_says_why(no_pml_module):
    """Never None and never silent: a refusal reports which sub-step and what for."""
    fields, pml = _build()
    plan = no_pml_module.plan_plain_step(fields, pml)
    assert plan.plans == {}
    assert set(plan.refusals) == {"step_B", "step_D"}
    for reasons in plan.refusals.values():
        assert any("cupy" in r for r in reasons), reasons


def test_the_whole_step_is_not_one_launch_and_the_object_refuses_to_pretend(
        no_pml_module):
    """The driver runs sources and wall wipes BETWEEN step_B and step_D.

    A ``PlainStepPlan.run()`` that launched both back to back would skip them, so it
    raises instead of composing something the array path does not do.
    """
    plan = no_pml_module.PlainStepPlan({}, {})
    with pytest.raises(NotImplementedError, match="boundary"):
        plan.run()


def test_the_plan_builder_returns_none_out_of_coverage(no_pml_module):
    fields, pml = _build(pml_thickness=2, storage=True)
    assert no_pml_module.plan_plain_curl(fields, pml, "step_B") is None
    assert no_pml_module.plan_plain_curl(fields, pml, "step_D") is None


# --- predicate mutations: the part that proves the clauses are load-bearing --

def _mutate(module, function, needle: str, replacement: str):
    """Recompile one predicate with one clause rewritten, in the module's namespace.

    Same device as ``test_triton_kernels._mutate``. Note this works only because
    these are plain Python functions: a ``@triton.jit`` body CANNOT be exec-ed —
    Triton reads it back with ``inspect.getsource`` and an exec-ed mutant dies at
    DECORATION, before any launch. Kernel mutations go through a real file on disk;
    the device gate does that.
    """
    import inspect
    import textwrap

    source = textwrap.dedent(inspect.getsource(function))
    mutated = source.replace(needle, replacement)
    assert mutated != source, f"the clause moved; update this mutation: {needle!r}"
    namespace = dict(module.__dict__)
    exec(compile(mutated, "<mutated no_pml>", "exec"), namespace)  # noqa: S102
    return namespace[function.__name__]


def test_mutation_dropping_the_inverted_pml_clause_is_caught(no_pml_module,
                                                             monkeypatch):
    """MUTATION 1: delete clause 3 and admit an active absorber.

    With the clause gone the configuration must be ADMITTED — that is what makes it
    load-bearing rather than decorative. If something else refused an active PML
    anyway, this assertion failing would be the signal that clause 3's coverage is
    not actually being measured by anything.
    """
    fields, pml = _build(pml_thickness=2, storage=False)
    assert any("active PML layer" in r
               for r in _reasons(no_pml_module, fields, pml))

    mutated = _mutate(no_pml_module, no_pml_module._no_pml_grid_reasons,
                      'if pml is not None and getattr(pml, "is_active", False):',
                      "if False:")
    monkeypatch.setattr(no_pml_module, "_no_pml_grid_reasons", mutated)
    assert _reasons(no_pml_module, fields, pml) == [], (
        "clause 3 is the only thing refusing an active absorber; if this fails, the "
        "mutation is being masked and clause 3's coverage is not measured")


def test_mutation_widening_the_boundary_set_by_one_axis_is_caught(no_pml_module,
                                                                  monkeypatch):
    """MUTATION 2: admit ``mirror`` as a covered ghost rule.

    Both halves are asserted, the way the PML suite does it: with the boundary set
    widened, the separate fold clause must still refuse — and with the fold clause
    ALSO gone, the predicate accepts a configuration the kernel cannot step, which
    is the failure the pair exists to prevent.
    """
    fields, pml = _build(symmetry=("X",))
    assert any("mirror" in r for r in _reasons(no_pml_module, fields, pml))

    monkeypatch.setattr(no_pml_module, "COVERED_BOUNDARIES",
                        no_pml_module.COVERED_BOUNDARIES + ("mirror",))
    still_refused = _reasons(no_pml_module, fields, pml)
    assert any("folded by a mirror plane" in r for r in still_refused), still_refused

    monkeypatch.setattr(no_pml_module, "_call",
                        lambda obj, name, *a, **kw: kw.get("default"))
    assert _reasons(no_pml_module, fields, pml) == []


def test_mutation_dropping_the_layout_check_is_caught(no_pml_module, monkeypatch):
    """MUTATION 3: stop asking coverage.py what a valid volume looks like.

    With ``_layout_reasons`` neutered, a reversed Bx — same shape, same dtype,
    wrong strides — is admitted. That is the exact silent wrong answer the landing
    pass closed, so it gets a mutation rather than a comment.
    """
    fields, pml = _build()
    fields.Bx = fields.Bx[::-1]
    assert any("C-contiguous" in r for r in _reasons(no_pml_module, fields, pml))

    monkeypatch.setattr(no_pml_module, "_layout_reasons",
                        lambda fields, shape, names: [])
    monkeypatch.setattr(no_pml_module, "_volume_reasons",
                        lambda label, array, shape: [])
    assert _reasons(no_pml_module, fields, pml) == []


# ---------------------------------------------------------------------------
# Structure — what the kernel must keep saying to stay bit-identical
# ---------------------------------------------------------------------------

def test_the_curl_grouping_is_the_one_stepping_uses():
    """``dtdx * ((sf - f) + (s - ss))``, parenthesised, three times.

    Read off the source rather than the imported module, so it holds with no
    Triton. C and Triton alike would associate ``sf - f + s - ss`` as
    ``((sf - f) + s) - ss``, which is a DIFFERENT float32 — and the gate's
    ``flatten_grouping`` mutation is caught 36/54 rather than 54/54 precisely
    because a one-cell axis makes one difference an exact zero. On the shapes this
    kernel is actually benchmarked on, that mutation is INVISIBLE, so the grouping
    needs a structural pin as well as a numerical one.
    """
    segment = _kernel_source_segment()
    assert "curl0 = dtdx * ((c_y - c) + (b - b_z))" in segment
    assert "curl1 = dtdx * ((a_z - a) + (c - c_x))" in segment
    assert "curl2 = dtdx * ((b_x - b) + (a - a_y))" in segment


def test_the_plain_tail_is_a_subtraction_and_carries_no_pml_arithmetic():
    """``target -= curl`` (stepping.py:537) and nothing else.

    The whole claim of this module is that the split-field recurrence, the
    auxiliary and the two coefficient tables are ABSENT — so their absence is
    asserted rather than assumed. A ``kms``/``sinv`` appearing here would mean the
    file drifted back toward ``kernels.pml_curl_step`` without the merge.
    """
    segment = _kernel_source_segment()
    for index in range(3):
        assert f"v{index} = tl.load(f{index} + idx, mask=live, other=0.0) - curl{index}" \
            in segment
    for absent in ("kms", "sinv", "fu_", "kps"):
        assert absent not in segment, f"{absent!r} is PML arithmetic and must not be here"


def test_the_derive_product_keeps_the_array_paths_operand_order():
    """D on the LEFT, as ``stepping._read_component`` writes it (stepping.py:2448).

    Float multiplication is bitwise commutative — the gate's
    ``swap_derive_operand_order`` leg is 0/54 CAUGHT and is labelled an EXPECTED
    miss for exactly that reason. So this is a transcription-fidelity pin, not a
    numerical one, and it is worth having: the next person to move part of
    ``update_E`` into a kernel should find the convention already stated.
    """
    segment = _kernel_source_segment()
    assert "tl.load(g0 + idx, mask=live, other=0.0) * tl.load(e0 + idx, mask=live, other=0.0)" \
        in segment


def test_the_module_does_not_reach_into_another_tracks_file():
    """Ownership is enforced while the public integration stays lazy.

    Checked on the IMPORT GRAPH rather than on the text: the predicate legitimately
    reads ``grid.has_symmetry`` in order to refuse a fold, so grepping for the bare
    word ``symmetry`` would fire on the clause that keeps the two tracks apart.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    for forbidden in ("fastpath", "cuda_kernels", "symmetry", "dispersive_update_e"):
        assert not any(forbidden in name for name in imported), (
            f"no_pml.py imports {forbidden}: {sorted(imported)}")
    assert imported <= {"__future__", "typing", "triton", "triton.language",
                        ".coverage", ".kernels", ".launch", "..stepping"}, imported
    code = code_of(MODULE_PATH)
    for forbidden in ("fastpath", "cuda_kernels"):
        assert forbidden not in code, f"no_pml.py references {forbidden}"
    # It is exported through lazy wrappers: importing the package must still not
    # import this module on a host without Triton.
    init = code_of(PACKAGE_DIR / "__init__.py")
    assert "def plain_curl_coverage" in init
    assert "def plan_plain_curl" in init
    assert "from .no_pml import" in init


# ---------------------------------------------------------------------------
# Composition — no-PML and PML products form a fail-closed partition
# ---------------------------------------------------------------------------

def _install_composition_stubs(monkeypatch, *, pml_curl, plain_by_step):
    refused = coverage_module.Coverage(False, ("outside this routing test",))
    monkeypatch.setattr(
        launch_module, "pml_curl_coverage",
        lambda fields, pml, sub_step: pml_curl)
    monkeypatch.setattr(
        launch_module, "conductive_pml_curl_coverage",
        lambda fields, pml, sub_step: refused,
        raising=False,
    )
    monkeypatch.setattr(
        launch_module,
        "plain_curl_coverage",
        lambda fields, pml, sub_step: plain_by_step[sub_step],
        raising=False,
    )
    monkeypatch.setattr(
        launch_module, "constitutive_coverage", lambda fields, pml, side: refused)
    monkeypatch.setattr(
        launch_module, "dispersive_constitutive_coverage", lambda fields, pml: refused)


def test_plan_step_installs_both_plain_curls_when_the_PML_product_refuses(monkeypatch):
    refused = coverage_module.Coverage(False, ("no active PML",))
    admitted = coverage_module.Coverage(True, ())
    _install_composition_stubs(
        monkeypatch,
        pml_curl=refused,
        plain_by_step={"step_B": admitted, "step_D": admitted},
    )
    sentinels = {"step_B": object(), "step_D": object()}
    monkeypatch.setattr(
        launch_module,
        "plan_plain_curl",
        lambda fields, pml, sub_step, block=None: sentinels[sub_step],
        raising=False,
    )

    plan = launch_module.plan_step(SimpleNamespace(polarizations=()), object())

    assert plan.plans["step_B"] is sentinels["step_B"]
    assert plan.plans["step_D"] is sentinels["step_D"]
    assert plan.replaces == ("step_B", "step_D")
    assert "step_B" not in plan.reasons and "step_D" not in plan.reasons


def test_plan_step_refuses_a_PML_plain_curl_overlap(monkeypatch):
    admitted = coverage_module.Coverage(True, ())
    _install_composition_stubs(
        monkeypatch,
        pml_curl=admitted,
        plain_by_step={"step_B": admitted, "step_D": admitted},
    )
    monkeypatch.setattr(launch_module, "plan_pml_curl", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        launch_module, "plan_plain_curl", lambda *args, **kwargs: object(), raising=False)

    plan = launch_module.plan_step(SimpleNamespace(polarizations=()), object())

    assert "step_B" not in plan.plans and "step_D" not in plan.plans
    for name in ("step_B", "step_D"):
        assert any("ambiguous" in reason and "curl" in reason
                   for reason in plan.reasons[name])


def test_the_module_file_is_readable_as_utf8_without_a_locale():
    """A measurement-host trap, recorded as a test because it cost a gate run.

    The gate host's default locale is ASCII, so ``open()`` without an explicit
    encoding dies on any kernel file whose prose carries a non-ASCII character —
    and this one's does. Every reader in this suite and in the gate passes
    ``encoding="utf-8"``; this asserts the file genuinely needs it, so the rule
    keeps earning its place.
    """
    raw = MODULE_PATH.read_bytes()
    with pytest.raises(UnicodeDecodeError):
        raw.decode("ascii")
    assert raw.decode("utf-8")


def test_the_destination_is_where_the_gate_expects_it():
    assert MODULE_PATH.exists()
    assert MODULE_PATH == pathlib.Path(__file__).parent / "triton_kernels" / "no_pml.py"
