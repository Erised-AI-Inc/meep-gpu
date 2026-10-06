"""Tests for the COMPLEX fused magnetic pair — complex ``step_B`` into ``update_H``.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware —
byte identity against the array path and against the two separately certified
complex products this launch replaces — lives in
``parity/meep_gpu/probe_triton_complex_fused_magnetic_pair.py``, WHICH HAS NEVER
RUN. That is stated in the module under test and re-stated here, because a test
suite that is green while the device gate is unrun must not be mistaken for a
certification.

What IS pinned here:

* the EXPANSION licence — which arm this product binds, on which pattern set, from
  which artifact, and that the shipped rule refuses the broken records the gate
  arms it with. This is the clause that decides whether a kernel reproduces the
  array path's bytes at all, and it is checked here rather than only on a device;
* the seam — that ``REPLACES`` names the driver's own call sites, that the two
  symmetry fills really are inert on every configuration the halves admit (measured,
  with a folded control that separates), and that the wall clear is carried on BOTH
  word planes;
* every clause of the seam predicate, in both directions, and that the plan builder
  refuses exactly where the predicate does;
* the transcription, read off the shipped source: the complex curl's
  parenthesisation, the constitutive half's two-accumulation grouping, and that the
  weld replaced a load rather than adding one;
* the corpus number the module's docstring quotes, re-derived from the census
  artifact rather than trusted;
* the release — ``launch.py`` routes this module and its label is in
  ``fastpath.RELEASED_FUSED_ARMS``, so clause (8) admits the whole plan at dispatch.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from meep_gpu import expansion_refusal
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import complex_fields
from meep_gpu.triton_kernels import complex_fused_magnetic_pair as product

PACKAGE_DIR = pathlib.Path(product.__file__).parent
API_ROOT = PACKAGE_DIR.parents[1]
GATE = (API_ROOT / "parity" / "meep_gpu"
        / "probe_triton_complex_fused_magnetic_pair.py")
#: The CURRENT census cut. An earlier directory (``..._census_2026-08-19``) carries
#: the same derivation and the same numbers; its manifest went stale because it
#: listed the predicate module, which this derivation does not read. A manifest may
#: not be overwritten, so the round was re-cut with a correctly scoped one.
CENSUS_DIR = (API_ROOT / "parity" / "meep_gpu" / "results"
              / "triton_complex_fused_magnetic_pair_census_2026-08-19T2355")
KEEP_PROBE = (API_ROOT / "parity" / "meep_gpu" / "results"
              / "expansion_probe_2026-08-17" / "expansion_probe_keep.json")


def build(cell=(3.0, 3.0, 0.0), boundaries="periodic", k_point=(0.23, 0.0, 0.0),
          mirrors=(), thickness=0.4, complex_storage=True, dimensions=2):
    """A real complex Grid/Fields/PML triple on NumPy, with PML storage allocated."""
    grid = Grid(resolution=10.0, cell_size=cell, dimensions=dimensions,
                boundaries=boundaries, k_point=k_point,
                symmetry=tuple(Mirror(axis, phase) for axis, phase in mirrors))
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=thickness)


def probe_record():
    return json.loads(KEEP_PROBE.read_text(encoding="utf-8"))


def residual(verdict):
    """The reasons that are not the NumPy-host backend clause.

    MATCHED ON THE CLAUSE, NOT ON THE WORD 'cupy'. The complex predicates' EXPANSION
    refusal names the backend the probe must be for, so a filter on the word swallows
    it — and a refusal that vanishes from the residual is a refusal the caller reads
    as an admission.
    """
    return [reason for reason in verdict.reasons
            if not ("array module is" in reason and "not cupy" in reason)]



def magnetic_deposit(fields):
    """A REAL source that publishes the index the injection writes.

    :class:`Source` below carries a ``field_type`` and nothing else: enough to PLACE
    a source in a seam, and deliberately not enough to CARRY one -- the repair
    refuses it by name for publishing no deposit index. A case about the carry needs
    the engine's own source, and one that deposits nothing would make the admitting
    assertion pass by measuring an empty scatter, so both are checked here rather
    than assumed at each call site.
    """
    from meep_gpu import deposit_repair  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    source = VolumeSource(grid=fields.grid, component="Hy",
                          center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    assert source._n_source_points, "the case deposits nothing and measures nothing"
    assert deposit_repair._deposit_index(source) is not None, (
        "the case cannot exercise the repair: this source publishes no deposit index")
    return source


class Source:
    def __init__(self, field_type: str) -> None:
        self.field_type = field_type


def load_gate():
    spec = importlib.util.spec_from_file_location("probe_complex_fused_B", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# The EXPANSION licence — which arm, on what evidence
# ---------------------------------------------------------------------------

def test_this_product_asks_the_licence_over_the_base_pattern_set_and_no_more():
    """The pattern-set claim, pinned to the base set the halves already cover.

    The sibling tranches pass a SUPERSET because each launches an operand
    orientation the base four do not cover; this product launches none, so it asks
    the same question over the same set. If a later edit adds one, this test is
    what forces the new pattern to be declared before the arm can be licensed.
    """
    assert product.PRODUCT_PROBE_PATTERNS == complex_fields.PROBE_PATTERNS


def test_the_kernel_body_calls_only_the_three_licensed_multiply_helpers():
    """The other half of the pattern-set claim, read off the shipped body.

    A fourth helper would be a fourth operand orientation. Declaring the base set
    while calling something outside it is exactly how a kernel gets an arm licensed
    for multiplications it does not perform.
    """
    gate = load_gate()
    body = gate._shipped_text("complex_fused_curl_constitutive_B")
    called = {name for name in product.LICENSED_MULTIPLY_HELPERS if name in body}
    assert called == set(product.LICENSED_MULTIPLY_HELPERS), called
    stray = [line.strip() for line in body.splitlines()
             if ("_mul_" in line or "_rotate_" in line)
             and not any(name in line for name in product.LICENSED_MULTIPLY_HELPERS)]
    assert not stray, stray


def test_the_keep_cut_probe_licenses_FMA_V1_on_this_products_pattern_set():
    """The arm this product binds, and the evidence it is bound from.

    Measured through the SHIPPED rule, not asserted: all four base patterns
    discriminate and agree, the basis is 'measured' rather than an
    ENVIRONMENT_DEFAULTS row, and the record's resolved policy is the one every
    complex arm in this package was certified under.
    """
    verdict = complex_fields.expansion_license(probe_record(),
                                               product.PRODUCT_PROBE_PATTERNS)
    assert verdict["refusals"] == []
    assert verdict["arm"] == "FMA_V1"
    assert verdict["basis"] == "measured"
    assert verdict["non_discriminating"] == []
    assert set(verdict["discriminating"]) == set(product.PRODUCT_PROBE_PATTERNS)
    assert verdict["policy_resolved"] == complex_fields.CERTIFIED_UNDER_SUBNORMAL_POLICY
    assert verdict["candidate_policy"] == verdict["policy_resolved"]


def test_a_keep_cut_licence_does_not_survive_a_flush_candidate_set():
    """The comparison, not the verdict, is what a policy boundary breaks.

    A record whose candidates were computed under one policy and whose bytes were
    cut under the other is broken whatever arm it named — the 2026-08-11
    54/48/125-word NEITHER is this same defect wearing a different answer.
    """
    record = probe_record()
    record["candidates"] = dict(record.get("candidates") or {})
    record["candidates"]["policy"] = "flush"
    verdict = complex_fields.expansion_license(record,
                                               product.PRODUCT_PROBE_PATTERNS)
    assert verdict["arm"] is None
    assert any("broken comparison" in reason for reason in verdict["refusals"]), \
        verdict["refusals"]


def test_the_gates_expansion_licence_leg_refuses_every_broken_record_it_arms():
    gate = load_gate()
    row = gate.expansion_licence_leg()
    assert row["passed"], row["findings"]
    assert row["licensed"], "the leg scored no artifact at all"
    assert {entry["arm"] for entry in row["licensed"]} == {"FMA_V1"}
    assert all(entry["basis"] == "measured" for entry in row["licensed"])
    assert {entry["case"] for entry in row["falsification"]} == {
        "neither_pattern", "missing_pattern",
        "candidates_cut_under_another_policy", "evidence_free_ambiguity",
        "discriminating_patterns_disagree", "not_a_record", "host_backend"}
    assert all(entry["passed"] for entry in row["falsification"]), row["falsification"]


# ---------------------------------------------------------------------------
# The seam: what the driver runs there, and what is inert
# ---------------------------------------------------------------------------

def test_the_product_declares_the_seam_passes_the_driver_actually_calls():
    driver_source = (PACKAGE_DIR.parent / "driver.py").read_text(encoding="utf-8")
    for name in product.REPLACES:
        assert f"{name}(self.fields" in driver_source or f'dispatch("{name}"' \
            in driver_source, name
    assert product.REPLACES == ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                                "fill_folded_far_ghosts_B", "update_H")


def test_both_symmetry_fills_return_on_the_has_symmetry_guard():
    """The claim that makes this weld a straight one, read off the source.

    ``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B`` both delegate to a
    helper whose FIRST statement is ``if not grid.has_symmetry(): return``. That is
    what lets this product carry neither pass, and it is checked structurally so a
    guard that moved or gained a condition is a test failure rather than a silently
    swallowed pass.
    """
    tree = ast.parse((PACKAGE_DIR.parent / "stepping.py").read_text(encoding="utf-8"))
    guarded = 0
    for name in ("_fill_symmetry_ghost_cells", "_fill_folded_far_ghosts"):
        node = next(found for found in ast.walk(tree)
                    if isinstance(found, ast.FunctionDef) and found.name == name)
        first_if = next((index for index, item in enumerate(node.body)
                         if isinstance(item, ast.If)), None)
        assert first_if is not None, name
        guard = node.body[first_if]
        assert "has_symmetry" in ast.unparse(guard.test), (name, ast.unparse(guard.test))
        assert isinstance(guard.body[0], ast.Return), name
        # NOTHING MAY WRITE BEFORE THE GUARD. A local binding (``grid =
        # fields.grid``) is fine; a loop or a subscript store ahead of it would mean
        # the pass touches a volume before deciding whether it should run at all,
        # and this product's whole claim is that it touches none.
        for statement in node.body[:first_if]:
            written = ast.unparse(statement)
            assert not isinstance(statement, (ast.For, ast.While)), (name, written)
            assert "[" not in written.split("=")[0], (name, written)
        guarded += 1
    assert guarded == 2


def test_the_gate_measures_the_two_fills_inert_here_and_live_on_a_fold():
    """Argument is not measurement, so the gate measures it — with a control.

    On an unfolded complex run the five-pass array-path seam and the three-pass
    weld must be bit-identical. The folded control must SEPARATE: without it the
    leg cannot tell an inert pass from a dead one.
    """
    gate = load_gate()
    row = gate.symmetry_inertness_leg()
    assert row["passed"], row["findings"]
    unfolded = [entry for entry in row["rows"] if not entry["folded"]]
    assert len(unfolded) == 3
    assert all(entry["identical"] for entry in unfolded), unfolded
    assert all(entry["arrays_moved"] for entry in unfolded), unfolded
    control = next(entry for entry in row["rows"] if entry["folded"])
    assert not control["identical"], control


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def test_a_complex_bloch_grid_is_admitted_modulo_the_numpy_host():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_magnetic_pair_coverage(
            fields, pml, (), probe=probe_record())
    assert residual(verdict) == []


def test_an_electric_source_is_admitted_and_a_magnetic_one_is_refused_by_name():
    """The clause that caps this family, in BOTH directions.

    The driver injects a magnetic source between the two halves (driver.py:3283)
    and an electric one in the D/E seam. A predicate that refused both would be
    correct and useless; one that refused neither would drop a deposit.
    """
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        record = probe_record()
        electric = product.complex_fused_magnetic_pair_coverage(
            fields, pml, (Source("D"),), probe=record)
        # (1) CARRIED since 2026-08-30: a real magnetic deposit in this seam.
        carried = product.complex_fused_magnetic_pair_coverage(
            fields, pml, (magnetic_deposit(fields),), probe=record)
        # (2) STILL REFUSED BY NAME: no publishable deposit index.
        magnetic = product.complex_fused_magnetic_pair_coverage(
            fields, pml, (Source("B"),), probe=record)
    assert residual(electric) == []
    assert residual(carried) == [], residual(carried)
    assert not magnetic.covered
    assert any("does not publish the index it writes" in reason
               for reason in magnetic.reasons), magnetic.reasons


def test_holding_the_carry_flag_False_puts_the_magnetic_seam_refusal_straight_back(
        monkeypatch):
    """(3) of the source clause: the admission above is the FLAG's doing, not a
    clause that quietly went away."""
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        monkeypatch.setattr(product, "CARRIES_DEPOSIT_REPAIR", False)
        held = product.complex_fused_magnetic_pair_coverage(
            fields, pml, (magnetic_deposit(fields),), probe=probe_record())
    assert not held.covered
    assert any("is magnetic" in reason for reason in held.reasons), held.reasons


def test_an_undeclared_source_list_is_a_refusal_and_not_an_empty_set():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_magnetic_pair_coverage(
            fields, pml, None, probe=probe_record())
    assert not verdict.covered
    assert any("was not declared" in reason for reason in verdict.reasons)


def test_a_fold_is_refused_by_this_products_own_clause_and_not_only_by_the_halves():
    """The seam clause must name the two passes, not lean on the halves' wording.

    If the complex tranche ever admits a fold, the halves' refusal disappears and
    this weld would silently swallow two passes that had started doing work. The
    clause is here for that day and the test is what keeps it honest.
    """
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build(boundaries=("metallic", "metallic", "periodic"),
                            k_point=(0.0, 0.0, 0.0), mirrors=(("Y", 1),))
        verdict = product.complex_fused_magnetic_pair_coverage(
            fields, pml, (), probe=probe_record())
    assert not verdict.covered
    seam = [reason for reason in verdict.reasons
            if "fill_symmetry_bc_B" in reason or "symmetry passes in this seam" in reason]
    assert seam, verdict.reasons


def test_a_missing_expansion_probe_is_a_refusal_not_a_guessed_arm():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_magnetic_pair_coverage(fields, pml, (),
                                                               probe={})
    assert not verdict.covered
    assert any("expansion probe artifact" in reason for reason in verdict.reasons)


def test_no_pml_is_refused_because_update_H_is_a_no_op_without_one():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_magnetic_pair_coverage(
            fields, None, (), probe=probe_record())
    assert not verdict.covered
    assert any("PML" in reason for reason in verdict.reasons)


def test_real_storage_is_refused_this_is_the_complex_arm():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build(complex_storage=False)
        verdict = product.complex_fused_magnetic_pair_coverage(
            fields, pml, (), probe=probe_record())
    assert not verdict.covered


def test_the_predicate_reports_both_halves_reasons_with_their_side_named():
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = product.complex_fused_magnetic_pair_coverage(fields, pml, ())
    prefixes = {reason.split(":")[0] for reason in verdict.reasons}
    assert "complex curl half" in prefixes, verdict.reasons
    assert "complex constitutive half" in prefixes, verdict.reasons


def test_the_builder_refuses_every_configuration_the_predicate_refuses():
    """``None`` is the only refusal, and it must track the predicate exactly.

    A licence refused at the coverage seam but still bound at the plan seam is how
    an unlicensed arm reaches a kernel.
    """
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        folded_fields, folded_pml = build(
            boundaries=("metallic", "metallic", "periodic"),
            k_point=(0.0, 0.0, 0.0), mirrors=(("Y", 1),))
        record = probe_record()
        for args, kwargs in (
            ((fields, pml, (Source("B"),)), {"probe": record}),
            ((fields, pml, None), {"probe": record}),
            ((fields, pml, ()), {"probe": {}}),
            ((fields, None, ()), {"probe": record}),
            ((folded_fields, folded_pml, ()), {"probe": record}),
        ):
            assert product.plan_complex_fused_magnetic_pair(*args, **kwargs) is None


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def _bare_arrays(shape=(4, 5, 6)):
    names = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz", "Ex", "Ey", "Ez",
             "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
    arrays = {name: np.ascontiguousarray(np.zeros(shape, dtype=np.complex64))
              for name in names}
    curl = {f"{stem}_{axis}": np.ones(shape[index], dtype=np.float32)
            for index, axis in enumerate("xyz") for stem in ("kms", "sinv")}
    constitutive = {f"{stem}_{axis}": np.ones(shape[index], dtype=np.float32)
                    for index, axis in enumerate("xyz") for stem in ("kps", "kms")}
    return arrays, curl, constitutive


def test_the_backward_constexpr_is_step_Bs_and_only_step_Bs():
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    assert product.BACKWARD == SUB_STEPS[product.CURL_SUB_STEP]["backward"] == 0
    arrays, curl, constitutive = _bare_arrays()
    plan = product.plan_complex_fused_magnetic_pair_from_arrays(
        arrays, curl, constitutive, (0, 0, 0), (False, False, False),
        (None, None, None), 0.35, 1)
    assert plan.backward == 0


def test_the_phase_table_is_rounded_to_complex64_and_not_conjugated_here():
    """Forward sub-step: the imaginary part is passed as measured.

    ``_phase_arguments`` negates it for a BACKWARD sub-step (the conjugate of
    ``_shift_down``). This product is forward-only, so a sign appearing here would
    be the D-half's convention leaking into the B half.
    """
    arrays, curl, constitutive = _bare_arrays()
    phase = complex(0.5, 0.25)
    plan = product.plan_complex_fused_magnetic_pair_from_arrays(
        arrays, curl, constitutive, (0, 0, 0), (False, False, False),
        (phase, None, None), 0.35, 1)
    assert plan.phased == (1, 0, 0)
    assert plan.phase_values[:2] == (float(np.float32(phase.real)),
                                     float(np.float32(phase.imag)))
    # An unphased axis passes (1.0, 0.0) under PH = 0 and emits no multiply.
    assert plan.phase_values[2:] == (1.0, 0.0, 1.0, 0.0)


def test_the_plan_refuses_a_cell_count_whose_word_index_overflows_int32():
    """``2 * idx`` is computed in int32; the bound is halved to compensate.

    The from-arrays route runs no predicate at all, so the bound is checked in the
    plan rather than only in coverage — an overflowed word index is a wild write,
    not a wrong number.
    """
    arrays, curl, constitutive = _bare_arrays()

    class _Huge:
        shape = (2 ** 11, 2 ** 11, 2 ** 10)  # 2**32 cells
        dtype = np.dtype("complex64")

        class flags:
            c_contiguous = True

        def view(self, _dtype):
            return self

    arrays = dict(arrays)
    arrays["Bx"] = _Huge()
    with pytest.raises(ValueError, match="overflows"):
        product.plan_complex_fused_magnetic_pair_from_arrays(
            arrays, curl, constitutive, (0, 0, 0), (False, False, False),
            (None, None, None), 0.35, 1)


def test_a_real_storage_volume_is_refused_at_the_word_view():
    arrays, curl, constitutive = _bare_arrays()
    arrays = dict(arrays)
    arrays["Hx"] = np.zeros((4, 5, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="complex64"):
        product.plan_complex_fused_magnetic_pair_from_arrays(
            arrays, curl, constitutive, (0, 0, 0), (False, False, False),
            (None, None, None), 0.35, 1)


# ---------------------------------------------------------------------------
# The deferral, and the absence of Triton
# ---------------------------------------------------------------------------

def test_the_module_imports_and_answers_coverage_with_triton_absent(monkeypatch):
    """The merge bar is a laptop, so the predicates must answer without Triton."""
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in list(sys.modules):
        if name.endswith("complex_fused_magnetic_pair"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(
        "meep_gpu.triton_kernels.complex_fused_magnetic_pair")
    module = importlib.reload(module)
    assert module.complex_fused_curl_constitutive_B is None
    with expansion_refusal.declaring_run_policy("keep"):
        fields, pml = build()
        verdict = module.complex_fused_magnetic_pair_coverage(fields, pml, ())
    assert isinstance(verdict.reasons, tuple)
    with pytest.raises(ImportError, match="triton"):
        module.complex_fused_curl_constitutive_B_kernel()


def test_the_composer_routes_this_product_and_the_driver_seam_released_it():
    """ROUTED 2026-09-02 by the installer wave, RELEASED 2026-09-13 — both halves.

    This test replaces the deferral it used to make. Until 2026-09-02 it asserted
    that ``launch.py`` and ``fastpath.py`` named this module NOWHERE, which was the
    seam that kept a certified-but-unrouted product deferred. The wave routed it:
    ``launch.CERTIFIED_FUSED_PRODUCTS`` holds its row and
    ``_install_certified_fused_products`` builds it through the shared
    ``_install_fused_pair``, taking both slots only when ``_pair_may_absorb`` finds
    the arm table has already given them to the arms this kernel implements.

    WIRING WAS NOT RELEASING, and this is the file where that sentence ended. It
    was named ``..._and_dispatch_still_refuses_it`` for as long as the label sat
    outside ``fastpath.RELEASED_FUSED_ARMS``; the 2026-09-13 batch
    (``dispatch_fused_route_2026-09-13_phaseB``) drove the complex pair through the
    driver's own consults and seeded its weld from the 2026-09-12 identity fleet,
    so the label this product writes is now in ``fastpath.RELEASED_FUSED_ARMS`` and
    clause (8) ADMITS the whole plan rather than refusing it.

    THE PARTITION IS STILL EXACTLY ONE OF TWO: the label is in
    ``ARM_CERTIFICATION`` (its gate has a tracked ledger entry) and out of
    ``PENDING_DEVICE_GATE_ARMS`` (it graduated past the pending rung on release). A
    label released while still pending would be a plan claiming a certification the
    rung says it lacks.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu.triton_kernels import launch as _launch  # noqa: PLC0415

    source = _pathlib.Path(_launch.__file__).read_text(encoding="utf-8")
    assert "plan_complex_fused_magnetic_pair" in source
    row = _launch.CERTIFIED_FUSED_PRODUCTS["complex_fused_magnetic_pair"]
    assert row["module"] == "complex_fused_magnetic_pair"
    assert row["builder"] == "plan_complex_fused_magnetic_pair"
    label = row["label"]
    assert label == 'fused pair B (complex)'
    assert _fastpath.arm_is_fused(label)
    assert label in _fastpath.RELEASED_FUSED_ARMS
    assert (label in _fastpath.ARM_CERTIFICATION) != (
        label in _fastpath.PENDING_DEVICE_GATE_ARMS)
    assert label in _fastpath.FUSED_ARM_CONSTITUENTS
def test_this_module_is_welded_and_the_weld_describes_the_shipped_bytes():
    """This asserted the module had NO fingerprints entry, because it had none.

    It was gated and released on 2026-08-20, so the absence it pinned is gone and
    the property worth pinning is the weld's SOUNDNESS: it must record a PASS,
    name this module, and name it at the bytes on disk. An entry that drifts from
    the file it certifies is worse than no entry, because it reads as evidence.
    """
    import hashlib

    fingerprints = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    weld = fingerprints["triton_complex_fused_magnetic_pair_device_gate"]
    assert weld["status"] == "PASS"
    # THE RUN FACTS LIVE PER ARCHITECTURE since 0.9.1 (``fastpath.RUNS``): each
    # record that still binds this entry's bytes must say which run it was.
    from meep_gpu import fastpath  # noqa: PLC0415
    live = fastpath.live_capabilities(weld)
    assert live, "no architecture's record binds the bytes this weld pins"
    for capability in live:
        run = weld[fastpath.RUNS][capability]
        assert run["artifact_sha256"] and run["recorded_utc"] and run["host"], capability
    # This weld records THIS TEST FILE's own hash among its sources, so a bytes
    # check that included the reader would score every edit to the test as a
    # "drift" -- a circular guard. Exclude the test file; the kernel module and
    # the probe (the bytes the gate actually measured) stay guarded, as the two
    # released siblings whose welds also name their test file already do.
    named = {name: digest for name, digest in weld["source_sha256"].items()
             if name.endswith("complex_fused_magnetic_pair.py")
             and not name.startswith("meep_gpu/test_")}
    assert named, "the weld does not name the module it certifies"
    for name, digest in named.items():
        live = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
        assert live == digest, f"{name} drifted from the bytes the gate executed"


def test_the_module_claims_identity_ONLY_through_the_weld_that_measured_it():
    """The docstring must report the run it took, and must not overclaim."""
    text = pathlib.Path(product.__file__).read_text(encoding="utf-8")
    head = text.split('"""')[1]
    assert "RELEASED 2026-08-20" in head
    # THE BOUND THAT DID NOT MOVE: a weld licenses a claim, not a dispatch.
    assert "NOT WIRED" in head


def test_the_gate_exists_names_the_product_and_reports_the_run_it_took():
    text = GATE.read_text(encoding="utf-8")
    head = text.split('"""')[1]
    assert "RELEASED 2026-08-20" in head
    assert "meep_gpu.triton_kernels.complex_fused_magnetic_pair" in text
    assert "complex_fused_curl_constitutive_B" in text


# ---------------------------------------------------------------------------
# The gate's own no-device legs, run here
# ---------------------------------------------------------------------------

def test_the_gates_transcription_leg_traces_every_arithmetic_line_to_its_source():
    gate = load_gate()
    row = gate.transcription_leg()
    assert row["passed"], row["findings"]
    assert row["curl_lines_checked"] == len(gate.CURL_LINES)
    assert row["constitutive_lines_checked"] == len(gate.CONSTITUTIVE_LINES)
    assert row["weld_lines_checked"] == len(gate.WELD_LINES)
    assert row["wall_lines_checked"] == len(gate.WALL_LINES)
    assert sorted(row["multiply_helpers_called"]) == sorted(
        product.LICENSED_MULTIPLY_HELPERS)


def test_the_gates_design_sweep_agrees_with_the_array_path_and_catches_its_flips():
    """The design, measured: which value reaches update_H, and in which order.

    A laptop cannot see a memory hazard inside one launch — that is the device
    gate's — but the ordering questions the weld answers are array-path questions
    and this is where they are decided.
    """
    gate = load_gate()
    row = gate.design_sweep_leg()
    assert row["passed"], row["findings"]
    by_knob = {}
    for entry in row["rows"]:
        by_knob.setdefault(entry["knob"], []).append(entry["identical"])
    assert all(by_knob["faithful"]), by_knob
    assert not all(by_knob["wall_clear_dropped"]), by_knob
    assert not all(by_knob["wall_clear_after_the_constitutive_read"]), by_knob
    assert not all(by_knob["history_written_before_it_is_read"]), by_knob


def test_the_gates_predicate_leg_exercises_every_clause_in_both_directions():
    gate = load_gate()
    row = gate.predicate_leg()
    assert row["passed"], row["findings"]
    assert {entry["case"] for entry in row["rows"]} == {
        "bloch_no_sources", "bloch_electric_source",
        "bloch_magnetic_deposit_is_carried",
        "bloch_magnetic_source_without_a_deposit_index",
        "bloch_magnetic_deposit_refused_when_the_flag_is_held_False",
        "undeclared_sources", "no_probe_artifact", "no_pml", "metallic_walls_k0",
        "folded_grid"}
    directions = {entry["case"]: entry["expect_admitted"] for entry in row["rows"]}
    # THE THREE ROWS THE 2026-08-30 CARRY ADDED, asserted by NAME and by direction.
    # A set equality alone would let a future round rename the carried row to the
    # refused one and stay green, so the directions are pinned individually: the
    # deposit is ADMITTED, the source that cannot publish its index is REFUSED, and
    # the deposit is REFUSED AGAIN once the flag is held down.
    assert directions["bloch_magnetic_deposit_is_carried"] is True
    assert directions["bloch_magnetic_source_without_a_deposit_index"] is False
    assert directions[
        "bloch_magnetic_deposit_refused_when_the_flag_is_held_False"] is False
    assert all(entry["passed"] for entry in row["builder_rows"]), row["builder_rows"]


def test_the_gates_seam_binding_leg_reads_the_driver_rather_than_remembering_it():
    gate = load_gate()
    row = gate.seam_binding_leg()
    assert row["passed"], row["findings"]
    assert row["windows"], "no step_B/update_H seam was found in driver.py"
    for window in row["windows"]:
        assert window["injects_a_source"], window
        assert window["passes_between"] == [
            "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B"]


def test_the_gate_arms_every_mutation_it_declares():
    """A rewrite that matches nothing reports a real defect as uncaught.

    That failure mode is not hypothetical in this tree: the fused-electric round
    lost its ``kernel=`` argument on the way to the plan and reported 4/4 real
    defects as 120/120 identical. Every rewrite here is applied to the SHIPPED text
    on the laptop, and a zero-hit rewrite fails at the merge bar rather than on the
    device.
    """
    gate = load_gate()
    source = gate._shipped_text("complex_fused_curl_constitutive_B")
    unarmed = []
    for name, _why, _expectation, rewrite in gate.mutation_table():
        mutated, hits = rewrite(source)
        if hits == 0 or mutated == source:
            unarmed.append(name)
    assert not unarmed, unarmed


def changed_lines(before, after):
    """Indices into ``before`` that a rewrite really replaced or deleted.

    A POSITIONAL zip IS WRONG HERE and was, on 2026-08-20: several rewrites
    change the line COUNT — ``m1_wall_clear_dropped`` turns three lines into two
    — so every line after the edit shifts by one and a zip comparison reports
    almost the whole kernel as changed. The guard analysis below then finds some
    top-level line among them, concludes the mutation is live, and passes. It
    passed on the very defect it was written to catch.

    SequenceMatcher reports the actual replaced and deleted ranges, which is what
    "the lines this mutation touches" means.
    """
    import difflib

    out = []
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            out.extend(range(i1, i2))
        elif tag == "insert":
            out.append(min(i1, len(before) - 1))   # the line the insert lands before
    return sorted(set(out))


def test_every_mutation_is_scored_on_a_grid_that_ENTERS_the_branch_it_rewrites():
    """A mutation run on a case that cannot see it is a null about the case.

    THIS TEST USED TO BE PER-GRID AND HAD TO BECOME PER-AXIS. It asserted that
    the one mutation case carried ``any(kind == "metallic")`` and a nonzero k,
    under the premise that "the mutation leg runs on ONE case — so that case must
    carry both". Both halves were wrong, and the first hid a real defect for four
    device rounds:

      * ``CASES[4]`` (bloch_x_metallic_y) declares ("periodic", "metallic",
        "periodic"). SOME axis is metallic, so the old assertion passed — but
        ``m1_wall_clear_dropped`` rewrites lines under ``if ZM_X:``, and x is
        PERIODIC there. The branch was never entered and the mutation reported
        UNCAUGHT while measuring nothing.
      * The two requirements genuinely conflict: a wall-clear mutation needs a
        metallic X and a phase mutation needs a nonzero k, and no case in the
        table has both. There is no single case that can carry every mutation,
        so ``MUTATION_CASE`` names one per mutation and this checks each against
        the branch it actually rewrites.

    Structural, so a mutation added later is covered without anyone remembering.
    """
    gate = load_gate()
    source = gate._shipped_text("complex_fused_curl_constitutive_B")
    lines = source.splitlines()

    def enclosing_guard(index: int) -> str:
        """The guard a changed line sits under — or IS.

        A LINE THAT IS ITSELF AN ``if`` COUNTS AS ITS OWN GUARD, and that is not a
        nicety. ``m1_wall_clear_dropped`` replaces three lines with two, DELETING
        the ``if ZM_X:`` header; asking for that line's ENCLOSING guard walks out
        to the surrounding scope, which is unguarded, so the mutation reads as
        live on any grid at all. Deleting a guard is only meaningful where the
        guard would have been entered, so the condition on the deleted line is
        exactly the requirement.
        """
        own = lines[index].strip()
        if own.startswith("if "):
            return own
        body_indent = len(lines[index]) - len(lines[index].lstrip())
        for above in range(index - 1, -1, -1):
            text = lines[above]
            if not text.strip():
                continue
            indent = len(text) - len(text.lstrip())
            if indent < body_indent and text.strip().startswith("if "):
                return text.strip()
            if indent < body_indent:
                body_indent = indent
        return ""

    AXES = "XYZ"
    offenders = []
    for name, _why, _expectation, rewrite in gate.mutation_table():
        mutated, hits = rewrite(source)
        assert hits, f"{name} is unarmed; the sibling test should have caught this"
        changed = changed_lines(lines, mutated.splitlines())
        if not changed:
            continue
        guards = {enclosing_guard(n) for n in changed}
        _index, case = gate.mutation_case_for(name)
        boundaries = case[3]
        kinds = ((boundaries,) * 3 if isinstance(boundaries, str)
                 else tuple(boundaries))
        k_point = case[4]

        def enterable(guard: str) -> bool:
            for axis, letter in enumerate(AXES):
                if f"ZM_{letter}" in guard or f"BC{letter} == METALLIC" in guard:
                    return kinds[axis] == "metallic"
                if f"PH{letter}" in guard:
                    return float(k_point[axis]) != 0.0
            return True                      # unclassified or unguarded: assume live

        # AT LEAST ONE, NOT ALL: most rewrites replace the same text in all three
        # components, so their lines sit under all three axis guards and a grid
        # satisfying one of them makes the mutation live.
        if not any(enterable(guard) for guard in guards):
            offenders.append(
                f"{name} rewrites lines under {sorted(guards)}, and it is scored "
                f"on {case[0]!r} with boundaries {kinds} and k {k_point} — NOT ONE "
                f"of those branches is entered, so the leg reports a verdict about "
                f"the case table rather than about the kernel")
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# The corpus number the module quotes
# ---------------------------------------------------------------------------

def test_the_docstrings_corpus_numbers_are_the_census_artifacts_own():
    """12 admitted against 1 for the electric twin, re-derived rather than trusted.

    The number in a docstring is a claim about a measurement; this reads the
    measurement. It also pins the ARM size and the four magnetic-source rows,
    because "16 rows" and "12 rows" are different statements and the difference is
    the whole point of the source clause.
    """
    record = json.loads((CENSUS_DIR / "corpus_admission.json").read_text(
        encoding="utf-8"))
    assert record["rows"] == 186
    assert record["coverage_column"] == "covered_modulo_backend"
    assert record["complex_fused_magnetic_pair_B_to_H"]["admitted"] == 12
    assert record["complex_fused_pair_D_to_E"]["admitted"] == 1
    assert record["arm"]["rows_clearing_both_B_to_H_halves"] == 16
    assert record["arm"]["of_those_declaring_a_MAGNETIC_source"] == 4
    assert record["admitted_if_the_magnetic_source_clause_were_dropped"] == 16
    assert record["is_an_upper_bound"] is True
    # The unlicensed control: the record every other Triton family's census was
    # derived from reports 0 complex rows, for a platform reason.
    control = record["unlicensed_census_control"]
    assert control["complex_pml_curl@step_B_covered_modulo_backend"] == 0
    assert control["rows"] == 186


def test_the_census_script_derives_from_the_licensed_record():
    """Reading the unlicensed census would report this family as admitting nothing.

    Not a style point: ``_wired``'s probe artifact states no resolved subnormal
    policy, so clause 13 refuses every complex family on every row and the complex
    column is 0 for a reason unconnected to any corpus row.
    """
    text = (CENSUS_DIR / "count_corpus_admission.py").read_text(encoding="utf-8")
    assert "predicate_coverage_2026-08-16_wired_convention" in text
    record = json.loads((CENSUS_DIR / "corpus_admission.json").read_text(
        encoding="utf-8"))
    assert record["census"].endswith("predicate_coverage_2026-08-16_wired_convention")
