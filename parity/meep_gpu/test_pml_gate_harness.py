"""The real-field PML gate's harness clauses, checked where there is no device.

THE GATE ITSELF NEEDS A GPU. Everything that DECIDES what the gate measures does
not: which operand classes it draws, whether its mutations still resolve against
the shipped kernel, which verdict key the runner reads, what its step budget is,
and whether a leg can report a pass for a mutation that never reached the
compiler. Every one of those is a silent wrong ANSWER rather than a crash — a
gate that measured nothing returns the same 120/120 a gate that measured
everything does — and until this file existed each was first exercised on the one
host that can launch a kernel, hours into a device run.

The clauses here are the ones the disposition names as missing on this track and
present on the sibling one (`the design notes (cuda-kernel-track-disposition)`
§1.3, §1.4, §1.5), plus one defect found while adding them: the mutation runner
was parsing a verdict key the probe had renamed, so every leg's `single_ran` read
0 and `ok` was False on all of them.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import inspect
import pathlib
import re
import sys
import types

import numpy as np
import pytest

HERE = pathlib.Path(__file__).parent
REPO_API = HERE.parents[1]
KERNEL_MODULE = REPO_API / "meep_gpu" / "cuda_kernels" / "step_curl_kernels.py"

CONSTITUTIVE_MODULE = REPO_API / "meep_gpu" / "cuda_kernels" / "constitutive_kernels.py"

#: The two kernels the certification covers; the CURL mutations must resolve on both.
CERTIFIED_SOURCE_ATTRIBUTES = ("_step_B_pml_real_kernel_code",
                               "_step_D_pml_real_kernel_code")

#: The real-storage constitutive pair — authored, not yet gated. Its mutations
#: must resolve on these and on nothing else.
CONSTITUTIVE_SOURCE_ATTRIBUTES = ("_update_H_pml_real_kernel_code",
                                  "_update_E_pml_real_kernel_code")

#: WHICH KERNELS EACH MUTATION TARGETS, and it is a partition, not a hint. A
#: mutation is written against one sub-step's expression tree, so "matches
#: nothing" is the correct answer in the other sub-step's kernels and a DEFECT in
#: its own. Before the constitutive pair landed there was only one family and the
#: distinction did not exist; folding the new mutations into the old check would
#: have demanded that a constitutive needle appear in a curl kernel.
#:
#: ``E_ONLY`` names the constitutive mutations that touch ``D * inv_eps``, which
#: exists only in the E kernel.
MUTATION_TARGETS = {
    "curl": ("regroup_stencil", "drop_metallic_mask", "swap_dsig_dsigu",
             "drop_fu_store", "commute_dtdx_scale", "reload_fu_from_memory",
             "read_fprev_after_store",
             # THE FOLD'S TWO MASKS, added with the BC_MIRROR_PERIODIC branch on
             # 2026-08-20. Curl-only for the same reason ``drop_metallic_mask`` is:
             # ``_mask_non_owned_cells`` is a CURL pass. The constitutive kernels
             # read their own cell and have no ownership mask at all, which is why
             # the fold cost them no device code and cost this pair a branch.
             "drop_folded_periodic_top_mask", "drop_folded_periodic_near_mask"),
    "constitutive": ("regroup_constitutive", "drop_fw_store",
                     "store_fw_before_reading_prev",
                     "commute_constitutive_scale",
                     # The component -> own-axis coefficient mapping. Only the
                     # constitutive kernels have a kps/kms pair per axis; the curl
                     # pair indexes kms/sinv and this needle finds nothing there.
                     "own_axis_to_x_for_all_three"),
    "constitutive_E_only": ("drop_inverse_epsilon", "inv_eps_left",
                            "bind_Ez_inv_eps_for_all_three"),
    # THE LINEAR-INDEX DECOMPOSITION IS THE SAME THREE LINES IN ALL FOUR KERNELS,
    # and this fourth family exists because the partition check found that out.
    # It was written for the constitutive pair — nothing armed the decomposition
    # there — and it resolves on the certified curl pair too, where nothing armed
    # it either. Rather than anchor the needle to one family and leave the other
    # hole open, it is declared against all four and armed as a leg on each
    # (``c8`` and ``m10``); each leg launches only its own sub-step, so the other
    # family's mutated-but-unlaunched source is inert.
    "shared_index_decomposition": ("fortran_order_index_decomposition",),
}

#: Every device source a mutation may target, in one tuple.
ALL_GATED_SOURCE_ATTRIBUTES = (CERTIFIED_SOURCE_ATTRIBUTES
                               + CONSTITUTIVE_SOURCE_ATTRIBUTES)


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    """The bit-identity probe. It imports CuPy in a try/except and runs without."""
    return _load(HERE / "probe_fused_kernel_bit_identity.py", "_probe_pml_gate")


@pytest.fixture(scope="module")
def runner():
    return _load(HERE / "run_pml_gate_mutations.py", "_runner_pml_gate")


@pytest.fixture(scope="module")
def lifted():
    return _load(HERE / "validate_pml_kernel_on_lifted_cases.py", "_lifted_pml_gate")


@pytest.fixture(scope="module")
def device_sources():
    """Every CUDA source string of the track, evaluated from the AST without CuPy.

    BOTH modules: the certified curl pair in ``step_curl_kernels.py`` and the
    real-storage constitutive pair in ``constitutive_kernels.py``. They are
    separate files because the first one's device strings are digest-pinned by
    ``certification.json``; they are one namespace here because a mutation is
    applied across whatever the adapter exposes.
    """
    if str(REPO_API) not in sys.path:
        sys.path.insert(0, str(REPO_API))
    from meep_gpu.cuda_kernels.test_certification_record import (  # noqa: PLC0415
        device_sources as extract, module_source)
    sources = dict(extract(module_source()))
    sources.update(extract(CONSTITUTIVE_MODULE.read_text(encoding="utf-8")))
    return sources


# --------------------------------------------------------------------------
# §1.4 — the operand class, and what each one provably can and cannot produce.
# --------------------------------------------------------------------------

def _census(arrays) -> dict:
    flat = np.concatenate([np.asarray(a, dtype=np.float32).ravel()
                           for a in arrays.values()])
    raw = flat.view(np.uint32)
    exponent = (raw >> 23) & 0xFF
    mantissa = raw & 0x7FFFFF
    return {
        "values": int(flat.size),
        "subnormals": int(((exponent == 0) & (mantissa != 0)).sum()),
        "negative_zeros": int((raw == 0x80000000).sum()),
        "zeros": int((raw == 0).sum()),
    }


@pytest.fixture
def keeps_subnormals():
    """The band classes are drawn with float32 arithmetic, so a process whose FPU
    flushes subnormals draws none of them. Measured: in a process that has imported
    MEEP on x86-64, MEEP's initialization leaves FTZ/DAZ set.

    Only that case is a declared skip. Any other flushing state is a flush leaked
    by an earlier test, which the band tests exist to catch, so it still fails."""
    import platform  # noqa: PLC0415

    from meep_gpu import backends  # noqa: PLC0415

    if (backends.subnormals_flushed() and "meep" in sys.modules
            and platform.machine() in ("x86_64", "AMD64")):
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip(
            "host_keeps_subnormals",
            "this process imported MEEP on x86-64, whose initialization flushes "
            "float32 subnormals, so the generator cannot draw the subnormal band")


def test_the_uniform_class_cannot_produce_a_subnormal_or_a_signed_zero(probe):
    """The claim the disposition makes about this leg, MEASURED rather than read.

    ``uniform(-1, 1)`` draws from the normal range and never lands on a zero, so a
    verdict cut from it says nothing about the value class a float32 subnormal
    policy governs. Re-running the gate under ``flush`` and under ``keep`` with
    only this class would return the same number twice and establish nothing —
    which is exactly what the 2026-08-09/10 record did.
    """
    census = _census(probe.pml_field_hosts((13, 11, 9), np.random.default_rng(4),
                                           "uniform"))
    assert census["values"] == 18 * 13 * 11 * 9
    assert census["subnormals"] == 0, census
    assert census["negative_zeros"] == 0, census
    assert census["zeros"] == 0, census


def test_the_band_class_produces_both_and_covers_every_array(
        probe, keeps_subnormals):
    """And the class that replaces it has to actually reach the band.

    Not just "contains a small number": subnormals in quantity, exact zeros,
    NEGATIVE zeros, and the smallest subnormal present as the minimum magnitude —
    otherwise the new axis is decoration.
    """
    arrays = probe.pml_field_hosts((13, 11, 9), np.random.default_rng(4),
                                   "subnormal_band")
    assert set(arrays) == set(probe.PML_FIELD_NAMES)
    census = _census(arrays)
    assert census["subnormals"] > census["values"] // 10, census
    assert census["negative_zeros"] > 0, census
    assert census["zeros"] > 0, census

    flat = np.concatenate([a.ravel() for a in arrays.values()])
    nonzero = np.abs(flat[flat != 0])
    assert float(nonzero.min()) == pytest.approx(1.4012985e-45, rel=1e-3), (
        "the smallest float32 subnormal is not in the draw, so the band's lower "
        "edge — where a flush policy and a keep policy differ most — is untested")
    # EVERY array, not just the fields: the auxiliaries are the recurrence's
    # state, and a policy that flushes fu compounds from the first launch.
    for name, array in arrays.items():
        raw = array.ravel().view(np.uint32)
        exponent, mantissa = (raw >> 23) & 0xFF, raw & 0x7FFFFF
        assert int(((exponent == 0) & (mantissa != 0)).sum()) > 0, name


def test_both_classes_are_in_the_gate_product_and_reported_apart(probe):
    """Reported apart, because in the band it is the ARRAY PATH that leaves IEEE.

    CuPy appends ``-ftz=true`` to every NVRTC compile, so folding the band class
    into the pass would report a kernel defect that is not there — and dropping
    it would report a byte identity the run does not have. The dispersive leg
    settled this shape already; this is the same move on the curl.
    """
    assert probe.PML_VALUE_CLASSES == ("uniform", "subnormal_band")
    cases = [
        {"guard_label": "fmad_false", "guard_is_primary": True,
         "value_class": "uniform",
         "vs_array_order": {"bit_identical": True}},
        {"guard_label": "fmad_false", "guard_is_primary": True,
         "value_class": "subnormal_band",
         "vs_array_order": {"bit_identical": False}},
    ]
    summary = probe.summarize_pml(cases)
    assert summary["primary_pass"] is True, (
        "the band class gated the verdict; a policy-visible difference in the "
        "band is not a kernel defect and must not read as one")
    assert summary["primary_ran"] == 1 and summary["primary_identical"] == 1
    assert summary["subnormal_band_ran"] == 1
    assert summary["subnormal_band_identical"] == 0
    assert summary["all_classes_ran"] == 2
    assert set(summary["per_value_class"]) == {"uniform", "subnormal_band"}


def test_the_split_leaves_the_other_legs_that_share_this_summarizer_alone(probe):
    """``summarize_pml`` is shared, and the dispersive leg has its own classes.

    The dispersive leg's are ``uniform``/``cancellation``, already split on top of
    this by ``summarize_dispersive``. Selecting "uniform" instead of excluding the
    band would have quietly dropped every dispersive ``cancellation`` case out of
    the counts its own log line calls OVERALL. A case with no value class at all
    still counts as a normal-number one, which is what keeps the fused-pair and
    whole-step legs reading the same as before.
    """
    unclassed = [{"guard_label": "fmad_false", "guard_is_primary": True,
                  "vs_array_order": {"bit_identical": True}}]
    assert probe.summarize_pml(unclassed)["primary_ran"] == 1

    dispersive_shaped = [
        {"guard_label": "fmad_false", "guard_is_primary": True,
         "value_class": klass, "vs_array_order": {"bit_identical": True}}
        for klass in probe.DISPERSIVE_VALUE_CLASSES]
    assert probe.summarize_pml(dispersive_shaped)["primary_ran"] == 2
    assert probe.summarize_pml(dispersive_shaped)["subnormal_band_ran"] == 0


# --------------------------------------------------------------------------
# §1.4 on the CONSTITUTIVE family: the axis the leg did not have.
# --------------------------------------------------------------------------

def _constitutive_hosts(probe, side, value_class, shape=(13, 11, 9), seed=4):
    return probe.constitutive_field_hosts(shape, side, np.random.default_rng(seed),
                                          value_class)


@pytest.mark.parametrize("side", ("H", "E"))
def test_the_constitutive_uniform_class_cannot_reach_the_band_it_is_run_under(
        probe, side):
    """WHY THE TWO-POLICY CONSTITUTIVE RUN WAS GOING TO ESTABLISH NOTHING.

    ``run_pml_gate_mutations.py`` prescribes running the battery twice, once per
    float32 subnormal policy, because "the two verdicts side by side are what
    closes §1.1". Under ``uniform`` no value in this leg can reach the band the
    policy governs, so the two runs agree by construction — the same finding the
    curl leg recorded and fixed, not carried into this family until 2026-08-15.

    Measured here rather than argued, on the leg's own fixture function.
    """
    census = _census(_constitutive_hosts(probe, side, "uniform"))
    assert census["subnormals"] == 0, census
    assert census["negative_zeros"] == 0, census
    assert census["zeros"] == 0, census
    flat = np.concatenate([a.ravel() for a in
                           _constitutive_hosts(probe, side, "uniform").values()])
    assert float(np.abs(flat[flat != 0]).min()) > 1e-10, (
        "the smallest magnitude the uniform class draws is still thirty orders "
        "above the largest subnormal")


@pytest.mark.parametrize("side", ("H", "E"))
def test_the_constitutive_band_class_reaches_it_and_leaves_the_material_normal(
        probe, side, keeps_subnormals):
    """The replacement axis has to actually land in the band — and only there.

    The FIELD state goes to the band; inverse epsilon and the absorber
    coefficients stay normal. Driving the material into the band too would make
    ``D * inv_eps`` an underflow to zero on both sides, which is a comparison of
    two zeros — the vacuity this file counts elsewhere, manufactured here.
    """
    hosts = _constitutive_hosts(probe, side, "subnormal_band")
    targets, auxiliaries, sources = probe.constitutive_names(side)
    stateful = {name: hosts[name] for name in targets + auxiliaries + sources}
    census = _census(stateful)
    assert census["subnormals"] > census["values"] // 10, census
    assert census["negative_zeros"] > 0, census
    assert census["zeros"] > 0, census
    for name, array in stateful.items():
        raw = array.ravel().view(np.uint32)
        exponent, mantissa = (raw >> 23) & 0xFF, raw & 0x7FFFFF
        assert int(((exponent == 0) & (mantissa != 0)).sum()) > 0, name

    if side == "E":
        material = {name: hosts[name] for name in hosts if name.startswith("inv_eps_")}
        assert _census(material)["subnormals"] == 0
        assert float(min(a.min() for a in material.values())) >= 0.2


def test_the_constitutive_fixture_draws_three_distinct_epsilon_volumes(probe):
    """LOAD-BEARING, not incidental — and nothing asserted it before.

    Binding one volume for all three components is defect 2 of the complex
    template and it is EXACTLY bit-identical under an isotropic epsilon. The only
    thing that can expose it is three independent volumes, so the leg
    ``c9_bind_Ez_inv_eps_for_all_three`` measures the kernel only for as long as
    this holds; a fixture that collapsed to one volume would report that leg
    UNCAUGHT and read as a correct kernel.
    """
    for value_class in probe.CONSTITUTIVE_VALUE_CLASSES:
        hosts = _constitutive_hosts(probe, "E", value_class)
        volumes = [hosts["inv_eps_Ex"], hosts["inv_eps_Ey"], hosts["inv_eps_Ez"]]
        for first in range(3):
            for second in range(first + 1, 3):
                assert volumes[first].tobytes() != volumes[second].tobytes(), (
                    f"{value_class}: inverse epsilon volumes {first} and {second} "
                    f"are byte-identical, so binding one for all three would be "
                    f"invisible")


def test_the_constitutive_band_is_in_the_gate_product_and_reported_apart(probe):
    """Reported apart, because in the band it is the ARRAY PATH that leaves IEEE."""
    assert probe.CONSTITUTIVE_VALUE_CLASSES == ("uniform", "subnormal_band")
    cases = [
        {"guard_label": "fmad_false", "guard_is_primary": True,
         "value_class": "uniform", "reference_moved_targets": 3,
         "vs_array_order": {"bit_identical": True}},
        {"guard_label": "fmad_false", "guard_is_primary": True,
         "value_class": "subnormal_band", "reference_moved_targets": 3,
         "vs_array_order": {"bit_identical": False}},
    ]
    summary = probe.summarize_pml(cases)
    assert summary["primary_pass"] is True
    assert summary["subnormal_band_ran"] == 1
    verdict = probe.pml_verdict({"constitutive_bit_identity": {"gate": summary}})
    assert verdict["constitutive_single_launch"] == "1/1"
    assert verdict["constitutive_single_launch_subnormal_band"] == "0/1"


def test_the_runner_reads_the_band_key_of_the_legs_own_family(runner):
    """A constitutive leg reading the curl's key reports None for a sweep that ran."""
    for family in ("curl", "constitutive"):
        assert runner.FAMILIES[family]["band_key"].startswith(
            "curl_" if family == "curl" else "constitutive_")
    verdict = {"constitutive_single_launch": "96/96",
               "constitutive_single_launch_subnormal_band": "40/96",
               "curl_single_launch_subnormal_band": None}
    leg = {"family": "constitutive"}
    assert verdict.get(runner.family_of(leg)["band_key"]) == "40/96"


def test_a_leg_whose_oracle_moved_nothing_cannot_report_a_pass(probe):
    """Zero equals zero, bitwise — the recurring way an unmeasurable run scores best.

    The band class makes this live rather than theoretical: seed every operand in
    the subnormal band, let both sides underflow to zero, and a byte comparison
    reports perfect agreement about nothing. So the pass requires at least one
    normal-number case in which the ARRAY-PATH reference actually moved a target,
    and the per-class counts carry ``moved_nothing`` so a vacuous class is visible
    in the summary rather than only in the cases.
    """
    dead = [{"guard_label": "fmad_false", "guard_is_primary": True,
             "value_class": "uniform", "reference_moved_targets": 0,
             "reference_moved_auxiliaries": 0,
             "vs_array_order": {"bit_identical": True}}]
    summary = probe.summarize_pml(dead)
    assert summary["primary_ran"] == 1
    assert summary["primary_identical"] == 1
    assert summary["primary_pass"] is False, (
        "every case was bit-identical and none of them moved anything; that is a "
        "comparison of two unchanged buffers")
    assert summary["per_value_class"]["uniform"]["moved_nothing"] == 1

    alive = dead + [{"guard_label": "fmad_false", "guard_is_primary": True,
                     "value_class": "uniform", "reference_moved_targets": 3,
                     "reference_moved_auxiliaries": 3,
                     "vs_array_order": {"bit_identical": True}}]
    assert probe.summarize_pml(alive)["primary_pass"] is True


def test_a_run_that_swept_only_the_band_does_not_report_a_pass(probe):
    """The other direction: no normal-number case means nothing the verdict is about."""
    summary = probe.summarize_pml([
        {"guard_label": "fmad_false", "guard_is_primary": True,
         "value_class": "subnormal_band",
         "vs_array_order": {"bit_identical": True}}])
    assert summary["primary_ran"] == 0
    assert summary["primary_pass"] is False


# --------------------------------------------------------------------------
# §1.3 — the step budget.
# --------------------------------------------------------------------------

def test_the_curl_multi_step_budget_is_the_one_the_sibling_track_certifies(probe):
    """8 is not a budget, it is a measured blind spot.

    ``triton_kernels/fingerprints.json``: "8, 10 and 6 consecutive steps all
    passed a divergence that 40 did not." Same probe, same comparator, a kernel of
    the same shape — so a CUDA record quoting 8 sits below a number the sibling
    track has already measured to be blind.
    """
    assert probe.PML_MULTI_STEP_COUNT == 60
    assert probe.MULTI_STEP_COUNT == 8, (
        "the dispersive and fused-pair legs share this constant and their "
        "CERTIFICATION RECORDS were cut at 8; raising it here re-scopes gates "
        "whose verdicts are already published, which is why the curl and the "
        "constitutive pair each have their own")


def test_the_constitutive_budget_is_the_one_its_record_was_cut_at(probe):
    """The budget and the record that quotes it must move together, or not at all.

    THIS TEST USED TO BE A TRIPWIRE and it fired as designed. While the pair was
    ungated it asserted ``CERTIFIED_KERNELS = ()`` in the kernel module, with the
    message "a constitutive kernel has been certified; its record now names a step
    budget and this constant can no longer move without re-scoping it". Both
    kernels were certified on 2026-08-15, so the tripwire has done its job and is
    replaced by the obligation it was pointing at.

    8 was never available here for the reason it is fixed on the dispersive legs:
    ``triton_kernels/fingerprints.json`` measured "8, 10 and 6 consecutive steps
    all passed a divergence that 40 did not", and the same tranche's laptop leg
    compares the expression tree over 60 — so at 8 the leg measuring the
    EXPRESSION TREE would have run deeper than the leg measuring the COMPILED
    KERNEL. Now that a record exists, the constant is pinned to it from both
    sides: lowering one without the other fails here.
    """
    assert probe.CONSTITUTIVE_MULTI_STEP_COUNT == 60
    record = json.loads(
        (REPO_API / "meep_gpu" / "cuda_kernels" / "certification.json")
        .read_text(encoding="utf-8"))
    block = record["constitutive_2026-08-15"]
    assert block["constitutive_multi_step_budget"] == probe.CONSTITUTIVE_MULTI_STEP_COUNT, (
        "the probe's constitutive multi-step budget and the budget its "
        "certification record was cut at have diverged; the record describes a "
        "gate that no longer exists")
    assert block["per_policy"]["keep"]["multi_step_budget"] == 60
    assert block["per_policy"]["flush"]["multi_step_budget"] == 60


def test_the_multi_step_budget_is_carried_into_the_artifact(probe):
    """A claim about N steps has to say which N it was — for BOTH families.

    The constitutive block carried no ``steps_budget`` at all, so every
    constitutive leg landed in ``MUTATION_SUMMARY.json`` with
    ``multi_step_budget: null`` and printed ``multi_identical=True@None``.
    """
    results = {"pml_multi_step": {"gate": {"steps_budget": 60,
                                           "all_steps_identical": True}}}
    verdict = probe.pml_verdict(results)
    assert verdict["curl_multi_step_budget"] == 60
    assert verdict["curl_multi_step_all_identical"] is True

    constitutive = probe._constitutive_multi_step_summary([
        {"identical_steps": probe.CONSTITUTIVE_MULTI_STEP_COUNT,
         "auxiliary_moved_after_the_first_step": True}])
    assert constitutive["steps_budget"] == probe.CONSTITUTIVE_MULTI_STEP_COUNT
    verdict = probe.pml_verdict({"constitutive_multi_step": {"gate": constitutive}})
    assert verdict["constitutive_multi_step_budget"] == 60
    assert verdict["constitutive_multi_step_all_identical"] is True


def test_a_constitutive_multi_step_run_whose_auxiliary_never_advanced_fails(probe):
    """``fw[i] = src`` is unconditional, so a fixed source is a fixed point.

    MEASURED on the leg as it stood: over 8 launches ``f_w`` changed against the
    previous step at step 1 only, and equalled the source exactly at every step —
    while the docstring said sources were "held fixed, so f_w compounds its own
    history". 59 of 60 launches adding no state makes "identical for 60 steps" a
    claim about one step repeated, so the budget alone is not enough and the leg
    has to show the auxiliary advancing.
    """
    runs = [{"identical_steps": probe.CONSTITUTIVE_MULTI_STEP_COUNT,
             "auxiliary_moved_after_the_first_step": False}]
    summary = probe._constitutive_multi_step_summary(runs)
    assert summary["all_steps_identical"] is True
    assert summary["auxiliary_advanced_past_the_first_step"] is False
    assert probe.pml_verdict({"constitutive_multi_step": {"gate": summary}})["pass"] \
        is False, (
            "every step was bit-identical and the auxiliary never moved past the "
            "first launch; that is one sub-step compared 60 times")


# --------------------------------------------------------------------------
# §1.5 — mutations that must be UNCAUGHT, and their discriminators.
# --------------------------------------------------------------------------

def _expected_attributes(name: str) -> tuple:
    """Which device sources ``name`` must rewrite, from :data:`MUTATION_TARGETS`."""
    if name in MUTATION_TARGETS["curl"]:
        return CERTIFIED_SOURCE_ATTRIBUTES
    if name in MUTATION_TARGETS["constitutive"]:
        return CONSTITUTIVE_SOURCE_ATTRIBUTES
    if name in MUTATION_TARGETS["constitutive_E_only"]:
        return ("_update_E_pml_real_kernel_code",)
    if name in MUTATION_TARGETS["shared_index_decomposition"]:
        return ALL_GATED_SOURCE_ATTRIBUTES
    raise AssertionError(
        f"{name} is in SOURCE_MUTATIONS and in no MUTATION_TARGETS family; the "
        f"partition has to say which kernels it is written against, or nothing "
        f"can tell 'matches nothing' apart from 'matches nothing YET'")


def test_every_source_mutation_still_resolves_on_the_kernels_it_targets(
        probe, device_sources):
    """Anchor rot, checked on every test run — the sibling clause this track lacked.

    A mutation whose needle no longer appears is never applied. The probe raises
    on a zero TOTAL at run time, but that is hours into a device run, and a total
    cannot see a needle that stopped matching one kernel of a pair while still
    matching the other. Here it is a second on a laptop, per kernel, both ways:

      * every mutation must resolve on every kernel it targets, and change it;
      * and must match NOTHING in the kernels it does not target. That direction
        is new with the constitutive pair and it is not decoration — a curl needle
        that started matching a constitutive kernel would be silently mutating a
        second sub-step, and the leg's verdict would no longer be about the defect
        it names.
    """
    stale = []
    for name, transform in probe.SOURCE_MUTATIONS.items():
        targets = _expected_attributes(name)
        for attribute, original in device_sources.items():
            if attribute not in (CERTIFIED_SOURCE_ATTRIBUTES
                                 + CONSTITUTIVE_SOURCE_ATTRIBUTES):
                continue  # the twelve dead complex kernels are not gated at all
            mutated, count = transform(original)
            if attribute in targets:
                if count == 0:
                    stale.append(f"{name}: matches nothing in {attribute}")
                elif mutated == original:
                    stale.append(f"{name}: matched {count}x in {attribute} and "
                                 f"changed nothing — it replaces its anchor with "
                                 f"itself")
            elif count:
                stale.append(f"{name}: matched {count}x in {attribute}, which it "
                             f"does not target; it is mutating a sub-step its "
                             f"verdict says nothing about")
    assert not stale, "\n".join(stale)


def test_the_gates_constitutive_reference_is_the_array_path(probe):
    """The ORACLE, checked against the engine it is supposed to be — on a laptop.

    ``reference_constitutive_step`` is a SECOND transcription of
    ``stepping._apply_constitutive_pml``, written into the probe so that
    "bit-identical" means agreement with the contract rather than agreement with
    whatever was typed twice. Nothing checked the second transcription against
    the first, and a reference that has drifted turns the whole gate into a
    kernel reproducing the probe's own arithmetic.

    Both sides, both groupings: ``array_order`` must match the engine and
    ``regrouped`` must not — the second half is the vacuity control, without
    which a comparison of two identical implementations of the same mistake
    would pass.
    """
    if str(REPO_API) not in sys.path:
        sys.path.insert(0, str(REPO_API))
    from meep_gpu import stepping  # noqa: PLC0415

    shape = (7, 5, 3)
    for side in ("H", "E"):
        rng = np.random.default_rng(20260815)
        targets, auxiliaries, sources = probe.constitutive_names(side)
        arrays = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                  for name in targets + auxiliaries + sources}
        if side == "E":
            for target in targets:
                arrays["inv_eps_" + target] = rng.uniform(
                    0.2, 0.9, size=shape).astype(np.float32)
        coefficients = {}
        for axis, name in enumerate(probe.AXIS_NAMES):
            broadcast = [1, 1, 1]
            broadcast[axis] = shape[axis]
            for label in ("kps", "kms"):
                coefficients[f"{label}_{name}"] = rng.uniform(
                    0.5, 1.0, size=shape[axis]).astype(np.float32).reshape(broadcast)

        engine = {name: arrays[name].copy() for name in targets + auxiliaries}
        for target, source_name, axis in probe.CONSTITUTIVE_TERMS[side]:
            axis_name = probe.AXIS_NAMES[axis]
            source = (arrays[source_name] * arrays["inv_eps_" + target]
                      if side == "E" else arrays[source_name])
            stepping._apply_constitutive_pml(
                engine[target], source,
                coefficients["kps_" + axis_name], coefficients["kms_" + axis_name],
                engine["f_w_" + target], scratch=None)

        for grouping, must_match in (("array_order", True), ("regrouped", False)):
            state = {name: arrays[name].copy() for name in targets + auxiliaries}
            probe.reference_constitutive_step(side, arrays, state, coefficients,
                                              grouping)
            same = all(state[name].tobytes() == engine[name].tobytes()
                       for name in targets + auxiliaries)
            assert same is must_match, (
                f"update_{side}/{grouping}: the gate's reference "
                f"{'diverges from' if must_match else 'agrees with'} "
                f"stepping._apply_constitutive_pml")


def test_the_null_mutations_are_declared_and_are_source_mutations(probe):
    """A null that is not in the mutation table is a leg that cannot run."""
    assert probe.PML_NULL_MUTATIONS == ("commute_dtdx_scale", "reload_fu_from_memory")
    assert probe.CONSTITUTIVE_NULL_MUTATIONS == ("inv_eps_left",
                                                 "commute_constitutive_scale")
    for name in probe.PML_NULL_MUTATIONS + probe.CONSTITUTIVE_NULL_MUTATIONS:
        assert name in probe.SOURCE_MUTATIONS, name


def test_the_constitutive_defects_the_gate_requires_all_have_a_hand_spelling(probe):
    """The gate owns the list; this track has to own a spelling for each.

    ``CONSTITUTIVE_SOURCE_MUTATIONS`` is the gate's statement of what the
    constitutive sub-step must be proof against. A name on that list with no
    entry in this track's table is a defect the hand kernels are never tested
    against — and it fails at run time with a KeyError hours into a device run,
    not here.
    """
    missing = [name for name in probe.CONSTITUTIVE_SOURCE_MUTATIONS
               if name not in probe.SOURCE_MUTATIONS]
    assert not missing, (
        f"the hand track has no spelling for {missing}; the constitutive gate "
        f"cannot arm those legs")


def test_each_constitutive_null_has_a_discriminating_leg_on_the_same_expression(
        probe, device_sources):
    """"Inert" only means something beside a defect on the same code that fails.

    ``inv_eps_left`` and ``drop_inverse_epsilon`` edit the same three
    multiplications; ``commute_constitutive_scale`` and ``regroup_constitutive``
    edit the same two accumulation lines. Each pair is one leg that must be
    UNCAUGHT and one that must be CAUGHT — measured here as "they touch the same
    bytes", so a null that drifted onto some other expression stops being a
    control and this says so.
    """
    source = probe.SOURCE_MUTATIONS
    # The SHIPPED E kernel, which carries both expressions — the accumulation in
    # its prelude and the three inverse-epsilon products in its body. Using the
    # real text rather than a hand-typed sample is what keeps this from passing
    # against a sample that has drifted away from the kernel.
    body = device_sources["_update_E_pml_real_kernel_code"]
    for null, discriminator in (("inv_eps_left", "drop_inverse_epsilon"),
                                ("commute_constitutive_scale", "regroup_constitutive")):
        null_out, null_hits = source[null](body)
        caught_out, caught_hits = source[discriminator](body)
        assert null_hits and caught_hits, (null, discriminator, null_hits, caught_hits)
        assert null_out != body and caught_out != body
        assert null_out != caught_out, (
            f"{null} and {discriminator} produce the same text; the pair is one "
            f"mutation twice, not a control and its discriminator")


def test_each_null_mutation_has_a_discriminating_leg_that_must_be_caught(runner):
    """"Inert" only means something beside a leg on the same code that can fail.

    ``reload_fu_from_memory`` and ``read_fprev_after_store`` touch the same two
    lines of ``pml_apply``; ``commute_dtdx_scale`` and ``regroup_stencil`` touch
    the same curl expression. Each pair is one leg that must be UNCAUGHT and one
    that must be CAUGHT, which is what turns "the comparator is not simply failing
    everything" into a measurement.
    """
    by_mutation: dict = {}
    for leg in runner.LEGS:
        args = list(leg["args"])
        if "--source-mutation" in args:
            # A mutation may carry MORE than one leg — regroup_stencil is both the
            # byte gate's m3 (caught) and the allclose harness leg's real defect.
            by_mutation.setdefault(
                args[args.index("--source-mutation") + 1], set()).add(leg["expect"])
    assert by_mutation.get("reload_fu_from_memory") == {"uncaught"}
    assert "caught" in by_mutation.get("read_fprev_after_store", set())
    assert by_mutation.get("commute_dtdx_scale") == {"uncaught"}
    assert "caught" in by_mutation.get("regroup_stencil", set())
    # The constitutive pair's two nulls and their discriminators.
    assert by_mutation.get("inv_eps_left") == {"uncaught"}
    assert "caught" in by_mutation.get("drop_inverse_epsilon", set())
    assert by_mutation.get("commute_constitutive_scale") == {"uncaught"}
    assert "caught" in by_mutation.get("regroup_constitutive", set())


def test_every_constitutive_leg_is_wired_to_the_constitutive_family(runner):
    """A constitutive leg run under the curl's wiring measures nothing and says so.

    The runner reads its counts out of a named artifact section and its fraction
    out of a named verdict key. Before the family table existed both were
    hard-coded to the curl's, so a constitutive leg would have read
    ``curl_single_launch`` — absent from its artifact, hence 0/0, hence
    ``single_ran == 0``, hence a failure whose message is about the harness and
    not about the kernel. This is what keeps the two wirings apart.
    """
    families = {leg["leg"]: leg.get("family", runner.DEFAULT_FAMILY)
                for leg in runner.LEGS}
    constitutive = {name for name, family in families.items()
                    if family == "constitutive"}
    assert constitutive, "the constitutive gate legs are gone"
    for name in constitutive:
        assert name.startswith(("c", "cn", "cm")), name
    for name, family in families.items():
        assert family in runner.FAMILIES, (name, family)
        wiring = runner.FAMILIES[family]
        assert set(wiring) == {"experiments", "single_section", "multi_section",
                               "single_key", "band_key", "multi_key"}
    # The curl legs must not have drifted onto the constitutive experiments, and
    # vice versa: the experiment string is what decides which kernel is launched.
    assert runner.FAMILIES["curl"]["experiments"] == "pml,multistep"
    assert runner.FAMILIES["constitutive"]["experiments"] == \
        "constitutive,constitutive_multistep"


def test_the_constitutive_gate_carries_every_defect_the_gate_requires(probe, runner):
    """Four required defects, two host mutations, two nulls, one harness leg.

    The four are ``CONSTITUTIVE_SOURCE_MUTATIONS`` — the gate's own statement of
    what this sub-step must be proof against. A leg table that armed three of
    them would still report a pass, which is why the list is checked against the
    legs rather than against a comment.
    """
    armed_source = set()
    armed_host = set()
    for leg in runner.LEGS:
        if leg.get("family") != "constitutive":
            continue
        args = list(leg["args"])
        if "--source-mutation" in args:
            armed_source.add(args[args.index("--source-mutation") + 1])
        if "--host-mutation" in args:
            armed_host.add(args[args.index("--host-mutation") + 1])
    missing = set(probe.CONSTITUTIVE_SOURCE_MUTATIONS) - armed_source
    assert not missing, f"the constitutive gate arms no leg for {sorted(missing)}"
    assert set(probe.CONSTITUTIVE_NULL_MUTATIONS) <= armed_source
    # The two host mutations: the sub-lattice swap (a half-cell error the KERNEL
    # cannot make and its caller can) and the kps/kms swap (bit-identical outside
    # the layer, so it also measures whether the sweep reaches the absorber).
    assert {"swap_constitutive_sublattice", "swap_kps_kms"} <= armed_host
    for name in armed_host:
        assert name in probe.HOST_MUTATIONS, name


def test_the_null_mutations_are_stamped_must_be_uncaught_in_the_artifact(probe):
    """The artifact has to say which kind of leg it is, or a reader cannot tell."""
    for name in probe.PML_NULL_MUTATIONS:
        assert name in (probe.FUSED_PAIR_NULL_MUTATIONS
                        + probe.DISPERSIVE_NULL_MUTATIONS
                        + probe.PML_NULL_MUTATIONS)


def test_every_leg_names_a_mutation_the_probe_implements(probe, runner):
    """A typo'd leg runs a gate with no mutation and reports its pass."""
    known_source = set(probe.SOURCE_MUTATIONS)
    unknown = []
    for leg in runner.LEGS:
        args = list(leg["args"])
        if "--source-mutation" in args:
            name = args[args.index("--source-mutation") + 1]
            if name not in known_source:
                unknown.append(f"{leg['leg']}: --source-mutation {name}")
        if "--host-mutation" in args:
            name = args[args.index("--host-mutation") + 1]
            if name not in probe.HOST_MUTATIONS:
                unknown.append(f"{leg['leg']}: --host-mutation {name}")
    assert not unknown, "\n".join(unknown)


# --------------------------------------------------------------------------
# §1.5 — armed-mutation accounting.
# --------------------------------------------------------------------------

class _StubCache:
    def __init__(self, digests):
        self._digests = list(digests)

    def compile_log(self):
        return tuple({"source_sha256": digest} for digest in self._digests)


def _stub_adapter(cache):
    module = types.SimpleNamespace()
    if cache is not None:
        module.compile_cache = cache
    return types.SimpleNamespace(module=module)


def test_the_accounting_says_when_the_mutated_bytes_never_reached_the_compiler(probe):
    """THE defect: a leg reporting a pass for a mutation it never applied.

    The sibling track hit it three times. Text matching is not enough — between
    the rewrite and NVRTC sit an adapter that has to re-install the attribute, a
    memo that has to miss, and a disk cache that has to not answer.
    """
    armed = {"original_source_sha256": {"B": "aaa", "D": "bbb"},
             "mutated_source_sha256": {"B": "ccc", "D": "ddd"}}
    never = probe.armed_mutation_accounting(_stub_adapter(_StubCache(["aaa", "bbb"])),
                                            armed)
    assert never["applies"] and never["measurable"]
    assert never["compiles_from_mutated_source"] == 0
    assert never["compiles_from_unmutated_source"] == 2
    assert never["mutated_source_reached_the_compiler"] is False

    landed = probe.armed_mutation_accounting(
        _stub_adapter(_StubCache(["ccc", "ddd"])), armed)
    assert landed["compiles_from_mutated_source"] == 2
    assert landed["mutated_source_reached_the_compiler"] is True


def test_a_track_without_a_compile_log_reports_unmeasurable_not_zero(probe):
    """A zero here would read as "the mutation never landed", which is worse than
    saying the counter is missing."""
    armed = {"original_source_sha256": {"B": "aaa"},
             "mutated_source_sha256": {"B": "ccc"}}
    record = probe.armed_mutation_accounting(_stub_adapter(None), armed)
    assert record["applies"] is True
    assert record["measurable"] is False
    assert "mutated_source_reached_the_compiler" not in record
    assert record["why"]


def test_a_leg_with_no_source_mutation_says_the_accounting_does_not_apply(probe):
    record = probe.armed_mutation_accounting(_stub_adapter(None), None)
    assert record["applies"] is False


def test_the_runner_fails_a_leg_whose_mutation_never_compiled(runner):
    """The accounting has to have teeth, not just a field in the JSON."""
    source = (HERE / "run_pml_gate_mutations.py").read_text(encoding="utf-8")
    assert "armed_as_required" in source
    assert "mutated_source_reached_the_compiler" in source
    assert "and armed_ok" in source, (
        "the armed accounting is recorded but not folded into the leg's verdict")


# --------------------------------------------------------------------------
# The defect found while adding the clauses above.
# --------------------------------------------------------------------------

def test_the_runner_parses_a_verdict_key_the_probe_actually_emits(probe, runner):
    """It did not, and every leg silently read 0/0.

    ``run_leg`` asked for ``verdict['single_launch']``; ``pml_verdict`` emits
    ``curl_single_launch``. ``_split_fraction(None)`` returns (0, 0), so ``ran``
    was 0, ``ok`` was False and every leg's ``as_required`` was False regardless
    of what the gate measured. This pins the two ends together.
    """
    verdict = probe.pml_verdict({
        "pml_bit_identity": {"gate": {"primary_identical": 120, "primary_ran": 120,
                                      "primary_pass": True,
                                      "subnormal_band_identical": 100,
                                      "subnormal_band_ran": 120}},
    })
    assert runner._split_fraction(verdict["curl_single_launch"]) == (120, 120)
    assert verdict["curl_single_launch_subnormal_band"] == "100/120"
    source = (HERE / "run_pml_gate_mutations.py").read_text(encoding="utf-8")
    assert "curl_single_launch" in source
    assert 'verdict.get("single_launch")' not in source


def test_an_unreadable_verdict_fails_the_leg_rather_than_scoring_zero(runner):
    assert runner._split_fraction(None) == (0, 0)
    assert runner._split_fraction("not a fraction") == (0, 0)


# --------------------------------------------------------------------------
# §1.2 — the second half the memo key cannot reach.
# --------------------------------------------------------------------------

def test_every_leg_gets_a_private_policy_token_carrying_cache_directory(runner):
    """CuPy's cache key is computed ABOVE the strip seam, so the directory IS the key.

    ``subnormal_policy.cupy_cache_reasons`` refuses a ``keep`` install whose
    directory does not carry ``ftz_stripped`` and refuses a ``flush`` install
    whose directory does; both refusals are satisfied by construction here, and
    the per-leg leaf also makes every mutation leg compile cold.
    """
    if str(REPO_API) not in sys.path:
        sys.path.insert(0, str(REPO_API))
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    keep = runner.leg_cache_dir("/out", "keep", "m3_regroup_stencil")
    flush = runner.leg_cache_dir("/out", "flush", "m3_regroup_stencil")
    none = runner.leg_cache_dir("/out", None, "m3_regroup_stencil")

    assert subnormal_policy.CUPY_CACHE_POLICY_TOKEN in keep
    assert subnormal_policy.CUPY_CACHE_POLICY_TOKEN not in flush
    assert subnormal_policy.CUPY_CACHE_POLICY_TOKEN not in none
    assert not subnormal_policy.cupy_cache_reasons("keep", keep)
    assert not subnormal_policy.cupy_cache_reasons("flush", flush)
    assert keep != flush != none
    for leg in runner.LEGS:
        assert runner.leg_cache_dir("/out", "keep", leg["leg"]) != keep or \
            leg["leg"] == "m3_regroup_stencil"


def test_the_guard_sweep_drops_the_kernel_memo_at_both_ends(probe):
    """Why the memo key is allowed to be blind to the probe's option override.

    ``CupyShim`` substitutes the compiler and overrides the option tuple from
    OUTSIDE the kernel module, so ``compile_cache``'s key — which records the
    options the module PASSES — cannot see the change. What makes that safe is
    that the adapter drops the memo on entering and on leaving every guard set.
    If that ever stops being true, the second guard set is served the first one's
    binaries and the 0/120 unguarded control silently becomes 120/120.
    """
    cleared = []
    module = types.SimpleNamespace(
        _clear_kernel_cache=lambda: cleared.append(True),
        _step_B_pml_real_kernel_code="B",
        _step_D_pml_real_kernel_code="D",
        cp=None)
    adapter = probe.HandTrackAdapter(module)
    adapter.begin_guard(("--fmad=false",))
    assert len(cleared) == 1
    assert isinstance(module.cp, probe.CupyShim)
    adapter.end_guard()
    assert len(cleared) == 2


def test_a_module_loaded_INSIDE_a_guard_still_gets_that_guard(probe, monkeypatch):
    """The leg that has no source mutation is the one this exists for.

    The constitutive kernels live in a sibling module the adapter loads lazily.
    ``cgate`` arms nothing, so nothing calls ``source_attributes()`` and the
    sibling first loads inside ``launch`` — AFTER ``begin_guard`` installed the
    shim on the modules it knew about. A sibling that missed the shim would
    compile under the file's own ``--fmad=false`` while the artifact reported the
    swept option set, and the ``default_no_options`` control would silently stop
    being a control: 0/N would read as N/N.

    Measured on a stand-in sibling rather than the real one, because the real one
    imports CuPy.
    """
    cleared = []
    sibling = types.SimpleNamespace(
        _clear_kernel_cache=lambda: cleared.append("sibling"),
        _update_H_pml_real_kernel_code="H",
        _update_E_pml_real_kernel_code="E",
        cp=None)
    module = types.SimpleNamespace(
        _clear_kernel_cache=lambda: cleared.append("curl"),
        _step_B_pml_real_kernel_code="B",
        _step_D_pml_real_kernel_code="D",
        cp=None)
    adapter = probe.HandTrackAdapter(module)

    def fake_load():
        """The by-path load's body, minus the filesystem — same guard block."""
        if adapter._constitutive is None:
            if adapter._guard_active:
                sibling.cp = probe.CupyShim(adapter._guard or ())
                for attribute, source in adapter._mutated_source.items():
                    if hasattr(sibling, attribute):
                        setattr(sibling, attribute, source)
                sibling._clear_kernel_cache()
            adapter._constitutive = sibling
        return sibling
    monkeypatch.setattr(adapter, "constitutive_module", fake_load)

    adapter.begin_guard(())            # the UNGUARDED control set
    assert sibling.cp is None, "the sibling was loaded before the guard began"
    adapter.constitutive_module()      # what launch() does on the cgate leg
    assert isinstance(sibling.cp, probe.CupyShim), (
        "a module loaded inside a guard did not receive the swept option tuple; "
        "its kernels would compile under the file's own options")
    assert "sibling" in cleared, "the late-loaded module kept a stale memo"
    adapter.end_guard()
    assert adapter._guard_active is False
    assert sibling.cp is probe.cp, "end_guard left the shim installed on the sibling"


# --------------------------------------------------------------------------
# §1.4 — the lifted-case leg's Courant axis.
# --------------------------------------------------------------------------

def test_the_lifted_leg_sweeps_a_dtdx_that_is_not_exactly_representable(lifted):
    """0.5 hides the guard; the record's six cases were all 0.5.

    ``dtdx`` IS the Courant number here (``dt = courant/resolution``,
    ``dx = 1/resolution``), so this is the lever, and the assertion is on the
    float32 property rather than on the literal.
    """
    assert float(np.float32(lifted.EXACT_COURANT)) == lifted.EXACT_COURANT
    assert float(np.float32(lifted.INEXACT_COURANT)) != lifted.INEXACT_COURANT
    assert lifted.INEXACT_COURANT == 0.35, (
        "the synthetic gate sweeps 0.35 as its inexact value; a different one "
        "here makes the corpus leg and the synthetic leg incommensurable")
    assert set(lifted.COURANTS) == {lifted.EXACT_COURANT, lifted.INEXACT_COURANT}
    assert len(lifted.PLAN) == 2 * len(lifted.SHAPES)
    assert {courant for _, _, courant in lifted.PLAN} == set(lifted.COURANTS)


def test_the_provenance_file_names_the_policy_the_bytes_were_cut_under(probe, tmp_path):
    """§1.1's actual requirement: the ARTIFACT has to say which policy it is.

    ``grep -ric`` for ``subnormal|ftz|denormal|policy`` over the 2026-08-09/10
    artifact's five files returned 0 on all five, so the recorded 120/120 cannot
    even be FILTERED by policy — and CuPy appends ``-ftz=true`` to every NVRTC
    compile while ``subnormal_policy`` strips it, so the same source is two
    binaries. ``CUPY_CACHE_DIR`` is on the same line for the same reason: CuPy's
    cache key is computed above the strip seam, so a run sharing a directory with
    the other policy can be served the other policy's bytes.
    """
    arguments = types.SimpleNamespace(
        track="hand", module=str(KERNEL_MODULE), source_mutation=None,
        host_mutation=None, allclose=False, no_fu_compare=False)
    results = {
        "started_utc": "2026-08-15T00:00:00Z",
        "environment": {"device_name": "none"},
        "pml_value_classes": list(probe.PML_VALUE_CLASSES),
        "pml_multi_step_budget": probe.PML_MULTI_STEP_COUNT,
        "subnormal_policy": {"policy": "ieee_keep_ftz_stripped", "resolved": "keep",
                             "CUPY_CACHE_DIR": "/tmp/ftz_stripped/gate"},
    }
    written = pathlib.Path(probe.write_provenance(str(tmp_path), arguments, results))
    text = written.read_text(encoding="utf-8")
    assert "subnormal_policy      ieee_keep_ftz_stripped" in text, text
    assert "subnormal_resolved    keep" in text, text
    assert "/tmp/ftz_stripped/gate" in text, text
    assert "subnormal_band" in text, text
    assert "pml_multi_step_budget 60" in text, text
    assert "module_sha256" in text, "the digest test_certification_record reads"


def test_the_certification_record_describes_this_harness_and_not_another(probe, runner,
                                                                        lifted):
    """The record's ``harness_2026-08-15`` block, welded to the harness it describes.

    ``certification.json`` now carries two step budgets on purpose — 60 for what
    the gate does and 8 for what produced the recorded verdict — and a reader has
    to be able to trust that the first number is the harness's real one. It is a
    transcription, so it rots the moment either side moves; this is the test that
    makes the rot loud instead of silent.
    """
    import json  # noqa: PLC0415

    record = json.loads(
        (REPO_API / "meep_gpu" / "cuda_kernels" / "certification.json")
        .read_text(encoding="utf-8"))
    harness = record["harness_2026-08-15"]
    assert harness["pml_value_classes"] == list(probe.PML_VALUE_CLASSES)
    assert harness["pml_multi_step_budget"] == probe.PML_MULTI_STEP_COUNT
    assert harness["pml_null_mutations"] == list(probe.PML_NULL_MUTATIONS)
    assert harness["lifted_case_courants"] == list(lifted.COURANTS)
    for path_key, module in (("probe", probe), ("runner", runner),
                             ("lifted_validator", lifted)):
        named = REPO_API / harness[path_key]
        assert named.exists(), harness[path_key]
        assert named.resolve() == pathlib.Path(module.__file__).resolve()


def test_the_lifted_leg_will_not_pass_on_exact_dtdx_cases_alone(lifted):
    """The summary has to count the half that tests the arithmetic guard.

    A run whose inexact-dtdx count is zero has not exercised ``--fmad=false`` at
    corpus scale whatever its total says, so ``pass`` requires that half to be
    non-empty AND whole.
    """
    source = (HERE / "validate_pml_kernel_on_lifted_cases.py").read_text(
        encoding="utf-8")
    assert "covered_at_inexact_dtdx" in source
    assert "and bool(inexact) and len(inexact_identical) == len(inexact)" in source


# --------------------------------------------------------------------------
# §1.4 again — the record's operand-class census against the live generator.
# --------------------------------------------------------------------------

def test_the_recorded_operand_census_is_what_the_shipped_generator_draws(
        probe, keeps_subnormals):
    """The record's numbers, re-derived rather than believed.

    An earlier revision of this row read "508 subnormals of 1080 values, 5
    negative zeros, 11 zeros" and labelled it "18 arrays x 13x11x9", which is
    23166 values — the counts had been taken at a different shape and stamped
    with the gate's. Nothing caught it, because the only assertions on this
    generator were "more than a tenth of the values" and "more than zero", which
    hold at every shape and every seed. So the record now carries a seed and a
    shape, and every count in it is asserted EXACTLY here: a record whose whole
    purpose is that its numbers can be trusted may not state one the code does
    not produce.
    """
    import json  # noqa: PLC0415

    entry = json.loads(
        (REPO_API / "meep_gpu" / "cuda_kernels" / "certification.json")
        .read_text(encoding="utf-8"))
    # The gap moved to resolved_gaps when the 2026-08-15 re-cut closed it, and the
    # census moved with it — it is the evidence the band class is non-vacuous, so
    # it stays asserted exactly whichever list it lives in.
    gaps = entry["open_gaps"] + entry.get("resolved_gaps", [])
    census = next(gap for gap in gaps
                  if gap["id"] == "operand-classes-cannot-see-the-policy")["operand_class_census"]
    shape = tuple(census["shape"])
    assert shape in probe.PML_SHAPES, (
        "the census is drawn at a shape the gate does not sweep")
    assert census["arrays"] == len(probe.PML_FIELD_NAMES)
    assert census["values"] == census["arrays"] * shape[0] * shape[1] * shape[2], (
        "arrays x shape does not equal the stated value count — the counts and "
        "the label are describing different draws, which is the defect this "
        "clause exists for")

    for value_class in ("uniform", "subnormal_band"):
        arrays = probe.pml_field_hosts(
            shape, np.random.default_rng(census["seed"]), value_class)
        got = _census(arrays)
        assert got["values"] == census["values"], (value_class, got)
        for field in ("subnormals", "negative_zeros", "zeros"):
            assert got[field] == census[value_class][field], (
                f"{value_class}.{field}: record says {census[value_class][field]}, "
                f"the shipped generator draws {got[field]}")

    flat = np.concatenate([a.ravel() for a in probe.pml_field_hosts(
        shape, np.random.default_rng(census["seed"]), "subnormal_band").values()])
    nonzero = np.abs(flat[flat != 0])
    assert float(nonzero.min()) == \
        census["subnormal_band"]["min_nonzero_magnitude"]


# --------------------------------------------------------------------------
# §1.1 — the runner's policy plumbing, keyed on the RESOLUTION.
# --------------------------------------------------------------------------

def test_match_meep_is_resolved_before_the_cache_directory_is_chosen(runner):
    """The shipped default policy has to be RUNNABLE, not merely accepted.

    ``--subnormal-policy match_meep`` is in ``choices`` and ``match_meep`` is the
    package default, so it is the likely thing an operator types. It is not a
    policy an executor can be driven to: it is a measurement (flush on x86, keep
    on arm64). Keyed on the literal request, every leg died at the install — the
    keep half because ``cupy_cache_reasons`` refuses a directory without the
    ``ftz_stripped`` token, the flush half because ``CUPY_ACCELERATORS`` was
    never emptied and ``install_cupy_policy`` refuses ``flush`` by name without
    it. Both halves are checked here against the REAL refusal functions, so this
    passes on whichever host it runs on and fails on the other's bug too.
    """
    if str(REPO_API) not in sys.path:
        sys.path.insert(0, str(REPO_API))
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    resolved = runner.resolved_policy("match_meep")
    assert resolved in ("flush", "keep"), resolved
    assert resolved == subnormal_policy.resolve_policy("match_meep")[0]
    assert runner.resolved_policy(None) is None
    assert runner.resolved_policy("keep") == "keep"
    assert runner.resolved_policy("flush") == "flush"

    directory = runner.leg_cache_dir("/out", resolved, "gate")
    assert not subnormal_policy.cupy_cache_reasons(resolved, directory), (
        f"match_meep resolved to {resolved!r} and the leg cache directory "
        f"{directory!r} is refused by the install this runner then performs")

    # And the accelerator lever, which is the other half on an x86 host.
    source = (HERE / "run_pml_gate_mutations.py").read_text(encoding="utf-8")
    assert 'if resolved == "flush":' in source, (
        "CUPY_ACCELERATORS is still keyed on the requested policy, so a "
        "match_meep run on x86 leaves the CUB reduction path outside the policy")
    assert "cache_dir = leg_cache_dir(out_dir, resolved, leg[\"leg\"])" in source


def test_every_argparse_policy_choice_reaches_an_installable_state(runner):
    """No argument the CLI accepts may be unrunnable on the host that accepts it.

    Driven off ``--subnormal-policy``'s OWN choices rather than a list retyped
    here, so a choice added later is covered the day it is added. ``match_meep``
    was accepted and could not execute a single leg; that it now can is the
    clause, and the mechanism is that the resolution — not the request — picks
    the cache directory.
    """
    if str(REPO_API) not in sys.path:
        sys.path.insert(0, str(REPO_API))
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    for choices in re.findall(
            r'--subnormal-policy"[^)]*?choices=\(([^)]*)\)',
            (HERE / "run_pml_gate_mutations.py").read_text(encoding="utf-8"),
            re.S):
        accepted = tuple(value.strip().strip('"\'')
                         for value in choices.split(",") if value.strip())
        break
    else:  # pragma: no cover - the flag was renamed
        pytest.fail("--subnormal-policy's choices could not be read from the runner")
    assert set(accepted) == {"flush", "keep", "match_meep"}, accepted

    for choice in accepted + (None,):
        resolved = runner.resolved_policy(choice)
        directory = runner.leg_cache_dir("/out", resolved, "m3_regroup_stencil")
        if resolved is None:
            continue
        assert not subnormal_policy.cupy_cache_reasons(resolved, directory), (
            choice, resolved, directory)


# --------------------------------------------------------------------------
# The policy-reach probe, which is certification.json's §1.1 evidence.
# --------------------------------------------------------------------------

def test_the_policy_reach_probe_can_still_read_the_kernel_memo():
    """It produced the measurement the record quotes; it has to keep running.

    ``certification.json`` cites this probe verbatim — "install-then-compile kept
    a planted 0x00004000 in fu_Bx; compile-then-install flushed it" — as the
    basis for saying the 120/120 was taken under flush. The probe read
    ``step_curl_kernels._compiled_kernels``, which the 2026-08-15 memo move
    deleted and ``test_compile_cache.py`` pins as staying deleted, so on a CUDA
    host it would have raised ``AttributeError`` inside ``run_kernel_once`` on
    its FIRST leg and that measurement could not be re-taken.

    Driven against the real ``compile_cache`` here, and against a module with
    neither accessor, so the fallback reports rather than raises.
    """
    if str(REPO_API) not in sys.path:
        sys.path.insert(0, str(REPO_API))
    from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

    reach = _load(HERE / "probe_cuda_policy_reach.py", "_probe_cuda_policy_reach")

    shipped = types.SimpleNamespace(compile_cache=compile_cache)
    compile_cache.clear_kernel_cache()
    assert reach.memo_size(shipped) == 0
    key = compile_cache.kernel_cache_key("step_B_pml_real", False, (), "src")
    compile_cache.get_or_compile(key, lambda: object())
    try:
        assert reach.memo_size(shipped) == 1
    finally:
        compile_cache.clear_kernel_cache()

    legacy = types.SimpleNamespace(_compiled_kernels={"a": 1, "b": 2})
    assert reach.memo_size(legacy) == 2
    assert reach.memo_size(types.SimpleNamespace()) == -1

    source = (HERE / "probe_cuda_policy_reach.py").read_text(encoding="utf-8")
    assert "len(sck._compiled_kernels)" not in source, (
        "the probe reaches for the deleted attribute again")


# --------------------------------------------------------------------------
# §1.2 — the NVRTC binary observer: the only thing that can tell two policies'
# BYTES apart, and the one wrapper that could silently disable the strip.
# --------------------------------------------------------------------------

def test_the_observer_is_installed_before_the_policy_so_it_sees_stripped_options(probe):
    """Order is the whole design, and it is readable in ``main`` rather than hoped.

    Installed AFTER the policy the observer sits OUTSIDE the strip and records the
    pre-strip option tuple — which still carries ``-ftz=true`` under both
    policies, making the two legs look alike in exactly the field the §1.2
    question is asked in. The install order is therefore a correctness property,
    not a style one, so it is asserted on the source.
    """
    source = (HERE / "probe_fused_kernel_bit_identity.py").read_text(encoding="utf-8")
    observer = source.index("binary_observer = install_nvrtc_binary_observer()")
    policy = source.index("policy_install = (install_subnormal_policy_for_run(")
    assert observer < policy, (
        "the NVRTC observer is installed after the policy, so it wraps the strip "
        "instead of being wrapped BY it and records pre-strip options")


def test_the_observer_does_not_inherit_the_policy_mark_and_suppress_the_strip(probe):
    """The failure this must not cause is the one it exists to detect.

    ``subnormal_policy`` marks the wrappers it installs and refuses to install a
    second time over its own mark. ``functools.wraps`` copies ``__dict__``, so an
    observer built with it inherits whatever mark the function it wrapped carried
    — and a later ``install_subnormal_policy('keep')`` would see the seam as
    already done and leave the run compiling with ``-ftz=true`` while the artifact
    said ``keep``. That is a run whose bytes belong to neither policy, reported as
    a clean one.

    Driven, not read: a fake compiler module is wrapped and the mark checked.
    """
    if str(REPO_API) not in sys.path:
        sys.path.insert(0, str(REPO_API))
    from meep_gpu import subnormal_policy  # noqa: PLC0415

    def original(source, options=(), *args, **kwargs):
        return (b"CUBIN-BYTES", "irrelevant")

    setattr(original, subnormal_policy._POLICY_MARK, subnormal_policy.KEEP)
    fake = types.SimpleNamespace(compile_using_nvrtc=original)
    cupy_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(compiler=fake))

    saved_cupy = sys.modules.get("cupy")
    saved_cuda = sys.modules.get("cupy.cuda")
    saved_compiler = sys.modules.get("cupy.cuda.compiler")
    sys.modules["cupy"] = cupy_stub
    sys.modules["cupy.cuda"] = cupy_stub.cuda
    sys.modules["cupy.cuda.compiler"] = fake
    try:
        probe._NVRTC_OBSERVATIONS.clear()
        report = probe.install_nvrtc_binary_observer()
        assert report["installed"] is True, report
        assert not hasattr(fake.compile_using_nvrtc, subnormal_policy._POLICY_MARK), (
            "the observer inherited the policy mark, so a later 'keep' install "
            "would treat the seam as already stripped and silently do nothing")
    finally:
        for key, value in (("cupy", saved_cupy), ("cupy.cuda", saved_cuda),
                           ("cupy.cuda.compiler", saved_compiler)):
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def test_the_observer_fingerprints_the_bytes_and_the_options_nvrtc_received(probe):
    """A source hash cannot answer §1.2; both policies compile the same source."""
    calls = []

    def original(source, options=(), *args, **kwargs):
        calls.append(tuple(options))
        # Distinct bytes per option tuple, as a real compiler produces.
        return (b"CUBIN:" + repr(tuple(options)).encode(), "log")

    fake = types.SimpleNamespace(compile_using_nvrtc=original)
    cupy_stub = types.SimpleNamespace(cuda=types.SimpleNamespace(compiler=fake))
    saved = {k: sys.modules.get(k) for k in
             ("cupy", "cupy.cuda", "cupy.cuda.compiler")}
    sys.modules["cupy"] = cupy_stub
    sys.modules["cupy.cuda"] = cupy_stub.cuda
    sys.modules["cupy.cuda.compiler"] = fake
    try:
        probe._NVRTC_OBSERVATIONS.clear()
        assert probe.install_nvrtc_binary_observer()["installed"] is True
        fake.compile_using_nvrtc("SOURCE", ("--fmad=false", "-ftz=true"))
        fake.compile_using_nvrtc("SOURCE", ("--fmad=false",))
        report = probe.nvrtc_binary_report()
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
        probe._NVRTC_OBSERVATIONS.clear()

    assert report["nvrtc_calls_observed"] == 2
    assert report["distinct_sources"] == 1, (
        "the two compiles are of ONE source — which is exactly why a source hash "
        "cannot distinguish the policies and this report exists")
    assert report["distinct_binaries"] == 2, report
    assert report["any_ftz_true_reached_nvrtc"] is True
    assert report["all_ftz_true_reached_nvrtc"] is False
    only_source = next(iter(report["binary_sha256_by_source"]))
    assert len(report["binary_sha256_by_source"][only_source]) == 2, (
        "one source mapping to one binary would mean the two legs cannot be "
        "told apart, which is the §1.2 question")
    assert [e["ftz_true_present"] for e in report["observations"]] == [True, False]


# --------------------------------------------------------------------------
# §1.4 — non-vacuity is carried by the RUN's own record, not by trust.
# --------------------------------------------------------------------------

def test_the_case_census_counts_the_values_that_actually_go_to_the_device(probe):
    """The census is of the drawn arrays, at the draw the case uses."""
    hosts = probe.pml_field_hosts((13, 11, 9),
                                  np.random.default_rng(probe.SEED + 7),
                                  "subnormal_band")
    assert probe.operand_census(hosts) == _census(hosts)

    uniform = probe.pml_field_hosts((13, 11, 9),
                                    np.random.default_rng(probe.SEED + 7),
                                    "uniform")
    census = probe.operand_census(uniform)
    assert census["subnormals"] == 0 and census["negative_zeros"] == 0, (
        "the class the 2026-08-09/10 gate ran on is supposed to be provably "
        "unable to reach the band")


def test_a_band_leg_whose_operands_hold_no_subnormal_is_marked_vacuous(probe):
    """The clause that stops 'subnormal_band 24/24' from being read as evidence.

    A band leg with no subnormals and no signed zeros in it exercised the policy
    exactly as much as the uniform class did, and would report a clean pass while
    doing it. Both the live shape and the vacuous one are driven here.
    """
    def case(value_class, census):
        return {"guard_label": "fmad_false", "guard_is_primary": True,
                "value_class": value_class,
                "operand_census": census, "reference_moved_targets": 1,
                "vs_array_order": {"bit_identical": True},
                "vs_kernel_order": {"bit_identical": True}}

    live = {"values": 100, "subnormals": 40, "negative_zeros": 3, "zeros": 5}
    dead = {"values": 100, "subnormals": 0, "negative_zeros": 0, "zeros": 0}
    normal = case("uniform", {"values": 100, "subnormals": 0,
                              "negative_zeros": 0, "zeros": 0})

    good = probe.summarize_pml([normal, case("subnormal_band", live)])
    assert good["subnormal_band_is_non_vacuous"] is True
    assert good["subnormal_band_operands"]["subnormals"] == 40

    bad = probe.summarize_pml([normal, case("subnormal_band", dead)])
    assert bad["subnormal_band_identical"] == 1, (
        "the band leg still reports identical — which is the point: the pass is "
        "real and it is worthless, and only the vacuity flag says so")
    assert bad["subnormal_band_is_non_vacuous"] is False

    absent = probe.summarize_pml([normal])
    assert absent["subnormal_band_is_non_vacuous"] is False, (
        "a run that swept no band case at all must not report non-vacuity")


def test_the_lifted_leg_can_name_the_policy_its_bytes_were_cut_under(lifted):
    """§1.1 applies to EVERY artifact this track cuts, not only to ``gate.json``.

    The lifted leg is the one that runs at corpus scale, and until this round it
    had no way to say which policy compiled the kernel it validated — the exact
    defect §1.1 records against the 2026-08-09/10 record. Leaving it that way
    while fixing the gate would move the unanswerable question rather than answer
    it, so the flag, the install and the stamp are asserted here.
    """
    source = (HERE / "validate_pml_kernel_on_lifted_cases.py").read_text(
        encoding="utf-8")
    assert '"--subnormal-policy"' in source
    install = source.index("probe.install_subnormal_policy_for_run(")
    observer = source.index("probe.install_nvrtc_binary_observer()")
    first_case = source.index("for index, (name, kwargs, courant) in enumerate(PLAN")
    assert observer < install < first_case, (
        "the policy is installed after the first case has already compiled a "
        "kernel, which is the compile-then-install order measured to leave the "
        "binary under the OTHER policy")
    assert '"subnormal_policy": probe.subnormal_policy_stamp(' in source, (
        "the lifted artifact does not stamp the policy, so its verdict cannot "
        "be filtered by one — disposition §1.1")


def test_a_flush_leg_asks_the_probe_to_import_meep_first(runner, probe):
    """MEASURED on device 2026-08-15: without this, the whole flush leg RAISES.

    ``install_subnormal_policy('flush')`` refuses in a process that has not
    imported MEEP — ``mp.set_zero_subnormals`` is the only route to this
    process's FTZ/DAZ bits and ``meep_gpu`` may not import MEEP itself — so the
    first flush re-cut attempt returned ``0/12 legs behaved as required`` in 7 s
    with no kernel ever compiled. The refusal is correct; the runner has to
    answer it.

    Keyed on the RESOLUTION, not the request, for the same reason
    ``CUPY_ACCELERATORS`` is: on x86 ``match_meep`` RESOLVES to flush, and a
    keep leg must NOT import MEEP (it initializes MPI and moves the host FPU for
    reasons unrelated to the kernel).
    """
    source = (HERE / "run_pml_gate_mutations.py").read_text(encoding="utf-8")
    assert '"--import-meep-for-host-policy"] if resolved == "flush"' in source, (
        "the runner does not ask for the MEEP import on a flush leg, so the "
        "policy install will refuse before the first compile")
    assert 'if policy == "flush"' not in source, (
        "keyed on the REQUEST rather than the resolution: match_meep resolves "
        "to flush on x86 and would be missed")

    probe_source = (HERE / "probe_fused_kernel_bit_identity.py").read_text(
        encoding="utf-8")
    imported = probe_source.index("meep_import = (import_meep_for_host_policy()")
    installed = probe_source.index(
        "policy_install = (install_subnormal_policy_for_run(")
    assert imported < installed, (
        "MEEP is imported after the policy install it exists to make attainable")


# ---------------------------------------------------------------------------
# Armed-mutation accounting must read EVERY loaded kernel module's compile log
# ---------------------------------------------------------------------------

class _FakeCache:
    """A stand-in for ``cuda_kernels/compile_cache.py``'s module-level log."""

    def __init__(self, digests):
        self._entries = [{"source_sha256": digest} for digest in digests]

    def compile_log(self):
        return list(self._entries)


class _FakeAdapter:
    """A ``HandTrackAdapter`` shaped just enough for the accounting to read it.

    The two kernel modules are loaded BY PATH by the probe, so each takes the
    ImportError fallback in its own header and ends up with its OWN
    ``compile_cache`` module object — which is why these two fakes deliberately
    do NOT share one.
    """

    def __init__(self, modules):
        self._loaded = modules
        self.module = modules[0]

    def _modules(self):
        return tuple(self._loaded)


def _module_with_log(name, digests):
    module = types.ModuleType(name)
    module.compile_cache = _FakeCache(digests)
    return module


def test_armed_accounting_reads_the_constitutive_modules_log_too(probe):
    """MEASURED FAILURE, 2026-08-15: every constitutive leg reported 0 of 0.

    ``armed_mutation_accounting`` read ``adapter.module.compile_cache`` — the
    CURL module's log — and a constitutive leg compiles nothing there. The
    driver then correctly REFUSED thirteen legs whose mutations had in fact been
    applied, because "0 compiles from mutated source out of 0 constructions" is
    indistinguishable from a mutation that never landed.

    Both modules' logs must be unioned. This pins it with the curl log EMPTY,
    which is exactly the shape a constitutive leg has.
    """
    curl = _module_with_log("_fake_curl", [])
    constitutive = _module_with_log("_fake_constitutive", ["MUT", "MUT", "ORIG"])
    adapter = _FakeAdapter([curl, constitutive])
    armed = {"name": "regroup_constitutive",
             "original_source_sha256": {"_a_kernel_code": "ORIG"},
             "mutated_source_sha256": {"_a_kernel_code": "MUT"}}

    accounting = probe.armed_mutation_accounting(adapter, armed)

    assert accounting["measurable"] is True
    assert accounting["compiles"] == 3, accounting
    assert accounting["compiles_from_mutated_source"] == 2, accounting
    assert accounting["mutated_source_reached_the_compiler"] is True
    assert accounting["compile_logs_read"] == ["_fake_curl", "_fake_constitutive"]
    assert accounting["compiles_per_log"] == {"_fake_curl": 0,
                                              "_fake_constitutive": 3}


def test_armed_accounting_counts_a_shared_memo_once(probe):
    """Two modules that DO share one memo must not double-count it.

    The modules' own docstrings describe the memo as shared, and it is when they
    are imported as a package. De-duplication is by identity, so both worlds give
    the same number instead of the union being right in one and doubled in the
    other.
    """
    shared = _FakeCache(["MUT", "ORIG"])
    first, second = types.ModuleType("_fake_a"), types.ModuleType("_fake_b")
    first.compile_cache = second.compile_cache = shared
    adapter = _FakeAdapter([first, second])
    armed = {"name": "regroup_constitutive",
             "original_source_sha256": {"_k": "ORIG"},
             "mutated_source_sha256": {"_k": "MUT"}}

    accounting = probe.armed_mutation_accounting(adapter, armed)

    assert accounting["compiles"] == 2, accounting
    assert accounting["compiles_from_mutated_source"] == 1, accounting
    assert accounting["compile_logs_read"] == ["_fake_a"]


# --------------------------------------------------------------------------
# The all-periodic reference — the leg nothing executed, and nothing welded.
# --------------------------------------------------------------------------
#
# The no-absorber leg's reference used to live in a SECOND function also called
# ``reference_curl``. The later definition rebound the name, so ``reference_step``
# was left calling a signature that no longer existed and the leg raised
# ``TypeError: reference_curl() missing 2 required positional arguments`` on its
# first case — for five days, unnoticed, because every test in this file above
# pins the PML leg, no validator welds the all-periodic step to stepping.py, and
# no driver passes ``--experiments legacy``. The clauses below EXECUTE the step
# and pin the expression it must produce, on NumPy, with no device: the reference
# is the one thing in a bit-identity gate that cannot be checked by the gate.

#: Every axis > 1, so no axis is invariant and every roll actually moves data.
PERIODIC_SHAPE = (7, 5, 3)
PERIODIC_SEED = 20260815
#: 0.35 is NOT exactly representable in binary32. At 0.5 the scale is exact, the
#: two groupings collapse onto the same bits, and a re-association regression is
#: invisible — the same reason the gate's own dtdx sweep carries 0.35.
PERIODIC_DTDX = np.float32(0.35)

#: (terms attribute, backward) per sub-step. ``backward`` is MEEP's negated
#: stride: step_D shifts down (``roll(f, +1)``), step_B shifts up (``roll(f, -1)``).
PERIODIC_SUB_STEPS = {"step_B": ("B_TERMS", False), "step_D": ("D_TERMS", True)}


def _periodic_fields(probe, sub_step):
    """Seeded float32 operands for one sub-step: three sources, three targets."""
    rng = np.random.default_rng(PERIODIC_SEED)
    terms = getattr(probe, PERIODIC_SUB_STEPS[sub_step][0])
    names = [term[0] for term in terms] + sorted(
        {term[1] for term in terms} | {term[3] for term in terms})
    return {name: rng.standard_normal(PERIODIC_SHAPE, dtype=np.float32)
            for name in names}


def _call_reference_step(probe, arrays, sub_step, grouping):
    """``reference_step`` on NumPy, with any binding failure named as one.

    NumPy is the stand-in for ``cp``: every operation the reference performs is
    ``roll``/``-``/``*``, which both libraries spell identically, and the probe
    already runs its PML reference this way in
    ``validate_pml_reference_vs_stepping.py`` on a host with no GPU.
    """
    terms_attribute, backward = PERIODIC_SUB_STEPS[sub_step]
    terms = getattr(probe, terms_attribute)
    sources = {name: arrays[name] for term in terms for name in (term[1], term[3])}
    targets = {term[0]: arrays[term[0]] for term in terms}
    try:
        return probe.reference_step(np, sources, targets, terms, PERIODIC_DTDX,
                                    backward, grouping)
    except TypeError as exc:
        pytest.fail(
            f"the all-periodic reference cannot be called at all ({sub_step}, "
            f"{grouping}): {type(exc).__name__}: {exc}. This is the leg's ONLY "
            "reference — a gate that cannot compute it certifies nothing, and "
            "the failure is argument binding, so no device would change it.")


@pytest.mark.parametrize("sub_step", sorted(PERIODIC_SUB_STEPS))
def test_the_all_periodic_reference_is_steppings_curl_contract(probe, sub_step):
    """``reference_step`` == stepping._curl_from_operands (:1601) then :510, to the bit.

    The expectation is built here from ``np.roll`` and four arithmetic
    operations, NOT by calling anything the probe exposes, so agreement is
    evidence about the transcription rather than about itself.
    """
    arrays = _periodic_fields(probe, sub_step)
    pristine = {name: array.copy() for name, array in arrays.items()}
    terms_attribute, backward = PERIODIC_SUB_STEPS[sub_step]
    terms = getattr(probe, terms_attribute)

    shift = 1 if backward else -1
    expected = {}
    for target, first, first_axis, second, second_axis in terms:
        f, s = arrays[first], arrays[second]
        shifted_first = np.roll(f, shift, axis=first_axis)
        shifted_second = np.roll(s, shift, axis=second_axis)
        curl = PERIODIC_DTDX * ((shifted_first - f) + (s - shifted_second))
        expected[target] = arrays[target] - curl

    produced = _call_reference_step(probe, arrays, sub_step, "array_order")

    assert sorted(produced) == sorted(expected), (
        f"{sub_step}: the reference updated {sorted(produced)}, not the three "
        f"targets {sorted(expected)}")
    for target in sorted(expected):
        got = np.ascontiguousarray(produced[target]).view(np.uint32)
        want = np.ascontiguousarray(expected[target]).view(np.uint32)
        differing = int(np.count_nonzero(got != want))
        assert differing == 0, (
            f"{sub_step} {target}: {differing}/{want.size} floats differ from "
            "dtdx * ((shifted_first - first) + (second - shifted_second)) "
            "computed independently in this test — the reference is no longer "
            "stepping.py's grouping, so every bit-identity number the leg "
            "reports is measured against the wrong expression")
        # NOT vacuous: a curl of exact zeros would make the comparison above
        # true no matter what the reference did with it.
        assert not np.array_equal(
            np.ascontiguousarray(pristine[target]).view(np.uint32), want), (
            f"{sub_step} {target}: the expected update equals the input target "
            "bit for bit, so this case compares a no-op against a no-op")

    for name, array in arrays.items():
        assert np.array_equal(np.ascontiguousarray(array).view(np.uint32),
                              np.ascontiguousarray(pristine[name]).view(np.uint32)), (
            f"{sub_step}: the reference mutated its input {name}; the gate hands "
            "the SAME arrays to both groupings and then to the kernel")


@pytest.mark.parametrize("sub_step", sorted(PERIODIC_SUB_STEPS))
def test_the_two_groupings_of_the_all_periodic_reference_are_different_bits(
        probe, sub_step):
    """The discriminating clause: if they agree, the pair measures nothing.

    ``array_order`` and ``kernel_order`` differ only by where the parentheses
    fall. Float addition is not associative, so on inexact operands they must
    differ in the last bits — and if they did NOT, the gate's whole "which
    grouping does the kernel reproduce" question would be unanswerable and the
    test above would pass against either expression.
    """
    arrays = _periodic_fields(probe, sub_step)
    array_order = _call_reference_step(probe, arrays, sub_step, "array_order")
    kernel_order = _call_reference_step(probe, arrays, sub_step, "kernel_order")

    per_target = {}
    for target in sorted(array_order):
        a = np.ascontiguousarray(array_order[target]).view(np.uint32)
        k = np.ascontiguousarray(kernel_order[target]).view(np.uint32)
        per_target[target] = int(np.count_nonzero(a != k))
    assert sum(per_target.values()) > 0, (
        f"{sub_step}: the two groupings produced identical bytes on these "
        f"operands ({per_target}), so this leg cannot tell an associativity "
        f"regression from a correct kernel — choose operands or a dtdx where "
        f"they separate rather than keeping a test that cannot fail")

    # And the grouping argument is validated, not silently accepted.
    with pytest.raises(ValueError):
        _call_reference_step(probe, arrays, sub_step, "left_to_right")


def test_every_call_of_a_reference_helper_binds_the_signature_it_resolves_to(probe):
    """The defect class itself, at file scope: a call that its callee cannot accept.

    The shadowed reference was invisible to every test because nothing executed
    it; it was NOT invisible to the module's own AST. This walks every bare-name
    call of a module-level ``reference_*``/``one_*`` helper and binds it against
    the signature that name ACTUALLY resolves to at import — which is the last
    definition, not the nearest one above the call.
    """
    source_path = HERE / "probe_fused_kernel_bit_identity.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    live = {name: obj for name, obj in vars(probe).items()
            if callable(obj) and getattr(obj, "__module__", None) == probe.__name__
            and (name.startswith("reference") or name.startswith("one_"))}

    violations = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        target = live.get(node.func.id)
        if target is None:
            continue
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
                kw.arg is None for kw in node.keywords):
            continue  # Unpacked at the call site: nothing to refute statically.
        positional = ["<positional>"] * len(node.args)
        keywords = {kw.arg: "<keyword>" for kw in node.keywords}
        try:
            inspect.signature(target).bind(*positional, **keywords)
        except TypeError as exc:
            violations.append(
                f"{source_path.name}:{node.lineno}: {node.func.id}"
                f"{inspect.signature(target)} cannot accept this call "
                f"({len(positional)} positional, {sorted(keywords)}): {exc}")

    assert not violations, (
        "a call site does not fit the definition its name resolves to — the "
        "shape that made the all-periodic reference unexecutable:\n"
        + "\n".join(violations))
