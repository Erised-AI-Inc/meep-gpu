"""The off-diagonal CUDA gate's clauses that need no device — which is most of them.

THE GATE ITSELF NEEDS A GPU for one thing only: comparing NVRTC's output against
the array path. Everything that decides WHAT THAT COMPARISON MEANS runs here, and
every one of them is a silent wrong ANSWER rather than a crash:

* whether the fixture can distinguish anything at all — a case whose coupling is
  identically zero, or whose oracle moved no word, is a pass nobody earned, and
  the gate must REFUSE it rather than count it;
* whether each planted defect still matches the text it was written against —
  a mutation that has drifted apart from the emitter exercises nothing and
  reports a pass while doing it;
* whether the defects predicted to be NULLS really are, and whether their
  discriminators really can fail — a battery of only-must-be-caught legs scores
  identically whether the comparator works or has degenerated into failing
  everything;
* whether the harness knows WHICH defects its off-device backend can see. The
  evaluator is a finditer over an anchored grammar, so a mutation that rewrites a
  statement out of that grammar is silently not applied; a leg scored off such a
  run is a statement about the parser. That classification is measured here.
* whether a run on the off-device backend can be mistaken for a certification.

The device legs are exercised here too, against a stand-in that is DELIBERATELY
WRONG, so "the gate reports DIVERGENT when it should" is a measurement rather
than a property nobody tried to break.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).parent
REPO_API = HERE.parents[1]
if str(REPO_API) not in sys.path:
    sys.path.insert(0, str(REPO_API))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

GATE_PATH = HERE / "gate_cuda_offdiag.py"
POLICY_PROBE_PATH = HERE / "probe_cuda_offdiag_policy_binaries.py"
KERNEL_MODULE = REPO_API / "meep_gpu" / "cuda_kernels" / "offdiag_constitutive_kernels.py"
RECORD = REPO_API / "meep_gpu" / "cuda_kernels" / "certification.json"


def load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = load(GATE_PATH, "gate_cuda_offdiag_under_test")
policy_probe = load(POLICY_PROBE_PATH, "probe_cuda_offdiag_policy_binaries_under_test")

from meep_gpu.cuda_kernels import coverage, offdiag_emitter  # noqa: E402


# ---------------------------------------------------------------------------
# The defects still describe the text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(gate.SOURCE_MUTATIONS))
def test_every_source_mutation_still_matches_the_emitted_text(name):
    """A mutation that matched nothing has drifted apart from the emitter.

    Checked on every mask the corresponding leg is armed on, because the emitted
    source VARIES with the mask: a needle anchored on a statement only the full
    tensor emits would match nothing on the corpus's 15-row mask, and the leg
    would report a verdict for a defect it never planted.
    """
    transform = gate.SOURCE_MUTATIONS[name]
    leg = next((l for l in gate.MUTATION_LEGS if l.get("mutation") == name), None)
    masks = leg.get("masks", gate.GATE_ROW_MASKS[:2]) if leg else gate.GATE_ROW_MASKS[:2]
    for mask in masks:
        original = offdiag_emitter.offdiag_source(mask)
        mutated, sites = transform(original)
        assert sites > 0, f"{name} matched nothing in mask {mask}"
        assert mutated != original, (
            f"{name} matched {sites} site(s) in mask {mask} and left the source "
            f"byte-identical; it replaces its anchor with itself")


def test_every_leg_names_a_mutation_the_gate_implements():
    for leg in gate.MUTATION_LEGS:
        if leg["kind"] == "source":
            assert leg["mutation"] in gate.SOURCE_MUTATIONS, leg["leg"]
        elif leg["kind"] == "host":
            assert leg["mutation"] in gate.HOST_MUTATIONS, leg["leg"]
        else:
            assert leg["mutation"] is None, leg["leg"]


def test_every_implemented_mutation_is_driven_by_a_leg():
    """The converse: a defect nobody arms is a defect nobody measured."""
    armed = {leg["mutation"] for leg in gate.MUTATION_LEGS}
    assert set(gate.SOURCE_MUTATIONS) <= armed
    assert set(gate.HOST_MUTATIONS) <= armed


def test_the_four_defects_this_round_owes_are_all_armed():
    """The off-diagonal STRUCTURE specifically, not the tail it shares.

    Coupling the wrong components, dropping a row from the volume sum, inverting
    the wall mask and transposing the coefficient table are the four this family
    exists to be proof against; everything else on the list is either shared with
    a certified sibling or a control.
    """
    armed = {leg["mutation"] for leg in gate.MUTATION_LEGS}
    for required in ("couple_the_wrong_components", "drop_a_row_from_the_volume_sum",
                     "invert_the_wall_mask", "transpose_the_chi1inv_term_table"):
        assert required in armed, required
        assert required in gate.SOURCE_MUTATIONS


def test_each_null_has_a_discriminating_leg_that_must_be_caught():
    """"Inert" only means something beside a demonstration that a leg can fail."""
    caught = {leg["mutation"] for leg in gate.MUTATION_LEGS
              if leg["expect"] in ("caught", "null_on_periodic")}
    assert gate.NULL_MUTATIONS, "a gate with no null control cannot distinguish a "
    for null in gate.NULL_MUTATIONS:
        legs = [l for l in gate.MUTATION_LEGS if l["mutation"] == null]
        assert legs and any(l["expect"] == "uncaught" for l in legs), null
    # distribute_the_quarter edits offdiag_term's scale; hoist_the_coefficient
    # edits the same expression's association and must be caught.
    assert "hoist_the_coefficient" in caught
    # commute_the_diagonal_product edits the operand ORDER of a product; the hoist
    # is again the association discriminator.
    assert len(caught) >= len(gate.NULL_MUTATIONS)


def _declared_rows(source: str):
    import re  # noqa: PLC0415
    return re.findall(r"const float\* (chi1inv_E[xyz]_E[xyz])", source)


def _rows_read_by_terms(source: str):
    import re  # noqa: PLC0415
    return re.findall(r"\n        D[xyz], (chi1inv_E[xyz]_E[xyz]), idx,", source)


def test_the_transpose_is_armed_only_on_transpose_closed_masks():
    """On any other mask the rewrite names a parameter the signature never declared
    and NVRTC fails to compile. A compile failure is not a catch, and scoring it as
    one would be a lie about what the leg measured."""
    leg = next(l for l in gate.MUTATION_LEGS
               if l["mutation"] == "transpose_the_chi1inv_term_table")
    for mask in leg["masks"]:
        live = {coverage.OFFDIAG_ROW_SLOTS[slot]
                for slot, flag in enumerate(mask) if flag}
        assert {(partner, row) for row, partner in live} == live, (
            f"{mask} is not closed under transpose")
        source = offdiag_emitter.offdiag_source(mask)
        mutated, _sites = gate.m_transpose_the_chi1inv_term_table(source)
        assert set(_rows_read_by_terms(mutated)) <= set(_declared_rows(mutated)), (
            f"{mask}: the transpose reads a row the signature never declared")


def test_the_transpose_moves_the_READS_and_leaves_the_DECLARATIONS_alone():
    """THE REGRESSION FOR A MEASURED FALSE NULL, 2026-08-16.

    The first spelling of this defect renamed the signature too, which is a
    consistent ALPHA RENAME: the parameter in each position stays bound to the
    same volume and is still read by the same term, so the arithmetic is
    untouched. The keep leg reported it 8/8 identical against an expectation of
    'caught', which is the leg doing its job on a defect that was not one. What
    makes it a defect is moving the USE while the caller's binding order is fixed.
    """
    for mask in (m for m in gate.TRANSPOSE_CLOSED_MASKS):
        source = offdiag_emitter.offdiag_source(mask)
        mutated, sites = gate.m_transpose_the_chi1inv_term_table(source)
        assert sites == sum(mask)
        assert _declared_rows(mutated) == _declared_rows(source), (
            "the signature moved: this is an alpha rename and cannot be a defect")
        reads_before, reads_after = _rows_read_by_terms(source), _rows_read_by_terms(mutated)
        assert reads_after != reads_before
        for before, after in zip(reads_before, reads_after):
            row, partner = before.split("_")[1:]
            assert after == f"chi1inv_{partner}_{row}"


def test_the_transpose_is_caught_off_device_before_a_slot_is_spent():
    """It edits text the evaluator EXECUTES, so the catch is measurable here — and
    was not, on the first spelling. A defect this leg cannot demonstrate off device
    is a defect the next device leg would have to discover."""
    backend = gate.build_backend("numpy")
    xp = gate._NumpyWearingCupysName()
    backend.set_source_mutation(gate.SOURCE_MUTATIONS[
        "transpose_the_chi1inv_term_table"])
    record = gate.one_case(backend, xp, (8.0, 8.0, 8.0), "8x8x8",
                           ("metallic", "metallic", "periodic"),
                           (1, 0, 0, 1, 0, 0), "varying", 0.35, "uniform",
                           "fmad_false", ("--fmad=false",), 4)
    assert gate.case_is_valid(record)[0]
    assert not record["bit_identical"], (
        "transposing the coefficient reads changed no bit; the fixture's row "
        "volumes must be independent per slot or this defect is invisible")


# ---------------------------------------------------------------------------
# The fixture can distinguish something — and a fixture that cannot is REFUSED
# ---------------------------------------------------------------------------

def test_the_shipped_source_reproduces_the_array_path_on_every_gate_mask():
    """The headline claim, off device: the emitted tree IS the array path's.

    Run through the gate's own ``one_case``, so what is measured is the code path
    the device leg takes and not a second arrangement of the same pieces.
    """
    backend = gate.build_backend("numpy")
    xp = gate._NumpyWearingCupysName()
    for mask in gate.GATE_ROW_MASKS:
        record = gate.one_case(backend, xp, (8.0, 8.0, 8.0), "8x8x8",
                               ("metallic", "metallic", "periodic"), mask,
                               "varying", 0.35, "uniform", "fmad_false",
                               ("--fmad=false",), 4)
        assert gate.case_is_valid(record)[0], record["refused_because"]
        assert record["bit_identical"], (
            f"{mask}: {record['differing_floats']} of {record['total_floats']} "
            f"words differ, max_ulp={record['max_ulp']}")


def test_a_case_whose_oracle_moved_nothing_is_refused_not_passed():
    """ZERO-INIT IS A FIXED POINT of this sub-step: with D, E and f_w all zero
    every tree agrees, so a gate that counted such a case would be certifying the
    seed. Driven by handing ``one_case``'s validity check a zeroed record."""
    record = {"oracle_moved": False, "coupling_is_live": True}
    valid, why = gate.case_is_valid(record)
    assert not valid and "fixed point" in why


def test_a_case_with_no_live_coupling_is_refused_not_passed():
    """The floor this family needs beyond the sibling's: without the coupling the
    kernel computes the certified PLAIN constitutive kernel's arithmetic, and a
    pass would be a statement about that one."""
    record = {"oracle_moved": True, "coupling_is_live": False}
    valid, why = gate.case_is_valid(record)
    assert not valid and "coupling" in why


def test_the_swept_cases_all_carry_a_live_coupling_and_a_moved_oracle():
    """Both floors, measured on the real product rather than asserted of it."""
    backend = gate.build_backend("numpy")
    xp = gate._NumpyWearingCupysName()
    for spec in gate.case_product("reduced", 1)[:6]:
        record = gate.one_case(backend, xp, spec["cell"], spec["shape_label"],
                               spec["boundaries"], spec["mask"], spec["row_form"],
                               spec["courant"], spec["value_class"], "fmad_false",
                               ("--fmad=false",), spec["steps"])
        assert record["oracle_moved"], spec
        assert record["coupling_is_live"], spec


def test_the_band_class_reaches_the_band_and_the_uniform_class_cannot():
    """A leg run under a subnormal policy whose operands hold no subnormal measures
    nothing and reports a pass while doing it. Measured per class, off device."""
    grid_shape = (8, 8, 8)
    rng = np.random.default_rng(7)
    uniform = {name: rng.uniform(-1.0, 1.0, size=grid_shape).astype(np.float32)
               for name in gate.STATE}
    census = gate.operand_census(uniform)
    assert census["subnormals"] == 0 and census["negative_zeros"] == 0
    band = gate.subnormal_band_hosts(gate.STATE, grid_shape,
                                     np.random.default_rng(7))
    census = gate.operand_census(band)
    assert census["subnormals"] > 0 and census["negative_zeros"] > 0


# ---------------------------------------------------------------------------
# The gate can FAIL — driven against a deliberately wrong stand-in
# ---------------------------------------------------------------------------

class _WrongBackend(gate.EvaluatorBackend):
    """A kernel side that is wrong by one word. Nothing more is needed: the
    comparator is on RAW BYTES, so a single word is a divergence."""

    def launch(self, fields, layer, rows, mask, tables, codes, walls):
        super().launch(fields, layer, rows, mask, tables, codes, walls)
        flat = fields.Ex.reshape(-1)
        flat[0] = np.float32(flat[0] + np.float32(1.0))


def test_the_gate_reports_divergent_when_the_kernel_side_is_wrong():
    """Without this, "identical" is a claim nobody showed could come out otherwise."""
    xp = gate._NumpyWearingCupysName()
    record = gate.one_case(_WrongBackend(), xp, (8.0, 8.0, 8.0), "8x8x8",
                           ("metallic", "metallic", "periodic"),
                           (1, 1, 1, 1, 1, 1), "varying", 0.35, "uniform",
                           "fmad_false", ("--fmad=false",), 1)
    assert not record["bit_identical"]
    assert record["differing_floats"] >= 1


def test_a_launch_that_raises_is_recorded_as_a_divergence_not_swallowed():
    """A kernel that could not run is not a kernel that agreed."""

    class _Raising(gate.EvaluatorBackend):
        def launch(self, *args, **kwargs):
            raise RuntimeError("planted")

    xp = gate._NumpyWearingCupysName()
    record = gate.one_case(_Raising(), xp, (8.0, 8.0, 8.0), "8x8x8",
                           ("periodic", "periodic", "periodic"),
                           (1, 0, 0, 1, 0, 0), "varying", 0.35, "uniform",
                           "fmad_false", ("--fmad=false",), 1)
    assert record["launch_error"] and "planted" in record["launch_error"]
    assert not record["bit_identical"]


# ---------------------------------------------------------------------------
# The off-device backend knows which defects it cannot see
# ---------------------------------------------------------------------------

def test_the_evaluator_grammar_covers_every_shipped_source():
    """The premise of the whole off-device backend: nothing in an unmutated
    emitted source falls outside the grammar."""
    for mask in gate.GATE_ROW_MASKS:
        assert gate.evaluator_blind_lines(
            offdiag_emitter.offdiag_source(mask)) == []


@pytest.mark.parametrize("name,expected", [
    ("couple_the_wrong_components", "arithmetic"),
    ("hoist_the_coefficient", "arithmetic"),
    ("drop_the_wall_mask", "arithmetic"),
    ("column_major_index", "pinned_half"),
    ("flatten_the_tail", "pinned_half"),
    ("store_fw_before_reading_prev", "pinned_half"),
    ("drop_fw_store", "pinned_half"),
    ("invert_the_wall_mask", "outside_the_grammar"),
    ("over_apply_the_wall_mask", "outside_the_grammar"),
    ("commute_the_diagonal_product", "outside_the_grammar"),
])
def test_the_classifier_says_which_defects_the_evaluator_can_see(name, expected):
    """THE MEASURED HAZARD, pinned per defect.

    On this gate's first laptop run, four defects planted in the pinned half came
    back UNCAUGHT and three planted outside the grammar came back with verdicts
    that were about the parser: inverting the wall-mask select leaves a line the
    ``_MASK`` pattern no longer matches, so off device it degenerates into
    DROPPING the mask — a different defect that happens to bite on a metallic
    grid — and commuting the diagonal product leaves ``src_Ez`` undefined, so a
    later line raises and the leg reads the raise as a catch.

    Both classes are now named, skipped off device with the reason recorded, and
    measured normally on device, where the compiler executes every line.
    """
    source = offdiag_emitter.offdiag_source((1, 0, 0, 1, 0, 0))
    mutated, sites = gate.SOURCE_MUTATIONS[name](source)
    assert sites > 0
    verdict = gate.classify_for_evaluator(source, mutated)
    assert verdict["why"] == expected, verdict
    assert verdict["evaluator_sees_it"] == (expected == "arithmetic")


def test_the_pinned_half_is_what_the_slice_tests_pin_by_exact_string():
    """The other half of the pinned-half claim: those defects are not unguarded off
    device, they are guarded by a different mechanism."""
    slice_tests = gate.slice_module()
    source = offdiag_emitter.offdiag_source((1, 1, 1, 1, 1, 1))
    for name in ("column_major_index", "flatten_the_tail",
                 "store_fw_before_reading_prev"):
        mutated, _sites = gate.SOURCE_MUTATIONS[name](source)
        pinned = dict(slice_tests.PINNED_HELPERS)
        pinned["constitutive_apply"] = (
            "    float prev = fw[idx];\n    fw[idx] = src;\n"
            "    float a = f[idx] + kps * src;\n    f[idx] = a - kms * prev;")
        assert any(body not in mutated for body in pinned.values()), name


def test_a_leg_the_backend_cannot_see_is_recorded_not_scored(tmp_path):
    """It must appear in the artifact naming the reason, with ``as_required`` None
    — a skipped leg silently counted as a pass is the failure this exists for."""
    results = {}
    backend = gate.build_backend("numpy")
    xp = gate._NumpyWearingCupysName()
    gate.run_mutations(backend, xp, results, str(tmp_path / "m.json"),
                       "reduced", 1)
    legs = {leg["leg"]: leg for leg in results["mutations"]["legs"]}
    skipped = [leg for leg in legs.values()
               if leg.get("measurable_on_this_backend") is False]
    assert skipped, "the numpy backend saw every defect; the classifier is inert"
    for leg in skipped:
        assert leg["as_required"] is None
        assert leg["why_not_measured_here"]
    assert results["mutations"]["legs_not_measurable_on_this_backend"] == len(skipped)


# ---------------------------------------------------------------------------
# The off-device backend cannot certify
# ---------------------------------------------------------------------------

def test_the_numpy_backend_never_reports_passed(tmp_path):
    out = tmp_path / "gate.json"
    rc = gate.main(["--backend", "numpy", "--product", "reduced",
                    "--legs", "bytes", "--out", str(out)])
    payload = json.loads(out.read_text())
    assert payload["certifies"] is False
    assert payload["verdict"]["passed"] is False
    assert "compiles nothing" in payload["verdict"]["why_not_certified"]
    assert rc == 0  # the harness legs still have to come out right


def test_the_artifact_carries_the_subject_digests_and_the_emitter_corpus_digest(tmp_path):
    """WHICH bytes produced this verdict. The corpus digest moves on any change to
    any of the 63 emittable sources, so a record cut against a moved emitter is
    visible rather than implied."""
    out = tmp_path / "gate.json"
    gate.main(["--backend", "numpy", "--product", "reduced", "--legs", "",
               "--out", str(out)])
    payload = json.loads(out.read_text())
    assert payload["emitter_corpus_digest"] == offdiag_emitter.corpus_digest()
    for name in ("offdiag_emitter.py", "offdiag_constitutive_kernels.py",
                 "coverage.py", "step_curl_kernels.py"):
        assert len(payload["subjects"][name]) == 64


# ---------------------------------------------------------------------------
# Budgets, guards and the product
# ---------------------------------------------------------------------------

def test_the_multi_step_budget_is_the_one_both_certified_records_are_cut_at():
    """8 is not a budget, it is a blind spot: the sibling track measured 8, 10 and
    6 consecutive steps all passing a divergence 40 did not."""
    assert gate.MULTI_STEP_BUDGET == 60
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["constitutive_2026-08-15"]["constitutive_multi_step_budget"] == 60


def test_the_guard_sweep_carries_a_set_with_the_guard_OFF():
    """The pair (identical with, non-identical without) is what makes the guard
    evidence rather than decoration; a gate with only the primary set asserts it."""
    labels = [label for label, _token, _primary in gate.GUARD_SETS]
    assert "fmad_false" in labels and "default_no_options" in labels
    primary = [label for label, _t, is_primary in gate.GUARD_SETS if is_primary]
    assert primary == ["fmad_false"]


def test_the_product_sweeps_a_courant_that_is_not_exactly_representable():
    """0.5 is exact in float32, so a contracted and an uncontracted expression
    round identically there and the guard cannot be measured. Both are swept and
    the control is scored at the inexact one."""
    assert gate.INEXACT_COURANT in gate.COURANTS
    assert float(np.float32(gate.INEXACT_COURANT)) != gate.INEXACT_COURANT
    assert float(np.float32(0.5)) == 0.5
    keys = {spec["courant"] for spec in gate.case_product("full", 1)}
    assert keys == set(gate.COURANTS)


def test_the_product_sweeps_both_row_forms_and_both_value_classes():
    specs = gate.case_product("full", 1)
    assert {s["row_form"] for s in specs} == set(gate.ROW_FORMS)
    assert {s["value_class"] for s in specs} == set(gate.VALUE_CLASSES)
    assert {tuple(s["mask"]) for s in specs} == {tuple(m) for m in gate.GATE_ROW_MASKS}


def test_the_product_includes_an_invariant_axis_and_refuses_the_impossible_pair():
    """An axis of one cell makes the partner pair ``2*g`` rather than a ghost zero
    — MEEP's ``stride(d) = 0`` double read — and it is the one configuration where
    the two shift helpers return the same cell."""
    assert any(1.0 in cell for cell, _label in gate.SHAPES)
    assert gate.config_viable((1.0, 16.0, 16.0),
                              ("metallic", "periodic", "periodic")) is not None
    assert gate.config_viable((1.0, 16.0, 16.0),
                              ("periodic", "metallic", "metallic")) is None


def test_the_corpus_row_masks_are_both_in_the_product():
    """The two masks the 186-row corpus actually drives, from the predicate
    battery's own census. A gate that swept neither would certify a variant no row
    asks for."""
    slice_tests = gate.slice_module()
    for mask in slice_tests.CORPUS_ROW_MASKS:
        assert tuple(mask) in {tuple(m) for m in gate.GATE_ROW_MASKS}


# ---------------------------------------------------------------------------
# The policy-binaries probe
# ---------------------------------------------------------------------------

def test_the_needle_arithmetic_is_what_the_probe_claims_it_is():
    """THE PROBE'S OWN PREDICTION, evaluated off device from the shipped text.

    ``2n`` on the coupled component and ``n`` on the uncoupled ones, exactly, as
    bit patterns — every intermediate is a power of two inside the subnormal band.
    A probe whose keep-leg expectation were wrong would fail a healthy kernel.
    """
    shape = policy_probe.SHAPE
    bits = policy_probe.NEEDLE_BITS
    needle = np.frombuffer(np.uint32(bits).tobytes(), dtype=np.float32)[0]
    arrays = {name: np.zeros(shape, dtype=np.float32)
              for name in gate.OUTPUTS}
    for name in ("Dx", "Dy", "Dz"):
        arrays[name] = np.full(shape, needle, dtype=np.float32)
    for component in gate.COMPONENTS:
        arrays[f"inv_eps_{component}"] = np.ones(shape, dtype=np.float32)
    for slot, (row, partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if policy_probe.NEEDLE_MASK[slot]:
            arrays[f"chi1inv_{row}_{partner}"] = np.ones(shape, dtype=np.float32)
    tables = {f"{stem}_{axis}": (np.ones(shape[i], dtype=np.float32) if stem == "kps"
                                 else np.zeros(shape[i], dtype=np.float32))
              for i, axis in enumerate("xyz") for stem in ("kps", "kms")}
    gate.evaluator()(offdiag_emitter.offdiag_source(policy_probe.NEEDLE_MASK),
                     arrays, tables, (0, 0, 0), (0, 0, 0), shape)

    def word(name):
        return int(arrays[name].ravel()[0].view(np.uint32))

    assert word("Ex") == 2 * bits, "the coupled component is not exactly 2n"
    assert word("Ey") == bits and word("Ez") == bits
    assert word("Ex") != word("Ey"), (
        "the coupled and uncoupled components agree, so the coupling never fired "
        "and the needle would measure the plain constitutive arithmetic")


def test_the_probe_compiles_the_two_corpus_masks_and_a_plain_arm_mask():
    slice_tests = gate.slice_module()
    for mask in slice_tests.CORPUS_ROW_MASKS:
        assert tuple(mask) in {tuple(m) for m in policy_probe.MASKS}
    assert policy_probe.NEEDLE_MASK in policy_probe.MASKS
    assert sum(policy_probe.NEEDLE_MASK) == 1, (
        "the needle mask must leave two components on the PLAIN arm, or the "
        "coupled/uncoupled discriminator has nothing to compare")


def test_the_probe_row_masks_emit_distinct_sources():
    """Three masks that compiled one source would be one variant counted three
    times, and the distinct-binary verdict would be about a single kernel."""
    sources = {offdiag_emitter.offdiag_source(mask) for mask in policy_probe.MASKS}
    assert len(sources) == len(policy_probe.MASKS)


# ---------------------------------------------------------------------------
# Platform facts this family depends on
# ---------------------------------------------------------------------------

def test_the_swept_primary_guard_is_the_tuple_the_kernel_module_ships():
    """``--fmad=false`` is CORRECTNESS, not tuning, and it is spelled in the kernel
    module so a by-path load cannot pick up a different tuple. The gate SWEEPS
    options, so its primary set must be exactly what the file ships — a gate whose
    primary guard drifted from the module's would certify bytes nothing launches.
    """
    import ast  # noqa: PLC0415

    shipped = None
    for node in ast.parse(KERNEL_MODULE.read_text(encoding="utf-8")).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_COMPILE_OPTIONS"):
            shipped = ast.literal_eval(node.value)
    assert shipped == ("--fmad=false",)
    primary = [token for _label, token, is_primary in gate.GUARD_SETS if is_primary]
    assert primary == [shipped]
    off = [token for _label, token, is_primary in gate.GUARD_SETS if not is_primary]
    assert off == [()], "the control set must be the guard OFF, not another guard"


def test_every_emitted_source_the_gate_and_probe_drive_is_pure_ascii():
    """A COMPILE REQUIREMENT: ``compile_using_nvrtc`` writes the source through a
    bare ``open(..., 'w')``, so the bytes go through the interpreter's LOCALE
    encoding — ASCII under C/POSIX, which is what a non-interactive shell on the
    validation host gets. Two em-dashes killed the certified sibling's E kernel at
    its first launch."""
    masks = set(tuple(m) for m in gate.GATE_ROW_MASKS) | set(
        tuple(m) for m in policy_probe.MASKS)
    for mask in masks:
        offdiag_emitter.offdiag_source(mask).encode("ascii")


def test_no_unary_minus_appears_on_any_float_path_of_a_mutated_source():
    """CUDA lowers ``-x`` to ``neg.f32`` rather than to ``0.0f - x``, so it does
    NOT canonicalize signed zeros the way the sibling track's platform does. That
    track's negation idiom must not arrive here through a mutation either."""
    for name, transform in gate.SOURCE_MUTATIONS.items():
        mutated, _sites = transform(
            offdiag_emitter.offdiag_source((1, 1, 1, 1, 1, 1)))
        code = "\n".join(line.split("//")[0] for line in mutated.split("\n"))
        assert "* -1.0" not in code and "-1.0f *" not in code, name
