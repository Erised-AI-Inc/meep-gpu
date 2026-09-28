"""The dynamic-loop gate's clauses that need no device — which is most of them.

THE GATE ITSELF NEEDS A GPU, for one thing only: the byte comparison between the
two compiled binaries. Everything that DECIDES WHAT THAT COMPARISON MEANS does
not, and each of those is a silent wrong ANSWER rather than a crash:

* whether the axis under test is REAL in the corpus and its arity actually VARIES
  (a synthetic axis nothing drives would make the whole measurement worthless);
* whether the transcribed array-path reference is the array path (rule 3);
* whether the operand class can separate two association orders at all -- if it
  cannot, "bit-identical" is a fact about the draw and not about the kernels;
* whether the controls can fire, at which arity, and whether the ones that are
  exact identities at low arity are correctly left disarmed there;
* whether the two variants really are one axis apart -- same prelude, same
  compile options, same index decomposition, same downstream arithmetic;
* whether the sources can be handed to NVRTC at all (the ASCII trap that killed
  the constitutive E kernel at its first launch on 2026-08-15).

Every one of those is checked here, on the laptop that is the merge bar.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import pathlib
import re
import sys

import numpy as np
import pytest

HERE = pathlib.Path(__file__).parent
REPO_API = HERE.parents[1]
if str(REPO_API) not in sys.path:
    sys.path.insert(0, str(REPO_API))

SUBJECT_PATH = HERE / "cuda_dynamic_loop_arity.py"
PROBE_PATH = HERE / "probe_cuda_dynamic_loop_arity.py"
CONSTITUTIVE_MODULE = REPO_API / "meep_gpu" / "cuda_kernels" / "constitutive_kernels.py"
CURL_MODULE = REPO_API / "meep_gpu" / "cuda_kernels" / "step_curl_kernels.py"

#: The coverage artifacts the corpus arity census is re-derived from.
COVERAGE_DIR = (REPO_API / "parity" / "meep_gpu" / "results"
                / "cuda_predicate_coverage_2026-08-15_constitutive")
COVERAGE_FILES = ("examples.jsonl", "tests.jsonl", "tests_param_matched.jsonl")


def load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


subject = load(SUBJECT_PATH, "cuda_dynamic_loop_arity_under_test")


# ---------------------------------------------------------------------------
# The axis is real, and its arity really varies
# ---------------------------------------------------------------------------

def test_the_corpus_arity_census_is_rederived_from_the_artifacts():
    """``CORPUS_ARITIES`` is a MEASUREMENT and this re-runs it.

    The whole justification for picking ``NP`` over the other arity axes is that
    the corpus drives four distinct values of it across fifteen rows. A constant
    transcribed once and never checked is how that justification would quietly
    become false — a re-cut coverage run that changed the row set would leave
    this module claiming a census nobody re-took.
    """
    if not COVERAGE_DIR.is_dir():
        pytest.skip(f"coverage artifacts absent: {COVERAGE_DIR}")
    rows = []
    for name in COVERAGE_FILES:
        path = COVERAGE_DIR / name
        if path.is_file():
            rows.extend(json.loads(line) for line in path.open())
    assert rows, "no coverage rows were read; the census would be vacuous"

    per_row = collections.Counter()
    for row in rows:
        states = row.get("polarization") or []
        if not states:
            continue
        counts = {"Ex": 0, "Ey": 0, "Ez": 0}
        for state in states:
            for component in state.get("driven", ()):
                if component in counts:
                    counts[component] += 1
        live = set(counts.values())
        # The measured fact the sweep leans on: every dispersive corpus row has
        # the SAME live pole count on all three components.
        assert len(live) == 1, (
            f"{row.get('row')}: expected a uniform per-component pole count, got "
            f"{counts} — the sweep's symmetric triples would no longer cover it")
        per_row[live.pop()] += 1

    assert dict(per_row) == subject.CORPUS_ARITIES, (
        f"the corpus census moved: measured {dict(per_row)}, module claims "
        f"{subject.CORPUS_ARITIES}")
    assert len(per_row) >= 3, (
        "an arity axis with fewer than three distinct corpus values is not an "
        "axis this measurement can speak about")


def test_the_sweep_covers_every_arity_the_corpus_drives():
    covered = {triple for case in subject.sweep_cases()
               for triple in (max(case["arities"]),)}
    missing = set(subject.CORPUS_ARITIES) - covered
    assert not missing, f"corpus arities not swept: {sorted(missing)}"
    assert 0 in subject.SWEEP_ARITIES, "the degenerate arity must be swept"
    assert subject.MAX_POLES in subject.SWEEP_ARITIES


# ---------------------------------------------------------------------------
# The reference IS the array path (the validate-against-the-reference rule)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("poles", [0, 1, 2, 5, 6, 8])
def test_the_reference_is_bit_identical_to_stepping_update_E(poles):
    """Transcription validated by MEASUREMENT against ``stepping.update_E``.

    Real ``Grid``/``Fields``/``PML``/``PolarizationState`` objects, the engine's
    own ones, driven through the engine's own sub-step; the reference is handed
    the same arrays and must return the same BYTES. This is the clause that makes
    the third leg of the gate — "does either kernel also match the array path" —
    mean anything.
    """
    from meep_gpu import stepping
    from meep_gpu.dispersion import PolarizationState, Susceptibility
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid
    from meep_gpu.pml import PML

    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    fields.enable_field_storage()
    pml = PML(grid=grid, thickness=2)
    shape = grid.shape
    rng = np.random.default_rng(20260815 + poles)

    states = []
    for index in range(poles):
        term = Susceptibility(frequency=1.0, gamma=0.1, kind="lorentzian")
        sigmas = {name: 0.3 + 0.05 * index for name in ("Ex", "Ey", "Ez")}
        state = PolarizationState(term, sigmas, grid, np.float32)
        fields.polarizations.append(state)
        states.append(state)

    for name in ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = subject.draw(shape, rng, "uniform")
    for state in states:
        for component in ("Ex", "Ey", "Ez"):
            state.P[component][...] = subject.draw(shape, rng, "uniform")

    snapshot = {name: np.array(getattr(fields, name), copy=True)
                for name in ("Ex", "Ey", "Ez", "Dx", "Dy", "Dz",
                             "f_w_Ex", "f_w_Ey", "f_w_Ez")}
    for component, prefix in (("Ex", "Px"), ("Ey", "Py"), ("Ez", "Pz")):
        for index in range(subject.MAX_POLES):
            snapshot[f"{prefix}{index}"] = (
                np.array(states[index].P[component], copy=True) if index < poles
                else np.zeros(shape, dtype=np.float32))
        snapshot["inv_eps_" + component] = np.asarray(
            fields.inverse_epsilon_for(component) * np.ones(shape, dtype=np.float32),
            dtype=np.float32)

    kps = [np.asarray(getattr(pml, f"kps_{axis}_h"), dtype=np.float32)
           for axis in ("x", "y", "z")]
    kms = [np.asarray(getattr(pml, f"kms_{axis}_h"), dtype=np.float32)
           for axis in ("x", "y", "z")]

    reference = subject.reference_update_E(snapshot, (poles,) * 3, kps, kms)
    stepping.update_E(fields, pml)

    for name in subject.OUTPUT_NAMES:
        engine = subject.as_words(np.asarray(getattr(fields, name), dtype=np.float32))
        mine = subject.as_words(reference[name])
        differing = int(np.count_nonzero(engine != mine))
        assert differing == 0, (
            f"{name} at {poles} poles: {differing}/{engine.size} words differ from "
            f"stepping.update_E — the reference is not the array path")


def test_the_reference_leg_is_not_vacuous_at_zero_poles():
    """At NP=0 the reference must still WRITE, or the zero-pole leg proves nothing.

    Zero-init is a fixed point of this sub-step and the zero arity is the one most
    likely to be silently a no-op. The reference at NP=0 must still move E and
    f_w_E away from their inputs.
    """
    shape = (4, 5, 6)
    rng = np.random.default_rng(11)
    case = subject.make_case(shape, (0, 0, 0), "uniform", 11)
    before = {name: np.array(case["state"][name], copy=True)
              for name in subject.OUTPUT_NAMES}
    after = subject.reference_update_E(
        case["state"], (0, 0, 0), case["kps_broadcast"], case["kms_broadcast"])
    moved = subject.moved_words(before, after)
    assert moved["moved_fraction"] > 0.9, moved
    assert rng is not None  # the seed path is make_case's, not this test's


# ---------------------------------------------------------------------------
# The operand class can separate association orders — the vacuity precondition
# ---------------------------------------------------------------------------

def _orders(shape, poles, value_class, seed):
    """Forward, reversed, pre-summed and one-short sums of the same terms."""
    rng = np.random.default_rng(seed)
    displacement = subject.draw(shape, rng, value_class)
    terms = [subject.draw(shape, rng, value_class) for _ in range(poles)]
    forward = displacement.copy()
    for term in terms:
        forward = (forward - term).astype(np.float32)
    backward = displacement.copy()
    for term in reversed(terms):
        backward = (backward - term).astype(np.float32)
    accumulated = np.zeros(shape, dtype=np.float32)
    for term in terms:
        accumulated = (accumulated + term).astype(np.float32)
    presummed = (displacement - accumulated).astype(np.float32)
    short = displacement.copy()
    for term in terms[:-1] if terms else []:
        short = (short - term).astype(np.float32)
    return forward, backward, presummed, short


def _fraction(left, right):
    a, b = subject.as_words(left).ravel(), subject.as_words(right).ravel()
    return float(np.count_nonzero(a != b)) / a.size


def _outputs(case, mode):
    """One sum order driven through the FULL reference chain to the OUTPUTS.

    THE QUANTITY THE GATE COMPARES. An earlier version of this file scored the
    raw pole sum ``s`` instead, which is the quantity the mutation targets but
    not the one the byte comparison reads, and it overstated the gate's
    discrimination by 41% at NP=2 — enough to set the floor above what the device
    could deliver and fail sixteen legs that were measuring correctly. See
    ``CONTROL_CATCH_FLOOR``.
    """
    state = case["state"]
    arities = case["arities"]
    out = {}
    for index, (field, source, prefix) in enumerate(
            (("Ex", "Dx", "Px"), ("Ey", "Dy", "Py"), ("Ez", "Dz", "Pz"))):
        terms = [state[f"{prefix}{p}"] for p in range(arities[index])]
        if mode == "reversed":
            terms = list(reversed(terms))
        accumulated = np.array(state[source], dtype=np.float32, copy=True)
        if mode == "presummed":
            total = np.zeros(case["shape"], dtype=np.float32)
            for term in terms:
                total = (total + term).astype(np.float32)
            accumulated = (accumulated - total).astype(np.float32)
        else:
            for term in terms:
                accumulated = (accumulated - term).astype(np.float32)
        src = (accumulated * state["inv_eps_" + field]).astype(np.float32)
        out["f_w_" + field] = src
        previous = np.array(state["f_w_" + field], copy=True)
        value = (state[field] + case["kps_broadcast"][index] * src).astype(np.float32)
        out[field] = (value - case["kms_broadcast"][index] * previous).astype(np.float32)
    return out


@pytest.mark.parametrize("poles", [2, 3, 5, 6, 8])
def test_the_headline_class_separates_association_orders_above_the_floor(poles):
    """THE CLAUSE THAT STOPS A VACUOUS "IDENTICAL".

    If reordering or regrouping these operands does not change the float32
    OUTPUT, then two kernels agreeing on the output says nothing about whether
    they associate the sum the same way. The floor the device leg enforces is
    checked here first, on the same class and ON THE SAME QUANTITY, so a bad
    class or a mis-set floor is caught on the laptop rather than after a device
    run.
    """
    case = subject.make_case((16, 16, 24), (poles,) * 3,
                             subject.HEADLINE_VALUE_CLASS, 4242 + poles)
    forward = _outputs(case, "forward")
    for name in ("reversed", "presummed"):
        result = subject.compare_words(forward, _outputs(case, name))
        assert result["differing_fraction"] >= subject.CONTROL_CATCH_FLOOR, (
            f"{name} at NP={poles} on the {subject.HEADLINE_VALUE_CLASS} class "
            f"moves only {result['differing_fraction']:.4f} of OUTPUT words, below "
            f"the {subject.CONTROL_CATCH_FLOOR} floor")
        assert result["differing_words"] >= subject.CONTROL_CATCH_MINIMUM_WORDS


def test_the_downstream_arithmetic_collapses_differences_and_the_floor_knows_it():
    """The measurement that corrected the floor, kept as a regression.

    The pole sum is the most order-sensitive point in the sub-step; every step
    after it rounds distinct sums back together. If a future edit made the gate
    score the sum instead of the outputs, the floor would silently become 40%
    too generous — which is the direction that admits a vacuous pass.
    """
    poles = 2
    case = subject.make_case((16, 16, 24), (poles,) * 3, "uniform", 5002)
    forward, backward = _outputs(case, "forward"), _outputs(case, "reversed")
    on_fw = subject.compare_words(forward, backward,
                                  ("f_w_Ex", "f_w_Ey", "f_w_Ez"))["differing_fraction"]
    on_field = subject.compare_words(forward, backward,
                                     ("Ex", "Ey", "Ez"))["differing_fraction"]
    on_both = subject.compare_words(forward, backward)["differing_fraction"]
    assert on_field < on_fw, (
        "the two accumulations must collapse differences the multiply left alive; "
        f"measured f_w {on_fw:.4f} vs E {on_field:.4f}")
    assert on_both < on_fw
    assert on_both >= subject.CONTROL_CATCH_FLOOR


def test_the_headline_class_was_chosen_by_measurement_not_by_argument():
    """``uniform`` really is the most order-sensitive class this module draws.

    The module docstring's table is the reason ``HEADLINE_VALUE_CLASS`` is what it
    is, and a table in a docstring is exactly the kind of claim that rots. This
    re-measures it: the headline class must beat every other class it offers, at
    every armed arity, on both the reordering and the regrouping control.
    """
    shape = (16, 16, 24)
    for poles in (2, 3, 5, 6):
        scores = {}
        for value_class in subject.VALUE_CLASSES:
            case = subject.make_case(shape, (poles,) * 3, value_class, 4242 + poles)
            forward = _outputs(case, "forward")
            scores[value_class] = tuple(
                subject.compare_words(forward, _outputs(case, mode))["differing_fraction"]
                for mode in ("reversed", "presummed"))
        best = max(scores, key=lambda name: min(scores[name]))
        assert best == subject.HEADLINE_VALUE_CLASS, (
            f"at NP={poles} the most order-sensitive class is {best!r}, not the "
            f"declared headline {subject.HEADLINE_VALUE_CLASS!r}: {scores}")


@pytest.mark.parametrize("poles", [0, 1])
def test_the_reordering_controls_are_exact_identities_at_low_arity(poles):
    """Why ``dynamic_reversed`` and ``dynamic_presummed`` are armed only at NP>=2.

    With one term there is one order and one grouping. A leg that counted a miss
    there would be reporting a failure of arithmetic to be non-commutative.
    """
    forward, backward, presummed, _ = _orders((8, 8, 8), poles, "uniform", 5)
    assert _fraction(forward, backward) == 0.0
    assert _fraction(forward, presummed) == 0.0
    assert subject.CONTROL_ARMED_AT["dynamic_reversed"] == 2
    assert subject.CONTROL_ARMED_AT["dynamic_presummed"] == 2


def test_the_short_control_is_armed_at_one_pole_and_catches_everything_there():
    forward, _, _, short = _orders((8, 8, 8), 1, "uniform", 6)
    assert subject.CONTROL_ARMED_AT["dynamic_short"] == 1
    assert _fraction(forward, short) > 0.99


# ---------------------------------------------------------------------------
# Eligibility — the asymmetric-triple trap
# ---------------------------------------------------------------------------

def test_eligible_outputs_scopes_a_control_to_the_components_it_can_reach():
    assert subject.eligible_outputs((0, 1, 2), 2) == ("Ez", "f_w_Ez")
    assert subject.eligible_outputs((0, 1, 2), 1) == ("Ey", "f_w_Ey", "Ez", "f_w_Ez")
    assert subject.eligible_outputs((0, 0, 0), 1) == ()
    assert subject.eligible_outputs((5, 5, 5), 2) == subject.OUTPUT_NAMES[0:1] + (
        "f_w_Ex", "Ey", "f_w_Ey", "Ez", "f_w_Ez")


def test_an_asymmetric_triple_is_in_the_sweep():
    """The corpus only drives symmetric triples; the sweep must not stop there.

    An anisotropic sigma makes a per-component arity reachable
    (dispersion.py:640-642 drops a component whose sigma is identically zero), and
    a per-component bound is exactly where a shared-loop-counter defect hides.
    """
    triples = {case["arities"] for case in subject.sweep_cases()}
    assert any(len(set(triple)) > 1 for triple in triples)


# ---------------------------------------------------------------------------
# The two variants are ONE AXIS APART
# ---------------------------------------------------------------------------

_SIGNATURE = re.compile(r"__device__ __forceinline__ void constitutive_apply\("
                        r".*?\n\}", re.S)


def test_the_prelude_is_the_certified_constitutive_apply():
    """Character for character, comments excluded, against the shipped module.

    The arity axis is the ``s`` line and NOTHING ELSE. If the downstream
    arithmetic in this file drifted from the certified body, a divergence found
    here would be unattributable — it could be the loop or it could be the drift.
    """
    shipped = CONSTITUTIVE_MODULE.read_text()
    mine = _SIGNATURE.search(subject._PRELUDE)
    theirs = _SIGNATURE.search(shipped)
    assert mine and theirs, "constitutive_apply not found in one of the two files"
    normalize = lambda text: "\n".join(  # noqa: E731 - a local, one line
        line.strip() for line in text.splitlines() if line.strip())
    assert normalize(mine.group(0)) == normalize(theirs.group(0))


def test_the_compile_options_equal_the_shipped_modules():
    """One tuple, three files, pinned. ``--fmad=false`` is correctness here.

    It also removes a confound from THIS measurement specifically: without it a
    loop and an unrolled chain could be contracted into FMAs differently, and the
    verdict would be about contraction rather than about association.
    """
    constitutive = re.search(r"_COMPILE_OPTIONS = (\([^)]*\))",
                             CONSTITUTIVE_MODULE.read_text())
    curl = re.search(r"_COMPILE_OPTIONS = (\([^)]*\))", CURL_MODULE.read_text())
    assert constitutive and curl
    expected = repr(subject.COMPILE_OPTIONS)
    assert eval(constitutive.group(1)) == subject.COMPILE_OPTIONS  # noqa: S307
    assert eval(curl.group(1)) == subject.COMPILE_OPTIONS  # noqa: S307
    assert "--fmad=false" in expected


def test_every_variant_shares_the_index_decomposition_and_the_prelude():
    for variant in subject.ALL_VARIANTS:
        code = subject.source_for(variant, (3, 3, 3))
        assert subject._INDEX_BLOCK in code, variant
        assert "constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);" in code
        assert "constitutive_apply(Ey, f_w_Ey, idx, src_y, kps_y[j], kms_y[j]);" in code
        assert "constitutive_apply(Ez, f_w_Ez, idx, src_z, kps_z[k], kms_z[k]);" in code
        # D on the LEFT of inv_eps (stepping.py:1011), in every variant.
        assert "float src_x = sx * inv_eps_Ex[idx];" in code, variant


@pytest.mark.parametrize("poles", list(range(0, 9)))
def test_the_unrolled_source_carries_exactly_the_arity_it_was_asked_for(poles):
    """The ``#if`` chain is the arity, so count what it actually admits.

    A preprocessor guard that never matched would compile a NON-DISPERSIVE kernel
    and the gate would compare two kernels that both ignored the poles — the
    purest form of the vacuous pass.
    """
    code = subject.unrolled_source(poles, poles, poles)
    assert f"#define NP0 {poles}\n" in code
    live = 0
    for component, prefix in (("x", "Px"), ("y", "Py"), ("z", "Pz")):
        macro = {"x": "NP0", "y": "NP1", "z": "NP2"}[component]
        for index in range(subject.MAX_POLES):
            guard = f"#if {macro} > {index}\n    s{component} = s{component} - {prefix}{index}[idx];"
            if guard in code:
                live += int(index < poles)
    assert live == 3 * poles


def test_the_dynamic_sources_carry_no_arity_at_all():
    """One compile covers the whole family — which IS the cost model under test."""
    for variant in subject.ALL_VARIANTS:
        if variant == "unrolled":
            continue
        first = subject.source_for(variant, (0, 0, 0))
        for triple in ((1, 1, 1), (5, 6, 8)):
            assert subject.source_for(variant, triple) == first, variant
        assert "#define NP0" not in first
        assert "int np0, int np1, int np2," in first


def test_only_the_dynamic_sources_contain_a_loop():
    assert "for (" not in subject.unrolled_source(8, 8, 8)
    for variant in subject.ALL_VARIANTS:
        if variant == "unrolled":
            continue
        code = subject.source_for(variant, (0, 0, 0))
        assert code.count("for (") == 3, f"{variant}: one loop per component"


def test_the_control_variants_differ_from_the_subject_in_source():
    """A control that is textually the subject is a control that cannot fail."""
    forward = subject.dynamic_source("dynamic")
    for control in subject.CONTROL_VARIANTS:
        assert subject.dynamic_source(control) != forward, control
    assert set(subject.CONTROL_VARIANTS) == set(subject.CONTROL_ARMED_AT)
    overlap = set(subject.SUBJECT_VARIANTS) & set(subject.CONTROL_VARIANTS)
    assert not overlap, overlap


def test_the_trip_counter_null_differs_only_in_the_store():
    with_store = subject.dynamic_source("dynamic")
    without = subject.dynamic_source("dynamic_notrips")
    assert with_store.replace("    trips[idx] = executed;\n", "") == without


# ---------------------------------------------------------------------------
# The sources can be handed to NVRTC at all
# ---------------------------------------------------------------------------

def test_every_device_string_is_pure_ascii_and_encodes():
    """The trap that killed the constitutive E kernel at its first launch.

    ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source through a bare
    ``open(..., 'w')`` (compiler.py:368), so the bytes go through the
    interpreter's LOCALE encoding — ASCII under C/POSIX, which is what a
    non-interactive shell on the validation host gets. Two em-dashes in a comment
    were enough. The scan is not sufficient on its own; the encode is performed.
    """
    for variant in subject.ALL_VARIANTS:
        for triple in ((0, 0, 0), (8, 8, 8), (1, 5, 2)):
            code = subject.source_for(variant, triple)
            offending = [(index, char) for index, char in enumerate(code)
                         if ord(char) > 127]
            assert not offending, f"{variant}: non-ASCII at {offending[:3]}"
            code.encode("ascii")


def test_a_source_file_is_hashed_as_bytes_and_a_device_string_as_ascii():
    """The two digests are not interchangeable, and confusing them is a crash.

    ``source_digest`` encodes ASCII on purpose — a device string that cannot be
    encoded cannot be compiled on a C-locale host, so the failure belongs at the
    digest and not at first launch. A source FILE has prose in it and is not so
    constrained. The probe hashed its own subject module with the ASCII digest
    and died on an em-dash in the docstring, on the device, having reached
    nothing; this pins the two apart.
    """
    assert subject.source_digest("plain ascii")
    with pytest.raises(UnicodeEncodeError):
        subject.source_digest("an em-dash — here")
    digest = subject.file_digest(str(SUBJECT_PATH))
    assert len(digest) == 64
    assert "—" in SUBJECT_PATH.read_text(), (
        "the module lost the non-ASCII prose this clause exists to survive")
    assert "SUBJECT.file_digest" in PROBE_PATH.read_text()
    assert "SUBJECT.source_digest(\n" not in PROBE_PATH.read_text()


def test_the_probe_declares_the_floors_it_enforces():
    """The runner must read its floors from the subject, not restate them."""
    text = PROBE_PATH.read_text()
    assert "SUBJECT.CONTROL_CATCH_FLOOR" in text
    assert "SUBJECT.MOVED_WORD_FLOOR_BY_CLASS" in text
    assert "SUBJECT.eligible_outputs" in text
    for constant in ("CONTROL_CATCH_FLOOR", "MOVED_WORD_FLOOR_BY_CLASS"):
        assert f"{constant} =" not in text, (
            f"{constant} is restated in the probe; two definitions is how a gate "
            f"and its artifact stop describing the same measurement")


# ---------------------------------------------------------------------------
# Byte comparison mechanics
# ---------------------------------------------------------------------------

def test_word_comparison_sees_signed_zero_and_nan():
    """``==`` on float32 is the wrong comparison and this pins that it is not used.

    ``0.0 == -0.0`` is True and ``nan == nan`` is False; neither is the question.
    The ulp figure for the signed-zero pair is 0 BY DESIGN — they are the same
    number — and the byte count is what reports the disagreement.
    """
    zeros = {"Ex": np.array([0.0], dtype=np.float32)}
    negative = {"Ex": np.array([-0.0], dtype=np.float32)}
    result = subject.compare_words(zeros, negative, ("Ex",))
    assert result["differing_words"] == 1 and result["max_ulp"] == 0
    nan = {"Ex": np.array([np.nan], dtype=np.float32)}
    assert subject.compare_words(nan, nan, ("Ex",))["identical"]


def test_the_ulp_distance_is_monotone_across_the_sign_boundary():
    """One ulp is one ulp wherever it is, or the "by how much" figure lies.

    The follow-up the reversal condition turns on is whether a divergence, if it
    happens, is a one-ulp rounding difference or a catastrophic one. A distance
    that read 2^31 for a step across zero would make that unanswerable.
    """
    tiny = np.float32(1e-45)              # smallest positive subnormal
    pairs = ((np.float32(1.0), np.nextafter(np.float32(1.0), np.float32(2.0)), 1),
             (np.float32(-1.0), np.nextafter(np.float32(-1.0), np.float32(0.0)), 1),
             (tiny, -tiny, 2))
    for left, right, expected in pairs:
        result = subject.compare_words({"Ex": np.array([left], dtype=np.float32)},
                                       {"Ex": np.array([right], dtype=np.float32)},
                                       ("Ex",))
        assert result["max_ulp"] == expected, (left, right, result)


def test_moved_words_would_fail_a_kernel_that_wrote_nothing():
    state = {name: np.ones((4, 4, 4), dtype=np.float32) for name in subject.OUTPUT_NAMES}
    assert subject.moved_words(state, state)["moved_fraction"] == 0.0
    floor = min(subject.MOVED_WORD_FLOOR_BY_CLASS.values())
    assert floor > 0.0, "a floor of zero would admit a kernel that wrote nothing"


# ---------------------------------------------------------------------------
# The RUNNER can fail — exercised against a NumPy stand-in for the device
# ---------------------------------------------------------------------------
#
# The device leg's verdict is only as good as the code that computes it. A runner
# that silently reported "identical" whatever the kernels did would look exactly
# like a runner that measured everything, and would do so after an hour of GPU
# time rather than before. So the runner is driven here against a NumPy
# emulation of the seven variants: once with the emulation honest (the verdict
# must be IDENTICAL and every armed control must be CAUGHT), once with the
# subject deliberately reordered (the verdict must flip to DIVERGENT), and once
# with every variant writing nothing (the vacuity floor must fire).

def _emulate(variant, case, state, sabotage=None):
    """One variant's arithmetic in float32 NumPy, in place on ``state``."""
    arities = case["arities"]
    order = {"x": 0, "y": 1, "z": 2}
    trips = 0
    sources = {}
    for letter, (field, source, prefix) in zip(
            "xyz", (("Ex", "Dx", "Px"), ("Ey", "Dy", "Py"), ("Ez", "Dz", "Pz"))):
        arity = arities[order[letter]]
        terms = [state[f"{prefix}{index}"] for index in range(arity)]
        effective = variant if sabotage is None else sabotage
        if effective == "dynamic_short":
            terms = terms[:-1] if terms else []
        if effective == "dynamic_reversed":
            terms = list(reversed(terms))
        accumulated = np.array(state[source], dtype=np.float32, copy=True)
        if effective == "dynamic_presummed":
            total = np.zeros(case["shape"], dtype=np.float32)
            for term in terms:
                total = (total + term).astype(np.float32)
            accumulated = (accumulated - total).astype(np.float32)
        else:
            for term in terms:
                accumulated = (accumulated - term).astype(np.float32)
        trips += len(terms)
        sources[letter] = (accumulated * state["inv_eps_" + field]).astype(np.float32)
    for letter, field in zip("xyz", ("Ex", "Ey", "Ez")):
        axis = order[letter]
        fw = "f_w_" + field
        previous = np.array(state[fw], dtype=np.float32, copy=True)
        state[fw] = sources[letter]
        value = (state[field] + case["kps_broadcast"][axis] * sources[letter]
                 ).astype(np.float32)
        state[field] = (value - case["kms_broadcast"][axis] * previous).astype(np.float32)
    state["trips"] = _HostArray(np.full(int(np.prod(case["shape"])), trips,
                                        dtype=np.int32))


class _HostArray(np.ndarray):
    """A NumPy array wearing CuPy's ``.get()``, so the probe needs no branch."""

    def __new__(cls, values):
        return np.asarray(values).view(cls)

    def get(self):
        return np.asarray(self)


def _device_state(case):
    state = {name: np.array(array, copy=True)
             for name, array in case["state"].items()}
    state["trips"] = _HostArray(np.zeros(int(np.prod(case["shape"])), dtype=np.int32))
    return state


@pytest.fixture()
def runner(monkeypatch):
    """The probe, with its three device seams replaced by NumPy.

    ``_device_state``, ``launch`` and ``harvest`` are the ONLY places the probe
    touches CuPy. Replacing exactly those three drives every line of the verdict,
    the floors, the arming and the failure accounting on a laptop.
    """
    probe = load(PROBE_PATH, "probe_cuda_dynamic_loop_arity_under_test")
    monkeypatch.setattr(probe, "SUBJECT", subject)
    monkeypatch.setattr(probe, "_device_state", _device_state)
    monkeypatch.setattr(probe, "harvest",
                        lambda device: {name: np.array(device[name], copy=True)
                                        for name in subject.OUTPUT_NAMES})
    return probe


def test_the_runner_reports_identical_and_catches_every_control_when_honest(runner, monkeypatch):
    monkeypatch.setattr(runner, "launch",
                        lambda variant, case, device: _emulate(variant, case, device))
    row = runner.run_case({"shape": (8, 8, 12), "arities": (5, 5, 5),
                           "value_class": "uniform", "seed": 77}, repeats=2)
    assert row["subject_identical"] is True
    assert row["ok"] is True, row["failures"]
    for control in subject.CONTROL_VARIANTS:
        assert row["controls"][control]["armed"] is True
        assert row["controls"][control]["caught"] is True, control
        assert row["controls"][control]["caught_fraction"] >= subject.CONTROL_CATCH_FLOOR
    assert row["counter_inert"] is True
    assert row["reference"]["unrolled"]["identical"] is True


def test_the_runner_flips_to_divergent_when_the_subject_reassociates(runner, monkeypatch):
    """THE CLAUSE THAT MAKES THE VERDICT MEAN SOMETHING.

    If the dynamic kernel summed in the opposite order, the runner must say so.
    A runner that could not distinguish that case would return IDENTICAL for a
    genuinely divergent pair, which is the one failure mode this whole gate is
    supposed to be immune to.
    """
    def launch(variant, case, device):
        _emulate(variant, case, device,
                 "dynamic_reversed" if variant == "dynamic" else None)

    monkeypatch.setattr(runner, "launch", launch)
    row = runner.run_case({"shape": (8, 8, 12), "arities": (5, 5, 5),
                           "value_class": "uniform", "seed": 77}, repeats=1)
    assert row["subject_identical"] is False
    assert row["comparisons"]["dynamic"]["differing_words"] > 0
    assert row["comparisons"]["dynamic"]["max_ulp"] > 0


def test_the_runner_fails_a_leg_whose_kernels_wrote_nothing(runner, monkeypatch):
    """Zero-init is a fixed point; two kernels that wrote nothing compare equal."""
    monkeypatch.setattr(runner, "launch", lambda variant, case, device: None)
    row = runner.run_case({"shape": (8, 8, 12), "arities": (5, 5, 5),
                           "value_class": "uniform", "seed": 77}, repeats=1)
    assert row["subject_identical"] is True      # they agree — on nothing
    assert row["ok"] is False
    assert any("vacuous" in failure for failure in row["failures"])
    assert any("ARMED" in failure and "NOT CAUGHT" in failure
               for failure in row["failures"])


def test_the_runner_fails_a_leg_whose_trip_count_is_wrong(runner, monkeypatch):
    """The loop must have run the number of times the arity says it did."""
    def launch(variant, case, device):
        _emulate(variant, case, device)
        if variant == "dynamic":
            device["trips"] = _HostArray(np.asarray(device["trips"]) + 1)

    monkeypatch.setattr(runner, "launch", launch)
    row = runner.run_case({"shape": (8, 8, 12), "arities": (5, 5, 5),
                           "value_class": "uniform", "seed": 77}, repeats=1)
    assert row["ok"] is False
    assert any("trip count" in failure for failure in row["failures"])


def test_the_runner_disarms_the_reordering_controls_at_one_pole(runner, monkeypatch):
    """At NP=1 reversed and presummed ARE the subject; a miss there is correct."""
    monkeypatch.setattr(runner, "launch",
                        lambda variant, case, device: _emulate(variant, case, device))
    row = runner.run_case({"shape": (8, 8, 12), "arities": (1, 1, 1),
                           "value_class": "uniform", "seed": 78}, repeats=1)
    assert row["controls"]["dynamic_reversed"]["armed"] is False
    assert row["controls"]["dynamic_presummed"]["armed"] is False
    assert row["controls"]["dynamic_short"]["armed"] is True
    assert row["ok"] is True, row["failures"]


# ---------------------------------------------------------------------------
# The PTX census reads structure, not vibes
# ---------------------------------------------------------------------------

def test_the_ptx_census_detects_a_back_edge():
    """The loop detector is the structural claim the trip-count leg leans on."""
    probe = load(PROBE_PATH, "probe_cuda_dynamic_loop_arity_ptx")
    looping = """
    .visible .entry k()
    {
    $L__BB0_1:
        add.f32 %f1, %f2, %f3;
        @%p1 bra $L__BB0_1;
        ret;
    }
    """
    straight = """
    .visible .entry k()
    {
        sub.f32 %f1, %f2, %f3;
        sub.f32 %f4, %f1, %f5;
        @%p1 bra $L__BB0_2;
    $L__BB0_2:
        ret;
    }
    """
    assert probe.ptx_census(looping)["backward_branches"] == 1
    assert probe.ptx_census(straight)["backward_branches"] == 0
    assert probe.ptx_census(straight)["forward_branches"] == 1
    assert probe.ptx_census(straight)["sub_f32"] == 2
    assert probe.ptx_census(looping)["fma_rn_f32"] == 0


def test_the_operand_census_reports_the_class_it_was_given():
    shape = (12, 12, 12)
    rng = np.random.default_rng(3)
    uniform = subject.operand_census(
        {"a": subject.draw(shape, rng, "uniform")})
    band = subject.operand_census(
        {"a": subject.draw(shape, rng, "subnormal_band")})
    wide = subject.operand_census(
        {"a": subject.draw(shape, rng, "wide_exponent")})
    assert uniform["subnormal"] == 0 and uniform["negative_zero"] == 0
    assert band["subnormal"] > 0 and band["negative_zero"] > 0
    assert wide["subnormal"] == 0
    assert wide["exponent_decades"] > uniform["exponent_decades"]
