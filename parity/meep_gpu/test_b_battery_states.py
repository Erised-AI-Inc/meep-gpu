"""``b_battery_states``: the B battery's states hold what each one is for, host-side.

Each constructor exists because a random state cannot reach what it builds, so each
clause here is the property a gate leg would silently lose: a zero word in a
"full-volume" state, a planted case whose two forms store the same word, a lattice
missing a signed-zero pair, a coverage check that passes a configuration with an
empty form.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import b_battery_states as battery  # noqa: E402
import no_pml_ref  # noqa: E402

SHAPE = (9, 8, 7)
NAMES = ("Ex", "Ey", "Ez", "Bx", "By", "Bz", "Hx", "Hy", "Hz", "Dx", "Dy", "Dz")


def test_a_full_volume_state_has_no_zero_no_subnormal_and_both_signs():
    state = battery.full_volume_state(NAMES, SHAPE, np.random.default_rng(1))
    for name, array in state.items():
        assert array.dtype == np.float32 and array.shape == SHAPE, name
        magnitude = np.abs(array)
        assert np.all(magnitude >= np.finfo(np.float32).tiny), name
        assert np.all(np.isfinite(array)), name
        assert (array > 0).any() and (array < 0).any(), name


def test_a_full_volume_state_is_the_same_words_for_the_same_seed():
    one = battery.full_volume_state(NAMES, SHAPE, np.random.default_rng(7))
    two = battery.full_volume_state(NAMES, SHAPE, np.random.default_rng(7))
    assert all(np.array_equal(battery.words(one[n]), battery.words(two[n])) for n in NAMES)


def test_magnitudes_that_reach_zero_or_the_subnormal_band_are_refused():
    with pytest.raises(ValueError):
        battery.full_volume_state(NAMES, SHAPE, np.random.default_rng(1), low=0.0)
    with pytest.raises(ValueError):
        battery.full_volume_state(NAMES, SHAPE, np.random.default_rng(1), low=1e-40)


@pytest.mark.parametrize("sub_step,terms,shift", [("step_B", no_pml_ref.B_TERMS, +1),
                                                  ("step_D", no_pml_ref.D_TERMS, -1)])
def test_the_planted_curl_case_separates_the_plain_form_from_the_collapse(
        sub_step, terms, shift):
    """MEEP's plain ``f − curl`` stores −0 and the p := 0 recurrence +0, on every term."""
    boundaries = ("periodic", "periodic", "periodic")
    for term in terms:
        state = battery.full_volume_state(NAMES, SHAPE, np.random.default_rng(3))
        record = battery.plant_curl_negative_zero(state, term, (4, 4, 3), shift,
                                                  boundaries, dtdx=0.35)
        assert record["expected_word"] == battery.NEGATIVE_ZERO, (sub_step, term)
        assert record["collapse_word"] == battery.POSITIVE_ZERO, (sub_step, term)
        # And the reference transcription agrees: no_pml_ref's plain step leaves -0.
        sources = {n: state[n] for n in (term[1], term[3])}
        others = [t for t in terms if t[0] != term[0]]
        for other in others:
            sources.setdefault(other[1], state[other[1]])
            sources.setdefault(other[3], state[other[3]])
        targets = {t[0]: state[t[0]].copy() for t in terms}
        no_pml_ref.reference_plain_step(np, sources, targets, np.float32(0.35),
                                        sub_step, boundaries, "kernel_order")
        assert int(battery.words(targets[term[0]])[4, 4, 3]) == battery.NEGATIVE_ZERO


def test_the_no_pml_reference_defaults_to_meeps_grouping():
    """The reference a caller gets by default is MEEP's grouping, and no other tree.

    ``kernel_order`` (C's left-to-right) is the default and ``pairwise_order`` (the
    package's tree until 0.10) differs from it at a Courant factor that is not a power
    of two; the name the pre-alignment tree carried before 0.10 is refused.
    """
    state = battery.full_volume_state(NAMES, SHAPE, np.random.default_rng(11))
    sources = {n: state[n] for n in ("Ex", "Ey", "Ez")}

    def stepped(*grouping):
        targets = {n: state[n].copy() for n in ("Bx", "By", "Bz")}
        no_pml_ref.reference_plain_step(np, sources, targets, np.float32(0.35), "step_B",
                                        ("periodic",) * 3, *grouping)
        return np.concatenate([battery.words(targets[n]).ravel()
                               for n in ("Bx", "By", "Bz")])

    default, meep, pairwise = stepped(), stepped("kernel_order"), stepped("pairwise_order")
    assert np.array_equal(default, meep)
    assert int(np.count_nonzero(default != pairwise)) > 0
    assert sorted(no_pml_ref.CURL_GROUPINGS) == ["kernel_order", "pairwise_order"]
    with pytest.raises(ValueError, match="unknown grouping"):
        stepped("array_order")


def test_the_planted_curl_case_is_refused_off_a_plain_cell_and_on_a_wall():
    state = battery.full_volume_state(NAMES, SHAPE, np.random.default_rng(3))
    plain = np.ones(SHAPE, dtype=bool)
    plain[4, 4, 3] = False
    term = no_pml_ref.B_TERMS[0]
    with pytest.raises(ValueError, match="not a plain cell"):
        battery.plant_curl_negative_zero(state, term, (4, 4, 3), +1,
                                         ("periodic",) * 3, plain=plain)
    with pytest.raises(ValueError, match="metallic face"):
        battery.plant_curl_negative_zero(state, term, (0, 4, 3), +1,
                                         ("metallic", "periodic", "periodic"))
    # A wrapping partner on a periodic face is an owned word, not a ghost.
    record = battery.plant_curl_negative_zero(state, term, (4, SHAPE[1] - 1, 3), +1,
                                              ("periodic",) * 3)
    assert record["partners"][term[1]][term[2]] == 0


def test_the_planted_constitutive_case_separates_the_plain_form_from_the_collapse():
    state = battery.full_volume_state(NAMES, SHAPE, np.random.default_rng(5))
    inverse = np.random.default_rng(6).uniform(0.2, 0.9, SHAPE).astype(np.float32)
    for source, scale in (("Bx", None), ("Dy", inverse)):
        record = battery.plant_constitutive_negative_zero(state, source, (2, 3, 4),
                                                          inverse=scale)
        assert record["expected_word"] == battery.NEGATIVE_ZERO, source
        assert record["collapse_word"] == battery.POSITIVE_ZERO, source
    with pytest.raises(ValueError, match="not positive"):
        battery.plant_constitutive_negative_zero(
            state, "Dz", (1, 1, 1), inverse=np.zeros(SHAPE, dtype=np.float32))


def test_the_lattice_holds_only_special_words_and_every_pair_on_every_axis():
    state = battery.signed_zero_lattice(("Ex", "Hz"), (12, 11, 10),
                                        np.random.default_rng(11))
    for name, array in state.items():
        assert set(np.unique(battery.words(array)).tolist()) == set(battery.SPECIAL_WORDS)
        for axis in range(3):
            assert battery.pair_coverage(array, axis) == 36, (name, axis)


def test_a_lattice_too_small_to_cover_every_pair_is_refused_by_name():
    with pytest.raises(ValueError, match="ordered special-word pairs"):
        battery.signed_zero_lattice(("Ex",), (3, 3, 3), np.random.default_rng(11))


def _forms(shape, inside):
    recurrence = np.zeros(shape, dtype=bool)
    recurrence[inside] = True
    return {"Bx": {"recurrence": recurrence, "dsig_or_plain": ~recurrence}}


def _membership(n=(9, 8, 7)):
    lines = {}
    for axis in range(3):
        line = np.zeros(n[axis], dtype=bool)
        line[:2] = True
        for lattice in battery.LATTICES:
            lines[(axis, lattice)] = line
    return lines


def test_a_covered_configuration_passes_and_reports_its_counts():
    forms = _forms(SHAPE, (slice(0, 2), slice(None), slice(None)))
    counts = battery.require_form_coverage(forms, _membership(), (True, False, False))
    assert counts == {"Bx": {"recurrence": 2 * 8 * 7, "dsig_or_plain": 7 * 8 * 7}}


def test_every_coverage_gap_is_named():
    empty = _forms(SHAPE, (slice(0, 0), slice(None), slice(None)))
    lines = _membership()
    lines[(1, "half")] = np.ones(8, dtype=bool)
    lines[(2, "integer")] = np.zeros(7, dtype=bool)
    del lines[(0, "half")]
    gaps = battery.coverage_gaps(empty, lines, (False, False, False))
    assert gaps == ["Bx:recurrence has no cell",
                    "axis 0 half: no membership line given",
                    "axis 1 half: no cell outside every run",
                    "axis 2 integer: no cell inside a run",
                    "no wrapping axis"]
    with pytest.raises(ValueError, match="form coverage incomplete"):
        battery.require_form_coverage(empty, lines, (False, False, False))


def test_masks_that_overlap_or_leave_a_cell_formless_are_refused():
    forms = _forms(SHAPE, (slice(0, 2), slice(None), slice(None)))
    forms["Bx"]["dsig_or_plain"] = np.ones(SHAPE, dtype=bool)
    with pytest.raises(ValueError, match="more than one form"):
        battery.form_counts(forms)
    forms["Bx"]["dsig_or_plain"] = np.zeros(SHAPE, dtype=bool)
    with pytest.raises(ValueError, match="take no form"):
        battery.form_counts(forms)


def test_an_invariant_axis_is_not_judged():
    shape = (9, 1, 7)
    forms = _forms(shape, (slice(0, 2), slice(None), slice(None)))
    lines = _membership(shape)
    lines[(1, "integer")] = np.zeros(1, dtype=bool)
    lines[(1, "half")] = np.ones(1, dtype=bool)
    assert battery.coverage_gaps(forms, lines, (True, False, False)) == []
