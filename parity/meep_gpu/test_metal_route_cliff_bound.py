"""The bound that separates a POLICY SPLIT from an arithmetic defect, armed off-device.

WHY THIS FILE EXISTS. ``gate_dispatch_metal_route.cliff_control`` drives a released
Metal pair with the host FPU kept while the device flushes natively, and then has to
say which of two things it is looking at: the flush-vs-keep cliff the policy rung
exists to refuse, or a real arithmetic difference in a dispatched kernel. The whole
of that judgement is :func:`band_confinement`, and until this file it could only be
exercised by a ~3 hour campaign on an Apple-silicon host -- so a bound that was wrong
stayed wrong for as long as it took a campaign to notice.

THE BOUND THAT WAS WRONG, and the case that is pinned here as a regression. Until
2026-09-11 the predicate required ``max_rel_diff < 1e-30``. That number is
``max|a-b|`` over ``max|a|`` -- the disagreement divided by THE ARRAY'S OWN PEAK --
so it grades amplitude, not disagreement. On ``pml_3d_diagonal`` five arrays peak
near 1e-04 and passed while ``fields.Bz``, which carries only numerical residue in a
diagonal 3D PML and peaks at 1.97e-08, failed by 78x on the second SMALLEST
disagreement of the six. Three campaign legs were held red by it. The numbers below
are that run's own forensics, transcribed from
``dispatch_metal_route_2026-09-11_dispatch/shipped/summary.json`` ->
``control_rows[subnormal-cliff, pml_3d_diagonal].at_24_steps.differences`` -- the
COMPARATOR's per-array rows, not the ``band`` projection beside them, because that
projection did not carry ``max_abs_diff`` until this same change added it. The
``_bandbound`` re-run reproduces all twenty-four numbers exactly, which is itself
worth knowing: the split is deterministic, not sampling noise.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import gate_dispatch_metal_route as route  # noqa: E402


#: ``pml_3d_diagonal`` at 24 steps, keeping host against flushing device. Every
#: differing element sits at the denormal floor; ``fields.Bz`` is the array whose
#: peak -- not whose disagreement -- used to fail the leg.
MEASURED_CLIFF = [
    {"array": "fields.Bx", "largest_differing_magnitude": 2.678e-29,
     "max_abs_diff": 3.009e-36, "max_rel_diff": 1.301e-32,
     "elements_one_side_exactly_zero": 168},
    {"array": "fields.By", "largest_differing_magnitude": 9.457e-32,
     "max_abs_diff": 1.563e-38, "max_rel_diff": 6.751e-35,
     "elements_one_side_exactly_zero": 144},
    {"array": "fields.Bz", "largest_differing_magnitude": 6.666e-32,
     "max_abs_diff": 1.549e-36, "max_rel_diff": 7.851e-29,
     "elements_one_side_exactly_zero": 668},
    {"array": "fields.Dx", "largest_differing_magnitude": 4.568e-30,
     "max_abs_diff": 3.762e-37, "max_rel_diff": 1.122e-33,
     "elements_one_side_exactly_zero": 142},
    {"array": "fields.Dy", "largest_differing_magnitude": 1.439e-29,
     "max_abs_diff": 1.505e-36, "max_rel_diff": 4.485e-33,
     "elements_one_side_exactly_zero": 164},
    {"array": "fields.Dz", "largest_differing_magnitude": 6.421e-32,
     "max_abs_diff": 3.233e-38, "max_rel_diff": 2.048e-35,
     "elements_one_side_exactly_zero": 180},
]


#: The array the 6-of-24 truncation hid, with the numbers the host flush replay
#: reproduces exactly (``parity/meep_gpu/probe_pml_intermediate_flush.py``).
MEASURED_PML_INTERMEDIATE = {
    "array": "fields.fu_Dz", "largest_differing_magnitude": 1.922e-28,
    "max_abs_diff": 1.204e-35, "max_rel_diff": 5.994e-33,
    "elements_one_side_exactly_zero": 180,
}


def test_a_pml_intermediate_over_the_field_ceiling_is_judged_on_its_own():
    """THE ARRAY THE TRUNCATION HID, and the reason it is not a defect.

    ``fields.fu_Dz`` is the stored stage-1 output of the split-field PML recurrence,
    and its flush perturbation reaches the field only through ``fu - fu_previous``,
    where it cancels by ~3000x -- so the stored intermediate carries at 1.922e-28 what
    ``fields.Dz`` shows at 6.421e-32, with the same 180 elements exactly zero on one
    side in both. The host replay reproduces that number with no device. It is over the
    FIELD ceiling and under the INTERMEDIATE one, and the record says which judged it.
    """
    verdict = route.band_confinement([MEASURED_PML_INTERMEDIATE])
    assert verdict["confined"], verdict["per_array"]
    row = verdict["per_array"][0]
    assert row["judged_as"] == "pml_intermediate"
    assert row["value_ceiling"] == route._BAND_INTERMEDIATE_VALUE_CEILING
    # And the field ceiling really would have refused it -- the regression half.
    assert MEASURED_PML_INTERMEDIATE["largest_differing_magnitude"] > \
        route._BAND_VALUE_CEILING
    as_a_field = dict(MEASURED_PML_INTERMEDIATE, array="fields.Dz")
    refused = route.band_confinement([as_a_field])
    assert not refused["confined"]
    assert refused["per_array"][0]["over_ceiling"] == ["largest_differing_magnitude"]
    assert refused["per_array"][0]["judged_as"] == "field"


def test_the_intermediate_ceiling_still_refuses_a_field_scale_disagreement():
    """The looser ceiling is not a blank cheque: 12 orders of room, not infinite."""
    defect = {"array": "fields.fu_Dz", "largest_differing_magnitude": 1.0e-03,
              "max_abs_diff": 5.96e-11, "max_rel_diff": 5.96e-08,
              "elements_one_side_exactly_zero": 0}
    verdict = route.band_confinement([defect])
    assert not verdict["confined"]
    assert verdict["per_array"][0]["over_ceiling"] == [
        "largest_differing_magnitude", "max_abs_diff", "max_rel_diff"]


def test_the_intermediate_ceiling_is_not_fitted_to_the_measurement():
    """Two orders of margin above the measured value, and twelve below its own peak.

    A ceiling set to the measurement's last digit is a ceiling fitted to one run.
    """
    measured = MEASURED_PML_INTERMEDIATE["largest_differing_magnitude"]
    peak = (MEASURED_PML_INTERMEDIATE["max_abs_diff"]
            / MEASURED_PML_INTERMEDIATE["max_rel_diff"])
    ceiling = route._BAND_INTERMEDIATE_VALUE_CEILING
    assert ceiling > measured * 50, (ceiling, measured)
    assert ceiling < peak * 1.0e-10, (ceiling, peak)


def test_band_ceiling_names_every_intermediate_fields_declares():
    """The ``fu_`` prefix IS the structural fact, pinned against fields.py.

    A prefix test is a weak instrument unless the prefix is exhaustive. It is: the
    six ``fu_*`` slots are the only stored values that are an intermediate of a
    two-stage recurrence, and a seventh cannot appear without editing the dataclass.
    """
    import ast  # noqa: PLC0415
    import pathlib  # noqa: PLC0415

    from meep_gpu import fields as fields_module  # noqa: PLC0415

    source = pathlib.Path(fields_module.__file__).read_text(encoding="utf-8")
    declared = {node.target.id for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id.startswith("fu_")}
    assert declared == {"fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz"}, declared
    for name in sorted(declared):
        assert route._is_pml_intermediate(f"fields.{name}"), name
    # And nothing else is swept in: the fields, the PML w-arrays and the conductive
    # and BFAST auxiliaries are all judged as fields.
    for name in ("fields.Dz", "fields.Ez", "fields.f_w_Ez", "fields.f_cond_Dz",
                 "fields.Hz", "fields.Bz"):
        assert not route._is_pml_intermediate(name), name


def test_the_ceilings_are_the_values_this_file_argues_for():
    """THE BOUND'S VALUE, not only its shape.

    Every other test here probes 8-16 orders away from the ceilings, so all of them
    stay green if the three constants are loosened by eight orders -- the file would
    pin the predicate's STRUCTURE while the numbers it exists to defend drifted. These
    are the values, byte for byte, and moving one means coming back here and saying
    why in the same change.
    """
    assert route._BAND_VALUE_CEILING == 1e-28
    assert route._BAND_DIFFERENCE_CEILING == 1e-28
    assert route._BAND_RATIO_CEILING == 1e-20
    # And the predicate reports them, so an artifact records the bound it was judged
    # against rather than leaving a reader to go and read the source of the day.
    assert route._BAND_INTERMEDIATE_VALUE_CEILING == 1e-26
    ceilings = route.band_confinement(MEASURED_CLIFF)["ceilings"]
    assert ceilings == {"largest_differing_magnitude": 1e-28,
                        "largest_differing_magnitude_pml_intermediate": 1e-26,
                        "max_abs_diff": 1e-28,
                        "max_rel_diff": 1e-20}


def test_the_measured_cliff_is_confined_and_names_no_array_over_a_ceiling():
    """The run three legs were held red on. Every array, every ceiling."""
    verdict = route.band_confinement(MEASURED_CLIFF)
    assert verdict["confined"], verdict["per_array"]
    assert [row["over_ceiling"] for row in verdict["per_array"]] == [[]] * 6


def test_the_old_ratio_ceiling_would_still_refuse_the_array_it_refused():
    """NON-VACUOUS AS A REGRESSION: the pinned numbers really do fail 1e-30.

    Without this the file could pass because the transcription drifted toward
    something no bound ever rejected.
    """
    bz, = [row for row in MEASURED_CLIFF if row["array"] == "fields.Bz"]
    assert bz["max_rel_diff"] > 1e-30
    assert all(row["max_rel_diff"] < 1e-30
               for row in MEASURED_CLIFF if row["array"] != "fields.Bz")


def test_a_normal_magnitude_disagreement_is_not_confined():
    """The defect the control exists to catch: one float32 ULP on a real field.

    A field peaking at 1e-03 has a ULP near 6e-11, so a one-ULP disagreement has
    differing VALUES at 1e-03 and a difference at 1e-10 -- nine and eighteen orders
    respectively above the ceilings. Every clause refuses it, which is what makes
    ``CLIFF-OUTSIDE-THE-BAND`` a finding rather than a formality.
    """
    verdict = route.band_confinement([
        {"array": "fields.Ez", "largest_differing_magnitude": 1.0e-03,
         "max_abs_diff": 5.96e-11, "max_rel_diff": 5.96e-08,
         "elements_one_side_exactly_zero": 0}])
    assert not verdict["confined"]
    assert verdict["per_array"][0]["over_ceiling"] == [
        "largest_differing_magnitude", "max_abs_diff", "max_rel_diff"]


@pytest.mark.parametrize("peak", [1.0e-03, 1.0e-08, 1.0e-12])
def test_the_verdict_does_not_move_with_the_array_s_own_amplitude(peak):
    """THE MUTATION THE OLD BOUND FAILED, as a property rather than as one case.

    The same band-scale disagreement is presented against three peaks spanning nine
    orders. A predicate that reads the disagreement confines all three; the old one
    confined only the loudest array, which is how a quiet field component held a
    release.
    """
    difference = 1.5e-36
    verdict = route.band_confinement([
        {"array": "fields.Bz", "largest_differing_magnitude": 6.7e-32,
         "max_abs_diff": difference, "max_rel_diff": difference / peak,
         "elements_one_side_exactly_zero": 668}])
    assert verdict["confined"], (peak, verdict["per_array"])


@pytest.mark.parametrize("field,over", [
    ("largest_differing_magnitude", 1.0e-20),
    ("max_abs_diff", 1.0e-20),
    ("max_rel_diff", 1.0e-06),   # one ULP-scale relative error: 13 orders over R
])
def test_each_ceiling_is_load_bearing_on_its_own(field, over):
    """Violate exactly one number and the row is refused, naming that number.

    Three ceilings that could not each refuse alone would mean two of them were
    decoration.
    """
    row = dict(MEASURED_CLIFF[2], **{field: over})
    verdict = route.band_confinement([row])
    assert not verdict["confined"]
    assert verdict["per_array"][0]["over_ceiling"] == [field]


@pytest.mark.parametrize("missing", ["largest_differing_magnitude",
                                     "max_abs_diff", "max_rel_diff"])
def test_forensics_that_stopped_reporting_a_number_refuse_rather_than_confine(missing):
    """FAIL CLOSED. A comparator that dropped a field must not licence a release."""
    row = {key: value for key, value in MEASURED_CLIFF[2].items() if key != missing}
    verdict = route.band_confinement([row])
    assert not verdict["confined"]
    assert verdict["per_array"][0]["over_ceiling"] == [missing]


def test_where_the_ratio_clause_re_couples_to_amplitude_is_stated_not_hidden():
    """R = 1e-20 is not amplitude-free, and pretending otherwise is how the last bound
    went wrong.

    A band-scale disagreement of 1.5e-36 passes the ratio clause while the array peak
    stays above 1.5e-16 and fails below it. That floor is eight orders beneath the
    smallest peak this corpus has ever produced (``fields.Bz`` at 1.97e-08), so the
    clause is amplitude-free over every array the control actually sees -- but the
    coupling exists, and it is pinned here so a future corpus with a quieter component
    finds a red test rather than a repeat of the 1e-30 argument.
    """
    difference = 1.5e-36
    couples_below = difference / route._BAND_RATIO_CEILING
    assert 1.0e-17 < couples_below < 1.0e-15, couples_below
    smallest_peak_measured = min(row["max_abs_diff"] / row["max_rel_diff"]
                                 for row in MEASURED_CLIFF)
    assert smallest_peak_measured > couples_below * 1.0e+06, (
        f"the corpus's quietest array peaks at {smallest_peak_measured:.2e}, within "
        f"six orders of where the ratio clause re-couples ({couples_below:.2e}) -- "
        f"the margin this test exists to keep has gone")


def test_the_ratio_ceiling_sits_below_float32_resolution():
    """R is not a fitted number, and this is what makes that checkable.

    float32 eps is 1.19e-07, so a one-ULP arithmetic error anywhere in an array shows
    up at ``max_rel_diff`` of about 1e-07. The ceiling has to be far below that or it
    admits real defects, and far above the worst value this control has measured
    (7.85e-29) or it is fitted to one run. Both margins are asserted, so retuning the
    constant in either direction turns this red.
    """
    float32_eps = 1.1920929e-07
    worst_measured = max(row["max_rel_diff"] for row in MEASURED_CLIFF)
    assert route._BAND_RATIO_CEILING < float32_eps / 1.0e+10, (
        "the ratio ceiling is within ten orders of float32 eps; a one-ULP arithmetic "
        "error would pass it")
    assert route._BAND_RATIO_CEILING > worst_measured * 1.0e+05, (
        "the ratio ceiling is within five orders of the worst value ever measured, "
        "which is a bound fitted to a run rather than to float32")


def test_a_seventh_array_is_judged_and_not_truncated_away():
    """THE DEFECT THAT MATTERED MOST, armed as a regression.

    Until 2026-09-11 ``cliff_control`` handed this predicate
    ``verdict["differences"][:6]``. Six of ``pml_3d_diagonal``'s twenty-four differing
    arrays decided the case, and six of twelve decided ``pml_2d``,
    ``magnetic_seam_2d`` and ``folded_2d`` -- all four then read
    CLIFF-CONFINED-TO-THE-BAND. A truncation for readability sat upstream of a
    verdict. Here a normal-magnitude disagreement is placed SEVENTH, where the old
    slice would have dropped it.
    """
    defect = {"array": "fields.Hz", "largest_differing_magnitude": 1.0e-03,
              "max_abs_diff": 5.96e-11, "max_rel_diff": 5.96e-08,
              "elements_one_side_exactly_zero": 0}
    assert route.band_confinement(MEASURED_CLIFF)["confined"], "precondition"
    verdict = route.band_confinement([*MEASURED_CLIFF, defect])
    assert not verdict["confined"], (
        "a seventh array carrying a normal-magnitude disagreement was not judged")
    assert verdict["per_array"][6]["array"] == "fields.Hz"
    assert verdict["per_array"][6]["over_ceiling"] == [
        "largest_differing_magnitude", "max_abs_diff", "max_rel_diff"]


def test_the_control_requires_a_flush_signature_and_not_only_confinement():
    """THE CALLER'S HALF, which no test read until the audit found it unguarded.

    ``band_confinement`` answers "is the disagreement at the band's scale". It does
    NOT answer "is this a flush" -- that is ``elements_one_side_exactly_zero``, and
    ``cliff_control`` requires BOTH before it writes CLIFF-CONFINED-TO-THE-BAND. A
    caller that dropped the second half would turn every tiny disagreement into a
    policy split, so the conjunction is pinned here by AST rather than by prose.
    """
    import ast  # noqa: PLC0415
    import pathlib  # noqa: PLC0415

    source = pathlib.Path(route.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    control = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "cliff_control")
    body = ast.get_source_segment(source, control) or ""
    assert "one_side_zero = any(" in body, (
        "cliff_control no longer derives one_side_zero from the forensics")
    assert '"CLIFF-CONFINED-TO-THE-BAND" if (confined and one_side_zero)' in body, (
        "the confinement verdict no longer requires the flush signature beside the "
        "band bound; band_confinement alone cannot tell a flush from a tiny defect")
    # And the short-list guard the truncation defect earned.
    assert "judged_every_differing_array" in body, (
        "cliff_control no longer checks that every differing array reached the "
        "predicate, which is how differences[:6] decided four cases")


def test_no_differences_at_all_is_not_a_confinement():
    """An empty forensics list means nothing diverged, which the caller handles as
    ``NO-CLIFF`` -- it must never arrive here as a confined cliff."""
    assert not route.band_confinement([])["confined"]


#: ``special_kz_2d`` at 24 steps, the rows the 2026-09-12 campaign actually refused
#: on. ``fields.Dy`` is over the FIELD ceiling at 1.894e-28 while its own
#: ``fields.fu_Dy`` is under the INTERMEDIATE ceiling at 1.896e-28 — the same
#: number, one step apart. Reproduced on the host with no device by
#: ``probe_pml_intermediate_flush.py`` (``FU_DZ_CASE=special_kz_2d``), which flushes
#: only stage-1 outputs and lands fu_Dy 1.896e-28 / Dy 1.894e-28 / max_abs 1.204e-35.
MEASURED_DOWNSTREAM_CLIFF = [
    {"array": "fields.fu_Dy", "largest_differing_magnitude": 1.896e-28,
     "max_abs_diff": 1.204e-35, "max_rel_diff": 2.306e-31},
    {"array": "fields.Dy", "largest_differing_magnitude": 1.894e-28,
     "max_abs_diff": 1.204e-35, "max_rel_diff": 2.306e-31},
]


def test_a_field_inherits_the_ceiling_of_the_intermediate_that_wrote_it():
    """The 2026-09-12 refusal, and why it was the BOUND and not the kernel.

    ``_apply_pml_update``'s stage 2 is ``field += fu; field -= fu_previous``, so the
    stage-1 intermediate's differing magnitude passes straight into the field. A
    bound that grants ``fu_Dy`` 1e-26 and holds ``Dy`` to 1e-28 refuses its own
    consumer for every intermediate flush between the two, which is what happened.
    """
    out = route.band_confinement(MEASURED_DOWNSTREAM_CLIFF)
    assert out["confined"] is True
    rows = {row["array"]: row for row in out["per_array"]}
    assert rows["fields.fu_Dy"]["judged_as"] == "pml_intermediate"
    assert rows["fields.Dy"]["judged_as"] == (
        "field_downstream_of_its_pml_intermediate")
    # The artifact must NAME the lender, not leave it to be inferred.
    assert rows["fields.Dy"]["ceiling_inherited_from"] == "fields.fu_Dy"
    assert rows["fields.fu_Dy"]["ceiling_inherited_from"] is None


def test_a_field_differing_ABOVE_its_own_intermediate_is_still_refused():
    """THE CASE THE INHERITANCE MUST NOT SWALLOW, and the reason it is bounded.

    The mechanism explains a field carrying AT MOST what its intermediate carried.
    A field carrying MORE is not downstream pass-through — it is arithmetic the
    recurrence does not account for — so the ceiling must not be lent to it.
    """
    entries = [
        {"array": "fields.fu_Dy", "largest_differing_magnitude": 1.9e-28,
         "max_abs_diff": 1.2e-35, "max_rel_diff": 2.3e-31},
        # One ULP-ish ABOVE its producer, and over the field ceiling.
        {"array": "fields.Dy", "largest_differing_magnitude": 1.0e-27,
         "max_abs_diff": 1.2e-35, "max_rel_diff": 2.3e-31},
    ]
    out = route.band_confinement(entries)
    assert out["confined"] is False
    rows = {row["array"]: row for row in out["per_array"]}
    assert rows["fields.Dy"]["judged_as"] == "field"
    assert rows["fields.Dy"]["ceiling_inherited_from"] is None
    assert rows["fields.Dy"]["over_ceiling"] == ["largest_differing_magnitude"]


def test_a_field_with_no_intermediate_in_the_run_keeps_the_field_ceiling():
    """No producer among the differing arrays means no ceiling to inherit.

    This is the ``E``/``H`` case as well as the general one: their companions are
    ``f_w_`` arrays, whose relationship to the field has not been measured, so the
    clause claims nothing about them.
    """
    out = route.band_confinement([
        {"array": "fields.Dy", "largest_differing_magnitude": 1.894e-28,
         "max_abs_diff": 1.2e-35, "max_rel_diff": 2.3e-31}])
    assert out["confined"] is False
    assert out["per_array"][0]["judged_as"] == "field"

    ey = route.band_confinement([
        {"array": "fields.f_w_Ey", "largest_differing_magnitude": 4.0e-28,
         "max_abs_diff": 1.2e-35, "max_rel_diff": 2.3e-31},
        {"array": "fields.Ey", "largest_differing_magnitude": 4.0e-28,
         "max_abs_diff": 1.2e-35, "max_rel_diff": 2.3e-31}])
    assert ey["confined"] is False
    assert {row["judged_as"] for row in ey["per_array"]} == {"field"}


def test_the_tight_guards_are_not_inherited_only_band_membership_is():
    """Inheritance moves the MAGNITUDE ceiling and nothing else.

    ``max_abs_diff`` and ``max_rel_diff`` are the guards on the actual disagreement
    rather than on where in the number line it sits, and a downstream field gets no
    relief on either. Without this, the clause would be a way to widen the whole
    bound by writing one intermediate into the row list.
    """
    over_difference = route.band_confinement([
        {"array": "fields.fu_Dy", "largest_differing_magnitude": 1.9e-28,
         "max_abs_diff": 1.2e-35, "max_rel_diff": 2.3e-31},
        {"array": "fields.Dy", "largest_differing_magnitude": 1.894e-28,
         "max_abs_diff": 1.0e-20, "max_rel_diff": 2.3e-31}])
    assert over_difference["confined"] is False
    rows = {row["array"]: row for row in over_difference["per_array"]}
    assert rows["fields.Dy"]["over_ceiling"] == ["max_abs_diff"]
    # ...and it still inherited the magnitude ceiling; that is not what refused it.
    assert rows["fields.Dy"]["judged_as"] == (
        "field_downstream_of_its_pml_intermediate")
