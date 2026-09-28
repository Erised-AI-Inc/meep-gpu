"""The IMAGE CLOSURE is REQUIRED on a folded seam, and these cases can show it.

WHAT THIS FILE IS FOR. ``deposit_repair.fill_image_rules`` / ``repair_cells``
(deposit_repair.py:217-258) extend a deposit repair's saved and restored cell set to the
cells the driver's post-injection ``fill_symmetry_bc_*`` / ``fill_folded_far_ghosts_*``
image the deposit into. That closure is the whole reason
``CARRIES_DEPOSIT_REPAIR`` could flip on the two FOLDED Triton fused families, and until
this file nothing on the merge bar executed it: ``test_deposit_repair.py`` pins the
module's own contracts, and the device gates that byte-compare a fused launch ran their
sources at the cell centre, which lands on a stored row NO fill reads from.

WHAT A CASE MUST HAVE TO SHOW ANYTHING, and it is the point of the whole file: the
deposit has to land ON a row a fill reads from, in the INJECTED component's own
constitutive target. A source at the centre of a folded axis does not — its
``repair_cells`` returns the deposit and nothing else, so a point-only repair is exact
there BY CONSTRUCTION and a byte comparison against the array path passes whether the
closure exists or not. :func:`test_the_centre_placement_is_vacuous_for_the_closure` pins
that vacuity as a measured fact rather than leaving it as an argument.

THE ARMED CONTROL IS THE TEST. Each case below runs three legs of complete driver steps
against the array path, compared as uint32:

    A   the shipped ``repair_cells``    -> identical
    B   a POINT-ONLY ``repair_cells``   -> MUST DIVERGE
    C   no repair at all                -> MUST DIVERGE

If leg B did not diverge the case would be measuring nothing, so both directions are
asserted on every case. The fused pair is HOST-EMULATED here (the shipped ``stepping``
passes the product's ``REPLACES`` list names, run against the same pre-injection field one
launch consumes) because the merge bar has no CUDA; the device leg lives in
``parity/meep_gpu/probe_triton_folded_deposit_closure.py`` and is what licenses the
kernel. What this file pins is the ARCHITECTURE and the ARMING — that these cases can
tell a working closure from a missing one at all.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

PARITY = pathlib.Path(__file__).resolve().parent.parent / "parity" / "meep_gpu"
PROBE = PARITY / "probe_triton_folded_deposit_closure.py"


def _probe():
    """The gate's own case table and legs, loaded rather than re-implemented.

    A test that restated the case construction would drift from the gate it is supposed
    to keep honest, and this project has measured that costing 86 of 87 tests their bite.
    """
    spec = importlib.util.spec_from_file_location(
        "folded_deposit_closure_probe_under_test", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: One near-fill case, one FAR-ghost case, one two-fold corner and one magnetic case —
#: the four distinct shapes of the closure, kept to four so the file stays a merge-bar
#: test. The gate runs all twelve on a device.
EXERCISED = ("near_metallic_Ez_row2", "far_periodic_Ey_row18",
             "twofold_metallic_Ez_corner", "near_metallic_Hy_row2")


def _case(probe, name):
    for entry in probe.CASES:
        if entry[0] == name:
            return entry
    raise AssertionError(f"the gate no longer declares a case named {name!r}")


def test_the_gate_still_declares_every_shape_this_file_exercises():
    """A renamed or dropped case must fail HERE, not silently reduce the coverage."""
    probe = _probe()
    declared = {entry[0] for entry in probe.CASES}
    missing = sorted(set(EXERCISED) - declared)
    assert not missing, f"the gate dropped {missing}; this file now exercises less"
    assert probe.ON_PLANE_CONTROL_CASE in declared


@pytest.mark.parametrize("name", EXERCISED)
def test_the_closure_is_required_and_a_point_only_repair_diverges(name):
    probe = _probe()
    case = _case(probe, name)
    _n, cell, boundaries, mirrors, declaration, seam, _steps, _why = case

    census = probe.arming_census(
        probe.build_driver(cell, boundaries, mirrors, declaration, None),
        seam, declaration["component"])
    assert census["closure_extra_cells"] > 0, (
        f"{name}: repair_cells adds no cell beyond the deposit, so a point-only repair "
        f"is exact here by construction and this case can prove nothing. Rules: "
        f"{census['fill_image_rules']}, deposit rows: {census['deposit_rows']}")

    reference = probe.reference_states(case, None)
    legs = {leg: probe.run_one_leg(case, leg, False, None) for leg in ("A", "B", "C")}

    def diverging_words(leg):
        got = legs[leg]["states"]
        assert len(got) == len(reference)
        return sum(int(np.count_nonzero(reference[-1][key] != got[-1][key]))
                   for key in sorted(set(reference[-1]) & set(got[-1])))

    assert probe.first_divergence(reference[-1], legs["A"]["states"][-1]) is None, (
        f"{name}: the fused bracket WITH the closure diverged from the array path: "
        f"{probe.first_divergence(reference[-1], legs['A']['states'][-1])}")
    for step, (ref, cand) in enumerate(zip(reference, legs["A"]["states"]), start=1):
        assert probe.first_divergence(ref, cand) is None, (
            f"{name}: leg A diverged at step {step}")

    # THE TWO ARMED CONTROLS. Without these the assertion above is a measurement that
    # cannot fail: an array path compared against itself is identical for free.
    assert diverging_words("B") > 0, (
        f"{name}: withholding the IMAGE CLOSURE changed nothing, so this case cannot "
        f"tell a working closure from a point-only repair and its leg-A pass is "
        f"vacuous. Arming census said {census['closure_extra_cells']} extra cells.")
    assert diverging_words("C") > 0, (
        f"{name}: withholding the repair ENTIRELY changed nothing, so this case is not "
        f"measuring the deposit repair at all")


def test_the_centre_placement_is_vacuous_for_the_closure():
    """The placement the 2026-08-30 device evidence used, measured rather than argued.

    A source at the centre of a folded axis lands on stored row 1. The near fill reads
    row ``stepping.MIRROR_SOURCE_INDEX`` (2) and the far ghost reads the reflect row, so
    row 1 is imaged by neither: ``repair_cells`` returns the deposit and nothing more,
    the closure and a point-only repair are the SAME SET, and leg B cannot diverge. The
    case still needs the repair — leg C must diverge — which is exactly why a run that
    only asked "does it match?" reported a pass and licensed nothing about the closure.
    """
    probe = _probe()
    case = _case(probe, probe.ON_PLANE_CONTROL_CASE)
    _n, cell, boundaries, mirrors, declaration, seam, _steps, _why = case

    census = probe.arming_census(
        probe.build_driver(cell, boundaries, mirrors, declaration, None),
        seam, declaration["component"])
    assert census["closure_extra_cells"] == 0, (
        "the centre placement now reaches an imaged row; if that is deliberate this "
        "control has to be re-chosen, because a control that is armed proves nothing "
        "about what an unarmed case measures")

    reference = probe.reference_states(case, None)
    legs = {leg: probe.run_one_leg(case, leg, False, None) for leg in ("A", "B", "C")}
    for leg in ("A", "B"):
        assert probe.first_divergence(reference[-1], legs[leg]["states"][-1]) is None, (
            f"leg {leg} diverged on the centre placement, where the closure adds no "
            f"cell: legs A and B run the SAME set here and must agree")
    assert probe.first_divergence(reference[-1], legs["C"]["states"][-1]) is not None, (
        "even the centre placement needs the POINT repair; if leg C matches here the "
        "harness is not withholding anything")


def test_the_complex_magnetic_pair_cannot_reach_the_closure_at_all():
    """The third product that flipped on 2026-08-30, and the honest limit of the flip.

    ``complex_fused_magnetic_pair`` refuses ``grid.has_symmetry()`` and every
    ``is_mirrored`` axis BY NAME, so no configuration it admits runs a post-injection
    fill. On the grids it does admit ``fill_image_rules`` is empty for all three magnetic
    targets, which makes the closure and a point-only repair the same set: its
    ``CARRIES_DEPOSIT_REPAIR`` is licensed by the POINT repair alone and no gate can show
    it needing the closure. Pinned so the limit is a fact in the suite rather than a
    sentence in an artifact.
    """
    probe = _probe()
    leg = probe.unfolded_product_leg()
    assert leg.get("passed"), leg
    assert leg["folded_case_covered"] is False
    assert leg["symmetry_refusals"], (
        "the predicate no longer refuses a fold by name; if this product ever admits "
        "one, its flip needs the closure and needs a gate that exercises it")
    assert leg["closure_is_empty_where_admitted"] is True


def test_the_prior_device_evidence_did_not_exercise_the_closure():
    """What ``probe_triton_symmetry_composition`` actually measured, executed.

    Kept as a test rather than only as prose because it is the reason this file exists,
    and because it is falsifiable: if that probe ever plans with fusion on and places a
    source on an imaged row, this goes red and the duplication can be retired.
    """
    probe = _probe()
    leg = probe.prior_evidence_leg()
    assert leg.get("passed"), leg
    assert leg["cases_armed_for_closure"] == 0, (
        f"{leg['cases_armed_for_closure']} of that probe's cases now place a deposit on "
        f"a row a fill reads from: {leg['finding']}")
    assert leg["run_case_fuse_flag"] == "fuse=False"
