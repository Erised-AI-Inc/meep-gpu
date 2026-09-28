"""Which kernel tables the dispatch seam can actually reach, measured off the tree.

THE QUESTION THIS FILE EXISTS FOR was the boards' worst failure mode before it was
answered: every board published ``served_in_dispatch`` from its own release table
while the dispatcher imported ONE composer, so two of the three numbers described
products ``plan_fast_path`` could not select at all. ``dispatch_reachability``
answers it from ``fastpath.py``'s parse tree instead of from a constant, and this
file pins the answer so a table that stops being reachable is a red test rather than
a silently unchanged headline.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import dispatch_reachability as reachability  # noqa: E402
import gate_dispatch_fused_route as route_gate  # noqa: E402

from meep_gpu import fastpath, fastpath_cuda  # noqa: E402


@pytest.mark.parametrize("backend,package", [
    ("triton", "triton_kernels"),
    ("cuda", "cuda_kernels"),
    ("metal", "metal_kernels"),
])
def test_every_wired_table_reaches_the_dispatch_seam(backend, package):
    """All three, and each through the route its own design states.

    Triton is imported at the composer rung; the hand-CUDA table through the one
    function-local ``from .cuda_kernels import arms`` inside ``_decide``; the Metal
    table one hop further out, through the ``metal_dispatch`` sibling — which is why
    the walk follows ``meep_gpu`` siblings that CALL a ``plan_step`` rather than
    reading ``fastpath.py`` alone.
    """
    verdict = reachability.backend_reaches_the_dispatch_seam(backend)
    assert verdict["kernel_package"] == package
    assert verdict["reaches_the_seam"] is True, verdict["why"]
    assert package in verdict["packages_fastpath_imports"]


def test_the_cuda_join_is_derived_from_the_composer_and_is_total():
    """Board product -> namespaced label, read off ``fused_pairs.py`` and checked.

    The row and the builder are compared inside
    ``cuda_product_arm_labels``; what is asserted here is the JOIN's own totality —
    every released arm is attributable to a product, every joined label is one the
    composer writes, and the two maps are disjoint.
    """
    report = reachability.assert_the_label_map_covers_the_release("cuda")
    assert report["backend"] == "cuda"
    joined = reachability.product_arm_labels("cuda")
    refused = reachability.certified_but_not_installed("cuda")
    assert not set(joined) & set(refused), "a product cannot be both"
    assert len(joined) + len(refused) == 38
    assert set(fastpath_cuda.CUDA_RELEASED_FUSED_ARMS) <= set(joined.values())
    for label in joined.values():
        assert label.startswith(fastpath_cuda.CUDA_LABEL_PREFIX), label
        assert fastpath.arm_is_fused(label), label


def test_a_cuda_dispatch_number_is_published_with_its_condition():
    """Under the shipped precedence Triton holds every contested slot.

    So every CUDA ``served_in_dispatch`` is a BY-PREFERENCE number, and publishing
    it without that clause in the same sentence would read as a default. The
    by-default count is carried beside it rather than left to be inferred.
    """
    instances = [("row", "cuda_fused_magnetic_pair",
                  {"pml": True, "shape": [200, 120, 1]})]
    report = reachability.served_in_dispatch("cuda", instances)
    assert report["counted_under"] == "by_preference"
    assert fastpath.BACKEND_PREFERENCE_SWITCH in report["requires"]
    assert report["by_default_precedence"]["served_in_dispatch"] == 0
    assert report["buckets_total_the_served_count"]["agrees"] is True


def test_the_yield_clause_the_board_quotes_is_the_shipped_one():
    assert (reachability.fastpath_cuda_yield()
            is fastpath_cuda.YIELD_PENDING_PRIMARY_SLOTS)


def _drive_cuda() -> dict:
    """``gate_dispatch_fused_route.DRIVE_CUDA``, read off the SOURCE.

    Off the parse tree rather than by import, for the same reason
    ``dispatch_reachability`` walks ``fastpath.py``'s: the gate module pulls a
    device-shaped world in at import and this question is about a literal.
    """
    import ast  # noqa: PLC0415
    import re  # noqa: PLC0415

    source = (pathlib.Path(__file__).resolve().parent
              / "gate_dispatch_fused_route.py").read_text(encoding="utf-8")
    opening = re.search(r"^DRIVE_CUDA[^=]*=\s*(\{)", source, re.M)
    assert opening, "gate_dispatch_fused_route.py declares no DRIVE_CUDA"
    start = opening.start(1)
    depth = 0
    for index, character in enumerate(source[start:], start):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(source[start:index + 1])
    raise AssertionError("DRIVE_CUDA is not a closed literal")


def test_the_release_table_and_the_gate_drive_the_same_arms_on_the_same_cases():
    """THE ONE DISAGREEMENT THAT IS ONLY DISCOVERED ON DEVICE, AND LATE.

    ``recut_driver_dispatch_record.py --backend cuda`` refuses to cut the record
    unless EVERY ``(arm, case, leg)`` row of ``CUDA_RELEASED_FUSED_ARMS`` appears in
    that leg's ``arms_driven``. A release row the gate does not drive, or a driven
    arm with no release row, therefore costs the WHOLE campaign — roughly 1.5 GPU-
    hours and a re-stage — and nothing off-device said so, because the two tables
    live in different files and neither imports the other.

    This is that check, run at the merge bar on three literals. It is not a
    substitute for the recut (which reads what the run actually drove, not what the
    table asked for); it is the tripwire that stops a typo reaching the queue.
    """
    drive = _drive_cuda()
    driven_cases: dict = {}
    driven_legs: dict = {}
    for case, spec in drive.items():
        for arm in spec["arms"]:
            driven_cases.setdefault(arm, set()).add(case)
            driven_legs.setdefault(arm, set()).add(spec["leg"])

    released = {arm: set(cases)
                for arm, cases in fastpath_cuda.CUDA_RELEASED_FUSED_ARMS.items()}
    assert set(released) == set(driven_cases), (
        "released but never driven: "
        f"{sorted(set(released) - set(driven_cases))}; driven but not released: "
        f"{sorted(set(driven_cases) - set(released))}")
    mismatched = {arm: (sorted(cases), sorted(driven_cases[arm]))
                  for arm, cases in released.items()
                  if cases != driven_cases[arm]}
    assert not mismatched, f"release cases != driven cases: {mismatched}"

    # AND THE LEG, which is the half a case list cannot carry: a complex-storage arm
    # dispatches on the expansion-probe leg and on NO other, so a release row that
    # claimed a shipped leg for one would be un-cuttable even with its cases right.
    for arm, legs in fastpath_cuda.CUDA_RELEASED_ARM_LEGS.items():
        assert driven_legs[arm] <= set(legs), (
            f"{arm} is driven on {sorted(driven_legs[arm])} but its release row "
            f"declares {sorted(legs)}")


def test_all_three_arbitration_verdicts_survive_into_the_summary():
    """The CUDA campaign measures three arbitration answers per case; keep three.

    THE FAILURE THIS PINS was silent and total. The summary keyed arbitration by
    CASE and stored one verdict, so the three controls overwrote each other and the
    LAST one -- ``arbitration-preference-rejected`` -- was the only survivor. The
    two verdicts the published by-preference number is conditioned on
    (``INCUMBENT-YIELDED``, and ``ARBITRATION-DEFAULT-HELD`` from the default-
    precedence leg) never reached the artifact, so ``recut_driver_dispatch_record``
    would refuse a campaign that had measured both of them perfectly, and the
    reason it gave would point at the measurement rather than at the serialization.
    """
    controls = [
        {"control": "arbitration-default-precedence", "case": "pml_2d",
         "verdict": "ARBITRATION-DEFAULT-HELD"},
        {"control": "arbitration-cuda-preferred", "case": "pml_2d",
         "verdict": "INCUMBENT-YIELDED"},
        {"control": "arbitration-preference-rejected", "case": "pml_2d",
         "verdict": "PREFERENCE-REFUSED-BY-NAME"},
        # A NON-ARBITRATION CONTROL ON THE SAME CASE must not enter the map: the
        # recut reads this key as the arbitration measurement and nothing else.
        {"control": "withheld-consult", "case": "pml_2d", "verdict": "WITHHELD"},
        {"control": "arbitration", "case": "folded_2d", "verdict": "NOT-APPLICABLE"},
    ]

    summary = route_gate.arbitration_summary(controls)

    assert summary["pml_2d"] == ["ARBITRATION-DEFAULT-HELD", "INCUMBENT-YIELDED",
                                 "PREFERENCE-REFUSED-BY-NAME"]
    assert summary["folded_2d"] == ["NOT-APPLICABLE"]
    # THE TWO THE RECUT REQUIRES BY NAME, stated here as the reason the list shape
    # exists rather than left implicit in the ordering above.
    assert {"INCUMBENT-YIELDED", "ARBITRATION-DEFAULT-HELD"} <= set(summary["pml_2d"])


def test_the_recut_refuses_a_scalar_arbitration_spelling():
    """A one-verdict-per-case artifact is refused, not read as a one-element list.

    Accepting the scalar would let a campaign run under the old serialization pass
    whichever of its three controls happened to land last -- and on that spelling
    ``PREFERENCE-REFUSED-BY-NAME`` always lands last, so every such artifact would
    fail the INCUMBENT-YIELDED check anyway, with a message blaming the run. The
    refusal names the serialization so the fix is a re-run of the leg.
    """
    source = (pathlib.Path(__file__).resolve().parent
              / "recut_driver_dispatch_record.py").read_text(encoding="utf-8")

    assert "isinstance(verdict, str)" in source
    assert "the campaign runs three" in source


@pytest.mark.parametrize("leg,preference,expected", [
    ("cuda_default_precedence", "cuda", "must run with --table-preference unset"),
    ("cuda_shipped", "unset", "must run with --table-preference cuda"),
    ("cuda_harness_keep", "unset", "must run with --table-preference cuda"),
])
def test_a_cuda_leg_run_under_the_wrong_switch_is_refused(tmp_path, leg,
                                                          preference, expected):
    """The leg's NAME and the backend-preference switch have to agree.

    WHY THIS IS A REFUSAL AND NOT A CONVENTION: the published record carries a
    ``legs`` map asserting that ``cuda_shipped`` ran with the preference set and
    ``cuda_default_precedence`` ran without it, and NOTHING downstream re-reads the
    environment to check. A leg run under the wrong switch therefore publishes a
    by-preference measurement as the by-default number -- two numbers that differ by
    exactly what the arbitration does -- and every artifact in the chain looks
    right. The directory leaf is the leg's name, which is how the recut discovers
    legs at all, so the gate can compare them before it does any work.
    """
    gate = pathlib.Path(__file__).resolve().parent / "gate_dispatch_fused_route.py"
    out = tmp_path / leg
    finished = subprocess.run(
        [sys.executable, str(gate), "--backend", "cuda",
         "--table-preference", preference, "--out", str(out),
         "--cases", "pml_2d", "--smoke"],
        capture_output=True, text=True, timeout=600)

    assert finished.returncode == 2, finished.stdout[-2000:]
    assert expected in finished.stderr
    # REFUSED BEFORE IT WROTE ANYTHING. A refusal that had already created the leg
    # directory would leave a shape the recut counts as a leg.
    assert not out.exists()


# ---------------------------------------------------------------------------
# WHAT THE COUNT EVALUATES, AND WHAT IT ONLY DECLARES (2026-09-11)
# ---------------------------------------------------------------------------


def test_the_pair_may_absorb_clause_is_evaluated_not_only_declared():
    """FLOOR-ADJACENT: the headline must count installs, not admissions.

    ``served_in_dispatch`` listed ``_pair_may_absorb`` in ``clauses_below_this_one``
    and then credited every instance anyway. That cost nothing while H_to_D was the
    only span reaching into a neighbour's slots and no H->D arm was released — and it
    was precisely the change that would have made it cost everything: releasing the
    H->D arms moves 173 instances PER BACKEND into the headline against at most 244
    installs across all three. The slot-contention half is evaluated now, whenever the
    board hands each instance its seam.
    """
    # The clause is reported as evaluated whenever every instance carries a seam,
    # and the three shipped boards all do since this change (asserted by text below).
    measured = reachability.served_in_dispatch("metal", [
        ("a_row", "fused_magnetic_pair", {}, "B_to_H")])
    assert measured["slot_contention"]["evaluated"] is True
    # And every board passes the seam, so none of them gets the declared-only path.
    for name in ("build_fusion_matrix.py", "build_triton_fusion_matrix.py",
                 "build_cuda_fusion_matrix.py"):
        text = (pathlib.Path(__file__).parent / name).read_text(encoding="utf-8")
        assert "h_to_d_seam.SEAM)" in text, name
    # THE UNIT, over both seam vocabularies: a pair owning its own slots contends
    # with nothing.
    for seam in ("B_to_H", "B->H"):
        verdict = reachability._slot_contention([("a_row", seam, "a_pair")])
        assert verdict["contended_slots"] == [], seam
        assert verdict["rows_examined"] == 1, seam
        assert verdict["instances_whose_seam_this_table_cannot_name"] == {}, seam
    # TWO DIFFERENT PRODUCTS WANTING ONE SLOT is the refusal this exists for: an
    # H->D span takes update_H from the B seam and step_D from the D seam.
    clash = reachability._slot_contention([
        ("a_row", "B_to_H", "a_pair"), ("a_row", "H_to_D", "an_hd_weld")])
    assert [c["slot"] for c in clash["contended_slots"]] == ["update_H"]
    assert clash["contended_slots"][0]["claimants"] == ["a_pair", "an_hd_weld"]


def test_a_board_that_passes_no_seam_says_so_instead_of_pretending():
    """A three-tuple gets the old count AND an honest evaluated: False."""
    measured = reachability.served_in_dispatch("metal", [("a_row", "x", {})])
    assert measured["slot_contention"]["evaluated"] is False
    assert "must pass" in measured["slot_contention"]["why_not"]


def test_a_seam_this_table_cannot_name_refuses_rather_than_examining_nothing():
    """THE VACUOUS-CHECK FAILURE, armed.

    Triton spells its seams ``B->H`` and the other two ``B_to_H``. Before
    ``fusion_taxonomy.canonical_seam`` was asked, the Triton board reported
    ``evaluated: True`` having examined ZERO rows — a clause that looked at nothing
    while claiming to have run. Both halves refuse now.
    """
    verdict = reachability._slot_contention([("a_row", "not_a_seam", "a_pair")])
    assert verdict["instances_whose_seam_this_table_cannot_name"] == {"not_a_seam": 1}
    assert verdict["rows_examined"] == 0
    # and the caller refuses on it rather than reporting an evaluated empty check
    source = (pathlib.Path(__file__).parent / "dispatch_reachability.py").read_text(
        encoding="utf-8")
    assert "could not name the seam of instances counted as dispatching" in source
    assert "examined 0 rows while" in source


def test_one_product_spanning_two_seams_is_recorded_not_refused():
    """``cuda_three_slot_dispersive_weld`` takes step_D -> update_E -> update_P in ONE
    launch and is credited a seam-instance at D_to_E and another at E_to_P, so
    ``update_E`` carries its name twice. Keying contention on the SEAM called that a
    double-claim and refused four correct CUDA rows; keying on the PRODUCT does not,
    and the span is recorded because a reader counting launches needs to know two
    instances shared one."""
    contention = reachability._slot_contention([
        ("a_row", "D_to_E", "cuda_three_slot_dispersive_weld"),
        ("a_row", "E_to_P", "cuda_three_slot_dispersive_weld")])
    assert contention["contended_slots"] == []
    spans = contention["slots_one_product_spans_under_two_seams"]
    assert [s["slot"] for s in spans] == ["update_E"], spans
    assert spans[0]["seams"] == ["D_to_E", "E_to_P"]


def test_the_specialized_family_clause_is_read_per_backend_not_interpolated():
    """It named a file that does not contain the guard on two of three backends.

    The clause was one string with the backend's own package interpolated, so the CUDA
    and Metal boards both published "the composer's specialized_family_owns_the_grid
    guard (<their>/launch.py)" while the guard exists only in triton_kernels. An
    over-declared clause makes the count look more conservative than it is.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415

    package_root = _pathlib.Path(_fastpath.__file__).parent
    for backend, package in reachability.BACKEND_PACKAGES.items():
        launch = package_root / package / "launch.py"
        present = (launch.is_file()
                   and "specialized_family_owns_the_grid"
                   in launch.read_text(encoding="utf-8"))
        clause = reachability._specialized_family_guard_clause(backend)
        if present:
            assert f"({package}/launch.py)" in clause, backend
            assert "NOT A CLAUSE" not in clause, backend
        else:
            assert clause.startswith("NOT A CLAUSE ON THIS BACKEND"), backend
            assert f"({package}/launch.py) refuses" not in clause, backend
    # And the shipped fact this is about, so a port of the guard turns this red.
    assert reachability._specialized_family_guard_clause("triton").startswith(
        "the composer's")
    for backend in ("cuda", "metal"):
        assert reachability._specialized_family_guard_clause(backend).startswith(
            "NOT A CLAUSE")



def test_the_metal_off_diagonal_false_clause_is_gone_and_the_axis_is_per_arm():
    """The debt this test used to pin was PAID on 2026-09-12. It now pins the payment.

    WHAT THE DEBT WAS. The shared Metal envelope's ``off_diagonal_epsilon`` row
    claimed "and none CAN be on the products released here: the composer's
    specialized-family guard refuses every ordinary and folded pair on a grid with
    off-diagonal chi1inv rows". That guard is ``specialized_family_owns_the_grid``
    and it exists only in the TRITON composer, so the refusal was resting on another
    backend's code. The two readings prescribe opposite work — a structural "none
    can be" says do not bother, a coverage boundary says drive the case — and the
    axis was the largest single lever on the Metal board.

    WHAT PAID IT. ``pml_3d`` (the sphere) is a DRIVE case now and the ordinary
    magnetic pair dispatches there on the RELEASE route, which is the measurement
    the false clause said could not exist. The axis left the shared table for the
    per-arm one, because the arms stopped agreeing on it: that one arm drives BOTH
    values and so carries no row, while every other arm pins it False.

    THE PREMISE HALF IS KEPT EXACTLY AS IT WAS. The guard must still be absent from
    the Metal package and present in the Triton one. If that ever stops being true
    the correction's reasoning is void, and this test says so rather than assuming a
    settled question stays settled.
    """
    import pathlib as _pathlib  # noqa: PLC0415

    from meep_gpu import fastpath as _fastpath  # noqa: PLC0415
    from meep_gpu import metal_dispatch  # noqa: PLC0415

    # THE PAYMENT: the axis is no longer a shared row at all.
    shared = [row[0] for row in metal_dispatch.METAL_FUSED_RELEASE_ENVELOPE]
    assert "off_diagonal_epsilon" not in shared, shared

    # ...and no surviving text in the module still makes the false claim.
    source = _pathlib.Path(metal_dispatch.__file__).read_text(encoding="utf-8")
    assert "none can be on the products released here" not in source

    # It is a per-arm CORNER bound, not a per-arm row, and the difference is the
    # whole correction: a plain row could only refuse every off-diagonal case
    # driven or admit every one that is not. Two arms have off-diagonal coverage,
    # each at exactly the corner its own off-diagonal cases sat at — the sphere and
    # the cylinder for the ordinary pair, the cylinder on the fold plane for the
    # folded one.
    assert metal_dispatch.METAL_OFF_DIAGONAL_CORNERS == {
        "fused magnetic B/H pair": {"dimensions": frozenset({2, 3}),
                                    "susceptibilities": frozenset({0}),
                                    "conductivity": frozenset({False})},
        "folded fused B/H pair": {"dimensions": frozenset({2}),
                                  "susceptibilities": frozenset({0})},
        # THE FOLDED COMPLEX B/H PAIR'S CORNER, added by the 2026-09-14 target round
        # (Metal L5): its two off-diagonal cases, folded_complex_offdiag_2d and
        # folded_complex_nobloch_offdiag_2d, drove both Bloch values at 2-D.
        "folded complex fused B/H pair": {"bloch": frozenset({False, True}),
                                          "dimensions": frozenset({2})},
        # THE FOURTH, 2026-09-17, and it is EMPTY for a reason that inverts the other
        # three: `folded off-diagonal fused electric D/E pair` is the arm whose kernel
        # IS the off-diagonal constitutive, so ABSENCE here would refuse it on every
        # grid it can run on. Its per-arm row pins every axis to a single value read
        # off folded_offdiag_magnetic_2d's own lift, so under this table's rule --
        # list the axes the row leaves open -- there is nothing left to list.
        "folded off-diagonal fused electric D/E pair": {},
        # AND ITS UNFOLDED TWIN, empty for the same reason.
        "off-diagonal fused electric D/E pair": {},
        # AND THE COMPLEX UNFOLDED ONE, 2026-09-17.
        "complex no-PML off-diagonal fused electric D/E pair": {},
    }
    # ...and the axis is NOT also spelled as a row, because one fact with two homes
    # is one that can disagree with itself.
    rows = {row[0] for axes in metal_dispatch.METAL_FUSED_RELEASE_ARM_AXES.values()
            for row in axes}
    assert "off_diagonal_epsilon" not in rows

    # THE PREMISE, unchanged: the guard really is another backend's.
    package_root = _pathlib.Path(_fastpath.__file__).parent
    assert "specialized_family_owns_the_grid" not in (
        package_root / "metal_kernels" / "launch.py").read_text(encoding="utf-8")
    assert "specialized_family_owns_the_grid" in (
        package_root / "triton_kernels" / "launch.py").read_text(encoding="utf-8")

    # AND THE SIBLING CORRECTION STANDS: the reachability clause reads per backend
    # rather than interpolating one backend's guard onto another.
    assert reachability._specialized_family_guard_clause("metal").startswith(
        "NOT A CLAUSE")


def test_the_census_dimensions_follow_meep_not_the_extent_count():
    """The two shapes the extent count got wrong on 2026-09-12, and the ones it got right.

    ``fastpath._run_shape`` reports ``grid.dimensions``, which ``from_meep`` transcribes
    from ``Simulation._infer_dimensions``; a board that derived the axis from the
    extents credited both folded complex pairs on the one ``kz_2d="3d"`` corpus row
    (a zero-thickness cell MEEP builds in 3-D) that the release rows refuse at runtime,
    and named every ``(0, 0, L)`` cell declared 3-D as 1-D. The census carries the lift
    facts the rule needs; this pins that the derivation reads them, and what it does
    without them.
    """
    from dispatch_reachability import run_shape_from_census  # noqa: PLC0415

    def row(shape, declared, cell, k=(0.0, 0.0, 0.0), beta=0.0, facts=True,
            cylindrical=False):
        configuration = {"shape": shape, "cylindrical": cylindrical, "k_point": list(k),
                         "beta": beta, "has_bloch": any(k), "pml_active": True}
        if facts:
            configuration["facts"] = {"dimensions_attr": declared, "cell_size": list(cell)}
        return configuration

    # kz_2d="3d": z-extent 1, k_point.z != 0, no special-kz beta -> MEEP keeps 3-D.
    assert run_shape_from_census(row([135, 92, 1], 3, (4.5, 6.0, 0.0),
                                     k=(2.797, 0.0, -1.085)))["dimensions"] == 3
    # The special-kz spellings lift beta != 0 and DO collapse to 2-D.
    assert run_shape_from_census(row([120, 120, 1], 3, (6.0, 6.0, 0.0),
                                     k=(0.0, 0.0, 0.4), beta=0.4))["dimensions"] == 2
    # An ordinary 2-D cell declared 3-D collapses; declared 1 or 2 wins outright.
    assert run_shape_from_census(row([200, 120, 1], 3, (10.0, 6.0, 0.0)))["dimensions"] == 2
    assert run_shape_from_census(row([1, 1, 2500], 1, (0.0, 0.0, 100.0)))["dimensions"] == 1
    assert run_shape_from_census(row([200, 120, 1], 2, (10.0, 6.0, 4.0)))["dimensions"] == 2
    # A (0, 0, L) cell declared 3-D is what MEEP builds in 3-D (one cell in x and y).
    assert run_shape_from_census(row([1, 1, 1800], 3, (0.0, 0.0, 72.0)))["dimensions"] == 3
    # Cylindrical is 2 whatever the extents say, facts or not.
    assert run_shape_from_census(row([80, 1, 80], 3, (4.0, 0.0, 4.0),
                                     cylindrical=True))["dimensions"] == 2
    # Without the facts the derivation is the extent count, as every board read
    # until 2026-09-13 -- and assert_every_row_shape_is_read counts the two routes.
    from dispatch_reachability import assert_every_row_shape_is_read  # noqa: PLC0415

    assert run_shape_from_census(row([1, 1, 1800], 3, (0.0, 0.0, 72.0),
                                     facts=False))["dimensions"] == 1
    read = assert_every_row_shape_is_read([
        row([135, 92, 1], 3, (4.5, 6.0, 0.0)), row([1, 1, 1800], 3, (0.0, 0.0, 72.0), facts=False)])
    assert read["with_facts"] == 1 and read["rows"] == 2
