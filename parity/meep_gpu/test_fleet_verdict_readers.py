"""Two verdicts the fleet's readers scored "not released" though each gate reached its own.

MEASURED on the precert pass of 2026-10-09 (tree 8e6ccf99):

* ``gate_triton_fused_offdiag_electric.py`` is a REFUTATION gate. It writes
  ``verdict["REFUSAL ESTABLISHED"]`` over ``verdict["clauses"]`` (10 / 10 true) and
  exits 0, and ``gate_provenance.read_verdict`` does not read that shape, so its
  ``canonical_verdict`` is UNREADABLE and the fleet row read ``released=None``.
* ``gate_dispatch_end_to_end.py --plan-only`` is the route chain's preflight. Its
  designed per-case verdict is PLAN-ONLY and it exits 0, but ``summary.json`` carried
  no outcome of its own, and the reader that re-derived one accepted only
  PASS-DISPATCHED / PASS-FELL-BACK.

The fixes: the fleet driver's row reads the refutation gate's own boolean where the
canonical reading is UNREADABLE (``drive_triton_weld_gates.row_verdict``), and the
end-to-end gate writes its outcome as a ``release`` block beside the rows
(``run_release``), which the shared normaliser reads first.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import drive_triton_weld_gates as fleet  # noqa: E402
import gate_dispatch_end_to_end as e2e  # noqa: E402
import gate_provenance  # noqa: E402
import recut_driver_dispatch_record as recut  # noqa: E402

CLAUSES = {"C1 the subject is bit-identical": True,
           "S1 the subject DISAGREES with the two-launch route somewhere": True}


def _refutation(established, clauses=None):
    payload = {"gate": "gate_triton_fused_offdiag_electric",
               "verdict": {"clauses": dict(clauses or CLAUSES),
                           "REFUSAL ESTABLISHED": established}}
    payload["canonical_verdict"] = gate_provenance.read_verdict(payload)
    return payload


# ---------------------------------------------------------------------------
# The refutation gate's own word
# ---------------------------------------------------------------------------

def test_the_normaliser_cannot_read_the_refutation_shape():
    """Why the widening is in the row reader: read_verdict stays UNREADABLE here."""
    canonical = _refutation(True)["canonical_verdict"]
    assert canonical["released"] is None and canonical.get("unreadable")


def test_the_row_reads_an_established_refusal_as_released():
    row = fleet.row_verdict(_refutation(True))
    assert row["released"] is True
    assert row["read_from"] == "verdict.REFUSAL ESTABLISHED"
    assert row["canonical_read_from"] is None and row["unreadable"]


def test_an_unestablished_refusal_is_refused_with_its_false_clauses():
    clauses = dict(CLAUSES, **{"S1 the subject DISAGREES with the two-launch route "
                                "somewhere": False})
    row = fleet.row_verdict(_refutation(False, clauses))
    assert row["released"] is False
    assert row["reasons"] == ["clause false: S1 the subject DISAGREES with the "
                              "two-launch route somewhere"]


def test_a_readable_canonical_verdict_always_wins():
    payload = _refutation(True)
    payload["canonical_verdict"] = {"released": False, "reasons": ["x"],
                                    "read_from": "release.released"}
    row = fleet.row_verdict(payload)
    assert row["released"] is False and row["read_from"] == "release.released"


def test_a_payload_with_no_known_shape_stays_unreadable():
    payload = {"gate": "g", "verdict": {"clauses": CLAUSES}}
    payload["canonical_verdict"] = gate_provenance.read_verdict(payload)
    row = fleet.row_verdict(payload)
    assert row["released"] is None and row["read_from"] is None


# ---------------------------------------------------------------------------
# The end-to-end gate's own outcome
# ---------------------------------------------------------------------------

def _summary(rows, *, null_control=False):
    verdicts = {row["case"]: row["verdict"] for row in rows}
    offenders = ({row["case"]: {"verdict": row["verdict"]} for row in rows
                  if row["verdict"] != "PASS-FELL-BACK"} if null_control else {})
    return {"null_control": null_control,
            "null_control_clean": (bool(rows) and not offenders) if null_control else None,
            "null_control_offenders": offenders, "verdicts": verdicts,
            "dispatched_cases": [c for c, v in verdicts.items() if v == "PASS-DISPATCHED"],
            "failures": [c for c, v in verdicts.items()
                         if v not in ("PASS-DISPATCHED", "PASS-FELL-BACK", "PLAN-ONLY")],
            "rows": rows}


def test_a_plan_only_preflight_reaches_its_designed_outcome():
    summary = _summary([{"case": "pml_2d", "verdict": "PLAN-ONLY",
                         "plan": {"decision": "dispatched"}}])
    release = e2e.run_release(summary, plan_only=True, smoke=False)
    assert release["released"] is True and release["outcome"] == "PLAN-ONLY"
    assert release["plan_decisions"] == {"pml_2d": "dispatched"}
    summary["release"] = release
    canonical = gate_provenance.read_verdict(summary)
    assert canonical["released"] is True and canonical["read_from"] == "release.released"


def test_a_plan_only_run_that_errored_is_refused():
    summary = _summary([{"case": "pml_2d", "verdict": "ERROR"}])
    release = e2e.run_release(summary, plan_only=True, smoke=False)
    assert release["released"] is False and release["outcome"] == "CASES-FAILED"


def test_a_ship_leg_that_dispatched_nothing_is_still_refused():
    """The guard the block must not lose: failures == [] and no dispatch is exit 1."""
    summary = _summary([{"case": "pml_2d", "verdict": "PASS-FELL-BACK"}])
    release = e2e.run_release(summary, plan_only=False, smoke=False)
    assert (release["released"], release["outcome"]) == (False, "NOTHING-DISPATCHED")
    assert e2e.run_release(summary, plan_only=False, smoke=True)["released"] is True
    summary["release"] = release
    assert gate_provenance.read_verdict(summary)["released"] is False
    assert "finished_utc" not in summary


def test_the_null_control_and_the_ship_leg_read_as_the_exit_code_does():
    clean = _summary([{"case": "pml_2d", "verdict": "PASS-FELL-BACK"}], null_control=True)
    assert e2e.run_release(clean, plan_only=False, smoke=False)["outcome"] == (
        "NULL-CONTROL-CLEAN")
    dirty = _summary([{"case": "pml_2d", "verdict": "PASS-DISPATCHED"}],
                     null_control=True)
    assert e2e.run_release(dirty, plan_only=False, smoke=False)["released"] is False
    ship = _summary([{"case": "pml_2d", "verdict": "PASS-DISPATCHED"},
                     {"case": "bloch_2d", "verdict": "PASS-FELL-BACK"}])
    assert e2e.run_release(ship, plan_only=False, smoke=False)["outcome"] == (
        "ALL-CASES-PASSED")


def test_plan_only_still_licenses_no_dispatch():
    assert "PLAN-ONLY" not in recut.LICENCE_VERDICTS
