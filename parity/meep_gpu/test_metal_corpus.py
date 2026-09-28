"""Host-safe checks for the live Metal corpus-census helpers."""

from __future__ import annotations

from types import SimpleNamespace

from parity.meep_gpu import metal_corpus


def test_slot_denominator_adds_update_p_only_for_live_polarizations():
    assert metal_corpus.slot_names(SimpleNamespace(polarizations=[])) == (
        "step_B", "step_D", "update_H", "update_E")
    assert metal_corpus.slot_names(SimpleNamespace(polarizations=[object()])) == (
        "step_B", "step_D", "update_H", "update_E", "update_P")


def test_summary_counts_only_live_selected_slots_and_checks_the_denominator():
    records = [
        {"measured": True, "slots": ["step_B", "step_D"],
         "selected": {"step_B": "PML"}},
        {"measured": False, "slots": ["update_E"], "selected": {}},
    ]
    assert metal_corpus.summary(records) == {
        "rows": 2,
        "measured_rows": 1,
        "slots": 2,
        "selected_slots": 1,
        "unselected_slots": 1,
        "expected_slots": 759,
        "slot_denominator_matches": False,
    }
