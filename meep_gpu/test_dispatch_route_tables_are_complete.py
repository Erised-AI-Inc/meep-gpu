"""Every DRIVE row can be built and described, checked here rather than on a device.

WHAT THIS EXISTS BECAUSE OF. On 2026-09-17 the CUDA route campaign recorded
``complex_no_pml_offdiag`` as ERROR after 0.0 s, on a leg where the honest verdict
was PASS-NOT-THIS-LEG. The row had been added to ``DRIVE_CUDA`` with a builder in
``CASES`` and a ``needs_probe`` licence, but no entry in ``CASE_INTENT`` — and
``run_case`` reads ``CASE_INTENT[case]`` two lines in, while assembling its row,
BEFORE the ``needs_probe`` clause that would have skipped the case. So the lookup
raised ``KeyError`` before any of the case's own logic ran.

THE COST IS NOT THE CRASH, IT IS THE AMBIGUITY. An ERROR verdict in a campaign
summary reads exactly like a product that failed to dispatch. Distinguishing the two
took a GPU campaign leg, a log, and an artifact extraction — for a missing dictionary
key that a local test answers in milliseconds.

THE TABLES ARE CHECKED AGAINST EACH OTHER, NOT AGAINST A LIST. A hand-written
expected set would have to be edited every time a row is added, which is the same
class of omission this file is here to catch.
"""

from __future__ import annotations

import pathlib
import sys

import pytest


@pytest.fixture(scope="module")
def route():
    directory = str(pathlib.Path(__file__).resolve().parents[1]
                    / "parity" / "meep_gpu")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import gate_dispatch_fused_route as module  # noqa: PLC0415
    return module


@pytest.mark.parametrize("table_name", ["DRIVE", "DRIVE_CUDA"])
def test_every_drive_row_has_a_case_builder(route, table_name):
    table = getattr(route, table_name)
    missing = sorted(name for name in table if name not in route.CASES)
    assert not missing, (
        f"{table_name} rows with no builder in CASES: {missing}. "
        f"``run_case`` does ``CASES[case]`` while building its row, so each of "
        f"these raises KeyError at 0.0 s and is recorded as ERROR — which in a "
        f"campaign summary is indistinguishable from a product that refused")


@pytest.mark.parametrize("table_name", ["DRIVE", "DRIVE_CUDA"])
def test_every_drive_row_has_an_intent(route, table_name):
    table = getattr(route, table_name)
    missing = sorted(name for name in table if name not in route.CASE_INTENT)
    assert not missing, (
        f"{table_name} rows with no CASE_INTENT entry: {missing}. This is the exact "
        f"omission that cost the 2026-09-17 CUDA route campaign its "
        f"``complex_no_pml_offdiag`` case: the lookup happens BEFORE the "
        f"``needs_probe`` clause, so the row cannot even reach the skip that a "
        f"licence-free leg owes it")


@pytest.mark.parametrize("table_name", ["DRIVE", "DRIVE_CUDA"])
def test_every_drive_row_states_what_it_expects(route, table_name):
    """``expect`` and ``why`` are read unconditionally; ``arms`` feeds the fuse env."""
    table = getattr(route, table_name)
    broken = {name: sorted(k for k in ("arms", "expect", "why")
                           if k not in spec)
              for name, spec in table.items()
              if not {"arms", "expect", "why"} <= set(spec)}
    assert not broken, f"{table_name} rows missing required keys: {broken}"


def test_a_probe_licensed_row_names_the_leg_that_carries_the_licence(route):
    """A row that needs a licence must say which leg exports it.

    ``run_case``'s skip message names ``spec['leg']`` when it refuses a case for a
    missing licence. Without it the campaign reports that some other leg drives the
    case while naming no leg at all, and the reader cannot tell whether the case is
    covered elsewhere or simply never runs.
    """
    for table_name in ("DRIVE", "DRIVE_CUDA"):
        table = getattr(route, table_name)
        unnamed = sorted(name for name, spec in table.items()
                         if spec.get("needs_probe") and not spec.get("leg"))
        assert not unnamed, (
            f"{table_name} rows needing a licence but naming no leg: {unnamed}")
