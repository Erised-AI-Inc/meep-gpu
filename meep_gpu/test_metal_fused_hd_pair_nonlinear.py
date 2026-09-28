"""Host-side contract for the H->D weld's NONLINEAR WIDENING.

THERE IS NO NEW KERNEL AND NO NEW PRODUCT HERE, and that is the thing this file exists
to hold. :mod:`meep_gpu.metal_kernels.fused_hd_pair`'s shipped predicate already admits
a chi2/chi3 run, through ``nonlinear_update_e``'s two SPINE arms; the widening is a
PREDICATE fact, and its device measurement is
``parity/meep_gpu/gate_metal_fused_hd_pair_nonlinear.py``.

WHAT A HOST SUITE CAN HOLD, then, is the three claims the widening rests on:

* **the spine arms build the CERTIFIED kernels**, through ``launch``'s own builders
  behind a scope view that hides chi2/chi3 and nothing else -- so the fused kernel's
  arithmetic IS the nonlinear run's arithmetic and not merely something like it;
* **the predicate really admits it**, and the two ORDINARY arms really refuse the same
  configuration, so the widening is an inversion and not an overlap;
* **a second product for this cell would be a defect**: two fused products admitting
  one slot leaves it UNSELECTED naming both, which is a coverage loss dressed as a
  gain.

Every test here runs without ``torch``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest

from meep_gpu.metal_kernels import (
    arms, fused_hd_pair as family, launch as metal_launch,
    nonlinear_update_e as nonlinear, shaders,
)

API_ROOT = Path(__file__).resolve().parent.parent
#: The 2026-09-19 board (`_components`), on which `PRODUCTS["fused_hd_pair"]` declares
#: the nonlinear cell beside its plain one and binds it to THIS gate's own run rather
#: than to the family's plain weld. RE-CUT 2026-09-19: the overhead round moved
#: `deposit_repair.py`, the fleet re-gated the cell
#: (`metal_fused_hd_pair_nonlinear_2026-09-19_witness`) and the ledger was re-welded
#: from it, so the 2026-09-13 board this pinned named a run the ledger no longer
#: welds. RE-CUT AGAIN 2026-09-21 at `_2026-09-20_restrict`, same shape and same cause:
#: the component-restricted deposit repair moved the cell's import closure, the fleet
#: re-gated it (`metal_fused_hd_pair_nonlinear_2026-09-20_restrict`) and the ledger was
#: re-welded from that fleet. The board's cell `key` and the ledger's weld both name the
#: 2026-09-20 run. RE-CUT AGAIN 2026-09-24 at `_2026-09-24_round3`: the night round's
#: held-step fixes moved the cell's import closure, the round-3 fleet re-gated it
#: (`metal_fused_hd_pair_nonlinear_2026-09-24_round3`, released, 256 digests, zero drift),
#: the ledger was re-welded from that fleet and the board was cut from it.
#: RE-CUT AGAIN 2026-09-25 at `_2026-09-25_night`: the citation re-point moved the cell's import
#: closure; the `_2026-09-25_night` fleet re-gated it (`metal_fused_hd_pair_nonlinear_2026-09-25_night`, released,
#: zero drift), the host fingerprints were re-cut, and `_b` is the board cut on that ledger.
CELLS_BOARD = (API_ROOT / "parity" / "meep_gpu" / "results"
               / "fusion_matrix_metal_2026-09-25_night_b" / "fusion_matrix.json")
#: The two slots this seam spans, and the two arm labels a nonlinear run selects on
#: them -- the cell the board files these two corpus rows under.
CELL_ARMS: Tuple[str, str] = ("nonlinear PML magnetic", "nonlinear PML curl")
CELL_LABEL = " -> ".join(CELL_ARMS)


# ---------------------------------------------------------------------------
# The spine arms build the certified kernels
# ---------------------------------------------------------------------------

def test_the_spine_curl_builds_through_the_certified_builder():
    """``plan_nonlinear_run_pml_curl`` calls ``launch.plan_pml_curl``, not its own.

    Building the plan THROUGH the certified builder -- rather than re-deriving one --
    is what makes "same kernel" a fact instead of a claim: a divergence would have to
    come from the predicate, which is the only thing the spine contributes.
    """
    body = inspect.getsource(nonlinear.plan_nonlinear_run_pml_curl)
    assert "launch.plan_pml_curl(" in body
    assert "_LinearScopeView(fields)" in body


def test_the_spine_constitutive_builds_through_the_certified_builder():
    body = inspect.getsource(nonlinear.plan_nonlinear_run_constitutive)
    assert "launch.plan_constitutive(" in body
    assert "_LinearScopeView(fields)" in body


def test_the_scope_view_hides_the_nonlinearity_and_nothing_else():
    """The view is what makes the certified predicate answer a nonlinear run.

    It must hide chi2/chi3 -- and it must not hide anything the certified builder
    reads to choose a specialisation, because that would be a different kernel wearing
    the certified one's name.
    """
    view = inspect.getsource(nonlinear._LinearScopeView)  # noqa: SLF001
    assert "has_nonlinearity" in view
    for attribute in ("grid", "shape"):
        assert attribute not in view.split("__slots__", 1)[-1].split("\n", 1)[0]


def test_the_weld_lifts_the_two_certified_emitters():
    """The fused kernel's two halves are ``shaders``' own text.

    The constitutive tail is spliced into ``h_cell`` under
    ``offdiag_weld_common.step_cell_function``'s four-space indent and nothing else;
    the curl half is the certified curl body with the magnetic reads redirected.
    """
    for codes in ((0, 0, 0), (1, 1, 1), (1, 0, 1)):
        source = family.fused_hd_pair_source(codes)
        tail = family.certified_constitutive_tail()
        indented = "\n".join("    " + line if line.strip() else line
                             for line in tail.splitlines())
        assert indented in family.h_cell_function()
        assert family.welded_curl_tail(codes) in source
        # And the curl the weld lifts is the CERTIFIED one for these codes.
        certified = shaders.curl_source(
            codes, bool(metal_launch.SUB_STEPS["step_D"]["backward"]))
        prologue = certified.split("uint idx [[thread_position_in_grid]])\n{\n", 1)[1]
        prologue = prologue.split(family.DECODE_END, 1)[0]
        assert prologue in source


# ---------------------------------------------------------------------------
# The predicate admits it, by an INVERSION rather than an overlap
# ---------------------------------------------------------------------------

class _Grid:
    def __init__(self, **overrides: Any) -> None:
        self.shape = (6, 5, 4)
        self.dt = 0.035
        self.dx = 0.1
        self.beta = 0.0
        self.cylindrical = False
        self.bfast_active = False
        self.dimensions = 3
        for key, value in overrides.items():
            setattr(self, key, value)

    def is_mirrored(self, axis: int) -> bool:  # noqa: ARG002
        return False


def test_the_predicate_consults_the_spine_on_both_halves():
    """The widening is a DISJUNCTION per half, and the two are disjoint on the flag.

    Each half is admitted by EITHER its ordinary predicate OR the spine arm's, and
    ``has_nonlinearity`` separates them, so exactly one can ever answer. A predicate
    that consulted only one side would refuse the cell the board credits.
    """
    body = inspect.getsource(family.metal_fused_hd_pair_coverage)
    assert "nonlinear_run_constitutive_coverage" in body
    assert "nonlinear_run_pml_curl_coverage" in body
    assert "nonlinear-run widening" in body


def test_the_spine_arms_refuse_a_LINEAR_grid_by_name():
    """The inversion, driven from the side that must refuse.

    A spine arm that admitted a linear run would co-admit with the ordinary arm and
    leave the slot UNSELECTED naming both -- the composer's fail-closed answer, and a
    coverage loss.
    """

    class _Fields:
        grid = _Grid()
        has_nonlinearity = False

    for coverage, slot in ((nonlinear.nonlinear_run_pml_curl_coverage, "step_D"),
                           (nonlinear.nonlinear_run_constitutive_coverage, "H")):
        verdict = coverage(_Fields(), None, slot)
        assert not verdict.covered
        assert any("chi2/chi3" in reason for reason in verdict.reasons), slot


def test_the_cell_arms_are_the_two_labels_the_spine_registers():
    """The board files these rows under two labels; both must be live arm labels.

    A cell key nothing measures would price this widening at zero admitted rows and
    read as a widening nothing admits -- the failure the folded H->D product met on
    its way in.
    """
    labels = {spec.label for spec in arms.registered("update_H")}
    labels |= {spec.label for spec in arms.registered("step_D")}
    for label in CELL_ARMS:
        assert label in labels, label


# ---------------------------------------------------------------------------
# A second product here would be a defect
# ---------------------------------------------------------------------------

def test_exactly_one_weld_is_registered_on_update_H_for_this_arm_pair():
    """``fused_hd_pair`` is the only weld whose primary arm pair this cell widens.

    Every OTHER registered ``update_H`` weld names a different cell in its own module,
    and this test reads that off the table rather than from a list here: a new weld
    added for the nonlinear cell would make two rows claim one arm pair, which
    ``arms._select_slot`` answers by selecting neither.
    """
    welds = [spec for spec in arms.registered("update_H") if spec.is_weld]
    assert len(welds) == len({spec.family for spec in welds})
    assert family.FAMILY in {spec.family for spec in welds}
    # And every one of them is unwired, so none can be selected by plan_step.
    assert all(spec.wired is False for spec in welds)


def test_the_extra_arms_row_is_what_reaches_the_nonlinear_cell():
    """``FUSED_PAIR_EXTRA_ARMS`` is how the seam loop asks about a widened cell.

    The primary ``FUSED_PAIR_ARMS`` row is the ``(ordinary, PML)`` pair; the nonlinear
    cell is reached through the EXTRA row, and if the table ever stops carrying one the
    widening becomes unreachable without anything failing.
    """
    extra = getattr(metal_launch, "FUSED_PAIR_EXTRA_ARMS", None)
    if extra is None:
        pytest.skip("launch carries no FUSED_PAIR_EXTRA_ARMS table on this tree")
    rows = extra.get(family.FAMILY)
    assert rows, "the H->D weld has no extra-arms row, so the widening is unreachable"
    flattened = {pair for pair in (rows if isinstance(rows, (list, tuple, set))
                                   else [rows])}
    assert any(tuple(pair) == CELL_ARMS or tuple(pair) == CELL_ARMS[::-1]
               for pair in flattened), (rows, CELL_ARMS)


def test_the_widening_does_not_change_the_seams_declarations():
    """A widened cell is still the same seam: same span, same flags, same rotation."""
    assert family.REPLACES == ("update_H", "step_D")
    assert family.CARRIES_DEPOSIT_REPAIR is False
    assert family.HOISTS_THE_WITHDRAW is False
    assert family.ROTATED_NAMES == ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")


def test_the_installable_reason_names_this_cell_as_the_one_gain():
    """The cell the flag's own measurement calls the single genuine gain.

    The gate drives the composer and finds zero installed pairs on every nonlinear
    fixture, which is what "neither neighbour installs" means in launches. Pinning the
    sentence here keeps the constant and the measurement pointing at the same cell.
    """
    reason = family.INSTALLABLE_REASON
    assert "nonlinear cell" in reason
    assert "3rd-harm-1d" in reason
    assert "neither neighbour installs" in reason


# ---------------------------------------------------------------------------
# The board's credit, read from the artifact side

def _cells_board() -> dict:
    if not CELLS_BOARD.is_file():
        from conftest import requires_resource_skip  # noqa: PLC0415

        requires_resource_skip("metal_fusion_board",
                               f"{CELLS_BOARD} is not in this checkout")
    return json.loads(CELLS_BOARD.read_text(encoding="utf-8"))


@pytest.mark.requires_resource("metal_fusion_board")
def test_the_board_credits_this_cell_from_its_own_bound_artifact():
    """Both corpus rows at this cell are SERVED, and served by this family.

    Until the 2026-09-10 per-cell round the board filed them buildable_not_built:
    the family's plain `halves` refused every nonlinear row before the cell clause
    was asked. Pinned from the artifact: the rows, the bucket, the product, and the
    binding -- which must be THIS gate's run, since the plain family artifact drove
    neither row and a credit on it would rest on a text identity.
    """
    payload = _cells_board()
    rows = [row for row in payload["h_to_d_seam"]["instances"]
            if (row["update_H"], row["step_D"]) == CELL_ARMS]
    assert sorted(row["row"] for row in rows) == [
        "examples:3rd-harm-1d.py", "tests:Test3rdHarm1d.test_3rd_harm_1d"], rows
    assert all(row["bucket"] == "served" for row in rows), rows
    assert all(row["served_by"] == family.FAMILY for row in rows), rows
    product = payload["products"][family.FAMILY]
    cell = product["cells"][CELL_LABEL]
    assert cell["bound_by"] == "cell"
    assert cell["binding"][1].startswith("metal_fused_hd_pair_nonlinear_"), cell
    assert not cell["binding"][1].startswith("metal_fused_hd_pair_2026"), cell
    assert sorted(cell["rows"]) == sorted(row["row"] for row in rows)
    assert product["admitted"] == 47 + 2, product["admitted"]


@pytest.mark.requires_resource("metal_fusion_board")
def test_the_two_pin_registries_agree_on_the_nonlinear_cell():
    """The board's cell binding and the ledger's weld name the SAME run.

    Two registries can each be internally consistent and still point at different
    artifacts (the CUDA track's `certification.json` / `fingerprints.json` split).
    So the cell's `fingerprint` key is resolved and the ledger's `artifact_sha256`
    is checked against the bytes of the artifact the board is bound to.
    """
    payload = _cells_board()
    bound = payload["release_binding_per_cell"][f"{family.FAMILY} / {CELL_LABEL}"]
    assert bound["fingerprint"] == "metal_fused_hd_pair_nonlinear_device_gate"
    key = bound["key"]
    artifact = (API_ROOT / "parity" / "meep_gpu" / "results"
                / (key if key.endswith(".json") else f"{key}/gate.json"))
    ledger = json.loads((API_ROOT / "meep_gpu" / "metal_kernels"
                         / "fingerprints.json").read_text(encoding="utf-8"))
    entry = ledger[bound["fingerprint"]]
    assert entry["artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()).hexdigest(), (key, entry["artifact_sha256"][:12])


# ---------------------------------------------------------------------------
# The per-row clause an extra cell is credited by, pinned without a device

def _board_module():
    import sys  # noqa: PLC0415
    parity = API_ROOT / "parity" / "meep_gpu"
    if str(parity) not in sys.path:
        sys.path.insert(0, str(parity))
    import build_fusion_matrix as board  # noqa: PLC0415
    return board


def _rows_file(tmp_path, records):
    """A gate-shaped `lift/rows.jsonl` under a results-relative directory."""
    board = _board_module()
    directory = tmp_path / "metal_fused_hd_pair_nonlinear_synthetic"
    (directory / "lift").mkdir(parents=True)
    (directory / "lift" / "rows.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8")
    # `_driven_bit_identical_by` resolves the binding under the board's RESULTS root;
    # point the root at tmp_path for the duration of the test.
    return board, directory.name


def _record(label, selected, **block):
    base = {"predicate_admits": True, "driven": True, "bit_identical": True,
            "passed": True}
    base.update(block)
    return {"label": label, "measured": True,
            "census_selected": {"update_H": selected[0], "step_D": selected[1]},
            "metal_fused_hd_pair_nonlinear_gate": base}


def test_the_driven_row_clause_admits_only_a_row_driven_at_this_cell(tmp_path, monkeypatch):
    """Identical bytes at ANOTHER cell are not this cell's evidence.

    The lift record is keyed by row label, and a row can select different arms
    under different censuses. The clause reads the record's own `census_selected`
    and refuses a record driven at a different cell, however identical it was.
    """
    board, name = _rows_file(tmp_path, [
        _record("examples:at_this_cell", CELL_ARMS),
        _record("examples:at_the_plain_cell", ("ordinary", "PML")),
        _record("examples:not_identical", CELL_ARMS, bit_identical=False),
    ])
    monkeypatch.setattr(board, "RESULTS", tmp_path)
    board._DRIVEN_ROWS_CACHE.clear()
    label, clause = board._driven_bit_identical_by(
        CELL_ARMS, ("gate", f"{name}/gate.json"),
        ("lift/rows.jsonl", "metal_fused_hd_pair_nonlinear_gate"))
    assert label == f"driven_bit_identical_by_{name}"
    assert clause({"leg": "examples", "row": "at_this_cell"}) is True
    assert clause({"leg": "examples", "row": "at_the_plain_cell"}) is False
    assert clause({"leg": "examples", "row": "not_identical"}) is False
    assert clause({"leg": "examples", "row": "never_lifted"}) is False


def test_the_driven_row_clause_refuses_a_record_that_does_not_name_its_cell(tmp_path, monkeypatch):
    """A record without `census_selected` is refused by name, not read as credited."""
    board, name = _rows_file(tmp_path, [
        {"label": "examples:silent", "measured": True,
         "metal_fused_hd_pair_nonlinear_gate": {
             "predicate_admits": True, "driven": True, "bit_identical": True,
             "passed": True}},
    ])
    monkeypatch.setattr(board, "RESULTS", tmp_path)
    board._DRIVEN_ROWS_CACHE.clear()
    _label, clause = board._driven_bit_identical_by(
        CELL_ARMS, ("gate", f"{name}/gate.json"),
        ("lift/rows.jsonl", "metal_fused_hd_pair_nonlinear_gate"))
    with pytest.raises(SystemExit, match="which arms the census selected"):
        clause({"leg": "examples", "row": "silent"})


def test_an_extra_cell_must_be_bound_to_a_gate_artifact():
    """The table floor: a per-cell credit rests on a run's own lift records.

    A fingerprint entry carries no per-row evidence, so an extra cell bound through
    one is refused by `_wiring` — and the binding walk refuses the same cell again
    rather than skipping its checks (belt and braces, both pinned here).
    """
    import contextlib
    import copy
    import io

    board = _board_module()
    table = copy.deepcopy(board.PRODUCTS)
    cells = list(table["fused_hd_pair"]["cells"])
    cells[1] = dict(cells[1], binding=("fingerprint",
                                       "metal_fused_hd_pair_nonlinear_device_gate"))
    table["fused_hd_pair"]["cells"] = tuple(cells)
    original = board.PRODUCTS
    board.PRODUCTS = table
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(SystemExit, match="needs a GATE artifact"):
                board._wiring()
    finally:
        board.PRODUCTS = original
