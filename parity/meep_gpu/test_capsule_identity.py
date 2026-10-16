"""Tests of ``capsule_identity.py``, the capsule's own identity check, without MEEP.

The lift leg needs MEEP and runs where it is installed. Everything after it is tested
here on a hand-built driver standing in for the lifted one: a load leg in its own
process (MEEP blocked) against a reference leg built from the same driver's stepped
digests, the comparer's word compare and planted flip, and the perturbed-current
control, which must differ.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import capsule_identity as tool  # noqa: E402

from meep_gpu import capsule  # noqa: E402
from meep_gpu.driver import FdtdDriver  # noqa: E402

N_STEPS = 24


def _driver():
    driver = FdtdDriver(cell_size=(3.0, 3.0, 0.0), dimensions=2, resolution=10,
                        force_complex_fields=False)
    driver.set_epsilon(np.full(driver.shape, 2.0, dtype=np.float32))
    driver.setup_pml({"x": (5, 5), "y": (5, 5)})
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 0.8, "center": (0.1, 0.0, 0.0), "size": (0.0, 0.0, 0.0)})
    return driver


@pytest.fixture()
def reference_leg(tmp_path):
    """What a lift leg leaves: a capsule, a currents table, digests and two dumps."""
    from parity.meep_gpu.gate_dispatch_end_to_end import collect_state  # noqa: PLC0415

    out = tmp_path / "lift"
    progress = tool._start(str(out), "reference")  # noqa: SLF001
    with capsule.recording_construction() as recorder:
        driver = _driver()
    info = capsule.write_capsule(out / "capsule.capsule", driver, recorder)
    schedule = tool.schedule_for(N_STEPS, [N_STEPS // 2])
    with capsule.recording_currents(driver) as currents:
        rows = tool._step_rows(driver, schedule, str(out), progress, collect_state, capsule)  # noqa: SLF001
    currents.table().write(out / "currents.json")
    manifest = {"format": tool.FORMAT, "leg": "lift", "schedule": schedule, "rows": rows,
                "capsule": {"file_sha256": info.file_sha256}}
    tool._finish(str(out), manifest, 0)  # noqa: SLF001
    return out


def _load_leg(reference: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "MEEP_GPU_DISPATCH": "0"}
    return subprocess.run(
        [sys.executable, str(HERE / "capsule_identity.py"), "load", "--capsule",
         str(reference / "capsule.capsule"), "--schedule-from", str(reference / "manifest.json"),
         "--out", str(out), *extra],
        capture_output=True, text=True, env=environment, timeout=600)


def test_the_schedule_is_every_step_to_twenty_plus_the_checkpoints_and_n():
    assert tool.schedule_for(200, [20, 40, 100, 200]) == list(range(0, 21)) + [40, 100, 200]
    assert tool.schedule_for(7, [3, 9]) == list(range(0, 8))


def test_a_load_leg_where_meep_cannot_be_imported_matches_the_reference(reference_leg, tmp_path):
    for name, extra in (("plain", ()), ("injected", ("--currents", str(reference_leg / "currents.json")))):
        result = _load_leg(reference_leg, tmp_path / name, *extra)
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        manifest = json.loads((tmp_path / name / "manifest.json").read_text())
        assert manifest["meep_imported"] is False
        assert manifest["load_report"]["arrays_rebuilt_differently"] == 0
        out = tmp_path / f"{name}.json"
        assert tool.main(["compare", "--reference", str(reference_leg), "--candidate",
                          str(tmp_path / name), "--out", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["identical"] and report["planted_caught"]
        assert report["steps_state_equal"] == report["steps_compared"] == 22
        assert report["dumps"]["0"]["differing_words"] == 0
        assert report["dumps"][str(N_STEPS)]["words"] > 0


def test_the_perturbed_current_control_differs_and_is_reported_as_such(reference_leg, tmp_path):
    result = _load_leg(reference_leg, tmp_path / "perturbed", "--currents",
                       str(reference_leg / "currents.json"), "--perturb-current")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    out = tmp_path / "perturbed.json"
    assert tool.main(["compare", "--reference", str(reference_leg), "--candidate",
                      str(tmp_path / "perturbed"), "--out", str(out)]) == 1
    assert tool.main(["compare", "--reference", str(reference_leg), "--candidate",
                      str(tmp_path / "perturbed"), "--expect", "differ", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert not report["identical"] and report["planted_caught"]
    assert report["steps_state_equal"] < report["steps_compared"]
    assert report["first_state_difference"] < N_STEPS, "the change must reach the fields before the end"
    assert report["dumps"]["0"]["differing_words"] == 0
    assert report["dumps"][str(N_STEPS)]["differing_words"] > 0


def test_the_word_compare_counts_a_signed_zero_and_catches_its_planted_flip(tmp_path):
    zero = np.zeros(8, dtype=np.float32)
    signed = zero.copy()
    signed[3] = -0.0
    values = np.arange(1, 9, dtype=np.float32)
    np.savez(tmp_path / "a.npz", **{"state:x": zero, "state:y": values})
    np.savez(tmp_path / "b.npz", **{"state:x": signed, "state:y": values})
    result = tool._compare_dump(str(tmp_path / "a.npz"), str(tmp_path / "b.npz"))  # noqa: SLF001
    assert result["differing_words"] == 1
    assert result["differing_arrays"] == {"state:x": 1}
    assert result["planted"]["caught"] and result["planted"]["array"] == "state:y"


def test_a_load_leg_refuses_a_process_that_already_imported_meep(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "meep", object())
    with pytest.raises(SystemExit, match="already imported"):
        tool.main(["load", "--capsule", str(tmp_path / "x"), "--schedule-from", str(tmp_path / "y"),
                   "--out", str(tmp_path / "z")])
