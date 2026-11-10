"""``tools/check_install.py`` says how the lift read MEEP's permittivity.

The package records the read on ``driver.chi1inv_read``: the path (``"bulk"`` or
``"per-point"``), why the bulk read was not taken or was restarted, the MEEP build
as the lift stamps it, and, where the package records them, whether that build is
certified for the bulk read and how many points its self-check compared. The check
carries those fields out of its child (``summarize_lift``) and prints one line
(``describe_lift``). A field the package does not record is left out and said to be
unrecorded, never guessed.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

CHECK_PATH = pathlib.Path(__file__).resolve().parents[2] / "tools" / "check_install.py"
_spec = importlib.util.spec_from_file_location("check_install_lift", CHECK_PATH)
check_install = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_install)

STAMP = ("1.35.0", "single", "Darwin", "arm64")


def _record(**fields):
    record = {"path": "bulk", "refusal": None, "restarted_by": "", "stamp": STAMP,
              "components": {"Ex": {"points": 10}}, "seconds": 0.25}
    record.update(fields)
    return record


def test_the_child_carries_the_scalar_fields_and_leaves_out_what_is_not_recorded():
    lift = check_install.summarize_lift(_record())
    assert lift == {"path": "bulk", "refusal": None, "restarted_by": "",
                    "stamp": list(STAMP), "seconds": 0.25}
    json.dumps(lift)  # it crosses the process boundary as JSON
    lift = check_install.summarize_lift(_record(certified_build=False, self_checked_points=930))
    assert lift["certified_build"] is False and lift["self_checked_points"] == 930
    assert "components" not in lift


@pytest.mark.parametrize("record", [None, "bulk", 3])
def test_no_record_is_none(record):
    assert check_install.summarize_lift(record) is None
    assert check_install.describe_lift(check_install.summarize_lift(record)) == (
        "permittivity read: not recorded (the run recorded no chi1inv read)")


@pytest.mark.parametrize("fields, words", [
    (dict(certified_build=True, self_checked_points=930),
     "permittivity read in bulk from MEEP 1.35.0 single precision on Darwin arm64; "
     "that build is certified for the bulk read"),
    (dict(certified_build=False, self_checked_points=21180),
     "permittivity read in bulk from MEEP 1.35.0 single precision on Darwin arm64; "
     "that build is NOT certified for the bulk read (its sampled self-check compared "
     "21,180 points with MEEP's own reader)"),
    (dict(),
     "permittivity read in bulk from MEEP 1.35.0 single precision on Darwin arm64; "
     "whether that build is certified is not recorded by this package"),
    (dict(path="per-point", refusal="MEEP_GPU_HARDWARE=certified restricts the bulk read",
          certified_build=False, self_checked_points=0),
     "permittivity read one point at a time from MEEP 1.35.0 single precision on Darwin "
     "arm64 (MEEP_GPU_HARDWARE=certified restricts the bulk read); that build is NOT "
     "certified for the bulk read"),
    (dict(path="per-point", refusal="restarted per point: Ex: corner disagreed",
          restarted_by="Ex: corner disagreed", certified_build=True),
     "permittivity read one point at a time from MEEP 1.35.0 single precision on Darwin "
     "arm64 (Ex: corner disagreed); that build is certified for the bulk read"),
])
def test_the_line_names_the_path_the_build_and_its_certification(fields, words):
    assert check_install.describe_lift(check_install.summarize_lift(_record(**fields))) == words
