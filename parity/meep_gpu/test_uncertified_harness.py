"""The harness keeps uncertified NVIDIA identities out of what it certifies and times.

The package runs a SUPPORTED NVIDIA device or Triton version that no cited gate ran on
by default, recorded ``certified: False``. Three harness tools must not let such a run
into evidence:

* ``recut_driver_dispatch_record.py`` refuses a route-campaign row that dispatched
  through a table whose device OR (for the Triton table) Triton version is not
  certified;
* ``drive_triton_weld_gates.py`` runs every gate child with
  ``MEEP_GPU_ALLOW_UNCERTIFIED=0``, whatever the launching shell says;
* ``timing_ladder.py`` exports ``0`` to its NVIDIA rows (``test_timing_ladder_plan``).

And ``tools/check_install.py`` expects the kernels where one table may run on what was
read, says which of supported and certified each fact is, and names each identity an
uncertified run was admitted on with its own verdict.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import drive_triton_weld_gates as weld_gates  # noqa: E402
import recut_driver_dispatch_record as recut  # noqa: E402

CHECK_PATH = pathlib.Path(__file__).resolve().parents[2] / "tools" / "check_install.py"
_spec = importlib.util.spec_from_file_location("check_install_uncertified", CHECK_PATH)
check_install = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_install)


def _plan(by_table, triton="3.1.0", triton_certified=True, capability="8.6"):
    return {"environment": {"device": {"compute_capability": capability},
                            "device_certified_by_table": by_table,
                            "triton": triton, "triton_certified": triton_certified}}


def test_a_certified_row_passes():
    plan = _plan({"triton": True, "cuda": True})
    assert recut._uncertified_dispatch_reasons(  # noqa: SLF001
        "pml_2d", plan, {"triton", "cuda"}) == []


@pytest.mark.parametrize("tables", [{"triton"}, {"triton", "cuda"}])
def test_a_row_served_by_an_uncertified_triton_is_refused(tables):
    """The device half was checked before; the compiler half is what the supported
    Triton range now lets through by default."""
    plan = _plan({"triton": True, "cuda": True}, triton="3.1.1", triton_certified=False)
    (reason,) = recut._uncertified_dispatch_reasons(  # noqa: SLF001
        "pml_2d", plan, tables)
    assert reason.startswith("pml_2d dispatched through 'triton' on Triton 3.1.1"), reason
    assert "triton_certified = False" in reason, reason


def test_a_cuda_row_is_not_judged_on_the_triton_version():
    plan = _plan({"triton": True, "cuda": True}, triton="3.1.1", triton_certified=False)
    assert recut._uncertified_dispatch_reasons(  # noqa: SLF001
        "pml_2d", plan, {"cuda"}) == []


def test_a_row_on_a_supported_uncertified_device_is_refused():
    plan = _plan({"triton": False, "cuda": False}, capability="8.9")
    reasons = recut._uncertified_dispatch_reasons(  # noqa: SLF001
        "pml_2d", plan, {"triton", "cuda"})
    assert len(reasons) == 2, reasons
    assert all("compute capability 8.9" in reason for reason in reasons), reasons


@pytest.mark.parametrize("inherited", [None, "1", "0"])
def test_every_gate_child_runs_certified_identities_only(monkeypatch, tmp_path, inherited):
    if inherited is None:
        monkeypatch.delenv("MEEP_GPU_ALLOW_UNCERTIFIED", raising=False)
    else:
        monkeypatch.setenv("MEEP_GPU_ALLOW_UNCERTIFIED", inherited)
    env = weld_gates.gate_environment(0, tmp_path, "a_gate", "keep")
    assert env["MEEP_GPU_ALLOW_UNCERTIFIED"] == "0"
    assert env["MEEP_GPU_DISPATCH"] == "0"


#: (certified, supported) -> would a table run, under unset, 0 and 1.
RUNS = [((True, True), (True, True, True)),
        ((True, None), (True, True, True)),
        ((False, True), (True, False, True)),
        ((False, False), (False, False, True)),
        ((None, None), (True, False, True))]


@pytest.mark.parametrize("fact,expected", RUNS)
def test_check_install_expects_the_kernels_where_a_table_may_run(fact, expected):
    certified, supported = fact
    assert tuple(check_install.nvidia_runs(certified, supported, value)
                 for value in (None, "0", "1")) == expected


@pytest.mark.parametrize("certified,supported,text", [
    (True, True, "on the certified list"),
    (False, True, "supported, NOT on the certified list"),
    (False, False, "NOT supported and NOT on the certified list"),
    (None, None, "could not be read, so it is not refused")])
def test_check_install_says_supported_and_certified_apart(certified, supported, text):
    assert check_install.mark(certified, supported) == text


def _served(what, read, supported, table="triton"):
    return {"table": table, "what": what, "read": read, "supported": supported}


def test_check_install_names_the_supported_identity_alone():
    """Two tables admitting one GPU name it once; a certified toolchain is not named."""
    note = check_install.nvidia_uncertified_note(
        [_served("GPU compute capability", "8.9", True),
         _served("GPU compute capability", "8.9", True, table="cuda")], "S")
    assert note.startswith("GPU compute capability 8.9 is supported but not certified "
                           "bit-identical, so the kernels ran uncertified"), note
    assert note.endswith("Set MEEP_GPU_ALLOW_UNCERTIFIED=0 to run them only on a "
                         "certified device and toolchain."), note
    assert "toolchain are" not in note, note


def test_check_install_says_which_identity_is_outside_the_range():
    note = check_install.nvidia_uncertified_note(
        [_served("Triton", "3.1.1", True),
         _served("GPU compute capability", "6.1", False),
         _served("GPU compute capability", "6.1", False, table="cuda")], "S")
    assert note.startswith("Triton 3.1.1 is supported but not certified bit-identical; "
                           "GPU compute capability 6.1 is outside the supported range, "
                           "which only MEEP_GPU_ALLOW_UNCERTIFIED=1 runs, so the kernels "
                           "ran uncertified"), note
    assert "MEEP_GPU_ALLOW_UNCERTIFIED=0" not in note, note
