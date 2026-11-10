"""The certified NVIDIA stack is recorded per compute capability, from its own lanes (host-only).

``MEEP_GPU_HARDWARE=certified`` judges a device on the stack its own compute capability was
certified on (``fastpath.certified_nvidia_stack``). These tests pin how that record is made
and used outside the package:

* each gate lane's toolchain step records the stack its gates run beside, once, and a lane
  that resumes beside another stack stops (``run_lane.py``);
* both bind printers name the record tool (``chunks.py bind``, ``legs.py bind``) before any
  line that records or rebinds (only the chunked bind's verify, which writes nothing, comes
  first), and the unchunked stage's stack directory is a one-shot root;
* ``record_nvidia_stack.py`` adds a new capability's stack, keeps a certified one on the same
  stack, adds a CUDA driver whose lanes read the recorded value on every other axis (the
  record then lists each driver with the stamps that ran on it), refuses any other change
  unless told to re-state it, and refuses lanes that disagree on any axis but the driver, an
  axis nobody read, another capability and an uncertified MEEP build, writing nothing; what
  it writes is what dispatch then reads, and a host on any listed driver is certified while
  one on a driver no lane ran on is refused;
* the route stage asks the gate interpreter for rung 4c's verdict before any leg runs, and
  the route gate's admitted-policy leg refuses a host off its capability's stack before it
  runs a case (a smoke run, which consults no kernel table, is not judged).
"""
from __future__ import annotations

import argparse
import json
import stat
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROUND = HERE / "rounds" / "1.0"
for path in (str(ROUND), str(HERE), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

import chunks as C  # noqa: E402
import legs as L  # noqa: E402
import record_nvidia_stack as R  # noqa: E402
import run_lane  # noqa: E402

from meep_gpu import fastpath  # noqa: E402

#: 8.6's recorded values, the first of each axis. The record lists two CUDA drivers since the
#: 1.0.1 round's bind: 12030 (the RTX A6000's chunks) and 13040 (the RTX 3090's), so a lane
#: built from these reads 12030 and a lane on 13040 is written with it named.
STACK_86 = {axis: values[0] for axis, values in fastpath.certified_nvidia_stack("8.6").items()}
#: 8.6's recorded CUDA drivers, typed out: the pin the tests below hold the shipped record to.
DRIVERS_86 = ("12030", "13040")
#: A driver no lane of the 8.6 record ran on, for the tool's "adds a driver" path.
FURTHER_DRIVER = "13060"


def lane_record(capability="9.0", lane="gates_0", utc="2026-10-21T01:00:00Z", **values):
    stack = dict(STACK_86, **values)
    return {"schema": R.STACK_SCHEMA, "lane": lane, "capability": capability,
            "stack": stack, "utc": utc}


def write_lanes(directory: Path, *records) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        (directory / f"{record['lane']}.json").write_text(json.dumps(record))
    return directory


@pytest.fixture()
def root(tmp_path):
    """A repository root holding a copy of the shipped Triton ledger."""
    ledger = tmp_path / R.LEDGER
    ledger.parent.mkdir(parents=True)
    ledger.write_text((REPO / R.LEDGER).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def recorded(root: Path) -> dict:
    ledger = json.loads((root / R.LEDGER).read_text(encoding="utf-8"))
    return ledger[fastpath.CERTIFIED_STACK_ENTRY][fastpath.STACK_BY_CAPABILITY]


def run(root: Path, *argv) -> int:
    return R.main(["--root", str(root), *argv])


# --------------------------------------------------------------------- the bind tool ---
def test_a_new_capability_records_the_stack_its_lanes_ran_beside(root, capsys):
    lanes = write_lanes(root / "results" / "chunk_x_cc90_c0" / "stack",
                        lane_record(cuda_driver="12080", lane="gates_0",
                                    utc="2026-10-21T02:00:00Z"),
                        lane_record(cuda_driver="12080", lane="gates_1"))
    before = (root / R.LEDGER).read_text(encoding="utf-8")
    assert run(root, "--stack", str(lanes), "--capability", "9.0") == 0
    assert (root / R.LEDGER).read_text(encoding="utf-8") == before, "report mode writes"
    assert "adds the certified stack of compute capability 9.0" in capsys.readouterr().out
    assert run(root, "--stack", str(lanes), "--capability", "9.0", "--write") == 0
    record = recorded(root)["9.0"]
    assert record["stack"] == {axis: [value] for axis, value in
                               dict(STACK_86, cuda_driver="12080").items()}
    assert record["read_from"] == ["results/chunk_x_cc90_c0/stack/gates_0.json",
                                   "results/chunk_x_cc90_c0/stack/gates_1.json"]
    assert record["recorded_utc"] == "2026-10-21T01:00:00Z", "the earliest lane"
    assert recorded(root)["8.6"] == recorded(REPO)["8.6"], "another capability is untouched"
    after = json.loads((root / R.LEDGER).read_text(encoding="utf-8"))
    shipped = json.loads(before)
    assert {k: v for k, v in after.items() if k != fastpath.CERTIFIED_STACK_ENTRY} == {
        k: v for k, v in shipped.items() if k != fastpath.CERTIFIED_STACK_ENTRY}
    assert json.dumps(after, **R.LEDGER_FORMAT) + "\n" == (root / R.LEDGER).read_text()


def test_what_the_bind_writes_is_what_dispatch_reads(root, monkeypatch):
    lanes = write_lanes(root / "stack", lane_record(cuda_driver="12080"))
    assert run(root, "--stack", str(lanes), "--write") == 0
    ledger = json.loads((root / R.LEDGER).read_text(encoding="utf-8"))
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {"triton_kernels": ledger})
    assert fastpath.certified_nvidia_stack("9.0")["cuda_driver"] == ("12080",)
    assert fastpath.certified_nvidia_stack("8.6")["cuda_driver"] == DRIVERS_86


def test_a_certified_capability_on_its_own_stack_keeps_it(root, capsys):
    """Lanes on 8.6's recorded stack, one on each of its two recorded drivers."""
    lanes = write_lanes(root / "stack", lane_record(capability="8.6"),
                        lane_record(capability="8.6", cuda_driver=DRIVERS_86[1],
                                    lane="gates_1"))
    assert run(root, "--stack", str(lanes), "--write") == 0
    assert "keeps the certified stack of compute capability 8.6" in capsys.readouterr().out
    assert recorded(root)["8.6"]["stack"] == recorded(REPO)["8.6"]["stack"]
    assert recorded(root)["8.6"]["stack"]["cuda_driver"] == list(DRIVERS_86)
    assert recorded(root)["8.6"]["read_from"] == ["stack/gates_0.json", "stack/gates_1.json"]


def test_a_certified_capability_on_another_driver_is_refused_unless_restated(root, capsys):
    lanes = write_lanes(root / "stack", lane_record(capability="8.6", cuda_driver="12090"))
    before = (root / R.LEDGER).read_text(encoding="utf-8")
    assert run(root, "--stack", str(lanes), "--write") == 1
    out = capsys.readouterr().out
    assert "REFUSED: compute capability 8.6 is certified on another stack" in out, out
    assert "CUDA driver ['12030', '13040'] -> ['12090']" in out, out
    assert (root / R.LEDGER).read_text(encoding="utf-8") == before
    assert run(root, "--stack", str(lanes), "--restate", "--write") == 0
    assert recorded(root)["8.6"]["stack"]["cuda_driver"] == ["12090"]


@pytest.mark.parametrize("records,needle", [
    ((lane_record(cuda_driver="12080"), lane_record(numpy="2.3", lane="gates_1")),
     "the lanes disagree on the NumPy: 2.2 in"),
    ((lane_record(), lane_record(capability="8.6", lane="gates_1")),
     "the lanes ran on ['8.6', '9.0']"),
    ((lane_record(nvrtc=None),), "leaves ['nvrtc'] unread"),
    ((dict(lane_record(), schema="other"),), "is not a lane stack record"),
    ((lane_record(capability="86"),), "'86' is not spelled as the ledger keys a capability"),
    ((lane_record(capability="sm_90"),), "is not a normalised compute capability"),
    ((lane_record(meep="1.35.0 double"),), "is not a certified MEEP build"),
    ((dict(lane_record(), stack=dict(STACK_86, gpu="x")),), "which are not axes"),
])
def test_what_the_bind_refuses_writes_nothing(root, capsys, records, needle):
    lanes = write_lanes(root / "stack", *records)
    before = (root / R.LEDGER).read_text(encoding="utf-8")
    assert run(root, "--stack", str(lanes), "--write") == 1
    out = capsys.readouterr().out
    assert needle in out, out
    assert (root / R.LEDGER).read_text(encoding="utf-8") == before


# ------------------------------------------------- one capability, several CUDA drivers ---
def two_driver_lanes(root: Path, second: dict, first: str = "12030") -> list:
    """A selection's stack directories: an A-card chunk on 8.6's first recorded driver and a
    B-card chunk on its second, 13040, with ``second`` read in place of the 8.6 values
    there."""
    return [write_lanes(root / "results" / "chunk_r_cc86_c0" / "stack",
                        lane_record(capability="8.6", cuda_driver=first,
                                    utc="2026-10-21T03:00:00Z"),
                        lane_record(capability="8.6", cuda_driver=first, lane="gates_1")),
            write_lanes(root / "results" / "chunk_r_cc86_c27" / "stack",
                        lane_record(capability="8.6", lane="gates_0",
                                    utc="2026-10-21T02:00:00Z",
                                    **dict({"cuda_driver": "13040"}, **second)))]


def stacks(*directories) -> list:
    return [arg for directory in directories for arg in ("--stack", str(directory))]


def test_a_further_driver_with_every_other_axis_equal_is_added_beside_the_recorded_ones(
        root, capsys):
    """The 8.6 record on drivers 12030 and 13040, as the round's bind wrote it, and a
    selection that ran on both and on a third, every other axis read alike: no
    ``--restate`` is needed; the record lists the three drivers, in version order, with the
    stamps that ran on each, and keeps every other value."""
    lanes = two_driver_lanes(root, {}) + [
        write_lanes(root / "results" / "chunk_r_cc86_c41" / "stack",
                    lane_record(capability="8.6", cuda_driver=FURTHER_DRIVER,
                                utc="2026-10-21T04:00:00Z"))]
    before = (root / R.LEDGER).read_text(encoding="utf-8")
    assert run(root, *stacks(*lanes), "--capability", "8.6") == 0
    out = capsys.readouterr().out
    assert (root / R.LEDGER).read_text(encoding="utf-8") == before, "report mode writes"
    assert (f"adds CUDA driver {FURTHER_DRIVER} to the certified stack of compute capability "
            f"8.6, beside {', '.join(DRIVERS_86)}") in out, out
    assert f"  CUDA driver: {', '.join(DRIVERS_86)}, {FURTHER_DRIVER}\n" in out, out
    assert f"  CUDA driver {FURTHER_DRIVER} ran in: r_cc86_c41\n" in out, out
    assert run(root, *stacks(*reversed(lanes)), "--capability", "8.6", "--write") == 0
    record = recorded(root)["8.6"]
    shipped = recorded(REPO)["8.6"]["stack"]
    assert shipped["cuda_driver"] == list(DRIVERS_86), shipped["cuda_driver"]
    assert record["stack"] == dict(shipped, cuda_driver=[*DRIVERS_86, FURTHER_DRIVER]), \
        record["stack"]
    assert record[R.STAMPS_BY_DRIVER] == {"12030": ["r_cc86_c0"], "13040": ["r_cc86_c27"],
                                          FURTHER_DRIVER: ["r_cc86_c41"]}
    assert record["read_from"] == ["results/chunk_r_cc86_c0/stack/gates_0.json",
                                   "results/chunk_r_cc86_c0/stack/gates_1.json",
                                   "results/chunk_r_cc86_c27/stack/gates_0.json",
                                   "results/chunk_r_cc86_c41/stack/gates_0.json"]
    assert record["recorded_utc"] == "2026-10-21T01:00:00Z", "the earliest lane"
    assert set(record) == {"read_from", "recorded_by", "recorded_utc", "stack",
                           R.STAMPS_BY_DRIVER}, "the stamps sit beside the stack, not in it"
    # The same selection again keeps the three drivers; a lower driver sorts first.
    assert run(root, *stacks(*lanes), "--write") == 0
    assert "keeps the certified stack of compute capability 8.6" in capsys.readouterr().out
    assert recorded(root)["8.6"] == record
    lower = write_lanes(root / "results" / "chunk_r_cc86_c28" / "stack",
                        lane_record(capability="8.6", cuda_driver="9020"))
    assert run(root, *stacks(*lanes, lower), "--write") == 0
    assert recorded(root)["8.6"]["stack"]["cuda_driver"] == ["9020", *DRIVERS_86,
                                                             FURTHER_DRIVER]


@pytest.mark.parametrize("axis,value", [
    ("cupy", "13.6.0"), ("nvrtc", "12.8"), ("cuda_runtime", "12080"), ("host", "Linux aarch64"),
    ("numpy", "2.3"), ("meep", "1.33.0 single")])
def test_a_second_driver_beside_another_value_on_any_other_axis_is_refused(root, capsys,
                                                                         axis, value):
    """The second driver's lanes read another value on one other axis: refused as lanes
    that disagree, writing nothing, and ``--restate`` does not lift it (two drivers on two
    values of another axis would certify the cross-products no lane ran)."""
    lanes = two_driver_lanes(root, {axis: value})
    before = (root / R.LEDGER).read_text(encoding="utf-8")
    for extra in ((), ("--restate",)):
        assert run(root, *stacks(*lanes), *extra, "--write") == 1
        out = capsys.readouterr().out
        assert f"REFUSED: the lanes disagree on the {fastpath.NVIDIA_STACK_NAMES[axis]}: " \
            in out, out
        assert (root / R.LEDGER).read_text(encoding="utf-8") == before


def test_the_recorded_drivers_on_another_recorded_value_is_a_restatement(root, capsys):
    """The lanes of both recorded drivers read NumPy 2.3 where 8.6 records 2.2: the lanes
    agree, so it is not a disagreement, but it moves a certified value, which only
    ``--restate`` writes; then the record is exactly the lanes' stack."""
    for directory in two_driver_lanes(root, {"numpy": "2.3"}, first="12030"):
        for path in directory.glob("*.json"):
            lane = json.loads(path.read_text())
            lane["stack"]["numpy"] = "2.3"
            path.write_text(json.dumps(lane))
    lanes = sorted((root / "results").glob("chunk_*/stack"))
    before = (root / R.LEDGER).read_text(encoding="utf-8")
    assert run(root, *stacks(*lanes), "--write") == 1
    out = capsys.readouterr().out
    assert "REFUSED: compute capability 8.6 is certified on another stack" in out, out
    assert "is certified on another stack (NumPy ['2.2'] -> ['2.3'])" in out, out
    assert (root / R.LEDGER).read_text(encoding="utf-8") == before
    assert run(root, *stacks(*lanes), "--restate", "--write") == 0
    assert recorded(root)["8.6"]["stack"]["numpy"] == ["2.3"]


def test_dropping_a_recorded_driver_needs_restate_and_drops_it(root, capsys):
    """8.6 recorded on 12030 and 13040; a selection that ran on 13040 alone would drop 12030:
    refused without ``--restate``; with it the record is the 13040 stack alone."""
    assert run(root, *stacks(*two_driver_lanes(root, {})), "--write") == 0
    capsys.readouterr()
    alone = write_lanes(root / "results" / "chunk_r_cc86_c29" / "stack",
                        lane_record(capability="8.6", cuda_driver="13040"))
    before = (root / R.LEDGER).read_text(encoding="utf-8")
    assert run(root, "--stack", str(alone), "--write") == 1
    out = capsys.readouterr().out
    assert "CUDA driver ['12030', '13040'] -> ['13040']" in out, out
    assert "pass --restate" in out, out
    assert (root / R.LEDGER).read_text(encoding="utf-8") == before
    assert run(root, "--stack", str(alone), "--restate", "--write") == 0
    assert recorded(root)["8.6"]["stack"]["cuda_driver"] == ["13040"]
    assert recorded(root)["8.6"][R.STAMPS_BY_DRIVER] == {"13040": ["r_cc86_c29"]}


def _rung_4c(monkeypatch, ledger: dict, **read) -> tuple:
    """Rung 4c's verdict on an 8.6 host reading the 8.6 values with ``read`` in their place,
    against ``ledger``: the stack block and the ``certified`` refusal, from dispatch's own
    reader, unchanged."""
    monkeypatch.setattr(fastpath, "_FINGERPRINTS", {"triton_kernels": ledger})
    identity = dict(STACK_86, **read)
    monkeypatch.setattr(fastpath, "_nvidia_stack_identity", lambda xp: dict(identity))
    block = fastpath.nvidia_stack_block(None, "8.6")
    return block, fastpath.certified_stack_refusal(block)


def test_dispatch_certifies_a_host_on_each_listed_driver_and_refuses_another(root, monkeypatch):
    """What the bind writes for two drivers is read by dispatch as written: a host on either
    driver is certified (rung 4c refuses nothing), a host on a driver no lane ran on is
    refused by name with both drivers named, and so is a listed driver beside another value
    of another axis. The shipped record is that record: the round's bind wrote 8.6 on both
    drivers."""
    assert run(root, *stacks(*two_driver_lanes(root, {})), "--write") == 0
    ledger = json.loads((root / R.LEDGER).read_text(encoding="utf-8"))
    for driver in ("12030", "13040"):
        block, refusal = _rung_4c(monkeypatch, ledger, cuda_driver=driver)
        assert block["certified"] is True and refusal is None, (driver, block)
        assert block["axes"]["cuda_driver"]["certified_values"] == ["12030", "13040"]
    assert fastpath.certified_nvidia_stack("8.6")["cuda_driver"] == ("12030", "13040")
    block, refusal = _rung_4c(monkeypatch, ledger, cuda_driver="12090")
    assert block["certified"] is False
    assert refusal == ("the stack the NVIDIA kernels would run beside is not the one they "
                       "were certified on for compute capability 8.6: CUDA driver 12090 is "
                       "not the certified 12030, 13040"), refusal
    block, refusal = _rung_4c(monkeypatch, ledger, cuda_driver="13040", numpy="2.3")
    assert refusal.endswith("NumPy 2.3 is not the certified 2.2"), refusal
    # The shipped record, untouched: 8.6 on either recorded driver is certified on it, and a
    # driver no lane of the round ran on is refused naming both.
    shipped = json.loads((REPO / R.LEDGER).read_text())
    for driver in DRIVERS_86:
        block, refusal = _rung_4c(monkeypatch, shipped, cuda_driver=driver)
        assert block["certified"] is True and refusal is None, (driver, block)
    block, refusal = _rung_4c(monkeypatch, shipped, cuda_driver="12090")
    assert refusal.endswith("CUDA driver 12090 is not the certified 12030, 13040"), refusal


def test_the_bind_refuses_another_capability_a_missing_directory_and_an_outside_one(
        root, capsys):
    lanes = write_lanes(root / "stack", lane_record())
    assert run(root, "--stack", str(lanes), "--capability", "8.6", "--write") == 1
    assert "ran on compute capability 9.0, not 8.6" in capsys.readouterr().out
    assert run(root, "--stack", str(root / "absent"), "--write") == 1
    assert "is not a directory" in capsys.readouterr().out
    (root / "empty").mkdir()
    assert run(root, "--stack", str(root / "empty"), "--write") == 1
    assert "holds no lane stack record" in capsys.readouterr().out
    outside = write_lanes(root.parent / f"{root.name}_outside", lane_record())
    before = (root / R.LEDGER).read_text(encoding="utf-8")
    assert run(root, "--stack", str(outside), "--write") == 1
    assert "lies outside the repository" in capsys.readouterr().out
    assert (root / R.LEDGER).read_text(encoding="utf-8") == before


# ------------------------------------------------------------------ the lane's record ---
def _fake_toolchain(tmp_path: Path, capability: str, stack: dict, ok: bool = True) -> str:
    """An interpreter that prints what the toolchain check prints on a card."""
    path = tmp_path / "bin" / "python"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"capability": capability, "stack": stack})
    verdict = "TOOLCHAIN OK" if ok else "TOOLCHAIN REFUSED: compute capability"
    path.write_text("#!/bin/sh\n"
                    f"echo 'TOOLCHAIN card NVIDIA H200 | cc {capability}'\n"
                    f"echo '{run_lane.STACK_LINE}{line}'\n"
                    f"echo '{verdict}'\n"
                    f"exit {0 if ok else 1}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _lane(tmp_path: Path, python: str) -> run_lane.Lane:
    args = argparse.Namespace(lane_dir=str(tmp_path / "home" / "gates_0"), repo=str(tmp_path),
                              results=str(tmp_path / "results"), python=python, gpu=0,
                              subs="", libcuda_stub="", corpus_root="", min_timeout=1800,
                              timeout_factor=4, heartbeat=120)
    return run_lane.Lane(args)


def test_a_gate_lane_records_its_stack_once_and_stops_beside_another(tmp_path):
    stack_dir = tmp_path / "results" / "chunk_s" / "stack"
    step = {"id": "toolchain", "kind": "toolchain", "capability": "9.0",
            "stack_dir": str(stack_dir)}
    stack = dict(STACK_86, cuda_driver="12080")
    lane = _lane(tmp_path, _fake_toolchain(tmp_path, "9.0", stack))
    assert lane.step_toolchain(step)["state"] == "released"
    written = json.loads((stack_dir / "gates_0.json").read_text())
    assert (written["schema"], written["lane"], written["capability"]) == (
        R.STACK_SCHEMA, "gates_0", "9.0")
    assert written["stack"] == stack and written["utc"]
    # What the lane wrote is what the bind reads.
    assert R.lane_stack("x", written) == ("9.0", stack, written["utc"])
    # A resume beside the same stack keeps the record; beside another, the lane stops.
    assert lane.step_toolchain(step)["state"] == "released"
    assert json.loads((stack_dir / "gates_0.json").read_text()) == written
    moved = _lane(tmp_path, _fake_toolchain(tmp_path, "9.0", dict(stack, cuda_driver="12090")))
    result = moved.step_toolchain(step)
    assert result["state"] == "refused"
    assert "two stacks under one stamp" in result["reasons"][0]
    assert json.loads((stack_dir / "gates_0.json").read_text()) == written
    # The lane names the record relative to its results tree, never by an absolute path.
    said = lane.progress.read_text(encoding="utf-8")
    assert "stack record chunk_s/stack/gates_0.json: " in said, said
    assert "stack record chunk_s/stack/gates_0.json kept (the same stack)" in said, said
    assert result["reasons"][0].startswith("chunk_s/stack/gates_0.json records another")
    assert str(tmp_path) not in said and str(tmp_path) not in result["reasons"][0]


def test_a_route_lane_and_a_refused_toolchain_record_nothing(tmp_path):
    stack_dir = tmp_path / "results" / "stack_s"
    lane = _lane(tmp_path, _fake_toolchain(tmp_path, "9.0", STACK_86, ok=False))
    step = {"id": "toolchain", "kind": "toolchain", "capability": "9.0",
            "stack_dir": str(stack_dir)}
    assert lane.step_toolchain(step)["state"] == "refused"
    assert not stack_dir.exists()
    lane = _lane(tmp_path, _fake_toolchain(tmp_path, "9.0", STACK_86))
    step.pop("stack_dir")
    assert lane.step_toolchain(step)["state"] == "released"
    assert not stack_dir.exists()


def test_the_toolchain_check_prints_the_stack_dispatch_reads():
    assert "fastpath._nvidia_stack_identity(cupy)" in run_lane.TOOLCHAIN_CHECK
    assert '"TOOLCHAIN STACK "' in run_lane.TOOLCHAIN_CHECK
    assert run_lane.STACK_LINE == "TOOLCHAIN STACK "
    assert run_lane.STACK_SCHEMA == R.STACK_SCHEMA


# ---------------------------------------------------------------- the bind printers ---
def test_the_unchunked_bind_records_the_stack_first():
    lines = L.bind_plan(Path("results"), "2099-01-01_t", 1, write=False)
    assert lines[0] == "python record_nvidia_stack.py --stack results/stack_2099-01-01_t"
    assert sum("record_nvidia_stack.py" in line for line in lines) == 1
    assert lines[-1] == C.bind_plan_last(False)
    assert L.bind_plan(Path("results"), "2099-01-01_t", 1, write=True)[0].endswith(" --write")
    assert L.stack_root("2099-01-01_t") in L.round_roots("2099-01-01_t", 1)


def test_a_chunk_records_its_stack_inside_its_own_root():
    assert C.stack_root("r_cc90_c3") == "chunk_r_cc90_c3/stack"
    assert C.stack_root("r_cc90_c3").split("/")[0] == C.launch_root("r_cc90_c3")
    assert C.device_capability("cc86") == "8.6" and C.device_capability("cc90") == "9.0"
    assert C.device_capability("cc120") == "12.0"
    for bad in ("86", "ccx", "cc8", "applegpu_g13s"):
        with pytest.raises(ValueError):
            C.device_capability(bad)


def test_the_chunked_bind_names_every_selected_stamps_stack(monkeypatch):
    index = {"machine": "nvidia", "device": "cc90", "device_mode": "new"}
    selection = {"selected": {"triton/bit_identity": "r_cc90_c1", "cuda/ade_recut": "r_cc90_c0"},
                 "units": {}, "counts": {}}
    monkeypatch.setattr(C, "_ordered", lambda sel, stamps: sorted(set(stamps)))
    monkeypatch.setattr(C, "units_for", lambda repo, machine: [
        {"id": "triton/bit_identity", "kind": "triton_bit_identity"},
        {"id": "cuda/ade_recut", "kind": "cuda_ade_recut"}])
    monkeypatch.setattr(C, "stamp_tool_conflicts", lambda *args: [])
    monkeypatch.setattr(C, "_literal_dict", lambda path, name: {})
    monkeypatch.setattr(C, "triton_ledger_keys", lambda repo: ("ledger", set()))
    monkeypatch.setattr(C, "_family_recert_families", lambda repo: ())
    lines = C.nvidia_bind_commands(index, Path("results"), "index.json", selection, "sel.json",
                                   REPO, write=True)
    # Second, after the verify (which writes nothing) and before every line that writes.
    assert lines[0].startswith("python rounds/1.0/chunks.py verify ") and \
        "--write" not in lines[0]
    assert lines[1] == ("python record_nvidia_stack.py --capability 9.0 "
                        "--stack results/chunk_r_cc90_c0/stack "
                        "--stack results/chunk_r_cc90_c1/stack --write"), lines[1]
    assert sum("record_nvidia_stack.py" in line for line in lines) == 1
    writers = [line for line in lines[2:] if line.startswith("python ")]
    assert writers and all(line.endswith(" --write") for line in writers), writers
    assert lines[-1] == C.bind_plan_last(True)


# ------------------------------------------------------------- the route stage's ask ---
def _answer(stdout: str, returncode: int = 0):
    def run(argv, **kwargs):
        assert argv[1:3] == ["-I", "-c"] and argv[3] == L.ROUTE_STACK_CHECK
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")
    return run


def test_the_route_stage_refuses_a_host_off_its_capabilitys_stack():
    block = {"capability": "9.0", "certified": False}
    refusal = ("the stack the NVIDIA kernels would run beside is not the one they were "
               "certified on for compute capability 9.0: CUDA driver 12030 is not the "
               "certified 12080")
    out = L.ROUTE_STACK_LINE + json.dumps({"block": block, "refusal": refusal}) + "\n"
    problems, read = L.route_stack_problems(REPO, "9.0", "python", run=_answer(out))
    assert read == block
    (problem,) = problems
    assert problem.startswith("the route's admitted-policy legs run under "
                              "MEEP_GPU_HARDWARE=certified, which would refuse every case")
    assert refusal in problem


def test_the_route_stage_goes_on_beside_the_certified_stack_and_names_an_unread_one():
    block = {"capability": "8.6", "certified": True}
    out = L.ROUTE_STACK_LINE + json.dumps({"block": block, "refusal": None})
    assert L.route_stack_problems(REPO, "8.6", "python", run=_answer(out)) == ([], block)
    problems, read = L.route_stack_problems(REPO, "8.6", "python",
                                            run=_answer("ModuleNotFoundError: cupy", 1))
    assert read is None and "could not be read with python (exit 1)" in problems[0]


def test_the_route_stack_check_asks_fastpath_its_own_question(tmp_path, monkeypatch):
    """The child script, run here against a stand-in CuPy and MEEP: rung 4c's verdict."""
    from meep_gpu.test_dispatch_contract import cupy_like_grid_with_device

    xp = cupy_like_grid_with_device().xp
    meep = types.ModuleType("meep")
    meep.__version__, meep.is_single_precision = "1.35.0", (lambda: True)
    printed = []
    namespace = {"__name__": "__main__", "print": lambda line, **_: printed.append(line)}
    monkeypatch.setitem(sys.modules, "cupy", xp)
    monkeypatch.setitem(sys.modules, "meep", meep)
    monkeypatch.setattr(sys, "argv", ["-c", str(REPO), "8.6"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(compile(L.ROUTE_STACK_CHECK, "<route stack check>", "exec"), namespace)  # noqa: S102
    read = json.loads(printed[-1][len(L.ROUTE_STACK_LINE):])
    assert read["block"]["capability"] == "8.6"
    assert read["block"]["axes"]["cuda_driver"] == {
        "what": "CUDA driver", "read": "12030", "certified_values": list(DRIVERS_86),
        "certified": True}
    assert read["block"]["axes"]["meep"]["read"] == "1.35.0 single"
    assert read["refusal"] == fastpath.certified_stack_refusal(read["block"])


# ------------------------------------------------------------------- the route gate ---
def test_the_admitted_policy_leg_judges_the_stack_by_rung_4cs_own_words(monkeypatch):
    import gate_dispatch_fused_route as gate
    from meep_gpu.test_dispatch_contract import cupy_like_grid_with_device

    monkeypatch.setitem(sys.modules, "cupy", cupy_like_grid_with_device((9, 0)).xp)
    block, refusal = gate.certified_scope_stack()
    assert block["capability"] == "9.0"
    assert refusal == ("the stack the NVIDIA kernels would run beside cannot be judged: "
                       "no certified stack is recorded for compute capability 9.0")
    identity = {axis: value for axis, value in STACK_86.items() if axis != "meep"}
    monkeypatch.setattr(fastpath, "_nvidia_stack_identity", lambda xp: dict(identity))
    monkeypatch.setitem(sys.modules, "cupy", cupy_like_grid_with_device((8, 6)).xp)
    block, refusal = gate.certified_scope_stack()
    assert block["capability"] == "8.6" and refusal is None
    monkeypatch.setattr(fastpath, "_nvidia_stack_identity",
                        lambda xp: dict(identity, cuda_driver="12090"))
    block, refusal = gate.certified_scope_stack()
    assert refusal == fastpath.certified_stack_refusal(block)
    assert refusal.endswith("CUDA driver 12090 is not the certified 12030, 13040"), refusal


def test_the_admitted_leg_precheck_records_the_stack_and_refuses_in_rung_4cs_words(
        monkeypatch):
    import gate_dispatch_fused_route as gate
    from meep_gpu.test_dispatch_contract import cupy_like_grid_with_device

    identity = {axis: value for axis, value in STACK_86.items() if axis != "meep"}
    monkeypatch.setattr(fastpath, "_nvidia_stack_identity", lambda xp: dict(identity))
    monkeypatch.setitem(sys.modules, "cupy", cupy_like_grid_with_device((8, 6)).xp)
    provenance = {}
    assert gate.admitted_leg_stack_refusal(provenance) is None
    assert provenance["certified_stack"]["capability"] == "8.6"
    assert provenance["certified_stack"]["certified"] is True
    head = (f"REFUSING: this leg runs under {fastpath.HARDWARE_SWITCH}="
            f"{fastpath.HARDWARE_CERTIFIED} and must dispatch, and that scope would refuse "
            "every case on this host: ")
    monkeypatch.setattr(fastpath, "_nvidia_stack_identity",
                        lambda xp: dict(identity, cuda_driver="12090"))
    provenance = {}
    refusal = gate.admitted_leg_stack_refusal(provenance)
    assert refusal == head + fastpath.certified_stack_refusal(provenance["certified_stack"])
    assert refusal.endswith("CUDA driver 12090 is not the certified 12030, 13040"), refusal
    monkeypatch.setitem(sys.modules, "cupy", cupy_like_grid_with_device((9, 0)).xp)
    provenance = {}
    assert gate.admitted_leg_stack_refusal(provenance) == head + (
        "the stack the NVIDIA kernels would run beside cannot be judged: no certified "
        "stack is recorded for compute capability 9.0")
    assert provenance["certified_stack"]["capability"] == "9.0"


class _PastThePrecheck(Exception):
    """Raised where the route gate reads the policy stamp, the step after the precheck."""


def _admitted_leg_main(monkeypatch, tmp_path, *extra, stack):
    """Run the route gate's ``main`` as the admitted-policy (flush) leg on the Triton table
    up to the step after its stack precheck, with the card, the counters, the provenance
    and the policy install stubbed (the real install drives the host FPU), and
    ``certified_scope_stack`` returning ``stack``. Returns the gate module, the leg
    directory and what the step after the precheck saw."""
    import gate_dispatch_fused_route as gate
    import meep_gpu

    for name in (fastpath.HARDWARE_SWITCH, fastpath.RETIRED_HARDWARE_SWITCH):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    for name in ("BACKEND", "TABLE_PREFERENCE", "ADMITTED_POLICY_INSTALLED",
                 "UNCERTIFIED_POLICY_INSTALLED", "ENVELOPE_CASE", "ENVELOPE_WITHHOLD"):
        monkeypatch.setattr(gate, name, getattr(gate, name))
    for name in ("PREFER_GPU", "_PROGRESS_PATH"):
        monkeypatch.setattr(gate.e2e, name, getattr(gate.e2e, name))

    class Counter:
        def install(self):
            return None

    seen = {}

    def policy_stamp():
        seen["scope"] = gate.os.environ.get(fastpath.HARDWARE_SWITCH)
        raise _PastThePrecheck

    monkeypatch.setattr(gate.e2e, "nvidia_host_refusal", lambda name: None)
    monkeypatch.setattr(gate, "card_capability", lambda gpu_id: "8.6")
    monkeypatch.setattr(gate, "installed_triton_version", lambda: "3.1.0")
    monkeypatch.setattr(gate, "uncertified_card_refusal", lambda *args: None)
    monkeypatch.setattr(gate.e2e, "TritonLaunchCounter", Counter)
    monkeypatch.setattr(gate, "_provenance", lambda gpu_id: {})
    monkeypatch.setitem(sys.modules, "meep", types.ModuleType("meep"))
    monkeypatch.setattr(meep_gpu, "install_subnormal_policy", lambda policy: {"policy": policy})
    monkeypatch.setattr(gate.e2e, "_policy_stamp", policy_stamp)
    monkeypatch.setattr(gate, "certified_scope_stack", stack)
    assert gate.policy_leg_role("flush", "triton") == "admitted"
    out = tmp_path / "flush"
    monkeypatch.setattr(sys, "argv", ["gate_dispatch_fused_route.py", "--out", str(out),
                                      "--install-policy", "flush", *extra])
    return gate, out, seen


def test_the_route_gate_refuses_an_admitted_leg_off_its_stack_before_its_first_case(
        monkeypatch, capsys, tmp_path):
    block = {"capability": "8.6", "certified": False}
    gate, out, seen = _admitted_leg_main(
        monkeypatch, tmp_path, stack=lambda: (block, "the stack is not the certified one"))
    assert gate.main() == 4
    printed = capsys.readouterr().out
    assert ("REFUSING: this leg runs under MEEP_GPU_HARDWARE=certified and must dispatch, "
            "and that scope would refuse every case on this host: the stack is not the "
            "certified one") in printed, printed
    assert seen == {}, "the leg went on past the refusal"
    assert not (out / "provenance.json").exists()


def test_the_route_gate_runs_an_admitted_leg_on_its_stack_under_the_certified_scope(
        monkeypatch, tmp_path):
    block = {"capability": "8.6", "certified": True}
    gate, _, seen = _admitted_leg_main(monkeypatch, tmp_path, stack=lambda: (block, None))
    with pytest.raises(_PastThePrecheck):
        gate.main()
    assert seen == {"scope": fastpath.HARDWARE_CERTIFIED}


def test_a_smoke_admitted_leg_is_not_judged_on_the_stack(monkeypatch, tmp_path):
    """A smoke run lifts the NumPy reference, which consults no kernel table, and may run
    on a host with no CuPy: the precheck is not asked."""
    def must_not_be_asked():
        raise AssertionError("a smoke run asked for the certified scope's stack verdict")

    gate, _, seen = _admitted_leg_main(monkeypatch, tmp_path, "--smoke",
                                       stack=must_not_be_asked)
    with pytest.raises(_PastThePrecheck):
        gate.main()
    assert seen == {"scope": fastpath.HARDWARE_CERTIFIED}
