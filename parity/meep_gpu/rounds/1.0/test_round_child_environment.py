"""The environments the 1.0 round's NVIDIA drivers give their children (host-only).

``run_lane.py`` starts every gate, fleet, route, toolchain and record child from the calling
shell with every package switch removed by prefix and the round's own set on top, and every
licence leg from nothing (``env -i``); ``fleet_launch.py`` hands the fleet driver an
environment it has checked. These tests pin what reaches a child: no ``MEEP_GPU_*`` or
``TRIDENT_FDTD_*`` variable the round did not set, whatever its name; ``CUPY_ACCELERATORS``
empty in every CUDA child but the licence legs, which the licence verdict refuses it in;
the user site directory off everywhere, and a lane or fleet refused by name before it
starts a gate when the interpreter keeps it on; the Triton launcher's toolchain in every
child on Linux (``CC`` and ``TRITON_LIBCUDA_PATH``: the caller's, else the reference
environment's own compiler and recorded libcuda link), and a lane, fleet or timing session
refused by name when a reference environment lacks its compiler. The timing ladder's legs
are not gated on the power source unless asked.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

ROUND = Path(__file__).resolve().parent
HARNESS = ROUND.parent.parent
for _path in (HARNESS, ROUND):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cert8  # noqa: E402
import fleet_launch  # noqa: E402
import legs  # noqa: E402
import run_lane  # noqa: E402
from recut_driver_dispatch_record import (  # noqa: E402
    LICENCE_ENVIRONMENT_ALLOWED, LICENCE_ENVIRONMENT_REFUSED_NAMES,
    LICENCE_ENVIRONMENT_REFUSED_PREFIXES)

#: A shell that exports switches no fixed list named.
SHELL = {"MEEP_GPU_SUBNORMAL_INSTALL": "0", "MEEP_GPU_WARM": "0", "MEEP_GPU_FUSED": "1",
         "MEEP_GPU_HD_GATE_MAX_CELLS": "10", "TRIDENT_FDTD_FUSED": "1",
         "MEEP_GPU_CORPUS_ROOT": "/elsewhere/python", "CUPY_CACHE_DIR": "/c",
         "TRITON_CACHE_DIR": "/t", "CUPY_ACCELERATORS": "cub", "OMP_NUM_THREADS": "8",
         "PATH": "/usr/bin:/bin", "HOME": "/h"}


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "meep" / "python"
    for name in ("examples", "tests"):
        (root / name).mkdir(parents=True)
    return root


def _lane(tmp_path: Path, monkeypatch, python: str = sys.executable, corpus: str = "",
          stub: str = "") -> run_lane.Lane:
    for name in list(os.environ):
        if name.startswith(run_lane.SCRUB_PREFIXES) or name in run_lane.SCRUB_NAMES:
            monkeypatch.delenv(name)
    # The lane sets it; a test process that exports it must not hide a lane that does not.
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    for name, value in SHELL.items():
        monkeypatch.setenv(name, value)
    args = argparse.Namespace(lane_dir=str(tmp_path / "lane"), repo=str(tmp_path),
                              results=str(tmp_path / "results"), python=python, gpu=2,
                              subs="", libcuda_stub=stub, corpus_root=corpus,
                              min_timeout=1800, timeout_factor=4, heartbeat=120)
    return run_lane.Lane(args)


def _fake_python(tmp_path: Path, answer: str) -> str:
    """An interpreter that answers the user-site probe with ``answer``."""
    path = tmp_path / "bin" / "python"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho {answer}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def test_a_lane_child_inherits_no_package_switch_whatever_its_name(tmp_path, monkeypatch):
    root = _corpus(tmp_path)
    env = _lane(tmp_path, monkeypatch, corpus=str(root), stub="/stub").base_env()
    assert {k for k in env if k.startswith(("MEEP_GPU_", "TRIDENT_FDTD_"))} == {
        "MEEP_GPU_CORPUS_ROOT"}
    assert env["MEEP_GPU_CORPUS_ROOT"] == str(root)
    assert "CUPY_CACHE_DIR" not in env and "TRITON_CACHE_DIR" not in env
    assert env["CUPY_ACCELERATORS"] == "" and env["PYTHONNOUSERSITE"] == "1"
    assert all(env[name] == "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
    assert env["CUDA_VISIBLE_DEVICES"] == "2" and env["TRITON_LIBCUDA_PATH"] == "/stub"
    assert env["HOME"] == "/h" and env["PATH"] == "/usr/bin:/bin"


def test_every_lane_step_starts_from_the_scrubbed_environment(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch)
    seen = []
    monkeypatch.setattr(lane, "run_child",
                        lambda argv, env, log, timeout, label: seen.append(env) or
                        {"rc": 0, "timed_out": False, "elapsed_s": 0.0})
    leg = tmp_path / "results" / "campaign" / "leg"
    lane.step_gate({"id": "g/keep", "policy": "keep", "script": "gate_cuda_x.py", "args": [],
                    "artifact": str(leg / "gate.json"), "leg_dir": str(leg), "expect_s": 1})
    lane.step_toolchain({"capability": "8.6"})
    for env in seen:
        assert not [k for k in env if k.startswith("TRIDENT_FDTD_")]
        assert not {"MEEP_GPU_SUBNORMAL_INSTALL", "MEEP_GPU_WARM", "MEEP_GPU_FUSED",
                    "MEEP_GPU_HD_GATE_MAX_CELLS"} & set(env)
        assert env["CUPY_ACCELERATORS"] == "" and env["PYTHONNOUSERSITE"] == "1"
    assert seen[0]["MEEP_GPU_DISPATCH"] == "0"


def test_a_licence_leg_holds_nothing_the_licence_verdict_refuses(tmp_path, monkeypatch):
    env = _lane(tmp_path, monkeypatch, stub="/stub").licence_env("ship", tmp_path / "ship")
    shaping = sorted(name for name in env if name not in LICENCE_ENVIRONMENT_ALLOWED and (
        name.startswith(LICENCE_ENVIRONMENT_REFUSED_PREFIXES)
        or name in LICENCE_ENVIRONMENT_REFUSED_NAMES))
    assert shaping == []
    assert "CUPY_ACCELERATORS" not in env and env["PYTHONNOUSERSITE"] == "1"
    assert env["MEEP_GPU_DISPATCH_LOG"] == str(tmp_path / "ship_dispatch_log.jsonl")
    assert env["OMP_NUM_THREADS"] == "1" and "HOME" in env and env["TRITON_LIBCUDA_PATH"] == "/stub"
    leaked = [name for name, value in SHELL.items() if name != "HOME" and env.get(name) == value]
    assert leaked == []
    assert "MEEP_GPU_CORPUS_ROOT" not in env and "CUPY_ACCELERATORS" not in env


def test_a_licence_leg_runs_in_the_licence_environment(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, stub="/stub")
    seen = []
    monkeypatch.setattr(lane, "run_child",
                        lambda argv, env, log, timeout, label: seen.append((argv, env)) or
                        {"rc": 0, "timed_out": False, "elapsed_s": 0.0})
    out = tmp_path / "results" / "e2e" / "ship"
    lane.step_e2e({"id": "e2e/ship", "kind": "e2e", "leg": "ship", "out": str(out),
                   "args": ["--enable-from-default"], "expect_s": 1})
    (argv, env), = seen
    assert argv[2].endswith("gate_dispatch_end_to_end.py")
    assert env == lane.licence_env("ship", out)
    assert "CUPY_ACCELERATORS" not in env and env["PYTHONNOUSERSITE"] == "1"
    assert [name for name, value in SHELL.items() if name != "HOME" and env.get(name) == value] == []


def test_the_round_s_interpreter_keeps_the_user_site_off_in_both_environments(tmp_path,
                                                                            monkeypatch):
    assert _lane(tmp_path, monkeypatch).user_site_problems() == []


def test_an_interpreter_that_keeps_the_user_site_on_is_named(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch, python=_fake_python(tmp_path, "True"))
    problems = lane.user_site_problems()
    assert len(problems) == 2
    assert "gate children" in problems[0] and "licence children" in problems[1]
    assert all("site.ENABLE_USER_SITE: True" in problem for problem in problems)


def test_such_a_lane_stops_before_it_takes_a_gpu(tmp_path, monkeypatch):
    for name in [n for n in os.environ if n.startswith(run_lane.SCRUB_PREFIXES)]:
        monkeypatch.delenv(name)
    # main() installs the lane's signal handlers; they must not outlive this test.
    monkeypatch.setattr(run_lane.signal, "signal", lambda *args: None)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"id": "t", "kind": "toolchain", "capability": "8.6"}]))
    lane_dir = tmp_path / "lane"
    # A lane that skipped the check would wait for a GPU: one poll, then it gives up.
    rc = run_lane.main(["--plan", str(plan), "--lane-dir", str(lane_dir), "--repo",
                        str(tmp_path), "--results", str(tmp_path / "results"), "--lock-dir",
                        str(tmp_path / "locks"), "--poll", "1", "--lock-patience", "1",
                        "--python", _fake_python(tmp_path, "True")])
    assert rc == 3
    assert "user site directory" in (lane_dir / "STOPPED").read_text()
    assert not (lane_dir / "steps.jsonl").exists() and not (tmp_path / "locks").exists()


def test_the_toolchain_check_refuses_the_user_site_on_the_card():
    assert "site.ENABLE_USER_SITE" in run_lane.TOOLCHAIN_CHECK
    assert "user site directory is enabled" in run_lane.TOOLCHAIN_CHECK


def test_no_timing_leg_is_gated_on_the_power_source_by_default():
    import timing_ladder  # noqa: PLC0415 - the harness module, beside the round

    assert timing_ladder.PARAM_BY_NAME["REQUIRE_AC_POWER"].default == 0


# ---------------------------------------------------------------------------------------
# fleet_launch.py
# ---------------------------------------------------------------------------------------

def test_the_fleet_is_refused_an_exported_switch_or_the_user_site_by_name():
    assert fleet_launch.environment_problems({"MEEP_GPU_CORPUS_ROOT": "/c", "PATH": "/b"},
                                             False) == []
    problems = fleet_launch.environment_problems(
        {"MEEP_GPU_SUBNORMAL_INSTALL": "0", "TRIDENT_FDTD_X": "1"}, True)
    assert len(problems) == 2 and "user site directory" in problems[0]
    assert "['MEEP_GPU_SUBNORMAL_INSTALL', 'TRIDENT_FDTD_X']" in problems[1]


def _stub_fleet(monkeypatch):
    seen = {}
    stub = types.SimpleNamespace(GATES={}, subprocess=None, log=lambda message: None)

    def main(argv):
        seen["env"] = dict(os.environ)
        seen["argv"] = list(argv)
        return 0

    stub.main = main
    monkeypatch.setitem(sys.modules, "drive_triton_weld_gates", stub)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return seen


def test_every_fleet_gate_starts_with_the_user_site_off_and_no_accelerator(tmp_path,
                                                                         monkeypatch):
    seen = _stub_fleet(monkeypatch)
    for name in [n for n in os.environ if n.startswith(fleet_launch.SHAPING_PREFIXES)]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("CUPY_ACCELERATORS", "cub")
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    monkeypatch.setattr(fleet_launch.site, "ENABLE_USER_SITE", False)
    assert fleet_launch.main(["--repo", str(tmp_path), "--", "--policy", "keep"]) == 0
    assert seen["env"]["PYTHONNOUSERSITE"] == "1" and seen["env"]["CUPY_ACCELERATORS"] == ""
    assert seen["argv"] == ["--policy", "keep"]


def test_a_refused_fleet_starts_no_driver(tmp_path, monkeypatch):
    seen = _stub_fleet(monkeypatch)
    monkeypatch.setenv("MEEP_GPU_FUSED", "1")
    monkeypatch.setattr(fleet_launch.site, "ENABLE_USER_SITE", False)
    assert fleet_launch.main(["--repo", str(tmp_path), "--", "--policy", "keep"]) == 3
    assert "env" not in seen
    monkeypatch.delenv("MEEP_GPU_FUSED")
    monkeypatch.setattr(fleet_launch.site, "ENABLE_USER_SITE", True)
    assert fleet_launch.main(["--repo", str(tmp_path), "--", "--policy", "keep"]) == 3
    assert "env" not in seen


# ---------------------------------------------------------------------------------------
# The Triton launcher's toolchain (legs.launcher_toolchain), on Linux
# ---------------------------------------------------------------------------------------

def _reference_prefix(tmp_path: Path, compiler: bool = True, libcuda: bool = True,
                      record: bool = True) -> Path:
    """A reference environment as its build leaves it: the build record, the compiler
    Triton's launchers need, the libcuda link and its record, and an interpreter that
    answers the sys.prefix probe with this prefix and the user-site probe with False."""
    prefix = tmp_path / "env"
    records = prefix / "share" / "meep-gpu-build"
    records.mkdir(parents=True)
    (prefix / "bin").mkdir()
    if record:
        (records / "reference_build.json").write_text("{}")
    python = prefix / "bin" / "python"
    python.write_text(f'#!/bin/sh\ncase "$*" in *sys.prefix*) echo {prefix} ;; *) echo False ;; '
                      f'esac\n')
    python.chmod(python.stat().st_mode | stat.S_IXUSR)
    if compiler:
        cc = prefix.joinpath(*legs.LAUNCHER_COMPILER)
        cc.write_text("#!/bin/sh\nexit 0\n")
        cc.chmod(cc.stat().st_mode | stat.S_IXUSR)
    if libcuda:
        (prefix / "lib" / "meep-gpu-libcuda").mkdir(parents=True)
        (prefix / "lib" / "meep-gpu-libcuda" / "libcuda.so").write_text("")
        (records / legs.LIBCUDA_RECORD).write_text(json.dumps(
            {"directory": "lib/meep-gpu-libcuda", "target": "/driver/libcuda.so.1"}))
    return prefix


@pytest.fixture
def on_linux(monkeypatch):
    """The toolchain rule as it runs on a Linux GPU host, whatever this host is; the
    caller's CC, TRITON_LIBCUDA_PATH and LD_LIBRARY_PATH unset, and restored afterwards."""
    monkeypatch.setattr(legs, "LAUNCHER_PLATFORMS", (sys.platform,))
    for name in ("CC", "TRITON_LIBCUDA_PATH", "LD_LIBRARY_PATH"):
        monkeypatch.setenv(name, "x")      # recorded, so whatever a test sets is undone
        monkeypatch.delenv(name)


def test_the_toolchain_is_the_reference_environment_s_unless_the_caller_set_it(tmp_path,
                                                                              on_linux):
    prefix = _reference_prefix(tmp_path)
    cc = str(prefix / "bin" / "x86_64-conda-linux-gnu-cc")
    lib = str(prefix / "lib" / "meep-gpu-libcuda")
    own = legs.launcher_toolchain(str(prefix), {})
    assert (own["cc"], own["libcuda"], own["problems"]) == (cc, lib, [])
    assert own["cc_source"] == own["libcuda_source"] == "reference environment"
    caller = legs.launcher_toolchain(str(prefix), {"CC": "/opt/cc", "TRITON_LIBCUDA_PATH": "/t"})
    assert (caller["cc"], caller["libcuda"]) == ("/opt/cc", "/t")
    stub = legs.launcher_toolchain(str(prefix), {"TRITON_LIBCUDA_PATH": "/t", "CC": ""}, "/s")
    assert (stub["cc"], stub["libcuda"]) == (cc, "/s")     # an empty CC is unset
    env = legs.launcher_environment({"LD_LIBRARY_PATH": f"/a:{lib}"}, own)
    assert env == {"CC": cc, "TRITON_LIBCUDA_PATH": lib, "LD_LIBRARY_PATH": f"{lib}:/a"}


def test_a_reference_environment_without_its_compiler_or_link_is_named(tmp_path, on_linux):
    prefix = _reference_prefix(tmp_path, compiler=False)
    (prefix / "lib" / "meep-gpu-libcuda" / "libcuda.so").unlink()
    found = legs.launcher_toolchain(str(prefix), {})
    assert found["cc"] is None and found["libcuda"] is None and len(found["problems"]) == 2
    assert "has no C compiler at" in found["problems"][0]
    assert "x86_64-conda-linux-gnu-cc" in found["problems"][0]
    assert "holds no libcuda.so" in found["problems"][1]
    # The caller's values need nothing from the environment.
    assert legs.launcher_toolchain(str(prefix), {"CC": "cc"}, "/s")["problems"] == []


def test_another_interpreter_or_host_gets_only_what_the_caller_set(tmp_path, on_linux,
                                                                    monkeypatch):
    plain = _reference_prefix(tmp_path, compiler=False, libcuda=False, record=False)
    found = legs.launcher_toolchain(str(plain), {"CC": "cc"})
    assert (found["cc"], found["libcuda"], found["problems"]) == ("cc", None, [])
    assert "not a reference environment" in found["notes"][0]
    monkeypatch.setattr(legs, "LAUNCHER_PLATFORMS", ("no-such-platform",))
    found = legs.launcher_toolchain(str(_reference_prefix(tmp_path / "b", compiler=False)),
                                    {}, "/s")
    assert (found["cc"], found["libcuda"], found["problems"]) == (None, "/s", [])


def test_every_lane_child_builds_launchers_with_the_environment_s_compiler(tmp_path,
                                                                          monkeypatch,
                                                                          on_linux):
    prefix = _reference_prefix(tmp_path)
    cc = str(prefix / "bin" / "x86_64-conda-linux-gnu-cc")
    lib = str(prefix / "lib" / "meep-gpu-libcuda")
    lane = _lane(tmp_path, monkeypatch, python=str(prefix / "bin" / "python"))
    for env in (lane.base_env(), lane.licence_env("ship", tmp_path / "ship")):
        assert env["CC"] == cc and env["TRITON_LIBCUDA_PATH"] == lib
        assert env["LD_LIBRARY_PATH"].split(":")[0] == lib
    monkeypatch.setenv("CC", "/opt/cc")
    lane = _lane(tmp_path, monkeypatch, python=str(prefix / "bin" / "python"), stub="/stub")
    for env in (lane.base_env(), lane.licence_env("ship", tmp_path / "ship")):
        assert env["CC"] == "/opt/cc" and env["TRITON_LIBCUDA_PATH"] == "/stub"


def test_a_lane_whose_environment_lacks_its_compiler_stops_before_it_takes_a_gpu(
        tmp_path, monkeypatch, on_linux):
    for name in [n for n in os.environ if n.startswith(run_lane.SCRUB_PREFIXES)]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(run_lane.signal, "signal", lambda *args: None)
    prefix = _reference_prefix(tmp_path, compiler=False)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"id": "t", "kind": "toolchain", "capability": "8.6"}]))
    lane_dir = tmp_path / "lane"
    rc = run_lane.main(["--plan", str(plan), "--lane-dir", str(lane_dir), "--repo",
                        str(tmp_path), "--results", str(tmp_path / "results"), "--lock-dir",
                        str(tmp_path / "locks"), "--poll", "1", "--lock-patience", "1",
                        "--python", str(prefix / "bin" / "python")])
    assert rc == 3
    stopped = (lane_dir / "STOPPED").read_text()
    assert "has no C compiler at" in stopped and "x86_64-conda-linux-gnu-cc" in stopped
    assert not (lane_dir / "steps.jsonl").exists() and not (tmp_path / "locks").exists()


def test_every_fleet_gate_inherits_the_toolchain_and_a_missing_compiler_is_refused(
        tmp_path, monkeypatch, capsys, on_linux):
    for name in [n for n in os.environ if n.startswith(fleet_launch.SHAPING_PREFIXES)]:
        monkeypatch.delenv(name)
    monkeypatch.setattr(fleet_launch.site, "ENABLE_USER_SITE", False)
    for name in ("PYTHONNOUSERSITE", "CUPY_ACCELERATORS"):
        monkeypatch.setenv(name, "x")
    prefix = _reference_prefix(tmp_path)
    monkeypatch.setattr(fleet_launch.sys, "prefix", str(prefix))
    seen = _stub_fleet(monkeypatch)
    assert fleet_launch.main(["--repo", str(tmp_path), "--", "--policy", "keep"]) == 0
    assert seen["env"]["CC"] == str(prefix / "bin" / "x86_64-conda-linux-gnu-cc")
    assert seen["env"]["TRITON_LIBCUDA_PATH"] == str(prefix / "lib" / "meep-gpu-libcuda")
    bare = _reference_prefix(tmp_path / "bare", compiler=False)
    monkeypatch.setattr(fleet_launch.sys, "prefix", str(bare))
    monkeypatch.delenv("CC")
    seen = _stub_fleet(monkeypatch)
    assert fleet_launch.main(["--repo", str(tmp_path), "--", "--policy", "keep"]) == 3
    assert "env" not in seen and "has no C compiler at" in capsys.readouterr().out


def test_a_timing_session_exports_the_toolchain_or_is_refused(tmp_path, on_linux):
    prefix = _reference_prefix(tmp_path)
    found = cert8.export_launcher_toolchain(str(prefix / "bin" / "python"))
    assert os.environ["CC"] == found["cc"] == str(prefix / "bin" / "x86_64-conda-linux-gnu-cc")
    assert os.environ["TRITON_LIBCUDA_PATH"] == str(prefix / "lib" / "meep-gpu-libcuda")
    assert "Triton launcher toolchain: CC=" in found["line"]
    del os.environ["CC"]
    bare = _reference_prefix(tmp_path / "bare", compiler=False)
    with pytest.raises(cert8.Refused) as refused:
        cert8.export_launcher_toolchain(str(bare / "bin" / "python"))
    assert "has no C compiler at" in refused.value.message and "CC" not in os.environ
