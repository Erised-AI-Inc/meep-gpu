"""The host tools a certification round on a new GPU runs: the preflight, the chunk runner,
the card log and the results packer (``docs/development/certify-a-new-gpu.md``).

Host-only: no GPU, no MEEP, no network. The machine the preflight reads is a fake that
answers each question from a table; the round's own tools the chunk runner starts are a fake
runner that records each command and writes what the real tool would leave; ``nvidia-smi``
is a stand-in script on ``PATH`` where a real child process reads it.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

ROUND = Path(__file__).resolve().parent
if str(ROUND) not in sys.path:
    sys.path.insert(0, str(ROUND))

import gpu_log  # noqa: E402
import pack_results  # noqa: E402
import preflight  # noqa: E402
import run_chunks  # noqa: E402

EXPECT = "0123456789abcdef0123456789abcdef01234567"


def fake_smi(directory: Path, body: str) -> Path:
    """A stand-in ``nvidia-smi`` on ``directory``: a short shell script that runs ``body`` (Python)
    with this interpreter, so no interpreter path goes on a ``#!`` line, whose length the kernel
    bounds."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nvidia_smi_stand_in.py").write_text(body)
    smi = directory / "nvidia-smi"
    smi.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$(dirname "$0")/nvidia_smi_stand_in.py" "$@"\n')
    smi.chmod(0o755)
    return smi


# ------------------------------------------------------------------- preflight -----
class FakeHost(preflight.Host):
    """A machine read from a table: the answers a native Linux host with two idle H200s gives."""

    def __init__(self, tmp: Path, **change) -> None:
        super().__init__(environ=change.pop("environ", {"PATH": "/usr/bin:/bin"}))
        self.prefix_dir = tmp / "env"
        records = self.prefix_dir / "share" / "meep-gpu-build"
        records.mkdir(parents=True)
        (records / "reference_build.json").write_text("{}")
        libcuda = change.pop("libcuda_directory", "lib/meep-gpu-libcuda")
        (records / "libcuda.json").write_text(json.dumps({"directory": libcuda}))
        if libcuda:
            (self.prefix_dir / libcuda).mkdir(parents=True)
            (self.prefix_dir / libcuda / "libcuda.so").write_text("")
        self.values = {
            "system": "Linux", "machine": "x86_64", "glibc": "2.35", "wsl2": False,
            "free_gib": 900.0, "cpus": 16, "driver": 12030, "head": EXPECT, "status": "",
            "cards": [("0", "NVIDIA H200", "9.0"), ("1", "NVIDIA H200", "9.0")],
            "mig": "Disabled", "packages": "", "tools": True,
            "libcuda": str(tmp / "driver_without_libcuda_so" / "libcuda.so.1"),
            "files": {"/proc/meminfo": "MemTotal:       2113600000 kB\n",
                      "/proc/mounts": "/dev/sda1 / ext4 rw 0 0\n",
                      "/proc/self/cgroup": "0::/\n",
                      "/etc/os-release": 'PRETTY_NAME="Ubuntu 22.04.4 LTS"\n'},
            "idle": None,
        }
        self.values.update(change)
        self.idle_asked: List[Tuple[Sequence[int], int]] = []

    def system(self): return self.values["system"]
    def machine(self): return self.values["machine"]
    def glibc(self): return self.values["glibc"]
    def wsl2(self): return self.values["wsl2"]
    def read(self, path): return self.values["files"].get(path)
    def free_gib(self, path): return self.values["free_gib"]
    def cpus(self): return self.values["cpus"]
    def driver_version(self): return self.values["driver"]
    def prefix(self): return self.prefix_dir

    def which(self, name):
        if name == "reg.exe":
            return None
        return f"/usr/bin/{name}" if self.values["tools"] else None

    def run(self, argv, cwd=None):
        argv = list(argv)
        if argv[:1] == ["git"]:
            if argv[-1] == "HEAD":
                return 0, self.values["head"] + "\n"
            if argv[-1] == "HEAD^{tree}":
                return 0, "f" * 40 + "\n"
            return 0, self.values["status"]
        if argv[-1] == "-p":
            return 0, f"\tlibcuda.so.1 (libc6,x86-64) => {self.values['libcuda']}\n"
        if argv[:1] == ["dpkg"]:
            return 0, self.values["packages"]
        if any(a.startswith("--query-gpu=index,name") for a in argv):
            return 0, "".join(f"{i}, {name}, {cap}, 143771 MiB, 570.86.15\n"
                              for i, name, cap in self.values["cards"])
        if any(a.startswith("--query-gpu=mig") for a in argv):
            return 0, self.values["mig"] + "\n"
        return 127, "not faked"

    def idle_reports(self, cards, poll):
        self.idle_asked.append((list(cards), poll))
        if self.values["idle"] is not None:
            return self.values["idle"]
        return {card: (0, f"CARD {card}: NVIDIA H200\nLANE: taken by the idle rule\n"
                          f"FLEET: taken: GPU {card} empty\n"
                          f"DECISION: TAKEN: card {card} is taken under the idle rule\n")
                for card in cards}


def run_preflight(tmp: Path, *extra: str, **change) -> Tuple[int, List[str], FakeHost]:
    host = FakeHost(tmp, **change)
    lines: List[str] = []
    say = preflight.Report(lines.append)
    ap_args = ["--capability", "9.0", "--expect", EXPECT, "--poll", "0", "--repo", str(tmp), *extra]
    rc = preflight.run(preflight.build_parser().parse_args(ap_args), host, say)
    return rc, lines, host


def fails(lines: List[str]) -> List[str]:
    return [line for line in lines if line.startswith("FAIL")]


def test_a_ready_native_host_passes_and_names_the_cards_to_run_on(tmp_path):
    rc, lines, host = run_preflight(tmp_path)
    assert rc == 0, fails(lines)
    assert lines[-1] == "PREFLIGHT OK (0 FAIL)"
    assert any("the cards to run on: 0,1" in line for line in lines)
    assert any(line.startswith("PASS  the driver supplies CUDA 12.3") for line in lines)
    assert host.idle_asked == [([0, 1], 0)], "every card is read at once, at the poll asked"
    assert sum("is taken under the idle rule" in line for line in lines) == 2


@pytest.mark.parametrize("change, needle", [
    ({"head": "f" * 40}, "the clone's HEAD is"),
    ({"status": " M meep_gpu/fastpath.py\n"}, "changed or untracked path"),
    ({"cards": [("0", "NVIDIA RTX A6000", "8.6")]}, "compute capability '8.6', not 9.0"),
    ({"mig": "Enabled"}, "MIG slices"),
    ({"glibc": "2.17"}, "glibc 2.17"),
    ({"free_gib": 12.0}, "free disk under the clone: 12 GiB"),
    ({"libcuda_directory": None}, "build_meep_reference_record.py link-libcuda"),
    ({"free_gib": 30.0}, "still needs about 40 GiB"),
    ({"environ": {"PATH": "/usr/bin", "MEEP_GPU_DISPATCH": "0"}}, "exports MEEP_GPU_DISPATCH"),
    ({"idle": {0: (1, "LANE: refused (1 compute process(es) listed on the card)\n"
                      "DECISION: REFUSED: card 0 is not taken; the lane would wait"),
               1: (0, "DECISION: TAKEN: card 1 is taken under the idle rule")}},
     "card 0 is not taken"),
    ({"idle": {0: (1, "Traceback"), 1: (0, "DECISION: TAKEN: card 1")}},
     "the card rule did not decide"),
])
def test_each_thing_the_round_would_refuse_is_a_named_fail(tmp_path, change, needle):
    rc, lines, _host = run_preflight(tmp_path, **change)
    assert rc == 1
    assert any(needle in line for line in fails(lines)), fails(lines)
    assert lines[-1].startswith("PREFLIGHT REFUSED")


def test_an_expect_that_is_not_a_commit_hash_is_refused(tmp_path):
    rc, lines, _host = run_preflight(tmp_path, "--expect", "PASTE-THE-COMMIT")
    assert rc == 1 and any("not a 40-character commit hash" in line for line in fails(lines))


def record_stack(repo: Path, capability: str, *drivers: str) -> None:
    """The clone's Triton ledger, holding ``capability``'s certified stack on ``drivers``."""
    path = repo.joinpath(*preflight.STACK_LEDGER)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({preflight.STACK_ENTRY: {preflight.STACK_BY_CAPABILITY: {
        capability: {"stack": {"cuda_driver": list(drivers)}}}}}))


def test_another_driver_than_the_capabilitys_warns_and_does_not_stop_the_gates(tmp_path):
    record_stack(tmp_path, "9.0", "12030")
    rc, lines, _host = run_preflight(tmp_path, driver=12080)
    assert rc == 0
    warn = [line for line in lines if line.startswith("WARN") and "12080" in line]
    assert warn and "CUDA 12.8" in warn[0] and "route stage" in warn[0]
    assert "not one compute capability 9.0 was certified on (12030)" in warn[0], warn
    assert "the bind adds it to the capability's certified drivers only if" in warn[0], warn


def test_the_capabilitys_own_driver_passes(tmp_path):
    record_stack(tmp_path, "9.0", "12080")
    rc, lines, _host = run_preflight(tmp_path, driver=12080)
    assert rc == 0
    assert any(line.startswith("PASS  the driver supplies CUDA 12.8 (driver version 12080), "
                               "one compute capability 9.0 was certified on (12080)")
               for line in lines), lines


@pytest.mark.parametrize("driver,level", [(12030, "PASS"), (13040, "PASS"), (12080, "WARN")])
def test_each_driver_a_capability_lists_passes_and_another_warns(tmp_path, driver, level):
    """A capability certified on two drivers: either passes, naming both; a third warns."""
    record_stack(tmp_path, "9.0", "12030", "13040")
    assert preflight.certified_drivers(tmp_path, "9.0") == (12030, 13040)
    rc, lines, _host = run_preflight(tmp_path, driver=driver)
    assert rc == 0
    line = [line for line in lines if line.startswith(f"{level}  the driver supplies")]
    assert line and "compute capability 9.0 was certified on (12030, 13040)" in line[0], lines


def test_a_new_capability_passes_on_any_driver_and_says_the_bind_records_it(tmp_path):
    rc, lines, _host = run_preflight(tmp_path, driver=12080)
    assert rc == 0
    passed = [line for line in lines if line.startswith("PASS  the driver supplies CUDA 12.8")]
    assert passed and "compute capability 9.0 has no certified stack yet" in passed[0], lines


def test_the_shipped_ledger_names_8_6s_drivers():
    """The 1.0.1 round's bind recorded 8.6 on the drivers its chunks ran on: 12030 (the RTX
    A6000's) and 13040 (the RTX 3090's)."""
    assert preflight.certified_drivers(preflight.REPO, "8.6") == (12030, 13040)
    assert preflight.certified_drivers(preflight.REPO, "9.9") is None


def test_a_missing_libcuda_link_is_a_warning_where_the_driver_ships_libcuda_so(tmp_path):
    driver = tmp_path / "driver"
    driver.mkdir()
    (driver / "libcuda.so.1").write_text("")
    (driver / "libcuda.so").write_text("")
    host = FakeHost(tmp_path / "h", libcuda_directory=None, libcuda=str(driver / "libcuda.so.1"))
    lines: List[str] = []
    preflight.check_environment(host, preflight.Report(lines.append))
    assert not fails(lines)
    assert any(line.startswith("WARN") and str(driver / "libcuda.so") in line for line in lines)


@pytest.mark.parametrize("module", [preflight, run_chunks, pack_results])
def test_each_tool_defaults_to_the_checkout_that_holds_it(module):
    assert (module.REPO / "parity" / "meep_gpu" / "rounds" / "1.0" / "display_card.py").is_file()
    assert (module.REPO / "meep_gpu").is_dir() and (module.REPO / "tools").is_dir()


def test_a_non_reference_interpreter_is_refused(tmp_path):
    host = FakeHost(tmp_path)
    (host.prefix_dir / "share" / "meep-gpu-build" / "reference_build.json").unlink()
    lines: List[str] = []
    preflight.check_environment(host, preflight.Report(lines.append))
    assert fails(lines) and "not a reference environment" in fails(lines)[0]


def test_wsl2_keeps_its_own_checks(tmp_path):
    files = dict(FakeHost(tmp_path / "a").values["files"])
    files["/proc/mounts"] = "C:\\134 /mnt/c 9p rw 0 0\n/dev/sdc / ext4 rw 0 0\n"
    rc, lines, _host = run_preflight(
        tmp_path / "b", wsl2=True, files=files, cards=[("0", "NVIDIA GeForce RTX 4080", "9.0")],
        libcuda="/usr/lib/wsl/lib/libcuda.so.1",
        packages="ii  nvidia-utils-550  550.1  amd64  NVIDIA driver support binaries\n")
    assert rc == 1
    assert any("Linux NVIDIA driver packages inside WSL2 (nvidia-utils-550)" in line
               for line in fails(lines))
    assert any(line.startswith("PASS  libcuda.so.1 in the linker cache: /usr/lib/wsl/lib/")
               for line in lines)
    assert any("Windows interop is off" in line for line in lines)


def test_a_clone_under_a_windows_drive_is_refused_under_wsl2(tmp_path):
    host = FakeHost(tmp_path, wsl2=True)
    host.values["files"]["/proc/mounts"] = "C:\\134 /mnt/c 9p rw 0 0\n/dev/sdc / ext4 rw 0 0\n"
    lines: List[str] = []
    preflight.check_disk(host, preflight.Report(lines.append), Path("/mnt/c/Users/x/meep-gpu"),
                         True, 60, 24)
    assert [line for line in fails(lines) if "under /mnt" in line]
    assert [line for line in fails(lines) if "'9p'" in line]


@pytest.mark.parametrize("listed, visible, asked, want", [
    ([0, 1, 2, 3], None, None, [0, 1, 2, 3]),
    ([0, 1], "0,1", None, [0, 1]),
    ([0, 1, 2, 3, 4, 5, 6, 7], "3,5", None, [3, 5]),
    ([0, 1, 2, 3], "0,1,2,3", None, [0, 1, 2, 3]),
    ([0, 1, 2, 3], "2,3", "1", [1]),
    ([0, 1], "4,5", None, [0, 1]),
])
def test_the_cards_come_from_the_job_as_the_scheduler_confines_it(listed, visible, asked, want):
    assert preflight.card_ids(listed, visible, asked)[0] == want


@pytest.mark.parametrize("listed, visible, asked, needle", [
    ([0], "GPU-5c1a6b2e-0000", None, "by UUID"),
    ([0, 1], "", None, "hides every card"),
    ([0, 1, 2], "5", None, "does not match"),
    ([0, 1], None, "2", "which nvidia-smi does not list"),
])
def test_a_mask_that_cannot_be_matched_is_refused(listed, visible, asked, needle):
    with pytest.raises(ValueError, match=needle):
        preflight.card_ids(listed, visible, asked)


def test_cards_only_prints_the_cards_for_a_batch_script(tmp_path):
    out: List[str] = []
    host = FakeHost(tmp_path, environ={"CUDA_VISIBLE_DEVICES": "0,1"})
    args = preflight.build_parser().parse_args(["--capability", "9.0", "--cards-only"])
    assert preflight.cards_only(args, host, out.append) == 0 and out == ["0,1"]
    out.clear()
    host = FakeHost(tmp_path / "b", environ={"CUDA_VISIBLE_DEVICES": "GPU-5c1a"})
    assert preflight.cards_only(args, host, out.append) == 1 and "by UUID" in out[0]


def test_the_memory_is_the_job_limit_when_it_is_smaller():
    meminfo = "MemTotal:       2113600000 kB\n"
    assert preflight.memory_gib(meminfo, ["68719476736\n"]) == 64
    assert preflight.memory_gib(meminfo, ["max\n"]) == pytest.approx(2113600000 / 2 ** 20)
    assert preflight.cgroup_memory_files("0::/slurm/uid_1/job_7\n") == [
        "/sys/fs/cgroup/slurm/uid_1/job_7/memory.max"]
    assert preflight.cgroup_memory_files("4:memory:/slurm/job_7\n1:cpu:/x\n") == [
        "/sys/fs/cgroup/memory/slurm/job_7/memory.limit_in_bytes"]
    assert [preflight.memory_needed_gib(n) for n in (1, 2, 4)] == [24, 36, 60]


def test_the_longest_mount_point_names_the_file_system():
    mounts = "/dev/a / ext4 rw 0 0\nnfs:/h /srv nfs4 rw 0 0\nlu /srv/scratch lustre rw 0 0\n"
    assert preflight.mount_type(Path("/srv/scratch/u/meep-gpu"), mounts) == "lustre"
    assert preflight.mount_type(Path("/srv/u"), mounts) == "nfs4"
    assert preflight.mount_type(Path("/opt/x"), mounts) == "ext4"


def test_the_preflight_runs_the_rounds_own_card_rule(tmp_path, monkeypatch):
    """The idle check is ``display_card.py`` itself, beside the preflight, as a child process
    that reads a stand-in ``nvidia-smi``: an idle card is taken, a busy one refused."""
    smi = fake_smi(tmp_path / "bin", '''import sys
args = " ".join(sys.argv[1:])
card = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--id=")), "0")
busy = card == "1"
if "query-compute-apps" in args:
    print("4242" if busy else "")
elif "memory.used,utilization.gpu" in args:
    print("2048, 97" if busy else "0, 0")
elif "display_attached" in args:
    print("No")
elif "display_active,display_mode" in args:
    print("Disabled, Disabled")
elif "name,compute_cap" in args:
    print("NVIDIA H200, 9.0, 143771 MiB")
''')
    monkeypatch.setenv("PATH", f"{smi.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    reports = preflight.Host().idle_reports([0, 1], 0)
    assert preflight.idle_decision(*reports[0])[0] == "PASS", reports[0]
    level, decision, _detail = preflight.idle_decision(*reports[1])
    assert level == "FAIL" and "card 1 is not taken" in decision, reports[1]


# ------------------------------------------------------------------ run_chunks -----
def write_index(path: Path, mode: str = "new", machine: str = "nvidia") -> Path:
    path.write_text(json.dumps({
        "schema": "meep_gpu.cert_chunks/2", "machine": machine, "device": "cc90",
        "device_mode": mode, "round": "2026-10-18_101", "chunks": [
            {"id": f"c{k}", "stamp": f"2026-10-18_101_cc90_c{k}", "units": []} for k in range(3)]}))
    return path


class FakeTools:
    """``round.py`` and ``legs.py`` as the runner calls them: each call recorded, each
    outcome from a table, and the files the real tools leave written."""

    def __init__(self, lanes: Path, check: str = "CHECK OK (0 problem(s))",
                 gates_rc: Optional[Dict[str, int]] = None) -> None:
        self.lanes = lanes
        self.calls: List[List[str]] = []
        self.check = check
        self.gates_rc = gates_rc or {}

    def __call__(self, argv: Sequence[str], cwd: Path, log: Optional[Path]) -> int:
        argv = list(argv)
        self.calls.append(argv)
        if "check" in argv and argv[2].endswith("legs.py"):
            log.write_text(f"... checks\n{self.check}\n")
            return 0 if self.check.startswith("CHECK OK") else 1
        if "status" in argv:
            home = Path(argv[argv.index("--home") + 1])
            if log is not None:
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("status\n")
            return 0 if (home / "GATES_DONE").exists() else 1
        if argv[4:6] == ["run", "gates"]:
            chunk = argv[argv.index("--chunk") + 1]
            rc = self.gates_rc.get(chunk, 0)
            if rc == 0:
                home = self.lanes / f"2026-10-18_101_cc90_{chunk}"
                home.mkdir(parents=True, exist_ok=True)
                (home / "GATES_DONE").write_text("")
            return rc
        raise AssertionError(f"unexpected call {argv}")

    def gates(self) -> List[List[str]]:
        return [c for c in self.calls if c[4:6] == ["run", "gates"]]

    def checks(self) -> List[List[str]]:
        return [c for c in self.calls if c[2].endswith("legs.py")]


def chunks_main(tmp: Path, tools: FakeTools, *argv: str, mode: str = "new") -> int:
    index = write_index(tmp / "INDEX_cc90.json", mode)
    (tmp / "repo" / "parity" / "meep_gpu" / "results").mkdir(parents=True, exist_ok=True)
    return run_chunks.main(["--index", str(index), "--lanes", str(tools.lanes), "--lock-dir",
                            str(tmp / "locks"), "--expect", EXPECT, "--repo", str(tmp / "repo"),
                            "--python", "/env/bin/python", *argv], run=tools)


def test_a_new_device_index_runs_with_new_capability_on_the_cards_named(tmp_path):
    tools = FakeTools(tmp_path / "lanes")
    rc = chunks_main(tmp_path, tools, "--gpu-ids", "2,3", "--max-locks", "4", "c0", "c1")
    assert rc == 0
    assert len(tools.checks()) == 2 and all("--new-capability" in c for c in tools.checks())
    gates = tools.gates()
    assert [g[g.index("--chunk") + 1] for g in gates] == ["c0", "c1"]
    for g in gates:
        assert g[g.index("--capability") + 1] == "9.0" and "--new-capability" in g
        assert g[g.index("--gpu-ids") + 1] == "2,3" and g[g.index("--max-locks") + 1] == "4"
        assert g[g.index("--python") + 1] == "/env/bin/python" and "--resume" not in g


def test_a_certified_device_index_runs_without_the_flag(tmp_path):
    tools = FakeTools(tmp_path / "lanes")
    assert chunks_main(tmp_path, tools, "c0", mode="certified") == 0
    assert not any("--new-capability" in c for c in tools.calls)


def test_a_check_that_does_not_read_check_ok_stops_before_any_gate(tmp_path):
    tools = FakeTools(tmp_path / "lanes", check="CHECK REFUSED (1 problem(s))")
    assert chunks_main(tmp_path, tools, "c0", "c1") == 3
    assert tools.gates() == [] and len(tools.checks()) == 1


def test_a_launched_chunk_resumes_and_a_done_chunk_is_skipped(tmp_path):
    tools = FakeTools(tmp_path / "lanes")
    launch = tmp_path / "repo" / "parity" / "meep_gpu" / "results" / \
        "chunk_2026-10-18_101_cc90_c1" / "LAUNCH.json"
    launch.parent.mkdir(parents=True)
    launch.write_text("{}")
    done = tmp_path / "lanes" / "2026-10-18_101_cc90_c0"
    done.mkdir(parents=True)
    (done / "GATES_DONE").write_text("")
    assert chunks_main(tmp_path, tools, "c0", "c1") == 0
    gates = tools.gates()
    assert [g[g.index("--chunk") + 1] for g in gates] == ["c1"] and "--resume" in gates[0]
    assert tools.checks() == [], "a resumed chunk is not checked again, a done one not run"


def test_a_chunk_that_does_not_pass_stops_the_run_unless_told_to_go_on(tmp_path):
    tools = FakeTools(tmp_path / "lanes", gates_rc={"c0": 1})
    assert chunks_main(tmp_path, tools, "c0", "c1") == 1
    assert [g[g.index("--chunk") + 1] for g in tools.gates()] == ["c0"]
    tools = FakeTools(tmp_path / "lanes2", gates_rc={"c0": 1})
    assert chunks_main(tmp_path, tools, "--continue-on-fail", "c0", "c1", "c2") == 1
    assert [g[g.index("--chunk") + 1] for g in tools.gates()] == ["c0", "c1", "c2"]


def test_the_runner_refuses_what_it_cannot_run(tmp_path, capsys):
    tools = FakeTools(tmp_path / "lanes")
    assert chunks_main(tmp_path, tools, "c9") == 2
    assert run_chunks.main(["--index", str(write_index(tmp_path / "m.json", machine="metal")),
                            "--lanes", str(tmp_path / "l"), "--lock-dir", str(tmp_path / "k"),
                            "--expect", EXPECT, "c0"], run=tools) == 2
    assert run_chunks.main(["--index", str(write_index(tmp_path / "i.json")), "--lanes",
                            str(tmp_path / "l"), "--lock-dir", str(tmp_path / "k"),
                            "--expect", "PASTE-THE-COMMIT", "c0"], run=tools) == 2
    out = capsys.readouterr().out
    assert "c9 is not a chunk" in out and "'metal'" in out and "not a 40-character" in out
    assert tools.gates() == []


def test_spread_runs_one_chunk_per_card_at_a_time_and_every_chunk_once(tmp_path):
    tools = FakeTools(tmp_path / "lanes")
    assert chunks_main(tmp_path, tools, "--gpu-ids", "0,1", "--spread", "c0", "c1", "c2") == 0
    gates = tools.gates()
    ran = sorted(g[g.index("--chunk") + 1] for g in gates)
    assert ran == ["c0", "c1", "c2"]
    assert {g[g.index("--gpu-ids") + 1] for g in gates} <= {"0", "1"}
    assert all(g[g.index("--max-locks") + 1] == "2" for g in gates), "the cap defaults to the cards"


def test_spread_stops_taking_chunks_after_one_that_does_not_pass(tmp_path):
    tools = FakeTools(tmp_path / "lanes", gates_rc={"c0": 1})
    assert chunks_main(tmp_path, tools, "--gpu-ids", "0", "--spread", "c0", "c1") == 1
    assert [g[g.index("--chunk") + 1] for g in tools.gates()] == ["c0"]
    tools = FakeTools(tmp_path / "lanes2", gates_rc={"c0": 1})
    assert chunks_main(tmp_path, tools, "--gpu-ids", "0", "--spread", "--continue-on-fail",
                       "c0", "c1") == 1
    assert [g[g.index("--chunk") + 1] for g in tools.gates()] == ["c0", "c1"]


@pytest.mark.parametrize("device, capability", [("cc90", "9.0"), ("cc86", "8.6"),
                                                ("cc100", "10.0"), ("cc120", "12.0")])
def test_the_capability_is_read_from_the_index_device(device, capability):
    assert run_chunks.capability_of(device) == capability


# --------------------------------------------------------------------- gpu_log -----
def test_the_card_log_reports_each_cards_peak_and_lowest_reading():
    text = (gpu_log.HEADER + "\n"
            "2026/10/21 10:00:00.000, 0, 3, 143771, 0, 30, 70\n"
            "2026/10/21 10:00:00.000, 1, 1, 143771, 0, 31, 71\n"
            "2026/10/21 10:00:30.000, 0, 12650, 143771, 97, 45, 400\n"
            "2026/10/21 10:01:00.000, 0, 400, 143771, 20, 40, 200\n"
            "2026/10/21 10:01:00.000, 1, [N/A], 143771, 0, 31, 71\n")
    found = gpu_log.peaks(text)
    assert found["0"] == {"peak": 12650, "at": "2026/10/21 10:00:30.000", "lowest": 3,
                          "readings": 3}
    assert found["1"]["peak"] == 1 and found["1"]["readings"] == 1
    assert gpu_log.reader_argv("0,1")[1] == "--id=0,1"
    assert not any(a.startswith("--id") for a in gpu_log.reader_argv(""))


def test_the_card_log_starts_in_its_own_session_and_stops(tmp_path, monkeypatch, capsys):
    smi = fake_smi(tmp_path / "bin", "import time\n"
                   "print('2026/10/21 10:00:00.000, 0, 5, 143771, 0, 30, 70', flush=True)\n"
                   "time.sleep(120)\n")
    monkeypatch.setenv("PATH", f"{smi.parent}{os.pathsep}{os.environ.get('PATH', '')}")
    where = tmp_path / "round"
    assert gpu_log.main(["start", "--dir", str(where)]) == 0
    pid = int((where / "gpu_log.pid").read_text())
    assert os.getsid(pid) == pid, "the reader leads a session of its own"
    deadline = time.time() + 20
    while len((where / "gpu_log.csv").read_text().splitlines()) < 2 and time.time() < deadline:
        time.sleep(0.1)
    assert gpu_log.main(["peak", "--dir", str(where)]) == 0
    assert gpu_log.main(["stop", "--dir", str(where)]) == 0
    deadline = time.time() + 20
    while gpu_log.alive(pid) and time.time() < deadline:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            break
        time.sleep(0.1)
    assert not (where / "gpu_log.pid").exists()
    out = capsys.readouterr().out
    assert "GPU 0: peak memory.used 5 MiB" in out and "1 readings" in out


# ---------------------------------------------------------------- pack_results -----
def results_tree(tmp: Path) -> Tuple[Path, Path, Path]:
    repo = tmp / "repo"
    results = repo / "parity" / "meep_gpu" / "results"
    for name in ("cuda_regate_2026-10-18_101_cc90_c0_g0", "triton_fleet_2026-10-18_101_cc90_c1_t2",
                 "chunk_2026-10-18_101_cc90_c0", "set_aside_2026-10-18_101_cc90_c0",
                 "cuda_regate_2026-10-18_101_cc86_c0_g0", "dispatch_fused_route_x_cc86",
                 "dispatch_fused_route_x_cc90", "dispatch_fused_route_x_cc90_preflight",
                 "dispatch_end_to_end_x_cc90", "triton_fleet_2026-10-15_10_cc90_c0"):
        (results / name).mkdir(parents=True)
        (results / name / "gate.json").write_text(json.dumps({"name": name}))
    caches = results / "cuda_regate_2026-10-18_101_cc90_c0_g0" / "leg" / "caches"
    caches.mkdir(parents=True)
    (caches / "kernel.cubin").write_text("compiled")
    for ledger in pack_results.LEDGERS:
        (repo / ledger).parent.mkdir(parents=True, exist_ok=True)
        (repo / ledger).write_text("{}")
    lanes = tmp / "round" / "lanes"
    (lanes / "2026-10-18_101_cc90_c0").mkdir(parents=True)
    (lanes / "2026-10-18_101_cc90_c0" / "round.log").write_text("log\n")
    index = write_index(tmp / "round" / "INDEX_cc90.json")
    (tmp / "round" / "preflight.txt").write_text("PREFLIGHT OK (0 FAIL)\n")
    (tmp / "round" / "meep-gpu-cert_41.out").write_text("job output\n")
    return repo, lanes, index


def members(archive: Path) -> List[str]:
    with tarfile.open(archive) as handle:
        return sorted(m.name for m in handle.getmembers())


def test_part1_packs_the_device_chunk_roots_and_set_asides_without_caches(tmp_path):
    repo, lanes, index = results_tree(tmp_path)
    results = repo / "parity" / "meep_gpu" / "results"
    assert pack_results.roots(results, "part1", "2026-10-18_101", "cc90") == [
        "chunk_2026-10-18_101_cc90_c0", "cuda_regate_2026-10-18_101_cc90_c0_g0",
        "set_aside_2026-10-18_101_cc90_c0", "triton_fleet_2026-10-18_101_cc90_c1_t2"]
    assert pack_results.roots(results, "part2", "2026-10-18_101", "cc90") == [
        "dispatch_end_to_end_x_cc90", "dispatch_fused_route_x_cc90",
        "dispatch_fused_route_x_cc90_preflight", "set_aside_2026-10-18_101_cc90_c0"]
    recorded = []
    archive = pack_results.pack("part1", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                                None, None, lambda out: recorded.append(out))
    assert archive.name == "meep-gpu-cc90-part1.tgz" and recorded
    out = tmp_path / "out" / "meep-gpu-cc90-part1"
    inner = members(out / "results.tgz")
    assert not any("caches" in name for name in inner)
    assert "cuda_regate_2026-10-18_101_cc90_c0_g0/gate.json" in inner
    assert not any("cc86" in name or "2026-10-15_10" in name for name in inner)
    assert "lanes/2026-10-18_101_cc90_c0/round.log" in members(out / "lanes.tgz")
    sums = (out / "SHA256SUMS").read_text().splitlines()
    listed = sorted(line.split("  ", 1)[1] for line in sums)
    assert "./results.tgz" in listed and "./preflight.txt" in listed and "./INDEX_cc90.json" in listed
    assert "./meep-gpu-cert_41.out" in listed
    assert not (out / "ledgers").exists()
    with pytest.raises(FileExistsError):
        pack_results.pack("part1", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                          None, None, lambda out: None)
    shutil.rmtree(out)
    with pytest.raises(FileExistsError, match="nothing is overwritten"):
        pack_results.pack("part1", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                          None, None, lambda out: None)


def test_part2_carries_the_bound_ledgers(tmp_path):
    repo, lanes, index = results_tree(tmp_path)
    pack_results.pack("part2", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                      None, None, lambda out: None)
    out = tmp_path / "out" / "meep-gpu-cc90-part2"
    for ledger in pack_results.LEDGERS:
        assert (out / "ledgers" / ledger).is_file()
    assert "dispatch_end_to_end_x_cc90/gate.json" in members(out / "results.tgz")


def test_part2_without_a_route_packs_the_bind_and_says_so(tmp_path, capsys):
    repo, lanes, index = results_tree(tmp_path)
    results = repo / "parity" / "meep_gpu" / "results"
    for name in ("dispatch_fused_route_x_cc90", "dispatch_fused_route_x_cc90_preflight",
                 "dispatch_end_to_end_x_cc90"):
        shutil.rmtree(results / name)
    (tmp_path / "round" / "bind_reports.txt").write_text("report\n")
    pack_results.pack("part2", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                      None, None, lambda out: None)
    out = tmp_path / "out" / "meep-gpu-cc90-part2"
    assert "no route campaign of cc90" in capsys.readouterr().out
    assert all((out / "ledgers" / ledger).is_file() for ledger in pack_results.LEDGERS)
    assert (out / "bind_reports.txt").is_file()


def test_a_part_with_no_roots_of_its_own_is_refused(tmp_path):
    repo, lanes, _index = results_tree(tmp_path)
    other = tmp_path / "round" / "INDEX_cc89.json"
    other.write_text(json.dumps({"machine": "nvidia", "device": "cc89", "device_mode": "new",
                                 "round": "2026-10-18_101", "chunks": []}))
    with pytest.raises(FileNotFoundError, match="no results root of part1"):
        pack_results.pack("part1", repo, other, lanes, tmp_path / "round", tmp_path / "out",
                          None, None, lambda out: None)


# ------------------------------------------------ pack_results: what a part refuses -----
#: The two home roots, from pieces: a source line spelling one in full is a hygiene-guard hit.
HOME_ROOT, USERS_ROOT = "/" + "ho" + "me", "/" + "Us" + "ers"


@pytest.fixture
def account(monkeypatch, tmp_path):
    """The packing machine as the packer reads it: its login and home directory are tokens
    (the home lies under no home root and does not hold the login, so only its own spelling
    names it)."""
    spell = pack_results.neutral_ids()
    home = tmp_path / "accounts" / "the-home"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(spell.os.path, "expanduser",
                        lambda p: p.replace("~", str(home), 1) if p.startswith("~") else p)
    monkeypatch.setattr(spell, "login", lambda: "someone")
    spell.reset()
    yield home
    spell.reset()


def plant(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload, indent=1),
                    encoding="utf-8")


def test_a_part_that_names_a_home_is_refused_by_file_and_key_before_anything_is_written(
        tmp_path, account):
    repo, lanes, index = results_tree(tmp_path)
    results = repo / "parity" / "meep_gpu" / "results"
    fleet = results / "triton_fleet_2026-10-18_101_cc90_c1_t2"
    other = f"{HOME_ROOT}/collaborator/meep-gpu-cert/parity/meep_gpu/results"
    # The composition probe's leak as a collaborator's pack carried it: two keys of one file.
    plant(fleet / "cylindrical_complex_composition" / "gate.json",
          {"environment": {"CUPY_CACHE_DIR": f"{other}/cupy_cache_ftz_stripped_c1"},
           "provenance": f"{other}/x/composition_provenance.json", "cases": []})
    # This account's own home (under no home root) in a row file; the account's name as a
    # directory elsewhere; a macOS and a Windows home in two lane logs; a JSON-escaped
    # Windows home and a home path as a key; this home in the verify report.
    plant(fleet / "fused_hd_pair" / "gate.jsonl",
          json.dumps({"leg": "lift", "corpus_root": f"{account}/envs/ref/python"}) + "\n")
    plant(fleet / "nonlinear" / "gate.json", {"rows": [{"log": "/data/someone/runs/a.log"}]})
    plant(lanes / "2026-10-18_101_cc90_c0" / "gates_0" / "step.log",
          f"ok\nopened {USERS_ROOT}/other/x.h5\n")
    plant(lanes / "2026-10-18_101_cc90_c0" / "gates_0" / "build.log",
          "ok\nok\nopened C:\\Users\\other\\y.h5\n")
    plant(fleet / "symmetry" / "gate.json", {"windows": {"log": "C:\\Users\\other\\z.h5"}})
    plant(fleet / "dispersive" / "gate.json", {"by_path": {f"{HOME_ROOT}/other/k": 1}})
    verify = tmp_path / "round" / "verify_cc90.json"
    plant(verify, {"ok": True, "repo": f"{account}/meep-gpu-cert"})
    with pytest.raises(pack_results.LeakRefusal) as refused:
        pack_results.pack("part1", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                          None, verify, lambda out: None)
    message = str(refused.value)
    assert message.startswith("8 of the ") and "nothing is packed" in message, message
    assert ("results/triton_fleet_2026-10-18_101_cc90_c1_t2/cylindrical_complex_composition/"
            "gate.json: environment.CUPY_CACHE_DIR, provenance") in message, message
    assert ("results/triton_fleet_2026-10-18_101_cc90_c1_t2/fused_hd_pair/gate.jsonl: "
            "line 1: corpus_root") in message, message
    assert "nonlinear/gate.json: rows[].log\n" in message, message
    assert "verify_cc90.json: repo" in message, message
    assert "lanes/2026-10-18_101_cc90_c0/gates_0/step.log: line 2" in message, message
    assert "lanes/2026-10-18_101_cc90_c0/gates_0/build.log: line 3" in message, message
    assert "symmetry/gate.json: windows.log\n" in message, message
    assert "dispersive/gate.json: by_path (a key)\n" in message, message
    # Named by file and key, never by value; and nothing was written.
    for value in ("collaborator", str(account), "/data/someone", "/other", "\\other"):
        assert value not in message, value
    assert not (tmp_path / "out").exists()
    # The same refusal through the command line: STOP, exit 2.
    assert pack_results.main(["part1", "--index", str(index), "--lanes", str(lanes),
                              "--round-dir", str(tmp_path / "round"), "--out-dir",
                              str(tmp_path / "out"), "--repo", str(repo), "--verify",
                              str(verify)],
                             records=lambda out: None) == 2
    assert not (tmp_path / "out").exists()


def test_a_clean_part_packs_its_artifacts_byte_for_byte_and_a_temporary_path_passes(
        tmp_path, account):
    repo, lanes, index = results_tree(tmp_path)
    results = repo / "parity" / "meep_gpu" / "results"
    # A temporary directory a gate made (no account in it) and paths under their labels.
    plant(results / "triton_fleet_2026-10-18_101_cc90_c1_t2" / "folded_offdiag" / "gate.json",
          {"mutant_file": {"path": "/tmp/tmpk2l9x0ab/mutant.py"},
           "environment": {"CUPY_CACHE_DIR": "<home>/meep-gpu-cert/results/cupy_cache_c1"},
           "argv": ["<staging-root>/meep-gpu-cert/parity/meep_gpu/gate.py", f"{HOME_ROOT}/"]})
    # What the part leaves out is not read: a compiler cache, a local pid file and a link.
    leg = results / "cuda_regate_2026-10-18_101_cc90_c0_g0" / "leg"
    plant(leg / "caches" / "kernel.log", f"{account}/envs/ref/lib")
    plant(lanes / "2026-10-18_101_cc90_c0" / "gates_0" / "lane.pid", f"{account}/run\n")
    (account / "outside.txt").write_text(f"{account}/x\n")
    (leg / "outside.txt").symlink_to(account / "outside.txt")
    archive = pack_results.pack("part1", repo, index, lanes, tmp_path / "round",
                                tmp_path / "out", None, None, lambda out: None)
    assert archive.is_file()
    out = tmp_path / "out" / "meep-gpu-cc90-part1"
    with tarfile.open(out / "results.tgz") as handle:
        packed = {m.name: handle.extractfile(m).read() for m in handle.getmembers() if m.isfile()}
    assert packed and all(data == (results / name).read_bytes() for name, data in packed.items())
    # The files read before packing are exactly the files packed.
    names = pack_results.roots(results, "part1", "2026-10-18_101", "cc90")
    assert sorted(packed) == sorted(p.relative_to(results).as_posix()
                                    for p in pack_results.packed_files(results, names))


def test_a_file_the_packer_writes_itself_is_read_again_before_the_archive(tmp_path, account):
    repo, lanes, index = results_tree(tmp_path)

    def records(out: Path) -> None:
        (out / "build_record.json").write_text(json.dumps({"prefix": f"{account}/envs/ref"}))

    with pytest.raises(pack_results.LeakRefusal, match="build_record.json: prefix"):
        pack_results.pack("part1", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                          None, None, records)
    assert not (tmp_path / "out" / "meep-gpu-cc90-part1.tgz").exists()


def test_a_checkout_in_the_home_directory_does_not_hide_a_home_path(tmp_path, account,
                                                                     monkeypatch):
    """A collaborator's clone lies in his home directory, so its staging root IS the home:
    a raw path under the clone is still read as a home path, not as ``<staging-root>``."""
    spell = pack_results.neutral_ids()
    monkeypatch.setattr(spell, "CHECKOUT", account / "meep-gpu-cert")
    spell.reset()
    repo, lanes, index = results_tree(tmp_path)
    plant(repo / "parity" / "meep_gpu" / "results" / "triton_fleet_2026-10-18_101_cc90_c1_t2"
          / "composition" / "gate.json",
          {"provenance": f"{account}/meep-gpu-cert/parity/meep_gpu/results/x/p.json"})
    with pytest.raises(pack_results.LeakRefusal, match="composition/gate.json: provenance"):
        pack_results.pack("part1", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                          None, None, lambda out: None)


def test_part2_reads_the_ledgers_it_carries(tmp_path, account):
    repo, lanes, index = results_tree(tmp_path)
    (repo / pack_results.LEDGERS[0]).write_text(json.dumps({"w": {"host": f"{account}/x"}}))
    with pytest.raises(pack_results.LeakRefusal, match="ledgers/meep_gpu/cuda_kernels/"
                                                       "certification.json: w.host"):
        pack_results.pack("part2", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                          None, None, lambda out: None)
    assert not (tmp_path / "out").exists()


# ------------------------------------- pack_results: a part1 of the chunks run since -----
def rerun_tree(tmp: Path) -> Tuple[Path, Path, Path, Path]:
    """The round of :func:`results_tree` after a chunk ran again: c1's roots and lanes, the
    runner's checks and log beside c0's, and a selection and verify report of the round."""
    repo, lanes, index = results_tree(tmp)
    results = repo / "parity" / "meep_gpu" / "results"
    for name in ("chunk_2026-10-18_101_cc90_c1", "triton_fleet_2026-10-18_101_cc90_c1_t2_flush"):
        plant(results / name / "gate.json", {"name": name})
    plant(lanes / "2026-10-18_101_cc90_c1" / "round.log", "log\n")
    for k in (0, 1):
        plant(lanes / f"check_c{k}.txt", "CHECK OK (0 problem(s))\n")
    plant(lanes / "run_chunks.log", "c0 GATES_DONE\nc1 GATES_DONE\n")
    plant(tmp / "round" / "selection_cc90.json", {"selected": {}})
    plant(tmp / "round" / "verify_cc90.json", {"ok": True})
    return repo, lanes, index, results


def test_a_part1_since_a_chunk_carries_only_the_newer_chunks_and_records_both(tmp_path,
                                                                              account):
    repo, lanes, index, results = rerun_tree(tmp_path)
    # c0's composition run wrote a home path; c1 ran the unit again on a fixed writer, so the
    # earlier run is superseded and never travels again: it does not stop this part.
    plant(results / "cuda_regate_2026-10-18_101_cc90_c0_g0" / "composition" / "gate.json",
          {"provenance": f"{HOME_ROOT}/collaborator/meep-gpu-cert/x/p.json"})
    selection, verify = tmp_path / "round" / "selection_cc90.json", tmp_path / "round" / "verify_cc90.json"
    archive = pack_results.pack("part1", repo, index, lanes, tmp_path / "round",
                                tmp_path / "out", selection, verify, lambda out: None,
                                since="c0")
    assert archive.name == "meep-gpu-cc90-part1-c1-c2.tgz"
    out = tmp_path / "out" / "meep-gpu-cc90-part1-c1-c2"
    inner = members(out / "results.tgz")
    assert {name.split("/")[0] for name in inner} == {
        "chunk_2026-10-18_101_cc90_c1", "triton_fleet_2026-10-18_101_cc90_c1_t2",
        "triton_fleet_2026-10-18_101_cc90_c1_t2_flush"}, inner
    assert members(out / "lanes.tgz") == [
        "lanes/2026-10-18_101_cc90_c1", "lanes/2026-10-18_101_cc90_c1/round.log",
        "lanes/check_c1.txt", "lanes/run_chunks.log"]
    record = json.loads((out / "CHUNKS.json").read_text())
    assert record["chosen"] == "newer than c0"
    assert [(c["id"], c["results_roots"]) for c in record["carried"]] == [
        ("c1", ["chunk_2026-10-18_101_cc90_c1", "triton_fleet_2026-10-18_101_cc90_c1_t2",
                "triton_fleet_2026-10-18_101_cc90_c1_t2_flush"]), ("c2", [])]
    assert record["left_out"] == [{"id": "c0", "stamp": "2026-10-18_101_cc90_c0"}]
    # The whole index, selection and verify report travel as in every part1.
    for name in ("INDEX_cc90.json", "selection_cc90.json", "verify_cc90.json"):
        assert (out / name).read_bytes() == (tmp_path / "round" / name).read_bytes()
    assert "chunks c1,c2 (newer than c0)" in (out / "ROUND.txt").read_text()
    # Named chunks, through the command line: the same choice, under its own name.
    assert pack_results.main(["part1", "--index", str(index), "--lanes", str(lanes),
                              "--round-dir", str(tmp_path / "round"), "--out-dir",
                              str(tmp_path / "out2"), "--repo", str(repo), "--chunks", "c1"],
                             records=lambda out: None) == 0
    named = json.loads((tmp_path / "out2" / "meep-gpu-cc90-part1-c1" / "CHUNKS.json").read_text())
    assert named["chosen"] == "named" and [c["id"] for c in named["carried"]] == ["c1"]
    assert [c["id"] for c in named["left_out"]] == ["c0", "c2"]


def test_a_part1_of_every_chunk_records_that_it_carries_them_all(tmp_path):
    repo, lanes, index, _results = rerun_tree(tmp_path)
    pack_results.pack("part1", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                      None, None, lambda out: None)
    record = json.loads((tmp_path / "out" / "meep-gpu-cc90-part1" / "CHUNKS.json").read_text())
    assert record["chosen"] == "every chunk" and record["left_out"] == []
    assert [c["id"] for c in record["carried"]] == ["c0", "c1", "c2"]
    assert record["roots_of_no_chunk"] == []
    assert "lanes/check_c0.txt" in members(tmp_path / "out" / "meep-gpu-cc90-part1" / "lanes.tgz")


def test_a_chosen_chunk_that_names_a_home_is_still_refused(tmp_path, account):
    repo, lanes, index, results = rerun_tree(tmp_path)
    plant(results / "triton_fleet_2026-10-18_101_cc90_c1_t2_flush" / "composition" / "gate.json",
          {"environment": {"CUPY_CACHE_DIR": f"{account}/meep-gpu-cert/cache"}})
    plant(lanes / "2026-10-18_101_cc90_c1" / "gates_0" / "step.log", f"ok\n{account}/x\n")
    with pytest.raises(pack_results.LeakRefusal) as refused:
        pack_results.pack("part1", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                          None, None, lambda out: None, named=["c1"])
    message = str(refused.value)
    assert ("results/triton_fleet_2026-10-18_101_cc90_c1_t2_flush/composition/gate.json: "
            "environment.CUPY_CACHE_DIR") in message, message
    assert "lanes/2026-10-18_101_cc90_c1/gates_0/step.log: line 2" in message, message
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("choice, why", [
    ({"named": ["c7"]}, "not a current chunk of the index: c7"),
    ({"since": "c9"}, "the index never allocated chunk c9"),
    ({"since": "c2"}, "no current chunk of the index is newer than c2"),
    ({"named": ["c1"], "since": "c0"}, "choose the chunks two ways"),
])
def test_a_chunk_choice_the_index_cannot_answer_is_refused_by_name(tmp_path, choice, why):
    repo, lanes, index, _results = rerun_tree(tmp_path)
    with pytest.raises(ValueError, match=why):
        pack_results.pack("part1", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                          None, None, lambda out: None, **choice)
    assert not (tmp_path / "out").exists()


def test_part2_carries_no_chunk_and_refuses_a_chunk_choice(tmp_path):
    repo, lanes, index, _results = rerun_tree(tmp_path)
    with pytest.raises(ValueError, match="part2 carries no chunk's results roots"):
        pack_results.pack("part2", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                          None, None, lambda out: None, since="c0")
    pack_results.pack("part2", repo, index, lanes, tmp_path / "round", tmp_path / "out",
                      None, None, lambda out: None)
    assert not (tmp_path / "out" / "meep-gpu-cc90-part2" / "CHUNKS.json").exists()
