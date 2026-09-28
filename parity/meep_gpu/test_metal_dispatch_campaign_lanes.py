"""Host-only tests for ``run_metal_dispatch_campaign.sh``: order, lanes, isolation.

NO DEVICE AND NO GATE. Two layers, because a dry run can only show a plan:

* the DRY RUN is parsed for what each leg WOULD be handed -- order, lane plan,
  arguments, environment, the state of each leg directory -- and for the promise
  that it creates nothing;
* the REAL launch path is driven with a STUB in place of ``metal_gate_runner.py``
  (the script's ``METAL_CAMPAIGN_RUNNER`` hook) that records the argv and the
  environment it was really handed, waits or sleeps as directed, and writes a
  ``gate.json`` with a directed verdict. That is the only way to observe the things a
  plan cannot show: the exclusive directory, the lane cap, a lane that dies, and
  who writes ``campaign.txt`` in what order.

The environment assertions are made under a HOSTILE calling shell: every name a leg
must not inherit is exported by the test before the script starts, which is the
situation the per-process environments exist for (``recut_metal_gates.sh`` exports
both the subnormal policy and the probe paths for its own children).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("run_metal_dispatch_campaign.sh")
_INTERPRETER = re.search(r"^PY=(\S+)$", SCRIPT.read_text(encoding="utf-8"), re.M)

pytestmark = [
    pytest.mark.requires_resource("zsh"),
    pytest.mark.skipif(
        shutil.which("zsh") is None or _INTERPRETER is None
        or "MGPU_SITE_PYTHON" not in _INTERPRETER.group(1),
        reason="the campaign script needs zsh and the interpreter it names"),
]

#: WHAT EACH LEG WAS HANDED BEFORE LANES EXISTED, typed here rather than read from
#: the script: the point of the comparison is that a lane change cannot move it.
LEG_ARGUMENTS = {
    "band_witness": ["--install-policy", "keep", "--band-witness"],
    "harness_keep": ["--install-policy", "keep"],
    "harness_flush": ["--install-policy", "flush"],
    "shipped": [],
    "shipped_expansion_probe": [],
}
SERIAL = ["band_witness", "harness_keep"]
LANES_LONGEST_FIRST = ["shipped_expansion_probe", "shipped", "harness_flush"]
ORDER = SERIAL + LANES_LONGEST_FIRST

PROBE_VARIABLES = (
    "MEEP_GPU_METAL_EXPANSION_PROBE",
    "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
    "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE",
    "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE",
)
SCRUBBED = (
    "MEEP_GPU_SUBNORMAL_POLICY", "MEEP_GPU_SUBNORMAL_INSTALL", "MEEP_GPU_DISPATCH",
    "MEEP_GPU_FUSED", "MEEP_GPU_FUSE_ARMS", "MEEP_GPU_DISPATCH_LOG",
    "MEEP_GPU_GATE_SOURCE_MANIFEST",
)
#: Recorded, not scrubbed: these reach every leg as the calling shell holds them.
RECORDED_SWITCHES = ("MEEP_GPU_WARM", "MEEP_GPU_KERNEL_TABLE",
                     "MEEP_GPU_BACKEND_PREFERENCE")
WITNESS_VARIABLE = "MEEP_GPU_METAL_BAND_WITNESS"
#: The one scrubbed name a leg is HANDED: band_witness gets exactly
#: ``MEEP_GPU_DISPATCH=0``, so "dispatches nothing" is its declaration rather than a
#: consequence of rung 8bM refusing the keep policy it installs. Every other leg still
#: has the enable scrubbed.
DISPATCH_ENABLE = "MEEP_GPU_DISPATCH"
PINNED_OFF = {"band_witness": {DISPATCH_ENABLE: "0"}}

_STUB = r'''
"""Stands in for metal_gate_runner.py: records what it was handed, runs no gate."""
import json, os, signal, sys, time
from pathlib import Path

here = Path(__file__).resolve().parent
argv = sys.argv[1:]
out = Path(argv[argv.index("--out") + 1])
leg = out.name
control = json.loads((here / "control.json").read_text()).get(leg, {})
journal = here / "journal.log"


def note(event):
    with open(journal, "a") as handle:
        handle.write(f"{leg} {event} {time.time():.6f}\n")


def started_legs():
    if not journal.exists():
        return set()
    return {line.split()[0] for line in journal.read_text().splitlines()
            if line.split()[1] == "start"}


sigint = signal.getsignal(signal.SIGINT)
(here / f"seen.{leg}.json").write_text(json.dumps(
    {"argv": argv, "environ": dict(os.environ), "out_existed": out.is_dir(),
     "out_listing": sorted(p.name for p in out.iterdir()) if out.is_dir() else None,
     "pid": os.getpid(), "sigint_ignored": sigint is signal.SIG_IGN}))
note("start")
together = control.get("alive_together")
if together:
    deadline = time.time() + 60
    while not set(together) <= started_legs():
        if time.time() > deadline:
            note("rendezvous-timeout")
            sys.exit(97)
        time.sleep(0.02)
time.sleep(control.get("sleep", 0.0))
if control.get("kill_lane"):
    note("end")
    os.kill(os.getppid(), signal.SIGKILL)
    sys.exit(0)
if control.get("write_gate", True):
    payload = {"release": {"released": control.get("released", True), "reasons": []},
               "leg": leg}
    if control.get("finalized", True):
        # What metal_gate_runner.finalize_artifact adds; without it the artifact is
        # the gate's own write, which no runner welded.
        payload["weld"] = {"passed": True, "source_count": 0, "manifest": "stub"}
    (out / "gate.json").write_text(json.dumps(payload))
note("end")
sys.exit(control.get("exit", 0))
'''


class Campaign:
    """One scratch results root, one stub runner, one hostile calling shell."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.results = root / "results"
        self.stub_dir = root / "stub"
        self.stub_dir.mkdir()
        self.stub = self.stub_dir / "stub_runner.py"
        self.stub.write_text(_STUB, encoding="utf-8")
        self.probes = root / "probes"
        self.probes.mkdir()
        (self.probes / "special_kz.json").write_text("{}")
        self.complex_probe = root / "complex_expansion_probe.json"
        self.complex_probe.write_text("{}")
        self.tools = root / "bin"
        self.tools.mkdir()
        # THE FAKE TOOL HAS THE REAL ONE'S TWO SHAPES: `-i -w <pid>` records and
        # waits; `-i <command>` records and execs the command IN PLACE, same pid.
        self.caffeinate_seen = root / "caffeinate.watch"
        self.caffeinate_legs = root / "caffeinate.legs"
        self.caffeinate = self.tools / "caffeinate"
        self.caffeinate.write_text(
            "#!/bin/sh\n"
            "case \" $* \" in\n"
            f"  *' -w '*) echo \"$@\" >> '{self.caffeinate_seen}'; exec sleep 120 ;;\n"
            "esac\n"
            f"echo \"$@\" >> '{self.caffeinate_legs}'\n"
            "shift\n"
            "exec \"$@\"\n")
        self.caffeinate.chmod(0o755)
        self.control({})

    @property
    def run(self) -> Path:
        return self.results / "dispatch_metal_route_stamp"

    def control(self, behaviour: dict) -> None:
        (self.stub_dir / "control.json").write_text(json.dumps(behaviour))

    def environment(self, path: str | None = None) -> dict:
        env = dict(os.environ)
        env.pop("LANES", None)
        env.pop("DRY_RUN", None)
        env["PATH"] = path or f"{self.tools}:{env.get('PATH', '/usr/bin:/bin')}"
        env["MGPU_SITE_PYTHON"] = sys.executable
        env.pop("MGPU_SITE_SOURCE_ROOT", None)
        env["METAL_CAMPAIGN_RESULTS_ROOT"] = str(self.results)
        env["METAL_CAMPAIGN_RUNNER"] = str(self.stub)
        env["METAL_CAMPAIGN_POLL_SECONDS"] = "0.2"
        # THE HOSTILE SHELL: everything a leg must not inherit, exported.
        for name in SCRUBBED:
            env[name] = "inherited-from-the-calling-shell"
        for name in PROBE_VARIABLES:
            env[name] = "/inherited/probe.json"
        env[WITNESS_VARIABLE] = "/inherited/stale/band_witness/gate.json"
        for name in RECORDED_SWITCHES:
            env.pop(name, None)
        env["MEEP_GPU_KERNEL_TABLE"] = "metal"
        env["MEEP_GPU_CORPUS_ROOT"] = "/somewhere/else"
        return env

    def invoke(self, *options: str, probes: bool = True, env: dict | None = None,
               positional_lanes: str | None = None):
        arguments = ["zsh", str(SCRIPT), *options, "stamp"]
        if probes:
            arguments += [str(self.probes), str(self.complex_probe)]
        elif positional_lanes is not None:
            arguments += ["", ""]
        if positional_lanes is not None:
            arguments.append(positional_lanes)
        return subprocess.run(arguments, env=env or self.environment(), text=True,
                              capture_output=True, timeout=180)

    def start(self, *options: str) -> subprocess.Popen:
        """The same invocation, left running, for tests that act on a live driver."""
        return subprocess.Popen(
            ["zsh", str(SCRIPT), *options, "stamp", str(self.probes),
             str(self.complex_probe)], env=self.environment(), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def wait_for_line(self, fragment: str, seconds: float = 30.0) -> str:
        deadline = time.time() + seconds
        path = self.run / "campaign.txt"
        while time.time() < deadline:
            if path.exists():
                for line in path.read_text().splitlines():
                    if fragment in line:
                        return line
            time.sleep(0.05)
        raise AssertionError(f"campaign.txt never showed {fragment!r}")

    def seen(self, leg: str) -> dict:
        return json.loads((self.stub_dir / f"seen.{leg}.json").read_text())

    def journal(self) -> list[tuple[str, str, float]]:
        path = self.stub_dir / "journal.log"
        if not path.exists():
            return []
        return [(leg, event, float(when)) for leg, event, when in
                (line.split() for line in path.read_text().splitlines())]

    def interval(self, leg: str) -> tuple[float, float]:
        rows = {event: when for name, event, when in self.journal() if name == leg}
        return rows["start"], rows["end"]

    def summary(self) -> list[str]:
        return (self.run / "campaign.txt").read_text().splitlines()


@pytest.fixture
def campaign(tmp_path: Path) -> Campaign:
    return Campaign(tmp_path)


def _plan(stdout: str) -> dict[str, dict]:
    """The dry run's LEG blocks: header fields, set / unset names, and the argv."""
    legs: dict[str, dict] = {}
    current = None
    for line in stdout.splitlines():
        if line.startswith("LEG "):
            fields = dict(item.split("=", 1) for item in line.split()[1:])
            current = legs[fields["leg"]] = {**fields, "set": {}, "unset": [],
                                             "command": None}
        elif current is not None and line.startswith("  set "):
            name, _, value = line.split(None, 1)[1].partition("=")
            current["set"][name] = value
        elif current is not None and line.startswith("  unset "):
            current["unset"].append(line.split()[1])
        elif current is not None and line.startswith("  command "):
            current["command"] = shlex.split(line.split(None, 1)[1])
    return legs


# ---------------------------------------------------------------------------
# The dry run: the plan, and the promise that it touches nothing
# ---------------------------------------------------------------------------


def test_dry_run_prints_the_order_and_lane_plan_and_creates_nothing(campaign):
    result = campaign.invoke("--dry-run", "--lanes", "3")
    assert result.returncode == 0, result.stdout + result.stderr
    plan = _plan(result.stdout)

    assert list(plan) == ORDER
    assert [plan[leg]["phase"] for leg in ORDER] == ["serial", "serial",
                                                     "lane", "lane", "lane"]
    assert [int(plan[leg]["order"]) for leg in ORDER] == [1, 2, 3, 4, 5]
    # LONGEST FIRST, by the numbers the plan itself prints.
    seconds = [int(plan[leg]["measured_seconds"]) for leg in LANES_LONGEST_FIRST]
    assert seconds == sorted(seconds, reverse=True), seconds
    assert "lanes=3" in result.stdout
    assert all(plan[leg]["state"] == "RUN" for leg in ORDER)

    assert not campaign.results.exists(), "a dry run created the results root"
    assert campaign.journal() == [], "a dry run launched a leg"
    assert not campaign.caffeinate_seen.exists(), "a dry run took a sleep assertion"


def test_every_leg_is_handed_the_arguments_and_directory_it_always_was(campaign):
    for lanes in ("1", "2", "3"):
        plan = _plan(campaign.invoke("--dry-run", "--lanes", lanes).stdout)
        for leg, expected in LEG_ARGUMENTS.items():
            command = plan[leg]["command"]
            python = command.index(sys.executable)
            assert command[python + 1] == "-u"
            assert command[python + 2] == str(campaign.stub)
            assert command[python + 3].endswith("gate_dispatch_metal_route.py")
            assert command[python + 4:] == ["--", "--out", str(campaign.run / leg),
                                            *expected], (lanes, leg)


def test_dry_run_environment_is_isolated_per_leg_under_a_hostile_shell(campaign):
    result = campaign.invoke("--dry-run", "--lanes", "3")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ENVIRONMENT CHECK ok" in result.stdout
    plan = _plan(result.stdout)
    witness = str(campaign.run / "band_witness" / "gate.json")
    for leg in ORDER:
        block = plan[leg]
        carried = [name for name in PROBE_VARIABLES if name in block["set"]]
        if leg == "shipped_expansion_probe":
            assert carried == list(PROBE_VARIABLES)
            assert block["set"]["MEEP_GPU_METAL_EXPANSION_PROBE"] == \
                str(campaign.probes / "special_kz.json")
            assert block["set"]["MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE"] == \
                str(campaign.complex_probe)
        else:
            assert carried == [], f"{leg} is handed a probe licence: {carried}"
            assert set(PROBE_VARIABLES) <= set(block["unset"]), leg
        pinned = PINNED_OFF.get(leg, {})
        assert set(SCRUBBED) - set(pinned) <= set(block["unset"]), leg
        assert not (set(SCRUBBED) - set(pinned)) & set(block["set"]), leg
        for name, value in pinned.items():
            assert block["set"][name] == value, (leg, name)
            assert name not in block["unset"], (leg, name)
        if leg == "band_witness":
            # The producer of the artifact is handed no path to it, as before.
            assert WITNESS_VARIABLE in block["unset"]
        else:
            assert block["set"][WITNESS_VARIABLE] == witness, leg
        # The command IS the plan: the leg's own sleep assertion, then `env`, where
        # every unset is a `-u NAME` and every set a NAME=value.
        command = block["command"]
        assert command[:3] == [str(campaign.caffeinate), "-i", "/usr/bin/env"]
        for name in block["unset"]:
            assert command[command.index(name) - 1] == "-u", (leg, name)
        for name, value in block["set"].items():
            assert f"{name}={value}" in command, (leg, name)

    # The script's own read-back of the EFFECTIVE environment agrees, leg by leg.
    reports = {line.split()[1]: line for line in result.stdout.splitlines()
               if line.startswith("ENVIRONMENT ") and "CHECK" not in line}
    assert set(reports) == set(ORDER)
    for leg, line in reports.items():
        if leg == "band_witness":
            assert "scrubbed_absent=6/6" in line and "dispatch=0" in line, line
        else:
            assert "scrubbed_absent=7/7" in line and "dispatch=unset" in line, line
        assert ("probe_variables=4/4" if leg == "shipped_expansion_probe"
                else "probe_variables=0/4") in line, line
        assert ("band_witness=no" if leg == "band_witness"
                else "band_witness=yes") in line, line


@pytest.mark.parametrize("removed, fault", [
    ("    leg_unset+=($PROBE_VARIABLES)\n",
     "shipped carries MEEP_GPU_METAL_EXPANSION_PROBE; only shipped_expansion_probe may"),
    ("  leg_unset=($SCRUBBED)\n",
     "shipped carries MEEP_GPU_SUBNORMAL_POLICY; no leg may"),
    ('    leg_set+=("$WITNESS_VARIABLE=$WITNESS")\n',
     "shipped carries a MEEP_GPU_METAL_BAND_WITNESS that is not this run's"),
    # The band witness's pin: dropped, the hostile shell's value reaches the leg.
    ('    leg_set+=("$DISPATCH_ENABLE=0")\n',
     "band_witness does not carry exactly MEEP_GPU_DISPATCH=0 (it carries "
     "'inherited-from-the-calling-shell')"),
])
def test_the_environment_check_is_armed_and_refuses_before_anything_is_launched(
        campaign, removed, fault):
    # THE CHECK MUST FIRE, not merely pass: each mutant drops one line of the per-leg
    # environment, and under the hostile shell the script's own read-back of what the
    # child would receive has to name the leak and launch nothing.
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count(removed) == 1, removed
    replacement = "  leg_unset=()\n" if "SCRUBBED" in removed else "    :\n"
    mutant = campaign.root / "mutant_campaign.sh"
    mutant.write_text(text.replace(removed, replacement), encoding="utf-8")
    for options in (("--dry-run", "--lanes", "3"), ("--lanes", "3")):
        result = subprocess.run(
            ["zsh", str(mutant), *options, "stamp", str(campaign.probes),
             str(campaign.complex_probe)],
            env=campaign.environment(), text=True, capture_output=True, timeout=60)
        assert result.returncode == 4, result.stdout + result.stderr
        assert f"ENVIRONMENT FAULT  {fault}" in result.stdout + result.stderr
        assert not campaign.results.exists(), "a refused campaign created its run"
        assert campaign.journal() == []


def test_dry_run_by_environment_variable_is_the_same_promise(campaign):
    env = campaign.environment()
    env["DRY_RUN"] = "1"
    result = campaign.invoke("--lanes", "2", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("DRY RUN: nothing is created, written or launched")
    assert not campaign.results.exists() and campaign.journal() == []


def test_lanes_setting_option_beats_positional_beats_environment(campaign):
    env = campaign.environment()
    env["LANES"] = "2"
    assert "lanes=2" in campaign.invoke("--dry-run", env=env).stdout
    assert "lanes=3" in campaign.invoke("--dry-run", env=env,
                                        positional_lanes="3").stdout
    assert "lanes=1" in campaign.invoke("--dry-run", "--lanes", "1", env=env,
                                        positional_lanes="3").stdout
    env.pop("LANES")
    assert "lanes=1" in campaign.invoke("--dry-run", env=env).stdout, \
        "the default must stay the sequential campaign"
    refused = campaign.invoke("--dry-run", "--lanes", "0")
    assert refused.returncode == 2 and "positive integer" in refused.stderr


def test_dry_run_names_a_used_directory_and_a_missing_probe_root(campaign):
    (campaign.run / "shipped").mkdir(parents=True)
    (campaign.run / "shipped" / "cases.jsonl").write_text("{}\n")
    (campaign.run / "harness_keep").mkdir()
    (campaign.run / "harness_keep" / "gate.json").write_text(
        json.dumps({"release": {"released": True}, "weld": {"passed": True}}))
    result = campaign.invoke("--dry-run", "--lanes", "2", probes=False)
    plan = _plan(result.stdout)
    assert plan["shipped"]["state"] == "REFUSE"
    assert plan["harness_keep"]["state"] == "SKIP"
    assert plan["shipped_expansion_probe"]["state"] == "NO-PROBE-ROOT"
    assert "WOULD REFUSE shipped" in result.stdout
    assert "WOULD SKIP shipped_expansion_probe" in result.stdout
    assert result.returncode == 1
    assert sorted(p.name for p in campaign.run.iterdir()) == ["harness_keep", "shipped"]


# ---------------------------------------------------------------------------
# The launch path, with a stub where the gate runner goes
# ---------------------------------------------------------------------------


def test_each_process_really_receives_only_its_own_environment(campaign):
    campaign.control({leg: {"alive_together": LANES_LONGEST_FIRST}
                      for leg in LANES_LONGEST_FIRST})
    result = campaign.invoke("--lanes", "3")
    assert result.returncode == 0, result.stdout + result.stderr
    witness = str(campaign.run / "band_witness" / "gate.json")
    for leg in ORDER:
        seen = campaign.seen(leg)
        environ = seen["environ"]
        assert seen["argv"][1:] == ["--", "--out", str(campaign.run / leg),
                                    *LEG_ARGUMENTS[leg]], leg
        pinned = PINNED_OFF.get(leg, {})
        for name in SCRUBBED:
            if name in pinned:
                assert environ.get(name) == pinned[name], (leg, name, environ.get(name))
            else:
                assert name not in environ, f"{leg} inherited {name}"
        carried = {name: environ[name] for name in PROBE_VARIABLES if name in environ}
        if leg == "shipped_expansion_probe":
            assert carried == {
                "MEEP_GPU_METAL_EXPANSION_PROBE": str(campaign.probes / "special_kz.json"),
                "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE":
                    str(campaign.probes / "folded_complex.json"),
                "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE":
                    str(campaign.probes / "cylindrical_complex.json"),
                "MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE": str(campaign.complex_probe),
            }
        else:
            # THE SILENT TRAP: a sibling started while the probe leg is alive must
            # not see its licence, nor the calling shell's.
            assert carried == {}, f"{leg} would consume a probe licence: {carried}"
        if leg == "band_witness":
            assert WITNESS_VARIABLE not in environ
        else:
            assert environ[WITNESS_VARIABLE] == witness, leg
        assert environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
        assert environ["MPLBACKEND"] == "Agg"
        assert environ["PYTHONPATH"] == str(SCRIPT.resolve().parents[2])
        # The driver made the directory, exclusively and empty, before the launch.
        assert seen["out_existed"] and seen["out_listing"] == [], leg


def test_serial_prefix_runs_alone_and_three_lanes_are_alive_together(campaign):
    # `alive_together` makes each lane WAIT for the other two to have started, so a
    # driver that ran them one after another would time out (exit 97), not pass.
    campaign.control({leg: {"alive_together": LANES_LONGEST_FIRST, "sleep": 0.1}
                      for leg in LANES_LONGEST_FIRST})
    result = campaign.invoke("--lanes", "3")
    assert result.returncode == 0, result.stdout + result.stderr
    starts = [leg for leg, event, _ in campaign.journal() if event == "start"]
    assert starts == ORDER, "launch order: serial prefix, then longest lane first"
    witness, keep = campaign.interval("band_witness"), campaign.interval("harness_keep")
    assert witness[1] <= keep[0], "harness_keep started before the witness landed"
    for leg in LANES_LONGEST_FIRST:
        assert keep[1] <= campaign.interval(leg)[0], f"{leg} overlapped harness_keep"


def test_lane_cap_two_starts_the_third_leg_only_when_a_lane_frees(campaign):
    campaign.control({
        "shipped_expansion_probe": {"alive_together": ["shipped_expansion_probe",
                                                       "shipped"], "sleep": 3.0},
        "shipped": {"alive_together": ["shipped_expansion_probe", "shipped"],
                    "sleep": 0.2},
        "harness_flush": {"sleep": 0.1},
    })
    result = campaign.invoke("--lanes", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    probe, shipped, flush = (campaign.interval(leg) for leg in LANES_LONGEST_FIRST)
    assert flush[0] >= shipped[1], "a third process started while two lanes were busy"
    assert flush[0] < probe[1], "the freed lane was left idle until the probe leg ended"
    events = sorted((when, 1 if event == "start" else -1)
                    for leg, event, when in campaign.journal()
                    if leg in LANES_LONGEST_FIRST and event in ("start", "end"))
    alive = peak = 0
    for _, step in events:
        alive += step
        peak = max(peak, alive)
    assert peak == 2, peak
    # Results reach campaign.txt as they land, not held for the slowest lane.
    landed = [line.split()[0] for line in campaign.summary() if " exit=" in line]
    assert landed == ["band_witness", "harness_keep", "shipped", "harness_flush",
                      "shipped_expansion_probe"]


def test_one_lane_is_strictly_sequential(campaign):
    campaign.control({leg: {"sleep": 0.05} for leg in ORDER})
    result = campaign.invoke("--lanes", "1")
    assert result.returncode == 0, result.stdout + result.stderr
    intervals = [campaign.interval(leg) for leg in ORDER]
    for earlier, later in zip(intervals, intervals[1:]):
        assert earlier[1] <= later[0], "two legs overlapped at LANES=1"
    assert [line.split()[0] for line in campaign.summary() if " exit=" in line] == ORDER


def test_a_used_leg_directory_is_refused_by_name_and_never_launched_into(campaign):
    used = campaign.run / "shipped"
    used.mkdir(parents=True)
    (used / "cases.jsonl").write_text('{"case": "pml_2d"}\n')
    result = campaign.invoke("--lanes", "3")
    assert result.returncode == 1, "a refused leg must fail the campaign"
    refusals = [line for line in campaign.summary() if "REFUSED" in line]
    assert len(refusals) == 1 and refusals[0].split()[0] == "shipped", refusals
    assert "without gate.json" in refusals[0]
    launched = {leg for leg, event, _ in campaign.journal() if event == "start"}
    assert launched == set(ORDER) - {"shipped"}
    assert sorted(p.name for p in used.iterdir()) == ["cases.jsonl"], \
        "the refused directory was written into"
    assert not (campaign.run / "shipped.result").exists()
    assert "fail=1" in campaign.summary()[-2]


def test_an_unreleased_leg_is_a_result_recorded_once_and_never_retried(campaign):
    campaign.control({"shipped": {"released": False, "exit": 1}})
    first = campaign.invoke("--lanes", "3")
    assert first.returncode == 1
    lines = [line for line in campaign.summary() if line.startswith("shipped ")]
    assert any("exit=1" in line and "released=False" in line for line in lines), lines
    assert "fail=0" not in (campaign.run / "campaign.txt").read_text()
    assert sum(1 for leg, event, _ in campaign.journal()
               if leg == "shipped" and event == "start") == 1

    # A SECOND INVOCATION RE-RUNS NOTHING: every leg holds a gate.json, the failed one
    # included, and its verdict still fails the campaign.
    second = campaign.invoke("--lanes", "3")
    assert second.returncode == 1
    assert sum(1 for _, event, _ in campaign.journal() if event == "start") == 5
    text = (campaign.run / "campaign.txt").read_text()
    assert "fail=0" not in text
    assert "resumed" in text and text.count("campaign dispatch_metal_route_stamp  started") == 2
    skips = [line for line in text.splitlines() if " SKIP " in line]
    assert len(skips) == 5
    assert any(line.startswith("shipped ") and "released=False" in line for line in skips)


def test_the_parent_waits_for_every_lane_and_reports_every_exit_code(campaign):
    # One lane fails fast with a distinctive code, one dies with no result at all,
    # and the slowest lane is still running when both have ended.
    campaign.control({
        "shipped_expansion_probe": {"alive_together": LANES_LONGEST_FIRST, "sleep": 1.0},
        "shipped": {"alive_together": LANES_LONGEST_FIRST, "exit": 7,
                    "write_gate": False},
        "harness_flush": {"alive_together": LANES_LONGEST_FIRST, "kill_lane": True},
    })
    result = campaign.invoke("--lanes", "3")
    assert result.returncode == 1
    summary = campaign.summary()
    by_leg = {line.split()[0]: line for line in summary if " exit=" in line}
    assert "exit=0" in by_leg["shipped_expansion_probe"]
    assert "released=True" in by_leg["shipped_expansion_probe"], \
        "the parent did not wait for the slowest lane"
    assert "exit=7" in by_leg["shipped"] and "released=unreadable" in by_leg["shipped"]
    assert "exit=?" in by_leg["harness_flush"] and "left no result" in by_leg["harness_flush"]
    outcome = next(line for line in summary if line.startswith("legs  "))
    for expected in ("band_witness=exit 0", "harness_keep=exit 0",
                     "shipped_expansion_probe=exit 0", "shipped=exit 7",
                     "harness_flush=exit ?"):
        assert expected in outcome, outcome
    assert summary[-2].endswith("fail=1")


def test_campaign_txt_has_one_writer_and_lands_lanes_in_a_fixed_leg_order(campaign):
    # The three lanes END IN REVERSE of the fixed order and all inside one poll
    # interval, so only a parent that lands them itself, in leg order, writes this.
    campaign.control({
        "shipped_expansion_probe": {"alive_together": LANES_LONGEST_FIRST, "sleep": 1.4},
        "shipped": {"alive_together": LANES_LONGEST_FIRST, "sleep": 0.8},
        "harness_flush": {"alive_together": LANES_LONGEST_FIRST, "sleep": 0.2},
    })
    env = campaign.environment()
    env["METAL_CAMPAIGN_POLL_SECONDS"] = "6"
    result = campaign.invoke("--lanes", "3", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    ended = [leg for leg, event, _ in campaign.journal()
             if event == "end" and leg in LANES_LONGEST_FIRST]
    assert ended == LANES_LONGEST_FIRST[::-1], "the stub legs did not end in reverse"

    summary = campaign.summary()
    landed = [line for line in summary if " exit=" in line]
    assert [line.split()[0] for line in landed] == ORDER
    shape = re.compile(r"^(\S+)\s+exit=(\d+)  released=(True|False)\s+(\d+)s  (\S+)  "
                       r"landed (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)$")
    for line in landed:
        match = shape.match(line)
        assert match, f"not one whole result line: {line!r}"
        leg = match.group(1)
        assert match.group(5) == str(campaign.run / f"{leg}.log")
        # THE LANE WROTE ITS LINE TO ITS OWN FILE; the parent copied it, verbatim.
        result_file = (campaign.run / f"{leg}.result").read_text().splitlines()
        assert f"line {line}" in result_file, (leg, result_file)
    starts = [line for line in summary if " START " in line]
    assert [line.split()[0] for line in starts] == ORDER
    assert all(re.search(r"START  \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$", line)
               for line in starts), starts
    for leg in ORDER:
        start = next(i for i, line in enumerate(summary) if line.startswith(f"{leg} ")
                     and " START " in line)
        end = next(i for i, line in enumerate(summary) if line.startswith(f"{leg} ")
                   and " exit=" in line)
        assert start < end
    # What the script printed is what it appended: one stream, one writer.
    assert result.stdout.splitlines() == summary
    # The closing lines: exit codes, wall against leg seconds, then the verdict the
    # Metal contract test reads, then the next step.
    assert summary[-4].startswith("legs  ")
    assert re.match(r"^wall clock \d+s against \d+s of leg seconds run by this driver "
                    r"\(\d+\.\d\dx\)  lanes=3$", summary[-3]), summary[-3]
    assert summary[-2].endswith("fail=0") and "finished" in summary[-2]
    assert summary[-1].startswith("NEXT: ")
    assert (campaign.run / "campaign_plan.txt").read_text().count("LEG order=") == 5


def test_band_witness_digest_is_recorded_and_its_absence_refuses_every_later_leg(
        campaign, tmp_path):
    assert campaign.invoke("--lanes", "2").returncode == 0
    artifact = campaign.run / "band_witness" / "gate.json"
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    summary = campaign.summary()
    recorded = [line for line in summary if line.startswith("band witness  sha256=")]
    assert len(recorded) == 1 and digest in recorded[0] and str(artifact) in recorded[0]
    assert summary.index(recorded[0]) < next(
        i for i, line in enumerate(summary) if line.startswith("harness_keep ")), \
        "the digest must be on record before any leg that reads the artifact starts"
    assert "band witness  sha256 unchanged at the end of the campaign" in summary

    missing = Campaign(tmp_path / "second")
    missing.control({"band_witness": {"write_gate": False, "exit": 3}})
    result = missing.invoke("--lanes", "3")
    assert result.returncode == 1
    assert {leg for leg, event, _ in missing.journal() if event == "start"} == \
        {"band_witness"}
    refused = [line.split()[0] for line in missing.summary() if " REFUSED " in line]
    assert refused == ORDER[1:], refused
    assert any(line.startswith("band witness  MISSING") for line in missing.summary())


def test_without_a_probe_root_the_probe_leg_is_skipped_by_name_and_fails(campaign):
    result = campaign.invoke("--lanes", "3", probes=False)
    assert result.returncode == 1
    skipped = [line for line in campaign.summary() if " SKIP " in line]
    assert len(skipped) == 1 and skipped[0].startswith("shipped_expansion_probe ")
    assert "no probe root given" in skipped[0]
    launched = {leg for leg, event, _ in campaign.journal() if event == "start"}
    assert launched == set(ORDER) - {"shipped_expansion_probe"}
    assert not (campaign.run / "shipped_expansion_probe").exists()


def test_the_machine_is_held_awake_by_the_driver_and_by_each_leg_or_the_absence_is_noted(
        campaign, tmp_path):
    assert campaign.invoke("--lanes", "3").returncode == 0
    summary = campaign.summary()
    driver = re.search(r"driver_pid=(\d+)", summary[0]).group(1)
    # ONE assertion watching the driver ...
    assert campaign.caffeinate_seen.read_text().splitlines() == [f"-i -w {driver}"]
    assert any(line.startswith(f"AWAKE  caffeinate -i -w {driver} held")
               and "each leg's command also holds its own" in line for line in summary)
    # ... AND ONE PER LEG, wrapped round the leg's own command, so a leg that outlives
    # a killed driver still holds one. The tool execs the command in place, so what
    # the gate is handed (asserted elsewhere, from what each stub received) is as it
    # was: here, that each leg went through it exactly once, `-i` then `env`.
    wrapped = [line.split() for line in campaign.caffeinate_legs.read_text().splitlines()]
    assert len(wrapped) == 5
    outs = []
    for words in wrapped:
        assert words[:2] == ["-i", "/usr/bin/env"], words[:3]
        outs.append(Path(words[words.index("--out") + 1]).name)
    assert sorted(outs) == sorted(ORDER)

    # A HOST WITHOUT THE TOOL: the campaign still runs, and says what it lacks.
    bare = Campaign(tmp_path / "bare")
    bare.caffeinate.rename(bare.tools / "not-on-the-path")
    for tool in ("date", "tee", "mkdir", "mv", "grep", "cut", "shasum", "sleep",
                 "zsh", "perl", "tail", "ps"):
        found = shutil.which(tool)
        assert found, tool
        (bare.tools / tool).symlink_to(found)
    env = bare.environment(path=str(bare.tools))
    plan = _plan(bare.invoke("--dry-run", "--lanes", "1", env=env).stdout)
    assert all(plan[leg]["command"][0] == "/usr/bin/env" for leg in ORDER)
    result = bare.invoke("--lanes", "1", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any("AWAKE  NOTE caffeinate is absent" in line for line in bare.summary())
    assert not bare.caffeinate_seen.exists() and not bare.caffeinate_legs.exists()


def test_a_gate_json_the_runner_never_finalized_is_refused_not_skipped(campaign, tmp_path):
    # THE WINDOW: the gate writes gate.json itself, already carrying
    # release.released, before the runner replaces it with the welded one. A process
    # killed in between leaves THIS -- released=True and no `weld` -- and it must not
    # read as a finished, released leg.
    unfinalized = campaign.run / "shipped"
    unfinalized.mkdir(parents=True)
    written = json.dumps({"release": {"released": True, "reasons": []}})
    (unfinalized / "gate.json").write_text(written)
    truncated = campaign.run / "harness_flush"
    truncated.mkdir()
    (truncated / "gate.json").write_text('{"release": {"released": tr')

    dry = campaign.invoke("--dry-run", "--lanes", "3")
    plan = _plan(dry.stdout)
    assert plan["shipped"]["state"] == "REFUSE" and plan["harness_flush"]["state"] == "REFUSE"
    assert "WOULD REFUSE shipped: " in dry.stdout and "never finalized" in dry.stdout
    assert dry.returncode == 1

    result = campaign.invoke("--lanes", "3")
    assert result.returncode == 1
    refusals = {line.split()[0]: line for line in campaign.summary() if " REFUSED " in line}
    assert set(refusals) == {"shipped", "harness_flush"}
    assert "never finalized (unfinalized(True))" in refusals["shipped"]
    assert "never finalized (unreadable(" in refusals["harness_flush"]
    assert not any(" SKIP " in line for line in campaign.summary())
    launched = {leg for leg, event, _ in campaign.journal() if event == "start"}
    assert launched == set(ORDER) - {"shipped", "harness_flush"}
    assert (unfinalized / "gate.json").read_text() == written
    assert sorted(p.name for p in unfinalized.iterdir()) == ["gate.json"]

    # A LEG THAT LANDS ONE under this driver fails the campaign by name, and a band
    # witness in that state is not handed to the legs that would read it.
    landing = Campaign(tmp_path / "landing")
    landing.control({"harness_flush": {"finalized": False}})
    assert landing.invoke("--lanes", "3").returncode == 1
    landed = next(line for line in landing.summary()
                  if line.startswith("harness_flush ") and " exit=" in line)
    assert "exit=0" in landed and "released=unfinalized(True)" in landed

    witness = Campaign(tmp_path / "witness")
    witness.control({"band_witness": {"finalized": False}})
    assert witness.invoke("--lanes", "3").returncode == 1
    assert {leg for leg, event, _ in witness.journal() if event == "start"} == \
        {"band_witness"}
    assert any(line.startswith("band witness  UNFINALIZED") for line in witness.summary())
    assert [line.split()[0] for line in witness.summary() if " REFUSED " in line] == ORDER[1:]


def test_each_lane_records_its_gates_pid_and_a_stopped_gate_lands_as_a_result(campaign):
    # The shipped stub would sleep for a minute; the test stops it the way the LANE
    # line says to, and the campaign must land that as a result like any other.
    campaign.control({"shipped": {"sleep": 60.0}})
    driver = campaign.start("--lanes", "3")
    try:
        line = campaign.wait_for_line("shipped                    LANE ")
        gate_pid = int(re.search(r"gate_pid=(\d+)", line).group(1))
        assert f"kill -TERM {gate_pid}" in line
        # THE PID ON RECORD IS THE GATE'S OWN, not the lane shell's.
        deadline = time.time() + 30
        while not (campaign.stub_dir / "seen.shipped.json").exists():
            assert time.time() < deadline
            time.sleep(0.05)
        time.sleep(0.2)
        assert campaign.seen("shipped")["pid"] == gate_pid
        assert int(re.search(r"lane_shell=(\d+)", line).group(1)) != gate_pid
        assert (campaign.run / "shipped.pid").read_text().split() == \
            ["driver", str(driver.pid), "gate_pid", str(gate_pid), "running"]
        os.kill(gate_pid, signal.SIGTERM)
        stdout, _ = driver.communicate(timeout=60)
    finally:
        if driver.poll() is None:
            driver.kill()
    assert driver.returncode == 1, stdout
    summary = campaign.summary()
    landed = next(line for line in summary if line.startswith("shipped ") and " exit=" in line)
    assert "exit=143" in landed and "released=unreadable(" in landed
    assert (campaign.run / "shipped.pid").read_text().split()[-1] == "ended"
    lanes = [line.split()[0] for line in summary if " LANE   gate_pid=" in line]
    assert sorted(lanes) == sorted(LANES_LONGEST_FIRST), "one LANE line per lane, none serial"
    assert any(line.startswith("SIGNALS  lanes ignore SIGINT and SIGQUIT") for line in summary)
    for leg in ("shipped_expansion_probe", "harness_flush"):
        assert any(line.startswith(f"{leg} ") and "exit=0" in line for line in summary)
    # WHAT THE SIGNALS LINE SAYS IS TRUE OF THE PROCESSES: a lane's gate starts with
    # SIGINT ignored, and a serial leg is handed whatever the driver itself has.
    serial = campaign.seen("band_witness")["sigint_ignored"]
    assert campaign.seen("harness_keep")["sigint_ignored"] == serial
    for leg in LANES_LONGEST_FIRST:
        assert campaign.seen(leg)["sigint_ignored"] is True, leg


def test_one_lane_launches_every_leg_in_the_foreground_and_records_no_lane_line(campaign):
    assert campaign.invoke("--lanes", "1").returncode == 0
    summary = campaign.summary()
    assert not any(" LANE   gate_pid=" in line or line.startswith("SIGNALS")
                   for line in summary)
    assert not list(campaign.run.glob("*.pid"))
    dispositions = {campaign.seen(leg)["sigint_ignored"] for leg in ORDER}
    assert len(dispositions) == 1, "at LANES=1 every leg is launched the same way"


def test_a_second_driver_is_refused_while_the_first_is_alive_and_a_dead_one_resumes(
        campaign, tmp_path):
    campaign.control({"band_witness": {"sleep": 4.0}})
    first = campaign.start("--lanes", "3")
    try:
        campaign.wait_for_line("band_witness               START")
        before = (campaign.run / "campaign.txt").read_text()
        dry = campaign.invoke("--dry-run", "--lanes", "3")
        assert dry.returncode == 1
        assert f"WOULD REFUSE the campaign (exit 3): driver pid {first.pid}" in dry.stdout
        second = campaign.invoke("--lanes", "3")
        assert second.returncode == 3, second.stdout + second.stderr
        assert f"REFUSING: driver pid {first.pid} started this campaign" in second.stderr
        assert second.stdout == ""
        assert (campaign.run / "campaign.txt").read_text().startswith(before)
        stdout, _ = first.communicate(timeout=120)
    finally:
        if first.poll() is None:
            first.kill()
    assert first.returncode == 0, stdout
    text = (campaign.run / "campaign.txt").read_text()
    assert text.count("  started ") == 1 and "resumed" not in text
    assert sum(1 for _, event, _ in campaign.journal() if event == "start") == 5

    # A DRIVER THAT DIED left `started` with no `finished`, and its pid is dead (or is
    # some other program by now): the next invocation resumes, as it always did.
    dead = Campaign(tmp_path / "dead")
    gone = subprocess.Popen(["true"])
    gone.wait()
    dead.run.mkdir(parents=True)
    (dead.run / "campaign.txt").write_text(
        "campaign dispatch_metal_route_stamp  started 2026-01-01T00:00:00Z  lanes=3  "
        f"driver_pid={gone.pid}\n")
    assert dead.invoke("--lanes", "3").returncode == 0
    assert "resumed" in (dead.run / "campaign.txt").read_text()
    # A LIVE pid that is not this script -- the test's own -- is not a driver either.
    alive = Campaign(tmp_path / "alive")
    alive.run.mkdir(parents=True)
    (alive.run / "campaign.txt").write_text(
        "campaign dispatch_metal_route_stamp  started 2026-01-01T00:00:00Z  lanes=3  "
        f"driver_pid={os.getpid()}\n")
    assert alive.invoke("--lanes", "3").returncode == 0


def test_the_callers_switches_are_put_on_record_and_the_scrub_list_is_named(campaign):
    dry = campaign.invoke("--dry-run", "--lanes", "3")
    ran = campaign.invoke("--lanes", "3")
    assert dry.returncode == 0 and ran.returncode == 0
    for lines in (dry.stdout.splitlines(), campaign.summary()):
        switches = next(line for line in lines if line.startswith("SWITCHES "))
        assert "MEEP_GPU_WARM=(unset)" in switches
        assert "MEEP_GPU_KERNEL_TABLE=metal" in switches
        assert "MEEP_GPU_BACKEND_PREFERENCE=(unset)" in switches
        exported = next(line for line in lines if line.startswith("CALLER exported: "))
        for name in SCRUBBED:
            if name == DISPATCH_ENABLE:
                assert (f"{name} (scrubbed from every leg; band_witness is handed 0)"
                        in exported), exported
            else:
                assert f"{name} (scrubbed from every leg)" in exported
        for name in (*PROBE_VARIABLES, WITNESS_VARIABLE):
            assert f"{name} (replaced or unset on each leg)" in exported
        assert "MEEP_GPU_CORPUS_ROOT=/somewhere/else (reaches every leg)" in exported
    # RECORDED MEANS NOT SCRUBBED: what the line says reaches every leg, did.
    for leg in ORDER:
        environ = campaign.seen(leg)["environ"]
        assert environ["MEEP_GPU_KERNEL_TABLE"] == "metal", leg
        assert "MEEP_GPU_WARM" not in environ and "MEEP_GPU_SUBNORMAL_INSTALL" not in environ


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
