"""What the round's own writers record names no machine, account, card or local path.

Each test runs a writer on a machine whose host name, login, home, temporary root and card
UUID are known tokens, then holds what it wrote to two checks: ``neutral_ids.residue`` finds
nothing, and no token appears in the serialized output. The readers that compare two stamps
for "the same machine" or follow a recorded path are run on what the writers wrote.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROUND = Path(__file__).resolve().parent
HARNESS = ROUND.parents[1]
for _path in (ROUND, HARNESS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import neutral_ids  # noqa: E402
import display_card  # noqa: E402
import drive_triton_weld_gates as fleet  # noqa: E402
import gate_provenance  # noqa: E402
import pack_results  # noqa: E402
import rebind_triton_welds  # noqa: E402
import round_status  # noqa: E402
import triton_device_identity  # noqa: E402

SECRET = bytes.fromhex("00112233445566778899aabbccddeeff")
UUID = "GPU-1a2b3c4d-0000-1111-2222-333344445555"
HOST = "labbox7.example.org"


@pytest.fixture
def machine(monkeypatch, tmp_path):
    home, tmp = tmp_path / "home" / "someone", tmp_path / "scratchtmp"
    home.mkdir(parents=True)
    tmp.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setattr(neutral_ids.tempfile, "gettempdir", lambda: str(tmp))
    monkeypatch.setattr(neutral_ids.os.path, "expanduser",
                        lambda p: p.replace("~", str(home), 1) if p.startswith("~") else p)
    monkeypatch.setattr(neutral_ids, "_machine_secret", lambda: (SECRET, "/etc/machine-id"))
    monkeypatch.setattr(neutral_ids.socket, "gethostname", lambda: HOST)
    monkeypatch.setattr(neutral_ids, "_node", lambda: HOST.split(".")[0])
    monkeypatch.setattr(neutral_ids, "login", lambda: "someone")
    neutral_ids.reset()
    yield {"home": str(home), "tmp": str(tmp)}
    neutral_ids.reset()


def machine_id() -> str:
    return "m-" + hmac.new(SECRET, neutral_ids.MACHINE_APP_ID, hashlib.sha256).hexdigest()[:16]


def assert_neutral(obj, machine) -> None:
    written = json.dumps(obj, default=str)
    for token in ("labbox7", "someone", machine["home"], UUID, "1a2b3c4d", "4030427"):
        assert token not in written, token
    assert neutral_ids.residue(obj) == []


class _Gate:
    """A gate process: prints its machine and paths, exits 0."""

    def __init__(self, argv, cwd=None, env=None, stdout=None, stderr=None):
        home = neutral_ids.os.environ["HOME"]
        self.stdout = io.BytesIO(f"gate on {HOST} writing {home}/runs/x card {UUID}\n".encode())
        self.returncode = 0
        self.pid = 4030427

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class _Subprocess:
    Popen = _Gate
    PIPE, STDOUT = subprocess.PIPE, subprocess.STDOUT

    def __getattr__(self, name):
        return getattr(subprocess, name)


def test_the_fleet_campaign_names_no_machine_and_its_rows_read_back(machine, monkeypatch):
    """campaign.json, gates.jsonl and gate.log hold the machine id, card ids and
    campaign-relative or labelled paths; round_status reads the rows back."""
    home = machine["home"]
    import test_display_card as td  # noqa: PLC0415 - its fake nvidia-smi

    # The driver holds no reader of its own: it asks the round's rule (display_card), which
    # reads nvidia-smi through display_card.run_smi.
    monkeypatch.setattr(fleet.display_card, "run_smi", td.FakeSmi(td.headless(0, used=3)))
    monkeypatch.setattr(fleet.display_card, "wsl2_host", lambda *a: False)
    monkeypatch.setattr(fleet, "PLACEMENT", None)
    monkeypatch.setattr(fleet, "record_device", lambda python, out_root, env: (
        {"compute_capability": "8.6", "device": "NVIDIA RTX A6000", "hostname": HOST,
         "cuda_visible_devices": "0"}, None))
    raw = {"canonical_verdict": {"released": True, "read_from": "release"},
           "environment": {"hostname": HOST, "python": f"{home}/env/bin/python",
                           "CUPY_CACHE_DIR": f"{home}/runs/caches/cupy_ftz_stripped_x",
                           "USER": "someone", "uuid": UUID}}
    monkeypatch.setattr(fleet, "read_artifact", lambda path: (json.loads(json.dumps(raw)), None))
    monkeypatch.setattr(fleet, "sha256_of", lambda path: "0" * 64)
    monkeypatch.setattr(fleet, "subprocess", _Subprocess())
    lines = []
    monkeypatch.setattr(fleet, "print", lambda message, flush=True: lines.append(message),
                        raising=False)
    out = Path(home) / "runs" / "campaign"
    assert fleet.main(["--gpu", "0", "--forbidden-gpus", "", "--out-root", str(out),
                       "--gates", "fused_electric", "--patience", "0",
                       "--python", f"{home}/env/bin/python"]) == 0
    summary = json.loads((out / "campaign.json").read_text())
    row = summary["rows"][0]
    assert summary["host"] == machine_id() and summary["device"]["hostname"] == machine_id()
    assert row["log"] == "fused_electric/gate.log" and row["cupy_cache_dir"].startswith("caches/")
    assert not Path(row["artifact"]).is_absolute()
    assert row["host_environment"]["USER"] == "<account>"
    assert "ftz_stripped" in row["host_environment"]["CUPY_CACHE_DIR"]
    assert_neutral(summary, machine)
    assert_neutral([json.loads(l) for l in (out / "gates.jsonl").read_text().splitlines()],
                   machine)
    log = (out / "fused_electric" / "gate.log").read_text()
    assert machine_id() in log and "labbox7" not in log and home not in log
    assert neutral_ids.card_id(UUID) in log and UUID not in log
    # The rebind reads "a machine was stamped" off the campaign and labels it.
    assert rebind_triton_welds.campaign_host_note(summary).startswith("the GPU host")
    # round_status resolves the campaign-relative artifact against the campaign directory.
    (out / row["artifact"]).parent.mkdir(parents=True, exist_ok=True)
    (out / row["artifact"]).write_text(json.dumps({"release": {"released": True}}))
    status = round_status.fleet_root(out)
    assert status[0]["gate"] == "fused_electric" and round_status.is_pass(status[0]["state"])


def test_the_stamp_writes_the_artifact_neutrally_in_place(machine):
    """gate_provenance.stamp: every string neutral, the same containers, and a second stamp
    of the same payload changes nothing."""
    rows = [{"out": f"{machine['home']}/runs/x.json", "card": UUID}]
    payload = {"rows": rows, "environment": {"hostname": HOST, "USER": "someone"},
               "verdict": "PASS"}
    stamped = gate_provenance.stamp(payload)
    assert stamped is payload and payload["rows"] is rows
    assert payload["environment"]["hostname"] == machine_id()
    assert_neutral(payload, machine)
    rows.append({"later": f"{machine['home']}/y"})
    gate_provenance.stamp(payload)
    assert payload["rows"][-1] == {"later": "<home>/y"}


def test_the_device_stamp_and_the_same_machine_check(machine, tmp_path):
    """device.json holds the machine id, and the rebind's same-machine rule still compares:
    one machine reads one id; a stamp from another machine is refused."""
    identity = triton_device_identity.device_identity()
    assert identity["hostname"] == machine_id()
    stamp_dir = tmp_path / "campaign"
    stamp_dir.mkdir()
    (stamp_dir / "device.json").write_text(json.dumps(
        {"compute_capability": "8.6", "hostname": identity["hostname"]}))
    fresh = {"environment": {"compute_capability": "8.6", "hostname": machine_id()}}
    capability, merged = rebind_triton_welds.run_capability(fresh, [stamp_dir])
    assert capability == "8.6" and merged["hostname"] == machine_id()
    other = {"environment": {"compute_capability": "8.6", "hostname": "m-" + "0" * 16}}
    capability, why = rebind_triton_welds.run_capability(other, [stamp_dir])
    assert capability is None and "two hostnames" in why


def test_the_pack_leaves_local_files_and_owner_names_out(machine, tmp_path):
    root = tmp_path / "lanes"
    (root / "LAUNCH.lock").mkdir(parents=True)
    (root / "LAUNCH.lock" / "pid").write_text("4030427\n")
    (root / "gates_0").mkdir()
    (root / "gates_0" / "lane.pid").write_text("4030427\n")
    (root / "gates_0" / "progress.log").write_text("ok\n")
    # An H->D lift leg's work directory: the MEEP examples linked in by absolute path, and
    # one file hard-linked under a second name (its link names the archive's first copy).
    corpus = Path(machine["home"]) / "meep" / "python" / "examples"
    corpus.mkdir(parents=True)
    (corpus / "data.h5").write_bytes(b"\x89HDF")
    workdir = root / "gates_0" / "lift" / "workdir"
    workdir.mkdir(parents=True)
    (workdir / "data.h5").symlink_to(corpus / "data.h5")
    (workdir / "examples").symlink_to(corpus)
    (root / "gates_0" / "lift" / "rows.jsonl").write_text("{}\n")
    (root / "gates_0" / "lift" / "rows_again.jsonl").hardlink_to(root / "gates_0" / "lift" / "rows.jsonl")
    archive = tmp_path / "lanes.tgz"
    pack_results.tar_into(archive, root.parent, [root.name])
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
    names = [m.name for m in members]
    owners = {(m.uname, m.gname, m.uid, m.gid) for m in members}
    assert "lanes/gates_0/progress.log" in names
    assert not any(n.endswith(("pid", ".pid")) or ".lock" in n for n in names)
    assert owners == {("", "", 0, 0)}
    assert not any(m.issym() for m in members), "a symbolic link left the machine"
    assert not any(m.linkname.startswith("/") or machine["home"] in m.linkname for m in members)
    assert {m.name: m.linkname for m in members if m.islnk()} in (
        {"lanes/gates_0/lift/rows_again.jsonl": "lanes/gates_0/lift/rows.jsonl"},
        {"lanes/gates_0/lift/rows.jsonl": "lanes/gates_0/lift/rows_again.jsonl"})
    # A results root that is itself a link is refused by name, not packed as a dangling link.
    results = tmp_path / "repo" / "parity" / "meep_gpu" / "results"
    results.mkdir(parents=True)
    (results / "fleet_r_cc86_c1").symlink_to(root)
    index = tmp_path / "INDEX.json"
    index.write_text(json.dumps({"round": "r", "device": "cc86"}))
    with pytest.raises(ValueError, match="symbolic links: fleet_r_cc86_c1"):
        pack_results.pack("part1", tmp_path / "repo", index, root, tmp_path, tmp_path / "out",
                          None, None, lambda out: None)
    assert not (tmp_path / "out").exists()
    source = tmp_path / "preflight.txt"
    source.write_text(f"PASS the reference environment: {machine['home']}/env on {HOST}\n")
    pack_results.copy_neutral(source, tmp_path / "copy.txt")
    copied = (tmp_path / "copy.txt").read_text()
    assert "<home>/env" in copied and machine_id() in copied
    assert machine["home"] not in copied and "labbox7" not in copied


def test_the_package_policy_stamp_labels_its_cache_directory(machine, monkeypatch):
    from meep_gpu import subnormal_policy as sp

    monkeypatch.setenv("CUPY_CACHE_DIR", f"{machine['home']}/caches/cupy_ftz_stripped_lane")
    sp._recorded_roots.cache_clear()
    sp._recorded_login.cache_clear()
    monkeypatch.setattr(sp, "_recorded_login", lambda: "someone")
    try:
        stamp = sp.policy_stamp()
    finally:
        sp._recorded_roots.cache_clear()
    assert stamp["cache_dir"] == "<home>/caches/cupy_ftz_stripped_lane"
    assert machine["home"] not in json.dumps(stamp, default=str)
    assert sp.recorded_value({"x": "/scratch/someone/a", "y": "/usr/lib/someone"}) == {
        "x": "/scratch/<account>/a", "y": "/usr/lib/someone"}
    assert sp.recorded_value([f"-I{machine['home']}/env/include"]) == ["-I<home>/env/include"]
    # The harness's rules, the same: the name leading a component or a hyphen-joined word
    # (under a root label too), a per-account temporary directory named for the numeric id.
    uid = str(sp.os.getuid())
    flat = "-" + "Us" + "ers-someone-project"
    assert sp.recorded_value([
        "/scratch/someone_runs/a", f"{machine['home']}/x/pytest-of-someone/y",
        "/" + "priv" + f"ate/tmp/tool-{uid}/{flat}/s1/cupy", f"/tmp/tool-{uid}0/a",
        "/data/x/metal_gate_someone.py"]) == [
        "/scratch/<account>_runs/a", "<home>/x/pytest-of-<account>/y", "<tmp>/s1/cupy",
        f"/tmp/tool-{uid}0/a", "/data/x/metal_gate_someone.py"]
    monkeypatch.setattr(sp, "_recorded_login", lambda: "meep")
    assert sp.recorded_value(f"{machine['home']}/e/site-packages/meep/x.py /data/meep_runs") == \
        "<home>/e/site-packages/meep/x.py /data/meep_runs"
    for raw in ("/scratch/someone_runs/a", f"{machine['home']}/x/pytest-of-someone/y",
                "/" + "priv" + f"ate/tmp/tool-{uid}/{flat}/s1/cupy"):
        monkeypatch.setattr(sp, "_recorded_login", lambda: "someone")
        assert sp.recorded_value(raw) == neutral_ids.path(raw), raw
    sp._recorded_roots.cache_clear()       # the roots read on the fixture machine


def test_the_package_and_the_harness_keep_one_list_of_record_words():
    from meep_gpu import subnormal_policy as sp

    assert sp._RECORDED_WORDS == neutral_ids.RECORD_WORDS
    assert sp._SYSTEM_ROOTS == neutral_ids._SYSTEM_ROOTS


def test_a_lane_s_rows_and_log_name_no_path_or_card(machine, tmp_path, monkeypatch):
    """run_lane writes every steps.jsonl row and progress.log line neutrally; the resume
    reader still finds the step by its id."""
    import test_display_card as td  # noqa: PLC0415 - its lane and fake nvidia-smi
    import run_lane  # noqa: PLC0415

    lane = td._lane(tmp_path, monkeypatch, td.FakeSmi(td.Card(0)))
    home = machine["home"]
    lane.row({"id": "cuda/x", "state": "released", "artifact": f"{home}/runs/x/gate.json",
              "log": f"{home}/lanes/gates_0/logs/x.log", "reasons": [f"on {HOST} card {UUID}"]})
    lane.say(f"ADE subject manifest -> {home}/runs/x/subjects.txt on {HOST}")
    rows = [json.loads(l) for l in (tmp_path / "lane" / "steps.jsonl").read_text().splitlines()]
    assert rows[0]["artifact"] == "<home>/runs/x/gate.json"
    assert_neutral(rows, machine)
    log = (tmp_path / "lane" / "progress.log").read_text()
    assert "<home>/runs/x/subjects.txt" in log and home not in log and "labbox7" not in log
    assert lane.last_rows()["cuda/x"]["state"] == "released"
    assert run_lane.N is neutral_ids


def test_the_display_report_counts_compute_processes(machine):
    """display_card's rule (the preflight's report asks it) refuses a card that carries another
    account's process without naming the process id; the report counts them
    (test_display_card.test_the_report_names_the_decision)."""
    import test_display_card as td  # noqa: PLC0415

    smi = td.FakeSmi(td.headless(0, apps={4030427: 4970}))
    taken, why = display_card.Placement(smi, wsl2=False, say=lambda m: None).check(0)
    assert taken is False and "GPU 0 carries a compute process" in why
    assert "4030427" not in why


def test_the_bit_identity_probe_writes_its_provenance_neutrally(machine, tmp_path):
    """PROVENANCE.txt beside the bit-identity probe's artifact: the module and probe paths, the
    interpreter and the cache directory under their labels, the machine as its id."""
    import argparse  # noqa: PLC0415
    import probe_fused_kernel_bit_identity as probe  # noqa: PLC0415

    home = machine["home"]
    module = Path(home) / "src" / "track.py"
    module.parent.mkdir(parents=True)
    module.write_text("x = 1\n")
    args = argparse.Namespace(track="cuda", module=str(module), source_mutation=None,
                              host_mutation=None, allclose=False, no_fu_compare=False)
    results = {"environment": {"python": f"{home}/env/bin/python", "device_name": "NVIDIA RTX A6000"},
               "subnormal_policy": {"policy": "keep", "CUPY_CACHE_DIR": f"{home}/c/cupy_ftz_stripped_x"}}
    path = probe.write_provenance(str(tmp_path), args, results)
    text = Path(path).read_text()
    assert "<home>/src/track.py" in text and "<home>/c/cupy_ftz_stripped_x" in text
    assert machine_id() in text and home not in text and "labbox7" not in text


# ------------------------------------------------------------------- a route leg's records ----
#: A route gate as the lane runs it: it writes its provenance, progress log, case rows and control
#: rows raw, as ``gate_dispatch_fused_route.py`` does (a case row copies the plan's subnormal gate,
#: whose ``cupy_cache_dir`` names the keep cache the lane set), then its stamped payloads. With
#: ``--stop-mid-row`` it leaves a row cut short and waits to be stopped.
ROUTE_GATE = r'''
import json, os, sys, time
out = sys.argv[sys.argv.index("--out") + 1]
os.makedirs(out, exist_ok=True)
home, cache = os.environ["HOME"], os.environ["CUPY_CACHE_DIR"]
checkout = os.path.join(home, "meep-gpu")
def say(text):
    with open(os.path.join(out, "progress.log"), "a", encoding="utf-8") as handle:
        handle.write(f"[00:00:01] {text}\n")
def append(name, row):
    with open(os.path.join(out, name), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")
        handle.flush()
with open(os.path.join(out, "provenance.json"), "w", encoding="utf-8") as handle:
    json.dump({"gate": "dispatch_fused_route", "script": checkout + "/parity/meep_gpu/x.py",
               "cuda_visible_devices": "0", "environment": {"LOGNAME": "someone"}},
              handle, indent=1, default=str)
say(f"writing {out}/cases.jsonl on HOSTNAME")
gate = {"cupy_cache_dir": {"before": cache, "after": cache, "changed": False,
                           "provenance": {"marker": cache + ".keep_policy", "marked": True}}}
plan = {"decision": "dispatched", "composition": {"tables_dispatched": ["triton"]},
        "subnormal": {"gate": gate, "stamp": {"cache_dir": "<home>/x", "nvrtc_calls": 3}}}
append("cases.jsonl", {"case": "pml_2d", "verdict": "PASS-FUSED", "case_wall_s": 1.25,
                       "legs": {"fused": {"plan": plan}, "unfused": {"plan": {"decision": "array"}}},
                       "substitution": {"verdict": "EXACT", "fused": {"launches_per_step": 6.0},
                                        "unfused": {"launches_per_step": 8.0}},
                       "max_abs": 1.401298464324817e-45, "ratio": float("nan"), "label": "µm"})
append("cases.jsonl", {"case": "conductive_2d", "verdict": "PASS-FUSED", "case_wall_s": 2.5})
append("cases.jsonl", {"case": "special_kz_2d", "verdict": "ERROR",
                       "error": "OSError: GPU-1a2b3c4d-0000-1111-2222-333344445555 on HOSTNAME",
                       "environment": {"USER": "someone", "CUDA_VISIBLE_DEVICES": "0"},
                       "traceback": f'File "{checkout}/parity/meep_gpu/gate.py", line 7'})
append("controls.jsonl", {"case": "pml_2d", "control": "withheld-consult", "verdict": "CAUGHT",
                          "plan": {"subnormal": {"gate": gate}}})
if "--stop-mid-row" in sys.argv:
    with open(os.path.join(out, "cases.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"case": "pml_3d", "cache": cache})[:-12])
        handle.flush()
    time.sleep(600)
for name in ("summary.json", "gate.json"):
    with open(os.path.join(out, name), "w", encoding="utf-8") as handle:
        json.dump({"gate": "dispatch_fused_route", "release": {"released": True, "reasons": []},
                   "rows": ["<home>/meep-gpu"]}, handle, indent=1)
'''.replace("HOSTNAME", HOST)


def _route_lane(machine, monkeypatch):
    """A lane whose directory, checkout and results lie in the home directory, as a round's do
    on a host that runs it there, with ``ROUTE_GATE`` as the checkout's route gate."""
    import argparse  # noqa: PLC0415
    import run_lane  # noqa: PLC0415

    home = Path(machine["home"])
    checkout = home / "meep-gpu"
    (checkout / "parity" / "meep_gpu").mkdir(parents=True)
    (checkout / "parity" / "meep_gpu" / "gate_dispatch_fused_route.py").write_text(ROUTE_GATE)
    for name in list(neutral_ids.os.environ):
        if name.startswith(run_lane.SCRUB_PREFIXES) or name in run_lane.SCRUB_NAMES:
            monkeypatch.delenv(name)
    args = argparse.Namespace(lane_dir=str(home / "round" / "lanes" / "route_0"),
                              repo=str(checkout), results=str(checkout / "parity" / "meep_gpu" / "results"),
                              python=sys.executable, gpu=0, subs="", libcuda_stub="",
                              corpus_root="", min_timeout=1800, timeout_factor=4, heartbeat=120)
    lane = run_lane.Lane(args)
    monkeypatch.setattr(lane, "mint_keep_cache", lambda cache: cache.mkdir(parents=True) and None)
    out = Path(args.results) / "dispatch_fused_route_r_cc86" / "shipped"
    return lane, out


def _route_step(out: Path, *extra: str, timeout_s=None) -> dict:
    step = {"id": "shipped", "kind": "route", "cache": "keep", "env": {}, "expect_s": 1,
            "args": ["--gpu-id", "0", "--out", str(out), *extra]}
    if timeout_s:
        step["timeout_s"] = timeout_s
    return step


def _leg_files(out: Path) -> dict:
    return {p.name: p for p in sorted(out.iterdir()) if p.is_file()}


def test_a_route_leg_s_records_name_no_home_once_the_lane_has_run_it(machine, monkeypatch):
    """A route leg's case rows, control rows, provenance and progress log leave the machine with
    Part 2: once the gate has exited the lane writes each through the record spellings, so the
    keep cache the rows copy (``legs.fused.plan.subnormal.gate.cupy_cache_dir``) is written
    under ``<home>``, the packer reads every file of the leg clean, and what the readers use
    (the case, its verdict, the composition, the substitution) is as the gate wrote it."""
    lane, out = _route_lane(machine, monkeypatch)
    result = lane.step_route(_route_step(out))
    assert result["rc"] == 0 and result["state"] == "released", result
    files = _leg_files(out)
    rows = [json.loads(l) for l in files["cases.jsonl"].read_text().splitlines()]
    cache = rows[0]["legs"]["fused"]["plan"]["subnormal"]["gate"]["cupy_cache_dir"]
    assert cache["after"] == "<home>/round/lanes/route_0/caches/cache_ftz_stripped_shipped"
    assert cache["before"] == cache["after"]
    assert cache["provenance"]["marker"] == cache["after"] + ".keep_policy"
    assert rows[2]["traceback"] == 'File "<home>/meep-gpu/parity/meep_gpu/gate.py", line 7'
    assert machine_id() in rows[2]["error"] and neutral_ids.card_id(UUID) in rows[2]["error"]
    assert [(r["case"], r["verdict"]) for r in rows] == [
        ("pml_2d", "PASS-FUSED"), ("conductive_2d", "PASS-FUSED"), ("special_kz_2d", "ERROR")]
    assert rows[0]["legs"]["fused"]["plan"]["composition"] == {"tables_dispatched": ["triton"]}
    assert rows[0]["substitution"]["fused"]["launches_per_step"] == 6.0
    assert rows[0]["max_abs"] == 1.401298464324817e-45 and rows[0]["label"] == "µm"
    needles = pack_results._needles()
    for name, path in files.items():
        assert pack_results.leaks_in(path, needles) == [], name
        text = path.read_text()
        for token in ("labbox7", machine["home"], UUID):
            assert token not in text, (name, token)
    assert_neutral(rows, machine)
    assert_neutral(json.loads(files["controls.jsonl"].read_text()), machine)
    provenance = json.loads(files["provenance.json"].read_text())
    assert provenance["script"] == "<home>/meep-gpu/parity/meep_gpu/x.py"
    assert provenance["environment"] == {"LOGNAME": "<account>"}
    assert_neutral(provenance, machine)
    assert rows[2]["environment"] == {"USER": "<account>", "CUDA_VISIBLE_DEVICES": "0"}
    log = files["progress.log"].read_text()
    assert log.startswith("[00:00:01] writing <home>/meep-gpu/parity/meep_gpu/results/"
                          "dispatch_fused_route_r_cc86/shipped/cases.jsonl on "), log
    assert machine_id() in log and log.count("\n") == 1
    # The row that names nothing local is written as the gate wrote it, byte for byte.
    assert files["cases.jsonl"].read_text().splitlines()[1] == json.dumps(
        {"case": "conductive_2d", "verdict": "PASS-FUSED", "case_wall_s": 2.5})
    assert result["finished"] == {"rewritten": {"cases.jsonl": 2, "controls.jsonl": 1,
                                                "progress.log": 1, "provenance.json": 1},
                                  "unfinished": {}}
    assert "shipped: records finished" in (lane.dir / "progress.log").read_text()


def test_a_route_leg_stopped_mid_row_is_finished_too(machine, monkeypatch):
    """A leg the lane stops (its time ran out) is finished all the same, before it is judged:
    the complete rows through ``N.value`` and the row cut short through ``N.text``, so a run
    set aside travels as clean as a released one."""
    lane, out = _route_lane(machine, monkeypatch)
    result = lane.step_route(_route_step(out, "--stop-mid-row", timeout_s=1))
    assert result["timed_out"] is True and result["state"] == "missing", result
    text = (out / "cases.jsonl").read_text()
    assert machine["home"] not in text and "labbox7" not in text
    last = text.splitlines()[-1]
    assert last.startswith('{"case": "pml_3d", "cache": "<home>/round/lanes/route_0/caches/')
    for path in _leg_files(out).values():
        assert pack_results.leaks_in(path) == [], path.name
    assert result["finished"]["rewritten"]["cases.jsonl"] == 3


def test_a_stopped_lane_finishes_its_route_leg_before_it_exits(machine, monkeypatch):
    """The signal handler stops the child and exits the lane (``SystemExit``): the leg's records
    are finished on the way out, and the exit still propagates."""
    lane, out = _route_lane(machine, monkeypatch)
    home = machine["home"]

    def stopped(argv, env, log, timeout, label):
        out.mkdir(parents=True)
        (out / "cases.jsonl").write_text(json.dumps({"case": "pml_2d", "cache": f"{home}/c"}) + "\n")
        raise SystemExit(143)

    monkeypatch.setattr(lane, "run_child", stopped)
    with pytest.raises(SystemExit):
        lane.step_route(_route_step(out))
    assert json.loads((out / "cases.jsonl").read_text())["cache"] == "<home>/c"


@pytest.mark.parametrize("released", [True, False])
def test_a_route_leg_that_existed_before_the_step_is_left_as_it_was(machine, monkeypatch, released):
    """Only the lane's own child is finished: a released leg is carried by the sha256 of every
    file (``carry.py``), and a leg that did not release may still be written by a child an
    interrupted lane left running, so neither is rewritten."""
    lane, out = _route_lane(machine, monkeypatch)
    out.mkdir(parents=True)
    raw = json.dumps({"case": "pml_2d", "cache": f"{machine['home']}/c"}) + "\n"
    (out / "cases.jsonl").write_text(raw)
    (out / "gate.json").write_text(json.dumps({"release": {"released": released}}))
    result = lane.step_route(_route_step(out))
    assert ("skipped" in result) is released and "finished" not in result
    assert (out / "cases.jsonl").read_text() == raw


def test_finishing_a_route_leg_writes_only_what_names_something_local(machine, tmp_path):
    """``finish_route_leg`` rewrites a file only when a string in it names the machine, the
    account, a card or a local path: a clean file is not touched (same inode), the gate's
    stamped payloads, symbolic links, caches and other files are left whatever they hold, a JSON
    file in another layout is read as text, and the report names what changed."""
    import run_lane  # noqa: PLC0415

    home = machine["home"]
    out = tmp_path / "leg"
    (out / "caches").mkdir(parents=True)
    clean = json.dumps({"case": "pml_2d", "ratio": float("inf"), "n": [1, 2.5, None, True]}) + "\n"
    (out / "cases.jsonl").write_text(clean + "\n" + clean)
    (out / "controls.jsonl").write_text(json.dumps({"x": f"{home}/a", "y": 1.0e-45}) + "\n")
    (out / "controls.jsonl").chmod(0o640)
    (out / "provenance.json").write_text(json.dumps({"z": f"{home}/b", "k": [1, 2]}, indent=2))
    (out / "launch.json").write_text(json.dumps({"w": f"{home}/d", "k": 1.5}, indent=1) + "\n")
    (out / "notes.txt").write_text(f"a note on {home}/e\nand a line naming nothing\n")
    (out / "gate.json").write_text(json.dumps({"stamped": f"{home}/kept"}))
    (out / "summary.json").write_text(json.dumps({"stamped": f"{home}/kept"}))
    (out / "plans.json").write_text(json.dumps({"stamped": f"{home}/kept"}))
    (out / "caches" / "index.json").write_text(json.dumps({"cache": f"{home}/kept"}))
    (out / "kernel.bin").write_bytes(f"{home}/kept".encode())
    (tmp_path / "elsewhere.jsonl").write_text(json.dumps({"x": f"{home}/kept"}) + "\n")
    (out / "linked.jsonl").symlink_to(tmp_path / "elsewhere.jsonl")
    inode = (out / "cases.jsonl").stat().st_ino
    report = run_lane.finish_route_leg(out)
    assert report == {"rewritten": {"controls.jsonl": 1, "launch.json": 1, "notes.txt": 1,
                                    "provenance.json": 1}, "unfinished": {}}
    assert (out / "cases.jsonl").read_text() == clean + "\n" + clean
    assert (out / "cases.jsonl").stat().st_ino == inode
    assert (out / "controls.jsonl").read_text() == json.dumps({"x": "<home>/a", "y": 1e-45}) + "\n"
    assert (out / "controls.jsonl").stat().st_mode & 0o777 == 0o640
    assert (out / "provenance.json").read_text() == json.dumps({"z": "<home>/b", "k": [1, 2]},
                                                               indent=2)
    assert (out / "launch.json").read_text() == json.dumps({"w": "<home>/d", "k": 1.5}, indent=1) + "\n"
    assert (out / "notes.txt").read_text() == "a note on <home>/e\nand a line naming nothing\n"
    assert (out / "linked.jsonl").is_symlink()
    for name in ("gate.json", "summary.json", "plans.json"):
        assert (out / name).read_text() == json.dumps({"stamped": f"{home}/kept"}), name
    assert (out / "caches" / "index.json").read_text() == json.dumps({"cache": f"{home}/kept"})
    assert (out / "kernel.bin").read_bytes() == f"{home}/kept".encode()
    assert (tmp_path / "elsewhere.jsonl").read_text() == json.dumps({"x": f"{home}/kept"}) + "\n"
    assert not [p for p in os.listdir(out) if p.endswith(".finishing")]
    assert run_lane.finish_route_leg(out) == {"rewritten": {}, "unfinished": {}}
    assert run_lane.finish_route_leg(tmp_path / "absent") == {"rewritten": {}, "unfinished": {}}


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="a directory's write bit does not hold the superuser back")
def test_a_route_record_that_cannot_be_rewritten_is_named_and_left(machine, tmp_path):
    """A record the lane cannot write back (here a leg directory it may not write in) is named
    under ``unfinished`` with the reason, left as it was, and does not stop the lane: the packer
    refuses it later by name."""
    import run_lane  # noqa: PLC0415

    out = Path(machine["home"]) / "leg"
    out.mkdir()
    raw = json.dumps({"case": "pml_2d", "cache": f"{machine['home']}/c"}) + "\n"
    (out / "cases.jsonl").write_text(raw)
    out.chmod(0o500)
    try:
        report = run_lane.finish_route_leg(out)
    finally:
        out.chmod(0o700)
    assert report["rewritten"] == {} and list(report["unfinished"]) == ["cases.jsonl"]
    assert report["unfinished"]["cases.jsonl"].startswith("PermissionError: ")
    assert machine["home"] not in report["unfinished"]["cases.jsonl"]
    assert "<home>/leg" in report["unfinished"]["cases.jsonl"]
    assert (out / "cases.jsonl").read_text() == raw
