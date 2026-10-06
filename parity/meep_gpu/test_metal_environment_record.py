"""A Metal campaign records the environment its welds are certified for, and binds it.

``metal_dispatch`` certifies a Metal weld for each GPU architecture it holds a live run
record for (``runs[<architecture>]``), with the torch and Metal frontend that run
recorded. ``recut_metal_gates.sh`` records the environment before its first gate and
after its last (``metal_environment.py``); the rebind writes the campaign's run under
the architecture that record names, refuses a campaign whose records are missing,
unread or changed, adds a second architecture BESIDE the first on unchanged bytes, and
on bytes that moved refuses to strand another architecture's run unless that
architecture is superseded by name.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metal_environment  # noqa: E402
import mint_metal_weld as mint  # noqa: E402
import rebind_metal_welds as rebind  # noqa: E402

from meep_gpu import fastpath, metal_dispatch, metal_runs  # noqa: E402

STAMP = "2026-10-02_test"
ENVIRONMENT = {"architecture": "applegpu_g13s", "torch": "2.10.0",
               "metal_frontend": "metalfe-32023.850.10", metal_environment.FAST_MATH: None}
#: A real package file, so the rebind's code digest has something to read.
PIN = "meep_gpu/metal_kernels/launch.py"


def _record(results, start=None, end=None):
    directory = results / f"metal_environment_{STAMP}"
    directory.mkdir(parents=True, exist_ok=True)
    for name, environment in (("start", start), ("end", end)):
        if environment is not None:
            (directory / f"{name}.json").write_text(json.dumps(environment))


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(metal_environment, "RESULTS", tmp_path)
    monkeypatch.setattr(rebind, "RESULTS", tmp_path)
    return tmp_path


def test_the_host_line_names_every_fact_the_dispatch_ladder_reads():
    line = metal_environment.host_line(ENVIRONMENT)
    assert metal_dispatch.host_environment(line) == {
        "architecture": "applegpu_g13s", "torch": "2.10.0", "metal_frontend": "32023.850.10"}


@pytest.mark.parametrize("fast_math", [None, "0"])
def test_a_campaign_with_agreeing_records_is_its_environment(results, fast_math):
    """``0`` compiles the same kernels as unset (``probe_metal_fast_math.py``)."""
    environment = dict(ENVIRONMENT, **{metal_environment.FAST_MATH: fast_math})
    _record(results, environment, environment)
    assert metal_environment.campaign(STAMP) == environment


@pytest.mark.parametrize("start,end,refusal", [
    (ENVIRONMENT, None, "end.json is missing"),
    (dict(ENVIRONMENT, architecture=None), ENVIRONMENT, "could not read ['architecture']"),
    (ENVIRONMENT, dict(ENVIRONMENT, torch="2.14.0"), "['torch'] changed"),
    (dict(ENVIRONMENT, **{metal_environment.FAST_MATH: "1"}),
     dict(ENVIRONMENT, **{metal_environment.FAST_MATH: "1"}), "fast-math"),
])
def test_a_campaign_whose_records_cannot_name_one_environment_is_refused(
        results, start, end, refusal):
    _record(results, start, end)
    with pytest.raises(SystemExit) as raised:
        metal_environment.campaign(STAMP)
    assert refusal in str(raised.value), raised.value


OTHER = {"architecture": "applegpu_g15p", "torch": "2.10.0",
         "metal_frontend": "metalfe-32023.850.10", metal_environment.FAST_MATH: None}
POLICY = "flush - native and uncontrollable on MPS; the oracle flushes too"
ENTRY = "metal_probe_device_gate"


def _live_digest():
    return hashlib.sha256((rebind._API / PIN).read_bytes()).hexdigest()


def _run_for(environment):
    """A run record as a writer would have filed it for ``environment``."""
    host = metal_environment.host_line(environment)
    named = metal_runs.environment_of(host)
    return named["architecture"], {
        "host": host, "torch": named["torch"], "metal_frontend": named["metal_frontend"],
        "records": "apps/api/parity/meep_gpu/results/metal_probe_earlier/ - 1 records",
        "recorded_utc": "2026-10-01T00:00:00Z", "artifact_sha256": "c" * 64,
        "subnormal_policy": POLICY,
        "environment_read_from": "metal_environment_earlier/start.json",
        "verdict_read_from": "metal_probe_earlier/gate.json:release.released"}


def _ledger(results, monkeypatch, *environments, pinned=None, artifact=None):
    """A per-architecture ledger with one entry holding a run per ``environments``.

    ``pinned`` is the digest the entry records for :data:`PIN` (default: the live
    file's, which is what the campaign's artifact records, so the bytes have not
    moved); ``artifact`` is the gate artifact the campaign wrote.
    """
    entry = {"source_sha256": {PIN: pinned or _live_digest()}, "code_sha256": {},
             "status": "PASS"}
    for environment in environments:
        architecture, run = _run_for(environment)
        metal_runs.bind_architecture(entry, bound_before=None,
                                     architecture=architecture, run=run)
    path = results / "fingerprints.json"
    path.write_text(json.dumps({ENTRY: entry}, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(rebind, "LEDGER", path)
    written = results / f"metal_probe_{STAMP}" / "gate.json"
    written.parent.mkdir(parents=True)
    written.write_text(json.dumps(artifact or {
        "release": {"released": True}, "imported_source_sha256": {PIN: _live_digest()}}))
    return path


def _entry(path):
    return json.loads(path.read_text())[ENTRY]


def test_a_rebind_files_the_run_under_the_campaigns_architecture(results, monkeypatch):
    _record(results, ENVIRONMENT, ENVIRONMENT)
    path = _ledger(results, monkeypatch, ENVIRONMENT)
    assert rebind.main(["--stamp", STAMP, "--write"]) == 0
    entry = _entry(path)
    assert metal_runs.shape_reasons(entry) == []
    assert metal_runs.live_architectures(entry) == ("applegpu_g13s",)
    run = entry["runs"]["applegpu_g13s"]
    assert run["host"] == metal_environment.host_line(ENVIRONMENT)
    assert run["environment_read_from"] == f"metal_environment_{STAMP}/start.json"
    assert (run["torch"], run["metal_frontend"]) == ("2.10.0", "32023.850.10")
    # THE CURATED POLICY LINE IS CARRIED from this architecture's previous run.
    assert run["subnormal_policy"] == POLICY
    assert "this architecture's previous run" in run["_subnormal_policy_read_from"]
    for field in metal_runs.RUN_FIELDS:
        assert field not in entry, f"{field} written beside the digests"


def test_a_second_architecture_on_the_same_bytes_joins_the_first(results, monkeypatch):
    """A second Mac's round ADDS its run; the first Mac's is byte-identical afterwards."""
    _record(results, ENVIRONMENT, ENVIRONMENT)
    path = _ledger(results, monkeypatch, OTHER)
    before = _entry(path)["runs"]["applegpu_g15p"]
    assert rebind.main(["--stamp", STAMP, "--write"]) == 0
    entry = _entry(path)
    assert metal_runs.live_architectures(entry) == ("applegpu_g13s", "applegpu_g15p")
    assert entry["runs"]["applegpu_g15p"] == before
    report = metal_runs.admission_report({ENTRY: entry}, [ENTRY])
    assert report["admitted"] == ("applegpu_g13s", "applegpu_g15p"), report
    # A FIRST RUN for this architecture: the artifact states no policy, so the line
    # every other live run carries is taken, and the run says from where.
    run = entry["runs"]["applegpu_g13s"]
    assert run["subnormal_policy"] == POLICY
    assert "carried from runs['applegpu_g15p']" in run["_subnormal_policy_read_from"]


def test_a_second_architectures_first_run_carries_the_line_the_first_one_states(
        results, monkeypatch):
    """The policy is the table's: another live run's line wins over the artifact's."""
    _record(results, ENVIRONMENT, ENVIRONMENT)
    path = _ledger(results, monkeypatch, OTHER, artifact={
        "release": {"released": True}, "imported_source_sha256": {PIN: _live_digest()},
        "subnormal_policy": "flush"})
    assert rebind.main(["--stamp", STAMP, "--write"]) == 0
    run = _entry(path)["runs"]["applegpu_g13s"]
    assert run["subnormal_policy"] == POLICY
    assert run["_subnormal_policy_read_from"].startswith(
        "carried from runs['applegpu_g15p']"), run["_subnormal_policy_read_from"]


#: A gate's policy REPORT, as 30 of the 2026-10-04 fleet's artifacts state it.
POLICY_REPORT = {"admitted": True, "attainable": ["flush"], "executor": "mps",
                 "in_force": "flush", "resolved": "flush", "requested": "flush"}


@pytest.mark.parametrize("field,stated", [
    ("subnormal_policy", "flush"),
    ("subnormal_policy", POLICY_REPORT),
    ("subnormal_policy_report", POLICY_REPORT),
])
def test_an_entrys_only_run_takes_the_policy_value_its_artifact_states(
        results, monkeypatch, field, stated):
    """With no run to carry from, the artifact's policy VALUE, never a report's repr."""
    _record(results, ENVIRONMENT, ENVIRONMENT)
    path = _ledger(results, monkeypatch, artifact={
        "release": {"released": True}, "imported_source_sha256": {PIN: _live_digest()},
        field: stated})
    document = json.loads(path.read_text())
    document[ENTRY]["runs"] = {}
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    assert rebind.main(["--stamp", STAMP, "--write"]) == 0
    run = _entry(path)["runs"]["applegpu_g13s"]
    assert run["subnormal_policy"] == "flush" + rebind.POLICY_SUFFIX, run
    assert run["_subnormal_policy_read_from"].startswith("derived from")


def test_an_artifact_that_names_another_gpu_is_skipped_by_name(results, monkeypatch,
                                                               capsys):
    """The artifact's own architecture, where it states one, must be the campaign's."""
    _record(results, ENVIRONMENT, ENVIRONMENT)
    stamped = {"release": {"released": True},
               "imported_source_sha256": {PIN: _live_digest()},
               "environment": {"apple_gpu": {"architecture": "applegpu_g15p"}}}
    path = _ledger(results, monkeypatch, OTHER, artifact=stamped)
    before = path.read_text()
    assert rebind.main(["--stamp", STAMP, "--write"]) == 1
    assert path.read_text() == before
    out = capsys.readouterr().out
    assert ("records environment.apple_gpu.architecture 'applegpu_g15p' and the "
            "campaign's environment records 'applegpu_g13s'") in out, out
    stamped["environment"]["apple_gpu"]["architecture"] = "applegpu_g13s"
    (results / f"metal_probe_{STAMP}" / "gate.json").write_text(json.dumps(stamped))
    assert rebind.main(["--stamp", STAMP, "--write"]) == 0
    assert metal_runs.live_architectures(_entry(path)) == (
        "applegpu_g13s", "applegpu_g15p")


def test_a_rebind_on_moved_bytes_names_the_architecture_it_would_strand(
        results, monkeypatch, capsys):
    _record(results, ENVIRONMENT, ENVIRONMENT)
    path = _ledger(results, monkeypatch, OTHER, pinned="0" * 64)
    before = path.read_text()
    assert rebind.main(["--stamp", STAMP, "--write"]) == 1
    assert path.read_text() == before
    out = capsys.readouterr().out
    assert "applegpu_g15p" in out and "--supersede applegpu_g15p" in out, out


def test_superseding_an_architecture_by_name_rebinds_and_leaves_its_run_stale(
        results, monkeypatch, capsys):
    _record(results, ENVIRONMENT, ENVIRONMENT)
    path = _ledger(results, monkeypatch, OTHER, pinned="0" * 64)
    assert rebind.main(["--stamp", STAMP, "--supersede", "applegpu_g15p", "--write"]) == 0
    entry = _entry(path)
    assert entry["source_sha256"][PIN] == _live_digest()
    assert metal_runs.live_architectures(entry) == ("applegpu_g13s",)
    assert "applegpu_g15p" in entry["runs"], "a superseded run is kept, stale"
    assert entry["runs"]["applegpu_g15p"]["bound_sha256"] != fastpath.bound_digest(entry)
    assert "SUPERSEDED ['applegpu_g15p']" in capsys.readouterr().out


def test_an_entry_in_the_one_run_shape_is_refused_by_name(results, monkeypatch, capsys):
    _record(results, ENVIRONMENT, ENVIRONMENT)
    path = _ledger(results, monkeypatch)
    document = json.loads(path.read_text())
    document[ENTRY]["host"] = "this machine: Apple MPS device applegpu_g13s, torch 2.10.0"
    assert "runs" not in document[ENTRY]
    path.write_text(json.dumps(document))
    assert rebind.main(["--stamp", STAMP, "--write"]) == 1
    assert json.loads(path.read_text()) == document
    assert "run migrate_metal_runs.py first" in capsys.readouterr().out


def test_a_supersede_that_names_no_architecture_is_refused(results, monkeypatch):
    _record(results, ENVIRONMENT, ENVIRONMENT)
    _ledger(results, monkeypatch, ENVIRONMENT)
    with pytest.raises(SystemExit) as raised:
        rebind.main(["--stamp", STAMP, "--supersede", "8.6"])
    assert "not a GPU architecture key" in str(raised.value)


def test_a_rebind_of_a_campaign_with_no_recorded_environment_is_refused(results,
                                                                        monkeypatch):
    _ledger(results, monkeypatch, ENVIRONMENT)
    with pytest.raises(SystemExit) as raised:
        rebind.main(["--stamp", STAMP])
    assert "start.json is missing" in str(raised.value)


# --- the mint: a family's first weld, filed under its architecture ----------------------

SECOND_STAMP = "2026-10-02_second"


@pytest.fixture
def minted(results, monkeypatch):
    """An empty ledger and one released artifact per stamp, as two Macs' fleets leave them."""
    monkeypatch.setattr(mint, "RESULTS", results)
    path = results / "fingerprints.json"
    path.write_text("{}\n")
    monkeypatch.setattr(mint, "LEDGER", path)
    for stamp, environment in ((STAMP, ENVIRONMENT), (SECOND_STAMP, OTHER)):
        directory = results / f"metal_environment_{stamp}"
        directory.mkdir(parents=True)
        for name in ("start", "end"):
            (directory / f"{name}.json").write_text(json.dumps(environment))
        artifact = results / f"metal_probe_{stamp}" / "gate.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(json.dumps({"release": {"released": True}, "records": 3,
                                        "subnormal_policy": "flush",
                                        "imported_source_sha256": {PIN: _live_digest()}}))
    return path


def _mint(stamp, *extra):
    return mint.main(["--family", "probe", "--stamp", stamp, "--purpose", "a test weld",
                      "--pins", PIN, "--write", *extra])


def test_a_minted_weld_holds_its_run_under_the_campaigns_architecture(minted):
    assert _mint(STAMP) == 0
    entry = json.loads(minted.read_text())["metal_probe_device_gate"]
    assert metal_runs.shape_reasons(entry) == []
    assert metal_runs.live_architectures(entry) == ("applegpu_g13s",)
    run = entry["runs"]["applegpu_g13s"]
    assert run["host"] == metal_environment.host_line(ENVIRONMENT)
    assert run["records"].endswith(f"metal_probe_{STAMP}/ - 3 records")
    assert entry["status"] == "PASS" and entry["purpose"] == "a test weld"


def test_a_re_mint_from_a_second_architecture_joins_the_first(minted):
    assert _mint(STAMP) == 0
    first = json.loads(minted.read_text())["metal_probe_device_gate"]["runs"]
    assert _mint(SECOND_STAMP, "--replace") == 0
    entry = json.loads(minted.read_text())["metal_probe_device_gate"]
    assert metal_runs.live_architectures(entry) == ("applegpu_g13s", "applegpu_g15p")
    assert entry["runs"]["applegpu_g13s"] == first["applegpu_g13s"]


def _restate(results, **fields):
    """Rewrite the first stamp's minted artifact with ``fields`` (``None`` removes one)."""
    artifact = results / f"metal_probe_{STAMP}" / "gate.json"
    data = json.loads(artifact.read_text())
    for name, value in fields.items():
        if value is None:
            data.pop(name, None)
        else:
            data[name] = value
    artifact.write_text(json.dumps(data))


def test_a_mint_writes_a_policy_reports_value_and_never_its_repr(minted, results):
    _restate(results, subnormal_policy=None, subnormal_policy_report=POLICY_REPORT)
    assert _mint(STAMP) == 0
    run = json.loads(minted.read_text())["metal_probe_device_gate"]["runs"]["applegpu_g13s"]
    assert run["subnormal_policy"] == "flush" + metal_runs.POLICY_SUFFIX, run


def test_a_mint_refuses_an_artifact_that_names_another_gpu(minted, results):
    _restate(results, environment={"apple_gpu": {"architecture": "applegpu_g15p"}})
    with pytest.raises(SystemExit) as raised:
        _mint(STAMP)
    assert "'applegpu_g15p'" in str(raised.value), raised.value
    assert json.loads(minted.read_text()) == {}


def test_a_mint_refuses_an_artifact_that_states_no_policy(minted, results):
    _restate(results, subnormal_policy=None)
    with pytest.raises(SystemExit) as raised:
        _mint(STAMP)
    assert "states no subnormal policy" in str(raised.value), raised.value
    assert json.loads(minted.read_text()) == {}
