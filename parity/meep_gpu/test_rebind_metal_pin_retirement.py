"""A pinned file that was deleted leaves a Metal weld by name, inside a rebind, and no other way.

``rebind_metal_welds.py`` keeps each entry's curated pin set, so an entry that pins a file
deleted from the package can never rebind: no fresh artifact records a file that is gone,
and the tool skips the entry with "artifact does not record" in every round.
``RETIRED_PINS`` names each such path. These tests pin the rule: the table names exactly
the deleted expansion-refusal module, and only the three welds that pinned it carry it; a
rebind drops a retired pin from both digest maps and reports ``RETIRED-PIN``; a retired
file back in the checkout refuses the run; a non-retired missing file still refuses its
entry; an entry that does not rebind keeps its pin; an artifact that recorded the retired
file cannot bind; and a second ``--write`` leaves the ledger byte-identical.

Host-only: no GPU, no MEEP. The real ledger is read, never written; the ``main()`` legs
write a temporary ledger under a temporary repository root.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import metal_environment  # noqa: E402
import rebind_metal_welds as R  # noqa: E402

from meep_gpu import metal_runs  # noqa: E402
from meep_gpu.code_identity import code_digest_of_path  # noqa: E402

RETIRED = "meep_gpu/expansion_refusal.py"

#: The three welds that pinned the deleted module when the expansion licence existed.
CARRIERS = {
    "metal_complex_conductive_fused_pair_device_gate",
    "metal_complex_no_pml_conductive_device_gate",
    "metal_cylindrical_real_fused_magnetic_pair_device_gate",
}

STAMP = "2026-10-11_test"
ENVIRONMENT = {"architecture": "applegpu_g13s", "torch": "2.10.0",
               "metal_frontend": "metalfe-32023.850.10", metal_environment.FAST_MATH: None}
POLICY = "flush" + R.POLICY_SUFFIX
KEPT = "meep_gpu/kept.py"
GONE = "meep_gpu/gone.py"
ENTRY = "metal_probe_device_gate"
IDLE = "metal_idle_device_gate"


# ---------------------------------------------------------------------------
# THE TABLE AND THE SHIPPED LEDGER
# ---------------------------------------------------------------------------


def test_the_one_retired_pin_is_the_deleted_expansion_refusal_module():
    assert set(R.RETIRED_PINS) == {RETIRED}
    assert R.RETIRED_PINS[RETIRED]
    assert not (R._API / RETIRED).exists(), "the retired module is back in the checkout"
    assert R.retired_files_in_checkout() == []


def test_only_the_three_named_welds_pin_a_retired_file():
    """Before the bind the three carry it; after it, none. No other entry ever may."""
    ledger = json.loads(R.LEDGER.read_text(encoding="utf-8"))
    carriers = {key for key in metal_runs.weld_keys(ledger)
                for field in ("source_sha256", "code_sha256")
                if set(ledger[key].get(field) or {}) & set(R.RETIRED_PINS)}
    assert carriers <= CARRIERS, sorted(carriers - CARRIERS)


# ---------------------------------------------------------------------------
# THE RULE, THROUGH main(), ON A TEMPORARY REPOSITORY ROOT
# ---------------------------------------------------------------------------


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(api: pathlib.Path, *pins: str) -> dict:
    """An entry in the per-architecture shape, pinning ``KEPT`` (live) and ``pins``."""
    source = {KEPT: _digest(api / KEPT)}
    code = {KEPT: code_digest_of_path(api / KEPT)}
    for name in pins:
        source[name] = "e" * 64
        code[name] = "f" * 64
    entry = {"source_sha256": source, "code_sha256": code, "status": "PASS",
             "purpose": "a weld"}
    host = metal_environment.host_line(ENVIRONMENT)
    named = metal_runs.environment_of(host)
    metal_runs.bind_architecture(entry, bound_before=None, architecture="applegpu_g13s", run={
        "host": host, "torch": named["torch"], "metal_frontend": named["metal_frontend"],
        "records": "apps/api/parity/meep_gpu/results/metal_probe_earlier/ - 1 records",
        "recorded_utc": "2026-10-01T00:00:00Z", "artifact_sha256": "c" * 64,
        "subnormal_policy": POLICY,
        "environment_read_from": "metal_environment_earlier/start.json",
        "verdict_read_from": "metal_probe_earlier/gate.json:release.released"})
    return entry


def _artifact(results: pathlib.Path, family: str, imported: dict, released=True):
    path = results / f"metal_{family}_{STAMP}" / "gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"release": {"released": released}, "records": 1,
                                "imported_source_sha256": imported}))


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """A temporary repository root: one live package file, a campaign's environment
    records, and a ledger the test fills. Returns ``(api, results, ledger_path)``."""
    api = tmp_path / "api"
    (api / "meep_gpu").mkdir(parents=True)
    (api / KEPT).write_text("KEPT = 1\n", encoding="utf-8")
    results = api / "parity" / "meep_gpu" / "results"
    environment = results / f"metal_environment_{STAMP}"
    environment.mkdir(parents=True)
    for name in ("start", "end"):
        (environment / f"{name}.json").write_text(json.dumps(ENVIRONMENT))
    ledger = api / "meep_gpu" / "fingerprints.json"
    monkeypatch.setattr(R, "_API", api)
    monkeypatch.setattr(R, "LEDGER", ledger)
    monkeypatch.setattr(R, "RESULTS", results)
    monkeypatch.setattr(metal_environment, "RESULTS", results)
    return api, results, ledger


def _write(ledger: pathlib.Path, entries: dict) -> str:
    text = json.dumps(entries, indent=2, sort_keys=True) + "\n"
    ledger.write_text(text, encoding="utf-8")
    return text


def test_report_mode_names_the_retired_pin_and_writes_nothing(scratch, capsys):
    api, results, ledger = scratch
    text = _write(ledger, {ENTRY: _entry(api, RETIRED)})
    _artifact(results, "probe", {KEPT: _digest(api / KEPT)})
    assert R.main(["--stamp", STAMP]) == 0
    out = capsys.readouterr().out
    assert f"RETIRED-PIN  {ENTRY}: {RETIRED} (" in out
    assert "1 rebound, 0 skipped, 0 without an artifact, 1 pin(s) retired" in out
    assert ledger.read_text(encoding="utf-8") == text


def test_write_drops_the_pin_from_both_maps_and_nothing_else(scratch, capsys):
    api, results, ledger = scratch
    _write(ledger, {ENTRY: _entry(api, RETIRED)})
    _artifact(results, "probe", {KEPT: _digest(api / KEPT)})
    assert R.main(["--stamp", STAMP, "--write"]) == 0
    entry = json.loads(ledger.read_text(encoding="utf-8"))[ENTRY]
    assert entry["source_sha256"] == {KEPT: _digest(api / KEPT)}
    assert entry["code_sha256"] == {KEPT: code_digest_of_path(api / KEPT)}
    assert entry["purpose"] == "a weld" and entry["status"] == "PASS"
    assert metal_runs.shape_reasons(entry) == []
    assert metal_runs.live_architectures(entry) == ("applegpu_g13s",)
    assert entry["runs"]["applegpu_g13s"]["subnormal_policy"] == POLICY


def test_an_entry_that_does_not_rebind_keeps_its_retired_pin(scratch, capsys):
    """No drop without a rebind: a dropped pin with no fresh run would leave the
    entry's recorded run bound to digests nobody re-measured."""
    api, results, ledger = scratch
    idle = _entry(api, RETIRED)
    _write(ledger, {ENTRY: _entry(api, RETIRED), IDLE: idle})
    _artifact(results, "probe", {KEPT: _digest(api / KEPT)})
    _artifact(results, "idle", {KEPT: _digest(api / KEPT)}, released=False)
    assert R.main(["--stamp", STAMP, "--write"]) == 1
    out = capsys.readouterr().out
    assert f"SKIPPED  {IDLE}: not released" in out
    assert f"RETIRED-PIN  {IDLE}:" not in out and f"RETIRED-PIN  {ENTRY}:" in out
    written = json.loads(ledger.read_text(encoding="utf-8"))
    assert written[IDLE] == idle
    assert RETIRED not in written[ENTRY]["source_sha256"]


def test_a_retired_file_back_in_the_checkout_refuses_and_writes_nothing(scratch):
    api, results, ledger = scratch
    text = _write(ledger, {ENTRY: _entry(api, RETIRED)})
    _artifact(results, "probe", {KEPT: _digest(api / KEPT)})
    (api / RETIRED).write_text("BACK = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="the file is in the checkout"):
        R.main(["--stamp", STAMP, "--write"])
    assert ledger.read_text(encoding="utf-8") == text


def test_a_missing_file_that_is_not_retired_still_refuses(scratch, capsys):
    """Every pin outside the table is compared exactly as before the rule existed."""
    api, results, ledger = scratch
    entry = _entry(api, RETIRED, GONE)
    text = _write(ledger, {ENTRY: entry})
    _artifact(results, "probe", {KEPT: _digest(api / KEPT)})
    assert R.main(["--stamp", STAMP, "--write"]) == 1
    out = capsys.readouterr().out
    assert f"SKIPPED  {ENTRY}: artifact does not record ['{GONE}']" in out
    assert "RETIRED-PIN" not in out and "0 rebound" in out
    assert ledger.read_text(encoding="utf-8") == text


def test_an_artifact_that_recorded_the_retired_file_cannot_bind(scratch, capsys):
    api, results, ledger = scratch
    text = _write(ledger, {ENTRY: _entry(api, RETIRED)})
    _artifact(results, "probe", {KEPT: _digest(api / KEPT), RETIRED: "e" * 64})
    assert R.main(["--stamp", STAMP, "--write"]) == 1
    out = capsys.readouterr().out
    assert f"SKIPPED  {ENTRY}: gate.json records retired pin(s) ['{RETIRED}']" in out
    assert "RETIRED-PIN" not in out
    assert ledger.read_text(encoding="utf-8") == text


def test_a_second_write_is_byte_identical_and_retires_nothing(scratch, capsys):
    api, results, ledger = scratch
    _write(ledger, {ENTRY: _entry(api, RETIRED)})
    _artifact(results, "probe", {KEPT: _digest(api / KEPT)})
    assert R.main(["--stamp", STAMP, "--write"]) == 0
    first = ledger.read_text(encoding="utf-8")
    capsys.readouterr()
    assert R.main(["--stamp", STAMP, "--write"]) == 0
    out = capsys.readouterr().out
    assert ledger.read_text(encoding="utf-8") == first
    assert "RETIRED-PIN" not in out and f"{ENTRY}  runs['applegpu_g13s']  (unchanged)" in out
    assert "0 pin(s) retired, 1 unchanged" in out
