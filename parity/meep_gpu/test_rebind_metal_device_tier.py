"""A Metal rebind re-cuts each weld's ``device_sha256`` from the run it binds, or refuses.

``device_sha256`` pins the shader text a weld's modules emit
(``device_identity.device_digests``), and ``test_metal_weld_contract`` recomputes it against
the tree. Until this rule only ``mint_metal_weld.py`` wrote it, so a re-gated and rebound
weld kept the device tier of whatever tree first minted it. These tests pin the rule
:func:`rebind_metal_welds.recut_device_tier` states: the re-cut is mint's computation, whole;
it is taken from the checkout only where the bound run imported each pinned path at the
checkout's bytes; a pinned path the run did not import, or whose device digest cannot be
established, skips the entry by name; the pinned path set never moves and an entry without
a device tier is never given one; and a second ``--write`` leaves the ledger byte-identical.

Host-only: no GPU, no MEEP, no Triton. The real ledger is read, never written. The
``main()`` legs pin a ``triton_kernels/`` path, whose device digest ``device_identity``
reads off the file's syntax tree, so a temporary repository root can carry one; a Metal
module's digest needs the module importable from the package, which only the real tree is.
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
from meep_gpu.device_identity import device_digests, triton_kernel_digests  # noqa: E402

STAMP = "2026-10-15_test"
ENVIRONMENT = {"architecture": "applegpu_g13s", "torch": "2.10.0",
               "metal_frontend": "metalfe-32023.850.10", metal_environment.FAST_MATH: None}
POLICY = "flush" + R.POLICY_SUFFIX
KEPT = "meep_gpu/kept.py"
DEVICE = "meep_gpu/triton_kernels/probe_kernels.py"
ENTRY = "metal_probe_device_gate"
PLAIN = "metal_plain_device_gate"
STALE = {"kind": "triton", "digests": {"probe_step": "d" * 64, "gone_step": "d" * 64}}

KERNELS = '''import triton

SCALE = 2.0


@triton.jit
def probe_step(x_ptr):
    x = SCALE


@triton.jit
def added_step(x_ptr):
    x = SCALE + 1.0
'''


# ---------------------------------------------------------------------------
# THE RE-CUT IS MINT'S COMPUTATION, ON THE SHIPPED LEDGER
# ---------------------------------------------------------------------------


def _digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_shipped_device_pin_recomputes_to_a_non_empty_block_of_its_kind():
    """A rebind refuses a pin with no establishable digest, so the shipped pins must have one."""
    ledger = json.loads(R.LEDGER.read_text(encoding="utf-8"))
    pins = {(key, name): block for key in metal_runs.weld_keys(ledger)
            for name, block in (ledger[key].get("device_sha256") or {}).items()}
    assert pins, "no Metal weld pins a device tier; this test would be inert"
    for (key, name), block in sorted(pins.items()):
        kind, digests = device_digests(R._API / name)
        assert digests, f"{key}: {name} recomputes to no device digest"
        assert kind == block["kind"], (key, name, kind, block["kind"])


def test_the_recut_is_what_mint_metal_weld_writes_for_the_family():
    family = "fused_magnetic_pair"
    name = f"meep_gpu/metal_kernels/{family}.py"
    path = R._API / name
    kind, digests = device_digests(R._API / "meep_gpu" / "metal_kernels" / f"{family}.py")
    assert digests, "the family publishes no device digest; pick another"
    recut, why = R.recut_device_tier({"device_sha256": {name: {"kind": kind, "digests": {}}}},
                                     {name: _digest(path)}, "probe/gate.json")
    assert why is None
    assert recut == {name: {"kind": kind, "digests": digests}}


def test_an_entry_without_a_device_tier_is_given_none():
    assert R.recut_device_tier({"source_sha256": {KEPT: "a" * 64}}, {}, "a/gate.json") == (
        None, None)


def test_bytes_the_run_did_not_import_refuse_by_name():
    """Unreachable through ``main()`` while the drift rule runs first; held here directly."""
    name = "meep_gpu/metal_kernels/fused_magnetic_pair.py"
    recut, why = R.recut_device_tier({"device_sha256": {name: {"kind": "metal", "digests": {}}}},
                                     {name: "0" * 64}, "probe/gate.json")
    assert recut is None
    assert why.startswith(f"device_sha256 pins {name}, whose bytes in the checkout differ")


def test_a_pinned_path_absent_from_the_checkout_refuses_by_name():
    name = "meep_gpu/metal_kernels/no_such_module.py"
    recut, why = R.recut_device_tier({"device_sha256": {name: {}}}, {name: "0" * 64},
                                     "probe/gate.json")
    assert recut is None and why == f"device_sha256 pins {name}, which is not in the checkout"


def test_a_device_tier_of_another_shape_refuses():
    recut, why = R.recut_device_tier({"device_sha256": ["meep_gpu/x.py"]}, {}, "a/gate.json")
    assert recut is None and "not a map of pinned paths" in why


# ---------------------------------------------------------------------------
# THE RULE, THROUGH main(), ON A TEMPORARY REPOSITORY ROOT
# ---------------------------------------------------------------------------


def _entry(api: pathlib.Path, *, device: bool = True, device_pinned: bool = True) -> dict:
    """An entry in the per-architecture shape pinning ``KEPT``, and ``DEVICE`` when
    ``device_pinned``, with a stale device tier on ``DEVICE`` when ``device``."""
    pins = [KEPT] + ([DEVICE] if device_pinned else [])
    entry = {"source_sha256": {name: _digest(api / name) for name in pins},
             "code_sha256": {name: code_digest_of_path(api / name) for name in pins},
             "status": "PASS", "purpose": "a weld"}
    if device:
        entry["device_sha256"] = {DEVICE: json.loads(json.dumps(STALE))}
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
    """A temporary repository root: a live package file, a Triton-spelled kernel module,
    a campaign's environment records, and a ledger the test fills."""
    api = tmp_path / "api"
    (api / "meep_gpu" / "triton_kernels").mkdir(parents=True)
    (api / KEPT).write_text("KEPT = 1\n", encoding="utf-8")
    (api / DEVICE).write_text(KERNELS, encoding="utf-8")
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


def _live(api: pathlib.Path) -> dict:
    return {name: _digest(api / name) for name in (KEPT, DEVICE)}


def test_report_mode_names_the_recut_and_writes_nothing(scratch, capsys):
    api, results, ledger = scratch
    text = _write(ledger, {ENTRY: _entry(api)})
    _artifact(results, "probe", _live(api))
    assert R.main(["--stamp", STAMP]) == 0
    out = capsys.readouterr().out
    assert (f"DEVICE-RECUT  {ENTRY}: {DEVICE}: 1 digest(s) moved, added ['added_step'], "
            f"removed ['gone_step']") in out
    assert "1 rebound, 0 skipped" in out and "0 unchanged, 1 device tier(s) moved" in out
    assert ledger.read_text(encoding="utf-8") == text


def test_write_re_cuts_the_device_tier_whole_and_nothing_else_moves(scratch, capsys):
    api, results, ledger = scratch
    before = _entry(api)
    _write(ledger, {ENTRY: before})
    _artifact(results, "probe", _live(api))
    assert R.main(["--stamp", STAMP, "--write"]) == 0
    entry = json.loads(ledger.read_text(encoding="utf-8"))[ENTRY]
    assert entry["device_sha256"] == {
        DEVICE: {"kind": "triton", "digests": triton_kernel_digests(KERNELS)}}
    assert entry["device_sha256"] == {DEVICE: dict(zip(("kind", "digests"),
                                                       device_digests(api / DEVICE)))}
    assert entry["source_sha256"] == before["source_sha256"]
    assert entry["code_sha256"] == before["code_sha256"]
    # THE BOUND DIGEST DOES NOT COVER THE DEVICE TIER, so the run stays live.
    assert entry["runs"]["applegpu_g13s"]["bound_sha256"] == \
        before["runs"]["applegpu_g13s"]["bound_sha256"]
    assert metal_runs.live_architectures(entry) == ("applegpu_g13s",)


def test_a_second_write_is_byte_identical(scratch, capsys):
    api, results, ledger = scratch
    _write(ledger, {ENTRY: _entry(api)})
    _artifact(results, "probe", _live(api))
    assert R.main(["--stamp", STAMP, "--write"]) == 0
    first = ledger.read_text(encoding="utf-8")
    capsys.readouterr()
    assert R.main(["--stamp", STAMP, "--write"]) == 0
    out = capsys.readouterr().out
    assert ledger.read_text(encoding="utf-8") == first
    assert "DEVICE-RECUT" not in out
    assert "1 unchanged, 0 device tier(s) moved" in out


def test_an_entry_without_a_device_tier_gets_none_on_a_write(scratch, capsys):
    api, results, ledger = scratch
    _write(ledger, {PLAIN: _entry(api, device=False)})
    _artifact(results, "plain", _live(api))
    assert R.main(["--stamp", STAMP, "--write"]) == 0
    assert "device_sha256" not in json.loads(ledger.read_text(encoding="utf-8"))[PLAIN]


def test_a_device_pin_the_run_did_not_import_skips_the_entry(scratch, capsys):
    """The pin sits in the device tier only, so the curated-set check cannot catch it first."""
    api, results, ledger = scratch
    entry = _entry(api, device_pinned=False)
    text = _write(ledger, {ENTRY: entry})
    _artifact(results, "probe", {KEPT: _digest(api / KEPT)})
    assert R.main(["--stamp", STAMP, "--write"]) == 1
    out = capsys.readouterr().out
    assert (f"SKIPPED  {ENTRY}: device_sha256 pins {DEVICE}, which "
            f"metal_probe_{STAMP}/gate.json does not record") in out
    assert "0 rebound" in out
    assert ledger.read_text(encoding="utf-8") == text


def test_a_device_pin_recorded_at_other_bytes_skips_the_entry(scratch, capsys):
    """Caught by the drift rule ahead of the device rule; either way, nothing is cut."""
    api, results, ledger = scratch
    text = _write(ledger, {ENTRY: _entry(api)})
    _artifact(results, "probe", {KEPT: _digest(api / KEPT), DEVICE: "e" * 64})
    assert R.main(["--stamp", STAMP, "--write"]) == 1
    out = capsys.readouterr().out
    assert f"SKIPPED  {ENTRY}:" in out and "DEVICE-RECUT" not in out
    assert ledger.read_text(encoding="utf-8") == text


def test_a_module_that_emits_nothing_refuses_rather_than_dropping_the_pin(scratch, capsys):
    api, results, ledger = scratch
    (api / DEVICE).write_text("SCALE = 2.0\n", encoding="utf-8")
    text = _write(ledger, {ENTRY: _entry(api)})
    _artifact(results, "probe", _live(api))
    assert R.main(["--stamp", STAMP, "--write"]) == 1
    out = capsys.readouterr().out
    assert (f"SKIPPED  {ENTRY}: device_sha256 pins {DEVICE}, and device_identity "
            f"establishes no device digest for it on this checkout") in out
    assert ledger.read_text(encoding="utf-8") == text


def test_a_recut_that_moved_the_pinned_path_set_writes_nothing(scratch, monkeypatch):
    """The survival check, armed: a re-cut naming another path is refused whole."""
    api, results, ledger = scratch
    text = _write(ledger, {ENTRY: _entry(api)})
    _artifact(results, "probe", _live(api))
    real = R.recut_device_tier

    def widened(entry, imported, artifact_name):
        recut, why = real(entry, imported, artifact_name)
        return ({**recut, KEPT: {"kind": "triton", "digests": {"x": "0" * 64}}}, why)

    monkeypatch.setattr(R, "recut_device_tier", widened)
    with pytest.raises(SystemExit, match="device_sha256 pins"):
        R.main(["--stamp", STAMP, "--write"])
    assert ledger.read_text(encoding="utf-8") == text
