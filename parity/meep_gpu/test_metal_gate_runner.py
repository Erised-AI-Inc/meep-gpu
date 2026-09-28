"""Host-only tests for the universal Metal gate provenance writer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from parity.meep_gpu import metal_gate_runner as runner


def _source(path: Path, text: str = "answer = 42\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


#: The two lines a Metal gate must carry to route through the universal writer.
_WRITER_IMPORT = "from metal_gate_runner import run_current_measurement"
_WRITER_EXIT = "raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))"

#: A FLOOR, NOT AN EQUALITY, and the distinction is the whole point of this
#: comment. This assertion read ``== 30`` while the tree held 42 gates, and
#: because it sat on line 2 of the test body the per-gate contract below — the
#: assertions this test exists for — NEVER RAN. Twelve gates, including several
#: added on 2026-08-21, were never checked against the writer contract at all.
#:
#: Measured when the pin was found: the two real assertions pass on all 42, so
#: the stale count was hiding nothing. That is luck, not design. An equality on a
#: growing set is a tripwire pointed at yourself: it fires on the ordinary act of
#: adding a gate, and the cheapest way to silence it — bump the number — rebuilds
#: it one gate later. A floor ratchets instead: it still catches the suite
#: silently collapsing, and it does not fire when the tree grows.
_GATE_FLOOR = 42


def test_every_checked_in_metal_gate_uses_the_universal_measurement_writer():
    gates = runner.discover_gates()

    # THE CONTRACT FIRST, AND OVER EVERY DISCOVERED GATE. Collected rather than
    # asserted per gate so one offender cannot hide the others behind it — the
    # same reason this test's own count must not stand in front of the loop.
    offenders = {}
    for gate in gates:
        text = gate.read_text(encoding="utf-8")
        missing = [needle for needle in (_WRITER_IMPORT, _WRITER_EXIT)
                   if needle not in text]
        if missing:
            offenders[gate.name] = missing
    assert not offenders, (
        f"{len(offenders)} of {len(gates)} Metal gates do not route through the "
        f"universal measurement writer: {offenders}")

    assert len(gates) >= _GATE_FLOOR, (
        f"discover_gates() found {len(gates)}, below the floor of {_GATE_FLOOR}. "
        f"Gates do not normally disappear — either discovery broke or the suite "
        f"shrank, and both are worth stopping for. Raise the floor deliberately "
        f"when gates are retired; do not lower it to make this pass.")


def test_no_metal_gate_escapes_discovery_by_being_named_off_pattern():
    """The property the stale count was a bad proxy for.

    ``discover_gates`` IS ``glob("gate_metal_*.py")``, so asserting it equals the
    directory listing would be tautological and asserting a NUMBER only tracks the
    tree until someone adds a file. The real hazard the count could never catch is
    a gate whose NAME misses the pattern — ``gate_mtl_x.py``,
    ``gate_metalx.py`` — which the glob skips in silence and which therefore never
    gets its writer contract checked.

    So this asks the question from the other side: every file here that both is
    named like a gate AND routes through the universal writer must be discovered.

    PROBES ARE EXCLUDED BY MEASUREMENT, not by assumption: on 2026-08-21 exactly
    two non-gate files carried the writer's exit line — this test module, which
    holds it as a string literal, and probe_metal_eop_interleave_dependency.py, a
    probe that legitimately routes through the same writer. Probes are a different
    population with their own discovery function, so the name filter below is what
    keeps this check about gates.
    """
    discovered = set(runner.discover_gates())
    routed_gates = set()
    for path in sorted(runner.HERE.glob("*.py")):
        if not path.name.startswith("gate_"):
            continue
        if _WRITER_EXIT in path.read_text(encoding="utf-8"):
            routed_gates.add(path)
    missed = sorted(path.name for path in routed_gates - discovered)
    assert not missed, (
        f"these files are named like Metal gates and route through the universal "
        f"writer, but discover_gates() does not find them, so their contract is "
        f"never checked: {missed}")


def test_every_standalone_expansion_probe_is_an_explicit_measurement_target():
    assert [path.name for path in runner.discover_probes()] == [
        "probe_metal_beta_expansion.py",
        "probe_metal_folded_complex_expansion.py",
        "probe_metal_cylindrical_complex.py",
    ]


def test_runtime_digests_follow_module_file_after_import_not_the_cwd(tmp_path: Path):
    root = tmp_path / "api"
    gate = _source(root / "parity" / "gate.py")
    imported = _source(root / "meep_gpu" / "actual_module.py", "value = 7\n")
    module = ModuleType("shadowed_name")
    module.__file__ = str(imported)
    recorded = runner.runtime_source_sha256(
        gate_path=gate, modules=(module,), root=root)
    assert recorded[str(imported)] == hashlib.sha256(imported.read_bytes()).hexdigest()
    assert str(gate) in recorded
    assert all(Path(path).is_absolute() for path in recorded)


def test_finalize_replaces_legacy_source_labels_and_welds_runtime_bytes(tmp_path: Path):
    root = tmp_path / "api"
    gate = _source(root / "parity" / "gate.py")
    family = _source(root / "meep_gpu" / "family.py", "value = 7\n")
    module = ModuleType("family")
    module.__file__ = str(family)
    artifact = tmp_path / "results" / "gate.json"
    artifact.parent.mkdir()
    artifact.write_text(json.dumps({"verdict": "PASS", "source_sha256": {"family": "old"}}),
                        encoding="utf-8")

    payload = runner.finalize_artifact(
        artifact_path=artifact, gate_path=gate, exit_code=0,
        modules=(module,), root=root)

    assert payload["release"] == {"released": True, "reasons": []}
    assert payload["gate_reported_source_sha256"] == {"family": "old"}
    assert payload["source_sha256"][str(family)] == hashlib.sha256(family.read_bytes()).hexdigest()
    assert payload["weld"]["passed"] is True
    assert (artifact.parent / "source_sha256.txt").is_file()
    assert runner.check_artifact_weld(payload, root=root) == []


def test_campaign_manifest_rejects_mixed_source_trees(tmp_path: Path):
    root = tmp_path / "api"
    gate = _source(root / "parity" / "gate.py")
    family = _source(root / "meep_gpu" / "family.py", "value = 7\n")
    module = ModuleType("family")
    module.__file__ = str(family)
    artifact = tmp_path / "results" / "gate.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    runner.finalize_artifact(artifact_path=artifact, gate_path=gate, exit_code=0,
                             modules=(module,), root=root)

    _source(family, "value = 8\n")
    artifact.write_text("{}", encoding="utf-8")
    payload = runner.finalize_artifact(
        artifact_path=artifact, gate_path=gate, exit_code=0,
        modules=(module,), root=root)

    assert payload["release"]["released"] is False
    assert any("campaign source manifest disagrees" in reason
               for reason in payload["release"]["reasons"])


def test_runner_intercepts_direct_main_and_writes_a_release_artifact(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "api"
    gate = _source(
        root / "parity" / "gate.py",
        "import argparse\nimport json\n"
        "def main():\n"
        "  parser=argparse.ArgumentParser()\n"
        "  parser.add_argument('--out', required=True)\n"
        "  args=parser.parse_args()\n"
        "  open(args.out, 'w').write(json.dumps({'verdict': 'PASS'}))\n"
        "  return 0\n",
    )
    artifact = tmp_path / "result" / "gate.json"
    artifact.parent.mkdir()
    monkeypatch.setattr(runner, "API_ROOT", root)
    monkeypatch.setattr(runner, "HERE", root / "parity")
    assert runner.run_current_measurement(gate, ("--out", str(artifact))) == 0
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["release"]["released"] is True
    assert payload["source_sha256"][str(gate)] == hashlib.sha256(gate.read_bytes()).hexdigest()


def test_runner_normalizes_the_whole_step_directory_output(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "api"
    gate = _source(
        root / "parity" / "gate_metal_whole_step.py",
        "import argparse\nimport json\nfrom pathlib import Path\n"
        "def main():\n"
        "  parser=argparse.ArgumentParser()\n"
        "  parser.add_argument('--out', required=True)\n"
        "  args=parser.parse_args()\n"
        "  out=Path(args.out)\n"
        "  out.mkdir(parents=True, exist_ok=True)\n"
        "  (out / 'whole_step.json').write_text(json.dumps({'verdict': 'PASS'}))\n"
        "  return 0\n",
    )
    output = tmp_path / "result"
    monkeypatch.setattr(runner, "API_ROOT", root)
    monkeypatch.setattr(runner, "HERE", root / "parity")
    assert runner.run_current_measurement(gate, ("--out", str(output))) == 0
    payload = json.loads((output / "whole_step.json").read_text(encoding="utf-8"))
    assert payload["release"]["released"] is True
    assert (output / "source_sha256.txt").is_file()


def test_explicit_campaign_manifest_is_shared_by_independent_gate_outputs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "api"
    gate = _source(root / "parity" / "gate.py")
    module_path = _source(root / "meep_gpu" / "family.py")
    module = ModuleType("family")
    module.__file__ = str(module_path)
    shared_manifest = tmp_path / "campaign" / "source_sha256.txt"
    monkeypatch.setenv(runner.SOURCE_MANIFEST_ENV, str(shared_manifest))
    for name in ("first", "second"):
        artifact = tmp_path / name / "gate.json"
        artifact.parent.mkdir()
        artifact.write_text("{}", encoding="utf-8")
        payload = runner.finalize_artifact(
            artifact_path=artifact, gate_path=gate, exit_code=0,
            modules=(module,), root=root)
        assert payload["release"]["released"] is True
    assert shared_manifest.is_file()


def test_related_artifact_inherits_only_a_released_and_current_parent_weld(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "api"
    gate = _source(root / "parity" / "gate.py")
    module_path = _source(root / "meep_gpu" / "family.py")
    module = ModuleType("family")
    module.__file__ = str(module_path)
    parent = tmp_path / "gate.json"
    parent.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "API_ROOT", root)
    runner.finalize_artifact(artifact_path=parent, gate_path=gate, exit_code=0,
                             modules=(module,), root=root)
    companion = tmp_path / "probe.json"
    companion.write_text(json.dumps({"backend": "numpy", "patterns": {}}),
                         encoding="utf-8")

    payload = runner.finalize_related_artifact(
        artifact_path=companion, parent_artifact=parent)

    assert payload["release"]["released"] is True
    assert payload["parent_artifact"] == str(parent.resolve())
    assert payload["source_sha256"] == json.loads(parent.read_text())["source_sha256"]
