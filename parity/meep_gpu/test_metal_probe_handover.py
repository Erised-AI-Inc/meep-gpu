"""How the Metal fleet hands each gate the expansion probes it cut, and what a gate records.

Host-only, no device. Three defects seen on a Mac without the evidence archive, each
pinned where it lived:

* ``recut_metal_gates.sh`` exported the complex probe only after
  ``gate_metal_complex`` had run in directory order, so earlier-sorting gates that bind
  it ran without it (the refusal leg of ``gate_metal_beta_complex_fused_hd_pair``
  failed there). The script is driven here with a stub interpreter in place of Python.
* ``metal_composition_matrix.prepare_environment`` returned the dated archive candidate
  it found rather than the probe in force, so ``gate_metal_residue_fused_pairs`` and
  ``gate_metal_tranche7_fused_pairs`` refused although every probe was exported.
* ``gate_metal_complex_conductive_fused_pair`` and ``gate_metal_complex_fused_ade_chain``
  named, hashed and refused on one dated archive file while their loader read the probe
  the environment named.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import metal_composition_matrix as matrix  # noqa: E402

SCRIPT = HERE / "recut_metal_gates.sh"
PROBES = ("MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE", "MEEP_GPU_METAL_EXPANSION_PROBE",
          "MEEP_GPU_METAL_FOLDED_COMPLEX_EXPANSION_PROBE",
          "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE")


# ---------------------------------------------------------------------------
# prepare_environment returns what is in force
# ---------------------------------------------------------------------------

@pytest.fixture
def no_probe_environment(monkeypatch, tmp_path):
    for name in PROBES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(matrix, "API_ROOT", str(tmp_path))
    for attribute in ("COMPLEX_PROBE_CANDIDATES", "BETA_PROBE_CANDIDATES",
                      "FOLDED_COMPLEX_PROBE_CANDIDATES",
                      "CYLINDRICAL_COMPLEX_PROBE_CANDIDATES"):
        monkeypatch.setattr(matrix, attribute, (f"archive/{attribute}.json",))
    return tmp_path


def test_a_probe_the_environment_names_is_returned_not_the_archive_candidate(
        no_probe_environment, monkeypatch):
    (no_probe_environment / "archive").mkdir()
    for attribute in ("COMPLEX", "BETA", "FOLDED_COMPLEX", "CYLINDRICAL_COMPLEX"):
        (no_probe_environment / "archive" / f"{attribute}_PROBE_CANDIDATES.json").write_text("{}")
    fresh = {name: str(no_probe_environment / f"fresh_{index}.json")
             for index, name in enumerate(PROBES)}
    for name, path in fresh.items():
        monkeypatch.setenv(name, path)
    returned = matrix.prepare_environment()
    for name in PROBES:
        assert returned[name] == fresh[name] == os.environ[name]


def test_with_nothing_named_the_archive_candidate_is_set_and_returned(
        no_probe_environment):
    (no_probe_environment / "archive").mkdir()
    candidate = no_probe_environment / "archive" / "BETA_PROBE_CANDIDATES.json"
    candidate.write_text("{}")
    returned = matrix.prepare_environment()
    assert returned["MEEP_GPU_METAL_EXPANSION_PROBE"] == str(candidate)
    assert os.environ["MEEP_GPU_METAL_EXPANSION_PROBE"] == str(candidate)
    assert returned["MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE"] is None


def test_on_a_mac_without_the_archive_the_exported_probes_satisfy_the_guards(
        no_probe_environment, monkeypatch):
    """The residue and tranche7 guards read this dict; ``None`` exits the gate."""
    for name in PROBES:
        monkeypatch.setenv(name, str(no_probe_environment / f"{name}.json"))
    returned = matrix.prepare_environment()
    assert all(returned[name] is not None for name in PROBES)


def test_an_empty_variable_is_no_probe_and_is_not_overwritten(no_probe_environment,
                                                              monkeypatch):
    (no_probe_environment / "archive").mkdir()
    (no_probe_environment / "archive" / "COMPLEX_PROBE_CANDIDATES.json").write_text("{}")
    monkeypatch.setenv("MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE", "")
    returned = matrix.prepare_environment()
    assert returned["MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE"] is None
    assert os.environ["MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE"] == ""


def test_the_guarded_gates_read_their_probes_through_prepare_environment():
    for name in ("gate_metal_residue_fused_pairs.py", "gate_metal_tranche7_fused_pairs.py"):
        text = (HERE / name).read_text(encoding="utf-8")
        assert "environment = matrix.prepare_environment()" in text
        assert 'environment.get("MEEP_GPU_METAL_EXPANSION_PROBE")' in text


# ---------------------------------------------------------------------------
# The two complex gates name the probe their loader reads
# ---------------------------------------------------------------------------

CHILD = ("import sys, importlib; sys.path[:0] = [sys.argv[1], sys.argv[1] + '/parity/meep_gpu']; "
         "m = importlib.import_module(sys.argv[2]); "
         "print('PROBE_ARTIFACT=' + str(m.PROBE_ARTIFACT))")


@pytest.mark.parametrize("module", ["gate_metal_complex_conductive_fused_pair",
                                    "gate_metal_complex_fused_ade_chain"])
@pytest.mark.parametrize("named", [True, False])
def test_the_complex_gate_names_the_probe_in_force_and_no_dated_file(module, named,
                                                                      tmp_path):
    text = (HERE / f"{module}.py").read_text(encoding="utf-8")
    assert "metal_complex_audit_2026-08-16" not in text
    environment = {k: v for k, v in os.environ.items() if k not in PROBES}
    environment.update(KMP_DUPLICATE_LIB_OK="TRUE", MEEP_GPU_DISPATCH="0")
    probe = tmp_path / "complex_expansion_probe.json"
    if named:
        environment["MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE"] = str(probe)
    completed = subprocess.run([sys.executable, "-c", CHILD, str(REPO), module],
                               env=environment, capture_output=True, text=True,
                               timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    printed = [text for text in completed.stdout.splitlines()
               if text.startswith("PROBE_ARTIFACT=")]
    assert printed == [f"PROBE_ARTIFACT={probe if named else None}"]


# ---------------------------------------------------------------------------
# recut_metal_gates.sh: the complex probe is cut before every gate that binds it
# ---------------------------------------------------------------------------

STUB = r"""#!/bin/sh
# Records, per call, the .py files named and the probe variables in force.
pys=""
out=""
prev=""
for a in "$@"; do
  case "$a" in *.py) pys="$pys $(basename "$a")";; esac
  [ "$prev" = "--out" ] && out="$a"
  prev="$a"
done
printf '%s|%s|%s\n' "$pys" "${MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE-<unset>}" \
  "${MEEP_GPU_METAL_EXPANSION_PROBE-<unset>}" >> "$STUB_LOG"
case "$pys" in
  " gate_metal_complex.py")
    if [ "${STUB_WRITE_COMPLEX_PROBE:-1}" = 1 ]; then
      mkdir -p "$(dirname "$out")"
      echo '{}' > "$(dirname "$out")/complex_expansion_probe.json"
    fi;;
esac
exit 0
"""

GATES = ("gate_metal_beta_complex_fused_hd_pair.py", "gate_metal_complex.py",
         "gate_metal_complex_conductive_fused_pair.py", "gate_metal_residue_fused_pairs.py",
         "gate_metal_whole_step.py")
HELPERS = ("metal_gate_runner.py", "metal_environment.py", "probe_metal_beta_expansion.py",
           "probe_metal_folded_complex_expansion.py", "probe_metal_cylindrical_complex.py")


def _fleet(tmp_path: Path, write_complex_probe: bool):
    api = tmp_path / "api"
    harness = api / "parity" / "meep_gpu"
    harness.mkdir(parents=True)
    for name in GATES + HELPERS:
        (harness / name).write_text("")
    stub = tmp_path / "stub_python"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.log"
    environment = {k: v for k, v in os.environ.items() if not k.startswith("MEEP_GPU_")}
    environment.update(
        MGPU_SITE_SOURCE_ROOT=str(api), MGPU_SITE_PYTHON=str(stub), STUB_LOG=str(log),
        STUB_WRITE_COMPLEX_PROBE="1" if write_complex_probe else "0",
        # a hostile calling shell: neither may reach any gate
        MEEP_GPU_METAL_COMPLEX_EXPANSION_PROBE=str(tmp_path / "elsewhere" / "complex.json"),
        MEEP_GPU_METAL_EXPANSION_PROBE=str(tmp_path / "elsewhere" / "beta.json"))
    root = tmp_path / "root"
    stamp = "2000-01-01_test"
    completed = subprocess.run(["zsh", str(SCRIPT), str(root), stamp], env=environment,
                               capture_output=True, text=True, timeout=120, check=False)
    calls = [line.split("|") for line in log.read_text().splitlines()]
    gates = [(pys.strip(), complex_probe, beta)
             for pys, complex_probe, beta in calls if pys.strip().startswith("gate_metal_")]
    summary = (root / "recut_summary.txt").read_text()
    fresh = str(harness / "results" / f"metal_complex_{stamp}" / "complex_expansion_probe.json")
    return completed, gates, summary, fresh, root


needs_zsh = pytest.mark.skipif(shutil.which("zsh") is None,
                               reason="the fleet script needs zsh")


@pytest.mark.requires_resource("zsh")
@needs_zsh
def test_gate_metal_complex_runs_first_and_every_later_gate_binds_its_probe(tmp_path):
    completed, gates, summary, fresh, root = _fleet(tmp_path, write_complex_probe=True)
    assert completed.returncode == 0, summary + completed.stderr
    assert [name for name, _c, _b in gates][0] == "gate_metal_complex.py"
    assert sorted(name for name, _c, _b in gates) == sorted(GATES)
    first = gates[0]
    assert first[1] == "<unset>", "the complex gate ran with an inherited complex probe"
    assert first[2] == str(root / "probes" / "special_kz.json")
    for name, complex_probe, beta in gates[1:]:
        assert complex_probe == fresh, f"{name} ran with complex probe {complex_probe}"
        assert beta == str(root / "probes" / "special_kz.json")
    assert summary.rstrip().endswith("VERDICT: ALL GREEN")


@pytest.mark.requires_resource("zsh")
@needs_zsh
def test_a_missing_complex_probe_is_named_and_no_gate_inherits_one(tmp_path):
    completed, gates, summary, _fresh, _root = _fleet(tmp_path, write_complex_probe=False)
    assert completed.returncode == 1
    assert "complex expansion probe" in summary and "MISSING" in summary
    assert summary.rstrip().endswith("VERDICT: AT LEAST ONE GATE FAILED")
    assert all(complex_probe == "<unset>" for _name, complex_probe, _b in gates)
