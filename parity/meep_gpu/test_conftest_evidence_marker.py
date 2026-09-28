"""The evidence-archive probe of the root ``conftest.py``, checked for being ARMED.

``tools/ci/declared_resources.txt`` lists, under ``[evidence_archive]``, the tests
that read a record of the evidence archive of the certification campaigns. The
archive is not part of this repository, so those tests skip by that resource when
it is absent. The harness also WRITES into ``parity/meep_gpu/results/``: a gate run
as ``docs/development/certification.md`` documents it creates the directory. So the
directory is not evidence of the archive, and the probe asks for the marker file
``parity/meep_gpu/results/EVIDENCE_ARCHIVE.json`` instead. When the probe asked only
for the directory, one gate run in a checkout turned every listed test from a
declared skip into a failure.

Two layers. The function the probe calls, ``evidence_archive_absence``, is checked
branch by branch on directories built here. Then the real root conftest is copied
into a scratch tree beside a one-entry ``declared_resources.txt`` and run in a
separate interpreter: no directory, a directory holding only gate output, a valid
marker, a valid marker whose record is missing, and two malformed markers. A
negative control puts the directory-only rule back and must see the listed test run
and fail on the gate-output tree, so a probe that stops reading the marker fails
here by name.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROOT_CONFTEST = ROOT / "conftest.py"


def _load_root_conftest():
    """The root conftest as a module of its own, so no module global is patched.

    Loaded under a name pytest never gives it: the copy that pytest registered as a
    plugin is left alone, and this one defines hooks that nothing calls.
    """
    spec = importlib.util.spec_from_file_location(
        "root_conftest_under_test", ROOT_CONFTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


root_conftest = _load_root_conftest()
absence = root_conftest.evidence_archive_absence
MARKER = root_conftest.EVIDENCE_ARCHIVE_MARKER
KIND = root_conftest.EVIDENCE_ARCHIVE_KIND

#: A record of the kind a gate run leaves: what the directory holds after the
#: documented ``gate_metal_pml.py --out parity/meep_gpu/results/<subject>_<date>/``.
GATE_OUTPUT = "metal_pml_2026-09-28/gate.json"
#: The archived record the listed scratch test reads. A gate run does not write it.
ARCHIVED_RECORD = "archived_record_2026-09-01/record.json"
DIGEST = "0123456789abcdef" * 4


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _results(tmp_path: pathlib.Path, *, marker: str | None = None,
             gate_output: bool = True, record: bool = False) -> pathlib.Path:
    """``<tmp>/parity/meep_gpu/results`` with the named contents."""
    results = tmp_path / "parity" / "meep_gpu" / "results"
    results.mkdir(parents=True)
    if gate_output:
        _write(results / GATE_OUTPUT, json.dumps({"released": True}))
    if record:
        _write(results / ARCHIVED_RECORD, json.dumps({"measured": 42}))
    if marker is not None:
        _write(results / MARKER, marker)
    return results


# ---------------------------------------------------------------------------
# The function the probe calls, branch by branch
# ---------------------------------------------------------------------------

def test_no_results_directory_is_absent_and_says_so(tmp_path):
    why = absence(tmp_path / "parity" / "meep_gpu" / "results")
    assert why is not None and "does not exist" in why, why


def test_a_directory_of_gate_output_without_the_marker_is_not_the_archive(tmp_path):
    why = absence(_results(tmp_path))
    assert why is not None, "a directory holding only gate output counted as the archive"
    assert MARKER in why and "working output directory" in why, why


@pytest.mark.parametrize("document", [
    {"kind": KIND},
    {"kind": KIND, "archive_version": "2026-09-28"},
    {"kind": KIND, "manifest_sha256": DIGEST},
    {"kind": KIND, "archive_version": "1", "manifest_sha256": DIGEST},
], ids=["kind_only", "with_version", "with_digest", "with_both"])
def test_a_valid_marker_makes_the_archive_present(tmp_path, document):
    assert absence(_results(tmp_path, marker=json.dumps(document))) is None


@pytest.mark.parametrize("text, expected", [
    ("{not json", "is not readable JSON"),
    ("", "is not readable JSON"),
    (json.dumps([KIND]), "is not a JSON object (it holds a list)"),
    (json.dumps({}), "has kind None"),
    (json.dumps({"kind": "meep-evidence"}), "has kind 'meep-evidence'"),
    (json.dumps({"kind": KIND, "records": 12}),
     "carries keys its schema does not define: records"),
    (json.dumps({"kind": KIND, "archive_version": ""}), "has archive_version ''"),
    (json.dumps({"kind": KIND, "archive_version": 3}), "has archive_version 3"),
    (json.dumps({"kind": KIND, "manifest_sha256": DIGEST[:-1]}),
     "not 64 lowercase hexadecimal digits"),
    (json.dumps({"kind": KIND, "manifest_sha256": DIGEST.upper()}),
     "not 64 lowercase hexadecimal digits"),
], ids=["unparsable", "empty", "list", "no_kind", "wrong_kind", "unknown_key",
        "empty_version", "numeric_version", "short_digest", "uppercase_digest"])
def test_a_malformed_marker_is_refused_with_its_reason(tmp_path, text, expected):
    why = absence(_results(tmp_path, marker=text))
    assert why is not None, f"a malformed marker counted as the archive: {text!r}"
    assert f"parity/meep_gpu/results/{MARKER}" in why and expected in why, why


def test_a_marker_that_is_not_utf8_is_refused(tmp_path):
    results = _results(tmp_path)
    (results / MARKER).write_bytes(b'{"kind": "\xff"}')
    why = absence(results)
    assert why is not None and "is not readable JSON" in why, why


def test_a_marker_that_is_a_directory_is_refused(tmp_path):
    results = _results(tmp_path)
    (results / MARKER).mkdir()
    why = absence(results)
    assert why is not None and "is not a file" in why, why


def test_the_probe_reads_the_results_directory_beside_the_conftest():
    assert root_conftest._ARCHIVE == ROOT / "parity" / "meep_gpu" / "results"
    assert root_conftest._PROBES["evidence_archive"] is root_conftest._probe_evidence_archive


# ---------------------------------------------------------------------------
# The real conftest, end to end, in a separate interpreter
# ---------------------------------------------------------------------------

_LISTED_TEST = f'''
import json
import pathlib

RECORD = (pathlib.Path(__file__).parent / "parity" / "meep_gpu" / "results"
          / {ARCHIVED_RECORD!r})


def test_reads_an_archived_record():
    assert json.loads(RECORD.read_text(encoding="utf-8"))["measured"] == 42


def test_needs_nothing():
    assert True
'''

_DECLARED = "[evidence_archive]\ntest_reads_the_archive::test_reads_an_archived_record\n"

#: The probe as it read before the marker: the directory alone counted.
_DIRECTORY_ONLY_PROBE = (
    "def _probe_evidence_archive():\n"
    "    if _ARCHIVE.is_dir():\n"
    "        return None\n"
    "    return 'the evidence archive is not part of this repository'\n")


def _run(tree: pathlib.Path, conftest_source: str | None = None) -> dict:
    """pytest over ``tree`` under the real root conftest (or ``conftest_source``).

    Returns the outcome and message of each test, read from the JUnit report,
    which carries skip reasons whole where the terminal summary truncates them.
    """
    if conftest_source is None:
        shutil.copy(ROOT_CONFTEST, tree / "conftest.py")
    else:
        (tree / "conftest.py").write_text(conftest_source, encoding="utf-8")
    _write(tree / "tools" / "ci" / "declared_resources.txt", _DECLARED)
    _write(tree / "test_reads_the_archive.py", _LISTED_TEST)
    report = tree / "report.xml"
    env = dict(os.environ, SUITE_ENFORCE_SKIPS="1")
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tree), "-q", "-p", "no:cacheprovider",
         f"--junitxml={report}"],
        capture_output=True, text=True, env=env, cwd=str(tree))
    assert report.is_file(), result.stdout + result.stderr
    outcomes = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome, message = tag, node.get("message") or ""
                break
        outcomes[case.get("name")] = (outcome, message)
    assert outcomes.get("test_needs_nothing") == ("passed", ""), (
        result.stdout + result.stderr)
    return outcomes


def _listed(outcomes: dict) -> tuple:
    return outcomes["test_reads_an_archived_record"]


def _declared_skip(outcome: tuple, expected: str) -> None:
    kind, message = outcome
    assert kind == "skipped", f"the listed test was not skipped: {outcome}"
    assert "[requires_resource][evidence_archive]" in message, message
    assert expected in message, message


def test_without_a_results_directory_the_listed_test_skips(tmp_path):
    _declared_skip(_listed(_run(tmp_path)), "does not exist")


def test_gate_output_without_the_marker_skips_and_names_the_marker(tmp_path):
    _results(tmp_path)
    _declared_skip(_listed(_run(tmp_path)),
                   f"carries no {MARKER}, so it is a working output directory")


def test_a_valid_marker_runs_the_listed_test_and_it_passes(tmp_path):
    _results(tmp_path, marker=json.dumps({"kind": KIND, "archive_version": "test"}),
             record=True)
    assert _listed(_run(tmp_path)) == ("passed", "")


def test_a_valid_marker_without_the_record_runs_the_listed_test_and_it_fails(tmp_path):
    """The marker claims the archive: a record missing from it is a failure, not a skip."""
    _results(tmp_path, marker=json.dumps({"kind": KIND}))
    kind, message = _listed(_run(tmp_path))
    assert kind == "failure", (kind, message)


@pytest.mark.parametrize("text, expected", [
    (json.dumps({"kind": "meep-evidence"}), "has kind 'meep-evidence'"),
    ("{not json", "is not readable JSON"),
], ids=["wrong_kind", "unparsable"])
def test_a_malformed_marker_skips_the_listed_test_with_its_reason(tmp_path, text,
                                                                  expected):
    _results(tmp_path, marker=text, record=True)
    _declared_skip(_listed(_run(tmp_path)), expected)


def test_the_marker_is_load_bearing(tmp_path):
    """Put the directory-only rule back and the gate-output tree runs the listed test."""
    source = ROOT_CONFTEST.read_text(encoding="utf-8")
    start = source.index("def _probe_evidence_archive():\n")
    end = source.index("\n\n\ndef ", start)
    mutant = source[:start] + _DIRECTORY_ONLY_PROBE + source[end + 1:]
    assert mutant != source and "_ARCHIVE.is_dir()" in mutant, \
        "DISARMED: the probe this control replaces no longer matches"
    _results(tmp_path)
    kind, message = _listed(_run(tmp_path, mutant))
    assert kind == "failure", (
        f"the directory-only probe skipped the listed test ({kind}: {message}); the "
        f"control no longer shows the defect the marker fixes")
