"""Session-integrity enforcement for the test suites of this repository.

Policy (no silent skips; a suite that does not fully finish must FAIL):

  1. **Every skip is a failure** unless the test is explicitly sanctioned as
     depending on a genuinely absent external resource: an external tool or
     library (MEEP, CuPy, Triton, PyTorch, a CUDA device, an MPS device, ...) or
     a generated or archived data artifact (a gate artifact of the evidence
     archive, a fusion board, a compiler cache, ...).
     A test is sanctioned by either:
       * the marker ``@pytest.mark.requires_resource("name")`` (works at
         function, class, or module ``pytestmark`` scope), or
       * calling ``requires_resource_skip("name", ...)`` (the only sanctioned
         way to do an ``allow_module_level`` skip: it stamps the sentinel the
         collection hook recognises).
     Any other skip (a bare ``pytest.skip(...)``, a ``skipif`` without the
     marker, a swallowed-exception "[SKIP]") is converted to a FAILURE so it
     cannot masquerade as a pass.

  2. **A ``test_*.py`` that collects zero tests fails.** A script wearing a
     ``test_`` name, or a module whose imports silently produced no tests,
     looks covered while validating nothing; that is a collection failure.

  3. ``xfail`` is left alone: it is a *visible, tracked* expected-failure
     mechanism (reported as ``xfailed``/``xpassed``), not a silent skip.

Sanctioned skips still run and are reported as skips. The point is that the
*reason* is declared, not that coverage silently disappears. The summary lists
how many sanctioned-resource skips fired and for which resources.

Two lists under ``tools/ci/`` feed this file:

  4. ``declared_resources.txt`` names tests that need a resource this repository
     or this host may not have (the evidence archive of the certification
     campaigns, PyTorch, MEEP, a single-precision MEEP, a certified Metal
     toolchain). A listed test whose resource is absent is given the marker and
     a skip that states why; a listed test whose resource is present runs as
     before. The evidence archive is present only when
     ``parity/meep_gpu/results/`` carries the marker file
     ``EVIDENCE_ARCHIVE.json`` (:func:`evidence_archive_absence`): the harness
     writes its own records into that directory, so the directory alone is a
     working output directory, not the archive.

  5. **The certification suite is selected, not run by default.** The ledger,
     weld and record contracts (the files in ``CERTIFICATION_FILES``) and the
     tests listed in ``pending_certification.txt`` carry the marker
     ``certification``. A run with no ``-m`` expression deselects them and says
     how many; ``-m certification`` runs them, and naming one of their files on
     the command line runs that file. Their failures wait on the certification
     round on the published files (``docs/development/certification.md``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Skip-enforcement is ON by default (CI, normal runs). Set
# ``SUITE_ENFORCE_SKIPS=0`` for a local *exploratory* run where you want to see
# raw skip behaviour WITHOUT every skip becoming a failure. This is an explicit
# opt-out, not a silent one, and the zero-collection structural guard stays
# active regardless, so a script wearing a test name still fails. The variable
# deliberately does not carry the ``MEEP_GPU_`` prefix: the package's own
# conftest saves and restores every variable with that prefix.
_ENFORCE_SKIPS = os.environ.get("SUITE_ENFORCE_SKIPS", "1").strip().lower() not in (
    "0", "false", "off", "no",
)

_MARKER = "requires_resource"
# Stamped into the skip reason by ``requires_resource_skip`` so the collection
# hook can recognise sanctioned module-level skips (which have no test item to
# carry a marker).
_SENTINEL = "[requires_resource]"

# Tally of sanctioned skips for the terminal summary: resource-name -> count.
_sanctioned_skips: dict[str, int] = {}


def requires_resource_skip(name: str, reason: str = "", *, allow_module_level: bool = False) -> None:
    """Sanctioned skip for a genuinely-absent external tool / data artifact.

    The ONLY blessed way to skip at module level. Stamps the sentinel + the
    resource name into the reason so the enforcement hooks allow it and the
    summary can tally it. ``reason`` is the human-readable detail (kept visible).
    """
    detail = f" {reason}" if reason else " resource unavailable"
    pytest.skip(f"{_SENTINEL}[{name}]{detail}", allow_module_level=allow_module_level)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{_MARKER}(name): the ONLY sanctioned reason a test may skip — the named "
        f"external tool or generated/fetched data artifact is unavailable. Every "
        f"other skip is treated as a silent coverage gap and fails the session.",
    )
    config.addinivalue_line(
        "markers",
        f"{_CERTIFICATION}: part of the certification suite (ledger, weld and record "
        f"contracts, and the tests that wait on the certification round). Not run "
        f"without an -m expression; run with -m {_CERTIFICATION}.",
    )


# ---------------------------------------------------------------------------------
# Declared resources (tools/ci/declared_resources.txt) and the certification suite
# ---------------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
_DECLARED = _ROOT / "tools" / "ci" / "declared_resources.txt"
_PENDING = _ROOT / "tools" / "ci" / "pending_certification.txt"
_ARCHIVE = _ROOT / "parity" / "meep_gpu" / "results"
_CERTIFICATION = "certification"

#: The file that makes ``parity/meep_gpu/results/`` the evidence archive rather than
#: a working output directory. It sits at the root of the archive and is written by
#: whoever puts the archive there; no gate, probe or benchmark writes it.
EVIDENCE_ARCHIVE_MARKER = "EVIDENCE_ARCHIVE.json"
#: The marker's ``kind``, its one required key.
EVIDENCE_ARCHIVE_KIND = "meep-gpu-evidence-archive"
#: Every key the marker may carry. ``archive_version`` and ``manifest_sha256`` are
#: optional and identify which archive was restored; neither is checked against
#: the records.
_EVIDENCE_ARCHIVE_KEYS = ("kind", "archive_version", "manifest_sha256")

#: The ledger, weld and record contracts: every test in these files compares a
#: certification record with the files of this tree or with the campaign that
#: minted it.
CERTIFICATION_FILES = (
    "meep_gpu/cuda_kernels/test_certification_metadata.py",
    "meep_gpu/cuda_kernels/test_certification_record.py",
    "meep_gpu/test_cuda_weld_contract.py",
    "meep_gpu/test_dispatch_contract.py",
    "meep_gpu/test_metal_weld_contract.py",
    "meep_gpu/test_regate_writers.py",
    "meep_gpu/test_triton_weld_contract.py",
)

_deselected_certification: list = []
_declared_unmatched: list = []


def junit_identifier(item: pytest.Item) -> str:
    """``CLASSNAME::NAME`` as the JUnit report and tools/ci/check_failures.py spell it."""
    address, bracket, parameters = item.nodeid.partition("[")
    names = address.split("::")
    module = names[0]
    if module.endswith(".py"):
        module = module[:-3]
    names[0] = module.replace("/", ".")
    names[-1] += bracket + parameters
    return ".".join(names[:-1]) + "::" + names[-1]


def _read_declared() -> dict:
    """Resource name -> set of test identifiers, from declared_resources.txt."""
    sections: dict = {}
    if not _DECLARED.is_file():
        return sections
    current = None
    for number, raw in enumerate(_DECLARED.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            if current not in _PROBES:
                raise pytest.UsageError(
                    f"{_DECLARED.name} line {number}: no probe for resource {current!r}")
            sections.setdefault(current, set())
            continue
        if current is None:
            raise pytest.UsageError(f"{_DECLARED.name} line {number}: a test before any "
                                    f"[resource] heading")
        sections[current].add(line)
    return sections


def _read_pending() -> set:
    if not _PENDING.is_file():
        return set()
    return {line.strip() for line in _PENDING.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def evidence_archive_absence(results: Path) -> str | None:
    """Why ``results`` is not the evidence archive, or ``None`` when it is.

    The archive counts as present only when the directory carries the marker
    ``EVIDENCE_ARCHIVE.json``: a JSON object whose ``kind`` is
    ``"meep-gpu-evidence-archive"``, with at most two more keys,
    ``archive_version`` (a non-empty string) and ``manifest_sha256`` (64 lowercase
    hexadecimal digits). The directory alone proves nothing, because the harness
    writes its own records there: after one gate run in a checkout the directory
    exists and holds none of the archived records. A marker that is present but
    malformed is refused with the reason, so a restore that went wrong is named
    rather than taken for the archive.
    """
    marker = f"parity/meep_gpu/results/{EVIDENCE_ARCHIVE_MARKER}"
    if not results.is_dir():
        return ("parity/meep_gpu/results/ does not exist: the evidence archive of the "
                "certification campaigns is not part of this repository")
    path = results / EVIDENCE_ARCHIVE_MARKER
    if not path.exists():
        return (f"parity/meep_gpu/results/ carries no {EVIDENCE_ARCHIVE_MARKER}, so it "
                f"is a working output directory (gate and benchmark runs write there), "
                f"not the evidence archive")
    if not path.is_file():
        return f"{marker} is not a file"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as problem:
        return f"{marker} is not readable JSON ({problem})"
    if not isinstance(document, dict):
        return f"{marker} is not a JSON object (it holds a {type(document).__name__})"
    kind = document.get("kind")
    if kind != EVIDENCE_ARCHIVE_KIND:
        return f"{marker} has kind {kind!r}, not {EVIDENCE_ARCHIVE_KIND!r}"
    unknown = sorted(set(document) - set(_EVIDENCE_ARCHIVE_KEYS))
    if unknown:
        return (f"{marker} carries keys its schema does not define: "
                f"{', '.join(unknown)} (defined: {', '.join(_EVIDENCE_ARCHIVE_KEYS)})")
    if "archive_version" in document:
        version = document["archive_version"]
        if not isinstance(version, str) or not version.strip():
            return f"{marker} has archive_version {version!r}, not a non-empty string"
    if "manifest_sha256" in document:
        digest = document["manifest_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            return (f"{marker} has manifest_sha256 {digest!r}, not 64 lowercase "
                    f"hexadecimal digits")
    return None


def _probe_evidence_archive():
    why = evidence_archive_absence(_ARCHIVE)
    if why is None:
        return None
    return (f"{why}; this test reads a record of the archive "
            f"(docs/development/testing.md)")


def _probe_torch():
    return None if importlib.util.find_spec("torch") is not None else "PyTorch is not installed"


def _probe_meep():
    return None if importlib.util.find_spec("meep") is not None else "MEEP is not installed"


def _probe_mps_device():
    if importlib.util.find_spec("torch") is None:
        return "PyTorch is not installed, so no Apple GPU is reachable"
    import numpy  # noqa: F401,PLC0415  NumPy before PyTorch (INSTALL.md, OpenMP)
    import torch  # noqa: PLC0415

    backend = getattr(getattr(torch, "backends", None), "mps", None)
    if backend is not None and backend.is_available():
        return None
    return "this PyTorch reports no usable Apple GPU (MPS)"


def _probe_single_precision_meep():
    # Asked in a child process: this process may import PyTorch, and MEEP is not
    # imported beside it here.
    if importlib.util.find_spec("meep") is None:
        return "MEEP is not installed"
    try:
        answer = subprocess.run(
            [sys.executable, "-c",
             "import meep; print('SINGLE' if meep.is_single_precision() else 'DOUBLE')"],
            capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError) as problem:
        return f"the precision of this MEEP could not be read ({problem!r})"
    words = answer.stdout.split()
    if "SINGLE" in words:
        return None
    if "DOUBLE" in words:
        return ("this MEEP is a double-precision build; the test compares with MEEP at "
                "single-precision bounds or asserts that MEEP returns float32 arrays "
                "(INSTALL.md, \"Precision\")")
    return (f"the precision of this MEEP could not be read (exit {answer.returncode}: "
            f"{answer.stderr.strip()[-200:]})")


_PROBES = {
    "evidence_archive": _probe_evidence_archive,
    "torch": _probe_torch,
    "meep": _probe_meep,
    "mps_device": _probe_mps_device,
    "single_precision_meep": _probe_single_precision_meep,
}


def _explicitly_named(config: pytest.Config) -> set:
    named = set()
    for argument in config.args:
        path = Path(str(argument).split("::", 1)[0])
        if not path.is_absolute():
            path = Path(config.invocation_params.dir) / path
        if path.suffix == ".py":
            named.add(path.resolve())
    return named


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config,
                                  items: list) -> None:
    declared = _read_declared()
    pending = _read_pending()
    answers: dict = {}
    matched: set = set()
    for item in items:
        identifier = junit_identifier(item)
        base = identifier.split("[", 1)[0]
        for resource, listed in declared.items():
            entry = identifier if identifier in listed else (base if base in listed else None)
            if entry is None:
                continue
            matched.add((resource, entry))
            if resource not in answers:
                answers[resource] = _PROBES[resource]()
            why = answers[resource]
            if why is not None:
                item.add_marker(pytest.mark.requires_resource(resource))
                item.add_marker(pytest.mark.skip(reason=f"{_SENTINEL}[{resource}] {why}"))
        relative = Path(str(item.path)).resolve()
        try:
            relative_text = relative.relative_to(_ROOT).as_posix()
        except ValueError:
            relative_text = ""
        if relative_text in CERTIFICATION_FILES or identifier in pending \
                or base in pending:
            item.add_marker(getattr(pytest.mark, _CERTIFICATION))

    collected_modules = {junit_identifier(item).split("::", 1)[0] for item in items}
    for resource, listed in declared.items():
        for entry in sorted(listed):
            if entry.split("::", 1)[0] in collected_modules and (resource, entry) not in matched:
                _declared_unmatched.append(f"[{resource}] {entry}")

    if config.option.markexpr:
        return
    named = _explicitly_named(config)
    kept, dropped = [], []
    for item in items:
        if item.get_closest_marker(_CERTIFICATION) is not None \
                and Path(str(item.path)).resolve() not in named:
            dropped.append(item)
        else:
            kept.append(item)
    if dropped:
        _deselected_certification.extend(junit_identifier(item) for item in dropped)
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


def _resource_name_from_marker(item: pytest.Item) -> str | None:
    marker = item.get_closest_marker(_MARKER)
    if marker is None:
        return None
    return str(marker.args[0]) if marker.args else "unspecified"


def _resource_name_from_reason(reason: str) -> str | None:
    if _SENTINEL not in reason:
        return None
    # Reason looks like "[requires_resource][meep] ..." — pull the second [...].
    tail = reason.split(_SENTINEL, 1)[1]
    if tail.startswith("[") and "]" in tail:
        return tail[1:tail.index("]")]
    return "unspecified"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if not report.skipped:
        return
    # xfail/xpass surface as their own outcomes and carry ``wasxfail`` — leave them.
    if getattr(report, "wasxfail", None) is not None:
        return

    reason = ""
    if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        reason = str(report.longrepr[2])
    else:
        reason = str(report.longrepr)

    # The reason first: it names the resource that actually caused this skip, where a
    # test's marker names the resource it needs to run at all (a test that needs MEEP
    # can still skip for want of something MEEP's build lacks).
    sanctioned = _resource_name_from_reason(reason) or _resource_name_from_marker(item)
    if sanctioned is not None:
        _sanctioned_skips[sanctioned] = _sanctioned_skips.get(sanctioned, 0) + 1
        return

    if not _ENFORCE_SKIPS:
        return  # explicit opt-out: leave the skip as a skip (exploratory run)

    # Unsanctioned skip → fail loudly.
    report.outcome = "failed"
    report.longrepr = (
        f"UNSANCTIONED SKIP — no @pytest.mark.{_MARKER}(...) and no sanctioned "
        f"sentinel in the reason. Under the no-silent-skip policy a skip must "
        f"declare a genuinely-absent external resource (tool or generated/fetched "
        f"data) or it is treated as a silent coverage gap and fails.\n"
        f"Original skip reason: {reason}"
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """Fail collection of a test_*.py that (a) skips at module level without a
    sanctioned resource, or (b) collects zero tests. Uses ``force_result`` so
    the mutated report is actually honoured (a plain ``pytest_collectreport``
    mutation is informational and ignored)."""
    outcome = yield
    report = outcome.get_result()

    # Police ONLY ``test_*.py`` Module collectors — never Class/Function/Package
    # collectors. (A ``Test*``-named imported class with no test methods is a Class
    # collector that legitimately yields 0 items; policing it would be a false
    # positive.)
    if not isinstance(collector, pytest.Module):
        return
    nodeid = getattr(collector, "nodeid", "") or ""
    fspath = getattr(collector, "path", None)
    fname = (fspath.name if fspath is not None else nodeid.rsplit("/", 1)[-1])
    if not (fname.startswith("test_") and fname.endswith(".py")):
        return

    # (a) Module-level skip: allow only if it carries the sanctioned sentinel.
    if report.skipped:
        name = _resource_name_from_reason(str(report.longrepr))
        if name is not None:
            _sanctioned_skips[name] = _sanctioned_skips.get(name, 0) + 1
            return
        if not _ENFORCE_SKIPS:
            return  # explicit opt-out: leave the module-level skip as a skip
        report.longrepr = (
            f"UNSANCTIONED MODULE-LEVEL SKIP in {nodeid}: use "
            f"requires_resource_skip('name', allow_module_level=True) so the skip "
            f"declares its absent external resource. Original: {report.longrepr}"
        )
        report.outcome = "failed"
        outcome.force_result(report)
        return

    # (b) Zero-collection guard.
    if report.outcome == "passed" and not report.result:
        report.longrepr = (
            f"ZERO-COLLECTION: {nodeid} matches test_*.py but collected no tests. "
            f"A script masquerading as a test module (or a collection that silently "
            f"imported nothing) validates nothing while looking covered. Give it "
            f"real tests, or rename it so it is not collected as a test module."
        )
        report.outcome = "failed"
        outcome.force_result(report)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if _deselected_certification:
        terminalreporter.write_sep("-", "certification suite (not run by default)")
        terminalreporter.write_line(
            f"  {len(_deselected_certification)} tests of the certification suite were "
            f"deselected; run them with: python -m pytest meep_gpu parity/meep_gpu "
            f"-m {_CERTIFICATION}")
    if _declared_unmatched:
        terminalreporter.write_sep("-", "declared_resources.txt entries that match no "
                                        "collected test")
        for entry in _declared_unmatched:
            terminalreporter.write_line(f"  {entry}")
    if not _sanctioned_skips:
        return
    terminalreporter.write_sep("-", "sanctioned resource skips (declared, not silent)")
    for name in sorted(_sanctioned_skips):
        terminalreporter.write_line(f"  requires_resource({name!r}): {_sanctioned_skips[name]} skipped")
