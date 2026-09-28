"""The dispatch pin in ``parity/meep_gpu/conftest.py``, checked for being ARMED.

Since ``DISPATCH_BY_DEFAULT`` turned on, an unset ``MEEP_GPU_DISPATCH`` dispatches on
every driver that plans. Several tests here lift or step a driver as the ARRAY-PATH
oracle; one lifted ``prefer_gpu=True`` (on a host with an MPS device, an Apple GPU
driver) or handed to the planner directly would plan through the host's kernel table
unpinned, so a byte comparison against it would be kernels against kernels. A
``prefer_gpu=False`` lift is the NumPy reference and never plans, whatever the enable
says, so the pin is redundant for it and stays for the rest. The conftest pins the enable
OFF at module scope (for module- and class-scoped reference fixtures) and again per
test (so a test that writes ``os.environ`` directly cannot decide what the next one
measures), and a test that exercises dispatch sets ``1`` itself.

Deleting either pin used to fail nothing, so each is exercised here in a separate
interpreter against the real conftest source — under a caller that exported nothing
and under one that exported ``1``, with this process's own pin removed from the
child's environment so a disarmed conftest cannot pass by inheriting it — and each
pin's removal carries a negative control that must fail by name. The package's own
suite has the same guard for ``meep_gpu/conftest.py`` in
``meep_gpu/test_conftest_guards.py``; the two directories are separate conftest
scopes and each pin is checked where it lives.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

CONFTEST = pathlib.Path(__file__).with_name("conftest.py")
_DISPATCH = "MEEP_GPU_DISPATCH"

_PIN_READERS = '''
import os
import pytest


@pytest.fixture(scope="module")
def module_reference():
    # Stands in for a module-scoped array-path oracle: what the enable read when
    # the fixture was set up, which is before any function-scoped fixture runs.
    return os.environ.get("MEEP_GPU_DISPATCH")


def test_a_module_scoped_fixture_sees_the_pin(module_reference):
    assert module_reference == "0", f"module-scoped fixture read {module_reference!r}"


def test_a_test_sees_the_pin():
    assert os.environ.get("MEEP_GPU_DISPATCH") == "0", os.environ.get("MEEP_GPU_DISPATCH")


def test_a_tests_own_setenv_wins(monkeypatch):
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "1")
    assert os.environ["MEEP_GPU_DISPATCH"] == "1"


def test_writes_the_enable_and_leaves_it():
    os.environ["MEEP_GPU_DISPATCH"] = "1"  # deliberately not restored


def test_the_next_test_sees_the_pin_again():
    assert os.environ.get("MEEP_GPU_DISPATCH") == "0", (
        f"the previous test's write leaked: {os.environ.get('MEEP_GPU_DISPATCH')!r}")
'''

_PIN_OPT_IN = '''
import os
import pytest


@pytest.fixture(autouse=True)
def _opted_in(monkeypatch):
    # A module-level autouse opt-in, the shape a dispatch-exercising module uses.
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "1")


def test_a_modules_own_autouse_opt_in_wins():
    assert os.environ.get("MEEP_GPU_DISPATCH") == "1", os.environ.get("MEEP_GPU_DISPATCH")
'''

#: Each pin as the exact text whose removal disarms it: the fixture decorator goes,
#: so the function is still defined and never runs.
_MODULE_PIN = ('@pytest.fixture(autouse=True, scope="module")\n'
               "def _dispatch_pinned_off_for_the_module():\n")
_FUNCTION_PIN = "@pytest.fixture(autouse=True)\ndef _dispatch_pinned_off():\n"


def _run(tmp_path: pathlib.Path, caller: str | None,
         conftest_source: str | None = None) -> subprocess.CompletedProcess:
    """Both generated modules under ``conftest_source`` (default: the real file).

    ``caller`` is what the calling shell exported: ``None`` removes the variable
    from the child's environment, which also removes this process's own pin.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    if conftest_source is None:
        shutil.copy(CONFTEST, tmp_path / "conftest.py")
    else:
        (tmp_path / "conftest.py").write_text(conftest_source, encoding="utf-8")
    (tmp_path / "test_pin_readers.py").write_text(_PIN_READERS, encoding="utf-8")
    (tmp_path / "test_pin_opt_in.py").write_text(_PIN_OPT_IN, encoding="utf-8")
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    env.pop(_DISPATCH, None)
    if caller is not None:
        env[_DISPATCH] = caller
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path))


def _summary(result: subprocess.CompletedProcess) -> str:
    """Just pytest's short test summary — the nodes it reports as not-passing."""
    return result.stdout.split("short test summary info")[-1]


def test_the_dispatch_pin_reaches_module_fixtures_and_tests_and_yields_to_an_opt_in(
        tmp_path):
    """ARMED: ``0`` in a module-scoped fixture and in every test, whatever the
    caller exported; a leaked direct write is contained; the test's own
    ``setenv("1")`` and a module's own autouse opt-in both win."""
    for caller in (None, "1"):
        result = _run(tmp_path / f"caller_{caller}", caller)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "6 passed" in result.stdout, result.stdout


def test_the_module_pin_is_load_bearing(tmp_path):
    """Cut the module pin and the module-scoped fixture stops reading ``0``."""
    source = CONFTEST.read_text(encoding="utf-8")
    assert source.count(_MODULE_PIN) == 1, \
        "DISARMED: the module pin this control removes no longer matches"
    mutant = source.replace(_MODULE_PIN, "def _dispatch_pinned_off_for_the_module():\n")
    for caller in (None, "1"):
        result = _run(tmp_path / f"caller_{caller}", caller, mutant)
        assert "test_a_module_scoped_fixture_sees_the_pin" in _summary(result), \
            result.stdout


def test_the_function_pin_is_load_bearing(tmp_path):
    """Cut the function pin and a test's unrestored write reaches the next test."""
    source = CONFTEST.read_text(encoding="utf-8")
    assert source.count(_FUNCTION_PIN) == 1, \
        "DISARMED: the function pin this control removes no longer matches"
    mutant = source.replace(_FUNCTION_PIN, "def _dispatch_pinned_off():\n")
    result = _run(tmp_path, None, mutant)
    summary = _summary(result)
    assert "test_the_next_test_sees_the_pin_again" in summary, result.stdout
    assert "test_a_module_scoped_fixture_sees_the_pin" not in summary, result.stdout
