"""The isolation guard in ``meep_gpu/conftest.py``, checked for being ARMED.

A guard that silently stops matching is worse than no guard: the suite reads as
protected while the coupling it was written for is free to come back. The
coupling in question is real and was measured (2026-08-13) — seven tests in
``test_triton_cylindrical_complex.py`` and ``test_triton_conductivity.py`` passed
or failed according to which files were collected beside them, because one test
module wrote NumPy into ``sys.modules["cupy"]`` and never took it out. See
``conftest.py``'s module docstring for the mechanism.

So the guard is exercised end-to-end, in a separate interpreter, against the real
``conftest.py`` source: a leaking test must FAIL, a properly scoped stub must NOT,
and the leak must be torn out so one offender does not cascade into every file
after it. Running it out-of-process is the point — a leak asserted in THIS process
would be the very pollution the guard exists to prevent.

The containment leg carries its own negative control: the same generated files
are re-run against a conftest with the teardown removal cut out, and the innocent
test must then FAIL. Without that, "the next test still saw a clean sys.modules"
would pass just as happily if nothing were being contained at all.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

CONFTEST = pathlib.Path(__file__).with_name("conftest.py")

#: The one line whose removal turns containment off, and nothing else.
_CONTAINMENT_LINE = "        del sys.modules[name]\n"

_LEAKING = '''
import sys
import numpy as np

def test_leaks_a_stub():
    sys.modules.setdefault("cupy", np)
    assert True

def test_runs_after_the_leak():
    # The guard tore the stub out, so this innocent test is unpolluted.
    assert "cupy" not in sys.modules, "the leak cascaded into the next test"
'''

_NAMED_STUB = '''
import sys
from types import ModuleType

def test_leaks_a_stub_that_calls_itself_cupy():
    # Passes the __name__ tell; still cannot answer cp.asnumpy / cp.ndarray, so
    # it poisons every gate imported after it exactly as an alias would.
    sys.modules.setdefault("cupy", ModuleType("cupy"))
    assert True
'''

_SCOPED = '''
import sys
import numpy as np

def test_scoped_stub_is_fine(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", np)
    assert sys.modules["cupy"] is np
'''


def _run(tmp_path: pathlib.Path, body: str,
         conftest_source: str | None = None) -> subprocess.CompletedProcess:
    """Run one generated test file under the REAL conftest, in its own process."""
    if conftest_source is None:
        shutil.copy(CONFTEST, tmp_path / "conftest.py")
    else:
        (tmp_path / "conftest.py").write_text(conftest_source, encoding="utf-8")
    (tmp_path / "test_generated.py").write_text(body, encoding="utf-8")
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, env=env, cwd=str(tmp_path))


def _summary(result: subprocess.CompletedProcess) -> str:
    """Just pytest's short test summary — the nodes it reports as not-passing."""
    return result.stdout.split("short test summary info")[-1]


def test_the_guard_fails_the_test_that_leaks_a_backend_stub(tmp_path):
    """ARMED: an unscoped ``sys.modules["cupy"] = numpy`` must not pass."""
    result = _run(tmp_path, _LEAKING)
    assert result.returncode != 0, result.stdout
    assert "BACKEND STUB LEAKED" in result.stdout, result.stdout
    # Named, so the failure lands on the offender rather than on its victims.
    assert "test_leaks_a_stub" in _summary(result), result.stdout


def test_the_guard_contains_the_leak_instead_of_cascading(tmp_path):
    """The test AFTER the leak still sees a clean ``sys.modules``.

    Without the teardown removal one leak poisons every module imported for the
    rest of the session — which is exactly how seven unrelated tests came to
    depend on collection order. The offender is reported; its neighbour is not.
    """
    result = _run(tmp_path, _LEAKING)
    summary = _summary(result)
    assert "test_leaks_a_stub" in summary, result.stdout
    assert "test_runs_after_the_leak" not in summary, result.stdout


def test_the_containment_is_load_bearing_and_not_incidental(tmp_path):
    """The negative control for the test above: cut the removal, and it cascades.

    One line out of ``conftest.py`` — the ``del sys.modules[name]`` — and the
    innocent neighbour must start failing. If it did not, the containment test
    would be passing for some reason other than the guard.
    """
    source = CONFTEST.read_text(encoding="utf-8")
    assert source.count(_CONTAINMENT_LINE) == 1, \
        "DISARMED: the containment line this control removes no longer matches"
    result = _run(tmp_path, _LEAKING, source.replace(_CONTAINMENT_LINE, ""))
    summary = _summary(result)
    assert "test_runs_after_the_leak" in summary, result.stdout
    assert "the leak cascaded into the next test" in result.stdout, result.stdout


def test_the_guard_catches_a_stub_that_wears_the_right_name(tmp_path):
    """The second tell. A bare ``ModuleType("cupy")`` reports ``__name__ ==
    "cupy"``, so a name check alone would wave it through — and it is just as
    poisonous, because what breaks downstream is ``cp is not None`` followed by a
    call the stand-in has no answer for. Catching it needs the attribute check,
    which is why the guard carries both."""
    result = _run(tmp_path, _NAMED_STUB)
    assert result.returncode != 0, result.stdout
    assert "BACKEND STUB LEAKED" in result.stdout, result.stdout
    assert "test_leaks_a_stub_that_calls_itself_cupy" in _summary(result), result.stdout


def test_the_guard_leaves_a_properly_scoped_stub_alone(tmp_path):
    """NOT over-armed: ``monkeypatch.setitem`` is undone at teardown, so a test
    that needs a stub for its own duration keeps working. If this failed, the
    guard would be banning the correct spelling along with the broken one."""
    result = _run(tmp_path, _SCOPED)
    assert result.returncode == 0, result.stdout
    assert "BACKEND STUB LEAKED" not in result.stdout, result.stdout


# ---------------------------------------------------------------------------
# The dispatch pin: MEEP_GPU_DISPATCH=0 at module AND function scope
# ---------------------------------------------------------------------------
#
# Since ``DISPATCH_BY_DEFAULT`` turned on, an unset enable dispatches on every
# driver that plans: a ``prefer_gpu=True`` driver (Metal over NumPy arrays on a host
# with an MPS device), a driver a test has given a ``gpu``, and a direct planner
# call. (A ``prefer_gpu=False`` driver is the NumPy reference and never plans, so the
# pin is redundant for it.) ``conftest.py`` pins the enable OFF twice — inside the module
# fixture's environment, so module- and class-scoped oracle fixtures see it, and
# again per test, so a test that writes ``os.environ`` directly cannot decide what
# the next one measures — and a test that exercises dispatch sets ``1`` itself.
# Deleting either pin failed nothing before these tests: the oracles would simply
# have started measuring kernels against kernels. So each pin is exercised in a
# separate interpreter against the real conftest, under a caller that exported
# nothing AND under one that exported ``1`` (the parent's own pin is removed from
# the child's environment, or a mutant conftest would read ``0`` off it and pass for
# the wrong reason), and each pin's removal has its own negative control.

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
    # The shape test_dispatch_contract uses: a module-level autouse opt-in.
    monkeypatch.setenv("MEEP_GPU_DISPATCH", "1")


def test_a_modules_own_autouse_opt_in_wins():
    assert os.environ.get("MEEP_GPU_DISPATCH") == "1", os.environ.get("MEEP_GPU_DISPATCH")
'''

#: The two pins, each as the exact text whose removal disarms it and nothing else.
#: The module pin is the entry that puts the enable into the module fixture's
#: ``MEEP_GPU_*`` environment; the function pin loses its fixture decorator, so the
#: function is still defined and never runs.
_MODULE_PIN = "    wanted = {**wanted, _dispatch_enable_name(): _DISPATCH_OFF}\n"
_FUNCTION_PIN = "@pytest.fixture(autouse=True)\ndef _dispatch_pinned_off():\n"


def _run_pin(tmp_path: pathlib.Path, caller: str | None,
             conftest_source: str | None = None) -> subprocess.CompletedProcess:
    """Both generated modules under ``conftest_source`` (default: the real file).

    ``caller`` is what the calling shell exported: ``None`` removes the variable
    from the child's environment, which also removes the PARENT's own pin.
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


def test_the_dispatch_pin_reaches_module_fixtures_and_tests_and_yields_to_an_opt_in(
        tmp_path):
    """ARMED: ``0`` in a module-scoped fixture and in every test, whatever the
    caller exported; a leaked direct write is contained; the test's own
    ``setenv("1")`` and a module's own autouse opt-in both win."""
    for caller in (None, "1"):
        result = _run_pin(tmp_path / f"caller_{caller}", caller)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "6 passed" in result.stdout, result.stdout


def test_the_module_pin_is_load_bearing(tmp_path):
    """Cut the module pin and the module-scoped fixture stops reading ``0``."""
    source = CONFTEST.read_text(encoding="utf-8")
    assert source.count(_MODULE_PIN) == 1, \
        "DISARMED: the module pin this control removes no longer matches"
    for caller in (None, "1"):
        result = _run_pin(tmp_path / f"caller_{caller}", caller,
                          source.replace(_MODULE_PIN, ""))
        summary = _summary(result)
        assert "test_a_module_scoped_fixture_sees_the_pin" in summary, result.stdout


def test_the_function_pin_is_load_bearing(tmp_path):
    """Cut the function pin and a test's unrestored write reaches the next test."""
    source = CONFTEST.read_text(encoding="utf-8")
    assert source.count(_FUNCTION_PIN) == 1, \
        "DISARMED: the function pin this control removes no longer matches"
    mutant = source.replace(_FUNCTION_PIN, "def _dispatch_pinned_off():\n")
    result = _run_pin(tmp_path, None, mutant)
    summary = _summary(result)
    assert "test_the_next_test_sees_the_pin_again" in summary, result.stdout
    assert "test_a_module_scoped_fixture_sees_the_pin" not in summary, result.stdout
