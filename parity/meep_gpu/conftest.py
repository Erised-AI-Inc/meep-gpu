"""The dispatch enable, pinned OFF for every test under ``parity/meep_gpu/``.

``MEEP_GPU_DISPATCH`` unset takes ``fastpath.DISPATCH_BY_DEFAULT``, and since
2026-09-27 that default dispatches. The tests here drive harness code — the route
gates' helpers, the bench, the census and reachability tools — and several of them
lift or step a driver as the ARRAY-PATH oracle. Left unset, an oracle that PLANS — a
driver lifted ``prefer_gpu=True``, or any direct ``plan_fast_path`` call — would plan
through whatever kernel table the host offers (on a laptop with an MPS device, the
Metal table: shader compiles and a process-wide subnormal policy), and a byte
comparison against it would be kernels against kernels. A driver lifted
``prefer_gpu=False`` is the NumPy reference: it never consults a kernel table,
whatever the enable says, so the pin is redundant for it and stays for the rest.

So every test starts with ``MEEP_GPU_DISPATCH=0``, at two scopes:

* module scope, so a module- or class-scoped fixture that steps a reference driver
  is covered — those are set up before any function-scoped fixture runs;
* function scope, so a test that writes ``os.environ`` directly and does not
  restore it cannot decide what the next test measures.

A test that exercises dispatch sets the enable itself (``monkeypatch.setenv``, in
the test or in a module-level autouse fixture — both run after these and win). A
test that means "unset" deletes it explicitly and gets the shipped default.

The same pin lives in ``meep_gpu/conftest.py`` for the package's own suite; the two
directories are separate conftest scopes, so each carries its own.
"""

from __future__ import annotations

import os
import sys

import pytest

#: ``fastpath.DISPATCH_ENABLE``, spelled here so this conftest imports nothing from
#: the package at collection time; :func:`_dispatch_enable_name` checks it against
#: the package whenever the package is already loaded.
_DISPATCH_ENABLE = "MEEP_GPU_DISPATCH"
_DISPATCH_OFF = "0"


def _dispatch_enable_name() -> str:
    """The enable's name, checked against ``fastpath`` when it is already imported.

    Read off ``sys.modules`` rather than imported, so the check never changes what
    a test module has loaded. A mismatch fails loudly instead of pinning a variable
    the ladder no longer reads.
    """
    fastpath = sys.modules.get("meep_gpu.fastpath")
    spelled = getattr(fastpath, "DISPATCH_ENABLE", _DISPATCH_ENABLE)
    if spelled != _DISPATCH_ENABLE:
        pytest.fail(
            f"DISPATCH PIN IS STALE: fastpath.DISPATCH_ENABLE is {spelled!r} and "
            f"parity/meep_gpu/conftest.py pins {_DISPATCH_ENABLE!r}. Rename the pin "
            "(and meep_gpu/conftest.py's) with the variable.")
    return _DISPATCH_ENABLE


def _pinned(name: str):
    previous = os.environ.get(name)
    os.environ[name] = _DISPATCH_OFF
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


@pytest.fixture(autouse=True, scope="module")
def _dispatch_pinned_off_for_the_module():
    """Pin the enable OFF for the module's lifetime; put the caller's value back after."""
    yield from _pinned(_dispatch_enable_name())


@pytest.fixture(autouse=True)
def _dispatch_pinned_off():
    """Every test starts with dispatch OFF; a test that wants it sets it.

    Saved and restored by hand rather than through ``monkeypatch``: a fixture that
    requests ``monkeypatch`` pulls its setup ahead of the test's own and its
    teardown behind every other autouse fixture, which reorders what those
    fixtures see (``meep_gpu/conftest.py`` measured that against its stub guard).
    """
    yield from _pinned(_dispatch_enable_name())
