"""Cross-file isolation guards for the ``meep_gpu`` suite.

This suite runs on a laptop with neither CuPy nor Triton installed, against
device-gate modules that ask ``cp is None`` to decide whether a device path is
reachable. That makes ``sys.modules["cupy"]`` a piece of PROCESS-GLOBAL STATE
that every gate reads at its own first import — and a gate is imported once and
cached, so whichever test file causes that import decides what ``cp`` means for
every test that runs afterwards.

Two parity scripts used to write ``sys.modules.setdefault("cupy", numpy)`` so a
gate could be imported without a device (the gates have imported ``cupy``
defensively for a while, so it bought nothing). Nothing ever undid it, and
NumPy-wearing-CuPy's-name is worse than no CuPy at all:

* ``cp is not None`` reads as "there is a device", so device-only policy fires on
  a laptop — ``gate_triton_complex.install_ftz_strip`` demanded a private
  ``CUPY_CACHE_DIR`` and raised, replacing the device-absent refusal that
  ``test_triton_cylindrical_complex.py`` is there to pin;
* ``isinstance(array, cp.ndarray)`` answers True for a plain NumPy array, so host
  bridges dispatched to a ``numpy.asnumpy`` that does not exist, breaking all six
  ``test_triton_conductivity.py`` bit-identity cases.

Measured 2026-08-13: seven tests that pass under ``pytest meep_gpu/test_triton_*.py``
failed under a different grouping of the same files, purely on which file imported
a gate first. That is greenness decided by collection order, and it blocks
sharding the suite — the obvious remedy for a serial run on a ~2 h trajectory.

The stubs are gone. This guard is what keeps them gone: a test that leaves a fake
``cupy`` behind FAILS, by name, instead of quietly re-coupling the files.
Scoped stubbing is untouched — ``monkeypatch.setitem(sys.modules, "cupy", ...)``
is undone at teardown and never trips this.

The subnormal policy is the SECOND piece of process-global state with the same
shape, and it leaks by a different route (measured 2026-08-15: ten failures in a
full ``pytest meep_gpu/`` run, nine of them this one cause; every affected module
passes alone).

The trigger is not a test — it is IMPORT. About twenty Metal parity modules open
with ``os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")`` at module
scope (``gate_metal_pml.py:91``, ``gate_metal_complex.py:178``, and their
siblings), and ``metal_composition_matrix.prepare_environment`` (:83) does the
same for anything that calls it, including ``test_metal_planner_composition.py``
at its own module scope (:56). Every one of those is CORRECT for a script run
standalone — the MPS executor cannot honour ``keep``, so a gate that did not ask
for ``flush`` would measure nothing. But pytest imports every test module during
COLLECTION, before the first test of any module runs, so importing one of those
gates sets the variable for the whole process with nothing anywhere to undo it.

The install is the amplifier. ``subnormal_policy`` caches the resolved policy in
its module-level ``_STATE``, and restoring the ENVIRONMENT VARIABLE does not undo
the INSTALL — which is why ``monkeypatch.setenv`` in ``test_metal_bfast.py`` and
``test_metal_complex_fields.py``, correctly scoped and genuinely undone, still
left ``flush`` in force. Every later module that resolves the policy then read
``flush`` where it expected ``match_meep``/``keep``, and the gate-side tests
compared a FLUSH stamp against a KEEP one.

So both halves are put back, at module teardown, by
:func:`_no_leaked_subnormal_policy` below.

The dispatch enable is the THIRD piece of process-global state, and it is pinned
rather than restored. ``MEEP_GPU_DISPATCH`` unset takes
``fastpath.DISPATCH_BY_DEFAULT``, and since 2026-09-27 that default dispatches on
every driver that PLANS: one built with ``prefer_gpu=True`` (CuPy on a CUDA host,
Metal kernels over NumPy arrays on an Apple GPU), one a test has made a GPU driver's
engine by setting ``driver.gpu``, and every direct ``plan_fast_path`` call. A test
that steps such a driver as the ARRAY-PATH ORACLE and leaves the enable unset would
compare kernels against kernels — and on a laptop with an MPS device it would
compile shaders and install a process-wide subnormal policy on the way. A driver
built with ``prefer_gpu=False`` is the NumPy reference and needs no pin: it never
consults a kernel table, whatever the enable says (``test_prefer_gpu.py``), so for
the reference oracles the pin is redundant and it stays for the rest. So every test
in this tree starts with
``MEEP_GPU_DISPATCH=0`` (:func:`_dispatch_pinned_off`, and inside every module's
own policy environment, so module- and class-scoped fixtures see it too). A test
that exercises dispatch says so itself — ``monkeypatch.setenv(DISPATCH_ENABLE,
"1")`` in the test or in a module-level autouse fixture, both of which run after
this pin and win. A test that means "unset" deletes the variable explicitly and
gets the shipped default, which is the only way to ask that question here.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest

#: Modules whose identity a gate reads to decide "is a device reachable here",
#: mapped to attributes the REAL package has and a stand-in would not. Stubbing
#: one of these process-globally makes every later import lie.
_BACKEND_MODULES = {
    # The two the gates actually reach for once ``cp is not None`` lets them.
    "cupy": ("ndarray", "asnumpy"),
    "triton": ("jit",),
}


def _impostor(name: str, attributes: tuple[str, ...]) -> ModuleType | None:
    """The module registered under ``name`` if it is not actually that package.

    Two independent tells, because a stub can defeat either one alone: a real
    ``import cupy`` binds a module whose own ``__name__`` is "cupy" (NumPy
    standing in reports "numpy"), and the real package has the attributes the
    gates call. A hand-rolled ``ModuleType("cupy")`` passes the name check and
    fails the second — and it is just as poisonous, since what breaks downstream
    is ``cp is not None`` followed by a call the stand-in cannot answer.
    """
    module = sys.modules.get(name)
    if module is None:
        return None
    if getattr(module, "__name__", None) != name:
        return module
    if any(not hasattr(module, attribute) for attribute in attributes):
        return module
    return None


@pytest.fixture(autouse=True)
def _no_leaked_backend_stub():
    """Fail the test that leaves NumPy (or anything else) wearing a backend's name.

    Checked after the test rather than before so the failure lands on the test
    that CAUSED it, not on the innocent one that ran next. The stub is also torn
    out, so a single leak fails once instead of cascading through every file
    collected after it — the leak stays loud, its blast radius does not.
    """
    yield
    for name, attributes in _BACKEND_MODULES.items():
        impostor = _impostor(name, attributes)
        if impostor is None:
            continue
        del sys.modules[name]
        pytest.fail(
            f"BACKEND STUB LEAKED: this test left sys.modules[{name!r}] bound to "
            f"{getattr(impostor, '__name__', impostor)!r}, which is process-global "
            f"and never undone. Every gate imported after it binds that stub as "
            f"``cp``, reads ``cp is not None`` as 'a device is present', and calls "
            f"device-only methods ({name}.{attributes[-1]}, the -ftz cache policy) "
            f"that the stand-in does not answer — so unrelated test files start "
            f"passing or failing on collection order. The gates import {name} "
            f"defensively and need no stub; if a test genuinely needs one, scope it "
            f"with monkeypatch.setitem(sys.modules, {name!r}, ...) so it is undone "
            f"at teardown.")


#: Every environment variable this package reads is spelled ``MEEP_GPU_*`` — the
#: policy itself and the expansion-probe paths alike. The prefix is the unit of
#: restoration rather than one hand-listed name, because the failure mode is a
#: module-scope ``setdefault`` in a file nobody remembers importing, and a list
#: would silently stop covering the next one somebody adds.
_POLICY_ENV_PREFIX = "MEEP_GPU_"


def _policy_environment() -> dict[str, str]:
    return {name: value for name, value in os.environ.items()
            if name.startswith(_POLICY_ENV_PREFIX)}


#: Captured at CONFTEST IMPORT, which is the whole trick. pytest imports the
#: conftest for a directory BEFORE it collects the test modules in it, so this
#: runs before the first module-scope ``os.environ.setdefault`` anywhere in the
#: suite. It is therefore the only moment in the process at which ``MEEP_GPU_*``
#: still means what the CALLER exported — read it any later and the "pristine"
#: value is already whatever collection happened to leave behind.
_PRISTINE_POLICY_ENV = _policy_environment()

#: The same variables AFTER collection has imported every test module — i.e.
#: exactly what the Metal modules' own import-time ``setdefault`` calls
#: established, the policy and both expansion-probe paths together. Captured
#: rather than hand-listed so that a Metal module which starts needing a fourth
#: variable gets it without this file being edited, and so the profile is always
#: the modules' own declaration rather than this guard's guess at it.
_COLLECTED_POLICY_ENV: dict[str, str] = {}


def pytest_collection_finish(session):  # noqa: ARG001 - pytest hook signature
    global _COLLECTED_POLICY_ENV
    if not _COLLECTED_POLICY_ENV:
        _COLLECTED_POLICY_ENV = _policy_environment()


#: A test module needs ``flush`` if and only if it exercises the Metal executor.
#: This is a PHYSICAL fact about the executor, not a maintained list of files:
#: MPS flushes subnormals and cannot be made to keep them, so a Metal predicate
#: asked for ``keep`` refuses and the suite measures nothing
#: (``test_metal_planner_composition.py:92``). Nothing outside the Metal modules
#: wants it, and every Metal module wants it, so the module's own name carries
#: the whole declaration and no file has to be edited to opt in.
_METAL_MODULE_PREFIX = "test_metal_"


def _module_needs_flush(module) -> bool:
    return os.path.basename(getattr(module, "__file__", "")).startswith(
        _METAL_MODULE_PREFIX)


#: The enable's name as the package spells it (``fastpath.DISPATCH_ENABLE``), and
#: the one value that turns dispatch off. Spelled here rather than imported: this
#: conftest is imported before every test module and must not pull the package in
#: (see :func:`_uninstall_policy`). :func:`_dispatch_enable_name` checks the
#: spelling against the package whenever the package is already loaded, so a
#: rename cannot leave this pin guarding a variable nothing reads.
_DISPATCH_ENABLE = "MEEP_GPU_DISPATCH"
_DISPATCH_OFF = "0"


def _dispatch_enable_name() -> str:
    """The enable's name, checked against ``fastpath`` when it is already imported.

    Read off ``sys.modules`` rather than imported, so the check never changes what
    a test module has loaded. A mismatch fails loudly instead of pinning a variable
    the ladder no longer reads — which would silently put every array-path oracle
    back on the default.
    """
    fastpath = sys.modules.get("meep_gpu.fastpath")
    spelled = getattr(fastpath, "DISPATCH_ENABLE", _DISPATCH_ENABLE)
    if spelled != _DISPATCH_ENABLE:
        pytest.fail(
            f"DISPATCH PIN IS STALE: fastpath.DISPATCH_ENABLE is {spelled!r} and "
            f"meep_gpu/conftest.py pins {_DISPATCH_ENABLE!r}. Rename the pin "
            "(and parity/meep_gpu/conftest.py's) with the variable.")
    return _DISPATCH_ENABLE


@pytest.fixture(autouse=True)
def _dispatch_pinned_off():
    """Every test starts with dispatch OFF; a test that wants it sets it.

    Function-scoped on top of the module-scoped pin in
    :func:`_no_leaked_subnormal_policy`, because a test that writes
    ``os.environ`` directly and does not restore it would otherwise decide what
    the next test in the same module measures. Autouse fixtures in this conftest
    are set up before a test module's own autouse fixtures of the same scope, so
    ``test_dispatch_contract._opted_in`` and its siblings still win.

    IT DOES NOT REQUEST ``monkeypatch``, and that is load-bearing. pytest orders a
    conftest's autouse fixtures by NAME, so this one is set up before
    :func:`_no_leaked_backend_stub`; asking for ``monkeypatch`` here would set
    monkeypatch up first too and tear it down LAST — after the stub guard had
    already inspected ``sys.modules`` — so a test's correctly scoped
    ``monkeypatch.setitem(sys.modules, "cupy", ...)`` would read as a leak
    (measured: ``test_conftest_guards`` failed exactly that way). The variable is
    saved and put back by hand instead.
    """
    name = _dispatch_enable_name()
    previous = os.environ.get(name)
    os.environ[name] = _DISPATCH_OFF
    yield
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _apply_policy_environment(wanted: dict[str, str]) -> None:
    """Make ``MEEP_GPU_*`` be exactly ``wanted`` — no more, no less."""
    for name in [n for n in os.environ if n.startswith(_POLICY_ENV_PREFIX)]:
        if name not in wanted:
            del os.environ[name]
    for name, value in wanted.items():
        os.environ[name] = value


def _uninstall_policy() -> None:
    # Imported here, not at module scope, for two reasons. This conftest is
    # imported before every test module in the directory, and pulling the policy
    # machinery in that early would make the suite's first import order depend on
    # the guard meant to make import order stop mattering. And the import can
    # legitimately FAIL: ``test_conftest_guards.py`` copies this file into a
    # tmp_path and runs a generated test there in a subprocess, precisely so the
    # guards are exercised as the pytest machinery really loads them, and
    # ``meep_gpu`` is not importable from that directory. A process that cannot
    # import the package cannot be holding an install of its policy, so the
    # no-op is the right answer rather than a swallowed error — and it is
    # narrowed to the package itself so an ImportError from INSIDE
    # ``subnormal_policy`` still propagates.
    try:
        from meep_gpu import subnormal_policy
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "meep_gpu":
            return
        raise

    if subnormal_policy.policy_is_installed():
        subnormal_policy.uninstall_subnormal_policy()


@pytest.fixture(autouse=True, scope="module")
def _no_leaked_subnormal_policy(request):
    """Give each module the policy IT declares, whatever collection left behind.

    Restoring only at teardown is not enough, and was measured not to be: the
    variable is established during COLLECTION, so by the time any module's
    teardown runs, every other module has already inherited it. Cleaning up after
    the first Metal module simply moved the breakage — ``test_metal_kernels.py``
    tore the variable out and ``test_metal_planner_composition.py``, which had
    set it at its own import and had not run yet, lost it and failed 34 tests.

    So the fixture SETS UP as well as tears down, and the module's own name is
    the declaration (:func:`_module_needs_flush`). Collection order stops being
    an input: a Metal module gets ``flush`` whether or not a sibling stripped it,
    and every other module gets the caller's pristine environment whether or not
    a sibling set it.

    MODULE-scoped, not function-scoped, and that is deliberate in both
    directions. Per-test teardown would be wrong: the Metal modules establish
    ``flush`` at their own import and every one of their tests needs it in force,
    so clearing it between tests would break the modules that are behaving
    correctly. Per-module teardown is the tightest boundary that leaves a module
    entirely alone and still refuses to let it decide what the NEXT file measures.

    Both halves are required, and MEASURED 2026-08-15 to be required for
    DIFFERENT reasons — neither is defence-in-depth for the other:

    * a successful install OUTRANKS the environment. After installing ``keep``,
      flipping ``MEEP_GPU_SUBNORMAL_POLICY`` to ``flush`` and then deleting it
      outright both leave ``get_subnormal_policy()`` answering ``keep`` with
      ``policy_is_installed()`` True; only ``uninstall_subnormal_policy`` clears
      it. Restoring the environment cannot reach an install that took.
    * on THIS host no install takes, so the environment is the whole story.
      ``flush`` is unattainable on arm64 without MEEP imported — ``set_zero_subnormals``
      is the only exposure of the FTZ/DAZ bits this package may use, and importing
      MEEP here would breach the no-MEEP-import boundary and initialize MPI as a
      side effect. The Metal modules ask for exactly ``flush``, so they leave the
      variable set and no install behind, and it is the variable that reaches the
      next module. On x86, where ``flush`` DOES install, it is the other half.

    Which is to say: the laptop and the device fail through different halves of
    the same leak, and a guard carrying only one of them would look green on
    whichever machine happened to run it.

    The two mechanisms in detail:

    * the environment goes back to :data:`_PRISTINE_POLICY_ENV` — keys the module
      introduced are removed, keys it overwrote are put back to the captured
      value, rather than merely unsetting the one name we happen to know about;
    * the INSTALL is torn out, because ``subnormal_policy`` caches the resolved
      policy in ``_STATE`` and no amount of environment restoration reaches it.
      ``uninstall_subnormal_policy`` is used rather than clearing ``_STATE``
      directly for the reason its own docstring gives — the Triton wrappers live
      on the ``CUDABackend`` CLASS, so forgetting the bookkeeping alone leaves the
      previous policy's ``add_stages`` and ``hash`` in place and the next install
      returns a stamp naming a policy the bytes will not obey.

    The host FPU is deliberately left where it is, matching
    ``uninstall_subnormal_policy``'s own contract: the caller that wants a
    particular FPU state installs a policy that says so.
    """
    # Re-established rather than assumed: a Metal module set these at its own
    # import, but a sibling's teardown may have run in between.
    wanted = (_COLLECTED_POLICY_ENV or _policy_environment()) \
        if _module_needs_flush(request.module) else _PRISTINE_POLICY_ENV
    # THE DISPATCH PIN RIDES IN THE MODULE'S ENVIRONMENT, not beside it: the
    # variable carries the ``MEEP_GPU_`` prefix, so an environment applied without
    # it would delete it, and a module- or class-scoped fixture that steps an
    # array-path oracle is set up here, before any function-scoped pin could run.
    # Teardown restores the caller's environment exactly, pin included or not.
    wanted = {**wanted, _dispatch_enable_name(): _DISPATCH_OFF}
    _apply_policy_environment(wanted)
    _uninstall_policy()

    yield

    _apply_policy_environment(_PRISTINE_POLICY_ENV)
    _uninstall_policy()


# ---------------------------------------------------------------------------
# The subnormal policy a predicate question is asked UNDER
# ---------------------------------------------------------------------------
#
# A complex family's coverage predicate cannot answer "may this kernel step
# this?" without knowing which float32 subnormal policy the run will use: the
# EXPANSION licence its probe artifact carries is conditional on the policy the
# artifact was cut under (``complex_fields.POLICY_CONDITIONAL_LICENCE``), so the
# same record licenses an arm under one policy and refuses under the other.
#
# That clause FAILS CLOSED as of 2026-08-16: asked when the policy can be
# neither read nor declared, it refuses. Production never lands there — dispatch
# declares the policy it is about to install, four rungs before it installs it —
# but a test process installs nothing, so a module that builds keep-stamped probe
# records and asks a predicate about them has to say so. This fixture is how it
# says so, and modules opt in by name::
#
#     pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")
#
# It states a premise those modules always had and could not previously write
# down. It is deliberately NOT autouse: the tests that pin the fail-closed
# behaviour itself must be able to ask the question with nothing declared.


@pytest.fixture
def run_policy_declared_keep():
    """Declare 'keep' as the policy this test's predicate questions are about."""
    from meep_gpu.expansion_refusal import declaring_run_policy

    with declaring_run_policy("keep"):
        yield "keep"
