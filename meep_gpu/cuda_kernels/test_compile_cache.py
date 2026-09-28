"""The compile memo's key, exercised where there is no device.

WHY THIS FILE EXISTS. Until 2026-08-15 ``step_curl_kernels`` memoized compiled
kernels on ``(name, is_complex)`` and nothing else. A float32 subnormal policy
installed after a kernel had first compiled was therefore never seen by that
module: the second caller got the first caller's binary, under the second
caller's policy name, with every counter reading zero. The 120/120 the track
holds names no policy at all, and could not be filtered by one, because the field
does not exist in the artifact (disposition §1.1, §1.2).

The fix is a KEY, not a hook, and the difference is the whole point: a hook has
to be called by :mod:`meep_gpu.subnormal_policy`'s install, which this package
does not edit, so it would be one more thing that silently does not happen. A key
needs nobody to remember. What follows measures that — including with the REAL
policy module driven through a REAL install, because the failure being pinned is
a late install and a mock of one proves nothing about the module that does it.

WHAT IT STILL DOES NOT PROVE, and no laptop can: that the recompiled binary is
different BYTES. CuPy's own on-disk cache key is computed above the seam the
policy strips ``-ftz=true`` at, so the rebuild this file proves happens can still
be served the previous policy's binary off disk. The second half is a fresh,
policy-token-carrying ``CUPY_CACHE_DIR`` per leg, which lives in the gate runner
and needs the device.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
import types

import pytest

from . import compile_cache

HERE = pathlib.Path(__file__).parent

#: A stand-in for a kernel source string. Its identity is what the memo keys on.
SOURCE_A = "extern \"C\" __global__ void k() { }"
SOURCE_B = "extern \"C\" __global__ void k() { /* mutated */ }"


class _CountingFactory:
    """A compile that is not a compile: it counts and returns a fresh object."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return object()


@pytest.fixture(autouse=True)
def _empty_memo():
    """Every test starts on a cold memo and leaves one behind.

    Module state shared across a pytest session is exactly how a stale entry gets
    served, which is the defect this file is about.
    """
    compile_cache.clear_kernel_cache()
    compile_cache.clear_compile_log()
    yield
    compile_cache.clear_kernel_cache()
    compile_cache.clear_compile_log()


# --------------------------------------------------------------------------
# The key: one entry per distinct (name, is_complex, options, policy, source).
# --------------------------------------------------------------------------

def test_one_kernel_under_two_policies_is_two_entries():
    """THE required half of the fix, stated as the disposition states it.

    Same name, same options, same source; two policies. Two entries, two
    compiles, and the two lookups may not hand back the same object — that
    object is a compiled binary and the policy decides its arithmetic.
    """
    factory = _CountingFactory()
    keep = compile_cache.kernel_cache_key(
        "step_B_pml_real", False, ("--fmad=false",), SOURCE_A,
        policy_token="keep|cupy_ftz_stripped=True|cache=/tmp/ftz_stripped")
    flush = compile_cache.kernel_cache_key(
        "step_B_pml_real", False, ("--fmad=false",), SOURCE_A,
        policy_token="flush|cupy_ftz_stripped=False|cache=/tmp/plain")

    assert keep != flush
    first = compile_cache.get_or_compile(keep, factory)
    second = compile_cache.get_or_compile(flush, factory)
    assert factory.calls == 2
    assert first is not second
    assert compile_cache.cache_size() == 2

    # And the memo is still a memo: a repeat under either policy compiles nothing.
    assert compile_cache.get_or_compile(keep, factory) is first
    assert compile_cache.get_or_compile(flush, factory) is second
    assert factory.calls == 2


def test_a_policy_installed_after_the_first_compile_is_not_served_the_earlier_binary(
        monkeypatch, tmp_path):
    """The late install, driven through the REAL policy module.

    This is the measured failure from the disposition: install-then-compile and
    compile-then-install produced different bytes on device, and the module could
    not tell the two apart. Here the same sequence runs on a laptop — compile
    first, install second — and the second lookup must MISS.

    The install is real (``install_subnormal_policy('keep')``), so what is being
    pinned is this module's reading of that module, not a mock of it.

    ``CUPY_CACHE_DIR`` IS SET, AND THAT IS NOT COSMETIC. A ``keep`` install with
    ``strict=True`` REFUSES a cache directory that does not carry the
    ``ftz_stripped`` token (``subnormal_policy.cupy_cache_reasons``), and it only
    reaches that check when CuPy is importable — so on a laptop without CuPy the
    install short-circuits to ``attained=True`` and the omission is invisible,
    while on the ONE host that runs the gate this test's outcome would depend on
    an ambient environment variable it does not control. A private tmp directory
    carrying the token makes the refusal satisfied by construction on every host.
    """
    from .. import subnormal_policy

    cache = tmp_path / f"cupy_cache_{subnormal_policy.CUPY_CACHE_POLICY_TOKEN}"
    cache.mkdir()
    monkeypatch.setenv(compile_cache.CUPY_CACHE_DIR_ENV, str(cache))
    assert not subnormal_policy.cupy_cache_reasons("keep", str(cache))

    factory = _CountingFactory()
    name, options = "step_D_pml_real", ("--fmad=false",)

    subnormal_policy._reset_for_tests()
    try:
        assert not subnormal_policy.policy_is_installed()
        before_token = compile_cache.compile_policy_token()
        assert before_token.startswith(compile_cache.UNINSTALLED), before_token

        cold = compile_cache.kernel_cache_key(name, False, options, SOURCE_A)
        first = compile_cache.get_or_compile(cold, factory)

        report = subnormal_policy.install_subnormal_policy("keep", strict=True)
        assert report["resolved"] == "keep", report
        assert subnormal_policy.policy_is_installed()

        after_token = compile_cache.compile_policy_token()
        assert after_token != before_token, (
            "the token did not move across a policy install, so the memo would "
            "serve the pre-install binary under the installed policy's name")
        warm = compile_cache.kernel_cache_key(name, False, options, SOURCE_A)
        assert warm != cold
        second = compile_cache.get_or_compile(warm, factory)
    finally:
        subnormal_policy._reset_for_tests()

    assert second is not first, (
        "the kernel compiled before the policy install was handed back after it")
    assert factory.calls == 2
    tokens = [entry["policy_token"] for entry in compile_cache.compile_log()]
    assert len(set(tokens)) == 2, tokens


def test_an_uninstalled_preference_is_not_a_policy_the_memo_separates_on(monkeypatch):
    """A resolved preference compiles nothing, so it may not split the memo.

    ``policy_stamp`` draws the same line and for the same reason: with
    ``MEEP_GPU_SUBNORMAL_POLICY=keep`` and no install, zero NVRTC calls have been
    made and zero options stripped. A token that read ``keep`` there would claim
    a binary property nothing in the process earned — and would make the memo
    miss on a change that moved no bytes.
    """
    from .. import subnormal_policy

    subnormal_policy._reset_for_tests()
    try:
        monkeypatch.setenv(subnormal_policy.POLICY_ENV, "keep")
        assert compile_cache.compile_policy_token().startswith(
            compile_cache.UNINSTALLED)
        monkeypatch.setenv(subnormal_policy.POLICY_ENV, "flush")
        assert compile_cache.compile_policy_token().startswith(
            compile_cache.UNINSTALLED)
    finally:
        subnormal_policy._reset_for_tests()


def test_the_token_moves_when_the_cupy_cache_directory_moves(monkeypatch):
    """The half a memo key cannot fix, folded in as far as it reaches.

    CuPy computes its disk-cache key ABOVE the seam the strip installs at
    (``subnormal_policy.py:87-89``), so the directory name is the only separator
    between one policy's binaries and another's. Two directories are two sets of
    bytes for the same source, and the memo has to be able to tell them apart —
    otherwise a runner that correctly took a private cache per leg would still be
    served the previous leg's kernel from THIS memo.
    """
    monkeypatch.setenv(compile_cache.CUPY_CACHE_DIR_ENV, "/tmp/leg_one")
    one = compile_cache.compile_policy_token()
    monkeypatch.setenv(compile_cache.CUPY_CACHE_DIR_ENV,
                       "/tmp/leg_two_ftz_stripped")
    two = compile_cache.compile_policy_token()
    assert one != two, (one, two)
    key_one = compile_cache.kernel_cache_key("step_B_pml_real", False, (), SOURCE_A,
                                             policy_token=one)
    key_two = compile_cache.kernel_cache_key("step_B_pml_real", False, (), SOURCE_A,
                                             policy_token=two)
    assert key_one != key_two


def test_a_mutated_source_is_never_served_the_unmutated_binary():
    """The armed-mutation trap, closed by construction rather than by discipline.

    The bit-identity probe rewrites a kernel's source string and re-launches. It
    also drops the memo when it does — but a leg that reports a pass for a
    mutation it never applied is a defect the sibling track hit THREE times, and
    the cheapest way to make it impossible is to put the source in the key.
    """
    factory = _CountingFactory()
    clean = compile_cache.kernel_cache_key("step_B_pml_real", False, (), SOURCE_A,
                                           policy_token="fixed")
    mutated = compile_cache.kernel_cache_key("step_B_pml_real", False, (), SOURCE_B,
                                             policy_token="fixed")
    assert compile_cache.get_or_compile(clean, factory) \
        is not compile_cache.get_or_compile(mutated, factory)
    assert factory.calls == 2
    digests = compile_cache.compiled_source_digests()
    assert len(digests) == 2, digests


def test_the_options_are_in_the_key():
    """``--fmad=false`` is a CORRECTNESS option here, not a tuning one."""
    factory = _CountingFactory()
    guarded = compile_cache.kernel_cache_key("step_B_pml_real", False,
                                             ("--fmad=false",), SOURCE_A,
                                             policy_token="fixed")
    plain = compile_cache.kernel_cache_key("step_B_pml_real", False, (), SOURCE_A,
                                           policy_token="fixed")
    assert guarded != plain
    compile_cache.get_or_compile(guarded, factory)
    compile_cache.get_or_compile(plain, factory)
    assert factory.calls == 2


# --------------------------------------------------------------------------
# The accounting the mutation legs read.
# --------------------------------------------------------------------------

def test_the_log_records_one_entry_per_compile_and_not_per_lookup():
    factory = _CountingFactory()
    key = compile_cache.kernel_cache_key("step_B_pml_real", False, (), SOURCE_A,
                                         policy_token="fixed")
    for _ in range(5):
        compile_cache.get_or_compile(key, factory)
    assert factory.calls == 1
    log = compile_cache.compile_log()
    assert len(log) == 1
    entry = log[0]
    assert entry["name"] == "step_B_pml_real"
    assert entry["policy_token"] == "fixed"
    assert entry["source_chars"] == len(SOURCE_A)
    assert len(entry["source_sha256"]) == 64


def test_clearing_reports_how_many_entries_went():
    """A nonzero drop where the caller believed the cache was cold is a signal."""
    factory = _CountingFactory()
    for source in (SOURCE_A, SOURCE_B):
        compile_cache.get_or_compile(
            compile_cache.kernel_cache_key("step_B_pml_real", False, (), source,
                                           policy_token="fixed"), factory)
    assert compile_cache.clear_kernel_cache() == 2
    assert compile_cache.cache_size() == 0
    # The LOG survives a cache clear on purpose: the mutation legs clear the
    # memo between guard sets and still have to account for what was compiled.
    assert len(compile_cache.compile_log()) == 2


# --------------------------------------------------------------------------
# The structural properties, pinned so they cannot be undone.
# --------------------------------------------------------------------------

def _module_level_imports(path: pathlib.Path) -> set:
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add((node.module or "").split(".")[0])
    return imported


def test_the_memo_module_pulls_in_no_device_dependency():
    """One ``import cupy`` here and every test above becomes a skip.

    The relative ``from .. import subnormal_policy`` is deliberately INSIDE a
    function, so it does not appear as a module-level import and cannot drag the
    package in during a by-path load.
    """
    allowed = {"__future__", "hashlib", "importlib", "os", "sys", "typing"}
    imported = _module_level_imports(HERE / "compile_cache.py")
    assert imported <= allowed, sorted(imported - allowed)


def test_the_policy_is_imported_rather_than_re_read():
    """One process-wide policy means one reading of it.

    ``compile_cache`` must reach :mod:`meep_gpu.subnormal_policy` and must not
    define its own ``FLUSH``/``KEEP``/``MATCH_MEEP`` — a second copy of the flag
    is a second answer, and the two would diverge on the first change to either.
    """
    source = (HERE / "compile_cache.py").read_text(encoding="utf-8")
    assert "subnormal_policy" in source
    tree = ast.parse(source)
    assigned = {target.id for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    forbidden = {"FLUSH", "KEEP", "MATCH_MEEP", "POLICIES", "REQUESTABLE"}
    assert not (assigned & forbidden), sorted(assigned & forbidden)


def test_the_kernel_module_holds_no_memo_of_its_own():
    """Read textually, so it holds without CuPy.

    ``step_curl_kernels`` must not carry a module-level ``_compiled_kernels``
    again, and ``_get_kernel`` must route through the shared memo. A second dict
    in that module is a second policy blind spot.
    """
    source = (HERE / "step_curl_kernels.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assigned = {target.id for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    assert "_compiled_kernels" not in assigned, (
        "step_curl_kernels re-grew its own memo; the policy-blind key is back")

    get_kernel = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_get_kernel")
    called = {ast.unparse(node.func) for node in ast.walk(get_kernel)
              if isinstance(node, ast.Call)}
    assert "compile_cache.get_or_compile" in called, sorted(called)
    assert "compile_cache.kernel_cache_key" in called, sorted(called)


def test_get_kernel_rebuilds_its_code_map_so_a_rewritten_source_is_seen():
    """Why the 14-entry map literal may NOT be memoized, measured on the module.

    ``probe_fused_kernel_bit_identity`` mutates a kernel by assigning over
    ``module._step_B_pml_real_kernel_code`` and re-launching. Until
    2026-08-15 the map was built inside ``_get_kernel``'s cache-MISS branch, so a
    rewrite plus ``_clear_kernel_cache()`` was enough; the source is now part of
    the memo key, so the map has to be read BEFORE a key exists to miss on, and
    it moved to the top of the call. Memoizing it there would look like a free
    0.784 us — and would pin the pre-mutation string for the life of the process,
    handing every armed leg the unmutated binary while its verdict read as a
    measurement. That is the sibling track's three-times defect, so it is pinned
    here rather than left to a comment.

    Driven with a stand-in ``cupy``: this asserts WHICH SOURCE reaches the
    factory, which is a host-side decision and needs no device.
    """
    constructed = []

    class _FakeRawKernel:
        def __init__(self, code, name, options=()):
            constructed.append((code, name, tuple(options)))
            self.code, self.name, self.options = code, name, tuple(options)

    stub = types.ModuleType("cupy")
    stub.RawKernel = _FakeRawKernel
    stub.ndarray = type("ndarray", (), {})
    saved_cupy = sys.modules.get("cupy")
    sys.modules["cupy"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "_kernels_under_a_cupy_stub", HERE / "step_curl_kernels.py")
        kernels = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = kernels
        spec.loader.exec_module(kernels)
    finally:
        if saved_cupy is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = saved_cupy

    # This by-path load got its OWN compile_cache (the ImportError fallback), so
    # drive that one rather than the package copy this file imported.
    cache = kernels.compile_cache
    cache.clear_kernel_cache()
    cache.clear_compile_log()

    original = kernels._step_B_pml_real_kernel_code
    first = kernels._get_kernel("step_B_pml_real", False)
    assert first.code == original

    mutated = original.replace("float curl = dtdx *", "float curl = 0.0f * dtdx *")
    assert mutated != original, "the mutation matched no text; the pin is vacuous"
    kernels._step_B_pml_real_kernel_code = mutated
    try:
        second = kernels._get_kernel("step_B_pml_real", False)
    finally:
        kernels._step_B_pml_real_kernel_code = original

    assert second is not first, (
        "_get_kernel handed back the pre-mutation kernel after the module's "
        "source string was rewritten: an armed mutation leg would report a "
        "verdict for a kernel it never mutated")
    assert second.code == mutated
    assert [entry["source_chars"] for entry in cache.compile_log()] == \
        [len(original), len(mutated)]
    assert len(cache.compiled_source_digests()) == 2
    cache.clear_kernel_cache()
    cache.clear_compile_log()
    sys.modules.pop("_kernels_under_a_cupy_stub", None)


def test_the_compile_log_counts_constructions_not_nvrtc_calls():
    """The log's own header may not claim more than the log measures.

    ``get_or_compile`` appends once per FACTORY INVOCATION — a ``cp.RawKernel``
    construction. A construction is not an NVRTC compile: CuPy's disk cache key
    is computed ABOVE the seam ``subnormal_policy`` strips ``-ftz=true`` at, and
    this project's own artifact records a leg
    (``cuda_policy_reach_policy_last.json``,
    ``after_install__cuda_kernel__module_memo_cleared``) where the memo was
    cleared, a fresh kernel was constructed and the bytes did not move. The
    docstrings said "one entry per NVRTC compile", which would have an auditor of
    an armed leg reading the wrong quantity.
    """
    source = (HERE / "compile_cache.py").read_text(encoding="utf-8")
    assert "per NVRTC compile" not in source, (
        "compile_cache claims to count NVRTC compiles; it counts factory "
        "invocations, and this project measured the two differing")
    for phrase in ("cp.RawKernel", "upper bound"):
        assert phrase in source, phrase
