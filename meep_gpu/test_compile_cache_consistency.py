"""One invariant, three backends: a compile cache must key on everything that can
change the binary it hands back.

WHY THIS FILE EXISTS. The three tracks memoize compiled kernels in three different
places, for three good reasons, and each is individually correct today:

* **CUDA** -- :func:`meep_gpu.cuda_kernels.compile_cache.kernel_cache_key` folds
  ``(name, is_complex, options, source, policy_token)``. The policy is IN THE KEY because
  CuPy appends ``-ftz=true`` to every NVRTC compile and the strip that removes it happens
  BELOW the seam CuPy computes its own disk-cache key from, so the same source under two
  policies is two different binaries. That module's docstring records what the weaker
  ``(name, is_complex)`` key cost: a policy installed after first compile was never seen,
  and the second caller got the first caller's binary under the second caller's name.
* **Triton** -- no cache module here; Triton's own JIT cache is keyed on the constexpr
  signature, and :func:`~meep_gpu.subnormal_policy.install_triton_policy` folds the policy
  into that key and drops the in-process caches, because the policy genuinely moves the
  binary (an LLVM denormal attribute under flush; a libdevice ftz rewrite under keep).
* **Metal** -- ``metal_kernels.device._LIBRARY_CACHE`` keys on the EXACT SOURCE STRING and
  nothing else. That is complete, and only because no policy lever reaches MPS: the policy
  machinery installs for ``host``, ``cupy`` and ``triton``, and there is no MPS executor
  at all.

THE RISK THIS FILE GUARDS, and it is a quiet one. Metal's key is complete because of an
ABSENCE. The day an MPS subnormal lever is added -- or any other per-run compile switch --
``_LIBRARY_CACHE`` silently starts serving one policy's library to another, with no error
and no test failing, which is exactly the defect CUDA already paid for once. So the
absence is asserted here rather than assumed: add an executor and this file fails, naming
the cache that now needs a wider key.

These are cheap structural checks, not device tests. They run anywhere.
"""

from __future__ import annotations

import inspect

import pytest


def test_the_cuda_cache_key_folds_policy_and_source():
    """Both are load-bearing: policy because CuPy's strip is below its own cache seam,
    source because a mutation harness that forgets to drop the memo must still miss."""
    from .cuda_kernels.compile_cache import kernel_cache_key

    parameters = list(inspect.signature(kernel_cache_key).parameters)
    for required in ("name", "is_complex", "options", "source", "policy_token"):
        assert required in parameters, (
            f"{required!r} left the CUDA cache key; a lookup that ignores it can serve a "
            f"binary compiled under a different {required}")


def test_the_cuda_key_actually_separates_two_policies():
    """The signature is not the claim -- the values are."""
    from .cuda_kernels.compile_cache import kernel_cache_key

    common = ("step_B_pml_real", False, ("--fmad=false",), "extern \"C\" ...")
    keep = kernel_cache_key(*common, policy_token="ieee_keep_ftz_stripped")
    flush = kernel_cache_key(*common, policy_token="meep_x86_flush")
    assert keep != flush, (
        "two policies produced one cache key; the second caller would be served the "
        "first caller's binary, which is the defect compile_cache.py was written for")


def test_the_cuda_key_separates_two_sources():
    from .cuda_kernels.compile_cache import kernel_cache_key

    a = kernel_cache_key("k", False, (), "SOURCE A", policy_token="p")
    b = kernel_cache_key("k", False, (), "SOURCE B", policy_token="p")
    assert a != b, "a mutated source shares a key with the original"


def test_the_metal_library_cache_is_keyed_on_the_whole_source():
    """Every specialisation is a distinct string, so the string is a complete key --
    given that nothing else can move a Metal binary. The next test holds that given."""
    from .metal_kernels import device

    assert isinstance(device._LIBRARY_CACHE, dict)
    source = inspect.getsource(device.compile_source)
    assert "_LIBRARY_CACHE.get(source)" in source, (
        "compile_source no longer keys on the source string; if it now keys on "
        "something narrower, two specialisations can collide")


def test_no_policy_lever_reaches_mps_and_metals_cache_may_stay_source_keyed():
    """THE TRIPWIRE. Metal's cache is complete because of an absence, so the absence is
    asserted. If an MPS executor appears, `_LIBRARY_CACHE` must gain that dimension the
    way the CUDA memo has `policy_token` -- otherwise it serves one policy's library
    under another's name, silently."""
    from meep_gpu import subnormal_policy

    installers = {name for name in dir(subnormal_policy)
                  if name.startswith("install_") and name.endswith("_policy")}
    unexpected = installers - {"install_host_policy", "install_cupy_policy",
                               "install_triton_policy", "install_subnormal_policy"}
    assert not unexpected, (
        f"a new policy executor appeared ({sorted(unexpected)}). If it reaches MPS, "
        f"metal_kernels.device._LIBRARY_CACHE is now keyed too narrowly: it holds only "
        f"the source string, so two policies would share one library. Widen that key, "
        f"then widen this test.")


def test_the_triton_policy_install_says_it_folds_into_the_cache_key():
    """Triton has no cache module here because its own cache is the one that matters;
    what the package owns is folding the policy into that key and dropping the caches."""
    from meep_gpu import subnormal_policy

    body = inspect.getsource(subnormal_policy.install_triton_policy)
    assert "cache" in body.lower(), (
        "install_triton_policy no longer mentions the cache; if it stopped folding the "
        "policy into Triton's key or stopped dropping the in-process caches, a binary "
        "compiled under the previous policy survives the install")


@pytest.mark.parametrize("backend,dimension", [
    ("cuda", "policy_token"),
    ("cuda", "source"),
])
def test_the_cuda_dimensions_are_named_in_its_own_docstring(backend, dimension):
    """A key whose reasons are not written down is a key someone narrows later."""
    from .cuda_kernels import compile_cache

    assert dimension.split("_")[0] in (compile_cache.__doc__ or "").lower(), dimension
