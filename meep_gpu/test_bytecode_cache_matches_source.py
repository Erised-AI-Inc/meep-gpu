"""The package's own bytecode caches must agree with the sources beside them.

WHY THIS EXISTS. On 2026-08-22 a case-only edit to `metal_kernels/complex_fields.py`
(`bloch_pml_curl_step` -> `..._STEP` in the Metal entry point) was reverted 14 ms after
CPython had written the `.pyc`. A timestamp-based pyc header validates on source mtime
and SIZE; the edit was case-only and therefore size-preserving, and the revert landed
inside the same integer second. Both of CPython's checks passed, so it kept serving
bytecode that no longer matched the file on disk.

The cost was not the failure but the DIAGNOSIS. Fifty tests failed with

    RuntimeError: Failed to create function state object for: bloch_pml_curl_step

which reads like a Metal pipeline-state failure and is not -- it is torch's message for
"no function of that name in this library". The source was clean against HEAD, so the
investigation went through toolchain versions, buffer limits and uncommitted-edit
interaction before anyone compared the RUNTIME constant to a fresh `compile()`. All 2944
legal specialisations of three shader emitters were affected; 1728 of them have no test
caller at all.

WHAT IT CHECKS, and what it deliberately does not. Only caches CPython would actually
SERVE: hash-based pycs verify their own content, and a cache whose header disagrees with
its source is recompiled on the next import and is harmless. The dangerous set is the one
whose header agrees while the code does not, which is exactly the shape above.
"""

from __future__ import annotations

import marshal
import pathlib
import struct

PACKAGE = pathlib.Path(__file__).parent


def _strings(code, out):
    for constant in code.co_consts:
        if isinstance(constant, str):
            out.append(constant)
        elif hasattr(constant, "co_consts"):
            _strings(constant, out)
    return out


def _served_caches():
    """Every (pyc, source) CPython would serve without recompiling."""
    for pyc in sorted(PACKAGE.rglob("__pycache__/*.pyc")):
        if "-pytest-" in pyc.name:
            continue                      # assertion-rewritten by pytest, by design
        source = pyc.parent.parent / (pyc.name.split(".")[0] + ".py")
        if not source.exists():
            continue                      # a cache for a deleted module is not served
        try:
            data = pyc.read_bytes()
            if len(data) < 16 or struct.unpack("<I", data[4:8])[0] != 0:
                continue                  # hash-based: CPython verifies the content itself
            mtime, size = struct.unpack("<II", data[8:16])
        except Exception:                 # noqa: BLE001 - an unreadable cache is not served
            continue
        stat = source.stat()
        if (int(stat.st_mtime) & 0xFFFFFFFF) != mtime or (stat.st_size & 0xFFFFFFFF) != size:
            continue                      # header disagrees -> recompiled -> harmless
        yield pyc, source


def test_no_served_bytecode_cache_disagrees_with_its_source():
    wrong = []
    checked = 0
    for pyc, source in _served_caches():
        try:
            cached = marshal.loads(pyc.read_bytes()[16:])
            # dont_inherit=True IS LOAD-BEARING. `compile` inherits the CALLING
            # module's __future__ statements by default, and this file carries
            # `from __future__ import annotations`. Inheriting it compiles every
            # annotation to a string constant that the real import never produced,
            # so five CUDA kernel modules reported a mismatch that did not exist --
            # a false positive that took longer to clear than the defect this test
            # was written for. Compile the source as PYTHON would at import.
            fresh = compile(source.read_text(encoding="utf-8"), str(source), "exec",
                            dont_inherit=True)
        except Exception as exc:          # noqa: BLE001
            wrong.append(f"{pyc.relative_to(PACKAGE)}: unreadable ({exc!r})")
            continue
        checked += 1
        if sorted(_strings(cached, [])) != sorted(_strings(fresh, [])):
            wrong.append(
                f"{pyc.relative_to(PACKAGE)} is SERVED but its string constants differ "
                f"from a fresh compile of {source.name}. `touch {source}` to force a "
                f"recompile; the source itself is fine.")
    assert not wrong, wrong
    # A run with no caches at all (a fresh clone, or PYTHONDONTWRITEBYTECODE) checks
    # nothing and must not read as a pass over the whole package. Zero is legitimate, so
    # this records rather than asserts a floor -- but a scan that silently found nothing
    # while caches exist on disk is a broken scan.
    if any(PACKAGE.rglob("__pycache__/*.pyc")):
        assert checked or not list(_served_caches()), (
            "caches exist on disk but the scan checked none of them")
