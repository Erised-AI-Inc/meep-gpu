"""What each backend's DEVICE SOURCE is, and how to hash it without a device.

WHY THIS EXISTS. A weld binds a gate's PASS verdict to the sha256 of the Python files the
gate imported. That is a sound rule and a blunt one: it cannot tell a rewritten docstring
from a changed coefficient, so both cost a device re-run. Measured 2026-08-22, five prose
mentions in three modules drifted 76 of 70 welds across two tracks.

The CUDA track already solved this properly, and the other two should converge on its
answer rather than on a weaker common denominator. ``cuda_kernels/certification.json``
records ``device_source_sha256`` -- the sha256 of the exact source strings NVRTC compiles
-- and ``test_the_gate_bound_the_device_bytes_that_ship_today`` re-checks it on every run.
Under that rule ANY edit is free provided the device text is untouched, and any edit to
the device text is refused. This module gives the same rule the other two spellings it
needs.

THE THREE SPELLINGS OF "DEVICE SOURCE", and why they differ for real reasons:

* **CUDA** -- module-level string literals named ``*_kernel_code`` (plus the preludes they
  are concatenated with). Hashing them is EXACT: that text is what the compiler receives.
* **Metal** -- shader text produced by host-callable emitters. A module that exposes
  ``corpus_digest()`` already enumerates every specialisation and hashes the emitted text
  (see ``metal_kernels/offdiag_update_e.py:535``), which is equally exact. Deterministic
  and needs no MPS device: measured, two calls of ``bloch_constitutive_source`` return
  byte-identical 4 461-character sources.
* **Triton** -- the ``@triton.jit`` function itself IS the device source; there is no
  emitted string to hash. Its identity is the function's AST, plus the module-level
  constants the function reads, because a ``tl.constexpr`` default or a shared coefficient
  table reaches the compiled kernel exactly as the body does.

WHAT IS DELIBERATELY NOT USED FOR TRITON: the compiled PTX. It is obtainable --
``CompiledKernel.asm['ptx']``, which ``parity/meep_gpu/probe_ftz_ptx_census.py`` reads --
and it is the wrong thing to pin. PTX is compiler OUTPUT: arch-specific (that probe records
``-arch=compute_86`` against ``sm_86`` on one host) and Triton-version-specific, so a
toolchain bump would drift every digest without a line of the package changing. It also
needs the Triton toolchain, so the check could not run on a laptop. CUDA and Metal pin what
goes INTO the compiler; Triton must do the same.

FAIL CLOSED, ALWAYS. Where a file's device source cannot be established -- a Metal module
with no ``corpus_digest``, an unparsable file -- this returns nothing rather than something
weaker, and the caller keeps the strict byte rule. A silent degrade would turn the weld
into an honour system, which is the failure the welds exist to prevent.

THE PRINCIPLED TIER. A file that CONTAINS device source is pinned by it, and any host-side
edit to that file is free. A file that contains NONE -- ``launch.py``, ``driver.py`` -- can
reach the device only through host logic, so only comments and docstrings are free there
(:mod:`meep_gpu.code_identity`). That is not a fallback; it is the correct rule for each.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from .code_identity import canonical_dump, strip_docstrings

__all__ = ["triton_kernel_digests", "cuda_kernel_digests", "metal_shader_digests",
           "device_digests", "weld_survives_edit", "TRITON", "CUDA", "METAL", "NONE"]

TRITON, CUDA, METAL, NONE = "triton", "cuda", "metal", "none"


def _dump(node: ast.AST) -> str:
    return canonical_dump(node)


def _module_constants(tree: ast.Module) -> Dict[str, ast.AST]:
    """Module-level simple assignments, by target name."""
    out: Dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                out[node.target.id] = node.value
    return out


def _is_jit(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        if name == "jit":
            return True
    return False


def triton_kernel_digests(source: str) -> Dict[str, str]:
    """One digest per ``@triton.jit`` kernel: its body, plus the constants it reads.

    THE CONSTANTS ARE NOT OPTIONAL. A kernel that reads a module-level coefficient table
    or a ``tl.constexpr`` default compiles that value into the program, so a digest over
    the function body alone would call a changed coefficient "unchanged" and release a
    weld on arithmetic nobody ran. Captured ONE level deep, which is the depth the shipped
    kernels use; a constant defined in terms of another module's constant is NOT followed,
    and that bound is why the caller keeps the byte rule for anything it cannot establish.
    """
    tree = ast.parse(source)
    constants = _module_constants(tree)
    out: Dict[str, str] = {}
    for node in tree.body:
        if not _is_jit(node):
            continue
        isolated = ast.Module(body=[node], type_ignores=[])
        strip_docstrings(isolated)
        parts = [_dump(isolated)]
        referenced = sorted({n.id for n in ast.walk(node)
                             if isinstance(n, ast.Name) and n.id in constants})
        for name in referenced:
            parts.append(f"{name}={_dump(constants[name])}")
        out[node.name] = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return out


def cuda_kernel_digests(source: str) -> Dict[str, str]:
    """One digest per module-level device-source string literal.

    EXACT, and the reason the other two converge here: the value hashed is the text handed
    to NVRTC. Names ending ``_kernel_code`` are the kernels; ``*_PRELUDE`` constants are
    included because they are concatenated into the compiled program.
    """
    environment: Dict[str, str] = {}
    out: Dict[str, str] = {}
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if not (name.endswith("_code") or name.endswith("PRELUDE")):
            continue
        # CONCATENATIONS ARE EVALUATED, not skipped. The shipped kernels are spelled
        # ``_REAL_PML_PRELUDE + r"""..."""``, so a reader that accepted only a bare
        # literal would find 3 of the 14 and silently call the other 11 "no device
        # source" -- releasing every weld that touched them. Evaluated in file order
        # against the constants already seen, which is how they are formed at compile
        # time. This mirrors cuda_kernels/test_certification_record.device_sources, and
        # test_device_identity asserts the two agree kernel for kernel.
        try:
            value = eval(  # noqa: S307 - a literal/concat expression over strings already read
                compile(ast.Expression(ast.fix_missing_locations(node.value)),
                        "<device-source>", "eval"),
                {"__builtins__": {}}, dict(environment))
        except Exception:      # noqa: BLE001 - anything unevaluable has no establishable identity
            continue
        if isinstance(value, str):
            environment[name] = value
            out[name] = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return out


def metal_shader_digests(module_path: Union[str, Path]) -> Optional[Dict[str, str]]:
    """The module's own ``corpus_digest()``, if it publishes one.

    Returns ``None`` when the module does not, rather than substituting something weaker.
    Importing is required because the emitted text is what is being hashed, and only the
    module can enumerate its own specialisations.
    """
    path = Path(module_path)
    package = f"meep_gpu.metal_kernels.{path.stem}"
    try:
        import importlib
        module = importlib.import_module(package)
    except Exception:      # noqa: BLE001 - an unimportable module has no establishable identity
        return None
    digest = getattr(module, "corpus_digest", None)
    if callable(digest):
        try:
            result = digest()
        except Exception:      # noqa: BLE001
            return None
        if isinstance(result, dict) and result.get("sha256"):
            return {"corpus": str(result["sha256"])}
        if isinstance(result, str):
            return {"corpus": result}
        return None

    # NO PUBLISHED DIGEST: derive one from the module's own emitters, from OUTSIDE the
    # module. Three of forty-one Metal modules published a corpus_digest, which left 40
    # welds unable to pin device source. The obvious repair -- adding corpus_digest to
    # each module -- is self-defeating: editing a kernel module drifts every weld that
    # records it, so publishing the pin would first break the thing it is meant to
    # protect. Computing it here edits nothing and pins the same bytes.
    from .metal_kernels import corpus as _corpus  # noqa: PLC0415
    emitters = sorted(
        name for name in dir(module)
        if name.endswith("_source") and not name.startswith(("_", "refuted"))
        and callable(getattr(module, name, None))
        and getattr(getattr(module, name), "__module__", "") == module.__name__)
    if not emitters:
        return None
    out: Dict[str, str] = {}
    overrides = _corpus.OVERRIDES.get(path.stem, {})
    for name in emitters:
        try:
            result = _corpus.corpus_digest_for(getattr(module, name), **overrides)
        except Exception:      # noqa: BLE001 - unknown domain, or a corpus that is empty
            continue           # FAIL CLOSED: this emitter simply is not pinned
        out[name] = str(result["sha256"])
    return out or None


def device_digests(path: Union[str, Path]) -> Tuple[str, Dict[str, str]]:
    """``(kind, digests)`` for a file, or ``(NONE, {})`` if it carries no device source."""
    path = Path(path)
    if path.suffix != ".py":
        return NONE, {}
    parts = path.as_posix()
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return NONE, {}
    try:
        if "/metal_kernels/" in parts:
            shaders = metal_shader_digests(path)
            return (METAL, shaders) if shaders else (NONE, {})
        if "/cuda_kernels/" in parts:
            found = cuda_kernel_digests(source)
            return (CUDA, found) if found else (NONE, {})
        if "/triton_kernels/" in parts:
            found = triton_kernel_digests(source)
            return (TRITON, found) if found else (NONE, {})
    except SyntaxError:
        return NONE, {}
    return NONE, {}


def weld_survives_edit(path: Union[str, Path], entry: dict, name: str) -> bool:
    """May a weld keep its PASS although ``name`` no longer hashes to what it recorded?

    ONE HOME FOR THE RULE. Every drift clause in the suite calls this rather than carrying
    its own copy, because a rule spelled in twenty places is twenty places to disagree --
    and the disagreement that matters is the one that silently ADMITS an edit.

    Two tiers, and which applies is decided by the file, not by the caller:

    * the file CARRIES DEVICE SOURCE and the weld pinned it -- then the device source is
      the whole question. Any host-side edit is free; any edit reaching the compiled
      program is refused. This is the CUDA track's long-standing rule.
    * the file carries NONE -- then only comments and docstrings are free, because
      everything else in it reaches the device through host logic.

    Returns ``False`` whenever the answer cannot be established: an unpinned file, an
    unparsable one, a record with neither digest. The caller then refuses, which is the
    behaviour the welds had before any of this existed.
    """
    path = Path(path)
    pinned = (entry.get("device_sha256") or {}).get(name)
    if pinned:
        kind, digests = device_digests(path)
        if kind == pinned.get("kind") and digests == pinned.get("digests"):
            return True
        return False        # a pinned file whose device source moved is refused outright
    recorded_code = (entry.get("code_sha256") or {}).get(name)
    if recorded_code is not None and path.suffix == ".py":
        try:
            from .code_identity import code_digest
            return code_digest(path.read_text(encoding="utf-8")) == recorded_code
        except (SyntaxError, OSError):
            return False
    return False
