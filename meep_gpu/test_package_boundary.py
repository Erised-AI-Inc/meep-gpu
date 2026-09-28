"""Structural tests for the standalone numerical-package boundary.

Plus one structural clause that spans the package AND the parity harnesses that
gate it, because the defect it catches was found in the second tree and could
land in either.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import meep_gpu


PACKAGE_ROOT = Path(__file__).resolve().parent
#: The gates, probes and validators that measure this package. Not importable as
#: a package and not shipped, but the shadowing guard below covers it: that is
#: where the defect it exists to catch was found.
PARITY_ROOT = PACKAGE_ROOT.parent / "parity" / "meep_gpu"
FORBIDDEN_IMPORT_ROOTS = {"app", "fastapi", "h5py", "pydantic"}
# The one directory allowed to import cupy at module level: raw-kernel source that
# only fastpath.py loads, lazily, after the backend is known to be CuPy.
CUDA_KERNEL_DIR = PACKAGE_ROOT / "cuda_kernels"


def test_production_modules_do_not_import_host_application_code():
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots = {alias.name.partition(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots = {node.module.partition(".")[0]}
            else:
                continue
            forbidden = sorted(imported_roots & FORBIDDEN_IMPORT_ROOTS)
            if forbidden:
                relative_path = path.relative_to(PACKAGE_ROOT)
                violations.append(f"{relative_path}:{node.lineno}: {', '.join(forbidden)}")
    assert not violations, "host dependencies crossed into meep_gpu:\n" + "\n".join(violations)


def test_public_package_root_exposes_the_solver_not_a_host_adapter():
    assert "FdtdDriver" in meep_gpu.__all__
    assert "GpuFdtdBackend" not in meep_gpu.__all__
    assert "FdtdJobSpec" not in meep_gpu.__all__


def test_no_module_level_cupy_import_outside_cuda_kernels():
    """The backends.py package rule, structurally: cupy only ever imports lazily.

    ``cuda_kernels/step_curl_kernels.py`` imports cupy at module level by design —
    which is exactly why nothing may import IT at module level either, and why
    ``fastpath.plan_fast_path`` defers both imports until the backend is CuPy.
    """
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name.startswith("test_") or CUDA_KERNEL_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:  # Module level only: function-local imports stay legal.
            roots: set[str] = set()
            if isinstance(node, ast.Import):
                roots = {alias.name.partition(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = {node.module.partition(".")[0]}
            if "cupy" in roots:
                violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = {alias.name for alias in node.names}
                if isinstance(node, ast.ImportFrom) and node.module and "cuda_kernels" in node.module:
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} (cuda_kernels)")
                elif any("cuda_kernels" in name for name in names):
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno} (cuda_kernels)")
    assert not violations, (
        "module-level cupy / cuda_kernels imports outside cuda_kernels/:\n" + "\n".join(violations)
    )


#: The one directory allowed to import torch at module level: the Metal kernel
#: sources, which launch through ``torch.mps.compile_shader`` and cannot be written
#: without it.
METAL_KERNEL_DIR = PACKAGE_ROOT / "metal_kernels"


def test_no_module_level_torch_or_metal_kernels_import_outside_metal_kernels():
    """The Metal mirror of the cupy rule, and it exists for a second reason too.

    ``metal_kernels`` imports torch at module level by design — which is why
    nothing outside it may import EITHER at module level. The cost of breaking this
    is not an ImportError: torch imports on any Mac, so a module-level import here
    would simply load several hundred megabytes of framework into every process
    that touches the fast path, including a pure NumPy array-path run that will
    never reach a device.

    THE SECOND REASON IS THE DISPATCH SEAM ITSELF. ``fastpath`` reaches the Metal
    table through ``meep_gpu/metal_dispatch.py``, and that module is imported at the
    backend rung — so a module-level torch there would be paid for by the CuPy table
    as well. Function-local stays legal everywhere, which is how both files reach
    what they need without anyone else paying for it.
    """
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name.startswith("test_") or METAL_KERNEL_DIR in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:  # Module level only: function-local imports stay legal.
            roots: set[str] = set()
            if isinstance(node, ast.Import):
                roots = {alias.name.partition(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                base = (node.module or "").partition(".")[0]
                if base:
                    roots.add(base)
                if node.level:
                    roots |= {alias.name.partition(".")[0] for alias in node.names}
            crossed = sorted(roots & {"torch", "metal_kernels"})
            if crossed:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}: "
                    f"{', '.join(crossed)}")
    assert not violations, (
        "module-level torch / metal_kernels imports outside metal_kernels/:\n"
        + "\n".join(violations))


def test_numpy_backend_stepping_never_imports_metal_kernels():
    """A NumPy step with ``MEEP_GPU_DISPATCH=0`` must not reach the Metal package at all.

    RUNG 2 REFUSES BEFORE RUNG 3, and this is what measures that ordering rather
    than trusting it. With ``MEEP_GPU_DISPATCH=0`` the ladder declines at the
    enable, before the candidate-table rung asks the hardware anything — so a run
    that has opted out must not pay for an MPS availability probe, a Metal frontend
    read, or the import of a kernel module. Set to ``0`` explicitly rather than
    unset: dispatch is on by default, so an unset enable now passes rung 2.

    THE DRIVER IS A GPU DRIVER'S ENGINE (``gpu="metal"``, set by hand so the test
    needs no device): only a GPU driver's freeze reaches the ladder at all. A
    ``prefer_gpu=False`` driver is the NumPy reference, which never plans and so
    imports nothing by construction; ``test_prefer_gpu.py`` holds that in a fresh
    process.

    Measured as "no NEW modules", not "no modules", so a session whose other tests
    legitimately loaded the Metal package still runs this test meaningfully.
    """
    import numpy as np

    from meep_gpu.driver import FdtdDriver

    already_loaded = {name for name in sys.modules
                      if name.startswith("meep_gpu.metal_kernels")}
    import os as _os

    previous = _os.environ.get("MEEP_GPU_DISPATCH")
    _os.environ["MEEP_GPU_DISPATCH"] = "0"
    try:
        driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8)
        driver.gpu = "metal"
        driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
        driver.step()  # Freezes the configuration: the fast-path plan is built here.
        assert driver.active_step_path == "array"
        assert driver.fast_path_report()["refused_because"] == (
            "dispatch disabled by MEEP_GPU_DISPATCH=0")
        newly_loaded = {name for name in sys.modules
                        if name.startswith("meep_gpu.metal_kernels")
                        and name not in already_loaded}
        driver.close()
    finally:
        if previous is None:
            _os.environ.pop("MEEP_GPU_DISPATCH", None)
        else:
            _os.environ["MEEP_GPU_DISPATCH"] = previous
    assert not newly_loaded, (
        f"an opted-out NumPy step imported Metal modules: {sorted(newly_loaded)}")


def test_the_cuda_table_sibling_pulls_in_neither_cupy_nor_the_kernel_package():
    """``fastpath_cuda.py`` is DATA plus functions, and importing it must cost that.

    It is imported at the backend rung on every CuPy host, beside the ladder — so a
    module-level ``cupy`` or ``cuda_kernels`` import there would be paid for by the
    Triton table as well, on a run that may never compose the second table at all.
    Everything that needs either is function-local inside it, which is the same rule
    ``fastpath.py`` and ``metal_dispatch.py`` are held to and for the same reason.
    """
    tree = ast.parse((PACKAGE_ROOT / "fastpath_cuda.py").read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in tree.body:
        roots: set[str] = set()
        names: set[str] = set()
        if isinstance(node, ast.Import):
            roots = {alias.name.partition(".")[0] for alias in node.names}
            names = {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or "").partition(".")[0]
            if base:
                roots.add(base)
            names = {alias.name for alias in node.names}
            if node.level:
                roots |= {alias.name.partition(".")[0] for alias in node.names}
        crossed = sorted(roots & {"cupy", "cuda_kernels", "torch"})
        if crossed or any("cuda_kernels" in name for name in names):
            violations.append(f"fastpath_cuda.py:{node.lineno}: "
                              f"{', '.join(crossed) or 'cuda_kernels'}")
    assert not violations, (
        "module-level device imports in fastpath_cuda.py:\n" + "\n".join(violations))


def test_numpy_backend_stepping_never_imports_cuda_kernels(monkeypatch):
    """The array path plus the fast-path PLANNING must stay off the kernel file.

    ``cuda_kernels/step_curl_kernels.py`` imports cupy at module level, so pulling
    it in on a NumPy host is an ImportError — and pulling it in lazily but
    needlessly would spend NVRTC/compile machinery on a run that cannot use it.
    Measured as "no NEW modules", not "no modules", so a CUDA host whose other
    tests legitimately loaded cupy still runs this test meaningfully.

    THE PLANNING RUNS PAST THE ENABLE. Dispatch is on by default, so the enable is
    left unset and the ladder reaches the candidate-table rung; the Metal hardware
    probe is answered "no" so a Mac refuses there as a NumPy host without a device
    does ("backend is not CuPy"), instead of planning through the Metal table.
    """
    import numpy as np

    from meep_gpu import fastpath
    from meep_gpu.driver import FdtdDriver

    monkeypatch.delenv(fastpath.DISPATCH_ENABLE, raising=False)
    monkeypatch.delenv(fastpath.FUSED_KILL_SWITCH, raising=False)
    monkeypatch.setattr(fastpath, "metal_hardware_present", lambda: False)

    already_loaded = {
        name for name in sys.modules
        if name.partition(".")[0] == "cupy" or name.startswith("meep_gpu.cuda_kernels")
    }
    driver = FdtdDriver(cell_size=(0.5, 0.5, 0.5), resolution=8)
    driver.gpu = "metal"  # a GPU driver's engine, so the planner runs to rung 3
    driver.set_epsilon(np.full(driver.shape, 1.0, dtype=np.float32))
    driver.step()  # Freezes the configuration: the fast-path plan is built here.
    assert driver.active_step_path == "array"
    reason = driver.fast_path_report()["refused_because"]
    assert "backend is not CuPy" in reason, reason
    newly_loaded = {
        name for name in sys.modules
        if (name.partition(".")[0] == "cupy" or name.startswith("meep_gpu.cuda_kernels"))
        and name not in already_loaded
    }
    driver.close()
    assert not newly_loaded, (
        f"a NumPy-backend step imported GPU-only modules: {sorted(newly_loaded)}"
    )


# ---------------------------------------------------------------------------
# One name, two definitions — the shadowing guard
# ---------------------------------------------------------------------------
#
# WHAT THIS CATCHES, MEASURED, not hypothesised. Two instances existed when this
# was written, one in each tree:
#
#   * ``parity/meep_gpu/probe_fused_kernel_bit_identity.py`` defined
#     ``reference_curl`` twice. The second definition rebound the name, so the
#     first one's caller was left passing a signature that no longer existed and
#     the all-periodic bit-identity leg raised ``TypeError`` on its first case —
#     for five days, on a gate whose entire job is to be the thing you trust.
#   * ``meep_gpu/test_triton_nonlinear_update_e.py`` defined one mutation test
#     twice. pytest binds by name, so the earlier body was collected under the
#     later one's and a real, passing assertion silently left the suite.
#
# Neither is visible to any test that does not call the shadowed name, which is
# exactly the shape a test suite cannot notice: the survivor works, the loser is
# unreachable, and nothing reports a count that changed.

DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
#: Branch labels for statements that run REGARDLESS of which branch was taken,
#: so a binding under one is never mutually exclusive with anything.
UNCONDITIONAL_BRANCHES = {"finally"}

#: (path relative to the repository root, bound name) -> why the rebinding is legitimate.
#:
#: EXPLICIT AND EMPTY, and it must stay that way unless a reviewer signs a
#: reason. The two rebinding classes the census over these trees adjudicated as
#: legitimate are excluded BY CONSTRUCTION below rather than by name, and neither
#: exclusion can hide a shadowed definition:
#:
#:   * ``import ctypes`` followed by ``import ctypes.util`` — backends.py:19-20,
#:     subnormal_policy.py:135-136, test_triton_conductivity.py:31-32,
#:     test_triton_no_pml_constitutive.py:47-48. Both statements bind the SAME
#:     module object, and an import is not a definition, so the "at least one
#:     definition" clause never admits the pair.
#:   * ``if triton is not None: @triton.jit def K(...)`` / ``else: K = None`` —
#:     the platform fallback in offdiag_update_e.py, cylindrical_triton.py,
#:     dispersive_fused_pair.py, dispersive_update_e.py, folded_offdiag_update_e.py,
#:     fused_ade_state.py, fused_dispersive_chain.py and nonlinear_update_e.py.
#:     The body and the orelse of one ``if`` cannot both execute, which
#:     ``_mutually_exclusive`` decides from the tree — it does NOT excuse a
#:     conditional definition that competes with an unconditional one.
LEGITIMATE_REBINDINGS: dict = {}


def _module_bindings(tree: ast.Module) -> dict:
    """Every module-scope binding: name -> [(line, kind, branch path)].

    Recurses through module-level ``if``/``try``/``for``/``while``/``with`` — a
    definition nested in one still binds a module global — recording the branch
    it sits in so mutually exclusive alternatives can be told from real
    competitors. Function and class BODIES are not entered: those bind locally.
    """
    found: dict = {}

    def visit(body, path):
        for node in body:
            names, kind = [], None
            if isinstance(node, DEFINITION_NODES):
                names, kind = [node.name], "definition"
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                kind = "assignment"
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names, kind = [node.target.id], "assignment"
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [(a.asname or a.name).partition(".")[0] for a in node.names]
                kind = "import"
            for name in names:
                found.setdefault(name, []).append((node.lineno, kind, path))

            if isinstance(node, ast.If):
                visit(node.body, path + ((id(node), "if"),))
                visit(node.orelse, path + ((id(node), "else"),))
            elif isinstance(node, ast.Try):
                # body and orelse BOTH run when nothing was raised, so they share
                # a label; only a handler excludes them.
                visit(node.body, path + ((id(node), "try"),))
                visit(node.orelse, path + ((id(node), "try"),))
                for index, handler in enumerate(node.handlers):
                    visit(handler.body, path + ((id(node), f"except{index}"),))
                visit(node.finalbody, path + ((id(node), "finally"),))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While,
                                   ast.With, ast.AsyncWith)):
                visit(node.body, path + ((id(node), "body"),))
                visit(getattr(node, "orelse", []), path + ((id(node), "body"),))

    visit(tree.body, ())
    return found


def _mutually_exclusive(left: tuple, right: tuple) -> bool:
    """True when no single execution can reach both bindings."""
    for (left_node, left_branch), (right_node, right_branch) in zip(left, right):
        if left_node != right_node:
            return False          # Different constructs: both can run.
        if left_branch != right_branch:
            return (left_branch not in UNCONDITIONAL_BRANCHES
                    and right_branch not in UNCONDITIONAL_BRANCHES)
    return False


def shadowed_definitions(path: Path, label: str) -> list:
    """Every name this module binds twice where at least one binding is a definition."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # A file that will not parse is its own defect.
        return [f"{label}: does not parse: {exc}"]
    violations = []
    for name, places in sorted(_module_bindings(tree).items()):
        if (label, name) in LEGITIMATE_REBINDINGS or len(places) < 2:
            continue
        clashes = [
            (first, second)
            for index, first in enumerate(places)
            for second in places[index + 1:]
            if (first[1] == "definition" or second[1] == "definition")
            and not _mutually_exclusive(first[2], second[2])
        ]
        if clashes:
            where = sorted({f"{line} ({kind})"
                            for pair in clashes for line, kind, _ in pair})
            violations.append(f"{label}: {name} bound at {', '.join(where)}")
    return violations


def test_no_module_binds_one_name_to_two_competing_definitions():
    """A definition whose name a later binding takes over is dead code at best.

    The rule is deliberately stricter than "the loser still has a caller": with a
    caller it is a TypeError waiting for someone to select that leg, without one
    it is an unreachable body that reads like live coverage. Both are defects and
    the fix for both is the same — rename one, outright.
    """
    violations: list[str] = []
    scanned = 0
    for root in (PACKAGE_ROOT, PARITY_ROOT):
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            violations.extend(shadowed_definitions(
                path, str(path.relative_to(PACKAGE_ROOT.parent))))

    assert scanned > 400, (
        f"the scan reached only {scanned} files; it is not looking at the trees "
        "it claims to cover")
    assert not violations, (
        "a module binds one name to two competing definitions — the later one "
        "wins at import and the earlier one is unreachable by name:\n  "
        + "\n  ".join(violations))
