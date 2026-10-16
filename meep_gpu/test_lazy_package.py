"""The package root binds its public names on first use (PEP 562), and that changes nothing else.

``import meep_gpu`` loads ``meep_gpu/__init__.py`` and no other module of the package; every
name in ``__all__`` is bound from its defining module the first time it is read, by attribute,
by ``from meep_gpu import name`` or by ``from meep_gpu import *``. Why: a kernel module imports
the package root before itself, so an eager root put the driver, the dispatch planners and the
MEEP lift into every process that touched a kernel, and every gate's import record held them
(``parity/meep_gpu/test_gate_import_closure.py`` holds the records to the gates' real needs).

Two halves are held here. THE PUBLIC SURFACE is the eager one's: the same names, the same
objects, ``dir()``, the submodules an eager import left bound on the package, and the lazy
``__version__``. NOTHING IS DECIDED AT IMPORT, so binding a name later decides nothing
differently: no module of the package reads the environment, installs or reads a subnormal
policy, or probes a backend at import, read off the source; and in a fresh interpreter the
host's floating-point state and the policy stamp are the same whichever order the modules
are imported in. The interpreter-level checks run in fresh processes, because this test
session has already imported half the package.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import meep_gpu

PACKAGE_ROOT = Path(meep_gpu.__file__).resolve().parent
CHECKOUT = PACKAGE_ROOT.parent


def _fresh(code: str) -> dict:
    """Run ``code`` in a fresh interpreter on this checkout; it prints one JSON line.

    The child sees what this process sees: this checkout first, then this process's
    ``PYTHONPATH``, and ``-S`` / ``-s`` when this process runs with them.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEEP_GPU_")}
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env.update(PYTHONPATH=os.pathsep.join([str(CHECKOUT), *inherited]), PYTHONDONTWRITEBYTECODE="1")
    flags = ["-B"] + (["-S"] if sys.flags.no_site else []) + (["-s"] if sys.flags.no_user_site else [])
    done = subprocess.run([sys.executable, *flags, "-c", code], capture_output=True, text=True,
                          cwd=str(CHECKOUT), timeout=300, env=env)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads([line for line in done.stdout.splitlines() if line.startswith("{")][-1])


def test_import_loads_the_package_root_alone():
    seen = _fresh("import json, sys\n"
                  "import meep_gpu\n"
                  "print(json.dumps({'loaded': sorted(k for k in sys.modules "
                  "if k == 'meep_gpu' or k.startswith('meep_gpu.'))}))\n")
    assert seen["loaded"] == ["meep_gpu"], seen


def test_every_exported_name_is_its_defining_modules_object():
    seen = _fresh(
        "import importlib, json\n"
        "import meep_gpu\n"
        "bad = []\n"
        "for name in meep_gpu.__all__:\n"
        "    value = getattr(meep_gpu, name)\n"
        "    if name == 'is_available':\n"
        "        continue\n"
        "    module, attribute = meep_gpu._LAZY[name]\n"
        "    if value is not getattr(importlib.import_module('meep_gpu.' + module), attribute):\n"
        "        bad.append(name)\n"
        "print(json.dumps({'bad': bad, 'all': sorted(meep_gpu.__all__), "
        "'lazy': sorted(meep_gpu._LAZY)}))\n")
    assert seen["bad"] == []
    assert sorted(set(seen["lazy"]) | {"is_available"}) == seen["all"]


def test_names_resolve_by_from_import_and_by_star():
    seen = _fresh(
        "import json\n"
        "from meep_gpu import lift_simulation, FdtdDriver, install_subnormal_policy\n"
        "import meep_gpu.from_meep, meep_gpu.driver, meep_gpu.subnormal_policy\n"
        "same = [lift_simulation is meep_gpu.from_meep.lift_simulation,\n"
        "        FdtdDriver is meep_gpu.driver.FdtdDriver,\n"
        "        install_subnormal_policy is meep_gpu.subnormal_policy.install_subnormal_policy]\n"
        "namespace = {}\n"
        "exec('from meep_gpu import *', namespace)\n"
        "missing = [n for n in meep_gpu.__all__ if n not in namespace]\n"
        "print(json.dumps({'same': same, 'missing': missing}))\n")
    assert seen == {"same": [True, True, True], "missing": []}


def test_dir_and_the_submodules_an_eager_import_left_bound():
    seen = _fresh(
        "import json, types\n"
        "import meep_gpu\n"
        "listed = dir(meep_gpu)\n"
        "kinds = {name: isinstance(getattr(meep_gpu, name), types.ModuleType)\n"
        "         for name in sorted(meep_gpu._SUBMODULES)}\n"
        "print(json.dumps({'listed': listed, 'kinds': kinds,\n"
        "                  'all': list(meep_gpu.__all__),\n"
        "                  'submodules': sorted(meep_gpu._SUBMODULES)}))\n")
    assert set(seen["all"]) <= set(seen["listed"])
    assert set(seen["submodules"]) <= set(seen["listed"])
    # ``meep_gpu.harminv`` and ``meep_gpu.subnormal_policy`` stay MODULES (the shadowing trap
    # the root's comments describe), and so does every other submodule it used to bind.
    assert all(seen["kinds"].values()), seen["kinds"]


def test_an_unlisted_submodule_is_still_an_attribute_error_until_imported():
    seen = _fresh(
        "import json\n"
        "import meep_gpu\n"
        "try:\n"
        "    meep_gpu.metal_kernels\n"
        "except AttributeError:\n"
        "    before = 'AttributeError'\n"
        "else:\n"
        "    before = 'bound'\n"
        "import meep_gpu.metal_kernels\n"
        "print(json.dumps({'before': before, 'after': hasattr(meep_gpu, 'metal_kernels')}))\n")
    assert seen == {"before": "AttributeError", "after": True}


def test_the_floating_point_state_and_the_policy_stamp_do_not_depend_on_import_order():
    """Root first and names bound late, against the modules imported eagerly first."""
    probe = (
        "import json\n"
        "{prelude}\n"
        "import meep_gpu\n"
        "for name in meep_gpu.__all__:\n"
        "    getattr(meep_gpu, name)\n"
        "from meep_gpu import backends, subnormal_policy\n"
        "print(json.dumps({{'flushed': bool(backends.subnormals_flushed()),\n"
        "                  'policy': subnormal_policy.get_subnormal_policy(),\n"
        "                  'stamp': subnormal_policy.policy_stamp()}}, sort_keys=True, default=repr))\n")
    eager = _fresh(probe.format(prelude="import meep_gpu.driver, meep_gpu.from_meep, "
                                        "meep_gpu.subnormal_policy, meep_gpu.backends"))
    lazy = _fresh(probe.format(prelude=""))
    assert lazy == eager
    assert lazy["flushed"] is False


def _is_main_block(node: ast.AST) -> bool:
    """``if __name__ == "__main__":``, which an import does not run."""
    test = getattr(node, "test", None) if isinstance(node, ast.If) else None
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__" and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__")


def _walk_executed(node: ast.AST):
    """``ast.walk`` that does not enter a function or lambda body (defining one runs nothing)."""
    todo = [node]
    while todo:
        current = todo.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                todo.extend(child.decorator_list if not isinstance(child, ast.Lambda) else ())
                if not isinstance(child, ast.Lambda):
                    todo.extend(d for d in child.args.defaults + child.args.kw_defaults if d is not None)
                continue
            todo.append(child)


def _module_level_reads(path: Path) -> list:
    """Environment reads, policy calls and backend probes a module makes AT IMPORT.

    The module-level statements (less an ``if __name__ == "__main__":`` block), and the
    bodies of the same-file functions they call (an import runs those too), followed through
    the calls those bodies make.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    probes = {"available_gpu", "cupy_available", "is_available", "install_subnormal_policy",
              "set_subnormal_policy", "get_subnormal_policy", "policy_stamp", "subnormals_flushed",
              "keep_subnormals", "missing_dependencies", "resolve_backend"}
    found: list = []
    pending = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and not _is_main_block(n)]
    visited: set = set()
    while pending:
        node = pending.pop()
        for sub in _walk_executed(node):
            if isinstance(sub, ast.Attribute) and sub.attr in ("environ", "getenv"):
                found.append(f"{path.name}:{sub.lineno} reads the environment")
            if isinstance(sub, ast.Call):
                name = sub.func.id if isinstance(sub.func, ast.Name) else (
                    sub.func.attr if isinstance(sub.func, ast.Attribute) else "")
                if name in probes:
                    found.append(f"{path.name}:{sub.lineno} calls {name}")
                if name in functions and name not in visited:
                    visited.add(name)
                    pending.extend(functions[name].body)
    return found


def test_no_module_reads_the_environment_installs_a_policy_or_probes_a_backend_at_import():
    """What makes a late binding safe: nothing is decided when a module is imported."""
    found = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name.startswith("test_") or path.name == "conftest.py":
            continue
        found.extend(_module_level_reads(path))
    assert found == [], "decided at import:\n" + "\n".join(found)


def test_the_import_time_scan_sees_a_planted_read():
    """The scan is not vacuous: a module-level call of a function that reads the environment."""
    text = "import os\n\ndef _mode():\n    return os.environ.get('X')\n\nMODE = _mode()\n"
    with tempfile.TemporaryDirectory() as scratch:  # never written into the package
        planted = Path(scratch) / "planted.py"
        planted.write_text(text, encoding="utf-8")
        assert _module_level_reads(planted) == ["planted.py:4 reads the environment"]
