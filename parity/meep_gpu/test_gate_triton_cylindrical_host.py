"""Host checks of the Triton cylindrical gates' kernel mutations (no device, no Triton).

The mutation legs of ``gate_triton_cylindrical`` (the real m = 0 family) and
``gate_triton_cylindrical_complex`` rewrite the shipped radial-scan and curl kernel text
and compile each copy on the device, under the campaign's subnormal policy. Two harness
defects read as gate failures there, and both are decided by text, so both are checked
here against the shipped source read off the kernel files:

* a needle that no longer matches the shipped text disarms its mutation:
  ``apply_edits`` raises and the battery stops;
* a copy that spells a plain ``/`` on floats compiles to ``div.full.f32``, emitted as
  inline asm that neither subnormal policy reaches. Under ``flush`` the policy's PTX
  audit refuses that copy at compile, so the gate dies mid-battery and every later row
  goes unmeasured. A planted division is spelled ``tl.math.div_rn`` (``div.rn.f32``
  under ``keep``, ``div.rn.ftz.f32`` under ``flush``).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
KERNEL_DIR = API_ROOT / "meep_gpu" / "triton_kernels"

#: (kernel file, function name) of every shipped kernel the two gates mutate, keyed by
#: the gate's ``kernels`` map: both gates mutate the real scan; each its own curl.
SHIPPED: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("real", "scan"): ("cylindrical_triton.py", "cyl_radial_difference"),
    ("real", "curl"): ("cylindrical_triton.py", "cyl_pml_curl_step"),
    ("complex", "scan"): ("cylindrical_triton.py", "cyl_radial_difference"),
    ("complex", "curl"): ("cylindrical_complex.py", "cyl_complex_pml_curl_step"),
}

#: The spelling the 1.0 pre-certification dry run found under ``flush``: a plain ``/``
#: in place of the ``rinv`` multiply, refused at compile by the PTX audit.
PLAIN_SLASH_DIVIDE = ("tl.load(rinv + (g - 1)) * (here - prev)",
                      "(here - prev) / tl.load(w + g)")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def real_gate():
    return _load("gate_triton_cylindrical")


@pytest.fixture(scope="module")
def complex_gate():
    _load("gate_triton_cylindrical")
    return _load("gate_triton_cylindrical_complex")


def shipped_source(family: str, which: str) -> str:
    """One shipped ``@triton.jit`` function's text, decorator included, dedented.

    Read off the file by its AST rather than through ``inspect``: the kernels are
    defined only where Triton imports, and this check needs none.
    """
    filename, function = SHIPPED[(family, which)]
    text = (KERNEL_DIR / filename).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.FunctionDef) and n.name == function)
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return textwrap.dedent("\n".join(text.splitlines()[first - 1:node.end_lineno]))


def true_divisions(source: str) -> List[str]:
    """Every float ``/`` in ``source`` (``a / b`` or ``a /= b``); ``//`` is not one."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            found.append(ast.unparse(node))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Div):
            found.append(ast.unparse(node))
    return found


def calls_named(source: str, dotted: str) -> int:
    return sum(1 for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Call) and ast.unparse(node.func) == dotted)


def _table(gate, family: str):
    return gate.REAL_MUTATIONS if family == "real" else gate.COMPLEX_MUTATIONS


def _rows():
    real = _load("gate_triton_cylindrical")
    cplx = _load("gate_triton_cylindrical_complex")
    return ([("real", row[0]) for row in real.REAL_MUTATIONS]
            + [("complex", row[0]) for row in cplx.COMPLEX_MUTATIONS])


ROWS = _rows()


def _mutant(real_gate, complex_gate, family: str, name: str) -> Tuple[str, str]:
    gate = real_gate if family == "real" else complex_gate
    row = next(r for r in _table(gate, family) if r[0] == name)
    _, which, edits, _expected, _seedings = row
    return which, real_gate.apply_edits(shipped_source(family, which), edits)


@pytest.mark.parametrize("family,name", ROWS, ids=[f"{f}-{n}" for f, n in ROWS])
def test_every_mutation_arms_on_the_shipped_text_and_the_copy_parses(
        real_gate, complex_gate, family, name):
    which, mutated = _mutant(real_gate, complex_gate, family, name)
    assert mutated != shipped_source(family, which), f"{name} changed nothing"
    ast.parse(mutated)


@pytest.mark.parametrize("family,name", ROWS, ids=[f"{f}-{n}" for f, n in ROWS])
def test_no_mutant_spells_a_plain_float_division(real_gate, complex_gate, family, name):
    """Under ``flush`` a plain ``/`` is refused at compile; the row would never measure."""
    _which, mutated = _mutant(real_gate, complex_gate, family, name)
    assert true_divisions(mutated) == [], name


@pytest.mark.parametrize("key", sorted(SHIPPED), ids=["-".join(k) for k in sorted(SHIPPED)])
def test_no_shipped_cylindrical_kernel_spells_a_plain_float_division(key):
    """The shipped scan multiplies by ``rinv``, as MEEP does; no kernel divides."""
    assert true_divisions(shipped_source(*key)) == []


def test_s2_plants_one_round_to_nearest_division_where_the_scan_multiplies(real_gate,
                                                                          complex_gate):
    """``s2_divide_not_multiply`` is still the divide-for-multiply defect, spelled so it
    compiles under both policies: one ``tl.math.div_rn`` and no ``rinv`` load left."""
    shipped = shipped_source("real", "scan")
    _which, mutated = _mutant(real_gate, complex_gate, "real", "s2_divide_not_multiply")
    assert calls_named(shipped, "tl.math.div_rn") == 0
    assert calls_named(mutated, "tl.math.div_rn") == 1
    assert "rinv" in shipped.split('"""')[-1]
    assert "rinv +" not in mutated.split('"""')[-1]


def test_the_division_check_bites_on_the_plain_slash_spelling(real_gate):
    """Positive control: the spelling the dry run refused reads one division here."""
    planted = real_gate.apply_edits(shipped_source("real", "scan"), (PLAIN_SLASH_DIVIDE,))
    assert true_divisions(planted) == ["(here - prev) / tl.load(w + g)"]
    assert true_divisions("q = a // b\nq /= 2.0\n") == ["q /= 2.0"]
