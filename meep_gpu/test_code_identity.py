"""What the weld's comment-only escape hatch may and may not release.

The hatch exists because a docstring fix cost 76 welds (see ``code_identity`` module
docstring). It is dangerous in exactly one direction: if it ever admits an edit that
reaches execution, every weld it releases becomes a claim nobody measured. So the tests
below are weighted accordingly -- one case confirming it admits comments and docstrings,
and a battery confirming it refuses everything else, INCLUDING the edit that motivated it.
"""

from __future__ import annotations

import ast
import sys

import pytest

from .code_identity import canonical_dump, code_digest

BASE = '''"""Module docstring."""
SWITCH = "TRIDENT_FDTD_TRITON"   # the enable

def plan(a, b, *, mode="fused"):
    """Docstring."""
    total = a + b
    return total if mode == "fused" else a
'''

#: Edits that CANNOT reach execution. Only these may keep a weld alive.
#:
#: EVERY ONE CHANGES THE LINE COUNT, deliberately. The first draft of this battery used
#: length-preserving replacements and passed against an implementation that hashed node
#: attributes -- so it certified a hatch that would have rejected essentially every real
#: docstring edit, because real ones add or remove lines. A prose-edit battery whose cases
#: keep the line count is not testing prose editing.
INERT = {
    "docstring expanded over several lines": (
        '"""Module docstring."""',
        '"""Module docstring.\n\nWith a much longer explanation\nspanning several lines.\n"""'),
    "docstring deleted entirely": ('    """Docstring."""\n', ''),
    "comment reworded and doubled": ("# the enable", "# the three-valued dispatch enable\n# with a second line"),
    "comment deleted": ("   # the enable", ""),
    "blank lines inserted": ("SWITCH =", "\n\nSWITCH ="),
    # THE HATCH REASONS ABOUT VALUES, NOT TEXT, and this is the clearest demonstration.
    # Python folds adjacent string literals at parse time, so `"AB"` and `"A" "B"` are one
    # Constant with the same value. Splitting a long literal across lines to fit a margin
    # therefore cannot change what executes -- and is admitted. This case was written as a
    # refusal first; the implementation was right and the expectation was wrong.
    "literal split by implicit concatenation": (
        '"TRIDENT_FDTD_TRITON"', '"TRIDENT_FDTD" "_TRITON"'),
}

#: Edits that DO reach execution. Every one must break the weld. This is the direction
#: that matters: a false "unchanged" here releases a weld describing bytes nobody ran.
LIVE = {
    "string literal renamed": ('"TRIDENT_FDTD_TRITON"', '"MEEP_GPU_DISPATCH"'),
    "default argument changed": ('mode="fused"', 'mode="array"'),
    "operator flipped": ("a + b", "a - b"),
    "comparison flipped": ('mode == "fused"', 'mode != "fused"'),
    "identity vs equality": ('mode == "fused"', 'mode is "fused"'),
    "parameters reordered": ("def plan(a, b, *,", "def plan(b, a, *,"),
    "local renamed": ("total = a + b\n    return total", "sum_ = a + b\n    return sum_"),
    "branch result swapped": ("else a", "else b"),
    "keyword-only made positional": ("def plan(a, b, *, mode=", "def plan(a, b, mode="),
    "decorator added": ("def plan(", "@staticmethod\ndef plan("),
    "type annotation added": ("def plan(a, b", "def plan(a: int, b"),
    "return annotation added": ('mode="fused"):', 'mode="fused") -> int:'),
    "statement order swapped": ("total = a + b\n    return", "total = b + a\n    return"),
    "a pass added": ("    total = a + b", "    pass\n    total = a + b"),
    "augmented assignment": ("total = a + b", "total = a\n    total += b"),
    "f-string content changed": ('SWITCH = "TRIDENT_FDTD_TRITON"', 'SWITCH = f"{a}TRIDENT_FDTD_TRITON"'),
}

@pytest.mark.parametrize("name", sorted(INERT))
def test_comments_and_docstrings_do_not_change_the_code_identity(name):
    """The hatch's whole purpose: prose may be corrected without a device run."""
    old, new = INERT[name]
    assert old in BASE, name
    assert code_digest(BASE.replace(old, new)) == code_digest(BASE), name


@pytest.mark.parametrize("name", sorted(LIVE))
def test_anything_that_reaches_execution_breaks_the_code_identity(name):
    """The direction that matters. A false 'unchanged' here releases an unmeasured weld."""
    old, new = LIVE[name]
    assert old in BASE, name
    assert code_digest(BASE.replace(old, new)) != code_digest(BASE), (
        f"{name} changed what executes and the code identity did not move; a weld "
        f"released on this basis would describe bytes nobody ran")


def test_the_switch_rename_that_motivated_the_hatch_is_NOT_released_by_it():
    """The 2026-08-22 environment-switch rename must still cost its gate re-runs.

    Renaming ``*_FDTD_*`` to ``MEEP_GPU_*`` changes which variable the package
    reads, which is observable behaviour. The hatch was built during that change and
    deliberately does not license it -- only the five prose mentions in
    ``driver.py``/``launch.py``/``folded_complex.py`` would have qualified. If this ever
    passes, the hatch has been widened into an honour system.
    """
    renamed = BASE.replace('"TRIDENT_FDTD_TRITON"', '"MEEP_GPU_DISPATCH"')
    assert code_digest(renamed) != code_digest(BASE)


def test_an_unparsable_file_has_no_code_identity():
    """Degrading to a byte hash would make the clause quietly stricter, not safer."""
    with pytest.raises(SyntaxError):
        code_digest("def broken(:\n")


def test_a_bare_string_that_is_not_a_docstring_is_kept():
    """Only the FIRST statement of a module/class/function is a docstring.

    A bare string elsewhere is an expression statement; stripping it would let a real
    edit hide behind the hatch.
    """
    a = 'def f():\n    x = 1\n    "not a docstring"\n    return x\n'
    b = 'def f():\n    x = 1\n    "a DIFFERENT string"\n    return x\n'
    assert code_digest(a) != code_digest(b)


#: Python 3.12's ``ast.dump`` of one function and one class, the grammar every recorded
#: digest was minted in. ``returns=None`` is left out (an optional field) while
#: ``Constant(value=None)`` is kept (a required one), and ``type_params=[]`` appears on
#: both definitions.
DUMP_312 = (
    "Module(body=[FunctionDef(name='f', args=arguments(posonlyargs=[], "
    "args=[arg(arg='x')], kwonlyargs=[arg(arg='y')], "
    "kw_defaults=[Constant(value=None)], defaults=[]), "
    "body=[Return(value=Name(id='x', ctx=Load()))], decorator_list=[], "
    "type_params=[]), ClassDef(name='C', bases=[], keywords=[], "
    "body=[Assign(targets=[Name(id='value', ctx=Store())], "
    "value=Constant(value=None))], decorator_list=[], type_params=[])], "
    "type_ignores=[])")


def test_the_dump_is_python_312s_on_every_interpreter():
    """A recorded digest is a fact about the source, not about the Python that hashed it.

    ``ast.dump`` prints a different text for the same tree on 3.10, 3.12 and 3.13, and
    measured under 3.10 all 78 Triton device digests failed to recompute on an
    unchanged tree. On 3.12 the canonical dump must also BE ``ast.dump``, or the
    recorded digests would stop recomputing there instead.
    """
    tree = ast.parse("def f(x, *, y=None):\n    return x\n\n"
                     "class C:\n    value = None\n")
    assert canonical_dump(tree) == DUMP_312
    if sys.version_info[:2] == (3, 12):
        assert ast.dump(tree, annotate_fields=True, include_attributes=False) == DUMP_312
