"""A Metal kernel gate's arm table: which gates close it, and what closing it may not do.

Production and every whole-table reader see the complete table. A gate that closes it
holds only the families it imports, so its import record is its own dependency set and
an edit to an unrelated family owes it nothing. A gate closes only where its record
reads the same under both tables, so the list below is measured, not chosen.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[2]
GATES = API_ROOT / "parity" / "meep_gpu"
PACKAGE = API_ROOT / "meep_gpu"
sys.path.insert(0, str(GATES))

import import_closure  # noqa: E402

CLOSED = frozenset((
    "complex_fused_hd_pair", "complex_no_pml_offdiag_update_e", "cylindrical_complex",
    "cylindrical_fused_electric_pair", "cylindrical_real",
    "cylindrical_real_fused_electric_pair", "folded_offdiag_dispersive",
    "fused_electric_pair", "fused_magnetic_pair", "no_pml_curl",
    "no_pml_fused_electric_pairs", "offdiag", "pml",
    "residue_fused_pairs", "tranche7_fused_pairs",
))

#: These read the complete table on purpose: co-admission, a neighbour's refusal, the
#: arbitration or the composition census, so their records differ under a closed table.
#: ``offdiag_stencil_welds`` reads the same under both, but its weld pins the registry, and
#: the rebind skips a weld whose run does not record a pin. ``fused_hd_pair`` and
#: ``cylindrical_real_fused_hd_pair`` start lift children, which never run ``main`` and so
#: read the complete table: closing the parent would drop those kernels from its record.
WHOLE = frozenset((
    "beta_complex_fused_hd_pair", "beta_real_fused_hd_pair", "bfast",
    "bfast_fused_hd_pair", "complex", "conductive_fused_hd_pair",
    "cylindrical_real_fused_hd_pair", "folded_beta",
    "folded_beta_complex_fused_magnetic_pair", "folded_complex_fused_hd_pair",
    "folded_fused_hd_pair", "fused_hd_pair", "fused_hd_pair_nonlinear",
    "no_pml_constitutive", "offdiag_stencil_welds", "special_kz", "symmetry", "whole_step",
))

REGISTRY = "meep_gpu/metal_kernels/registry.py"


def _calls_close_table(path: Path) -> bool:
    return any(isinstance(node, ast.Call)
               and "close_table" in (getattr(node.func, "attr", None),
                                     getattr(node.func, "id", None))
               for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))))


def _run(script: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(API_ROOT), str(GATES))))
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=API_ROOT,
                          capture_output=True, text=True, check=False, env=environment)


def test_the_gates_that_close_are_exactly_the_measured_list():
    found = {path.stem[len("gate_metal_"):] for path in GATES.glob("gate_metal_*.py")
             if _calls_close_table(path)}
    assert found == CLOSED
    assert not CLOSED & WHOLE


def test_a_closed_gate_starts_no_child_process():
    # A child never runs the gate's main, so it reads the complete table and the kernels it
    # launches stay out of the parent's record.
    for name in CLOSED:
        source = (GATES / f"gate_metal_{name}.py").read_text(encoding="utf-8")
        assert "sys.executable" not in source and "subprocess" not in source, name


def test_every_gate_that_reads_the_table_is_listed():
    resolver = import_closure.Resolver(API_ROOT)
    reads = {path.stem[len("gate_metal_"):] for path in GATES.glob("gate_metal_*.py")
             if REGISTRY in resolver.run_closure(f"parity/meep_gpu/{path.name}")}
    # Its static closure reaches the registry through a leg its run does not take.
    assert reads - STATIC_UPPER_BOUND - WHOLE == {"complex_conductive_fused_pair"}
    assert WHOLE <= reads


def test_nothing_in_the_package_closes_the_table():
    # A closed table in a run would change a selection with no ambiguity detected.
    callers = [path.relative_to(API_ROOT).as_posix() for path in PACKAGE.rglob("*.py")
               if _calls_close_table(path)]
    assert callers == []


#: Their legs reach the lift and census modules, whose whole-table reads the static
#: closure cannot rule out; the import record of a real run is the authority for these.
STATIC_UPPER_BOUND = frozenset(("complex_fused_hd_pair", "fused_magnetic_pair"))


@pytest.mark.parametrize("name", sorted(CLOSED - STATIC_UPPER_BOUND))
def test_a_closed_gate_plans_without_the_registry(name):
    closure = import_closure.Resolver(API_ROOT).run_closure(
        f"parity/meep_gpu/gate_metal_{name}.py")
    assert REGISTRY not in closure


def test_the_closure_rule_reads_the_gate_and_not_a_name(tmp_path):
    # The planted control: the same gate text without the call reaches the registry.
    source = (GATES / "gate_metal_pml.py").read_text(encoding="utf-8")
    assert "close_table()" in source
    root = tmp_path / "tree"
    (root / "parity" / "meep_gpu").mkdir(parents=True)
    for rel in ("meep_gpu",):
        (root / rel).symlink_to(API_ROOT / rel)
    for path in GATES.glob("*.py"):
        if path.name != "gate_metal_pml.py":
            (root / "parity" / "meep_gpu" / path.name).symlink_to(path)
    planted = root / "parity" / "meep_gpu" / "gate_metal_pml.py"
    planted.write_text(source.replace("    close_table()\n", ""),
                       encoding="utf-8")
    closure = import_closure.Resolver(root).run_closure("parity/meep_gpu/gate_metal_pml.py")
    assert REGISTRY in closure


_UNARMED_SLOT = """
    import numpy as np
    from meep_gpu.metal_kernels import arms, launch
    {close}
    arms.arms_for = lambda slot, context: ()
    launch.chunk_division = lambda *args: (None, None, ())
    try:
        plan = launch.plan_step(None, None)
    except KeyError as exc:
        print("KeyError", exc)
    else:
        print(plan.reasons["fill_B"][0])
"""


def test_a_closed_table_refuses_an_unarmed_slot_by_name_and_loads_no_family():
    result = _run("""
        import sys
        from metal_closed_table import close_table
        from meep_gpu.metal_kernels import arms
        close_table()
        arms.ensure_registered()
        assert "meep_gpu.metal_kernels.registry" not in sys.modules
        assert arms.registered("fill_B") == ()
        print("ok")
    """)
    assert result.stdout.strip() == "ok", result.stderr


def test_the_table_cannot_close_once_it_has_been_read():
    result = _run("""
        from metal_closed_table import close_table
        from meep_gpu.metal_kernels import arms
        arms.ensure_registered()
        try:
            close_table()
        except RuntimeError as exc:
            print(exc)
    """)
    assert "cannot close" in result.stdout, result.stderr


def test_an_unarmed_slot_is_a_defect_on_the_complete_table_and_a_refusal_on_a_closed_one():
    whole = _run(_UNARMED_SLOT.format(close="arms.ensure_registered()"))
    assert "KeyError" in whole.stdout, whole.stderr
    closed = _run(_UNARMED_SLOT.format(
        close="from metal_closed_table import close_table; close_table()"))
    assert "fill_B: no family in this gate's closed table" in closed.stdout, closed.stderr


def test_the_package_has_no_closed_mode():
    # The closure is the gate's: the bytes a run executes do not carry it.
    for name in ("arms.py", "launch.py"):
        source = (PACKAGE / "metal_kernels" / name).read_text(encoding="utf-8")
        assert "close_table" not in source and '"closed"' not in source
