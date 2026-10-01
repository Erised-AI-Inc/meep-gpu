"""Tests for the COMPLEX cylindrical (Dcyl, |m| >= 1) tranche — laptop only.

Everything here runs with no GPU, no CuPy and no Triton. What needs hardware —
byte identity of ``cyl_complex_pml_curl_step`` against ``stepping.step_B``/
``step_D``, the kernel-source mutation legs, the fusion guard control — lives in
``parity/meep_gpu/gate_triton_cylindrical_complex.py`` and **has not been run**:
no device job was submitted for this family, so no byte-identity claim exists for
the kernel. The one identity claim that carries out of the laptop is the gate's
reference transcription equalling the shipped array path, and a slice of that leg
runs here (:func:`test_the_gate_reference_transcription_equals_the_array_path`)
so the merge bar carries it rather than only an artifact.

What is pinned HERE is everything that decides whether the kernel is ever ALLOWED
to run, plus the host-rounded numbers it is handed, which would be wrong SILENTLY
rather than loudly:

* **the two predicates**, clause by clause, each shown load-bearing by breaking
  exactly one thing on an otherwise admitted real ``Fields``/``PML``/``Grid``;
* **disjointness with ``cylindrical_triton`` in both directions** — that module
  refuses ``m != 0`` by name and this one refuses ``m == 0`` by name, which is
  what stops two products that compute different arithmetic from both admitting
  a run once either is wired in;
* **the host-rounded constants**, byte-compared against ``stepping.py`` itself
  rather than transcribed twice: the i*m/r coefficient row (including that its
  real word is ``+0.0`` and not ``-0.0``), the |m| = 1 axis-increment scalars
  (whose real word IS a signed zero, the opposite laundering), and the radial
  prefix on both sides including the B-side zero wall row;
* **the tables** — i*m/r call sites, signs and partner registers, the axis
  increment's target, the prefix component and ``ir0`` — read back off
  ``stepping``'s own curl-term tables and source;
* **the plan shape**, built through the gate's own ``from_arrays`` route on NumPy
  arrays, since ``CupyPointer`` defers the address until launch;
* **the module staying unwired and importing no sibling kernel module**;
* **the optional dependency staying optional.**
"""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import math
import pathlib
import sys
import types

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import IYEE_SHIFTS, Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import complex_fields as complex_module
from meep_gpu.triton_kernels import cylindrical_complex as module
from meep_gpu.triton_kernels import cylindrical_triton as real_cylindrical
from meep_gpu.test_triton_complex_fields import stamp_probe_record

# Every probe record this module builds is stamped 'keep'; the complex
# families' policy clause fails closed when the run policy can be neither
# read nor declared, so the premise is declared rather than left implicit.
# See ``run_policy_declared_keep`` in conftest.py.
pytestmark = pytest.mark.usefixtures("run_policy_declared_keep")


PACKAGE_DIR = pathlib.Path(module.__file__).parent
API_DIR = PACKAGE_DIR.parent.parent
GATE_PATH = API_DIR / "parity" / "meep_gpu" / "gate_triton_cylindrical_complex.py"
COMPOSITION_PATH = (API_DIR / "parity" / "meep_gpu"
                    / "probe_triton_cylindrical_complex_composition.py")


# ---------------------------------------------------------------------------
# The step budget, stated rather than implied
# ---------------------------------------------------------------------------
#
# THREE SEPARATE NUMBERS. Conflating them is how a family claims more than it
# measured, so they are named apart and two of the three are deliberately absent.
#
# 1. the gate's NumPy REFERENCE transcription is bytewise identical to
#    ``stepping.py`` for 8 sub-steps per row over 544 rows (8 (m, accurate)
#    classes x 4 shapes x 5 Courant numbers, four of them non-power-of-two, x
#    both z terminations x both sub-steps, minus 24 (m, Courant) pairs above the
#    accurate branch's stability bound, which are RECORDED as skips), plus
#    480/480 rows of the constitutive identity leg; measured on this laptop,
#    ``--self-check``;
# 2. the KERNEL has never been compiled, launched or compared. There is no
#    device budget and no byte-identity claim;
# 3. when the device leg does run, the INHERITED exposure to watch is
#    ``cylindrical_triton``'s: its consecutive-step leg first diverged at step
#    23-72 on a subnormal against CuPy's flushed zero. This family shares that
#    module's array-path prefix and the same split-field recurrence.
REFERENCE_SUB_STEPS_PROVEN_ON_NUMPY = 8
REFERENCE_ROWS_PROVEN_ON_NUMPY = 624       # 544 before the m = 0 arm (2026-09-04)
CONSTITUTIVE_IDENTITY_ROWS = 560           # 480 before the m = 0 arm (2026-09-04)
KERNEL_CERTIFIED_CONSECUTIVE_STEPS = None      # no device run has been taken
INHERITED_FIRST_DIVERGENCE_STEP_TO_WATCH = 23  # cylindrical_triton's, not ours


#: r-high plus both z faces, in CELLS — what a lifted Dcyl script actually asks
#: for. There is no absorber at r = 0 and none on phi, which has one cell.
DCYL_PML_CELLS = {"x": (0, 4), "z": 4}

#: A probe record that classifies every pattern the same way. The predicates
#: refuse an absent or disagreeing one, so every admitted case has to pass one in
#: — which is itself a clause under test.
PROBE = stamp_probe_record(
    {"backend": "cupy",
     "patterns": {name: "FMA_V1" for name in complex_module.PROBE_PATTERNS}})


class _Missing:
    """Sentinel: an :class:`Override` key bound to this makes the attribute absent."""


MISSING = _Missing()


class Override:
    """A real engine object with named attributes replaced or removed.

    Used instead of a hand-built stub so every refusal test starts from a
    configuration the array path itself accepts and breaks exactly ONE thing. A
    stub that happens to omit an attribute proves nothing about which clause
    fired; this proves the clause is the only difference.
    """

    def __init__(self, wrapped, **overrides):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            value = overrides[name]
            if value is MISSING:
                raise AttributeError(name)
            return value
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Override({object.__getattribute__(self, '_wrapped')!r})"


def build(shape=(16, 1, 20), m=-1, accurate=False, courant=0.37, z_metallic=True,
          pml_cells=None, force_complex_fields=True):
    """A real complex Dcyl ``Grid``/``Fields``/``PML``, built as the engine builds one.

    ``resolution = 1`` with a cell size equal to the cell count lands on ``shape``
    exactly (``meep_cell_count`` is ``int(size*a + 0.5)``). z is declared
    explicitly because ``Grid``'s own default is periodic and BOTH terminations
    are in the sixteen corpus rows.
    """
    grid = Grid(resolution=1.0,
                cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=int(m),
                boundaries=({"z": "metallic"} if z_metallic else None),
                accurate_fields_near_cylorigin=bool(accurate),
                courant=float(courant))
    assert tuple(grid.shape) == tuple(shape), (tuple(grid.shape), tuple(shape))
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.enable_pml_storage()
    cells = DCYL_PML_CELLS if pml_cells is None else pml_cells
    return fields, PML(grid=grid, thickness=cells)


def cartesian():
    """A real Cartesian complex grid — the coordinate-system refusal's subject."""
    grid = Grid(resolution=10.0, cell_size=(0.8, 0.8, 0.8))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=2)


def curl_reasons(fields, pml, sub_step="step_B", probe=PROBE):
    """Curl-predicate reasons, minus the CuPy-backend one every laptop run carries."""
    verdict = module.cylindrical_complex_curl_coverage(fields, pml, sub_step, probe=probe)
    return [r for r in verdict.reasons if "not cupy" not in r]


def constitutive_reasons(fields, pml, side="E", probe=PROBE):
    verdict = module.cylindrical_complex_constitutive_coverage(fields, pml, side,
                                                              probe=probe)
    return [r for r in verdict.reasons if "not cupy" not in r]


def code_of(path: pathlib.Path) -> str:
    """A module's source with comments and docstrings removed.

    Source-level assertions have to read EXECUTABLE text: this module's prose
    names the files integration will eventually edit, and a substring check on
    the raw file would fire on the prose rather than on an import.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


def imported_names(path: pathlib.Path) -> set:
    """Every name the module imports, as an import GRAPH rather than a substring."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def words(array) -> np.ndarray:
    """The uint32 word view. Byte compares only — never ``allclose``."""
    return np.ascontiguousarray(np.asarray(array)).view(np.uint32).ravel()


def same_bytes(left, right) -> bool:
    a, b = words(left), words(right)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def constexpr_value(value):
    """Unwrap a ``tl.constexpr``, or pass a plain int through.

    On this laptop Triton is absent and the module's stand-in returns the raw
    value, so a bare ``==`` would pass here and compare ``constexpr`` objects on a
    Triton host. Unwrapping keeps the assertion meaning the same thing on both.
    """
    return getattr(value, "value", value)


def is_power_of_two(value: float) -> bool:
    """True when the float's mantissa is exactly 0.5 — i.e. it is 2**k.

    Why it is worth a helper: an exact binary Courant makes the FMA and
    associativity discrepancies vanish, so a sweep carrying only powers of two
    certifies broken kernels. This is what the sweep test checks against.
    """
    return math.frexp(float(value))[0] == 0.5


def kernel_function_source(name: str) -> str:
    """One module-level function's SOURCE TEXT, by AST segment.

    Read from the file rather than through ``inspect``: with Triton absent the
    jitted kernel is an ``_UnavailableKernel`` with no ``.fn``, and the
    source-text pins below are exactly the ones that must hold on the laptop that
    is the merge bar.
    """
    path = PACKAGE_DIR / "cylindrical_complex.py"
    text = path.read_text(encoding="utf-8")
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"{name} is not a module-level function of {path}")


def gate_module():
    """Import the gate by path — ``parity/`` is not a package on ``sys.path``."""
    spec = importlib.util.spec_from_file_location(
        "_gate_triton_cylindrical_complex", GATE_PATH)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


# ---------------------------------------------------------------------------
# The optional dependency must stay optional
# ---------------------------------------------------------------------------

@pytest.fixture
def restored_module():
    """Reload the module cleanly AFTER a test that reloaded it with Triton blocked.

    ``importlib.reload`` mutates the module object every other test holds, so an
    absence test without this leaves the kernel as an ``_UnavailableKernel`` for
    the rest of the session — which on a Triton host turns the source tests into
    a failure far from its cause. Requested FIRST in the signature so it is set
    up first and therefore torn down LAST, after ``monkeypatch`` has put
    ``__import__`` back.
    """
    yield
    importlib.reload(importlib.import_module(
        "meep_gpu.triton_kernels.cylindrical_complex"))


def test_the_module_imports_and_answers_coverage_with_triton_absent(restored_module,
                                                                   monkeypatch):
    """A missing optional dependency must not break the engine — or this module.

    ``@triton.jit`` runs at IMPORT time, so the predicate and the plan builders
    have to be importable and answerable on the machine that is the merge bar.
    This laptop has no Triton at all, which is why the fallback is the ordinary
    path here rather than an exotic one — but the test blocks the import outright
    so it still means something on a Triton host.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is blocked for this test")
        return real_import(name, *args, **kwargs)

    for name in [n for n in sys.modules if n.startswith("triton")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    reloaded = importlib.reload(importlib.import_module(
        "meep_gpu.triton_kernels.cylindrical_complex"))

    fields, pml = build()
    verdict = reloaded.cylindrical_complex_curl_coverage(fields, pml, "step_B",
                                                        probe=PROBE)
    assert verdict.covered is False                 # NumPy backend, and it says so
    assert any("not cupy" in reason for reason in verdict.reasons)
    assert reloaded.plan_cylindrical_complex_curl(fields, pml, "step_B",
                                                  probe=PROBE) is None
    assert reloaded.plan_cylindrical_complex_constitutive(fields, pml, "E",
                                                          probe=PROBE) is None
    # The host-side tables still answer: they are pure Python and Triton-free.
    assert reloaded.m_class(3) == 2
    assert reloaded.zero_rows(3, False) == 3


def test_the_kernel_refuses_by_raising_rather_than_being_silently_none(restored_module,
                                                                      monkeypatch):
    """Without Triton the kernel object must RAISE when launched, not be ``None``.

    ``None`` would let a caller that forgot to check coverage fail with an
    ``AttributeError`` from somewhere unrelated; ``_UnavailableKernel`` names the
    missing dependency at the launch site. Certified by the complex tranche and
    reused here rather than re-invented, which is what this pins.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is blocked for this test")
        return real_import(name, *args, **kwargs)

    for name in [n for n in sys.modules if n.startswith("triton")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    reloaded = importlib.reload(importlib.import_module(
        "meep_gpu.triton_kernels.cylindrical_complex"))

    kernel = reloaded.cyl_complex_pml_curl_step
    assert kernel is not None
    assert isinstance(kernel, complex_module._UnavailableKernel)
    with pytest.raises(ImportError):
        kernel[(1,)](1, 2, 3)


# ---------------------------------------------------------------------------
# Not wired, and importing no sibling kernel module
# ---------------------------------------------------------------------------

def test_the_module_is_wired_into_the_planner_but_not_into_dispatch():
    """THE CLAIM CHANGED IN ONE PLACE AND MUST NOT HAVE CHANGED IN THE OTHERS.

    This family is now CONSULTED by ``launch.plan_step`` behind the
    ``has_cylindrical and has_complex`` gate — that is the round that wired it,
    and asserting its absence from ``launch.py`` would now be asserting that the
    wiring did not happen. So the planner leg is inverted: the module must be
    named there, and named through a LAZY forwarder.

    Everything else is unchanged and is what "dispatch stays disabled" now means:
    ``fastpath.py`` and ``backends.py`` may not mention this module at all, and
    ``fingerprints.json`` may not carry a weld for it — no
    ``specialized_kernel_sources`` entry and no ``host_sha256`` entry — because
    no device gate has certified its bytes into that record.

    Checked on the shipped files rather than on intent, because "we did not wire
    it into a launch" is exactly the kind of claim that decays silently.
    """
    init_code = code_of(PACKAGE_DIR / "__init__.py")
    assert "cylindrical_complex" in init_code, \
        "the package no longer exports this family's planner entry points"

    launch_code = code_of(PACKAGE_DIR / "launch.py")
    assert "cylindrical_complex" in launch_code, \
        "launch.py no longer consults this family; the planner wiring regressed"
    tree = ast.parse((PACKAGE_DIR / "launch.py").read_text(encoding="utf-8"))
    module_scope = {node.module for node in tree.body
                    if isinstance(node, ast.ImportFrom) and node.module}
    assert "cylindrical_complex" not in module_scope, (
        "launch.py imports this module at module scope; that closes an import "
        "cycle (this file imports FROM launch at its own module scope) and it "
        "makes a bare package import pull Triton-optional code")

    for owned in ("fastpath.py", "backends.py"):
        path = PACKAGE_DIR / owned
        if not path.exists():
            path = PACKAGE_DIR.parent / owned
        # Not "if it exists": a moved or renamed dispatch file must FAIL here
        # rather than quietly skip, or this whole check goes vacuous exactly when
        # the wiring is being changed.
        assert path.exists(), f"{owned} not found; the dispatch check is now blind"
        # THE DISPATCH CLAIM MOVED, and only here. ``fastpath.py`` now dispatches
        # the track, so it NAMES this family — its run artifact reports which
        # certification each dispatched arm rides on, and that is a lookup table
        # of strings. What it must still not do is IMPORT this module: a
        # dispatcher that reached a plan any way other than through
        # ``plan_step`` would be selecting a numerical product outside the
        # composer's fail-closed arm table, which is the whole hazard.
        assert not any("cylindrical_complex" in name
                       for name in imported_names(path)), \
            f"{owned} imports cylindrical_complex; dispatch must go through plan_step"
        if owned == "backends.py":
            assert "cylindrical_complex" not in code_of(path), \
                "backends.py references cylindrical_complex"
    from meep_gpu.fastpath import ARM_CERTIFICATION
    assert "cylindrical_complex" in {family for family, _ in ARM_CERTIFICATION.values()}, \
        ("the dispatcher no longer records which gate certified this family; a "
         "dispatched arm with no named provenance is what the artifact exists to "
         "prevent")

    fingerprints = json.loads(
        (PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    assert "cylindrical_complex.py" not in fingerprints.get(
        "specialized_kernel_sources", {}), \
        "fingerprints.json carries a weld for a module with no device gate result"
    assert "cylindrical_complex.py" not in fingerprints.get("host_sha256", {}), \
        "fingerprints.json welds a host digest this family never gated"


def test_the_module_imports_no_sibling_kernel_module():
    """Module independence, enforced rather than agreed.

    Each kernel family is certified on its own. This module defines its own kernel,
    its own predicates and its own plans, and imports shared helpers READ-ONLY. The
    sibling kernel modules are certified separately — above all
    ``folded_complex`` (a NaN-policy re-run) — and must not be imported at all.
    """
    path = PACKAGE_DIR / "cylindrical_complex.py"
    code = code_of(path)
    for banned in ("fastpath", "cuda_kernels", "backends"):
        assert banned not in code, f"cylindrical_complex.py references {banned}"

    imports = imported_names(path)
    for banned in ("folded_complex", ".folded_complex", "special_kz", ".special_kz",
                   "bfast_curl", ".bfast_curl", "nonlinear_update_e",
                   ".nonlinear_update_e", "offdiag_update_e", ".offdiag_update_e",
                   "cylindrical_triton", ".cylindrical_triton", "symmetry",
                   ".symmetry", "conductivity", ".conductivity"):
        assert banned not in imports, f"cylindrical_complex.py imports {banned}"

    # What it MAY import, read-only: the certified complex base, the shared
    # coverage helpers and the shared launch interop. Nothing else.
    assert {"ComplexConstitutivePlan", "_mul_coefficient_left", "_word_view",
            "Coverage", "_boundary_kinds", "CupyPointer", "_flat"} <= imports


def test_the_fusion_guard_has_exactly_one_launch_site_spelled_the_house_way():
    """``enable_fp_fusion`` rides as a launch keyword, once, never in an options dict.

    It is not byte-uniform across tranches — measured: complex 68/68 fusion-on
    rows identical, special_kz 24/96, nonlinear 0/108, offdiag 0/28, bfast 0/52 —
    so which configuration a family certifies under is a fact about the family,
    and it can only be one fact if there is one launch site.
    """
    code = code_of(PACKAGE_DIR / "cylindrical_complex.py")
    assert "options=" not in code, "the guard must never ride in an options dict"
    assert code.count("enable_fp_fusion=") == 1
    assert "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)" in code


def test_no_negation_is_spelled_with_a_unary_minus_in_the_kernel():
    """Triton lowers ``-x`` as ``0.0 - x``, which canonicalizes signed zeros.

    Measured REACHABLE at driver level — not a theoretical
    hazard. Every negated addend in a kernel body must therefore be spelled
    ``* -1.0``. Checked on the kernel function's AST so a minus sign in host code
    (where Python's semantics are the array path's own) does not trip it.
    """
    tree = ast.parse((PACKAGE_DIR / "cylindrical_complex.py").read_text(encoding="utf-8"))
    kernels = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name in ("cyl_complex_pml_curl_step",
                              "_mul_general_coefficient_left")]
    assert len(kernels) == 2, "the kernel bodies moved; this check is now blind"
    for kernel in kernels:
        for node in ast.walk(kernel):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                # A negative literal is a constant, not a lowered subtraction.
                assert isinstance(node.operand, ast.Constant), (
                    f"{kernel.name} spells a negation with unary minus at line "
                    f"{node.lineno}; Triton lowers it as 0.0 - x")


# ---------------------------------------------------------------------------
# The tables are stepping's own
# ---------------------------------------------------------------------------

def test_the_imr_call_sites_signs_and_partners_are_steppings_own():
    """Four i*m/r call sites and only four, read back off ``stepping``'s source.

    A fifth site, a dropped one or a flipped sign is a smooth, plausible, wrong
    field. The gate measures all three as live needles (``imr_dropped`` 899712
    words, ``imr_sign_flipped`` 899712, ``imr_partner_swapped`` 449856 at 8
    sub-steps); this pins the same facts where they are cheap to check.
    """
    source = pathlib.Path(stepping.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {"step_B": [], "step_D": []}
    for function in [n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name in found]:
        for node in ast.walk(function):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_cylindrical_imr_term"):
                target = node.args[1].value
                partner = ast.unparse(node.args[2])
                sign = float(ast.literal_eval(node.args[3]))
                found[function.name].append((target, partner, sign))

    assert found["step_B"] == [("Bx", "electric['Ez']", +1.0),
                               ("Bz", "electric['Ex']", -1.0)]
    assert found["step_D"] == [("Dx", "magnetic['Hz']", -1.0),
                               ("Dz", "magnetic['Hx']", +1.0)]

    # And that is exactly what the module compiles in, expressed as (target index,
    # partner REGISTER, sign) — the register naming which centre load of the
    # certified body the term reuses, so no new pointer and no extra traffic.
    for sub_step, terms, sources in (
            ("step_B", stepping.B_CURL_TERMS, ("Ex", "Ey", "Ez")),
            ("step_D", stepping.D_CURL_TERMS, ("Hx", "Hy", "Hz"))):
        targets = tuple(term.target for term in terms)
        expected = []
        for target, partner, sign in found[sub_step]:
            index = targets.index(target)
            partner_name = partner.split("'")[1]
            register = "abc"[sources.index(partner_name)]
            expected.append((index, register, sign))
        assert module.IMR_TERMS[sub_step] == tuple(expected)

    # Target 1 (By / Dy) gets NOTHING on either side.
    for sub_step in ("step_B", "step_D"):
        assert 1 not in [index for index, _register, _sign in module.IMR_TERMS[sub_step]]


def test_the_imr_partners_are_centre_loads_the_certified_body_already_makes():
    """``c`` = g2 = Ez/Hz and ``a`` = g0 = Ex/Hx — no new pointer, no extra traffic.

    The property special_kz's beta term has, and the reason this family reuses
    that structural precedent rather than inventing one. If a partner were ever a
    SHIFTED operand the kernel would need a load the body does not make, and the
    "one new kernel" accounting would be wrong.
    """
    source = pathlib.Path(stepping.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for sub_step, sources in (("step_B", ("Ex", "Ey", "Ez")),
                              ("step_D", ("Hx", "Hy", "Hz"))):
        function = next(n for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) and n.name == sub_step)
        partners = {}
        for node in ast.walk(function):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "_cylindrical_imr_term"):
                partners[node.args[1].value] = ast.unparse(node.args[2]).split("'")[1]
        targets = tuple(term.target for term in
                        (stepping.B_CURL_TERMS if sub_step == "step_B"
                         else stepping.D_CURL_TERMS))
        for index, register, _sign in module.IMR_TERMS[sub_step]:
            assert register in ("a", "c"), "only g0 and g2 are centre loads"
            # The register the module names must BE the source register holding
            # the partner stepping actually passes — not merely some source.
            assert sources["abc".index(register)] == partners[targets[index]], (
                f"{sub_step} {targets[index]}: module says register {register!r} "
                f"(= {sources['abc'.index(register)]}) but stepping passes "
                f"{partners[targets[index]]}")
        # And never register 'b' (g1), whose only uses in the body are SHIFTED —
        # a partner there would need a load the certified body does not make.
        assert "b" not in {r for _i, r, _s in module.IMR_TERMS[sub_step]}


def test_the_axis_increment_targets_are_the_ones_stepping_replaces():
    """Bx on B, Dy on D — and BOTH replace curl row 0, they do not accumulate.

    Read off ``stepping``'s own returns. ``axis_increment_accumulates`` is the
    gate's zero-init-only needle (0 random words, 16 zero-init at 8 sub-steps):
    after the ownership mask the row is exactly ``+0.0`` and ``+0.0 + x == x``
    for every x EXCEPT ``-0.0``, so a random-seeded sweep is provably blind to
    the difference and only a thin negative-coefficient absorber reaches it.
    """
    assert module.AXIS_INCREMENT_TARGET == {"step_B": 0, "step_D": 1}
    fields, pml = build(m=1)
    grid = fields.grid
    boundaries = stepping._boundary_kinds(grid, pml)
    phases = stepping._bloch_phases(grid, boundaries, fields.Ex, pml)
    dtdx = grid.dt / grid.dx

    electric = stepping._component_snapshot(fields, ("Ex", "Ey", "Ez"))
    magnetic = stepping._component_snapshot(fields, ("Hx", "Hy", "Hz"))
    b_target, _ = stepping._cylindrical_axis_increment_B(fields, electric, boundaries,
                                                        phases, dtdx)
    d_target, _ = stepping._cylindrical_axis_increment_D(fields, magnetic, boundaries,
                                                        phases, dtdx)
    assert stepping.B_CURL_TERMS[module.AXIS_INCREMENT_TARGET["step_B"]].target == b_target
    assert stepping.D_CURL_TERMS[module.AXIS_INCREMENT_TARGET["step_D"]].target == d_target

    # The replacement is an ASSIGNMENT in both, never a +=. Read off the source.
    source = pathlib.Path(stepping.__file__).read_text(encoding="utf-8")
    assert source.count("curl[_face(0, 0)] = -axis_increment[1]") == 2
    assert "curl[_face(0, 0)] += " not in source


def test_the_axis_increment_is_absent_at_abs_m_above_one():
    """|m| >= 2 has no axis-row increment at all — the two rules are exclusive."""
    for m in (2, -2, 3, 5):
        fields, pml = build(m=m, courant=0.3)
        grid = fields.grid
        boundaries = stepping._boundary_kinds(grid, pml)
        phases = stepping._bloch_phases(grid, boundaries, fields.Ex, pml)
        dtdx = grid.dt / grid.dx
        electric = stepping._component_snapshot(fields, ("Ex", "Ey", "Ez"))
        magnetic = stepping._component_snapshot(fields, ("Hx", "Hy", "Hz"))
        assert stepping._cylindrical_axis_increment_B(fields, electric, boundaries,
                                                      phases, dtdx) is None
        assert stepping._cylindrical_axis_increment_D(fields, magnetic, boundaries,
                                                      phases, dtdx) is None
        assert module.m_class(m) == 2


def test_the_prefix_spec_is_the_one_step_B_and_step_D_pass():
    """Ep at ir0 = 0.0 WITH a wall row; Hp at ir0 = 0.5 WITHOUT one.

    ``ir0`` is half the component's own radial Yee shift: Ep sits at the node
    (r-shift 0) and Hp half a cell out (r-shift 1). The wall row is the historical
    defect this project already paid for once — without it the last row's forward
    difference becomes MINUS THE WHOLE ACCUMULATED SUM, and the gate's
    ``no_wall_row`` needle is live at 12032 words.
    """
    assert module.PREFIX["step_B"]["component"] == "Ey"      # Ep
    assert module.PREFIX["step_D"]["component"] == "Hy"      # Hp
    assert module.PREFIX["step_B"]["ir0"] == 0.0
    assert module.PREFIX["step_D"]["ir0"] == 0.5
    assert module.PREFIX["step_B"]["extend_wall_row"] is True
    assert module.PREFIX["step_D"]["extend_wall_row"] is False
    assert IYEE_SHIFTS["Ey"][0] == 0 and IYEE_SHIFTS["Hy"][0] == 1
    assert module.PREFIX["step_B"]["ir0"] == 0.5 * IYEE_SHIFTS["Ey"][0]
    assert module.PREFIX["step_D"]["ir0"] == 0.5 * IYEE_SHIFTS["Hy"][0]


@pytest.mark.parametrize("z_metallic,expected_z", [(True, "metallic"),
                                                   (False, "periodic")])
def test_the_boundary_triple_is_what_stepping_resolves_for_a_dcyl_grid(z_metallic,
                                                                      expected_z):
    """``('axis', 'periodic', <z>)`` — measured off a real grid, not assumed.

    BOTH z terminations, because 13 of the sixteen corpus rows are metallic and 3
    are periodic, and they mask different planes AND take different ghost arms.
    """
    fields, pml = build(z_metallic=z_metallic)
    kinds = tuple(stepping._boundary_kinds(fields.grid, pml))
    assert kinds == (module.REQUIRED_R_KIND, module.REQUIRED_PHI_KIND, expected_z)
    assert kinds[2] in module.ADMITTED_Z_KINDS
    assert tuple(fields.grid.is_axis(a) for a in range(3)) == module.CYLINDRICAL_AXIS_FLAGS
    assert fields.grid.shape[1] == 1


def test_the_r_axis_far_ghost_is_the_metallic_zero():
    """Why the kernel compiles ``BCX = METALLIC`` and needs no cylindrical branch.

    ``_shift_up``'s CYL_AXIS branch SHARES the METALLIC one — a hard zero at the
    far r face. The gate measured the consequence end to end (running the whole
    reference with ``boundaries[0]`` forced to ``'metallic'`` is 0 differing words
    over 80 rows); this pins the shift function itself, which is where the fact
    actually lives.
    """
    field = np.arange(5 * 1 * 3, dtype=np.complex64).reshape(5, 1, 3) + 1j
    as_axis = stepping._shift_up(np, field, 0, stepping.CYL_AXIS, None)
    as_metallic = stepping._shift_up(np, field, 0, stepping.METALLIC, None)
    assert same_bytes(as_axis, as_metallic)
    assert not same_bytes(as_axis, stepping._shift_up(np, field, 0,
                                                      stepping.PERIODIC, None))


# ---------------------------------------------------------------------------
# m_class and zero_rows
# ---------------------------------------------------------------------------

def test_m_class_carries_the_m_zero_arm():
    """m = 0 is the THIRD arm since 2026-09-04 (complex storage), not a refusal.

    The disjointness story moved from m to STORAGE: the real m = 0 run is still
    ``cylindrical_triton``'s and is refused here on ``force_complex_fields``
    (see ``test_the_two_cylindrical_products_are_disjoint_by_storage``).
    """
    assert module.m_class(0) == 0 == constexpr_value(module.M_ZERO)
    assert module.zero_rows(0, False) == 0 == module.zero_rows(0, True)


def test_the_m_zero_scalar_is_rounded_the_way_the_array_path_rounds_it():
    """``four_dtdx_scalar`` is the float32 rounding of the float64 ``4.0 * dtdx``
    — the NEP-50 cast stepping :585 makes of a Python float times a complex64
    row — and never the product of an fp32-rounded ``dtdx``."""
    for dtdx in (0.37, 0.3141592653589793, 0.2777777777777778, 0.5):
        assert module.four_dtdx_scalar(dtdx) == float(np.float32(4.0 * dtdx))
        assert np.float32(module.four_dtdx_scalar(dtdx)).tobytes() == \
            (np.complex64(4.0 * dtdx) * np.ones(1, np.complex64))[0].real.tobytes()


@pytest.mark.parametrize("m,expected", [(1, 1), (-1, 1), (2, 2), (-2, 2), (3, 2),
                                        (-5, 2), (7, 2)])
def test_m_class_splits_at_abs_m_two(m, expected):
    """|m| = 1 takes axis-row increments and no zeroing; |m| >= 2 the reverse.

    The two constexpr arms are compared through :func:`constexpr_value` so the
    assertion means the same thing on a Triton host, where ``M_ONE`` is a
    ``tl.constexpr`` object rather than the plain int this laptop sees.
    """
    assert module.m_class(m) == expected
    arm = module.M_ONE if abs(m) == 1 else module.M_MANY
    assert module.m_class(m) == constexpr_value(arm)
    assert constexpr_value(module.M_ONE) == 1
    assert constexpr_value(module.M_MANY) == 2


@pytest.mark.parametrize("m", [1, -1, 2, -2, 3, -3, 5])
@pytest.mark.parametrize("accurate", [False, True])
def test_zero_rows_matches_the_slice_stepping_would_take(m, accurate):
    """``zero_rows`` against ``stepping._cylindrical_axis_rows`` itself.

    The ACCURATE flag selects the row count and may not be assumed — a wrong
    constexpr is a plane of wrong values, not a crash. The gate's
    ``zero_rows_accurate_always`` needle is live at 55268 random / 72 zero-init
    words, which is what makes this cheap check worth having beside it.
    """
    stub = types.SimpleNamespace(m=int(m), accurate_fields_near_cylorigin=accurate)
    rows = stepping._cylindrical_axis_rows(stub)
    expected = 0 if abs(m) < 2 else (rows.stop - (rows.start or 0))
    assert module.zero_rows(m, accurate) == expected
    if abs(m) >= 2:
        assert module.zero_rows(m, True) == 1
        assert module.zero_rows(m, False) == abs(m)


# ---------------------------------------------------------------------------
# The host-rounded constants, byte-compared against stepping itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [1, -1, 2, -3, 5])
@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_imr_coefficient_row_reproduces_steppings_own_term_bytewise(m, sub_step):
    """End-to-end byte identity against ``stepping._cylindrical_imr_term``.

    Compared THROUGH the term rather than against a second transcription of the
    row: a re-transcription proves only that the test and the module agree. A
    random complex partner makes every word of the row observable, so any
    differing coefficient word shows up in the product.

    The row is host-built and bound as a device array for three reasons the
    module names — the ``arange`` is float64, the divide is float64 and rounded
    ONCE, and the numerator's ``(-1j) * X`` spelling launders the real word. The
    consequence is that THERE IS NO DIVISION IN THE KERNEL, so the platform's
    ``div.full.f32`` hazard (~2 ulp, not correctly rounded) never arises here.
    """
    fields, _pml = build(m=m, courant=0.3)
    rng = np.random.default_rng(4242)
    shape = fields.grid.shape
    partner = (rng.standard_normal(shape).astype(np.float32)
               + 1j * rng.standard_normal(shape).astype(np.float32)).astype(np.complex64)
    targets = (stepping.B_CURL_TERMS if sub_step == "step_B" else stepping.D_CURL_TERMS)

    for index, _register, sign in module.IMR_TERMS[sub_step]:
        target = targets[index].target
        expected = stepping._cylindrical_imr_term(fields, target, partner, sign)
        row = module.imr_coefficient_row(np, target, sign, m, fields.grid.dt / fields.grid.dx,
                                         shape[0], np.complex64)
        assert row.shape == (shape[0], 1, 1)
        # stepping returns the term already negated into curl sign convention.
        assert same_bytes(expected, -(row * partner)), \
            f"{sub_step} {target} sign={sign:+g} m={m}: coefficient row differs"


@pytest.mark.parametrize("m", [1, -1, 2, -2, 3, -3, 5])
def test_the_imr_coefficient_rows_real_word_is_a_positive_zero(m):
    """+0.0, for BOTH signs of m and both signs of ``sign`` — and it must survive.

    ``stepping`` spells the numerator ``(-1j) * X``: multiplied by MINUS the
    imaginary unit, CPython computes ``re = (-0.0)*X - (-1.0)*0.0``, whose
    cross-term subtraction launders the sign to +0.0 either way. Multiplying by
    PLUS the unit instead gives ``re = 0.0*X - 1.0*0.0`` and PRESERVES a -0.0 for
    X < 0 — which is exactly what the |m| = 1 axis scalar does, four lines away.

    THE SIGN OF THE UNIT IS THE CAUSE, NOT THE OPERAND ORDER. Measured at
    X = +-0.37 and +-0.74: ``(-1j)*X`` and ``X*(-1j)`` are bit-identical, as are
    ``1j*X`` and ``X*1j``; only flipping the unit's sign (or moving the negation
    outside, ``-(1j*X)``, which gives -0.0 for X > 0) moves the real word. Pinned
    here because a comment that misattributes it invites a "harmless" reorder
    that really is harmless, and discourages the sign edit that is not.

    Why it matters that the zero is a real operand and not elided: ``(+0.0) -
    (+0.0)`` is ``+0.0`` where ``-(+0.0)`` is ``-0.0``, so a plane-wise shortcut
    that drops the zero cross terms is byte-wrong on signed-zero rows. That is the
    device leg's ``imr_planewise_zero_cross_terms`` mutation, named as data here
    so the module and the gate cannot drift about which choice is pinned.
    """
    assert module.MUTATION_IMR_PLANEWISE == "imr_planewise_zero_cross_terms"
    for sub_step in ("step_B", "step_D"):
        targets = (stepping.B_CURL_TERMS if sub_step == "step_B"
                   else stepping.D_CURL_TERMS)
        for index, _register, sign in module.IMR_TERMS[sub_step]:
            row = module.imr_coefficient_row(np, targets[index].target, sign, m,
                                             0.37, 8, np.complex64)
            real_words = np.ascontiguousarray(row.real.copy()).view(np.uint32).ravel()
            assert set(real_words.tolist()) == {0}, \
                (f"{targets[index].target} sign={sign:+g} m={m}: real word is not "
                 f"+0.0 (got {[hex(w) for w in set(real_words.tolist())]})")
            # The imaginary word is genuinely nonzero, so the row is not vacuous.
            assert np.count_nonzero(row.imag) == row.size


@pytest.mark.parametrize("m,dtdx", [(1, 0.37), (-1, 0.37), (1, 0.5),
                                    (-1, 0.3141592653589793)])
def test_the_axis_increment_scalars_reproduce_steppings_increment_bytewise(m, dtdx):
    """The |m| = 1 B-side scalars, checked THROUGH the increment ``stepping`` builds.

    ``(-dtdx) * (ep[0] - ep_above[0]) - (1j*(m*dtdx)) * ez_off_axis``, where
    ``ez_off_axis`` is the FIRST OFF-AXIS r row (MEEP's ``f[Ez][1-cmp] + (nz+1)``).
    Both scalars meet a complex64 array as WEAK Python scalars, so each is
    converted once and the multiply is a FULL complex product with the zero cross
    terms. Rebuilt here from the module's two numbers and byte-compared.
    """
    fields, pml = build(m=m, courant=dtdx)
    grid = fields.grid
    rng = np.random.default_rng(99)
    for name in ("Ex", "Ey", "Ez"):
        host = (rng.standard_normal(grid.shape).astype(np.float32)
                + 1j * rng.standard_normal(grid.shape).astype(np.float32))
        getattr(fields, name)[...] = host.astype(np.complex64)

    boundaries = stepping._boundary_kinds(grid, pml)
    phases = stepping._bloch_phases(grid, boundaries, fields.Ex, pml)
    step_dtdx = grid.dt / grid.dx
    electric = stepping._component_snapshot(fields, ("Ex", "Ey", "Ez"))
    target, expected = stepping._cylindrical_axis_increment_B(
        fields, electric, boundaries, phases, step_dtdx)
    assert target == "Bx"

    minus_dtdx, (inc_re, inc_im) = module.axis_increment_scalars(m, step_dtdx)
    ep = electric["Ey"]
    ep_above = stepping._shift_up(np, ep, 2, boundaries[2], phases[2])
    ez_off_axis = np.take(electric["Ez"], 1, axis=0)
    rebuilt = (minus_dtdx * (ep[stepping._face(0, 0)] - ep_above[stepping._face(0, 0)])
               - np.complex64(complex(inc_re, inc_im)) * ez_off_axis)
    assert same_bytes(expected, rebuilt)


@pytest.mark.parametrize("m,dtdx,expect_negative_zero",
                         [(-1, 0.37, True), (1, 0.37, False),
                          (-1, 0.5, True), (1, 0.5, False)])
def test_the_axis_increment_scalar_carries_a_signed_zero_real_word(m, dtdx,
                                                                  expect_negative_zero):
    """THE ASYMMETRY: this scalar's real word IS ``-0.0`` where the i*m/r row's is ``+0.0``.

    ``1j * (m*dtdx)`` multiplies by PLUS the imaginary unit, so CPython computes
    ``re = 0.0*X - 1.0*0.0`` — ``-0.0`` for X < 0. The i*m/r row multiplies by
    MINUS the unit and launders the same position to ``+0.0``. Both are
    host-rounded and passed through as data for exactly this reason; neither may
    be synthesized in-kernel.

    The four-way measurement behind the attribution is in
    :func:`test_the_imr_coefficient_rows_real_word_is_a_positive_zero`: operand
    order is a null, the unit's sign is not.
    """
    _minus, (inc_re, inc_im) = module.axis_increment_scalars(m, dtdx)
    # The sibling spelling, checked side by side so the asymmetry is one fact in
    # one place rather than two tests that could drift apart.
    row = module.imr_coefficient_row(np, "Bx", 1.0, m, dtdx, 4, np.complex64)
    assert int(np.float32(row.real.ravel()[1]).view(np.uint32)) == 0, \
        "the i*m/r row's real word must stay +0.0 while this scalar's is signed"
    word = np.float32(inc_re).view(np.uint32)
    assert inc_re == 0.0, "the real word is a zero of one sign or the other"
    assert bool(word == np.uint32(0x80000000)) is expect_negative_zero, \
        f"m={m}: real word {hex(int(word))}"
    assert np.float32(inc_im) == np.float32(m * dtdx)


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
@pytest.mark.parametrize("m", [-1, 3])
def test_the_prefix_is_byte_identical_to_the_one_the_array_path_builds(sub_step, m,
                                                                      monkeypatch):
    """The radial prefix, captured OUT OF ``step_B``/``step_D`` as they run.

    Not compared against a second transcription: ``stepping.cylindrical_rderiv_prefix``
    is wrapped so the real sub-step's own scan input and output are recorded, then
    :func:`cylindrical_complex_prefix` is asked for the same thing and the two are
    byte-compared. The scan STAYS on the array path — ``cupy.cumsum`` in float32
    is deterministic but is not a sequential accumulation and ``tl.cumsum``
    matches neither, so CuPy's scan order is the oracle — and this is the seam
    where that decision is checkable.
    """
    fields, pml = build(m=m, courant=0.3)
    grid = fields.grid
    rng = np.random.default_rng(7)
    for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Bx", "By", "Bz",
                 "Dx", "Dy", "Dz"):
        array = getattr(fields, name, None)
        if array is None:
            continue
        host = (rng.standard_normal(grid.shape).astype(np.float32)
                + 1j * rng.standard_normal(grid.shape).astype(np.float32))
        array[...] = host.astype(np.complex64)

    real_scan = stepping.cylindrical_rderiv_prefix
    seen = []

    def recording(xp, f_p, ir0, scratch=None):
        result = real_scan(xp, f_p, ir0, scratch=scratch)
        seen.append((np.array(f_p, copy=True), ir0, np.array(result, copy=True)))
        return result

    monkeypatch.setattr(stepping, "cylindrical_rderiv_prefix", recording)
    (stepping.step_B if sub_step == "step_B" else stepping.step_D)(fields, pml)
    assert len(seen) == 1, "the sub-step ran the radial scan once, or the seam moved"
    scan_input, ir0, scan_output = seen[0]
    assert ir0 == module.PREFIX[sub_step]["ir0"]

    names = ("Ex", "Ey", "Ez") if sub_step == "step_B" else ("Hx", "Hy", "Hz")
    sources = stepping._component_snapshot(fields, names)
    ours = module.cylindrical_complex_prefix(np, sub_step, sources)
    assert ours.shape == scan_output.shape
    assert same_bytes(ours, scan_output), f"{sub_step}: prefix differs from step's own"

    # And the B side really is EXTENDED BY ONE ZERO WALL ROW, D not.
    rows = grid.shape[0]
    if sub_step == "step_B":
        assert scan_input.shape[0] == rows + 1
        assert not np.any(scan_input[rows])
        assert ours.shape[0] == rows + 1
    else:
        assert scan_input.shape[0] == rows
        assert ours.shape[0] == rows


# ---------------------------------------------------------------------------
# Coverage — the positive admission
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
@pytest.mark.parametrize("m,accurate,courant", [(1, False, 0.37), (-1, False, 0.5),
                                                (2, False, 0.37), (3, True, 1 / 3.6),
                                                (5, False, 0.3)])
@pytest.mark.parametrize("z_metallic", [True, False])
def test_a_real_complex_dcyl_grid_is_admitted_but_for_the_cupy_backend(sub_step, m,
                                                                      accurate, courant,
                                                                      z_metallic):
    """Every |m| class, both z terminations, both sub-steps: ONE reason, the backend.

    The positive half of the enumeration. Without it a predicate that refused
    everything would pass every refusal test in this file, which is the failure
    mode a pile of negative tests has.
    """
    fields, pml = build(m=m, accurate=accurate, courant=courant, z_metallic=z_metallic)
    assert curl_reasons(fields, pml, sub_step) == []
    verdict = module.cylindrical_complex_curl_coverage(fields, pml, sub_step, probe=PROBE)
    assert verdict.covered is False and any("not cupy" in r for r in verdict.reasons)


@pytest.mark.parametrize("side", ["H", "E"])
@pytest.mark.parametrize("m", [1, -1, 3])
def test_the_constitutive_side_is_admitted_on_the_same_grids(side, m):
    """32 of the family's 64 sub-step slots, with NO new kernel.

    The finding this tranche turns on: ``stepping.update_H`` (:907-924) and
    ``update_E`` (:926-993) carry no cylindrical branch at all, so the certified
    ``complex_fields.bloch_constitutive_step`` body already computes both sides.
    Measured, not argued — the gate's identity leg is 480/480 rows, 0 differing
    uint32 words, every row asserting its own non-vacuity. This pins the
    predicate that lets the delegation happen.
    """
    fields, pml = build(m=m, courant=0.3)
    assert constitutive_reasons(fields, pml, side) == []


def test_both_predicates_reject_an_unknown_sub_step_or_side_loudly():
    """A typo is a ``ValueError``, never a quiet ``covered=False``.

    A silent False here would read as "this configuration is not covered" and send
    a correct run to the array path forever, which is the failure that never gets
    investigated.
    """
    fields, pml = build()
    with pytest.raises(ValueError, match="sub_step must be one of"):
        module.cylindrical_complex_curl_coverage(fields, pml, "step_E", probe=PROBE)
    with pytest.raises(ValueError, match="side must be one of"):
        module.cylindrical_complex_constitutive_coverage(fields, pml, "B", probe=PROBE)
    with pytest.raises(ValueError, match="sub_step must be one of"):
        module.plan_cylindrical_complex_curl(fields, pml, "step_E", probe=PROBE)
    with pytest.raises(ValueError, match="side must be one of"):
        module.plan_cylindrical_complex_constitutive(fields, pml, "B", probe=PROBE)


# ---------------------------------------------------------------------------
# Coverage — every clause, broken one at a time
# ---------------------------------------------------------------------------

def test_real_storage_is_refused_by_name_because_it_has_no_form_at_abs_m_one():
    """``force_complex_fields`` unset: |m| >= 1 has NO real-storage form.

    ``stepping`` raises on the combination, the driver refuses it and MEEP's
    ``change_m`` aborts — so a real Dcyl run is m = 0 and belongs to
    ``cylindrical_triton``. Refusing here by NAME is what keeps the two products
    from ever both admitting a run.
    """
    fields, pml = build()
    broken = Override(fields, force_complex_fields=False)
    reported = curl_reasons(broken, pml)
    assert any("force_complex_fields is not set" in r for r in reported)
    assert any("cylindrical_triton" in r for r in reported)


@pytest.mark.parametrize("pml_value,fragment",
                         [(None, "no active PML"),
                          ("inactive", "no active PML")])
def test_an_absent_or_inactive_absorber_is_refused(pml_value, fragment):
    """This is the SPLIT-FIELD product only; the no-PML path is a different kernel."""
    fields, pml = build()
    subject = None if pml_value is None else Override(pml, is_active=False)
    assert any(fragment in r for r in curl_reasons(fields, subject))


def test_an_unreadable_conductivity_table_is_refused_outright():
    """A MISSING ``condfac_for`` is not an absent conductivity.

    Inferring "no conductivity" from the absence of the reader is admission by
    attribute absence — the exact reasoning coverage exists to refuse. Stricter
    than the shipped predicate's magnetic-flag fallback, deliberately.
    """
    fields, pml = build()
    assert any("does not expose condfac_for" in r
               for r in curl_reasons(Override(fields, condfac_for=MISSING), pml))


def test_a_conductivity_on_any_curl_target_is_refused_by_target_name():
    """Named per target, so the refusal says WHICH one disqualified the run."""
    fields, pml = build()
    for target in ("Bx", "Dz"):
        conductive = Override(fields,
                              condfac_for=lambda name, t=target: object() if name == t else None)
        reported = curl_reasons(conductive, pml)
        assert any(f"a conductivity is installed on {target}" in r for r in reported)
    raiser = Override(fields, condfac_for=lambda name: (_ for _ in ()).throw(RuntimeError("x")))
    assert any("raised" in r for r in curl_reasons(raiser, pml))


def test_a_magnetic_conductivity_is_refused():
    fields, pml = build()
    assert any("magnetic (B) conductivity" in r
               for r in curl_reasons(Override(fields, has_magnetic_conductivity=True), pml))


@pytest.mark.parametrize("attribute,value,fragment", [
    ("has_polarizations", True, "a susceptibility is registered"),
    ("has_nonlinearity", True, "chi2/chi3 is installed"),
    ("stores_E", False, "E is recomputed from D"),
])
def test_the_fields_side_refusals_each_fire_alone(attribute, value, fragment):
    """One attribute at a time, from an otherwise admitted real ``Fields``."""
    fields, pml = build()
    assert curl_reasons(fields, pml) == []          # the control
    assert any(fragment in r for r in curl_reasons(Override(fields, **{attribute: value}), pml))


@pytest.mark.parametrize("attribute,value,fragment", [
    ("bfast_active", True, "BFAST is active"),
    ("beta", 0.25, "beta=0.25 is nonzero"),
    ("has_bloch", True, "compiles no Bloch rotation"),
    ("k_point", (0.0, 0.0, 0.3), "is not exactly zero"),
    ("has_symmetry", True, "mirror plane is active"),
    ("cylindrical", False, "grid is not cylindrical"),
    ("accurate_fields_near_cylorigin", None, "does not report accurate_fields_near_cylorigin"),
])
def test_the_grid_side_refusals_each_fire_alone(attribute, value, fragment):
    """One grid attribute at a time.

    ``Grid`` already refuses several of these on a cylindrical cell. The clauses
    exist anyway, because "another module already guards it" is exactly the
    reasoning this file exists to refuse — the guard has to be where the
    admission decision is made.
    """
    fields, pml = build()
    broken = Override(fields, grid=Override(fields.grid, **{attribute: value}))
    assert any(fragment in r for r in curl_reasons(broken, pml)), \
        f"{attribute}={value!r} was admitted"


def test_m_zero_under_complex_storage_is_admitted_and_real_storage_is_named():
    """The disjointness clause, in this direction, since 2026-09-04.

    m = 0 with ``force_complex_fields=True`` is this family's M_ZERO arm and
    carries NO m clause; m = 0 with REAL storage is ``cylindrical_triton``'s and
    the refusal NAMES the storage flag and the other product. A silent overlap
    between two products that compute different arithmetic is still the failure
    this prevents — the split just moved from m to storage.
    """
    fields, pml = build(m=0, courant=0.37)
    assert curl_reasons(fields, pml) == []
    assert not any("grid.m = 0" in r for r in curl_reasons(fields, pml))
    real, real_pml = build(m=0, force_complex_fields=False, courant=0.5)
    reported = curl_reasons(real, real_pml)
    assert any("force_complex_fields is not set" in r and "cylindrical_triton" in r
               for r in reported), reported


def test_an_unreadable_m_is_refused_rather_than_defaulted():
    fields, pml = build()
    for value, fragment in ((MISSING, "does not report m"),
                            ("three", "is not an integer")):
        broken = Override(fields, grid=Override(fields.grid, m=value))
        assert any(fragment in r for r in curl_reasons(broken, pml))


def test_a_cartesian_grid_is_refused_on_geometry_not_only_on_the_name():
    """A real Cartesian complex grid: the coordinate system AND the axis flags."""
    fields, pml = cartesian()
    reported = curl_reasons(fields, pml)
    assert any("grid is not cylindrical" in r for r in reported)
    assert any("is not (True, False, False)" in r for r in reported)
    assert any("axis 0 boundary" in r for r in reported)


def test_a_wrong_axis_flag_triple_is_refused():
    """r must be the LEADING axis; the kernel indexes the coefficient row by ``i``."""
    fields, pml = build()
    broken = Override(fields, grid=Override(fields.grid, is_axis=lambda axis: axis == 1))
    assert any("is not (True, False, False)" in r for r in curl_reasons(broken, pml))


def test_a_z_termination_outside_the_admitted_pair_is_refused():
    """Only ``metallic`` and ``periodic`` on z; anything else takes a ghost arm
    the kernel does not compile.

    Forced with a mirror on z, which is the only other kind
    ``stepping._boundary_kinds`` can return for that axis. The absorber is built
    WITHOUT a z layer for this case, because ``_require_consistent_pml`` refuses
    an absorber sitting on a fold's mirror plane and the predicate would then
    report the unresolvable grid instead of the clause under test. The fold
    clauses still fire alongside — that is honest, not a confound: the assertion
    is that the Z clause is among the reasons.
    """
    fields, pml = build(pml_cells={"x": (0, 4), "z": 0})
    broken = Override(fields, grid=Override(fields.grid,
                                            is_mirrored=lambda axis: axis == 2))
    reported = curl_reasons(broken, pml)
    assert any("axis 2 boundary 'mirror' is outside" in r for r in reported)
    assert any("folded by a mirror plane" in r for r in reported)


def test_a_phi_extent_above_one_cell_is_refused():
    """Dcyl is 2.5-D: the exp(i*m*phi) dependence is analytic, not sampled."""
    fields, pml = build()
    broken = Override(fields, grid=Override(fields.grid, shape=(16, 3, 20)))
    assert any("phi extent is 3 cells" in r for r in curl_reasons(broken, pml))
    flat = Override(fields, grid=Override(fields.grid, shape=(16, 20)))
    assert any("is not three-dimensional" in r for r in curl_reasons(flat, pml))


def test_an_unallocated_volume_or_auxiliary_is_refused_by_name():
    """Every target, every ``fu_*`` and every source — named individually.

    The auxiliaries matter as much as the fields: the split-field recurrence reads
    ``fu`` before it writes it.
    """
    fields, pml = build()
    for name in ("Bz", "fu_Bx", "Ey", "Ez"):
        broken = Override(fields, **{name: None})
        assert any(f"{name} is not allocated" in r for r in curl_reasons(broken, pml)), name


def test_float32_storage_is_refused_by_the_layout_clause():
    """The DECLARATION and the DTYPE are separate clauses, and both must bite.

    A run that claims ``force_complex_fields`` but stores float32 would be read by
    the kernel as word pairs of adjacent cells.
    """
    fields, pml = build(m=0, force_complex_fields=False)
    reported = curl_reasons(Override(fields, force_complex_fields=True), pml)
    assert any("is not complex64" in r for r in reported)


def test_a_non_contiguous_volume_is_refused_on_contiguity_not_on_shape():
    """A strided view would be read in the wrong order by the flat word index.

    The stride is made with a view whose SHAPE and DTYPE both still match the
    grid, so the contiguity clause is the only one that can fire. A slice that
    also changed the shape would be caught by the shape clause and prove nothing
    about contiguity — which is the property the word-pair addressing depends on.
    """
    fields, pml = build()
    shape = tuple(fields.grid.shape)
    backing = np.zeros((shape[0], shape[1], shape[2] * 2), dtype=np.complex64)
    strided = backing[:, :, ::2]
    assert strided.shape == shape and str(strided.dtype) == "complex64"
    assert not strided.flags.c_contiguous
    reported = curl_reasons(Override(fields, Bz=strided), pml)
    assert any("Bz is not C-contiguous" in r for r in reported), reported
    assert not any("shape" in r for r in reported), reported


@pytest.mark.parametrize("probe,fragment", [
    (None, "no complex-multiply expansion probe artifact"),
    ({"backend": "numpy", "patterns": {}}, "missing, ambiguous or not for the"),
    ({"backend": "cupy", "patterns": {"c8_mul_c8": "FMA_V1"}},
     "missing, ambiguous or not for the"),
])
def test_the_expansion_constexpr_may_not_be_guessed(probe, fragment):
    """``EXPANSION`` is a MEASURED platform fact bound from an artifact.

    Refused three ways: no artifact, an artifact for the wrong backend, and an
    artifact that does not classify every orientation. A disagreeing platform
    earns a new arm, not a shrug.
    """
    fields, pml = build()
    reported = curl_reasons(fields, pml, probe=probe)
    assert any(fragment in r for r in reported)
    assert module.plan_cylindrical_complex_curl(fields, pml, "step_B", probe=probe) is None


# ---------------------------------------------------------------------------
# Disjointness with cylindrical_triton, in BOTH directions
# ---------------------------------------------------------------------------

def test_the_two_cylindrical_products_are_disjoint_by_storage_in_both_directions():
    """Neither predicate can admit a grid the other admits.

    Until 2026-09-04 the split was by m (``cylindrical_triton`` refused
    ``m != 0``, this module refused ``m == 0``); it is now by STORAGE —
    ``cylindrical_triton`` refuses ``force_complex_fields`` by name and this
    module requires it — so a real m = 0 grid is theirs alone, a complex m = 0
    grid is ours alone, and an |m| >= 1 grid (complex by necessity) is ours
    alone. Checked on real grids of each kind, with the backend clause filtered
    out of both so the comparison is about the storage clause alone.
    """
    def other_reasons(fields, pml):
        verdict = real_cylindrical.cylindrical_curl_coverage(fields, pml)
        return [r for r in verdict.reasons if "not cupy" not in r]

    real_storage, real_pml = build(m=0, force_complex_fields=False, courant=0.5)
    assert other_reasons(real_storage, real_pml) == []           # theirs admits
    mine = curl_reasons(real_storage, real_pml)                  # mine refuses
    assert any("force_complex_fields is not set" in r for r in mine), mine

    complex_zero, complex_zero_pml = build(m=0, courant=0.37)
    assert curl_reasons(complex_zero, complex_zero_pml) == []    # mine admits
    theirs = other_reasons(complex_zero, complex_zero_pml)       # theirs refuses
    assert any("force_complex_fields=True" in r for r in theirs), theirs

    complex_storage, complex_pml = build(m=-1, courant=0.37)
    assert curl_reasons(complex_storage, complex_pml) == []      # mine admits
    theirs = other_reasons(complex_storage, complex_pml)         # theirs refuses
    assert theirs, "cylindrical_triton admitted an |m| >= 1 grid"
    assert any("m" in r for r in theirs)


def test_the_shared_blanket_clauses_still_refuse_this_domain_for_their_own_products():
    """``coverage._grid_reasons`` and ``complex_fields._complex_grid_reasons`` must
    KEEP refusing Dcyl — and this module must not compose either.

    Each is now load-bearing in two directions at once: it is right for its own
    product, and it is what makes leaving this family unwired safe. Widening
    either to admit Dcyl would create the overlap the disjointness argument rules
    out, so the refusal is pinned rather than assumed.
    """
    fields, pml = build()
    verdict = complex_module.complex_pml_curl_coverage(fields, pml, "step_B", probe=PROBE)
    assert verdict.covered is False
    assert any("cylindrical" in r.lower() for r in verdict.reasons), verdict.reasons

    code = code_of(PACKAGE_DIR / "cylindrical_complex.py")
    assert "_grid_reasons" not in code, \
        "this module composes a blanket clause that refuses its own whole domain"
    assert "_complex_grid_reasons" not in code


# ---------------------------------------------------------------------------
# The per-sub-step split coverage.py makes
# ---------------------------------------------------------------------------

def test_an_offdiagonal_epsilon_is_admitted_by_the_curl_and_refused_by_update_E():
    """Constitutive-only, so it splits by sub-step exactly as ``coverage.py`` does.

    Its whole effect is inside ``update_E``, where the row product reads
    neighbours and the sub-step stops being element-wise. The curl never sees it.
    Refusing it everywhere would be over-strict; admitting it on E would be wrong.
    """
    fields, pml = build()
    offdiag = Override(fields, has_offdiagonal_epsilon=True)
    assert curl_reasons(offdiag, pml, "step_B") == []
    assert curl_reasons(offdiag, pml, "step_D") == []
    assert constitutive_reasons(offdiag, pml, "H") == []
    reported = constitutive_reasons(offdiag, pml, "E")
    assert any("off-diagonal chi1inv row" in r for r in reported)


# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

class NumpyClaimingCupy:
    """NumPy wearing CuPy's name, so the ENGINE route can be built on a laptop.

    The backend clause is the only thing standing between an admitted Dcyl
    configuration and a real plan here, and it is the one clause that cannot be
    satisfied on this box. Delegating every attribute to NumPy keeps the plan
    builder doing real work — ``imr_coefficient_row`` really runs its ``arange``,
    ``maximum`` and float64 divide — while ``__name__`` answers the predicate.

    This is a TEST-ONLY fiction and it does NOT make anything here a device
    result: ``run()`` is never called, no kernel is compiled and no bytes are
    compared against hardware.
    """

    __name__ = "cupy"

    def __getattr__(self, name):
        return getattr(np, name)


@pytest.mark.parametrize("z_metallic,expected_bcz", [(True, 1), (False, 0)])
@pytest.mark.parametrize("m,accurate,courant,expected_class,expected_zero_rows",
                         [(-1, False, 0.37, 1, 0), (2, False, 0.37, 2, 2),
                          (3, True, 1 / 3.6, 2, 1)])
@pytest.mark.parametrize("sub_step,suffix", [("step_B", "_h"), ("step_D", "")])
def test_the_engine_route_derives_every_constexpr_from_the_grid(sub_step, suffix, m,
                                                                accurate, courant,
                                                                expected_class,
                                                                expected_zero_rows,
                                                                z_metallic,
                                                                expected_bcz):
    """``plan_cylindrical_complex_curl`` — the route the driver would take.

    Distinct from the ``from_arrays`` test, which is HANDED its constexprs: here
    the builder has to DERIVE them, and the derivations are where a silent error
    lives. ``BCZ`` in particular is computed from the resolved boundary triple
    (``1 if kinds[2] == 'metallic' else 0``), and inverting it selects the wrong
    ghost arm and the wrong mask planes — a converged, smooth, wrong field rather
    than a crash. Without this test that mapping is unreachable on a laptop,
    because the backend clause refuses every plan.
    """
    fields, pml = build(m=m, accurate=accurate, courant=courant, z_metallic=z_metallic)
    claims_cupy = Override(fields, grid=Override(fields.grid, xp=NumpyClaimingCupy()))
    plan = module.plan_cylindrical_complex_curl(claims_cupy, pml, sub_step, probe=PROBE)

    assert plan is not None, "the engine route refused an admitted configuration"
    assert plan.bcz == expected_bcz
    assert plan.m_class == expected_class
    assert plan.zero_rows == expected_zero_rows
    assert plan.sub_step == sub_step
    assert plan.shape == tuple(fields.grid.shape)
    assert plan.dtdx == float(fields.grid.dt / fields.grid.dx)
    assert plan.expansion == 1                       # FMA_V1, from the probe
    # The Yee sub-lattice comes from the SUB-STEP, and a swap is a silent
    # half-cell error in the absorber profile.
    expected_coefficients = [getattr(pml, f"{stem}_{axis}{suffix}").reshape(-1)
                             for axis in "xyz" for stem in ("kms", "sinv")]
    assert len(plan._coefficients) == 6
    for pointer, wanted in zip(plan._coefficients, expected_coefficients):
        assert same_bytes(pointer.array, wanted)
    # The i*m/r rows are grid invariants built ONCE at plan time.
    assert len(plan._imr_rows) == 2
    targets = (stepping.B_CURL_TERMS if sub_step == "step_B" else stepping.D_CURL_TERMS)
    for pointer, (index, _register, sign) in zip(plan._imr_rows,
                                                 module.IMR_TERMS[sub_step]):
        wanted = module.imr_coefficient_row(
            np, targets[index].target, sign, m, fields.grid.dt / fields.grid.dx,
            fields.grid.shape[0], np.complex64).reshape(-1)
        assert same_bytes(pointer.array, np.ascontiguousarray(wanted).view(np.float32))


def test_the_engine_route_still_refuses_a_real_numpy_backend():
    """The backend clause is what :class:`NumpyClaimingCupy` bypasses, so it is
    checked here separately — otherwise the fiction could hide its removal."""
    fields, pml = build()
    assert module.plan_cylindrical_complex_curl(fields, pml, "step_B",
                                                probe=PROBE) is None
    verdict = module.cylindrical_complex_curl_coverage(fields, pml, "step_B", probe=PROBE)
    assert any("not cupy" in r for r in verdict.reasons)


def test_the_engine_route_refuses_to_none_and_never_raises():
    """None is the ONLY refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly."""
    fields, pml = build()
    for sub_step in ("step_B", "step_D"):
        assert module.plan_cylindrical_complex_curl(fields, pml, sub_step,
                                                    probe=PROBE) is None
    for side in ("H", "E"):
        assert module.plan_cylindrical_complex_constitutive(fields, pml, side,
                                                            probe=PROBE) is None


def arrays_for(fields, sub_step):
    """The gate's ``from_arrays`` inputs, off a real ``Fields``."""
    from meep_gpu.triton_kernels.launch import SUB_STEPS

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
             + tuple(spec["sources"]))
    return {name: getattr(fields, name) for name in names}


def flat_for(pml, sub_step):
    suffix = "_h" if sub_step == "step_B" else ""
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


@pytest.mark.parametrize("sub_step,backward", [("step_B", 0), ("step_D", 1)])
@pytest.mark.parametrize("m,accurate,expected_class,expected_zero_rows",
                         [(1, False, 1, 0), (-1, False, 1, 0), (2, False, 2, 2),
                          (3, False, 2, 3), (3, True, 2, 1), (5, False, 2, 5)])
@pytest.mark.parametrize("z_metallic,bcz", [(True, 1), (False, 0)])
def test_the_plan_carries_the_constexprs_the_grid_implies(sub_step, backward, m,
                                                          accurate, expected_class,
                                                          expected_zero_rows,
                                                          z_metallic, bcz):
    """Built through the gate's own ``from_arrays`` route, on NumPy arrays.

    Constructible here because ``CupyPointer`` defers the device address until
    launch — so the plan's SHAPE is laptop-checkable even though its ``run()`` is
    not. Every constexpr is a compile-time fact about the configuration and a
    wrong one is a plane of wrong values rather than a crash: ``M_CLASS`` selects
    increments versus zeroing, ``ZERO_ROWS`` how many rows the zeroing holds, and
    ``BCZ`` which ghost arm and which mask planes.
    """
    courant = 1 / 3.6 if accurate else 0.37
    fields, pml = build(m=m, accurate=accurate, courant=courant, z_metallic=z_metallic)
    plan = module.plan_cylindrical_complex_curl_from_arrays(
        sub_step, arrays_for(fields, sub_step), flat_for(pml, sub_step),
        fields.grid.dt / fields.grid.dx, m, accurate, bcz, 1, np)

    assert plan.sub_step == sub_step
    assert plan.backward == backward
    assert plan.bcz == bcz
    assert plan.m_class == expected_class
    assert plan.zero_rows == expected_zero_rows
    assert plan.shape == tuple(fields.grid.shape)
    assert plan.n_elem == int(np.prod(fields.grid.shape))
    assert plan.block == module.DEFAULT_BLOCK
    assert plan._grid == ((plan.n_elem + plan.block - 1) // plan.block,)
    # Two i*m/r rows, one per coupled target; target 1 has none.
    assert len(plan._imr_rows) == len(module.IMR_TERMS[sub_step]) == 2
    assert plan.increment_scalars == module.axis_increment_scalars(
        m, fields.grid.dt / fields.grid.dx)


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_plans_prefix_is_the_array_paths_own_scan(sub_step):
    """``plan.prefix()`` recomputes per launch from the CURRENT sources, and equals
    :func:`cylindrical_complex_prefix` bytewise.

    The plan is deliberately NOT allocation-free — it cannot be, the prefix
    depends on the sources — and this pins that the per-launch recomputation goes
    through the shipped scan rather than a cached stale volume.
    """
    fields, pml = build(m=-1)
    rng = np.random.default_rng(11)
    for name in ("Ey", "Hy"):
        host = (rng.standard_normal(fields.grid.shape).astype(np.float32)
                + 1j * rng.standard_normal(fields.grid.shape).astype(np.float32))
        getattr(fields, name)[...] = host.astype(np.complex64)

    arrays = arrays_for(fields, sub_step)
    plan = module.plan_cylindrical_complex_curl_from_arrays(
        sub_step, arrays, flat_for(pml, sub_step), 0.37, -1, False, 1, 1, np)
    expected = module.cylindrical_complex_prefix(np, sub_step, arrays)
    assert same_bytes(plan.prefix(), expected)

    # It re-reads the sources: mutate one and the prefix must move.
    before = np.array(plan.prefix(), copy=True)
    getattr(fields, module.PREFIX[sub_step]["component"])[...] += np.complex64(1 + 1j)
    assert not same_bytes(plan.prefix(), before)


def test_the_constitutive_plan_delegates_to_the_certified_complex_plan():
    """No new constitutive kernel: the builder returns the CERTIFIED plan type.

    Forced past the backend clause with a grid whose array module claims to be
    CuPy, which is enough because ``ComplexConstitutivePlan`` never touches ``xp``
    — it wraps pointers and defers the address to launch. What that shows is the
    thing the finding rests on: this family's E and H sub-steps ARE
    ``complex_fields.bloch_constitutive_step``, certified, with
    no cylindrical arithmetic added anywhere.
    """
    fields, pml = build(m=-1)
    fake_cupy = types.SimpleNamespace(__name__="cupy")
    claims_cupy = Override(fields, grid=Override(fields.grid, xp=fake_cupy))
    for side, half_integer in (("H", False), ("E", True)):
        plan = module.plan_cylindrical_complex_constitutive(claims_cupy, pml, side,
                                                            probe=PROBE)
        assert isinstance(plan, complex_module.ComplexConstitutivePlan), side
        assert plan.side == side
        assert plan.shape == tuple(fields.grid.shape)
        assert plan.scale == (1 if side == "E" else 0)
        # The Yee sub-lattice is chosen at build time and nowhere else; a swap is a
        # silent half-cell error in the absorber profile.
        suffix = "_h" if half_integer else ""
        expected = [getattr(pml, f"{stem}_{axis}{suffix}").reshape(-1)
                    for axis in "xyz" for stem in ("kps", "kms")]
        assert len(plan._coefficients) == len(expected) == 6
        for pointer, wanted in zip(plan._coefficients, expected):
            assert same_bytes(pointer.array, wanted)


def test_the_curl_plan_rejects_an_unknown_sub_step():
    fields, pml = build()
    with pytest.raises(ValueError, match="sub_step must be one of"):
        module.CylindricalComplexCurlPlan(
            "step_E", fields.grid.shape, 0.37, 1, False, 1, 1, 256,
            [], [], [], [], np)


# ---------------------------------------------------------------------------
# The reference leg — the ONE identity this round actually claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m,accurate,z_metallic,courant", [
    (-1, False, True, 0.37),                       # |m| = 1, metallic z
    (3, False, False, 0.3141592653589793),         # |m| >= 2, periodic z, non-p2 Courant
])
@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_gate_reference_transcription_equals_the_array_path(m, accurate,
                                                                z_metallic, courant,
                                                                sub_step):
    """THE leg that stops "bit-identical" meaning the harness reproduced itself.

    The gate's in-file reference is compared against ``stepping.step_B``/``step_D``
    on a REAL complex Dcyl ``Grid``/``Fields``/``PML``, PER SUB-STEP and by uint32
    word — never ``allclose``. A slice runs here so the merge bar carries it; the
    full sweep (416 rows x 8 sub-steps, four Courant numbers with three
    non-power-of-two, four shapes, both z terminations, both |m| classes) is the
    gate's ``--self-check`` and reported 416/416 identical.

    PER SUB-STEP is load-bearing and was learned the hard way: the zero-init
    divergence is TRANSIENT — a ``-0.0`` that survives one sub-step is laundered
    back by the next ``fu *= kms`` sign flip — and an end-of-run compare reported
    0 differing words on a state that had genuinely diverged at step 1.

    NOTE what this does NOT establish: nothing about the Triton kernel. No device
    run has been taken for this family.
    """
    gate = gate_module()
    row = gate._reference_row((13, 1, 11), m, accurate, courant, z_metallic,
                              sub_step, 4)
    assert row["first_divergence"] is None, row
    assert row["steps"] == 4
    assert tuple(row["boundaries"]) == ("axis", "periodic",
                                        "metallic" if z_metallic else "periodic")


@pytest.mark.parametrize("seeding", ["random", "signed_zero"])
@pytest.mark.parametrize("m,side", [(-1, "H"), (-1, "E"), (3, "H"), (3, "E")])
def test_the_constitutive_identity_leg_holds_on_this_host(m, side, seeding):
    """The evidence for "32 of the 64 slots need no new kernel".

    ``stepping.update_H``/``update_E`` against a plain complex elementwise
    ``dsigw`` reference carrying NO cylindrical clause of any kind. If any row
    differed, the delegation to the certified constitutive body would be wrong and
    this tranche would need two kernels rather than one. The gate's full leg is
    480/480 rows, 0 differing uint32 words.

    BOTH SEEDINGS run here, and the row asserts its OWN non-vacuity — it moved
    words, and a ``signed_zero`` row carried a nonzero ±0 census. The leg used to
    sweep ``zero_init`` instead, where the constitutive sub-step leaves every word
    +0.0 forever; half its rows could not fail and reported a pass anyway.
    """
    gate = gate_module()
    row = gate.one_constitutive_case(np, (13, 1, 11), m, 0.37, True, side, seeding)
    assert row["first_divergence"] is None, row
    assert row["moved_words"] > 0, "a row that moved nothing cannot fail"
    if seeding == "signed_zero":
        assert row["census_peak"] > 0, "the ±0 class was never reached"


def test_a_zero_initialised_constitutive_row_is_refused_as_vacuous():
    """The vacuity guard, exercised on the state that fooled the first cut.

    An all-zero constitutive state is a FIXED POINT of ``fw = value;
    field += kps*fw; field -= kms*prev`` — every word stays +0.0 — so the row
    compares two untouched states and passes whatever the reference does. The
    gate must refuse it by name rather than count it.
    """
    gate = gate_module()

    def all_zero(xp, fields, shape, seeding):
        for name in gate.CONSTITUTIVE_VOLUMES:
            array = getattr(fields, name, None)
            if array is not None:
                array[...] = 0
        return 1                     # claims a seed, writes only zeros

    original = gate.seed_constitutive
    gate.seed_constitutive = all_zero
    try:
        with pytest.raises(AssertionError, match="VACUOUS"):
            gate.one_constitutive_case(np, (13, 1, 11), -1, 0.37, True, "H",
                                       "random")
    finally:
        gate.seed_constitutive = original


@pytest.mark.parametrize("name", ["imr_dropped", "axis_increment_not_negated",
                                  "bz_flat_grouping"])
def test_representative_reference_needles_are_live_not_merely_armed(name):
    """A mutation that changes nothing is a test that proves nothing.

    Each of these edits the gate's reference transcription, recompiles the WHOLE
    file as a throwaway module (so a mutation cannot pass by landing in a helper
    the reference never calls) and must move bytes. ``bz_flat_grouping`` is the
    one that would be invisible under an exact-binary Courant alone, which is why
    the sweep carries three non-power-of-two values.
    """
    gate = gate_module()
    source = GATE_PATH.read_text(encoding="utf-8")
    spec = {entry[0]: entry for entry in gate.REFERENCE_MUTATIONS}[name]
    _name, old, new, verdict = spec
    assert verdict == "caught"
    assert source.count(old) >= 1, f"{name}: pattern DISARMED, it no longer matches"
    mutated = gate._load_module(source.replace(old, new), f"_mutant_{name}")
    assert mutated is not None

    clean = gate._reference_row((13, 1, 11), -1, False, 0.37, True, "step_B", 3)
    assert clean["first_divergence"] is None
    dirty = mutated._reference_row((13, 1, 11), -1, False, 0.37, True, "step_B", 3)
    assert dirty["first_divergence"] is not None, \
        f"{name}: NEEDLE-MISSED — the mutation changed no bytes"


def test_the_predicted_null_that_licenses_the_kernels_single_subtract():
    """``curl - (c*g)`` and ``curl + (-(c*g))`` are the same bits, on every input.

    IEEE-754 defines subtraction AS addition of the negation, including on signed
    zeros — which is what licenses the kernel to spell the i*m/r fold as one
    subtract per plane instead of a negate-then-add. Recorded as a PREDICTED NULL
    with its reason rather than left as an unexamined pass: a null that is not
    predicted is indistinguishable from a dead needle.
    """
    gate = gate_module()
    spec = {entry[0]: entry for entry in gate.REFERENCE_MUTATIONS}["imr_negation_as_subtract"]
    _name, old, new, verdict = spec
    assert isinstance(verdict, tuple) and verdict[0] == "null"
    assert "IEEE-754" in verdict[1]

    source = GATE_PATH.read_text(encoding="utf-8")
    assert source.count(old) >= 1, "the null's pattern DISARMED; it no longer matches"
    mutated = gate._load_module(source.replace(old, new), "_mutant_null_subtract")
    row = mutated._reference_row((13, 1, 11), -1, False, 0.37, True, "step_B", 3)
    assert row["first_divergence"] is None, \
        "the two spellings differ; the kernel's single-subtract fold is unlicensed"


def test_the_zero_init_leg_refuses_to_pass_vacuously():
    """CENSUS 0 IS VACUOUS, NOT PASSED.

    A zero-initialised row can only reach the signed-zero class if some ``kms``
    goes NEGATIVE — which needs a thin absorber. With a 5-cell layer the minimum
    ``kms_z`` is positive, no ``-0.0`` is ever produced, and the row reports a
    needle-miss for a reason that has nothing to do with the mutation. The gate
    raises with the measured numbers rather than reporting a pass, and that
    behaviour is what this pins.
    """
    gate = gate_module()
    # A negative coefficient IS present and the class was still never reached:
    # the row could have discriminated and did not. That is the harness fault.
    with pytest.raises(AssertionError, match="VACUOUS"):
        gate.assert_zero_init_non_vacuous("probe", [{"neg_zero_words": 0},
                                                    {"neg_zero_words": 0}], -0.6)
    assert gate.assert_zero_init_non_vacuous(
        "probe", [{"neg_zero_words": 0}, {"neg_zero_words": 85}], -0.6) == "live"

    # AND THE THIRD CLASS, which the first device run is what found: with NO
    # negative bound coefficient the ±0 class is out of reach BY CONSTRUCTION,
    # and failing the row would be failing the physics. Measured on the gate's
    # own build_engine at 2 absorber cells, Courant 0.5, shape (20, 1, 24):
    # z PERIODIC + step_D binds the INTEGER sub-lattice, whose minimum kms is
    # +0.190 (a z-periodic grid carries no z absorber at all), and its peak
    # census is 0 for every m — where the other three (z, sub_step) pairs
    # measure -0.821 / -2.238 / -0.821 and peaks of 66 / 19 / 24.
    assert gate.assert_zero_init_non_vacuous(
        "probe", [{"neg_zero_words": 0}], +0.19) == \
        "unreachable_no_negative_coefficient"
    assert gate.signed_zero_reach(0, -1.0) == "VACUOUS"
    assert gate.signed_zero_reach(0, 0.0) == "unreachable_no_negative_coefficient"
    assert gate.signed_zero_reach(3, 0.5) == "live"

    # THE LEG-LEVEL FLOOR the row-level one cannot carry: a sweep whose every
    # zero-init row is out of the class swept it without measuring it.
    out_of_reach = [{"zero_init": True,
                     "signed_zero_reach": "unreachable_no_negative_coefficient"}]
    with pytest.raises(AssertionError, match="VACUOUS zero-init half"):
        gate.assert_leg_reaches_signed_zero("probe", out_of_reach)
    gate.assert_leg_reaches_signed_zero(
        "probe", out_of_reach + [{"zero_init": True, "signed_zero_reach": "live"}])
    gate.assert_leg_reaches_signed_zero("probe", [{"zero_init": False}])

    # And the minimum is taken over the kms tables ONLY: sinv is not a producer.
    assert gate.minimum_bound_kms(
        {"kms_x": np.array([0.5], dtype=np.float32),
         "kms_z": np.array([-0.25], dtype=np.float32),
         "sinv_x": np.array([-9.0], dtype=np.float32)}) == pytest.approx(-0.25)

    # AND THE ABSORBER DEPTH REALLY IS WHAT DECIDES IT, measured here rather than
    # asserted in a comment. The earlier version of this block built an UNSTEPPED
    # Fields and checked the census dict's KEYS — on a state that has never been
    # touched the census is structurally 0, so it would have passed identically
    # with the thick absorber it claims to discriminate.
    census_by_depth = {}
    for cells in (gate.ZERO_INIT_PML_CELLS, gate.SEEDED_PML_CELLS):
        grid, fields, pml = gate.build_engine(np, (20, 1, 24), -1, 0.37, True,
                                              False, cells)
        gate.seed_state(np, fields, (20, 1, 24), True)
        volumes = {name: getattr(fields, name)
                   for name in gate.COMPARED + gate.SOURCES}
        coefficients = gate.coefficient_dict(pml, half_integer=True)
        boundaries = stepping._boundary_kinds(grid, pml)
        peak = 0
        for _ in range(4):
            gate.reference_cyl_complex_step(np, "step_B", volumes, coefficients,
                                            grid.dt / grid.dx, boundaries, -1,
                                            False)
            peak = max(peak, gate.negative_zero_census(fields)["neg_zero_words"])
        census_by_depth[cells] = peak
    thin, thick = census_by_depth[gate.ZERO_INIT_PML_CELLS], \
        census_by_depth[gate.SEEDED_PML_CELLS]
    assert thin > 0, ("the THIN absorber must reach the signed-zero class; "
                      f"census {census_by_depth}")
    assert thick == 0, ("the THICK absorber must NOT reach it — that is why the "
                        f"zero-init rows use the thin one; census {census_by_depth}")


def test_the_sweep_carries_a_non_power_of_two_courant_and_both_terminations():
    """Case discipline, pinned as data rather than trusted to the sweep's author.

    0.5 alone certifies broken kernels — exact binary scaling makes the FMA and
    associativity discrepancies vanish — so three non-power-of-two Courant numbers
    are in the product. Both z terminations, both signs at |m| = 1 (the i*m/r
    coefficient and the ghost phase both flip with the sign), both sub-steps, and
    the accurate branch at its own stability bound.
    """
    gate = gate_module()
    assert 0.5 in gate.DTDX and is_power_of_two(0.5)
    assert gate.DTDX[0] == 0.5, \
        "0.5 goes FIRST so a failure at it is visibly not a rounding failure"
    non_power_of_two = [c for c in gate.DTDX if not is_power_of_two(c)]
    assert len(non_power_of_two) >= 3, (gate.DTDX, non_power_of_two)
    assert set(gate.Z_METALLIC) == {True, False}
    assert set(gate.SUB_STEPS) == {"step_B", "step_D"}
    ms = [m for m, _accurate in gate.M_CASES]
    assert 1 in ms and -1 in ms, "both signs at |m| = 1"
    assert gate.ZERO_INIT_PML_CELLS < gate.SEEDED_PML_CELLS, \
        "the zero-init rows need the THIN absorber to reach a negative kms"

    # EVERY DECLARED CASE MUST BE ABLE TO RUN. "the accurate branch is swept" was
    # asserted as a TABLE property — ``any(accurate for _m, accurate in
    # M_CASES)`` — while the (3, True) entry contributed ZERO rows, because every
    # Courant in DTDX was above its 1/(|m| + 0.5) = 0.2857 stability bound and the
    # sweep skipped it with a bare ``continue``. A table entry that runs nothing
    # advertises coverage it does not have.
    for m, accurate in gate.M_CASES:
        runnable = [c for c in gate.DTDX
                    if not (accurate and c > 1.0 / (abs(m) + 0.5))]
        assert runnable, (
            f"M_CASES entry (m={m}, accurate={accurate}) has NO Courant below "
            f"its stability bound {1.0 / (abs(m) + 0.5):.4f}: it would contribute "
            f"zero rows while appearing in the sweep's own record")
    assert any(accurate for _m, accurate in gate.M_CASES)
    # And the one the corpus actually demands: test_pml_cyl idx3 is m = 3 with
    # accurate_fields_near_cylorigin at Courant 1/3.6.
    assert any(abs(c - 1.0 / 3.6) < 1e-12 for c in gate.DTDX), \
        "the m = 3 accurate corpus row's own Courant is not in the sweep"


def test_no_device_step_budget_may_be_claimed_for_this_family():
    """The "no byte-identity claim" is ENFORCED here, not just written down.

    ``KERNEL_CERTIFIED_CONSECUTIVE_STEPS`` stays ``None`` until a device leg has
    actually run and reported a number. Whoever fills it in has to delete this
    test deliberately, which is the point: a budget that appears without a run is
    exactly the claim this round must not make. The number beside it is the
    INHERITED exposure from ``cylindrical_triton`` — a different family's
    measurement, kept separate so it can never be read as ours.
    """
    assert KERNEL_CERTIFIED_CONSECUTIVE_STEPS is None
    assert REFERENCE_SUB_STEPS_PROVEN_ON_NUMPY == 8
    assert INHERITED_FIRST_DIVERGENCE_STEP_TO_WATCH == 23

    # THE ROW COUNTS ARE DERIVED FROM THE GATE'S OWN AXES, never transcribed:
    # the numbers above were 416 and 512 while the module docstring advertised
    # 640 over an m set the harness did not carry, and nothing could tell.
    gate = gate_module()
    assert CONSTITUTIVE_IDENTITY_ROWS == gate.constitutive_identity_row_count()
    expected = 0
    for m, accurate in gate.M_CASES:
        for courant in gate.DTDX:
            if accurate and courant > 1.0 / (abs(m) + 0.5):
                continue
            expected += len(gate.SHAPES) * len(gate.Z_METALLIC) * len(gate.SUB_STEPS)
    assert REFERENCE_ROWS_PROVEN_ON_NUMPY == expected

    # The module's own prose must not have quietly acquired a certification claim.
    text = (PACKAGE_DIR / "cylindrical_complex.py").read_text(encoding="utf-8")
    assert "NOT WIRED" in text and "NO BYTE-IDENTITY CLAIM" in text
    assert "NO DEVICE RUN HAS BEEN TAKEN" in text


def test_the_gate_refuses_to_report_a_device_result_without_a_device():
    """A device leg asked for on a laptop must SKIP, loudly, and measure nothing.

    ``--legs synthetic`` names a device leg only. With no CuPy and no Triton the
    gate must run nothing, say why, and return 2 — never emit an artifact a
    reader would take for a byte-identity result. (The whole-laptop half is
    exercised by the legs' own tests; running ``main([])`` here would re-run it.)
    """
    gate = gate_module()
    available, _why = gate.device_status()
    if available:                                   # pragma: no cover - device host
        pytest.skip("this host HAS cupy+triton; the refusal path is not reachable")
    assert gate.main(["--legs", "synthetic"]) == 2


# ---------------------------------------------------------------------------
# THE DEVICE HALF EXISTS — checked here, because "written" was not true once
# ---------------------------------------------------------------------------
#
# Every claim in this block failed when it was first written against the gate:
# the mutation batteries had no driver, the only function that built or launched
# a plan had no caller, ``main`` routed one flag, and the host battery's entries
# were prose with no callable. None of that is visible from reading the gate's
# docstring, which described all of it as present. These tests are what make the
# difference between written and described checkable at the merge bar.

def test_every_device_leg_named_in_the_gate_is_reachable_from_main():
    """A leg described in prose and unreachable in code has measured nothing."""
    gate = gate_module()
    source = GATE_PATH.read_text(encoding="utf-8")
    for leg in gate.DEVICE_LEGS:
        runner = getattr(gate, "run_" + leg, None)
        assert callable(runner), f"{leg}: no run_{leg} function exists"
        assert f'leg == "{leg}"' in source, \
            f"{leg}: main() never dispatches it, so it is dead code"
    for leg in gate.LAPTOP_LEGS:
        assert leg in source


def test_the_kernel_source_needles_are_armed_on_this_laptop():
    """DISARMED must surface HERE, not after a GPU has been spent.

    Every kernel-source mutation is applied to the shipped kernel's text and has
    to match something and change it. The text is read WITHOUT Triton (the module
    exposes an ``_UnavailableKernel`` with no ``.fn`` here, so the gate falls back
    to the module file's AST), which is the whole reason this can run at the merge
    bar. Nothing in the repository used to apply these transforms at all.
    """
    gate = gate_module()
    source = gate.shipped_kernel_source()
    assert "def cyl_complex_pml_curl_step(" in source
    assert gate.MUTATIONS, "the kernel battery is empty"
    for name, transform, _predicted in gate.MUTATIONS:
        mutated, hits = transform(source)
        assert hits > 0, f"{name}: DISARMED — the pattern matches nothing"
        assert mutated != source, f"{name}: matched but changed nothing"


def test_the_host_needles_are_callables_that_bend_something_real():
    """The host battery used to be ``(name, prose, expectation)`` three-tuples.

    No driver could have applied them: there was no transform to apply. Each entry
    now carries a callable returning ``(patches, overrides)``, and each must
    actually change a patched attribute or a plan argument.
    """
    gate = gate_module()
    context = {"sub_step": "step_B", "half_integer": True, "bcz": 0,
               "expansion": 1, "m": -1, "accurate": False, "dtdx": 0.37}
    assert gate.HOST_MUTATIONS
    for name, description, _predicted, apply in gate.HOST_MUTATIONS:
        assert callable(apply), f"{name}: not a callable"
        assert description
        patches, overrides = apply(module, dict(context))
        changed = any(context.get(key) != value for key, value in overrides.items())
        for obj, attribute, replacement in patches:
            assert getattr(obj, attribute, None) is not None, \
                f"{name}: patches {attribute!r}, which the module does not expose"
            changed = changed or getattr(obj, attribute) is not replacement
        assert changed, f"{name}: DISARMED — it bends nothing"


def test_every_grouping_choice_names_a_needle_that_exists_in_a_battery():
    """The module's nine grouping choices, mapped to needles as DATA.

    Two of the nine used to be pinned by prose alone, and one of those cited a
    word count for a mutation name that appeared in NO battery — a constraint
    documented, measured once by hand, and never armed.
    """
    gate = gate_module()
    assert gate.check_needle_layers() == []
    assert sorted(gate.GROUPING_CHOICE_NEEDLES) == list(range(1, 10))
    layers = gate.needle_layer_map()
    for choice, needles in gate.GROUPING_CHOICE_NEEDLES.items():
        assert needles, f"grouping choice {choice} names no needle"
        for needle in needles:
            assert needle.split("(")[0] in layers, (choice, needle)


def test_the_two_needle_layers_are_a_map_and_the_kernel_side_is_covered():
    """Every KERNEL needle has a same-named REFERENCE needle.

    That is what lets a laptop measurement license what the device leg expects to
    see. The gate used to claim the two batteries "mirror one for one"; they were
    6 against 20, and the mismatch was exactly the dead needle.
    """
    gate = gate_module()
    layers = gate.needle_layer_map()
    kernel = [name for name, where in layers.items() if where["kernel"]]
    assert kernel
    for name in kernel:
        assert layers[name]["reference"], \
            f"{name}: armed on the kernel with nothing on the laptop to say it is live"


@pytest.mark.parametrize("row,expected", [
    ({"covered": False, "launches": 0, "steps": 4}, "NOT-COVERED"),
    ({"covered": True, "launches": 0, "steps": 4}, "NO-LAUNCH"),
    ({"covered": True, "launches": 2, "steps": 4}, "SHORT-LAUNCH"),
    ({"covered": True, "launches": 4, "steps": 4, "moved_words": 0}, "VACUOUS"),
    ({"covered": True, "launches": 4, "steps": 4, "zero_init": True,
      "census_peak": 0}, "VACUOUS"),
    ({"covered": True, "launches": 4, "steps": 4, "moved_words": 9,
      "first_divergence": None}, "IDENTICAL"),
    ({"covered": True, "launches": 4, "steps": 4, "moved_words": 9,
      "first_divergence": {"step": 1}}, "DIVERGED"),
])
def test_the_curl_row_classifier_never_calls_an_unlaunched_row_identical(row,
                                                                        expected):
    """THE defect this classifier exists for, pinned as a pure function.

    ``one_curl_case(route='engine')`` was called without a probe artifact, so the
    shipped predicate refused, ``plan`` was None, the comparison loop was skipped
    entirely and the row recorded ``first_divergence: None`` — which every reader
    and every summary counts as IDENTICAL. A row that launched nothing measured
    nothing and must say so.
    """
    gate = gate_module()
    assert gate.verdict_for_curl_row(row) == expected


def test_the_mutation_classifier_treats_a_stale_binary_as_a_failure():
    """Platform fact (c): Triton's cache can serve a shipped binary to a mutant.

    Both independent checks are pinned — the JIT cache key and, when PTX is
    readable, the mutant's PTX against every shipped specialization's. The gate
    had five mentions of PTX, all in comments, and no implementation.
    """
    gate = gate_module()
    distinct = {"cache_keys_differ": True, "ptx_available": True,
                "ptx_differs_from_every_shipped": True}
    stale_key = {"cache_keys_differ": False, "ptx_available": True,
                 "ptx_differs_from_every_shipped": True}
    stale_ptx = {"cache_keys_differ": True, "ptx_available": True,
                 "ptx_differs_from_every_shipped": False}
    moved = {"random": 12, "zero_init": 0}

    assert gate.classify_mutation("caught", moved, 8, distinct, 1) == "CAUGHT"
    assert gate.classify_mutation("caught", moved, 8, stale_key, 1) == "STALE-BINARY"
    assert gate.classify_mutation("caught", moved, 8, stale_ptx, 1) == "STALE-BINARY"
    assert gate.classify_mutation("caught", moved, 0, distinct, 1) == "NO-LAUNCH"
    assert gate.classify_mutation("caught", moved, 8, distinct, 0) == "DISARMED"
    assert gate.classify_mutation("caught", {"random": 0, "zero_init": 0}, 8,
                                  distinct, 1) == "NEEDLE-MISSED"
    null = ("null", "recorded with its reason")
    assert gate.classify_mutation(null, {"random": 0, "zero_init": 0}, 8,
                                  distinct, 1) == "NULL-AS-PREDICTED"
    assert gate.classify_mutation(null, moved, 8, distinct, 1) == "UNEXPECTEDLY-CAUGHT"
    zero_only = ("caught_zero_init_only", "the ±0 class")
    assert gate.classify_mutation(zero_only, {"random": 0, "zero_init": 16}, 8,
                                  distinct, 1) == "CAUGHT-ZERO-INIT-ONLY"
    assert gate.classify_mutation(zero_only, {"random": 4, "zero_init": 16}, 8,
                                  distinct, 1) == "PREDICTION-BROKEN"
    for status in ("DISARMED", "NO-LAUNCH", "STALE-BINARY", "NEEDLE-MISSED",
                   "UNEXPECTEDLY-CAUGHT", "PREDICTION-BROKEN"):
        assert status in gate.MUTATION_FAILURE_STATUSES


def test_a_mutation_may_not_pass_by_editing_its_own_table():
    """A needle whose pattern only matches inside the mutation TABLE is DISARMED.

    Measured failure mode: after the i*m/r fold was rewritten to call
    ``imr_product``, two entries still carried the old spelling, which still
    occurred in the file — inside the table's own tuple. ``replace(..., 1)`` then
    edited a string literal, the reference was untouched, and the leg reported
    NEEDLE-MISSED. A disarmed needle wearing the mask of a measured null is worse
    than an absent one.
    """
    gate = gate_module()
    source = GATE_PATH.read_text(encoding="utf-8")
    only_in_table = "curl = curl + -(factor * snap[partner])"
    assert only_in_table not in source, \
        "the retired spelling is back; this test's premise needs rechecking"
    assert gate.mutate_region(source, "reference", "NOT-IN-THIS-FILE", "x") is None

    # A pattern placed only in the table region must be refused, not applied.
    marker = gate.TABLE_MARKERS["reference"]
    doctored = source.replace(marker, "SENTINEL_PATTERN_XYZ = 1\n" + marker, 1)
    assert gate.mutate_region(doctored, "reference",
                              "SENTINEL_PATTERN_XYZ", "y") is not None
    tail_only = source[:source.index(marker)] + "SENTINEL_PATTERN_XYZ\n" \
        + source[source.index(marker):]
    assert "SENTINEL_PATTERN_XYZ" in tail_only


def test_a_single_radial_row_is_refused_because_the_array_path_raises_there():
    """OVER-COVERAGE, measured: the predicate admitted a grid stepping REFUSES.

    ``Grid(cell_size=(1, 0, 20), m=1, cylindrical)`` is one radial cell.
    ``stepping.step_B`` raises ``IndexError`` on it — the |m| = 1 axis increment
    reads the first OFF-AXIS row — while the curl predicate returned covered with
    no reasons at all, and the kernel's matching masked load would have read a
    full plane past the end of Ez and stored a silently wrong axis row.
    """
    grid = Grid(resolution=1.0, cell_size=(1.0, 0.0, 20.0), cylindrical=True,
                m=1, boundaries={"z": "metallic"}, courant=0.37)
    assert tuple(grid.shape) == (1, 1, 20)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness={"x": (0, 0), "z": 4})

    with pytest.raises(IndexError):
        stepping.step_B(fields, pml)

    reasons = curl_reasons(fields, pml, "step_B")
    assert any("radial extent" in reason for reason in reasons), reasons
    assert module.plan_cylindrical_complex_curl(fields, pml, "step_B",
                                                probe=PROBE) is None


@pytest.mark.parametrize("m", [1, -1, 2, 3])
def test_the_minimum_radial_clause_admits_two_rows_and_refuses_one(m):
    """The clause is a threshold, not a blanket refusal of small grids."""
    for rows, admitted in ((1, False), (2, True)):
        grid = Grid(resolution=1.0, cell_size=(float(rows), 0.0, 12.0),
                    cylindrical=True, m=m, boundaries={"z": "metallic"},
                    courant=0.2)
        fields = Fields(grid=grid, force_complex_fields=True)
        fields.enable_pml_storage()
        pml = PML(grid=grid, thickness={"x": (0, 0), "z": 3})
        reasons = [r for r in curl_reasons(fields, pml, "step_B")
                   if "radial extent" in r]
        assert bool(reasons) is (not admitted), (m, rows, reasons)


# ---------------------------------------------------------------------------
# The BYTE-INVISIBLE choices — pinned where they can be pinned
# ---------------------------------------------------------------------------
#
# No byte gate can hold these: the gate MEASURED each one at 0 differing words
# and records it as a predicted null with its reason. Claiming the gate holds
# them would be the artifact-overclaim inversion the BFAST tranche names, so they
# are held HERE, by source text, and the module names them as data.

def test_the_byte_invisible_choices_are_named_as_data_and_measured_as_nulls():
    """The module's list and the gate's predicted nulls must agree."""
    gate = gate_module()
    predicted = {name: verdict for name, _t, verdict in gate.MUTATIONS}
    predicted.update({name: verdict
                      for name, _p, _r, verdict in gate.REFERENCE_MUTATIONS})
    assert module.BYTE_INVISIBLE_CHOICES
    for name, reason in module.BYTE_INVISIBLE_CHOICES.items():
        assert reason and len(reason) > 40, f"{name}: no reason recorded"
        assert name in predicted, f"{name}: named invisible but armed nowhere"
        verdict = predicted[name]
        assert isinstance(verdict, tuple) and verdict[0] == "null", \
            f"{name}: named byte-invisible but the battery expects {verdict!r}"


def test_the_zero_cross_terms_of_the_imr_product_survive_in_the_source():
    """Grouping choice: the i*m/r product keeps BOTH arms' cross terms.

    Byte-invisible through this kernel's fold (the gate's minuend census is 0 of
    491,520 words), so it is pinned here. The plane-wise shortcut the module
    refuses would spell the real part as a bare negated product.
    """
    helper = kernel_function_source("_mul_general_coefficient_left")
    assert "tl.math.fma(c_re, z_re, (c_im * z_im) * -1.0)" in helper
    assert "(c_re * z_re) - (c_im * z_im)" in helper
    assert "(c_re * z_im) + (c_im * z_re)" in helper
    # The shortcut, in either arm, would drop the c_re factor entirely.
    assert "(c_im * z_im) * -1.0," in helper or "- (c_im * z_im)" in helper
    for forbidden in ("out_re = (c_im * z_im) * -1.0",
                      "out_im = c_im * z_re"):
        assert forbidden not in helper, f"the plane-wise shortcut is in the kernel: {forbidden}"


def test_the_invariant_axis_difference_is_still_spelled_out_in_the_kernel():
    """Grouping choice 8, pinned by source text because it measures 0 words.

    phi has one cell, so ``(c_p - c)`` is an exact +0.0 addend that an optimizer
    would drop. The array path computes it, so the kernel computes it.
    """
    body = kernel_function_source("cyl_complex_pml_curl_step")
    for line in ("t0_re = ((c_p_re - c_re) + (b_re - b_z_re))",
                 "t0_im = ((c_p_im - c_im) + (b_im - b_z_im))",
                 "t2_re = ((b_r_re - b_re) + (a_re - a_p_re))",
                 "t2_im = ((b_r_im - b_im) + (a_im - a_p_im))"):
        assert line in body, f"the invariant-axis addend was elided: {line}"


def test_the_axis_increment_operand_order_is_the_array_paths_own():
    """Grouping choice 5's B side: ``(-dtdx) * (A - B)``, never ``dtdx * (B - A)``.

    Measured byte-invisible on the B side alone (Bx's dsig axis is phi, whose kms
    never goes negative), which is exactly why it is pinned here AND why the D
    side of the same choice is armed as a catch rather than assumed to follow.
    """
    body = kernel_function_source("cyl_complex_pml_curl_step")
    assert "d_re = b_re - b_z_re" in body and "d_im = b_im - b_z_im" in body
    assert "_mul_coefficient_left(minus_dtdx, d_re, d_im, EXPANSION)" in body
    assert "s_re = (a_re - a_z_re) - two_c_re" in body


def _canned_row(**over):
    row = {"route": "synthetic", "shape": [20, 1, 24], "m": -1, "accurate": False,
           "courant": 0.37, "z": "metallic", "sub_step": "step_B",
           "zero_init": False, "guard": None, "steps": 8, "covered": True,
           "launches": 8, "moved_words": 4096, "first_divergence": None,
           "census_peak": None, "nan_census": {"nan_words": 0, "inf_words": 0},
           "min_bound_kms": -0.82, "signed_zero_reach": None}
    row.update(over)
    if row["zero_init"] and row["signed_zero_reach"] is None:
        row["signed_zero_reach"] = ("live" if row.get("census_peak")
                                    else "unreachable_no_negative_coefficient")
    return row


def test_every_device_driver_actually_iterates_its_cases():
    """The drivers are exercised with the launch stubbed — dead code cannot pass.

    THE DEFECT THIS PINS: the whole device half used to be unreachable. The only
    function that built or launched a plan had no caller, both mutation tables had
    no driver, and ``main`` routed one flag. Nothing failed, because nothing ran.
    Here every driver is called with :func:`one_curl_case` and the Triton-only
    seams replaced, so the test measures that each leg ENUMERATES ITS CASES,
    counts launches, summarises and writes its record.
    """
    gate = gate_module()
    calls = []

    def fake_case(xp, shape, m, accurate, courant, z_metallic, sub_step,
                  zero_init, guard, steps, expansion, kernel=None,
                  route="synthetic", probe=None, host_mutation=None):
        calls.append({"m": m, "sub_step": sub_step, "zero_init": zero_init,
                      "guard": guard, "steps": steps, "route": route,
                      "kernel": kernel, "host_mutation": host_mutation})
        return _canned_row(m=m, sub_step=sub_step, zero_init=zero_init,
                           guard=guard, steps=steps, route=route,
                           shape=list(shape), courant=courant,
                           z="metallic" if z_metallic else "periodic",
                           census_peak=17 if zero_init else None)

    gate.one_curl_case = fake_case
    gate.compile_mutated_kernel = lambda source, entry: types.SimpleNamespace(
        fn=types.SimpleNamespace(__name__=entry), cache_key=entry, cache={})
    gate.mutant_distinctness = lambda kernel: {
        "cache_keys_differ": True, "ptx_available": False,
        "ptx_differs_from_every_shipped": True}
    gate.shared = types.SimpleNamespace(
        CountingKernel=lambda kernel: types.SimpleNamespace(kernel=kernel,
                                                            launches=1))

    results = {}
    synthetic = gate.run_synthetic(None, results, 1, 8)
    assert synthetic["total_rows"] > 100, synthetic["total_rows"]
    assert synthetic["total_launches"] == 8 * synthetic["total_rows"]
    assert synthetic["identical_rows"] == synthetic["total_rows"]
    assert results["synthetic"] is synthetic
    # Both |m| classes, both z terminations, both sub-steps, both seedings and a
    # non-power-of-two Courant have to be in the rows the leg actually generated.
    assert {abs(row["m"]) == 1 for row in synthetic["rows"]} == {True, False}
    assert {row["z"] for row in synthetic["rows"]} == {"metallic", "periodic"}
    assert {row["sub_step"] for row in synthetic["rows"]} == {"step_B", "step_D"}
    assert {row["zero_init"] for row in synthetic["rows"]} == {True, False}
    assert {tuple(row["shape"]) for row in synthetic["rows"]} >= {
        tuple(shape) for shape in gate.SHAPES}

    calls.clear()
    guard = gate.run_guard(None, results, 1, 8)
    assert guard["total_rows"] > 100
    assert all(call["guard"] is True for call in calls), \
        "the fusion-on control must actually pass guard=True"

    calls.clear()
    multi = gate.run_multi_step(None, results, 1)
    assert multi["budget"] == gate.MULTI_STEP_BUDGET
    assert all(call["steps"] == gate.MULTI_STEP_BUDGET for call in calls)
    assert multi["claimable_budget"] == gate.MULTI_STEP_BUDGET

    calls.clear()
    engine = gate.run_engine(None, results, {"backend": "cupy"}, 8)
    assert engine["total_rows"] > 0 and engine["refused_rows"] == 0
    assert all(call["route"] == "engine" for call in calls)

    calls.clear()
    mutations = gate.run_mutations(None, results, 1, 8)
    assert [row["name"] for row in mutations["rows"]] == \
        [name for name, _t, _p in gate.MUTATIONS]
    assert mutations["verdict"].startswith("FAILED"), \
        "a stubbed case moves no bytes, so every 'caught' needle must MISS here"
    assert all(call["kernel"] is not None for call in calls), \
        "the mutant kernel must reach the plan — launch.py:516-522's lesson"

    calls.clear()
    host = gate.run_host_mutations(None, results, 1, 8)
    assert [row["name"] for row in host["rows"]] == \
        [name for name, _d, _p, _a in gate.HOST_MUTATIONS]
    assert all(call["host_mutation"] is not None for call in calls)


def test_a_leg_that_never_launched_is_not_reported_as_a_pass():
    """The summary must count NO-LAUNCH rows as refusals, not as identical."""
    gate = gate_module()
    rows = [_canned_row(covered=False, launches=0),
            _canned_row(launches=0),
            _canned_row()]
    for row in rows:
        row["verdict"] = gate.verdict_for_curl_row(row)
    summary = gate._summarize_rows(rows)
    assert summary["identical_rows"] == 1
    assert summary["by_verdict"]["NOT-COVERED"] == 1
    assert summary["by_verdict"]["NO-LAUNCH"] == 1


def test_the_device_legs_refuse_a_probe_cut_under_the_wrong_policy(tmp_path):
    """EXPANSION comes from a MEASURED artifact, and the policy is part of it.

    A device leg that guesses the constexpr is not a byte gate, and one that
    accepts a probe cut under CuPy's default ``-ftz=true`` certifies the wrong
    binaries. Both refusals are reasons, not exceptions, so the artifact records
    why nothing was measured.
    """
    gate = gate_module()
    shared = sys.modules["gate_triton_complex"]

    record, reasons = gate.probe_record(None)
    assert record is None and reasons and "may not be guessed" in reasons[0]

    # ``resolved`` and ``candidates`` are what the real cutter writes
    # (gate_triton_complex.policy_stamp / measure_expansion_record). This fixture
    # carried only the policy NAME and its strip counters, which since 2026-08-15
    # is a refusal in its own right: the licence rule requires every record to say
    # which policy its bytes resolved to AND that its candidate arms were cut
    # under the same one, on every path rather than only when something is
    # ambiguous. An abbreviated fixture must not be the thing that proves a
    # correct-policy artifact passes.
    good = {"backend": "cupy",
            "patterns": {name: "FMA_V1" for name in complex_module.PROBE_PATTERNS},
            "subnormal_policy": {"policy": shared.STRIPPED_POLICY,
                                 "resolved": "keep",
                                 "nvrtc_calls": 3, "ftz_removed": 3},
            "candidates": {"policy": "keep"}}
    stamp_probe_record(good, policy=shared.STRIPPED_POLICY, resolved="keep")
    good["subnormal_policy"].update(nvrtc_calls=3, ftz_removed=3)
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    record, reasons = gate.probe_record(str(path))
    assert reasons == [] and complex_module.expansion_from_probe(record) is not None

    wrong = dict(good, subnormal_policy={"policy": "cupy_default_ftz"})
    path.write_text(json.dumps(wrong), encoding="utf-8")
    record, reasons = gate.probe_record(str(path))
    assert record is None and reasons, "a wrong-policy probe must be refused"


def test_the_gate_carries_no_function_or_table_that_nothing_reaches():
    """DEAD CODE IS THE DEFECT THIS FAMILY WAS CAUGHT WITH.

    ``one_curl_case``, ``MUTATIONS``, ``HOST_MUTATIONS``, ``BENCH_SHAPE`` and
    ``DEFAULT_STEPS`` all had exactly one reference — their own definition — while
    the gate's docstring described the legs they belong to as written. This walks
    the gate's AST and requires every module-level function, class and constant to
    be referenced somewhere, with ONE declared exception:
    ``curl_from_invariant_elided``, which is reached only through a mutation's
    replacement TEXT and is checked by name below.
    """
    gate = gate_module()
    source = GATE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = [node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    constants = [target.id for node in tree.body
                 if isinstance(node, (ast.Assign, ast.AnnAssign))
                 for target in ([node.target] if isinstance(node, ast.AnnAssign)
                                else node.targets)
                 if isinstance(target, ast.Name)]
    used = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used[node.id] = used.get(node.id, 0) + 1
        elif isinstance(node, ast.Attribute):
            used[node.attr] = used.get(node.attr, 0) + 1

    text_only = {"curl_from_invariant_elided"}
    dead = [name for name in names if used.get(name, 0) == 0
            and name not in text_only]
    dead += [name for name in constants if used.get(name, 0) <= 1
             and name not in text_only]
    assert dead == [], f"the gate carries unreachable code: {dead}"

    # The one exception, reached through a mutation's replacement text.
    replacements = [entry[2] for entry in gate.REFERENCE_MUTATIONS]
    assert any("curl_from_invariant_elided" in text for text in replacements)
    assert hasattr(gate, "curl_from_invariant_elided")


# ---------------------------------------------------------------------------
# The two device-side obligations the batteries did not carry
# ---------------------------------------------------------------------------

def test_the_per_seeding_vacuity_floor_names_a_seeding_that_caught_nothing():
    """A SEEDING that catches nothing across a whole battery is VACUOUS.

    The two reference-side batteries carried this floor and the two DEVICE
    batteries did not: every individual row could report CAUGHT on its random
    column while the zero-init column was all zeros, and the battery's verdict
    would read "all needles live". The function is pure, so the branch the
    device leg decides on is driven here rather than on a GPU.
    """
    gate = gate_module()

    live = {"rows": [{"status": "CAUGHT", "differing_words":
                      {"random": 12, "zero_init": 4}}]}
    totals, vacuous = gate._per_seeding_vacuity(live)
    assert totals == {"random": 12, "zero_init": 4} and vacuous == []
    assert live["per_seeding_total_words"] == totals
    assert live["vacuous_seedings"] == []

    half = {"rows": [{"status": "CAUGHT", "differing_words":
                      {"random": 12, "zero_init": 0}}]}
    totals, vacuous = gate._per_seeding_vacuity(half)
    assert vacuous == ["zero_init"], totals

    # A PREDICTED NULL contributes nothing and must not be counted as evidence
    # that a seeding is live: a battery of nothing but nulls has no floor.
    nulls = {"rows": [{"status": "NULL-AS-PREDICTED", "differing_words":
                       {"random": 0, "zero_init": 0}}]}
    totals, vacuous = gate._per_seeding_vacuity(nulls)
    assert sorted(vacuous) == ["random", "zero_init"] and totals == {
        "random": 0, "zero_init": 0}


def test_the_nan_census_counts_both_non_finite_classes():
    """Platform fact (f): a NaN's sign and payload are IEEE-unspecified, so a
    raw uint32 compare over one is not a measurement. The census is what lets
    the artifact say so rather than compare through it."""
    gate = gate_module()

    clean = np.ones((2, 1, 3), dtype=np.complex64)
    assert gate.nan_census({"Bx": clean}) == {"nan_words": 0, "inf_words": 0}

    dirty = np.ones((2, 1, 3), dtype=np.complex64)
    dirty[0, 0, 0] = complex(np.nan, 0.0)
    dirty[0, 0, 1] = complex(0.0, np.inf)
    census = gate.nan_census({"Bx": dirty, "By": None})
    assert census == {"nan_words": 1, "inf_words": 1}


def test_the_minuend_census_names_the_backend_it_was_cut_on():
    """The predicted null's evidence must say WHERE it was measured.

    ``imr_planewise_zero_cross_terms`` is a null about DEVICE bytes; a census
    cut on NumPy is evidence about NumPy arithmetic. The record now carries the
    backend so the two cannot be read as one number.
    """
    gate = gate_module()
    record = gate.run_minuend_census(1)
    assert record["backend"] == "numpy"
    assert record["words_scanned"] > 0
    assert record["negative_zero_minuend_words"] == 0, record


# ---------------------------------------------------------------------------
# The composition probe's enumeration, checked without a device
# ---------------------------------------------------------------------------

def composition_module():
    """Import the composition probe by path (``parity/`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "_probe_cylindrical_complex_composition", COMPOSITION_PATH)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_the_composition_cases_span_both_m_classes_and_both_z_terminations():
    """The corpus demands both of each, so the composition may not carry one.

    |m| = 1 has the axis-row increments and no near-axis zeroing; |m| >= 2 has
    the zeroing and no increments; thirteen lifted rows terminate z METALLIC and
    three PERIODIC. A composition that swept one class would certify half the
    family's additions through a complete driver step and none of the other.
    """
    probe = composition_module()
    classes = {module.m_class(case["m"]) for case in probe.CASES}
    assert classes == {module.m_class(0), module.m_class(1), module.m_class(3)}, \
        classes
    # The m = 0 arm (2026-09-04): a sourced case, the corpus row's own shape and
    # termination, and a +-0 lattice case, all under complex storage.
    zero = [case for case in probe.CASES if case["m"] == 0]
    assert len(zero) >= 3, [case["name"] for case in zero]
    assert any(case["sources"] != "none" for case in zero)
    assert any(case["seeding"] == "signed_zero_lattice" for case in zero)
    corpus = [case for case in zero if case.get("courant_is_the_corpus_rows_own")]
    assert len(corpus) == 1 and corpus[0]["cell"] == (3.0, 0.0, 6.0) \
        and corpus[0]["resolution"] == 50.0 and corpus[0]["courant"] == 0.5 \
        and corpus[0]["z_boundary"] == "metallic", corpus
    assert {case["z_boundary"] for case in probe.CASES} == {"metallic",
                                                            "periodic"}
    assert any(case["accurate"] for case in probe.CASES), \
        "no case carries accurate_fields_near_cylorigin, where ZERO_ROWS is 1"
    assert any(case["m"] < 0 for case in probe.CASES), \
        "no case carries a NEGATIVE m; seven of the sixteen lifted rows do"
    # The |m| >= 2 cases must not all share one zero-row count, or ZERO_ROWS is
    # a constexpr the composition never varied.
    counts = {module.zero_rows(case["m"], case["accurate"])
              for case in probe.CASES}
    assert len(counts) >= 3, counts


def test_every_composition_case_carries_a_non_power_of_two_courant():
    """At 0.5 the dt/dx scaling is exact in binary and the associativity
    discrepancies vanish; 0.5 alone certifies broken kernels."""
    probe = composition_module()
    for case in probe.CASES:
        scaled = round(case["courant"] * 2 ** 8)
        power_of_two = (abs(case["courant"] * 2 ** 8 - scaled) < 1e-12
                        and scaled and (scaled & (scaled - 1)) == 0)
        if case.get("courant_is_the_corpus_rows_own"):
            # The ONE exception, and it says so: the corpus row's own Courant
            # (0.5). Every other case of its m class carries a non-power-of-two
            # one, asserted below.
            assert power_of_two and case["courant"] == 0.5, case["name"]
            others = [c for c in probe.CASES
                      if c["m"] == case["m"] and c is not case]
            assert others, "the corpus-Courant case is its m class's only case"
            continue
        assert not power_of_two, f"{case['name']} has Courant {case['courant']}"


def test_the_composition_mutations_are_scheduled_where_they_can_bite():
    """Two of the three needles only exist in one |m| class, and a needle armed
    on the wrong case is a null that reads as a measurement."""
    probe = composition_module()
    by_name = {case["name"]: case for case in probe.CASES}
    for name, case_name, apply, expected_step, why in probe.MUTATIONS:
        assert case_name in by_name, f"{name} names no case"
        assert callable(apply) and why
        assert expected_step >= 1, f"{name} declares no catch step"
        case = by_name[case_name]
        assert expected_step < case["steps"], (
            f"{name} declares a catch at step {expected_step} on a case that "
            f"runs {case['steps']}; the leg must have headroom to observe a "
            f"LATER catch, or 'caught at the declared step' is unfalsifiable")
        if name == "axis_increment_negated":
            assert module.m_class(case["m"]) == module.m_class(1)
            # The declared step is a STRUCTURAL fact about the driver's order,
            # so it is pinned here as well as in the probe: the B-side scalars
            # are the only thing this needle bends, and E is exactly +0.0 at
            # every run's first step_B because update_E runs at the END.
            assert expected_step == 2, expected_step
        if name == "zero_rows_reduced":
            assert module.zero_rows(case["m"], case["accurate"]) >= 1


def test_the_axis_increment_scalars_are_the_b_side_pair_only():
    """The composition's step-2 declaration rests on this, so it is pinned.

    ``axis_increment_scalars`` returns the B-side ``(-dtdx, 1j*m*dtdx)``; the
    D-side arm takes the plain ``dtdx`` the whole curl already carries. If a
    later change routed the D arm through these scalars the needle would become
    live at step 1 and the composition's exact-step assertion would fail — which
    is the intended outcome, but the reason belongs beside the signature.
    """
    source = gate_module().shipped_kernel_source()
    assert "unused on D" in source, \
        "the kernel no longer declares minus_dtdx unused on the D side"
    assert "minus_dtdx" in source


def test_the_composition_serves_the_four_core_slots_in_driver_order():
    """The install is a monkeypatch of the driver's own sub-step functions, so
    the slot names and their ORDER are a contract with driver.py:3205-3228."""
    probe = composition_module()
    assert probe.SLOTS == ("step_B", "update_H", "step_D", "update_E")
    driver_source = (PACKAGE_DIR.parent / "driver.py").read_text(encoding="utf-8")
    positions = [driver_source.index(f"        {slot}(self.fields")
                 for slot in probe.SLOTS]
    assert positions == sorted(positions), \
        "the driver no longer calls the four slots in this order"


def test_the_composition_probe_does_not_wire_production_dispatch():
    """Dispatch stays disabled: the probe may monkeypatch the driver module for
    ONE fields object and may not touch the fast path or the launch planner.

    Checked on the AST rather than on the text, because the docstring NAMES the
    hooks it does not touch — a text scan would refuse the very sentence that
    records the decision.
    """
    tree = ast.parse(COMPOSITION_PATH.read_text(encoding="utf-8"))
    banned = {"plan_fast_path", "fastpath", "plan_step",
              "invalidate_fast_path"}
    reached = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    reached |= {node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)}
    reached |= {alias.name.split(".")[-1] for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert not (banned & reached), \
        f"the composition probe reaches production dispatch: {banned & reached}"
