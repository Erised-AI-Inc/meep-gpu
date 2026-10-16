"""Tests for the COMPLEX cylindrical (Dcyl, complex64) Triton family -- host only.

Everything here runs with no GPU, no CuPy and no Triton. What needs hardware -- byte
identity of ``cyl_complex_pml_curl_step`` and the shared radial scan against
``stepping.step_B`` / ``step_D``, the kernel-source mutations, the contraction control --
lives in ``parity/meep_gpu/gate_triton_cylindrical_complex.py``, whose reference is the
array path itself (twin ``Fields``, one stepped by ``stepping``, one by the plan).

What is pinned HERE is everything that decides whether the kernel is ever ALLOWED to run,
plus the host-built numbers it is handed, which would be wrong SILENTLY rather than
loudly:

* **the two predicates**, clause by clause, each shown load-bearing by breaking exactly
  one thing on an otherwise admitted real ``Fields``/``PML``/``Grid``;
* **disjointness with ``cylindrical_triton`` by storage**, in both directions;
* **the host-built rows and words**, byte-compared THROUGH ``stepping``'s own functions:
  the i*m/r pass's real ``rinv`` row (a float32 quotient), the |m| = 1 axis increments
  (a double product with the double Courant) and the radial scan's bindings;
* **the tables** -- the i*m/r terms, the axis increment's target, the radial-sum source
  and ``ir0`` -- read back off ``stepping``'s own tables;
* **the plan shape**, built through the harness ``from_arrays`` route on NumPy arrays;
* **the gate's structure**: its cases, its mutation table's needles against the shipped
  text, its legs reachable from ``main``;
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

PACKAGE_DIR = pathlib.Path(module.__file__).parent
API_DIR = PACKAGE_DIR.parent.parent
GATE_PATH = API_DIR / "parity" / "meep_gpu" / "gate_triton_cylindrical_complex.py"
COMPOSITION_PATH = (API_DIR / "parity" / "meep_gpu"
                    / "probe_triton_cylindrical_complex_composition.py")


#: r-high plus both z faces, in CELLS — what a lifted Dcyl script actually asks
#: for. There is no absorber at r = 0 and none on phi, which has one cell.
DCYL_PML_CELLS = {"x": (0, 4), "z": 4}

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


def curl_reasons(fields, pml, sub_step="step_B"):
    """Curl-predicate reasons, minus the CuPy-backend one every laptop run carries."""
    verdict = module.cylindrical_complex_curl_coverage(fields, pml, sub_step)
    return [r for r in verdict.reasons if "not cupy" not in r]


def constitutive_reasons(fields, pml, side="E"):
    verdict = module.cylindrical_complex_constitutive_coverage(fields, pml, side)
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
    verdict = reloaded.cylindrical_complex_curl_coverage(fields, pml, "step_B")
    assert verdict.covered is False                 # NumPy backend, and it says so
    assert any("not cupy" in reason for reason in verdict.reasons)
    assert reloaded.plan_cylindrical_complex_curl(fields, pml, "step_B") is None
    assert reloaded.plan_cylindrical_complex_constitutive(fields, pml, "E") is None
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


def test_m_class_carries_the_m_zero_arm():
    """m = 0 is the THIRD arm since 2026-09-04 (complex storage), not a refusal.

    The disjointness story moved from m to STORAGE: the real m = 0 run is still
    ``cylindrical_triton``'s and is refused here on ``force_complex_fields``
    (see ``test_the_two_cylindrical_products_are_disjoint_by_storage``).
    """
    assert module.m_class(0) == 0 == constexpr_value(module.M_ZERO)
    assert module.zero_rows(0, False) == 0 == module.zero_rows(0, True)


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
    verdict = module.cylindrical_complex_curl_coverage(fields, pml, sub_step)
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
        module.cylindrical_complex_curl_coverage(fields, pml, "step_E")
    with pytest.raises(ValueError, match="side must be one of"):
        module.cylindrical_complex_constitutive_coverage(fields, pml, "B")
    with pytest.raises(ValueError, match="sub_step must be one of"):
        module.plan_cylindrical_complex_curl(fields, pml, "step_E")
    with pytest.raises(ValueError, match="side must be one of"):
        module.plan_cylindrical_complex_constitutive(fields, pml, "B")


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
    verdict = complex_module.complex_pml_curl_coverage(fields, pml, "step_B")
    assert verdict.covered is False
    assert any("cylindrical" in r.lower() for r in verdict.reasons), verdict.reasons

    code = code_of(PACKAGE_DIR / "cylindrical_complex.py")
    assert "_grid_reasons" not in code, \
        "this module composes a blanket clause that refuses its own whole domain"
    assert "_complex_grid_reasons" not in code


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


class NumpyClaimingCupy:
    """NumPy wearing CuPy's name, so the ENGINE route can be built on a laptop.

    The backend clause is the only thing standing between an admitted Dcyl
    configuration and a real plan here, and it is the one clause that cannot be
    satisfied on this box. Delegating every attribute to NumPy keeps the plan
    builder doing real work — ``imr_rinv_row`` really runs its ``arange``,
    ``maximum`` and float32 quotient — while ``__name__`` answers the predicate.

    This is a TEST-ONLY fiction and it does NOT make anything here a device
    result: ``run()`` is never called, no kernel is compiled and no bytes are
    compared against hardware.
    """

    __name__ = "cupy"

    def __getattr__(self, name):
        return getattr(np, name)



def test_the_engine_route_still_refuses_a_real_numpy_backend():
    """The backend clause is what :class:`NumpyClaimingCupy` bypasses, so it is
    checked here separately — otherwise the fiction could hide its removal."""
    fields, pml = build()
    assert module.plan_cylindrical_complex_curl(fields, pml, "step_B") is None
    verdict = module.cylindrical_complex_curl_coverage(fields, pml, "step_B")
    assert any("not cupy" in r for r in verdict.reasons)


def test_the_engine_route_refuses_to_none_and_never_raises():
    """None is the ONLY refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly."""
    fields, pml = build()
    for sub_step in ("step_B", "step_D"):
        assert module.plan_cylindrical_complex_curl(fields, pml, sub_step) is None
    for side in ("H", "E"):
        assert module.plan_cylindrical_complex_constitutive(fields, pml, side) is None


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
    assert module.plan_cylindrical_complex_curl(fields, pml, "step_B") is None


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


def test_the_module_imports_no_sibling_kernel_module():
    """Module independence, enforced rather than agreed.

    The family imports shared helpers READ-ONLY: the complex base, the shared coverage
    and launch interop, and the radial scan's one home (:mod:`.cylindrical_triton`).
    No other sibling kernel module may be imported.
    """
    path = PACKAGE_DIR / "cylindrical_complex.py"
    code = code_of(path)
    for banned in ("fastpath", "cuda_kernels", "backends"):
        assert banned not in code, f"cylindrical_complex.py references {banned}"
    imports = imported_names(path)
    for banned in ("folded_complex", ".folded_complex", "special_kz", ".special_kz",
                   "bfast_curl", ".bfast_curl", "nonlinear_update_e",
                   ".nonlinear_update_e", "offdiag_update_e", ".offdiag_update_e",
                   "symmetry", ".symmetry", "conductivity", ".conductivity"):
        assert banned not in imports, f"cylindrical_complex.py imports {banned}"
    assert {"ComplexConstitutivePlan", "_word_view", "Coverage", "_boundary_kinds",
            "CupyPointer", "_flat", "RadialScan", "radial_scan_starts",
            "courant_constants"} <= imports


def test_no_negation_is_spelled_with_a_unary_minus_and_no_product_is_complex():
    """Triton lowers ``-x`` as ``0.0 - x``, which canonicalizes signed zeros, so every
    negation in the kernel is ``* -1.0``. And no complex product, no multiply-add: every
    product has a real operand and is formed per part."""
    tree = ast.parse((PACKAGE_DIR / "cylindrical_complex.py").read_text(encoding="utf-8"))
    kernels = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "cyl_complex_pml_curl_step"]
    assert len(kernels) == 1, "the kernel body moved; this check is now blind"
    for node in ast.walk(kernels[0]):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            assert isinstance(node.operand, ast.Constant), (
                f"a negation is spelled with unary minus at line {node.lineno}")
    body = kernel_function_source("cyl_complex_pml_curl_step")
    for banned in ("tl.math.fma", "_mul_coefficient_left", "_mul_field_left",
                   "_mul_general_coefficient_left", "EXPANSION"):
        assert banned not in body, banned


def test_the_imr_call_sites_signs_and_partners_are_steppings_own():
    """Four i*m/r terms and only four, read off ``stepping``'s own pass tables.

    ``stepping._CYLINDRICAL_IMR_B`` / ``_D`` hold (curl term, partner, sign) for the pass
    MEEP runs after every component's curl (step_db.cpp:177-294). The module compiles in
    (target index, partner REGISTER, sign): the register names which centre load of the
    curl the term reuses, so no new pointer and no extra traffic. Target 1 (By / Dy) has
    no term on either side.
    """
    for sub_step, table, terms, sources in (
            ("step_B", stepping._CYLINDRICAL_IMR_B, stepping.B_CURL_TERMS, ("Ex", "Ey", "Ez")),
            ("step_D", stepping._CYLINDRICAL_IMR_D, stepping.D_CURL_TERMS, ("Hx", "Hy", "Hz"))):
        targets = tuple(term.target for term in terms)
        expected = tuple((targets.index(term.target), "abc"[sources.index(partner)], sign)
                         for term, partner, sign in table)
        assert module.IMR_TERMS[sub_step] == expected
        assert 1 not in [index for index, _register, _sign in module.IMR_TERMS[sub_step]]


def test_the_imr_partners_are_centre_loads_the_curl_already_makes():
    """``c`` = g2 = Ez/Hz and ``a`` = g0 = Ex/Hx -- never ``b``, whose only uses in the
    curl are shifted; the kernel names exactly those registers in the pass."""
    for sub_step in ("step_B", "step_D"):
        assert {r for _i, r, _s in module.IMR_TERMS[sub_step]} == {"a", "c"}
    body = kernel_function_source("cyl_complex_pml_curl_step")
    assert "i0_re = (r0 * c_im) * -1.0" in body and "i0_im = r0 * c_re" in body
    assert "i2_re = (r2 * a_im) * -1.0" in body and "i2_im = r2 * a_re" in body


@pytest.mark.parametrize("m", [1, -1, 2, -3, 5])
@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_imr_coefficient_row_reproduces_steppings_own_term_bytewise(m, sub_step):
    """End-to-end byte identity THROUGH ``stepping._cylindrical_imr_increment``.

    The kernel's increment is the quarter turn of the partner by the real row,
    ``(-(rinv*g.im), rinv*g.re)``; rebuilt here from :func:`imr_rinv_row` and compared
    with the array path's increment on a random complex partner, so every word of the
    row is observable. The row is a float32 QUOTIENT (``the_m / here_r``, both realnum),
    built on the host, so the kernel performs no division.
    """
    fields, _pml = build(m=m, courant=0.3)
    rng = np.random.default_rng(4242)
    shape = fields.grid.shape
    partner = (rng.standard_normal(shape).astype(np.float32)
               + 1j * rng.standard_normal(shape).astype(np.float32)).astype(np.complex64)
    targets = stepping.B_CURL_TERMS if sub_step == "step_B" else stepping.D_CURL_TERMS
    for index, _register, sign in module.IMR_TERMS[sub_step]:
        target = targets[index].target
        expected = stepping._cylindrical_imr_increment(fields, target, partner, sign)
        row = module.imr_rinv_row(target, sign, m, fields.grid.courant, shape[0])
        assert row.dtype == np.float32 and row.shape == (shape[0],)
        rinv = row.reshape(-1, 1, 1)
        rebuilt = np.empty_like(partner)
        rebuilt.real[...] = (rinv * partner.imag) * np.float32(-1.0)
        rebuilt.imag[...] = rinv * partner.real
        assert same_bytes(expected, rebuilt), \
            f"{sub_step} {target} sign={sign:+g} m={m}: rinv row differs"


@pytest.mark.parametrize("m,dtdx", [(1, 0.37), (-1, 0.37), (1, 0.5),
                                    (-1, 0.3141592653589793)])
def test_the_axis_increment_scalars_reproduce_steppings_increment_bytewise(m, dtdx):
    """The |m| = 1 B-side increment, rebuilt the KERNEL's way and byte-compared.

    The kernel forms, per part, ``bracket = (Ep[0] - Ep[0, z+1]) - partner`` with
    ``partner = (m*Ez[1].im, -(m*Ez[1].re))`` and multiplies it in DOUBLE by the double
    ``-Courant`` it reads from :func:`cylindrical_triton.courant_constants`, rounding
    once -- ``stepping._cylindrical_axis_increment_B``'s ``_realnum_double_product``.
    """
    from meep_gpu.triton_kernels import cylindrical_triton as cyl

    fields, pml = build(m=m, courant=dtdx)
    grid = fields.grid
    rng = np.random.default_rng(99)
    for name in ("Ex", "Ey", "Ez"):
        host = (rng.standard_normal(grid.shape).astype(np.float32)
                + 1j * rng.standard_normal(grid.shape).astype(np.float32))
        getattr(fields, name)[...] = host.astype(np.complex64)
    boundaries = stepping._boundary_kinds(grid, pml)
    phases = stepping._bloch_phases(grid, boundaries, fields.Ex, pml)
    electric = stepping._component_snapshot(fields, ("Ex", "Ey", "Ez"))
    target, expected = stepping._cylindrical_axis_increment_B(
        fields, electric, boundaries, phases, grid.dt / grid.dx)
    assert target == "Bx"
    factor = cyl.courant_constants(grid.courant)[cyl.COURANT_WORDS["minus"]]
    ep = electric["Ey"][stepping._face(0, 0)]
    ep_above = stepping._shift_up(np, electric["Ey"], 2, boundaries[2], phases[2])[
        stepping._face(0, 0)]
    e1 = np.take(electric["Ez"], 1, axis=0)
    mf = np.float32(m)
    rebuilt = np.empty_like(expected)
    for part, partner in (("real", mf * e1.imag), ("imag", (mf * e1.real) * np.float32(-1.0))):
        bracket = (getattr(ep, part) - getattr(ep_above, part)) - partner
        getattr(rebuilt, part)[...] = (bracket.astype(np.float64) * factor).astype(np.float32)
    assert same_bytes(expected, rebuilt)


def test_the_m_zero_axis_add_is_a_double_product_in_the_kernel():
    """``Dz[0] += realnum(double(4*Courant) * double(Hp[0]))`` per part, the double words
    read by pointer; never an fp32 scalar argument."""
    body = kernel_function_source("cyl_complex_pml_curl_step")
    assert "four = tl.load(courant + COURANT_FOUR)" in body
    assert "h_re = tl.cast(tl.cast(b_re, tl.float64) * four, tl.float32)" in body
    assert "four_dtdx" not in body


def test_the_axis_increment_operand_order_is_the_array_paths_own():
    """``(Ep - Ep_above) - partner`` on B and ``(Hr - Hr_below) - 2*Hz`` on D, each
    negated by ``* -1.0`` when it replaces curl row 0 -- ``-(A - B)`` is not ``B - A`` on
    +0 operands, so the order is load-bearing."""
    body = kernel_function_source("cyl_complex_pml_curl_step")
    assert "s_re = (b_re - b_z_re) - partner_re" in body
    assert "s_re = (a_re - a_z_re) - (2.0 * c_re)" in body
    assert "curl0_re = tl.where(at_r, inc_re * -1.0, curl0_re)" in body
    assert "curl1_re = tl.where(at_r, inc_re * -1.0, curl1_re)" in body


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
@pytest.mark.parametrize("m", [-1, 3])
def test_the_prefix_is_byte_identical_to_the_one_the_array_path_builds(sub_step, m):
    """The radial scan's bindings are the array path's own, on complex storage.

    The rows are ``stepping._cylindrical_rderiv_rows`` over the side's source rows (Ep
    with its wall row on B, Hp on D), the starts line is
    ``stepping._cylindrical_scan_starts``, and a complex64 source is scanned as float32
    words -- one column per stored part, which IS the array path's per-part sum.
    """
    from meep_gpu.triton_kernels import cylindrical_triton as cyl

    fields, pml = build(m=m, courant=0.3)
    scan = module.RadialScan(sub_step, getattr(fields, module.PREFIX[sub_step]["component"]),
                             cyl.radial_scan_starts(fields, pml, sub_step, None), np)
    rows = fields.grid.shape[0]
    wall = 1 if module.PREFIX[sub_step]["extend_wall_row"] else 0
    weights, rinv = stepping._cylindrical_rderiv_rows(
        np, rows + wall, module.PREFIX[sub_step]["ir0"], np.dtype(np.float32))
    assert same_bytes(scan._w, weights.reshape(-1))
    assert same_bytes(scan._rinv, rinv.reshape(-1))
    target = "Bz" if sub_step == "step_B" else "Dz"
    expected = stepping._cylindrical_scan_starts(fields, pml, target)
    assert np.array_equal(scan.starts, np.asarray(expected).reshape(-1))
    assert scan.ncol == 2 * fields.grid.shape[1] * fields.grid.shape[2]
    assert scan.out.dtype == np.complex64 and scan.out.shape == fields.Bz.shape


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
    """``plan_cylindrical_complex_curl`` -- the route the driver takes -- DERIVES its
    constexprs: ``BCZ`` from the resolved boundary triple, ``M_CLASS`` and ``ZERO_ROWS``
    from m and the near-axis policy, the coefficient sub-lattice from the sub-step, the
    i*m/r rows and the Courant words from the grid, the chunk record and the scan's starts
    from the run's division."""
    from meep_gpu.triton_kernels import cylindrical_triton as cyl
    from meep_gpu.triton_kernels.launch import chunk_bounds_for

    fields, pml = build(m=m, accurate=accurate, courant=courant, z_metallic=z_metallic)
    claims_cupy = Override(fields, grid=Override(fields.grid, xp=NumpyClaimingCupy()))
    plan = module.plan_cylindrical_complex_curl(claims_cupy, pml, sub_step)
    assert plan is not None, "the engine route refused an admitted configuration"
    assert plan.bcz == expected_bcz
    assert plan.m_class == expected_class
    assert plan.zero_rows == expected_zero_rows
    assert plan.sub_step == sub_step
    assert plan.shape == tuple(fields.grid.shape)
    assert plan.dtdx == float(fields.grid.dt / fields.grid.dx)
    assert plan.courant == float(fields.grid.courant)
    expected_coefficients = [getattr(pml, f"{stem}_{axis}{suffix}").reshape(-1)
                             for axis in "xyz" for stem in ("kms", "sinv")]
    assert len(plan._coefficients) == 6
    for pointer, wanted in zip(plan._coefficients, expected_coefficients):
        assert same_bytes(pointer.array, wanted)
    targets = stepping.B_CURL_TERMS if sub_step == "step_B" else stepping.D_CURL_TERMS
    assert len(plan._imr_rows) == 2
    for pointer, (index, _register, sign) in zip(plan._imr_rows, module.IMR_TERMS[sub_step]):
        assert same_bytes(pointer.array, module.imr_rinv_row(
            targets[index].target, sign, m, fields.grid.courant, fields.grid.shape[0]))
    assert same_bytes(plan._courant.array, cyl.courant_constants(fields.grid.courant))
    assert np.array_equal(plan.chunk_bounds, chunk_bounds_for(fields, pml, None))
    assert np.array_equal(plan.scan.starts,
                          cyl.radial_scan_starts(fields, pml, sub_step, None))


@pytest.mark.parametrize("sub_step,backward", [("step_B", 0), ("step_D", 1)])
@pytest.mark.parametrize("m,accurate,expected_class,expected_zero_rows",
                         [(1, False, 1, 0), (-1, False, 1, 0), (2, False, 2, 2),
                          (3, False, 2, 3), (3, True, 2, 1), (5, False, 2, 5)])
@pytest.mark.parametrize("z_metallic,bcz", [(True, 1), (False, 0)])
def test_the_plan_carries_the_constexprs_the_grid_implies(sub_step, backward, m,
                                                          accurate, expected_class,
                                                          expected_zero_rows,
                                                          z_metallic, bcz):
    """Built through the harness ``from_arrays`` route, on NumPy arrays: ``CupyPointer``
    defers the device address until launch, so the plan's shape is host-checkable."""
    from meep_gpu.triton_kernels import cylindrical_triton as cyl
    from meep_gpu.triton_kernels.launch import chunk_bounds_for

    courant = 1 / 3.6 if accurate else 0.37
    fields, pml = build(m=m, accurate=accurate, courant=courant, z_metallic=z_metallic)
    plan = module.plan_cylindrical_complex_curl_from_arrays(
        sub_step, arrays_for(fields, sub_step), flat_for(pml, sub_step),
        fields.grid.dt / fields.grid.dx, fields.grid.courant, m, accurate, bcz, np,
        chunk_bounds=chunk_bounds_for(fields, pml, None),
        scan_starts=cyl.radial_scan_starts(fields, pml, sub_step, None))
    assert plan.sub_step == sub_step
    assert plan.backward == backward
    assert plan.bcz == bcz
    assert plan.m_class == expected_class
    assert plan.zero_rows == expected_zero_rows
    assert plan.shape == tuple(fields.grid.shape)
    assert plan.n_elem == int(np.prod(fields.grid.shape))
    assert plan.block == module.DEFAULT_BLOCK
    assert plan._grid == ((plan.n_elem + plan.block - 1) // plan.block,)
    assert len(plan._imr_rows) == len(module.IMR_TERMS[sub_step]) == 2
    assert plan.scan.sub_step == sub_step


@pytest.mark.parametrize("sub_step", ["step_B", "step_D"])
def test_the_plans_prefix_is_the_array_paths_own_scan(sub_step):
    """The plan launches the radial scan on every run, before the curl, on its bound
    source, into the one output volume the curl reads -- the scan reads the CURRENT
    sources, which the previous sub-step wrote, so nothing may be cached."""
    from meep_gpu.triton_kernels import cylindrical_triton as cyl
    from meep_gpu.triton_kernels.launch import chunk_bounds_for

    fields, pml = build(m=-1)
    arrays = arrays_for(fields, sub_step)
    plan = module.plan_cylindrical_complex_curl_from_arrays(
        sub_step, arrays, flat_for(pml, sub_step), 0.37, 0.37, -1, False, 1, np,
        chunk_bounds=chunk_bounds_for(fields, pml, None),
        scan_starts=cyl.radial_scan_starts(fields, pml, sub_step, None))
    source = arrays[module.PREFIX[sub_step]["component"]]
    assert np.shares_memory(plan.scan._source, source)
    run = code_of(PACKAGE_DIR / "cylindrical_complex.py").split(
        "class CylindricalComplexCurlPlan")[1].split("def run(")[1].split("def __repr__")[0]
    assert run.index("self.scan.run(guard)") < run.index("kernel[self._grid](")
    assert "_word_view(self.scan.out)" in run


def test_the_constitutive_plan_delegates_to_the_complex_plan_with_the_start_of_h():
    """No new constitutive kernel: the builder returns ``ComplexConstitutivePlan``, bound
    on the run's division, and the H side carries MEEP's start of H (``h_state``)."""
    fields, pml = build(m=-1)
    fake_cupy = types.SimpleNamespace(__name__="cupy")
    claims_cupy = Override(fields, grid=Override(fields.grid, xp=fake_cupy))
    for side, half_integer in (("H", False), ("E", True)):
        plan = module.plan_cylindrical_complex_constitutive(claims_cupy, pml, side)
        assert isinstance(plan, complex_module.ComplexConstitutivePlan), side
        assert plan.side == side
        assert plan.scale == (1 if side == "E" else 0)
        suffix = "_h" if half_integer else ""
        expected = [getattr(pml, f"{stem}_{axis}{suffix}").reshape(-1)
                    for axis in "xyz" for stem in ("kps", "kms")]
        for pointer, wanted in zip(plan._coefficients, expected):
            assert same_bytes(pointer.array, wanted)
        assert (plan._h_state is claims_cupy) is (side == "H")


def test_the_curl_plan_rejects_an_unknown_sub_step():
    fields, pml = build()
    with pytest.raises(ValueError, match="sub_step must be one of"):
        module.CylindricalComplexCurlPlan(
            "step_E", fields.grid.shape, 0.37, 0.37, 1, False, 1, 256,
            [], [], [], [], np, chunk_bounds=None, scan_starts=None)


# ---------------------------------------------------------------------------
# The gate: its reference is the array path; its tables are checked here
# ---------------------------------------------------------------------------

def test_the_gate_cases_span_every_m_class_both_terminations_and_both_divisions():
    """m = 0 (complex), |m| = 1 of both signs, |m| >= 2 on both near-axis branches, z
    metallic and periodic, one r chunk and two, and Courant numbers that are not float32
    words (the double products are visible only there)."""
    gate = gate_module()
    cases = gate.COMPLEX_CASES
    assert {module.m_class(c["m"]) for c in cases} == {0, 1, 2}
    assert {c.get("z", "metallic") for c in cases} == {"metallic", "periodic"}
    assert any(c.get("accurate") for c in cases) and any(not c.get("accurate") for c in cases)
    assert any("x" in c["thickness"] for c in cases)
    assert any("x" not in c["thickness"] for c in cases)
    assert sum(1 for c in cases if not is_power_of_two(c["courant"])) >= 3
    assert {c["m"] for c in cases} >= {1, -1}


def test_every_gate_mutation_needle_is_present_in_the_shipped_text():
    """A needle that is not in the shipped text DISARMS its mutation; each must be."""
    gate = gate_module()
    common = gate.common
    curl = kernel_function_source("cyl_complex_pml_curl_step")
    scan_text = (PACKAGE_DIR / "cylindrical_triton.py").read_text(encoding="utf-8")
    scan = scan_text.split("def cyl_radial_difference(")[1].split("def cyl_pml_curl_step(")[0]
    for name, which, edits, expected, seedings in gate.COMPLEX_MUTATIONS:
        source = curl if which == "curl" else scan
        common.apply_edits(source, edits)                  # raises on a missing needle
        assert expected in ("caught", "null"), name
        assert set(seedings) <= set(common.SEEDINGS), name


def test_the_gate_mutation_table_covers_the_battery_it_owes():
    """The Appendix A inversion, the section 7.2 battery, the i*m/r pass, the axis rules,
    the scan, and the declared nulls -- by name."""
    names = {row[0] for row in gate_module().COMPLEX_MUTATIONS}
    owed = {"a1_pairwise_curl_grouping", "b1_whole_grid_recurrence",
            "b2_membership_one_row_off", "b3_lattices_swapped",
            "b4_fu_stored_on_every_cell", "b5_predicate_on_coefficient_identity",
            "n1_fu_load_unconditional", "i1_imr_dropped", "x1_axis_increment_in_float32",
            "z1_near_rows_fu_kept_outside", "m0_dz_axis_add_in_float32",
            "g1_near_ghost_restored", "s1_one_segment", "s3_first_row_not_from_zero"}
    assert owed <= names, owed - names
    nulls = {row[0] for row in gate_module().COMPLEX_MUTATIONS if row[3] == "null"}
    assert nulls == {"n1_fu_load_unconditional", "g1_near_ghost_restored"}


def test_every_device_leg_named_in_the_gate_is_reachable_from_main():
    gate_text = GATE_PATH.read_text(encoding="utf-8")
    main = gate_text.split("def main(")[1]
    for leg in ("synthetic", "guard", "multi_step", "constitutive", "mutations"):
        assert f'"{leg}" in legs' in main, leg


def test_the_gate_refuses_to_report_a_device_result_without_a_device(monkeypatch):
    """No CuPy, no device: the gate exits naming the device it needs, and writes no
    result."""
    gate = gate_module()
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise ImportError("cupy is blocked for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(SystemExit, match="CUDA device"):
        gate.common.device_xp()
