"""Tests for the Triton CYLINDRICAL (Dcyl, m = 0) curl kernel — coverage, tables, plan.

Everything here runs on a laptop: no GPU, no CuPy, no Triton. What needs hardware —
byte identity against ``stepping.step_B``/``step_D``, the mutation legs, throughput —
lives in ``parity/meep_gpu/gate_triton_cylindrical.py``, because a byte comparison
against the array path is only meaningful on the device that runs it.

What is pinned HERE is everything that decides whether the kernel is ever ALLOWED to
run, plus the facts about it that would be wrong SILENTLY rather than loudly:

* **coverage**, with mutations proving each added clause is load-bearing — above all
  the ``m == 0`` clause, since |m| >= 1 is a different kernel and admitting it would
  produce a smooth, plausible, entirely wrong field;
* **the ownership mask**, derived from ``fields.IYEE_SHIFTS`` and the Dcyl boundary
  triple and compared against what the kernel hard-codes;
* **the prefix**, including the ZERO WALL ROW on the B side — the historical defect
  this project already paid for once;
* **the two-directional load-bearingness of ``coverage._grid_reasons`` clause 6**:
  it must keep refusing Dcyl for the Cartesian predicates, and this module must not
  compose it;
* **the optional dependency staying optional.**
"""

from __future__ import annotations

import ast
import builtins
import importlib
import pathlib
import sys

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import IYEE_SHIFTS, Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.triton_kernels import coverage as coverage_module
from meep_gpu.triton_kernels import cylindrical_triton as module

PACKAGE_DIR = pathlib.Path(module.__file__).parent

#: THE STEP BUDGET, as three separate numbers, because conflating them is how §16's
#: mistake was made. Stated as data so a reader of the test file finds them without
#: opening the gate artifact.
#:
#: 1. the NumPy TRANSCRIPTION is bytewise identical to ``stepping.py`` for 400 steps
#:    at four Courant numbers (three non-power-of-two);
#: 2. the gate RAN 1000 steps on hardware;
#: 3. the KERNEL is bytewise identical for **one launch everywhere in the sweep** and
#:    over consecutive steps only **to step 23** at the worst row measured — first
#:    divergence at steps 23/24/37/54 (engine leg) and 67/67/72/72 (synthetic leg),
#:    always ONE float in an ``fu_*`` auxiliary, always a subnormal against CuPy's
#:    flushed zero. That is plan §16's amplifying disagreement, and the same run's
#:    controls say so: array-path-against-itself is identical 4/4 over 1000 steps, and
#:    with ONE mantissa bit flipped it diverges at step 1 in 4/4.
CERTIFIED_STEPS_NUMPY_TRANSCRIPTION = 400
GATE_STEPS_RUN_ON_HARDWARE = 1000
KERNEL_CERTIFIED_CONSECUTIVE_STEPS = 23


#: r-high plus both z faces, in CELLS — the layer a Dcyl script actually asks for
#: (``mp.PML(d, direction=mp.R, side=mp.High)`` + ``mp.PML(d, direction=mp.Z)``).
#: There is no absorber at r = 0 and none on phi, which has one cell.
DCYL_PML_CELLS = {"x": (0, 5), "z": 5}


#: z is METALLIC, not periodic, and that is what a lifted Dcyl script gets: an
#: ``mp.Simulation`` with no ``k_point`` runs ``use_bloch = false``, i.e. PEC walls
#: on every face MEEP has one for. ``Grid``'s own default is periodic, so a directly
#: constructed grid has to say so — and the difference is exactly the z ghost rule
#: this kernel compiles in, which makes it worth stating rather than inheriting.
DCYL_BOUNDARIES = {"z": "metallic"}


def build(cell_size=(2.0, 0.0, 4.0), m=0, resolution=10, thickness=None,
          boundaries=None, **kwargs):
    """A real cylindrical ``Grid``/``Fields``/``PML`` triple, built as the engine builds one."""
    grid = Grid(resolution=resolution, cell_size=cell_size, cylindrical=True, m=m,
                boundaries=DCYL_BOUNDARIES if boundaries is None else boundaries,
                **kwargs)
    fields = Fields(grid=grid, force_complex_fields=(m != 0))
    fields.enable_pml_storage()
    return fields, PML(grid=grid,
                       thickness=DCYL_PML_CELLS if thickness is None else thickness)


def cartesian(cell_size=(0.8, 0.8, 0.8), **kwargs):
    grid = Grid(resolution=10.0, cell_size=cell_size, **kwargs)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=2)


def reasons(fields, pml):
    """The refusal reasons, minus the CuPy-backend one every laptop run carries."""
    return [r for r in module.cylindrical_curl_coverage(fields, pml).reasons
            if "not cupy" not in r]


def code_of(path: pathlib.Path) -> str:
    """A module's source with comments and docstrings removed — see test_triton_kernels."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and ast.get_docstring(node):
            node.body = node.body[1:]
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# The optional dependency must stay optional
# ---------------------------------------------------------------------------

@pytest.fixture
def restored_module():
    """Reload the module cleanly AFTER a test that reloaded it with Triton blocked.

    ``importlib.reload`` mutates the module object every other test holds, so an
    absence test without this leaves ``cyl_pml_curl_step`` at None for the rest of
    the session — which on a Triton host turns the source tests into an
    ``AttributeError`` far from its cause. MEASURED: that is exactly how it failed
    the first time it ran on the measurement host.

    Requested FIRST in the signature so it is set up first and therefore torn down
    LAST, after ``monkeypatch`` has restored ``__import__``.
    """
    yield
    importlib.reload(importlib.import_module("meep_gpu.triton_kernels.cylindrical_triton"))


def test_the_module_imports_and_answers_coverage_with_triton_absent(restored_module,
                                                                   monkeypatch):
    """A missing optional dependency must not break the engine — or this module.

    The kernel sits behind ``if triton is not None`` for exactly this reason:
    ``@triton.jit`` runs at import time, and the predicate has to be importable and
    answerable on the machine that is the merge bar.
    """
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise ImportError("triton is blocked for this test")
        return real_import(name, *args, **kwargs)

    for name in [n for n in sys.modules if n.startswith("triton")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)
    reloaded = importlib.reload(
        importlib.import_module("meep_gpu.triton_kernels.cylindrical_triton"))
    fields, pml = build()
    verdict = reloaded.cylindrical_curl_coverage(fields, pml)
    assert verdict.covered is False            # NumPy backend, and it says so
    assert any("not cupy" in reason for reason in verdict.reasons)
    with pytest.raises(ImportError, match="optional"):
        reloaded.cylindrical_curl_kernel()
    assert reloaded.plan_cylindrical_curl(fields, pml, "step_B") is None


def test_the_module_touches_no_file_another_track_owns():
    """File ownership, enforced rather than agreed.

    Three concurrent tracks share this package. This module defines its own kernel,
    its own predicates and its own plan, and imports shared helpers READ-ONLY; the
    prose names the modules integration will have to edit, which is why the check is
    on executable text rather than on the raw file.
    """
    path = PACKAGE_DIR / "cylindrical_triton.py"
    code = code_of(path)
    for banned in ("fastpath", "cuda_kernels"):
        assert banned not in code, f"cylindrical_triton.py references {banned}"
    # The sibling kernel modules are owned by other live tracks; this one must not
    # import them at all. Checked on the IMPORT GRAPH rather than by substring,
    # because "symmetry" is also a word the grid vocabulary uses (``has_symmetry``).
    imported = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    for banned in ("symmetry", "dispersive_update_e", ".symmetry",
                   ".dispersive_update_e"):
        assert banned not in imported, f"cylindrical_triton.py imports {banned}"
    # What it MAY import, read-only, is the shared host-side helpers and the shared
    # guard constant. Nothing else from a file another track owns.
    assert {"CupyPointer", "_flat", "ENABLE_FP_FUSION"} <= imported
    assert "options=" not in code, "the guard must never ride in an options dict"
    # The one launch site, spelled exactly as every other launch site in the package.
    assert code.count("enable_fp_fusion=") == 1
    assert "enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard)" in code


# ---------------------------------------------------------------------------
# The tables are the engine's own
# ---------------------------------------------------------------------------

def test_the_sub_step_tables_are_steppings_own():
    """Targets and sources per sub-step, against ``stepping.B_CURL_TERMS``/``D_CURL_TERMS``."""
    for sub_step, terms in (("step_B", stepping.B_CURL_TERMS),
                            ("step_D", stepping.D_CURL_TERMS)):
        spec = module.SUB_STEPS[sub_step]
        assert spec["targets"] == tuple(term.target for term in terms)
    assert module.SUB_STEPS["step_B"]["sources"] == ("Ex", "Ey", "Ez")
    assert module.SUB_STEPS["step_D"]["sources"] == ("Hx", "Hy", "Hz")
    assert module.SUB_STEPS["step_B"]["backward"] == 0
    assert module.SUB_STEPS["step_D"]["backward"] == 1


def test_the_half_integer_pairing_is_the_one_stepping_uses():
    """B reads HALF-INTEGER curl coefficients, D integer. Swapped, it is a silent
    half-cell error in the absorber profile — converged, smooth and wrong."""
    assert module.SUB_STEPS["step_B"]["suffix"] == "_h"
    assert module.SUB_STEPS["step_D"]["suffix"] == ""
    fields, pml = build()
    _, half = stepping._curl_coefficients(pml, "x", half_integer=True)
    assert half is pml.sinv_x_h
    _, integer = stepping._curl_coefficients(pml, "x", half_integer=False)
    assert integer is pml.sinv_x


def test_the_prefix_component_and_ir0_are_the_ones_step_B_and_step_D_pass():
    """Ep at ir0 = 0.0 with a wall row; Hp at ir0 = 0.5 without one.

    ``ir0`` is ``origin_r*a + 0.5*iyee_shift(Fp).in_direction(R)``: Ep sits at the
    node (r-shift 0) and Hp half a cell out (r-shift 1). Exchanging the two is the
    ``ir0_swapped`` mutation, which moved 1560 Bz + 1559 Dz floats on step 0.
    """
    assert module.SUB_STEPS["step_B"]["prefix_component"] == "Ey"   # Ep
    assert module.SUB_STEPS["step_D"]["prefix_component"] == "Hy"   # Hp
    assert module.SUB_STEPS["step_B"]["prefix_ir0"] == 0.0
    assert module.SUB_STEPS["step_D"]["prefix_ir0"] == 0.5
    assert IYEE_SHIFTS["Ey"][0] == 0 and IYEE_SHIFTS["Hy"][0] == 1
    assert module.SUB_STEPS["step_B"]["extend_wall_row"] is True
    assert module.SUB_STEPS["step_D"]["extend_wall_row"] is False


def test_the_boundary_triple_is_what_stepping_resolves_for_a_dcyl_grid():
    """``('axis', 'periodic', 'metallic')`` — measured off a real grid, not assumed."""
    fields, pml = build()
    assert tuple(stepping._boundary_kinds(fields.grid, pml)) == \
        module.CYLINDRICAL_BOUNDARY_KINDS
    assert tuple(fields.grid.is_axis(a) for a in range(3)) == \
        module.CYLINDRICAL_AXIS_FLAGS
    assert fields.grid.shape[1] == 1


def test_m_zero_storage_is_real_and_m_one_is_not():
    """The whole reason this kernel needs no complex arithmetic."""
    real_fields, _ = build(m=0)
    assert real_fields.Ez.dtype == np.float32
    complex_fields, _ = build(m=1)
    assert complex_fields.Ez.dtype == np.complex64


# ---------------------------------------------------------------------------
# The ownership mask, derived and compared against what the kernel hard-codes
# ---------------------------------------------------------------------------

def test_the_ownership_mask_is_derived_from_iyee_shifts_and_the_boundary_triple():
    """The table the kernel compiles in, recomputed from the engine's own inputs."""
    assert module.ownership_mask_axes("step_B") == ((0,), (), (2,))
    assert module.ownership_mask_axes("step_D") == ((2,), (0, 2), (0,))
    # And it really is what _mask_non_owned_cells would do on this grid.
    fields, pml = build()
    grid = fields.grid
    kinds = stepping._boundary_kinds(grid, pml)
    for sub_step, terms in (("step_B", stepping.B_CURL_TERMS),
                            ("step_D", stepping.D_CURL_TERMS)):
        expected = []
        for term in terms:
            axes = []
            for axis in range(3):
                if term.iyee[axis] != 0:
                    assert not stepping._stored_past_owned(grid, axis)
                    continue
                if (grid.is_mirrored(axis) or grid.is_metallic(axis)
                        or grid.is_axis(axis)):
                    axes.append(axis)
            expected.append(tuple(axes))
        assert module.ownership_mask_axes(sub_step) == tuple(expected), kinds


requires_triton = pytest.mark.requires_resource("triton")


def _triton_available() -> bool:
    """Ask the DEPENDENCY, never ``module.cyl_pml_curl_step``.

    The absence test reloads this module; reading its kernel attribute as the
    availability signal makes the skip depend on test ORDER.
    """
    try:
        import triton  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001
        return False
    return True


@requires_triton
@pytest.mark.skipif(not _triton_available(), reason="Triton is not installed")
def test_the_kernel_source_masks_exactly_the_derived_axes():
    """The hard-coded ``tl.where(at_r|at_z, 0.0, curlN)`` set equals the derived table.

    Hard-coding the mask in the kernel is right — it is a compile-time fact for this
    one configuration. Leaving it uncheckable would not be, which is what this test
    is for. It needs Triton only because ``.fn`` is where the source lives.
    """
    import inspect
    import re

    # Re-import rather than reading the module object another test may have reloaded
    # with Triton blocked.
    live = importlib.import_module("meep_gpu.triton_kernels.cylindrical_triton")
    if live.cyl_pml_curl_step is None:
        live = importlib.reload(live)
    source = inspect.getsource(live.cyl_pml_curl_step.fn)
    body = source.split("# --- ownership mask")[1].split("# --- split-field")[0]
    backward, forward = body.split("else:")[0], body.split("else:")[1]
    found = {"step_D": set(), "step_B": set()}
    for label, chunk in (("step_D", backward), ("step_B", forward)):
        for axis_letter, target in re.findall(
                r"tl\.where\(at_([rz]), 0\.0, curl(\d)\)", chunk):
            found[label].add((int(target), axis_letter))
    for sub_step in ("step_B", "step_D"):
        expected = {(index, module.MASK_AXIS_LETTERS[axis])
                    for index, axes in enumerate(module.ownership_mask_axes(sub_step))
                    for axis in axes}
        assert found[sub_step] == expected, sub_step


# ---------------------------------------------------------------------------
# The prefix — the part that stays on the array path
# ---------------------------------------------------------------------------

def test_the_B_side_prefix_carries_the_zero_wall_row():
    """The historical defect, pinned. Without the wall row the last row's forward
    difference becomes MINUS the whole accumulated sum (``no_wall_row``: 39 Bz floats,
    all in the last rows)."""
    fields, _ = build()
    rng = np.random.default_rng(20260810)
    fields.Ey[...] = rng.uniform(-1, 1, size=fields.Ey.shape).astype(np.float32)
    sources = {name: getattr(fields, name) for name in ("Ex", "Ey", "Ez")}
    prefix = module.cylindrical_prefix(np, "step_B", sources)
    assert prefix.shape == (fields.Ey.shape[0] + 1,) + fields.Ey.shape[1:]

    rows = fields.Ey.shape[0]
    extended = np.empty((rows + 1,) + fields.Ey.shape[1:], dtype=np.float32)
    extended[:rows] = fields.Ey
    extended[rows] = 0
    expected = stepping.cylindrical_rderiv_prefix(np, extended, 0.0)
    assert prefix.tobytes() == expected.tobytes()

    # And it is NOT the unextended prefix — the mutation must have somewhere to go.
    unextended = stepping.cylindrical_rderiv_prefix(np, fields.Ey, 0.0)
    assert prefix[:rows].tobytes() == unextended.tobytes()
    assert prefix.shape != unextended.shape


def test_the_D_side_prefix_is_Hp_at_half_and_carries_no_wall_row():
    fields, _ = build()
    rng = np.random.default_rng(20260811)
    fields.Hy[...] = rng.uniform(-1, 1, size=fields.Hy.shape).astype(np.float32)
    sources = {name: getattr(fields, name) for name in ("Hx", "Hy", "Hz")}
    prefix = module.cylindrical_prefix(np, "step_D", sources)
    assert prefix.shape == fields.Hy.shape
    assert prefix.tobytes() == stepping.cylindrical_rderiv_prefix(
        np, fields.Hy, 0.5).tobytes()
    # ir0 is load-bearing: 0.0 is a different array.
    assert prefix.tobytes() != stepping.cylindrical_rderiv_prefix(
        np, fields.Hy, 0.0).tobytes()


def test_the_prefix_cannot_be_algebraically_eliminated():
    """``prefix[i+1] - prefix[i] != increment[i+1]`` in float32.

    This looks like a free optimization and is not; ``bz_flat_grouping`` is the same
    class of error and it moved 211 Bz floats on step 0.
    """
    fields, _ = build()
    rng = np.random.default_rng(20260812)
    fields.Hy[...] = rng.uniform(-1, 1, size=fields.Hy.shape).astype(np.float32)
    prefix = stepping.cylindrical_rderiv_prefix(np, fields.Hy, 0.5)
    counts = np.arange(fields.Hy.shape[0], dtype=np.float64) + 0.5
    weights = counts.reshape(-1, 1, 1).astype(np.float32)
    divisor = (counts[1:] - 0.5).reshape(-1, 1, 1).astype(np.float32)
    weighted = fields.Hy * weights
    increment = np.zeros_like(fields.Hy)
    increment[1:] = (weighted[1:] - weighted[:-1]) / divisor
    difference = prefix[1:] - prefix[:-1]
    assert difference.tobytes() != increment[1:].tobytes()


# ---------------------------------------------------------------------------
# Coverage — the positive clauses, and a mutation per added one
# ---------------------------------------------------------------------------

def test_a_real_m0_cylindrical_pml_configuration_is_refused_only_for_the_backend():
    """The whole point: everything else about this configuration is admitted."""
    fields, pml = build()
    assert reasons(fields, pml) == []
    verdict = module.cylindrical_curl_coverage(fields, pml)
    assert verdict.covered is False
    assert [r for r in verdict.reasons if "not cupy" in r]


@pytest.mark.parametrize("m", [1, -1, 2, -2])
def test_a_nonzero_m_is_refused_BY_NAME(m):
    """Refused because m != 0, not because an i*m/r term happened not to be found.

    |m| >= 1 forces complex64 storage, adds the i*m/r coupling, adds the |m| = 1 axis
    increments folded into curl row 0 on both sides, and adds the |m| >= 2
    six-component zeroing of rows [0:|m|]. That is a second kernel.
    """
    fields, pml = build(m=m)
    named = [r for r in reasons(fields, pml) if "m = " in r and "ONLY" in r]
    assert named, reasons(fields, pml)


def test_dropping_the_m_clause_admits_a_configuration_the_kernel_would_get_wrong(
        monkeypatch):
    """The mutation that proves the m clause is load-bearing, not decoration."""
    fields, pml = build(m=1)
    before = reasons(fields, pml)
    assert any("ONLY" in r for r in before)

    original = module._cylindrical_geometry_reasons

    def without_m_clause(f, p, g):
        return [r for r in original(f, p, g) if "ONLY" not in r]

    monkeypatch.setattr(module, "_cylindrical_geometry_reasons", without_m_clause)
    after = reasons(fields, pml)
    # Complex storage is still caught (a second, independent clause) — which is
    # exactly why the m clause must be its own: a future real-storage |m| >= 1 mode
    # would sail through on the storage clause alone.
    assert len(after) < len(before)


def test_a_cartesian_grid_is_refused_by_this_predicate():
    """These predicates are the Dcyl ones. A Cartesian grid belongs to ``coverage``."""
    fields, pml = cartesian()
    assert any("not cylindrical" in r for r in reasons(fields, pml))


def test_the_shared_cartesian_predicate_still_refuses_dcyl():
    """``coverage._grid_reasons`` clause 6 is now LOAD-BEARING IN TWO DIRECTIONS.

    It must keep refusing Dcyl for the Cartesian kernels (whose coefficient index and
    ghost rules are wrong there), and this module must not compose it — composing it
    would make every predicate here vacuously False. Both halves are asserted so a
    later "tidy-up" that widens the shared clause fails here.
    """
    fields, pml = build()
    verdict = coverage_module.pml_curl_coverage(fields, pml)
    assert verdict.covered is False
    assert any("cylindrical" in r.lower() for r in verdict.reasons)
    assert "_grid_reasons" not in code_of(PACKAGE_DIR / "cylindrical_triton.py")


def test_the_shared_clause_builders_this_module_composes_from_still_exist():
    """A rename in the shared file must fail at the merge bar, not silently drop a clause."""
    for name in module.SHARED_CLAUSES:
        assert hasattr(coverage_module, name), name


def test_complex_storage_is_refused():
    grid = Grid(resolution=10, cell_size=(2.0, 0.0, 4.0), cylindrical=True, m=0,
                boundaries=DCYL_BOUNDARIES)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    assert any("force_complex_fields" in r
               for r in reasons(fields, PML(grid=grid, thickness=DCYL_PML_CELLS)))


def test_no_active_layer_is_refused_with_its_own_reason():
    fields, _ = build()
    assert any("no active PML" in r for r in reasons(fields, None))


def test_a_missing_auxiliary_is_refused_by_name():
    fields, pml = build()
    fields.fu_Dz = None
    assert any("fu_Dz is not allocated" in r for r in reasons(fields, pml))


def test_a_float64_volume_is_refused():
    fields, pml = build()
    fields.Bz = fields.Bz.astype(np.float64)
    assert any("Bz dtype" in r for r in reasons(fields, pml))


def test_a_non_contiguous_volume_is_refused():
    fields, pml = build()
    fields.Ez = np.asfortranarray(np.zeros((2, 2, 2), dtype=np.float32))
    assert any("Ez" in r for r in reasons(fields, pml))


def test_a_grid_that_cannot_report_the_near_axis_policy_is_refused(monkeypatch):
    """The flag is irrelevant at m = 0 and is READ anyway.

    An |m| >= 2 admission later must not inherit this predicate's silence about
    ``accurate_fields_near_cylorigin`` — which is what selects between the r = 0 row
    and every row within |m| pixels of it.
    """
    fields, pml = build()
    assert module.cylindrical_axis_policy(fields.grid) in (True, False)

    class Blind:
        def __getattr__(self, name):
            if name == "accurate_fields_near_cylorigin":
                raise AttributeError(name)
            return getattr(fields.grid, name)

    assert module.cylindrical_axis_policy(Blind()) is None
    assert any("accurate_fields_near_cylorigin" in r
               for r in module._cylindrical_geometry_reasons(fields, pml, Blind()))


def test_a_boundary_triple_that_is_not_the_dcyl_one_is_refused(monkeypatch):
    """The mutation for the boundary clause: a resolved triple that is not ours."""
    fields, pml = build()
    monkeypatch.setattr(coverage_module, "_boundary_kinds",
                        lambda grid, layer: ("axis", "metallic", "metallic"))
    assert any("boundary kinds" in r for r in reasons(fields, pml))


def test_a_conductivity_on_a_curl_target_is_refused():
    fields, pml = build()
    monkeypatch_target = {"Dz"}
    original = fields.condfac_for
    fields.condfac_for = lambda name: (object() if name in monkeypatch_target
                                       else original(name))
    assert any("conductivity is installed on Dz" in r for r in reasons(fields, pml))


# ---------------------------------------------------------------------------
# The constitutive side — no new kernel, only a predicate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", ["H", "E"])
def test_the_existing_constitutive_kernel_is_admitted_on_a_dcyl_grid(side):
    """``update_H``/``update_E`` carry NO cylindrical branch — measured by reading them.

    So the cylindrical tranche is ONE new kernel plus a predicate, not three. The
    byte-level confirmation on a Dcyl grid is a hardware leg and is NOT yet run; the
    module docstring says so in the same words.
    """
    fields, pml = build()
    verdict = module.cylindrical_constitutive_coverage(fields, pml, side)
    assert [r for r in verdict.reasons if "not cupy" not in r] == []
    for name in ("update_H", "update_E"):
        source = stepping.__dict__[name].__doc__ or ""
        assert "cylindrical" not in source.lower()


def test_the_constitutive_predicate_refuses_a_nonzero_m_too():
    fields, pml = build(m=1)
    verdict = module.cylindrical_constitutive_coverage(fields, pml, "H")
    assert any("ONLY" in r for r in verdict.reasons)


def test_an_unknown_constitutive_side_raises():
    fields, pml = build()
    with pytest.raises(ValueError, match="side must be"):
        module.cylindrical_constitutive_coverage(fields, pml, "B")


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_planning_a_refused_configuration_returns_None_without_needing_triton():
    fields, pml = build()                       # NumPy backend => refused
    assert module.plan_cylindrical_curl(fields, pml, "step_B") is None
    with pytest.raises(ValueError, match="sub_step must be"):
        module.plan_cylindrical_curl(fields, pml, "update_H")


def build_plan(sub_step):
    """A plan over NumPy arrays. It cannot LAUNCH here; everything else is real."""
    fields, pml = build()
    spec = module.SUB_STEPS[sub_step]
    arrays = {name: getattr(fields, name)
              for name in module.CURL_TARGETS + module.CURL_SOURCES}
    arrays.update({"fu_" + n: getattr(fields, "fu_" + n) for n in module.CURL_TARGETS})
    flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{spec['suffix']}")
            for axis in "xyz" for stem in ("kms", "sinv")}
    dtdx = fields.grid.dt / fields.grid.dx
    return module.plan_cylindrical_curl_from_arrays(
        sub_step, arrays, flat, dtdx, np), fields, dtdx


def test_four_dtdx_is_the_float32_rounding_of_the_array_paths_python_float():
    """``(4.0 * (dt/dx))`` is formed in float64 and cast WEAKLY onto float32 storage.

    Recomputing ``4.0 * dtdx`` inside the kernel from an fp32 ``dtdx`` is a different
    number, and the difference is invisible to anything but a byte comparison.
    """
    plan, fields, dtdx = build_plan("step_D")
    assert plan.four_dtdx == float(np.float32(4.0 * dtdx))
    # The array path's own spelling, run for real: a Python float times a float32
    # volume, which NEP-50 casts weakly. The product's dtype IS float32, which is
    # what makes the plan's rounding the right one.
    probe = np.ones((1,), dtype=np.float32)
    weak = (4.0 * dtdx) * probe
    assert weak.dtype == np.float32
    assert np.float32(plan.four_dtdx) == weak[0]


def test_the_block_size_is_the_one_kernels_py_ships():
    """``DEFAULT_BLOCK`` is restated here so a plan builds without Triton; it must not
    drift from the shared value, which is read off the SOURCE rather than imported."""
    source = (PACKAGE_DIR / "kernels.py").read_text(encoding="utf-8")
    assert f"\nDEFAULT_BLOCK = {module.DEFAULT_BLOCK}\n" in source


def test_the_plan_reports_its_shape_grid_and_sub_step():
    plan, fields, _ = build_plan("step_B")
    assert plan.shape == tuple(fields.grid.shape)
    assert plan.n_elem == int(np.prod(fields.grid.shape))
    assert plan.backward == 0
    assert "m=0" in repr(plan)
    assert module.plan_cylindrical_curl_from_arrays.__doc__


def test_the_plan_recomputes_the_prefix_every_launch():
    """A cached prefix is stale after one sub-step: it is a function of the CURRENT
    sources, which the previous sub-step wrote."""
    plan, fields, _ = build_plan("step_D")
    rng = np.random.default_rng(20260813)
    fields.Hy[...] = rng.uniform(-1, 1, size=fields.Hy.shape).astype(np.float32)
    first = plan.prefix().copy()
    fields.Hy[...] = rng.uniform(-1, 1, size=fields.Hy.shape).astype(np.float32)
    assert plan.prefix().tobytes() != first.tobytes()


def test_the_three_step_budgets_are_stated_separately_and_not_conflated():
    """The claimable budget is the SMALLEST of the three, and it is not the one run.

    §16's mistake was reading a claim about step 40 as a claim about the run. Here the
    three numbers are held apart by assertion: the transcription's 400, the 1000 the
    gate executed, and the 23 consecutive steps the KERNEL is bytewise identical for.
    A later edit that raises the claim has to raise the measurement first.
    """
    assert CERTIFIED_STEPS_NUMPY_TRANSCRIPTION == 400
    assert GATE_STEPS_RUN_ON_HARDWARE == 1000
    assert KERNEL_CERTIFIED_CONSECUTIVE_STEPS == 23
    assert (KERNEL_CERTIFIED_CONSECUTIVE_STEPS
            < CERTIFIED_STEPS_NUMPY_TRANSCRIPTION
            < GATE_STEPS_RUN_ON_HARDWARE), (
        "the kernel's budget is the smallest of the three and must stay labelled as "
        "the kernel's — running longer is not certifying longer")


def test_the_module_states_that_the_dead_ghost_is_uncertifiable():
    """The one design decision no byte gate can check must be written down as such.

    ``axis_ghost_sign`` came back 16/16 identical on hardware — uncaught, as predicted
    — so the module's account of it is the only record there is. A tidy-up that
    removes the paragraph removes the only evidence a reader has.
    """
    text = (PACKAGE_DIR / "cylindrical_triton.py").read_text(encoding="utf-8")
    assert "NOT IMPLEMENTED" in text
    assert "no byte gate" in text
    assert "r_to_minus_r" in text
