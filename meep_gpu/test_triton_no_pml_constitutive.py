"""Merge-bar tests for :mod:`meep_gpu.triton_kernels.no_pml_constitutive`.

THIS FAMILY SHIPS NO KERNEL, and that changes what a laptop suite is FOR here.
For every other module in this package the laptop holds the predicate while a CUDA
host holds the arithmetic. There is no arithmetic to hold: ``stepping.update_H``
returns at stepping.py:944-945 and ``stepping.update_E`` returns at
stepping.py:983-984, before any statement that reads an array. So this file carries
BOTH halves — the predicate AND the numerical claim — and the numerical claim is
checkable here precisely because the product is the absence of a computation:

* the IDENTITY property (the family's whole content) is measured against the REAL
  ``stepping`` functions over multiple complete ``update_H``+``update_E`` cycles on
  real ``Grid``/``Fields``/``PML`` objects, with a seeded state that carries signed
  zeros and subnormals and an in-test census that fails the assertion as VACUOUS if
  either class is missing;
* its CONTROL — the same assertion on configurations the predicate refuses, where
  the array path must move bytes — is here too, because an identity test with no
  failing control measures nothing;
* the ARM S measurements the module's docstring cites (the stored-E product is
  ``D * inv_eps`` exactly; binding ``kps=1/kms=0/f=0`` on the certified accumulation
  does NOT recover a store, because ``0.0 + (-0.0)`` canonicalises a negative zero)
  are re-measured rather than quoted.

What is NOT here: a kernel-source hash, a launch-guard spelling, an
``@triton.jit`` pin. Those exist in the sibling test files because those modules
ship a kernel body whose MLIR lowering was certified on one Triton version. This
module has no body to weld, so the tests assert the ABSENCE of that surface — a
later hand that adds a launch here must add its weld too, and the assertions below
are what make that a failure rather than a quiet gap.

The optional-dependency contract is also stronger here than anywhere else in the
package: this module must import and answer with BOTH ``triton`` and ``cupy``
absent, and must be COVERED (not merely answerable) on NumPy, because it launches
nothing against a device pointer.

The device-side record lives in ``parity/meep_gpu/gate_triton_no_pml_constitutive.py``
— whose release contract, :func:`validate_payload`, is itself mutation-tested below.
NO byte-identity claim on a device is made by this file or by that gate: no device
leg has been run.
"""

from __future__ import annotations

import ast
import builtins
import copy
import importlib
import importlib.util
import json
import os
import pathlib
import sys

import numpy
import pytest

from meep_gpu.dispersion import Susceptibility
from meep_gpu.driver import FdtdDriver
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.pml import PML
from meep_gpu.test_triton_kernels import PACKAGE_DIR
from meep_gpu.triton_kernels import coverage as coverage_module

MODULE_PATH = PACKAGE_DIR / "no_pml_constitutive.py"
MODULE_NAME = "meep_gpu.triton_kernels.no_pml_constitutive"
GATE_PATH = (PACKAGE_DIR.parents[1] / "parity" / "meep_gpu"
             / "gate_triton_no_pml_constitutive.py")
PROBE_PATH = (PACKAGE_DIR.parents[1] / "parity" / "meep_gpu"
              / "probe_triton_no_pml_constitutive_composition.py")

SEED = 20260813


@pytest.fixture(scope="module")
def family():
    """The shipped module itself — never a test-local stub."""
    return importlib.import_module(MODULE_NAME)


# ---------------------------------------------------------------------------
# Construction helpers — the engine's own objects
# ---------------------------------------------------------------------------

def _build(cell_size=(1.2, 1.0, 1.4), pml_thickness=0, storage=False,
           complex_storage=False, **grid_kwargs):
    """A real Grid/Fields/PML triple on NumPy. Default: NO absorber, NO storage.

    This is the family's TARGET configuration, and every refusal test below
    perturbs exactly one clause off it.
    """
    grid = Grid(resolution=10.0, cell_size=cell_size, **grid_kwargs)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.set_background_eps(2.25)
    if storage:
        fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=pml_thickness)


def _driver(cell_size=(1.2, 1.0, 1.4), courant=0.4056, pml=None,
            dispersion=False, zero_sigma=False, source=False, **kwargs):
    """A real ``FdtdDriver``, because the reference leg has to step a real engine."""
    fdtd = FdtdDriver(cell_size=cell_size, resolution=10.0, courant=courant,
                      **kwargs)
    shape = tuple(fdtd.grid.shape)
    index = numpy.arange(int(numpy.prod(shape)), dtype=numpy.float32).reshape(shape)
    fdtd.set_epsilon(numpy.ascontiguousarray(
        (1.45 + 0.30 * numpy.sin(index * numpy.float32(0.037))).astype(numpy.float32)))
    if dispersion:
        fdtd.add_susceptibility(Susceptibility(frequency=1.1, gamma=0.05),
                                numpy.full(shape, numpy.float32(0.35),
                                           dtype=numpy.float32))
    if zero_sigma:
        # driver.py:1538-1542 — sigma identically zero drives nothing and must NOT
        # switch storage on. The predicate asks driven(), not "is one registered".
        fdtd.add_susceptibility(Susceptibility(frequency=1.1, gamma=0.05),
                                numpy.zeros(shape, dtype=numpy.float32))
    if pml:
        fdtd.setup_pml(pml)
    if source:
        fdtd.add_source({"component": "Ez", "center": (0.0, 0.0, 0.0),
                         "size": (0.0, 0.0, 0.0), "source_type": "gaussian",
                         "frequency": 1.0, "fwidth": 0.4, "amplitude": 1.0})
    return fdtd


#: Every array a ``Fields`` may own that a constitutive sub-step could conceivably
#: write. Read by name and skipped when unallocated — WHICH of them exist is itself
#: configuration-dependent, and the point of the identity claim is that none of them
#: moves regardless.
CANDIDATE_ARRAYS = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
    "f_cond_Bx", "f_cond_By", "f_cond_Bz",
    "f_cond_Dx", "f_cond_Dy", "f_cond_Dz",
    "f_bfast_Bx", "f_bfast_By", "f_bfast_Bz",
    "f_bfast_Dx", "f_bfast_Dy", "f_bfast_Dz",
)


def _words(array):
    """The uint32 view of an array's bytes. The ONLY comparison made here."""
    host = numpy.ascontiguousarray(numpy.asarray(array))
    if host.dtype == numpy.complex64:
        host = host.view(numpy.float32)
    return host.ravel().view(numpy.uint32).copy()


def _live_arrays(fields):
    out = {}
    for name in CANDIDATE_ARRAYS:
        array = getattr(fields, name, None)
        if array is not None:
            out[name] = array
    for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        for label in ("P", "P_prev"):
            for component, array in (getattr(state, label, None) or {}).items():
                if array is not None:
                    out[f"pol{index}.{label}[{component}]"] = array
    return out


def _snapshot(fields):
    return {name: _words(array) for name, array in _live_arrays(fields).items()}


def _differing_words(before, after):
    assert set(before) == set(after), "an array appeared or vanished"
    return sum(int(numpy.count_nonzero(before[name] != after[name]))
               for name in sorted(before))


def _seed_state(fields, seed=SEED):
    """Fill every live array with normals + a signed-zero plane + dense subnormals.

    THREE CLASSES ON PURPOSE. The bulk is ordinary normals; the negative zeros are
    the class ``rng.uniform`` provably cannot produce and the exact class that
    separates a plain store from ``kernels.constitutive_step``'s accumulation; the
    subnormals are the class CuPy flushes and Triton keeps. No inf and no NaN — a
    NaN's sign and payload are IEEE-unspecified, so a raw-word comparison over one
    is version-sensitive, and every value here stays finite through both arms.

    The per-array ORDINAL PLANT is not decoration: a strided plant misses a
    one-element array, and a 1-voxel cell is a real corpus row
    (``material-dispersion.py``), so each array also gets one class planted at a
    rotated index.
    """
    rng = numpy.random.default_rng(seed)
    shape = tuple(fields.grid.shape)
    size = int(numpy.prod(shape))
    classes = (numpy.float32(-0.0), numpy.float32(1e-45), numpy.float32(-3e-44))
    for ordinal, (_, array) in enumerate(sorted(_live_arrays(fields).items())):
        host = rng.uniform(-1.0, 1.0, size=size).astype(numpy.float32)
        host[3::11] = numpy.float32(-0.0)
        host[5::23] = numpy.float32(1e-45)
        host[7::29] = numpy.float32(-3e-44)
        host[(ordinal * 5) % size] = classes[ordinal % len(classes)]
        if str(getattr(array, "dtype", "")) == "complex64":
            array[...] = host.reshape(shape) + 1j * host.reshape(shape)
        else:
            array[...] = numpy.ascontiguousarray(host.reshape(shape))


def _census(fields):
    """nonzero / signed-zero / subnormal word counts. A zero in any is VACUOUS."""
    nonzero = signed_zero = subnormal = total = 0
    for _, array in _live_arrays(fields).items():
        w = _words(array)
        total += int(w.size)
        nonzero += int(numpy.count_nonzero(w != numpy.uint32(0)))
        signed_zero += int(numpy.count_nonzero(w == numpy.uint32(0x80000000)))
        exponent = (w >> numpy.uint32(23)) & numpy.uint32(0xFF)
        mantissa = w & numpy.uint32(0x7FFFFF)
        subnormal += int(numpy.count_nonzero((exponent == 0) & (mantissa != 0)))
    return {"total": total, "nonzero": nonzero, "signed_zero": signed_zero,
            "subnormal": subnormal}


def _reasons(family, fields, pml, side):
    return list(family.null_constitutive_coverage(fields, pml, side).reasons)


class _BlankStrings(ast.NodeTransformer):
    def visit_Constant(self, node):  # noqa: N802 - ast's own naming
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=""), node)
        return node


def _executable_text(path: pathlib.Path) -> str:
    """Source with comments, docstrings AND string constants removed.

    ``code_of`` strips comments and docstrings, which is enough for a module whose
    prose is all in docstrings. It is not enough here: ``STORED_E_ARM`` is a dict of
    string VALUES that quote the kernel body Arm S would need. Those are a record of
    a decision, not code, and a check that fired on them would be measuring the
    documentation.
    """
    tree = _BlankStrings().visit(ast.parse(path.read_text(encoding="utf-8")))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


# ---------------------------------------------------------------------------
# The optional dependency, which for THIS module means both of them
# ---------------------------------------------------------------------------

def _block(monkeypatch, *names):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        for blocked_name in names:
            if name == blocked_name or name.startswith(blocked_name + "."):
                raise ImportError(f"{blocked_name} is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for name in list(sys.modules):
        if any(name == n or name.startswith(n + ".") for n in names):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in (MODULE_NAME, "meep_gpu.triton_kernels.kernels",
                 "meep_gpu.triton_kernels"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_the_module_imports_and_is_COVERED_with_triton_and_cupy_both_absent(
        monkeypatch):
    """The property no other family in this package has, asserted as behaviour.

    Every kernel predicate opens with "array module is not cupy" because a kernel
    launches against a device pointer. This one launches nothing, so it must be
    covered — verdict True, reasons empty — on a machine that has neither library.
    That is not a convenience: it is what makes the whole gate runnable on a laptop
    and what makes the ``breadth`` claim measurable without a device.
    """
    _block(monkeypatch, "triton", "cupy")
    module = importlib.import_module(MODULE_NAME)
    fields, pml = _build()
    for side in ("H", "E"):
        verdict = module.null_constitutive_coverage(fields, pml, side)
        assert verdict.covered is True, verdict.reasons
        assert verdict.reasons == (), verdict.reasons
    plan = module.plan_null_constitutive_step(fields, pml)
    assert set(plan.covered) == {"update_H", "update_E"}


def test_importing_the_package_does_not_pull_this_module_in(monkeypatch):
    """Dispatch is deferred; the package must not grow a dependency on this file."""
    _block(monkeypatch, "triton")
    package = importlib.import_module("meep_gpu.triton_kernels")
    assert package.triton_available() is False
    assert MODULE_NAME not in sys.modules, (
        "importing the package pulled in no_pml_constitutive")


# ---------------------------------------------------------------------------
# The module ships NO kernel, and that absence is load-bearing
# ---------------------------------------------------------------------------

def test_the_module_ships_no_kernel_and_no_launch_surface():
    """The shape of the file IS the finding, so it is pinned like one.

    A later hand that adds a launch here also has to add the guard spelling, a
    ``KERNEL_SOURCE_SHA256`` weld and a fingerprints entry — the three things every
    other module in this package carries. This test is what turns "they forgot" into
    a failure instead of a quiet gap.
    """
    # EXECUTABLE TEXT ONLY, and that means string CONSTANTS are stripped too, not
    # just docstrings. This module's prose names the surface it does not have — the
    # WIRING note describes the seam a later round adds, and ``STORED_E_ARM`` is a
    # RECORD whose values quote the kernel body Arm S would need, ``tl.store``
    # included. Grepping the raw file (or even ``code_of``'s output) would fire on
    # the module's own documentation of what it deliberately did not build.
    code = _executable_text(MODULE_PATH)
    for absent in ("@triton.jit", "import triton", "tl.load", "tl.store",
                   "enable_fp_fusion", "KERNEL_SOURCE_SHA256", "cupy",
                   "_UnavailableKernel"):
        assert absent not in code, (
            f"{absent!r} appeared in a module documented as kernel-free; if a kernel "
            "was added, add its weld, its guard spelling and its fingerprint too")
    # And the same check from the other side, structurally: no decorated kernel, no
    # ``tl.<anything>`` attribute access. A future edit that reached for a kernel
    # under a different spelling still trips this.
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorators = {ast.unparse(d) for d in node.decorator_list}
            assert decorators <= {"property"}, (
                f"{node.name} carries {sorted(decorators)}; this module defines no "
                "kernel, so the only decorator it may use is @property")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id != "tl", "a Triton language call appeared"


def test_the_module_imports_nothing_but_coverage_from_the_package():
    """Allocation-free and import-free is a claim about the import graph."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported <= {"__future__", "typing", ".coverage", ".no_pml_stored_e"}, imported


def test_the_family_is_wired_into_the_planner_but_reaches_no_dispatch():
    """THE ROUND THAT WIRED IT RAN, so this measures the new claim, not the old.

    ``launch.plan_step`` owns the ``update_H``/``update_E`` selection and now
    consults this family behind ``absorber_inactive`` — the gate that transcribes
    this file's inverted clause 2. Asserting the module's absence from
    ``launch.py`` would now assert that the wiring did not happen.

    What is measured instead is the shape the wiring had to take:

    1. ``launch.py`` names this module and imports it ONLY inside a function body,
       so a bare package import still pulls nothing;
    2. the two entry points are exported, which is what makes the planner's arm
       reachable from outside the package;
    3. ``fastpath.py`` still does not mention it — a planner arm is not a
       dispatch, and ``plan_fast_path`` returns None on every branch;
    4. ``fingerprints.json`` still carries no entry for a kernel-free module, the
       claim the sibling test below owns.
    """
    from meep_gpu.triton_kernels import launch as launch_module

    launch_path = pathlib.Path(launch_module.__file__)
    launch_source = launch_path.read_text(encoding="utf-8")
    assert "no_pml_constitutive" in launch_source, \
        "launch.py no longer consults this family; the planner wiring regressed"

    tree = ast.parse(launch_source)
    module_scope = {node.module for node in tree.body
                    if isinstance(node, ast.ImportFrom) and node.module}
    assert "no_pml_constitutive" not in module_scope, \
        "launch.py imports this module at module scope rather than lazily"

    package = importlib.import_module("meep_gpu.triton_kernels")
    for name in ("null_constitutive_coverage", "plan_null_constitutive"):
        assert name in getattr(package, "__all__", ()), (
            f"{name} is consulted by plan_step but not exported")
        assert callable(getattr(package, name)), name

    # THE DISPATCH CLAIM MOVED, and only here. ``fastpath.py`` dispatches the
    # track now, so it NAMES this family: its run artifact reports which
    # certification each arm rides on, and the null arm is the one it DROPS
    # before dispatching (both sides of that slot are empty). What it must still
    # not do is IMPORT this module — a dispatcher that reached a plan any way
    # other than through ``plan_step`` would be selecting a numerical product
    # outside the composer's fail-closed arm table.
    fastpath_path = launch_path.parent.parent / "fastpath.py"
    fastpath_imports = set()
    for node in ast.walk(ast.parse(fastpath_path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            fastpath_imports.add(node.module)
            fastpath_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            fastpath_imports.update(alias.name for alias in node.names)
    assert not any("no_pml_constitutive" in name for name in fastpath_imports), \
        "fastpath.py imports this family; dispatch must go through plan_step"

    from meep_gpu.fastpath import ARM_CERTIFICATION, NULL_ARM_LABEL

    assert ARM_CERTIFICATION[NULL_ARM_LABEL][0] == "no_pml_constitutive", \
        "the dispatcher no longer records which family owns the null arm"


def test_fingerprints_json_deliberately_has_no_entry_for_a_kernel_free_module():
    """Welding a host predicate to a source digest records a fact about a FILE.

    ``fingerprints.json``'s census is over kernel source hashes. This family ships no
    kernel, so an entry here would assert something the census does not mean. The
    gate's identity leg is this family's record, and it re-runs on any host.
    """
    record = json.loads((PACKAGE_DIR / "fingerprints.json").read_text(encoding="utf-8"))
    assert "no_pml_constitutive.py" not in record.get(
        "specialized_kernel_sources", {})


# ---------------------------------------------------------------------------
# The predicate: the positive verdict
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", ["H", "E"])
def test_the_target_configuration_is_covered_with_zero_reasons(family, side):
    """A predicate no configuration satisfies is as useless as one everything does."""
    fields, pml = _build()
    verdict = family.null_constitutive_coverage(fields, pml, side)
    assert verdict.covered is True
    assert verdict.reasons == ()


def test_pml_is_none_is_covered_as_well_as_an_all_zero_face_layer(family):
    """``_pml_is_active`` (stepping.py:2498-2506) treats the two identically.

    ``PML.is_active``'s own docstring (pml.py:427-434) says a zero-thickness table
    "steps bit-identically to ``pml=None``", so a driver that installed one must land
    on this family and not on an absorber product.
    """
    fields, zero_layer = _build(pml_thickness=0)
    for pml in (None, zero_layer):
        for side in ("H", "E"):
            assert family.null_constitutive_coverage(fields, pml, side).covered


@pytest.mark.parametrize("side", ["H", "E"])
def test_an_unknown_side_raises_rather_than_quietly_refusing(family, side):
    fields, pml = _build()
    assert family.null_constitutive_coverage(fields, pml, side).covered
    with pytest.raises(ValueError, match="side must be one of"):
        family.null_constitutive_coverage(fields, pml, "step_B")


def test_a_fields_without_a_grid_is_refused(family):
    class _NoGrid:
        grid = None
        stores_E = False
        polarizations = ()

    verdict = family.null_constitutive_coverage(_NoGrid(), None, "H")
    assert verdict.covered is False
    assert any("no grid" in reason for reason in verdict.reasons)


# ---------------------------------------------------------------------------
# The predicate: one test per refusal, each naming its clause
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side", ["H", "E"])
def test_an_active_layer_is_refused_on_both_sides_and_names_the_dsigw_accumulation(
        family, side):
    """Clause 2 — the ONE clause both sides rest on (stepping.py:944, :982)."""
    fields, pml = _build(pml_thickness=2, storage=True)
    reasons = _reasons(family, fields, pml, side)
    assert any("active PML layer" in r and "dsigw" in r for r in reasons), reasons


def test_stores_e_refuses_the_E_side_and_leaves_the_H_side_covered(family):
    """Clause 3 — stepping.py:983 branches on exactly this, and only for E.

    The asymmetry is the family's most valuable property: H is NEVER stored without
    PML (fields.py:664-666), so ``update_H`` stays covered on all 14 no-PML corpus
    rows even where ``update_E`` is Arm S's.
    """
    fields, pml = _build()
    fields.enable_field_storage()
    assert fields.stores_E is True
    e_reasons = _reasons(family, fields, pml, "E")
    assert any("stores_E is True" in r and "STORED_E_ARM" in r for r in e_reasons)
    assert family.null_constitutive_coverage(fields, pml, "H").covered is True


def test_a_driving_polarization_refuses_E_and_names_the_driver_line(family):
    fdtd = _driver(dispersion=True)
    reasons = _reasons(family, fdtd.fields, fdtd.pml, "E")
    assert any("drives Ex" in r or "drives Ey" in r or "drives Ez" in r
               for r in reasons), reasons
    assert any("driver.py:1542" in r for r in reasons), reasons
    assert family.null_constitutive_coverage(fdtd.fields, fdtd.pml, "H").covered


def test_a_zero_sigma_polarization_is_STILL_covered_on_both_sides(family):
    """The clause asks ``driven()``, not "is one registered". That is the difference.

    driver.py:1538-1542 refuses to switch storage on for a term whose sigma is
    identically zero, "what makes the zero-strength case byte-identical rather than
    merely close". A predicate that refused on registration would refuse a
    configuration where the array path really does return at stepping.py:983 — a
    coverage loss with no defect behind it. Mutation m6 in the gate is this test's
    device-side twin.
    """
    fdtd = _driver(zero_sigma=True)
    assert fdtd.fields.polarizations, "the needle needs a registered susceptibility"
    assert fdtd.fields.stores_E is False
    for side in ("H", "E"):
        verdict = family.null_constitutive_coverage(fdtd.fields, fdtd.pml, side)
        assert verdict.covered is True, verdict.reasons


def test_a_polarization_that_cannot_report_driven_is_refused(family):
    """An unreadable susceptibility is not a covered one."""

    class _Opaque:
        pass

    fields, pml = _build()
    fields.polarizations.append(_Opaque())
    reasons = _reasons(family, fields, pml, "E")
    assert any("does not report" in r and "driven()" in r for r in reasons), reasons


def test_a_nonlinearity_is_refused_by_name_and_points_at_its_own_product(family):
    fields, pml = _build()
    zeros = {name: 0.0 for name in ("Ex", "Ey", "Ez")}
    fields.set_nonlinear_volumes(zeros, {name: 0.4 for name in ("Ex", "Ey", "Ez")})
    assert fields.has_nonlinearity
    reasons = _reasons(family, fields, pml, "E")
    assert any("chi2/chi3" in r and "nonlinear_update_e" in r for r in reasons), reasons


def test_a_surviving_offdiagonal_row_is_refused_by_name(family):
    fields, pml = _build()
    shape = fields.grid.shape
    eps = {name: numpy.full(shape, numpy.float32(2.25), dtype=numpy.float32)
           for name in ("Ex", "Ey", "Ez")}
    inv = {name: numpy.full(shape, numpy.float32(1.0 / 2.25), dtype=numpy.float32)
           for name in ("Ex", "Ey", "Ez")}
    fields.set_epsilon_volumes(
        eps, inv,
        chi1inv_offdiagonal={"Ez": {"Ex": numpy.full(shape, numpy.float32(0.05),
                                                     dtype=numpy.float32)}})
    assert fields.has_offdiagonal_epsilon
    reasons = _reasons(family, fields, pml, "E")
    assert any("off-diagonal" in r and "offdiag_update_e" in r for r in reasons), reasons


def test_a_magnetic_susceptibility_refuses_the_H_side_only(family):
    """Clause 4 — vacuous against today's engine, and written anyway.

    ``update_H`` returns regardless of what is registered, so this clause cannot be
    protecting the arithmetic. It protects the CLAIM: stepping.py:1406-1407 records
    that MEEP's H-side polarization slot is deliberately empty here, and a null must
    not report coverage of a sub-step that has grown a term. The needle is duck-typed
    because the driver refuses magnetic susceptibilities outright — which is exactly
    the "another module already guards it" reasoning this package forbids.
    """

    class _MagneticState:
        def driven(self):
            return ()

        def drives(self, component):
            return component in ("Bx", "Hx")

    fields, pml = _build()
    fields.polarizations.append(_MagneticState())
    h_reasons = _reasons(family, fields, pml, "H")
    assert any("drives magnetic component" in r for r in h_reasons), h_reasons
    assert any("stepping.py:1406-1407" in r for r in h_reasons), h_reasons
    # And it must NOT leak onto E: that side's clause is stores_E, not this.
    assert family.null_constitutive_coverage(fields, pml, "E").covered is True


@pytest.mark.parametrize("side", ["H", "E"])
def test_pml_storage_behind_an_inert_layer_is_refused_and_LABELLED_conservative(
        family, side):
    """The one clause the arithmetic does not require, and it says so in the reason.

    The null is BYTE-EXACT on this configuration — doing nothing is exactly what the
    array path does. It refuses so the package does not supply half a step to a
    configuration whose curl ``no_pml.plain_curl_coverage`` clause 3b refuses. A
    reader has to be able to tell that apart from a correctness refusal, so the
    reason text carries it.
    """
    fields, pml = _build(pml_thickness=0, storage=True)
    reasons = _reasons(family, fields, pml, side)
    assert any("inert" in r and "conservatively" in r for r in reasons), reasons


@pytest.mark.parametrize("side", ["H", "E"])
def test_the_conservative_reason_does_not_fire_where_the_layer_is_active(family, side):
    """A right verdict with a false reason is still a defect in a reasons list.

    ``no_pml.py`` carried exactly this defect once: every active-PML refusal listed
    "while the layer is inert", a sentence that is false for that case.
    """
    fields, pml = _build(pml_thickness=2, storage=True)
    reasons = _reasons(family, fields, pml, side)
    assert any("active PML layer" in r for r in reasons), reasons
    assert not any("inert" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# The CONSERVATIVE clauses, measured — not asserted, and not counted by hand
# ---------------------------------------------------------------------------

def _conservative_needle(name):
    """Build the exact object each CONSERVATIVE_CLAUSES row names."""
    if name == "pml_storage_behind_an_inert_layer":
        return _build(pml_thickness=0, storage=True)
    if name == "chi2_chi3_installed_without_stored_e":
        fields, pml = _build()
        fields.set_nonlinear_volumes({n: 0.0 for n in ("Ex", "Ey", "Ez")},
                                     {n: 0.4 for n in ("Ex", "Ey", "Ez")})
        return fields, pml
    if name == "magnetic_susceptibility_on_the_h_side":
        class _MagneticState:
            def driven(self):
                return ()

            def drives(self, component):
                return component in ("Bx", "Hx")

        fields, pml = _build()
        fields.polarizations.append(_MagneticState())
        return fields, pml
    if name == "a_polarization_that_cannot_answer_driven":
        class _Opaque:
            pass

        fields, pml = _build()
        fields.polarizations.append(_Opaque())
        return fields, pml
    raise AssertionError(f"no needle for {name}")


@pytest.mark.parametrize("name", sorted(
    importlib.import_module(MODULE_NAME).CONSERVATIVE_CLAUSES))
def test_every_clause_the_module_calls_conservative_really_is(family, name):
    """The table is a claim about bytes, so it is re-measured rather than trusted.

    "Conservative" means TWO things at once and both are checked here: the predicate
    REFUSES, and the real ``stepping`` sub-step moves ZERO words on the same seeded
    object — i.e. the null would have been byte-exact and the refusal costs coverage
    rather than preventing a wrong answer. A row that stops being either fails here.

    This exists because the module used to claim exactly ONE conservative clause
    (``_storage_switch_reasons``) while three others were, and one of the three
    carried a reason sentence that was FALSE about the object it fired on.
    """
    from meep_gpu import stepping

    row = family.CONSERVATIVE_CLAUSES[name]
    fields, pml = _conservative_needle(name)
    side = row["side"]
    verdict = family.null_constitutive_coverage(fields, pml, side)
    assert verdict.covered is False, (name, verdict.reasons)
    _seed_state(fields)
    counts = _census(fields)
    assert counts["nonzero"] > 0 and counts["signed_zero"] > 0, (name, counts)
    before = _snapshot(fields)
    (stepping.update_H if side == "H" else stepping.update_E)(fields, pml)
    moved = _differing_words(before, _snapshot(fields))
    assert moved == row["moved_words"] == 0, (name, moved)


def test_the_chi2_chi3_reason_does_not_claim_a_storage_switch_that_did_not_happen(
        family):
    """The measured defect: the reason was FALSE about the object it fires on.

    ``driver.set_chi2``/``set_chi3`` call ``enable_field_storage`` (driver.py:2016-2022);
    ``Fields.set_nonlinear_volumes`` does not. On a ``Fields`` built the second way —
    which is the way this very suite builds it two tests above —
    ``has_nonlinearity`` is True and ``stores_E`` is False, so the old sentence "the
    Pade constitutive factor forces stored E" described a switch that had not been
    thrown. The verdict was safe; the reason was not.
    """
    fields, pml = _build()
    fields.set_nonlinear_volumes({n: 0.0 for n in ("Ex", "Ey", "Ez")},
                                 {n: 0.4 for n in ("Ex", "Ey", "Ez")})
    assert fields.has_nonlinearity is True
    assert fields.stores_E is False
    reasons = _reasons(family, fields, pml, "E")
    chi = [r for r in reasons if "chi2/chi3" in r]
    assert len(chi) == 1, reasons
    assert "forces stored E" not in chi[0], chi[0]
    assert "CONSERVATIVE" in chi[0], chi[0]
    assert "nonlinear_update_e" in chi[0], chi[0]
    # And the stores_E clause must NOT have fired, or the sentence above would be
    # describing a configuration this object is not in.
    assert not any("stores_E is True" in r for r in reasons), reasons


def test_the_offdiagonal_clause_is_vacuous_and_that_is_a_measurement(family):
    """The sibling clause, and the reason it is NOT in CONSERVATIVE_CLAUSES.

    ``Fields.set_epsilon_volumes`` really does call ``enable_field_storage()`` on a
    surviving row (fields.py:1253-1255), so ``stores_E`` is already True, clause 3 has
    already refused, and the array path WRITES. The null would have been wrong there —
    which is the opposite of conservative — and this test is what keeps the two
    categories from being decided by argument.
    """
    from meep_gpu import stepping

    fields, pml = _build()
    shape = fields.grid.shape
    eps = {name: numpy.full(shape, numpy.float32(2.25), dtype=numpy.float32)
           for name in ("Ex", "Ey", "Ez")}
    inv = {name: numpy.full(shape, numpy.float32(1.0 / 2.25), dtype=numpy.float32)
           for name in ("Ex", "Ey", "Ez")}
    fields.set_epsilon_volumes(
        eps, inv,
        chi1inv_offdiagonal={"Ez": {"Ex": numpy.full(shape, numpy.float32(0.05),
                                                     dtype=numpy.float32)}})
    assert fields.stores_E is True
    assert "offdiagonal" not in " ".join(family.CONSERVATIVE_CLAUSES)
    _seed_state(fields)
    before = _snapshot(fields)
    stepping.update_E(fields, None)
    assert _differing_words(before, _snapshot(fields)) > 0


# ---------------------------------------------------------------------------
# "Cannot answer" is not "yes": the two load-bearing flags, and the raise
# ---------------------------------------------------------------------------

class _UnreadableAttribute:
    """A proxy that forwards everything except one attribute, which raises."""

    def __init__(self, inner, name):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_hidden", name)

    def __getattr__(self, name):
        if name == object.__getattribute__(self, "_hidden"):
            raise AttributeError(f"{name} is unreadable on this object")
        return getattr(object.__getattribute__(self, "_inner"), name)


@pytest.mark.parametrize("attribute,fragment", [
    ("stores_E", "does not report stores_E"),
    ("has_nonlinearity", "does not report has_nonlinearity"),
    ("has_offdiagonal_epsilon", "does not report has_offdiagonal_epsilon"),
])
def test_a_fields_that_cannot_answer_a_clause_is_refused_not_admitted(
        family, attribute, fragment):
    """``getattr(..., False)`` turned "cannot answer" into ADMIT on the E side.

    Measured before the fix: a ``Fields`` proxy whose ``stores_E`` raised was covered
    with NO reasons, while the underlying run's ``stepping.update_E`` wrote 4618
    words. The module already applied the opposite rule to polarizations two clauses
    later ("an unreadable susceptibility is not a covered one"); it now applies it to
    the flags it branches on as well.
    """
    fields, pml = _build()
    fields.enable_field_storage()
    verdict = family.null_constitutive_coverage(
        _UnreadableAttribute(fields, attribute), pml, "E")
    assert verdict.covered is False
    assert any(fragment in r for r in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("side", ["H", "E"])
def test_a_layer_that_cannot_answer_is_active_is_refused(family, side):
    """The other load-bearing flag. ``pml is None`` stays the legitimate no-PML case."""
    fields, pml = _build()

    class _MuteLayer:
        @property
        def is_active(self):
            raise RuntimeError("this layer cannot say")

    verdict = family.null_constitutive_coverage(fields, _MuteLayer(), side)
    assert verdict.covered is False
    assert any("does not report is_active" in r for r in verdict.reasons), verdict
    assert family.null_constitutive_coverage(fields, None, side).covered is True


@pytest.mark.parametrize("answer", [True, 7, object()])
@pytest.mark.parametrize("side", ["H", "E"])
def test_a_polarization_whose_driven_is_not_iterable_refuses_and_does_not_raise(
        family, answer, side):
    """The E side RAISED where the H side refused, and a raise is not a verdict.

    ``for name in tuple(driven)`` sat outside the guard that swallows an exception
    raised INSIDE ``driven()``, so a duck-typed state answering ``True`` produced
    ``TypeError: 'bool' object is not iterable`` — measured. The H side cannot do this
    because it routes every question through ``coverage._call``. Once
    ``launch.plan_step`` consults this predicate, a raising clause is a crash rather
    than a refusal.
    """

    class _NonIterableDriven:
        def driven(self):
            return answer

        def drives(self, component):
            return False

    fields, pml = _build()
    fields.polarizations.append(_NonIterableDriven())
    verdict = family.null_constitutive_coverage(fields, pml, side)
    if side == "E":
        assert verdict.covered is False
        assert any("does not report" in r and "driven()" in r
                   for r in verdict.reasons), verdict.reasons
    else:
        # The H side never asked driven() and must be unaffected.
        assert verdict.covered is True, verdict.reasons


# ---------------------------------------------------------------------------
# The DELIBERATE non-refusals — the module's distinguishing claim
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,kwargs", [
    ("fold_y", {"symmetry": ("y",), "cell_size": (1.2, 1.6, 1.4)}),
    ("complex_storage", {"complex_storage": True}),
    ("bloch_k", {"complex_storage": True, "k_point": (0.4, -1.3, 0.7)}),
    ("cylindrical_m0", {"complex_storage": True, "cylindrical": True, "m": 0,
                        "dimensions": 2, "cell_size": (1.2, 0.0, 1.4)}),
    ("beta_kz", {"complex_storage": True, "beta": 0.2, "dimensions": 2,
                 "cell_size": (1.2, 1.0, 0.0)}),
    ("bfast", {"bfast_scaled_k": (0.31, 0.0, 0.0), "cell_size": (0.1, 0.1, 4.8)}),
    ("metallic", {"boundaries": "metallic"}),
    ("np2_courant", {"courant": 0.4056}),
])
@pytest.mark.parametrize("side", ["H", "E"])
def test_the_clauses_every_other_predicate_carries_are_deliberately_absent(
        family, label, kwargs, side):
    """A null indexes nothing, so it cannot be wrong about an index.

    ``coverage._grid_reasons`` clause 5 gives the reason those clauses exist — "a
    folded axis changes n_a and therefore every cell's coefficient index". Carrying
    them here would refuse ``update_H`` on the six complex and two off-diagonal
    no-PML corpus rows for a hazard that provably cannot exist. The gate's
    ``breadth`` leg measures the same eight configurations at the byte level; this
    is the verdict half, and it fails if a later hand "hardens" the predicate by
    pasting the shared clause list in.
    """
    fields, pml = _build(**kwargs)
    verdict = family.null_constitutive_coverage(fields, pml, side)
    assert verdict.covered is True, (label, verdict.reasons)


@pytest.mark.parametrize("side", ["H", "E"])
def test_a_conductivity_is_covered_here_even_though_the_curl_refuses_it(family, side):
    """Coverage is a set PER SUB-STEP, not per run — this is the sharpest case.

    ``no_pml`` clause 8 refuses a conductive run's CURL because ``mp.Absorber``
    routes to a three-factor update. The constitutive sub-step is untouched by it
    (``coverage._grid_reasons`` keeps the conductivity per curl for that reason), so
    refusing here would cost ``TestAbsorber.test_absorber_2d`` two sub-steps for a
    hazard in a different one.
    """
    fields, pml = _build()
    fields.set_d_conductivity(numpy.full(fields.grid.shape, 0.5, dtype=numpy.float32))
    assert family.null_constitutive_coverage(fields, pml, side).covered is True


# ---------------------------------------------------------------------------
# The partition against every other constitutive predicate
# ---------------------------------------------------------------------------

def _residual(verdict):
    """Refusal reasons minus the backend clause — a laptop has no CuPy.

    Without this the overlap question is vacuous here: every kernel predicate opens
    with "array module is not cupy", so on NumPy they all refuse and any partition
    holds trivially. Dropping exactly that one reason is what the predicate-coverage
    harness's ``residual_reasons`` does, and it asks the question a device asks.
    """
    return [reason for reason in verdict.reasons if "not cupy" not in reason]


@pytest.mark.parametrize("side", ["H", "E"])
def test_the_null_and_the_ordinary_constitutive_predicate_are_a_partition(family, side):
    """Both ask ``pml.is_active``; exactly one may admit any configuration."""
    plain_fields, no_layer = _build()
    pml_fields, layer = _build(pml_thickness=2, storage=True)

    plain_here = family.null_constitutive_coverage(plain_fields, no_layer,
                                                   side).covered
    plain_there = _residual(coverage_module.constitutive_coverage(
        plain_fields, no_layer, side)) == []
    pml_here = family.null_constitutive_coverage(pml_fields, layer, side).covered
    pml_there = _residual(coverage_module.constitutive_coverage(
        pml_fields, layer, side)) == []

    assert (plain_here, plain_there) == (True, False)
    assert (pml_here, pml_there) == (False, True)


def test_the_two_side_tables_name_the_same_slots(family):
    """``NULL_SIDES`` spells its keys out rather than importing ``coverage``'s.

    Its VALUES are deliberately not array names — this family reads and writes no
    array — but the KEY sets have to agree or a composer that passes one ``side``
    string to either would silently drop a sub-step.
    """
    assert set(family.NULL_SIDES) == set(coverage_module.CONSTITUTIVE_SIDES)
    assert {spec["sub_step"] for spec in family.NULL_SIDES.values()} == {
        "update_H", "update_E"}


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_plan_builder_refuses_exactly_where_the_predicate_does(family):
    fields, pml = _build(pml_thickness=2, storage=True)
    for side in ("H", "E"):
        assert family.plan_null_constitutive(fields, pml, side) is None
    plain_fields, no_layer = _build()
    for side in ("H", "E"):
        assert family.plan_null_constitutive(plain_fields, no_layer, side) is not None


def test_the_plan_counts_its_runs_because_there_is_no_launch_to_count(family):
    """``runs`` is the DISARMED guard for a kernel-less family.

    A leg that certifies "the plan reproduced the array path" by comparing bytes is
    trivially satisfied by a plan that was never invoked. Here the thing to count
    cannot be a launch, so it is this.
    """
    plan = family.NullConstitutivePlan("E")
    assert plan.runs == 0 and plan.sub_step == "update_E"
    plan.run()
    plan.run(guard=True)
    plan.run(guard=False)
    assert plan.runs == 3


def test_the_plans_run_body_is_the_counter_and_nothing_else():
    """The SHIPPED body, pinned structurally — source-text evidence, and labelled so.

    ``NullConstitutivePlan.run`` reproduces a bare ``return``. Anything it does beyond
    advancing its own counter is work the array path does not do, and a hook that
    currently does nothing is a hook a later edit fills in.

    ITS RELATIONSHIP TO THE GATE'S m3 IS NOT WHAT THIS DOCSTRING USED TO SAY. It read
    as "the merge-bar twin of m3, which arms exactly that edit"; m3 in fact injected a
    method whose body was ``pass``, moved 0 words, and reported a ``hasattr`` check —
    true by construction of its own edit — as its flip. So this AST pin was the ONLY
    evidence in the tranche about the shipped body, and it is the lowest rank this
    package recognises. m3 now binds the ``Fields`` and clears ``Bx``, and is caught
    by the gate's own uint32 identity comparison
    (``test_the_m3_mutant_really_writes_and_the_shipped_run_body_does_not``). This
    test is kept as the cheap structural guard it always was, not as the measurement.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    body = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == "NullConstitutivePlan"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "run":
                    body = item.body
    assert body is not None, "NullConstitutivePlan.run is gone"
    statements = [ast.unparse(node) for node in body
                  if not (isinstance(node, ast.Expr)
                          and isinstance(node.value, ast.Constant))]
    assert statements == ["del guard", "self.runs += 1"], statements
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        assert not isinstance(node, (ast.Call, ast.Subscript)), (
            f"run() performs {ast.unparse(node)}; the array path performs nothing")


def test_running_the_plan_by_itself_moves_no_byte_of_a_seeded_fields(family):
    """The behavioural half: the plan is not the thing that writes.

    ``run()`` is called in place of a sub-step, so if it touched an array the
    substitution would differ from the array path even where the predicate is right.
    Measured on a fully seeded state so "no bytes moved" is not measured on zeros.
    """
    fdtd = _driver()
    _seed_state(fdtd.fields)
    census = _census(fdtd.fields)
    assert census["signed_zero"] > 0 and census["subnormal"] > 0, census
    plan = family.plan_null_constitutive_step(fdtd.fields, fdtd.pml)
    before = _snapshot(fdtd.fields)
    for _ in range(4):
        plan.run()
    assert _differing_words(before, _snapshot(fdtd.fields)) == 0
    assert all(p.runs == 4 for p in plan.plans.values())


def test_the_plan_is_not_a_NoopPlan_and_carries_no_absorbed_by(family):
    """The distinction is the reason the class exists.

    ``launch.NoopPlan`` is the sentinel a FUSED PAIR leaves in an absorbed sub-step's
    slot: it means "the work happened, over there". This means "there was no work".
    Reporting the first where the second is true would tell a composer to look for a
    kernel that does not exist.
    """
    from meep_gpu.triton_kernels import launch as launch_module

    plan = family.NullConstitutivePlan("H")
    assert not isinstance(plan, getattr(launch_module, "NoopPlan", ()))
    assert not hasattr(plan, "absorbed_by")


def test_the_block_argument_is_accepted_and_ignored(family):
    """No launch, no grid, no block — accepted only so the signature matches."""
    fields, pml = _build()
    plan = family.plan_null_constitutive(fields, pml, "H", block=256)
    assert plan is not None and plan.block == 256
    plan.run()
    assert plan.runs == 1


def test_the_step_plan_reports_the_covered_subset_with_reasons_for_the_rest(family):
    fields, pml = _build()
    fields.enable_field_storage()
    plan = family.plan_null_constitutive_step(fields, pml)
    assert plan.covered == ("update_H",)
    assert "update_E" in plan.refusals and plan.refusals["update_E"]
    plan.run()
    assert plan.plans["update_H"].runs == 1


def test_an_unknown_side_is_rejected_by_the_plan_constructor(family):
    with pytest.raises(ValueError, match="side must be one of"):
        family.NullConstitutivePlan("step_B")


# ---------------------------------------------------------------------------
# Arm S — the record, and the two measurements behind the decision not to build it
# ---------------------------------------------------------------------------

def test_the_stored_e_arm_is_recorded_as_built_by_its_dedicated_family(family):
    record = family.STORED_E_ARM
    assert record["built"] is True
    assert "no_pml_stored_e.py" in record["built_as"]
    assert record["sub_step"] == "update_E"
    for key in ("array_path", "displacement_minus_polarization",
                "differs_from_certified_body", "compose_from",
                "drive_field_consequence", "measured_demand"):
        assert record[key], key


def test_the_stored_e_refusal_record_is_always_non_empty(family):
    """It is a record, not a predicate — it must never read as "covered"."""
    fields, pml = _build()
    assert family.stored_e_constitutive_reasons(fields, pml)
    fields.enable_field_storage()
    assert family.stored_e_constitutive_reasons(fields, pml)
    pml_fields, layer = _build(pml_thickness=2, storage=True)
    assert family.stored_e_constitutive_reasons(pml_fields, layer)


def test_the_no_pml_store_really_is_D_times_inv_eps_to_the_word():
    """The transcription of stepping.py:1010-1013 + :1022, measured not quoted.

    This is the product Arm S would have to reproduce, so the claim that it is a
    PLAIN STORE — and not the certified accumulation — starts here.
    """
    from meep_gpu import stepping

    fields, pml = _build()
    fields.enable_field_storage()
    _seed_state(fields)
    expected = {
        component: numpy.asarray(getattr(fields, "D" + component[1])
                                 * fields.inverse_epsilon_for(component))
        for component in ("Ex", "Ey", "Ez")}
    stepping.update_E(fields, pml)
    moved = 0
    for component, want in expected.items():
        got = getattr(fields, component)
        assert numpy.array_equal(_words(want), _words(got)), component
        moved += int(numpy.count_nonzero(_words(want) != _words(numpy.zeros_like(got))))
    assert moved > 0, "the store wrote nothing; the measurement is vacuous"


def test_binding_the_certified_accumulation_to_a_store_canonicalises_signed_zero():
    """The measurement that makes Arm S a NEW kernel rather than a restated predicate.

    ``kernels.constitutive_step``'s body is ``f = (f + kps*src) - kms*prev``. Binding
    ``kps=1``, ``kms=0`` and a zeroed ``f`` looks like a store and is not one:
    ``0.0 + (-0.0)`` is ``+0.0`` in IEEE-754 round-to-nearest, so every negative zero
    in the source is canonicalised away. One differing word in four, on this host.
    """
    src = numpy.array([-0.0, 0.0, 1.5, -2.5], dtype=numpy.float32)
    stored = src.copy()
    accumulated = (numpy.float32(0.0) + numpy.float32(1.0) * src).astype(numpy.float32)
    differing = int(numpy.count_nonzero(_words(stored) != _words(accumulated)))
    assert differing == 1, (_words(stored), _words(accumulated))
    assert _words(stored)[0] == numpy.uint32(0x80000000)
    assert _words(accumulated)[0] == numpy.uint32(0x00000000)


# ---------------------------------------------------------------------------
# THE REFERENCE LEG — the null against real ``stepping``, over multiple cycles
# ---------------------------------------------------------------------------

IDENTITY_CYCLES = 4


@pytest.mark.parametrize("label,kwargs", [
    ("plain_3d_np2", {"courant": 0.4056}),
    ("plain_3d_p2", {"courant": 0.5}),
    ("plain_2d_metallic", {"cell_size": (2.0, 2.0, 0.0), "dimensions": 2,
                           "boundaries": {"x": "metallic", "y": "metallic"}}),
    ("plain_1voxel", {"cell_size": (0.0, 0.0, 0.1), "dimensions": 1}),
    ("plain_zero_sigma_pol", {"zero_sigma": True}),
    ("plain_with_source", {"source": True}),
])
def test_the_array_path_moves_no_byte_over_multiple_full_cycles(family, label, kwargs):
    """THE claim of the family, against the real ``stepping`` functions.

    Seed every live array non-degenerately, snapshot its uint32 words, run
    ``IDENTITY_CYCLES`` complete ``update_H``+``update_E`` cycles with the inert
    layer, and require zero differing words against the ORIGINAL snapshot after each
    one. Multiple cycles because a sub-step that wrote back exactly what it read
    would pass a single call and drift on the second; against the original because a
    drift that cancels between two cycles must still be caught.

    The census is a PASS CONDITION, not a statistic: a zero-init ``Fields`` satisfies
    "no bytes moved" trivially, which is the failure mode for a null family. A
    strided plant misses a one-element array, and a 1-voxel cell is a real corpus row
    — so the seeder rotates a class per array and ``plain_1voxel`` is carried here
    for exactly that reason.
    """
    from meep_gpu import stepping

    fdtd = _driver(**kwargs)
    fields, pml = fdtd.fields, fdtd.pml
    _seed_state(fields)
    census = _census(fields)
    assert census["nonzero"] > 0 and census["signed_zero"] > 0, census
    assert census["subnormal"] > 0, census

    for side in ("H", "E"):
        assert family.null_constitutive_coverage(fields, pml, side).covered

    plan = family.plan_null_constitutive_step(fields, pml)
    before = _snapshot(fields)
    for cycle in range(1, IDENTITY_CYCLES + 1):
        stepping.update_H(fields, pml)
        stepping.update_E(fields, pml)
        plan.run()
        assert _differing_words(before, _snapshot(fields)) == 0, (label, cycle)
    assert all(p.runs == IDENTITY_CYCLES for p in plan.plans.values())
    assert _census(fields) == census, "the seeded classes did not survive the cycles"


@pytest.mark.parametrize("label,kwargs,must_move", [
    ("active_pml", {"pml": 3}, ("H", "E")),
    ("active_pml_p2_courant", {"pml": 3, "courant": 0.5}, ("H", "E")),
    ("no_pml_dispersive", {"dispersion": True}, ("E",)),
    ("no_pml_dispersive_1voxel", {"dispersion": True, "cell_size": (0.0, 0.0, 0.1),
                                  "dimensions": 1}, ("E",)),
])
def test_the_control_configurations_move_bytes_AND_are_refused(family, label, kwargs,
                                                               must_move):
    """The leg that makes the identity test a measurement instead of a tautology.

    BOTH halves are pass conditions. If the array path moved no bytes here, "no bytes
    moved" above would be unfalsified; if the predicate admitted these, the family
    would be substituting a null for a sub-step that writes.
    """
    from meep_gpu import stepping

    fdtd = _driver(**kwargs)
    fields, pml = fdtd.fields, fdtd.pml
    _seed_state(fields)
    call = {"H": stepping.update_H, "E": stepping.update_E}
    for side in must_move:
        assert family.null_constitutive_coverage(fields, pml, side).covered is False
        before = _snapshot(fields)
        call[side](fields, pml)
        assert _differing_words(before, _snapshot(fields)) > 0, (label, side)


def test_the_null_forced_past_its_own_predicate_diverges_on_a_real_driver(family):
    """The negative control, in the suite as well as in the gate.

    Everything above reports "identical", which is also what a harness that never ran
    reports. Force the null into a driver whose layer IS active — the substitution
    the predicate exists to forbid — and the same loop must diverge.
    """
    from meep_gpu import driver as driver_module

    steps = 4
    reference = _driver(pml=3, source=True)
    forced = _driver(pml=3, source=True)
    for side in ("H", "E"):
        assert family.null_constitutive_coverage(forced.fields, forced.pml,
                                                 side).covered is False

    plans = {"update_H": family.NullConstitutivePlan("H"),
             "update_E": family.NullConstitutivePlan("E")}
    calls = {"update_H": 0, "update_E": 0}

    def make(name):
        def substituted(fields, pml):
            del pml
            assert fields is forced.fields, "the patch leaked onto the reference driver"
            plans[name].run()
            calls[name] += 1
        return substituted

    originals = (driver_module.update_H, driver_module.update_E)
    divergence = None
    try:
        for step in range(1, steps + 1):
            reference.step()
            driver_module.update_H = make("update_H")
            driver_module.update_E = make("update_E")
            try:
                forced.step()
            finally:
                driver_module.update_H, driver_module.update_E = originals
            moved = _differing_words(_snapshot(reference.fields),
                                     _snapshot(forced.fields))
            if moved and divergence is None:
                divergence = (step, moved)
    finally:
        driver_module.update_H, driver_module.update_E = originals

    assert calls == {"update_H": steps, "update_E": steps}, calls
    assert divergence is not None, "the forbidden substitution did not diverge"
    assert divergence[0] == 1, divergence


def test_a_driven_run_and_a_source_free_run_do_not_agree(family):
    """Source non-vacuity: without it, the driven identity case proves nothing."""
    steps = 6
    driven = _driver(source=True)
    free = _driver()
    for _ in range(steps):
        driven.step()
        free.step()
    assert _differing_words(_snapshot(free.fields), _snapshot(driven.fields)) > 0


# ---------------------------------------------------------------------------
# The gate's own release contract
# ---------------------------------------------------------------------------

def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "gate_triton_no_pml_constitutive_contract", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _identity_shaped_case(name, np2=True, **extra):
    """One passing identity/breadth case, DEEP-COPIED so no two share a sub-dict.

    The fixture this replaces built its three cases with ``dict(identity_case, ...)``
    — a SHALLOW copy — so ``comparison``, ``census_before``, ``covered`` and
    ``plan_runs`` were the same objects in the identity leg's two cases AND in the
    breadth leg's one. Measured: ``identity[0]['comparison'] is breadth[0]['comparison']``
    was True. Every one of the contract-mutation tests below therefore mutated three
    cases at once, which is the mechanism that concealed the breadth leg's four
    contract holes: with the breadth case sharing the identity case's dicts, all four
    probes read as CLOSED; detached, all four were OPEN. A battery that cannot
    distinguish "the contract catches one bad case" from "the contract catches an
    all-bad leg" is not measuring the contract.
    """
    case = {
        "name": name, "np2_courant": bool(np2), "cycles": 3,
        "comparison": {"bit_identical": True, "differing_words": 0,
                       "total_words": 30240, "arrays_compared": 6, "per_array": {}},
        "census_before": {"total_words": 30240, "nonzero_words": 10080,
                          "signed_zero_words": 847, "subnormal_words": 777},
        "covered": {"H": True, "E": True},
        "plan_runs": {"update_H": 3, "update_E": 3},
        "polarization_arrays_compared": 0,
        "field_arrays_compared": 6,
        "pml_object_present": False,
        "requires_pml_object": False,
    }
    case.update(extra)
    return copy.deepcopy(case)


def _passing_payload(gate):
    """A payload that passes the release contract, built from the gate's own tables.

    Generated rather than hand-listed, because the contract now pins the LEG CONTENT:
    a leg stripped to its non-distinguishing control row is exactly the vacuity the
    breadth hole let through, so "these case names must all be here" is part of the
    contract and the fixture has to carry them.
    """
    identity = [
        _identity_shaped_case(
            spec["name"],
            np2=spec.get("courant", gate.COURANT_NP2) != gate.COURANT_P2,
            requires_pml_object=bool(spec.get("requires_pml_object")),
            pml_object_present=bool(spec.get("requires_pml_object")))
        for spec in gate.IDENTITY_CASES]
    breadth = [
        _identity_shaped_case(spec["name"],
                              refused_elsewhere=spec["refused_elsewhere"])
        for spec in gate.BREADTH_CASES]
    controls = []
    for spec in gate.CONTROL_CASES:
        dispersive = bool(spec.get("dispersion"))
        controls.append(copy.deepcopy({
            "name": spec["name"], "must_move": list(spec["must_move"]),
            "covered": {"H": not spec["must_move"] == ("H", "E") and dispersive,
                        "E": False},
            "polarization_arrays_compared": 6 if dispersive else 0,
            "polarization_held": True,
            "moved": {"H": {"differing_words": 0 if dispersive else 8972,
                            "polarization_words_moved": 0},
                      "E": {"differing_words": 9195,
                            "polarization_words_moved": 0}}}))
    overlap = [copy.deepcopy({
        "name": spec["name"], "side_tables_agree": True,
        "predicates_consulted": [entry["label"]
                                 for entry in gate.SIBLING_CONSTITUTIVE_PREDICATES],
        "admitted_modulo_backend": {"H": ["ordinary"], "E": ["ordinary"]}})
        for spec in (gate.CONTROL_CASES[0],)]
    engine = [copy.deepcopy({
        "name": "engine_driven", "first_divergence": None, "steps": gate.ENGINE_STEPS,
        "step_and_call_counts_agree": True, "substitution_leaks": 0,
        "substitution_calls": {"update_H": gate.ENGINE_STEPS,
                               "update_E": gate.ENGINE_STEPS},
        "step_count": {"reference": gate.ENGINE_STEPS,
                       "substituted": gate.ENGINE_STEPS},
        "reference_state_moved": {"differing_words": 9791, "total_words": 10080}})]
    return {
        "policy": {"policy": "host_ieee_keep"},
        "identity": {"numpy": {"ran": len(identity), "ok": len(identity),
                               "cases": identity}},
        "breadth": {"numpy": {"ran": len(breadth), "ok": len(breadth),
                              "cases": breadth}},
        "controls": {"numpy": {"ran": len(controls), "ok": len(controls),
                               "cases": controls}},
        "overlap": {"numpy": {"ran": len(overlap), "ok": len(overlap),
                              "cases": overlap}},
        "mutations": {"numpy": {
            "predicted_nulls": [{"name": "np2_courant", "reason": "nothing rounds"}],
            "mutations": {
                spec["name"]: {
                    "ok": True, "kind": spec["kind"], "sites": 1,
                    "mutant_sha256": "beef", "distinct_from_shipped": True,
                    "configurations_evaluated": 4, "verdict_flips": ["x@H"],
                    "runs": 1 if spec["kind"] == "byte_visible" else 0,
                    "caught_on": ["x@H"] if spec["kind"] == "byte_visible" else []}
                for spec in gate.MUTATIONS}}},
        "engine": {"numpy": {
            "ran": len(engine), "ok": len(engine), "cases": engine,
            "source_non_vacuity": {"ok": True, "differing_words": 9791},
            "negative_control": {"ok": True,
                                 "first_divergence": {"step": 1,
                                                      "differing_words": 30240},
                                 "predicate_refused": {"H": True, "E": True}}}},
    }


def test_the_gates_release_contract_passes_a_passing_artifact():
    gate = _load_gate()
    summary = gate.validate_payload(_passing_payload(gate))
    assert summary["status"] == "passed"
    assert set(summary["legs_validated"]) == {
        "identity", "breadth", "controls", "overlap", "mutations", "engine"}


@pytest.mark.parametrize("label,mutate,expected", [
    ("identity_moved_bytes",
     lambda p: p["identity"]["numpy"]["cases"][0]["comparison"].update(
         {"bit_identical": False, "differing_words": 7}),
     "moved bytes"),
    ("identity_vacuous_census",
     lambda p: p["identity"]["numpy"]["cases"][0]["census_before"].update(
         {"signed_zero_words": 0}),
     "VACUOUS"),
    ("identity_single_cycle",
     lambda p: p["identity"]["numpy"]["cases"][0].update({"cycles": 1}),
     "cycle"),
    ("identity_plan_never_ran",
     lambda p: p["identity"]["numpy"]["cases"][0]["plan_runs"].update(
         {"update_H": 0}),
     "DISARMED"),
    ("courant_control_dropped",
     lambda p: [c.update({"np2_courant": True})
                for c in p["identity"]["numpy"]["cases"]],
     "power-of-two"),
    # --- the identity holes the contract used to leave open ---------------
    ("identity_plan_runs_empty",
     lambda p: p["identity"]["numpy"]["cases"][0].update({"plan_runs": {}}),
     "DISARMED"),
    ("identity_plan_runs_names_one_side",
     lambda p: p["identity"]["numpy"]["cases"][0].update(
         {"plan_runs": {"update_E": 3}}),
     "DISARMED"),
    ("identity_compared_no_array",
     lambda p: p["identity"]["numpy"]["cases"][0]["comparison"].update(
         {"arrays_compared": 0}),
     "compared NO array"),
    ("identity_compared_no_word",
     lambda p: p["identity"]["numpy"]["cases"][0]["comparison"].update(
         {"total_words": 0}),
     "compared NO word"),
    ("inert_layer_row_carries_no_pml_object",
     lambda p: [c.update({"pml_object_present": False})
                for c in p["identity"]["numpy"]["cases"]
                if c.get("requires_pml_object")],
     "INERT LAYER"),
    # --- the breadth leg, which the contract did not check at all ---------
    ("breadth_moved_bytes",
     lambda p: p["breadth"]["numpy"]["cases"][0]["comparison"].update(
         {"bit_identical": False, "differing_words": 7}),
     "breadth case .* moved bytes"),
    ("breadth_vacuous_census",
     lambda p: p["breadth"]["numpy"]["cases"][0]["census_before"].update(
         {"signed_zero_words": 0}),
     "breadth case .* VACUOUS"),
    ("breadth_single_cycle",
     lambda p: p["breadth"]["numpy"]["cases"][0].update({"cycles": 1}),
     "breadth case .* cycle"),
    ("breadth_plan_never_ran",
     lambda p: p["breadth"]["numpy"]["cases"][0]["plan_runs"].update(
         {"update_H": 0}),
     r"breadth case .*DISARMED"),
    ("breadth_compared_no_array",
     lambda p: p["breadth"]["numpy"]["cases"][0]["comparison"].update(
         {"arrays_compared": 0}),
     "breadth case .* compared NO array"),
    ("breadth_stripped_to_its_wall_control",
     lambda p: p["breadth"]["numpy"].update(
         {"cases": [c for c in p["breadth"]["numpy"]["cases"]
                    if c["name"] == "metallic_walls"], "ran": 1, "ok": 1}),
     "missing its distinguishing cases"),
    # --- the overlap leg's own coverage of the package --------------------
    ("overlap_predicate_absent_from_the_leg",
     lambda p: p["overlap"]["numpy"]["cases"][0].update(
         {"predicates_consulted": ["ordinary"]}),
     "missing"),
    ("overlap_predicate_raised",
     lambda p: p["overlap"]["numpy"]["cases"][0].update(
         {"raised": {"bfast@H": "TypeError('side')"}}),
     "RAISED rather than refusing"),
    # --- the polarization pin, which lives on controls or nowhere ---------
    ("no_control_compared_a_polarization",
     lambda p: [c.update({"polarization_arrays_compared": 0})
                for c in p["controls"]["numpy"]["cases"]],
     "P/P_prev"),
    ("control_advanced_a_polarization",
     lambda p: p["controls"]["numpy"]["cases"][-1].update(
         {"polarization_held": False}),
     "advanced a polarization"),
    # --- the engine leg's raw counters, not its own boolean ---------------
    ("engine_substitution_never_took",
     lambda p: p["engine"]["numpy"]["cases"][0].update(
         {"substitution_calls": {"update_H": 0, "update_E": 0},
          "step_count": {"reference": 0, "substituted": 0}}),
     "was substituted 0 times"),
    ("engine_substitution_ran_twice_per_step",
     lambda p: p["engine"]["numpy"]["cases"][0]["substitution_calls"].update(
         {"update_H": 48}),
     "was substituted 48 times"),
    ("engine_patch_leaked",
     lambda p: p["engine"]["numpy"]["cases"][0].update({"substitution_leaks": 2}),
     "leaked onto"),
    ("engine_step_budget_shrank",
     lambda p: p["engine"]["numpy"]["cases"][0].update({"steps": 2}),
     "the stated budget"),
    # --- the vacuity hole a NULL family is uniquely exposed to -------------
    # Two drivers that both moved nothing agree bit for bit. Without this the engine
    # leg's "identical" verdict is satisfied by a build that failed to seed, a source
    # that never fired, or a spec zeroed everywhere — and the artifact would read as a
    # byte claim.
    ("engine_step_did_no_work",
     lambda p: p["engine"]["numpy"]["cases"][0]["reference_state_moved"].update(
         {"differing_words": 0}),
     "did not move across the whole run"),
    ("engine_stopped_recording_whether_the_step_worked",
     lambda p: p["engine"]["numpy"]["cases"][0].pop("reference_state_moved"),
     "did not move across the whole run"),
    # --- the leg set the release claim rests on ---------------------------
    ("engine_leg_missing_entirely",
     lambda p: p.pop("engine"),
     "a release claim needs every leg"),
    ("every_leg_but_identity_missing",
     lambda p: [p.pop(leg) for leg in
                ("controls", "breadth", "overlap", "mutations", "engine")],
     "a release claim needs every leg"),
    ("control_did_not_bite",
     lambda p: p["controls"]["numpy"]["cases"][0]["moved"]["H"].update(
         {"differing_words": 0}),
     "moved no bytes"),
    ("control_was_admitted",
     lambda p: p["controls"]["numpy"]["cases"][0]["covered"].update({"E": True}),
     "ADMITTED"),
    ("overlap_double_admission",
     lambda p: p["overlap"]["numpy"]["cases"][0]["admitted_modulo_backend"].update(
         {"H": ["null", "ordinary"]}),
     "OVERLAP"),
    ("overlap_vacuous",
     lambda p: p["overlap"]["numpy"]["cases"][0]["admitted_modulo_backend"].update(
         {"H": [], "E": []}),
     "vacuous"),
    ("side_tables_drifted",
     lambda p: p["overlap"]["numpy"]["cases"][0].update(
         {"side_tables_agree": False}),
     "disagree"),
    ("engine_diverged",
     lambda p: p["engine"]["numpy"]["cases"][0].update(
         {"first_divergence": {"step": 3}}),
     "diverged"),
    ("engine_call_counts_disagree",
     lambda p: p["engine"]["numpy"]["cases"][0].update(
         {"step_and_call_counts_agree": False}),
     "disagree"),
    ("source_control_vacuous",
     lambda p: p["engine"]["numpy"]["source_non_vacuity"].update({"ok": False}),
     "source-free control"),
    ("negative_control_did_not_diverge",
     lambda p: p["engine"]["numpy"]["negative_control"].update({"ok": False}),
     "negative control"),
    ("policy_stamp_missing",
     lambda p: p.pop("policy"),
     "policy"),
])
def test_the_gates_release_contract_fails_each_way_it_is_meant_to(label, mutate,
                                                                 expected):
    """The contract is mutation-tested, because an unfalsified contract is decoration.

    Every entry here is a way the artifact could look green while the measurement
    underneath it means nothing — a vacuous census, a control that did not bite, a
    partition leg that would pass on a package with one predicate in it.
    """
    gate = _load_gate()
    payload = copy.deepcopy(_passing_payload(gate))
    mutate(payload)
    with pytest.raises(AssertionError, match=expected):
        gate.validate_payload(payload)


def test_the_contract_fixtures_cases_are_independent_objects():
    """The shallow-copy defect, pinned so it cannot come back.

    Three cases used to share ``comparison``, ``census_before``, ``covered`` and
    ``plan_runs``, so every mutation test above mutated three cases at once. That is
    what made the breadth leg's holes read as closed: the breadth case was literally
    the identity case's dicts.
    """
    gate = _load_gate()
    payload = _passing_payload(gate)
    cases = (payload["identity"]["numpy"]["cases"]
             + payload["breadth"]["numpy"]["cases"])
    for key in ("comparison", "census_before", "covered", "plan_runs"):
        identities = [id(case[key]) for case in cases]
        assert len(set(identities)) == len(identities), key


def test_mutating_one_case_produces_a_failure_about_exactly_that_case():
    """The battery must be able to tell one bad case from an all-bad leg."""
    gate = _load_gate()
    payload = copy.deepcopy(_passing_payload(gate))
    target = payload["identity"]["numpy"]["cases"][0]["name"]
    payload["identity"]["numpy"]["cases"][0]["comparison"].update(
        {"bit_identical": False, "differing_words": 7})
    with pytest.raises(AssertionError) as excinfo:
        gate.validate_payload(payload)
    moved = [part for part in str(excinfo.value).split("; ") if "moved bytes" in part]
    assert moved == [f"identity case {target} moved bytes: 7"], moved


@pytest.mark.parametrize("name", ["m1_drop_inactive_layer_clause",
                                  "m2_drop_stores_e_clause",
                                  "m3_plan_run_writes"])
def test_each_byte_visible_mutation_must_have_run_a_plan_and_caught_something(name):
    gate = _load_gate()
    payload = copy.deepcopy(_passing_payload(gate))
    payload["mutations"]["numpy"]["mutations"][name]["runs"] = 0
    with pytest.raises(AssertionError, match="ran no null plan"):
        gate.validate_payload(payload)

    payload = copy.deepcopy(_passing_payload(gate))
    payload["mutations"]["numpy"]["mutations"][name]["caught_on"] = []
    with pytest.raises(AssertionError, match="moved no bytes"):
        gate.validate_payload(payload)


def test_the_gates_mutation_needles_still_match_the_shipped_source():
    """A text-edit mutation that matches nothing measures the SHIPPED module.

    ``sites == 0`` is the gate's own NEEDLE MISSED path, but it only fires when the
    gate runs. This is the merge-bar version: a prose edit to the predicate that
    silently detunes every needle fails HERE, on a laptop, before anything is
    submitted.
    """
    gate = _load_gate()
    source = gate.shipped_source()
    for spec in gate.MUTATIONS:
        mutated, sites = spec["fn"](source)
        assert sites > 0, f"{spec['name']} matched nothing in the shipped source"
        assert mutated != source, spec["name"]


def test_the_gate_records_its_predicted_nulls_with_reasons():
    """Predicted nulls are RECORDED, never silently dropped."""
    gate = _load_gate()
    names = {entry["name"] for entry in gate.PREDICTED_NULLS}
    assert "non_power_of_two_courant_on_the_covered_arm" in names
    assert "ptx_distinctness_of_mutants" in names
    assert "cupy_ftz_vs_triton_ieee_subnormal_policy" in names
    for entry in gate.PREDICTED_NULLS:
        assert entry["reason"].strip(), entry["name"]


# ---------------------------------------------------------------------------
# The gate's case builder — the leg that could not be built at all
# ---------------------------------------------------------------------------

def test_the_gates_device_path_asks_the_driver_for_a_device_rather_than_relabelling():
    """``--backend cupy`` used to raise TypeError on the FIRST case.

    ``build`` constructed a HOST driver and then reassigned ``fdtd.fields.grid.xp``,
    which allocates every array on the host (``FdtdDriver`` has no ``xp`` parameter and
    resolves its module at driver.py:943), leaves ``fdtd.xp`` NumPy, and makes
    ``seed_arrays`` assign a device array into a NumPy one. Measured with a stand-in
    module whose ``asarray`` refuses host conversion exactly as ``cupy.ndarray`` does:
    ``TypeError: Implicit conversion to a NumPy array is not allowed``. Under the
    slurm's ``set -euo pipefail`` that aborted the job before its only GPU leg.

    What is pinned is that the request goes to the ENGINE and that it cannot degrade
    silently. THE REFUSAL THAT FIRES DEPENDS ON THE HOST, and this test used to assert
    only the laptop's one, which made it fail on the only kind of machine the device
    leg can run on (measured on the GPU host, 2026-08-13: 1 of 2 suite failures there):

    * no GPU at all — ``resolve_backend(prefer_gpu=True)`` raises "GPU backend
      requested but unavailable" (``backends._gpu_route``) instead of quietly handing
      back a host driver wearing a device label;
    * an Apple GPU — ``prefer_gpu=True`` resolves Metal kernels over NumPy host
      arrays (``driver.gpu == "metal"``, ``xp`` NumPy), which is not the module the
      case asked for, so ``build``'s own post-condition raises and names both modules;
    * a CUDA device — the engine succeeds and resolves ``cupy``, which is not the
      module the case asked for either, and the same post-condition raises.

    Either way the guarantee is identical and is what the test asserts: a ``RuntimeError``
    naming the mismatch, raised BEFORE seeding. ``_NotNumpy.asarray`` raises
    ``AssertionError`` rather than ``RuntimeError``, so ``pytest.raises(RuntimeError)``
    also excludes the path where seeding was reached at all.
    """
    gate = _load_gate()
    spec = dict(gate.IDENTITY_CASES[0])

    class _NotNumpy:
        __name__ = "not_numpy"

        def asarray(self, host):  # pragma: no cover - reached only by the defect
            raise AssertionError("build must not reach seeding on a mislabelled driver")

    with pytest.raises(RuntimeError) as excinfo:
        gate.build(spec, _NotNumpy())
    message = str(excinfo.value)
    assert ("GPU backend requested but unavailable" in message
            or ("not_numpy" in message and "would not be attributable" in message)), (
        "build must refuse a mislabelled backend by name — on a host with no CUDA via "
        "resolve_backend, on a CUDA host via its own post-condition — and it raised "
        f"neither: {message!r}")

    # The NumPy path still resolves to NumPy, and the post-condition is real.
    fdtd = gate.build(spec, numpy)
    assert fdtd.xp is numpy and fdtd.fields.grid.xp is numpy

    # Secondary, and named as the weaker evidence it is: the relabelling assignment is
    # gone from the EXECUTABLE text (``_executable_text`` blanks every string, so the
    # docstring that records the defect does not satisfy its own pin).
    assert "grid.xp = xp" not in _executable_text(GATE_PATH)


def test_the_policy_stamp_is_taken_after_the_legs_and_not_only_before_them(tmp_path):
    """The stamp used to be a number that could not have been anything but zero.

    ``policy_stamp`` reads LIVE counters off the installed -ftz strip
    (``gate_triton_complex.install_ftz_strip``: ``calls`` and ``removed`` are
    incremented by the NVRTC wrapper). The gate called it once, in the dict literal
    that opens ``main``, BEFORE the first leg — so every artifact recorded
    ``nvrtc_calls: 0, ftz_removed: 0`` whatever ran afterwards, and this family's own
    docstring then cited that zero as confirmation that it "compiles nothing".

    MEASURED on the GPU host, 2026-08-13: the CuPy run that printed ``nvrtc_calls: 0``
    left 48 ``.cubin`` files in its own private ``CUPY_CACHE_DIR``. The family's
    PRODUCT does compile nothing — it ships no kernel — but the gate's harness is
    ordinary CuPy elementwise work and the controls leg does real active-PML
    arithmetic, so the seam is exercised and the zero was false.

    Pinned by counting calls rather than by reading the artifact's numbers, so it
    holds on a host with no CuPy at all: the stamp must be taken again AFTER the legs,
    and the later reading is the one the artifact reports.
    """
    gate = _load_gate()
    seen = {"calls": 0}
    real = gate.policy_stamp

    def counting_stamp(backend_name):
        seen["calls"] += 1
        stamp = dict(real(backend_name))
        stamp["nvrtc_calls"] = seen["calls"] * 10  # a number that must change
        return stamp

    gate.policy_stamp = counting_stamp
    try:
        out = tmp_path / "gate.json"
        assert gate.main(["--backend", "numpy", "--out", str(out)]) == 0
    finally:
        gate.policy_stamp = real

    payload = json.loads(out.read_text())
    policy = payload["policy"]
    assert seen["calls"] >= 2, (
        "the policy must be stamped again after the legs; a single stamp taken before "
        f"the first leg measures nothing (calls={seen['calls']})")
    assert "measured_after_legs" in policy and "at_startup" in policy, sorted(policy)
    assert policy["at_startup"]["nvrtc_calls"] == 10
    assert policy["measured_after_legs"]["nvrtc_calls"] > 10, (
        "the artifact must report the reading taken after the legs ran, and keep the "
        "startup reading labelled beside it rather than in its place")


def test_the_byte_identity_claim_is_derived_from_the_backend_that_ran(tmp_path):
    """The record used to deny the existence of the run that wrote it.

    ``provenance`` carried a frozen ``"NOT CLAIMED ON A DEVICE. No device leg has been
    run."``, emitted verbatim by every artifact — so the first green ``--backend cupy``
    run on the GPU host produced a device artifact stating that no device leg had run. A
    provenance field that cannot change is not provenance.
    """
    gate = _load_gate()
    host = gate.provenance(numpy, "numpy")["byte_identity_claim"]
    device = gate.provenance(numpy, "cupy")["byte_identity_claim"]
    assert host != device
    assert "HOST ONLY" in host
    assert "No device leg has been run" not in device
    # And the narrow claim stays narrow: this family certifies a SUBSTITUTION, never a
    # kernel, because it ships none.
    assert "NO KERNEL IS CERTIFIED HERE" in device


def test_the_gate_reads_and_writes_its_own_source_under_an_ascii_process_locale():
    """The mutation battery could not run on ANY CUDA host, and this is the pin.

    MEASURED on the GPU host, 2026-08-13, in one process:
    ``locale.getpreferredencoding(False)`` is ``UTF-8`` at start, still ``UTF-8``
    after ``import cupy``, and ``ANSI_X3.4-1968`` after the first device allocation —
    CUDA context initialisation calls ``setlocale`` and resets ``LC_CTYPE``. Every
    ``open`` that defers to the locale therefore decodes as ASCII from that moment on,
    and the shipped module's text is full of em-dashes.

    What that cost, before the fix: ``gate --backend cupy`` completed identity,
    controls, breadth and overlap and then died at the mutations leg with
    ``UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 240``, and
    the suite failed the same way in the test that immediately follows the first one
    to build a device driver. A gate that cannot read its own source on the only host
    with a GPU has no armed mutations there at all — the leg is not weakened, it is
    absent.

    Pinned WITHOUT a GPU by reproducing the state rather than the cause:
    ``PYTHONCOERCECLOCALE=0 LC_ALL=C`` gives the interpreter the same ASCII preferred
    encoding, and produced the identical traceback on this laptop before the fix. A
    subprocess is the only honest way to hold it — the encoding is chosen at ``open``
    time from process-global state that a fixture cannot restore.

    Both directions are exercised: ``shipped_source`` reads, and ``compile_mutant``
    writes the same non-ASCII text back out and imports it.
    """
    import subprocess  # noqa: PLC0415 - only this test needs a second interpreter

    program = (
        "import locale, sys\n"
        f"sys.path.insert(0, {str(GATE_PATH.parent)!r})\n"
        f"sys.path.insert(0, {str(PACKAGE_DIR.parents[1])!r})\n"
        "import gate_triton_no_pml_constitutive as gate\n"
        "assert locale.getpreferredencoding(False).lower() in "
        "('ascii', 'us-ascii', 'ansi_x3.4-1968'), locale.getpreferredencoding(False)\n"
        "source = gate.shipped_source()\n"
        "assert '\\u2014' in source, 'the needle character is gone; this test is vacuous'\n"
        "module = gate.compile_mutant(source, 'locale_probe')\n"
        "assert module.null_constitutive_coverage is not None\n"
        "print('OK', len(source))\n"
    )
    environment = dict(os.environ,
                       PYTHONCOERCECLOCALE="0", LC_ALL="C", LANG="C", PYTHONUTF8="0")
    completed = subprocess.run([sys.executable, "-c", program], env=environment,
                               capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, (
        "the gate must read and rewrite its own source under an ASCII process locale "
        "— the state every CUDA host is in after its first device allocation. "
        f"stdout={completed.stdout!r} stderr={completed.stderr[-2000:]!r}")
    assert completed.stdout.startswith("OK"), completed.stdout


def test_the_inert_layer_case_really_installs_an_inert_layer(family):
    """The row named ``plain_inert_pml_object`` used to carry no PML object at all.

    ``if spec.get("pml")`` is falsy for ``0.0``, so ``setup_pml`` was skipped and the
    case was a salt-only duplicate of ``plain_3d_np2`` — measured: identical shape,
    identical ``arrays_compared``, identical census. ``setup_pml(0.0)`` could not have
    produced it either (driver.py:2162-2166 raises on a request that absorbs nowhere).
    The clause this family calls its load-bearing partition test — a layer PRESENT and
    inactive — was measured by no byte-level leg anywhere in the tranche.
    """
    gate = _load_gate()
    spec = [case for case in gate.IDENTITY_CASES
            if case["name"] == "plain_inert_pml_object"][0]
    assert spec.get("requires_pml_object") is True
    fdtd = gate.build(dict(spec), numpy)
    assert fdtd.pml is not None
    assert fdtd.pml.is_active is False
    entry = gate.one_identity_case(dict(spec), numpy)
    assert entry["pml_object_present"] is True
    assert entry["pml_active"] is False
    assert entry["comparison"]["bit_identical"] is True
    assert entry["comparison"]["arrays_compared"] > 0
    assert entry["ok"] is True


def test_the_overlap_leg_consults_every_constitutive_predicate_in_the_package():
    """It asked four of fifteen, and the eleven it skipped own the breadth grids.

    The partition claim was therefore unmeasured exactly where it is most interesting:
    a folded, complex, Bloch, beta, BFAST or cylindrical grid is where a SIBLING
    predicate is designed to admit. The declared table is checked against the package
    by name here, so a predicate added beside this family cannot quietly stay out.
    """
    gate = _load_gate()
    declared = {entry["label"] for entry in gate.SIBLING_CONSTITUTIVE_PREDICATES}
    predicates, failures = gate.sibling_constitutive_predicates()
    assert failures == {}, failures
    assert set(predicates) == declared

    # DISCOVERY IS FAIL-CLOSED, AND IT IS NOT BY NAME.
    #
    # This test used to look for functions ending in ``constitutive_coverage``.
    # Measured 2026-08-17: THREE constitutive predicates across two wired
    # families do not carry that suffix — complex_stored_e_coverage and both of
    # complex_offdiag_update_e's — so the guard reported a completeness it was
    # not checking, and the overlap leg never consulted them. A predicate that
    # is never asked cannot be found to double-admit, and a double admission
    # leaves the slot UNSELECTED and silently on the array path.
    #
    # So: EVERY ``*_coverage`` function in the package must be classified. A new
    # one is a constitutive predicate that must be declared, UNLESS its name
    # says it serves another slot — a curl, a ghost fill, an update_P, or one of
    # the opt-in fused products that are not arms at all. Adding a predicate
    # whose name matches none of those patterns FAILS here until someone
    # decides which it is, which is the point: the cost of a new family is one
    # deliberate classification, not a silent gap.
    NON_CONSTITUTIVE_MARKERS = (
        "_curl_coverage",        # step_B / step_D products
        "_fill_",                # mirror ghost fills — seam passes, not arms
        "update_p_coverage",     # the polarization slot, its own overlap question
        "fused_pair_coverage",   # opt-in cross-sub-step, no slot (launch.py:942)
        "fused_ade_state_coverage",
        "fused_dispersive_chain_coverage",
        # The deferred E->P weld, 2026-08-21: update_E welded to update_P. It
        # spans TWO slots and claims no constitutive slot of its own, so the
        # overlap leg must not ask it whether it double-admits one. Its own
        # disjointness question is a different one — exactly one ARM admits each
        # configuration — and test_triton_fused_ade_chain asks it directly.
        "fused_ade_chain_coverage",
        # The deferred folded B/H fusion: it spans step_B, the near mirror fill,
        # zero_metal_B and update_H, so it claims no constitutive slot and the
        # overlap leg must not ask it whether it double-admits one.
        "fused_magnetic_pair_coverage",
        # ...and its D/E mirror image, 2026-08-30. `complex_fused_electric_pair`
        # spans step_D, both symmetry fills, zero_metal_D and update_E — FIVE
        # driver call sites — so it claims no constitutive slot either, for exactly
        # the reason the magnetic marker above gives. The classification is the
        # only thing that is new: the predicate is a cross-sub-step SEAM predicate,
        # not a slot claimant, and asking it whether it double-admits update_E
        # would be asking the wrong question of the wrong function.
        "fused_electric_pair_coverage",
        # THE H->D WELD, 2026-09-06, and it is the same classification for a seam the
        # other two markers do not cover. `fused_hd_pair_coverage` spans `update_H`
        # and `step_D` -- the FOURTH seam, and the first whose two halves sit in the
        # order constitutive-then-curl -- so it claims no constitutive slot of its
        # own. Asking it whether it double-admits `update_H` would be asking the wrong
        # question of the wrong function: its own disjointness question is whether the
        # composer may install it beside a neighbouring pair, and
        # `test_triton_fused_hd_pair` drives BOTH brakes that answer it
        # (`_pair_may_absorb` against the live `selected`, and the declared
        # `INSTALLABLE = False`).
        "fused_hd_pair_coverage",
        "_composition_coverage",  # composition helpers, not slot claimants
    )

    found, non_constitutive = set(), set()
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        if path.name in ("__init__.py", "launch.py"):
            continue  # lazy forwarders, not implementations
        module_name = f"meep_gpu.triton_kernels.{path.stem}"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.FunctionDef)
                    and node.name.endswith("_coverage")):
                continue
            # PRIVATE HELPERS ARE NOT SLOT PREDICATES, and cannot be: the overlap
            # leg looks its siblings up by MODULE AND PUBLIC NAME, so a leading
            # underscore is already outside the set it can consult. The only one
            # in the package today is fused_ade_chain._ade_coverage, a selector
            # that RETURNS one of the two update_P predicates the marker below
            # already excludes. The non-vacuity floor at the end of this test is
            # what stops this skip quietly emptying the scan.
            if node.name.startswith("_"):
                continue
            pair = (module_name, node.name)
            if any(marker in node.name for marker in NON_CONSTITUTIVE_MARKERS):
                non_constitutive.add(pair)
            else:
                found.add(pair)

    found.discard((MODULE_NAME, "null_constitutive_coverage"))
    declared_pairs = {(entry["module"], entry["name"])
                      for entry in gate.SIBLING_CONSTITUTIVE_PREDICATES}
    declared_pairs.add(("meep_gpu.triton_kernels.no_pml_stored_e",
                        "stored_e_constitutive_coverage"))
    assert found == declared_pairs, (
        "every constitutive predicate must be consulted by the overlap leg; "
        f"symmetric difference: {sorted(found ^ declared_pairs)}")
    # Non-vacuity: if the marker list ever swallowed everything, the assertion
    # above would pass against an empty set and prove nothing.
    assert len(found) >= 15, f"only {len(found)} constitutive predicates discovered"
    assert non_constitutive, "the marker list classified nothing as non-constitutive"


@pytest.mark.parametrize("side", ["H", "E"])
def test_no_sibling_predicate_admits_where_the_null_does(family, side):
    """The partition, asked of ALL of them rather than of the four general ones."""
    gate = _load_gate()
    predicates, _ = gate.sibling_constitutive_predicates()
    fields, pml = _build()
    assert family.null_constitutive_coverage(fields, pml, side).covered is True
    for label, predicate in sorted(predicates.items()):
        verdict = predicate(fields, pml, side)
        if verdict is None:
            continue
        residual = [r for r in verdict.reasons if "not cupy" not in r]
        assert residual, f"{label} admits modulo the backend clause where the null does"


def test_the_m3_mutant_really_writes_and_the_shipped_run_body_does_not():
    """m3 used to inject a method whose body was ``pass``.

    Its recorded "flip" was ``hasattr(plan, '_mutant_side_effect')`` — true by
    construction of the edit — and the compiled mutant moved 0 words against a fully
    seeded ``Fields``. The defect class it exists to arm ("the null plan performs
    work") was armed by nothing but a source-text AST pin. The mutant now binds the
    ``Fields`` it was built for and clears ``Bx``, and the gate drives it through
    ``one_identity_case`` — the ``module`` parameter that previously had no caller.
    """
    gate = _load_gate()
    source = gate.shipped_source()
    mutated, sites = gate.mutate_plan_run_writes(source)
    assert sites == 3, sites
    module = gate.compile_mutant(mutated, "suite_m3")
    spec = dict(gate.IDENTITY_CASES[0])

    mutant_entry = gate.one_identity_case(spec, numpy, module=module)
    assert mutant_entry["comparison"]["bit_identical"] is False
    assert mutant_entry["comparison"]["differing_words"] > 0
    assert "Bx" in mutant_entry["comparison"]["per_array"]
    assert mutant_entry["ok"] is False

    shipped_entry = gate.one_identity_case(spec, numpy, module=_load_family())
    assert shipped_entry["comparison"]["bit_identical"] is True
    assert shipped_entry["ok"] is True

    # And the classification is honest: m3 is byte_visible, not a structural note.
    row = [spec for spec in gate.MUTATIONS if spec["name"] == "m3_plan_run_writes"][0]
    assert row["kind"] == "byte_visible"


def _load_family():
    return importlib.import_module(MODULE_NAME)


def test_the_identity_leg_cannot_carry_the_polarization_pin_and_says_so(family):
    """The gate claimed a pin the covered arm structurally cannot hold.

    P/P_prev exist on a DRIVEN run, and a driven run switches ``stores_E`` on, which
    the predicate refuses. The one identity case that registers a susceptibility
    registers a zero-sigma one whose ``P`` table is empty. So the count is recorded
    per case and the docstring points at ``controls``.
    """
    gate = _load_gate()
    for spec in gate.IDENTITY_CASES:
        entry = gate.one_identity_case(dict(spec), numpy)
        assert entry["polarization_arrays_compared"] == 0, spec["name"]
        assert entry["field_arrays_compared"] > 0, spec["name"]
    assert "The pin lives on ``controls``." in gate.__doc__


def test_update_E_consumes_a_polarization_without_advancing_it(family):
    """The pin itself, measured where the arrays exist: on the refused arm.

    stepping.py:969-973 is the contract. A control leg that measured "bytes moved"
    alone would pass on an ``update_E`` that advanced the polarization instead.
    """
    from meep_gpu import stepping

    gate = _load_gate()
    spec = [case for case in gate.CONTROL_CASES
            if case["name"] == "no_pml_dispersive"][0]
    fdtd = gate.build(dict(spec), numpy)
    polarization = sorted(gate.polarization_arrays(fdtd.fields))
    assert polarization, "the control carries no polarization array to pin"
    before = gate.snapshot(fdtd.fields)
    stepping.update_E(fdtd.fields, fdtd.pml)
    after = gate.snapshot(fdtd.fields)
    assert gate.diff(before, after)["differing_words"] > 0
    for name in polarization:
        assert int(numpy.count_nonzero(before[name] != after[name])) == 0, name




def test_the_gate_imports_the_shared_policy_machinery_rather_than_reimplementing_it():
    """``install_ftz_strip`` is measured machinery; a second copy is a second policy."""
    gate = _load_gate()
    assert gate.install_ftz_strip.__module__.endswith("gate_triton_complex")
    assert gate.STRIPPED_POLICY == "ieee_keep_ftz_stripped"
    source = GATE_PATH.read_text(encoding="utf-8")
    assert "def install_ftz_strip" not in source, "the strip was re-implemented"


def test_a_host_artifact_never_claims_device_bytes_and_never_claims_a_kernel():
    """What this family may claim, on each backend, and what it may never claim.

    This test used to read "Nothing here has run on a device. The artifacts must say
    so" and pinned the literal ``NOT CLAIMED``. A device leg HAS since run — direct,
    pinned to a verified-empty GPU on the GPU host, 2026-08-13 — so that sentence became
    false and the pin would have kept the artifact lying on purpose. The durable
    property is not "no device leg exists"; it is that the claim MATCHES the run, and
    that neither backend is ever allowed to claim a kernel, because the family ships
    none. Its backend-derivation is pinned separately, above.
    """
    gate = _load_gate()
    for backend in ("numpy", "cupy"):
        record = gate.provenance(numpy, backend)
        assert record["ships_a_kernel"] is False
        assert record["backend"] == backend
        claim = record["byte_identity_claim"]
        assert "kernel" in claim.lower(), (
            f"the {backend} claim must say what it does NOT certify: {claim!r}")
    host = gate.provenance(numpy, "numpy")["byte_identity_claim"]
    assert "device" in host.lower() and "says nothing about device" in host, (
        "a host artifact must disclaim device bytes in so many words")
    assert PROBE_PATH.exists(), "the composition probe is part of the family's trio"


# ---------------------------------------------------------------------------
# The composition probe's contract
# ---------------------------------------------------------------------------

def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_triton_no_pml_constitutive_contract", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _passing_probe_payload():
    stepped = {"name": "harminv_warnings_class", "np2_courant": True,
               "first_divergence": None, "step_and_call_counts_agree": True,
               "non_vacuous": True}
    return {
        "policy": {"policy": "host_ieee_keep"},
        "census": {"numpy": {"ran": 2, "ok": 2, "cases": [
            {"name": "harminv_warnings_class", "whole_step": True,
             "expected_whole_step": True},
            {"name": "control_active_pml", "whole_step": False,
             "expected_whole_step": False}]}},
        "constitutive": {"numpy": {"ran": 2, "ok": 2, "cases": [
            stepped, dict(stepped, name="p2_courant_control", np2_courant=False)],
            "source_non_vacuity": {"ok": True, "differing_words": 9786},
            "negative_control": {"ok": True,
                                 "first_divergence": {"step": 1}}}},
        "whole_step": {"numpy": {"ran": 0, "ok": 0, "skipped": True,
                                 "blockers": ["no CuPy array module"]}},
    }


def _ran_whole_step_payload():
    """A payload whose whole_step leg RAN — the shape only a CUDA host produces.

    The fixture above always records ``skipped: True``, which is honest for a laptop
    and is why every whole_step contract clause went unarmed on the merge bar: the
    guards are all inside ``if not block.get("skipped")``. The device leg on the GPU host
    is what makes this shape real, and the rows below are the numbers it produced
    (2026-08-13, 5 rows x 24 steps, all four heavy slots substituted).
    """
    payload = _passing_probe_payload()
    case = {"name": "harminv_warnings_class", "row": "TestSimulation.test_harminv_warnings",
            "steps": 24, "np2_courant": True, "courant": "0.4056",
            "first_divergence": None, "step_and_call_counts_agree": True,
            "substitution_leaks": 0,
            "slots_substituted": ["step_B", "step_D", "update_E", "update_H"],
            "substitution_calls": {"step_B": 24, "step_D": 24,
                                   "update_E": 24, "update_H": 24},
            "totals": {"differing_words": 0, "total_words": 241920, "comparisons": 24},
            "reference_state_moved": {"differing_words": 10080, "total_words": 10080,
                                      "arrays_moved": ["Bx", "By", "Bz",
                                                       "Dx", "Dy", "Dz"]}}
    payload["whole_step"] = {"numpy": {"ran": 1, "ok": 1, "skipped": False,
                                       "steps_per_case": 24, "total_steps": 24,
                                       "cases": [copy.deepcopy(case)]}}
    return payload


def test_the_probes_contract_passes_a_passing_artifact():
    probe = _load_probe()
    assert probe.validate_payload(_passing_probe_payload())["status"] == "passed"


def test_the_probes_contract_passes_a_payload_whose_whole_step_leg_ran():
    probe = _load_probe()
    summary = probe.validate_payload(_ran_whole_step_payload())
    assert summary["status"] == "passed"
    assert summary["whole_step_skipped"] is False


@pytest.mark.parametrize("label,mutate,expected", [
    # The vacuity hole this leg is uniquely exposed to: TWO of its four substituted
    # slots are nulls, so a run in which nothing moved reports "all four heavy slots
    # off the array path, bit-identical" and means nothing.
    ("whole_step_did_no_work",
     lambda p: p["whole_step"]["numpy"]["cases"][0]["reference_state_moved"].update(
         {"differing_words": 0}),
     "did not move across the run"),
    ("whole_step_stopped_recording_whether_it_worked",
     lambda p: p["whole_step"]["numpy"]["cases"][0].pop("reference_state_moved"),
     "did not move across the run"),
    # The denominator: "bit-identical over 24 steps" needs 24 comparisons, not the
    # mere absence of a divergence record.
    ("whole_step_compared_fewer_steps_than_it_claims",
     lambda p: p["whole_step"]["numpy"]["cases"][0]["totals"].update(
         {"comparisons": 3}),
     "comparisons for 24 steps"),
    ("whole_step_compared_no_words",
     lambda p: p["whole_step"]["numpy"]["cases"][0]["totals"].update(
         {"total_words": 0}),
     "compared 0 words"),
    ("whole_step_dropped_a_heavy_slot",
     lambda p: p["whole_step"]["numpy"]["cases"][0].update(
         {"slots_substituted": ["step_B", "step_D", "update_H"]}),
     "all four heavy slots"),
    ("whole_step_diverged",
     lambda p: p["whole_step"]["numpy"]["cases"][0].update(
         {"first_divergence": {"step": 7}}),
     "diverged"),
])
def test_the_probes_contract_catches_a_hollow_whole_step_leg(label, mutate, expected):
    probe = _load_probe()
    payload = _ran_whole_step_payload()
    mutate(payload)
    with pytest.raises(AssertionError, match=expected):
        probe.validate_payload(payload)


@pytest.mark.parametrize("label,mutate,expected", [
    ("no_row_closed_whole_step",
     lambda p: p["census"]["numpy"]["cases"][0].update(
         {"whole_step": False, "expected_whole_step": False}),
     "closed NO row whole-step"),
    ("every_row_closed",
     lambda p: p["census"]["numpy"]["cases"][1].update(
         {"whole_step": True, "expected_whole_step": True}),
     "not falsifiable"),
    ("prediction_disagrees",
     lambda p: p["census"]["numpy"]["cases"][0].update({"whole_step": False}),
     "recorded prediction"),
    ("constitutive_diverged",
     lambda p: p["constitutive"]["numpy"]["cases"][0].update(
         {"first_divergence": {"step": 2}}),
     "diverged"),
    ("patch_did_not_take",
     lambda p: p["constitutive"]["numpy"]["cases"][0].update(
         {"step_and_call_counts_agree": False}),
     "may not have taken"),
    ("courant_control_dropped",
     lambda p: p["constitutive"]["numpy"]["cases"].pop(),
     "power-of-two"),
    ("source_control_vacuous",
     lambda p: p["constitutive"]["numpy"]["source_non_vacuity"].update({"ok": False}),
     "source-free control"),
    ("negative_control_flat",
     lambda p: p["constitutive"]["numpy"]["negative_control"].update({"ok": False}),
     "negative control"),
    ("whole_step_skipped_silently",
     lambda p: p["whole_step"]["numpy"].pop("blockers"),
     "skipped without recording why"),
    # THE LEG THAT DISAPPEARS ENTIRELY. The guard used to fire only when the block
    # EXISTED and lacked blockers, so ``--legs census,constitutive`` — which produces
    # no whole_step block at all — passed with no statement about the device leg.
    # Measured before the fix: removing the block, and removing its backend row, both
    # returned status='passed'.
    ("whole_step_block_absent",
     lambda p: p.pop("whole_step"),
     "have no block at all"),
    ("whole_step_backend_row_absent",
     lambda p: p["whole_step"].pop("numpy"),
     "have no block at all"),
    ("census_leg_absent",
     lambda p: p.pop("census"),
     "have no block at all"),
    ("constitutive_leg_absent",
     lambda p: p.pop("constitutive"),
     "have no block at all"),
])
def test_the_probes_contract_fails_each_way_it_is_meant_to(label, mutate, expected):
    probe = _load_probe()
    payload = copy.deepcopy(_passing_probe_payload())
    mutate(payload)
    with pytest.raises(AssertionError, match=expected):
        probe.validate_payload(payload)


def test_the_probes_contract_records_which_legs_it_validated():
    probe = _load_probe()
    summary = probe.validate_payload(_passing_probe_payload())
    assert summary["legs_absent"] == []
    assert summary["whole_step_skipped"] is True
    assert summary["all_legs_required"] is True


def test_the_probe_enumerates_every_slot_of_a_whole_step():
    """A whole-step claim that quietly omits a slot is not a whole-step claim.

    ``update_P`` is in the list even though this family does not serve it: on the rows
    it closes, ``stepping.update_P`` returns at stepping.py:1425-1426 because nothing
    is registered.

    THE RECORD AND THE ARITHMETIC NOW AGREE ABOUT IT. The probe used to say that no-op
    was "recorded rather than counted as coverage" while the whole-step conjunction
    counted it — visible in the artifact as an absorbing control reporting
    ``cov ['update_P']``. Counting it is the right reading of "whole step" ("no heavy
    work is left on the array path"), so the wording was corrected and a SECOND column,
    :data:`HEAVY_SLOTS`, carries the other question: how much of it a product took.
    """
    probe = _load_probe()
    assert probe.STEP_SLOTS == ("step_B", "update_H", "step_D", "update_E",
                                "update_P")
    assert probe.HEAVY_SLOTS == ("step_B", "update_H", "step_D", "update_E")
    assert "update_P" not in probe.HEAVY_SLOTS
    assert set(probe.HEAVY_SLOTS) | {"update_P"} == set(probe.STEP_SLOTS)


def test_the_census_reports_the_heavy_slots_separately_from_the_whole_step_verdict():
    """An absorbing run must read as one array-path no-op, never as coverage."""
    probe = _load_probe()
    absorbing = [row for row in probe.ROWS if row["name"] == "control_active_pml"][0]
    fdtd = _load_gate().build(dict(absorbing), numpy)
    slots = probe.slot_coverage(fdtd.fields, fdtd.pml)
    covered = [name for name in probe.STEP_SLOTS
               if slots[name]["covered_modulo_backend"]]
    assert covered == ["update_P"]
    heavy = [name for name in probe.HEAVY_SLOTS
             if slots[name]["covered_modulo_backend"]]
    assert heavy == []
    assert "array-path no-op" in slots["update_P"]["product"]


def test_the_probe_refuses_to_fake_a_device_leg_and_says_why(tmp_path):
    """The whole_step leg needs CuPy AND Triton. Skipped is recorded, never silent."""
    probe = _load_probe()
    results = {}
    # tmp_path, not the parity directory: the leg writes its artifact as it goes, and
    # a merge-bar test must not leave one beside the gate it is testing.
    block = probe.run_whole_step(results, str(tmp_path / "whole_step.json"),
                                 numpy, "numpy")
    assert block["skipped"] is True
    assert block["ran"] == 0
    assert block["blockers"], "a skipped leg must record its blockers"
    assert "NO whole-step byte-identity claim" in block["claim"]


def test_the_probe_predicts_which_rows_close_and_which_do_not():
    """The predictions are recorded BEFORE the run, so the census can disagree.

    A census that only reports cannot fail. Each row carries ``expect_whole_step``
    and the contract compares the two — including three rows that must NOT close,
    each for a different named reason (a conductive curl, an absorbing run, and a
    dispersive run whose ``update_E`` is Arm S).
    """
    probe = _load_probe()
    expected = {row["name"]: row["expect_whole_step"] for row in probe.ROWS}
    assert sum(expected.values()) >= 4, "the four measured corpus rows are the claim"
    assert sum(1 for value in expected.values() if not value) >= 3, (
        "without negative rows the census is not falsifiable")
