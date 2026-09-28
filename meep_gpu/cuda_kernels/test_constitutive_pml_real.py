"""The real-storage constitutive slice: its predicate, its partition, its device text.

THE BIT-IDENTITY GATE LIVES ELSEWHERE — it needs a device and a sweep, so it is
``parity/meep_gpu/probe_fused_kernel_bit_identity.py --track hand --experiments
constitutive,constitutive_multistep``. What is pinned HERE is everything about
this slice that can be decided without a GPU, which after ``coverage.py`` went
CuPy-free is most of it:

  * the predicate that decides whether the kernels may run at all, case by case,
    with its two required mutations applied to the REAL function's source;
  * the ONE clause where this predicate is deliberately WIDER than the curl's —
    conductivity — checked against ``stepping.py``'s own text rather than
    against a belief about it;
  * the tables the predicate and the launcher share, pinned against
    ``stepping``'s term tables so a transcription cannot drift;
  * the properties of the CUDA source strings that decide bit-identity — the
    grouping, the load-before-store, the absence of a stencil and of a wall
    mask — read textually, so they fail on a laptop rather than hours into a
    device run;
  * the partition: every kernel shipped in ``constitutive_kernels.py`` is either
    certified with a record or listed as owing the gate. Right now all of it is
    the second kind, and the file says so.

WHY THE DEVICE-TEXT ASSERTIONS ARE TEXTUAL. ``constitutive_kernels`` imports
``cupy`` at module scope because it holds ``cp.RawKernel`` objects, so importing
it on this laptop is impossible. The kernel strings are still just text in a
file, and the three defects that would silently break the sub-step — a flattened
grouping, a store before the load, a dropped inverse epsilon — are all visible in
that text. ``test_certification_record.py`` established the pattern (it evaluates
the sibling module's device strings from the syntax tree without importing it);
this file reuses it.

THE BACKEND STAND-IN, AND WHAT IT IS NOT. Off device the fixture hands ``Grid``
a module that is NumPy wearing CuPy's ``__name__``, because the predicate's first
question is whether the backend is CuPy at all. Everything else it reads — dtype,
shape, contiguity, the PML vectors, the grid's own boundary resolution — is
exercised on REAL ``Grid``/``Fields``/``PML`` objects. On a CUDA host the same
tests run again against real CuPy arrays.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import pathlib
import re
import textwrap
import types

import numpy
import pytest

try:
    import cupy
except ImportError:  # pragma: no cover - exercised on any NumPy-only host
    cupy = None

from .. import stepping
from ..device_identity import weld_survives_edit
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import coverage

HERE = pathlib.Path(__file__).parent
KERNEL_MODULE = HERE / "constitutive_kernels.py"
SIBLING_MODULE = HERE / "step_curl_kernels.py"
RECORD = HERE / "certification.json"

#: The multi-step budget this pair was gated at, and the one the laptop
#: expression-tree comparison runs at. 60 because the sibling Triton track
#: MEASURED 8, 10 and 6 consecutive steps all passing a divergence that 40 did
#: not (triton_kernels/fingerprints.json, bit_identity_gate.legs.whole_step).
MULTI_STEP_BUDGET = 60

KERNEL_DECLARATION = re.compile(r'extern "C" __global__ void (\w+)\(')


def kernel_source() -> str:
    return KERNEL_MODULE.read_text(encoding="utf-8")


def module_level_literal(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned at module level in {KERNEL_MODULE.name}")


def device_sources(source: str) -> dict:
    """The exact CUDA strings ``_get_kernel`` hands ``cp.RawKernel``.

    Evaluated from the syntax tree rather than by importing the module, because
    importing it needs CuPy — which is the whole reason this helper exists. The
    ``_PRELUDE + r'''…'''`` concatenations are evaluated in order against the
    constants already seen, so each kernel string includes its shared prelude
    exactly as it does at compile time.
    """
    environment: dict = {}
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if not (name.endswith("_code") or name.endswith("PRELUDE")):
            continue
        expression = ast.Expression(node.value)
        ast.fix_missing_locations(expression)
        environment[name] = eval(compile(expression, "<device source>", "eval"),
                                 {}, dict(environment))
    return {name: text for name, text in environment.items() if name.endswith("_code")}


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__``.

    The predicate refuses any backend whose module is not named "cupy", which is
    the one thing about the real device library that cannot be reproduced off
    device — and the only thing this stands in for.
    """

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


_BACKENDS = [pytest.param(_NumpyWearingCupysName(), id="numpy-as-cupy")]
if cupy is not None:  # pragma: no cover - only on a CUDA host
    _BACKENDS.append(pytest.param(cupy, id="cupy"))


@pytest.fixture(params=_BACKENDS)
def xp(request):
    """The array module the grid is built on, named "cupy" either way."""
    return request.param


SIDES = ("H", "E")


def build(xp, boundaries=("periodic", "periodic", "periodic"), symmetry=(),
          force_complex_fields=False, cell=(8.0, 8.0, 8.0), **grid_kwargs):
    """A frozen (fields, pml, grid) triple with PML storage allocated.

    The same builder the curl slice uses, for the same reason: the layer skips an
    axis too thin to hold it, and skips the low face of a mirrored axis where
    cell 0 is the plane rather than a wall.
    """
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=symmetry, xp=xp, **grid_kwargs)
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


# --------------------------------------------------------------------------
# The verdict table. One entry per refusal the predicate can state, per side.
#
# Unlike the curl slice's table this one is NOT a record of a pre-extraction
# function — there is no earlier version to preserve. It is a specification: the
# reason string is what a caller reads when a configuration unexpectedly stays on
# the array path, and a refusal that fires for a different clause than the one
# intended is a refusal nobody can act on.
# --------------------------------------------------------------------------

_SHARED_GOLDEN = {
    "covered_periodic": (True, "covered"),
    "covered_m_m_p": (True, "covered"),
    "covered_m_m_m": (True, "covered"),
    "covered_m_p_m": (True, "covered"),
    # THE WIDENING. A conductivity disqualifies the CURL and leaves both
    # constitutive sub-steps alone; see test_a_conductive_run_is_covered_here.
    "conductivity": (True, "covered"),
    "complex_storage": (
        False, "complex64 storage: the recurrence is the same but the storage is not"),
    # THE 2026-08-19 WIDENING. A fold used to be refused here on the reading
    # that "the extent is what turns a cell index into a coefficient index"; the
    # device gate ran the shipped pair on 96 folded cases per policy at 112/112
    # bit-identical and the reading did not survive it. Every fold shape the gate
    # swept is a row of this table, so a narrowing shows up as a verdict change
    # and not as a quietly smaller number somewhere else.
    "mirror_plane": (True, "covered"),
    "mirror_plane_odd_phase": (True, "covered"),
    "mirror_plane_odd_count": (True, "covered"),
    "mirror_plane_x": (True, "covered"),
    "mirror_plane_z_metallic": (True, "covered"),
    "mirror_two_planes": (True, "covered"),
    # ADMITTED since 2026-08-20, when the gate was RE-RUN with three ``fold_XYZ_*``
    # specs at both terminations and two of them in the mutation plan: 136 cases
    # per policy, 136/136 identical at one launch and at 60, every mutation as
    # required (``CONSTITUTIVE_FOLD_ADMISSION["planes_round_artifacts"]``). Three
    # is every plane a 3-D grid has, so the cap clause below is now unreachable
    # from a real ``Grid`` -- it is on ``_UNREACHED_REFUSALS`` for that reason, and
    # ``test_the_fold_plane_cap_is_the_number_the_gate_swept`` lowers the record
    # instead of building a four-plane grid nothing can build.
    "mirror_three_planes": (True, "covered"),
    # ADMITTED since 2026-08-20, and the case stays in the table for the reason
    # the folded rows do: a verdict that moved has to be pinned where it moved to,
    # or the next reader cannot tell an admission from an omission. What licenses
    # it is coverage.CONSTITUTIVE_CYLINDRICAL_ADMISSION -- 96/96 cases
    # bit-identical on the GPU host, 80 of them resolving the r axis to CYL_AXIS.
    "cylindrical": (True, "covered"),
    "special_kz": (
        False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"),
    "inactive_layer": (False, "no active PML layer"),
    "numpy_backend": (False, "backend is not CuPy"),
    "bloch_k": (
        False, "nonzero Bloch k: the wrapped plane carries a phase real storage "
               "cannot hold"),
    "bfast": (False, "BFAST: a second additive term on every curl target"),
    "chi2": (False, "instantaneous chi2/chi3: a Pade factor replaces the constitutive product"),
    "chi3": (False, "instantaneous chi2/chi3: a Pade factor replaces the constitutive product"),
}

_GOLDEN = {
    "H": dict(_SHARED_GOLDEN, **{
        # The H side is blind to everything that changes update_E's SOURCE, and
        # that is the per-sub-step split doing its job rather than an oversight.
        "polarizations": (True, "covered"),
        "offdiagonal": (True, "covered"),
        "no_storage": (False, "Hx is not allocated"),
        # `stores_E` is an E-side clause, so the H side reaches the allocation
        # loop and names the first array it finds missing.
        "wrong_dtype": (False, "Hz is float64, not float32"),
        "wrong_shape": (False, "Bx has shape (2, 2, 2), not the grid's (8, 8, 8)"),
        "not_contiguous": (False, "f_w_Hy is not C-contiguous"),
        "pml_vector_missing": (False, "pml.kps_y is missing"),
        "pml_vector_dtype": (False, "pml.kms_z is float64, not float32"),
        "pml_vector_size": (False, "pml.kps_x has 7 entries, not the axis's 8"),
    }),
    "E": dict(_SHARED_GOLDEN, **{
        "polarizations": (
            False, "dispersion: update_E's source is (D - sum P), not D, and "
                   "update_P closes the step"),
        "offdiagonal": (
            False, "off-diagonal chi1inv: the row product reads the other "
                   "components' volumes and this sub-step is element-wise"),
        # The E side never reaches the allocation loop: `stores_E` is false
        # before it, and that is the clause that names the real problem.
        "no_storage": (False, "E is recomputed from D rather than stored"),
        "wrong_dtype": (False, "Ez is float64, not float32"),
        "wrong_shape": (False, "Dx has shape (2, 2, 2), not the grid's (8, 8, 8)"),
        "not_contiguous": (False, "f_w_Ey is not C-contiguous"),
        "inv_eps_dtype": (
            False, "inverse_epsilon_for('Ey') is float64, not float32"),
        "inv_eps_scalar": (
            False, "inverse_epsilon_for('Ez') is a scalar, not a volume"),
        "inv_eps_shape": (
            False, "inverse_epsilon_for('Ex') has shape (2, 2, 2), not the "
                   "grid's (8, 8, 8)"),
        "inv_eps_not_contiguous": (
            False, "inverse_epsilon_for('Ez') is not C-contiguous"),
        "pml_vector_missing": (False, "pml.kps_y_h is missing"),
        "pml_vector_dtype": (False, "pml.kms_z_h is float64, not float32"),
        "pml_vector_size": (False, "pml.kps_x_h has 7 entries, not the axis's 8"),
    }),
}


def configuration(case: str, side: str, xp):
    """Build one named configuration for one side."""
    if case == "covered_periodic":
        return build(xp)
    if case.startswith("covered_"):
        kinds = {"m": "metallic", "p": "periodic"}
        return build(xp, boundaries=tuple(kinds[c] for c in case.split("_")[1:]))
    if case == "complex_storage":
        return build(xp, force_complex_fields=True)
    if case == "mirror_plane":
        return build(xp, symmetry=("y",))
    if case == "mirror_plane_odd_phase":
        return build(xp, symmetry=(Mirror("y", -1),))
    if case == "mirror_plane_odd_count":
        # An ODD full cell count on the folded axis. MEEP's fold lands on a grid
        # point at both parities but shifts the window, and the stored extent is
        # ``n - n//2 + 1`` either way — a different arithmetic from the even case
        # even though nothing in the predicate reads it.
        return build(xp, symmetry=("y",), cell=(8.0, 9.0, 8.0))
    if case == "mirror_plane_x":
        # The folded axis whose coefficient index is the SLOWEST stride, against
        # ``mirror_plane``'s middle one and ``mirror_plane_z_metallic``'s fastest.
        return build(xp, symmetry=("x",))
    if case == "mirror_plane_z_metallic":
        # The other folded TERMINATION: a folded periodic axis stores one slot
        # past MEEP's owned window and a folded metallic one does not, so the two
        # are structurally different arrays and the gate swept both.
        return build(xp, symmetry=("z",),
                     boundaries=("periodic", "periodic", "metallic"))
    if case == "mirror_two_planes":
        # A DOUBLED cell on the folded axes, one terminated each way. At the
        # table's default 8 the fold leaves a stored extent below ``build``'s
        # six-cell floor and the layer comes back inactive, which would score
        # this row on "no active PML layer" and say nothing about the fold.
        return build(xp, symmetry=("x", "y"), cell=(16.0, 16.0, 8.0),
                     boundaries=("periodic", "metallic", "periodic"))
    if case == "mirror_three_planes":
        return build(xp, symmetry=("x", "y", "z"), cell=(16.0, 16.0, 16.0),
                     boundaries=("metallic", "metallic", "metallic"))
    if case == "cylindrical":
        grid = Grid(resolution=1.0, cell_size=(8.0, 0.0, 8.0), cylindrical=True,
                    m=0, xp=xp)
        layer = PML(grid=grid, thickness={"x": (0, 2), "z": 2})
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        return fields, layer, grid
    if case == "special_kz":
        return build(xp, cell=(8.0, 8.0, 0.0), dimensions=2, beta=0.25)
    if case == "conductivity":
        fields, layer, grid = build(xp)
        fields.set_d_conductivity(xp.full(grid.shape, 0.4, dtype=xp.float32))
        return fields, layer, grid
    if case == "inactive_layer":
        grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=xp)
        layer = PML(grid=grid, thickness=0)
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        return fields, layer, grid
    if case == "numpy_backend":
        grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=numpy)
        layer = PML(grid=grid, thickness=2)
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        return fields, layer, grid
    if case == "no_storage":
        grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=xp)
        layer = PML(grid=grid, thickness=2)
        return Fields(grid=grid), layer, grid  # no enable_*_storage(): no arrays
    if case == "bloch_k":
        return build(xp, k_point=(0.25, 0.0, 0.0))
    if case == "bfast":
        return build(xp, bfast_scaled_k=(0.1, 0.0, 0.0))
    if case == "polarizations":
        fields, layer, grid = build(xp)
        fields.polarizations.append(object())
        return fields, layer, grid
    if case == "offdiagonal":
        fields, layer, grid = build(xp)
        # The predicate reads the PROPERTY, so this stands in for an installed
        # row without needing a chi1inv tensor the fixture would have to
        # validate. What is being pinned is the refusal, not the installer.
        fields._chi1inv_offdiagonal = {"Ex": {"Ey": None}}
        return fields, layer, grid
    if case in ("chi2", "chi3"):
        fields, layer, grid = build(xp)
        setattr(fields, f"_{case}_components", {"Ex": 0.5})
        return fields, layer, grid
    if case == "wrong_dtype":
        fields, layer, grid = build(xp)
        target = "Hz" if side == "H" else "Ez"
        setattr(fields, target, getattr(fields, target).astype(xp.float64))
        return fields, layer, grid
    if case == "wrong_shape":
        fields, layer, grid = build(xp)
        source = "Bx" if side == "H" else "Dx"
        setattr(fields, source, xp.zeros((2, 2, 2), dtype=xp.float32))
        return fields, layer, grid
    if case == "not_contiguous":
        fields, layer, grid = build(xp)
        aux = "f_w_Hy" if side == "H" else "f_w_Ey"
        setattr(fields, aux, xp.asfortranarray(getattr(fields, aux)))
        return fields, layer, grid
    if case == "inv_eps_dtype":
        fields, layer, grid = build(xp)
        fields._inv_eps_components["Ey"] = \
            fields._inv_eps_components["Ey"].astype(xp.float64)
        return fields, layer, grid
    if case == "inv_eps_scalar":
        fields, layer, grid = build(xp)
        fields._inv_eps_components["Ez"] = xp.float32(0.5)
        return fields, layer, grid
    if case == "inv_eps_shape":
        fields, layer, grid = build(xp)
        fields._inv_eps_components["Ex"] = xp.zeros((2, 2, 2), dtype=xp.float32)
        return fields, layer, grid
    if case == "inv_eps_not_contiguous":
        fields, layer, grid = build(xp)
        fields._inv_eps_components["Ez"] = xp.asfortranarray(
            fields._inv_eps_components["Ez"])
        return fields, layer, grid
    if case == "pml_vector_missing":
        fields, layer, grid = build(xp)
        setattr(layer, "kps_y" + ("_h" if side == "E" else ""), None)
        return fields, layer, grid
    if case == "pml_vector_dtype":
        fields, layer, grid = build(xp)
        name = "kms_z" + ("_h" if side == "E" else "")
        setattr(layer, name, getattr(layer, name).astype(xp.float64))
        return fields, layer, grid
    if case == "pml_vector_size":
        fields, layer, grid = build(xp)
        name = "kps_x" + ("_h" if side == "E" else "")
        setattr(layer, name, getattr(layer, name).reshape(-1)[:-1])
        return fields, layer, grid
    raise AssertionError(f"no builder for {case!r}")


_CASES = [(side, case) for side in SIDES for case in sorted(_GOLDEN[side])]


@pytest.mark.parametrize("side,case", _CASES, ids=[f"{s}-{c}" for s, c in _CASES])
def test_the_predicate_states_the_verdict_the_table_specifies(side, case, xp):
    """Verdict AND reason, case by case, side by side."""
    fields, layer, grid = configuration(case, side, xp)
    assert coverage.covers_real_pml_constitutive(fields, layer, grid, side) \
        == _GOLDEN[side][case], (
            f"{side}/{case}: the predicate's verdict is not the specified one")


# Every ``return False`` in the shipped predicate that NO configuration in the
# tables reaches, identified by a distinctive fragment of its own source line.
# Each one is here because it is genuinely unreachable from a real
# ``Grid``/``Fields``/``PML`` triple, not because nobody got round to it:
#
#     (the has_symmetry/is_mirrored DISAGREEMENT clause is NOT on this list: it
#     is unreachable from a real ``Grid``, where both facts are built from the
#     same list — grid.py:1091-1095, :1445-1446 — but it is REACHED through
#     ``grid_contradicts_itself_about_the_fold``, which is the discipline the
#     fail-closed clauses added on 2026-08-15 already follow; the cylindrical
#     r = 0 clause that replaced the whole-grid refusal on 2026-08-20 follows the
#     same discipline through
#     ``grid_reports_the_axis_rule_without_cylindrical``, which is why it is not
#     on this list either);
#   * a boundary kind outside ``CONSTITUTIVE_BOUNDARY_KINDS`` is now unreachable
#     from a real triple altogether: ``mirror`` joined the list on 2026-08-19 and
#     ``axis`` on 2026-08-20, so the four kinds ``_boundary_kinds_from`` can
#     return are all admitted. The clause stays as a guard against a kind added
#     to ``Grid`` later, which is exactly the case it cannot be exercised for
#     today;
#   * ``Fields`` always exposes ``inverse_epsilon_for``, it never returns None for
#     Ex/Ey/Ez, and its ValueError path (fields.py:1341-1344) is unreachable for
#     the three components the predicate asks for — the guard around it is there
#     because a raise escaping a fail-closed predicate is a crashed run where "no"
#     was the correct answer, not because a fixture can trigger it.
#
# A clause added to the predicate and left unexercised fails the test below with
# its own text, which is the point: this is a list of admissions, not a tolerance.
_UNREACHED_REFUSALS = (
    'which has no kernel"',
    "fields does not expose inverse_epsilon_for",
    'is None"',
    "raised {exc!r}",
    # THE PLANE CAP, since 2026-08-20. ``folded`` counts mirrored axes out of
    # ``range(3)`` and the cap is 3, so ``folded > cap`` cannot be true on any
    # grid this engine can build. The clause is kept rather than deleted because
    # it is what makes the predicate read the record's number rather than restate
    # it, and ``test_the_fold_plane_cap_is_the_number_the_gate_swept`` measures
    # that by lowering the record and requiring the refusal back.
    "mirror planes at once",
)


# Configurations that are NOT buildable from ``Grid``/``Fields``/``PML`` and are
# built by hand instead. Each one is a fail-closed clause added on 2026-08-15
# after four of them were measured ESCAPING both predicates as exceptions — a
# crashed run where "no" was the correct answer, on a module that argues for
# exactly that property by name.
#
# They are in the trace set as well as in their own tests, so the new clauses are
# REACHED rather than admitted: a fail-closed guard nothing exercises is the same
# shape of unmeasured code as the thing it guards against.
def _synthetic(case: str, side: str, xp):
    """One hand-built triple whose objects cannot answer some question."""
    fields, layer, grid = build(xp)

    class _Proxy:
        """A stand-in that delegates everything except what the case breaks."""

        def __init__(self, wrapped):
            object.__setattr__(self, "_wrapped", wrapped)

        def __getattr__(self, item):
            return getattr(object.__getattribute__(self, "_wrapped"), item)

    if case == "grid_cannot_answer":
        class _Broken(_Proxy):
            def is_axis(self, axis):
                raise RuntimeError("this grid does not know its axes")
        return fields, layer, _Broken(grid)
    if case == "grid_reports_the_axis_rule_without_cylindrical":
        class _Broken(_Proxy):
            def is_axis(self, axis):
                return axis == 0
            @property
            def cylindrical(self):
                return False
        return fields, layer, _Broken(grid)
    if case == "grid_missing_a_method":
        class _Broken(_Proxy):
            def __getattr__(self, item):
                if item == "is_metallic":
                    raise AttributeError("is_metallic")
                return getattr(object.__getattribute__(self, "_wrapped"), item)
        return fields, layer, _Broken(grid)
    if case == "array_cannot_be_inspected":
        target = "Hx" if side == "H" else "Ex"

        class _Opaque:
            dtype = property(lambda self: (_ for _ in ()).throw(
                RuntimeError("this array will not say what it holds")))
        setattr(fields, target, _Opaque())
        return fields, layer, grid
    if case == "pml_vector_not_contiguous":
        # A (2n,1,1) view strided by 2: `.size` is right, and `reshape(-1)` of it
        # is NON-contiguous because the trailing size-1 dims let NumPy re-stride
        # rather than copy — so the launcher's real_constitutive_tables raises
        # ValueError on an input the predicate used to admit.
        name = "kps_x" + ("_h" if side == "E" else "")
        values = getattr(layer, name)
        doubled = xp.zeros((values.size * 2, 1, 1), dtype=xp.float32)
        doubled[::2] = values.reshape(-1, 1, 1)
        setattr(layer, name, doubled[::2])
        return fields, layer, grid
    if case == "pml_vector_wrong_axis":
        # A cubic grid makes every axis's extent equal, so a size comparison
        # cannot tell one axis's profile from another's. The broadcast SHAPE can.
        name = "kms_y" + ("_h" if side == "E" else "")
        setattr(layer, name, getattr(layer, name).reshape(-1, 1, 1))
        return fields, layer, grid
    if case == "grid_contradicts_itself_about_the_fold":
        # UNREACHABLE from a real Grid — ``has_symmetry()`` is
        # ``bool(self.symmetry)`` and ``is_mirrored(axis)`` reads
        # ``mirrors_by_axis[axis]``, both built from the same list. Reached here
        # because the fold admission is decided PER AXIS since 2026-08-19, so a
        # grid that says it is folded and names no folded axis would be admitted
        # by a per-axis reading and refused by a whole-grid one, and a predicate
        # that has to pick must pick "no".
        class _Contradictory(_Proxy):
            def has_symmetry(self):
                return True
        return fields, layer, _Contradictory(grid)
    if case == "grid_is_four_dimensional":
        # Unreachable from a real Grid (grid.py assigns a 3-tuple) and admitted by
        # covers_real_pml_curl until 2026-08-15 — the kernels take (nx, ny, nz),
        # so a 4-D admission is a wrong-index answer, not a launch failure.
        class _FourD(_Proxy):
            shape = (2, 3, 4, 5)
        return fields, layer, _FourD(grid)
    if case == "cells_exceed_int32":
        # 2048 * 2048 * 512 is 2**31 EXACTLY, the first count the kernels' ``int``
        # index cannot address. Reached through a grid stand-in rather than an
        # allocation for the obvious reason — the real thing is 8 GiB per volume
        # and eighteen volumes deep — which is why both predicates' int32 clause
        # was ADMITTED as unreachable until 2026-08-15. An admitted clause is an
        # unexercised clause: the shape read is what the predicate acts on, so a
        # grid that merely REPORTS the count exercises it exactly.
        class _Huge(_Proxy):
            shape = (2048, 2048, 512)
        return fields, layer, _Huge(grid)
    raise AssertionError(f"no synthetic builder for {case!r}")


_SYNTHETIC_GOLDEN = {
    "grid_cannot_answer": (
        False, "the grid could not answer a question this predicate has to ask: "
               "RuntimeError: this grid does not know its axes"),
    "grid_missing_a_method": (
        False, "the grid could not answer a question this predicate has to ask: "
               "AttributeError: is_metallic"),
}
_SYNTHETIC_REASON_PREFIX = {
    "grid_contradicts_itself_about_the_fold":
        "the grid reports has_symmetry() and no mirrored axis, or the reverse",
    # The clause that replaced the cylindrical refusal on 2026-08-20. Dcyl is now
    # ADMITTED, so the only thing left to refuse is a grid that reports the
    # r = 0 axis rule WITHOUT reporting cylindrical coordinates -- which a real
    # Grid never does, which is exactly why it needs a synthetic case: a
    # fail-closed guard nothing exercises is the same shape of unmeasured code as
    # the hole it was written to close.
    "grid_reports_the_axis_rule_without_cylindrical":
        "reports the cylindrical r = 0 rule on a grid that does not report",
    "array_cannot_be_inspected": "could not be inspected: RuntimeError",
    "pml_vector_not_contiguous": "is not C-contiguous; the kernel indexes",
    "pml_vector_wrong_axis": "not axis 1's broadcast shape",
    "grid_is_four_dimensional": "is not three-dimensional",
    "cells_exceed_int32": "2147483648 cells exceeds the kernel's int32 index range",
}
_SYNTHETIC_CASES = tuple(_SYNTHETIC_GOLDEN) + tuple(_SYNTHETIC_REASON_PREFIX)


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("case", _SYNTHETIC_CASES)
def test_the_predicate_refuses_rather_than_raising_on_an_object_that_cannot_answer(
        case, side, xp):
    """A RAISE IS NOT A REFUSAL, applied everywhere instead of at one call site.

    ``coverage.py`` states the rule in as many words and wrapped exactly one call
    (``inverse_epsilon_for``). Constructed and measured: a grid whose ``is_axis``
    raises, a grid with no ``is_metallic``, and an array whose ``dtype`` raises
    each escaped BOTH predicates as an exception. Latent — a real
    ``Grid``/``Fields`` always answers, which is why the 186-row battery reports
    "predicate RAISED on: 0" — but latency is a statement about today's callers.
    """
    fields, layer, grid = _synthetic(case, side, xp)
    covered, reason = coverage.covers_real_pml_constitutive(fields, layer, grid, side)
    assert covered is False
    if case in _SYNTHETIC_GOLDEN:
        assert (covered, reason) == _SYNTHETIC_GOLDEN[case]
    else:
        assert _SYNTHETIC_REASON_PREFIX[case] in reason, reason


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
@pytest.mark.parametrize("case", ("grid_cannot_answer", "grid_missing_a_method",
                                  "array_cannot_be_inspected",
                                  "grid_is_four_dimensional",
                                  "cells_exceed_int32"))
def test_the_curl_predicate_refuses_the_same_way(case, sub_step, xp):
    """Both predicates, or the asymmetry is the next silent wrong answer.

    ``grid_is_four_dimensional`` is the asymmetry that was there: the constitutive
    predicate carried a dimensionality clause from the start and the curl did not,
    so a 4-D grid came back ``(True, 'covered')`` from one and refused by the
    other. Measured, then closed. ``cells_exceed_int32`` was the same asymmetry one
    clause along — the constitutive predicate refused past 2**31 cells and the curl
    admitted, though ``step_B_pml_real`` indexes with ``int`` too — and it
    was closed on 2026-08-15 by the same census.
    """
    fields, layer, grid = _synthetic(case, "H", xp)
    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
    assert covered is False, reason


@pytest.mark.parametrize("sub_step", ("step_B", "step_D"))
def test_a_fields_that_cannot_be_asked_for_a_conductivity_is_refused(sub_step, xp):
    """The curl's own read, which is not on any grid: ``fields.condfac_for``."""
    fields, layer, grid = build(xp)

    class _Mute:
        def __getattr__(self, item):
            if item == "condfac_for":
                raise AttributeError("condfac_for")
            return getattr(fields, item)

    covered, reason = coverage.covers_real_pml_curl(_Mute(), layer, grid, sub_step)
    assert covered is False
    assert "could not be asked for a conductivity" in reason, reason


def _refusal_lines() -> dict:
    """Absolute line number -> source text, for every ``return False`` site."""
    lines, start = inspect.getsourcelines(coverage.covers_real_pml_constitutive)
    return {start + offset: text.strip()
            for offset, text in enumerate(lines)
            if text.strip().startswith("return False")}


def _lines_executed(function, *args) -> set:
    """The line numbers ``function`` actually ran for these arguments."""
    import sys

    code = function.__code__
    seen: set = set()

    def tracer(frame, event, arg):
        if frame.f_code is not code:
            return None
        if event == "line":
            seen.add(frame.f_lineno)
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        function(*args)
    finally:
        sys.settrace(previous)
    return seen


def test_the_tables_reach_every_refusal_the_predicate_can_state(xp):
    """A verdict table that has drifted away from the predicate pins nothing.

    Not a count of strings — a TRACE. Every case in both tables is run under a
    line tracer, the ``return False`` sites it reaches are collected, and what is
    left over must be exactly ``_UNREACHED_REFUSALS``. That makes the two
    failure directions distinguishable: a clause added without a case fails
    naming its own source line, and a case that stops reaching the clause it was
    written for fails naming that one.
    """
    refusals = _refusal_lines()
    # 25 UNTIL 2026-08-20, when the cylindrical clause was removed on a device
    # verdict (CONSTITUTIVE_CYLINDRICAL_ADMISSION). The floor tracks the clause
    # count so that a scan which silently stops finding sites fails; it is not a
    # claim that the predicate may never lose a clause, only that a clause may
    # not disappear from the SCAN while remaining in the source.
    assert len(refusals) >= 24, (
        f"only {len(refusals)} refusal sites found; the source scan has drifted")

    reached: set = set()
    for side, case in _CASES:
        fields, layer, grid = configuration(case, side, xp)
        reached |= _lines_executed(
            coverage.covers_real_pml_constitutive, fields, layer, grid, side)
    # The hand-built triples too, so the fail-closed clauses added on 2026-08-15
    # are REACHED rather than admitted. A guard nothing exercises is the same
    # shape of unmeasured code as the hole it was written to close.
    for side in SIDES:
        for case in _SYNTHETIC_CASES:
            fields, layer, grid = _synthetic(case, side, xp)
            reached |= _lines_executed(
                coverage.covers_real_pml_constitutive, fields, layer, grid, side)

    unreached = sorted(text for line, text in refusals.items()
                       if line not in reached)
    expected = sorted(text for text in unreached
                      if any(fragment in text for fragment in _UNREACHED_REFUSALS))
    assert unreached == expected, {
        "unreached and not admitted": sorted(set(unreached) - set(expected)),
    }
    assert len(unreached) == len(_UNREACHED_REFUSALS), {
        "admitted but now reachable": sorted(
            fragment for fragment in _UNREACHED_REFUSALS
            if not any(fragment in text for text in unreached)),
        "unreached": unreached,
    }


def test_an_unknown_side_raises_rather_than_refusing(xp):
    """A typo'd side must not read as "not covered" — that is a silent skip.

    ``covers_real_pml_constitutive(fields, pml, grid, "B")`` returning
    ``(False, ...)`` would look exactly like a legitimate refusal in a plan log,
    and the sub-step would quietly stay on the array path forever.
    """
    fields, layer, grid = build(xp)
    with pytest.raises(ValueError, match="side must be one of"):
        coverage.covers_real_pml_constitutive(fields, layer, grid, "B")
    with pytest.raises(ValueError, match="side must be one of"):
        coverage.constitutive_sub_lattice("B")


# --------------------------------------------------------------------------
# The one clause where this predicate is WIDER than the curl's.
# --------------------------------------------------------------------------

def test_a_conductive_run_is_covered_here_and_refused_by_one_curl_only(xp):
    """The per-sub-step split, measured across all four sub-steps at once.

    A D conductivity routes ``step_D`` to the three-history conductive-PML
    recurrence and leaves ``step_B`` and both constitutive sub-steps as ordinary
    products — three of the four admitted, one refused, on the configuration the
    corpus row ``TestAdjointSolver.test_damping`` carries. If this ever inverts —
    the constitutive predicate refusing, or ``step_D`` admitting, or ``step_B``
    going back to refusing — one of the pair is wrong and this says which.
    """
    fields, layer, grid = configuration("conductivity", "E", xp)
    assert fields.has_conductivity

    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, "step_D")
    assert not covered
    assert "conductivity" in reason

    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, "step_B")
    assert covered, (
        f"step_B was refused on a D-only conductivity ({reason!r}); the sigma "
        f"reaches step_D's targets and nothing step_B writes")

    for side in SIDES:
        covered, reason = coverage.covers_real_pml_constitutive(
            fields, layer, grid, side)
        assert covered, f"update_{side} was refused on a conductive run: {reason}"


def test_the_constitutive_path_in_stepping_never_reads_a_conductivity():
    """The widening is not a belief about ``stepping.py``; it is its text.

    ``condfac_for`` is read inside ``_apply_curl`` and nowhere else in the
    module's constitutive chain. Read off the shipped source rather than
    restated, so the day someone threads a conductivity through
    ``_apply_constitutive_pml`` this fails instead of the engine going quietly
    wrong on every conductive PML run.
    """
    for function in (stepping.update_H, stepping.update_E,
                     stepping._apply_constitutive_pml):
        body = inspect.getsource(function)
        assert "condfac" not in body and "f_cond" not in body, (
            f"{function.__name__} now reads a conductivity; "
            f"covers_real_pml_constitutive admits conductive runs on the "
            f"strength of it not doing so")


# --------------------------------------------------------------------------
# The tables, pinned against stepping's own.
# --------------------------------------------------------------------------

def test_the_side_table_matches_steppings_constitutive_terms():
    """``CONSTITUTIVE_SIDES`` is a transcription and this is what stops it drifting.

    ``coverage.py`` imports nothing, so the tables are written out by hand
    there; here — in a test, where an import costs nothing — they are checked
    against ``stepping.H_CONSTITUTIVE_TERMS`` / ``E_CONSTITUTIVE_TERMS``
    (stepping.py:227-228), which is where the engine reads them from.
    """
    engine = {"H": stepping.H_CONSTITUTIVE_TERMS,
              "E": stepping.E_CONSTITUTIVE_TERMS}
    for side, terms in engine.items():
        spec = coverage.CONSTITUTIVE_SIDES[side]
        assert spec["targets"] == tuple(t[0] for t in terms)
        assert spec["sources"] == tuple(t[1] for t in terms)
        assert spec["aux"] == tuple("f_w_" + t[0] for t in terms)
        # The component's OWN axis, in order: x for component 0, y for 1, z for 2.
        assert tuple(t[2] for t in terms) == ("x", "y", "z")


def test_the_sub_lattice_pairing_is_the_one_stepping_passes():
    """Integer for H, half-integer for E — read off ``update_H``/``update_E``.

    Swapped, it is a half-cell error in the absorber profile: converged, smooth
    and wrong. The pairing is decided in exactly one place
    (``constitutive_sub_lattice``) and checked here against the two call sites
    that make it real.
    """
    assert coverage.constitutive_sub_lattice("H") is False
    assert coverage.constitutive_sub_lattice("E") is True
    assert "half_integer=False" in inspect.getsource(stepping.update_H)
    assert "half_integer=True" in inspect.getsource(stepping.update_E)


def test_the_launcher_and_the_predicate_ask_the_same_function_for_the_pairing():
    """One decision, one function — read structurally, because CuPy is not here.

    ``real_constitutive_tables`` takes ``half_integer`` as an argument, so every
    call site is one more place to get the H/E pairing backwards.
    ``constitutive_tables_for`` removes the argument by asking
    ``constitutive_sub_lattice``, which is what the predicate asks.

    AND THE LIVE ENTRY POINT HAS TO REACH IT, which is the half this test used to
    miss. It inspected ``constitutive_tables_for`` — a function that, until
    2026-08-15, had no callers anywhere in the repository. The only path that ever
    launches these kernels is ``update_fused_pml_real``, and it took ``tables`` as
    a positional parameter and passed it through unchecked, so the invariant the
    module's docstring claims ("the two cannot disagree") held for a function
    nothing called and not for the one that runs.
    """
    tree = ast.parse(kernel_source())

    def function(name):
        return next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    def calls_of(node):
        return {inner.func.id for inner in ast.walk(node)
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)}

    assert {"real_constitutive_tables", "constitutive_sub_lattice"} <= \
        calls_of(function("constitutive_tables_for"))

    entry = function("update_fused_pml_real")
    assert "constitutive_tables_for" in calls_of(entry), (
        "the live entry point does not derive the coefficient tables from the "
        "side; the pairing is then a convention its callers may keep, and a "
        "half-cell error in the absorber profile is one argument away")

    # ``tables`` must be KEYWORD-ONLY on that entry point. It exists for the byte
    # gate, which supplies synthetic tables and deliberately mis-paired ones; a
    # positional slot is one a planner fills by accident.
    assert not any(argument.arg == "tables" for argument in entry.args.args), (
        "tables is positional on update_fused_pml_real")
    assert any(argument.arg == "tables" for argument in entry.args.kwonlyargs)

    # And no literal True/False sub-lattice is hard-coded anywhere in the module's
    # own call sites — the argument exists for the gate, which sweeps both.
    hard_coded = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)
                  and node.func.id == "real_constitutive_tables"
                  and any(isinstance(a, ast.Constant) and isinstance(a.value, bool)
                          for a in node.args)]
    assert not hard_coded, (
        "a call site hard-codes the sub-lattice; that decision belongs to "
        "constitutive_sub_lattice so the predicate and the launcher move together")


def test_every_module_function_the_slice_ships_is_reachable_from_the_entry_point():
    """A helper with no callers is validation nobody runs.

    ``real_constitutive_tables``' dtype and contiguity guards were dead code for
    exactly as long as ``constitutive_tables_for`` had no callers — they refuse
    inputs that could never arrive. Walking the call graph from the two public
    entry points is what turns "it is wired" into something that fails when it
    stops being.
    """
    tree = ast.parse(kernel_source())
    defined = {node.name: node for node in tree.body
               if isinstance(node, ast.FunctionDef)
               and not node.name.startswith("_")}
    reachable, frontier = set(), ["update_fused_pml_real"]
    while frontier:
        name = frontier.pop()
        if name in reachable:
            continue
        reachable.add(name)
        node = defined.get(name)
        if node is None:
            continue
        frontier.extend(inner.func.id for inner in ast.walk(node)
                        if isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Name))
    orphans = sorted(set(defined) - reachable)
    assert orphans == [], (
        f"{orphans} are shipped, public and unreachable from update_fused_pml_real; "
        f"whatever they validate, nothing validates")


def test_the_predicate_checks_only_its_own_sub_lattice(xp):
    """A side must not be held hostage to the other side's tables.

    Deleting the half-integer vectors leaves ``update_H`` covered and refuses
    ``update_E``, and vice versa. The curl predicate checks BOTH sub-lattices
    because the curl kernel plans either sub-step; a constitutive side reads one.
    """
    fields, layer, grid = build(xp)
    layer.kps_y_h = None
    assert coverage.covers_real_pml_constitutive(fields, layer, grid, "H")[0]
    covered, reason = coverage.covers_real_pml_constitutive(fields, layer, grid, "E")
    assert not covered and reason == "pml.kps_y_h is missing"

    fields, layer, grid = build(xp)
    layer.kms_x = None
    assert coverage.covers_real_pml_constitutive(fields, layer, grid, "E")[0]
    covered, reason = coverage.covers_real_pml_constitutive(fields, layer, grid, "H")
    assert not covered and reason == "pml.kms_x is missing"


STDLIB_ONLY = {"__future__", "typing"}


def device_free_import_report(source: str) -> dict:
    """Every way this module could acquire a dependency, as one report.

    THE GUARD THIS REPLACES MISSED FIVE OF SEVEN ARMED MUTATIONS. It filtered
    ``isinstance(node, ast.ImportFrom) and node.level == 0``, so every RELATIVE
    import was invisible: ``from .step_curl_kernels import _get_kernel`` (that
    module does ``import cupy as cp`` at module scope), ``from ..stepping import
    _boundary_kinds``, ``from . import step_curl_kernels``, the same relative
    import placed INSIDE the predicate, and ``cp = __import__('cupy')`` all read
    clean; only the two absolute forms were caught. The pattern is not
    hypothetical — the sibling ``triton_kernels/coverage.py`` uses exactly it
    (``..stepping``, ``.dispersive_update_e``), so it is live in this tree.

    Returns the three channels separately so a failure says which one opened.
    """
    absolute, relative, dynamic = set(), [], set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            absolute.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative.append("." * node.level + (node.module or ""))
            else:
                absolute.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("__import__", "eval", "exec"):
            dynamic.add(node.func.id)
        elif isinstance(node, ast.Attribute) and node.attr in (
                "import_module", "spec_from_file_location"):
            dynamic.add(node.attr)
    return {"absolute": absolute, "relative": relative, "dynamic": dynamic}


def test_the_predicate_module_still_pulls_in_no_device_dependency():
    """The round's own precondition, re-checked after this slice landed.

    ``coverage.py`` grew a second predicate. One ``import cupy`` in it and every
    test in this file and in the curl slice's collapses back to a sanctioned
    skip on any laptop — which is the state the whole hand-CUDA track was in
    before the predicate moved out of the kernel module. A RELATIVE import is the
    same failure by another spelling: ``step_curl_kernels`` imports CuPy at module
    scope, so ``from . import step_curl_kernels`` here would be identical in
    effect and invisible to the guard this replaces.
    """
    report = device_free_import_report((HERE / "coverage.py").read_text(encoding="utf-8"))
    assert report["absolute"] <= STDLIB_ONLY, (
        f"coverage.py imports {sorted(report['absolute'] - STDLIB_ONLY)}")
    assert report["relative"] == [], (
        f"coverage.py imports from the package: {report['relative']}; every "
        f"sibling module in it reaches CuPy, so this is 'import cupy' spelled "
        f"differently")
    assert report["dynamic"] == set(), (
        f"coverage.py can acquire a module at run time via "
        f"{sorted(report['dynamic'])}, which no static import check can see")


@pytest.mark.parametrize("mutation,channel", [
    ("import cupy\n", "absolute"),
    ("from cupy import float32\n", "absolute"),
    ("from .step_curl_kernels import _get_kernel\n", "relative"),
    ("from ..stepping import _boundary_kinds\n", "relative"),
    ("from . import step_curl_kernels\n", "relative"),
    ("cp = __import__('cupy')\n", "dynamic"),
    ("import importlib\ncp = importlib.import_module('cupy')\n", "absolute"),
])
def test_the_import_guard_catches_every_way_the_dependency_could_come_back(
        mutation, channel):
    """The guard's own mutation battery — seven armed edits, and it must catch all.

    Measured against the version this replaces: 2 caught, 5 MISSED. A regression
    guard that misses the exact regression it exists to prevent is the same shape
    of defect as the predicate hole it protects.
    """
    source = (HERE / "coverage.py").read_text(encoding="utf-8")
    clean = device_free_import_report(source)
    assert clean["absolute"] <= STDLIB_ONLY and not clean["relative"] \
        and not clean["dynamic"], "the unmutated control is already dirty"

    report = device_free_import_report(source + "\n" + mutation)
    caught = (not report["absolute"] <= STDLIB_ONLY
              or report["relative"] or report["dynamic"])
    assert caught, f"{mutation!r} slipped through the {channel} channel"


def test_the_guard_reads_an_import_hidden_inside_the_predicate_itself():
    """A function-scope relative import is the same acquisition, one indent in.

    ``ast.walk`` descends into function bodies, so this is a property of walking
    rather than of scanning ``tree.body`` — which is worth pinning, because a
    guard rewritten to look only at module level would pass every test above.
    """
    source = (HERE / "coverage.py").read_text(encoding="utf-8")
    hidden = source.replace(
        '    if side not in CONSTITUTIVE_SIDES:',
        '    from . import step_curl_kernels  # noqa\n'
        '    if side not in CONSTITUTIVE_SIDES:', 1)
    assert hidden != source, "the anchor for this mutation has moved"
    assert device_free_import_report(hidden)["relative"] == [".", ]


# --------------------------------------------------------------------------
# The device text: the properties that decide bit-identity, read on a laptop.
# --------------------------------------------------------------------------

def test_both_kernels_are_shipped_and_partitioned():
    """A kernel added without a gate verdict must fail here, not ship unmeasured.

    Same partition discipline as ``step_curl_kernels.py``'s. Both kernels were
    certified 2026-08-15, so ``UNCERTIFIED_KERNELS`` is now empty and the
    partition is carried entirely by ``CERTIFIED_KERNELS`` — which means a THIRD
    kernel added to this file lands in neither set and fails here.

    ``certified == set()`` is deliberately NOT asserted any more; what replaces
    it is stricter, in :func:`test_the_record_backs_every_certified_kernel`: a
    name may be certified only if the record names it too.
    """
    source = kernel_source()
    shipped = set(KERNEL_DECLARATION.findall(source))
    certified = set(module_level_literal(source, "CERTIFIED_KERNELS"))
    dead = module_level_literal(source, "UNCERTIFIED_KERNELS")

    assert shipped == {"update_H_pml_real", "update_E_pml_real"}
    assert not certified & set(dead)
    assert certified | set(dead) == shipped, {
        "in neither set": sorted(shipped - certified - set(dead)),
        "named but not shipped": sorted((certified | set(dead)) - shipped),
    }
    for name, owes in dead.items():
        assert "probe_fused_kernel_bit_identity" in owes, (
            f"{name} does not name the gate it owes: {owes!r}")


def test_the_record_backs_every_certified_kernel():
    """A name in ``CERTIFIED_KERNELS`` needs a device run behind it, in the record.

    This is what stops "certified" from being a word someone typed. The record's
    ``constitutive_2026-08-15`` block must name exactly the certified set, and it
    must carry the verdicts that make the claim mean something — both policies,
    distinct binaries, the multi-step budget, and a non-vacuous subnormal band.

    Everything asserted here is DERIVED from the run's artifacts by
    ``summarize_constitutive_recut.py``; none of it is transcribed by hand.
    """
    source = kernel_source()
    certified = set(module_level_literal(source, "CERTIFIED_KERNELS"))
    entry = json.loads(RECORD.read_text(encoding="utf-8"))
    block = entry.get("constitutive_2026-08-15")
    assert block is not None, (
        "certification.json has no constitutive_2026-08-15 block, so nothing in "
        "the record backs CERTIFIED_KERNELS")
    assert set(block["certified_kernels"]) == certified, (
        "certification.json and constitutive_kernels.py disagree about which "
        "kernels are certified")
    assert block["shipped_kernel_count"] == len(
        set(KERNEL_DECLARATION.findall(source)))

    # The claims that make it a certification rather than a note.
    assert sorted(block["policies_cut_under"]) == ["ieee_keep_ftz_stripped",
                                                   "meep_x86_flush"]
    assert block["constitutive_multi_step_budget"] == MULTI_STEP_BUDGET
    binaries = block["distinct_binaries_per_policy"]
    assert binaries["two_policies_produced_distinct_binaries"] is True
    for name in certified:
        assert binaries["per_kernel"][name]["distinct"] is True, name
    for policy in ("keep", "flush"):
        per_policy = block["per_policy"][policy]
        assert per_policy["all_legs_as_required"] is True, policy
        assert per_policy["multi_step_all_identical"] is True, policy
        assert per_policy["subnormal_band_is_non_vacuous"] is True, policy
        identical, ran = per_policy["single_launch_normal_numbers"].split("/")
        assert identical == ran and int(ran) > 0, per_policy
        armed = per_policy["armed_mutation_accounting"]
        assert armed["legs_reporting_a_verdict_for_a_mutation_never_compiled"] == []

        # THE GUARD MUST BE SHOWN TO BE LOAD-BEARING, and this is where the
        # evidence lives: the SYNTHETIC gate sweeps both option sets, and the
        # unguarded one must fail essentially everywhere. Without this the
        # 120/120 is a number with no scale.
        guards = per_policy["per_guard"]
        assert guards["fmad_false"]["identical"] == guards["fmad_false"]["ran"], guards
        assert guards["default_no_options"]["identical"] == 0, (
            f"the unguarded control agreed on "
            f"{guards['default_no_options']['identical']} cases under {policy}; "
            f"--fmad=false is then decorative and the guarded pass says nothing")

    # THE CORPUS GUARD CONTROL IS NOT ASSERTED TO DIVERGE, and that is a measured
    # correction rather than a missing check. It was written expecting divergence,
    # by analogy with the curl; it agreed on all twelve cases at both Courants.
    # The reason is structural: pml.py:686 builds the interior as the exact
    # identity kps == kms == 1.0, so fma(1.0f, src, f) == f + 1.0f*src bit for
    # bit, and in the 2 warm-up + 8 compared steps the pulse never reaches the
    # cells where a coefficient could bite. Requiring divergence there would fail
    # a healthy kernel for a property of the fixture, so the record states what
    # the corpus leg IS evidence for and points at the synthetic control above.
    control = block["guard_control_at_corpus_scale"]
    assert control["leg_ran_validly"] is True
    assert "_where_the_guard_evidence_actually_IS" in control
    assert control["diverged_at_inexact_dtdx"] is not None


def test_the_record_pins_the_device_source_that_was_gated():
    """The digests in the record must be of the strings this file compiles TODAY.

    A device string edited after the gate ran leaves a record describing a kernel
    that no longer exists. The em-dash repair of 2026-08-15 is exactly such an
    edit — it changed the E string and therefore its digest — and it landed
    BEFORE the gate, which is why the two agree. A later one would not.
    """
    entry = json.loads(RECORD.read_text(encoding="utf-8"))
    pinned = entry["constitutive_2026-08-15"]["device_source_sha256"]
    sources = device_sources(kernel_source())
    for name, text in sources.items():
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert pinned[name] == digest, (
            f"{name} has changed since the gate ran: the record pins "
            f"{pinned[name]} and the file now compiles {digest}")
    concatenated = "".join(sources[name] for name in sorted(sources))
    assert pinned["both_concatenated"] == hashlib.sha256(
        concatenated.encode("utf-8")).hexdigest()


def test_the_grouping_is_two_separate_accumulations():
    """THE clause the whole slice's bit-identity rests on.

    ``((f + kps*src) - kms*prev)``, which is ``stepping._apply_constitutive_pml``
    (:2086-2087) and its scratch branch (:2093-2096) alike. The flattened
    ``f + (kps*src - kms*prev)`` is a different float32 number and is what the
    two complex constitutive kernels in the sibling module write.
    """
    prelude = device_sources(kernel_source())["_update_H_pml_real_kernel_code"]
    assert "float a = f[idx] + kps * src;" in prelude
    assert "f[idx] = a - kms * prev;" in prelude
    # The flattened form, in either spelling, must not appear anywhere.
    for flattened in ("kps * src - kms * prev", "kps*src - kms*prev",
                      "+= kps", "+= kms"):
        assert flattened not in prelude, (
            f"the flattened grouping {flattened!r} is in the device source; the "
            f"array path accumulates twice, left to right")


def test_prev_is_loaded_before_the_auxiliary_is_stored():
    """The aliasing trap the array path's explicit ``fw.copy()`` exists to prevent.

    Store first and ``prev`` becomes the value just written, which is wrong only
    where ``kms != 0`` — i.e. inside the absorber — and reads as a slightly
    weaker PML rather than as a bug.
    """
    body = device_sources(kernel_source())["_update_E_pml_real_kernel_code"]
    load = body.index("float prev = fw[idx];")
    store = body.index("fw[idx] = src;")
    assert load < store, "the auxiliary is stored before its previous value is read"


def test_the_constitutive_kernels_carry_no_stencil_and_no_wall_mask():
    """Note 4: the sub-step reads no neighbour and masks no cell.

    ``_apply_constitutive_pml`` writes every cell of the volume. Anyone porting
    the certified curl kernel's shape into this one will reach for a stencil and a
    mask; this is what stops it.
    """
    for name, body in device_sources(kernel_source()).items():
        for borrowed in ("shift_up", "shift_dn", "BC_METALLIC", "BC_PERIODIC",
                         "curl = 0.0f", "dtdx"):
            assert borrowed not in body, (
                f"{name} carries {borrowed!r}, which belongs to the curl sub-step")


def test_no_constitutive_function_in_stepping_masks_an_unowned_cell():
    """The other half of note 4, READ OFF ``stepping.py`` instead of remembered.

    The note used to assert that ``_mask_non_owned_cells`` is "called from
    ``_apply_curl`` and from nowhere else". Both halves were wrong: there are
    THREE call sites and ``_apply_curl`` is not one of them — ``step_B``,
    ``step_D`` and ``_bfast_term``, the last of which sits directly above
    ``update_H`` and is the one a reader skimming for the constitutive chain would
    walk into.

    What the kernel actually needs is the weaker and checkable claim: no function
    on the constitutive path masks anything. That is what this measures, by
    reading each function's source, so the day a mask is threaded through
    ``_apply_constitutive_pml`` this fails instead of the kernel silently writing
    cells the array path leaves alone.
    """
    for function in (stepping.update_H, stepping.update_E,
                     stepping._apply_constitutive_pml):
        body = inspect.getsource(function)
        assert "_mask_non_owned_cells" not in body, (
            f"{function.__name__} now masks unowned cells; the constitutive "
            f"kernels write every cell of the volume and would stop agreeing")

    # And the call sites are what the note says they are — a count, so that a
    # fourth one appearing is a failure rather than a sentence quietly going stale.
    module = (HERE.parents[1] / "meep_gpu" / "stepping.py").read_text(encoding="utf-8")
    tree = ast.parse(module)
    owners = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) \
                    and inner.func.id == "_mask_non_owned_cells":
                owners.append(node.name)
    assert sorted(set(owners)) == ["_bfast_term", "step_B", "step_D"], sorted(set(owners))


def test_each_component_reads_its_own_axis_coefficient_table():
    """NOTE 2, the highest-consequence confusion in the file, pinned at last.

    MEEP's ``dsigw`` is the absorption a component accumulates along the direction
    IT POINTS IN (``H_CONSTITUTIVE_TERMS`` / ``E_CONSTITUTIVE_TERMS``,
    stepping.py:227-228): component 0 reads ``kps_x``/``kms_x``, 1 the ``_y`` pair,
    2 the ``_z`` pair. Confuse it with the dsig/dsigu cycle the CURL recurrence
    uses and the result is a smooth, converged, entirely wrong absorber — the
    kernel's own note says so, and until 2026-08-15 nothing anywhere checked it.

    Every earlier device-text assertion in this file stops one argument short of
    the coefficient pair: ``test_the_h_source_is_b_with_no_permeability_volume``
    asserts ``constitutive_apply(Hy, f_w_Hy, idx, By[idx],`` and no more. The
    arithmetic legs cannot reach it either — ``_seeded`` broadcasts on axis 0 only,
    so the y and z tables are never indexed there.
    """
    sources = device_sources(kernel_source())
    for prefix, kernel in (("H", "_update_H_pml_real_kernel_code"),
                           ("E", "_update_E_pml_real_kernel_code")):
        body = sources[kernel]
        for component, axis, index in ((0, "x", "i"), (1, "y", "j"), (2, "z", "k")):
            target = prefix + "xyz"[component]
            assert f"kps_{axis}[{index}], kms_{axis}[{index}]);" in body, (
                f"{target} does not read the {axis} coefficient pair at [{index}]")
        # ...and no component reads another axis's pair. Two components both
        # reading `kps_x[i]` is exactly the armed gate mutation
        # `own_axis_to_x_for_all_three`, and it must not be the shipped text.
        for axis, index in (("x", "i"), ("y", "j"), ("z", "k")):
            assert body.count(f"kps_{axis}[{index}], kms_{axis}[{index}]);") == 1, (
                f"{kernel}: the {axis} coefficient pair is read by more than one "
                f"component")


def test_the_linear_index_is_decomposed_for_c_contiguous_storage():
    """The other indexing question: which axis is fastest.

    The predicate refuses anything that is not C-contiguous, so the last axis is
    the fastest and the decomposition is ``k = idx % nz`` / ``i = idx / (ny*nz)``.
    The Fortran-order decomposition is in bounds on every shape and swaps which
    coefficient each component reads — the same silent-absorber class as the axis
    mapping, and the gate mutation ``fortran_order_index_decomposition``.

    THE SAME THREE LINES APPEAR IN THE CERTIFIED CURL KERNELS, which is how that
    mutation turned out to resolve on all four gated kernels and to have been
    armed against none of them.
    """
    for name, body in device_sources(kernel_source()).items():
        assert "int k = idx % nz;" in body, name
        assert "int j = (idx / nz) % ny;" in body, name
        assert "int i = idx / (ny * nz);" in body, name
        for column_major in ("idx % nx", "idx / (nx * ny)"):
            assert column_major not in body, (
                f"{name} decomposes the index for column-major storage")


def test_the_e_kernel_binds_three_inverse_epsilon_pointers():
    """Defect 2 of the complex template, refused structurally.

    ``update_E_pml_complex``'s wrapper passes ``fields.inv_eps`` — the Ez
    view (fields.py:1259-1260) — for all three components, so a diagonal
    anisotropic epsilon updates Ex and Ey with Ez's material. This kernel takes
    three pointers and the wrapper fills them from ``inverse_epsilon_for``.
    """
    body = device_sources(kernel_source())["_update_E_pml_real_kernel_code"]
    for component in ("Ex", "Ey", "Ez"):
        assert f"inv_eps_{component}" in body

    # Read as CODE, not as text: the docstring right above names the defect and
    # would satisfy a substring search for it.
    tree = ast.parse(kernel_source())
    wrapper = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "_update_E_fused_pml_real")
    calls = [node for node in ast.walk(wrapper)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "inverse_epsilon_for"]
    assert len(calls) == 3, (
        f"the wrapper calls inverse_epsilon_for {len(calls)} times; one per "
        f"component is what binds three distinct pointers")
    assert {ast.literal_eval(call.args[0]) for call in calls} == {"Ex", "Ey", "Ez"}
    attributes = {node.attr for node in ast.walk(wrapper)
                  if isinstance(node, ast.Attribute)}
    assert "inv_eps" not in attributes, (
        "the wrapper reads fields.inv_eps — the Ez view — which is exactly the "
        "complex kernel's defect")


def test_the_e_source_keeps_steppings_operand_order():
    """``D * inv_eps``, D on the left (stepping.py:1011) — transcription discipline.

    INERT ON THE BITS and pinned anyway: float32 multiply is bitwise commutative
    and the sibling track measured this exact null at 30/30 identical
    (``inv_eps_left``). What it buys is that a reader comparing this line to
    ``stepping.py`` sees the same expression, not a reordered one they then have
    to reason about.
    """
    body = device_sources(kernel_source())["_update_E_pml_real_kernel_code"]
    for component, source_name in (("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz")):
        assert f"{source_name}[idx] * inv_eps_{component}[idx]" in body


def test_the_h_source_is_b_with_no_permeability_volume():
    """mu = 1 is baked into the array path, not applied here.

    ``stepping.update_H`` (:921) passes ``getattr(fields, source)`` straight
    through with ``source`` from ``H_CONSTITUTIVE_TERMS``. A kernel that
    introduced a permeability lookup would be a second engine, not a
    transcription.
    """
    from meep_gpu.cuda_kernels import own_cell_hoist  # CuPy-free

    body = device_sources(kernel_source())["_update_H_pml_real_kernel_code"]
    # The hoisted single reads its source into src_* ahead of any store; the statement
    # form -- what every lifter splices -- is the inverse's, byte for byte.
    statements = own_cell_hoist.unhoisted_kernel_code(
        body, "update_H_pml_real", "_update_H_pml_real_kernel_code")
    for component, source_name in (("Hx", "Bx"), ("Hy", "By"), ("Hz", "Bz")):
        assert f"    float src_{component[-1]} = {source_name}[idx];\n" in body
        assert f"constitutive_apply({component}, f_w_{component}, idx, " \
               f"{source_name}[idx]," in statements
    # The signature, not the comments: no permeability array is bound at all.
    signature = body[body.index("update_H_pml_real("):body.index(") {")]
    assert "mu" not in signature and "inv_mu" not in signature, signature


def test_every_device_string_is_pure_ascii():
    """MEASURED FAILURE, on device, 2026-08-15: a comment em-dash made the E
    kernel uncompilable.

    ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source to a ``.cu``
    file with a bare ``open(..., 'w')`` (compiler.py:368), so the bytes go
    through the interpreter's LOCALE encoding. Under the C/POSIX locale that is
    ASCII, and the first kernel launch died with::

        UnicodeEncodeError: 'ascii' codec can't encode character '\\u2014'
        in position 2535: ordinal not in range(128)

    Position 2535 of the concatenated E string was the ' -- ' in
    "stops being true -- it", then written with U+2014. The H string was pure
    ASCII and compiled; the E string did not and could not, on any host whose
    locale is not UTF-8. Nothing about the arithmetic was wrong, which is what
    makes it worth a test: the file was correct, reviewed, and unable to run.

    The certified sibling has ZERO non-ASCII across all fifteen of its device
    strings, so this is a property of the house's device code and not a
    concession this file is making. Both modules are checked, because a
    regression in either has the same consequence and this is the only test that
    looks.
    """
    for module in (KERNEL_MODULE, SIBLING_MODULE):
        for name, text in device_sources(module.read_text(encoding="utf-8")).items():
            offenders = [(index, character) for index, character in enumerate(text)
                         if ord(character) > 127]
            assert not offenders, (
                f"{module.name}:{name} carries {len(offenders)} non-ASCII "
                f"character(s) NVRTC cannot be handed under a C-locale host; "
                f"first at offset {offenders[0][0]} ({offenders[0][1]!r}): "
                f"...{text[max(0, offenders[0][0] - 60):offenders[0][0] + 30]!r}")


def test_the_device_strings_survive_an_ascii_locale_write():
    """The same fact as the test above, exercised the way CuPy exercises it.

    The ASCII scan says WHAT is wrong; this says the consequence follows, by
    running the identical operation ``compile_using_nvrtc`` performs — an
    ``encode`` under the strictest encoding a host can present. A future edit
    that reintroduces a non-ASCII character in some form the scan missed (an
    escape that decodes to one, a different dash) still fails here.
    """
    for module in (KERNEL_MODULE, SIBLING_MODULE):
        for name, text in device_sources(module.read_text(encoding="utf-8")).items():
            try:
                text.encode("ascii")
            except UnicodeEncodeError as error:
                raise AssertionError(
                    f"{module.name}:{name} cannot be written to a .cu file on a "
                    f"C-locale host: {error}") from error


def test_the_contraction_guard_is_the_siblings_guard():
    """``--fmad=false`` is correctness on both modules, and must be the same string.

    Two spellings would mean two compile behaviours certified by one gate. NVRTC
    accepts both ``--fmad=false`` and ``-fmad=false``; the certified pair
    compiles under the first, and so does this pair.
    """
    ours = module_level_literal(kernel_source(), "_COMPILE_OPTIONS")
    theirs = module_level_literal(
        SIBLING_MODULE.read_text(encoding="utf-8"), "_COMPILE_OPTIONS")
    assert ours == theirs == ("--fmad=false",)


def test_the_launch_geometry_matches_the_certified_curls():
    """256 lanes per block, flat 1-D grid — the same shape both tracks landed on.

    The sub-step is element-wise, so there is no tile to shape; what this pins is
    that the number is not a third, unmeasured choice.
    """
    ours = module_level_literal(kernel_source(), "_CONSTITUTIVE_THREADS")
    theirs = module_level_literal(
        SIBLING_MODULE.read_text(encoding="utf-8"), "_REAL_PML_THREADS")
    assert ours == theirs == 256


def test_the_kernel_module_re_exports_the_predicate_rather_than_copying_it():
    """Two copies of a fail-closed predicate is one predicate nobody maintains."""
    tree = ast.parse(kernel_source())
    names = {"covers_real_pml_constitutive", "constitutive_sub_lattice",
             "CONSTITUTIVE_SIDES"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("coverage"):
            imported.update(alias.name for alias in node.names)
    assert names <= imported, f"not imported from coverage: {sorted(names - imported)}"

    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    defined |= {target.id for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    assert not (names & defined), (
        f"constitutive_kernels re-defines {sorted(names & defined)}")


def test_the_probe_can_load_the_kernel_module_by_path():
    """The bit-identity probe imports these modules BY PATH, outside the package.

    ``from .coverage import …`` raises ``ImportError`` there, so the by-path
    fallback beside it is what keeps the probe working — and it would otherwise
    be exercised for the first time on the one host that runs the gate, hours
    into a device run. Checked textually, because executing the module needs
    CuPy.
    """
    tree = ast.parse(kernel_source())
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    relative = [t for t in handlers
                if any(isinstance(n, ast.ImportFrom) and n.level > 0
                       for n in ast.walk(t))]
    assert len(relative) == 2, (
        "expected the coverage and compile_cache relative imports each to carry "
        "an ImportError fallback for the by-path load")
    for block in relative:
        assert any(isinstance(handler.type, ast.Name)
                   and handler.type.id == "ImportError"
                   for handler in block.handlers)
        assert "spec_from_file_location" in ast.dump(block)


# --------------------------------------------------------------------------
# The arithmetic, measured against stepping.py — on a laptop, in float32.
#
# THE BYTE GATE NEEDS A GPU AND THIS DOES NOT. What the gate measures is whether
# the COMPILED kernel reproduces the array path; what these measure is whether
# the EXPRESSION TREE the kernel is written in reproduces it, which is the half a
# compiler cannot fix and the half that is wrong in every one of the four plain
# curl kernels (0/16 against the array path under every option set). If the tree
# is wrong here it is wrong on the device, and finding that out on a laptop in a
# tenth of a second rather than in an A6000 slot is the whole point.
#
# Every leg carries its own vacuity control: a leg that passes because the
# operands cannot tell the two groupings apart has measured nothing. The control
# is ``test_the_flattened_grouping_diverges_at_these_operands``, and the
# assumption it was WRITTEN to protect against turned out not to apply here —
# see ``test_a_unit_coefficient_does_not_hide_this_groupings_divergence``, which
# was written expecting an agreement and measured a divergence.
# --------------------------------------------------------------------------

def _broadcast(values, axis: int):
    """A per-axis coefficient vector in the shape ``PML`` stores it in."""
    shape = [1, 1, 1]
    shape[axis] = values.size
    return values.reshape(shape)


def _kernel_tree(field, fw, source, kps, kms):
    """``constitutive_apply``'s arithmetic, transcribed from the shipped device source.

    Element-wise and in float32, so it is the same tree the kernel evaluates per
    thread::

        float prev = fw[idx];
        fw[idx] = src;
        float a = f[idx] + kps * src;
        f[idx] = a - kms * prev;

    ``test_the_grouping_is_two_separate_accumulations`` pins those four lines in
    the device source, so this stays a transcription of what ships rather than of
    what it once said.
    """
    prev = fw.copy()
    fw[...] = source
    a = field + kps * source
    field[...] = a - kms * prev


def _flattened_tree(field, fw, source, kps, kms):
    """The complex kernels' tree: ``f + (kps*src - kms*prev)``. The control."""
    prev = fw.copy()
    fw[...] = source
    field[...] = field + (kps * source - kms * prev)


def _seeded(shape=(7, 5, 3), seed=20260815):
    """float32 state and non-representable coefficients, drawn once.

    The coefficients are drawn in [0.5, 1.0) rather than left at 1.0 for the
    reason the gate's own fixture states: a uniform table hides the coefficient
    indexing completely, and a power-of-two one makes the multiply exact and
    hides the grouping with it.
    """
    rng = numpy.random.default_rng(seed)
    state = {name: rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
             for name in ("field", "fw", "source")}
    coefficients = {
        "kps": _broadcast(rng.uniform(0.5, 1.0, size=shape[0]).astype(numpy.float32), 0),
        "kms": _broadcast(rng.uniform(0.5, 1.0, size=shape[0]).astype(numpy.float32), 0),
    }
    return state, coefficients


def _bytes(array) -> bytes:
    """Raw bits — never allclose. -0.0 == 0.0 and NaN != NaN both lie here."""
    return numpy.ascontiguousarray(array, dtype=numpy.float32).tobytes()


def _run(tree, steps: int, state, coefficients):
    """``steps`` consecutive sub-steps, returning the final (field, fw) bits.

    MULTI-STEP BECAUSE THE AUXILIARY IS STATE. A tree that gets ``field`` right
    and ``fw`` wrong is correct for exactly one launch and wrong forever after,
    and a single-shot comparison cannot see it.
    """
    field = state["field"].copy()
    fw = state["fw"].copy()
    source = state["source"].copy()
    for step in range(steps):
        # The source moves between sub-steps in a real run — the curl writes B/D
        # before each constitutive call — so it is perturbed here rather than held
        # fixed, which is what keeps `fw` from settling into a fixed point.
        stepped = (source * numpy.float32(0.97)).astype(numpy.float32)
        tree(field, fw, stepped, coefficients["kps"], coefficients["kms"])
        source = stepped
    return _bytes(field), _bytes(fw)


def _array_path(field, fw, source, kps, kms):
    """``stepping._apply_constitutive_pml`` itself, no scratch — the oracle."""
    stepping._apply_constitutive_pml(field, source, kps, kms, fw, scratch=None)


def test_the_kernels_expression_tree_is_bit_identical_to_stepping():
    """The transcription, measured — 60 consecutive sub-steps, byte-compared.

    60 is the budget the certified curl pair's record carries, taken from the
    sibling track's finding that 8, 10 and 6 steps all passed a divergence 40 did
    not. "Identical for N steps" is a claim about N.
    """
    state, coefficients = _seeded()
    assert _run(_kernel_tree, 60, state, coefficients) == \
        _run(_array_path, 60, state, coefficients), (
            "the kernel's expression tree diverges from "
            "stepping._apply_constitutive_pml in float32; it will diverge on the "
            "device too, and no compile option fixes a grouping")


def test_the_scratch_branch_of_the_array_path_is_the_same_tree():
    """Both array-path branches, so there is no ambiguity about which one is the oracle.

    ``_apply_constitutive_pml`` has a no-scratch branch (stepping.py:2130-2135)
    and a ``StepScratch`` branch (:2136-2143) that reaches the same tree by a
    different route. If they ever stopped agreeing, "the kernel matches the array
    path" would depend on which array path — so this pins that they do.
    """
    from ..fields import StepScratch

    state, coefficients = _seeded()
    plain = _run(_array_path, 60, state, coefficients)

    scratch = StepScratch(numpy)

    def with_scratch(field, fw, source, kps, kms):
        stepping._apply_constitutive_pml(field, source, kps, kms, fw,
                                         scratch=scratch)

    assert _run(with_scratch, 60, state, coefficients) == plain


def test_the_flattened_grouping_diverges_at_these_operands():
    """THE VACUITY CONTROL. Without it the test above measures nothing.

    ``f + (kps*src - kms*prev)`` is the tree both complex constitutive kernels in
    ``step_curl_kernels.py`` are written in. If it agreed with the array path at
    these operands, then so would anything, and the pass above would be a
    statement about the seed rather than about the transcription.
    """
    state, coefficients = _seeded()
    assert _run(_flattened_tree, 60, state, coefficients) != \
        _run(_array_path, 60, state, coefficients), (
            "the flattened grouping agrees with the array path at these operands, "
            "so the seed cannot distinguish the two trees and the bit-identity "
            "leg above is vacuous")


def test_a_unit_coefficient_does_not_hide_this_groupings_divergence():
    """MEASURED, AND IT CORRECTS AN ASSUMPTION CARRIED OVER FROM THE CURL LEG.

    The curl gate's rule is that an exactly-representable scale hides the effect:
    a Courant of 0.5 makes ``dtdx * curl`` exact, so contracted and uncontracted
    round identically and the guard becomes untestable (§1.4 — the reason that
    leg's six lifted corpus cases, all at 0.5, could not distinguish it).

    THAT RULE DOES NOT TRANSFER TO THIS SUB-STEP, and this test is the
    measurement. What the flattened grouping changes here is the association of
    the two ADDITIONS — ``((f + s) - p)`` against ``f + (s - p)`` — and that is a
    different float32 number whatever the coefficients are. Setting
    ``kps = kms = 1.0`` makes both multiplies exact and the two trees still
    diverge, which was written here as an expected AGREEMENT and measured as a
    divergence on the first run.

    The consequence for the gate is worth stating: the constitutive grouping leg
    cannot be silently disarmed by a benign coefficient table the way the curl's
    contraction leg can be by a benign Courant. What a unit table WOULD still
    hide is the FMA contraction the ``--fmad=false`` guard exists for, since that
    lives in the multiplies — which is a device-side question this file cannot
    answer and the byte gate is what answers.
    """
    state, _ = _seeded()
    ones = {"kps": numpy.float32(1.0), "kms": numpy.float32(1.0)}
    assert _run(_flattened_tree, 60, state, ones) != \
        _run(_array_path, 60, state, ones)
    # ...and the shipped tree still agrees, so the divergence above is the
    # flattening rather than the unit table upsetting something else.
    assert _run(_kernel_tree, 60, state, ones) == \
        _run(_array_path, 60, state, ones)


def test_the_e_side_source_product_is_bit_identical_either_way():
    """``D * inv_eps`` vs ``inv_eps * D`` — the null, measured here too.

    float32 multiply is bitwise commutative, so the operand order the kernel
    keeps (stepping.py:1011, D on the left) is transcription discipline and not a
    bit decision. Measuring it on the laptop is what lets the device leg
    ``cn1_inv_eps_left_must_be_uncaught`` be read as a check on the COMPARATOR
    rather than as a claim about IEEE.
    """
    rng = numpy.random.default_rng(20260815)
    d = rng.uniform(-1.0, 1.0, size=(9, 4, 5)).astype(numpy.float32)
    inv_eps = rng.uniform(0.2, 0.9, size=(9, 4, 5)).astype(numpy.float32)
    assert _bytes(d * inv_eps) == _bytes(inv_eps * d)


# --------------------------------------------------------------------------
# THE INDEXING, measured — the half the legs above provably could not reach.
#
# Everything before this point compares ONE component under coefficients
# broadcast on axis 0, so the y and z tables are never indexed and neither the
# component -> own-axis mapping nor the linear-index decomposition is exercised at
# all. Both are silent: they produce a smooth, converged absorber with the wrong
# profile rather than a visibly broken field, and note 2 of the kernel calls the
# first of them the highest-consequence confusion in the file.
#
# The two gate legs that arm them (``c7``, ``c8``) need an A6000. What is measured
# here is the same question one level up — that the SHIPPED index arithmetic
# reproduces the array path and the two wrong ones do not — which is what makes
# those legs worth a device slot rather than the first place anyone finds out.
# --------------------------------------------------------------------------

_C_ORDER, _FORTRAN_ORDER = "c", "fortran"


def _index_volumes(shape, order: str):
    """Per-cell (i, j, k) as the kernel computes them, for either memory order.

    C order is what ships and what the predicate's contiguity clause guarantees::

        int k = idx % nz;  int j = (idx / nz) % ny;  int i = idx / (ny * nz);

    The Fortran order is the mutation. It is IN BOUNDS ON EVERY SHAPE — ``idx % nx``
    is below ``nx``, and ``idx / (nx*ny)`` is below ``nz`` because
    ``idx < nx*ny*nz`` — which is why the gate mutation is spelled this way rather
    than by swapping ``i`` and ``k`` in place: a decomposition that read PAST a
    coefficient vector would be caught by a fault on some shapes and by garbage on
    others, and neither is the defect.
    """
    nx, ny, nz = shape
    flat = numpy.arange(nx * ny * nz)
    if order == _C_ORDER:
        i, j, k = flat // (ny * nz), (flat // nz) % ny, flat % nz
    else:
        i, j, k = flat % nx, (flat // nx) % ny, flat // (nx * ny)
    assert i.max() < nx and j.max() < ny and k.max() < nz, (
        "the decomposition leaves the coefficient vectors' range; this leg is "
        "measuring an out-of-bounds read rather than a wrong coefficient")
    return [a.reshape(shape) for a in (i, j, k)]


def _three_component_run(shape, tables, state, steps: int, axis_of_component,
                         order: str):
    """``steps`` sub-steps of all three components, indexing exactly as the kernel does.

    ``axis_of_component`` is the mapping under test: ``(0, 1, 2)`` is the shipped
    one — each component reads the table for the direction it points in — and
    ``(0, 0, 0)`` is the armed mutation ``own_axis_to_x_for_all_three``.
    """
    volumes = _index_volumes(shape, order)
    fields = [state["field"][c].copy() for c in range(3)]
    fws = [state["fw"][c].copy() for c in range(3)]
    sources = [state["source"][c].copy() for c in range(3)]
    for _ in range(steps):
        for component in range(3):
            axis = axis_of_component[component]
            kps = tables["kps"][axis][volumes[axis]]
            kms = tables["kms"][axis][volumes[axis]]
            source = (sources[component] * numpy.float32(0.97)).astype(numpy.float32)
            _kernel_tree(fields[component], fws[component], source, kps, kms)
            sources[component] = source
    return b"".join(_bytes(a) for a in fields + fws)


def _array_path_run(shape, tables, state, steps: int):
    """The oracle: ``stepping._apply_constitutive_pml`` per component, own axis.

    Coefficients enter as ``PML`` stores them — broadcast on the component's own
    axis — which is how ``stepping.update_H`` / ``update_E`` pass them
    (``_constitutive_coefficients(pml, axis, ...)``). No index arithmetic of its
    own, so agreement with it is a statement about the kernel's.
    """
    fields = [state["field"][c].copy() for c in range(3)]
    fws = [state["fw"][c].copy() for c in range(3)]
    sources = [state["source"][c].copy() for c in range(3)]
    for _ in range(steps):
        for component in range(3):
            source = (sources[component] * numpy.float32(0.97)).astype(numpy.float32)
            stepping._apply_constitutive_pml(
                fields[component], source,
                _broadcast(tables["kps"][component], component),
                _broadcast(tables["kms"][component], component),
                fws[component], scratch=None)
            sources[component] = source
    return b"".join(_bytes(a) for a in fields + fws)


def _three_component_state(shape, seed=20260815):
    rng = numpy.random.default_rng(seed)
    state = {name: [rng.uniform(-1.0, 1.0, size=shape).astype(numpy.float32)
                    for _ in range(3)]
             for name in ("field", "fw", "source")}
    tables = {label: [rng.uniform(0.5, 1.0, size=shape[axis]).astype(numpy.float32)
                      for axis in range(3)]
              for label in ("kps", "kms")}
    return state, tables


# (7, 5, 3) is the ordinary case; (5, 5, 5) is the one where a mis-decomposed
# index stays in bounds on every axis and the error is silent rather than a fault.
_INDEXING_SHAPES = ((7, 5, 3), (5, 5, 5))


@pytest.mark.parametrize("shape", _INDEXING_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_the_shipped_axis_mapping_and_decomposition_reproduce_the_array_path(shape):
    """The control both mutations below are measured against."""
    state, tables = _three_component_state(shape)
    assert _three_component_run(shape, tables, state, 60, (0, 1, 2), _C_ORDER) == \
        _array_path_run(shape, tables, state, 60), (
            "the kernel's index arithmetic does not reproduce the array path; "
            "every component must read the coefficient table for the direction it "
            "points in, gathered through the C-order decomposition")


@pytest.mark.parametrize("shape", _INDEXING_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_binding_every_component_to_the_x_table_diverges(shape):
    """Gate mutation ``c7``, measured on the expression tree.

    If this agreed, the device leg would be unfalsifiable and the tables would be
    interchangeable — which they are not, by construction: they are drawn per axis
    and never 1.0, for exactly this reason.
    """
    state, tables = _three_component_state(shape)
    assert _three_component_run(shape, tables, state, 60, (0, 0, 0), _C_ORDER) != \
        _array_path_run(shape, tables, state, 60)


@pytest.mark.parametrize("shape", _INDEXING_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_the_column_major_decomposition_diverges_and_stays_in_bounds(shape):
    """Gate mutation ``c8``, measured — including that it is a wrong ANSWER.

    ``_index_volumes`` asserts the mutated decomposition never leaves the
    coefficient vectors' range, so what this measures is a silently wrong
    absorber and not an out-of-bounds read that a device would catch for free.
    """
    state, tables = _three_component_state(shape)
    assert _three_component_run(shape, tables, state, 60, (0, 1, 2),
                                _FORTRAN_ORDER) != \
        _array_path_run(shape, tables, state, 60)


def test_the_one_component_legs_above_provably_cannot_see_the_axis_mapping():
    """WHY THIS SECTION EXISTS, stated as a measurement rather than as a claim.

    ``_seeded`` broadcasts on axis 0 and steps ONE component, which is component 0
    — and component 0 reads ``kps_x`` under the shipped mapping AND under the
    all-x mutation. The two are the same computation there, so a battery made of
    those legs alone would report the mutation correct however lethal it is on the
    other two components.

    Measured by running all three under both mappings and comparing only the first
    component's bytes: identical, while the full three-component comparison above
    diverges.
    """
    shape = (7, 5, 3)
    state, tables = _three_component_state(shape)

    def first_component_only(axis_of_component):
        volumes = _index_volumes(shape, _C_ORDER)
        axis = axis_of_component[0]
        field, fw = state["field"][0].copy(), state["fw"][0].copy()
        source = state["source"][0].copy()
        for _ in range(8):
            stepped = (source * numpy.float32(0.97)).astype(numpy.float32)
            _kernel_tree(field, fw, stepped,
                         tables["kps"][axis][volumes[axis]],
                         tables["kms"][axis][volumes[axis]])
            source = stepped
        return _bytes(field) + _bytes(fw)

    assert first_component_only((0, 1, 2)) == first_component_only((0, 0, 0)), (
        "component 0 distinguishes the two mappings after all; then this "
        "section's premise is wrong and it should say so")


# --------------------------------------------------------------------------
# The two required predicate mutations (R4). Applied to the REAL source.
# --------------------------------------------------------------------------

def mutated_predicate(*patterns: str):
    """Recompile ``covers_real_pml_constitutive`` with refusals edited out.

    Textual, against the shipping source, and it raises if any pattern matches
    nothing — so a mutation that has drifted away from the code fails loudly
    instead of quietly testing an unmutated function.
    """
    source = textwrap.dedent(
        inspect.getsource(coverage.covers_real_pml_constitutive))
    for pattern in patterns:
        source, count = re.subn(pattern, "", source)
        if count != 1:
            raise AssertionError(
                f"predicate mutation {pattern!r} matched {count} times, expected 1; "
                f"the mutation and the predicate have drifted apart and it is "
                f"pinning nothing.")
    return _predicate_from(source)


def _predicate_from(source: str):
    """Compile one edited copy of the predicate against the real module globals.

    Split out of :func:`mutated_predicate` on 2026-08-19 because the fold clause
    is pinned by INSERTING a refusal rather than deleting one, and a mutation
    helper that can only delete cannot arm a narrowing.
    """
    namespace = dict(vars(coverage))
    namespace["__builtins__"] = __builtins__
    module = types.ModuleType("mutated_predicate")
    module.__dict__.update(namespace)
    exec(compile(source, "<mutated covers_real_pml_constitutive>", "exec"),
         module.__dict__)
    return module.covers_real_pml_constitutive


@pytest.mark.parametrize("side", SIDES)
def test_the_fold_is_admitted_and_the_admission_is_not_an_accident(side, xp):
    """The 2026-08-19 widening, pinned in the direction it can now fail.

    Until 2026-08-19 this was ``test_mutation_widen_by_one_axis_is_caught``: it
    built a folded grid, asserted the refusal, and then had to delete THREE
    statements (the whole-grid ``has_symmetry``, the per-axis ``is_mirrored``,
    and the boundary-kind list a mirrored axis resolves to ``"mirror"`` on) to
    make the predicate leak. All three are gone, so the mutation has nothing left
    to defeat and the test that pinned them is now a test that pins the opposite
    thing: the fold is ADMITTED, and re-adding any ONE of those statements is a
    NARROWING that must show up as a refusal.

    THE MUTATION IS THE OLD CLAUSE. Re-inserting it is the edit a reader who
    knows the curl family and not the gate would make on sight, so it is the
    edit worth arming — and each of the three is armed separately, because the
    old refusal was stated three times and a narrowing only has to land once.
    """
    fields, layer, grid = build(xp, symmetry=("y",))
    covered, reason = coverage.covers_real_pml_constitutive(fields, layer, grid, side)
    assert covered, (
        f"the shipped predicate refuses a folded grid on update_{side} "
        f"({reason}); the fold admission CONSTITUTIVE_FOLD_ADMISSION records has "
        f"been narrowed and the census slot count no longer describes it")

    # THE MARKER MOVED when the cylindrical clause was removed on 2026-08-20
    # (CONSTITUTIVE_CYLINDRICAL_ADMISSION). Anchored on the fold-count clause
    # instead, which is the nearest statement that is still about the fold -- and
    # which this test is about. A marker that no longer matches fails loudly here
    # rather than inserting nothing and reporting three passes.
    marker = '    folded = sum(1 for axis in range(3) if facts["mirrored"][axis])\n'
    narrowings = {
        "whole-grid has_symmetry": (
            '    if facts["has_symmetry"]:\n'
            '        return False, "NARROWED: has_symmetry"\n',
            "NARROWED: has_symmetry"),
        "per-axis is_mirrored": (
            '    if any(facts["mirrored"]):\n'
            '        return False, "NARROWED: is_mirrored"\n',
            "NARROWED: is_mirrored"),
        "the curl family's boundary-kind list": (
            '    if any(kind not in BC_CODES\n'
            '           for kind in _boundary_kinds_from(facts)):\n'
            '        return False, "NARROWED: BC_CODES"\n',
            "NARROWED: BC_CODES"),
    }
    for label, (clause, expected) in narrowings.items():
        source = textwrap.dedent(
            inspect.getsource(coverage.covers_real_pml_constitutive))
        assert marker in source, "the insertion point has moved"
        narrowed = _predicate_from(source.replace(marker, clause + marker, 1))
        still_covered, why = narrowed(fields, layer, grid, side)
        assert not still_covered, (
            f"re-adding {label} did NOT refuse the folded grid, so this leg is "
            f"measuring nothing and the admission is unpinned in that direction")
        # THE REASON, not merely a refusal: each leg's inserted clause carries a
        # distinct string, so a refusal from some OTHER clause would pass a
        # verdict-only check while measuring nothing about the one named.
        assert why == expected, (
            f"{label} was defeated by some OTHER clause ({why!r}), so this leg "
            f"is not measuring the narrowing it names")


def test_mutation_drop_the_dispersion_refusal_is_caught(xp):
    """Drop it and ``update_E`` covers a run whose source is not D at all.

    With a susceptibility registered the array path's source is ``D - sum P``
    (stepping.py:1010, fields.py:1096-1105) and ``update_P`` closes the step. The
    kernel binds ``fields.D*`` directly, so what leaks here is not a last-bit
    regrouping — it is the polarization silently not being subtracted, on a run
    that still converges and still looks like a dielectric.

    This clause has NO second line of defence, which is the finding: unlike the
    complex-storage refusal (stated twice, flag and per-array dtype) and the
    fold (stated three times), one deletion is enough.
    """
    fields, layer, grid = configuration("polarizations", "E", xp)
    assert not coverage.covers_real_pml_constitutive(fields, layer, grid, "E")[0]

    dropped = mutated_predicate(
        r'        if getattr\(fields, "polarizations", None\):\n'
        r'            return False, \("dispersion[\s\S]*?closes the step"\)\n')
    covered, reason = dropped(fields, layer, grid, "E")
    assert covered, (
        f"dropping the dispersion refusal did not leak, so something else is "
        f"refusing it and the clause is unpinned (it refused: {reason})")

    # And the H side never had it: the same configuration is covered there
    # unmutated, which is the per-sub-step split rather than a hole.
    assert coverage.covers_real_pml_constitutive(fields, layer, grid, "H")[0]


# --------------------------------------------------------------------------
# THE 2026-08-19 FOLD ADMISSION — the clause, its data, and what did NOT move
# --------------------------------------------------------------------------

def test_the_admitted_boundary_kinds_carry_the_fold_and_the_axis():
    """The clauses of this predicate that diverge from the curl's, as data.

    BOTH divergences are paid for by a device gate on the SHIPPED pair, and
    neither by an argument. The fold came first
    (:data:`coverage.CONSTITUTIVE_FOLD_ADMISSION`, 2026-08-19) and the cylindrical
    axis followed (:data:`coverage.CONSTITUTIVE_CYLINDRICAL_ADMISSION`,
    2026-08-20, 96/96 cases bit-identical with 80 of them resolving r to
    ``CYL_AXIS``). The obvious tidy-up — putting ``BC_CODES`` back here so this
    family reads like the curl's — costs 128 slots for the fold and 32 more for
    Dcyl, so it fails here rather than only in a census nobody re-runs.

    THE ADE FAMILY IS NO LONGER REQUIRED TO MATCH. It did while both lists were
    (PERIODIC, METALLIC, MIRROR); the constitutive pair has now been measured on
    a kind the ADE pair has not, and asserting they are equal would force the
    unmeasured one to move with the measured one. What is asserted instead is the
    direction that is actually safe: this family admits everything the ADE family
    does.
    """
    assert coverage.MIRROR in coverage.CONSTITUTIVE_BOUNDARY_KINDS
    assert coverage.CYL_AXIS in coverage.CONSTITUTIVE_BOUNDARY_KINDS
    assert set(coverage.BC_CODES) < set(coverage.CONSTITUTIVE_BOUNDARY_KINDS)
    assert set(coverage.ADE_BOUNDARY_KINDS) <= \
        set(coverage.CONSTITUTIVE_BOUNDARY_KINDS)


def test_the_fold_plane_cap_is_the_number_the_gate_swept():
    """The refusal that names its own price.

    Two simultaneous planes were measured and three were not, so three is
    refused. Written as a number the clause READS rather than a literal in the
    clause, because the number is what a further fold spec in the gate moves —
    and the refusal string quotes it, so a cap raised without a gate leg shows up
    as a changed message rather than as silence.
    """
    table = coverage.CONSTITUTIVE_FOLD_ADMISSION
    # THE NUMBER MUST BE A RUN'S, and the run is named. Raised 2 -> 3 on
    # 2026-08-20 by adding three fold_XYZ_* specs and RE-CUTTING; the block below
    # is that run, and the cap has to equal what its cases actually reached rather
    # than what its spec tuple listed.
    block = json.loads(RECORD.read_text(encoding="utf-8"))[
        "folded_constitutive_2026-08-20_threeplane"]
    assert block["passed"] is True
    assert table["folded_planes_swept"] == block["folded_planes_swept"] == 3, (
        "the cap and the run that produced it disagree; if a further fold spec "
        "was added, RE-CUT the gate and regenerate the block with "
        "results/cuda_folded_constitutive_2026-08-20_threeplane/"
        "build_record_block.py — otherwise put the number back")
    # ...and the run's own CASES must carry the count, not merely its spec list.
    counted = {int(k) for k in
               block["fold_structure_swept"]["cases_by_number_of_fold_planes"]}
    assert max(counted) == table["folded_planes_swept"], (counted, table)
    # THE CORPUS ROW THE CAP USED TO COST, and what it costs now.
    assert table["slots_refused_for_a_third_fold_plane"] == 0
    assert table["row_with_three_fold_planes"] == "TestLDOS.test_ldos_3D"
    source = inspect.getsource(coverage.covers_real_pml_constitutive)
    assert 'CONSTITUTIVE_FOLD_ADMISSION["folded_planes_swept"]' in source, (
        "the cap is a literal in the clause rather than a read of the table; a "
        "reader who changes the table would then not change the predicate")


def test_lowering_the_fold_plane_cap_brings_the_refusal_back(xp):
    """The cap clause is unreachable at 3 -- so what is measured is the READ.

    ``folded`` counts mirrored axes out of ``range(3)`` and the cap is 3, so
    ``folded > cap`` cannot be true on any grid this engine can build; the clause
    is on ``_UNREACHED_REFUSALS`` for exactly that reason. What can still be
    measured is that the number DECIDING a three-plane grid is the record's: put
    2 back and the refusal returns, by name and quoting the number.

    Without this, "the cap is a read of the table" rests on a string search of the
    source, and a clause that read the table and ignored it would pass.
    """
    for side in ("H", "E"):
        fields, layer, grid = configuration("mirror_three_planes", side, xp)
        assert coverage.covers_real_pml_constitutive(fields, layer, grid, side)[0]

        table = coverage.CONSTITUTIVE_FOLD_ADMISSION
        shipped = table["folded_planes_swept"]
        try:
            table["folded_planes_swept"] = 2
            covered, reason = coverage.covers_real_pml_constitutive(
                fields, layer, grid, side)
            assert not covered, (
                "lowering the record's plane count did not change the verdict, so "
                "the predicate is not reading the record and the cap is unpinned")
            assert "3 mirror planes at once" in reason, reason
            assert "swept 2 simultaneous" in reason, reason
        finally:
            table["folded_planes_swept"] = shipped
        assert coverage.covers_real_pml_constitutive(fields, layer, grid, side)[0]


def test_the_fold_admission_names_a_record_block_that_reports_a_pass():
    """A widening is a claim about a DEVICE RUN, so the record has to hold one.

    Both directions, the shape ``test_the_record_backs_every_certified_kernel``
    uses: the table's numbers must be the block's numbers, and the block must
    report a released verdict under BOTH policies with a diverging guard control.
    The table is what the predicate reads; the block is what ran. Letting them
    drift is how a slot count outlives the measurement behind it.
    """
    table = coverage.CONSTITUTIVE_FOLD_ADMISSION
    entry = json.loads(RECORD.read_text(encoding="utf-8"))
    # RE-POINTED 2026-08-22, from folded_constitutive_2026-08-19 to the block that
    # describes the bytes that SHIP. The kernel rename moved constitutive_kernels.py,
    # and the clause at the foot of this test refuses to let a moved kernel module be
    # declared away -- a fold verdict cut on bytes that no longer exist licenses
    # nothing. The two-plane 2026-08-19 sweep cannot be re-cut either: FOLD_SPECS now
    # carries the fold_XYZ_* specs and the gate has no flag to restrict them, so that
    # run is not reproducible. The three-plane block IS re-cuttable and was re-cut on
    # cuda_folded_constitutive_2026-08-21_rename, which compiled the shipped module.
    block = entry.get("folded_constitutive_2026-08-20_threeplane")
    assert block is not None, (
        "coverage.py admits a fold and certification.json has no "
        "folded_constitutive_2026-08-20_threeplane block behind it")
    superseded = entry.get("folded_constitutive_2026-08-19")
    assert superseded is not None and superseded.get("superseded_by") == (
        "folded_constitutive_2026-08-20_threeplane"), (
        "the 2026-08-19 block is still present and does not say what replaced it; a "
        "reader would take it for a live licence")
    assert block["passed"] is True
    assert block["kernel_module"] == "meep_gpu/cuda_kernels/constitutive_kernels.py"
    assert set(block["kernels"]) == set(
        module_level_literal(kernel_source(), "CERTIFIED_KERNELS")), (
        "the fold verdict was cut on a different set of kernels than this module "
        "certifies")
    assert sorted(block["policies_cut_under"]) == ["ieee_keep_ftz_stripped",
                                                   "meep_x86_flush"]
    assert block["recorded_utc"] == table["recorded_utc"]
    assert block["host"] == table["host"]
    assert list(block["folded_terminations_swept"]) == \
        list(table["folded_terminations_swept"])
    # THE BLOCK'S PLANE COUNT AND THE CAP ARE NOW ONE RUN'S. The superseded
    # 2026-08-19 sweep stopped at two; this block reached three, which is where the
    # cap comes from and what
    # ``test_the_fold_plane_cap_is_the_number_the_gate_swept`` reads. The <= clause
    # is kept: it still catches a sweep that reached MORE planes than the shipped
    # cap, which would mean the predicate is narrower than a run that happened.
    assert block["folded_planes_swept"] == 3, block["folded_planes_swept"]
    assert block["folded_planes_swept"] <= table["folded_planes_swept"], (
        "the fold sweep reached MORE planes than the shipped cap, so the "
        "cap is narrower than a run that already happened")
    for policy in ("keep", "flush"):
        leg = block["per_policy"][policy]
        assert leg["released"] is True, policy
        assert leg["cases_scored"] == table["cases_per_policy"], policy
        assert leg["cases_folded"] == table["folded_cases_per_policy"], policy
        assert leg["single_launch_identical"] == table["identical_single_launch"]
        assert leg["multi_step_identical"] == table["identical_at_sixty_launches"]
        assert leg["multi_step_launches_each"] == MULTI_STEP_BUDGET, policy
        # THE GUARD IS EVIDENCE, NOT DECORATION.
        guard = leg["guard_control"]
        assert guard["scored_at_inexact_courant"] > 0 and \
            guard["diverged"] == guard["scored_at_inexact_courant"], policy
        # THE FOLD-SPECIFIC MUTATION — the refusal armed as a defect. Without it
        # "the fold does not move the coefficient index" rests on a leg nobody
        # showed could fail on a folded grid.
        reversed_leg = leg["host_mutations"]["reverse_folded_axis_coefficients"]
        assert reversed_leg["verdict"] == "CAUGHT", policy
        assert reversed_leg["caught"] == reversed_leg["ran"] > 0, policy
        # AND A MUTATION THAT MUST BE UNCAUGHT, or the battery scores the same
        # whether the comparator works or fails everything.
        null = leg["source_mutations"]["E:commute_constitutive_scale"]
        assert null["verdict"] == "NULL CONFIRMED" and null["caught"] == 0, policy
    # THE FOLD STRUCTURE, COUNTED. The block's own census of what was folded has
    # to agree with the cap the predicate reads, and every folded axis set in it
    # has to be within that cap — a widening whose evidence is described rather
    # than counted is how the sibling ADE block shipped a fold claim its case
    # records disproved.
    structure = block["fold_structure_swept"]
    assert structure["folded_cases_guarded"] > 0
    multiplicities = {int(k) for k in structure["cases_by_number_of_fold_planes"]}
    assert max(multiplicities) == block["folded_planes_swept"], (
        f"the cases carry up to {max(multiplicities)} fold planes and this block "
        f"says its sweep reached {block['folded_planes_swept']}")
    assert set(structure["folded_axis_terminations"]) == \
        set(table["folded_terminations_swept"])
    # EVERY AXIS FOLDED SOMEWHERE: a sweep carrying one folded axis proves
    # nothing about the other two, since the coefficient index is a different
    # stride on each.
    folded_axes = {axis for key in structure["cases_by_folded_axis_set"]
                   for axis in key if axis in "XYZ"}
    assert folded_axes == {"X", "Y", "Z"}, folded_axes
    identical, ran = structure[
        "folded_cases_bit_identical_at_sixty_launches"].split("/")
    assert identical == ran == str(structure["folded_cases_guarded"])
    assert structure["multi_step_launches_on_every_folded_case"] == \
        [MULTI_STEP_BUDGET]

    # THE FLOOR THAT REFUSES A VACUOUS VERDICT: on the folded axis the absorber
    # must actually absorb, or a coefficient-index error there is invisible.
    low, high = block["per_policy"]["keep"]["folded_axis_absorbs_deviation_range"]
    assert low > 0.0 and high >= low

    # AND THE SUBJECTS, INCLUDING THE GATE ITSELF. This block licenses a
    # widening rather than certifying a kernel, so what has to stay put is the
    # instrument: an edited gate is a different experiment, and a kernel module
    # edited after the run is a verdict about bytes that no longer ship. A
    # subject that has moved must be declared, and the declaration must describe
    # the file as it is now.
    edited = {name for name in block["revision_sha256"] if not name.startswith("_")}
    for name, recorded in block["subject_sha256"].items():
        path = HERE.parents[1] / name
        if not path.exists():
            continue
        live = hashlib.sha256(path.read_bytes()).hexdigest()
        if live == recorded:
            continue
        # ONE HOME FOR THE RULE (device_identity.py:209). A subject whose edit
        # cannot reach the device -- device source unchanged where the file
        # carries some, prose only where it carries none -- needs no
        # declaration; there is nothing for a declaration to say. The helper
        # returns False for anything it cannot establish, so a block recording
        # no device_sha256/code_sha256 keeps the declared-edit rule below.
        if weld_survives_edit(path, block, name):
            continue
        assert name in edited, (
            f"{name} has changed since the fold gate ran and no "
            f"post_certification_edits entry says why")
        assert block["revision_sha256"][name] == live, (
            f"{name} has changed AGAIN since the recorded post-gate edit")
    for entry in block["post_certification_edits"]:
        assert entry["touches_device_code"] is False, (
            "a post-gate edit that touched device code invalidates the verdict")
    # The kernel module is the one subject that must NOT be on that list: the
    # whole claim is that no device source changed.
    assert "meep_gpu/cuda_kernels/constitutive_kernels.py" not in edited, (
        "the fold verdict was cut on constitutive_kernels.py and that file has "
        "been edited since; nothing here licenses a widening any more")


@pytest.mark.parametrize("side", SIDES)
def test_the_sibling_predicates_did_not_widen_with_this_one(side, xp):
    """No family may inherit another's fold verdict. EACH needs its own gate.

    This pinned "the curl still refuses" until 2026-08-19, and the reason was
    right: the gate that widened THIS pair ran constitutive_kernels.py's two
    kernels and nothing else, so copying the widening across is the obvious edit
    for a reader who sees three predicates with one shape.

    THE CURL NOW HAS ITS OWN VERDICT AND ITS OWN DEVICE CODE, and neither came
    from here. Measured on the GPU host (results/cuda_folded_curl_2026-08-19/ and its
    2026-08-20 re-run): a folded METALLIC axis was bit-identical in the SHIPPED
    curl pair with nothing written, 48/48 at one launch AND at 60 under both
    policies; a folded PERIODIC axis DIVERGED 0/64, with all 19,649 differing words
    on one plane and zero elsewhere, and was admitted only after a KERNEL BRANCH
    (``BC_MIRROR_PERIODIC``) was added to close it. That asymmetry — one family
    taking the fold for free, the other paying device code for half of it — is the
    whole reason no family may inherit another's fold verdict.

    THE GUARD STILL BITES, and it caught a real leak the same day: the folded
    resolution was first written into ``_boundary_kinds_from`` - "one copy, two
    callers" - which widened THIS pair as a side effect, with no device evidence
    for the configurations it admitted. This test failed, and the fix was to scope
    the resolution to the curl predicate. What is pinned now is that each family's
    admission traces to its own gate, not that the others stay refused forever —
    and the family with NO fold gate at all is still refused.
    """
    fields, layer, grid = build(xp, symmetry=("y",))
    assert coverage.covers_real_pml_constitutive(fields, layer, grid, side)[0]

    # The curl's fold admission is its own, and its RE-CUT is its own too: on
    # 2026-08-20 both families' gates were re-run at three planes, separately, and
    # each raised its own record. Agreeing on a grid is not inheritance; what
    # would be inheritance is one number deciding both.
    for sub_step in ("step_B", "step_D"):
        covered, why = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
        assert covered, (sub_step, why)
    _fields3, _layer3, three = build(xp, symmetry=("x", "y", "z"))
    for sub_step in ("step_B", "step_D"):
        covered, why = coverage.covers_real_pml_curl(_fields3, _layer3, three, sub_step)
        assert covered, (sub_step, why)
    assert coverage.CURL_FOLD_ADMISSION["gate"] != \
        coverage.CONSTITUTIVE_FOLD_ADMISSION["gate"], (
        "the two fold admissions name the same gate, so one family's verdict IS "
        "the other's and this test is measuring a tautology")
    assert coverage.CURL_FOLD_ADMISSION["planes_round_artifacts"] != \
        coverage.CONSTITUTIVE_FOLD_ADMISSION["planes_round_artifacts"]

    # AND TWO FAMILIES THAT DID NOT MOVE WITH THEM, which is where the guard now
    # bites. Both were one shared constant away from being widened for free on
    # 2026-08-20, and both were pinned to their OWN gate's number instead.
    #
    # THE ADE update_P FAMILY read this record's cap directly until that day. Its
    # gate has never scored a three-plane case, and the corpus drives no
    # three-plane dispersive row -- so inheriting the raise would have bought zero
    # slots and been invisible to the census: an over-claim with nothing anywhere
    # behind it.
    from . import dispersive_kernels  # noqa: PLC0415

    assert dispersive_kernels.ADE_FOLD_PLANES_SWEPT == 2, (
        "the ADE family's plane cap moved; if gate_cuda_ade.py gained a "
        "three-plane spec and was RE-RUN, update it there — otherwise it is "
        "inheriting this pair's measurement")
    assert dispersive_kernels.ADE_FOLD_PLANES_SWEPT < \
        coverage.CONSTITUTIVE_FOLD_ADMISSION["folded_planes_swept"]
    refusal = dispersive_kernels._shared_dispersive_refusals(_fields3, three)
    assert refusal is not None and "3 mirror planes at once" in refusal, refusal
    assert "this family's gate" in refusal, refusal

    # THE NO-ABSORBER CURL delegates to covers_real_pml_curl and would have
    # inherited BOTH of that predicate's 2026-08-20 widenings. Its own gate stops
    # at two planes and installs no chi, so it now states both refusals itself.
    from . import no_pml_curl  # noqa: PLC0415

    assert no_pml_curl.NO_PML_CURL_NARROWER_THAN_THE_SIBLING[
        "folded_planes_swept"] == 2
    covered, why = no_pml_curl.covers_no_pml_curl(_fields3, None, three, "step_B")
    assert not covered and "3 mirror planes at once" in why, why

    # THE OFF-DIAGONAL HAS NO FOLD EVIDENCE AT ALL and must still refuse: its row
    # product reads neighbouring cells of the OTHER components' volumes, which is
    # precisely what a fold changes, and no gate has ever run it folded. This is
    # the clause that keeps the guard meaningful now that two of the three families
    # admit a fold.
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": None}}
    covered, why = coverage.covers_real_pml_offdiag_constitutive(fields, layer, grid)
    assert not covered
    assert why.startswith("mirror symmetry:"), why


def test_the_widening_did_not_open_the_offdiagonal_disjointness_seam(xp):
    """No configuration may be admitted by both E-side predicates.

    The census measures this over all 186 rows and reports zero; the widening
    moves 53 rows into this predicate's ``update_E`` admission, so the seam is
    worth re-measuring here rather than trusting that an E-side clause the fold
    never touched still fires. It does fire, and it is a DIFFERENT clause from
    the fold one — the off-diagonal refusal, which is why the two stay disjoint
    on a folded grid as well as on a flat one.
    """
    fields, layer, grid = build(xp, symmetry=("y",))
    fields._chi1inv_offdiagonal = {"Ex": {"Ey": None}}
    plain = coverage.covers_real_pml_constitutive(fields, layer, grid, "E")
    tensor = coverage.covers_real_pml_offdiag_constitutive(fields, layer, grid)
    assert not (plain[0] and tensor[0]), (plain, tensor)
    assert plain[1].startswith("off-diagonal chi1inv:"), plain[1]


# --------------------------------------------------------------------------
# THE SLOT COUNT, RE-DERIVED FROM THE CENSUS RECORD RATHER THAN QUOTED
# --------------------------------------------------------------------------
#
# ``CONSTITUTIVE_FOLD_ADMISSION`` carries two numbers that describe a corpus:
# what this predicate admitted before the widening and what it admits after. A
# number in a table nothing recomputes is a number that outlives its measurement,
# which is the failure mode ``artifact_sha256`` exists for on the device side.
# Here the recompute is cheap and needs no GPU, so it runs at the merge bar.
#
# results/ is untracked, so this is a declared skip on a host without the census.

CENSUS = (HERE.parents[1] / "parity/meep_gpu/results"
          / "cuda_predicate_coverage_2026-08-16_offdiag")

#: The refusal ``covers_real_pml_constitutive`` USED to state on every folded run.
#: Rows the census recorded under it are the ones the widening acts on; every
#: other recorded refusal must still be reproduced verbatim, which is what makes
#: the stand-ins below a replay rather than a second predicate.
_RETIRED_FOLD_REFUSAL = (
    "mirror symmetry: the fold changes the stored extent, and the extent is "
    "what turns a cell index into a coefficient index")

#: The refusal it stated on every Dcyl run, retired on 2026-08-20 by
#: :data:`coverage.CONSTITUTIVE_CYLINDRICAL_ADMISSION`. Same treatment and same
#: reason: the census predates the widening, so rows it recorded under this
#: refusal are the ones the widening acts on and are counted as GAINED rather
#: than replayed. Every other recorded refusal must still be reproduced verbatim.
_RETIRED_CYLINDRICAL_REFUSAL = (
    "cylindrical (Dcyl): the axial extent moves every coefficient index")

#: Both, for the one check that has to skip either.
_RETIRED_REFUSALS = (_RETIRED_FOLD_REFUSAL, _RETIRED_CYLINDRICAL_REFUSAL)


class _CensusArray:
    """A float32 C-contiguous volume of ``shape`` that allocates nothing.

    The predicate reads ``dtype``, ``shape``, ``size`` and ``flags.c_contiguous``
    and nothing else, and the corpus runs up to 89**3 cells across eighteen
    volumes a row. Standing in for them is also what makes the derived count an
    explicit UPPER BOUND: the census record cannot say whether a row's real
    buffers were well-formed, only what the decision logic above them did.
    """

    class _Flags:
        c_contiguous = True

    def __init__(self, shape, xp):
        self.shape = tuple(shape)
        self.dtype = xp.float32
        self.flags = self._Flags()
        self.size = int(numpy.prod(self.shape)) if self.shape else 1


class _CensusGrid:
    def __init__(self, configuration, xp):
        self._c = configuration
        self.xp = xp
        self.shape = tuple(configuration["shape"])
        self.cylindrical = configuration["cylindrical"]
        self.has_bloch = configuration["has_bloch"]
        self.bfast_active = configuration["bfast_active"]
        self.beta = configuration["beta"]

    def has_symmetry(self):
        return self._c["has_symmetry"]

    def is_mirrored(self, axis):
        return self._c["mirrored"][axis]

    def is_axis(self, axis):
        return self._c["is_axis"][axis]

    def is_metallic(self, axis):
        return self._c["metallic"][axis]


class _CensusFields:
    def __init__(self, configuration, xp):
        shape = tuple(configuration["shape"])
        self.force_complex_fields = configuration["force_complex_fields"]
        self.stores_E = configuration["stores_E"]
        self.has_offdiagonal_epsilon = configuration["has_offdiagonal_epsilon"]
        self.polarizations = [None] * configuration["n_polarizations"]
        # ``Fields.has_nonlinearity`` reads ``_chi2_components`` alone, which is
        # the attribute the census recorded; the predicate reads both maps by
        # name. The two agree on this corpus, and the reproduction leg below is
        # what says so rather than this comment.
        self._chi2_components = {"Ex": 0.5} if configuration["has_nonlinearity"] else None
        self._chi3_components = None
        for side in coverage.CONSTITUTIVE_SIDES.values():
            for name in tuple(side["targets"]) + tuple(side["aux"]) + tuple(side["sources"]):
                setattr(self, name, _CensusArray(shape, xp))
        self._inverse = _CensusArray(shape, xp)

    def inverse_epsilon_for(self, component):
        return self._inverse


class _CensusPML:
    def __init__(self, configuration, xp):
        shape = tuple(configuration["shape"])
        self.is_active = configuration["pml_active"]
        for axis, name in enumerate(("x", "y", "z")):
            broadcast = tuple(shape[axis] if a == axis else 1 for a in range(3))
            for label in ("kps", "kms"):
                for suffix in ("", "_h"):
                    setattr(self, f"{label}_{name}{suffix}",
                            _CensusArray(broadcast, xp))


def _census_rows():
    """The analyzer's own loader: examples + tests, parameterized rows recovered."""
    def load(path):
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]

    rows = []
    for leg in ("examples", "tests"):
        rows.extend(load(CENSUS / f"{leg}.jsonl"))
    replacements = {(row.get("leg"), row.get("row")): row
                    for row in load(CENSUS / "tests_param_matched.jsonl")}
    rows = [replacements.get((row.get("leg"), row.get("row")), row) for row in rows]
    return [row for row in rows if row.get("measured")]


@pytest.mark.requires_resource("cuda-coverage-census")
def test_the_fold_admission_slot_counts_recompute_from_the_census(xp):
    """Both numbers in the table, re-derived by running the SHIPPED predicate.

    THE REPLAY VALIDATES ITSELF FIRST. Every row the census recorded with a
    refusal OTHER than the retired fold one must come back with that same string,
    and every row it recorded as covered must still be covered — 372 verdicts,
    reproduced through stand-ins built from the record's own ``configuration``
    blocks. Only then is the widening's number read off, so a stand-in that had
    drifted from what the battery measured would fail as a disagreement rather
    than quietly move the count.

    IT IS AN UPPER BOUND AND THE TABLE SAYS SO. The stand-ins answer the array
    and coefficient-vector questions by construction, because the record does not
    carry them. What is measured here is exactly the decision logic above that
    tail — which is where the fold clause lives.
    """
    if not CENSUS.is_dir():
        pytest.skip("[requires_resource][cuda-coverage-census] "
                    "the 2026-08-16 predicate-coverage record is not on this host")
    rows = _census_rows()
    assert len(rows) == 186, (
        f"the census record holds {len(rows)} measured rows, not the 186 the "
        f"slot counts were derived over")

    reproduced, admitted = 0, {"H": 0, "E": 0}
    gained = []
    for row in rows:
        configuration = row["configuration"]
        triple = (_CensusFields(configuration, xp), _CensusPML(configuration, xp),
                  _CensusGrid(configuration, xp))
        for sub_step, side in (("update_H", "H"), ("update_E", "E")):
            recorded = row["cuda_constitutive"][sub_step]
            covered, reason = coverage.covers_real_pml_constitutive(*triple, side)
            if covered:
                admitted[side] += 1
                if not recorded["covered_modulo_backend"]:
                    gained.append((row["row"], sub_step, recorded["first_refusal"],
                                   configuration))
            if recorded["first_refusal"] in _RETIRED_REFUSALS:
                continue  # the rows the widenings act on
            reproduced += 1
            assert covered == recorded["covered_modulo_backend"], (
                f"{row['row']}/{sub_step}: the census recorded "
                f"covered={recorded['covered_modulo_backend']} and the shipped "
                f"predicate now says {covered} ({reason})")
            if not covered:
                assert reason == recorded["first_refusal"], (
                    f"{row['row']}/{sub_step}: refusal moved from "
                    f"{recorded['first_refusal']!r} to {reason!r}")
    assert reproduced >= 200, (
        f"only {reproduced} verdicts were reproducible, so the validation leg is "
        f"too thin to license the count")

    # EVERY SLOT GAINED MUST HAVE BEEN REFUSED BY THE FOLD CLAUSE AND NOTHING
    # ELSE. This is what separates a widening from a leak: a slot that used to be
    # refused for dispersion or complex storage and is now admitted would mean
    # some other clause stopped firing, and the totals alone cannot see that.
    leaked = [(name, sub_step, first) for name, sub_step, first, _ in gained
              if first not in _RETIRED_REFUSALS]
    assert not leaked, (
        f"{len(leaked)} slots were gained for a reason that is NOT the retired "
        f"fold refusal, so the widening reaches further than the fold: {leaked[:5]}")
    # WHAT A GAINED ROW MAY BE. Two widenings have landed, so a gained row must be
    # PML-active real storage that is EITHER folded (2026-08-19) OR cylindrical
    # (2026-08-20) -- and nothing else. Complex storage is not admitted by either:
    # the complex pair is a different kernel and the Dcyl gate did not score it.
    unfolded = [(name, sub_step) for name, sub_step, _, configuration in gained
                if not (configuration["pml_active"]
                        and not configuration["force_complex_fields"]
                        and (configuration["has_symmetry"]
                             or configuration["cylindrical"]))]
    assert not unfolded, (
        f"slots were gained on rows that are neither folded nor cylindrical, "
        f"PML-active, real-storage runs: {unfolded[:5]}")
    # THE PLANE CAP APPLIES TO THE FOLDED GAINS ONLY. A Dcyl row is not folded,
    # so its plane count is 0 and including it would let an empty set of folded
    # gains pass this check by dilution.
    planes = {sum(configuration["mirrored"])
              for _, _, _, configuration in gained
              if configuration["has_symmetry"]}
    assert planes and max(planes) <= coverage.CONSTITUTIVE_FOLD_ADMISSION[
        "folded_planes_swept"], (
        f"a slot was gained on a row folded across {max(planes)} planes and the "
        f"gate swept fewer")
    # AND THE CAP IS NOT DECORATION: the corpus really does drive a row at the
    # count the cap names, so a cap raised without a gate leg would be visible
    # here as an admitted row rather than only as a changed constant.
    assert max(planes) == coverage.CONSTITUTIVE_FOLD_ADMISSION[
        "folded_planes_swept"], (
        f"the corpus's deepest gained fold is {max(planes)} planes and the cap is "
        f"{coverage.CONSTITUTIVE_FOLD_ADMISSION['folded_planes_swept']}; the cap "
        f"is wider than anything it was priced against")

    # TWO WIDENINGS, ONE ARITHMETIC. The fold table's slots_after is the Dcyl
    # table's slots_before, so the chain has to close end to end: a number that
    # only checked the latest link would let an earlier one drift unnoticed.
    fold = coverage.CONSTITUTIVE_FOLD_ADMISSION
    dcyl = coverage.CONSTITUTIVE_CYLINDRICAL_ADMISSION
    assert dcyl["slots_before"] == fold["slots_after"], (
        f"the Dcyl admission says it started from {dcyl['slots_before']} slots "
        f"and the fold admission says it left {fold['slots_after']}; the two "
        f"widenings no longer compose")

    # EACH GAIN ATTRIBUTED TO THE WIDENING THAT CAUSED IT, not to the total. A
    # fold gain and a Dcyl gain are different measurements behind different
    # gates, and a count that pooled them could not tell one shrinking from the
    # other growing.
    #
    # THREE WIDENINGS NOW, and the third is split OUT OF the fold count rather
    # than folded into it: the 2026-08-19 gate swept at most two simultaneous
    # planes, so the rows it licensed are the folded rows with one or two, and the
    # 2026-08-20 re-cut is what licensed the third. Pooling them would let a
    # three-plane row be counted as evidence for the earlier run.
    by_widening = {"fold": 0, "third_plane": 0, "cylindrical": 0}
    for _, _, _, configuration in gained:
        if not configuration["has_symmetry"]:
            by_widening["cylindrical"] += 1
        elif sum(configuration["mirrored"]) > 2:
            by_widening["third_plane"] += 1
        else:
            by_widening["fold"] += 1
    assert by_widening["fold"] == fold["slots_after"] - fold["slots_before"], (
        f"{by_widening['fold']} slots were gained on rows folded across at most "
        f"two planes and CONSTITUTIVE_FOLD_ADMISSION describes "
        f"{fold['slots_after'] - fold['slots_before']}")
    assert by_widening["cylindrical"] == dcyl["slots_after"] - dcyl["slots_before"], (
        f"{by_widening['cylindrical']} slots were gained on Dcyl rows and "
        f"CONSTITUTIVE_CYLINDRICAL_ADMISSION describes "
        f"{dcyl['slots_after'] - dcyl['slots_before']}")
    assert by_widening["third_plane"] == \
        fold["planes_round_slots_after"] - fold["planes_round_slots_before"], (
        f"{by_widening['third_plane']} slots were gained on rows folded across "
        f"three planes and the plane round describes "
        f"{fold['planes_round_slots_after'] - fold['planes_round_slots_before']}")
    # THE CHAIN CLOSES END TO END: the plane round starts where the Dcyl one
    # stopped, so a number that only checked the latest link cannot let an earlier
    # one drift.
    assert fold["planes_round_slots_before"] == dcyl["slots_after"], (
        f"the plane round says it started from "
        f"{fold['planes_round_slots_before']} slots and the Dcyl admission says "
        f"it left {dcyl['slots_after']}; the widenings no longer compose")

    assert admitted["H"] + admitted["E"] == fold["planes_round_slots_after"], (
        f"the shipped predicate admits {admitted} = "
        f"{admitted['H'] + admitted['E']} constitutive slots over the census, and "
        f"the latest admission says {fold['planes_round_slots_after']}")
    before = sum(1 for row in rows for sub_step in ("update_H", "update_E")
                 if row["cuda_constitutive"][sub_step]["covered_modulo_backend"])
    assert before == fold["slots_before"], (
        f"the census recorded {before} constitutive slots and the table says "
        f"{fold['slots_before']}; the BEFORE number is a transcription of the "
        f"record and must match it")


@pytest.mark.requires_resource("cuda-coverage-census")
def test_the_third_fold_plane_bought_exactly_what_the_table_says(xp):
    """The raise's price, counted rather than asserted -- and now counted BACKWARDS.

    Until 2026-08-20 the cap was 2 and this test lifted it to 3 on a copy of the
    predicate to price what a further gate leg WOULD buy. The leg was run, the cap
    is 3, and the same question is now asked the other way: put 2 back and count
    what the shipped predicate would LOSE. The number is the same number and it is
    the one the table carries as ``slots_refused_for_a_third_fold_plane``, which
    is 0 now that nothing is refused for it.
    """
    if not CENSUS.is_dir():
        pytest.skip("[requires_resource][cuda-coverage-census] "
                    "the 2026-08-16 predicate-coverage record is not on this host")
    source = textwrap.dedent(
        inspect.getsource(coverage.covers_real_pml_constitutive))
    recapped = _predicate_from(source.replace(
        'CONSTITUTIVE_FOLD_ADMISSION["folded_planes_swept"]', "2", 1))

    lost, rows_lost = 0, set()
    for row in _census_rows():
        configuration = row["configuration"]
        triple = (_CensusFields(configuration, xp), _CensusPML(configuration, xp),
                  _CensusGrid(configuration, xp))
        for side in SIDES:
            if coverage.covers_real_pml_constitutive(*triple, side)[0] and not \
                    recapped(*triple, side)[0]:
                lost += 1
                rows_lost.add(row["row"])
    table = coverage.CONSTITUTIVE_FOLD_ADMISSION
    # THE RAISE IS WORTH WHAT THE OLD CAP COST, and the row is the same one.
    assert lost == 2, (
        f"lowering the cap to 2 loses {lost} slots; the 2026-08-19 record priced "
        f"the third plane at 2 and the two must agree or one of them is stale")
    assert rows_lost == {table["row_with_three_fold_planes"]}, rows_lost
    # AND NOTHING IS REFUSED FOR IT ANY MORE, which is the table's own number.
    assert table["slots_refused_for_a_third_fold_plane"] == 0
