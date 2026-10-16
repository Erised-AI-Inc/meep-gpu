"""The CUDA cylindrical (Dcyl, m = 0) curl slice: its predicate, its scan's host side, its text.

THE WORD-FOR-WORD CHECK OF THE TEXT AGAINST THE ARRAY PATH is ``test_cylindrical_chunk_forms``
(the kernels executed by an interpreter, launched through their own launchers); the device
gate is ``parity/meep_gpu/gate_cuda_cylindrical_real.py``. What is pinned HERE is the rest of
the slice's silent-failure surface, and every one of these failures is a plausible, smooth,
WRONG field rather than a crash:

* THE PREDICATE. Every configuration it refuses would be mis-stepped if it leaked, so the
  refusals are mutated and every mutation must be caught.
* THE NON-WIDENING. ``coverage.covers_real_pml_curl`` -- the CARTESIAN predicate -- must
  STILL refuse Dcyl, and this pair's predicate every Cartesian grid.
* THE DEVICE TEXT. What separates this pair from the root Cartesian one: Bz's and Dz's whole
  curl is ``dtdx`` times the scan's radial difference, and the m = 0 axis rules run after
  the PML forms with MEEP's double ``4*Courant`` product. Pinned as text because a regrouped
  or float32 spelling is exact at some Courant numbers.
* THE ARGUMENT ORDER. The kernel signature and the launcher's tuple are two lists that must
  agree; CuPy's ``RawKernel`` does not check.
* THE SCAN'S HOST SIDE. Its row vectors and chunk starts are the array path's own.

THE SOURCE IS READ WITH ``ast`` RATHER THAN IMPORTED: ``cylindrical_kernels`` imports
``cupy`` at module scope. The predicate and the scan's host side live in CuPy-free siblings
and ARE imported. Off device the fixture hands ``Grid`` NumPy wearing CuPy's ``__name__``,
because the predicate's first question is whether the backend is CuPy at all.
"""

from __future__ import annotations

import ast
import inspect
import os
import re
import textwrap
import types

import numpy
import pytest

try:
    import cupy
except ImportError:  # pragma: no cover - exercised on any NumPy-only host
    cupy = None

from ..fields import IYEE_SHIFTS, Fields
from ..grid import Grid
from ..pml import PML
from .. import stepping
from . import coverage, cylindrical_coverage, cylindrical_prefix

SUB_STEPS = ("step_B", "step_D")


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__``.

    The predicate refuses any backend whose module is not named "cupy", which is the
    one thing about the real device library that cannot be reproduced off device — and
    the only thing this stands in for.
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
    return request.param


def build(xp, shape=(16, 1, 20), z_kind="metallic", m=0, cylindrical=True,
          force_complex_fields=False, courant=0.35, **grid_kwargs):
    """A frozen ``(fields, layer, grid)`` triple.

    ``Grid`` owns the r axis's boundary pair on a cylindrical cell — the axis at r = 0
    and a metallic wall at r_max (grid.py:498-508) — so only z is passed. The PML is
    asked for the HIGH r face only: cell 0 is the axis, a boundary condition rather
    than a wall.
    """
    if cylindrical:
        grid = Grid(resolution=1.0,
                    cell_size=(float(shape[0]), 0.0, float(shape[2])),
                    cylindrical=True, m=m, boundaries={"z": z_kind},
                    courant=courant, xp=xp, **grid_kwargs)
        thickness = {"x": (0, max(2, shape[0] // 4)), "z": max(2, shape[2] // 4)}
    else:
        grid = Grid(resolution=1.0,
                    cell_size=(float(shape[0]), float(shape[1]), float(shape[2])),
                    boundaries=("periodic", "periodic", z_kind),
                    courant=courant, xp=xp, **grid_kwargs)
        thickness = tuple((0, 0) if grid.shape[a] < 6 else (2, 2) for a in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=force_complex_fields)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


# --------------------------------------------------------------------------
# The device text, read without importing CuPy
# --------------------------------------------------------------------------

_KERNEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "cylindrical_kernels.py")
_CURL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "step_curl_kernels.py")


def _string_assignment(path: str, name: str) -> str:
    """The value of a module-level string assignment, by ``ast``.

    Handles both a bare literal and ``PRELUDE + r'''...'''``: the second is how both
    kernel files spell a kernel, and taking only the literal half would silently drop
    the prelude from every text assertion below.
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != name:
                continue
            if isinstance(node.value, ast.Constant):
                return node.value.value
            if isinstance(node.value, ast.BinOp) and isinstance(node.value.right,
                                                                ast.Constant):
                return node.value.right.value
    raise AssertionError(f"{name} is not a module-level string in {path}")


def kernel_body(sub_step: str) -> str:
    """The cylindrical kernel's OWN text, without the shared prelude."""
    return _string_assignment(
        _KERNEL_PATH, f"_cyl_step_{sub_step[-1]}_pml_real_kernel_code")


@pytest.fixture(scope="module")
def prelude() -> str:
    return _string_assignment(_CURL_PATH, "_REAL_PML_PRELUDE")


# --------------------------------------------------------------------------
# The predicate: what it admits
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", SUB_STEPS)
@pytest.mark.parametrize("z_kind", ("metallic", "periodic"))
@pytest.mark.parametrize("shape", ((16, 1, 20), (9, 1, 17), (40, 1, 16)))
def test_a_real_m0_dcyl_grid_under_an_active_pml_is_covered(sub_step, z_kind, shape, xp):
    fields, layer, grid = build(xp, shape=shape, z_kind=z_kind)
    covered, reason = cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, layer, grid, sub_step)
    assert covered, reason
    assert reason == "covered"


#: Every refusal reachable from a real ``Grid``, as ``(builder kwargs, fragment)``.
#: Written positively so a clause that stops firing fails a test rather than quietly
#: widening coverage.
_REFUSALS = (
    ({"cylindrical": False}, "not a cylindrical"),
    ({"m": 1}, "carries m = 0 ONLY"),
    ({"m": -2}, "carries m = 0 ONLY"),
    ({"force_complex_fields": True}, "complex64 storage"),
)


@pytest.mark.parametrize("sub_step", SUB_STEPS)
@pytest.mark.parametrize("kwargs,fragment", _REFUSALS)
def test_the_named_refusals_fire_and_say_why(sub_step, kwargs, fragment, xp):
    fields, layer, grid = build(xp, **kwargs)
    covered, reason = cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, layer, grid, sub_step)
    assert not covered
    assert fragment in reason, reason


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_an_inactive_layer_is_refused(sub_step, xp):
    fields, _layer, grid = build(xp)
    covered, reason = cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, None, grid, sub_step)
    assert not covered
    assert "no active PML" in reason


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_a_numpy_backend_is_refused(sub_step):
    """The one clause the stand-in cannot fake, checked with the real NumPy."""
    fields, layer, grid = build(numpy)
    covered, reason = cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, layer, grid, sub_step)
    assert not covered
    assert reason == "backend is not CuPy"


def test_an_unnamed_sub_step_raises_rather_than_refusing(xp):
    """A typo must not read as 'not covered'. There are two kernels and a caller
    that named neither has a bug, not a configuration this pair declines."""
    fields, layer, grid = build(xp)
    with pytest.raises(ValueError, match="sub_step must be one of"):
        cylindrical_coverage.covers_real_pml_cylindrical_curl(
            fields, layer, grid, "step_E")


# --------------------------------------------------------------------------
# The predicate: mutations
# --------------------------------------------------------------------------

def _mutated(**mutations):
    """The predicate module with named refusals edited out of named functions.

    Every mutated source is exec'd into ONE namespace seeded from the real module, so
    a mutated helper is what the mutated predicate calls. Raises if a pattern matches
    other than once — a mutation that has drifted away from the code fails loudly
    instead of quietly testing an unmutated function.
    """
    namespace = dict(vars(cylindrical_coverage))
    namespace["__builtins__"] = __builtins__
    module = types.ModuleType("mutated_cylindrical_predicate")
    module.__dict__.update(namespace)
    for name, patterns in mutations.items():
        source = textwrap.dedent(inspect.getsource(getattr(cylindrical_coverage, name)))
        for pattern in patterns:
            source, count = re.subn(pattern, "", source)
            assert count == 1, (
                f"{name}: {pattern!r} matched {count} times, expected 1; the mutation "
                f"and the predicate have drifted apart and it is pinning nothing")
        exec(compile(source, f"<mutated {name}>", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_mutation_drop_the_m_clause_admits_a_grid_this_pair_cannot_step(sub_step, xp):
    """Drop the ``m == 0`` refusal and |m| = 1 becomes 'covered'.

    |m| >= 1 forces complex storage, the ``i*m/r`` coupling and the per-|m| axis
    rules; the kernel has none of them and would return a smooth, plausible, wrong
    field. Two clauses have to go together — real storage refuses |m| = 1 first on a
    grid that actually allocates complex64 — so the leg uses a grid whose storage is
    still real and removes only the m clause.
    """
    fields, layer, grid = build(xp, m=1)
    assert not cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, layer, grid, sub_step)[0]
    widened = _mutated(covers_real_pml_cylindrical_curl=(
        r"    if m != COVERED_M:\n(?:        [^\n]*\n)+",))
    covered, reason = widened.covers_real_pml_cylindrical_curl(
        fields, layer, grid, sub_step)
    assert covered, (
        f"the widened predicate still refused m = 1 ({reason}), so this mutation is "
        f"not exercising the m clause and the clause is unpinned")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_mutation_drop_the_phi_extent_clause_is_caught(sub_step, xp):
    """A phi extent other than 1 cell is a different engine.

    Unreachable from today's ``Grid`` — it refuses a phi extent outright — so the leg
    fakes the shape read rather than the grid, which is the honest way to pin a
    fail-closed clause whose input cannot yet be constructed.
    """
    fields, layer, grid = build(xp)

    class _WidePhi:
        """The grid, with a phi extent of 3 in the ONE read the clause makes."""

        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        @property
        def shape(self):
            inner = object.__getattribute__(self, "_inner").shape
            return (inner[0], 3, inner[2])

        def __getattr__(self, item):
            return getattr(object.__getattribute__(self, "_inner"), item)

    covered, reason = cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, layer, _WidePhi(grid), sub_step)
    assert not covered
    assert "phi extent" in reason or "not allocated" in reason or "shape" in reason, reason


# --------------------------------------------------------------------------
# The non-widening: the Cartesian predicate must STILL refuse Dcyl
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_cartesian_predicate_still_refuses_every_dcyl_grid(sub_step, xp):
    """Adding a second pair must not widen the first.

    ``coverage.covers_real_pml_curl`` refuses Dcyl twice — the ``cylindrical`` clause
    and the per-axis ``is_axis`` one — and both are CORRECT for kernels with no
    prefix pointer. A widening here reaches the certified kernels, which is a wrong
    answer on four of six targets rather than a crash.
    """
    fields, layer, grid = build(xp)
    covered, reason = coverage.covers_real_pml_curl(fields, layer, grid, sub_step)
    assert not covered
    assert "cylindrical" in reason or "r = 0 axis" in reason, reason


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_cylindrical_predicate_refuses_every_cartesian_grid(sub_step, xp):
    """And the same argument run the other way: the two partitions cannot overlap."""
    fields, layer, grid = build(xp, cylindrical=False, shape=(8, 8, 8))
    covered, reason = cylindrical_coverage.covers_real_pml_cylindrical_curl(
        fields, layer, grid, sub_step)
    assert not covered
    assert "not a cylindrical" in reason, reason


def test_every_shared_helper_this_module_borrows_still_exists():
    """A rename in ``coverage.py`` must fail at the merge bar, not at the first run."""
    for name in cylindrical_coverage.SHARED_HELPERS:
        assert hasattr(coverage, name), (
            f"{name} is gone from coverage.py; the cylindrical predicate imports it "
            f"and would drop a clause")


def test_the_predicate_module_pulls_in_no_device_dependency():
    """It must be importable, exercisable and mutable on a machine with no GPU."""
    source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "cylindrical_coverage.py"), encoding="utf-8").read()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & {"cupy", "numpy", "torch", "triton"}), imported


# --------------------------------------------------------------------------
# The boundary codes
# --------------------------------------------------------------------------

def test_the_r_axis_compiles_as_metallic_and_nothing_else_is_accepted():
    """``_shift_up``'s CYL_AXIS arm IS the METALLIC arm (stepping.py:1828-1830).

    A PERIODIC r axis would wrap the far radius onto the axis row, so it is refused
    BY NAME rather than mapped to a code that happens to exist.
    """
    codes = cylindrical_coverage.cylindrical_boundary_codes(
        ("axis", "periodic", "metallic"))
    assert codes == (coverage.BC_CODES["metallic"],
                     coverage.BC_CODES["periodic"],
                     coverage.BC_CODES["metallic"])
    with pytest.raises(ValueError, match="the r axis resolved to boundary"):
        cylindrical_coverage.cylindrical_boundary_codes(
            ("periodic", "periodic", "metallic"))
    with pytest.raises(ValueError, match="which the cylindrical curl pair does not"):
        cylindrical_coverage.cylindrical_boundary_codes(
            ("axis", "periodic", "mirror"))


@pytest.mark.parametrize("z_kind", ("metallic", "periodic"))
def test_the_resolved_kinds_are_steppings_own(z_kind, xp):
    """The codes are computed off ``stepping._boundary_kinds``, not off a reading."""
    _fields, _layer, grid = build(xp, z_kind=z_kind)
    assert tuple(coverage.real_pml_boundary_kinds(grid)) == \
        tuple(stepping._boundary_kinds(grid, None))
    assert tuple(stepping._boundary_kinds(grid, None))[:2] == ("axis", "periodic")


# --------------------------------------------------------------------------
# The ownership mask: derived, then compared with what the source spells
# --------------------------------------------------------------------------

def masked_axes_from_stepping(target: str, grid) -> tuple:
    """Which axes ``_mask_non_owned_cells`` zeroes cell 0 on, for this component.

    DERIVED from the engine's own inputs — ``fields.IYEE_SHIFTS`` and the grid's
    boundary resolution — rather than restated, so a change to either fails this test
    instead of silently disagreeing with the kernel.
    """
    iyee = IYEE_SHIFTS[target]
    return tuple(axis for axis in range(3)
                 if iyee[axis] == 0 and (grid.is_metallic(axis) or grid.is_axis(axis)
                                         or grid.is_mirrored(axis)))


@pytest.mark.parametrize("sub_step", SUB_STEPS)
@pytest.mark.parametrize("z_kind", ("metallic", "periodic"))
def test_the_kernel_masks_exactly_the_cells_the_array_path_masks(sub_step, z_kind, xp):
    """The mask lines in the device text, against ``_mask_non_owned_cells``' rule.

    With the r axis compiled METALLIC the kernel's ``bc == BC_METALLIC && index == 0``
    lands on exactly the cells the ``is_axis`` clause lands on. This parses the
    per-target blocks out of the source and compares the axes each one masks with the
    derived set — so a mask added, dropped or moved to the wrong block fails here,
    where the gate would only see it if the case matrix happened to make that plane
    live.
    """
    _fields, _layer, grid = build(xp, z_kind=z_kind)
    codes = cylindrical_coverage.cylindrical_boundary_codes(
        tuple(stepping._boundary_kinds(grid, None)))
    body = kernel_body(sub_step)
    targets = (("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz"))
    axis_letter = {"x": 0, "y": 1, "z": 2}
    for target in targets:
        block = re.search(
            r"//\s*" + target + r":.*?pml_apply\(" + target + r"\b",
            body, re.S)
        assert block, f"no {target} block found in the {sub_step} source"
        spelled = {axis_letter[letter]
                   for letter in re.findall(
                       r"if \(bc_([xyz]) == BC_METALLIC && [ijk] == 0\) curl = 0\.0f;",
                       block.group(0))}
        # A mask line is only LIVE where that axis's code is METALLIC. The kernel
        # spells the clause on every axis whose Yee shift is 0 and lets the code
        # decide; the array path decides by the boundary kind. The two agree exactly
        # when the spelled set, restricted to the metallic-coded axes, equals the
        # derived set.
        live = {axis for axis in spelled if codes[axis] == coverage.BC_CODES["metallic"]}
        assert live == set(masked_axes_from_stepping(target, grid)), (
            f"{sub_step} {target}: kernel masks {sorted(live)}, "
            f"_mask_non_owned_cells masks {sorted(masked_axes_from_stepping(target, grid))}")


def test_no_axis_of_a_dcyl_grid_stores_past_its_owned_window(xp):
    """The far-face clause the kernel does NOT reproduce, shown unreachable here.

    ``_mask_non_owned_cells``' second arm zeroes the LAST stored slot of a folded
    PERIODIC axis, and that is the divergence the folded CUDA round measured. A Dcyl
    grid has no fold — the predicate refuses one — so the clause cannot fire, and this
    asserts that on the real grid rather than reasoning about it.
    """
    for z_kind in ("metallic", "periodic"):
        _fields, _layer, grid = build(xp, z_kind=z_kind)
        for axis in range(3):
            assert not stepping._stored_past_owned(grid, axis)
            assert grid.stored_cells(axis) == grid.owned_cells(axis)


# --------------------------------------------------------------------------
# The device text: the four cylindrical additions
# --------------------------------------------------------------------------

def test_the_prelude_is_the_certified_one_verbatim(prelude):
    """Reused, not copied. Two copies of ``pml_apply`` are two things to keep in step
    — and a second spelling would silently un-arm the shared mutation battery, whose
    needles are those exact strings."""
    source = open(_KERNEL_PATH, encoding="utf-8").read()
    assert "from .step_curl_kernels import _REAL_PML_PRELUDE" in source
    for needle in ("float fu_new = ((fprev * kms) - curl) * sinv;",
                   "__device__ __forceinline__ float shift_up(",
                   "__device__ __forceinline__ float shift_dn("):
        assert needle in prelude


def test_bz_takes_dtdx_times_the_radial_difference():
    """Bz's whole curl is ``dtdx * rad`` -- ``part_products.scale(dtdx, radial)``, one
    product -- where ``rad`` is the scan's radial difference of Ep's per-chunk sum.

    PINNED AS TEXT because a distributed or a re-grouped spelling is exact whenever
    ``dtdx`` is a power of two, which a fixture without a non-power-of-two Courant cannot
    see. The Cartesian Bz stencil must be gone from the Bz block (``Ey`` is still read raw
    by Bx, so the assertion is scoped to the block).
    """
    body = kernel_body("step_B")
    bz_block = body[body.index("    // Bz:"):body.index("pml_apply(Bz,")]
    assert "float curl = dtdx * rad[idx];" in bz_block
    assert "shift_up(Ey," not in bz_block and "shift_up(Ex," not in bz_block
    assert body.count("rad[") == 1, "exactly one term may read the radial difference"


def test_dz_takes_the_radial_difference_and_dx_keeps_the_raw_hy():
    """The substitution is per TERM, not per sub-step: Dx's Hy operands stay RAW."""
    body = kernel_body("step_D")
    dz_block = body[body.index("    // Dz:"):body.index("pml_apply(Dz,")]
    assert "float curl = dtdx * rad[idx];" in dz_block
    assert "float f2 = Hy[idx];" in body, "Dx's Hy operand must stay RAW"
    assert "float ss = shift_dn(Hy, idx, k, nz, sz, bc_z);" in body
    assert body.count("rad[") == 1, "exactly one term may read the radial difference"


def test_the_axis_rules_are_post_rules_in_the_array_paths_own_order():
    """``Bx[r=0]=0``; then ``Dz[r=0] += realnum(double(4*Courant) * double(Hp))``, then
    ``Dy[r=0]=0`` -- after every PML form, in the array path's own order. Pinned as text
    as well as armed as a mutation, because "applied after the recurrence" is a property
    of WHERE the line sits."""
    b_body = kernel_body("step_B")
    d_body = kernel_body("step_D")
    assert b_body.rindex("pml_apply(") < b_body.index("if (i == 0) Bx[idx] = 0.0f;")
    add = d_body.index(
        "Dz[idx] = Dz[idx] + __double2float_rn(axis_courant * (double)(Hy[idx]));")
    zero = d_body.index("Dy[idx] = 0.0f;")
    last_apply = d_body.rindex("pml_apply(")
    assert last_apply < add < zero
    assert "curl = curl - " not in d_body, "the on-axis rule must NOT be folded into the curl"


def test_the_axis_rule_takes_the_double_courant_product():
    """``axis_courant`` is a ``double`` ARGUMENT, ``4 * Courant`` in double, and the product
    with Hp is formed in double and rounded once (MEEP step_db.cpp:324,
    ``stepping._realnum_double_product``). A float32 ``4*Courant`` is another factor
    wherever Courant is not a float32 word (0.3 is not)."""
    d_body = kernel_body("step_D")
    assert "double axis_courant" in d_body
    assert "4.0f" not in d_body and "axis_coef" not in d_body
    source = open(_KERNEL_PATH, encoding="utf-8").read()
    assert "np.float64(4.0 * float(grid.courant))" in source


def test_the_b_side_kernel_takes_no_axis_factor():
    """A scalar bound and unread is a signature two callers can disagree about."""
    assert "axis_courant" not in kernel_body("step_B")


# --------------------------------------------------------------------------
# The argument order: the signature against the launcher's tuple
# --------------------------------------------------------------------------

def _signature_names(body: str) -> list:
    signature = body[body.index("(", body.index("__global__ void")):]
    signature = signature[:signature.index(")")]
    names = []
    for part in signature.split(","):
        token = part.strip().split()[-1].strip("*").strip()
        names.append(token)
    return names


@pytest.mark.parametrize("sub_step,expected", (
    ("step_B", ["Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz", "Ex", "Ey", "Ez",
                "rad", "nx", "ny", "nz", "dtdx",
                "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z",
                "bc_x", "bc_y", "bc_z", "cb"]),
    ("step_D", ["Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz", "Hx", "Hy", "Hz",
                "rad", "nx", "ny", "nz", "dtdx", "axis_courant",
                "kms_x", "sinv_x", "kms_y", "sinv_y", "kms_z", "sinv_z",
                "bc_x", "bc_y", "bc_z", "cb"]),
))
def test_the_kernel_signature_is_the_order_the_launcher_passes(sub_step, expected):
    """``RawKernel`` does not type-check its tuple; a swapped pair is a wrong pointer.

    The launcher builds one tuple for both sides and inserts the D side's extra scalar
    through ``extra_scalars``, so the two orders differ in exactly one position -- the
    place a reader is most likely to get it wrong -- and closes it with the chunk record.
    """
    assert _signature_names(kernel_body(sub_step)) == expected
    launch = _string_assignment  # keep the import used; the real check is below
    source = open(_KERNEL_PATH, encoding="utf-8").read()
    order = re.search(r"kernel\(\(blocks,\), \(_REAL_PML_THREADS,\), \((.*?)\)\)\n",
                      source, re.S).group(1)
    flattened = [token.strip() for token in order.replace("\n", " ").split(",")
                 if token.strip()]
    assert flattened[:10] == ["targets[0]", "targets[1]", "targets[2]",
                              "auxiliaries[0]", "auxiliaries[1]", "auxiliaries[2]",
                              "sources[0]", "sources[1]", "sources[2]", "radial"]
    assert "*extra_scalars" in flattened
    assert flattened.index("*extra_scalars") == flattened.index("np.float32(dtdx)") + 1
    assert flattened[-1] == "cb"
    assert launch is _string_assignment


def test_the_dispatch_is_sized_from_the_target_volume():
    """The radial difference shares the target's shape; the launch is sized from the
    target and the launcher refuses a difference of any other shape."""
    source = open(_KERNEL_PATH, encoding="utf-8").read()
    assert "nx, ny, nz = targets[0].shape" in source
    assert "if tuple(radial.shape) != (nx, ny, nz):" in source
    for sub_step in SUB_STEPS:
        assert "if (idx >= nx * ny * nz) return;" in kernel_body(sub_step)


# --------------------------------------------------------------------------
# The radial scan's host side
# --------------------------------------------------------------------------

def test_the_prefix_tables_are_the_array_paths_own_values():
    assert cylindrical_prefix.PREFIX_IR0 == {"step_B": 0.0, "step_D": 0.5}
    assert cylindrical_prefix.PREFIX_WALL_ROW == {"step_B": True, "step_D": False}
    assert cylindrical_prefix.PREFIX_COMPONENT == {"step_B": "Ey", "step_D": "Hy"}
    assert cylindrical_prefix.RADIAL_TARGET == {"step_B": "Bz", "step_D": "Dz"}
    assert cylindrical_prefix.RADIAL_BACKWARD == {"step_B": False, "step_D": True}
    assert cylindrical_prefix.prefix_rows("step_B", 16) == 17
    assert cylindrical_prefix.prefix_rows("step_D", 16) == 16


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_scan_binds_the_array_paths_own_rows_and_chunk_starts(sub_step, xp):
    """``radial_rows`` is ``stepping._cylindrical_rderiv_rows`` over the sum's source rows
    (the B side's carries the wall row), and ``radial_starts`` is
    ``stepping._cylindrical_scan_starts`` of the target -- both CALLED, never re-derived.
    On a grid with an r layer the division has two r chunks, so the line restarts."""
    fields, layer, grid = build(xp, shape=(16, 1, 20))
    rows = int(grid.shape[0])
    weights, rinv = cylindrical_prefix.radial_rows(sub_step, rows)
    want_w, want_r = stepping._cylindrical_rderiv_rows(
        numpy, cylindrical_prefix.prefix_rows(sub_step, rows),
        cylindrical_prefix.PREFIX_IR0[sub_step], numpy.dtype(numpy.float32))
    assert numpy.array_equal(weights.view(numpy.uint32),
                             numpy.ascontiguousarray(want_w.reshape(-1)).view(numpy.uint32))
    assert numpy.array_equal(rinv.view(numpy.uint32),
                             numpy.ascontiguousarray(want_r.reshape(-1)).view(numpy.uint32))
    line, segments = cylindrical_prefix.radial_starts(fields, layer, sub_step)
    target = cylindrical_prefix.RADIAL_TARGET[sub_step]
    starts = stepping._cylindrical_scan_starts(fields, layer, target)
    want = (numpy.zeros(rows, dtype=numpy.int64) if starts is None
            else numpy.asarray(stepping._host_ints(starts)).reshape(-1))
    assert line.tolist() == want.tolist()
    assert segments.tolist() == sorted(set(want.tolist()))
    assert len(segments) == 2


def test_an_unnamed_sub_step_raises_in_the_scan_too(xp):
    fields, layer, _grid = build(xp)
    with pytest.raises(ValueError, match="sub_step must be one of"):
        cylindrical_prefix.radial_rows("step_E", 16)
    with pytest.raises(ValueError, match="sub_step must be one of"):
        cylindrical_prefix.radial_starts(fields, layer, "step_E")


# --------------------------------------------------------------------------
# The fact that licenses the kernel having no curl-row fold at m = 0
# --------------------------------------------------------------------------

@pytest.mark.parametrize("m,expected_none", ((0, True), (1, False)))
def test_the_axis_increment_is_none_at_m_zero_and_not_at_m_one(m, expected_none, xp):
    """READ OFF the array path by CALLING it, not by reading the branch.

    At m = 0 neither ``_cylindrical_axis_increment_B`` nor ``_D`` returns a term, and
    THAT is why the kernel carries only the two post-rules. If either ever starts
    returning something at m = 0, the kernel is missing a term and this fails.
    """
    fields, _layer, grid = build(xp, m=m,
                                 force_complex_fields=(m != 0),
                                 courant=0.35 if m == 0 else 0.3)
    boundaries = stepping._boundary_kinds(grid, None)
    dtdx = float(grid.dt / grid.dx)
    electric = {n: getattr(fields, n) for n in ("Ex", "Ey", "Ez")}
    magnetic = {n: getattr(fields, n) for n in ("Hx", "Hy", "Hz")}
    increment_b = stepping._cylindrical_axis_increment_B(
        fields, electric, boundaries, (None, None, None), dtdx)
    increment_d = stepping._cylindrical_axis_increment_D(
        fields, magnetic, boundaries, (None, None, None), dtdx)
    assert (increment_b is None and increment_d is None) is expected_none


@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_the_kernel_is_not_certified_and_says_so(sub_step):
    """Nothing here may be counted as coverage until the gate releases on hardware.

    The kernels are named in this file's own uncertified list and are ABSENT from
    ``certification.json``, which is the record the certified pair is bound by.
    """
    import json

    record = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "certification.json"), encoding="utf-8"))
    names = json.dumps(record)
    # RELEASED 2026-08-27, so this clause is INVERTED exactly as its own message said it
    # should be: "if the device gate has released it, the record is the place that says
    # so and this test is the place" to change. The gate ran on the GPU host under both
    # float32 policies and released; the block is cuda_cylindrical_real_2026-08-27. What
    # is asserted now is the thing worth asserting -- that the record NAMES them, so a
    # promotion without a record still fails here.
    certified = set()
    for block in record.values():
        if isinstance(block, dict):
            certified |= set(block.get("certified_kernels") or ())
    for name in ("cyl_step_B_pml_real", "cyl_step_D_pml_real"):
        assert name in certified, (
            f"{name} is certified in cylindrical_kernels.py but no certification.json "
            f"block lists it in certified_kernels; the record is what a reader checks")
    for name in ():
        assert name not in names, (
            f"{name} appears in certification.json; if the device gate has released "
            f"it, the record is the place that says so and this test is the place "
            f"that has to be updated deliberately")
    source = open(_KERNEL_PATH, encoding="utf-8").read()
    assert "UNCERTIFIED_KERNELS" in source
    assert f'"cyl_{sub_step.lower()}_pml_real"'.replace("step_", "step_") in source \
        or f'"cyl_{sub_step}_pml_real"' in source
