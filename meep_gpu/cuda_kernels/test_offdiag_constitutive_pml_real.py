"""The off-diagonal (tensor epsilon) ``update_E`` slice: predicate, emitter, arithmetic.

THE BYTE GATE LIVES ELSEWHERE — it needs a device — and it has not run for this
family. What is pinned HERE is everything that can be decided without a GPU, which
for this slice is more than for either certified sibling, because the device source
is EMITTED rather than a pair of string constants: the emitter is a stdlib-only
module (``offdiag_emitter.py``), so a test can call it, read what it wrote, and
EVALUATE it.

WHAT "EVALUATE IT" MEANS, precisely, because the strength of every arithmetic leg
below rests on the split:

  * READ AND EXECUTED FROM THE EMITTED TEXT: every per-component statement — the
    three flat-index expressions of each term, which partner volume and which
    coefficient each term takes, the accumulation order, the wall-mask lines, the
    diagonal-first row sum, and which coefficient table and index the tail reads —
    plus ``offdiag_term``'s three arithmetic lines, parsed and evaluated as
    expressions. A ``dj`` where ``dk`` belongs, a mispaired coefficient, a dropped
    mask or a hoisted multiply is therefore caught by running the SHIPPED TEXT,
    not by re-deriving what it ought to say.
  * PINNED BY EXACT STRING and transcribed by hand: the four small helpers
    (``coord_up``, ``coord_dn``, ``flat``, ``ghosted``) and the tail
    ``constitutive_apply``. Those are pinned character-for-character — and
    ``constitutive_apply`` is additionally welded to the CERTIFIED sibling's copy
    of itself, whose expression tree the sibling file already measures against
    ``stepping._apply_constitutive_pml`` over 60 consecutive sub-steps.

The two together cover the whole source: nothing in it is neither executed nor
pinned, and :func:`test_every_line_of_the_device_source_is_executed_or_pinned`
measures that rather than leaving it as a claim.

THE ORACLE IS ``stepping.update_E`` ITSELF, on real ``Grid``/``Fields``/``PML``
objects with rows installed through the PUBLIC installer
(``Fields.set_epsilon_volumes``), so the install-time validation, the zero-row
drop and the stored-E switch are all in force. Comparison is on RAW BYTES, never
``allclose``: -0.0 == 0.0 and NaN != NaN both lie, and this family has a live
signed-zero question.

THE BACKEND STAND-IN, AND WHAT IT IS NOT. Off device the fixture hands ``Grid`` a
module that is NumPy wearing CuPy's ``__name__``, because the predicate's first
question is whether the backend is CuPy at all. Everything else it reads — dtype,
shape, contiguity, base addresses, the PML vectors, the grid's own boundary
resolution — is exercised on REAL objects. On a CUDA host the same tests run again
against real CuPy arrays.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
import re

import numpy
import pytest

try:
    import cupy
except ImportError:  # pragma: no cover - exercised on any NumPy-only host
    cupy = None

from .. import stepping
from ..device_identity import weld_survives_edit
from ..fields import IYEE_SHIFTS, Fields
from ..grid import Grid
from ..pml import PML
from . import coverage, offdiag_emitter

HERE = pathlib.Path(__file__).parent
EMITTER_MODULE = HERE / "offdiag_emitter.py"
KERNEL_MODULE = HERE / "offdiag_constitutive_kernels.py"
CONSTITUTIVE_MODULE = HERE / "constitutive_kernels.py"
CURL_MODULE = HERE / "step_curl_kernels.py"
RECORD = HERE / "certification.json"

KERNEL_DECLARATION = re.compile(r'extern "C" __global__ void (\w+)\(')

#: The two row masks the 186-row corpus drives, measured by the predicate battery
#: (``parity/meep_gpu/results/cuda_predicate_coverage_*/analysis.txt``, the
#: ``offdiag_constitutive_step`` census). Carried as data so a later census that
#: moves them is a visible change rather than a silent one.
CORPUS_ROW_MASKS = ((1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1))

#: Boundary triples the arithmetic legs sweep. Both ghost rules on every axis, and
#: the mixed cases, because the two shifts of one term run on DIFFERENT axes and a
#: rule applied to the wrong one is invisible when all three agree.
BOUNDARY_TRIPLES = (
    ("periodic", "periodic", "periodic"),
    ("metallic", "metallic", "periodic"),
    ("metallic", "periodic", "metallic"),
    ("periodic", "metallic", "metallic"),
    ("metallic", "metallic", "metallic"),
)

#: Consecutive sub-steps every arithmetic leg runs. MULTI-STEP BECAUSE ``f_w`` IS
#: STATE: a tree that gets E right and ``f_w`` wrong is correct for exactly one
#: launch and wrong forever after, and one shot cannot see it. 60 is the budget the
#: certified constitutive pair's record carries.
MULTI_STEP_BUDGET = 60


def module_level_literal(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned at module level")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__`` — the one thing about the real device
    library that cannot be reproduced off device, and the only thing this stands
    in for."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(numpy, item)


_BACKENDS = [pytest.param(_NumpyWearingCupysName(), id="numpy-as-cupy")]
if cupy is not None:  # pragma: no cover - only on a CUDA host
    _BACKENDS.append(pytest.param(cupy, id="cupy"))


@pytest.fixture(params=_BACKENDS)
def xp(request):
    return request.param


def build(xp, boundaries=("periodic", "periodic", "periodic"), symmetry=(),
          force_complex_fields=False, cell=(8.0, 8.0, 8.0), **grid_kwargs):
    """A frozen (fields, pml, grid) triple with PML storage allocated.

    The builder both certified slices use, unchanged: the layer skips an axis too
    thin to hold it, and skips the low face of a mirrored axis where cell 0 is the
    plane rather than a wall.
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


def install_rows(fields, grid, mask, seed=20260816, uniform=False):
    """Install one row mask through the PUBLIC installer, plus a diagonal epsilon.

    The coefficients are SPATIALLY VARYING by default and drawn straddling zero,
    because a uniform coefficient is exactly what makes the registration (rather
    than merely the rounding) invisible: ``u[i]`` and ``u[i+s]`` are then the same
    number and the four-point hoist becomes an algebraic identity. ``uniform=True``
    is the control that shows the difference.
    """
    rng = numpy.random.default_rng(seed)
    xp = grid.xp
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=grid.shape).astype(numpy.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray(
            (numpy.float32(1.0) / values).astype(numpy.float32))
    rows = {}
    for slot, (row, partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if not mask[slot]:
            continue
        values = (numpy.full(grid.shape, 0.3125, dtype=numpy.float32) if uniform
                  else rng.uniform(-0.45, 0.45,
                                   size=grid.shape).astype(numpy.float32))
        rows.setdefault(row, {})[partner] = xp.asarray(values)
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=rows)
    return rows


def seed_state(fields, grid, seed=7):
    """Physical-band values in every array the sub-step reads or writes.

    A GATE THAT CANNOT FAIL CERTIFIES NOTHING, and zero-init is a FIXED POINT of
    this sub-step: with D, E and f_w all zero every tree agrees. Every leg below
    therefore asserts a vacuity floor on the oracle's own output as well.
    """
    rng = numpy.random.default_rng(seed)
    xp = grid.xp
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = xp.asarray(
            rng.uniform(-1.0, 1.0, size=grid.shape).astype(numpy.float32))


# --------------------------------------------------------------------------
# THE EVALUATOR: the emitted device source, executed in float32 NumPy.
# --------------------------------------------------------------------------
#
# A targeted interpreter over a KNOWN GRAMMAR -- the statements this emitter can
# write and no others -- rather than a C parser. Every pattern below is anchored,
# and :func:`test_every_line_of_the_device_source_is_executed_or_pinned` fails if a
# line of an emitted source matches none of them, so the grammar cannot silently
# grow a statement nothing executes.

_TERM_CALL = re.compile(
    r"float term_(?P<tag>\w+) = offdiag_term\(\s*"
    r"(?P<volume>\w+), (?P<coefficient>\w+), idx,\s*"
    r"flat\((?P<down>[^)]*), nyz, nz\),\s*"
    r"flat\((?P<up>[^)]*), nyz, nz\),\s*"
    r"flat\((?P<corner>[^)]*), nyz, nz\)\);")
_TOTAL_FIRST = re.compile(r"float total_(?P<name>\w+) = term_(?P<tag>\w+);")
_TOTAL_ADD = re.compile(
    r"total_(?P<name>\w+) = total_(?P=name) \+ term_(?P<tag>\w+);")
_MASK = re.compile(r"total_(?P<name>\w+) = \((?P<wall>wm_[xyz]) && "
                   r"(?P<face>at_[xyz])\) \? 0\.0f : total_(?P=name);")
_GS = re.compile(r"float gs_(?P<name>\w+) = (?P<volume>\w+)\[idx\];")
_US = re.compile(r"float us_(?P<name>\w+) = (?P<volume>\w+)\[idx\];")
_SRC_COUPLED = re.compile(
    r"float src_(?P<name>\w+) = \(gs_(?P=name) \* us_(?P=name)\)"
    r" \+ total_(?P=name);")
_SRC_PLAIN = re.compile(r"float src_(?P<name>\w+) = gs_(?P=name) \* us_(?P=name);")
_TAIL = re.compile(
    r"constitutive_apply\((?P<field>\w+), (?P<aux>f_w_\w+), idx, src_(?P<name>\w+),"
    r" (?P<kps>kps_[xyz])\[(?P<index>[ijk])\], (?P<kms>kms_[xyz])\[(?P=index)\]\);")
_TERM_BODY = re.compile(
    r"float near_pair = (?P<near>.+);\n"
    r"    float far_pair = (?P<far>.+);\n"
    r"    return (?P<term>.+);")


def _f32(values):
    """Assert float32 at every step: a silent promotion to float64 would make the
    comparison a statement about NumPy's promotion rules, not about the tree."""
    assert values.dtype == numpy.float32, values.dtype
    return values


def _coord_dn(a, n, bc):
    """``coord_dn`` (PINNED by exact string below), transcribed."""
    return numpy.where(a > 0, a - 1, -1 if bc == 1 else n - 1)


def _coord_up(a, n, bc):
    """``coord_up`` (PINNED by exact string below), transcribed."""
    return numpy.where(a + 1 < n, a + 1, -1 if bc == 1 else 0)


def evaluate_emitted_source(source, arrays, tables, codes, walls, shape):
    """Run one launch of the emitted kernel over the whole grid, in float32.

    ``arrays`` maps device parameter names to (nx, ny, nz) float32 arrays and is
    MUTATED in place, exactly as the kernel mutates its outputs.
    """
    nx, ny, nz = shape
    i, j, k = numpy.indices(shape)
    nyz = ny * nz
    home = i * nyz + j * nz + k
    coordinates = {
        "i": i, "j": j, "k": k,
        "di": _coord_dn(i, nx, codes[0]), "ui": _coord_up(i, nx, codes[0]),
        "dj": _coord_dn(j, ny, codes[1]), "uj": _coord_up(j, ny, codes[1]),
        "dk": _coord_dn(k, nz, codes[2]), "uk": _coord_up(k, nz, codes[2])}
    wall_flags = {"wm_x": walls[0], "wm_y": walls[1], "wm_z": walls[2]}
    faces = {"at_x": i == 0, "at_y": j == 0, "at_z": k == 0}

    def flat(expression):
        a, b, c = [coordinates[name.strip()] for name in expression.split(",")]
        return numpy.where((a < 0) | (b < 0) | (c < 0), -1, a * nyz + b * nz + c)

    def ghosted(volume, index):
        """``ghosted`` (PINNED by exact string below), transcribed."""
        flat_volume = numpy.asarray(volume).reshape(-1)
        return _f32(numpy.where(index < 0, numpy.float32(0.0),
                                flat_volume[numpy.maximum(index, 0)]))

    body = source[source.index("int at_x"):]
    scalars = {}
    for match in _GS.finditer(body):
        scalars[f"gs_{match['name']}"] = _f32(numpy.asarray(arrays[match["volume"]]))
    for match in _US.finditer(body):
        scalars[f"us_{match['name']}"] = _f32(numpy.asarray(arrays[match["volume"]]))

    # offdiag_term's three arithmetic lines, PARSED AND EVALUATED as expressions.
    near_text, far_text, term_text = _TERM_BODY.search(source).groups()

    def offdiag_term(volume, coefficient, down, up, corner):
        environment = {"g": volume, "u": coefficient, "home": home,
                       "down": down, "up": up, "corner": corner,
                       "ghosted": ghosted}

        def evaluate(text):
            python = text.replace("0.25f", "numpy.float32(0.25)")
            python = re.sub(r"\b(g|u)\[home\]", r"ghosted(\1, home)", python)
            return _f32(eval(python, {"numpy": numpy}, environment))

        environment["near_pair"] = evaluate(near_text)
        environment["far_pair"] = evaluate(far_text)
        return evaluate(term_text)

    terms, totals, sources = {}, {}, {}
    for match in _TERM_CALL.finditer(body):
        terms[match["tag"]] = offdiag_term(
            numpy.asarray(arrays[match["volume"]]),
            numpy.asarray(arrays[match["coefficient"]]),
            flat(match["down"]), flat(match["up"]), flat(match["corner"]))
    for match in _TOTAL_FIRST.finditer(body):
        totals[match["name"]] = terms[match["tag"]].copy()
    for match in _TOTAL_ADD.finditer(body):
        totals[match["name"]] = _f32(totals[match["name"]] + terms[match["tag"]])
    for match in _MASK.finditer(body):
        if wall_flags[match["wall"]]:
            totals[match["name"]] = _f32(numpy.where(
                faces[match["face"]], numpy.float32(0.0), totals[match["name"]]))
    for match in _SRC_COUPLED.finditer(body):
        name = match["name"]
        sources[name] = _f32(_f32(scalars[f"gs_{name}"] * scalars[f"us_{name}"])
                             + totals[name])
    for match in _SRC_PLAIN.finditer(body):
        name = match["name"]
        sources[name] = _f32(scalars[f"gs_{name}"] * scalars[f"us_{name}"])
    for match in _TAIL.finditer(body):
        axis = "ijk".index(match["index"])
        broadcast = [1, 1, 1]
        broadcast[axis] = shape[axis]
        kps = _f32(numpy.asarray(tables[match["kps"]]).reshape(broadcast))
        kms = _f32(numpy.asarray(tables[match["kms"]]).reshape(broadcast))
        field = arrays[match["field"]]
        aux = arrays[match["aux"]]
        # constitutive_apply (PINNED by exact string below, and welded to the
        # certified sibling's), transcribed: prev is read BEFORE fw is written,
        # then two separate accumulations left to right.
        previous = numpy.asarray(aux).copy()
        source_values = sources[match["name"]]
        aux[...] = source_values
        accumulated = _f32(numpy.asarray(field) + _f32(kps * source_values))
        field[...] = _f32(accumulated - _f32(kms * previous))


def _tables_for(layer):
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}_h").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def _device_arrays(fields, rows):
    arrays = {name: getattr(fields, name)
              for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez",
                           "Dx", "Dy", "Dz")}
    for component in ("Ex", "Ey", "Ez"):
        arrays[f"inv_eps_{component}"] = fields.inverse_epsilon_for(component)
    for row, partners in rows.items():
        for partner, volume in partners.items():
            arrays[f"chi1inv_{row}_{partner}"] = volume
    return arrays


def _bits(fields):
    return {name: numpy.asarray(getattr(fields, name)).tobytes()
            for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")}


def _advance_sources(fields):
    """Move D between sub-steps, as a real run's curl does.

    Held fixed, ``f_w`` settles and the recurrence stops being exercised — the
    multi-step leg would then be a slow single-step leg.
    """
    for name in ("Dx", "Dy", "Dz"):
        getattr(fields, name)[...] = (getattr(fields, name)
                                      * numpy.float32(0.97))


def run_both_paths(mask, boundaries, steps=8, source=None, rows_override=None,
                   uniform=False, seed=20260816):
    """The oracle and the emitted source, from one frozen state. Returns both bit sets.

    ONE fixture and one restore: two separately built configurations could differ
    in their inputs, silently, and the comparison would be of two runs rather than
    of two trees.

    PLAIN NUMPY, NOT THE ``xp`` FIXTURE, and deliberately: the arithmetic legs
    EXECUTE the emitted source in NumPy, so on a CUDA host a CuPy-backed fixture
    would hand the evaluator arrays ``numpy.asarray`` refuses to convert. The
    fixture belongs to the PREDICATE legs, where the backend name is the question.
    What is being measured here is a float32 expression tree, which is the same
    tree whatever allocated the arrays.
    """
    fields, layer, grid = build(numpy, boundaries=boundaries)
    rows = install_rows(fields, grid, mask, uniform=uniform)
    seed_state(fields, grid, seed=seed % 1000)
    frozen = {name: numpy.asarray(getattr(fields, name)).copy()
              for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                           "f_w_Ex", "f_w_Ey", "f_w_Ez")}

    for _ in range(steps):
        stepping.update_E(fields, layer)
        _advance_sources(fields)
    oracle = _bits(fields)
    magnitude = max(float(numpy.abs(numpy.asarray(getattr(fields, name))).max())
                    for name in ("Ex", "Ey", "Ez"))

    for name, values in frozen.items():
        getattr(fields, name)[...] = values
    codes = coverage.offdiag_boundary_codes_for_test = tuple(
        coverage.BC_CODES[kind]
        for kind in coverage.real_pml_boundary_kinds(grid))
    walls = coverage.offdiag_wall_mask_flags(grid)
    arrays = _device_arrays(fields, rows_override if rows_override else rows)
    text = offdiag_emitter.offdiag_source(mask) if source is None else source
    for _ in range(steps):
        evaluate_emitted_source(text, arrays, _tables_for(layer), codes, walls,
                                grid.shape)
        _advance_sources(fields)
        arrays = _device_arrays(fields, rows_override if rows_override else rows)
    return oracle, _bits(fields), magnitude


# --------------------------------------------------------------------------
# The predicate: verdict and reason, case by case.
# --------------------------------------------------------------------------

_GOLDEN = {
    "covered_periodic": (True, "covered"),
    "covered_m_m_p": (True, "covered"),
    "covered_m_m_m": (True, "covered"),
    "covered_m_p_m": (True, "covered"),
    "covered_one_row": (True, "covered"),
    # The widening both constitutive predicates share: a conductivity reaches
    # stepping._apply_curl (stepping.py:508) and no constitutive sub-step.
    "conductivity": (True, "covered"),
    "no_rows": (
        False, "no off-diagonal chi1inv row survived installation: that "
               "configuration is covers_real_pml_constitutive(side='E')'s and "
               "this predicate must not overlap it"),
    "zero_rows_dropped": (
        False, "no off-diagonal chi1inv row survived installation: that "
               "configuration is covers_real_pml_constitutive(side='E')'s and "
               "this predicate must not overlap it"),
    "flag_set_slots_dead": (
        False, "no off-diagonal chi1inv row survived installation: that "
               "configuration is covers_real_pml_constitutive(side='E')'s and "
               "this predicate must not overlap it"),
    "complex_storage": (
        False, "complex64 storage: the recurrence is the same but the storage is not"),
    "mirror_plane": (
        False, "mirror symmetry: the fold changes the stored extent, and the "
               "extent is what turns a cell index into a coefficient index"),
    "cylindrical": (
        False, "cylindrical (Dcyl): the axial extent moves every coefficient index"),
    "special_kz": (
        False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"),
    "inactive_layer": (False, "no active PML layer"),
    "numpy_backend": (False, "backend is not CuPy"),
    "bloch_k": (
        False, "nonzero Bloch k: the wrapped plane carries a phase real storage "
               "cannot hold"),
    "bfast": (False, "BFAST: a second additive term on every curl target"),
    "chi2": (False, "instantaneous chi2/chi3: the Pade factor scales the whole "
                    "row product, coupling included"),
    "chi3": (False, "instantaneous chi2/chi3: the Pade factor scales the whole "
                    "row product, coupling included"),
    "polarizations": (
        False, "dispersion: update_E's source is (D - sum P), not D, and "
               "update_P closes the step"),
    "no_stored_E": (False, "E is recomputed from D rather than stored"),
    "row_dtype": (
        False, "chi1inv_offdiagonal['Ex']['Ey'] is float64, not float32"),
    "row_shape": (
        False, "chi1inv_offdiagonal['Ex']['Ey'] has shape (2, 2, 2), not the "
               "grid's (8, 8, 8)"),
    "row_not_contiguous": (
        False, "chi1inv_offdiagonal['Ex']['Ey'] is not C-contiguous"),
    "row_scalar": (
        False, "chi1inv_offdiagonal['Ex']['Ey'] is a scalar, not a volume"),
    "row_aliases_output": (
        False, "chi1inv_offdiagonal['Ex']['Ey'] aliases output f_w_Ez: the "
               "coupling re-reads the partner volumes at neighbour offsets while "
               "the outputs are written, so the answer would depend on block "
               "schedule"),
    "wrong_dtype": (False, "Ez is float64, not float32"),
    "wrong_shape": (False, "Dx has shape (2, 2, 2), not the grid's (8, 8, 8)"),
    "not_contiguous": (False, "f_w_Ey is not C-contiguous"),
    "inv_eps_scalar": (
        False, "inverse_epsilon_for('Ez') is a scalar, not a volume"),
    "pml_vector_missing": (False, "pml.kps_y_h is missing"),
    "pml_vector_dtype": (False, "pml.kms_z_h is float64, not float32"),
    "pml_vector_size": (False, "pml.kps_x_h has 7 entries, not the axis's 8"),
    "unreadable_grid": (True, None),  # asserted separately: the message varies
}

_ALL_ROWS = (1, 1, 1, 1, 1, 1)


def configuration(case, xp):
    """Build one named configuration. Rows are installed through the installer
    unless the case is specifically about defeating it."""
    if case == "covered_periodic":
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        return fields, layer, grid
    if case.startswith("covered_") and case != "covered_one_row":
        kinds = {"m": "metallic", "p": "periodic"}
        fields, layer, grid = build(
            xp, boundaries=tuple(kinds[c] for c in case.split("_")[1:]))
        install_rows(fields, grid, _ALL_ROWS)
        return fields, layer, grid
    if case == "covered_one_row":
        fields, layer, grid = build(xp)
        install_rows(fields, grid, (0, 0, 0, 0, 1, 0))
        return fields, layer, grid
    if case == "conductivity":
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        fields.set_d_conductivity(xp.full(grid.shape, 0.4, dtype=xp.float32))
        return fields, layer, grid
    if case == "no_rows":
        fields, layer, grid = build(xp)
        return fields, layer, grid
    if case == "zero_rows_dropped":
        # An IDENTICALLY ZERO row is dropped at install (fields.py:1302-1303), so
        # this run is bit-identically the diagonal engine's and belongs to the
        # certified plain kernel. The disjointness rests on that drop.
        fields, layer, grid = build(xp)
        install_rows(fields, grid, (0,) * 6)
        fields.set_epsilon_volumes(
            {c: fields.epsilon_for(c) for c in ("Ex", "Ey", "Ez")},
            {c: fields.inverse_epsilon_for(c) for c in ("Ex", "Ey", "Ez")},
            chi1inv_offdiagonal={"Ex": {"Ey": xp.zeros(grid.shape,
                                                       dtype=xp.float32)}})
        return fields, layer, grid
    if case == "flag_set_slots_dead":
        # A row planted PAST the installer under a key no slot names: the flag is
        # True and every slot is dead. Counting the flag would admit it and the
        # emitter would then raise on the all-dead mask.
        fields, layer, grid = build(xp)
        fields._chi1inv_offdiagonal = {"Ex": {"Ex": object()}}
        fields.enable_field_storage()
        return fields, layer, grid
    if case == "complex_storage":
        fields, layer, grid = build(xp, force_complex_fields=True)
        install_rows(fields, grid, _ALL_ROWS)
        return fields, layer, grid
    if case == "mirror_plane":
        fields, layer, grid = build(xp, symmetry=("y",))
        install_rows(fields, grid, _ALL_ROWS)
        return fields, layer, grid
    if case == "cylindrical":
        grid = Grid(resolution=1.0, cell_size=(8.0, 0.0, 8.0), cylindrical=True,
                    m=0, xp=xp)
        layer = PML(grid=grid, thickness={"x": (0, 2), "z": 2})
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        install_rows(fields, grid, _ALL_ROWS)
        return fields, layer, grid
    if case == "special_kz":
        fields, layer, grid = build(xp, cell=(8.0, 8.0, 0.0), dimensions=2,
                                    beta=0.25)
        install_rows(fields, grid, _ALL_ROWS)
        return fields, layer, grid
    if case == "inactive_layer":
        grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=xp)
        layer = PML(grid=grid, thickness=0)
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        install_rows(fields, grid, _ALL_ROWS)
        return fields, layer, grid
    if case == "numpy_backend":
        grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=numpy)
        layer = PML(grid=grid, thickness=2)
        fields = Fields(grid=grid)
        fields.enable_field_storage()
        fields.enable_pml_storage()
        install_rows(fields, grid, _ALL_ROWS)
        return fields, layer, grid
    if case == "bloch_k":
        fields, layer, grid = build(xp, k_point=(0.25, 0.0, 0.0))
        install_rows(fields, grid, _ALL_ROWS)
        return fields, layer, grid
    if case == "bfast":
        fields, layer, grid = build(xp, bfast_scaled_k=(0.1, 0.0, 0.0))
        install_rows(fields, grid, _ALL_ROWS)
        return fields, layer, grid
    if case == "polarizations":
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        fields.polarizations.append(object())
        return fields, layer, grid
    if case in ("chi2", "chi3"):
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        setattr(fields, f"_{case}_components", {"Ex": 0.5})
        return fields, layer, grid
    if case == "no_stored_E":
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        fields._stored_E = False
        fields._pml_active = False
        return fields, layer, grid
    if case in ("row_dtype", "row_shape", "row_not_contiguous", "row_scalar",
                "row_aliases_output"):
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        entry = fields._chi1inv_offdiagonal["Ex"]
        if case == "row_dtype":
            entry["Ey"] = entry["Ey"].astype(xp.float64)
        elif case == "row_shape":
            entry["Ey"] = xp.zeros((2, 2, 2), dtype=xp.float32)
        elif case == "row_not_contiguous":
            entry["Ey"] = xp.asfortranarray(entry["Ey"])
        elif case == "row_scalar":
            entry["Ey"] = xp.float32(0.5)
        else:
            # REACHABLE THROUGH THE PUBLIC INSTALLER: set_epsilon_volumes keeps
            # the caller's array without copying (fields.py:1296), so a row that
            # IS an output arrives legally installed.
            entry["Ey"] = fields.f_w_Ez
        return fields, layer, grid
    if case == "wrong_dtype":
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        fields.Ez = fields.Ez.astype(xp.float64)
        return fields, layer, grid
    if case == "wrong_shape":
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        fields.Dx = xp.zeros((2, 2, 2), dtype=xp.float32)
        return fields, layer, grid
    if case == "not_contiguous":
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        fields.f_w_Ey = xp.asfortranarray(fields.f_w_Ey)
        return fields, layer, grid
    if case == "inv_eps_scalar":
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        fields._inv_eps_components["Ez"] = xp.float32(0.5)
        return fields, layer, grid
    if case in ("pml_vector_missing", "pml_vector_dtype", "pml_vector_size"):
        fields, layer, grid = build(xp)
        install_rows(fields, grid, _ALL_ROWS)
        if case == "pml_vector_missing":
            layer.kps_y_h = None
        elif case == "pml_vector_dtype":
            layer.kms_z_h = layer.kms_z_h.astype(xp.float64)
        else:
            layer.kps_x_h = layer.kps_x_h.reshape(-1)[:-1]
        return fields, layer, grid
    raise AssertionError(f"no builder for {case!r}")


_CASES = sorted(case for case in _GOLDEN if case != "unreadable_grid")


@pytest.mark.parametrize("case", _CASES)
def test_the_predicate_states_the_verdict_the_table_specifies(case, xp):
    """Verdict AND reason, case by case. A refusal that fires for a different
    clause than the one intended is a refusal nobody can act on."""
    fields, layer, grid = configuration(case, xp)
    assert coverage.covers_real_pml_offdiag_constitutive(fields, layer, grid) \
        == _GOLDEN[case], (
            f"{case}: got "
            f"{coverage.covers_real_pml_offdiag_constitutive(fields, layer, grid)!r}")


def test_the_predicate_refuses_rather_than_raising_on_a_grid_that_cannot_answer(xp):
    """A RAISE IS NOT A REFUSAL unless something catches it: this predicate is read
    by a planner deciding whether to dispatch, so an exception escaping it is a
    CRASHED RUN where a fail-closed no was the correct answer."""
    fields, layer, grid = configuration("covered_periodic", xp)

    class Unanswerable:
        xp = grid.xp

        def __getattr__(self, name):
            if name in ("is_metallic", "has_symmetry", "is_axis", "is_mirrored"):
                raise RuntimeError("this grid cannot say")
            return getattr(grid, name)

    covered, reason = coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, Unanswerable())
    assert covered is False
    assert "could not answer" in reason


def test_a_fields_whose_row_reader_raises_is_refused_rather_than_crashing(xp):
    """``chi1inv_offdiagonal_for`` is read six times and every one is fail-closed:
    a reader that raises yields six dead slots, which the inverted clause then
    refuses BY NAME rather than letting the exception escape a planner."""
    fields, layer, grid = configuration("covered_periodic", xp)

    def explode(component):
        raise RuntimeError("no rows here")

    fields.chi1inv_offdiagonal_for = explode
    covered, reason = coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, grid)
    assert covered is False
    assert "no off-diagonal chi1inv row survived installation" in reason


def test_the_two_e_side_predicates_are_disjoint_on_every_configuration(xp):
    """THE SEAM. ``covers_real_pml_constitutive(side='E')`` refuses every
    off-diagonal run and this predicate REQUIRES one, so no configuration may be
    admitted by both — otherwise two families would claim the same sub-step and
    which one ran would depend on the planner's branch order.

    Measured over the whole case table rather than argued from the two clauses.
    """
    both = []
    for case in _CASES:
        fields, layer, grid = configuration(case, xp)
        plain = coverage.covers_real_pml_constitutive(fields, layer, grid, "E")[0]
        offdiag = coverage.covers_real_pml_offdiag_constitutive(
            fields, layer, grid)[0]
        if plain and offdiag:
            both.append(case)
    assert both == [], f"admitted by both E-side predicates: {both}"


def test_the_plain_predicate_still_carries_the_refusal_this_family_depends_on(xp):
    """The disjointness is only as good as the sibling's clause, so it is checked
    here too: deleting that refusal as "stale" would make the two overlap."""
    fields, layer, grid = configuration("covered_periodic", xp)
    covered, reason = coverage.covers_real_pml_constitutive(
        fields, layer, grid, "E")
    assert covered is False
    assert "off-diagonal chi1inv" in reason


@pytest.mark.parametrize("case", ["covered_periodic", "covered_m_m_m",
                                  "covered_one_row"])
def test_every_admitted_configuration_yields_a_row_mask_the_emitter_accepts(case, xp):
    """A covered verdict must never meet a raise: the predicate's inverted clause
    and the emitter's all-dead refusal are the same rule stated twice, and this is
    where the two are checked to agree."""
    fields, layer, grid = configuration(case, xp)
    assert coverage.covers_real_pml_offdiag_constitutive(fields, layer, grid)[0]
    mask = coverage.offdiag_row_mask(fields)
    assert offdiag_emitter.normalized_row_mask(mask) == mask
    assert mask in offdiag_emitter.LIVE_ROW_MASKS


# --------------------------------------------------------------------------
# The transcribed tables, pinned against the modules they were transcribed from.
# --------------------------------------------------------------------------

def test_the_term_table_matches_steppings_constitutive_terms():
    """``E_TERMS`` is ``stepping.E_CONSTITUTIVE_TERMS`` with the axis NAME resolved
    to an index; the sub-step writes those components, in that order."""
    assert tuple((name, source, "xyz".index(axis))
                 for name, source, axis in stepping.E_CONSTITUTIVE_TERMS) \
        == offdiag_emitter.E_TERMS


def test_the_partner_table_is_meeps_cycle_direction():
    """``cycle_direction`` X -> Y -> Z: own axis + 1 then own axis + 2
    (stepping.py:1235-1237, ``partner_axis = (own_axis + offset) % 3``)."""
    assert coverage.OFFDIAG_TRANSVERSE_PARTNERS == tuple(
        tuple((own + offset) % 3 for offset in (1, 2)) for own in range(3))


def test_the_row_slots_are_the_partner_table_spelled_as_component_pairs():
    """``OFFDIAG_ROW_SLOTS`` must be exactly the (row, partner) pairs
    ``_offdiagonal_terms`` looks up (``rows.get('E' + AXIS_NAMES[partner_axis])``,
    stepping.py:1237-1238), in the same order the plan binds them."""
    expected = tuple(
        (name, "E" + "xyz"[partner])
        for component, (name, _source, own) in enumerate(offdiag_emitter.E_TERMS)
        for partner in coverage.OFFDIAG_TRANSVERSE_PARTNERS[component])
    assert coverage.OFFDIAG_ROW_SLOTS == expected


def test_the_wall_mask_axes_are_the_yee_shift_zero_axes():
    """``_mask_metallic_wall_coupling`` skips an axis whose ``iyee`` is non-zero
    (stepping.py:1280-1281) and loops ascending, so the table is exactly the
    ascending shift-0 axes of each row component in ``fields.IYEE_SHIFTS``."""
    for component, (name, _source, _own) in enumerate(offdiag_emitter.E_TERMS):
        shifts = IYEE_SHIFTS[name]
        assert coverage.OFFDIAG_WALL_MASK_AXES[component] == tuple(
            axis for axis in range(3) if shifts[axis] == 0), name


def test_the_wall_mask_asks_the_declaration_and_not_the_resolved_ghost_rule(xp):
    """The mask reads ``grid.is_metallic(axis) and not grid.is_mirrored(axis)``
    (stepping.py:1282) — the DECLARATION. The ghost codes are the RESOLUTION, and
    the two coincide only while folds are refused.

    WITHIN THIS FAMILY'S ADMITTED SPACE THEY ARE LOCKED, and that is measured over
    all eight boundary triples rather than left implied: the separation is
    future-proofing for the day a fold is admitted, not a live degree of freedom.
    """
    kinds = ("periodic", "metallic")
    for bx in kinds:
        for by in kinds:
            for bz in kinds:
                _fields, layer, grid = build(xp, boundaries=(bx, by, bz))
                walls = coverage.offdiag_wall_mask_flags(grid)
                codes = tuple(coverage.BC_CODES[kind] for kind in
                              coverage.real_pml_boundary_kinds(grid))
                assert walls == tuple(int(code == coverage.BC_CODES["metallic"])
                                      for code in codes), (bx, by, bz)


def test_the_helpers_agree_with_the_shipped_sibling_helpers(xp):
    """THE ANTI-DESYNC CLAUSE. The corpus census that produced this family's
    variant count calls the sibling track's ``row_volumes_for`` and
    ``wall_mask_axes``; if this track's answers drifted from those, the count
    would describe a different kernel than the one shipped here.

    The sibling module is importable without its optional dependency by design, so
    this runs at the merge bar.
    """
    from ..triton_kernels import offdiag_update_e as sibling  # noqa: PLC0415

    assert sibling.ROW_SLOTS == coverage.OFFDIAG_ROW_SLOTS
    assert sibling.WALL_MASK_AXES == coverage.OFFDIAG_WALL_MASK_AXES
    assert sibling.TRANSVERSE_PARTNERS == coverage.OFFDIAG_TRANSVERSE_PARTNERS
    for mask in ((1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1), (0, 0, 1, 0, 0, 0)):
        fields, _layer, grid = build(xp)
        install_rows(fields, grid, mask)
        theirs = tuple(int(v is not None) for v in sibling.row_volumes_for(fields))
        assert theirs == coverage.offdiag_row_mask(fields) == mask
        assert tuple(sibling.wall_mask_axes(grid)) == \
            coverage.offdiag_wall_mask_flags(grid)


# --------------------------------------------------------------------------
# The emitter and its device text.
# --------------------------------------------------------------------------

def test_the_family_ships_one_kernel_and_it_is_partitioned():
    """A kernel shipped without a gate verdict must fail here, not ship unmeasured."""
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    certified = set(module_level_literal(source, "CERTIFIED_KERNELS"))
    dead = module_level_literal(source, "UNCERTIFIED_KERNELS")
    shipped = set()
    for mask in offdiag_emitter.LIVE_ROW_MASKS:
        shipped |= set(KERNEL_DECLARATION.findall(
            offdiag_emitter.offdiag_source(mask)))

    assert shipped == {offdiag_emitter.KERNEL_NAME}
    assert not certified & set(dead)
    assert certified | set(dead) == shipped
    for name, owes in dead.items():
        assert "gate_cuda_offdiag" in owes or "probe_fused_kernel_bit_identity" in owes, (
            f"{name} does not name the gate it owes: {owes!r}")


def test_every_certified_name_has_a_record_block_behind_it():
    """"Certified" must be a VERDICT, never a word someone typed.

    Both directions: a name in ``CERTIFIED_KERNELS`` needs a
    ``certification.json`` block that claims it and reports a pass, and a block
    claiming a name this module does not certify is equally a drift. Until
    2026-08-16 this file asserted the opposite — that no block claimed the family
    — because none had run.
    """
    import json  # noqa: PLC0415

    source = KERNEL_MODULE.read_text(encoding="utf-8")
    certified = set(module_level_literal(source, "CERTIFIED_KERNELS"))
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    claimed = {name: key for key, block in record.items()
               if isinstance(block, dict)
               for name in block.get("certified_kernels", ())}
    for name in certified:
        assert name in claimed, (
            f"{name} is claimed certified with no certification.json block "
            f"behind it")
    block = record["offdiag_2026-08-16"]
    assert block["passed"] is True
    assert block["certified_kernels"] == [offdiag_emitter.KERNEL_NAME]
    assert sorted(block["policies_cut_under"]) == [
        "ieee_keep_ftz_stripped", "meep_x86_flush"]
    for policy in ("keep", "flush"):
        leg = block["per_policy"][policy]
        assert leg["passed"] is True
        assert leg["all_legs_as_required"] is True
        assert leg["legs_not_measurable_on_this_backend"] == 0
        assert leg["cases_refused_as_vacuous"] == 0
        assert leg["every_case_had_live_coupling"] is True
        assert leg["subnormal_band_is_non_vacuous"] is True
        # THE GUARD IS EVIDENCE, NOT DECORATION: the unguarded control has to
        # DIVERGE, or --fmad=false is being asserted rather than measured.
        identical, ran = leg["guard_control_identical_at_inexact_courant"].split("/")
        assert int(ran) > 0 and int(identical) < int(ran)
    assert block["distinct_binaries_per_policy"]["binaries_are_distinct"] is True


def test_the_record_block_was_cut_against_the_emitter_that_ships_today():
    """THE DIGEST IS THE WHOLE CLAIM for a family whose device code is emitted.

    The gate sweeps four of the 63 row masks; what carries the other 59 is that
    the emitter has not moved. ``corpus_digest`` hashes every source it can emit,
    so a single changed character anywhere in it separates the shipped emitter
    from the one the verdict was taken on — and this is where that shows.
    """
    import json  # noqa: PLC0415

    block = json.loads(RECORD.read_text(encoding="utf-8"))["offdiag_2026-08-16"]
    assert block["emitter_corpus_digest"] == offdiag_emitter.corpus_digest(), (
        "the emitter has changed since the gate ran; the certification does not "
        "describe the bytes that ship")
    edited = {name for name in block["revision_sha256"]
              if not name.startswith("_")}
    assert edited and block["post_certification_edits"], (
        "a revision digest with no edit entry behind it says nothing")
    for name, digest in block["subject_sha256"].items():
        live = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if live == digest:
            continue
        # ONE HOME FOR THE RULE (device_identity.py:209). This family's device
        # text comes from the emitter, whose corpus digest is asserted above, so
        # a subject edit that provably reaches no device source has nothing left
        # to invalidate. The helper returns False for anything it cannot
        # establish -- a block with no device_sha256/code_sha256 among them --
        # and the declared-edit rule below then applies unchanged.
        if weld_survives_edit(HERE / name, block, name):
            continue
        assert name in edited, (
            f"{name} has changed since the gate ran and no post_certification_edits "
            f"entry says why")
        assert block["revision_sha256"][name] == live, (
            f"{name} has changed AGAIN since the recorded post-gate edit; the "
            f"record's revision digest no longer describes the shipped file")
    for entry in block["post_certification_edits"]:
        assert entry["touches_device_code"] is False, (
            "a post-gate edit that touched device code invalidates the verdict")


def test_the_record_block_was_cut_against_the_launch_texts_that_ship_today():
    """The compiled bytes, not only the certified ones.

    The single launches :func:`offdiag_emitter.offdiag_launch_source`, the certified
    text through the own-cell hoist. ``emitter_corpus_digest`` cannot see an edit to the
    hoist, so the block also records the digest of all 63 launch texts, taken when the
    gate ran on them.
    """
    import json  # noqa: PLC0415

    block = json.loads(RECORD.read_text(encoding="utf-8"))["offdiag_2026-08-16"]
    assert "launch_corpus_digest" in block, (
        "the block records no launch-text digest; the 59 masks the gate does not sweep "
        "rest on nothing that sees the compiled bytes")
    assert block["launch_corpus_digest"] == offdiag_emitter.launch_corpus_digest(), (
        "a launch text has changed since the gate ran; the certification does not "
        "describe the bytes that compile")


def test_the_launch_digest_covers_every_mask_and_is_not_the_certified_one(monkeypatch):
    """The two digests answer different questions, so they must differ; and the launch
    digest is recomputed from an empty memo each time, so equality is determinism."""
    import hashlib  # noqa: PLC0415

    assert len(offdiag_emitter.LIVE_ROW_MASKS) == 63
    monkeypatch.setattr(offdiag_emitter, "_LAUNCH_TEXTS", {})
    first = offdiag_emitter.launch_corpus_digest()
    monkeypatch.setattr(offdiag_emitter, "_LAUNCH_TEXTS", {})
    assert offdiag_emitter.launch_corpus_digest() == first
    assert first != offdiag_emitter.corpus_digest()
    digest = hashlib.sha256()
    for mask in offdiag_emitter.LIVE_ROW_MASKS:
        digest.update(repr(mask).encode("ascii"))
        digest.update(offdiag_emitter.offdiag_launch_source(mask).encode("utf-8"))
    assert digest.hexdigest() == first


def test_there_are_sixty_three_live_row_masks_and_the_all_dead_one_is_refused():
    """63 = 2**6 - 1. The all-dead mask is the certified plain kernel's, and
    emitting this family's source for it would overlap the two families."""
    assert len(offdiag_emitter.LIVE_ROW_MASKS) == 63
    assert len(set(offdiag_emitter.LIVE_ROW_MASKS)) == 63
    assert all(any(mask) for mask in offdiag_emitter.LIVE_ROW_MASKS)
    with pytest.raises(ValueError, match="no row slot survives"):
        offdiag_emitter.offdiag_source((0, 0, 0, 0, 0, 0))
    for bad in ((1, 0, 0), (1, 0, 0, 1, 0, 2)):
        with pytest.raises(ValueError, match="row mask"):
            offdiag_emitter.offdiag_source(bad)


def test_every_emitted_source_is_pure_ascii_and_survives_an_ascii_locale_write():
    """A COMPILE REQUIREMENT, NOT A STYLE RULE.
    ``compile_using_nvrtc`` writes the source with a bare ``open(..., 'w')``
    (compiler.py:368), so the bytes go through the interpreter's LOCALE encoding —
    ASCII under C/POSIX. Two em-dashes in a comment killed the certified sibling's
    E kernel at its first launch on 2026-08-15. Scanned AND encoded, per mask."""
    for mask in offdiag_emitter.LIVE_ROW_MASKS:
        text = offdiag_emitter.offdiag_source(mask)
        offending = [(index, character) for index, character in enumerate(text)
                     if ord(character) > 127]
        assert offending == [], f"{mask}: non-ASCII at {offending[:3]}"
        text.encode("ascii")  # the write compile_using_nvrtc performs


def test_the_compile_options_are_both_siblings_options():
    """``--fmad=false`` is CORRECTNESS, not tuning: the two tail accumulations, the
    ``D * inv_eps`` product and the two ``pair * coefficient`` products are all
    contraction candidates the array path rounds twice. Spelled in each file so a
    by-path load cannot pick up a different tuple; pinned equal here so the three
    cannot drift apart."""
    ours = module_level_literal(KERNEL_MODULE.read_text(encoding="utf-8"),
                                "_COMPILE_OPTIONS")
    constitutive = module_level_literal(
        CONSTITUTIVE_MODULE.read_text(encoding="utf-8"), "_COMPILE_OPTIONS")
    curl = module_level_literal(CURL_MODULE.read_text(encoding="utf-8"),
                                "_COMPILE_OPTIONS")
    assert ours == constitutive == curl == ("--fmad=false",)


def test_the_tail_is_character_for_character_the_certified_siblings():
    """THE WELD. Both families run the same recurrence, and the certified one's
    expression tree is already measured against ``_apply_constitutive_pml`` over 60
    sub-steps. Copying the text is only sound while the copy stays a copy."""
    def extract(text):
        start = text.index("__device__ __forceinline__ void constitutive_apply(")
        end = text.index("\n}\n", start) + len("\n}\n")
        return text[start:end]

    assert extract(offdiag_emitter.PRELUDE) == \
        extract(CONSTITUTIVE_MODULE.read_text(encoding="utf-8"))


#: The four helpers the evaluator transcribes by hand rather than executing from
#: the text. Pinned character-for-character, so a change to any of them fails here
#: instead of silently passing an arithmetic leg that no longer describes them.
PINNED_HELPERS = {
    "coord_up": """__device__ __forceinline__ int coord_up(int a, int n, int bc) {
    if (a + 1 < n) return a + 1;
    return (bc == BC_METALLIC) ? -1 : 0;
}""",
    "coord_dn": """__device__ __forceinline__ int coord_dn(int a, int n, int bc) {
    if (a > 0) return a - 1;
    return (bc == BC_METALLIC) ? -1 : n - 1;
}""",
    "flat": """__device__ __forceinline__ int flat(int i, int j, int k, int nyz, int nz) {
    return (i < 0 || j < 0 || k < 0) ? -1 : (i * nyz + j * nz + k);
}""",
    "ghosted": """__device__ __forceinline__ float ghosted(const float* g, int index) {
    return (index < 0) ? 0.0f : g[index];
}""",
}


@pytest.mark.parametrize("name", sorted(PINNED_HELPERS))
def test_the_hand_transcribed_helpers_are_pinned(name):
    assert PINNED_HELPERS[name] in offdiag_emitter.PRELUDE, (
        f"{name} has changed; the evaluator's Python transcription of it no "
        f"longer describes the shipped text")


def _code_only(text):
    """The emitted source with its ``//`` comments removed.

    Every textual leg that asserts something is ABSENT reads this rather than the
    raw text: the comments name the very things the code must not contain (the
    mirror branch, the curl's ``dtdx``), so scanning them would make an
    absence test fail on its own documentation.
    """
    return "\n".join(line.split("//")[0] for line in text.split("\n"))


def test_the_ghost_rules_are_the_two_stepping_branches_and_no_others():
    """MIRROR and CYL_AXIS are refused by the predicate and real storage cannot
    carry a Bloch phase, so those branches must not exist in the device text — a
    branch nothing can reach is a branch nothing gates."""
    for mask in CORPUS_ROW_MASKS:
        text = _code_only(offdiag_emitter.offdiag_source(mask))
        for absent in ("BC_MIRROR", "CYL", "bloch", "phase", "parity"):
            assert absent not in text, absent
        assert "BC_METALLIC" in text and "BC_PERIODIC" in text


def test_the_boundary_and_wall_axes_are_runtime_arguments_not_baked_in():
    """THE SPLIT THIS FAMILY'S DESIGN RESTS ON. The boundary and wall axes are
    BRANCH axes — they select an index or a predicated zero and change no float
    operation — so they are runtime ``int`` arguments, as the certified
    ``step_B_pml_real`` already takes ``bc_x``/``bc_y``/``bc_z``. If they
    were baked in, the source count would be 64x what it is and every one of them
    would owe a byte gate.

    Measured, not asserted from the docstring: every emitted source is identical
    whatever the boundaries, because the boundaries are not an emitter input at
    all — the signature carries them and the emitter has no parameter for them.
    """
    for mask in CORPUS_ROW_MASKS:
        text = offdiag_emitter.offdiag_source(mask)
        assert "int bc_x, int bc_y, int bc_z," in text
        assert "int wm_x, int wm_y, int wm_z" in text
    assert offdiag_emitter.offdiag_source.__code__.co_argcount == 1
    # And the sibling kernel that established the precedent still takes them.
    assert "int bc_x, int bc_y, int bc_z" in CURL_MODULE.read_text(
        encoding="utf-8")


def test_only_the_live_coefficients_are_declared():
    """No pointer is bound to an array the kernel must never touch. The sibling
    track binds all six and points the dead ones at a D volume; that works, and it
    leaves a shape a later edit can read by accident."""
    for mask in offdiag_emitter.LIVE_ROW_MASKS:
        text = offdiag_emitter.offdiag_source(mask)
        for slot, parameter in enumerate(offdiag_emitter.ROW_PARAMETERS):
            declared = f"const float* {parameter},"
            assert (declared in text) == bool(mask[slot]), (mask, parameter)
            if not mask[slot]:
                assert parameter not in text, (mask, parameter)


def test_the_read_only_volumes_carry_no_restrict_promise_the_caller_cannot_keep():
    """An isotropic install hands the same inverse-permittivity pointer three times
    (fields.py:1321-1326) and a row coefficient may legally alias another row's or
    a D volume, so restrict on the read-only VOLUMES is a promise the caller cannot
    keep. The outputs and the PML vectors do carry it: the predicate's alias clause
    and the launcher's check are what make it true."""
    text = offdiag_emitter.offdiag_source(CORPUS_ROW_MASKS[1])
    start = text.index("extern \"C\"")
    signature = text[start:text.index(") {", start)]
    for promised in ("float* __restrict__ Ex", "float* __restrict__ f_w_Ez",
                     "const float* __restrict__ kps_x"):
        assert promised in signature, promised
    for plain in ("const float* Dx,", "const float* inv_eps_Ex,",
                  "const float* chi1inv_Ex_Ey,"):
        assert plain in signature, plain
    assert "__restrict__ Dx" not in signature
    assert "__restrict__ inv_eps" not in signature
    assert "__restrict__ chi1inv" not in signature


def test_the_emitted_source_borrows_nothing_from_the_curl_sub_step():
    """Anyone porting the certified curl kernel's shape into this one will reach
    for its ownership mask and its Courant factor. Neither belongs here: this
    sub-step masks a WALL PLANE OF THE COUPLING (a different rule, a different
    reason) and has no ``dtdx`` at all."""
    for mask in CORPUS_ROW_MASKS:
        text = _code_only(offdiag_emitter.offdiag_source(mask))
        for borrowed in ("dtdx", "curl", "fu_", "sinv", "shift_up(", "shift_dn("):
            assert borrowed not in text, borrowed


def test_the_corpus_digest_is_stable_and_moves_with_the_emitter():
    """One sha256 over all 63 sources. Deterministic across calls, and sensitive to
    a single character anywhere in the emitter — which is the property a per-source
    list would provide, without 63 digests burying the file."""
    first = offdiag_emitter.corpus_digest()
    assert first == offdiag_emitter.corpus_digest()
    assert len(first) == 64
    concatenated = hashlib.sha256()
    for mask in offdiag_emitter.LIVE_ROW_MASKS:
        concatenated.update(repr(mask).encode("ascii"))
        concatenated.update(offdiag_emitter.offdiag_source(mask).encode("utf-8"))
    assert first == concatenated.hexdigest()


def test_the_emitter_module_still_pulls_in_no_device_dependency():
    """The emitter has to run at the merge bar, and it can only do that while it
    imports nothing that needs a GPU. Read off the syntax tree, so a future import
    fails here on a laptop rather than being discovered on a device host."""
    tree = ast.parse(EMITTER_MODULE.read_text(encoding="utf-8"))
    forbidden = {"cupy", "numpy", "triton", "torch"}
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            assert name.partition(".")[0] not in forbidden, (
                f"{EMITTER_MODULE.name}:{node.lineno} imports {name!r}")


def test_every_line_of_the_device_source_is_executed_or_pinned():
    """NOTHING IN THE SHIPPED TEXT IS NEITHER RUN NOR PINNED.

    The evaluator's grammar covers the per-component statements; the pinned
    helpers and the welded tail cover the prelude; the template's fixed scaffolding
    (the signature, the index decomposition, the coordinate lines, the face
    predicates) is listed here explicitly. A line matching none of the three is a
    line no leg in this file can see, and this is where that is caught.
    """
    scaffolding = (
        "extern \"C\" __global__", "float* __restrict__", "const float*",
        "int nx, int ny, int nz,", "int bc_x", "int wm_x",
        "int idx = blockIdx.x", "if (idx >= nx * ny * nz) return;",
        "int nyz = ny * nz;", "int k = idx % nz;", "int j = (idx / nz) % ny;",
        "int i = idx / (ny * nz);", "int di = coord_dn", "int dj = coord_dn",
        "int dk = coord_dn", "int at_x = (i == 0)", ") {", "}", "{",
    )
    grammar = (_TERM_CALL, _TOTAL_FIRST, _TOTAL_ADD, _MASK, _GS, _US,
               _SRC_COUPLED, _SRC_PLAIN, _TAIL)
    for mask in offdiag_emitter.LIVE_ROW_MASKS:
        text = offdiag_emitter.offdiag_source(mask)
        body = text[text.index("extern \"C\""):]
        # The multi-line term call is matched as a whole, so its continuation
        # lines are accounted for by the span it covers.
        covered = set()
        for pattern in grammar:
            for match in pattern.finditer(body):
                covered.update(range(match.start(), match.end()))
        offset = 0
        for line in body.split("\n"):
            stripped = line.strip()
            span = set(range(offset, offset + len(line)))
            offset += len(line) + 1
            if not stripped or stripped.startswith("//"):
                continue
            if span & covered:
                continue
            assert any(token in line for token in scaffolding), (
                f"{mask}: no leg in this file executes or pins {stripped!r}")


def test_every_line_of_the_prelude_is_executed_or_pinned():
    """The same claim for the OTHER half of the source, which the scan above stops at.

    ``offdiag_term``'s three arithmetic lines are EXECUTED (the evaluator parses
    them with :data:`_TERM_BODY` and evaluates them as expressions); the four small
    helpers and the tail are PINNED by exact string. Everything else in the prelude
    is a preprocessor line, a signature or a brace, listed here. A prelude line
    matching none of the three would be device code no leg in this file can see —
    which is exactly the state the arithmetic legs are worth nothing in.
    """
    executed = _TERM_BODY.search(offdiag_emitter.PRELUDE)
    assert executed is not None, (
        "the evaluator can no longer find offdiag_term's arithmetic lines; every "
        "arithmetic leg below is then running a transcription of a body it is not "
        "reading")
    pinned = "\n".join(PINNED_HELPERS.values())
    tail = offdiag_emitter.PRELUDE[
        offdiag_emitter.PRELUDE.index(
            "__device__ __forceinline__ void constitutive_apply("):]
    scaffolding = ("#define ", "__device__ __forceinline__", "const float*",
                   "int a, int n, int bc", "int i, int j, int k, int nyz, int nz",
                   "float kps, float kms", ") {", "{", "}")
    for line in offdiag_emitter.PRELUDE.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if stripped in executed.group(0) or stripped in pinned or stripped in tail:
            continue
        assert any(token in line for token in scaffolding), (
            f"no leg in this file executes or pins {stripped!r}")


# --------------------------------------------------------------------------
# THE ARITHMETIC, measured against stepping.py — on a laptop, in float32.
#
# What the byte gate would measure is whether the COMPILED kernel reproduces the
# array path; what these measure is whether the EXPRESSION TREE THE EMITTER WROTE
# reproduces it, which is the half a compiler cannot fix and the half that is
# wrong in every one of the four plain curl kernels (0/16 against the array path
# under every option set). If the tree is wrong here it is wrong on the device.
#
# EVERY LEG CARRIES A VACUITY FLOOR. Zero-init is a FIXED POINT of this sub-step,
# so a leg run on zeros passes for every tree; the floor asserts the oracle's own
# output is in the physical band.
# --------------------------------------------------------------------------

_ARITHMETIC_CASES = [(mask, boundaries)
                     for mask in CORPUS_ROW_MASKS
                     for boundaries in BOUNDARY_TRIPLES]
_ARITHMETIC_CASES += [((0, 0, 1, 0, 0, 0), ("metallic", "metallic", "periodic")),
                      ((0, 1, 0, 0, 1, 0), ("periodic", "metallic", "metallic")),
                      ((1, 1, 0, 0, 0, 0), ("metallic", "periodic", "metallic"))]


@pytest.mark.parametrize("mask,boundaries", _ARITHMETIC_CASES,
                         ids=[f"{''.join(str(f) for f in m)}-"
                              f"{b[0][0]}{b[1][0]}{b[2][0]}"
                              for m, b in _ARITHMETIC_CASES])
def test_the_emitted_source_is_bit_identical_to_stepping(mask, boundaries):
    """THE HEADLINE LEG: eight consecutive sub-steps, byte-compared.

    Both stored quantities are compared — E and ``f_w_E`` — because ``f_w`` is what
    the next sub-step reads and a tree that gets E right and the auxiliary wrong is
    correct for exactly one launch.
    """
    oracle, mine, magnitude = run_both_paths(mask, boundaries, steps=8)
    assert magnitude > 1e-3, (
        f"vacuity floor: the oracle produced |E| <= {magnitude}, so every tree "
        f"would agree and this leg measures nothing")
    assert oracle == mine, [name for name in oracle if oracle[name] != mine[name]]


def test_the_transcription_holds_over_the_multi_step_budget():
    """"Identical for N steps" is a claim about N, and 60 is the budget the
    certified constitutive pair's record carries — taken from the sibling track's
    finding that 8, 10 and 6 consecutive steps all passed a divergence 40 did
    not."""
    oracle, mine, magnitude = run_both_paths(
        (1, 1, 1, 1, 1, 1), ("metallic", "metallic", "periodic"),
        steps=MULTI_STEP_BUDGET)
    assert magnitude > 1e-3
    assert oracle == mine


def test_a_uniform_coefficient_is_also_reproduced():
    """The degenerate coefficient the registration hides in, kept as its own leg so
    a pass on the varying rows cannot stand in for it."""
    oracle, mine, magnitude = run_both_paths(
        (1, 1, 1, 1, 1, 1), ("metallic", "metallic", "metallic"), steps=8,
        uniform=True)
    assert magnitude > 1e-3
    assert oracle == mine


def test_an_invariant_axis_doubles_the_partner_pair_rather_than_zeroing_it():
    """A reduced (n = 1) axis wraps to itself, so the partner pair is ``2*g`` — NOT
    the curl's exact zero — matching MEEP's ``stride(d) = 0`` double-read. The
    2-D configurations most of the corpus is made of run through this."""
    fields, layer, grid = build(numpy, cell=(8.0, 8.0, 0.0), dimensions=2)
    assert grid.shape[2] == 1
    rows = install_rows(fields, grid, (1, 1, 1, 1, 1, 1))
    seed_state(fields, grid)
    frozen = {name: numpy.asarray(getattr(fields, name)).copy()
              for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                           "f_w_Ex", "f_w_Ey", "f_w_Ez")}
    for _ in range(6):
        stepping.update_E(fields, layer)
        _advance_sources(fields)
    oracle = _bits(fields)
    magnitude = float(numpy.abs(numpy.asarray(fields.Ez)).max())

    for name, values in frozen.items():
        getattr(fields, name)[...] = values
    codes = tuple(coverage.BC_CODES[kind]
                  for kind in coverage.real_pml_boundary_kinds(grid))
    walls = coverage.offdiag_wall_mask_flags(grid)
    text = offdiag_emitter.offdiag_source((1, 1, 1, 1, 1, 1))
    for _ in range(6):
        evaluate_emitted_source(text, _device_arrays(fields, rows),
                                _tables_for(layer), codes, walls, grid.shape)
        _advance_sources(fields)
    assert magnitude > 1e-3
    assert oracle == _bits(fields)


# --------------------------------------------------------------------------
# The vacuity controls: each defect the transcription protects against, planted
# in the SHIPPED TEXT and required to be caught.
# --------------------------------------------------------------------------

def _mutate(text, kind):
    """Plant one defect in an emitted source. Every one is a wrong answer that
    stays smooth, converged and plausible — which is why none of them can be left
    to review."""
    if kind == "same_direction_shifts":
        # Both shifts UP: the half-cell registration error.
        return re.sub(
            r"flat\((\w+), (\w+), (\w+), nyz, nz\)",
            lambda m: "flat({}, {}, {}, nyz, nz)".format(
                *[{"di": "ui", "dj": "uj", "dk": "uk"}.get(g, g)
                  for g in m.groups()]), text)
    if kind == "own_axis_shift_down":
        # The own-axis shift taken DOWN instead of up.
        return text.replace("flat(ui, j, k, nyz, nz)", "flat(di, j, k, nyz, nz)") \
                   .replace("flat(i, uj, k, nyz, nz)", "flat(i, dj, k, nyz, nz)") \
                   .replace("flat(i, j, uk, nyz, nz)", "flat(i, j, dk, nyz, nz)")
    if kind == "hoist_coefficient":
        # The four-point-average-times-u[i] hoist: the same ALGEBRA only for a
        # uniform coefficient, and a different float32 number even then.
        return text.replace(
            "return 0.25f * ((near_pair * u[home])"
            " + (far_pair * ghosted(u, up)));",
            "return 0.25f * ((near_pair + far_pair) * u[home]);")
    if kind == "drop_wall_mask":
        return re.sub(r"\n *total_\w+ = \(wm_[xyz] && at_[xyz]\)"
                      r" \? 0\.0f : total_\w+;", "", text)
    if kind == "over_apply_wall_mask":
        # The mask applied wherever the FACE is, whatever the grid declares.
        return re.sub(r"\(wm_([xyz]) && at_([xyz])\)", r"(at_\2 && at_\2)", text)
    if kind == "wrong_own_axis_table":
        # Every component reads the x table: MEEP's dsigw confused for one axis.
        return re.sub(r"kps_[xyz]\[[ijk]\], kms_[xyz]\[[ijk]\]",
                      "kps_x[i], kms_x[i]", text)
    if kind == "column_major_index":
        return text.replace("(i * nyz + j * nz + k)", "(k * nyz + j * nz + i)")
    if kind == "flatten_tail":
        return text.replace(
            "    float a = f[idx] + kps * src;\n    f[idx] = a - kms * prev;",
            "    f[idx] = f[idx] + (kps * src - kms * prev);")
    if kind == "store_before_load":
        return text.replace(
            "    float prev = fw[idx];\n    fw[idx] = src;",
            "    fw[idx] = src;\n    float prev = fw[idx];")
    if kind == "distribute_quarter":  # a predicted NULL, not a mutation
        return text.replace(
            "return 0.25f * ((near_pair * u[home])"
            " + (far_pair * ghosted(u, up)));",
            "return (0.25f * (near_pair * u[home]))"
            " + (0.25f * (far_pair * ghosted(u, up)));")
    raise AssertionError(kind)


#: Defects the EVALUATOR can see, because it executes the text they are planted
#: in: the per-component statements and ``offdiag_term``'s three arithmetic lines.
CAUGHT_MUTATIONS = ("same_direction_shifts", "own_axis_shift_down",
                    "hoist_coefficient", "drop_wall_mask",
                    "over_apply_wall_mask", "wrong_own_axis_table")

#: Defects planted in the PINNED half of the source — the four small helpers and
#: the welded tail, which the evaluator transcribes rather than executes. The
#: arithmetic legs are BLIND to these by construction, and saying so is the point:
#: what catches them is the exact-string pin, and
#: :func:`test_the_pinned_half_is_guarded_by_its_pins_not_by_the_arithmetic_legs`
#: measures BOTH halves of that claim — the pin fires, and the arithmetic leg does
#: not. A defect that no leg at all could see would be invisible; these are not.
PINNED_HALF_MUTATIONS = ("column_major_index", "flatten_tail", "store_before_load")

#: Mutations predicted to change NO bit, carried as controls rather than as
#: catches. A null that turned into a catch is as much a finding as the reverse.
NULL_MUTATIONS = ("distribute_quarter",)


@pytest.mark.parametrize("kind", CAUGHT_MUTATIONS)
def test_the_planted_defect_is_caught(kind):
    """Without these the headline leg measures nothing: a tree that agrees with the
    array path at operands where EVERY tree agrees has been tested against the
    seed, not against the transcription."""
    mask, boundaries = (1, 1, 1, 1, 1, 1), ("metallic", "metallic", "periodic")
    original = offdiag_emitter.offdiag_source(mask)
    mutated = _mutate(original, kind)
    assert mutated != original, f"{kind} did not change the source it was given"
    oracle, mine, magnitude = run_both_paths(mask, boundaries, steps=8,
                                             source=mutated)
    assert magnitude > 1e-3
    assert oracle != mine, (
        f"{kind} produced identical bytes: this configuration cannot distinguish "
        f"the defect, so the leg that passes on the shipped source is vacuous "
        f"with respect to it")


@pytest.mark.parametrize("kind", NULL_MUTATIONS)
def test_the_predicted_null_changes_no_bit(kind):
    """0.25 is an exact power of two, so distributing it commutes with
    round-to-nearest away from underflow. Recorded as a measurement so the
    transcribed association is not mistaken for a grouping the gate must pin."""
    mask, boundaries = (1, 1, 1, 1, 1, 1), ("metallic", "metallic", "periodic")
    mutated = _mutate(offdiag_emitter.offdiag_source(mask), kind)
    oracle, mine, magnitude = run_both_paths(mask, boundaries, steps=8,
                                             source=mutated)
    assert magnitude > 1e-3
    assert oracle == mine, (
        f"{kind} was predicted to be a null and changed bits; the transcription's "
        f"association is then load-bearing and the gate owes it a mutation")


def test_the_row_sum_is_diagonal_first_and_the_order_is_a_measured_null():
    """``(D*inv_eps) + total`` (stepping.py:1007-1008), and the order is fidelity.

    Float32 addition is bitwise commutative, so this cannot be measured by planting
    the commuted spelling in the source — the evaluator's grammar is anchored on the
    spelling the emitter actually writes, and a form it never emits has no business
    being parseable. It is measured directly instead, on the operands a real run
    produces: the emitted text is pinned diagonal-first, and the commutation is
    shown to be a null at those operands.
    """
    for mask in CORPUS_ROW_MASKS:
        text = offdiag_emitter.offdiag_source(mask)
        for name, _source, _own in offdiag_emitter.E_TERMS:
            coupled = f"float src_{name} = (gs_{name} * us_{name}) + total_{name};"
            plain = f"float src_{name} = gs_{name} * us_{name};"
            assert coupled in text or plain in text, name

    fields, layer, grid = build(numpy,
                                boundaries=("metallic", "metallic", "periodic"))
    install_rows(fields, grid, (1, 1, 1, 1, 1, 1))
    seed_state(fields, grid)
    stepping.update_E(fields, layer)
    diagonal = numpy.asarray(fields.Dx) * numpy.asarray(
        fields.inverse_epsilon_for("Ex"))
    coupling = numpy.asarray(fields.f_w_Ex) - diagonal
    assert float(numpy.abs(coupling).max()) > 1e-6, (
        "the coupling is numerically absent, so commuting the row sum could not "
        "be distinguished from anything and this control measures nothing")
    assert (diagonal + coupling).tobytes() == (coupling + diagonal).tobytes()


@pytest.mark.parametrize("kind", PINNED_HALF_MUTATIONS)
def test_the_pinned_half_is_guarded_by_its_pins_not_by_the_arithmetic_legs(kind):
    """BOTH HALVES OF THE CLAIM, measured.

    The four small helpers and the tail are transcribed by the evaluator rather
    than executed from the text, so a defect planted in them is INVISIBLE to the
    arithmetic legs — and it must be, or the transcription would be executing its
    own copy of the mutation. What catches it is the exact-string pin. This asserts
    the pin fires AND that the arithmetic leg does not, so the division of labour
    is a measurement rather than an assumption.

    That the trees themselves are distinguishable in float32 — a flattened
    accumulation is a different number, a store-before-load is a different
    recurrence — is already measured for this exact text by the certified sibling's
    file, over 60 consecutive sub-steps, which is what the character-for-character
    weld lets this file inherit.
    """
    mask, boundaries = (1, 1, 1, 1, 1, 1), ("metallic", "metallic", "periodic")
    original = offdiag_emitter.offdiag_source(mask)
    mutated = _mutate(original, kind)
    assert mutated != original, f"{kind} did not change the source it was given"

    pinned = dict(PINNED_HELPERS)
    pinned["constitutive_apply"] = (
        "    float prev = fw[idx];\n    fw[idx] = src;\n"
        "    float a = f[idx] + kps * src;\n    f[idx] = a - kms * prev;")
    broken = [name for name, text in pinned.items() if text not in mutated]
    assert broken, (
        f"{kind} passed every exact-string pin, and the arithmetic legs cannot "
        f"see it either; nothing in this file would catch it")

    oracle, mine, _magnitude = run_both_paths(mask, boundaries, steps=8,
                                              source=mutated)
    assert oracle == mine, (
        f"{kind} changed the evaluator's answer, so the evaluator is reading that "
        f"part of the text after all and this test's premise is wrong")


def test_mispairing_the_coefficients_with_their_partners_is_caught():
    """The gate's m3, planted at the BINDING rather than in the text: the two
    coefficients of one component swapped. Bit-identity would then hold for a
    symmetric tensor and fail for every real one, which is exactly the failure a
    smooth field hides."""
    mask, boundaries = (1, 1, 1, 1, 1, 1), ("metallic", "metallic", "periodic")
    fields, _layer, grid = build(numpy, boundaries=boundaries)
    rows = install_rows(fields, grid, mask)
    swapped = {row: dict(partners) for row, partners in rows.items()}
    swapped["Ex"]["Ey"], swapped["Ex"]["Ez"] = rows["Ex"]["Ez"], rows["Ex"]["Ey"]
    oracle, mine, magnitude = run_both_paths(mask, boundaries, steps=8,
                                             rows_override=swapped)
    assert magnitude > 1e-3
    assert oracle != mine


def test_the_wall_mask_leg_is_reachable_and_the_periodic_control_is_not():
    """The mask can only be measured where the grid DECLARES a metallic axis, so
    the same planted defect must be a catch there and a null on an all-periodic
    grid. Without the second half, "the mask is caught" could be an artefact of
    something else the metallic case changes."""
    mask = (1, 1, 1, 1, 1, 1)
    mutated = _mutate(offdiag_emitter.offdiag_source(mask), "drop_wall_mask")
    metallic = run_both_paths(mask, ("metallic", "metallic", "metallic"),
                              steps=8, source=mutated)
    periodic = run_both_paths(mask, ("periodic", "periodic", "periodic"),
                              steps=8, source=mutated)
    assert metallic[0] != metallic[1], "the mask never fired on an all-metallic grid"
    assert periodic[0] == periodic[1], (
        "dropping the mask changed bits on an all-periodic grid, where no wall is "
        "declared; something other than the mask is being measured")


# --------------------------------------------------------------------------
# THE ARITY WITNESS — why the row mask is a compile-time axis.
# --------------------------------------------------------------------------

def test_padding_a_dead_slot_with_a_zero_coefficient_is_not_bit_identical():
    """THE MEASUREMENT BEHIND THIS FAMILY'S SPECIALIZATION SPLIT.

    The obvious way to make the row mask a runtime argument is to bind every dead
    slot to an all-zero coefficient volume, so a dropped term contributes ``+0.0``
    rather than not existing. ``x + 0.0f`` is the identity on the bits for every
    float32 x EXCEPT ``-0.0f``, which it turns into ``+0.0f``.

    The witness is explicit rather than statistical: a coupling total of ``-0.0f``
    is reachable whenever both partner products are negative zero, which needs only
    a zero partner volume and a negative coefficient — an ordinary interior cell of
    an off-diagonal tensor beside a null field region.

    THIS IS NOT A CLAIM ABOUT A DYNAMIC LOOP, AND THAT FOLD IS LICENSED. A
    runtime-bounded loop over an arity axis was measured bit-identical to the
    unrolled form at every arity 0-8 — 156/156 cases, max ULP 0
    (``the design notes (cuda-kernel-triton-transfer-assessment)`` §3.2,
    ``results/cuda_dynamic_loop_2026-08-16/``) — because NVCC leaves the arithmetic
    a strictly serial dependence chain on one accumulator, which this family's row
    accumulation also is. What this test rules out is the ZERO-PADDED sum, and what
    a dynamic loop would additionally need here is the final ``+ total`` made
    conditional on a non-zero trip count rather than absorbed into the diagonal:
    the array path sums the coupling separately and adds it once (stepping.py:1250
    then :1007-1008), so ``((gs*us) + t1) + t2`` is the wrong tree. Both are selects
    rather than arithmetic, so neither can round.
    """
    total = numpy.float32(-0.0)
    padded = numpy.float32(total + numpy.float32(0.0))
    assert numpy.signbit(total) and not numpy.signbit(padded)
    assert total.tobytes() != padded.tobytes()

    # And the total really can be a negative zero, formed the way the kernel forms
    # it: 0.25 * ((0.0 * -c) + (0.0 * -c)) with the partner volume at rest.
    zero_pair = numpy.float32(0.0)
    coefficient = numpy.float32(-0.5)
    term = numpy.float32(numpy.float32(0.25)
                         * numpy.float32(numpy.float32(zero_pair * coefficient)
                                         + numpy.float32(zero_pair * coefficient)))
    assert numpy.signbit(term), "the negative-zero total is not reachable"
    assert numpy.float32(term + numpy.float32(0.0)).tobytes() != term.tobytes()

    # The same hole sits in a `total = 0.0f` accumulator spelling, which is the
    # other obvious runtime fold.
    assert numpy.float32(numpy.float32(0.0) + term).tobytes() != term.tobytes()


@pytest.mark.parametrize("mask", offdiag_emitter.LIVE_ROW_MASKS,
                         ids=["".join(str(f) for f in m)
                              for m in offdiag_emitter.LIVE_ROW_MASKS])
def test_the_arity_axis_folds_into_the_term_count_and_nothing_else(mask):
    """THE ARITY AXIS, SPELLED OUT: what a row flag changes is HOW MANY terms exist.

    For every one of the 63 live masks: the emitted source carries exactly
    ``popcount(mask)`` term calls; each is the right (partner volume, coefficient)
    pair for its slot; a component with live rows accumulates exactly its own count
    and a component with none emits no ``total`` at all and takes the certified
    plain kernel's ``D * inv_eps`` body; and the wall-mask lines exist only for the
    components that have a total to mask.

    That is the whole of what the axis does — which is why it is the one axis this
    family keeps at compile time, and why the boundary and wall axes, which change
    no count, are runtime arguments.
    """
    text = offdiag_emitter.offdiag_source(mask)
    calls = list(_TERM_CALL.finditer(text))
    assert len(calls) == sum(mask), (
        f"{mask}: {len(calls)} term calls emitted for {sum(mask)} live slots")

    expected = []
    for component, (name, _source, own_axis) in enumerate(offdiag_emitter.E_TERMS):
        for offset in (0, 1):
            slot = 2 * component + offset
            if not mask[slot]:
                continue
            partner_axis = coverage.OFFDIAG_TRANSVERSE_PARTNERS[component][offset]
            expected.append((f"{name}_{offset}",
                             offdiag_emitter.PARTNER_VOLUMES[partner_axis],
                             offdiag_emitter.ROW_PARAMETERS[slot]))
    assert [(m["tag"], m["volume"], m["coefficient"]) for m in calls] == expected

    for component, (name, _source, _own) in enumerate(offdiag_emitter.E_TERMS):
        live = sum(mask[2 * component:2 * component + 2])
        adds = [m for m in _TOTAL_ADD.finditer(text) if m["name"] == name]
        firsts = [m for m in _TOTAL_FIRST.finditer(text) if m["name"] == name]
        masks = [m for m in _MASK.finditer(text) if m["name"] == name]
        assert len(firsts) == (1 if live else 0), (mask, name)
        assert len(adds) == max(live - 1, 0), (mask, name)
        assert len(masks) == (len(coverage.OFFDIAG_WALL_MASK_AXES[component])
                              if live else 0), (mask, name)
        coupled = f"float src_{name} = (gs_{name} * us_{name}) + total_{name};"
        plain = f"float src_{name} = gs_{name} * us_{name};"
        assert (coupled in text) == bool(live), (mask, name)
        assert (plain in text) == (not live), (mask, name)


def test_the_corpus_drives_two_row_masks_and_both_emit(xp):
    """The census number, carried as data and exercised. 4,096 static Triton
    variants; on this track's split the boundary and wall axes cost nothing and the
    row mask is the only source axis — of whose 63 values the 186-row corpus asks
    for these two."""
    assert set(CORPUS_ROW_MASKS) <= set(offdiag_emitter.LIVE_ROW_MASKS)
    for mask in CORPUS_ROW_MASKS:
        fields, layer, grid = build(xp)
        install_rows(fields, grid, mask)
        assert coverage.covers_real_pml_offdiag_constitutive(
            fields, layer, grid) == (True, "covered")
        assert coverage.offdiag_row_mask(fields) == mask
        assert offdiag_emitter.KERNEL_NAME in offdiag_emitter.offdiag_source(mask)


# --------------------------------------------------------------------------
# Predicate mutations: a fail-closed predicate has to be shown to fail closed.
# --------------------------------------------------------------------------

def _predicate_with(*edits):
    """Recompile ``covers_real_pml_offdiag_constitutive`` with edits applied.

    Each edit is an ``(original, replacement)`` pair applied to the REAL function's
    source, so an edit that no longer applies is a failure here rather than a leg
    quietly reporting a pass for a defect it never planted.
    """
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    source = textwrap.dedent(inspect.getsource(
        coverage.covers_real_pml_offdiag_constitutive))
    for original, replacement in edits:
        assert original in source, f"the mutation no longer applies: {original!r}"
        source = source.replace(original, replacement)
    namespace = dict(vars(coverage))
    exec(compile(source, "<mutated predicate>", "exec"), namespace)
    return namespace["covers_real_pml_offdiag_constitutive"]


@pytest.mark.parametrize("case,original,replacement", [
    ("row_aliases_output", "if address is not None and address in output_addresses:",
     "if False:"),
    ("no_rows", "if not any(volume is not None for volume in rows):", "if False:"),
    ("no_stored_E", 'if not getattr(fields, "stores_E", False):', "if False:"),
    ("polarizations", 'if getattr(fields, "polarizations", None):', "if False:"),
    ("row_dtype", "problem = _array_problem(label, volume, xp, shape)",
     "problem = None"),
    ("chi2", 'if getattr(fields, "_chi2_components", None) '
             'or getattr(fields, "_chi3_components", None):', "if False:"),
])
def test_dropping_a_clause_is_caught(case, original, replacement, xp):
    """Every refusal here is a SILENT WRONG ANSWER if it leaks, so each one is
    removed and the configuration it protects required to become admitted."""
    mutated = _predicate_with((original, replacement))
    fields, layer, grid = configuration(case, xp)
    assert coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, grid)[0] is False
    assert mutated(fields, layer, grid)[0] is True, (
        f"dropping the clause for {case} did not admit it; the clause is not what "
        f"refuses that configuration and the test measures nothing")


#: The two configurations no single clause decides, with the full set of clauses
#: that refuse them. BELT AND BRACES IS A MEASURABLE PROPERTY, and it has to be
#: measured rather than asserted, because a one-clause drop that changes no verdict
#: reads exactly like a clause that is not load-bearing.
_BELT_AND_BRACES = {
    "mirror_plane": (
        ('if facts["has_symmetry"]:', "if False:"),
        ('if facts["mirrored"][axis]:', "if False:"),
        ("if kind not in BC_CODES:", "if False:"),
    ),
    "complex_storage": (
        ('if getattr(fields, "force_complex_fields", False):', "if False:"),
        ("problem = _array_problem(name, getattr(fields, name, None), xp, shape)",
         "problem = None"),
    ),
}


@pytest.mark.parametrize("case", sorted(_BELT_AND_BRACES))
def test_the_belt_and_braces_refusals_need_every_clause_dropped(case, xp):
    """MEASURED, and it corrects the shape a one-clause mutation would have taken.

    A folded grid is refused THREE times — ``has_symmetry``, the per-axis
    ``mirrored`` clause, and the boundary-kind clause, since a mirrored axis
    resolves to a kind with no code — and complex storage TWICE, by the storage
    flag and by the dtype of the arrays it produces. Dropping any one changes no
    verdict, so each is peeled in turn and required to leave the configuration
    refused, and only the full set is required to admit it. That is what makes each
    clause's presence a statement rather than decoration.

    THE ENGINE DOES STEP THE FOLDED CONFIGURATION. ``_validated_offdiagonal_rows``
    installs rows unchanged on a folded grid (fields.py:1262-1310), with
    fold-equivalence measured 8.3e-13..4.7e-12 — so the refusal is this KERNEL
    FAMILY'S (the stored extent moves every coefficient index and the wall mask
    abstains on a mirror), not the engine's. The claim at stepping.py:1219-1220
    that the installer refuses the combination is stale, and the fix belongs in
    that file, not in this kernel family.
    """
    edits = _BELT_AND_BRACES[case]
    fields, layer, grid = configuration(case, xp)
    assert coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, grid)[0] is False
    for index in range(len(edits)):
        partial = edits[:index] + edits[index + 1:]
        assert _predicate_with(*partial)(fields, layer, grid)[0] is False, (
            f"{case} was admitted with clause {index} still in place, so the "
            f"others are decoration rather than belt and braces")
    assert _predicate_with(*edits)(fields, layer, grid)[0] is True, (
        f"{case} is still refused with every listed clause dropped; something "
        f"else decides it and the list is incomplete")


def test_widening_the_row_clause_to_the_bare_flag_is_caught(xp):
    """The inverted clause counts SLOTS, not ``has_offdiagonal_epsilon``. A row
    planted past the installer under a diagonal key sets the flag with every slot
    dead; counting the flag admits it and the emitter then raises on the all-dead
    mask — a covered verdict meeting a ValueError."""
    mutated = _predicate_with(
        ("if not any(volume is not None for volume in rows):",
         'if not getattr(fields, "has_offdiagonal_epsilon", False):'))
    fields, layer, grid = configuration("flag_set_slots_dead", xp)
    assert coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, grid)[0] is False
    assert mutated(fields, layer, grid)[0] is True
    with pytest.raises(ValueError, match="no row slot survives"):
        offdiag_emitter.normalized_row_mask(coverage.offdiag_row_mask(fields))


def test_dropping_the_int32_bound_is_caught(xp):
    """The kernel indexes with ``int`` and its neighbour offsets add ``ny*nz`` to
    the same int; past 2**31 cells that is a wrapped index into a live volume, not
    a launch failure. The clause is unreachable from any grid this machine can
    allocate, so it is reached with a stand-in that reports the shape."""
    fields, layer, grid = configuration("covered_periodic", xp)

    class HugeGrid:
        shape = (2048, 2048, 512)  # 2**31 exactly

        def __getattr__(self, name):
            return getattr(grid, name)

    covered, reason = coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, HugeGrid())
    assert covered is False and "int32 index range" in reason
    mutated = _predicate_with(("if cells >= 2 ** 31:", "if False:"))
    # The stand-in's arrays are the real grid's, so the clauses after this one
    # refuse for a shape mismatch; what is pinned is that the int32 clause is the
    # one that fires FIRST, and that it fires at all.
    assert mutated(fields, layer, HugeGrid())[1] != reason


# --------------------------------------------------------------------------
# The launcher's own guards, checked without a device.
# --------------------------------------------------------------------------

def test_the_launcher_refuses_both_a_layer_and_an_override():
    """A caller with two answers to a one-answer question has a bug either way, and
    the half-cell error the derived route exists to prevent is exactly what an
    override can reintroduce."""
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    launcher = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "update_E_offdiag_fused_pml_real")
    raises = [node for node in ast.walk(launcher) if isinstance(node, ast.Raise)]
    assert len(raises) >= 2, (
        "the launcher must refuse both the pml-plus-override and the "
        "partial-override calls; a silent default is the half-cell error")
    arguments = [argument.arg for argument in launcher.args.kwonlyargs]
    assert arguments == ["tables", "codes", "walls"], (
        "the gate's overrides must be keyword-only: a positional table is how the "
        "certified sibling's sub-lattice pairing stayed one argument away from "
        "being wrong for a year")


def test_the_launcher_derives_the_sub_lattice_from_the_shared_function():
    """The E side reads the HALF-INTEGER tables (stepping.py:1015). Asked of
    ``constitutive_sub_lattice``, the same function the predicate asks, so the two
    cannot disagree; swapped, it is a converged, smooth, wrong absorber."""
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    assert "real_constitutive_tables(pml, constitutive_sub_lattice(\"E\"))" in source
    assert coverage.constitutive_sub_lattice("E") is True
    assert coverage.constitutive_sub_lattice("H") is False


def test_the_kernel_module_re_exports_rather_than_copying(xp):
    """Two copies of a table is two things to keep in step. The CuPy half must name
    the stdlib halves, not restate them."""
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    for name in ("OFFDIAG_ROW_SLOTS", "LIVE_ROW_MASKS", "E_TERMS",
                 "covers_real_pml_offdiag_constitutive", "offdiag_source"):
        assignments = re.findall(rf"^{name} = (.+)$", source, flags=re.M)
        assert assignments, f"{name} is not re-exported"
        assert all(value.startswith(("_coverage.", "offdiag_emitter."))
                   for value in assignments), (name, assignments)


# --------------------------------------------------------------------------
# THE FOLD: the measured boundary, and why the refusal is broader than it
# --------------------------------------------------------------------------
#
# The shipped predicate refuses every mirror fold. On 2026-08-20 a device gate
# (``parity/meep_gpu/gate_cuda_folded_offdiag.py``, the GPU host GPU 7, both float32
# subnormal policies) measured that refusal instead of inheriting it, and found it
# REAL BUT BROADER THAN THE ARITHMETIC: on 96 folded cases per policy the SHIPPED
# kernel is byte-identical to ``stepping.update_E``, and on the complementary 352
# every differing word lies on the ghost-delta planes.
#
# ``coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION`` carries that measurement and
# ``coverage.offdiag_fold_roles`` carries its rule. The legs below do three
# different jobs, and the difference matters:
#
#   * REPLAY — the rule reproduces the recorded device table. Cheap, and only as
#     good as the record.
#   * MEASURE — the rule is re-derived HERE, by running the emitted source against
#     ``stepping.update_E`` on real folded grids. This is the load-bearing one: it
#     does not consult the artifact at all, and it fails on a laptop if the rule is
#     wrong. The two agree on all 52 combinations.
#   * PIN THE REFUSAL — the predicate still refuses the measured-safe arm, the
#     launcher still cannot serve it, and the record still says the arm is worth
#     zero corpus slots. A widening that leaked in later has to break one of these.

#: The four row masks the device sweep drove, chosen so a folded axis lands in each
#: of its three roles (partner, own-axis-only, neither). Restated here rather than
#: imported from the gate, which needs CuPy.
FOLD_ROW_MASKS = ((1, 1, 1, 1, 1, 1), (1, 0, 0, 1, 0, 0),
                  (1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0))

#: Fold specs for the LOCAL measurement: every axis alone, two planes at once and
#: three, plus the unfolded control. A folded axis gets a cell long enough to hold
#: a PML layer after halving.
FOLD_SYMMETRIES = ((), ("x",), ("y",), ("z",),
                   ("x", "y"), ("x", "z"), ("x", "y", "z"))

#: The two codes a folded axis can be handed. ``mirror`` has none — ``BC_CODES``
#: carries ``periodic`` and ``metallic`` only — so the choice is an experimental
#: variable and BOTH are measured. It is also the variable the record's second
#: reason turns on.
FOLD_SUBSTITUTIONS = ("metallic", "periodic")


def _fold_cell(symmetry):
    return tuple(16.0 if "xyz"[axis] in symmetry else (8.0, 10.0, 12.0)[axis]
                 for axis in range(3))


def run_both_paths_folded(mask, symmetry, substitution, steps=2,
                          boundaries=("periodic", "periodic", "periodic"),
                          seed=20260820):
    """The oracle and the emitted source on a FOLDED grid, from one frozen state.

    WHY THIS CANNOT REUSE :func:`run_both_paths`: that helper asks
    ``coverage.BC_CODES[kind]`` for every axis, and a folded axis resolves to
    ``mirror``, which has no entry — the KeyError the launcher deliberately lets
    escape. Choosing a stand-in is exactly what is being measured, so it is named
    per call here, as the device gate names it per case.

    Returns ``(oracle_bits, produced_bits, oracle_moved, folded_axis_deviation,
    mirror_ghost_min_abs)``. The last two are the non-vacuity floors: a folded axis
    whose coefficient profile is the identity cannot show a coefficient-index
    error, and a mirror ghost equal to the metallic zero it is compared against
    makes an "exact" result a fact about the fixture.
    """
    fields, layer, grid = build(numpy, boundaries=boundaries, symmetry=symmetry,
                                cell=_fold_cell(symmetry))
    rows = install_rows(fields, grid, mask, seed=seed)
    seed_state(fields, grid, seed=seed % 1000)
    frozen = {name: numpy.asarray(getattr(fields, name)).copy()
              for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                           "f_w_Ex", "f_w_Ey", "f_w_Ez")}

    # The mirror ghost is ``parity * g[MIRROR_SOURCE_INDEX]`` (stepping.py:
    # 1873-1875). Asked of ``stepping`` rather than spelled, so a second spelling
    # of the reflected row cannot drift from the first.
    ghost = None
    for axis in range(3):
        if not grid.is_mirrored(axis):
            continue
        for name in ("Dx", "Dy", "Dz"):
            index = [slice(None)] * 3
            index[axis] = stepping.MIRROR_SOURCE_INDEX
            value = float(numpy.abs(frozen[name][tuple(index)]).max())
            ghost = value if ghost is None else min(ghost, value)

    deviation = 0.0
    for axis, name in enumerate("xyz"):
        if not grid.is_mirrored(axis):
            continue
        for stem in ("kps", "kms"):
            values = numpy.asarray(getattr(layer, f"{stem}_{name}_h"),
                                   dtype=numpy.float64)
            deviation = max(deviation, float(numpy.abs(values - 1.0).max()))

    for _ in range(steps):
        stepping.update_E(fields, layer)
        _advance_sources(fields)
    oracle = _bits(fields)
    moved = sum(numpy.asarray(getattr(fields, name)).tobytes() != frozen[name].tobytes()
                for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"))

    for name, values in frozen.items():
        getattr(fields, name)[...] = values
    codes = tuple(
        coverage.BC_CODES[substitution if kind == "mirror" else kind]
        for kind in coverage.real_pml_boundary_kinds(grid))
    walls = coverage.offdiag_wall_mask_flags(grid)
    text = offdiag_emitter.offdiag_source(mask)
    arrays = _device_arrays(fields, rows)
    for _ in range(steps):
        evaluate_emitted_source(text, arrays, _tables_for(layer), codes, walls,
                                grid.shape)
        _advance_sources(fields)
        arrays = _device_arrays(fields, rows)
    return oracle, _bits(fields), moved, deviation, ghost


def _recorded_combinations():
    return {(folded, mask, substitution): (ran, identical)
            for folded, mask, substitution, ran, identical
            in coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION["combinations"]}


def _rule_says_identical(folded, mask, substitution):
    roles = coverage.offdiag_fold_roles(
        mask, tuple(axis in folded for axis in range(3)))
    return not roles["changes_a_byte"][substitution]


def test_the_fold_roles_are_read_off_the_row_slot_table_and_nothing_else():
    """``offdiag_fold_roles`` must be the row slots, not a second table of axes.

    Derived here from :data:`coverage.OFFDIAG_ROW_SLOTS` one slot at a time: a
    single live slot has exactly one partner axis and one own axis, and they are
    never the same axis — which is what makes the two roles disjoint per slot and
    the rule a pair rather than a scalar.
    """
    for slot, (row, partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        mask = tuple(int(index == slot) for index in range(6))
        roles = coverage.offdiag_fold_roles(mask, (False, False, False))
        assert roles["live_partner_axes"] == ("xyz".index(partner[1]),)
        assert roles["live_own_axes"] == ("xyz".index(row[1]),)
        assert roles["live_partner_axes"] != roles["live_own_axes"]
    dead = coverage.offdiag_fold_roles((0,) * 6, (True, True, True))
    assert dead["live_partner_axes"] == () and dead["live_own_axes"] == ()
    assert dead["changes_a_byte"] == {"metallic": False, "periodic": False}


def test_an_unfolded_grid_is_never_touched_by_the_fold_rule():
    """No fold, no ghost: every mask must read ``changes_a_byte`` False on both
    codes. A rule that answered otherwise would refuse the 113 unfolded corpus
    slots the shipped predicate already serves."""
    for mask in FOLD_ROW_MASKS:
        roles = coverage.offdiag_fold_roles(mask, (False, False, False))
        assert roles["folded_axes"] == ()
        assert roles["changes_a_byte"] == {"metallic": False, "periodic": False}


#: Every case the local sweep measures, enumerated rather than crossed: with no
#: folded axis there is nothing to substitute FOR, so the unfolded control appears
#: once and the pair (unfolded, periodic) does not exist. Built as a list because
#: a skipped parametrisation is a coverage gap wearing a pass.
FOLD_CASES = tuple(
    (mask, symmetry, substitution)
    for symmetry in FOLD_SYMMETRIES
    for mask in FOLD_ROW_MASKS
    for substitution in (FOLD_SUBSTITUTIONS if symmetry else ("metallic",)))


@pytest.mark.parametrize("mask,symmetry,substitution", FOLD_CASES,
                         ids=lambda value: str(value).replace(" ", ""))
def test_the_fold_rule_is_the_measured_boundary(mask, symmetry, substitution):
    """THE LOAD-BEARING LEG: bit-identity iff ``changes_a_byte`` is False.

    Measured here, on this machine, against ``stepping.update_E`` on a real folded
    grid — the artifact is not consulted. Both directions are asserted, so a rule
    one row too WIDE (admitting a case that diverges) and one row too NARROW
    (refusing a case that is exact) both fail. The vacuity floors are asserted per
    case for the reason the device gate asserts them: without them an "exact"
    verdict can be a fact about a fixture whose folded axis does not absorb and
    whose mirror ghost equals the zero it is compared against.
    """
    oracle, produced, moved, deviation, ghost = run_both_paths_folded(
        mask, symmetry, substitution)
    assert moved == 6, "the oracle moved no words: the comparison is vacuous"
    if symmetry:
        assert deviation > 0.0, (
            "the folded axis's half-integer profile is the identity, so a "
            "coefficient-index error on it would be invisible")
        assert ghost is not None and ghost > 0.0, (
            "the mirror ghost is identically zero, so it is indistinguishable "
            "from the metallic zero ghost it is being compared against")
    folded = tuple(axis for axis in range(3) if "xyz"[axis] in symmetry)
    identical = oracle == produced
    assert identical == _rule_says_identical(folded, mask, substitution), (
        f"folded={folded} mask={mask} code={substitution}: measured "
        f"identical={identical}, rule said "
        f"{_rule_says_identical(folded, mask, substitution)}")


def test_the_local_measurement_reproduces_the_recorded_device_table():
    """Every combination measured here appears in the record with the same verdict.

    The device sweep and this file build their fixtures independently — different
    cell sizes, courants, plane parities and value classes — so agreement across
    all 52 combinations is a second draw, not a re-read of one.
    """
    recorded = _recorded_combinations()
    seen = 0
    for mask, symmetry, substitution in FOLD_CASES:
        folded = tuple(axis for axis in range(3) if "xyz"[axis] in symmetry)
        key = (folded, mask, substitution)
        assert key in recorded, f"{key} is not in the record"
        ran, identical = recorded[key]
        oracle, produced, _, _, _ = run_both_paths_folded(
            mask, symmetry, substitution)
        assert (identical == ran) == (oracle == produced), key
        seen += 1
    assert seen == len(recorded) == len(FOLD_CASES) == 52


def test_every_recorded_combination_is_reproduced_by_the_rule():
    """The record's 480 guarded cases per policy, replayed against the rule.

    ``ident in (0, ran)`` is asserted first and is not bookkeeping: a MIXED
    combination would mean the verdict depends on the courant, the value class or
    the plane parity, and the rule — which reads none of those — could not be the
    boundary at all.
    """
    combinations = coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION["combinations"]
    cases = identical_cases = 0
    for folded, mask, substitution, ran, identical in combinations:
        assert identical in (0, ran), (folded, mask, substitution, ran, identical)
        assert (identical == ran) == _rule_says_identical(
            folded, mask, substitution), (folded, mask, substitution)
        cases += ran
        identical_cases += identical
    record = coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION
    assert cases == record["guarded_cases_per_policy"] == 480
    assert identical_cases == record["unfolded_controls"] + record[
        "folded_bit_identical"] == 128
    assert record["folded_divergent"] == cases - identical_cases == 352
    assert record["rule_disagreements_with_measurement"] == 0


def test_the_recorded_table_is_two_sided_on_every_lever_the_rule_reads():
    """A rule is only measured where the sweep put a case on BOTH sides.

    Three roles for a folded axis and two codes for it: each combination of the
    two that the rule distinguishes must appear, with a divergent case and (where
    the rule predicts one) an exact case. Without this the table could be
    all-divergent and every leg above would pass on a rule that always says "no".
    """
    combinations = coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION["combinations"]
    seen = {}
    for folded, mask, substitution, ran, identical in combinations:
        roles = coverage.offdiag_fold_roles(
            mask, tuple(axis in folded for axis in range(3)))
        role = (bool(folded), roles["fold_is_a_live_partner_axis"],
                roles["fold_is_a_live_own_axis"], substitution)
        seen.setdefault(role, set()).add(identical == ran)
    # Every role the rule distinguishes, and the verdict it must carry.
    expected = {
        (False, False, False, "metallic"): {True},
        (True, False, False, "metallic"): {True},
        (True, False, False, "periodic"): {True},
        (True, False, True, "metallic"): {True},
        (True, False, True, "periodic"): {False},
        (True, True, False, "metallic"): {False},
        (True, True, False, "periodic"): {False},
        (True, True, True, "metallic"): {False},
        (True, True, True, "periodic"): {False},
    }
    assert seen == expected
    # Both plane counts beyond one, so "depth of fold" is not an untested axis.
    depths = {len(folded) for folded, _, _, _, _ in combinations}
    assert depths == {0, 1, 2, 3}
    # Every axis folded somewhere: a rule that confused the slowest coefficient
    # stride with the fastest would otherwise pass.
    assert set().union(*(set(folded) for folded, _, _, _, _ in combinations)) == {0, 1, 2}


@pytest.mark.parametrize("leg,combinations,cases", (
    ("drop_the_partner_leg", 23, 196),
    ("drop_the_own_axis_leg", 3, 32),
    ("read_the_partner_axis_as_the_own_axis", 7, 68),
    ("ignore_the_code_handed_the_fold", 3, 32),
    ("admit_every_fold", 41, 352),
))
def test_a_rule_that_is_one_row_wider_is_caught(leg, combinations, cases):
    """THE MUTATION BATTERY, on the RULE rather than on the kernel.

    Each leg is a plausible mis-statement of the boundary — the ones the reading
    invites — and each must disagree with the recorded table. ``drop_the_own_axis
    _leg`` and ``ignore_the_code_handed_the_fold`` are the same three combinations
    seen from two sides, and they are exactly the 32 guarded cases per policy that
    make "read a fold as periodic" wrong: without them, the clause the artifact's
    ``proposed_predicate_clause.md`` sketches (a partner test alone) would look
    complete.
    """
    def mutated(folded, mask, substitution):
        roles = coverage.offdiag_fold_roles(
            mask, tuple(axis in folded for axis in range(3)))
        partner = roles["fold_is_a_live_partner_axis"]
        own = roles["fold_is_a_live_own_axis"]
        if leg == "drop_the_partner_leg":
            return not (own and substitution == "periodic")
        if leg == "drop_the_own_axis_leg":
            return not partner
        if leg == "read_the_partner_axis_as_the_own_axis":
            return not (own or (partner and substitution == "periodic"))
        if leg == "ignore_the_code_handed_the_fold":
            return not (partner or own)
        if leg == "admit_every_fold":
            return True
        raise AssertionError(leg)

    disagreements = [
        (folded, mask, substitution, ran)
        for folded, mask, substitution, ran, identical
        in coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION["combinations"]
        if mutated(folded, mask, substitution) != (identical == ran)]
    assert len(disagreements) == combinations, disagreements
    assert sum(entry[3] for entry in disagreements) == cases, disagreements


def test_the_predicate_still_refuses_the_arm_the_measurement_licensed(xp):
    """NOT INSTALLED, and that is the point: the measured-safe arm is refused.

    Three configurations the device gate measured bit-identical. Each must still
    come back refused, and refused by the FOLD clause rather than by some later
    one, or "the refusal stands" would be a statement about a different clause.
    """
    mirror_refusal = (
        "mirror symmetry: the fold changes the stored extent, and the extent is "
        "what turns a cell index into a coefficient index")
    for symmetry, mask in ((("z",), (1, 0, 0, 1, 0, 0)),
                           (("x",), (0, 0, 1, 0, 0, 0)),
                           (("x",), (1, 1, 0, 0, 0, 0))):
        folded = tuple(axis for axis in range(3) if "xyz"[axis] in symmetry)
        assert _rule_says_identical(folded, mask, "metallic"), (symmetry, mask)
        fields, layer, grid = build(xp, symmetry=symmetry,
                                    cell=_fold_cell(symmetry))
        install_rows(fields, grid, mask)
        covered, reason = coverage.covers_real_pml_offdiag_constitutive(
            fields, layer, grid)
        assert covered is False and reason == mirror_refusal, (symmetry, mask)


def test_the_launcher_has_no_code_to_hand_a_folded_axis(xp):
    """The record's SECOND reason, measured rather than argued.

    The launcher module is the CuPy half and cannot be imported on this machine,
    so its mapping is taken from its own SOURCE — one expression, pinned — and
    then EVALUATED here on a real folded grid. That is a measurement of the same
    expression the launcher runs, not a reading of what it probably does:
    ``real_pml_boundary_kinds`` answers ``mirror``, ``BC_CODES`` has no such key,
    and the KeyError escapes because the launcher deliberately does not default.

    So a predicate that admitted a fold would promise a planner a dispatch that
    raises — and the obvious default, reading the fold as periodic, is the one the
    mutation battery above shows is wrong on three of the swept combinations.
    """
    source = KERNEL_MODULE.read_text(encoding="utf-8")
    assert ("return tuple(BC_CODES[kind] for kind in real_pml_boundary_kinds(grid))"
            in source), "offdiag_boundary_codes no longer maps through BC_CODES"
    fields, layer, grid = build(xp, symmetry=("y",), cell=_fold_cell(("y",)))
    install_rows(fields, grid, (1, 0, 0, 1, 0, 0))
    kinds = coverage.real_pml_boundary_kinds(grid)
    assert kinds[1] == "mirror"
    assert "mirror" not in coverage.BC_CODES
    with pytest.raises(KeyError):
        tuple(coverage.BC_CODES[kind] for kind in kinds)


def test_the_record_says_what_the_arm_is_worth_and_why_it_is_not_installed():
    """The corpus arithmetic and the two reasons, pinned as data.

    ZERO IS THE FINDING, so it is pinned rather than left implicit: the widened
    predicate admits the same 16 update_E slots as the shipped one on the 186-row
    census, the union is unmoved at 557/759, and the 20 slots a fold could reach
    all need a mirror branch in the kernel. A later census that moves any of these
    should break this test rather than pass quietly.
    """
    record = coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION
    assert record["installed"] is False
    corpus = record["corpus"]
    assert corpus["rows"] == 186 and corpus["slots"] == 759
    assert corpus["union_admitted_before"] == corpus["union_admitted_after"] == 557
    assert (corpus["offdiag_update_E_slots_admitted_before"]
            == corpus["offdiag_update_E_slots_admitted_after"] == 16)
    assert corpus["slots_licensed"] == 0
    assert corpus["overlaps_introduced"] == 0
    assert (corpus["of_those_a_fold_is_a_live_partner_axis"]
            + corpus["of_those_no_off_diagonal_row_survived_at_all"]
            == corpus["of_those_refused_first_by_the_mirror_clause"] == 26)
    assert corpus["slots_needing_a_mirror_branch_in_the_kernel"] == 20
    # The sharp end: a fold-blind widening is not free, it is WRONG on 19 slots.
    assert corpus["slots_a_fold_blind_widening_would_gain"] == 19
    assert corpus["and_all_nineteen_were_measured_divergent"] is True
    assert record["corpus_counts_are_an_upper_bound"] is True
    reasons = record["not_installed_because"]
    assert "zero corpus slots" in reasons and "offdiag_boundary_codes" in reasons


@pytest.mark.requires_resource("cuda_folded_offdiag_artifact")
def test_the_record_was_cut_against_the_artifact_it_names():
    """Anti-transcription: re-derive the table from the device artifact.

    A SANCTIONED skip where the artifact is absent — ``parity/meep_gpu/results/``
    is gitignored — which is exactly why this leg is not the only thing measuring
    the table: :func:`test_the_local_measurement_reproduces_the_recorded_device_table`
    re-measures all 52 combinations from scratch on this machine.
    """
    import collections  # noqa: PLC0415
    import json  # noqa: PLC0415

    from conftest import requires_resource_skip  # noqa: PLC0415

    record = coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION
    root = HERE.parent.parent / record["artifact"]
    for policy_directory, words in (("keep", record["differing_words_keep"]),
                                    ("flush", record["differing_words_flush"])):
        path = root / policy_directory / "gate.json"
        if not path.exists():
            requires_resource_skip(
                "cuda_folded_offdiag_artifact",
                f"{path} is not present (parity/meep_gpu/results/ is gitignored)")
        payload = json.loads(path.read_text(encoding="utf-8"))
        tally = collections.defaultdict(lambda: [0, 0])
        for case in payload["sweep"]["cases"]:
            if case["guard"] != "fmad_false":
                continue
            key = (tuple(axis for axis in range(3) if case["mirrored"][axis]),
                   tuple(case["row_mask"]),
                   "metallic" if case["substitution"] == "mirror_as_metallic"
                   else "periodic")
            tally[key][0] += 1
            tally[key][1] += bool(case["bit_identical"])
        assert {key: tuple(value) for key, value in tally.items()} == \
            _recorded_combinations(), policy_directory
        # And every scalar the record quotes, from the same file.
        summary = payload["sweep"]["summary"]
        assert record["kernel"] == payload["kernel"]
        assert (record["guarded_cases_per_policy"]
                == summary["per_guard"]["fmad_false"]["ran"])
        assert (record["folded_bit_identical"]
                == summary["predicted_exact_folded_identical"])
        assert (record["folded_divergent"]
                == summary["predicted_diverging_that_diverged"])
        assert (record["unfolded_controls"]
                == summary["predicted_exact"] - summary["predicted_exact_folded"])
        assert words == summary["differing_words_total"]
        assert (record["differing_words_off_the_ghost_delta_planes"]
                == summary["differing_elsewhere"] == 0)
        assert record["multi_step_budget"] == payload["multi_step_budget"]
        assert (record["multi_step_folded_bit_identical"]
                == payload["multistep"]["summary"]["predicted_exact_folded_identical"])
        assert (payload["subnormal_policy_install"]["policy"]
                in record["policies"])
