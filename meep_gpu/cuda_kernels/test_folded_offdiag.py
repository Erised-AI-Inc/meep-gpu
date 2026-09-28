"""The FOLDED off-diagonal ``update_E`` slice, at the merge bar.

WHAT A LAPTOP CAN AND CANNOT SETTLE FOR THIS FAMILY, stated first because the
distinction is the whole reason this file exists beside a device gate.

A byte gate measures whether the COMPILED kernel reproduces ``stepping.update_E``.
These legs measure whether the EXPRESSION TREE THE EMITTER WROTE reproduces it --
which is the half a compiler cannot fix, and the half that was wrong in every one
of the four plain curl kernels when they were first written (0/16 against the
array path under every option set). If the tree is wrong here it is wrong on the
device, and it is wrong in a tenth of a second rather than in a device slot.

THE ARITHMETIC LEGS EXECUTE THE EMITTED DEVICE SOURCE. They do not re-implement
it: :func:`evaluate_folded_source` parses the emitted text over a KNOWN GRAMMAR
and evaluates its arithmetic lines as float32 expressions, and
:func:`test_every_line_of_the_device_source_is_executed_or_pinned` fails if a line
of any emitted source is neither matched by that grammar nor pinned by exact
string. So "the tests cover the shipped text" is measured rather than hoped.

WHAT IS NEW IN THIS FAMILY, AND THEREFORE WHAT THESE LEGS ARE FOR. Everything
except the mirror arm is ``offdiag_emitter``'s, certified 2026-08-16 on an RTX
A6000, and is not re-litigated here beyond the reduction leg. The new arithmetic
is exactly three things:

1. ``coord_dn``'s MIRROR branch returning stored row ``MIRROR_ROW``
   (``stepping._shift_down`` :1823-1825, ``_mirror_source`` :1535-1541);
2. the parity WEIGHT applied on that lane and no other
   (``fields.mirror_parity`` through ``stepping._symmetry_phase`` :2402-2414);
3. ``coord_up`` sharing METALLIC's ``-1`` arm on a fold, because
   ``_offdiagonal_terms`` calls ``_shift_up`` without ``component`` or
   ``reflect_row`` (:1219-1220) and both mirror terminations then fall to
   :1781-1783.

Each has a leg that MEASURES it against ``stepping.update_E`` on a real folded
``Grid``/``Fields``/``PML`` triple, and a MUTATION leg that plants the wrong
answer and requires it to be caught.

THE FIXTURE IS THE CERTIFIED SLICE'S, IMPORTED. ``build``, ``install_rows``,
``seed_state``, ``_tables_for``, ``_device_arrays``, ``_bits`` and
``_advance_sources`` come from ``test_offdiag_constitutive_pml_real`` rather than
being copied: two fixtures that differ silently would make the reduction leg a
comparison of two runs instead of of two trees.

NO DEVICE VERDICT IS CLAIMED HERE. ``folded_offdiag_kernels.FOLDED_OFFDIAG_
ADMISSION["host"]`` is None until
``parity/meep_gpu/gate_cuda_folded_offdiag_kernel.py`` records one, and a leg
below asserts that this file does not quietly imply otherwise.
"""

from __future__ import annotations

import io
import pathlib
import re

import numpy
import pytest

from .. import stepping
from ..fields import IYEE_SHIFTS, mirror_parity
from . import coverage, folded_offdiag_kernels as folded, offdiag_emitter
from .test_offdiag_constitutive_pml_real import (
    _advance_sources, _bits, _device_arrays, _tables_for, build, install_rows,
    seed_state)

HERE = pathlib.Path(__file__).parent
MODULE = HERE / "folded_offdiag_kernels.py"

#: The two row masks the 186-row corpus drives, plus two the corpus does not: a
#: single live slot, and a mask that leaves a component on the PLAIN arm. The last
#: two are here because "a component with no surviving row keeps the pure diagonal
#: arithmetic" is a property of the emitter, and a sweep over corpus masks alone
#: would never emit that arm beside a coupled one.
ROW_MASKS = ((1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1),
             (1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0))
CORPUS_ROW_MASKS = ((1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1))

#: Fold specs: every axis alone, two planes at once, three at once, and the
#: unfolded control. EVERY AXIS IS FOLDED SOMEWHERE, because a decomposition
#: defect that confuses the slowest coefficient stride with the fastest is only
#: visible if both are folded in the sweep.
FOLD_SYMMETRIES = ((), ("x",), ("y",), ("z",),
                   ("x", "y"), ("x", "z"), ("x", "y", "z"))

#: Both plane parities. The ghost weight is ``-phase``, so an even plane weights
#: the ghost by -1 and an odd one by +1 -- which is what makes "drop the weight" a
#: DISCRIMINATOR rather than a defect, and a sweep at one parity could not tell.
FOLD_PHASES = (1, -1)

#: Declared terminations for a folded axis. ``stepping._boundary_kinds`` answers
#: ``mirror`` for both, and FACT 3 of the module docstring predicts the two are
#: bit-identical HERE (unlike in the curl pair). Swept so the prediction is
#: measured.
FOLD_TERMINATIONS = ("periodic", "metallic")

MULTI_STEP_BUDGET = 60


def _fold_cell(symmetry):
    """A cell long enough on every folded axis to hold a layer after halving."""
    return tuple(16.0 if "xyz"[axis] in symmetry else (8.0, 10.0, 12.0)[axis]
                 for axis in range(3))


class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__`` -- the one thing about the real device
    library a laptop cannot supply, and the only thing this stands in for."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(numpy, item)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


# --------------------------------------------------------------------------
# THE EVALUATOR: the emitted device source, executed in float32 NumPy.
# --------------------------------------------------------------------------
#
# A targeted interpreter over a KNOWN GRAMMAR -- the statements this emitter can
# write and no others -- rather than a C parser. Every pattern is anchored, and
# :func:`test_every_line_of_the_device_source_is_executed_or_pinned` fails if a
# line matches none of them, so the grammar cannot silently grow a statement
# nothing executes.

_TERM_CALL = re.compile(
    r"float term_(?P<tag>\w+) = folded_offdiag_term\(\s*"
    r"(?P<volume>\w+), (?P<coefficient>\w+), idx,\s*"
    r"flat\((?P<down>[^)]*), nyz, nz\),\s*"
    r"flat\((?P<up>[^)]*), nyz, nz\),\s*"
    r"flat\((?P<corner>[^)]*), nyz, nz\),\s*"
    r"(?P<lane>mg_[xyz]), (?P<weight>gw_[xyz])\);")
_TOTAL_FIRST = re.compile(r"float total_(?P<name>\w+) = term_(?P<tag>\w+);")
_TOTAL_ADD = re.compile(
    r"total_(?P<name>\w+) = total_(?P=name) \+ term_(?P<tag>\w+);")
_MASK = re.compile(r"total_(?P<name>\w+) = \((?P<wall>wm_[xyz]) && "
                   r"(?P<face>at_[xyz])\) \? 0\.0f : total_(?P=name);")
#: The ghost-lane definitions, matched by SHAPE rather than by their exact text so
#: the evaluator EXECUTES whatever the emitter wrote. Anchoring the expression
#: would make a leg that rewrote it fail as a harness error instead of as a caught
#: defect -- and "the lane is derived from ``bc_*`` and nowhere else" is exactly
#: the property the kernel's correctness rests on.
_LANE = re.compile(r"int (?P<lane>mg_[xyz]) = (?P<expr>[^;]+);")

#: The three ghost RULES, parsed out of the prelude and evaluated rather than
#: re-implemented in Python. The certified slice transcribes its two coordinate
#: helpers and pins them by exact string; that is the right balance for arithmetic
#: another suite already certified, and the WRONG one here, because the mirror arm
#: is the only genuinely new thing in this family and a pinned string is weaker
#: evidence than an executed one. Both are done: these parse, and
#: :data:`PINNED_HELPERS` still pins.
_MIRROR_ROW = re.compile(r"#define MIRROR_ROW (?P<row>\d+)")
_COORD_DN = re.compile(
    r"int coord_dn\(int a, int n, int bc\) \{\n"
    r"    if \(a > 0\) return a - 1;\n"
    r"(?P<branches>(?:    if \([^)]+\) return [^;]+;\n)*)"
    r"    return (?P<fallback>[^;]+);\n\}")
_COORD_UP = re.compile(
    r"int coord_up\(int a, int n, int bc\) \{\n"
    r"    if \(a \+ 1 < n\) return a \+ 1;\n"
    r"    return (?P<far>[^;]+);\n\}")
_BRANCH = re.compile(r"if \((?P<condition>[^)]+)\) return (?P<value>[^;]+);")
_DEVICE_CONSTANTS = {"BC_PERIODIC": 0, "BC_METALLIC": 1, "BC_MIRROR": 2}
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
_GHOST_MIRROR_BODY = re.compile(
    r"if \(index < 0\) return 0\.0f;\n"
    r"    float value = g\[index\];\n"
    r"    return (?P<expr>[^;]+);")


def _f32(values):
    """Assert float32 at every step: a silent promotion to float64 would make the
    comparison a statement about NumPy's promotion rules, not about the tree."""
    assert values.dtype == numpy.float32, values.dtype
    return values


def _as_python(expression):
    """One C expression this grammar can contain, as Python.

    ``A ? B : C`` becomes ``numpy.where(A, B, C)`` -- the ternaries here are
    per-lane selects and their conditions are arrays -- and ``&&`` becomes ``&``.
    At most one ternary per expression, which the grammar test enforces by
    covering every line.
    """
    text = expression.replace("&&", "&").replace("0.0f", "numpy.float32(0.0)")
    match = re.match(r"^\s*(?P<c>.+?)\s*\?\s*(?P<a>.+?)\s*:\s*(?P<b>.+?)\s*$", text)
    if match is None:
        return text
    return f"numpy.where({match['c']}, {match['a']}, {match['b']})"


def _ghost_rules(source):
    """``MIRROR_ROW``, ``coord_dn`` and ``coord_up``, PARSED out of the emitted text.

    THE CERTIFIED SLICE TRANSCRIBES ITS COORDINATE HELPERS INTO PYTHON AND PINS THE
    C BY EXACT STRING. That is the right balance for arithmetic another suite has
    already certified on a device, and the wrong one here: the mirror arm is the
    only new rule in this family, and a leg that plants a wrong mirror arm must be
    CAUGHT BY ARITHMETIC rather than by a string comparison, or the merge bar is
    testing that nobody edited the file rather than that the file is right.
    """
    row = _MIRROR_ROW.search(source)
    down = _COORD_DN.search(source)
    up = _COORD_UP.search(source)
    assert row and down and up, (
        "the evaluator can no longer find the ghost rules; every arithmetic leg "
        "below would then be running a transcription of a body it is not reading")
    constants = dict(_DEVICE_CONSTANTS, MIRROR_ROW=int(row["row"]), numpy=numpy)
    branches = [(match["condition"], match["value"])
                for match in _BRANCH.finditer(down["branches"])]
    fallback = down["fallback"]
    far = _as_python(up["far"])

    def coord_dn(a, n, bc):
        scope = dict(constants, a=a, n=n, bc=bc)
        for condition, value in branches:
            if eval(_as_python(condition), {"numpy": numpy}, scope):  # noqa: S307
                low = eval(_as_python(value), {"numpy": numpy}, scope)  # noqa: S307
                break
        else:
            low = eval(_as_python(fallback), {"numpy": numpy}, scope)  # noqa: S307
        return numpy.where(a > 0, a - 1, low)

    def coord_up(a, n, bc):
        scope = dict(constants, a=a, n=n, bc=bc)
        return numpy.where(a + 1 < n, a + 1,
                           eval(far, {"numpy": numpy}, scope))  # noqa: S307

    return coord_dn, coord_up


def evaluate_folded_source(source, arrays, tables, codes, walls, weights, shape):
    """Run one launch of the emitted kernel over the whole grid, in float32.

    ``arrays`` maps device parameter names to (nx, ny, nz) float32 arrays and is
    MUTATED in place, exactly as the kernel mutates its outputs.
    """
    nx, ny, nz = shape
    i, j, k = numpy.indices(shape)
    nyz = ny * nz
    home = i * nyz + j * nz + k
    coord_dn, coord_up = _ghost_rules(source)
    coordinates = {
        "i": i, "j": j, "k": k,
        "di": coord_dn(i, nx, codes[0]), "ui": coord_up(i, nx, codes[0]),
        "dj": coord_dn(j, ny, codes[1]), "uj": coord_up(j, ny, codes[1]),
        "dk": coord_dn(k, nz, codes[2]), "uk": coord_up(k, nz, codes[2])}
    wall_flags = {"wm_x": walls[0], "wm_y": walls[1], "wm_z": walls[2]}
    ghost_weights = {"gw_x": numpy.float32(weights[0]),
                     "gw_y": numpy.float32(weights[1]),
                     "gw_z": numpy.float32(weights[2])}
    faces = {"at_x": i == 0, "at_y": j == 0, "at_z": k == 0}

    body = source[source.index("int at_x"):]

    # The ghost LANES are EVALUATED off the emitted lines rather than assumed:
    # "derived from bc_* and nowhere else" is the property the kernel's
    # correctness rests on, and an evaluator that computed them independently
    # could not see it break.
    lane_scope = dict(_DEVICE_CONSTANTS, numpy=numpy)
    for axis, name in enumerate("xyz"):
        lane_scope[f"bc_{name}"] = int(codes[axis])
        lane_scope[f"wm_{name}"] = int(walls[axis])
        lane_scope[f"at_{name}"] = faces[f"at_{name}"]
    lanes = {}
    for match in _LANE.finditer(body):
        lanes[match["lane"]] = eval(  # noqa: S307 - this module's own emitted text
            _as_python(match["expr"]), {"numpy": numpy}, lane_scope)
    assert set(lanes) == {"mg_x", "mg_y", "mg_z"}, sorted(lanes)

    def flat(expression):
        a, b, c = [coordinates[name.strip()] for name in expression.split(",")]
        return numpy.where((a < 0) | (b < 0) | (c < 0), -1, a * nyz + b * nz + c)

    def ghosted(volume, index):
        """``ghosted`` (PINNED by exact string below), transcribed."""
        flat_volume = numpy.asarray(volume).reshape(-1)
        return _f32(numpy.where(index < 0, numpy.float32(0.0),
                                flat_volume[numpy.maximum(index, 0)]))

    mirror_body = _GHOST_MIRROR_BODY.search(source)
    assert mirror_body is not None, (
        "the evaluator can no longer find ghosted_mirror's body; the parity "
        "weight would then be a transcription nothing reads")
    mirror_expression = _as_python(mirror_body["expr"])

    def ghosted_mirror(volume, index, mg, w):
        """``ghosted_mirror`` -- this family's new helper, its RETURN EXPRESSION
        parsed out of the emitted text and evaluated, so a leg that rewrites the
        parity rule is caught by arithmetic rather than by a string comparison."""
        flat_volume = numpy.asarray(volume).reshape(-1)
        value = _f32(flat_volume[numpy.maximum(index, 0)])
        weighted = _f32(eval(  # noqa: S307 - this module's own emitted text
            mirror_expression, {"numpy": numpy},
            {"value": value, "mg": mg, "w": numpy.float32(w), "numpy": numpy}))
        return _f32(numpy.where(index < 0, numpy.float32(0.0), weighted))

    scalars = {}
    for match in _GS.finditer(body):
        scalars[f"gs_{match['name']}"] = _f32(numpy.asarray(arrays[match["volume"]]))
    for match in _US.finditer(body):
        scalars[f"us_{match['name']}"] = _f32(numpy.asarray(arrays[match["volume"]]))

    # folded_offdiag_term's three arithmetic lines, PARSED AND EVALUATED as
    # expressions -- the certified slice's device, extended by two arguments.
    near_text, far_text, term_text = _TERM_BODY.search(source).groups()

    def folded_offdiag_term(volume, coefficient, down, up, corner, mg, w):
        environment = {"g": volume, "u": coefficient, "home": home,
                       "down": down, "up": up, "corner": corner,
                       "mg": mg, "w": w,
                       "ghosted": ghosted, "ghosted_mirror": ghosted_mirror}

        def evaluate(text):
            python = text.replace("0.25f", "numpy.float32(0.25)")
            python = re.sub(r"\b(g|u)\[home\]", r"ghosted(\1, home)", python)
            return _f32(eval(python, {"numpy": numpy}, environment))

        environment["near_pair"] = evaluate(near_text)
        environment["far_pair"] = evaluate(far_text)
        return evaluate(term_text)

    terms, totals, sources = {}, {}, {}
    for match in _TERM_CALL.finditer(body):
        terms[match["tag"]] = folded_offdiag_term(
            numpy.asarray(arrays[match["volume"]]),
            numpy.asarray(arrays[match["coefficient"]]),
            flat(match["down"]), flat(match["up"]), flat(match["corner"]),
            lanes[match["lane"]], ghost_weights[match["weight"]])
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
        # constitutive_apply (PINNED by exact string, and welded to the certified
        # sibling's): prev is read BEFORE fw is written, then two separate
        # accumulations left to right.
        previous = numpy.asarray(aux).copy()
        source_values = sources[match["name"]]
        aux[...] = source_values
        accumulated = _f32(numpy.asarray(field) + _f32(kps * source_values))
        field[...] = _f32(accumulated - _f32(kms * previous))


# --------------------------------------------------------------------------
# The fixture: one frozen state, run twice
# --------------------------------------------------------------------------

def run_both_paths(mask, symmetry=(), phase=1, termination="periodic", steps=2,
                   source=None, seed=20260820, boundaries=None,
                   codes_override=None, weights_override=None,
                   emitter="folded"):
    """``stepping.update_E`` and the emitted source from ONE frozen state.

    Returns ``(oracle_bits, produced_bits, floors)``. ``floors`` carries the three
    non-vacuity measurements every leg asserts, because a leg that cannot fail
    certifies nothing:

    * ``oracle_moved`` -- zero-init is a FIXED POINT of this recurrence, so a case
      that moved nothing passes for every tree;
    * ``coupling_max_abs`` -- what the off-diagonal rows actually contributed. A
      case whose coupling is identically zero is testing the PLAIN constitutive
      kernel with extra steps;
    * ``folded_axis_deviation`` and ``mirror_ghost_min_abs`` -- a folded axis whose
      half-integer profile is the identity everywhere cannot show a
      coefficient-index error on the axis the fold moved, and a mirror ghost whose
      source row is identically zero EQUALS the metallic zero it is being
      distinguished from.

    ``emitter="certified"`` runs the CERTIFIED family's source through the
    CERTIFIED family's evaluator on the same frozen state -- the only way the
    reduction leg can compare two trees rather than two runs. It is unfolded-only
    by construction: ``coverage.BC_CODES`` has no ``mirror`` entry and the KeyError
    is deliberately left to escape.

    PLAIN NUMPY, NOT AN ``xp`` FIXTURE: what is measured is a float32 expression
    tree, which is the same tree whatever allocated the arrays. The backend name
    is the PREDICATE legs' question, not this one's.
    """
    from .test_offdiag_constitutive_pml_real import (  # noqa: PLC0415
        evaluate_emitted_source)
    planes = tuple(_mirror(axis_name, phase) for axis_name in symmetry)
    if boundaries is None:
        boundaries = tuple(termination if "xyz"[axis] in symmetry else "periodic"
                           for axis in range(3))
    fields, layer, grid = build(numpy, boundaries=boundaries, symmetry=planes,
                                cell=_fold_cell(symmetry))
    rows = install_rows(fields, grid, mask, seed=seed)
    seed_state(fields, grid, seed=seed % 1000)
    frozen = {name: numpy.asarray(getattr(fields, name)).copy()
              for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                           "f_w_Ex", "f_w_Ey", "f_w_Ez")}

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
    moved = sum(numpy.asarray(getattr(fields, name)).tobytes()
                != frozen[name].tobytes()
                for name in ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"))
    coupling = float(numpy.abs(
        numpy.asarray(fields.f_w_Ex)
        - numpy.asarray(fields.Dx) * numpy.asarray(
            fields.inverse_epsilon_for("Ex"))).max())

    for name, values in frozen.items():
        getattr(fields, name)[...] = values
    walls = coverage.offdiag_wall_mask_flags(grid)
    weights = (tuple(weights_override) if weights_override is not None
               else folded.mirror_ghost_weights(grid))
    if emitter == "certified":
        codes = tuple(coverage.BC_CODES[kind]
                      for kind in coverage.real_pml_boundary_kinds(grid))
        text = offdiag_emitter.offdiag_source(mask) if source is None else source
    else:
        codes = (tuple(codes_override) if codes_override is not None
                 else folded.folded_offdiag_boundary_codes(grid))
        text = folded.folded_offdiag_source(mask) if source is None else source
    arrays = _device_arrays(fields, rows)
    for _ in range(steps):
        if emitter == "certified":
            evaluate_emitted_source(text, arrays, _tables_for(layer), codes,
                                    walls, grid.shape)
        else:
            evaluate_folded_source(text, arrays, _tables_for(layer), codes,
                                   walls, weights, grid.shape)
        _advance_sources(fields)
        arrays = _device_arrays(fields, rows)
    floors = {"oracle_moved": moved, "coupling_max_abs": coupling,
              "folded_axis_deviation": deviation, "mirror_ghost_min_abs": ghost,
              "shape": tuple(grid.shape), "codes": codes, "walls": walls,
              "weights": weights}
    return oracle, _bits(fields), floors


def _mirror(axis_name, phase):
    from ..grid import Mirror  # noqa: PLC0415 - one import site for the fixture

    return Mirror(axis_name.upper(), phase)


def _assert_floors(floors, folded_expected=True):
    assert floors["oracle_moved"] > 0, (
        "the array path changed no output word: zero-init is a fixed point of "
        "this recurrence and a case that moved nothing certifies nothing")
    assert floors["coupling_max_abs"] > 0.0, (
        "the off-diagonal coupling is identically zero: this case cannot "
        "distinguish any defect in this family")
    if folded_expected:
        assert floors["folded_axis_deviation"] > 0.0, (
            "the folded axis's half-integer profile is the identity everywhere; "
            "a coefficient-index error on the axis the fold moved is invisible")
        assert (floors["mirror_ghost_min_abs"] or 0.0) > 0.0, (
            "stored row 2 of a partner volume is identically zero on a folded "
            "axis, so the mirror ghost equals the metallic zero ghost")


# ==========================================================================
# THE TRANSCRIBED CONSTANTS
# ==========================================================================

def test_the_mirror_source_row_is_steppings():
    """Restated for the package's engine-import-free contract, pinned here so the
    two cannot drift. Off by one is a half-cell registration error on the fold
    plane: smooth, converged and wrong."""
    assert folded.MIRROR_SOURCE_INDEX == stepping.MIRROR_SOURCE_INDEX
    assert "#define MIRROR_ROW 2" in folded.prelude()


def test_the_boundary_codes_are_the_certified_table_plus_mirror():
    """The seam this family exists across. ``offdiag_boundary_codes`` maps through
    ``coverage.BC_CODES``, which has no ``mirror`` entry and deliberately lets the
    KeyError escape -- the second of the two reasons
    ``FOLDED_OFFDIAG_ROW_MASK_ADMISSION`` gives for not widening the certified
    predicate. This family adds the code the launcher needs, and adds NOTHING to
    the certified table."""
    assert folded.FOLDED_BC_CODES == dict(coverage.BC_CODES, mirror=2)
    for kind, code in coverage.BC_CODES.items():
        assert folded.FOLDED_BC_CODES[kind] == code
    assert "mirror" not in coverage.BC_CODES


def test_the_tables_are_the_certified_families_own_objects():
    """One home per fact. Imported by value rather than re-typed, so a rename in
    ``offdiag_emitter`` fails here instead of forking the arithmetic."""
    assert folded.E_TERMS is offdiag_emitter.E_TERMS
    assert folded.ROW_PARAMETERS is offdiag_emitter.ROW_PARAMETERS
    assert folded.PARTNER_VOLUMES is offdiag_emitter.PARTNER_VOLUMES
    assert folded.LIVE_ROW_MASKS is offdiag_emitter.LIVE_ROW_MASKS
    assert folded.normalized_row_mask is not offdiag_emitter.normalized_row_mask
    assert (folded.normalized_row_mask((1, 0, 0, 0, 0, 0))
            == offdiag_emitter.normalized_row_mask((1, 0, 0, 0, 0, 0)))


def test_the_family_does_not_wear_the_certified_kernels_name():
    """Two kernels sharing a name would make ``certification.json``'s partition and
    any NVRTC binary observation ambiguous about which body they saw."""
    assert folded.KERNEL_NAME != offdiag_emitter.KERNEL_NAME
    for mask in CORPUS_ROW_MASKS:
        assert folded.shipped_kernel_names(
            folded.folded_offdiag_source(mask)) == {folded.KERNEL_NAME}


def test_the_compile_options_are_the_certified_siblings_options():
    """``--fmad=false`` is CORRECTNESS on this sub-step, not tuning: the certified
    sibling measured its unguarded control diverging on 384/384, and this family's
    row product is that one plus a select.

    Read off the sibling's SOURCE rather than imported, because importing that
    module needs CuPy and this file is the merge bar on a laptop.
    """
    sibling = (HERE / "offdiag_constitutive_kernels.py").read_text(encoding="utf-8")
    assert "_COMPILE_OPTIONS = ('--fmad=false',)" in sibling
    assert folded._COMPILE_OPTIONS == ('--fmad=false',)


def test_the_ghost_weight_is_minus_the_plane_phase_on_every_axis():
    """THE ONE SIGNED SCALAR THE WHOLE PARITY INPUT COLLAPSES TO.

    ``_shift_down`` is called with ``component = "D" + AXIS_NAMES[partner_axis]``
    on ``axis = partner_axis`` (stepping.py:1243-1245), and every D component has
    Yee shift 1 on its own axis, so ``mirror_parity`` collapses to ``-phase`` for
    every partner and every axis. Measured through ``fields.mirror_parity``
    itself, exhaustively, rather than transcribed.
    """
    for axis in range(3):
        assert IYEE_SHIFTS["D" + "xyz"[axis]][axis] == 1
        for phase in (1, -1):
            assert mirror_parity("D" + "xyz"[axis], axis, phase) == -phase


def test_the_launched_weight_is_the_derivation_and_not_a_default(xp):
    """The value a launch binds, per axis, against the collapse -- and ``1.0`` on
    an unfolded axis, where the lane never fires."""
    for phase in (1, -1):
        for axis_name in ("x", "y", "z"):
            _f, _l, grid = build(xp, symmetry=(_mirror(axis_name, phase),),
                                 cell=_fold_cell((axis_name,)))
            weights = folded.mirror_ghost_weights(grid)
            axis = "xyz".index(axis_name)
            assert weights[axis] == float(-phase)
            for other in range(3):
                if other != axis:
                    assert weights[other] == 1.0


def test_an_unreadable_mirror_phase_is_a_nan_the_predicate_names(xp):
    """The weight is a RUNTIME argument, so a plane that cannot be read would
    LAUNCH a NaN rather than raise. It comes back NaN here and the predicate
    refuses it by name."""

    class _Broken:
        def __init__(self, grid):
            self._grid = grid

        def __getattr__(self, item):
            return getattr(self._grid, item)

        def mirror_phase(self, axis):
            return 0

    fields, layer, grid = build(xp, symmetry=(_mirror("x", 1),),
                                cell=_fold_cell(("x",)))
    install_rows(fields, grid, (1, 0, 0, 1, 0, 0))
    broken = _Broken(grid)
    weights = folded.mirror_ghost_weights(broken)
    assert weights[0] != weights[0], weights
    covered, reason = folded.covers_folded_offdiag_constitutive(
        fields, layer, broken)
    assert not covered and "ghost weight" in reason, reason


# ==========================================================================
# THE DEVICE TEXT
# ==========================================================================

PINNED_HELPERS = {
    "coord_up": """__device__ __forceinline__ int coord_up(int a, int n, int bc) {
    if (a + 1 < n) return a + 1;
    return (bc == BC_PERIODIC) ? 0 : -1;
}""",
    "coord_dn": """__device__ __forceinline__ int coord_dn(int a, int n, int bc) {
    if (a > 0) return a - 1;
    if (bc == BC_METALLIC) return -1;
    if (bc == BC_MIRROR) return MIRROR_ROW;
    return n - 1;
}""",
}


@pytest.mark.parametrize("name", sorted(PINNED_HELPERS))
def test_the_hand_transcribed_helpers_are_pinned(name):
    assert PINNED_HELPERS[name] in folded.prelude(), (
        f"{name} has changed; the evaluator's Python transcription of it no "
        f"longer describes the shipped text")


@pytest.mark.parametrize("name", folded.SIBLING_HELPERS)
def test_the_shared_helpers_are_the_certified_familys_bytes(name):
    """READ out of ``offdiag_emitter.PRELUDE``, never copied.

    Two copies of ``flat``, ``ghosted`` or the welded ``constitutive_apply`` tail
    are equal only until someone edits one, and a silent divergence there would be
    a wrong answer no diff of ``folded_offdiag_kernels.py`` would show. This
    asserts the extracted block is still in the sibling AND still lands in every
    emitted source.
    """
    block = folded._sibling_block(name)
    assert block in offdiag_emitter.PRELUDE
    assert f" {name}(" in block
    for mask in CORPUS_ROW_MASKS:
        assert block in folded.folded_offdiag_source(mask)


def test_a_renamed_sibling_helper_fails_loudly_rather_than_forking():
    """The indirection's whole point: a silent fallback to a local copy is the one
    failure it exists to prevent."""
    with pytest.raises(RuntimeError, match="no longer declares"):
        folded._sibling_block("a_helper_that_does_not_exist")


def test_the_welded_tail_is_character_for_character_the_certified_siblings():
    """``constitutive_apply`` is shared by three families now. The weld is the
    extraction itself, and this states it as an assertion so a future edit that
    inlines a copy fails here."""
    tail = folded._sibling_block("constitutive_apply")
    assert "float prev = fw[idx];" in tail and "fw[idx] = src;" in tail
    constitutive = (HERE / "constitutive_kernels.py").read_text(encoding="utf-8")
    body = tail[tail.index("__device__"):]
    assert body in constitutive, (
        "the certified plain constitutive pair no longer carries this exact tail")


def test_every_emitted_source_is_pure_ascii_and_survives_an_ascii_locale_write():
    """A COMPILE REQUIREMENT, not a style rule. ``compile_using_nvrtc`` writes the
    source through a bare ``open(..., 'w')``, so the bytes go through the
    interpreter's LOCALE encoding -- ASCII under C/POSIX, which is what a
    non-interactive shell on the validation host gets. Two em-dashes in a comment
    killed the sibling track's E kernel at its first launch on 2026-08-15."""
    for mask in folded.LIVE_ROW_MASKS:
        text = folded.folded_offdiag_source(mask)
        text.encode("ascii")
        handle = io.TextIOWrapper(io.BytesIO(), encoding="ascii")
        handle.write(text)
        handle.flush()


def test_there_are_sixty_three_live_row_masks_and_the_all_dead_one_is_refused():
    assert len(folded.LIVE_ROW_MASKS) == 63
    with pytest.raises(ValueError, match="no row slot survives"):
        folded.folded_offdiag_source((0, 0, 0, 0, 0, 0))


def test_the_boundary_wall_and_weight_axes_are_runtime_arguments():
    """THE SPECIALIZATION SPLIT, measured. The row mask is the emitter's only
    parameter; the boundary codes, wall flags and ghost weights ride the
    signature. Baked in, the source count would be 64x63x8 and every one of them
    would owe a byte gate."""
    for mask in CORPUS_ROW_MASKS:
        text = folded.folded_offdiag_source(mask)
        assert "int bc_x, int bc_y, int bc_z," in text
        assert "int wm_x, int wm_y, int wm_z," in text
        assert "float gw_x, float gw_y, float gw_z" in text
    assert folded.folded_offdiag_source.__code__.co_argcount == 1


def test_only_the_live_coefficients_are_declared():
    """No pointer bound to an array the kernel must never touch -- the certified
    emitter's choice, kept."""
    for mask in folded.LIVE_ROW_MASKS:
        text = folded.folded_offdiag_source(mask)
        for slot, flag in enumerate(mask):
            declaration = f"    const float* {folded.ROW_PARAMETERS[slot]},"
            assert (declaration in text) == bool(flag), (mask, slot)


def test_the_ghost_lane_is_derived_from_the_boundary_code_and_nowhere_else():
    """ONE FACT, ONE PLACE. The index redirect keys off ``bc_*`` and the parity
    weight off ``mg_*``; if the two could disagree the kernel would read stored
    row 2 without the parity, or apply the parity to an ordinary neighbour -- a
    plane of wrong values, not a crash. The sibling track carries the same fact
    twice as a constexpr; here it is derived in one line."""
    for mask in folded.LIVE_ROW_MASKS:
        text = folded.folded_offdiag_source(mask)
        for axis_name in "xyz":
            line = (f"    int mg_{axis_name} = (bc_{axis_name} == BC_MIRROR) "
                    f"&& at_{axis_name};")
            assert line in text
        # and no OTHER definition of a lane exists
        assert len(re.findall(r"int mg_[xyz] =", text)) == 3


def _code_only(text):
    """The emitted source with its ``//`` comments removed.

    Every leg that asserts something is ABSENT reads this rather than the raw
    text: the comments name the very things the code must not contain, so scanning
    them would make an absence test fail on its own documentation.
    """
    return "\n".join(line.split("//")[0] for line in text.split("\n"))


def test_the_ghost_rules_are_the_three_stepping_branches_and_no_others():
    """CYL_AXIS is refused by the predicate and real storage cannot carry a Bloch
    phase, so those branches must not exist in the device text: a branch nothing
    can reach is a branch nothing gates."""
    for mask in CORPUS_ROW_MASKS:
        text = _code_only(folded.folded_offdiag_source(mask))
        for absent in ("CYL", "bloch", "dtdx", "reflect"):
            assert absent not in text, absent
        for present in ("BC_METALLIC", "BC_PERIODIC", "BC_MIRROR", "MIRROR_ROW"):
            assert present in text, present


def test_every_line_of_the_device_source_is_executed_or_pinned():
    """NOTHING IN THE SHIPPED TEXT IS NEITHER RUN NOR PINNED.

    The evaluator's grammar covers the per-component statements and the ghost
    lanes; the pinned helpers and the extracted sibling blocks cover the prelude;
    the template's fixed scaffolding is listed here explicitly. A line matching
    none of the three is a line no leg in this file can see, and this is where
    that is caught.
    """
    scaffolding = (
        'extern "C" __global__', "float* __restrict__", "const float*",
        "int nx, int ny, int nz,", "int bc_x", "int wm_x", "float gw_x",
        "int idx = blockIdx.x", "if (idx >= nx * ny * nz) return;",
        "int nyz = ny * nz;", "int k = idx % nz;", "int j = (idx / nz) % ny;",
        "int i = idx / (ny * nz);", "int di = coord_dn", "int dj = coord_dn",
        "int dk = coord_dn", "int at_x = (i == 0)", ") {", "}", "{",
    )
    grammar = (_TERM_CALL, _TOTAL_FIRST, _TOTAL_ADD, _MASK, _LANE, _GS, _US,
               _SRC_COUPLED, _SRC_PLAIN, _TAIL)
    for mask in folded.LIVE_ROW_MASKS:
        text = folded.folded_offdiag_source(mask)
        body = text[text.index('extern "C"'):]
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
    """The same claim for the OTHER half of the source.

    ``folded_offdiag_term``'s three arithmetic lines and ``ghosted_mirror``'s
    three are EXECUTED (parsed and evaluated as expressions); ``coord_up`` and
    ``coord_dn`` are PINNED by exact string; ``flat``, ``ghosted`` and
    ``constitutive_apply`` are the certified family's extracted bytes, which its
    own suite executes and pins. Everything else is a preprocessor line, a
    signature or a brace, listed here.
    """
    text = folded.prelude()
    executed = _TERM_BODY.search(text)
    assert executed is not None, (
        "the evaluator can no longer find folded_offdiag_term's arithmetic "
        "lines; every arithmetic leg below is then running a transcription of a "
        "body it is not reading")
    mirror_body = _GHOST_MIRROR_BODY.search(text)
    assert mirror_body is not None, (
        "the evaluator can no longer find ghosted_mirror's body; the parity "
        "weight would then be a transcription nothing reads")
    pinned = "\n".join(PINNED_HELPERS.values())
    inherited = "\n".join(folded._sibling_block(name)
                          for name in folded.SIBLING_HELPERS)
    scaffolding = ("#define ", "__device__ __forceinline__", "const float*",
                   "int a, int n, int bc", "int i, int j, int k, int nyz, int nz",
                   "float kps, float kms", "int mg, float w", ") {", "{", "}")
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if (stripped in executed.group(0) or stripped in mirror_body.group(0)
                or stripped in pinned or stripped in inherited):
            continue
        assert any(token in line for token in scaffolding), (
            f"no leg in this file executes or pins {stripped!r}")


# ==========================================================================
# THE ARITHMETIC, measured against stepping.py -- on a laptop, in float32
# ==========================================================================

_FOLDED_CASES = [(mask, symmetry, phase, termination)
                 for mask in ROW_MASKS
                 for symmetry in FOLD_SYMMETRIES if symmetry
                 for phase in FOLD_PHASES
                 for termination in FOLD_TERMINATIONS]


@pytest.mark.parametrize("mask,symmetry,phase,termination", _FOLDED_CASES)
def test_the_emitted_source_is_bit_identical_to_stepping_on_a_fold(
        mask, symmetry, phase, termination):
    """THE MEASUREMENT THIS FAMILY EXISTS FOR.

    ``stepping.update_E`` on a real folded ``Grid``/``Fields``/``PML`` triple with
    the rows installed through the PUBLIC installer, against the emitted device
    source executed in float32 NumPy, from one frozen state, two chained launches.
    Every axis folded somewhere, both plane parities, both declared terminations,
    four row masks -- so the fold lands in the PARTNER role, in the OWN-axis role
    and in neither.
    """
    oracle, produced, floors = run_both_paths(mask, symmetry, phase, termination)
    _assert_floors(floors)
    for name in oracle:
        assert oracle[name] == produced[name], (
            f"{name} differs on {symmetry} phase={phase} {termination} "
            f"mask={mask}")


def test_the_transcription_holds_over_the_multi_step_budget():
    """A divergence that only appears from step two is the one a single launch
    cannot see: ``f_w`` starts nonzero here so the ``kms * prev`` term is live
    from launch one, but the RECURRENCE only compounds over steps. 60 is the
    budget every certified hand-CUDA record is cut at."""
    oracle, produced, floors = run_both_paths(
        (1, 1, 1, 1, 1, 1), ("x", "y"), phase=-1, termination="metallic",
        steps=MULTI_STEP_BUDGET)
    _assert_floors(floors)
    assert oracle == produced


def test_the_two_fold_terminations_are_bit_identical_here():
    """FACT 3, MEASURED. The curl pair splits a folded PERIODIC axis from a folded
    METALLIC one because it masks the folded-periodic top plane and reflects past
    it. ``update_E`` has no ownership mask (``_mask_non_owned_cells`` has three
    call sites and no constitutive function is among them) and, because
    ``_offdiagonal_terms`` passes no ``reflect_row``, no reflect row either. So
    ONE code serves both -- and this measures the array path itself agreeing,
    which is the claim, rather than the kernel agreeing with itself.
    """
    for mask in CORPUS_ROW_MASKS:
        for symmetry in (("x",), ("z",)):
            runs = {}
            for termination in FOLD_TERMINATIONS:
                oracle, produced, floors = run_both_paths(
                    mask, symmetry, phase=1, termination=termination)
                _assert_floors(floors)
                assert oracle == produced
                runs[termination] = oracle
            # The two grids differ in extent (a folded metallic axis stores one
            # cell fewer), so the BYTES cannot be compared; what is compared is
            # that ONE boundary code served both and both reproduced stepping.
            assert set(runs) == set(FOLD_TERMINATIONS)


@pytest.mark.parametrize("mask", ROW_MASKS)
@pytest.mark.parametrize("boundaries", [
    ("periodic", "periodic", "periodic"),
    ("metallic", "metallic", "periodic"),
    ("metallic", "periodic", "metallic"),
    ("metallic", "metallic", "metallic"),
])
def test_an_unfolded_grid_reduces_to_the_certified_family_bit_for_bit(
        mask, boundaries):
    """THE REDUCTION, MEASURED RATHER THAN CLAIMED.

    With no axis folded every ``mg`` lane is 0, ``ghosted_mirror`` returns
    ``ghosted``'s value BY BRANCH rather than by a multiply a compiler is trusted
    to fold away, and the emitted body must therefore be the certified family's
    arithmetic exactly. Both trees are executed on the SAME frozen state through
    the SAME evaluator machinery and compared as bytes.

    This is also the disjointness argument's other half: an unfolded run is the
    certified family's, and here that is a byte statement rather than a policy.
    """
    oracle_flat, produced_flat, floors_flat = run_both_paths(
        mask, symmetry=(), boundaries=boundaries, steps=4, seed=20260820,
        emitter="certified")
    oracle, produced, floors = run_both_paths(
        mask, symmetry=(), boundaries=boundaries, steps=4, seed=20260820)
    _assert_floors(floors, folded_expected=False)
    _assert_floors(floors_flat, folded_expected=False)
    assert oracle == oracle_flat, "the two fixtures did not build the same run"
    assert produced_flat == oracle_flat, (
        "the certified family is not bit-identical on its own unfolded run; the "
        "reduction leg would then be comparing against a broken reference")
    assert produced == produced_flat, (
        f"the folded emitter's unfolded arithmetic is not the certified "
        f"family's on {boundaries} mask={mask}")
    assert oracle == produced


def test_the_fold_plane_coupling_is_alive_and_is_the_mirror_ghosts():
    """THE ARM THE CERTIFIED KERNEL CANNOT SERVE, isolated.

    On a fold whose axis is a live slot's PARTNER, plane 0 of that axis carries a
    coupling computed from ``parity * g[2]``. Handing the kernel METALLIC instead
    -- which is the shipped kernel's only available answer -- must MOVE BYTES, and
    move them ON THAT PLANE. If it did not, this family would be worth nothing and
    the 2026-08-20 device sweep would have been measuring noise.
    """
    mask = (1, 0, 0, 1, 0, 0)          # slots (Ex,Ey) and (Ey,Ex): partners Y and X
    oracle, correct, floors = run_both_paths(mask, ("x",), phase=1)
    _assert_floors(floors)
    assert oracle == correct
    metallic = tuple(coverage.BC_CODES["metallic"] if code == folded.BC_MIRROR_CODE
                     else code for code in floors["codes"])
    _o, wrong, _f = run_both_paths(mask, ("x",), phase=1,
                                   codes_override=metallic)
    assert wrong != correct, (
        "handing the folded axis the metallic code changed nothing; the mirror "
        "ghost is then not observable in this fixture and every leg below is "
        "measuring the fixture")
    # And the difference is ON the fold plane of the row whose PARTNER axis is X.
    shape = floors["shape"]
    for name in ("Ey", "f_w_Ey"):
        a = numpy.frombuffer(correct[name], dtype=numpy.float32).reshape(shape)
        b = numpy.frombuffer(wrong[name], dtype=numpy.float32).reshape(shape)
        differing = a.view(numpy.uint32) != b.view(numpy.uint32)
        assert differing[0].any(), name
        assert not differing[1:].any(), (
            f"{name} differs off the fold plane; the mirror ghost reaches "
            f"exactly one plane (stepping.py:1870-1872)")


# ==========================================================================
# PLANTED DEFECTS
# ==========================================================================
#
# Each is a SILENT wrong answer -- a smooth, converged, plausible field with the
# wrong tensor in it -- and none would be caught by a magnitude comparison at any
# tolerance a physicist would accept. Every leg is scoped to a case whose
# UNMUTATED baseline is bit-identical, because a defect planted on an already
# diverging baseline scores CAUGHT for free.


def _mutate(text, kind):
    """One planted defect in the emitted device source. Returns ``(text, sites)``."""
    needle = "if (bc == BC_MIRROR) return MIRROR_ROW;"
    if kind == "mirror_ghost_as_the_metallic_zero":
        # The SHIPPED kernel's answer for a folded axis handed BC_METALLIC,
        # planted here as the defect it is. This is the arm the 2026-08-20 device
        # sweep measured DIVERGENT on 352/352.
        return (text.replace(needle, "if (bc == BC_MIRROR) return -1;"),
                text.count(needle))
    if kind == "mirror_ghost_as_the_periodic_wrap":
        return (text.replace(needle, "if (bc == BC_MIRROR) return n - 1;"),
                text.count(needle))
    if kind == "mirror_row_off_by_one":
        return (text.replace("#define MIRROR_ROW 2", "#define MIRROR_ROW 1"),
                text.count("#define MIRROR_ROW 2"))
    if kind == "drop_the_parity_weight":
        return (text.replace("return mg ? (w * value) : value;",
                             "return value;"),
                text.count("return mg ? (w * value) : value;"))
    if kind == "flip_the_parity_sign":
        return (text.replace("return mg ? (w * value) : value;",
                             "return mg ? (-w * value) : value;"),
                text.count("return mg ? (w * value) : value;"))
    if kind == "weight_every_lane":
        return (text.replace("return mg ? (w * value) : value;",
                             "return w * value;"),
                text.count("return mg ? (w * value) : value;"))
    if kind == "folded_own_axis_wraps":
        return (text.replace("return (bc == BC_PERIODIC) ? 0 : -1;",
                             "return (bc == BC_METALLIC) ? -1 : 0;"),
                text.count("return (bc == BC_PERIODIC) ? 0 : -1;"))
    if kind == "weight_the_own_axis_up_leg":
        return (text.replace(
            "float far_pair = ghosted(g, up) + ghosted_mirror(g, corner, mg, w);",
            "float far_pair = ghosted_mirror(g, up, mg, w)"
            " + ghosted_mirror(g, corner, mg, w);"),
            text.count("float far_pair = ghosted(g, up)"))
    if kind == "ghost_lane_from_the_wall_flag":
        # The ``wm_*``/``bc_*`` split collapsed: the ghost lane read off the
        # DECLARATION instead of the resolved ghost rule. A folded axis reports
        # ``wm = 0``, so the parity would never be applied.
        return re.subn(
            r"int mg_([xyz]) = \(bc_[xyz] == BC_MIRROR\) && at_[xyz];",
            r"int mg_\1 = wm_\1 && at_\1;", text)
    if kind == "same_direction_shifts":
        rename = {"di": "ui", "dj": "uj", "dk": "uk"}
        return re.subn(r"flat\((\w+), (\w+), (\w+), nyz, nz\)",
                       lambda m: "flat({}, {}, {}, nyz, nz)".format(
                           *[rename.get(g, g) for g in m.groups()]), text)
    if kind == "hoist_the_coefficient":
        needle = ("return 0.25f * ((near_pair * u[home])"
                  " + (far_pair * ghosted(u, up)));")
        return (text.replace(
            needle, "return 0.25f * ((near_pair + far_pair) * u[home]);"),
            text.count(needle))
    if kind == "distribute_the_quarter":
        needle = ("return 0.25f * ((near_pair * u[home])"
                  " + (far_pair * ghosted(u, up)));")
        return (text.replace(
            needle,
            "return (0.25f * (near_pair * u[home]))"
            " + (0.25f * (far_pair * ghosted(u, up)));"),
            text.count(needle))
    if kind == "commute_the_near_product":
        # NULL: IEEE multiply commutes bitwise. Its discriminator is
        # ``hoist_the_coefficient``, which edits the same expression's
        # ASSOCIATION and must be caught -- so the pair says the comparator is
        # sensitive to grouping and insensitive to operand order, which is a
        # measurement rather than a claim.
        return (text.replace("(near_pair * u[home])", "(u[home] * near_pair)"),
                text.count("(near_pair * u[home])"))
    raise ValueError(f"unknown mutation {kind!r}")


#: Defects that MUST be caught on a fold whose axis is a live slot's PARTNER.
CAUGHT_ON_A_LIVE_PARTNER = (
    "mirror_ghost_as_the_metallic_zero",
    "mirror_ghost_as_the_periodic_wrap",
    "mirror_row_off_by_one",
    "flip_the_parity_sign",
    "ghost_lane_from_the_wall_flag",
    "same_direction_shifts",
    "hoist_the_coefficient",
)

#: Defects that MUST be caught wherever a folded axis is some component's OWN
#: axis, and are a NULL where it is not.
CAUGHT_ON_A_LIVE_OWN_AXIS = ("folded_own_axis_wraps",)

#: NULLS, each paired with a leg editing the same kind of expression that must be
#: caught. A battery of only-must-be-caught legs scores identically whether the
#: comparator works or has degenerated into failing everything.
PREDICTED_NULLS = ("distribute_the_quarter", "commute_the_near_product")


@pytest.mark.parametrize("kind", CAUGHT_ON_A_LIVE_PARTNER)
def test_the_planted_defect_is_caught_on_a_live_partner_fold(kind):
    """Armed on the corpus's own mask with a fold on a live PARTNER axis, an EVEN
    plane (weight -1) so the parity is observable, and a baseline measured
    bit-identical first."""
    mask = (1, 0, 0, 1, 0, 0)
    oracle, clean, floors = run_both_paths(mask, ("x",), phase=1)
    _assert_floors(floors)
    assert oracle == clean, "the baseline is not bit-identical; the leg is void"
    text, sites = _mutate(folded.folded_offdiag_source(mask), kind)
    assert sites > 0, f"{kind} matched nothing in the emitted source"
    _o, produced, _f = run_both_paths(mask, ("x",), phase=1, source=text)
    assert produced != oracle, f"{kind} changed no bit"


@pytest.mark.parametrize("kind", CAUGHT_ON_A_LIVE_OWN_AXIS)
def test_the_own_axis_defect_discriminates_on_the_role_the_fold_plays(kind):
    """A DISCRIMINATOR, not a catch. ``coord_up``'s mirror arm only matters where
    the folded axis is some live component's OWN axis; on a fold that is only a
    partner it is a genuine NULL, and scoring it a miss there would be a statement
    about the fixture."""
    # (1, 1, 0, 0, 0, 0) is row Ex alone: X is its OWN axis and never its partner.
    catch_mask, catch_fold = (1, 1, 0, 0, 0, 0), ("x",)
    oracle, clean, floors = run_both_paths(catch_mask, catch_fold, phase=1)
    _assert_floors(floors)
    assert oracle == clean
    text, sites = _mutate(folded.folded_offdiag_source(catch_mask), kind)
    assert sites > 0
    _o, produced, _f = run_both_paths(catch_mask, catch_fold, phase=1, source=text)
    assert produced != oracle, f"{kind} was not caught where the fold is an OWN axis"

    # (0, 0, 1, 0, 0, 0) is the single slot Ey <- Ez: own axis Y, partner Z, so a
    # folded X is in NEITHER role and every arm of this family must be inert.
    null_mask, null_fold = (0, 0, 1, 0, 0, 0), ("x",)
    oracle, clean, floors = run_both_paths(null_mask, null_fold, phase=1)
    assert oracle == clean
    text, _sites = _mutate(folded.folded_offdiag_source(null_mask), kind)
    _o, produced, _f = run_both_paths(null_mask, null_fold, phase=1, source=text)
    assert produced == oracle, (
        f"{kind} moved bytes on a fold that is in NEITHER role for the live "
        f"slot; the mirror rule reaches this sub-step through the partner-axis "
        f"down shift alone (stepping.py:1243-1245)")


def test_dropping_the_parity_weight_discriminates_on_the_plane_phase():
    """THE SHARPEST LEG IN THIS FILE, and the reason both parities are swept.

    The weight is ``-phase``. On an EVEN plane it is -1 and dropping it must be
    caught; on an ODD plane it is +1, dropping it is the identity, and the leg
    must come back a NULL. A battery run at one parity would score this a catch
    and learn nothing about whether the SIGN is right.
    """
    mask = (1, 0, 0, 1, 0, 0)
    for phase, must_be_caught in ((1, True), (-1, False)):
        oracle, clean, floors = run_both_paths(mask, ("x",), phase=phase)
        _assert_floors(floors)
        assert oracle == clean
        text, sites = _mutate(folded.folded_offdiag_source(mask),
                              "drop_the_parity_weight")
        assert sites > 0
        _o, produced, _f = run_both_paths(mask, ("x",), phase=phase, source=text)
        assert (produced != oracle) == must_be_caught, (
            f"phase={phase}: dropping the weight was "
            f"{'not caught' if must_be_caught else 'caught'}")


def test_flipping_the_launched_weight_is_caught_on_both_parities():
    """The HOST half of the same question: the kernel is right and the value the
    launcher derived is wrong. ``+phase`` instead of ``-phase`` is the sign error
    a reader of ``mp.Mirror(direction, phase)`` makes."""
    mask = (1, 0, 0, 1, 0, 0)
    for phase in FOLD_PHASES:
        oracle, clean, floors = run_both_paths(mask, ("x",), phase=phase)
        assert oracle == clean
        flipped = tuple(-w for w in floors["weights"])
        _o, produced, _f = run_both_paths(mask, ("x",), phase=phase,
                                          weights_override=flipped)
        assert produced != oracle, f"phase={phase}: a flipped weight changed no bit"


@pytest.mark.parametrize("kind", PREDICTED_NULLS)
def test_the_predicted_null_changes_no_bit(kind):
    """0.25 is an exact power of two, so scaling by it commutes with
    round-to-nearest away from underflow; IEEE multiply commutes bitwise. Both are
    transcription fidelity rather than pinned groupings, and saying so without
    measuring it is how a real difference gets waved away."""
    mask = (1, 1, 1, 1, 1, 1)
    oracle, clean, floors = run_both_paths(mask, ("x", "y"), phase=1)
    _assert_floors(floors)
    assert oracle == clean
    text, sites = _mutate(folded.folded_offdiag_source(mask), kind)
    assert sites > 0, f"{kind} matched nothing"
    _o, produced, _f = run_both_paths(mask, ("x", "y"), phase=1, source=text)
    assert produced == oracle, f"{kind} was predicted inert and moved bytes"


def test_weighting_every_lane_is_caught_where_a_lane_is_subnormal():
    """WHY THE WEIGHT IS A SELECT AND NOT A BLANKET MULTIPLY.

    ``1.0f * x`` is the identity on the bits under ``-ftz=false``, so a blanket
    weight would usually be invisible -- which is exactly why the property is
    stated as "the operation is not there" rather than "the operation is
    harmless". The observable consequence is on the ODD plane, where the weight is
    +1 and a blanket multiply is inert, versus the EVEN plane, where it is -1 and
    every lane's sign flips. Measured on the even plane, where it must be caught.
    """
    mask = (1, 0, 0, 1, 0, 0)
    oracle, clean, _floors = run_both_paths(mask, ("x",), phase=1)
    assert oracle == clean
    text, sites = _mutate(folded.folded_offdiag_source(mask), "weight_every_lane")
    assert sites > 0
    _o, produced, _f = run_both_paths(mask, ("x",), phase=1, source=text)
    assert produced != oracle


# ==========================================================================
# THE PREDICATE
# ==========================================================================

def test_the_predicate_admits_a_fold_the_certified_one_refuses(xp):
    """The whole point, as a verdict pair on one object."""
    fields, layer, grid = build(xp, symmetry=(_mirror("x", 1),),
                                cell=_fold_cell(("x",)))
    install_rows(fields, grid, (1, 0, 0, 1, 0, 0))
    covered, reason = folded.covers_folded_offdiag_constitutive(fields, layer, grid)
    assert covered, reason
    covered, reason = folded.covers_folded_offdiag_composition(fields, layer, grid)
    assert covered, reason
    certified, why = coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, grid)
    assert not certified and "mirror symmetry" in why, why


def test_the_composition_verdict_requires_a_real_fold(xp):
    """THE DISJOINTNESS SEAM, as a measurement rather than a policy. The standalone
    verdict admits an unfolded grid so the reduction leg above can exist; the
    COMPOSITION verdict -- the one the coverage census asks -- does not, so no
    corpus slot can be claimed by two families."""
    fields, layer, grid = build(xp)
    install_rows(fields, grid, (1, 0, 0, 1, 0, 0))
    standalone, _why = folded.covers_folded_offdiag_constitutive(
        fields, layer, grid)
    assert standalone
    composed, reason = folded.covers_folded_offdiag_composition(
        fields, layer, grid)
    assert not composed and "no mirror plane" in reason, reason
    certified, _why = coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, grid)
    assert certified, "the certified family owns the unfolded run"


@pytest.mark.parametrize("symmetry", [s for s in FOLD_SYMMETRIES if s])
@pytest.mark.parametrize("mask", CORPUS_ROW_MASKS)
def test_no_configuration_is_admitted_by_two_families(xp, symmetry, mask):
    """Measured over every fold shape and both corpus masks: the composition
    verdict and the certified off-diagonal verdict are never both True, and
    neither is ever True at the same time as the PLAIN constitutive pair's E
    side."""
    fields, layer, grid = build(xp, symmetry=tuple(_mirror(a, 1) for a in symmetry),
                                cell=_fold_cell(symmetry))
    install_rows(fields, grid, mask)
    mine, _a = folded.covers_folded_offdiag_composition(fields, layer, grid)
    certified, _b = coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, grid)
    plain, _c = coverage.covers_real_pml_constitutive(fields, layer, grid, "E")
    assert not (mine and certified)
    assert not (mine and plain)
    assert not (certified and plain)


class _ThinFold:
    """A grid that reports a folded axis too thin to reflect from.

    MEASURED FIRST: a real ``Grid`` cannot produce one. Folding X at cell sizes
    1.0, 2.0 and 2.5 gives stored extents 3, 3 and 4, so
    ``shape[axis] <= MIRROR_SOURCE_INDEX`` is LATENT rather than live -- exactly
    the status ``coverage.py``'s fail-closed clauses carry, and stated the same
    way. "Latent" is a statement about today's ``Grid``, not about the clause,
    and ``stepping._mirror_source`` RAISES on this shape, so a kernel admitted
    here would read a neighbouring plane where the array path refuses to run.
    """

    def __init__(self, grid, extent=2):
        self._grid = grid
        self.shape = (extent,) + tuple(grid.shape[1:])

    def __getattr__(self, item):
        return getattr(self._grid, item)

    def stored_cells(self, axis):
        return self.shape[axis]

    def owned_cells(self, axis):
        return self.shape[axis] - (0 if self._grid.is_metallic(axis) else 1)


def test_a_real_grid_cannot_fold_an_axis_too_thin_to_reflect():
    """The measurement behind the clause below being latent rather than live."""
    from ..grid import Grid  # noqa: PLC0415

    for extent in (1.0, 2.0, 2.5):
        grid = Grid(resolution=1.0, cell_size=(extent, 10.0, 12.0),
                    boundaries=("periodic",) * 3,
                    symmetry=(_mirror("x", 1),), xp=numpy)
        assert grid.is_mirrored(0)
        assert grid.shape[0] > folded.MIRROR_SOURCE_INDEX, (extent, grid.shape)


def test_a_folded_axis_too_thin_to_reflect_is_refused_by_name(xp):
    """``coord_dn``'s mirror arm returns stored row 2 unconditionally, and
    ``stepping._mirror_source`` RAISES below that many cells."""
    fields, layer, real = build(xp, symmetry=(_mirror("x", 1),),
                                cell=_fold_cell(("x",)))
    install_rows(fields, real, (1, 0, 0, 1, 0, 0))
    grid = _ThinFold(real)
    covered, reason = folded.covers_folded_offdiag_constitutive(fields, layer, grid)
    assert not covered and "stored row" in reason, reason
    with pytest.raises(ValueError, match="mirror ghost images stored row"):
        folded._validate_launch_arguments(
            grid.shape, folded.folded_offdiag_boundary_codes(real),
            coverage.offdiag_wall_mask_flags(real),
            folded.mirror_ghost_weights(real))


def test_a_folded_run_with_no_surviving_row_belongs_to_the_plain_pair(xp):
    """The other disjointness seam: a zero-slot run is
    ``covers_real_pml_constitutive(side='E')``'s, which admits a fold on its own
    2026-08-19 evidence."""
    fields, layer, grid = build(xp, symmetry=(_mirror("x", 1),),
                                cell=_fold_cell(("x",)))
    install_rows(fields, grid, (0, 0, 0, 0, 0, 0))
    covered, reason = folded.covers_folded_offdiag_constitutive(fields, layer, grid)
    assert not covered and "no off-diagonal chi1inv row survived" in reason, reason
    plain, why = coverage.covers_real_pml_constitutive(fields, layer, grid, "E")
    assert plain, why


def test_the_clause_chain_matches_the_certified_predicate_off_the_fold(xp):
    """WRITTEN OUT, NOT DELEGATED -- and therefore pinned against drift.

    The certified predicate SHORT-CIRCUITS, so "refused for the fold and nothing
    else" is not a question its return value can answer and delegation would admit
    a configuration whose later clauses were never reached. The price is two
    chains, and this is what keeps them equal on every axis except the fold: on an
    UNFOLDED grid the two must return the same verdict AND the same reason, over
    every configuration this file can build.
    """
    cases = [
        {}, {"boundaries": ("metallic", "metallic", "metallic")},
        {"force_complex_fields": True},
        {"cell": (1.0, 16.0, 16.0)},
    ]
    for extra in cases:
        for mask in CORPUS_ROW_MASKS:
            fields, layer, grid = build(xp, **extra)
            install_rows(fields, grid, mask)
            mine = folded.covers_folded_offdiag_constitutive(fields, layer, grid)
            theirs = coverage.covers_real_pml_offdiag_constitutive(
                fields, layer, grid)
            assert bool(mine[0]) == bool(theirs[0]), (extra, mask, mine, theirs)
            if not mine[0]:
                assert mine[1] == theirs[1], (extra, mask)


def test_the_predicate_refuses_a_grid_that_cannot_answer(xp):
    """FAIL-CLOSED. These predicates are read by a planner deciding whether to
    dispatch, so an exception escaping one is a CRASHED RUN where a fail-closed
    'no' was the right answer."""

    class _Mute:
        xp = _NumpyWearingCupysName()
        shape = (8, 8, 8)

        def has_symmetry(self):
            raise RuntimeError("this grid cannot say")

    fields, layer, _grid = build(xp)
    install_rows(fields, _grid, (1, 0, 0, 1, 0, 0))
    covered, reason = folded.covers_folded_offdiag_constitutive(
        fields, layer, _Mute())
    assert not covered and "could not answer" in reason, reason


def test_the_two_fold_accessors_must_agree(xp):
    """An object that answers ``has_symmetry`` and ``is_mirrored`` inconsistently
    is refused rather than dispatched -- the reason the certified predicate keeps
    two fold refusals rather than one."""

    class _Inconsistent:
        def __init__(self, grid):
            self._grid = grid

        def __getattr__(self, item):
            return getattr(self._grid, item)

        def has_symmetry(self):
            return True

    fields, layer, grid = build(xp)
    install_rows(fields, grid, (1, 0, 0, 1, 0, 0))
    covered, reason = folded.covers_folded_offdiag_constitutive(
        fields, layer, _Inconsistent(grid))
    assert not covered and "disagree" in reason, reason


def _predicate_with(*edits):
    """A copy of the predicate with one clause removed -- the drop-a-clause probe.

    Compiled from the module's own SOURCE so the leg cannot drift from the shipped
    text: every clause below has to be found verbatim or the probe fails rather
    than silently testing nothing.
    """
    source = MODULE.read_text(encoding="utf-8")
    for old, new in edits:
        assert old in source, f"clause not found verbatim: {old!r}"
        source = source.replace(old, new, 1)
    namespace = {}
    module = compile(source, str(MODULE), "exec")
    import sys  # noqa: PLC0415
    namespace["__name__"] = "meep_gpu.cuda_kernels._folded_probe"
    namespace["__package__"] = "meep_gpu.cuda_kernels"
    exec(module, namespace)  # noqa: S102 - a probe over this module's own text
    assert sys is not None
    return namespace


#: clause -> (needle, replacement, what the drop must produce). Two of the three
#: LEAK -- the configuration the clause exists for is admitted without it, which
#: is the sharpest possible statement. The thickness clause cannot leak, and the
#: reason is measured rather than assumed: no real ``Grid`` can build a folded
#: axis that thin (see the leg above), so the stand-in that reports one has arrays
#: of the real shape and a LATER clause refuses it. What the probe shows there is
#: that the thickness clause is the one that refuses FIRST, which is what a
#: first-refusal predicate can be held to.
DROPPED_CLAUSES = {
    "the_fold_thickness_clause": (
        "if int(facts[\"shape\"][axis]) <= MIRROR_SOURCE_INDEX:",
        "if False:", "refused_later"),
    "the_ghost_weight_clause": (
        "if weight != weight or weight not in (1.0, -1.0):",
        "if False:", "leaks"),
    "the_row_survival_clause": (
        "if not any(volume is not None for volume in rows):",
        "if False:", "leaks"),
}


@pytest.mark.parametrize("case", sorted(DROPPED_CLAUSES))
def test_dropping_a_clause_is_caught(case, xp):
    """Each clause is a SILENT wrong answer if it leaks, so each is armed and the
    configuration it exists for is measured to slip through without it."""
    needle, replacement, expectation = DROPPED_CLAUSES[case]
    probe = _predicate_with((needle, replacement))
    if case == "the_fold_thickness_clause":
        fields, layer, real = build(xp, symmetry=(_mirror("x", 1),),
                                    cell=_fold_cell(("x",)))
        install_rows(fields, real, (1, 0, 0, 1, 0, 0))
        grid = _ThinFold(real)
    elif case == "the_ghost_weight_clause":
        class _Broken:
            def __init__(self, grid):
                self._grid = grid

            def __getattr__(self, item):
                return getattr(self._grid, item)

            def mirror_phase(self, axis):
                return 0

        fields, layer, real = build(xp, symmetry=(_mirror("x", 1),),
                                    cell=_fold_cell(("x",)))
        install_rows(fields, real, (1, 0, 0, 1, 0, 0))
        grid = _Broken(real)
    else:
        fields, layer, grid = build(xp, symmetry=(_mirror("x", 1),),
                                    cell=_fold_cell(("x",)))
        install_rows(fields, grid, (0, 0, 0, 0, 0, 0))
    shipped, why = folded.covers_folded_offdiag_constitutive(fields, layer, grid)
    leaked, why_without = probe["covers_folded_offdiag_constitutive"](
        fields, layer, grid)
    assert not shipped
    if expectation == "leaks":
        assert leaked, f"{case}: the clause is not what refuses this configuration"
    else:
        assert not leaked
        assert why != why_without, (
            f"{case}: removing the clause did not change the first refusal, so "
            f"the clause is not what refuses this configuration")


# ==========================================================================
# THE LAUNCHER'S OWN REFUSALS -- reachable without a device
# ==========================================================================

def test_the_launcher_refuses_both_a_layer_and_an_override():
    """A caller with two answers to a one-answer question has a bug either way."""
    with pytest.raises(ValueError, match="both supplied"):
        folded.update_E_folded_offdiag_fused_pml_real(
            object(), object(), tables={}, codes=(0, 0, 0), walls=(0, 0, 0),
            weights=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="exactly one of pml"):
        folded.update_E_folded_offdiag_fused_pml_real(object())


@pytest.mark.parametrize("shape,codes,walls,weights,match", [
    ((8, 8, 8), (0, 1, 9), (0, 0, 0), (1.0, 1.0, 1.0), "boundary codes"),
    ((8, 8, 8), (0, 1, 2), (0, 0, 1), (1.0, 1.0, 1.0), "wall-masked"),
    ((8, 8, 8), (2, 1, 0), (0, 0, 0), (float("nan"), 1.0, 1.0), "ghost weight"),
    ((2, 8, 8), (2, 1, 0), (0, 0, 0), (-1.0, 1.0, 1.0), "stored row"),
])
def test_the_launcher_validates_what_would_otherwise_launch(shape, codes, walls,
                                                            weights, match):
    """Every one of these is a WRONG ANSWER rather than a crash if it is not
    checked: a NaN weight multiplies one plane into NaN, a folded axis that is
    also wall-masked zeroes a plane MEEP steps, a folded axis with too few cells
    reads a neighbouring plane. The predicate names all three and a gate can reach
    the launcher without the predicate, so they are checked twice."""
    with pytest.raises(ValueError, match=match):
        folded._validate_launch_arguments(shape, codes, walls, weights)


def test_the_source_override_seam_routes_the_gates_bytes():
    """LOAD-BEARING: a mutation harness that could not route its own bytes through
    the launcher would launch the shipped kernel and report a pass for a defect it
    never introduced."""
    mask = (1, 0, 0, 1, 0, 0)
    assert folded.kernel_source(mask) == folded.folded_offdiag_source(mask)
    folded.set_kernel_source(mask, "// mutated\n")
    try:
        assert folded.kernel_source(mask) == "// mutated\n"
    finally:
        folded.set_kernel_source(mask, None)
    assert folded.kernel_source(mask) == folded.folded_offdiag_source(mask)


def test_the_boundary_codes_come_from_the_shared_resolution(xp):
    """The launcher and the predicate ask ONE function what a folded axis
    resolves to, which is what stops them from answering differently about one
    grid -- the defect the certified curl pair carried for a day."""
    for symmetry in (("x",), ("y", "z")):
        _f, _l, grid = build(xp, symmetry=tuple(_mirror(a, 1) for a in symmetry),
                             cell=_fold_cell(symmetry))
        codes = folded.folded_offdiag_boundary_codes(grid)
        kinds = coverage.real_pml_boundary_kinds(grid)
        assert codes == tuple(folded.FOLDED_BC_CODES[k] for k in kinds)
        for axis in range(3):
            assert (codes[axis] == folded.BC_MIRROR_CODE) == grid.is_mirrored(axis)


# ==========================================================================
# THE RECORD
# ==========================================================================

def test_no_device_verdict_is_claimed_without_a_host_and_an_artifact():
    """``certified`` may not be True without a HOST and an ARTIFACT.

    The three move together or the record is a claim with nothing behind it --
    which is the shape every over-claim on this track has had.
    """
    record = folded.FOLDED_OFFDIAG_ADMISSION
    assert record["kernel"] == folded.KERNEL_NAME
    assert bool(record["certified"]) == (record["host"] is not None)
    assert bool(record["certified"]) == bool(record["artifacts"])
    assert record["specified_by"] == "coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION"
    # The certified predicate's own fold refusal STAYS: this family is its
    # complement, not its widening.
    assert coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION["installed"] is False


def test_the_record_was_cut_against_the_emitter_that_ships_today():
    """A digest that no longer matches means the record describes bytes this tree
    does not emit -- so the verdict is about a kernel nobody can compile from here.
    """
    record = folded.FOLDED_OFFDIAG_ADMISSION
    assert record["emitter_corpus_digest"] == folded.corpus_digest(), (
        "the emitter has changed since the gate ran; re-cut the record or "
        "re-run parity/meep_gpu/gate_cuda_folded_offdiag_kernel.py")


def test_the_record_names_artifacts_that_exist_and_report_a_release():
    """A named artifact that is absent, or present and NOT released, is a record
    citing evidence it does not have."""
    import json  # noqa: PLC0415

    api_root = HERE.parents[1]
    for relative in folded.FOLDED_OFFDIAG_ADMISSION["artifacts"]:
        path = api_root / relative
        assert path.exists(), f"the record names a missing artifact: {relative}"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["kernel"] == folded.KERNEL_NAME
        assert record["release"]["released"] is True, record["release"]["reasons"]
        assert record["verdict"]["passed"] is True
        assert record["verdict"]["falsification_as_required"] is True
        assert record["certifies"] is True, "a laptop run is not a certification"
        assert record["emitter_corpus_digest"] == folded.corpus_digest()
        assert (record["gate_sha256"]
                == folded.FOLDED_OFFDIAG_ADMISSION["gate_sha256_at_run"]), (
            "the artifact was cut by a different gate than the record names")
        # The HOST the record names must be the host the artifact was cut on.
        device = (record.get("environment") or {}).get("device_name", "")
        assert device and device in folded.FOLDED_OFFDIAG_ADMISSION["host"], (
            f"the record names host "
            f"{folded.FOLDED_OFFDIAG_ADMISSION['host']!r} but the artifact was "
            f"cut on {device!r}")


def test_both_subnormal_policies_are_covered_by_a_released_artifact():
    """One policy is half a measurement: the two differ in what the HOST does to
    subnormals, and this sub-step's operands reach that band."""
    import json  # noqa: PLC0415

    api_root = HERE.parents[1]
    stamped = set()
    for relative in folded.FOLDED_OFFDIAG_ADMISSION["artifacts"]:
        record = json.loads((api_root / relative).read_text(encoding="utf-8"))
        stamped.add(record["subnormal_policy_stamp"]["stamp"]["policy"])
    assert stamped == set(folded.FOLDED_OFFDIAG_ADMISSION["policies"]), stamped


def test_the_specification_this_family_was_built_from_still_says_what_it_said():
    """The measured record that named this work. If its numbers move, the family's
    reason for existing has to be re-read rather than assumed."""
    record = coverage.FOLDED_OFFDIAG_ROW_MASK_ADMISSION
    assert record["folded_divergent"] == 352
    assert record["differing_words_off_the_ghost_delta_planes"] == 0
    assert record["corpus"]["slots_needing_a_mirror_branch_in_the_kernel"] == 20
    assert "MIRROR branch in the emitted coord_dn" in record[
        "what_would_move_the_twenty"]


def test_the_package_still_imports_nothing_device_shaped_at_scope():
    """The predicate is consumed by the coverage census, which runs on a laptop.
    ``cupy`` must be imported inside the launcher and nowhere else."""
    source = MODULE.read_text(encoding="utf-8")
    for line in source.splitlines():
        if line.startswith("import cupy") or line.startswith("from cupy"):
            raise AssertionError(f"module-scope device import: {line!r}")
    assert "import cupy as cp  # noqa: PLC0415" in source
