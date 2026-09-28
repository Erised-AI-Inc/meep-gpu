"""The DISPERSIVE off-diagonal ``update_E`` slice, at the merge bar.

WHAT A LAPTOP CAN AND CANNOT SETTLE FOR THIS FAMILY, stated first because the
distinction is the whole reason this file exists beside a device gate.

A byte gate measures whether the COMPILED kernel reproduces ``stepping.update_E``.
These legs measure whether the EXPRESSION TREE THE EMITTER WROTE reproduces it --
which is the half a compiler cannot fix, and the half that was wrong in every one
of the four plain curl kernels when they were first written (0/16 against the
array path under every option set). If the tree is wrong here it is wrong on the
device, and it is wrong in a tenth of a second rather than in a device slot.

THE ARITHMETIC LEGS EXECUTE THE EMITTED DEVICE SOURCE. They do not re-implement
it: :func:`evaluate_dispersive_offdiag_source` parses the emitted text over a
KNOWN GRAMMAR and evaluates its arithmetic lines as float32 expressions, and
:func:`test_every_line_of_the_device_source_is_executed_or_pinned` fails if a line
of any emitted source is neither matched by that grammar nor pinned by exact
string.

WHAT IS NEW IN THIS FAMILY, AND THEREFORE WHAT THESE LEGS ARE FOR. The fold arm
is ``folded_offdiag_kernels``', certified 2026-08-20 on an RTX A6000; the row
product and its tail are ``offdiag_emitter``'s, certified 2026-08-16; the pole
chain's SHAPE is ``dispersive_kernels``', released 2026-08-20. The new arithmetic
is exactly one thing:

    THE FOUR GATHERED FIELD READS OF EVERY TENSOR-ROW TERM ARE ``D - sum P``
    CHAINS OVER THE **PARTNER** COMPONENT'S POLES, AT NEIGHBOURING CELLS,
    AND THE MIRROR PARITY WEIGHTS THE **FORMED CHAIN**.

Every leg below is about some part of that sentence, measured against
``stepping.update_E`` on a real ``Grid``/``Fields``/``PML`` triple with real
``PolarizationState`` objects registered.

NO DEVICE VERDICT IS CLAIMED HERE.
``dispersive_offdiag_update_e.DISPERSIVE_OFFDIAG_ADMISSION["host"]`` is None until
``parity/meep_gpu/gate_cuda_dispersive_offdiag.py`` records one, and a leg below
asserts that this file does not quietly imply otherwise.
"""

from __future__ import annotations

import pathlib
import re

import numpy
import pytest

from .. import stepping
from ..dispersion import PolarizationState, Susceptibility
from ..grid import Mirror
from . import (coverage, dispersive_kernels, dispersive_offdiag_update_e as fam,
               folded_offdiag_kernels as folded, offdiag_emitter)
from .test_folded_offdiag import (_as_python, _f32, _ghost_rules,
                                  evaluate_folded_source)
from .test_offdiag_constitutive_pml_real import (
    _advance_sources, _bits, _device_arrays, _tables_for, build, install_rows,
    seed_state, evaluate_emitted_source)

HERE = pathlib.Path(__file__).parent
MODULE = HERE / "dispersive_offdiag_update_e.py"

#: The row masks swept here: the one the corpus drives, the full tensor, a mask
#: that leaves a component on the PLAIN arm, and a single slot. The last two are
#: here because "a component with no surviving row keeps the pure diagonal
#: arithmetic" is a property of the emitter, and a sweep over corpus masks alone
#: would never emit that arm beside a coupled one.
ROW_MASKS = ((1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1),
             (1, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0))
CORPUS_ROW_MASK = (1, 0, 0, 1, 0, 0)

#: The arities swept here. ``fam.POLE_COUNTS_SWEPT`` is the gate's list and this
#: is it, imported, so a merge-bar leg cannot exercise an arity the device sweep
#: does not carry -- or miss one it does.
ARITIES = fam.POLE_COUNTS_SWEPT
CORPUS_ARITY = (1, 1, 1)

#: Fold specs: every axis alone, two planes at once, three at once, and the
#: unfolded control. EVERY AXIS IS FOLDED SOMEWHERE, because a decomposition
#: defect that confuses the slowest coefficient stride with the fastest is only
#: visible if both are folded in the sweep.
FOLD_SYMMETRIES = ((), ("x",), ("y",), ("z",), ("x", "y"), ("x", "y", "z"))

#: Both plane parities. The ghost weight is ``-phase``, so an even plane weights
#: the ghost by -1 and an odd one by +1 -- which is what makes "drop the weight" a
#: DISCRIMINATOR rather than a defect, and a sweep at one parity could not tell.
FOLD_PHASES = (1, -1)

#: Declared terminations for a folded axis. ``stepping._boundary_kinds`` answers
#: ``mirror`` for both, and the borrowed ``coord_up``/``coord_dn`` pair predicts
#: the two are bit-identical here. Swept so the prediction is measured.
FOLD_TERMINATIONS = ("periodic", "metallic")


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
# write and no others. The GHOST RULES and the POLE CHAINS are PARSED out of the
# emitted text and evaluated rather than re-implemented in Python, because they
# are the only genuinely new arithmetic in this family and a pinned string is
# weaker evidence than an executed one. The helpers INHERITED verbatim from the
# two certified emitters (``flat``, ``ghosted``, ``constitutive_apply``,
# ``coord_up``, ``coord_dn``, ``MIRROR_ROW``) are pinned as well, and the
# coordinate pair is parsed through the folded slice's own ``_ghost_rules`` --
# one grammar, one copy.

_TERM_CALL = re.compile(
    r"float term_(?P<tag>\w+) = folded_dispersive_offdiag_term_(?P<partner>\w+)\(\s*"
    r"(?P<volume>\w+)(?P<poles>(?:, P_\w+)*), (?P<coefficient>\w+), idx,\s*"
    r"flat\((?P<down>[^)]*), nyz, nz\),\s*"
    r"flat\((?P<up>[^)]*), nyz, nz\),\s*"
    r"flat\((?P<corner>[^)]*), nyz, nz\),\s*"
    r"(?P<lane>mg_[xyz]), (?P<weight>gw_[xyz])\);")
_TOTAL_FIRST = re.compile(r"float total_(?P<name>\w+) = term_(?P<tag>\w+);")
_TOTAL_ADD = re.compile(
    r"total_(?P<name>\w+) = total_(?P=name) \+ term_(?P<tag>\w+);")
#: The metallic wall-coupling select. BOTH ARMS OF THE TERNARY ARE PARSED rather
#: than the pristine ``? 0.0f : total`` being anchored, so a leg that INVERTS the
#: select is executed and caught by arithmetic instead of being silently skipped
#: as an unmatched line -- which is what the first laptop run reported it as.
_MASK = re.compile(r"total_(?P<name>\w+) = \((?P<wall>wm_[xyz]) && "
                   r"(?P<face>at_[xyz])\) \? (?P<hit>[^:;]+) : (?P<miss>[^;]+);")
_LANE = re.compile(r"int (?P<lane>mg_[xyz]) = (?P<expr>[^;]+);")
#: The diagonal source. THE PLAIN-LOAD ALTERNATIVE IS IN THE GRAMMAR ON PURPOSE:
#: it is the shape ``folded_offdiag_kernels`` emits, so a leg that plants it here
#: is caught by ARITHMETIC rather than silently skipped as an unmatched line --
#: which is the dead-branch hazard this suite's grammar-coverage leg exists for.
_GS = re.compile(
    r"float gs_(?P<name>\w+) = (?:dmp_(?P=name)\("
    r"(?P<volume>\w+)(?P<poles>(?:, P_\w+)*), idx\)"
    r"|(?P<raw>\w+)\[idx\]);")
_US = re.compile(r"float us_(?P<name>\w+) = (?P<volume>\w+)\[idx\];")
_SRC_COUPLED = re.compile(
    r"float src_(?P<name>\w+) = \(gs_(?P=name) \* us_(?P=name)\)"
    r" \+ total_(?P=name);")
_SRC_PLAIN = re.compile(r"float src_(?P<name>\w+) = gs_(?P=name) \* us_(?P=name);")
_TAIL = re.compile(
    r"constitutive_apply\((?P<field>\w+), (?P<aux>f_w_\w+), idx, src_(?P<name>\w+),"
    r" (?P<kps>kps_[xyz])\[(?P<index>[ijk])\], (?P<kms>kms_[xyz])\[(?P=index)\]\);")

#: The three device helper bodies this family owns, matched by SHAPE so the
#: evaluator EXECUTES whatever the emitter wrote. Anchoring the expressions would
#: make a leg that rewrote one fail as a harness error instead of as a caught
#: defect.
_DMP_BODY = re.compile(
    r"float dmp_(?P<name>E[xyz])\(\n"
    r"    const float\* g(?P<poles>(?:, const float\* P_\w+)*), int index\n"
    r"\) \{\n(?P<body>.*?)\n\}", re.S)
_DMP_GHOSTED_BODY = re.compile(
    r"float dmp_ghosted_(?P<name>E[xyz])\(\n"
    r"    const float\* g(?P<poles>(?:, const float\* P_\w+)*), int index\n"
    r"\) \{\n    return (?P<expr>[^;]+);\n\}")
_DMP_MIRROR_BODY = re.compile(
    r"float dmp_ghosted_mirror_(?P<name>E[xyz])\(\n"
    r"    const float\* g(?P<poles>(?:, const float\* P_\w+)*), int index,\n"
    r"    int mg, float w\n"
    r"\) \{\n"
    r"    if \(index < 0\) return 0\.0f;\n"
    r"    float value = (?P<load>[^;]+);\n"
    r"    return (?P<expr>[^;]+);\n\}")
_TERM_BODY = re.compile(
    r"float folded_dispersive_offdiag_term_(?P<name>E[xyz])\(\n"
    r"    const float\* g(?P<poles>(?:, const float\* P_\w+)*), const float\* u,\n"
    r"    int home, int down, int up, int corner, int mg, float w\n"
    r"\) \{\n"
    # ``[^;]+`` spans the emitter's line continuations WITHOUT ``re.S``, and stops
    # at the first ``;``. A DOTALL ``.+`` here matched from the first term helper
    # to the last statement of the kernel and reported one helper where there are
    # three -- measured, and the reason this is spelled by exclusion.
    r"    float near_pair = (?P<near>[^;]+);\n"
    r"    float far_pair = (?P<far>[^;]+);\n"
    r"    return (?P<term>[^;]+);\n\}")

#: Lines every emitted source must carry verbatim, and their owner. A helper
#: carried without its citation is a helper whose provenance has to be remembered.
PINNED_LINES = (
    (fam.MIRROR_SELECT_LINE, "folded_offdiag_kernels.ghosted_mirror"),
    (fam.TERM_RETURN_LINE, "folded_offdiag_kernels.folded_offdiag_term"),
)

#: Scaffolding the evaluator neither executes nor pins because it carries no
#: arithmetic. Kept in step with the gate's own copy.
_SCAFFOLDING = (
    'extern "C" __global__', "float* __restrict__", "const float*",
    "int nx, int ny, int nz,", "int bc_x", "int wm_x", "float gw_x",
    "int idx = blockIdx.x", "if (idx >= nx * ny * nz) return;",
    "int nyz = ny * nz;", "int k = idx % nz;", "int j = (idx / nz) % ny;",
    "int i = idx / (ny * nz);", "int di = coord_dn", "int dj = coord_dn",
    "int dk = coord_dn", "int at_x = (i == 0)", ") {", "}", "{",
)


def _flatten_continuations(text):
    """Join a C expression written across lines into one, as the compiler sees it."""
    return re.sub(r"\s*\n\s*", " ", text).strip()


#: Any ``pointer[coordinate]`` in a helper body. Rewritten to a GATHER so that a
#: leg planting ``g[home]`` where a chain call belongs is evaluated as the device
#: would evaluate it -- not silently indexed on NumPy's first axis, which would
#: make the leg a harness error instead of a caught defect.
_INDEXED = re.compile(r"\b([A-Za-z_]\w*)\[(\w+)\]")


def _c_expression(text):
    """One C expression from a helper body, as Python, with loads as gathers."""
    python = _as_python(text).replace("0.25f", "numpy.float32(0.25)")
    return _INDEXED.sub(r"_gather(\1, \2)", python)


def _gather(volume, index):
    """``pointer[index]`` on a flattened volume, with no ghost rule of its own."""
    flat = numpy.asarray(volume).reshape(-1)
    return _f32(flat[numpy.maximum(index, 0)])


def _pole_names(text):
    """The pole parameter names out of a signature or call fragment, in order."""
    return tuple(re.findall(r"P_\w+_\d+", text or ""))


def device_helpers(source, ghosted):
    """Every device helper this family EMITS, as POSITIONAL Python callables.

    POSITIONAL IS THE WHOLE POINT. A C call binds arguments by position, so a leg
    that plants ``folded_dispersive_offdiag_term_Ey(Dy, P_Ex_0, ...)`` hands the
    body's ``P_Ey_0`` parameter the wrong array and computes a wrong answer. An
    evaluator that bound by NAME would raise a ``NameError`` instead -- a harness
    failure that looks nothing like the defect and proves nothing about it.

    THE BODIES ARE PARSED AND EVALUATED, not counted or transcribed.
    ``sum_then_subtract``, ``drop_one_pole`` and ``reverse_pole_order`` all
    rewrite exactly the chain lines, so an evaluator that reconstructed the chain
    from the pole COUNT would score every one of them UNCAUGHT while measuring
    nothing.

    The ONE line transcribed rather than parsed is ``dmp_ghosted_mirror``'s
    ``if (index < 0) return 0.0f;`` guard, which the pattern requires verbatim and
    which is therefore pinned rather than executed.
    """
    helpers = {}

    def _scope(parameters, arguments, extra=None):
        scope = {"numpy": numpy, "_gather": _gather, "ghosted": ghosted}
        scope.update(helpers)
        scope.update(zip(parameters, arguments))
        if extra:
            scope.update(extra)
        return scope

    for match in _DMP_BODY.finditer(source):
        parameters = ("g",) + _pole_names(match["poles"]) + ("index",)
        statements = [line.strip() for line in match["body"].split("\n")
                      if line.strip()]

        def make_dmp(parameters=parameters, statements=statements):
            def call(*arguments):
                scope = _scope(parameters, arguments)
                for statement in statements:
                    if statement.startswith("return "):
                        return _f32(eval(  # noqa: S307 - this package's own text
                            _c_expression(statement[7:-1]),
                            {"numpy": numpy}, scope))
                    target, _, expression = statement.partition(" = ")
                    scope[target.split()[-1]] = _f32(eval(  # noqa: S307
                        _c_expression(expression[:-1]),
                        {"numpy": numpy}, scope))
                raise AssertionError("a dmp body has no return statement")
            return call
        helpers[f"dmp_{match['name']}"] = make_dmp()

    for match in _DMP_GHOSTED_BODY.finditer(source):
        parameters = ("g",) + _pole_names(match["poles"]) + ("index",)

        def make_ghosted(parameters=parameters,
                         expression=_flatten_continuations(match["expr"])):
            def call(*arguments):
                return _f32(eval(_c_expression(expression),  # noqa: S307
                                 {"numpy": numpy}, _scope(parameters, arguments)))
            return call
        helpers[f"dmp_ghosted_{match['name']}"] = make_ghosted()

    for match in _DMP_MIRROR_BODY.finditer(source):
        parameters = (("g",) + _pole_names(match["poles"])
                      + ("index", "mg", "w"))

        def make_mirror(parameters=parameters,
                        load=_flatten_continuations(match["load"]),
                        select=_flatten_continuations(match["expr"])):
            def call(*arguments):
                scope = _scope(parameters, arguments)
                scope["w"] = numpy.float32(scope["w"])
                value = _f32(eval(_c_expression(load),  # noqa: S307
                                  {"numpy": numpy}, scope))
                scope["value"] = value
                weighted = _f32(eval(_c_expression(select),  # noqa: S307
                                     {"numpy": numpy}, scope))
                # ``if (index < 0) return 0.0f;`` -- PINNED by the pattern above
                # and transcribed here, the one line of these bodies that is not
                # executed off the emitted text.
                return _f32(numpy.where(scope["index"] < 0,
                                        numpy.float32(0.0), weighted))
            return call
        helpers[f"dmp_ghosted_mirror_{match['name']}"] = make_mirror()

    for match in _TERM_BODY.finditer(source):
        parameters = (("g",) + _pole_names(match["poles"])
                      + ("u", "home", "down", "up", "corner", "mg", "w"))

        def make_term(parameters=parameters,
                      near=_flatten_continuations(match["near"]),
                      far=_flatten_continuations(match["far"]),
                      ret=_flatten_continuations(match["term"])):
            def call(*arguments):
                scope = _scope(parameters, arguments)

                def evaluate(text):
                    return _f32(eval(_c_expression(text),  # noqa: S307
                                     {"numpy": numpy}, scope))
                scope["near_pair"] = evaluate(near)
                scope["far_pair"] = evaluate(far)
                return evaluate(ret)
            return call
        helpers[f"folded_dispersive_offdiag_term_{match['name']}"] = make_term()

    return helpers


def evaluate_dispersive_offdiag_source(source, arrays, tables, codes, walls,
                                       weights, shape):
    """Run one launch of the emitted kernel over the whole grid, in float32.

    ``arrays`` maps device parameter names to (nx, ny, nz) float32 arrays --
    including the ``P_<component>_<index>`` pole volumes -- and is MUTATED in
    place, exactly as the kernel mutates its outputs.
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
    lane_scope = {"BC_PERIODIC": 0, "BC_METALLIC": 1, "BC_MIRROR": 2,
                  "numpy": numpy}
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

    helpers = device_helpers(source, ghosted)
    assert {name for name in helpers if name.startswith("dmp_E")} == {
        "dmp_Ex", "dmp_Ey", "dmp_Ez"}, sorted(helpers)

    scalars = {}
    for match in _GS.finditer(body):
        name = match["name"]
        if match["raw"] is not None:
            # The PLAIN-LOAD alternative, executed rather than skipped. See the
            # note on ``_GS``: this shape is a defect here, not an emitter arm.
            scalars[f"gs_{name}"] = _gather(arrays[match["raw"]], home)
            continue
        arguments = [arrays[match["volume"]]]
        arguments.extend(arrays[pole] for pole in _pole_names(match["poles"]))
        scalars[f"gs_{name}"] = _f32(helpers[f"dmp_{name}"](*arguments, home))
    for match in _US.finditer(body):
        scalars[f"us_{match['name']}"] = _f32(numpy.asarray(arrays[match["volume"]]))

    terms, totals, sources = {}, {}, {}
    for match in _TERM_CALL.finditer(body):
        # POSITIONAL, exactly as the call site binds: the arrays in call order,
        # then the coefficient, then the four indices, the lane and the weight.
        arguments = [arrays[match["volume"]]]
        arguments.extend(arrays[pole] for pole in _pole_names(match["poles"]))
        arguments.append(numpy.asarray(arrays[match["coefficient"]]))
        arguments.extend([home, flat(match["down"]), flat(match["up"]),
                          flat(match["corner"]), lanes[match["lane"]],
                          ghost_weights[match["weight"]]])
        terms[match["tag"]] = helpers[
            f"folded_dispersive_offdiag_term_{match['partner']}"](*arguments)
    for match in _TOTAL_FIRST.finditer(body):
        totals[match["name"]] = terms[match["tag"]].copy()
    for match in _TOTAL_ADD.finditer(body):
        totals[match["name"]] = _f32(totals[match["name"]] + terms[match["tag"]])
    for match in _MASK.finditer(body):
        # THE SELECT IS ALWAYS EVALUATED, never skipped when ``wm`` is 0. Skipping
        # it encodes the PRISTINE semantics -- "a dead wall flag changes nothing"
        # -- which is false of an INVERTED select, where a dead flag zeroes the
        # coupling everywhere. The first laptop run scored that defect PARTIAL for
        # exactly that reason.
        scope = {"numpy": numpy, "_gather": _gather,
                 f"total_{match['name']}": totals[match["name"]]}
        arms = [_f32(numpy.asarray(eval(  # noqa: S307 - this module's own text
            _c_expression(match[key]), {"numpy": numpy}, scope),
            dtype=numpy.float32)) for key in ("hit", "miss")]
        condition = numpy.logical_and(bool(wall_flags[match["wall"]]),
                                      faces[match["face"]])
        totals[match["name"]] = _f32(numpy.where(
            condition, arms[0], arms[1]).astype(numpy.float32))
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
        # constitutive_apply (PINNED by exact string, and welded to both certified
        # siblings'): prev is read BEFORE fw is written, then two separate
        # accumulations left to right.
        previous = numpy.asarray(aux).copy()
        source_values = sources[match["name"]]
        aux[...] = source_values
        accumulated = _f32(numpy.asarray(field) + _f32(kps * source_values))
        field[...] = _f32(accumulated - _f32(kms * previous))


# --------------------------------------------------------------------------
# The fixture: one frozen state, run twice
# --------------------------------------------------------------------------

def register_poles(fields, grid, arity, seed=20260820):
    """Build ``max(arity)`` real ``PolarizationState`` objects and seed their P.

    THE ARITY IS BUILT FROM REAL OBJECTS, never faked: a per-component sigma of
    zero is what ``PolarizationState`` filters on (dispersion.py:641-643), which
    is the engine's own spelling of "this term does not drive that component" and
    is exactly what MEEP's ``needs_P`` / ``trivial_sigma`` pair does. A fixture
    that instead poked ``_driven`` would be measuring an object the engine cannot
    build.

    ``P`` IS SEEDED AWAY FROM ZERO. An all-zero pole chain is a NON-dispersive
    case wearing dispersive clothes: every subtraction would be exact, the
    ordering defects would all come back UNCAUGHT, and the reason would be about
    the fixture rather than about the kernel.
    """
    rng = numpy.random.default_rng(seed)
    xp = grid.xp
    kinds = ("lorentzian", "drude")
    # ARITY (0, 0, 0) STILL REGISTERS ONE STATE, with an all-zero sigma. That is
    # what the configuration IS: ``fields.polarizations`` truthy (which both
    # non-dispersive off-diagonal predicates refuse on) with a chain of zero on
    # every component (which the array path serves by ALIASING D). A fixture that
    # registered nothing would build the shipped folded family's configuration
    # and the reduction legs would be comparing that family to itself.
    for index in range(max(1, max(arity))):
        term = Susceptibility(frequency=0.20 + 0.03 * index, gamma=0.008,
                              kind=kinds[index % 2])
        sigma = {name: (0.35 + 0.04 * index if index < arity[axis] else 0.0)
                 for axis, name in enumerate(("Ex", "Ey", "Ez"))}
        state = PolarizationState(term, sigma, grid, fields._field_dtype())
        for component in state.driven():
            state.P[component][...] = xp.asarray(
                rng.uniform(-1.0, 1.0, size=grid.shape).astype(numpy.float32))
            state.P_prev[component][...] = xp.asarray(
                rng.uniform(-1.0, 1.0, size=grid.shape).astype(numpy.float32))
        fields.polarizations.append(state)
    return fields.polarizations


def pole_arrays(fields):
    """``{'P_Ex_0': array, ...}`` in the signature's own order."""
    plan = fam.resolve_pole_plan(fields)
    out = {}
    for target, _source, _axis in fam.E_TERMS:
        for index, array in enumerate(plan[target]):
            out[f"P_{target}_{index}"] = array
    return out


def run_both_paths(mask, arity, symmetry=(), phase=1, termination="periodic",
                   steps=2, source=None, seed=20260820, boundaries=None,
                   codes_override=None, weights_override=None,
                   emitter="dispersive_offdiag"):
    """``stepping.update_E`` and the emitted source from ONE frozen state.

    Returns ``(oracle_bits, produced_bits, floors)``. ``floors`` carries the
    non-vacuity measurements every leg asserts, because a leg that cannot fail
    certifies nothing:

    * ``oracle_moved`` -- zero-init is a FIXED POINT of this recurrence;
    * ``coupling_max_abs`` -- what the off-diagonal rows actually contributed;
    * ``pole_moved`` -- whether ``D - sum P`` differs BITWISE from ``D``. A case
      whose chain is inert is testing the non-dispersive family with extra steps;
    * ``folded_axis_deviation`` and ``mirror_ghost_min_abs`` -- a folded axis
      whose half-integer profile is the identity cannot show a coefficient-index
      error on the axis the fold moved, and a mirror ghost whose source row is
      identically zero EQUALS the metallic zero it is being distinguished from.

    ``emitter="folded"`` and ``emitter="certified"`` run the two SIBLING families'
    sources through their own evaluators on the same frozen state -- the only way
    a reduction leg can compare two trees rather than two runs.

    PLAIN NUMPY, NOT AN ``xp`` FIXTURE: what is measured is a float32 expression
    tree, which is the same tree whatever allocated the arrays.
    """
    planes = tuple(Mirror(name.upper(), phase) for name in symmetry)
    if boundaries is None:
        boundaries = tuple(termination if "xyz"[axis] in symmetry else "periodic"
                           for axis in range(3))
    fields, layer, grid = build(numpy, boundaries=boundaries, symmetry=planes,
                                cell=_fold_cell(symmetry))
    rows = install_rows(fields, grid, mask, seed=seed)
    register_poles(fields, grid, arity, seed=seed)
    seed_state(fields, grid, seed=seed % 1000)
    poles = pole_arrays(fields)
    frozen = {name: numpy.asarray(getattr(fields, name)).copy()
              for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                           "f_w_Ex", "f_w_Ey", "f_w_Ez")}

    # The mirror ghost reads stored row MIRROR_SOURCE_INDEX of the PARTNER's
    # D - sum P volume, so the floor is measured on THAT and not on D. The
    # volumes are COPIED because ``update_E`` rewrites the same per-component
    # scratch buffers (fields.py:1131) and a floor read after the loop would be
    # a fact about the last step.
    displacement = stepping._nonlinear_displacement(fields, layer)
    volumes = {name: numpy.asarray(volume).copy()
               for name, volume in displacement["volumes"].items()}

    # WHAT THE OFF-DIAGONAL ROWS ACTUALLY CONTRIBUTED, from the engine's own
    # function on the frozen state and over EVERY component -- not from Ex alone.
    # A mask whose only live slot is (Ey, Ez) leaves Ex's coupling identically
    # zero, and a floor that read Ex would refuse a case that is perfectly able
    # to discriminate.
    coupling = 0.0
    for component in ("Ex", "Ey", "Ez"):
        term = stepping._offdiagonal_terms(fields, component, displacement)
        if term is None:
            continue
        coupling = max(coupling,
                       float(numpy.abs(numpy.asarray(term)).max()))

    ghost = None
    for axis in range(3):
        if not grid.is_mirrored(axis):
            continue
        for component in ("Ex", "Ey", "Ez"):
            index = [slice(None)] * 3
            index[axis] = stepping.MIRROR_SOURCE_INDEX
            value = float(numpy.abs(
                numpy.asarray(volumes[component])[tuple(index)]).max())
            ghost = value if ghost is None else min(ghost, value)

    pole_moved = 0
    for component, source_name, _axis in fam.E_TERMS:
        chain = numpy.asarray(volumes[component])
        raw = numpy.asarray(getattr(fields, source_name))
        pole_moved += int(numpy.count_nonzero(
            numpy.ascontiguousarray(chain, dtype=numpy.float32).ravel().view(
                numpy.uint32)
            != numpy.ascontiguousarray(raw, dtype=numpy.float32).ravel().view(
                numpy.uint32)))

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

    for name, values in frozen.items():
        getattr(fields, name)[...] = values
    walls = coverage.offdiag_wall_mask_flags(grid)
    weights = (tuple(weights_override) if weights_override is not None
               else fam.mirror_ghost_weights(grid))
    if emitter == "certified":
        codes = tuple(coverage.BC_CODES[kind]
                      for kind in coverage.real_pml_boundary_kinds(grid))
        text = offdiag_emitter.offdiag_source(mask) if source is None else source
    elif emitter == "folded":
        codes = (tuple(codes_override) if codes_override is not None
                 else folded.folded_offdiag_boundary_codes(grid))
        text = folded.folded_offdiag_source(mask) if source is None else source
    else:
        codes = (tuple(codes_override) if codes_override is not None
                 else fam.dispersive_offdiag_boundary_codes(grid))
        text = (fam.dispersive_offdiag_source(mask, arity) if source is None
                else source)

    def arrays_now():
        arrays = _device_arrays(fields, rows)
        arrays.update(poles)
        return arrays

    for _ in range(steps):
        arrays = arrays_now()
        if emitter == "certified":
            evaluate_emitted_source(text, arrays, _tables_for(layer), codes,
                                    walls, grid.shape)
        elif emitter == "folded":
            evaluate_folded_source(text, arrays, _tables_for(layer), codes,
                                   walls, weights, grid.shape)
        else:
            evaluate_dispersive_offdiag_source(
                text, arrays, _tables_for(layer), codes, walls, weights,
                grid.shape)
        _advance_sources(fields)
    floors = {"oracle_moved": moved, "coupling_max_abs": coupling,
              "pole_moved": pole_moved,
              "folded_axis_deviation": deviation, "mirror_ghost_min_abs": ghost,
              "shape": tuple(grid.shape), "codes": codes, "walls": walls,
              "weights": weights, "fields": fields, "layer": layer,
              "grid": grid, "rows": rows}
    return oracle, _bits(fields), floors


def _assert_floors(floors, folded_expected=True, poles_expected=True):
    assert floors["oracle_moved"] > 0, (
        "the array path changed no output word: zero-init is a fixed point of "
        "this recurrence and a case that moved nothing certifies nothing")
    assert floors["coupling_max_abs"] > 0.0, (
        "the off-diagonal coupling is identically zero: this case cannot "
        "distinguish any defect in this family")
    if poles_expected:
        assert floors["pole_moved"] > 0, (
            "D - sum P is bitwise equal to D everywhere: this case is a "
            "NON-dispersive case wearing dispersive clothes and every ordering "
            "defect would come back UNCAUGHT for a reason about the fixture")
    if folded_expected:
        assert floors["folded_axis_deviation"] > 0.0, (
            "the folded axis's half-integer profile is the identity everywhere; "
            "a coefficient-index error on the axis the fold moved is invisible")
        assert (floors["mirror_ghost_min_abs"] or 0.0) > 0.0, (
            "stored row 2 of a partner D - sum P volume is identically zero on a "
            "folded axis, so the mirror ghost equals the metallic zero ghost")


# ==========================================================================
# PROVENANCE: nothing in the shared half of the device source is typed twice
# ==========================================================================

def test_every_borrowed_block_still_lives_in_its_owner():
    """The extraction must FAIL LOUDLY rather than fall back to a local copy.

    Two copies of a helper are equal only until someone edits one, and a silent
    divergence in ``coord_dn`` or in the welded ``constitutive_apply`` tail would
    be a wrong answer no diff of this family's module would show.
    """
    for kind, name, owner in fam.prelude_provenance():
        donor = (offdiag_emitter.PRELUDE if owner == "offdiag_emitter"
                 else folded._OWN_PRELUDE)
        block = (fam._device_block(donor, name) if kind == "helper"
                 else fam._define_block(donor, name))
        assert block.strip(), (kind, name, owner)
        assert block in donor, (name, owner)


def test_the_borrowed_extraction_agrees_with_the_folded_familys_own():
    """This family's generic extractor and the folded family's hard-wired one must
    return the same text for the three helpers both borrow, or the two tracks are
    reading the certified prelude differently."""
    for name in folded.SIBLING_HELPERS:
        assert (fam._device_block(offdiag_emitter.PRELUDE, name)
                == folded._sibling_block(name)), name


@pytest.mark.parametrize("mask", ROW_MASKS)
@pytest.mark.parametrize("arity", ARITIES)
def test_every_borrowed_block_lands_in_every_emitted_source(mask, arity):
    source = fam.dispersive_offdiag_source(mask, arity)
    for kind, name, owner in fam.prelude_provenance():
        donor = (offdiag_emitter.PRELUDE if owner == "offdiag_emitter"
                 else folded._OWN_PRELUDE)
        block = (fam._device_block(donor, name) if kind == "helper"
                 else fam._define_block(donor, name))
        assert block in source, (name, owner)


@pytest.mark.parametrize("mask", ROW_MASKS)
@pytest.mark.parametrize("arity", ARITIES)
def test_the_pinned_lines_are_the_folded_familys_own(mask, arity):
    """The mirror SELECT and the term RETURN are the folded family's, character
    for character. Only the loaded VALUE changes in this family; if the weight
    rule or the association forked, the fold arm would be a different kernel
    wearing an inherited certification."""
    source = fam.dispersive_offdiag_source(mask, arity)
    folded_prelude = folded.prelude()
    for line, owner in PINNED_LINES:
        assert line in folded_prelude, (line, owner)
        assert line in source, (line, owner)


def test_the_shared_tables_are_the_certified_families_own_objects():
    """One home per fact. Imported by value rather than re-typed, so a rename in
    a sibling fails here instead of forking the arithmetic."""
    assert fam.E_TERMS is folded.E_TERMS is offdiag_emitter.E_TERMS
    assert fam.PARTNER_VOLUMES is offdiag_emitter.PARTNER_VOLUMES
    assert fam.ROW_PARAMETERS is offdiag_emitter.ROW_PARAMETERS
    assert fam.LIVE_ROW_MASKS is offdiag_emitter.LIVE_ROW_MASKS
    assert fam.MIRROR_SOURCE_INDEX == stepping.MIRROR_SOURCE_INDEX
    assert fam.FOLDED_BC_CODES == dict(coverage.BC_CODES, mirror=2)
    assert fam.BC_MIRROR_CODE == folded.BC_MIRROR_CODE


def test_the_compile_options_match_both_siblings():
    """``--fmad=false`` is CORRECTNESS on this sub-step. Spelled in each module so
    a by-path load cannot pick up a different tuple than a gate compiled, and
    pinned equal here so the three cannot drift apart."""
    assert fam._COMPILE_OPTIONS == folded._COMPILE_OPTIONS
    assert fam._COMPILE_OPTIONS == dispersive_kernels._COMPILE_OPTIONS
    assert fam._COMPILE_OPTIONS == ('--fmad=false',)


def test_the_pole_cap_is_this_familys_own_number():
    """A CAP IS A FACT ABOUT A GATE and a gate answers for one family. Reading
    ``dispersive_kernels.POLE_COUNT_CAP`` would widen this body -- which gathers
    each pole at four addresses per term -- to arities nobody ran on it."""
    assert fam.POLE_COUNT_CAP == max(max(counts)
                                     for counts in fam.POLE_COUNTS_SWEPT)
    assert fam.POLE_COUNT_CAP <= dispersive_kernels.POLE_COUNT_CAP
    source = MODULE.read_text(encoding="utf-8")
    assert "dispersive_kernels.POLE_COUNT_CAP" not in source.split(
        "THE COVERAGE PREDICATE")[0].replace(
            "``dispersive_kernels.POLE_COUNT_CAP``", "")


# ==========================================================================
# THE EMITTER
# ==========================================================================

@pytest.mark.parametrize("mask", ROW_MASKS)
@pytest.mark.parametrize("arity", ARITIES)
def test_every_emitted_source_is_pure_ascii_and_encodable(mask, arity):
    """A COMPILE REQUIREMENT, not a style rule: ``compile_using_nvrtc`` writes the
    source through a bare ``open(..., 'w')``, so the bytes go through the
    interpreter's LOCALE encoding -- ASCII under C/POSIX."""
    source = fam.dispersive_offdiag_source(mask, arity)
    assert source.isascii(), [c for c in source if not c.isascii()][:8]
    source.encode("ascii")


@pytest.mark.parametrize("mask", ROW_MASKS)
@pytest.mark.parametrize("arity", ARITIES)
def test_exactly_one_kernel_is_declared_and_it_is_this_familys(mask, arity):
    """A second kernel wearing a certified sibling's name would make any NVRTC
    binary observation ambiguous about which body it saw."""
    source = fam.dispersive_offdiag_source(mask, arity)
    assert fam.shipped_kernel_names(source) == {fam.KERNEL_NAME}
    assert fam.KERNEL_NAME != folded.KERNEL_NAME
    assert fam.KERNEL_NAME != offdiag_emitter.KERNEL_NAME
    assert fam.KERNEL_NAME not in dispersive_kernels.ARM_KERNEL_NAMES.values()


@pytest.mark.parametrize("mask", ROW_MASKS)
@pytest.mark.parametrize("arity", ARITIES)
def test_the_signature_binds_only_the_live_rows_and_the_live_poles(mask, arity):
    """Binding a dead slot to some other volume leaves a pointer aimed at an array
    the kernel must never touch -- a shape a later edit can read by accident."""
    source = fam.dispersive_offdiag_source(mask, arity)
    start = source.index('extern "C"')
    signature = source[start:source.index(") {", start)]
    for slot, flag in enumerate(mask):
        name = fam.ROW_PARAMETERS[slot]
        assert (name in signature) == bool(flag), (name, flag)
    expected = set(fam.pole_parameter_names(arity))
    declared = set(re.findall(r"P_E[xyz]_\d+", signature))
    assert declared == expected, (sorted(declared), sorted(expected))


@pytest.mark.parametrize("mask", ROW_MASKS)
def test_a_zero_arity_body_emits_no_subtraction_at_all(mask):
    """The array path forms NO subtraction for an undriven component and hands
    back the D array itself, aliased and uncopied (fields.py:1113-1115). The
    emitter must do the same, or the arity-(0,0,0) reduction would be a claim
    about ``x - 0.0f`` instead of a structural identity."""
    source = fam.dispersive_offdiag_source(mask, (0, 0, 0))
    body = source[source.index("__device__ __forceinline__ float dmp_Ex"):]
    assert "value = value -" not in body
    for component in ("Ex", "Ey", "Ez"):
        assert f"float dmp_{component}(\n    const float* g, int index\n)" in body


def test_a_partly_zero_arity_emits_the_aliasing_arm_beside_a_chain():
    """``(2, 0, 3)`` is in the swept list precisely so the two arms are emitted in
    one body: a sweep of uniform arities alone would never compile them together
    and the aliasing branch would be an untested special case."""
    source = fam.dispersive_offdiag_source((1, 1, 1, 1, 1, 1), (2, 0, 3))
    assert "float dmp_Ey(\n    const float* g, int index\n)" in source
    assert source.count("value = value - P_Ex_") == 2
    assert source.count("value = value - P_Ez_") == 3
    assert "P_Ey_0" not in source


def test_the_row_mask_and_the_arity_are_the_only_source_axes():
    """The boundary codes, the wall flags and the ghost weights are RUNTIME
    arguments. Baking them in would multiply 63 x 4 sources by another 64 x 8 and
    buy nothing: they select an index or a predicated zero and change no float
    operation and no association."""
    source = fam.dispersive_offdiag_source(CORPUS_ROW_MASK, CORPUS_ARITY)
    for name in ("int bc_x, int bc_y, int bc_z", "int wm_x, int wm_y, int wm_z",
                 "float gw_x, float gw_y, float gw_z"):
        assert name in source


@pytest.mark.parametrize("bad", [(-1, 0, 0), (0, 0, fam.POLE_COUNT_CAP + 1),
                                 (1, 1)])
def test_an_arity_outside_the_swept_range_is_refused_by_name(bad):
    """Raises rather than clamping. An arity past the cap has never been run on a
    device for THIS body, and emitting for it anyway is the 'admitted by argument
    alone' move this package refuses everywhere else."""
    with pytest.raises(ValueError):
        fam.normalized_pole_counts(bad)


def test_an_all_dead_row_mask_is_refused():
    """That configuration is ``dispersive_kernels``' arm 1, and the two families
    must not overlap on it."""
    with pytest.raises(ValueError):
        fam.dispersive_offdiag_source((0, 0, 0, 0, 0, 0), CORPUS_ARITY)


def test_the_corpus_digest_moves_with_any_emitted_character(monkeypatch):
    """A file hash stops matching when a docstring gains a comma; this pins the
    strings NVRTC actually compiles."""
    before = fam.corpus_digest()
    assert before == fam.corpus_digest()
    original = fam.TERM_RETURN_LINE
    monkeypatch.setattr(fam, "TERM_RETURN_LINE",
                        original.replace("0.25f", "0.250f"))
    assert fam.corpus_digest() != before


def test_only_the_partner_components_get_a_term_helper():
    """Emitting the ghosted/mirror/term trio for a component nothing couples would
    leave three device functions nothing calls -- dead code a later edit can wire
    up by accident."""
    source = fam.dispersive_offdiag_source((1, 0, 0, 0, 0, 0), CORPUS_ARITY)
    assert "folded_dispersive_offdiag_term_Ey" in source
    assert "folded_dispersive_offdiag_term_Ex" not in source
    assert "folded_dispersive_offdiag_term_Ez" not in source
    # every component still gets its own diagonal chain
    for component in ("Ex", "Ey", "Ez"):
        assert f"float dmp_{component}(" in source


# ==========================================================================
# THE ARITHMETIC: the emitted tree against stepping.update_E
# ==========================================================================

@pytest.mark.parametrize("arity", ARITIES)
def test_the_corpus_configuration_reproduces_the_array_path(arity):
    """ONE FOLD PLANE, the corpus row mask, at every swept arity.

    ``examples/absorbed_power_density.py`` is 800 x 402 x 1 with a Y mirror, one
    polarization and row mask (1, 0, 0, 1, 0, 0). This is that shape's arithmetic
    at a size a laptop can run.
    """
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, arity, symmetry=("y",), phase=1)
    _assert_floors(floors, poles_expected=max(arity) > 0)
    assert produced == oracle


@pytest.mark.parametrize("symmetry", FOLD_SYMMETRIES)
@pytest.mark.parametrize("mask", ROW_MASKS)
def test_every_fold_and_every_row_mask_reproduce_the_array_path(symmetry, mask):
    oracle, produced, floors = run_both_paths(
        mask, CORPUS_ARITY, symmetry=symmetry, phase=1)
    _assert_floors(floors, folded_expected=bool(symmetry))
    assert produced == oracle


@pytest.mark.parametrize("phase", FOLD_PHASES)
@pytest.mark.parametrize("termination", FOLD_TERMINATIONS)
def test_both_plane_parities_and_both_terminations_reproduce_the_array_path(
        phase, termination):
    """The weight is ``-phase``, so an even plane weights the ghost by -1 and an
    odd one by +1. A sweep at one parity would score 'drop the weight' a catch and
    learn nothing about the SIGN. Both declared terminations are swept because one
    ``BC_MIRROR`` code serves both here, which is a PREDICTION about ``update_E``
    having no ownership mask and no reflect row."""
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, (2, 2, 2), symmetry=("x",), phase=phase,
        termination=termination)
    _assert_floors(floors)
    assert produced == oracle


def test_a_metallic_wall_beside_a_fold_reproduces_the_array_path():
    """A folded axis always reports ``wm = 0`` (``is_metallic and not
    is_mirrored``), so a sweep of folded specs alone leaves every wall flag zero
    and the wall mask untested on a fold."""
    oracle, produced, floors = run_both_paths(
        (1, 1, 1, 1, 1, 1), CORPUS_ARITY, symmetry=("x",), phase=1,
        boundaries=("periodic", "metallic", "periodic"))
    _assert_floors(floors)
    assert any(floors["walls"]), floors["walls"]
    assert produced == oracle


def test_the_multi_step_recurrence_reproduces_the_array_path():
    """``f_w`` is STATE. A single launch cannot see a defect that only shows once
    the auxiliary carries a wrong value into the next step."""
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, (2, 0, 3), symmetry=("y",), phase=1, steps=12)
    _assert_floors(floors)
    assert produced == oracle


# ==========================================================================
# THE REDUCTIONS: measured, not claimed
# ==========================================================================

@pytest.mark.parametrize("mask", ROW_MASKS)
def test_zero_arity_on_a_fold_reduces_to_the_folded_family(mask):
    """At arity (0, 0, 0) the emitted tree must be the FOLDED OFF-DIAGONAL
    family's, word for word on the outputs, from the same frozen state.

    MEASURED rather than claimed: the two sources differ in their function NAMES
    and in one layer of indirection, so a string comparison would be vacuous while
    an output comparison is not. A susceptibility is still REGISTERED (the
    predicate requires one), it simply drives nothing -- which is exactly the
    configuration ``fields.polarizations`` truthy plus a zero chain describes.
    """
    _oracle, mine, floors = run_both_paths(mask, (0, 0, 0), symmetry=("y",))
    _oracle2, theirs, _floors = run_both_paths(
        mask, (0, 0, 0), symmetry=("y",), emitter="folded")
    _assert_floors(floors, poles_expected=False)
    assert mine == theirs


@pytest.mark.parametrize("mask", ROW_MASKS)
def test_zero_arity_unfolded_reduces_to_the_certified_offdiagonal_family(mask):
    """With no fold and no pole the emitted tree must be ``offdiag_emitter``'s,
    which was certified on a device on 2026-08-16. That is the deepest reduction
    this family has and it is measured on outputs for the same reason."""
    _oracle, mine, floors = run_both_paths(mask, (0, 0, 0), symmetry=())
    _oracle2, theirs, _floors = run_both_paths(
        mask, (0, 0, 0), symmetry=(), emitter="certified")
    _assert_floors(floors, folded_expected=False, poles_expected=False)
    assert mine == theirs


def test_a_live_pole_chain_is_NOT_the_folded_familys_answer():
    """THE DISCRIMINATOR FOR THE TWO REDUCTIONS ABOVE. If the shipped folded
    off-diagonal kernel already reproduced a dispersive run, this family would buy
    nothing and the two reduction legs would be passing for a reason about the
    fixture. It does not: its operands are D and this one's are D - sum P."""
    _oracle, mine, floors = run_both_paths(
        CORPUS_ROW_MASK, CORPUS_ARITY, symmetry=("y",))
    _oracle2, theirs, _floors = run_both_paths(
        CORPUS_ROW_MASK, CORPUS_ARITY, symmetry=("y",), emitter="folded")
    _assert_floors(floors)
    assert mine != theirs


# ==========================================================================
# THE DEFECTS THE ARITHMETIC LEGS MUST CATCH
# ==========================================================================
#
# Every one is a SILENT wrong answer -- a smooth, converged, plausible field with
# the wrong tensor in it -- and none would be caught by a magnitude comparison at
# any tolerance a physicist would accept. Each is applied to the EMITTED TEXT and
# the leg requires the result to STOP matching the array path.

def _mutate(mask, arity, needle, replacement, count=None):
    source = fam.dispersive_offdiag_source(mask, arity)
    assert needle in source, needle
    if count is not None:
        assert source.count(needle) == count, (needle, source.count(needle))
    return source.replace(needle, replacement)


def test_coupling_the_raw_D_instead_of_the_chain_is_caught():
    """THE MIDDLE REFUSAL, planted. Subtracting the polarization from the diagonal
    term while coupling the raw D volumes is the defect the phrase 'update_E's
    source is D - sum P' invites, and it is wrong on every cell where any pole is
    nonzero. The rewrite makes every term helper's four field reads plain loads."""
    mutated = _mutate(
        CORPUS_ROW_MASK, CORPUS_ARITY,
        "    float near_pair = dmp_Ey(g, P_Ey_0, home)\n"
        "        + dmp_ghosted_mirror_Ey(g, P_Ey_0, down, mg, w);\n"
        "    float far_pair = dmp_ghosted_Ey(g, P_Ey_0, up)\n"
        "        + dmp_ghosted_mirror_Ey(g, P_Ey_0, corner, mg, w);",
        "    float near_pair = g[home] + ghosted(g, down);\n"
        "    float far_pair = ghosted(g, up) + ghosted(g, corner);")
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, CORPUS_ARITY, symmetry=("y",), source=mutated)
    _assert_floors(floors)
    assert produced != oracle


def test_the_parity_weighting_the_raw_D_instead_of_the_chain_is_caught():
    """The array path shifts the ALREADY-FORMED ``D - sum P`` volume and weights
    face 0 (stepping.py:1870-1872), so the parity multiplies the chain. Weighting
    D and then subtracting unweighted poles is a different number wherever the
    parity is -1 -- which is every EVEN plane."""
    mutated = _mutate(
        CORPUS_ROW_MASK, CORPUS_ARITY,
        "    if (index < 0) return 0.0f;\n"
        "    float value = dmp_Ey(g, P_Ey_0, index);\n"
        "    return mg ? (w * value) : value;",
        "    if (index < 0) return 0.0f;\n"
        "    float value = mg ? (w * g[index]) : g[index];\n"
        "    return value - P_Ey_0[index];")
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, CORPUS_ARITY, symmetry=("y",), phase=1, source=mutated)
    _assert_floors(floors)
    assert produced != oracle


def test_pre_accumulating_the_pole_sum_is_caught_at_two_poles():
    """``D - (P0 + P1)`` instead of ``(D - P0) - P1``. float32 addition is not
    associative; the two agree EXACTLY at one pole, which is why this leg is
    scored at two and why a battery at arity one would learn nothing."""
    mutated = _mutate(
        (1, 1, 1, 1, 1, 1), (2, 2, 2),
        "    float value = g[index];\n"
        "    value = value - P_Ex_0[index];\n"
        "    value = value - P_Ex_1[index];",
        "    float value = g[index] - (P_Ex_0[index] + P_Ex_1[index]);")
    oracle, produced, floors = run_both_paths(
        (1, 1, 1, 1, 1, 1), (2, 2, 2), symmetry=("y",), source=mutated)
    _assert_floors(floors)
    assert produced != oracle


def test_reversing_the_pole_order_is_caught_at_two_poles():
    """The chain order is ``fields.polarizations`` order filtered by ``drives``
    (fields.py:1125). Reversed, the result is a different float32 number for the
    same reason pre-accumulation is."""
    mutated = _mutate(
        (1, 1, 1, 1, 1, 1), (2, 2, 2),
        "    value = value - P_Ex_0[index];\n"
        "    value = value - P_Ex_1[index];",
        "    value = value - P_Ex_1[index];\n"
        "    value = value - P_Ex_0[index];")
    oracle, produced, floors = run_both_paths(
        (1, 1, 1, 1, 1, 1), (2, 2, 2), symmetry=("y",), source=mutated)
    _assert_floors(floors)
    assert produced != oracle


def test_dropping_one_pole_from_the_chain_is_caught():
    mutated = _mutate((1, 1, 1, 1, 1, 1), (2, 2, 2),
                      "    value = value - P_Ey_1[index];\n", "")
    oracle, produced, floors = run_both_paths(
        (1, 1, 1, 1, 1, 1), (2, 2, 2), symmetry=("y",), source=mutated)
    _assert_floors(floors)
    assert produced != oracle


def test_coupling_the_wrong_components_chain_is_caught():
    """Each term reads its PARTNER's chain. Reading the ROW component's chain
    instead -- the defect an author makes by carrying one component name through
    the helper suite -- is a tensor with the wrong anisotropy in it."""
    mutated = _mutate(
        CORPUS_ROW_MASK, CORPUS_ARITY,
        "        Dy, P_Ey_0, chi1inv_Ex_Ey, idx,",
        "        Dy, P_Ex_0, chi1inv_Ex_Ey, idx,")
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, CORPUS_ARITY, symmetry=("y",), source=mutated)
    _assert_floors(floors)
    assert produced != oracle


def test_the_own_axis_up_leg_reading_raw_D_instead_of_the_chain_is_caught():
    """THE HALF-DONE EDIT. ``dmp_ghosted_<c>`` is the OWN-axis up leg of every
    term. Leaving its interior load as the raw ``D`` while the near leg reads the
    chain is the shape an author reaches by fixing the obvious read and missing
    the gathered one -- and it is wrong on every cell where any pole is nonzero.
    The ghost value itself is untouched, so this leg is about the chain and not
    about the guard."""
    mutated = _mutate(
        (1, 1, 1, 1, 1, 1), CORPUS_ARITY,
        "    return (index < 0) ? 0.0f : dmp_Ex(g, P_Ex_0, index);",
        "    return (index < 0) ? 0.0f : g[index];")
    oracle, produced, floors = run_both_paths(
        (1, 1, 1, 1, 1, 1), CORPUS_ARITY, symmetry=(),
        boundaries=("metallic", "metallic", "metallic"), source=mutated)
    _assert_floors(floors, folded_expected=False)
    assert produced != oracle


def test_the_diagonal_source_reading_raw_D_is_caught():
    """``gs_c`` is ``dmp_c(D_c, ..., idx)``. Reading ``D_c[idx]`` instead is the
    non-dispersive body with a dispersive coupling bolted on."""
    mutated = _mutate(CORPUS_ROW_MASK, CORPUS_ARITY,
                      "    float gs_Ex = dmp_Ex(Dx, P_Ex_0, idx);",
                      "    float gs_Ex = Dx[idx];")
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, CORPUS_ARITY, symmetry=("y",), source=mutated)
    _assert_floors(floors)
    assert produced != oracle


def test_the_mirror_ghost_read_as_the_metallic_zero_is_caught():
    """The shipped CERTIFIED off-diagonal kernel's only available answer for a
    folded axis, planted. If this leg did not catch it, this family would have no
    reason to exist over ``cuda_offdiag`` plus a pole chain."""
    mutated = _mutate(CORPUS_ROW_MASK, CORPUS_ARITY,
                      "    if (bc == BC_MIRROR) return MIRROR_ROW;",
                      "    if (bc == BC_MIRROR) return -1;")
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, CORPUS_ARITY, symmetry=("y",), source=mutated)
    _assert_floors(floors)
    assert produced != oracle


def test_flipping_the_launched_ghost_weight_is_caught_on_an_even_plane():
    """The weight is ``-phase``, so on an EVEN plane it is exactly -1 and its sign
    is observable. This is the launcher's argument, not the source's."""
    weights = fam.mirror_ghost_weights
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, CORPUS_ARITY, symmetry=("y",), phase=1,
        weights_override=(1.0, 1.0, 1.0))
    _assert_floors(floors)
    assert floors["weights"] == (1.0, 1.0, 1.0)
    assert produced != oracle
    assert weights is fam.mirror_ghost_weights


def test_the_wall_mask_running_after_the_row_sum_is_caught():
    """stepping.py:1252 precedes :1006-1007: the mask zeroes the COUPLING, not the
    whole source. Moved after the sum it would zero the diagonal term too."""
    mutated = _mutate(
        (1, 1, 1, 1, 1, 1), CORPUS_ARITY,
        "    total_Ex = (wm_y && at_y) ? 0.0f : total_Ex;\n"
        "    total_Ex = (wm_z && at_z) ? 0.0f : total_Ex;\n"
        "    float src_Ex = (gs_Ex * us_Ex) + total_Ex;",
        "    float src_Ex = (gs_Ex * us_Ex) + total_Ex;\n"
        "    src_Ex = (wm_y && at_y) ? 0.0f : src_Ex;\n"
        "    src_Ex = (wm_z && at_z) ? 0.0f : src_Ex;")
    oracle, produced, floors = run_both_paths(
        (1, 1, 1, 1, 1, 1), CORPUS_ARITY, symmetry=(),
        boundaries=("periodic", "metallic", "periodic"), source=mutated)
    _assert_floors(floors, folded_expected=False)
    assert produced != oracle


def test_hoisting_the_coefficient_out_of_the_pair_is_caught():
    """The four-point average times ``u[home]``. The same ALGEBRA only for a
    uniform coefficient, and a different float32 number even then, because MEEP
    registers the entry at the component's Yee site minus half a cell along its
    own axis."""
    mutated = _mutate(CORPUS_ROW_MASK, CORPUS_ARITY, fam.TERM_RETURN_LINE,
                      "    return 0.25f * ((near_pair + far_pair) * u[home]);")
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, CORPUS_ARITY, symmetry=("y",), source=mutated)
    _assert_floors(floors)
    assert produced != oracle


def test_commuting_the_near_product_is_a_NULL():
    """IEEE multiply commutes bitwise. Paired with the hoist above, which edits the
    same expression's ASSOCIATION and must be caught -- so the pair says the
    comparator is sensitive to grouping and insensitive to operand order, which is
    a measurement rather than a claim. A battery of only-must-be-caught legs
    scores identically whether the comparator works or fails everything."""
    mutated = _mutate(CORPUS_ROW_MASK, CORPUS_ARITY,
                      "(near_pair * u[home])", "(u[home] * near_pair)")
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, CORPUS_ARITY, symmetry=("y",), source=mutated)
    _assert_floors(floors)
    assert produced == oracle


def test_pre_accumulating_the_pole_sum_is_a_NULL_at_one_pole():
    """The other half of the two-sidedness, and the reason the multi-pole arities
    are swept: at ONE pole ``D - (P0)`` and ``(D - P0)`` are the same expression,
    so a battery scored only at arity one would report the ordering legs UNCAUGHT
    and be measuring the fixture."""
    oracle, produced, floors = run_both_paths(
        CORPUS_ROW_MASK, (1, 1, 1), symmetry=("y",))
    _assert_floors(floors)
    assert produced == oracle
    assert fam.dispersive_offdiag_source(CORPUS_ROW_MASK, (1, 1, 1)).count(
        "value = value - P_Ex_") == 1


# ==========================================================================
# THE GRAMMAR: the tests cover the shipped text
# ==========================================================================

def _grammar_patterns():
    return (_TERM_CALL, _TOTAL_FIRST, _TOTAL_ADD, _MASK, _LANE, _GS, _US,
            _SRC_COUPLED, _SRC_PLAIN, _TAIL)


@pytest.mark.parametrize("mask", ROW_MASKS)
@pytest.mark.parametrize("arity", ARITIES)
def test_every_line_of_the_device_source_is_executed_or_pinned(mask, arity):
    """The evaluator is a ``finditer`` over an anchored grammar, so a statement the
    grammar does not match is SILENTLY SKIPPED -- a leg would then report a verdict
    that says nothing. This fails instead."""
    source = fam.dispersive_offdiag_source(mask, arity)
    body = source[source.index('extern "C"'):]
    covered = set()
    for pattern in _grammar_patterns():
        for match in pattern.finditer(body):
            covered.update(range(match.start(), match.end()))
    blind, offset = [], 0
    for line in body.split("\n"):
        stripped = line.strip()
        span = set(range(offset, offset + len(line)))
        offset += len(line) + 1
        if not stripped or stripped.startswith("//"):
            continue
        if span & covered:
            continue
        if any(token in line for token in _SCAFFOLDING):
            continue
        blind.append(stripped)
    assert not blind, blind


@pytest.mark.parametrize("mask", ROW_MASKS)
@pytest.mark.parametrize("arity", ARITIES)
def test_every_helper_body_is_parsed_by_the_evaluator(mask, arity):
    """The three helper bodies this family OWNS are parsed and executed, not
    transcribed. If a pattern stops matching, every arithmetic leg above would be
    running a transcription of a body it is not reading."""
    source = fam.dispersive_offdiag_source(mask, arity)
    assert set(m["name"] for m in _DMP_BODY.finditer(source)) == {
        "Ex", "Ey", "Ez"}
    partners = set(fam._live_partner_components(mask))
    assert set(m["name"] for m in _DMP_GHOSTED_BODY.finditer(source)) == partners
    assert set(m["name"] for m in _DMP_MIRROR_BODY.finditer(source)) == partners
    assert set(m["name"] for m in _TERM_BODY.finditer(source)) == partners
    assert _ghost_rules(source)


# ==========================================================================
# THE PREDICATE
# ==========================================================================

def _covered(fields, layer, grid):
    return fam.covers_real_pml_dispersive_offdiag_constitutive(
        fields, layer, grid)


def _build_covered(xp, symmetry=("y",), arity=CORPUS_ARITY,
                   mask=CORPUS_ROW_MASK):
    planes = tuple(Mirror(name.upper(), 1) for name in symmetry)
    boundaries = tuple("periodic" for _ in range(3))
    fields, layer, grid = build(xp, boundaries=boundaries, symmetry=planes,
                                cell=_fold_cell(symmetry))
    install_rows(fields, grid, mask, seed=20260820)
    register_poles(fields, grid, arity)
    return fields, layer, grid


def test_the_corpus_configuration_is_covered(xp):
    fields, layer, grid = _build_covered(xp)
    covered, reason = _covered(fields, layer, grid)
    assert covered, reason


def test_an_unfolded_dispersive_offdiagonal_run_is_also_covered(xp):
    """ADMITTED, and unlike the folded off-diagonal family that is not a
    reduction-only courtesy a narrower census verdict takes back: there is NO
    shipped family to overlap on the unfolded arm, because ``cuda_offdiag``
    refuses the dispersion and ``cuda_dispersive`` refuses the row."""
    fields, layer, grid = _build_covered(xp, symmetry=())
    covered, reason = _covered(fields, layer, grid)
    assert covered, reason


@pytest.mark.parametrize("arity", ARITIES)
def test_every_swept_arity_is_covered(xp, arity):
    fields, layer, grid = _build_covered(xp, arity=arity)
    covered, reason = _covered(fields, layer, grid)
    assert covered, reason


def test_an_arity_past_the_cap_is_refused_by_name(xp):
    fields, layer, grid = _build_covered(
        xp, arity=(fam.POLE_COUNT_CAP + 1,) * 3)
    covered, reason = _covered(fields, layer, grid)
    assert not covered
    assert "POLE_COUNT_CAP" in reason


def test_no_registered_susceptibility_is_refused_to_the_two_siblings(xp):
    """THE DISJOINTNESS CLAUSE against both NON-dispersive off-diagonal families,
    spelled on ``fields.polarizations`` being truthy because THAT is the question
    both of them ask. A run carrying a trivial-sigma susceptibility is refused by
    both of them and admitted here, so no slot falls between."""
    fields, layer, grid = _build_covered(xp, arity=(0, 0, 0))
    del fields.polarizations[:]
    covered, reason = _covered(fields, layer, grid)
    assert not covered
    assert "no susceptibility is registered" in reason


def test_a_trivial_sigma_susceptibility_is_admitted_here_and_by_nobody_else(xp):
    """The seam measured rather than argued: with a registered state that drives
    nothing, this predicate covers and both sibling off-diagonal predicates
    refuse, so the slot is owned by exactly one family."""
    fields, layer, grid = _build_covered(xp, arity=(0, 0, 0))
    assert fields.polarizations
    assert _covered(fields, layer, grid)[0]
    assert not coverage.covers_real_pml_offdiag_constitutive(
        fields, layer, grid)[0]
    assert not folded.covers_folded_offdiag_composition(fields, layer, grid)[0]


def test_no_offdiagonal_row_is_refused_to_the_dispersive_family(xp):
    planes = (Mirror("Y", 1),)
    fields, layer, grid = build(xp, boundaries=("periodic",) * 3,
                                symmetry=planes, cell=_fold_cell(("y",)))
    epsilon, inverse = {}, {}
    rng = numpy.random.default_rng(1)
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=grid.shape).astype(numpy.float32)
        epsilon[component] = grid.xp.asarray(values)
        inverse[component] = grid.xp.asarray(
            (numpy.float32(1.0) / values).astype(numpy.float32))
    fields.set_epsilon_volumes(epsilon, inverse)
    register_poles(fields, grid, CORPUS_ARITY)
    covered, reason = _covered(fields, layer, grid)
    assert not covered
    assert "no off-diagonal chi1inv row survived installation" in reason


@pytest.mark.parametrize("symmetry", FOLD_SYMMETRIES)
def test_the_four_shipped_families_and_this_one_partition_every_case(
        xp, symmetry):
    """DISJOINTNESS, measured on every fold spec this slice sweeps. Two families
    admitting one slot is not extra coverage -- it means one of them widened past
    its evidence."""
    fields, layer, grid = _build_covered(xp, symmetry=symmetry)
    verdicts = {
        "this": _covered(fields, layer, grid)[0],
        "cuda_constitutive": coverage.covers_real_pml_constitutive(
            fields, layer, grid, side="E")[0],
        "cuda_offdiag": coverage.covers_real_pml_offdiag_constitutive(
            fields, layer, grid)[0],
        "cuda_folded_offdiag": folded.covers_folded_offdiag_composition(
            fields, layer, grid)[0],
        "cuda_dispersive": (
            dispersive_kernels.covers_real_pml_dispersive_constitutive(
                fields, layer, grid)[0]),
    }
    assert verdicts["this"], verdicts
    assert sum(1 for value in verdicts.values() if value) == 1, verdicts


def test_a_numpy_backend_is_refused(xp):
    fields, layer, grid = _build_covered(xp)
    grid.xp = numpy
    covered, reason = _covered(fields, layer, grid)
    assert not covered
    assert reason == "backend is not CuPy"


def test_no_active_layer_is_refused_by_name(xp):
    fields, layer, grid = _build_covered(xp)
    covered, reason = _covered(fields, None, grid)
    assert not covered
    assert "no active PML layer" in reason


def test_a_susceptibility_kind_outside_the_pair_is_refused(xp):
    fields, layer, grid = _build_covered(xp)
    fields.polarizations[0].susceptibility = type(
        "Other", (), {"kind": "nonlinear_something"})()
    covered, reason = _covered(fields, layer, grid)
    assert not covered
    assert "outside" in reason


def test_a_pole_volume_aliasing_an_output_is_refused(xp):
    """The coupling re-reads the partner volumes at neighbour offsets -- on a
    folded axis including interior stored row 2 -- while the outputs are written,
    so an alias makes the answer depend on block schedule."""
    fields, layer, grid = _build_covered(xp)
    fields.polarizations[0].P["Ex"] = fields.f_w_Ex
    covered, reason = _covered(fields, layer, grid)
    assert not covered
    assert "aliases" in reason


def test_two_poles_sharing_a_buffer_are_refused(xp):
    fields, layer, grid = _build_covered(xp, arity=(2, 2, 2))
    fields.polarizations[1].P["Ex"] = fields.polarizations[0].P["Ex"]
    covered, reason = _covered(fields, layer, grid)
    assert not covered
    assert "aliases" in reason


def test_more_planes_than_the_gate_swept_are_refused_by_name(xp,
                                                             monkeypatch):
    monkeypatch.setattr(fam, "FOLD_PLANES_SWEPT", 2)
    fields, layer, grid = _build_covered(xp, symmetry=("x", "y", "z"))
    covered, reason = _covered(fields, layer, grid)
    assert not covered
    assert "mirror planes at once" in reason


# ==========================================================================
# THE LAUNCHER'S OWN REFUSALS
# ==========================================================================

def test_the_launcher_refuses_both_a_layer_and_an_override(xp):
    fields, layer, grid = _build_covered(xp)
    with pytest.raises(ValueError, match="both supplied"):
        fam.update_E_dispersive_offdiag_fused_pml_real(
            fields, layer, codes=(0, 0, 0))


def test_the_launcher_refuses_neither_a_layer_nor_a_full_override(xp):
    fields, layer, grid = _build_covered(xp)
    with pytest.raises(ValueError, match="exactly one of pml"):
        fam.update_E_dispersive_offdiag_fused_pml_real(fields)


def test_the_launcher_refuses_a_folded_axis_that_is_also_wall_masked():
    with pytest.raises(ValueError, match="wall-masked"):
        fam._validate_launch_arguments((8, 8, 8), (2, 0, 0), (1, 0, 0),
                                       (-1.0, 1.0, 1.0))


def test_the_launcher_refuses_an_unreadable_ghost_weight():
    with pytest.raises(ValueError, match="ghost weight"):
        fam._validate_launch_arguments((8, 8, 8), (2, 0, 0), (0, 0, 0),
                                       (float("nan"), 1.0, 1.0))


def test_the_launcher_refuses_a_folded_axis_thinner_than_the_mirror_row():
    with pytest.raises(ValueError, match="stored cells"):
        fam._validate_launch_arguments((2, 8, 8), (2, 0, 0), (0, 0, 0),
                                       (-1.0, 1.0, 1.0))


def test_the_pole_plan_is_the_dispersive_familys_own_reader(xp):
    """ONE SPELLING of ``fields.polarizations`` filtered by ``drives``: the chain
    ORDER is the arithmetic here, and two readers would be two places for it to
    drift."""
    fields, layer, grid = _build_covered(xp, arity=(2, 0, 3))
    assert (fam.resolve_pole_plan(fields)
            == dispersive_kernels.resolve_pole_plan(fields))
    assert fam.pole_counts_of(fam.resolve_pole_plan(fields)) == (2, 0, 3)


def test_the_source_override_door_round_trips(xp):
    """A mutation harness that could not route its own bytes through the launcher
    would launch the shipped kernel and report a pass for a defect it never
    introduced."""
    pristine = fam.kernel_source(CORPUS_ROW_MASK, CORPUS_ARITY)
    fam.set_kernel_source(CORPUS_ROW_MASK, CORPUS_ARITY, "PLANTED")
    try:
        assert fam.kernel_source(CORPUS_ROW_MASK, CORPUS_ARITY) == "PLANTED"
        # a DIFFERENT arity must be untouched: the override is keyed on the pair
        assert fam.kernel_source(CORPUS_ROW_MASK, (2, 2, 2)) != "PLANTED"
    finally:
        fam.set_kernel_source(CORPUS_ROW_MASK, CORPUS_ARITY, None)
    assert fam.kernel_source(CORPUS_ROW_MASK, CORPUS_ARITY) == pristine


# ==========================================================================
# THE RECORD
# ==========================================================================

def test_no_device_verdict_is_claimed_until_a_gate_records_one():
    """``host`` is the only thing that makes any number in the admission record a
    device verdict. While it is None, a predicate returning True licenses a
    MEASUREMENT and nothing else, and this file must not imply otherwise."""
    record = fam.DISPERSIVE_OFFDIAG_ADMISSION
    if record["host"] is None:
        assert record["certified"] is False
        assert record["artifacts"] == ()
        assert record["release_verdict_shown_to_flip_against_a_planted_defect"] \
            is None
    else:
        assert record["artifacts"], record
        assert record["policies"], record
        assert record["release_verdict_shown_to_flip_against_a_planted_defect"]


def test_the_admission_record_names_its_gate_and_the_gate_exists():
    path = HERE.parents[1] / fam.DISPERSIVE_OFFDIAG_ADMISSION["gate"]
    assert path.exists(), path


#: The two engine modules that may name this package at all, and nothing else may.
#: ``fastpath.py`` composes the hand-CUDA table as its second kernel table (one
#: function-local import of ``cuda_kernels.arms`` inside ``_decide``, below the
#: backend rung); ``fastpath_cuda.py`` holds that table's rows and reads the
#: registry to pin them. Every other file under ``meep_gpu/`` is still forbidden.
DISPATCH_SEAM_MODULES = ("fastpath.py", "fastpath_cuda.py")


def test_only_the_dispatch_seam_names_this_package_and_never_this_family():
    """A predicate returning True licenses a MEASUREMENT, not a choice of arithmetic.

    THE CLAIM NARROWED ON 2026-09-11 AND IT IS STILL A CLAIM. Until Phase 2 nothing
    under ``meep_gpu/`` imported ``cuda_kernels`` at all, and this leg asserted that
    absence. The seam now composes this package as its second table, so the absence
    is gone — but what replaced it is the fact worth holding: the seam is exactly TWO
    files, and neither of them, nor any other engine module, may name a FAMILY.

    That second half is the one this family cares about. ``cuda_kernels.arms``
    selects a family by predicate; an engine file that named
    ``dispersive_offdiag_update_e`` directly would be choosing arithmetic for a
    configuration instead of asking the table which arithmetic covers it, and the
    predicate — the thing every gate in this file measures — would no longer be what
    decides. ``test_package_boundary.py`` pins the module-level half of the same
    seam; this leg names it beside the family it protects.
    """
    root = HERE.parent
    statement = re.compile(
        r"^\s*(?:from\s+\S*cuda_kernels\S*\s+import|import\s+\S*cuda_kernels)")
    offenders = []
    families = []
    for path in root.glob("*.py"):
        if path.name.startswith("test_"):
            # A TEST may import the package it tests; what must not import it is
            # the ENGINE. ``test_package_boundary.py`` pins the same seam and
            # names the test files as the exception there too.
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if statement.match(line) and path.name not in DISPATCH_SEAM_MODULES:
                offenders.append(f"{path.name}:{number}: {line.strip()}")
        if "dispersive_offdiag_update_e" in text:
            families.append(path.name)
    assert not offenders, offenders
    assert not families, (
        "an engine module names this FAMILY rather than asking the table for it: "
        + ", ".join(families))
