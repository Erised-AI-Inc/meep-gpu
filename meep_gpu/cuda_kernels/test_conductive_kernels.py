"""What ``conductive_kernels.py`` claims, measured on this laptop.

Four things are settled here and nothing else is:

1. **The device text is the siblings' bytes.** Every template is built by
   transforming ``no_pml_curl``'s or ``step_curl_kernels``'s shipped string, so
   the curl half -- term table, ghost rule, grouping, wall masks -- cannot fork.
   That is a property of the bytes and is measured on the bytes: at ``COND == 0``
   on every component, stripping the preprocessor guards and the added parameters
   must recover the sibling's kernel EXACTLY.

2. **The arithmetic.** A NumPy transcription of the two device tails -- written as
   the KERNEL writes them, per cell, selecting a branch rather than computing all
   four over the volume -- is compared against ``stepping.step_B`` /
   ``stepping.step_D`` as uint32 words. This is the local half of the question the
   brief asked ("decide by measurement whether one kernel can serve a mixed
   run"): the mixed masks are swept, and the answer is measured rather than
   argued. It is NOT a device verdict -- no CUDA is compiled here -- and
   :data:`conductive_kernels.CONDUCTIVE_CURL_ADMISSION` is where that lives.

3. **The four subchunk cases are REACHED.** A PML identity assertion that only
   ever exercised case A would pass for a reason about the fixture. Every
   arithmetic case here asserts a nonzero cell count in each of MEEP's four cases
   it claims to cover, and the identity assertions carry an ``oracle moved`` floor.

4. **The partition.** The admission is the exact per-sub-step inverse of the two
   incumbents' conductivity clause, so no slot can be admitted twice and none can
   fall between them. Both directions are checked on real ``Grid``/``Fields``
   pairs, including the corpus's own mixed case (a D conductivity and no B one).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import hashlib
import pathlib
import re
import types

import numpy
import pytest

from .. import stepping
from ..fields import IYEE_SHIFTS, Fields
from ..grid import Grid, Mirror
from ..pml import PML
from . import conductive_kernels, coverage, no_pml_curl
from .conductive_kernels import covers_conductive_curl as covers

HERE = pathlib.Path(__file__).parent
SIBLING_PATH = HERE / "step_curl_kernels.py"

SUB_STEPS = ("step_B", "step_D")
LOSSLESS = (False, False, False)


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- this directory's standing stand-in.

    The delegated predicate's first question is whether the backend is CuPy at
    all, and that is the one thing about the device library a laptop cannot
    supply. Everything else -- the fold, the dtype, the contiguity, the
    conductivity volumes, the PML tables -- is a real object either way.
    """

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


def make(xp, *, boundaries=("periodic", "metallic", "periodic"),
         cell=(8.0, 10.0, 12.0), axes="", phase=1, dimensions=3, courant=0.5,
         absorber=None, stored=None, d_sigma=None, b_sigma=None,
         inhomogeneous_epsilon=False, seed=3):
    """A real ``(fields, layer, grid)`` triple.

    ``d_sigma``/``b_sigma`` are per-component mappings of sigma volumes, so a run
    with ONE lossy component and two lossless is expressible -- which is the
    configuration the whole per-component question is about.
    """
    planes = tuple(Mirror(name, phase) for name in axes)
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=planes, dimensions=dimensions, xp=xp, courant=courant)
    fields = Fields(grid=grid)
    rng = numpy.random.default_rng(seed)
    if inhomogeneous_epsilon:
        inverse, epsilon = {}, {}
        for component in ("Ex", "Ey", "Ez"):
            values = rng.uniform(0.2, 0.9, size=grid.shape).astype(numpy.float32)
            inverse[component] = values
            epsilon[component] = (1.0 / values).astype(numpy.float32)
        fields.set_epsilon_volumes(epsilon, inverse)
    layer = PML(grid=grid, thickness=absorber) if absorber is not None else None
    if layer is not None and layer.is_active:
        fields.enable_pml_storage()
    elif stored:
        fields.enable_field_storage()
    if d_sigma is not None:
        fields.set_d_conductivity(d_sigma)
    if b_sigma is not None:
        fields.set_b_conductivity(b_sigma)
    return fields, layer, grid


def sigma_map(grid, components, rng, low=0.05, high=0.9):
    """One INHOMOGENEOUS sigma volume per named component.

    Drawn away from zero and with a spread inside each volume, and the volumes are
    pairwise distinct. Against a uniform sigma a coefficient-index defect is
    invisible; against zero the whole tail is; and against one shared volume a
    component-binding defect is. Each of those is a mutation the device gate arms.
    """
    return {component: rng.uniform(low, high, size=grid.shape).astype(numpy.float32)
            for component in components}


def seed_arrays(fields, grid, names, seed=11):
    rng = numpy.random.default_rng(seed)
    for name in names:
        getattr(fields, name)[...] = rng.uniform(
            -1.0, 1.0, size=grid.shape).astype(numpy.float32)


def words(array) -> numpy.ndarray:
    return numpy.ascontiguousarray(numpy.asarray(array),
                                   dtype=numpy.float32).ravel().view(numpy.uint32)


# ---------------------------------------------------------------------------
# 1. THE DEVICE TEXT IS THE SIBLINGS' BYTES
# ---------------------------------------------------------------------------

def _body(source: str, entry: str) -> str:
    marker = f'extern "C" __global__ void {entry}'
    return source[source.index(marker):]


def _strip_guards(body: str, plain_only: bool = True) -> str:
    """Collapse the ``#if COND<n> ... #else <plain> #endif`` blocks to their plain arm.

    Written as a line filter rather than a regex over the whole block so a
    malformed nest raises here instead of silently deleting a tail.
    """
    out, state = [], None
    for line in body.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#if COND"):
            state = "conductive"
            continue
        if stripped == "#else" and state == "conductive":
            state = "plain"
            continue
        if stripped == "#endif" and state in ("conductive", "plain"):
            state = None
            continue
        if state == "conductive" and plain_only:
            continue
        out.append(line)
    assert state is None, "an #if COND block was left open"
    return "".join(out)


@pytest.mark.parametrize("mine,theirs", [
    ("step_B_no_pml_conductive", "step_B_no_pml_real"),
    ("step_B_no_pml_conductive_derived", "step_B_no_pml_real_derived"),
    ("step_D_no_pml_conductive", "step_D_no_pml_real"),
])
def test_the_no_pml_templates_are_the_siblings_bytes_plus_a_tail(mine, theirs):
    """Strip the guards and the six added parameters and the sibling comes back.

    THE WHOLE SAFETY ARGUMENT of this family is that the curl is unedited. That is
    a property of the bytes, so it is measured on the bytes rather than asserted
    in a comment: an equality, not a diff with an allow-list.
    """
    ours = _strip_guards(_body(conductive_kernels.kernel_source(mine, LOSSLESS), mine))
    ours = ours.replace(conductive_kernels._COND_PARAMS, "")
    ours = ours.replace(f"void {mine}(", f"void {theirs}(")
    assert ours == _body(no_pml_curl.kernel_source(theirs), theirs)


@pytest.mark.parametrize("mine,theirs", [
    ("step_B_pml_conductive", "step_B_pml_real"),
    ("step_D_pml_conductive", "step_D_pml_real"),
])
def test_the_pml_templates_are_the_certified_pairs_bytes_plus_a_tail(mine, theirs):
    """Same, against the CERTIFIED curl pair -- read out of the sibling's source.

    ``step_curl_kernels.py`` imports ``cupy`` at scope, so the comparison text is
    read the way the module builds its template: out of the file. A rename there
    fails here rather than yielding a stale string.
    """
    sibling = SIBLING_PATH.read_text(encoding="utf-8")
    start = sibling.index(f'extern "C" __global__ void {theirs}')
    certified = sibling[start:sibling.index("'''", start)]
    ours = _strip_guards(_body(conductive_kernels.kernel_source(mine, LOSSLESS), mine))
    ours = ours.replace(conductive_kernels._COND_PARAMS, "")
    ours = ours.replace(conductive_kernels._FCOND_PARAMS, "")
    ours = ours.replace(f"void {mine}(", f"void {theirs}(")
    assert ours == certified


def _preprocess(text: str) -> str:
    """The text NVRTC would see: the ``#if COND<n>`` arms resolved by the defines.

    Written out rather than shelled to a preprocessor because the whole claim is
    that the guard is the mechanism -- so the mechanism has to be evaluated here,
    on the same defines the kernel is built with, and an unbalanced nest has to
    raise rather than be silently dropped.
    """
    defines = {}
    out, keeping = [], []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#define COND"):
            name, value = stripped[len("#define "):].split()
            defines[name] = int(value)
            continue
        if stripped.startswith("#if COND"):
            expression = stripped[len("#if "):].strip()
            keeping.append(any(defines[name.strip()]
                               for name in expression.split("||")))
            continue
        if stripped == "#else":
            keeping[-1] = not keeping[-1]
            continue
        if stripped == "#endif":
            keeping.pop()
            continue
        if all(keeping):
            out.append(line)
    assert not keeping, "an #if COND block was left open"
    return "".join(out)


@pytest.mark.parametrize("name", conductive_kernels.CONDUCTIVE_KERNELS)
def test_a_lossless_component_compiles_out_of_the_conductive_branch(name):
    """``COND<n> == 0`` must leave NO reference to that component's tables.

    "The lossless component is identical by construction" is the claim, the
    construction is the preprocessor, and this evaluates it: after the defines are
    applied, a lossless component's ``cf``/``ci``/``fc`` pointers must appear only
    in the signature and nowhere in the body, and its tail must be the plain
    family's line.
    """
    for index in range(3):
        cond = [True, True, True]
        cond[index] = False
        compiled = _preprocess(conductive_kernels.kernel_source(name, tuple(cond)))
        body = _body(compiled, name)
        body = body[body.index(") {"):]
        for pointer in (f"cf{index}", f"ci{index}", f"fc{index}"):
            assert pointer not in body, (
                f"{name}: component {index} still reads {pointer} at "
                f"COND{index} == 0")
        for other in (i for i in range(3) if i != index):
            assert f"cf{other}" in body, (
                f"{name}: component {other} lost its conductive tail")


@pytest.mark.parametrize("name,helper,absent", [
    ("step_D_pml_conductive", "cond_pml_apply", "cond_apply"),
    ("step_D_no_pml_conductive", "cond_apply", "cond_pml_apply"),
])
def test_only_the_reachable_tail_helper_is_compiled(name, helper, absent):
    """A mask-0 build defines NEITHER helper, and a live build defines only its own.

    Not tidiness, and it is the lesson of a device run: the gate's battery arms
    every expression in both helpers, so a helper compiled into a kernel that
    never calls it is a site that must score UNCAUGHT -- for a reason about dead
    code rather than about the family. The two preludes are split per layer so
    every armed site is reachable from some launch.
    """
    lossless = _preprocess(conductive_kernels.kernel_source(name, LOSSLESS))
    assert "__forceinline__ void cond_pml_apply" not in lossless
    assert "__forceinline__ void cond_apply" not in lossless
    live = _preprocess(conductive_kernels.kernel_source(name, (True, False, False)))
    assert f"__forceinline__ void {helper}" in live
    assert f"__forceinline__ void {absent}" not in live


def test_the_four_case_tail_carries_the_exact_predicate_not_a_tolerance():
    """``!= 1.0f``, transcribed from stepping.py:2055-2058.

    A ``>=`` or an epsilon here is a defect the REAL coefficient tables cannot
    expose -- there is no near-one band in them -- so it has to be pinned on the
    text as well as driven by the gate's synthetic near-one table.
    """
    text = conductive_kernels._COND_PML_APPLY_PRELUDE
    assert "(km1 != 1.0f) || (si1 != 1.0f)" in text
    assert "(km2 != 1.0f) || (si2 != 1.0f)" in text
    assert not re.search(r"fabs|epsilon|1e-", text)


def test_a_restructured_sibling_fails_loudly_rather_than_silently():
    """The count check is what makes "the sibling's bytes" mean anything."""
    with pytest.raises(RuntimeError, match="expected 1 occurrence"):
        conductive_kernels._replace_exactly("nothing here", "needle", "x", 1, "a needle")


# ---------------------------------------------------------------------------
# 2. THE ARITHMETIC, MEASURED AGAINST ``stepping`` AS UINT32 WORDS
#
# The reference below is a transcription of the DEVICE text, not a second call
# into ``stepping``: it forms the curl the way the kernel forms it and then
# selects ONE branch per cell, which is precisely the step the array path does
# not take (it computes all four over the volume and copies). That difference is
# the thing being measured.
# ---------------------------------------------------------------------------

def _shift(field, axis, boundary, up):
    shifted = numpy.roll(field, -1 if up else 1, axis=axis)
    if boundary != "periodic":
        selector = [slice(None)] * 3
        selector[axis] = -1 if up else 0
        shifted[tuple(selector)] = numpy.float32(0.0)
    return shifted


def _curls(grid, boundaries, terms, operands, backward):
    """The six curl expressions, exactly as the device text groups them."""
    dtdx = numpy.float32(grid.dt / grid.dx)
    out = {}
    for term in terms:
        f1, f2 = operands[term.first], operands[term.second]
        sf = _shift(f1, term.first_axis, boundaries[term.first_axis], not backward)
        ss = _shift(f2, term.second_axis, boundaries[term.second_axis], not backward)
        curl = (dtdx * ((sf - f1) + (f2 - ss))).astype(numpy.float32)
        for axis in range(3):
            if IYEE_SHIFTS[term.target][axis] == 0 and boundaries[axis] == "metallic":
                selector = [slice(None)] * 3
                selector[axis] = 0
                curl[tuple(selector)] = numpy.float32(0.0)
        out[term.target] = curl
    return out


def _reference_no_pml(fields, grid, boundaries, terms, operands, backward):
    """``cond_apply`` / the plain tail, per component, in float32."""
    out = {}
    for term, curl in ((term, curl) for term, curl
                       in zip(terms, _curls(grid, boundaries, terms, operands,
                                            backward).values())):
        field = numpy.asarray(getattr(fields, term.target)).copy()
        condfac = fields.condfac_for(term.target)
        if condfac is None:
            out[term.target] = (field - curl).astype(numpy.float32)
        else:
            condinv = fields.condinv_for(term.target)
            out[term.target] = (((field * condfac) - curl) * condinv).astype(numpy.float32)
    return out


def _reference_pml(fields, layer, grid, boundaries, terms, operands, backward,
                   half_integer):
    """``cond_pml_apply`` / ``pml_apply``, per CELL, selecting one branch.

    The four case expressions are the module's own transcription of
    ``stepping._apply_conductive_pml_update``; here they are evaluated and then
    SELECTED, which is what a kernel does and what the array path does not.
    Returns the three fields, the three auxiliaries, the three histories, and the
    per-target case census the non-vacuity floor reads.
    """
    curls = _curls(grid, boundaries, terms, operands, backward)
    fields_out, aux_out, hist_out, census = {}, {}, {}, {}
    for term in terms:
        curl = curls[term.target]
        km1, si1 = stepping._curl_coefficients(layer, term.dsig, half_integer)
        km2, si2 = stepping._curl_coefficients(layer, term.dsigu, half_integer)
        km1, si1 = numpy.asarray(km1), numpy.asarray(si1)
        km2, si2 = numpy.asarray(km2), numpy.asarray(si2)
        field = numpy.asarray(getattr(fields, term.target)).copy()
        aux = numpy.asarray(getattr(fields, "fu_" + term.target)).copy()
        dsig = (km1 != 1.0) | (si1 != 1.0)
        dsigu = (km2 != 1.0) | (si2 != 1.0)
        dsig, dsigu = numpy.broadcast_to(dsig, field.shape), numpy.broadcast_to(dsigu, field.shape)
        census[term.target] = {
            "A": int((dsig & dsigu).sum()), "B": int((~dsig & dsigu).sum()),
            "C": int((dsig & ~dsigu).sum()), "D": int((~dsig & ~dsigu).sum())}
        condfac = fields.condfac_for(term.target)
        if condfac is None:
            new_aux = (((aux * km1) - curl) * si1).astype(numpy.float32)
            fields_out[term.target] = (
                (((field * km2) + new_aux) - aux) * si2).astype(numpy.float32)
            aux_out[term.target] = new_aux
            hist_out[term.target] = None
            continue
        condinv = fields.condinv_for(term.target)
        history = numpy.asarray(getattr(fields, "f_cond_" + term.target)).copy()
        c_new = (((history * condfac) - curl) * condinv).astype(numpy.float32)
        u_cond = (((aux * condfac) - curl) * condinv).astype(numpy.float32)
        u_split = ((((aux * km1) + c_new) - history) * si1).astype(numpy.float32)
        u_new = numpy.where(dsig, u_split, u_cond).astype(numpy.float32)
        f_split = ((((field * km2) + u_new) - aux) * si2).astype(numpy.float32)
        f_first = ((((field * km1) + c_new) - history) * si1).astype(numpy.float32)
        f_direct = (((field * condfac) - curl) * condinv).astype(numpy.float32)
        fields_out[term.target] = numpy.where(
            dsigu, f_split, numpy.where(dsig, f_first, f_direct)).astype(numpy.float32)
        aux_out[term.target] = numpy.where(dsigu, u_new, aux).astype(numpy.float32)
        hist_out[term.target] = numpy.where(dsig, c_new, history).astype(numpy.float32)
    return fields_out, aux_out, hist_out, census


def _operands(fields, sub_step):
    if sub_step == "step_B":
        return {name: numpy.asarray(fields.get_E(name)).copy()
                for name in ("Ex", "Ey", "Ez")}
    return {name: numpy.asarray(fields.get_H(name)).copy()
            for name in ("Hx", "Hy", "Hz")}


def _terms(sub_step):
    return (stepping.B_CURL_TERMS if sub_step == "step_B" else stepping.D_CURL_TERMS)


def _moved(before, after) -> int:
    return int((words(before) != words(after)).sum())


@pytest.mark.parametrize("sub_step", SUB_STEPS)
@pytest.mark.parametrize("mask", [(True, True, True), (True, False, False),
                                  (False, True, False), (False, False, True),
                                  (True, False, True)],
                         ids=["all", "x_only", "y_only", "z_only", "x_and_z"])
def test_the_no_pml_conductive_tail_is_stepping_bit_for_bit(sub_step, mask, xp):
    """ONE KERNEL SERVES A MIXED RUN, and this is the measurement that says so.

    A component with no sigma takes ``f - curl`` while its neighbours take
    ``((f*cf) - curl) * ci`` -- inside one sub-step, because ``_apply_curl`` reads
    the conductivity per TERM (stepping.py:508). Every mask is swept, including
    three with exactly one lossy component, and the comparison is uint32 words
    with a nonzero moved floor.
    """
    rng = numpy.random.default_rng(97)
    boundaries = ("periodic", "metallic", "periodic")
    cell = (8.0, 10.0, 12.0)
    probe = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries, xp=xp,
                 courant=0.5)
    targets = conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]
    live = [name for name, flag in zip(targets, mask) if flag]
    side = "d_sigma" if sub_step == "step_D" else "b_sigma"
    fields, layer, grid = make(xp, boundaries=boundaries, cell=cell, stored=True,
                               **{side: sigma_map(probe, live, rng)})
    assert layer is None
    seed_arrays(fields, grid, ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
                               "Ex", "Ey", "Ez"))
    assert conductive_kernels.conductive_targets(fields, sub_step) == mask

    kinds = stepping._boundary_kinds(grid, None)
    operands = _operands(fields, sub_step)
    before = {name: numpy.asarray(getattr(fields, name)).copy() for name in targets}
    expected = _reference_no_pml(fields, grid, kinds, _terms(sub_step),
                                 operands, backward=(sub_step == "step_D"))
    getattr(stepping, sub_step)(fields, None)

    for name in targets:
        assert _moved(before[name], getattr(fields, name)) > 0, (
            f"{name} did not move: an identity assertion over a sub-step that "
            f"changed nothing measures nothing")
        assert numpy.array_equal(words(expected[name]), words(getattr(fields, name))), (
            f"{sub_step} {name} at mask {mask}: the kernel's per-component tail is "
            f"not the array path's")


@pytest.mark.parametrize("sub_step", SUB_STEPS)
@pytest.mark.parametrize("mask", [(True, True, True), (True, False, False)],
                         ids=["all", "x_only"])
def test_the_four_case_pml_tail_is_stepping_bit_for_bit(sub_step, mask, xp):
    """The four subchunk cases, selected PER CELL, against the array path's copyto.

    THE NON-VACUITY FLOOR IS THE POINT. A layer absorbing on two axes and not on
    the third puts every one of MEEP's four cases somewhere in the volume for at
    least one target; the census below asserts a nonzero cell count in each, so a
    pass cannot mean "case A was right and the other three were never reached".
    """
    rng = numpy.random.default_rng(31)
    boundaries = ("metallic", "metallic", "periodic")
    cell = (12.0, 14.0, 10.0)
    probe = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries, xp=xp,
                 courant=0.5)
    targets = conductive_kernels.CONDUCTIVE_SUB_STEPS[sub_step]
    live = [name for name, flag in zip(targets, mask) if flag]
    side = "d_sigma" if sub_step == "step_D" else "b_sigma"
    fields, layer, grid = make(
        xp, boundaries=boundaries, cell=cell,
        absorber=((3, 3), (3, 3), (0, 0)),
        **{side: sigma_map(probe, live, rng)})
    assert layer is not None and layer.is_active
    seed_arrays(fields, grid, ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                               "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
                               "fu_Dx", "fu_Dy", "fu_Dz"))
    for name in live:
        getattr(fields, "f_cond_" + name)[...] = rng.uniform(
            -1.0, 1.0, size=grid.shape).astype(numpy.float32)

    kinds = stepping._boundary_kinds(grid, layer)
    operands = _operands(fields, sub_step)
    before = {name: numpy.asarray(getattr(fields, name)).copy() for name in targets}
    expected, aux, hist, census = _reference_pml(
        fields, layer, grid, kinds, _terms(sub_step), operands,
        backward=(sub_step == "step_D"), half_integer=(sub_step == "step_B"))

    reached = {case: max(counts[case] for counts in census.values())
               for case in ("A", "B", "C", "D")}
    assert all(count > 0 for count in reached.values()), (
        f"the fixture did not reach all four subchunk cases: {reached}; a pass "
        f"here would be a statement about the layer, not about the kernel")

    getattr(stepping, sub_step)(fields, layer)
    for name in targets:
        assert _moved(before[name], getattr(fields, name)) > 0
        assert numpy.array_equal(words(expected[name]), words(getattr(fields, name))), (
            f"{sub_step} {name} at mask {mask}: the per-cell four-case selection is "
            f"not the array path's copyto selection")
        assert numpy.array_equal(words(aux[name]),
                                 words(getattr(fields, "fu_" + name))), (
            f"{sub_step} fu_{name}: a tree that gets the field right and the "
            f"auxiliary wrong is correct for exactly one launch")
        if hist[name] is not None:
            assert numpy.array_equal(words(hist[name]),
                                     words(getattr(fields, "f_cond_" + name)))


def test_flattening_the_four_case_parentheses_diverges(xp):
    """The mutation control: the grouping the transcription refuses to lose.

    Without it, "bit-identical" above could be a statement about a comparator that
    cannot tell two float32 numbers apart. ``(((f*km2) + u_new) - u) * si2``
    flattened to ``((f*km2) + (u_new - u)) * si2`` is a different float32 number,
    and this measures that the comparison sees it.
    """
    rng = numpy.random.default_rng(5)
    boundaries = ("metallic", "metallic", "periodic")
    cell = (12.0, 14.0, 10.0)
    probe = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries, xp=xp,
                 courant=0.35)
    fields, layer, grid = make(xp, boundaries=boundaries, cell=cell, courant=0.35,
                               absorber=((3, 3), (3, 3), (0, 0)),
                               d_sigma=sigma_map(probe, ("Dx", "Dy", "Dz"), rng))
    seed_arrays(fields, grid, ("Dx", "Dy", "Dz", "Hx", "Hy", "Hz",
                               "fu_Dx", "fu_Dy", "fu_Dz"))
    for name in ("Dx", "Dy", "Dz"):
        getattr(fields, "f_cond_" + name)[...] = rng.uniform(
            -1.0, 1.0, size=grid.shape).astype(numpy.float32)

    kinds = stepping._boundary_kinds(grid, layer)
    operands = _operands(fields, "step_D")
    expected, _, _, _ = _reference_pml(fields, layer, grid, kinds,
                                       stepping.D_CURL_TERMS, operands,
                                       backward=True, half_integer=False)
    curls = _curls(grid, kinds, stepping.D_CURL_TERMS, operands, True)
    flattened = {}
    for term in stepping.D_CURL_TERMS:
        km1, si1 = stepping._curl_coefficients(layer, term.dsig, False)
        km2, si2 = stepping._curl_coefficients(layer, term.dsigu, False)
        field = numpy.asarray(getattr(fields, term.target)).copy()
        aux = numpy.asarray(getattr(fields, "fu_" + term.target)).copy()
        history = numpy.asarray(getattr(fields, "f_cond_" + term.target)).copy()
        cf, ci = fields.condfac_for(term.target), fields.condinv_for(term.target)
        c_new = (((history * cf) - curls[term.target]) * ci).astype(numpy.float32)
        u_split = ((((aux * km1) + c_new) - history) * si1).astype(numpy.float32)
        # THE FLATTENING, and only it.
        flattened[term.target] = (((field * km2) + (u_split - aux)) * si2).astype(
            numpy.float32)
    differing = sum(int((words(flattened[name]) != words(expected[name])).sum())
                    for name in ("Dx", "Dy", "Dz"))
    assert differing > 0, ("the flattened grouping produced the same float32 words; "
                           "the comparison above cannot be measuring the grouping")


# ---------------------------------------------------------------------------
# 3. THE PARTITION
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_step", SUB_STEPS)
def test_a_lossless_run_is_refused_because_admitting_it_would_be_an_overlap(sub_step, xp):
    fields, layer, grid = make(xp, stored=True)
    covered, reason = covers(fields, layer, grid, sub_step)
    assert not covered
    assert "no target of" in reason and "overlap" in reason


@pytest.mark.parametrize("sub_step,side", [("step_B", "b_sigma"),
                                           ("step_D", "d_sigma")])
def test_the_absorber_rows_configuration_is_admitted_at_both_curls(sub_step, side, xp):
    """MEEP's ``Absorber`` -- sigma on BOTH sides, no active layer -- at both curls.

    This is ``absorber-1d.py`` / ``TestAbsorber.test_absorber``'s shape and walls
    (1-D, periodic/periodic/metallic, ``stores_E`` True), which is 4 of the 7
    slots this family exists for.
    """
    rng = numpy.random.default_rng(7)
    cell = (0.0, 0.0, 24.0)
    boundaries = ("periodic", "periodic", "metallic")
    probe = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                 dimensions=1, xp=xp, courant=0.5)
    fields, layer, grid = make(
        xp, boundaries=boundaries, cell=cell, dimensions=1, stored=True,
        d_sigma=sigma_map(probe, ("Dx", "Dy", "Dz"), rng),
        b_sigma=sigma_map(probe, ("Bx", "By", "Bz"), rng))
    covered, reason = covers(fields, layer, grid, sub_step)
    assert covered, reason
    assert not coverage.covers_real_pml_curl(fields, layer, grid, sub_step)[0]
    assert not no_pml_curl.covers_no_pml_curl(fields, layer, grid, sub_step)[0]


def test_the_derived_arm_is_taken_when_nothing_stores_E(xp):
    """``TestAbsorber.test_absorber_2d``: sigma on both sides and ``stores_E`` False."""
    rng = numpy.random.default_rng(19)
    cell, boundaries = (12.0, 14.0, 0.0), ("metallic", "metallic", "periodic")
    probe = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                 dimensions=2, xp=xp, courant=0.5)
    fields, layer, grid = make(
        xp, boundaries=boundaries, cell=cell, dimensions=2,
        inhomogeneous_epsilon=True,
        d_sigma=sigma_map(probe, ("Dx", "Dy", "Dz"), rng),
        b_sigma=sigma_map(probe, ("Bx", "By", "Bz"), rng))
    assert not fields.stores_E
    assert conductive_kernels.conductive_arm(fields, layer, "step_B") == ("none", "derived_E")
    assert conductive_kernels.conductive_arm(fields, layer, "step_D") == ("none", "magnetic")
    for sub_step in SUB_STEPS:
        covered, reason = covers(fields, layer, grid, sub_step)
        assert covered, reason


def test_a_d_only_conductivity_under_pml_splits_the_two_curls(xp):
    """``TestAdjointSolver.test_damping``: the seventh slot, and the partition proof.

    ``step_B`` is the CERTIFIED pair's (no B sigma) and ``step_D`` is this
    family's. Both predicates are asked at both sub-steps, and the four verdicts
    must partition -- exactly one True per sub-step. An overlap here is what the
    union census scores as a widening rather than as coverage.
    """
    rng = numpy.random.default_rng(23)
    cell, boundaries = (12.0, 14.0, 0.0), ("metallic", "metallic", "periodic")
    probe = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                 dimensions=2, xp=xp, courant=0.5)
    fields, layer, grid = make(
        xp, boundaries=boundaries, cell=cell, dimensions=2,
        absorber=((3, 3), (3, 3), (0, 0)),
        d_sigma=sigma_map(probe, ("Dx", "Dy", "Dz"), rng))
    assert layer.is_active
    assert covers(fields, layer, grid, "step_D")[0]
    assert not covers(fields, layer, grid, "step_B")[0]
    assert coverage.covers_real_pml_curl(fields, layer, grid, "step_B")[0]
    assert not coverage.covers_real_pml_curl(fields, layer, grid, "step_D")[0]


def test_a_partially_conductive_sub_step_is_admitted_whole(xp):
    """One lossy component and two lossless is ONE launch, not two.

    The alternative -- refusing a mixed sub-step -- would give up a slot MEEP's
    own allocation granularity makes ordinary (``s->conductivity[c][d]``).
    """
    rng = numpy.random.default_rng(29)
    probe = Grid(resolution=1.0, cell_size=(8.0, 10.0, 12.0),
                 boundaries=("periodic", "metallic", "periodic"), xp=xp, courant=0.5)
    fields, layer, grid = make(
        xp, stored=True, d_sigma=sigma_map(probe, ("Dy",), rng))
    assert conductive_kernels.conductive_targets(fields, "step_D") == (False, True, False)
    covered, reason = covers(fields, layer, grid, "step_D")
    assert covered, reason


def test_an_unnamed_sub_step_raises_rather_than_refusing(xp):
    fields, layer, grid = make(xp, stored=True)
    with pytest.raises(ValueError, match="sub_step must be one of"):
        covers(fields, layer, grid, "update_E")


def test_a_float64_conductivity_volume_is_refused(xp):
    """The dtype is the ARITHMETIC, not just the launch.

    ``field * condfac`` is formed in ``result_type`` (stepping.py:1996), so a
    float64 table makes the whole tail a float64 tail and this float32 kernel is a
    different computation rather than a different rounding.
    """
    rng = numpy.random.default_rng(41)
    probe = Grid(resolution=1.0, cell_size=(8.0, 10.0, 12.0),
                 boundaries=("periodic", "metallic", "periodic"), xp=xp, courant=0.5)
    fields, layer, grid = make(xp, stored=True,
                               d_sigma=sigma_map(probe, ("Dx",), rng))
    fields._condfac["Dx"] = fields._condfac["Dx"].astype(numpy.float64)
    covered, reason = covers(fields, layer, grid, "step_D")
    assert not covered
    assert "float64" in reason and "condfac" in reason


def test_a_stray_history_on_a_lossless_component_is_refused_by_name(xp):
    """An allocated ``f_cond`` the COND == 0 branch would never write is stale state."""
    rng = numpy.random.default_rng(43)
    cell, boundaries = (12.0, 14.0, 0.0), ("metallic", "metallic", "periodic")
    probe = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                 dimensions=2, xp=xp, courant=0.5)
    fields, layer, grid = make(
        xp, boundaries=boundaries, cell=cell, dimensions=2,
        absorber=((3, 3), (3, 3), (0, 0)),
        d_sigma=sigma_map(probe, ("Dx",), rng))
    fields.f_cond_Dy = numpy.zeros(grid.shape, dtype=numpy.float32)
    covered, reason = covers(fields, layer, grid, "step_D")
    assert not covered
    assert "f_cond_Dy is allocated" in reason and "stale" in reason


def test_a_missing_history_under_an_active_layer_is_refused(xp):
    rng = numpy.random.default_rng(47)
    cell, boundaries = (12.0, 14.0, 0.0), ("metallic", "metallic", "periodic")
    probe = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                 dimensions=2, xp=xp, courant=0.5)
    fields, layer, grid = make(
        xp, boundaries=boundaries, cell=cell, dimensions=2,
        absorber=((3, 3), (3, 3), (0, 0)),
        d_sigma=sigma_map(probe, ("Dx",), rng))
    fields.f_cond_Dx = None
    covered, reason = covers(fields, layer, grid, "step_D")
    assert not covered
    assert "f_cond_Dx is not allocated" in reason


def test_the_clauses_behind_the_conductivity_one_are_still_reached(xp):
    """The proxy answers ONE question; every later clause must still run.

    A string filter over the certified predicate's refusal would have skipped
    these -- it SHORT-CIRCUITS, so forgiving its conductivity refusal after the
    fact would silently skip every clause behind it. The two probed here are
    chosen because they are structurally last (``stores_E`` and the array
    inventory) rather than by name: the clause list between them and the
    conductivity one is another family's to edit, and a test that pinned a
    particular neighbour would fail on someone else's widening instead of on a
    defect in this file.
    """
    rng = numpy.random.default_rng(53)
    cell, boundaries = (12.0, 14.0, 0.0), ("metallic", "metallic", "periodic")
    probe = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                 dimensions=2, xp=xp, courant=0.5)
    fields, layer, grid = make(
        xp, boundaries=boundaries, cell=cell, dimensions=2,
        absorber=((3, 3), (3, 3), (0, 0)),
        d_sigma=sigma_map(probe, ("Dx",), rng))
    assert covers(fields, layer, grid, "step_D")[0]

    # THE ARRAY INVENTORY, the delegated predicate's LAST clause. Ex is read by
    # neither this file's conductivity check nor step_D's launcher, so only the
    # delegation can refuse it.
    original = fields.Ex
    fields.Ex = original.astype(numpy.float64)
    covered, reason = covers(fields, layer, grid, "step_D")
    assert not covered
    assert "Ex" in reason and "float64" in reason
    fields.Ex = original

    # ``stores_E``, which sits between the conductivity clause and the inventory.
    fields._stored_E = False
    covered, reason = covers(fields, layer, grid, "step_D")
    assert not covered
    assert "stored" in reason


def test_every_kernel_the_arm_table_names_exists_and_vice_versa():
    assert set(conductive_kernels.ARM_KERNELS.values()) == set(
        conductive_kernels.CONDUCTIVE_KERNELS)


def test_sub_step_targets_match_coverage():
    """One transcription of MEEP's two term tables, pinned against the sibling's."""
    assert conductive_kernels.CONDUCTIVE_SUB_STEPS == dict(coverage.CURL_SUB_STEPS)
    for sub_step, targets in conductive_kernels.CONDUCTIVE_SUB_STEPS.items():
        terms = _terms(sub_step)
        assert tuple(term.target for term in terms) == targets


def test_the_coefficient_pairs_are_stepping_s_own_cycle():
    """``_COEFFICIENT_PAIRS`` is matched as TEXT against the certified body.

    It is also a transcription of ``vec.hpp``'s cycle, so it is pinned against
    ``stepping``'s term table here -- a half-cell error introduced by "fixing" the
    pair table would otherwise show up only as a wrong absorber.
    """
    letters = {"x": 0, "y": 1, "z": 2}
    for terms in (stepping.B_CURL_TERMS, stepping.D_CURL_TERMS):
        for term, (first, second) in zip(terms, conductive_kernels._COEFFICIENT_PAIRS):
            assert first[0] == term.dsig and second[0] == term.dsigu
            assert letters[first[0]] in range(3)


def test_admission_record_is_all_or_nothing():
    """No half-written device verdict.

    While ``host`` is None nothing in the record may claim a device measurement;
    once it is set, the fields a verdict rests on must all be present. The two
    states are the only legal ones -- the same rule ``no_pml_curl`` carries.
    """
    record = conductive_kernels.CONDUCTIVE_CURL_ADMISSION
    claims = ("artifacts", "recorded_utc", "device", "cases_scored",
              "single_launch_identical", "multi_step_identical",
              "kernel_template_sha256", "source_mutations_caught",
              "source_mutations_escaped", "verdict_flips_against_planted_defect")
    if record["host"] is None:
        assert all(record.get(key) is None for key in claims), (
            "a device claim is recorded with no host: the record is half-written")
        return
    assert all(record.get(key) is not None for key in claims)
    assert record["source_mutations_escaped"] == 0
    assert record["verdict_flips_against_planted_defect"] is True
    # THE DIGESTS PIN THE TEMPLATES THAT SHIP TODAY. A record cut against one
    # emitter and read beside another is a device verdict for bytes that no longer
    # exist -- and because the templates are BUILT from two siblings' source, an
    # edit over there moves them without anything in this file changing.
    for name, digest in record["kernel_template_sha256"].items():
        live = hashlib.sha256(
            conductive_kernels.kernel_template(name).encode("utf-8")).hexdigest()
        assert live == digest, (
            f"{name}'s device text has changed since the gate ran: the record "
            f"pins {digest[:12]} and the shipped template hashes {live[:12]}")
    assert set(record["kernel_template_sha256"]) == set(
        conductive_kernels.CONDUCTIVE_KERNELS)
    assert record["single_launch_identical"] == record["multi_step_identical"]
    assert (record["slots_after"] - record["slots_before"]
            == record["slots_gained"] == 7)


# ---------------------------------------------------------------------------
# 5. THE GATE'S NUMPY BACKEND, IN THE MERGE BAR
#
# The device gate is the artifact that certifies this family, and its NumPy leg
# is a transcription of the SAME device tree this file pins textually. Running a
# few of its cases here keeps the two from drifting between device runs: a
# transformation that changed the emitted tail would keep passing the byte
# equalities above (they strip the tail) and fail here.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gate():
    import importlib.util
    import sys

    path = (HERE.parent.parent / "parity" / "meep_gpu"
            / "gate_cuda_conductive.py")
    spec = importlib.util.spec_from_file_location("gate_cuda_conductive", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gate_cuda_conductive", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("label", ["3d_mixed", "1d_metallic", "3d_pml_all_axes",
                                   "2d_pml_two_axes"])
@pytest.mark.parametrize("sub_step", SUB_STEPS)
@pytest.mark.parametrize("mask_index", [0, 1], ids=["CCC", "C--"])
def test_the_gate_numpy_backend_reproduces_stepping(gate, label, sub_step,
                                                    mask_index):
    """Each case is IDENTICAL, or FLOORED for a stated reason -- never divergent.

    A floored case is asserted rather than skipped: the floor's own numbers are
    checked, so "this fixture could not reach two of the four cases" is recorded
    as the measurement it is instead of vanishing as a skip.
    """
    spec = gate._spec_by_label(label)
    mask = gate.COND_MASKS[mask_index]
    storage = "stored" if gate.spec_is_active(spec) else "derived"
    case = gate.one_case("numpy", spec, sub_step, mask, storage, 0.35, "uniform",
                         "shipped", "fmad_false", gate.LICENSED_SUBSTITUTION)
    if case.get("skipped"):
        assert "four of MEEP's subchunk cases" in case["skipped"], case["skipped"]
        reached = case["pml_case_census"]["reached"]
        assert sum(reached.values()) > 0, (
            "a case floored for reaching no subchunk case at all is a broken "
            "fixture, not a reduced-dimension fact")
        assert min(reached.values()) == 0
        return
    assert case["single_launch"]["bit_identical"], case["localization_summary"]
    assert case["multi_step"]["bit_identical"]
    assert case["oracle_moved"] > 0.0


def test_the_gate_can_fail(gate):
    """A misdeclared code triple must DIVERGE, or the gate measures nothing."""
    spec = gate._spec_by_label("3d_mixed")
    case = gate.one_case("numpy", spec, "step_D", gate.COND_MASKS[0], "stored",
                         0.35, "uniform", "shipped", "fmad_false", "all_periodic")
    assert not case.get("skipped"), case.get("skipped")
    assert not case["single_launch"]["bit_identical"]
    assert case["localization_summary"]["differing_words"] > 0


def test_the_gate_refuses_a_layer_that_reaches_only_some_subchunk_cases(gate):
    """The non-vacuity floor SKIPS rather than passes, and prints the census.

    On a 2-D grid, a target whose ``dsigu`` ladder runs along the invariant axis
    can never reach cases A or B. That is a true fact about 2-D and not a
    degenerate fixture, so the case is skipped with the counts in the message
    rather than silently counted as evidence for a branch it never ran.
    """
    spec = gate._spec_by_label("2d_pml_two_axes")
    case = gate.one_case("numpy", spec, "step_B", gate.COND_MASKS[1], "stored",
                         0.35, "uniform", "shipped", "fmad_false",
                         gate.LICENSED_SUBSTITUTION)
    assert case.get("skipped")
    assert "four of MEEP's subchunk cases" in case["skipped"]
    assert case["pml_case_census"]["reached"]["D"] > 0


def test_the_near_one_table_lands_strictly_below_one(gate, xp):
    """``1.0 - 1e-7`` rounds to 1.0 in float32; the fixture must not be a no-op.

    The whole tolerance experiment turns on the perturbed entry being a DIFFERENT
    float32 from 1.0 while a 1e-7 tolerance still calls it one. Measured here
    rather than assumed, because a fixture that silently collapsed to 1.0 would
    make the tolerance mutation come back UNCAUGHT on both tables and read as
    evidence that the exact comparison does not matter.
    """
    assert gate.NEAR_ONE != numpy.float32(1.0)
    assert abs(float(gate.NEAR_ONE) - 1.0) < 1e-7
    # ...and it is the LARGEST such float32: nothing sits between it and 1.0, so
    # the fixture puts the entry as close to the boundary as float32 allows.
    assert numpy.nextafter(gate.NEAR_ONE, numpy.float32(2.0),
                           dtype=numpy.float32) == numpy.float32(1.0)
    grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0),
                boundaries=("metallic", "metallic", "periodic"), xp=xp,
                courant=0.5)
    layer = PML(grid=grid, thickness=((2, 2), (2, 2), (0, 0)))
    record = gate.install_near_one(layer)
    assert record["installed"] and record["entries"]
    touched = [numpy.asarray(getattr(layer, e["attribute"])).reshape(-1)[e["index"]]
               for e in record["entries"]]
    assert all(value == gate.NEAR_ONE for value in touched)


def test_every_named_mutation_resolves_and_arms(gate):
    """Every name in the battery exists, and arms where the record says it should.

    THIS TEST EXISTS BECAUSE A DEVICE LEG DIED WITHOUT IT. A mutation registered
    after ``SOURCE_MUTATIONS_ALL`` was built raised ``KeyError`` forty minutes
    into a GPU-host run -- a failure of the battery's own bookkeeping that looks
    nothing like a kernel defect and costs a GPU hour to find. Resolving every
    name and applying every transform to every shipped template is a laptop check.
    """
    for name in gate.SOURCE_MUTATIONS:
        assert name in gate.SOURCE_MUTATIONS_ALL, (
            f"{name} is in the battery but not in SOURCE_MUTATIONS_ALL -- most "
            f"likely registered after that dict was built")
    for name in gate.NULL_SOURCE_MUTATIONS:
        assert name in gate.SOURCE_MUTATIONS, f"{name} is a null of nothing"
    for kernel in conductive_kernels.CONDUCTIVE_KERNELS:
        template = conductive_kernels.kernel_template(kernel)
        layer = ("active" if kernel in conductive_kernels._PML_KERNELS else "none")
        for name in gate.SOURCE_MUTATIONS:
            mutated, sites = gate.SOURCE_MUTATIONS_ALL[name](template)
            required = gate.SOURCE_MUTATION_REQUIRES_LAYER.get(name, "unset")
            expected_absent = required not in ("unset", None) and required != layer
            if expected_absent:
                assert sites == 0, (
                    f"{kernel}: {name} matched {sites} site(s) in a kernel whose "
                    f"tail it does not belong to -- it would arm on dead code")
            else:
                assert sites > 0 and mutated != template, (
                    f"{kernel}: {name} matched {sites} site(s) and changed "
                    f"nothing; a mutation that cannot arm scores NOT ARMED and "
                    f"measures nothing")


def test_every_filtered_mutation_has_a_leg_somewhere_in_the_full_plan(gate):
    """A leg filter that excludes every spec has dropped a defect, not protected one.

    The filters are what stop a structurally inert leg (a mask mutation on a grid
    with no such wall, the index-decomposition swap on a flat grid) from reading
    as a kernel defect. That machinery can itself silence a defect, so the full
    spec list is checked to carry at least one leg for each filtered mutation --
    on the laptop, before a device leg spends an hour discovering it.
    """
    for name, leg_filter in gate.SOURCE_MUTATION_LEG_FILTER.items():
        carriers = [spec["label"] for spec in gate.SPECS
                    if leg_filter(*gate.spec_codes_and_shape(spec))]
        assert carriers, (
            f"no spec in the sweep satisfies {name}'s requirement "
            f"({gate.SOURCE_MUTATION_LEG_REQUIREMENT.get(name)})")
        assert any(label in gate.MUTATION_SPEC_LABELS for label in carriers), (
            f"{name} has carriers {carriers} but none is in MUTATION_SPEC_LABELS, "
            f"so no mutation leg would ever be built for it")
