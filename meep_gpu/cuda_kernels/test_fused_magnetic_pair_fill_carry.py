"""The two in-seam mirror fills, carried inside the fused magnetic launch.

WHAT THIS FILE PINS THAT NO OTHER ONE DOES. Until 2026-08-28
``fused_magnetic_pair`` refused every mirrored axis by name, on a reason that was
true of the NAIVE carry and only of it: a thread standing on a fill's destination
would READ a post-curl ``B`` produced by a thread in another block, and an
ordinary CUDA launch has no grid-wide barrier at which that read is defined. The
OWNERSHIP INVERSION removes the read instead of synchronising it -- the SOURCE
thread writes the destination from the register it already holds -- and that
turns one refusal into two claims, each of which needs its own measurement:

1. **THE ARITHMETIC.** The composed closed form
   :func:`~.fused_magnetic_pair.carried_destinations` implements really is what
   the array path's ``fill_symmetry_bc_B`` -> ``zero_metal_B`` ->
   ``fill_folded_far_ghosts_B`` leave behind, to the WORD, including the signed
   zero a far image of a cleared row carries. Measured here against
   ``stepping``'s own three passes in driver order, over nine configurations --
   the CUDA twin of ``results/metal_folded_far_carry_2026-08-20/ownership.json``,
   run on NumPy so it is merge-bar evidence rather than a device-gated claim.

2. **THE RACE.** No cell is written by two threads, and no cell is left unwritten.
   Measured off the EMITTED DEVICE TEXT rather than off the Python tables that
   produced it: :func:`write_map` scans the kernel body, tracks the guard stack by
   brace depth, and runs every thread of a launch. A carry that raced would show
   as a cell with two writers; a carry that dropped a cell would show as a hole,
   and BOTH directions are asserted, because a kernel that wrote nothing at all
   would pass the first check on its own.

WHY THE FIRST CHECK CANNOT BE SKIPPED EVEN THOUGH A DEVICE GATE EXISTS. The
device gate compares complete driver steps and would catch a wrong parity -- but
only on a fixture that drives one. The ordering question the closed form settles
(the fills are applied axis by axis, the wall clear sits between them, and a
corner carries the PRODUCT of two parities) is invisible unless two axes are
folded at once, which is a fixture property, not a kernel property. This file
sweeps the fold shapes; the device gate measures the bytes.

CUPY IS NOT INSTALLED ON THE MERGE-BAR HOST and this file does not pretend
otherwise. The predicate and the emitter need no device -- the emitter needs only
the three certified modules' device strings -- so the certified halves are loaded
BY PATH under private names, with ``cupy`` stubbed only for the duration of that
load (``conftest``'s sanctioned scope). Nothing here launches anything.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import types

import numpy
import pytest

from .. import stepping
from ..fields import Fields, mirror_parity
from ..grid import Grid, Mirror
from ..pml import PML
from . import fused_magnetic_pair as family
from . import in_seam_coverage

B_NAMES = ("Bx", "By", "Bz")

#: Every array the fused kernel writes, in the order the signature binds them.
#: The write enumeration requires EVERY cell of EVERY one of these to be written
#: exactly once, which is what makes it a coverage floor and not only a race check.
WRITTEN_ARRAYS = ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz",
                  "Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")


# ---------------------------------------------------------------------------
# The emitter, on a host with no CuPy
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- this directory's standing stand-in."""

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


@pytest.fixture(scope="module")
def emitted() -> str:
    """The shipped device text, spliced on a laptop.

    The three certified modules are loaded BY PATH under private names -- the same
    route ``fused_magnetic_pair`` itself falls back to when it is loaded outside the
    package -- so nothing lands in ``sys.modules`` under a package name and the
    ``cupy`` stand-in is removed before the first assertion runs. A test that needed
    a device would be a skip here, and a skip is not evidence.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    previous = sys.modules.get("cupy")
    sys.modules["cupy"] = _NumpyWearingCupysName()
    loaded = {}
    try:
        for stem in ("step_curl_kernels", "constitutive_kernels", "in_seam_passes"):
            spec = importlib.util.spec_from_file_location(
                f"_certified_{stem}", os.path.join(here, f"{stem}.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded[stem] = module
    finally:
        if previous is None:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = previous
    held = {name: getattr(family, name) for name in loaded}
    for name, module in loaded.items():
        setattr(family, name, module)
    try:
        return family.fused_magnetic_pair_source()
    finally:
        for name, module in held.items():
            setattr(family, name, module)


# ---------------------------------------------------------------------------
# The fixtures: folded grids, on NumPy
# ---------------------------------------------------------------------------

def build(axes: str = "", phases=(), boundaries=None, extent: float = 2.0,
          cross: float = 2.4, depth: float = 1.6, resolution: float = 4.0):
    """A ``(fields, grid, pml)`` triple with the named axes mirror-folded.

    ``boundaries`` is what decides a folded axis's TERMINATION, and the two are
    different kernels' worth of behaviour rather than a detail: a folded PERIODIC
    axis stores MEEP's not-owned slot past ``big_corner`` and runs the FAR fill, a
    folded METALLIC one does not and runs only the NEAR fill.

    ``extent`` decides the full count's PARITY, which decides the reflect row --
    ``_far_reflect_rows`` is ``n_full - stored + 2``, which is ``stored - 2`` at an
    even full count and ``stored - 3`` at an odd one. A sweep carrying only the even
    one measures nothing about the row. At this resolution 2.0 is the even count and
    2.25 the odd one, and :func:`test_the_fold_sweep_reaches_every_shape_the_carry_emits`
    requires both.

    THE GRIDS ARE SMALL ON PURPOSE. :func:`write_map` runs every thread of a launch
    in the interpreter, so the shape is a runtime budget; nothing measured here
    scales with it, because both fills write whole PLANES and a 6-cell axis carries
    the same destination structure a 60-cell one does.
    """
    size = [float(cross), float(cross), float(depth)]
    for name in axes:
        size["XYZ".index(name)] = float(extent)
    grid = Grid(resolution=float(resolution), cell_size=tuple(size),
                dimensions=3 if depth else 2, courant=0.35,
                symmetry=tuple(Mirror(name, int(phase))
                               for name, phase in zip(axes, phases)),
                boundaries=boundaries, xp=numpy)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    folded = {"XYZ".index(name) for name in axes}
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if axis in folded
                      else (2, 2)
                      for axis in range(3))
    return fields, grid, PML(grid=grid, thickness=thickness)


#: The fold shapes swept, and every one is here for a property no other row has.
#: ``label, kwargs`` -- the shapes and reflect rows are read off the built grid.
FOLDS = (
    ("unfolded", {}),
    ("unfolded_wall_z", {"boundaries": {"z": "metallic"}}),
    ("y_periodic_even", {"axes": "Y", "phases": (1,)}),
    ("y_periodic_odd", {"axes": "Y", "phases": (-1,), "extent": 2.25}),
    ("y_metallic", {"axes": "Y", "phases": (1,), "boundaries": {"y": "metallic"}}),
    ("y_periodic_wall_z", {"axes": "Y", "phases": (1,),
                           "boundaries": {"z": "metallic"}}),
    ("xy_periodic_mixed_phase", {"axes": "XY", "phases": (1, -1)}),
    ("xy_periodic_odd", {"axes": "XY", "phases": (1, -1), "extent": 2.25}),
    ("xy_mixed_termination", {"axes": "XY", "phases": (1, 1),
                              "boundaries": {"y": "metallic"}}),
    ("xyz_periodic", {"axes": "XYZ", "phases": (1, -1, 1)}),
)


# ---------------------------------------------------------------------------
# 1. THE ARITHMETIC -- the closed form against the array path's own three passes
# ---------------------------------------------------------------------------

def _face(axis, index):
    return tuple(index if d == axis else slice(None) for d in range(3))


def _words(array):
    return numpy.ascontiguousarray(array, dtype=numpy.float32).ravel().view(
        numpy.uint32)


@pytest.mark.parametrize("label,kwargs", FOLDS, ids=[row[0] for row in FOLDS])
def test_the_closed_form_is_what_the_three_passes_leave(label, kwargs):
    """``carried_destinations``' composition, byte-compared to ``stepping`` itself.

    THE CUDA TWIN OF ``metal_folded_far_carry_2026-08-20/ownership.json``. The
    array path runs its own ``fill_symmetry_bc_B`` -> ``zero_metal_B`` ->
    ``fill_folded_far_ghosts_B`` in DRIVER ORDER (driver.py:3294-3296); the model
    applies the wall clear FIRST and then one composed multiply per destination
    subset. Those two orders agree only because the near fill's destination
    (stored 0 of a FOLDED axis) and the clear's (stored 0 of a WALLED axis) are
    disjoint per component -- and this comparison is what says so rather than the
    comment that claims it.

    Compared as uint32 WORDS, never ``allclose``: the far weight is ``-phase`` and
    a far image of a cleared row is ``-1.0f * +0.0f``, which is ``-0.0f`` on both
    paths and which ``==`` would call equal to ``+0.0f``.
    """
    fields, grid, _pml = build(**kwargs)
    phases = in_seam_coverage.mirror_fill_phases(grid)
    reflect = in_seam_coverage.folded_far_rows(grid)
    walls = in_seam_coverage.zero_metal_axes(grid)
    rng = numpy.random.default_rng(20260828)
    stepped = {name: (rng.standard_normal(grid.shape) * 0.37).astype(numpy.float32)
               for name in B_NAMES}
    for name in B_NAMES:
        getattr(fields, name)[...] = stepped[name]

    stepping.fill_symmetry_bc_B(fields)
    stepping.zero_metal_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)

    moved = 0
    for target, name in enumerate(B_NAMES):
        near = tuple(axis for axis in family.near_fill_axes(target)
                     if phases[axis] is not None)
        far = tuple(axis for axis in family.far_fill_axes(target)
                    if reflect[axis] is not None)
        # The wall clear, on the value the fills read. zero_metal reaches component
        # m at stored 0 of axis m only, and skips a folded axis.
        base = stepped[name].copy()
        for axis in range(3):
            if walls[axis] and in_seam_coverage.IYEE_SHIFTS[name][axis] == 0:
                base[_face(axis, 0)] = numpy.float32(0.0)
        model = base.copy()
        for subset, carries_near in family.carried_destinations(near, far):
            source = [slice(None)] * 3
            destination = [slice(None)] * 3
            weight = 1
            for axis in subset:
                source[axis] = int(reflect[axis])
                destination[axis] = int(grid.shape[axis]) - 1
                weight *= int(mirror_parity(name, axis, phases[axis]))
            if carries_near:
                axis = near[0]
                source[axis] = family.NEAR_SOURCE_INDEX
                destination[axis] = 0
                weight *= int(mirror_parity(name, axis, phases[axis]))
            model[tuple(destination)] = numpy.float32(weight) * base[tuple(source)]
        got = numpy.array(getattr(fields, name), copy=True)
        differing = int(numpy.count_nonzero(_words(got) != _words(model)))
        assert differing == 0, (
            f"{label}/{name}: the closed form differs from stepping's own three "
            f"passes in {differing} words (near={near} far={far} "
            f"reflect={reflect} walls={walls})")
        moved += int(numpy.count_nonzero(_words(got) != _words(stepped[name])))
    if any(phases) or any(walls):
        assert moved > 0, (
            f"{label}: the three passes changed no word at all, so this row agrees "
            f"with the closed form only because neither did anything")


def test_the_fold_sweep_reaches_every_shape_the_carry_emits():
    """The sweep is not vacuous: one, two and three folded axes, both terminations.

    A NULL CONTROL FOR THE PARAMETRISATION ITSELF. The composition question --
    does a corner carry the PRODUCT of two parities -- is invisible with one
    folded axis, so a sweep that happened to build only single folds would agree
    with any closed form at all.
    """
    seen = set()
    terminations = set()
    for _label, kwargs in FOLDS:
        _fields, grid, _pml = build(**kwargs)
        phases = in_seam_coverage.mirror_fill_phases(grid)
        rows = in_seam_coverage.folded_far_rows(grid)
        seen.add(sum(1 for axis in range(3) if phases[axis] is not None))
        for axis in range(3):
            if phases[axis] is not None:
                terminations.add(rows[axis] is not None)
    assert seen == {0, 1, 2, 3}, f"folded-axis counts swept: {sorted(seen)}"
    assert terminations == {True, False}, (
        "the sweep carries only one fold termination; a folded METALLIC axis runs "
        "the near fill and NOT the far one, and a mutation caught on one is not "
        "caught on the other")


def test_a_far_image_of_a_cleared_row_carries_the_signed_zero():
    """The one place the driver's ORDER between the clear and the far fill shows.

    ``zero_metal_B`` (driver.py:3295) runs BEFORE ``fill_folded_far_ghosts_B``
    (:3296), so a far image of a cleared row is ``-phase * +0.0f``. The kernel
    reads its register AFTER the clear lines for exactly this reason, and if it
    read before it, the ghost would carry a stepped value instead of a signed zero
    -- which ``allclose`` and ``==`` would both call correct.
    """
    fields, grid, _pml = build(axes="Y", phases=(1,),
                               boundaries={"z": "metallic"})
    walls = in_seam_coverage.zero_metal_axes(grid)
    reflect = in_seam_coverage.folded_far_rows(grid)
    assert walls[2] and reflect[1] is not None, (walls, reflect)
    for name in B_NAMES:
        getattr(fields, name)[...] = numpy.float32(0.25)
    stepping.fill_symmetry_bc_B(fields)
    stepping.zero_metal_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    # Bz clears on the z wall (the B diagonal) and is far-imaged along y
    # (iyee[Bz][y] == 1), so its top-y plane at k == 0 is a far image of a
    # cleared row.
    ghost = numpy.array(fields.Bz[:, grid.shape[1] - 1, 0])
    assert numpy.all(_words(ghost) == numpy.uint32(0x80000000)), (
        "the far image of the cleared plane is not -0.0f; the register was read "
        "before the wall clear rather than after it")


# ---------------------------------------------------------------------------
# 2. THE RACE -- every subscripted write, enumerated off the emitted device text
# ---------------------------------------------------------------------------

_IF_BLOCK = re.compile(r"^\s*if\s*\((.+)\)\s*\{$")
_IF_INLINE = re.compile(r"^\s*if\s*\((.+?)\)\s*\{(.+)\}$")
_IF_STATEMENT = re.compile(r"^\s*if\s*\((.+?)\)\s+([^{}]+;)$")
_STORE = re.compile(r"^\s*(\w+)\[([^\]]+)\]\s*=\s*[^;]+;$")
_PML_APPLY = re.compile(
    r"^\s*(\w+)\s*=\s*pml_apply_reg_pre\((\w+),\s*(\w+),\s*(\w+),.*,\s*(\w+),"
    r"\s*pre_fu_\w+,\s*pre_b_\w+\);$")
_CONSTITUTIVE = re.compile(
    r"^\s*constitutive_apply(?:_pre)?\((\w+),\s*(\w+),\s*(\w+),.*\);$")
#: THE OWN-CELL HOIST: one guarded (or, for fu, unguarded) load of this thread's own
#: word into a pre_* register. It writes no memory, and its register is never read by
#: a guard or an index, so it contributes nothing to the write set.
_HOISTED_LOAD = re.compile(
    r"^\s*float\s+pre_\w+\s*=\s*(?:own_[xyz]\s*\?\s*)?\w+\[idx\](?:\s*:\s*0\.0f)?;$")
_DECLARATION = re.compile(r"^\s*(?:const\s+)?(?:int|float)\s+(\w+)\s*=\s*(.+);$")
_ASSIGNMENT = re.compile(r"^\s*(\w+)\s*=\s*([^;]+);$")


def _pythonic(expression: str) -> str:
    """One C expression as Python, for the guards and the index arithmetic.

    ``/`` BECOMES ``//``, and that is the difference between reading the kernel and
    reading something else: every division reachable from a guard or an index in
    this body is C INTEGER division on the thread index (``idx / nz``,
    ``idx / (ny * nz)`` -- the certified decode). Python's ``/`` is float division,
    which turns ``i`` into ``0.666`` and every ``i == 0`` into False -- an
    enumeration that would report a clean write map for a kernel that races.
    :func:`test_the_enumerations_decode_is_the_kernels_own` is the control.
    """
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?=[\(A-Za-z_])", " not ", expression)
    expression = expression.replace("/", "//")
    return expression.replace("0.0f", "0.0").strip()


def _scan(line, guards):
    """One statement as ``(guards, kind, payload)`` entries. Raises on anything new.

    RAISING IS THE POINT. A statement shape this scanner does not know is a write
    it might be missing, and a race check that silently skipped a store would
    report "no cell written twice" about a kernel it had not read.
    """
    if line.strip() == "return;":
        return [(guards, "unreachable", None)]
    if _HOISTED_LOAD.match(line):
        return []
    match = _PML_APPLY.match(line)
    if match:
        register, field, auxiliary, index, owned = match.groups()
        return [(guards, "clear", register),
                (guards, "write", (auxiliary, index)),
                (guards + [owned], "write", (field, index)),
                (guards + [owned], "let", (register, "1.0"))]
    match = _CONSTITUTIVE.match(line)
    if match:
        field, auxiliary, index = match.groups()
        return [(guards, "write", (field, index)),
                (guards, "write", (auxiliary, index))]
    match = _STORE.match(line)
    if match:
        return [(guards, "write", (match.group(1), match.group(2)))]
    match = _DECLARATION.match(line) or _ASSIGNMENT.match(line)
    if match:
        return [(guards, "let", (match.group(1), match.group(2)))]
    raise AssertionError(
        f"the write enumeration does not recognise {line.strip()!r}; a statement "
        f"shape it cannot read is a write it may be missing")


def kernel_body(source: str) -> str:
    """The emitted entry point's body: everything between its brace and its close."""
    body = source.split("\n) {\n")[-1]
    assert body.endswith("}\n"), body[-60:]
    return body[: -len("}\n")]


def program(source: str):
    """Every write the emitted kernel body performs, with the guards in scope."""
    out, stack, depth = [], [], 0
    for raw in kernel_body(source).splitlines():
        line = raw.split("//")[0].rstrip()
        if not line.strip():
            continue
        inline = _IF_INLINE.match(line)
        if inline:
            for part in inline.group(2).split(";"):
                if part.strip():
                    out.extend(_scan("    " + part.strip() + ";",
                                     stack + [inline.group(1)]))
            continue
        single = _IF_STATEMENT.match(line)
        if single and not line.rstrip().endswith("{"):
            out.extend(_scan("    " + single.group(2), stack + [single.group(1)]))
            continue
        block = _IF_BLOCK.match(line)
        if block:
            stack.append(block.group(1))
            depth += 1
            continue
        if line.strip() == "{":
            stack.append("1")
            depth += 1
            continue
        if line.strip() == "}":
            stack.pop()
            depth -= 1
            continue
        out.extend(_scan(line, list(stack)))
    assert depth == 0, f"unbalanced braces in the emitted body ({depth})"
    # COMPILED ONCE, not per thread. A launch is tens of thousands of guard
    # evaluations and ``eval`` on a string re-parses every one of them; the
    # enumeration is a merge-bar test, so this is the difference between two
    # minutes and two seconds.
    return tuple(
        (tuple(compile(_pythonic(g), "<guard>", "eval") for g in guards),
         kind,
         (payload[0], compile(_pythonic(payload[1]), "<expr>", "eval"))
         if kind in ("write", "let") else payload,
         (payload[1] if kind in ("write", "let") else payload))
        for guards, kind, payload in out)


def write_map(source, shape, near, reflect, walls, codes, phases):
    """``(array, flat cell) -> [thread ids]`` for one whole launch."""
    nx, ny, nz = shape
    statements = program(source)
    writes = {}
    for idx in range(nx * ny * nz):
        env = {
            "idx": idx, "nx": nx, "ny": ny, "nz": nz,
            "sx": ny * nz, "sy": nz, "sz": 1,
            "k": idx % nz, "j": (idx // nz) % ny, "i": idx // (ny * nz),
            "near_x": near[0], "near_y": near[1], "near_z": near[2],
            "reflect_x": reflect[0], "reflect_y": reflect[1], "reflect_z": reflect[2],
            "wall_x": walls[0], "wall_y": walls[1], "wall_z": walls[2],
            "bc_x": codes[0], "bc_y": codes[1], "bc_z": codes[2],
            "phase_x": phases[0], "phase_y": phases[1], "phase_z": phases[2],
            "BC_PERIODIC": 0, "BC_METALLIC": 1, "BC_MIRROR_PERIODIC": 2,
        }
        for guards, kind, payload, spelling in statements:
            if not all(eval(g, {}, env) for g in guards):  # noqa: S307
                continue
            if kind == "unreachable":
                raise AssertionError(
                    f"thread {idx} of {shape} reached an early return; this "
                    f"enumeration runs only in-range threads")
            if kind == "clear":
                env[payload] = 0.0
            elif kind == "let":
                name, value = payload
                try:
                    env[name] = eval(value, {}, env)  # noqa: S307
                except (NameError, TypeError):
                    # A value this pass does not model -- the thread index, a
                    # ghost gather, a coefficient load. All are READS. A guard or
                    # an index that needed one would raise below rather than
                    # silently dropping a write.
                    if name != "idx":
                        env.pop(name, None)
            else:
                array, index = payload
                cell = int(eval(index, {}, env))  # noqa: S307
                assert 0 <= cell < nx * ny * nz, (
                    f"{array}[{spelling}] = {cell} is outside the {shape} volume "
                    f"from thread {idx}")
                writes.setdefault((array, cell), []).append(idx)
    return writes


def test_the_enumerations_decode_is_the_kernels_own(emitted):
    """The control for :func:`_pythonic`'s integer-division rewrite.

    The enumeration SUPPLIES ``i``, ``j``, ``k`` and then lets the kernel's own
    decode overwrite them. Those two must agree at every thread, or every guard
    below is being evaluated at a coordinate the kernel never has -- which reads as
    a clean write map for a kernel that races.
    """
    shape = (5, 6, 7)
    statements = program(emitted)
    decode = [payload for _guards, kind, payload, _spelling in statements
              if kind == "let" and payload[0] in ("i", "j", "k", "sx", "sy", "sz")]
    assert len(decode) == 6, [name for name, _ in decode]
    nx, ny, nz = shape
    for idx in range(nx * ny * nz):
        env = {"idx": idx, "nx": nx, "ny": ny, "nz": nz}
        for name, code in decode:
            env[name] = eval(code, {}, env)  # noqa: S307
        assert (env["i"], env["j"], env["k"]) == (
            idx // (ny * nz), (idx // nz) % ny, idx % nz), (idx, env)
        assert (env["sx"], env["sy"], env["sz"]) == (ny * nz, nz, 1)


def _launch_arguments(grid):
    """The runtime plan the launcher would build, plus the curl's boundary codes."""
    fills = family.fused_magnetic_pair_fills(grid)
    codes = tuple(
        2 if in_seam_coverage.stored_past_owned(grid)[axis]
        else 1 if (grid.is_metallic(axis) or grid.is_mirrored(axis))
        else 0
        for axis in range(3))
    return fills, codes


@pytest.mark.parametrize("label,kwargs", FOLDS, ids=[row[0] for row in FOLDS])
def test_no_cell_is_written_by_two_threads(label, kwargs, emitted):
    """THE RACE PROOF, off the emitted device text and not off the tables.

    A launch has no grid-wide barrier, so two threads writing one word is not a
    torn value the gate would catch as a divergence -- it is an UNDEFINED one that
    can be right on the machine it was measured on and wrong on the next.
    """
    _fields, grid, _pml = build(**kwargs)
    fills, codes = _launch_arguments(grid)
    shape = tuple(int(n) for n in grid.shape)
    writes = write_map(emitted, shape, fills["near"], fills["reflect"],
                       in_seam_coverage.zero_metal_axes(grid), codes,
                       fills["phase"])
    contended = {key: sorted(set(threads)) for key, threads in writes.items()
                 if len(set(threads)) > 1}
    assert not contended, (
        f"{label}: {len(contended)} words are written by more than one thread, "
        f"e.g. {list(contended.items())[:3]}")


@pytest.mark.parametrize("label,kwargs", FOLDS, ids=[row[0] for row in FOLDS])
def test_every_cell_of_every_written_array_is_written_exactly_once(label, kwargs,
                                                                   emitted):
    """The other direction, and it is what stops the race check being vacuous.

    A kernel that wrote nothing would have no contended word at all. Every one of
    the twelve arrays the signature binds writable must come out fully covered:
    the destination threads stop after ``fu``, so their ``B``, ``H`` and ``f_w_H``
    are owed by their source threads, and a carry that forgot one destination
    subset shows up here as a hole rather than as a wrong number.
    """
    _fields, grid, _pml = build(**kwargs)
    fills, codes = _launch_arguments(grid)
    shape = tuple(int(n) for n in grid.shape)
    cells = shape[0] * shape[1] * shape[2]
    writes = write_map(emitted, shape, fills["near"], fills["reflect"],
                       in_seam_coverage.zero_metal_axes(grid), codes,
                       fills["phase"])
    covered = {name: 0 for name in WRITTEN_ARRAYS}
    for array, _cell in writes:
        assert array in covered, f"{label}: the kernel writes {array}, which is " \
                                 f"not in WRITTEN_ARRAYS"
        covered[array] += 1
    holes = {name: cells - count for name, count in covered.items() if count != cells}
    assert not holes, f"{label}: arrays with unwritten cells {holes} of {cells}"


@pytest.mark.parametrize("label,kwargs", FOLDS, ids=[row[0] for row in FOLDS])
def test_the_carry_moves_exactly_the_cells_the_two_fills_write(label, kwargs,
                                                               emitted):
    """The POSITIVE direction: ownership moved for these cells and no others.

    A launch in which every thread wrote only its own cell would pass both the race
    check and the coverage check and carry nothing at all. The count that separates
    them is measured against ``stepping``'s own two fills rather than against the
    emitter: a boolean map of the cells the array path's fills write, per component,
    built by running them on a marked field.
    """
    fields, grid, _pml = build(**kwargs)
    fills, codes = _launch_arguments(grid)
    shape = tuple(int(n) for n in grid.shape)
    nx, ny, nz = shape

    # Which cells do the two fills actually write? MARKED rather than derived, and
    # the marker has to be per-cell distinct: a constant field is imaged onto
    # itself wherever the phase is +1, so a uniform sentinel would report the near
    # fill as writing nothing at all. zero_metal_B is deliberately NOT run -- it
    # writes cells the thread still owns.
    cells = nx * ny * nz
    marks = {}
    for target, name in enumerate(B_NAMES):
        marks[name] = (numpy.arange(cells, dtype=numpy.float32).reshape(shape)
                       + 1.0 + 10.0 * cells * target)
        getattr(fields, name)[...] = marks[name]
    stepping.fill_symmetry_bc_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    destinations = {name: numpy.asarray(getattr(fields, name)) != marks[name]
                    for name in B_NAMES}

    writes = write_map(emitted, shape, fills["near"], fills["reflect"],
                       in_seam_coverage.zero_metal_axes(grid), codes,
                       fills["phase"])
    for target, name in enumerate(B_NAMES):
        # SET comparison, not list: a walled cell is stored twice by the SAME
        # thread (the curl's store, then zero_metal_B's), which is a repeat and
        # not a move.
        moved = {cell for (array, cell), threads in writes.items()
                 if array == name and set(threads) != {cell}}
        expected = {int(cell) for cell in
                    numpy.flatnonzero(destinations[name].reshape(-1))}
        assert moved == expected, (
            f"{label}/{name}: the launch moves ownership of {len(moved)} cells and "
            f"the array path's two fills write {len(expected)}; symmetric "
            f"difference {sorted(moved ^ expected)[:6]}")
        # And every one of them is written by a thread that owns its own cell.
        for cell in moved:
            writer = writes[(name, cell)][0]
            assert set(writes[(name, writer)]) == {writer}, (
                f"{label}/{name}: cell {cell}'s writer {writer} does not own its "
                f"own cell, so it wrote a ghost from a register it never formed")
    _ = target, nx


def test_the_ghost_writer_is_never_itself_a_destination(emitted):
    """The defect the sibling track's device gate found, stated as a measurement.

    With two fills live a thread can be the SOURCE of one and the DESTINATION of
    the other. If a carry block were not nested inside the ownership guard, that
    thread would write a ghost from a register it never formed -- and the ghost it
    would write is already owned by the fully back-substituted thread, so the
    symptom is a contended word rather than a missing one.
    """
    _fields, grid, _pml = build(axes="XY", phases=(1, -1))
    fills, codes = _launch_arguments(grid)
    shape = tuple(int(n) for n in grid.shape)
    nx, ny, nz = shape
    assert fills["near"][0] and fills["reflect"][1] >= 0, fills
    writes = write_map(emitted, shape, fills["near"], fills["reflect"],
                       in_seam_coverage.zero_metal_axes(grid), codes,
                       fills["phase"])
    # Bx is near-imaged on x (i == 0) and far-imaged on y (j == ny - 1). The thread
    # at (i == 2, j == ny - 1) stands on a Bx DESTINATION while sitting on the near
    # fill's SOURCE row, so an unnested carry would have it write the corner ghost
    # from a b_x it never formed.
    trap = 2 * (ny * nz) + (ny - 1) * nz
    wrote = sorted(array for (array, _cell), threads in writes.items()
                   if trap in threads)
    assert "fu_Bx" in wrote, "the split-field auxiliary is owed at every cell"
    assert not {"Bx", "Hx", "f_w_Hx"} & set(wrote), (
        f"the thread at (2, ny-1, 0) wrote {sorted({'Bx', 'Hx', 'f_w_Hx'} & set(wrote))}; "
        f"it is a DESTINATION of the far y fill for Bx and owes no Bx word at all")
    # And the corner it would have written IS written -- by the fully
    # back-substituted thread, which is what makes this an ownership move rather
    # than a dropped ghost.
    corner = 0 * (ny * nz) + (ny - 1) * nz
    owner = 2 * (ny * nz) + int(fills["reflect"][1]) * nz
    assert set(writes[("Bx", corner)]) == {owner}, (
        f"Bx at (0, ny-1, 0) is written by {writes[('Bx', corner)]}, not by the "
        f"back-substituted thread at (2, reflect_y, 0) = {owner}")


def test_the_write_enumeration_would_catch_a_dropped_ownership_guard(emitted):
    """The enumeration's own null control -- a detector never shown to fire.

    The mutation is applied to the EMITTED TEXT, so it is the shipped kernel's own
    bytes with one guard removed rather than a hand-written strawman.
    """
    _fields, grid, _pml = build(axes="XY", phases=(1, -1))
    fills, codes = _launch_arguments(grid)
    shape = tuple(int(n) for n in grid.shape)
    broken = emitted.replace("    if (own_x) {\n", "    if (1) {\n", 1)
    assert broken != emitted, "the ownership guard anchor no longer matches"
    writes = write_map(broken, shape, fills["near"], fills["reflect"],
                       in_seam_coverage.zero_metal_axes(grid), codes,
                       fills["phase"])
    contended = [key for key, threads in writes.items() if len(set(threads)) > 1]
    assert contended, (
        "dropping own_x from the constitutive block left every word with one "
        "writer, so the race check cannot see an ownership failure at all")


# ---------------------------------------------------------------------------
# 3. THE DEVICE TEXT -- the terms that measured zero before this round
# ---------------------------------------------------------------------------

def test_the_kernel_now_carries_the_fill_and_reflect_terms(emitted):
    """The fill, the reflect row and the parity are IN the emitted kernel.

    Before this round the board measured ``fill_symmetry=0, fill_folded=0,
    reflect_row=0`` in this family's device text: the two passes were refused on
    the host and nothing about them reached the kernel. These are the terms that
    were zero.
    """
    assert emitted.count("near_x") >= 2 and emitted.count("near_y") >= 2 \
        and emitted.count("near_z") >= 2
    for axis in ("x", "y", "z"):
        assert f"reflect_{axis} >= 0" in emitted, axis
        assert f"phase_{axis}" in emitted, axis
    assert emitted.count("int own_") == 3
    # Seven ghosts per component, three components: 21 carried destinations, each
    # with one B store and one constitutive statement.
    assert emitted.count("_i = idx ") == 21, emitted.count("_i = idx ")
    assert emitted.count("constitutive_apply(") == 1 + 21
    assert emitted.count("constitutive_apply_pre(") == 1 + 3


def test_every_carry_block_sits_inside_its_components_ownership_guard(emitted):
    """Structural, and it is the property :func:`fill_carry_blocks` depends on.

    Read off the SAME scanner the race check uses, so the two cannot disagree about
    which guards are in scope where.
    """
    ghosts = [(payload[0], guards)
              for guards, kind, payload, spelling in program(emitted)
              if kind == "let" and payload[0].startswith("g")
              and payload[0].endswith("_i")]
    assert len(ghosts) == 21, [name for name, _ in ghosts]
    for name, guards in ghosts:
        component = name[1]                                  # g<x|y|z>_..._i
        outermost = guards[0].co_consts if hasattr(guards[0], "co_consts") else guards[0]
        assert f"own_{component}" in guards[0].co_names, (
            f"the outermost guard over {name} does not name own_{component} "
            f"(names: {guards[0].co_names}, consts: {outermost})")


def test_the_near_ghost_takes_the_destinations_coefficient_pair(emitted):
    """The one asymmetry between the near and far carries, in the emitted text.

    ``update_H`` indexes component ``m`` on axis ``m``
    (``stepping.H_CONSTITUTIVE_TERMS``:226) and the NEAR fill images along that
    same axis, so the near ghost's coefficient pair is the DESTINATION's at stored
    index 0, not this thread's at stored 2. A FAR ghost does not move that axis
    and takes the source thread's pair unchanged -- reloading there would reload
    the identical word. Getting this backwards is the absorber profile of the
    wrong cell: smooth, converged and wrong inside the PML, invisible outside it.
    """
    body = emitted.split("\n) {\n")[-1]
    near_ghosts = far_ghosts = 0
    for line in body.splitlines():
        if "constitutive_apply(" not in line or "_i, g" not in line:
            continue
        tag = line.split("constitutive_apply(")[1].split(",")[0][-1]
        if line.rstrip().endswith(f"kps_{tag}[0], kms_int_{tag}[0]);"):
            assert "n_i, g" in line, f"a pair at index 0 on a FAR ghost: {line!r}"
            near_ghosts += 1
        else:
            coordinate = ("i", "j", "k")["xyz".index(tag)]
            assert line.rstrip().endswith(
                f"kps_{tag}[{coordinate}], kms_int_{tag}[{coordinate}]);"), line
            assert "n_i, g" not in line, f"the source's pair on a NEAR ghost: {line!r}"
            far_ghosts += 1
    # Per component: four of the seven destinations carry the near image.
    assert (near_ghosts, far_ghosts) == (12, 9), (near_ghosts, far_ghosts)


def test_the_split_field_auxiliary_is_written_at_every_cell(emitted):
    """``fu`` stays ABOVE the ownership guard, and the array path is why.

    ``step_B`` writes ``fu`` at every cell and neither fill touches it
    (stepping.py:1451 writes ``field``, never ``fu_field``). Masking it would damp
    the recurrence on every destination plane -- a smooth, converged, wrong
    absorber, one timestep later.
    """
    prelude = emitted.split("extern \"C\"")[0]
    guard = prelude.index("if (!owned) return 0.0f;")
    assert prelude.index("fu[idx] = fu_new;") < guard, (
        "the fu store moved below the ownership guard; a destination thread would "
        "leave the split-field auxiliary un-stepped")
    assert prelude.index("f[idx] = value;") > guard


def test_replaces_is_the_five_passes_the_driver_runs_in_this_seam():
    """Declared, in driver order, and matching the passes ``stepping`` exposes."""
    assert family.REPLACES == ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                               "fill_folded_far_ghosts_B", "update_H")
    for name in family.REPLACES:
        assert callable(getattr(stepping, name)), name


# ---------------------------------------------------------------------------
# 4. THE PREDICATE -- what the carry admits, and what it still refuses BY NAME
# ---------------------------------------------------------------------------

@pytest.fixture
def cupy_named_numpy():
    return _NumpyWearingCupysName()


def covered(fields, pml, grid):
    return family.covers_fused_magnetic_pair(fields, pml, grid, ())


@pytest.mark.parametrize("label,kwargs", FOLDS, ids=[row[0] for row in FOLDS])
def test_a_folded_grid_is_admitted_now(label, kwargs, cupy_named_numpy):
    """The whole point of the round: the fold is no longer a refusal.

    Built on the CuPy-named stand-in because every CUDA predicate's first question
    is whether the array module is CuPy at all, and that is the one thing about the
    device library a laptop cannot supply.
    """
    size = [1.6, 1.6, 1.2]
    for name in kwargs.get("axes", ""):
        size["XYZ".index(name)] = kwargs.get("extent", 2.0)
    grid = Grid(resolution=10.0, cell_size=tuple(size), dimensions=3, courant=0.35,
                symmetry=tuple(Mirror(name, int(phase)) for name, phase
                               in zip(kwargs.get("axes", ""),
                                      kwargs.get("phases", ()))),
                boundaries=kwargs.get("boundaries"), xp=cupy_named_numpy)
    folded = {"XYZ".index(name) for name in kwargs.get("axes", "")}
    pml = PML(grid=grid, thickness=tuple(
        (0, 0) if grid.shape[axis] < 6 else (0, 2) if axis in folded else (2, 2)
        for axis in range(3)))
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    ok, reason = covered(fields, pml, grid)
    assert ok, f"{label}: {reason}"


def test_a_grid_that_cannot_answer_is_mirrored_is_refused_by_name(cupy_named_numpy):
    grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=cupy_named_numpy)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=2)

    class _Blind:
        def __getattr__(self, item):
            if item == "is_mirrored":
                raise AttributeError(item)
            return getattr(grid, item)

    ok, reason = family.covers_fused_magnetic_pair(fields, pml, _Blind(), ())
    assert not ok and "is_mirrored" in reason, reason


def test_an_axis_reported_both_folded_and_walled_is_refused_by_name(
        cupy_named_numpy, monkeypatch):
    """The clause the NEAR carry rests on, and it is unreachable from a real Grid.

    ``zero_metal_axes`` is ``is_metallic and not is_mirrored`` (stepping.py:2284-2286),
    so the emitted kernel writes no wall line at the cell the near fill images -- and
    a grid that reported both would silently lose that clear. The state is reachable
    only by making ``zero_metal_axes`` itself answer differently, which is what is
    armed here: a fail-closed clause nothing can reach is indistinguishable from one
    that returns True, and this is the difference.
    """
    grid = Grid(resolution=4.0, cell_size=(2.0, 2.4, 1.6), dimensions=3,
                courant=0.35, symmetry=(Mirror("X", 1),), xp=cupy_named_numpy)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((0, 2), (2, 2), (2, 2)))
    assert covered(fields, pml, grid)[0], covered(fields, pml, grid)[1]
    assert grid.is_mirrored(0)

    monkeypatch.setattr(family, "zero_metal_axes", lambda _grid: (True, False, False))
    ok, reason = covered(fields, pml, grid)
    assert not ok and "both folded and walled" in reason, reason


def test_a_folded_axis_too_short_for_the_near_source_row_is_refused_by_name(
        cupy_named_numpy):
    """``stepping._mirror_source`` raises there; this kernel would index outside."""
    grid = Grid(resolution=10.0, cell_size=(2.0, 1.6, 1.2), dimensions=3,
                courant=0.35, symmetry=(Mirror("X", 1),), xp=cupy_named_numpy)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((0, 2), (2, 2), (0, 0)))

    class _TooShort:
        def __getattr__(self, item):
            return getattr(grid, item)

        def stored_cells(self, axis):
            return 2 if axis == 0 else grid.stored_cells(axis)

    ok, reason = family.covers_fused_magnetic_pair(fields, pml, _TooShort(), ())
    assert not ok and ("stored_cells" in reason or "does not exist" in reason), reason


def test_the_fill_predicates_own_refusals_reach_this_seam(cupy_named_numpy):
    """Delegation, measured: a clause in ``in_seam_coverage`` refuses here too.

    A mirrored axis whose declared phase is not +/-1 is refused by
    ``covers_fill_symmetry``; this seam must not have its own weaker copy of that
    question.
    """
    grid = Grid(resolution=10.0, cell_size=(2.0, 1.6, 1.2), dimensions=3,
                courant=0.35, symmetry=(Mirror("X", 1),), xp=cupy_named_numpy)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((0, 2), (2, 2), (0, 0)))

    class _OddPhase:
        def __getattr__(self, item):
            return getattr(grid, item)

        def mirror_phase(self, axis):
            return 0 if axis == 0 else grid.mirror_phase(axis)

    ok, reason = family.covers_fused_magnetic_pair(fields, pml, _OddPhase(), ())
    assert not ok and "fill_symmetry_bc_B" in reason and "mirror phase" in reason, \
        reason


def test_the_fill_plan_refuses_a_grid_the_predicate_would_have(cupy_named_numpy,
                                                               monkeypatch):
    """The launcher does not build a plan the predicate would have refused.

    A gate hands ``launch_fused_magnetic_pair`` its own arguments, so the plan
    builder is reachable WITHOUT the predicate and must refuse on its own. Both
    facts the carry rests on are asserted twice for that reason.
    """
    grid = Grid(resolution=4.0, cell_size=(2.0, 2.4, 1.6), dimensions=3,
                courant=0.35, symmetry=(Mirror("X", 1),), xp=cupy_named_numpy)
    assert family.fused_magnetic_pair_fills(grid)["near"] == (1, 0, 0)

    monkeypatch.setattr(family, "zero_metal_axes", lambda _grid: (True, False, False))
    with pytest.raises(ValueError, match="both folded and walled"):
        family.fused_magnetic_pair_fills(grid)

    monkeypatch.setattr(family, "zero_metal_axes", lambda _grid: (False, False, False))
    monkeypatch.setattr(family, "mirror_fill_phases", lambda _grid: (0, None, None))
    with pytest.raises(ValueError, match="mirror phase"):
        family.fused_magnetic_pair_fills(grid)

    monkeypatch.setattr(family, "mirror_fill_phases", lambda _grid: (None, None, None))
    monkeypatch.setattr(family, "folded_far_rows", lambda _grid: (4, None, None))
    with pytest.raises(ValueError, match="far reflect row"):
        family.fused_magnetic_pair_fills(grid)


def test_the_fill_plan_is_the_same_reading_in_seam_coverage_gives(cupy_named_numpy):
    """One source of truth for which axes each fill visits."""
    for _label, kwargs in FOLDS:
        _fields, grid, _pml = build(**kwargs)
        fills = family.fused_magnetic_pair_fills(grid)
        phases = in_seam_coverage.mirror_fill_phases(grid)
        rows = in_seam_coverage.folded_far_rows(grid)
        assert fills["near"] == tuple(int(phases[a] is not None) for a in range(3))
        assert fills["reflect"] == tuple(
            -1 if rows[a] is None else int(rows[a]) for a in range(3))
        for axis in range(3):
            near_plan = in_seam_coverage.plan("fill_symmetry", grid)
            far_plan = in_seam_coverage.plan("fill_folded_far", grid)
            assert bool(fills["near"][axis]) == any(
                entry["axis"] == axis for entry in near_plan)
            assert (fills["reflect"][axis] >= 0) == any(
                entry["axis"] == axis for entry in far_plan)


def test_carried_destinations_is_the_powerset_minus_the_threads_own_cell():
    """Seven per component when all three axes fold, and never the own cell."""
    assert family.carried_destinations((), ()) == ()
    assert family.carried_destinations((0,), ()) == (((), True),)
    assert len(family.carried_destinations((0,), (1, 2))) == 7
    assert len(family.carried_destinations((), (1, 2))) == 3
    for subset, near in family.carried_destinations((0,), (1, 2)):
        assert subset or near, "the empty destination is the thread's own cell"
    seen = set(family.carried_destinations((0,), (1, 2)))
    assert len(seen) == 7, "a destination is emitted twice"


def test_the_near_and_far_axis_sets_are_complementary():
    """Read off ``IYEE_SHIFTS``, and the D family's answer is the exact inverse."""
    for target, name in enumerate(B_NAMES):
        near = family.near_fill_axes(target)
        far = family.far_fill_axes(target)
        assert near == (target,), (name, near)
        assert set(near) | set(far) == {0, 1, 2}
        assert not set(near) & set(far)
        assert in_seam_coverage.IYEE_SHIFTS[name][target] == 0
