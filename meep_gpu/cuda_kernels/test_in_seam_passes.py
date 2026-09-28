"""The laptop merge bar for the CUDA in-seam passes.

WHAT THIS FILE CAN AND CANNOT SETTLE. It never launches a kernel -- there is no
CuPy on the machine that runs the merge bar -- so it settles nothing about
bit-identity. What it settles is everything a device gate would have to ASSUME:

* that ``in_seam_coverage``'s restatements of ``stepping``'s derivations are the
  same function, on every grid this file builds. Those restatements exist so the
  predicate stays engine-import-free, and a restatement that has drifted is a
  silent wrong answer on one plane -- never a crash;
* that the parity closed form the kernels rely on IS ``fields.mirror_parity``,
  over all 72 component/axis/phase combinations rather than on the handful the
  fixture happens to draw;
* that each pass's component set is the set ``stepping``'s own loop would touch,
  INCLUDING the B/D inversion, which is where an argument by analogy is plausible
  and wrong;
* that the device strings are pure ASCII and encode under the C locale, which is
  a compile requirement rather than a style rule;
* that the device text still says what the transcription claims -- a multiply and
  not a copy, ``2 * stride`` and not ``1``, ``-phase`` on the far fill;
* that every shipped kernel is in exactly one of ``CERTIFIED_KERNELS`` and
  ``UNCERTIFIED_KERNELS``, so a kernel cannot ship unmeasured, and that a name in
  ``CERTIFIED_KERNELS`` has a record block behind it.

Everything is read off the SOURCE TEXT through the syntax tree, never by
importing the kernel module: importing it needs CuPy, which is the whole reason
these helpers exist.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re
import types

import numpy
import pytest

from .. import stepping
from ..device_identity import weld_survives_edit
from ..fields import Fields, IYEE_SHIFTS, mirror_parity
from ..grid import Grid, Mirror
from . import in_seam_coverage

HERE = pathlib.Path(__file__).parent
KERNEL_MODULE = HERE / "in_seam_passes.py"
RECORD = HERE / "certification.json"

KERNEL_DECLARATION = re.compile(r'extern "C" __global__ void (\w+)\(')


def kernel_source() -> str:
    return KERNEL_MODULE.read_text(encoding="utf-8")


def module_level_literal(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned at module level in "
                         f"{KERNEL_MODULE.name}")


def device_sources(source: str) -> dict:
    """The exact CUDA strings ``_get_kernel`` hands ``cp.RawKernel``.

    Evaluated from the syntax tree rather than by importing, because importing
    needs CuPy. The ``_PRELUDE + r'''...'''`` concatenations are evaluated in
    order against the constants already seen, so each kernel string carries its
    shared prelude exactly as it does at compile time.
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
    return {name: text for name, text in environment.items()
            if name.endswith("_code")}


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- the one thing a laptop cannot supply."""

    def __init__(self):
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


# ---------------------------------------------------------------------------
# The grid battery
# ---------------------------------------------------------------------------
#
# EVERY AXIS IS FOLDED SOMEWHERE, EVERY AXIS IS WALLED SOMEWHERE, both fold
# terminations appear, both plane phases appear, and both full-count parities
# appear on a folded PERIODIC axis. A battery carrying one of each proves nothing
# about the other.

GRID_SPECS = (
    ("wall_XYZ", "", 1, ("metallic", "metallic", "metallic"), (9.0, 10.0, 11.0)),
    ("wall_X_only", "", 1, ("metallic", "periodic", "periodic"), (9.0, 10.0, 11.0)),
    ("no_wall_no_fold", "", 1, ("periodic", "periodic", "periodic"), (9.0, 10.0, 11.0)),
    ("fold_X_periodic", "X", 1, ("periodic", "periodic", "periodic"), (16.0, 10.0, 11.0)),
    ("fold_X_metallic", "X", 1, ("metallic", "periodic", "periodic"), (16.0, 10.0, 11.0)),
    ("fold_Y_periodic_odd_plane", "Y", -1, ("periodic", "periodic", "periodic"),
     (9.0, 16.0, 11.0)),
    ("fold_Y_metallic", "Y", 1, ("periodic", "metallic", "periodic"), (9.0, 16.0, 11.0)),
    ("fold_Z_periodic", "Z", -1, ("periodic", "periodic", "periodic"), (9.0, 10.0, 16.0)),
    ("fold_X_periodic_odd_count", "X", 1, ("periodic", "periodic", "periodic"),
     (17.0, 10.0, 11.0)),
    ("fold_Y_periodic_odd_count", "Y", -1, ("periodic", "periodic", "periodic"),
     (9.0, 17.0, 11.0)),
    ("fold_Z_periodic_odd_count", "Z", 1, ("periodic", "periodic", "periodic"),
     (9.0, 10.0, 17.0)),
    ("fold_X_wall_Y", "X", 1, ("periodic", "metallic", "periodic"), (16.0, 10.0, 11.0)),
    ("fold_X_metallic_wall_Y", "X", -1, ("metallic", "metallic", "periodic"),
     (16.0, 10.0, 11.0)),
    ("fold_XY_mixed", "XY", -1, ("periodic", "metallic", "periodic"), (16.0, 18.0, 11.0)),
    ("fold_XYZ_periodic", "XYZ", 1, ("periodic", "periodic", "periodic"),
     (16.0, 18.0, 20.0)),
    ("fold_XYZ_periodic_odd", "XYZ", -1, ("periodic", "periodic", "periodic"),
     (17.0, 19.0, 21.0)),
    ("fold_XYZ_metallic", "XYZ", 1, ("metallic", "metallic", "metallic"),
     (16.0, 18.0, 20.0)),
    ("fold_XYZ_mixed", "XYZ", -1, ("metallic", "periodic", "metallic"),
     (16.0, 18.0, 20.0)),
)


def build_grid(spec):
    _label, axes, phase, boundaries, cell = spec
    planes = tuple(Mirror(name, phase) for name in axes)
    return Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=planes, xp=numpy, courant=0.5)


@pytest.fixture(params=GRID_SPECS, ids=[spec[0] for spec in GRID_SPECS])
def grid(request):
    return build_grid(request.param)


# ---------------------------------------------------------------------------
# The transcriptions
# ---------------------------------------------------------------------------

def test_the_restated_yee_shifts_are_the_engines():
    """A second copy of ``IYEE_SHIFTS`` is the whole component selection.

    ``in_seam_coverage`` restates the six triples so it can stay
    engine-import-free. Every component set in every one of the three passes is
    derived from that table, so a drifted digit is a wrong plane on a wrong array
    -- silent, and on one component only.
    """
    for name, shifts in in_seam_coverage.IYEE_SHIFTS.items():
        assert shifts == IYEE_SHIFTS[name], name
    assert set(in_seam_coverage.IYEE_SHIFTS) == {
        "Bx", "By", "Bz", "Dx", "Dy", "Dz"}


def test_the_restated_mirror_source_index_is_the_engines():
    assert in_seam_coverage.MIRROR_SOURCE_INDEX == stepping.MIRROR_SOURCE_INDEX


def test_the_family_components_are_the_names_stepping_uses():
    """The wipe's component tuple and the fills' curl-term targets name one set.

    ``stepping._zero_metal`` is handed ``B_COMPONENTS``/``D_COMPONENTS``; the two
    fills walk ``B_CURL_TERMS``/``D_CURL_TERMS`` and take ``term.target``. That
    they are the same six arrays IN THE SAME ORDER is what lets one table serve
    all three passes, and it is checked rather than assumed.
    """
    assert in_seam_coverage.FAMILY_COMPONENTS["B"] == tuple(stepping.B_COMPONENTS)
    assert in_seam_coverage.FAMILY_COMPONENTS["D"] == tuple(stepping.D_COMPONENTS)
    assert in_seam_coverage.FAMILY_COMPONENTS["B"] == tuple(
        term.target for term in stepping.B_CURL_TERMS)
    assert in_seam_coverage.FAMILY_COMPONENTS["D"] == tuple(
        term.target for term in stepping.D_CURL_TERMS)


def test_the_parity_closed_form_is_mirror_parity_on_every_combination():
    """``ph * (1 - 2*iyee[c][a])`` over all 12 x 3 x 2, not over a sample.

    The kernels carry ONE signed float per launch because of this identity: the
    near fill only touches shift-0 components (parity ``+phase``) and the far
    fill only shift-1 ones (``-phase``). If it fails anywhere the single scalar is
    the wrong instrument, whatever the fixture happens to draw.
    """
    checked = 0
    for component, shifts in IYEE_SHIFTS.items():
        for axis in range(3):
            for phase in (1, -1):
                checked += 1
                assert mirror_parity(component, axis, phase) == \
                    phase * (1 - 2 * shifts[axis]), (component, axis, phase)
    assert checked == 72


def test_the_component_sets_are_the_ones_steppings_loops_touch(grid):
    """``shifts[axis] == 0`` / ``!= 1``, read back off the engine's own tables."""
    for family, terms in (("B", stepping.B_CURL_TERMS), ("D", stepping.D_CURL_TERMS)):
        for axis in range(3):
            near = {index for index, term in enumerate(terms)
                    if term.iyee[axis] == 0}
            far = {index for index, term in enumerate(terms)
                   if term.iyee[axis] == 1}
            assert set(in_seam_coverage.shift_zero_components(family, axis)) == near
            assert set(in_seam_coverage.shift_one_components(family, axis)) == far


def test_the_two_families_invert_each_other_on_every_axis():
    """The near fill reaches ONE B component and TWO D ones; the far fill flips it.

    Stated as its own test because it is the fact a port by analogy gets wrong,
    and because the fused track measured exactly this inversion this week. A D
    kernel written as "the B kernel with the letters changed" would put one array
    where two belong on both fills at once.
    """
    for axis in range(3):
        assert len(in_seam_coverage.shift_zero_components("B", axis)) == 1
        assert len(in_seam_coverage.shift_zero_components("D", axis)) == 2
        assert len(in_seam_coverage.shift_one_components("B", axis)) == 2
        assert len(in_seam_coverage.shift_one_components("D", axis)) == 1
        # The single B component on the near fill is the one whose OWN axis this
        # is; the single D component on the far fill is likewise.
        assert in_seam_coverage.shift_zero_components("B", axis) == (axis,)
        assert in_seam_coverage.shift_one_components("D", axis) == (axis,)


# ---------------------------------------------------------------------------
# The derivations, against stepping's own
# ---------------------------------------------------------------------------

def test_zero_metal_axes_is_steppings_walled_set(grid):
    """The predicate's walled set is ``_zero_metal``'s, folded-metallic exclusion included."""
    expected = tuple(
        bool(grid.has_metallic) and bool(grid.is_metallic(axis))
        and not bool(grid.is_mirrored(axis))
        for axis in range(3))
    assert in_seam_coverage.zero_metal_axes(grid) == expected


def test_zero_metal_never_walls_a_folded_metallic_axis(grid):
    """The 1.28e+00-relative-L2 clause, checked as a property of the derivation."""
    walls = in_seam_coverage.zero_metal_axes(grid)
    for axis in range(3):
        if grid.is_mirrored(axis):
            assert not walls[axis], (
                f"axis {axis} is folded; its stored cell 0 is the fold's "
                f"parity-weighted ghost, not a wall")


def test_mirror_fill_phases_agree_with_stepping(grid):
    engine = stepping._mirror_phases(grid)
    ours = in_seam_coverage.mirror_fill_phases(grid)
    for axis in range(3):
        if not grid.has_symmetry() or not grid.is_mirrored(axis):
            assert ours[axis] is None
        else:
            assert ours[axis] == engine[axis]


def test_stored_past_owned_agrees_with_stepping(grid):
    ours = in_seam_coverage.stored_past_owned(grid)
    for axis in range(3):
        assert bool(ours[axis]) == bool(stepping._stored_past_owned(grid, axis))


def test_far_reflect_rows_agree_with_stepping_on_the_axes_the_pass_visits(grid):
    """Ours is the engine's, MASKED BY THE VISIT LIST.

    ``_far_reflect_rows`` reports a row on every folded non-metallic axis;
    ``_fill_folded_far_ghosts`` only iterates ``_stored_past_owned`` axes. The two
    coincide on every grid this battery builds, and this test is what would say
    so if they ever stopped.
    """
    engine = stepping._far_reflect_rows(grid)
    ours = in_seam_coverage.folded_far_rows(grid)
    for axis in range(3):
        if stepping._stored_past_owned(grid, axis):
            assert ours[axis] == engine[axis]
        else:
            assert ours[axis] is None


def test_the_image_row_is_stored_minus_two_at_even_and_minus_three_at_odd(grid):
    """The fact a baked ``n - 2`` hides behind at one count parity.

    MEASURED here rather than quoted: the row is ``stored - 2`` when the FULL
    count is even and ``stored - 3`` when it is odd, which is why the gate carries
    both parities and plants ``reflect_row_n_minus_two``.
    """
    rows = in_seam_coverage.folded_far_rows(grid)
    for axis in range(3):
        if rows[axis] is None:
            continue
        stored = int(grid.stored_cells(axis))
        full = int(grid.shape_full[axis])
        expected = stored - 2 if full % 2 == 0 else stored - 3
        assert rows[axis] == expected, (axis, full, stored, rows[axis])


def test_the_image_row_is_in_bounds_and_below_the_slot_it_feeds(grid):
    """No launch may read the plane it is about to write."""
    rows = in_seam_coverage.folded_far_rows(grid)
    for axis in range(3):
        if rows[axis] is None:
            continue
        stored = int(grid.stored_cells(axis))
        assert 0 <= rows[axis] <= stored - 2


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_fill_plans_walk_the_axes_in_x_y_z_order(grid):
    """The corner of a doubly unowned component carries the product of both parities.

    ``_fill_symmetry_ghost_cells`` walks ``for term ... for axis in range(3)``, so
    the second axis's fill reads a plane the first already wrote. Only an ordered
    walk reproduces that, which is why the plan is a list of launches rather than
    one launch with three flags.
    """
    for pass_name in ("fill_symmetry", "fill_folded_far"):
        axes = [entry["axis"] for entry in in_seam_coverage.plan(pass_name, grid)]
        assert axes == sorted(axes)


def test_the_wipe_plan_is_one_launch_and_the_fills_are_one_per_axis(grid):
    wipe = in_seam_coverage.plan("zero_metal", grid)
    assert len(wipe) <= 1
    if wipe:
        assert tuple(wipe[0]["axes"]) == tuple(
            int(bool(w)) for w in in_seam_coverage.zero_metal_axes(grid))
    near = in_seam_coverage.plan("fill_symmetry", grid)
    assert len(near) == sum(1 for a in range(3)
                            if in_seam_coverage.mirror_fill_phases(grid)[a] is not None)
    far = in_seam_coverage.plan("fill_folded_far", grid)
    assert len(far) == sum(1 for a in range(3)
                           if stepping._stored_past_owned(grid, a))


def test_every_planned_phase_is_plus_or_minus_one(grid):
    for pass_name in ("fill_symmetry", "fill_folded_far"):
        for entry in in_seam_coverage.plan(pass_name, grid):
            assert entry["phase"] in (1, -1)


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def _fields_on(grid):
    return Fields(grid=grid)


def test_the_predicate_admits_a_real_float32_cupy_grid(grid):
    """With CuPy's name in place, every grid in this battery is covered."""
    grid.xp = _NumpyWearingCupysName()
    fields = _fields_on(grid)
    for pass_name in in_seam_coverage.PASSES:
        for family in in_seam_coverage.FAMILIES:
            covered, reason = in_seam_coverage.covers(pass_name, fields, grid, family)
            assert covered, f"{pass_name}:{family}: {reason}"


def test_the_predicate_refuses_a_numpy_backend(grid):
    fields = _fields_on(grid)
    for pass_name in in_seam_coverage.PASSES:
        covered, reason = in_seam_coverage.covers(pass_name, fields, grid, "B")
        assert not covered
        assert "not cupy" in reason


def test_the_predicate_refuses_complex_storage(grid):
    """A Bloch run stores complex64; these kernels index float32.

    The largest single gap in this slice, refused BY NAME rather than discovered
    by a launch that read half a complex word as a float.
    """
    grid.xp = _NumpyWearingCupysName()
    fields = _fields_on(grid)
    fields.Bx = numpy.zeros(grid.shape, dtype=numpy.complex64)
    covered, reason = in_seam_coverage.covers("zero_metal", fields, grid, "B")
    assert not covered
    assert "complex64" in reason


def test_the_predicate_refuses_the_cylindrical_radial_axis(grid):
    """``_mirror_phases`` puts ``(-1)**m`` in the mirror-phase slot for r."""
    grid.xp = _NumpyWearingCupysName()
    fields = _fields_on(grid)
    real_is_axis = grid.is_axis
    grid.is_axis = lambda axis: axis == 0
    try:
        for pass_name in in_seam_coverage.PASSES:
            covered, reason = in_seam_coverage.covers(pass_name, fields, grid, "B")
            assert not covered
            assert "cylindrical" in reason
    finally:
        grid.is_axis = real_is_axis


def test_the_predicate_refuses_a_fold_whose_two_termination_routes_disagree():
    """``is_metallic`` and ``stored_cells > owned_cells`` are two readings of one fact."""
    grid = build_grid(("fold_X_periodic", "X", 1,
                       ("periodic", "periodic", "periodic"), (16.0, 10.0, 11.0)))
    grid.xp = _NumpyWearingCupysName()
    fields = _fields_on(grid)
    covered, _reason = in_seam_coverage.covers("fill_folded_far", fields, grid, "B")
    assert covered
    real_owned = grid.owned_cells
    grid.owned_cells = lambda axis: int(grid.stored_cells(axis))
    try:
        covered, reason = in_seam_coverage.covers("fill_folded_far", fields, grid, "B")
        assert not covered
        assert "disagree" in reason
    finally:
        grid.owned_cells = real_owned


def test_the_predicate_refuses_an_unknown_family_and_pass():
    grid = build_grid(GRID_SPECS[0])
    fields = _fields_on(grid)
    covered, reason = in_seam_coverage.covers("zero_metal", fields, grid, "H")
    assert not covered and "family" in reason
    with pytest.raises(ValueError):
        in_seam_coverage.covers("update_E", fields, grid, "B")


# ---------------------------------------------------------------------------
# The device text
# ---------------------------------------------------------------------------

def test_the_device_strings_are_pure_ascii_and_encode_under_the_c_locale():
    """A compile requirement, not a style rule.

    ``cupy.cuda.compiler.compile_using_nvrtc`` writes the source with a bare
    ``open(..., 'w')``, so the bytes go through the interpreter's LOCALE
    encoding -- ASCII under C/POSIX, which is what a non-interactive shell on the
    validation host gets. Two em-dashes in a comment once killed a sibling kernel
    at its first launch with ``UnicodeEncodeError``.
    """
    for name, text in device_sources(kernel_source()).items():
        assert text.isascii(), name
        text.encode("ascii")  # the encode itself, not only the predicate


def test_every_declared_kernel_is_named_in_the_module_and_partitioned():
    """Shipped, named, and in exactly one of the two verdict sets."""
    source = kernel_source()
    declared = set()
    for text in device_sources(source).values():
        declared.update(KERNEL_DECLARATION.findall(text))
    named = set(module_level_literal(source, "KERNEL_NAMES"))
    certified = set(module_level_literal(source, "CERTIFIED_KERNELS"))
    uncertified = set(module_level_literal(source, "UNCERTIFIED_KERNELS"))
    assert declared == named, (declared, named)
    assert certified & uncertified == set()
    assert certified | uncertified == declared


def in_seam_blocks() -> list:
    """Every record block whose subject is this module. Enumerated, not named."""
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    return [block for block in record.values()
            if isinstance(block, dict)
            and block.get("kernel_module") == "meep_gpu/cuda_kernels/in_seam_passes.py"]


def test_a_certified_kernel_has_a_record_block_behind_it():
    """A name in ``CERTIFIED_KERNELS`` with no record is the failure this catches."""
    certified = set(module_level_literal(kernel_source(), "CERTIFIED_KERNELS"))
    recorded = set()
    for block in in_seam_blocks():
        recorded.update(block.get("certified_kernels", []))
    assert certified <= recorded, sorted(certified - recorded)


def test_the_record_pins_the_DEVICE_BYTES_that_ship_today():
    """THE load-bearing weld, and the one a laptop can check.

    A record that binds the MODULE digest binds Python bookkeeping as much as it
    binds arithmetic; a record that binds each DEVICE STRING binds exactly what
    NVRTC compiled. Recomputed here off the syntax tree, so a comment moved inside
    a kernel body -- which changes the compiled text -- fails at the merge bar
    rather than at the next device run.
    """
    live = {name.replace("_kernel_code", "").lstrip("_"):
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in device_sources(kernel_source()).items()}
    blocks = in_seam_blocks()
    assert blocks, "no record block names in_seam_passes.py as its subject"
    for block in blocks:
        recorded = block.get("device_source_sha256")
        assert recorded, "the block records no per-kernel device digest"
        assert recorded == live, (
            "the device text moved since the gate ran. Re-run "
            "parity/meep_gpu/gate_cuda_in_seam_passes.py on a CUDA host and re-cut "
            "the block; a post_certification_edits entry cannot cover a change to "
            "the bytes NVRTC compiles.\n"
            f"  recorded: {json.dumps(recorded, indent=2, sort_keys=True)}\n"
            f"  live:     {json.dumps(live, indent=2, sort_keys=True)}")


def test_a_module_that_has_drifted_from_the_gated_bytes_declares_the_edit():
    """The module may move after the gate, and then it must SAY it moved.

    ``subject_sha256`` is the module as the gate ran it; ``revision_sha256`` is the
    module as it ships. Where they differ there must be a
    ``post_certification_edits`` entry, and its ``touches_device_code`` claim is
    not taken on trust -- the device-digest weld above is what makes it checkable.
    """
    live = hashlib.sha256(KERNEL_MODULE.read_bytes()).hexdigest()
    for block in in_seam_blocks():
        # ONE HOME FOR THE RULE (device_identity.py:209). A drift the shared rule
        # can prove harmless -- the device source NVRTC compiles is untouched --
        # needs neither a fresh revision digest nor an edit entry, because
        # nothing a reader could act on has moved. The helper returns False for
        # everything it cannot establish, and this block records its device text
        # under ``device_source_sha256`` rather than the ``device_sha256`` shape
        # the helper reads, so today it always falls through to the byte rule.
        if block["revision_sha256"] != live and not weld_survives_edit(
                KERNEL_MODULE, block, block["kernel_module"]):
            assert block["revision_sha256"] == live, (
                "the shipped module is neither the gated bytes nor the declared "
                "revision, and the change reaches executable code; update "
                "revision_sha256 and add a post_certification_edits entry saying "
                "what moved")
        if block["subject_sha256"] != block["revision_sha256"]:
            edits = block.get("post_certification_edits")
            assert edits, ("the module moved after the gate and the block declares "
                           "no edit")
            for edit in edits:
                assert set(edit) >= {"what", "touches_device_code",
                                     "how_that_was_established"}, edit


def test_the_record_names_an_artifact_tree_that_exists_and_reports_a_release():
    """A block whose artifacts are gone, or report a refusal, is not evidence."""
    for block in in_seam_blocks():
        # HERE is meep_gpu/cuda_kernels; the record's paths are relative
        # to the repository root, which is two levels up.
        directory = HERE.resolve().parents[1] / block["artifacts"]
        assert directory.is_dir(), directory
        legs = sorted(directory.rglob("gate.json"))
        assert len(legs) == 6, [str(p) for p in legs]  # 3 passes x 2 policies
        for leg in legs:
            summary = json.loads(leg.read_text(encoding="utf-8"))["summary"]
            assert summary["released"], (leg, summary["reasons"])
            assert summary["scored_cases"] == summary["single_launch_identical"]
            assert summary["multi_step_cases"] == summary["multi_step_identical"]


def test_the_record_does_not_claim_a_fusion():
    """The one over-claim this tranche is positioned to invite.

    Porting the in-seam passes is the PREREQUISITE for a fused CUDA pair and is
    not a fused pair. A record that let the two blur would be read, later, as
    evidence for a product nobody measured.
    """
    for block in in_seam_blocks():
        disclaimers = " ".join(block["_what_it_does_not_claim"]).lower()
        assert "no fusion" in disclaimers
        assert "no throughput" in disclaimers
        assert "no dispatch" in disclaimers
        for pass_name in ("zero_metal", "fill_symmetry", "fill_folded_far"):
            assert pass_name in block["per_pass"]


def test_the_parity_is_a_multiply_and_never_a_copy():
    """The ``phase == +1`` case is NOT special-cased into an assignment.

    Under ``"flush"`` CuPy compiles with ``-ftz=true``, so the array path's
    ``1 * field[...]`` FLUSHES a subnormal operand; a copy would preserve it and
    diverge on exactly the values the policy exists to decide.
    """
    sources = device_sources(kernel_source())
    for name in ("_fill_symmetry_B_kernel_code", "_fill_symmetry_D_kernel_code"):
        text = sources[name]
        assert "= phase * " in text, name
        assert re.search(r"= (\w+)\[base \+ 2 \* stride\];", text) is None, (
            f"{name} assigns the source plane without the parity multiply")
    for name in ("_fill_folded_far_B_kernel_code", "_fill_folded_far_D_kernel_code"):
        text = sources[name]
        assert "float parity = -phase;" in text, name
        assert "= parity * " in text, name


def test_the_near_fill_images_stored_cell_two():
    """``MIRROR_SOURCE_INDEX`` is 2 and the device text says 2."""
    assert stepping.MIRROR_SOURCE_INDEX == 2
    sources = device_sources(kernel_source())
    for name in ("_fill_symmetry_B_kernel_code", "_fill_symmetry_D_kernel_code"):
        assert "base + 2 * stride" in sources[name], name


def test_the_far_fill_writes_the_last_row_and_reads_the_derived_one():
    sources = device_sources(kernel_source())
    for name in ("_fill_folded_far_B_kernel_code", "_fill_folded_far_D_kernel_code"):
        text = sources[name]
        assert "base + last * stride] = " in text, name
        assert "base + reflect_row * stride]" in text, name
        # The image row is a RUNTIME argument, never a baked n - 2.
        assert "stride - 2" not in text and "- 2) * stride" not in text, name


def test_the_wipe_writes_positive_zero_only():
    sources = device_sources(kernel_source())
    for name in ("_zero_metal_B_kernel_code", "_zero_metal_D_kernel_code"):
        text = sources[name]
        assert "= 0.0f;" in text, name
        assert "-0.0f" not in text, name


def test_no_device_string_contains_a_contractible_expression():
    """``--fmad=false`` has nothing to disable here, and the record says so.

    Every device expression below is a single multiply or a store, so the guard is
    carried for consistency with the track rather than for correctness -- which is
    a claim about the TEXT and is checked as one.
    """
    # THE INDEX ARITHMETIC IS INTEGER and is not a contraction candidate at all:
    # ``(t / nz) * (ny * nz) + (t % nz)`` lowers to integer ops, and NVRTC's
    # ``--fmad`` governs float multiply-add only. So the scan is restricted to the
    # FLOAT-VALUED statements -- the ones mentioning ``phase``, ``parity`` or a
    # ``float`` declaration -- with the bracketed subscripts stripped first.
    additive = re.compile(r"[+-]")
    for name, text in device_sources(kernel_source()).items():
        for line in text.splitlines():
            code = line.split("//", 1)[0]
            if not any(token in code for token in ("phase", "parity", "float ")):
                continue
            body = re.sub(r"\[[^\]]*\]", "[]", code)
            if "*" not in body:
                continue
            # The one permitted minus is the parity's own, on the scalar and
            # nowhere near a multiply: ``float parity = -phase;``.
            body = body.replace("= -phase;", "= phase;")
            assert not additive.search(body), (name, line)


def test_the_compile_options_are_the_tracks():
    assert module_level_literal(kernel_source(), "_COMPILE_OPTIONS") == \
        ("--fmad=false",)


def test_the_kernel_module_imports_nothing_from_the_engine_at_module_scope():
    """The device module must load by path, outside the package, for the probe.

    Both sibling helpers (``in_seam_coverage``, ``compile_cache``) are imported
    relatively with a by-path fallback; nothing else from ``meep_gpu`` may be
    imported at module scope or the standalone load fails.
    """
    tree = ast.parse(kernel_source())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            assert not (node.module or "").startswith("meep_gpu"), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("meep_gpu"), alias.name


def test_the_coverage_module_imports_nothing_at_all():
    """``in_seam_coverage`` is the part that must be exercisable on a laptop."""
    source = (HERE / "in_seam_coverage.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in ("typing", "__future__"), alias.name
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "") in ("typing", "__future__"), node.module
