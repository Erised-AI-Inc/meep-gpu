"""The two SCRATCH-OUTPUT stencil welds: lift, predicate, rotation and record.

WHAT THIS FILE SETTLES, and what it deliberately does not. ``offdiag_fused_electric_
pair`` and ``folded_offdiag_fused_electric_pair`` are the first products this track
has put on cells the board recorded UNBUILDABLE, so the questions that decide whether
they are real are sharper than usual -- and all but one of them are answerable without
a device:

1. **Is the lift the certified text?** Every anchor the splice takes is asserted
   against the two certified strings, every edit is declared in ``LIFT_EDITS``, and
   the edits are asserted to have landed where they were meant to and NOWHERE else.
   The one whose failure is silent is the sub-lattice rename: ``kms_*`` is the INTEGER
   split-field vector for the D curl and the HALF-INTEGER one for ``update_E``, both
   real, both indexed the same way, half a cell apart in the absorber profile.

2. **Does every D read go through the resolution?** This is the whole design. A
   single surviving ``D*[...]`` load in the constitutive body would be a read of a
   volume the launch is not writing -- the PRE-launch value where the post-pass one
   belongs -- and it would be smooth, converged and wrong. The check is textual and
   exhaustive over the emitted source.

3. **Do the predicates refuse by name, and do the two partition?** A bit comparison
   on an admitted row says nothing about a row that should never have been admitted.
   The fold clause is the one that matters most here: it is what keeps the unfolded
   weld off a grid whose two mirror fills it does not carry, and what keeps the two
   products off each other's rows.

4. **Is ``CARRIES_DEPOSIT_REPAIR = False`` a MEASUREMENT?** Rule 3 of this campaign
   says the flag is a per-cell measurement off the census, not a copied constant.
   Both cells' rows are taken off the shipped census by replaying the board's own
   ``selection`` and every one is checked to carry the off-diagonal chi1inv row that
   ``deposit_repair.repairable`` refuses BY NAME -- so ``True`` could not have been
   declared, whatever a reader might have preferred.

5. **Is the ROTATION invisible to every consumer of D?** The brief for this round
   required a consumer audit of pointer identity across the step boundary, and a grep
   is not an audit. This file drives a real ``FdtdDriver`` with a source and three
   monitors for several steps, splices in an equivalent rotation at exactly the point
   the launcher rotates, and requires every stored volume AND every monitor
   accumulation to be byte-identical to the unrotated run.

NOTHING HERE COMPILES OR LAUNCHES ANYTHING. The fused kernels' arithmetic is a DEVICE
claim and no test on this host may speak to it
(``parity/meep_gpu/gate_cuda_offdiag_stencil_welds.py`` is where that is measured);
the host arithmetic of the RESOLUTION is measured separately and off-device by
``parity/meep_gpu/probe_cuda_offdiag_scratch_weld.py``. What is settled here is the
lift, the predicate, the flag and the protocol -- exactly the half a byte comparison
is blind to.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re

import numpy
import pytest

from .. import deposit_repair, stepping
from ..fields import IYEE_SHIFTS, Fields, mirror_parity
from ..grid import Grid, Mirror
from ..pml import PML
from ..sources import GaussianEnvelope, VolumeSource
from . import fused_pairs, in_seam_coverage, offdiag_emitter, offdiag_stencil_weld
from . import folded_offdiag_fused_electric_pair as folded
from . import offdiag_fused_electric_pair as flat
from .test_fused_pairs import _NumpyWearingCupysName

#: The census this round's numbers are read off. Named so a re-cut that moved the
#: rows is a NAMED failure here rather than a silently different denominator.
CENSUS = (pathlib.Path(__file__).resolve().parents[2] / "parity" / "meep_gpu"
          / "results" / "cuda_predicate_coverage_2026-09-02_electrictwins_c")

#: The two board cells these products occupy, as (step_D family, update_E family),
#: and what the census says about each: how many rows the composer lands there, and
#: how many of those carry an electric deposit inside the seam.
CELLS = {
    "flat": {"families": ("cuda_curl", "cuda_offdiag"),
             "arms": ("PML", "off-diagonal"), "rows": 16, "electric_deposit": 8},
    "folded": {"families": ("cuda_curl", "cuda_folded_offdiag"),
               "arms": ("PML", "folded off-diagonal"), "rows": 19,
               "electric_deposit": 10},
}

MODULES = {"flat": flat, "folded": folded}

#: Every row mask the two emitters are exercised on here. The corpus drives two of
#: the 63; the all-live one is the widest tree and the singleton the narrowest, and
#: the emitter's own ``corpus_digest`` covers the rest.
MASKS = ((1, 1, 1, 1, 1, 1), (1, 0, 0, 1, 0, 0), (0, 0, 1, 0, 0, 0))


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


# ---------------------------------------------------------------------------
# The certified text, without the device library and without a skip
# ---------------------------------------------------------------------------

def _string_constants(module_name):
    """Every module-level ``name = <string expression>`` in one sibling, by ``ast``.

    ``step_curl_kernels`` imports CuPy at module scope, so on the merge-bar host it
    cannot be imported -- and a test that skipped for that reason would leave the
    SPLICE, the part of these products that can be silently wrong, unchecked
    everywhere a laptop runs. The device strings are plain module-level assignments,
    so they are read from the SOURCE instead. Same reader ``test_fused_electric_pair``
    uses, and for the same reason.
    """
    path = pathlib.Path(flat.__file__).with_name(f"{module_name}.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bound = {}

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return bound.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = value(node.left), value(node.right)
            return None if left is None or right is None else left + right
        return None

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            text = value(node.value)
            if text is not None:
                bound[target.id] = text
    return bound


class _CertifiedText:
    """A stand-in for the CuPy-importing curl sibling, holding its device strings."""

    def __init__(self, module_name):
        for name, text in _string_constants(module_name).items():
            setattr(self, name, text)


@pytest.fixture
def certified(monkeypatch):
    """Both emitters, with the certified curl text supplied from source."""
    monkeypatch.setattr(offdiag_stencil_weld, "step_curl_kernels",
                        offdiag_stencil_weld.step_curl_kernels
                        or _CertifiedText("step_curl_kernels"))
    return {name: {mask: module.kernel_source(mask) for mask in MASKS}
            for name, module in MODULES.items()}


# ---------------------------------------------------------------------------
# 1. THE LIFT
# ---------------------------------------------------------------------------

def test_the_emitters_refuse_by_name_where_the_certified_curl_is_unreachable():
    """THE SPLICE IS THE LIFT, so a missing half is not a degraded emit.

    The refusal has to name the module one frame from the caller rather than surface
    as an AttributeError inside a string operation -- and both predicates must still
    answer, which is why that import is defensive in the first place.
    """
    saved = offdiag_stencil_weld.step_curl_kernels
    try:
        offdiag_stencil_weld.step_curl_kernels = None
        for module in MODULES.values():
            with pytest.raises(RuntimeError) as raised:
                module.kernel_source((1, 1, 1, 1, 1, 1))
            assert "step_curl_kernels" in str(raised.value)
        assert flat.covers_offdiag_fused_electric_pair(None, None, None, ())[0] is False
        assert folded.covers_folded_offdiag_fused_electric_pair(
            None, None, None, ())[0] is False
    finally:
        offdiag_stencil_weld.step_curl_kernels = saved


def test_a_moved_certified_anchor_fails_the_emit_rather_than_emitting(monkeypatch):
    """A certified string that changed under this family must STOP the splice.

    The alternative is a kernel that compiles and is quietly not the certified
    arithmetic, which is the one failure mode nothing downstream catches.
    """
    text = _CertifiedText("step_curl_kernels")
    text._REAL_PML_PRELUDE = text._REAL_PML_PRELUDE.replace(
        "__device__ __forceinline__ void pml_apply(", "__device__ void pml_apply(")
    monkeypatch.setattr(offdiag_stencil_weld, "step_curl_kernels", text)
    with pytest.raises(AssertionError, match="pml_apply"):
        offdiag_stencil_weld.curl_prelude()


def test_the_pure_helper_keeps_the_certified_expression_character_for_character(
        certified):
    """``pml_apply_pure`` may move both stores and may not touch either right-hand side.

    The split-field recurrence and the displacement update are the certified
    arithmetic; what this weld changes is only where the two results go.
    """
    source = certified["flat"][(1, 1, 1, 1, 1, 1)]
    assert "float fu_new = ((fprev * kms) - curl) * sinv;" in source
    assert "return (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;" in source
    # And nothing stores through the certified names any more: the whole point.
    assert "    fu[idx] = fu_new;" not in source
    assert "    f[idx] = (((f[idx] * kms_u)" not in source


def test_the_curl_body_is_lifted_whole_with_only_the_cell_edits(certified):
    """Every line of the certified ``step_D`` body survives except the two blocks
    that tie it to a thread index and the three ``pml_apply`` call lines."""
    body = offdiag_stencil_weld.split_body(
        _CertifiedText("step_curl_kernels")._step_D_pml_real_kernel_code,
        _CertifiedText("step_curl_kernels")._REAL_PML_PRELUDE,
        "_step_D_pml_real_kernel_code")
    source = certified["flat"][(1, 1, 1, 1, 1, 1)]
    edited = {"    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
              "    if (idx >= nx * ny * nz) return;",
              "    int k = idx % nz;", "    int j = (idx / nz) % ny;",
              "    int i = idx / (ny * nz);"}
    for line in body.splitlines():
        if not line.strip() or line in edited or line.lstrip().startswith("pml_apply("):
            continue
        assert line in source, f"the certified curl body's line vanished: {line!r}"
    # The three captures landed, one per component, each on the certified argument
    # list with only the two results added.
    for component, target in enumerate(offdiag_stencil_weld.D_TARGETS):
        assert (f"d_out[{component}] = pml_apply_pure({target}, fu_{target}, idx, "
                f"curl, ") in source
        assert f", &fu_out[{component}]);" in source


@pytest.mark.parametrize("name", sorted(MODULES))
@pytest.mark.parametrize("mask", MASKS)
def test_every_flux_density_read_goes_through_the_resolution(certified, name, mask):
    """THE DESIGN, CHECKED TEXTUALLY. No ``D*[...]`` load may survive below the
    kernel signature: the constitutive half reads its own cell from a register and
    every foreign cell through ``resolve_D``, and a single surviving direct load
    would be the PRE-launch value where the post-pass one belongs."""
    source = certified[name][mask]
    body = source[source.index("extern \"C\" __global__"):]
    body = body.split("\n) {\n", 1)[1]
    stripped = re.sub(r"//[^\n]*", "", body)
    offenders = re.findall(r"\b(D[xyz]|fu_D[xyz])\s*\[", stripped)
    assert not offenders, (
        f"{name}/{mask}: the kernel body loads {sorted(set(offenders))} directly; "
        f"every flux-density read must go through resolve_D or a register")
    # The scratch is written and nothing else is.
    for component, letter in enumerate("xyz"):
        assert f"D{letter}_out[idx] = v_{letter};" in stripped
        assert f"fu_D{letter}_out[idx] = fu_raw[{component}];" in stripped


@pytest.mark.parametrize("name", sorted(MODULES))
def test_the_sub_lattice_rename_landed_on_every_component_and_nowhere_else(
        certified, name):
    """``kms_*`` is the curl's INTEGER vector and ``kms_half_*`` update_E's
    HALF-INTEGER one. Both are real, both are indexed the same way, and they are half
    a cell apart in the absorber profile -- a shadow here is not a compile failure."""
    source = certified[name][(1, 1, 1, 1, 1, 1)]
    for component, axis in enumerate("xyz"):
        index = "ijk"[component]
        name_e = ("Ex", "Ey", "Ez")[component]
        assert (f"constitutive_apply({name_e}, f_w_{name_e}, idx, src_{name_e}, "
                f"kps_{axis}[{index}], kms_half_{axis}[{index}]);") in source
    # The curl's own three calls keep the INTEGER names, untouched.
    assert "curl, kms_y[j], sinv_y[j], kms_z[k], sinv_z[k]" in source
    assert "curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]" in source
    assert "curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]" in source


def test_only_the_folded_weld_renames_the_constitutive_boundary_codes(certified):
    """The two readings of a grid's boundaries differ ON A FOLD and nowhere else.

    The folded weld binds both and its constitutive body reads ``cbc_*``; the
    unfolded one binds ONE triple and cross-checks the two readings in its launcher,
    which is why its body must still read ``bc_*``.
    """
    folded_source = certified["folded"][(1, 1, 1, 1, 1, 1)]
    flat_source = certified["flat"][(1, 1, 1, 1, 1, 1)]
    for axis, letter in enumerate("xyz"):
        coordinate = "ijk"[axis]
        extent = ("nx", "ny", "nz")[axis]
        assert (f"coord_dn({coordinate}, {extent}, cbc_{letter})") in folded_source
        assert (f"mg_{letter} = (cbc_{letter} == BC_MIRROR)") in folded_source
        assert (f"coord_dn({coordinate}, {extent}, bc_{letter})") in flat_source
    assert "cbc_" not in flat_source


@pytest.mark.parametrize("name", sorted(MODULES))
def test_the_lift_edits_are_data_and_carry_no_silent_extra(name):
    """``LIFT_EDITS`` is DATA so a gate can assert it rather than a docstring.

    The shared CURL edits are the same six on both families -- they are one lift --
    and each family's own edits are counted separately, so an edit that appeared in
    the emitter without a row here is a failure.
    """
    edits = MODULES[name].LIFT_EDITS
    assert all(set(edit) == {"line", "became", "why"} for edit in edits)
    assert edits[:len(offdiag_stencil_weld.CURL_LIFT_EDITS)] == \
        offdiag_stencil_weld.CURL_LIFT_EDITS
    own = edits[len(offdiag_stencil_weld.CURL_LIFT_EDITS):]
    assert len(own) == {"flat": 9, "folded": 13}[name], (
        f"{name} declares {len(own)} of its own lift edits; a count that moved "
        f"without this number moving is an edit nobody wrote down")
    assert all(edit["why"] for edit in edits)


def test_the_two_emitters_agree_on_the_shared_helpers(certified):
    """The curl lift, the argument pack and the resolution are ONE piece of text.

    Two copies of a resolution are equal only until someone edits one, which is why
    they live in a shared module -- and this is the check that they still do.
    """
    shared = (offdiag_stencil_weld.curl_prelude()
              + offdiag_stencil_weld.weld_args_struct()
              + offdiag_stencil_weld.raw_step_D_cell_source()
              + offdiag_stencil_weld.resolution_source())
    for name in MODULES:
        assert certified[name][(1, 1, 1, 1, 1, 1)].startswith(shared)


# ---------------------------------------------------------------------------
# 2. THE RESOLUTION, against the engine's own tables
# ---------------------------------------------------------------------------

def test_the_axis_sets_are_read_off_the_yee_table_not_assumed():
    """A D component's NEAR axes and its CLEARED axes are the same pair (both are
    the Yee-shift-0 test), and its FAR axis is its one shift-1 axis. That coincidence
    is why a near ghost can land in a cleared plane at all."""
    for component, target in enumerate(offdiag_stencil_weld.D_TARGETS):
        shifts = IYEE_SHIFTS[target]
        near = offdiag_stencil_weld.near_axes(component)
        assert near == tuple(a for a in range(3) if shifts[a] == 0)
        assert offdiag_stencil_weld.clear_axes(component) == near
        assert shifts[offdiag_stencil_weld.far_axis(component)] == 1
        assert len(near) == 2


def test_the_parities_are_the_engines_own_and_collapse_per_axis(xp):
    """``fill_plan`` hands the kernel two float triples so it performs NO negation.

    Each is ``fields.mirror_parity`` of the axis's own shift-0 / shift-1 components,
    and the collapse to one value per axis is ASSERTED in the plan rather than
    assumed -- a drifted Yee table would put a wrong SIGN on a whole plane.
    """
    grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=xp,
                symmetry=(Mirror("Y", -1),))
    plan = offdiag_stencil_weld.fill_plan(grid)
    assert plan["near"] == (0, 1, 0)
    phase = int(grid.mirror_phase(1))
    assert plan["near_phase"][1] == float(mirror_parity("Dx", 1, phase))
    assert plan["far_phase"][1] == float(mirror_parity("Dy", 1, phase))
    assert plan["near_phase"][1] == -plan["far_phase"][1]
    # An unfolded axis carries 0.0 and a -1 reflect sentinel, never a live value.
    assert plan["near_phase"][0] == plan["far_phase"][0] == 0.0
    assert plan["reflect"][0] == -1


class _Stub:
    """A grid that answers exactly the questions the fill plan asks, and lies."""

    def __init__(self, **answers):
        self.__dict__.update(answers)
        self.shape = (8, 8, 8)
        self.has_metallic = bool(answers.get("metallic"))

    def has_symmetry(self):
        return True

    def is_mirrored(self, axis):
        return axis == 1

    def mirror_phase(self, axis):
        return self.phase

    def is_metallic(self, axis):
        return bool(self.metallic) and axis == 1

    def stored_cells(self, axis):
        return self.stored if axis == 1 else 8

    def owned_cells(self, axis):
        return 8

    @property
    def shape_full(self):
        return (8, 16, 8)


def test_the_fill_plan_raises_rather_than_planning_what_it_cannot_stand_behind():
    """A launcher handed a grid the predicate would have refused must not quietly
    build a plan for it: the failure mode is a plane of wrong values, not a crash.

    THE THREE CLAUSES ARE NOT EQUALLY REACHABLE and the test says which is which.
    An unreadable plane phase is reachable and armed below. The orphan far row --
    a reflect row on an axis with no declared phase -- is reachable and armed. The
    "folded AND walled" clause is BELT AND BRACES: ``in_seam_coverage.zero_metal_axes``
    answers ``is_metallic(axis) and not is_mirrored(axis)``, so it can never report a
    mirrored axis as walled, and the clause exists for the day that reader changes.
    Recording that here rather than arming it with a stub that could not occur is the
    vacuity discipline this campaign runs on.
    """
    with pytest.raises(ValueError, match="mirror phase"):
        offdiag_stencil_weld.fill_plan(_Stub(phase=0, metallic=False, stored=8))
    assert in_seam_coverage.zero_metal_axes(
        _Stub(phase=1, metallic=True, stored=8)) == (False, False, False), (
        "zero_metal_axes reported a MIRRORED axis as walled; the fill plan's "
        "folded-and-walled clause has become reachable and now needs its own arm")


# ---------------------------------------------------------------------------
# 3. THE PREDICATES, ON REAL ENGINE OBJECTS
# ---------------------------------------------------------------------------

def _rows(fields, rows):
    shape = tuple(fields.grid.shape)
    rng = numpy.random.default_rng(11)
    names = ("Ex", "Ey", "Ez")
    epsilon = {n: numpy.full(shape, v, numpy.float32)
               for n, v in zip(names, (2.0, 2.5, 3.0))}
    inverse = {n: numpy.full(shape, numpy.float32(1.0 / v), numpy.float32)
               for n, v in zip(names, (2.0, 2.5, 3.0))}
    built = {row: {partner: rng.uniform(-0.25, 0.25, shape).astype(numpy.float32)
                   for partner in partners}
             for row, partners in rows.items()}
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=built)
    return fields


def _build(xp, *, symmetry=(), boundaries=("periodic", "periodic", "periodic"),
           rows=None, cell=(8.0, 8.0, 8.0)):
    grid = Grid(resolution=1.0, cell_size=cell, boundaries=boundaries,
                symmetry=symmetry, xp=xp)
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2) for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    _rows(fields, rows or {"Ex": ("Ey",)})
    return fields, layer, grid


def _admitters(fields, layer, grid):
    admitted = []
    if flat.covers_offdiag_fused_electric_pair(fields, layer, grid, ())[0]:
        admitted.append("flat")
    if folded.covers_folded_offdiag_fused_electric_pair(fields, layer, grid, ())[0]:
        admitted.append("folded")
    return admitted


def test_the_unfolded_weld_admits_its_cell_alone(xp):
    fields, layer, grid = _build(xp)
    assert _admitters(fields, layer, grid) == ["flat"]


def test_the_folded_weld_admits_its_cell_alone(xp):
    fields, layer, grid = _build(xp, symmetry=(Mirror("Y", 1),))
    assert _admitters(fields, layer, grid) == ["folded"]


def test_the_two_welds_partition_on_the_fold_at_both_terminations(xp):
    """The one boolean the whole partition rests on, measured at BOTH terminations:
    a mutation caught on one is not caught on the other."""
    for boundaries in (("periodic", "periodic", "periodic"),
                       ("periodic", "metallic", "periodic")):
        fields, layer, grid = _build(xp, symmetry=(Mirror("Y", 1),),
                                     boundaries=boundaries)
        assert _admitters(fields, layer, grid) == ["folded"], boundaries
        fields, layer, grid = _build(xp, boundaries=boundaries)
        assert _admitters(fields, layer, grid) == ["flat"], boundaries


def test_a_run_with_no_off_diagonal_row_is_refused_by_both(xp):
    """The seam against ``covers_real_pml_constitutive(side='E')``: that predicate
    refuses every off-diagonal run, and these two REQUIRE one, so a diagonal run
    belongs to the plain electric pair and to neither of these."""
    grid = Grid(resolution=1.0, cell_size=(8.0, 8.0, 8.0), xp=xp)
    layer = PML(grid=grid, thickness=(2, 2, 2))
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    assert _admitters(fields, layer, grid) == []


def test_an_undeclared_source_set_is_refused_by_both(xp):
    """IGNORANCE IS NEVER AN EMPTY SET. ``Fields`` does not hold the source list, so
    a predicate that inferred "no sources" from not being told would be exactly the
    over-covering the seam clause exists to prevent."""
    fields, layer, grid = _build(xp)
    covered, why = flat.covers_offdiag_fused_electric_pair(fields, layer, grid, None)
    assert not covered and "was not declared" in why
    fields, layer, grid = _build(xp, symmetry=(Mirror("Y", 1),))
    covered, why = folded.covers_folded_offdiag_fused_electric_pair(
        fields, layer, grid, None)
    assert not covered and "was not declared" in why


def test_an_electric_deposit_is_refused_by_name_and_names_the_repair_clause(xp):
    """The source seam, REFUSED rather than carried, and the refusal has to say WHY
    the carry is unavailable rather than merely that it was not declared."""
    fields, layer, grid = _build(xp)
    source = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    covered, why = flat.covers_offdiag_fused_electric_pair(
        fields, layer, grid, (source,))
    assert not covered
    assert "off-diagonal chi1inv row" in why and "no deposit repair" in why


def test_a_magnetic_deposit_does_not_disqualify_either_weld(xp):
    """A MAGNETIC source is injected in the B/H half, outside this seam. If it
    disqualified these products the clause would be asking about the wrong list --
    the exact defect the shared clause was introduced to prevent."""
    fields, layer, grid = _build(xp)
    source = VolumeSource(grid=grid, component="Hz", center=(0.0, 0.0, 0.0),
                          size=(0.0, 0.0, 0.0),
                          envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                          amplitude=1.0)
    assert flat.covers_offdiag_fused_electric_pair(
        fields, layer, grid, (source,))[0]


def test_the_unfolded_weld_refuses_a_fold_in_its_own_words(xp):
    """The clause is written even though the constitutive arm already refuses a fold
    twice: the guarantee that the two mirror fills do nothing inside this seam is
    what makes REPLACES honest, and it may not depend on another module's clause."""
    fields, layer, grid = _build(xp, symmetry=(Mirror("Y", 1),))
    covered, why = flat.covers_offdiag_fused_electric_pair(fields, layer, grid, ())
    assert not covered
    assert "mirror" in why.lower()


def test_the_folded_weld_refuses_a_fold_free_grid_in_its_own_words(xp):
    """Its constitutive arm's STANDALONE predicate admits zero folded axes on
    purpose; the COMPOSITION predicate requires one, and this clause says so from
    the seam's side so two products can never claim one row."""
    fields, layer, grid = _build(xp)
    covered, why = folded.covers_folded_offdiag_fused_electric_pair(
        fields, layer, grid, ())
    assert not covered


# ---------------------------------------------------------------------------
# 4. CARRIES_DEPOSIT_REPAIR, MEASURED OFF THE CENSUS
# ---------------------------------------------------------------------------

def _census_rows():
    rows = []
    for name in ("examples.jsonl", "tests.jsonl", "tests_param_matched.jsonl"):
        path = CENSUS / name
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


@pytest.mark.parametrize("name", sorted(CELLS))
def test_the_flag_is_false_because_every_row_of_the_cell_refuses_the_repair(name):
    """RULE 3: the flag is a per-cell MEASUREMENT off the census, never a constant
    copied from a sibling.

    ``deposit_repair.repairable(fields, "D")`` refuses an off-diagonal chi1inv row BY
    NAME, and both constitutive arms REQUIRE one to be installed. So every row the
    composer lands on either cell carries the thing that makes the repair
    unavailable, and ``True`` could not have been declared however convenient it
    would have been -- which is the opposite of the five electric twins, where the
    flag was load-bearing and measured True.
    """
    if not CENSUS.is_dir():  # pragma: no cover - the census travels with the tree
        pytest.skip(f"{CENSUS} is not in this tree")
    import sys  # noqa: PLC0415 - the board builder is a parity script, not a package
    parity = str(CENSUS.parents[1])
    if parity not in sys.path:
        sys.path.insert(0, parity)
    import build_cuda_fusion_matrix as board  # noqa: PLC0415

    cell = CELLS[name]
    hits = []
    for row in _census_rows():
        curl = board.selection(row, "step_D")
        constitutive = board.selection(row, "update_E")
        if not curl or not constitutive:
            continue
        if (curl["family"], constitutive["family"]) != cell["families"]:
            continue
        assert (curl["arm"], constitutive["arm"]) == cell["arms"]
        hits.append(row)
    assert len(hits) == cell["rows"], (
        f"{name}: the census lands {len(hits)} rows on this cell, not "
        f"{cell['rows']}; the denominator moved and every number below it is stale")
    for row in hits:
        assert row["configuration"]["has_offdiagonal_epsilon"], (
            f"{name}: {row.get('row')} sits on this cell with NO off-diagonal "
            f"chi1inv row, which the constitutive arm requires")
    deposits = sum(1 for row in hits
                   if "D" in (row["configuration"].get("source_field_types") or []))
    assert deposits == cell["electric_deposit"], (
        f"{name}: {deposits} of the cell's rows deposit inside the seam, not "
        f"{cell['electric_deposit']}; the served ceiling moved")
    assert MODULES[name].CARRIES_DEPOSIT_REPAIR is False


def test_the_repair_itself_refuses_the_configuration_by_name(xp):
    """The clause the flag rests on, asked of a real engine object rather than read
    out of a source file."""
    fields, layer, grid = _build(xp)
    ok, reasons = deposit_repair.repairable(fields, "D", layer)
    assert not ok
    assert any("off-diagonal chi1inv row" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# 5. THE ROTATION IS INVISIBLE TO EVERY CONSUMER
# ---------------------------------------------------------------------------

D_STATE = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")
ALL_STATE = tuple(f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz")
ALL_STATE += tuple(f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz")
ALL_STATE += tuple(f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz")


def _driver_with_monitors(seed):
    from ..driver import FdtdDriver  # noqa: PLC0415 - the engine, not the kernels

    driver = FdtdDriver(cell_size=(1.6, 1.6, 0.0), resolution=8.0, dimensions=2,
                        courant=0.35, force_complex_fields=False)
    driver.add_source({"component": "Ez", "center": (0.0, 0.0, 0.0),
                       "size": (0.0, 0.0, 0.0), "source_type": "gaussian",
                       "frequency": 1.0, "fwidth": 0.4, "amplitude": 1.0})
    # THREE MONITORS, AND ONE OF THEM READS D DIRECTLY. A rotation the D consumers
    # could not see would be an untested claim; the Dx DFT monitor is the consumer
    # this leg exists for.
    driver.add_dft_monitor(frequencies=[1.0], components=("Dx", "Dy"))
    driver.add_dft_monitor(frequencies=[1.0, 1.3], components=("Ez",))
    driver.add_flux_monitor(frequencies=[1.0], center=(0.0, 0.0, 0.0),
                            size=(0.8, 0.0, 0.0))
    rng = numpy.random.default_rng(seed)
    for name in ALL_STATE:
        array = getattr(driver.fields, name, None)
        if array is not None:
            array[...] = rng.normal(0.0, 0.2, array.shape).astype(array.dtype)
    return driver


def _monitor_words(driver):
    """Every monitor's accumulated spectrum, as raw bytes.

    Read through ``_dft`` -- the accumulator dict every DFT and flux monitor keeps --
    rather than through a public getter, because what this leg has to compare is the
    STATE the monitors carry, not a derived view that could round two divergent
    accumulations onto the same number.
    """
    out = {}
    for index, monitor in enumerate(list(driver._dft_monitors)
                                    + list(driver._flux_monitors)):
        store = getattr(monitor, "_dft", None)
        if not isinstance(store, dict):
            continue
        for key in sorted(store, key=repr):
            value = store[key]
            if value is None:
                continue
            out[f"{index}:{key!r}"] = numpy.frombuffer(
                numpy.ascontiguousarray(value).tobytes(), dtype=numpy.uint8)
    assert out, "no monitor accumulation was found; this leg would be vacuous"
    assert any(numpy.any(words) for words in out.values()), (
        "every monitor accumulation is zero; the comparison below would pass on a "
        "run that did nothing")
    return out


def test_rotating_the_flux_density_between_steps_is_invisible_to_every_consumer():
    """THE CONSUMER AUDIT THE SCRATCH DESIGN RESTS ON, MEASURED RATHER THAN GREPPED.

    The launcher rebinds ``fields.Dx``/``fu_Dx`` (and the other four) to different
    ALLOCATIONS after every launch. That is only sound if nothing in the engine holds
    a D array across a step boundary -- monitors, sources, the driver's decay probes,
    ``stepping``'s derived-E branch. Every one of them resolves the volume BY NAME at
    use time, and this leg is what turns that reading into a measurement: an
    equivalent rotation is spliced in at exactly the point the fused launcher rotates
    (immediately after ``update_E``, before ``update_P`` and the monitor updates), and
    every stored volume AND every monitor accumulation must come out byte-identical.

    A rotation that copied VALUES would pass this vacuously, so the shim copies into a
    FRESH allocation each step and asserts the identity of the arrays it replaced has
    actually changed.
    """
    plain = _driver_with_monitors(4242)
    rotated = _driver_with_monitors(4242)
    for name in ALL_STATE:
        a = getattr(plain.fields, name, None)
        b = getattr(rotated.fields, name, None)
        if a is not None and b is not None:
            numpy.testing.assert_array_equal(a, b)

    addresses_before = {name: id(getattr(rotated.fields, name)) for name in D_STATE}
    for _ in range(4):
        plain.step()
        rotated.step()
        # THE SHIM: the same rebinding the launcher performs, with the values
        # unchanged. A fresh allocation each time, so the ARRAY IDENTITY really moves.
        for name in D_STATE:
            current = getattr(rotated.fields, name)
            replacement = numpy.array(current, copy=True)
            setattr(rotated.fields, name, replacement)
    assert all(id(getattr(rotated.fields, name)) != addresses_before[name]
               for name in D_STATE), (
        "the shim did not actually move any array identity; this leg would be "
        "vacuous")

    for name in ALL_STATE:
        a = getattr(plain.fields, name, None)
        b = getattr(rotated.fields, name, None)
        if a is None and b is None:
            continue
        numpy.testing.assert_array_equal(
            a, b, err_msg=f"{name} diverged under the rotation shim")
    left, right = _monitor_words(plain), _monitor_words(rotated)
    assert set(left) == set(right)
    for key in sorted(left):
        numpy.testing.assert_array_equal(
            left[key], right[key],
            err_msg=f"monitor accumulation {key} diverged under the rotation shim")


@pytest.mark.parametrize("name", sorted(MODULES))
def test_the_rotation_hands_back_the_retired_pair(name, xp):
    """The ping-pong: what the launcher returns is the NEXT launch's scratch, and the
    field now holds what the launch wrote. Checked on plain arrays, because the
    rebinding is host bookkeeping and needs no device."""
    fields, _layer, _grid = _build(xp)
    module = MODULES[name]
    scratch = {volume: numpy.empty_like(getattr(fields, volume))
               for volume in module.SCRATCH_VOLUMES}
    before = {volume: getattr(fields, volume) for volume in module.SCRATCH_VOLUMES}
    retired = module.rotate_into_fields(fields, scratch)
    for volume in module.SCRATCH_VOLUMES:
        assert getattr(fields, volume) is scratch[volume]
        assert retired[volume] is before[volume]


# ---------------------------------------------------------------------------
# 6. THE ALIASING CHECK IS THE ONE THAT REFUSES THE IN-PLACE WELD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MODULES))
def test_a_scratch_that_aliases_the_storage_is_refused_by_name(name, xp):
    """If ``Dx_out`` were ``Dx`` the launch would BE the in-place weld the board
    refused, and every foreign recompute would read words other blocks had already
    overwritten. The check runs before every launch, and this is the leg that proves
    it fires -- the device gate arms the same defect as a mutation."""
    module = MODULES[name]

    class _Address:
        """A base address, which is all the check reads. NumPy forbids assigning
        ``ndarray.data``, and a CuPy array is what the shipped path binds -- so the
        thing under test is handed the one attribute it consults."""

        def __init__(self, tag):
            self.data = type("P", (), {"ptr": tag})()
            self.shape = (4,)

    class _Fields:
        def __init__(self):
            self._by_name = {}
            for index, volume in enumerate(
                    ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz", "Hx", "Hy", "Hz",
                     "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")):
                self._by_name[volume] = _Address(1000 + index)

        def __getattr__(self, item):
            try:
                return self._by_name[item]
            except KeyError as exc:  # pragma: no cover - a typo in the test itself
                raise AttributeError(item) from exc

        def inverse_epsilon_for(self, component):
            return _Address(2000 + "xyz".index(component[1]))

    fields = _Fields()
    tables = {group: {key: _Address(3000 + offset + index * 10)
                      for index, key in enumerate(
                          ("kps_x", "kms_x", "kps_y", "kms_y", "kps_z", "kms_z",
                           "sinv_x", "sinv_y", "sinv_z"))}
              for offset, group in enumerate(("curl", "constitutive"))}
    # A DISJOINT scratch passes, so the refusal below is a verdict about the ALIAS
    # rather than about the stub.
    clean = {volume: _Address(9000 + index)
             for index, volume in enumerate(module.SCRATCH_VOLUMES)}
    assert module.assert_scratch_is_disjoint(fields, clean, tables) > 0
    aliased = dict(clean)
    aliased["Dx"] = fields.Dx
    with pytest.raises(ValueError, match="in-place weld"):
        module.assert_scratch_is_disjoint(fields, aliased, tables)


# ---------------------------------------------------------------------------
# 7. THE WIRING TABLES AGREE, MECHANICALLY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MODULES))
def test_each_weld_row_names_its_own_module_seam_and_arms(name):
    """A product with no row in every table is a half-built product, and a row that
    names another module's kernel binds this configuration's values to a different
    kernel's parameters."""
    module = MODULES[name]
    family = module.FAMILY
    assert fused_pairs.FUSED_PRODUCTS[family]["module"] == module.__name__.rsplit(
        ".", 1)[-1]
    assert fused_pairs.FUSED_PRODUCTS[family]["curl_slot"] == "step_D"
    assert fused_pairs.FUSED_PAIR_ARMS[family] == CELLS[name]["arms"]
    assert module.SLOT == "step_D"
    assert module.REPLACES[0] == "step_D" and module.REPLACES[-1] == "update_E"
    # RELEASED 2026-09-02: the kernel moved from UNCERTIFIED to CERTIFIED as the
    # FINAL BYTES the gate then ran against, which is this campaign's rule 4 --
    # a module that moved its own name after the run would leave the record bound
    # to bytes that no longer ship. test_kernel_partition.py refuses a name in
    # CERTIFIED_KERNELS that no certification.json block claims, so this line and
    # that record move together or the tree goes red.
    assert module.CERTIFIED_KERNELS == (module.KERNEL_NAME,)
    assert module.UNCERTIFIED_KERNELS == {}


def test_the_folded_weld_replaces_all_three_in_seam_passes_and_the_flat_one_does_not():
    """REPLACES is DECLARED, and the declaration is what a composition reports. The
    unfolded weld's predicate refuses a fold, so the two mirror fills cannot run
    inside its seam and claiming them would be false; the folded one carries all
    three."""
    assert flat.REPLACES == ("step_D", "zero_metal_D", "update_E")
    assert folded.REPLACES == ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                               "fill_folded_far_ghosts_D", "update_E")
    assert set(folded.REPLACES) >= set(in_seam_coverage.PASSES and
                                       ("fill_symmetry_bc_D", "zero_metal_D",
                                        "fill_folded_far_ghosts_D"))


def test_the_emitters_are_the_certified_row_mask_vocabulary():
    """The row mask is ``offdiag_emitter``'s, by value, so a weld cannot emit a mask
    its constitutive half would refuse."""
    for module in MODULES.values():
        with pytest.raises(ValueError, match="no row slot survives"):
            module.kernel_source((0, 0, 0, 0, 0, 0))
    assert len(offdiag_emitter.LIVE_ROW_MASKS) == 63


def test_the_carried_passes_are_the_ones_the_engine_names():
    """``REPLACES`` names driver passes, and the names have to be the engine's own --
    a composition reporting a pass the driver does not call is unfalsifiable."""
    for name in ("fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D"):
        assert callable(getattr(stepping, name))
