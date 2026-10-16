"""The conductive x ordinary D/E weld: emitter, predicate and the signed-zero cell.

WHAT THIS SUITE PINS, and each section says which failure it is the tripwire for:

1. THE LIFT. The emitted kernel is the two CERTIFIED bodies with a named list of
   eight edits and nothing else, and ONE decode of the chunk record serves both. Both curl helpers become VALUES and BOTH KEEP THEIR
   AUXILIARY STORES -- ``fu`` and ``f_cond`` are the curl's own histories, and a
   rewrite that swallowed one would leave the recurrence a step behind at every cell,
   which no seam test would notice.
2. THE COEFFICIENT RENAME. ``kms_*`` is the INTEGER sigma sub-lattice in the curl body
   and the HALF-INTEGER one in the constitutive body. On the bare name the two
   parameters collide and one launch binds a HALF-CELL ERROR in the absorber profile
   -- not a crash, and not something a coarse comparison would catch.
3. THE SEAM ORDER. The wall clear sits between the curl's register capture and both
   its consumers.
4. THE PREDICATE, on real engine objects, including the disjointness against the real
   twin -- an overlap would leave the seam UNFUSED and cost the 79 rows that product
   serves.
5. THE WHOLE-VOLUME RESCALE REFUSAL, which is what remains of the clause that used to
   refuse this cell entirely.
6. THE CENSUS CELL: one corpus row, its arms, and ``CARRIES_DEPOSIT_REPAIR`` as a
   per-cell MEASUREMENT.

``constitutive_kernels`` imports CuPy at scope, so its device text is supplied FROM
SOURCE by ``ast`` rather than skipped -- a skip here would be a silent coverage gap on
the half this family exchanges nothing in.
"""

from __future__ import annotations

import ast
import json
import pathlib

import numpy
import pytest

from .. import deposit_repair
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from ..sources import GaussianEnvelope, VolumeSource
from . import conductive_fused_electric_pair as family
from . import fused_pairs
from . import fused_electric_pair as real
from .test_fused_pairs import _NumpyWearingCupysName

#: The census this round's numbers are read off.
CENSUS = (pathlib.Path(__file__).resolve().parents[2] / "parity" / "meep_gpu"
          / "results" / "cuda_predicate_coverage_2026-09-02_conductive_final")

#: The board cell this product occupies. THE DENOMINATOR IS ONE.
CELL = {
    "seam": "D_to_E",
    "curl_arm": "cuda_conductive",
    "constitutive_arm": "cuda_constitutive",
    "rows": 1,
}

CELL_ROWS = ("tests:TestAdjointSolver.test_damping",)


@pytest.fixture
def xp():
    return _NumpyWearingCupysName()


# ---------------------------------------------------------------------------
# The certified text, without the device library and without a skip
# ---------------------------------------------------------------------------

def _string_constants(module_name):
    """Every module-level ``name = <string expression>`` in one sibling, by ``ast``."""
    path = pathlib.Path(real.__file__).with_name(f"{module_name}.py")
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
        if not isinstance(target, ast.Name):
            continue
        text = value(node.value)
        if text is not None:
            bound[target.id] = text
    return bound


class _CertifiedText:
    """A stand-in for one CuPy-importing sibling, holding only its device strings."""

    def __init__(self, module_name):
        for name, text in _string_constants(module_name).items():
            setattr(self, name, text)


@pytest.fixture
def source(monkeypatch):
    """The emitter, with the CuPy-importing constitutive half supplied from source."""
    if getattr(family, "constitutive_kernels", None) is None:
        monkeypatch.setattr(family, "constitutive_kernels",
                            _CertifiedText("constitutive_kernels"), raising=False)
    return family.conductive_fused_electric_pair_source


# ---------------------------------------------------------------------------
# 1. THE LIFT
# ---------------------------------------------------------------------------

def test_the_curl_stencil_is_reused_unchanged(source):
    """MEEP's stencil, once, in the lifted ``curl_stencil`` helper; called thrice."""
    text = source((True, True, True))
    assert text.count("dtdx * (((sf - f1) + f2) - ss)") == 1
    assert text.count("        float curl = curl_stencil(dtdx, ") == 3


def test_one_decode_of_the_chunk_record_serves_both_halves(source):
    """D10: each lifted body's block is removed and the real twin's union block emitted.

    The D curl reads the integer record and the presence word, update_E the
    half-integer record; every record word is decoded exactly once, directly after the
    index decode, and the kernel takes the record as its last argument.
    """
    text = source((True, True, True))
    block = real.union_bounds_block()
    assert text.count(block) == 1
    assert text.index("    int i = idx / (ny * nz);\n") < text.index(block)
    for word in range(25):
        assert text.count(f"cb[{word}]") == 1, word
    signature = text.split('extern "C" __global__ void', 1)[1].split("\n) {\n", 1)[0]
    assert signature.rstrip().endswith("const int* __restrict__ cb")


def test_both_curl_helpers_become_values_and_keep_their_history_stores(source):
    """The two helpers return the displacement; ``fu`` and ``f_cond`` still store.

    THE FAILURE THIS CATCHES IS SILENT. The weld needs the displacement in a register
    so the wall clear can reach it before the store -- but ``fu`` (the split-field
    auxiliary) and ``f_cond`` (the conductive third history) are the CURL's own state,
    which ``zero_metal_D`` does not touch and ``update_E`` does not read. A value
    rewrite that dropped either store would leave the recurrence one step behind at
    every cell of the grid, and every seam-order test here would still pass.
    """
    text = source((True, True, True))
    assert "__device__ __forceinline__ float pml_apply_reg(" in text
    assert "__device__ __forceinline__ float cond_pml_apply_reg(" in text
    assert "    if (dsigu) fu[idx] = fu_new;" in text
    assert "    if (dsigu) u[idx] = u_new;" in text
    assert "    if (dsig) c[idx] = c_new;" in text
    assert "    return value;" in text


def test_the_conductive_case_selection_is_the_certified_one(source):
    """The per-chunk form choice and its flags are lifted, not retyped.

    ``cond_pml_apply`` selects MEEP's conductive forms per cell from the chunk record's
    membership flags and the keep flag; a chunk that keeps no conductivity steps the
    certified ``pml_apply``, whose value is carried like the others'. No coefficient
    identity predicate is left.
    """
    text = source((True, True, True))
    assert "    int dsig, int dsigu, int kept\n" in text
    assert "    float value = dsigu ? ((((fv * km2) + u_new) - uv) * si2)" in text
    assert "        return pml_apply_reg(f, u, idx, curl, km1, si1, km2, si2, dsigu);" in text
    assert "!= 1.0f" not in text


def test_both_branches_of_the_certified_conditional_are_captured(source):
    """Every per-target tail writes a REGISTER, on both branches of the ``#if``.

    The ARGUMENT LIST is carried whole -- every operand, its order and the coefficient
    indices are the certified ones. A capture that sliced arguments off by an offset
    would silently drop the leading ones.
    """
    text = source((True, False, True))
    assert ("        d_x = cond_pml_apply_reg(Dx, fu_Dx, fc0, cf0, ci0, idx, curl, "
            "kms_y[j], sinv_y[j], kms_z[k], sinv_z[k], in_i_y, in_i_z, KEPT0);") in text
    assert ("        d_x = pml_apply_reg(Dx, fu_Dx, idx, curl, kms_y[j], sinv_y[j], "
            "kms_z[k], sinv_z[k], in_i_z);") in text


def test_no_curl_tail_stores_into_the_displacement(source):
    text = source((True, True, True))
    body = text.split("float d_z = 0.0f;", 1)[1].split("stepping._zero_metal", 1)[0]
    for axis in "xyz":
        assert f"D{axis}[idx] = " not in body


def test_the_lift_edits_are_declared_and_no_more():
    assert len(family.LIFT_EDITS) == 8
    for edit in family.LIFT_EDITS:
        assert set(edit) == {"what", "from", "to", "why"}
        assert edit["why"]


# ---------------------------------------------------------------------------
# 2. THE COEFFICIENT RENAME -- a half-cell error, not a crash
# ---------------------------------------------------------------------------

def test_the_two_sigma_sub_lattices_do_not_collide(source):
    """The constitutive half reads ``kms_half_*``; the curl half reads ``kms_*``.

    ``step_D`` takes the INTEGER sub-lattice and ``update_E`` the HALF-INTEGER one
    (stepping.py:1015). On the bare name one launch would bind one vector where the
    other belongs -- a half-cell error in the absorber profile that produces plausible
    fields, which is precisely why it is pinned as text rather than trusted to a
    numeric comparison.
    """
    text = source((True, True, True))
    signature = text.split('extern "C" __global__ void', 1)[1].split("\n) {\n", 1)[0]
    for axis in "xyz":
        assert f"const float* __restrict__ kms_{axis}," in signature
        assert f"const float* __restrict__ kms_half_{axis}," in signature
    for axis, coordinate in zip("xyz", "ijk"):
        assert (f"    constitutive_apply(E{axis}, f_w_E{axis}, idx, src_{axis}, "
                f"kps_{axis}[{coordinate}], kms_half_{axis}[{coordinate}], "
                f"in_h_{axis});") in text


def test_the_curl_and_constitutive_tables_are_the_two_sub_lattices():
    """The launcher's two key groups are the two sub-lattices, not one repeated."""
    assert family._CURL_TABLE_KEYS == ("kms_x", "sinv_x", "kms_y", "sinv_y",
                                       "kms_z", "sinv_z")
    assert family._CONSTITUTIVE_TABLE_KEYS == ("kps_x", "kms_x", "kps_y", "kms_y",
                                               "kps_z", "kms_z")


# ---------------------------------------------------------------------------
# 3. THE SEAM ORDER
# ---------------------------------------------------------------------------

def test_the_clear_sits_between_the_capture_and_both_consumers(source):
    text = source((True, True, True))
    capture = text.index("d_z = cond_pml_apply_reg(Dz,")
    clear = text.index("if (wall_z && k == 0)")
    store = text.index("    Dz[idx] = d_z;")
    read = text.index("float src_z = d_z * inv_eps_Ez[idx];")
    assert capture < clear < store, "the wall clear must precede the store"
    assert clear < read, "the wall clear must precede the constitutive read"


def test_the_wall_clears_the_off_diagonal_pair_on_each_axis(source):
    text = source((True, True, True))
    assert "if (wall_x && i == 0) { d_y = 0.0f; d_z = 0.0f; }" in text
    assert "if (wall_y && j == 0) { d_x = 0.0f; d_z = 0.0f; }" in text
    assert "if (wall_z && k == 0) { d_x = 0.0f; d_y = 0.0f; }" in text


def test_the_displacement_is_bound_exactly_once(source):
    """D appears once in the signature, and the constitutive half reads no D volume."""
    text = source((True, True, True))
    signature = text.split('extern "C" __global__ void', 1)[1].split("\n) {\n", 1)[0]
    for axis in "xyz":
        assert signature.count(f" D{axis},") + signature.count(f" D{axis}\n") == 1
    constitutive = text.split("Dz[idx] = d_z;", 1)[1]
    for axis in "xyz":
        assert f"D{axis}[idx]" not in constitutive


def test_the_fully_lossless_mask_is_refused_by_name():
    """``(False, False, False)`` is a body no admitted configuration can reach.

    ``covers_conductive_curl`` refuses a run where no target of ``step_D`` carries a
    sigma BY NAME, and that run belongs to the real twin. Emitting it would put a
    kernel in the digest that nothing can ever launch.
    """
    with pytest.raises(ValueError, match="not a body this family emits"):
        family.conductive_fused_electric_pair_source((False, False, False))
    assert (False, False, False) not in family.COND_MASKS_SWEPT


def test_every_swept_source_emits_and_is_pure_ascii(source):
    sources = family.device_sources()
    assert len(sources) == len(family.COND_MASKS_SWEPT) * len(family.KEEP_MASKS_SWEPT)
    assert len(set(sources.values())) == len(sources)
    for text in sources.values():
        text.encode("ascii")


def test_a_moved_certified_anchor_fails_the_emit(monkeypatch, source):
    """A certified helper store that moved must fail the EMIT, by name."""
    from . import conductive_kernels

    original = conductive_kernels.kernel_template("step_D_pml_conductive")
    head, tail = original.split("void cond_pml_apply(", 1)
    broken = head + "void cond_pml_apply(" + tail.replace(
        "    f[idx] = value;\n", "    f[idx] = (value);\n", 1)
    assert broken != original
    monkeypatch.setattr(conductive_kernels, "_TEMPLATES",
                        dict(conductive_kernels._TEMPLATES,
                             step_D_pml_conductive=broken))
    with pytest.raises(AssertionError, match="its closing store 0 times"):
        family.conductive_fused_electric_pair_source((True, True, True))


# ---------------------------------------------------------------------------
# 4. THE PREDICATE, ON REAL ENGINE OBJECTS
# ---------------------------------------------------------------------------

def electric_source(grid, component="Ez"):
    return VolumeSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


def build(xp, *, conductivity=0.4, boundaries=("metallic", "metallic", "periodic"),
          symmetry=(), pml=True):
    """One engine on the CONDUCTIVE-PML branch.

    The corpus row's own shape: METALLIC on x and y, periodic on z, an ACTIVE
    absorber, a conductivity on every D component and NO susceptibility.
    """
    rng = numpy.random.default_rng(20260902)
    grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0),
                boundaries=tuple(boundaries),
                symmetry=tuple(Mirror(axis, int(phase)) for axis, phase in symmetry),
                xp=xp, courant=0.5)
    fields = Fields(grid=grid)
    shape = tuple(int(n) for n in grid.shape)
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=shape).astype(numpy.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray(
            (numpy.float32(1.0) / values).astype(numpy.float32))
    fields.set_epsilon_volumes(epsilon, inverse)
    if conductivity is not None:
        fields.set_d_conductivity(xp.asarray(
            numpy.full(shape, float(conductivity), dtype=numpy.float32)))
    if pml:
        fields.enable_pml_storage()
        folded = {"XYZ".index(axis) for axis, _phase in symmetry}
        layer = PML(grid=grid, thickness=tuple((0, 2) if axis in folded else (2, 2)
                                               for axis in range(3)))
    else:
        fields.enable_field_storage()
        layer = PML(grid=grid, thickness=0)
    return fields, grid, layer


def test_a_conductivity_some_chunk_drops_is_admitted_with_its_keep_mask(xp):
    """The corpus row's own shape: a conductivity the layer's chunks keep none of.

    ``TestAdjointSolver.test_damping`` and the route case ``conductive_2d`` both carry
    the conductivity in an interior region, so MEEP's layer chunks step the
    non-conductive forms there. The family admits it and binds the plan's mask.
    """
    fields, grid, layer = build(xp)
    shape = tuple(int(n) for n in grid.shape)
    half = numpy.zeros(shape, dtype=bool)
    half[shape[0] // 2:] = True
    fields.set_d_conductivity(xp.asarray(
        numpy.where(half, numpy.float32(0.4), numpy.float32(0.0)).astype(numpy.float32)))
    covered, why = family.covers_conductive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert covered, why
    keep, masks = family.conductive_keeps(fields, layer)
    assert keep == (False, False, False)
    bound = family.keep_bindings(fields, keep, masks)
    assert all(numpy.asarray(mask).dtype == bool for mask in bound)


def test_the_corpus_shape_is_admitted(xp):
    """``tests:TestAdjointSolver.test_damping``: an active absorber, a conductivity on
    every D component, no susceptibility, METALLIC on x and y, a D deposit."""
    fields, grid, layer = build(xp)
    covered, why = family.covers_conductive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert covered, why


def test_it_partitions_against_the_real_twin_on_the_conductivity(xp):
    """THE DISJOINTNESS MARGIN, and it is load-bearing rather than tidy.

    ``install_fused_pairs`` leaves a seam UNFUSED when two products admit it, so an
    overlap would COST the 79 rows the real twin serves rather than adding one.
    """
    for sigma in (0.4, None):
        fields, grid, layer = build(xp, conductivity=sigma)
        sources = (electric_source(grid),)
        mine = family.covers_conductive_fused_electric_pair(
            fields, layer, grid, sources)[0]
        theirs = real.covers_fused_electric_pair(fields, layer, grid, sources)[0]
        assert mine is (sigma is not None), sigma
        assert theirs is (sigma is None), sigma
        assert not (mine and theirs)


def test_an_inert_absorber_is_refused(xp):
    """The ordinary constitutive half runs the dsigw accumulation and needs a layer."""
    fields, grid, layer = build(xp, pml=False)
    covered, why = family.covers_conductive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert not covered


def test_a_mirror_plane_is_refused_by_name(xp):
    fields, grid, layer = build(xp, boundaries=("periodic",) * 3,
                                symmetry=(("Y", 1),))
    covered, why = family.covers_conductive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert not covered
    assert "mirror plane" in why or "folded" in why


def test_an_undeclared_source_set_is_refused(xp):
    fields, grid, layer = build(xp)
    covered, _why = family.covers_conductive_fused_electric_pair(
        fields, layer, grid, None)
    assert not covered


# ---------------------------------------------------------------------------
# 5. THE WHOLE-VOLUME RESCALE REFUSAL -- what remains of the old blanket clause
# ---------------------------------------------------------------------------

class _NoDepositTable:
    """A scaled electric source that publishes NO deposit table.

    NOT A MOCK OF AN IN-TREE CLASS -- no in-tree electric source is one, which is why
    the clause it triggers is about a duck-typed stranger. The driver's fallback
    applies the condinv scaling as three WHOLE-VOLUME passes for such a source, and
    those rewrite ``-0.0`` to ``+0.0`` at every cell of the component rather than only
    at the deposit; a point repair cannot reconstruct that.
    """

    is_integrated = False
    component = "Ez"


def test_a_source_with_no_deposit_table_is_refused_by_name(xp):
    """The clause that survives the 2026-09-01 lift, and it is a TRUE refusal.

    The blanket refusal of every conductive row was lifted on the driver's sparse
    replay. THIS case still takes the dense branch, so it is still refused -- and the
    reason names the branch rather than the feature, so a reader learns which code
    path is the problem.
    """
    fields, grid, layer = build(xp)
    covered, why = family.covers_conductive_fused_electric_pair(
        fields, layer, grid, (_NoDepositTable(),))
    assert not covered
    assert "WHOLE-VOLUME passes" in why
    assert "_inject_electric_through_conductivity" in why


def test_the_conductive_deposit_moves_only_its_own_cells():
    """THE PREMISE OF THIS WHOLE FAMILY, measured on the shipped deposit.

    This product's cell is servable only because a conductive run's electric deposit
    changes D at the source's own cells and nowhere else. The driver used to scale the
    current by a replay over the injected increment (whole-volume for a source with no
    deposit table); the double-precision deposit puts the ``condinv`` scale inside each
    source's own expression (``sources._deposit``, MEEP step.cpp:299-307) and the
    driver has no pass over the component at all. If that ever changes -- a pass over
    the volume, or a deposit that writes a cell off its table -- this product's
    byte-identity claims are void, so it is MEASURED here on the words: a ``-0.0``
    far from the deposit survives as ``-0.0``, every cell off the table is unchanged,
    and the deposit cells moved.
    """
    from ..sources import GaussianPulsedSource

    fields, grid, _layer = build(numpy)
    assert fields.condinv_for("Dz") is not None, "the build is not conductive"
    source = GaussianPulsedSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                                  size=(0.0, 0.0, 0.0), frequency=1.0, fwidth=0.2,
                                  amplitude=1.0)
    table = numpy.zeros(fields.Dz.shape, dtype=bool)
    table[source._point_ix, source._point_iy, source._point_iz] = True  # noqa: SLF001
    far = (1, 1, 0)
    assert not table[far]
    fields.Dz[far] = numpy.float32(-0.0)
    before = fields.Dz.copy()
    source.inject(fields, source.peak_time)
    after = fields.Dz
    assert after[far].view(numpy.uint32) == numpy.uint32(0x80000000), "the far -0.0 was rewritten"
    assert bool(numpy.all(after[~table].view(numpy.uint32) == before[~table].view(numpy.uint32))), (
        "the deposit wrote a cell off the source's table")
    assert bool(numpy.any(after[table].view(numpy.uint32) != before[table].view(numpy.uint32)))


# ---------------------------------------------------------------------------
# 6. THE REPAIR DECLARATION AND THE CENSUS CELL
# ---------------------------------------------------------------------------

def test_the_declared_path_is_the_one_the_layer_selects(xp):
    assert family.REPAIR_PATHS == (deposit_repair.SPLIT_FIELD_PATH,)
    _fields, _grid, layer = build(xp)
    assert deposit_repair.repair_path_for(layer) == deposit_repair.SPLIT_FIELD_PATH
    assert fused_pairs._repair_paths_of(
        fused_pairs.FUSED_PRODUCTS["cuda_conductive_fused_electric_pair"]) == (
            deposit_repair.SPLIT_FIELD_PATH,)


def test_the_repair_admits_this_cells_seam(xp):
    """The split-field inverse HAS something to invert here: ``f_w_E*`` is allocated
    and the layer is active, which is what separates this cell from its no-absorber
    sibling."""
    fields, _grid, layer = build(xp)
    covered, why = deposit_repair.repairable(fields, "D", pml=layer)
    assert covered, why


def _census_rows():
    rows = {}
    for leg in ("examples.jsonl", "tests.jsonl"):
        path = CENSUS / leg
        # A MISSING CENSUS IS A FAILURE, NOT A SKIP: this suite's cell facts are
        # MEASUREMENTS off that file, and skipping would turn the round's central
        # claim into a silent coverage gap.
        assert path.exists(), (
            f"the census leg {path} has not been cut; this suite reads the cell's "
            f"membership and its deposit denominator off it")
        # THE LEG PREFIX IS THIS SUITE'S, NOT THE CENSUS'S. The census stores a bare
        # `row` ("absorber-1d.py", "TestAbsorber.test_absorber"); the BOARD keys its
        # instances by "<leg>:<row>", which is the name a reader of
        # taxonomy.instances sees and the name CELL_ROWS spells. Prefixing here means
        # the two vocabularies meet in one place instead of a test quietly comparing
        # a bare name against a prefixed one and reporting a moved cell.
        leg_name = leg.split(".", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            rows[f"{leg_name}:{record.get('row')}"] = record
    return rows


def test_the_census_names_the_row_and_both_arms_admit_it():
    rows = _census_rows()
    for name in CELL_ROWS:
        assert name in rows, f"the census does not carry {name}"
        record = rows[name]
        assert record[CELL["curl_arm"]]["step_D"]["covered_modulo_backend"], name
        assert record[CELL["constitutive_arm"]]["update_E"][
            "covered_modulo_backend"], name


def test_the_row_carries_an_electric_deposit_and_an_active_absorber():
    """CARRIES_DEPOSIT_REPAIR IS THE PRODUCT, measured off the census.

    At False this module would compile, pass every arithmetic leg and serve ZERO.
    """
    rows = _census_rows()
    for name in CELL_ROWS:
        configuration = rows[name]["configuration"]
        assert "D" in (configuration.get("source_field_types") or ()), name
        assert configuration["pml_active"], name
        assert configuration["has_conductivity"], name
        assert configuration["n_polarizations"] == 0, name
        assert not configuration["has_symmetry"], name
    assert family.CARRIES_DEPOSIT_REPAIR is True


def test_the_product_column_admits_exactly_the_cell_row():
    rows = _census_rows()
    column = "cuda_conductive_fused_electric_pair"
    assert column in next(iter(rows.values())), (
        "this census predates the product column; the battery edit and the census cut "
        "belong to the same round")
    admitted = {name for name, record in rows.items()
                if (record.get(column) or {}).get("covered_modulo_backend")}
    assert admitted == set(CELL_ROWS), (
        f"admitted {sorted(admitted)}, expected {sorted(CELL_ROWS)}")


# ---------------------------------------------------------------------------
# 7. THE ARGUMENT ORDER -- a silent wrong answer if it ever drifts
# ---------------------------------------------------------------------------

def _signature_parameters(text):
    """The entry point's parameter list, one entry per declared name."""
    signature = text.split('extern "C" __global__ void', 1)[1].split("\n) {\n", 1)[0]
    names = []
    for line in signature.splitlines()[1:]:
        code = line.split("//")[0].strip().rstrip(",")
        if not code:
            continue
        for part in code.split(","):
            token = part.strip().split()[-1].lstrip("*")
            if token:
                names.append(token)
    return names


def test_the_launcher_binds_the_signature_in_order(source):
    """The launcher's argument sequence IS the signature's parameter sequence.

    CUDA binds by POSITION and checks no names, so a launcher assembling its tuple in
    a different order than the signature declares would compile, launch and produce a
    plausible wrong field. ON THIS FAMILY THE HAZARD IS SHARPEST AT THE TWO
    COEFFICIENT GROUPS: they are the same size and the same type, so swapping them is
    a half-cell error in the absorber profile that nothing downstream would name.
    """
    text = source((True, True, True))
    declared = _signature_parameters(text)
    expected = (
        list(family._FIELD_BINDINGS)
        + ["cf0", "cf1", "cf2", "ci0", "ci1", "ci2"]
        + ["fc0", "fc1", "fc2"]
        + ["ck0", "ck1", "ck2"]
        + list(family._ELECTRIC_BINDINGS)
        + ["inv_eps_Ex", "inv_eps_Ey", "inv_eps_Ez"]
        + ["nx", "ny", "nz", "dtdx"]
        + list(family._CURL_TABLE_KEYS)
        + ["kps_x", "kms_half_x", "kps_y", "kms_half_y", "kps_z", "kms_half_z"]
        + ["bc_x", "bc_y", "bc_z", "wall_x", "wall_y", "wall_z"]
        + ["cb"])
    assert declared == expected, (
        f"the signature declares\n  {declared}\nand the launcher assembles\n"
        f"  {expected}\nCUDA binds by position, so a difference here is a silent "
        f"wrong answer rather than an error")
    # THE TWO COEFFICIENT GROUPS ARE THE SAME SIZE, which is exactly why their order
    # is pinned as text: a swap is type-correct and arity-correct.
    assert len(family._CURL_TABLE_KEYS) == len(family._CONSTITUTIVE_TABLE_KEYS) == 6
