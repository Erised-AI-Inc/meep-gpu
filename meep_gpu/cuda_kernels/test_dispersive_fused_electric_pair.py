"""The DISPERSIVE hand-CUDA D/E weld -- the laptop half.

WHAT THIS FILE PINS. The device verdict lives in
``parity/meep_gpu/gate_cuda_dispersive_fused_electric_pair.py``; what a laptop can pin
is the half whose failure mode is a SILENT WRONG ANSWER before any kernel runs:

* THE GHOST CHAIN'S INDEX. The one thing this product does that no other backend's
  dispersive pair does: an imaged ghost's ``D - sum P`` is re-formed from the
  DESTINATION's pole words, not the source thread's registers. Both siblings refuse a
  folded grid on this family and never meet the question; four of the seven corpus
  rows this cell owns are folded, so it is the product.
* THE SPLICE ANCHORS AS WELDS -- the certified dispersive body's opening read, its
  subtraction chain, its ``constitutive_apply`` call and the real twin's ghost product
  are all matched as WHOLE statements, so a moved certified line fails the emit rather
  than emitting a kernel that is quietly not the certified arithmetic.
* THE DEGENERATE ARITY. At zero poles on a component the emitted lines must be
  character-identical to the non-dispersive family's, because ``fields.py:1096-1098``
  returns the D array itself and forms no subtraction at all.
* THE DISJOINTNESS MARGIN, which is one boolean and is load-bearing: two admitters on
  ``step_D`` leave the seam UNFUSED, so an overlap with the real twin would cost the
  79 rows that one serves rather than adding seven.
* ``CARRIES_DEPOSIT_REPAIR`` AS A PER-CELL MEASUREMENT off the census rather than as a
  constant: all seven rows of this cell carry an electric deposit inside the seam, so
  the flag is the whole product and a monkeypatched ``False`` must cost every row.

WHERE THE CERTIFIED TEXT COMES FROM ON A LAPTOP. ``step_curl_kernels`` and
``constitutive_kernels`` import CuPy at module scope, so they cannot be imported on the
merge-bar host -- and a test that skipped for that reason would leave the SPLICE
unchecked everywhere a laptop runs. Their device strings are plain module-level
assignments and are read from the SOURCE by ``ast``, the same reader
``test_fused_electric_pair`` and ``test_offdiag_stencil_welds`` use.
"""

from __future__ import annotations

import ast
import json
import pathlib

import numpy
import pytest

from .. import deposit_repair
from ..dispersion import PolarizationState, Susceptibility
from ..fields import Fields
from ..grid import Grid, Mirror
from ..pml import PML
from ..sources import GaussianEnvelope, VolumeSource
from . import arms, dispersive_kernels, fused_pairs
from . import dispersive_fused_electric_pair as family
from . import fused_electric_pair as real
from .test_fused_pairs import _NumpyWearingCupysName

#: The census this round's numbers are read off. Named so a re-cut that moved the rows
#: is a NAMED failure here rather than a silently different denominator.
CENSUS = (pathlib.Path(__file__).resolve().parents[2] / "parity" / "meep_gpu"
          / "results" / "cuda_predicate_coverage_2026-09-02_dispersive_pair")

#: The board cell this product occupies, and what the census says about it. THE
#: DENOMINATOR IS SEVEN and every one of the seven carries an electric deposit inside
#: the seam, which is why ``CARRIES_DEPOSIT_REPAIR`` is the product rather than a
#: clause of it.
CELL = {
    "seam": "D_to_E",
    "curl_arm": "cuda_curl/PML",
    "constitutive_arm": "cuda_dispersive/dispersive",
    "rows": 7,
    "rows_with_an_electric_deposit": 7,
    "rows_folded": 4,
}

#: The seven corpus rows, by name. Spelled out so a census whose cell moved is a named
#: failure listing WHICH row moved rather than a changed integer.
CELL_ROWS = (
    "examples:stochastic_emitter.py",
    "examples:stochastic_emitter_line.py",
    "examples:stochastic_emitter_reciprocity.py",
    "tests:TestLoadDump.test_load_dump_chunk_layout_file_2d",
    "tests:TestLoadDump.test_load_dump_chunk_layout_sim_2d",
    "tests:TestLoadDump.test_load_dump_structure_2d",
    "tests:TestLoadDump.test_load_dump_structure_sharded_2d",
)


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
def certified(monkeypatch):
    """Both emitters, with every certified half supplied from source.

    ``real`` is patched as well as ``family``: this family calls the real twin's own
    emitters for the curl body, the pre-clear copy, the wall carry and the ghost
    blocks, so a fixture that patched only its own module would refuse inside the
    first delegated call.
    """
    for module in (family, real):
        for name in ("step_curl_kernels", "constitutive_kernels"):
            if getattr(module, name, None) is None:
                monkeypatch.setattr(module, name, _CertifiedText(name), raising=False)
    return family.dispersive_fused_electric_pair_source


# ---------------------------------------------------------------------------
# 1. THE GHOST CHAIN -- what neither sibling backend does
# ---------------------------------------------------------------------------

def test_every_ghost_subtracts_the_destinations_poles_not_the_sources(certified):
    """THE FINDING. ``update_E`` is element-wise over the WHOLE volume
    (fields.py:1096-1105), so an imaged ghost subtracts THAT CELL'S ``P``. Re-using the
    source thread's registers is the obvious port and is a different field on every
    imaged plane -- measured by
    ``parity/meep_gpu/probe_cuda_dispersive_electric_pair.py``'s ``poles_at_source``
    mutation, which diverges on 6 of its 8 configurations.
    """
    source = certified((2, 2, 2))
    body = source.split("\n) {", 1)[1]
    subtractions = [line.strip() for line in body.splitlines()
                    if "_p = " in line and "- P_E" in line]
    assert subtractions, "no ghost chain was emitted at arity (2, 2, 2)"
    for line in subtractions:
        # `gx_ny_p = gx_ny_p - P_Ex_0[gx_ny_i];` -- the pole index and the tag index
        # are the SAME symbol, and that symbol is the destination's.
        tag = line.split("_p = ", 1)[0]
        assert f"[{tag}_i]" in line, (
            f"{line!r} does not read its pole at the destination index {tag}_i")
        assert "[idx]" not in line, (
            f"{line!r} reads a pole at this thread's own idx inside a ghost block")


def test_every_ghost_product_multiplies_the_destinations_inverse_epsilon(certified):
    source = certified((1, 1, 1))
    body = source.split("\n) {", 1)[1]
    products = [line.strip() for line in body.splitlines()
                if "_s = " in line and "inv_eps_E" in line and line.strip().startswith(
                    "float g")]
    assert len(products) == 21, (
        f"{len(products)} ghost products, not 21: three components at seven "
        f"destinations each (carried_destinations' closure)")
    for line in products:
        tag = line.split("float ", 1)[1].split("_s = ", 1)[0]
        assert f"inv_eps_E{tag[1]}[{tag}_i]" in line, (
            f"{line!r} does not read inverse epsilon at the destination index")


def test_the_ghost_chain_sits_between_the_imaged_value_and_the_product(certified):
    """ORDER: the imaged displacement, then the chain, then the inverse-epsilon
    multiply -- the array path's own order, with the subtraction never crossing the
    multiply it feeds."""
    source = certified((2, 0, 1))
    block = source.split("int gx_ny_i", 1)[1].split("constitutive_apply", 1)[0]
    value = block.index("float gx_ny_p = gx_ny_v;")
    first = block.index("gx_ny_p = gx_ny_p - P_Ex_0[gx_ny_i];")
    second = block.index("gx_ny_p = gx_ny_p - P_Ex_1[gx_ny_i];")
    product = block.index("float gx_ny_s = gx_ny_p * inv_eps_Ex[gx_ny_i];")
    assert value < first < second < product


def test_a_zero_arity_component_forms_no_chain_at_its_ghosts(certified):
    """``displacement_minus_polarization`` returns the D array ITSELF when nothing
    drives the component, so the emitter must form NO subtraction -- which is what
    makes the degenerate case reduce to the certified non-dispersive text by
    construction rather than by rounding."""
    source = certified((2, 0, 1))
    body = source.split("\n) {", 1)[1]
    own = body.split("if (own_y) {", 1)[1].split("if (own_z) {", 1)[0]
    assert "P_Ey" not in own
    assert "_p = " not in own
    assert "float gy_nx_s = gy_nx_v * inv_eps_Ey[gy_nx_i];" in own


def test_the_zero_arity_ghost_blocks_are_the_real_twins_own_text(certified):
    """At arity zero on every component the ghost blocks must be the REAL twin's
    verbatim -- the strongest form of "nothing was inserted"."""
    for target in range(3):
        assert (family.dispersive_fill_carry_blocks(target, 0)
                == real.fill_carry_blocks(target))


def test_the_pole_chain_at_this_threads_own_cell_reads_idx(certified):
    """The OWN cell's chain is the certified emitter's own text, unmoved: it reads
    ``[idx]`` and sits above the ownership guard, where the certified body puts it."""
    source = certified((2, 2, 2))
    body = source.split("\n) {", 1)[1]
    head = body.split("if (own_x) {", 1)[0]
    assert "float s_x = d_x;" in head
    assert "s_x = s_x - P_Ex_0[idx];" in head
    assert "s_x = s_x - P_Ex_1[idx];" in head
    assert "float src_x = s_x * inv_eps_Ex[idx];" in head


# ---------------------------------------------------------------------------
# 2. THE ORDER AGAINST THE METALLIC CLEANUP
# ---------------------------------------------------------------------------

def test_the_own_cell_chain_consumes_the_post_clear_register(certified):
    """BOTH SIBLING BACKENDS PUT THE CHAIN AFTER THE CLEAR and they agree:
    ``metal_kernels/fused_dispersive_pair`` emits ``zero_metal_mask`` then
    ``float source = v{axis};``, and ``triton_kernels/dispersive_fused_pair`` applies
    its ZM block to ``v0/v1/v2`` then opens with ``s0 = v0``. The probe's
    ``subtract_before_clear`` mutation shows the agreement is load-bearing rather than
    incidental -- it diverges on 3 of 8 configurations.
    """
    source = certified((2, 2, 2))
    body = source.split("\n) {", 1)[1]
    clear = body.index("if (own_x && clr_x) { d_x = 0.0f; Dx[idx] = d_x; }")
    chain = body.index("float s_x = d_x;")
    assert clear < chain, (
        "the D - sum P chain consumes d_x BEFORE zero_metal_D clears it; the wall "
        "plane would carry a pole-subtracted nonzero displacement")


def test_the_near_ghost_reads_the_pre_clear_register_and_the_far_one_does_not(certified):
    """The driver's own order between :3309 and :3310, inherited from the real twin
    and re-asserted here because the chain now sits between the two."""
    source = certified((1, 1, 1))
    near_only = source.split("int gx_ny_i", 1)[1].split("}", 1)[0]
    assert "float gx_ny_v = phase_y * pre_x;" in near_only
    far_only = source.split("int gx_x_i", 1)[1].split("}", 1)[0]
    assert "float gx_x_v = d_x;" in far_only
    assert "pre_x" not in far_only


def test_all_three_sources_are_formed_before_any_store(certified):
    """THE CERTIFIED BODY'S OWN REORDERING, kept. Neither sibling has it -- Metal
    dispatches one component per launch and Triton writes three sequential E blocks --
    and adopting either interleave would silently retype the lifted body."""
    source = certified((2, 2, 2))
    body = source.split("\n) {", 1)[1]
    last_source = max(body.index(f"float src_{letter} = ") for letter in "xyz")
    first_store = body.index("if (own_x) {")
    assert last_source < first_store


# ---------------------------------------------------------------------------
# 3. THE LIFT AND ITS ANCHORS
# ---------------------------------------------------------------------------

def test_the_lift_is_the_real_twins_edits_plus_exactly_three():
    """``LIFT_EDITS`` is DATA so a gate can assert it. The first rows ARE the real
    twin's object -- this family calls its emitters -- so a tenth edit appearing there
    reaches this list without a second record."""
    assert family.LIFT_EDITS[:len(real.LIFT_EDITS)] == real.LIFT_EDITS
    assert len(family.LIFT_EDITS) == len(real.LIFT_EDITS) + 3
    assert all(set(edit) == {"line", "became", "why"} for edit in family.LIFT_EDITS)


def test_the_shared_flux_density_is_bound_exactly_once(certified):
    """THE ALIASING HAZARD. Two ``__restrict__`` pointers to one allocation is UB that
    NVRTC miscompiles without a diagnostic, so the constitutive half must have no D
    source pointers at all -- at EVERY arity, including the degenerate one whose line
    is the non-dispersive family's."""
    for arity in ((0, 0, 0), (2, 0, 1), (6, 6, 6)):
        source = certified(arity)
        signature = source.split(f"{family.KERNEL_NAME}(", 1)[1].split("\n) {", 1)[0]
        for target in ("Dx", "Dy", "Dz"):
            assert (signature.count(f" {target},")
                    + signature.count(f" {target}\n")) == 1, (
                f"{target} is bound more than once at arity {arity}")
        body = source.split("\n) {", 1)[1]
        for target in ("Dx", "Dy", "Dz"):
            assert f"{target}[idx] *" not in body
            assert f"float s_{target[-1]} = {target}[idx];" not in body


def test_the_pole_pointers_are_restrict_and_the_inverse_epsilons_are_not(certified):
    """Lifted from ``dispersive_kernels.dispersive_source`` rather than decided here:
    two susceptibilities never share a P buffer (dispersion.py:645-647) while an
    isotropic run hands ONE inverse-epsilon pointer three times (fields.py:1321-1326).
    """
    source = certified((2, 1, 0))
    signature = source.split(f"{family.KERNEL_NAME}(", 1)[1].split("\n) {", 1)[0]
    for name in ("P_Ex_0", "P_Ex_1", "P_Ey_0"):
        assert f"const float* __restrict__ {name}," in signature
    assert "P_Ez_0" not in signature
    for name in ("inv_eps_Ex", "inv_eps_Ey", "inv_eps_Ez"):
        assert f"const float* {name}" in signature
        assert f"__restrict__ {name}" not in signature


def test_the_pole_block_count_is_the_arity_sum(certified):
    for arity in ((0, 0, 0), (1, 1, 1), (2, 0, 3), (6, 6, 6)):
        signature = certified(arity).split(f"{family.KERNEL_NAME}(", 1)[1].split(
            "\n) {", 1)[0]
        assert signature.count("__restrict__ P_E") == sum(arity)


@pytest.mark.parametrize("arity", [(0, 0, 0), (1, 1, 1), (2, 2, 2), (2, 0, 3),
                                   (5, 5, 5), (6, 6, 6)])
def test_the_swept_arities_all_emit_and_are_pure_ascii(certified, arity):
    """Every arity ``dispersive_kernels`` sweeps must emit here too: a triple this
    family emitted and that family never swept would be a body no device has run.

    PURE ASCII IS A COMPILE REQUIREMENT: NVRTC's source file is written through the
    interpreter's locale encoding, so a non-ASCII comment fails at first launch rather
    than at import."""
    assert arity in family.swept_arities()
    source = certified(arity)
    source.encode("ascii")
    assert source.count(f"__global__ void {family.KERNEL_NAME}(") == 1


def test_device_sources_keys_every_swept_arity(certified):
    sources = family.device_sources()
    assert len(sources) == len(family.swept_arities())
    assert len(set(sources.values())) == len(sources), (
        "two swept arities emitted the SAME body; a digest over them would pin one "
        "string and let the other drift")


@pytest.mark.parametrize("anchor", [
    "    float s_x = Dx[idx];",
    "    float src_y = Dy[idx] * inv_eps_Ey[idx];",
    "    constitutive_apply(Ez, f_w_Ez, idx, src_z, kps_z[k], kms_z[k]);",
])
def test_a_moved_certified_anchor_fails_the_emit_rather_than_emitting(
        monkeypatch, certified, anchor):
    """THE SPLICE ANCHORS ARE WELDS. A certified line that moved must fail the emit,
    not be silently splice-around -- the failure mode otherwise is a kernel that
    compiles and is quietly not the certified arithmetic."""
    original = dispersive_kernels.dispersive_source

    def mutated(arm, counts):
        text = original(arm, counts)
        return text.replace(anchor, anchor.replace("    ", "      ", 1))

    monkeypatch.setattr(dispersive_kernels, "dispersive_source", mutated)
    with pytest.raises(AssertionError):
        # (2, 0, 1) reaches all three anchors: a driven x, a degenerate y, a driven z.
        family.dispersive_fused_electric_pair_source((2, 0, 1))


def test_a_reworded_ghost_product_fails_the_carry_splice(monkeypatch):
    """The real twin's ghost product is matched as a WHOLE statement; a partial match
    would attach the pole chain to a line this module has not read."""
    original = real.fill_carry_blocks

    def mutated(target, indent="        "):
        return [line.replace("_v * inv_eps_", "_v*inv_eps_")
                for line in original(target, indent)]

    monkeypatch.setattr(real, "fill_carry_blocks", mutated)
    with pytest.raises(AssertionError, match="this family splices into"):
        family.dispersive_fill_carry_blocks(0, 2)


def test_a_vanished_ghost_product_fails_the_carry_splice(monkeypatch):
    """The other half: if the real twin stopped emitting an inverse-epsilon product at
    every ghost, this family would have nowhere to splice and must SAY so rather than
    return a block list with no chain in it."""
    original = real.fill_carry_blocks

    def mutated(target, indent="        "):
        return [line for line in original(target, indent)
                if "inv_eps_" not in line]

    monkeypatch.setattr(real, "fill_carry_blocks", mutated)
    with pytest.raises(AssertionError, match="nowhere to splice"):
        family.dispersive_fill_carry_blocks(0, 2)


def test_the_emitter_refuses_by_name_where_the_certified_text_is_unreachable():
    """THE SPLICE IS THE LIFT, so a missing half is not a degraded emit. The refusal
    names the modules one frame from the caller, and the predicate still answers --
    which is why those imports are defensive in the first place."""
    saved = (family.dispersive_kernels, family.constitutive_kernels)
    try:
        family.dispersive_kernels = None
        family.constitutive_kernels = None
        with pytest.raises(RuntimeError) as raised:
            family.dispersive_fused_electric_pair_source((1, 1, 1))
        assert "dispersive_kernels" in str(raised.value)
        assert "constitutive_kernels" in str(raised.value)
        assert family.covers_dispersive_fused_electric_pair(
            None, None, None, ())[0] is False
    finally:
        family.dispersive_kernels, family.constitutive_kernels = saved


def test_the_prelude_is_the_dispersive_familys_constitutive_apply(certified):
    """``dispersive_kernels`` keeps its OWN copy of ``constitutive_apply`` so that a
    bit-identity probe rewriting one family's string does not silently mutate the
    other's. A product that borrowed the wrong copy would sit outside every mutation
    its own gate arms."""
    source = certified((1, 1, 1))
    assert source.count("void constitutive_apply(") == 1
    assert dispersive_kernels._PML_ACCUMULATION_NOTE.split(
        "__device__", 1)[1] in source
    assert "float pml_apply_reg(" in source


# ---------------------------------------------------------------------------
# 4. THE PREDICATE, ON REAL ENGINE OBJECTS
# ---------------------------------------------------------------------------

def electric_source(grid, component="Ez"):
    return VolumeSource(grid=grid, component=component, center=(0.0, 0.0, 0.0),
                        size=(0.0, 0.0, 0.0),
                        envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                        amplitude=1.0)


def build(xp, *, poles=2, boundaries=("periodic",) * 3, symmetry=(), pml=True):
    """One engine at one shape and one pole arity, with LIVE poles.

    The two corpus shapes this cell owns are the defaults' two callers: all-periodic
    with an active layer (the three ``stochastic_emitter`` rows) and metallic x with a
    mirror-folded y (the four ``TestLoadDump.*_2d`` rows).
    """
    rng = numpy.random.default_rng(20260902)
    grid = Grid(resolution=1.0, cell_size=(9.0, 10.0, 11.0),
                boundaries=tuple(boundaries),
                symmetry=tuple(Mirror(axis, int(phase)) for axis, phase in symmetry),
                xp=xp, courant=0.5)
    fields = Fields(grid=grid)
    layer = None
    if pml:
        fields.enable_pml_storage()
        folded = {"XYZ".index(axis) for axis, _phase in symmetry}
        layer = PML(grid=grid, thickness=tuple((0, 2) if axis in folded else (2, 2)
                                               for axis in range(3)))
    else:
        fields.enable_field_storage()
    shape = tuple(int(n) for n in grid.shape)
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=shape).astype(numpy.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray(
            (numpy.float32(1.0) / values).astype(numpy.float32))
    fields.set_epsilon_volumes(epsilon, inverse)
    kinds = ("lorentzian", "drude")
    for index in range(poles):
        term = Susceptibility(frequency=0.20 + 0.03 * index, gamma=0.008,
                              kind=kinds[index % 2])
        sigma = {name: 0.35 + 0.04 * index for name in ("Ex", "Ey", "Ez")}
        fields.polarizations.append(
            PolarizationState(term, sigma, grid, fields._field_dtype()))
    # SEEDED NON-ZERO: zero is a fixed point of ``s = s - P``, so a zero fixture would
    # make every pole question a null.
    for state in fields.polarizations:
        for store in (state.P, state.P_prev):
            for _component, array in store.items():
                array[...] = xp.asarray(
                    rng.uniform(-0.6, 0.6, size=shape).astype(numpy.float32))
    return fields, grid, layer



def test_the_all_periodic_corpus_shape_is_admitted(xp):
    """``examples:stochastic_emitter*.py`` -- an active layer, all periodic, six
    susceptibilities driving all three components, electric deposits in the seam."""
    fields, grid, layer = build(xp, poles=6)
    covered, why = family.covers_dispersive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert covered, why


def test_the_folded_and_walled_corpus_shape_is_admitted(xp):
    """``tests:TestLoadDump.*_2d`` -- metallic x AND y with y ALSO mirror-folded, so
    ``zero_metal_D`` walls x alone and the near fill runs on y. BOTH SIBLING BACKENDS
    REFUSE THIS SHAPE on their dispersive pair, which is why it is named here."""
    fields, grid, layer = build(
        xp, poles=5, boundaries=("metallic", "metallic", "periodic"),
        symmetry=(("Y", 1),))
    covered, why = family.covers_dispersive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert covered, why
    assert grid.is_mirrored(1) and grid.is_metallic(1)
    from .in_seam_coverage import zero_metal_axes  # noqa: PLC0415
    assert tuple(bool(v) for v in zero_metal_axes(grid)) == (True, False, False), (
        "a folded metallic axis carries no stored wall (stepping.py:2284-2286); if it "
        "did, the near ghost would land in a plane the clear also owns")


def test_the_two_electric_products_partition_on_one_boolean(xp):
    """THE DISJOINTNESS MARGIN, and it is load-bearing rather than tidy:
    ``install_fused_pairs`` leaves a seam UNFUSED when two products admit it, so an
    overlap would COST the 79 rows the real twin serves rather than adding seven."""
    for poles in (0, 1, 2, 6):
        fields, grid, layer = build(xp, poles=poles)
        sources = (electric_source(grid),)
        mine = family.covers_dispersive_fused_electric_pair(
            fields, layer, grid, sources)[0]
        theirs = real.covers_fused_electric_pair(fields, layer, grid, sources)[0]
        assert mine is (poles > 0), poles
        assert theirs is (poles == 0), poles
        assert not (mine and theirs)


def test_no_absorber_is_refused_to_the_no_pml_arm(xp):
    fields, grid, layer = build(xp, poles=2, pml=False)
    covered, why = family.covers_dispersive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert not covered
    assert "no active PML layer" in why


def test_an_undeclared_source_set_is_refused(xp):
    """IGNORANCE IS NEVER AN EMPTY SET: ``Fields`` does not hold the source list, so a
    predicate that inferred "no sources" from not being told would be exactly the
    over-covering the seam clause exists to prevent."""
    fields, grid, layer = build(xp, poles=2)
    covered, why = family.covers_dispersive_fused_electric_pair(
        fields, layer, grid, None)
    assert not covered
    assert "was not declared" in why


def test_a_conductive_engine_is_refused_on_the_injection_route(xp):
    """``_inject_electric_through_conductivity`` (driver.py:3305) rescales the
    increment by ``condinv``; the deposit repair has no verdict on that route."""
    fields, grid, layer = build(xp, poles=2)

    class _Conductive:
        has_conductivity = True

        def __getattr__(self, name):
            return getattr(fields, name)

    covered, why = family.covers_dispersive_fused_electric_pair(
        _Conductive(), layer, grid, (electric_source(grid),))
    assert not covered
    assert "condinv" in why or "conductivity" in why


def test_a_pole_count_past_the_cap_is_refused_by_name(xp):
    """``POLE_COUNT_CAP`` is MEASURED, not chosen: ``gate_cuda_dispersive`` swept
    0..CAP and a longer chain has never been run on a device."""
    fields, grid, layer = build(xp, poles=dispersive_kernels.POLE_COUNT_CAP + 1)
    covered, why = family.covers_dispersive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert not covered
    assert "POLE_COUNT_CAP" in why


def test_the_proxy_hides_only_the_polarizations(xp):
    """``_WithoutPolarizations`` exists so the SEAM clauses can be delegated whole. It
    must not hide anything else: every other clause has to read the live run."""
    fields, _grid, _layer = build(xp, poles=3)
    proxy = family._WithoutPolarizations(fields)
    assert proxy.polarizations == ()
    assert fields.polarizations
    for name in ("Dx", "Ex", "f_w_Ex", "grid", "has_conductivity", "stores_E"):
        assert getattr(proxy, name) is getattr(fields, name)


# ---------------------------------------------------------------------------
# 5. THE DEPOSIT REPAIR, AS A PER-CELL MEASUREMENT
# ---------------------------------------------------------------------------

def test_the_flag_is_declared_and_reaches_the_shared_clause():
    """A product that declared ``CARRIES_DEPOSIT_REPAIR`` without the two slots would
    compute the constitutive half against a pre-injection field and report success.
    The flag and the wiring change together or not at all."""
    assert family.CARRIES_DEPOSIT_REPAIR is True
    assert family.FAMILY in fused_pairs.FUSED_PAIR_ARMS
    assert family.FAMILY in fused_pairs.FUSED_PRODUCTS
    assert fused_pairs.FUSED_PRODUCTS[family.FAMILY]["curl_slot"] == "step_D"
    assert fused_pairs.FUSED_PAIR_SEAMS["step_D"] == ("update_E", "D")
    source = pathlib.Path(real.__file__).read_text(encoding="utf-8")
    assert "carries_repair=CARRIES_DEPOSIT_REPAIR" in source, (
        "the delegated seam clause no longer passes the flag; this family's own "
        "declaration would then decide nothing")


def test_the_flag_is_worth_every_row_of_the_cell(xp):
    """THE MEASUREMENT, not the constant. Every row of this cell carries an electric
    deposit inside the seam, so the same configuration asked twice -- once with the
    flag held at False -- must flip from admitted to refused BY NAME."""
    fields, grid, layer = build(xp, poles=5,
                                boundaries=("metallic", "metallic", "periodic"),
                                symmetry=(("Y", 1),))
    sources = (electric_source(grid),)
    assert family.covers_dispersive_fused_electric_pair(
        fields, layer, grid, sources)[0]

    saved = real.CARRIES_DEPOSIT_REPAIR
    try:
        real.CARRIES_DEPOSIT_REPAIR = False
        covered, why = family.covers_dispersive_fused_electric_pair(
            fields, layer, grid, sources)
        assert not covered
        assert "injects it BETWEEN step_D and update_E" in why
    finally:
        real.CARRIES_DEPOSIT_REPAIR = saved


def test_the_repair_admits_a_dispersive_electric_seam(xp):
    """``deposit_repair.repairable`` does not refuse a pole -- it refuses only an
    off-diagonal chi1inv and an instantaneous chi2/chi3 -- and NOT REFUSED IS NOT
    MEASURED. What makes the admission sound here is the driver's order:
    ``update_P`` runs AFTER ``update_E`` (driver.py:3315), so the polarization arrays
    the repair reads are the ones the launch consumed, unchanged."""
    fields, grid, layer = build(xp, poles=4)
    ok, why = deposit_repair.repairable(fields, "D", layer)
    assert ok, why
    assert fields.polarizations, "the fixture must carry a live pole or this is vacuous"


# ---------------------------------------------------------------------------
# 6. THE CELL, READ OFF THE CENSUS
# ---------------------------------------------------------------------------

def _census_rows():
    """Every row of the pinned census, keyed ``leg:row``.

    A MISSING CENSUS IS A FAILURE, NOT A SKIP. The census is a checked-in artifact
    under ``parity/meep_gpu/results/``, and skipping when it is absent would let this
    product's whole denominator disappear silently -- which is the one thing the cell
    numbers below exist to prevent.
    """
    rows = {}
    for leg in ("examples", "tests", "tests_param"):
        path = CENSUS / f"{leg}.jsonl"
        assert path.exists(), (
            f"the census {CENSUS.name} has no {leg} leg; this product's cell numbers "
            f"are read off it and cannot be established without it")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["row"] if ":" in record["row"]
                   else f"{leg}:{record['row']}")
            rows[key] = record
    return rows


def test_the_census_names_the_seven_rows_and_this_product_admits_them_all():
    """THE DENOMINATOR IS SEVEN, and it is named row by row so a census whose cell
    moved fails with WHICH row moved rather than with a changed integer.

    ``covered_modulo_backend`` is the reading: the census is cut on a backend-free
    host, so every predicate's raw answer is "backend is not CuPy" and the factored
    verdict is what says whether anything else refused.
    """
    rows = _census_rows()
    missing = [row for row in CELL_ROWS if row not in rows]
    assert not missing, f"the census does not carry {missing}"
    served, refused = [], {}
    for name in CELL_ROWS:
        verdict = rows[name].get(family.FAMILY)
        assert verdict is not None, (
            f"the census carries no {family.FAMILY} column; the battery column and "
            f"the census were cut out of step")
        if verdict.get("covered_modulo_backend"):
            served.append(name)
        else:
            refused[name] = verdict.get("first_refusal")
    assert len(served) == CELL["rows"], (
        f"{len(served)} of {CELL['rows']} rows served; refused: {refused}")


def test_every_row_of_the_cell_carries_an_electric_deposit():
    """Which is why ``CARRIES_DEPOSIT_REPAIR`` is the whole product: at False this
    module would compile, gate green on every arithmetic leg and serve ZERO."""
    rows = _census_rows()
    with_deposit = [name for name in CELL_ROWS
                    if "D" in (rows[name]["configuration"].get("source_field_types")
                               or ())]
    assert len(with_deposit) == CELL["rows_with_an_electric_deposit"]


def test_four_of_the_seven_rows_are_folded():
    """The fact that separates this product from BOTH sibling backends' dispersive
    pairs, which refuse a mirror plane by name and would serve three of seven."""
    rows = _census_rows()
    folded = [name for name in CELL_ROWS
              if any(rows[name]["configuration"].get("mirrored") or ())]
    assert len(folded) == CELL["rows_folded"], folded


def test_the_real_twin_still_refuses_every_row_of_this_cell():
    """The other half of the partition, measured rather than argued: if the real twin
    admitted one of these rows the seam would go UNFUSED on it and both products would
    serve nothing there."""
    rows = _census_rows()
    for name in CELL_ROWS:
        verdict = rows[name].get("cuda_fused_electric_pair")
        assert not verdict.get("covered_modulo_backend"), name
        assert "dispersion" in str(verdict.get("first_refusal")), name


def test_the_installer_gives_the_span_to_the_three_slot_weld(xp):
    """SLOT ARBITRATION, and it is a fact about the COMPOSER not this file.

    THE HISTORY IS THE POINT OF THE ASSERTION. This product's predicate ADMITS every
    row of its cell -- the tests above measure that and its gate certifies the
    arithmetic -- and it has never been about to install here, for a reason that
    changed shape once:

    * 2026-09-02, FIRST READING: it owns ``step_D`` AND ``update_E``,
      ``cuda_fused_polarization_pair`` owns ``update_E`` AND ``update_P``, and all
      SEVEN rows of this cell are exactly the seven that pair serves at E->P.
      Installing here was +7 at D_to_E and -7 at E_to_P -- NET ZERO, paid for by
      displacing a released product -- so ``fused_pairs._later_seam_claimant`` left
      the slot where it was.
    * 2026-09-02, SECOND READING, and strictly better: the THREE-SLOT weld
      ``cuda_three_slot_dispersive_weld`` now takes ``step_D``, ``update_E`` AND
      ``update_P``, so BOTH seams are served and there is no trade to arbitrate.
      This product is refused by ``_superseded_by_a_longer_span``, whose reason names
      the containing span rather than a trade.

    WHAT IS PINNED IS THE INVARIANT BOTH READINGS SHARE and it is stronger now, not
    weaker: this product never silently displaces the later seam's product, the
    refusal is BY NAME, and its own predicate still admits the run.
    """
    fields, grid, layer = build(xp, poles=6)
    plan = arms.plan_step(fields, layer, grid, fuse=True,
                          sources=(electric_source(grid),))
    assert plan.selected["step_D"] == plan.selected["update_E"] == \
        plan.selected["update_P"] == "three-slot dispersive weld", (
            "the three-slot weld serves both seams; nothing here is traded")
    refusal = plan.reasons["fused_pair_cuda_dispersive_fused_electric_pair"][0]
    assert "strictly contains" in refusal and "slot arbitration" in refusal
    covered, _why = family.covers_dispersive_fused_electric_pair(
        fields, layer, grid, (electric_source(grid),))
    assert covered, ("the PREDICATE must still admit it -- the refusal is the "
                     "installer's arbitration, not a verdict about the weld")
