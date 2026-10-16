"""Host contracts for Metal's real conductive split-field PML curl family.

What a host suite can hold without a device: the emitted forms (MEEP's four conductive
forms per structure chunk, keyed on the chunk record and never on coefficient identity,
with the ``f_cond`` increment formed first), the auxiliary guards, the lifted stencil,
the record binding, the per-chunk conductivity presence words (and their refusal where
the record cannot carry them), and the plan's wiring.
Byte identity against the array path is the device gate's claim
(``parity/meep_gpu/gate_metal_conductive_pml.py``).
"""

from __future__ import annotations

import inspect
from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.chunk_map import ChunkLayout, ChunkLayoutRefusal, chunk_bounds_record
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, chunk_bounds, shaders
from meep_gpu.metal_kernels import conductive_pml as family
from meep_gpu.metal_kernels.device import MAX_BUFFER_BINDINGS
from meep_gpu.pml import PML


class HostResidency:
    #: The record tensor a plan uploads goes to this device; a host run never reaches MPS.
    device = "cpu"

    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        previous = self.arrays.get(name)
        if previous is not None and previous is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def build(*, sides=("B", "D"), seed=63):
    grid = Grid(resolution=8.0, cell_size=(1.25, 1.0, 0.75), courant=0.35,
                boundaries=("metallic", "periodic", "metallic"), xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    pml = PML(grid=grid, thickness=1)
    fields.enable_pml_storage()
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz", "fu_Dx",
                 "fu_Dy", "fu_Dz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
    sigma = np.linspace(np.float32(0.06), np.float32(0.22), grid.shape[0],
                        dtype=np.float32)[:, None, None]
    sigma = np.broadcast_to(sigma, grid.shape).copy()
    if "B" in sides:
        fields.set_b_conductivity(sigma)
    if "D" in sides:
        fields.set_d_conductivity(sigma)
    for name in ("f_cond_Bx", "f_cond_By", "f_cond_Bz", "f_cond_Dx",
                 "f_cond_Dy", "f_cond_Dz"):
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
    return fields, pml


def record_of(fields, pml):
    """The one-rank record: the division the array path steps a bare ``Fields`` with."""
    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(family, "_metal_backend_reasons", lambda grid: [])


def _reference_function(reference, actual, pml, sub_step, target_names):
    def run(*args):
        targets, auxiliary, history = args[:3], args[3:6], args[6:9]
        for index, name in enumerate(target_names):
            assert targets[index] is getattr(actual, name)
            assert auxiliary[index] is getattr(actual, "fu_" + name)
            if actual.condfac_for(name) is None:
                assert history[index] is targets[index]
            else:
                assert history[index] is getattr(actual, "f_cond_" + name)
        getattr(stepping, sub_step)(reference, pml)
        for target, name in zip(targets, target_names):
            target[...] = getattr(reference, name)
        for item, name in zip(auxiliary, target_names):
            item[...] = getattr(reference, "fu_" + name)
        for item, name in zip(history, target_names):
            if reference.condfac_for(name) is not None:
                item[...] = getattr(reference, "f_cond_" + name)
    return run


# ---------------------------------------------------------------------------
# The emitted forms
# ---------------------------------------------------------------------------

def test_the_four_conductive_forms_are_keyed_on_chunk_membership():
    """MEEP's four forms per structure chunk, selected on the record, never on ``!= 1``."""
    for backward, tag in ((False, "h"), (True, "i")):
        source = family.conductive_pml_curl_source(
            (0, 1, 0), backward, (True, False, True))
        assert "!= 1.0f" not in source
        assert f"const bool ds0 = in_{tag}_y, du0 = in_{tag}_z;" in source
        assert f"const bool ds2 = in_{tag}_x, du2 = in_{tag}_y;" in source
        assert "if (ds0 && du0) {" in source
        assert "} else if (du0) {" in source
        assert "} else if (ds0) {" in source
        assert "f0[ii] = ((f0_previous * cf0[ii]) - curl0) * ci0[ii];" in source
        # A lossless target of the same sub-step takes the base curl's two forms,
        # keyed on its dsigu membership alone.
        assert f"const bool du1 = in_{tag}_x;" in source
        assert "const bool ds1" not in source


def test_the_f_cond_increment_is_formed_before_it_is_added():
    """MEEP's grouping (step_generic.cpp:167, :209): ``(fcnd' - fcnd)`` first."""
    source = family.conductive_pml_curl_source((0, 0, 0), False, (True, True, True))
    assert source.count("float c0_step = c0_new - c0_previous;") == 2
    assert "float u0_new = ((u0_previous * km_y) + c0_step) * si_y;" in source
    assert "float f0_new = ((f0_previous * km_y) + c0_step) * si_y;" in source
    # The package's previous grouping adds the two words one at a time.
    assert "+ c0_new) - c0_previous)" not in source


def test_each_auxiliary_is_touched_only_on_its_forms_cells():
    """The auxiliary contract: ``fu`` on dsigu cells, ``f_cond`` on dsig cells."""
    source = family.conductive_pml_curl_source((0, 0, 0), True, (True, False, True))
    body = source.split("float km_z = kmz[k], si_z = sinvz[k];", 1)[1]
    branches = body.split("const bool ds2", 1)[0]
    both, rest = branches.split("} else if (du0) {", 1)
    dsigu_only, rest = rest.split("} else if (ds0) {", 1)
    dsig_only, neither = rest.split("} else {", 1)
    assert "c0[ii]" in both and "u0[ii]" in both
    assert "c0[" not in dsigu_only and "u0[ii]" in dsigu_only
    assert "u0[" not in dsig_only and "c0[ii]" in dsig_only
    assert "u0[" not in neither and "c0[" not in neither
    # The lossless target reads and writes fu only behind its dsigu flag.
    assert "float p1 = du1 ? u1[ii] : f1[ii];" in source
    assert "if (du1) { u1[ii] = n1; }" in source
    assert source.count("u1[ii]") == 2


def test_the_stencil_is_the_certified_curls_own_text():
    """Lifted from :func:`.shaders.curl_source`, never re-spelled."""
    stencil = family.certified_stencil()
    assert stencil.count("float curl") == 3
    assert "dtdx * (((c_y - c) + b) - b_z)" in stencil
    assert "dtdx * ((c_y - c) + (b - b_z))" not in stencil
    for backward in (False, True):
        for codes in ((0, 0, 0), (1, 1, 1)):
            assert stencil in shaders.curl_source(codes, backward)
            assert stencil in family.conductive_pml_curl_source(
                codes, backward, (True, True, True))


def test_the_record_and_the_presence_words_bind_last_under_the_ceiling():
    source = family.conductive_pml_curl_source((0, 0, 0), False, (True, True, True))
    assert chunk_bounds.binding(family.RECORD_BUFFER) in source
    assert family.RECORD_BUFFER == 29
    assert family.PRESENCE_BUFFER == family.RECORD_BUFFER + 1
    assert family.presence_binding(family.PRESENCE_BUFFER) in source
    assert f"[[buffer({family.PRESENCE_BUFFER + 1})]]" not in source
    assert family.PRESENCE_BUFFER + 1 <= MAX_BUFFER_BINDINGS
    assert source.index(chunk_bounds.struct_source()) < source.index("kernel void")
    assert source.index(family.presence_struct_source()) < source.index("kernel void")
    assert source.count(chunk_bounds.BLOCK_BEGIN) == 1
    assert source.count(chunk_bounds.BLOCK_END) == 1
    # The block sits after the decode, which the H->D weld cuts its prologue at.
    from meep_gpu.metal_kernels import conductive_fused_hd_pair

    assert (source.index(conductive_fused_hd_pair.DECODE_END)
            < source.index(chunk_bounds.BLOCK_BEGIN))


def test_the_membership_lattice_is_the_coefficient_tables():
    """Half-integer on the B side, integer on the D side, per (dsig, dsigu)."""
    assert family.membership_flags(False) == (
        ("in_h_y", "in_h_z"), ("in_h_z", "in_h_x"), ("in_h_x", "in_h_y"))
    assert family.membership_flags(True) == (
        ("in_i_y", "in_i_z"), ("in_i_z", "in_i_x"), ("in_i_x", "in_i_y"))


# ---------------------------------------------------------------------------
# Coverage and the per-chunk conductivity presence
# ---------------------------------------------------------------------------

def test_each_pml_curl_requires_its_own_conductive_side(backend_ready):
    fields, pml = build(sides=("D",))
    b = family.metal_conductive_pml_curl_coverage(fields, pml, "step_B", HostResidency())
    d = family.metal_conductive_pml_curl_coverage(fields, pml, "step_D", HostResidency())
    assert not b.covered
    assert any("no step_B target" in reason for reason in b.reasons)
    assert d.covered, d.reasons


def _partial_presence(fields, pml):
    """Dx lossy in the interior chunk along x only: 9 of the 27 chunks keep it."""
    shape = fields.grid.shape
    sigma = np.zeros(shape, dtype=np.float32)
    sigma[4:6] = np.float32(0.15)          # the interior chunk along x only
    fields.set_d_conductivity({"Dx": sigma, "Dy": None, "Dz": None})
    kept = ChunkLayout.one_rank(fields.grid, pml).conductive_chunks(sigma, (1, 0, 0))
    assert any(kept) and not all(kept)
    return sigma


def test_a_chunk_that_keeps_no_conductivity_is_served_through_the_presence_words(
        backend_ready):
    """MEEP steps the lossless forms there, and the words say which chunks those are.

    The words are the array path's own per-cell presence (``RegionPlan.conductive_mask``)
    read through the record, and expand back to it on every stored cell; a target every
    chunk keeps, and a lossless one, is the all-kept word.
    """
    from meep_gpu.metal_kernels import offdiag_update_e

    fields, pml = build(sides=())
    sigma = _partial_presence(fields, pml)
    verdict = family.metal_conductive_pml_curl_coverage(
        fields, pml, "step_D", HostResidency())
    assert verdict.covered, verdict.reasons
    record = record_of(fields, pml)
    plan = family.plan_metal_conductive_pml_curl(
        fields, pml, "step_D", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None}, chunk_bounds=record)
    assert plan is not None
    word, lossless_y, lossless_z = plan.presence
    assert lossless_y == lossless_z == family.ALL_CHUNKS_KEPT
    assert 0 < word < family.ALL_CHUNKS_KEPT
    mask = stepping._region_plan(fields, pml).conductive_mask("Dx", sigma, np)  # noqa: SLF001
    codes = offdiag_update_e.cell_chunk_codes(record, (1, 0, 0), fields.grid.shape,
                                              plan.bc)
    assert np.array_equal(offdiag_update_e.expand_presence_word(word, codes), mask)
    # The same volume on every chunk is the all-kept word.
    fields.set_d_conductivity({"Dx": np.full(fields.grid.shape, np.float32(0.15)),
                               "Dy": None, "Dz": None})
    assert family.presence_for(fields, pml, "step_D", record) == family.ALL_KEPT


def test_a_record_that_cannot_carry_the_presence_is_refused_by_name(backend_ready):
    """A record whose chunk codes disagree with the array path's presence refuses."""
    fields, pml = build(sides=())
    _partial_presence(fields, pml)
    wrong = np.array(record_of(fields, pml), copy=True)
    from meep_gpu.chunk_map import record_index

    # The x-axis runs moved by two rows on the D side's half-integer lattice: the
    # interior chunk's codes then cover cells the array path steps losslessly.
    wrong[record_index(True, 0, 1)] += 2
    reasons = family.presence_reasons(fields, pml, "step_D", chunk_bounds=wrong)
    assert reasons and "conductivity presence is refused" in reasons[0], reasons
    for function in (family._arm_coverage, family._arm_plan):  # noqa: SLF001
        text = inspect.getsource(function)
        assert "chunk_layout=context.chunk_layout" in text
    assert "chunk_bounds=context.chunk_bounds" in inspect.getsource(
        family._arm_plan)  # noqa: SLF001


def test_a_lossy_target_steps_the_lossless_forms_where_its_chunk_keeps_none():
    """The lossless branch comes first, keyed on the presence flag, and touches no f_cond."""
    for backward in (False, True):
        source = family.conductive_pml_curl_source((0, 0, 0), backward,
                                                   (True, False, True))
        assert source.count("const bool kc0 = ") == 1
        assert source.count("const bool kc2 = ") == 1
        assert "kc1" not in source            # a lossless target reads no word
        for index in (0, 2):
            branch = source.split(f"    if (!kc{index}) {{\n", 1)[1].split(
                f"    }} else if (ds{index} && du{index}) {{\n", 1)[0]
            assert f"c{index}[" not in branch and f"cf{index}" not in branch
            assert f"if (du{index}) {{ u{index}[ii] = n{index}; }}" in branch
            assert (source.index(f"const bool kc{index} = ")
                    < source.index(f"    if (!kc{index}) {{"))


@pytest.mark.parametrize("product, needle", [
    ("inactive", "active PML"), ("real_lossless", "no step_B target"),
    ("complex", "complex64"),
])
def test_distinct_products_are_refused_by_name(backend_ready, product, needle):
    fields, pml = build(sides=(() if product == "real_lossless" else ("B", "D")))
    if product == "inactive":
        pml = PML(grid=fields.grid, thickness=0)
    elif product == "complex":
        fields.force_complex_fields = True
    verdict = family.metal_conductive_pml_curl_coverage(
        fields, pml, "step_B", HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_plan_requires_the_record_and_binds_it_last(backend_ready):
    fields, pml = build()
    parameter = inspect.signature(family.plan_metal_conductive_pml_curl).parameters[
        "chunk_bounds"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
    with pytest.raises(ChunkLayoutRefusal):
        family.plan_metal_conductive_pml_curl(
            fields, pml, "step_B", HostResidency(),
            functions={shaders.CONTRACT_OFF: lambda *args: None}, chunk_bounds=None)
    record = record_of(fields, pml)
    plan = family.plan_metal_conductive_pml_curl(
        fields, pml, "step_B", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None}, chunk_bounds=record)
    assert plan is not None
    assert plan.chunk_bounds.tolist() == np.asarray(record).tolist()
    bound = plan._args[-2]  # noqa: SLF001
    assert bound.dtype.__str__() == "torch.int32"
    assert bound.cpu().numpy().tolist() == np.asarray(record).tolist()
    words = plan._args[-1]  # noqa: SLF001
    assert words.dtype.__str__() == "torch.int32"
    assert words.cpu().numpy().tolist() == list(plan.presence) == list(family.ALL_KEPT)
    assert len(plan._args) == family.PRESENCE_BUFFER + 1  # noqa: SLF001


def test_mixed_pml_targets_bind_only_their_own_histories(backend_ready):
    fields, pml = build(sides=())
    sigma = np.full(fields.grid.shape, np.float32(0.15))
    fields.set_b_conductivity({"Bx": sigma, "By": None, "Bz": None})
    fields.set_d_conductivity({"Dx": None, "Dy": None, "Dz": sigma})
    record = record_of(fields, pml)
    b = family.plan_metal_conductive_pml_curl(
        fields, pml, "step_B", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None}, chunk_bounds=record)
    d = family.plan_metal_conductive_pml_curl(
        fields, pml, "step_D", HostResidency(),
        functions={shaders.CONTRACT_OFF: lambda *args: None}, chunk_bounds=record)
    assert b is not None and d is not None
    assert b.conductive == (True, False, False)
    assert d.conductive == (False, False, True)


def test_two_active_pml_curl_cycles_match_the_array_path_and_histories(backend_ready):
    reference, reference_pml = build()
    actual, actual_pml = build()
    residency = HostResidency()
    record = record_of(actual, actual_pml)
    b_targets, d_targets = ("Bx", "By", "Bz"), ("Dx", "Dy", "Dz")
    b = family.plan_metal_conductive_pml_curl(
        actual, actual_pml, "step_B", residency,
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_B", b_targets)},
        chunk_bounds=record)
    d = family.plan_metal_conductive_pml_curl(
        actual, actual_pml, "step_D", residency,
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, actual, reference_pml, "step_D", d_targets)},
        chunk_bounds=record)
    assert b is not None and d is not None
    for _ in range(2):
        b.run()
        d.run()
    for stem in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        for name in (stem, "fu_" + stem, "f_cond_" + stem):
            assert np.array_equal(getattr(actual, name).view(np.uint32),
                                  getattr(reference, name).view(np.uint32)), name
    assert b.launches == d.launches == 2


def test_arm_is_wired_after_complete_residency_composition():
    """The whole-step gate makes this a dispatchable composition arm.

    The direct family gate owns the stencil and history recurrence.  The
    whole-step gate additionally proves that the B/D mirror can remain resident
    while ordinary H/E execute in the same step, including both directions of
    conductivity.  Keep the assertion here so a future regression cannot make
    the product silently array-only while its direct gate stays green.
    """
    for slot in family.CONDUCTIVE_SUB_STEPS:
        arm = next(spec for spec in arms.registered(slot) if spec.family == family.FAMILY)
        assert arm.label == "conductive PML curl"
        assert arm.wired
