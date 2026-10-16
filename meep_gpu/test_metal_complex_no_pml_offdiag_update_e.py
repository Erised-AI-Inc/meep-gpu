"""Host contract for Metal's complex/Bloch no-PML tensor-row ``update_E``.

Under ruling R7 (software binary64) the family forms the array path's double row per
part with the array path's Bloch connection (plane 0 of the same-index operands) and
the per-chunk presence words, binds the chunk record and the words last, and its arm
is wired; the weld that lifts it carries the words in ``Params`` before the record.
One device test measures the row against ``stepping.update_E`` on a generic state
with a phased and a negated connection.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.chunk_map import ChunkLayout, chunk_bounds_record
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, complex_fields, shaders, softfp64, subnormal
from meep_gpu.metal_kernels import complex_no_pml_offdiag_update_e as family
from meep_gpu.metal_kernels import complex_no_pml_conductive as shared
from meep_gpu.metal_kernels import offdiag_update_e as offdiag
from meep_gpu.pml import PML


def _torch_mps() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch is no device
        return False
    return bool(getattr(getattr(torch, "backends", None), "mps", None)
                and torch.backends.mps.is_available())


# The sentinel names the resource, so the skip is a declared one (the root
# conftest.py's no-silent-skip policy) rather than a coverage gap.
needs_mps = pytest.mark.skipif(
    not _torch_mps(), reason="[requires_resource][mps_device] needs an Apple GPU (MPS)")


class HostResidency:
    #: Where a plan uploads a record or a Params tensor: the host, so no GPU is reached.
    device = "cpu"

    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        previous = self.arrays.get(name)
        if previous is not None and previous is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host




def build(*, rows=True, active_pml=False, complex_storage=True, bloch=True, seed=137,
          k_point=(0.2, -0.125, 0.0)):
    grid = Grid(resolution=8.0, cell_size=(1.125, 0.875, 0.75), courant=0.35,
                boundaries="periodic",
                k_point=tuple(k_point) if bloch else (0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_field_storage()
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = np.full(shape, np.float32(2.25))
    inverse = np.full(shape, np.float32(1.0 / 2.25))
    tensor_rows = ({
        "Ex": {"Ey": (0.04 * rng.standard_normal(shape)).astype(np.float32)},
        "Ey": {"Ez": (0.03 * rng.standard_normal(shape)).astype(np.float32)},
        "Ez": {"Ex": (0.02 * rng.standard_normal(shape)).astype(np.float32)},
    } if rows else {})
    fields.set_epsilon_volumes(
        {name: epsilon for name in ("Ex", "Ey", "Ez")},
        {name: inverse for name in ("Ex", "Ey", "Ez")}, tensor_rows)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez"):
        real = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
        getattr(fields, name)[...] = (
            real + 1j * rng.uniform(-0.4, 0.4, shape).astype(np.float32)
            if complex_storage else real)
    return fields, PML(grid=grid, thickness=1 if active_pml else 0)


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(shared, "_metal_backend_reasons", lambda grid: [])


def _record(fields, pml):
    """The chunk record of the division this no-absorber run steps with: one chunk."""
    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, None))


def _reference_function(reference, reference_pml, actual, record):
    def run(*args):
        targets, sources, inverse, rows = args[:3], args[3:6], args[6:9], args[9:15]
        assert targets == (actual.Ex, actual.Ey, actual.Ez)
        assert sources == (actual.Dx, actual.Dy, actual.Dz)
        assert inverse == tuple(actual.inverse_epsilon_for(name) for name in ("Ex", "Ey", "Ez"))
        assert rows[0] is actual.chi1inv_offdiagonal_for("Ex")["Ey"]
        # Four scalars, the six phase pairs (the backward side's three, then the
        # forward side's), then the chunk record and the presence words, last.
        assert len(args) == 15 + 4 + 6 + 2
        assert np.array_equal(np.asarray(args[-2].cpu()), record.astype(np.int32))
        assert tuple(int(word) for word in args[-1].cpu()) == tuple(
            np.asarray(offdiag.ALL_KEPT, dtype=np.uint32).view(np.int32))
        stepping.update_E(reference, reference_pml)
        for target, name in zip(targets, ("Ex", "Ey", "Ez")):
            target[...] = getattr(reference, name)
    return run


def _unresolvable(grid, pml):  # noqa: ARG001
    """What the array path does with a grid it will not resolve.

    ``coverage._boundary_kinds`` turns exactly this into ``None`` (coverage.py:105-121),
    which is the input the predicate has to REFUSE rather than crash on.
    """
    raise RuntimeError("the array path will not resolve this grid")


def test_the_connection_lands_on_the_same_index_operands_plane_zero():
    """``stepping._bloch_ghost_operand``'s placement, on two ``CONNECT_PHASE`` axes:
    the partner pair's own word (and its own-axis-up twin) on the partner axis's
    plane 0 with ``eikna`` (``dpy``), the formed product on the component's own
    axis's plane 0 with ``conj(eikna)`` (``upx``); the shifted reads are plain wraps
    and the far product is the unconnected product shifted up."""
    source = family.complex_no_pml_offdiag_source(
        (1, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0), (2, 2, 0))
    assert source.count("[[buffer(") == family.BINDING_COUNT == 27
    assert "    home_00 = (j == 0) ? c_mul(home_00, dpy) : home_00;" in source
    assert "    far_a_00 = (j == 0) ? c_mul(far_a_00, dpy) : far_a_00;" in source
    assert ("    near_product_00 = (i == 0) ? c_mul(near_product_00, upx) : "
            "near_product_00;") in source
    assert "down_00 = (" not in source and "far_product_00 = (" not in source
    assert "corner_00 = (" not in source.split("float2 corner_00", 1)[1]
    assert "f0[ii] = src0;" in source
    assert "f_w_Ex" not in source


def test_the_row_is_the_double_row_per_part_rounded_once():
    """``widen(gs*us)`` per part, each kept term's float32 pair sum scaled by MEEP's
    double 0.25 and added left to right, one rounding per part, through the
    software-binary64 helpers; the float32 ``0.25f`` row and its ``total`` are gone;
    a component with no live row keeps the float32 diagonal."""
    source = family.complex_no_pml_offdiag_source(
        (1, 1, 0, 0, 0, 0), (1, 0, 0), (1, 0, 0), (0, 2, 0))
    assert softfp64.MSL_HELPERS.strip("\n") in source
    assert offdiag.presence_block((1, 1, 0, 0, 0, 0), (1, 0, 0)) in source
    lines = [
        "    float2 od_d0 = c_mul_field_left(g0[ii], e0[ii]);",
        "    ulong od_re0 = sd_widen_f(od_d0.x, true);",
        "    ulong od_im0 = sd_widen_f(od_d0.y, true);",
        "    if (od_kept_00) { od_re0 = sd_add(od_re0, sd_scale_pow2(sd_widen_f("
        "sum_00.x, true), -2, true), true); }",
        "    if (od_kept_00) { od_im0 = sd_add(od_im0, sd_scale_pow2(sd_widen_f("
        "sum_00.y, true), -2, true), true); }",
        "    if (od_kept_01) { od_re0 = sd_add(od_re0, sd_scale_pow2(sd_widen_f("
        "sum_01.x, true), -2, true), true); }",
        "    if (od_kept_01) { od_im0 = sd_add(od_im0, sd_scale_pow2(sd_widen_f("
        "sum_01.y, true), -2, true), true); }",
        "    float2 src0 = float2(sd_narrow_f(od_re0, true), sd_narrow_f(od_im0, true));",
    ]
    assert "\n".join(lines) in source
    # The wall mask on each term's pair sum, both parts, before the scale.
    assert "    sum_00 = at_y ? float2(0.0f, 0.0f) : sum_00;" not in source
    assert "    sum_01 = at_x ? float2(0.0f, 0.0f) : sum_01;" not in source
    walled = family.complex_no_pml_offdiag_source(
        (1, 1, 0, 0, 0, 0), (0, 1, 1), (0, 1, 1), (0, 0, 0))
    assert "    sum_00 = at_y ? float2(0.0f, 0.0f) : sum_00;" in walled
    assert "    sum_01 = at_z ? float2(0.0f, 0.0f) : sum_01;" in walled
    assert "    float2 src1 = c_mul_field_left(g1[ii], e1[ii]);" in source
    assert "0.25f" not in source and "total" not in source


def test_the_record_and_the_presence_bind_last():
    from meep_gpu.metal_kernels import chunk_bounds

    source = family.complex_no_pml_offdiag_source(
        (1, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0), (2, 2, 0))
    assert chunk_bounds.binding(family.RECORD_BUFFER) in source
    assert offdiag.presence_binding(family.PRESENCE_BUFFER) in source
    assert family.RECORD_BUFFER == 25 and family.PRESENCE_BUFFER == 26
    import inspect

    parameter = inspect.signature(
        family.plan_metal_complex_no_pml_offdiag).parameters["chunk_bounds"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_no_ruling_refuses_the_row(backend_ready):
    """Every clause holds, so the predicate admits: no reason names R7."""
    fields, pml = build()
    verdict = family.metal_complex_no_pml_offdiag_coverage(
        fields, pml, HostResidency(), _record(fields, pml))
    assert verdict.covered, verdict.reasons
    assert not hasattr(family, "R7_REFUSAL") and not hasattr(family, "r7_reasons")


def test_the_structural_clauses_admit_and_bind_in_order(backend_ready):
    """The builder's structural contract: the volumes, the scalars, the six phases,
    then the record and the presence words last."""
    reference, reference_pml = build()
    actual, actual_pml = build()
    record = _record(actual, actual_pml)
    plan = family.plan_metal_complex_no_pml_offdiag(
        actual, actual_pml, HostResidency(), chunk_bounds=record,
        functions={shaders.CONTRACT_OFF: _reference_function(
            reference, reference_pml, actual, record)})
    assert plan is not None
    assert plan.presence == offdiag.ALL_KEPT
    for _ in range(3):
        plan.run()
    for name in ("Ex", "Ey", "Ez"):
        assert np.array_equal(getattr(actual, name).view(np.uint32),
                              getattr(reference, name).view(np.uint32)), name
    assert plan.launches == 3


def test_an_unreadable_boundary_rule_is_a_refusal_and_not_a_crash(monkeypatch):
    """UNREADABLE IS A REFUSAL, NEVER A CRASH — measured on an otherwise REAL grid.

    ``coverage._boundary_kinds`` answers ``None`` for a grid the array path will not
    resolve, and ``complex_no_pml_conductive._base_reasons`` — the clause list this
    family shares with the other complex no-PML curls, the off-diagonal E and the
    fused conductive pair — reads that triple TWICE: once for the covered ghost rule
    and once for per-axis Bloch legality. The grid here is real, so ``bloch_phase``
    is callable and the SECOND read is genuinely reached; a guard on the first read
    alone would still raise here.
    """
    fields, pml = build()
    monkeypatch.setattr(stepping, "_boundary_kinds", _unresolvable)
    verdict = family.metal_complex_no_pml_offdiag_coverage(
        fields, pml, HostResidency())
    assert verdict.covered is False
    assert any("the per-axis boundary rule is unreadable" in reason
               for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("no_rows", "no live off-diagonal"),
    ("active_pml", "active PML"),
    ("real", "complex64 storage"),
])
def test_distinct_products_are_refused_by_name(backend_ready, mutation, needle):
    fields, pml = build(rows=mutation != "no_rows", active_pml=mutation == "active_pml")
    if mutation == "real":
        fields, pml = build(complex_storage=False, bloch=False)
    verdict = family.metal_complex_no_pml_offdiag_coverage(
        fields, pml, HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_the_presence_clause_reads_the_composers_record_last(backend_ready,
                                                             monkeypatch):
    """Clause (p) runs on an otherwise admitted run, through the record the composer
    binds, and a presence the record cannot carry is refused by name."""
    seen = []

    def refuse(fields, pml, codes, chunk_bounds):
        seen.append((tuple(codes), chunk_bounds))
        return ["the off-diagonal presence is refused: planted"]

    monkeypatch.setattr(offdiag, "presence_reasons", refuse)
    fields, pml = build()
    record = _record(fields, pml)
    verdict = family.metal_complex_no_pml_offdiag_coverage(
        fields, pml, HostResidency(), record)
    assert verdict.reasons == ("the off-diagonal presence is refused: planted",)
    assert len(seen) == 1 and seen[0][1] is record and seen[0][0] == (0, 0, 0)
    seen.clear()
    fields, pml = build(rows=False)
    verdict = family.metal_complex_no_pml_offdiag_coverage(
        fields, pml, HostResidency(), record)
    assert not verdict.covered and not seen, "the clause ran on a refused run"


def test_arm_is_registered_wired_and_visible_to_package_registry():
    from meep_gpu.metal_kernels import registry
    import meep_gpu.metal_kernels as package

    arm = next(spec for spec in arms.registered(family.SLOT) if spec.family == family.FAMILY)
    assert arm.label == "complex no-PML off-diagonal"
    assert arm.wired
    assert family.FAMILY in registry.FAMILY_MODULES
    assert package.complex_no_pml_offdiag_update_e is family


def test_a_negated_connection_is_a_per_part_negation_on_plane_zero():
    """``CONNECT_NEGATE`` (a phase of -1) is MEEP's negation of the word, a sign-bit
    operation per part, never a product by the placeholder word the plan binds."""
    source = family.complex_no_pml_offdiag_source(
        (1, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0),
        (complex_fields.CONNECT_NEGATE, complex_fields.CONNECT_NEGATE, 0))
    assert "    home_00 = (j == 0) ? -home_00 : home_00;" in source
    assert "    far_a_00 = (j == 0) ? -far_a_00 : far_a_00;" in source
    assert "    near_product_00 = (i == 0) ? -near_product_00 : near_product_00;" in source
    assert "c_mul(home_00" not in source
    with pytest.raises(ValueError, match="requires a periodic boundary"):
        family.complex_no_pml_offdiag_source(
            (1, 0, 0, 0, 0, 0), (1, 0, 0), (1, 0, 0),
            (complex_fields.CONNECT_NEGATE, 0, 0))


@needs_mps
@pytest.mark.parametrize("k_point", ((0.2, -0.125, 0.0), (1.0 / 2.25, 0.0, 0.0)),
                         ids=("phased", "negated_x"))
def test_the_double_row_is_the_array_paths_word_on_a_generic_state(monkeypatch,
                                                                   k_point):
    """THE RULING'S CLAIM, on the device: on a generic complex state, where a float32
    row differs from the array path's double row, the shipped row (Bloch connection
    and presence included) is 0 words from ``stepping.update_E`` per part. The
    ``negated_x`` case puts a phase of -1 on x (k_x * L_x = 1/2)."""
    from meep_gpu.metal_kernels import device

    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    reference, reference_pml = build(seed=20261009, k_point=k_point)
    actual, actual_pml = build(seed=20261009, k_point=k_point)
    kinds = complex_fields.phase_arguments(complex_fields.bloch_phase_table(
        actual.grid, stepping._boundary_kinds(actual.grid, None)), backward=True)[0]
    if k_point[0] > 0.3:
        assert kinds[0] == complex_fields.CONNECT_NEGATE, kinds
    residency = device.Residency()
    plan = family.plan_metal_complex_no_pml_offdiag(
        actual, actual_pml, residency,
        chunk_bounds=_record(actual, actual_pml))
    assert plan is not None
    before = {name: np.array(getattr(reference, name), copy=True)
              for name in ("Ex", "Ey", "Ez")}
    residency.sync_in()
    plan.run()
    residency.sync_out()
    stepping.update_E(reference, reference_pml)

    def words(value):
        return np.ascontiguousarray(value).reshape(-1).view(np.uint32)

    moved = sum(int(np.count_nonzero(words(before[n]) != words(getattr(reference, n))))
                for n in ("Ex", "Ey", "Ez"))
    assert moved > 0, "the oracle did nothing; identity here would be vacuous"
    differing = {n: int(np.count_nonzero(words(getattr(actual, n))
                                         != words(getattr(reference, n))))
                 for n in ("Ex", "Ey", "Ez")}
    assert sum(differing.values()) == 0, differing


# ---------------------------------------------------------------------------
# The weld that lifts this body: the record and the presence in Params
# ---------------------------------------------------------------------------

def test_no_ruling_refuses_the_weld_through_its_constitutive_half(backend_ready):
    from meep_gpu.metal_kernels import complex_no_pml_offdiag_fused_electric_pair as weld

    fields, pml = build()
    verdict = weld.metal_complex_no_pml_offdiag_fused_electric_pair_coverage(
        fields, pml, (), HostResidency(), _record(fields, pml))
    assert not any("constitutive half" in reason for reason in verdict.reasons), (
        verdict.reasons)
    assert not any("R7" in reason or "float64" in reason
                   for reason in verdict.reasons), verdict.reasons


def test_the_weld_carries_the_record_last_in_params_into_every_lifted_body():
    """``ChunkBounds cb`` is the last ``Params`` member, aliased before the body,
    and the last parameter of ``step_cell`` and every ``d_final_c``: the lifted
    curl's presence word is read at the cell being stepped, a re-derived foreign
    cell included."""
    from meep_gpu.metal_kernels import chunk_bounds
    from meep_gpu.metal_kernels import complex_no_pml_offdiag_fused_electric_pair as weld

    source = weld.complex_no_pml_offdiag_fused_electric_pair_source(
        (1, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0), (2, 2, 0))
    struct = source.split("struct Params {", 1)[1].split("};", 1)[0]
    assert struct.rstrip().endswith(chunk_bounds.params_member().strip())
    lines = [line.strip() for line in struct.strip().splitlines()]
    assert lines[-2:] == ["OffdiagPresence kw;", "ChunkBounds cb;"], lines[-2:]
    assert offdiag.presence_params_alias() in source
    assert softfp64.MSL_HELPERS.strip("\n") in source
    assert "kw.kept01" in source.split("kernel void", 1)[1]
    assert source.index("struct ChunkBounds {") < source.index("struct Params {")
    assert chunk_bounds.params_alias() in source
    assert "constant ChunkBounds& cb) {" in source
    for c in range(3):
        header = source.split(f"d_final_{c}(", 1)[1].split(")", 1)[0]
        assert "constant ChunkBounds& cb" in header, c
    helper = source.split("static inline step_cell_result step_cell(", 1)[1]
    assert chunk_bounds.BLOCK_BEGIN in helper.split("\n}\n", 1)[0]
    record = weld.params_record_dtype()
    assert record.names[-2:] == ("kw", "cb")
    assert record.fields["cb"][0] == chunk_bounds.record_dtype()
    assert record.fields["kw"][0] == offdiag.presence_record_dtype()
    assert (record.fields["kw"][1] + offdiag.presence_record_dtype().itemsize
            == record.fields["cb"][1] == 116)
    assert record.fields["cb"][1] + chunk_bounds.record_dtype().itemsize <= record.itemsize
    assert record.itemsize == weld.PARAMS_ITEMSIZE == 216


def test_the_weld_builder_takes_the_record_with_no_default():
    import inspect

    from meep_gpu.metal_kernels import complex_no_pml_offdiag_fused_electric_pair as weld

    parameter = inspect.signature(
        weld.plan_metal_complex_no_pml_offdiag_fused_electric_pair).parameters[
            "chunk_bounds"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty
