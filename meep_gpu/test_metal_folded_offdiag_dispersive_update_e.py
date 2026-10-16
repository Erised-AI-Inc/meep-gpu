"""Host contracts for Metal's folded tensor-row dispersive intersection.

Under ruling R7 (software binary64) the family forms the off-diagonal single's double
row over the ordered ``D - P`` gathers, adds each term where the cell's chunk keeps
it (the six presence words ride in ``Params`` before the chunk record), and its arm
is wired; its PML tail is the per-chunk select on the chunk record its ``Params``
carries. Two device tests measure the row and the presence against
``stepping.update_E``.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.chunk_map import ChunkLayout, ChunkLayoutRefusal, chunk_bounds_record
from meep_gpu.dispersion import PolarizationState, Susceptibility
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.metal_kernels import arms, device
from meep_gpu.metal_kernels import folded_offdiag_dispersive_update_e as family
from meep_gpu.metal_kernels import offdiag_update_e as offdiag
from meep_gpu.metal_kernels import shaders, softfp64, subnormal, symmetry
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
    device = "cpu"

    def __init__(self):
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        prior = self.arrays.get(name)
        if prior is not None and prior is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def build(*, counts: Sequence[int] = (2, 1, 2), rows: bool = True,
          fold: bool = True, seed: int = 906) -> Tuple[Fields, PML]:
    grid = Grid(resolution=8.0, cell_size=(2.0, 1.5, 1.25), courant=0.35,
                boundaries="periodic", symmetry=(Mirror("Y", 1),) if fold else (),
                xp=np)
    fields = Fields(grid=grid)
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = {name: np.full(shape, value, np.float32)
               for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))}
    inverse = {name: np.float32(1.0) / value for name, value in epsilon.items()}
    tensor_rows = ({"Ex": {"Ey": rng.uniform(-0.2, 0.2, shape).astype(np.float32),
                            "Ez": rng.uniform(-0.2, 0.2, shape).astype(np.float32)},
                    "Ey": {"Ez": rng.uniform(-0.2, 0.2, shape).astype(np.float32)}}
                   if rows else None)
    fields.set_epsilon_volumes(epsilon, inverse, tensor_rows)
    for index in range(max(counts)):
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.8 + 0.1 * index, 0.06 + 0.01 * index, "lorentzian"),
            {name: 0.18 + 0.02 * index if counts[axis] > index else 0.0
             for axis, name in enumerate(("Ex", "Ey", "Ez"))}, grid, np.float32))
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((2, 2), (0, 2) if fold else (2, 2), (2, 2)))
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey",
                 "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = rng.uniform(-0.2, 0.2, shape).astype(np.float32)
            state.P_prev[component][...] = rng.uniform(-0.2, 0.2, shape).astype(np.float32)
    return fields, pml


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(
        "meep_gpu.metal_kernels.folded_offdiag_update_e._metal_backend_reasons",
        lambda grid: [])


def _record(fields, pml):
    return chunk_bounds_record(ChunkLayout.one_rank(fields.grid, pml))


def test_the_scalars_the_presence_and_the_record_ride_in_one_params_binding():
    source = family.folded_offdiag_dispersive_source(
        (1, 1, 1, 0, 0, 0), (0, 3, 0), (0, 0, 0), (0, 1, 0), (8, 8, 8))
    assert family.BINDING_COUNT == 28 < device.MAX_BUFFER_BINDINGS
    assert source.count("[[buffer(") == family.BINDING_COUNT
    assert f"constant Params& prm [[buffer({family.PARAMS_BUFFER})]]" in source
    struct = source.split("struct Params {", 1)[1].split("};", 1)[0]
    lines = [line.strip() for line in struct.strip().splitlines()]
    assert lines[-2:] == ["OffdiagPresence kw;", "ChunkBounds cb;"], lines
    assert offdiag.presence_struct_source() in source
    assert offdiag.presence_params_alias() in source
    assert "    constant ChunkBounds& cb = prm.cb;" in source
    for name in ("nx", "ny", "nz", "n_elem"):
        assert f"    uint {name} = prm.{name};" in source
        assert f"constant uint& {name}" not in source
    assert "p[7u * n_elem + uint(index)]" in source
    assert source.index("p[0u * n_elem + uint(index)]") < source.index(
        "p[7u * n_elem + uint(index)]")


def test_the_params_record_is_four_scalars_the_presence_then_the_record_word_for_word():
    dtype = family.params_record_dtype()
    assert dtype.itemsize == 16 + 24 + 100
    assert [dtype.fields[name][1]
            for name in ("nx", "ny", "nz", "n_elem", "kw", "cb")] == [0, 4, 8, 12, 16, 40]
    assert dtype.fields["kw"][0] == offdiag.presence_record_dtype()
    fields, pml = build()
    record = _record(fields, pml)
    presence = (1, 2, 3, 4, 5, offdiag.ALL_CHUNKS_KEPT)
    words = family.params_words(fields.grid.shape, record, presence)
    assert tuple(words[:4]) == (*fields.grid.shape, int(np.prod(fields.grid.shape)))
    assert tuple(words[4:10]) == presence
    assert np.array_equal(words[10:], record.astype(np.int32))
    with pytest.raises(ChunkLayoutRefusal):
        family.params_words(fields.grid.shape, None, offdiag.ALL_KEPT)
    with pytest.raises(ValueError, match="no off-diagonal presence words"):
        family.params_words(fields.grid.shape, record, None)


def test_the_tail_is_the_base_constitutive_kernels_per_chunk_select():
    source = family.folded_offdiag_dispersive_source(
        (1, 1, 1, 0, 0, 0), (0, 3, 0), (0, 0, 0), (0, 1, 0), (2, 1, 2))
    assert shaders.constitutive_bounds_block("E") in source
    flags = shaders.constitutive_membership_flags("E")
    assert (f"    const bool dw0 = {flags[0]}, dw1 = {flags[1]}, dw2 = {flags[2]};"
            in source)
    certified = shaders.constitutive_source("E")
    for t in range(3):
        for line in (f"    float prev{t} = dw{t} ? w{t}[ii] : 0.0f;",
                     f"    if (dw{t}) {{ w{t}[ii] = src{t}; }}",
                     f"    float a{t} = src{t};",
                     f"    if (dw{t}) {{ a{t} = f{t}[ii] + ((kp_{t} * src{t}) - "
                     f"(km_{t} * prev{t})); }}",
                     f"    f{t}[ii] = a{t};"):
            assert line in certified and line in source, line
    assert "a0 = a0 + kp_0 * src0;" not in source


def test_the_row_is_the_off_diagonal_singles_double_row_over_d_minus_p():
    """``gs`` is the ordered ``D - P`` gather and the row is
    :func:`.offdiag_update_e._row_assembly`'s text: each term's float32 pair sum
    wall-masked, scaled by MEEP's double 0.25 and added left to right where the
    chunk keeps it, one rounding. The float32 ``0.25f`` row and its ``total`` are
    gone; a component with no live row keeps the float32 ``gs*us``."""
    walls = (1, 0, 1)
    source = family.folded_offdiag_dispersive_source(
        (1, 1, 1, 0, 0, 0), (1, 3, 1), walls, (0, 1, 0), (2, 1, 2))
    assert softfp64.MSL_HELPERS.strip("\n") in source
    assert "    float gs0 = dmp0(g0, p0, ii, n_elem);" in source
    assert "    float us0 = e0[ii];" in source
    assert offdiag._row_assembly(0, ("00", "01"), walls) in source
    assert offdiag._row_assembly(1, ("10",), walls) in source
    assert "    float src2 = dmp2(g2, p2, ii, n_elem) * e2[ii];" in source
    assert "0.25f" not in source and "total" not in source
    assert "float sum_00 = (near_00 * unear_00) + (far_00 * ufar_00);" in source
    assert offdiag.presence_block((1, 1, 1, 0, 0, 0), (1, 3, 1)) in source


def test_partner_stencil_reads_ordered_d_minus_p_not_raw_d():
    source = family.folded_offdiag_dispersive_source(
        (1, 0, 0, 0, 0, 0), (0, 3, 0), (0, 0, 0), (0, 1, 0), (2, 1, 2))
    assert "dmp1(g1, p1, i * nyz + dj * nzi + k, n_elem)" in source
    assert "float near_00 = dmp1(g1, p1, ii, n_elem)" in source
    assert "at_y ? -dn_00 : dn_00" in source


def test_folded_tensor_dispersive_product_and_max_pole_count_are_admitted(
        backend_ready):
    """No ruling refuses the row: every clause holds and the predicate admits."""
    fields, pml = build(counts=(family.MAX_POLES,) * 3)
    verdict = family.folded_offdiag_dispersive_coverage(fields, pml, HostResidency(),
                                                        _record(fields, pml))
    assert verdict.covered, verdict.reasons
    assert not hasattr(offdiag, "R7_REFUSAL") and not hasattr(offdiag, "r7_reasons")


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
    verdict = family.folded_offdiag_dispersive_coverage(fields, pml, HostResidency(),
                                                        record)
    assert verdict.reasons == ("the off-diagonal presence is refused: planted",)
    assert len(seen) == 1 and seen[0][1] is record
    fields.polarizations.clear()
    seen.clear()
    verdict = family.folded_offdiag_dispersive_coverage(fields, pml, HostResidency(),
                                                        record)
    assert not verdict.covered and not seen, "the clause ran on a refused run"


@pytest.mark.parametrize("mutation, needle", [
    ("no_rows", "no off-diagonal"),
    ("no_poles", "no susceptibility"),
    ("unfolded", "no mirror plane"),
])
def test_intersection_boundaries_refuse_by_name(backend_ready, mutation, needle):
    fields, pml = build(rows=mutation != "no_rows", fold=mutation != "unfolded")
    if mutation == "no_poles":
        fields.polarizations.clear()
    verdict = family.folded_offdiag_dispersive_coverage(fields, pml, HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_builder_allocates_three_persistent_max_pole_packs(backend_ready):
    fields, pml = build(counts=(2, 1, 2))
    residency = HostResidency()
    with pytest.raises(TypeError):
        family.plan_folded_offdiag_dispersive(fields, pml, residency,
                                              contract_variants=())
    plan = family.plan_folded_offdiag_dispersive(
        fields, pml, residency, contract_variants=(), chunk_bounds=_record(fields, pml))
    assert plan is not None
    assert np.array_equal(plan.chunk_bounds, _record(fields, pml))
    # Rows nonzero on every cell: no chunk deletes them, so every word is all-kept,
    # and the words ride in Params between the scalars and the record.
    assert plan.presence == offdiag.ALL_KEPT
    assert tuple(plan.params_words[4:10]) == tuple(
        np.asarray(offdiag.ALL_KEPT, dtype=np.uint32).view(np.int32))
    assert plan.counts == (2, 1, 2)
    assert tuple(plan._packs) == ("Ex", "Ey", "Ez")
    assert all(pack.shape == (family.MAX_POLES, plan.n_elem)
               for pack in plan._packs.values())
    assert residency.arrays["row:Ey:Ex"] is fields.inverse_epsilon_for("Ey")
    assert residency.arrays["row:Ez:Ex"] is fields.inverse_epsilon_for("Ez")


def test_arm_is_wired_offered_on_its_intersection_and_keeps_the_device_side_pole_refresh():
    arm = next(spec for spec in arms.registered("update_E") if spec.family == family.FAMILY)
    assert arm.label == family.LABEL
    assert arm.wired

    def consulted(fields, pml):
        context = arms.StepContext(fields, pml, None, (shaders.CONTRACT_OFF,),
                                   chunk_bounds=_record(fields, pml))
        return [bound for bound in arms.arms_for("update_E", context)
                if bound.label == family.LABEL and bound.gate]

    assert len(consulted(*build())) == 1
    assert consulted(*build(fold=False)) == []
    source = inspect.getsource(family.MetalFoldedOffdiagDispersivePlan._refresh_packs)
    assert "tensor_for_host(state.P[component])" in source


# ---------------------------------------------------------------------------
# Device-backed: the row and the presence against stepping.update_E
# ---------------------------------------------------------------------------

_COMPARED = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")


def _words(value):
    return np.ascontiguousarray(value).reshape(-1).view(np.uint32)


def _device_step(actual, pml):
    """One launch through the engine builder on ``actual`` (a twin of the oracle's
    fixture), with the record of the division ``stepping.update_E`` steps with."""
    residency = device.Residency()
    plan = family.plan_folded_offdiag_dispersive(
        actual, pml, residency, chunk_bounds=chunk_bounds_record(
            stepping._region_plan(actual, pml).layout))
    assert plan is not None
    residency.sync_in()
    plan.run()
    residency.sync_out()
    assert plan.launches == 1
    return actual


@needs_mps
def test_the_double_row_is_the_array_paths_word_on_a_generic_state(monkeypatch):
    """THE RULING'S CLAIM: on a generic state, where a float32 row differs from the
    array path's double row, the shipped software-binary64 row over ``D - P`` is 0
    words from ``stepping.update_E``, ``f_w`` included."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = build(counts=(2, 1, 2))
    twin, twin_pml = build(counts=(2, 1, 2))
    before = {name: np.array(getattr(fields, name), copy=True) for name in _COMPARED}
    actual = _device_step(twin, twin_pml)
    stepping.update_E(fields, pml)
    moved = sum(int(np.count_nonzero(_words(before[n]) != _words(getattr(fields, n))))
                for n in _COMPARED)
    assert moved > 0, "the oracle did nothing; identity here would be vacuous"
    differing = {n: int(np.count_nonzero(_words(getattr(actual, n))
                                         != _words(getattr(fields, n))))
                 for n in _COMPARED}
    assert sum(differing.values()) == 0, differing


@needs_mps
def test_a_term_its_chunk_dropped_keeps_a_negative_zero_row(monkeypatch):
    """PRESENCE ON THE DEVICE. The rows are zero in every PML chunk of the metallic
    axes, so MEEP drops the terms there; with ``Dx - P = -0`` the row so far is
    ``-0`` and adding a term's ``+0`` would give ``+0``. The kernel must keep ``-0``
    where the chunk dropped the term, which is the array path's word."""
    monkeypatch.setenv(subnormal.POLICY_ENV, subnormal.FLUSH)
    fields, pml = _presence_build()
    twin, twin_pml = _presence_build()
    codes, _notes = symmetry.folded_axis_kinds(fields.grid, pml)
    layout = stepping._region_plan(fields, pml).layout
    words = offdiag.presence_for(fields, pml, codes, chunk_bounds_record(layout))
    assert words[0] != offdiag.ALL_CHUNKS_KEPT and words[1] != offdiag.ALL_CHUNKS_KEPT
    actual = _device_step(twin, twin_pml)
    stepping.update_E(fields, pml)
    negative = int(np.count_nonzero(_words(fields.Ex) == np.uint32(0x80000000)))
    assert negative > 0, "no -0 row survived: the case does not exercise presence"
    differing = {n: int(np.count_nonzero(_words(getattr(actual, n))
                                         != _words(getattr(fields, n))))
                 for n in _COMPARED}
    assert sum(differing.values()) == 0, differing


def _presence_build():
    """Rows zero outside the interior of the metallic x and z axes; ``Dx - P = -0``."""
    grid = Grid(resolution=10.0, cell_size=(1.6, 1.2, 1.6), courant=0.35,
                boundaries={"x": "metallic", "y": "periodic", "z": "metallic"},
                symmetry=(Mirror("Y", 1),), xp=np)
    fields = Fields(grid=grid)
    shape = grid.shape
    rng = np.random.default_rng(1009)
    keep = np.zeros(shape, dtype=bool)
    keep[3:-3, :, 3:-3] = True
    epsilon = {name: np.full(shape, 2.0, np.float32) for name in ("Ex", "Ey", "Ez")}
    inverse = {name: np.float32(1.0) / value for name, value in epsilon.items()}
    rows = {"Ex": {partner: np.where(keep, rng.uniform(0.05, 0.2, shape), 0.0
                                     ).astype(np.float32) for partner in ("Ey", "Ez")}}
    fields.set_epsilon_volumes(epsilon, inverse, rows)
    fields.polarizations.append(PolarizationState(
        Susceptibility(0.8, 0.06, "lorentzian"),
        {"Ex": 0.18, "Ey": 0.18, "Ez": 0.18}, grid, np.float32))
    fields.enable_pml_storage()
    pml = PML(grid=grid, thickness=((2, 2), (0, 2), (2, 2)))
    for name in ("Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(0.01, 0.4, shape).astype(np.float32)
    fields.Dx[...] = np.float32(-0.0)
    state = fields.polarizations[0]
    for component in state.driven():
        state.P[component][...] = (np.float32(0.0) if component == "Ex" else
                                   rng.uniform(-0.005, 0.0, shape).astype(np.float32))
        state.P_prev[component][...] = rng.uniform(-0.2, 0.2, shape).astype(np.float32)
    return fields, pml
