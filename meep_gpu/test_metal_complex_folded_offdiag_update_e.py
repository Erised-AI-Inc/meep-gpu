"""Host contracts for Metal's complex folded tensor-row PML ``update_E`` arm.

Under ruling R7 the row is MEEP's double row per part, in software binary64
(:mod:`.metal_kernels.softfp64`), with the real family's per-chunk presence words; the
Bloch connection lands on the same-index operand's plane 0, as the array path's
``_bloch_ghost_operand`` places it. The kernel's PML tail is the per-chunk select; the
chunk record and then the presence words bind last. The arm is wired.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytest

from meep_gpu import chunk_map
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.metal_kernels import arms
from meep_gpu.metal_kernels import complex_folded_offdiag_update_e as family
from meep_gpu.metal_kernels import folded_complex, shaders
from meep_gpu.metal_kernels import offdiag_update_e as offdiag
from meep_gpu.pml import PML


class HostResidency:
    """Array-preserving mirror double: these contracts never compile Metal."""

    #: The device the plan-owned record tensor is placed on: the host's.
    device = "cpu"

    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        prior = self.arrays.get(name)
        if prior is not None and prior is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def build(*, axes: str = "Y", phases: Sequence[int] = (1,), rows: bool = True,
          active_pml: bool = True, complex_storage: bool = True,
          seed: int = 17081,
          k_point: Sequence[float] = (0.0, 0.0, 0.0)) -> Tuple[Fields, PML]:
    grid = Grid(
        resolution=8.0,
        cell_size=(2.0, 1.5, 1.25),
        courant=0.35,
        boundaries="periodic",
        k_point=tuple(k_point),
        symmetry=tuple(Mirror(axis, int(phase))
                       for axis, phase in zip(axes, phases)),
        xp=np,
    )
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = np.full(shape, np.float32(2.25))
    inverse = np.full(shape, np.float32(1.0 / 2.25))
    tensor_rows = ({
        "Ex": {"Ey": (0.04 * rng.standard_normal(shape)).astype(np.float32)},
        "Ey": {"Ez": (0.03 * rng.standard_normal(shape)).astype(np.float32)},
        "Ez": {"Ex": (0.02 * rng.standard_normal(shape)).astype(np.float32)},
    } if rows else None)
    fields.set_epsilon_volumes(
        {name: epsilon for name in ("Ex", "Ey", "Ez")},
        {name: inverse for name in ("Ex", "Ey", "Ez")}, tensor_rows)
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        real = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
        getattr(fields, name)[...] = (
            real + 1j * rng.uniform(-0.4, 0.4, shape).astype(np.float32)
            if complex_storage else real)
    folded_axes = {"XYZ".index(axis) for axis in axes}
    thickness = tuple(
        (0, 2) if axis in folded_axes else (2, 2)
        for axis in range(3)) if active_pml else 0
    return fields, PML(grid=grid, thickness=thickness)


@pytest.fixture
def backend_ready(monkeypatch):
    monkeypatch.setattr(folded_complex, "_metal_backend_reasons", lambda grid: [])


def test_source_keeps_folded_partner_ghost_then_complex_row_product_then_pml_tail():
    """Connections: x PHASE (kind 2), z NEGATE (kind 1); fold on y."""
    source = family.complex_folded_offdiag_source(
        (1, 1, 1, 1, 1, 1), (0, 3, 0), (0, 0, 0), (2, 0, 1), (0, 1, 0))
    assert source.count("[[buffer(") == family.BINDING_COUNT == 31
    # The partner pair's SAME-INDEX word takes eikna on the partner axis's plane 0
    # (ph[partner]); the shifted reads are plain wraps.
    assert "home_10 = (k == 0) ? -home_10 : home_10;" in source
    assert "home_20 = (i == 0) ? c_mul(home_20, ph[0]) : home_20;" in source
    assert "far_a_20 = (i == 0) ? c_mul(far_a_20, ph[0]) : far_a_20;" in source
    assert "down_10 = (k == 0)" not in source
    # The formed product takes conj(eikna) on the OWN axis's plane 0 (ph[own + 3]).
    assert ("near_product_00 = (i == 0) ? c_mul(near_product_00, ph[3]) : "
            "near_product_00;") in source
    assert "near_product_20 = (k == 0) ? -near_product_20 : near_product_20;" in source
    assert "nxi - 1) ? c_mul(" not in source, "the far product is not phased"
    assert "down_00 = (at_y ? c_mul_coefficient_left(-1.0f, down_00) : down_00);" in source
    assert "near_product_00 = c_mul_field_left(near_00, u01[ii]);" in source
    # The row per part, in software binary64, rounded once per part.
    assert "    ulong od_re0 = sd_widen_f(od_d0.x, true);" in source
    assert "    ulong od_im0 = sd_widen_f(od_d0.y, true);" in source
    assert ("    float2 src0 = float2(sd_narrow_f(od_re0, true), "
            "sd_narrow_f(od_im0, true));") in source
    assert "c_mul_coefficient_left(0.25f" not in source
    # The PML tail: complex_fields' per-chunk select; the record, then the presence.
    assert (f"constant ChunkBounds& cb [[buffer({family.RECORD_BUFFER})]]"
            in source)
    assert (f"constant OffdiagPresence& kw [[buffer({family.PRESENCE_BUFFER})]]"
            in source)
    assert family.PRESENCE_BUFFER == family.BINDING_COUNT - 1
    assert family.RECORD_BUFFER == family.BINDING_COUNT - 2
    assert shaders.constitutive_bounds_block("E") in source
    for t in range(3):
        assert f"float2 prev{t} = dw{t} ? w{t}[ii] : float2(0.0f, 0.0f);" in source
        assert f"if (dw{t}) {{ w{t}[ii] = src{t}; }}" in source
        assert (f"if (dw{t}) {{ a{t} = f{t}[ii] + (c_mul_coefficient_left(kp_{t}, "
                f"src{t}) - c_mul_coefficient_left(km_{t}, prev{t})); }}") in source
    # The previous grouping, ((f + kps*src) - kms*prev), is gone.
    assert "a0 = a0 + c_mul_coefficient_left(kp_0, src0);" not in source
    assert "a0 = a0 - c_mul_coefficient_left(km_0, prev0);" not in source


def test_a_connection_needs_an_unfolded_periodic_axis_and_a_known_kind():
    with pytest.raises(ValueError, match="unfolded periodic axis"):
        family.complex_folded_offdiag_source(
            (1, 0, 0, 0, 0, 0), (0, 3, 0), (0, 0, 0), (0, 2, 0), (0, 1, 0))
    with pytest.raises(ValueError, match="connection kinds"):
        family.complex_folded_offdiag_source(
            (1, 0, 0, 0, 0, 0), (0, 3, 0), (0, 0, 0), (3, 0, 0), (0, 1, 0))


def _record(fields, pml):
    return chunk_map.chunk_bounds_record(chunk_map.ChunkLayout.one_rank(fields.grid, pml))


def test_folded_complex_tensor_product_builds_without_compiling_a_device_shader(
        backend_ready, monkeypatch):
    """The builder's structural contract: the predicate admits, the plan builds."""
    fields, pml = build()
    plan = family.plan_metal_complex_folded_offdiag(
        fields, pml, HostResidency(), contract_variants=(),
        chunk_bounds=_record(fields, pml))
    assert plan is not None
    assert plan.variants == ()
    assert plan.row_mask == (1, 0, 1, 0, 1, 0)
    assert plan.codes[1] in family._folded_real.MIRROR_CODES
    assert len(plan._args) == family.BINDING_COUNT


@pytest.mark.parametrize("mutation, needle", [
    ("no_fold", "no mirror plane"),
    ("no_rows", "no live off-diagonal"),
    ("inactive_pml", "no active PML"),
    ("real", "storage is real float32"),
])
def test_distinct_products_are_refused_by_name(backend_ready, mutation, needle):
    fields, pml = build(
        axes="" if mutation == "no_fold" else "Y",
        rows=mutation != "no_rows",
        active_pml=mutation != "inactive_pml",
        complex_storage=mutation != "real",
    )
    verdict = family.metal_complex_folded_offdiag_coverage(
        fields, pml, HostResidency())
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("k_point", ((0.0, 0.0, 0.0), (0.2, 0.0, 0.125)),
                         ids=("unphased", "bloch"))
def test_an_otherwise_admitted_run_is_admitted_with_or_without_a_bloch_phase(
        backend_ready, k_point):
    """No ruling refuses the row any more, and the Bloch connection is carried."""
    fields, pml = build(k_point=k_point)
    verdict = family.metal_complex_folded_offdiag_coverage(
        fields, pml, HostResidency(), _record(fields, pml))
    assert verdict.covered, verdict.reasons
    plan = family.plan_metal_complex_folded_offdiag(
        fields, pml, HostResidency(), contract_variants=(),
        chunk_bounds=_record(fields, pml))
    assert plan is not None
    assert plan.presence == offdiag.ALL_KEPT
    if any(k_point):
        assert any(kind != 0 for kind in plan.phased), plan.phased


def test_the_plan_refuses_a_missing_record_by_name(backend_ready):
    fields, pml = build()
    with pytest.raises(chunk_map.ChunkLayoutRefusal, match="no chunk record"):
        family.plan_metal_complex_folded_offdiag(
            fields, pml, HostResidency(), contract_variants=(),
            chunk_bounds=None)


def test_arm_is_registered_and_wired():
    from meep_gpu.metal_kernels import registry
    import meep_gpu.metal_kernels as package

    arm = next(spec for spec in arms.registered(family.SLOT) if spec.family == family.FAMILY)
    assert arm.label == family.LABEL
    assert arm.wired
    assert family.FAMILY in registry.FAMILY_MODULES
    assert package.complex_folded_offdiag_update_e is family
