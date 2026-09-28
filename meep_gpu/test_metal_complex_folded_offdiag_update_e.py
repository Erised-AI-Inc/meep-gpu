"""Host contracts for Metal's complex folded tensor-row PML ``update_E`` arm."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytest

from meep_gpu.fields import Fields
from meep_gpu.grid import Grid, Mirror
from meep_gpu.metal_kernels import arms
from meep_gpu.metal_kernels import complex_folded_offdiag_update_e as family
from meep_gpu.metal_kernels import folded_complex
from meep_gpu.pml import PML


class HostResidency:
    """Array-preserving mirror double: these contracts never compile Metal."""

    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        prior = self.arrays.get(name)
        if prior is not None and prior is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


PROBE = {"backend": "numpy", "patterns": {
    name: "FMA_V1" for name in folded_complex.PARITY_PROBE_PATTERNS}}


def build(*, axes: str = "Y", phases: Sequence[int] = (1,), rows: bool = True,
          active_pml: bool = True, complex_storage: bool = True,
          seed: int = 17081) -> Tuple[Fields, PML]:
    grid = Grid(
        resolution=8.0,
        cell_size=(2.0, 1.5, 1.25),
        courant=0.35,
        boundaries="periodic",
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
    source = family.complex_folded_offdiag_source(
        (1, 1, 1, 1, 1, 1), (0, 3, 0), (0, 0, 0), (1, 0, 1), (0, 1, 0),
        "FMA_V1")
    assert source.count("[[buffer(") == family.BINDING_COUNT == 29
    assert "down_10 = (k == 0) ? c_mul(down_10, ph[2]) : down_10;" in source
    assert "down_00 = (at_y ? c_mul_coefficient_left(-1.0f, down_00) : down_00);" in source
    assert "near_product_00 = c_mul_field_left(near_00, u01[ii]);" in source
    assert "far_product_00 = (i == nxi - 1) ? c_mul(far_product_00, ph[3]) : far_product_00;" in source
    assert "a0 = a0 + c_mul_coefficient_left(kp_0, src0);" in source
    assert "a0 = a0 - c_mul_coefficient_left(km_0, prev0);" in source


def test_folded_complex_tensor_product_builds_without_compiling_a_device_shader(backend_ready):
    fields, pml = build()
    plan = family.plan_metal_complex_folded_offdiag(
        fields, pml, HostResidency(), contract_variants=(), probe=PROBE)
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
    ("missing_probe", "probe artifact"),
])
def test_distinct_products_are_refused_by_name(backend_ready, mutation, needle):
    fields, pml = build(
        axes="" if mutation == "no_fold" else "Y",
        rows=mutation != "no_rows",
        active_pml=mutation != "inactive_pml",
        complex_storage=mutation != "real",
    )
    verdict = family.metal_complex_folded_offdiag_coverage(
        fields, pml, HostResidency(), {} if mutation == "missing_probe" else PROBE)
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_arm_is_wired_in_the_experimental_composer_only():
    from meep_gpu.metal_kernels import registry
    import meep_gpu.metal_kernels as package

    arm = next(spec for spec in arms.registered(family.SLOT) if spec.family == family.FAMILY)
    assert arm.label == family.LABEL
    assert arm.wired
    assert family.FAMILY in registry.FAMILY_MODULES
    assert package.complex_folded_offdiag_update_e is family
