"""Host contract for Metal's complex/Bloch no-PML tensor-row ``update_E``."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from meep_gpu import stepping
from meep_gpu.fields import Fields
from meep_gpu.grid import Grid
from meep_gpu.metal_kernels import arms, shaders
from meep_gpu.metal_kernels import complex_no_pml_offdiag_update_e as family
from meep_gpu.metal_kernels import complex_no_pml_conductive as shared
from meep_gpu.pml import PML


class HostResidency:
    def __init__(self) -> None:
        self.arrays: Dict[str, Any] = {}

    def mirror(self, name, host, constant=False, dtype=None):  # noqa: ARG002
        previous = self.arrays.get(name)
        if previous is not None and previous is not host:
            raise ValueError(f"{name} rebound")
        self.arrays[name] = host
        return host


def probe():
    return {"backend": "numpy", "patterns": {
        name: "FMA_V1" for name in (
            "c8_mul_c8", "c8_mul_c8_scalar_right", "c8_mul_f4_field_left",
            "f4_mul_c8_coefficient_left", "python_float_left")}}


def build(*, rows=True, active_pml=False, complex_storage=True, bloch=True, seed=137):
    grid = Grid(resolution=8.0, cell_size=(1.125, 0.875, 0.75), courant=0.35,
                boundaries="periodic",
                k_point=(0.2, -0.125, 0.0) if bloch else (0.0, 0.0, 0.0), xp=np)
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


def _reference_function(reference, reference_pml, actual):
    def run(*args):
        targets, sources, inverse, rows = args[:3], args[3:6], args[6:9], args[9:15]
        assert targets == (actual.Ex, actual.Ey, actual.Ez)
        assert sources == (actual.Dx, actual.Dy, actual.Dz)
        assert inverse == tuple(actual.inverse_epsilon_for(name) for name in ("Ex", "Ey", "Ez"))
        assert rows[0] is actual.chi1inv_offdiagonal_for("Ex")["Ey"]
        # The final six scalar arguments are the conjugated-down and forward-up
        # phase pairs; their separate positions are the operation this family owns.
        assert len(args[-6:]) == 6
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


def test_source_keeps_partner_down_phase_before_product_and_own_up_phase_after_it():
    source = family.complex_no_pml_offdiag_source(
        (1, 0, 0, 0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 1, 0), "FMA_V1")
    assert source.count("[[buffer(") == family.BINDING_COUNT
    assert "down_00 = (j == 0) ? c_mul(down_00, dpy) : down_00;" in source
    assert "near_product_00 = c_mul_field_left(near_00, u01[ii]);" in source
    assert "far_product_00 = (i == nxi - 1) ? c_mul(far_product_00, upx) : far_product_00;" in source
    assert "f0[ii] = src0;" in source
    assert "f_w_Ex" not in source


def test_complex_bloch_tensor_row_is_admitted_and_matches_array_update(backend_ready):
    reference, reference_pml = build()
    actual, actual_pml = build()
    plan = family.plan_metal_complex_no_pml_offdiag(
        actual, actual_pml, HostResidency(), probe=probe(),
        functions={shaders.CONTRACT_OFF: _reference_function(reference, reference_pml, actual)})
    assert plan is not None
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
        fields, pml, HostResidency(), probe())
    assert verdict.covered is False
    assert any("the per-axis boundary rule is unreadable" in reason
               for reason in verdict.reasons), verdict.reasons


@pytest.mark.parametrize("mutation, needle", [
    ("no_rows", "no live off-diagonal"),
    ("active_pml", "active PML"),
    ("real", "complex64 storage"),
    ("missing_probe", "probe artifact"),
])
def test_distinct_products_are_refused_by_name(backend_ready, mutation, needle):
    fields, pml = build(rows=mutation != "no_rows", active_pml=mutation == "active_pml")
    record = probe()
    if mutation == "real":
        fields, pml = build(complex_storage=False, bloch=False)
    elif mutation == "missing_probe":
        record = {}
    verdict = family.metal_complex_no_pml_offdiag_coverage(
        fields, pml, HostResidency(), record)
    assert not verdict.covered
    assert any(needle in reason for reason in verdict.reasons), verdict.reasons


def test_arm_is_wired_and_visible_to_package_registry():
    from meep_gpu.metal_kernels import registry
    import meep_gpu.metal_kernels as package

    arm = next(spec for spec in arms.registered(family.SLOT) if spec.family == family.FAMILY)
    assert arm.label == "complex no-PML off-diagonal"
    assert arm.wired
    assert family.FAMILY in registry.FAMILY_MODULES
    assert package.complex_no_pml_offdiag_update_e is family
