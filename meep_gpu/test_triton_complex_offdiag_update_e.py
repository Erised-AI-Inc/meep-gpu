"""Laptop merge bar for the complex off-diagonal ``update_E`` Triton family.

The compiled body is a CUDA-gate question.  This suite pins the independently
settleable contract: the two tails partition active/inactive PML, both require a
surviving tensor row and complex storage, their builders bind the correct Yee
tables and phase directions, and a direct transcription of the shared row core
is bit-identical to :func:`stepping.update_E` on the two residual shapes.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from . import stepping
from .fields import Fields, IYEE_SHIFTS, mirror_parity
from .grid import Grid, Mirror
from .pml import PML
from .test_triton_complex_fields import _probe_record
from .triton_kernels import complex_fields
from .triton_kernels import complex_offdiag_update_e as family
from .triton_kernels import folded_complex
from .triton_kernels import offdiag_update_e


E = ("Ex", "Ey", "Ez")
D = ("Dx", "Dy", "Dz")
AXES = "xyz"
SEED = 20260817


class _CupyNamed:
    """NumPy operations with the backend name used by host-only predicates."""

    __name__ = "cupy"

    def __getattr__(self, name):
        return getattr(np, name)


@pytest.fixture(scope="module", autouse=True)
def _keep_policy():
    from .subnormal_policy import install_subnormal_policy

    install_subnormal_policy("keep")


def _build(*, folded=False, active_pml=False, dimensions=3,
           k_point=(0.23, -0.17, 0.35), seed=SEED):
    symmetry = (Mirror("Y", -1),) if folded else ()
    cell = (2.0, 2.0, 1.2) if folded else (
        (2.5, 2.5, 0.0) if dimensions == 2 else (2.5, 2.5, 2.5))
    grid = Grid(resolution=8.0, cell_size=cell, dimensions=dimensions,
                boundaries="periodic", symmetry=symmetry, k_point=k_point,
                xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    shape = tuple(grid.shape)
    rng = np.random.default_rng(seed)
    eps = {name: np.full(shape, value, np.float32)
           for name, value in zip(E, (2.0, 2.5, 3.0))}
    inv = {name: np.full(shape, np.float32(1.0 / value), np.float32)
           for name, value in zip(E, (2.0, 2.5, 3.0))}
    rows = {
        "Ex": {"Ey": rng.uniform(-0.19, 0.17, shape).astype(np.float32)},
        "Ey": {"Ez": rng.uniform(-0.13, 0.21, shape).astype(np.float32)},
        "Ez": {"Ex": rng.uniform(-0.23, 0.11, shape).astype(np.float32)},
    }
    fields.set_epsilon_volumes(eps, inv, chi1inv_offdiagonal=rows)
    if active_pml:
        fields.enable_pml_storage()
    else:
        fields.enable_field_storage()
    pml = PML(grid=grid, thickness=((0, 2), (0, 2), (0, 0))
              if active_pml else 0)
    for name in E + D:
        a = getattr(fields, name)
        a.real = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
        a.imag = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
    for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez"):
        a = getattr(fields, name, None)
        if a is not None:
            a.real = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
            a.imag = rng.uniform(-0.5, 0.5, shape).astype(np.float32)
    grid.xp = _CupyNamed()
    return fields, pml


def _probe():
    record = _probe_record()
    name = folded_complex.PARITY_PROBE_PATTERN
    record["patterns"][name] = "FMA_V1"
    record["vectors"][name] = 2792
    record["detail"][name] = dict(record["detail"]["c8_mul_c8"])
    return record


def _strip_backend(reasons):
    return tuple(r for r in reasons if "not cupy" not in r)


@pytest.mark.parametrize("dimensions,k", [
    (3, (0.23, -0.17, 0.35)),
    (2, (0.3892, 0.1597, 0.0)),
])
def test_no_pml_tail_admits_the_two_group_j_rows(dimensions, k):
    fields, pml = _build(dimensions=dimensions, k_point=k)
    verdict = family.complex_no_pml_offdiag_update_e_coverage(
        fields, pml, probe=_probe())
    assert verdict.covered, verdict.reasons
    assert family.plan_complex_no_pml_offdiag_update_e(
        fields, pml, probe=_probe()) is not None


def test_folded_pml_tail_admits_group_d_and_incumbent_still_refuses_e():
    fields, pml = _build(folded=True, active_pml=True,
                         k_point=(0.35, 0.0, 0.0))
    verdict = family.complex_folded_offdiag_update_e_coverage(
        fields, pml, probe=_probe())
    assert verdict.covered, verdict.reasons
    plan = family.plan_complex_folded_offdiag_update_e(
        fields, pml, probe=_probe())
    assert plan is not None
    assert plan.pml is True
    assert plan.mirror_axes == (0, 1, 0)
    assert plan.phase_flags == (1, 0, 0)
    incumbent = folded_complex.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "E", probe=_probe())
    assert not incumbent.covered
    assert any("needs a complex folded off-diagonal kernel" in r
               for r in incumbent.reasons)


def test_the_two_tails_are_disjoint_on_the_array_paths_branch():
    plain, inert = _build(active_pml=False)
    assert family.complex_no_pml_offdiag_update_e_coverage(
        plain, inert, probe=_probe()).covered
    assert not family.complex_folded_offdiag_update_e_coverage(
        plain, inert, probe=_probe()).covered

    folded, active = _build(folded=True, active_pml=True,
                            k_point=(0.35, 0.0, 0.0))
    assert family.complex_folded_offdiag_update_e_coverage(
        folded, active, probe=_probe()).covered
    assert not family.complex_no_pml_offdiag_update_e_coverage(
        folded, active, probe=_probe()).covered


@pytest.mark.parametrize("mutation,needle", [
    ("real", "storage is real"),
    ("no_rows", "off-diagonal"),
    ("poles", "susceptibility"),
    ("nonlinear", "chi2/chi3"),
])
def test_adjacent_families_are_refused_by_name(mutation, needle):
    fields, pml = _build(k_point=((0.0, 0.0, 0.0)
                                  if mutation == "real"
                                  else (0.23, -0.17, 0.35)))
    if mutation == "real":
        fields.force_complex_fields = False
    elif mutation == "no_rows":
        fields._chi1inv_offdiagonal = {}
    elif mutation == "poles":
        fields.polarizations = [SimpleNamespace(
            driven=lambda: ("Ex",), drives=lambda name: name == "Ex",
            susceptibility=SimpleNamespace(kind="lorentzian"))]
    else:
        fields.set_nonlinear_volumes({"Ex": np.float32(0.03)}, {})
    reasons = family.complex_no_pml_offdiag_update_e_coverage(
        fields, pml, probe=_probe()).reasons
    assert any(needle.lower() in reason.lower() for reason in reasons), reasons


def test_row_and_yee_tables_are_the_engine_tables():
    assert family.E_TERMS is offdiag_update_e.E_TERMS
    assert family.ROW_SLOTS is offdiag_update_e.ROW_SLOTS
    for row_index, (component, _source, own_axis) in enumerate(family.E_TERMS):
        assert own_axis == row_index
        assert IYEE_SHIFTS[component][own_axis] == 1
        assert family.WALL_MASK_AXES[row_index] == tuple(
            axis for axis in range(3) if IYEE_SHIFTS[component][axis] == 0)


def test_phase_arguments_carry_both_shift_directions_without_device_negation():
    fields, _ = _build(k_point=(0.23, -0.17, 0.35))
    phases = complex_fields.bloch_phase_table(
        fields.grid, stepping._boundary_kinds(fields.grid, None))
    flags_f, forward = complex_fields._phase_arguments(phases, backward=False)
    flags_b, backward = complex_fields._phase_arguments(phases, backward=True)
    assert flags_f == flags_b == (1, 1, 1)
    for axis in range(3):
        assert forward[2 * axis] == backward[2 * axis]
        assert forward[2 * axis + 1] == -backward[2 * axis + 1]


def test_mirror_weight_uses_the_partner_displacement_parity():
    fields, _ = _build(folded=True, active_pml=True,
                       k_point=(0.35, 0.0, 0.0))
    weights = family.mirror_ghost_weights(fields.grid)
    assert weights[1] == float(mirror_parity("Dy", 1, -1))


def _face(axis, index):
    return (slice(None),) * axis + (index,)


def _reference(fields, pml, state, *, coefficient_after_shift=False,
               pml_tail=None):
    """Direct transcription with mutation knobs used for non-vacuity controls."""
    if pml_tail is None:
        pml_tail = bool(pml is not None and pml.is_active)
    grid = fields.grid
    kinds = stepping._boundary_kinds(grid, pml if pml_tail else None)
    phases = stepping._bloch_phases(grid, kinds, state["Dx"], pml if pml_tail else None)
    mirror_phases = stepping._mirror_phases(grid)
    walls = offdiag_update_e.wall_mask_axes(grid)
    volumes = {e: state[d] for e, d in zip(E, D)}
    for own, target in enumerate(E):
        diagonal = volumes[target] * fields.inverse_epsilon_for(target)
        total = None
        rows = fields.chi1inv_offdiagonal_for(target)
        for offset in (1, 2):
            partner_axis = (own + offset) % 3
            partner = E[partner_axis]
            u = rows.get(partner)
            if u is None:
                continue
            g = volumes[partner]
            down = np.roll(g, 1, axis=partner_axis)
            if kinds[partner_axis] == stepping.PERIODIC:
                if phases[partner_axis] is not None:
                    down[_face(partner_axis, 0)] *= np.conj(
                        np.complex64(phases[partner_axis]))
            elif kinds[partner_axis] == stepping.METALLIC:
                down[_face(partner_axis, 0)] = 0
            else:
                weight = mirror_parity("D" + AXES[partner_axis], partner_axis,
                                       int(mirror_phases[partner_axis]))
                down[_face(partner_axis, 0)] = weight * g[
                    _face(partner_axis, stepping.MIRROR_SOURCE_INDEX)]
            if coefficient_after_shift:
                product = (g + down)
            else:
                product = (g + down) * u
            up = np.roll(product, -1, axis=own)
            if kinds[own] == stepping.PERIODIC:
                if phases[own] is not None:
                    up[_face(own, -1)] *= np.complex64(phases[own])
            else:
                up[_face(own, -1)] = 0
            term = 0.25 * (product + up)
            if coefficient_after_shift:
                term *= u
            total = term if total is None else total + term
        source = diagonal
        if total is not None:
            for axis in range(3):
                if walls[axis] and IYEE_SHIFTS[target][axis] == 0:
                    total[_face(axis, 0)] = 0
            source = source + total
        if pml_tail:
            previous = state["f_w_" + target].copy()
            state["f_w_" + target][...] = source
            state[target] += getattr(pml, f"kps_{AXES[own]}_h") * source
            state[target] -= getattr(pml, f"kms_{AXES[own]}_h") * previous
        else:
            state[target][...] = source


def _snapshot(fields, pml_tail):
    names = E + D + (("f_w_Ex", "f_w_Ey", "f_w_Ez") if pml_tail else ())
    return {name: getattr(fields, name).copy() for name in names}


def _word_bytes(state, pml_tail):
    names = E + (("f_w_Ex", "f_w_Ey", "f_w_Ez") if pml_tail else ())
    return b"".join(np.asarray(state[name]).view(np.uint32).tobytes()
                    for name in names)


@pytest.mark.parametrize("folded,pml_tail,k", [
    (False, False, (0.23, -0.17, 0.35)),
    (True, True, (0.35, 0.0, 0.0)),
])
def test_host_transcription_is_bit_identical_and_mutation_armed(folded, pml_tail, k):
    fields, pml = _build(folded=folded, active_pml=pml_tail, k_point=k)
    reference = _snapshot(fields, pml_tail)
    mutation = {name: value.copy() for name, value in reference.items()}
    for _ in range(3):
        stepping.update_E(fields, pml)
        _reference(fields, pml, reference, pml_tail=pml_tail)
        _reference(fields, pml, mutation, coefficient_after_shift=True,
                   pml_tail=pml_tail)
    assert _word_bytes(reference, pml_tail) == _word_bytes(
        {name: getattr(fields, name) for name in reference}, pml_tail)
    assert _word_bytes(mutation, pml_tail) != _word_bytes(reference, pml_tail)


def test_kernel_source_preserves_the_load_bearing_operand_orientations():
    source = inspect.getsource(family)
    assert "_mul_field_left" in source       # D/partner field LEFT, real row RIGHT
    assert "_mul_coefficient_left" in source  # PML/parity/0.25 coefficient LEFT
    assert "PML: tl.constexpr" in source
    assert "PHX" in source and "MG_X" in source
