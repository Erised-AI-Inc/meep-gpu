"""Laptop merge bar for the folded + off-diagonal + dispersive ``update_E``.

The compiled-kernel byte verdict belongs to the CUDA gate.  These tests pin the
array-path transcription, predicate disjointness, ordered ``D-P`` arithmetic,
folded Yee reads, PML recurrence and live polarization-pointer contract without
requiring Triton or a CUDA device.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from . import stepping
from .dispersion import PolarizationState, Susceptibility
from .fields import Fields, IYEE_SHIFTS, mirror_parity
from .grid import Grid, Mirror
from .pml import PML
from .triton_kernels import coverage
from .triton_kernels import dispersive_update_e
from .triton_kernels import folded_dispersive_update_e
from .triton_kernels import folded_offdiag_dispersive_update_e as module
from .triton_kernels import folded_offdiag_update_e
from .triton_kernels import offdiag_update_e
from .triton_kernels import symmetry

E_NAMES = ("Ex", "Ey", "Ez")
FW_NAMES = tuple("f_w_" + name for name in E_NAMES)
AXES = "xyz"


def _build(*, fold="X", poles=(2, 1, 2), rows=True, seed=41,
           complex_storage=False, pml_active=True, nonlinear=False,
           boundaries=None):
    grid = Grid(
        resolution=10.0,
        cell_size=(2.0, 1.2, 1.2),
        dimensions=3,
        courant=0.35,
        symmetry=tuple(Mirror(axis, 1) for axis in fold),
        boundaries=boundaries,
        xp=np,
    )
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    fields.enable_pml_storage()
    fields.enable_field_storage()
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = {name: np.full(shape, value, np.float32)
               for name, value in zip(E_NAMES, (2.0, 2.5, 3.0))}
    inverse = {name: np.full(shape, 1.0 / value, np.float32)
               for name, value in zip(E_NAMES, (2.0, 2.5, 3.0))}
    installed_rows = None
    if rows:
        installed_rows = {
            row: {partner: rng.uniform(-0.2, 0.2, shape).astype(np.float32)
                  for partner in E_NAMES if partner != row}
            for row in E_NAMES
        }
    fields.set_epsilon_volumes(
        epsilon, inverse, chi1inv_offdiagonal=installed_rows)

    for pole_index in range(max(poles) if poles else 0):
        susceptibility = Susceptibility(
            frequency=0.8 + 0.1 * pole_index,
            gamma=0.07 + 0.01 * pole_index,
            kind="lorentzian",
        )
        sigma = {name: (0.2 + 0.03 * pole_index)
                 if poles[index] > pole_index else 0.0
                 for index, name in enumerate(E_NAMES)}
        fields.polarizations.append(
            PolarizationState(susceptibility, sigma, grid, np.float32))

    dtype = np.complex64 if complex_storage else np.float32
    for name in E_NAMES + FW_NAMES + ("Dx", "Dy", "Dz"):
        array = getattr(fields, name)
        real = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
        if complex_storage:
            imag = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
            array[...] = real + 1j * imag
        else:
            array[...] = real.astype(dtype)
    for state in fields.polarizations:
        for slot in ("P", "P_prev"):
            for array in getattr(state, slot).values():
                array[...] = rng.uniform(-0.15, 0.15, shape).astype(array.dtype)
    if nonlinear:
        fields.set_nonlinear_volumes({"Ez": 0.04}, {"Ez": 0.02})

    thickness = tuple(
        (0, 2) if grid.is_mirrored(axis) else (2, 2)
        for axis in range(3))
    pml = PML(grid=grid, thickness=thickness if pml_active else 0)
    return fields, pml


def _face(axis, index):
    return (slice(None),) * axis + (index,)


def _d_minus_p(fields, reverse=False, drop=False):
    volumes = {}
    states = list(fields.polarizations)
    if reverse:
        states.reverse()
    if drop:
        states = []
    for name in E_NAMES:
        source = getattr(fields, "D" + name[1]).copy()
        for state in states:
            if state.drives(name):
                source -= state.P[name]
        volumes[name] = source
    return volumes


def _reference_update_e(fields, pml, state, *, reverse_poles=False,
                        drop_poles=False, mirror_down="parity",
                        integer_coefficients=False):
    """Independent transcription with mutations for discriminating controls."""
    grid = fields.grid
    kinds = tuple(stepping._boundary_kinds(grid, pml))
    phases = stepping._mirror_phases(grid)
    walls = offdiag_update_e.wall_mask_axes(grid)
    volumes = _d_minus_p(fields, reverse=reverse_poles, drop=drop_poles)

    for own_axis, target in enumerate(E_NAMES):
        constitutive = volumes[target] * fields.inverse_epsilon_for(target)
        rows = fields.chi1inv_offdiagonal_for(target)
        total = None
        for offset in (1, 2):
            partner_axis = (own_axis + offset) % 3
            partner = "E" + AXES[partner_axis]
            coefficient = rows.get(partner)
            if coefficient is None:
                continue
            values = volumes[partner]
            shifted = np.roll(values, 1, axis=partner_axis)
            if kinds[partner_axis] == stepping.METALLIC:
                shifted[_face(partner_axis, 0)] = 0
            elif kinds[partner_axis] == stepping.MIRROR:
                if mirror_down == "zero":
                    shifted[_face(partner_axis, 0)] = 0
                elif mirror_down == "wrap":
                    pass
                else:
                    weight = mirror_parity(
                        "D" + AXES[partner_axis], partner_axis,
                        int(phases[partner_axis]))
                    shifted[_face(partner_axis, 0)] = weight * values[
                        _face(partner_axis, module.MIRROR_SOURCE_INDEX)]
            product = (values + shifted) * coefficient
            shifted_up = np.roll(product, -1, axis=own_axis)
            if kinds[own_axis] in (stepping.METALLIC, stepping.MIRROR):
                shifted_up[_face(own_axis, -1)] = 0
            term = 0.25 * (product + shifted_up)
            total = term if total is None else total + term
        if total is not None:
            for axis in range(3):
                if IYEE_SHIFTS[target][axis] == 0 and walls[axis]:
                    total[_face(axis, 0)] = 0
            constitutive = constitutive + total

        previous = state["f_w_" + target].copy()
        state["f_w_" + target][...] = constitutive
        suffix = "" if integer_coefficients else "_h"
        state[target] += getattr(
            pml, f"kps_{AXES[own_axis]}{suffix}") * constitutive
        state[target] -= getattr(
            pml, f"kms_{AXES[own_axis]}{suffix}") * previous


def _state(fields):
    return {name: getattr(fields, name).copy() for name in E_NAMES + FW_NAMES}


def _state_bytes(state):
    return b"".join(state[name].tobytes() for name in E_NAMES + FW_NAMES)


def _field_bytes(fields):
    return b"".join(getattr(fields, name).tobytes()
                    for name in E_NAMES + FW_NAMES)


def _reasons(fields, pml):
    return module.folded_offdiag_dispersive_constitutive_coverage(
        fields, pml).reasons


def _non_backend_reasons(fields, pml):
    return tuple(reason for reason in _reasons(fields, pml)
                 if "not cupy" not in reason)


def test_reference_matches_stepping_bytewise_and_moves_state():
    fields, pml = _build()
    reference = _state(fields)
    before = _field_bytes(fields)
    for cycle in range(3):
        _reference_update_e(fields, pml, reference)
        stepping.update_E(fields, pml)
        assert _state_bytes(reference) == _field_bytes(fields), cycle
    assert _field_bytes(fields) != before


@pytest.mark.parametrize(
    "mutation", ["drop_poles", "reverse_poles", "zero_ghost", "integer_pml"])
def test_reference_controls_are_discriminating(mutation):
    fields, pml = _build()
    reference = _state(fields)
    kwargs = {
        "drop_poles": mutation == "drop_poles",
        "reverse_poles": mutation == "reverse_poles",
        "mirror_down": "zero" if mutation == "zero_ghost" else "parity",
        "integer_coefficients": mutation == "integer_pml",
    }
    _reference_update_e(fields, pml, reference, **kwargs)
    stepping.update_E(fields, pml)
    assert _state_bytes(reference) != _field_bytes(fields), (
        f"{mutation} changed no output; its CUDA mutation would be vacuous")


def test_reference_reads_d_minus_p_in_partner_stencil_not_only_diagonal():
    fields, pml = _build()
    # Kill the diagonal source for Ex while leaving Ey/Ez poles and Ex's tensor
    # partners live.  A kernel subtracting P only from gs0 would miss this.
    fields.Dx[...] = 0
    for state in fields.polarizations:
        if state.drives("Ex"):
            state.P["Ex"][...] = 0
    correct = _state(fields)
    wrong = _state(fields)
    _reference_update_e(fields, pml, correct)

    saved = []
    for state in fields.polarizations:
        for name in ("Ey", "Ez"):
            if state.drives(name):
                saved.append((state, name, state.P[name]))
                state.P[name] = np.zeros_like(state.P[name])
    _reference_update_e(fields, pml, wrong)
    for state, name, array in saved:
        state.P[name] = array
    assert correct["Ex"].tobytes() != wrong["Ex"].tobytes()


def test_live_binding_resolves_rotated_pointers_after_ade_update():
    fields, _pml = _build()
    binding = dispersive_update_e.LivePoleBinding(
        fields, dispersive_update_e.poles_per_component(fields))
    before = binding.arrays()
    before_ids = tuple(tuple(id(array) for array in group) for group in before)
    for state in fields.polarizations:
        state.update(lambda name: getattr(fields, "f_w_" + name), fields.grid.dt)
    after = binding.arrays()
    after_ids = tuple(tuple(id(array) for array in group) for group in after)
    assert after_ids != before_ids
    for component_index, name in enumerate(E_NAMES):
        expected = tuple(state.P[name] for state in fields.polarizations
                         if state.drives(name))
        assert after[component_index] == expected


def test_live_binding_rejects_a_changed_pole_order():
    fields, _pml = _build()
    binding = dispersive_update_e.LivePoleBinding(
        fields, dispersive_update_e.poles_per_component(fields))
    fields.polarizations.reverse()
    with pytest.raises(RuntimeError, match="ORDER"):
        binding.arrays()


def test_target_predicate_admits_except_for_the_numpy_backend():
    fields, pml = _build()
    assert _non_backend_reasons(fields, pml) == ()
    assert any("not cupy" in reason for reason in _reasons(fields, pml))


def test_both_incumbents_refuse_the_intersection_by_name():
    fields, pml = _build()
    assert any("susceptibility is registered" in reason for reason in
               folded_offdiag_update_e.folded_offdiag_constitutive_coverage(
                   fields, pml).reasons)
    assert any("off-diagonal chi1inv row" in reason for reason in
               folded_dispersive_update_e.folded_dispersive_constitutive_coverage(
                   fields, pml).reasons)


def test_no_pole_no_row_and_no_fold_are_disjoint_refusals():
    fields, pml = _build(poles=(0, 0, 0))
    assert any("no susceptibility" in reason for reason in _reasons(fields, pml))

    fields, pml = _build(rows=False)
    assert any("no off-diagonal" in reason for reason in _reasons(fields, pml))

    fields, pml = _build(fold="")
    assert any("no mirror plane" in reason for reason in _reasons(fields, pml))


def test_malformed_and_excess_pole_storage_is_refused_fail_closed():
    fields, pml = _build()
    fields.polarizations[0].P_prev["Ez"] = \
        fields.polarizations[0].P_prev["Ez"][::-1]
    assert any("P_prev[Ez]" in reason and "C-contiguous" in reason
               for reason in _reasons(fields, pml))

    fields, pml = _build(poles=(dispersive_update_e.MAX_POLES + 1, 1, 1))
    assert any(f"MAX_POLES={dispersive_update_e.MAX_POLES}" in reason
               for reason in _reasons(fields, pml))


def test_unsupported_intersections_remain_refused():
    fields, pml = _build(complex_storage=True)
    assert any("force_complex_fields" in reason for reason in _reasons(fields, pml))

    fields, pml = _build(nonlinear=True)
    assert any("chi2/chi3" in reason for reason in _reasons(fields, pml))

    fields, pml = _build(pml_active=False)
    assert any("no active PML" in reason for reason in _reasons(fields, pml))

    fields, pml = _build()
    fields.polarizations.append(SimpleNamespace(
        susceptibility=SimpleNamespace(kind="debye"),
        driven=lambda: ("Ex",), drives=lambda name: name == "Ex",
        P={"Ex": fields.Dx.copy()}, P_prev={"Ex": fields.Dx.copy()}))
    assert any("outside" in reason and "debye" in reason
               for reason in _reasons(fields, pml))


def test_optional_import_contract_and_kernel_error_are_diagnosable():
    fields, pml = _build()
    assert isinstance(module.explain_folded_offdiag_dispersive(
        fields, pml).covered, bool)
    if module.folded_offdiag_dispersive_constitutive_step is None:
        with pytest.raises(ImportError, match="triton"):
            module.folded_offdiag_dispersive_constitutive_step_kernel()


def test_shared_helpers_and_restatements_have_not_drifted():
    lookup = {
        "coverage": coverage,
        "folded_offdiag_update_e": folded_offdiag_update_e,
        "offdiag_update_e": offdiag_update_e,
        "symmetry": symmetry,
        "dispersive_update_e": dispersive_update_e,
    }
    for entry in module.SHARED_CLAUSES:
        owner, name = entry.split(".")
        assert hasattr(lookup[owner], name), entry
    assert module.E_TERMS is dispersive_update_e.E_TERMS
    assert module.ROW_SLOTS is offdiag_update_e.ROW_SLOTS
    assert module.MIRROR_SOURCE_INDEX == stepping.MIRROR_SOURCE_INDEX
    assert module.MIRROR_CODES == folded_offdiag_update_e.MIRROR_CODES


def test_engine_builder_binds_raw_d_and_live_poles_not_materialized_dmp():
    source = inspect.getsource(
        module.plan_folded_offdiag_dispersive_constitutive)
    assert "displacement_minus_polarization_volumes" not in source
    assert "LivePoleBinding" in source
    assert "getattr(fields, term[1])" in source


def test_kernel_source_carries_all_three_arithmetic_layers():
    text = inspect.getsource(module)
    assert "value = value - tl.load(p7" in text
    assert "_folded_dispersive_term" in text
    assert "MIRROR_ROW" in text
    assert "value0 = value0 + kp_0 * src0" in text
    assert "value0 = value0 - km_0 * prev0" in text
    assert "NP0=self.counts[0]" in text


def test_central_wiring_spec_names_the_row_product_veto():
    text = inspect.getsource(module)
    assert "folded off-diagonal dispersive" in text
    assert "ROW_PRODUCT_ARMS" in text
    assert "_veto_dropped_offdiagonal_coupling" in text
    assert "launch.FAMILY_MODULES" in text
