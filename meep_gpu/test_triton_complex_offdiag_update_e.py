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

from . import part_products, readback, stepping
from .fields import Fields, IYEE_SHIFTS, mirror_parity
from .grid import Grid, Mirror
from .pml import PML
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


def _strip_backend(reasons):
    return tuple(r for r in reasons if "not cupy" not in r)


@pytest.mark.parametrize("dimensions,k", [
    (3, (0.23, -0.17, 0.35)),
    (2, (0.3892, 0.1597, 0.0)),
])
def test_no_pml_tail_admits_the_two_group_j_rows(dimensions, k):
    fields, pml = _build(dimensions=dimensions, k_point=k)
    verdict = family.complex_no_pml_offdiag_update_e_coverage(
        fields, pml)
    assert verdict.covered, verdict.reasons
    assert family.plan_complex_no_pml_offdiag_update_e(
        fields, pml) is not None


def test_folded_pml_tail_admits_group_d_and_incumbent_still_refuses_e():
    fields, pml = _build(folded=True, active_pml=True,
                         k_point=(0.35, 0.0, 0.0))
    verdict = family.complex_folded_offdiag_update_e_coverage(
        fields, pml)
    assert verdict.covered, verdict.reasons
    plan = family.plan_complex_folded_offdiag_update_e(
        fields, pml)
    assert plan is not None
    assert plan.pml is True
    assert plan.mirror_axes == (0, 1, 0)
    assert plan.connections == (complex_fields.CONNECTION_KINDS["phase"], 0, 0)
    incumbent = folded_complex.folded_complex_offdiag_constitutive_coverage(
        fields, pml, "E")
    assert not incumbent.covered
    assert any("needs a complex folded off-diagonal kernel" in r
               for r in incumbent.reasons)


def test_the_two_tails_are_disjoint_on_the_array_paths_branch():
    plain, inert = _build(active_pml=False)
    assert family.complex_no_pml_offdiag_update_e_coverage(
        plain, inert).covered
    assert not family.complex_folded_offdiag_update_e_coverage(
        plain, inert).covered

    folded, active = _build(folded=True, active_pml=True,
                            k_point=(0.35, 0.0, 0.0))
    assert family.complex_folded_offdiag_update_e_coverage(
        folded, active).covered
    assert not family.complex_no_pml_offdiag_update_e_coverage(
        folded, active).covered


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
        fields, pml).reasons
    assert any(needle.lower() in reason.lower() for reason in reasons), reasons


def test_row_and_yee_tables_are_the_engine_tables():
    assert family.E_TERMS is offdiag_update_e.E_TERMS
    assert family.ROW_SLOTS is offdiag_update_e.ROW_SLOTS
    for row_index, (component, _source, own_axis) in enumerate(family.E_TERMS):
        assert own_axis == row_index
        assert IYEE_SHIFTS[component][own_axis] == 1
        assert family.WALL_MASK_AXES[row_index] == tuple(
            axis for axis in range(3) if IYEE_SHIFTS[component][axis] == 0)


def test_the_two_connections_are_conjugates_and_the_device_negates_nothing():
    """The partner pair reads ABOVE the window (``eikna``), the near product BELOW it
    (``conj(eikna)``): two host-built factor sets, the second the first's conjugate
    word for word, and the same connection kind on every axis."""
    fields, pml = _build(k_point=(0.23, -0.17, 0.35))
    phases = complex_fields.bloch_phase_table(
        fields.grid, stepping._boundary_kinds(fields.grid, None))
    kinds_above, above = complex_fields.connection_arguments(phases, backward=True)
    kinds_below, below = complex_fields.connection_arguments(phases, backward=False)
    assert kinds_above == kinds_below == (complex_fields.CONNECTION_KINDS["phase"],) * 3
    for axis in range(3):
        assert above[2 * axis] == below[2 * axis]
        assert above[2 * axis + 1] == -below[2 * axis + 1]
    plan = family.plan_complex_no_pml_offdiag_update_e(fields, pml)
    assert plan is not None
    assert plan.above_values == above and plan.below_values == below


def test_mirror_weight_uses_the_partner_displacement_parity():
    fields, _ = _build(folded=True, active_pml=True,
                       k_point=(0.35, 0.0, 0.0))
    weights = family.mirror_ghost_weights(fields.grid)
    assert weights[1] == float(mirror_parity("Dy", 1, -1))


def _face(axis, index):
    return (slice(None),) * axis + (index,)


def _connected(volume, axis, phase, below):
    """Plane 0 of ``volume`` along ``axis`` as MEEP's not-owned word; a copy elsewhere.

    ``thephase = 1.0 * conj(eikna)`` (``below``) or ``1.0 * eikna``, in double, and the
    word connected by MEEP's rule (copy within 1e-13 of +1, negation of -1, else the
    product with every part rounded on its own): ``part_products.connect``.
    """
    out = volume.copy()
    if phase is None:
        return out
    eikna = complex(phase)
    factor = (eikna.real, -eikna.imag) if below else (eikna.real, eikna.imag)
    thephase = readback.cmul((1.0, 0.0), factor)
    out[_face(axis, 0)] = part_products.connect(volume[_face(axis, 0)], thephase)
    return out


def _reference(fields, pml, state, *, coefficient_after_shift=False,
               pml_tail=None):
    """Direct transcription of the row core, with a mutation knob for non-vacuity.

    MEEP's ``fw[i] = gs*us + OFFDIAG(u1, g1, s1) + OFFDIAG(u2, g2, s2)`` per stored part:
    ``gs*us`` one float32 product widened to double; each OFFDIAG the float32 pair
    ``g[i] + g[i-sx]`` (the unshifted operand's plane 0 connected ABOVE the window on a
    phased partner axis, the down neighbour a plain roll with the near-face ghost of
    its boundary), times ``u`` per part, then the product at this node (plane 0
    connected BELOW the window on a phased own axis) plus the product one node up (a
    plain roll, zero past a non-periodic top face), wall-masked, times the double
    literal 0.25; the row summed in double left to right and rounded once per part.
    The PML tail is the array path's own region forms (not this test's subject).
    """
    if pml_tail is None:
        pml_tail = bool(pml is not None and pml.is_active)
    grid = fields.grid
    kinds = stepping._boundary_kinds(grid, pml if pml_tail else None)
    phases = stepping._bloch_phases(grid, kinds, state["Dx"], pml if pml_tail else None)
    mirror_phases = stepping._mirror_phases(grid)
    walls = offdiag_update_e.wall_mask_axes(grid)
    volumes = {e: state[d].copy() for e, d in zip(E, D)}
    for own, target in enumerate(E):
        inverse = fields.inverse_epsilon_for(target)
        row = [(part * inverse).astype(np.float64)
               for part in (volumes[target].real, volumes[target].imag)]
        rows = fields.chi1inv_offdiagonal_for(target)
        for offset in (1, 2):
            partner_axis = (own + offset) % 3
            partner = E[partner_axis]
            u = rows.get(partner)
            if u is None:
                continue
            g = volumes[partner]
            here_g = (_connected(g, partner_axis, phases[partner_axis], below=False)
                      if kinds[partner_axis] == stepping.PERIODIC else g)
            down = np.roll(g, 1, axis=partner_axis)
            if kinds[partner_axis] == stepping.METALLIC:
                down[_face(partner_axis, 0)] = 0
            elif kinds[partner_axis] != stepping.PERIODIC:
                weight = np.float32(mirror_parity("D" + AXES[partner_axis], partner_axis,
                                                  int(mirror_phases[partner_axis])))
                source = g[_face(partner_axis, stepping.MIRROR_SOURCE_INDEX)]
                down.real[_face(partner_axis, 0)] = weight * source.real
                down.imag[_face(partner_axis, 0)] = weight * source.imag
            pair = here_g + down
            if coefficient_after_shift:
                product = pair
            else:
                product = np.empty_like(pair)
                product.real = pair.real * u
                product.imag = pair.imag * u
            here = (_connected(product, own, phases[own], below=True)
                    if kinds[own] == stepping.PERIODIC else product)
            up = np.roll(product, -1, axis=own)
            if kinds[own] != stepping.PERIODIC:
                up[_face(own, -1)] = 0
            for index, (low, high) in enumerate(((here.real, up.real),
                                                 (here.imag, up.imag))):
                summed = low + high
                if coefficient_after_shift:
                    summed = summed * u
                for axis in range(3):
                    if walls[axis] and IYEE_SHIFTS[target][axis] == 0:
                        summed[_face(axis, 0)] = 0
                row[index] = row[index] + 0.25 * summed.astype(np.float64)
        constitutive = np.empty(volumes[target].shape, np.complex64)
        constitutive.real = row[0].astype(np.float32)
        constitutive.imag = row[1].astype(np.float32)
        if pml_tail:
            plan = stepping._region_plan(fields, pml)
            kps, kms = stepping._constitutive_coefficients(pml, AXES[own], half_integer=True)
            stepping._apply_constitutive_regions(plan, target, state[target], constitutive,
                                                 kps, kms, state["f_w_" + target], np)
        else:
            state[target][...] = constitutive


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
    # MEEP's (g[i] + g[i-sx])*u[i]: the float32 pair LEFT, the real row RIGHT, per part.
    assert "near_re = (c_re + d_re) * u_c" in source
    assert "far_im = (p_im + q_im) * u_u" in source
    # The mirror ghost's weight LEFT of the stored word, exact (+1 or -1).
    assert "d_re = w_d * d_re" in source
    # Connections through the one helper; no whole-complex rotation remains.
    assert "_connect(near_re, near_im, obr, obi, CNO)" in source
    assert "_rotate_field_left" not in source and "_mul_field_left" not in source
    assert "PML: tl.constexpr" in source
    assert "CNX: tl.constexpr" in source and "MG_X" in source
    assert "tl.math.fma" not in source
