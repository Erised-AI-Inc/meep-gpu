"""The complex no-PML ``update_P`` text takes MEEP's recurrence one part at a time.

MEEP steps complex storage as two real passes (``DOCMP2``), and the array path forms
every product part by part (``part_products.scale``), so each part of P advances by

    gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s * w)))

on that part's words alone. The 0.9.2 complex form multiplied by zero-imaginary
complex coefficients, which carries the OTHER part's sign into an exactly-zero addend.

What runs here, on a host with no CuPy: the emitted ``ade_part`` helper, executed by
the root interpreter inside a probe kernel over every lane, against
``PolarizationState.update`` on complex64 volumes whose real planes are half signed
zeros (the class that separates the two forms); the 0.9.2 cross term, planted in the
same probe, must change a word there. The rest is text: ``ade_step`` is the two part
calls, and every complex kernel that advances P -- the stand-alone launch, the E->P
pair and the three-slot weld -- calls it with the four constants in MEEP's order.
"""

from __future__ import annotations

import re

import numpy
import pytest

from ..dispersion import PolarizationState, Susceptibility
from ..grid import Grid
from . import complex_no_pml_kernels as family
from .test_root_chunk_forms import F32, Interpreter

_ADE_PART = re.search(r"__device__ __forceinline__ float ade_part\(.*?\n\}\n",
                      family._NO_PML_HELPERS, re.S).group(0)  # noqa: SLF001

_PROBE = _ADE_PART + r'''
extern "C" __global__ void probe(float* out, const float* p, const float* q,
                                 const float* w, const float* s,
                                 float two_minus_od, float gamma1, float omega0dtsqr,
                                 float gamma1inv, int nx, int ny, int nz) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;
    out[idx] = ade_part(p[idx], q[idx], w[idx], s[idx], two_minus_od, gamma1,
                        omega0dtsqr, gamma1inv);
}
'''

#: The 0.9.2 real part: the drive term formed as a complex product with a
#: zero-imaginary coefficient, ``re = a*w.re - 0*w.im``.
_PACKAGE_REAL_PART = _PROBE.replace(
    "float* out, const float* p,", "float* out, const float* w_other, const float* p,"
).replace(
    "    out[idx] = ade_part(p[idx], q[idx], w[idx], s[idx], two_minus_od, gamma1,\n"
    "                        omega0dtsqr, gamma1inv);",
    "    out[idx] = gamma1inv * (((p[idx] * two_minus_od) - (gamma1 * q[idx])) + "
    "((omega0dtsqr * (s[idx] * w[idx])) - (0.0f * (s[idx] * w_other[idx]))));")


def _complex(rng, shape, zero_real: bool):
    real = rng.uniform(-1.0, 1.0, size=shape)
    if zero_real:
        zeroed = rng.integers(0, 2, size=shape) == 0
        signs = numpy.where(rng.integers(0, 2, size=shape) == 0, -0.0, 0.0)
        real = numpy.where(zeroed, signs, real)
    out = numpy.empty(shape, dtype=numpy.complex64)
    out.real = real.astype(F32)  # assigned per plane: ``re + 1j*im`` loses -0
    out.imag = rng.uniform(-1.0, 1.0, size=shape).astype(F32)
    return out


def _case(seed: int, uniform_sigma: bool, zero_real: bool):
    grid = Grid(resolution=1.0, cell_size=(16.0, 16.0, 12.0),
                boundaries=("periodic",) * 3, xp=numpy, courant=0.5)
    shape = tuple(int(n) for n in grid.shape)
    rng = numpy.random.default_rng(seed)
    sigma = 0.33 if uniform_sigma else rng.uniform(0.1, 0.9, size=shape).astype(F32)
    state = PolarizationState(Susceptibility(frequency=1.05, gamma=0.05),
                              {"Ex": sigma, "Ey": 0.0, "Ez": 0.0}, grid, numpy.complex64)
    p = _complex(rng, shape, zero_real)
    q = _complex(rng, shape, zero_real)
    w = _complex(rng, shape, zero_real)
    state.P["Ex"][...] = p
    state.P_prev["Ex"][...] = q
    state.update(lambda _component: w, grid.dt)
    want = numpy.array(state.P["Ex"], copy=True)
    s = numpy.broadcast_to(numpy.asarray(state.sigma["Ex"], dtype=F32), shape)
    return state, p, q, w, s, want


def _part(text, values, state, other=None):
    p, q, w, s = values
    out = numpy.zeros(p.size, dtype=F32)
    named = {"out": out, "p": numpy.ascontiguousarray(p, dtype=F32).ravel(),
             "q": numpy.ascontiguousarray(q, dtype=F32).ravel(),
             "w": numpy.ascontiguousarray(w, dtype=F32).ravel(),
             "s": numpy.ascontiguousarray(s, dtype=F32).ravel(),
             "nx": int(p.shape[0]), "ny": int(p.shape[1]), "nz": int(p.shape[2])}
    for name, value in zip(("two_minus_od", "gamma1", "omega0dtsqr", "gamma1inv"),
                           state.constants):
        named[name] = F32(value)
    if other is not None:
        named["w_other"] = numpy.ascontiguousarray(other, dtype=F32).ravel()
    Interpreter(text).launch(named, int(p.size))
    return out.reshape(p.shape)


def _bits(array):
    return numpy.ascontiguousarray(array, dtype=F32).view(numpy.uint32)


@pytest.mark.parametrize("uniform_sigma", (False, True))
@pytest.mark.parametrize("zero_real", (False, True))
def test_each_part_is_meeps_recurrence_word_for_word(uniform_sigma, zero_real):
    state, p, q, w, s, want = _case(5, uniform_sigma, zero_real)
    real = _part(_PROBE, (p.real, q.real, w.real, s), state)
    imag = _part(_PROBE, (p.imag, q.imag, w.imag, s), state)
    assert numpy.count_nonzero(_bits(real) != _bits(want.real)) == 0
    assert numpy.count_nonzero(_bits(imag) != _bits(want.imag)) == 0
    if zero_real:
        assert numpy.count_nonzero(_bits(want.real) == _bits(F32(-0.0))) > 0, (
            "no -0 real word: the class did not reach the case the forms differ on")


def test_the_package_cross_term_changes_a_word_on_signed_zero_real_planes():
    state, p, q, w, s, want = _case(5, False, True)
    planted = _part(_PACKAGE_REAL_PART, (p.real, q.real, w.real, s), state,
                    other=w.imag)
    assert numpy.count_nonzero(_bits(planted) != _bits(want.real)) > 0


def test_ade_step_is_the_two_part_calls():
    text = family._NO_PML_HELPERS  # noqa: SLF001
    assert ("    out.re = ade_part(p.re, q.re, w.re, s, two_minus_od, gamma1, "
            "omega0dtsqr, gamma1inv);\n"
            "    out.im = ade_part(p.im, q.im, w.im, s, two_minus_od, gamma1, "
            "omega0dtsqr, gamma1inv);") in text
    assert "mul_coefficient_left" not in _ADE_PART


@pytest.mark.parametrize("key", ("update_P", "update_P_uniform"))
def test_the_stand_alone_kernel_passes_the_constants_in_meeps_order(key):
    text = family.kernel_source(key)
    assert "two_minus_od, gamma1, omega0dtsqr, gamma1inv));" in text
    assert re.search(r"cf_store\(p_out, idx, ade_step\(\s*cf_load\(p_now, idx\), "
                     r"cf_load\(p_prev, idx\), cf_load\(drive, idx\),", text)


def test_the_fused_kernels_pass_the_constants_in_meeps_order():
    from . import (complex_no_pml_fused_polarization_pair as pair,  # noqa: PLC0415
                   complex_no_pml_three_slot_dispersive_weld as weld)

    texts = [pair.fused_polarization_pair_no_pml_complex_source(c, 2, (True, False))
             for c in range(3)]
    # A variant with no pole advances nothing and calls no recurrence.
    texts += [text for text in weld.device_sources().values() if "p_out_0" in text]
    assert len(texts) > 3
    for text in texts:
        calls = re.findall(r"cf_store\(p_out_\d+, idx, ade_step\(.*?\)\);", text, re.S)
        assert calls, "no ade_step call"
        for call in calls:
            index = re.search(r"two_minus_od_(\d+), gamma1_\1, omega0dtsqr_\1, "
                              r"gamma1inv_\1\)\);", call)
            assert index is not None, call
