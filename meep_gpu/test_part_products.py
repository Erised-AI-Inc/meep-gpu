"""``part_products``: MEEP's per-part arithmetic on complex storage, word for word.

What each pins:

* a real coefficient times complex64 storage is one float32 product per part -- equal to
  the part products themselves, including the sign of zero that NumPy's complex multiply
  against ``(k + 0j)`` gets wrong (MEASURED in phase 0: ``kap - sig < 0`` times ``+0``);
* real storage takes the plain ufunc, so a real run's words cannot move;
* the coefficient is MEEP's ``realnum``: a scalar rounds to the part dtype, a float64
  array or a complex coefficient is refused by name;
* strided views (a slice along any axis) are written in place;
* a Bloch connection is MEEP's ``connect_phase`` rule: a copy within 1e-13 of +1, a unary
  minus of -1, else ``complex<float>(phase) * word`` with each operation rounded, which
  is NOT NumPy's complex64 multiply on a host that contracts it.
"""

from __future__ import annotations

import numpy
import pytest

from . import part_products


def _complex(shape, seed=11):
    rng = numpy.random.default_rng(seed)
    z = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(numpy.complex64)
    flat = z.reshape(-1)
    flat[0] = complex(0.0, 0.0)
    flat[1] = complex(-0.0, 0.0)
    flat[2] = complex(0.0, -0.0)
    return z


def _words(array):
    return numpy.ascontiguousarray(array).view(numpy.uint32)


def test_a_real_scale_of_complex_storage_is_one_float32_product_per_part():
    z = _complex((4, 5, 3))
    for coefficient in (numpy.float32(-0.23955), -0.23955, 2, numpy.float32(1.0)):
        out = part_products.scale(coefficient, z)
        k = numpy.float32(coefficient)
        assert out.dtype == numpy.complex64
        assert numpy.array_equal(_words(out.real), _words(k * z.real))
        assert numpy.array_equal(_words(out.imag), _words(k * z.imag))
    # The sign of zero, which the complex multiply against (k + 0j) cannot keep: a
    # negative realnum times (+0, -0) is (-0, +0) per part.
    out = part_products.scale(numpy.float32(-0.5), z)
    assert numpy.signbit(out.reshape(-1)[0].real) and numpy.signbit(out.reshape(-1)[0].imag)
    assert not numpy.signbit(out.reshape(-1)[2].imag)


def test_a_coefficient_array_broadcasts_and_writes_in_place_through_strided_views():
    z = _complex((6, 5, 4), seed=3)
    k = numpy.random.default_rng(4).standard_normal((6, 1, 1)).astype(numpy.float32)
    expected_re, expected_im = k * z.real, k * z.imag
    view = z.copy()
    part_products.scale(k, view, out=view)
    assert numpy.array_equal(_words(view.real), _words(expected_re))
    assert numpy.array_equal(_words(view.imag), _words(expected_im))
    # A slice along the last axis is not contiguous; the part views write through it.
    base = z.copy()
    sliced = base[:, :, 1:3]
    part_products.scale(k, sliced, out=sliced)
    assert numpy.array_equal(_words(base[:, :, 1:3].real), _words(k * z[:, :, 1:3].real))
    assert numpy.array_equal(base[:, :, 0], z[:, :, 0])  # outside the slice: untouched


def test_real_storage_takes_the_plain_ufunc():
    r = numpy.random.default_rng(5).standard_normal((3, 4, 2)).astype(numpy.float32)
    k = numpy.random.default_rng(6).standard_normal((3, 1, 1)).astype(numpy.float32)
    assert numpy.array_equal(_words(part_products.scale(k, r)), _words(k * r))
    assert numpy.array_equal(_words(part_products.scale(0.1, r)), _words(0.1 * r))
    assert numpy.array_equal(_words(part_products.divide(r, k)), _words(r / k))


def test_the_coefficient_is_a_realnum_and_a_complex_one_is_refused():
    z = _complex((2, 3, 1))
    with pytest.raises(TypeError, match="float32 coefficient"):
        part_products.scale(numpy.ones((2, 1, 1)), z)  # float64: each part would round twice
    with pytest.raises(TypeError, match="complex product"):
        part_products.scale(1j, z)
    with pytest.raises(TypeError, match="complex product"):
        part_products.scale(numpy.complex64(2.0), z)
    with pytest.raises(TypeError, match="needs complex storage"):
        part_products.quarter_turn(1.0, numpy.zeros(3, dtype=numpy.float32))


def test_divide_quarter_turn_and_from_parts_form_no_complex_product():
    z = _complex((3, 4, 1), seed=8)
    d = numpy.random.default_rng(9).uniform(0.5, 2.0, (3, 1, 1)).astype(numpy.float32)
    q = part_products.divide(z, d)
    assert numpy.array_equal(_words(q.real), _words(z.real / d))
    assert numpy.array_equal(_words(q.imag), _words(z.imag / d))
    c = numpy.float32(0.37)
    t = part_products.quarter_turn(c, z)  # (i * c) * z = -(c * im) + i * (c * re)
    assert numpy.array_equal(_words(t.real), _words(-(c * z.imag)))
    assert numpy.array_equal(_words(t.imag), _words(c * z.real))
    assert t == pytest.approx((1j * 0.37) * z.astype(numpy.complex128), abs=1e-6)
    re, im = z.real.copy(), z.imag.copy()
    assert numpy.array_equal(_words(part_products.from_parts(re, im, numpy.complex64)), _words(z))


def test_parts_are_views_of_the_stored_words_and_from_parts_inverts_them():
    z = _complex((4, 3, 2), seed=21)
    real, imaginary = part_products.parts(z)
    assert real.dtype == imaginary.dtype == numpy.float32
    assert numpy.shares_memory(real, z) and numpy.shares_memory(imaginary, z)
    assert numpy.array_equal(_words(real), _words(z.real))
    assert numpy.array_equal(_words(imaginary), _words(z.imag))
    assert numpy.array_equal(_words(part_products.from_parts(real, imaginary, numpy.complex64)),
                             _words(z))
    x = numpy.random.default_rng(22).standard_normal((4, 3)).astype(numpy.float32)
    (only,) = part_products.parts(x)
    assert only is x


def test_a_connection_is_meeps_copy_negate_or_separately_rounded_product():
    z = _complex((50, 7, 1), seed=12)
    assert numpy.array_equal(_words(part_products.connect(z, (1.0, 0.0))), _words(z))
    assert numpy.array_equal(_words(part_products.connect(z, (1.0 + 5e-14, -5e-14))), _words(z))
    negated = part_products.connect(z, (-1.0, 0.0))
    assert numpy.array_equal(_words(negated), _words(-z))  # unary minus: exact, signs flip
    phase = (0.30901699437494745, 0.9510565162951535)
    out = part_products.connect(z, phase)
    a, b = numpy.float32(phase[0]), numpy.float32(phase[1])
    assert numpy.array_equal(_words(out.real), _words((a * z.real) - (b * z.imag)))
    assert numpy.array_equal(_words(out.imag), _words((a * z.imag) + (b * z.real)))
    with pytest.raises(ValueError, match="complex fields"):
        part_products.connect(numpy.ones(3, dtype=numpy.float32), phase)
    # On real storage the copy and the negation are still connections.
    r = numpy.arange(3, dtype=numpy.float32)
    assert numpy.array_equal(part_products.connect(r, (-1.0, 0.0)), -r)
