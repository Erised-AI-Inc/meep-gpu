"""
Products on complex field storage, formed part by part as MEEP forms them.

MEEP stores a complex field as two ``realnum`` arrays, ``f[c][0]`` and ``f[c][1]``,
and never multiplies a complex number by a complex number while it steps:

* a real coefficient times a field runs once per part, in the ``DOCMP`` loops
  (step_db.cpp:47, update_eh.cpp:149): ``(kap - sig) * f[c][cmp][i]``,
  ``chi1inv * dmp[cmp][i]`` -- one float32 product per word;
* a Bloch connection is ``complex<realnum>(thephase) * word`` (step.cpp:171-221,
  boundaries.cpp:359-366, :495), which the contraction-off build forms as libc++'s
  ``(a*c) - (b*d), (a*d) + (b*c)`` with every product and every sum rounded on its
  own; a phase within 1e-13 of +1 or -1 is a copy or a unary minus instead.

NumPy's and CuPy's complex multiply is neither. It contracts to ``fma(a, c,
-(b*d))`` wherever the host has fused multiply-add (NumPy 2.x on arm64 and on
x86-64 with FMA3, MEASURED in phase 0 on 2,000,000 of 2,000,000 words; CuPy under
NVRTC's default ``--fmad=true``), and there is no float-times-complex loop, so a
real coefficient times complex64 storage is a full complex multiply against
``(k + 0j)`` whose zero cross terms can flip the sign of a zero part (MEASURED:
``(kap - sig) < 0`` times ``+0`` gives MEEP's ``-0`` per part and NumPy's ``+0`` in
the imaginary part). So every product on the array path goes through this module:
:func:`scale` for a real coefficient, :func:`quarter_turn` for ``+-i`` times a real
coefficient, :func:`connect` for a Bloch connection, :func:`parts` to step a field
part by part, :func:`from_parts` to assemble a complex value from parts without a
product. A runtime probe (``parity/meep_gpu/complex_product_interceptor.py``) checks
that the stepping path forms no other complex product.

Every function takes NumPy or CuPy arrays and uses the module that owns them; the
``.real`` and ``.imag`` attributes of a complex array are writable strided views on
both, so a part product writes in place without an interleaved copy. Real storage
takes the plain ufunc, so a real run's arithmetic is unchanged by construction.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy

#: A ``complex<double>`` as its two parts, Python floats, signed zeros kept.
ComplexParts = Tuple[float, float]

#: ``phase_isclose`` (boundaries.cpp:359-360): within this of +1 or -1 a connection is
#: a copy or a negation, not a product.
CONNECT_TOLERANCE = 1e-13


def array_module(values: Any) -> Any:
    """``cupy`` for a device array, ``numpy`` otherwise."""
    if type(values).__module__.split(".")[0] == "cupy":
        import cupy  # noqa: PLC0415

        return cupy
    return numpy


def _is_scalar(coefficient: Any) -> bool:
    """A Python number or a NumPy scalar / 0-d host array (never a device array)."""
    if isinstance(coefficient, (bool, int, float, complex)):
        return True
    return getattr(coefficient, "ndim", None) == 0 and type(coefficient).__module__.split(".")[0] == "numpy"


def _realnum(coefficient: Any, real_dtype: Any) -> Any:
    """MEEP's ``realnum`` coefficient for complex storage whose parts are ``real_dtype``.

    A scalar is rounded to ``real_dtype`` (a ``realnum``); an array must already be
    ``real_dtype``, because promoting the parts to a wider type would round twice.
    A complex coefficient is refused: it is not a scale but a product.
    """
    if _is_scalar(coefficient):
        if isinstance(coefficient, complex) or getattr(getattr(coefficient, "dtype", None), "kind", "") == "c":
            raise TypeError(
                f"a complex coefficient {coefficient!r} is a complex product, not a part-wise scale; "
                f"MEEP's step carries none (form it with quarter_turn or connect)")
        return real_dtype.type(coefficient)
    dtype = getattr(coefficient, "dtype", None)
    if dtype is None or numpy.dtype(dtype) != real_dtype:
        raise TypeError(
            f"a part-wise scale of {real_dtype}-part storage needs a {real_dtype} coefficient "
            f"array (MEEP's realnum), got {dtype}; a wider coefficient would round each part twice")
    return coefficient


def _real_dtype(values: Any) -> Any:
    return numpy.dtype(numpy.float32) if values.dtype == numpy.complex64 else numpy.dtype(numpy.float64)


def scale(coefficient: Any, values: Any, out: Optional[Any] = None) -> Any:
    """``coefficient * values`` with a real coefficient, one product per stored part.

    Real storage takes ``xp.multiply(coefficient, values, out=out)``, the same call the
    expression ``coefficient * values`` makes. Complex storage multiplies the real and
    the imaginary part separately by the ``realnum`` coefficient (:func:`_realnum`),
    into ``out`` (which may be ``values`` itself) or a new array. Multiplication is
    commutative in IEEE arithmetic, so the operand order is immaterial.
    """
    xp = array_module(values)
    if values.dtype.kind != "c":
        return xp.multiply(coefficient, values, out=out)
    coeff = _realnum(coefficient, _real_dtype(values))
    if out is None:
        shape = numpy.broadcast_shapes(getattr(coeff, "shape", ()), values.shape)
        out = xp.empty(shape, dtype=values.dtype)
    elif out.dtype != values.dtype:
        raise TypeError(f"a part-wise scale writes {values.dtype} parts, got an out of {out.dtype}")
    xp.multiply(coeff, values.real, out=out.real)
    xp.multiply(coeff, values.imag, out=out.imag)
    return out


def divide(values: Any, divisor: Any, out: Optional[Any] = None) -> Any:
    """``values / divisor`` with a real divisor, one quotient per stored part.

    Real storage takes ``xp.true_divide(values, divisor, out=out)``. A complex
    quotient by ``(d + 0j)`` is NumPy's complex division, several operations per part;
    this divides each part by the ``realnum`` divisor once.
    """
    xp = array_module(values)
    if values.dtype.kind != "c":
        return xp.true_divide(values, divisor, out=out)
    d = _realnum(divisor, _real_dtype(values))
    if out is None:
        shape = numpy.broadcast_shapes(getattr(d, "shape", ()), values.shape)
        out = xp.empty(shape, dtype=values.dtype)
    elif out.dtype != values.dtype:
        raise TypeError(f"a part-wise quotient writes {values.dtype} parts, got an out of {out.dtype}")
    xp.true_divide(values.real, d, out=out.real)
    xp.true_divide(values.imag, d, out=out.imag)
    return out


def quarter_turn(coefficient: Any, values: Any) -> Any:
    """``(1j * coefficient) * values`` for complex storage, part by part, as a new array.

    ``i*c*(x + iy) = -(c*y) + i*(c*x)``: two real products and a negation, which is
    exact. MEEP spells the same coupling as the other part times a signed realnum
    (``f[c_g][1 - cmp]``, step_db.cpp:148-294); which spelling the identity work
    package of a family adopts is that family's decision, and this one forms no
    complex product either way. Real storage is refused: it has no other part.
    """
    xp = array_module(values)
    if values.dtype.kind != "c":
        raise TypeError("a quarter turn needs complex storage: a real field has no other part")
    coeff = _realnum(coefficient, _real_dtype(values))
    shape = numpy.broadcast_shapes(getattr(coeff, "shape", ()), values.shape)
    out = xp.empty(shape, dtype=values.dtype)
    xp.multiply(coeff, values.imag, out=out.real)
    xp.negative(out.real, out=out.real)
    xp.multiply(coeff, values.real, out=out.imag)
    return out


def parts(values: Any) -> Tuple[Any, ...]:
    """MEEP's per-part arrays of one field: ``(values,)`` for real storage, and
    ``(values.real, values.imag)`` for complex storage.

    Views of the stored words, never a complex operation: each part is then stepped
    with real operands alone, as MEEP's ``DOCMP`` loop steps ``f[c][0]`` and
    ``f[c][1]`` (update_eh.cpp:149). :func:`from_parts` is the inverse.
    """
    if values.dtype.kind != "c":
        return (values,)
    return (values.real, values.imag)


def from_parts(real: Any, imag: Any, dtype: Any) -> Any:
    """A complex array whose parts are ``real`` and ``imag`` (copied), with no product.

    ``real + 1j * imag`` is a complex multiply of ``imag`` against ``(0 + 1j)``; this
    writes the two parts instead.
    """
    xp = array_module(real)
    shape = numpy.broadcast_shapes(real.shape, imag.shape)
    out = xp.empty(shape, dtype=dtype)
    out.real[...] = real
    out.imag[...] = imag
    return out


def connect_parts(parts: Sequence[Any], phase: ComplexParts) -> List[Any]:
    """One not-owned word from its owned source, by MEEP's connection rule, part by part.

    ``parts`` is ``[re]`` (real storage) or ``[re, im]`` (complex storage), float arrays
    of the storage's ``realnum``; ``phase`` is ``thephase`` as ``complex<double>`` parts.
    Within :data:`CONNECT_TOLERANCE` of +1 a copy, of -1 a unary minus per part
    (CONNECT_COPY / CONNECT_NEGATE, boundaries.cpp:359-366); otherwise CONNECT_PHASE,
    ``complex<realnum>(phase) * word`` (step.cpp:183-192): the phase rounded to the
    part dtype (boundaries.cpp:495), then four products and two sums, each its own
    operation. Real storage cannot carry a phase and is refused by name.
    """
    pr, pi = float(phase[0]), float(phase[1])
    if abs(pi) < CONNECT_TOLERANCE and abs(pr - 1.0) < CONNECT_TOLERANCE:
        return [p.copy() for p in parts]
    if abs(pi) < CONNECT_TOLERANCE and abs(pr + 1.0) < CONNECT_TOLERANCE:
        return [-p for p in parts]
    if len(parts) != 2:
        raise ValueError(
            f"Bloch boundaries need complex fields: the connection phase {complex(pr, pi)!r} has an "
            f"imaginary part and the field storage is real. Build the Fields with "
            f"force_complex_fields=True, or use k_point = 0 on this axis.")
    vr, vi = parts
    a = vr.dtype.type(pr)
    b = vr.dtype.type(pi)
    re = (a * vr) - (b * vi)
    im = (a * vi) + (b * vr)
    return [re, im]


def connect(values: Any, phase: ComplexParts) -> Any:
    """:func:`connect_parts` on one array of complex or real storage, as a new array."""
    if values.dtype.kind != "c":
        return connect_parts([values], phase)[0]
    re, im = connect_parts([values.real, values.imag], phase)
    return from_parts(re, im, values.dtype)
