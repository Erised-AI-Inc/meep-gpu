"""MEEP's Gaussian-beam source fields, transcribed and vectorized.

``mp.GaussianBeamSource`` is not a current distribution the user writes down; MEEP
synthesizes one at setup. ``GaussianBeam3DSource.add_source`` (python/source.py:748-777)
marshals the beam parameters into a C++ ``gaussianbeam`` object and hands it to
``fields::add_volume_source`` (src/sources.cpp:526-567), which expands it into the
same four Love-equivalence current sheets the eigenmode source uses and deposits each
through ``add_volume_source`` with an amp_func that samples ``gaussianbeam::get_fields``
(src/sources.cpp:509-524).

This module is the first half of that: :func:`beam_fields` is a line-by-line
transcription of ``gaussianbeam::get_fields`` (src/sources.cpp:624-746 — itself
adapted from M. T. Homer Reid's SCUFF-EM, following Sheppard & Saghafi,
J. Opt. Soc. Am. A 16, 1381: a complex-point-source solution that satisfies Maxwell's
equations exactly rather than only paraxially), vectorized over an array of points.
The second half is this package's own ``sources._build_source_points``, which is
already the transcription of ``add_volume_source`` itself, so ``from_meep`` only has
to hand it the same four (component, +/-1, profile) requests MEEP builds.

:func:`compiled_beam_fields` calls MEEP's own compiled ``get_fields`` through ctypes
and exists ONLY as the test oracle for the transcription. It must never be the
production path: the mangled symbol it looks up is libc++-specific (a libstdc++ build
spells ``std::complex<double>`` differently) and a per-point round trip through the
C++ ABI costs ~1e5 calls per sheet on a 3-D plane.

TWO THINGS HERE ARE NOT THE PHYSICS, AND ARE DELIBERATE.

1. ``gb_Hx`` is zero. sources.cpp:691 and :716 read ``EH[1]`` — the caller's OUTPUT
   buffer — before anything has been written to it, so the value is whatever was on
   the stack. It is plainly meant to be ``gb_Ey`` (introduced in MEEP commit d8bd779b,
   "complex source point Gaussian beam", and untouched since). MEEP's own path
   declares ``std::complex<double> EH[6];`` fresh in ``gaussianbeam_ampfunc``
   (sources.cpp:511) and the value it reads is reliably 0. Reproducing MEEP means
   reproducing that zero, not the intended field: see the measurements in
   ``test_gaussian_beam.py``, which pin MEEP's zero directly so that a repair
   upstream turns the suite red instead of quietly moving the parity number.

2. The dimension-dependent vector algebra. ``meep::vec`` carries its own ``dim`` and
   every operator loops only over that dimensionality (``LOOP_OVER_DIRECTIONS``,
   src/meep/vec.hpp:640-690), while ``get_fields`` mixes vectors of different rank:
   ``x``, ``x0``, ``kdir`` and ``zhat`` are D2 in a 2-D cell, but the cross products
   and the polarization axes are built with the THREE-argument constructor and are
   always D3. So in 2-D the z components survive in some places and not others —
   ``py_v3_to_vec`` deliberately keeps ``v3.z`` on a D2 vec (python/simulation.py:130),
   ``x - x0`` never subtracts it, ``kdir`` is normalized by its in-plane length alone,
   and ``zhat & xrel`` sums only x and y, while ``rho``, ``xhat``, ``yhat`` and every
   final projection use all three. Reducing the beam vectors to the cell's
   dimensionality — the instinct ``_lift_source`` correctly applies to a source's
   centre and size — is measurably wrong here.
"""

# Derived from MEEP (https://github.com/NanoComp/meep) and, through it, from SCUFF-EM.
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# Copyright (C) 2005-2011 M. T. Homer Reid.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import ctypes
import math

import numpy as np

__all__ = ["beam_fields", "compiled_beam_fields", "GET_FIELDS_SYMBOL"]


# The number of directions ``LOOP_OVER_DIRECTIONS`` covers for a vec of each
# dimensionality: D1 is the z axis alone, D2 is x and y, D3 is all three
# (src/meep/vec.hpp, LOOP_OVER_DIRECTIONS / ndim).
_LOOPED_DIRECTIONS = {2: 2, 3: 3}


def _gb_hx(gb_Ey):
    """MEEP's ``gb_Hx``, sources.cpp:691 and :716 — zero, and not the field.

    The C++ reads ``complex<double> gb_Hx = EH[1];``: ``EH`` is the OUTPUT buffer,
    nothing has been written to it yet, and its caller ``gaussianbeam_ampfunc``
    (sources.cpp:511) hands it a fresh, uninitialized stack array. The value is
    plainly meant to be ``gb_Ey`` — the two blocks are otherwise a transliteration of
    each other — but what MEEP actually deposits is 0, measured on both a 1.29 and a
    1.33 build, so 0 is what parity with MEEP means.

    A separate function so the divergence has one home, and so the suite can flip it
    to the intended ``gb_Ey`` and measure what the "correct" reading would cost:
    0.0671 against a 9.068 peak sheet amplitude in the configuration where it reaches
    a sheet at all. See ``test_gaussian_beam.py``.
    """
    return 0.0 * gb_Ey


def _cross(a, b):  # zhat x b, written out exactly as sources.cpp:632-634 does.
    return np.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], axis=-1)


def beam_fields(offsets, x0, kdir, w0, freq, eps, mu, E0, dimensions: int):
    """``gaussianbeam::get_fields`` (sources.cpp:624-746) over many points at once.

    ``offsets`` is (N, 3) and holds the vectors MEEP passes as ``x``: the location
    RELATIVE to the source centre, because ``src_vol_chunkloop`` calls the amp_func
    with ``rel_loc = loc - center`` (sources.cpp:266-268) and ``gaussianbeam_ampfunc``
    forwards it unchanged. ``x0`` (the beam focus) is therefore also measured from the
    source centre, which is what ``mp.GaussianBeamSource.beam_x0`` documents.

    ``x0``, ``kdir`` are (3,) real and ``E0`` is (3,) complex — all three carried at
    full length even for a 2-D cell, because ``py_v3_to_vec`` keeps ``v3.z`` on a D2
    vec (simulation.py:130) and this routine reads it. ``dimensions`` selects how many
    directions the ``meep::vec`` operators loop over; see the module docstring.

    Returns (N, 6) complex128: Ex, Ey, Ez, Hx, Hy, Hz.
    """
    looped = _LOOPED_DIRECTIONS.get(int(dimensions))
    if looped is None:
        raise ValueError(
            f"gaussianbeam has no 1-D or cylindrical form (MEEP aborts on Dcyl, "
            f"sources.cpp:562); got dimensions={dimensions}."
        )
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
    x0 = np.asarray(x0, dtype=np.float64).reshape(3)
    kdir = np.asarray(kdir, dtype=np.float64).reshape(3)
    E0 = np.asarray(E0, dtype=np.complex128).reshape(3)

    n = math.sqrt(eps * mu)
    k = 2.0 * math.pi * freq * n
    ZR = math.sqrt(mu / eps)
    z0 = k * w0 * w0 / 2.0
    kz0 = k * z0

    # sources.cpp:630 ``vec xrel = x - x0``. vec::operator- copies the RIGHT operand
    # and then overwrites only the looped directions (vec.hpp:640-644), so on a D2 vec
    # the z slot survives as x0's own z — not as x.z - x0.z, and not as zero.
    xrel = np.broadcast_to(x0, offsets.shape).copy()
    xrel[:, :looped] = offsets[:, :looped] - x0[:looped]

    # sources.cpp:632 ``vec zhat = kdir / abs(kdir)``. ``abs`` is sqrt(v & v)
    # (vec.hpp:718) and both it and operator/ loop over the vec's own dimensionality,
    # so a 2-D beam is normalized by its IN-PLANE length and keeps kdir.z unscaled.
    zhat = kdir.copy()
    zhat[:looped] /= math.sqrt(float(np.dot(kdir[:looped], kdir[:looped])))

    # sources.cpp:633-635: the cross product is built with the three-argument vec
    # constructor, so ``rho`` is a full 3-D magnitude even in a 2-D cell.
    rho = np.linalg.norm(_cross(np.broadcast_to(zhat, xrel.shape), xrel), axis=-1)
    # sources.cpp:636 ``zhat & xrel`` — the dot product loops over ZHAT's dim.
    zhatdotxrel = xrel[:, :looped] @ zhat[:looped]

    zc = zhatdotxrel - 1j * z0                                      # sources.cpp:640
    Rsq = rho * rho + zc * zc                                       # sources.cpp:641
    R = np.sqrt(Rsq.astype(np.complex128))                          # sources.cpp:642
    kR = k * R
    kR2 = kR * kR
    kR3 = kR2 * kR

    big = np.abs(kR) > 1e-4                                         # sources.cpp:645
    rescaled = big & (np.abs(kR.imag) > 30.0)                       # sources.cpp:647
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coskR = np.cos(kR)
        sinkR = np.sin(kR)
        if rescaled.any():
            # sources.cpp:648-653 — cos/sin rewritten so the growing exponential is
            # divided by exp(kz0) before it overflows.
            ExpI = np.exp(1j * kR.real)
            ExpPlus = np.exp(kR.imag - kz0)
            ExpMinus = np.exp(-(kR.imag + kz0))
            coskR = np.where(rescaled, 0.5 * (ExpI * ExpMinus + np.conj(ExpI) * ExpPlus), coskR)
            sinkR = np.where(rescaled,
                             -0.5j * (ExpI * ExpMinus - np.conj(ExpI) * ExpPlus), sinkR)
        f_big = -3.0 * (coskR / kR2 - sinkR / kR3)                  # sources.cpp:659
        g_big = 1.5 * (sinkR / kR + coskR / kR2 - sinkR / kR3)      # sources.cpp:660
        fmg_big = (f_big - g_big) / Rsq                             # sources.cpp:661
        # sources.cpp:664-669 — the small-R Taylor series, which is also the only
        # branch defined at R = 0 (the other divides by kR3).
        kR4 = kR2 * kR2
        f_small = kR4 / 280.0 - kR2 / 10.0 + 1.0
        g_small = 3.0 * kR4 / 280.0 - kR2 / 5.0 + 1.0
        fmg_small = (kR4 / 5040.0 - kR2 / 140.0 + 0.1) * (k * k)
        f = np.where(big, f_big, f_small)
        g = np.where(big, g_big, g_small)
        fmgbRsq = np.where(big, fmg_big, fmg_small)

    i2fk = 0.5j * f * k                                             # sources.cpp:671

    E = np.zeros((offsets.shape[0], 3), dtype=np.complex128)
    H = np.zeros((offsets.shape[0], 3), dtype=np.complex128)
    # sources.cpp:680-731: the real and imaginary parts of E0 each define their own
    # polarization frame and are accumulated separately, the second with a factor i.
    for part, phase in ((E0.real, 1.0 + 0j), (E0.imag, 1j)):
        norm = float(np.linalg.norm(part))
        if norm <= 1e-13:                                           # sources.cpp:681, :709
            continue
        xhat = part / norm                                          # D3: sources.cpp:682
        yhat = _cross(zhat, xhat)                                   # D3: sources.cpp:683
        xhatdotxrel = xrel @ xhat                                   # D3 dot: all three
        yhatdotxrel = xrel @ yhat
        gb_Ex = g + fmgbRsq * xhatdotxrel * xhatdotxrel + i2fk * zc
        gb_Ey = fmgbRsq * xhatdotxrel * yhatdotxrel
        gb_Ez = fmgbRsq * xhatdotxrel * zc - i2fk * xhatdotxrel
        gb_Hx = _gb_hx(gb_Ey)  # sources.cpp:691 / :716 — an uninitialized read, not a field.
        gb_Hy = g + fmgbRsq * yhatdotxrel * yhatdotxrel + i2fk * zc
        gb_Hz = fmgbRsq * yhatdotxrel * zc - i2fk * yhatdotxrel
        for axis in range(3):
            E[:, axis] += phase * norm * (
                gb_Ex * xhat[axis] + gb_Ey * yhat[axis] + gb_Ez * zhat[axis])
            H[:, axis] += phase * norm * (
                gb_Hx * xhat[axis] + gb_Hy * yhat[axis] + gb_Hz * zhat[axis])

    # sources.cpp:734-739 — the on-axis amplitude the whole profile is divided by, so
    # that E0 is the field at the focus. Its branch is the point's own f/g branch.
    with np.errstate(over="ignore"):
        Eorig_plain = 3.0 / (2 * kz0 * kz0 * kz0) * (math.exp(kz0) * kz0 * (kz0 - 1)
                                                     + math.sinh(kz0))
        Eorig_rescaled = 3.0 / (2 * kz0 * kz0 * kz0) * (kz0 * (kz0 - 1)
                                                        + 0.5 * (1.0 - math.exp(-2.0 * kz0)))
    Eorig = np.where(rescaled, Eorig_rescaled, Eorig_plain)

    out = np.empty((offsets.shape[0], 6), dtype=np.complex128)
    out[:, 0:3] = E / Eorig[:, None]                                # sources.cpp:741-743
    out[:, 3:6] = H / (Eorig[:, None] * ZR)                         # sources.cpp:744-746
    return out


# ``gaussianbeam::get_fields(std::complex<double>*, meep::vec const&) const`` under
# the Itanium C++ ABI as libc++ spells it. Test-oracle only — see the module docstring.
GET_FIELDS_SYMBOL = "_ZNK4meep12gaussianbeam10get_fieldsEPNSt3__17complexIdEERKNS_3vecE"


def compiled_beam_fields(mp, offsets, x0, kdir, w0, freq, eps, mu, E0, dimensions: int):
    """MEEP's OWN compiled ``get_fields``, one point at a time, as the test oracle.

    ``get_fields`` has no SWIG typemap for its ``std::complex<double>*`` output
    argument, so it is unreachable through the Python bindings; the symbol is exported
    from the extension module, though, and ``ctypes.CDLL(None)`` resolves it once meep
    has been imported. Raises rather than falling back: an unresolved symbol means the
    oracle is unavailable, and a silent substitution here would validate the
    transcription against itself.
    """
    try:
        get_fields = getattr(ctypes.CDLL(None), GET_FIELDS_SYMBOL)
    except AttributeError as exc:  # a libstdc++ build spells the symbol differently
        raise RuntimeError(
            f"MEEP's compiled gaussianbeam::get_fields is not resolvable as "
            f"{GET_FIELDS_SYMBOL!r} in this process; the C++ oracle is unavailable on "
            f"this build. This is a TEST oracle only — meep_gpu.gaussian_beam."
            f"beam_fields does not depend on it."
        ) from exc
    get_fields.restype = None
    get_fields.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

    # A contiguous complex128 array, not a list: the SWIG typemap for
    # ``std::complex<double> E0_[3]`` takes the buffer directly, and a list segfaults
    # in ``_wrap_new_gaussianbeam``. source.py:766-768 spells it the same way.
    amplitudes = np.ascontiguousarray(np.asarray(E0, dtype=np.complex128).reshape(3))
    beam = mp.gaussianbeam(
        mp.py_v3_to_vec(dimensions, mp.Vector3(*x0)),
        mp.py_v3_to_vec(dimensions, mp.Vector3(*kdir)),
        float(w0), float(freq), float(eps), float(mu), amplitudes)
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 3)
    out = np.empty((offsets.shape[0], 6), dtype=np.complex128)
    for index, offset in enumerate(offsets):
        buffer = np.zeros(6, dtype=np.complex128)
        point = mp.py_v3_to_vec(dimensions, mp.Vector3(*offset))
        get_fields(int(beam.this), buffer.ctypes.data, int(point.this))
        out[index] = buffer
    return out
