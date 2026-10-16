"""The source deposit in MEEP's double precision: ``f = float(double(f) - real((amp * current) * dt))``.

MEEP's ``fields_chunk::step_source`` (step.cpp:295-317) subtracts a ``complex<double>``
from a ``realnum`` word, so the stored float32 word is widened, ONE double subtraction
runs and ONE rounding brings it back; ``A = sv.current(j) * dt`` (times
``double(cndinv[i])`` under a conductivity) is formed in double, the complex product
first. Every test here compares stored WORDS (uint32), sign of zero included, against a
scalar transcription of those statements in Python floats -- which are IEEE doubles,
evaluated one operation at a time with no contraction -- so nothing here is a tolerance.

What is pinned:

* :func:`host_writes.subtract_double` is the one-subtract-one-rounding door, on real and
  complex storage, with nothing held and under a residency (gather, host arithmetic,
  scatter; or the residency's own ``subtract_double_index`` when it offers one);
* every source class's deposit is the transcription, word for word, on real and complex
  storage, with and without a conductivity, electric and magnetic;
* a silent table returns early only where MEEP's subtraction leaves every word: the
  live deposit stores the frozen always-deposit form's words on every sign class of
  amplitude, current, ``dt`` and ``cndinv``, on both storages, under keep and under flush,
  it skips exactly the deposits whose every subtrahend is ``+0``, and a silent current
  with a ``-0`` subtrahend still subtracts, which turns a stored ``-0`` into ``+0``
  exactly where MEEP does; under flush the skip relies on no subnormal being stored at a
  deposit cell, which a flushing run is pinned not to do; the skip set is judged again
  against MEEP's statements in Python doubles, on real ``Fields`` with a conductivity of
  either sign, and on a CUDA device, and the door's refusal of a non-contiguous array is
  kept on a silent step;
* the amplitudes are complex128, and a deposition table that lists a cell twice is
  refused by name;
* THE MUST-CATCH BATTERY: each planted defect -- a float32 subtraction, a contracted
  (fused) complex product, ``dt`` folded in before the product, the conductivity scale in
  float32 or before ``dt``, a dropped imaginary line -- is caught by the same comparison,
  and so is each wrong silent path -- a skip on any silent current, a blanket ``-0`` to
  ``+0``, a magnitude threshold, a sign-blind zero test, the real plane's predicate alone
  on complex storage, a ``cndinv`` sign left unread, and a ``last_time()`` predicate --
  on planted signed zeros, so the comparisons can tell MEEP's arithmetic from its near
  misses.
"""

from __future__ import annotations

import contextlib
import math
import struct
from fractions import Fraction

import numpy as np
import pytest

from . import host_writes
from . import sources as src
from .fields import Fields
from .grid import Grid
from .sources import (ContinuousEnvelope, ContinuousSource, ExtendedSource,
                      GaussianPulsedSource, VolumeSource)


def _word(value) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _words(array) -> np.ndarray:
    return np.ascontiguousarray(array).reshape(-1).view(np.uint32)


def _seeded_words(rng, n: int) -> np.ndarray:
    """float32 words across the whole range: normals, subnormals, both zeros, near-ties."""
    bits = rng.integers(0, 2 ** 32, size=n, dtype=np.uint64).astype(np.uint32)
    exponent = (bits >> 23) & 0xFF
    bits = np.where(exponent == 0xFF, bits & ~np.uint32(0x40000000), bits)  # no inf/NaN
    words = bits.view(np.float32).copy()
    words[:: 7] = np.float32(0.0)
    words[3:: 11] = np.float32(-0.0)
    words[5:: 13] = (bits[5:: 13] & np.uint32(0x807FFFFF)).view(np.float32)  # subnormals
    words[1:: 17] = rng.standard_normal(words[1:: 17].size).astype(np.float32)
    return words


def _reference_difference(word: float, x: float) -> int:
    """``float(double(f) - x)`` as one IEEE subtraction in double and one rounding."""
    return _word(np.float32(float(np.float32(word)) - x))


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

def test_subtract_double_is_one_double_subtraction_and_one_rounding():
    rng = np.random.default_rng(20261007)
    words = _seeded_words(rng, 4096)
    x = rng.standard_normal(4096) * np.float64(2.0) ** rng.integers(-140, 4, 4096)
    x[::9] = 0.0
    x[4::9] = -0.0
    array = words.copy().reshape(16, 16, 16)
    linear = np.arange(4096, dtype=np.int64)
    host_writes.subtract_double(array, linear, x)
    expected = np.array([_reference_difference(w, v) for w, v in zip(words, x)], dtype=np.uint32)
    np.testing.assert_array_equal(_words(array), expected)
    # The float32 form the door replaces is a different function on these words.
    single = (words - x.astype(np.float32)).astype(np.float32)
    assert np.count_nonzero(_words(single) != expected) > 0


def test_subtract_double_takes_meeps_second_line_on_complex_storage():
    rng = np.random.default_rng(7)
    re_words = _seeded_words(rng, 512)
    im_words = _seeded_words(rng, 512)
    array = np.empty(512, dtype=np.complex64)
    parts = array.view(np.float32).reshape(-1, 2)
    parts[:, 0], parts[:, 1] = re_words, im_words
    xr = rng.standard_normal(512) * 1e-3
    xi = rng.standard_normal(512) * 1e-3
    xr[::5] = -0.0
    linear = rng.permutation(512)[:300].astype(np.int64)
    host_writes.subtract_double(array, linear, xr[:300], xi[:300])
    got = array.view(np.float32).reshape(-1, 2)
    for k, i in enumerate(linear):
        assert _word(got[i, 0]) == _reference_difference(re_words[i], xr[k])
        assert _word(got[i, 1]) == _reference_difference(im_words[i], xi[k])
    untouched = np.setdiff1d(np.arange(512), linear)
    np.testing.assert_array_equal(_words(got[untouched, 0]), _words(re_words[untouched]))
    np.testing.assert_array_equal(_words(got[untouched, 1]), _words(im_words[untouched]))


def test_subtract_double_refuses_what_it_cannot_write_exactly():
    linear = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="not C-contiguous"):
        host_writes.subtract_double(np.zeros((4, 4), np.float32)[:, ::2], linear, np.zeros(2))
    with pytest.raises(ValueError, match="no imaginary plane"):
        host_writes.subtract_double(np.zeros(4, np.float32), linear, np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="needs imag"):
        host_writes.subtract_double(np.zeros(4, np.complex64), linear, np.zeros(2))
    with pytest.raises(ValueError, match="float32 or complex64"):
        host_writes.subtract_double(np.zeros(4, np.float64), linear, np.zeros(2))


class _HeldResidency:
    """A residency stand-in that owns one array and keeps its words elsewhere.

    The host array is stale by design (as a held Metal mirror's is) and the "device"
    copy is the truth; ``gather`` and ``scatter`` are exact copies, as the Metal doors are.
    """

    def __init__(self, array, door=False):
        self.array = array
        self.device = array.copy()
        array[...] = np.float32(1234.5) if array.dtype == np.float32 else np.complex64(1234.5)
        self.gathers = self.scatters = self.door_calls = 0
        if door:
            self.subtract_double_index = self._door

    def _by_host_identity(self, array):
        return self if array is self.array else None

    def gather(self, array, linear):
        self.gathers += 1
        return self.device.reshape(-1)[linear].copy()

    def scatter(self, array, linear, values):
        self.scatters += 1
        self.device.reshape(-1)[linear] = values

    def _door(self, array, linear, real64, imag64=None):
        self.door_calls += 1
        flat = self.device.reshape(-1)
        flat[linear] = host_writes._rounded_difference(flat[linear], real64, imag64)  # noqa: SLF001


def test_under_a_residency_the_difference_is_formed_on_the_host_from_the_owners_words():
    rng = np.random.default_rng(3)
    words = _seeded_words(rng, 256)
    plain = words.copy()
    held_host = words.copy()
    linear = np.array([3, 17, 64, 200, 255], dtype=np.int64)
    x = rng.standard_normal(5) * 1e-30
    host_writes.subtract_double(plain, linear, x)
    residency = _HeldResidency(held_host)
    host_writes.register(residency)
    try:
        host_writes.subtract_double(held_host, linear, x)
    finally:
        host_writes.unregister(residency)
    assert (residency.gathers, residency.scatters) == (1, 1)
    np.testing.assert_array_equal(_words(residency.device), _words(plain))
    assert np.all(held_host == np.float32(1234.5)), "the stale host side was written"


def test_a_residency_with_its_own_double_door_is_handed_the_deposit():
    rng = np.random.default_rng(4)
    words = _seeded_words(rng, 64)
    plain, held_host = words.copy(), words.copy()
    linear = np.array([1, 2, 30], dtype=np.int64)
    x = np.array([1e-40, -2.5, 0.0])
    host_writes.subtract_double(plain, linear, x)
    residency = _HeldResidency(held_host, door=True)
    host_writes.register(residency)
    try:
        host_writes.subtract_double(held_host, linear, x)
    finally:
        host_writes.unregister(residency)
    assert (residency.door_calls, residency.gathers, residency.scatters) == (1, 0, 0)
    np.testing.assert_array_equal(_words(residency.device), _words(plain))


# ---------------------------------------------------------------------------
# The deposit of every source class, against a scalar transcription of step.cpp
# ---------------------------------------------------------------------------

def _grid(complex_fields=False, magnetic_conductivity=False, electric_conductivity=False):
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.2, 0.0), dimensions=2)
    fields = Fields(grid=grid, force_complex_fields=complex_fields)
    rng = np.random.default_rng(99)
    for name in ("Dx", "Dy", "Dz", "Bx", "By", "Bz"):
        array = getattr(fields, name)
        if array.dtype == np.complex64:
            parts = array.view(np.float32).reshape(-1)
            parts[:] = _seeded_words(rng, parts.size)
        else:
            array.reshape(-1)[:] = _seeded_words(rng, array.size)
    if electric_conductivity:
        sigma = np.linspace(0.0, 0.7, grid.shape[0], dtype=np.float32)[:, None, None]
        fields.set_d_conductivity(np.broadcast_to(sigma, grid.shape).copy())
    if magnetic_conductivity:
        sigma = np.linspace(0.6, 0.0, grid.shape[1], dtype=np.float32)[None, :, None]
        fields.set_b_conductivity(np.broadcast_to(sigma, grid.shape).copy())
    return grid, fields


def _sources(grid):
    """One instance of every source class and spelling the deposit serves."""
    sheet = dict(center=(0.05, -0.1, 0.0), size=(0.4, 0.3, 0.0))
    return {
        "continuous point": ContinuousSource(grid=grid, frequency=0.8, component="Ez",
                                             center=(0.03, 0.07, 0.0), amplitude=0.7 - 0.2j),
        "extended sheet": ExtendedSource(grid=grid, frequency=0.8, component="Ey",
                                         amp_func=lambda x, y, z: complex(1.0 + x, 0.5 - y),
                                         **sheet),
        "gaussian sheet": GaussianPulsedSource(grid=grid, frequency=0.9, fwidth=0.4, component="Ex",
                                               amplitude=-1.3 + 0.4j, **sheet),
        "volume magnetic": VolumeSource(grid=grid, component="Hz", amplitude=0.6 + 0.9j,
                                        envelope=ContinuousEnvelope(frequency=0.7), **sheet),
        "volume electric": VolumeSource(grid=grid, component="Ez", amplitude=1.1,
                                        envelope=ContinuousEnvelope(frequency=0.6),
                                        center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0)),
    }


def _array_name(source) -> str:
    return src._ARRAY_FOR_COMPONENT[source.component]  # noqa: SLF001


def _bits64(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _transcription(fields, source, time):
    """step.cpp:302-317, point by point in Python doubles.

    Returns, per deposit cell, the double operands MEEP subtracts (``real(A)``, and
    ``imag(A)`` on complex storage) and the float32 words it would then store.
    """
    name = _array_name(source)
    array = getattr(fields, name)
    dt = source.grid.dt
    current = complex(source.current(time, dt))
    c_re, c_im = current.real, current.imag
    cndinv = fields.condinv_for(name)
    amps = np.asarray(source._point_amps).reshape(-1)  # noqa: SLF001
    cells = zip(*(np.asarray(v).reshape(-1) for v in (source._point_ix, source._point_iy,  # noqa: SLF001
                                                      source._point_iz)))
    out = {}
    for j, (i, k, m) in enumerate(cells):
        a_re, a_im = float(amps[j].real), float(amps[j].imag)
        re = (a_re * c_re) - (a_im * c_im)
        im = (a_re * c_im) + (a_im * c_re)
        re = re * dt
        im = im * dt
        if cndinv is not None:
            scale = float(cndinv[i, k, m])
            re = re * scale
            im = im * scale
        word = array[i, k, m]
        if array.dtype == np.complex64:
            out[(i, k, m)] = ((_bits64(re), _bits64(im)),
                              (_reference_difference(word.real, re), _reference_difference(word.imag, im)))
        else:
            out[(i, k, m)] = ((_bits64(re),), (_reference_difference(word, re),))
    return name, out


def _check(fields, source, time, deposit=None):
    """Inject once; count the differences from the transcription.

    Two levels, both exact: the DOUBLE operands handed to the door (64-bit words, so an
    ordering that moves only the last bit of ``real(A)`` is seen even where the float32
    rounding would hide it), and the stored float32 words (sign of zero included), with
    every cell off the deposit required unchanged.
    """
    name, expected = _transcription(fields, source, time)
    before = getattr(fields, name).copy()
    recorded = {}
    door = host_writes.subtract_double

    def recording(array, linear, real64, imag64=None):
        shape = array.shape
        for n, flat in enumerate(np.asarray(linear).reshape(-1).tolist()):
            cell = tuple(int(v) for v in np.unravel_index(flat, shape))
            parts = (_bits64(real64[n]),) if imag64 is None else (_bits64(real64[n]), _bits64(imag64[n]))
            recorded[cell] = parts
        return door(array, linear, real64, imag64)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(host_writes, "subtract_double", recording)
        if deposit is not None:
            patch.setattr(src, "_deposit", deposit)
        source.inject(fields, time)
    after = getattr(fields, name)
    differing = 0
    touched = np.zeros(after.shape, dtype=bool)
    for cell, (operands, want) in expected.items():
        touched[cell] = True
        value = after[cell]
        got = (_word(value.real), _word(value.imag)) if after.dtype == np.complex64 else (_word(value),)
        differing += sum(g != w for g, w in zip(got, want))
        handed = recorded.get(cell)
        if handed is None and all(bits == 0 for bits in operands):
            # Every subtrahend is +0, which leaves every word: the deposit may skip the
            # door (a silent table, ``sources._deposit``), and the words are still judged.
            handed = operands
        differing += sum(g != w for g, w in zip(handed or (None,) * len(operands), operands))
    # Nothing else moves, sign of zero included.
    differing += int(np.count_nonzero(_words(after[~touched]) != _words(before[~touched])))
    return differing, len(expected)


CONFIGURATIONS = {
    "real": dict(),
    "complex": dict(complex_fields=True),
    "real, D and B conductivity": dict(electric_conductivity=True, magnetic_conductivity=True),
    "complex, D and B conductivity": dict(complex_fields=True, electric_conductivity=True,
                                          magnetic_conductivity=True),
}


@pytest.mark.parametrize("configuration", sorted(CONFIGURATIONS))
def test_every_source_class_deposits_meeps_double_expression_word_for_word(configuration):
    grid, fields = _grid(**CONFIGURATIONS[configuration])
    total_points = 0
    for label, source in _sources(grid).items():
        for time in (0.35, 1.05, 2.4):
            differing, points = _check(fields, source, time)
            assert differing == 0, f"{configuration}, {label}, t={time}: {differing} words differ"
            total_points += points
    assert total_points > 0
    if "conductivity" in configuration:
        # Both families are scaled: the D conductivity on the electric deposits, the
        # B conductivity on the magnetic one (step.cpp:299-307).
        assert fields.condinv_for("Dz") is not None and fields.condinv_for("Bz") is not None


def test_an_integrated_source_is_not_scaled_and_forms_amp_times_dipole_in_double():
    """f_minus_p (update_eh.cpp:131-134): ``real(amp[j] * dipole)``, no ``dt`` and no
    ``condinv``, subtracted in double at the constitutive slot; ``inject`` deposits nothing."""
    grid, fields = _grid(electric_conductivity=True)
    source = VolumeSource(grid=grid, component="Ez", amplitude=0.9 - 0.3j,
                          center=(0.0, 0.0, 0.0), size=(0.3, 0.2, 0.0),
                          envelope=ContinuousEnvelope(frequency=0.8, is_integrated=True))
    time = 1.3
    before = fields.Dz.copy()
    source.inject(fields, time)
    assert np.array_equal(np.asarray(fields.Dz).view(np.uint32), np.asarray(before).view(np.uint32))
    dipole = complex(source.envelope.dipole(time + grid.dt))
    held = src.f_minus_p_begin(fields, [source], "D", time + grid.dt)
    amps = np.asarray(source._point_amps)  # noqa: SLF001
    for j, cell in enumerate(zip(source._point_ix, source._point_iy, source._point_iz)):  # noqa: SLF001
        a = complex(amps[j])
        re = (a.real * dipole.real) - (a.imag * dipole.imag)  # no dt, no condinv
        assert _word(fields.Dz[cell]) == _reference_difference(before[cell], re)
    src.f_minus_p_end(fields, held)
    assert np.array_equal(np.asarray(fields.Dz).view(np.uint32), np.asarray(before).view(np.uint32))


def test_a_silent_current_with_a_minus_zero_subtrahend_turns_a_stored_minus_zero_into_plus_zero():
    """MEEP always subtracts. ``(-0) - (-0) = +0``, so this deposit cannot be skipped.

    A negative real amplitude and the current ``(+0, +0)`` form ``real(A) = -0`` at every
    point: the table is not word-neutral, and the deposit must still reach the door.
    """
    grid, fields = _grid()
    pulse = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component="Ez",
                                 amplitude=-1.0, center=(0.0, 0.0, 0.0), size=(0.2, 0.0, 0.0))
    far = 1.0e4  # far outside the pulse: the current is exactly zero
    assert pulse.current(far, grid.dt) == 0
    cells = list(zip(pulse._point_ix, pulse._point_iy, pulse._point_iz))  # noqa: SLF001
    for cell in cells:
        fields.Dz[cell] = np.float32(-0.0)
    pulse.inject(fields, far)
    assert [_word(fields.Dz[c]) for c in cells] == [0x00000000] * len(cells), (
        "a stored -0 met real(A) = -0 (negative amplitude, zero current) and MEEP's one "
        "double subtraction gives +0; the deposit skipped it")

    def early_return(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
        if complex(factor) == 0:
            return None
        return deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv)

    deposit = src._deposit  # noqa: SLF001
    for cell in cells:
        fields.Dz[cell] = np.float32(-0.0)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(src, "_deposit", early_return)
        pulse.inject(fields, far)
    assert [_word(fields.Dz[c]) for c in cells] == [0x80000000] * len(cells), (
        "the control did not keep the -0, so the assertion above cannot see an early return")


def test_the_amplitudes_are_complex128():
    grid, _fields = _grid()
    for label, source in _sources(grid).items():
        assert source._point_amps.dtype == np.complex128, label  # noqa: SLF001


def test_a_table_that_lists_a_cell_twice_is_refused_by_name():
    grid, fields = _grid()
    with pytest.raises(ValueError, match="same cell more than once"):
        src._point_index_arrays(grid, [(1, 2, 0), (3, 3, 0), (1, 2, 0)])  # noqa: SLF001
    ix = np.array([4, 4], dtype=np.int64)
    iy = np.array([5, 5], dtype=np.int64)
    iz = np.array([0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="same cell more than once"):
        src._deposit(fields.Dz, ix, iy, iz, np.array([1.0, 2.0], dtype=np.complex128),  # noqa: SLF001
                     1.0 + 0j, grid.dt, "Ez")


# ---------------------------------------------------------------------------
# The must-catch battery
# ---------------------------------------------------------------------------

def _float32_subtraction(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    product = (amps * complex(factor)) * dt
    if cndinv is not None:
        product = product * cndinv[ix, iy, iz]
    if array.dtype == np.complex64:
        array[ix, iy, iz] -= product.astype(np.complex64)
    else:
        array[ix, iy, iz] -= product.real.astype(np.float32)


def _fma(a: float, b: float, c: float) -> float:
    """``a * b + c`` rounded ONCE, as a contracting compiler forms it (exact rationals)."""
    return float(Fraction(a) * Fraction(b) + Fraction(c))


def _fused_complex_product(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    """The contracted complex product a fusing host forms: ``fma(ar, cr, -(ai*ci))``.

    Spelled with exact rationals so the mutant is the same on every host: NumPy's own
    complex multiply contracts on some hosts and not on others (phase-0 complex family:
    491,977 of 2,000,000 words on an AVX-512 x86 host), which would make a battery
    built on it host-dependent.
    """
    c = complex(factor)
    real = np.array([_fma(a.real, c.real, -(a.imag * c.imag)) for a in amps.tolist()]) * dt
    imag = np.array([_fma(a.real, c.imag, a.imag * c.real) for a in amps.tolist()]) * dt
    if cndinv is not None:
        scale = cndinv[ix, iy, iz].astype(np.float64)
        real, imag = real * scale, imag * scale
    linear = src._linear_points(array, ix, iy, iz)  # noqa: SLF001
    host_writes.subtract_double(array, linear, real, imag if array.dtype == np.complex64 else None)


def _dt_folded_first(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    return DEPOSIT(array, ix, iy, iz, amps, complex(factor) * dt, None, component, cndinv)


def _condinv_before_dt(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    if cndinv is None:
        return DEPOSIT(array, ix, iy, iz, amps, factor, dt, component, None)
    scaled = amps * cndinv[ix, iy, iz].astype(np.float64)
    return DEPOSIT(array, ix, iy, iz, scaled, factor, dt, component, None)


def _condinv_in_float32(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    if cndinv is None:
        return DEPOSIT(array, ix, iy, iz, amps, factor, dt, component, None)
    rounded = (amps * complex(factor) * dt).astype(np.complex64) * cndinv[ix, iy, iz]
    linear = src._linear_points(array, ix, iy, iz)  # noqa: SLF001
    real = rounded.real.astype(np.float64)
    imag = rounded.imag.astype(np.float64) if array.dtype == np.complex64 else None
    host_writes.subtract_double(array, linear, real, imag)


def _imaginary_line_dropped(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    if array.dtype != np.complex64:
        return DEPOSIT(array, ix, iy, iz, amps, factor, dt, component, cndinv)
    linear = src._linear_points(array, ix, iy, iz)  # noqa: SLF001
    c = complex(factor)
    real = (amps.real * c.real - amps.imag * c.imag) * dt
    if cndinv is not None:
        real = real * cndinv[ix, iy, iz].astype(np.float64)
    host_writes.subtract_double(array, linear, real, np.zeros_like(real))


DEPOSIT = src._deposit  # noqa: SLF001

MUTANTS = {
    "float32 subtraction": (_float32_subtraction, "real"),
    "fused complex product": (_fused_complex_product, "complex"),
    "dt folded in before the product": (_dt_folded_first, "real"),
    "condinv applied before dt": (_condinv_before_dt, "real, D and B conductivity"),
    "condinv in float32": (_condinv_in_float32, "real, D and B conductivity"),
    "imaginary line dropped": (_imaginary_line_dropped, "complex"),
}


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_the_comparison_catches_every_planted_defect(mutant):
    """Each mutant is a plausible spelling of the deposit; the transcription sees each one.

    A comparison that cannot tell these from MEEP's expression would pass for the wrong
    reason, so every one must move at least one stored word across the five sources and
    three times on the configuration it is planted on.
    """
    deposit, configuration = MUTANTS[mutant]
    grid, fields = _grid(**CONFIGURATIONS[configuration])
    caught = 0
    for source in _sources(grid).values():
        for time in (0.35, 1.05, 2.4):
            differing, _points = _check(fields, source, time, deposit=deposit)
            caught += differing
    assert caught > 0, f"the planted defect '{mutant}' left every stored word as MEEP's"


# ---------------------------------------------------------------------------
# The silent table: an early return only where MEEP's subtraction leaves every word
# ---------------------------------------------------------------------------

def _always_deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    """MEEP's statement on every call, with no silent return: the deposit as 1.0 shipped it."""
    keep_real = src._injects_real(array, component)  # noqa: SLF001
    linear = src._linear_points(array, ix, iy, iz)  # noqa: SLF001
    current = complex(factor)
    real = amps.real * current.real
    real -= amps.imag * current.imag
    if dt is not None:
        real *= float(dt)
    scale = None
    if cndinv is not None:
        scale = cndinv[ix, iy, iz].astype(np.float64)
        real *= scale
    if keep_real:
        host_writes.subtract_double(array, linear, real)
        return
    imag = amps.real * current.imag
    imag += amps.imag * current.real
    if dt is not None:
        imag *= float(dt)
    if scale is not None:
        imag *= scale
    host_writes.subtract_double(array, linear, real, imag)


@contextlib.contextmanager
def _host_policy(policy):
    """Run in ``keep`` or ``flush``; skip where this host's FPU cannot be driven there."""
    from . import backends, subnormal_policy  # noqa: PLC0415

    before = bool(backends.subnormals_flushed())
    want = policy == "flush"
    try:
        if before != want:
            subnormal_policy._drive_host_fpu(want)  # noqa: SLF001
        if bool(backends.subnormals_flushed()) != want:
            from conftest import requires_resource_skip  # noqa: PLC0415 - the root conftest

            requires_resource_skip("host_fpu_mode", f"this host's FPU cannot be driven to {policy}")
        yield
    finally:
        if bool(backends.subnormals_flushed()) != before:
            subnormal_policy._drive_host_fpu(before)  # noqa: SLF001


def _bitwise_plus_zero_current(factor) -> bool:
    current = complex(factor)
    return (current.real == 0.0 and current.imag == 0.0 and math.copysign(1.0, current.real) > 0
            and math.copysign(1.0, current.imag) > 0)


def _is_subnormal(words):
    return ((words & np.uint32(0x7F800000)) == 0) & ((words & np.uint32(0x007FFFFF)) != 0)


#: Amplitude parts: both zeros, both signs of a normal and of a tiny double, both
#: infinities and NaN -- every sign class the predicate reads and the non-finite ones it
#: must refuse.
_PARTS = (0.0, -0.0, 0.75, -0.75, 1e-300, -1e-300, math.inf, -math.inf, math.nan)
#: Currents: the four signed zeros, two tail currents (a Gaussian's last on-steps reach
#: |J| near 1e-100) and an ordinary one.
_CURRENTS = (complex(0.0, 0.0), complex(-0.0, 0.0), complex(0.0, -0.0), complex(-0.0, -0.0),
             complex(1e-100, 0.0), complex(0.0, -1e-100), complex(0.5, -0.25))
#: Stored words: both zeros, the extreme subnormals of each sign, +-1, +-inf, a quiet NaN.
_STORED = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,
                    0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000], dtype=np.uint32)
#: ``cndinv`` at the table's cells: lossless, positive, +0, negative (sigma below -2/dt),
#: +inf, NaN, and one negative cell among positive ones.
_SCALES = ("lossless", 1.0, 0.4, 0.0, -2.0, math.inf, math.nan, "one negative cell")


def _silent_sweep(policy, deposit):
    """Every (amplitude, current, ``dt``, ``cndinv``, storage) class: ``deposit`` against the frozen form.

    Each table repeats one amplitude at every stored-word class (a flagged point is then
    met by a stored ``-0`` in each plane), plus two tables with one flagged point among
    unflagged ones. Returns the counts the tests read: differing words, deposits, the
    deposits ``deposit`` skipped (handed the door nothing), those of them whose frozen
    operands were not all ``+0``, and the silent, sign-clear, all-``+0`` deposits it did
    not skip. Under flush the stored subnormals are left out: the skip is exact there
    only because a flushing run stores none (pinned on a run below).
    """
    stored = _STORED[~_is_subnormal(_STORED)] if policy == "flush" else _STORED
    n = int(stored.size)
    ix = np.arange(n, dtype=np.int64)
    iy = np.zeros(n, dtype=np.int64)
    iz = np.zeros(n, dtype=np.int64)
    volumes = {}
    for scale in _SCALES:
        if scale == "lossless":
            volumes[scale] = None
        elif scale == "one negative cell":
            volumes[scale] = np.full((n, 1, 1), 0.4, dtype=np.float32)
            volumes[scale][n // 2] = -2.0
        else:
            volumes[scale] = np.full((n, 1, 1), scale, dtype=np.float32)
    tables = [np.full(n, complex(re, im), dtype=np.complex128) for re in _PARTS for im in _PARTS]
    for odd in (complex(-0.75, 0.0), complex(-0.75, -0.75)):  # flagged: real plane; imaginary
        mixed = np.full(n, complex(0.75, 0.25), dtype=np.complex128)
        mixed[n // 3] = odd
        tables.append(mixed)
    counts = {"deposits": 0, "differing": 0, "skipped": 0, "skipped_not_neutral": 0,
              "neutral_not_skipped": 0}
    door = host_writes.subtract_double
    handed = []

    def recording(array, linear, real64, imag64=None):
        handed.extend(part.view(np.uint64).tolist() for part in (real64, imag64) if part is not None)
        return door(array, linear, real64, imag64)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(host_writes, "subtract_double", recording)
        for dtype, shifts in ((np.float32, (0,)), (np.complex64, (0, n // 2))):
            for shift in shifts:
                base = np.zeros((n, 1, 1), dtype=dtype)
                words = base.reshape(-1).view(np.uint32)
                if dtype == np.float32:
                    words[:] = stored
                else:
                    words.reshape(n, 2)[:, 0] = stored
                    words.reshape(n, 2)[:, 1] = np.roll(stored, shift)
                for amps in tables:
                    for current in _CURRENTS:
                        for dt in (0.05, None):
                            for cndinv in volumes.values():
                                live, frozen = base.copy(), base.copy()
                                handed.clear()
                                deposit(live, ix, iy, iz, amps, current, dt, "Ez", cndinv)
                                skipped = not handed
                                handed.clear()
                                _always_deposit(frozen, ix, iy, iz, amps, current, dt, "Ez", cndinv)
                                neutral = all(bits == 0 for part in handed for bits in part)
                                sign_clear = cndinv is None or not np.signbit(cndinv).any()
                                counts["deposits"] += 1
                                counts["differing"] += int(np.count_nonzero(_words(live) != _words(frozen)))
                                counts["skipped"] += skipped
                                counts["skipped_not_neutral"] += skipped and not neutral
                                counts["neutral_not_skipped"] += (
                                    _bitwise_plus_zero_current(current) and neutral and sign_clear
                                    and not skipped)
    return counts


@pytest.mark.parametrize("policy", ("keep", "flush"))
def test_the_silent_return_stores_the_always_deposit_forms_words_on_every_sign_class(policy):
    """The live deposit against MEEP's statement run on every call, word for word.

    It must store the same words (signs of zero, NaN and infinities included), skip only
    a deposit whose every subtrahend is ``+0``, and skip every silent one that is: the
    current bitwise ``(+0, +0)``, a sign-clear ``cndinv``, and all-``+0`` operands.
    """
    with np.errstate(all="ignore"), _host_policy(policy):
        counts = _silent_sweep(policy, src._deposit)  # noqa: SLF001
    assert counts["differing"] == 0, counts
    assert counts["skipped_not_neutral"] == 0, counts
    assert counts["neutral_not_skipped"] == 0, counts
    assert 0 < counts["skipped"] < counts["deposits"], counts


def test_under_flush_the_skip_leaves_a_stored_subnormal_that_meeps_subtraction_zeroes():
    """Why the skip needs a flushing run to store no subnormal at a deposit cell.

    Under flush the widening reads a subnormal word as a zero of its sign, so MEEP's
    subtraction of ``+0`` stores that zero; the skip leaves the word. A flushing run
    stores none -- every arithmetic result is flushed, and the fills and wall clears copy,
    negate or zero -- which the run below pins at every silent deposit.
    """
    words = np.array([0x00000001, 0x807FFFFF, 0x00400000, 0x80000001], dtype=np.uint32)
    ix, iy, iz = np.arange(4, dtype=np.int64), np.zeros(4, np.int64), np.zeros(4, np.int64)
    amps = np.full(4, complex(0.75, 0.0), dtype=np.complex128)
    live = words.view(np.float32).reshape(4, 1, 1).copy()
    frozen = live.copy()
    with _host_policy("flush"):
        src._deposit(live, ix, iy, iz, amps, 0j, 0.05, "Ez")  # noqa: SLF001
        _always_deposit(frozen, ix, iy, iz, amps, 0j, 0.05, "Ez")
    assert _words(live).tolist() == words.tolist()
    assert _words(frozen).tolist() == [0x00000000, 0x80000000, 0x00000000, 0x80000000]


def _conductive_run(amplitude, deposit, steps=300):
    """A 2-D run in a strong D and B conductivity, its pulse over by step 40, every step digested.

    ``sigma * dt / 2 = 1.5`` makes ``condfac`` negative, so once the run decays every quiet
    word passes through signed zeros, the deposit cells included. ``deposit`` stands in
    for ``sources._deposit``; the counts record what each silent deposit found at its cells.
    """
    from . import meep_identity  # noqa: PLC0415
    from .driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(cell_size=(3.0, 3.0, 0.0), resolution=10.0, dimensions=2)
    sigma = 1.5 / (0.5 * driver.grid.dt)
    driver.set_conductivity(sigma)
    driver.set_b_conductivity(sigma)
    driver.add_source({"component": "Ez", "source_type": "gaussian", "frequency": 1.0,
                       "fwidth": 2.0, "cutoff": 2.0, "center": (0.15, -0.05, 0.0),
                       "amplitude": amplitude})
    masks = {name: meep_identity.cell_classes(driver.grid, name)["compared"]
             for name in meep_identity.COMPONENTS}
    counts = {"silent": 0, "skipped": 0, "minus_zero_at_cells": 0, "subnormal_at_cells": 0}
    door = host_writes.subtract_double
    calls = [0]

    def counting_door(array, linear, real64, imag64=None):
        calls[0] += 1
        return door(array, linear, real64, imag64)

    def watching(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
        if complex(factor) == 0:
            counts["silent"] += 1
            cells = _words(array)[src._linear_points(array, ix, iy, iz)]  # noqa: SLF001
            counts["minus_zero_at_cells"] += int(np.count_nonzero(cells == 0x80000000))
            counts["subnormal_at_cells"] += int(np.count_nonzero(_is_subnormal(cells)))
            before = calls[0]
            deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv)
            counts["skipped"] += calls[0] == before
            return None
        return deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv)

    digests = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(host_writes, "subtract_double", counting_door)
        patch.setattr(src, "_deposit", watching)
        for _ in range(steps):
            driver.step()
            digests.append(meep_identity.step_digests(driver, masks, planes=False,
                                                      reader=False).stored)
    return digests, counts


@pytest.mark.parametrize("policy", ("keep", "flush"))
@pytest.mark.parametrize("amplitude", (1.0, -1.0), ids=("unflagged", "flagged"))
def test_a_run_past_its_pulse_stores_the_always_deposit_forms_words_at_every_step(policy, amplitude):
    """Every stored word at every step, live against the frozen form, on a run whose silent
    deposits meet ``-0`` at their cells. Amplitude +1 is unflagged (skipped, the ``-0`` kept,
    as MEEP keeps it); -1 is flagged (deposited, the ``-0`` turned ``+0``, as MEEP does).
    Under flush no silent deposit finds a subnormal at its cells: the skip's condition."""
    deposit = src._deposit  # noqa: SLF001
    with np.errstate(all="ignore"), _host_policy(policy):
        live, counts = _conductive_run(amplitude, deposit)
        frozen, _frozen_counts = _conductive_run(amplitude, _always_deposit)
    differing = [(step, name) for step, (a, b) in enumerate(zip(live, frozen))
                 for name in a if a[name] != b[name]]
    assert not differing, (counts, differing[:5])
    assert counts["silent"] >= 250 and counts["minus_zero_at_cells"] > 0, counts
    assert counts["skipped"] == (counts["silent"] if amplitude > 0 else 0), counts
    if policy == "flush":
        assert counts["subnormal_at_cells"] == 0, counts


def _silent_sources(grid):
    """Every source class at a time it is silent, in both flag classes."""
    sheet = dict(center=(0.05, -0.1, 0.0), size=(0.4, 0.3, 0.0))
    point = dict(center=(0.03, 0.07, 0.0), size=(0.0, 0.0, 0.0))
    return {
        "continuous point, before its start": (ContinuousSource(
            grid=grid, frequency=0.8, component="Ez", amplitude=0.7 - 0.2j, start_time=5.0,
            center=point["center"]), 1.0),
        "continuous point, negative, after its end": (ContinuousSource(
            grid=grid, frequency=0.8, component="Ez", amplitude=-0.7, end_time=1.0,
            center=point["center"]), 3.0),
        "extended sheet, after its end": (ExtendedSource(
            grid=grid, frequency=0.8, component="Ey", end_time=1.0,
            amp_func=lambda x, y, z: complex(1.0 + 4 * x, 0.5 - 4 * y), **sheet), 3.0),
        "gaussian sheet, flagged, after its pulse": (GaussianPulsedSource(
            grid=grid, frequency=0.9, fwidth=0.4, component="Ex", amplitude=-1.3 + 0.4j,
            **sheet), 1.0e4),
        "gaussian sheet, unflagged, after its pulse": (GaussianPulsedSource(
            grid=grid, frequency=0.9, fwidth=0.4, component="Ex", amplitude=1.3, **sheet), 1.0e4),
        "gaussian point, imaginary flag, after its pulse": (GaussianPulsedSource(
            grid=grid, frequency=0.9, fwidth=0.4, component="Ez", amplitude=-1.0 - 1.0j,
            **point), 1.0e4),
        "volume magnetic, after its end": (VolumeSource(
            grid=grid, component="Hz", amplitude=0.6 + 0.9j,
            envelope=ContinuousEnvelope(frequency=0.7, end_time=1.0), **sheet), 3.0),
        "volume electric, negative, before its start": (VolumeSource(
            grid=grid, component="Ez", amplitude=-1.1, envelope=ContinuousEnvelope(
                frequency=0.6, start_time=4.0), **point), 0.5),
    }


@pytest.mark.parametrize("configuration", sorted(CONFIGURATIONS))
def test_every_source_class_at_a_silent_time_stores_meeps_words_on_planted_signed_zeros(configuration):
    """step.cpp:302-317 at a silent time, with ``-0`` planted at every deposit cell.

    The transcription (:func:`_check`) forms each subtrahend and the word MEEP stores, so
    a flagged table must flip each ``-0`` it forms a ``-0`` subtrahend against, and an
    unflagged one must leave it; the classes cover both, before a start and after an end.
    """
    grid, fields = _grid(**CONFIGURATIONS[configuration])
    seen = {"skipped": 0, "deposited": 0}
    for label, (source, time) in _silent_sources(grid).items():
        assert _bitwise_plus_zero_current(source.current(time, grid.dt)), label
        array = getattr(fields, _array_name(source))
        cells = tuple(np.asarray(v).reshape(-1) for v in (source._point_ix, source._point_iy,  # noqa: SLF001
                                                           source._point_iz))
        array[cells] = np.complex64(complex(-0.0, -0.0)) if array.dtype == np.complex64 else np.float32(-0.0)
        differing, points = _check(fields, source, time)
        assert differing == 0, f"{configuration}, {label}: {differing} words differ"
        cndinv = fields.condinv_for(_array_name(source))
        skipped = src._silent_deposit_is_identity(  # noqa: SLF001
            source._point_amps, source._point_ix, source._point_iy, source._point_iz,  # noqa: SLF001
            grid.dt, cndinv, array.dtype == np.float32)
        seen["skipped" if skipped else "deposited"] += 1
    assert seen["skipped"] > 0 and seen["deposited"] > 0, seen


def test_an_integrated_sources_silent_dipole_keeps_or_flips_minus_zero_as_meep_does():
    """``f_minus_p`` (update_eh.cpp:131-134) with a silent dipole: ``real(amp * 0)``, no ``dt``."""
    for amplitude, want in ((0.9, 0x80000000), (-0.9, 0x00000000)):
        grid, fields = _grid()
        source = VolumeSource(grid=grid, component="Ez", amplitude=amplitude,
                              center=(0.0, 0.0, 0.0), size=(0.3, 0.2, 0.0),
                              envelope=ContinuousEnvelope(frequency=0.8, start_time=5.0,
                                                          is_integrated=True))
        time = 1.0
        assert _bitwise_plus_zero_current(source.envelope.dipole(time + grid.dt))
        cells = list(zip(source._point_ix, source._point_iy, source._point_iz))  # noqa: SLF001
        for cell in cells:
            fields.Dz[cell] = np.float32(-0.0)
        before = fields.Dz.copy()
        held = src.f_minus_p_begin(fields, [source], "D", time + grid.dt)
        assert [_word(fields.Dz[c]) for c in cells] == [want] * len(cells), amplitude
        src.f_minus_p_end(fields, held)
        assert np.array_equal(_words(fields.Dz), _words(before))


def test_a_conductivity_replaced_after_a_silent_deposit_is_read_again():
    """The cached predicate follows the ``cndinv`` volume by identity, never by a stale id.

    A positive conductivity first (the table skips, its ``-0`` kept), then one below
    ``-2 / dt`` (``cndinv`` negative, so ``real(A) = +0 * -cndinv = -0``): MEEP flips the
    ``-0``, and the deposit must too, though the table and the cells are the same.
    """
    grid, fields = _grid()
    source = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.4, component="Ez",
                                  amplitude=1.0, center=(0.0, 0.0, 0.0), size=(0.2, 0.0, 0.0))
    cells = list(zip(source._point_ix, source._point_iy, source._point_iz))  # noqa: SLF001
    for sigma, want in ((0.5, 0x80000000), (-3.0 / grid.dt, 0x00000000), (0.5, 0x80000000)):
        fields.set_d_conductivity(np.full(grid.shape, sigma, dtype=np.float32))
        for cell in cells:
            fields.Dz[cell] = np.float32(-0.0)
        source.inject(fields, 1.0e4)
        assert [_word(fields.Dz[c]) for c in cells] == [want] * len(cells), sigma


def _skip_any_silent_current(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    if complex(factor) == 0:
        return None
    return _always_deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv)


def _canonicalise_minus_zero(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    if complex(factor) != 0:
        return _always_deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv)
    words = _words(array)
    linear = src._linear_points(array, ix, iy, iz)  # noqa: SLF001
    flat = np.concatenate([2 * linear, 2 * linear + 1]) if array.dtype == np.complex64 else linear
    words[flat[words[flat] == 0x80000000]] = 0
    return None


def _threshold(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    if abs(complex(factor)) < 1e-30:  # 0.9.2's Gaussian class
        return None
    return _always_deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv)


def _skip_with(read_scale, sign_blind=False, imaginary_plane=True, finite=True):
    """A predicate one reading short of the shipped one, then the always-deposit statement."""
    def mutant(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
        current = complex(factor)
        silent = current == 0 if sign_blind else _bitwise_plus_zero_current(current)
        if silent:
            values = np.asarray(amps).reshape(-1)
            negative_re, negative_im = np.signbit(values.real), np.signbit(values.imag)
            usable = (not finite) or bool(np.isfinite(values.real).all() and np.isfinite(values.imag).all())
            if read_scale and cndinv is not None:
                scale = cndinv[ix, iy, iz]
                usable = usable and bool(np.isfinite(scale).all() and not np.signbit(scale).any())
            real_left = not (negative_re & ~negative_im).any()
            imag_left = not (negative_re & negative_im).any()
            if usable and real_left and (array.dtype == np.float32 or imag_left or not imaginary_plane):
                return None
        return _always_deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv)
    return mutant


def _one_zero_sign_flipped(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    """The live deposit, then a planted wrong sign of zero on one stored zero at its cells."""
    src._deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv)  # noqa: SLF001
    words = _words(array)
    linear = src._linear_points(array, ix, iy, iz)  # noqa: SLF001
    real_words = 2 * linear if array.dtype == np.complex64 else linear
    zeros = real_words[(words[real_words] & np.uint32(0x7FFFFFFF)) == 0]
    if zeros.size:
        words[zeros[0]] ^= np.uint32(0x80000000)


SILENT_MUTANTS = {
    "one stored zero's sign flipped after the live deposit": _one_zero_sign_flipped,
    "skip on any silent current": _skip_any_silent_current,
    "blanket -0 to +0 on a silent current": _canonicalise_minus_zero,
    "abs(current) < 1e-30": _threshold,
    "sign-blind zero test on the current": _skip_with(True, sign_blind=True),
    "real plane's predicate alone on complex storage": _skip_with(True, imaginary_plane=False),
    "cndinv sign left unread": _skip_with(False),
    "amplitude finiteness left unread": _skip_with(True, finite=False),
}


@pytest.mark.parametrize("mutant", sorted(SILENT_MUTANTS))
def test_the_silent_sweep_catches_every_wrong_silent_path(mutant):
    """Each is a plausible early return; the sweep must see each move a stored word."""
    with np.errstate(all="ignore"), _host_policy("keep"):
        counts = _silent_sweep("keep", SILENT_MUTANTS[mutant])
    assert counts["differing"] > 0, (mutant, counts)


def test_a_last_time_predicate_is_caught_on_the_pulses_last_on_step():
    """``last_time()`` is not the zero test (bands.cpp:35-41 reads it; step_source never does).

    On the slot whose later sample is past the window bound, the current is
    ``-dipole(t) / dt``, not zero; a source that returns once ``t + dt`` passes
    ``peak + cutoff`` skips a real deposit.
    """
    grid, fields = _grid()
    pulse = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=2.0, cutoff=2.0, component="Ez",
                                 amplitude=1.0, center=(0.0, 0.0, 0.0), size=(0.2, 0.0, 0.0))
    end = pulse._peak_time + pulse._cutoff_time  # noqa: SLF001
    slots = [(n + 0.5) * grid.dt for n in range(int(end / grid.dt) + 4)]
    time = next(t for t in slots if t + grid.dt > end and complex(pulse.current(t, grid.dt)) != 0)
    differing, _points = _check(fields, pulse, time)
    assert differing == 0
    read = GaussianPulsedSource.inject

    def last_time_inject(self, fields, time):
        if time + self.grid.dt > self._peak_time + self._cutoff_time:  # noqa: SLF001
            return None
        return read(self, fields, time)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(GaussianPulsedSource, "inject", last_time_inject)
        caught, _points = _check(fields, pulse, time)
    assert caught > 0, "the last_time() predicate skipped no deposit that MEEP makes"


# ---------------------------------------------------------------------------
# MEEP's src_vol list: per chunk, merged in double (fields.cpp:576-585, sources.cpp:179-188)
# ---------------------------------------------------------------------------

def _merge_grid():
    from .chunk_map import ChunkLayout  # noqa: PLC0415
    from .pml import PML  # noqa: PLC0415

    grid = Grid(resolution=10, cell_size=(3.0, 3.0, 0.0), dimensions=2)
    pml = PML(grid=grid, thickness=(4, 4, 0))
    return grid, ChunkLayout.one_rank(grid, pml)


def _line(grid, amplitude, frequency=1.0, fwidth=0.7, component="Ez", integrated=False, cls=None):
    if cls is None and not integrated:
        return GaussianPulsedSource(grid=grid, frequency=frequency, fwidth=fwidth, component=component,
                                    center=(0.25, 0.0, 0.0), size=(0.0, 3.0, 0.0), amplitude=amplitude)
    from .sources import GaussianEnvelope  # noqa: PLC0415

    return VolumeSource(grid=grid, component=component, center=(0.25, 0.0, 0.0), size=(0.0, 3.0, 0.0),
                        amplitude=amplitude, envelope=GaussianEnvelope(frequency=frequency, fwidth=fwidth,
                                                                       is_integrated=integrated))


def test_equal_src_times_on_equal_pieces_merge_per_chunk_with_the_amplitudes_summed_in_double():
    """MEEP combines a piece into an entry with the same component, src_time and index vector.

    The line crosses three chunks of the one-rank division (two layers and the interior),
    so it is three pieces, and each merges into the first source's piece in that chunk:
    the first source deposits the summed amplitudes, the second deposits nothing.
    """
    grid, layout = _merge_grid()
    first, second = _line(grid, 0.9 - 0.1j), _line(grid, -0.4 + 0.25j)
    counts = src.bind_src_vols([first, second], layout, grid)
    assert counts["merged"] == counts["pieces"] // 2 >= 3, counts
    assert second._src_vol == ()  # noqa: SLF001
    entries = first._src_vol  # noqa: SLF001
    assert sum(entry.points for entry in entries) >= len(first._indices)  # noqa: SLF001
    first_pieces = src._src_vol_pieces(first._src_geometry, layout.boxes, grid)  # noqa: SLF001
    second_pieces = src._src_vol_pieces(second._src_geometry, layout.boxes, grid)  # noqa: SLF001
    assert len(entries) == len(first_pieces) == len(second_pieces)
    for entry, (_c1, cells, a1), (_c2, cells2, a2) in zip(entries, first_pieces, second_pieces):
        assert cells == cells2
        for ix, iy, iz, amp, x, y in zip(entry.ix, entry.iy, entry.iz, np.asarray(entry.amps), a1, a2):
            assert (int(ix), int(iy), int(iz)) in cells
            assert complex(amp) == complex(x.real + y.real, x.imag + y.imag)


@pytest.mark.parametrize("change", ["src_time", "integrated", "array"])
def test_pieces_that_differ_in_src_time_flag_or_array_do_not_merge(change):
    grid, layout = _merge_grid()
    first = _line(grid, 0.9)
    second = {"src_time": lambda: _line(grid, 0.5, frequency=1.2),
              "integrated": lambda: _line(grid, 0.5, integrated=True),
              "array": lambda: _line(grid, 0.5, component="Ey")}[change]()
    counts = src.bind_src_vols([first, second], layout, grid)
    assert counts["merged"] == 0, counts
    assert len(first._src_vol) + len(second._src_vol) == counts["entries"] == counts["pieces"]  # noqa: SLF001


def test_the_d_and_e_spellings_of_one_component_merge():
    """MEEP folds a D-family component onto its E name before comparing (sources.cpp:167-173)."""
    grid, layout = _merge_grid()
    from .sources import GaussianEnvelope  # noqa: PLC0415

    as_e = _line(grid, 0.9)
    as_d = VolumeSource(grid=grid, component="Dz", center=(0.25, 0.0, 0.0), size=(0.0, 3.0, 0.0),
                        amplitude=0.3, envelope=GaussianEnvelope(frequency=1.0, fwidth=0.7))
    assert src.source_time_key(as_e) == src.source_time_key(as_d)
    counts = src.bind_src_vols([as_e, as_d], layout, grid)
    assert counts["merged"] >= 3, counts


def test_a_merged_pair_deposits_once_and_rounds_once():
    """``f -= real(((a1 + a2) * J) * dt)`` per merged entry: one subtraction per entry, where
    the two sources' own tables deposited one after the other round twice."""
    grid, layout = _merge_grid()
    rng = np.random.default_rng(1007)
    first, second = _line(grid, 0.9 - 0.1j), _line(grid, -0.4 + 0.25j)
    src.bind_src_vols([first, second], layout, grid)
    pieces1 = src._src_vol_pieces(first._src_geometry, layout.boxes, grid)  # noqa: SLF001
    pieces2 = src._src_vol_pieces(second._src_geometry, layout.boxes, grid)  # noqa: SLF001
    merged, separate = Fields(grid=grid), Fields(grid=grid)
    merged.Dz[...] = rng.standard_normal(merged.Dz.shape).astype(np.float32)
    differing = 0
    for step in range(40):
        time = 2.0 + step * grid.dt
        before = merged.Dz.copy()
        first.inject(merged, time)
        second.inject(merged, time)
        current = first.current(time, grid.dt)
        expected = before.copy()
        for (_chunk, cells, a1), (_chunk2, cells2, a2) in zip(pieces1, pieces2):
            assert cells == cells2
            for cell, x, y in zip(cells, a1, a2):
                a = complex(x.real + y.real, x.imag + y.imag)
                re = ((a.real * current.real) - (a.imag * current.imag)) * grid.dt
                expected[cell] = np.float32(float(np.float32(expected[cell])) - re)
        assert np.array_equal(_words(merged.Dz), _words(expected))
        # The unmerged order: each source's own table, one rounding each.
        separate.Dz[...] = before
        for source in (first, second):
            src._deposit(separate.Dz, source._point_ix, source._point_iy, source._point_iz,  # noqa: SLF001
                         source._point_amps, source.current(time, grid.dt), grid.dt, "Ez")
        differing += int(np.count_nonzero(_words(merged.Dz) != _words(separate.Dz)))
    assert differing > 0, "the merge changed no word; the comparison cannot see it"


# ---------------------------------------------------------------------------
# A periodic seam: two entries on one cell, deposited one after the other
# (loop_in_chunks.cpp:390-500, sources.cpp:311, fields.cpp:576-585, step.cpp:309-314)
# ---------------------------------------------------------------------------

def _seam_grid():
    """A periodic y with no absorber: a full-height line's two ends are one lattice vector apart."""
    from .chunk_map import ChunkLayout  # noqa: PLC0415

    grid = Grid(resolution=10, cell_size=(2.0, 3.0, 0.0), dimensions=2)
    return grid, ChunkLayout.one_rank(grid, None)


def test_a_full_height_line_on_a_periodic_axis_places_its_far_end_as_a_piece_of_its_own():
    """The ishift = 0 piece ends on the seam row with its edge weight and the ishift = -1
    piece is that row alone with the other end's weight: two entries, not one summed cell."""
    grid, layout = _seam_grid()
    line = _line(grid, 1.0)
    shifts = [source_pass.ishift for source_pass in line._src_geometry.passes]  # noqa: SLF001
    assert (0, 0, 0) in shifts and len(shifts) == 2, shifts
    pieces = src._src_vol_pieces(line._src_geometry, layout.boxes, grid)  # noqa: SLF001
    assert len(pieces) == 2
    image, direct = sorted(pieces, key=lambda piece: len(piece[1]))
    # The image is the seam row alone (both x columns of the off-sample line).
    assert {cell[1] for cell in image[1]} == {0} and set(image[1]) <= set(direct[1])
    totals = dict(zip(line._indices, line._amplitudes))  # noqa: SLF001
    for cell, other in zip(image[1], image[2]):
        one = direct[2][direct[1].index(cell)]
        assert 0.0 < abs(one) < abs(totals[cell]) and 0.0 < abs(other) < abs(totals[cell])
        assert totals[cell] == complex(one.real + other.real, one.imag + other.imag)
    src.bind_src_vols([line], layout, grid)
    assert len(line._src_vol) == 2  # noqa: SLF001


def test_the_seam_cell_is_deposited_twice_and_rounds_twice():
    """``f -= real(A1); f -= real(A2)`` in double, each rounded to float: at the seam cell
    MEEP's two entries are not the one deposit of their sum."""
    grid, layout = _seam_grid()
    line = _line(grid, 1.0)
    src.bind_src_vols([line], layout, grid)
    pieces = src._src_vol_pieces(line._src_geometry, layout.boxes, grid)  # noqa: SLF001
    seam = (set(pieces[0][1]) & set(pieces[1][1])).pop()
    rng = np.random.default_rng(311)
    differing = 0
    for step in range(200):
        time = 1.0 + step * grid.dt
        fields = Fields(grid=grid)
        fields.Dz[...] = rng.standard_normal(fields.Dz.shape).astype(np.float32)
        before = float(fields.Dz[seam])
        line.inject(fields, time)
        current = line.current(time, grid.dt)
        word = np.float32(before)
        for _chunk, cells, amps in pieces:
            a = amps[cells.index(seam)]
            word = np.float32(float(word) - ((a.real * current.real) - (a.imag * current.imag)) * grid.dt)
        assert _word(fields.Dz[seam]) == _word(word)
        total = dict(zip(line._indices, line._amplitudes))[seam]  # noqa: SLF001
        once = np.float32(before - ((total.real * current.real) - (total.imag * current.imag)) * grid.dt)
        differing += int(_word(once) != _word(word))
    assert differing > 0, "two roundings never differed from one; the case cannot see the seam"


def test_an_axis_the_dimension_does_not_carry_is_one_point_sampled_at_zero():
    """vec2py hands the amplitude function 0 on z in 2-D, and z is one point of weight 1
    (src_vol_chunkloop's vec of the run's dimension): a half-integer-z component is not
    split into two z rungs sampled at its own +-dx/2."""
    grid, _layout = _seam_grid()
    seen = []

    def amp_func(x, y, z):
        seen.append(z)
        return complex(1.0, 0.25 * z)

    from .sources import GaussianEnvelope  # noqa: PLC0415

    source = VolumeSource(grid=grid, component="Ez", center=(0.25, 0.0, 0.0), size=(0.0, 0.5, 0.0),
                          amp_func=amp_func, envelope=GaussianEnvelope(frequency=1.0, fwidth=0.7))
    assert seen and all(z == 0.0 and math.copysign(1.0, z) > 0 for z in seen)
    # One z point per (x, y) site, sampled once: as many samples as cells.
    assert len(seen) == len(source._indices)  # noqa: SLF001
    assert all(cell[2] == 0 for cell in source._indices)  # noqa: SLF001
    assert all(complex(a).imag == 0.0 for a in source._amplitudes)  # noqa: SLF001


def _lattice_seam_grid(name):
    """An unfolded periodic y, or a periodic y folded by a mirror at each parity and stencil."""
    from .chunk_map import ChunkLayout  # noqa: PLC0415
    from .grid import Mirror  # noqa: PLC0415

    folded = {"even": 0.32, "odd": 0.30}
    if name == "unfolded":
        grid, span = Grid(resolution=10.0, cell_size=(2.0, 3.0, 0.0), dimensions=2), 3.0
    else:
        parity, plane = name.split()
        symmetry = ('Y',) if plane == "Ez" else (Mirror('Y', -1),)
        grid = Grid(resolution=50.0, cell_size=(6.0, folded[parity], 0.0), dimensions=2,
                    symmetry=symmetry)
        span = folded[parity]
    return grid, ChunkLayout.one_rank(grid, None), span


SEAMS = ("unfolded", "even Ez", "even Ey", "odd Ez", "odd Ey")


@pytest.mark.parametrize("seam", SEAMS)
def test_a_full_width_source_takes_its_lattice_image_as_a_separate_subtraction(seam):
    """MEEP's seam cell is two ``src_vol`` entries, subtracted one after the other.

    A request spanning a periodic axis reaches the seam row through the direct pass
    (its edge weight) and through the ishift = -1 image of its other end: two calls of
    ``src_vol_chunkloop`` (loop_in_chunks.cpp:390-531, sources.cpp:243-312) whose index
    vectors differ, so ``fields_chunk::add_source`` keeps two entries
    (fields.cpp:576-585, sources.cpp:179-181) and ``step_source`` rounds twice there
    (step.cpp:295-317). Pinned on an unfolded periodic y and on a mirror-folded one at
    both parities and both stencils: the pieces sum to the footprint, the image comes
    first (``loop_in_chunks`` starts ``ishift`` at its minimum), and the stored words of
    the bound source are the two-subtraction transcription; the summed amplitude
    deposited once is the must-catch control.
    """
    grid, layout, span = _lattice_seam_grid(seam)
    component = "Ez" if seam in ("unfolded", "even Ez", "odd Ez") else "Ey"
    source = ExtendedSource(grid=grid, frequency=0.8, component=component,
                            center=(-0.45 if seam == "unfolded" else -1.8, 0.0, 0.0),
                            size=(0.0, span, 0.0), amp_func=lambda x, y, z: complex(1.0 + y, 0.3 - y))
    pieces = src._src_vol_pieces(source._src_geometry, layout.boxes, grid)  # noqa: SLF001
    sequences: dict = {}
    for _chunk, cells, amps in pieces:
        for cell, amp in zip(cells, amps):
            sequences.setdefault(tuple(cell), []).append(complex(amp))
    seams = {cell: amps for cell, amps in sequences.items() if len(amps) == 2}
    assert seams and max(len(amps) for amps in sequences.values()) == 2, sequences
    assert len(pieces[0][1]) < len(pieces[-1][1]), "the image piece is listed first"
    totals = {tuple(p): complex(a) for p, a in zip(source._indices, source._amplitudes)}  # noqa: SLF001
    assert set(totals) == set(sequences)
    for cell, amps in sequences.items():
        assert complex(sum(amps)) == pytest.approx(totals[cell], rel=1e-15, abs=0.0)
    src.bind_src_vols([source], layout, grid)
    fields = Fields(grid=grid)
    array = getattr(fields, src._ARRAY_FOR_COMPONENT[component])  # noqa: SLF001
    rng = np.random.default_rng(4242)
    array[...] = rng.standard_normal(array.shape).astype(np.float32) * np.float32(1e-3)
    summed = 0
    for step in range(30):
        time = 1.0 + step * grid.dt
        before = array.copy()
        source.inject(fields, time)
        current = complex(source.current(time, grid.dt))
        for cell, amps in sequences.items():
            word = before[cell]
            for a in amps:
                re = ((a.real * current.real) - (a.imag * current.imag)) * grid.dt
                word = np.uint32(_reference_difference(word, re)).view(np.float32)
            assert _word(array[cell]) == _word(word), (seam, cell, step)
        for cell, amps in seams.items():
            a = totals[cell]
            re = ((a.real * current.real) - (a.imag * current.imag)) * grid.dt
            summed += int(_word(array[cell]) != _reference_difference(before[cell], re))
    assert summed > 0, "depositing the seam's summed amplitude once moved no word"


# ---------------------------------------------------------------------------
# The lift's MEEP src_time (owner ruling R3)
# ---------------------------------------------------------------------------

class _MeepTime:
    """A stand-in for MEEP's ``src_time``: values no package evaluator produces."""

    is_integrated = False

    def __init__(self):
        self.calls = []

    def dipole(self, time):
        self.calls.append(("dipole", time))
        return complex(3.0 + time, -0.5)

    def current(self, time, dt):
        self.calls.append(("current", time, dt))
        return complex(7.0 * time, dt)


def test_a_bound_meep_src_time_forms_every_current_and_dipole():
    grid, fields = _grid()
    source = GaussianPulsedSource(grid=grid, frequency=0.8, fwidth=0.2, component="Ez",
                                  center=(0.0, 0.0, 0.0), size=(0.2, 0.0, 0.0))
    meep_time = _MeepTime()
    source.bind_meep_src_time(meep_time)
    assert source.dipole(1.5) == complex(4.5, -0.5)
    assert source.current(1.5, grid.dt) == complex(10.5, grid.dt)
    before = fields.Dz.copy()
    source.inject(fields, 1.5)
    amps = np.asarray(source._point_amps)  # noqa: SLF001
    for j, cell in enumerate(zip(source._point_ix, source._point_iy, source._point_iz)):  # noqa: SLF001
        a = complex(amps[j])
        re = ((a.real * 10.5) - (a.imag * grid.dt)) * grid.dt
        assert _word(fields.Dz[cell]) == _reference_difference(before[cell], re)
    assert ("current", 1.5, grid.dt) in meep_time.calls


def test_a_volume_source_binds_its_envelope_and_the_integrated_flag_must_agree():
    grid, _fields = _grid()
    integrated = VolumeSource(grid=grid, component="Hz", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                              envelope=ContinuousEnvelope(frequency=0.8, is_integrated=True))
    with pytest.raises(ValueError, match="is_integrated"):
        integrated.bind_meep_src_time(_MeepTime())
    meep_time = _MeepTime()
    meep_time.is_integrated = True
    integrated.bind_meep_src_time(meep_time)
    assert integrated.envelope._meep_src_time is meep_time  # noqa: SLF001
    assert integrated.dipole(2.0) == complex(5.0, -0.5)
    with pytest.raises(ValueError, match="src_time"):
        integrated.bind_meep_src_time(object())


def test_the_continuous_point_source_folds_its_amplitude_into_the_weights_as_meep_does():
    """``IVEC_LOOP_WEIGHT * (amplitude * a) * A`` (sources.cpp:243-312): the current is the unit
    waveform's, and the point amplitudes carry the user amplitude."""
    grid, _fields = _grid()
    unit = ContinuousSource(grid=grid, frequency=0.8, component="Ez", center=(0.03, 0.02, 0.0))
    scaled = ContinuousSource(grid=grid, frequency=0.8, component="Ez", center=(0.03, 0.02, 0.0),
                              amplitude=1.2 - 0.4j)
    assert scaled.dipole(1.1) == unit.dipole(1.1)
    for w_unit, w_scaled in zip(unit._point_weights, scaled._point_weights):  # noqa: SLF001
        assert w_scaled != w_unit
        assert abs(w_scaled - (1.2 - 0.4j) * w_unit) <= 1e-12 * abs(w_scaled)


# ---------------------------------------------------------------------------
# The silent skip set against step.cpp's own arithmetic, the door's refusal, real Fields
# with a conductivity of either sign, and the CUDA device
# ---------------------------------------------------------------------------

#: ``condinv`` classes for the skip-set oracle: lossless, positive, +0, negative, +inf, NaN.
_ORACLE_SCALES = (None, 0.4, 0.0, -2.0, math.inf, math.nan)


def _subtrahends_are_plus_zero(amplitude: complex, dt, scale, complex_storage: bool) -> bool:
    """step.cpp:302-314 at the current ``(+0, +0)``, in Python doubles, one operation at a
    time and in MEEP's order: is every part the deposit would subtract ``+0``?"""
    re = (amplitude.real * 0.0) - (amplitude.imag * 0.0)
    im = (amplitude.real * 0.0) + (amplitude.imag * 0.0)
    parts = [re, im] if complex_storage else [re]
    if dt is not None:
        parts = [part * dt for part in parts]
    if scale is not None:
        parts = [part * float(np.float32(scale)) for part in parts]
    return all(part == 0.0 and math.copysign(1.0, part) > 0.0 for part in parts)


@pytest.mark.parametrize("storage", ("real", "complex"))
def test_the_skip_set_is_where_step_cpps_own_arithmetic_subtracts_plus_zero(storage):
    """The deposits ``_deposit`` skips, against an oracle independent of its predicate.

    At the current ``(+0, +0)`` a table is skipped iff MEEP's own statements, in Python
    doubles, subtract ``+0`` at every point and its ``condinv`` is finite and sign-clear (a
    negative ``condinv`` that turns a flagged ``-0`` into ``+0`` still deposits: exact
    either way, and the rule stays one rule); at every other current -- the three other
    signed zeros, a tail of 1e-100, an ordinary one -- never.
    """
    complex_storage = storage == "complex"
    dtype = np.complex64 if complex_storage else np.float32
    ix, iy, iz = np.arange(2, dtype=np.int64), np.zeros(2, np.int64), np.zeros(2, np.int64)
    door = host_writes.subtract_double
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return door(*args, **kwargs)

    skipped = deposited = 0
    with pytest.MonkeyPatch.context() as patch, np.errstate(all="ignore"):
        patch.setattr(host_writes, "subtract_double", counting)
        for re_part in _PARTS:
            for im_part in _PARTS:
                amplitude = complex(re_part, im_part)
                amps = np.full(2, amplitude, dtype=np.complex128)
                for scale in _ORACLE_SCALES:
                    cndinv = None if scale is None else np.full((2, 1, 1), scale, dtype=np.float32)
                    usable = scale is None or (math.isfinite(scale) and math.copysign(1.0, scale) > 0)
                    for dt in (0.05, None):
                        expected = usable and _subtrahends_are_plus_zero(amplitude, dt, scale,
                                                                         complex_storage)
                        for current in _CURRENTS:
                            calls.clear()
                            src._deposit(np.zeros((2, 1, 1), dtype=dtype), ix, iy, iz, amps,  # noqa: SLF001
                                         current, dt, "Ez", cndinv)
                            skip = not calls
                            want = expected and _bitwise_plus_zero_current(current)
                            assert skip == want, (storage, amplitude, scale, dt, current)
                            skipped += skip
                            deposited += not skip
    assert skipped > 0 and deposited > 0, (skipped, deposited)


def test_a_silent_deposit_into_a_non_contiguous_array_is_still_refused_by_name():
    """The skip never swallows the door's refusal: an unflagged silent table aimed at an
    array that is not C-contiguous is refused exactly as the always-deposit form refuses it."""
    array = np.zeros((4, 4, 1), dtype=np.float32, order="F")
    ix, iy, iz = np.arange(2, dtype=np.int64), np.ones(2, np.int64), np.zeros(2, np.int64)
    amps = np.full(2, 1.0 + 0.0j, dtype=np.complex128)
    assert src._silent_deposit_is_identity(amps, ix, iy, iz, 0.05, None, True)  # noqa: SLF001
    for deposit in (_always_deposit, src._deposit):  # noqa: SLF001
        with pytest.raises(ValueError, match="not C-contiguous"):
            deposit(array, ix, iy, iz, amps, 0j, 0.05, "Ez")


_MIN_SUB = np.uint32(0x00000001).view(np.float32)

#: The words planted at the deposit cells, cycled: both zeros, both smallest subnormals,
#: a large subnormal and normals of both signs.
_PLANTS = (np.float32(-0.0), np.float32(0.0), -_MIN_SUB, _MIN_SUB,
           np.uint32(0x807FFFFF).view(np.float32), np.float32(1.5), np.float32(-2.25))
_NORMAL_PLANTS = tuple(w for w in _PLANTS if not (w != 0 and abs(float(w)) < 1.1754944e-38))

#: Amplitude tables, and whether a silent deposit of each is skipped (real, complex storage):
#: a part is -0 where a_re has the sign bit and a_im not (real), or both have it (imaginary).
FIELD_TABLES = {
    "positive": ([1.0, 0.5, 2.5], (True, True)),
    "negative real": ([1.0, -1.0], (False, False)),
    "minus i": ([complex(-0.0, -1.0)], (True, False)),
    "zero amplitudes": ([complex(0.0, -0.0), complex(0.0, 0.0), complex(-0.0, -0.0)], (True, False)),
    "a -0 real amplitude": ([complex(-0.0, 0.0)], (False, False)),
    "phases": ([0.3 - 0.7j, 2.0 + 1.0j, complex(0.0, 1.0)], (True, True)),
    "a negative phase": ([0.3 - 0.7j, -0.3 + 0.7j], (False, False)),
    "non-finite": ([complex(math.inf, 0.0)], (False, False)),
}

#: Currents at the deposit: the silent one, the three other signed zeros, and two tiny
#: non-zero ones (a threshold below 1e-30 would treat them as silent).
FIELD_CURRENTS = {
    "(+0, +0)": complex(0.0, 0.0), "(-0, +0)": complex(-0.0, 0.0), "(+0, -0)": complex(0.0, -0.0),
    "(-0, -0)": complex(-0.0, -0.0), "(1e-35, 0)": complex(1e-35, 0.0), "(0, 1e-35)": complex(0.0, 1e-35),
}

FIELD_CONFIGURATIONS = dict(CONFIGURATIONS, **{
    "real, negative D conductivity": dict(negative_conductivity=-50.0),
    "complex, negative D conductivity": dict(complex_fields=True, negative_conductivity=-50.0),
    "real, infinite condinv": dict(negative_conductivity=-40.0),
})


def _field_grid(complex_fields=False, magnetic_conductivity=False, electric_conductivity=False,
                negative_conductivity=None):
    grid, fields = _grid(complex_fields, magnetic_conductivity, electric_conductivity)
    if negative_conductivity is not None:  # condinv < 0 (or inf): (+0) * condinv is not +0
        with np.errstate(divide="ignore"):
            fields.set_d_conductivity(np.full(grid.shape, negative_conductivity, dtype=np.float32))
    return grid, fields


def _field_case(fields, component, amplitudes, plants):
    """The deposit cells of ``component``'s array, planted; their index arrays; the amplitudes."""
    name = src._ARRAY_FOR_COMPONENT[component]  # noqa: SLF001
    array = getattr(fields, name)
    n = len(amplitudes) * len(plants)
    cells = [(2 + k % 8, 2 + k // 8, 0) for k in range(n)]
    ix, iy, iz = (np.array([c[a] for c in cells], dtype=np.int64) for a in range(3))
    for k, cell in enumerate(cells):
        word = plants[k % len(plants)]
        if array.dtype == np.complex64:  # word by word: a complex() round trip could move a sign
            parts = array.view(np.float32).reshape(array.shape + (2,))
            parts[cell + (0,)] = word
            parts[cell + (1,)] = plants[(k + 3) % len(plants)]
        else:
            array[cell] = word
    amps = np.array([amplitudes[k % len(amplitudes)] for k in range(n)], dtype=np.complex128)
    return name, ix, iy, iz, amps


def _field_deposit_words(deposit, configuration, component, amplitudes, factor, plants):
    """Words of ``component``'s array after one ``deposit``, and the door's call count."""
    grid, fields = _field_grid(**FIELD_CONFIGURATIONS[configuration])
    name, ix, iy, iz, amps = _field_case(fields, component, amplitudes, plants)
    array = getattr(fields, name)
    calls = []
    door = host_writes.subtract_double

    def counting(*args, **kwargs):
        calls.append(1)
        return door(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch, np.errstate(all="ignore"):
        patch.setattr(host_writes, "subtract_double", counting)
        deposit(array, ix, iy, iz, amps, factor, grid.dt, component, fields.condinv_for(name))
    return _words(array).copy(), len(calls), array.dtype == np.complex64


def _field_expects_skip(configuration, component, table, label, is_complex) -> bool:
    skips = FIELD_TABLES[table][1]
    conductive_sign = "negative" not in configuration and "infinite" not in configuration
    return (label == "(+0, +0)" and skips[1 if is_complex else 0]
            and (conductive_sign or component == "Hz"))


@pytest.mark.parametrize("configuration", sorted(FIELD_CONFIGURATIONS))
def test_a_silent_deposit_on_real_fields_stores_the_always_deposit_words_and_skips_only_plus_zero(configuration):
    """On a ``Fields`` with a D conductivity of either sign (and B), electric and magnetic:
    every case's words equal the always-deposit form's, sign of zero included, and the door
    is skipped exactly where every deposited part is ``+0``."""
    skipped = deposited = 0
    for component in ("Ez", "Hz"):
        for table, (amplitudes, _skips) in FIELD_TABLES.items():
            for label, current in FIELD_CURRENTS.items():
                want, _, is_complex = _field_deposit_words(_always_deposit, configuration, component,
                                                           amplitudes, current, _PLANTS)
                got, calls, _ = _field_deposit_words(src._deposit, configuration, component,  # noqa: SLF001
                                                     amplitudes, current, _PLANTS)
                where = f"{configuration}, {component}, {table}, current {label}"
                assert np.count_nonzero(want != got) == 0, f"{where}: words differ"
                expect_skip = _field_expects_skip(configuration, component, table, label, is_complex)
                assert calls == (0 if expect_skip else 1), f"{where}: door calls {calls}"
                skipped += expect_skip
                deposited += not expect_skip
    assert skipped > 0 and deposited > 0


# The CUDA table's deposit: the same statements on CuPy arrays, the door a device gather,
# widen, subtract, round and scatter.

def _device_blanket_skip(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    if complex(factor) == 0:
        return None
    return _always_deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv)


def _device_blanket_canonicalise(array, ix, iy, iz, amps, factor, dt, component, cndinv=None):
    """A silent deposit spelled as ``-0 -> +0`` at the cells, through the gather and scatter doors."""
    if complex(factor) != 0:
        return _always_deposit(array, ix, iy, iz, amps, factor, dt, component, cndinv)
    linear = src._linear_points(array, ix, iy, iz)  # noqa: SLF001
    words = host_writes.gather(array, linear)
    parts = words.view(np.float32)
    parts[parts == 0] = np.float32(0.0)
    host_writes.scatter(array, linear, words)
    return None


DEVICE_MUTANTS = {
    "blanket skip of a zero current": _device_blanket_skip,
    "blanket -0 -> +0 on a zero current": _device_blanket_canonicalise,
}


def _device_deposit_words(cupy, deposit, configuration, component, amplitudes, factor, plants):
    """:func:`_field_deposit_words` with the array, the index arrays, the amplitudes and
    ``condinv`` on the device; the words are read back after the one deposit."""
    grid, fields = _field_grid(**FIELD_CONFIGURATIONS[configuration])
    name, ix, iy, iz, amps = _field_case(fields, component, amplitudes, plants)
    array = cupy.asarray(getattr(fields, name))
    cndinv = fields.condinv_for(name)
    cndinv = None if cndinv is None else cupy.asarray(cndinv)
    ix, iy, iz, amps = (cupy.asarray(v) for v in (ix, iy, iz, amps))
    calls = []
    door = host_writes.subtract_double

    def counting(*args, **kwargs):
        calls.append(1)
        return door(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch, np.errstate(all="ignore"):
        patch.setattr(host_writes, "subtract_double", counting)
        deposit(array, ix, iy, iz, amps, factor, grid.dt, component, cndinv)
    return _words(cupy.asnumpy(array)).copy(), len(calls), array.dtype == cupy.complex64


@pytest.mark.requires_resource("cupy")
def test_on_a_cuda_device_the_silent_skip_stores_the_always_deposit_and_host_words():
    """On the zeros and normals a run stores, whatever subnormal policy the device kernels
    carry: the skip's words equal the always-deposit form's on the device and the host
    deposit's, sign of zero included, and the door is skipped exactly where every part is
    ``+0`` (the verdicts are read once, from host copies of the device tables). The planted
    blanket skip and blanket ``-0 -> +0`` are caught on the device."""
    from .backends import cupy_available  # noqa: PLC0415

    if not cupy_available():
        from conftest import requires_resource_skip  # noqa: PLC0415 - the root conftest

        requires_resource_skip("cupy", "CuPy is not installed or no CUDA device is visible")
    import cupy  # noqa: PLC0415

    skipped = deposited = 0
    caught = dict.fromkeys(DEVICE_MUTANTS, 0)
    for configuration in sorted(FIELD_CONFIGURATIONS):
        for component in ("Ez", "Hz"):
            for table, (amplitudes, _skips) in FIELD_TABLES.items():
                for label, current in FIELD_CURRENTS.items():
                    want, _, is_complex = _device_deposit_words(cupy, _always_deposit, configuration,
                                                                component, amplitudes, current,
                                                                _NORMAL_PLANTS)
                    got, calls, _ = _device_deposit_words(cupy, src._deposit, configuration,  # noqa: SLF001
                                                          component, amplitudes, current,
                                                          _NORMAL_PLANTS)
                    host, _, _ = _field_deposit_words(src._deposit, configuration, component,  # noqa: SLF001
                                                      amplitudes, current, _NORMAL_PLANTS)
                    where = f"{configuration}, {component}, {table}, current {label}"
                    assert np.count_nonzero(want != got) == 0, f"{where}: device words differ"
                    assert np.count_nonzero(host != got) == 0, f"{where}: device and host words differ"
                    expect_skip = _field_expects_skip(configuration, component, table, label, is_complex)
                    assert calls == (0 if expect_skip else 1), f"{where}: door calls {calls}"
                    skipped += expect_skip
                    deposited += not expect_skip
                    for mutant, deposit in DEVICE_MUTANTS.items():
                        planted, _, _ = _device_deposit_words(cupy, deposit, configuration,
                                                              component, amplitudes, current,
                                                              _NORMAL_PLANTS)
                        caught[mutant] += int(np.count_nonzero(planted != want))
    assert skipped > 0 and deposited > 0
    assert all(caught.values()), f"a planted skip rule left every device word as the deposit's: {caught}"
