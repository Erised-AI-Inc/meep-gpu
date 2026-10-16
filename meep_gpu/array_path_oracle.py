"""A behavioural fingerprint of the array path, recorded in every kernel ledger.

WHAT THIS IS. Every kernel table is certified against the array path: a gate passes when
the kernel returns the same float32 words as ``stepping`` on the same state. The kernel
ledgers bind the bytes of the KERNEL sources, and only a handful of their entries bind an
array-path module at all, so an arithmetic change to the array path leaves almost every
ledger reading green while every gate run against the new array path would fail. This
module closes that gap with one judged value, ``fingerprint_sha256``: the SHA-256 over the
float32 words a fixed, seeded fixture set produces when the array path steps it.

* An ARITHMETIC change in any function a kernel transcribes moves the fingerprint, and the
  three weld contracts then fail by name: ``test_<table>_weld_contract.py::
  test_the_<table>_ledger_records_the_live_array_path_oracle``.
* A comment, docstring or layout edit leaves it unchanged, because the fingerprint is a
  property of what the code computes, not of its bytes.
* ``parity/meep_gpu/record_array_path_oracle.py`` is the only writer. It refuses to mint
  unless every declared (case, sub-step, component, form) class carries a nonzero cell
  count (:data:`DECLARED_FORM_CLASSES`), so a fixture that stops reaching a form refuses by
  name instead of shrinking the tripwire.

WHAT THE FIXTURE AVOIDS, so that the fingerprint is a property of the array path and not
of the host that computed it:

* No transcendental function while the array path steps. :func:`run` replaces every
  libm-backed function of ``numpy``, ``math`` and ``cmath`` with a refusal for the
  stepping phase, and the build phase keeps libm out of the state: the PML grading
  prefactor is the stored constant :data:`PML_LN_REFLECTION` instead of
  ``-numpy.log(R)`` (:func:`_libm_free_pml_grading`), the susceptibility frequencies are
  chosen so that ``(2*pi*f*dt) ** 2`` has an exactly representable result
  (:data:`LORENTZ_ANGULAR`), and the dt is a power of two. The two certification hosts'
  libm already disagree on cosine at 30 of 800 and sine at 18 of 800 sampled arguments
  (measured 2026-10-05, Apple arm64 against glibc x86-64).
* No source envelope evaluation. Every source is a ``custom`` source whose waveform reads
  STORED current words (:class:`_StoredCurrents`), so the deposit is exercised and
  ``src_time`` is never evaluated.
* No complex storage. The array path forms every product on complex storage part by part
  (``part_products``), so its words no longer depend on whether a host contracts a complex
  multiply; complex fixtures are a new fixture-set identity, not yet minted, and a Bloch
  phase still evaluates libm. Those classes are listed in :data:`DEFERRED_FORM_CLASSES`
  with the reason. The deposit forms
  ``real(amp * current)`` per part in float64 ufuncs (``sources._deposit``), which no host
  contracts; the fixture's amplitudes are still real (:func:`_require_real_amplitudes`),
  which keeps its deposit words the ones a real-amplitude fixture set was minted with.
* No random-number algorithm that could change between releases: the words come from
  SplitMix64 written here in integer arithmetic (:func:`_splitmix64`).
* No subnormal flush. :func:`run` measures the host FPU, drives it to keep when it is
  flushing, refuses by name when that does not hold, and restores the entry state.

THE READER IS EXCLUDED (``get_array``, ``get_field``, the point reader): no kernel
transcribes it. DFT and flux monitors are excluded for the same reason.

Provenance is reported and never judged: :func:`provenance_code_sha256` records the
``code_identity`` digest of each array-path module beside the fingerprint, so a reader
can see which code it was minted from without a refactor that keeps behaviour turning the
tripwire red.

NumPy only, and every package import beyond NumPy is deferred, so the weld contracts can
call this at the laptop merge bar.
"""

from __future__ import annotations

import cmath
import contextlib
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, Dict, Iterator, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple)

import numpy

__all__ = [
    "FIXTURE_SET", "LEAF_KEY", "STEPS", "CASES", "DECLARED_FORM_CLASSES",
    "DEFERRED_FORM_CLASSES", "PROVENANCE_MODULES", "OracleRefusal", "OracleRun", "run",
    "live", "coverage_problems", "provenance_code_sha256", "entry_leaf", "dispatch_leaf",
    "binds_bytes", "ledger_problems", "membership_line",
]

#: The fixture-set identity, recorded beside the fingerprint. A change to the fixture
#: set (a new case, a new seed rule, complex fixtures once complex products are
#: removed from the array path) is a NEW identity, so it reads as a change of what is
#: measured rather than as a change of the array path.
FIXTURE_SET = "array_path_oracle/real-v6"

#: The ledger key of the leaf, in every byte-binding entry and in ``driver_dispatch``.
LEAF_KEY = "array_path_oracle"

#: The minting tool and this module, as the leaf names them.
TOOL = "parity/meep_gpu/record_array_path_oracle.py"
ORACLE_MODULE = "meep_gpu/array_path_oracle.py"

#: Complete driver steps per case. Every case is digested after seeding and after each.
STEPS = 6

#: dt = COURANT / RESOLUTION = 2**-4 and dx = 2**-3 exactly, so dt/dx, dt/2 and every
#: product with dt are exact scalings.
RESOLUTION = 8.0
COURANT = 0.5

#: The PML grading prefactor's numerator, stored: it stands in for ``-ln(1e-15)``
#: (34.5387...), which ``PML.prefac`` computes with ``numpy.log``.
PML_LN_REFLECTION = 34.5

#: Angular frequencies (2*pi*f) of the two susceptibilities, each a float32-exact value
#: whose round trip ``2*pi*(w / (2*pi)) == w`` holds in IEEE double arithmetic. The
#: polarization constants (``Susceptibility.realnum_constants``) are formed with IEEE
#: basic operations only, so they are the same words on every host; dt = 2**-4 keeps
#: every product with dt an exact scaling.
LORENTZ_ANGULAR = 3.0
DRUDE_ANGULAR = 2.0

#: The array-path modules whose code identity is reported beside the fingerprint.
#: ``meep_gpu/chunk_map.py`` does not exist before the chunk-map work package lands; a
#: module that is absent is listed under ``provenance_absent`` rather than refused.
PROVENANCE_MODULES: Tuple[str, ...] = (
    "meep_gpu/stepping.py", "meep_gpu/chunk_map.py", "meep_gpu/sources.py",
    "meep_gpu/pml.py", "meep_gpu/fields.py", "meep_gpu/host_writes.py",
    "meep_gpu/deposit_repair.py", "meep_gpu/driver.py", "meep_gpu/dispersion.py",
    "meep_gpu/withdraw_hoist.py", "meep_gpu/grid.py", "meep_gpu/from_meep.py",
    "meep_gpu/part_products.py",
)

#: Form classes the real fixture set cannot carry, and why. Recorded in the
#: ``driver_dispatch`` leaf so the absence is a statement rather than an omission.
DEFERRED_FORM_CLASSES: Dict[str, str] = {
    "complex_storage/*": (
        "complex64 storage: the array path forms its products part by part (part_products), "
        "so no host contraction reaches it; the complex fixtures are added with a new "
        "fixture-set identity, a change of what is measured"),
    "bloch_phase/*": (
        "a nonzero k_point needs complex storage and evaluates exp(i*2*pi*k*L) with libm"),
    "cylindrical_m_nonzero/*": (
        "cylindrical m != 0 couples the real and imaginary parts through i*m/r and needs "
        "complex storage"),
    "source_envelopes/*": (
        "continuous and gaussian envelopes evaluate exp, sin and cos; the deposit is "
        "exercised through stored current words instead"),
}

#: The six components of each family, in stepping's own order.
_B = ("Bx", "By", "Bz")
_D = ("Dx", "Dy", "Dz")
_H = ("Hx", "Hy", "Hz")
_E = ("Ex", "Ey", "Ez")
_AXES = ("x", "y", "z")

#: The functions whose result can differ between two IEEE hosts because a libm computes
#: them. Square root and the four arithmetic operations are correctly rounded everywhere
#: and stay callable.
_LIBM_NUMPY = ("exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sin", "cos",
               "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh",
               "arcsinh", "arccosh", "arctanh", "power", "float_power", "hypot", "cbrt",
               "logaddexp", "logaddexp2")
_LIBM_MATH = ("exp", "expm1", "log", "log2", "log10", "log1p", "sin", "cos", "tan",
              "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh",
              "atanh", "pow", "hypot", "cbrt", "erf", "erfc", "gamma", "lgamma")
_LIBM_CMATH = ("exp", "log", "log10", "sqrt", "sin", "cos", "tan", "asin", "acos",
               "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "phase",
               "polar", "rect")

_MASK64 = (1 << 64) - 1


class OracleRefusal(RuntimeError):
    """A named reason the fixture set did not run, or ran outside its preconditions."""


class OracleRun(NamedTuple):
    """One run of the fixture set."""

    fixture_set: str
    fingerprint_sha256: str
    case_sha256: Dict[str, str]
    form_classes: Dict[str, int]
    host: Dict[str, Any]
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Words: SplitMix64 in integer arithmetic, so no generator algorithm can move them
# ---------------------------------------------------------------------------


def _seed(*parts: str) -> int:
    text = "/".join((FIXTURE_SET,) + tuple(parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def _splitmix64(seed: int, count: int) -> numpy.ndarray:
    """``count`` SplitMix64 outputs from ``seed``, as uint64. Integer operations only."""
    index = numpy.arange(1, count + 1, dtype=numpy.uint64)
    with numpy.errstate(over="ignore"):
        z = numpy.uint64(seed & _MASK64) + index * numpy.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> numpy.uint64(30))) * numpy.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> numpy.uint64(27))) * numpy.uint64(0x94D049BB133111EB)
        z = z ^ (z >> numpy.uint64(31))
    return z


def _unit_float32(seed: int, shape: Tuple[int, ...]) -> numpy.ndarray:
    """float32 in [0, 1) with a 23-bit mantissa: an exact integer times 2**-23."""
    count = int(numpy.prod(shape))
    mantissa = (_splitmix64(seed, count) >> numpy.uint64(41)).astype(numpy.float32)
    return (mantissa * numpy.float32(2.0 ** -23)).reshape(shape)


def _state_words(seed: int, shape: Tuple[int, ...]) -> numpy.ndarray:
    """A field state as float32 bit patterns.

    Normal values of either sign between 2**-6 and 2**2, with about 2 % exact +0, 2 %
    exact -0 and 1 % subnormal words planted, so the signed-zero rules and the subnormal
    mode both reach the fingerprint.
    """
    count = int(numpy.prod(shape))
    words = _splitmix64(seed, count)
    mantissa = words & numpy.uint64(0x7FFFFF)
    exponent = numpy.uint64(121) + ((words >> numpy.uint64(23)) & numpy.uint64(7))
    sign = (words >> numpy.uint64(31)) & numpy.uint64(1)
    bits = (sign << numpy.uint64(31)) | (exponent << numpy.uint64(23)) | mantissa
    selector = (words >> numpy.uint64(40)) & numpy.uint64(0xFF)
    bits = numpy.where(selector < 5, numpy.uint64(0), bits)
    bits = numpy.where((selector >= 5) & (selector < 10), numpy.uint64(0x80000000), bits)
    subnormal = (sign << numpy.uint64(31)) | (mantissa | numpy.uint64(1))
    bits = numpy.where((selector >= 10) & (selector < 13), subnormal, bits)
    return bits.astype(numpy.uint32).view(numpy.float32).reshape(shape)


def _volume(seed: int, shape: Tuple[int, ...], low: float, high: float,
            zero_fraction: float = 0.0) -> numpy.ndarray:
    """A material volume in [low, high) as float32, with a fraction of exact zeros."""
    unit = _unit_float32(seed, shape)
    values = (numpy.float32(low) + numpy.float32(high - low) * unit).astype(numpy.float32)
    if zero_fraction > 0.0:
        selector = _unit_float32(seed ^ 0x5A5A5A5A, shape)
        values = numpy.where(selector < numpy.float32(zero_fraction),
                             numpy.float32(0.0), values).astype(numpy.float32)
    return values


class _StoredCurrents:
    """A custom waveform that returns STORED complex words, one per step.

    The driver asks for the current at its slot times, ``n * dt`` (magnetic) and
    ``(n + 1/2) * dt`` (electric, and the integrated dipole). ``floor(t/dt + 1/4)`` maps
    both to ``n`` with a margin of a quarter step, so the table is read by step, never
    evaluated. Real and imaginary parts are multiples of ``2**-20`` in [-1, 1) with
    magnitude at least ``2**-4``, so ``dt * current`` is an exact scaling and no part is
    zero.
    """

    def __init__(self, seed: int, dt: float, count: int = STEPS + 4):
        words = _splitmix64(seed, 2 * count)
        parts = ((words >> numpy.uint64(44)).astype(numpy.int64) - (1 << 19)).astype(float)
        parts = parts * 2.0 ** -19
        parts = numpy.where(numpy.abs(parts) < 2.0 ** -4,
                            numpy.copysign(0.5, parts + 2.0 ** -30), parts)
        self.table = [complex(float(parts[2 * i]), float(parts[2 * i + 1]))
                      for i in range(count)]
        self.dt = float(dt)
        self.reads = 0

    def __call__(self, time_value: float) -> complex:
        step = int(math.floor(float(time_value) / self.dt + 0.25))
        if not 0 <= step < len(self.table):
            raise OracleRefusal(
                f"the driver asked a stored-current source for step {step} (t={time_value!r}), "
                f"outside the {len(self.table)} stored words")
        self.reads += 1
        return self.table[step]


# ---------------------------------------------------------------------------
# What the fixture keeps out of the state
# ---------------------------------------------------------------------------


def _libm_free_prefac(self: Any, cells: Optional[float] = None) -> float:
    """``PML.prefac`` with the stored constant :data:`PML_LN_REFLECTION` for ``-ln(R)``."""
    n_pml = self.thickness if cells is None else cells
    dx_pml = n_pml * self.grid.dx
    if dx_pml <= 0:
        return 0.0
    profile_integral = 1.0 / (self.order + 1)
    return float(PML_LN_REFLECTION / (4.0 * dx_pml * profile_integral))


@contextlib.contextmanager
def _libm_free_pml_grading() -> Iterator[None]:
    """Build every PML of the fixture with :func:`_libm_free_prefac`.

    The grading itself is ``scale * u ** order`` with order 2, which NumPy evaluates as a
    square; the prefactor is the only libm value a PML holds.
    """
    from .pml import PML  # noqa: PLC0415

    original = PML.prefac
    PML.prefac = _libm_free_prefac
    try:
        yield
    finally:
        PML.prefac = original


class _LibmCall(RuntimeError):
    pass


@contextlib.contextmanager
def _no_libm_calls(case: str) -> Iterator[None]:
    """Refuse every libm-backed function of numpy, math and cmath while the array path steps."""
    replaced: List[Tuple[Any, str, Any]] = []

    def refuse(module_name: str, name: str) -> Callable[..., Any]:
        def call(*_args: Any, **_kwargs: Any) -> Any:
            raise _LibmCall(f"{module_name}.{name}")
        return call

    for module, names, module_name in ((numpy, _LIBM_NUMPY, "numpy"),
                                       (math, _LIBM_MATH, "math"),
                                       (cmath, _LIBM_CMATH, "cmath")):
        for name in names:
            if hasattr(module, name):
                replaced.append((module, name, getattr(module, name)))
                setattr(module, name, refuse(module_name, name))
    try:
        yield
    except _LibmCall as exc:
        raise OracleRefusal(
            f"case {case}: the array path called {exc} while the fixture stepped; a libm "
            f"result differs between hosts, so the fingerprint would not be a property of "
            f"the array path") from None
    finally:
        for module, name, original in reversed(replaced):
            setattr(module, name, original)


def _require_real_amplitudes(case: str, source: Any) -> None:
    """The fixture set's sources carry real, nonzero amplitudes (its minted shape)."""
    amps = getattr(source, "_point_amps", None)
    if amps is None:
        raise OracleRefusal(f"case {case}: source {source!r} publishes no deposit table")
    values = numpy.asarray(amps)
    if values.dtype.kind != "c" or numpy.any(values.imag != 0) or numpy.any(values.real == 0):
        raise OracleRefusal(
            f"case {case}: source {source!r} carries an amplitude with a nonzero imaginary "
            f"part or a zero real part; the fixture set is minted with real, nonzero "
            f"amplitudes, so this would be a different fixture set")


@contextlib.contextmanager
def _subnormals_kept() -> Iterator[Dict[str, Any]]:
    """Run with the host FPU keeping subnormals; refuse by name when it cannot."""
    from . import backends, subnormal_policy  # noqa: PLC0415

    report: Dict[str, Any] = {"flushing_at_entry": bool(backends.subnormals_flushed()),
                              "mechanism": None}
    if report["flushing_at_entry"]:
        _module, mechanism = subnormal_policy._drive_host_fpu(False)  # noqa: SLF001
        report["mechanism"] = mechanism
        if backends.subnormals_flushed():
            raise OracleRefusal(
                f"the host FPU flushes subnormals and could not be driven to keep "
                f"({mechanism}); the fixture's subnormal words would be flushed, so the "
                f"fingerprint would describe the FPU mode rather than the array path")
    try:
        yield report
    finally:
        if report["flushing_at_entry"]:
            subnormal_policy._drive_host_fpu(True)  # noqa: SLF001


def _require_keep(case: str) -> None:
    from . import backends  # noqa: PLC0415

    if backends.subnormals_flushed():
        raise OracleRefusal(
            f"case {case}: the host FPU began flushing subnormals while the case ran; "
            f"something in the step changed the FPU mode")


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One stored-current source of a case."""

    component: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    integrated: bool = False


@dataclass(frozen=True)
class CaseSpec:
    """One fixture case. Every field is data; :func:`_build` reads it."""

    name: str
    cell: Tuple[float, float, float]
    dimensions: int = 3
    pml: Optional[Mapping[str, Any]] = None
    boundaries: Optional[Mapping[str, str]] = None
    symmetry: Tuple[Tuple[str, int], ...] = ()
    cylindrical: bool = False
    beta: float = 0.0
    bfast: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    d_conductivity: bool = False
    b_conductivity: bool = False
    #: Where the conductivity lives, as a per-axis [start, stop) fraction of the stored
    #: rows; zero elsewhere. ``None`` seeds it on every cell.
    conductivity_block: Optional[Tuple[Tuple[float, float], ...]] = None
    offdiagonal: bool = False
    susceptibilities: Tuple[str, ...] = ()
    nonlinear: bool = False
    sources: Tuple[SourceSpec, ...] = ()
    repair: bool = False


CASES: Tuple[CaseSpec, ...] = (
    CaseSpec("pml_3d", (1.5, 1.25, 1.0),
             pml={"x": (2, 3), "y": (2.5, 2), "z": (0, 2)}),
    CaseSpec("pml_2d", (2.0, 1.5, 0.0), dimensions=2,
             pml={"x": (2.5, 1.5), "y": (1.75, 3)}),
    CaseSpec("no_pml_3d", (1.0, 0.75, 0.5)),
    CaseSpec("metallic_2d", (1.5, 1.25, 0.0), dimensions=2,
             boundaries={"x": "metallic"}, pml={"y": 2}),
    CaseSpec("folded_3d", (1.5, 1.25, 1.0), symmetry=(("X", 1), ("Y", -1)),
             pml={"x": (0, 2), "y": (0, 2), "z": 2}),
    CaseSpec("conductive_3d", (1.0, 0.75, 0.75), d_conductivity=True, b_conductivity=True),
    CaseSpec("conductive_pml_3d", (1.5, 1.25, 1.25), pml=2, d_conductivity=True,
             b_conductivity=True),
    # The conductivity on the upper-x, upper-y quadrant only: the interior chunk, the x-high
    # and y-high edge chunks and their corner keep it; the other five chunks keep none
    # and step the lossless PML forms (structure.cpp:883-900).
    CaseSpec("conductive_chunks_pml_2d", (2.0, 2.0, 0.0), dimensions=2,
             boundaries={"x": "metallic", "y": "metallic"}, pml={"x": 2, "y": 2},
             d_conductivity=True, b_conductivity=True,
             conductivity_block=((0.5, 1.0), (0.5, 1.0), (0.0, 1.0))),
    CaseSpec("conductive_source_2d", (1.5, 1.25, 0.0), dimensions=2, d_conductivity=True,
             sources=(SourceSpec("Ez", (0.0625, -0.0625, 0.0)),
                      SourceSpec("Ex", (0.0, 0.125, 0.0), size=(0.5, 0.0, 0.0)),
                      SourceSpec("Ey", (-0.125, 0.0, 0.0), integrated=True))),
    CaseSpec("offdiag_3d", (1.0, 0.75, 0.75), offdiagonal=True),
    CaseSpec("offdiag_pml_3d", (1.5, 1.25, 1.25), pml=2, offdiagonal=True),
    CaseSpec("dispersive_pml_2d", (1.5, 1.25, 0.0), dimensions=2, pml={"x": 2, "y": 2},
             susceptibilities=("lorentz", "drude")),
    CaseSpec("dispersive_3d", (1.0, 0.75, 0.75), susceptibilities=("lorentz",)),
    CaseSpec("nonlinear_pml_2d", (1.5, 1.25, 0.0), dimensions=2, pml={"x": 2, "y": 2},
             nonlinear=True),
    CaseSpec("nonlinear_folded_3d", (1.5, 1.25, 1.0), symmetry=(("X", 1), ("Y", -1)),
             pml={"x": (0, 2), "y": (0, 2), "z": 2}, nonlinear=True),
    CaseSpec("cylindrical_m0", (2.0, 0.0, 1.5), cylindrical=True,
             pml={"x": (0, 2), "z": 2}),
    CaseSpec("beta_pml_2d", (1.5, 1.25, 0.0), dimensions=2, pml={"x": 2, "y": 2},
             beta=0.375),
    CaseSpec("bfast_pml_2d", (1.5, 1.25, 0.0), dimensions=2, pml={"y": 2},
             bfast=(0.25, 0.125, 0.0)),
    CaseSpec("sources_pml_2d", (2.0, 2.0, 0.0), dimensions=2, pml={"x": 3, "y": 3},
             sources=(SourceSpec("Ez", (-0.8125, 0.0625, 0.0)),
                      SourceSpec("Ex", (0.0, 0.25, 0.0), size=(0.75, 0.0, 0.0)),
                      SourceSpec("Hz", (0.1875, -0.1875, 0.0)),
                      SourceSpec("Ey", (0.125, 0.0, 0.0), integrated=True))),
    CaseSpec("repair_pml_2d", (1.5, 1.5, 0.0), dimensions=2, pml={"x": 2, "y": 2},
             sources=(SourceSpec("Ez", (0.0625, 0.0625, 0.0)),
                      SourceSpec("Hz", (-0.1875, 0.125, 0.0))),
             repair=True),
)


def _declared() -> Tuple[str, ...]:
    """The form classes every mint must reach, written out per case.

    STATIC ON PURPOSE: derived from nothing a case builds, so a case that stops reaching a
    form leaves its class declared and counted zero, and the mint refuses by name.
    """
    out: List[str] = []

    def add(case: str, substep: str, components: Sequence[str], forms: Sequence[str]) -> None:
        out.extend(f"{case}/{substep}/{c}/{form}" for c in components for form in forms)

    pml_curl = ("recurrence", "plain")
    pml_constitutive = ("aux", "plain")
    quadrants = ("dsig_and_dsigu", "dsig_only", "dsigu_only", "neither")

    add("pml_3d", "curl_B", _B, pml_curl)
    add("pml_3d", "curl_D", _D, pml_curl)
    add("pml_3d", "constitutive_H", _H, pml_constitutive)
    add("pml_3d", "constitutive_E", _E, pml_constitutive)
    # The run's first update_H starts H and f_w_H from B on the dsigw cells.
    add("pml_3d", "h_start", _H, ("copied",))

    # 2-D: z is invariant, so a curl whose dsigu is z and a constitutive whose own axis
    # is z take the plain form only.
    add("pml_2d", "curl_B", ("By", "Bz"), pml_curl)
    add("pml_2d", "curl_B", ("Bx",), ("plain",))
    add("pml_2d", "curl_D", ("Dy", "Dz"), pml_curl)
    add("pml_2d", "curl_D", ("Dx",), ("plain",))
    add("pml_2d", "constitutive_H", ("Hx", "Hy"), pml_constitutive)
    add("pml_2d", "constitutive_H", ("Hz",), ("plain",))
    add("pml_2d", "constitutive_E", ("Ex", "Ey"), pml_constitutive)
    add("pml_2d", "constitutive_E", ("Ez",), ("plain",))

    add("no_pml_3d", "curl_B", _B, ("plain",))
    add("no_pml_3d", "curl_D", _D, ("plain",))
    add("no_pml_3d", "constitutive_H", _H, ("alias",))
    add("no_pml_3d", "constitutive_E", _E, ("on_demand",))

    add("metallic_2d", "zero_metal_B", ("Bx",), ("wall_x",))
    add("metallic_2d", "zero_metal_D", ("Dy", "Dz"), ("wall_x",))
    add("metallic_2d", "curl_D", ("Dz",), pml_curl)

    # The near fill writes the components whose Yee shift on the folded axis is 0, the
    # far fill those whose shift is 1 (fields.IYEE_SHIFTS).
    add("folded_3d", "fill_B", ("Bx",), ("mirror_x",))
    add("folded_3d", "fill_B", ("By",), ("mirror_y",))
    add("folded_3d", "fill_D", ("Dy", "Dz"), ("mirror_x",))
    add("folded_3d", "fill_D", ("Dx", "Dz"), ("mirror_y",))
    add("folded_3d", "fill", ("parity",), ("even", "odd"))
    add("folded_3d", "far_fill_B", ("By", "Bz"), ("x",))
    add("folded_3d", "far_fill_B", ("Bx", "Bz"), ("y",))
    add("folded_3d", "far_fill_D", ("Dx",), ("x",))
    add("folded_3d", "far_fill_D", ("Dy",), ("y",))
    add("folded_3d", "curl_D", _D, pml_curl)

    add("conductive_3d", "curl_B", _B, ("conductive",))
    add("conductive_3d", "curl_D", _D, ("conductive",))

    add("conductive_pml_3d", "curl_B", _B, quadrants)
    add("conductive_pml_3d", "curl_D", _D, quadrants)

    # Per chunk: whether the chunk keeps the component's conductivity (``kept``) or not
    # (``none``), crossed with the quadrant. In 2-D z is invariant, so the z-normal
    # component reaches every quadrant and the other two the two their resolved axis
    # gives; the interior chunk is the only one with neither layer, and it keeps.
    # (x-normal: dsig y; y-normal: dsigu x; z-normal: dsig x and dsigu y.)
    for family, (x_normal, y_normal, z_normal) in (("B", _B), ("D", _D)):
        add("conductive_chunks_pml_2d", f"curl_{family}_chunk", (z_normal,),
            tuple(f"kept_{q}" for q in quadrants)
            + tuple(f"none_{q}" for q in quadrants if q != "neither"))
        add("conductive_chunks_pml_2d", f"curl_{family}_chunk", (x_normal,),
            ("kept_dsig_only", "kept_neither", "none_dsig_only", "none_neither"))
        add("conductive_chunks_pml_2d", f"curl_{family}_chunk", (y_normal,),
            ("kept_dsigu_only", "kept_neither", "none_dsigu_only", "none_neither"))

    add("conductive_source_2d", "deposit_D", ("Ez",), ("point",))
    add("conductive_source_2d", "deposit_D", ("Ex",), ("sheet",))
    add("conductive_source_2d", "deposit_D", ("Ey",), ("integrated",))
    add("conductive_source_2d", "conductive_replay", ("Ez", "Ex"), ("points",))

    for case in ("offdiag_3d", "offdiag_pml_3d"):
        add(case, "offdiag_E", _E, ("row", "diagonal_cell"))
    add("offdiag_pml_3d", "constitutive_E", _E, pml_constitutive)

    for index, kind in enumerate(("lorentz", "drude")):
        add("dispersive_pml_2d", f"update_P{index}_{kind}", _E, ("sigma", "zero_sigma"))
    add("dispersive_pml_2d", "constitutive_E", ("Ex", "Ey"), pml_constitutive)
    add("dispersive_3d", "update_P0_lorentz", _E, ("sigma", "zero_sigma"))
    add("dispersive_3d", "constitutive_E", _E, ("stored",))

    add("nonlinear_pml_2d", "nonlinear_E", _E, ("pade", "linear"))
    # Folded: E, and f_w on the accumulation's cells, imaged at each ghost after the update.
    add("nonlinear_folded_3d", "nonlinear_E", _E, ("pade", "linear"))
    for substep in ("nonlinear_ghost_E", "nonlinear_ghost_fw"):
        add("nonlinear_folded_3d", substep, ("Ey", "Ez"), ("mirror_x",))
        add("nonlinear_folded_3d", substep, ("Ex", "Ez"), ("mirror_y",))

    add("cylindrical_m0", "cyl_axis_D", ("Dz",), ("axis_row",))
    add("cylindrical_m0", "cyl_axis_D", ("Dy",), ("zeroed",))
    # The r layer's chunk cut: the cells past it read a prefix sum that restarts at the
    # cut (step_db.cpp:104-118, per structure chunk).
    add("cylindrical_m0", "cyl_prefix_D", ("Dz",), ("prefix", "chunk_restart"))
    add("cylindrical_m0", "cyl_prefix_B", ("Bz",), ("prefix", "chunk_restart"))
    # r = x, phi = y (invariant), z = z: Dz's dsigu is phi, so it steps plain only.
    add("cylindrical_m0", "curl_D", ("Dx", "Dy"), pml_curl)
    add("cylindrical_m0", "curl_D", ("Dz",), ("plain",))

    add("beta_pml_2d", "beta_B", ("Bx", "By"), ("coupled",))
    add("beta_pml_2d", "beta_D", ("Dx", "Dy"), ("coupled",))
    # The beta ladder's two forms: Bx and Dx have the invariant z as dsigu, so the plain
    # form only; By and Dy have x, which carries a layer.
    add("beta_pml_2d", "beta_B", ("Bx", "By"), ("ladder_plain",))
    add("beta_pml_2d", "beta_B", ("By",), ("ladder_aux",))
    add("beta_pml_2d", "beta_D", ("Dx", "Dy"), ("ladder_plain",))
    add("beta_pml_2d", "beta_D", ("Dy",), ("ladder_aux",))

    add("bfast_pml_2d", "bfast_B", _B, ("iir",))
    add("bfast_pml_2d", "bfast_D", _D, ("iir",))
    # A layer on y only: Bz and Dz (dsigu y) take both ladder forms, the rest the plain
    # one. The one-partner components (z invariant) are x and y: Bx and Dx have dsig y,
    # so both of BFAST's one-partner lines; By and Dy carry no layer on either axis.
    add("bfast_pml_2d", "bfast_B", _B, ("ladder_plain",))
    add("bfast_pml_2d", "bfast_B", ("Bz",), ("ladder_aux",))
    add("bfast_pml_2d", "bfast_D", _D, ("ladder_plain",))
    add("bfast_pml_2d", "bfast_D", ("Dz",), ("ladder_aux",))
    add("bfast_pml_2d", "bfast_B", ("Bx", "By"), ("one_partner_plain",))
    add("bfast_pml_2d", "bfast_B", ("Bx",), ("one_partner_layer",))
    add("bfast_pml_2d", "bfast_D", ("Dx", "Dy"), ("one_partner_plain",))
    add("bfast_pml_2d", "bfast_D", ("Dx",), ("one_partner_layer",))

    add("sources_pml_2d", "deposit_D", ("Ez",), ("point",))
    add("sources_pml_2d", "deposit_D", ("Ex",), ("sheet",))
    add("sources_pml_2d", "deposit_D", ("Ey",), ("integrated",))
    add("sources_pml_2d", "deposit_B", ("Hz",), ("point",))

    add("repair_pml_2d", "repair_D", ("Ez",), ("points",))
    add("repair_pml_2d", "repair_B", ("Hz",), ("points",))
    return tuple(out)


DECLARED_FORM_CLASSES: Tuple[str, ...] = _declared()


# ---------------------------------------------------------------------------
# Building one case
# ---------------------------------------------------------------------------


def _within(spec: CaseSpec, values: numpy.ndarray) -> numpy.ndarray:
    """``values`` inside the case's ``conductivity_block`` and exact zero outside it."""
    if spec.conductivity_block is None:
        return values
    kept = numpy.zeros(values.shape, dtype=bool)
    kept[tuple(slice(int(round(lo * n)), int(round(hi * n)))
               for (lo, hi), n in zip(spec.conductivity_block, values.shape))] = True
    return numpy.where(kept, values, numpy.float32(0.0)).astype(numpy.float32)


def _build(spec: CaseSpec) -> Tuple[Any, List[Any], List[_StoredCurrents]]:
    """A NumPy ``FdtdDriver`` for one case, with every material and state volume seeded."""
    from .dispersion import DRUDE, LORENTZIAN, Susceptibility  # noqa: PLC0415
    from .driver import FdtdDriver  # noqa: PLC0415
    from .grid import Mirror  # noqa: PLC0415

    with _libm_free_pml_grading():
        driver = FdtdDriver(
            cell_size=spec.cell, resolution=RESOLUTION, courant=COURANT,
            force_complex_fields=False,
            symmetry=tuple(Mirror(axis, phase) for axis, phase in spec.symmetry),
            boundaries=dict(spec.boundaries) if spec.boundaries else None,
            dimensions=spec.dimensions, cylindrical=spec.cylindrical, m=0,
            beta=spec.beta, bfast_scaled_k=spec.bfast, prefer_gpu=False)
        if driver.dt != 2.0 ** -4:
            raise OracleRefusal(f"case {spec.name}: dt is {driver.dt!r}, not 2**-4")
        shape = tuple(int(n) for n in driver.grid.shape)
        epsilon = {c: _volume(_seed(spec.name, "eps", c), shape, 1.0, 3.0) for c in _E}
        offdiagonal = None
        if spec.offdiagonal:
            offdiagonal = {
                row: {partner: _volume(_seed(spec.name, "chi1inv", row, partner), shape,
                                       -0.0625, 0.0625, zero_fraction=0.4)
                      for partner in _E if partner != row}
                for row in _E}
        driver.set_epsilon_components(epsilon, chi1inv_offdiagonal=offdiagonal)
        if spec.d_conductivity:
            driver.set_conductivity({c: _within(spec, _volume(_seed(spec.name, "sigma_d", c),
                                                              shape, 0.0, 2.0, zero_fraction=0.3))
                                     for c in _D})
        if spec.b_conductivity:
            driver.set_b_conductivity({c: _within(spec, _volume(_seed(spec.name, "sigma_b", c),
                                                                shape, 0.0, 2.0,
                                                                zero_fraction=0.3))
                                       for c in _B})
        for index, kind in enumerate(spec.susceptibilities):
            angular = LORENTZ_ANGULAR if kind == "lorentz" else DRUDE_ANGULAR
            frequency = angular / (2.0 * math.pi)
            if 2.0 * math.pi * frequency != angular:
                raise OracleRefusal(f"case {spec.name}: 2*pi*f does not round-trip to "
                                    f"{angular!r}, so (2*pi*f*dt)**2 would be libm output")
            term = Susceptibility(frequency=frequency, gamma=0.0625 * (index + 1),
                                  kind=LORENTZIAN if kind == "lorentz" else DRUDE)
            driver.add_susceptibility(term, {c: _volume(_seed(spec.name, "pol", str(index), c),
                                                        shape, 0.0, 1.0, zero_fraction=0.3)
                                             for c in _E})
        if spec.nonlinear and not spec.symmetry:
            # chi2 is not mirror-symmetric, and a plane that makes its component odd
            # refuses it (Fields.set_nonlinear_volumes): a folded case takes chi3 alone.
            driver.set_chi2({c: _volume(_seed(spec.name, "chi2", c), shape, 0.0, 0.03125,
                                        zero_fraction=0.3) for c in _E})
        if spec.nonlinear:
            driver.set_chi3({c: _volume(_seed(spec.name, "chi3", c), shape, 0.0, 0.03125,
                                        zero_fraction=0.3) for c in _E})
        if spec.pml is not None:
            driver.setup_pml(spec.pml)
        sources: List[Any] = []
        currents: List[_StoredCurrents] = []
        for index, source in enumerate(spec.sources):
            table = _StoredCurrents(_seed(spec.name, "current", str(index)), driver.dt)
            created = driver.add_source({
                "component": source.component, "center": source.center,
                "size": source.size, "source_type": "custom", "src_func": table,
                "is_integrated": source.integrated})
            _require_real_amplitudes(spec.name, created)
            sources.append(created)
            currents.append(table)
    for name, array in _stored_volumes(driver.fields).items():
        array[...] = _state_words(_seed(spec.name, "state", name), tuple(array.shape))
    return driver, sources, currents


def _stored_volumes(fields: Any) -> Dict[str, Any]:
    """Every allocated state volume, live, by name: fields, auxiliaries and polarizations."""
    out: Dict[str, Any] = {}
    for component in _D + _B + _E + _H:
        for prefix in ("", "fu_", "f_cond_", "f_bfast_", "f_w_"):
            array = getattr(fields, prefix + component, None)
            if array is not None:
                out[prefix + component] = array
    for index, state in enumerate(getattr(fields, "polarizations", ()) or ()):
        for component, array in state.P.items():
            out[f"P[{index}].{component}"] = array
        for component, array in state.P_prev.items():
            out[f"P_prev[{index}].{component}"] = array
    return out


# ---------------------------------------------------------------------------
# Form classes: predicates on the fixture's own state, never on code branches
# ---------------------------------------------------------------------------


def membership_line(n_cells: int, faces: Tuple[float, float], half_integer: bool,
                    wraps: bool) -> numpy.ndarray:
    """Per stored cell of one axis: does a chunk carrying this axis's layer own it?

    The one-rank chunk rule for a layer of thickness ``t`` cells: the chunk spans
    ``int(t + 1.5)`` cells from its wall; at integer positions the low chunk owns cells
    ``1 .. span`` and the high chunk the top ``span - 1`` planes plus, on a wrapping axis,
    cell 0; at half-integer positions ``0 .. span - 1`` and the top ``span``
    (structure.cpp:509-524, vec.hpp:1100-1104). Written here, not read from the PML, so
    this module's class counts do not move when the package's own membership code does.
    """
    low, high = faces
    inside = numpy.zeros(n_cells, dtype=bool)
    if low > 0:
        span = min(n_cells, int(low + 1.5))
        if half_integer:
            inside[0:span] = True
        else:
            inside[1:min(n_cells, span + 1)] = True
    if high > 0:
        span = min(n_cells, int(high + 1.5))
        if half_integer:
            inside[n_cells - span:] = True
        else:
            inside[max(0, n_cells - span + 1):] = True
            if wraps:
                inside[0] = True
    return inside


def chunk_samples(n_cells: int, faces: Tuple[float, float], half_integer: bool,
                  wraps: bool) -> List[numpy.ndarray]:
    """Per one-rank chunk interval of one axis (low layer, interior, high layer, those that
    exist): the stored rows its conductivity is sampled on.

    ``LOOP_OVER_VOL`` (vec.hpp:170-172) visits the interval's owned rows plus one more:
    the row below them on the integer lattice and the row above them on the half-integer
    one, at its periodic image on a wrapping axis and at the last stored row past the top
    of one that does not. The owned rows are :func:`membership_line`'s rule, written here
    and not read from the package, so the class counts do not move when its code does.
    """
    if n_cells == 1:
        return [numpy.zeros(1, dtype=numpy.int64)]  # an invariant axis: one row, one interval
    low, high = faces
    rows = numpy.arange(n_cells)
    layer = membership_line(n_cells, faces, half_integer, wraps)
    owned = numpy.ones(n_cells, dtype=bool)
    if not half_integer and not wraps:
        owned[0] = False  # the low wall plane
    high_rows = layer & (rows >= n_cells // 2)
    if wraps and not half_integer and high > 0:
        high_rows[0] = True
    low_rows = layer & ~high_rows
    intervals = []
    if low > 0:
        intervals.append(low_rows)
    intervals.append(owned & ~layer)
    if high > 0:
        intervals.append(high_rows)
    out = []
    for member in intervals:
        where = numpy.flatnonzero(member)
        if where.size == 0:
            raise OracleRefusal(f"an axis of {n_cells} rows with layers {faces} leaves a "
                                f"one-rank chunk interval with no owned row")
        if wraps and not half_integer and high > 0 and member is intervals[-1]:
            run = where[where > 0]
            extra = int(run[0]) - 1
        elif half_integer:
            extra = int(where[-1]) + 1
        else:
            extra = int(where[0]) - 1
        extra = extra % n_cells if wraps else min(max(extra, 0), n_cells - 1)
        out.append(numpy.union1d(where, [extra]))
    return out


def chunk_keeps(sigma: numpy.ndarray, frames: Sequence[Tuple[int, Tuple[float, float], bool, bool]],
                ) -> numpy.ndarray:
    """Per stored cell: does the one-rank chunk owning it keep a nonzero conductivity sample?

    ``frames`` gives, per axis, (stored rows, layer faces, half-integer lattice, wraps); an
    invariant axis is one row with no layer. A chunk keeps its array when any of its
    :func:`chunk_samples` is nonzero (structure.cpp:883-900); an unowned wall row reads
    the chunk beside it.
    """
    per_axis = []
    for n_cells, faces, half_integer, wraps in frames:
        samples = chunk_samples(n_cells, faces, half_integer, wraps)
        label = numpy.zeros(n_cells, dtype=numpy.int64)
        if len(samples) > 1:
            layer = membership_line(n_cells, faces, half_integer, wraps)
            rows = numpy.arange(n_cells)
            low, high = faces
            high_rows = layer & (rows >= n_cells // 2)
            if wraps and not half_integer and high > 0:
                high_rows[0] = True
            index_interior = 1 if low > 0 else 0
            label[:] = index_interior
            if low > 0:
                label[layer & ~high_rows] = 0
                if not half_integer and not wraps:
                    label[0] = 0
            if high > 0:
                label[high_rows] = len(samples) - 1
        per_axis.append((samples, label))
    keeps = numpy.zeros(tuple(len(samples) for samples, _ in per_axis), dtype=bool)
    for chunk in numpy.ndindex(keeps.shape):
        box = numpy.ix_(*(per_axis[a][0][chunk[a]] for a in range(3)))
        keeps[chunk] = bool(numpy.any(sigma[box] != 0))
    labels = [per_axis[a][1] for a in range(3)]
    return keeps[labels[0][:, None, None], labels[1][None, :, None], labels[2][None, None, :]]


def _broadcast(line: numpy.ndarray, axis: int) -> numpy.ndarray:
    shape = [1, 1, 1]
    shape[axis] = line.size
    return line.reshape(shape)


def _count(mask: Any, shape: Tuple[int, ...]) -> int:
    return int(numpy.count_nonzero(numpy.broadcast_to(mask, shape)))


def _nonzero(value: Any, shape: Tuple[int, ...]) -> numpy.ndarray:
    return numpy.broadcast_to(numpy.asarray(value) != 0, shape)


def _form_classes(spec: CaseSpec, driver: Any, sources: Sequence[Any]) -> Dict[str, int]:
    """Cell counts per (case, sub-step, component, form), from the fixture's state."""
    from . import fields as fields_module  # noqa: PLC0415
    from .stepping import B_CURL_TERMS, D_CURL_TERMS  # noqa: PLC0415

    fields, grid, pml = driver.fields, driver.grid, driver.pml
    shape = tuple(int(n) for n in grid.shape)
    total = int(numpy.prod(shape))
    counts: Dict[str, int] = {}

    def add(substep: str, component: str, form: str, value: int) -> None:
        key = f"{spec.name}/{substep}/{component}/{form}"
        counts[key] = counts.get(key, 0) + int(value)

    pml_active = pml is not None and pml.is_active

    def line(axis_name: str, half_integer: bool) -> numpy.ndarray:
        axis = _AXES.index(axis_name)
        return _broadcast(membership_line(shape[axis], pml.thickness_by_face[axis],
                                          half_integer, grid.axis_wraps(axis)), axis)

    for family, terms in (("B", B_CURL_TERMS), ("D", D_CURL_TERMS)):
        for term in terms:
            half = family == "B"
            conductive = fields.condfac_for(term.target) is not None
            if pml_active:
                dsigu = numpy.broadcast_to(line(term.dsigu, half), shape)
                if conductive:
                    dsig = numpy.broadcast_to(line(term.dsig, half), shape)
                    add(f"curl_{family}", term.target, "dsig_and_dsigu", _count(dsig & dsigu, shape))
                    add(f"curl_{family}", term.target, "dsig_only", _count(dsig & ~dsigu, shape))
                    add(f"curl_{family}", term.target, "dsigu_only", _count(~dsig & dsigu, shape))
                    add(f"curl_{family}", term.target, "neither", _count(~dsig & ~dsigu, shape))
                    shifts = fields_module.IYEE_SHIFTS[term.target]
                    keeps = chunk_keeps(
                        numpy.asarray(fields.conductivity_for(term.target)),
                        [(shape[a], pml.thickness_by_face[a] if not grid.is_invariant(a)
                          else (0, 0), bool(shifts[a]), bool(grid.axis_wraps(a))
                          and not grid.is_invariant(a)) for a in range(3)])
                    for kept, where in (("kept", keeps), ("none", ~keeps)):
                        for quadrant, cells in (("dsig_and_dsigu", dsig & dsigu),
                                                ("dsig_only", dsig & ~dsigu),
                                                ("dsigu_only", ~dsig & dsigu),
                                                ("neither", ~dsig & ~dsigu)):
                            add(f"curl_{family}_chunk", term.target, f"{kept}_{quadrant}",
                                _count(cells & where, shape))
                else:
                    add(f"curl_{family}", term.target, "recurrence", _count(dsigu, shape))
                    add(f"curl_{family}", term.target, "plain", _count(~dsigu, shape))
            else:
                add(f"curl_{family}", term.target, "conductive" if conductive else "plain", total)
            # The folded passes (stepping._apply_folded_increment) take the fu stage on
            # the dsigu cells and f += raw*siginv elsewhere; BFAST's one-partner plain
            # line (no layer on either axis, no conductivity) stores no previous state.
            folded = []
            if grid.beta != 0.0 and term.target in ("Bx", "By", "Dx", "Dy"):
                add(f"beta_{family}", term.target, "coupled", total)
                folded.append("beta")
            if grid.bfast_active:
                add(f"bfast_{family}", term.target, "iir", total)
                folded.append("bfast")
            if folded and pml_active:
                dsigu = numpy.broadcast_to(line(term.dsigu, half), shape)
                for name in folded:
                    add(f"{name}_{family}", term.target, "ladder_aux", _count(dsigu, shape))
                    add(f"{name}_{family}", term.target, "ladder_plain", _count(~dsigu, shape))
            if grid.bfast_active:
                one_partner = (grid.is_invariant(term.first_axis)
                               != grid.is_invariant(term.second_axis))
                if one_partner:
                    plain = numpy.ones(shape, dtype=bool)
                    if conductive:
                        plain[...] = False
                    elif pml_active:
                        plain = (~numpy.broadcast_to(line(term.dsig, half), shape)
                                 & ~numpy.broadcast_to(line(term.dsigu, half), shape))
                    add(f"bfast_{family}", term.target, "one_partner_plain", _count(plain, shape))
                    add(f"bfast_{family}", term.target, "one_partner_layer", _count(~plain, shape))

    for family, components, half in (("H", _H, False), ("E", _E, True)):
        for component in components:
            own = component[1]
            if pml_active:
                member = numpy.broadcast_to(line(own, half), shape)
                add(f"constitutive_{family}", component, "aux", _count(member, shape))
                add(f"constitutive_{family}", component, "plain", _count(~member, shape))
                if family == "H" and not fields.h_separated:
                    add("h_start", component, "copied", _count(member, shape))
            elif family == "H":
                add("constitutive_H", component, "alias", total)
            else:
                add("constitutive_E", component, "stored" if fields.stores_E else "on_demand",
                    total)

    if fields.has_offdiagonal_epsilon:
        for row in _E:
            partners = fields.chi1inv_offdiagonal_for(row)
            coupled = numpy.zeros(shape, dtype=bool)
            for volume in partners.values():
                coupled |= _nonzero(volume, shape)
            add("offdiag_E", row, "row", _count(coupled, shape))
            add("offdiag_E", row, "diagonal_cell", _count(~coupled, shape))

    if fields.has_nonlinearity:
        for component in _E:
            active = _nonzero(fields.chi2_for(component), shape) | \
                _nonzero(fields.chi3_for(component), shape)
            add("nonlinear_E", component, "pade", _count(active, shape))
            add("nonlinear_E", component, "linear", _count(~active, shape))
        if grid.has_symmetry():
            # stepping._image_nonlinear_mirror_ghosts: every ghost plane of E, and f_w on
            # the accumulation's cells of that plane (its own axis is never the folded one).
            for component in _E:
                shifts = fields_module.IYEE_SHIFTS[component]
                member = (numpy.broadcast_to(line(component[1], True), shape) if pml_active
                          else numpy.zeros(shape, dtype=bool))
                for axis in range(3):
                    if not grid.is_mirrored(axis) or shifts[axis] != 0:
                        continue
                    ghost = (slice(None),) * axis + (0,)
                    add("nonlinear_ghost_E", component, f"mirror_{_AXES[axis]}",
                        total // shape[axis])
                    add("nonlinear_ghost_fw", component, f"mirror_{_AXES[axis]}",
                        int(numpy.count_nonzero(member[ghost])))

    for index, state in enumerate(fields.polarizations):
        kind = "drude" if state.susceptibility.is_drude else "lorentz"
        for component in state.driven():
            driven = _nonzero(state.sigma[component], shape)
            add(f"update_P{index}_{kind}", component, "sigma", _count(driven, shape))
            add(f"update_P{index}_{kind}", component, "zero_sigma", _count(~driven, shape))

    if grid.has_symmetry():
        for family, components in (("B", _B), ("D", _D)):
            for component in components:
                shifts = fields_module.IYEE_SHIFTS[component]
                for axis in range(3):
                    if not grid.is_mirrored(axis):
                        continue
                    face = total // shape[axis]
                    if shifts[axis] == 0:
                        parity = fields_module.mirror_parity(component, axis,
                                                             grid.mirror_phase(axis))
                        add(f"fill_{family}", component, f"mirror_{_AXES[axis]}", face)
                        add("fill", "parity", "even" if parity > 0 else "odd", face)
                    elif grid.stored_cells(axis) > grid.owned_cells(axis):
                        add(f"far_fill_{family}", component, _AXES[axis], face)

    walled = [axis for axis in range(3) if grid.is_metallic(axis) and not grid.is_mirrored(axis)]
    for family, components in (("B", _B), ("D", _D)):
        for component in components:
            shifts = fields_module.IYEE_SHIFTS[component]
            for axis in walled:
                if shifts[axis] == 0:
                    add(f"zero_metal_{family}", component, f"wall_{_AXES[axis]}",
                        total // shape[axis])

    if getattr(grid, "cylindrical", False):
        row = total // shape[0]
        add("cyl_axis_D", "Dz", "axis_row", row)
        add("cyl_axis_D", "Dy", "zeroed", row)
        add("cyl_prefix_D", "Dz", "prefix", total)
        add("cyl_prefix_B", "Bz", "prefix", total)
        from .stepping import _cylindrical_scan_starts  # noqa: PLC0415
        for substep, component in (("cyl_prefix_D", "Dz"), ("cyl_prefix_B", "Bz")):
            starts = _cylindrical_scan_starts(fields, pml if pml_active else None, component)
            if starts is not None:
                add(substep, component, "chunk_restart",
                    _count(numpy.asarray(starts) > 0, shape))

    for source in sources:
        family = "B" if source.field_type == "B" else "D"
        points = int(source._n_source_points)  # noqa: SLF001
        if source.is_integrated:
            kind = "integrated"
        elif any(extent > 0.0 for extent in source.size):
            kind = "sheet"
        else:
            kind = "point"
        add(f"deposit_{family}", source.component, kind, points)
        if (family == "D" and not source.is_integrated and fields.has_conductivity
                and fields.condinv_for("D" + source.component[1]) is not None):
            add("conductive_replay", source.component, "points", points)
    return counts


# ---------------------------------------------------------------------------
# Stepping and digesting
# ---------------------------------------------------------------------------


def _digest_state(digest: Any, driver: Any, sources: Sequence[Any]) -> None:
    fields = driver.fields
    for name, array in sorted(_stored_volumes(fields).items()):
        values = numpy.ascontiguousarray(array)
        digest.update(f"{name}|{values.dtype.str}|{values.shape}\n".encode("utf-8"))
        digest.update(values.tobytes())
    if not fields.stores_E:
        for component in _E:
            values = numpy.ascontiguousarray(fields.get_E(component))
            digest.update(f"get_E:{component}|{values.dtype.str}|{values.shape}\n".encode())
            digest.update(values.tobytes())


def _repair_step(driver: Any, sources: Sequence[Any], counts: Dict[str, int],
                 case: str) -> None:
    """One step with each seam's curl and constitutive run before its deposit, then repaired.

    The fused route's order on the array path: the deposit points are saved, the curl and
    the constitutive run on the uninjected field (with the wall clear the fused kernels
    carry inline), the driver's injection, fills and wall clear run, the integrated
    sources' ``f_minus_p`` words are written (``sources.f_minus_p_begin``, where the
    driver writes them, before the constitutive consult the repair occupies),
    ``deposit_repair.apply`` recomputes the constitutive at the deposit points, and the
    words are restored.
    """
    from . import deposit_repair, sources as sources_module, stepping  # noqa: PLC0415

    fields, pml = driver.fields, driver.pml
    time_value = driver.time
    dt = driver.dt
    driver._require_src_vols()  # noqa: SLF001 - the deposit tables the driver's step binds
    magnetic = [s for s in sources if s.field_type == "B"]
    electric = [s for s in sources if s.field_type != "B"]

    saved = deposit_repair.save(fields, magnetic, "B", pml)
    stepping.step_B(fields, pml)
    stepping.zero_metal_B(fields)
    stepping.update_H(fields, pml)
    for source in magnetic:
        source.inject(fields, time_value)
    stepping.fill_symmetry_bc_B(fields)
    stepping.zero_metal_B(fields)
    stepping.fill_folded_far_ghosts_B(fields)
    held = sources_module.f_minus_p_begin(fields, magnetic, "B", time_value + 0.5 * dt)
    repaired = deposit_repair.apply(fields, pml, magnetic, "B", saved)
    sources_module.f_minus_p_end(fields, held)
    counts[f"{case}/repair_B/Hz/points"] = counts.get(f"{case}/repair_B/Hz/points", 0) + repaired

    saved = deposit_repair.save(fields, electric, "D", pml)
    stepping.step_D(fields, pml)
    stepping.zero_metal_D(fields)
    stepping.update_E(fields, pml)
    for source in electric:
        source.inject(fields, time_value + 0.5 * dt)
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    held = sources_module.f_minus_p_begin(fields, electric, "D", time_value + dt)
    repaired = deposit_repair.apply(fields, pml, electric, "D", saved)
    sources_module.f_minus_p_end(fields, held)
    counts[f"{case}/repair_D/Ez/points"] = counts.get(f"{case}/repair_D/Ez/points", 0) + repaired
    stepping.update_P(fields, pml)
    driver.step_count += 1


#: The primary volumes every case must move: a case that steps without changing them
#: measures nothing, and its digest would be the seed's.
_PRIMARIES = _D + _B


def _run_case(spec: CaseSpec) -> Tuple[str, Dict[str, int]]:
    driver, sources, currents = _build(spec)
    counts = _form_classes(spec, driver, sources)
    seeded = {name: numpy.array(getattr(driver.fields, name), copy=True) for name in _PRIMARIES}
    digest = hashlib.sha256(f"{FIXTURE_SET}|{spec.name}\n".encode("utf-8"))
    _digest_state(digest, driver, sources)
    with _no_libm_calls(spec.name):
        for _step in range(STEPS):
            if spec.repair:
                _repair_step(driver, sources, counts, spec.name)
            else:
                driver.step()
            _digest_state(digest, driver, sources)
    still = [name for name in _PRIMARIES
             if numpy.array_equal(seeded[name].view(numpy.uint32),
                                  numpy.asarray(getattr(driver.fields, name)).view(numpy.uint32))]
    if still:
        raise OracleRefusal(f"case {spec.name}: {still} did not move over {STEPS} steps, so "
                            f"the case does not exercise the sub-steps that write them")
    if sources and not all(table.reads for table in currents):
        raise OracleRefusal(f"case {spec.name}: a stored-current source was never read, so "
                            f"its deposit did not reach the state")
    driver.close()
    return digest.hexdigest(), counts


def _host() -> Dict[str, Any]:
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "machine": platform.machine(), "system": platform.system()}


def run(cases: Optional[Sequence[CaseSpec]] = None,
        progress: Optional[Callable[[str], None]] = None) -> OracleRun:
    """Step every case and return the fingerprint, the per-case digests and the counts.

    ``progress`` receives one line per case: which case of how many, its digest prefix,
    and the elapsed time.
    """
    chosen = tuple(CASES if cases is None else cases)
    names = [spec.name for spec in chosen]
    if len(set(names)) != len(names):
        raise OracleRefusal(f"case names repeat: {sorted(names)}")
    started = time.perf_counter()
    case_sha256: Dict[str, str] = {}
    form_classes: Dict[str, int] = {}
    with _subnormals_kept() as fpu:
        for position, spec in enumerate(chosen, 1):
            case_started = time.perf_counter()
            digest, counts = _run_case(spec)
            _require_keep(spec.name)
            case_sha256[spec.name] = digest
            for key, value in counts.items():
                form_classes[key] = form_classes.get(key, 0) + value
            if progress is not None:
                progress(f"case {position}/{len(chosen)} {spec.name}: {digest[:16]} "
                         f"({time.perf_counter() - case_started:.2f} s)")
    combined = hashlib.sha256(f"{FIXTURE_SET}\n".encode("utf-8"))
    for name in names:
        combined.update(f"{name}\t{case_sha256[name]}\n".encode("utf-8"))
    host = _host()
    host["fpu_flushing_at_entry"] = fpu["flushing_at_entry"]
    host["fpu_mechanism"] = fpu["mechanism"]
    return OracleRun(FIXTURE_SET, combined.hexdigest(), case_sha256,
                     dict(sorted(form_classes.items())), host,
                     round(time.perf_counter() - started, 3))


_LIVE: Dict[str, OracleRun] = {}


def live() -> OracleRun:
    """The fixture set's run on this tree, computed once per process."""
    if FIXTURE_SET not in _LIVE:
        _LIVE[FIXTURE_SET] = run()
    return _LIVE[FIXTURE_SET]


def coverage_problems(form_classes: Mapping[str, int]) -> List[str]:
    """Why a run may not be minted: a declared class with no cell, or an undeclared one."""
    problems = []
    empty = [name for name in DECLARED_FORM_CLASSES if form_classes.get(name, 0) <= 0]
    if empty:
        problems.append(f"{len(empty)} of {len(DECLARED_FORM_CLASSES)} declared form "
                        f"classes were reached by no cell: {empty}")
    declared_cases = {name.split("/", 1)[0] for name in DECLARED_FORM_CLASSES}
    stray_cases = sorted({name.split("/", 1)[0] for name in form_classes} - declared_cases)
    if stray_cases:
        problems.append(f"cases count form classes but declare none: {stray_cases}")
    return problems


# ---------------------------------------------------------------------------
# Provenance and the ledger leaves
# ---------------------------------------------------------------------------


def provenance_code_sha256(root: Path) -> Tuple[Dict[str, str], List[str]]:
    """``code_identity`` digests of :data:`PROVENANCE_MODULES` under ``root``, and the absent ones."""
    from .code_identity import code_digest_of_path  # noqa: PLC0415

    digests: Dict[str, str] = {}
    absent: List[str] = []
    for name in PROVENANCE_MODULES:
        path = Path(root) / name
        if path.is_file():
            digests[name] = code_digest_of_path(path)
        else:
            absent.append(name)
    return digests, absent


def entry_leaf(result: OracleRun) -> Dict[str, str]:
    """The leaf every byte-binding weld entry carries."""
    return {"fixture_set": result.fixture_set,
            "fingerprint_sha256": result.fingerprint_sha256}


def dispatch_leaf(result: OracleRun, root: Path) -> Dict[str, Any]:
    """The leaf ``driver_dispatch`` carries: the judged values plus what was measured."""
    digests, absent = provenance_code_sha256(root)
    return {
        "fixture_set": result.fixture_set,
        "fingerprint_sha256": result.fingerprint_sha256,
        "case_sha256": dict(result.case_sha256),
        "form_classes": dict(result.form_classes),
        "deferred_form_classes": dict(DEFERRED_FORM_CLASSES),
        "provenance_code_sha256": digests,
        "provenance_absent": absent,
        "minted_with": {"tool": TOOL, "oracle_module": ORACLE_MODULE,
                        "steps_per_case": STEPS,
                        "python": result.host["python"], "numpy": result.host["numpy"],
                        "machine": result.host["machine"], "system": result.host["system"]},
    }


#: The fields whose presence makes an entry bind bytes (``fastpath.BOUND_FIELDS``),
#: restated so this module imports nothing from the planner; the oracle's own test
#: asserts the two are equal.
BOUND_FIELDS: Tuple[str, ...] = ("source_sha256", "source_sha256_at_recert", "probe",
                                 "probe_sha256", "adapter", "adapter_sha256")
DISPATCH_KEY = "driver_dispatch"


def binds_bytes(entry: Any) -> bool:
    """Does this ledger entry bind source bytes (the mechanical weld rule)?"""
    return isinstance(entry, Mapping) and any(field in entry for field in BOUND_FIELDS)


def ledger_problems(ledger: Mapping[str, Any], result: OracleRun, table: str) -> List[str]:
    """Why ``ledger`` does not record the array path ``result`` describes. Empty means it does.

    Every byte-binding entry carries :func:`entry_leaf`'s values; no other entry carries
    a leaf; ``driver_dispatch`` carries the per-case digests and form-class counts of the
    same run. A moved fingerprint is reported with the cases whose digest moved.
    """
    problems: List[str] = []
    binding = sorted(key for key, entry in ledger.items() if binds_bytes(entry))
    expected = entry_leaf(result)
    missing, stale = [], []
    for key in binding:
        leaf = ledger[key].get(LEAF_KEY)
        if not isinstance(leaf, Mapping):
            missing.append(key)
        elif {field: leaf.get(field) for field in expected} != expected:
            stale.append(key)
    stray = sorted(key for key, entry in ledger.items()
                   if isinstance(entry, Mapping) and LEAF_KEY in entry and key not in binding)
    if missing:
        problems.append(f"{table}: {len(missing)} of {len(binding)} byte-binding entries carry "
                        f"no {LEAF_KEY} leaf: {missing[:12]}")
    if stale:
        problems.append(f"{table}: {len(stale)} of {len(binding)} byte-binding entries record "
                        f"another array path than this tree's ({result.fixture_set} "
                        f"{result.fingerprint_sha256[:16]}): {stale[:12]}")
    if stray:
        problems.append(f"{table}: entries that bind no bytes carry a {LEAF_KEY} leaf: {stray}")
    dispatch = ledger.get(DISPATCH_KEY)
    leaf = dispatch.get(LEAF_KEY) if isinstance(dispatch, Mapping) else None
    if not isinstance(leaf, Mapping):
        problems.append(f"{table}: {DISPATCH_KEY} carries no {LEAF_KEY} leaf")
        return problems
    recorded_cases = leaf.get("case_sha256")
    if not isinstance(recorded_cases, Mapping):
        problems.append(f"{table}: {DISPATCH_KEY}.{LEAF_KEY} records no case_sha256")
    else:
        moved = sorted(name for name in set(recorded_cases) | set(result.case_sha256)
                       if recorded_cases.get(name) != result.case_sha256.get(name))
        if moved:
            problems.append(
                f"{table}: the array path's behaviour moved since {LEAF_KEY} was minted: "
                f"{len(moved)} of {len(result.case_sha256)} fixture cases step to other words "
                f"({moved}); fingerprint recorded "
                f"{str(leaf.get('fingerprint_sha256'))[:16]} live "
                f"{result.fingerprint_sha256[:16]}. Every weld of this table was certified "
                f"against the recorded behaviour")
    if leaf.get("fixture_set") != result.fixture_set or \
            leaf.get("fingerprint_sha256") != result.fingerprint_sha256:
        if not any("moved since" in problem for problem in problems):
            problems.append(f"{table}: {DISPATCH_KEY}.{LEAF_KEY} records "
                            f"{leaf.get('fixture_set')} {str(leaf.get('fingerprint_sha256'))[:16]}, "
                            f"this tree steps {result.fixture_set} "
                            f"{result.fingerprint_sha256[:16]}")
    if leaf.get("form_classes") != result.form_classes:
        problems.append(f"{table}: {DISPATCH_KEY}.{LEAF_KEY}.form_classes differ from the "
                        f"counts this tree's fixture reaches")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - a diagnostic entry
    """Print the live run as JSON: ``python -m meep_gpu.array_path_oracle``."""
    del argv
    result = run(progress=lambda line: print(line, flush=True))
    print(json.dumps(result._asdict(), indent=2, sort_keys=True))
    return 0 if not coverage_problems(result.form_classes) else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
