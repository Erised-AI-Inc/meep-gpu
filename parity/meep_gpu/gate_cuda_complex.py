"""BIT-IDENTITY GATE for the hand-CUDA COMPLEX-storage PML family.

WHAT IS UNDER TEST. The four kernels ``complex_emitter`` emits --
``step_B_pml_complex_bloch``, ``step_D_pml_complex_bloch``,
``update_H_pml_complex_bloch``, ``update_E_pml_complex_bloch`` --
against ``stepping.step_B`` / ``step_D`` / ``update_H`` / ``update_E`` on complex64
storage under a split-field PML, compared as WORDS over every volume the sub-step
writes.

THE STRUCTURAL ZERO THIS CLOSES. The 2026-08-16 predicate census measured
``complex64 storage: the recurrence is the same but the storage is not`` as the
FIRST refusal on 47 of the 186 corpus rows at ``update_H`` and 47 at ``update_E``,
and the curl predicate refused the same class. That was a correct refusal about the
REAL kernels and an ABSENCE of a complex one -- the same shape ``update_P`` had
before ``ade_kernels``.

=============================================================================
THE ARM, AND WHY THIS GATE HAS A LEG NO SIBLING HAS
=============================================================================

Every real-coefficient multiply on the complex array path is a FULL complex
multiply with a zero-imaginary operand, because ``np.multiply`` carries only
'FF->F' for complex. Whether the platform fuses that multiply (``FMA_V1``) or
rounds every product separately (``NAIVE``) is a MEASURED PLATFORM FACT carried in
a probe artifact, and the arbiter is
``triton_kernels.complex_fields.expansion_license``. This gate:

* READS the artifact named by ``--expansion-probe`` (or
  ``MEEP_GPU_COMPLEX_EXPANSION_PROBE``), runs the licence, and REFUSES to release
  without an arm -- there is no default anywhere in this family;
* checks the licence against the policy this run installs
  (``expansion_policy_reasons``), because a licence cut under one float32
  subnormal policy reproduces the platform's bytes under that policy only;
* carries the OTHER arm as a mutation, and reports where the two arms are
  measurably apart and where they coincide. THE COINCIDENCE IS A FINDING, not a
  failure: on the zero-imaginary orientations the arms differ only where the
  product is exactly zero or underflows, so a leg that reports "uncaught" there is
  reporting that those operand classes cannot tell the arms apart -- and the
  number of words they ARE apart on the classes that can is what makes the
  measurement mean something.

=============================================================================
TWO BACKENDS, AND WHAT EACH ONE LICENSES
=============================================================================

``--backend cupy`` is the CERTIFYING backend: NVRTC compiles the emitted source
and the GPU runs it. Nothing else can certify, because the questions this family
turns on -- does ``z.im * 0.0f`` survive constant folding, does ``__fmaf_rn`` lower
to one ``fma.rn.f32``, does ``--fmad=false`` remove the accidental contractions --
are questions about a compiler.

``--backend host`` compiles THE SAME SOURCE TEXT with the host C++ compiler
(``clang++``/``g++``), shimming ``__device__``, ``__global__`` and ``__fmaf_rn``
and driving it with a block/thread loop, then runs it against the NumPy array path.
It is a LAPTOP leg and it certifies nothing about NVRTC. What it does buy is real
and was worth building: it executes the ACTUAL emitted characters rather than a
re-transcription, so an index, a stencil, a mask, an operand order or a
coefficient sub-lattice that is wrong is caught at the merge bar in seconds
instead of in a device slot. Every sibling gate's laptop leg is a hand-written
evaluator of a grammar the device source has to stay inside; this one has no
grammar to stay inside and no second transcription to drift.

``--backend numpy`` runs the harness legs with no kernel at all.

=============================================================================
WHAT A RELEASE REQUIRES
=============================================================================

* every valid single-launch case bit-identical under ``--fmad=false``;
* every valid case bit-identical at the multi-step budget, which is what sees the
  auxiliary recurrence rather than one launch of it;
* the GUARD CLAUSE, satisfied by EITHER of two routes and the artifact says which.
  Every certified sibling required the output control (compiled with no options) to
  DIVERGE, on the reasoning that a flag whose removal changes nothing is
  decoration. That reasoning has a second branch, and this family lands in it: the
  flag can also change nothing because the source gives it nothing to do. Measured
  on an RTX A6000 -- under the licensed ``FMA_V1`` arm all four cubins are
  BIT-IDENTICAL with and without ``--fmad=false``, and under ``NAIVE`` all four
  DIFFER. Every product under FMA_V1 is either an explicit ``__fmaf_rn`` -- already
  one ``fma.rn.f32``, which no contraction flag reaches -- or a multiply by a
  literal whose result feeds another multiply or an fma addend, and contraction
  needs a ``mul`` feeding an ``add``. So the clause passes on identical images
  PLUS the other arm differing, which is what keeps "inert" apart from "unmeasured";
  an identical control with differing images would be a gate that cannot see and
  fails;
* every mutation leg as required, every leg disarmed afterwards, and no leg
  unmeasurable on this backend;
* the non-vacuity floors: the oracle moved words in every case, the subnormal band
  actually contained subnormals, every phased case actually rotated a plane, and
  every case was ADMITTED BY THE PREDICATE (a gate that certifies a configuration
  its own predicate refuses has measured the wrong thing).

Usage, on the validation host (from ``parity/meep_gpu``)::

    PYTHONPATH=../.. CUDA_VISIBLE_DEVICES=<one empty index> \\
    CUPY_CACHE_DIR=<fresh, policy-token-carrying> \\
    python -u gate_cuda_complex.py --backend cupy --product full \\
        --subnormal-policy keep --expansion-probe <probe>.json \\
        --out <fresh dir>/gate.json

and on a laptop::

    PYTHONPATH=../.. python -u gate_cuda_complex.py --backend host \\
        --product reduced --out /tmp/complex_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten after
every case, so an interrupted run keeps everything up to the failure. Correctness
only: no throughput or timing claim is made or possible.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: the host-compiled and harness legs still run
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import complex_emitter, coverage  # noqa: E402
from meep_gpu.triton_kernels import complex_fields  # noqa: E402

log = probe.log
save = probe.save
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

SEED = 20260819

#: Consecutive launches in the multi-step leg. 60 is the budget the four certified
#: hand-CUDA records are cut at, kept so every family's claim stays commensurable.
#: "Identical for N steps" is a claim about N, and for the CURL it is the claim
#: that carries the split-field auxiliary: at one launch ``fu`` is read once, and a
#: defect that mis-writes it shows only when the next launch reads it back.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the recurrence driven. Held fixed, the source stops
#: changing and a 60-launch leg becomes a slow single-launch leg. Applied to the
#: SAME bytes on both fixtures so it cancels out of the comparison; not exactly
#: representable in float32, deliberately -- an exact scale would leave the
#: mantissas untouched.
_ADVANCE = np.complex64(complex(0.97, -0.03))

#: (cell extents in cells, label). The MIDDLE one is the point: 13*17*11 = 2431
#: complex cells is not a multiple of the 256-lane block, so the tail guard
#: ``if (idx >= nx*ny*nz) return;`` decides real threads there. The THIRD carries an
#: INVARIANT axis, where ``xp.roll`` returns the cell itself and every lane is a
#: wrap lane -- so a phase on a collapsed axis is applied to the whole volume, and
#: that is the array path's behaviour rather than an edge case to be avoided.
SHAPES: Tuple[Tuple[Tuple[float, float, float], str], ...] = (
    ((8.0, 8.0, 8.0), "8x8x8"),
    ((13.0, 17.0, 11.0), "13x17x11"),
    ((1.0, 16.0, 16.0), "1x16x16"),
)

#: The two ghost rules this family serves, in every mixture. A metallic axis
#: exercises BOTH the predicated-zero ghost and the ownership mask; a periodic axis
#: exercises the wrap, and only a periodic axis can carry a phase.
BOUNDARY_TRIPLES: Tuple[Tuple[str, str, str], ...] = (
    ("periodic", "periodic", "periodic"),
    ("metallic", "periodic", "periodic"),
    ("periodic", "metallic", "periodic"),
    ("periodic", "periodic", "metallic"),
    ("metallic", "metallic", "metallic"),
)

#: The k_point axis. ``(0,0,0)`` is the COMPLEX-STORAGE-AT-K=0 class -- wvg-src.py
#: and solve-cw.py -- where every phase flag is 0 and the kernel must reduce to the
#: plain complex path; it is not a degenerate case to be tolerated but one of the
#: two things this family is for. The Brillouin edge is carried because
#: ``Grid._axis_bloch_phase`` returns EXACTLY -1+0j there (grid.py:1011-1017),
#: which is a different word pair from ``exp(i*pi)``'s rounding and reaches the
#: rotation's zero cross terms.
K_POINTS: Tuple[Tuple[Tuple[float, float, float], str], ...] = (
    ((0.0, 0.0, 0.0), "k0"),
    ((0.3, 0.0, 0.0), "kx"),
    ((0.0, 0.17, -0.11), "kyz"),
    ((0.5, 0.0, 0.0), "kedge"),
)

#: 0.5 is exactly representable and 0.35 is not; only the second can distinguish a
#: contracted expression from an uncontracted one, so the guard control is scored
#: at the inexact one.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

#: ``uniform`` is the physical band. ``subnormal_band`` is what a float32 subnormal
#: policy decides the fate of, and it is the ONE class measured to separate the two
#: expansion arms on a zero-imaginary orientation (the E-side constitutive).
#: ``zero_init`` is the class the zero cross terms exist for: from all-zero state
#: the array path stores ``-0.0`` words wherever a thin absorber's deepest
#: ``kms = kappa - sigma`` goes negative, which a kernel that folded
#: ``z.im * 0.0f`` to a literal cannot reproduce.
#: ``signed_zero`` is the class ENGINEERED for the zero cross terms, and it is a
#: real operand class rather than a contrivance: a complex word whose REAL plane is
#: an exact zero and whose imaginary plane is not is what a flushed subnormal, a
#: freshly-initialised volume and a wall plane all produce, and it is the only class
#: on which ``z.im * 0.0f`` can change a stored word. ``fma(z.re, c, -0.0f)``
#: differs from ``fma(z.re, c, +0.0f)`` exactly when ``z.re * c`` is ``-0.0f``, so
#: the class needs signed zeros on one plane and live values on the other.
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "zero_init",
                                  "signed_zero")

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

CURL_SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")
CONSTITUTIVE_SIDES: Tuple[str, ...] = ("H", "E")
SUB_STEPS: Tuple[str, ...] = CURL_SUB_STEPS + ("update_H", "update_E")

#: Which volumes each sub-step WRITES -- the words a case compares -- and which it
#: reads. Transcribed from ``stepping.B_CURL_TERMS`` / ``D_CURL_TERMS`` (:213-223)
#: and ``H_``/``E_CONSTITUTIVE_TERMS`` (:226-227).
OUTPUTS: Dict[str, Tuple[str, ...]] = {
    "step_B": ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz"),
    "step_D": ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"),
    "update_H": ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz"),
    "update_E": ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"),
}

#: What a case ADVANCES between launches in the multi-step leg -- the sub-step's
#: sources, which nothing in the sub-step writes.
DRIVERS: Dict[str, Tuple[str, ...]] = {
    "step_B": ("Ex", "Ey", "Ez"),
    "step_D": ("Hx", "Hy", "Hz"),
    "update_H": ("Bx", "By", "Bz"),
    "update_E": ("Dx", "Dy", "Dz"),
}

#: Every complex volume the fixture seeds and snapshots. One list for all four
#: sub-steps so a case can be built once and re-frozen for any of them.
STATE: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and that
    is the one thing about the device library a laptop cannot supply. Everything
    else it asks -- dtype, shape, contiguity, the PML vectors, the grid's boundary
    resolution, the phase table -- is a real object either way.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def config_viable(cell: Sequence[float], boundaries: Sequence[str],
                  k_point: Sequence[float]) -> Optional[str]:
    """Why this (shape, boundaries, k) triple cannot be built, or None.

    ``Grid`` REFUSES a metallic axis holding exactly one cell: MEEP overrides a
    unit direction to periodic whatever the k_point says, and a conductor on both
    faces of one cell zeroes the whole family. ``stepping._bloch_phases`` REFUSES a
    nonzero k on a non-periodic axis, and so does the predicate. Filtering here
    keeps both refusals out of the case list instead of turning them into leg
    failures that say nothing about the kernel.
    """
    for axis, extent in enumerate(cell):
        if extent <= 1.0 and boundaries[axis] == "metallic":
            return (f"axis {axis} holds one cell and cannot be metallic; Grid "
                    f"refuses it and MEEP has no such run")
        if boundaries[axis] != "periodic" and float(k_point[axis]) != 0.0:
            return (f"axis {axis} is {boundaries[axis]} and carries k="
                    f"{k_point[axis]}; a PEC wall gives the axis no lattice vector "
                    f"for the phase and stepping._bloch_phases raises")
    return None


def build(xp, cell, boundaries, k_point, courant: float, needs_epsilon: bool,
          rng):
    """A frozen ``(fields, layer, grid)`` triple, complex storage, PML allocated.

    ``force_complex_fields=True`` ALWAYS, even at k = 0: that is the
    complex-storage-at-k=0 class (wvg-src.py, solve-cw.py) and it is half of what
    this family exists for. A nonzero k_point would force complex storage by
    itself; forcing it explicitly is what makes the k = 0 rows real rows rather
    than an unreachable branch.

    The thickness rule is the one every certified slice's fixture uses: skip an
    axis too thin to hold a layer. A mirrored axis cannot occur -- the predicate
    refuses folds -- so the low-face special case those fixtures carry is
    deliberately absent rather than copied.
    """
    grid = Grid(resolution=1.0, cell_size=tuple(cell), boundaries=tuple(boundaries),
                xp=xp, courant=courant, k_point=tuple(k_point))
    thickness = tuple((0, 0) if grid.shape[axis] < 6 else (2, 2)
                      for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    if needs_epsilon:
        # THREE DISTINCT VOLUMES, never one. Binding a single volume for all three
        # components is the defect ``update_E_pml_complex`` carries through
        # ``fields.inv_eps`` (fields.py:1259-1260), and a fixture that handed one
        # array to all three could not see it.
        epsilon, inverse = {}, {}
        for component in ("Ex", "Ey", "Ez"):
            values = rng.uniform(1.2, 3.4, size=grid.shape).astype(np.float32)
            epsilon[component] = xp.asarray(values)
            inverse[component] = xp.asarray(
                (np.float32(1.0) / values).astype(np.float32))
        fields.set_epsilon_volumes(epsilon, inverse)
    return fields, layer, grid


def seed_state(fields, grid, value_class: str, rng) -> Dict[str, np.ndarray]:
    """Seed every complex volume the sub-steps touch, and return the host words.

    The AUXILIARIES start nonzero in the two live classes. A zero ``fu``/``f_w``
    makes ``kms * prev`` exactly zero on the first launch whatever ``kms`` holds,
    so a mis-indexed coefficient would only show from launch two.

    ``zero_init`` is the exception and is the whole point of that class: from
    all-zero state the ONLY thing that can move a word is a signed zero, and the
    array path does move them -- ``0.0 * kms`` with ``kms`` negative is ``-0.0``,
    which the zero cross terms carry and a folded literal does not.
    """
    xp = grid.xp
    shape = tuple(grid.shape)
    if value_class == "uniform":
        planes = {name: (rng.uniform(-1.0, 1.0, size=shape),
                         rng.uniform(-1.0, 1.0, size=shape)) for name in STATE}
    elif value_class == "subnormal_band":
        real = subnormal_band_hosts(STATE, shape, rng)
        imag = subnormal_band_hosts(STATE, shape, rng)
        planes = {name: (real[name], imag[name]) for name in STATE}
    elif value_class == "zero_init":
        zeros = np.zeros(shape, dtype=np.float32)
        planes = {name: (zeros, zeros) for name in STATE}
    elif value_class == "signed_zero":
        planes = {}
        for name in STATE:
            values = rng.uniform(-1.0, 1.0, size=shape)
            # Half the words get an exact zero in the REAL plane, with the SIGN
            # drawn independently: +0.0 and -0.0 are different words and the cross
            # term is what carries the difference between them through.
            zeroed = rng.integers(0, 2, size=shape) == 0
            signs = np.where(rng.integers(0, 2, size=shape) == 0, -0.0, 0.0)
            planes[name] = (np.where(zeroed, signs, values),
                            rng.uniform(-1.0, 1.0, size=shape))
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    # THE TWO PLANES ARE ASSIGNED, NEVER COMBINED AS ``re + 1j*im``. Measured here
    # before it was reasoned about: ``1j * im`` carries a real part of ``0.0 * im``,
    # so ``re + 1j*im`` computes ``(-0.0) + (+0.0) = +0.0`` and DESTROYS every
    # negative zero in the real plane. A signed-zero fixture built that way seeds no
    # negative zeros at all in the plane the zero cross terms act on, and the legs
    # that exist to catch a folded cross term then report "uncaught" for a reason
    # that belongs to the fixture.
    host: Dict[str, np.ndarray] = {}
    for name, (real_plane, imag_plane) in planes.items():
        values = np.empty(shape, dtype=np.complex64)
        values.real = real_plane.astype(np.float32)
        values.imag = imag_plane.astype(np.float32)
        host[name] = values
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def word_census(host: Dict[str, np.ndarray]) -> Dict[str, int]:
    """The operand census over the float32 WORDS of the complex volumes.

    ``operand_census`` counts subnormals, zeros and negative zeros in float32
    arrays; a complex volume's words are exactly that array, so the complex class's
    non-vacuity is measured on the same instrument the real families use.
    """
    return operand_census({name: values.view(np.float32).ravel()
                           for name, values in host.items()})


def snapshot(fields) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy() for name in STATE}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, sub_step: str) -> None:
    for name in DRIVERS[sub_step]:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


def words_differ(left: np.ndarray, right: np.ndarray) -> int:
    """Differing float32 WORDS between two complex volumes, compared as bits."""
    a = np.ascontiguousarray(left).view(np.float32).ravel().view(np.uint32)
    b = np.ascontiguousarray(right).view(np.float32).ravel().view(np.uint32)
    return int(np.count_nonzero(a != b))


# ---------------------------------------------------------------------------
# The expansion licence
# ---------------------------------------------------------------------------

def load_licence(path: Optional[str], policy: Optional[str]) -> Dict[str, Any]:
    """Read the probe artifact, run the licence, and report the whole verdict.

    THE ARBITER IS ``complex_fields.expansion_license`` AND IS NOT REIMPLEMENTED.
    What this adds is the two things a verdict cannot know: which artifact was
    read, and which policy the consuming run installs
    (``expansion_policy_reasons``).
    """
    out: Dict[str, Any] = {"path": path, "policy_required": policy,
                           "record_sha256": None, "verdict": None,
                           "policy_reasons": [], "certification_reasons": [],
                           "arm": None, "expansion": None, "usable": False}
    if not path:
        path = os.environ.get(complex_fields.PROBE_PATH_ENVIRONMENT)
        out["path"] = path
    if not path:
        out["policy_reasons"] = [
            "no expansion probe artifact was given and "
            f"{complex_fields.PROBE_PATH_ENVIRONMENT} is unset; the arm is a "
            "measured platform fact and may not be guessed"]
        return out
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        out["policy_reasons"] = [f"the probe artifact could not be read: {exc}"]
        return out
    out["record_sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        record = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        out["policy_reasons"] = [f"the probe artifact is not readable JSON: {exc}"]
        return out
    verdict = complex_fields.expansion_license(record)
    out["verdict"] = verdict
    out["policy_reasons"] = list(
        complex_fields.expansion_policy_reasons(record, policy))
    out["certification_reasons"] = list(
        complex_fields.expansion_certification_reasons(policy))
    out["arm"] = verdict.get("arm")
    out["expansion"] = verdict.get("expansion")
    out["basis"] = verdict.get("basis")
    out["usable"] = bool(verdict.get("arm") and not verdict.get("refusals")
                         and not out["policy_reasons"])
    return out


def other_arm(arm: str) -> str:
    """The arm this platform was NOT licensed for -- the mutation leg's arm."""
    return "NAIVE" if arm == "FMA_V1" else "FMA_V1"


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------

_SIGNATURE = re.compile(r'extern "C" __global__ void (\w+)\(([^)]*)\)', re.S)

#: What the host compiler needs in place of CUDA. ``__fmaf_rn`` maps to C99
#: ``fmaf``, which is a correctly-rounded single fused multiply-add on every host
#: this runs on -- the same contract the intrinsic has. Everything else is a
#: keyword the host compiler does not know.
_HOST_SHIM = r'''
#include <math.h>
#define __device__
#define __forceinline__ inline
#define __global__
#define __restrict__ __restrict
#define __fmaf_rn(a, b, c) fmaf((a), (b), (c))
struct meep_gpu_uint3 { int x, y, z; };
static struct meep_gpu_uint3 blockIdx, blockDim, threadIdx;
'''

#: ``-ffp-contract=off`` is the host compiler's ``--fmad=false``: without it the
#: host compiler fuses ``a*b + c`` on its own and the comparison stops being about
#: the transcription. ``-fno-fast-math`` keeps ``z.im * 0.0f`` alive for the same
#: reason NVRTC's default does.
_HOST_OPTIONS: Tuple[str, ...] = ("-O1", "-ffp-contract=off", "-fno-fast-math")


def parse_signature(source: str) -> Tuple[str, List[Tuple[str, str]]]:
    """The kernel's name and its (type, identifier) parameter list, from the text."""
    match = _SIGNATURE.search(source)
    if match is None:
        raise ValueError("no extern \"C\" __global__ kernel found in the source")
    parameters: List[Tuple[str, str]] = []
    for raw in match.group(2).split(","):
        cleaned = " ".join(raw.replace("__restrict__", "").split())
        kind, identifier = cleaned.rsplit(" ", 1)
        parameters.append((kind.strip(), identifier.lstrip("*")))
    return match.group(1), parameters


class KernelBackend:
    """NVRTC + the GPU. The only backend that can certify anything."""

    certifies = True
    name = "cupy"

    def __init__(self) -> None:
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        self.module = complex_pml_kernels
        self.options: Tuple[str, ...] = ("--fmad=false",)
        self._original_options = complex_pml_kernels._COMPILE_OPTIONS
        self._original_source = complex_emitter.complex_source

    def set_guard(self, options: Sequence[str]) -> None:
        self.options = tuple(options)
        self.module._COMPILE_OPTIONS = tuple(options)
        self.module._clear_kernel_cache()

    def set_source_transform(self, transform: Optional[Callable[[str], Tuple[str, int]]]
                             ) -> int:
        """Install a text mutation on the emitted source; returns the site count.

        Monkeypatching ``complex_emitter.complex_source`` rather than a string
        constant is what this family's emitter forces, and it is the stronger
        arrangement: the mutation is applied to the SAME text the shipped path
        emits, per sub-step and per arm, so a leg cannot mutate one specialization
        and score another.
        """
        self.module._clear_kernel_cache()
        if transform is None:
            complex_emitter.complex_source = self._original_source
            return 0
        sites = {"n": 0}
        original = self._original_source

        def mutated(sub_step, expansion):
            text, count = transform(original(sub_step, expansion))
            sites["n"] = max(sites["n"], count)
            return text

        complex_emitter.complex_source = mutated
        # Emit once per sub-step so the site count is known before any launch: a
        # mutation whose needle matched nothing exercises nothing and would
        # otherwise report a pass while doing it.
        for sub_step in complex_emitter.KERNELS:
            mutated(sub_step, "FMA_V1")
        return sites["n"]

    def launch_curl(self, sub_step, fields, layer, grid, expansion, *,
                    tables=None, boundary_codes=None, phase_flags=None,
                    phase_values=None, dtdx=None) -> None:
        if tables is None and boundary_codes is None:
            self.module.step_fused_pml_complex(sub_step, fields, expansion,
                                               grid=grid, pml=layer)
            return
        derived_flags, derived_values = self.module.bloch_phase_arguments(
            grid, complex_emitter.KERNELS[sub_step][1])
        self.module.step_fused_pml_complex(
            sub_step, fields, expansion,
            tables=(tables if tables is not None
                    else self.module.complex_curl_tables(
                        layer, complex_emitter.HALF_INTEGER[sub_step])),
            boundary_codes=(boundary_codes if boundary_codes is not None
                            else self.module.complex_boundary_codes(grid)),
            phase_flags=phase_flags if phase_flags is not None else derived_flags,
            phase_values=(phase_values if phase_values is not None
                          else derived_values),
            dtdx=dtdx if dtdx is not None else grid.dt / grid.dx)

    def launch_constitutive(self, side, fields, layer, grid, expansion, *,
                            tables=None) -> None:
        if tables is None:
            self.module.update_fused_pml_complex(side, fields, expansion, pml=layer)
        else:
            self.module.update_fused_pml_complex(side, fields, expansion,
                                                 tables=tables)

    def guard_effect(self, expansion) -> Dict[str, Any]:
        """What ``--fmad=false`` does to the COMPILED IMAGE, per kernel and arm.

        THE GUARD CONTROL IS AN OUTPUT COMPARISON AND THIS IS A BYTE ONE, and the
        pair is what makes an identical control mean something. A control that
        comes back identical is either a gate that cannot see or a source the flag
        has nothing to do to; two bit-identical NVRTC images settle it -- they
        cannot produce different words -- and two that differ while the outputs
        agree would be the weaker claim, which this reports rather than hides.

        Measured 2026-08-19 on an RTX A6000 by
        ``probe_cuda_complex_fmad.py``: under FMA_V1 all four cubins are IDENTICAL
        across the flag, and under NAIVE all four DIFFER. The transcription is why
        -- under FMA_V1 every product is either an explicit ``__fmaf_rn`` (already
        one ``fma.rn.f32``, which no contraction flag reaches) or a multiply by a
        literal whose result feeds another multiply or an fma addend, and
        contraction needs a ``mul`` feeding an ``add``. ``NAIVE``'s
        ``(z.re * c) - (z.im * 0.0f)`` is exactly that shape.
        """
        from cupy.cuda import compiler  # noqa: PLC0415

        device = cp.cuda.Device()
        arch = f"{device.compute_capability[0]}{device.compute_capability[1:]}"
        out: Dict[str, Any] = {"arch": arch, "rows": []}
        for sub_step in complex_emitter.KERNELS:
            for arm in ("FMA_V1", "NAIVE"):
                source = self._original_source(sub_step, arm)
                digests = {}
                for label, options in (("fmad_false", ("--fmad=false",)),
                                       ("default_no_options", ())):
                    image = compiler.compile_using_nvrtc(
                        source, options=options, arch=arch, filename="guard.cu")
                    if isinstance(image, tuple):
                        image = image[0]
                    if isinstance(image, str):
                        image = image.encode("utf-8")
                    image = bytes(image)
                    digests[label] = {
                        "sha256": hashlib.sha256(image).hexdigest(),
                        "bytes": len(image),
                        "kind": ("cubin" if image[:4] == b"\x7fELF" else "ptx")}
                out["rows"].append({
                    "sub_step": sub_step, "arm": arm, **digests,
                    "identical_across_guards":
                        digests["fmad_false"]["sha256"]
                        == digests["default_no_options"]["sha256"]})
        shipped = complex_emitter.EXPANSION_NAMES[
            complex_emitter.normalized_expansion(expansion)]
        live = [r for r in out["rows"] if r["arm"] == shipped]
        other = [r for r in out["rows"] if r["arm"] != shipped]
        out["shipped_arm"] = shipped
        out["shipped_arm_images_identical_across_guards"] = all(
            r["identical_across_guards"] for r in live)
        out["other_arm_images_identical_across_guards"] = all(
            r["identical_across_guards"] for r in other)
        out["kernels_measured"] = len(out["rows"])
        return out

    def restore(self) -> None:
        complex_emitter.complex_source = self._original_source
        self.module._COMPILE_OPTIONS = self._original_options
        self.module._clear_kernel_cache()


class HostCompiledBackend:
    """The emitted source, compiled by the HOST C++ compiler and driven on NumPy.

    CERTIFIES NOTHING about NVRTC and says so in the artifact. What it measures is
    the TRANSCRIPTION: the indices, the stencils, the masks, the operand orders,
    the coefficient sub-lattices and the phase table, executed from the actual
    emitted characters at the merge bar.
    """

    certifies = False
    name = "host"

    def __init__(self, compiler: str) -> None:
        self.compiler = compiler
        self.options: Tuple[str, ...] = _HOST_OPTIONS
        self._transform: Optional[Callable[[str], Tuple[str, int]]] = None
        self._cache: Dict[str, Any] = {}
        self._workdir = tempfile.mkdtemp(prefix="gate_cuda_complex_")

    def set_guard(self, options: Sequence[str]) -> None:
        # The NVRTC guard names have no host equivalent; the host leg always
        # compiles with contraction off, and the guard CONTROL is a device-only
        # measurement. Recorded rather than silently ignored.
        self.options = _HOST_OPTIONS
        self._cache.clear()

    def set_source_transform(self, transform) -> int:
        self._transform = transform
        self._cache.clear()
        if transform is None:
            return 0
        sites = 0
        for sub_step in complex_emitter.KERNELS:
            _, count = transform(complex_emitter.complex_source(sub_step, "FMA_V1"))
            sites = max(sites, count)
        return sites

    def _entry(self, sub_step: str, expansion):
        arm = complex_emitter.normalized_expansion(expansion)
        key = f"{sub_step}|{arm}"
        if key in self._cache:
            return self._cache[key]
        source = complex_emitter.complex_source(sub_step, arm)
        if self._transform is not None:
            source, _ = self._transform(source)
        name, parameters = parse_signature(source)
        declaration = ", ".join(f"{kind} {identifier}"
                                for kind, identifier in parameters)
        call = ", ".join(identifier for _, identifier in parameters)
        driver = (f'\nextern "C" void launch_{name}(int nblocks, int nthreads, '
                  f'{declaration}) {{\n'
                  f'    blockDim.x = nthreads;\n'
                  f'    for (int b = 0; b < nblocks; ++b) {{\n'
                  f'        blockIdx.x = b;\n'
                  f'        for (int t = 0; t < nthreads; ++t) {{\n'
                  f'            threadIdx.x = t;\n'
                  f'            {name}({call});\n'
                  f'        }}\n    }}\n}}\n')
        stem = os.path.join(self._workdir,
                            hashlib.sha256((source + driver).encode()).hexdigest()[:16])
        with open(stem + ".cpp", "w", encoding="ascii") as handle:
            handle.write(_HOST_SHIM + source + driver)
        subprocess.run([self.compiler, *self.options, "-shared", "-fPIC",
                        "-o", stem + ".so", stem + ".cpp"], check=True,
                       capture_output=True)
        library = ctypes.CDLL(stem + ".so")
        function = getattr(library, f"launch_{name}")
        argtypes: List[Any] = [ctypes.c_int, ctypes.c_int]
        for kind, _ in parameters:
            if "*" in kind:
                argtypes.append(ctypes.c_void_p)
            elif kind == "float":
                argtypes.append(ctypes.c_float)
            elif kind == "int":
                argtypes.append(ctypes.c_int)
            else:
                raise ValueError(f"unhandled parameter type {kind!r}")
        function.argtypes = argtypes
        function.restype = None
        self._cache[key] = (function, parameters)
        return self._cache[key]

    @staticmethod
    def _pointer(array):
        return np.ascontiguousarray(array).ctypes.data_as(ctypes.c_void_p)

    def _words(self, array):
        if not array.flags.c_contiguous:
            raise ValueError("the host leg needs C-contiguous complex volumes")
        return array.view(np.float32).ctypes.data_as(ctypes.c_void_p)

    def launch_curl(self, sub_step, fields, layer, grid, expansion, *,
                    tables=None, boundary_codes=None, phase_flags=None,
                    phase_values=None, dtdx=None) -> None:
        function, _ = self._entry(sub_step, expansion)
        backward = complex_emitter.KERNELS[sub_step][1]
        suffix = "_h" if complex_emitter.HALF_INTEGER[sub_step] else ""
        if tables is None:
            tables = {f"{stem}_{axis}": np.ascontiguousarray(
                getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1))
                for axis in "xyz" for stem in ("kms", "sinv")}
        if boundary_codes is None:
            boundary_codes = tuple(
                coverage.BC_CODES[kind]
                for kind in coverage.real_pml_boundary_kinds(grid))
        if phase_flags is None or phase_values is None:
            flags, values = host_phase_arguments(grid, backward)
            phase_flags = phase_flags if phase_flags is not None else flags
            phase_values = phase_values if phase_values is not None else values
        if dtdx is None:
            dtdx = grid.dt / grid.dx
        targets = ("Bx", "By", "Bz") if sub_step == "step_B" else ("Dx", "Dy", "Dz")
        sources = ("Ex", "Ey", "Ez") if sub_step == "step_B" else ("Hx", "Hy", "Hz")
        nx, ny, nz = grid.shape
        arguments: List[Any] = [self._words(getattr(fields, n)) for n in targets]
        arguments += [self._words(getattr(fields, "fu_" + n)) for n in targets]
        arguments += [self._words(getattr(fields, n)) for n in sources]
        arguments += [int(nx), int(ny), int(nz), ctypes.c_float(float(dtdx))]
        for axis in "xyz":
            arguments += [self._pointer(tables[f"kms_{axis}"]),
                          self._pointer(tables[f"sinv_{axis}"])]
        arguments += [int(code) for code in boundary_codes]
        arguments += [int(flag) for flag in phase_flags]
        arguments += [ctypes.c_float(float(value)) for value in phase_values]
        cells = int(nx) * int(ny) * int(nz)
        function((cells + 255) // 256, 256, *arguments)

    def launch_constitutive(self, side, fields, layer, grid, expansion, *,
                            tables=None) -> None:
        sub_step = "update_H" if side == "H" else "update_E"
        function, _ = self._entry(sub_step, expansion)
        suffix = "_h" if complex_emitter.HALF_INTEGER[side] else ""
        if tables is None:
            tables = {f"{stem}_{axis}": np.ascontiguousarray(
                getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1))
                for axis in "xyz" for stem in ("kps", "kms")}
        targets = ("Hx", "Hy", "Hz") if side == "H" else ("Ex", "Ey", "Ez")
        sources = ("Bx", "By", "Bz") if side == "H" else ("Dx", "Dy", "Dz")
        nx, ny, nz = grid.shape
        arguments: List[Any] = [self._words(getattr(fields, n)) for n in targets]
        arguments += [self._words(getattr(fields, "f_w_" + n)) for n in targets]
        arguments += [self._words(getattr(fields, n)) for n in sources]
        if side == "E":
            arguments += [self._pointer(fields.inverse_epsilon_for(n))
                          for n in targets]
        arguments += [int(nx), int(ny), int(nz)]
        for axis in "xyz":
            arguments += [self._pointer(tables[f"kps_{axis}"]),
                          self._pointer(tables[f"kms_{axis}"])]
        cells = int(nx) * int(ny) * int(nz)
        function((cells + 255) // 256, 256, *arguments)

    def restore(self) -> None:
        self._transform = None
        self._cache.clear()
        shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = tempfile.mkdtemp(prefix="gate_cuda_complex_")


class HarnessBackend:
    """No kernel at all. Exercises the fixture, the predicate and the arming."""

    certifies = False
    name = "numpy"

    def set_guard(self, options: Sequence[str]) -> None:
        return None

    def set_source_transform(self, transform) -> int:
        if transform is None:
            return 0
        sites = 0
        for sub_step in complex_emitter.KERNELS:
            _, count = transform(complex_emitter.complex_source(sub_step, "FMA_V1"))
            sites = max(sites, count)
        return sites

    def launch_curl(self, *args, **kwargs) -> None:
        raise RuntimeError("the numpy backend launches no kernel")

    def launch_constitutive(self, *args, **kwargs) -> None:
        raise RuntimeError("the numpy backend launches no kernel")

    def restore(self) -> None:
        return None


def host_phase_arguments(grid, backward: bool):
    """The phase encoding, for the backend that cannot call the CuPy launcher.

    IT CALLS THE SHIPPED FUNCTION when CuPy is importable and reproduces it
    otherwise; the reproduction is four lines and is checked against the shipped
    one by ``test_complex_pml.py``, because a gate that encoded the phase its own
    way would be measuring its own encoder.
    """
    kinds = coverage.real_pml_boundary_kinds(grid)
    flags: List[int] = []
    values: List[float] = []
    for axis in range(3):
        phase = grid.bloch_phase(axis) if getattr(grid, "has_bloch", False) else None
        if phase is None:
            flags.append(0)
            values.extend((1.0, 0.0))
            continue
        if kinds[axis] != "periodic":
            raise ValueError(
                f"axis {axis} carries Bloch phase {phase!r} but resolved to "
                f"{kinds[axis]!r}")
        rounded = np.complex64(phase)
        imaginary = float(np.float32(rounded.imag))
        flags.append(1)
        values.extend((float(np.float32(rounded.real)),
                       -imaginary if backward else imaginary))
    return tuple(flags), tuple(values)


def build_backend(name: str):
    if name == "cupy":
        if cp is None:
            raise SystemExit("--backend cupy needs CuPy and a device")
        return KernelBackend()
    if name == "host":
        for candidate in ("clang++", "g++", "c++"):
            found = shutil.which(candidate)
            if found:
                return HostCompiledBackend(found)
        raise SystemExit("--backend host needs a host C++ compiler on PATH")
    if name == "numpy":
        return HarnessBackend()
    raise SystemExit(f"unknown backend {name!r}")


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def _sub(pattern: str, replacement: str, text: str) -> Tuple[str, int]:
    out, count = re.subn(pattern, replacement, text)
    return out, count


def m_fold_the_zero_cross_terms(text: str) -> Tuple[str, int]:
    """Replace every zero cross term with a literal ``0.0f``.

    THE MUTATION THIS FAMILY EXISTS TO BE PROOF AGAINST, and the one whose
    non-arming would be a compiler doing it for us. ``z.im * 0.0f`` carries the
    SIGN of the other plane into an addend that is exactly zero; a literal does
    not, and ``fma(x, y, -0.0f)`` differs from ``fma(x, y, +0.0f)`` wherever
    ``x*y`` is exactly zero. If this leg comes out UNCAUGHT on a class that
    reaches a zero product, the pipeline folded the term itself and the shipped
    kernel is not the transcription it claims to be.
    """
    out, a = _sub(r"\(z\.im \* 0\.0f\) \* -1\.0f", "0.0f", text)
    out, b = _sub(r"z\.re, 0\.0f, z\.im \* c", "z.re, 0.0f, z.im * c", out)
    out, c = _sub(r"z\.im \* 0\.0f", "0.0f", out)
    out, d = _sub(r"z\.re \* 0\.0f", "0.0f", out)
    out, e = _sub(r"\(0\.0f \* z\.im\) \* -1\.0f", "0.0f", out)
    out, f = _sub(r"0\.0f \* z\.im", "0.0f", out)
    out, g = _sub(r"0\.0f \* z\.re", "0.0f", out)
    return out, a + c + d + e + f + g


def m_plane_wise_scaling(text: str) -> Tuple[str, int]:
    """Scale the two planes independently -- the obvious wrong transcription.

    ``{re*c, im*c}`` is what anyone writes who has not read that ``np.multiply``
    carries no scalar-times-complex loop. It is right on every value that is not a
    signed zero and wrong on the ones that are.
    """
    out, a = _sub(r"o\.re = __fmaf_rn\(z\.re, c, [^;]*;", "o.re = z.re * c;", text)
    out, b = _sub(r"o\.im = __fmaf_rn\(z\.re, 0\.0f, z\.im \* c\);",
                  "o.im = z.im * c;", out)
    out, c = _sub(r"o\.re = \(z\.re \* c\) - \(z\.im \* 0\.0f\);",
                  "o.re = z.re * c;", out)
    out, d = _sub(r"o\.im = \(z\.re \* 0\.0f\) \+ \(z\.im \* c\);",
                  "o.im = z.im * c;", out)
    # AND THE COEFFICIENT-LEFT ORIENTATION. ``update_H`` calls no field-left
    # multiply at all, so a mutation that rewrote only ``mul_field_left`` would
    # score that sub-step against an UNMUTATED source and report it uncaught --
    # measured, and the reason this half exists.
    out, e = _sub(r"o\.re = __fmaf_rn\(c, z\.re, [^;]*;", "o.re = c * z.re;", out)
    out, f = _sub(r"o\.im = __fmaf_rn\(c, z\.im, 0\.0f \* z\.re\);",
                  "o.im = c * z.im;", out)
    out, g = _sub(r"o\.re = \(c \* z\.re\) - \(0\.0f \* z\.im\);",
                  "o.re = c * z.re;", out)
    out, h = _sub(r"o\.im = \(c \* z\.im\) \+ \(0\.0f \* z\.re\);",
                  "o.im = c * z.im;", out)
    return out, a + b + c + d + e + f + g + h


def m_left_to_right_curl(text: str) -> Tuple[str, int]:
    """Flatten the curl's grouping to what C associates from an unparenthesised sum.

    ``((sf - f_1) + f_2) - ss`` instead of ``((sf - f_1) + (f_2 - ss))``. Float32
    addition is not associative and ``stepping._curl_from_operands`` (:1601-1636)
    fixes the grouping.
    """
    return _sub(r"cf_add\(cf_sub\(sf, f_1\), cf_sub\(f_2, ss\)\)",
                "cf_sub(cf_add(cf_sub(sf, f_1), f_2), ss)", text)


def m_flatten_the_constitutive_tail(text: str) -> Tuple[str, int]:
    """``f + (kps*src - kms*prev)`` instead of two separate accumulations.

    This is precisely what the twelve uncertified complex kernels in
    ``step_curl_kernels`` write, and it is a different float32 number from the
    array path's two ``+=``.
    """
    return _sub(
        r"    a = cf_add\(a, mul_coefficient_left\(kps, src\)\);\n"
        r"    a = cf_sub\(a, mul_coefficient_left\(kms, prev\)\);",
        "    a = cf_add(a, cf_sub(mul_coefficient_left(kps, src),\n"
        "                         mul_coefficient_left(kms, prev)));", text)


def m_store_fw_before_reading_prev(text: str) -> Tuple[str, int]:
    """Write ``fw`` before reading ``prev`` -- a slightly weaker absorber.

    Wrong only where ``kms != 0``, i.e. INSIDE THE PML ONLY, which is why it reads
    as physics rather than as a bug.
    """
    return _sub(r"    cf prev = cf_load\(fw, idx\);\n    cf_store\(fw, idx, src\);",
                "    cf_store(fw, idx, src);\n    cf prev = cf_load(fw, idx);", text)


def m_drop_the_ownership_mask(text: str) -> Tuple[str, int]:
    """Delete every ``curl = cf_zero()`` guard.

    ``stepping._mask_non_owned_cells`` (:1865-1902) zeroes the curl at cell 0 of
    every metallic axis on which the target's Yee shift is 0. Without it the
    auxiliary at the wall integrates a curl assembled from a ghost the cell does
    not own -- indefinitely.
    """
    return _sub(r" *if \(bc_[xyz] == BC_METALLIC && [ijk] == 0\) curl = cf_zero\(\);\n",
                "", text)


def m_metallic_ghost_wraps(text: str) -> Tuple[str, int]:
    """Serve the periodic wrap past a PEC wall instead of the zero ghost.

    The defect the twelve uncertified kernels carry in the other direction (they
    clamp where MEEP wraps); here it is planted deliberately so the gate shows the
    ghost rule is measured and not assumed.
    """
    return _sub(r"    if \(bc == BC_METALLIC\) return cf_zero\(\);\n", "", text)


def m_phase_on_every_lane(text: str) -> Tuple[str, int]:
    """Rotate every loaded neighbour, not only the wrapped plane.

    ``_apply_bloch_phase`` multiplies ONE plane of the rolled buffer
    (stepping.py:1893-1909). Rotating the interior is a smooth, plausible,
    completely wrong dispersion relation.
    """
    out, a = _sub(r"    if \(ia \+ 1 < na\) return cf_load\(g, idx \+ stride\);",
                  "    if (ia + 1 < na) { cf v = cf_load(g, idx + stride);\n"
                  "        if (ph) v = rotate_field_left(v, phase); return v; }", text)
    out, b = _sub(r"    if \(ia > 0\) return cf_load\(g, idx - stride\);",
                  "    if (ia > 0) { cf v = cf_load(g, idx - stride);\n"
                  "        if (ph) v = rotate_field_left(v, phase); return v; }", out)
    return out, a + b


def m_conjugate_the_rotation(text: str) -> Tuple[str, int]:
    """Take the conjugate of the Bloch factor inside the rotation.

    The classic sign error: every magnitude stays plausible and only the phase
    moves, which is what a band structure is made of. The host passes the
    conjugate for the backward sub-step (stepping.py:1865-1869); conjugating again
    in the kernel undoes it on one side and doubles it on the other.
    """
    out, a = _sub(r"o\.re = __fmaf_rn\(g\.re, p\.re, \(g\.im \* p\.im\) \* -1\.0f\);",
                  "o.re = __fmaf_rn(g.re, p.re, g.im * p.im);", text)
    out, b = _sub(r"o\.im = __fmaf_rn\(g\.re, p\.im, g\.im \* p\.re\);",
                  "o.im = __fmaf_rn(g.re, p.im, (g.im * p.re) * -1.0f);", out)
    out, c = _sub(r"o\.re = \(g\.re \* p\.re\) - \(g\.im \* p\.im\);",
                  "o.re = (g.re * p.re) + (g.im * p.im);", out)
    out, d = _sub(r"o\.im = \(g\.re \* p\.im\) \+ \(g\.im \* p\.re\);",
                  "o.im = (g.re * p.im) - (g.im * p.re);", out)
    return out, a + b + c + d


def m_hoist_dtdx_into_the_difference(text: str) -> Tuple[str, int]:
    """Distribute ``dtdx`` over the four stencil operands.

    Algebraically identical, numerically not: the array path forms the whole
    difference and multiplies once (stepping.py:1682).
    """
    return _sub(
        r"cf curl = mul_coefficient_left\(dtdx, "
        r"cf_add\(cf_sub\(sf, f_1\), cf_sub\(f_2, ss\)\)\);",
        "cf curl = cf_add(cf_sub(mul_coefficient_left(dtdx, sf),\n"
        "                            mul_coefficient_left(dtdx, f_1)),\n"
        "                     cf_sub(mul_coefficient_left(dtdx, f_2),\n"
        "                            mul_coefficient_left(dtdx, ss)));", text)


def m_word_pair_transposed(text: str) -> Tuple[str, int]:
    """Read the imaginary word first -- the whole volume with its planes swapped.

    A complex64 volume read as ``(im, re)`` pairs is a scrambled field, not a
    launch failure, and it is the failure mode word-pair addressing invites.
    """
    return _sub(r"    z\.re = g\[2 \* idx\];\n    z\.im = g\[2 \* idx \+ 1\];",
                "    z.re = g[2 * idx + 1];\n    z.im = g[2 * idx];", text)


def m_column_major_index(text: str) -> Tuple[str, int]:
    """Decompose the flat index in the wrong order.

    ``i`` and ``k`` swapped: every coefficient lookup and every ghost test then
    names the wrong axis, on a grid whose extents differ.
    """
    return _sub(r"    int k = idx % nz;\n    int j = \(idx / nz\) % ny;\n"
                r"    int i = idx / \(ny \* nz\);",
                "    int i = idx % nx;\n    int j = (idx / nx) % ny;\n"
                "    int k = idx / (nx * ny);", text)


def n_reload_the_target_from_memory(text: str) -> Tuple[str, int]:
    """A NULL mutation: store the target, load it back, store it again.

    A float32 written to global memory and read back is the identity on the bits.
    A leg that CAUGHT this would mean the harness reports differences that are not
    arithmetic, which is the thing an uncaught null is there to rule out.
    """
    return _sub(
        r"    cf_store\(f, idx, mul_field_left\(cf_sub\(cf_add\(a, fu_new\), fprev\), sinv_u\)\);",
        "    cf_store(f, idx, mul_field_left(cf_sub(cf_add(a, fu_new), fprev), sinv_u));\n"
        "    cf_store(f, idx, cf_load(f, idx));", text)


def n_commute_the_plane_sum(text: str) -> Tuple[str, int]:
    """A NULL mutation: commute the two operands of a float ADD.

    IEEE addition is commutative on the bits (it is associativity that is not), so
    this must not be caught. It is the control that separates "the gate sees
    grouping" from "the gate sees any edit at all".
    """
    return _sub(r"    o\.re = a\.re \+ b\.re;\n    o\.im = a\.im \+ b\.im;",
                "    o.re = b.re + a.re;\n    o.im = b.im + a.im;", text)


SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "fold_the_zero_cross_terms": m_fold_the_zero_cross_terms,
    "plane_wise_scaling": m_plane_wise_scaling,
    "left_to_right_curl": m_left_to_right_curl,
    "flatten_the_constitutive_tail": m_flatten_the_constitutive_tail,
    "store_fw_before_reading_prev": m_store_fw_before_reading_prev,
    "drop_the_ownership_mask": m_drop_the_ownership_mask,
    "metallic_ghost_wraps": m_metallic_ghost_wraps,
    "phase_on_every_lane": m_phase_on_every_lane,
    "conjugate_the_rotation": m_conjugate_the_rotation,
    "hoist_dtdx_into_the_difference": m_hoist_dtdx_into_the_difference,
    "word_pair_transposed": m_word_pair_transposed,
    "column_major_index": m_column_major_index,
    "reload_the_target_from_memory": n_reload_the_target_from_memory,
    "commute_the_plane_sum": n_commute_the_plane_sum,
}

#: Host mutations: nothing in the device text changes, the LAUNCHER is handed
#: something wrong. Each is a real defect a call site can carry.
HOST_MUTATIONS: Tuple[str, ...] = (
    "swap_sub_lattice", "drop_the_bloch_flags", "unconjugated_backward_phase",
    "drop_the_metallic_codes", "the_other_expansion_arm")


def apply_host_mutation(name: Optional[str], sub_step: str, layer, grid,
                        expansion) -> Dict[str, Any]:
    """Build the launcher overrides one host mutation needs."""
    overrides: Dict[str, Any] = {}
    if name is None:
        return overrides
    backward = complex_emitter.KERNELS[sub_step][1]
    is_curl = sub_step in CURL_SUB_STEPS
    key = sub_step if is_curl else ("H" if sub_step == "update_H" else "E")
    if name == "swap_sub_lattice":
        # THE HALF-CELL ERROR: hand in the OTHER Yee sub-lattice's coefficient
        # vectors. Converged, smooth and wrong.
        suffix = "" if complex_emitter.HALF_INTEGER[key] else "_h"
        stems = ("kms", "sinv") if is_curl else ("kps", "kms")
        overrides["tables"] = {
            f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in stems}
        return overrides
    if name == "the_other_expansion_arm":
        overrides["expansion"] = other_arm(
            complex_emitter.EXPANSION_NAMES[
                complex_emitter.normalized_expansion(expansion)])
        return overrides
    if not is_curl:
        return overrides
    flags, values = host_phase_arguments(grid, backward)
    codes = tuple(coverage.BC_CODES[kind]
                  for kind in coverage.real_pml_boundary_kinds(grid))
    if name == "drop_the_bloch_flags":
        overrides["phase_flags"] = (0, 0, 0)
        overrides["phase_values"] = values
        overrides["boundary_codes"] = codes
        overrides["tables"] = _shipped_tables(layer, sub_step)
    elif name == "unconjugated_backward_phase":
        overrides["phase_flags"] = flags
        overrides["phase_values"] = tuple(
            value if index % 2 == 0 else -value
            for index, value in enumerate(values))
        overrides["boundary_codes"] = codes
        overrides["tables"] = _shipped_tables(layer, sub_step)
    elif name == "drop_the_metallic_codes":
        overrides["phase_flags"] = flags
        overrides["phase_values"] = values
        overrides["boundary_codes"] = (0, 0, 0)
        overrides["tables"] = _shipped_tables(layer, sub_step)
    return overrides


def _shipped_tables(layer, sub_step: str) -> Dict[str, Any]:
    suffix = "_h" if complex_emitter.HALF_INTEGER[sub_step] else ""
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def case_key(spec: Dict[str, Any], guard_label: str, steps: int) -> str:
    return "|".join([
        spec["sub_step"], spec["shape_label"],
        "".join(b[0] for b in spec["boundaries"]), spec["k_label"],
        f"C{spec['courant']}", spec["value_class"], guard_label, f"n{steps}"])


def one_case(backend, xp, spec: Dict[str, Any], guard_label: str,
             guard: Sequence[str], steps: int, expansion,
             source_mutation: Optional[str] = None,
             host_mutation: Optional[str] = None,
             comparator=None) -> Dict[str, Any]:
    """Oracle and kernel from ONE frozen state; every written word compared.

    One fixture and one restore. Two separately built configurations could differ
    in their inputs, silently, and the comparison would then be of two runs rather
    than of two trees.
    """
    key = case_key(spec, guard_label, steps)
    rng = np.random.default_rng(
        (SEED + int(hashlib.sha256(key.encode("ascii")).hexdigest()[:8], 16))
        % (2 ** 32))
    sub_step = spec["sub_step"]
    is_curl = sub_step in CURL_SUB_STEPS
    fields, layer, grid = build(xp, spec["cell"], spec["boundaries"],
                                spec["k_point"], spec["courant"],
                                needs_epsilon=(sub_step == "update_E"), rng=rng)
    host = seed_state(fields, grid, spec["value_class"], rng)
    frozen = snapshot(fields)
    outputs = OUTPUTS[sub_step]

    # --- what the predicate says about this configuration -------------------
    licence_stub = {"arm": complex_emitter.EXPANSION_NAMES[
        complex_emitter.normalized_expansion(expansion)],
        "expansion": complex_emitter.normalized_expansion(expansion),
        "basis": "measured", "refusals": [],
        "policy_resolved": spec.get("policy")}
    if is_curl:
        admitted, reason = coverage.covers_real_pml_complex_curl(
            fields, layer, grid, sub_step, licence_stub, spec.get("policy"))
    else:
        admitted, reason = coverage.covers_real_pml_complex_constitutive(
            fields, layer, grid, "H" if sub_step == "update_H" else "E",
            licence_stub, spec.get("policy"))

    # --- the oracle: the array path itself ----------------------------------
    array_path = getattr(stepping, sub_step)
    for _ in range(steps):
        array_path(fields, layer)
        advance_sources(fields, sub_step)
    oracle = {name: to_host(getattr(fields, name)).copy() for name in outputs}
    moved = sum(words_differ(oracle[name], frozen[name]) for name in outputs)
    total_words = sum(int(oracle[name].size) * 2 for name in outputs)

    # THE PHASE FLOOR. A "phased" case that rotated nothing measures the unphased
    # kernel under a phased label, which is how a phase defect goes unseen.
    flags, _values = host_phase_arguments(grid, complex_emitter.KERNELS[sub_step][1]) \
        if is_curl else ((0, 0, 0), ())
    phased_axes = int(sum(flags))

    # --- the kernel, from the same frozen state -----------------------------
    restore(fields, frozen)
    backend.set_guard(guard)
    overrides = apply_host_mutation(host_mutation, sub_step, layer, grid, expansion)
    arm = overrides.pop("expansion", expansion)
    launch_error = None
    try:
        for _ in range(steps):
            if is_curl:
                backend.launch_curl(sub_step, fields, layer, grid, arm, **overrides)
            else:
                backend.launch_constitutive(
                    "H" if sub_step == "update_H" else "E", fields, layer, grid,
                    arm, tables=overrides.get("tables"))
            advance_sources(fields, sub_step)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, recorded
        launch_error = f"{type(exc).__name__}: {exc}"[:600]

    if launch_error is None and comparator is None:
        differing = sum(words_differ(oracle[name], to_host(getattr(fields, name)))
                        for name in outputs)
        identical = differing == 0
    elif launch_error is None:
        parts = [comparator(oracle[name], to_host(getattr(fields, name)))
                 for name in outputs]
        identical = all(parts)
        differing = 0 if identical else 1
    else:
        differing, identical = total_words, False

    return {
        "key": key,
        "sub_step": sub_step,
        "shape": list(grid.shape),
        "boundaries": list(spec["boundaries"]),
        "k_point": list(spec["k_point"]),
        "k_label": spec["k_label"],
        "courant": spec["courant"],
        "value_class": spec["value_class"],
        "guard": guard_label,
        "steps": steps,
        "expansion_arm": complex_emitter.EXPANSION_NAMES[
            complex_emitter.normalized_expansion(arm)],
        "source_mutation": source_mutation,
        "host_mutation": host_mutation,
        "bit_identical": bool(identical),
        "differing_words": int(differing),
        "total_words": int(total_words),
        "launch_error": launch_error,
        "oracle_moved_words": int(moved),
        "oracle_moved": moved > 0,
        "phased_axes": phased_axes,
        "phase_is_live": phased_axes > 0,
        "predicate_admits": bool(admitted),
        "predicate_reason": reason,
        "operand_census": word_census(host),
    }


def case_is_valid(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """A case that could not distinguish anything is REFUSED, not passed."""
    if not record["oracle_moved"]:
        return False, ("the array path changed no output word: this fixture is a "
                       "fixed point and cannot distinguish anything")
    return True, None


# ---------------------------------------------------------------------------
# The legs
# ---------------------------------------------------------------------------

def case_product(product: str, policy: Optional[str]) -> List[Dict[str, Any]]:
    """The swept cases, before the guard axis multiplies them."""
    shapes = SHAPES if product == "full" else SHAPES[1:2]
    boundaries = BOUNDARY_TRIPLES if product == "full" else BOUNDARY_TRIPLES[:2]
    ks = K_POINTS if product == "full" else K_POINTS[:2]
    courants = COURANTS if product == "full" else COURANTS[1:]
    # THE REDUCED SET KEEPS ``signed_zero`` AND DROPS ``zero_init``, deliberately.
    # The reduced product is what every mutation leg is scored on, and
    # ``signed_zero`` is the only class on which the zero cross terms can change a
    # word -- so dropping it would leave the two legs this family most needs with
    # nothing to be caught on. ``zero_init`` is dropped because it is a FIXED POINT
    # for both constitutive sub-steps (source zero in, zero out), so those cases
    # are refused as non-distinguishing anyway; the byte leg's full product still
    # carries it for the curl.
    classes = (VALUE_CLASSES if product == "full"
               else ("uniform", "subnormal_band", "signed_zero"))
    cases: List[Dict[str, Any]] = []
    for sub_step in SUB_STEPS:
        for cell, label in shapes:
            for triple in boundaries:
                for k_point, k_label in ks:
                    if config_viable(cell, triple, k_point) is not None:
                        continue
                    for courant in courants:
                        for value_class in classes:
                            cases.append({
                                "sub_step": sub_step, "cell": cell,
                                "shape_label": label, "boundaries": triple,
                                "k_point": k_point, "k_label": k_label,
                                "courant": courant, "value_class": value_class,
                                "policy": policy})
    return cases


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_guard: Dict[str, Dict[str, int]] = {}
    for case in cases:
        bucket = per_guard.setdefault(case["guard"], {
            "identical": 0, "ran": 0, "inexact_identical": 0, "inexact_ran": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["bit_identical"])
        if case["courant"] == INEXACT_COURANT:
            bucket["inexact_ran"] += 1
            bucket["inexact_identical"] += int(case["bit_identical"])
    def bucket_by(field: str) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for case in cases:
            if case["guard"] != "fmad_false":
                continue
            slot = out.setdefault(str(case[field]), {"identical": 0, "ran": 0})
            slot["ran"] += 1
            slot["identical"] += int(case["bit_identical"])
        return out
    census = {"values": 0, "subnormals": 0, "negative_zeros": 0, "zeros": 0}
    for case in cases:
        if case["value_class"] != "subnormal_band":
            continue
        for key in census:
            census[key] += case["operand_census"][key]
    phased = [c for c in cases if c["k_label"] != "k0" and c["sub_step"] in CURL_SUB_STEPS]
    return {
        "per_guard": per_guard,
        "by_sub_step": bucket_by("sub_step"),
        "by_value_class": bucket_by("value_class"),
        "by_k_point": bucket_by("k_label"),
        "subnormal_band_operands": census,
        "subnormal_band_is_non_vacuous": census["subnormals"] > 0,
        "every_case_moved_the_oracle": all(c["oracle_moved"] for c in cases),
        "every_case_admitted_by_the_predicate": all(c["predicate_admits"] for c in cases),
        "predicate_refusals": sorted({c["predicate_reason"] for c in cases
                                      if not c["predicate_admits"]}),
        "phased_curl_cases": len(phased),
        "every_phased_case_rotated_a_plane": all(c["phase_is_live"] for c in phased),
        "every_sub_step_ran": sorted({c["sub_step"] for c in cases}) == sorted(SUB_STEPS),
    }


def run_bytes(backend, xp, results: Dict[str, Any], out_path: str, product: str,
              steps: int, leg_name: str, expansion,
              guards: Sequence[Tuple[str, Tuple[str, ...], bool]] = GUARD_SETS,
              policy: Optional[str] = None) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []
    plan = case_product(product, policy)
    swept = [g for g in guards if backend.certifies or g[0] == "fmad_false"]
    total = len(plan) * len(swept)
    index = 0
    started = time.time()
    for guard_label, guard, _primary in swept:
        for spec in plan:
            index += 1
            record = one_case(backend, xp, spec, guard_label, guard, steps, expansion)
            valid, why = case_is_valid(record)
            record["case_is_valid"] = valid
            record["refused_because"] = why
            (cases if valid else refused).append(record)
            log(f"[{leg_name}] {index}/{total} {record['key']} "
                f"identical={record['bit_identical']} "
                f"diff={record['differing_words']}/{record['total_words']} "
                f"moved={record['oracle_moved_words']} "
                f"admits={record['predicate_admits']}"
                + ("" if valid else f" REFUSED: {why}"))
            results[leg_name] = {"cases": cases, "refused": refused,
                                 "summary": summarize(cases),
                                 "seconds": round(time.time() - started, 1)}
            save(results, out_path)
    return results[leg_name]


def allclose_compare(a: Any, b: Any) -> bool:
    """The magnitude comparison a byte gate exists to replace, for the harness leg."""
    return bool(np.allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-8,
                            equal_nan=True))


#: (leg, what is armed, expectation, why).
#: "pass"     -> every valid case must stay identical
#: "caught"   -> every valid case must DIVERGE
#: "caught_where_live" -> must diverge on the cases the defect can REACH, and stay
#:              identical elsewhere; the pair is the claim, and the reach is named
#: "uncaught" -> every valid case must stay identical, and that IS the measurement
#: "measure"  -> a number is reported, not a bar
MUTATION_LEGS: Tuple[Dict[str, Any], ...] = (
    {"leg": "gate", "mutation": None, "kind": None, "expect": "pass",
     "why": "the shipped source, unmutated, on the mutation leg's own reduced "
            "product -- so a divergence there is attributable to the product and "
            "not to the arming"},
    {"leg": "c1_fold_the_zero_cross_terms", "mutation": "fold_the_zero_cross_terms",
     "kind": "source", "expect": "caught_where_live", "reach": "signed_zero",
     "why": "the zero cross terms carry the sign of the other plane into an "
            "addend that is exactly zero; folding them to a literal is the "
            "transformation the compiler must NOT be doing for us"},
    {"leg": "c2_plane_wise_scaling", "mutation": "plane_wise_scaling",
     "kind": "source", "expect": "caught_where_live", "reach": "signed_zero",
     "why": "the obvious wrong transcription: np.multiply carries no "
            "scalar-times-complex loop, so a real coefficient IS a full complex "
            "multiply on the array path"},
    {"leg": "c3_left_to_right_curl", "mutation": "left_to_right_curl",
     "kind": "source", "expect": "caught", "curl_only": True,
     "why": "float32 addition is not associative and _curl_from_operands fixes "
            "the grouping"},
    {"leg": "c4_flatten_the_constitutive_tail",
     "mutation": "flatten_the_constitutive_tail", "kind": "source",
     "expect": "caught", "constitutive_only": True,
     "why": "two separate accumulations, left to right, is what the array path's "
            "two += statements fix -- and flattening them is exactly what the "
            "twelve uncertified complex kernels write"},
    {"leg": "c5_store_fw_before_reading_prev",
     "mutation": "store_fw_before_reading_prev", "kind": "source",
     "expect": "caught", "constitutive_only": True,
     "why": "prev must be read before the fw store; the wrong order is a slightly "
            "weaker absorber, wrong INSIDE THE PML ONLY"},
    {"leg": "c6_drop_the_ownership_mask", "mutation": "drop_the_ownership_mask",
     "kind": "source", "expect": "caught_where_live", "reach": "metallic",
     "curl_only": True, "dead_must_be_identical": True,
     "why": "cell 0 of a shift-0 component on a metallic axis is the wall MEEP "
            "never steps; without the mask its auxiliary integrates a ghost it "
            "does not own"},
    {"leg": "c7_metallic_ghost_wraps", "mutation": "metallic_ghost_wraps",
     "kind": "source", "expect": "caught_where_live", "reach": "metallic_forward",
     "curl_only": True, "dead_must_be_identical": True,
     "why": "past a perfect electric conductor the field is zero; wrapping there "
            "is the defect the twelve uncertified kernels carry in reverse. THE "
            "REACH IS step_B ONLY, measured first and explained after: on step_D "
            "every down-shifted operand's near ghost is consumed by a cell whose "
            "curl the ownership mask drops, so the D-side metallic ghost is "
            "structurally unobservable -- which is exactly why _shift_down writes "
            "it 'so the stencil is right on its own terms rather than depending "
            "on that mask for its validity'"},
    {"leg": "c8_phase_on_every_lane", "mutation": "phase_on_every_lane",
     "kind": "source", "expect": "caught_where_live", "reach": "phased",
     "curl_only": True, "dead_must_be_identical": True,
     "why": "_apply_bloch_phase multiplies ONE plane of the rolled buffer; "
            "rotating the interior is a smooth, plausible, wrong dispersion"},
    {"leg": "c9_conjugate_the_rotation", "mutation": "conjugate_the_rotation",
     "kind": "source", "expect": "caught_where_live", "reach": "phased",
     "curl_only": True, "dead_must_be_identical": True,
     "why": "the classic sign error: every magnitude stays plausible and only the "
            "phase moves"},
    {"leg": "c10_hoist_dtdx_into_the_difference",
     "mutation": "hoist_dtdx_into_the_difference", "kind": "source",
     "expect": "caught", "curl_only": True,
     "why": "algebraically identical, numerically not: the array path forms the "
            "whole difference and multiplies once"},
    {"leg": "c11_word_pair_transposed", "mutation": "word_pair_transposed",
     "kind": "source", "expect": "caught",
     "why": "a complex64 volume read as (im, re) pairs is a scrambled field, not "
            "a launch failure -- the failure mode word-pair addressing invites"},
    {"leg": "c12_column_major_index", "mutation": "column_major_index",
     "kind": "source", "expect": "caught",
     "why": "every coefficient lookup and every ghost test then names the wrong "
            "axis, on a grid whose extents differ"},
    {"leg": "h1_swap_sub_lattice", "mutation": "swap_sub_lattice", "kind": "host",
     "expect": "caught",
     "why": "the half-cell error the sub-lattice pairing exists to prevent: "
            "converged, smooth and wrong"},
    {"leg": "h2_drop_the_bloch_flags", "mutation": "drop_the_bloch_flags",
     "kind": "host", "expect": "caught_where_live", "reach": "phased",
     "curl_only": True, "dead_must_be_identical": True,
     "why": "PH=0 must be the k=0 kernel and nothing else; dropping the flag on a "
            "PHASED axis runs the unphased kernel under a phased configuration -- "
            "and the uncaught half at k=0 is what shows the reduction is exact"},
    {"leg": "h3_unconjugated_backward_phase",
     "mutation": "unconjugated_backward_phase", "kind": "host",
     "expect": "caught_where_live", "reach": "phased", "curl_only": True,
     "dead_must_be_identical": True,
     "why": "the near face wraps DOWN by one lattice vector and carries the "
            "conjugate; taking the same factor both ways is the sign error MEEP's "
            "own locate_point_in_user_volume avoids"},
    {"leg": "h4_drop_the_metallic_codes", "mutation": "drop_the_metallic_codes",
     "kind": "host", "expect": "caught_where_live", "reach": "metallic",
     "curl_only": True, "dead_must_be_identical": True,
     "why": "an axis that merely absorbs still WRAPS, and one that is metallic "
            "does not; the code is what tells the kernel which"},
    {"leg": "h5_the_other_expansion_arm", "mutation": "the_other_expansion_arm",
     "kind": "host", "expect": "measure",
     "why": "the arm this platform was NOT licensed for. Where the two arms are "
            "measurably apart this must DIVERGE; where they coincide the "
            "coincidence is the finding, and the number of cases in each bucket "
            "is what the licence's discrimination claim is worth here"},
    {"leg": "n1_reload_the_target_from_memory",
     "mutation": "reload_the_target_from_memory", "kind": "source",
     "expect": "uncaught", "curl_only": True,
     "why": "a float32 stored to global memory and loaded back is the identity on "
            "the bits; a leg that caught this would mean the harness reports "
            "differences that are not arithmetic"},
    {"leg": "n2_commute_the_plane_sum", "mutation": "commute_the_plane_sum",
     "kind": "source", "expect": "uncaught",
     "why": "IEEE addition is commutative on the bits -- it is associativity that "
            "is not -- so this separates 'the gate sees grouping' from 'the gate "
            "sees any edit at all'"},
)


def mutation_case_plan(leg: Dict[str, Any], product: str,
                       policy: Optional[str]) -> List[Dict[str, Any]]:
    """The cases one mutation leg is scored on, filtered to what it can reach."""
    plan = case_product("reduced" if product == "full" else product, policy)
    if leg.get("curl_only"):
        plan = [s for s in plan if s["sub_step"] in CURL_SUB_STEPS]
    if leg.get("constitutive_only"):
        plan = [s for s in plan if s["sub_step"] not in CURL_SUB_STEPS]
    return plan


def _reaches(leg: Dict[str, Any], case: Dict[str, Any]) -> bool:
    """Whether this case is one the leg's defect can be seen on at all.

    A REACH IS A CLAIM AND EACH ONE IS DERIVED, not guessed:

    * ``metallic`` -- the defect touches the predicated-zero ghost or the
      ownership mask, so a grid with no metallic axis has nothing to see;
    * ``metallic_forward`` -- the FAR-face ghost only. The NEAR-face ghost is
      structurally unobservable on ``step_D`` and this was measured before it was
      explained: every down-shifted operand of a D term is consumed by the cell
      whose curl ``_mask_non_owned_cells`` drops. Dx masks y and z and reads Hz
      down-y and Hy down-z; Dy masks x and z and reads Hx down-z and Hz down-x;
      Dz masks x and y and reads Hy down-x and Hx down-y (fields.IYEE_SHIFTS,
      fields.py:214-219; stepping.py:1912-1949). So the D near ghost is written
      "so the stencil is right on its own terms rather than depending on that mask
      for its validity", which is what ``_shift_down``'s own docstring says, and a
      gate cannot see it. The B far ghost has no such cover and is caught.
    * ``phased`` -- at least one axis actually carries a rotation;
    * ``signed_zero`` -- the ONE class on which the zero cross terms can change a
      stored word: ``fma(z.re, c, -0.0f)`` differs from ``fma(z.re, c, +0.0f)``
      exactly when ``z.re * c`` is ``-0.0f``.
    """
    reach = leg.get("reach")
    if reach is None:
        return True
    if reach == "metallic":
        return "metallic" in case["boundaries"]
    if reach == "metallic_forward":
        return "metallic" in case["boundaries"] and case["sub_step"] == "step_B"
    if reach == "phased":
        return case["phased_axes"] > 0
    if reach == "signed_zero":
        return case["value_class"] == "signed_zero"
    raise ValueError(f"unknown reach {reach!r}")


def adjudicate(leg: Dict[str, Any], cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Did this leg's verdict come out the way the leg requires?"""
    valid = [c for c in cases if c["case_is_valid"]]
    identical = sum(1 for c in valid if c["bit_identical"])
    ran = len(valid)
    expect = leg["expect"]
    out: Dict[str, Any] = {"identical": identical, "ran": ran,
                           "single": f"{identical}/{ran}"}
    if expect == "pass":
        out["as_required"] = ran > 0 and identical == ran
    elif expect == "caught":
        out["as_required"] = ran > 0 and identical == 0
    elif expect == "uncaught":
        out["as_required"] = ran > 0 and identical == ran
    elif expect == "caught_where_live":
        # THE BAR IS ON THE LIVE HALF ALWAYS: every case the defect can reach must
        # DIVERGE. The dead half is a bar only where the leg says so
        # (``dead_must_be_identical``), and that flag is not decoration -- for the
        # phase and mask legs the null half IS the claim (PH=0 must be the k=0
        # kernel and nothing else; a mask must not fire on a periodic grid). For
        # the two zero-cross-term legs it is deliberately NOT a bar: a defect that
        # also shows on some subnormal-band words is MORE catching, not a failure,
        # and requiring silence there would turn extra sensitivity into a red leg.
        live = [c for c in valid if _reaches(leg, c)]
        dead = [c for c in valid if not _reaches(leg, c)]
        out["reach"] = leg["reach"]
        out["live"] = f"{sum(c['bit_identical'] for c in live)}/{len(live)}"
        out["dead"] = f"{sum(c['bit_identical'] for c in dead)}/{len(dead)}"
        out["dead_is_a_bar"] = bool(leg.get("dead_must_be_identical"))
        out["as_required"] = bool(
            live and not any(c["bit_identical"] for c in live)
            and (all(c["bit_identical"] for c in dead)
                 if leg.get("dead_must_be_identical") else True))
    elif expect == "measure":
        diverged = [c for c in valid if not c["bit_identical"]]
        out["diverged"] = len(diverged)
        out["coincided"] = ran - len(diverged)
        out["diverged_on"] = sorted({
            f"{c['sub_step']}/{c['k_label']}/{c['value_class']}" for c in diverged})
        out["coincided_on"] = sorted({
            f"{c['sub_step']}/{c['k_label']}/{c['value_class']}"
            for c in valid if c["bit_identical"]})
        out["max_differing_words"] = max((c["differing_words"] for c in valid),
                                         default=0)
        # A measurement leg passes when it MEASURED something -- and, for the arm
        # swap specifically, when at least one case could tell the arms apart. A
        # leg on which nothing ever diverges has not shown the arm matters here.
        out["as_required"] = ran > 0 and len(diverged) > 0
    else:
        raise ValueError(f"unknown expectation {expect!r}")
    return out


def run_mutations(backend, xp, results: Dict[str, Any], out_path: str,
                  product: str, expansion, policy: Optional[str]) -> Dict[str, Any]:
    legs: List[Dict[str, Any]] = []
    started = time.time()
    for leg in MUTATION_LEGS:
        leg_started = time.time()
        transform = (SOURCE_MUTATIONS[leg["mutation"]]
                     if leg.get("kind") == "source" else None)
        host_mutation = leg["mutation"] if leg.get("kind") == "host" else None
        sites = backend.set_source_transform(transform)
        if transform is not None and sites == 0:
            legs.append({**leg, "sites": 0, "as_required": False,
                         "failure": "the mutation needle matched nothing: a leg "
                                    "that arms nothing reports a pass while "
                                    "exercising nothing"})
            backend.set_source_transform(None)
            log(f"[mutations] {leg['leg']} ARMED NOTHING")
            results["mutations"] = {"legs": legs}
            save(results, out_path)
            continue
        cases: List[Dict[str, Any]] = []
        plan = mutation_case_plan(leg, product, policy)
        for spec in plan:
            record = one_case(backend, xp, spec, "fmad_false", ("--fmad=false",),
                              1, expansion, source_mutation=leg.get("mutation")
                              if leg.get("kind") == "source" else None,
                              host_mutation=host_mutation)
            valid, why = case_is_valid(record)
            record["case_is_valid"] = valid
            record["refused_because"] = why
            cases.append(record)
        verdict = adjudicate(leg, cases)
        entry = {**leg, "sites": sites, **verdict,
                 "seconds": round(time.time() - leg_started, 1),
                 "cases": cases}
        legs.append(entry)
        log(f"[mutations] {leg['leg']} expect={leg['expect']} "
            f"identical={verdict['single']} as_required={verdict['as_required']}")
        # DISARM AND CHECK IT. A leg that left a mutation installed would score
        # every later leg against mutated bytes.
        backend.set_source_transform(None)
        results["mutations"] = {
            "legs": legs,
            "legs_as_required": f"{sum(1 for l in legs if l.get('as_required'))}/{len(legs)}",
            "all_legs_as_required": all(l.get("as_required") for l in legs),
            "seconds": round(time.time() - started, 1)}
        save(results, out_path)
    disarmed = complex_emitter.complex_source("step_B", "FMA_V1")
    results["mutations"]["every_leg_disarmed"] = (
        hashlib.sha256(disarmed.encode("utf-8")).hexdigest()
        == results["device_source_sha256"]["step_B_pml_complex_bloch"])
    save(results, out_path)
    return results["mutations"]


LEG_NAMES = ("bytes", "multistep", "mutations")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", default="auto",
                        choices=("auto", "cupy", "host", "numpy"))
    parser.add_argument("--legs", default=",".join(LEG_NAMES))
    parser.add_argument("--product", default="auto",
                        choices=("auto", "full", "reduced"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--multi-step-budget", type=int, default=MULTI_STEP_BUDGET)
    parser.add_argument("--subnormal-policy", default=None,
                        choices=("keep", "flush", "match_meep", "ieee"))
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--expansion-probe", default=None,
                        help="the probe artifact the arm is bound from")
    parser.add_argument("--expansion-arm", default=None,
                        choices=("NAIVE", "FMA_V1"),
                        help="bind this arm WITHOUT a licence; the artifact then "
                             "records that it was not certified")
    args = parser.parse_args(argv)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    legs = tuple(name for name in args.legs.split(",") if name)
    backend_name = args.backend
    if backend_name == "auto":
        backend_name = "cupy" if cp is not None else "host"
    product = args.product
    if product == "auto":
        product = "full" if backend_name == "cupy" else "reduced"

    results: Dict[str, Any] = {
        "gate": "cuda_complex_pml",
        "kernels": [complex_emitter.KERNELS[s][0] for s in SUB_STEPS],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "backend": backend_name,
        "certifies": backend_name == "cupy",
        "product": product,
        "legs_requested": list(legs),
        "seed": SEED,
        "multi_step_budget": args.multi_step_budget,
        "emitter_corpus_digest": complex_emitter.corpus_digest(),
        "subjects": {
            name: probe.source_digest(os.path.join(
                _REPO_API, "meep_gpu", "cuda_kernels", name))
            for name in ("complex_emitter.py", "complex_pml_kernels.py",
                         "coverage.py", "compile_cache.py")},
        "oracle_subjects": {
            name: probe.source_digest(os.path.join(_REPO_API, "meep_gpu", name))
            for name in ("stepping.py", "fields.py", "pml.py", "grid.py")},
        "licence_subject": probe.source_digest(os.path.join(
            _REPO_API, "meep_gpu", "triton_kernels", "complex_fields.py")),
        "gate_sha256": probe.source_digest(os.path.abspath(__file__)),
        # SET FALSE BEFORE ANYTHING RUNS: an artifact from a run that died mid-leg
        # must not read as RELEASED.
        "passed": False,
    }

    if backend_name == "cupy":
        results["nvrtc_binary_observer"] = probe.install_nvrtc_binary_observer()
    if args.import_meep_for_host_policy:
        results["meep_import_for_host_policy"] = probe.import_meep_for_host_policy()
    if args.subnormal_policy and backend_name == "cupy":
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    elif args.subnormal_policy:
        # THE POLICY IS A LABEL ON A BACKEND THAT CANNOT INSTALL IT, and the
        # artifact says so rather than implying the run was cut under it. The
        # installer drives CuPy's NVRTC seam; there is nothing for it to drive on a
        # host-compiled or harness run. The NAME is still carried because the
        # predicate's licence clause is a policy comparison, and a laptop leg that
        # could not exercise that clause would leave the one refusal path most
        # likely to be got wrong untested.
        results["subnormal_policy_install"] = {
            "installed": False, "policy": args.subnormal_policy,
            "why": "the requested policy is a LABEL on this backend: only the cupy "
                   "backend has an NVRTC seam to install it at, so these bytes "
                   "belong to whatever this host's default arithmetic is"}
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    # --- THE ARM ---------------------------------------------------------
    licence = load_licence(args.expansion_probe, args.subnormal_policy)
    results["expansion_licence"] = licence
    if licence["usable"]:
        expansion = licence["arm"]
        results["expansion_arm"] = expansion
        results["expansion_arm_basis"] = licence.get("basis")
    elif args.expansion_arm:
        expansion = args.expansion_arm
        results["expansion_arm"] = expansion
        results["expansion_arm_basis"] = "asserted_without_a_licence"
    else:
        results["expansion_arm"] = None
        results["verdict"] = {
            "passed": False,
            "why_not_certified": (
                "no usable expansion licence: the complex multiply's arm is a "
                "measured platform fact and this gate will not guess it. Cut a "
                "probe artifact with cut_expansion_probe.py under the SAME "
                "subnormal policy this run installs, or pass --expansion-arm to "
                "run uncertified."),
            "licence_refusals": (licence.get("verdict") or {}).get("refusals", []),
            "policy_reasons": licence.get("policy_reasons", [])}
        save(results, out_path)
        log("[done] REFUSED: no usable expansion licence")
        return 2

    backend = build_backend(backend_name)
    results["device_source_sha256"] = {
        complex_emitter.KERNELS[sub_step][0]: hashlib.sha256(
            complex_emitter.complex_source(sub_step, expansion).encode("utf-8")
        ).hexdigest() for sub_step in SUB_STEPS}
    results["device_source_sha256_both_arms"] = {
        f"{complex_emitter.KERNELS[sub_step][0]}|{arm}": hashlib.sha256(
            complex_emitter.complex_source(sub_step, arm).encode("utf-8")).hexdigest()
        for sub_step in SUB_STEPS for arm in ("NAIVE", "FMA_V1")}
    xp = cp if backend_name == "cupy" else _NumpyWearingCupysName()
    if backend_name == "cupy":
        results["environment"] = probe.device_info()
        results["guard_effect"] = backend.guard_effect(expansion)
    if backend_name == "host":
        results["host_compiler"] = {
            "path": backend.compiler, "options": list(backend.options),
            "certifies": False,
            "what_it_measures": (
                "the emitted source text, compiled by the host C++ compiler and "
                "driven with a block/thread loop, against the NumPy array path. "
                "It measures the TRANSCRIPTION -- indices, stencils, masks, "
                "operand orders, sub-lattices, the phase table -- and nothing "
                "about NVRTC.")}
    save(results, out_path)

    try:
        if "bytes" in legs:
            run_bytes(backend, xp, results, out_path, product, args.steps, "bytes",
                      expansion, policy=args.subnormal_policy)
        if "multistep" in legs:
            run_bytes(backend, xp, results, out_path,
                      "reduced" if product == "full" else product,
                      args.multi_step_budget, "multistep", expansion,
                      guards=GUARD_SETS[:1], policy=args.subnormal_policy)
        if "mutations" in legs:
            run_mutations(backend, xp, results, out_path, product, expansion,
                          args.subnormal_policy)
    finally:
        backend.restore()

    if backend_name == "cupy":
        results["nvrtc_binaries"] = probe.nvrtc_binary_report()

    single = results.get("bytes", {}).get("summary", {})
    multi = results.get("multistep", {}).get("summary", {})
    mutations = results.get("mutations", {})
    primary = single.get("per_guard", {}).get("fmad_false", {})
    control = single.get("per_guard", {}).get("default_no_options", {})
    multi_primary = multi.get("per_guard", {}).get("fmad_false", {})
    verdict = {
        "single_launch": f"{primary.get('identical', 0)}/{primary.get('ran', 0)}",
        "multi_step": f"{multi_primary.get('identical', 0)}/{multi_primary.get('ran', 0)}",
        "multi_step_budget": args.multi_step_budget,
        "guard_control_identical": f"{control.get('identical', 0)}/{control.get('ran', 0)}",
        "guard_control_diverged": (control.get("ran", 0) > 0
                                   and control.get("identical", 0) < control.get("ran", 0)),
        # THE GUARD CLAUSE IS SATISFIED TWO WAYS AND THE ARTIFACT SAYS WHICH.
        # Every certified sibling required the OUTPUT control to diverge, on the
        # reasoning that a flag whose removal changes nothing is decoration. That
        # reasoning has a second branch this family lands in: the flag can also
        # change nothing because the source gives it nothing to do, and the way to
        # tell the two apart is not the outputs but the COMPILED IMAGE. Measured
        # here per kernel: under the licensed FMA_V1 arm the cubins are IDENTICAL
        # with and without --fmad=false (so no run can differ), and under NAIVE
        # they DIFFER on all four (so the flag is real and the measurement is not
        # blind). An identical control with differing images would be a gate that
        # cannot see, and it would fail this clause.
        "guard_images_identical_for_the_shipped_arm":
            (results.get("guard_effect") or {}).get(
                "shipped_arm_images_identical_across_guards"),
        "guard_images_identical_for_the_other_arm":
            (results.get("guard_effect") or {}).get(
                "other_arm_images_identical_across_guards"),
        "expansion_arm": results.get("expansion_arm"),
        "expansion_arm_basis": results.get("expansion_arm_basis"),
        "by_sub_step": single.get("by_sub_step"),
        "by_k_point": single.get("by_k_point"),
        "by_value_class": single.get("by_value_class"),
        "subnormal_band_is_non_vacuous": single.get("subnormal_band_is_non_vacuous"),
        "every_case_moved_the_oracle": single.get("every_case_moved_the_oracle"),
        "every_case_admitted_by_the_predicate":
            single.get("every_case_admitted_by_the_predicate"),
        "every_phased_case_rotated_a_plane":
            single.get("every_phased_case_rotated_a_plane"),
        "every_sub_step_ran": single.get("every_sub_step_ran"),
        "mutation_legs_as_required": mutations.get("legs_as_required"),
        "all_mutation_legs_as_required": mutations.get("all_legs_as_required"),
        "every_leg_disarmed": mutations.get("every_leg_disarmed"),
    }
    # Either the control DIVERGED (the flag changes the answer and the gate saw
    # it), or the flag provably cannot change the answer because it leaves the
    # compiled image bit-identical -- AND the other arm shows the measurement is
    # not blind. Nothing else satisfies the clause.
    verdict["guard_clause_satisfied"] = bool(
        verdict["guard_control_diverged"]
        or (verdict["guard_images_identical_for_the_shipped_arm"]
            and verdict["guard_images_identical_for_the_other_arm"] is False))
    verdict["guard_clause_route"] = (
        "the control diverged" if verdict["guard_control_diverged"]
        else "the flag leaves the shipped arm's compiled image bit-identical, and "
             "changes the other arm's, so it is inert here rather than unmeasured"
        if verdict["guard_clause_satisfied"] else "neither route")
    verdict["passed"] = bool(
        results["certifies"]
        and results.get("expansion_arm_basis") in ("measured", "environment_default")
        and primary.get("ran", 0) > 0 and primary["identical"] == primary["ran"]
        and multi_primary.get("ran", 0) > 0
        and multi_primary["identical"] == multi_primary["ran"]
        and verdict["guard_clause_satisfied"]
        and single.get("subnormal_band_is_non_vacuous")
        and single.get("every_case_moved_the_oracle")
        and single.get("every_case_admitted_by_the_predicate")
        and single.get("every_phased_case_rotated_a_plane")
        and single.get("every_sub_step_ran")
        and mutations.get("all_legs_as_required", False)
        and mutations.get("every_leg_disarmed", False))
    if not results["certifies"]:
        verdict["why_not_certified"] = (
            "only the cupy backend compiles with NVRTC and runs on a device; every "
            "question this family turns on -- whether the zero cross terms survive "
            "folding, whether __fmaf_rn lowers to one fma.rn.f32, what --fmad=false "
            "removes -- is a question about that compiler")
    results["verdict"] = verdict
    results["passed"] = verdict["passed"]
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, out_path)

    log(f"[done] backend={backend_name} arm={results.get('expansion_arm')} "
        f"single={verdict['single_launch']} multi={verdict['multi_step']} "
        f"mutations={verdict['mutation_legs_as_required']} "
        f"guard={verdict['guard_clause_route']} "
        f"passed={verdict['passed']}")
    if not results["certifies"]:
        return 0 if (mutations.get("all_legs_as_required", True)
                     and primary.get("identical", 0) == primary.get("ran", 0)) else 1
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
