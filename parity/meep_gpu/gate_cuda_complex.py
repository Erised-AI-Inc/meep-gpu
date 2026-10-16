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
THE FORMS, AND WHAT THE ARM NO LONGER DECIDES
=============================================================================

The kernels transcribe the array path's part-wise, per-chunk forms (bit_parity_0_10
DESIGN.md D1, D6, D10, D11, D16): every real-coefficient product is one float32
product per part (``part_products.scale``), each cell takes MEEP's PML form from the
run's chunk record ``cb`` (the split-field recurrence where its chunk carries the
dsigu layer, the dsig-only form elsewhere; the constitutive accumulation where its
chunk carries the dsigw layer, ``f = src`` elsewhere), the curl is MEEP's C-order
bracket with its partner-presence forms, and a Bloch-phased axis takes MEEP's
not-owned word on plane 0 of the operand read at the cell itself
(``part_products.connect``'s rule: a copy, a per-part negation or the unfused
product). Every launch binds the record.

ONE BODY. The emitter has no expansion arm: every complex product is formed
part by part, as the array path and MEEP form it.

=============================================================================
TWO BACKENDS, AND WHAT EACH ONE LICENSES
=============================================================================

``--backend cupy`` is the CERTIFYING backend: NVRTC compiles the emitted source
and the GPU runs it. Nothing else can certify, because the question this family
turns on -- does ``--fmad=false`` keep every part-wise product and every sum of the
connection's unfused product separately rounded -- is a question about a compiler.

``--backend host`` compiles THE SAME SOURCE TEXT with the host C++ compiler
(``clang++``/``g++``), shimming ``__device__``, ``__global__`` and ``__restrict__``
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
* the GUARD CLAUSE: the output control (compiled with no options) DIVERGES. The
  part-wise body carries contraction candidates (the connection's
  ``(p.re * g.re) - (p.im * g.im)``, the constitutive's
  ``f + ((kps*src) - (kms*prev))``), so the flag has work to do; the artifact also
  records whether the compiled image differs across the flag;
* every mutation leg as required -- including the inverted MEEP-grouping legs
  (KERNELS.md section 7.1: the previous package grouping is the mutant now) and the
  section 7.2 B battery on the chunk forms -- every leg disarmed afterwards, and no
  leg unmeasurable on this backend;
* the non-vacuity floors: the oracle moved words in every case, the subnormal band
  actually contained subnormals, every case the array path connects (MEEP's connection
  rule, not the k-point's label: a nonzero k that wraps to +1 is a copy) actually
  rotated a plane, each axis encoded as the array path connects it, and
  every case was ADMITTED BY THE PREDICATE (a gate that certifies a configuration
  its own predicate refuses has measured the wrong thing).

Usage, on the validation host (from ``parity/meep_gpu``)::

    PYTHONPATH=../.. CUDA_VISIBLE_DEVICES=<one empty index> \\
    CUPY_CACHE_DIR=<fresh, policy-token-carrying> \\
    python -u gate_cuda_complex.py --backend cupy --product full \\
        --subnormal-policy keep \\
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
import gate_chunk_records  # noqa: E402

from meep_gpu import part_products, stepping  # noqa: E402
from meep_gpu.chunk_map import record_index  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import complex_emitter, coverage  # noqa: E402

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
#: rotation's zero cross terms. On an axis of even cell count at resolution 1 the edge
#: wraps to +1 instead, a COPY by MEEP's connection rule (the 8x8x8 shape); the phase
#: floor grades by the connection and requires the edge to negate on some shape.
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
#: policy decides the fate of.
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
    # MID-RUN: H is already separated from B (stepping.separate_h_from_b ran at the
    # run's first update_H), so the oracle and the kernel both step the seeded state
    # and neither performs the start -- which the launcher would otherwise run on the
    # array path before the kernel.
    fields.h_separated = True
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
# The two kernel-side backends
# ---------------------------------------------------------------------------

_SIGNATURE = re.compile(r'extern "C" __global__ void (\w+)\(([^)]*)\)', re.S)

#: What the host compiler needs in place of CUDA: the keywords it does not know.
_HOST_SHIM = r'''
#include <math.h>
#define __device__
#define __forceinline__ inline
#define __global__
#define __restrict__ __restrict
struct meep_gpu_uint3 { int x, y, z; };
static struct meep_gpu_uint3 blockIdx, blockDim, threadIdx;
'''

#: ``-ffp-contract=off`` is the host compiler's ``--fmad=false``: without it the
#: host compiler fuses ``a*b + c`` on its own and the comparison stops being about
#: the transcription. ``-fno-fast-math`` keeps every signed zero a product makes.
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
        emits, per sub-step, so a leg cannot mutate one specialization and score
        another.
        """
        self.module._clear_kernel_cache()
        if transform is None:
            complex_emitter.complex_source = self._original_source
            return 0
        sites = {"n": 0}
        original = self._original_source

        def mutated(sub_step):
            text, count = transform(original(sub_step))
            sites["n"] = max(sites["n"], count)
            return text

        complex_emitter.complex_source = mutated
        # Emit once per sub-step so the site count is known before any launch: a
        # mutation whose needle matched nothing exercises nothing and would
        # otherwise report a pass while doing it.
        for sub_step in complex_emitter.KERNELS:
            mutated(sub_step)
        return sites["n"]

    def launch_curl(self, sub_step, fields, layer, grid, *,
                    record, tables=None, boundary_codes=None, phase_flags=None,
                    phase_values=None, dtdx=None) -> None:
        cb = cp.asarray(np.ascontiguousarray(record, dtype=np.int32))
        if tables is None and boundary_codes is None and phase_flags is None:
            self.module.step_fused_pml_complex(sub_step, fields,
                                               grid=grid, pml=layer, chunk_bounds=cb)
            return
        derived_flags, derived_values = self.module.bloch_phase_arguments(
            grid, complex_emitter.KERNELS[sub_step][1])
        self.module.step_fused_pml_complex(
            sub_step, fields,
            tables=(tables if tables is not None
                    else self.module.complex_curl_tables(
                        layer, complex_emitter.HALF_INTEGER[sub_step])),
            boundary_codes=(boundary_codes if boundary_codes is not None
                            else self.module.complex_boundary_codes(grid)),
            phase_flags=phase_flags if phase_flags is not None else derived_flags,
            phase_values=(phase_values if phase_values is not None
                          else derived_values),
            dtdx=dtdx if dtdx is not None else grid.dt / grid.dx, chunk_bounds=cb)

    def launch_constitutive(self, side, fields, layer, grid, *,
                            record, tables=None) -> None:
        cb = cp.asarray(np.ascontiguousarray(record, dtype=np.int32))
        if tables is None:
            self.module.update_fused_pml_complex(side, fields, pml=layer,
                                                 chunk_bounds=cb)
        else:
            self.module.update_fused_pml_complex(side, fields,
                                                 tables=tables, chunk_bounds=cb)

    def guard_effect(self) -> Dict[str, Any]:
        """What ``--fmad=false`` does to the COMPILED IMAGE, per kernel.

        THE GUARD CONTROL IS AN OUTPUT COMPARISON AND THIS IS A BYTE ONE, and the
        pair is what makes an identical control mean something. A control that
        comes back identical is either a gate that cannot see or a source the flag
        has nothing to do to; two bit-identical NVRTC images settle it -- they
        cannot produce different words -- and two that differ while the outputs
        agree would be the weaker claim, which this reports rather than hides.

        The part-wise body carries a ``mul`` feeding an ``add`` (the connection's
        unfused product, the constitutive's right-hand side), which is what
        ``--fmad=false`` exists to keep apart.
        """
        from cupy.cuda import compiler  # noqa: PLC0415

        device = cp.cuda.Device()
        arch = f"{device.compute_capability[0]}{device.compute_capability[1:]}"
        out: Dict[str, Any] = {"arch": arch, "rows": []}
        for sub_step in complex_emitter.KERNELS:
            source = self._original_source(sub_step)
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
                "sub_step": sub_step, **digests,
                "identical_across_guards":
                    digests["fmad_false"]["sha256"]
                    == digests["default_no_options"]["sha256"]})
        out["images_identical_across_guards"] = all(
            r["identical_across_guards"] for r in out["rows"])
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
            _, count = transform(complex_emitter.complex_source(sub_step))
            sites = max(sites, count)
        return sites

    def _entry(self, sub_step: str):
        key = sub_step
        if key in self._cache:
            return self._cache[key]
        source = complex_emitter.complex_source(sub_step)
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

    def launch_curl(self, sub_step, fields, layer, grid, *,
                    record, tables=None, boundary_codes=None, phase_flags=None,
                    phase_values=None, dtdx=None) -> None:
        function, _ = self._entry(sub_step)
        record = np.ascontiguousarray(record, dtype=np.int32)
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
        arguments += [record.ctypes.data_as(ctypes.c_void_p)]
        cells = int(nx) * int(ny) * int(nz)
        function((cells + 255) // 256, 256, *arguments)

    def launch_constitutive(self, side, fields, layer, grid, *,
                            record, tables=None) -> None:
        sub_step = "update_H" if side == "H" else "update_E"
        function, _ = self._entry(sub_step)
        record = np.ascontiguousarray(record, dtype=np.int32)
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
        arguments += [record.ctypes.data_as(ctypes.c_void_p)]
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
            _, count = transform(complex_emitter.complex_source(sub_step))
            sites = max(sites, count)
        return sites

    def launch_curl(self, *args, **kwargs) -> None:
        raise RuntimeError("the numpy backend launches no kernel")

    def launch_constitutive(self, *args, **kwargs) -> None:
        raise RuntimeError("the numpy backend launches no kernel")

    def restore(self) -> None:
        return None


def host_phase_arguments(grid, backward: bool):
    """The connection encoding, for the backend that cannot call the CuPy launcher.

    ``complex_pml_kernels.bloch_phase_arguments`` when CuPy is importable, and its
    reproduction otherwise: ``readback.cmul`` forms ``thephase = 1 * factor`` (the
    D curl -- ``backward`` -- takes ``eikna``, the word above the stored window; the B
    curl ``conj(eikna)``, the word below it) and ``part_products.connect_parts``' rule
    resolves it to a kind (a copy, a negation, the unfused product) and float32 parts.
    ``test_complex_chunk_forms.py`` pins the shipped function's kinds against the
    connection the array path applies.
    """
    try:
        from meep_gpu.cuda_kernels.complex_pml_kernels import (  # noqa: PLC0415
            bloch_phase_arguments)
    except ImportError:
        bloch_phase_arguments = None
    if bloch_phase_arguments is not None:
        kinds, parts = bloch_phase_arguments(grid, backward)
        return (tuple(int(kind) for kind in kinds),
                tuple(float(value) for value in parts))
    from meep_gpu.readback import cmul  # noqa: PLC0415

    boundary = coverage.real_pml_boundary_kinds(grid)
    flags: List[int] = []
    values: List[float] = []
    for axis in range(3):
        phase = grid.bloch_phase(axis) if getattr(grid, "has_bloch", False) else None
        if phase is None:
            flags.append(complex_emitter.CONNECT_NONE)
            values.extend((1.0, 0.0))
            continue
        if boundary[axis] != "periodic":
            raise ValueError(
                f"axis {axis} carries Bloch phase {phase!r} but resolved to "
                f"{boundary[axis]!r}")
        eikna = complex(phase)
        factor = (eikna.real, eikna.imag) if backward else (eikna.real, -eikna.imag)
        pr, pi = (float(part) for part in cmul((1.0, 0.0), factor))
        tolerance = part_products.CONNECT_TOLERANCE
        if abs(pi) < tolerance and abs(pr - 1.0) < tolerance:
            flags.append(complex_emitter.CONNECT_NONE)
            values.extend((1.0, 0.0))
        elif abs(pi) < tolerance and abs(pr + 1.0) < tolerance:
            flags.append(complex_emitter.CONNECT_NEGATE)
            values.extend((-1.0, 0.0))
        else:
            flags.append(complex_emitter.CONNECT_PHASE)
            values.extend((float(np.float32(pr)), float(np.float32(pi))))
    return tuple(flags), tuple(values)


#: The kernel's kind for each connection the array path can apply. An axis with no Bloch
#: phase and an axis whose phase resolves to +1 both read the stored word: kind 0.
ARRAY_CONNECTION_KIND: Dict[str, int] = {
    "unconnected": complex_emitter.CONNECT_NONE, "copy": complex_emitter.CONNECT_NONE,
    "negate": complex_emitter.CONNECT_NEGATE, "product": complex_emitter.CONNECT_PHASE}


def array_path_connection(grid, backward: bool) -> Tuple[str, ...]:
    """The connection the ARRAY PATH applies on each axis, read without the kernel module.

    The phase floor's reference. ``host_phase_arguments`` prefers the kernel module's own
    ``bloch_phase_arguments``, so a floor that read only it would grade the kernel's
    encoding against itself. This reads what the oracle applies: ``grid.bloch_phase``
    (``eikna``, the Brillouin edge exactly -1+0j), ``thephase = 1 * factor`` by
    ``readback.cmul`` with the direction's factor, and ``part_products.connect_parts``'
    rule (boundaries.cpp:359-366): within ``CONNECT_TOLERANCE`` of +1 a COPY, of -1 a
    NEGATION, otherwise the unfused PRODUCT. A nonzero k-point can connect nothing: k = 0.5
    on an 8-cell axis at resolution 1 is eikna = exp(2*pi*i*4) = +1, a copy, in MEEP and in
    the array path alike.
    """
    from meep_gpu.readback import cmul  # noqa: PLC0415

    tolerance = part_products.CONNECT_TOLERANCE
    kinds: List[str] = []
    for axis in range(3):
        phase = grid.bloch_phase(axis) if getattr(grid, "has_bloch", False) else None
        if phase is None:
            kinds.append("unconnected")
            continue
        eikna = complex(phase)
        factor = (eikna.real, eikna.imag) if backward else (eikna.real, -eikna.imag)
        pr, pi = (float(part) for part in cmul((1.0, 0.0), factor))
        if abs(pi) < tolerance and abs(pr - 1.0) < tolerance:
            kinds.append("copy")
        elif abs(pi) < tolerance and abs(pr + 1.0) < tolerance:
            kinds.append("negate")
        else:
            kinds.append("product")
    return tuple(kinds)


def case_record(grid, layer) -> np.ndarray:
    """The chunk record of the case's own division: the one-rank division of ``grid``
    with ``layer``, which is the division the oracle's region plan steps with (no driver
    installs another; stepping's first PML sub-step plans the one-rank layout)."""
    return np.ascontiguousarray(gate_chunk_records.layer_record(grid, layer),
                                dtype=np.int32)


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


def m_full_complex_coefficient_products(text: str) -> Tuple[str, int]:
    """INVERTED: the previous family's full complex product against ``(c + 0j)``.

    The shipped helpers are ``part_products.scale``'s one product per part. The
    previous family wrote the zero-imaginary complex multiply with its literal zero
    cross terms (``(z.re * c) - (z.im * 0.0f)``), which carries the OTHER part's sign
    into an addend that is exactly zero -- a different word wherever a part is a
    signed zero. MUST BE CAUGHT on the signed-zero class.
    """
    out, a = _sub(r"    o\.re = z\.re \* c;\n    o\.im = z\.im \* c;",
                  "    o.re = (z.re * c) - (z.im * 0.0f);\n"
                  "    o.im = (z.re * 0.0f) + (z.im * c);", text)
    out, b = _sub(r"    o\.re = c \* z\.re;\n    o\.im = c \* z\.im;",
                  "    o.re = (c * z.re) - (0.0f * z.im);\n"
                  "    o.im = (c * z.im) + (0.0f * z.re);", out)
    return out, a + b


def m_old_package_curl_grouping(text: str) -> Tuple[str, int]:
    """INVERTED (KERNELS.md section 7.1): the previous package grouping of the curl.

    MEEP's ``step_curl`` brackets ``((g1s - g1) + g2) - g2s`` and the shipped
    ``curl_stencil`` is that bracket; the previous package grouping
    ``(g1s - g1) + (g2 - g2s)`` is a different float32 number. MUST BE CAUGHT.
    """
    return _sub(r"cf_sub\(cf_add\(cf_sub\(sf, f1\), f2\), ss\)",
                "cf_add(cf_sub(sf, f1), cf_sub(f2, ss))", text)


def m_old_package_constitutive_grouping(text: str) -> Tuple[str, int]:
    """INVERTED (KERNELS.md section 7.1): two accumulations instead of MEEP's grouping.

    ``step_update_EDHB`` forms the right-hand side first, ``f + ((kps*src) -
    (kms*prev))``; the previous package grouping ``(f + kps*src) - kms*prev`` is a
    different float32 number on the cells whose chunk carries the dsigw layer. MUST BE
    CAUGHT.
    """
    return _sub(
        r"cf_add\(cf_load\(f, idx\), cf_sub\(mul_coefficient_left\(kps, src\), "
        r"mul_coefficient_left\(kms, prev\)\)\)",
        "cf_sub(cf_add(cf_load(f, idx), mul_coefficient_left(kps, src)), "
        "mul_coefficient_left(kms, prev))", text)


def m_store_fw_before_reading_prev(text: str) -> Tuple[str, int]:
    """Write ``fw`` before reading ``prev`` -- a slightly weaker absorber.

    Wrong only where ``kms != 0`` on a cell whose chunk carries the dsigw layer, i.e.
    INSIDE THE PML ONLY, which is why it reads as physics rather than as a bug.
    """
    return _sub(r"    cf prev = dsigw \? cf_load\(fw, idx\) : cf_zero\(\);\n"
                r"    if \(dsigw\) cf_store\(fw, idx, src\);",
                "    if (dsigw) cf_store(fw, idx, src);\n"
                "    cf prev = dsigw ? cf_load(fw, idx) : cf_zero();", text)


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


def m_connection_on_every_plane(text: str) -> Tuple[str, int]:
    """Connect the operand read at the cell itself on EVERY plane, not plane 0 only.

    ``stepping._bloch_ghost_operand`` takes MEEP's not-owned word on plane 0 of a
    phased axis and the stored word everywhere else; rotating the interior is a
    smooth, plausible, completely wrong dispersion relation.
    """
    return _sub(r"    if \(ia != 0 \|\| ph == 0\) return w;",
                "    if (ph == 0) return w;", text)


def m_conjugate_the_rotation(text: str) -> Tuple[str, int]:
    """Take the conjugate of the Bloch factor inside the unfused product.

    The classic sign error: every magnitude stays plausible and only the phase moves,
    which is what a band structure is made of. The host resolves the factor (the D
    curl's ``eikna``, the B curl's ``conj(eikna)``); conjugating again in the kernel
    undoes it on one sub-step and doubles it on the other.
    """
    return _sub(r"    o\.re = \(p\.re \* g\.re\) - \(p\.im \* g\.im\);\n"
                r"    o\.im = \(p\.re \* g\.im\) \+ \(p\.im \* g\.re\);",
                "    o.re = (p.re * g.re) + (p.im * g.im);\n"
                "    o.im = (p.re * g.im) - (p.im * g.re);", text)


def m_hoist_dtdx_into_the_difference(text: str) -> Tuple[str, int]:
    """Distribute ``dtdx`` over the four stencil operands.

    Algebraically identical, numerically not: MEEP forms the whole bracket and
    multiplies once (``part_products.scale(dtdx, ...)`` on the array path).
    """
    return _sub(
        r"if \(pr1 && pr2\) return mul_coefficient_left\(dtdx, "
        r"cf_sub\(cf_add\(cf_sub\(sf, f1\), f2\), ss\)\);",
        "if (pr1 && pr2) return cf_sub(cf_add(cf_sub(mul_coefficient_left(dtdx, sf), "
        "mul_coefficient_left(dtdx, f1)), mul_coefficient_left(dtdx, f2)), "
        "mul_coefficient_left(dtdx, ss));", text)


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
    return _sub(r"    cf_store\(f, idx, value\);\n",
                "    cf_store(f, idx, value);\n    cf_store(f, idx, cf_load(f, idx));\n",
                text)


def n_commute_the_plane_sum(text: str) -> Tuple[str, int]:
    """A NULL mutation: commute the two operands of a float ADD.

    IEEE addition is commutative on the bits (it is associativity that is not), so
    this must not be caught. It is the control that separates "the gate sees
    grouping" from "the gate sees any edit at all".
    """
    return _sub(r"    o\.re = a\.re \+ b\.re;\n    o\.im = a\.im \+ b\.im;",
                "    o.re = b.re + a.re;\n    o.im = b.im + a.im;", text)


# --- KERNELS.md section 7.2: the B battery on the per-chunk forms -------------------

def b_select_removed(text: str) -> Tuple[str, int]:
    """The whole-grid recurrence (the previous form): every cell takes the dsigu /
    dsigw form whatever its chunk carries. MUST BE CAUGHT."""
    out, a = _sub(r"    cf fprev = dsigu \? cf_load\(fu, idx\) : cf_zero\(\);",
                  "    dsigu = 1;\n    cf fprev = dsigu ? cf_load(fu, idx) : cf_zero();",
                  text)
    out, b = _sub(r"    cf prev = dsigw \? cf_load\(fw, idx\) : cf_zero\(\);",
                  "    dsigw = 1;\n    cf prev = dsigw ? cf_load(fw, idx) : cf_zero();", out)
    return out, a + b


def b_predicate_on_coefficient_identity(text: str) -> Tuple[str, int]:
    """The form chosen by ``kms != 1`` instead of chunk membership. MUST BE CAUGHT.

    KERNELS.md section 7.2: the predicate the family carried until 0.10. A cell
    inside a layer's chunk whose own coefficients are exactly unity takes a different
    form, and so does one outside every chunk that carries the layer whose
    coefficients are not.
    """
    out, a = re.subn(
        r"(pml_apply\([^;]*?kms_([xyz])\[([ijk])\], sinv_\2\[\3\]), in_[ih]_\2\);",
        r"\1, ((kms_\2[\3] != 1.0f) || (sinv_\2[\3] != 1.0f)));", text)
    out, b = re.subn(
        r"(constitutive_apply\([^;]*?kps_([xyz])\[([ijk])\], kms_\2\[\3\]), in_[ih]_\2\);",
        r"\1, ((kps_\2[\3] != 1.0f) || (kms_\2[\3] != 1.0f)));", out)
    return out, a + b


def b_swap_dsig_and_dsigu(text: str) -> Tuple[str, int]:
    """The dsig axis's membership handed in as the dsigu flag. MUST BE CAUGHT.

    KERNELS.md section 7.2: dsig-only against dsigu-only, on every cell whose chunk
    carries exactly one of the two layers.
    """
    return re.subn(
        r"(pml_apply\([^;]*?kms_([xyz])\[[ijk]\], sinv_\2\[[ijk]\], "
        r"kms_[xyz]\[[ijk]\], sinv_[xyz]\[[ijk]\]), in_([ih])_[xyz]\);",
        r"\1, in_\3_\2);", text)


def b_auxiliary_stored_on_a_plain_cell(text: str) -> Tuple[str, int]:
    """The auxiliary written on every cell. MUST BE CAUGHT: the auxiliary contract
    leaves it untouched on a plain cell and every auxiliary is compared."""
    out, a = _sub(r"    if \(dsigu\) cf_store\(fu, idx, fu_new\);",
                  "    cf_store(fu, idx, fu_new);", text)
    out, b = _sub(r"    if \(dsigw\) cf_store\(fw, idx, src\);",
                  "    cf_store(fw, idx, src);", out)
    return out, a + b


def b_auxiliary_load_unguarded(text: str) -> Tuple[str, int]:
    """A DECLARED NULL (KERNELS.md section 7.2): the auxiliary load issued on every
    cell, the store still guarded. The loaded word feeds only the forms of the cells
    whose chunk carries the layer, so no stored word changes; the static needle in the
    module's own test pins the guard. MUST BE UNCAUGHT."""
    out, a = _sub(r"    cf fprev = dsigu \? cf_load\(fu, idx\) : cf_zero\(\);",
                  "    cf fprev = cf_load(fu, idx);", text)
    out, b = _sub(r"    cf prev = dsigw \? cf_load\(fw, idx\) : cf_zero\(\);",
                  "    cf prev = cf_load(fw, idx);", out)
    return out, a + b


SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "full_complex_coefficient_products": m_full_complex_coefficient_products,
    "old_package_curl_grouping": m_old_package_curl_grouping,
    "old_package_constitutive_grouping": m_old_package_constitutive_grouping,
    "store_fw_before_reading_prev": m_store_fw_before_reading_prev,
    "drop_the_ownership_mask": m_drop_the_ownership_mask,
    "metallic_ghost_wraps": m_metallic_ghost_wraps,
    "connection_on_every_plane": m_connection_on_every_plane,
    "conjugate_the_rotation": m_conjugate_the_rotation,
    "hoist_dtdx_into_the_difference": m_hoist_dtdx_into_the_difference,
    "word_pair_transposed": m_word_pair_transposed,
    "column_major_index": m_column_major_index,
    "select_removed": b_select_removed,
    "predicate_on_coefficient_identity": b_predicate_on_coefficient_identity,
    "swap_dsig_and_dsigu": b_swap_dsig_and_dsigu,
    "auxiliary_stored_on_a_plain_cell": b_auxiliary_stored_on_a_plain_cell,
    "auxiliary_load_unguarded": b_auxiliary_load_unguarded,
    "reload_the_target_from_memory": n_reload_the_target_from_memory,
    "commute_the_plane_sum": n_commute_the_plane_sum,
}

#: Host mutations: nothing in the device text changes, the LAUNCHER is handed
#: something wrong. Each is a real defect a call site can carry.
HOST_MUTATIONS: Tuple[str, ...] = (
    "swap_sub_lattice", "drop_the_bloch_connection", "the_other_direction_s_connection",
    "drop_the_metallic_codes", "record_bound_off_by_one", "record_lattices_swapped")

#: The host mutations that corrupt the chunk record a launch binds (the oracle's
#: division is untouched).
RECORD_HOST_MUTATIONS: Tuple[str, ...] = ("record_bound_off_by_one",
                                          "record_lattices_swapped")


def host_mutated_record(name: Optional[str], record: np.ndarray, grid) -> np.ndarray:
    """Corrupt the chunk record a LAUNCH BINDS, leaving the oracle's division alone.

    KERNELS.md section 7.2: ``record_bound_off_by_one`` moves each layered axis's low
    run end one row up on both lattices (the boundary plane changes form);
    ``record_lattices_swapped`` hands each lattice the other's runs (the boundary plane
    moves by one cell). The record itself for any other name.
    """
    if name not in RECORD_HOST_MUTATIONS:
        return record
    out = np.array(record, dtype=np.int32, copy=True)
    for axis in range(3):
        if int(grid.shape[axis]) <= 1:
            continue
        for half in (False, True):
            a0 = out[record_index(half, axis, 0)]
            a1 = out[record_index(half, axis, 1)]
            if name == "record_bound_off_by_one" and a1 > a0:
                out[record_index(half, axis, 1)] = a1 + 1
        if name == "record_lattices_swapped":
            for word in range(4):
                low = record_index(False, axis, word)
                high = record_index(True, axis, word)
                out[low], out[high] = record[high], record[low]
    return out


def apply_host_mutation(name: Optional[str], sub_step: str, layer, grid) -> Dict[str, Any]:
    """Build the launcher overrides one host mutation needs."""
    overrides: Dict[str, Any] = {}
    if name is None or name in RECORD_HOST_MUTATIONS:
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
    if not is_curl:
        return overrides
    flags, values = host_phase_arguments(grid, backward)
    codes = tuple(coverage.BC_CODES[kind]
                  for kind in coverage.real_pml_boundary_kinds(grid))
    if name == "drop_the_bloch_connection":
        overrides["phase_flags"] = (complex_emitter.CONNECT_NONE,) * 3
        overrides["phase_values"] = values
        overrides["boundary_codes"] = codes
        overrides["tables"] = _shipped_tables(layer, sub_step)
    elif name == "the_other_direction_s_connection":
        # The D curl handed the B curl's conjugate and the reverse: the word on the
        # wrong side of the stored window.
        other_flags, other_values = host_phase_arguments(grid, not backward)
        overrides["phase_flags"] = other_flags
        overrides["phase_values"] = other_values
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
             guard: Sequence[str], steps: int,
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
    if is_curl:
        admitted, reason = coverage.covers_real_pml_complex_curl(
            fields, layer, grid, sub_step)
    else:
        admitted, reason = coverage.covers_real_pml_complex_constitutive(
            fields, layer, grid, "H" if sub_step == "update_H" else "E")

    # --- the oracle: the array path itself ----------------------------------
    array_path = getattr(stepping, sub_step)
    for _ in range(steps):
        array_path(fields, layer)
        advance_sources(fields, sub_step)
    oracle = {name: to_host(getattr(fields, name)).copy() for name in outputs}
    moved = sum(words_differ(oracle[name], frozen[name]) for name in outputs)
    total_words = sum(int(oracle[name].size) * 2 for name in outputs)

    # THE PHASE FLOOR. A "phased" case that connected nothing measures the unphased
    # kernel under a phased label, which is how a connection defect goes unseen. PHASED
    # means the ARRAY PATH connects a plane (MEEP's rule, read without the kernel
    # module), not that the k-point is nonzero: a nonzero k that resolves to +1 is a copy.
    flags, _values = host_phase_arguments(grid, complex_emitter.KERNELS[sub_step][1]) \
        if is_curl else ((0, 0, 0), ())
    phased_axes = int(sum(1 for flag in flags if flag != complex_emitter.CONNECT_NONE))
    product_axes = int(sum(1 for flag in flags if flag == complex_emitter.CONNECT_PHASE))
    array_kinds = (array_path_connection(grid, complex_emitter.KERNELS[sub_step][1])
                   if is_curl else ("unconnected",) * 3)

    # --- the kernel, from the same frozen state -----------------------------
    restore(fields, frozen)
    backend.set_guard(guard)
    record = host_mutated_record(host_mutation, case_record(grid, layer), grid)
    overrides = apply_host_mutation(host_mutation, sub_step, layer, grid)
    launch_error = None
    try:
        for _ in range(steps):
            if is_curl:
                backend.launch_curl(sub_step, fields, layer, grid, record=record,
                                    **overrides)
            else:
                backend.launch_constitutive(
                    "H" if sub_step == "update_H" else "E", fields, layer, grid,
                    record=record, tables=overrides.get("tables"))
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
        "source_mutation": source_mutation,
        "host_mutation": host_mutation,
        "bit_identical": bool(identical),
        "differing_words": int(differing),
        "total_words": int(total_words),
        "launch_error": launch_error,
        "oracle_moved_words": int(moved),
        "oracle_moved": moved > 0,
        "phased_axes": phased_axes,
        "product_axes": product_axes,
        "phase_is_live": phased_axes > 0,
        "array_connection": list(array_kinds),
        "array_connects_a_plane": any(kind in ("negate", "product") for kind in array_kinds),
        "connection_agrees": ([int(flag) for flag in flags]
                              == [ARRAY_CONNECTION_KIND[kind] for kind in array_kinds]),
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
    floor = phase_floor(cases)
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
        "phased_curl_cases": floor["phased_curl_cases"],
        "phase_floor": floor,
        "every_phased_case_rotated_a_plane": floor["passed"],
        "every_sub_step_ran": sorted({c["sub_step"] for c in cases}) == sorted(SUB_STEPS),
    }


def phase_floor(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Did every case the ARRAY PATH connects rotate a plane in the kernel?

    Graded by the connection, not by the k-point's label (``array_path_connection``):
    a phase-labelled case whose every axis resolves to a copy under MEEP's rule is
    recorded with its configuration, not counted as phased. The clauses keep the floor's
    teeth: every curl case encodes each axis as the array path connects it (a kernel
    that copied where the oracle negates fails here, not only in the bytes); every
    phase-labelled k-point connects a plane on at least one shape; and the Brillouin
    edge, carried for its exact -1+0j (``K_POINTS``), negates on at least one shape.
    """
    curl = [c for c in cases if c["sub_step"] in CURL_SUB_STEPS]
    labelled = [c for c in curl if c["k_label"] != "k0"]
    phased = [c for c in curl if c["array_connects_a_plane"]]
    copies = [c for c in labelled if not c["array_connects_a_plane"]]
    live_labels = {c["k_label"] for c in phased if c["phase_is_live"]}
    edge = [c for c in labelled if c["k_label"] == "kedge"]
    clauses = {
        "every_case_connects_as_the_array_path": all(c["connection_agrees"] for c in curl),
        "every_phased_case_is_live_in_the_kernel": all(c["phase_is_live"] for c in phased),
        "every_phased_k_point_connects_a_plane_somewhere": all(
            label in live_labels for label in {c["k_label"] for c in labelled}),
        "the_brillouin_edge_negates_somewhere": not edge or any(
            "negate" in c["array_connection"] and c["connection_agrees"] for c in edge),
    }
    return {
        "clauses": clauses,
        "passed": all(clauses.values()),
        "phased_curl_cases": len(phased),
        "phase_labelled_curl_cases": len(labelled),
        "phase_labelled_cases_that_copy": len(copies),
        "configurations_that_copy": sorted({
            ("x".join(str(n) for n in c["shape"]), c["k_label"]) for c in copies}),
    }


def run_bytes(backend, xp, results: Dict[str, Any], out_path: str, product: str,
              steps: int, leg_name: str,
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
            record = one_case(backend, xp, spec, guard_label, guard, steps)
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
    {"leg": "c1_full_complex_coefficient_products",
     "mutation": "full_complex_coefficient_products",
     "kind": "source", "expect": "caught_where_live", "reach": "signed_zero",
     "why": "INVERTED: the previous family's full complex product against (c + 0j), "
            "with its literal zero cross terms, carries the other part's sign into an "
            "addend that is exactly zero; part_products.scale forms one product per "
            "part"},
    {"leg": "c3_old_package_curl_grouping", "mutation": "old_package_curl_grouping",
     "kind": "source", "expect": "caught", "curl_only": True,
     "why": "INVERTED (KERNELS.md section 7.1): MEEP's bracket ((g1s - g1) + g2) - g2s "
            "is shipped; the previous package grouping is a different float32 number"},
    {"leg": "c4_old_package_constitutive_grouping",
     "mutation": "old_package_constitutive_grouping", "kind": "source",
     "expect": "caught", "constitutive_only": True,
     "why": "INVERTED (KERNELS.md section 7.1): MEEP's f + ((kps*src) - (kms*prev)) is "
            "shipped; two accumulations are a different float32 number"},
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
    {"leg": "c8_connection_on_every_plane", "mutation": "connection_on_every_plane",
     "kind": "source", "expect": "caught_where_live", "reach": "phased",
     "curl_only": True, "dead_must_be_identical": True,
     "why": "MEEP's not-owned word is taken on plane 0 of a phased axis only; "
            "connecting the interior is a smooth, plausible, wrong dispersion"},
    {"leg": "c9_conjugate_the_rotation", "mutation": "conjugate_the_rotation",
     "kind": "source", "expect": "caught_where_live", "reach": "product_phase",
     "curl_only": True, "dead_must_be_identical": True,
     "why": "the classic sign error: every magnitude stays plausible and only the "
            "phase moves. It reaches the unfused product only: a copy and a negation "
            "never call it"},
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
    {"leg": "h2_drop_the_bloch_connection", "mutation": "drop_the_bloch_connection",
     "kind": "host", "expect": "caught_where_live", "reach": "phased",
     "curl_only": True, "dead_must_be_identical": True,
     "why": "kind 0 must be the k=0 kernel and nothing else; dropping the connection on "
            "a PHASED axis runs the unphased kernel under a phased configuration -- and "
            "the uncaught half at k=0 is what shows the reduction is exact"},
    {"leg": "h3_the_other_direction_s_connection",
     "mutation": "the_other_direction_s_connection", "kind": "host",
     "expect": "caught_where_live", "reach": "product_phase", "curl_only": True,
     "dead_must_be_identical": True,
     "why": "the D curl reads the word ABOVE the stored window (eikna) and the B curl "
            "the word BELOW it (conj(eikna)); handing either the other's factor is the "
            "sign error. A copy and a negation are their own conjugates, so only the "
            "unfused product can see it"},
    {"leg": "h4_drop_the_metallic_codes", "mutation": "drop_the_metallic_codes",
     "kind": "host", "expect": "caught_where_live", "reach": "metallic",
     "curl_only": True, "dead_must_be_identical": True,
     "why": "an axis that merely absorbs still WRAPS, and one that is metallic "
            "does not; the code is what tells the kernel which"},
    {"leg": "h6_record_bound_off_by_one", "mutation": "record_bound_off_by_one",
     "kind": "host", "expect": "caught",
     "why": "KERNELS.md section 7.2: the boundary plane of each layered axis changes "
            "form"},
    {"leg": "h7_record_lattices_swapped", "mutation": "record_lattices_swapped",
     "kind": "host", "expect": "caught",
     "why": "KERNELS.md section 7.2: each sub-lattice reads the other's runs, and the "
            "boundary plane moves by one cell"},
    {"leg": "b1_select_removed", "mutation": "select_removed", "kind": "source",
     "expect": "caught",
     "why": "KERNELS.md section 7.2: the whole-grid recurrence, the form the family "
            "stepped before the chunk forms"},
    {"leg": "b2_predicate_on_coefficient_identity",
     "mutation": "predicate_on_coefficient_identity", "kind": "source",
     "expect": "caught",
     "why": "KERNELS.md section 7.2: the form chosen by the coefficient's value rather "
            "than by the chunk's membership"},
    {"leg": "b3_swap_dsig_and_dsigu", "mutation": "swap_dsig_and_dsigu",
     "kind": "source", "expect": "caught", "curl_only": True,
     "why": "KERNELS.md section 7.2: dsig-only against dsigu-only on the cells whose "
            "chunk carries exactly one of the two layers"},
    {"leg": "b4_auxiliary_stored_on_a_plain_cell",
     "mutation": "auxiliary_stored_on_a_plain_cell", "kind": "source",
     "expect": "caught",
     "why": "KERNELS.md section 7.2: the auxiliary contract leaves it untouched on a "
            "plain cell, and every auxiliary is compared"},
    {"leg": "b5_auxiliary_load_unguarded", "mutation": "auxiliary_load_unguarded",
     "kind": "source", "expect": "uncaught",
     "why": "A DECLARED NULL (KERNELS.md section 7.2): the loaded word feeds only the "
            "layered forms, so no stored word changes; the module's own test pins the "
            "guard by needle"},
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
    * ``phased`` -- at least one axis actually carries a connection (a negation or
      the unfused product);
    * ``product_phase`` -- at least one axis carries the UNFUSED PRODUCT, the only
      connection a conjugation or a direction swap can change;
    * ``signed_zero`` -- the class on which a literal zero cross term changes a stored
      word: a part that is a signed zero.
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
    if reach == "product_phase":
        return case["product_axes"] > 0
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
        # A measurement leg passes when it MEASURED something: at least one case
        # diverged. A leg on which nothing ever diverges has shown nothing.
        out["as_required"] = ran > 0 and len(diverged) > 0
    else:
        raise ValueError(f"unknown expectation {expect!r}")
    return out


def run_mutations(backend, xp, results: Dict[str, Any], out_path: str,
                  product: str, policy: Optional[str]) -> Dict[str, Any]:
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
                              1, source_mutation=leg.get("mutation")
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
    disarmed = complex_emitter.complex_source("step_B")
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
        # host-compiled or harness run. The NAME is still carried so the artifact
        # says which policy was asked for.
        results["subnormal_policy_install"] = {
            "installed": False, "policy": args.subnormal_policy,
            "why": "the requested policy is a LABEL on this backend: only the cupy "
                   "backend has an NVRTC seam to install it at, so these bytes "
                   "belong to whatever this host's default arithmetic is"}
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    backend = build_backend(backend_name)
    results["device_source_sha256"] = {
        complex_emitter.KERNELS[sub_step][0]: hashlib.sha256(
            complex_emitter.complex_source(sub_step).encode("utf-8")
        ).hexdigest() for sub_step in SUB_STEPS}
    xp = cp if backend_name == "cupy" else _NumpyWearingCupysName()
    if backend_name == "cupy":
        results["environment"] = probe.device_info()
        results["guard_effect"] = backend.guard_effect()
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
            run_bytes(backend, xp, results, out_path, product, args.steps, "bytes", policy=args.subnormal_policy)
        if "multistep" in legs:
            run_bytes(backend, xp, results, out_path,
                      "reduced" if product == "full" else product,
                      args.multi_step_budget, "multistep",
                      guards=GUARD_SETS[:1], policy=args.subnormal_policy)
        if "mutations" in legs:
            run_mutations(backend, xp, results, out_path, product,
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
        # THE GUARD CLAUSE IS THE OUTPUT CONTROL: it must diverge. The compiled
        # images across the flag are recorded beside it, so an identical control
        # says whether the flag had nothing to do or the gate could not see.
        "guard_images_identical_across_the_flag":
            (results.get("guard_effect") or {}).get(
                "images_identical_across_guards"),
        "by_sub_step": single.get("by_sub_step"),
        "by_k_point": single.get("by_k_point"),
        "by_value_class": single.get("by_value_class"),
        "subnormal_band_is_non_vacuous": single.get("subnormal_band_is_non_vacuous"),
        "every_case_moved_the_oracle": single.get("every_case_moved_the_oracle"),
        "every_case_admitted_by_the_predicate":
            single.get("every_case_admitted_by_the_predicate"),
        "every_phased_case_rotated_a_plane":
            single.get("every_phased_case_rotated_a_plane"),
        "phase_floor": single.get("phase_floor"),
        "every_sub_step_ran": single.get("every_sub_step_ran"),
        "mutation_legs_as_required": mutations.get("legs_as_required"),
        "all_mutation_legs_as_required": mutations.get("all_legs_as_required"),
        "every_leg_disarmed": mutations.get("every_leg_disarmed"),
    }
    verdict["guard_clause_satisfied"] = bool(verdict["guard_control_diverged"])
    verdict["guard_clause_route"] = (
        "the control diverged" if verdict["guard_control_diverged"]
        else "the control did not diverge")
    verdict["passed"] = bool(
        results["certifies"]
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
            "only the cupy backend compiles with NVRTC and runs on a device; the "
            "question this family turns on -- whether --fmad=false keeps every "
            "part-wise product and every sum separately rounded -- is a question about "
            "that compiler")
    results["verdict"] = verdict
    results["passed"] = verdict["passed"]
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, out_path)

    log(f"[done] backend={backend_name} "
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
