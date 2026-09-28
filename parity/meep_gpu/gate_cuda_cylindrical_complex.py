"""Sub-step byte-identity gate for the NEW hand-CUDA COMPLEX cylindrical curl pair.

THE CLAIM UNDER TEST. ``meep_gpu/cuda_kernels/cylindrical_complex_kernels.py`` emits
two kernels that did not exist before this run -- ``cyl_step_B_pml_complex`` and
``cyl_step_D_pml_complex`` -- and this gate asks whether either reproduces
``stepping.step_B`` / ``stepping.step_D`` WORD FOR WORD on a real Dcyl grid at
``abs(grid.m) >= 1`` with complex64 storage under a split-field PML. Nothing in
``coverage.py``, ``step_curl_kernels.py``, ``complex_emitter.py`` or
``cylindrical_kernels.py`` is touched; the pair carries its own predicate
(``cylindrical_coverage.covers_pml_cylindrical_complex_curl``).

CUDA HAS NEVER BEEN MEASURED ON A COMPLEX Dcyl CURL. There is no standing record to
reproduce and nothing to inherit: every number below is this run's.

=============================================================================
WHAT THE FAMILY ADDS, AND THEREFORE WHAT THIS GATE HAS TO BE ABLE TO SEE
=============================================================================

Five things (``cylindrical_complex_kernels``' docstring derives them off
``stepping.py`` with line numbers). The battery is built around them, because a gate
that only re-ran the certified complex battery would certify a Dcyl kernel against
Cartesian defects:

1. the RADIAL PREFIX SUM, its ZERO WALL ROW on the B side and its ``ir0`` per side;
2. Bz's whole curl REPLACED by the prefix's forward difference; Dz's ``first`` source
   SWAPPED for the prefix while Dx's Hy operands stay RAW;
3. the i*m/r COUPLING on targets 0 and 2, from a HOST-BUILT per-r row, added AFTER the
   dtdx curl and BEFORE the ownership mask;
4. the |m| = 1 AXIS-ROW INCREMENTS, which REPLACE curl row 0 after the mask;
5. the PER-|m| AXIS RULES on the stored value -- ``Dz[0] = 0`` at |m| = 1 (field only)
   and the six-volume near-axis zeroing at |m| >= 2.

=============================================================================
THREE BACKENDS, AND ONLY ONE OF THEM CERTIFIES
=============================================================================

* ``cuda`` -- NVRTC and a GPU. The only backend whose bytes are the shipped bytes and
  the only one that can release.
* ``host`` -- the SAME EMITTED CHARACTERS compiled by the host C++ compiler and driven
  on NumPy. It certifies nothing about NVRTC and says so in the artifact; what it
  measures is the TRANSCRIPTION -- indices, stencils, masks, operand orders,
  coefficient sub-lattices, the prefix wiring and the axis rules -- at the merge bar.
  It is the ``gate_cuda_complex`` pattern, borrowed rather than re-invented.
* neither -- ``--backend host`` on a machine with no compiler skips with its reason.

THE HOST LEG COMPILES AT ``-O0`` AND THAT IS A CORRECTNESS SETTING, MEASURED. This
family carries the only negation of an fma OUTPUT in the package (the axis-increment
row, ``curl.re = inc.re * -1.0f``), and Apple clang 21 at ``-O1`` rewrites
``-(fma(a,b,c))`` as ``fma(-a,b,-c)`` -- exact for every finite value and wrong on
signed zeros. Measured over 96 host cases per cell (two sub-steps x four seedings x
two absorber depths x six grids):

    -O0  0/96 diverged      -O1  3/96 (one uint32 word each)      -O2  0/96

every divergence the same cell -- ``step_D``, ``zero_init``, thin absorber, |m| = 1,
``fu_Dy`` at r = 0 -- and an explicit sign-bit negation does NO better (it is wrong at
``-O2`` where ``* -1.0f`` is right). The optimization level is the variable, not the
source. ``-O0`` is the setting under which the host compiler runs the transcription
rather than an algebraic rewrite of it. NVRTC IS A DIFFERENT COMPILER and was ASKED:
the device leg's ``zero_init`` cases at |m| = 1 came back bit-identical, 129/129 under
both float32 subnormal policies, so it does not share the fold. The leg stays armed --
if it ever reports that word the fold is REAL on CUDA and this family is not
releasable.

=============================================================================
THE CONTROLS
=============================================================================

* ``stripped_control`` -- the CERTIFIED complex CARTESIAN body run on this gate's own
  Dcyl grids. It must DIVERGE, and by a lot: a zero there would mean this module
  should not exist. The Triton sibling measured 1,281,955 differing words; this leg
  makes the number this backend's.
* ``axis_identity`` -- the array path with ``boundaries[0]`` forced to ``'metallic'``,
  i.e. exactly the ghost rule the kernel compiles. It must be BIT-IDENTICAL to the
  unforced array path, which is what licenses ``bc_x = BC_METALLIC`` and makes the
  cylindrical axis cost the kernel no branch.
* the GUARD control -- ``--fmad=false`` against no options, scored where the guarded
  leg was identical, so "the guard is load-bearing" is measured rather than assumed.
* the mutation battery, INCLUDING legs that must be UNCAUGHT.

=============================================================================
THE FLOORS THAT REFUSE A VACUOUS PASS
=============================================================================

* ``oracle_moved`` -- the fraction of output words the array path changed. Zero is a
  refusal, not a pass.
* ``axis_row_is_live`` -- the rows the per-|m| rules touch must carry MOVED words.
* ``prefix_is_live`` -- the prefix must be non-constant down r.
* ``imr_is_live`` -- the two bound coefficient rows must be non-constant down r and
  not the identity, or a row-index error is invisible.
* ``absorbs`` -- ``max|kms-1|`` and ``max|sinv-1|`` on r and z.
* ``signed_zero_census`` -- on the ``zero_init`` and ``signed_zero`` seedings the
  reference must carry NEGATIVE ZERO words, or the whole class is VACUOUS and is
  reported as such rather than as a pass. That is the sibling tranche's rule and it
  is why those cases run with a THIN absorber: measured on this family's own grids, a
  quarter-cell absorber pushes ``min kms_z`` positive and the census to 0.
* ``INEXACT_COURANT`` -- 0.5 is exactly representable and distributing ``dtdx`` over a
  difference is then EXACT, so a matrix without a non-power-of-two Courant cannot see
  a grouping error at all.
* every mutation declares a REACH and is scored ONLY where its defect can be seen. A
  reach is a claim and each one is derived, not guessed.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU; a FRESH CuPy cache dir per leg, because the
disk-cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=6 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_cylindrical_complex.py --backend cuda \\
        --subnormal-policy keep \\
        --expansion-probe results/expansion_probe_2026-08-17/expansion_probe_keep.json \\
        --out $OUT/keep/gate.json

Laptop (no CUDA)::

    python -u gate_cuda_cylindrical_complex.py --backend host \\
        --expansion-probe results/expansion_probe_2026-08-17/expansion_probe_keep.json \\
        --out /tmp/cylcx_host.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten atomically
after every case, so an interrupted run keeps everything up to the failure.
Correctness only -- no throughput claim is made or possible, because the radial prefix
stays on the array path.
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
except ImportError:  # laptop: the host backend still runs
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402
import gate_cuda_complex as complex_gate  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import complex_emitter  # noqa: E402
from meep_gpu.cuda_kernels import coverage as cuda_coverage  # noqa: E402
from meep_gpu.cuda_kernels import cylindrical_complex_kernels as family  # noqa: E402
from meep_gpu.cuda_kernels import cylindrical_coverage as cyl_coverage  # noqa: E402
from meep_gpu.cuda_kernels import cylindrical_prefix as cyl_prefix  # noqa: E402
from meep_gpu.triton_kernels import complex_fields  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host

SEED = 20260820

#: Consecutive launches in the multi-step leg. 60 is the budget both certified
#: hand-CUDA records are cut at; "identical for N steps" is a claim about N.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the recurrence live. Held fixed, the curl operands
#: never move, ``fu`` reaches a fixed point and 60 launches measure what one does --
#: and the PREFIX, recomputed from those sources, would be a constant the kernel could
#: have cached. The same exact float32 scale is applied on both paths.
_ADVANCE = np.float32(0.97)

BC_PERIODIC = 0
BC_METALLIC = 1

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: Every complex volume a case seeds and compares. BOTH sub-steps' volumes are seeded
#: on every case, because ``step_B``'s prefix reads ``Ey`` and the axis increments read
#: ``Ez``/``Hx``/``Hz``: a zero source there is a fixed point that hides the rule.
STATE: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")


def outputs(sub_step: str) -> Tuple[str, ...]:
    """The six volumes this sub-step writes.

    ``fu`` IS STATE. A tree that gets the field right and the auxiliary wrong is
    correct for exactly one launch and wrong forever after, so the auxiliary is
    compared on every case, not only in the multi-step leg.
    """
    targets, _sources = family.CURL_ARRAYS[sub_step]
    return tuple(targets) + tuple("fu_" + name for name in targets)


# ---------------------------------------------------------------------------
# The case product
# ---------------------------------------------------------------------------
#
# EVERY CASE IS A REAL Dcyl GRID at |m| >= 1 with complex64 storage. ``Grid`` builds
# the r axis itself, so the only boundary a case chooses is z -- and BOTH terminations
# appear, because they select different code in the ghost rule and the ownership mask.

CASES: Tuple[Dict[str, Any], ...] = (
    # |m| = 1, both signs, both z terminations. m = -1 is the corpus's own majority
    # (seven of the eight |m| = 1 rows) and is the sign whose axis-increment scalar
    # carries a NEGATIVE ZERO real word.
    {"label": "m+1_r16_z20_metallic", "shape": (16, 1, 20), "m": 1, "z_kind": "metallic"},
    {"label": "m-1_r16_z20_metallic", "shape": (16, 1, 20), "m": -1, "z_kind": "metallic"},
    {"label": "m-1_r16_z20_periodic", "shape": (16, 1, 20), "m": -1, "z_kind": "periodic"},
    # ODD EXTENTS on both axes: the wall row lands on an odd row count and the linear
    # index decomposition has no power-of-two stride to hide behind.
    {"label": "m+1_r9_z17_metallic", "shape": (9, 1, 17), "m": 1, "z_kind": "metallic"},
    # |m| >= 2 -- the near-axis zeroing, at three different ZERO_ROWS counts.
    {"label": "m+2_r9_z17_metallic", "shape": (9, 1, 17), "m": 2, "z_kind": "metallic"},
    {"label": "m+3_r9_z17_periodic", "shape": (9, 1, 17), "m": 3, "z_kind": "periodic"},
    {"label": "m+5_r12_z48_metallic", "shape": (12, 1, 48), "m": 5, "z_kind": "metallic"},
    # A WIDER RADIAL EXTENT: the prefix's accumulated sum runs further, which is where
    # a decaying field reaches the subnormal band first.
    {"label": "m-1_r40_z16_metallic", "shape": (40, 1, 16), "m": -1, "z_kind": "metallic"},
    # THE ACCURATE NEAR-AXIS BRANCH, which is ZERO_ROWS = 1 at |m| = 3 rather than 3.
    # ``Grid`` refuses a Courant above ~1/(|m| + 0.5) here, so this case carries its
    # own; it is the one case whose Courant is not from COURANTS.
    {"label": "m+3_accurate_r16_z20", "shape": (16, 1, 20), "m": 3, "z_kind": "metallic",
     "accurate": True, "courant": 0.2},
    # m = 0 UNDER COMPLEX STORAGE, the third m class (2026-09-04): no i*m/r term, no
    # curl-row fold, Bx[0] = 0 on the B side, the Dz post-add and Dy[0] = 0 on the D
    # side. Both z terminations, and the CORPUS ROW'S OWN SHAPE -- examples:
    # dipole_in_vacuum_cyl_off_axis.py lifts to (150, 1, 300), m = 0, PML, z
    # metallic -- so the configuration the board prices is one the gate stepped.
    {"label": "m0_r16_z20_metallic", "shape": (16, 1, 20), "m": 0, "z_kind": "metallic"},
    {"label": "m0_r16_z20_periodic", "shape": (16, 1, 20), "m": 0, "z_kind": "periodic"},
    {"label": "m0_corpus_r150_z300_metallic", "shape": (150, 1, 300), "m": 0,
     "z_kind": "metallic"},
)

#: 0.5 is exactly representable in float32 and distributing ``dtdx`` over a difference
#: is then EXACT, so only 0.35 can distinguish a contracted expression from an
#: uncontracted one or a distributed ``dtdx`` from a factored one. The Courant also
#: moves dt, which moves every PML coefficient and the i*m/r row, so it is a real
#: second draw of the tables.
COURANTS: Tuple[float, ...] = (0.35, 0.5)
INEXACT_COURANT = 0.35

#: ``zero_init`` and ``signed_zero`` are the two classes on which a signed zero can
#: change a stored word -- the zero cross terms and the axis-increment replacement live
#: there and NOWHERE else. Both run with a THIN absorber (see ABSORBERS).
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "signed_zero",
                                  "zero_init")
SIGNED_ZERO_CLASSES: Tuple[str, ...] = ("signed_zero", "zero_init")

#: Absorber depth per value class. MEASURED, not chosen: on this family's own grids a
#: quarter-cell absorber pushes ``min kms_z`` POSITIVE, the negative-zero census to 0
#: and every signed-zero mutation to NEEDLE-MISSED -- for a reason that has nothing to
#: do with the mutation. A two-cell absorber holds ``min kms_z`` at about -1.27.
ABSORBERS: Dict[str, str] = {"uniform": "quarter", "subnormal_band": "quarter",
                             "signed_zero": "thin", "zero_init": "thin"}

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

#: The host compiler's flags. ``-O0`` is CORRECTNESS on this family -- see the module
#: docstring. ``-ffp-contract=off`` is the host equivalent of ``--fmad=false`` and
#: ``-fno-fast-math`` keeps ``z.im * 0.0f`` alive.
_HOST_OPTIONS: Tuple[str, ...] = ("-O0", "-ffp-contract=off", "-fno-fast-math")

#: What the host compiler needs in place of CUDA. Extends ``gate_cuda_complex``'s
#: shim with nothing: this family emits no intrinsic that one does not already carry.
_HOST_SHIM = complex_gate._HOST_SHIM


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__`` -- the one thing a laptop cannot supply."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def build(xp, spec: Dict[str, Any], courant: float, absorber: str):
    """A frozen ``(fields, layer, grid)`` triple for one complex Dcyl case."""
    shape = tuple(int(n) for n in spec["shape"])
    grid = Grid(resolution=1.0,
                cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=int(spec["m"]),
                boundaries={"z": spec["z_kind"]},
                accurate_fields_near_cylorigin=bool(spec.get("accurate", False)),
                courant=float(courant), xp=xp)
    if tuple(grid.shape) != shape:
        raise ValueError(f"grid built {tuple(grid.shape)}, case asked for {shape}")
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    if absorber == "thin":
        thickness = {"x": (0, 2), "z": 2}
    else:
        thickness = {"x": (0, max(2, shape[0] // 4)), "z": max(2, shape[2] // 4)}
    layer = PML(grid=grid, thickness=thickness)
    return fields, layer, grid


def seed_state(fields, grid, value_class: str, rng) -> None:
    """Seed every complex volume either sub-step touches.

    THE TWO PLANES ARE ASSIGNED, NEVER COMBINED AS ``re + 1j*im``. Measured by the
    sibling tranche before it was reasoned about: ``1j * im`` carries a real part of
    ``0.0 * im``, so ``re + 1j*im`` computes ``(-0.0) + (+0.0) = +0.0`` and DESTROYS
    every negative zero in the plane the zero cross terms act on.

    THE AUXILIARIES START NONZERO in the live classes: a zero ``fu`` makes
    ``kms * prev`` exactly zero on the first launch whatever ``kms`` holds, so a
    mis-indexed coefficient would only show from launch two. ``zero_init`` is the
    exception and is the whole point of that class -- from all-zero state the ONLY
    thing that can move a word is a signed zero, and the array path does move them.
    """
    xp = grid.xp
    shape = tuple(grid.shape)
    for name in STATE:
        if value_class == "uniform":
            real = rng.uniform(-1.0, 1.0, size=shape)
            imag = rng.uniform(-1.0, 1.0, size=shape)
        elif value_class == "subnormal_band":
            real = probe.subnormal_band_hosts((name,), shape, rng)[name]
            imag = probe.subnormal_band_hosts((name,), shape, rng)[name]
        elif value_class == "zero_init":
            real = imag = np.zeros(shape, dtype=np.float32)
        elif value_class == "signed_zero":
            values = rng.uniform(-1.0, 1.0, size=shape)
            zeroed = rng.integers(0, 2, size=shape) == 0
            signs = np.where(rng.integers(0, 2, size=shape) == 0, -0.0, 0.0)
            real = np.where(zeroed, signs, values)
            imag = rng.uniform(-1.0, 1.0, size=shape)
        else:
            raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
        host = np.empty(shape, dtype=np.complex64)
        host.real = np.asarray(real, dtype=np.float32)
        host.imag = np.asarray(imag, dtype=np.float32)
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(host))


def snapshot(fields) -> Dict[str, np.ndarray]:
    return {name: np.ascontiguousarray(to_host(getattr(fields, name))).copy()
            for name in STATE}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, sub_step: str) -> None:
    """Move the curl operands between launches, identically on both paths."""
    _targets, sources = family.CURL_ARRAYS[sub_step]
    for name in sources:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# Word arithmetic
# ---------------------------------------------------------------------------

def _words(array) -> np.ndarray:
    return np.ascontiguousarray(array).view(np.float32).ravel().view(np.uint32)


def differing_words(a, b) -> int:
    return int(np.count_nonzero(_words(a) != _words(b)))


def negative_zero_census(arrays: Sequence[np.ndarray]) -> int:
    return int(sum(np.count_nonzero(_words(a) == 0x80000000) for a in arrays))


# ---------------------------------------------------------------------------
# The two backends
# ---------------------------------------------------------------------------

_SIGNATURE = re.compile(r'extern "C" __global__ void (\w+)\(([^)]*)\)', re.S)


def parse_signature(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    match = _SIGNATURE.search(text)
    if match is None:
        raise ValueError('no extern "C" __global__ kernel found in the source')
    parameters = []
    for raw in match.group(2).split(","):
        cleaned = " ".join(raw.replace("__restrict__", "").split())
        kind, identifier = cleaned.rsplit(" ", 1)
        parameters.append((kind.strip(), identifier.lstrip("*")))
    return match.group(1), parameters


class _Backend:
    """The two things a backend must do, and the one thing only one of them can."""

    certifies = False
    name = "none"

    def set_guard(self, options: Sequence[str]) -> Dict[str, Any]:
        raise NotImplementedError

    def set_source_transform(self, transform) -> int:
        raise NotImplementedError

    def launch(self, sub_step, fields, layer, grid, arm, prefix, **overrides) -> None:
        raise NotImplementedError

    def restore(self) -> None:
        pass


class KernelBackend(_Backend):
    """NVRTC + the GPU. The only backend that can certify anything."""

    certifies = True
    name = "cuda"

    def __init__(self) -> None:
        self.module = family
        self._original_source = family.cylindrical_complex_source
        self._original_options = family._COMPILE_OPTIONS

    def set_guard(self, options: Sequence[str]) -> Dict[str, Any]:
        family._COMPILE_OPTIONS = tuple(options)
        family._clear_kernel_cache()
        return {"options": list(options)}

    def set_source_transform(self, transform) -> int:
        """Install a text mutation on the EMITTED source; returns the site count.

        Monkeypatching the emitter rather than a string constant is what an emitted
        family forces, and it is the stronger arrangement: the mutation is applied to
        the SAME text the shipped path emits, per sub-step and per arm, so a leg
        cannot mutate one specialization and score another.
        """
        family._clear_kernel_cache()
        if transform is None:
            family.cylindrical_complex_source = self._original_source
            return 0
        sites = 0
        for sub_step in SUB_STEPS:
            _text, count = transform(self._original_source(sub_step, "FMA_V1"))
            sites = max(sites, count)

        def mutated(sub_step, expansion, _original=self._original_source):
            text, _count = transform(_original(sub_step, expansion))
            return text

        family.cylindrical_complex_source = mutated
        return sites

    def launch(self, sub_step, fields, layer, grid, arm, prefix, **overrides) -> None:
        # THE LAUNCHER REFUSES TWO ANSWERS TO ONE QUESTION -- pass ``pml`` AND
        # ``tables`` and it raises, deliberately (a gate that could hand it both would
        # not know which one the kernel read). So a host mutation that supplies an
        # override REPLACES the object it derives from, rather than shadowing it.
        arguments: Dict[str, Any] = dict(overrides)
        if "tables" not in arguments:
            arguments["pml"] = layer
        if "boundary_codes" in arguments:
            # Without a grid the launcher derives nothing, so everything it would have
            # derived is supplied here from the same functions the predicate asks.
            dtdx = float(grid.dt / grid.dx)
            arguments.setdefault("dtdx", dtdx)
            arguments.setdefault("m_class_code", family.m_class(int(grid.m)))
            arguments.setdefault("zero_rows_count", family.zero_rows(
                int(grid.m), bool(grid.accurate_fields_near_cylorigin)))
            arguments.setdefault("imr_rows", family.imr_rows_for(
                sub_step, grid.xp, int(grid.m), dtdx, int(grid.shape[0]),
                getattr(fields, family.CURL_ARRAYS[sub_step][0][0]).dtype))
            arguments.setdefault("increment_scalars",
                                 family.axis_increment_scalars(int(grid.m), dtdx))
        else:
            arguments["grid"] = grid
        family.step_cylindrical_complex(sub_step, fields, arm, prefix=prefix,
                                        **arguments)
        cp.cuda.runtime.deviceSynchronize()

    def restore(self) -> None:
        family.cylindrical_complex_source = self._original_source
        family._COMPILE_OPTIONS = self._original_options
        family._clear_kernel_cache()


class HostCompiledBackend(_Backend):
    """The emitted source, compiled by the HOST C++ compiler and driven on NumPy.

    CERTIFIES NOTHING about NVRTC and says so in the artifact. What it measures is the
    TRANSCRIPTION, executed from the actual emitted characters at the merge bar.
    """

    certifies = False
    name = "host"

    def __init__(self, compiler: str) -> None:
        self.compiler = compiler
        self.options = _HOST_OPTIONS
        self._transform = None
        self._cache: Dict[str, Any] = {}
        self._workdir = tempfile.mkdtemp(prefix="gate_cuda_cylcx_")

    def set_guard(self, options: Sequence[str]) -> Dict[str, Any]:
        # The NVRTC guard names have no host equivalent; the host leg always compiles
        # with contraction off, and the guard CONTROL is a device-only measurement.
        # Recorded rather than silently ignored.
        return {"options": list(self.options),
                "note": "the NVRTC guard is not a host flag; this leg is fixed"}

    def set_source_transform(self, transform) -> int:
        self._transform = transform
        self._cache.clear()
        if transform is None:
            return 0
        sites = 0
        for sub_step in SUB_STEPS:
            _text, count = transform(family.cylindrical_complex_source(sub_step, "FMA_V1"))
            sites = max(sites, count)
        return sites

    def _entry(self, sub_step: str, arm):
        text = family.cylindrical_complex_source(sub_step, arm)
        if self._transform is not None:
            text, _count = self._transform(text)
        key = hashlib.sha256((text + str(self.options)).encode()).hexdigest()[:24]
        if key in self._cache:
            return self._cache[key]
        name, parameters = parse_signature(text)
        declaration = ", ".join(f"{kind} {ident}" for kind, ident in parameters)
        call = ", ".join(ident for _kind, ident in parameters)
        driver = (f'\nextern "C" void launch_{name}(int nblocks, int nthreads, '
                  f'{declaration}) {{\n'
                  f'    blockDim.x = nthreads;\n'
                  f'    for (int b = 0; b < nblocks; ++b) {{\n'
                  f'        blockIdx.x = b;\n'
                  f'        for (int t = 0; t < nthreads; ++t) {{\n'
                  f'            threadIdx.x = t;\n'
                  f'            {name}({call});\n'
                  f'        }}\n    }}\n}}\n')
        stem = os.path.join(self._workdir, key)
        with open(stem + ".cpp", "w", encoding="ascii") as handle:
            handle.write(_HOST_SHIM + text + driver)
        subprocess.run([self.compiler, *self.options, "-shared", "-fPIC",
                        "-o", stem + ".so", stem + ".cpp"], check=True,
                       capture_output=True)
        function = getattr(ctypes.CDLL(stem + ".so"), f"launch_{name}")
        argtypes: List[Any] = [ctypes.c_int, ctypes.c_int]
        for kind, _ident in parameters:
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
        self._cache[key] = function
        return function

    @staticmethod
    def _pointer(array):
        return np.ascontiguousarray(array).ctypes.data_as(ctypes.c_void_p)

    @staticmethod
    def _word_pointer(array):
        if not array.flags.c_contiguous:
            raise ValueError("the host leg needs C-contiguous complex volumes")
        return array.view(np.float32).ctypes.data_as(ctypes.c_void_p)

    def launch(self, sub_step, fields, layer, grid, arm, prefix, **overrides) -> None:
        function = self._entry(sub_step, arm)
        targets, sources = family.CURL_ARRAYS[sub_step]
        shape = tuple(int(n) for n in grid.shape)
        dtdx = overrides.get("dtdx") or float(grid.dt / grid.dx)
        tables = overrides.get("tables")
        if tables is None:
            suffix = "_h" if family.HALF_INTEGER[sub_step] else ""
            tables = {f"{stem}_{axis}": np.ascontiguousarray(
                getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1))
                for axis in "xyz" for stem in ("kms", "sinv")}
        codes = overrides.get("boundary_codes")
        if codes is None:
            codes = family.cylindrical_complex_boundary_codes(grid)
        imr = overrides.get("imr_rows")
        if imr is None:
            imr = family.imr_rows_for(sub_step, np, int(grid.m), dtdx, shape[0],
                                      np.complex64)
        increment = overrides.get("increment_scalars")
        if increment is None:
            increment = family.axis_increment_scalars(int(grid.m), dtdx)
        m_class = overrides.get("m_class_code")
        if m_class is None:
            m_class = family.m_class(int(grid.m))
        zero_rows = overrides.get("zero_rows_count")
        if zero_rows is None:
            zero_rows = family.zero_rows(
                int(grid.m), bool(grid.accurate_fields_near_cylorigin))
        minus_dtdx, (inc_re, inc_im) = increment

        arguments: List[Any] = [self._word_pointer(getattr(fields, n)) for n in targets]
        arguments += [self._word_pointer(getattr(fields, "fu_" + n)) for n in targets]
        arguments += [self._word_pointer(getattr(fields, n)) for n in sources]
        arguments += [self._word_pointer(np.ascontiguousarray(prefix))]
        arguments += [self._word_pointer(np.ascontiguousarray(row)) for row in imr]
        arguments += [int(shape[0]), int(shape[1]), int(shape[2]),
                      ctypes.c_float(float(dtdx)), ctypes.c_float(float(minus_dtdx)),
                      ctypes.c_float(float(inc_re)), ctypes.c_float(float(inc_im))]
        if sub_step == "step_D":
            axis_coef = overrides.get("axis_coef")
            if axis_coef is None:
                axis_coef = family.axis_coefficient(dtdx)
            arguments += [ctypes.c_float(float(axis_coef))]
        for axis in "xyz":
            arguments += [self._pointer(tables[f"kms_{axis}"]),
                          self._pointer(tables[f"sinv_{axis}"])]
        arguments += [int(code) for code in codes] + [0, 0, 0]
        arguments += [int(m_class), int(zero_rows)]
        cells = shape[0] * shape[1] * shape[2]
        function((cells + 255) // 256, 256, *arguments)

    def restore(self) -> None:
        self._transform = None
        self._cache.clear()
        shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = tempfile.mkdtemp(prefix="gate_cuda_cylcx_")


# ---------------------------------------------------------------------------
# The expansion licence
# ---------------------------------------------------------------------------

def load_licence(path: Optional[str], policy: Optional[str]) -> Dict[str, Any]:
    """Read the probe artifact and run the licence. THE ARBITER IS NOT REIMPLEMENTED."""
    out: Dict[str, Any] = {"path": path, "policy_required": policy,
                           "record_sha256": None, "verdict": None,
                           "arm": None, "expansion": None, "usable": False,
                           "policy_reasons": []}
    if not path:
        path = os.environ.get(complex_fields.PROBE_PATH_ENVIRONMENT)
        out["path"] = path
    if not path:
        out["policy_reasons"] = [
            "no expansion probe artifact was given and "
            f"{complex_fields.PROBE_PATH_ENVIRONMENT} is unset; the arm is a measured "
            "platform fact and may not be guessed"]
        return out
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        out["policy_reasons"] = [f"the probe artifact could not be read: {exc}"]
        return out
    out["record_sha256"] = hashlib.sha256(raw).hexdigest()
    record = json.loads(raw.decode("utf-8"))
    verdict = complex_fields.expansion_license(record)
    out["verdict"] = verdict
    out["arm"] = verdict.get("arm")
    out["expansion"] = verdict.get("expansion")
    out["policy_reasons"] = list(
        complex_fields.expansion_policy_reasons(record, policy)) if policy else []
    out["usable"] = bool(verdict.get("arm") and not verdict.get("refusals")
                         and not out["policy_reasons"])
    return out


def other_arm(arm: str) -> str:
    return "NAIVE" if arm == "FMA_V1" else "FMA_V1"


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def oracle_moved(before, after, sub_step: str) -> float:
    """Fraction of output WORDS the array path changed, compared as raw uint32."""
    moved = total = 0
    for name in outputs(sub_step):
        a, b = _words(before[name]), _words(after[name])
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


#: Which rows each per-|m| rule touches, and on which targets. Used by the
#: ``axis_row_is_live`` floor: a rule applied to rows the oracle never moves is a rule
#: the case cannot score, and its mutation would come back UNCAUGHT for a reason about
#: the fixture.
def axis_rows_touched(m: int, accurate: bool, sub_step: str) -> Tuple[Tuple[str, ...], int]:
    if int(m) == 0:
        # The m = 0 tail writes Bx at r = 0 (B) and Dz (post-add) and Dy (zero) at
        # r = 0 (D) -- stepping._cylindrical_axis_zero_B :657-659 / _D :583-587.
        targets = ("Bx",) if sub_step == "step_B" else ("Dy", "Dz")
        return targets, 1
    if abs(int(m)) == 1:
        # The curl-row replacement writes target 0 (B) / target 1 (D) at r = 0, and
        # the D side additionally zeroes Dz there.
        targets = ("Bx",) if sub_step == "step_B" else ("Dy", "Dz")
        return targets, 1
    targets = outputs(sub_step)
    return targets, family.zero_rows(m, accurate)


def axis_row_is_live(frozen, reference, sub_step: str, shape, m: int,
                     accurate: bool) -> Dict[str, Any]:
    targets, rows = axis_rows_touched(m, accurate, sub_step)
    out: Dict[str, Any] = {"targets": list(targets), "rows": int(rows),
                           "moved": 0, "words": 0}
    for target in targets:
        before = np.ascontiguousarray(frozen[target])[:rows]
        after = np.ascontiguousarray(reference[target])[:rows]
        out["moved"] += int(np.count_nonzero(_words(before) != _words(after)))
        out["words"] += int(_words(before).size)
    out["meets_floor"] = out["moved"] > 0
    return out


def prefix_is_live(prefix, sub_step: str) -> Dict[str, Any]:
    """Is the prefix non-constant down r? A flat one hides every prefix defect."""
    host = np.ascontiguousarray(to_host(prefix))
    diffs = host[1:] - host[:-1]
    return {"rows": int(host.shape[0]),
            "distinct_radial_differences": int(np.count_nonzero(diffs)),
            "prefixed_target": "Bz" if sub_step == "step_B" else "Dz",
            "meets_floor": int(np.count_nonzero(diffs)) > 0}


def imr_is_live(rows: Sequence[Any]) -> Dict[str, Any]:
    """Are the two bound coefficient rows non-constant down r and not the identity?

    A constant row makes the per-target Yee shift and the row index invisible, and a
    zero row makes the whole i*m/r term invisible.
    """
    out: Dict[str, Any] = {"rows": []}
    live = True
    for index, row in enumerate(rows):
        host = np.ascontiguousarray(to_host(row))
        distinct = int(np.unique(_words(host)).size)
        magnitude = float(np.max(np.abs(host))) if host.size else 0.0
        out["rows"].append({"index": index, "distinct_words": distinct,
                            "max_abs": magnitude})
        live = live and distinct > 2 and magnitude > 0.0
    out["meets_floor"] = live
    return out


def absorbs(sub_step: str, layer) -> Dict[str, Any]:
    """Do the r and z coefficient profiles differ from the identity?"""
    suffix = "_h" if family.HALF_INTEGER[sub_step] else ""
    out: Dict[str, Any] = {"axes": [], "min_deviation": None, "min_kms_z": None}
    for axis, role in (("x", "r"), ("z", "z")):
        deviation = 0.0
        for stem in ("kms", "sinv"):
            values = to_host(getattr(layer, f"{stem}_{axis}{suffix}")).astype(np.float64)
            deviation = max(deviation, float(np.max(np.abs(values - 1.0))))
        out["axes"].append({"axis": axis, "role": role,
                            "max_deviation_from_identity": deviation})
        out["min_deviation"] = (deviation if out["min_deviation"] is None
                                else min(out["min_deviation"], deviation))
    out["min_kms_z"] = float(np.min(to_host(getattr(layer, f"kms_z{suffix}"))))
    out["meets_floor"] = (out["min_deviation"] or 0.0) > 0.0
    return out


def structure_facts(grid) -> Dict[str, Any]:
    return {
        "shape": [int(n) for n in grid.shape],
        "cylindrical": bool(getattr(grid, "cylindrical", False)),
        "m": int(getattr(grid, "m", 0)),
        "m_class": family.m_class(int(grid.m)),
        "zero_rows": family.zero_rows(int(grid.m),
                                      bool(grid.accurate_fields_near_cylorigin)),
        "accurate_fields_near_cylorigin": bool(grid.accurate_fields_near_cylorigin),
        "is_axis": [bool(grid.is_axis(a)) for a in range(3)],
        "boundary_kinds": list(stepping._boundary_kinds(grid, None)),
    }


# ---------------------------------------------------------------------------
# The two array-path controls
# ---------------------------------------------------------------------------

def axis_identity_leg(rng) -> Dict[str, Any]:
    """Is the CYL_AXIS ghost rule the METALLIC one, on real grids?

    The kernel compiles ``bc_x = BC_METALLIC``. This runs the ARRAY PATH twice on the
    same frozen state -- once as it stands, once with ``_boundary_kinds`` forced to
    report ``'metallic'`` on r -- and requires ZERO differing words. That is what
    licenses the constant, and it is measured rather than read off
    ``_shift_up``'s source.
    """
    out: Dict[str, Any] = {"rows": [], "differing_words": 0, "cases": 0}
    original = stepping._boundary_kinds

    def forced(grid, pml=None):
        kinds = list(original(grid, pml))
        kinds[0] = "metallic"
        return tuple(kinds)

    for spec in CASES:
        for sub_step in SUB_STEPS:
            fields, layer, grid = build(np, spec, spec.get("courant", INEXACT_COURANT),
                                        "quarter")
            seed_state(fields, grid, "uniform", rng)
            frozen = snapshot(fields)
            getattr(stepping, sub_step)(fields, layer)
            reference = snapshot(fields)
            restore(fields, frozen)
            stepping._boundary_kinds = forced
            try:
                getattr(stepping, sub_step)(fields, layer)
            finally:
                stepping._boundary_kinds = original
            differing = sum(differing_words(reference[n], getattr(fields, n))
                            for n in outputs(sub_step))
            out["rows"].append({"label": spec["label"], "sub_step": sub_step,
                                "differing_words": differing})
            out["differing_words"] += differing
            out["cases"] += 1
    out["meets_floor"] = out["differing_words"] == 0
    out["reading"] = ("the CYL_AXIS ghost IS the METALLIC ghost on every case, which "
                      "is what licenses bc_x = BC_METALLIC"
                      if out["meets_floor"] else
                      "forcing r metallic CHANGED the array path; the kernel's "
                      "compiled boundary code is not the array path's rule")
    return out


def stripped_control(backend, arm, rng) -> Dict[str, Any]:
    """The CERTIFIED complex CARTESIAN body on a Dcyl grid. It MUST diverge.

    A zero here would mean this module should not exist. Run through
    ``complex_emitter``'s own source and ``gate_cuda_complex``'s host driver shape, so
    the comparison is against the certified characters and not a re-transcription.
    """
    out: Dict[str, Any] = {"rows": [], "differing_words": 0, "rows_differing": 0,
                           "cases": 0}
    if backend.name != "host":
        out["skipped"] = ("run on the host leg only: the control's question is about "
                          "the certified body's ARITHMETIC, which is the same "
                          "characters on both backends")
        return out
    host = backend
    for spec in CASES:
        for sub_step in SUB_STEPS:
            fields, layer, grid = build(np, spec, spec.get("courant", INEXACT_COURANT),
                                        "quarter")
            seed_state(fields, grid, "uniform", rng)
            frozen = snapshot(fields)
            getattr(stepping, sub_step)(fields, layer)
            reference = snapshot(fields)
            restore(fields, frozen)
            _launch_certified_cartesian(host, sub_step, fields, layer, grid, arm)
            differing = sum(differing_words(reference[n], getattr(fields, n))
                            for n in outputs(sub_step))
            out["rows"].append({"label": spec["label"], "sub_step": sub_step,
                                "differing_words": differing})
            out["differing_words"] += differing
            out["rows_differing"] += 1 if differing else 0
            out["cases"] += 1
    out["meets_floor"] = out["differing_words"] > 0
    out["reading"] = (
        f"the certified complex CARTESIAN body differs from the array path by "
        f"{out['differing_words']} uint32 words on {out['rows_differing']} of "
        f"{out['cases']} Dcyl cases; the cylindrical kernel is not a widening"
        if out["meets_floor"] else
        "the certified Cartesian body is ALREADY bit-identical on Dcyl, which would "
        "mean this module should not exist")
    return out


_CARTESIAN_CACHE: Dict[str, Any] = {}


def _launch_certified_cartesian(host, sub_step, fields, layer, grid, arm) -> None:
    """Compile and drive ``complex_emitter``'s Cartesian curl on a Dcyl grid."""
    text = complex_emitter.complex_source(sub_step, arm)
    key = hashlib.sha256((text + str(host.options)).encode()).hexdigest()[:24]
    if key not in _CARTESIAN_CACHE:
        name, parameters = parse_signature(text)
        declaration = ", ".join(f"{k} {i}" for k, i in parameters)
        call = ", ".join(i for _k, i in parameters)
        driver = (f'\nextern "C" void launch_{name}(int nblocks, int nthreads, '
                  f'{declaration}) {{\n    blockDim.x = nthreads;\n'
                  f'    for (int b = 0; b < nblocks; ++b) {{\n        blockIdx.x = b;\n'
                  f'        for (int t = 0; t < nthreads; ++t) {{\n'
                  f'            threadIdx.x = t;\n            {name}({call});\n'
                  f'        }}\n    }}\n}}\n')
        stem = os.path.join(host._workdir, "cart_" + key)
        with open(stem + ".cpp", "w", encoding="ascii") as handle:
            handle.write(_HOST_SHIM + text + driver)
        subprocess.run([host.compiler, *host.options, "-shared", "-fPIC",
                        "-o", stem + ".so", stem + ".cpp"], check=True,
                       capture_output=True)
        function = getattr(ctypes.CDLL(stem + ".so"), f"launch_{name}")
        argtypes: List[Any] = [ctypes.c_int, ctypes.c_int]
        for kind, _ident in parameters:
            argtypes.append(ctypes.c_void_p if "*" in kind else
                            (ctypes.c_float if kind == "float" else ctypes.c_int))
        function.argtypes = argtypes
        function.restype = None
        _CARTESIAN_CACHE[key] = function
    function = _CARTESIAN_CACHE[key]
    targets, sources = family.CURL_ARRAYS[sub_step]
    shape = tuple(int(n) for n in grid.shape)
    suffix = "_h" if complex_emitter.HALF_INTEGER[sub_step] else ""
    tables = {f"{stem}_{axis}": np.ascontiguousarray(
        getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1))
        for axis in "xyz" for stem in ("kms", "sinv")}
    # r resolves to 'axis', which the CARTESIAN family has no code for -- so the
    # control passes METALLIC, which is the strictly most favourable choice for it.
    codes = (BC_METALLIC, BC_PERIODIC,
             BC_METALLIC if stepping._boundary_kinds(grid, None)[2] == "metallic"
             else BC_PERIODIC)
    arguments: List[Any] = [host._word_pointer(getattr(fields, n)) for n in targets]
    arguments += [host._word_pointer(getattr(fields, "fu_" + n)) for n in targets]
    arguments += [host._word_pointer(getattr(fields, n)) for n in sources]
    arguments += [int(shape[0]), int(shape[1]), int(shape[2]),
                  ctypes.c_float(float(grid.dt / grid.dx))]
    for axis in "xyz":
        arguments += [host._pointer(tables[f"kms_{axis}"]),
                      host._pointer(tables[f"sinv_{axis}"])]
    arguments += [int(c) for c in codes] + [0, 0, 0]
    arguments += [ctypes.c_float(1.0), ctypes.c_float(0.0)] * 3
    cells = shape[0] * shape[1] * shape[2]
    function((cells + 255) // 256, 256, *arguments)


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def _sub(pattern: str, replacement: str, text: str) -> Tuple[str, int]:
    out, count = re.subn(pattern, replacement, text)
    return out, count


def m_drop_imr(text):
    """Drop BOTH i*m/r couplings -- the whole |m| >= 1 physics."""
    return _sub(r"\n *curl = cf_sub\(curl, mul_complex_left\([^;]*;", "", text)


def m_imr_after_mask(text):
    """Add the i*m/r term AFTER the ownership mask instead of before it.

    The whole GUARDED block moves (the ``if (m_class != 0) {`` line, the term and its
    closing brace), so the mutated text is still the m = 0 arm's on the m = 0 cases
    and the defect lives at |m| >= 1 exactly as before the guard landed.
    """
    out, count = _sub(
        r"( *if \(m_class != 0\) \{\n *curl = cf_sub\(curl, mul_complex_left\([^;]*;\n"
        r" *\}\n)"
        r"((?: *if \(bc_[xyz] == BC_METALLIC && \w == 0\) curl = cf_zero\(\);\n)+)",
        r"\2\1", text)
    return out, count


def m_m0_drop_bx_axis_zero(text):
    """Drop the m = 0 B-side tail: Br keeps the recurrence value on the axis row."""
    return _sub(r" *if \(m_class == 0 && i == 0\) \{\n"
                r" *cf_store\(Bx, idx, cf_zero\(\)\);\n *\}\n", "", text)


def m_m0_drop_dz_axis_post_add(text):
    """Drop the m = 0 on-axis Dz post-add (stepping.py:585); Dy's zero stays."""
    return _sub(r" *cf_store\(Dz, idx, cf_add\(cf_load\(Dz, idx\),\n"
                r" *mul_coefficient_left\(axis_coef, cf_load\(Hy, idx\)\)\)\);\n",
                "", text)


def m_m0_drop_dy_axis_zero(text):
    """Drop Dp = 0 on the axis at m = 0 (stepping.py:586); the post-add stays."""
    return _sub(r"(if \(m_class == 0 && i == 0\) \{\n[^}]*\)\);\n)"
                r" *cf_store\(Dy, idx, cf_zero\(\)\);\n", r"\1", text)


def m_m0_post_add_reads_prefixed_hp(text):
    """Feed the post-add the PREFIXED Hp instead of the raw stored one."""
    return _sub(r"mul_coefficient_left\(axis_coef, cf_load\(Hy, idx\)\)",
                "mul_coefficient_left(axis_coef, cf_load(pfx, idx))", text)


def m_m0_post_add_folded_before_dy_zero_swapped(text):
    """Zero Dp BEFORE the post-add -- the same two stores in the other order.

    They touch different volumes, so this is a PREDICTED NULL: order-independent
    stores are order-independent bytes, and a leg that caught it would mean the
    harness reports differences that are not arithmetic.
    """
    return _sub(r"( *)(cf_store\(Dz, idx, cf_add\(cf_load\(Dz, idx\),\n"
                r" *mul_coefficient_left\(axis_coef, cf_load\(Hy, idx\)\)\)\);\n)"
                r"( *cf_store\(Dy, idx, cf_zero\(\)\);\n)", r"\3\1\2", text)


def m_m0_imr_unguarded(text):
    """Subtract the (zero) i*m/r rows at m = 0 instead of skipping them.

    MEASURED, NOT REQUIRED. The array path adds no term at m = 0 (stepping.py:376,
    :459). ``curl - (0 * f)`` is the identity on every word except a ``-0.0`` curl
    meeting a ``-0.0`` subtrahend, so on ordinary values this leg is a null and on
    the signed-zero classes it may move a word; the guard is kept because the array
    path SKIPS the term, and the reading records what the unguarded form costs.
    """
    return _sub(r"if \(m_class != 0\) \{", "if (1) {", text)


def m_imr_partner_swapped(text):
    """Target 0's coupling takes the OTHER center load -- a real mis-wiring."""
    return _sub(r"mul_complex_left\(cf_load\(imr0, i\), f_1\)",
                "mul_complex_left(cf_load(imr0, i), f_2)", text)


def m_imr_row_indexed_by_k(text):
    """Index the per-r row by the z lane: the wrong radius on every cell."""
    return _sub(r"cf_load\(imr(\d), i\)", r"cf_load(imr\1, k)", text)


def m_imr_added_not_subtracted(text):
    """``curl + m`` instead of ``curl - m``: the sign of the whole coupling."""
    return _sub(r"curl = cf_sub\(curl, (mul_complex_left\([^;]*)\);",
                r"curl = cf_add(curl, \1);", text)


def m_drop_axis_increment(text):
    return _sub(r" *curl\.re = inc\.re \* -1\.0f;\n *curl\.im = inc\.im \* -1\.0f;\n",
                "", text)


def m_axis_increment_accumulates(text):
    """REPLACE vs ACCUMULATE -- byte-visible only on a stored signed zero."""
    return _sub(r"curl\.re = inc\.re \* -1\.0f;\n( *)curl\.im = inc\.im \* -1\.0f;",
                r"curl.re = curl.re + inc.re * -1.0f;\n"
                r"\1curl.im = curl.im + inc.im * -1.0f;", text)


def m_axis_increment_not_negated(text):
    return _sub(r"curl\.re = inc\.re \* -1\.0f;\n( *)curl\.im = inc\.im \* -1\.0f;",
                r"curl.re = inc.re;\n\1curl.im = inc.im;", text)


def m_axis_increment_operands_swapped(text):
    """``-(A - B)`` is NOT ``B - A`` when both are +0.0 (grouping choice)."""
    out, count = _sub(r"mul_coefficient_left\(minus_dtdx, cf_sub\(f_2, ss\)\)",
                      "mul_coefficient_left(minus_dtdx, cf_sub(ss, f_2))", text)
    if count:
        return out, count
    return _sub(r"cf_sub\(cf_sub\(f_1, sf\), two_c\)",
                "cf_sub(cf_sub(sf, f_1), two_c)", text)


def m_axis_increment_reads_row0_ez(text):
    """Read the AXIS row of Ez instead of the FIRST OFF-AXIS row (stepping.py:671)."""
    return _sub(r"cf_load\(Ez, sx \+ j \* sy \+ k\)", "cf_load(Ez, j * sy + k)", text)


def m_axis_increment_before_mask(text):
    """Apply the |m| = 1 replacement BEFORE the mask, so the mask wipes it."""
    return _sub(
        r"((?: *if \(bc_[xyz] == BC_METALLIC && \w == 0\) curl = cf_zero\(\);\n)+)"
        r"( *// stepping\.py:[^\n]*\n(?: *//[^\n]*\n)* *if \(m_class == 1 && i == 0\) \{"
        r"(?:[^}]*)\}\n)",
        r"\2\1", text)


def m_drop_near_axis_zeroing(text):
    return _sub(r" *if \(m_class == 2 && i < zero_rows\) \{[^}]*\}\n", "", text)


def m_near_axis_leaves_fu(text):
    """Zero the fields but not their auxiliaries: right for one launch, wrong after."""
    return _sub(r"( *)cf_store\(fu_\wx, idx, cf_zero\(\)\);\n"
                r" *cf_store\(fu_\wy, idx, cf_zero\(\)\);\n"
                r" *cf_store\(fu_\wz, idx, cf_zero\(\)\);\n", "", text)


def m_near_axis_row_zero_only(text):
    """Hold only row 0 at zero -- i.e. the ACCURATE branch on every grid."""
    return _sub(r"if \(m_class == 2 && i < zero_rows\)",
                "if (m_class == 2 && i < 1)", text)


def m_drop_dz_axis_zero(text):
    return _sub(r" *if \(m_class == 1 && i == 0\) \{\n"
                r" *cf_store\(Dz, idx, cf_zero\(\)\);\n *\}\n", "", text)


def m_dz_axis_zero_takes_fu(text):
    """Zero ``fu_Dz`` too -- the array path zeroes the FIELD ONLY at |m| = 1."""
    return _sub(r"(if \(m_class == 1 && i == 0\) \{\n *cf_store\(Dz, idx, cf_zero\(\)\);\n)",
                r"\1        cf_store(fu_Dz, idx, cf_zero());\n", text)


def m_drop_bz_prefix_substitution(text):
    """Bz's curl becomes the four-operand Cartesian grouping again."""
    return _sub(
        r"cf pfx_here = cf_load\(pfx, idx\);\n"
        r" *cf pfx_up = cf_load\(pfx, idx \+ sx\);\n"
        r" *cf curl = mul_coefficient_left\(dtdx, cf_sub\(pfx_up, pfx_here\)\);",
        "cf f_1 = cf_load(Ey, idx);\n"
        "        cf sf = cshift_up(Ey, idx, i, nx, sx, bc_x, ph_x, noph);\n"
        "        cf f_2 = cf_load(Ex, idx);\n"
        "        cf ss = cshift_up(Ex, idx, j, ny, sy, bc_y, ph_y, noph);\n"
        "        cf curl = mul_coefficient_left(dtdx, "
        "cf_add(cf_sub(sf, f_1), cf_sub(f_2, ss)));", text)


def m_bz_flat_grouping(text):
    """``dtdx*a - dtdx*b`` instead of ``dtdx*(a-b)`` -- exact at 0.5, wrong at 0.35."""
    return _sub(
        r"cf curl = mul_coefficient_left\(dtdx, cf_sub\(pfx_up, pfx_here\)\);",
        "cf curl = cf_sub(mul_coefficient_left(dtdx, pfx_up), "
        "mul_coefficient_left(dtdx, pfx_here));", text)


def m_bz_prefix_reads_same_row(text):
    """Drop the wall row's reach: the forward difference becomes zero."""
    return _sub(r"cf pfx_up = cf_load\(pfx, idx \+ sx\);",
                "cf pfx_up = cf_load(pfx, idx);", text)


def m_drop_dz_prefix_substitution(text):
    return _sub(r"cf f_1 = cf_load\(pfx, idx\);\n"
                r"( *)cf sf = cshift_dn\(pfx, idx, i, nx, sx, bc_x, ph_x, noph\);",
                r"cf f_1 = cf_load(Hy, idx);\n"
                r"\1cf sf = cshift_dn(Hy, idx, i, nx, sx, bc_x, ph_x, noph);", text)


def m_prefix_into_dx(text):
    """Leak the prefix into Dx, whose Hy operands must stay RAW."""
    return _sub(r"cf f_2 = cf_load\(Hy, idx\);\n"
                r"( *)cf ss = cshift_dn\(Hy, idx, k, nz, sz, bc_z, ph_z, noph\);",
                r"cf f_2 = cf_load(pfx, idx);\n"
                r"\1cf ss = cshift_dn(pfx, idx, k, nz, sz, bc_z, ph_z, noph);", text)


def _r_near_ghost(variable: str, volume: str, sign: str):
    """Serve the array path's CYL_AXIS near ghost instead of the metallic zero.

    ``stepping._shift_down``'s CYL_AXIS branch (:1830-1846) images stored row 0 with a
    direction sign and the ``(-1)^m`` phase. Both the imaged value and its NEGATION are
    armed, because the claim is that the row is UNOBSERVABLE and neither may move a
    word. If either is ever CAUGHT, the ownership mask has broken, not the ghost.
    """
    call = f"cf {variable} = cshift_dn({volume}, idx, i, nx, sx, bc_x, ph_x, noph);"
    replacement = (
        f"cf {variable};\n"
        f"        if (i == 0) {{ {variable} = cf_load({volume}, idx); "
        f"{variable}.re = {variable}.re * {sign}f; "
        f"{variable}.im = {variable}.im * {sign}f; }}\n"
        f"        else {{ {variable} = cshift_dn({volume}, idx, i, nx, sx, bc_x, "
        f"ph_x, noph); }}")

    def transform(text):
        if call not in text:
            return text, 0
        return text.replace(call, replacement), 1

    return transform


def n_reload_the_target_from_memory(text):
    """A NULL: store, load back, store again. A float32 round trip is the identity."""
    return _sub(r"( *)pml_apply\((\w+), (\w+), idx, curl,",
                r"\1cf_store(\2, idx, cf_load(\2, idx));\n"
                r"\1pml_apply(\2, \3, idx, curl,", text)


def n_commute_the_imr_subtrahend(text):
    """A NULL: ``a - b`` with the SAME operands, spelled through a named temporary.

    IEEE subtraction is not commutative, so this does NOT swap them; it only moves the
    product into a local. A leg that CAUGHT this would mean the harness reports
    differences that are not arithmetic.
    """
    return _sub(r"curl = cf_sub\(curl, (mul_complex_left\([^;]*)\);",
                r"{ cf mterm = \1; curl = cf_sub(curl, mterm); }", text)


#: The family's own battery. ``reach`` names the cases the defect can be seen on and
#: is DERIVED, not guessed -- see :func:`reaches`. ``must`` is True for a catch and
#: False for a predicted null.
OWN_MUTATIONS: Dict[str, Dict[str, Any]] = {
    # THE i*m/r BATTERY REACHES |m| >= 1 ONLY (2026-09-04): at m = 0 every i*m/r site
    # is guarded off by ``m_class != 0`` -- the arm the array path takes (stepping.py
    # :376, :459) -- so a defect in the term is unreachable there BY DESIGN, and
    # ``m0_imr_unguarded`` below is the leg that measures the guard itself.
    "drop_imr": {"fn": m_drop_imr, "must": True, "reach": "imr_live"},
    "imr_after_mask": {"fn": m_imr_after_mask, "must": True,
                       "reach": "imr_metallic_live"},
    "imr_partner_swapped": {"fn": m_imr_partner_swapped, "must": True, "reach": "imr_live"},
    "imr_row_indexed_by_k": {"fn": m_imr_row_indexed_by_k, "must": True, "reach": "imr_live"},
    "imr_added_not_subtracted": {"fn": m_imr_added_not_subtracted, "must": True,
                                 "reach": "imr_live"},
    "drop_axis_increment": {"fn": m_drop_axis_increment, "must": True,
                            "reach": "m1_live"},
    "axis_increment_accumulates": {"fn": m_axis_increment_accumulates, "must": True,
                                   "reach": "m1_zero_init"},
    "axis_increment_not_negated": {"fn": m_axis_increment_not_negated, "must": True,
                                   "reach": "m1_live"},
    "axis_increment_operands_swapped": {"fn": m_axis_increment_operands_swapped,
                                        "must": True, "reach": "m1_live"},
    "axis_increment_before_mask": {"fn": m_axis_increment_before_mask, "must": True,
                                   "reach": "m1"},
    "drop_near_axis_zeroing": {"fn": m_drop_near_axis_zeroing, "must": True,
                               "reach": "mmany_live"},
    "near_axis_leaves_fu": {"fn": m_near_axis_leaves_fu, "must": True, "reach": "mmany_live"},
    "near_axis_row_zero_only": {"fn": m_near_axis_row_zero_only, "must": True,
                                "reach": "mmany_wide_live"},
    "drop_bz_prefix_substitution": {"fn": m_drop_bz_prefix_substitution, "must": True,
                                    "reach": "live_state", "sub_step": "step_B"},
    "bz_flat_grouping": {"fn": m_bz_flat_grouping, "must": True,
                         "reach": "inexact_courant", "sub_step": "step_B"},
    "bz_prefix_reads_same_row": {"fn": m_bz_prefix_reads_same_row, "must": True,
                                 "reach": "live_state", "sub_step": "step_B"},
    "axis_increment_reads_row0_ez": {"fn": m_axis_increment_reads_row0_ez, "must": True,
                                     "reach": "m1_live", "sub_step": "step_B"},
    "drop_dz_prefix_substitution": {"fn": m_drop_dz_prefix_substitution, "must": True,
                                    "reach": "live_state", "sub_step": "step_D"},
    "prefix_into_dx": {"fn": m_prefix_into_dx, "must": True, "reach": "live_state",
                       "sub_step": "step_D"},
    "drop_dz_axis_zero": {"fn": m_drop_dz_axis_zero, "must": True, "reach": "m1_live",
                          "sub_step": "step_D"},
    "dz_axis_zero_takes_fu": {"fn": m_dz_axis_zero_takes_fu, "must": True, "reach": "m1_live",
                              "sub_step": "step_D"},
    # THE m = 0 ARM'S OWN BATTERY (2026-09-04): each of the array path's three m = 0
    # rules dropped, the post-add fed the wrong Hp, and the guard on the i*m/r sites
    # removed. The store-order swap is a predicted null.
    "m0_drop_bx_axis_zero": {"fn": m_m0_drop_bx_axis_zero, "must": True,
                             "reach": "m0_live", "sub_step": "step_B"},
    "m0_drop_dz_axis_post_add": {"fn": m_m0_drop_dz_axis_post_add, "must": True,
                                 "reach": "m0_live", "sub_step": "step_D"},
    "m0_drop_dy_axis_zero": {"fn": m_m0_drop_dy_axis_zero, "must": True,
                             "reach": "m0_live", "sub_step": "step_D"},
    "m0_post_add_reads_prefixed_hp": {"fn": m_m0_post_add_reads_prefixed_hp,
                                      "must": True, "reach": "m0_live",
                                      "sub_step": "step_D"},
    "m0_dy_zero_before_dz_post_add": {"fn": m_m0_post_add_folded_before_dy_zero_swapped,
                                      "must": False, "reach": "any",
                                      "sub_step": "step_D"},
    "m0_imr_unguarded": {"fn": m_m0_imr_unguarded, "must": "measure", "reach": "m0",
                         "reading": (
                             "the UNGUARDED form -- subtracting the (zero) i*m/r rows "
                             "at m = 0 -- moved a word on {caught} of {live} m = 0 "
                             "cases across all four value classes; the guard is kept "
                             "because the array path SKIPS the term (stepping.py:376, "
                             ":430) and `curl - (0 * f)` is not the identity on a "
                             "-0.0 curl meeting a -0.0 subtrahend, which no seeding "
                             "here produced")},
    # THE TWO r-NEAR-GHOST NULLS, on the D side only. On the B side ``cshift_dn`` is
    # compiled but never called, so a mutation to it there is armed in the TEXT and
    # unreachable in fact -- which is a vacuous leg, not a null.
    "r_near_ghost_dy_image": {"fn": _r_near_ghost("ss", "Hz", "1.0"), "must": False,
                              "reach": "any", "sub_step": "step_D"},
    "r_near_ghost_dy_image_negated": {"fn": _r_near_ghost("ss", "Hz", "-1.0"),
                                      "must": False, "reach": "any",
                                      "sub_step": "step_D"},
    "r_near_ghost_dz_image": {"fn": _r_near_ghost("sf", "pfx", "-1.0"), "must": False,
                              "reach": "any", "sub_step": "step_D"},
    "r_near_ghost_dz_image_negated": {"fn": _r_near_ghost("sf", "pfx", "1.0"),
                                      "must": False, "reach": "any",
                                      "sub_step": "step_D"},
    # THE TWO HARNESS NULLS.
    "reload_the_target_from_memory": {"fn": n_reload_the_target_from_memory,
                                      "must": False, "reach": "any"},
    "commute_the_imr_subtrahend": {"fn": n_commute_the_imr_subtrahend, "must": False,
                                   "reach": "any"},
}

#: The SHARED battery, borrowed from ``gate_cuda_complex`` so this gate cannot own a
#: second spelling of a defect the certified complex record was cut against. Every one
#: of them arms on the IMPORTED ``_HEAD``/``_ARM_SOURCE``/``_TAIL`` strings or on the
#: Cartesian target blocks this pair keeps unchanged -- which is itself the check that
#: "the certified body, character for character" is true rather than claimed.
SHARED_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "fold_the_zero_cross_terms": {"must": True, "reach": "signed_zero"},
    "plane_wise_scaling": {"must": True, "reach": "signed_zero"},
    "left_to_right_curl": {"must": True, "reach": "live_state"},
    "drop_the_ownership_mask": {"must": True, "reach": "mask_observable"},
    "metallic_ghost_wraps": {"must": True, "reach": "metallic_ghost"},
    "hoist_dtdx_into_the_difference": {"must": True, "reach": "inexact_courant"},
    "word_pair_transposed": {"must": True, "reach": "live_state"},
    "column_major_index": {"must": True, "reach": "live_state"},
    "reload_the_target_from_memory": {"must": False, "reach": "any",
                                      "rename": "shared_reload_the_target"},
    "commute_the_plane_sum": {"must": False, "reach": "any"},
}

#: Shared mutations this family EXCLUDES, each with the reason. A mutation that cannot
#: fire and a mutation that is not caught look identical in a verdict, so an
#: unreachable leg is excluded by name rather than scored as a null.
SHARED_EXCLUDED: Dict[str, str] = {
    "flatten_the_constitutive_tail":
        "arms on constitutive_apply, which these curl kernels never call",
    "store_fw_before_reading_prev":
        "arms on constitutive_apply, which these curl kernels never call",
    "phase_on_every_lane":
        "arms on the Bloch rotation; this family passes every phase flag 0 and the "
        "predicate refuses a phase, so the mutated path is unreachable",
    "conjugate_the_rotation":
        "arms on the Bloch rotation; see phase_on_every_lane",
}

#: HOST mutations: nothing in the device text changes, the LAUNCHER is handed
#: something wrong. Each is a real defect a call site can carry.
HOST_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "swap_curl_sublattice": {"must": True, "reach": "live_state"},
    "swap_prefix_ir0": {"must": True, "reach": "live_state"},
    "wall_row_not_zero": {"must": True, "reach": "live_state", "sub_step": "step_B"},
    "imr_iyee_zero": {"must": True, "reach": "imr_live"},
    "imr_rows_swapped": {"must": True, "reach": "imr_live"},
    "imr_clamp_raised": {"must": True, "reach": "m1_live"},
    "zero_rows_forced_one": {"must": True, "reach": "mmany_wide_live"},
    "m_class_forced_one": {"must": True, "reach": "not_m1_live"},
    # THE THIRD ARM FORCED ON THE OTHER TWO: at |m| >= 1 the i*m/r sites go silent,
    # the increments and the near-axis zeroing never fire and the m = 0 tails do.
    "m_class_forced_zero": {"must": True, "reach": "not_m0_live"},
    "the_other_expansion_arm": {"must": "measure", "reach": "arm_separable"},
    "drop_the_metallic_codes": {"must": True, "reach": "boundary_code_observable"},
    # A PREDICTED NULL: the clamp bites only on row 0 of a shift-0 target, whose curl
    # the ownership mask zeroes -- so dropping it cannot move a compared word (the
    # divide-by-zero's non-finite never reaches a compared volume).
    "imr_clamp_dropped": {"must": False, "reach": "any"},
}


def reaches(reach: str, case: Dict[str, Any]) -> bool:
    """Whether this case is one the leg's defect can be seen on at all.

    A REACH IS A CLAIM AND EACH ONE IS DERIVED:

    * ``m1`` / ``mmany`` -- the two axis-rule arms are compiled branches; a mutation
      to one cannot be seen from the other;
    * ``mmany_wide`` -- ``ZERO_ROWS >= 2``, which is the only place "rows [0:|m|]" and
      "row 0 alone" differ;
    * ``metallic`` -- the defect touches the ownership mask or the predicated-zero
      ghost, so a grid with no metallic axis has nothing to see. The r axis is ALWAYS
      metallic here, so this is about z;
    * ``metallic_mask`` -- the mask fires on at least one axis this sub-step masks;
    * ``metallic_ghost`` -- the FAR-face ghost only, which on ``step_D`` is
      structurally unobservable (every down-shifted operand is consumed by the cell
      whose curl the mask drops) EXCEPT through the |m| = 1 axis increment, which reads
      a z-down ghost at a row the mask does not keep;
    * ``signed_zero`` -- the ONE class on which the zero cross terms can change a
      stored word;
    * ``m1_signed_zero`` -- both at once: the axis-increment REPLACE/ACCUMULATE choice
      is byte-visible only on a stored ``-0.0``, and the reachable producer is a
      negative ``kms`` times a quiet ``+0.0``;
    * ``inexact_courant`` -- distributing ``dtdx`` over a difference is EXACT at 0.5;
    * ``arm_separable`` -- the two expansion arms must actually differ on this case.
    """
    if reach == "any":
        return True
    if reach == "live_state":
        # A defect in ORDINARY arithmetic needs ordinary values. Under ``zero_init``
        # every operand is an exact zero and under ``signed_zero`` half of one plane
        # is: the prefix is flat, the i*m/r products are +-0 and a grouping change
        # moves nothing. Those two classes exist for the +-0 defects and are scored on
        # those; scoring an arithmetic leg there would report UNCAUGHT for a reason
        # that belongs to the fixture.
        return (case["value_class"] not in SIGNED_ZERO_CLASSES
                and case.get("prefix_live", True))
    metallic = case["z_kind"] == "metallic"
    m_one = abs(int(case["m"])) == 1
    m_zero = int(case["m"]) == 0
    m_many = abs(int(case["m"])) >= 2
    if reach == "m1":
        return m_one and case.get("axis_row_live", True)
    if reach == "m1_live":
        return m_one and case.get("axis_row_live", True) and reaches("live_state", case)
    if reach == "m0":
        return m_zero
    if reach == "m0_live":
        return m_zero and case.get("axis_row_live", True) and reaches("live_state", case)
    if reach == "not_m0_live":
        return (not m_zero) and reaches("live_state", case)
    if reach == "imr_live":
        # The i*m/r term exists at |m| >= 1 only; at m = 0 the kernel skips it.
        return (not m_zero) and reaches("live_state", case)
    if reach == "imr_metallic_live":
        return (not m_zero) and metallic and reaches("live_state", case)
    if reach == "not_m1_live":
        return (not m_one) and reaches("live_state", case)
    if reach == "mmany":
        return m_many
    if reach == "mmany_live":
        return m_many and reaches("live_state", case)
    if reach == "mmany_wide":
        return m_many and case["zero_rows"] >= 2
    if reach == "mmany_wide_live":
        return (m_many and case["zero_rows"] >= 2
                and reaches("live_state", case))
    if reach == "metallic":
        return metallic
    if reach == "metallic_live":
        return metallic and reaches("live_state", case)
    if reach == "mask_observable":
        # WHICH ROWS THE MASK ACTUALLY DECIDES, derived rather than assumed. It fires
        # on (a) r at i = 0 for every shift-0 target and (b) z at k = 0 when z is
        # metallic. Row 0 of the r-masked targets is then OVERWRITTEN by this family's
        # own rules -- by the axis increment at |m| = 1 (Bx, Dy) and by the near-axis
        # zeroing at |m| >= 2 (all six volumes) -- so on a PERIODIC z the only place
        # the mask still decides a stored word is ``fu_Dz`` at |m| = 1, which the
        # |m| = 1 rule zeroes the FIELD of and not the auxiliary.
        # AT m = 0 the mask is observable on BOTH sides through the auxiliary: the
        # tail zeroes the FIELD only (Bx; Dy) and post-adds into Dz, so fu_Bx / fu_Dy
        # keep the masked recurrence and Dz keeps the masked curl under the add.
        return reaches("live_state", case) and (
            metallic or m_zero or (case["sub_step"] == "step_D" and m_one))
    if reach == "boundary_code_observable":
        # Dropping the metallic CODES changes both the near and the FAR r ghost. The
        # far face (i = nx - 1) is reached only by a forward difference, i.e. by
        # ``step_B``; on ``step_D`` every r ghost is a near one, and those land on row
        # 0, which the per-|m| rules overwrite unless |m| = 1 leaves ``fu_Dz``.
        return reaches("live_state", case) and (
            case["sub_step"] == "step_B" or metallic or m_one or m_zero)
    if reach == "metallic_ghost":
        # THE FAR-face ghost only, which is ``step_B``'s: on ``step_D`` every r and z
        # ghost is a NEAR one, and each is consumed by the cell whose curl the
        # ownership mask drops -- EXCEPT the one cell the |m| = 1 axis increment reads
        # past the mask, ``(i, k) = (0, 0)``, through its z-down operand. A leg whose
        # whole visibility is ONE CELL is scored on the seeding whose values are
        # ordinary: measured, both misses on the full product were ``step_D`` /
        # ``subnormal_band``, where that cell's operands are ~1e-40 and the difference
        # rounds back to the same word.
        if not (metallic and reaches("live_state", case)):
            return False
        if case["sub_step"] == "step_B":
            return True
        return m_one and case["value_class"] == "uniform"
    if reach == "signed_zero":
        return case["value_class"] in SIGNED_ZERO_CLASSES
    if reach == "m1_zero_init":
        # ``zero_init`` ON ``step_D`` ONLY, and both halves are measured.
        #
        # THE CLASS. ``curl + x`` and ``x`` differ only at ``x = -0.0``, so the
        # increment itself must be an EXACT +0.0 before its negation. ``zero_init`` is
        # the seeding that makes it one on every word; under ``signed_zero`` the other
        # operands are live, the increment is generically nonzero at the axis row and
        # the choice moves nothing -- measured, 6 of 6 ``signed_zero`` legs
        # NEEDLE-MISSED while every ``zero_init`` leg was CAUGHT.
        #
        # THE SUB-STEP. The difference then has to SURVIVE the recurrence, which
        # subtracts the curl from ``fu * kms``; that minuend is a -0.0 only where
        # ``kms`` on the target's own dsig axis is NEGATIVE. Bx's dsig axis is phi,
        # whose kms is 1.0 everywhere on a Dcyl cell, so the B side launders it. Dy's
        # is z, which does go negative under the thin absorber.
        return (m_one and case["value_class"] == "zero_init"
                and case["sub_step"] == "step_D")
    if reach == "inexact_courant":
        return case["courant"] != 0.5 and reaches("live_state", case)
    if reach == "arm_separable":
        return case["value_class"] in ("uniform", "subnormal_band")
    if reach == "prefix_live":
        return bool(case.get("prefix_live", True))
    raise ValueError(f"unknown reach {reach!r}")


# ---------------------------------------------------------------------------
# The host-side overrides a host mutation installs
# ---------------------------------------------------------------------------

def host_overrides(name: Optional[str], sub_step: str, fields, layer, grid,
                   arm) -> Dict[str, Any]:
    """The launcher arguments one host mutation corrupts; ``{}`` when there is none."""
    if name is None:
        return {}
    dtdx = float(grid.dt / grid.dx)
    shape = tuple(int(n) for n in grid.shape)
    if name == "swap_curl_sublattice":
        suffix = "" if family.HALF_INTEGER[sub_step] else "_h"
        return {"tables": {f"{stem}_{axis}":
                           getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
                           for axis in "xyz" for stem in ("kms", "sinv")}}
    if name == "imr_iyee_zero":
        rows = tuple(_imr_row_with_shift(grid, sub_step, target, sign, 0)
                     for _index, target, sign in family.IMR_TERMS[sub_step])
        return {"imr_rows": rows}
    if name == "imr_rows_swapped":
        rows = family.imr_rows_for(sub_step, grid.xp, int(grid.m), dtdx, shape[0],
                                   np.complex64)
        return {"imr_rows": (rows[1], rows[0])}
    if name == "imr_clamp_dropped":
        return {"imr_rows": tuple(
            _imr_row_with_clamp(grid, sub_step, target, sign, None)
            for _index, target, sign in family.IMR_TERMS[sub_step])}
    if name == "imr_clamp_raised":
        return {"imr_rows": tuple(
            _imr_row_with_clamp(grid, sub_step, target, sign, 2.0)
            for _index, target, sign in family.IMR_TERMS[sub_step])}
    if name == "zero_rows_forced_one":
        return {"zero_rows_count": 1}
    if name == "m_class_forced_one":
        return {"m_class_code": family.M_ONE}
    if name == "m_class_forced_zero":
        return {"m_class_code": family.M_ZERO}
    if name == "the_other_expansion_arm":
        return {"__arm__": other_arm(arm)}
    if name == "drop_the_metallic_codes":
        return {"boundary_codes": (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)}
    if name in ("swap_prefix_ir0", "wall_row_not_zero"):
        return {"__prefix__": name}
    raise KeyError(f"unknown host mutation {name!r}")


def _imr_row_with_shift(grid, sub_step, target, sign, iyee_r):
    """The i*m/r row built with a WRONG radial Yee shift."""
    xp = grid.xp
    rows = int(grid.shape[0])
    dtdx = float(grid.dt / grid.dx)
    r_doubled = 2 * xp.arange(rows, dtype=xp.float64) + iyee_r
    divisor = xp.maximum(r_doubled, 1.0)
    row = ((-1j) * (sign * 2.0 * int(grid.m) * dtdx)) / divisor
    return xp.ascontiguousarray(row.astype(xp.complex64))


def _imr_row_with_clamp(grid, sub_step, target, sign, clamp):
    """The i*m/r row with the domain clamp dropped or raised."""
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

    xp = grid.xp
    rows = int(grid.shape[0])
    dtdx = float(grid.dt / grid.dx)
    r_doubled = 2 * xp.arange(rows, dtype=xp.float64) + IYEE_SHIFTS[target][0]
    divisor = r_doubled if clamp is None else xp.maximum(r_doubled, clamp)
    with np.errstate(divide="ignore", invalid="ignore"):
        row = ((-1j) * (sign * 2.0 * int(grid.m) * dtdx)) / divisor
    return xp.ascontiguousarray(row.astype(xp.complex64))


def prefix_for(fields, sub_step: str, mutation: Optional[str]):
    """This launch's prefix, from the SHIPPED helper unless a mutation corrupts it."""
    from meep_gpu.stepping import _face, _span, cylindrical_rderiv_prefix  # noqa: PLC0415

    if mutation is None:
        return cyl_prefix.cylindrical_prefix(fields, sub_step, scratch=None)
    xp = fields.grid.xp
    source = getattr(fields, cyl_prefix.PREFIX_COMPONENT[sub_step])
    if mutation == "swap_prefix_ir0":
        other = "step_D" if sub_step == "step_B" else "step_B"
        ir0 = cyl_prefix.PREFIX_IR0[other]
    else:
        ir0 = cyl_prefix.PREFIX_IR0[sub_step]
    if not cyl_prefix.PREFIX_WALL_ROW[sub_step]:
        return cylindrical_rderiv_prefix(xp, source, ir0)
    rows = source.shape[0]
    extended = xp.empty((rows + 1,) + source.shape[1:], dtype=source.dtype)
    extended[_span(0, 0, rows)] = source
    if mutation == "wall_row_not_zero":
        extended[_face(0, rows)] = source[_face(0, rows - 1)]
    else:
        extended[_face(0, rows)] = 0
    return cylindrical_rderiv_prefix(xp, extended, ir0)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend, spec: Dict[str, Any], sub_step: str, courant: float,
             value_class: str, guard: str, arm: str,
             host_mutation: Optional[str] = None,
             multi_step: bool = True) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML`` objects.
    There is no second transcription on the oracle leg to drift: the kernel is compared
    against the thing it claims to reproduce, byte for byte, from ONE frozen state.
    """
    started = time.time()
    xp = cp if backend.name == "cuda" else _NumpyWearingCupysName()
    # A STABLE per-case seed. ``hash()`` on a tuple of strings is salted by
    # PYTHONHASHSEED, so a gate keyed on it draws a DIFFERENT fixture every process --
    # measured here as a case count that moved between two runs of the same command.
    # A digest of the case's own name is reproducible across processes and hosts.
    rng = np.random.default_rng(SEED + int.from_bytes(hashlib.sha256(
        f"{spec['label']}|{sub_step}|{courant}|{value_class}".encode()).digest()[:4],
        "big"))
    absorber = ABSORBERS[value_class]

    fields, layer, grid = build(xp, spec, courant, absorber)
    seed_state(fields, grid, value_class, rng)
    dtdx = float(grid.dt / grid.dx)

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend.name,
        "arm": arm, "host_mutation": host_mutation, "absorber": absorber,
        "z_kind": spec["z_kind"], "m": int(spec["m"]),
        "zero_rows": family.zero_rows(int(spec["m"]),
                                      bool(spec.get("accurate", False))),
        "dtdx": dtdx, "structure": structure_facts(grid),
    }

    covered, reason = cyl_coverage.covers_pml_cylindrical_complex_curl(
        fields, layer, grid, sub_step,
        license={"arm": arm, "expansion": complex_emitter.EXPANSIONS[arm],
                 "refusals": [], "basis": "measured", "policy_resolved": "keep"},
        subnormal_policy="keep")
    case["predicate_today"] = {"covered": bool(covered), "reason": reason}

    absorption = absorbs(sub_step, layer)
    case["absorbs"] = absorption
    if not absorption["meets_floor"]:
        case["skipped"] = ("an axis's coefficient profile is the identity everywhere; "
                           "this case cannot distinguish a coefficient index error")
        case["seconds"] = time.time() - started
        return case

    frozen = snapshot(fields)

    # Leg 1: the oracle.
    getattr(stepping, sub_step)(fields, layer)
    reference = snapshot(fields)

    moved = oracle_moved(frozen, reference, sub_step)
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    live = axis_row_is_live(frozen, reference, sub_step, grid.shape, int(spec["m"]),
                            bool(spec.get("accurate", False)))
    case["axis_row_is_live"] = live
    case["axis_row_live"] = bool(live["meets_floor"])
    # THE FLOOR IS PER SEEDING, and that split is measured rather than tidy. On the
    # LIVE classes the axis rows must carry moved words or the per-|m| rules cannot be
    # scored at all. On the +-0 classes they need not: ``zero_init`` moves a word only
    # where a negative ``kms`` meets a quiet ``+0.0``, which on the B side is nowhere
    # near the axis row (Bx's dsig axis is phi, whose kms is 1.0) -- so demanding it
    # there would refuse the ONE class the signed-zero mutations live on for a reason
    # that belongs to the fixture. Those classes carry their own floor below, and the
    # ``m1`` reach consults ``axis_row_live`` so an axis-rule mutation is never scored
    # where its row is dead.
    if value_class not in SIGNED_ZERO_CLASSES and not live["meets_floor"]:
        case["skipped"] = ("the rows the per-|m| axis rules touch carry no moved word; "
                           "this case cannot score them")
        case["seconds"] = time.time() - started
        return case

    census = negative_zero_census([reference[n] for n in outputs(sub_step)])
    case["negative_zero_census"] = census
    if value_class in SIGNED_ZERO_CLASSES and census == 0:
        case["skipped"] = (f"the {value_class} seeding produced NO negative-zero word "
                           f"in the reference; the class is VACUOUS on this case and "
                           f"is reported as such rather than as a pass")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, from the SAME frozen state, with the prefix computed from the
    # SAME sources the oracle differenced.
    restore(fields, frozen)
    overrides = host_overrides(host_mutation, sub_step, fields, layer, grid, arm)
    prefix_mutation = overrides.pop("__prefix__", None)
    launch_arm = overrides.pop("__arm__", arm)
    prefix = prefix_for(fields, sub_step, prefix_mutation)

    liveness = prefix_is_live(prefix, sub_step)
    case["prefix_is_live"] = liveness
    case["prefix_live"] = bool(liveness["meets_floor"])
    if value_class not in SIGNED_ZERO_CLASSES and not liveness["meets_floor"]:
        case["skipped"] = ("the prefix is constant down r; every prefix defect is "
                           "invisible on this case")
        case["seconds"] = time.time() - started
        return case

    rows = overrides.get("imr_rows") or family.imr_rows_for(
        sub_step, grid.xp, int(grid.m), dtdx, int(grid.shape[0]), xp.complex64)
    imr_live = imr_is_live(rows)
    case["imr_is_live"] = imr_live
    if not imr_live["meets_floor"] and host_mutation is None and int(spec["m"]) != 0:
        # AT m = 0 THE ROWS ARE ZERO BY CONSTRUCTION AND NEVER READ: the kernel guards
        # every i*m/r site on m_class != 0, so a dead row there is the arm, not a
        # vacuous fixture. The floor stays for every |m| >= 1 case.
        case["skipped"] = ("a bound i*m/r row is constant or zero; a row-index or "
                           "Yee-shift defect is invisible on this case")
        case["seconds"] = time.time() - started
        return case

    backend.launch(sub_step, fields, layer, grid, launch_arm, prefix, **overrides)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(sub_step)}
    case["single_launch"] = combine(parts)
    case["per_array"] = {
        name: {"differing_words": int(parts[name]["differing_floats"]),
               "words": int(parts[name]["total_floats"])}
        for name in outputs(sub_step)}

    # Leg 3: the multi-step. ``fu`` IS STATE, and the prefix is RECOMPUTED every
    # launch on both paths -- which is what a real driver does and what a cached
    # prefix would get wrong.
    if multi_step and host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            getattr(stepping, sub_step)(fields, layer)
            advance_sources(fields, sub_step)
        oracle_multi = snapshot(fields)

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            backend.launch(sub_step, fields, layer, grid, launch_arm,
                           prefix_for(fields, sub_step, None), **overrides)
            advance_sources(fields, sub_step)
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in outputs(sub_step)}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

#: The reduced product's cases -- CHOSEN, not sliced off the front, because both
#: |m| classes, both z terminations and both axis-rule arms must appear or the
#: summary's own "no |m| >= 2 case was scored" refusal fires for a reason about the
#: slice rather than about the kernel.
REDUCED_LABELS: Tuple[str, ...] = ("m-1_r16_z20_metallic", "m-1_r16_z20_periodic",
                                   "m+3_r9_z17_periodic", "m+2_r9_z17_metallic",
                                   "m0_r16_z20_metallic")


def case_product(product: str):
    specs = (CASES if product == "full"
             else tuple(c for c in CASES if c["label"] in REDUCED_LABELS))
    classes = VALUE_CLASSES if product == "full" else ("uniform", "zero_init")
    plan = []
    for spec in specs:
        courants = ((spec["courant"],) if "courant" in spec
                    else (COURANTS if product == "full" else (INEXACT_COURANT,)))
        for sub_step in SUB_STEPS:
            for courant in courants:
                for value_class in classes:
                    plan.append((spec, sub_step, courant, value_class))
    return plan


def run_sweep(results, out_path, backend, product, guard, arm) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product)
    for index, (spec, sub_step, courant, value_class) in enumerate(plan, start=1):
        case = one_case(backend, spec, sub_step, courant, value_class, guard, arm)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
                f"{value_class} SKIPPED: {case['skipped'][:64]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
            f"c={courant} {value_class} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"moved={case['oracle_moved']:.3f} census={case['negative_zero_census']} "
            f"({case['seconds']:.1f} s)")
    return cases


# ---------------------------------------------------------------------------
# The mutation legs
# ---------------------------------------------------------------------------

def scorable_baselines(results) -> Tuple[set, List[Dict[str, Any]]]:
    """The ``(label, sub_step, value_class)`` triples whose UNMUTATED case passed.

    A MUTATION LEG ONLY MEANS SOMETHING WHERE THE BASELINE IS IDENTICAL. On an arm
    that already diverges, every leg -- including one that must be UNCAUGHT -- scores
    CAUGHT for free.
    """
    scorable, excluded = set(), []
    for case in results.get("sweep", {}).get("fmad_false", []):
        key = (case["label"], case["sub_step"], case["value_class"], case["courant"])
        if case.get("skipped"):
            excluded.append({"key": list(key), "why": case["skipped"][:90]})
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical", True)
        if single and multi:
            scorable.add(key)
        else:
            excluded.append({"key": list(key),
                             "why": "the unmutated baseline already diverges here"})
    return scorable, excluded


def mutation_plan(scorable: set, product: str):
    by_label = {spec["label"]: spec for spec in CASES}
    plan = []
    for label, sub_step, value_class, courant in sorted(scorable):
        plan.append((by_label[label], sub_step, courant, value_class))
    return plan


def leg_caught(case: Dict[str, Any]) -> bool:
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def score(legs: List[Dict[str, Any]], must_be_caught, reach: str,
          reading: Optional[str] = None) -> Dict[str, Any]:
    scored = [c for c in legs if not c.get("skipped")]
    live = [c for c in scored if reaches(reach, c)]
    dead = [c for c in scored if not reaches(reach, c)]
    caught_live = sum(1 for c in live if leg_caught(c))
    caught_dead = sum(1 for c in dead if leg_caught(c))
    out = {"ran": len(scored), "live": len(live), "dead": len(dead),
           "caught_live": caught_live, "caught_dead": caught_dead,
           "must_be_caught": must_be_caught, "reach": reach}
    if not scored:
        out["verdict"] = "NO LEGS"
    elif must_be_caught == "measure":
        # A MEASUREMENT, not a bar. Some legs answer "does this choice change a byte
        # HERE" rather than "can the gate see this defect", and forcing them into
        # CAUGHT/NULL is what turns a finding into a red leg or an overclaim.
        out["verdict"] = "MEASURED"
        out["separated_cases"] = caught_live
        if reading is not None:
            out["reading"] = reading.format(caught=caught_live, live=len(live))
            return out
        out["reading"] = (
            f"the two expansion arms differ on {caught_live} of {len(live)} live cases"
            if caught_live else
            "the two expansion arms are DEGENERATE on every case this family reaches: "
            "every complex multiply it makes carries a zero real operand (the "
            "real-coefficient products by construction, the i*m/r row and the |m| = 1 "
            "scalar by measurement), and fma(+-0, x, y) is the same single rounding as "
            "(+-0 * x) + y. The arm binding is therefore NOT evidenced by bytes here")
    elif must_be_caught:
        # THE BAR, AND WHAT IT DELIBERATELY IS NOT. "Every live case must diverge" is
        # the default and is what CAUGHT means. A leg that catches the defect on most
        # of its live cases and misses one is NOT the same failure as a leg that never
        # catches it, and collapsing the two would either refuse a working gate for a
        # fixture reason or hide a needle that never lands. So the second bar is
        # STRUCTURAL: the defect must be caught somewhere in EVERY (sub_step, m class)
        # the reach names, and every case that missed is NAMED in the artifact. A leg
        # that clears the structural bar with misses is "PARTIAL (misses named)" and
        # does not block release; one that leaves a whole sub-step or m class untouched
        # is PARTIAL and does.
        missed = [c for c in live if not leg_caught(c)]
        out["missed_cases"] = [
            {"label": c["label"], "sub_step": c["sub_step"], "m": c["m"],
             "value_class": c["value_class"], "courant": c["courant"],
             "negative_zero_census": c.get("negative_zero_census")}
            for c in missed]
        classes = {(c["sub_step"], family.m_class(c["m"])) for c in live}
        caught_classes = {(c["sub_step"], family.m_class(c["m"]))
                          for c in live if leg_caught(c)}
        class_label = {family.M_ZERO: "m=0", family.M_ONE: "|m|=1",
                       family.M_MANY: "|m|>=2"}
        if not live:
            out["verdict"] = "NO LIVE LEGS"
        elif caught_live == len(live):
            out["verdict"] = "CAUGHT"
        elif not caught_live:
            out["verdict"] = "UNCAUGHT"
        elif caught_classes == classes:
            out["verdict"] = "PARTIAL (misses named)"
        else:
            out["verdict"] = "PARTIAL"
            out["structural_gap"] = sorted(
                f"{sub}/{class_label[cls]}"
                for sub, cls in (classes - caught_classes))
    else:
        out["verdict"] = ("NULL CONFIRMED" if caught_live + caught_dead == 0
                          else "NULL VIOLATED")
    return out


def run_host_mutations(results, out_path, backend, scorable, product,
                       arm) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    plan = mutation_plan(scorable, product)
    for name, spec in HOST_MUTATIONS.items():
        legs: List[Dict[str, Any]] = []
        for case_spec, sub_step, courant, value_class in plan:
            if spec.get("sub_step") and spec["sub_step"] != sub_step:
                continue
            legs.append(one_case(backend, case_spec, sub_step, courant, value_class,
                                 "fmad_false", arm, host_mutation=name,
                                 multi_step=False))
        out[name] = dict(score(legs, spec["must"], spec["reach"],
                               spec.get("reading")), cases=legs)
        log(f"[host-mut] {name}: live {out[name]['caught_live']}/{out[name]['live']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def run_source_mutations(results, out_path, backend, scorable, product,
                         arm) -> Dict[str, Any]:
    """Every device-text defect, applied to the EMITTED source and recompiled."""
    out: Dict[str, Any] = {}
    plan = mutation_plan(scorable, product)
    battery: List[Tuple[str, Callable, Dict[str, Any]]] = []
    for name, spec in OWN_MUTATIONS.items():
        battery.append((name, spec["fn"], spec))
    for name, spec in SHARED_MUTATIONS.items():
        key = spec.get("rename", "shared_" + name)
        battery.append((key, complex_gate.SOURCE_MUTATIONS[name], spec))

    for name, transform, spec in battery:
        sites = backend.set_source_transform(transform)
        if not sites:
            out[name] = {"armed": False, "must_be_caught": spec["must"],
                         "why": "matched 0 sites; the mutation and the kernel have "
                                "drifted apart"}
            log(f"[src-mut] {name}: NOT ARMED")
            backend.set_source_transform(None)
            results["source_mutations"] = out
            save(results, out_path)
            continue
        legs = []
        for case_spec, sub_step, courant, value_class in plan:
            if spec.get("sub_step") and spec["sub_step"] != sub_step:
                continue
            legs.append(one_case(backend, case_spec, sub_step, courant, value_class,
                                 "fmad_false", arm, multi_step=False))
        backend.set_source_transform(None)
        out[name] = dict(score(legs, spec["must"], spec["reach"],
                               spec.get("reading")), armed=True,
                         sites=sites, cases=legs)
        log(f"[src-mut] {name}: live {out[name]['caught_live']}/{out[name]['live']} "
            f"dead-caught {out[name]['caught_dead']}/{out[name]['dead']} "
            f"-> {out[name]['verdict']}")
        results["source_mutations"] = out
        save(results, out_path)
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results) -> Dict[str, Any]:
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]

    per_sub_step: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        entry = per_sub_step.setdefault(case["sub_step"], {
            "cases": 0, "single_identical": 0, "multi_cases": 0, "multi_identical": 0,
            "differing_words": 0, "m_values": set(), "value_classes": set()})
        entry["cases"] += 1
        entry["m_values"].add(case["m"])
        entry["value_classes"].add(case["value_class"])
        if case["single_launch"]["bit_identical"]:
            entry["single_identical"] += 1
        else:
            entry["differing_words"] += sum(
                v["differing_words"] for v in case.get("per_array", {}).values())
        if "multi_step" in case:
            entry["multi_cases"] += 1
            if case["multi_step"]["bit_identical"]:
                entry["multi_identical"] += 1
    for entry in per_sub_step.values():
        entry["m_values"] = sorted(entry["m_values"])
        entry["value_classes"] = sorted(entry["value_classes"])

    reasons: List[str] = []
    for sub_step in SUB_STEPS:
        entry = per_sub_step.get(sub_step)
        if entry is None:
            reasons.append(f"no case was scored for {sub_step}")
            continue
        if entry["single_identical"] != entry["cases"]:
            reasons.append(
                f"{sub_step}: single-launch divergence on "
                f"{entry['cases'] - entry['single_identical']} of {entry['cases']} cases")
        if entry["multi_identical"] != entry["multi_cases"]:
            reasons.append(
                f"{sub_step}: multi-step divergence on "
                f"{entry['multi_cases'] - entry['multi_identical']} of "
                f"{entry['multi_cases']} cases")
        for m_class, label in ((0, "m = 0"), (1, "|m| = 1"), (2, "|m| >= 2")):
            if not any(family.m_class(c["m"]) == m_class for c in scored
                       if c["sub_step"] == sub_step):
                reasons.append(f"{sub_step}: no {label} case was scored")

    axis = results.get("axis_identity") or {}
    if axis and not axis.get("meets_floor"):
        reasons.append(f"the r-axis identity control failed: {axis.get('reading')}")
    stripped = results.get("stripped_control") or {}
    if stripped and not stripped.get("skipped") and not stripped.get("meets_floor"):
        reasons.append(f"the stripped control failed: {stripped.get('reading')}")

    for name, leg in results.get("host_mutations", {}).items():
        if leg.get("verdict") in ("NO LEGS", "NO LIVE LEGS"):
            reasons.append(f"host mutation {name} was never scored on a live case")
        elif leg.get("verdict") not in ("CAUGHT", "NULL CONFIRMED", "MEASURED", "PARTIAL (misses named)"):
            reasons.append(f"host mutation {name} is {leg.get('verdict')}")
    for name, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"source mutation {name} was not armed: {leg.get('why')}")
        elif leg.get("verdict") not in ("CAUGHT", "NULL CONFIRMED", "MEASURED", "PARTIAL (misses named)"):
            reasons.append(f"source mutation {name} is {leg.get('verdict')}")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT]
    guarded = {(c["label"], c["sub_step"], c["courant"], c["value_class"])
               for c in scored if c["single_launch"]["bit_identical"]}
    comparable = [c for c in control
                  if (c["label"], c["sub_step"], c["courant"], c["value_class"])
                  in guarded]
    diverged = [c for c in comparable if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "comparable_where_guarded_leg_was_identical": len(comparable),
        "diverged": len(diverged),
        "reading": ("NOT MEASURED on this run" if not comparable else
                    "the contraction guard is load-bearing on this family"
                    if diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too"),
    }

    certifies = bool(results.get("backend_certifies"))
    return {
        "released": bool(certifies and not reasons),
        "certifies": certifies,
        "reasons": reasons,
        "per_sub_step": per_sub_step,
        "guard_control": guard_control,
        "axis_identity": axis.get("reading"),
        "stripped_control": stripped.get("reading"),
        "scored_cases": len(scored),
        "claim": (
            "on a real Dcyl grid at ANY m (m = 0, |m| = 1 and |m| >= 2) with complex64 "
            "storage and an active split-field PML, cyl_step_B_pml_complex and "
            "cyl_step_D_pml_complex are byte-identical to stepping.step_B / "
            f"stepping.step_D per sub-step, and over {MULTI_STEP_BUDGET} consecutive "
            "launches, on the case matrix in CASES -- including the corpus row's own "
            "(150, 1, 300) m = 0 shape"
            if certifies else
            "NOTHING IS CERTIFIED BY THIS RUN: the host backend compiles the emitted "
            "characters with a host C++ compiler and drives them on NumPy. What it "
            "measures is the TRANSCRIPTION; the shipped bytes are NVRTC's"),
        "does_not_claim": [
            "no throughput claim: the radial prefix stays on the array path, so this "
            "pair cannot go below the prefix's own launches and nothing here was timed",
            "m = 0 with REAL float32 storage is NOT covered and is refused by name: "
            "that is the real-storage pair's kernel (cylindrical_kernels). m = 0 with "
            "complex64 storage IS covered since 2026-09-04, as the third m_class arm",
            "no CONSTITUTIVE claim: this family ships no constitutive predicate and "
            "admits no update_H/update_E slot",
            "nothing dispatches these kernels; fastpath.plan_fast_path is untouched",
            "the expansion ARM is barely evidenced by bytes here and the "
            "the_other_expansion_arm leg carries the number: every complex multiply "
            "this family makes has a zero real operand -- the real-coefficient "
            "products by construction, the i*m/r row and the |m| = 1 scalar by "
            "measurement -- so the two arms coincide almost everywhere. The arm is "
            "still REQUIRED and bound from a policy-matched licence, and a future call "
            "site with a nonzero real coefficient would separate them everywhere",
            "the r near ghost is served as the METALLIC zero; that choice is measured "
            "as a NULL here and cannot be certified by any byte gate from inside the "
            "kernel",
            "a folded, Bloch, BFAST, beta, conductive, nonlinear, dispersive or "
            "real-storage Dcyl run is refused by the predicate, not measured here",
        ],
    }


def save(results, out_path: str) -> None:
    """Atomic rewrite, with the bytes THIS process imported recorded first."""
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cuda", "host"), default="cuda")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--expansion-probe", default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true", help=(
        "import MEEP before installing the subnormal policy. REQUIRED for 'flush': "
        "MEEP's set_zero_subnormals is the only exposure of this process's FTZ/DAZ "
        "bits the package may use, and without it the policy is unattainable on the "
        "host executor."))
    parser.add_argument("--compiler", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    parser.add_argument("--falsify", default=None, help=(
        "install one of OWN_MUTATIONS on the emitted source for the WHOLE run, "
        "sweep included. THE VERDICT ITSELF MUST FLIP: a gate whose release verdict "
        "cannot be made False by a planted defect is not a gate, and no mutation leg "
        "answers that question -- each of those re-runs a case, not the release."))
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_cylindrical_complex",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("do the hand-CUDA complex cylindrical curl kernels reproduce "
                     "stepping.step_B / step_D word for word on a Dcyl grid at any m "
                     "(m = 0 since 2026-09-04, |m| >= 1) with complex64 storage?"),
        "kernel_source_sha256": {
            f"{sub_step}|{arm}": hashlib.sha256(
                family.cylindrical_complex_source(sub_step, arm).encode()).hexdigest()
            for sub_step in SUB_STEPS for arm in ("FMA_V1", "NAIVE")},
        "corpus_digest": family.corpus_digest(),
        "host_compile_options": list(_HOST_OPTIONS),
        "host_optimization_level_is_a_correctness_setting": (
            "-O0: this family carries the only negation of an fma OUTPUT in the "
            "package, and clang rewrites -(fma(a,b,c)) as fma(-a,b,-c) at -O1, which "
            "is wrong on signed zeros. Measured 0/96, 3/96, 0/96 at -O0/-O1/-O2."),
    }

    licence = load_licence(args.expansion_probe, args.subnormal_policy)
    results["expansion_licence"] = licence
    if not licence["usable"]:
        log(f"[fatal] no usable expansion licence: {licence['policy_reasons']}")
        results["status"] = "refused: no usable expansion arm"
        results["summary"] = {"released": False,
                              "reasons": ["no usable expansion licence"]}
        save(results, args.out)
        return 2
    arm = licence["arm"]
    log(f"[arm] {arm} (basis {licence['verdict'].get('basis')!r})")

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        observer = probe.install_nvrtc_binary_observer()
        results["nvrtc_observer"] = observer
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
        backend = KernelBackend()
    else:
        if shutil.which(args.compiler) is None:
            log(f"[fatal] --backend host but {args.compiler!r} is not on PATH")
            results["status"] = f"refused: no host compiler {args.compiler!r}"
            save(results, args.out)
            return 2
        results["environment"] = {"python": sys.version.split()[0],
                                  "numpy_version": np.__version__,
                                  "compiler": args.compiler,
                                  "note": ("host backend: compiles the emitted "
                                           "characters, certifies nothing about NVRTC")}
        backend = HostCompiledBackend(args.compiler)
    results["backend_certifies"] = bool(backend.certifies)
    if args.falsify and not args.skip_mutations:
        # THE MUTATION LEGS CLEAR THE TRANSFORM when they finish, so a falsify run
        # that also ran them would emit the SHIPPED source for most of its cases and
        # report a verdict about nothing. Refused rather than silently half-planted.
        log("[fatal] --falsify requires --skip-mutations: the mutation legs install "
            "and clear their own transform and would wipe the planted defect")
        results["status"] = "refused: --falsify without --skip-mutations"
        save(results, args.out)
        return 2
    if args.falsify:
        if args.falsify not in OWN_MUTATIONS:
            log(f"[fatal] --falsify {args.falsify!r} is not one of {sorted(OWN_MUTATIONS)}")
            results["status"] = "refused: unknown falsify mutation"
            save(results, args.out)
            return 2
        sites = backend.set_source_transform(OWN_MUTATIONS[args.falsify]["fn"])
        results["falsify"] = {"mutation": args.falsify, "sites": sites,
                              "why": ("this run carries a PLANTED DEFECT and its "
                                      "summary.released MUST be False; that is the "
                                      "only leg that measures the verdict itself")}
        log(f"[falsify] {args.falsify} installed on {sites} site(s)")
    save(results, args.out)

    try:
        results["axis_identity"] = axis_identity_leg(np.random.default_rng(SEED + 1))
        log(f"[control] axis_identity: {results['axis_identity']['reading'][:80]}")
        save(results, args.out)

        results["stripped_control"] = stripped_control(
            backend, arm, np.random.default_rng(SEED + 2))
        log(f"[control] stripped: {results['stripped_control'].get('reading', results['stripped_control'].get('skipped'))[:90]}")
        save(results, args.out)

        for guard, options, _primary in GUARD_SETS:
            if args.falsify and guard != "fmad_false":
                continue
            if not backend.certifies and guard != "fmad_false":
                continue
            results.setdefault("guards", {})[guard] = backend.set_guard(options)
            log(f"[guard] {guard} options={options}")
            run_sweep(results, args.out, backend, args.product, guard, arm)
        backend.set_guard(("--fmad=false",))

        if not args.skip_mutations:
            scorable, excluded = scorable_baselines(results)
            results["mutation_scope"] = {
                "scorable": sorted("::".join(str(x) for x in k) for k in scorable),
                "excluded": excluded,
                "why": ("a mutation leg answers 'can this gate see this defect', and "
                        "only an arm whose unmutated baseline is bit-identical can "
                        "answer it")}
            log(f"[mut-scope] {len(scorable)} scorable keys; {len(excluded)} excluded")
            save(results, args.out)
            run_host_mutations(results, args.out, backend, scorable, args.product, arm)
            run_source_mutations(results, args.out, backend, scorable, args.product, arm)
    finally:
        backend.restore()

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
        results["subnormal_policy_stamp_after_run"] = \
            probe.subnormal_policy_stamp(_REPO_API)
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    if args.falsify:
        verdict["falsify_flipped_the_verdict"] = not verdict["released"]
        verdict["falsify_mutation"] = args.falsify
        save(results, args.out)
        log(f"[falsify] released={verdict['released']} -> "
            f"{'FLIPPED as required' if not verdict['released'] else 'DID NOT FLIP'}")
    log(f"[verdict] released={verdict['released']} certifies={verdict['certifies']} "
        f"scored={verdict['scored_cases']}")
    for name, entry in sorted(verdict["per_sub_step"].items()):
        log(f"[verdict]   {name}: single {entry['single_identical']}/{entry['cases']} "
            f"multi {entry['multi_identical']}/{entry['multi_cases']} "
            f"m={entry['m_values']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    if args.falsify:
        # INVERTED on purpose: this leg PASSES when the verdict went False.
        return 0 if not verdict["released"] else 1
    return 0 if not verdict["reasons"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
