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
``stepping.py``). The battery is built around them, because a gate that only re-ran the
certified complex battery would certify a Dcyl kernel against Cartesian defects:

1. the RADIAL DIFFERENCE of MEEP's per-chunk ``f_rderiv_int`` (the column-serial scan,
   ``cylindrical_prefix``; its own battery is ``gate_cuda_cylindrical_scan.py``'s);
2. Bz's and Dz's whole curl REPLACED by ``dtdx`` times that difference, while Dx's Hy
   operands stay RAW;
3. the i*m/r PASS on targets 0 and 2, MEEP's own pass after the curl
   (``stepping._cylindrical_imr_increment``): a quarter turn of the partner by the
   bound realnum ``rinv`` row, masked on the target's unowned cells, folded through the
   additive ladder (``_apply_folded_increment``);
4. the |m| = 1 AXIS-ROW INCREMENTS, which REPLACE curl row 0 after the mask, formed as
   MEEP's double product with Courant (``_realnum_double_product``);
5. the PER-m AXIS RULES on the stored value -- the m = 0 tails (``Bx[0] = 0``; the Dz
   post-add as a double product, then ``Dy[0] = 0``), ``Dz[0] = 0`` at |m| = 1 (field
   only) and the six-volume near-axis zeroing at |m| >= 2.

And the per-chunk PML forms of the certified complex prelude (KERNELS.md section 7.2),
armed through ``gate_cuda_complex``'s B battery and two record host mutations.

=============================================================================
THREE BACKENDS, AND ONLY ONE OF THEM CERTIFIES
=============================================================================

* ``cuda`` -- NVRTC and a GPU. The only backend whose bytes are the shipped bytes and
  the only one that can release.
* ``host`` -- the SAME EMITTED CHARACTERS compiled by the host C++ compiler and driven
  on NumPy. It certifies nothing about NVRTC and says so in the artifact; what it
  measures is the TRANSCRIPTION -- indices, stencils, masks, operand orders,
  coefficient sub-lattices, the radial wiring and the axis rules -- at the merge bar.
  It is the ``gate_cuda_complex`` pattern, borrowed rather than re-invented.
* neither -- ``--backend host`` on a machine with no compiler skips with its reason.

THE HOST LEG COMPILES AT ``-O0``. Under the 0.9.2 text the axis-increment row negated
an fma OUTPUT, and Apple clang 21 at ``-O1`` rewrote ``-(fma(a,b,c))`` as
``fma(-a,b,-c)`` -- exact for every finite value and wrong on signed zeros (measured
0/96, 3/96 and 0/96 host cases diverging at ``-O0``/``-O1``/``-O2``). The part-wise
forms carry no fma, but the row still negates a rounded product and the ladder still
subtracts masked signed zeros, so ``-O0`` stays: it is the setting under which the
host compiler runs the transcription rather than an algebraic rewrite of it. NVRTC is a
different compiler, and only the device leg certifies.

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
* ``radial_is_live`` -- the radial difference must be non-zero and non-constant down r.
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
        --out $OUT/keep/gate.json

Laptop (no CUDA)::

    python -u gate_cuda_cylindrical_complex.py --backend host \\
        --out /tmp/cylcx_host.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten atomically
after every case, so an interrupted run keeps everything up to the failure.
Correctness only -- no throughput claim is made.
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
import gate_chunk_records  # noqa: E402
import gate_provenance  # noqa: E402
import gate_cuda_complex as complex_gate  # noqa: E402
import gate_cuda_cylindrical_scan as scan_gate  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.chunk_map import record_index  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import complex_emitter  # noqa: E402
from meep_gpu.cuda_kernels import coverage as cuda_coverage  # noqa: E402
from meep_gpu.cuda_kernels import cylindrical_complex_kernels as family  # noqa: E402
from meep_gpu.cuda_kernels import cylindrical_coverage as cyl_coverage  # noqa: E402
from meep_gpu.cuda_kernels import cylindrical_prefix as cyl_prefix  # noqa: E402

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
#: on every case, because ``step_B``'s scan reads ``Ey`` and the axis increments read
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
    # A WIDER RADIAL EXTENT: the scan's accumulated sum runs further, which is where
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

#: What the host compiler needs in place of CUDA: ``gate_cuda_complex``'s shim and the
#: one intrinsic this family adds, ``__double2float_rn`` (the axis rules' double
#: product rounded once to float32), which C's double-to-float conversion is under the
#: default rounding mode.
_HOST_SHIM = complex_gate._HOST_SHIM + "#define __double2float_rn(x) ((float)(x))\n"


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

    def launch(self, sub_step, fields, layer, grid, radial, record,
               **overrides) -> None:
        raise NotImplementedError

    def restore(self) -> None:
        pass


def launch_scalars(sub_step: str, grid, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """The launch's grid readings, derived by the functions the shipped launcher calls
    (``step_cylindrical_complex``) unless a host mutation hands one in."""
    m = int(grid.m)
    rows = int(grid.shape[0])
    out = {
        "dtdx": float(grid.dt / grid.dx),
        "m": m,
        "imr_rows": overrides.get("imr_rows"),
        "courant_words": overrides.get("courant_words"),
        "m_class_code": overrides.get("m_class_code"),
        "zero_rows_count": overrides.get("zero_rows_count"),
        "boundary_codes": overrides.get("boundary_codes"),
    }
    if out["imr_rows"] is None:
        out["imr_rows"] = family.imr_rows_for(sub_step, grid, rows)
    if out["courant_words"] is None:
        out["courant_words"] = family.axis_courant_words(sub_step, grid)
    if out["m_class_code"] is None:
        out["m_class_code"] = family.m_class(m)
    if out["zero_rows_count"] is None:
        out["zero_rows_count"] = family.zero_rows(
            m, bool(grid.accurate_fields_near_cylorigin))
    if out["boundary_codes"] is None:
        out["boundary_codes"] = family.cylindrical_complex_boundary_codes(grid)
    return out


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
        the SAME text the shipped path emits, per sub-step, so a leg cannot mutate
        one specialization and score another.
        """
        family._clear_kernel_cache()
        if transform is None:
            family.cylindrical_complex_source = self._original_source
            return 0
        sites = 0
        for sub_step in SUB_STEPS:
            _text, count = transform(self._original_source(sub_step))
            sites = max(sites, count)

        def mutated(sub_step, _original=self._original_source):
            text, _count = transform(_original(sub_step))
            return text

        family.cylindrical_complex_source = mutated
        return sites

    def launch(self, sub_step, fields, layer, grid, radial, record,
               **overrides) -> None:
        """The SHIPPED launcher when nothing is overridden; otherwise the shipped
        argument builder (``curl_arguments``) with the overridden readings, so a host
        mutation corrupts exactly one input and the kernel is the shipped one."""
        cb = cp.ascontiguousarray(cp.asarray(np.ascontiguousarray(record, dtype=np.int32)))
        tables = overrides.pop("tables", None)
        if not overrides and tables is None:
            family.step_cylindrical_complex(sub_step, fields, grid, layer,
                                            chunk_bounds=cb, radial=radial)
        else:
            if tables is None:
                tables = family.cylindrical_complex_curl_tables(
                    layer, family.HALF_INTEGER[sub_step])
            scalars = launch_scalars(sub_step, grid, overrides)
            arguments = family.curl_arguments(
                sub_step, fields, radial, family.device_rows(scalars["imr_rows"], cp),
                scalars["dtdx"], scalars["m"], scalars["courant_words"], tables,
                scalars["boundary_codes"], scalars["m_class_code"],
                scalars["zero_rows_count"], family.chunk_record_argument(cb))
            cells = int(getattr(fields, family.CURL_ARRAYS[sub_step][0][0]).size)
            threads = family._COMPLEX_THREADS
            family._get_kernel(sub_step)(((cells + threads - 1) // threads,),
                                         (threads,), arguments)
        cp.cuda.runtime.deviceSynchronize()

    def restore(self) -> None:
        family.cylindrical_complex_source = self._original_source
        family._COMPILE_OPTIONS = self._original_options
        family._clear_kernel_cache()


def _host_function(compiler: str, options: Sequence[str], workdir: str, text: str,
                   cache: Dict[str, Any], tag: str = ""):
    """Compile one emitted kernel with the HOST compiler behind a serial grid driver;
    returns ``(function, parameters)``, memoized on the text and the options."""
    key = hashlib.sha256((text + str(tuple(options))).encode()).hexdigest()[:24]
    if key in cache:
        return cache[key]
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
    stem = os.path.join(workdir, tag + key)
    with open(stem + ".cpp", "w", encoding="ascii") as handle:
        handle.write(_HOST_SHIM + text + driver)
    subprocess.run([compiler, *options, "-shared", "-fPIC", "-o", stem + ".so",
                    stem + ".cpp"], check=True, capture_output=True)
    function = getattr(ctypes.CDLL(stem + ".so"), f"launch_{name}")
    argtypes: List[Any] = [ctypes.c_int, ctypes.c_int]
    for kind, _ident in parameters:
        if "*" in kind:
            argtypes.append(ctypes.c_void_p)
        elif kind == "float":
            argtypes.append(ctypes.c_float)
        elif kind == "double":
            argtypes.append(ctypes.c_double)
        elif kind == "int":
            argtypes.append(ctypes.c_int)
        else:
            raise ValueError(f"unhandled parameter type {kind!r}")
    function.argtypes = argtypes
    function.restype = None
    cache[key] = (function, parameters)
    return cache[key]


def _host_call(function, parameters, values: Dict[str, Any], cells: int) -> None:
    """Call a host-compiled kernel with its arguments looked up BY PARAMETER NAME, each
    converted to the C type its signature declares -- so a reordered signature is a
    lookup, not a wrong pointer."""
    arguments: List[Any] = []
    keep: List[Any] = []
    for kind, name in parameters:
        value = values[name]
        if "*" in kind:
            array = np.asarray(value)
            if not array.flags.c_contiguous:
                raise ValueError(f"{name}: the host leg needs C-contiguous arrays")
            if array.dtype.kind == "c":
                array = array.view(np.float32)
            keep.append(array)
            arguments.append(array.ctypes.data_as(ctypes.c_void_p))
        elif kind == "float":
            arguments.append(ctypes.c_float(float(value)))
        elif kind == "double":
            arguments.append(ctypes.c_double(float(value)))
        else:
            arguments.append(int(value))
    function((cells + 255) // 256, 256, *arguments)
    del keep


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
            _text, count = transform(family.cylindrical_complex_source(sub_step))
            sites = max(sites, count)
        return sites

    def _entry(self, sub_step: str):
        text = family.cylindrical_complex_source(sub_step)
        if self._transform is not None:
            text, _count = self._transform(text)
        return _host_function(self.compiler, self.options, self._workdir, text,
                              self._cache)

    def launch(self, sub_step, fields, layer, grid, radial, record,
               **overrides) -> None:
        function, parameters = self._entry(sub_step)
        targets, sources = family.CURL_ARRAYS[sub_step]
        shape = tuple(int(n) for n in grid.shape)
        tables = overrides.pop("tables", None)
        if tables is None:
            suffix = "_h" if family.HALF_INTEGER[sub_step] else ""
            tables = {f"{stem}_{axis}": np.ascontiguousarray(
                getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1))
                for axis in "xyz" for stem in ("kms", "sinv")}
        scalars = launch_scalars(sub_step, grid, overrides)
        values: Dict[str, Any] = {name: getattr(fields, name) for name in targets}
        values.update({"fu_" + name: getattr(fields, "fu_" + name) for name in targets})
        values.update({name: getattr(fields, name) for name in sources})
        values["rad"] = np.ascontiguousarray(to_host(radial))
        values["imr0"], values["imr2"] = (np.ascontiguousarray(row, dtype=np.float32)
                                          for row in scalars["imr_rows"])
        values.update(nx=shape[0], ny=shape[1], nz=shape[2], dtdx=scalars["dtdx"],
                      m_word=float(np.float32(scalars["m"])))
        words = tuple(scalars["courant_words"])
        if sub_step == "step_B":
            values["minus_courant"] = words[0]
        else:
            values["plus_courant"], values["four_courant"] = words
        for key, vector in tables.items():
            values[key] = np.ascontiguousarray(to_host(vector), dtype=np.float32)
        codes = scalars["boundary_codes"]
        values.update(bc_x=codes[0], bc_y=codes[1], bc_z=codes[2],
                      m_class=scalars["m_class_code"],
                      zero_rows=scalars["zero_rows_count"],
                      cb=np.ascontiguousarray(record, dtype=np.int32))
        _host_call(function, parameters, values, shape[0] * shape[1] * shape[2])

    def restore(self) -> None:
        self._transform = None
        self._cache.clear()
        shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = tempfile.mkdtemp(prefix="gate_cuda_cylcx_")



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


def radial_is_live(radial, sub_step: str) -> Dict[str, Any]:
    """Is the radial difference non-zero and non-constant down r? A flat one hides every
    radial-substitution defect."""
    host = np.ascontiguousarray(to_host(radial))
    words = _words(host)
    varies = int(np.count_nonzero(host[1:] - host[:-1])) if host.shape[0] > 1 else 0
    return {"rows": int(host.shape[0]),
            "nonzero_words": int(np.count_nonzero(words)),
            "rows_that_differ_from_the_one_below": varies,
            "radial_target": "Bz" if sub_step == "step_B" else "Dz",
            "meets_floor": int(np.count_nonzero(words)) > 0 and varies > 0}


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


def stripped_control(backend, rng) -> Dict[str, Any]:
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
            _launch_certified_cartesian(host, sub_step, fields, layer, grid)
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


def _launch_certified_cartesian(host, sub_step, fields, layer, grid) -> None:
    """Compile and drive ``complex_emitter``'s Cartesian curl on a Dcyl grid, its
    arguments looked up by the certified signature's own parameter names."""
    text = complex_emitter.complex_source(sub_step)
    function, parameters = _host_function(host.compiler, host.options, host._workdir,
                                          text, _CARTESIAN_CACHE, tag="cart_")
    targets, sources = family.CURL_ARRAYS[sub_step]
    shape = tuple(int(n) for n in grid.shape)
    suffix = "_h" if complex_emitter.HALF_INTEGER[sub_step] else ""
    values: Dict[str, Any] = {}
    for index, name in enumerate(targets):
        values[f"f{index}"] = getattr(fields, name)
        values[f"u{index}"] = getattr(fields, "fu_" + name)
    for index, name in enumerate(sources):
        values[f"g{index}"] = getattr(fields, name)
    values.update(nx=shape[0], ny=shape[1], nz=shape[2],
                  dtdx=float(grid.dt / grid.dx))
    for axis in "xyz":
        for stem in ("kms", "sinv"):
            values[f"{stem}_{axis}"] = np.ascontiguousarray(
                getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1), dtype=np.float32)
    # r resolves to 'axis', which the CARTESIAN family has no code for -- so the
    # control passes METALLIC, which is the strictly most favourable choice for it.
    z_metallic = stepping._boundary_kinds(grid, None)[2] == "metallic"
    values.update(bc_x=BC_METALLIC, bc_y=BC_PERIODIC,
                  bc_z=BC_METALLIC if z_metallic else BC_PERIODIC)
    for axis in "xyz":
        values[f"ph_{axis}"] = complex_emitter.CONNECT_NONE
        values[f"p{axis}r"] = 1.0
        values[f"p{axis}i"] = 0.0
    values["cb"] = case_record(grid, layer)
    _host_call(function, parameters, values, shape[0] * shape[1] * shape[2])


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def _sub(pattern: str, replacement: str, text: str) -> Tuple[str, int]:
    out, count = re.subn(pattern, replacement, text)
    return out, count


#: One i*m/r pass block as the kernels spell it: the guard, the increment, its masks,
#: the folded ladder, the closing brace.
_IMR_BLOCK = (r"( *)if \(m_class != 0\) \{\n *cf inc = imr_increment\([^\n]*\n"
              r"(?: *if \([^\n]*inc = cf_zero\(\);\n)* *fold_apply\([^\n]*\n *\}\n")


def m_drop_imr(text):
    """Drop BOTH i*m/r passes -- the whole |m| >= 1 physics."""
    return _sub(_IMR_BLOCK, "", text)


def m_imr_increment_unmasked(text):
    """The pass's increment no longer masked on the target's unowned cells.

    MEEP's pass loops over its owned cells only (LOOP_OVER_VOL_OWNED0) and the array
    path masks the increment to +0 there; unmasked, the z-wall plane of Bz / Dx takes
    the increment. MUST BE CAUGHT where z is metallic.
    """
    return _sub(r" *if \(bc_[xyz] == BC_METALLIC && \w == 0\) inc = cf_zero\(\);\n", "",
                text)


def m_imr_partner_swapped(text):
    """Target 0's pass takes the OTHER centre load -- a real mis-wiring."""
    return _sub(r"imr_increment\(imr0\[i\], f_1\)", "imr_increment(imr0[i], f_2)", text)


def m_imr_row_reversed(text):
    """Index the per-r row from the far wall: the wrong radius on every cell."""
    return _sub(r"imr_increment\(imr(\d)\[i\],", r"imr_increment(imr\1[nx - 1 - i],",
                text)


def m_imr_increment_conjugated(text):
    """The quarter turn's sign on the other part: the coupling rotates the wrong way.

    MEEP's ``(1-2*cmp)`` puts ``-rinv`` on the imaginary part; swapped, the run is
    the conjugate one -- smooth, plausible and wrong.
    """
    return _sub(r"    o\.re = \(rinv \* g\.im\) \* -1\.0f;\n    o\.im = rinv \* g\.re;\n",
                "    o.re = rinv * g.im;\n    o.im = (rinv * g.re) * -1.0f;\n", text)


def m_imr_ladder_skips_sinv(text):
    """The ladder's ``df = sinv * inc`` dropped: the increment enters unscaled where the
    target's dsig layer is not unity (``stepping._apply_folded_increment``)."""
    return _sub(r"    cf df = mul_coefficient_left\(sinv, inc\);\n", "    cf df = inc;\n",
                text)


def m_m0_drop_bx_axis_zero(text):
    """Drop the m = 0 B-side tail: Br keeps the recurrence value on the axis row."""
    return _sub(r" *if \(m_class == 0 && i == 0\) \{\n"
                r" *cf_store\(Bx, idx, cf_zero\(\)\);\n *\}\n", "", text)


_M0_POST_ADD = (r" *cf_store\(Dz, idx, cf_add\(cf_load\(Dz, idx\), "
                r"double_product\(four_courant, cf_load\(Hy, idx\)\)\)\);\n")


def m_m0_drop_dz_axis_post_add(text):
    """Drop the m = 0 on-axis Dz post-add; Dy's zero stays."""
    return _sub(_M0_POST_ADD, "", text)


def m_m0_drop_dy_axis_zero(text):
    """Drop Dp = 0 on the axis at m = 0; the post-add stays."""
    return _sub(r"(if \(m_class == 0 && i == 0\) \{\n[^}]*\)\);\n)"
                r" *cf_store\(Dy, idx, cf_zero\(\)\);\n", r"\1", text)


def m_m0_post_add_reads_the_radial_difference(text):
    """Feed the post-add the radial difference instead of the raw stored Hp."""
    return _sub(r"double_product\(four_courant, cf_load\(Hy, idx\)\)",
                "double_product(four_courant, cf_load(rad, idx))", text)


def m_m0_post_add_in_float32(text):
    """The on-axis product through a float32 factor instead of MEEP's double product.

    ``realnum(double(4*Courant) * double(Hp))`` (step_db.cpp:324) is another word
    wherever ``4*Courant`` is not a float32 word. MUST BE CAUGHT at the inexact Courant.
    """
    return _sub(r"double_product\(four_courant, cf_load\(Hy, idx\)\)",
                "mul_coefficient_left((float)four_courant, cf_load(Hy, idx))", text)


def m_m0_post_add_folded_before_dy_zero_swapped(text):
    """Zero Dp BEFORE the post-add -- the same two stores in the other order.

    They touch different volumes, so this is a PREDICTED NULL: order-independent
    stores are order-independent bytes.
    """
    return _sub(r"( *)(cf_store\(Dz, idx, cf_add\(cf_load\(Dz, idx\), "
                r"double_product\(four_courant, cf_load\(Hy, idx\)\)\)\);\n)"
                r"( *cf_store\(Dy, idx, cf_zero\(\)\);\n)", r"\3\1\2", text)


def m_m0_imr_unguarded(text):
    """Run the i*m/r pass at m = 0 instead of skipping it.

    MEASURED, NOT REQUIRED. The array path runs no pass at m = 0. The m = 0 row is a
    signed zero (``realnum(sign * 0)``), so the pass adds signed-zero increments, which
    move a word only where a ``-0`` meets a ``+0`` in the ladder; the guard is kept
    because the array path SKIPS the pass.
    """
    return _sub(r"if \(m_class != 0\) \{", "if (1) {", text)


_AXIS_REPLACE = r"curl = cf_neg\(double_product\(((?:minus|plus)_courant), ([^\n]*)\)\);"


def m_drop_axis_increment(text):
    return _sub(r" *curl = cf_neg\(double_product\((?:minus|plus)_courant, [^\n]*\n", "",
                text)


def m_axis_increment_accumulates(text):
    """REPLACE vs ACCUMULATE -- byte-visible only on a stored signed zero."""
    return _sub(_AXIS_REPLACE, r"curl = cf_add(curl, cf_neg(double_product(\1, \2)));",
                text)


def m_axis_increment_not_negated(text):
    return _sub(_AXIS_REPLACE, r"curl = double_product(\1, \2);", text)


def m_axis_increment_in_float32(text):
    """The |m| = 1 increment through a float32 Courant factor instead of MEEP's double
    product (step_db.cpp:377). MUST BE CAUGHT at the inexact Courant."""
    return _sub(_AXIS_REPLACE,
                r"curl = cf_neg(mul_coefficient_left((float)\1, \2));", text)


def m_axis_increment_operands_swapped(text):
    """``(A - B)`` is NOT ``-(B - A)`` (grouping choice)."""
    out, count = _sub(r"cf_sub\(cf_sub\(f_2, ss\), turn\)", "cf_sub(cf_sub(ss, f_2), turn)",
                      text)
    if count:
        return out, count
    return _sub(r"cf_sub\(cf_sub\(f_1, sf\), mul_coefficient_left\(2\.0f, f_2\)\)",
                "cf_sub(cf_sub(sf, f_1), mul_coefficient_left(2.0f, f_2))", text)


def m_axis_increment_reads_row0_ez(text):
    """Read the AXIS row of Ez instead of the FIRST OFF-AXIS row."""
    return _sub(r"cf e1 = cf_load\(Ez, idx \+ sx\);", "cf e1 = cf_load(Ez, idx);", text)


def m_axis_increment_before_mask(text):
    """Apply the |m| = 1 replacement BEFORE the mask, so the mask wipes it."""
    return _sub(
        r"((?: *if \(bc_[xyz] == BC_METALLIC && \w == 0\) curl = cf_zero\(\);\n)+)"
        r"((?: *//[^\n]*\n)* *if \(m_class == 1 && i == 0\) \{\n(?:[^}]*)\}\n)",
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


_RADIAL_CURL = r"cf curl = mul_coefficient_left\(dtdx, cf_load\(rad, idx\)\);"


def m_drop_bz_radial_substitution(text):
    """Bz's curl becomes the four-operand Cartesian one again."""
    return _sub(_RADIAL_CURL,
                "cf f_1 = cf_load(Ey, idx);\n"
                "        cf sf = cshift_up(Ey, idx, i, nx, sx, bc_x);\n"
                "        cf f_2 = cf_load(Ex, idx);\n"
                "        cf ss = cshift_up(Ex, idx, j, ny, sy, bc_y);\n"
                "        cf curl = curl_stencil(dtdx, sf, f_1, f_2, ss, pr_x, pr_y);", text)


def m_drop_dz_radial_substitution(text):
    """Dz's curl becomes the four-operand Cartesian one over the raw Hp."""
    return _sub(_RADIAL_CURL,
                "cf f_1 = cf_load(Hy, idx);\n"
                "        cf sf = cshift_dn(Hy, idx, i, nx, sx, bc_x);\n"
                "        cf f_2 = cf_load(Hx, idx);\n"
                "        cf ss = cshift_dn(Hx, idx, j, ny, sy, bc_y);\n"
                "        cf curl = curl_stencil(dtdx, sf, f_1, f_2, ss, pr_x, pr_y);", text)


def n_radial_scale_commuted(text):
    """A NULL: ``rad * dtdx`` per part for ``dtdx * rad``. IEEE multiplication commutes
    bitwise; paired with the substitution legs on the same line."""
    return _sub(_RADIAL_CURL, "cf curl = mul_field_left(cf_load(rad, idx), dtdx);", text)


def m_radial_into_dx(text):
    """Leak the radial difference into Dx, whose Hy operands must stay RAW."""
    return _sub(r"cf f_2 = cf_load\(Hy, idx\);\n"
                r"( *)cf ss = cshift_dn\(Hy, idx, k, nz, sz, bc_z\);",
                r"cf f_2 = cf_load(rad, idx);\n"
                r"\1cf ss = cshift_dn(rad, idx, k, nz, sz, bc_z);", text)


def _r_near_ghost(variable: str, volume: str, sign: str):
    """Serve the array path's CYL_AXIS near ghost instead of the metallic zero.

    ``stepping._shift_down``'s CYL_AXIS branch images stored row 0 with a direction sign
    and the ``(-1)^m`` phase. Both the imaged value and its NEGATION are armed, because
    the claim is that the row is UNOBSERVABLE and neither may move a word. If either is
    ever CAUGHT, the ownership mask has broken, not the ghost.
    """
    call = f"cf {variable} = cshift_dn({volume}, idx, i, nx, sx, bc_x);"
    replacement = (
        f"cf {variable};\n"
        f"        if (i == 0) {{ {variable} = cf_load({volume}, idx); "
        f"{variable}.re = {variable}.re * {sign}f; "
        f"{variable}.im = {variable}.im * {sign}f; }}\n"
        f"        else {{ {variable} = cshift_dn({volume}, idx, i, nx, sx, bc_x); }}")

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


def n_imr_partner_through_a_temporary(text):
    """A NULL: the pass's partner named through a local first. A copy is the identity on
    the bits; a leg that CAUGHT it would mean the harness reports differences that are
    not arithmetic."""
    return _sub(r"cf inc = imr_increment\((imr\d\[i\]), ([^;]*)\);",
                r"cf partner = \2; cf inc = imr_increment(\1, partner);", text)


#: The family's own battery. ``reach`` names the cases the defect can be seen on and
#: is DERIVED, not guessed -- see :func:`reaches`. ``must`` is True for a catch and
#: False for a predicted null.
OWN_MUTATIONS: Dict[str, Dict[str, Any]] = {
    # THE i*m/r BATTERY REACHES |m| >= 1 ONLY: at m = 0 every pass is guarded off by
    # ``m_class != 0`` -- the arm the array path takes -- so a defect in the pass is
    # unreachable there BY DESIGN, and ``m0_imr_unguarded`` measures the guard itself.
    "drop_imr": {"fn": m_drop_imr, "must": True, "reach": "imr_live"},
    "imr_increment_unmasked": {"fn": m_imr_increment_unmasked, "must": True,
                               "reach": "imr_metallic_live"},
    "imr_partner_swapped": {"fn": m_imr_partner_swapped, "must": True, "reach": "imr_live"},
    "imr_row_reversed": {"fn": m_imr_row_reversed, "must": True, "reach": "imr_live"},
    "imr_increment_conjugated": {"fn": m_imr_increment_conjugated, "must": True,
                                 "reach": "imr_live"},
    "imr_ladder_skips_sinv": {"fn": m_imr_ladder_skips_sinv, "must": True,
                              "reach": "imr_live"},
    "drop_axis_increment": {"fn": m_drop_axis_increment, "must": True,
                            "reach": "m1_live"},
    "axis_increment_accumulates": {"fn": m_axis_increment_accumulates, "must": True,
                                   "reach": "m1_zero_init"},
    "axis_increment_not_negated": {"fn": m_axis_increment_not_negated, "must": True,
                                   "reach": "m1_live"},
    "axis_increment_in_float32": {"fn": m_axis_increment_in_float32, "must": True,
                                  "reach": "m1_inexact"},
    "axis_increment_operands_swapped": {"fn": m_axis_increment_operands_swapped,
                                        "must": True, "reach": "m1_live"},
    "axis_increment_before_mask": {"fn": m_axis_increment_before_mask, "must": True,
                                   "reach": "m1"},
    "drop_near_axis_zeroing": {"fn": m_drop_near_axis_zeroing, "must": True,
                               "reach": "mmany_live"},
    "near_axis_leaves_fu": {"fn": m_near_axis_leaves_fu, "must": True, "reach": "mmany_live"},
    "near_axis_row_zero_only": {"fn": m_near_axis_row_zero_only, "must": True,
                                "reach": "mmany_wide_live"},
    "drop_bz_radial_substitution": {"fn": m_drop_bz_radial_substitution, "must": True,
                                    "reach": "live_state", "sub_step": "step_B"},
    "axis_increment_reads_row0_ez": {"fn": m_axis_increment_reads_row0_ez, "must": True,
                                     "reach": "m1_live", "sub_step": "step_B"},
    "drop_dz_radial_substitution": {"fn": m_drop_dz_radial_substitution, "must": True,
                                    "reach": "live_state", "sub_step": "step_D"},
    "radial_into_dx": {"fn": m_radial_into_dx, "must": True, "reach": "live_state",
                       "sub_step": "step_D"},
    "drop_dz_axis_zero": {"fn": m_drop_dz_axis_zero, "must": True, "reach": "m1_live",
                          "sub_step": "step_D"},
    "dz_axis_zero_takes_fu": {"fn": m_dz_axis_zero_takes_fu, "must": True, "reach": "m1_live",
                              "sub_step": "step_D"},
    # THE m = 0 ARM'S OWN BATTERY: each of the array path's three m = 0 rules dropped,
    # the post-add fed the wrong volume or a float32 factor, and the guard on the
    # passes removed. The store-order swap is a predicted null.
    "m0_drop_bx_axis_zero": {"fn": m_m0_drop_bx_axis_zero, "must": True,
                             "reach": "m0_live", "sub_step": "step_B"},
    "m0_drop_dz_axis_post_add": {"fn": m_m0_drop_dz_axis_post_add, "must": True,
                                 "reach": "m0_live", "sub_step": "step_D"},
    "m0_drop_dy_axis_zero": {"fn": m_m0_drop_dy_axis_zero, "must": True,
                             "reach": "m0_live", "sub_step": "step_D"},
    "m0_post_add_reads_the_radial_difference": {
        "fn": m_m0_post_add_reads_the_radial_difference, "must": True,
        "reach": "m0_live", "sub_step": "step_D"},
    "m0_post_add_in_float32": {"fn": m_m0_post_add_in_float32, "must": True,
                               "reach": "m0_inexact", "sub_step": "step_D"},
    "m0_dy_zero_before_dz_post_add": {"fn": m_m0_post_add_folded_before_dy_zero_swapped,
                                      "must": False, "reach": "any",
                                      "sub_step": "step_D"},
    "m0_imr_unguarded": {"fn": m_m0_imr_unguarded, "must": "measure", "reach": "m0",
                         "reading": (
                             "the UNGUARDED form -- the i*m/r pass run at m = 0 with its "
                             "signed-zero row -- moved a word on {caught} of {live} m = 0 "
                             "cases across all four value classes; the guard is kept "
                             "because the array path SKIPS the pass")},
    # THE TWO r-NEAR-GHOST NULLS, on the D side only (Dy's Hz operand). On the B side
    # ``cshift_dn`` is never called, so a mutation to it there is armed in the TEXT and
    # unreachable in fact -- a vacuous leg, not a null. Dz reads no shifted operand: its
    # whole curl is the radial difference.
    "r_near_ghost_dy_image": {"fn": _r_near_ghost("ss", "Hz", "1.0"), "must": False,
                              "reach": "any", "sub_step": "step_D"},
    "r_near_ghost_dy_image_negated": {"fn": _r_near_ghost("ss", "Hz", "-1.0"),
                                      "must": False, "reach": "any",
                                      "sub_step": "step_D"},
    # THE HARNESS NULLS.
    "radial_scale_commuted": {"fn": n_radial_scale_commuted, "must": False,
                              "reach": "any"},
    "reload_the_target_from_memory": {"fn": n_reload_the_target_from_memory,
                                      "must": False, "reach": "any"},
    "imr_partner_through_a_temporary": {"fn": n_imr_partner_through_a_temporary,
                                        "must": False, "reach": "any"},
}

#: The SHARED battery, borrowed from ``gate_cuda_complex`` so this gate cannot own a
#: second spelling of a defect the certified complex record was cut against. Every one
#: of them arms on the IMPORTED ``_HEAD``/``_ARM_SOURCE``/``_TAIL`` strings or on the
#: Cartesian target blocks this pair keeps unchanged -- which is itself the check that
#: "the certified body, character for character" is true rather than claimed. The
#: grouping legs are KERNELS.md section 7.1's inversions (the previous package grouping
#: is the mutant) and the select legs its section 7.2 B battery.
SHARED_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "full_complex_coefficient_products": {"must": True, "reach": "signed_zero"},
    "old_package_curl_grouping": {"must": True, "reach": "live_state"},
    "drop_the_ownership_mask": {"must": True, "reach": "mask_observable"},
    "metallic_ghost_wraps": {"must": True, "reach": "metallic_ghost"},
    "hoist_dtdx_into_the_difference": {"must": True, "reach": "inexact_courant"},
    "word_pair_transposed": {"must": True, "reach": "live_state"},
    "column_major_index": {"must": True, "reach": "live_state"},
    "select_removed": {"must": True, "reach": "live_state"},
    "predicate_on_coefficient_identity": {"must": True, "reach": "live_state"},
    "swap_dsig_and_dsigu": {"must": True, "reach": "live_state"},
    "auxiliary_stored_on_a_plain_cell": {"must": True, "reach": "live_state"},
    "auxiliary_load_unguarded": {"must": False, "reach": "any"},
    "reload_the_target_from_memory": {"must": False, "reach": "any",
                                      "rename": "shared_reload_the_target"},
    "commute_the_plane_sum": {"must": False, "reach": "any"},
}

#: Shared mutations this family EXCLUDES, each with the reason. A mutation that cannot
#: fire and a mutation that is not caught look identical in a verdict, so an
#: unreachable leg is excluded by name rather than scored as a null.
SHARED_EXCLUDED: Dict[str, str] = {
    "old_package_constitutive_grouping":
        "arms on constitutive_apply, which these curl kernels never call",
    "store_fw_before_reading_prev":
        "arms on constitutive_apply, which these curl kernels never call",
    "connection_on_every_plane":
        "arms on the Bloch connection; this family's predicate refuses a phase and its "
        "shift helpers carry none, so the mutated path is unreachable",
    "conjugate_the_rotation":
        "arms on the Bloch connection; see connection_on_every_plane",
}

#: HOST mutations: nothing in the device text changes, the LAUNCHER is handed
#: something wrong. Each is a real defect a call site can carry. The scan's own host
#: defects (the starts line, the weights' ir0, the wall row) are
#: gate_cuda_cylindrical_scan.py's.
HOST_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "swap_curl_sublattice": {"must": True, "reach": "live_state"},
    "imr_iyee_zero": {"must": True, "reach": "imr_live"},
    "imr_rows_swapped": {"must": True, "reach": "imr_live"},
    "imr_clamp_raised": {"must": True, "reach": "m1_live"},
    "zero_rows_forced_one": {"must": True, "reach": "mmany_wide_live"},
    "m_class_forced_one": {"must": True, "reach": "not_m1_live"},
    # THE THIRD ARM FORCED ON THE OTHER TWO: at |m| >= 1 the passes go silent, the
    # increments and the near-axis zeroing never fire and the m = 0 tails do.
    "m_class_forced_zero": {"must": True, "reach": "not_m0_live"},
    "courant_words_in_float32": {"must": True, "reach": "axis_rule_inexact"},
    "record_bound_off_by_one": {"must": True, "reach": "live_state"},
    "record_lattices_swapped": {"must": True, "reach": "live_state"},
    "drop_the_metallic_codes": {"must": True, "reach": "boundary_code_observable"},
    # A PREDICTED NULL: the clamp bites only on row 0 of a shift-0 target, whose
    # increment the mask zeroes -- so dropping it cannot move a compared word (the
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
    """
    if reach == "any":
        return True
    if reach == "live_state":
        # A defect in ORDINARY arithmetic needs ordinary values. Under ``zero_init``
        # every operand is an exact zero and under ``signed_zero`` half of one plane
        # is: the radial difference is flat, the i*m/r products are +-0 and a grouping change
        # moves nothing. Those two classes exist for the +-0 defects and are scored on
        # those; scoring an arithmetic leg there would report UNCAUGHT for a reason
        # that belongs to the fixture.
        return (case["value_class"] not in SIGNED_ZERO_CLASSES
                and case.get("radial_live", True))
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
    if reach == "radial_live":
        return bool(case.get("radial_live", True))
    if reach == "m1_inexact":
        # The double product against a float32 factor: the same word wherever Courant
        # is a float32 word (0.5), so only the inexact Courant can see it.
        return reaches("m1_live", case) and case["courant"] != 0.5
    if reach == "m0_inexact":
        return reaches("m0_live", case) and case["courant"] != 0.5
    if reach == "axis_rule_inexact":
        # Every axis rule that multiplies by Courant: the |m| = 1 increments and the
        # m = 0 on-axis Dz post-add (the B side's m = 0 rule is a zero store).
        return (reaches("m1_inexact", case)
                or (case["sub_step"] == "step_D" and reaches("m0_inexact", case)))
    raise ValueError(f"unknown reach {reach!r}")


# ---------------------------------------------------------------------------
# The host-side overrides a host mutation installs
# ---------------------------------------------------------------------------

def host_overrides(name: Optional[str], sub_step: str, fields, layer, grid) -> Dict[str, Any]:
    """The launcher arguments one host mutation corrupts; ``{}`` when there is none."""
    if name is None:
        return {}
    shape = tuple(int(n) for n in grid.shape)
    if name == "swap_curl_sublattice":
        suffix = "" if family.HALF_INTEGER[sub_step] else "_h"
        return {"tables": {f"{stem}_{axis}":
                           getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
                           for axis in "xyz" for stem in ("kms", "sinv")}}
    if name == "imr_iyee_zero":
        rows = tuple(_imr_row_with_shift(grid, target, sign, 0)
                     for _index, target, sign in family.IMR_TERMS[sub_step])
        return {"imr_rows": rows}
    if name == "imr_rows_swapped":
        rows = family.imr_rows_for(sub_step, grid, shape[0])
        return {"imr_rows": (rows[1], rows[0])}
    if name == "imr_clamp_dropped":
        return {"imr_rows": tuple(
            _imr_row_with_clamp(grid, target, sign, None)
            for _index, target, sign in family.IMR_TERMS[sub_step])}
    if name == "imr_clamp_raised":
        return {"imr_rows": tuple(
            _imr_row_with_clamp(grid, target, sign, 2)
            for _index, target, sign in family.IMR_TERMS[sub_step])}
    if name == "zero_rows_forced_one":
        return {"zero_rows_count": 1}
    if name == "m_class_forced_one":
        return {"m_class_code": family.M_ONE}
    if name == "m_class_forced_zero":
        return {"m_class_code": family.M_ZERO}
    if name == "courant_words_in_float32":
        # The axis rules' double factors rounded through float32 first: another word
        # wherever Courant is not a float32 word (stepping._realnum_double_product).
        return {"courant_words": tuple(np.float64(np.float32(word)) for word in
                                       family.axis_courant_words(sub_step, grid))}
    if name == "drop_the_metallic_codes":
        return {"boundary_codes": (BC_PERIODIC, BC_PERIODIC, BC_PERIODIC)}
    if name in ("record_bound_off_by_one", "record_lattices_swapped"):
        return {"__record__": name}
    raise KeyError(f"unknown host mutation {name!r}")


def _imr_row_with_shift(grid, target, sign, iyee_r):
    """The i*m/r ``rinv`` row built with a WRONG radial Yee shift (the array path's
    realnum quotient otherwise)."""
    rows = int(grid.shape[0])
    the_m = np.float32(sign * 2.0 * int(grid.m) * float(grid.courant))
    here_r = np.maximum(2 * np.arange(rows, dtype=np.int64) + iyee_r, 1).astype(np.float32)
    return np.ascontiguousarray(the_m / here_r, dtype=np.float32)


def _imr_row_with_clamp(grid, target, sign, clamp):
    """The i*m/r ``rinv`` row with the divisor's clamp dropped or raised."""
    from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

    rows = int(grid.shape[0])
    the_m = np.float32(sign * 2.0 * int(grid.m) * float(grid.courant))
    doubled = 2 * np.arange(rows, dtype=np.int64) + IYEE_SHIFTS[target][0]
    here_r = (doubled if clamp is None else np.maximum(doubled, clamp)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.ascontiguousarray(the_m / here_r, dtype=np.float32)


def case_record(grid, layer) -> np.ndarray:
    """The chunk record of the case's own division: the one-rank division of ``grid``
    with ``layer``, the division the oracle steps with (``stepping``'s region plan when
    no driver installed one)."""
    return gate_chunk_records.layer_record(grid, layer)


def host_mutated_record(name: Optional[str], record: np.ndarray, grid) -> np.ndarray:
    """Corrupt the chunk record a LAUNCH BINDS, leaving the oracle's division alone
    (KERNELS.md section 7.2): the low run's end one row up on both lattices, or each
    lattice handed the other's runs."""
    if name is None:
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


def radial_for(backend_name: str, fields, layer, sub_step: str):
    """This launch's radial difference: the SHIPPED scan on the device
    (``cylindrical_prefix.radial_difference``), and the scan gate's transcription of the
    column-serial loop on the host leg (``gate_cuda_cylindrical_scan.scan_numpy``). The
    scan's own defects are that gate's."""
    if backend_name == "cuda":
        return cyl_prefix.radial_difference(fields, layer, sub_step)
    tables = cyl_prefix.radial_scan_tables(fields, layer, sub_step)
    source = getattr(fields, cyl_prefix.PREFIX_COMPONENT[sub_step])
    return scan_gate.scan_numpy(np.ascontiguousarray(to_host(source)), tables)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend, spec: Dict[str, Any], sub_step: str, courant: float,
             value_class: str, guard: str,
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
        "host_mutation": host_mutation, "absorber": absorber,
        "z_kind": spec["z_kind"], "m": int(spec["m"]),
        "zero_rows": family.zero_rows(int(spec["m"]),
                                      bool(spec.get("accurate", False))),
        "dtdx": dtdx, "structure": structure_facts(grid),
    }

    covered, reason = cyl_coverage.covers_pml_cylindrical_complex_curl(
        fields, layer, grid, sub_step)
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

    # Leg 2: the kernel, from the SAME frozen state, with the radial difference scanned
    # from the SAME sources the oracle differenced and the chunk record of the division
    # the oracle stepped with.
    restore(fields, frozen)
    overrides = host_overrides(host_mutation, sub_step, fields, layer, grid)
    record = host_mutated_record(overrides.pop("__record__", None),
                                 case_record(grid, layer), grid)
    case["chunk_record"] = [int(word) for word in record]
    radial = radial_for(backend.name, fields, layer, sub_step)

    liveness = radial_is_live(radial, sub_step)
    case["radial_is_live"] = liveness
    case["radial_live"] = bool(liveness["meets_floor"])
    if value_class not in SIGNED_ZERO_CLASSES and not liveness["meets_floor"]:
        case["skipped"] = ("the radial difference is +0 everywhere or constant down r; "
                           "every radial defect is invisible on this case")
        case["seconds"] = time.time() - started
        return case

    rows = overrides.get("imr_rows")
    if rows is None:
        rows = family.imr_rows_for(sub_step, grid, int(grid.shape[0]))
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

    backend.launch(sub_step, fields, layer, grid, radial, record, **overrides)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(sub_step)}
    case["single_launch"] = combine(parts)
    case["per_array"] = {
        name: {"differing_words": int(parts[name]["differing_floats"]),
               "words": int(parts[name]["total_floats"])}
        for name in outputs(sub_step)}

    # Leg 3: the multi-step. ``fu`` IS STATE, and the radial difference is RESCANNED
    # every launch on both paths -- which is what a real driver does and what a cached
    # one would get wrong.
    if multi_step and host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            getattr(stepping, sub_step)(fields, layer)
            advance_sources(fields, sub_step)
        oracle_multi = snapshot(fields)

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            backend.launch(sub_step, fields, layer, grid,
                           radial_for(backend.name, fields, layer, sub_step), record,
                           **overrides)
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


def run_sweep(results, out_path, backend, product, guard) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product)
    for index, (spec, sub_step, courant, value_class) in enumerate(plan, start=1):
        case = one_case(backend, spec, sub_step, courant, value_class, guard)
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
        out["reading"] = f"{caught_live} of {len(live)} live cases separated"
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


def run_host_mutations(results, out_path, backend, scorable, product) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    plan = mutation_plan(scorable, product)
    for name, spec in HOST_MUTATIONS.items():
        legs: List[Dict[str, Any]] = []
        for case_spec, sub_step, courant, value_class in plan:
            if spec.get("sub_step") and spec["sub_step"] != sub_step:
                continue
            legs.append(one_case(backend, case_spec, sub_step, courant, value_class,
                                 "fmad_false", host_mutation=name,
                                 multi_step=False))
        out[name] = dict(score(legs, spec["must"], spec["reach"],
                               spec.get("reading")), cases=legs)
        log(f"[host-mut] {name}: live {out[name]['caught_live']}/{out[name]['live']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def run_source_mutations(results, out_path, backend, scorable, product) -> Dict[str, Any]:
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
                                 "fmad_false", multi_step=False))
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
            "no throughput claim: nothing here was timed",
            "m = 0 with REAL float32 storage is NOT covered and is refused by name: "
            "that is the real-storage pair's kernel (cylindrical_kernels). m = 0 with "
            "complex64 storage IS covered since 2026-09-04, as the third m_class arm",
            "no CONSTITUTIVE claim: this family ships no constitutive predicate and "
            "admits no update_H/update_E slot",
            "nothing dispatches these kernels; fastpath.plan_fast_path is untouched",
            "the complex products are part-wise: the family emits one body and binds "
            "no expansion arm",
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
            sub_step: hashlib.sha256(
                family.cylindrical_complex_source(sub_step).encode()).hexdigest()
            for sub_step in SUB_STEPS},
        "corpus_digest": family.corpus_digest(),
        "host_compile_options": list(_HOST_OPTIONS),
        "host_optimization_level_is_a_correctness_setting": (
            "-O0: the host compiler must run the transcription, not an algebraic "
            "rewrite of it; under the 0.9.2 text clang rewrote -(fma(a,b,c)) as "
            "fma(-a,b,-c) at -O1 (0/96, 3/96, 0/96 at -O0/-O1/-O2)."),
    }

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
            backend, np.random.default_rng(SEED + 2))
        log(f"[control] stripped: {results['stripped_control'].get('reading', results['stripped_control'].get('skipped'))[:90]}")
        save(results, args.out)

        for guard, options, _primary in GUARD_SETS:
            if args.falsify and guard != "fmad_false":
                continue
            if not backend.certifies and guard != "fmad_false":
                continue
            results.setdefault("guards", {})[guard] = backend.set_guard(options)
            log(f"[guard] {guard} options={options}")
            run_sweep(results, args.out, backend, args.product, guard)
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
            run_host_mutations(results, args.out, backend, scorable, args.product)
            run_source_mutations(results, args.out, backend, scorable, args.product)
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
