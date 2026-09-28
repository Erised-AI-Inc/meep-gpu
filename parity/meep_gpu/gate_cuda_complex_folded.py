"""Sub-step byte-identity gate for the hand-CUDA COMPLEX FOLDED curl pair.

THE QUESTION, in three parts, each measured separately:

1. **The curl.** Is ``cuda_kernels/complex_folded_kernels.py``'s pair byte-identical,
   as uint32 WORDS, to ``stepping.step_B`` / ``step_D`` on a complex64 grid carrying
   a MIRROR FOLD -- at one launch and at 60, at an exactly representable courant and
   at one that is not, on physical-band and subnormal-band operands, at BOTH fold
   terminations, under both float32 subnormal policies?
2. **The constitutive sides.** ``covers_complex_folded_constitutive`` admits the
   CERTIFIED complex constitutive pair on a folded run, on the reading that
   ``stepping.update_H`` / ``update_E`` read no neighbour and no mask. A reading is
   not evidence. The constitutive arm runs the certified kernels against the array
   path on a FOLDED complex grid whose two curls have ALREADY moved the state, per
   side, scored SEPARATELY -- because the Metal track's tranche 6 ended at 757/759
   with the two open slots being exactly one row's constitutive sides while its
   curls were served, so "the curl works therefore the constitutive works" is a
   shape that has already been wrong once.
3. **Is the FOLD doing anything at all?** Every curl case measures the same frozen
   state through the array path twice -- once on the folded grid and once with the
   fold's own mask arms dropped from the kernel -- and refuses itself if the two
   agree. A fold whose mask arms are dead on a fixture measures the CERTIFIED
   complex pair and reports it as a fold verdict.

WHY THIS FAMILY EXISTS, AND WHAT THE CENSUS SAYS IT IS WORTH
------------------------------------------------------------
``coverage.covers_real_pml_complex_curl`` refuses every fold by name, and the
refusal is honest: ``complex_emitter`` carries two boundary codes and no fold branch
at all. On the 2026-08-20 union census that clause is the FIRST refusal on 32
unserved slots over 8 complex-storage rows -- of which three rows also carry
``grid.beta`` (``complex_beta_kernels``') and three carry an off-diagonal epsilon
whose ``update_E`` stays refused, so this family's own share is 17 slots over 5 rows.
The number is recomputed from the census record; it is not this gate's to claim.

WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
----------------------------------------------------------------
* ``oracle_moved`` -- the fraction of output words the array path changed from the
  frozen input. A case that moved nothing is refused, not passed.
* ``absorber_is_not_the_identity`` -- ``max|kms-1|`` over every axis and both Yee
  sub-lattices. An identity profile hides a coefficient-index error behind a table
  of ones.
* ``fold_is_live`` -- running the kernel with the fold's mask arms dropped must
  CHANGE the answer, per case. This is the fold's analogue of the special_kz gate's
  beta-on/beta-off control, and it is measured on the KERNEL rather than argued
  from the fixture's shape.
* ``mirror_ghost_is_nonzero`` -- ``stepping``'s MIRROR ghost is a parity-weighted
  image of a stored interior plane, and the whole fold argument is that it never
  reaches an output word. If the image happens to be all zeros the argument is
  vacuous, so the ghost's own word census is taken and asserted nonzero on a
  folded PERIODIC axis.
* The mutation battery, including legs that MUST BE UNCAUGHT. A battery of only
  must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything.

RUNNING IT
----------
Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy token
because CuPy's disk-cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=$GPU CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_complex_folded.py --backend cupy \\
        --probe parity/meep_gpu/results/expansion_probe_2026-08-17/expansion_probe_keep.json \\
        --subnormal-policy keep --out $OUT/keep/gate.json

Laptop, NO GPU: the HOST backend compiles the SAME emitted characters with the host
C++ compiler through ``gate_cuda_complex``'s shim and drives them on NumPy arrays.
IT CERTIFIES NOTHING ABOUT NVRTC and says so in the artifact -- but it does execute
the actual device text, so the indices, the stencils, the masks and the phase table
are measured at the merge bar rather than hours later::

    python -u gate_cuda_complex_folded.py --backend host --out /tmp/fold.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten atomically
after every case, so an interrupted run keeps everything up to the failure.
Correctness only -- no throughput claim is made or possible.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
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

import gate_provenance  # noqa: E402
import probe_fused_kernel_bit_identity as probe  # noqa: E402

# The certified complex gate is this family's PARENT and its machinery is imported
# rather than copied: the licence reader, the host-compiled backend's shim and
# signature parser, the word census and the word comparator are the same questions
# asked of the same emitter, and two spellings of them is one too many.
import gate_cuda_complex as parent  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels import complex_emitter, coverage  # noqa: E402
from meep_gpu.cuda_kernels import complex_folded_kernels as family  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host

SEED = 20260820
MULTI_STEP_BUDGET = 60
_ADVANCE = np.complex64(complex(0.97, -0.03))

BC_PERIODIC = 0
BC_METALLIC = 1
BC_MIRROR_PERIODIC = 2

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")
SIDES: Tuple[str, ...] = ("H", "E")

#: Every complex volume the four sub-steps read or write.
STATE: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")

CURL_ARRAYS: Dict[str, Dict[str, Any]] = {
    "step_B": {"targets": ("Bx", "By", "Bz"), "aux": ("fu_Bx", "fu_By", "fu_Bz"),
               "sources": ("Ex", "Ey", "Ez"), "half_integer": True,
               "backward": False},
    "step_D": {"targets": ("Dx", "Dy", "Dz"), "aux": ("fu_Dx", "fu_Dy", "fu_Dz"),
               "sources": ("Hx", "Hy", "Hz"), "half_integer": False,
               "backward": True},
}

CONSTITUTIVE_ARRAYS: Dict[str, Dict[str, Any]] = {
    "H": {"targets": ("Hx", "Hy", "Hz"), "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
          "sources": ("Bx", "By", "Bz")},
    "E": {"targets": ("Ex", "Ey", "Ez"), "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
          "sources": ("Dx", "Dy", "Dz")},
}

COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("fmad_false", ("--fmad=false",)),
    ("default_no_options", ()),
)

#: The fold fixtures. Two are the CORPUS's own classes and the rest exist because a
#: family must not release on the half of a split it happens to need: the corpus
#: carries no folded METALLIC complex row and no odd full count, and both are arms
#: of ``stepping._stored_past_owned``'s split.
FOLD_SPECS: Tuple[Dict[str, Any], ...] = (
    # tests_param/TestEigCoeffs.test_binary_grating_special_kz__idx2's class, minus
    # its beta: Mirror(Y) PERIODIC with an in-plane kx on the PML'd X axis.
    {"label": "fold_Y_periodic_kx", "cell": (12.0, 16.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.27, 0.0, 0.0)},
    # tests/TestHoleyWvgBands.test_fields_at_kx's class: the same shape at k = 0.
    {"label": "fold_Y_periodic", "cell": (12.0, 16.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0)},
    # The ODD full count on the folded axis: `_stored_past_owned` holds at both
    # parities and what moves is the IMAGE ROW (n_full 16 reflects stored-2, 15
    # reflects stored-3), so a sweep of the even arm alone would bake `n - 2`.
    {"label": "fold_Y_periodic_odd_count", "cell": (12.0, 15.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0)},
    # THE ODD PLANE PARITY. No corpus row carries one.
    {"label": "fold_Y_periodic_odd_plane", "cell": (12.0, 16.0, 0.0), "axes": "Y",
     "phase": -1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.0, 0.0)},
    # THE OTHER TERMINATION. A folded METALLIC axis is handed BC_METALLIC and must
    # get NO top-plane mask, because MEEP steps that plane.
    {"label": "fold_Y_metallic", "cell": (12.0, 16.0, 0.0), "axes": "Y",
     "phase": 1, "boundaries": ("periodic", "metallic", "periodic"),
     "k": (0.0, 0.0, 0.0)},
    # THE SLOWEST-STRIDE AXIS: an index defect confusing a plane with its
    # neighbour moves with the stride.
    {"label": "fold_X_periodic", "cell": (16.0, 12.0, 0.0), "axes": "X",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.19, 0.0)},
    # examples/solve-cw.py and tests/TestArrayMetadata's class: TWO planes at once,
    # both METALLIC.
    {"label": "fold_XY_metallic", "cell": (16.0, 16.0, 0.0), "axes": "XY",
     "phase": 1, "boundaries": ("metallic", "metallic", "periodic"),
     "k": (0.0, 0.0, 0.0)},
    # tests_param/TestModeDecomposition.test_triangular_lattice_oblique's class: a
    # 3-D fold with phases on the two unfolded axes, so the fold, the wrap rotation
    # and the z stencil are all live at once.
    {"label": "fold_X_periodic_3d", "cell": (12.0, 8.0, 10.0), "axes": "X",
     "phase": 1, "boundaries": ("periodic", "periodic", "periodic"),
     "k": (0.0, 0.19, 0.27)},
)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(xp, spec: Dict[str, Any], courant: float, rng):
    """A frozen ``(fields, layer, grid)`` triple: COMPLEX storage, a live PML, a fold.

    ``force_complex_fields=True`` ALWAYS, even at k = 0: the complex-storage-at-k=0
    class is half of what this family serves (``solve-cw.py``,
    ``TestArrayMetadata``), and forcing it explicitly is what makes those rows real
    rather than an unreachable branch.

    The layer goes on the FAR face only of a folded axis, whose near face is the
    mirror plane -- the same thickness rule every folded fixture in this tree uses.
    THREE DISTINCT INVERSE-EPSILON VOLUMES, never one: binding a single volume for
    all three components is the defect the twelve dead complex kernels carry, and a
    fixture that handed one array to all three could not see it.
    """
    planes = tuple(Mirror(name, spec["phase"]) for name in spec["axes"])
    kwargs = {"dimensions": 2} if float(spec["cell"][2]) == 0.0 else {}
    # ``beta`` RIDES IN THE SPEC and is read here rather than in a second builder:
    # ``gate_cuda_complex_beta`` uses this exact fixture with a nonzero beta, and
    # two fixtures that differ only in one constructor argument is two things to
    # keep in step. A fold spec carries no beta key and gets none.
    if spec.get("beta"):
        kwargs["beta"] = float(spec["beta"])
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), symmetry=planes,
                xp=xp, courant=courant, k_point=tuple(spec["k"]), **kwargs)
    thickness = tuple(
        (0, 0) if grid.shape[axis] < 6
        else (0, 2) if grid.is_mirrored(axis)
        else (2, 2)
        for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    epsilon, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=grid.shape).astype(np.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray((np.float32(1.0) / values).astype(np.float32))
    fields.set_epsilon_volumes(epsilon, inverse)
    return fields, layer, grid


def seed_state(fields, grid, value_class: str, rng) -> Dict[str, np.ndarray]:
    """Seed every complex volume the sub-steps touch; return the host words.

    THE AUXILIARIES START NONZERO in both classes: a zero ``fu``/``f_w`` makes
    ``kms * prev`` exactly zero on the first launch whatever ``kms`` holds, which
    would hide a mis-indexed coefficient until step two.
    """
    xp = grid.xp
    shape = tuple(grid.shape)
    host: Dict[str, np.ndarray] = {}
    if value_class == "uniform":
        for name in STATE:
            host[name] = (rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                          + 1j * rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                          ).astype(np.complex64)
    elif value_class == "subnormal_band":
        planes = probe.subnormal_band_hosts(
            tuple(f"{name}|{part}" for name in STATE for part in ("re", "im")),
            shape, rng)
        for name in STATE:
            host[name] = (planes[f"{name}|re"].astype(np.float32)
                          + 1j * planes[f"{name}|im"].astype(np.float32)
                          ).astype(np.complex64)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def snapshot(fields) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy() for name in STATE}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, names: Sequence[str]) -> None:
    """Move the operands between launches, identically on both paths.

    A COMPLEX factor, not a real one: a real scale leaves the phase alone, and a
    defect in the wrap rotation is a phase defect.
    """
    for name in names:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


def words_differ(left, right) -> int:
    return parent.words_differ(left, right)


def moved_fraction(before, after, names) -> float:
    moved = total = 0
    for name in names:
        a = np.ascontiguousarray(before[name]).view(np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name]).view(np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def structure_facts(grid) -> Dict[str, Any]:
    codes, refusal = family.folded_complex_boundary_codes(grid)
    return {
        "shape": [int(n) for n in grid.shape],
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "stored_cells": [int(grid.stored_cells(a)) for a in range(3)],
        "owned_cells": [int(grid.owned_cells(a)) for a in range(3)],
        "stored_past_owned": [bool(stepping._stored_past_owned(grid, a))
                              for a in range(3)],
        "far_reflect_rows": [None if r is None else int(r)
                             for r in stepping._far_reflect_rows(grid)],
        "boundary_kinds": list(coverage.real_pml_boundary_kinds(grid)),
        "boundary_codes": None if refusal else [int(c) for c in codes],
        "boundary_refusal": refusal,
        "k_point": [float(k) for k in grid.k_point],
        "has_bloch": bool(getattr(grid, "has_bloch", False)),
    }


def absorber_is_not_the_identity(layer) -> Dict[str, Any]:
    worst = 0.0
    for axis in "xyz":
        for stem in ("kms", "sinv", "kps"):
            for suffix in ("", "_h"):
                vector = getattr(layer, f"{stem}_{axis}{suffix}", None)
                if vector is None:
                    continue
                values = to_host(vector).astype(np.float64).ravel()
                worst = max(worst, float(np.max(np.abs(values - 1.0))))
    return {"max_deviation_from_identity": worst, "meets_floor": worst > 0.0}


def mirror_ghost_is_nonzero(fields, grid) -> Dict[str, Any]:
    """The array path's own MIRROR ghost, and whether it carries any nonzero word.

    THE WHOLE FOLD ARGUMENT is that ``stepping``'s parity-weighted ghost never
    reaches an output word because its one consumer plane is masked. If the image
    plane happened to be all zeros the argument would be vacuous and the curl case
    would be passing for the wrong reason. This measures it rather than assuming.

    A folded METALLIC axis writes an EXACT ZERO there by construction
    (stepping.py:1828-1830), so the floor applies to the PERIODIC arm only and says
    which arm it is looking at.
    """
    axes = [a for a in range(3) if grid.is_mirrored(a)]
    rows = stepping._far_reflect_rows(grid)
    parities = stepping._mirror_phases(grid)
    out: Dict[str, Any] = {"per_axis": {}, "meets_floor": True,
                           "arm": "metallic"}
    for axis in axes:
        if rows[axis] is None:
            out["per_axis"][axis] = {"reflect_row": None,
                                     "note": "folded METALLIC: the ghost is an "
                                             "exact zero by construction"}
            continue
        out["arm"] = "periodic"
        shifted = stepping._shift_up(
            grid.xp, fields.Ez, axis, stepping.MIRROR, component="Ez",
            mirror_phase=parities[axis], reflect_row=rows[axis])
        key = [slice(None)] * 3
        key[axis] = -1
        plane = to_host(shifted[tuple(key)])
        nonzero = int(np.count_nonzero(
            np.ascontiguousarray(plane).view(np.float32).ravel().view(np.uint32)))
        out["per_axis"][axis] = {"reflect_row": int(rows[axis]),
                                 "parity": int(parities[axis]),
                                 "nonzero_ghost_words": nonzero}
        if nonzero == 0:
            out["meets_floor"] = False
    return out


# ---------------------------------------------------------------------------
# The backends
# ---------------------------------------------------------------------------

class CupyBackend:
    """NVRTC + the GPU. The only backend that can certify anything."""

    certifies = True
    name = "cupy"

    def __init__(self) -> None:
        from meep_gpu.cuda_kernels import complex_pml_kernels  # noqa: PLC0415

        self.constitutive_module = complex_pml_kernels
        self._original_constitutive_options = complex_pml_kernels._COMPILE_OPTIONS
        self._original_curl_options = family._COMPILE_OPTIONS
        self._transform: Optional[Callable[[str], Tuple[str, int]]] = None

    def set_guard(self, options: Sequence[str]) -> None:
        family._COMPILE_OPTIONS = tuple(options)
        self.constitutive_module._COMPILE_OPTIONS = tuple(options)
        family._clear_kernel_cache()
        self.constitutive_module._clear_kernel_cache()

    def set_source_transform(self, transform, arm) -> int:
        """Install a text mutation on the emitted source; returns the site count.

        Mutating through ``family.set_kernel_source`` is the ONE seam the module
        exposes, and the compile memo keys on the source string, so a rewritten body
        is a miss and reaches NVRTC rather than being served an earlier binary.
        """
        family.reset_kernel_sources()
        family._clear_kernel_cache()
        self._transform = transform
        if transform is None:
            return 0
        sites = 0
        for sub_step in family.FOLDED_KERNELS:
            text, count = transform(family.folded_source(sub_step, arm))
            sites = max(sites, count)
            family.set_kernel_source(sub_step, arm, text)
        return sites

    def launch_curl(self, sub_step, fields, layer, grid, arm, **overrides) -> None:
        # THE LAUNCHER'S EXACTLY-ONE CONTRACT, honoured rather than argued with:
        # ``step_folded_complex`` refuses a grid AND a boundary-code override in
        # one call, and a layer AND a table override likewise -- "passing a layer
        # AND its override is refused" is its own docstring. A control leg that
        # supplies its own codes therefore hands the launcher NO grid, and this
        # backend derives the grid's other three quantities exactly as the
        # launcher would have (same functions, same arguments), so the mutation
        # moves ONLY the quantity it names. Measured 2026-09-01 on the GPU host: the
        # unfixed call died on case 1's fold_is_live leg with the launcher's own
        # ValueError, which is the contract working and this seam being the
        # right place for the translation.
        kwargs = dict(overrides)
        if "boundary_codes" in kwargs:
            from meep_gpu.cuda_kernels.complex_pml_kernels import (  # noqa: PLC0415
                bloch_phase_arguments,
            )

            backward = complex_emitter.KERNELS[sub_step][1]
            flags, values = bloch_phase_arguments(grid, backward)
            kwargs.setdefault("phase_flags", flags)
            kwargs.setdefault("phase_values", values)
            kwargs.setdefault("dtdx", grid.dt / grid.dx)
            grid = None
        if "tables" in kwargs:
            layer = None
        family.step_folded_complex(sub_step, fields, arm, grid=grid, pml=layer,
                                   **kwargs)
        cp.cuda.runtime.deviceSynchronize()

    def launch_constitutive(self, side, fields, layer, grid, arm, *,
                            tables=None) -> None:
        if tables is None:
            self.constitutive_module.update_fused_pml_complex(
                side, fields, arm, pml=layer)
        else:
            self.constitutive_module.update_fused_pml_complex(
                side, fields, arm, tables=tables)
        cp.cuda.runtime.deviceSynchronize()

    def restore(self) -> None:
        family.reset_kernel_sources()
        family._COMPILE_OPTIONS = self._original_curl_options
        self.constitutive_module._COMPILE_OPTIONS = \
            self._original_constitutive_options
        family._clear_kernel_cache()
        self.constitutive_module._clear_kernel_cache()


class HostBackend:
    """The EMITTED SOURCE, compiled by the host C++ compiler, driven on NumPy.

    CERTIFIES NOTHING about NVRTC and says so in the artifact. What it measures is
    the TRANSCRIPTION -- the indices, the stencils, the three mask arms, the ghost
    rules, the coefficient sub-lattices and the phase table -- executed from the
    actual emitted characters at the merge bar rather than hours later on a device.

    The shim, the option set and the signature parser are ``gate_cuda_complex``'s,
    imported: this family emits the same dialect from the same emitter, and a second
    copy of a shim is a second thing to keep in step.
    """

    certifies = False
    name = "host"

    def __init__(self, compiler: str) -> None:
        import tempfile  # noqa: PLC0415

        self.compiler = compiler
        self.options = parent._HOST_OPTIONS
        self._transform: Optional[Callable[[str], Tuple[str, int]]] = None
        self._cache: Dict[str, Any] = {}
        self._workdir = tempfile.mkdtemp(prefix="gate_cuda_complex_folded_")
        self._parent_host = parent.HostCompiledBackend(compiler)

    def set_guard(self, options: Sequence[str]) -> None:
        # The NVRTC guard names have no host equivalent; the host leg always
        # compiles with contraction off, and the guard CONTROL is device-only.
        self._cache.clear()

    def set_source_transform(self, transform, arm) -> int:
        self._transform = transform
        self._cache.clear()
        if transform is None:
            return 0
        sites = 0
        for sub_step in family.FOLDED_KERNELS:
            _, count = transform(family.folded_source(sub_step, arm))
            sites = max(sites, count)
        return sites

    def _entry(self, sub_step: str, arm):
        import ctypes  # noqa: PLC0415
        import subprocess  # noqa: PLC0415

        key = f"{sub_step}|{complex_emitter.normalized_expansion(arm)}"
        if key in self._cache:
            return self._cache[key]
        source = family.folded_source(sub_step, arm)
        if self._transform is not None:
            source, _ = self._transform(source)
        name, parameters = parent.parse_signature(source)
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
        stem = os.path.join(
            self._workdir,
            hashlib.sha256((source + driver).encode()).hexdigest()[:16])
        with open(stem + ".cpp", "w", encoding="ascii") as handle:
            handle.write(parent._HOST_SHIM + source + driver)
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
        import ctypes  # noqa: PLC0415

        return np.ascontiguousarray(array).ctypes.data_as(ctypes.c_void_p)

    @staticmethod
    def _words(array):
        import ctypes  # noqa: PLC0415

        if not array.flags.c_contiguous:
            raise ValueError("the host leg needs C-contiguous complex volumes")
        return array.view(np.float32).ctypes.data_as(ctypes.c_void_p)

    def curl_arguments(self, sub_step, fields, layer, grid, *, tables=None,
                       boundary_codes=None, phase_flags=None, phase_values=None,
                       dtdx=None):
        import ctypes  # noqa: PLC0415

        backward = CURL_ARRAYS[sub_step]["backward"]
        suffix = "_h" if CURL_ARRAYS[sub_step]["half_integer"] else ""
        if tables is None:
            tables = {f"{stem}_{axis}": np.ascontiguousarray(
                getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1))
                for axis in "xyz" for stem in ("kms", "sinv")}
        if boundary_codes is None:
            codes, refusal = family.folded_complex_boundary_codes(grid)
            if refusal is not None:
                raise ValueError(refusal)
            boundary_codes = tuple(int(c) for c in codes)
        if phase_flags is None or phase_values is None:
            flags, values = parent.host_phase_arguments(grid, backward)
            phase_flags = flags if phase_flags is None else phase_flags
            phase_values = values if phase_values is None else phase_values
        if dtdx is None:
            dtdx = grid.dt / grid.dx
        targets = CURL_ARRAYS[sub_step]["targets"]
        sources = CURL_ARRAYS[sub_step]["sources"]
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
        return arguments, int(nx) * int(ny) * int(nz)

    def launch_curl(self, sub_step, fields, layer, grid, arm, **overrides) -> None:
        function, _ = self._entry(sub_step, arm)
        arguments, cells = self.curl_arguments(sub_step, fields, layer, grid,
                                               **overrides)
        function((cells + 255) // 256, 256, *arguments)

    def launch_constitutive(self, side, fields, layer, grid, arm, *,
                            tables=None) -> None:
        self._parent_host.launch_constitutive(side, fields, layer, grid, arm,
                                              tables=tables)

    def restore(self) -> None:
        import shutil  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        self._transform = None
        self._cache.clear()
        self._parent_host.restore()
        shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = tempfile.mkdtemp(prefix="gate_cuda_complex_folded_")


class _GridWithCupysName:
    """The run's grid behind a backend whose ``__name__`` is ``cupy``.

    The ONE clause a host-compiled run cannot satisfy, satisfied at the INPUT
    rather than filtered out of a refusal string -- the technique the coverage
    census uses, and necessary for the same reason: these predicates short-circuit,
    so there is no accumulated refusal list to filter afterwards. On the CuPy
    backend the proxy is the identity (``xp.__name__`` is already ``cupy``) and is
    still applied, so both backends ask the predicate the same question.
    """

    class _Xp:
        __name__ = "cupy"

        def __init__(self, real):
            self._real = real

        def __getattr__(self, item):
            return getattr(self._real, item)

    def __init__(self, grid):
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "xp", self._Xp(grid.xp))

    def has_symmetry(self):
        return object.__getattribute__(self, "_grid").has_symmetry()

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_grid"), item)


def build_backend(name: str):
    if name == "cupy":
        if cp is None:
            raise SystemExit("--backend cupy needs CuPy and a device")
        return CupyBackend()
    if name == "host":
        import shutil  # noqa: PLC0415

        for candidate in ("clang++", "g++", "c++"):
            found = shutil.which(candidate)
            if found:
                return HostBackend(found)
        raise SystemExit("--backend host needs a host C++ compiler on PATH")
    raise SystemExit(f"unknown backend {name!r}")


# ---------------------------------------------------------------------------
# THE FAMILY'S OWN SOURCE MUTATIONS
#
# Each is a text transform on the EMITTED source, returning ``(mutated, sites)``.
# ``sites`` is asserted nonzero and the mutated text asserted different, because a
# mutation that matched nothing and a mutation that cannot bite look identical in a
# verdict.
# ---------------------------------------------------------------------------

def _sub(pattern: str, replacement: str, text: str) -> Tuple[str, int]:
    out, count = re.subn(pattern, replacement, text)
    return out, count


def m_drop_the_top_plane_mask(text: str) -> Tuple[str, int]:
    """The ``_stored_past_owned`` arm deleted -- the fold's own half of the split.

    The plane MEEP does not own is then stepped from a curl assembled out of a
    ghost the cell does not own. Interior-identical, wrong on one plane per
    component: exactly the defect the REAL fold gate measured at 19,649 words, all
    of them on this plane.
    """
    return _sub(r"\n *if \(bc_\w == BC_MIRROR_PERIODIC && \w == n\w - 1\) "
                r"curl = cf_zero\(\);", "", text)


def m_drop_the_fold_cell_zero_mask(text: str) -> Tuple[str, int]:
    """The MIRROR arm of the cell-0 mask deleted, leaving only METALLIC's.

    A folded PERIODIC axis is not metallic, so its mirror plane stops being masked
    and the run steps a cell ``_mask_non_owned_cells`` drops. Silent on a folded
    METALLIC grid, which is why the leg is scored only where a fold code is live.
    """
    return _sub(r"\n *if \(bc_\w == BC_MIRROR_PERIODIC && \w == 0\) "
                r"curl = cf_zero\(\);", "", text)


def m_fold_ghost_wraps(text: str) -> Tuple[str, int]:
    """``BC_MIRROR_PERIODIC`` falls through to the PERIODIC wrap instead of zero.

    The ghost then carries the opposite face's value instead of the zero the mask
    makes irrelevant. THE INTERESTING PART is that this SHOULD be undetectable if
    the "ghost is dead" argument holds -- the ghost's one consumer plane is masked
    -- so this leg is a NULL and its verdict is the argument, measured.
    """
    return _sub(r"if \(bc == BC_METALLIC \|\| bc == BC_MIRROR_PERIODIC\) "
                r"return cf_zero\(\);",
                "if (bc == BC_METALLIC) return cf_zero();", text)


def m_top_plane_masks_cell_zero(text: str) -> Tuple[str, int]:
    """The top-plane mask aimed at cell 0 -- an off-by-the-whole-extent index.

    Two planes go wrong at once: the owned top plane is stepped and the shift-1
    cell 0, which MEEP DOES own, is zeroed.
    """
    return _sub(r"(if \(bc_(\w) == BC_MIRROR_PERIODIC && )(\w) == n\2 - 1\)",
                r"\1\3 == 0)", text)


def m_mask_the_metallic_top_plane(text: str) -> Tuple[str, int]:
    """The top-plane mask widened to BC_METALLIC -- the half that must NOT fire.

    On a folded METALLIC axis MEEP steps the top plane. A family that masked it too
    would still reproduce every folded PERIODIC fixture, which is why the
    metallic arm is in the sweep at all.
    """
    return _sub(r"if \(bc_(\w) == BC_MIRROR_PERIODIC && (\w) == n\1 - 1\)",
                r"if (bc_\1 != BC_PERIODIC && \2 == n\1 - 1)", text)


#: THE NULLS. A battery of only must-be-caught legs scores identically whether the
#: comparator works or has degenerated into failing everything.
NULL_MUTATIONS: Tuple[str, ...] = ("fold_ghost_wraps", "reload_the_target",)


def n_reload_the_target(text: str) -> Tuple[str, int]:
    """Store and reload the curl before the recurrence -- a value-preserving no-op.

    IEEE round-trip through memory changes nothing, so this MUST be uncaught. If it
    is caught, the comparator is reacting to something other than the arithmetic.
    """
    return _sub(r"( *)(pml_apply\(f\d, u\d, idx, curl,)",
                r"\1cf curl_rt = curl; curl = curl_rt;\n\1\2", text)


SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "drop_the_top_plane_mask": m_drop_the_top_plane_mask,
    "drop_the_fold_cell_zero_mask": m_drop_the_fold_cell_zero_mask,
    "top_plane_masks_cell_zero": m_top_plane_masks_cell_zero,
    "mask_the_metallic_top_plane": m_mask_the_metallic_top_plane,
    "fold_ghost_wraps": m_fold_ghost_wraps,
    "reload_the_target": n_reload_the_target,
    # THE CERTIFIED FAMILY'S OWN DEFECT CLASSES, re-armed here because this family
    # emits its bytes: a fold delta that broke one of them would otherwise be
    # invisible to a battery that only tested the fold.
    "fold_the_zero_cross_terms": parent.m_fold_the_zero_cross_terms,
    "plane_wise_scaling": parent.m_plane_wise_scaling,
    "left_to_right_curl": parent.m_left_to_right_curl,
    "phase_on_every_lane": parent.m_phase_on_every_lane,
    "conjugate_the_rotation": parent.m_conjugate_the_rotation,
    "word_pair_transposed": parent.m_word_pair_transposed,
    "column_major_index": parent.m_column_major_index,
}

#: WHICH FOLD CODE A LEG NEEDS TO BE ABLE TO BITE. A leg run only on grids that
#: cannot express its defect scores UNCAUGHT for a reason that has nothing to do
#: with the kernel.
#: Legs whose defect needs a PHASED axis to be able to bite. A rotation mutation on
#: an unphased grid is a no-op: the kernel emits no multiply at all when every phase
#: flag is 0, which is the bit-identity of k = 0. Measured 2026-08-20 on the host
#: leg: both legs caught 6/6 on the three phased fixtures and 0/10 on the unphased
#: ones, exactly. Scoring them over the unphased fixtures reported PARTIAL for a
#: reason about the FIXTURE rather than about the kernel.
MUTATION_REQUIRES_A_PHASE: Tuple[str, ...] = ("phase_on_every_lane",
                                              "conjugate_the_rotation")

#: THE CERTIFIED COMPLEX FAMILY'S OWN SIGNED-ZERO DEFECT CLASSES, re-armed here as a
#: REGRESSION CHECK rather than as this family's verdict. Their full scoring belongs
#: to ``gate_cuda_complex``, which released on them under its own case product; what
#: this gate asks is narrower and is stated as such: did the fold delta DISABLE one?
#: So the bar is "caught on at least one fixture", and the per-fixture table stays in
#: the artifact rather than being collapsed into a fraction.
#:
#: WHY THEY DO NOT FIRE EVERYWHERE, measured rather than guessed: both need a word
#: that is EXACTLY zero for the cross term's sign to reach an output, and under
#: uniform random operands the only exact zeros in the step are the ones the kernel
#: itself writes -- the zero ghost and the masked planes. On the 2026-08-20 host run
#: they were caught on every ``step_B`` case with a wrapping axis and on no
#: ``step_D`` case at all, which is where those zeros land.
INHERITED_MUTATIONS: Tuple[str, ...] = ("fold_the_zero_cross_terms",
                                        "plane_wise_scaling")

MUTATION_REQUIRES_CODE: Dict[str, int] = {
    "drop_the_top_plane_mask": BC_MIRROR_PERIODIC,
    "drop_the_fold_cell_zero_mask": BC_MIRROR_PERIODIC,
    "top_plane_masks_cell_zero": BC_MIRROR_PERIODIC,
    "mask_the_metallic_top_plane": BC_METALLIC,
    "fold_ghost_wraps": BC_MIRROR_PERIODIC,
}

#: HOST-side mutations: the inputs corrupted rather than the device text.
HOST_MUTATIONS: Tuple[str, ...] = (
    "swap_the_yee_sub_lattice",
    "fold_code_read_as_periodic",
    "fold_code_read_as_metallic",
    "unconjugated_backward_phase",
)
HOST_NULL_MUTATIONS: Tuple[str, ...] = ()


def host_mutated_curl_inputs(name: Optional[str], sub_step: str, layer, grid):
    """The keyword overrides one HOST mutation installs; ``{}`` when there is none."""
    if name is None:
        return {}
    codes, refusal = family.folded_complex_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(refusal)
    codes = [int(c) for c in codes]
    if name == "swap_the_yee_sub_lattice":
        # HALF-INTEGER for step_B and INTEGER for step_D; swapped, it is a
        # half-cell error in the absorber profile -- converged, smooth and wrong.
        suffix = "" if CURL_ARRAYS[sub_step]["half_integer"] else "_h"
        return {"tables": {
            f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}}
    if name == "fold_code_read_as_periodic":
        return {"boundary_codes": tuple(
            BC_PERIODIC if c == BC_MIRROR_PERIODIC else c for c in codes)}
    if name == "fold_code_read_as_metallic":
        # The 2026-08-19 measurement on the REAL pair, re-armed: handing a folded
        # PERIODIC axis BC_METALLIC is exactly the configuration that diverged 0/64
        # there, and it must diverge here too or the top-plane mask is not reached.
        return {"boundary_codes": tuple(
            BC_METALLIC if c == BC_MIRROR_PERIODIC else c for c in codes)}
    if name == "unconjugated_backward_phase":
        flags, values = parent.host_phase_arguments(
            grid, not CURL_ARRAYS[sub_step]["backward"])
        return {"phase_flags": flags, "phase_values": values}
    raise ValueError(f"unknown host mutation {name!r}")


def spec_codes(spec: Dict[str, Any]) -> Tuple[int, ...]:
    rng = np.random.default_rng(1)
    _f, _l, grid = build(np, spec, COURANTS[0], rng)
    codes, refusal = family.folded_complex_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(f"the shipped resolver refuses fixture {spec['label']!r}: "
                         f"{refusal}")
    return tuple(int(c) for c in codes)


# ---------------------------------------------------------------------------
# One curl case
# ---------------------------------------------------------------------------

def curl_case(backend, xp, spec, sub_step, courant, value_class, guard, arm,
              host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE SEED COMES FROM A DIGEST, never from ``hash()``: Python salts ``hash()`` of a
    tuple containing strings with ``PYTHONHASHSEED``, so such a gate draws a
    different fixture every process and a failing case cannot be replayed.
    """
    started = time.time()
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{sub_step}|{courant}|{value_class}".encode()
        ).digest()[:4], "big"))
    fields, layer, grid = build(xp, spec, courant, rng)
    host = seed_state(fields, grid, value_class, rng)

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend.name,
        "arm": complex_emitter.EXPANSION_NAMES[
            complex_emitter.normalized_expansion(arm)],
        "host_mutation": host_mutation, "fold_axes": spec["axes"],
        "structure": structure_facts(grid),
        "word_census": parent.word_census(host),
    }
    asked = _GridWithCupysName(grid)
    covered, reason = family.covers_complex_folded_curl(
        fields, layer, asked, sub_step, license=LICENCE_SHAPE,
        subnormal_policy=POLICY_SHAPE)
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    if not covered:
        case["skipped"] = f"the predicate refuses this configuration: {reason}"
        case["seconds"] = time.time() - started
        return case
    certified, certified_reason = coverage.covers_real_pml_complex_curl(
        fields, layer, asked, sub_step, license=LICENCE_SHAPE,
        subnormal_policy=POLICY_SHAPE)
    case["certified_predicate"] = {"covered": bool(certified),
                                   "reason": certified_reason}
    if certified:
        case["skipped"] = ("the CERTIFIED complex predicate also admits this "
                           "configuration; two families on one slot is a widening")
        case["seconds"] = time.time() - started
        return case

    absorbs = absorber_is_not_the_identity(layer)
    case["absorber_is_not_the_identity"] = absorbs
    if not absorbs["meets_floor"]:
        case["skipped"] = ("every absorber coefficient is the identity; a "
                           "coefficient-index error would be invisible here")
        case["seconds"] = time.time() - started
        return case

    ghost = mirror_ghost_is_nonzero(fields, grid)
    case["mirror_ghost_is_nonzero"] = ghost
    if not ghost["meets_floor"]:
        case["skipped"] = ("the array path's mirror ghost is all-zero words, so "
                           "'the kernel writes a zero instead' is not a claim this "
                           "fixture tests")
        case["seconds"] = time.time() - started
        return case

    frozen = snapshot(fields)
    outputs = (tuple(CURL_ARRAYS[sub_step]["targets"])
               + tuple(CURL_ARRAYS[sub_step]["aux"]))

    getattr(stepping, sub_step)(fields, layer)
    reference = snapshot(fields)
    case["oracle_moved"] = moved_fraction(frozen, reference, outputs)
    if case["oracle_moved"] == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    overrides = host_mutated_curl_inputs(host_mutation, sub_step, layer, grid)
    case["host_overrides"] = sorted(overrides)

    restore(fields, frozen)
    backend.launch_curl(sub_step, fields, layer, grid, arm, **overrides)
    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs}
    case["single_launch"] = combine(parts)
    case["single_launch_per_array"] = {
        name: parts[name]["differing_floats"] for name in outputs}

    # THE FOLD'S OWN LIVENESS FLOOR, measured on the KERNEL: hand the folded axis
    # the code the CERTIFIED family would have to use (PERIODIC on a folded
    # periodic axis, which is what "no fold branch" means) and require the answer
    # to change. A fixture where it does not measures the certified pair.
    if host_mutation is None:
        codes, _ = family.folded_complex_boundary_codes(grid)
        blind = tuple(BC_PERIODIC if int(c) == BC_MIRROR_PERIODIC
                      else (BC_PERIODIC if int(c) == BC_METALLIC else int(c))
                      for c in codes)
        restore(fields, frozen)
        backend.launch_curl(sub_step, fields, layer, grid, arm,
                            boundary_codes=blind)
        # ``to_host`` FIRST: ``reference`` is a host snapshot and the fields are
        # device arrays on this backend, and ``parent.words_differ`` views NumPy
        # words -- CuPy refuses the implicit conversion (measured 2026-09-01 on
        # the GPU host: the first device run of this gate died here). The host
        # backend passes NumPy straight through ``to_host`` unchanged.
        blind_words = sum(words_differ(reference[name],
                                       to_host(getattr(fields, name)))
                          for name in outputs)
        case["fold_is_live"] = {"blind_codes": list(blind),
                                "differing_words": blind_words,
                                "meets_floor": blind_words > 0}
        if not case["fold_is_live"]["meets_floor"]:
            case["skipped"] = ("running the kernel fold-blind changed nothing: this "
                               "case measures the certified complex pair, not this "
                               "family")
            case["seconds"] = time.time() - started
            return case

        # Leg 3: the multi-step. ``fu`` IS STATE.
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            getattr(stepping, sub_step)(fields, layer)
            advance_sources(fields, CURL_ARRAYS[sub_step]["sources"])
        oracle_multi = snapshot(fields)
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            backend.launch_curl(sub_step, fields, layer, grid, arm)
            advance_sources(fields, CURL_ARRAYS[sub_step]["sources"])
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in outputs}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


def constitutive_case(backend, xp, spec, side, courant, value_class, guard,
                      arm) -> Dict[str, Any]:
    """The CERTIFIED complex constitutive kernel against the array path, on a FOLD.

    THE CURLS RUN FIRST, both of them, so the state this sub-step reads is a state
    the fold produced. A constitutive arm seeded with fresh random numbers would
    measure the certified kernel on a grid that merely HAS a mirror attribute.
    """
    started = time.time()
    rng = np.random.default_rng(
        SEED + 977 + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{side}|{courant}|{value_class}".encode()
        ).digest()[:4], "big"))
    fields, layer, grid = build(xp, spec, courant, rng)
    seed_state(fields, grid, value_class, rng)

    case: Dict[str, Any] = {
        "label": spec["label"], "side": side, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend.name,
        "arm": complex_emitter.EXPANSION_NAMES[
            complex_emitter.normalized_expansion(arm)],
        "structure": structure_facts(grid),
    }
    covered, reason = family.covers_complex_folded_constitutive(
        fields, layer, _GridWithCupysName(grid), side, license=LICENCE_SHAPE,
        subnormal_policy=POLICY_SHAPE)
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    if not covered:
        case["skipped"] = f"the predicate refuses this configuration: {reason}"
        case["seconds"] = time.time() - started
        return case

    stepping.step_B(fields, layer)
    stepping.step_D(fields, layer)

    frozen = snapshot(fields)
    outputs = (tuple(CONSTITUTIVE_ARRAYS[side]["targets"])
               + tuple(CONSTITUTIVE_ARRAYS[side]["aux"]))
    if side == "H":
        stepping.update_H(fields, layer)
    else:
        stepping.update_E(fields, layer)
    reference = snapshot(fields)
    case["oracle_moved"] = moved_fraction(frozen, reference, outputs)
    if case["oracle_moved"] == 0.0:
        case["skipped"] = ("the array path changed no output word; a constitutive "
                           "case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    restore(fields, frozen)
    backend.launch_constitutive(side, fields, layer, grid, arm)
    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs}
    case["single_launch"] = combine(parts)

    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        if side == "H":
            stepping.update_H(fields, layer)
        else:
            stepping.update_E(fields, layer)
        advance_sources(fields, CONSTITUTIVE_ARRAYS[side]["sources"])
    oracle_multi = snapshot(fields)
    restore(fields, frozen)
    for _ in range(MULTI_STEP_BUDGET):
        backend.launch_constitutive(side, fields, layer, grid, arm)
        advance_sources(fields, CONSTITUTIVE_ARRAYS[side]["sources"])
    multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
             for name in outputs}
    case["multi_step"] = combine(multi)
    case["multi_step"]["launches"] = MULTI_STEP_BUDGET
    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweeps
# ---------------------------------------------------------------------------

#: The shape the predicate's licence clause checks. On a device run this is
#: REPLACED by the verdict the probe artifact licenses; the placeholder exists so a
#: host run (which compiles no NVRTC and binds no arm from an artifact) can still
#: ask the predicate its other clauses. ``main`` overwrites both.
LICENCE_SHAPE: Dict[str, Any] = {"arm": "FMA_V1", "expansion": 1,
                                 "basis": "measured", "refusals": [],
                                 "policy_resolved": "keep"}
POLICY_SHAPE: str = "keep"


def case_product(product: str):
    specs = FOLD_SPECS if product == "full" else FOLD_SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    return [(spec, sub_step, courant, value_class)
            for spec in specs for sub_step in SUB_STEPS
            for courant in courants for value_class in classes]


def run_sweep(results, out_path, backend, xp, product, guard, arm):
    cases = []
    plan = case_product(product)
    for index, (spec, sub_step, courant, value_class) in enumerate(plan, start=1):
        case = curl_case(backend, xp, spec, sub_step, courant, value_class, guard, arm)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
                f"SKIPPED: {case['skipped'][:70]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
            f"c={courant} {value_class} shape={case['structure']['shape']} "
            f"codes={case['structure']['boundary_codes']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"diff={case['single_launch']['differing_floats']} "
            f"fold_live={case.get('fold_is_live', {}).get('differing_words')} "
            f"({case['seconds']:.1f} s)")
    return cases


def run_constitutive(results, out_path, backend, xp, product, guard, arm):
    cases = []
    specs = FOLD_SPECS if product == "full" else FOLD_SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    plan = [(spec, side, courant, value_class)
            for spec in specs for side in SIDES
            for courant in courants for value_class in classes]
    for index, (spec, side, courant, value_class) in enumerate(plan, start=1):
        case = constitutive_case(backend, xp, spec, side, courant, value_class,
                                 guard, arm)
        cases.append(case)
        results.setdefault("constitutive", {})[guard] = cases
        save(results, out_path)
        if case.get("skipped"):
            log(f"[const {guard}] {index}/{len(plan)} {spec['label']} update_{side} "
                f"SKIPPED: {case['skipped'][:70]}")
            continue
        log(f"[const {guard}] {index}/{len(plan)} {spec['label']} update_{side} "
            f"c={courant} {value_class} "
            f"single={'IDENTICAL' if case['single_launch']['bit_identical'] else 'DIVERGED'} "
            f"multi={'IDENTICAL' if case['multi_step']['bit_identical'] else 'DIVERGED'} "
            f"diff={case['single_launch']['differing_floats']} "
            f"({case['seconds']:.1f} s)")
    return cases


def scorable_baselines(results) -> Tuple[set, List[Dict[str, Any]]]:
    """``(label, sub_step)`` pairs whose UNMUTATED case was bit-identical.

    A mutation leg only means something where the baseline is identical: on an arm
    that already diverges, every leg -- including one that must be UNCAUGHT --
    scores CAUGHT for free.
    """
    baseline = {}
    for case in results.get("sweep", {}).get("fmad_false", []):
        if (case.get("courant") != INEXACT_COURANT
                or case.get("value_class") != "uniform"):
            continue
        baseline[(case["label"], case["sub_step"])] = case
    scorable, excluded = set(), []
    for key, case in baseline.items():
        if case.get("skipped"):
            excluded.append({"label": key[0], "sub_step": key[1],
                             "why": f"baseline skipped: {case['skipped'][:80]}"})
            continue
        if (case["single_launch"]["bit_identical"]
                and case.get("multi_step", {}).get("bit_identical", True)):
            scorable.add(key)
        else:
            excluded.append({"label": key[0], "sub_step": key[1],
                             "why": "the unmutated baseline already diverges here",
                             "differing_words":
                                 case["single_launch"]["differing_floats"]})
    return scorable, excluded


def mutation_plan(product: str, scorable: Optional[set] = None):
    specs = FOLD_SPECS if product == "full" else FOLD_SPECS[:3]
    plan = [(spec, sub_step, INEXACT_COURANT, "uniform")
            for spec in specs for sub_step in SUB_STEPS]
    if scorable is None:
        return plan
    return [e for e in plan if (e[0]["label"], e[1]) in scorable]


def leg_caught(case: Dict[str, Any]) -> bool:
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def _score(legs, must_be_caught: bool) -> Dict[str, Any]:
    scored = [c for c in legs if not c.get("skipped")]
    caught = sum(1 for c in scored if leg_caught(c))
    if not scored:
        verdict = "NO LEGS"
    elif must_be_caught:
        verdict = ("CAUGHT" if caught == len(scored)
                   else "PARTIAL" if caught else "UNCAUGHT")
    else:
        verdict = "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"
    return {"ran": len(scored), "caught": caught, "uncaught": len(scored) - caught,
            "must_be_caught": must_be_caught, "verdict": verdict}


def run_host_mutations(results, out_path, backend, xp, product, scorable, arm):
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable)
    for name in HOST_MUTATIONS:
        legs = []
        for spec, sub_step, courant, value_class in plan:
            if name.startswith("fold_code_read_as") and \
                    BC_MIRROR_PERIODIC not in spec_codes(spec):
                continue
            if name == "unconjugated_backward_phase" and not any(spec["k"]):
                continue
            legs.append(curl_case(backend, xp, spec, sub_step, courant,
                                  value_class, "fmad_false", arm,
                                  host_mutation=name))
        out[name] = dict(_score(legs, name not in HOST_NULL_MUTATIONS), cases=legs)
        log(f"[host-mut] {name}: caught {out[name]['caught']}/{out[name]['ran']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def run_source_mutations(results, out_path, backend, xp, product, scorable, arm):
    """Every device-text defect, applied to the EMITTED strings and recompiled."""
    out: Dict[str, Any] = {}
    plan = mutation_plan(product, scorable)
    try:
        for name, transform in SOURCE_MUTATIONS.items():
            required = MUTATION_REQUIRES_CODE.get(name)
            leg_plan = list(plan)
            if required is not None:
                leg_plan = [e for e in leg_plan if required in spec_codes(e[0])]
            if name in MUTATION_REQUIRES_A_PHASE:
                leg_plan = [e for e in leg_plan if any(e[0]["k"])]
            sites = backend.set_source_transform(transform, arm)
            if sites == 0:
                out[name] = {"armed": False,
                             "why": (f"matched {sites} site(s); the mutation and "
                                     f"the emitted source have drifted apart")}
                log(f"[src-mut] {name}: NOT ARMED")
                backend.set_source_transform(None, arm)
                results["source_mutations"] = out
                save(results, out_path)
                continue
            legs = [curl_case(backend, xp, spec, sub_step, courant, value_class,
                              "fmad_false", arm)
                    for spec, sub_step, courant, value_class in leg_plan]
            backend.set_source_transform(None, arm)
            scored = _score(legs, name not in NULL_MUTATIONS)
            if name in INHERITED_MUTATIONS:
                # THE NARROWER QUESTION, asked and labelled as such: this leg
                # belongs to gate_cuda_complex and is re-armed here only to show
                # the fold delta did not disable it. The per-fixture table below
                # is the evidence; the fraction is not the claim.
                scored["verdict"] = ("CAUGHT" if scored["caught"] else "UNCAUGHT")
                scored["bar"] = ("inherited: caught on at least one fixture "
                                 "(full scoring is gate_cuda_complex's)")
            out[name] = dict(scored, armed=True, sites=sites,
                             requires_code=required,
                             requires_a_phase=name in MUTATION_REQUIRES_A_PHASE,
                             inherited=name in INHERITED_MUTATIONS,
                             legs_in_plan=len(plan), legs_scored=len(leg_plan),
                             caught_on=[f"{c['label']}::{c['sub_step']}"
                                        for c in legs if leg_caught(c)],
                             cases=legs)
            log(f"[src-mut] {name}: caught {scored['caught']}/{scored['ran']} "
                f"sites={sites} -> {out[name]['verdict']}")
            results["source_mutations"] = out
            save(results, out_path)
    finally:
        backend.set_source_transform(None, arm)
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results) -> Dict[str, Any]:
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]
    reasons: List[str] = []

    arms: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        codes = case["structure"]["boundary_codes"] or []
        key = "folded_periodic" if BC_MIRROR_PERIODIC in codes else "folded_metallic"
        entry = arms.setdefault(key, {"cases": 0, "single_identical": 0,
                                      "multi_cases": 0, "multi_identical": 0,
                                      "differing_words": 0, "labels": set()})
        entry["cases"] += 1
        entry["labels"].add(case["label"])
        entry["single_identical"] += bool(case["single_launch"]["bit_identical"])
        if "multi_step" in case:
            entry["multi_cases"] += 1
            entry["multi_identical"] += bool(case["multi_step"]["bit_identical"])
        entry["differing_words"] += case["single_launch"]["differing_floats"]
    for entry in arms.values():
        entry["labels"] = sorted(entry["labels"])

    if not scored:
        reasons.append("no curl case was scored at all")
    for arm in ("folded_periodic", "folded_metallic"):
        if arm not in arms:
            reasons.append(f"the sweep contained no {arm} case; the fold split has "
                           f"two terminations and releasing on one is releasing on "
                           f"half the admitted set")
            continue
        entry = arms[arm]
        if entry["single_identical"] != entry["cases"]:
            reasons.append(
                f"{arm}: single-launch divergence on "
                f"{entry['cases'] - entry['single_identical']} of {entry['cases']} "
                f"cases ({entry['differing_words']} words)")
        if entry["multi_identical"] != entry["multi_cases"]:
            reasons.append(
                f"{arm}: multi-step divergence on "
                f"{entry['multi_cases'] - entry['multi_identical']} of "
                f"{entry['multi_cases']} cases")

    constitutive = results.get("constitutive", {}).get("fmad_false", [])
    per_side: Dict[str, Dict[str, Any]] = {}
    for case in constitutive:
        entry = per_side.setdefault(case["side"], {"cases": 0, "single": 0,
                                                   "multi_cases": 0, "multi": 0,
                                                   "skipped": 0})
        if case.get("skipped"):
            entry["skipped"] += 1
            continue
        entry["cases"] += 1
        entry["single"] += bool(case["single_launch"]["bit_identical"])
        entry["multi_cases"] += 1
        entry["multi"] += bool(case["multi_step"]["bit_identical"])
    deferred = collections.Counter()
    for case in constitutive:
        if "already admits this configuration directly" in str(
                (case.get("predicate") or {}).get("reason") or ""):
            deferred[case["side"]] += 1
    deferral_notes: List[str] = []
    for side in SIDES:
        entry = per_side.get(side)
        if not entry or entry["cases"] == 0:
            if deferred[side] and deferred[side] == (entry or {}).get("skipped"):
                # NOT A GAP. On 2026-08-20 coverage.covers_real_pml_complex_
                # constitutive gained admit_fold on a device verdict of its own, so
                # the CERTIFIED pair serves every folded constitutive slot and this
                # family defers to it by name. Scoring zero cases is then the
                # PARTITION working, and calling it a missing leg would push this
                # gate to measure a slot it does not own.
                #
                # RECORDED AS A NOTE, NOT A RELEASE-BLOCKING REASON -- which is
                # what this whole comment already argued while the line below
                # still appended into ``reasons``. The first device run
                # (the GPU host, 2026-09-01) surfaced the contradiction: every curl
                # leg identical, every mutation caught, and ``released`` False
                # on exactly these two lines. The deferral is itself measured
                # (every skipped case named the certified predicate's own
                # admission), so it is reported beside the verdict rather than
                # inside it.
                deferral_notes.append(
                    f"update_{side}: NOT SCORED and NOT A GAP -- all "
                    f"{deferred[side]} cases were deferred to "
                    f"coverage.covers_real_pml_complex_constitutive, which admits "
                    f"a fold directly since 2026-08-20; its verdict is that "
                    f"family's gate's, not this one's")
                continue
            reasons.append(f"update_{side} was never scored; the constitutive "
                           f"admission is an ADMISSION and needs its own device leg")
            continue
        if entry["single"] != entry["cases"] or entry["multi"] != entry["multi_cases"]:
            reasons.append(f"update_{side}: divergence on "
                           f"{entry['cases'] - entry['single']} single / "
                           f"{entry['multi_cases'] - entry['multi']} multi cases")

    mutations = {}
    for group in ("host_mutations", "source_mutations"):
        for name, entry in (results.get(group) or {}).items():
            if not entry.get("armed", True):
                mutations[f"{group}:{name}"] = "NOT ARMED"
                reasons.append(f"{group} {name} was not armed; a mutation that "
                               f"matched nothing tests nothing")
                continue
            mutations[f"{group}:{name}"] = entry["verdict"]
            if entry["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
                reasons.append(f"{group} {name}: {entry['verdict']} "
                               f"({entry['caught']}/{entry['ran']})")

    certifies = bool(results.get("backend_certifies"))
    if not certifies:
        reasons.append("this backend compiles no NVRTC and certifies nothing; the "
                       "verdict below is about the TRANSCRIPTION only")
    return {
        "released": not reasons and certifies,
        "scored_cases": len(scored),
        "deferrals_not_gaps": deferral_notes,
        "arms": {k: dict(v) for k, v in arms.items()},
        "constitutive_per_side": per_side,
        "mutations": mutations,
        "reasons": reasons,
        "claim": ("the hand-CUDA complex FOLDED curl pair is byte-identical to "
                  "stepping.step_B / step_D on a complex64 grid carrying a mirror "
                  "fold, per sub-step, at one launch and at 60, at BOTH fold "
                  "terminations and both plane parities, at an exactly "
                  "representable courant and at one that is not, on physical-band "
                  "and subnormal-band operands; and the CERTIFIED complex "
                  "constitutive pair is byte-identical to stepping.update_H / "
                  "update_E on the same folded runs, scored per side"),
        "does_not_claim": [
            "nothing dispatches these kernels; this gate licenses a predicate "
            "clause, not a wiring",
            "grid.beta on a folded complex grid: that is complex_beta_kernels' and "
            "is refused here by an inherited clause",
            "a folded axis carrying a Bloch phase: stepping's PERIODIC and MIRROR "
            "ghost arms are mutually exclusive, so there is no such run",
            "an off-diagonal chi1inv at update_E: refused by the certified complex "
            "constitutive predicate, fold or no fold",
            "cylindrical coordinates, a conductivity, BFAST, or three simultaneous "
            "fold planes: refused by inherited clauses and untested here",
            "no throughput claim: this is a correctness gate and times nothing",
        ],
    }


def save(results, out_path: str) -> None:
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    global LICENCE_SHAPE, POLICY_SHAPE

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cupy", "host"), default="cupy")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--probe", default=None,
                        help="the expansion probe artifact that licenses the arm")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"),
                        default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    results: Dict[str, Any] = {
        "gate": "cuda_complex_folded",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("is the hand-CUDA complex FOLDED curl pair byte-identical to "
                     "stepping.step_B / step_D on a folded complex64 grid, and is "
                     "the CERTIFIED complex constitutive pair byte-identical to "
                     "update_H / update_E on the same run, per sub-step and side?"),
        "family_corpus_digest": family.corpus_digest(),
        "emitter_corpus_digest": complex_emitter.corpus_digest(),
        "deltas": [label for label, _o, _n, _s in family._deltas_for("step_B")],
    }

    licence = parent.load_licence(args.probe, args.subnormal_policy)
    results["expansion_licence"] = licence
    if licence["usable"]:
        LICENCE_SHAPE = dict(licence["verdict"])
        LICENCE_SHAPE["policy_resolved"] = args.subnormal_policy
        POLICY_SHAPE = args.subnormal_policy
        arm = licence["arm"]
    else:
        # A HOST run binds no arm from an artifact and says so; a DEVICE run
        # without a usable licence is refused outright, because the arm is a
        # measured platform fact and compiling one anyway is guessing.
        if args.backend == "cupy":
            log("[fatal] no usable expansion licence; the arm may not be guessed")
            results["status"] = "refused: no usable expansion licence"
            save(results, args.out)
            return 2
        arm = "FMA_V1"
        results["arm_note"] = ("the host backend binds FMA_V1 by default and "
                               "certifies nothing about which arm this platform "
                               "reproduces; that is the device gate's question")
    results["arm"] = arm

    if args.backend == "cupy":
        if cp is None:
            log("[fatal] --backend cupy but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy,
                                                       _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
        xp = cp
    else:
        results["environment"] = {"python": sys.version.split()[0],
                                  "numpy_version": np.__version__,
                                  "note": ("host-compiled backend: it executes the "
                                           "EMITTED characters but compiles no "
                                           "NVRTC and certifies nothing about it")}
        xp = np

    backend = build_backend(args.backend)
    results["backend_certifies"] = bool(backend.certifies)
    save(results, args.out)

    try:
        for guard, options in GUARD_SETS:
            if args.backend == "host" and guard != "fmad_false":
                continue
            backend.set_guard(options)
            log(f"[guard] {guard} options={options}")
            run_sweep(results, args.out, backend, xp, args.product, guard, arm)
            run_constitutive(results, args.out, backend, xp, args.product, guard, arm)
        backend.set_guard(("--fmad=false",))

        if not args.skip_mutations:
            scorable, excluded = scorable_baselines(results)
            results["mutation_scope"] = {
                "scorable": sorted("::".join(k) for k in scorable),
                "excluded": excluded,
                "why": ("a mutation leg answers 'can this gate see this defect', "
                        "and only an arm whose unmutated baseline is bit-identical "
                        "can answer it"),
            }
            log(f"[mut-scope] {len(scorable)} scorable pairs; "
                f"{len(excluded)} excluded")
            save(results, args.out)
            run_host_mutations(results, args.out, backend, xp, args.product,
                               scorable, arm)
            run_source_mutations(results, args.out, backend, xp, args.product,
                                 scorable, arm)
    finally:
        backend.restore()

    if args.backend == "cupy":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} scored={verdict['scored_cases']}")
    for name, entry in sorted(verdict["arms"].items()):
        log(f"[verdict]   {name}: single {entry['single_identical']}/{entry['cases']} "
            f"multi {entry['multi_identical']}/{entry['multi_cases']} "
            f"diff={entry['differing_words']}")
    for side, entry in sorted(verdict["constitutive_per_side"].items()):
        log(f"[verdict]   update_{side}: single {entry['single']}/{entry['cases']} "
            f"multi {entry['multi']}/{entry['multi_cases']} "
            f"skipped {entry['skipped']}")
    for name, value in sorted(verdict["mutations"].items()):
        log(f"[verdict]   mut {name}: {value}")
    for note in verdict.get("deferrals_not_gaps", ()):
        log(f"[verdict]   (deferral, not a gap) {note}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
