"""Byte-identity gate for the hand-CUDA off-diagonal (tensor epsilon) ``update_E``.

The family under test is ``meep_gpu/cuda_kernels/offdiag_constitutive_kernels.py``
plus its stdlib-only emitter ``offdiag_emitter.py``. Nothing dispatches it; this
gate is what would move ``CERTIFIED_KERNELS`` off empty.

=============================================================================
WHY THIS IS A FILE OF ITS OWN RATHER THAN A FAMILY IN THE SHARED PROBE
=============================================================================

``probe_fused_kernel_bit_identity.py`` drives the curl pair and the plain
constitutive pair, and its mutation machinery rewrites MODULE-LEVEL STRING
CONSTANTS. This family has none: its device source is EMITTED per row mask, so
the thing a mutation has to rewrite is a function's OUTPUT, and the thing a sweep
has to vary is which of the 63 sources gets compiled. Bending the shared probe
around that would change a file two other certified records are cut against.

What is NOT re-implemented: the byte comparator, the operand census, the
subnormal-band draw, the policy install, the NVRTC binary observer and the
device stamp all come from that probe by import. The sibling gates on the other
two tracks are laid out the same way (``gate_triton_offdiag.py`` imports the same
helpers), so the three families' claims stay commensurable.

=============================================================================
THE ORACLE IS THE ARRAY PATH ITSELF
=============================================================================

``stepping.update_E`` on real ``Grid``/``Fields``/``PML`` objects, with the rows
installed through the PUBLIC installer (``Fields.set_epsilon_volumes``), so the
install-time validation, the zero-row drop and the stored-E switch are all in
force. There is no second transcription on the device leg to drift: the kernel is
compared against the thing it claims to reproduce, byte for byte, from ONE frozen
state — the oracle runs, the state is restored, the kernel runs, the six output
arrays are compared as raw words. Never ``allclose``: ``-0.0 == 0.0`` and
``NaN != NaN`` both lie, and this family has a live signed-zero question (the
emitter's zero-padding witness).

``--backend numpy`` swaps the KERNEL side for the emitted source EXECUTED in
float32 by the evaluator in the slice's own test module — one grammar, one copy,
imported rather than duplicated. That backend compiles nothing, so it certifies
nothing; what it does is let the whole harness — case product, arming, catch
accounting, vacuity floors — be exercised and shown to FAIL where it must,
on a laptop, before a device slot is spent. Every mutation's catch rate is
therefore a number twice: once off device and once on it.

=============================================================================
A GATE THAT CANNOT FAIL CERTIFIES NOTHING
=============================================================================

Zero-init is a fixed point of this sub-step, so every case carries two floors and
records both:

* ``oracle_moved`` — the fraction of output words the array path changed from the
  frozen input. A case that moved nothing is refused, not passed.
* ``coupling_max_abs`` — ``f_w - D*inv_eps``, i.e. what the off-diagonal rows
  actually contributed. A case where the coupling is identically zero cannot
  distinguish ANY defect in this family and is refused by name; it is the
  difference between testing this kernel and testing the certified plain one.

The mutation legs are the other half. Source defects are planted in the EMITTED
TEXT and compiled from it — the compile memo is keyed through the source, so a
mutated body is a miss and reaches NVRTC — and every leg records how many
compiles came from the mutated bytes, because a leg reporting a pass for a
mutation it never applied is worse than no leg (§13.4 of the disposition, three
measured instances). Legs that MUST BE UNCAUGHT are carried beside legs that edit
the same expression and must be caught, so "inert" is a measurement.

=============================================================================
GUARDS, POLICIES AND COURANTS
=============================================================================

``--fmad=false`` is CORRECTNESS here, not tuning: the two tail accumulations, the
``D * inv_eps`` product and this family's own two ``pair * coefficient`` products
are all contraction candidates the array path rounds twice. The gate sweeps the
guard ON (primary) and OFF (control), and the control is expected to DIVERGE —
if it does not, the guard is decorative on this sub-step and the record says so
rather than implying evidence it does not have.

COURANT 0.5 IS EXACT IN FLOAT32, so a contracted and an uncontracted expression
round identically and the guard cannot be measured there. 0.35 is not
representable and is where it becomes visible. Both are swept; the summary
reports the control's divergence at the inexact Courant separately.

Run (the GPU host, one verified-empty device; the cache dir MUST carry the policy
token because CuPy's cache key is computed above the strip seam)::

    CUDA_VISIBLE_DEVICES=3 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_offdiag.py --subnormal-policy keep \\
        --out $OUT/keep/gate.json

Laptop (no CUDA; the harness legs run against the evaluator and say so)::

    python -u gate_cuda_offdiag.py --backend numpy --product reduced \\
        --out /tmp/offdiag_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only: no throughput or timing claim is made or possible from
a shared box.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
except ImportError:  # laptop: the harness legs still run against the evaluator
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage, offdiag_emitter  # noqa: E402

log = probe.log
save = probe.save
bit_compare = probe.bit_compare
combine = probe.combine
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

SEED = 20260816

#: The six arrays this sub-step writes. ``f_w`` IS STATE: a tree that gets E right
#: and ``f_w`` wrong is correct for exactly one launch and wrong forever after, so
#: it is compared on every case rather than only in the multi-step leg.
OUTPUTS: Tuple[str, ...] = ("Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez")
STATE: Tuple[str, ...] = ("Dx", "Dy", "Dz") + OUTPUTS
COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: Consecutive launches in the multi-step leg. 60 is the budget both certified
#: hand-CUDA records are cut at, taken from the sibling track's measurement that
#: 8, 10 and 6 steps all passed a divergence 40 did not. "Identical for N steps"
#: is a claim about N.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the recurrence exercised. Held fixed, ``f_w``
#: settles and a 60-step leg becomes a slow single-step leg. The same exact
#: float32 scale is applied on both paths, so it cancels out of the comparison.
_ADVANCE = np.float32(0.97)

#: The two row masks the 186-row corpus drives (the predicate battery's
#: ``offdiag_constitutive_step`` census), plus two the corpus does not: a single
#: live slot, and a mask that leaves two components on the PLAIN arm. The last
#: two are here because "a component with no surviving row keeps the pure diagonal
#: arithmetic" is a property of the emitter (stepping.py:1222-1227), and a sweep
#: over corpus masks alone would never emit that arm beside a coupled one.
GATE_ROW_MASKS: Tuple[Tuple[int, ...], ...] = (
    (1, 0, 0, 1, 0, 0),   # the corpus's 15-row mask
    (1, 1, 1, 1, 1, 1),   # the corpus's full tensor
    (1, 0, 0, 0, 0, 0),   # one slot: Ey and Ez take the plain arm
    (0, 0, 1, 1, 0, 0),   # Ey coupled on both partners, Ex and Ez plain
)

#: Masks closed under the (row, partner) -> (partner, row) transpose. The
#: transpose mutation renames coefficient parameters, so on a mask whose image is
#: not itself it would name a parameter the emitted signature never declared and
#: fail to COMPILE — which is a different event from a defect being caught, and
#: scoring it as one would be a lie about what the leg measured.
TRANSPOSE_CLOSED_MASKS: Tuple[Tuple[int, ...], ...] = (
    (1, 0, 0, 1, 0, 0), (1, 1, 1, 1, 1, 1))

#: (cell size, label). Shape == cell_size * resolution at resolution 1.
SHAPES: Tuple[Tuple[Tuple[float, float, float], str], ...] = (
    ((8.0, 8.0, 8.0), "8x8x8"),
    ((13.0, 17.0, 11.0), "13x17x11"),      # non-power-of-two, unequal extents
    ((1.0, 16.0, 16.0), "1x16x16"),        # an INVARIANT axis: the partner pair
)                                          # is 2*g rather than a ghost zero

BOUNDARY_TRIPLES: Tuple[Tuple[str, str, str], ...] = (
    ("periodic", "periodic", "periodic"),
    ("metallic", "metallic", "periodic"),
    ("metallic", "periodic", "metallic"),
    ("periodic", "metallic", "metallic"),
    ("metallic", "metallic", "metallic"),
)

#: 0.5 is exactly representable and 0.35 is not; only the second can distinguish a
#: contracted expression from an uncontracted one, so the guard control is scored
#: at the inexact one.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: Spatially VARYING coefficients by default. A uniform coefficient is exactly
#: what makes the registration (rather than merely the rounding) invisible:
#: ``u[i]`` and ``u[i+s]`` are then the same number and the four-point hoist
#: becomes an algebraic identity. ``uniform`` is carried as the control that shows
#: the difference.
ROW_FORMS: Tuple[str, ...] = ("varying", "uniform")

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and that
    is the one thing about the device library a laptop cannot supply. Everything
    else the fixture exercises — dtype, shape, contiguity, base addresses, the PML
    vectors, the grid's own boundary resolution — is a real object either way.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def config_viable(cell: Sequence[float],
                  boundaries: Sequence[str]) -> Optional[str]:
    """Why this (shape, boundary) pair cannot be built, or None.

    ``Grid`` REFUSES a metallic axis holding exactly one cell, and says why at
    length: MEEP overrides a unit direction to periodic whatever the k_point says,
    and a conductor on both faces of one cell zeroes the whole family. Filtering
    here keeps that refusal out of the case list instead of turning it into a leg
    failure that says nothing about the kernel.
    """
    for axis, extent in enumerate(cell):
        if extent <= 1.0 and boundaries[axis] == "metallic":
            return (f"axis {axis} holds one cell and cannot be metallic; Grid "
                    f"refuses it and MEEP has no such run")
    return None


def build(xp, cell, boundaries, courant: float):
    """A frozen ``(fields, layer, grid)`` triple with PML storage allocated.

    The thickness rule is the one both certified slices' fixtures use: skip an
    axis too thin to hold a layer. A mirrored axis cannot occur here — the
    predicate refuses folds — so the low-face special case those fixtures carry
    is deliberately absent rather than copied.
    """
    grid = Grid(resolution=1.0, cell_size=tuple(cell), boundaries=tuple(boundaries),
                xp=xp, courant=courant)
    thickness = tuple((0, 0) if grid.shape[axis] < 6 else (2, 2)
                      for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, layer, grid


def install_rows(fields, grid, mask, row_form: str, rng) -> Dict[str, Any]:
    """Install one row mask and a DIAGONAL epsilon through the public installer.

    THREE DISTINCT INVERSE-EPSILON VOLUMES, never one. Binding a single volume for
    all three components is the complex template's own defect, and a fixture that
    handed one array to all three could not see it — measured on the certified
    sibling, where exactly that leg (``c9``) is what the shared-epsilon defect
    needed to become visible.
    """
    xp = grid.xp
    epsilon, inverse = {}, {}
    for component in COMPONENTS:
        values = rng.uniform(1.2, 3.4, size=grid.shape).astype(np.float32)
        epsilon[component] = xp.asarray(values)
        inverse[component] = xp.asarray((np.float32(1.0) / values).astype(np.float32))
    rows: Dict[str, Dict[str, Any]] = {}
    for slot, (row, partner) in enumerate(coverage.OFFDIAG_ROW_SLOTS):
        if not mask[slot]:
            continue
        if row_form == "uniform":
            values = np.full(grid.shape, 0.3125, dtype=np.float32)
        else:
            # Straddling zero: the inverse of a real tensor carries NEGATIVE
            # off-diagonals, and a sign error that only shows on one side of zero
            # is exactly the kind a positive-only draw hides.
            values = rng.uniform(-0.45, 0.45, size=grid.shape).astype(np.float32)
        rows.setdefault(row, {})[partner] = xp.asarray(values)
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=rows)
    return rows


def seed_state(fields, grid, value_class: str, rng) -> Dict[str, np.ndarray]:
    """Physical-band or subnormal-band values in every array the sub-step touches.

    The AUXILIARIES start nonzero in both classes. A zero ``f_w`` makes
    ``kms * prev`` exactly zero on the first launch whatever ``kms`` holds, so a
    mis-indexed coefficient would only show from step two — precisely the
    divergence the multi-step leg exists to catch, and no reason to hide it from
    the single-launch leg as well.

    THE MATERIAL STAYS NORMAL under the band class. Driving the coefficients into
    the band too would make every product underflow and the leg would measure the
    fixture rather than the policy's reach into this arithmetic.
    """
    xp = grid.xp
    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
                for name in STATE}
    elif value_class == "subnormal_band":
        host = subnormal_band_hosts(STATE, tuple(grid.shape), rng)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def half_integer_tables(layer) -> Dict[str, Any]:
    """The six HALF-INTEGER kps/kms views this sub-step reads, flattened.

    Asked of the layer directly rather than of the launcher, because the launcher
    deriving them is one of the things under test: the ``swap_sub_lattice`` host
    mutation hands in the INTEGER pair instead, which is a half-cell error in the
    absorber profile — converged, smooth and wrong.
    """
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}_h").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def integer_tables(layer) -> Dict[str, Any]:
    """The INTEGER pair — the H side's sub-lattice, wrong for E by half a cell."""
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def device_arrays(fields, rows) -> Dict[str, Any]:
    """Device parameter name -> array, for the evaluator backend."""
    arrays = {name: getattr(fields, name) for name in OUTPUTS + ("Dx", "Dy", "Dz")}
    for component in COMPONENTS:
        arrays[f"inv_eps_{component}"] = fields.inverse_epsilon_for(component)
    for row, partners in rows.items():
        for partner, volume in partners.items():
            arrays[f"chi1inv_{row}_{partner}"] = volume
    return arrays


def snapshot(fields) -> Dict[str, np.ndarray]:
    return {name: probe.to_host(getattr(fields, name)).copy() for name in STATE}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields) -> None:
    for name in ("Dx", "Dy", "Dz"):
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------

_SLICE_TESTS: Optional[Any] = None


def slice_module():
    """The slice's own test module: the evaluator AND the grammar it is anchored on.

    Imported, never copied. That module is where the grammar is PINNED —
    ``test_every_line_of_the_device_source_is_executed_or_pinned`` fails if a line
    of an emitted source matches none of its anchored patterns — so a second copy
    here could drift from it, and the laptop leg would then be measuring a tree the
    merge bar does not check.
    """
    global _SLICE_TESTS
    if _SLICE_TESTS is None:
        from meep_gpu.cuda_kernels import (  # noqa: PLC0415
            test_offdiag_constitutive_pml_real as slice_tests)
        _SLICE_TESTS = slice_tests
    return _SLICE_TESTS


def evaluator():
    return slice_module().evaluate_emitted_source


#: Template text the evaluator neither executes nor pins because it carries no
#: arithmetic: the signature, the index decomposition, the coordinate resolution
#: and the face predicates. Kept in step with the slice test's own list.
_SCAFFOLDING: Tuple[str, ...] = (
    'extern "C" __global__', "float* __restrict__", "const float*",
    "int nx, int ny, int nz,", "int bc_x", "int wm_x",
    "int idx = blockIdx.x", "if (idx >= nx * ny * nz) return;",
    "int nyz = ny * nz;", "int k = idx % nz;", "int j = (idx / nz) % ny;",
    "int i = idx / (ny * nz);", "int di = coord_dn", "int dj = coord_dn",
    "int dk = coord_dn", "int at_x = (i == 0)", ") {", "}", "{",
)


def evaluator_blind_lines(text: str) -> List[str]:
    """Lines of an emitted body the NumPy evaluator neither EXECUTES nor pins.

    THE HAZARD THIS EXISTS FOR, measured on this gate's first laptop run. The
    evaluator is a ``finditer`` over an anchored grammar, so a mutation that
    rewrites a statement into a form the grammar does not match is not applied at
    all — it is SILENTLY SKIPPED. Two of this file's defects do exactly that
    (inverting the wall-mask select, commuting the diagonal product), and both
    scored a verdict off the laptop backend that said nothing about the defect:
    one read as a catch because a later line then raised, the other as a partial
    catch because skipping the mask is a different defect that happens to bite on
    a metallic grid.

    A leg whose mutated text leaves a line here is therefore NOT MEASURABLE on the
    evaluator backend and is recorded as such rather than scored. On the device
    backend every line is executed by the compiler and the question does not
    arise, which is precisely why the two backends' catch tables differ and why
    both are reported.
    """
    slice_tests = slice_module()
    grammar = (slice_tests._TERM_CALL, slice_tests._TOTAL_FIRST,
               slice_tests._TOTAL_ADD, slice_tests._MASK, slice_tests._GS,
               slice_tests._US, slice_tests._SRC_COUPLED,
               slice_tests._SRC_PLAIN, slice_tests._TAIL)
    body = text[text.index('extern "C"'):]
    covered = set()
    for pattern in grammar:
        for match in pattern.finditer(body):
            covered.update(range(match.start(), match.end()))
    blind, offset = [], 0
    for line in body.split("\n"):
        stripped = line.strip()
        span = set(range(offset, offset + len(line)))
        offset += len(line) + 1
        if not stripped or stripped.startswith("//"):
            continue
        if span & covered:
            continue
        if any(token in line for token in _SCAFFOLDING):
            continue
        blind.append(stripped)
    return blind


def classify_for_evaluator(pristine: str, mutated: str) -> Dict[str, Any]:
    """Can the NumPy evaluator see this defect at all, and if not, what guards it?

    Three outcomes, decided mechanically rather than by a hand-kept list:

    * ``arithmetic`` — the edit lands in text the evaluator executes, so the
      laptop backend measures the same defect the device will.
    * ``pinned_half`` — the edit lands in the four small helpers or the welded
      tail, which the evaluator TRANSCRIBES rather than executes. Invisible to the
      laptop backend BY CONSTRUCTION, and what guards it there is the exact-string
      pin in the slice's own tests. On device it is an ordinary catch.
    * ``outside_the_grammar`` — the edit rewrites a statement into a form the
      anchored grammar does not match, so on the laptop backend the mutation is
      not applied. Not scored there; scored normally on device.
    """
    pinned = dict(slice_module().PINNED_HELPERS)
    pinned["constitutive_apply"] = (
        "    float prev = fw[idx];\n    fw[idx] = src;\n"
        "    float a = f[idx] + kps * src;\n    f[idx] = a - kms * prev;")
    broken_pins = [name for name, body in pinned.items() if body not in mutated]
    if broken_pins:
        return {"evaluator_sees_it": False, "why": "pinned_half",
                "broken_pins": broken_pins}
    blind = evaluator_blind_lines(mutated)
    if blind and blind != evaluator_blind_lines(pristine):
        return {"evaluator_sees_it": False, "why": "outside_the_grammar",
                "unmatched_lines": blind[:4]}
    return {"evaluator_sees_it": True, "why": "arithmetic"}


class EvaluatorBackend:
    """The kernel side, executed in NumPy from the emitted text. Compiles nothing.

    Present so the harness can be shown to FAIL where it must without a device.
    It answers ``certifies = False`` and every summary carries that through, so no
    run on this backend can be mistaken for a certification.
    """

    name = "numpy"
    certifies = False

    def __init__(self):
        self.source_transform: Optional[Callable[[str], Tuple[str, int]]] = None
        self.sources_used: Dict[str, str] = {}

    def set_guard(self, guard: Sequence[str]) -> None:
        """No compiler, so no guard. Recorded rather than silently ignored."""

    def set_source_mutation(self, transform) -> None:
        self.source_transform = transform

    def pristine_source(self, mask) -> str:
        return offdiag_emitter.offdiag_source(mask)

    def source_for(self, mask) -> Tuple[str, int]:
        text = self.pristine_source(mask)
        if self.source_transform is None:
            return text, 0
        return self.source_transform(text)

    def launch(self, fields, layer, rows, mask, tables, codes, walls) -> None:
        text, _sites = self.source_for(mask)
        self.sources_used[str(tuple(mask))] = hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        evaluator()(text, device_arrays(fields, rows), tables, codes, walls,
                    tuple(fields.grid.shape))

    def compile_log(self) -> List[Dict[str, Any]]:
        return []

    def clear(self) -> int:
        return 0


class KernelBackend:
    """The shipped CuPy launcher, driven through its own gate door.

    THE GUARD AND THE MUTATION BOTH REACH NVRTC THROUGH THE SOURCE-KEYED MEMO.
    ``_get_kernel`` keys on ``(name, options, policy, source)``, so overriding
    ``_COMPILE_OPTIONS`` or wrapping ``offdiag_launch_source`` is a MISS and the
    compiler sees the bytes this leg intends. The memo is dropped at both ends anyway,
    because a leg that measured a guard it never applied is the failure family the
    disposition records three instances of.

    THE TEXT THIS BACKEND MUTATES AND DIGESTS IS THE LAUNCH TEXT,
    ``offdiag_emitter.offdiag_launch_source``: the certified statement form in the
    own-cell register view, which is what ``_get_kernel`` compiles. A mutation is
    applied AFTER the hoist, so the port rule never sees a mutated helper (two of the
    defects below rewrite ``constitutive_apply`` into a form its hazard check refuses),
    and every digest this backend records -- ``sources_used`` and the compile-log match
    of each mutation leg -- is of bytes NVRTC received.
    """

    name = "cupy"
    certifies = True

    def __init__(self, kernels):
        self.kernels = kernels
        self.source_transform: Optional[Callable[[str], Tuple[str, int]]] = None
        self._pristine_emitter = offdiag_emitter.offdiag_launch_source
        self.sources_used: Dict[str, str] = {}

    def set_guard(self, guard: Sequence[str]) -> None:
        self.kernels._COMPILE_OPTIONS = tuple(guard)
        self.kernels._clear_kernel_cache()

    def set_source_mutation(self, transform) -> None:
        self.source_transform = transform
        backend = self

        def emitted(mask):
            text = backend._pristine_emitter(mask)
            if backend.source_transform is None:
                return text
            mutated, _sites = backend.source_transform(text)
            return mutated

        # Patched on the EMITTER MODULE, which is where ``_get_kernel`` looks the
        # launch text up at call time.
        offdiag_emitter.offdiag_launch_source = emitted
        self.kernels._clear_kernel_cache()

    def pristine_source(self, mask) -> str:
        """The UNMUTATED source, whatever is currently patched onto the emitter.

        Site counting and digesting must never go through the patched name: with a
        transform installed that would apply it TWICE, and the leg would then hash
        a body it never compiled and score its own accounting a false zero.
        """
        return self._pristine_emitter(mask)

    def source_for(self, mask) -> Tuple[str, int]:
        text = self.pristine_source(mask)
        if self.source_transform is None:
            return text, 0
        return self.source_transform(text)

    def launch(self, fields, layer, rows, mask, tables, codes, walls) -> None:
        text, _sites = self.source_for(mask)
        self.sources_used[str(tuple(mask))] = hashlib.sha256(
            text.encode("utf-8")).hexdigest()
        self.kernels.update_E_offdiag_fused_pml_real(
            fields, tables=tables, codes=codes, walls=walls)
        cp.cuda.runtime.deviceSynchronize()

    def compile_log(self) -> List[Dict[str, Any]]:
        return list(self.kernels.compile_cache.compile_log())

    def clear(self) -> int:
        return self.kernels._clear_kernel_cache()


def build_backend(name: str):
    if name == "numpy":
        return EvaluatorBackend()
    if cp is None:
        raise SystemExit("--backend cupy needs CuPy; this host has none")
    from meep_gpu.cuda_kernels import offdiag_constitutive_kernels as kernels
    return KernelBackend(kernels)


# ---------------------------------------------------------------------------
# The defects
# ---------------------------------------------------------------------------
#
# Every one of these is a SILENT wrong answer: a smooth, converged, plausible
# field with the wrong tensor in it. None of them is a crash, and none would be
# caught by a magnitude comparison at any tolerance a physicist would accept.

def _sub(pattern: str, replacement, text: str) -> Tuple[str, int]:
    new, count = re.subn(pattern, replacement, text)
    return new, count


def m_couple_the_wrong_components(text: str) -> Tuple[str, int]:
    """Each term reads the NEXT axis's D volume instead of its partner's.

    The row product then samples the wrong field entirely while every coefficient,
    every shift and every mask stays right — the defect an author makes by
    reading ``cycle_direction`` once and applying it to the volume as well as to
    the axis.
    """
    rotate = {"Dx": "Dy", "Dy": "Dz", "Dz": "Dx"}
    return _sub(r"offdiag_term\(\n        (D[xyz]),",
                lambda m: f"offdiag_term(\n        {rotate[m.group(1)]},", text)


def m_drop_a_row_from_the_volume_sum(text: str) -> Tuple[str, int]:
    """One partner's term silently missing from a component's total.

    Half a tensor. The remaining term is right, the field stays smooth, and the
    anisotropy is simply wrong. Only applies where a component has BOTH slots
    live, which is why the leg names the masks it is armed on.
    """
    return _sub(r"\n    total_(\w+) = total_\1 \+ term_\w+;", "", text)


def m_invert_the_wall_mask(text: str) -> Tuple[str, int]:
    """The metallic wall-coupling select inverted: kept at the wall, zeroed inside.

    ``stepping._mask_metallic_wall_coupling`` zeroes the coupling ON face 0 of the
    declared-metallic axes and leaves it everywhere else (stepping.py:1279-1283).
    """
    return _sub(r"\? 0\.0f : total_(\w+);", r"? total_\1 : 0.0f;", text)


def m_drop_the_wall_mask(text: str) -> Tuple[str, int]:
    """The mask never applied. Its discriminator: it must be a NULL on a grid that
    declares no metallic axis, which is what makes 'the mask is caught' a statement
    about the mask rather than about something else the metallic case changes."""
    return _sub(r"\n *total_\w+ = \(wm_[xyz] && at_[xyz]\) \? 0\.0f : total_\w+;",
                "", text)


def m_over_apply_the_wall_mask(text: str) -> Tuple[str, int]:
    """The mask applied wherever the FACE is, whatever the grid declares — the
    predicate's ``wm_*``/``bc_*`` split collapsed back into one question."""
    return _sub(r"\(wm_([xyz]) && at_([xyz])\)", r"(at_\2 && at_\2)", text)


def m_transpose_the_chi1inv_term_table(text: str) -> Tuple[str, int]:
    """Slot (row, partner) READ where (partner, row) belongs.

    Bit-identity would then hold for a SYMMETRIC tensor and fail for every real
    one. Restricted to the transpose-closed masks: on any other mask the rename
    would name a parameter the emitted signature never declared, and a compile
    failure is not a catch.

    ONLY THE USE IS REWRITTEN, NEVER THE DECLARATION, and that is the whole
    mutation rather than a detail. MEASURED, 2026-08-16, on the first keep leg:
    the first spelling of this defect renamed both, which is a consistent ALPHA
    RENAME — the parameter in each position stays bound to the same volume and is
    still read by the same term, so the arithmetic is untouched and the kernel was
    correctly bit-identical, 8/8. The leg reported "must be caught, was not",
    which is the leg doing its job on a defect that was not one. Rewriting the
    argument of ``offdiag_term`` alone leaves the caller's binding order fixed and
    makes each term read its transpose's volume, which is the silent half-tensor
    this leg is for.
    """
    return _sub(r"(\n        D[xyz], )chi1inv_(E[xyz])_(E[xyz])(, idx,)",
                lambda m: f"{m.group(1)}chi1inv_{m.group(3)}_{m.group(2)}"
                          f"{m.group(4)}", text)


def m_same_direction_shifts(text: str) -> Tuple[str, int]:
    """Both shifts UP: the half-cell registration error the opposite directions
    exist to prevent (stepping.py:1243-1249)."""
    rename = {"di": "ui", "dj": "uj", "dk": "uk"}
    return _sub(r"flat\((\w+), (\w+), (\w+), nyz, nz\)",
                lambda m: "flat({}, {}, {}, nyz, nz)".format(
                    *[rename.get(g, g) for g in m.groups()]), text)


def m_hoist_the_coefficient(text: str) -> Tuple[str, int]:
    """The four-point average times ``u[home]``.

    The same ALGEBRA only for a uniform coefficient, and a different float32
    number even then: ``a*u + b*u`` and ``(a+b)*u`` do not round alike. This is
    the family's one genuinely new arithmetic element — the multiply sitting
    BETWEEN the two shifts — so it is the defect the gate most owes a leg.
    """
    return _sub(
        re.escape("return 0.25f * ((near_pair * u[home])"
                  " + (far_pair * ghosted(u, up)));"),
        "return 0.25f * ((near_pair + far_pair) * u[home]);", text)


def m_own_axis_shift_down(text: str) -> Tuple[str, int]:
    """The component's OWN-axis shift taken down instead of up."""
    out, hits = text, 0
    for up, down in (("flat(ui, j, k, nyz, nz)", "flat(di, j, k, nyz, nz)"),
                     ("flat(i, uj, k, nyz, nz)", "flat(i, dj, k, nyz, nz)"),
                     ("flat(i, j, uk, nyz, nz)", "flat(i, j, dk, nyz, nz)")):
        hits += out.count(up)
        out = out.replace(up, down)
    return out, hits


def m_wrong_own_axis_table(text: str) -> Tuple[str, int]:
    """Every component reads the x coefficient table.

    MEEP's ``dsigw`` is the absorption a component accumulates along the direction
    it POINTS IN, so component 0 reads x, 1 reads y, 2 reads z. Getting it wrong
    is a converged, smooth, entirely wrong absorber rather than a broken field.
    """
    return _sub(r"kps_[xyz]\[[ijk]\], kms_[xyz]\[[ijk]\]", "kps_x[i], kms_x[i]",
                text)


def m_column_major_index(text: str) -> Tuple[str, int]:
    """The linear index decomposed for column-major storage: i and k swap, and
    every component reads another axis's profile. In bounds on every swept shape by
    construction, so it must be caught as a wrong ANSWER and not as a fault."""
    return _sub(re.escape("(i * nyz + j * nz + k)"), "(k * nyz + j * nz + i)", text)


def m_flatten_the_tail(text: str) -> Tuple[str, int]:
    """``((f + kps*src) - kms*prev)`` flattened to ``f + (kps*src - kms*prev)`` —
    the form BOTH complex constitutive kernels in the certified file are written
    as, so it is the defect the template would have introduced here."""
    return _sub(
        re.escape("    float a = f[idx] + kps * src;\n    f[idx] = a - kms * prev;"),
        "    f[idx] = f[idx] + (kps * src - kms * prev);", text)


def m_store_fw_before_reading_prev(text: str) -> Tuple[str, int]:
    """The aliasing trap the array path's copy exists to prevent: ``prev`` becomes
    the value just stored and the recurrence loses its history term."""
    return _sub(
        re.escape("    float prev = fw[idx];\n    fw[idx] = src;"),
        "    fw[idx] = src;\n    float prev = fw[idx];", text)


def m_drop_fw_store(text: str) -> Tuple[str, int]:
    """The auxiliary-only defect: ``f_w`` is never written. E is bit-identical on
    launch one and wrong from launch two, so only the auxiliary comparison and the
    multi-step leg can see it."""
    return _sub(re.escape("    fw[idx] = src;\n"), "", text)


def n_distribute_the_quarter(text: str) -> Tuple[str, int]:
    """NULL: 0.25 distributed over the sum.

    0.25 is an exact power of two, so scaling by it commutes with round-to-nearest
    — the rounding grid scales exactly — and the distributed form is bitwise
    identical AWAY FROM UNDERFLOW. It is NOT a null once ``0.25*x`` is subnormal,
    which is why this leg is armed on the uniform class only and says so.
    """
    return _sub(
        re.escape("return 0.25f * ((near_pair * u[home])"
                  " + (far_pair * ghosted(u, up)));"),
        "return (0.25f * (near_pair * u[home]))"
        " + (0.25f * (far_pair * ghosted(u, up)));", text)


def n_commute_the_diagonal_product(text: str) -> Tuple[str, int]:
    """NULL: ``gs * us`` written ``us * gs``. IEEE multiply commutes bitwise.

    Its discriminator is ``hoist_the_coefficient``, which edits the same kind of
    expression's ASSOCIATION and must be caught — so the pair says the comparator
    is sensitive to grouping and insensitive to operand order, which is a
    measurement rather than a claim.

    BOTH ARMS, not only the plain one. Anchored on the trailing ``;`` this needle
    matched only ``src = gs * us;`` — the arm a component with no surviving row
    takes — so on the full tensor, where every component is coupled, it matched
    NOTHING and the leg ran an unmutated kernel and scored it a null. Matching the
    product itself reaches the coupled arm's ``(gs * us) + total`` as well.
    """
    return _sub(r"gs_(\w+) \* us_(\w+)", r"us_\2 * gs_\1", text)


SOURCE_MUTATIONS: Dict[str, Callable[[str], Tuple[str, int]]] = {
    "couple_the_wrong_components": m_couple_the_wrong_components,
    "drop_a_row_from_the_volume_sum": m_drop_a_row_from_the_volume_sum,
    "invert_the_wall_mask": m_invert_the_wall_mask,
    "drop_the_wall_mask": m_drop_the_wall_mask,
    "over_apply_the_wall_mask": m_over_apply_the_wall_mask,
    "transpose_the_chi1inv_term_table": m_transpose_the_chi1inv_term_table,
    "same_direction_shifts": m_same_direction_shifts,
    "hoist_the_coefficient": m_hoist_the_coefficient,
    "own_axis_shift_down": m_own_axis_shift_down,
    "wrong_own_axis_table": m_wrong_own_axis_table,
    "column_major_index": m_column_major_index,
    "flatten_the_tail": m_flatten_the_tail,
    "store_fw_before_reading_prev": m_store_fw_before_reading_prev,
    "drop_fw_store": m_drop_fw_store,
    "distribute_the_quarter": n_distribute_the_quarter,
    "commute_the_diagonal_product": n_commute_the_diagonal_product,
}

#: Mutations whose verdict must be UNCAUGHT. A battery of only-must-be-caught legs
#: scores identically whether the comparator works or has degenerated into failing
#: everything, so each null is paired with a leg editing the same kind of
#: expression that must be caught.
NULL_MUTATIONS: Tuple[str, ...] = ("distribute_the_quarter",
                                   "commute_the_diagonal_product")

#: Host-side defects: the kernel never chooses these and cannot choose them wrong,
#: but its caller can. They are armed through the launcher's keyword-only gate
#: door, which exists for exactly this.
HOST_MUTATIONS: Tuple[str, ...] = ("swap_sub_lattice", "drop_the_wall_flags",
                                   "metallic_as_periodic")


def apply_host_mutation(name: Optional[str], layer, grid,
                        tables, codes, walls):
    if name is None:
        return tables, codes, walls
    if name == "swap_sub_lattice":
        # The INTEGER coefficients on the E side: half a cell wrong in the
        # absorber profile. A host mutation because the sub-lattice is chosen by
        # the wrapper (stepping.py:948 vs :1015), not by the kernel.
        return integer_tables(layer), codes, walls
    if name == "drop_the_wall_flags":
        return tables, codes, (0, 0, 0)
    if name == "metallic_as_periodic":
        # A wall told to the kernel as a wrap: what porting a hard-coded
        # all-periodic layout produces.
        return tables, (0, 0, 0), walls
    raise ValueError(f"unknown host mutation {name!r}")


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def case_key(shape_label, boundaries, mask, row_form, courant, value_class,
             guard_label, steps) -> str:
    return "|".join([shape_label, "".join(b[0] for b in boundaries),
                     "".join(str(f) for f in mask), row_form,
                     f"C{courant}", value_class, guard_label, f"n{steps}"])


def one_case(backend, xp, cell, shape_label, boundaries, mask, row_form,
             courant, value_class, guard_label, guard, steps,
             source_mutation: Optional[str] = None,
             host_mutation: Optional[str] = None,
             comparator=None,
             compare_fw: bool = True) -> Dict[str, Any]:
    """Oracle and kernel from ONE frozen state; the six outputs compared as words.

    One fixture and one restore. Two separately built configurations could differ
    in their inputs, silently, and the comparison would then be of two runs rather
    than of two trees.
    """
    # A STABLE per-case seed. ``hash()`` of a str is salted per process, so a run
    # seeded from it is not re-runnable to the same bytes — which is the one thing
    # a gate artifact has to be.
    key = case_key(shape_label, boundaries, mask, row_form, courant,
                   value_class, guard_label, steps)
    rng = np.random.default_rng(
        (SEED + int(hashlib.sha256(key.encode("ascii")).hexdigest()[:8], 16))
        % (2 ** 32))
    fields, layer, grid = build(xp, cell, boundaries, courant)
    rows = install_rows(fields, grid, mask, row_form, rng)
    host = seed_state(fields, grid, value_class, rng)
    frozen = snapshot(fields)

    # --- the oracle: the array path itself -------------------------------
    for _ in range(steps):
        stepping.update_E(fields, layer)
        advance_sources(fields)
    oracle = {name: probe.to_host(getattr(fields, name)).copy() for name in OUTPUTS}

    # THE TWO VACUITY FLOORS, taken on the ORACLE's own output.
    moved = sum(int(np.count_nonzero(
        oracle[name].ravel().view(np.uint32)
        != frozen[name].ravel().view(np.uint32))) for name in OUTPUTS)
    total_words = sum(int(oracle[name].size) for name in OUTPUTS)
    diagonal = probe.to_host(fields.Dx) * probe.to_host(
        fields.inverse_epsilon_for("Ex"))
    coupling = probe.to_host(fields.f_w_Ex) - diagonal
    coupling_max = float(np.max(np.abs(coupling.astype(np.float64))))

    # --- the kernel, from the same frozen state --------------------------
    restore(fields, frozen)
    backend.set_guard(guard)
    tables = half_integer_tables(layer)
    codes = tuple(coverage.BC_CODES[kind]
                  for kind in coverage.real_pml_boundary_kinds(grid))
    walls = coverage.offdiag_wall_mask_flags(grid)
    tables, codes, walls = apply_host_mutation(host_mutation, layer, grid,
                                               tables, codes, walls)
    launch_error = None
    try:
        for _ in range(steps):
            backend.launch(fields, layer, rows, mask, tables, codes, walls)
            advance_sources(fields)
    except Exception as exc:  # noqa: BLE001 - a refusal is a result, recorded
        launch_error = f"{type(exc).__name__}: {exc}"[:600]

    compare = comparator or bit_compare
    names = OUTPUTS if compare_fw else COMPONENTS
    parts = ({name: compare(oracle[name], getattr(fields, name)) for name in names}
             if launch_error is None else {})
    verdict = combine(parts) if parts else {"bit_identical": False,
                                            "differing_floats": 0,
                                            "total_floats": 0,
                                            "per_component": {}}

    record: Dict[str, Any] = {
        "key": case_key(shape_label, boundaries, mask, row_form, courant,
                        value_class, guard_label, steps),
        "shape": list(grid.shape),
        "boundaries": list(boundaries),
        "row_mask": list(mask),
        "row_form": row_form,
        "courant": courant,
        "value_class": value_class,
        "guard": guard_label,
        "steps": steps,
        "source_mutation": source_mutation,
        "host_mutation": host_mutation,
        "compare_fw": compare_fw,
        "bit_identical": bool(verdict["bit_identical"]) and launch_error is None,
        "differing_floats": verdict["differing_floats"],
        "total_floats": verdict["total_floats"],
        "max_ulp": verdict.get("max_ulp"),
        "launch_error": launch_error,
        # The floors, per case, so non-vacuity is a property of the RECORD rather
        # than of a reader's trust in the generator.
        "oracle_moved_words": moved,
        "oracle_total_words": total_words,
        "oracle_moved": moved > 0,
        "coupling_max_abs": coupling_max,
        "coupling_is_live": coupling_max > 0.0,
        "operand_census": operand_census(host),
    }
    return record


def case_is_valid(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """A case that could not distinguish anything is REFUSED, not passed."""
    if not record["oracle_moved"]:
        return False, "the array path changed no output word: this fixture is a fixed point"
    if not record["coupling_is_live"]:
        return False, ("the off-diagonal coupling is identically zero: this case "
                       "cannot distinguish any defect in this family")
    return True, None


# ---------------------------------------------------------------------------
# The legs
# ---------------------------------------------------------------------------

def case_product(product: str, steps: int) -> List[Dict[str, Any]]:
    """The swept cases, before the guard axis multiplies them."""
    shapes = SHAPES if product == "full" else SHAPES[:2]
    boundaries = BOUNDARY_TRIPLES if product == "full" else BOUNDARY_TRIPLES[:2]
    masks = GATE_ROW_MASKS if product == "full" else GATE_ROW_MASKS[:2]
    forms = ROW_FORMS if product == "full" else ROW_FORMS[:1]
    courants = COURANTS
    classes = VALUE_CLASSES
    cases = []
    for cell, label in shapes:
        for triple in boundaries:
            if config_viable(cell, triple) is not None:
                continue
            for mask in masks:
                for form in forms:
                    for courant in courants:
                        for value_class in classes:
                            cases.append({
                                "cell": cell, "shape_label": label,
                                "boundaries": triple, "mask": mask,
                                "row_form": form, "courant": courant,
                                "value_class": value_class, "steps": steps})
    return cases


def summarize(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    per_guard: Dict[str, Dict[str, int]] = {}
    for case in cases:
        bucket = per_guard.setdefault(case["guard"], {"identical": 0, "ran": 0,
                                                      "inexact_identical": 0,
                                                      "inexact_ran": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["bit_identical"])
        if case["courant"] == INEXACT_COURANT:
            bucket["inexact_ran"] += 1
            bucket["inexact_identical"] += int(case["bit_identical"])
    by_class: Dict[str, Dict[str, int]] = {}
    for case in cases:
        if case["guard"] != "fmad_false":
            continue
        bucket = by_class.setdefault(case["value_class"], {"identical": 0, "ran": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["bit_identical"])
    census = {"values": 0, "subnormals": 0, "negative_zeros": 0, "zeros": 0}
    for case in cases:
        if case["value_class"] != "subnormal_band":
            continue
        for key in census:
            census[key] += case["operand_census"][key]
    return {
        "per_guard": per_guard,
        "by_value_class": by_class,
        "subnormal_band_operands": census,
        "subnormal_band_is_non_vacuous": census["subnormals"] > 0,
        "every_case_moved_the_oracle": all(c["oracle_moved"] for c in cases),
        "every_case_had_live_coupling": all(c["coupling_is_live"] for c in cases),
        "min_coupling_max_abs": (min(c["coupling_max_abs"] for c in cases)
                                 if cases else None),
    }


def run_bytes(backend, xp, results: Dict[str, Any], out_path: str,
              product: str, steps: int, leg_name: str,
              guards: Sequence[Tuple[str, Tuple[str, ...], bool]] = GUARD_SETS
              ) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []
    plan = case_product(product, steps)
    swept = [g for g in guards if backend.certifies or g[0] == "fmad_false"]
    total = len(plan) * len(swept)
    index = 0
    started = time.time()
    for guard_label, guard, _primary in swept:
        for spec in plan:
            index += 1
            record = one_case(backend, xp, spec["cell"], spec["shape_label"],
                              spec["boundaries"], spec["mask"], spec["row_form"],
                              spec["courant"], spec["value_class"],
                              guard_label, guard, spec["steps"])
            valid, why = case_is_valid(record)
            record["case_is_valid"] = valid
            record["refused_because"] = why
            (cases if valid else refused).append(record)
            log(f"[{leg_name}] {index}/{total} {record['key']} "
                f"identical={record['bit_identical']} "
                f"diff={record['differing_floats']}/{record['total_floats']} "
                f"coupling={record['coupling_max_abs']:.3e}"
                + ("" if valid else f" REFUSED: {why}"))
            results[leg_name] = {"cases": cases, "refused": refused,
                                 "summary": summarize(cases),
                                 "seconds": round(time.time() - started, 1)}
            save(results, out_path)
    return results[leg_name]


def allclose_compare(a: Any, b: Any) -> Dict[str, Any]:
    """The magnitude comparison a byte gate exists to replace, for the harness leg."""
    host_a, host_b = probe.to_host(a), probe.to_host(b)
    identical = bool(np.allclose(host_a, host_b, rtol=1e-5, atol=1e-8,
                                 equal_nan=True))
    return {"bit_identical": identical, "differing_floats": 0 if identical else 1,
            "total_floats": int(host_a.size)}


#: (leg, mutation, kind, expectation, masks it is armed on, why).
#: expectation "caught"   -> every valid case must DIVERGE
#: expectation "uncaught" -> every valid case must stay identical, and that is the
#:                           measurement
#: expectation "null_on_periodic" -> uncaught on an all-periodic grid, caught on a
#:                           metallic one; the pair is what makes the first half
#:                           mean something
MUTATION_LEGS: Tuple[Dict[str, Any], ...] = (
    {"leg": "gate", "mutation": None, "kind": None, "expect": "pass",
     "why": "the shipped source, unmutated, on the mutation leg's own reduced "
            "product — so a divergence there is attributable to the product and "
            "not to the arming."},
    {"leg": "o1_couple_the_wrong_components", "mutation": "couple_the_wrong_components",
     "kind": "source", "expect": "caught",
     "why": "the row product samples the wrong D volume while every coefficient, "
            "shift and mask stays right"},
    {"leg": "o2_drop_a_row_from_the_volume_sum",
     "mutation": "drop_a_row_from_the_volume_sum", "kind": "source",
     "expect": "caught", "masks": ((1, 1, 1, 1, 1, 1), (0, 0, 1, 1, 0, 0)),
     "why": "half a tensor: one partner's term missing from a component's total. "
            "Armed only on masks where a component HAS both slots live."},
    {"leg": "o3_invert_the_wall_mask", "mutation": "invert_the_wall_mask",
     "kind": "source", "expect": "caught",
     "why": "the coupling kept at the wall and zeroed inside it"},
    {"leg": "o4_transpose_the_chi1inv_term_table",
     "mutation": "transpose_the_chi1inv_term_table", "kind": "source",
     "expect": "caught", "masks": TRANSPOSE_CLOSED_MASKS,
     "why": "slot (row, partner) bound where (partner, row) belongs: identical "
            "for a symmetric tensor, wrong for every real one"},
    {"leg": "o5_drop_the_wall_mask", "mutation": "drop_the_wall_mask",
     "kind": "source", "expect": "null_on_periodic",
     "why": "THE DISCRIMINATOR for the mask legs: the same planted defect must be "
            "a catch where a metallic axis is declared and a NULL where none is, "
            "or 'the mask is caught' is a statement about something else"},
    {"leg": "o6_over_apply_the_wall_mask", "mutation": "over_apply_the_wall_mask",
     "kind": "source", "expect": "caught",
     "why": "the wm_*/bc_* split collapsed back into one question"},
    {"leg": "o7_same_direction_shifts", "mutation": "same_direction_shifts",
     "kind": "source", "expect": "caught",
     "why": "both shifts up: the half-cell registration error"},
    {"leg": "o8_hoist_the_coefficient", "mutation": "hoist_the_coefficient",
     "kind": "source", "expect": "caught",
     "why": "the family's one genuinely new arithmetic element — the multiply "
            "between the two shifts — flattened to a four-point average"},
    {"leg": "o9_own_axis_shift_down", "mutation": "own_axis_shift_down",
     "kind": "source", "expect": "caught",
     "why": "the own-axis shift taken down instead of up"},
    {"leg": "o10_wrong_own_axis_table", "mutation": "wrong_own_axis_table",
     "kind": "source", "expect": "caught",
     "why": "MEEP's dsigw confused for one axis: a converged, smooth, entirely "
            "wrong absorber"},
    {"leg": "o11_column_major_index", "mutation": "column_major_index",
     "kind": "source", "expect": "caught",
     "why": "which cell a thread believes it is; in bounds on every swept shape "
            "by construction, so it must be caught as a wrong answer"},
    {"leg": "o12_flatten_the_tail", "mutation": "flatten_the_tail",
     "kind": "source", "expect": "caught",
     "why": "the welded tail's two accumulations flattened into one — the form "
            "both complex constitutive kernels are written as"},
    {"leg": "o13_store_fw_before_reading_prev",
     "mutation": "store_fw_before_reading_prev", "kind": "source",
     "expect": "caught",
     "why": "the aliasing trap the array path's copy prevents"},
    {"leg": "o14_drop_fw_store", "mutation": "drop_fw_store", "kind": "source",
     "expect": "caught",
     "why": "the auxiliary-only defect; E is right on launch one, so only the f_w "
            "comparison and the multi-step leg can see it"},
    {"leg": "h1_swap_sub_lattice", "mutation": "swap_sub_lattice", "kind": "host",
     "expect": "caught",
     "why": "the INTEGER coefficient pair on the E side: half a cell wrong in the "
            "absorber profile. The kernel never chooses; its caller can."},
    {"leg": "h2_drop_the_wall_flags", "mutation": "drop_the_wall_flags",
     "kind": "host", "expect": "null_on_periodic",
     "why": "the grid's declaration lost on the way to the launch. A grid that "
            "declares no metallic axis already asks for (0,0,0), so this leg is "
            "necessarily a null there and a catch where a wall exists — the pair "
            "is the measurement, and scoring it 'caught everywhere' would be "
            "false of the physics rather than of the kernel."},
    {"leg": "h3_metallic_as_periodic", "mutation": "metallic_as_periodic",
     "kind": "host", "expect": "null_on_periodic",
     "why": "a wall told to the kernel as a wrap; an all-periodic grid already "
            "resolves to (0,0,0), so the same pair applies"},
    {"leg": "n1_distribute_the_quarter", "mutation": "distribute_the_quarter",
     "kind": "source", "expect": "uncaught", "classes": ("uniform",),
     "why": "NULL: 0.25 is an exact power of two and scaling by it commutes with "
            "round-to-nearest. NOT a null under the band class, which is why this "
            "leg names the class it is armed on."},
    {"leg": "n2_commute_the_diagonal_product",
     "mutation": "commute_the_diagonal_product", "kind": "source",
     "expect": "uncaught",
     "why": "NULL: IEEE multiply commutes bitwise. Its discriminator is o8, which "
            "edits the same kind of expression's association and must be caught."},
    {"leg": "x1_allclose_hides_the_hoist", "mutation": "hoist_the_coefficient",
     "kind": "source", "expect": "measure", "comparator": "allclose",
     "row_form": "uniform",
     "why": "HARNESS: relax the byte comparison to np.allclose and COUNT how many "
            "cases of a real defect walk through. ON UNIFORM ROWS, which is where "
            "the blindness lives and where the sibling track measured it: there "
            "the hoist is algebraically EQUAL and differs only through "
            "distributivity rounding (~1 ulp relative), far below rtol=1e-5. On "
            "VARYING rows it is a first-order registration defect that allclose "
            "correctly catches at any amplitude, so a leg run there would measure "
            "the wrong claim -- and did, on this gate's first laptop run (0 of 4 "
            "walked through)."},
    {"leg": "x2_no_fw_compare_hides_drop_fw_store", "mutation": "drop_fw_store",
     "kind": "source", "expect": "single_all_uncaught", "compare_fw": False,
     "steps": 1,
     "why": "HARNESS: drop the auxiliary comparison against the defect that really "
            "is auxiliary-only. A single-launch, target-only gate must be TOTALLY "
            "blind to it — R2 measured rather than argued — while o14 catches it "
            "with f_w and the multi-step leg catches it from step two."},
)


def mutation_case_plan(leg: Dict[str, Any], product: str,
                       steps: int) -> List[Dict[str, Any]]:
    """The reduced product one mutation leg is armed over."""
    plan = []
    masks = leg.get("masks", GATE_ROW_MASKS[:2])
    classes = leg.get("classes", ("uniform",))
    triples = (BOUNDARY_TRIPLES[:2] if leg["expect"] != "null_on_periodic"
               else (BOUNDARY_TRIPLES[0], BOUNDARY_TRIPLES[4]))
    shapes = SHAPES[:1] if product == "reduced" else SHAPES[:2]
    for cell, label in shapes:
        for triple in triples:
            if config_viable(cell, triple) is not None:
                continue
            for mask in masks:
                for value_class in classes:
                    plan.append({"cell": cell, "shape_label": label,
                                 "boundaries": triple, "mask": mask,
                                 "row_form": leg.get("row_form", "varying"),
                                 "courant": INEXACT_COURANT,
                                 "value_class": value_class,
                                 "steps": leg.get("steps", steps)})
    return plan


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
    elif expect == "null_on_periodic":
        periodic = [c for c in valid if set(c["boundaries"]) == {"periodic"}]
        metallic = [c for c in valid if "metallic" in c["boundaries"]]
        out["periodic_identical"] = f"{sum(c['bit_identical'] for c in periodic)}/{len(periodic)}"
        out["metallic_identical"] = f"{sum(c['bit_identical'] for c in metallic)}/{len(metallic)}"
        out["as_required"] = bool(
            periodic and metallic
            and all(c["bit_identical"] for c in periodic)
            and not any(c["bit_identical"] for c in metallic))
    elif expect == "single_all_uncaught":
        out["as_required"] = ran > 0 and identical == ran
    elif expect == "measure":
        # A measurement, not a bar: how many cases of a real defect the relaxed
        # comparator lets past. Reported, and the leg is scored on having RUN.
        out["as_required"] = ran > 0
        out["walked_through_allclose"] = identical
    else:
        raise ValueError(f"unknown expectation {expect!r}")
    return out


def run_mutations(backend, xp, results: Dict[str, Any], out_path: str,
                  product: str, steps: int) -> Dict[str, Any]:
    legs: List[Dict[str, Any]] = []
    started = time.time()
    for leg in MUTATION_LEGS:
        leg_started = time.time()
        transform = (SOURCE_MUTATIONS[leg["mutation"]]
                     if leg["kind"] == "source" else None)
        host = leg["mutation"] if leg["kind"] == "host" else None
        comparator = allclose_compare if leg.get("comparator") == "allclose" else None
        compare_fw = leg.get("compare_fw", True)

        # ARM IT, and refuse the leg if the needle matched nothing: a mutation
        # that has drifted away from the text exercises nothing, and a leg
        # reporting a verdict for it is reporting on the shipped source.
        sites = None
        classification: Optional[Dict[str, Any]] = None
        if transform is not None:
            sites = 0
            for mask in leg.get("masks", GATE_ROW_MASKS[:2]):
                _mutated, count = transform(backend.pristine_source(mask))
                sites += count
                if classification is None or classification["evaluator_sees_it"]:
                    # Asked of the certified statement form on both backends: that is
                    # the text the evaluator executes, whichever text is compiled.
                    statement = offdiag_emitter.offdiag_source(mask)
                    classification = classify_for_evaluator(
                        statement, transform(statement)[0])
            backend.set_source_mutation(transform)

        # THE EVALUATOR BACKEND CANNOT SEE EVERY DEFECT, and a leg it cannot see
        # must say so rather than score a number. Skipped here, measured on device.
        skip_reason = None
        if (not backend.certifies and classification is not None
                and not classification["evaluator_sees_it"]):
            skip_reason = classification["why"]
        before_compiles = len(backend.compile_log())

        cases: List[Dict[str, Any]] = []
        if skip_reason is not None:
            backend.set_source_mutation(None)
            record = dict(leg)
            record.update({
                "identical": 0, "ran": 0, "single": "0/0",
                "as_required": None, "mutation_sites": sites,
                "measurable_on_this_backend": False,
                "evaluator_classification": classification,
                "why_not_measured_here": {
                    "pinned_half":
                        "the edit lands in the four pinned helpers or the welded "
                        "tail, which the evaluator TRANSCRIBES rather than "
                        "executes; what guards it off device is the exact-string "
                        "pin in the slice's own tests, and on device it is an "
                        "ordinary catch",
                    "outside_the_grammar":
                        "the mutation rewrites a statement into a form the "
                        "evaluator's anchored grammar does not match, so off "
                        "device it is not applied at all; a verdict here would be "
                        "about the parser, not the kernel",
                }[skip_reason],
                "cases": [], "seconds": 0.0})
            legs.append(record)
            log(f"[mutations] {leg['leg']}: NOT MEASURABLE on backend="
                f"{backend.name} ({skip_reason}); sites={sites}")
            results["mutations"] = {
                "legs": legs,
                "legs_as_required":
                    f"{sum(1 for l in legs if l['as_required'])}/"
                    f"{sum(1 for l in legs if l['as_required'] is not None)}",
                "legs_not_measurable_on_this_backend":
                    sum(1 for l in legs if l.get("measurable_on_this_backend")
                        is False),
                "all_legs_as_required": all(
                    l["as_required"] for l in legs if l["as_required"] is not None),
                "seconds": round(time.time() - started, 1)}
            save(results, out_path)
            continue
        for spec in mutation_case_plan(leg, product, steps):
            record = one_case(backend, xp, spec["cell"], spec["shape_label"],
                              spec["boundaries"], spec["mask"], spec["row_form"],
                              spec["courant"], spec["value_class"],
                              "fmad_false", ("--fmad=false",), spec["steps"],
                              source_mutation=(leg["mutation"] if transform else None),
                              host_mutation=host, comparator=comparator,
                              compare_fw=compare_fw)
            valid, why = case_is_valid(record)
            record["case_is_valid"] = valid
            record["refused_because"] = why
            cases.append(record)

        # DID THE MUTATED BYTES REACH THE COMPILER? apply-time text matching is
        # necessary and not sufficient: between the rewrite and NVRTC sit a memo
        # that has to miss and a disk cache that has to not answer.
        compiles = backend.compile_log()[before_compiles:]
        mutated_digests = set()
        if transform is not None:
            for mask in leg.get("masks", GATE_ROW_MASKS[:2]):
                text, _n = transform(backend.pristine_source(mask))
                mutated_digests.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
        from_mutated = sum(1 for entry in compiles
                           if entry.get("source_sha256") in mutated_digests)

        backend.set_source_mutation(None)
        backend.clear()

        record = dict(leg)
        record.update(adjudicate(leg, cases))
        record["mutation_sites"] = sites
        record["measurable_on_this_backend"] = True
        record["evaluator_classification"] = classification
        record["compiles_on_this_leg"] = len(compiles)
        record["compiles_from_mutated_source"] = from_mutated
        record["mutated_source_reached_the_compiler"] = (
            None if transform is None else (from_mutated > 0))
        record["cases"] = cases
        record["seconds"] = round(time.time() - leg_started, 1)
        if transform is not None and sites == 0:
            record["as_required"] = False
            record["refused"] = ("the mutation matched nothing in any emitted "
                                 "source; it has drifted apart from the emitter "
                                 "and is exercising nothing")
        if backend.certifies and transform is not None and from_mutated == 0:
            record["as_required"] = False
            record["refused"] = ("the mutated bytes never reached the compiler; a "
                                 "verdict for a mutation that never compiled is "
                                 "indistinguishable from one that found nothing")
        legs.append(record)
        log(f"[mutations] {leg['leg']}: expect={leg['expect']} "
            f"{record['single']} as_required={record['as_required']} "
            f"sites={sites} compiles_from_mutated={from_mutated} "
            f"({record['seconds']} s)")
        results["mutations"] = {
            "legs": legs,
            "legs_as_required":
                f"{sum(1 for l in legs if l['as_required'])}/"
                f"{sum(1 for l in legs if l['as_required'] is not None)}",
            "legs_not_measurable_on_this_backend":
                sum(1 for l in legs
                    if l.get("measurable_on_this_backend") is False),
            "all_legs_as_required": all(
                l["as_required"] for l in legs if l["as_required"] is not None),
            "seconds": round(time.time() - started, 1)}
        save(results, out_path)
    return results["mutations"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

LEG_NAMES = ("bytes", "multistep", "mutations")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", default="auto",
                        choices=("auto", "cupy", "numpy"))
    parser.add_argument("--legs", default=",".join(LEG_NAMES))
    parser.add_argument("--product", default="auto", choices=("auto", "full", "reduced"))
    parser.add_argument("--steps", type=int, default=1,
                        help="launches per case in the single-launch leg")
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
        backend_name = "cupy" if cp is not None else "numpy"
    product = args.product
    if product == "auto":
        product = "full" if backend_name == "cupy" else "reduced"

    results: Dict[str, Any] = {
        "gate": "cuda_offdiag_constitutive",
        "kernel": offdiag_emitter.KERNEL_NAME,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": backend_name,
        "certifies": backend_name == "cupy",
        "product": product,
        "legs_requested": list(legs),
        "seed": SEED,
        "multi_step_budget": args.multi_step_budget,
        "emitter_corpus_digest": offdiag_emitter.corpus_digest(),
        # The texts the single COMPILES: the certified ones through the own-cell hoist.
        # The corpus digest above cannot see an edit to the hoist; this one can.
        "launch_corpus_digest": offdiag_emitter.launch_corpus_digest(),
        "subjects": {
            name: probe.source_digest(os.path.join(
                _REPO_API, "meep_gpu", "cuda_kernels", name))
            for name in ("offdiag_emitter.py", "offdiag_constitutive_kernels.py",
                         "own_cell_hoist.py", "coverage.py", "constitutive_kernels.py",
                         "step_curl_kernels.py")},
        "gate_sha256": probe.source_digest(os.path.abspath(__file__)),
    }

    # THE ORDER IS THE WHOLE POINT: the observer wraps NVRTC BELOW the strip so it
    # records the options NVRTC really received, and the policy installs before the
    # first compile. Install-then-compile keeps a planted subnormal;
    # compile-then-install flushes it and stays flushed through a cache clear.
    if backend_name == "cupy":
        results["nvrtc_binary_observer"] = probe.install_nvrtc_binary_observer()
    if args.import_meep_for_host_policy:
        results["meep_import_for_host_policy"] = probe.import_meep_for_host_policy()
    if args.subnormal_policy:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)

    backend = build_backend(backend_name)
    xp = cp if backend_name == "cupy" else _NumpyWearingCupysName()
    if backend_name == "cupy":
        results["environment"] = probe.device_info()
    save(results, out_path)

    if "bytes" in legs:
        run_bytes(backend, xp, results, out_path, product, args.steps, "bytes")
    if "multistep" in legs:
        # THE PRIMARY GUARD ONLY. The control's job is to show the guard is doing
        # work, and one launch per case already shows it; paying 60x for the same
        # demonstration buys nothing the single-launch leg has not already
        # measured, and this leg's cost is what a device slot is spent on.
        run_bytes(backend, xp, results, out_path,
                  "reduced" if product == "full" else product,
                  args.multi_step_budget, "multistep", guards=GUARD_SETS[:1])
    if "mutations" in legs:
        run_mutations(backend, xp, results, out_path, product, args.steps)

    if backend_name == "cupy":
        results["nvrtc_binaries"] = probe.nvrtc_binary_report()

    single = results.get("bytes", {}).get("summary", {})
    multi = results.get("multistep", {}).get("summary", {})
    mutations = results.get("mutations", {})
    primary = single.get("per_guard", {}).get("fmad_false", {})
    control = single.get("per_guard", {}).get("default_no_options", {})
    verdict = {
        "single_launch": f"{primary.get('identical', 0)}/{primary.get('ran', 0)}",
        "multi_step": f"{multi.get('per_guard', {}).get('fmad_false', {}).get('identical', 0)}"
                      f"/{multi.get('per_guard', {}).get('fmad_false', {}).get('ran', 0)}",
        "guard_control_identical_at_inexact_courant":
            f"{control.get('inexact_identical', 0)}/{control.get('inexact_ran', 0)}",
        "guard_control_diverged": (control.get("inexact_ran", 0) > 0
                                   and control.get("inexact_identical", 0)
                                   < control.get("inexact_ran", 0)),
        "subnormal_band_is_non_vacuous": single.get("subnormal_band_is_non_vacuous"),
        "every_case_had_live_coupling": single.get("every_case_had_live_coupling"),
        "mutation_legs_as_required": mutations.get("legs_as_required"),
        "all_mutation_legs_as_required": mutations.get("all_legs_as_required"),
        # ON A CERTIFYING BACKEND THIS MUST BE ZERO. The compiler executes every
        # line, so no defect is invisible for a reason of the harness; a nonzero
        # count would mean a leg was skipped on the run the record is cut from.
        "mutation_legs_not_measurable":
            mutations.get("legs_not_measurable_on_this_backend"),
    }
    verdict["passed"] = bool(
        results["certifies"]
        and primary.get("ran", 0) > 0 and primary["identical"] == primary["ran"]
        and multi.get("per_guard", {}).get("fmad_false", {}).get("ran", 0) > 0
        and (multi["per_guard"]["fmad_false"]["identical"]
             == multi["per_guard"]["fmad_false"]["ran"])
        and single.get("subnormal_band_is_non_vacuous")
        and single.get("every_case_had_live_coupling")
        and single.get("every_case_moved_the_oracle")
        and mutations.get("all_legs_as_required", False)
        and mutations.get("legs_not_measurable_on_this_backend", 1) == 0)
    if not results["certifies"]:
        verdict["why_not_certified"] = (
            "the numpy backend executes the emitted source and compiles nothing; "
            "it exercises the harness, it does not measure NVRTC's output")
    results["verdict"] = verdict
    save(results, out_path)

    log(f"[done] backend={backend_name} single={verdict['single_launch']} "
        f"multi={verdict['multi_step']} "
        f"mutations={verdict['mutation_legs_as_required']} "
        f"guard_control_diverged={verdict['guard_control_diverged']} "
        f"passed={verdict['passed']}")
    if not results["certifies"]:
        return 0 if (mutations.get("all_legs_as_required", True)
                     and primary.get("identical", 0) == primary.get("ran", 0)) else 1
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
