"""Word-identity gate for the CUDA column-serial radial scan, ``cyl_radial_difference``.

THE CLAIM UNDER TEST. ``meep_gpu/cuda_kernels/cylindrical_prefix.py`` ships one kernel,
``cyl_radial_difference`` (bit_parity_0_10 DESIGN.md D26): MEEP's per-chunk
``f_rderiv_int`` (step_db.cpp:99-119) and the radial difference a Bz or Dz cell reads
from the sum of the chunk that owns it. This gate asks whether it reproduces the array
path's ``stepping.cylindrical_rderiv_difference`` -- called exactly as ``stepping.step_B``
and ``stepping.step_D`` call it -- WORD FOR WORD, on both sides, on real and complex64
storage, on one- and many-chunk divisions, under both float32 subnormal policies.

THE ORACLE IS THE ARRAY PATH ON THE SAME BACKEND. On CuPy,
``cylindrical_rderiv_difference`` takes its row loop (one rounded add per row,
``stepping._cylindrical_segment_scan``), because a library ``cumsum`` is a parallel scan
with another association; on NumPy, ``numpy.cumsum``, which accumulates row after row.
Either is MEEP's order, and the kernel is compared with the one its backend runs.

=============================================================================
THE LEGS
=============================================================================

* ``identity`` -- every case, one launch, compared as uint32 words against the oracle
  over the whole target-shaped difference, rows no segment writes included. The launch
  is handed an output buffer pre-filled with a NaN word, so a row the kernel fails to
  write is a differing word rather than a lucky zero.
* source mutations, each MUST BE CAUGHT: the ``rinv`` factor dropped; ``rinv`` read one
  row late; the forward and backward differences swapped; the running total carried
  across segments; the weights dropped; the B side's wall row filled from the last
  stored row; the clearing loop removed.
* host mutations, each MUST BE CAUGHT: the starts line one row late; a segment dropped
  from the list; the other side's ``ir0`` in the weights (the half-cell error); the B
  side's wall row dropped from the sum.
* the declared NULLS, each MUST BE UNCAUGHT: the weight product commuted and the
  ``rinv`` product commuted (IEEE multiplication commutes bitwise). Each is paired with
  a caught leg on the same expression.
* the contraction guard: the same text compiled without ``--fmad=false``.
  ``total + rinv * (here - prev)`` is a contraction candidate, so the unguarded control
  is expected to diverge on some case; a control identical everywhere is reported as
  the guard MEASURED DECORATIVE on these cases, not as a failure.

=============================================================================
THE FLOORS
=============================================================================

* a case whose oracle difference is ``+0`` on every word is refused, not passed;
* a many-chunk case must bind more than one segment, and its oracle must differ from the
  one-chunk sum somewhere (else the restart is invisible on it);
* a source mutation must match a site, and its leg must have compiled the mutated bytes.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU, a fresh CuPy cache per policy)::

    CUDA_VISIBLE_DEVICES=6 CUPY_CACHE_DIR=$OUT/cupy_cache/keep \\
        python -u gate_cuda_cylindrical_scan.py --subnormal-policy keep \\
        --out $OUT/keep/gate.json

Laptop (no CUDA): the NumPy backend runs the same cases and host mutations against a
NumPy transcription of the column-serial loop. IT COMPILES NOTHING AND CERTIFIES NOTHING;
what it settles is whether the harness can fail::

    python -u gate_cuda_cylindrical_scan.py --backend numpy --out /tmp/scan_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case.

:func:`identity_leg` is the identity comparison as a function, so the curl gate
(``gate_cuda_cylindrical_real.py``, leg ``scan_order``) runs it rather than a second
spelling of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
except ImportError:  # laptop: the NumPy backend still runs
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import cylindrical_prefix as cyl_prefix  # noqa: E402

log = probe.log
to_host = probe.to_host

SEED = 20261009
SIDES: Tuple[str, ...] = ("step_B", "step_D")
STORAGES: Tuple[str, ...] = ("real", "complex")
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band", "signed_zeros")

#: The divisions a case binds. ``one_chunk``: no layer, one chunk from the axis.
#: ``layer``: the one-rank division of a high-r layer (the r axis cut at the layer's
#: chunk), read through ``stepping._cylindrical_scan_starts`` as the array path reads
#: it. ``three_chunks``: a hand-made line with three starts, which no one-rank division
#: produces but a lifted run's division can, so the restart is exercised twice.
DIVISIONS: Tuple[str, ...] = ("one_chunk", "layer", "three_chunks")

#: Radial extents and z extents. The odd and the tall ones are where an index or a
#: column-layout defect shows; the long r is where a late restart shows.
SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (16, 1, 20), (9, 1, 17), (12, 1, 48), (40, 1, 6), (5, 1, 8))

#: The word every output buffer is pre-filled with: a quiet NaN whose payload no
#: arithmetic in the kernel can produce, so a row left unwritten is a differing word.
_UNWRITTEN = np.uint32(0x7FC0ABCD)

#: The float32 special words the ``signed_zeros`` class draws from: +0, -0, +1, -1 and
#: the smallest subnormals of both signs (b_battery_states.SPECIAL_WORDS).
_SPECIAL_WORDS = np.array([0x00000000, 0x80000000, 0x3F800000, 0xBF800000,
                           0x00000001, 0x80000001], dtype=np.uint32)


class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``: the fixture's grids are real objects either way."""

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

def build(xp, shape: Tuple[int, int, int], storage: str, with_layer: bool):
    """A Dcyl ``(fields, layer, grid)`` with the source volume of both sides allocated.

    Real storage runs at m = 0 and complex storage at m = 1 (the scan does not read m;
    the storage is what changes the column layout). The layer, when asked for, is on the
    high r face and on z, as the cylindrical curl gates build it.
    """
    nr, _ny, nz = (int(n) for n in shape)
    grid = Grid(resolution=1.0, cell_size=(float(nr), 0.0, float(nz)), cylindrical=True,
                m=0 if storage == "real" else 1, boundaries={"z": "metallic"},
                courant=0.35, xp=xp)
    fields = Fields(grid=grid, force_complex_fields=(storage == "complex"))
    fields.enable_field_storage()
    fields.enable_pml_storage()
    layer = (PML(grid=grid, thickness={"x": (0, max(2, nr // 4)), "z": max(2, nz // 4)})
             if with_layer else None)
    return fields, layer, grid


def seed_source(fields, side: str, value_class: str, rng) -> np.ndarray:
    """Seed the side's source volume (Ep or Hp) and return its host words."""
    source = getattr(fields, cyl_prefix.PREFIX_COMPONENT[side])
    xp = fields.grid.xp
    complex_storage = source.dtype.kind == "c"
    shape = tuple(int(n) for n in source.shape) + ((2,) if complex_storage else ())
    if value_class == "uniform":
        words = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    elif value_class == "subnormal_band":
        words = probe.subnormal_band_hosts(("w",), shape, rng)["w"]
    elif value_class == "signed_zeros":
        words = _SPECIAL_WORDS[rng.integers(0, _SPECIAL_WORDS.size, size=shape)].view(
            np.float32)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    words = np.ascontiguousarray(words, dtype=np.float32)
    host = words.view(np.complex64).reshape(source.shape) if complex_storage else words
    source[...] = xp.asarray(host)
    return words


def division_starts(fields, layer, side: str, division: str):
    """``(starts as the array path takes it, the scan's host tables)`` for one division.

    The array path's argument is ``stepping._cylindrical_scan_starts``' own answer
    (``None`` for one chunk) for the two real divisions, and the hand-made line shaped
    ``(n, 1, 1)`` for ``three_chunks``. The scan's tables are
    ``cylindrical_prefix.radial_scan_tables`` -- the shipped host side -- with the
    hand-made line substituted for ``three_chunks``.
    """
    target = cyl_prefix.RADIAL_TARGET[side]
    xp = fields.grid.xp
    if division == "three_chunks":
        tables = cyl_prefix.radial_scan_tables(fields, None, side)
        n_target = int(tables["n_target"])
        cuts = (0, max(1, n_target // 3), max(2, (2 * n_target) // 3))
        line = np.zeros(n_target, dtype=np.int32)
        for cut in cuts:
            line[cut:] = cut
        tables["starts"] = line
        tables["segs"] = np.ascontiguousarray(np.unique(line), dtype=np.int32)
        return xp.asarray(line.reshape(-1, 1, 1).astype(np.int64)), tables
    pml = layer if division == "layer" else None
    return (stepping._cylindrical_scan_starts(fields, pml, target),
            cyl_prefix.radial_scan_tables(fields, pml, side))


def array_path_difference(fields, side: str, starts):
    """``stepping.cylindrical_rderiv_difference`` called as ``step_B`` / ``step_D`` call it.

    B: Ep extended by the zero wall row, ``ir0 = 0``, ``rows`` target rows, forward.
    D: Hp itself, ``ir0 = 0.5``, ``rows`` target rows, backward (stepping.py, the two
    call sites of the function).
    """
    xp = fields.grid.xp
    source = getattr(fields, cyl_prefix.PREFIX_COMPONENT[side])
    rows = int(source.shape[0])
    if side == "step_B":
        extended = xp.empty((rows + 1,) + tuple(source.shape[1:]), dtype=source.dtype)
        extended[:rows] = source
        extended[rows] = 0
        return stepping.cylindrical_rderiv_difference(xp, extended, 0.0, starts, rows,
                                                      backward=False)
    return stepping.cylindrical_rderiv_difference(xp, source, 0.5, starts, rows,
                                                  backward=True)


# ---------------------------------------------------------------------------
# The two scan backends
# ---------------------------------------------------------------------------

def _unwritten_buffer(xp, shape, dtype):
    out = xp.empty(shape, dtype=dtype)
    words = out.view(xp.uint32) if hasattr(out, "view") else out
    words[...] = _UNWRITTEN
    return out


def scan_numpy(source: np.ndarray, tables: Dict[str, Any]) -> np.ndarray:
    """The column-serial loop, transcribed in float32 NumPy over all columns at once.

    Rounding per operation as the kernel rounds it (``--fmad=false``): NumPy's float32
    elementwise multiply and add round each result once and fuse nothing.
    """
    words = source.view(np.float32) if source.dtype.kind == "c" else source
    rows_stored = int(tables["rows_stored"])
    flat = np.ascontiguousarray(words).reshape(rows_stored, -1)
    ncol = flat.shape[1]
    n_target = int(tables["n_target"])
    out = np.zeros((n_target, ncol), dtype=np.float32)
    w = np.asarray(tables["w"], dtype=np.float32)
    rinv = np.asarray(tables["rinv"], dtype=np.float32)
    starts = np.asarray(tables["starts"])
    backward = bool(tables["backward"])
    zero = np.zeros(ncol, dtype=np.float32)
    for s in (int(v) for v in np.asarray(tables["segs"])):
        total = np.zeros(ncol, dtype=np.float32)
        prev = w[s] * (flat[s] if s < rows_stored else zero)
        for g in range(s + 1, int(tables["rows_src"])):
            here = w[g] * (flat[g] if g < rows_stored else zero)
            after = total + (rinv[g - 1] * (here - prev))
            row = g if backward else g - 1
            if int(starts[row]) == s:
                out[row] = (total - after) if backward else (after - total)
            total, prev = after, here
    shape = (n_target,) + tuple(source.shape[1:])
    if source.dtype.kind == "c":
        return out.reshape(shape + (2,)).view(np.complex64).reshape(shape)
    return out.reshape(shape)


def scan_cuda(source, tables: Dict[str, Any], kernel) -> Any:
    """The SHIPPED launcher (``launch_radial_scan``) into a NaN-filled buffer."""
    device_tables = cyl_prefix.radial_scan_device_tables(tables, cp)
    shape = (int(tables["n_target"]),) + tuple(int(n) for n in source.shape[1:])
    out = _unwritten_buffer(cp, shape, source.dtype)
    cyl_prefix.launch_radial_scan(kernel, source, device_tables, out, cp)
    cp.cuda.runtime.deviceSynchronize()
    return out


def differing_words(left, right) -> Tuple[int, int]:
    a = np.ascontiguousarray(to_host(left)).view(np.uint32).ravel()
    b = np.ascontiguousarray(to_host(right)).view(np.uint32).ravel()
    if a.size != b.size:
        return max(a.size, b.size), max(a.size, b.size)
    return int(np.count_nonzero(a != b)), int(a.size)


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def _needle(old: str, new: str) -> Callable[[str], Tuple[str, int]]:
    def transform(source: str) -> Tuple[str, int]:
        return source.replace(old, new), source.count(old)
    transform.needle = (old, new)  # type: ignore[attr-defined]
    return transform


#: Device-text defects: ``name -> (transform, must_be_caught, why)``.
SOURCE_MUTATIONS: Dict[str, Tuple[Callable[[str], Tuple[str, int]], bool, str]] = {
    "rinv_dropped": (
        _needle("float after = total + (rinv[g - 1] * (here - prev));",
                "float after = total + (here - prev);"), True,
        "the 1/r factor gone: d(r*Fp)/dr summed where (1/r)d(r*Fp)/dr is"),
    "rinv_one_row_late": (
        _needle("(rinv[g - 1] * (here - prev))", "(rinv[g] * (here - prev))"), True,
        "the next row's 1/r: a half-cell error in the radial weights, silent"),
    "forward_backward_swapped": (
        _needle("out[row * ncol + idx] = backward ? (total - after) : (after - total);",
                "out[row * ncol + idx] = backward ? (after - total) : (total - after);"),
        True, "the D side's negated stride on the B side and the reverse"),
    "total_carried_across_segments": (
        _needle("    for (int t = 0; t < n_segs; ++t) {\n"
                "        const int s = segs[t];\n        float total = 0.0f;\n",
                "    float total = 0.0f;\n    for (int t = 0; t < n_segs; ++t) {\n"
                "        const int s = segs[t];\n"),
        True, "MEEP's restart at +0 in every chunk dropped: each chunk's sum starts "
              "from the previous chunk's total"),
    "weights_dropped": (
        _needle("float here = w[g] * (g < rows_stored ? fp[g * ncol + idx] : 0.0f);",
                "float here = (g < rows_stored ? fp[g * ncol + idx] : 0.0f);"), True,
        "the r in d(r*Fp)/dr gone on every row but the segment's first"),
    "wall_row_from_the_last_stored_row": (
        _needle("float here = w[g] * (g < rows_stored ? fp[g * ncol + idx] : 0.0f);",
                "float here = w[g] * (g < rows_stored ? fp[g * ncol + idx] "
                ": fp[(rows_stored - 1) * ncol + idx]);"), True,
        "the B side's wall row filled from the last stored row instead of held at +0: "
        "the last row's forward difference is no longer MEEP's local 0 - F[nr-1]"),
    "clear_loop_removed": (
        _needle("    for (int g = 0; g < n_target; ++g) {\n"
                "        out[g * ncol + idx] = 0.0f;\n    }\n", ""), True,
        "a row no segment writes keeps the buffer's word instead of +0: on the D side "
        "every segment's first row (the axis row included)"),
    # THE NULLS.
    "weight_product_commuted": (
        _needle("float here = w[g] * (g < rows_stored ? fp[g * ncol + idx] : 0.0f);",
                "float here = (g < rows_stored ? fp[g * ncol + idx] : 0.0f) * w[g];"),
        False, "IEEE multiplication commutes bitwise; paired with weights_dropped"),
    "rinv_product_commuted": (
        _needle("(rinv[g - 1] * (here - prev))", "((here - prev) * rinv[g - 1])"),
        False, "IEEE multiplication commutes bitwise; paired with rinv_dropped"),
}

HOST_MUTATIONS: Tuple[str, ...] = ("starts_one_row_late", "segment_dropped",
                                   "ir0_of_the_other_side", "wall_row_dropped")

#: Which divisions and sides a host mutation can bite on. A starts line with one segment
#: has nothing to drop; the wall row exists only on the B side.
HOST_MUTATION_SCOPE: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "starts_one_row_late": (("layer", "three_chunks"), SIDES),
    "segment_dropped": (("layer", "three_chunks"), SIDES),
    "ir0_of_the_other_side": (DIVISIONS, SIDES),
    "wall_row_dropped": (DIVISIONS, ("step_B",)),
}


def host_mutated_tables(name: Optional[str], side: str,
                        tables: Dict[str, Any]) -> Dict[str, Any]:
    """The scan's host tables with one defect a call site can carry."""
    if name is None:
        return tables
    out = dict(tables)
    if name == "starts_one_row_late":
        line = np.array(tables["starts"], dtype=np.int32, copy=True)
        for s in (int(v) for v in np.asarray(tables["segs"])):
            if s > 0:
                first = int(np.argmax(line == s))
                line[first] = line[first - 1] if first > 0 else line[first]
        out["starts"] = line
    elif name == "segment_dropped":
        out["segs"] = np.ascontiguousarray(np.asarray(tables["segs"])[:-1],
                                           dtype=np.int32)
    elif name == "ir0_of_the_other_side":
        other = "step_D" if side == "step_B" else "step_B"
        weights, rinv = stepping._cylindrical_rderiv_rows(
            np, int(tables["rows_src"]), cyl_prefix.PREFIX_IR0[other],
            np.dtype(np.float32))
        out["w"] = np.ascontiguousarray(weights.reshape(-1), dtype=np.float32)
        out["rinv"] = np.ascontiguousarray(rinv.reshape(-1), dtype=np.float32)
    elif name == "wall_row_dropped":
        out["rows_src"] = int(tables["rows_stored"])
    else:
        raise KeyError(f"unknown host mutation {name!r}")
    return out


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def case_plan(product: str) -> List[Tuple[Tuple[int, int, int], str, str, str, str]]:
    shapes = SHAPES if product == "full" else SHAPES[:2]
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    return [(shape, side, storage, division, value_class)
            for shape in shapes for side in SIDES for storage in STORAGES
            for division in DIVISIONS for value_class in classes]


def one_case(backend: str, shape, side: str, storage: str, division: str,
             value_class: str, kernel=None, host_mutation: Optional[str] = None
             ) -> Dict[str, Any]:
    """One seeded source, the array path's difference, the scan's, compared as words."""
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    label = f"{shape}|{side}|{storage}|{division}|{value_class}"
    rng = np.random.default_rng(SEED + int.from_bytes(
        hashlib.sha256(label.encode()).digest()[:4], "big"))
    fields, layer, _grid = build(xp, shape, storage, with_layer=(division == "layer"))
    seed_source(fields, side, value_class, rng)
    case: Dict[str, Any] = {"shape": list(shape), "side": side, "storage": storage,
                            "division": division, "value_class": value_class,
                            "backend": backend, "host_mutation": host_mutation}
    try:
        starts, tables = division_starts(fields, layer, side, division)
    except cyl_prefix.RadialScanRefusal as exc:
        case["skipped"] = f"the scan refuses this division: {exc}"
        case["seconds"] = time.time() - started
        return case
    case["segments"] = [int(v) for v in np.asarray(tables["segs"])]
    reference = array_path_difference(fields, side, starts)
    nonzero = int(np.count_nonzero(np.ascontiguousarray(to_host(reference))
                                   .view(np.uint32)))
    case["oracle_nonzero_words"] = nonzero
    if nonzero == 0:
        case["skipped"] = "the oracle difference is +0 on every word"
        case["seconds"] = time.time() - started
        return case
    if division != "one_chunk":
        one = array_path_difference(fields, side, None)
        restart_visible = differing_words(reference, one)[0]
        case["restart_visible_words"] = restart_visible
        if len(case["segments"]) < 2 or restart_visible == 0:
            case["skipped"] = ("a many-chunk division whose restart is invisible on "
                               "this case (one segment, or the same words as one sum)")
            case["seconds"] = time.time() - started
            return case
    source = getattr(fields, cyl_prefix.PREFIX_COMPONENT[side])
    mutated = host_mutated_tables(host_mutation, side, tables)
    if backend == "cuda":
        produced = scan_cuda(source, mutated,
                             kernel if kernel is not None
                             else cyl_prefix.get_radial_scan_kernel())
    else:
        produced = scan_numpy(np.asarray(to_host(source)), mutated)
    differing, words = differing_words(reference, produced)
    case.update(differing_words=differing, words=words, identical=(differing == 0),
                seconds=time.time() - started)
    return case


def identity_leg(shapes: Sequence[Tuple[int, int, int]], backend: str = "cuda",
                 storages: Sequence[str] = ("real",),
                 divisions: Sequence[str] = ("layer",)) -> Dict[str, Any]:
    """The identity comparison as a function: every shape, both sides, uniform values.

    The curl gate's ``scan_order`` leg runs this on its own shapes, so the scan its
    cases read is measured once, here, rather than spelled twice.
    """
    cases = []
    for shape in shapes:
        for side in SIDES:
            for storage in storages:
                for division in divisions:
                    case = one_case(backend, tuple(shape), side, storage, division,
                                    "uniform")
                    cases.append(case)
                    log(f"[scan] {shape} {side} {storage} {division} "
                        + (f"SKIPPED: {case['skipped'][:60]}" if case.get("skipped")
                           else f"{case['differing_words']}/{case['words']} differing"))
    scored = [c for c in cases if not c.get("skipped")]
    identical = bool(scored) and all(c["identical"] for c in scored)
    return {"cases": cases, "scored": len(scored),
            "identical": sum(1 for c in scored if c["identical"]),
            "all_identical": identical,
            "reading": ("the shipped column-serial scan reproduces "
                        "stepping.cylindrical_rderiv_difference word for word on these "
                        "shapes" if identical else
                        "the shipped scan DIVERGES from the array path's difference, or "
                        "no case was scored")}


# ---------------------------------------------------------------------------
# The sweep, the mutations, the verdict
# ---------------------------------------------------------------------------

def save(results, out_path: str) -> None:
    """Atomic rewrite, with the bytes THIS process imported recorded first."""
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def run_sweep(results, out_path, backend, product, guard, kernel=None) -> None:
    cases: List[Dict[str, Any]] = []
    plan = case_plan(product)
    for index, (shape, side, storage, division, value_class) in enumerate(plan, start=1):
        case = one_case(backend, shape, side, storage, division, value_class, kernel)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        state = (f"SKIPPED: {case['skipped'][:60]}" if case.get("skipped") else
                 f"{'IDENTICAL' if case['identical'] else 'DIVERGED'} "
                 f"{case['differing_words']}/{case['words']}")
        log(f"[{guard}] case {index}/{len(plan)} {shape} {side} {storage} {division} "
            f"{value_class} segs={case.get('segments')} {state} "
            f"({case.get('seconds', 0.0):.2f} s)")


def mutation_plan() -> List[Tuple[Tuple[int, int, int], str, str, str, str]]:
    """The cases a mutation leg is scored on: every side, storage and division on two
    shapes (an odd radial extent among them), uniform values."""
    return [(shape, side, storage, division, "uniform")
            for shape in ((16, 1, 20), (9, 1, 17)) for side in SIDES
            for storage in STORAGES for division in DIVISIONS]


def _score(legs: List[Dict[str, Any]], must_be_caught: bool) -> Dict[str, Any]:
    scored = [c for c in legs if not c.get("skipped")]
    caught = sum(1 for c in scored if not c["identical"])
    if not scored:
        verdict = "NO LEGS"
    elif must_be_caught:
        verdict = "CAUGHT" if caught else "UNCAUGHT"
    else:
        verdict = "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"
    return {"ran": len(scored), "caught": caught, "must_be_caught": must_be_caught,
            "verdict": verdict}


def run_host_mutations(results, out_path, backend) -> None:
    out: Dict[str, Any] = {}
    for name in HOST_MUTATIONS:
        divisions, sides = HOST_MUTATION_SCOPE[name]
        legs = [one_case(backend, shape, side, storage, division, value_class,
                         host_mutation=name)
                for shape, side, storage, division, value_class in mutation_plan()
                if division in divisions and side in sides]
        out[name] = dict(_score(legs, True), cases=legs)
        log(f"[host-mut] {name}: caught {out[name]['caught']}/{out[name]['ran']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)


def run_source_mutations(results, out_path) -> None:
    """Every device-text defect, compiled from the mutated bytes through the shared memo."""
    from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415

    original = cyl_prefix.RADIAL_SCAN_SOURCE
    out: Dict[str, Any] = {}
    for name, (transform, must_be_caught, why) in SOURCE_MUTATIONS.items():
        mutated, sites = transform(original)
        if sites == 0 or mutated == original:
            out[name] = {"armed": False, "must_be_caught": must_be_caught,
                         "why": f"matched {sites} site(s); the needle and the kernel "
                                f"have drifted apart"}
            log(f"[src-mut] {name}: NOT ARMED")
            results["source_mutations"] = out
            save(results, out_path)
            continue
        compile_cache.clear_compile_log()
        digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
        kernel = cyl_prefix.get_radial_scan_kernel(mutated)
        legs = [one_case("cuda", shape, side, storage, division, value_class, kernel)
                for shape, side, storage, division, value_class in mutation_plan()]
        from_mutated = sum(1 for entry in compile_cache.compile_log()
                           if entry["source_sha256"] == digest)
        out[name] = dict(_score(legs, must_be_caught), armed=True, sites=sites, why=why,
                         mutated_source_sha256=digest,
                         kernel_constructions_from_mutated_bytes=from_mutated,
                         cases=legs)
        if from_mutated == 0:
            out[name]["verdict"] = "UNACCOUNTED"
        log(f"[src-mut] {name}: caught {out[name]['caught']}/{out[name]['ran']} "
            f"builds_from_mutated={from_mutated} -> {out[name]['verdict']}")
        results["source_mutations"] = out
        save(results, out_path)


def policy_evidence(results) -> Dict[str, Any]:
    """Did this run compile under the policy it claims? (``keep``: the strip fired and
    no ``-ftz=true`` reached NVRTC; ``flush``: ``-ftz=true`` reached NVRTC.)"""
    stamp = (results.get("subnormal_policy_stamp_after_run") or {}).get("stamp") or {}
    report = results.get("nvrtc_binary_report") or {}
    resolved = stamp.get("resolved")
    calls = int(report.get("nvrtc_calls_observed") or 0)
    reached = report.get("any_ftz_true_reached_nvrtc")
    if calls == 0:
        return {"meets_floor": False, "why": "NVRTC was never observed compiling"}
    if resolved == "keep":
        ok = int(stamp.get("ftz_removed") or 0) > 0 and reached is False
        return {"meets_floor": ok, "resolved": resolved,
                "why": "the strip fired" if ok else "the strip is not evidenced"}
    if resolved == "flush":
        ok = reached is True
        return {"meets_floor": ok, "resolved": resolved,
                "why": "-ftz=true reached NVRTC" if ok else "-ftz=true never reached NVRTC"}
    return {"meets_floor": False, "resolved": resolved, "why": "no policy was installed"}


def summarize(results) -> Dict[str, Any]:
    primary = [c for c in results.get("sweep", {}).get("fmad_false", [])
               if not c.get("skipped")]
    reasons: List[str] = []
    diverged = [c for c in primary if not c["identical"]]
    if not primary:
        reasons.append("no case was scored")
    if diverged:
        reasons.append(f"{len(diverged)} of {len(primary)} cases diverge from the "
                       f"array path's difference")
    for side in SIDES:
        for storage in STORAGES:
            for division in DIVISIONS:
                if not any(c["side"] == side and c["storage"] == storage
                           and c["division"] == division for c in primary):
                    reasons.append(f"no {side} {storage} {division} case was scored")
    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] != "CAUGHT":
            reasons.append(f"host mutation {name} is {leg['verdict']}")
    for name, leg in results.get("source_mutations", {}).items():
        expected = "CAUGHT" if leg.get("must_be_caught") else "NULL CONFIRMED"
        if not leg.get("armed"):
            reasons.append(f"source mutation {name} was not armed")
        elif leg["verdict"] != expected:
            reasons.append(f"source mutation {name} is {leg['verdict']}")
    if results.get("backend") == "cuda":
        evidence = policy_evidence(results)
        results["policy_evidence"] = evidence
        if not evidence["meets_floor"]:
            reasons.append(f"the subnormal policy is not evidenced: {evidence['why']}")
    control = [c for c in results.get("sweep", {}).get("default_no_options", [])
               if not c.get("skipped")]
    guard = {"scored": len(control),
             "diverged": sum(1 for c in control if not c["identical"])}
    guard["reading"] = ("NOT MEASURED" if not control else
                        "the contraction guard is load-bearing" if guard["diverged"] else
                        "MEASURED DECORATIVE on these cases")
    return {"released": not reasons, "reasons": reasons, "scored_cases": len(primary),
            "identical_cases": len(primary) - len(diverged), "guard_control": guard,
            "claim": ("cyl_radial_difference reproduces stepping.cylindrical_rderiv_"
                      "difference word for word, as step_B and step_D call it, on both "
                      "sides, real and complex64 storage, one-, two- and three-chunk "
                      "divisions, on the case matrix in case_plan"),
            "does_not_claim": [
                "a division whose r cuts differ along z: the scan refuses it by name and "
                "the array path steps it (RadialScanRefusal)",
                "no timing of any kind"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    results: Dict[str, Any] = {
        "gate": "cuda_cylindrical_scan",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend, "product": args.product,
        "question": ("does cyl_radial_difference reproduce "
                     "stepping.cylindrical_rderiv_difference word for word?"),
        "kernel_source_sha256": hashlib.sha256(
            cyl_prefix.RADIAL_SCAN_SOURCE.encode("utf-8")).hexdigest(),
    }
    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
        if args.subnormal_policy:
            results["subnormal_policy_install"] = \
                probe.install_subnormal_policy_for_run(args.subnormal_policy, _REPO_API)
        results["environment"] = probe.device_info()
        results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    else:
        results["environment"] = {"python": sys.version.split()[0],
                                  "numpy_version": np.__version__,
                                  "note": "NumPy backend: compiles nothing, certifies nothing"}
    save(results, args.out)

    run_sweep(results, args.out, args.backend, args.product, "fmad_false")
    if args.backend == "cuda":
        saved = cyl_prefix.RADIAL_SCAN_OPTIONS
        try:
            cyl_prefix.RADIAL_SCAN_OPTIONS = ()
            log("[guard] default_no_options")
            run_sweep(results, args.out, args.backend, "reduced", "default_no_options",
                      cyl_prefix.get_radial_scan_kernel())
        finally:
            cyl_prefix.RADIAL_SCAN_OPTIONS = saved
    if not args.skip_mutations:
        run_host_mutations(results, args.out, args.backend)
        if args.backend == "cuda":
            run_source_mutations(results, args.out)
    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
        results["subnormal_policy_stamp_after_run"] = \
            probe.subnormal_policy_stamp(_REPO_API)
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)
    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} "
        f"identical={verdict['identical_cases']}/{verdict['scored_cases']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
