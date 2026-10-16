"""Byte gate for the Metal real-field PML curl and ``dsigw`` constitutive kernels.

THE CLAIM THIS GATE MAKES, and the only one: **byte-identity to ``stepping.py`` on
this host, subject to a declared and CHECKED subnormal-free precondition.** Not a
stated tolerance. The justification is a measured cliff rather than an argument —
scaling the whole field state and re-running the same kernel gives identity at
1e+00, 1e-20 and 1e-30 with an operand census of zero, and total divergence at
1e-38 and 1e-40 with a census in the thousands. A tolerance would have to be
either meaninglessly loose or would falsely admit the 1e-38 case; byte-identity
plus a checked precondition is the only honest shape. Leg 6 is that check, and it
is required to FIRE on the scaled control — a precondition never demonstrated to
fire is decorative.

THE ORACLE IS IN PROCESS. On the Triton track it lived on another box, so gates
emitted per-step state hashes for offline comparison. Here ``stepping.step_B(fields,
pml)`` and the Metal launch run in ONE process against ONE seed, on real
``Grid``/``Fields``/``PML`` objects. That collapses the reference/synthetic split
for the identity question — but the in-file transcription STAYS, in a reduced
role: it is the mutation substrate and the only way to attribute a defect to a
sub-step or to exercise a deliberately wrong configuration. LEG 0 pins the
transcription against ``stepping.py`` over multiple full cycles BEFORE any Metal
kernel is trusted, which is what stops "bit-identical" from meaning the kernel
reproduced whatever the harness wrote twice.

WHAT IS COMPARED: uint32 word equality on float32 storage, never ``allclose``. The
compared arrays are the sub-step's TARGETS **and** its AUXILIARIES — ``fu_*`` for
the curl, ``f_w_*`` for the constitutive. The auxiliaries are STATE, so a kernel
that is right for one launch and wrong forever after only diverges in the
multi-step leg; omitting them is how that defect hides.

THE PML FORM IS CHOSEN PER STRUCTURE CHUNK, as MEEP chooses it and as the array
path steps it (``stepping._apply_curl_regions`` / ``_apply_constitutive_regions``):
the split-field recurrence on a cell whose chunk carries the target's dsigu layer
and the dsig-only form elsewhere; the accumulation on a cell whose chunk carries
the component's own (dsigw) layer and ``f = src`` elsewhere. Every plan binds the
chunk record (``chunk_map.chunk_bounds_record``) its reference steps with — the
one-rank division of each case's layer (``gate_chunk_records.layer_record``), the
division the array path steps a driverless ``Fields`` with, and a two-run record
with random coefficients on the synthetic leg (``two_run_record``) — and the
in-gate transcription reads the same lines.

THE TWELVE LEGS:

  0  transcription  the in-gate NumPy reference vs ``stepping.py``, full cycles
  1  reference      the Metal kernels vs ``stepping.py``, real engine objects
  2  synthetic      odd shapes x boundary triples x dtdx x GUARD x sub-step
  3  constitutive   ``update_H``/``update_E`` with the three named defects
  4  multi_step     B -> H -> D -> E cycles from one state, auxiliaries included
  5  signed_zero    zero-init + a negative-coefficient absorber, census-floored
  6  precondition   the subnormal census, and the control that makes it fire
  7  mutations      armed, launch-counted, three-valued must_catch: the
                    inverted grouping mutations and the per-chunk form battery
  8  engine         plans from the engine's own objects + the disjointness sweep
  9  forms          the form-coverage census: every (component x form) class,
                    in-run and out-of-run cells on each axis and lattice
 10  planted_zero   the two signed-zero cases only a planted -0 reaches: the
                    plain curl spelled through the recurrence with p := 0, and
                    the plain constitutive spelled as the accumulation
 11  first_update_h the run's FIRST update_H (h_separated False) through
                    plan_step + wrap_for_residency: MEEP's start of H, carried
                    around the slot, against the array path; the same
                    composition with the start stripped must differ

CASE DISCIPLINE, inherited and non-negotiable: uint32 compares never allclose;
a non-power-of-two Courant FIRST; every leg proves the step MOVED STATE (a no-op
agreeing with a no-op is trivially identical); predicted nulls recorded WITH
reasons; armed mutations launch-counted with DISARMED / NEEDLE-MISSED / caught
classification; the signed-zero census floored, because a census of 0 is VACUOUS
and not a pass.

OVERFLOW AND NaN NEEDLES ARE EXCLUDED, on this track as on the Triton one: NaN
sign and payload are IEEE-unspecified and every Metal op carries NaN-class
residuals, so a needle there measures the toolchain's mood rather than the
transcription.

STATED WEAKNESS, recorded because it is a real difference from the Triton
certification: there is NO PTX-EQUIVALENT AUDIT here. ``subnormal_policy`` can read
every generated instruction and refuse a compile on a policy violation;
``compile_shader`` exposes no disassembly, so the mutation legs and this gate are
the only arbiters on this executor.

Progress is one flushed line per case and the artifact is rewritten atomically
after every case (the progress-reporting rule), so ``tail`` on the log is the whole status
check and an interrupted run keeps everything up to the failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

# The MPS executor delivers `flush` natively and cannot deliver `keep`, and this
# arm64 host's DEFAULT resolves to keep (MEEP's set_zero_subnormals is a no-op
# under #if HAVE_IMMINTRIN_H, so match_meep measures "keep"). The gate requests
# flush EXPLICITLY, before anything resolves a policy, and stamps the resolution
# into the artifact — the claim is only as good as the precondition it was
# certified under. Under the checked subnormal-free precondition the two policies
# are indistinguishable, which is exactly what leg 6 measures.
os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

from gate_chunk_records import layer_record, two_run_record  # noqa: E402
from meep_gpu import stepping  # noqa: E402
from meep_gpu.chunk_map import (  # noqa: E402
    RECORD_PRESENCE_WORD,
    membership_from_record,
    record_index,
)
from meep_gpu.fields import IYEE_SHIFTS, Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import coverage as metal_coverage  # noqa: E402
from meep_gpu.metal_kernels import launch as metal_launch  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402


# ---------------------------------------------------------------------------
# Artifact plumbing
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Dict[str, Any], path: str) -> None:
    """Atomic rewrite: tmp + fsync + os.replace, after EVERY case."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(payload)  # bytes THIS process imported; see gate_provenance
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def provenance(out_dir: str) -> Dict[str, str]:
    """sha256 of every source whose bytes decide what this gate certified."""
    files = {
        "stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
        "metal_kernels/shaders.py": os.path.join(API_ROOT, "meep_gpu", "metal_kernels",
                                                 "shaders.py"),
        "metal_kernels/coverage.py": os.path.join(API_ROOT, "meep_gpu", "metal_kernels",
                                                  "coverage.py"),
        "metal_kernels/launch.py": os.path.join(API_ROOT, "meep_gpu", "metal_kernels",
                                                "launch.py"),
        "metal_kernels/subnormal.py": os.path.join(API_ROOT, "meep_gpu", "metal_kernels",
                                                   "subnormal.py"),
        "metal_kernels/chunk_bounds.py": os.path.join(API_ROOT, "meep_gpu",
                                                      "metal_kernels", "chunk_bounds.py"),
        "chunk_map.py": os.path.join(API_ROOT, "meep_gpu", "chunk_map.py"),
        "gate_metal_pml.py": os.path.abspath(__file__),
    }
    record = {}
    for label, path in files.items():
        with open(path, "rb") as handle:
            record[label] = hashlib.sha256(handle.read()).hexdigest()
    save({"sources": record, "kernel_sources": {
        label: shaders.source_sha256(source)
        for label, source in shaders.enumerate_sources().items()}},
        os.path.join(out_dir, "provenance.json"))
    return record


def words(array: Any) -> Any:
    return np.ascontiguousarray(array, dtype=np.float32).reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


# ---------------------------------------------------------------------------
# The case matrix
# ---------------------------------------------------------------------------

#: Non-power-of-two FIRST. 0.35 is the odd one, and the PML coefficients are the
#: same family of non-representable multiplicands, so the grouping guard matters
#: at every Courant number and not only at the odd one.
COURANTS: Tuple[float, ...] = (0.35, 0.5)

CONFIGS: Tuple[Tuple[str, Tuple[float, float, float], Any], ...] = (
    ("periodic_all", (1.2, 1.0, 0.9), "periodic"),
    ("metallic_xy", (1.2, 1.0, 0.9), ("metallic", "metallic", "periodic")),
    ("metallic_all", (1.1, 1.0, 0.9), "metallic"),
)

#: PINNED BY NAME, and an edit here silently disarms leg 5. The signed-zero OUTPUT
#: class on the CURL is non-vacuous ONLY WITH A METALLIC AXIS: censused on
#: identical seeding (186 x -0.0 and 141 x +0.0 words in), periodic_all produced
#: 0 x -0.0 on both sub-steps while metallic_xy produced 12 (step_B) and 21
#: (step_D). The ownership mask writing exact zeros through the recurrence is what
#: makes them. A signed-zero leg run only on periodic configurations would census
#: zero and prove nothing — which is why the census has a FLOOR below.
SIGNED_ZERO_CONFIGS: Tuple[str, ...] = ("metallic_xy", "metallic_all")

SUB_STEPS = {
    "step_B": {"targets": ("Bx", "By", "Bz"), "sources": ("Ex", "Ey", "Ez"),
               "backward": False, "suffix": "_h"},
    "step_D": {"targets": ("Dx", "Dy", "Dz"), "sources": ("Hx", "Hy", "Hz"),
               "backward": True, "suffix": ""},
}
SIDES = {
    "H": {"targets": ("Hx", "Hy", "Hz"), "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
          "sources": ("Bx", "By", "Bz"), "suffix": "", "step": "update_H"},
    "E": {"targets": ("Ex", "Ey", "Ez"), "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
          "sources": ("Dx", "Dy", "Dz"), "suffix": "_h", "step": "update_E"},
}

STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
         "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

PERIODIC, METALLIC = shaders.PERIODIC, shaders.METALLIC


def build(cell, boundaries, courant: float, seed: int, scale: float = 1.0,
          signed_zeros: bool = True) -> Tuple[Any, Any, Any]:
    """A real Grid/Fields/PML triple, seeded in the physical band.

    ``signed_zeros`` seeds a +-0 LATTICE into every volume. MEEP keeps float32
    subnormals on arm64 (``set_zero_subnormals`` is a no-op under
    ``#if HAVE_IMMINTRIN_H``), so the signed-zero class is constructible on the
    HOST side here — unlike on a flushing x86 host.
    """
    grid = Grid(resolution=10.0, cell_size=cell, boundaries=boundaries,
                dimensions=3, courant=courant, k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=False)
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(
        epsilon, (np.float32(1.0) / epsilon).astype(np.float32))
    pml = PML(grid=grid,
              thickness=tuple((2, 2) if grid.shape[a] >= 6 else (0, 0)
                              for a in range(3)))
    rng = np.random.default_rng(seed)
    for name in STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        host = (rng.standard_normal(grid.shape) * (0.37 * scale)).astype(np.float32)
        if signed_zeros:
            host.reshape(-1)[::17] = np.float32(-0.0)
            host.reshape(-1)[7::23] = np.float32(0.0)
        array[...] = host
    # A SEEDED STATE IS A MID-RUN ONE: MEEP's start of H from B
    # (stepping.separate_h_from_b) is long past, so the array path's first update_H
    # must not run it again — the kernels never do.
    fields.h_separated = True
    return grid, fields, pml


def snapshot(fields) -> Dict[str, Any]:
    return {name: np.array(getattr(fields, name), copy=True) for name in STATE
            if getattr(fields, name, None) is not None}


def restore(fields, state: Dict[str, Any]) -> None:
    for name, value in state.items():
        getattr(fields, name)[...] = value


def boundary_codes(grid, pml) -> Tuple[int, int, int]:
    kinds = stepping._boundary_kinds(grid, pml)
    return tuple(METALLIC if kind == "metallic" else PERIODIC for kind in kinds)


# ---------------------------------------------------------------------------
# The in-gate transcription: the MUTATION SUBSTRATE, pinned by leg 0
# ---------------------------------------------------------------------------
#
# NOT the reference for the identity question — stepping.py itself is, in process.
# This exists because a whole-step comparison cannot attribute a defect to a
# sub-step and cannot exercise a deliberately wrong configuration at all.

def _shifted(array: Any, axis: int, backward: bool, code: int) -> Any:
    """``stepping._shift_up`` / ``_shift_down`` for one axis, as a gather.

    ``np.roll`` is a pure gather and preserves every bit including signed zeros;
    the metallic wall is then overwritten with an exact ``+0.0``, which is what
    ``other=0.0`` and the kernel's ternary both deliver.
    """
    out = np.roll(array, 1 if backward else -1, axis=axis)
    if code == METALLIC:
        index = [slice(None)] * 3
        index[axis] = 0 if backward else array.shape[axis] - 1
        out[tuple(index)] = np.float32(0.0)
    return out


def _line(record: Any, half: bool, axis: int, n: int) -> Any:
    """One membership line of the record, shaped to broadcast along ``axis``."""
    shape = [1, 1, 1]
    shape[axis] = n
    return membership_from_record(record, half, axis, n).reshape(shape)


def _present(record: Any, axis: int) -> bool:
    """The record's partner-presence bit: a curl partner differenced along ``axis``."""
    return bool(int(record[RECORD_PRESENCE_WORD]) >> axis & 1)


#: Per target, MEEP's (g1 source, g1 axis, g2 source, g2 axis), the source being an
#: index into the sub-step's three source volumes — ``stepping.B_CURL_TERMS`` /
#: ``D_CURL_TERMS``, which leg 0 measures this against.
CURL_OPERANDS: Tuple[Tuple[int, int, int, int], ...] = ((2, 1, 1, 2), (0, 2, 2, 0),
                                                       (1, 0, 0, 1))
CURL_CYCLE: Tuple[Tuple[str, str], ...] = (("y", "z"), ("z", "x"), ("x", "y"))


def reference_curl(targets: Sequence[Any], aux: Sequence[Any],
                   sources: Sequence[Any], coefficients: Dict[str, Any],
                   codes: Sequence[int], backward: bool, dtdx: float,
                   record: Any) -> None:
    """``stepping.step_B`` / ``step_D``, transcribed elementwise, in place.

    The curl is ``stepping._curl_from_operands``: MEEP's ``((g1s - g1) + g2) - g2s``
    with both partners, ``dtdx * (g1s - g1)`` without g2 and ``(-dtdx) * (g2s - g2)``
    without g1, on the record's presence bits. The two forms are
    ``stepping._apply_curl_regions``: on the record's dsigu line for the target's
    own lattice, the split-field recurrence, and ``fu`` takes its first stage;
    everywhere else ``((kms * f) - curl) * sinv`` and ``fu`` keeps its word.
    """
    names = SUB_STEPS["step_D" if backward else "step_B"]["targets"]
    curls = []
    for first, first_axis, second, second_axis in CURL_OPERANDS:
        g1, g2 = sources[first], sources[second]
        g1s = _shifted(g1, first_axis, backward, codes[first_axis])
        g2s = _shifted(g2, second_axis, backward, codes[second_axis])
        has1, has2 = _present(record, first_axis), _present(record, second_axis)
        if has1 and has2:
            curls.append(dtdx * (((g1s - g1) + g2) - g2s))
        elif has1:
            curls.append(dtdx * (g1s - g1))
        elif has2:
            curls.append((-dtdx) * (g2s - g2))
        else:
            curls.append(np.zeros_like(g1))

    # stepping._mask_non_owned_cells: cell 0 of a metallic axis, per target.
    pairs = (((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)) if backward
             else ((0, 0), (1, 1), (2, 2)))
    for target, axis in pairs:
        if codes[axis] == METALLIC:
            index = [slice(None)] * 3
            index[axis] = 0
            curls[target][tuple(index)] = np.float32(0.0)

    # Target 0 takes (dsig, dsigu) = (y, z), 1 (z, x), 2 (x, y) — vec.hpp's
    # cycle_direction, the same triple on both sides.
    for target, (first, second) in enumerate(CURL_CYCLE):
        kms, sinv = coefficients["kms_" + first], coefficients["sinv_" + first]
        kms_u, sinv_u = coefficients["kms_" + second], coefficients["sinv_" + second]
        axis = "xyz".index(second)
        half = IYEE_SHIFTS[names[target]][axis] == 1
        inside = _line(record, half, axis, targets[target].shape[axis])
        previous = aux[target].copy()
        stage = ((previous * kms) - curls[target]) * sinv
        recurrence = (((targets[target] * kms_u) + stage) - previous) * sinv_u
        dsig_only = ((targets[target] * kms) - curls[target]) * sinv
        targets[target][...] = np.where(inside, recurrence, dsig_only)
        aux[target][...] = np.where(inside, stage, previous)


def reference_constitutive(targets: Sequence[Any], aux: Sequence[Any],
                           sources: Sequence[Any],
                           inverse_epsilon: Optional[Sequence[Any]],
                           coefficients: Dict[str, Any], record: Any,
                           side: str) -> None:
    """``stepping._apply_constitutive_regions`` on the component's OWN axis, in place.

    On the record's dsigw line for the component's own lattice, MEEP's
    ``f + ((kps * src) - (kms * prev))`` with ``f_w`` taking ``src``; everywhere else
    ``f = src`` and ``f_w`` keeps its word.
    """
    names = SIDES[side]["targets"]
    for target, axis_name in enumerate("xyz"):
        kps, kms = coefficients["kps_" + axis_name], coefficients["kms_" + axis_name]
        product = (sources[target] if inverse_epsilon is None
                   else sources[target] * inverse_epsilon[target])
        half = IYEE_SHIFTS[names[target]][target] == 1
        inside = _line(record, half, target, targets[target].shape[target])
        previous = aux[target].copy()
        accumulated = targets[target] + ((kps * product) - (kms * previous))
        targets[target][...] = np.where(inside, accumulated, product)
        aux[target][...] = np.where(inside, product, previous)


def curl_coefficients(pml, suffix: str) -> Dict[str, Any]:
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kms", "sinv")}


def constitutive_coefficients(pml, suffix: str) -> Dict[str, Any]:
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kps", "kms")}


# ---------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------

class Counter:
    """A launch-counted wrapper — the DISARMED classification's evidence.

    A mutation leg whose kernel never launched reports its defect as uncaught. The
    count is what makes "the mutant ran and the bytes still matched" a different
    statement from "nothing happened".
    """

    __slots__ = ("function", "launches")

    def __init__(self, function: Any) -> None:
        self.function = function
        self.launches = 0

    def __call__(self, *args: Any) -> Any:
        self.launches += 1
        return self.function(*args)


def run_curl_on_device(state: Dict[str, Any], grid, pml, sub_step: str,
                       codes, dtdx: float, source: Optional[str] = None,
                       contract: str = shaders.CONTRACT_OFF,
                       counter: Optional[List[Counter]] = None,
                       record: Any = None) -> Dict[str, Any]:
    """One curl launch through the SHIPPED plan object, returning the host results.

    Everything goes through ``plan_from_arrays`` -> ``PmlCurlPlan.run`` so the
    bytes this gate certifies are the bytes the engine route would launch.
    ``record`` is the chunk record bound; ``None`` is the one-rank division of the
    case's own layer (``layer_record``), the one the array path steps it with.
    """
    spec = SUB_STEPS[sub_step]
    arrays = {name: np.array(state[name], copy=True)
              for name in tuple(spec["targets"]) + tuple(spec["sources"])}
    arrays.update({"fu_" + n: np.array(state["fu_" + n], copy=True)
                   for n in spec["targets"]})
    flat = {key: np.asarray(value).reshape(-1)
            for key, value in curl_coefficients(pml, spec["suffix"]).items()}
    residency = metal_launch.Residency()
    functions = None
    if source is not None or contract != shaders.CONTRACT_OFF:
        text = source if source is not None else shaders.curl_source(
            codes, spec["backward"], contract)
        function = metal_launch.compile_source(text).pml_curl_step
        if counter is not None:
            function = Counter(function)
            counter.append(function)
        functions = {shaders.CONTRACT_OFF: function}
    plan = metal_launch.plan_from_arrays(
        sub_step, arrays, flat, codes, dtdx, residency, functions=functions,
        chunk_bounds=layer_record(grid, pml) if record is None else record)
    plan.run()
    residency.sync_out()
    return arrays


def run_constitutive_on_device(state: Dict[str, Any], pml, side: str,
                               inverse_epsilon: Optional[Dict[str, Any]] = None,
                               source: Optional[str] = None,
                               contract: str = shaders.CONTRACT_OFF,
                               suffix: Optional[str] = None,
                               counter: Optional[List[Counter]] = None,
                               record: Any = None) -> Dict[str, Any]:
    """One constitutive launch through the SHIPPED plan; ``record`` as
    :func:`run_curl_on_device` takes it, ``None`` being the layer's one-rank record."""
    spec = SIDES[side]
    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    arrays = {name: np.array(state[name], copy=True) for name in names}
    if side == "E":
        arrays.update({"inv_eps_" + n: inverse_epsilon[n] for n in spec["targets"]})
    flat = {key: np.asarray(value).reshape(-1) for key, value in
            constitutive_coefficients(
                pml, spec["suffix"] if suffix is None else suffix).items()}
    residency = metal_launch.Residency()
    functions = None
    if source is not None or contract != shaders.CONTRACT_OFF:
        text = source if source is not None else shaders.constitutive_source(
            side, contract)
        function = metal_launch.compile_source(text).constitutive_step
        if counter is not None:
            function = Counter(function)
            counter.append(function)
        functions = {shaders.CONTRACT_OFF: function}
    plan = metal_launch.plan_constitutive_from_arrays(
        side, arrays, flat, residency, functions=functions,
        chunk_bounds=layer_record(pml.grid, pml) if record is None else record)
    plan.run()
    residency.sync_out()
    return arrays


# ---------------------------------------------------------------------------
# LEG 0 — the transcription, pinned against stepping.py over full cycles
# ---------------------------------------------------------------------------

def leg_transcription(payload: Dict[str, Any], out: str, cycles: int = 3) -> None:
    """The in-gate reference vs ``stepping.py`` itself, over multiple full cycles.

    THIS RUNS FIRST AND NOTHING ELSE IS TRUSTED UNTIL IT PASSES. Without it,
    "bit-identical" could mean the kernel faithfully reproduced whatever this file
    wrote, twice.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        for courant in COURANTS:
            grid, fields, pml = build(cell, boundaries, courant, 20260814)
            initial = snapshot(fields)
            codes = boundary_codes(grid, pml)
            dtdx = float(grid.dt / grid.dx)

            for _ in range(cycles):
                stepping.step_B(fields, pml)
                stepping.update_H(fields, pml)
                stepping.step_D(fields, pml)
                stepping.update_E(fields, pml)
            oracle = snapshot(fields)

            work = {key: np.array(value, copy=True) for key, value in initial.items()}
            inverse = {n: fields.inverse_epsilon_for(n) for n in ("Ex", "Ey", "Ez")}
            record = layer_record(grid, pml)
            for _ in range(cycles):
                reference_curl([work[n] for n in ("Bx", "By", "Bz")],
                               [work["fu_" + n] for n in ("Bx", "By", "Bz")],
                               [work[n] for n in ("Ex", "Ey", "Ez")],
                               curl_coefficients(pml, "_h"), codes, False, dtdx, record)
                reference_constitutive([work[n] for n in ("Hx", "Hy", "Hz")],
                                       [work["f_w_" + n] for n in ("Hx", "Hy", "Hz")],
                                       [work[n] for n in ("Bx", "By", "Bz")],
                                       None, constitutive_coefficients(pml, ""), record,
                                       "H")
                reference_curl([work[n] for n in ("Dx", "Dy", "Dz")],
                               [work["fu_" + n] for n in ("Dx", "Dy", "Dz")],
                               [work[n] for n in ("Hx", "Hy", "Hz")],
                               curl_coefficients(pml, ""), codes, True, dtdx, record)
                reference_constitutive([work[n] for n in ("Ex", "Ey", "Ez")],
                                       [work["f_w_" + n] for n in ("Ex", "Ey", "Ez")],
                                       [work[n] for n in ("Dx", "Dy", "Dz")],
                                       [inverse[n] for n in ("Ex", "Ey", "Ez")],
                                       constitutive_coefficients(pml, "_h"), record, "E")

            bad = {n: differing(work[n], oracle[n]) for n in oracle}
            moved = sum(differing(oracle[n], initial[n]) for n in oracle)
            row = {"case": name, "courant": courant, "cycles": cycles,
                   "shape": [int(v) for v in grid.shape], "moved": moved,
                   "differing": {k: v for k, v in bad.items() if v}}
            rows.append(row)
            log(f"[leg0 transcription] {name:<13} C={courant} cycles={cycles} "
                f"moved={moved:<7} "
                f"{'IDENTICAL' if not row['differing'] else 'DIFFERS ' + str(row['differing'])}"
                f" ({time.time() - started:.1f}s)")
            payload["legs"]["transcription"] = rows
            save(payload, out)
            assert moved > 0, f"VACUOUS: {name} moved no state over {cycles} cycles"
            assert not row["differing"], (
                f"the in-gate transcription is not stepping.py: {row['differing']}")
    # The operand table the hand reference reads is the array path's own term table.
    for terms, sources in ((stepping.B_CURL_TERMS, ("Ex", "Ey", "Ez")),
                           (stepping.D_CURL_TERMS, ("Hx", "Hy", "Hz"))):
        for (first, first_axis, second, second_axis), (dsig, dsigu), term in zip(
                CURL_OPERANDS, CURL_CYCLE, terms):
            assert (sources[first], first_axis, sources[second], second_axis) == (
                term.first, term.first_axis, term.second, term.second_axis), term
            assert (dsig, dsigu) == (term.dsig, term.dsigu), term


# ---------------------------------------------------------------------------
# LEG 1 — the Metal kernels vs stepping.py, on real engine objects
# ---------------------------------------------------------------------------

def leg_reference(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    started = time.time()
    total = 0
    for name, cell, boundaries in CONFIGS:
        for courant in COURANTS:
            for sub_step in SUB_STEPS:
                grid, fields, pml = build(cell, boundaries, courant, 20260814)
                before = snapshot(fields)
                getattr(stepping, sub_step)(fields, pml)
                after = snapshot(fields)
                codes = boundary_codes(grid, pml)
                got = run_curl_on_device(before, grid, pml, sub_step, codes,
                                         float(grid.dt / grid.dx))
                spec = SUB_STEPS[sub_step]
                compared = tuple(spec["targets"]) + tuple(
                    "fu_" + n for n in spec["targets"])
                bad = {n: differing(got[n], after[n]) for n in compared}
                moved = sum(differing(after[n], before[n]) for n in compared)
                total += 1
                row = {"case": name, "courant": courant, "sub_step": sub_step,
                       "shape": [int(v) for v in grid.shape], "moved": moved,
                       "compared": list(compared),
                       "differing": {k: v for k, v in bad.items() if v}}
                rows.append(row)
                log(f"[leg1 reference] case {total}/12 {name:<13} C={courant} "
                    f"{sub_step} shape={tuple(grid.shape)} moved={moved:<6} "
                    f"{'IDENTICAL' if not row['differing'] else 'DIFFERS ' + str(row['differing'])}"
                    f" ({time.time() - started:.1f}s)")
                payload["legs"]["reference"] = rows
                save(payload, out)
                assert moved > 0, f"VACUOUS: {name}/{sub_step} moved no state"
                assert not row["differing"], row


# ---------------------------------------------------------------------------
# LEG 2 — synthetic shapes, and THE GUARD
# ---------------------------------------------------------------------------

SYNTHETIC_SHAPES = (("three_d", (7, 5, 4)), ("one_cell_z", (7, 5, 1)),
                    ("line_z", (1, 1, 11)))
SYNTHETIC_CODES = (("periodic_all", (PERIODIC,) * 3),
                   ("metallic_xy", (METALLIC, METALLIC, PERIODIC)),
                   ("metallic_all", (METALLIC,) * 3))
SYNTHETIC_DTDX = (0.35, 0.5)

#: The shapes on which the contraction guard MUST be measurably effective. The
#: three-dimensional one is the shape the engine steps; the degenerate ones are
#: kept in the matrix because they exercise the ghost rule and the ownership mask
#: at extents where an off-by-one is fatal, and their guard result is recorded
#: rather than required — see :func:`leg_synthetic`.
GUARD_REQUIRED_SHAPES: Tuple[str, ...] = ("three_d",)


def synthetic_state(shape, seed: int, scale: float = 1.0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = {}
    for name in STATE:
        host = (rng.standard_normal(shape) * (0.37 * scale)).astype(np.float32)
        host.reshape(-1)[::13] = np.float32(-0.0)
        state[name] = host
    return state


def synthetic_coefficients(shape, seed: int, stems=("kms", "sinv")) -> Dict[str, Any]:
    """Coefficient vectors in the shape and the RANGE a real PML produces.

    A real absorber's leading ``kms`` entry is NEGATIVE (measured: -1.267 at cell 0
    of a 2-cell layer), so the range below straddles zero deliberately — the
    signed-zero and sign-flip behaviour of the recurrence is only reachable there.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for index, axis in enumerate("xyz"):
        broadcast = [1, 1, 1]
        broadcast[index] = shape[index]
        for stem in stems:
            values = rng.uniform(-1.4, 1.4, size=shape[index]).astype(np.float32)
            out[f"{stem}_{axis}"] = values.reshape(broadcast)
    return out


def leg_synthetic(payload: Dict[str, Any], out: str) -> None:
    """Odd shapes x boundary triples x dtdx x sub-step, plus THE GUARD LEG.

    The guard leg compiles the SAME case twice, once with the contraction
    directive and once without, and asserts identical for the first and NON-
    identical for the second. The guard's effect is MEASURED here; it is never
    asserted from the pragma's presence.

    MEASURED, AND IT IS TRITON FACT (d)'s METAL ANALOGUE: **the guard is not
    byte-uniform across kernel shapes.** On the DEGENERATE shapes — a single cell
    on two axes, with metallic walls that serve exact zeros past them — removing
    the contraction directive changes NO BIT on some sub-steps, because the
    surviving multiply-adds have an operand that makes ``fma`` and ``mul``+``add``
    agree. Those cases are RECORDED AS VACUOUS FOR THE GUARD with their reason and
    counted; they are not silently passed and they are not asserted away. The
    guard IS required to be effective on every three-dimensional case — that is
    the shape the engine steps — and :data:`GUARD_REQUIRED_SHAPES` names them.
    """
    rows: List[Dict[str, Any]] = []
    vacuous: List[Dict[str, Any]] = []
    started = time.time()
    total = 0
    for shape_name, shape in SYNTHETIC_SHAPES:
        for code_name, codes in SYNTHETIC_CODES:
            for dtdx in SYNTHETIC_DTDX:
                for sub_step, spec in SUB_STEPS.items():
                    state = synthetic_state(shape, 424242)
                    coefficients = synthetic_coefficients(shape, 99)
                    record = two_run_record(shape)
                    expect = {n: np.array(state[n], copy=True)
                              for n in tuple(spec["targets"])
                              + tuple("fu_" + n for n in spec["targets"])}
                    reference_curl([expect[n] for n in spec["targets"]],
                                   [expect["fu_" + n] for n in spec["targets"]],
                                   [state[n] for n in spec["sources"]],
                                   coefficients, codes, spec["backward"], dtdx, record)
                    flat = {k: np.asarray(v).reshape(-1)
                            for k, v in coefficients.items()}
                    arrays = {n: np.array(state[n], copy=True)
                              for n in tuple(spec["targets"]) + tuple(spec["sources"])}
                    arrays.update({"fu_" + n: np.array(state["fu_" + n], copy=True)
                                   for n in spec["targets"]})

                    guarded: Dict[str, Dict[str, int]] = {}
                    for mode in shaders.CONTRACT_MODES:
                        work = {k: np.array(v, copy=True) for k, v in arrays.items()}
                        residency = metal_launch.Residency()
                        function = metal_launch.compile_source(
                            shaders.curl_source(codes, spec["backward"],
                                                mode)).pml_curl_step
                        plan = metal_launch.plan_from_arrays(
                            sub_step, work, flat, codes, dtdx, residency,
                            functions={shaders.CONTRACT_OFF: function},
                            chunk_bounds=record)
                        plan.run()
                        residency.sync_out()
                        guarded[mode] = {n: differing(work[n], expect[n])
                                         for n in expect}

                    moved = sum(differing(expect[n], state[n]) for n in expect)
                    total += 1
                    off_bad = {k: v for k, v in guarded[shaders.CONTRACT_OFF].items() if v}
                    fast_total = sum(guarded[shaders.CONTRACT_FAST].values())
                    guard_required = shape_name in GUARD_REQUIRED_SHAPES
                    row = {"shape": shape_name, "dims": list(shape),
                           "boundaries": code_name, "dtdx": dtdx,
                           "sub_step": sub_step, "moved": moved,
                           "contract_off_differing": off_bad,
                           "contract_fast_differing_total": fast_total,
                           "guard_effective": fast_total > 0,
                           "guard_required": guard_required}
                    if fast_total == 0:
                        row["guard_vacuous_reason"] = (
                            "removing the contraction directive changed no bit on "
                            "this shape: with a single cell on two axes and "
                            "metallic walls serving exact zeros past them, the "
                            "surviving multiply-adds have an operand that makes "
                            "fma and mul+add agree. This case certifies the "
                            "arithmetic and NOT the guard")
                        vacuous.append(row)
                    rows.append(row)
                    log(f"[leg2 synthetic] case {total}/36 {shape_name:<10} "
                        f"{code_name:<12} dtdx={dtdx} {sub_step} moved={moved:<6} "
                        f"off={'IDENTICAL' if not off_bad else off_bad} "
                        f"fast_differs={fast_total}"
                        f"{'' if fast_total else ' GUARD-VACUOUS'} "
                        f"({time.time() - started:.1f}s)")
                    payload["legs"]["synthetic"] = rows
                    payload["legs"]["synthetic_guard_vacuous"] = vacuous
                    save(payload, out)
                    assert moved > 0, f"VACUOUS: {shape_name}/{sub_step} moved nothing"
                    assert not off_bad, row
                    assert fast_total > 0 or not guard_required, (
                        f"THE GUARD LEG WENT VACUOUS on a shape where it is "
                        f"REQUIRED: {row}. Removing the contraction directive "
                        f"changed no bit, so this case certifies nothing about the "
                        f"guard")
    effective = sum(1 for row in rows if row["guard_effective"])
    log(f"[leg2 synthetic] guard effective on {effective}/{len(rows)} cases; "
        f"{len(vacuous)} recorded VACUOUS-FOR-THE-GUARD (degenerate shapes only)")
    assert effective > 0, "the guard leg measured nothing anywhere"


# ---------------------------------------------------------------------------
# LEG 3 — the constitutive sub-step, and its three named defects
# ---------------------------------------------------------------------------

def leg_constitutive(payload: Dict[str, Any], out: str) -> None:
    rows: List[Dict[str, Any]] = []
    started = time.time()
    total = 0
    for name, cell, boundaries in CONFIGS:
        for courant in COURANTS:
            for side in SIDES:
                grid, fields, pml = build(cell, boundaries, courant, 20260814)
                before = snapshot(fields)
                inverse = {n: fields.inverse_epsilon_for(n)
                           for n in SIDES["E"]["targets"]}
                getattr(stepping, SIDES[side]["step"])(fields, pml)
                after = snapshot(fields)
                got = run_constitutive_on_device(before, pml, side, inverse)
                spec = SIDES[side]
                compared = tuple(spec["targets"]) + tuple(spec["aux"])
                bad = {n: differing(got[n], after[n]) for n in compared}
                moved = sum(differing(after[n], before[n]) for n in compared)
                total += 1
                row = {"case": name, "courant": courant, "side": side,
                       "shape": [int(v) for v in grid.shape], "moved": moved,
                       "compared": list(compared),
                       "differing": {k: v for k, v in bad.items() if v}}
                rows.append(row)
                log(f"[leg3 constitutive] case {total}/12 {name:<13} C={courant} "
                    f"update_{side} moved={moved:<6} "
                    f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
                    f"({time.time() - started:.1f}s)")
                payload["legs"]["constitutive"] = rows
                save(payload, out)
                assert moved > 0, f"VACUOUS: {name}/update_{side} moved no state"
                assert not row["differing"], row


# ---------------------------------------------------------------------------
# LEG 4 — multi-step cycles, where a stateful auxiliary defect surfaces
# ---------------------------------------------------------------------------

def leg_multi_step(payload: Dict[str, Any], out: str, cycles: int = 4) -> None:
    """B -> H -> D -> E, ``cycles`` times, from one state, WITH THE WALL SEAM.

    The auxiliaries are STATE. A kernel that is right for one launch and wrong
    forever after is identical in legs 1 and 3 and only diverges here.

    ON A WALLED GRID THE SEAM IS REAL AND IS DRIVEN. ``zero_metal_B``/``_D`` clear
    stored cell 0 between the curl and the constitutive sub-step (driver.py:3167 vs
    :3169) and they run on the ARRAY PATH, writing the host arrays the mirrors
    shadow. So this leg brackets them with an explicit ``sync_out``/``sync_in`` and
    DECLARES them synced to the composer — and the residency verdict is asserted
    with that declaration in hand, rather than being worked around. Undeclared,
    the composer refuses this exact composition; leg 7's ``m10`` measures that the
    refusal is load-bearing by holding the mirror across the write and watching
    the field diverge.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        for courant in COURANTS:
            grid, fields, pml = build(cell, boundaries, courant, 20260814)
            walled = any(metal_coverage.zero_metal_axes(grid))
            initial = snapshot(fields)
            for _ in range(cycles):
                stepping.step_B(fields, pml)
                if walled:
                    stepping.zero_metal_B(fields)
                stepping.update_H(fields, pml)
                stepping.step_D(fields, pml)
                if walled:
                    stepping.zero_metal_D(fields)
                stepping.update_E(fields, pml)
            oracle = snapshot(fields)

            restore(fields, initial)
            residency = metal_launch.Residency()
            synced = ("zero_metal_B", "zero_metal_D") if walled else ()
            plan = metal_launch.plan_step(fields, pml, residency=residency,
                                          sources=(), synced=synced)
            assert plan.replaces == ("step_B", "update_H", "step_D", "update_E"), (
                f"{name}: plan_step covered {plan.replaces}, reasons={plan.reasons}")
            assert plan.residency.covered, plan.residency.reasons
            for _ in range(cycles):
                plan.plans["step_B"].run()
                if walled:
                    residency.sync_out()
                    stepping.zero_metal_B(fields)
                    residency.sync_in()
                plan.plans["update_H"].run()
                plan.plans["step_D"].run()
                if walled:
                    residency.sync_out()
                    stepping.zero_metal_D(fields)
                    residency.sync_in()
                plan.plans["update_E"].run()
            residency.sync_out()
            got = snapshot(fields)

            bad = {n: differing(got[n], oracle[n]) for n in oracle}
            moved = sum(differing(oracle[n], initial[n]) for n in oracle)
            mirror_drift = residency.verify()
            row = {"case": name, "courant": courant, "cycles": cycles,
                   "shape": [int(v) for v in grid.shape], "moved": moved,
                   "walled": walled, "synced": list(synced),
                   "selected": dict(plan.selected),
                   "mirror_drift": mirror_drift,
                   "differing": {k: v for k, v in bad.items() if v}}
            rows.append(row)
            log(f"[leg4 multi_step] {name:<13} C={courant} cycles={cycles} "
                f"moved={moved:<7} mirrors={len(residency.names)} "
                f"walled={walled} syncs={residency.syncs_out}/{residency.syncs_in} "
                f"{'IDENTICAL' if not row['differing'] else 'DIFFERS ' + str(row['differing'])}"
                f" ({time.time() - started:.1f}s)")
            payload["legs"]["multi_step"] = rows
            save(payload, out)
            assert moved > 0, f"VACUOUS: {name} moved no state over {cycles} cycles"
            assert not row["differing"], row
            assert not mirror_drift, f"residency drift after the cycle: {mirror_drift}"


# ---------------------------------------------------------------------------
# LEG 5 — zero init and the signed-zero census
# ---------------------------------------------------------------------------

def leg_signed_zero(payload: Dict[str, Any], out: str) -> None:
    """Zero-init with a thin negative-coefficient absorber, plus a +-0 lattice.

    TWO SEEDINGS, because the two sub-step families reach the class differently:

    * the CURL is seeded to exact zeros. A real PML's leading ``kms`` is negative,
      so the recurrence produces ``-0.0`` out of ``+0.0`` — but ONLY where the
      ownership mask writes exact zeros through it, i.e. ONLY WITH A METALLIC AXIS
      (measured: periodic_all censused 0 on both sub-steps). The metallic
      configurations are pinned in :data:`SIGNED_ZERO_CONFIGS` for exactly that
      reason, and the periodic result is recorded as a PREDICTED NULL with its
      reason rather than quietly dropped;
    * the CONSTITUTIVE sub-step is a FIXED POINT under zero init — ``fw`` becomes
      the source, the accumulation adds and subtracts the same zeros, and nothing
      moves. So it gets a ``+-0`` LATTICE instead, with its own vacuity floor.

    A CENSUS OF ZERO IS VACUOUS, NOT PASSED. Every leg below carries a floor.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        grid, fields, pml = build(cell, boundaries, 0.35, 20260814)
        # --- the curl: exact zero init ------------------------------------
        for volume in STATE:
            array = getattr(fields, volume, None)
            if array is not None:
                array[...] = np.float32(0.0)
        zero_state = snapshot(fields)
        codes = boundary_codes(grid, pml)
        for sub_step, spec in SUB_STEPS.items():
            restore(fields, zero_state)
            getattr(stepping, sub_step)(fields, pml)
            after = snapshot(fields)
            got = run_curl_on_device(zero_state, grid, pml, sub_step, codes,
                                     float(grid.dt / grid.dx))
            compared = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
            bad = {n: differing(got[n], after[n]) for n in compared}
            census = sum(subnormal.signed_zero_census(after[n])["negative_zero"]
                         for n in compared)
            expected_nonvacuous = name in SIGNED_ZERO_CONFIGS
            row = {"case": name, "sub_step": sub_step, "seeding": "zero_init",
                   "negative_zero_words": census,
                   "expected_nonvacuous": expected_nonvacuous,
                   "differing": {k: v for k, v in bad.items() if v}}
            if not expected_nonvacuous:
                row["predicted_null_reason"] = (
                    "no metallic axis: the ownership mask writes no exact zeros "
                    "through the recurrence, so no -0.0 output is reachable")
            rows.append(row)
            log(f"[leg5 signed_zero] {name:<13} {sub_step} zero_init "
                f"neg_zero_words={census} "
                f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["signed_zero"] = rows
            save(payload, out)
            assert not row["differing"], row
            if expected_nonvacuous:
                assert census > 0, (
                    f"VACUOUS signed-zero census on {name}/{sub_step}: a census of "
                    f"0 means the class was never constructed, which is not a pass")

        # --- the constitutive: +-0 lattice, because zero init is a fixed point --
        for volume in STATE:
            array = getattr(fields, volume, None)
            if array is None:
                continue
            lattice = np.zeros(grid.shape, dtype=np.float32)
            lattice.reshape(-1)[::3] = np.float32(-0.0)
            lattice.reshape(-1)[1::5] = np.float32(0.25)
            array[...] = lattice
        seeded = snapshot(fields)
        inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}
        for side, spec in SIDES.items():
            restore(fields, seeded)
            getattr(stepping, spec["step"])(fields, pml)
            after = snapshot(fields)
            got = run_constitutive_on_device(seeded, pml, side, inverse)
            compared = tuple(spec["targets"]) + tuple(spec["aux"])
            bad = {n: differing(got[n], after[n]) for n in compared}
            moved = sum(differing(after[n], seeded[n]) for n in compared)
            census = sum(subnormal.signed_zero_census(after[n])["negative_zero"]
                         for n in compared)
            row = {"case": name, "side": side, "seeding": "signed_zero_lattice",
                   "moved": moved, "negative_zero_words": census,
                   "differing": {k: v for k, v in bad.items() if v}}
            rows.append(row)
            log(f"[leg5 signed_zero] {name:<13} update_{side} lattice moved={moved} "
                f"neg_zero_words={census} "
                f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["signed_zero"] = rows
            save(payload, out)
            assert not row["differing"], row
            assert moved > 0, (
                f"VACUOUS: update_{side} on a {name} +-0 lattice moved no state — "
                f"the constitutive sub-step is a fixed point of zero init and this "
                f"seeding exists to escape it")
            assert census > 0, f"VACUOUS signed-zero census on {name}/update_{side}"


# ---------------------------------------------------------------------------
# LEG 6 — the subnormal precondition, and the control that makes it fire
# ---------------------------------------------------------------------------

SUBNORMAL_SCALES = (("physical", 1.0, True), ("small_normal", 1e-20, True),
                    ("subnormal_band", 1e-38, False))


def leg_precondition(payload: Dict[str, Any], out: str) -> None:
    """The census that BOUNDS the claim, and its own non-vacuity proof.

    Two halves, and the second is the one that matters:

    1. on the physical band the census over every OPERAND and every RESULT is
       zero and the bytes are identical;
    2. on the 1e-38 control the census FIRES and the gate REFUSES the case.

    A precondition never demonstrated to fire is decorative. The census covers
    RESULTS as well as operands because the curl PRODUCES subnormals it was not
    given — measured, the 1e-38 output census (6432) is nearly double its input
    census (3233).

    THE WINDOW IS REPORTED, NOT ONLY THE COUNT, and the window here is ONE STEP.
    Band entry is a RUN-and-WINDOW fact rather than a family fact: measured on the
    chi3 round, a Q~20 narrow-band Gaussian turn-on sweeps ~40 decades and drags
    the leading edge through the whole subnormal band — FIRST subnormal at step 55,
    LAST at step 3,726, then clean for 16,274 more steps, while a chi3=0 control
    with no nonlinear path at all reached the band in SIX field arrays. So a census
    that fired zero licenses NOTHING about a run with a narrow-band source, and a
    row that records only a count cannot be read for what it covers. This leg
    steps each case ONCE, so its window is ``[0, 0]`` and it is recorded as such
    rather than left to be inferred — an honest small window beats an unbounded
    claim. A multi-step window belongs to ``gate_metal_whole_step``.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for label, scale, expect_clean in SUBNORMAL_SCALES:
        for sub_step, spec in SUB_STEPS.items():
            grid, fields, pml = build(CONFIGS[1][1], CONFIGS[1][2], 0.35, 20260814,
                                      scale=scale, signed_zeros=False)
            before = snapshot(fields)
            getattr(stepping, sub_step)(fields, pml)
            after = snapshot(fields)
            codes = boundary_codes(grid, pml)
            got = run_curl_on_device(before, grid, pml, sub_step, codes,
                                     float(grid.dt / grid.dx))
            compared = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
            operand_names = tuple(spec["sources"]) + compared
            operand_census = sum(subnormal.census(before[n]) for n in operand_names)
            result_census = sum(subnormal.census(after[n]) for n in compared)
            bad = {n: differing(got[n], after[n]) for n in compared}
            total_bad = sum(bad.values())
            census_fired = (operand_census + result_census) > 0
            # THE WINDOW, not a scalar. This leg steps once, so it is [0, 0] and
            # the row says so; `first_fired`/`last_fired` are the step indices the
            # census actually fired on, which is what a reader needs to know how
            # far a "clean" verdict reaches. See this function's docstring for the
            # measurement that makes the distinction load-bearing.
            fired_steps = [0] if census_fired else []
            row = {"scale": label, "factor": scale, "sub_step": sub_step,
                   "window": [0, 0],
                   "steps_censused": [0],
                   "first_fired": fired_steps[0] if fired_steps else None,
                   "last_fired": fired_steps[-1] if fired_steps else None,
                   "operand_subnormal_words": operand_census,
                   "result_subnormal_words": result_census,
                   "census_fired": census_fired,
                   "gate_verdict": "REFUSED (precondition)" if census_fired
                                   else ("identical" if not total_bad else "DIFFERS"),
                   "differing_words": total_bad}
            rows.append(row)
            log(f"[leg6 precondition] {label:<14} {sub_step} window=[0,0] "
                f"first_fired={row['first_fired']} last_fired={row['last_fired']} "
                f"operands={operand_census} results={result_census} "
                f"census_fired={census_fired} differing={total_bad} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["precondition"] = rows
            save(payload, out)
            if expect_clean:
                assert not census_fired, (
                    f"{label}: the precondition fired on a band it must not — "
                    f"operands={operand_census} results={result_census}")
                assert total_bad == 0, row
            else:
                assert census_fired, (
                    f"{label}: the census DID NOT FIRE on the scaled control. A "
                    f"precondition never demonstrated to fire is decorative, and "
                    f"this leg exists to demonstrate it")
                assert total_bad > 0, (
                    f"{label}: the control was expected to diverge (measured 6480 "
                    f"words); it did not, so the cliff this claim rests on was not "
                    f"reproduced")


# ---------------------------------------------------------------------------
# LEG 7 — mutations: armed, launch-counted, three-valued
# ---------------------------------------------------------------------------

def _needle(source: str, old: str, new: str) -> str:
    if old not in source:
        raise LookupError(old)
    return source.replace(old, new)


def _flag_line(source: str, flag: str) -> str:
    """The one ``const bool <flag>0 = ..., <flag>1 = ..., <flag>2 = ...;`` line."""
    lines = [line for line in source.splitlines()
             if line.strip().startswith(f"const bool {flag}0 = ")]
    if len(lines) != 1:
        raise LookupError(f"the {flag} flag line ({len(lines)} found)")
    return lines[0]


def _set_flags(source: str, flag: str, values: Sequence[str]) -> str:
    """Replace the three membership flags of one body with three C expressions."""
    return source.replace(_flag_line(source, flag),
                          f"    const bool {flag}0 = {values[0]}, {flag}1 = {values[1]}, "
                          f"{flag}2 = {values[2]};")


def _lattice_tag(source: str, flag: str) -> str:
    """``h`` or ``i``: the sub-lattice tag of the membership the flags read."""
    return _flag_line(source, flag).split(" = in_", 1)[1][0]


def _dsig_flags(source: str) -> str:
    """The dsig/dsigu swap: each curl target keyed on its DSIG axis's membership."""
    tag = _lattice_tag(source, "du")
    return _set_flags(source, "du", [f"in_{tag}_{dsig}" for dsig, _ in CURL_CYCLE])


#: ``must_catch``: True (must be caught) | None (record only) | False (must NOT be
#: caught — a declared null, with its reason).
#:
#: THE GROUPING MUTATIONS ARE INVERTED (KERNELS.md §7.1). Before the alignment the
#: shipped groupings were the package's and these mutations planted MEEP's; the
#: shipped form now IS MEEP's, so each plants the PREVIOUS PACKAGE GROUPING and stays
#: must-catch. The ``b*`` rows are the per-chunk form battery (KERNELS.md §7.2).
CURL_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m1_previous_curl_grouping": {
        "must_catch": True,
        "why": "the package's previous pairwise grouping (g1s - g1) + (g2 - g2s) "
               "planted in place of MEEP's left-to-right ((g1s - g1) + g2) - g2s",
        "apply": lambda s: _needle(
            _needle(_needle(s, "dtdx * (((c_y - c) + b) - b_z)",
                            "dtdx * ((c_y - c) + (b - b_z))"),
                    "dtdx * (((a_z - a) + c) - c_x)", "dtdx * ((a_z - a) + (c - c_x))"),
            "dtdx * (((b_x - b) + a) - a_y)", "dtdx * ((b_x - b) + (a - a_y))"),
    },
    "m2_contract_on": {
        "must_catch": True,
        "why": "removes the one compile option; the recurrence's multiply-adds "
               "contract into fmas",
        "apply": lambda s: _needle(s, shaders.contraction_pragma("off"),
                                   shaders.contraction_pragma("fast")),
    },
    "m3_recurrence_axis_swap": {
        "must_catch": True,
        "why": "breaks vec.hpp's dsig/dsigu cycle on target 0: (y,z) -> (z,y)",
        "apply": lambda s: _needle(
            _needle(s, "float n0 = ((p0 * km_y) - curl0) * si_y;",
                    "float n0 = ((p0 * km_z) - curl0) * si_z;"),
            "float v0 = du0 ? ((((f0[ii] * km_z) + n0) - p0) * si_z) : n0;",
            "float v0 = du0 ? ((((f0[ii] * km_y) + n0) - p0) * si_y) : n0;"),
    },
    "m4_store_before_load": {
        "must_catch": True,
        "why": "the auxiliary's previous value read after its own store",
        "apply": lambda s: _needle(s, "float p0 = du0 ? u0[ii] : f0[ii];",
                                   "if (du0) { u0[ii] = 0.0f; } "
                                   "float p0 = du0 ? u0[ii] : f0[ii];"),
    },
    "m7_drop_ownership_mask_one_axis": {
        "must_catch": True,
        "why": "one metallic axis stops dropping its unowned cell — a plane of "
               "values MEEP's loop never touches",
        "apply": lambda s: _needle(s, "curl0 = at_x ? 0.0f : curl0;",
                                   "// ownership mask dropped on x"),
        "codes": (METALLIC, METALLIC, PERIODIC),
        "sub_steps": ("step_B",),
    },
    "m8_ghost_periodic_for_metallic": {
        "must_catch": True,
        "why": "a metallic axis wrapped like a periodic one: the wall reads the "
               "far face instead of an exact zero",
        "apply": lambda s: _needle(s, "vx = (si >= 0) && (si < nxi);",
                                   "si = (si == nxi) ? 0 : si;"),
        "codes": (METALLIC, METALLIC, PERIODIC),
        "sub_steps": ("step_B",),
    },
    "b1_select_removed": {
        "must_catch": True,
        "why": "the whole-grid recurrence (the previous package form): every cell "
               "steps the dsigu form and fu is written everywhere",
        "apply": lambda s: _set_flags(s, "du", ("true", "true", "true")),
    },
    "b2_predicate_on_coefficient_identity": {
        "must_catch": True,
        "why": "the form keyed on kms != 1 instead of chunk membership: a sigma = 0 "
               "row inside a layer chunk takes the other form",
        "apply": lambda s: _set_flags(s, "du", ("(kmz[k] != 1.0f)", "(kmx[i] != 1.0f)",
                                                "(kmy[j] != 1.0f)")),
    },
    "b5_dsig_dsigu_swapped": {
        "must_catch": True,
        "why": "each target's form keyed on its dsig axis's chunk instead of its "
               "dsigu axis's: dsig-only cells and dsigu-only cells trade forms",
        "apply": _dsig_flags,
    },
    "b6_auxiliary_stored_on_a_plain_cell": {
        "must_catch": True,
        "why": "fu written outside the recurrence's cells; the array path leaves "
               "those words alone and the gate compares them",
        "apply": lambda s: _needle(s, "if (du0) { u0[ii] = n0; }", "u0[ii] = n0;"),
    },
    "b7_auxiliary_load_unconditional": {
        "must_catch": False,
        "why": "DECLARED NULL: an fu load outside the guard with the store still "
               "guarded changes no word (the plain form reads f); the guard is "
               "traffic, pinned by test_metal_kernels' static needle instead",
        "apply": lambda s: _needle(s, "float p0 = du0 ? u0[ii] : f0[ii];",
                                   "float p0u = u0[ii]; "
                                   "float p0 = du0 ? p0u : f0[ii];"),
    },
    "b8_plain_curl_through_the_recurrence": {
        "must_catch": True,
        "why": "the plain form spelled as the recurrence with p := 0. With unit "
               "coefficients it differs only for f = -0, curl = +0 (leg 10 plants "
               "that cell); this leg catches it on the dsig-only cells whose dsig "
               "coefficients are not 1",
        "apply": lambda s: _needle(
            _needle(s, "float p0 = du0 ? u0[ii] : f0[ii];",
                    "float p0 = du0 ? u0[ii] : 0.0f;"),
            "float v0 = du0 ? ((((f0[ii] * km_z) + n0) - p0) * si_z) : n0;",
            "float v0 = (((f0[ii] * km_z) + n0) - p0) * si_z;"),
    },
}

CONSTITUTIVE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "m11_previous_accumulation_grouping": {
        "must_catch": True,
        "why": "the package's previous grouping ((f + kps*src) - kms*prev) planted "
               "in place of MEEP's f + ((kps*src) - (kms*prev))",
        "apply": lambda s: _needle(
            s, "if (dw0) { a0 = f0[ii] + ((kp_0 * src0) - (km_0 * prev0)); }",
            "if (dw0) { a0 = (f0[ii] + (kp_0 * src0)) - (km_0 * prev0); }"),
    },
    "m12_constitutive_store_before_load": {
        "must_catch": True,
        "why": "`prev` read after `fw` is written — wrong only where kms != 0, "
               "i.e. INSIDE THE PML ONLY, so it looks like a worse absorber",
        "apply": lambda s: _needle(
            s, "    float prev0 = dw0 ? w0[ii] : 0.0f;\n    float src0 =",
            "    if (dw0) { w0[ii] = 0.0f; }\n    float prev0 = dw0 ? w0[ii] : 0.0f;"
            "\n    float src0 ="),
    },
    "m13_dsigw_to_dsig_cycle": {
        "must_catch": True,
        "why": "the coefficient index moved off the component's OWN axis onto the "
               "curl's cycle — MEEP's dsigw is not dsig",
        "apply": lambda s: _needle(s, "float kp_0 = kp0[i], km_0 = km0[i];",
                                   "float kp_0 = kp0[j], km_0 = km0[j];"),
    },
    "b9_constitutive_select_removed": {
        "must_catch": True,
        "why": "the whole-grid accumulation (the previous package form): every cell "
               "accumulates and f_w is written everywhere",
        "apply": lambda s: _set_flags(s, "dw", ("true", "true", "true")),
    },
    "b10_constitutive_predicate_on_coefficient_identity": {
        "must_catch": True,
        "why": "the form keyed on kps != 1 instead of chunk membership",
        "apply": lambda s: _set_flags(s, "dw", ("(kp0[i] != 1.0f)", "(kp1[j] != 1.0f)",
                                                "(kp2[k] != 1.0f)")),
    },
    "b11_f_w_stored_on_a_plain_cell": {
        "must_catch": True,
        "why": "f_w written outside the accumulation's cells",
        "apply": lambda s: _needle(s, "if (dw0) { w0[ii] = src0; }", "w0[ii] = src0;"),
    },
    "b12_constitutive_load_unconditional": {
        "must_catch": False,
        "why": "DECLARED NULL: prev is read only on the accumulation's cells, so an "
               "unguarded load changes no word; pinned statically in "
               "test_metal_kernels instead",
        "apply": lambda s: _needle(s, "float prev0 = dw0 ? w0[ii] : 0.0f;",
                                   "float prev0 = w0[ii];"),
    },
    "b13_plain_constitutive_through_the_accumulation": {
        "must_catch": True,
        "why": "f = src spelled as the accumulation with kps = 1, kms = 0 on zeroed "
               "inputs: wrong on every -0 source (the seeded +-0 lattice reaches it; "
               "leg 10 plants one)",
        "apply": lambda s: _needle(s, "float a0 = src0;",
                                   "float a0 = 0.0f + ((1.0f * src0) - (0.0f * prev0));"),
    },
}


#: The record mutations (KERNELS.md §7.2): the kernel is right and the record it is
#: handed is not. Each is a function of the true record.
def _bound_off_by_one(record: Any) -> Any:
    """Every non-empty low run one row longer: the boundary plane changes form."""
    out = np.array(record, copy=True)
    for half in (False, True):
        for axis in range(3):
            a0, a1, b0 = (int(out[record_index(half, axis, w)]) for w in range(3))
            if a1 > a0 and (b0 == 0 or a1 + 1 < b0):
                out[record_index(half, axis, 1)] = a1 + 1
    return out


def _lattices_swapped(record: Any) -> Any:
    """The integer and half-integer entries exchanged."""
    out = np.array(record, copy=True)
    for axis in range(3):
        for word in range(4):
            i, h = record_index(False, axis, word), record_index(True, axis, word)
            out[i], out[h] = record[h], record[i]
    return out


RECORD_MUTATIONS: Dict[str, Tuple[Callable[[Any], Any], str]] = {
    "b3_bound_off_by_one": (_bound_off_by_one, "a run end one row past the chunk"),
    "b4_lattices_swapped": (_lattices_swapped,
                            "the integer and half-integer records swapped"),
}


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Every mutation ARMED, LAUNCH-COUNTED and classified three ways.

    * ``NEEDLE-MISSED`` — the transform matched nothing, so nothing was tested;
    * ``DISARMED`` — the mutant compiled but never launched. FAILS: a leg that
      certifies a defect it never ran is a hollow pass;
    * otherwise the caught/ran count, checked against ``must_catch``.

    The HOST mutations at the end have no source transform at all: they are the
    plan's own choices — the Yee sub-lattice suffix, the stencil direction — and
    the one with no Triton analogue, a device mirror left STALE across a launch.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    name, cell, boundaries = CONFIGS[1]  # metallic_xy: masks and walls both live
    grid, fields, pml = build(cell, boundaries, 0.35, 20260814)
    codes = boundary_codes(grid, pml)
    dtdx = float(grid.dt / grid.dx)
    base = snapshot(fields)
    inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}

    oracle: Dict[str, Dict[str, Any]] = {}
    for sub_step in SUB_STEPS:
        restore(fields, base)
        getattr(stepping, sub_step)(fields, pml)
        oracle[sub_step] = snapshot(fields)
    for side, spec in SIDES.items():
        restore(fields, base)
        getattr(stepping, spec["step"])(fields, pml)
        oracle["update_" + side] = snapshot(fields)

    def record(label: str, verdict: str, launches: int, caught: int, ran: int,
               must_catch: Optional[bool], why: str) -> None:
        row = {"mutation": label, "verdict": verdict, "launches": launches,
               "caught": caught, "ran": ran, "must_catch": must_catch, "why": why}
        rows.append(row)
        log(f"[leg7 mutation] {label:<34} launches={launches} {verdict} "
            f"({time.time() - started:.1f}s)")
        payload["legs"]["mutations"] = rows
        save(payload, out)
        assert verdict != "DISARMED", (
            f"{label} never launched: a mutation leg that certifies a defect it "
            f"did not run is a hollow pass")
        assert verdict != "NEEDLE-MISSED", (
            f"{label}: the transform matched nothing in the shipped source, so the "
            f"defect it names was never planted")
        if must_catch is True:
            assert caught == ran and ran > 0, f"{label} was not caught: {row}"
        elif must_catch is False:
            assert caught == 0, f"{label} was caught but must not be: {row}"

    for label, entry in CURL_MUTATIONS.items():
        mutation_codes = entry.get("codes", codes)
        counters: List[Counter] = []
        caught = ran = 0
        missed = False
        for sub_step in entry.get("sub_steps", tuple(SUB_STEPS)):
            spec = SUB_STEPS[sub_step]
            shipped = shaders.curl_source(mutation_codes, spec["backward"])
            try:
                mutated = entry["apply"](shipped)
            except LookupError:
                missed = True
                continue
            got = run_curl_on_device(base, grid, pml, sub_step, mutation_codes, dtdx,
                                     source=mutated, counter=counters)
            compared = tuple(spec["targets"]) + tuple(
                "fu_" + n for n in spec["targets"])
            ran += 1
            caught += int(any(differing(got[n], oracle[sub_step][n])
                              for n in compared))
        launches = sum(counter.launches for counter in counters)
        verdict = ("NEEDLE-MISSED" if missed and ran == 0
                   else "DISARMED" if launches == 0 else f"CAUGHT {caught}/{ran}")
        record(label, verdict, launches, caught, ran, entry["must_catch"],
               entry["why"])

    for label, entry in CONSTITUTIVE_MUTATIONS.items():
        counters = []
        caught = ran = 0
        missed = False
        for side in SIDES:
            shipped = shaders.constitutive_source(side)
            try:
                mutated = entry["apply"](shipped)
            except LookupError:
                missed = True
                continue
            got = run_constitutive_on_device(base, pml, side, inverse,
                                             source=mutated, counter=counters)
            spec = SIDES[side]
            compared = tuple(spec["targets"]) + tuple(spec["aux"])
            ran += 1
            caught += int(any(differing(got[n], oracle["update_" + side][n])
                              for n in compared))
        launches = sum(counter.launches for counter in counters)
        verdict = ("NEEDLE-MISSED" if missed and ran == 0
                   else "DISARMED" if launches == 0 else f"CAUGHT {caught}/{ran}")
        record(label, verdict, launches, caught, ran, entry["must_catch"],
               entry["why"])

    # --- HOST mutations: the plan's own choices, not the kernel's source -----
    # m5: the Yee sub-lattice suffix swapped. A half-cell error in the absorber
    # profile — converged, smooth, and wrong.
    caught = 0
    for sub_step, spec in SUB_STEPS.items():
        wrong_suffix = "" if spec["suffix"] == "_h" else "_h"
        flat = {k: np.asarray(v).reshape(-1)
                for k, v in curl_coefficients(pml, wrong_suffix).items()}
        arrays = {n: np.array(base[n], copy=True)
                  for n in tuple(spec["targets"]) + tuple(spec["sources"])}
        arrays.update({"fu_" + n: np.array(base["fu_" + n], copy=True)
                       for n in spec["targets"]})
        residency = metal_launch.Residency()
        function = Counter(metal_launch.compile_curl(codes, spec["backward"]))
        plan = metal_launch.plan_from_arrays(
            sub_step, arrays, flat, codes, dtdx, residency,
            functions={shaders.CONTRACT_OFF: function},
            chunk_bounds=layer_record(grid, pml))
        plan.run()
        residency.sync_out()
        compared = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
        caught += int(any(differing(arrays[n], oracle[sub_step][n])
                          for n in compared))
        assert function.launches == 1
    record("m5_half_integer_suffix_swap", f"CAUGHT {caught}/2", 2, caught, 2, True,
           "the curl's PML coefficients taken from the other Yee sub-lattice")

    # m6: the constitutive side's sub-lattice swapped, the same defect on the
    # other family — chosen by the HOST at stepping.py:948 vs :1015.
    caught = 0
    for side, spec in SIDES.items():
        wrong_suffix = "" if spec["suffix"] == "_h" else "_h"
        got = run_constitutive_on_device(base, pml, side, inverse,
                                         suffix=wrong_suffix)
        compared = tuple(spec["targets"]) + tuple(spec["aux"])
        caught += int(any(differing(got[n], oracle["update_" + side][n])
                          for n in compared))
    record("m6_constitutive_suffix_swap", f"CAUGHT {caught}/2", 2, caught, 2, True,
           "the constitutive coefficients taken from the other Yee sub-lattice")

    # m9: the D stencil direction flipped — step_D stepped with forward
    # differences. MEEP's negated strides are what make it the adjoint curl.
    counters = []
    spec = SUB_STEPS["step_D"]
    forward_source = shaders.curl_source(codes, False)
    got = run_curl_on_device(base, grid, pml, "step_D", codes, dtdx,
                             source=forward_source, counter=counters)
    compared = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
    caught = int(any(differing(got[n], oracle["step_D"][n]) for n in compared))
    record("m9_d_stencil_direction_flipped",
           "DISARMED" if not counters or counters[0].launches == 0
           else f"CAUGHT {caught}/1",
           sum(c.launches for c in counters), caught, 1, True,
           "step_D stepped with the forward stencil")

    # m10: A RESIDENCY MUTATION WITH NO TRITON ANALOGUE. An array-path seam writes
    # the HOST array between two device sub-steps and nothing syncs it in. The
    # defect is not that the device reads garbage — it is that the host write is
    # SILENTLY DISCARDED, and the second launch consumes the pre-seam value. So
    # the oracle here is the array path WITH the seam write applied, which is the
    # answer the driver would have produced, and the mutant must diverge from it.
    # If it did not, the mirror would not be load-bearing and the residency clause
    # would be certifying nothing.
    restore(fields, base)
    stepping.step_B(fields, pml)
    for component in ("Bx", "By", "Bz"):
        getattr(fields, component)[...] = np.float32(0.0)
    stepping.update_H(fields, pml)
    two_step_oracle = snapshot(fields)
    restore(fields, base)
    residency = metal_launch.Residency()
    plan = metal_launch.plan_step(fields, pml, residency=residency, sources=())
    plan.plans["step_B"].run()
    # The stale write: something on the array path touches the mirrored volume.
    for component in ("Bx", "By", "Bz"):
        getattr(fields, component)[...] = np.float32(0.0)
    plan.plans["update_H"].run()
    residency.sync_out()
    stale = snapshot(fields)
    compared = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
    caught = int(any(differing(stale[n], two_step_oracle[n]) for n in compared))
    record("m10_stale_mirror_across_a_launch", f"CAUGHT {caught}/1", 2, caught, 1,
           True,
           "a host write between two device sub-steps that no sync carried; the "
           "residency clause refuses this composition and this measures that the "
           "refusal is load-bearing")

    # b3/b4: THE RECORD IS WRONG, the kernel is the shipped one. Both sub-step
    # families, against the oracle stepped on the true division.
    truth = layer_record(grid, pml)
    for label, (mutate, why) in RECORD_MUTATIONS.items():
        wrong = mutate(truth)
        assert not np.array_equal(wrong, truth), f"{label} changed no record word"
        caught = ran = launches = 0
        for sub_step, spec in SUB_STEPS.items():
            counters: List[Counter] = []
            got = run_curl_on_device(base, grid, pml, sub_step, codes, dtdx,
                                     source=shaders.curl_source(codes, spec["backward"]),
                                     counter=counters, record=wrong)
            compared = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
            ran += 1
            launches += sum(c.launches for c in counters)
            caught += int(any(differing(got[n], oracle[sub_step][n]) for n in compared))
        for side, spec in SIDES.items():
            counters = []
            got = run_constitutive_on_device(base, pml, side, inverse,
                                             source=shaders.constitutive_source(side),
                                             counter=counters, record=wrong)
            compared = tuple(spec["targets"]) + tuple(spec["aux"])
            ran += 1
            launches += sum(c.launches for c in counters)
            caught += int(any(differing(got[n], oracle["update_" + side][n])
                              for n in compared))
        record(label, "DISARMED" if launches == 0 else f"CAUGHT {caught}/{ran}",
               launches, caught, ran, True, why)


# ---------------------------------------------------------------------------
# LEG 8 — the engine route and the disjointness sweep
# ---------------------------------------------------------------------------

def leg_engine(payload: Dict[str, Any], out: str) -> None:
    """Plans built from the engine's own objects, and no two arms co-admitting.

    The engine route and the ``from_arrays`` route produce the same object and go
    through the same ``run``, which is what makes the gate's bytes the engine's
    bytes. The sweep then asks every arm of every slot for its verdict on every
    configuration in the matrix and asserts at most one admits.
    """
    from meep_gpu.triton_kernels.no_pml_constitutive import (
        null_constitutive_coverage,
    )

    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        for absorbing in (True, False):
            grid, fields, pml = build(cell, boundaries, 0.35, 20260814)
            if not absorbing:
                pml = PML(grid=grid, thickness=tuple((0, 0) for _ in range(3)))
            residency = metal_launch.Residency()
            plan = metal_launch.plan_step(fields, pml, residency=residency,
                                          sources=())
            admitters: Dict[str, List[str]] = {}
            for slot, side in (("update_H", "H"), ("update_E", "E")):
                admitting = []
                if metal_coverage.constitutive_coverage(fields, pml, side,
                                                        residency).covered:
                    admitting.append("ordinary")
                if null_constitutive_coverage(fields, pml, side).covered:
                    admitting.append("no-PML null")
                admitters[slot] = admitting
            for slot in ("step_B", "step_D"):
                admitters[slot] = (["PML"] if metal_coverage.pml_curl_coverage(
                    fields, pml, slot, residency).covered else [])
            row = {"case": name, "absorbing": absorbing,
                   "replaces": list(plan.replaces),
                   "selected": dict(plan.selected),
                   "admitters": admitters,
                   "residency_covered": bool(plan.residency.covered)}
            rows.append(row)
            log(f"[leg8 engine] {name:<13} absorbing={absorbing} "
                f"replaces={plan.replaces} selected={plan.selected} "
                f"({time.time() - started:.1f}s)")
            payload["legs"]["engine"] = rows
            save(payload, out)
            for slot, admitting in admitters.items():
                assert len(admitting) <= 1, (
                    f"{name}/absorbing={absorbing}: {slot} is co-admitted by "
                    f"{admitting}; two products on one slot is the over-covering "
                    f"dispatch the composer fails closed on")
            if absorbing:
                assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
                assert set(plan.selected.values()) == {"PML", "ordinary"}
            else:
                # PREDICTED NULL, recorded with the reason that was MEASURED and
                # not the one that was expected. The curl kernels refuse because
                # they implement the split-field path only. The null family ALSO
                # refuses here, and its reason is its own conservative clause 4:
                # this ``Fields`` had ``enable_pml_storage()`` called on it, so
                # ``get_H`` would serve a stored H that ``update_H`` never writes.
                # That is a refusal by name, not an omission — and it is why the
                # null arm gets its own case below rather than being assumed live.
                row["predicted_null_reason"] = (
                    "no active PML: the curl kernels implement the split-field "
                    "path only; and the null family refuses this Fields because "
                    "PML STORAGE is switched on behind an inert layer "
                    "(no_pml_constitutive conservative clause 4)")
                assert "step_B" not in plan.replaces
                assert not plan.replaces, plan.replaces

    # THE NULL ARM, MADE NON-VACUOUS. Without this case the arm is consulted on
    # every configuration above and admits on none of them, which certifies
    # nothing about it. Here the layer is inert AND no PML storage was switched
    # on, which is the configuration stepping.py:944-945 / :983-984 return from.
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries="periodic",
                dimensions=3, courant=0.35, k_point=(0.0, 0.0, 0.0), xp=np)
    plain_fields = Fields(grid=grid, force_complex_fields=False)
    inert = PML(grid=grid, thickness=tuple((0, 0) for _ in range(3)))
    residency = metal_launch.Residency()
    plan = metal_launch.plan_step(plain_fields, inert, residency=residency,
                                  sources=())
    row = {"case": "no_pml_no_storage", "absorbing": False,
           "replaces": list(plan.replaces), "selected": dict(plan.selected),
           "admitters": {slot: [plan.selected[slot]] if slot in plan.selected else []
                         for slot in ("step_B", "update_H", "step_D", "update_E")},
           "residency_covered": bool(plan.residency.covered),
           "note": ("the null family's product launches nothing: update_H returns "
                    "at stepping.py:944-945 and update_E at :983-984")}
    rows.append(row)
    log(f"[leg8 engine] no_pml_no_storage replaces={plan.replaces} "
        f"selected={plan.selected} ({time.time() - started:.1f}s)")
    payload["legs"]["engine"] = rows
    save(payload, out)
    assert plan.selected.get("update_H") == "no-PML null", plan.reasons
    assert plan.selected.get("update_E") == "no-PML null", plan.reasons
    assert plan.selected.get("step_B") == "no-PML curl", plan.reasons
    assert plan.selected.get("step_D") == "no-PML curl", plan.reasons
    assert plan.replaces == ("step_B", "update_H", "step_D", "update_E")
    for slot in ("update_H", "update_E"):
        plan.plans[slot].run()
    assert all(plan.plans[slot].runs == 1 for slot in ("update_H", "update_E")), (
        "the null plan's run counter did not move: a leg that certifies "
        "'the plan reproduced the array path' by comparing bytes is trivially "
        "satisfied by a plan that was never invoked")


# ---------------------------------------------------------------------------
# LEG 9 — the form-coverage census (DESIGN.md D13)
# ---------------------------------------------------------------------------

def _form_counts(record: Any, shape: Sequence[int]) -> Dict[str, Dict[str, int]]:
    """Cells per (component, form) under one record, on each component's own line."""
    counts: Dict[str, Dict[str, int]] = {}
    total = int(np.prod(shape))
    for backward in (False, True):
        names = SUB_STEPS["step_D" if backward else "step_B"]["targets"]
        for target, (_dsig, dsigu) in enumerate(CURL_CYCLE):
            axis = "xyz".index(dsigu)
            half = IYEE_SHIFTS[names[target]][axis] == 1
            line = membership_from_record(record, half, axis, int(shape[axis]))
            inside = int(line.sum()) * total // int(shape[axis])
            counts[names[target]] = {"recurrence": inside, "dsig_only": total - inside}
    for side, spec in SIDES.items():
        for target, name in enumerate(spec["targets"]):
            half = IYEE_SHIFTS[name][target] == 1
            line = membership_from_record(record, half, target, int(shape[target]))
            inside = int(line.sum()) * total // int(shape[target])
            counts[name] = {"accumulate": inside, "plain": total - inside}
    return counts


def leg_forms(payload: Dict[str, Any], out: str) -> None:
    """Every gate case's FORM COVERAGE, recorded and floored.

    The rule (DESIGN.md D13): a nonzero count for every (component x form) class,
    and in-run and out-of-run rows on each axis and on both Yee sub-lattices, with
    at least one wrapping axis among the cases. A gate whose cases never stepped one
    form certifies nothing about it — the per-chunk forms are exactly the place a
    case drawn wholly inside or wholly outside a layer would hide a defect.
    """
    rows: List[Dict[str, Any]] = []
    classes: Dict[Tuple[str, str], int] = {}
    lines: Dict[Tuple[str, bool], Dict[str, int]] = {}
    wrapping = 0
    started = time.time()
    for name, cell, boundaries in CONFIGS:
        grid, _fields, pml = build(cell, boundaries, 0.35, 20260814)
        record = layer_record(grid, pml)
        counts = _form_counts(record, grid.shape)
        for component, forms in counts.items():
            for form, count in forms.items():
                classes[(component, form)] = classes.get((component, form), 0) + count
        for axis in range(3):
            for half in (False, True):
                line = membership_from_record(record, half, axis, int(grid.shape[axis]))
                entry = lines.setdefault(("xyz"[axis], half), {"in": 0, "out": 0})
                entry["in"] += int(line.sum())
                entry["out"] += int((~line).sum())
            wrapping += int(bool(grid.axis_wraps(axis)))
        row = {"case": name, "shape": [int(v) for v in grid.shape],
               "record": [int(v) for v in record], "forms": counts}
        rows.append(row)
        log(f"[leg9 forms] {name:<13} shape={tuple(grid.shape)} "
            f"forms={{{', '.join(f'{c}: {f}' for c, f in sorted(counts.items()))}}} "
            f"({time.time() - started:.1f}s)")
    payload["legs"]["forms"] = {
        "cases": rows,
        "classes": {f"{c}/{f}": n for (c, f), n in sorted(classes.items())},
        "lines": {f"{a}/{'half' if h else 'integer'}": v for (a, h), v in sorted(lines.items())},
        "wrapping_axes": wrapping,
    }
    save(payload, out)
    empty = sorted(f"{c}/{f}" for (c, f), n in classes.items() if n == 0)
    assert not empty, f"VACUOUS form classes: {empty}"
    assert len(classes) == 24, sorted(classes)
    flat = sorted(f"{a}/{h}" for (a, h), v in lines.items() if not v["in"] or not v["out"])
    assert not flat, f"lines with no in-run or no out-of-run row: {flat}"
    assert wrapping > 0, "no case carries a wrapping axis"


# ---------------------------------------------------------------------------
# LEG 10 — the two signed-zero cases only a planted -0 reaches
# ---------------------------------------------------------------------------

NEGATIVE_ZERO = np.float32(-0.0)


def _word(array: Any, index: Tuple[int, int, int]) -> int:
    return int(np.asarray(array[index], dtype=np.float32).view(np.uint32))


def leg_planted_zero(payload: Dict[str, Any], out: str) -> None:
    """A plain cell whose f is ``-0`` and whose curl is exactly ``+0``; a plain cell
    whose source is ``-0``.

    MEEP's plain curl ``f - curl`` keeps ``-0 - +0 = -0`` and its plain constitutive
    ``f = src`` keeps a ``-0`` source. Spelled through the recurrence with
    ``p := 0`` (``b8``) or as the accumulation with ``kps = 1, kms = 0`` on zeroed
    inputs (``b13``), both give ``+0`` — on unit coefficients, the ONLY cells where
    the two spellings disagree, so random states do not reach them. The curl's
    ``+0`` is built, not hoped for: each stencil partner is set equal at the cell and
    at its neighbour, so ``((g1s - g1) + g2) - g2s = (+0 + g2) - g2 = +0``.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    name, cell, boundaries = CONFIGS[1]
    grid, fields, pml = build(cell, boundaries, 0.35, 20260814)
    record = layer_record(grid, pml)
    codes = boundary_codes(grid, pml)
    dtdx = float(grid.dt / grid.dx)
    inverse = {n: fields.inverse_epsilon_for(n) for n in SIDES["E"]["targets"]}
    seeded = snapshot(fields)
    centre = tuple(int(n) // 2 for n in grid.shape)
    i, j, k = centre

    def plain(component: str, axis: int) -> bool:
        half = IYEE_SHIFTS[component][axis] == 1
        line = membership_from_record(record, half, axis, int(grid.shape[axis]))
        return not bool(line[centre[axis]])

    for sub_step, spec in SUB_STEPS.items():
        target = spec["targets"][0]
        assert plain(target, 2) and plain(target, 1), (
            f"the centre is not a plain {target} cell, so this leg would plant "
            f"nothing it measures")
        state = {n: np.array(v, copy=True) for n, v in seeded.items()}
        step = -1 if spec["backward"] else 1
        state[target][centre] = NEGATIVE_ZERO
        first, second = spec["sources"][2], spec["sources"][1]
        state[first][i, j + step, k] = state[first][centre]
        state[second][i, j, k + step] = state[second][centre]
        restore(fields, state)
        getattr(stepping, sub_step)(fields, pml)
        oracle = snapshot(fields)
        shipped = run_curl_on_device(state, grid, pml, sub_step, codes, dtdx,
                                     record=record)
        counters: List[Counter] = []
        mutant_source = CURL_MUTATIONS["b8_plain_curl_through_the_recurrence"]["apply"](
            shaders.curl_source(codes, spec["backward"]))
        mutant = run_curl_on_device(state, grid, pml, sub_step, codes, dtdx,
                                    source=mutant_source, counter=counters,
                                    record=record)
        compared = tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
        row = {"case": name, "sub_step": sub_step, "cell": list(centre),
               "oracle_word": f"{_word(oracle[target], centre):#010x}",
               "shipped_differing": {n: differing(shipped[n], oracle[n])
                                     for n in compared if differing(shipped[n], oracle[n])},
               "mutant_word": f"{_word(mutant[target], centre):#010x}",
               "mutant_launches": sum(c.launches for c in counters)}
        rows.append(row)
        log(f"[leg10 planted_zero] {sub_step} {target}{centre} oracle={row['oracle_word']} "
            f"shipped={'IDENTICAL' if not row['shipped_differing'] else row['shipped_differing']} "
            f"b8={row['mutant_word']} ({time.time() - started:.1f}s)")
        payload["legs"]["planted_zero"] = rows
        save(payload, out)
        assert _word(oracle[target], centre) == 0x80000000, (
            f"VACUOUS: the planted cell's array-path word is {row['oracle_word']}, "
            f"not -0, so the case was not constructed")
        assert not row["shipped_differing"], row
        assert row["mutant_launches"] > 0, "b8 never launched"
        assert _word(mutant[target], centre) != 0x80000000, (
            f"b8 kept the planted -0 on {target}: the plain form's signed zero is not "
            f"what this leg measures")

    for side, spec in SIDES.items():
        target, source_name = spec["targets"][0], spec["sources"][0]
        assert plain(target, 0), f"the centre is not a plain {target} cell"
        state = {n: np.array(v, copy=True) for n, v in seeded.items()}
        state[source_name][centre] = NEGATIVE_ZERO
        restore(fields, state)
        getattr(stepping, spec["step"])(fields, pml)
        oracle = snapshot(fields)
        shipped = run_constitutive_on_device(state, pml, side, inverse, record=record)
        counters = []
        mutant_source = CONSTITUTIVE_MUTATIONS[
            "b13_plain_constitutive_through_the_accumulation"]["apply"](
            shaders.constitutive_source(side))
        mutant = run_constitutive_on_device(state, pml, side, inverse,
                                            source=mutant_source, counter=counters,
                                            record=record)
        compared = tuple(spec["targets"]) + tuple(spec["aux"])
        row = {"case": name, "side": side, "cell": list(centre),
               "oracle_word": f"{_word(oracle[target], centre):#010x}",
               "shipped_differing": {n: differing(shipped[n], oracle[n])
                                     for n in compared if differing(shipped[n], oracle[n])},
               "mutant_word": f"{_word(mutant[target], centre):#010x}",
               "mutant_launches": sum(c.launches for c in counters)}
        rows.append(row)
        log(f"[leg10 planted_zero] update_{side} {target}{centre} "
            f"oracle={row['oracle_word']} "
            f"shipped={'IDENTICAL' if not row['shipped_differing'] else row['shipped_differing']} "
            f"b13={row['mutant_word']} ({time.time() - started:.1f}s)")
        payload["legs"]["planted_zero"] = rows
        save(payload, out)
        assert _word(oracle[target], centre) == 0x80000000, (
            f"VACUOUS: {target}{centre} is {row['oracle_word']}, not -0")
        assert not row["shipped_differing"], row
        assert row["mutant_launches"] > 0, "b13 never launched"
        assert _word(mutant[target], centre) != 0x80000000, (
            f"b13 kept the planted -0 on {target}")


# ---------------------------------------------------------------------------
# LEG 12 — the run's FIRST update_H, through the composition
# ---------------------------------------------------------------------------

def leg_first_update_h(payload: Dict[str, Any], out: str) -> None:
    """MEEP's start of H on the run's first ``update_H``, carried by the composition.

    Every leg above seeds a mid-run state, which has started H already, so none of
    them reaches the start (``stepping.separate_h_from_b``). This one composes the
    certified singles ("PML" + "ordinary") through ``launch.plan_step`` and
    ``wrap_for_residency`` on a run frozen before its first step, on a layer whose
    negative ``kms`` lands the first B on -0, and requires 0 differing words against
    the array path -- and requires the same composition with the start stripped from
    the ``update_H`` slot to differ. ``metal_first_update_h`` holds the case and
    the score; ``gate_metal_fused_magnetic_pair`` runs the same leg on the pair.
    """
    import metal_first_update_h as first  # noqa: PLC0415 - sibling harness module

    row = first.run_leg((), "ordinary", log=log)
    payload["legs"]["first_update_h"] = row
    save(payload, out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

LEGS: Tuple[Tuple[str, Callable[[Dict[str, Any], str], None]], ...] = (
    ("transcription", leg_transcription),
    ("reference", leg_reference),
    ("synthetic", leg_synthetic),
    ("constitutive", leg_constitutive),
    ("multi_step", leg_multi_step),
    ("signed_zero", leg_signed_zero),
    ("precondition", leg_precondition),
    ("mutations", leg_mutations),
    ("engine", leg_engine),
    ("forms", leg_forms),
    ("planted_zero", leg_planted_zero),
    ("first_update_h", leg_first_update_h),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="artifact JSON path")
    parser.add_argument("--legs", default="", help="comma-separated leg subset")
    arguments = parser.parse_args()

    import torch  # noqa: PLC0415

    out_dir = os.path.dirname(os.path.abspath(arguments.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()
    payload: Dict[str, Any] = {
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mps_available": bool(torch.backends.mps.is_available()),
            "metal_frontend": metal_launch.metal_frontend_version(),
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        # THE CLAIM IS ONLY AS GOOD AS THE PRECONDITION IT WAS CERTIFIED UNDER.
        "subnormal_policy": subnormal.mps_policy_report(),
        "provenance": provenance(out_dir),
        "fingerprints": metal_launch.compute_fingerprints()["metal_kernels"],
        "legs": {},
    }
    save(payload, arguments.out)

    wanted = tuple(name.strip() for name in arguments.legs.split(",") if name.strip())
    log(f"[env] torch={torch.__version__} numpy={np.__version__} "
        f"frontend={payload['environment']['metal_frontend']} "
        f"policy={payload['subnormal_policy']['resolved']}")
    assert payload["subnormal_policy"]["admitted"], (
        f"the MPS executor refused the resolved subnormal policy: "
        f"{payload['subnormal_policy']['reasons']}")

    for name, leg in LEGS:
        if wanted and name not in wanted:
            log(f"[skip] leg {name} (not in --legs)")
            continue
        log(f"=== LEG {name} ===")
        leg(payload, arguments.out)

    payload["summary"] = {
        "status": "passed",
        "legs_run": [name for name, _ in LEGS if not wanted or name in wanted],
        "claim": ("byte-identity to stepping.py on this host, under a CHECKED "
                  "subnormal-free precondition (leg 6) — not a stated tolerance"),
        "scope": ("real-field PML curl (step_B/step_D) and constitutive "
                  "(update_H/update_E), both forms of each chosen per structure "
                  "chunk from the bound chunk record; periodic and "
                  "metallic ghost rules; no fold, no cylindrical axis, no Bloch "
                  "phase, no beta, no BFAST, no chi2/chi3, no dispersion, no "
                  "off-diagonal row, real float32 storage only"),
        "stated_weakness": ("no PTX-equivalent audit exists on this executor: "
                            "compile_shader exposes no disassembly, so the "
                            "mutation legs and this gate are the only arbiters"),
        "elapsed_s": round(time.time() - started, 1),
    }
    save(payload, arguments.out)
    log(f"METAL PML GATE PASSED in {payload['summary']['elapsed_s']}s -> "
        f"{arguments.out}")
    return 0


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
