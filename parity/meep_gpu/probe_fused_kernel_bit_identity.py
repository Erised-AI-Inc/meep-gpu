"""Hardware preconditions for a bit-identical fused CUDA curl fast path.

Phase 1 of ``the design notes (fdtd-fused-cuda-kernel-plan)`` gates the fused
kernels on **bit-identity** against the CuPy array path (plan section 3.2). Two
hardware-level facts have to be measured on the GPU before that gate can mean
anything, and neither can be settled by reading source:

1. **FMA contraction.** NVRTC contracts ``a*b + c`` into a single fused
   multiply-add by default (``--fmad=true``), which rounds once where the array
   path rounds twice. Is ``--fmad=false`` *necessary* (does the default output
   actually differ?) and is it *sufficient* (does it alone reach bit-identity?).
2. **Curl associativity.** ``stepping.py:1648`` contracts the stencil as
   ``dtdx * ((shifted_first - first) + (second - shifted_second))``. The fused
   kernels write ``dtdx * (a - b + c - d)``, which C associates left to right as
   ``((a - b) + c) - d``. Float addition is not associative, so these differ in
   the last bits even with FMA off.

This probe is deliberately **standalone**: it imports only the kernel-string
module ``meep_gpu/cuda_kernels/step_curl_kernels.py`` (loaded by path, never as
part of the ``meep_gpu`` package, so nothing pulls in ``driver``/``pml``), and
re-derives the array path's expression order here by transcription from
``stepping.py`` rather than importing it. Two independent transcriptions of the
same contract is the point: an import would compare the kernel against whatever
the array path happens to do, not against what the plan says it must do.

The reference is computed twice per case:

* ``array_order``  — ``dtdx * ((sf - f) + (s - ss))``, stepping.py's contract.
* ``kernel_order`` — ``dtdx * (((sf - f) + s) - ss)``, the kernels' C grouping,
  evaluated with the same CuPy array ops.

Comparing against both separates the two effects: if the kernel matches
``kernel_order`` bitwise but not ``array_order``, associativity is the only
remaining divergence and the fix is a source edit to the stencil parentheses.

Usage (the GPU host, one GPU)::

    CUDA_VISIBLE_DEVICES=0 python -u probe_fused_kernel_bit_identity.py \
        --kernels /path/to/cuda_kernels/step_curl_kernels.py \
        --out results/fused_kernel_bit_identity.json

Every case prints one flushed line as it lands and is appended to the JSON
artifact incrementally (the progress-reporting rule).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cupy as cp
except ImportError:  # No device here — the reference transcriptions still load.
    # Every experiment needs CuPy and main() refuses without it. The import is
    # optional only so ``reference_pml_step`` can be exercised on NumPy against
    # ``stepping.py`` on a host with no GPU, which is what pins the transcription
    # BEFORE the device run measures the kernel against it
    # (``validate_pml_reference_vs_stepping.py``).
    cp = None

# Fixed literal seed: the probe must be re-runnable to the same bytes.
SEED = 20260804

# Compile-option spellings under test. The empty tuple reproduces what
# ``_get_kernel`` does today (``cp.RawKernel(code, func_name)``, no options).
OPTION_SETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("default_no_options", ()),
    ("fmad_false_double_dash", ("--fmad=false",)),
    ("fmad_false_single_dash", ("-fmad=false",)),
    ("fmad_true_double_dash", ("--fmad=true",)),
)

# The Cartesian curl term tables, transcribed from stepping.py:212-220.
#   CurlTerm(target, first, first_axis, second, second_axis, dsig, dsigu)
# Axis codes: x = 0, y = 1, z = 2 (stepping.py AXIS_X/AXIS_Y/AXIS_Z).
B_TERMS: Tuple[Tuple[str, str, int, str, int], ...] = (
    ("Bx", "Ez", 1, "Ey", 2),   # curl_x = dEz/dy - dEy/dz
    ("By", "Ex", 2, "Ez", 0),   # curl_y = dEx/dz - dEz/dx
    ("Bz", "Ey", 0, "Ex", 1),   # curl_z = dEy/dx - dEx/dy
)
D_TERMS: Tuple[Tuple[str, str, int, str, int], ...] = (
    ("Dx", "Hz", 1, "Hy", 2),   # curl_x = dHz/dy - dHy/dz
    ("Dy", "Hx", 2, "Hz", 0),   # curl_y = dHx/dz - dHz/dx
    ("Dz", "Hy", 0, "Hx", 1),   # curl_z = dHy/dx - dHx/dy
)


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Loading the kernel-string module without importing the meep_gpu package
# ---------------------------------------------------------------------------

def load_kernel_module(path: str):
    """Import ``step_curl_kernels.py`` by path, outside any package.

    The module's only runtime imports are ``cupy``/``numpy``/``typing``; the
    ``from ..fields import`` lines sit under ``TYPE_CHECKING`` or inside the
    symmetry wrappers, which this probe never calls.
    """
    spec = importlib.util.spec_from_file_location("probe_step_curl_kernels", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load kernel module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CupyShim:
    """Stand-in for the ``cp`` name inside the kernel module.

    This substitutes the probe's option tuple for whatever ``_get_kernel`` passes,
    without editing the file, so the probe always compiles exactly the code map
    the file declares while sweeping the option spellings itself. The ``options``
    keyword is accepted and deliberately overridden — that is the point of the
    sweep, and it keeps the probe working whether or not the file pins options.
    """

    def __init__(self, options: Sequence[str]):
        self._options = tuple(options)

    def __getattr__(self, item: str) -> Any:
        return getattr(cp, item)

    def RawKernel(self, code: str, func_name: str, options=None):  # noqa: N802
        if self._options:
            return cp.RawKernel(code, func_name, options=self._options)
        return cp.RawKernel(code, func_name)


def kernel_inventory(module) -> List[Dict[str, Any]]:
    """Derive the registered kernel list from ``_get_kernel``'s own code map."""
    source = inspect.getsource(module._get_kernel)
    found = re.findall(
        r"\('(\w+)',\s*(True|False)\):\s*\((\w+),\s*'(\w+)'\)", source)
    return [
        {"name": name, "is_complex": flag == "True",
         "code_symbol": symbol, "func_name": func}
        for name, flag, symbol, func in found
    ]


# ---------------------------------------------------------------------------
# Bit comparison
# ---------------------------------------------------------------------------

def _ordered_key(raw: np.ndarray) -> np.ndarray:
    """Monotone integer key for IEEE-754 float32, so |ka - kb| is the ULP gap."""
    signed = raw.view(np.int32).astype(np.int64)
    return np.where(signed < 0, np.int64(-2147483648) - signed, signed)


def to_host(array: Any) -> np.ndarray:
    """Host copy of a device or host array, decided by the OBJECT not by ``cp``.

    ``isinstance(array, cp.ndarray)`` + ``cp.asnumpy`` was the spelling here, and
    it makes every caller depend on what ``import cupy`` happened to bind in this
    process. A module that stubs ``sys.modules["cupy"]`` with NumPy — parity
    scripts used to, to import a gate without a device — makes that ``isinstance``
    answer True for a plain NumPy array and sends it to a ``numpy.asnumpy`` that
    does not exist. Asking the array for its own ``.get`` needs no module identity
    and is correct on both backends: CuPy's ``ndarray.get()`` is what ``asnumpy``
    calls, and it is bit-preserving, which is the only property the uint32 word
    comparisons downstream depend on.
    """
    if isinstance(array, np.ndarray):
        return np.asarray(array)          # exactly the old host branch: a no-op
    getter = getattr(array, "get", None)
    return np.asarray(getter() if callable(getter) else array)


def bit_compare(a_dev: Any, b_dev: Any) -> Dict[str, Any]:
    """Exact-byte comparison of two arrays, with the ULP gap when unequal."""
    a = to_host(a_dev)
    b = to_host(b_dev)
    if a.dtype == np.complex64:
        af = np.ascontiguousarray(a).view(np.float32).ravel()
        bf = np.ascontiguousarray(b).view(np.float32).ravel()
    else:
        af = np.ascontiguousarray(a).ravel()
        bf = np.ascontiguousarray(b).ravel()
    ua = af.view(np.uint32)
    ub = bf.view(np.uint32)
    identical = bool(np.array_equal(ua, ub))
    differing = int(np.count_nonzero(ua != ub))
    result: Dict[str, Any] = {
        "bit_identical": identical,
        "differing_floats": differing,
        "total_floats": int(ua.size),
    }
    if not identical:
        ulp = np.abs(_ordered_key(af) - _ordered_key(bf))
        result["max_ulp"] = int(ulp.max())
        result["mean_ulp_over_differing"] = float(
            ulp[ulp > 0].mean()) if differing else 0.0
        result["max_abs_diff"] = float(
            np.max(np.abs(af.astype(np.float64) - bf.astype(np.float64))))
        denom = np.maximum(np.abs(af.astype(np.float64)), 1e-30)
        result["max_rel_diff"] = float(
            np.max(np.abs(af.astype(np.float64) - bf.astype(np.float64)) / denom))
    return result


def combine(parts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Roll three per-component comparisons into one verdict."""
    identical = all(p["bit_identical"] for p in parts.values())
    out: Dict[str, Any] = {
        "bit_identical": identical,
        "differing_floats": sum(p["differing_floats"] for p in parts.values()),
        "total_floats": sum(p["total_floats"] for p in parts.values()),
        "per_component": parts,
    }
    if not identical:
        out["max_ulp"] = max(p.get("max_ulp", 0) for p in parts.values())
        out["max_abs_diff"] = max(p.get("max_abs_diff", 0.0) for p in parts.values())
    return out


# ---------------------------------------------------------------------------
# The array-path reference, transcribed from stepping.py — ONE curl, two legs
# ---------------------------------------------------------------------------
#
# There used to be two curls under one name: an all-periodic copy for the
# no-absorber leg and the ghost-rule-carrying one below for the PML leg. They
# were the same four ufuncs on the same operands — with every axis PERIODIC
# ``reference_shift`` returns the bare roll and the METALLIC branch is
# unreachable — so the copy bought no second reading and cost the no-absorber leg
# its reference: both were spelled ``reference_curl``, the later definition
# rebound the name, and the earlier one's caller was left passing a signature
# that no longer existed. What survives is one curl, called by
# ``reference_pml_step`` under the grid's real boundaries and by
# ``reference_step`` under ``ALL_PERIODIC``.
#
# ---------------------------------------------------------------------------
# The real-field PML sub-step reference — a SECOND transcription, not an import
# ---------------------------------------------------------------------------
#
# The gate compares three things, not two: the kernel, this transcription, and
# (in validate_pml_reference_vs_stepping.py, on a NumPy host with no device)
# stepping.py itself. Two independent transcriptions agreeing with each other
# AND with the array path is what makes "bit-identical" mean the contract rather
# than mean "the kernel reproduces whatever I wrote twice".
#
# Term tables transcribed from stepping.B_CURL_TERMS / D_CURL_TERMS
# (stepping.py:214-223) with the Yee shifts from fields.IYEE_SHIFTS
# (fields.py:214-219). Axis codes x = 0, y = 1, z = 2.
#   (target, g1, axis(g1), g2, axis(g2), dsig, dsigu, iyee)
B_PML_TERMS = (
    ("Bx", "Ez", 1, "Ey", 2, "y", "z", (0, 1, 1)),
    ("By", "Ex", 2, "Ez", 0, "z", "x", (1, 0, 1)),
    ("Bz", "Ey", 0, "Ex", 1, "x", "y", (1, 1, 0)),
)
D_PML_TERMS = (
    ("Dx", "Hz", 1, "Hy", 2, "y", "z", (1, 0, 0)),
    ("Dy", "Hx", 2, "Hz", 0, "z", "x", (0, 1, 0)),
    ("Dz", "Hy", 0, "Hx", 1, "x", "y", (0, 0, 1)),
)

AXIS_NAMES = ("x", "y", "z")
PERIODIC = "periodic"
METALLIC = "metallic"
#: The no-absorber leg's grid: every axis wraps, so no ghost plane is ever
#: rewritten and no cell is un-owned. ``reference_step`` is the only caller.
ALL_PERIODIC: Tuple[str, str, str] = (PERIODIC, PERIODIC, PERIODIC)


def _face(axis: int, index: int) -> Tuple[Any, ...]:
    """The single-plane index tuple ``[..., index, ...]`` on one axis."""
    return tuple(index if a == axis else slice(None) for a in range(3))


def reference_shift(xp: Any, field: Any, axis: int, boundary: str,
                    backward: bool) -> Any:
    """``field[i-1]`` (backward) or ``field[i+1]`` (forward) under one axis's ghost rule.

    stepping._shift_up (:1723) / _shift_down (:1787), PERIODIC and METALLIC only.
    Both are a plain one-cell roll; METALLIC then overwrites the single wrapped
    plane with zero — the far plane going up, the near plane going down. No Bloch
    multiply: real storage refuses a phase, so ``phases`` is (None, None, None)
    by construction in this slice.

    An invariant axis (n == 1, always PERIODIC) needs no case of its own — the
    wrap returns the same plane and the difference is an exact zero.
    """
    shifted = xp.roll(field, 1 if backward else -1, axis=axis)
    if boundary == PERIODIC:
        return shifted
    if boundary == METALLIC:
        shifted[_face(axis, 0 if backward else -1)] = 0
        return shifted
    raise ValueError(f"boundary {boundary!r} has no real-field PML reference")


def reference_curl(xp: Any, first: Any, second: Any, first_axis: int,
                   second_axis: int, dtdx: Any, backward: bool,
                   boundaries: Sequence[str], grouping: str) -> Any:
    """dtdx * the discrete curl, in one of the two groupings.

    ``array_order`` is stepping._curl_from_operands (:1601), the contract::

        dtdx * ((shifted_first - first) + (second - shifted_second))

    ``kernel_order`` is what C makes of ``dtdx * (a - b + c - d)`` written
    without parentheses — ``((a - b) + c) - d``. Float addition is not
    associative, so keeping both separates an associativity regression from
    everything else the gate can catch.
    """
    shifted_first = reference_shift(xp, first, first_axis,
                                    boundaries[first_axis], backward)
    shifted_second = reference_shift(xp, second, second_axis,
                                     boundaries[second_axis], backward)
    if grouping == "array_order":
        return dtdx * ((shifted_first - first) + (second - shifted_second))
    if grouping == "kernel_order":
        return dtdx * (((shifted_first - first) + second) - shifted_second)
    raise ValueError(f"unknown grouping {grouping!r}")


def reference_mask(curl: Any, iyee: Sequence[int],
                   boundaries: Sequence[str]) -> None:
    """Zero the curl on every metallic wall the target does not own, in place.

    stepping._mask_non_owned_cells (:1865), the metallic branch. Cell 0 of a
    component whose Yee shift is 0 on a metallic axis sits ON the wall. The
    recurrence still runs on the zeroed curl — masking the curl rather than
    skipping the write is what makes the auxiliary decay instead of integrating
    a curl assembled from a ghost the cell does not own.
    """
    for axis in range(3):
        if iyee[axis] == 0 and boundaries[axis] == METALLIC:
            curl[_face(axis, 0)] = 0


def reference_recurrence(field: Any, fu: Any, curl: Any, kms: Any, sinv: Any,
                         kms_u: Any, sinv_u: Any) -> None:
    """The split-field PML curl recurrence, in place, in the array path's order.

    stepping._apply_pml_update (:1905), whose in-place sequence is exactly::

        fu *= kms; fu -= curl; fu *= sinv
        field *= kms_u; field += fu; field -= fu_previous; field *= sinv_u

    Both arrays are state. ``fu`` is why the gate compares two arrays per
    component and runs consecutive sub-steps: a kernel that gets ``field`` right
    and ``fu`` wrong is correct for exactly one step and wrong forever after.
    """
    fu_previous = fu.copy()
    fu *= kms
    fu -= curl
    fu *= sinv
    field *= kms_u
    field += fu
    field -= fu_previous
    field *= sinv_u


def reference_pml_step(xp: Any, sources: Dict[str, Any], targets: Dict[str, Any],
                       auxiliaries: Dict[str, Any], coefficients: Dict[str, Any],
                       dtdx: Any, sub_step: str, boundaries: Sequence[str],
                       grouping: str = "array_order") -> None:
    """One real-field PML curl sub-step, in place on ``targets`` and ``auxiliaries``.

    ``coefficients`` is keyed ``kms_x``/``sinv_x``/... in BROADCAST shape
    ((n,1,1), (1,n,1), (1,1,n)) — the layout ``PML._reshape_for_broadcast``
    produces, which is what the array path multiplies by.
    """
    terms = B_PML_TERMS if sub_step == "step_B" else D_PML_TERMS
    backward = sub_step == "step_D"
    for target, g1, a1, g2, a2, dsig, dsigu, iyee in terms:
        curl = reference_curl(xp, sources[g1], sources[g2], a1, a2, dtdx,
                              backward, boundaries, grouping)
        reference_mask(curl, iyee, boundaries)
        reference_recurrence(
            targets[target], auxiliaries["fu_" + target], curl,
            coefficients["kms_" + dsig], coefficients["sinv_" + dsig],
            coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu])


def reference_step(xp: Any, sources: Dict[str, Any], targets: Dict[str, Any],
                   terms: Sequence[Tuple[str, str, int, str, int]],
                   dtdx: Any, backward: bool, grouping: str) -> Dict[str, Any]:
    """``target -= curl`` for all three components, no absorber, all axes periodic.

    stepping._apply_curl's plain branch (:510) — the update a cell takes when it
    is in no PML and its material carries no conductivity, which is what the
    all-periodic fused core does.

    ``backward=True`` is MEEP's negated stride (the D sub-step, ``_shift_down``,
    ``roll(f, +1)[i] = f[i-1]``); ``False`` is the B sub-step (``_shift_up``,
    ``roll(f, -1)[i] = f[i+1]``).

    ``reference_mask`` is not called and must not be: under ``ALL_PERIODIC`` its
    only clause tests ``boundaries[axis] == METALLIC``, so it is a proven no-op
    here, and invoking it would put a metallic rule in a leg that has no wall.
    """
    out = {}
    for target, first, first_axis, second, second_axis in terms:
        curl = reference_curl(xp, sources[first], sources[second], first_axis,
                              second_axis, dtdx, backward, ALL_PERIODIC, grouping)
        updated = targets[target].copy()
        updated -= curl
        out[target] = updated
    return out


# ---------------------------------------------------------------------------
# Field shim for the module's launch wrappers
# ---------------------------------------------------------------------------

class FieldsShim:
    """Minimum surface ``_step_B_fused`` / ``_step_D_fused`` touch.

    The real ``Fields.get_E`` returns ``D * inv_eps``; handing the kernels raw E
    and H arrays isolates the curl, which is the only thing under test here.
    """

    def __init__(self, arrays: Dict[str, Any]):
        for key, value in arrays.items():
            setattr(self, key, value)

    def get_E(self, name: str) -> Any:
        return getattr(self, name)

    def get_H(self, name: str) -> Any:
        return getattr(self, name)


def make_fields(shape: Tuple[int, int, int], dtype: Any, rng) -> Dict[str, Any]:
    """Random, seeded, C-contiguous device arrays for all twelve volumes."""
    names = ("Bx", "By", "Bz", "Ex", "Ey", "Ez",
             "Dx", "Dy", "Dz", "Hx", "Hy", "Hz")
    arrays: Dict[str, Any] = {}
    for name in names:
        real = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        if dtype == np.complex64:
            imag = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            host = (real + 1j * imag).astype(np.complex64)
        else:
            host = real
        arrays[name] = cp.asarray(np.ascontiguousarray(host))
    return arrays


# ---------------------------------------------------------------------------
# PML coefficient tables — two sources, both required
# ---------------------------------------------------------------------------

def broadcast_shape(axis: int, n: int) -> Tuple[int, int, int]:
    """PML._reshape_for_broadcast's layout: (n,1,1) for x, (1,n,1) y, (1,1,n) z."""
    return tuple(n if a == axis else 1 for a in range(3))  # type: ignore[return-value]


def synthetic_coefficients(xp: Any, shape: Tuple[int, int, int],
                           half_integer: bool) -> Dict[str, Any]:
    """Seeded random kms/sinv in [0.5, 1.0], different per axis AND per Yee offset.

    A uniform table of ones hides the coefficient indexing completely: every axis
    would multiply by the same 1.0, so a swapped ``dsig``/``dsigu``, a swapped
    axis, or the integer set used where the half-integer one belongs would all
    reproduce bit-for-bit. These are drawn from a different stream per offset for
    exactly that reason, and never touch 1.0.
    """
    rng = np.random.default_rng(SEED + (11 if half_integer else 12))
    coefficients: Dict[str, Any] = {}
    for axis, name in enumerate(AXIS_NAMES):
        n = shape[axis]
        for label in ("kms", "sinv"):
            values = rng.uniform(0.5, 1.0, size=n).astype(np.float32)
            coefficients[f"{label}_{name}"] = xp.asarray(
                np.ascontiguousarray(values.reshape(broadcast_shape(axis, n))))
    return coefficients


def real_pml_layer(xp: Any, shape: Tuple[int, int, int],
                   boundaries: Sequence[str], repo_root: str,
                   courant: float = 0.5):
    """Build a real ``meep_gpu.pml.PML`` whose tables cover exactly ``shape``.

    The synthetic tables exercise the indexing; this one exercises the physics
    the indexing is for — the true graded profile, the integer/half-integer split
    and the axis pairing, all together. ``meep_cell_count`` is
    ``int(size * a + 0.5)``, so resolution 1.0 with a cell size equal to the cell
    count lands on the shape exactly.

    An axis of one cell is INVARIANT (Grid refuses a wall there) and carries no
    layer: ``low + high > n_cells`` would not fit anyway.
    """
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from meep_gpu.grid import Grid  # noqa: PLC0415 - deliberately late
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(
        resolution=1.0,
        cell_size=(float(shape[0]), float(shape[1]), float(shape[2])),
        boundaries=tuple(boundaries),
        dimensions=3,
        courant=float(courant),
        xp=xp,
    )
    if tuple(grid.shape) != tuple(shape):
        raise RuntimeError(
            f"Grid built {tuple(grid.shape)} for a requested {tuple(shape)}; the "
            f"real-PML leg needs the tables to match the field arrays exactly.")
    thickness = tuple((2, 2) if shape[axis] >= 6 else (0, 0) for axis in range(3))
    return grid, PML(grid=grid, thickness=thickness)


def layer_coefficients(pml, half_integer: bool) -> Dict[str, Any]:
    """The six broadcast-shaped curl coefficients of one sub-step, off a real layer.

    stepping._curl_coefficients (:2418): the B curl reads the HALF-INTEGER set
    and the D curl the integer one. Backwards is a half-cell error, not a crash,
    which is why the gate's mutation 6 drives exactly this pairing.
    """
    suffix = "_h" if half_integer else ""
    return {f"{label}_{name}": getattr(pml, f"{label}_{name}{suffix}")
            for name in AXIS_NAMES for label in ("kms", "sinv")}


def flatten_coefficients(coefficients: Dict[str, Any]) -> Dict[str, Any]:
    """Contiguous 1-D views of the broadcast tables, for the kernel's bare indexing.

    Flattened ONCE here and handed to the launcher, never per launch: the complex
    wrappers' ``ascontiguousarray(...ravel())`` is six device allocations a
    sub-step for tables that never change (port-reference defect 5.14).
    ``reshape(-1)`` on an (n,1,1) array is a view, which the assertions confirm
    rather than assume.
    """
    flat: Dict[str, Any] = {}
    for key, table in coefficients.items():
        view = table.reshape(-1)
        if not view.flags.c_contiguous:
            raise RuntimeError(f"{key} did not flatten to a contiguous view")
        if view.dtype != np.float32:
            raise RuntimeError(f"{key} is {view.dtype}, not float32")
        flat[key] = view
    return flat


#: Every array the real-field PML curl reads or writes.
PML_FIELD_NAMES: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Ex", "Ey", "Ez",
    "Dx", "Dy", "Dz", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")

#: How the real-PML leg's arrays are drawn, and it is a COVERAGE AXIS.
#:
#: ``uniform`` is what this leg had until 2026-08-15 and reaches NORMAL NUMBERS
#: ONLY: ``uniform(-1, 1)`` cannot produce a subnormal, and it cannot produce a
#: signed zero either. A verdict from it says nothing about the value class the
#: float32 subnormal policy governs — so the 120/120 could be re-cut under
#: ``flush`` and under ``keep`` and return 120/120 twice, establishing nothing
#: about the policy (disposition §1.4, §1.1).
#:
#: ``subnormal_band`` is the class the dispersive leg has carried since its
#: recon, ported here. It seeds every field IN the band rather than constructing
#: a cancellation into it, because a curl has no cancellation of its own to
#: exploit: two float32 neighbours near 1.0 are at least one ulp (~6e-8) apart, so
#: no seeding around a normal base can drive ``sf - f1`` below FLT_MIN. Values
#: spanning 1e-45 to 1e-30 put the operands, the differences, ``dtdx * curl``,
#: ``fprev * kms`` and the whole recurrence in and around the band, and one cell in
#: sixteen is forced to an exact zero, a NEGATIVE zero, the smallest normal or the
#: largest subnormal.
#:
#: THE TWO VERDICTS ARE REPORTED SEPARATELY and only the normal-number one gates,
#: exactly as on the dispersive leg and for the same measured reason: in the
#: subnormal band it is the ARRAY PATH, not the kernel, that leaves IEEE (CuPy
#: appends ``-ftz=true`` to every NVRTC compile), so folding the two into one
#: boolean would report a kernel defect that is not there.
PML_VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: Exact zero, negative zero, the smallest normal float32, the largest subnormal,
#: and the smallest subnormal in both signs. The values a policy decides the fate
#: of, and the ones ``uniform(-1, 1)`` provably never draws.
SUBNORMAL_NEEDLES = np.array(
    [0.0, -0.0, 1.1754944e-38, -1.1754944e-38, 1.1754942e-38, 1e-45, -1e-45],
    dtype=np.float32)


def pml_field_hosts(shape: Tuple[int, int, int], rng,
                    value_class: str = "uniform") -> Dict[str, np.ndarray]:
    """The HOST arrays of one real-PML case, before they go to the device.

    Split out of :func:`make_pml_fields` so the operand classes can be exercised
    and asserted on a laptop — that the uniform class cannot produce a subnormal
    or a signed zero, and that the band class does, is a property of this
    function and of no device (``test_pml_gate_harness.py``).

    The auxiliaries start NONZERO in both classes. A zero ``fu`` makes ``fu*kms``
    exactly zero on the first sub-step whatever ``kms`` is, so a mis-indexed
    coefficient would only show from step two — precisely the divergence the
    multi-step leg exists to catch, and no reason to hide it from the
    single-launch leg as well.
    """
    if value_class not in PML_VALUE_CLASSES:
        raise ValueError(f"value class {value_class!r} is not one of {PML_VALUE_CLASSES}")
    if value_class == "uniform":
        return {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                for name in PML_FIELD_NAMES}
    return subnormal_band_hosts(PML_FIELD_NAMES, shape, rng)


def subnormal_band_hosts(names: Sequence[str], shape: Tuple[int, int, int],
                         rng) -> Dict[str, np.ndarray]:
    """Every named array seeded IN the subnormal band, with the needles forced in.

    Factored out of :func:`pml_field_hosts` when the constitutive leg gained the
    same axis, so there is ONE band draw on this track rather than two that drift.
    The draw order is unchanged — one pass assigning the band magnitudes, then a
    second forcing one cell in sixteen to a needle — because
    ``test_pml_gate_harness.py`` pins the curl class's census exactly, against a
    named seed and shape, and a re-ordered draw would move those numbers without
    moving anything the gate measures.

    Values spanning 1e-45 to 1e-30 put the operands, their products with the
    absorber coefficients and the whole recurrence in and around the band. The
    needles are the values a policy decides the fate of and that ``uniform(-1, 1)``
    provably never draws.
    """
    host: Dict[str, np.ndarray] = {}
    for name in names:
        exponents = rng.uniform(-45.0, -30.0, size=shape)
        signs = np.where(rng.integers(0, 2, size=shape) == 0, -1.0, 1.0)
        host[name] = (signs * np.power(10.0, exponents)).astype(np.float32)
    for name in names:
        picks = rng.integers(0, 16, size=shape) == 0
        choice = SUBNORMAL_NEEDLES[rng.integers(0, SUBNORMAL_NEEDLES.size, size=shape)]
        host[name] = np.where(picks, choice, host[name]).astype(np.float32)
    return host


def make_pml_fields(shape: Tuple[int, int, int], rng,
                    value_class: str = "uniform") -> Dict[str, Any]:
    """Seeded float32 device arrays for the twelve fields and the six auxiliaries."""
    return {name: cp.asarray(np.ascontiguousarray(array))
            for name, array in pml_field_hosts(shape, rng, value_class).items()}


def operand_census(arrays: Dict[str, np.ndarray]) -> Dict[str, int]:
    """How many subnormals, signed zeros and zeros a leg's operands ACTUALLY hold.

    A LEG WHOSE OPERANDS CANNOT CONTAIN THE CLASS IT CLAIMS TO TEST MEASURES
    NOTHING, and reports a pass while doing it. That is not hypothetical here:
    the whole reason the 2026-08-09/10 certification says nothing about the
    subnormal policy is that its only operand class was ``uniform(-1, 1)``, which
    provably draws no subnormal and no signed zero — so it would have returned
    120/120 under either policy (disposition §1.4).

    Classified off the BITS, not by comparison: ``x < FLT_MIN`` is true for zero
    too, and ``x == -0.0`` is true for ``+0.0``. Exponent field zero with a
    nonzero mantissa is exactly the subnormal encoding, and ``0x80000000`` is
    exactly negative zero.

    Recorded per case in the artifact so NON-VACUITY is a property of the record
    rather than of a reader's trust in the generator. ``test_pml_gate_harness.py``
    pins the same counts off-device, exactly, against a seed and a shape the
    record names.
    """
    flat = np.concatenate([np.asarray(a, dtype=np.float32).ravel()
                           for a in arrays.values()])
    raw = flat.view(np.uint32)
    exponent = (raw >> 23) & 0xFF
    mantissa = raw & 0x7FFFFF
    return {
        "values": int(flat.size),
        "subnormals": int(((exponent == 0) & (mantissa != 0)).sum()),
        "negative_zeros": int((raw == 0x80000000).sum()),
        "zeros": int((raw == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_compilation(module, results: Dict[str, Any], out_path: str) -> None:
    inventory = kernel_inventory(module)
    results["kernel_inventory"] = inventory
    log(f"[compile] {len(inventory)} registered kernels found in _get_kernel's code map")
    results["compilation"] = {}
    for label, options in OPTION_SETS:
        per_option: Dict[str, Any] = {}
        for entry in inventory:
            key = f"{entry['name']}:{'complex' if entry['is_complex'] else 'real'}"
            module.cp = CupyShim(options)
            module._clear_kernel_cache()
            started = time.time()
            try:
                kernel = module._get_kernel(entry["name"], entry["is_complex"])
                _ = kernel.kernel  # force NVRTC compilation now
                per_option[key] = {"ok": True,
                                   "seconds": round(time.time() - started, 3)}
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - the error text IS the result
                per_option[key] = {"ok": False,
                                   "error_type": type(exc).__name__,
                                   "error": str(exc)[:4000],
                                   "seconds": round(time.time() - started, 3)}
                status = f"FAIL {type(exc).__name__}"
            log(f"[compile] options={label or '()'} {key}: {status} "
                f"({per_option[key]['seconds']} s)")
        results["compilation"][label] = {"options": list(options),
                                         "kernels": per_option}
        save(results, out_path)
    module.cp = cp
    module._clear_kernel_cache()


def run_scalar_dtype_probe(shape: Tuple[int, int, int], results: Dict[str, Any],
                           dtdx_values: Sequence[float]) -> None:
    """Does a Python-float scalar multiply differ from an np.float32 one?

    ``stepping.py`` multiplies by ``dtdx = grid.dt / grid.dx``, a Python float
    (double); the kernel wrappers pass ``np.float32(dtdx)``. If CuPy evaluated
    the scalar in double this would be a third, independent divergence, so it is
    measured rather than assumed.
    """
    rng = np.random.default_rng(SEED + 1)
    out: Dict[str, Any] = {}
    for dtype_name, dtype in (("complex64", np.complex64), ("float32", np.float32)):
        host = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        if dtype == np.complex64:
            host = (host + 1j * rng.uniform(-1.0, 1.0, size=shape)).astype(np.complex64)
        arr = cp.asarray(np.ascontiguousarray(host))
        for dtdx in dtdx_values:
            as_double = float(dtdx) * arr
            as_float32 = np.float32(dtdx) * arr
            key = f"{dtype_name}@dtdx={dtdx!r}"
            out[key] = {
                "float32_exactly_representable":
                    bool(float(np.float32(dtdx)) == float(dtdx)),
                "python_float_vs_np_float32": bit_compare(as_double, as_float32),
                "result_dtype_python_float": str(as_double.dtype),
                "result_dtype_np_float32": str(as_float32.dtype),
            }
            log(f"[scalar] {key}: python-float == np.float32 -> "
                f"{out[key]['python_float_vs_np_float32']['bit_identical']}")
    results["scalar_dtype"] = out


def run_bit_identity(module, results: Dict[str, Any], out_path: str,
                     shapes: Sequence[Tuple[int, int, int]],
                     dtdx_values: Sequence[float]) -> None:
    cases: List[Dict[str, Any]] = results.setdefault("bit_identity", [])
    dtypes = (("complex64", np.complex64), ("float32", np.float32))
    total = (len(OPTION_SETS) * len(shapes) * len(dtypes)
             * len(dtdx_values) * 2)
    index = 0
    for label, options in OPTION_SETS:
        for shape in shapes:
            for dtype_name, dtype in dtypes:
                for dtdx in dtdx_values:
                    for sub_step in ("step_B", "step_D"):
                        index += 1
                        started = time.time()
                        case = one_case(module, options, label, shape, dtype_name,
                                        dtype, dtdx, sub_step)
                        case["seconds"] = round(time.time() - started, 3)
                        cases.append(case)
                        log(
                            f"[bits] case {index}/{total} options={label} "
                            f"{sub_step} {dtype_name} shape={shape} dtdx={dtdx!r}: "
                            f"vs array_order identical="
                            f"{case['vs_array_order']['bit_identical']} "
                            f"(maxulp={case['vs_array_order'].get('max_ulp', 0)}, "
                            f"maxabs={case['vs_array_order'].get('max_abs_diff', 0.0):.3e}) | "
                            f"vs kernel_order identical="
                            f"{case['vs_kernel_order']['bit_identical']} "
                            f"(maxulp={case['vs_kernel_order'].get('max_ulp', 0)}) "
                            f"({case['seconds']} s)"
                        )
                        save(results, out_path)


def one_case(module, options: Sequence[str], label: str,
             shape: Tuple[int, int, int], dtype_name: str, dtype: Any,
             dtdx: float, sub_step: str) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    arrays = make_fields(shape, dtype, rng)

    # RETIRED 2026-08-26, AND REFUSED BY NAME RATHER THAN LEFT TO AttributeError.
    # This leg drove ``_step_B_fused`` / ``_step_D_fused``, the all-periodic triple-curl
    # launchers over two of the twelve kernels INHERITED from the original meep_gpu
    # bundle (commit 2cd0921, 2026-07-31). Those twelve were deleted: every surface they
    # targeted is reimplemented in this package's own modules, and the upstream
    # repository still exists separately as a build reference.
    #
    # WHAT THIS LEG ESTABLISHED IS NOT LOST, because it was never this probe that kept
    # it true. Its two findings -- that NVRTC's default FMA contraction breaks
    # bit-identity against the array path, and that the curl's associativity differs
    # from ``stepping.py``'s grouping -- are permanently expressed as
    # ``_COMPILE_OPTIONS = ('--fmad=false',)`` in every kernel module, and every
    # certified family's gate re-measures bit-identity against ``stepping`` on every
    # run. The probe measured a precondition once; the gates enforce it continuously.
    raise SystemExit(
        "the hand-track curl legs of this probe are RETIRED: they drove "
        "_step_B_fused / _step_D_fused, launchers over inherited kernels deleted on "
        "2026-08-26. The FMA and associativity findings they established are carried "
        "by _COMPILE_OPTIONS=('--fmad=false',) and re-measured by every certified "
        "family's bit-identity gate; see certification.json post_certification_edits.")

    sources = {name: arrays[name] for name in source_names}
    targets = {name: arrays[name] for name in target_names}

    # The same reference in both groupings, from the pristine inputs. This pair
    # is an ASSOCIATIVITY discriminator, not two independent readings: it says
    # which parenthesisation the kernel reproduces, and nothing about whether
    # either matches stepping.py. That second question belongs to a validator.
    ref_array = reference_step(cp, sources, targets, terms, dtdx, backward,
                               "array_order")
    ref_kernel = reference_step(cp, sources, targets, terms, dtdx, backward,
                                "kernel_order")

    # The kernel runs in place on its own copies of the targets.
    kernel_arrays = dict(arrays)
    for name in target_names:
        kernel_arrays[name] = arrays[name].copy()
    fields = FieldsShim(kernel_arrays)

    module.cp = CupyShim(options)
    module._clear_kernel_cache()
    error: Optional[str] = None
    try:
        launcher(fields, dtdx)
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        module.cp = cp
        module._clear_kernel_cache()

    case: Dict[str, Any] = {
        "option_label": label,
        "options": list(options),
        "sub_step": sub_step,
        "dtype": dtype_name,
        "shape": list(shape),
        "dtdx": repr(dtdx),
        "dtdx_float32_exact": bool(float(np.float32(dtdx)) == float(dtdx)),
        "launch_error": error,
    }
    if error is not None:
        case["vs_array_order"] = {"bit_identical": False, "differing_floats": -1,
                                  "total_floats": 0}
        case["vs_kernel_order"] = {"bit_identical": False, "differing_floats": -1,
                                   "total_floats": 0}
        return case

    case["vs_array_order"] = combine({
        name: bit_compare(getattr(fields, name), ref_array[name])
        for name in target_names})
    case["vs_kernel_order"] = combine({
        name: bit_compare(getattr(fields, name), ref_kernel[name])
        for name in target_names})
    case["array_vs_kernel_order_reference"] = combine({
        name: bit_compare(ref_array[name], ref_kernel[name])
        for name in target_names})
    return case


# ---------------------------------------------------------------------------
# The real-field PML gate
# ---------------------------------------------------------------------------
#
# Sweep axes, and why each one is in the product rather than trimmed:
#
# * dtdx — 0.35 is MANDATORY. At 0.5 the scaling is exact in binary and the
#   FMA/associativity discrepancies vanish entirely, so a gate testing only 0.5
#   certifies broken kernels. Both are kept so the pair separates scalar
#   rounding from the arithmetic.
# * shapes — an odd non-cube, a cube, a 2-D shape with an invariant nz = 1, and
#   a thin asymmetric one. The non-cube is what stops a swapped-axis index bug
#   (i <-> k, sx <-> sz) from passing: on 32^3 every stride is interchangeable.
# * boundaries — the first three are measured-real configurations; the fourth is
#   asymmetric (metallic, periodic, metallic) so a per-axis bug cannot hide
#   behind a uniform declaration. This axis is the one that catches porting the
#   complex kernels' hard-coded "X/Y metallic, Z periodic" layout, which is
#   bit-identical on a 2-D XY sheet and wrong on everything else.
# * sub_step — step_B and step_D differ in shift direction, coefficient
#   sub-lattice AND which axes carry the wall mask (one per component on B, two
#   on D). Neither side implies the other.
# * value class — see :data:`PML_VALUE_CLASSES`. The uniform class reaches normal
#   numbers only, so a gate carrying it alone is blind to every question the
#   float32 subnormal policy decides.

PML_SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (13, 11, 9), (32, 32, 32), (64, 48, 1), (17, 5, 3))
PML_BOUNDARY_SETS: Tuple[Tuple[str, str, str], ...] = (
    (PERIODIC, PERIODIC, PERIODIC),
    (METALLIC, METALLIC, PERIODIC),
    (METALLIC, METALLIC, METALLIC),
    (METALLIC, PERIODIC, METALLIC),
)
PML_DTDX: Tuple[float, ...] = (0.5, 0.35)

#: Consecutive launches in the dispersive and fused-pair multi-step legs. Left at
#: 8 deliberately: those legs belong to Triton kernel families whose CERTIFICATION
#: RECORDS were cut at this budget, and raising it here would silently re-scope
#: gates whose verdicts are already published. The curl leg has its own, below,
#: and so — since 2026-08-15 — does the constitutive pair.
MULTI_STEP_COUNT = 8

#: Consecutive launches in the REAL-STORAGE CONSTITUTIVE multi-step leg. Raised
#: from the shared 8 on 2026-08-15, and the reason the shared constant could not
#: be raised does not apply to this family: it has NO record to re-scope.
#: ``constitutive_kernels.CERTIFIED_KERNELS`` is empty and a test asserts it stays
#: empty until this gate runs, so nothing has ever been certified at 8 here.
#:
#: What DID apply was the measurement behind the curl's 60 — "8, 10 and 6
#: consecutive steps all passed a divergence that 40 did not"
#: (``triton_kernels/fingerprints.json``, ``bit_identity_gate.legs.whole_step._why_60``).
#: A CUDA record quoting 8 sits below a budget the sibling track has already
#: measured to be blind. Worse, the tranche's own laptop leg
#: (``test_constitutive_pml_real.py``) runs its expression-tree comparison at 60,
#: so before this the leg measuring the EXPRESSION TREE ran at 60 and the leg
#: measuring the COMPILED KERNEL ran at 8.
CONSTITUTIVE_MULTI_STEP_COUNT = 60

#: Consecutive launches in the REAL-FIELD PML CURL multi-step leg — the one this
#: track's certification quotes. Raised from 8 on 2026-08-15.
#:
#: 8 is not a budget, it is a blind spot, and the sibling track measured it:
#: ``triton_kernels/fingerprints.json``, ``bit_identity_gate.legs.whole_step._why_60``
#: — "8, 10 and 6 consecutive steps all passed a divergence that 40 did not.
#: 'Identical for N steps' is a claim about N, so the step budget is a coverage
#: axis and 60 is the budget this record certifies." Same probe, same comparator,
#: a kernel of the same shape. 60 is taken from there so the two tracks' claims
#: stay commensurable, and the whole gate leg cost 13.92 s at 8.
PML_MULTI_STEP_COUNT = 60


class HandTrackAdapter:
    """The hand-written CUDA track, driven through its own launch wrappers.

    Both tracks are judged by THIS file. The hand track's adapter lives here
    because its module is a plain kernel-string module; a track whose module has
    a different shape supplies its own by exporting ``PROBE_ADAPTER`` (see
    :func:`build_adapter`).
    """

    track = "hand"
    # (label, guard token, is the gate's primary guard). The primary must reach
    # bit-identity; the secondary exists to DEMONSTRATE that the guard is doing
    # the work rather than to assert it.
    guard_sets = (
        ("fmad_false", ("--fmad=false",), True),
        ("default_no_options", (), False),
    )
    guard_description = "NVRTC compile options"

    #: The constitutive kernels live in a SIBLING module, and deliberately so:
    #: ``step_curl_kernels.py``'s fourteen device strings are pinned by digest in
    #: ``certification.json`` (individually, concatenated, and as a whole-file
    #: hash), so adding a kernel there would force a record edit no device run
    #: backs. The adapter loads the sibling from the same directory rather than
    #: taking a second ``--module`` path, so a gate command line is unchanged.
    CONSTITUTIVE_MODULE = "constitutive_kernels.py"
    #: The source-mutation machinery rewrites module-level attributes; two
    #: modules means the attribute name alone no longer says which module owns
    #: it, so the map is explicit.
    _SOURCE_MODULE_OF = {
        "_step_B_pml_real_kernel_code": "curl",
        "_step_D_pml_real_kernel_code": "curl",
        "_update_H_pml_real_kernel_code": "constitutive",
        "_update_E_pml_real_kernel_code": "constitutive",
    }

    def __init__(self, module):
        self.module = module
        self._mutated_source: Dict[str, str] = {}
        self._constitutive = None
        self._guard: Optional[Sequence[str]] = None
        self._guard_active = False

    def constitutive_module(self):
        """Load ``constitutive_kernels.py`` beside the curl module, once.

        Lazily, because a run that sweeps only the curl legs must not need the
        sibling to exist — and because on a host with no CuPy neither module
        imports at all, which is exactly the state every laptop is in.

        A LATE LOAD MUST NOT ESCAPE THE SWEPT GUARD, and that is not hypothetical:
        the ``cgate`` leg arms no source mutation, so nothing calls
        :meth:`source_attributes` and this module first loads INSIDE
        :meth:`launch`, i.e. after ``begin_guard`` has already installed the shim
        on the modules it knew about. Without the block below, that leg would
        compile the constitutive kernels under the FILE's own option tuple while
        the artifact reported the swept one — and the ``default_no_options``
        control would silently run with ``--fmad=false``, turning a 0/N control
        into an N/N one. Same failure family as §13.4's three: a leg reporting a
        result for a condition it never applied.
        """
        if self._constitutive is None:
            path = os.path.join(os.path.dirname(os.path.abspath(
                self.module.__file__)), self.CONSTITUTIVE_MODULE)
            spec = importlib.util.spec_from_file_location(
                "probe_constitutive_kernels", path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load constitutive module from {path}")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded
            spec.loader.exec_module(loaded)
            if self._guard_active:
                loaded.cp = CupyShim(self._guard or ())
                for attribute, source in self._mutated_source.items():
                    if hasattr(loaded, attribute):
                        setattr(loaded, attribute, source)
                loaded._clear_kernel_cache()
            self._constitutive = loaded
        return self._constitutive

    def _modules(self) -> Tuple[Any, ...]:
        """Every module whose kernels this adapter can launch, curl first.

        The sibling is included only once it has been loaded — a curl-only run
        never touches it, and a guard change must not import it as a side effect.
        """
        return ((self.module,) if self._constitutive is None
                else (self.module, self._constitutive))

    def set_source_mutation(self, mutated: Dict[str, str]) -> None:
        self._mutated_source = mutated

    def begin_guard(self, guard) -> None:
        """Install the compile options and drop the cache so the next call compiles.

        BOTH modules get the shim and both get the cache drop. Installing it on
        one only would leave the other compiling under the file's own options
        while the artifact reported the swept ones — a leg measuring a guard it
        never applied, which is the failure family §13.4 records three instances
        of.
        """
        for module in self._modules():
            module.cp = CupyShim(guard)
        for attribute, source in self._mutated_source.items():
            self._source_module(attribute)
            for module in self._modules():
                if hasattr(module, attribute):
                    setattr(module, attribute, source)
        for module in self._modules():
            module._clear_kernel_cache()
        # Recorded LAST, so ``constitutive_module``'s late-load block sees the
        # mutations that were just installed rather than an empty map.
        self._guard, self._guard_active = guard, True

    def _source_module(self, attribute: str):
        """Load the module that owns ``attribute`` so a mutation can reach it."""
        if self._SOURCE_MODULE_OF.get(attribute) == "constitutive":
            return self.constitutive_module()
        return self.module

    def end_guard(self) -> None:
        self._guard_active = False
        for module in self._modules():
            module.cp = cp
            module._clear_kernel_cache()

    def launch(self, sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any],
               codes: Sequence[Any], dtdx: float,
               extra: Optional[Dict[str, Any]] = None) -> None:
        if sub_step.startswith("fused_pair"):
            raise NotImplementedError(
                "the hand-CUDA track has no cross-sub-step fused kernel; the pair "
                "is a Triton-track kernel and this leg does not apply to it.")
        shim = FieldsShim(arrays)
        if sub_step in ("update_H", "update_E"):
            side = sub_step[-1]
            # The gate's fixture keys inverse epsilon by TARGET
            # (``inv_eps_Ex``/``_Ey``/``_Ez``); the kernel wrapper reads it
            # through ``Fields.inverse_epsilon_for``, so the shim answers that
            # call from those arrays. Three distinct volumes, never one — binding
            # a single volume for all three is the complex kernel's own defect
            # (it passes ``fields.inv_eps``, which is the Ez view) and a fixture
            # that handed one array to all three could not see it.
            shim.inverse_epsilon_for = lambda component: getattr(
                shim, "inv_eps_" + component)
            module = self.constitutive_module()
            # ``tables=`` is keyword-only and the gate is what it exists for: the
            # fixture's synthetic tables come from no PML, and the host mutation
            # ``swap_constitutive_sublattice`` hands in deliberately mis-paired
            # ones. A planner passes ``pml=`` instead and the launcher derives the
            # pairing from the side, which is the invariant this door bypasses ON
            # PURPOSE and no other caller may.
            module.update_fused_pml_real(side, shim, tables=flat)
            return
        if sub_step == "step_B":
            self.module._step_B_fused_pml_real(shim, flat, codes, dtdx)
        else:
            self.module._step_D_fused_pml_real(shim, flat, codes, dtdx)

    def source_attributes(self) -> Tuple[str, ...]:
        """Every kernel source string a mutation may rewrite, across both modules.

        The constitutive pair is named unconditionally, which forces its module
        to load: ``apply_source_mutation`` raises when a mutation matches nothing
        ANYWHERE, and a mutation aimed at the constitutive kernels would
        otherwise "match nothing" purely because the module had not been imported
        yet — a false drift report that would read as a real one.
        """
        self.constitutive_module()
        return ("_step_B_pml_real_kernel_code",
                "_step_D_pml_real_kernel_code",
                "_update_H_pml_real_kernel_code",
                "_update_E_pml_real_kernel_code")

    def source_attribute_owner(self, attribute: str):
        """Which loaded module holds ``attribute`` — used by the mutation reader."""
        return self._source_module(attribute)

    def mutated_kernels(self) -> Dict[str, Any]:
        return {}


class ExternalTrackAdapter:
    """A track that supplies its own adapter through ``module.PROBE_ADAPTER``.

    The contract, so a second implementation is judged by the same file without
    this one dictating its internals. ``PROBE_ADAPTER`` is a mapping with:

    * ``track``            — the track name, for the artifact directory;
    * ``guard_sets``       — ``((label, token, is_primary), ...)``; the token is
      whatever that track's rounding guard is (a compile flag, a kwarg). The
      track MUST supply a non-primary set with the guard OFF, so the pair
      (identical with, non-identical without) is measured rather than asserted.
    * ``guard_description``— what the token means, one line, for the artifact;
    * ``begin_guard(token)`` / ``end_guard()`` — install and drop it;
    * ``launch(sub_step, arrays, flat, codes, dtdx)`` — run one sub-step in
      place on ``arrays``. ``arrays`` is keyed by component name
      (``Bx``/``fu_Bx``/``Ex``/...), ``flat`` by ``kms_x``/``sinv_x``/... as
      contiguous 1-D views, ``codes`` is the per-axis 0/1 periodic/metallic
      triple, and ``dtdx`` is a Python float.
    * ``source_attributes()`` and ``set_source_mutation(mapping)`` — optional;
      present only if the track's kernels are source strings the mutation leg
      can rewrite.
    """

    def __init__(self, module, spec: Dict[str, Any]):
        self.module = module
        self._spec = spec
        self.track = spec["track"]
        self.guard_sets = tuple(spec["guard_sets"])
        self.guard_description = spec.get("guard_description", "<undeclared>")

    def set_source_mutation(self, mutated: Dict[str, str]) -> None:
        hook = self._spec.get("set_source_mutation")
        if hook is None:
            raise NotImplementedError("this track exposes no source-mutation hook")
        hook(mutated)

    def begin_guard(self, guard) -> None:
        self._spec["begin_guard"](guard)

    def end_guard(self) -> None:
        self._spec["end_guard"]()

    def launch(self, sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any],
               codes: Sequence[Any], dtdx: float,
               extra: Optional[Dict[str, Any]] = None) -> None:
        """``extra`` carries what only the fused pair needs: the constitutive
        half's own coefficient tables and the ``zero_metal`` triple.

        Forwarded only when the track's ``launch`` accepts it, so an adapter
        written before the cross-sub-step pair existed keeps its four-argument
        signature and simply cannot be asked for the fused leg.
        """
        launcher = self._spec["launch"]
        if "extra" in inspect.signature(launcher).parameters:
            launcher(sub_step, arrays, flat, codes, dtdx, extra)
            return
        if extra:
            raise NotImplementedError(
                f"track {self.track!r} does not accept the fused pair's extra "
                f"bindings ({sorted(extra)}); its launch() takes no `extra`.")
        launcher(sub_step, arrays, flat, codes, dtdx)

    def source_attributes(self) -> Tuple[str, ...]:
        hook = self._spec.get("source_attributes")
        return tuple(hook()) if hook is not None else ()

    def mutated_kernels(self) -> Dict[str, Any]:
        """The currently-mutated kernel per PLAN CLASS, for the whole-step leg.

        WITHOUT THIS THE WHOLE-STEP LEG IS DISARMED, and it was — measured. The
        source-mutation machinery routes a mutated kernel through the
        ``plan_*_from_arrays(kernel=...)`` argument, which only the synthetic legs
        use; ``plan_step`` builds engine-route plans that hold ``kernel=None`` and
        therefore launch the SHIPPED kernel. So every whole-step mutation leg ran
        unmutated code and reported a pass: ``reload_B_from_memory_but_shift``, a
        defect the synthetic leg catches 0/60, came back **30/30 identical** here.

        That is the third instance of §13.4's family in this file (the dropped
        ``kernel=`` argument, the host mutation plumbed into one leg and not the
        other, and now this), and the same sentence applies: a leg that reports a
        pass for a mutation it never applied is worse than no leg.
        """
        hook = self._spec.get("mutated_kernels")
        return dict(hook()) if hook is not None else {}

    @property
    def source_mutations(self):
        """This track's spellings of the gate's four source defects, if it has them.

        WHICH defects are injected, and what must happen when they are, stays with
        the gate; only the needles are per-track, because one track's kernel is a
        CUDA string and the other's is Python. A track that supplies none falls
        back to :data:`SOURCE_MUTATIONS`, which is the hand track's set.
        """
        return self._spec.get("source_mutations")


def build_adapter(track: str, module):
    if track == "hand":
        return HandTrackAdapter(module)
    spec = getattr(module, "PROBE_ADAPTER", None)
    if spec is None:
        raise SystemExit(
            f"--track {track} needs the module to export PROBE_ADAPTER; see "
            f"ExternalTrackAdapter's docstring for the contract.")
    return ExternalTrackAdapter(module, spec)


def pml_case_key(coefficient_source: str, shape, boundaries, dtdx, sub_step,
                 value_class: str = "uniform") -> str:
    return (f"{coefficient_source}|{'x'.join(str(v) for v in shape)}|"
            f"{'/'.join(b[0] for b in boundaries)}|dtdx={dtdx!r}|{sub_step}|"
            f"{value_class}")


def _reference_arrays(arrays: Dict[str, Any], sub_step: str):
    """The (sources, targets, auxiliaries) views one sub-step reads and writes."""
    if sub_step == "step_B":
        return (("Ex", "Ey", "Ez"), ("Bx", "By", "Bz"),
                ("fu_Bx", "fu_By", "fu_Bz"))
    return (("Hx", "Hy", "Hz"), ("Dx", "Dy", "Dz"),
            ("fu_Dx", "fu_Dy", "fu_Dz"))


def compare_pml_state(arrays: Dict[str, Any], reference: Dict[str, Any],
                      target_names, aux_names, compare_fu: bool,
                      comparator=bit_compare) -> Dict[str, Any]:
    """Compare BOTH the target field AND its auxiliary, per component.

    ``compare_fu=False`` is the harness's own mutation: it exists only to
    measure whether the auxiliary comparison is load-bearing, and a gate run
    never sets it.
    """
    parts = {name: comparator(arrays[name], reference[name]) for name in target_names}
    if compare_fu:
        parts.update({name: comparator(arrays[name], reference[name])
                      for name in aux_names})
    return combine(parts)


def allclose_compare(a_dev: Any, b_dev: Any) -> Dict[str, Any]:
    """The harness's mutation 1: a magnitude comparison in place of the bytes.

    Kept beside :func:`bit_compare` so the difference is one argument. A
    magnitude test is what let a signed-zero defect survive earlier in this
    project — ``abs((1+0j) - (1-0j))`` is exactly 0.0 — and it is the most
    plausible way a last-bit regrouping reaches dispatch.
    """
    a = to_host(a_dev)
    b = to_host(b_dev)
    close = bool(np.allclose(a, b, rtol=1e-5, atol=1e-8))
    return {"bit_identical": close, "differing_floats": 0 if close else -1,
            "total_floats": int(a.size), "comparator": "np.allclose"}


def run_pml_bit_identity(adapter, results: Dict[str, Any], out_path: str,
                         repo_root: str, coefficient_sources: Sequence[str],
                         compare_fu: bool = True,
                         comparator=bit_compare,
                         guard_filter: Optional[str] = None,
                         host_mutation: Optional[str] = None,
                         label: str = "gate",
                         value_classes: Sequence[str] = PML_VALUE_CLASSES) -> Dict[str, Any]:
    """The real-field PML gate: the full product, both arrays, both groupings.

    PASS is every NORMAL-NUMBER case bit-identical to ``array_order`` on both the
    target and the auxiliary. ``kernel_order`` is carried alongside so a failure
    separates associativity from everything else in one read, and the
    subnormal-band class is counted SEPARATELY — see :data:`PML_VALUE_CLASSES`
    for why it is published rather than folded into the pass.
    """
    cases: List[Dict[str, Any]] = []
    guards = [g for g in adapter.guard_sets
              if guard_filter is None or g[0] == guard_filter]
    combinations = [
        (source, shape, boundaries, dtdx, sub_step, value_class)
        for source in coefficient_sources
        for shape in PML_SHAPES
        for boundaries in PML_BOUNDARY_SETS
        for dtdx in PML_DTDX
        for sub_step in ("step_B", "step_D")
        for value_class in value_classes
    ]
    total = len(guards) * len(combinations)
    index = 0
    for guard_label, guard, is_primary in guards:
        adapter.begin_guard(guard)
        try:
            for source, shape, boundaries, dtdx, sub_step, value_class in combinations:
                index += 1
                started = time.time()
                case = one_pml_case(adapter, shape, boundaries, dtdx, sub_step,
                                    source, repo_root, compare_fu, comparator,
                                    host_mutation, value_class)
                case.update({"guard_label": guard_label, "guard": list(guard),
                             "guard_is_primary": is_primary,
                             "seconds": round(time.time() - started, 3)})
                cases.append(case)
                if case.get("skipped"):
                    log(f"[pml:{label}] case {index}/{total} guard={guard_label} "
                        f"{pml_case_key(source, shape, boundaries, dtdx, sub_step, value_class)}: "
                        f"SKIPPED ({case['skipped']})")
                else:
                    verdict = case["vs_array_order"]
                    log(f"[pml:{label}] case {index}/{total} guard={guard_label} "
                        f"{pml_case_key(source, shape, boundaries, dtdx, sub_step, value_class)}: "
                        f"identical={verdict['bit_identical']} "
                        f"(differing={verdict['differing_floats']}/"
                        f"{verdict['total_floats']}, "
                        f"maxulp={verdict.get('max_ulp', 0)}) "
                        f"vs_kernel_order={case['vs_kernel_order']['bit_identical']} "
                        f"({case['seconds']} s)")
                results.setdefault("pml_bit_identity", {})[label] = summarize_pml(cases)
                results["pml_bit_identity"][label]["cases"] = cases
                save(results, out_path)
        finally:
            adapter.end_guard()
    summary = summarize_pml(cases)
    summary["cases"] = cases
    results.setdefault("pml_bit_identity", {})[label] = summary
    save(results, out_path)
    return summary


def summarize_pml(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass counts as N/N per guard, never as the word 'matches'.

    THE PASS IS THE NORMAL-NUMBER ONE and the block says so out loud. The
    subnormal-band class is counted beside it, per guard and overall, because in
    that band it is the array path — not the kernel — that leaves IEEE, and a
    single boolean over both classes would report a kernel defect that is not
    there. ``primary_pass`` therefore ranges over the ``uniform`` class; a run
    that swept only the band would have no normal-number cases and
    ``primary_pass`` would be False, which is the correct answer for a run that
    measured nothing the verdict is about.
    """
    per_guard: Dict[str, Dict[str, int]] = {}
    per_value_class: Dict[str, Dict[str, int]] = {}
    for case in cases:
        bucket = per_guard.setdefault(case["guard_label"],
                                      {"identical": 0, "ran": 0, "skipped": 0})
        klass = per_value_class.setdefault(
            case.get("value_class", "uniform"),
            {"identical": 0, "ran": 0, "skipped": 0, "moved_nothing": 0})
        if case.get("skipped"):
            bucket["skipped"] += 1
            klass["skipped"] += 1
            continue
        identical = int(case["vs_array_order"]["bit_identical"])
        bucket["ran"] += 1
        bucket["identical"] += identical
        klass["ran"] += 1
        klass["identical"] += identical
        # A case where the ORACLE moved no target and no auxiliary compared zero
        # against zero and would report identical whatever the kernel did.
        if (case.get("reference_moved_targets", 1) == 0
                and case.get("reference_moved_auxiliaries", 1) == 0):
            klass["moved_nothing"] += 1
    primary = [c for c in cases
               if c.get("guard_is_primary") and not c.get("skipped")]
    # EXCLUDING the band, rather than selecting "uniform", on purpose: this
    # summarizer is shared with the constitutive leg (whose cases carry no value
    # class at all) and with the dispersive leg (whose classes are "uniform" and
    # "cancellation", already split by summarize_dispersive on top of this). Only
    # the curl leg produces "subnormal_band", so only its accounting moves.
    normal = [c for c in primary if c.get("value_class") != "subnormal_band"]
    band = [c for c in primary if c.get("value_class") == "subnormal_band"]
    return {
        "per_guard": per_guard,
        "per_value_class": per_value_class,
        "primary_ran": len(normal),
        "primary_identical": sum(int(c["vs_array_order"]["bit_identical"])
                                 for c in normal),
        "primary_pass": bool(normal) and all(
            c["vs_array_order"]["bit_identical"] for c in normal)
        and any(c.get("reference_moved_targets", 1) for c in normal),
        "_primary_excludes_the_subnormal_band_class": True,
        "_primary_pass_requires_liveness": (
            "at least one normal-number case in which the array-path reference "
            "actually moved a target: zero == zero is bit-identical, and a leg "
            "that moved nothing would otherwise report a perfect pass"),
        "subnormal_band_ran": len(band),
        "subnormal_band_identical": sum(int(c["vs_array_order"]["bit_identical"])
                                        for c in band),
        # NON-VACUITY, summed off the per-case censuses. A band leg that ran
        # cases containing no subnormal and no signed zero exercised the policy
        # exactly as much as the uniform class did — i.e. not at all — and would
        # report a clean pass for it. The counts are carried so that reading
        # "subnormal_band 24/24" without reading this is not possible.
        "subnormal_band_operands": _class_operand_totals(band),
        "subnormal_band_is_non_vacuous": bool(band) and all(
            c.get("operand_census", {}).get("subnormals", 0) > 0
            and c.get("operand_census", {}).get("negative_zeros", 0) > 0
            for c in band),
        "normal_class_operands": _class_operand_totals(normal),
        "all_classes_ran": len(primary),
        "all_classes_identical": sum(int(c["vs_array_order"]["bit_identical"])
                                     for c in primary),
    }


def _class_operand_totals(cases: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Summed operand census over one value class's cases — the non-vacuity total."""
    totals = {"cases_with_a_census": 0, "values": 0, "subnormals": 0,
              "negative_zeros": 0, "zeros": 0}
    for case in cases:
        census = case.get("operand_census")
        if not census:
            continue
        totals["cases_with_a_census"] += 1
        for field in ("values", "subnormals", "negative_zeros", "zeros"):
            totals[field] += int(census.get(field, 0))
    return totals


def one_pml_case(adapter, shape, boundaries, dtdx, sub_step, coefficient_source,
                 repo_root, compare_fu, comparator,
                 host_mutation: Optional[str],
                 value_class: str = "uniform") -> Dict[str, Any]:
    """One launch of one sub-step, against both reference groupings."""
    case: Dict[str, Any] = {
        "sub_step": sub_step,
        "dtype": "float32",
        "shape": list(shape),
        "boundaries": list(boundaries),
        "dtdx": repr(dtdx),
        "dtdx_float32_exact": bool(float(np.float32(dtdx)) == float(dtdx)),
        "coefficient_source": coefficient_source,
        "host_mutation": host_mutation,
        "compare_fu": compare_fu,
        "value_class": value_class,
    }
    half_integer = sub_step == "step_B"

    if coefficient_source == "synthetic":
        coefficients = synthetic_coefficients(cp, shape, half_integer)
        kernel_coefficients = coefficients
        if host_mutation == "integer_coefficients_on_B":
            # Mutation 6: the D-side (integer) table handed to the B side — a
            # half-cell error, not a crash. It must be fed to the KERNEL ONLY.
            # Swapping the table under both legs is not a mutation at all: the
            # reference would then step the same wrong coefficients and agree,
            # which is exactly how this leg first reported 64/64 UNCAUGHT.
            kernel_coefficients = synthetic_coefficients(cp, shape, False)
    else:
        walled_invariant = [axis for axis in range(3)
                            if shape[axis] == 1 and boundaries[axis] == METALLIC]
        if walled_invariant:
            case["skipped"] = (
                f"axis {walled_invariant[0]} has one cell and a metallic wall; Grid "
                f"refuses that configuration, so no real layer exists for it")
            return case
        try:
            _, layer = real_pml_layer(cp, shape, boundaries, repo_root)
        except Exception as exc:  # noqa: BLE001 - the refusal IS the result
            case["skipped"] = f"{type(exc).__name__}: {exc}"[:400]
            return case
        coefficients = layer_coefficients(layer, half_integer)
        kernel_coefficients = (layer_coefficients(layer, False)
                               if host_mutation == "integer_coefficients_on_B"
                               else coefficients)

    rng = np.random.default_rng(SEED + 7)
    # Drawn on the host so the census is taken of the EXACT values that go to the
    # device, not of a second draw that would merely be distributed alike.
    hosts = pml_field_hosts(shape, rng, value_class)
    case["operand_census"] = operand_census(hosts)
    arrays = {name: cp.asarray(np.ascontiguousarray(array))
              for name, array in hosts.items()}
    source_names, target_names, aux_names = _reference_arrays(arrays, sub_step)

    references: Dict[str, Dict[str, Any]] = {}
    for grouping in ("array_order", "kernel_order"):
        state = {name: arrays[name].copy()
                 for name in target_names + aux_names}
        reference_pml_step(
            cp,
            {name: arrays[name] for name in source_names},
            {name: state[name] for name in target_names},
            {name: state[name] for name in aux_names},
            coefficients, np.float32(dtdx), sub_step, boundaries, grouping)
        references[grouping] = state

    kernel_arrays = dict(arrays)
    for name in target_names + aux_names:
        kernel_arrays[name] = arrays[name].copy()

    codes = list(0 if kind == PERIODIC else 1 for kind in boundaries)
    if host_mutation == "metallic_as_periodic":
        # Mutation 7: a wall told to the kernel as a wrap. Silently wrong, and
        # exactly what porting the complex kernels' hard-coded layout produces.
        codes = [0, 0, 0]
    codes = [np.int32(c) for c in codes]

    error: Optional[str] = None
    try:
        adapter.launch(sub_step, kernel_arrays,
                       flatten_coefficients(kernel_coefficients), codes, dtdx)
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:2000]
    case["launch_error"] = error
    if error is not None:
        for key in ("vs_array_order", "vs_kernel_order"):
            case[key] = {"bit_identical": False, "differing_floats": -1,
                         "total_floats": 0}
        return case

    # LIVENESS, and it is not decoration on the subnormal-band class. Zero equals
    # zero bitwise, so a case whose operands all underflowed to zero on BOTH sides
    # reports "identical" for the worst possible reason — the recurring failure
    # mode where an unmeasurable comparison scores as a perfect one. Counted from
    # the ARRAY-PATH reference (the oracle), never from the kernel, and
    # summarize_pml reports it per value class so a vacuous pass is visible in the
    # summary rather than only in the cases.
    case["reference_moved_targets"] = sum(
        int(bool((references["array_order"][name] != arrays[name]).any()))
        for name in target_names)
    case["reference_moved_auxiliaries"] = sum(
        int(bool((references["array_order"][name] != arrays[name]).any()))
        for name in aux_names)

    for grouping, key in (("array_order", "vs_array_order"),
                          ("kernel_order", "vs_kernel_order")):
        case[key] = compare_pml_state(kernel_arrays, references[grouping],
                                      target_names, aux_names, compare_fu,
                                      comparator)
    case["array_vs_kernel_order_reference"] = compare_pml_state(
        references["array_order"], references["kernel_order"],
        target_names, aux_names, compare_fu, bit_compare)
    return case


def run_pml_multi_step(adapter, results: Dict[str, Any], out_path: str,
                       repo_root: str, compare_fu: bool = True,
                       comparator=bit_compare,
                       host_mutation: Optional[str] = None,
                       label: str = "gate",
                       steps: int = PML_MULTI_STEP_COUNT) -> Dict[str, Any]:
    """``steps`` consecutive sub-steps per shape, byte-compared at EVERY step.

    THE AUXILIARY IS STATE. A kernel that gets ``f`` right and ``fu`` wrong is
    correct for exactly one launch and wrong forever after, and a single-shot
    comparison cannot see it. Sources are held fixed (the constitutive
    sub-steps are not part of this slice), so ``fu`` compounds its own history
    and any divergence in it grows rather than cancels.

    ``steps`` IS A COVERAGE AXIS, not a runtime knob — see
    :data:`PML_MULTI_STEP_COUNT`. The 2026-08-09/10 record was cut at 8, which the
    sibling track has since measured to be blind to a divergence 40 catches.

    THIS LEG STAYS ON THE UNIFORM OPERAND CLASS, deliberately. The recurrence
    decays ``fu`` by ``sinv`` every step with the sources held fixed, so a run
    seeded in the subnormal band walks both sides to zero within a few steps and
    the remaining fifty-odd comparisons are zero against zero — bit-identical, and
    about nothing. The band class belongs on the single-launch leg, where the
    operands are still in the band when the comparison is taken, and there it
    carries a liveness count for the same reason (:func:`summarize_pml`).
    """
    guard_label, guard, _ = adapter.guard_sets[0]
    boundaries = PML_BOUNDARY_SETS[3]  # the asymmetric one
    dtdx = 0.35
    runs: List[Dict[str, Any]] = []
    adapter.begin_guard(guard)
    try:
        for shape in PML_SHAPES:
            for sub_step in ("step_B", "step_D"):
                run = one_pml_multi_step(adapter, shape, boundaries, dtdx,
                                         sub_step, compare_fu, comparator,
                                         host_mutation, steps=steps)
                run.update({"guard_label": guard_label, "shape": list(shape),
                            "sub_step": sub_step, "steps_budget": int(steps),
                            "boundaries": list(boundaries), "dtdx": repr(dtdx)})
                runs.append(run)
                log(f"[pml:{label}:multi] shape={shape} {sub_step} "
                    f"{steps} steps: "
                    f"identical_steps={run['identical_steps']}/{steps}"
                    + (f" first_divergence_at_step={run['first_divergence']}"
                       if run["first_divergence"] is not None else ""))
                results.setdefault("pml_multi_step", {})[label] = {
                    "runs": runs,
                    "steps_budget": int(steps),
                    "all_steps_identical": all(
                        r["identical_steps"] == steps for r in runs),
                }
                save(results, out_path)
    finally:
        adapter.end_guard()
    summary = {"runs": runs,
               "steps_budget": int(steps),
               "all_steps_identical": all(
                   r["identical_steps"] == steps for r in runs)}
    results.setdefault("pml_multi_step", {})[label] = summary
    save(results, out_path)
    return summary


def one_pml_multi_step(adapter, shape, boundaries, dtdx, sub_step, compare_fu,
                       comparator, host_mutation,
                       steps: int = PML_MULTI_STEP_COUNT) -> Dict[str, Any]:
    half_integer = sub_step == "step_B"
    coefficients = synthetic_coefficients(cp, shape, half_integer)
    # Mutation 6 feeds the wrong sub-lattice to the KERNEL only; the reference
    # keeps the right one, or the two agree on being wrong together.
    flat = flatten_coefficients(
        synthetic_coefficients(cp, shape, False)
        if host_mutation == "integer_coefficients_on_B" else coefficients)
    rng = np.random.default_rng(SEED + 9)
    arrays = make_pml_fields(shape, rng)
    source_names, target_names, aux_names = _reference_arrays(arrays, sub_step)

    reference_state = {name: arrays[name].copy()
                       for name in target_names + aux_names}
    kernel_state = dict(arrays)
    for name in target_names + aux_names:
        kernel_state[name] = arrays[name].copy()

    codes = [0 if kind == PERIODIC else 1 for kind in boundaries]
    if host_mutation == "metallic_as_periodic":
        codes = [0, 0, 0]
    codes = [np.int32(c) for c in codes]

    sources = {name: arrays[name] for name in source_names}
    step_records: List[Dict[str, Any]] = []
    identical_steps = 0
    first_divergence: Optional[int] = None
    for step in range(1, int(steps) + 1):
        reference_pml_step(
            cp, sources,
            {name: reference_state[name] for name in target_names},
            {name: reference_state[name] for name in aux_names},
            coefficients, np.float32(dtdx), sub_step, boundaries, "array_order")
        try:
            adapter.launch(sub_step, kernel_state, flat, codes, dtdx)
            cp.cuda.runtime.deviceSynchronize()
        except Exception as exc:  # noqa: BLE001
            step_records.append({"step": step, "launch_error":
                                 f"{type(exc).__name__}: {exc}"[:1000]})
            first_divergence = first_divergence or step
            break
        verdict = compare_pml_state(kernel_state, reference_state, target_names,
                                    aux_names, compare_fu, comparator)
        step_records.append(
            {"step": step, "bit_identical": verdict["bit_identical"],
             "differing_floats": verdict["differing_floats"],
             "max_ulp": verdict.get("max_ulp", 0),
             "per_component": {k: v["bit_identical"]
                               for k, v in verdict["per_component"].items()}})
        if verdict["bit_identical"]:
            identical_steps += 1
        elif first_divergence is None:
            first_divergence = step
    return {"steps": step_records, "identical_steps": identical_steps,
            "first_divergence": first_divergence}


# ---------------------------------------------------------------------------
# The constitutive gate — update_H and update_E
# ---------------------------------------------------------------------------
#
# A SECOND transcription of stepping._apply_constitutive_pml (:2065), for the
# same reason the curl has one: two independent transcriptions agreeing with
# each other AND with the array path is what makes "bit-identical" mean the
# contract rather than "the kernel reproduces whatever I wrote twice".
#
# The sub-step reads NO NEIGHBOUR, so there is no ghost rule, no ownership mask
# and no boundary constexpr. The boundary axis stays in the sweep anyway: the
# gate certifies a covered SET, and a configuration whose ghost rule the curl
# refuses must not be half-fused by this kernel. It also changes the real
# coefficient tables (a metallic axis carries a wall), so it is not inert.
#
# THE COURANT AXIS IS NOT DECORATION HERE EITHER, and it enters differently
# from the curl's. This sub-step has no dtdx scalar; the non-representable
# multiplicands are the PML coefficients themselves — which §12.2 measured to be
# the real source of the contraction discrepancy on the curl too. The Courant
# reaches them through the dt/2 factor baked into the stored sigma, so the real
# layer is built at BOTH 0.5 and 0.35 and the two produce genuinely different
# tables. The synthetic tables are random in [0.5, 1.0) and non-representable by
# construction.
#
#   (target, source, own axis) — stepping.H_CONSTITUTIVE_TERMS / E_... (:226-227)
H_CONSTITUTIVE_TERMS = (("Hx", "Bx", 0), ("Hy", "By", 1), ("Hz", "Bz", 2))
E_CONSTITUTIVE_TERMS = (("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))
CONSTITUTIVE_TERMS = {"H": H_CONSTITUTIVE_TERMS, "E": E_CONSTITUTIVE_TERMS}
#: update_eh reads INTEGER positions for H and half-integer ones for E
#: (stepping.py:948 vs :1015). Swapped, it is a half-cell error in the absorber
#: profile — converged, smooth, and wrong — which is host mutation
#: ``swap_constitutive_sublattice``.
CONSTITUTIVE_HALF_INTEGER = {"H": False, "E": True}
CONSTITUTIVE_SIDES = ("H", "E")


def constitutive_names(side: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """(targets, auxiliaries, sources) for one side, in component order."""
    terms = CONSTITUTIVE_TERMS[side]
    return (tuple(t[0] for t in terms),
            tuple("f_w_" + t[0] for t in terms),
            tuple(t[1] for t in terms))


#: How the CONSTITUTIVE leg's arrays are drawn, and it is a COVERAGE AXIS for the
#: same measured reason it is one on the curl (:data:`PML_VALUE_CLASSES`).
#:
#: THIS LEG HAD ONLY ``uniform`` UNTIL 2026-08-15, and that made the two-policy
#: run the gate's own docstring prescribes — once under ``flush``, once under
#: ``keep``, "the two verdicts side by side are what closes §1.1" — agree BY
#: CONSTRUCTION. Measured over the four swept shapes and both sides before the
#: change: 1,907,550 float32 operands and intermediates, 0 subnormals, 0 signed
#: zeros, 0 zeros, smallest magnitude anywhere 4.49e-06 against a smallest normal
#: of 1.175e-38 — thirty-two orders of margin. Two runs of that establish nothing
#: about the policy and cost two device slots to say so.
#:
#: THE COEFFICIENTS STAY NORMAL in both classes, and so does inverse epsilon. They
#: are an absorber profile and a material, not field state; what the policy
#: decides the fate of is the recurrence, and driving ``kps`` into the band would
#: make ``kps * src`` an underflow to zero on both sides — a comparison of two
#: zeros, which is the vacuity this file counts rather than a measurement.
#:
#: THE TWO VERDICTS ARE REPORTED SEPARATELY and only the normal-number one gates,
#: exactly as on the curl and the dispersive legs: in the band it is the ARRAY
#: PATH, not the kernel, that leaves IEEE.
CONSTITUTIVE_VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")


def constitutive_field_hosts(shape: Tuple[int, int, int], side: str, rng,
                             value_class: str = "uniform") -> Dict[str, np.ndarray]:
    """The HOST arrays of one constitutive case, before they go to the device.

    Split out of :func:`make_constitutive_fields` for the reason the curl's
    :func:`pml_field_hosts` was: which value class an operand draw can and cannot
    reach is a property of THIS function and of no device, so it is asserted on a
    laptop (``test_pml_gate_harness.py``) rather than inferred from a device
    verdict that would read the same either way.

    The auxiliaries start NONZERO in both classes: a zero ``f_w`` makes
    ``kms * fw_previous`` exactly zero on the first launch whatever ``kms`` is, so
    a mis-indexed coefficient would only show from step two.

    THE THREE INVERSE-EPSILON VOLUMES ARE DRAWN INDEPENDENTLY, and that is
    load-bearing rather than incidental: binding one volume for all three
    components is defect 2 of the complex template (it passes ``fields.inv_eps``,
    the Ez view, fields.py:1259-1260), it is EXACTLY bit-identical under an
    isotropic epsilon, and the only thing that can expose it is three distinct
    volumes. Source mutation ``bind_Ez_inv_eps_for_all_three`` arms it and
    ``test_pml_gate_harness.py`` pins that the draw stays distinct.

    They are also drawn AWAY FROM 1.0 — a table of ones would make ``D * inv_eps``
    bit-equal to ``D`` and hide a kernel that dropped the multiply entirely
    (source mutation ``drop_inverse_epsilon``) — and they stay NORMAL in both
    value classes, so that ``D * inv_eps`` in the band is a subnormal times a
    normal rather than an underflow of two subnormals.
    """
    if value_class not in CONSTITUTIVE_VALUE_CLASSES:
        raise ValueError(f"value class {value_class!r} is not one of "
                         f"{CONSTITUTIVE_VALUE_CLASSES}")
    targets, auxiliaries, sources = constitutive_names(side)
    stateful = targets + auxiliaries + sources
    if value_class == "uniform":
        hosts = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                 for name in stateful}
    else:
        hosts = subnormal_band_hosts(stateful, shape, rng)
    if side == "E":
        for target in targets:
            hosts["inv_eps_" + target] = rng.uniform(
                0.2, 0.9, size=shape).astype(np.float32)
    return hosts


def make_constitutive_fields(shape: Tuple[int, int, int], side: str, rng,
                             value_class: str = "uniform") -> Dict[str, Any]:
    """Seeded float32 device arrays for one side's targets, auxiliaries and sources."""
    return {name: cp.asarray(np.ascontiguousarray(array))
            for name, array in constitutive_field_hosts(
                shape, side, rng, value_class).items()}


def synthetic_constitutive_coefficients(xp: Any, shape: Tuple[int, int, int],
                                        half_integer: bool) -> Dict[str, Any]:
    """Seeded random kps/kms per axis, different per Yee offset, never 1.0.

    Same argument as :func:`synthetic_coefficients`: a uniform table hides the
    coefficient indexing completely, so a swapped axis or the wrong sub-lattice
    would reproduce bit-for-bit.
    """
    rng = np.random.default_rng(SEED + (21 if half_integer else 22))
    coefficients: Dict[str, Any] = {}
    for axis, name in enumerate(AXIS_NAMES):
        n = shape[axis]
        for label in ("kps", "kms"):
            values = rng.uniform(0.5, 1.0, size=n).astype(np.float32)
            coefficients[f"{label}_{name}"] = xp.asarray(
                np.ascontiguousarray(values.reshape(broadcast_shape(axis, n))))
    return coefficients


def layer_constitutive_coefficients(pml, half_integer: bool) -> Dict[str, Any]:
    """The six broadcast-shaped kps/kms of one side, off a real layer."""
    suffix = "_h" if half_integer else ""
    return {f"{label}_{name}": getattr(pml, f"{label}_{name}{suffix}")
            for name in AXIS_NAMES for label in ("kps", "kms")}


def reference_constitutive_step(side: str, arrays: Dict[str, Any],
                                state: Dict[str, Any],
                                coefficients: Dict[str, Any],
                                grouping: str = "array_order") -> None:
    """One constitutive PML sub-step, in place on ``state``.

    ``stepping._apply_constitutive_pml`` (:2065), whose no-scratch branch is::

        fw_previous = fw.copy()
        fw[...] = source
        field += kps * fw
        field -= kms * fw_previous

    TWO SEPARATE ACCUMULATIONS, LEFT TO RIGHT. ``grouping="regrouped"`` is the
    alternative association ``field + (kps*fw - kms*fw_previous)``, carried so a
    failure separates associativity from everything else in one read — the same
    role ``kernel_order`` plays in the curl leg.
    """
    for target, source_name, axis in CONSTITUTIVE_TERMS[side]:
        axis_name = AXIS_NAMES[axis]
        kps = coefficients["kps_" + axis_name]
        kms = coefficients["kms_" + axis_name]
        if side == "E":
            source = arrays[source_name] * arrays["inv_eps_" + target]
        else:
            source = arrays[source_name]
        fw = state["f_w_" + target]
        field = state[target]
        fw_previous = fw.copy()
        fw[...] = source
        if grouping == "regrouped":
            field[...] = field + (kps * fw - kms * fw_previous)
        else:
            field += kps * fw
            field -= kms * fw_previous


def _swapped_kps_kms(coefficients: Dict[str, Any]) -> Dict[str, Any]:
    """Host mutation: ``(kap+sig)`` and ``(kap-sig)`` exchanged, per axis.

    Fed to the KERNEL only. It is the sign of the absorption reversed on the
    auxiliary term — a wrong answer with no error, and outside the layer (where
    kps == kms == 1) it is exactly bit-identical, so it can only be caught inside
    the absorber.
    """
    swapped: Dict[str, Any] = {}
    for key, value in coefficients.items():
        stem, axis = key.split("_", 1)
        swapped[("kms_" if stem == "kps" else "kps_") + axis] = value
    return swapped


def constitutive_case_key(coefficient_source, shape, boundaries, courant, side,
                          value_class="uniform") -> str:
    return (f"{coefficient_source}|{'x'.join(str(v) for v in shape)}|"
            f"{'/'.join(b[0] for b in boundaries)}|courant={courant!r}|update_{side}"
            f"|{value_class}")


def one_constitutive_case(adapter, shape, boundaries, courant, side,
                          coefficient_source, repo_root, compare_fw, comparator,
                          host_mutation: Optional[str],
                          value_class: str = "uniform") -> Dict[str, Any]:
    """One launch of one constitutive sub-step, against both reference groupings."""
    case: Dict[str, Any] = {
        "sub_step": "update_" + side,
        "side": side,
        "dtype": "float32",
        "shape": list(shape),
        "boundaries": list(boundaries),
        "courant": repr(courant),
        "courant_float32_exact": bool(float(np.float32(courant)) == float(courant)),
        "coefficient_source": coefficient_source,
        "host_mutation": host_mutation,
        "value_class": value_class,
        "compare_fw": compare_fw,
    }
    half_integer = CONSTITUTIVE_HALF_INTEGER[side]

    if coefficient_source == "synthetic":
        coefficients = synthetic_constitutive_coefficients(cp, shape, half_integer)
        kernel_coefficients = coefficients
        if host_mutation == "swap_constitutive_sublattice":
            # The half-cell error: the OTHER sub-lattice's tables, fed to the KERNEL
            # only. Swapping them under both legs is not a mutation — the reference
            # would step the same wrong coefficients and agree.
            kernel_coefficients = synthetic_constitutive_coefficients(
                cp, shape, not half_integer)
        elif host_mutation == "swap_kps_kms":
            kernel_coefficients = _swapped_kps_kms(coefficients)
    else:
        walled_invariant = [axis for axis in range(3)
                            if shape[axis] == 1 and boundaries[axis] == METALLIC]
        if walled_invariant:
            case["skipped"] = (
                f"axis {walled_invariant[0]} has one cell and a metallic wall; Grid "
                f"refuses that configuration, so no real layer exists for it")
            return case
        try:
            _, layer = real_pml_layer(cp, shape, boundaries, repo_root, courant=courant)
        except Exception as exc:  # noqa: BLE001 - the refusal IS the result
            case["skipped"] = f"{type(exc).__name__}: {exc}"[:400]
            return case
        coefficients = layer_constitutive_coefficients(layer, half_integer)
        if host_mutation == "swap_constitutive_sublattice":
            kernel_coefficients = layer_constitutive_coefficients(layer, not half_integer)
        elif host_mutation == "swap_kps_kms":
            kernel_coefficients = _swapped_kps_kms(coefficients)
        else:
            kernel_coefficients = coefficients

    rng = np.random.default_rng(SEED + 13)
    # Drawn on the HOST so the census is of the exact values that go to the
    # device, not of a second draw that would merely be distributed alike.
    hosts = constitutive_field_hosts(shape, side, rng, value_class)
    case["operand_census"] = operand_census(hosts)
    arrays = {name: cp.asarray(np.ascontiguousarray(array))
              for name, array in hosts.items()}
    targets, auxiliaries, _ = constitutive_names(side)

    references: Dict[str, Dict[str, Any]] = {}
    for grouping in ("array_order", "regrouped"):
        state = {name: arrays[name].copy() for name in targets + auxiliaries}
        reference_constitutive_step(side, arrays, state, coefficients, grouping)
        references[grouping] = state

    kernel_arrays = dict(arrays)
    for name in targets + auxiliaries:
        kernel_arrays[name] = arrays[name].copy()

    error: Optional[str] = None
    try:
        adapter.launch("update_" + side, kernel_arrays,
                       flatten_coefficients(kernel_coefficients),
                       [np.int32(0), np.int32(0), np.int32(0)], 0.0)
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:2000]
    case["launch_error"] = error
    if error is not None:
        for key in ("vs_array_order", "vs_regrouped"):
            case[key] = {"bit_identical": False, "differing_floats": -1,
                         "total_floats": 0}
        return case

    # LIVENESS, and the band class is what makes it live rather than theoretical:
    # seed every operand in the subnormal band, let both sides underflow to zero,
    # and a byte comparison reports perfect agreement about nothing. Counted from
    # the ARRAY-PATH reference (the oracle), never from the kernel, and
    # ``summarize_pml`` reports it per value class so a vacuous pass is visible in
    # the summary rather than only in the cases.
    case["reference_moved_targets"] = sum(
        int(bool((references["array_order"][name] != arrays[name]).any()))
        for name in targets)
    case["reference_moved_auxiliaries"] = sum(
        int(bool((references["array_order"][name] != arrays[name]).any()))
        for name in auxiliaries)

    for grouping, key in (("array_order", "vs_array_order"),
                          ("regrouped", "vs_regrouped")):
        case[key] = compare_pml_state(kernel_arrays, references[grouping],
                                      targets, auxiliaries, compare_fw, comparator)
    case["array_vs_regrouped_reference"] = compare_pml_state(
        references["array_order"], references["regrouped"],
        targets, auxiliaries, compare_fw, bit_compare)
    return case


def run_constitutive_bit_identity(adapter, results: Dict[str, Any], out_path: str,
                                  repo_root: str, coefficient_sources: Sequence[str],
                                  compare_fw: bool = True, comparator=bit_compare,
                                  guard_filter: Optional[str] = None,
                                  host_mutation: Optional[str] = None,
                                  label: str = "gate") -> Dict[str, Any]:
    """The constitutive gate: shapes x boundary sets x Courants x sides x tables x value classes.

    PASS is every NORMAL-NUMBER case bit-identical to ``array_order`` on BOTH the
    target and the auxiliary. §12.3 measured a dropped auxiliary store at 120/120
    UNCAUGHT on the curl when ``fu`` was not compared; ``f_w`` is this sub-step's
    ``fu`` and the identical blindness applies.

    THE VALUE CLASS IS A SWEPT AXIS (:data:`CONSTITUTIVE_VALUE_CLASSES`) and the
    band's count is reported beside the pass rather than folded into it. Without
    it this leg's operands provably could not reach the band, so the two-policy
    run the gate is specified to be cut under would return the same number twice.
    """
    cases: List[Dict[str, Any]] = []
    guards = [g for g in adapter.guard_sets
              if guard_filter is None or g[0] == guard_filter]
    combinations = [
        (source, shape, boundaries, courant, side, value_class)
        for source in coefficient_sources
        for shape in PML_SHAPES
        for boundaries in PML_BOUNDARY_SETS
        for courant in PML_DTDX
        for side in CONSTITUTIVE_SIDES
        for value_class in CONSTITUTIVE_VALUE_CLASSES
    ]
    total = len(guards) * len(combinations)
    index = 0
    for guard_label, guard, is_primary in guards:
        adapter.begin_guard(guard)
        try:
            for source, shape, boundaries, courant, side, value_class in combinations:
                index += 1
                started = time.time()
                case = one_constitutive_case(adapter, shape, boundaries, courant,
                                             side, source, repo_root, compare_fw,
                                             comparator, host_mutation, value_class)
                case.update({"guard_label": guard_label, "guard": list(guard),
                             "guard_is_primary": is_primary,
                             "seconds": round(time.time() - started, 3)})
                cases.append(case)
                key = constitutive_case_key(source, shape, boundaries, courant,
                                            side, value_class)
                if case.get("skipped"):
                    log(f"[const:{label}] case {index}/{total} guard={guard_label} "
                        f"{key}: SKIPPED ({case['skipped']})")
                else:
                    verdict = case["vs_array_order"]
                    census = case.get("operand_census", {})
                    log(f"[const:{label}] case {index}/{total} guard={guard_label} "
                        f"{key}: identical={verdict['bit_identical']} "
                        f"(differing={verdict['differing_floats']}/"
                        f"{verdict['total_floats']}, "
                        f"maxulp={verdict.get('max_ulp', 0)}) "
                        f"vs_regrouped={case['vs_regrouped']['bit_identical']} "
                        f"subnormals={census.get('subnormals', 0)} "
                        f"moved={case.get('reference_moved_targets')}/"
                        f"{case.get('reference_moved_auxiliaries')} "
                        f"({case['seconds']} s)")
                summary = summarize_pml(cases)
                summary["cases"] = cases
                results.setdefault("constitutive_bit_identity", {})[label] = summary
                save(results, out_path)
        finally:
            adapter.end_guard()
    summary = summarize_pml(cases)
    summary["cases"] = cases
    results.setdefault("constitutive_bit_identity", {})[label] = summary
    save(results, out_path)
    return summary


def run_constitutive_multi_step(adapter, results: Dict[str, Any], out_path: str,
                                compare_fw: bool = True, comparator=bit_compare,
                                host_mutation: Optional[str] = None,
                                label: str = "gate") -> Dict[str, Any]:
    """``CONSTITUTIVE_MULTI_STEP_COUNT`` consecutive sub-steps per shape, byte-compared at EVERY step.

    ``f_w`` IS STATE. A kernel that gets ``f`` right and ``f_w`` wrong is correct
    for exactly one launch and wrong forever after — measured on the curl at
    120/120 blind to a single-launch, target-only comparison.

    THE SOURCE MOVES BETWEEN SUB-STEPS, and it has to. This leg used to hold it
    fixed and claim in this docstring that ``f_w`` therefore "compounds its own
    history"; the recurrence writes ``fw[i] = src`` unconditionally, so a FIXED
    source drives ``f_w`` to a fixed point after launch one and 59 of 60 launches
    add no new auxiliary state. Measured over 8 launches before the change: ``f_w``
    changed against the previous step at step 1 only, and ``f_w == source``
    exactly at every step. In a real run the curl writes B/D before each
    constitutive call, so perturbing it here is the faithful thing as well as the
    load-bearing one — the same move the tranche's laptop sibling already makes
    (``test_constitutive_pml_real.py::_run``).
    """
    guard_label, guard, _ = adapter.guard_sets[0]
    boundaries = PML_BOUNDARY_SETS[3]
    runs: List[Dict[str, Any]] = []
    adapter.begin_guard(guard)
    try:
        for shape in PML_SHAPES:
            for side in CONSTITUTIVE_SIDES:
                run = one_constitutive_multi_step(adapter, shape, side, compare_fw,
                                                  comparator, host_mutation)
                run.update({"guard_label": guard_label, "shape": list(shape),
                            "sub_step": "update_" + side,
                            # NORMAL NUMBERS, like the curl's multi-step leg: the
                            # band is a single-launch axis, where the array path's
                            # own departure from IEEE is reported apart rather
                            # than compounded over 60 launches into a number
                            # nothing could attribute.
                            "value_class": "uniform",
                            "boundaries": list(boundaries)})
                runs.append(run)
                log(f"[const:{label}:multi] shape={shape} update_{side} "
                    f"{CONSTITUTIVE_MULTI_STEP_COUNT} steps: "
                    f"identical_steps={run['identical_steps']}"
                    f"/{CONSTITUTIVE_MULTI_STEP_COUNT} "
                    f"auxiliary_moved_after_step_1="
                    f"{run.get('auxiliary_moved_after_the_first_step')}"
                    + (f" first_divergence_at_step={run['first_divergence']}"
                       if run["first_divergence"] is not None else ""))
                results.setdefault("constitutive_multi_step", {})[label] = \
                    _constitutive_multi_step_summary(runs)
                save(results, out_path)
    finally:
        adapter.end_guard()
    summary = _constitutive_multi_step_summary(runs)
    results.setdefault("constitutive_multi_step", {})[label] = summary
    save(results, out_path)
    return summary


def _constitutive_multi_step_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The multi-step block, WITH the budget it was cut at.

    ``steps_budget`` is what ``run_pml_gate_mutations.py`` reads to fill
    ``multi_step_budget`` in the mutation summary; without it every constitutive
    leg recorded ``null`` and printed ``multi_identical=True@None``, against this
    track's own rule that a claim about N steps has to say which N it was.

    ``auxiliary_advanced`` is the non-vacuity floor for this leg specifically: a
    run whose ``f_w`` never changes after the first launch has 59 launches that
    add no state, and a multi-step claim over them is a claim about one step
    repeated.
    """
    return {
        "runs": runs,
        "steps_budget": CONSTITUTIVE_MULTI_STEP_COUNT,
        "all_steps_identical": all(
            r["identical_steps"] == CONSTITUTIVE_MULTI_STEP_COUNT for r in runs),
        "auxiliary_advanced_past_the_first_step": all(
            r.get("auxiliary_moved_after_the_first_step") for r in runs),
    }


def one_constitutive_multi_step(adapter, shape, side, compare_fw, comparator,
                                host_mutation) -> Dict[str, Any]:
    half_integer = CONSTITUTIVE_HALF_INTEGER[side]
    coefficients = synthetic_constitutive_coefficients(cp, shape, half_integer)
    # Every host mutation must be plumbed into EVERY leg it applies to. Measured:
    # `swap_kps_kms` was caught 0/120 in the single-launch leg and reported 8/8
    # IDENTICAL here, because this leg silently ran the unmutated tables — a leg
    # that reports a pass for a mutation it never applied is worse than no leg.
    if host_mutation == "swap_constitutive_sublattice":
        kernel_coefficients = synthetic_constitutive_coefficients(
            cp, shape, not half_integer)
    elif host_mutation == "swap_kps_kms":
        kernel_coefficients = _swapped_kps_kms(coefficients)
    else:
        kernel_coefficients = coefficients
    flat = flatten_coefficients(kernel_coefficients)
    rng = np.random.default_rng(SEED + 17)
    arrays = make_constitutive_fields(shape, side, rng)
    targets, auxiliaries, _ = constitutive_names(side)

    reference_state = {name: arrays[name].copy() for name in targets + auxiliaries}
    kernel_state = dict(arrays)
    for name in targets + auxiliaries:
        kernel_state[name] = arrays[name].copy()

    steps: List[Dict[str, Any]] = []
    identical_steps = 0
    first_divergence: Optional[int] = None
    codes = [np.int32(0), np.int32(0), np.int32(0)]
    source_names = constitutive_names(side)[2]
    auxiliary_advanced = False
    for step in range(1, CONSTITUTIVE_MULTI_STEP_COUNT + 1):
        # THE SOURCE MOVES. ``fw[i] = src`` is unconditional, so a fixed source
        # makes ``f_w`` a fixed point from launch one and every later launch adds
        # no state. ``kernel_state`` shares the source arrays with ``arrays`` (only
        # targets and auxiliaries were copied), so scaling in place perturbs the
        # reference and the kernel identically — which is what keeps this a
        # comparison rather than a divergence the harness manufactured. 0.97 is
        # the tranche's laptop sibling's factor; over 60 steps it stays normal.
        for name in source_names:
            arrays[name] *= np.float32(0.97)
        previous_auxiliaries = {name: reference_state[name].copy()
                                for name in auxiliaries}
        reference_constitutive_step(side, arrays, reference_state, coefficients,
                                    "array_order")
        if step > 1 and any(bool((reference_state[name] != previous_auxiliaries[name]).any())
                            for name in auxiliaries):
            auxiliary_advanced = True
        try:
            adapter.launch("update_" + side, kernel_state, flat, codes, 0.0)
            cp.cuda.runtime.deviceSynchronize()
        except Exception as exc:  # noqa: BLE001
            steps.append({"step": step,
                          "launch_error": f"{type(exc).__name__}: {exc}"[:1000]})
            first_divergence = first_divergence or step
            break
        verdict = compare_pml_state(kernel_state, reference_state, targets,
                                    auxiliaries, compare_fw, comparator)
        steps.append({"step": step, "bit_identical": verdict["bit_identical"],
                      "differing_floats": verdict["differing_floats"],
                      "max_ulp": verdict.get("max_ulp", 0),
                      "per_component": {k: v["bit_identical"]
                                        for k, v in verdict["per_component"].items()}})
        if verdict["bit_identical"]:
            identical_steps += 1
        elif first_divergence is None:
            first_divergence = step
    return {"steps": steps, "identical_steps": identical_steps,
            "steps_budget": CONSTITUTIVE_MULTI_STEP_COUNT,
            "auxiliary_moved_after_the_first_step": auxiliary_advanced,
            "first_divergence": first_divergence}


# ---------------------------------------------------------------------------
# The DISPERSIVE constitutive gate — update_E with poles registered
# ---------------------------------------------------------------------------
#
# A THIRD transcription, of ``stepping.update_E`` (:926) taking its dispersive
# branch, i.e. ``_apply_constitutive_pml`` fed by
# ``Fields.displacement_minus_polarization`` (fields.py:1079-1105) instead of by
# ``D`` directly. The contract, per component:
#
#     s    = ((D_c - P_0[c]) - P_1[c]) - ...    left to right, polarizations order
#     src  = s * inv_eps_c                      D-minus-P on the LEFT
#     prev = f_w_c ; f_w_c = src
#     E_c  = (E_c + kps_h[a]*src) - kms_h[a]*prev
#
# THREE AXES ARE IN THE PRODUCT HERE THAT ARE IN NO OTHER LEG, and each one is a
# defect family that only exists once a pole list enters the sub-step:
#
# * POLE COUNT PER COMPONENT, including a component with NONE. ``sigma_is_trivial``
#   (dispersion.py:600) drops any component whose sigma is identically zero, so a
#   real anisotropic material gives Ex, Ey and Ez different subsets — and a kernel
#   that carries one pole list for all three is wrong SILENTLY. ``(1, 0, 2)`` is in
#   the sweep for exactly that.
# * MULTI-POLE ARITHMETIC. ``D - (P0 + P1)`` is a different float32 number from
#   ``(D - P0) - P1``, so the pre-summed design — the one anyone writes from the
#   docstring sentence — is a real defect, and it is INVISIBLE at one pole. The
#   recon measured both mutations at 20/20 caught on two poles and 0/10 at one.
# * VALUE CLASS. Every other leg in this file seeds ``uniform(-1, 1)``, which
#   cannot produce a subnormal or a signed zero. ``D - sum P`` is a CANCELLATION
#   between two nearly equal volumes and is the only one of its kind in the step;
#   plan §16 traced the whole fast path's first divergence to a subnormal in
#   ``f_w_Ez``, which is the array THIS sub-step writes. So the leg carries a
#   second value class that constructs the cancellation deliberately, and the two
#   verdicts are reported SEPARATELY. A verdict from the uniform class alone is a
#   statement about normal numbers only and must be worded as one.

#: (target, source, own axis) — ``stepping.E_CONSTITUTIVE_TERMS`` (:227). Same
#: table as the non-dispersive E side; restated here so a reader of this leg does
#: not have to hold the other one in mind.
DISPERSIVE_TERMS = E_CONSTITUTIVE_TERMS

#: Pole counts per component, in ``E_CONSTITUTIVE_TERMS`` order. ``(1, 0, 2)`` is
#: the anisotropic-sigma case (Ey carries no P at all, so its arm must compile to
#: the non-dispersive body); ``(3, 2, 1)`` makes every component's list a different
#: length AND a different order; ``(6, 6, 6)`` is Ag, the corpus's only real
#: multi-pole material (1 Drude + 5 Lorentz), against the kernel's 8 slots.
DISPERSIVE_POLE_COUNTS: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 1), (2, 2, 2), (3, 2, 1), (1, 0, 2), (6, 6, 6))

#: How the seeded arrays are drawn. ``uniform`` is every other leg's class and
#: reaches NORMAL NUMBERS ONLY. ``cancellation`` constructs ``D`` and ``P`` so the
#: difference lands in and around the subnormal band, and injects exact zeros and
#: negative zeros — the value classes §16's divergence actually lives in.
DISPERSIVE_VALUE_CLASSES: Tuple[str, ...] = ("uniform", "cancellation")


def dispersive_pole_names(counts: Sequence[int]) -> Dict[str, Tuple[str, ...]]:
    """``{'Ex': ('P0_Ex', 'P1_Ex'), ...}`` — the ordered pole array names per component."""
    return {target: tuple(f"P{n}_{target}" for n in range(int(count)))
            for (target, _, _), count in zip(DISPERSIVE_TERMS, counts)}


def make_dispersive_fields(shape: Tuple[int, int, int], counts: Sequence[int],
                           rng, value_class: str = "uniform") -> Dict[str, Any]:
    """Seeded float32 device arrays for the dispersive E sub-step.

    ``uniform`` is the standard class: everything in [-1, 1), auxiliaries NONZERO
    (a zero ``f_w`` makes ``kms * fw_previous`` zero whatever ``kms`` is, hiding a
    coefficient-indexing error on the first launch) and inverse epsilon drawn away
    from 1.0 (a table of ones makes ``D*inv_eps`` bit-equal to ``D`` and hides a
    kernel that dropped the multiply).

    ``cancellation`` is the class this sub-step needs and no other leg has. Each
    pole is built as ``D/npoles`` MINUS a tiny perturbation whose magnitude sweeps
    1e-45 to 1e-30, so ``D - sum P`` lands in and around the subnormal band; on top
    of that, one cell in sixteen is forced to an exact zero, a negative zero, the
    smallest normal, or the largest subnormal, in D, in the poles and in ``f_w``.
    THE POINT IS NOT REALISM — it is that the gate's verdict otherwise says nothing
    about the band plan §16 measured the divergence in.
    """
    targets = tuple(t[0] for t in DISPERSIVE_TERMS)
    sources = tuple(t[1] for t in DISPERSIVE_TERMS)
    poles = dispersive_pole_names(counts)

    def draw() -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)

    host: Dict[str, np.ndarray] = {}
    for name in targets + tuple("f_w_" + t for t in targets) + sources:
        host[name] = draw()
    for target in targets:
        host["inv_eps_" + target] = rng.uniform(0.2, 0.9, size=shape).astype(np.float32)
    for target, source_name, _ in DISPERSIVE_TERMS:
        names = poles[target]
        if not names:
            continue
        if value_class == "uniform":
            for name in names:
                host[name] = draw()
            continue
        # The cancellation class. Each pole carries an equal share of D, so the
        # sequential subtraction walks the sum down to ~0 and the LAST subtraction
        # is the one that cancels; the perturbation decides where it lands.
        share = (host[source_name] / np.float32(len(names))).astype(np.float32)
        exponents = rng.uniform(-45.0, -30.0, size=shape)
        signs = np.where(rng.integers(0, 2, size=shape) == 0, -1.0, 1.0)
        perturbation = (signs * np.power(10.0, exponents)).astype(np.float32)
        for index, name in enumerate(names):
            host[name] = share.copy()
        host[names[-1]] = (share - perturbation).astype(np.float32)
    if value_class == "cancellation":
        # Exact zeros, negative zeros and the two ends of the subnormal boundary,
        # sprinkled into every array the sub-step reads or writes. CuPy's own
        # flush-to-zero is a measurement hazard here (plan §16.5), which is why the
        # comparison stays bytewise and never asks for a magnitude.
        specials = np.array([0.0, -0.0, np.float32(1.1754944e-38),
                             np.float32(1.1754942e-38), np.float32(1e-45),
                             np.float32(-1e-45)], dtype=np.float32)
        for name, array in host.items():
            if name.startswith("inv_eps_"):
                continue  # a zero inverse epsilon is not a configuration the engine builds
            picks = rng.integers(0, 16, size=shape) == 0
            choice = specials[rng.integers(0, specials.size, size=shape)]
            host[name] = np.where(picks, choice, array).astype(np.float32)
    return {name: cp.asarray(np.ascontiguousarray(array))
            for name, array in host.items()}


def reference_dispersive_constitutive_step(arrays: Dict[str, Any],
                                           state: Dict[str, Any],
                                           coefficients: Dict[str, Any],
                                           poles: Dict[str, Sequence[str]],
                                           grouping: str = "array_order",
                                           pole_grouping: str = "sequential") -> None:
    """One dispersive electric constitutive sub-step, in place on ``state``.

    ``stepping.update_E``'s dispersive branch, transcribed::

        source       = fields.displacement_minus_polarization(c)   # :981
        constitutive = source * fields.inverse_epsilon_for(c)      # :982
        fw_previous  = fw.copy(); fw[...] = constitutive
        field += kps * fw ; field -= kms * fw_previous              # :2065-2097

    ``grouping="regrouped"`` is the alternative association of the accumulation,
    exactly as in :func:`reference_constitutive_step`. ``pole_grouping`` is this
    leg's own second control and has no counterpart anywhere else:

    * ``"summed"``     — ``D - (P0 + P1 + ...)``, the pre-accumulated design;
    * ``"reversed"``   — the pole list walked backwards.

    Both are bit-identical to the array order at ONE pole and differ at two or
    more, which is the measured signature (recon: 20/20 and 0/10 respectively) and
    is what makes the multi-pole rows of the sweep load-bearing rather than
    decorative.

    ``displacement_minus_polarization`` returns the D array ITSELF, unaliased and
    uncopied, when nothing drives the component (fields.py:1096-1097). That is
    reproduced here rather than smoothed over: a component with no pole must reduce
    to the non-dispersive sub-step exactly.
    """
    for target, source_name, axis in DISPERSIVE_TERMS:
        axis_name = AXIS_NAMES[axis]
        kps = coefficients["kps_" + axis_name]
        kms = coefficients["kms_" + axis_name]
        contributors = list(poles.get(target, ()))
        if pole_grouping == "reversed":
            contributors = list(reversed(contributors))
        if not contributors:
            difference = arrays[source_name]
        elif pole_grouping == "summed":
            total = arrays[contributors[0]].copy()
            for name in contributors[1:]:
                total = total + arrays[name]
            difference = arrays[source_name] - total
        else:
            difference = arrays[source_name].copy()
            for name in contributors:
                difference = difference - arrays[name]
        source = difference * arrays["inv_eps_" + target]
        fw = state["f_w_" + target]
        field = state[target]
        fw_previous = fw.copy()
        fw[...] = source
        if grouping == "regrouped":
            field[...] = field + (kps * fw - kms * fw_previous)
        else:
            field += kps * fw
            field -= kms * fw_previous


def dispersive_case_key(coefficient_source, shape, boundaries, courant, counts,
                        value_class) -> str:
    return (f"{coefficient_source}|{'x'.join(str(v) for v in shape)}|"
            f"{'/'.join(b[0] for b in boundaries)}|courant={courant!r}|"
            f"poles={'-'.join(str(c) for c in counts)}|{value_class}")


def _dispersive_bindings(arrays: Dict[str, Any], counts: Sequence[int],
                         host_mutation: Optional[str]) -> Dict[str, Any]:
    """The ``poles`` binding handed to the KERNEL, host mutations included.

    ``reverse_pole_order`` is a HOST mutation and not a source one, deliberately:
    the kernel's slots are filled by the plan, so reversing them there is the
    faithful spelling of "the plan resolved the order backwards" — which is the
    defect that actually threatens this kernel (``fields.polarizations`` order is
    what ``displacement_minus_polarization`` walks). Applied to the kernel only;
    reversing it under the reference too would agree.
    """
    names = dispersive_pole_names(counts)
    ordered = {target: list(group) for target, group in names.items()}
    if host_mutation == "reverse_pole_order":
        ordered = {target: list(reversed(group)) for target, group in ordered.items()}
    return {target: [arrays[name] for name in group]
            for target, group in ordered.items()}


def one_dispersive_case(adapter, shape, boundaries, courant, counts,
                        coefficient_source, value_class, repo_root, compare_fw,
                        comparator, host_mutation: Optional[str]) -> Dict[str, Any]:
    """One launch of the dispersive ``update_E``, against every reference control."""
    case: Dict[str, Any] = {
        "sub_step": "update_E_dispersive",
        "dtype": "float32",
        "shape": list(shape),
        "boundaries": list(boundaries),
        "courant": repr(courant),
        "courant_float32_exact": bool(float(np.float32(courant)) == float(courant)),
        "coefficient_source": coefficient_source,
        "pole_counts": list(counts),
        "value_class": value_class,
        "host_mutation": host_mutation,
        "compare_fw": compare_fw,
    }
    if coefficient_source == "synthetic":
        coefficients = synthetic_constitutive_coefficients(cp, shape, True)
        if host_mutation == "swap_constitutive_sublattice":
            kernel_coefficients = synthetic_constitutive_coefficients(cp, shape, False)
        elif host_mutation == "swap_kps_kms":
            kernel_coefficients = _swapped_kps_kms(coefficients)
        else:
            kernel_coefficients = coefficients
    else:
        walled_invariant = [axis for axis in range(3)
                            if shape[axis] == 1 and boundaries[axis] == METALLIC]
        if walled_invariant:
            case["skipped"] = (
                f"axis {walled_invariant[0]} has one cell and a metallic wall; Grid "
                f"refuses that configuration, so no real layer exists for it")
            return case
        try:
            _, layer = real_pml_layer(cp, shape, boundaries, repo_root, courant=courant)
        except Exception as exc:  # noqa: BLE001 - the refusal IS the result
            case["skipped"] = f"{type(exc).__name__}: {exc}"[:400]
            return case
        coefficients = layer_constitutive_coefficients(layer, True)
        if host_mutation == "swap_constitutive_sublattice":
            kernel_coefficients = layer_constitutive_coefficients(layer, False)
        elif host_mutation == "swap_kps_kms":
            kernel_coefficients = _swapped_kps_kms(coefficients)
        else:
            kernel_coefficients = coefficients

    rng = np.random.default_rng(SEED + 31)
    arrays = make_dispersive_fields(shape, counts, rng, value_class)
    names = dispersive_pole_names(counts)
    targets = tuple(t[0] for t in DISPERSIVE_TERMS)
    auxiliaries = tuple("f_w_" + t for t in targets)

    references: Dict[str, Dict[str, Any]] = {}
    for label, grouping, pole_grouping in (
            ("array_order", "array_order", "sequential"),
            ("regrouped", "regrouped", "sequential"),
            ("summed_poles", "array_order", "summed"),
            ("reversed_poles", "array_order", "reversed")):
        state = {name: arrays[name].copy() for name in targets + auxiliaries}
        reference_dispersive_constitutive_step(arrays, state, coefficients, names,
                                               grouping, pole_grouping)
        references[label] = state

    kernel_arrays = dict(arrays)
    for name in targets + auxiliaries:
        kernel_arrays[name] = arrays[name].copy()

    error: Optional[str] = None
    try:
        adapter.launch("update_E_dispersive", kernel_arrays,
                       flatten_coefficients(kernel_coefficients),
                       [np.int32(0), np.int32(0), np.int32(0)], 0.0,
                       extra={"poles": _dispersive_bindings(kernel_arrays, counts,
                                                            host_mutation)})
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:2000]
    case["launch_error"] = error
    if error is not None:
        for key in ("vs_array_order", "vs_regrouped", "vs_summed_poles",
                    "vs_reversed_poles"):
            case[key] = {"bit_identical": False, "differing_floats": -1,
                         "total_floats": 0}
        return case

    for label in ("array_order", "regrouped", "summed_poles", "reversed_poles"):
        case["vs_" + label] = compare_pml_state(
            kernel_arrays, references[label], targets, auxiliaries, compare_fw,
            comparator)
    # The three reference controls against the array order itself. At ONE pole the
    # pole controls MUST coincide with it — that is not a harness failure, it is
    # the measured signature of a multi-pole-only defect — so the artifact records
    # which rows could discriminate rather than leaving a reader to infer it.
    case["reference_controls"] = {
        label: compare_pml_state(references["array_order"], references[label],
                                 targets, auxiliaries, compare_fw, bit_compare)
        for label in ("regrouped", "summed_poles", "reversed_poles")}
    case["multi_pole"] = bool(max(counts) > 1)
    return case


def summarize_dispersive(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pass counts per guard AND per value class, never merged into one number.

    The split is the point. ``uniform`` is a verdict about NORMAL NUMBERS and
    ``cancellation`` is a verdict about the band plan §16 measured the fast path's
    divergence in; a single N/N over both would let one hide inside the other, and
    the word "bit-identical" would then be doing work it has not earned.
    """
    summary = summarize_pml(cases)
    per_class: Dict[str, Dict[str, int]] = {}
    for case in cases:
        if case.get("skipped") or not case.get("guard_is_primary"):
            continue
        bucket = per_class.setdefault(case["value_class"],
                                      {"ran": 0, "identical": 0})
        bucket["ran"] += 1
        bucket["identical"] += int(case["vs_array_order"]["bit_identical"])
    summary["per_value_class"] = per_class
    summary["normal_numbers_pass"] = bool(
        per_class.get("uniform", {}).get("ran")
        and per_class["uniform"]["ran"] == per_class["uniform"]["identical"])
    return summary


def run_dispersive_constitutive_bit_identity(
        adapter, results: Dict[str, Any], out_path: str, repo_root: str,
        coefficient_sources: Sequence[str], compare_fw: bool = True,
        comparator=bit_compare, guard_filter: Optional[str] = None,
        host_mutation: Optional[str] = None, label: str = "gate") -> Dict[str, Any]:
    """The dispersive gate: shapes x boundary sets x Courants x pole sets x value classes.

    PASS is every primary-guard case bit-identical to ``array_order`` on BOTH the
    target and the auxiliary — and the auxiliary half is not a formality here. §12.3
    measured a dropped auxiliary store at 120/120 UNCAUGHT on the curl when ``fu``
    was not compared, and ``f_w_E`` is worse than the curl's ``fu``: it is what
    ``Fields.drive_field`` hands ``update_P``, so a wrong ``f_w`` is a wrong
    polarization from the next sub-step onwards while ``E`` still reads correct.

    The NON-POWER-OF-TWO COURANT is mandatory and reaches the coefficients through
    the ``dt/2`` baked into the stored sigma, exactly as it does on the
    non-dispersive side.
    """
    cases: List[Dict[str, Any]] = []
    guards = [g for g in adapter.guard_sets
              if guard_filter is None or g[0] == guard_filter]
    combinations = [
        (source, shape, boundaries, courant, counts, value_class)
        for source in coefficient_sources
        for shape in PML_SHAPES
        for boundaries in PML_BOUNDARY_SETS
        for courant in PML_DTDX
        for counts in DISPERSIVE_POLE_COUNTS
        for value_class in DISPERSIVE_VALUE_CLASSES
    ]
    total = len(guards) * len(combinations)
    index = 0
    for guard_label, guard, is_primary in guards:
        adapter.begin_guard(guard)
        try:
            for source, shape, boundaries, courant, counts, value_class in combinations:
                index += 1
                started = time.time()
                case = one_dispersive_case(adapter, shape, boundaries, courant,
                                           counts, source, value_class, repo_root,
                                           compare_fw, comparator, host_mutation)
                case.update({"guard_label": guard_label, "guard": list(guard),
                             "guard_is_primary": is_primary,
                             "seconds": round(time.time() - started, 3)})
                cases.append(case)
                key = dispersive_case_key(source, shape, boundaries, courant,
                                          counts, value_class)
                if case.get("skipped"):
                    log(f"[disp:{label}] case {index}/{total} guard={guard_label} "
                        f"{key}: SKIPPED ({case['skipped']})")
                else:
                    verdict = case["vs_array_order"]
                    log(f"[disp:{label}] case {index}/{total} guard={guard_label} "
                        f"{key}: identical={verdict['bit_identical']} "
                        f"(differing={verdict['differing_floats']}/"
                        f"{verdict['total_floats']}, "
                        f"maxulp={verdict.get('max_ulp', 0)}) "
                        f"vs_summed={case['vs_summed_poles']['bit_identical']} "
                        f"vs_reversed={case['vs_reversed_poles']['bit_identical']} "
                        f"({case['seconds']} s)")
                results.setdefault("dispersive_bit_identity", {})[label] = dict(
                    summarize_dispersive(cases), cases=cases)
                save(results, out_path)
        finally:
            adapter.end_guard()
    summary = dict(summarize_dispersive(cases), cases=cases)
    results.setdefault("dispersive_bit_identity", {})[label] = summary
    save(results, out_path)
    return summary


def run_dispersive_constitutive_multi_step(adapter, results: Dict[str, Any],
                                           out_path: str, compare_fw: bool = True,
                                           comparator=bit_compare,
                                           host_mutation: Optional[str] = None,
                                           steps: int = MULTI_STEP_COUNT,
                                           label: str = "gate") -> Dict[str, Any]:
    """Consecutive dispersive sub-steps per shape, byte-compared at EVERY step.

    ``f_w`` IS STATE and it is this sub-step's output to ``update_P``. A kernel that
    gets ``E`` right and ``f_w`` wrong is correct for exactly one launch and wrong
    forever after. Sources and poles are held fixed, so ``f_w`` compounds its own
    history and a divergence grows rather than cancelling.

    THE STEP BUDGET IS A COVERAGE AXIS, not a runtime knob: 8, 10 and 60 steps all
    passed for the existing kernels and 40 and 67 did not (plan §14.5, §16). It is
    therefore an argument, and the artifact records it.
    """
    guard_label, guard, _ = adapter.guard_sets[0]
    boundaries = PML_BOUNDARY_SETS[3]
    runs: List[Dict[str, Any]] = []
    adapter.begin_guard(guard)
    try:
        for shape in PML_SHAPES:
            for counts in DISPERSIVE_POLE_COUNTS:
                for value_class in DISPERSIVE_VALUE_CLASSES:
                    run = one_dispersive_multi_step(adapter, shape, counts,
                                                    value_class, compare_fw,
                                                    comparator, host_mutation, steps)
                    run.update({"guard_label": guard_label, "shape": list(shape),
                                "sub_step": "update_E_dispersive",
                                "pole_counts": list(counts),
                                "value_class": value_class,
                                "steps_requested": int(steps),
                                "boundaries": list(boundaries)})
                    runs.append(run)
                    log(f"[disp:{label}:multi] shape={shape} poles={counts} "
                        f"{value_class} {steps} steps: "
                        f"identical_steps={run['identical_steps']}/{steps}"
                        + (f" first_divergence_at_step={run['first_divergence']}"
                           if run["first_divergence"] is not None else ""))
                    results.setdefault("dispersive_multi_step", {})[label] = {
                        "runs": runs,
                        "all_steps_identical": all(
                            r["identical_steps"] == steps for r in runs),
                        "normal_numbers_all_steps_identical": all(
                            r["identical_steps"] == steps for r in runs
                            if r["value_class"] == "uniform")}
                    save(results, out_path)
    finally:
        adapter.end_guard()
    summary = {"runs": runs,
               "all_steps_identical": all(r["identical_steps"] == steps for r in runs),
               "normal_numbers_all_steps_identical": all(
                   r["identical_steps"] == steps for r in runs
                   if r["value_class"] == "uniform")}
    results.setdefault("dispersive_multi_step", {})[label] = summary
    save(results, out_path)
    return summary


def one_dispersive_multi_step(adapter, shape, counts, value_class, compare_fw,
                              comparator, host_mutation, steps) -> Dict[str, Any]:
    coefficients = synthetic_constitutive_coefficients(cp, shape, True)
    # Every host mutation must be plumbed into EVERY leg it applies to. Measured
    # twice in this project: a leg that reports a pass for a mutation it never
    # applied is worse than no leg (§13.4, §15.8).
    if host_mutation == "swap_constitutive_sublattice":
        kernel_coefficients = synthetic_constitutive_coefficients(cp, shape, False)
    elif host_mutation == "swap_kps_kms":
        kernel_coefficients = _swapped_kps_kms(coefficients)
    else:
        kernel_coefficients = coefficients
    flat = flatten_coefficients(kernel_coefficients)
    rng = np.random.default_rng(SEED + 37)
    arrays = make_dispersive_fields(shape, counts, rng, value_class)
    names = dispersive_pole_names(counts)
    targets = tuple(t[0] for t in DISPERSIVE_TERMS)
    auxiliaries = tuple("f_w_" + t for t in targets)

    reference_state = {name: arrays[name].copy() for name in targets + auxiliaries}
    kernel_state = dict(arrays)
    for name in targets + auxiliaries:
        kernel_state[name] = arrays[name].copy()
    bindings = _dispersive_bindings(kernel_state, counts, host_mutation)

    step_records: List[Dict[str, Any]] = []
    identical_steps = 0
    first_divergence: Optional[int] = None
    codes = [np.int32(0), np.int32(0), np.int32(0)]
    for step in range(1, int(steps) + 1):
        reference_dispersive_constitutive_step(arrays, reference_state, coefficients,
                                               names, "array_order", "sequential")
        try:
            adapter.launch("update_E_dispersive", kernel_state, flat, codes, 0.0,
                           extra={"poles": bindings})
            cp.cuda.runtime.deviceSynchronize()
        except Exception as exc:  # noqa: BLE001
            step_records.append({"step": step,
                                 "launch_error": f"{type(exc).__name__}: {exc}"[:1000]})
            first_divergence = first_divergence or step
            break
        verdict = compare_pml_state(kernel_state, reference_state, targets,
                                    auxiliaries, compare_fw, comparator)
        step_records.append({"step": step, "bit_identical": verdict["bit_identical"],
                             "differing_floats": verdict["differing_floats"],
                             "max_ulp": verdict.get("max_ulp", 0),
                             "per_component": {k: v["bit_identical"] for k, v
                                               in verdict["per_component"].items()}})
        if verdict["bit_identical"]:
            identical_steps += 1
        elif first_divergence is None:
            first_divergence = step
    return {"steps": step_records, "identical_steps": identical_steps,
            "first_divergence": first_divergence}


# ---------------------------------------------------------------------------
# The CROSS-SUB-STEP fused pair — step_B + zero_metal_B + update_H in ONE launch
# ---------------------------------------------------------------------------
#
# Every leg above times/compares ONE sub-step. This one compares a kernel that
# performs TWO, and the extra thing it has to certify is the SEAM: the value the
# curl half leaves in a register is the value the constitutive half must read, and
# in the array path that value made a round trip through float32 memory with
# ``zero_metal_B`` writing over one plane of it on the way.
#
# THREE COMPARISONS PER CASE, not two, and the third is the one that makes the
# other two readable:
#
#   fused vs two kernels  — does removing the round trip change any bit?
#   fused vs array path   — the verdict that matters (the reference transcription)
#   two kernels vs array  — the control: the unfused composition still holds
#
# The auxiliaries are compared throughout. §12.3 and §13.3 both measured a real
# defect passing 120/120 when only the primary was compared, one sub-step apart,
# and a fused kernel writes FOUR auxiliaries (fu_B* and f_w_H*) rather than three.

#: Every array the pair writes. Both primaries, both auxiliaries.
FUSED_PAIR_TARGETS = ("Bx", "By", "Bz")
FUSED_PAIR_AUX = ("fu_Bx", "fu_By", "fu_Bz")
FUSED_PAIR_H_TARGETS = ("Hx", "Hy", "Hz")
FUSED_PAIR_H_AUX = ("f_w_Hx", "f_w_Hy", "f_w_Hz")
FUSED_PAIR_SOURCES = ("Ex", "Ey", "Ez")
FUSED_PAIR_STATE = (FUSED_PAIR_TARGETS + FUSED_PAIR_AUX
                    + FUSED_PAIR_H_TARGETS + FUSED_PAIR_H_AUX)


def make_fused_pair_fields(shape: Tuple[int, int, int], rng) -> Dict[str, Any]:
    """Seeded float32 device arrays for everything the fused pair touches.

    Both auxiliary families start NONZERO for the reason the curl and constitutive
    legs give: a zero ``fu`` or ``f_w`` makes its coefficient product exactly zero
    on the first launch whatever the coefficient is, which would hide a mis-indexed
    table until step two.
    """
    names = FUSED_PAIR_STATE + FUSED_PAIR_SOURCES
    return {name: cp.asarray(np.ascontiguousarray(
        rng.uniform(-1.0, 1.0, size=shape).astype(np.float32))) for name in names}


def zero_metal_flags(boundaries: Sequence[str]) -> Tuple[bool, bool, bool]:
    """``ZM_X/Y/Z`` for a harness configuration: a declared wall is a wall.

    On the engine route ``coverage.zero_metal_axes`` asks the Grid; here the
    harness declared the boundaries itself, so the same question has the same
    answer without a Grid. Fed to BOTH legs — this is the configuration, not a
    mutation; ``drop_zero_metal_inline`` is the mutation, and it is applied to the
    kernel only.
    """
    return tuple(kind == METALLIC for kind in boundaries)  # type: ignore[return-value]


def reference_zero_metal_B(state: Dict[str, Any],
                           zero_metal: Sequence[bool]) -> None:
    """``stepping.zero_metal_B``, transcribed: stored cell 0 of the wall plane.

    Component ``d`` is the B component whose Yee shift on axis ``d`` is 0 — ``Bx``
    on an x wall, ``By`` on y, ``Bz`` on z (``IYEE_SHIFTS``, fields.py:214-219) —
    which is the one-to-one mapping that lets the kernel spell the wipe as one
    constexpr per axis.
    """
    for axis, (component, flag) in enumerate(zip(FUSED_PAIR_TARGETS, zero_metal)):
        if flag:
            state[component][_face(axis, 0)] = 0


def reference_fused_pair_step(xp: Any, arrays: Dict[str, Any], state: Dict[str, Any],
                              curl_coefficients: Dict[str, Any],
                              h_coefficients: Dict[str, Any], dtdx: Any,
                              boundaries: Sequence[str],
                              zero_metal: Sequence[bool],
                              grouping: str = "array_order") -> None:
    """The three array-path passes the one kernel replaces, in the driver's order.

    ``step_B`` (driver.py:3163), ``zero_metal_B`` (:3167), ``update_H`` (:3169).
    Nothing else runs between them in a covered configuration: the magnetic source
    inject is refused by the predicate, and ``fill_symmetry_bc_B`` /
    ``fill_folded_far_ghosts_B`` need a fold, which the curl half refuses.

    ``grouping="regrouped"`` re-associates BOTH halves — the curl's stencil (its
    own label for that is ``kernel_order``) and the constitutive side's two
    accumulations. It is the associativity control, the role ``kernel_order``
    plays in the curl leg: a fused kernel that matched THIS reference would have
    re-associated something across the seam.
    """
    reference_pml_step(
        xp, {name: arrays[name] for name in FUSED_PAIR_SOURCES},
        {name: state[name] for name in FUSED_PAIR_TARGETS},
        {name: state[name] for name in FUSED_PAIR_AUX},
        curl_coefficients, dtdx, "step_B", boundaries,
        "kernel_order" if grouping == "regrouped" else "array_order")
    reference_zero_metal_B(state, zero_metal)
    reference_constitutive_step(
        "H", {name: state[name] for name in FUSED_PAIR_TARGETS}, state,
        h_coefficients, "regrouped" if grouping == "regrouped" else "array_order")


def two_kernel_pair_step(adapter, arrays: Dict[str, Any], curl_flat, h_flat,
                         codes, dtdx: float, zero_metal: Sequence[bool]) -> None:
    """The CONTROL: the composed-but-unfused path, one launch per sub-step.

    This is what the fusion is measured against and it must stay runnable — which
    is why ``plan_step``'s ``fuse`` defaults to False and why the two separate
    kernels were not deleted. The host performs ``zero_metal_B`` between them,
    exactly as ``FdtdDriver.step`` does.
    """
    adapter.launch("step_B", arrays, curl_flat, codes, dtdx)
    reference_zero_metal_B(arrays, zero_metal)
    adapter.launch("update_H", arrays, h_flat, codes, dtdx)


def fused_pair_case_key(coefficient_source, shape, boundaries, dtdx) -> str:
    return (f"{coefficient_source}|{'x'.join(str(v) for v in shape)}|"
            f"{'/'.join(b[0] for b in boundaries)}|dtdx={dtdx!r}|fused_pair_B")


def one_fused_pair_case(adapter, shape, boundaries, dtdx, coefficient_source,
                        repo_root, compare_aux, comparator,
                        host_mutation: Optional[str],
                        num_warps: Optional[int] = None) -> Dict[str, Any]:
    """One launch of the fused pair against the array path AND against two kernels."""
    case: Dict[str, Any] = {
        "sub_step": "fused_pair_B",
        "dtype": "float32",
        "shape": list(shape),
        "boundaries": list(boundaries),
        "dtdx": repr(dtdx),
        "dtdx_float32_exact": bool(float(np.float32(dtdx)) == float(dtdx)),
        "coefficient_source": coefficient_source,
        "host_mutation": host_mutation,
        "compare_aux": compare_aux,
        "num_warps": num_warps,
    }
    zero_metal = zero_metal_flags(boundaries)
    case["zero_metal"] = [bool(f) for f in zero_metal]

    if coefficient_source == "synthetic":
        curl_coefficients = synthetic_coefficients(cp, shape, True)
        h_coefficients = synthetic_constitutive_coefficients(cp, shape, False)
        kernel_curl = curl_coefficients
        kernel_h = h_coefficients
        if host_mutation == "integer_coefficients_on_B":
            kernel_curl = synthetic_coefficients(cp, shape, False)
        if host_mutation == "swap_constitutive_sublattice":
            kernel_h = synthetic_constitutive_coefficients(cp, shape, True)
    else:
        walled_invariant = [axis for axis in range(3)
                            if shape[axis] == 1 and boundaries[axis] == METALLIC]
        if walled_invariant:
            case["skipped"] = (
                f"axis {walled_invariant[0]} has one cell and a metallic wall; Grid "
                f"refuses that configuration, so no real layer exists for it")
            return case
        try:
            _, layer = real_pml_layer(cp, shape, boundaries, repo_root)
        except Exception as exc:  # noqa: BLE001 - the refusal IS the result
            case["skipped"] = f"{type(exc).__name__}: {exc}"[:400]
            return case
        curl_coefficients = layer_coefficients(layer, True)
        h_coefficients = layer_constitutive_coefficients(layer, False)
        kernel_curl = (layer_coefficients(layer, False)
                       if host_mutation == "integer_coefficients_on_B"
                       else curl_coefficients)
        kernel_h = (layer_constitutive_coefficients(layer, True)
                    if host_mutation == "swap_constitutive_sublattice"
                    else h_coefficients)
    if host_mutation == "swap_kps_kms":
        kernel_h = _swapped_kps_kms(kernel_h)

    rng = np.random.default_rng(SEED + 31)
    arrays = make_fused_pair_fields(shape, rng)

    references: Dict[str, Dict[str, Any]] = {}
    for grouping in ("array_order", "regrouped"):
        state = {name: arrays[name].copy() for name in FUSED_PAIR_STATE}
        reference_fused_pair_step(cp, arrays, state, curl_coefficients,
                                  h_coefficients, np.float32(dtdx), boundaries,
                                  zero_metal, grouping)
        references[grouping] = state

    codes = [0 if kind == PERIODIC else 1 for kind in boundaries]
    if host_mutation == "metallic_as_periodic":
        codes = [0, 0, 0]
    codes = [np.int32(c) for c in codes]

    fused_arrays = dict(arrays)
    unfused_arrays = dict(arrays)
    for name in FUSED_PAIR_STATE:
        fused_arrays[name] = arrays[name].copy()
        unfused_arrays[name] = arrays[name].copy()

    curl_flat = flatten_coefficients(kernel_curl)
    h_flat = flatten_coefficients(kernel_h)
    error: Optional[str] = None
    try:
        adapter.launch("fused_pair_B", fused_arrays, curl_flat, codes, dtdx,
                       extra={"h_flat": h_flat, "zero_metal": zero_metal,
                              "num_warps": num_warps})
        two_kernel_pair_step(adapter, unfused_arrays, curl_flat, h_flat, codes,
                             dtdx, zero_metal)
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:2000]
    case["launch_error"] = error
    if error is not None:
        for key in ("vs_array_order", "vs_two_kernels", "two_kernels_vs_array_order"):
            case[key] = {"bit_identical": False, "differing_floats": -1,
                         "total_floats": 0}
        return case

    names = (FUSED_PAIR_TARGETS + FUSED_PAIR_H_TARGETS
             + (FUSED_PAIR_AUX + FUSED_PAIR_H_AUX if compare_aux else ()))
    case["vs_array_order"] = combine(
        {name: comparator(fused_arrays[name], references["array_order"][name])
         for name in names})
    case["vs_two_kernels"] = combine(
        {name: comparator(fused_arrays[name], unfused_arrays[name])
         for name in names})
    case["two_kernels_vs_array_order"] = combine(
        {name: comparator(unfused_arrays[name], references["array_order"][name])
         for name in names})
    # The regrouped reference is the associativity control, as in the curl leg:
    # a fused kernel that matched THIS one would have re-associated the seam.
    case["vs_regrouped_reference"] = combine(
        {name: bit_compare(fused_arrays[name], references["regrouped"][name])
         for name in names})
    return case


def run_fused_pair_bit_identity(adapter, results: Dict[str, Any], out_path: str,
                                repo_root: str, coefficient_sources: Sequence[str],
                                compare_aux: bool = True, comparator=bit_compare,
                                guard_filter: Optional[str] = None,
                                host_mutation: Optional[str] = None,
                                label: str = "gate",
                                num_warps: Optional[int] = None) -> Dict[str, Any]:
    """The full product for the fused pair, under every guard set.

    THE PRODUCT IS 64 CASES, NOT 128, and the difference is not a trim: there is
    exactly one fused pair (the magnetic one), so the ``sub_step`` axis that
    doubles the curl and constitutive products does not exist here. Every other
    axis is the shared one — four shapes, four boundary sets, both Courants, both
    coefficient sources — and the non-power-of-two Courant is as mandatory here as
    everywhere else: at dtdx = 0.5 the scaling is exact in binary and the
    discrepancies this gate exists to catch vanish.
    """
    cases: List[Dict[str, Any]] = []
    for guard_label, guard, primary in adapter.guard_sets:
        if guard_filter is not None and guard_label != guard_filter:
            continue
        adapter.begin_guard(guard)
        try:
            for coefficient_source in coefficient_sources:
                for shape in PML_SHAPES:
                    for boundaries in PML_BOUNDARY_SETS:
                        for dtdx in PML_DTDX:
                            case = one_fused_pair_case(
                                adapter, shape, boundaries, dtdx,
                                coefficient_source, repo_root, compare_aux,
                                comparator, host_mutation, num_warps)
                            case.update({"guard_label": guard_label,
                                         "guard": list(guard),
                                         "primary_guard": primary,
                                         "key": fused_pair_case_key(
                                             coefficient_source, shape,
                                             boundaries, dtdx)})
                            cases.append(case)
                            log(f"[fused:{label}] {guard_label} {case['key']}: "
                                + (f"SKIPPED ({case['skipped']})"
                                   if case.get("skipped") else
                                   f"vs_array={case['vs_array_order']['bit_identical']} "
                                   f"vs_two_kernels={case['vs_two_kernels']['bit_identical']} "
                                   f"two_vs_array={case['two_kernels_vs_array_order']['bit_identical']} "
                                   f"differing={case['vs_array_order']['differing_floats']}"))
                            results.setdefault("fused_pair_bit_identity", {})[label] = \
                                summarize_fused_pair(cases)
                            save(results, out_path)
        finally:
            adapter.end_guard()
    summary = summarize_fused_pair(cases)
    results.setdefault("fused_pair_bit_identity", {})[label] = summary
    save(results, out_path)
    return summary


def summarize_fused_pair(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts per guard set, and the primary's three verdicts kept apart.

    ``primary_identical`` is against the ARRAY PATH, because that is the contract.
    ``primary_matches_two_kernels`` is reported beside it rather than folded in: a
    fused kernel that agrees with the two separate kernels but not with the array
    path, or the reverse, is a different defect from one that fails both.
    """
    out: Dict[str, Any] = {"cases": list(cases), "by_guard": {}}
    for case in cases:
        bucket = out["by_guard"].setdefault(
            case["guard_label"],
            {"ran": 0, "skipped": 0, "identical_vs_array": 0,
             "identical_vs_two_kernels": 0, "two_kernels_identical_vs_array": 0,
             "primary": case.get("primary_guard", False)})
        if case.get("skipped"):
            bucket["skipped"] += 1
            continue
        bucket["ran"] += 1
        bucket["identical_vs_array"] += int(case["vs_array_order"]["bit_identical"])
        bucket["identical_vs_two_kernels"] += int(case["vs_two_kernels"]["bit_identical"])
        bucket["two_kernels_identical_vs_array"] += int(
            case["two_kernels_vs_array_order"]["bit_identical"])
    primary = next((b for b in out["by_guard"].values() if b["primary"]), None)
    out["primary_ran"] = primary["ran"] if primary else 0
    out["primary_identical"] = primary["identical_vs_array"] if primary else 0
    out["primary_matches_two_kernels"] = (
        primary["identical_vs_two_kernels"] if primary else 0)
    out["primary_two_kernels_vs_array"] = (
        primary["two_kernels_identical_vs_array"] if primary else 0)
    out["primary_pass"] = bool(primary) and (
        primary["ran"] > 0 and primary["identical_vs_array"] == primary["ran"])
    return out


def run_fused_pair_multi_step(adapter, results: Dict[str, Any], out_path: str,
                              compare_aux: bool = True, comparator=bit_compare,
                              host_mutation: Optional[str] = None,
                              label: str = "gate",
                              num_warps: Optional[int] = None) -> Dict[str, Any]:
    """Consecutive launches of the pair, byte-compared at EVERY step.

    BOTH auxiliary families are state: ``fu_B*`` compounds the split-field history
    and ``f_w_H*`` is literally last step's B. A kernel right in the primaries and
    wrong in either is correct for one launch and wrong forever after, and a
    single-shot comparison cannot see it.
    """
    guard_label, guard, _ = adapter.guard_sets[0]
    boundaries = PML_BOUNDARY_SETS[3]  # the asymmetric one
    dtdx = 0.35
    zero_metal = zero_metal_flags(boundaries)
    runs: List[Dict[str, Any]] = []
    adapter.begin_guard(guard)
    try:
        for shape in PML_SHAPES:
            curl_coefficients = synthetic_coefficients(cp, shape, True)
            h_coefficients = synthetic_constitutive_coefficients(cp, shape, False)
            kernel_h = (_swapped_kps_kms(h_coefficients)
                        if host_mutation == "swap_kps_kms" else
                        synthetic_constitutive_coefficients(cp, shape, True)
                        if host_mutation == "swap_constitutive_sublattice"
                        else h_coefficients)
            kernel_curl = (synthetic_coefficients(cp, shape, False)
                           if host_mutation == "integer_coefficients_on_B"
                           else curl_coefficients)
            codes = [0 if kind == PERIODIC else 1 for kind in boundaries]
            if host_mutation == "metallic_as_periodic":
                codes = [0, 0, 0]
            codes = [np.int32(c) for c in codes]
            rng = np.random.default_rng(SEED + 33)
            arrays = make_fused_pair_fields(shape, rng)
            reference_state = {n: arrays[n].copy() for n in FUSED_PAIR_STATE}
            kernel_state = dict(arrays)
            for name in FUSED_PAIR_STATE:
                kernel_state[name] = arrays[name].copy()
            curl_flat = flatten_coefficients(kernel_curl)
            h_flat = flatten_coefficients(kernel_h)
            names = (FUSED_PAIR_TARGETS + FUSED_PAIR_H_TARGETS
                     + (FUSED_PAIR_AUX + FUSED_PAIR_H_AUX if compare_aux else ()))
            steps: List[Dict[str, Any]] = []
            identical_steps = 0
            first_divergence: Optional[int] = None
            for step in range(1, MULTI_STEP_COUNT + 1):
                reference_fused_pair_step(
                    cp, arrays, reference_state, curl_coefficients, h_coefficients,
                    np.float32(dtdx), boundaries, zero_metal, "array_order")
                try:
                    adapter.launch("fused_pair_B", kernel_state, curl_flat, codes,
                                   dtdx, extra={"h_flat": h_flat,
                                                "zero_metal": zero_metal,
                                                "num_warps": num_warps})
                    cp.cuda.runtime.deviceSynchronize()
                except Exception as exc:  # noqa: BLE001
                    steps.append({"step": step, "launch_error":
                                  f"{type(exc).__name__}: {exc}"[:1000]})
                    first_divergence = first_divergence or step
                    break
                verdict = combine({n: comparator(kernel_state[n], reference_state[n])
                                   for n in names})
                steps.append({"step": step,
                              "bit_identical": verdict["bit_identical"],
                              "differing_floats": verdict["differing_floats"],
                              "max_ulp": verdict.get("max_ulp", 0),
                              "per_component": {k: v["bit_identical"] for k, v
                                                in verdict["per_component"].items()}})
                if verdict["bit_identical"]:
                    identical_steps += 1
                elif first_divergence is None:
                    first_divergence = step
            run = {"shape": list(shape), "boundaries": list(boundaries),
                   "dtdx": repr(dtdx), "zero_metal": [bool(f) for f in zero_metal],
                   "guard_label": guard_label, "steps": steps,
                   "identical_steps": identical_steps,
                   "first_divergence": first_divergence}
            runs.append(run)
            log(f"[fused:{label}:multi] shape={shape} {MULTI_STEP_COUNT} steps: "
                f"identical_steps={identical_steps}/{MULTI_STEP_COUNT}"
                + (f" first_divergence_at_step={first_divergence}"
                   if first_divergence is not None else ""))
            results.setdefault("fused_pair_multi_step", {})[label] = {
                "runs": runs,
                "all_steps_identical": all(
                    r["identical_steps"] == MULTI_STEP_COUNT for r in runs)}
            save(results, out_path)
    finally:
        adapter.end_guard()
    summary = {"runs": runs,
               "all_steps_identical": all(
                   r["identical_steps"] == MULTI_STEP_COUNT for r in runs)}
    results.setdefault("fused_pair_multi_step", {})[label] = summary
    save(results, out_path)
    return summary


# ---------------------------------------------------------------------------
# The WHOLE-STEP leg — one real configuration, end to end
# ---------------------------------------------------------------------------
#
# Every leg above isolates one sub-step on synthetic arrays. This one runs a real
# ``FdtdDriver`` through ``FdtdDriver.step`` and substitutes the covered sub-steps
# for the Triton plans, comparing EVERY state array bytewise at EVERY step. It is
# the first configuration this track covers end to end and the first number that
# is a whole-step measurement rather than a projection from a sub-step share.
#
# WHAT IS AND IS NOT CLAIMED. Dispatch is NOT wired: ``fastpath.plan_fast_path``
# still returns None on every branch and this round does not touch it. The
# substitution below is performed by the harness, on the driver module's own
# names, which is what makes it a MEASUREMENT of the composed path rather than a
# claim that the engine takes it. When dispatch lands, this leg is the regression
# test that the wiring changed nothing.
#
# The substitution seam is the driver module's globals — ``driver.step_B`` and its
# four siblings are module-level names ``FdtdDriver.step`` looks up per call — so
# neither ``driver.py`` nor ``stepping.py`` is edited to run it.

WHOLE_STEP_COUNT = 10
#: Every array ``FdtdDriver.step`` may write, by name. A whole-step gate that
#: compares the primaries only is blind to exactly what §12.3 measured on the
#: curl: the auxiliaries are state, and a kernel that gets ``f`` right and its
#: auxiliary wrong is correct for one step and wrong forever after.
WHOLE_STEP_ARRAYS = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz",
    "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz",
)


def build_whole_step_driver(repo_root: str, shape, courant: float, boundaries,
                            dispersive: bool, gpu_id: int = 0,
                            seed_auxiliaries: bool = False):
    """One real dispersive/non-dispersive PML driver, seeded deterministically.

    No source: the fields are SEEDED and left to evolve, which exercises every
    sub-step without adding the source slots' host-side work to a comparison that
    is about the kernels. Epsilon is graded rather than uniform so the E side's
    ``inv_eps`` multiply is not the identity, and sigma is a VOLUME so the ADE
    kernel's ``SIGMA_IS_VOLUME`` arm is the one under test.

    ``seed_auxiliaries`` SEEDS ``fu_B*`` TOO, AND IT IS NOT COSMETIC. MEASURED on
    this harness: with the split-field auxiliary starting at zero, a metallic
    wall's B plane is identically zero for every step — the ownership mask makes
    ``curl0`` zero at cell 0, so ``fu`` there stays at whatever it started as and
    the recurrence returns ``B * kms * sinv`` on a B the previous wipe already
    zeroed. ``zero_metal_B`` is then a NO-OP FROM STEP ONE, and a whole-step leg
    cannot see it being dropped: ``drop_zero_metal_inline`` came back 60/60
    UNCAUGHT on the unseeded driver. ``stepping.zero_metal_B``'s own docstring
    records the same thing from the other side — deleting the call survived the
    whole CPU-MEEP metallic cross-validation, and only a magnetic current placed ON
    a wall separates the two.

    Seeding the auxiliary is the smallest change that makes the wall plane live, so
    the wipe becomes observable and the mutation becomes catchable. (The other
    separating configuration, a magnetic source on the wall, is one the fused pair
    REFUSES by predicate — which is the reason the refusal is there.)
    """
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from meep_gpu.dispersion import Susceptibility  # noqa: PLC0415
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=(float(shape[0]), float(shape[1]), float(shape[2])),
        resolution=1.0, courant=float(courant), force_complex_fields=False,
        boundaries=tuple(boundaries), dimensions=3, prefer_gpu=True, gpu_id=gpu_id)
    xp = driver.grid.xp
    if xp.__name__ != "cupy":
        raise RuntimeError("the whole-step leg needs the CuPy backend; got "
                           f"{xp.__name__}")
    index = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    epsilon = (1.5 + 0.75 * np.sin(index * 0.031)).astype(np.float32)
    driver.set_epsilon(xp.asarray(np.ascontiguousarray(epsilon)))
    driver.setup_pml(2)
    if dispersive:
        sigma = np.zeros(shape, dtype=np.float32)
        sigma[1:-1, 1:-1, 1:-1] = 0.4
        driver.add_susceptibility(Susceptibility(frequency=0.6, gamma=0.05),
                                  xp.asarray(np.ascontiguousarray(sigma)))
    rng = np.random.default_rng(SEED + 23)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        driver.set_field(name, xp.asarray(np.ascontiguousarray(
            rng.uniform(-0.5, 0.5, size=shape).astype(np.float32))))
    if seed_auxiliaries:
        for name in ("fu_Bx", "fu_By", "fu_Bz"):
            getattr(driver.fields, name)[...] = xp.asarray(np.ascontiguousarray(
                rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)))
    return driver


def whole_step_state(driver) -> Dict[str, Any]:
    """Every array the step may have written, including each pole's P and P_prev."""
    state = {name: getattr(driver.fields, name) for name in WHOLE_STEP_ARRAYS
             if getattr(driver.fields, name, None) is not None}
    for index, pole in enumerate(driver.fields.polarizations):
        for component in pole.driven():
            state[f"P{index}_{component}"] = pole.P[component]
            state[f"Pprev{index}_{component}"] = pole.P_prev[component]
    return state


def install_triton_substeps(driver_module, plan, owner, mutation: Optional[str] = None):
    """Point the driver module's sub-step names at the plan, and hand back the undo.

    SCOPED TO ONE ``Fields``, and that is not a nicety. The substitution seam is the
    driver MODULE's globals, which every ``FdtdDriver`` in the process shares — so an
    unscoped patch sends the reference driver's steps through the plan too, and
    through a plan whose cached device pointers belong to the OTHER driver's arrays.
    Measured before the ``owner`` check existed: the fused state was stepped twice
    per timestep and every array diverged at step 1, which reads exactly like a
    broken kernel and is not one.

    ``mutation='drop_update_P'`` is spec §G.m10: dispatch the covered sub-steps and
    silently drop one from the array path — the "coverage is a set" mis-plumbing.
    It must diverge at step 1.
    """
    names = ("step_B", "step_D", "update_H", "update_E", "update_P")
    originals = {name: getattr(driver_module, name) for name in names}

    def make(name):
        def wrapper(fields, pml=None):
            if (name == "update_P" and mutation == "drop_update_P"
                    and fields is owner):
                # m10 is "the covered sub-step was dropped from the step entirely",
                # so it must fire whether or not update_P is in the plan. Aimed at
                # the refusal branch only, it never fired at all: update_P IS
                # covered for a dispersive run, so the dispatch branch ran first and
                # the leg reported 10/10 identical for a mutation it never applied.
                return None
            entry = plan.plans.get(name) if fields is owner else None
            if mutation == "replace_nothing":
                # The null control: the wrapper is installed on every name and
                # delegates on every name. Anything that diverges under it is the
                # harness, which is exactly what §13.4's third defect looked like.
                entry = None
            if entry is None:
                return originals[name](fields, pml)
            if name == "update_P":
                for sub_plan in entry:
                    sub_plan.run(fields.drive_field)
                return None
            entry.run()
            return None
        return wrapper

    for name in names:
        setattr(driver_module, name, make(name))
    return lambda: [setattr(driver_module, n, f) for n, f in originals.items()]


def install_mutated_kernels(plan, mutated: Dict[str, Any]) -> Dict[str, int]:
    """Point every engine-route plan at the mutated kernel of its own class.

    ``plan_step`` builds plans with ``kernel=None``, which means "launch the
    shipped kernel" — correct for the engine and disarming for a mutation leg. The
    per-class mapping is applied here, after the plan is built and before a single
    step runs, and the COUNT of plans actually re-pointed is returned so a leg that
    installed nothing cannot look like a leg that installed something.
    """
    installed: Dict[str, int] = {}
    if not mutated:
        return installed
    for name, entry in plan.plans.items():
        for item in (entry if isinstance(entry, (list, tuple)) else (entry,)):
            kernel = mutated.get(type(item).__name__)
            if kernel is None or not hasattr(item, "_kernel"):
                continue
            item._kernel = kernel
            installed[name] = installed.get(name, 0) + 1
    return installed


def run_whole_step(results: Dict[str, Any], out_path: str, repo_root: str,
                   gpu_id: int = 0, mutation: Optional[str] = None,
                   label: str = "gate", fuse: bool = False,
                   steps_budget: int = WHOLE_STEP_COUNT,
                   seed_auxiliaries: bool = False,
                   adapter: Any = None,
                   num_warps: Optional[int] = None) -> Dict[str, Any]:
    """N real steps on the array path vs N with the covered sub-steps on Triton.

    Two drivers are built identically and seeded identically; one runs untouched,
    the other with the driver module's sub-step names pointed at the plan. Every
    state array is compared as BYTES after every step.

    ``steps_budget`` IS A COVERAGE AXIS, not a runtime knob. §14.5 measured a
    divergence that 8, 10 and 6 consecutive steps all passed and 40 did not:
    "identical for N steps" is a claim about N. The fused-pair gate runs 60.

    ``fuse=True`` asks ``plan_step`` for the cross-sub-step pair. It is off by
    default because the composed-but-unfused plan is the control this round is
    measured against, and both must keep running from the same harness.
    """
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from meep_gpu import driver as driver_module  # noqa: PLC0415
    from meep_gpu.triton_kernels import plan_step  # noqa: PLC0415

    runs: List[Dict[str, Any]] = []
    configurations = [
        ("nondispersive", (24, 20, 16), 0.35, (METALLIC, METALLIC, PERIODIC), False),
        ("dispersive", (24, 20, 16), 0.35, (METALLIC, METALLIC, PERIODIC), True),
        ("dispersive_cube", (16, 16, 16), 0.5, (METALLIC, METALLIC, METALLIC), True),
    ]
    for name, shape, courant, boundaries, dispersive in configurations:
        started = time.time()
        run: Dict[str, Any] = {
            "configuration": name, "shape": list(shape), "courant": repr(courant),
            "boundaries": list(boundaries), "dispersive": dispersive,
            "mutation": mutation, "steps": steps_budget,
            "seed_auxiliaries": seed_auxiliaries,
        }
        try:
            reference = build_whole_step_driver(repo_root, shape, courant,
                                                boundaries, dispersive, gpu_id,
                                                seed_auxiliaries)
            fused = build_whole_step_driver(repo_root, shape, courant, boundaries,
                                            dispersive, gpu_id, seed_auxiliaries)
            # The driver owns the source list; the fused pair's predicate refuses
            # `None` rather than assuming an empty one, so it is passed explicitly.
            plan = plan_step(fused.fields, fused.pml, fuse=fuse,
                             sources=tuple(getattr(fused, "_sources", ())),
                             num_warps=num_warps)
            run["replaces"] = list(plan.replaces)
            run["fuse"] = fuse
            run["fused_pair_installed"] = bool(
                type(plan.plans.get("update_H")).__name__ == "NoopPlan")
            run["mutated_kernels_installed"] = install_mutated_kernels(
                plan, adapter.mutated_kernels() if adapter is not None else {})
            run["refusals"] = {k: list(v) for k, v in plan.reasons.items()}
            run["array_step_path"] = fused.active_step_path
        except Exception as exc:  # noqa: BLE001 - a refusal IS the result
            run["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            runs.append(run)
            log(f"[whole:{label}] {name}: ERROR {run['error']}")
            results.setdefault("whole_step", {})[label] = {"runs": runs}
            save(results, out_path)
            continue

        # The initial states must be byte-equal or nothing below means anything.
        run["seeded_identically"] = combine({
            key: bit_compare(value, whole_step_state(reference)[key])
            for key, value in whole_step_state(fused).items()})["bit_identical"]

        undo = install_triton_substeps(driver_module, plan, fused.fields, mutation)
        steps: List[Dict[str, Any]] = []
        identical_steps = 0
        first_divergence: Optional[int] = None
        try:
            for step in range(1, steps_budget + 1):
                reference.step()
                fused.step()
                cp.cuda.runtime.deviceSynchronize()
                ref_state = whole_step_state(reference)
                fused_state = whole_step_state(fused)
                verdict = combine({key: bit_compare(fused_state[key], ref_state[key])
                                   for key in sorted(ref_state)})
                steps.append({
                    "step": step, "bit_identical": verdict["bit_identical"],
                    "differing_floats": verdict["differing_floats"],
                    "max_ulp": verdict.get("max_ulp", 0),
                    "differing_arrays": sorted(
                        k for k, v in verdict["per_component"].items()
                        if not v["bit_identical"]),
                })
                if verdict["bit_identical"]:
                    identical_steps += 1
                elif first_divergence is None:
                    first_divergence = step
        except Exception as exc:  # noqa: BLE001
            run["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        finally:
            undo()
        run.update({"steps": steps, "identical_steps": identical_steps,
                    "first_divergence": first_divergence,
                    "seconds": round(time.time() - started, 2)})
        runs.append(run)
        log(f"[whole:{label}] {name} replaces={run.get('replaces')} "
            f"fused_pair={run.get('fused_pair_installed')} "
            f"mutated={run.get('mutated_kernels_installed')} "
            f"identical_steps={identical_steps}/{steps_budget}"
            + (f" first_divergence_at_step={first_divergence}"
               if first_divergence is not None else "")
            + f" ({run['seconds']} s)")
        results.setdefault("whole_step", {})[label] = {
            "runs": runs, "steps_budget": steps_budget, "fuse": fuse,
            "all_steps_identical": all(
                r.get("identical_steps") == steps_budget for r in runs)}
        save(results, out_path)
    summary = {"runs": runs, "steps_budget": steps_budget, "fuse": fuse,
               "all_steps_identical": all(
                   r.get("identical_steps") == steps_budget for r in runs)}
    results.setdefault("whole_step", {})[label] = summary
    save(results, out_path)
    return summary


# ---------------------------------------------------------------------------
# Mutations — a gate no mutation exercises is indistinguishable from one that
# returns True. Every entry here must be verified CAUGHT (or, for the two
# harness mutations, verified to make a real defect go UNCAUGHT).
# ---------------------------------------------------------------------------

def _regroup_stencil(source: str) -> Tuple[str, int]:
    """Mutation 3: ``((a-b) + (c-d))`` -> ``(a-b+c-d)``, which C reads as ``((a-b)+c)-d``."""
    needle = "dtdx * ((sf - f1) + (f2 - ss))"
    return source.replace(needle, "dtdx * (sf - f1 + f2 - ss)"), source.count(needle)


def _drop_metallic_mask(source: str) -> Tuple[str, int]:
    """Mutation 4: delete every ``_mask_non_owned_cells`` wall line."""
    pattern = r"\n *if \(bc_[xyz] == BC_METALLIC && [ijk] == 0\) curl = 0\.0f;"
    return re.subn(pattern, "", source)[0], len(re.findall(pattern, source))


def _drop_folded_periodic_top_mask(source: str) -> Tuple[str, int]:
    """Delete the Yee-shift-1 TOP-plane mask on a folded PERIODIC axis.

    THE DEFECT THIS ARMS IS THE ONE THAT WAS MEASURED. Before the branch existed,
    the shipped real-PML curl pair diverged 0/64 on a folded PERIODIC axis with
    every one of 19,649 differing words on this exact plane and none elsewhere
    (``results/cuda_folded_curl_2026-08-19/``). Removing it should reproduce that
    divergence, which is what makes "the branch closed the gap" a measurement
    rather than a coincidence of the fixture.

    Scored only on legs that HAVE a folded periodic axis: on a folded metallic
    grid no axis carries ``BC_MIRROR_PERIODIC`` and the deletion is inert, so a
    leg there would report UNCAUGHT for a reason about the grid.
    """
    pattern = (r"\n *if \(bc_[xyz] == BC_MIRROR_PERIODIC && [ijk] == n[xyz] - 1\)"
               r" curl = 0\.0f;")
    return re.subn(pattern, "", source)[0], len(re.findall(pattern, source))


def _drop_folded_periodic_near_mask(source: str) -> Tuple[str, int]:
    """Delete the cell-0 mask on a folded PERIODIC axis, leaving the METALLIC one.

    The fold's OTHER mask, and a separate leg because it is a separate branch on a
    separate plane. ``_mask_non_owned_cells`` zeroes cell 0 for every Yee-shift-0
    component on a MIRRORED axis (stepping.py:1943-1947) — a folded METALLIC axis
    inherits that from ``BC_METALLIC`` and a folded PERIODIC one does not, so this
    line is the only thing supplying it there. Deliberately narrow: it leaves
    ``if (bc_x == BC_METALLIC && i == 0)`` in place, so a leg that scores CAUGHT
    scored it on the fold's own mask and not on the wall mask beside it.
    """
    pattern = r"\n *if \(bc_[xyz] == BC_MIRROR_PERIODIC && [ijk] == 0\) curl = 0\.0f;"
    return re.subn(pattern, "", source)[0], len(re.findall(pattern, source))


def _swap_dsig_dsigu(source: str) -> Tuple[str, int]:
    """Mutation 5: swap kms/sinv with kms_u/sinv_u inside the recurrence."""
    needle = ("    float fu_new = ((fprev * kms) - curl) * sinv;\n"
              "    fu[idx] = fu_new;\n"
              "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;")
    replacement = ("    float fu_new = ((fprev * kms_u) - curl) * sinv_u;\n"
                   "    fu[idx] = fu_new;\n"
                   "    f[idx] = (((f[idx] * kms) + fu_new) - fprev) * sinv;")
    return source.replace(needle, replacement), source.count(needle)


def _drop_fu_store(source: str) -> Tuple[str, int]:
    """The auxiliary-only defect: compute ``fu_new``, use it for ``f``, never store it.

    ``f`` is bit-identical on every single launch — the recurrence reads the
    register, not the array — and ``fu`` is wrong from the first one. Nothing but
    the auxiliary comparison can see it, which is what makes it the measurement
    of whether that comparison is load-bearing. Spec mutation 8 nominates the
    dropped mask for that role; whether the mask is in fact ``f``-visible is
    measured here rather than assumed.
    """
    needle = "    fu[idx] = fu_new;\n"
    return source.replace(needle, ""), source.count(needle)


def _commute_dtdx_scale(source: str) -> Tuple[str, int]:
    """A NULL: ``dtdx * curl`` written ``curl * dtdx``. IEEE multiply commutes.

    Every other mutation on the curl must be CAUGHT, and a battery of only those
    scores identically whether the comparator works or has degenerated into
    failing everything. This one must come back N/N IDENTICAL, and it is not a
    tautology: it edits the SAME expression ``regroup_stencil`` edits, one
    character away, and that leg is measured caught. So the pair says the
    comparator is sensitive to the association of that expression and insensitive
    to the operand order of its multiply — which is exactly what IEEE-754 says,
    and now what the artifact says too.
    """
    needle = "float curl = dtdx * ((sf - f1) + (f2 - ss));"
    return (source.replace(needle, "float curl = ((sf - f1) + (f2 - ss)) * dtdx;"),
            source.count(needle))


def _reload_fu_from_memory(source: str) -> Tuple[str, int]:
    """A NULL: use the value just STORED to ``fu`` instead of the register.

    A float32 held in a register, stored to global memory and loaded straight
    back is the identity on the bits — there is no wider accumulator on this
    path, so the round trip cannot re-round. The sibling track measured the same
    claim as ``reload_B_from_memory`` and it came back 60/60 identical.

    Its discriminator is :func:`_read_fprev_after_store`, which touches the same
    two lines and MUST be caught. Without that pair, "the reload is inert" would
    be a statement about a leg nobody showed could fail.
    """
    needle = "    f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;"
    replacement = "    f[idx] = (((f[idx] * kms_u) + fu[idx]) - fprev) * sinv_u;"
    return source.replace(needle, replacement), source.count(needle)


def _read_fprev_after_store(source: str) -> Tuple[str, int]:
    """The aliasing trap: read ``fu`` for ``fprev`` AFTER the new value lands.

    ``fprev`` becomes ``fu_new`` and the recurrence loses its history term. This
    is the defect the array path's explicit copy exists to prevent, and it is the
    discriminating sibling that makes :func:`_reload_fu_from_memory`'s pass a
    measurement: same two lines, one inert and one lethal.
    """
    needle = ("    float fprev = fu[idx];\n"
              "    float fu_new = ((fprev * kms) - curl) * sinv;\n"
              "    fu[idx] = fu_new;\n")
    replacement = ("    float fu_new = ((fu[idx] * kms) - curl) * sinv;\n"
                   "    fu[idx] = fu_new;\n"
                   "    float fprev = fu[idx];\n")
    return source.replace(needle, replacement), source.count(needle)


# ---------------------------------------------------------------------------
# The hand track's spellings of the CONSTITUTIVE defects.
#
# The gate owns WHICH defects the constitutive sub-step must be proof against
# (:data:`CONSTITUTIVE_SOURCE_MUTATIONS`); each track owns the spelling, because
# one track's kernel is a CUDA string and the other's is Python. These four are
# the hand track's, plus two NULLS that must come back UNCAUGHT — each editing
# the same expression as one of the four, one character away, so "inert" is a
# measurement rather than a claim about a leg nobody showed could fail.
#
# All of them target ``constitutive_kernels.py``. The first three live in the
# shared ``_REAL_CONSTITUTIVE_PRELUDE``, so each matches twice (once per kernel
# string); the inverse-epsilon pair lives in the E kernel only and matches three
# times, once per component.
# ---------------------------------------------------------------------------

def _regroup_constitutive(source: str) -> Tuple[str, int]:
    """``((f + kps*src) - kms*prev)`` flattened to ``f + (kps*src - kms*prev)``.

    The array path accumulates TWICE, left to right (stepping.py:2133-2134, and
    the scratch branch :2140-2143 by a different route). The flattened form is a
    different float32 number, and it is what both complex constitutive kernels in
    ``step_curl_kernels.py`` are written as. MUST BE CAUGHT.
    """
    needle = ("    float a = f[idx] + kps * src;\n"
              "    f[idx] = a - kms * prev;\n")
    replacement = "    f[idx] = f[idx] + (kps * src - kms * prev);\n"
    return source.replace(needle, replacement), source.count(needle)


def _drop_fw_store(source: str) -> Tuple[str, int]:
    """The auxiliary-only defect: never write the new ``f_w``.

    ``f`` is bit-identical on the FIRST launch — ``prev`` is loaded before the
    store, so nothing in the launch reads it back — and ``f_w`` is wrong from
    that launch onwards, which makes ``f`` wrong from the second. Only the
    auxiliary comparison can see it on a single shot, which is what makes this
    the measurement of whether that comparison is load-bearing. §12.3 measured a
    dropped auxiliary store at 120/120 UNCAUGHT on the curl when ``fu`` was not
    compared. MUST BE CAUGHT.
    """
    needle = "    fw[idx] = src;\n"
    return source.replace(needle, ""), source.count(needle)


def _store_fw_before_reading_prev(source: str) -> Tuple[str, int]:
    """The aliasing trap: read ``prev`` AFTER the new value lands.

    ``prev`` becomes ``src`` and the recurrence loses its history term. This is
    the defect the array path's explicit ``fw.copy()`` (stepping.py:2131) exists
    to prevent, and it is wrong only where ``kms != 0`` — i.e. INSIDE THE PML —
    so it reads as a slightly weaker absorber rather than as a bug. MUST BE
    CAUGHT.
    """
    needle = ("    float prev = fw[idx];\n"
              "    fw[idx] = src;\n")
    replacement = ("    fw[idx] = src;\n"
                   "    float prev = fw[idx];\n")
    return source.replace(needle, replacement), source.count(needle)


def _drop_inverse_epsilon(source: str) -> Tuple[str, int]:
    """E side: the source becomes D instead of ``D * inv_eps``.

    Invisible against a table of ones, which is why the gate's fixture draws
    inverse epsilon in [0.2, 0.9) and away from 1.0. MUST BE CAUGHT.
    """
    pattern = r"(D([xyz])\[idx\]) \* inv_eps_E\2\[idx\]"
    return re.subn(pattern, r"\1", source)[0], len(re.findall(pattern, source))


def _inv_eps_left(source: str) -> Tuple[str, int]:
    """A NULL: ``inv_eps * D`` in place of ``D * inv_eps``. IEEE multiply commutes.

    stepping.py:1011 writes ``source * fields.inverse_epsilon_for(component)``
    with D on the left and this kernel keeps that order — for transcription
    discipline, not because it changes a bit. The sibling track measured this
    exact null at 30/30 identical on real-engine cases. Its discriminating
    sibling is :func:`_drop_inverse_epsilon`, which edits the same expression and
    MUST be caught; without that pair, "the operand order is inert" would be a
    statement about a leg nobody showed could fail. MUST BE UNCAUGHT.
    """
    pattern = r"(D([xyz])\[idx\]) \* (inv_eps_E\2\[idx\])"
    return re.subn(pattern, r"\3 * \1", source)[0], len(re.findall(pattern, source))


def _commute_constitutive_scale(source: str) -> Tuple[str, int]:
    """A NULL: ``src * kps`` in place of ``kps * src``, on the accumulation itself.

    The array path writes the coefficient on the left in both branches
    (``field += kps * fw``, :2086; ``multiply(kps, fw, out=product)``, :2093) and
    so does this kernel. The paired defect on the same two lines is
    :func:`_regroup_constitutive`, which MUST be caught — so the pair says the
    comparator is sensitive to the ASSOCIATION of that expression and insensitive
    to the operand order of its multiply, which is exactly what IEEE-754 says.
    MUST BE UNCAUGHT.
    """
    needle = ("    float a = f[idx] + kps * src;\n"
              "    f[idx] = a - kms * prev;\n")
    replacement = ("    float a = f[idx] + src * kps;\n"
                   "    f[idx] = a - prev * kms;\n")
    return source.replace(needle, replacement), source.count(needle)


def _own_axis_to_x_for_all_three(source: str) -> Tuple[str, int]:
    """THE AXIS MAPPING: every component reads the x table instead of its own.

    Note 2 of ``constitutive_kernels.py`` calls this the highest-consequence
    confusion in the file — "a smooth, converged, entirely wrong absorber" — and
    until 2026-08-15 nothing armed it. MEEP's ``dsigw`` is the absorption a
    component accumulates along the direction IT POINTS IN
    (``H_CONSTITUTIVE_TERMS`` / ``E_CONSTITUTIVE_TERMS``, stepping.py:227-228), so
    ``kps_x``/``kms_x`` for component 0, ``_y`` for 1, ``_z`` for 2.

    IN BOUNDS ON EVERY SHAPE BY CONSTRUCTION — ``i`` indexes ``kps_x``, which has
    ``nx`` entries, whatever the component. That matters: this must be caught as a
    WRONG ANSWER, not as a launch failure, because a wrong answer is what it would
    be in production. MEASURED lethal on the laptop against the same expression
    tree, caught at step 1, with a same-fixture unmutated control identical.
    MUST BE CAUGHT.
    """
    pattern = r"kps_([yz])\[([jk])\], kms_\1\[\2\]"
    return re.subn(pattern, "kps_x[i], kms_x[i]", source)[0], \
        len(re.findall(pattern, source))


def _fortran_order_index_decomposition(source: str) -> Tuple[str, int]:
    """THE LINEAR-INDEX DECOMPOSITION, written for the wrong memory order.

    The arrays are C-contiguous — the predicate refuses anything else — so the
    last axis is the fastest, ``k = idx % nz`` and ``i = idx / (ny*nz)``. Writing
    the Fortran-order decomposition instead swaps which axis each coefficient
    lookup reads without touching the field access, which is the same class of
    silent absorber error as the axis mapping and the one a reader porting an
    index block from column-major code introduces.

    IN BOUNDS ON EVERY SHAPE BY CONSTRUCTION, which is why this spelling was
    chosen over swapping ``i`` and ``k`` in place: ``i = idx % nx`` stays below
    ``nx``, ``j`` below ``ny``, and ``k = idx / (nx*ny)`` below ``nz`` because
    ``idx < nx*ny*nz``. A mutation that read past a coefficient vector would be
    caught by a fault on some shapes and by garbage on others, and neither is the
    defect being measured. MUST BE CAUGHT.
    """
    needle = ("    int k = idx % nz;\n"
              "    int j = (idx / nz) % ny;\n"
              "    int i = idx / (ny * nz);\n")
    replacement = ("    int i = idx % nx;\n"
                   "    int j = (idx / nx) % ny;\n"
                   "    int k = idx / (nx * ny);\n")
    return source.replace(needle, replacement), source.count(needle)


def _bind_Ez_inv_eps_for_all_three(source: str) -> Tuple[str, int]:
    """DEFECT 2 OF THE COMPLEX TEMPLATE, armed: one epsilon volume for all three.

    ``update_E_pml_complex``'s wrapper passes ``fields.inv_eps``, which
    ``Fields.set_epsilon_volumes`` re-points at the Ez view (fields.py:1259-1260)
    as a "representative material array". On any run whose three components do not
    alias one volume — a diagonal anisotropic epsilon — Ex and Ey are then updated
    with Ez's inverse permittivity. The kernel this gate measures binds three
    pointers; this mutation makes it bind one, which is the defect the template
    would have carried in.

    EXACTLY BIT-IDENTICAL UNDER AN ISOTROPIC EPSILON, measured. Its whole power
    comes from ``constitutive_field_hosts`` drawing three INDEPENDENT volumes, so
    a leg reporting this UNCAUGHT is reporting that the fixture collapsed rather
    than that the kernel is right. ``drop_inverse_epsilon`` removes the multiply
    entirely and could never see it. MUST BE CAUGHT.
    """
    pattern = r"(D([xy])\[idx\] \* )inv_eps_E\2\[idx\]"
    return re.subn(pattern, r"\1inv_eps_Ez[idx]", source)[0], \
        len(re.findall(pattern, source))


SOURCE_MUTATIONS = {
    "regroup_stencil": _regroup_stencil,
    "drop_metallic_mask": _drop_metallic_mask,
    "drop_folded_periodic_top_mask": _drop_folded_periodic_top_mask,
    "drop_folded_periodic_near_mask": _drop_folded_periodic_near_mask,
    "swap_dsig_dsigu": _swap_dsig_dsigu,
    "drop_fu_store": _drop_fu_store,
    "commute_dtdx_scale": _commute_dtdx_scale,
    "reload_fu_from_memory": _reload_fu_from_memory,
    "read_fprev_after_store": _read_fprev_after_store,
    # The constitutive pair's four defects and their two nulls.
    "regroup_constitutive": _regroup_constitutive,
    "drop_fw_store": _drop_fw_store,
    "store_fw_before_reading_prev": _store_fw_before_reading_prev,
    "drop_inverse_epsilon": _drop_inverse_epsilon,
    "inv_eps_left": _inv_eps_left,
    "commute_constitutive_scale": _commute_constitutive_scale,
    # The two indexing defects and the shared-epsilon defect, armed 2026-08-15.
    "own_axis_to_x_for_all_three": _own_axis_to_x_for_all_three,
    "fortran_order_index_decomposition": _fortran_order_index_decomposition,
    "bind_Ez_inv_eps_for_all_three": _bind_Ez_inv_eps_for_all_three,
}
#: ``inv_eps_left`` MUST BE UNCAUGHT and ``commute_constitutive_scale`` MUST BE
#: UNCAUGHT, each paired with a defect on the same expression that must be caught
#: (``drop_inverse_epsilon`` and ``regroup_constitutive`` respectively). A leg
#: reporting either as CAUGHT is reporting a defect in the kernel or in the
#: harness, not a working gate — and a battery of only-must-be-caught mutations
#: scores identically whether the comparator works or has degenerated into
#: failing everything.
CONSTITUTIVE_NULL_MUTATIONS = ("inv_eps_left", "commute_constitutive_scale")
#: ``commute_dtdx_scale`` MUST BE UNCAUGHT and ``reload_fu_from_memory`` MUST BE
#: UNCAUGHT — the curl leg's answer to the clause the sibling record carries as
#: ``mutations_that_must_be_UNCAUGHT`` and this one had none of. A leg reporting
#: either as CAUGHT is reporting a defect in the kernel or in the harness, not a
#: working gate. Each is paired with a leg on the same expression that must be
#: caught (``regroup_stencil`` and ``read_fprev_after_store``), because "inert"
#: only means something beside a demonstration that the leg can fail at all.
PML_NULL_MUTATIONS = ("commute_dtdx_scale", "reload_fu_from_memory")
#: Defects that only exist for a track carrying a constitutive kernel. The gate
#: owns the LIST — what defects the constitutive sub-step must be proof against —
#: and each track owns the spelling, as with the four above. A track that does not
#: implement one fails with a KeyError naming it, which is the correct answer: it
#: has no such kernel to break.
CONSTITUTIVE_SOURCE_MUTATIONS = (
    "regroup_constitutive",          # ((f + kps*src) - kms*prev) flattened
    "drop_fw_store",                 # the auxiliary-only defect
    "store_fw_before_reading_prev",  # the aliasing trap the array path's copy prevents
    "drop_inverse_epsilon",          # E side: source becomes D instead of D*inv_eps
    # THE TWO INDEXING DEFECTS, added 2026-08-15. Everything above edits the
    # ARITHMETIC; nothing edited WHICH COEFFICIENT A CELL READS, which the
    # kernel's own note 2 calls the highest-consequence confusion in the file and
    # which is the only defect on this list that produces a converged, smooth,
    # entirely wrong absorber rather than a visibly broken field.
    "own_axis_to_x_for_all_three",   # every component reads the x table
    "fortran_order_index_decomposition",  # i and k swapped by writing the wrong order
    # E side: one epsilon volume bound three times — defect 2 of the complex
    # template, and invisible under an isotropic epsilon, so it also measures that
    # the fixture's three independent volumes are load-bearing.
    "bind_Ez_inv_eps_for_all_three",
)
#: Defects that only exist once two sub-steps share one launch. The first is not a
#: defect at all and is listed anyway, deliberately — see the note below.
FUSED_PAIR_SOURCE_MUTATIONS = (
    "reload_B_from_memory",           # THE TRUNCATION QUESTION, asked as a mutation
    "reload_B_from_memory_but_shift",  # ...and its discriminating control
    "drop_zero_metal_inline",         # the wall wipe the seam has to carry
    "zero_metal_only_on_the_store",   # ...carried for B but not for the H read
    "hoist_constitutive_above_curl_store",  # the ordering; a NULL mutation, see below
)
#: ``reload_B_from_memory`` MUST BE UNCAUGHT and ``hoist_constitutive_above_curl_store``
#: MUST BE UNCAUGHT. Both are here because a claim that they cannot matter is worth
#: more measured than argued:
#:
#: * the reload puts the float32 round trip back while keeping the fusion. On
#:   NVIDIA a Triton fp32 value is a 32-bit register, so store-then-load is the
#:   identity on the bits and the leg must come back identical. Its shifted sibling
#:   reads the neighbouring cell and must be caught — that is what makes the first
#:   result a measurement rather than a tautology.
#: * the hoist moves the constitutive block above the curl's stores. Nothing reads
#:   ``f0..f2`` back, so it is provably unobservable, and measuring it is how the
#:   claim "the seam is a pure register substitution" stops being an assertion.
#:
#: A leg that reports these two as CAUGHT is reporting a defect in the kernel or in
#: the harness, not a working gate.
FUSED_PAIR_NULL_MUTATIONS = ("reload_B_from_memory",
                             "hoist_constitutive_above_curl_store")
#: Defects that only exist once a POLE LIST enters the constitutive sub-step. The
#: gate owns the list; each track owns the spelling.
#:
#: The first two are the ones a reader of the docstring sentence "E = (D - sum P) *
#: inv_eps" will write by accident, and both are INVISIBLE AT ONE POLE — measured
#: 20/20 caught at two poles and 0/10 at one. A leg that reports them caught on a
#: single-pole row is reporting a harness defect, not a working gate.
DISPERSIVE_SOURCE_MUTATIONS = (
    "sum_then_subtract",              # D - (P0 + P1), the pre-accumulated design
    "regroup_constitutive_dispersive",  # ((E + kps*src) - kms*prev) flattened
    "drop_fw_store_dispersive",       # the auxiliary-only defect, on update_P's input
    "store_fw_before_reading_prev_dispersive",  # prev becomes the value just stored
    "drop_one_pole",                  # one pole silently missing from the chain
    "drop_inverse_epsilon_dispersive",  # source becomes (D - sum P), unscaled
    "inv_eps_left",                   # a NULL: float32 multiply IS commutative
)
#: ``inv_eps_left`` MUST BE UNCAUGHT. ``inv_eps * s`` versus ``s * inv_eps`` was
#: measured bitwise identical on 30/30 real-engine cases; the array path's operand
#: order is kept for transcription discipline, not because it changes a bit. A leg
#: that reports this CAUGHT is comparing something other than bytes.
DISPERSIVE_NULL_MUTATIONS = ("inv_eps_left",)
#: ``reverse_pole_order`` is a HOST mutation rather than a source one because the
#: PLAN fills the kernel's pole slots: reversing them there is the faithful spelling
#: of "the plan resolved ``fields.polarizations`` order backwards", which is the
#: defect that actually threatens this kernel.
HOST_MUTATIONS = ("integer_coefficients_on_B", "metallic_as_periodic",
                  "swap_constitutive_sublattice", "swap_kps_kms",
                  "reverse_pole_order")
#: Whole-step mutations: defects in the COMPOSITION rather than in any one kernel.
#: ``replace_nothing`` is the NULL CONTROL: the substitution machinery is installed
#: and every sub-step delegates, so a divergence under it is the harness's, not a
#: kernel's. §13.4's third defect is why that control exists at all.
WHOLE_STEP_MUTATIONS = ("drop_update_P", "replace_nothing")


def apply_source_mutation(adapter, name: str) -> Dict[str, Any]:
    """Rewrite the track's kernel sources and report how many sites were hit.

    A track now exposes MORE THAN ONE kernel source, and a mutation aimed at one
    kernel legitimately matches nothing in the others — the regroup needle for the
    constitutive kernel does not appear in the curl. So the drift check is on the
    TOTAL: a mutation that matched nothing ANYWHERE has drifted away from every
    kernel and is exercising nothing, which still raises. The per-attribute counts
    go into the artifact so a reader sees which kernel a leg actually hit.
    """
    # A track may own the SPELLING of these defects (its kernel may not be CUDA);
    # the gate still owns which defects exist and what their verdict must be.
    transform = (getattr(adapter, "source_mutations", None) or SOURCE_MUTATIONS)[name]
    mutated: Dict[str, str] = {}
    sites: Dict[str, int] = {}
    before: Dict[str, str] = {}
    after: Dict[str, str] = {}
    # A track may spread its kernels over more than one module — the hand track
    # does, because ``step_curl_kernels.py``'s device strings are digest-pinned
    # and the constitutive pair had to land beside them rather than inside them.
    # Reading every attribute off ``adapter.module`` would then raise
    # AttributeError on the sibling's kernels, or worse, silently read a stale
    # copy if one were ever mirrored there.
    owner = getattr(adapter, "source_attribute_owner", None)
    for attribute in adapter.source_attributes():
        holder = owner(attribute) if callable(owner) else adapter.module
        original = getattr(holder, attribute)
        new_source, count = transform(original)
        sites[attribute] = count
        before[attribute] = hashlib.sha256(original.encode("utf-8")).hexdigest()
        if count:
            mutated[attribute] = new_source
            after[attribute] = hashlib.sha256(new_source.encode("utf-8")).hexdigest()
    if sum(sites.values()) == 0:
        raise RuntimeError(
            f"mutation {name!r} matched nothing in any of {tuple(sites)}; the "
            f"mutation and the kernels have drifted apart and the gate is not "
            f"being exercised by it.")
    for attribute, count in sites.items():
        if count and before[attribute] == after.get(attribute):
            raise RuntimeError(
                f"mutation {name!r} matched {count} site(s) in {attribute} and "
                f"left the source byte-identical; it replaces its anchor with "
                f"itself and is exercising nothing.")
    adapter.set_source_mutation(mutated)
    return {
        "name": name,
        "sites": sites,
        "total_sites": sum(sites.values()),
        "attributes_rewritten": sorted(mutated),
        "original_source_sha256": before,
        "mutated_source_sha256": after,
    }


def armed_mutation_accounting(adapter, armed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Did the mutated bytes actually reach the COMPILER, or only the text?

    THE DEFECT THIS EXISTS FOR is a leg that reports a pass for a mutation it
    never applied — indistinguishable, in a summary, from a leg that applied it
    and found nothing. The sibling track hit it three times, and only caught it
    because ``install_mutated_kernels()`` reports its re-point count.

    ``apply_source_mutation`` already refuses a mutation that matched no text.
    That is necessary and not sufficient: between the rewrite and NVRTC sit an
    adapter that has to re-install the attribute, a memo that has to miss, and a
    disk cache that has to not answer. So this reads the kernel module's OWN
    compile log — one entry per ``cp.RawKernel`` CONSTRUCTION, carrying the
    sha256 of the source string handed to the compiler — and reports how many of
    those were of the mutated bytes.

    A construction is an UPPER BOUND on NVRTC calls, not a count of them: the
    last item on that list, CuPy's disk cache, is keyed above the strip seam and
    can answer a construction with no compiler involved (measured in this
    project's ``cuda_policy_reach`` artifact). For the ARMED case this counter
    exists for the two coincide, because mutated source is a miss in that
    source-keyed cache — but the field names the quantity it counts, so an
    auditor is not reading a compile count that was never taken.

    A track whose module keeps no such log reports ``measurable: False`` by name
    rather than a zero, because a zero here would read as "the mutation never
    landed", which is the exact wrong conclusion to draw from a missing counter.

    EVERY LOADED MODULE'S LOG IS READ, NOT JUST THE CURL'S. MEASURED FAILURE,
    2026-08-15, on the constitutive pair's first gate run: every source-mutation
    leg reported ``0 out of 0 kernel constructions`` and was refused. Both kernel
    modules are loaded BY PATH by this probe, so each one takes the ImportError
    fallback in its own header and builds its OWN ``compile_cache`` module
    object — they are not the shared memo the modules' docstrings describe when
    imported as a package. Reading ``adapter.module.compile_cache`` alone
    therefore read the CURL module's log, which is empty on a constitutive leg,
    and the accounting said "the mutation never reached the compiler" about a
    mutation that had. The logs are unioned across ``adapter._modules()`` and
    de-duplicated by identity, so a track whose modules DO share one memo counts
    it once and a track whose modules do not still counts both.
    """
    if not armed:
        return {"applies": False,
                "why": "no source mutation was armed on this leg"}
    if hasattr(adapter, "_modules"):
        modules = list(adapter._modules())
    else:
        modules = [m for m in (getattr(adapter, "module", None),) if m is not None]
    caches, seen = [], set()
    for module in modules:
        cache = getattr(module, "compile_cache", None)
        if cache is not None and hasattr(cache, "compile_log") and id(cache) not in seen:
            seen.add(id(cache))
            caches.append((getattr(module, "__name__", "<unnamed>"), cache))
    if not caches:
        return {"applies": True, "measurable": False,
                "why": "this track's kernel module keeps no compile log, so the "
                       "count of compiles taken from the mutated source cannot "
                       "be read here"}
    compiled = [entry["source_sha256"]
                for _, cache in caches for entry in cache.compile_log()]
    mutated = set(armed["mutated_source_sha256"].values())
    original = set(armed["original_source_sha256"].values())
    from_mutated = sum(1 for digest in compiled if digest in mutated)
    return {
        "applies": True,
        "measurable": True,
        # Which logs this count came from. A constitutive leg that reported zero
        # because only the curl module's log was read is exactly the failure this
        # field makes visible in the artifact rather than only in a verdict.
        "compile_logs_read": [name for name, _ in caches],
        "compiles_per_log": {name: len(cache.compile_log()) for name, cache in caches},
        "compiles": len(compiled),
        "distinct_sources_compiled": len(set(compiled)),
        "compiles_from_mutated_source": from_mutated,
        "compiles_from_unmutated_source": sum(1 for digest in compiled
                                              if digest in original),
        "mutated_source_reached_the_compiler": from_mutated > 0,
    }


def compile_to_ptx(code: str, options: Sequence[str], arch: str = "compute_86") -> str:
    """Real PTX text for one kernel, straight from NVRTC.

    ``cupy.cuda.compiler.compile_using_nvrtc(..., arch=None)`` DOES NOT RETURN
    PTX. On CuPy 13.5.1 it returns a 2-tuple whose first element is a CUBIN —
    an ELF binary — so counting ``"fma.rn.f32"`` in ``str()`` of it returns 0
    for every kernel under every compile option, which is exactly what this
    probe reported before this function existed. Zeros that look like a
    measurement are worse than no measurement: they read as "no FMA anywhere",
    which is the answer the guard is supposed to produce, so the broken census
    corroborated the right conclusion for no reason at all.

    NVRTC's own ``getPTX`` is asked for the PTX instead, so the text being
    counted is the text the counts are about.
    """
    from cupy.cuda import nvrtc  # noqa: PLC0415

    program = nvrtc.createProgram(code, "kernel.cu", [], [])
    try:
        nvrtc.compileProgram(program, tuple(options) + (f"--gpu-architecture={arch}",))
        return nvrtc.getPTX(program).decode()
    finally:
        nvrtc.destroyProgram(program)


def run_ptx_probe(module, results: Dict[str, Any]) -> None:
    """Corroborate the FMA measurement by counting fma.rn.f32 in the PTX."""
    try:
        from cupy.cuda import nvrtc  # noqa: PLC0415, F401
    except Exception as exc:  # noqa: BLE001
        results["ptx"] = {"available": False, "error": str(exc)}
        return
    codes = {
        "step_B": module._triple_curl_B_kernel_code,
        "step_B_complex": module._triple_curl_B_kernel_complex_code,
        "step_D": module._triple_curl_D_kernel_code,
        "step_D_complex": module._triple_curl_D_kernel_complex_code,
        # The real-field PML pair. Their recurrence carries TWO more FMA
        # candidates than the plain curls — ``(fu*kms) - curl`` and
        # ``(f*kms_u) + fu_new`` — so the fma.rn.f32 count here is the direct
        # corroboration of why --fmad=false is load-bearing for this slice.
        "step_B_pml_real": module._step_B_pml_real_kernel_code,
        "step_D_pml_real": module._step_D_pml_real_kernel_code,
    }
    out: Dict[str, Any] = {"available": True, "counts": {}}
    for label, options in OPTION_SETS:
        for name, code in codes.items():
            key = f"{label}:{name}"
            try:
                text = compile_to_ptx(code, tuple(options) + (
                    "-I" + os.path.join(
                        os.path.dirname(cp.__file__), "_core", "include"),))
                if ".visible .entry" not in text:
                    raise RuntimeError(
                        "NVRTC returned something that is not PTX (no '.visible "
                        ".entry'); the counts below would be counts in a binary.")
                out["counts"][key] = {
                    "fma_rn_f32": text.count("fma.rn.f32"),
                    # Contractible spellings: these are what the DEFAULT options
                    # emit and what ptxas is free to fuse. Their disappearance
                    # under --fmad=false, in favour of the .rn forms, is the
                    # mechanism the bit-identity result rides on.
                    "mul_f32_contractible": len(re.findall(r"\bmul\.f32\b", text)),
                    "add_f32_contractible": len(re.findall(r"\badd\.f32\b", text)),
                    "sub_f32_contractible": len(re.findall(r"\bsub\.f32\b", text)),
                    "mul_rn_f32": text.count("mul.rn.f32"),
                    "add_rn_f32": text.count("add.rn.f32"),
                    "sub_rn_f32": text.count("sub.rn.f32"),
                    "ptx_lines": text.count("\n"),
                }
                log(f"[ptx] {key}: {out['counts'][key]}")
            except Exception as exc:  # noqa: BLE001
                out["counts"][key] = {"error": f"{type(exc).__name__}: {exc}"[:600]}
                log(f"[ptx] {key}: unavailable ({type(exc).__name__})")
    results["ptx"] = out


def save(results: Dict[str, Any], out_path: str) -> None:
    """Atomic rewrite, with the bytes THIS process imported recorded first.

    THE THIRD WRITER. gate_triton_complex.save and metal_gate_kit.save already
    stamp; this one is bound by gate_cuda_offdiag as ``save = probe.save``, so
    without this the CUDA track wrote verdicts with no record of the source behind
    them while satisfying a ratchet that only checked for an IMPORT of a stamping
    module. All three tracks now answer the same question in the same key.
    """
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(results)

    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False)
    os.replace(tmp, out_path)


def device_info() -> Dict[str, Any]:
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props["name"]
    return {
        "device_index": int(cp.cuda.runtime.getDevice()),
        "device_name": name.decode() if isinstance(name, bytes) else str(name),
        "compute_capability": f"{props['major']}.{props['minor']}",
        "cupy_version": cp.__version__,
        "numpy_version": np.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "nvrtc_version": ".".join(str(v) for v in cp.cuda.nvrtc.getVersion()),
        "python": sys.version.split()[0],
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }


def source_digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def subnormal_policy_stamp(repo_root: str) -> Dict[str, Any]:
    """WHICH float32 subnormal policy these bytes were cut under.

    The 2026-08-09/10 hand-CUDA artifact names none — ``grep -ric`` for
    ``subnormal|ftz|denormal|policy`` over its five files returned 0 on all five,
    and its ``environment`` block has no policy field of any kind. That is the
    live gap in §1.1 of the disposition, and it is a gap in the ARTIFACT before
    it is a gap in the kernel: CuPy appends ``-ftz=true`` to every NVRTC compile
    and ``subnormal_policy`` strips it at the ``compile_using_nvrtc`` seam, so
    the same source compiles to two different binaries and the record could not
    even be filtered by which.

    READ, NOT DRIVEN — driving is :func:`install_subnormal_policy_for_run`, and
    only when ``--subnormal-policy`` asks for it. This says what the process it
    ran in was doing, including "nothing was installed", which is itself the
    answer for a run that took CuPy's default (which is to flush). The stamp carries
    ``policy``, ``resolved``, the ``match_meep`` measurement behind it, the NVRTC
    call and strip counters, and ``CUPY_CACHE_DIR`` — which matters here because
    CuPy's cache key is computed ABOVE the strip seam, so a run sharing a cache
    with the other policy can be served the other policy's bytes.
    """
    record: Dict[str, Any] = {"CUPY_CACHE_DIR": os.environ.get("CUPY_CACHE_DIR", ""),
                              "MEEP_GPU_SUBNORMAL_POLICY":
                                  os.environ.get("MEEP_GPU_SUBNORMAL_POLICY", ""),
                              "CUPY_ACCELERATORS":
                                  os.environ.get("CUPY_ACCELERATORS", "<unset>")}
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from meep_gpu import subnormal_policy  # noqa: PLC0415 - deliberately late
    except Exception as exc:  # noqa: BLE001 - the reason IS the record
        record.update(available=False, policy=None,
                      why=f"meep_gpu.subnormal_policy is not importable from "
                          f"{repo_root!r}: {type(exc).__name__}: {exc}"[:400])
        return record
    record.update(available=True, stamp=subnormal_policy.policy_stamp())
    record["policy"] = record["stamp"]["policy"]
    record["resolved"] = record["stamp"]["resolved"]
    record["installed"] = record["stamp"]["installed"]
    return record


#: Every ``compile_using_nvrtc`` call this process made, innermost: the OPTIONS
#: the real compiler received and the sha256 of the BYTES it handed back.
_NVRTC_OBSERVATIONS: List[Dict[str, Any]] = []


def install_nvrtc_binary_observer() -> Dict[str, Any]:
    """Fingerprint the BYTES NVRTC returns, at the seam the policy acts on.

    WHY THE COMPILE LOG IS NOT ENOUGH. ``compile_cache.compile_log()`` records the
    sha256 of the SOURCE handed to each ``cp.RawKernel`` construction, which
    settles the armed-mutation question (§1.5) and settles nothing about the
    policy: both policies compile the SAME source, so both legs' compile logs
    agree by construction. The §1.2 question is whether two policies produced two
    distinct BINARIES or whether one binary was served twice, and only the
    compiler's own output answers it.

    WHERE THIS SITS, AND WHY THAT IS THE WHOLE DESIGN. It is installed BEFORE
    ``install_subnormal_policy_for_run``, so under ``"keep"`` the policy's strip
    wraps THIS and this becomes the innermost layer — it therefore sees the
    option tuple AFTER ``-ftz=true`` was removed, i.e. what NVRTC was really
    given. Installed afterwards it would sit outside the strip and record the
    pre-strip tuple, which is the one thing that would make the two legs look
    alike. Under ``"flush"`` no strip is installed at all (``subnormal_policy``
    leaves CuPy native, because CuPy already appends ``-ftz=true``), so this is
    the only layer and records that tuple with the ``-ftz=true`` present.

    NOT ``functools.wraps``, DELIBERATELY. ``wraps`` copies ``__dict__``, so a
    wrapper built with it inherits any marker the original carries —
    ``subnormal_policy`` documents that exact mechanism at its own seam
    (``:672-679``). Inheriting ``_POLICY_MARK`` here would make the later strip
    install look already-done and silently leave the run un-policed, which is the
    failure this observer exists to detect rather than cause. Only ``__name__``
    is copied.

    A CUBIN HASH IS EVIDENCE OF DISTINCTNESS, NOT OF A COMPILE. CuPy's disk cache
    is keyed above this seam, so a cache hit never reaches here at all — which is
    why each leg takes a private, policy-token-carrying ``CUPY_CACHE_DIR``
    (§1.2). With that in place a hit is impossible across legs, so two legs
    reporting different digests for one kernel proves two binaries, and two legs
    reporting the SAME digest proves the bytes genuinely do not depend on the
    policy. Both are real results; neither can be read off a source hash.
    """
    record: Dict[str, Any] = {"installed": False}
    try:
        from cupy.cuda import compiler as cupy_compiler  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        record["why"] = f"cupy.cuda.compiler unreachable: {type(exc).__name__}: {exc}"
        return record
    original = getattr(cupy_compiler, "compile_using_nvrtc", None)
    if not callable(original):
        record["why"] = "cupy.cuda.compiler has no compile_using_nvrtc to observe"
        return record

    def observed(source, options=(), *args, **kwargs):
        result = original(source, options, *args, **kwargs)
        # The 2-tuple's first element is the CUBIN (an ELF), not PTX -- the same
        # fact compile_to_ptx() documents. Hash whatever object came back, by
        # its bytes when it has them and by its repr when it does not, so an
        # unexpected return shape is visible rather than silently unhashed.
        payload = result[0] if isinstance(result, tuple) and result else result
        if isinstance(payload, (bytes, bytearray, memoryview)):
            digest = hashlib.sha256(bytes(payload)).hexdigest()
            size = len(bytes(payload))
            kind = "bytes"
        else:
            digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
            size = -1
            kind = type(payload).__name__
        _NVRTC_OBSERVATIONS.append({
            "options_at_nvrtc": list(options),
            "ftz_true_present": any(str(o) in ("-ftz=true", "--ftz=true")
                                    for o in tuple(options)),
            "source_sha256": hashlib.sha256(str(source).encode("utf-8")).hexdigest(),
            "binary_sha256": digest,
            "binary_bytes": size,
            "binary_kind": kind,
        })
        return result

    observed.__name__ = getattr(original, "__name__", "compile_using_nvrtc")
    cupy_compiler.compile_using_nvrtc = observed
    record.update(installed=True,
                  seam="cupy.cuda.compiler.compile_using_nvrtc",
                  layer="innermost (installed before any policy strip)")
    log("[observer] NVRTC binary observer installed at "
        "cupy.cuda.compiler.compile_using_nvrtc")
    return record


def nvrtc_binary_report() -> Dict[str, Any]:
    """The observed compiles, plus the distinct-binary summary a reader wants."""
    observations = [dict(entry) for entry in _NVRTC_OBSERVATIONS]
    distinct_binaries = sorted({entry["binary_sha256"] for entry in observations})
    distinct_sources = sorted({entry["source_sha256"] for entry in observations})
    return {
        "nvrtc_calls_observed": len(observations),
        "distinct_binaries": len(distinct_binaries),
        "distinct_sources": len(distinct_sources),
        "any_ftz_true_reached_nvrtc": any(e["ftz_true_present"] for e in observations),
        "all_ftz_true_reached_nvrtc": bool(observations) and all(
            e["ftz_true_present"] for e in observations),
        # The cross-leg comparison is done by the RUNNER, which is the only thing
        # that sees both policies; this side just has to make it possible.
        "binary_sha256_by_source": {
            source: sorted({e["binary_sha256"] for e in observations
                            if e["source_sha256"] == source})
            for source in distinct_sources},
        "observations": observations,
    }


def import_meep_for_host_policy() -> Dict[str, Any]:
    """Import MEEP so the HOST half of a ``flush`` policy can be attained.

    MEASURED REFUSAL, not a precaution. ``install_subnormal_policy('flush')``
    raises ``SubnormalPolicyUnattainable`` in a process that has not imported
    MEEP, and says why in its own words: ``mp.set_zero_subnormals`` is the only
    exposure of this process's FTZ/DAZ bits the package may use, and
    ``subnormal_policy`` will not import MEEP itself because that would break
    ``meep_gpu``'s no-MEEP-import boundary and initialize MPI as a side effect.
    Its instruction to a caller that needs flush is exactly this: "Import MEEP
    before installing the policy."

    A PARITY PROBE MAY DO WHAT THE PACKAGE MAY NOT. The boundary protects
    ``meep_gpu/**``, which ships; this file is a comparator that already runs
    against MEEP-derived oracles. Doing it here, explicitly and recorded, is what
    lets the flush leg exist at all — the alternative is a run that requested
    flush, was refused, and reported nothing.

    WHY IT IS NOT UNCONDITIONAL. Importing MEEP initializes MPI and moves the
    host FPU, so a ``keep`` leg that imported it would differ from a ``keep`` leg
    that did not for reasons unrelated to the kernel. The caller asks only when
    the RESOLVED policy needs the host to flush.
    """
    record: Dict[str, Any] = {"requested": True}
    try:
        import meep  # noqa: PLC0415 - see docstring; parity probe, not package
    except Exception as exc:  # noqa: BLE001 - a refusal is the result
        record.update(imported=False,
                      why=f"{type(exc).__name__}: {exc}"[:400])
        log(f"[policy] MEEP import FAILED ({type(exc).__name__}); a flush leg "
            f"will be refused by the policy module")
        return record
    record.update(imported=True,
                  meep_version=getattr(meep, "__version__", "<unknown>"),
                  has_set_zero_subnormals=hasattr(meep, "set_zero_subnormals"))
    log(f"[policy] imported MEEP {record['meep_version']} so the host half of a "
        f"flush policy is reachable (set_zero_subnormals="
        f"{record['has_set_zero_subnormals']})")
    return record


def install_subnormal_policy_for_run(policy: str, repo_root: str) -> Dict[str, Any]:
    """DRIVE the float32 subnormal policy, before anything has compiled.

    THE ORDER IS THE WHOLE POINT and it was measured on device with this exact
    kernel: install-then-compile kept a planted ``0x00004000`` in ``fu_Bx``;
    compile-then-install flushed it and stayed flushed even through
    ``_clear_kernel_cache()``. So this runs before the first ``cp.RawKernel``,
    which is why it sits at the top of :func:`main` and not beside the leg that
    wants it.

    ``strict=True`` deliberately: a run that asked to flush and quietly did not
    is a run whose bytes belong to neither policy, and the whole reason this flag
    exists is to be able to write the answer into the artifact. The refusals are
    informative — ``"keep"`` refuses without a ``CUPY_CACHE_DIR`` carrying
    ``ftz_stripped`` (CuPy's cache key is computed above the strip seam, so a
    shared directory serves the other policy's binaries), and ``"flush"`` refuses
    unless CuPy imported with ``CUPY_ACCELERATORS=''``.
    """
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from meep_gpu import subnormal_policy  # noqa: PLC0415 - deliberately late

    report = subnormal_policy.install_subnormal_policy(policy, strict=True)
    log(f"[policy] installed {policy!r} -> resolved={report['resolved']!r} "
        f"stamp={report['policy']!r} unattained={report['unattained']}")
    return report


def write_provenance(directory: str, args, results: Dict[str, Any]) -> str:
    """Everything needed to say WHICH bytes produced this verdict, on which device."""
    path = os.path.join(directory, "PROVENANCE.txt")
    environment = results.get("environment", {})
    lines = [
        f"track                 {args.track}",
        f"module                {os.path.abspath(args.module)}",
        f"module_sha256         {source_digest(args.module)}",
        f"probe                 {os.path.abspath(__file__)}",
        f"probe_sha256          {source_digest(__file__)}",
        f"started_utc           {results.get('started_utc')}",
        f"seed                  {SEED}",
        f"device                {environment.get('device_name')} "
        f"(cc {environment.get('compute_capability')})",
        f"CUDA_VISIBLE_DEVICES  {environment.get('CUDA_VISIBLE_DEVICES')}",
        f"cupy                  {environment.get('cupy_version')}",
        f"numpy                 {environment.get('numpy_version')}",
        f"nvrtc                 {environment.get('nvrtc_version')}",
        f"cuda_runtime          {environment.get('cuda_runtime_version')}",
        f"driver                {environment.get('driver_version')}",
        f"python                {environment.get('python')}",
        f"hostname              {os.uname().nodename}",
        f"source_mutation       {args.source_mutation or '<none>'}",
        f"host_mutation         {args.host_mutation or '<none>'}",
        f"comparator            {'np.allclose' if args.allclose else 'bytes'}",
        f"compare_fu            {not args.no_fu_compare}",
        f"subnormal_policy      "
        f"{(results.get('subnormal_policy') or {}).get('policy') or '<unavailable>'}",
        f"subnormal_resolved    "
        f"{(results.get('subnormal_policy') or {}).get('resolved') or '<unavailable>'}",
        f"CUPY_CACHE_DIR        "
        f"{(results.get('subnormal_policy') or {}).get('CUPY_CACHE_DIR') or '<unset>'}",
        f"pml_value_classes     {','.join(results.get('pml_value_classes', ()))}",
        f"pml_multi_step_budget {results.get('pml_multi_step_budget')}",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", "--kernels", dest="module", required=True,
                        help="path to the track's kernel module "
                             "(hand: meep_gpu/cuda_kernels/step_curl_kernels.py)")
    parser.add_argument("--track", default="hand",
                        help="which track's adapter to drive; 'hand' is built in, "
                             "any other name requires module.PROBE_ADAPTER")
    parser.add_argument("--out", required=True, help="JSON artifact path")
    parser.add_argument("--experiments", default="pml",
                        help="comma-separated: pml, multistep, constitutive, "
                             "constitutive_multistep, dispersive, "
                             "dispersive_multistep, fused_pair, "
                             "fused_pair_multistep, wholestep, legacy, ptx, "
                             "compile, all")
    parser.add_argument("--coefficients", default="synthetic,real",
                        help="comma-separated PML coefficient sources")
    parser.add_argument("--shapes", default="13,11,9;32,32,32",
                        help="semicolon-separated nx,ny,nz grids (legacy experiment only)")
    parser.add_argument("--repo-root", default=None,
                        help="apps/api, for the real-PML leg's meep_gpu import")
    parser.add_argument("--source-mutation", default=None,
                        choices=sorted(set(SOURCE_MUTATIONS)
                                       | set(CONSTITUTIVE_SOURCE_MUTATIONS)
                                       | set(FUSED_PAIR_SOURCE_MUTATIONS)
                                       | set(DISPERSIVE_SOURCE_MUTATIONS)),
                        help="rewrite the kernel source before compiling; the gate "
                             "must then FAIL")
    parser.add_argument("--host-mutation", default=None, choices=HOST_MUTATIONS,
                        help="feed the kernel a wrong table or boundary code; the "
                             "gate must then FAIL")
    parser.add_argument("--whole-step-mutation", default=None,
                        choices=WHOLE_STEP_MUTATIONS,
                        help="break the COMPOSITION rather than a kernel; the "
                             "whole-step leg must then FAIL")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="device index for the whole-step leg's drivers")
    parser.add_argument("--fuse", action="store_true",
                        help="ask plan_step for the cross-sub-step fused pair; off "
                             "by default because the unfused composition is the "
                             "control this round is measured against")
    parser.add_argument("--num-warps", type=int, default=None,
                        help="warps per fused-pair program. CHANGES THE COMPILED "
                             "KERNEL, so it is a gate axis: a value that was never "
                             "swept is a value the byte gate never certified. "
                             "Default: Triton's own.")
    parser.add_argument("--seed-auxiliaries", action="store_true",
                        help="seed fu_B* as well as the primaries in the whole-step "
                             "leg. MEASURED necessity: with fu_B* at zero a "
                             "metallic wall's B plane is identically zero for every "
                             "step, so zero_metal_B is a no-op and dropping it is "
                             "60/60 UNCAUGHT.")
    parser.add_argument("--whole-steps", type=int, default=WHOLE_STEP_COUNT,
                        help="consecutive driver steps in the whole-step leg. This "
                             "is a COVERAGE AXIS: 14.5 measured a divergence that "
                             "8, 10 and 6 steps all passed and 40 did not.")
    parser.add_argument("--pml-steps", type=int, default=PML_MULTI_STEP_COUNT,
                        help="consecutive launches in the real-field PML curl "
                             "multi-step leg. A COVERAGE AXIS, not a runtime "
                             "knob: the sibling track measured 8, 10 and 6 steps "
                             "all passing a divergence 40 caught, and 60 is the "
                             "budget its record certifies. The 2026-08-09/10 CUDA "
                             "record was cut at 8.")
    parser.add_argument("--pml-value-classes", default=",".join(PML_VALUE_CLASSES),
                        help="comma-separated operand classes for the real-PML "
                             "leg: 'uniform' reaches normal numbers only and is "
                             "the class the verdict gates on; 'subnormal_band' "
                             "seeds the band and injects signed zeros, and is "
                             "the only class any float32 subnormal policy is "
                             "visible in. Reported separately, never folded.")
    parser.add_argument("--dispersive-steps", type=int, default=MULTI_STEP_COUNT,
                        help="consecutive launches in the dispersive multi-step "
                             "leg. A COVERAGE AXIS, not a runtime knob: 8, 10 and "
                             "60 steps all passed for the existing kernels and 40 "
                             "and 67 did not (plan 14.5, 16).")
    parser.add_argument("--allclose", action="store_true",
                        help="harness mutation 1: magnitude comparison in place of "
                             "the bytes")
    parser.add_argument("--no-fu-compare", action="store_true",
                        help="harness mutation 8: compare only the target field, "
                             "not the auxiliary")
    parser.add_argument("--subnormal-policy", default=None,
                        choices=("flush", "keep", "match_meep"),
                        help="INSTALL this float32 subnormal policy before the "
                             "first compile, and stamp the artifact with it. "
                             "Omitted, nothing is installed and the artifact "
                             "records that — which is what the 2026-08-09/10 "
                             "record silently was.")
    parser.add_argument("--import-meep-for-host-policy", action="store_true",
                        help="import MEEP before installing the policy, so the "
                             "HOST half of 'flush' is attainable. Measured: "
                             "without it install_subnormal_policy('flush') "
                             "RAISES, because mp.set_zero_subnormals is the only "
                             "route to this process's FTZ/DAZ bits and the "
                             "package may not import MEEP itself. Off by "
                             "default: importing MEEP initializes MPI and moves "
                             "the host FPU, which a 'keep' leg has no reason to "
                             "do.")
    parser.add_argument("--guard", default=None,
                        help="run only this guard set (default: all of them)")
    parser.add_argument("--label", default="gate",
                        help="key this run is filed under in the artifact")
    args = parser.parse_args(argv)

    if cp is None:
        raise SystemExit("cupy is not importable; every experiment here needs a device.")

    experiments = {name.strip() for name in args.experiments.split(",") if name.strip()}
    if "all" in experiments:
        experiments = {"pml", "multistep", "constitutive", "constitutive_multistep",
                       "dispersive", "dispersive_multistep",
                       "fused_pair", "fused_pair_multistep",
                       "wholestep", "legacy", "ptx", "compile"}
    repo_root = args.repo_root or os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

    shapes = tuple(
        tuple(int(v) for v in group.split(","))  # type: ignore[misc]
        for group in args.shapes.split(";") if group.strip())
    # 0.5 is exact in float32; 0.34999999999999998 (the double nearest 0.35) is
    # not, so the pair separates scalar rounding from the arithmetic questions.
    dtdx_values = (0.5, 0.35)

    value_classes = tuple(name.strip() for name in args.pml_value_classes.split(",")
                          if name.strip())
    unknown = [name for name in value_classes if name not in PML_VALUE_CLASSES]
    if unknown:
        raise SystemExit(f"--pml-value-classes: {unknown} not in {PML_VALUE_CLASSES}")

    # BEFORE the policy, so the policy's strip wraps THIS and the options
    # recorded are the ones NVRTC really got — see install_nvrtc_binary_observer.
    binary_observer = install_nvrtc_binary_observer()

    # BEFORE the policy too: the host FPU knob it needs lives in MEEP.
    meep_import = (import_meep_for_host_policy()
                   if args.import_meep_for_host_policy else
                   {"requested": False})

    # BEFORE the first compile — see install_subnormal_policy_for_run.
    policy_install = (install_subnormal_policy_for_run(args.subnormal_policy, repo_root)
                      if args.subnormal_policy else None)

    started = time.time()
    results: Dict[str, Any] = {
        "probe": "fused_kernel_bit_identity",
        "track": args.track,
        "seed": SEED,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kernel_source": os.path.abspath(args.module),
        "kernel_source_sha256": source_digest(args.module),
        "experiments": sorted(experiments),
        "shapes": [list(s) for s in shapes],
        "pml_shapes": [list(s) for s in PML_SHAPES],
        "pml_boundary_sets": [list(b) for b in PML_BOUNDARY_SETS],
        "pml_value_classes": list(value_classes),
        "pml_multi_step_budget": int(args.pml_steps),
        "dtdx_values": [repr(v) for v in dtdx_values],
        "mutation": {"source": args.source_mutation, "host": args.host_mutation,
                     "whole_step": args.whole_step_mutation,
                     "comparator": "np.allclose" if args.allclose else "bytes",
                     "compare_fu": not args.no_fu_compare,
                     "must_be_uncaught": args.source_mutation in
                                        (FUSED_PAIR_NULL_MUTATIONS
                                         + DISPERSIVE_NULL_MUTATIONS
                                         + PML_NULL_MUTATIONS)},
        "fuse": bool(args.fuse),
        "num_warps": args.num_warps,
        "whole_steps": int(args.whole_steps),
        "dispersive_steps": int(args.dispersive_steps),
        "dispersive_pole_counts": [list(c) for c in DISPERSIVE_POLE_COUNTS],
        "dispersive_value_classes": list(DISPERSIVE_VALUE_CLASSES),
        "seed_auxiliaries": bool(args.seed_auxiliaries),
        "environment": device_info(),
        "subnormal_policy_requested": args.subnormal_policy,
        "subnormal_policy_install": policy_install,
        "subnormal_policy": subnormal_policy_stamp(repo_root),
        "meep_import_for_host_policy": meep_import,
        "nvrtc_binary_observer": binary_observer,
    }
    log(f"[env] {json.dumps(results['environment'])}")
    log(f"[policy] {json.dumps({k: v for k, v in results['subnormal_policy'].items() if k != 'stamp'})}")
    out_directory = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_directory, exist_ok=True)
    save(results, args.out)
    write_provenance(out_directory, args, results)

    module = load_kernel_module(args.module)
    adapter = build_adapter(args.track, module)
    results["guard_sets"] = [{"label": label, "guard": list(guard),
                              "primary": primary}
                             for label, guard, primary in adapter.guard_sets]
    results["guard_description"] = adapter.guard_description

    armed: Optional[Dict[str, Any]] = None
    if args.source_mutation:
        armed = apply_source_mutation(adapter, args.source_mutation)
        results["mutation"]["armed"] = armed
        results["mutation"]["sites"] = armed["sites"]
        log(f"[mutate] source mutation {args.source_mutation!r} applied at "
            f"{armed['sites']} ({armed['total_sites']} sites across "
            f"{len(armed['attributes_rewritten'])} kernel source(s))")
    comparator = allclose_compare if args.allclose else bit_compare
    compare_fu = not args.no_fu_compare

    if "compile" in experiments:
        run_compilation(module, results, args.out)
        run_scalar_dtype_probe(shapes[0], results, dtdx_values)
        save(results, args.out)
    if "pml" in experiments:
        summary = run_pml_bit_identity(
            adapter, results, args.out, repo_root,
            [c.strip() for c in args.coefficients.split(",") if c.strip()],
            compare_fu=compare_fu, comparator=comparator,
            guard_filter=args.guard,
            host_mutation=args.host_mutation, label=args.label,
            value_classes=value_classes)
        log(f"[pml:{args.label}] primary guard, NORMAL NUMBERS: "
            f"{summary['primary_identical']}/{summary['primary_ran']} bit-identical; "
            f"subnormal band: {summary['subnormal_band_identical']}/"
            f"{summary['subnormal_band_ran']}")
    if "multistep" in experiments:
        summary = run_pml_multi_step(
            adapter, results, args.out, repo_root, compare_fu=compare_fu,
            comparator=comparator, host_mutation=args.host_mutation,
            label=args.label, steps=int(args.pml_steps))
        log(f"[pml:{args.label}:multi] all {args.pml_steps}-step runs identical: "
            f"{summary['all_steps_identical']}")
    if "constitutive" in experiments:
        summary = run_constitutive_bit_identity(
            adapter, results, args.out, repo_root,
            [c.strip() for c in args.coefficients.split(",") if c.strip()],
            compare_fw=compare_fu, comparator=comparator,
            guard_filter=args.guard, host_mutation=args.host_mutation,
            label=args.label)
        # The band count beside the pass, never folded into it — the same shape the
        # curl leg prints, and the reason this leg is worth running twice.
        log(f"[const:{args.label}] primary guard, NORMAL NUMBERS: "
            f"{summary['primary_identical']}/{summary['primary_ran']} bit-identical; "
            f"subnormal band: {summary['subnormal_band_identical']}/"
            f"{summary['subnormal_band_ran']} "
            f"(non-vacuous: {summary['subnormal_band_is_non_vacuous']})")
    if "constitutive_multistep" in experiments:
        summary = run_constitutive_multi_step(
            adapter, results, args.out, compare_fw=compare_fu,
            comparator=comparator, host_mutation=args.host_mutation,
            label=args.label)
        # The BUDGET comes off the summary, not off a constant this line happens to
        # name: it used to print MULTI_STEP_COUNT (8) while the leg had its own.
        log(f"[const:{args.label}:multi] all {summary['steps_budget']}-step runs "
            f"identical: {summary['all_steps_identical']}; auxiliary advanced past "
            f"the first launch: "
            f"{summary['auxiliary_advanced_past_the_first_step']}")
    if "dispersive" in experiments:
        summary = run_dispersive_constitutive_bit_identity(
            adapter, results, args.out, repo_root,
            [c.strip() for c in args.coefficients.split(",") if c.strip()],
            compare_fw=compare_fu, comparator=comparator,
            guard_filter=args.guard, host_mutation=args.host_mutation,
            label=args.label)
        per_class = summary.get("per_value_class", {})
        log(f"[disp:{args.label}] primary guard: "
            f"{summary['primary_identical']}/{summary['primary_ran']} bit-identical "
            f"OVERALL; by value class: " + "; ".join(
                f"{name} {bucket['identical']}/{bucket['ran']}"
                for name, bucket in sorted(per_class.items())))
    if "dispersive_multistep" in experiments:
        summary = run_dispersive_constitutive_multi_step(
            adapter, results, args.out, compare_fw=compare_fu,
            comparator=comparator, host_mutation=args.host_mutation,
            steps=args.dispersive_steps, label=args.label)
        log(f"[disp:{args.label}:multi] all {args.dispersive_steps}-step runs "
            f"identical: {summary['all_steps_identical']} "
            f"(normal numbers only: "
            f"{summary['normal_numbers_all_steps_identical']})")
    if "fused_pair" in experiments:
        summary = run_fused_pair_bit_identity(
            adapter, results, args.out, repo_root,
            [c.strip() for c in args.coefficients.split(",") if c.strip()],
            compare_aux=compare_fu, comparator=comparator,
            guard_filter=args.guard, host_mutation=args.host_mutation,
            label=args.label, num_warps=args.num_warps)
        log(f"[fused:{args.label}] primary guard vs the ARRAY PATH: "
            f"{summary['primary_identical']}/{summary['primary_ran']} bit-identical; "
            f"vs the two separate kernels: "
            f"{summary['primary_matches_two_kernels']}/{summary['primary_ran']}; "
            f"the unfused control vs the array path: "
            f"{summary['primary_two_kernels_vs_array']}/{summary['primary_ran']}")
    if "fused_pair_multistep" in experiments:
        summary = run_fused_pair_multi_step(
            adapter, results, args.out, compare_aux=compare_fu,
            comparator=comparator, host_mutation=args.host_mutation,
            label=args.label, num_warps=args.num_warps)
        log(f"[fused:{args.label}:multi] all {MULTI_STEP_COUNT}-step runs identical: "
            f"{summary['all_steps_identical']}")
    if "wholestep" in experiments:
        summary = run_whole_step(results, args.out, repo_root, gpu_id=args.gpu_id,
                                 mutation=args.whole_step_mutation,
                                 label=args.label, fuse=args.fuse,
                                 steps_budget=int(args.whole_steps),
                                 seed_auxiliaries=args.seed_auxiliaries,
                                 adapter=adapter, num_warps=args.num_warps)
        log(f"[whole:{args.label}] all {args.whole_steps}-step runs identical: "
            f"{summary['all_steps_identical']}")
    if "legacy" in experiments:
        run_bit_identity(module, results, args.out, shapes, dtdx_values)
    if "ptx" in experiments:
        run_ptx_probe(module, results)

    results["mutation"]["armed_accounting"] = armed_mutation_accounting(adapter, armed)
    log(f"[mutate] armed accounting: "
        f"{json.dumps(results['mutation']['armed_accounting'])}")
    results["subnormal_policy"] = subnormal_policy_stamp(repo_root)
    # The §1.2 evidence: which BYTES this leg's policy actually produced. Written
    # after every leg has run, so it covers every compile the run took.
    results["nvrtc_binaries"] = nvrtc_binary_report()
    log(f"[observer] nvrtc calls={results['nvrtc_binaries']['nvrtc_calls_observed']} "
        f"distinct_binaries={results['nvrtc_binaries']['distinct_binaries']} "
        f"distinct_sources={results['nvrtc_binaries']['distinct_sources']} "
        f"ftz_true_reached_nvrtc="
        f"{results['nvrtc_binaries']['any_ftz_true_reached_nvrtc']}")
    results["verdict"] = pml_verdict(results)
    results["elapsed_seconds"] = round(time.time() - started, 2)
    save(results, args.out)
    write_provenance(out_directory, args, results)
    log(f"[done] {results['elapsed_seconds']} s -> {args.out} | "
        f"verdict={json.dumps(results['verdict'])}")
    return 0


def pml_verdict(results: Dict[str, Any]) -> Dict[str, Any]:
    """PASS only when EVERY leg that ran is identical — reported leg by leg.

    A single boolean over five legs would hide which one carried it, so the counts
    stay separate and ``pass`` is their conjunction over the legs present.
    """
    def leg(section: str) -> Dict[str, Any]:
        block = results.get(section, {})
        label = None
        for key in block:
            label = key
        return block.get(label, {}) if label else {}

    single = leg("pml_bit_identity")
    multi = leg("pml_multi_step")
    constitutive = leg("constitutive_bit_identity")
    constitutive_multi = leg("constitutive_multi_step")
    fused = leg("fused_pair_bit_identity")
    fused_multi = leg("fused_pair_multi_step")
    dispersive = leg("dispersive_bit_identity")
    dispersive_multi = leg("dispersive_multi_step")
    whole = leg("whole_step")

    checks = []
    for summary, key in ((single, "primary_pass"), (constitutive, "primary_pass"),
                         (fused, "primary_pass")):
        if summary:
            checks.append(bool(summary.get(key)))
    for summary in (multi, constitutive_multi, fused_multi, whole):
        if summary:
            checks.append(bool(summary.get("all_steps_identical")))
    # THE CONSTITUTIVE MULTI-STEP LEG'S OWN VACUITY FLOOR. ``fw[i] = src`` is
    # unconditional, so a leg whose source never moves drives ``f_w`` to a fixed
    # point after launch one and its remaining launches add no state — measured:
    # ``f_w`` changed against the previous step at step 1 only, and equalled the
    # source exactly at every step. "Identical for 60 steps" over that is a claim
    # about one step repeated 60 times, so the leg must show the auxiliary
    # advancing past the first launch before its identity counts.
    if constitutive_multi:
        checks.append(bool(constitutive_multi.get(
            "auxiliary_advanced_past_the_first_step")))
    # THE DISPERSIVE LEG'S PASS IS THE NORMAL-NUMBER ONE, AND THE VERDICT SAYS SO.
    # Its seed carries a deliberate cancellation class that lands in the subnormal
    # band, where the ARRAY PATH — not the kernel — leaves IEEE: CuPy flushes a
    # subnormal result to zero and NumPy and Triton both keep it (measured, this
    # round, per operation). Folding that class into one boolean would report a
    # kernel defect that is not there; dropping it would report a byte identity the
    # run does not have. So both counts are published and only the normal-number one
    # gates.
    if dispersive:
        checks.append(bool(dispersive.get("normal_numbers_pass")))
    if dispersive_multi:
        checks.append(bool(dispersive_multi.get(
            "normal_numbers_all_steps_identical")))
    label = None
    for section in ("pml_bit_identity", "constitutive_bit_identity",
                    "fused_pair_bit_identity", "whole_step"):
        for key in results.get(section, {}):
            label = key
    return {
        "label": label,
        "subnormal_policy": (results.get("subnormal_policy") or {}).get("policy"),
        "curl_single_launch": (f"{single.get('primary_identical', 0)}/"
                               f"{single.get('primary_ran', 0)}") if single else None,
        "_curl_single_launch_is_the_normal_number_class": True,
        "curl_single_launch_subnormal_band": (
            f"{single.get('subnormal_band_identical', 0)}/"
            f"{single.get('subnormal_band_ran', 0)}") if single else None,
        "curl_multi_step_budget": multi.get("steps_budget"),
        "curl_multi_step_all_identical": multi.get("all_steps_identical"),
        "constitutive_single_launch": (
            f"{constitutive.get('primary_identical', 0)}/"
            f"{constitutive.get('primary_ran', 0)}") if constitutive else None,
        "_constitutive_single_launch_is_the_normal_number_class": True,
        # THE BAND COUNT, published beside the pass and not folded into it — the
        # same split the curl and dispersive legs carry, for the same measured
        # reason (in the band it is the ARRAY PATH that leaves IEEE). Before this
        # existed the constitutive leg had no value-class axis at all, so the
        # two-policy run the gate prescribes was identical by construction.
        "constitutive_single_launch_subnormal_band": (
            f"{constitutive.get('subnormal_band_identical', 0)}/"
            f"{constitutive.get('subnormal_band_ran', 0)}") if constitutive else None,
        "constitutive_single_launch_band_is_non_vacuous":
            constitutive.get("subnormal_band_is_non_vacuous"),
        "constitutive_multi_step_budget": constitutive_multi.get("steps_budget"),
        "constitutive_multi_step_all_identical":
            constitutive_multi.get("all_steps_identical"),
        "constitutive_multi_step_auxiliary_advanced":
            constitutive_multi.get("auxiliary_advanced_past_the_first_step"),
        "fused_pair_single_launch_vs_array": (
            f"{fused.get('primary_identical', 0)}/"
            f"{fused.get('primary_ran', 0)}") if fused else None,
        "fused_pair_single_launch_vs_two_kernels": (
            f"{fused.get('primary_matches_two_kernels', 0)}/"
            f"{fused.get('primary_ran', 0)}") if fused else None,
        "unfused_control_vs_array": (
            f"{fused.get('primary_two_kernels_vs_array', 0)}/"
            f"{fused.get('primary_ran', 0)}") if fused else None,
        "fused_pair_multi_step_all_identical":
            fused_multi.get("all_steps_identical"),
        "dispersive_single_launch_normal_numbers": (
            f"{dispersive.get('per_value_class', {}).get('uniform', {}).get('identical', 0)}/"
            f"{dispersive.get('per_value_class', {}).get('uniform', {}).get('ran', 0)}"
        ) if dispersive else None,
        "dispersive_single_launch_subnormal_band": (
            f"{dispersive.get('per_value_class', {}).get('cancellation', {}).get('identical', 0)}/"
            f"{dispersive.get('per_value_class', {}).get('cancellation', {}).get('ran', 0)}"
        ) if dispersive else None,
        "dispersive_multi_step_normal_numbers_all_identical":
            dispersive_multi.get("normal_numbers_all_steps_identical"),
        "whole_step_all_identical": whole.get("all_steps_identical"),
        "whole_step_replaces": [r.get("replaces") for r in whole.get("runs", [])],
        "pass": bool(checks) and all(checks),
    }


if __name__ == "__main__":
    raise SystemExit(main())
