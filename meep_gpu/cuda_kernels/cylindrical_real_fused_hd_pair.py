"""The REAL m = 0 Dcyl H->D product: ``update_H`` into scratch, the radial scan over the new
Hp, the rotation, the cylindrical ``step_D`` curl -- three launches per seam.

THE CELL: ``H_to_D (cuda_constitutive/ordinary, cuda_cylindrical/cylindrical)`` -- 3
seam-instances, ``tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_{0_0,1_0}``
and ``tests:TestPMLCylindrical.test_pml_cyl_0_0_0``, all ``pml_active``
(``parity/meep_gpu/results/fusion_matrix_cuda_2026-09-06_hd/fusion_matrix_cuda.json``,
``h_to_d_seam.instances``).

WHAT THE THREE LAUNCHES ARE. Launch 1 (slot ``update_H``, :data:`KERNEL_NAME`) steps
``update_H`` at the thread's own cell into launch-local SCRATCH through
:mod:`.fused_hd_pair`'s pure cell function -- the root ``update_H_pml_real`` body lifted,
the chunk record decoded at the cell, imported and never retyped -- so H, ``f_w_H`` and B
are const for the whole launch. Launch 2 is the column-serial radial scan
(:mod:`.cylindrical_prefix`) over the SCRATCH Hy: MEEP's per-chunk ``f_rderiv_int`` of the
Hp ``update_H`` just produced, its difference written to a target-shaped buffer. Then the
H/``f_w_H`` bindings rotate against the scratch (``fused_hd_pair.rotate_into_fields``), and
launch 3 is ``cylindrical_kernels``' ``cyl_step_D_pml_real`` over the rotated H with the
difference handed in.

THE RUN'S FIRST ``update_H`` STARTS H FROM B (``stepping.separate_h_from_b``): B is final
at this seam, so the start runs on the array path BEFORE launch 1, with the layer -- the
:mod:`.fused_hd_pair` arrangement.

THE LAUNCH ALGEBRA, AND THE INSTALLATION VERDICT. Under the per-chunk radial sum the D
side's own baseline is three launches -- ``update_H``, the scan, ``step_D`` -- because the
scan computes its increments from Hp itself. The 0.9.2 product fused the pre-``cumsum``
increment into launch 1 and saved four launches against a seven-launch baseline; that
saving is gone with the elementwise passes it removed. This product therefore saves NO
launch, and it ships ``INSTALLABLE = False`` (:data:`INSTALLABLE_REASON`). A two-launch
shape (``update_H`` row by row inside the column-serial scan) exists and is an owner's
design ruling, not this module's to make.

The seam's electric withdraw is not carried (:data:`HOISTS_THE_WITHDRAW`): a row with a
standing integrated electric withdraw is refused by name. Nothing is injected in this
seam (:data:`CARRIES_DEPOSIT_REPAIR` is False).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import covers_real_pml_constitutive
from .cylindrical_coverage import (covers_real_pml_cylindrical_curl,
                                   cylindrical_boundary_codes)
from .cylindrical_prefix import (PREFIX_COMPONENT, get_radial_scan_kernel,
                                 launch_radial_scan, radial_scan_device_tables,
                                 radial_scan_tables)
from . import fused_hd_pair as _hd
from .. import withdraw_hoist as _withdraw_hoist

# The halves' device modules. Taken DEFENSIVELY: both import CuPy at scope, and the
# PREDICATE half of this family must answer on a host with no device (the census asks
# it there). The emitter and the launcher refuse BY NAME where they are absent.
try:
    from . import constitutive_kernels
except Exception:  # noqa: BLE001 - no CuPy on this host
    constitutive_kernels = None  # type: ignore[assignment]
try:
    from . import cylindrical_kernels
except Exception:  # noqa: BLE001
    cylindrical_kernels = None  # type: ignore[assignment]

FAMILY = "cuda_cylindrical_real_fused_hd_pair"

#: Launch 1's entry-point symbol, spelled once and NOT either half's name.
KERNEL_NAME = "update_H_scratch_pml_cyl_real"

#: Launch 3 is the cylindrical curl, by its own name, launched through its own module.
CURL_KERNEL_NAME = "cyl_step_D_pml_real"

#: The sub-step slot this product is registered on: the FIRST half of the seam.
SLOT = "update_H"

#: The driver passes the product's launches perform, in driver order. The withdraw
#: between them is NOT carried (:data:`HOISTS_THE_WITHDRAW`).
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``fused_pairs.FUSED_PAIR_SEAMS`` files this row under.
SEAM: str = _withdraw_hoist.SEAM

#: The volumes launch 1 writes to scratch and the launcher rotates afterwards -- the
#: :mod:`.fused_hd_pair` six, imported so the two products cannot disagree.
SCRATCH_VOLUMES: Tuple[str, ...] = _hd.SCRATCH_VOLUMES
H_TARGETS: Tuple[str, str, str] = _hd.H_TARGETS

#: The component the D-side radial sum runs over (``Hy``), read off the shared table.
PREFIX_SOURCE: str = PREFIX_COMPONENT["step_D"]

#: Launches per complete seam: launch 1, the scan, launch 3. THE SCAN IS COUNTED.
LAUNCHES_PER_RUN = 3

#: The entry point is NEW with the per-chunk radial sum and no record names it yet: its
#: 0.9.2 predecessor (``update_H_increment_pml_cyl_real``, which fused the pre-cumsum
#: increment) is gone, and the record that certified it is re-cut in the combined
#: certification round (bit_parity_0_10 CERT-3).
CERTIFIED_KERNELS = ()

#: Spelled as a dict WITHOUT an annotation (the partition reader matches ``ast.Assign``).
UNCERTIFIED_KERNELS = {
    "update_H_scratch_pml_cyl_real":
        "gated by parity/meep_gpu/gate_cuda_cylindrical_real_fused_hd_pair.py under both "
        "float32 subnormal policies; the certification block that would move this name "
        "into CERTIFIED_KERNELS is cut from that gate's artifacts in the certification "
        "round",
}

#: Nothing is injected between the two consults; ``deposit_repair`` is not consulted.
CARRIES_DEPOSIT_REPAIR = False

#: The seam's electric withdraw is NOT performed by this product; rows with a standing
#: one are refused by name.
HOISTS_THE_WITHDRAW = False

#: The composer may not install this product.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "no launch saved. Under MEEP's per-chunk radial sum the D side's own baseline is "
    "three launches (update_H, the column-serial scan, step_D), because the scan forms "
    "its increments from Hp itself; this product is also three (update_H into scratch, "
    "the scan over the scratch Hp, step_D over the rotated H). The 0.9.2 product saved "
    "four launches against a seven-launch baseline by fusing the pre-cumsum increment, "
    "and those elementwise passes no longer exist. A two-launch shape -- update_H row "
    "by row inside the column-serial scan -- is an owner's design ruling. Until then "
    "INSTALLABLE stays False and the board credits the cell by predicate admission, as "
    "fused_hd_pair is credited")

WHAT_A_RELEASE_DOES_NOT_LICENSE = (
    "SERVED on the fusion board is PREDICATE ADMISSION by the board's own definition, "
    "so a released product credits its admitted seam-instances while executing "
    "NOWHERE: this family declares INSTALLABLE = False so the composer refuses to "
    "install it on every configuration, and no timing of any kind has been taken of "
    "this shape. The launch figures in this module are COUNTS.")

#: Lanes per block, the halves' own.
_FUSED_THREADS = 256

#: NVRTC options. ``--fmad=false`` is CORRECTNESS: the constitutive accumulation is a
#: contraction candidate the array path rounds twice.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "CURL_KERNEL_NAME", "FAMILY",
    "HOISTS_THE_WITHDRAW", "H_TARGETS", "INSTALLABLE", "INSTALLABLE_REASON",
    "KERNEL_NAME", "LAUNCHES_PER_RUN", "PREFIX_SOURCE", "REPLACES", "SCRATCH_VOLUMES",
    "SEAM", "SLOT", "UNCERTIFIED_KERNELS", "WHAT_A_RELEASE_DOES_NOT_LICENSE",
    "assert_bindings_are_disjoint", "covers_cylindrical_real_fused_hd_pair",
    "device_sources", "kernel_source", "launch_curl", "launch_update_H",
    "resolve", "run_cylindrical_real_fused_hd_pair", "scan_and_rotate", "signature",
    "source_digest",
]


# ---------------------------------------------------------------------------
# The device source
# ---------------------------------------------------------------------------

def _certified() -> None:
    if constitutive_kernels is None or cylindrical_kernels is None:
        raise RuntimeError(
            "constitutive_kernels / cylindrical_kernels are not importable on this host "
            "(they import CuPy at module scope), so there is no text to splice and no "
            "curl to launch. covers_cylindrical_real_fused_hd_pair needs neither and "
            "still answers")


_SIGNATURE = '''
extern "C" __global__ void update_H_scratch_pml_cyl_real(
    // THE SCRATCH OUTPUTS: a DIFFERENT allocation from every pre-launch volume below
    // (assert_bindings_are_disjoint checks it by base address before every launch).
    float* __restrict__ Hx_out, float* __restrict__ Hy_out,
    float* __restrict__ Hz_out,
    float* __restrict__ f_w_Hx_out, float* __restrict__ f_w_Hy_out,
    float* __restrict__ f_w_Hz_out,
    // THE PRE-LAUNCH STATE, read and written by none.
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    const float* __restrict__ f_w_Hx, const float* __restrict__ f_w_Hy,
    const float* __restrict__ f_w_Hz,
    const float* __restrict__ Bx, const float* __restrict__ By,
    const float* __restrict__ Bz,
    // update_H's INTEGER-sub-lattice coefficients.
    const float* __restrict__ kps_x, const float* __restrict__ kps_y,
    const float* __restrict__ kps_z,
    const float* __restrict__ kms_x, const float* __restrict__ kms_y,
    const float* __restrict__ kms_z,
    int nx, int ny, int nz,
    // THE CHUNK RECORD (chunk_map.chunk_bounds_record), int32[25], decoded by the
    // lifted body at the cell it is evaluated for.
    const int* __restrict__ cb
) {
'''


def signature() -> str:
    """Launch 1's parameter list. Every pointer this launch writes is a scratch twin."""
    return _SIGNATURE


def kernel_source() -> str:
    """The whole device source of launch 1: :mod:`.fused_hd_pair`'s pure constitutive
    prelude, argument pack and cell function, imported, and the scratch stores."""
    _certified()
    body = "\n".join([
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (idx >= nx * ny * nz) return;",
        "",
        _hd.weld_args_construction().rstrip("\n"),
        "",
        "    // --- update_H at the thread's OWN cell, into registers, stored to SCRATCH.",
        "    // Nothing written here is read by this launch; the launcher scans the",
        "    // scratch Hy and rotates H/f_w_H against the scratch afterwards.",
        "    float own_h[3];",
        "    float own_w[3];",
        "    raw_update_H_cell(idx, weld, own_h, own_w);",
        "    Hx_out[idx] = own_h[0]; Hy_out[idx] = own_h[1]; Hz_out[idx] = own_h[2];",
        "    f_w_Hx_out[idx] = own_w[0]; f_w_Hy_out[idx] = own_w[1];",
        "    f_w_Hz_out[idx] = own_w[2];",
        "",
    ])
    source = (_hd.constitutive_prelude() + _hd.weld_args_struct()
              + _hd.raw_update_H_cell_source() + signature() + body + "}\n")
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """``{kernel name: source}`` for launch 1. Launch 3's text is ``cylindrical_kernels``'
    and the scan's ``cylindrical_prefix``'s."""
    return {KERNEL_NAME: kernel_source()}


def source_digest() -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(kernel_source().encode("utf-8")).hexdigest()


def _get_kernel(source: Optional[str] = None):
    """Compile on first use, memoized on (name, options, policy, source)."""
    import cupy as cp  # noqa: PLC0415

    code = kernel_source() if source is None else source
    key = _kernel_cache_key(KERNEL_NAME, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def covers_cylindrical_real_fused_hd_pair(fields: Any, pml: Any, grid: Any,
                                          sources: Any = None) -> Tuple[bool, str]:
    """May this product span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    A CONJUNCTION, nothing weakened: the certified ordinary constitutive predicate on
    the H side (asked FIRST, the driver's order), the certified real Dcyl curl predicate
    on ``step_D`` (which REQUIRES a real m = 0 Dcyl grid under an active PML), the
    seam's withdraw clause through :mod:`..withdraw_hoist`, the r-axis boundary
    resolution, and the rotation's requirement that the six magnetic volumes exist.
    """
    covered, reason = covers_real_pml_constitutive(fields, pml, grid, "H")
    if not covered:
        return False, f"constitutive half: {reason}"
    covered, reason = covers_real_pml_cylindrical_curl(fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"

    # THE SEAM'S ONE PASS. IGNORANCE IS NEVER AN EMPTY SET: ``Fields`` does not hold
    # the source list, so an undeclared set is a refusal, not an empty seam.
    seam_reasons = _withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3313-3314); this product declares "
            f"HOISTS_THE_WITHDRAW = False, so nothing would perform the withdraw "
            f"before its launches"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES)
    if seam_reasons:
        return False, seam_reasons[0]

    # THE BOUNDARY TRIPLE LAUNCH 3 BINDS, resolved the way the certified curl resolves
    # it; a grid it refuses is a grid this product must not bind.
    try:
        from .coverage import real_pml_boundary_kinds  # noqa: PLC0415
        cylindrical_boundary_codes(tuple(real_pml_boundary_kinds(grid)))
    except Exception as exc:  # noqa: BLE001 - a raise is a refusal here
        return False, (f"the cylindrical curl's boundary resolution refuses this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE SCAN'S SOURCE MUST EXIST: the radial sum runs over the new Hy.
    source = getattr(fields, PREFIX_SOURCE, None)
    if source is None:
        return False, f"{PREFIX_SOURCE} is not allocated; the radial sum is built from it"

    # THE ROTATION'S OWN INVARIANT.
    for name in SCRATCH_VOLUMES:
        if getattr(fields, name, None) is None:
            return False, (f"{name} is not allocated; this product rotates it against "
                           f"a launch-local scratch twin after every launch")
    return True, "covered"


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def resolve(fields: Any, grid: Any, pml: Any, *, chunk_bounds: Any) -> Dict[str, Any]:
    """Everything a run needs, resolved ONCE per frozen configuration: the tables, the
    boundary codes, the scratch twins, the scan's division readings and its output
    buffer, the chunk record (the composition's), the layer. Only BUFFERS are carried
    between steps; every word is recomputed every step."""
    _certified()
    from .coverage import constitutive_sub_lattice, real_pml_boundary_kinds  # noqa: PLC0415

    codes = cylindrical_boundary_codes(tuple(real_pml_boundary_kinds(grid)))
    source = getattr(fields, PREFIX_SOURCE)
    xp = grid.xp
    return {
        "constitutive": constitutive_kernels.real_constitutive_tables(
            pml, constitutive_sub_lattice("H")),
        "curl": cylindrical_kernels.cylindrical_curl_tables(
            pml, cylindrical_kernels.HALF_INTEGER["step_D"]),
        "codes": tuple(np.int32(code) for code in codes),
        "scratch": _hd.fused_hd_pair_scratch(fields),
        "scan": radial_scan_device_tables(radial_scan_tables(fields, pml, "step_D"), xp),
        "radial": xp.empty(tuple(int(n) for n in fields.Dz.shape), dtype=source.dtype),
        "dtdx": float(grid.dt / grid.dx), "chunk_bounds": chunk_bounds, "layer": pml,
        "launches": 0, "scan_launches": 0, "curl_launches": 0,
    }


def assert_bindings_are_disjoint(fields: Any, state: Dict[str, Any]) -> int:
    """Every ``__restrict__`` argument of launch 1 is a distinct allocation, the scratch
    twins are not the pre-launch volumes (the in-place weld the board refused), and the
    scan's buffer is nobody's field. Returns how many allocations were inspected."""
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in SCRATCH_VOLUMES:
        visit(f"{name}_out", state["scratch"][name])
    visit("radial", state["radial"])
    for name in ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz",
                 "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"):
        visit(name, getattr(fields, name))
    for axis in "xyz":
        visit(f"kps_{axis}", state["constitutive"][f"kps_{axis}"])
        visit(f"kms_{axis}", state["constitutive"][f"kms_{axis}"])
    if collisions:
        raise ValueError(
            "this product binds every field, scratch, buffer and table argument "
            "__restrict__, and these arguments alias -- undefined behaviour NVRTC "
            "miscompiles silently, and for the scratch twins the in-place weld the "
            "board refused: " + "; ".join(collisions))
    return len(bound)


def launch_update_H(fields: Any, state: Dict[str, Any], kernel: Optional[Any] = None,
                    threads: int = _FUSED_THREADS) -> Dict[str, Any]:
    """LAUNCH 1: ``update_H`` into scratch. The run's first ``update_H`` starts H from B
    on the array path BEFORE the launch (``stepping.separate_h_from_b``; B is final at
    this seam), refused by name without a layer. ``kernel`` is the gate's door,
    ``threads`` the schedule door. Returns the launch geometry."""
    from .cylindrical_complex_kernels import chunk_record_argument  # noqa: PLC0415

    cb = chunk_record_argument(state["chunk_bounds"])
    started = False
    if not bool(getattr(fields, "h_separated", True)):
        if state.get("layer") is None:
            raise RuntimeError(
                "the run's first update_H starts H from B (stepping.separate_h_from_b) "
                "and this H->D run was resolved with no layer to start it with")
        from ..stepping import separate_h_from_b  # noqa: PLC0415 - NumPy/CuPy array path

        separate_h_from_b(fields, state["layer"])
        started = True
    nx, ny, nz = (int(n) for n in getattr(fields, PREFIX_SOURCE).shape)
    blocks = (nx * ny * nz + threads - 1) // threads
    tables = state["constitutive"]
    arguments: List[Any] = [state["scratch"][name] for name in SCRATCH_VOLUMES]
    arguments += [getattr(fields, name) for name in
                  ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz")]
    arguments += [tables[f"kps_{axis}"] for axis in "xyz"]
    arguments += [tables[f"kms_{axis}"] for axis in "xyz"]
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz), cb]
    (kernel or _get_kernel())((blocks,), (threads,), tuple(arguments))
    state["launches"] += 1
    return {"launched": True, "blocks": blocks, "threads": threads,
            "elements": nx * ny * nz, "kernel": KERNEL_NAME, "h_started": started}


def scan_and_rotate(fields: Any, state: Dict[str, Any], rotate: bool = True,
                    scan_kernel: Optional[Any] = None) -> Any:
    """LAUNCH 2, the column-serial radial scan over the SCRATCH Hy -- the Hp ``update_H``
    just produced -- into the held difference buffer; then the rotation of the
    ``H``/``f_w_H`` bindings against the scratch. ``scan_kernel`` is the gate's door."""
    launch_radial_scan(scan_kernel or get_radial_scan_kernel(), state["scratch"][PREFIX_SOURCE],
                       state["scan"], state["radial"], fields.grid.xp)
    state["scan_launches"] += 1
    state["launches"] += 1
    if rotate:
        state["scratch"] = _hd.rotate_into_fields(fields, state["scratch"])
    return state["radial"]


def launch_curl(fields: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """LAUNCH 3: ``cyl_step_D_pml_real`` through its own launcher, the difference handed
    in and the ROTATED ``H`` read from ``fields``."""
    _certified()
    cylindrical_kernels._step_D_fused_pml_cylindrical(  # noqa: SLF001
        fields, state["curl"], state["codes"], state["dtdx"], radial=state["radial"],
        chunk_bounds=state["chunk_bounds"])
    state["launches"] += 1
    state["curl_launches"] += 1
    return {"launched": True, "kernel": CURL_KERNEL_NAME}


def run_cylindrical_real_fused_hd_pair(fields: Any, grid: Any, pml: Any, *,
                                       chunk_bounds: Any = None,
                                       sources: Any = None,
                                       state: Optional[Dict[str, Any]] = None,
                                       kernel: Optional[Any] = None,
                                       scan_kernel: Optional[Any] = None,
                                       threads: int = _FUSED_THREADS,
                                       rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve once, then launch 1 -> scan -> rotate -> launch 3.

    A REFUSAL IS RETURNED, NOT RAISED. ``state`` carries the resolved tables and the
    scratch between steps (the caller keeps it; ``chunk_bounds`` is required to build
    one); ``kernel``, ``scan_kernel``, ``threads`` and ``rotate`` are the gate's doors.
    """
    covered, reason = covers_cylindrical_real_fused_hd_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if state is None:
        if chunk_bounds is None:
            raise ValueError("resolving a run needs the run's chunk record (chunk_bounds)")
        state = resolve(fields, grid, pml, chunk_bounds=chunk_bounds)
    assert_bindings_are_disjoint(fields, state)
    before = state["launches"]
    first = launch_update_H(fields, state, kernel, threads)
    scan_and_rotate(fields, state, rotate, scan_kernel)
    launch_curl(fields, state)
    return {"launched": True, "state": state, "rotated": bool(rotate),
            "h_started": first["h_started"],
            "launches_this_run": state["launches"] - before,
            "launches_per_run": LAUNCHES_PER_RUN, "replaces": REPLACES}
