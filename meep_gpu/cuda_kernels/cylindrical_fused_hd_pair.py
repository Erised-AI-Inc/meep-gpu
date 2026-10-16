"""The COMPLEX Dcyl H->D product: ``update_H`` into scratch, the radial scan over the new
Hp, the rotation, the cylindrical complex ``step_D`` curl -- three launches per seam.

THE CELL: ``H_to_D (cuda_complex/complex, cuda_cyl_complex/cylindrical complex)``, every
integer m -- 18 seam-instances on the census ``cuda_predicate_coverage_2026-09-06_hd``: 16
``buildable_not_built`` and 2 ``withdraw_seam`` (``cylinder_cross_section.py``,
``zone_plate.py``), which this product REFUSES BY NAME as :mod:`.fused_hd_pair` refuses
its five.

WHAT THE THREE LAUNCHES ARE. Launch 1 (slot ``update_H``, :data:`KERNEL_NAME`) steps
``update_H`` at the thread's own cell into launch-local SCRATCH through
:mod:`.complex_fused_hd_pair`'s pure cell function -- ``complex_emitter``'s
``update_H_pml_complex_bloch`` body lifted, the chunk record decoded at the cell, imported
and never retyped -- so H, ``f_w_H`` and B are const for the whole launch. Launch 2 is the
column-serial radial scan (:mod:`.cylindrical_prefix`) over the SCRATCH Hy, per stored
part. Then the H/``f_w_H`` bindings rotate against the scratch, and launch 3 is
``cylindrical_complex_kernels``' ``cyl_step_D_pml_complex`` over the rotated H with the
difference handed in.

THE RUN'S FIRST ``update_H`` STARTS H FROM B on the array path BEFORE launch 1
(``stepping.separate_h_from_b``), the :mod:`.complex_fused_hd_pair` arrangement.

THE INSTALLATION VERDICT is the real twin's (:mod:`.cylindrical_real_fused_hd_pair`): under
MEEP's per-chunk radial sum the D side's own baseline is three launches, this product is
three, and it ships ``INSTALLABLE = False``. The 0.9.2 product fused the pre-``cumsum``
increment (CuPy's complex multiply and scaled divide) into launch 1; that stage is gone.

The seam's electric withdraw is not carried (:data:`HOISTS_THE_WITHDRAW`); nothing is
injected in this seam.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import complex_fused_hd_pair as _chd
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import covers_real_pml_complex_constitutive
from .cylindrical_coverage import covers_pml_cylindrical_complex_curl
from .cylindrical_prefix import (PREFIX_COMPONENT, get_radial_scan_kernel,
                                 launch_radial_scan, radial_scan_device_tables,
                                 radial_scan_tables)
from . import cylindrical_complex_kernels as _cyl
from . import fused_hd_pair as _hd
from .. import withdraw_hoist as _withdraw_hoist

# The complex constitutive TABLE builder imports CuPy at scope; taken defensively so the
# predicate and the emitter answer on a host with no device.
try:
    from . import complex_pml_kernels
except Exception:  # noqa: BLE001 - no CuPy on this host
    complex_pml_kernels = None  # type: ignore[assignment]

FAMILY = "cuda_cylindrical_fused_hd_pair"
KERNEL_NAME = "update_H_scratch_pml_cyl_complex"
CURL_KERNEL_NAME = "cyl_step_D_pml_complex"
SLOT = "update_H"
REPLACES: Tuple[str, ...] = ("update_H", "step_D")
SEAM: str = _withdraw_hoist.SEAM
SCRATCH_VOLUMES: Tuple[str, ...] = _hd.SCRATCH_VOLUMES
H_TARGETS: Tuple[str, str, str] = _hd.H_TARGETS
PREFIX_SOURCE: str = PREFIX_COMPONENT["step_D"]
LAUNCHES_PER_RUN = 3

#: The entry point is NEW with the per-chunk radial sum and no record names it yet: its
#: 0.9.2 predecessor (``update_H_increment_pml_cyl_complex``) is gone, and the record that
#: certified it is re-cut in the combined certification round (bit_parity_0_10 CERT-3).
CERTIFIED_KERNELS = ()
UNCERTIFIED_KERNELS = {
    "update_H_scratch_pml_cyl_complex":
        "gated by parity/meep_gpu/gate_cuda_cylindrical_fused_hd_pair.py under both "
        "float32 subnormal policies; the certification block that would move this name "
        "into CERTIFIED_KERNELS is cut from that gate's artifacts in the certification "
        "round",
}

#: Nothing is injected between the two consults; ``deposit_repair`` is not consulted.
CARRIES_DEPOSIT_REPAIR = False

#: The seam's electric withdraw is NOT performed by this product.
HOISTS_THE_WITHDRAW = False

#: The composer may not install this product (the real twin's reason).
INSTALLABLE = False

INSTALLABLE_REASON = (
    "no launch saved: under MEEP's per-chunk radial sum the D side's own baseline is "
    "three launches (update_H, the column-serial scan, step_D) and this product is also "
    "three; the 0.9.2 saving came from fusing the pre-cumsum increment, whose elementwise "
    "passes no longer exist. A two-launch shape (update_H row by row inside the scan) is "
    "an owner's design ruling. The board credits the cell by predicate admission")

WHAT_A_RELEASE_DOES_NOT_LICENSE = (
    "SERVED on the fusion board is PREDICATE ADMISSION by the board's own definition, "
    "so a released product credits its admitted seam-instances while executing "
    "NOWHERE: INSTALLABLE = False keeps the composer from installing it, and no timing "
    "of any kind has been taken of this shape. The launch figures here are COUNTS.")

_FUSED_THREADS = 256

#: ``--fmad=false`` is CORRECTNESS: the constitutive accumulation is a contraction
#: candidate the array path rounds twice.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "CURL_KERNEL_NAME", "FAMILY",
    "HOISTS_THE_WITHDRAW", "H_TARGETS", "INSTALLABLE", "INSTALLABLE_REASON",
    "KERNEL_NAME", "LAUNCHES_PER_RUN", "PREFIX_SOURCE", "REPLACES", "SCRATCH_VOLUMES",
    "SEAM", "SLOT", "UNCERTIFIED_KERNELS", "WHAT_A_RELEASE_DOES_NOT_LICENSE",
    "assert_bindings_are_disjoint", "covers_cylindrical_fused_hd_pair",
    "device_sources", "kernel_source", "launch_curl", "launch_update_H", "resolve",
    "run_cylindrical_fused_hd_pair", "scan_and_rotate", "signature", "source_digest",
]


# ---------------------------------------------------------------------------
# The device source
# ---------------------------------------------------------------------------


_SIGNATURE = '''
extern "C" __global__ void update_H_scratch_pml_cyl_complex(
    // THE SCRATCH OUTPUTS (float32 word views of complex64), a DIFFERENT allocation
    // from every pre-launch volume below.
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
    // Extents in COMPLEX CELLS.
    int nx, int ny, int nz,
    // THE CHUNK RECORD (chunk_map.chunk_bounds_record), int32[25], decoded by the
    // lifted body at the cell it is evaluated for.
    const int* __restrict__ cb
) {
'''


def signature() -> str:
    """Launch 1's parameter list. Every pointer this launch writes is a scratch twin."""
    return _SIGNATURE


def _weld_args_construction() -> str:
    """The pack built from this kernel's own parameters, which carry the pack's names."""
    lines = ["    WeldArgs weld;"]
    for _kind, name in _chd.WELD_ARGS_FIELDS:
        lines.append(f"    weld.{name} = {name};")
    return "\n".join(lines)


def kernel_source() -> str:
    """The whole device source of launch 1: :mod:`.complex_fused_hd_pair`'s pure
    constitutive prelude, argument pack and cell function, imported, and the scratch
    stores."""
    body = "\n".join([
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (idx >= nx * ny * nz) return;",
        "",
        _weld_args_construction(),
        "",
        "    // --- update_H at the thread's OWN cell into registers, stored to SCRATCH.",
        "    cf own_h[3];",
        "    cf own_w[3];",
        "    raw_update_H_cell(idx, weld, own_h, own_w);",
        "    cf_store(Hx_out, idx, own_h[0]); cf_store(Hy_out, idx, own_h[1]);",
        "    cf_store(Hz_out, idx, own_h[2]);",
        "    cf_store(f_w_Hx_out, idx, own_w[0]); cf_store(f_w_Hy_out, idx, own_w[1]);",
        "    cf_store(f_w_Hz_out, idx, own_w[2]);",
        "",
    ])
    source = (_chd.constitutive_prelude("plain") + _chd.weld_args_struct()
              + _chd.raw_update_H_cell_source() + signature() + body + "}\n")
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """The source launch 1 emits, keyed by kernel name -- for a digest."""
    return {KERNEL_NAME: kernel_source()}


def source_digest() -> str:
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    for name, source in device_sources().items():
        digest.update(name.encode("ascii"))
        digest.update(source.encode("ascii"))
    return digest.hexdigest()


def _get_kernel(source: Optional[str] = None):
    import cupy as cp  # noqa: PLC0415

    code = kernel_source() if source is None else source
    key = _kernel_cache_key(KERNEL_NAME, True, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def covers_cylindrical_fused_hd_pair(fields: Any, pml: Any, grid: Any,
                                     sources: Any = None) -> Tuple[bool, str]:
    """May this product span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    A CONJUNCTION: the certified complex constitutive predicate on the H side (asked
    FIRST), the certified complex Dcyl curl predicate
    on ``step_D`` (every integer m), the seam's withdraw clause, the boundary
    resolution launch 3 binds, the scan's source, and the rotation's six volumes.
    """
    covered, reason = covers_real_pml_complex_constitutive(
        fields, pml, grid, "H")
    if not covered:
        return False, f"constitutive half: {reason}"
    covered, reason = covers_pml_cylindrical_complex_curl(
        fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"
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
    try:
        _cyl.cylindrical_complex_boundary_codes(grid)
    except Exception as exc:  # noqa: BLE001
        return False, (f"the cylindrical curl's boundary resolution refuses this grid: "
                       f"{type(exc).__name__}: {exc}")
    source = getattr(fields, PREFIX_SOURCE, None)
    if source is None:
        return False, f"{PREFIX_SOURCE} is not allocated; the radial sum is built from it"
    for name in SCRATCH_VOLUMES:
        if getattr(fields, name, None) is None:
            return False, (f"{name} is not allocated; this product rotates it against "
                           f"a launch-local scratch twin after every launch")
    return True, "covered"


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def _scratch(fields: Any) -> Dict[str, Any]:
    """The six launch-local complex volumes, shaped like the live ones."""
    xp = fields.grid.xp
    return {name: xp.empty_like(getattr(fields, name)) for name in SCRATCH_VOLUMES}


def resolve(fields: Any, grid: Any, pml: Any, *,
            chunk_bounds: Any = None) -> Dict[str, Any]:
    """Everything a run needs, resolved ONCE per frozen configuration. ``chunk_bounds``
    is the composition's record (required to launch)."""
    if complex_pml_kernels is None:
        raise RuntimeError("the CuPy-backed complex constitutive tables are not "
                           "importable on this host")
    source = getattr(fields, PREFIX_SOURCE)
    xp = grid.xp
    return {
        "constitutive": complex_pml_kernels.complex_constitutive_tables(pml, False),
        "curl": _cyl.cylindrical_complex_curl_tables(pml, _cyl.HALF_INTEGER["step_D"]),
        "scratch": _scratch(fields),
        "scan": radial_scan_device_tables(radial_scan_tables(fields, pml, "step_D"), xp),
        "radial": xp.empty(tuple(int(n) for n in fields.Dz.shape), dtype=source.dtype),
        "chunk_bounds": chunk_bounds, "layer": pml,
        "launches": 0, "scan_launches": 0, "curl_launches": 0,
    }


def assert_bindings_are_disjoint(fields: Any, state: Dict[str, Any]) -> int:
    """Every ``__restrict__`` argument of launch 1 is a distinct allocation and the
    scratch twins are not the pre-launch volumes. Returns the allocations inspected."""
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
            "__restrict__, and these arguments alias: " + "; ".join(collisions))
    return len(bound)


def launch_update_H(fields: Any, state: Dict[str, Any], kernel: Optional[Any] = None,
                    threads: int = _FUSED_THREADS) -> Dict[str, Any]:
    """LAUNCH 1: ``update_H`` into scratch, after MEEP's start of H on the run's first
    update (``stepping.separate_h_from_b``, refused by name without a layer)."""
    word_view = _cyl.word_view
    cb = _cyl.chunk_record_argument(state["chunk_bounds"])
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
    arguments: List[Any] = [word_view(state["scratch"][name]) for name in SCRATCH_VOLUMES]
    arguments += [word_view(getattr(fields, name)) for name in
                  ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz")]
    arguments += [tables[f"kps_{axis}"] for axis in "xyz"]
    arguments += [tables[f"kms_{axis}"] for axis in "xyz"]
    import numpy as np  # noqa: PLC0415

    arguments += [np.int32(nx), np.int32(ny), np.int32(nz), cb]
    (kernel or _get_kernel())((blocks,), (threads,), tuple(arguments))
    state["launches"] += 1
    return {"launched": True, "blocks": blocks, "threads": threads,
            "elements": nx * ny * nz, "kernel": KERNEL_NAME, "h_started": started}


def scan_and_rotate(fields: Any, state: Dict[str, Any], rotate: bool = True,
                    scan_kernel: Optional[Any] = None) -> Any:
    """LAUNCH 2, the column-serial radial scan over the SCRATCH Hy (per stored part),
    then the rotation against the scratch."""
    launch_radial_scan(scan_kernel or get_radial_scan_kernel(), state["scratch"][PREFIX_SOURCE],
                       state["scan"], state["radial"], fields.grid.xp)
    state["scan_launches"] += 1
    state["launches"] += 1
    if rotate:
        state["scratch"] = _hd.rotate_into_fields(fields, state["scratch"])
    return state["radial"]


def launch_curl(fields: Any, grid: Any, pml: Any, state: Dict[str, Any],
                curl_kernel: Optional[Any] = None) -> Dict[str, Any]:
    """LAUNCH 3: ``cyl_step_D_pml_complex`` through its own launcher, the difference
    handed in and the ROTATED ``H`` read from ``fields``."""
    _cyl.step_cylindrical_complex("step_D", fields, grid, pml,
                                  chunk_bounds=state["chunk_bounds"], tables=state["curl"],
                                  radial=state["radial"], kernel=curl_kernel)
    state["launches"] += 1
    state["curl_launches"] += 1
    return {"launched": True, "kernel": CURL_KERNEL_NAME}


def run_cylindrical_fused_hd_pair(fields: Any, grid: Any, pml: Any, *,
                                  chunk_bounds: Any = None, sources: Any = None,
                                  state: Optional[Dict[str, Any]] = None,
                                  kernel: Optional[Any] = None,
                                  scan_kernel: Optional[Any] = None,
                                  curl_kernel: Optional[Any] = None,
                                  threads: int = _FUSED_THREADS,
                                  rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve once, then launch 1 -> scan -> rotate -> launch 3.
    A REFUSAL IS RETURNED, NOT RAISED; the keyword doors are the gate's."""
    covered, reason = covers_cylindrical_fused_hd_pair(
        fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if state is None:
        state = resolve(fields, grid, pml, chunk_bounds=chunk_bounds)
    assert_bindings_are_disjoint(fields, state)
    before = state["launches"]
    first = launch_update_H(fields, state, kernel, threads)
    scan_and_rotate(fields, state, rotate, scan_kernel)
    launch_curl(fields, grid, pml, state, curl_kernel)
    return {"launched": True, "state": state, "rotated": bool(rotate),
            "h_started": first["h_started"],
            "launches_this_run": state["launches"] - before,
            "launches_per_run": LAUNCHES_PER_RUN, "replaces": REPLACES}
