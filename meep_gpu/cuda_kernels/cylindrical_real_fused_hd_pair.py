"""The REAL m = 0 Dcyl H->D product: ``update_H`` + the radial increment in one launch,
``xp.cumsum`` untouched on the array path, the certified cylindrical ``step_D`` curl in
the second launch. THREE launches per seam where the D side takes seven today.

THE CELL: ``H_to_D (cuda_constitutive/ordinary, cuda_cylindrical/cylindrical)`` -- 3
seam-instances, ``tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_{0_0,1_0}``
and ``tests:TestPMLCylindrical.test_pml_cyl_0_0_0``, all ``pml_active`` and all
``buildable_not_built`` on every board until this module
(``parity/meep_gpu/results/fusion_matrix_cuda_2026-09-06_hd/fusion_matrix_cuda.json``,
``h_to_d_seam.instances``).

=============================================================================
WHY THIS SEAM WAS FILED AS REFUSED, AND WHAT THE MEASUREMENT SAID INSTEAD
=============================================================================

The cylindrical ``step_D`` differences ``cylindrical_rderiv_prefix(Hy)``
(stepping.py:447), a radial PREFIX SUM over the very ``Hy`` that ``update_H`` -- the
seam's first half -- writes. A one-launch weld would have to reproduce ``cupy.cumsum``'s
float32 summation order in-kernel, and a column-serial CUDA scan does NOT
(``results/cuda_regate_2026-09-02/cuda_cylindrical_real_2026-08-26/keep/gate.json``,
leg ``scan_order``: 115/320 .. 320/640 words differing against ``cupy.cumsum``, 0
against ``numpy.cumsum``). That refuses the IN-KERNEL SCAN. It does not refuse the
seam. The pre-``cumsum`` stage of the prefix is four elementwise CuPy passes
(stepping.py:1322-1331)::

    xp.multiply(f_p, weights, out=weighted)             # field LEFT of the weight
    increment[_face(0, 0)] = 0
    xp.subtract(weighted[1:], weighted[:-1], out=increment[1:])
    increment[1:] /= divisor
    xp.cumsum(increment, axis=0, out=...)                # THE ORACLE, untouched here

and the first four are pointwise in ``Hy`` at rows ``i`` and ``i - 1`` -- a ONE-CELL
BACKWARD RADIAL HALO, exactly the shape :mod:`.fused_hd_pair`'s foreign-cell recompute
already handles. So one launch computes ``update_H`` at the thread's own cell into
SCRATCH, recomputes the certified ``update_H`` body at the backward radial neighbour
(no store), and writes ``increment``; ``xp.cumsum`` runs unchanged; and the certified
``cyl_step_D_pml_real`` runs in the ``step_D`` slot over the rotated ``H`` with the
prefix handed in. Launches on the D-side seam: ``update_H`` 1 + prefix 5 (multiply,
row-0 fill, subtract, divide, cumsum) + ``step_D`` 1 = **7 -> 3**.

THE ARITHMETIC OF THE INCREMENT IS MEASURED, NOT CHOSEN. The fused pre-``cumsum``
stage was compiled as one NVRTC kernel and compared as uint32 words against the array
path's own four passes on every real corpus radial extent (13 shapes), both ir0, four
value classes including a bit-assembled edge class of signed zeros, subnormals and
tiny normals, under BOTH float32 subnormal policies: **0 of 7,193,552 float32 words**
per policy, 104 of 104 cases, and ``xp.cumsum`` of the fused increment reproduces the
SHIPPED ``cylindrical_rderiv_prefix`` on 0 words. The spelling that measured 0 is

    ``wi = Hy_new[i] * w[i]; wim1 = Hy_new[i-1] * w[i-1]; inc = (wi - wim1) / d[i-1]``

with IEEE ``/`` and ``--fmad=false``. The controls BIT on every case: the same source
built without ``--fmad=false`` (NVRTC contracts the subtract with one product) 1,168,184
/ 1,149,775 words; a reciprocal multiply ``diff * (1.0f / d)`` 1,816,293 / 1,802,148; a
distributed divide ``wi/d - wim1/d`` 2,114,294 / 2,065,183. Swapping the multiply's
operand order moved 0 words -- IEEE-754 multiplication commutes bitwise -- so "field on
the left" is a documentation fact here and is carried as a NULL control, not a claim.

THE DIVIDE IS ``/`` ON THIS FAMILY AND THAT IS NOT THE COMPLEX FAMILY'S ANSWER. The
sibling :mod:`.cylindrical_fused_hd_pair` (complex64 storage) spells its divide as
CuPy's SCALED complex division and its multiply as the FMA_V1 ``mul_field_left``; the
Metal backend's complex scan spells ``a * (float32(1)/d)`` because ITS oracle is NumPy.
Each spelling is right on its own backend and storage and wrong on the others -- the
reciprocal multiply is the FIRST armed mutation of this family's gate, and it must be
CAUGHT.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND A ONE-CELL RECOMPUTE, THEN THE CERTIFIED CURL
=============================================================================

Launch 1 (slot ``update_H``, kernel :data:`KERNEL_NAME`):

* the constitutive half writes NOTHING in place. ``H_new`` and ``f_w_H_new`` go to
  launch-local scratch, so ``H``, ``f_w_H`` and ``B`` are ``const`` for the whole
  launch -- the :mod:`.fused_hd_pair` shape, and the same
  :func:`.fused_hd_pair.raw_update_H_cell_source` body, imported and never retyped;
* the increment's ``Hy_new[i-1]`` is a RECOMPUTE of that body at ``idx - ny*nz`` from
  the same unwritten state, never a read of another thread's store. A pure function of
  unwritten memory has no schedule to depend on;
* ``weights`` and ``divisor`` are the SAME device row vectors the array path caches
  (``scratch.constant(("cyl_rderiv", nr, ir0, real_dtype), ...)``, stepping.py:1320),
  bound as arguments and never rebuilt in-kernel.

Between the launches: ``xp.cumsum(increment, axis=0, out=prefix)`` -- the shipped oracle,
untouched, counted as a launch (:data:`LAUNCHES_PER_RUN` is 3) -- then the ``H``/``f_w_H``
bindings rotate against the scratch (:func:`.fused_hd_pair.rotate_into_fields`).

Launch 2 (slot ``step_D``): ``cylindrical_kernels._step_D_fused_pml_cylindrical`` with
``prefix=`` -- the certified ``cyl_step_D_pml_real`` verbatim, reading the rotated ``H``
(raw ``Hy`` for ``Dx`` and for the m = 0 axis post-add, the prefix for ``Dz``). Nothing
in that module changes.

=============================================================================
WHAT SITS IN THE SEAM, AND THE INSTALLATION VERDICT
=============================================================================

The only statement between the two consults is the electric integrated-source withdraw
(driver.py:3313-3314); :mod:`..withdraw_hoist` owns it. :data:`HOISTS_THE_WITHDRAW` is
False in this round, exactly as :mod:`.fused_hd_pair` declares it, so every row with a
standing integrated electric withdraw is refused BY NAME -- on this cell there are none;
the two cylindrical withdraw rows (``cylinder_cross_section.py``, ``zone_plate.py``) are
complex-storage rows and belong to the sibling family, which refuses them the same way.
:data:`CARRIES_DEPOSIT_REPAIR` is False because nothing is injected in this seam.

:data:`INSTALLABLE` is False, and here the algebra is the OPPOSITE of the Cartesian
cell's and the verdict is still False -- on purpose. ``fused_hd_pair.INSTALLABLE_REASON``
prices its cell as a LOSS because Cartesian launches are ``4 - pairs`` and an H->D span
takes one slot from each released neighbour. On a cylindrical row the D-side baseline is
7 launches, this product saves 4, and each neighbouring cylindrical pair saves 1, so
displacing both neighbours is ``+4 - 2 = +2`` launches per step per row IN THIS PRODUCT'S
FAVOUR. But the composer's rule (``fused_pairs._neighbouring_seam_claimant`` /
``_spans_may_absorb``) counts launches as ``4 - pairs`` and would refuse or lose the
tie, and no composition gate has yet counted device launches per step on a lifted
cylindrical row with this product installed against the two neighbours installed. So
this ships ``INSTALLABLE = False`` -- credited on the board as served by PREDICATE
ADMISSION, as ``fused_hd_pair`` is for its 122 rows -- and the flip is a MEASURED step:
that composition gate, then a per-family ``launches_saved`` weight in the arbitration,
which is a rule change in ``fused_pairs.py`` and not this module's to make.

=============================================================================
NOT CERTIFIED UNTIL ITS GATE RELEASES; NOT WIRED; NOT DISPATCHED
=============================================================================

``parity/meep_gpu/gate_cuda_cylindrical_real_fused_hd_pair.py`` is the gate. This
module is not in ``fused_pairs.FUSED_PRODUCTS`` / ``FUSED_PAIR_ARMS`` and its predicate
is not in ``registry.NOT_REGISTERED`` until the wiring change that adds them lands;
``fastpath.plan_fast_path`` never names this package. Nothing launches this from a run.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import covers_real_pml_constitutive
from .cylindrical_coverage import (covers_real_pml_cylindrical_curl,
                                   cylindrical_boundary_codes)
from .cylindrical_prefix import PREFIX_COMPONENT, PREFIX_IR0
from . import fused_hd_pair as _hd
from .. import withdraw_hoist as _withdraw_hoist

# The two certified halves' device modules. Taken DEFENSIVELY: both import CuPy at
# scope, and the PREDICATE half of this family must answer on a host with no device
# (the census asks it there). The emitter and the launcher refuse BY NAME where they
# are absent.
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
KERNEL_NAME = "update_H_increment_pml_cyl_real"

#: Launch 2 is the certified cylindrical curl, by its own name, launched through its
#: own module. Recorded here so a reader of this file knows which kernel the second
#: slot runs without opening ``cylindrical_kernels``.
CURL_KERNEL_NAME = "cyl_step_D_pml_real"

#: The sub-step slot this product is registered on: the FIRST half of the seam.
SLOT = "update_H"

#: The driver passes the product's launches perform, in driver order (driver.py:3311,
#: :3315). The withdraw between them is NOT carried (:data:`HOISTS_THE_WITHDRAW`).
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``fused_pairs.FUSED_PAIR_SEAMS`` files this row under.
SEAM: str = _withdraw_hoist.SEAM

#: The volumes launch 1 writes to scratch and the launcher rotates afterwards -- the
#: :mod:`.fused_hd_pair` six, imported so the two products cannot disagree.
SCRATCH_VOLUMES: Tuple[str, ...] = _hd.SCRATCH_VOLUMES
H_TARGETS: Tuple[str, str, str] = _hd.H_TARGETS

#: Which stored component the D-side prefix is built from and at which ``ir0`` --
#: ``Hy`` at 0.5 -- read off the shared table rather than retyped: getting either
#: backwards is a half-cell error in the radial weights and is silent.
PREFIX_SOURCE: str = PREFIX_COMPONENT["step_D"]
PREFIX_IR0_VALUE: float = PREFIX_IR0["step_D"]

#: Launches per complete seam: launch 1, the CuPy scan, launch 2. THE SCAN IS COUNTED.
#: It is a CuPy kernel the memo counter cannot see, so :func:`run_cylindrical_real_fused_hd_pair`
#: keeps its own tally beside the memo's two.
LAUNCHES_PER_RUN = 3

#: CERTIFIED BY ``parity/meep_gpu/gate_cuda_cylindrical_real_fused_hd_pair.py`` on
#: device under both float32 subnormal policies (campaign
#: ``cuda_cylindrical_real_fused_hd_pair_2026-09-07``). THE DECLARATION IS THE FINAL
#: BYTES, this track's rule: the campaign was cut against THESE bytes with the name
#: already here, so the record binds what ships. Until the record block is written by
#: the record tools, ``test_kernel_partition.py`` reports the name as certified without
#: a block -- the wiring change that adds the block is what turns it green.
CERTIFIED_KERNELS = (
    "update_H_increment_pml_cyl_real",
)

#: Spelled as a dict WITHOUT an annotation (the partition reader matches
#: ``ast.Assign``). Empty: this family ships one kernel and it is the one above.
UNCERTIFIED_KERNELS = {}

#: Nothing is injected between the two consults; ``deposit_repair`` is not consulted.
CARRIES_DEPOSIT_REPAIR = False

#: The seam's electric withdraw is NOT performed by this product in this round; rows
#: with a standing one are refused by name (none on this cell today).
HOISTS_THE_WITHDRAW = False

#: The composer may not install this product. See the module docstring: the
#: cylindrical launch algebra FAVOURS it, and the flip is a measured step, not a flag.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "slot arbitration, measured on the Cartesian cell and NOT YET measured on this one. "
    "fused_hd_pair.INSTALLABLE_REASON prices an H->D span as a TIE or a LOSS because "
    "Cartesian launches over step_B - update_H - step_D - update_E are 4 - (installed "
    "pairs) and the span takes one slot from each released neighbour. On a cylindrical "
    "row the algebra is different: the D-side seam is 7 launches today (update_H, the "
    "prefix's multiply / row-0 fill / subtract / divide / cumsum, step_D) and this "
    "product takes it to 3, saving 4, while each neighbouring cylindrical pair saves 1 "
    "-- displacing both neighbours is +4 - 2 = +2 launches per step per row in this "
    "product's favour. But the composer's rule (_neighbouring_seam_claimant / "
    "_spans_may_absorb) counts launches as 4 - pairs and would refuse or lose the tie, "
    "and no composition gate has yet counted device launches per step on a lifted "
    "cylindrical row with this product installed against the two neighbours installed. "
    "So INSTALLABLE stays False -- credited on the board as served by predicate "
    "admission, as fused_hd_pair is -- and the flip is a MEASURED step: that composition "
    "gate, then a per-family launches_saved weight in the arbitration, which is a rule "
    "change in fused_pairs.py")

WHAT_A_RELEASE_DOES_NOT_LICENSE = (
    "SERVED on the fusion board is PREDICATE ADMISSION by the board's own definition, "
    "so a released product credits its admitted seam-instances while executing "
    "NOWHERE: nothing under meep_gpu/ imports cuda_kernels outside tests, this family "
    "declares INSTALLABLE = False so the composer refuses to install it on every "
    "configuration once wired, it is not in the composer's tables until the wiring "
    "change lands, and no timing of any kind has been taken of this shape. The launch "
    "figures in this module are COUNTS.")

#: Lanes per block, the certified halves' own.
_FUSED_THREADS = 256

#: NVRTC options. ``--fmad=false`` is CORRECTNESS on this launch and it is MEASURED:
#: the increment's ``(wi - wim1)`` with ``wi = h * w`` is a contraction candidate, and
#: the same source built without the guard moved 1,168,184 / 1,149,775 of 7,193,552
#: words (keep / flush). The constitutive half's two accumulations are the other two
#: candidates, as in every constitutive kernel on this track.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "CURL_KERNEL_NAME", "FAMILY",
    "HOISTS_THE_WITHDRAW", "H_TARGETS", "INCREMENT_SPELLING", "INSTALLABLE",
    "INSTALLABLE_REASON", "KERNEL_NAME", "LAUNCHES_PER_RUN", "PREFIX_IR0_VALUE",
    "PREFIX_SOURCE", "REPLACES", "SCRATCH_VOLUMES", "SEAM", "SLOT",
    "UNCERTIFIED_KERNELS", "WHAT_A_RELEASE_DOES_NOT_LICENSE",
    "assert_bindings_are_disjoint", "covers_cylindrical_real_fused_hd_pair",
    "device_sources", "increment_source", "kernel_source",
    "launch_constitutive_and_increment", "launch_curl", "prefix_row_vectors",
    "resolve", "run_cylindrical_real_fused_hd_pair", "scan_and_rotate", "signature",
    "source_digest",
]


# ---------------------------------------------------------------------------
# The device source
# ---------------------------------------------------------------------------

#: THE MEASURED SPELLING OF THE INCREMENT, as data so a gate and the host suite can
#: assert the emitted text carries exactly these lines. Each entry is (statement, the
#: array-path statement it transcribes, the control that bites if it is misspelled).
INCREMENT_SPELLING: Tuple[Dict[str, str], ...] = (
    {"line": "float wi = own_h[1] * weights[i];",
     "transcribes": "xp.multiply(f_p, weights, out=weighted) at row i, field LEFT",
     "control": "operand order swapped: 0 words (IEEE multiplication commutes) -- a "
                "documentation fact carried as a NULL control"},
    {"line": "float wim1 = halo_h[1] * weights[i - 1];",
     "transcribes": "the same multiply at row i - 1, on the RECOMPUTED Hy_new[i-1]",
     "control": "reading the stored pre-launch Hy[idx - nyz] instead of recomputing: "
                "the array path differences update_H's OUTPUT, so this must be caught"},
    {"line": "float diff = wi - wim1;",
     "transcribes": "xp.subtract(weighted[1:], weighted[:-1], out=increment[1:])",
     "control": "the same source without --fmad=false contracts this subtract with "
                "one product: 1,168,184 / 1,149,775 of 7,193,552 words"},
    {"line": "increment[idx] = diff / divisor[i - 1];",
     "transcribes": "increment[1:] /= divisor -- CuPy true_divide, IEEE '/'",
     "control": "diff * (1.0f / divisor[i - 1]): 1,816,293 / 1,802,148 words; "
                "(wi / d) - (wim1 / d): 2,114,294 / 2,065,183 words"},
    {"line": "increment[idx] = 0.0f;",
     "transcribes": "increment[_face(0, 0)] = 0 -- the integer 0 is +0.0f",
     "control": "a sign-carrying zero or a non-zero row 0 moves the whole prefix"},
)


def _certified() -> None:
    if constitutive_kernels is None or cylindrical_kernels is None:
        raise RuntimeError(
            "constitutive_kernels / cylindrical_kernels are not importable on this host "
            "(they import CuPy at module scope), so there is no certified text to "
            "splice and no curl to launch. covers_cylindrical_real_fused_hd_pair needs "
            "neither and still answers")


def signature() -> str:
    """Launch 1's parameter list. The kernel name is a LITERAL so the partition reader
    sees it; every pointer this launch writes is a scratch twin or the increment."""
    return '''
extern "C" __global__ void update_H_increment_pml_cyl_real(
    // THE SCRATCH OUTPUTS: a DIFFERENT allocation from every pre-launch volume below
    // (assert_bindings_are_disjoint checks it by base address before every launch).
    float* __restrict__ Hx_out, float* __restrict__ Hy_out,
    float* __restrict__ Hz_out,
    float* __restrict__ f_w_Hx_out, float* __restrict__ f_w_Hy_out,
    float* __restrict__ f_w_Hz_out,
    // THE INCREMENT: the pre-cumsum stage of cylindrical_rderiv_prefix(Hy_new, 0.5),
    // one word per cell, row 0 an exact +0.0f. xp.cumsum runs over it afterwards.
    float* __restrict__ increment,
    // THE PRE-LAUNCH STATE, read at ANY cell by ANY thread and written by none.
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    const float* __restrict__ f_w_Hx, const float* __restrict__ f_w_Hy,
    const float* __restrict__ f_w_Hz,
    const float* __restrict__ Bx, const float* __restrict__ By,
    const float* __restrict__ Bz,
    // The array path's own cached row vectors: weights has nx rows, divisor nx - 1.
    const float* __restrict__ weights, const float* __restrict__ divisor,
    int nx, int ny, int nz,
    // update_H's INTEGER-sub-lattice coefficients (constitutive_sub_lattice("H")).
    const float* __restrict__ kps_x, const float* __restrict__ kps_y,
    const float* __restrict__ kps_z,
    const float* __restrict__ kms_x, const float* __restrict__ kms_y,
    const float* __restrict__ kms_z
) {
'''


def increment_source() -> str:
    """The body below the constitutive store: the fused pre-``cumsum`` stage.

    Every statement is one of :data:`INCREMENT_SPELLING`'s lines; the gate's
    transcription leg asserts that correspondence rather than trusting this docstring.
    """
    lines = [
        "",
        "    // --- THE INCREMENT: stepping.cylindrical_rderiv_prefix's pre-cumsum",
        "    // stage (stepping.py:1293-1302) at THIS cell. Row 0 is the sum's start",  # stepping.py live lines for the frozen device-text citation(s) in this string: 1293-1302->1322-1331
        "    // (the array path assigns the integer 0 = +0.0f). Rows i >= 1 need",
        "    // Hy_new at i and i - 1: the own cell is in a register, the backward",
        "    // radial neighbour is RECOMPUTED from pre-launch state through the same",
        "    // raw_update_H_cell -- never a read of another thread's store. The",
        "    // spelling is the MEASURED one (module docstring): field-left multiply,",
        "    // one subtract, IEEE '/', under --fmad=false. NOT a reciprocal multiply",
        "    // (1,816,293 words), NOT distributed over the divide (2,114,294 words).",
        "    const int nyz = ny * nz;",
        "    const int i = idx / nyz;",
        "    if (i == 0) {",
        "        increment[idx] = 0.0f;",
        "        return;",
        "    }",
        "    float halo_h[3];",
        "    float halo_w[3];",
        "    raw_update_H_cell(idx - nyz, weld, halo_h, halo_w);",
        "    float wi = own_h[1] * weights[i];",
        "    float wim1 = halo_h[1] * weights[i - 1];",
        "    float diff = wi - wim1;",
        "    increment[idx] = diff / divisor[i - 1];",
        "",
    ]
    return "\n".join(lines)


def kernel_source() -> str:
    """The whole device source of launch 1. ONE string for this family.

    The constitutive prelude, the argument pack, the pure cell function and the pack
    construction are :mod:`.fused_hd_pair`'s own emitters -- imported, so the recompute
    IS the H->D weld's certified recompute and the two cannot drift.
    """
    _certified()
    weld = "\n".join([
        "    int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        "    if (idx >= nx * ny * nz) return;",
        "",
        _hd.weld_args_construction().rstrip("\n"),
        "",
        "    // --- update_H at the thread's OWN cell, computed into registers and",
        "    // stored to SCRATCH. Nothing written here is read by this launch; the",
        "    // launcher rotates H/f_w_H against the scratch after the cumsum.",
        "    float own_h[3];",
        "    float own_w[3];",
        "    raw_update_H_cell(idx, weld, own_h, own_w);",
        "    Hx_out[idx] = own_h[0]; Hy_out[idx] = own_h[1]; Hz_out[idx] = own_h[2];",
        "    f_w_Hx_out[idx] = own_w[0]; f_w_Hy_out[idx] = own_w[1];",
        "    f_w_Hz_out[idx] = own_w[2];",
    ])
    source = (_hd.constitutive_prelude() + _hd.weld_args_struct()
              + _hd.raw_update_H_cell_source() + signature() + weld
              + increment_source() + "}\n")
    # PURE ASCII IS A COMPILE REQUIREMENT on this track (the NVRTC locale trap).
    source.encode("ascii")
    return source


def device_sources() -> Dict[str, str]:
    """``{kernel name: source}`` for launch 1 -- the shape every family publishes.
    Launch 2's text is ``cylindrical_kernels``' and is pinned by that family's record."""
    return {KERNEL_NAME: kernel_source()}


def source_digest() -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(kernel_source().encode("utf-8")).hexdigest()


def _get_kernel(source: Optional[str] = None):
    """Compile on first use, memoized on (name, options, policy, source). ``cupy`` is
    imported HERE, never at scope. The source is in the key, so a gate's mutated string
    cannot be served the shipped binary."""
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

    # THE BOUNDARY TRIPLE LAUNCH 2 BINDS, resolved the way the certified curl resolves
    # it; a grid it refuses is a grid this product must not bind.
    try:
        from .coverage import real_pml_boundary_kinds  # noqa: PLC0415
        cylindrical_boundary_codes(tuple(real_pml_boundary_kinds(grid)))
    except Exception as exc:  # noqa: BLE001 - a raise is a refusal here
        return False, (f"the cylindrical curl's boundary resolution refuses this grid: "
                       f"{type(exc).__name__}: {exc}")

    # THE PREFIX SOURCE MUST BE THE VOLUME THE INCREMENT INDEXES: nx rows of weights,
    # nx - 1 of divisor, a C-contiguous float32 (nr, 1, nz) Hy.
    source = getattr(fields, PREFIX_SOURCE, None)
    if source is None:
        return False, f"{PREFIX_SOURCE} is not allocated; the increment is built from it"
    if int(source.shape[0]) < 2:
        return False, (f"{PREFIX_SOURCE} has {int(source.shape[0])} radial row(s); the "
                       f"increment differences rows i and i - 1 and the array path's "
                       f"divisor has nr - 1 rows")

    # THE ROTATION'S OWN INVARIANT.
    for name in SCRATCH_VOLUMES:
        if getattr(fields, name, None) is None:
            return False, (f"{name} is not allocated; this product rotates it against "
                           f"a launch-local scratch twin after every launch")
    return True, "covered"


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def prefix_row_vectors(fields: Any) -> Tuple[Any, Any]:
    """``(weights, divisor)`` -- THE ARRAY PATH'S OWN device row vectors, flat.

    Resolved through ``fields.scratch.constant`` under the SAME key
    ``stepping.cylindrical_rderiv_prefix`` uses (stepping.py:1320), so on an engine that
    steps the array path the very same cached arrays are bound; on a bare ``Fields``
    with no scratch they are built by the same function. Never rebuilt in-kernel.
    """
    from ..stepping import _cylindrical_rderiv_weights  # noqa: PLC0415

    source = getattr(fields, PREFIX_SOURCE)
    xp = fields.grid.xp
    rows = int(source.shape[0])
    real_dtype = source.real.dtype
    key = ("cyl_rderiv", rows, float(PREFIX_IR0_VALUE), real_dtype)
    scratch = getattr(fields, "scratch", None)
    if scratch is not None and hasattr(scratch, "constant"):
        weights, divisor = scratch.constant(
            key, lambda: _cylindrical_rderiv_weights(xp, rows, PREFIX_IR0_VALUE,
                                                     real_dtype))
    else:
        weights, divisor = _cylindrical_rderiv_weights(xp, rows, PREFIX_IR0_VALUE,
                                                       real_dtype)
    flat_w, flat_d = weights.reshape(-1), divisor.reshape(-1)
    if int(flat_w.shape[0]) != rows or int(flat_d.shape[0]) != rows - 1:
        raise ValueError(
            f"the prefix row vectors have {int(flat_w.shape[0])} / {int(flat_d.shape[0])} "
            f"rows for a {rows}-row volume; the kernel indexes weights[i] and "
            f"divisor[i - 1]")
    if not (flat_w.flags.c_contiguous and flat_d.flags.c_contiguous):
        raise ValueError("the prefix row vectors did not flatten to contiguous views")
    return flat_w, flat_d


def resolve(fields: Any, grid: Any, pml: Any) -> Dict[str, Any]:
    """Everything a launch pair needs, resolved ONCE per frozen configuration.

    Tables, boundary codes, the scratch twins, the increment and prefix buffers and the
    two row vectors. The PREFIX is NOT cached across steps -- it is recomputed by the
    scan every step from this step's increment -- only its BUFFER is.
    """
    import cupy as cp  # noqa: PLC0415

    _certified()
    source = getattr(fields, PREFIX_SOURCE)
    weights, divisor = prefix_row_vectors(fields)
    from .coverage import constitutive_sub_lattice, real_pml_boundary_kinds  # noqa: PLC0415

    codes = cylindrical_boundary_codes(tuple(real_pml_boundary_kinds(grid)))
    return {
        "constitutive": constitutive_kernels.real_constitutive_tables(
            pml, constitutive_sub_lattice("H")),
        "curl": cylindrical_kernels.cylindrical_curl_tables(
            pml, cylindrical_kernels.HALF_INTEGER["step_D"]),
        "codes": tuple(np.int32(code) for code in codes),
        "scratch": _hd.fused_hd_pair_scratch(fields),
        "increment": cp.empty_like(source),
        "prefix": cp.empty_like(source),
        "weights": weights, "divisor": divisor,
        "dtdx": float(grid.dt / grid.dx),
        "launches": 0, "cumsum_calls": 0, "curl_launches": 0,
    }


def assert_bindings_are_disjoint(fields: Any, state: Dict[str, Any]) -> int:
    """Every ``__restrict__`` argument of launch 1 is a distinct allocation, the
    scratch twins are not the pre-launch volumes (the in-place weld the board
    refused), and the increment / prefix buffers are nobody's field. Returns how many
    distinct allocations were inspected."""
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
    visit("increment", state["increment"])
    visit("prefix", state["prefix"])
    for name in ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz",
                 "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"):
        visit(name, getattr(fields, name))
    visit("weights", state["weights"])
    visit("divisor", state["divisor"])
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


def launch_constitutive_and_increment(fields: Any, state: Dict[str, Any],
                                      kernel: Optional[Any] = None,
                                      threads: int = _FUSED_THREADS) -> Dict[str, Any]:
    """LAUNCH 1: ``update_H`` into scratch plus the increment. ``kernel`` is the gate's
    door; ``threads`` the schedule door. Returns the launch geometry."""
    nx, ny, nz = (int(n) for n in getattr(fields, PREFIX_SOURCE).shape)
    blocks = (nx * ny * nz + threads - 1) // threads
    tables = state["constitutive"]
    arguments: List[Any] = [state["scratch"][name] for name in SCRATCH_VOLUMES]
    arguments.append(state["increment"])
    arguments += [getattr(fields, name) for name in
                  ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz", "Bx", "By", "Bz")]
    arguments += [state["weights"], state["divisor"]]
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz)]
    arguments += [tables[f"kps_{axis}"] for axis in "xyz"]
    arguments += [tables[f"kms_{axis}"] for axis in "xyz"]
    (kernel or _get_kernel())((blocks,), (threads,), tuple(arguments))
    state["launches"] += 1
    return {"launched": True, "blocks": blocks, "threads": threads,
            "elements": nx * ny * nz, "kernel": KERNEL_NAME}


def scan_and_rotate(fields: Any, state: Dict[str, Any], rotate: bool = True) -> Any:
    """THE ORACLE, UNTOUCHED: ``xp.cumsum(increment, axis=0, out=prefix)`` -- the exact
    statement ``stepping.cylindrical_rderiv_prefix`` ends with -- then the rotation of
    the ``H``/``f_w_H`` bindings against the scratch. Counted as one launch."""
    xp = fields.grid.xp
    xp.cumsum(state["increment"], axis=0, out=state["prefix"])
    state["cumsum_calls"] += 1
    state["launches"] += 1
    if rotate:
        state["scratch"] = _hd.rotate_into_fields(fields, state["scratch"])
    return state["prefix"]


def launch_curl(fields: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    """LAUNCH 2: the certified ``cyl_step_D_pml_real`` through its own launcher, with
    the prefix handed in and the ROTATED ``H`` read from ``fields``."""
    _certified()
    cylindrical_kernels._step_D_fused_pml_cylindrical(  # noqa: SLF001
        fields, state["curl"], state["codes"], state["dtdx"], prefix=state["prefix"])
    state["launches"] += 1
    state["curl_launches"] += 1
    return {"launched": True, "kernel": CURL_KERNEL_NAME}


def run_cylindrical_real_fused_hd_pair(fields: Any, grid: Any, pml: Any, *,
                                       sources: Any = None,
                                       state: Optional[Dict[str, Any]] = None,
                                       kernel: Optional[Any] = None,
                                       threads: int = _FUSED_THREADS,
                                       rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve once, then launch 1 -> cumsum -> rotate -> launch 2.

    A REFUSAL IS RETURNED, NOT RAISED. ``state`` carries the resolved tables and the
    scratch between steps (the caller keeps it); ``kernel``, ``threads`` and ``rotate``
    are the gate's doors. The record carries the family's own launch tally (3 per run)
    beside the count the compile memo can see (2 RawKernel launches).
    """
    covered, reason = covers_cylindrical_real_fused_hd_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if state is None:
        state = resolve(fields, grid, pml)
    assert_bindings_are_disjoint(fields, state)
    before = state["launches"]
    launch_constitutive_and_increment(fields, state, kernel, threads)
    scan_and_rotate(fields, state, rotate)
    launch_curl(fields, state)
    return {"launched": True, "state": state, "rotated": bool(rotate),
            "launches_this_run": state["launches"] - before,
            "launches_per_run": LAUNCHES_PER_RUN,
            "raw_kernel_launches_this_run": 2, "cumsum_calls_this_run": 1,
            "replaces": REPLACES}
