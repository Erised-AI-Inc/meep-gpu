"""Sub-step byte-identity gate for the NEW hand-CUDA CYLINDRICAL curl pair, m = 0.

THE CLAIM UNDER TEST. ``meep_gpu/cuda_kernels/cylindrical_kernels.py`` ships two
kernels that did not exist before this run — ``cyl_step_B_pml_real`` and
``cyl_step_D_pml_real`` — and this gate asks whether either reproduces
``stepping.step_B`` / ``stepping.step_D`` WORD FOR WORD on a real Dcyl grid at m = 0
under a split-field PML. Nothing in ``step_curl_kernels.py`` or ``coverage.py`` is
touched; the Cartesian pair keeps its cylindrical refusal and this pair carries its
own predicate.

CUDA HAS NEVER BEEN MEASURED ON A Dcyl RUN AT ANY SUB-STEP. So there is no standing
record to reproduce here and nothing to inherit: every number below is this run's.

=============================================================================
WHAT CYLINDRICAL ADDS, AND THEREFORE WHAT THIS GATE HAS TO BE ABLE TO SEE
=============================================================================

Four things (``cylindrical_kernels``' docstring derives them off ``stepping.py`` with
line numbers). The battery is built around them, because a gate that only re-runs the
Cartesian battery would certify a Dcyl kernel against Cartesian defects:

1. the RADIAL PREFIX SUM and its ZERO WALL ROW on the B side;
2. Bz's whole curl REPLACED by the prefix's forward difference — one subtract, one
   multiply, NOT the four-operand grouping;
3. Dz's ``first`` source SWAPPED for the prefix while Dx's Hy operands stay RAW;
4. the m = 0 AXIS RULES: ``Bx[r=0] = 0``; ``Dz[r=0] += (4*Courant)*Hp[r=0]`` as a
   POST-ADD, then ``Dy[r=0] = 0``.

=============================================================================
THE SCAN IS MEASURED HERE, NOT ASSUMED
=============================================================================

The kernels do NOT own the radial prefix: it stays on the array path, because on this
backend the oracle is ``cupy.cumsum``'s float32 summation order and the Triton track
measured that order to be reproducible by no other scan. That is a citation from
another tranche until this box says it, so leg ``scan_order`` compiles a column-serial
CUDA scan — the shape the Metal port ships, where the same premise INVERTS because
that engine holds NumPy — and compares it word for word against ``cupy.cumsum`` on
this gate's own shapes. The verdict does not change the design either way; what it
settles is whether "the scan cannot be fused bit-identically on CuPy" is a
measurement on this backend or a borrowed reading.

=============================================================================
THE FLOORS THAT REFUSE A VACUOUS PASS
=============================================================================

* ``oracle_moved`` — the fraction of output words the array path changed from the
  frozen input. A case that moved nothing is refused, not passed.
* ``axis_row_is_live`` — the r = 0 row must carry MOVED words on the targets the axis
  rules touch. An axis rule applied to a row nothing writes is a rule this case
  cannot score, and all three axis mutations would come back UNCAUGHT for a reason
  about the fixture.
* ``prefix_is_live`` — the prefix must be non-constant down r, and Bz must move. A
  flat prefix makes every prefix defect invisible.
* ``absorbs`` — ``max|kms-1|`` and ``max|sinv-1|`` on the r AND z axes. If a
  coefficient vector is the identity everywhere, a coefficient-index error on that
  axis cannot be seen.
* ``INEXACT_COURANT`` — 0.5 is exactly representable in float32 and distributing
  ``dtdx`` over a difference is then EXACT, so a case matrix without a
  non-power-of-two Courant cannot see a grouping error in the Bz substitution at all.
  The Metal port measured exactly that (``curl_bz_flat_grouping`` uncaught at 0.5,
  caught at 0.314159), so the primary courant here is 0.35 and 0.5 is carried beside
  it rather than instead of it.
* the mutation battery, INCLUDING legs that must be UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU; the cache dir MUST be fresh per policy
because CuPy's disk-cache key is computed above the strip seam)::

    export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda_stub"
    export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES=6 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_cylindrical_real.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend runs the SAME oracle, the same fixture, the same
floors and the same HOST mutations against a transcription of the device tree. IT
COMPILES NOTHING AND CERTIFIES NOTHING — what it settles is whether the harness can
fail::

    python -u gate_cuda_cylindrical_real.py --backend numpy --out /tmp/cyl_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten atomically
after every case, so an interrupted run keeps everything up to the failure.
Correctness only — no throughput claim is made or possible, because the prefix stays
on the array path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from meep_gpu.fields import Fields, IYEE_SHIFTS  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import cylindrical_coverage as cyl_coverage  # noqa: E402
from meep_gpu.cuda_kernels import cylindrical_prefix as cyl_prefix  # noqa: E402

log = probe.log
bit_compare = probe.bit_compare
combine = probe.combine
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

SEED = 20260820

#: Consecutive launches in the multi-step leg. 60 is the budget both certified
#: hand-CUDA records are cut at. "Identical for N steps" is a claim about N.
MULTI_STEP_BUDGET = 60

#: How the multi-step leg keeps the recurrence live. Held fixed, the curl operands
#: never move and a 60-launch leg becomes a slow single-launch leg. The same exact
#: float32 scale is applied on both paths, so it cancels out of the comparison. It
#: also moves the PREFIX every launch, which is the point on this family: a prefix
#: recomputed from a frozen source is a constant the kernel could have cached.
_ADVANCE = np.float32(0.97)

BC_PERIODIC = 0
BC_METALLIC = 1

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

SUB_STEP_ARRAYS: Dict[str, Dict[str, Any]] = {
    "step_B": {"targets": ("Bx", "By", "Bz"),
               "aux": ("fu_Bx", "fu_By", "fu_Bz"),
               "sources": ("Ex", "Ey", "Ez"),
               "half_integer": True,
               "backward": False},
    "step_D": {"targets": ("Dx", "Dy", "Dz"),
               "aux": ("fu_Dx", "fu_Dy", "fu_Dz"),
               "sources": ("Hx", "Hy", "Hz"),
               "half_integer": False,
               "backward": True},
}

#: The six curl terms as the SHIPPED CYLINDRICAL KERNELS spell them, transcribed from
#: ``cylindrical_kernels.py``'s two device strings. Read as ``(target, aux, first,
#: first's axis, second, second's axis, dsig axis, dsigu axis)``. ``Bz``'s entry is
#: the Cartesian one and is NOT what the kernel computes — the B-side substitution
#: replaces that whole curl — but it is kept so
#: :func:`check_terms_against_stepping` can pin the table against ``stepping``'s own
#: ``B_CURL_TERMS`` / ``D_CURL_TERMS``, which is what stops the NumPy leg quietly
#: drifting into comparing the oracle with itself.
CURL_TERMS: Dict[str, Tuple[Tuple[str, str, str, int, str, int, int, int], ...]] = {
    "step_B": (
        ("Bx", "fu_Bx", "Ez", 1, "Ey", 2, 1, 2),
        ("By", "fu_By", "Ex", 2, "Ez", 0, 2, 0),
        ("Bz", "fu_Bz", "Ey", 0, "Ex", 1, 0, 1),
    ),
    "step_D": (
        ("Dx", "fu_Dx", "Hz", 1, "Hy", 2, 1, 2),
        ("Dy", "fu_Dy", "Hx", 2, "Hz", 0, 2, 0),
        ("Dz", "fu_Dz", "Hy", 0, "Hx", 1, 0, 1),
    ),
}

#: Every array a case seeds and compares. Both sub-steps' volumes are seeded on every
#: case, because ``step_D``'s axis rule READS ``Hy`` and ``step_B``'s prefix reads
#: ``Ey``: a zero source there is a fixed point that hides the rule.
STATE: Tuple[str, ...] = (
    "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
    "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz")


def outputs(sub_step: str) -> Tuple[str, ...]:
    """The six arrays this sub-step writes.

    ``fu`` IS STATE. A tree that gets the field right and the auxiliary wrong is
    correct for exactly one launch and wrong forever after, so the auxiliary is
    compared on every case, not only in the multi-step leg. The sibling track
    measured a dropped auxiliary store at 120/120 UNCAUGHT when ``fu`` was not
    compared.
    """
    spec = SUB_STEP_ARRAYS[sub_step]
    return tuple(spec["targets"]) + tuple(spec["aux"])


# ---------------------------------------------------------------------------
# The case product
# ---------------------------------------------------------------------------
#
# EVERY CASE IS A REAL Dcyl GRID at m = 0. `Grid` builds the r axis itself — its
# boundary pair is not caller-facing (grid.py:498-508) — so the only boundary this
# sweep chooses is z, and BOTH terminations appear, because they select different
# code in the kernel's ghost rule and its ownership mask.

CASES: Tuple[Dict[str, Any], ...] = (
    # The reference shape, both z terminations.
    {"label": "r16_z20_metallic", "shape": (16, 1, 20), "z_kind": "metallic"},
    {"label": "r16_z20_periodic", "shape": (16, 1, 20), "z_kind": "periodic"},
    # A WIDER RADIAL EXTENT: the prefix's accumulated sum runs further, which is
    # where a decaying field reaches the subnormal band first.
    {"label": "r40_z16_metallic", "shape": (40, 1, 16), "z_kind": "metallic"},
    # ODD EXTENTS on both axes. The wall row lands on an odd row count and the
    # linear-index decomposition has no power-of-two stride to hide behind.
    {"label": "r9_z17_metallic", "shape": (9, 1, 17), "z_kind": "metallic"},
    {"label": "r9_z17_periodic", "shape": (9, 1, 17), "z_kind": "periodic"},
    # A TALL z EXTENT with a short r: the fastest-stride axis dominates, which is the
    # arm a Fortran-order index defect is most visible on.
    {"label": "r12_z48_metallic", "shape": (12, 1, 48), "z_kind": "metallic"},
)

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one OR a distributed
#: ``dtdx`` from a factored one, so both the guard control and the grouping mutations
#: are scored at the inexact one. The courant also moves dt, which moves every PML
#: coefficient, so it is a real second draw of the tables.
COURANTS: Tuple[float, ...] = (0.35, 0.5)
INEXACT_COURANT = 0.35

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("fmad_false", ("--fmad=false",), True),
    ("default_no_options", (), False),
)

#: The SHARED battery, borrowed from the probe so this gate cannot own a second
#: spelling of a defect the certified record was cut against. Every one of them arms
#: on the imported ``_REAL_PML_PRELUDE`` or on the Cartesian blocks this pair keeps
#: unchanged — which is itself the check that "the certified body, character for
#: character" is true rather than claimed.
SHARED_SOURCE_MUTATIONS: Tuple[str, ...] = (
    "regroup_stencil",                    # the parenthesisation the record pins
    "drop_metallic_mask",                 # the ownership mask, r AND z at once
    "swap_dsig_dsigu",                    # the coefficient pairing
    "drop_fu_store",                      # the auxiliary-only defect
    "read_fprev_after_store",             # the aliasing trap
    "fortran_order_index_decomposition",  # i and k swapped: the curl's index defect
    # THE TWO SHARED NULLS.
    "commute_dtdx_scale",
    "reload_fu_from_memory",
)
SHARED_NULL_MUTATIONS: Tuple[str, ...] = probe.PML_NULL_MUTATIONS


# ---------------------------------------------------------------------------
# The CYLINDRICAL mutations — this family's own, one per addition it makes
# ---------------------------------------------------------------------------

def _drop_bz_prefix_substitution(source: str) -> Tuple[str, int]:
    """Bz's curl becomes the ordinary Cartesian four-operand one.

    THE WHOLE B-SIDE SUBSTITUTION, deleted. ``stepping.step_B``:343-347 replaces that
    curl outright, and a kernel that computed ``dEy/dr - dEx/dphi`` instead would be
    solving Cartesian Maxwell on a cylindrical grid — smooth, plausible and wrong
    everywhere off the axis. MUST BE CAUGHT.
    """
    needle = ("        float pfx_here = pfx[idx];\n"
              "        float pfx_up   = pfx[idx + sx];\n"
              "        float curl = dtdx * (pfx_up - pfx_here);\n")
    replacement = ("        float f1 = Ey[idx];\n"
                   "        float sf = shift_up(Ey, idx, i, nx, sx, bc_x);\n"
                   "        float f2 = Ex[idx];\n"
                   "        float ss = shift_up(Ex, idx, j, ny, sy, bc_y);\n"
                   "        float curl = dtdx * ((sf - f1) + (f2 - ss));\n")
    return source.replace(needle, replacement), source.count(needle)


def _bz_flat_grouping(source: str) -> Tuple[str, int]:
    """``dtdx*pfx_up - dtdx*pfx_here`` in place of ``dtdx*(pfx_up - pfx_here)``.

    Distributing the scale over the difference is EXACT whenever ``dtdx`` is a power
    of two, so this leg is expected to be caught at :data:`INEXACT_COURANT` and NOT
    at 0.5. That is a property of the CASES, not of the kernel, and the Metal port
    measured it (uncaught at 0.5, 4,390 words at 0.314159). MUST BE CAUGHT at the
    inexact courant, which is the only courant the battery runs.
    """
    needle = "float curl = dtdx * (pfx_up - pfx_here);"
    replacement = "float curl = dtdx * pfx_up - dtdx * pfx_here;"
    return source.replace(needle, replacement), source.count(needle)


def _bz_metallic_ghost_for_wall_row(source: str) -> Tuple[str, int]:
    """The WALL ROW replaced by the generic metallic zero ghost.

    ``stepping.step_B``:326-333 extends Ep by a zero wall row and prefixes the
    EXTENDED volume, so the last row's forward difference reads a real prefix entry.
    Serving a zero there instead turns that difference into MINUS THE WHOLE
    ACCUMULATED SUM — the symptom :303-314 records (the first divergence born at the
    last two rows and growing inward). MUST BE CAUGHT.
    """
    needle = "        float pfx_up   = pfx[idx + sx];\n"
    replacement = "        float pfx_up   = (i + 1 < nx) ? pfx[idx + sx] : 0.0f;\n"
    return source.replace(needle, replacement), source.count(needle)


def _drop_dz_prefix_substitution(source: str) -> Tuple[str, int]:
    """Dz's ``first`` source reverts to the RAW Hp.

    ``stepping.step_D``:424-426 substitutes the prefix for that term only. Without
    it the Dz update computes ``dHp/dr`` where cylindrical Maxwell needs
    ``(1/r)d(r*Hp)/dr``. MUST BE CAUGHT.
    """
    needle = ("        float f1 = pfx[idx];\n"
              "        float sf = shift_dn(pfx, idx, i, nx, sx, bc_x);\n")
    replacement = ("        float f1 = Hy[idx];\n"
                   "        float sf = shift_dn(Hy, idx, i, nx, sx, bc_x);\n")
    return source.replace(needle, replacement), source.count(needle)


def _prefix_into_dx(source: str) -> Tuple[str, int]:
    """The prefix LEAKS into Dx, whose Hy operands must stay RAW.

    ``stepping.step_D``:423-426 swaps the source dict for the Dz TERM only; ``Dx``
    reads the untouched ``magnetic`` mapping. The two operands differ by a whole
    radial integral, so this is a wrong answer on a component the substitution has
    nothing to do with. MUST BE CAUGHT.
    """
    needle = ("        float f2 = Hy[idx];\n"
              "        float ss = shift_dn(Hy, idx, k, nz, sz, bc_z);\n")
    replacement = ("        float f2 = pfx[idx];\n"
                   "        float ss = shift_dn(pfx, idx, k, nz, sz, bc_z);\n")
    return source.replace(needle, replacement), source.count(needle)


def _drop_axis_zero_bx(source: str) -> Tuple[str, int]:
    """``Bx[r=0] = 0`` deleted (``stepping._cylindrical_axis_zero_B``:659-661).

    A radial vector at r = 0 points nowhere. MUST BE CAUGHT.
    """
    needle = "    if (i == 0) Bx[idx] = 0.0f;\n"
    return source.replace(needle, ""), source.count(needle)


def _drop_axis_zero_dy(source: str) -> Tuple[str, int]:
    """``Dy[r=0] = 0`` deleted (``stepping._cylindrical_axis_zero_D``:587).

    An azimuthal vector at r = 0 has no direction to point. MUST BE CAUGHT.
    """
    needle = "        Dy[idx] = 0.0f;\n"
    return source.replace(needle, ""), source.count(needle)


def _drop_axis_add_dz(source: str) -> Tuple[str, int]:
    """The on-axis Dz post-add deleted (``stepping._cylindrical_axis_zero_D``:585).

    ``4*Courant*Hp`` at the axis row is the analytic limit of ``(1/r)d(r*Hp)/dr``
    there (step_db.cpp:299-341), and the prefix machinery cannot supply it because
    the curl is masked on that row. MUST BE CAUGHT.
    """
    needle = "        Dz[idx] = Dz[idx] + (axis_coef * Hy[idx]);\n"
    return source.replace(needle, ""), source.count(needle)


def _fold_axis_add_into_curl(source: str) -> Tuple[str, int]:
    """The on-axis Dz increment FOLDED INTO THE CURL instead of post-added.

    THE EXACT ERROR ``stepping.py``:545-551 RECORDS AS MEASURED WRONG for m = 0 under
    PML (Er 2.7e-01 / Hp 4.5e-01 against 3.6e-07 for the post-add). The split-field
    routing curl -> fu -> field is not MEEP's axis ladder for this rule; the plain
    post-add is exact because Dz's dsig is R, whose sigma is zero on the axis row.
    The curl enters the recurrence NEGATED (``fu = fu*kms - curl``), so an increment
    of ``+axis_coef*Hy`` on the field is ``-axis_coef*Hy`` on the curl. MUST BE
    CAUGHT — and if it ever is not, the fixture has no absorption on the axis row and
    the ``absorbs`` floor has stopped working.
    """
    fold = ("        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;\n"
            "        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);\n")
    replacement = ("        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;\n"
                   "        if (i == 0) curl = curl - (axis_coef * Hy[idx]);\n"
                   "        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);\n")
    out, sites = source.replace(fold, replacement), source.count(fold)
    if sites:
        out, _ = _drop_axis_add_dz(out)
    return out, sites


def _axis_coef_recomputed_in_kernel(source: str) -> Tuple[str, int]:
    """A MEASURED NULL: ``4.0f * dtdx`` in place of the host-rounded ``axis_coef``.

    The array path forms ``4.0 * (dt/dx)`` in float64 and NEP-50 casts that ONE
    scalar to float32 before the multiply (stepping.py:585), so binding the
    already-rounded word is the literal transcription. Scaling by four is EXACT in
    float32 — it only moves the exponent — so the two spellings should agree, and the
    Metal port pins that they do rather than leaving it to luck. Carried as a leg
    rather than as a comment because a claim nobody armed is not a measurement.
    MUST BE UNCAUGHT.
    """
    needle = "Dz[idx] + (axis_coef * Hy[idx])"
    replacement = "Dz[idx] + ((4.0f * dtdx) * Hy[idx])"
    return source.replace(needle, replacement), source.count(needle)


def _restore_r_ghost_hz(source: str) -> Tuple[str, int]:
    """A NULL: the r_to_minus_r near ghost put BACK on Dy's live shift-down operand.

    ``stepping._shift_down``'s CYL_AXIS branch (:1830-1841) images stored row 0 with
    a direction sign and the ``(-1)^m`` phase; at m = 0 the phase is +1 and Hz is a
    z-direction component, so its ghost is ``+Hz[row 0]``. The kernel serves the
    METALLIC zero instead. It is UNOBSERVABLE because ``Dy`` has r-Yee shift 0 and the
    ownership mask zeroes its curl at exactly that row — the claim both sibling tracks
    measured. MUST BE UNCAUGHT; its paired twin
    :func:`_restore_r_ghost_hz_no_mask` drops the mask and MUST be caught, which is
    what makes "unobservable" a demonstration rather than an untested arm.
    """
    needle = "        float ss = shift_dn(Hz, idx, i, nx, sx, bc_x);\n"
    replacement = "        float ss = (i > 0) ? Hz[idx - sx] : (1.0f * Hz[idx]);\n"
    return source.replace(needle, replacement), source.count(needle)


def _restore_r_ghost_hz_no_mask(source: str) -> Tuple[str, int]:
    """The ghost restored AND Dy's r mask dropped — the paired twin. MUST BE CAUGHT.

    The Metal gate found this pairing necessary after arming the ghost on a DEAD
    operand and reading the resulting null as evidence. Here the ghost lands on the
    operand ``Dy``'s curl actually consumes, and with the mask gone the row is no
    longer dropped, so the ghost's value reaches an output word.
    """
    out, sites = _restore_r_ghost_hz(source)
    if not sites:
        return source, 0
    needle = ("        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
              "        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;\n"
              "        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;\n"
              "        pml_apply(Dy, fu_Dy, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);\n")
    replacement = ("        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
                   "        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;\n"
                   "        pml_apply(Dy, fu_Dy, idx, curl, kms_z[k], sinv_z[k], kms_x[i], sinv_x[i]);\n")
    if needle not in out:
        return source, 0
    return out.replace(needle, replacement), sites


def _restore_r_ghost_prefix(source: str) -> Tuple[str, int]:
    """A NULL: the near ghost put back on Dz's live shift-down operand, the PREFIX.

    The term's component name is still ``Hy`` — the substitution swaps the ARRAY, not
    the name (``stepping.step_D``:417-419) — so ``_shift_down``'s CYL_AXIS branch
    would give ``-prefix[row 0]`` at m = 0 (a phi-direction component, sign -1). The
    Metal port measured that on Dz the mask is REDUNDANT: prefix row 0 is an exact
    ``+0.0`` by construction (stepping.py:1314/:1329 zeros_like), so the ghost is
    ``-0.0``, ``-0.0 - (+0.0)`` is ``-0.0``, and the phi self-difference on the
    one-cell invariant axis turns it back into ``+0.0`` at the add. MUST BE UNCAUGHT.
    """
    needle = "        float sf = shift_dn(pfx, idx, i, nx, sx, bc_x);\n"
    replacement = "        float sf = (i > 0) ? pfx[idx - sx] : (-1.0f * pfx[idx]);\n"
    return source.replace(needle, replacement), source.count(needle)


def _restore_r_ghost_prefix_no_mask(source: str) -> Tuple[str, int]:
    """The prefix ghost restored AND Dz's r mask dropped. PREDICTED UNCAUGHT.

    The Metal port measured this twin at 0/8 and gave the reason: the prefix's own
    zero row absorbs the ghost, so the mask on Dz is redundant and dropping it changes
    nothing. This leg carries that CROSS-BACKEND PREDICTION as an armed expectation.
    A CAUGHT verdict here is a FINDING — it would say the prefix row 0 is not an exact
    ``+0.0`` on this backend — not a failure to be hidden.
    """
    out, sites = _restore_r_ghost_prefix(source)
    if not sites:
        return source, 0
    needle = ("        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
              "        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;\n"
              "        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;\n"
              "        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);\n")
    replacement = ("        float curl = dtdx * ((sf - f1) + (f2 - ss));\n"
                   "        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;\n"
                   "        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], sinv_y[j]);\n")
    if needle not in out:
        return source, 0
    return out.replace(needle, replacement), sites


#: This family's own battery, keyed by the sub-step whose source it edits. A mutation
#: armed against the wrong side matches zero sites and is reported NOT ARMED rather
#: than scored, because a mutation silently absent from a battery and a mutation that
#: cannot bite look identical in a verdict.
CYLINDRICAL_SOURCE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "step_B": {
        "drop_bz_prefix_substitution": (_drop_bz_prefix_substitution, True),
        "bz_flat_grouping": (_bz_flat_grouping, True),
        "bz_metallic_ghost_for_wall_row": (_bz_metallic_ghost_for_wall_row, True),
        "drop_axis_zero_bx": (_drop_axis_zero_bx, True),
    },
    "step_D": {
        "drop_dz_prefix_substitution": (_drop_dz_prefix_substitution, True),
        "prefix_into_dx": (_prefix_into_dx, True),
        "drop_axis_zero_dy": (_drop_axis_zero_dy, True),
        "drop_axis_add_dz": (_drop_axis_add_dz, True),
        "fold_axis_add_into_curl": (_fold_axis_add_into_curl, True),
        "restore_r_ghost_hz_no_mask": (_restore_r_ghost_hz_no_mask, True),
        # THE NULLS.
        "axis_coef_recomputed_in_kernel": (_axis_coef_recomputed_in_kernel, False),
        "restore_r_ghost_hz": (_restore_r_ghost_hz, False),
        "restore_r_ghost_prefix": (_restore_r_ghost_prefix, False),
        "restore_r_ghost_prefix_no_mask": (_restore_r_ghost_prefix_no_mask, False),
    },
}

#: HOST mutations: they corrupt the TABLES or the PREFIX rather than the device text,
#: so they run on both backends.
HOST_MUTATIONS: Tuple[str, ...] = (
    "swap_curl_sublattice",   # half-integer vs integer PML positions
    "swap_prefix_ir0",        # the other sub-step's radial weights
    "wall_row_not_zero",      # B side: the wall row filled from the last source row
)
B_ONLY_HOST_MUTATIONS: Tuple[str, ...] = ("wall_row_not_zero",)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and that is
    the one thing about the device library a laptop cannot supply. Everything else the
    fixture exercises — the Dcyl axis table, the phi extent, the coefficient vector
    lengths, the dtype and contiguity — is a real object either way.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def build(xp, spec: Dict[str, Any], courant: float):
    """A frozen ``(fields, layer, grid)`` triple for one Dcyl case.

    ``Grid`` owns the r axis's boundary pair — the axis at r = 0 and a metallic wall
    at r_max — so it is not passed (grid.py:498-508); only z is chosen. The PML is
    asked for the HIGH r face only, for the same reason: cell 0 is the axis, a
    boundary condition rather than a wall.
    """
    shape = tuple(int(n) for n in spec["shape"])
    grid = Grid(resolution=1.0,
                cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=0,
                boundaries={"z": spec["z_kind"]},
                courant=float(courant), xp=xp)
    if tuple(grid.shape) != shape:
        raise ValueError(f"grid built {tuple(grid.shape)}, case asked for {shape}")
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    layer = PML(grid=grid, thickness={"x": (0, max(2, shape[0] // 4)),
                                      "z": max(2, shape[2] // 4)})
    return fields, layer, grid


def seed_state(fields, grid, value_class: str, rng) -> Dict[str, np.ndarray]:
    """Physical-band or subnormal-band values in EVERY array either sub-step touches.

    THE AUXILIARIES START NONZERO in both classes. A zero ``fu`` makes ``kms*prev``
    exactly zero on the first launch whatever ``kms`` holds, so a mis-indexed
    coefficient would only show from step two.

    BOTH sub-steps' volumes are seeded on every case even though one is stepped: the
    D-side axis rule READS ``Hy`` and the B-side prefix reads ``Ey``, and a zero there
    is a fixed point that hides the rule this gate exists to score.
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


def snapshot(fields) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy() for name in STATE}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, sub_step: str) -> None:
    """Move the curl operands between launches, identically on both paths.

    A real run's constitutive sub-step rewrites E (or H) before every curl. Held
    fixed, ``fu`` reaches a fixed point and 60 launches measure what one does — and
    the PREFIX, which is recomputed from those sources, would be a constant.
    """
    for name in SUB_STEP_ARRAYS[sub_step]["sources"]:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# The boundary codes
# ---------------------------------------------------------------------------

def boundary_codes_for(grid) -> Tuple[Tuple[int, int, int], Tuple[str, str, str]]:
    """The SHIPPED mapping, through the predicate module's own function.

    There is no experimental variable here, unlike the folded gate: ``'axis'`` has
    exactly one defensible code and :func:`cylindrical_boundary_codes` REFUSES any
    other r kind by name.
    """
    kinds = tuple(stepping._boundary_kinds(grid, None))
    codes = cyl_coverage.cylindrical_boundary_codes(kinds)
    return tuple(int(c) for c in codes), kinds


# ---------------------------------------------------------------------------
# The coefficient tables and the prefix, plus the host mutations that corrupt them
# ---------------------------------------------------------------------------

def tables_for(sub_step: str, layer) -> Dict[str, Any]:
    suffix = "_h" if SUB_STEP_ARRAYS[sub_step]["half_integer"] else ""
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


def wrong_sub_lattice_tables(sub_step: str, layer) -> Dict[str, Any]:
    """The OTHER sub-step's sub-lattice: a half-cell error, not a crash."""
    suffix = "" if SUB_STEP_ARRAYS[sub_step]["half_integer"] else "_h"
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kms", "sinv")}


def prefix_for(fields, sub_step: str, host_mutation: Optional[str] = None):
    """The prefix this launch feeds the kernel.

    THE SHIPPED HELPER IS CALLED, not re-derived: ``cylindrical_prefix`` extends Ep by
    the zero wall row on the B side and calls ``stepping.cylindrical_rderiv_prefix``
    itself, so the words the kernel reads are the words the array path computed.
    """
    if host_mutation == "swap_prefix_ir0":
        # The other sub-step's ir0: a HALF-CELL error in the radial weights, silent.
        other = "step_D" if sub_step == "step_B" else "step_B"
        from meep_gpu.stepping import _face, _span, cylindrical_rderiv_prefix
        source = getattr(fields, cyl_prefix.PREFIX_COMPONENT[sub_step])
        xp = fields.grid.xp
        if not cyl_prefix.PREFIX_WALL_ROW[sub_step]:
            return cylindrical_rderiv_prefix(xp, source,
                                             cyl_prefix.PREFIX_IR0[other])
        rows = source.shape[0]
        extended = xp.empty((rows + 1,) + source.shape[1:], dtype=source.dtype)
        extended[_span(0, 0, rows)] = source
        extended[_face(0, rows)] = 0
        return cylindrical_rderiv_prefix(xp, extended, cyl_prefix.PREFIX_IR0[other])
    if host_mutation == "wall_row_not_zero":
        # The wall row filled from the LAST SOURCE ROW instead of held at zero
        # (stepping.py:360 assigns 0). In bounds on every shape, and wrong only on the
        # last radial row's forward difference — the plane the array path's zero wall
        # exists to make right.
        from meep_gpu.stepping import _face, _span, cylindrical_rderiv_prefix
        source = getattr(fields, cyl_prefix.PREFIX_COMPONENT[sub_step])
        xp = fields.grid.xp
        rows = source.shape[0]
        extended = xp.empty((rows + 1,) + source.shape[1:], dtype=source.dtype)
        extended[_span(0, 0, rows)] = source
        extended[_face(0, rows)] = source[_face(0, rows - 1)]
        return cylindrical_rderiv_prefix(xp, extended,
                                         cyl_prefix.PREFIX_IR0[sub_step])
    return cyl_prefix.cylindrical_prefix(fields, sub_step, scratch=None)


def host_mutated_tables(name: str, sub_step: str, layer) -> Dict[str, Any]:
    if name == "swap_curl_sublattice":
        return wrong_sub_lattice_tables(sub_step, layer)
    if name in ("swap_prefix_ir0", "wall_row_not_zero"):
        return tables_for(sub_step, layer)  # this one corrupts the PREFIX, not tables
    raise KeyError(f"unknown host mutation {name!r}")


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------

def run_kernel_cuda(sub_step: str, fields, tables, codes, dtdx, prefix) -> None:
    """The SHIPPED kernel, through its own launch wrapper, with the prefix handed in."""
    from meep_gpu.cuda_kernels import cylindrical_kernels  # noqa: PLC0415

    entry = cylindrical_kernels.CYLINDRICAL_LAUNCHERS[sub_step]
    entry(fields, tables, tuple(np.int32(c) for c in codes), dtdx, prefix=prefix)
    cp.cuda.runtime.deviceSynchronize()


def _plane_index(axis: int, index: int, shape) -> Tuple[Any, ...]:
    key: List[Any] = [slice(None)] * 3
    key[axis] = index if index >= 0 else shape[axis] - 1
    return tuple(key)


def _shift_up_numpy(field: np.ndarray, axis: int, bc: int) -> np.ndarray:
    shifted = np.roll(field, -1, axis=axis)
    if bc == BC_METALLIC:
        shifted[_plane_index(axis, -1, field.shape)] = np.float32(0.0)
    return shifted


def _shift_dn_numpy(field: np.ndarray, axis: int, bc: int) -> np.ndarray:
    shifted = np.roll(field, 1, axis=axis)
    if bc == BC_METALLIC:
        shifted[_plane_index(axis, 0, field.shape)] = np.float32(0.0)
    return shifted


def _broadcast(vector, axis: int) -> np.ndarray:
    shape = [1, 1, 1]
    shape[axis] = int(np.asarray(vector).size)
    return np.asarray(vector, dtype=np.float32).reshape(shape)


def run_kernel_numpy(sub_step: str, fields, tables, codes, dtdx, prefix) -> None:
    """The device tree, transcribed, in float32 — the laptop backend.

    THE GROUPING IS THE SHIPPED ONE and it is load-bearing: ``dtdx * ((sf - f1) +
    (f2 - ss))``, never the left-to-right form, and ``dtdx * (pfx_up - pfx_here)``
    on Bz, never the distributed one.

    THIS COMPILES NOTHING AND CERTIFIES NOTHING. It cannot see a defect the NVRTC
    contraction guard exists for and it is not the shipped bytes. What it settles is
    whether the harness can fail.
    """
    spec = SUB_STEP_ARRAYS[sub_step]
    shift = _shift_dn_numpy if spec["backward"] else _shift_up_numpy
    scale = np.float32(dtdx)
    names = "xyz"
    prefix_host = np.asarray(to_host(prefix), dtype=np.float32)
    for (target, aux, first, first_axis, second, second_axis,
         dsig, dsigu) in CURL_TERMS[sub_step]:
        f = getattr(fields, target)
        fu = getattr(fields, aux)
        if sub_step == "step_B" and target == "Bz":
            curl = (scale * (prefix_host[1:] - prefix_host[:-1])).astype(np.float32)
        else:
            if sub_step == "step_D" and target == "Dz":
                f1 = prefix_host
                sf = shift(prefix_host, first_axis, codes[first_axis])
            else:
                f1 = getattr(fields, first)
                sf = shift(f1, first_axis, codes[first_axis])
            f2 = getattr(fields, second)
            ss = shift(f2, second_axis, codes[second_axis])
            curl = (scale * ((sf - f1) + (f2 - ss))).astype(np.float32)
        iyee = IYEE_SHIFTS[target]
        for axis in range(3):
            if iyee[axis] == 0 and codes[axis] == BC_METALLIC:
                curl[_plane_index(axis, 0, curl.shape)] = np.float32(0.0)
        kms = _broadcast(tables[f"kms_{names[dsig]}"], dsig)
        sinv = _broadcast(tables[f"sinv_{names[dsig]}"], dsig)
        kms_u = _broadcast(tables[f"kms_{names[dsigu]}"], dsigu)
        sinv_u = _broadcast(tables[f"sinv_{names[dsigu]}"], dsigu)
        fprev = fu.copy()
        fu_new = (((fprev * kms).astype(np.float32) - curl).astype(np.float32)
                  * sinv).astype(np.float32)
        fu[...] = fu_new
        a = ((f * kms_u).astype(np.float32) + fu_new).astype(np.float32)
        f[...] = ((a - fprev).astype(np.float32) * sinv_u).astype(np.float32)
    axis_row = _plane_index(0, 0, tuple(fields.grid.shape))
    if sub_step == "step_B":
        fields.Bx[axis_row] = np.float32(0.0)
    else:
        axis_coef = np.float32(4.0 * float(dtdx))
        fields.Dz[axis_row] = (fields.Dz[axis_row]
                               + (axis_coef * fields.Hy[axis_row])).astype(np.float32)
        fields.Dy[axis_row] = np.float32(0.0)


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def oracle_moved(before, after, sub_step: str) -> float:
    """Fraction of output WORDS the array path changed from the frozen input.

    Compared as raw uint32, never with ``allclose``: ``-0.0 == 0.0`` and
    ``NaN != NaN`` both lie, and the subnormal-band class puts signed zeros in the
    operands deliberately.
    """
    moved = total = 0
    for name in outputs(sub_step):
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


#: Which target each sub-step's axis rules touch, and what the rule is. Used by the
#: ``axis_row_is_live`` floor: an axis rule applied to a row the oracle never moves is
#: a rule the case cannot score, and the mutation that deletes it would come back
#: UNCAUGHT for a reason about the fixture.
AXIS_RULE_TARGETS: Dict[str, Tuple[str, ...]] = {"step_B": ("Bx",),
                                                 "step_D": ("Dy", "Dz")}


def axis_row_is_live(frozen, reference, sub_step: str, shape) -> Dict[str, Any]:
    out: Dict[str, Any] = {"targets": [], "moved": 0, "words": 0}
    axis_row = _plane_index(0, 0, tuple(shape))
    for target in AXIS_RULE_TARGETS[sub_step]:
        before = np.ascontiguousarray(frozen[target], dtype=np.float32)[axis_row]
        after = np.ascontiguousarray(reference[target], dtype=np.float32)[axis_row]
        moved = int(np.count_nonzero(before.ravel().view(np.uint32)
                                     != after.ravel().view(np.uint32)))
        out["targets"].append({"target": target, "moved": moved,
                               "words": int(before.size)})
        out["moved"] += moved
        out["words"] += int(before.size)
    out["meets_floor"] = out["moved"] > 0
    return out


def prefix_is_live(prefix, reference, sub_step: str) -> Dict[str, Any]:
    """Is the prefix non-constant down r, and did the prefixed target move?

    A flat prefix makes every prefix defect invisible: the forward difference is zero
    everywhere and the substitution and its absence agree.
    """
    host = np.ascontiguousarray(to_host(prefix), dtype=np.float32)
    diffs = host[1:] - host[:-1]
    target = "Bz" if sub_step == "step_B" else "Dz"
    return {
        "rows": int(host.shape[0]),
        "distinct_radial_differences": int(np.count_nonzero(diffs)),
        "max_abs": float(np.max(np.abs(host))) if host.size else 0.0,
        "row0_all_exact_positive_zero": bool(
            np.all(np.ascontiguousarray(host[0], dtype=np.float32)
                   .ravel().view(np.uint32) == 0)),
        "prefixed_target": target,
        "meets_floor": int(np.count_nonzero(diffs)) > 0,
    }


def absorbs(sub_step: str, layer) -> Dict[str, Any]:
    """Do the r and z coefficient profiles differ from the identity?"""
    tables = tables_for(sub_step, layer)
    out: Dict[str, Any] = {"axes": [], "min_deviation": None}
    for axis, name in (("x", "r"), ("z", "z")):
        deviation = 0.0
        for stem in ("kms", "sinv"):
            values = to_host(tables[f"{stem}_{axis}"]).astype(np.float64)
            deviation = max(deviation, float(np.max(np.abs(values - 1.0))))
        out["axes"].append({"axis": axis, "role": name,
                            "max_deviation_from_identity": deviation})
        out["min_deviation"] = (deviation if out["min_deviation"] is None
                                else min(out["min_deviation"], deviation))
    out["meets_floor"] = (out["min_deviation"] or 0.0) > 0.0
    return out


def structure_facts(grid) -> Dict[str, Any]:
    return {
        "shape": [int(n) for n in grid.shape],
        "cylindrical": bool(getattr(grid, "cylindrical", False)),
        "m": int(getattr(grid, "m", -1)),
        "is_axis": [bool(grid.is_axis(a)) for a in range(3)],
        "metallic": [bool(grid.is_metallic(a)) for a in range(3)],
        "mirrored": [bool(grid.is_mirrored(a)) for a in range(3)],
        "stored_past_owned": [int(grid.stored_cells(a)) - int(grid.owned_cells(a))
                              for a in range(3)],
        "accurate_fields_near_cylorigin": bool(
            getattr(grid, "accurate_fields_near_cylorigin", False)),
        "boundary_kinds": list(stepping._boundary_kinds(grid, None)),
    }


def check_terms_against_stepping() -> Dict[str, Any]:
    """Pin :data:`CURL_TERMS` against ``stepping``'s own term tables."""
    out: Dict[str, Any] = {"agreed": True, "checked": [], "disagreements": []}
    for sub_step, source in (("step_B", stepping.B_CURL_TERMS),
                             ("step_D", stepping.D_CURL_TERMS)):
        for mine, theirs in zip(CURL_TERMS[sub_step], source):
            axis_of = {"x": 0, "y": 1, "z": 2}
            record = {
                "sub_step": sub_step, "target": mine[0],
                "mine": [mine[0], mine[2], mine[3], mine[4], mine[5], mine[6], mine[7]],
                "stepping": [theirs.target, theirs.first, theirs.first_axis,
                             theirs.second, theirs.second_axis,
                             axis_of[theirs.dsig], axis_of[theirs.dsigu]],
            }
            record["agrees"] = record["mine"] == record["stepping"]
            out["checked"].append(record)
            if not record["agrees"]:
                out["agreed"] = False
                out["disagreements"].append(record)
    return out


def check_axis_rules_against_stepping() -> Dict[str, Any]:
    """Read the m = 0 axis rules OFF ``stepping``'s source, not off memory.

    The kernel hard-codes three rules and one of them is an ARITHMETIC one. This
    checks that ``_cylindrical_axis_increment_B`` and ``_D`` really do return ``None``
    at m = 0 — the fact that licenses the kernel having no curl-row fold — by CALLING
    them on a real m = 0 grid rather than by reading the branch.
    """
    grid = Grid(resolution=1.0, cell_size=(8.0, 0.0, 8.0), cylindrical=True, m=0,
                boundaries={"z": "metallic"}, courant=0.5, xp=np)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    electric = {n: getattr(fields, n) for n in ("Ex", "Ey", "Ez")}
    magnetic = {n: getattr(fields, n) for n in ("Hx", "Hy", "Hz")}
    boundaries = stepping._boundary_kinds(grid, None)
    phases = (None, None, None)
    dtdx = float(grid.dt / grid.dx)
    increment_B = stepping._cylindrical_axis_increment_B(fields, electric, boundaries,
                                                         phases, dtdx)
    increment_D = stepping._cylindrical_axis_increment_D(fields, magnetic, boundaries,
                                                         phases, dtdx)
    return {
        "axis_increment_B_at_m0": increment_B,
        "axis_increment_D_at_m0": increment_D,
        "both_none": increment_B is None and increment_D is None,
        "why_it_matters": ("at m = 0 neither side folds an increment into the curl "
                           "row, which is why the kernel carries only the two post-"
                           "rules; a non-None here would mean the kernel is missing a "
                           "term"),
    }


# ---------------------------------------------------------------------------
# The scan-order leg: is cupy.cumsum reproducible by a column-serial CUDA scan?
# ---------------------------------------------------------------------------

_SERIAL_SCAN_SOURCE = r'''
extern "C" __global__ void column_serial_scan(
    float* __restrict__ out, const float* __restrict__ inc,
    int nx, int ny, int nz, int n_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_cols) return;
    int nyz = ny * nz;
    int k = idx % nz;
    int j = idx / nz;
    if (j >= ny) return;
    int base = j * nz + k;
    float acc = inc[base];
    out[base] = acc;
    for (int i = 1; i < nx; ++i) {
        int o = i * nyz + base;
        acc = acc + inc[o];
        out[o] = acc;
    }
}
'''


def scan_order_leg(shapes: Sequence[Tuple[int, int, int]], rng) -> Dict[str, Any]:
    """Does a column-serial CUDA scan reproduce ``cupy.cumsum`` word for word?

    THE POINT IS NOT TO SHIP THIS SCAN. It is to convert the sibling track's reading —
    "the CuPy prefix cannot be reproduced by any other summation order, so the scan
    stays on the array path" — into a measurement ON THIS BACKEND. Whatever it says,
    the kernels do not change: they consume the array path's prefix either way. What
    changes is whether the design decision is measured here or borrowed.
    """
    out: Dict[str, Any] = {"cases": [], "cupy_version": getattr(cp, "__version__", None)}
    kernel = cp.RawKernel(_SERIAL_SCAN_SOURCE, "column_serial_scan",
                          options=("--fmad=false",))
    threads = 128
    for shape in shapes:
        nx, ny, nz = shape
        host = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        inc = cp.asarray(host)
        reference = cp.cumsum(inc, axis=0)
        produced = cp.empty_like(inc)
        cols = ny * nz
        blocks = (cols + threads - 1) // threads
        kernel((blocks,), (threads,), (produced, inc, np.int32(nx), np.int32(ny),
                                       np.int32(nz), np.int32(cols)))
        cp.cuda.runtime.deviceSynchronize()
        a = to_host(reference).astype(np.float32).ravel().view(np.uint32)
        b = to_host(produced).astype(np.float32).ravel().view(np.uint32)
        differing = int(np.count_nonzero(a != b))
        # NumPy's cumsum IS a sequential accumulation; comparing the serial CUDA scan
        # against it too says which of the two the device kernel actually reproduces.
        numpy_reference = np.cumsum(host, axis=0, dtype=np.float32)
        c = np.ascontiguousarray(numpy_reference, np.float32).ravel().view(np.uint32)
        out["cases"].append({
            "shape": list(shape),
            "words": int(a.size),
            "serial_cuda_vs_cupy_cumsum_differing": differing,
            "serial_cuda_vs_numpy_cumsum_differing": int(np.count_nonzero(c != b)),
            "cupy_vs_numpy_cumsum_differing": int(np.count_nonzero(a != c)),
        })
        log(f"[scan] shape={shape} serial_vs_cupy={differing}/{a.size} "
            f"serial_vs_numpy={out['cases'][-1]['serial_cuda_vs_numpy_cumsum_differing']} "
            f"cupy_vs_numpy={out['cases'][-1]['cupy_vs_numpy_cumsum_differing']}")
    out["serial_cuda_reproduces_cupy"] = all(
        c["serial_cuda_vs_cupy_cumsum_differing"] == 0 for c in out["cases"])
    out["reading"] = (
        "a column-serial CUDA scan IS bit-equal to cupy.cumsum on these shapes: the "
        "prefix COULD be fused, and the array-path decision is a cost choice rather "
        "than a correctness one"
        if out["serial_cuda_reproduces_cupy"] else
        "a column-serial CUDA scan is NOT bit-equal to cupy.cumsum: the oracle's "
        "summation order is CuPy's own and the prefix cannot be fused bit-identically "
        "on this backend — MEASURED HERE, not inherited")
    return out


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend: str, spec: Dict[str, Any], sub_step: str, courant: float,
             value_class: str, guard: str,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML`` objects.
    There is no second transcription on the oracle leg to drift: the kernel is
    compared against the thing it claims to reproduce, byte for byte, from ONE frozen
    state — oracle runs, state is restored, kernel runs.
    """
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{sub_step}|{courant}|{value_class}".encode()).digest()[:4], "big"))

    fields, layer, grid = build(xp, spec, courant)
    host = seed_state(fields, grid, value_class, rng)
    dtdx = float(grid.dt / grid.dx)
    codes, kinds = boundary_codes_for(grid)

    case: Dict[str, Any] = {
        "label": spec["label"], "sub_step": sub_step, "courant": courant,
        "value_class": value_class, "guard": guard, "backend": backend,
        "host_mutation": host_mutation,
        "z_kind": spec["z_kind"],
        "boundary_codes": [int(c) for c in codes],
        "resolved_kinds": list(kinds),
        "dtdx": dtdx,
        "structure": structure_facts(grid),
        "operand_census": operand_census(host),
    }

    covered, reason = cyl_coverage.covers_real_pml_cylindrical_curl(
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
    (stepping.step_B if sub_step == "step_B" else stepping.step_D)(fields, layer)
    reference = snapshot(fields)

    moved = oracle_moved(frozen, reference, sub_step)
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; a case that moved nothing certifies nothing")
        case["seconds"] = time.time() - started
        return case

    live = axis_row_is_live(frozen, reference, sub_step, grid.shape)
    case["axis_row_is_live"] = live
    if not live["meets_floor"]:
        case["skipped"] = ("the r = 0 row carries no moved word on the targets the "
                           "axis rules touch; this case cannot score them")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, from the SAME frozen state, with the prefix computed from
    # the SAME sources the oracle differenced.
    restore(fields, frozen)
    tables = (tables_for(sub_step, layer) if host_mutation is None
              else host_mutated_tables(host_mutation, sub_step, layer))
    prefix = prefix_for(fields, sub_step, host_mutation)
    liveness = prefix_is_live(prefix, reference, sub_step)
    case["prefix_is_live"] = liveness
    if not liveness["meets_floor"]:
        case["skipped"] = ("the prefix is constant down r; every prefix defect is "
                           "invisible on this case")
        case["seconds"] = time.time() - started
        return case

    runner = run_kernel_cuda if backend == "cuda" else run_kernel_numpy
    runner(sub_step, fields, tables, codes, dtdx, prefix)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(sub_step)}
    case["single_launch"] = combine(parts)
    case["per_array"] = {
        name: {"differing_words": int(parts[name]["differing_floats"]),
               "words": int(parts[name]["total_floats"])}
        for name in outputs(sub_step)}

    # Leg 3: the multi-step. ``fu`` IS STATE. ONE ``Fields`` object, run twice from
    # the SAME frozen state, with the prefix RECOMPUTED every launch on both paths —
    # which is what a real driver does and what a cached prefix would get wrong.
    if host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            (stepping.step_B if sub_step == "step_B" else stepping.step_D)(fields, layer)
            advance_sources(fields, sub_step)
        oracle_multi = snapshot(fields)

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(sub_step, fields, tables, codes, dtdx,
                   prefix_for(fields, sub_step, None))
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

def case_product(product: str):
    specs = CASES if product == "full" else CASES[:2]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = VALUE_CLASSES if product == "full" else ("uniform",)
    return [(spec, sub_step, courant, value_class)
            for spec in specs for sub_step in SUB_STEPS
            for courant in courants for value_class in classes]


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
                f"SKIPPED: {case['skipped'][:70]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        log(f"[{guard}] case {index}/{len(plan)} {spec['label']} {sub_step} "
            f"c={courant} {value_class} shape={case['structure']['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"moved={case['oracle_moved']:.3f} ({case['seconds']:.1f} s)")
    return cases


#: The specs a mutation leg is scored on. CHOSEN, not sliced off the front: both z
#: terminations and an odd radial extent must appear, because the ownership mask and
#: the wall row live on those.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "r16_z20_metallic", "r16_z20_periodic", "r9_z17_metallic", "r12_z48_metallic")


def scorable_baselines(results) -> Tuple[set, List[Dict[str, Any]]]:
    """The ``(label, sub_step)`` pairs whose UNMUTATED case was bit-identical.

    A MUTATION LEG ONLY MEANS SOMETHING WHERE THE BASELINE IS IDENTICAL. On an arm
    that already diverges, every leg — including one that must be UNCAUGHT — scores
    CAUGHT for free, because the divergence it is credited with was already there.
    Read off the primary sweep rather than re-run, so it cannot disagree with itself.
    """
    baseline: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for case in results.get("sweep", {}).get("fmad_false", []):
        if (case.get("courant") != INEXACT_COURANT
                or case.get("value_class") != "uniform"):
            continue
        baseline[(case["label"], case["sub_step"])] = case
    scorable, excluded = set(), []
    for key, case in baseline.items():
        if key[0] not in MUTATION_SPEC_LABELS:
            continue
        if case.get("skipped"):
            excluded.append({"label": key[0], "sub_step": key[1],
                             "why": f"baseline skipped: {case['skipped'][:80]}"})
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical", True)
        if single and multi:
            scorable.add(key)
        else:
            excluded.append({"label": key[0], "sub_step": key[1],
                             "why": ("the unmutated baseline already diverges here, "
                                     "so every leg would score CAUGHT for free")})
    return scorable, excluded


def mutation_plan(scorable: Optional[set] = None):
    by_label = {spec["label"]: spec for spec in CASES}
    plan = [(by_label[label], sub_step, INEXACT_COURANT, "uniform")
            for label in MUTATION_SPEC_LABELS for sub_step in SUB_STEPS]
    if scorable is None:
        return plan
    return [entry for entry in plan if (entry[0]["label"], entry[1]) in scorable]


def leg_caught(case: Dict[str, Any]) -> bool:
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def _score(legs: List[Dict[str, Any]], must_be_caught: bool) -> Dict[str, Any]:
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


def run_host_mutations(results, out_path, backend, scorable) -> Dict[str, Any]:
    """Every table- and prefix-level defect. A gate that cannot fail certifies nothing."""
    out: Dict[str, Any] = {}
    plan = mutation_plan(scorable)
    for name in HOST_MUTATIONS:
        legs: List[Dict[str, Any]] = []
        for spec, sub_step, courant, value_class in plan:
            if name in B_ONLY_HOST_MUTATIONS and sub_step != "step_B":
                continue
            legs.append(one_case(backend, spec, sub_step, courant, value_class,
                                 "fmad_false", host_mutation=name))
        out[name] = dict(_score(legs, True), cases=legs)
        log(f"[host-mut] {name}: caught {out[name]['caught']}/{out[name]['ran']} "
            f"-> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def run_source_mutations(results, out_path, scorable) -> Dict[str, Any]:
    """Every device-text defect, applied to the SHIPPED strings and recompiled.

    The compile memo is keyed through the SOURCE, so a mutated body is a miss and
    reaches NVRTC. Every leg records how many constructions came from the mutated
    bytes: a leg reporting a pass for a mutation it never applied is worse than no leg.
    """
    from meep_gpu.cuda_kernels import compile_cache, cylindrical_kernels  # noqa: PLC0415

    attribute_for = {"step_B": "_cyl_step_B_pml_real_kernel_code",
                     "step_D": "_cyl_step_D_pml_real_kernel_code"}
    originals = {name: getattr(cylindrical_kernels, attribute)
                 for name, attribute in attribute_for.items()}
    out: Dict[str, Any] = {}
    plan = mutation_plan(scorable)

    def _battery(sub_step: str):
        for name in SHARED_SOURCE_MUTATIONS:
            yield name, probe.SOURCE_MUTATIONS[name], name not in SHARED_NULL_MUTATIONS
        for name, (transform, must) in CYLINDRICAL_SOURCE_MUTATIONS[sub_step].items():
            yield name, transform, must

    try:
        for sub_step in SUB_STEPS:
            attribute = attribute_for[sub_step]
            for name, transform, must_be_caught in _battery(sub_step):
                mutated, sites = transform(originals[sub_step])
                key = f"{sub_step}:{name}"
                if sites == 0 or mutated == originals[sub_step]:
                    out[key] = {"armed": False, "must_be_caught": must_be_caught,
                                "why": (f"matched {sites} site(s) and changed nothing; "
                                        f"the mutation and the kernel have drifted "
                                        f"apart")}
                    log(f"[src-mut] {key}: NOT ARMED ({sites} sites)")
                    results["source_mutations"] = out
                    save(results, out_path)
                    continue
                setattr(cylindrical_kernels, attribute, mutated)
                cylindrical_kernels._clear_kernel_cache()
                compile_cache.clear_compile_log()
                digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
                legs = []
                for spec, leg_sub_step, courant, value_class in plan:
                    if leg_sub_step != sub_step:
                        continue
                    legs.append(one_case("cuda", spec, sub_step, courant,
                                         value_class, "fmad_false"))
                setattr(cylindrical_kernels, attribute, originals[sub_step])
                cylindrical_kernels._clear_kernel_cache()
                from_mutated = sum(1 for entry in compile_cache.compile_log()
                                   if entry["source_sha256"] == digest)
                scored = _score(legs, must_be_caught)
                out[key] = dict(scored, armed=True, sites=sites,
                                mutated_source_sha256=digest,
                                kernel_constructions_from_mutated_bytes=from_mutated,
                                caught_at_single_launch=sum(
                                    1 for c in legs if not c.get("skipped")
                                    and not c["single_launch"]["bit_identical"]),
                                cases=legs)
                if from_mutated == 0 and scored["ran"]:
                    out[key]["verdict"] = "UNACCOUNTED"
                    out[key]["why"] = ("no kernel construction used the mutated bytes; "
                                       "this leg did not exercise the mutation")
                log(f"[src-mut] {key}: caught {scored['caught']}/{scored['ran']} "
                    f"builds_from_mutated={from_mutated} -> {out[key]['verdict']}")
                results["source_mutations"] = out
                save(results, out_path)
    finally:
        for sub_step, attribute in attribute_for.items():
            setattr(cylindrical_kernels, attribute, originals[sub_step])
        cylindrical_kernels._clear_kernel_cache()
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def policy_is_non_vacuous(results, after_stamp) -> Dict[str, Any]:
    """Did this run actually compile UNDER the policy it claims, and can it prove it?

    THE TWO POLICIES ARE EVIDENCED DIFFERENTLY, and a floor that demanded one number
    of both is what the first cut of this gate got wrong — it refused a perfectly
    good ``flush`` leg for reporting zero strips, which is precisely what a policy
    that installs NO STRIP is supposed to report. The refusal was a defect in the
    floor, and the gate surfacing it is the floor working.

    * ``keep`` (``ieee_keep_ftz_stripped``) WRAPS ``compile_using_nvrtc`` and removes
      CuPy's ``-ftz=true``. Its evidence is the STRIP COUNTERS: ``ftz_removed`` must
      be nonzero after the run, and no ``-ftz=true`` may have reached NVRTC. Zero
      strips here is exactly the shared-cache failure this track has already paid for
      — an artifact whose counters read zero cannot tell an installed policy from an
      inert one.
    * ``flush`` (``meep_x86_flush``) installs NO NVRTC wrapper at all: it moves the
      HOST FPU and lets CuPy's default through. Its counters are zero BY DESIGN, and
      its evidence is the opposite observation — ``-ftz=true`` DID reach NVRTC.

    Both need the observer to have seen NVRTC run at all: a leg served entirely from
    a warm disk cache compiled nothing and is evidence for neither.
    """
    stamp = (after_stamp or {}).get("stamp") or {}
    report = results.get("nvrtc_binary_report") or {}
    resolved = stamp.get("resolved")
    calls = int(report.get("nvrtc_calls_observed") or 0)
    reached = report.get("any_ftz_true_reached_nvrtc")
    out: Dict[str, Any] = {
        "resolved": resolved, "policy": stamp.get("policy"),
        "installed": stamp.get("installed"),
        "nvrtc_calls_observed": calls,
        "post_run_nvrtc_calls": stamp.get("nvrtc_calls"),
        "post_run_ftz_removed": stamp.get("ftz_removed"),
        "any_ftz_true_reached_nvrtc": reached,
    }
    if calls == 0:
        out.update(meets_floor=False,
                   why=("NVRTC was never observed compiling: everything came from a "
                        "warm cache, so this artifact is evidence for no policy"))
        return out
    if resolved == "keep":
        removed = int(stamp.get("ftz_removed") or 0)
        ok = removed > 0 and reached is False
        out.update(meets_floor=ok, why=(
            "ok: the strip fired and no -ftz=true reached NVRTC" if ok else
            f"policy 'keep' strips -ftz=true, but ftz_removed={removed} and "
            f"any_ftz_true_reached_nvrtc={reached}; an artifact whose strip counters "
            f"read zero cannot tell an installed policy from an inert one"))
        return out
    if resolved == "flush":
        ok = reached is True
        out.update(meets_floor=ok, why=(
            "ok: no strip is installed and -ftz=true reached NVRTC, which is what "
            "this policy IS" if ok else
            f"policy 'flush' installs no strip, so its evidence is that -ftz=true "
            f"reached NVRTC; the observer says {reached}"))
        return out
    out.update(meets_floor=False,
               why=f"no policy was installed (resolved={resolved!r}); these bytes "
                   f"belong to whatever CuPy defaulted to")
    return out


def summarize(results) -> Dict[str, Any]:
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]

    per_sub_step: Dict[str, Dict[str, Any]] = {}
    for case in scored:
        entry = per_sub_step.setdefault(case["sub_step"], {
            "cases": 0, "single_identical": 0, "multi_cases": 0,
            "multi_identical": 0, "labels": set(), "courants": set(),
            "value_classes": set(), "differing_words": 0})
        entry["cases"] += 1
        entry["labels"].add(case["label"])
        entry["courants"].add(case["courant"])
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
        for field in ("labels", "courants", "value_classes"):
            entry[field] = sorted(entry[field])

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

    stamps = (results.get("subnormal_policy_stamp") or {},
              results.get("subnormal_policy_stamp_after_run") or {})
    policy_evidence = policy_is_non_vacuous(results, stamps[1])
    if results.get("backend") == "cuda" and not policy_evidence["meets_floor"]:
        reasons.append(f"the subnormal policy is not evidenced by this run: "
                       f"{policy_evidence['why']}")
    if not results.get("curl_terms_vs_stepping", {}).get("agreed", False):
        reasons.append("the transcribed curl term table disagrees with stepping's")
    if not results.get("axis_rules_vs_stepping", {}).get("both_none", False):
        reasons.append("stepping folds an axis increment into the curl at m = 0; the "
                       "kernel carries no such term")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is not inert")
        elif leg["verdict"] != "CAUGHT":
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for key, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"source mutation {key} was not armed: {leg.get('why')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"source mutation {key} is {leg['verdict']}")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT]
    guarded_identical = {(c["label"], c["sub_step"], c["courant"], c["value_class"])
                         for c in scored if c["single_launch"]["bit_identical"]}
    comparable = [c for c in control
                  if (c["label"], c["sub_step"], c["courant"], c["value_class"])
                  in guarded_identical]
    control_diverged = [c for c in comparable if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "comparable_where_guarded_leg_was_identical": len(comparable),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run: no unguarded leg was scored where the "
                    "guarded leg was identical" if not comparable else
                    "the contraction guard is load-bearing on this sub-step"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too, so this run carries no evidence that "
                    "--fmad=false changed an answer here"),
    }

    return {
        "released": not reasons,
        "reasons": reasons,
        "per_sub_step": per_sub_step,
        "guard_control": guard_control,
        "policy_stamps": {"at_install": stamps[0].get("stamp"),
                          "after_run": stamps[1].get("stamp")},
        "policy_evidence": policy_evidence,
        "scan_order": results.get("scan_order", {}).get("reading"),
        "scored_cases": len(scored),
        "claim": ("on a real Dcyl grid at m = 0 with real float32 storage and an "
                  "active split-field PML, cyl_step_B_pml_real and "
                  "cyl_step_D_pml_real are byte-identical to stepping.step_B / "
                  "stepping.step_D per sub-step, and over "
                  f"{MULTI_STEP_BUDGET} consecutive launches, on the case matrix in "
                  "CASES"),
        "does_not_claim": [
            "no throughput claim: the radial prefix stays on the array path, so this "
            "pair cannot go below the prefix's own launches and nothing here was timed",
            "|m| >= 1 is NOT covered and is refused by name: complex storage, the "
            "i*m/r coupling and the per-|m| axis rules are a second kernel",
            "nothing dispatches these kernels; fastpath.plan_fast_path is untouched",
            "the r near ghost is served as the METALLIC zero; that choice is measured "
            "as a NULL here and cannot be certified by any byte gate from inside the "
            "kernel",
            "a folded, Bloch, BFAST, beta, conductive, nonlinear or complex-storage "
            "Dcyl run is refused by the predicate, not measured here",
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
    parser.add_argument("--backend", choices=("cuda", "numpy"), default="cuda")
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_cylindrical_real",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("do the NEW hand-CUDA cylindrical curl kernels reproduce "
                     "stepping.step_B / step_D word for word on a Dcyl grid at m = 0?"),
        "curl_terms_vs_stepping": check_terms_against_stepping(),
        "axis_rules_vs_stepping": check_axis_rules_against_stepping(),
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
    else:
        results["environment"] = {"python": sys.version.split()[0],
                                  "numpy_version": np.__version__,
                                  "note": ("NumPy backend: compiles nothing, "
                                           "certifies nothing")}
    save(results, args.out)

    if args.backend == "cuda":
        results["scan_order"] = scan_order_leg(
            tuple(spec["shape"] for spec in CASES),
            np.random.default_rng(SEED + 1))
        save(results, args.out)

    for guard, options, _is_primary in GUARD_SETS:
        if args.backend == "numpy" and guard != "fmad_false":
            continue
        if args.backend == "cuda":
            from meep_gpu.cuda_kernels import cylindrical_kernels  # noqa: PLC0415
            cylindrical_kernels._COMPILE_OPTIONS = tuple(options)
            cylindrical_kernels._clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.backend, args.product, guard)

    if args.backend == "cuda":
        from meep_gpu.cuda_kernels import cylindrical_kernels  # noqa: PLC0415
        cylindrical_kernels._COMPILE_OPTIONS = ("--fmad=false",)
        cylindrical_kernels._clear_kernel_cache()

    if not args.skip_mutations:
        scorable, excluded = scorable_baselines(results)
        results["mutation_scope"] = {
            "scorable": sorted("::".join(k) for k in scorable),
            "excluded": excluded,
            "why": ("a mutation leg answers 'can this gate see this defect', and only "
                    "an arm whose unmutated baseline is bit-identical can answer it"),
        }
        log(f"[mut-scope] {len(scorable)} scorable (label, sub_step) pairs; "
            f"{len(excluded)} excluded")
        save(results, args.out)
        run_host_mutations(results, args.out, args.backend, scorable)
        if args.backend == "cuda":
            run_source_mutations(results, args.out, scorable)

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
        # THE POLICY STAMP, RE-READ AFTER EVERYTHING COMPILED. The stamp taken at
        # install time necessarily reports zero NVRTC calls and zero strips — nothing
        # had compiled yet — and an artifact whose strip counters read zero cannot
        # tell an INSTALLED policy from an INERT one, which is the exact failure a
        # shared CuPy cache produced on this track before. Read again at the end, the
        # counters are this run's, and the pair of stamps says which is which.
        results["subnormal_policy_stamp_after_run"] = \
            probe.subnormal_policy_stamp(_REPO_API)
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} scored={verdict['scored_cases']}")
    for name, entry in sorted(verdict["per_sub_step"].items()):
        log(f"[verdict]   {name}: single {entry['single_identical']}/{entry['cases']} "
            f"multi {entry['multi_identical']}/{entry['multi_cases']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
