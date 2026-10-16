"""Sub-step byte-identity gate for the NEW hand-CUDA CYLINDRICAL curl pair, m = 0.

THE CLAIM UNDER TEST. ``meep_gpu/cuda_kernels/cylindrical_kernels.py`` ships two
kernels -- ``cyl_step_B_pml_real`` and ``cyl_step_D_pml_real`` -- and this gate asks
whether, with the column-serial radial scan in front of each (``cylindrical_prefix``),
they reproduce ``stepping.step_B`` / ``stepping.step_D`` WORD FOR WORD on a real Dcyl
grid at m = 0 under a split-field PML, with MEEP's per-chunk forms (bit_parity_0_10
DESIGN.md D1, D6, D10, D11; AP6c's Dcyl forms). Every launch binds the run's chunk
record ``cb``.

=============================================================================
WHAT CYLINDRICAL ADDS, AND THEREFORE WHAT THIS GATE HAS TO BE ABLE TO SEE
=============================================================================

Four things (``cylindrical_kernels``' docstring derives them off ``stepping.py``). The
battery is built around them, because a gate that only re-runs the Cartesian battery
would certify a Dcyl kernel against Cartesian defects:

1. the RADIAL DIFFERENCE of MEEP's per-chunk ``f_rderiv_int``, with the zero wall row on
   the B side -- the scan's, measured by ``gate_cuda_cylindrical_scan.py`` and re-read
   here by leg ``scan_order``;
2. Bz's whole curl REPLACED by ``dtdx`` times that difference, forward;
3. Dz's whole curl REPLACED by ``dtdx`` times the backward difference, while Dx's Hy
   operands stay RAW;
4. the m = 0 AXIS RULES: ``Bx[r=0] = 0``; ``Dz[r=0] += realnum(double(4*Courant) *
   double(Hp[r=0]))`` as a POST-ADD (one double product, rounded once), then
   ``Dy[r=0] = 0``.

And the per-chunk PML forms every root kernel now carries (KERNELS.md section 7.2):
the recurrence on the cells whose chunk carries the dsigu layer, the dsig-only form
elsewhere, the auxiliary touched only on the recurrence's cells -- armed as the B
battery, with the record's bounds, lattices and flags as host and source mutations.

=============================================================================
THE SCAN IS MEASURED, AND ITS VERDICT IS ITS OWN GATE'S
=============================================================================

The kernels read the radial difference the shipped scan writes, so a diverging case
could be the scan's or the curl's. Leg ``scan_order`` runs the scan gate's identity
comparison (``gate_cuda_cylindrical_scan.identity_leg``) on this gate's shapes, so the
answer to the first question is in this artifact beside the second. The scan's whole
battery -- complex storage, many-chunk divisions, its own mutations, both policies --
is ``gate_cuda_cylindrical_scan.py``'s.

=============================================================================
THE FLOORS THAT REFUSE A VACUOUS PASS
=============================================================================

* ``oracle_moved`` — the fraction of output words the array path changed from the
  frozen input. A case that moved nothing is refused, not passed.
* ``axis_row_is_live`` — the r = 0 row must carry MOVED words on the targets the axis
  rules touch. An axis rule applied to a row nothing writes is a rule this case
  cannot score, and all three axis mutations would come back UNCAUGHT for a reason
  about the fixture.
* ``radial_is_live`` — the radial difference must be non-zero and non-constant down
  r. A flat difference makes every radial-substitution defect invisible.
* ``absorbs`` — ``max|kms-1|`` and ``max|sinv-1|`` on the r AND z axes. If a
  coefficient vector is the identity everywhere, a coefficient-index error on that
  axis cannot be seen.
* ``INEXACT_COURANT`` — 0.5 is exactly representable in float32, so ``4*Courant`` is a
  float32 word there and the on-axis product formed in float32 is the same word as
  MEEP's double product. A case matrix without a non-power-of-two Courant cannot see
  that defect at all, so the primary courant here is 0.35 and 0.5 is carried beside it
  rather than instead of it.
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
Correctness only — no throughput claim is made.
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
import gate_chunk_records  # noqa: E402
import gate_cuda_cylindrical_scan as scan_gate  # noqa: E402
import gate_needles  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.chunk_map import RECORD_PRESENCE_WORD, record_index  # noqa: E402
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
#: also moves the RADIAL DIFFERENCE every launch, which is the point on this family: a
#: difference rescanned from a frozen source is a constant the kernel could have cached.
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
#: case, because ``step_D``'s axis rule READS ``Hy`` and ``step_B``'s scan reads
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
    # A WIDER RADIAL EXTENT: the scan's accumulated sum runs further, which is
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
#: THIS GATE'S NEEDLE TABLE for the shared battery above, every one borrowed from the
#: probe's shared default. The pair lifts the root pair's prelude (``curl_stencil``,
#: the chunk-keyed ``pml_apply``), which is the text the shared default is spelled on
#: (``regroup_stencil`` inverted to the pairwise grouping the pair carried until 0.10;
#: ``gate_needles.py``).
SHARED_NEEDLES = gate_needles.gate_table(
    "gate_cuda_cylindrical_real", probe.SOURCE_MUTATIONS, SHARED_SOURCE_MUTATIONS)
SHARED_NULL_MUTATIONS: Tuple[str, ...] = probe.PML_NULL_MUTATIONS


# ---------------------------------------------------------------------------
# The CYLINDRICAL mutations — this family's own, one per addition it makes
# ---------------------------------------------------------------------------

def _needle(old: str, new: str):
    """A plain text rewrite: ``(mutated source, sites)``."""
    def transform(source: str) -> Tuple[str, int]:
        return source.replace(old, new), source.count(old)
    return transform


#: Bz's and Dz's whole curl, as the kernels spell it: dtdx times the scan's radial
#: difference. One site per side.
_RADIAL_CURL = "        float curl = dtdx * rad[idx];\n"

#: The Cartesian four-operand curls the radial substitution replaces, on the term
#: table's operands (``stepping.B_CURL_TERMS[2]`` / ``D_CURL_TERMS[2]``).
_CARTESIAN_BZ = ("        float f1 = Ey[idx];\n"
                 "        float sf = shift_up(Ey, idx, i, nx, sx, bc_x);\n"
                 "        float f2 = Ex[idx];\n"
                 "        float ss = shift_up(Ex, idx, j, ny, sy, bc_y);\n"
                 "        float curl = curl_stencil(dtdx, sf, f1, f2, ss, pr_x, pr_y);\n")
_CARTESIAN_DZ = ("        float f1 = Hy[idx];\n"
                 "        float sf = shift_dn(Hy, idx, i, nx, sx, bc_x);\n"
                 "        float f2 = Hx[idx];\n"
                 "        float ss = shift_dn(Hx, idx, j, ny, sy, bc_y);\n"
                 "        float curl = curl_stencil(dtdx, sf, f1, f2, ss, pr_x, pr_y);\n")

#: The m = 0 on-axis Dz rule as the D kernel spells it (MEEP's double product,
#: step_db.cpp:324, rounded once) and the Dz block's last two lines.
_AXIS_ADD = ("        Dz[idx] = Dz[idx] + __double2float_rn(axis_courant * "
             "(double)(Hy[idx]));\n")
_DZ_TAIL = ("        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;\n"
            "        pml_apply(Dz, fu_Dz, idx, curl, kms_x[i], sinv_x[i], kms_y[j], "
            "sinv_y[j], in_i_y);\n")


def _drop_bz_radial_substitution(source: str) -> Tuple[str, int]:
    """Bz's curl becomes the ordinary Cartesian four-operand one.

    THE WHOLE B-SIDE SUBSTITUTION, deleted. ``stepping.step_B`` replaces that curl with
    dtdx times the radial difference of MEEP's per-chunk ``f_rderiv_int``; a kernel that
    computed ``dEy/dr - dEx/dphi`` instead solves Cartesian Maxwell on a cylindrical grid
    -- smooth, plausible and wrong everywhere off the axis. MUST BE CAUGHT.
    """
    return _needle(_RADIAL_CURL, _CARTESIAN_BZ)(source)


def _radial_curl_commuted(source: str) -> Tuple[str, int]:
    """A NULL: ``rad[idx] * dtdx`` for ``dtdx * rad[idx]``. IEEE multiplication commutes
    bitwise; paired with the substitution legs on the same line. MUST BE UNCAUGHT."""
    return _needle(_RADIAL_CURL, "        float curl = rad[idx] * dtdx;\n")(source)


def _drop_axis_zero_bx(source: str) -> Tuple[str, int]:
    """``Bx[r=0] = 0`` deleted (``stepping._cylindrical_axis_zero_B``, m = 0).

    A radial vector at r = 0 points nowhere. MUST BE CAUGHT.
    """
    return _needle("    if (i == 0) Bx[idx] = 0.0f;\n", "")(source)


def _drop_dz_radial_substitution(source: str) -> Tuple[str, int]:
    """Dz's curl becomes the Cartesian one over the RAW Hp.

    ``stepping.step_D`` replaces that curl with dtdx times the backward radial difference
    of Hp's per-chunk sum. Without it the Dz update computes ``dHp/dr`` where cylindrical
    Maxwell needs ``(1/r)d(r*Hp)/dr``. MUST BE CAUGHT.
    """
    return _needle(_RADIAL_CURL, _CARTESIAN_DZ)(source)


def _radial_into_dx(source: str) -> Tuple[str, int]:
    """The radial difference LEAKS into Dx, whose Hy operands must stay RAW.

    ``stepping.step_D`` substitutes for the Dz TERM only; ``Dx`` reads the untouched
    ``magnetic`` mapping. The two operands differ by a whole radial integral, so this is
    a wrong answer on a component the substitution has nothing to do with. MUST BE
    CAUGHT.
    """
    return _needle("        float f2 = Hy[idx];\n"
                   "        float ss = shift_dn(Hy, idx, k, nz, sz, bc_z);\n",
                   "        float f2 = rad[idx];\n"
                   "        float ss = shift_dn(rad, idx, k, nz, sz, bc_z);\n")(source)


def _drop_axis_zero_dy(source: str) -> Tuple[str, int]:
    """``Dy[r=0] = 0`` deleted (``stepping._cylindrical_axis_zero_D``, m = 0).

    An azimuthal vector at r = 0 has no direction to point. MUST BE CAUGHT.
    """
    return _needle("        Dy[idx] = 0.0f;\n", "")(source)


def _drop_axis_add_dz(source: str) -> Tuple[str, int]:
    """The on-axis Dz post-add deleted (``stepping._cylindrical_axis_zero_D``, m = 0).

    ``4*Courant*Hp`` at the axis row is the analytic limit of ``(1/r)d(r*Hp)/dr`` there
    (step_db.cpp:299-341), and the radial difference cannot supply it because the curl is
    masked on that row. MUST BE CAUGHT.
    """
    return _needle(_AXIS_ADD, "")(source)


def _fold_axis_add_into_curl(source: str) -> Tuple[str, int]:
    """The on-axis Dz increment FOLDED INTO THE CURL instead of post-added.

    A DECLARED NULL SINCE THE PER-CHUNK FORMS (re-read under KERNELS.md section 7.1, and
    measured UNCAUGHT on 4 of 4 specs on the device, keep). It was caught while the
    split-field recurrence ran on every cell, because the folded increment reached
    ``fu_Dz`` on the axis row. Now the axis row carries no Dz layer (dsig R: the r
    layer sits at r-high only; dsigu phi: never a layer), so Dz takes the plain form
    there and ``fu_Dz`` is untouched (the auxiliary contract, C2); and the ownership
    mask has made the row's curl ``+0``. The fold is then ``f - (+0 - a)`` against
    MEEP's ``(f - +0) + a``: the same word exactly, except for the signed-zero pair
    ``f = -0``, ``a = +0`` (``Hp[0] = +0``), which this case matrix does not plant.
    MUST BE UNCAUGHT here.
    """
    folded = _DZ_TAIL.replace(
        "        pml_apply(",
        "        if (i == 0) curl = curl - __double2float_rn(axis_courant * "
        "(double)(Hy[idx]));\n        pml_apply(", 1)
    out, sites = _needle(_DZ_TAIL, folded)(source)
    if sites:
        out, _ = _drop_axis_add_dz(out)
    return out, sites


def _axis_product_in_float32(source: str) -> Tuple[str, int]:
    """The on-axis product rounded through a float32 factor: ``(float)axis_courant * Hy``.

    MEEP forms ``realnum(double(4*Courant) * double(Hp))`` and rounds once
    (``stepping._realnum_double_product``); a float32 factor is a different word wherever
    ``4*Courant`` is not a float32 word, which 0.35 is not. MUST BE CAUGHT at the inexact
    Courant, the only one the battery runs.
    """
    return _needle(_AXIS_ADD, "        Dz[idx] = Dz[idx] + (((float)axis_courant) * "
                              "Hy[idx]);\n")(source)


def _axis_product_commuted(source: str) -> Tuple[str, int]:
    """A NULL: the double product's two factors commuted. Double multiplication commutes
    bitwise; paired with :func:`_axis_product_in_float32`. MUST BE UNCAUGHT."""
    return _needle("__double2float_rn(axis_courant * (double)(Hy[idx]))",
                   "__double2float_rn((double)(Hy[idx]) * axis_courant)")(source)


def _restore_r_ghost_hz(source: str) -> Tuple[str, int]:
    """A NULL: the r_to_minus_r near ghost put BACK on Dy's live shift-down operand.

    ``stepping._shift_down``'s CYL_AXIS branch images stored row 0 with a direction sign
    and the ``(-1)^m`` phase; at m = 0 the phase is +1 and Hz is a z-direction
    component, so its ghost is ``+Hz[row 0]``. The kernel serves the METALLIC zero
    instead. It is UNOBSERVABLE because ``Dy`` has r-Yee shift 0 and the ownership mask
    zeroes its curl at exactly that row. MUST BE UNCAUGHT; its paired twin
    :func:`_restore_r_ghost_hz_no_mask` drops the mask and MUST be caught.
    """
    return _needle("        float ss = shift_dn(Hz, idx, i, nx, sx, bc_x);\n",
                   "        float ss = (i > 0) ? Hz[idx - sx] : (1.0f * Hz[idx]);\n")(source)


def _restore_r_ghost_hz_no_mask(source: str) -> Tuple[str, int]:
    """The ghost restored AND Dy's r mask dropped -- the paired twin.

    A DECLARED NULL SINCE THE PER-CHUNK FORMS (re-read under KERNELS.md section 7.1, and
    measured UNCAUGHT on 4 of 4 specs on the device, keep). The ghost reaches only Dy's
    axis-row curl. While the recurrence ran on every cell it reached ``fu_Dy`` there
    and was caught; now the axis row carries no Dy dsigu (r) layer, so ``fu_Dy`` is
    untouched there (C2), and the m = 0 axis rule stores ``Dy[0] = +0`` after the
    recurrence (``stepping._cylindrical_axis_zero_D``; :func:`_drop_axis_zero_dy` is the
    caught leg on that store). MUST BE UNCAUGHT here.
    """
    out, sites = _restore_r_ghost_hz(source)
    if not sites:
        return source, 0
    needle = ("        float curl = curl_stencil(dtdx, sf, f1, f2, ss, pr_z, pr_x);\n"
              "        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;\n"
              "        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;\n")
    replacement = ("        float curl = curl_stencil(dtdx, sf, f1, f2, ss, pr_z, pr_x);\n"
                   "        if (bc_z == BC_METALLIC && k == 0) curl = 0.0f;\n")
    if needle not in out:
        return source, 0
    return out.replace(needle, replacement), sites


# --- KERNELS.md section 7.2: the B battery on the per-chunk forms -------------------
#
# The record decode is the kernel's own (bounds_block.curl_entries), so each leg edits
# the side's own lattice: the HALF-INTEGER record on B, the INTEGER one on D.

def _b_battery(side: str) -> Dict[str, Any]:
    lat = "h" if side == "step_B" else "i"
    words_z = "cb[20], cbh_z_a1 = cb[21], cbh_z_b0 = cb[22], cbh_z_b1 = cb[23]" \
        if side == "step_B" else "cb[8], cbi_z_a1 = cb[9], cbi_z_b0 = cb[10], cbi_z_b1 = cb[11]"
    swapped_z = "cb[8], cbh_z_a1 = cb[9], cbh_z_b0 = cb[10], cbh_z_b1 = cb[11]" \
        if side == "step_B" else "cb[20], cbi_z_a1 = cb[21], cbi_z_b0 = cb[22], cbi_z_b1 = cb[23]"
    first = "Bx" if side == "step_B" else "Dx"
    second = "By" if side == "step_B" else "Dy"
    membership_z = (f"const int in_{lat}_z = (k >= cb{lat}_z_a0 && k < cb{lat}_z_a1) || "
                    f"(k >= cb{lat}_z_b0 && k < cb{lat}_z_b1);")
    membership_x = (f"const int in_{lat}_x = (i >= cb{lat}_x_a0 && i < cb{lat}_x_a1) || "
                    f"(i >= cb{lat}_x_b0 && i < cb{lat}_x_b1);")
    return {
        # The select removed on the z and r memberships: the recurrence on every cell,
        # the 0.9.2 form. Off the run the dsig-only form is a different word.
        "recurrence_on_every_cell_z": (
            _needle(membership_z, f"const int in_{lat}_z = 1;"), True),
        "recurrence_on_every_cell_r": (
            _needle(membership_x, f"const int in_{lat}_x = 1;"), True),
        # The form chosen by coefficient identity instead of chunk membership.
        "predicate_on_coefficient_identity": (
            _needle(f"pml_apply({first}, fu_{first}, idx, curl, kms_y[j], sinv_y[j], "
                    f"kms_z[k], sinv_z[k], in_{lat}_z);",
                    f"pml_apply({first}, fu_{first}, idx, curl, kms_y[j], sinv_y[j], "
                    f"kms_z[k], sinv_z[k], (kms_z[k] != 1.0f));"), True),
        # The low run's end moved one row: the boundary plane changes form.
        "membership_bound_off_by_one": (
            _needle(f"const int in_{lat}_z = (k >= cb{lat}_z_a0 && k < cb{lat}_z_a1)",
                    f"const int in_{lat}_z = (k >= cb{lat}_z_a0 && k < cb{lat}_z_a1 + 1)"),
            True),
        # The other lattice's record: the boundary plane moves by one cell.
        "record_of_the_other_lattice": (
            _needle(f"const int cb{lat}_z_a0 = {words_z};",
                    f"const int cb{lat}_z_a0 = {swapped_z};"), True),
        # The dsig axis's membership handed in as the dsigu flag (dsig z, dsigu r on the
        # second component): dsig-only against dsigu-only.
        "dsig_and_dsigu_flags_swapped": (
            _needle(f"pml_apply({second}, fu_{second}, idx, curl, kms_z[k], sinv_z[k], "
                    f"kms_x[i], sinv_x[i], in_{lat}_x);",
                    f"pml_apply({second}, fu_{second}, idx, curl, kms_z[k], sinv_z[k], "
                    f"kms_x[i], sinv_x[i], in_{lat}_z);"), True),
        # The auxiliary stored on a plain cell (the auxiliary contract, C2).
        "auxiliary_stored_on_a_plain_cell": (
            _needle("    if (dsigu) fu[idx] = fu_new;\n", "    fu[idx] = fu_new;\n"),
            True),
        # A DECLARED NULL: the auxiliary load issued on every cell, the store still
        # guarded; the loaded word feeds only the recurrence's cells.
        "auxiliary_load_unguarded": (
            _needle("    float fprev = dsigu ? fu[idx] : 0.0f;\n"
                    "    float fu_new = ((fprev * kms) - curl) * sinv;\n",
                    "    float fu_any = fu[idx];\n"
                    "    float fprev = dsigu ? fu_any : 0.0f;\n"
                    "    float fu_new = ((fprev * kms) - curl) * sinv;\n"), False),
    }


#: This family's own battery, keyed by the sub-step whose source it edits. A mutation
#: armed against the wrong side matches zero sites and is reported NOT ARMED rather
#: than scored, because a mutation silently absent from a battery and a mutation that
#: cannot bite look identical in a verdict.
CYLINDRICAL_SOURCE_MUTATIONS: Dict[str, Dict[str, Any]] = {
    "step_B": {
        "drop_bz_radial_substitution": (_drop_bz_radial_substitution, True),
        "drop_axis_zero_bx": (_drop_axis_zero_bx, True),
        **_b_battery("step_B"),
        # THE NULLS.
        "radial_curl_commuted": (_radial_curl_commuted, False),
    },
    "step_D": {
        "drop_dz_radial_substitution": (_drop_dz_radial_substitution, True),
        "radial_into_dx": (_radial_into_dx, True),
        "drop_axis_zero_dy": (_drop_axis_zero_dy, True),
        "drop_axis_add_dz": (_drop_axis_add_dz, True),
        "axis_product_in_float32": (_axis_product_in_float32, True),
        **_b_battery("step_D"),
        # THE NULLS.
        "radial_curl_commuted": (_radial_curl_commuted, False),
        "axis_product_commuted": (_axis_product_commuted, False),
        "restore_r_ghost_hz": (_restore_r_ghost_hz, False),
        "restore_r_ghost_hz_no_mask": (_restore_r_ghost_hz_no_mask, False),
        "fold_axis_add_into_curl": (_fold_axis_add_into_curl, False),
    },
}

#: Source mutations LIVE ONLY ON SOME SPECS, by construction: each maps to the spec
#: labels it must be caught on, and is a null control (must move no word) on the others.
#: ``drop_metallic_mask`` removes the r AND z wall masks at once. On the D side the r mask
#: is unobservable since the per-chunk forms (measured: uncaught on the periodic-z spec
#: only, device, keep): the axis row of Dy is stored ``+0`` by the m = 0 axis rule after
#: the recurrence, Dz's axis-row curl is ``dtdx`` times the backward difference's row 0,
#: which no segment writes (``+0``, the word the mask stores), and no auxiliary is
#: touched on that row (C2). Only the z wall mask bites, so the leg is live where z is
#: metallic.
SOURCE_MUTATION_LIVE_ON: Dict[str, Tuple[str, ...]] = {
    "step_D:drop_metallic_mask": ("r16_z20_metallic", "r9_z17_metallic",
                                  "r12_z48_metallic"),
}

#: HOST mutations: they corrupt the TABLES or the CHUNK RECORD a launch binds rather
#: than the device text, so they run on both backends. The scan's own host defects (the
#: starts line, the segment list, the weights' ir0, the wall row) are
#: gate_cuda_cylindrical_scan.py's.
HOST_MUTATIONS: Tuple[str, ...] = (
    "swap_curl_sublattice",       # half-integer vs integer PML positions
    "record_bound_off_by_one",    # the low run's end one row up: the plane changes form
    "record_lattices_swapped",    # each lattice handed the other's runs
)
B_ONLY_HOST_MUTATIONS: Tuple[str, ...] = ()
RECORD_HOST_MUTATIONS: Tuple[str, ...] = ("record_bound_off_by_one",
                                          "record_lattices_swapped")


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
    D-side axis rule READS ``Hy`` and the B-side scan reads ``Ey``, and a zero there
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
# The coefficient tables, the chunk record and the radial difference, plus the host
# mutations that corrupt them
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


def host_mutated_tables(name: Optional[str], sub_step: str, layer) -> Dict[str, Any]:
    if name == "swap_curl_sublattice":
        return wrong_sub_lattice_tables(sub_step, layer)
    if name is None or name in RECORD_HOST_MUTATIONS:
        return tables_for(sub_step, layer)
    raise KeyError(f"unknown host mutation {name!r}")


def case_record(grid, layer) -> np.ndarray:
    """The chunk record of the case's own division: the one-rank division of ``grid``
    with ``layer``, which is the division the oracle steps with (``stepping``'s region
    plan when no driver installed one)."""
    return gate_chunk_records.layer_record(grid, layer)


def host_mutated_record(name: Optional[str], record: np.ndarray, grid) -> np.ndarray:
    """Corrupt the chunk record a LAUNCH BINDS, leaving the oracle's division alone.

    KERNELS.md section 7.2: ``record_bound_off_by_one`` moves each layered axis's low
    run end one row up on both lattices (the boundary plane changes form);
    ``record_lattices_swapped`` hands each lattice the other's runs (the boundary plane
    moves by one cell). The record itself for any other name.
    """
    if name not in RECORD_HOST_MUTATIONS:
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


def radial_for(backend: str, fields, layer, sub_step: str):
    """The radial difference this launch's Bz / Dz curl reads.

    On the device it is the SHIPPED scan (``cylindrical_prefix.radial_difference``) over
    the current source, as ``cylindrical_kernels.step_cylindrical`` runs it, so a case
    certifies the scan and the curl together; the scan alone is
    ``gate_cuda_cylindrical_scan.py``'s. The NumPy backend runs that gate's transcription
    of the column-serial loop.
    """
    if backend == "cuda":
        return cyl_prefix.radial_difference(fields, layer, sub_step)
    tables = cyl_prefix.radial_scan_tables(fields, layer, sub_step)
    source = getattr(fields, cyl_prefix.PREFIX_COMPONENT[sub_step])
    return scan_gate.scan_numpy(np.asarray(to_host(source)), tables)


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------

def run_kernel_cuda(sub_step: str, fields, tables, codes, dtdx, radial, record) -> None:
    """The SHIPPED kernel, through its own launch wrapper, with the radial difference
    and the chunk record handed in."""
    from meep_gpu.cuda_kernels import cylindrical_kernels  # noqa: PLC0415

    entry = cylindrical_kernels.CYLINDRICAL_LAUNCHERS[sub_step]
    cb = cp.ascontiguousarray(cp.asarray(np.ascontiguousarray(record, dtype=np.int32)))
    entry(fields, tables, tuple(np.int32(c) for c in codes), dtdx, radial=radial,
          chunk_bounds=cb)
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


def _membership(record: np.ndarray, half: bool, axis: int, n: int) -> np.ndarray:
    """The record's ``(a0 <= r < a1) || (b0 <= r < b1)`` along one axis, shaped to
    broadcast along it."""
    a0, a1, b0, b1 = (int(record[record_index(half, axis, word)]) for word in range(4))
    rows = np.arange(n)
    inside = ((rows >= a0) & (rows < a1)) | ((rows >= b0) & (rows < b1))
    shape = [1, 1, 1]
    shape[axis] = n
    return inside.reshape(shape)


def _curl_stencil_numpy(scale, sf, f1, f2, ss, pr1: int, pr2: int) -> np.ndarray:
    """``curl_stencil``: MEEP's left-to-right bracket, or the one-partner forms."""
    if pr1 and pr2:
        return (scale * (((sf - f1) + f2) - ss)).astype(np.float32)
    if pr1:
        return (scale * (sf - f1)).astype(np.float32)
    if pr2:
        return ((-scale) * (ss - f2)).astype(np.float32)
    return np.zeros_like(f1, dtype=np.float32)


def run_kernel_numpy(sub_step: str, fields, tables, codes, dtdx, radial, record) -> None:
    """The device tree, transcribed, in float32 -- the laptop backend.

    The forms are the kernel's: ``curl_stencil``'s left-to-right bracket with the
    partner-presence forms, the radial curl ``dtdx * rad``, ``pml_apply``'s two forms
    chosen per cell by the chunk record (the recurrence where the cell's chunk carries
    the dsigu layer, the dsig-only form elsewhere, the auxiliary touched only on the
    recurrence's cells), and the m = 0 axis rules with the on-axis Dz product formed in
    double and rounded once.

    THIS COMPILES NOTHING AND CERTIFIES NOTHING. It cannot see a defect the NVRTC
    contraction guard exists for and it is not the shipped bytes. What it settles is
    whether the harness can fail.
    """
    spec = SUB_STEP_ARRAYS[sub_step]
    shift = _shift_dn_numpy if spec["backward"] else _shift_up_numpy
    half = bool(spec["half_integer"])
    scale = np.float32(dtdx)
    names = "xyz"
    shape = tuple(int(n) for n in fields.grid.shape)
    record = np.asarray(record, dtype=np.int32)
    presence = int(record[RECORD_PRESENCE_WORD])
    present = [(presence >> axis) & 1 for axis in range(3)]
    radial_host = np.asarray(to_host(radial), dtype=np.float32)
    for (target, aux, first, first_axis, second, second_axis,
         dsig, dsigu) in CURL_TERMS[sub_step]:
        f = getattr(fields, target)
        fu = getattr(fields, aux)
        if target in ("Bz", "Dz"):
            curl = (scale * radial_host).astype(np.float32)
        else:
            f1 = getattr(fields, first)
            sf = shift(f1, first_axis, codes[first_axis])
            f2 = getattr(fields, second)
            ss = shift(f2, second_axis, codes[second_axis])
            curl = _curl_stencil_numpy(scale, sf, f1, f2, ss, present[first_axis],
                                       present[second_axis])
        iyee = IYEE_SHIFTS[target]
        for axis in range(3):
            if iyee[axis] == 0 and codes[axis] == BC_METALLIC:
                curl[_plane_index(axis, 0, curl.shape)] = np.float32(0.0)
        kms = _broadcast(tables[f"kms_{names[dsig]}"], dsig)
        sinv = _broadcast(tables[f"sinv_{names[dsig]}"], dsig)
        kms_u = _broadcast(tables[f"kms_{names[dsigu]}"], dsigu)
        sinv_u = _broadcast(tables[f"sinv_{names[dsigu]}"], dsigu)
        inside = np.broadcast_to(_membership(record, half, dsigu, shape[dsigu]), shape)
        fprev = np.where(inside, fu, np.float32(0.0)).astype(np.float32)
        fu_new = (((fprev * kms).astype(np.float32) - curl).astype(np.float32)
                  * sinv).astype(np.float32)
        recurrence = ((((f * kms_u).astype(np.float32) + fu_new).astype(np.float32)
                       - fprev).astype(np.float32) * sinv_u).astype(np.float32)
        plain = ((((f * kms).astype(np.float32) - curl).astype(np.float32))
                 * sinv).astype(np.float32)
        fu[...] = np.where(inside, fu_new, fu)
        f[...] = np.where(inside, recurrence, plain)
    axis_row = _plane_index(0, 0, shape)
    if sub_step == "step_B":
        fields.Bx[axis_row] = np.float32(0.0)
    else:
        factor = np.float64(4.0 * float(fields.grid.courant))
        product = (factor * fields.Hy[axis_row].astype(np.float64)).astype(np.float32)
        fields.Dz[axis_row] = (fields.Dz[axis_row] + product).astype(np.float32)
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


def radial_is_live(radial, sub_step: str) -> Dict[str, Any]:
    """Is the radial difference non-zero somewhere, and non-constant down r?

    A difference that is ``+0`` everywhere makes every radial-substitution defect
    invisible: the substitution and its absence would both feed the recurrence nothing.
    """
    host = np.ascontiguousarray(to_host(radial), dtype=np.float32)
    target = "Bz" if sub_step == "step_B" else "Dz"
    nonzero = int(np.count_nonzero(host.ravel().view(np.uint32)))
    varies = int(np.count_nonzero(host[1:] - host[:-1])) if host.shape[0] > 1 else 0
    return {
        "rows": int(host.shape[0]),
        "nonzero_words": nonzero,
        "rows_that_differ_from_the_one_below": varies,
        "max_abs": float(np.max(np.abs(host))) if host.size else 0.0,
        "radial_target": target,
        "meets_floor": nonzero > 0 and varies > 0,
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
# The scan-order leg: the shipped scan against the array path's difference
# ---------------------------------------------------------------------------

def scan_order_leg(shapes: Sequence[Tuple[int, int, int]]) -> Dict[str, Any]:
    """Does the SHIPPED column-serial scan reproduce the array path's radial difference?

    The cases below read the scan's output, so a case that diverged could be the scan's
    or the curl's; this leg answers the first on this gate's own shapes, through the scan
    gate's own comparison (``gate_cuda_cylindrical_scan.identity_leg``: real storage, the
    one-rank division of the case's layer, both sides). The scan's full battery --
    complex storage, many-chunk divisions, its mutations and both policies -- is that
    gate's.
    """
    out = scan_gate.identity_leg(shapes, backend="cuda")
    out["cupy_version"] = getattr(cp, "__version__", None)
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

    # Leg 2: the kernel, from the SAME frozen state, with the radial difference
    # scanned from the SAME sources the oracle differenced and the chunk record of the
    # division the oracle stepped with.
    restore(fields, frozen)
    tables = host_mutated_tables(host_mutation, sub_step, layer)
    record = host_mutated_record(host_mutation, case_record(grid, layer), grid)
    case["chunk_record"] = [int(word) for word in record]
    radial = radial_for(backend, fields, layer, sub_step)
    liveness = radial_is_live(radial, sub_step)
    case["radial_is_live"] = liveness
    if not liveness["meets_floor"]:
        case["skipped"] = ("the radial difference is +0 everywhere or constant down r; "
                           "every radial defect is invisible on this case")
        case["seconds"] = time.time() - started
        return case

    runner = run_kernel_cuda if backend == "cuda" else run_kernel_numpy
    runner(sub_step, fields, tables, codes, dtdx, radial, record)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(sub_step)}
    case["single_launch"] = combine(parts)
    case["per_array"] = {
        name: {"differing_words": int(parts[name]["differing_floats"]),
               "words": int(parts[name]["total_floats"])}
        for name in outputs(sub_step)}

    # Leg 3: the multi-step. ``fu`` IS STATE. ONE ``Fields`` object, run twice from
    # the SAME frozen state, with the radial difference RESCANNED every launch on both
    # paths -- which is what a real driver does and what a cached one would get wrong.
    if host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            (stepping.step_B if sub_step == "step_B" else stepping.step_D)(fields, layer)
            advance_sources(fields, sub_step)
        oracle_multi = snapshot(fields)

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(sub_step, fields, tables, codes, dtdx,
                   radial_for(backend, fields, layer, sub_step), record)
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
    """Every table- and record-level defect. A gate that cannot fail certifies nothing."""
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
            yield name, SHARED_NEEDLES[name], name not in SHARED_NULL_MUTATIONS
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
                if key in SOURCE_MUTATION_LIVE_ON:
                    live = SOURCE_MUTATION_LIVE_ON[key]
                    on = _score([c for c in legs if c.get("label") in live], True)
                    off = _score([c for c in legs if c.get("label") not in live], False)
                    scored = dict(on, live_on=list(live), null_control=off)
                    if off["ran"] and off["verdict"] != "NULL CONFIRMED":
                        scored["verdict"] = "NULL VIOLATED"
                else:
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
    scan = results.get("scan_order")
    if results.get("backend") == "cuda" and not (scan or {}).get("all_identical", False):
        reasons.append("the shipped radial scan is not word-identical to the array path's "
                       "difference on this gate's shapes (leg scan_order), so a case here "
                       "cannot say whether the curl or the scan diverged")
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
            "no throughput claim: nothing here was timed",
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
        results["scan_order"] = scan_order_leg(tuple(spec["shape"] for spec in CASES))
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
