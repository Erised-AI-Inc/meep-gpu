"""Sub-step byte-identity gate for the INSTANTANEOUS chi2/chi3 constitutive family.

THE CENSUS REFUSAL THIS ANSWERS, quoted::

    instantaneous chi2/chi3: a Pade factor replaces the constitutive product

``coverage.covers_real_pml_constitutive`` states it at coverage.py:1056-1057 and
fires it on BOTH sides for the two corpus rows that carry a nonlinearity --
``3rd-harm-1d.py`` and ``Test3rdHarm1d.test_3rd_harm_1d``. Four slots. The
refusal is two claims and this gate measures them separately, because they are
not the same claim.

=============================================================================
ARM E -- A NEW KERNEL, MEASURED AGAINST THE ARRAY PATH
=============================================================================

``meep_gpu/cuda_kernels/nonlinear_constitutive.update_E_pml_real_nonlinear``
is written for this family and has never run. The refusal is TRUE on this side:
``stepping.update_E`` branches at :969-971 into ``_nonlinear_constitutive``
(:1051), whose product is ``(gs*us) * calc_nonlinear_u(...)`` (:1078-1080) with a
four-point transverse average of the OTHER two D volumes inside it. What is being
measured is whether the transcription reproduces that arithmetic word for word.

=============================================================================
ARM H -- A NULL, AND THE PREMISE IS ARMED SO IT IS NOT A BLIND SPOT
=============================================================================

``stepping.update_H`` (stepping.py:907-923) is six lines and none of them reads
the nonlinearity; ``test_nonlinear_constitutive.test_update_H_names_nothing_nonlinear``
parses the function and pins that. So the refusal is INHERITED on this side and
the certified ``constitutive_kernels.update_H_pml_real`` should already be
byte-exact on a grid carrying chi.

"Should already be" is a hypothesis. This gate runs the SHIPPED H kernel, with no
change to one byte of it, on grids whose chi is live and whose E sub-step the same
fixture measures -- and arms the refusal's own premise as a defect:
``pade_scale_the_H_source`` rewrites the H device string so that it DOES apply a
Pade factor. It must be CAUGHT. Without it, "chi does not reach update_H" would be
a claim resting on a leg nobody showed could fail.

=============================================================================
WHAT WOULD MAKE THIS GATE VACUOUS, AND THE FLOORS THAT REFUSE IT
=============================================================================

* ``oracle_moved`` -- the fraction of output words the array path changed from the
  frozen input. Zero-init is a fixed point of this recurrence; a case that moved
  nothing is refused, not passed.
* ``pade_departure`` -- ``max|u - 1|`` over the live components. THE FLOOR THIS
  FAMILY EXISTS FOR. If the Pade factor is the identity everywhere, every one of
  the nine nonlinear source mutations is invisible and the case measures the
  certified linear kernel under a new name. Hard floor on the two ``uniform``
  amplitude classes; recorded on the others, where a degenerate ``u`` is the
  MEASUREMENT rather than a defect (see the subnormal classes below).
* ``expansion`` -- ``max(|c2| + |c3|)`` from ``stepping.nonlinear_margin``
  (:1315-1357). Must be strictly inside ``(0, 1/3)``: zero means no live
  nonlinearity, and past 1/3 the run has left the domain
  ``driver._NonlinearityGuard`` admits, so the case would be certifying arithmetic
  the engine refuses to perform.
* ``absorber_departure`` -- ``max|kps-1|``, ``max|kms-1|``. An identity profile
  cannot distinguish a coefficient-index error.
* ``epsilon_departure`` -- inverse epsilon away from 1.0 and the three volumes
  distinct, or ``drop_inverse_epsilon`` and ``bind_Ez_inv_eps_for_all_three`` are
  bit-identical for a reason about the fixture.
* ``transverse_departure`` -- ``Dsqr`` must differ from ``gs*gs`` somewhere, or
  the whole four-corner stencil is untested and four of the mutations are inert.
* The mutation battery, INCLUDING FOUR THAT MUST BE UNCAUGHT. A battery of
  only-must-be-caught legs scores identically whether the comparator works or has
  degenerated into failing everything.

=============================================================================
THE FOUR CLASSES, AND WHY THE DIVISION NEEDS ITS OWN
=============================================================================

``constitutive_kernels.py``'s header records the one PTX exception this package
knows of: ptxas expands ``div.rn.f32`` into a Newton-Raphson sequence whose range
checks carry ``.FTZ`` in SASS regardless of the PTX modifier, so a PTX audit does
not fully enforce a subnormal *keep* policy where a division appears. It ends "if
a future constitutive variant ever divides ... that exception applies to it".
THIS IS THAT VARIANT, and these classes are how the exception is measured rather
than argued:

* ``uniform / moderate`` -- the workhorse. ``u`` departs from 1 by a few percent.
* ``uniform / near_pole`` -- chi scaled so the expansion approaches (and stays
  under) 1/3. ``u`` is far from 1, which is what makes ANY divergence in the
  quotient byte-visible in float32 rather than lost in the rounding of a number
  that is 1.0 to seven digits.
* ``subnormal_band / moderate`` -- subnormals, signed zeros and the needle values
  in EVERY operand array. Here ``c2`` and ``c3`` underflow and ``u`` becomes
  exactly 1.0 on both paths: this class does NOT test the Pade factor, it tests
  the tail and the transverse loads under the policy, and the record says so.
* ``uniform / subnormal_chi`` -- normal fields, chi chosen so that ``c2`` and
  ``c3`` themselves land IN the subnormal band. This is the class that puts a
  subnormal into the quotient's addends, which is the closest anything in this
  arithmetic gets to the divide. ``pade_operand_census`` records how many of
  ``num``, ``den`` and the quotient are subnormal (expected: zero, because both
  operands are anchored at 1.0) and ``chi_term_census`` records how many ``c2`` /
  ``c3`` words are (must be non-zero under ``keep``, or the class is vacuous).

=============================================================================
THE DIVISION SPELLING PROBE
=============================================================================

The kernel divides with ``__fdiv_rn``. The sibling Triton track MEASURED that the
plain operator was not IEEE division on THAT platform (``div.full.f32``, ~2 ulp;
jobs 2336/2338), and CUDA's default is ``-prec-div=true``, so the two spellings
are expected to agree HERE. Expected is not measured: ``plain_division`` rewrites
``__fdiv_rn(num, den)`` to ``num / den`` and the result is REPORTED WITH A READING
rather than gated on, exactly as the contraction guard's control is -- because it
is a claim about the compiler, not about this kernel, and the kernel is correct
under either answer.

=============================================================================
RUNNING IT
=============================================================================

Device (the GPU host, ONE verified-empty GPU; the cache dir MUST carry the policy
token because CuPy's disk-cache key is computed above the strip seam)::

    export TRITON_LIBCUDA_PATH="$HOME/triton_libcuda_stub"
    export LD_LIBRARY_PATH="$TRITON_LIBCUDA_PATH:$LD_LIBRARY_PATH"
    CUDA_VISIBLE_DEVICES=3 CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
        python -u gate_cuda_nonlinear.py --subnormal-policy keep \\
        --out $OUT/keep/gate.json

Laptop (no CUDA). The NumPy backend runs the SAME oracle, the same fixture, the
same floors and the same HOST mutations against a transcription of the shipped
device tree. IT COMPILES NOTHING AND CERTIFIES NOTHING::

    python -u gate_cuda_nonlinear.py --backend numpy --out /tmp/nl_local.json

One flushed line per case (the progress-reporting rule); the artifact is rewritten
atomically after every case, so an interrupted run keeps everything up to the
failure. Correctness only -- no throughput claim is made or possible.
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
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.cuda_kernels import coverage  # noqa: E402
from meep_gpu.cuda_kernels import nonlinear_constitutive as nlc  # noqa: E402

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

#: How the multi-step leg keeps the recurrence live. Held fixed, ``f_w`` settles
#: and a 60-step leg becomes a slow single-step leg. The same exact float32 scale
#: is applied on both paths, so it cancels out of the comparison -- AND it keeps
#: the amplitude falling, which keeps the run inside the pole guard's domain for
#: all 60 launches rather than walking out of it.
_ADVANCE = np.float32(0.97)

BC_PERIODIC, BC_METALLIC = 0, 1

SIDE_ARRAYS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "H": {"targets": ("Hx", "Hy", "Hz"),
          "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
          "sources": ("Bx", "By", "Bz")},
    "E": {"targets": ("Ex", "Ey", "Ez"),
          "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
          "sources": ("Dx", "Dy", "Dz")},
}

SIDES: Tuple[str, ...] = ("H", "E")


def outputs(side: str) -> Tuple[str, ...]:
    """The six arrays this sub-step writes.

    ``f_w`` IS STATE. A tree that gets the field right and ``f_w`` wrong is
    correct for exactly one launch and wrong forever after, so the auxiliary is
    compared on every case and not only in the multi-step leg. The sibling track
    measured a dropped auxiliary store at 120/120 UNCAUGHT when ``fu`` was not
    compared.
    """
    spec = SIDE_ARRAYS[side]
    return tuple(spec["targets"]) + tuple(spec["aux"])


def state_names(side: str) -> Tuple[str, ...]:
    """Everything the case has to freeze, seed and restore.

    ON THE E SIDE THAT IS ALL THREE D VOLUMES, and it is the one structural
    difference between this family and the element-wise pair: the transverse
    average reads the OTHER two components at NEIGHBOURING cells, so a fixture
    that seeded only the component's own source would leave two thirds of the
    stencil reading zeros and ``transverse_departure`` would fail its floor.
    ``E_CONSTITUTIVE_TERMS`` already names all three as this side's sources, so
    the general spelling is also the correct one.
    """
    return outputs(side) + tuple(SIDE_ARRAYS[side]["sources"])


# ---------------------------------------------------------------------------
# The case product
# ---------------------------------------------------------------------------
#
# GHOST RULES ARE THE AXIS THIS FAMILY ADDS. The certified constitutive pair
# reads no neighbour, so its sweep did not have to care which face a cell sat on;
# this one reads four corners of two partner volumes, so every case below names a
# boundary triple and the sweep carries all-periodic, all-metallic, and both
# mixed orders -- plus the corpus's own 1-D shape, where two axes are a single
# cell and the wrap returns the cell itself.

SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "all_periodic", "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (8.0, 10.0, 12.0)},
    {"label": "all_metallic", "boundaries": ("metallic", "metallic", "metallic"),
     "cell": (8.0, 10.0, 12.0)},
    {"label": "mixed_pmp", "boundaries": ("periodic", "metallic", "periodic"),
     "cell": (8.0, 10.0, 12.0)},
    {"label": "mixed_mpm", "boundaries": ("metallic", "periodic", "metallic"),
     "cell": (10.0, 8.0, 12.0)},
    # THE CORPUS SHAPE. 3rd-harm-1d.py lifts to (1, 1, nz): two invariant axes
    # whose periodic wrap returns the same cell, which is what MEEP's stride(d)=0
    # computes for a direction it does not have. The absorber lives on z alone.
    {"label": "one_dimensional", "boundaries": ("periodic", "periodic", "periodic"),
     "cell": (1.0, 1.0, 40.0)},
    # A PARTLY NONLINEAR run: one component takes MEEP's ``else if (u)`` branch
    # (stepping.py:1100-1102) and must compile to the certified plain body, while
    # its D volume is still read as a NEIGHBOUR by the other two.
    {"label": "partly_nonlinear", "boundaries": ("periodic", "metallic", "periodic"),
     "cell": (8.0, 10.0, 12.0), "linear_components": ("Ey",)},
    # A VOLUME chi, spatially varying. A uniform chi cannot distinguish a chi
    # lookup that reads the wrong cell from one that reads the right one.
    {"label": "volume_chi", "boundaries": ("metallic", "periodic", "periodic"),
     "cell": (8.0, 10.0, 12.0), "chi_is_volume": True},
)

#: 0.5 is exactly representable in float32 and 0.35 is not. Only the second can
#: distinguish a contracted expression from an uncontracted one, so the guard
#: control is scored at the inexact one. The courant also moves dt, which moves
#: every PML coefficient, so it is a real second draw of the tables.
COURANTS: Tuple[float, ...] = (0.5, 0.35)
INEXACT_COURANT = 0.35

#: ``(value_class, amplitude_class)``. See the header for what each measures.
CLASSES: Tuple[Tuple[str, str], ...] = (
    ("uniform", "moderate"),
    ("uniform", "near_pole"),
    ("subnormal_band", "moderate"),
    ("uniform", "subnormal_chi"),
)

#: The ``(value_class, amplitude)`` pairs on which ``pade_departure`` is a HARD
#: floor. On the other two a degenerate ``u`` is THE MEASUREMENT rather than a
#: defect, and skipping them would throw away exactly the two classes that exist
#: to probe the policy: ``subnormal_band`` drives every operand under the
#: normal-magnitude floor so ``c2``/``c3`` underflow and ``u`` becomes exactly
#: 1.0 -- which leaves the TAIL and the transverse LOADS still under test -- and
#: ``subnormal_chi`` does the same to the chi terms alone, which is the class that
#: answers the ``div.rn`` FTZ question with a census.
PADE_FLOOR_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("uniform", "moderate"), ("uniform", "near_pole"))

#: chi magnitudes per amplitude class, chosen against the pole guard's derived
#: bound ``|c2| + |c3| < 1/3`` (stepping.py:1356-1359) and MEASURED per case --
#: ``expansion`` is recorded and floored, never assumed from these numbers.
CHI_BY_AMPLITUDE: Dict[str, Tuple[float, float]] = {
    "moderate": (0.02, 0.05),
    # Large enough that ``u`` is visibly off 1 and small enough to stay inside
    # the guard. The measured expansion is the release-relevant number.
    "near_pole": (0.10, 0.16),
    # c2 = gs*chi2*us^2 and c3 = dsqr*chi3*us^3 land IN the subnormal band for
    # normal-magnitude fields: float32's smallest normal is 1.18e-38.
    "subnormal_chi": (3e-40, 3e-40),
}

GUARD_SETS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("fmad_false", ("--fmad=false",)),
    ("default_no_options", ()),
)


# ---------------------------------------------------------------------------
# The source mutations
# ---------------------------------------------------------------------------
#
# The four that key on ``constitutive_apply`` are the SHARED ones from
# ``probe_fused_kernel_bit_identity`` -- this kernel copies that tail character
# for character precisely so they arm here too, and the slice test pins the copy.
# Everything below is this family's own, and every one is a defect a reader could
# plausibly write.

def _four_point_left_to_right(source: str) -> Tuple[str, int]:
    """MEEP C's summation order in place of stepping's shifted-pair association.

    ``stepping._nonlinear_transverse_sums`` (:1155-1163) forms the near pair, then
    that pair shifted; MEEP C sums left to right (step_generic.cpp:646-648). The
    BYTE ARBITER is stepping.py, so this is a defect here even though it is MEEP's
    own spelling. MUST BE CAUGHT -- and if it is not, the two associations agree
    on this fixture and the transcription note is decorative.
    """
    needle = ("    float near_pair = g[o_c] + (v_d ? g[o_d] : 0.0f);\n"
              "    float far_pair = (v_u ? g[o_u] : 0.0f) + (v_ud ? g[o_ud] : 0.0f);\n"
              "    return near_pair + far_pair;\n")
    replacement = ("    float near_pair = g[o_c] + (v_u ? g[o_u] : 0.0f);\n"
                   "    float far_pair = (v_d ? g[o_d] : 0.0f);\n"
                   "    return (near_pair + far_pair) + (v_ud ? g[o_ud] : 0.0f);\n")
    return source.replace(needle, replacement), source.count(needle)


#: The argument block of one ``four_point_sum`` call: everything between the
#: partner volume and the closing paren. Rewritten by the two stencil mutations,
#: and ONLY there, so nothing in the index-declaration block is touched and every
#: index stays clamped in bounds.
_CALL_SITE = re.compile(r"(four_point_sum\(\s*\n\s*D[xyz], idx,\n)(.*?)(\);)",
                        re.DOTALL)


def _dsqr_same_direction_shifts(source: str) -> Tuple[str, int]:
    """THE HALF-CELL REGISTRATION ERROR: both shifts the same way round.

    The OWN axis is shifted DOWN instead of up, so the four corners become a 2x2
    block on the wrong side of the component's Yee position. stepping.py:1160-1171
    states the geometry that forces the two shifts to be opposite: for Ez at
    ``(x_i, y_j, z_k+1/2)`` the partner Dx sits at ``(x_i+1/2, y_j, z_k)``, half a
    cell DOWN in x and half a cell UP in z. This survives every scalar test and
    shows up only in the mixed-polarization terms. MUST BE CAUGHT.

    Spelled as a rewrite of the CALL SITES' argument text -- the up-neighbour
    index and its validity flag swapped for the down pair -- because both are
    already computed and clamped in bounds, so the leg measures a WRONG ANSWER
    and never a memory fault.
    """
    swaps = (("ui", "di"), ("uj", "dj"), ("uk", "dk"),
             ("uvx", "dvx"), ("uvy", "dvy"), ("uvz", "dvz"))

    def rewrite(match):
        body = match.group(2)
        for before, after in swaps:
            # WHOLE TOKENS. An earlier spelling matched "uk *" and so silently
            # skipped the component whose own axis is the LAST one -- the index
            # there is written "+ uk" with no trailing "*". MEASURED: that left
            # the mutation a complete no-op on the (1, 1, nz) corpus shape, where
            # the other two components' own axes have extent 1 and their up- and
            # down-neighbours coincide, and the leg came back UNCAUGHT for a
            # reason about the rewrite rather than about the kernel.
            body = re.sub(rf"\b{before}\b", after, body)
        return match.group(1) + body + match.group(3)

    return _CALL_SITE.sub(rewrite, source), len(_CALL_SITE.findall(source))


def _dsqr_drop_transverse(source: str) -> Tuple[str, int]:
    """``Dsqr = gs*gs`` alone: the four-point average dropped entirely.

    ``0.0625 = (1/4)^2`` is what turns each unnormalized four-point sum back into
    a mean before it is squared (stepping.py:1131-1147); without the term ``Dsqr``
    is the component's own square and the interpolation onto its Yee position is
    gone. MUST BE CAUGHT.
    """
    needle = " + 0.0625f * (g1s * g1s + g2s * g2s)"
    return source.replace(needle, ""), source.count(needle)


def _drop_pade_factor(source: str) -> Tuple[str, int]:
    """THE REFUSAL'S PREMISE, ARMED ON THE E SIDE: no Pade factor at all.

    "a Pade factor REPLACES the constitutive product" is why this family is a
    kernel rather than a widening. If removing the factor leaves the bytes
    unchanged, the fixture has no live nonlinearity and every other nonlinear leg
    on that case is measuring the certified linear kernel. MUST BE CAUGHT.
    """
    pattern = r"src_([xyz]) = src_\1 \* pade_u\([^;]*\);"
    return re.subn(pattern, "", source)[0], len(re.findall(pattern, source))


def _regroup_pade_numerator(source: str) -> Tuple[str, int]:
    """``1 + (c2 + 2*c3)`` for ``(1 + c2) + 2*c3``. Float addition is not associative."""
    needle = "    float num = (1.0f + c2) + 2.0f * c3;\n"
    replacement = "    float num = 1.0f + (c2 + 2.0f * c3);\n"
    return source.replace(needle, replacement), source.count(needle)


def _swap_epsilon_powers(source: str) -> Tuple[str, int]:
    """c2 takes chi1inv^3 and c3 takes chi1inv^2 -- the powers exchanged.

    ``calc_nonlinear_u`` (stepping.py:1054-1055) pairs the LINEAR term with the
    square and the CUBIC term with the cube; MEEP's own comment derives it from
    inverting ``D = eps*E + chi2*E^2 + chi3*|E|^2 E``. Exchanged, the factor is
    smooth, finite and wrong. MUST BE CAUGHT.
    """
    needle = ("    float c2 = (gs * chi2) * us_sq;\n"
              "    float c3 = (dsqr * chi3) * us_cu;\n")
    replacement = ("    float c2 = (gs * chi2) * us_cu;\n"
                   "    float c3 = (dsqr * chi3) * us_sq;\n")
    return source.replace(needle, replacement), source.count(needle)


def _scale_row_before_product(source: str) -> Tuple[str, int]:
    """``gs * (us * u)`` for ``(gs * us) * u``: the row scaled before it is formed.

    stepping.py:1107-1109 forms the row and THEN scales it. MUST BE CAUGHT.
    """
    pattern = r"src_([xyz]) = src_\1 \* (pade_u\([^;]*\));"
    replacement = r"src_\1 = gs_\1 * (us_\1 * \2);"
    return re.subn(pattern, replacement, source)[0], len(re.findall(pattern, source))


def _drop_metallic_transverse_ghost(source: str) -> Tuple[str, int]:
    """A metallic face wraps instead of taking the zero ghost.

    ``_shift_up`` / ``_shift_down`` write an exact zero past a perfect conductor
    (stepping.py:1830, :1857). Wrapping instead reads the far wall's field into
    the near wall's average -- the classic ghost-rule confusion, and one that is
    completely invisible on an all-periodic fixture. MUST BE CAUGHT on the
    metallic specs; the sweep is what makes sure such a spec is scored.
    """
    out, count = source, 0
    for axis in "xyz":
        needle = f"    if (bc_{axis} == BC_METALLIC) {{"
        count += out.count(needle)
        out = out.replace(needle, "    if (false) {")
    return out, count


def _drop_inverse_epsilon_nonlinear(source: str) -> Tuple[str, int]:
    """The source becomes D instead of ``D * inv_eps``.

    Invisible against a table of ones, which is why the fixture draws inverse
    epsilon away from 1.0. MUST BE CAUGHT.
    """
    pattern = r"float src_([xyz]) = gs_\1 \* us_\1;"
    return re.subn(pattern, r"float src_\1 = gs_\1;", source)[0], \
        len(re.findall(pattern, source))


def _bind_Ez_inv_eps_for_all_three_nonlinear(source: str) -> Tuple[str, int]:
    """DEFECT 2 OF THE COMPLEX TEMPLATE, armed: one epsilon volume for all three.

    ``update_E_pml_complex``'s wrapper passes ``fields.inv_eps``, the Ez
    view (fields.py:1259-1260). Exactly bit-identical under an isotropic epsilon,
    so its whole power comes from the fixture drawing three INDEPENDENT volumes.
    MUST BE CAUGHT.
    """
    out, count = source, 0
    for axis in "xy":
        needle = f"float us_{axis} = inv_eps_E{axis}[idx];"
        replacement = f"float us_{axis} = inv_eps_Ez[idx];"
        count += out.count(needle)
        out = out.replace(needle, replacement)
    return out, count


def _swap_chi2_chi3_operands(source: str) -> Tuple[str, int]:
    """The two susceptibilities exchanged at the call site. MUST BE CAUGHT."""
    pattern = r"pade_u\((gs_[xyz]), (dsqr), q2, q3, (us_sq), (us_cu)\)"
    return re.subn(pattern, r"pade_u(\1, \2, q3, q2, \3, \4)", source)[0], \
        len(re.findall(pattern, source))


def _inv_eps_left_nonlinear(source: str) -> Tuple[str, int]:
    """A NULL: ``inv_eps * D`` for ``D * inv_eps``. IEEE multiply commutes.

    stepping.py:1011 writes D on the left and this kernel keeps that order for
    transcription discipline, not because it changes a bit. Its discriminating
    sibling is :func:`_drop_inverse_epsilon_nonlinear`, which edits the same
    expression and MUST be caught. MUST BE UNCAUGHT.
    """
    pattern = r"float src_([xyz]) = gs_\1 \* us_\1;"
    return re.subn(pattern, r"float src_\1 = us_\1 * gs_\1;", source)[0], \
        len(re.findall(pattern, source))


def _commute_pade_scale(source: str) -> Tuple[str, int]:
    """A NULL: ``u * (gs*us)`` for ``(gs*us) * u``. MUST BE UNCAUGHT.

    Paired with :func:`_scale_row_before_product`, which regroups the SAME
    expression and must be caught -- so the pair says the comparator is sensitive
    to the association and insensitive to the operand order, which is exactly what
    IEEE-754 says.
    """
    pattern = r"src_([xyz]) = src_\1 \* (pade_u\([^;]*\));"
    return re.subn(pattern, r"src_\1 = \2 * src_\1;", source)[0], \
        len(re.findall(pattern, source))


def _commute_dsqr_transverse_sum(source: str) -> Tuple[str, int]:
    """A NULL: ``g2s*g2s + g1s*g1s``. Float addition commutes bitwise.

    This is the swapped-partner defect in the only form that reaches the bits: the
    partner ORDER itself is pinned as TEXT by the slice test, because ``Dsqr``'s
    sum forgives it exactly. MUST BE UNCAUGHT.
    """
    needle = "0.0625f * (g1s * g1s + g2s * g2s)"
    replacement = "0.0625f * (g2s * g2s + g1s * g1s)"
    return source.replace(needle, replacement), source.count(needle)


def _plain_division(source: str) -> Tuple[str, int]:
    """THE SPELLING PROBE, not a release clause. ``num / den`` for ``__fdiv_rn``.

    The sibling Triton track measured the plain operator to be ``div.full.f32``
    (~2 ulp) on that platform. CUDA's default is ``-prec-div=true``, so these are
    expected to agree here -- and the reading is reported rather than gated on,
    because it is a claim about the compiler and this kernel is correct either way.
    """
    needle = "    return __fdiv_rn(num, den);\n"
    replacement = "    return num / den;\n"
    return source.replace(needle, replacement), source.count(needle)


def _pade_scale_the_H_source(source: str) -> Tuple[str, int]:
    """ARM H's REFUSAL, ARMED: make the certified H kernel apply a Pade factor.

    The census refuses ``update_H`` for the reason "a Pade factor replaces the
    constitutive product". ``stepping.update_H`` contains no such factor, so the
    refusal is inherited -- but "the comparator would notice if it did" is a claim
    that has to be shown. This rewrites the H device string so the source really is
    scaled by a quotient of the same shape, and it MUST BE CAUGHT. Without it the
    H arm's 100% pass rate would be consistent with a comparator that sees nothing.

    Applied to ``constitutive_kernels._update_H_pml_real_kernel_code``, which
    is restored immediately afterwards.
    """
    # THE SINGLE IS HOISTED (register view): its source is the preload
    # ``float src_a = Ba[idx];``, so that is where the Pade factor goes -- the same
    # word, scaled by the same quotient, before any statement reads it.
    pattern = r"float src_([xyz]) = (B[xyz])\[idx\];"
    replacement = (r"float src_\1 = \2[idx] * __fdiv_rn(1.0f + 0.05f * \2[idx], "
                   r"1.0f + 0.10f * \2[idx]);")
    return re.subn(pattern, replacement, source)[0], len(re.findall(pattern, source))


#: This family's own device-text defects. ``True`` means MUST BE CAUGHT.
E_SOURCE_MUTATIONS: Dict[str, Tuple[Any, bool]] = {
    "four_point_left_to_right": (_four_point_left_to_right, True),
    "dsqr_same_direction_shifts": (_dsqr_same_direction_shifts, True),
    "dsqr_drop_transverse": (_dsqr_drop_transverse, True),
    "drop_pade_factor": (_drop_pade_factor, True),
    "regroup_pade_numerator": (_regroup_pade_numerator, True),
    "swap_epsilon_powers": (_swap_epsilon_powers, True),
    "scale_row_before_product": (_scale_row_before_product, True),
    "swap_chi2_chi3_operands": (_swap_chi2_chi3_operands, True),
    "drop_metallic_transverse_ghost": (_drop_metallic_transverse_ghost, True),
    "drop_inverse_epsilon": (_drop_inverse_epsilon_nonlinear, True),
    "bind_Ez_inv_eps_for_all_three": (_bind_Ez_inv_eps_for_all_three_nonlinear, True),
    "inv_eps_left": (_inv_eps_left_nonlinear, False),
    "commute_pade_scale": (_commute_pade_scale, False),
    "commute_dsqr_transverse_sum": (_commute_dsqr_transverse_sum, False),
}

#: Borrowed from the shared probe so this gate cannot own a second spelling of a
#: defect the certified record was cut against. They key on ``constitutive_apply``,
#: which this kernel copies character for character.
SHARED_E_SOURCE_MUTATIONS: Dict[str, bool] = {
    "regroup_constitutive": True,
    "drop_fw_store": True,
    "store_fw_before_reading_prev": True,
    "own_axis_to_x_for_all_three": True,
    "fortran_order_index_decomposition": True,
    "commute_constitutive_scale": False,
}

#: Arm H's battery, applied to the SHIPPED H device string and restored after.
H_SOURCE_MUTATIONS: Dict[str, bool] = {
    "pade_scale_the_H_source": True,
    "regroup_constitutive": True,
    "drop_fw_store": True,
    "own_axis_to_x_for_all_three": True,
    "commute_constitutive_scale": False,
}

#: MUTATIONS THAT ARE PROVABLY INERT ON SOME SPECS, with the reason and the specs
#: they ARE scored on. A leg on which a defect cannot change a byte scores
#: UNCAUGHT and drags the verdict to PARTIAL for a reason about the fixture, which
#: is a harness defect that reads exactly like a kernel defect -- the folded gate
#: hit the mirror image of this and its note is why these are written out rather
#: than discovered on the device hours later.
SOURCE_MUTATION_SPECS: Dict[str, Dict[str, Any]] = {
    "drop_metallic_transverse_ghost": {
        "specs": ("all_metallic", "mixed_pmp", "mixed_mpm", "partly_nonlinear",
                  "volume_chi"),
        "why": ("the METALLIC branch is never taken on an all-periodic grid, so "
                "deleting it cannot change a byte there"),
    },
    "fortran_order_index_decomposition": {
        "specs": ("all_periodic", "all_metallic", "mixed_pmp", "mixed_mpm",
                  "partly_nonlinear", "volume_chi"),
        "why": ("on the (1, 1, nz) corpus shape the C-order and Fortran-order "
                "decompositions agree exactly -- i = j = 0 and k = idx either "
                "way -- so the rewrite is arithmetically the identity there"),
    },
}


def specs_for_source_mutation(name: str, plan):
    """The legs one device-text defect is scored on, with the inert ones dropped."""
    entry = SOURCE_MUTATION_SPECS.get(name)
    if entry is None:
        return plan
    allowed = set(entry["specs"])
    return [leg for leg in plan if leg[0]["label"] in allowed]


#: HOST mutations: they corrupt the TABLES and the operand BINDINGS rather than
#: the device text, so they run on both backends.
HOST_MUTATIONS: Dict[str, bool] = {
    "swap_constitutive_sublattice": True,
    "swap_kps_kms": True,
    "linearize_the_bindings": True,
    "swap_chi_bindings": True,
}

#: Host mutations that only exist on the E side (they touch the chi operands).
E_ONLY_HOST_MUTATIONS: Tuple[str, ...] = ("linearize_the_bindings", "swap_chi_bindings")


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

class _NumpyWearingCupysName:
    """NumPy behind CuPy's ``__name__``.

    The predicate's first question is whether the backend is CuPy at all, and
    that is the one thing about the device library a laptop cannot supply.
    """

    __name__ = "cupy"

    def __getattr__(self, item):
        return getattr(np, item)


def build(xp, spec: Dict[str, Any], courant: float, amplitude: str, rng):
    """A frozen ``(fields, layer, grid)`` triple carrying a live nonlinearity.

    THE THICKNESS RULE IS THE SLICE'S OWN: skip an axis too thin to hold a layer.
    On the 1-D spec that leaves the absorber on z alone, which is what the corpus
    rows carry and is also what makes ``absorber_departure`` a per-axis reading
    rather than a single number.
    """
    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]), xp=xp, courant=courant)
    thickness = tuple((0, 0) if grid.shape[axis] < 6 else (2, 2) for axis in range(3))
    layer = PML(grid=grid, thickness=thickness)
    fields = Fields(grid=grid)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    install_epsilon(fields, grid, rng)
    install_nonlinearity(fields, grid, spec, amplitude, rng)
    return fields, layer, grid


def install_epsilon(fields, grid, rng) -> None:
    """THREE INDEPENDENT inverse-epsilon volumes, never one array bound thrice.

    Binding one volume for all three components is defect 2 of the complex
    template. A fixture that handed one array to all three could not see it, and
    ``bind_Ez_inv_eps_for_all_three`` would come back UNCAUGHT for a reason that
    says nothing about the kernel. Drawn AWAY FROM 1.0 so that
    ``drop_inverse_epsilon`` is not bit-identical.
    """
    xp = grid.xp
    forward, inverse = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        values = rng.uniform(1.2, 3.4, size=grid.shape).astype(np.float32)
        forward[component] = xp.asarray(values)
        inverse[component] = xp.asarray((np.float32(1.0) / values).astype(np.float32))
    fields.set_epsilon_volumes(forward, inverse)


def install_nonlinearity(fields, grid, spec: Dict[str, Any], amplitude: str,
                         rng) -> None:
    """MEEP's both-or-neither chi2/chi3 pair, per component.

    ``spec["linear_components"]`` names components that get NO pair at all --
    MEEP deletes a trivial pair (structure.cpp:822-826, fields.py:879-880), which
    is what makes ``covers_real_pml_constitutive`` and this family disjoint, and
    what puts a component on MEEP's ``else if (u)`` branch.

    ``spec["chi_is_volume"]`` draws a SPATIALLY VARYING volume: a uniform chi
    cannot distinguish a chi lookup that reads the wrong cell.
    """
    xp = grid.xp
    second, third = CHI_BY_AMPLITUDE[amplitude]
    linear = set(spec.get("linear_components", ()))
    chi2: Dict[str, Any] = {}
    chi3: Dict[str, Any] = {}
    for component in ("Ex", "Ey", "Ez"):
        if component in linear:
            continue
        if spec.get("chi_is_volume"):
            shape = grid.shape
            chi2[component] = xp.asarray(
                (np.float32(second) * rng.uniform(0.4, 1.6, size=shape)
                 ).astype(np.float32))
            chi3[component] = xp.asarray(
                (np.float32(third) * rng.uniform(0.4, 1.6, size=shape)
                 ).astype(np.float32))
        else:
            chi2[component] = float(second)
            chi3[component] = float(third)
    fields.set_nonlinear_volumes(chi2, chi3)


def seed_state(fields, grid, side: str, value_class: str, rng) -> Dict[str, np.ndarray]:
    """Physical-band or subnormal-band values in every array the sub-step touches.

    THE AUXILIARIES START NONZERO in both classes. A zero ``f_w`` makes
    ``kms * prev`` exactly zero on the first launch whatever ``kms`` holds, so a
    mis-indexed coefficient would only show from step two.

    THE MATERIAL STAYS NORMAL under the band class: driving inverse epsilon into
    the band too would make every product underflow and the leg would measure the
    fixture rather than the policy's reach into this arithmetic.
    """
    xp = grid.xp
    names = state_names(side)
    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=grid.shape).astype(np.float32)
                for name in names}
    elif value_class == "subnormal_band":
        host = subnormal_band_hosts(names, tuple(grid.shape), rng)
    else:
        raise ValueError(f"value class {value_class!r} is not one of the swept ones")
    for name, values in host.items():
        getattr(fields, name)[...] = xp.asarray(np.ascontiguousarray(values))
    return host


def snapshot(fields, side: str) -> Dict[str, np.ndarray]:
    return {name: to_host(getattr(fields, name)).copy() for name in state_names(side)}


def restore(fields, frozen: Dict[str, np.ndarray]) -> None:
    xp = fields.grid.xp
    for name, values in frozen.items():
        getattr(fields, name)[...] = xp.asarray(values)


def advance_sources(fields, side: str) -> None:
    """Move the source between launches, exactly the same way on both paths.

    A real run's curl rewrites B/D before every constitutive call. Held fixed,
    ``f_w`` reaches a fixed point and 60 launches measure what one launch does.
    ALL THREE D volumes move on the E side, because all three are read.
    """
    names = ("Dx", "Dy", "Dz") if side == "E" else SIDE_ARRAYS[side]["sources"]
    for name in names:
        getattr(fields, name)[...] = getattr(fields, name) * _ADVANCE


# ---------------------------------------------------------------------------
# The tables, the bindings, and the host mutations that corrupt them
# ---------------------------------------------------------------------------

def tables_for(side: str, layer) -> Dict[str, Any]:
    """The six flattened kps/kms views this side reads.

    ``constitutive_sub_lattice`` is asked here rather than hard-coding the suffix,
    because it is the same function the PREDICATE asks; getting it backwards is a
    half-cell error in the absorber profile -- converged, smooth and wrong.
    """
    suffix = "_h" if coverage.constitutive_sub_lattice(side) else ""
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def wrong_sub_lattice_tables(side: str, layer) -> Dict[str, Any]:
    suffix = "" if coverage.constitutive_sub_lattice(side) else "_h"
    return {f"{stem}_{axis}": getattr(layer, f"{stem}_{axis}{suffix}").reshape(-1)
            for axis in "xyz" for stem in ("kps", "kms")}


def swapped_kps_kms_tables(side: str, layer) -> Dict[str, Any]:
    base = tables_for(side, layer)
    out = dict(base)
    for axis in "xyz":
        out[f"kps_{axis}"] = base[f"kms_{axis}"]
        out[f"kms_{axis}"] = base[f"kps_{axis}"]
    return out


def linearized_bindings(fields) -> Dict[str, Any]:
    """THE E ARM'S REFUSAL, ARMED AT THE HOST: every component reported linear.

    The kernel then runs its NL=0 arm -- the certified plain body -- on a run whose
    array path applies a Pade factor. If that is bit-identical, the case has no
    live nonlinearity in it and every nonlinear leg on it is measuring nothing.
    MUST BE CAUGHT, and it is the host-side twin of ``drop_pade_factor``.
    """
    bindings = nlc.nonlinear_bindings(fields)
    bindings = dict(bindings)
    bindings["nonlinear"] = [False, False, False]
    return bindings


def swapped_chi_bindings(fields) -> Dict[str, Any]:
    """chi2 and chi3 exchanged at the BINDING, not in the device text.

    The device-text twin is ``swap_chi2_chi3_operands``; this one proves the same
    defect is visible when it enters through the plan, which is where a real
    wiring error would introduce it.
    """
    bindings = dict(nlc.nonlinear_bindings(fields))
    for key_a, key_b in (("chi2", "chi3"), ("chi2_is_volume", "chi3_is_volume"),
                         ("chi2_scalar", "chi3_scalar")):
        bindings[key_a], bindings[key_b] = bindings[key_b], bindings[key_a]
    return bindings


def host_mutated(name: str, side: str, fields, layer):
    """Returns ``(tables, bindings)`` with exactly one of them corrupted."""
    tables = tables_for(side, layer)
    bindings = nlc.nonlinear_bindings(fields) if side == "E" else None
    if name == "swap_constitutive_sublattice":
        return wrong_sub_lattice_tables(side, layer), bindings
    if name == "swap_kps_kms":
        return swapped_kps_kms_tables(side, layer), bindings
    if name == "linearize_the_bindings":
        return tables, linearized_bindings(fields)
    if name == "swap_chi_bindings":
        return tables, swapped_chi_bindings(fields)
    raise KeyError(f"unknown host mutation {name!r}")


# ---------------------------------------------------------------------------
# The two kernel-side backends
# ---------------------------------------------------------------------------

def run_kernel_cuda(side: str, fields, tables, bindings, codes) -> None:
    """The SHIPPED kernels, through their own public entry points.

    ``tables`` (and on the E side ``bindings``) are passed explicitly -- the
    entry points' keyword-only gate doors, which exist so a harness can hand in
    deliberately corrupted ones. Passing the layer instead would derive them
    correctly and disarm every host mutation.
    """
    if side == "H":
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        constitutive_kernels.update_fused_pml_real("H", fields, tables=tables)
    else:
        nlc.update_E_fused_pml_real_nonlinear(
            fields, tables=tables, codes=codes, bindings=bindings)
    cp.cuda.runtime.deviceSynchronize()


def _broadcast(vector, axis: int):
    shape = [1, 1, 1]
    shape[axis] = int(np.asarray(vector).size)
    return np.asarray(vector).reshape(shape)


def _neighbour(n: int, code: int, delta: int):
    """One axis's neighbour index and validity, from the device string's branch."""
    index = np.arange(n) + delta
    if code == BC_METALLIC:
        valid = (index >= 0) & (index < n)
        return np.where(valid, index, 0), valid
    return np.mod(index, n), np.ones(n, bool)


def four_point_sum_numpy(g, own_axis: int, partner_axis: int, codes, shape):
    """``four_point_sum`` from the device string, in float32 NumPy.

    THE SHIFTED-PAIR ASSOCIATION: ``(g[i] + g[i-s1]) + (g[i+s] + g[i+s-s1])``.
    """
    identity = [np.arange(shape[axis]) for axis in range(3)]
    up_index, up_valid = _neighbour(shape[own_axis], codes[own_axis], +1)
    down_index, down_valid = _neighbour(shape[partner_axis], codes[partner_axis], -1)

    def corner(own_up: bool, partner_down: bool):
        picks = list(identity)
        mask = np.ones(shape, bool)
        if own_up:
            picks[own_axis] = up_index
            bshape = [1, 1, 1]
            bshape[own_axis] = shape[own_axis]
            mask = mask & up_valid.reshape(bshape)
        if partner_down:
            picks[partner_axis] = down_index
            bshape = [1, 1, 1]
            bshape[partner_axis] = shape[partner_axis]
            mask = mask & down_valid.reshape(bshape)
        return np.where(mask, g[np.ix_(*picks)], np.float32(0.0)).astype(np.float32)

    return ((corner(False, False) + corner(False, True))
            + (corner(True, False) + corner(True, True))).astype(np.float32)


def pade_pieces(fields, bindings, codes) -> Dict[str, Any]:
    """``u``, ``c2``, ``c3`` per live component -- the floors' raw material.

    Computed from the device tree's own expressions rather than re-derived, so
    ``pade_departure`` describes the arithmetic the kernel performs and not an
    approximation of it.
    """
    shape = tuple(fields.grid.shape)
    displacement = {axis: to_host(getattr(fields, f"D{'xyz'[axis]}"))
                    for axis in range(3)}
    one, two, three = np.float32(1.0), np.float32(2.0), np.float32(3.0)
    out: Dict[str, Any] = {}
    for component, _source, axis in nlc.E_TERMS:
        if not bindings["nonlinear"][axis]:
            continue
        gs = displacement[axis]
        us = to_host(fields.inverse_epsilon_for(component))
        first, second = nlc.TRANSVERSE_PARTNERS[axis]
        g1s = four_point_sum_numpy(displacement[first], axis, first, codes, shape)
        g2s = four_point_sum_numpy(displacement[second], axis, second, codes, shape)
        dsqr = (gs * gs + np.float32(0.0625) * (g1s * g1s + g2s * g2s)).astype(np.float32)
        chi2 = (to_host(bindings["chi2"][axis]) if bindings["chi2_is_volume"][axis]
                else bindings["chi2_scalar"][axis])
        chi3 = (to_host(bindings["chi3"][axis]) if bindings["chi3_is_volume"][axis]
                else bindings["chi3_scalar"][axis])
        us_sq = (us * us).astype(np.float32)
        us_cu = (us_sq * us).astype(np.float32)
        c2 = ((gs * chi2) * us_sq).astype(np.float32)
        c3 = ((dsqr * chi3) * us_cu).astype(np.float32)
        num = ((one + c2) + two * c3).astype(np.float32)
        den = ((one + two * c2) + three * c3).astype(np.float32)
        out[component] = {"u": (num / den).astype(np.float32), "c2": c2, "c3": c3,
                          "num": num, "den": den, "dsqr": dsqr,
                          "gs_squared": (gs * gs).astype(np.float32)}
    return out


def run_kernel_numpy(side: str, fields, tables, bindings, codes) -> None:
    """The device tree, transcribed, in float32 -- the laptop backend.

    THIS COMPILES NOTHING AND CERTIFIES NOTHING. It cannot see a defect the NVRTC
    contraction guard exists for, it does not exercise the subnormal policy, and it
    is not the shipped bytes. What it settles is (a) whether the transcription
    reproduces ``stepping`` at all and (b) whether this harness can fail -- both on
    a laptop, before a device slot is spent.
    """
    shape = tuple(fields.grid.shape)
    spec = SIDE_ARRAYS[side]
    sources: List[Any] = []
    if side == "E":
        displacement = {axis: np.asarray(getattr(fields, f"D{'xyz'[axis]}"))
                        for axis in range(3)}
        one, two, three = np.float32(1.0), np.float32(2.0), np.float32(3.0)
        for component, _source, axis in nlc.E_TERMS:
            gs = displacement[axis]
            us = np.asarray(fields.inverse_epsilon_for(component))
            src = (gs * us).astype(np.float32)
            if bindings["nonlinear"][axis]:
                first, second = nlc.TRANSVERSE_PARTNERS[axis]
                g1s = four_point_sum_numpy(displacement[first], axis, first, codes, shape)
                g2s = four_point_sum_numpy(displacement[second], axis, second, codes, shape)
                dsqr = (gs * gs
                        + np.float32(0.0625) * (g1s * g1s + g2s * g2s)).astype(np.float32)
                chi2 = (np.asarray(bindings["chi2"][axis])
                        if bindings["chi2_is_volume"][axis]
                        else bindings["chi2_scalar"][axis])
                chi3 = (np.asarray(bindings["chi3"][axis])
                        if bindings["chi3_is_volume"][axis]
                        else bindings["chi3_scalar"][axis])
                us_sq = (us * us).astype(np.float32)
                us_cu = (us_sq * us).astype(np.float32)
                c2 = ((gs * chi2) * us_sq).astype(np.float32)
                c3 = ((dsqr * chi3) * us_cu).astype(np.float32)
                num = ((one + c2) + two * c3).astype(np.float32)
                den = ((one + two * c2) + three * c3).astype(np.float32)
                src = (src * (num / den).astype(np.float32)).astype(np.float32)
            sources.append(src)
    else:
        sources = [np.asarray(getattr(fields, name)).astype(np.float32)
                   for name in spec["sources"]]
    for axis, name in enumerate("xyz"):
        target = np.asarray(getattr(fields, spec["targets"][axis]))
        auxiliary = np.asarray(getattr(fields, spec["aux"][axis]))
        kps = _broadcast(tables[f"kps_{name}"], axis)
        kms = _broadcast(tables[f"kms_{name}"], axis)
        previous = auxiliary.copy()
        auxiliary[...] = sources[axis]
        accumulated = (target + kps * sources[axis]).astype(np.float32)
        target[...] = (accumulated - kms * previous).astype(np.float32)


# ---------------------------------------------------------------------------
# The floors
# ---------------------------------------------------------------------------

def oracle_moved(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray],
                 side: str) -> float:
    """Fraction of output WORDS the array path changed from the frozen input.

    Compared as raw uint32, never with ``allclose``: ``-0.0 == 0.0`` and
    ``NaN != NaN`` both lie, and the subnormal-band class puts signed zeros in the
    operands deliberately.
    """
    moved = total = 0
    for name in outputs(side):
        a = np.ascontiguousarray(before[name], dtype=np.float32).ravel().view(np.uint32)
        b = np.ascontiguousarray(after[name], dtype=np.float32).ravel().view(np.uint32)
        moved += int(np.count_nonzero(a != b))
        total += int(a.size)
    return moved / total if total else 0.0


def absorber_departure(side: str, layer, grid) -> Dict[str, Any]:
    """``max|kps-1|`` and ``max|kms-1|`` per axis. Identity means untestable."""
    tables = tables_for(side, layer)
    per_axis = []
    worst = 0.0
    for axis, name in enumerate("xyz"):
        deviation = 0.0
        for stem in ("kps", "kms"):
            values = to_host(tables[f"{stem}_{name}"]).astype(np.float64)
            deviation = max(deviation, float(np.max(np.abs(values - 1.0))))
        per_axis.append({"axis": axis, "name": name, "max_departure": deviation})
        worst = max(worst, deviation)
    return {"per_axis": per_axis, "max_departure": worst,
            "meets_floor": worst > 0.0}


def epsilon_departure(fields) -> Dict[str, Any]:
    """Inverse epsilon away from 1.0, and the three volumes genuinely distinct."""
    volumes = [to_host(fields.inverse_epsilon_for(c)) for c in ("Ex", "Ey", "Ez")]
    from_identity = max(float(np.max(np.abs(v.astype(np.float64) - 1.0)))
                        for v in volumes)
    distinct = not (np.array_equal(volumes[0], volumes[1])
                    and np.array_equal(volumes[1], volumes[2]))
    return {"max_departure_from_identity": from_identity,
            "three_volumes_distinct": bool(distinct),
            "meets_floor": from_identity > 0.0 and bool(distinct)}


def _subnormal_census(values: np.ndarray) -> Dict[str, int]:
    raw = np.ascontiguousarray(values, dtype=np.float32).ravel().view(np.uint32)
    exponent = (raw >> 23) & 0xFF
    mantissa = raw & 0x7FFFFF
    return {"values": int(raw.size),
            "subnormals": int(((exponent == 0) & (mantissa != 0)).sum()),
            "zeros": int((raw == 0).sum())}


def pade_facts(pieces: Dict[str, Any]) -> Dict[str, Any]:
    """``pade_departure``, ``transverse_departure`` and the two subnormal censuses.

    ``pade_departure`` is THE floor this family exists for: with ``u == 1``
    everywhere the kernel is the certified linear one wearing a new name and every
    nonlinear mutation is inert.

    ``pade_operand_census`` answers the ``div.rn`` .FTZ question with a number
    rather than an argument: if no word of ``num``, ``den`` or ``u`` is subnormal,
    ptxas's FTZ-carrying range checks have nothing in this expression to reach.
    """
    if not pieces:
        return {"live_components": [], "pade_departure": 0.0,
                "transverse_departure": 0.0, "meets_floor": False}
    departure = 0.0
    transverse = 0.0
    quotient = []
    terms = []
    for _component, part in pieces.items():
        departure = max(departure, float(np.max(np.abs(
            part["u"].astype(np.float64) - 1.0))))
        transverse = max(transverse, float(np.max(np.abs(
            part["dsqr"].astype(np.float64) - part["gs_squared"].astype(np.float64)))))
        quotient.extend([part["num"], part["den"], part["u"]])
        terms.extend([part["c2"], part["c3"]])
    return {
        "live_components": sorted(pieces),
        "pade_departure": departure,
        "transverse_departure": transverse,
        "pade_operand_census": _subnormal_census(np.concatenate(
            [q.ravel() for q in quotient])),
        "chi_term_census": _subnormal_census(np.concatenate(
            [t.ravel() for t in terms])),
        "meets_floor": departure > 0.0 and transverse > 0.0,
    }


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def one_case(backend: str, spec: Dict[str, Any], side: str, courant: float,
             value_class: str, amplitude: str, guard: str,
             host_mutation: Optional[str] = None) -> Dict[str, Any]:
    """One frozen state, run twice: the array path, then the kernel.

    THE ORACLE IS ``stepping`` ITSELF on real ``Grid``/``Fields``/``PML`` objects.
    There is no second transcription on the oracle leg to drift: the kernel is
    compared against the thing it claims to reproduce, byte for byte, from ONE
    frozen state -- oracle runs, state is restored, kernel runs.
    """
    started = time.time()
    xp = cp if backend == "cuda" else _NumpyWearingCupysName()
    rng = np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            f"{spec['label']}|{side}|{courant}|{value_class}|{amplitude}".encode()).digest()[:4], "big"))

    fields, layer, grid = build(xp, spec, courant, amplitude, rng)
    host = seed_state(fields, grid, side, value_class, rng)
    codes, code_refusal = nlc.boundary_codes(grid)

    case: Dict[str, Any] = {
        "label": spec["label"], "side": side, "courant": courant,
        "value_class": value_class, "amplitude": amplitude, "guard": guard,
        "backend": backend, "host_mutation": host_mutation,
        "boundaries": list(spec["boundaries"]),
        "boundary_codes": list(codes) if codes else None,
        "shape": [int(n) for n in grid.shape],
        "linear_components": list(spec.get("linear_components", ())),
        "chi_is_volume": bool(spec.get("chi_is_volume", False)),
        "operand_census": operand_census(host),
    }
    if code_refusal is not None:
        case["skipped"] = f"the grid has no ghost-rule code: {code_refusal}"
        case["seconds"] = time.time() - started
        return case

    # BOTH PREDICATES' ANSWERS, recorded rather than acted on. This gate decides
    # whether the census's refusal should change, so it must not be gated by it --
    # but a record that did not carry both could not show that the two PARTITION.
    covered, reason = nlc.covers_real_pml_nonlinear_constitutive(fields, layer, grid, side)
    sibling, sibling_reason = coverage.covers_real_pml_constitutive(
        fields, layer, grid, side)
    case["predicate"] = {
        "nonlinear_family": {"covered": bool(covered), "reason": reason},
        "linear_family": {"covered": bool(sibling), "reason": sibling_reason},
        "disjoint": not (bool(covered) and bool(sibling)),
    }

    if not covered and reason and reason.startswith("no instantaneous chi2/chi3"):
        # MEASURED, and it is a finding rather than a nuisance: under the
        # ``meep_x86_flush`` policy a float32 chi VOLUME drawn in the subnormal
        # band is annihilated on the host at install time, so
        # ``Fields.set_nonlinear_volumes`` sees a trivial pair and drops it
        # (fields.py:879-880, MEEP's structure.cpp:822-826) and the run is
        # LINEAR. That configuration belongs to
        # ``coverage.covers_real_pml_constitutive``, not to this family, and the
        # predicate is right to refuse it. Measuring it here anyway would be
        # measuring the certified linear pair under this gate's name.
        case["skipped"] = ("the installed chi2/chi3 pair is trivial on this "
                           "backend and policy, so this run is LINEAR and belongs "
                           "to the linear constitutive family; "
                           f"the predicate says: {reason}")
        case["seconds"] = time.time() - started
        return case

    absorber = absorber_departure(side, layer, grid)
    case["absorber_departure"] = absorber
    if not absorber["meets_floor"]:
        case["skipped"] = ("every coefficient vector is the identity; a "
                           "coefficient-index error would be invisible")
        case["seconds"] = time.time() - started
        return case
    if side == "E":
        case["epsilon_departure"] = epsilon_departure(fields)
        if not case["epsilon_departure"]["meets_floor"]:
            case["skipped"] = ("inverse epsilon is the identity or one volume is "
                               "bound thrice; two mutations would be invisible")
            case["seconds"] = time.time() - started
            return case

    bindings = nlc.nonlinear_bindings(fields) if side == "E" else None
    if side == "E":
        pieces = pade_pieces(fields, bindings, codes)
        case["pade"] = pade_facts(pieces)
        margin = stepping.nonlinear_margin(fields, layer)
        case["margin"] = (None if margin is None else
                          {"component": margin.component,
                           "denominator": margin.denominator,
                           "expansion": margin.expansion,
                           "inside_guard": bool(
                               margin.expansion < nlc.POLE_EXPANSION_BOUND)})
        if margin is None or not case["margin"]["inside_guard"]:
            case["skipped"] = ("the run is outside the pole guard's admitted domain "
                               "(expansion >= 1/3); the engine refuses to perform "
                               "this arithmetic, so measuring it certifies nothing")
            case["seconds"] = time.time() - started
            return case
        if ((value_class, amplitude) in PADE_FLOOR_CLASSES
                and not case["pade"]["meets_floor"]):
            case["skipped"] = ("the Pade factor is the identity everywhere, or Dsqr "
                               "equals gs*gs; on this amplitude class that makes "
                               "every nonlinear mutation inert")
            case["seconds"] = time.time() - started
            return case

    frozen = snapshot(fields, side)

    # Leg 1: the oracle.
    if side == "H":
        stepping.update_H(fields, layer)
    else:
        stepping.update_E(fields, layer)
    reference = snapshot(fields, side)

    moved = oracle_moved(frozen, reference, side)
    case["oracle_moved"] = moved
    if moved == 0.0:
        case["skipped"] = ("the array path changed no output word from the frozen "
                           "input; zero-init is a fixed point of this recurrence")
        case["seconds"] = time.time() - started
        return case

    # Leg 2: the kernel, from the SAME frozen state.
    restore(fields, frozen)
    if host_mutation is None:
        tables, use_bindings = tables_for(side, layer), bindings
    else:
        tables, use_bindings = host_mutated(host_mutation, side, fields, layer)
    runner = run_kernel_cuda if backend == "cuda" else run_kernel_numpy
    runner(side, fields, tables, use_bindings, codes)

    parts = {name: bit_compare(reference[name], getattr(fields, name))
             for name in outputs(side)}
    case["single_launch"] = combine(parts)

    # Leg 3: the multi-step. The auxiliary is STATE.
    #
    # ONE ``Fields`` OBJECT, RUN TWICE FROM THE SAME FROZEN STATE, rather than a
    # second object built beside it: two objects means two epsilon draws unless
    # the volumes are copied across, and a fixture that solved two different
    # problems would report a divergence that says nothing about the kernel.
    if host_mutation is None:
        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            if side == "H":
                stepping.update_H(fields, layer)
            else:
                stepping.update_E(fields, layer)
            advance_sources(fields, side)
        oracle_multi = snapshot(fields, side)

        restore(fields, frozen)
        for _ in range(MULTI_STEP_BUDGET):
            runner(side, fields, tables, use_bindings, codes)
            advance_sources(fields, side)
        multi = {name: bit_compare(oracle_multi[name], getattr(fields, name))
                 for name in outputs(side)}
        case["multi_step"] = combine(multi)
        case["multi_step"]["launches"] = MULTI_STEP_BUDGET

    case["seconds"] = time.time() - started
    return case


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def case_product(product: str):
    specs = SPECS if product == "full" else SPECS[:3]
    courants = COURANTS if product == "full" else (INEXACT_COURANT,)
    classes = CLASSES if product == "full" else CLASSES[:1]
    return [(spec, side, courant, value_class, amplitude)
            for spec in specs
            for side in SIDES
            for courant in courants
            for value_class, amplitude in classes]


def run_sweep(results: Dict[str, Any], out_path: str, backend: str,
              product: str, guard: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    plan = case_product(product)
    for index, (spec, side, courant, value_class, amplitude) in enumerate(plan, start=1):
        case = one_case(backend, spec, side, courant, value_class, amplitude, guard)
        cases.append(case)
        results.setdefault("sweep", {})[guard] = cases
        save(results, out_path)
        head = (f"[{guard}] case {index}/{len(plan)} {spec['label']} {side} "
                f"c={courant} {value_class}/{amplitude}")
        if case.get("skipped"):
            log(f"{head} SKIPPED: {case['skipped'][:70]}")
            continue
        single = case["single_launch"]["bit_identical"]
        multi = case.get("multi_step", {}).get("bit_identical")
        pade = case.get("pade", {}).get("pade_departure")
        log(f"{head} shape={case['shape']} "
            f"single={'IDENTICAL' if single else 'DIVERGED'} "
            f"multi={'IDENTICAL' if multi else ('DIVERGED' if multi is False else '-')} "
            f"moved={case['oracle_moved']:.3f} "
            f"|u-1|={'-' if pade is None else format(pade, '.3e')} "
            f"({case['seconds']:.1f} s)")
    return cases


#: The specs a mutation leg is scored on. CHOSEN, not sliced off the front of the
#: sweep -- the folded gate learned that the hard way, scoring its one fold-only
#: mutation 0/0 UNCAUGHT because every selected leg was an unfolded control. Every
#: ghost rule and both chi spellings appear here.
MUTATION_SPEC_LABELS: Tuple[str, ...] = (
    "all_periodic",       # the wrap, on every axis
    "all_metallic",       # the zero ghost, on every axis
    "mixed_pmp",          # one of each, so a swapped axis code cannot hide
    "one_dimensional",    # the corpus shape: two axes of one cell
    "partly_nonlinear",   # one component on MEEP's ``else if (u)`` branch
    "volume_chi",         # a spatially varying chi, so a chi index error bites
)


def mutation_plan(product: str, side_filter: Optional[str] = None):
    """One case per (spec, side) for the mutation legs, at the INEXACT courant.

    Held to one courant and one class deliberately: a mutation leg answers "can
    this gate see this defect at all", and multiplying it by the whole sweep buys
    repetitions of that answer rather than a second question.

    THE CLASS IS ``uniform/near_pole``, AND THAT IS A MEASURED CHOICE. The two
    degenerate classes make ``u`` exactly 1 by construction and would score every
    nonlinear mutation UNCAUGHT for a reason about the fixture. Between the two
    live ones, a defect that enters through ``Dsqr`` reaches the output only
    through ``c3 = (Dsqr * chi3) * chi1inv^3`` -- a weight of order
    ``0.0625 * chi3 * chi1inv^3``, which at the moderate amplitude is ~3e-4, small
    enough to round a one-ulp stencil difference away entirely. MEASURED on
    the GPU host: at ``moderate``, ``four_point_left_to_right`` came back 0 differing
    words on the (1, 1, nz) corpus shape under the keep policy and NONZERO under
    flush -- the same defect sitting exactly on the edge of float32 visibility.
    ``near_pole`` raises that weight by 3.2x and is where the question "can this
    gate see this defect" is actually asked.

    WHAT THAT DOES NOT ESTABLISH is carried in the summary's ``does_not_claim``:
    the mutation legs do not show that every defect is visible at every amplitude
    the family admits, only that the comparator can see it.
    """
    by_label = {spec["label"]: spec for spec in SPECS}
    labels = MUTATION_SPEC_LABELS if product == "full" else MUTATION_SPEC_LABELS[:3]
    return [(by_label[label], side, INEXACT_COURANT, "uniform", "near_pole")
            for label in labels for side in SIDES
            if side_filter is None or side == side_filter]


def leg_caught(case: Dict[str, Any]) -> bool:
    """Did this mutated leg diverge from the array path at EITHER granularity?"""
    if case.get("skipped"):
        return False
    if not case["single_launch"]["bit_identical"]:
        return True
    multi = case.get("multi_step")
    return bool(multi is not None and not multi["bit_identical"])


def _verdict(scored: int, caught: int, must_be_caught: bool) -> str:
    if not scored:
        # NO LEGS is its own verdict, never "UNCAUGHT". A mutation that was never
        # run has not been shown inert; it has not been asked.
        return "NO LEGS"
    if must_be_caught:
        return "CAUGHT" if caught == scored else "PARTIAL" if caught else "UNCAUGHT"
    return "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"


def run_host_mutations(results: Dict[str, Any], out_path: str, backend: str,
                       product: str) -> Dict[str, Any]:
    """Every table- and binding-level defect, on both backends."""
    out: Dict[str, Any] = {}
    for name, must_be_caught in HOST_MUTATIONS.items():
        legs: List[Dict[str, Any]] = []
        for spec, side, courant, value_class, amplitude in mutation_plan(product):
            if name in E_ONLY_HOST_MUTATIONS and side != "E":
                continue
            legs.append(one_case(backend, spec, side, courant, value_class,
                                 amplitude, "fmad_false", host_mutation=name))
        scored = [c for c in legs if not c.get("skipped")]
        caught = sum(1 for c in scored if leg_caught(c))
        out[name] = {"ran": len(scored), "caught": caught,
                     "uncaught": len(scored) - caught,
                     "must_be_caught": must_be_caught,
                     "verdict": _verdict(len(scored), caught, must_be_caught),
                     "cases": legs}
        log(f"[host-mut] {name}: caught {caught}/{len(scored)} -> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


def _score_source_mutation(key: str, module, attribute: str, transform,
                           must_be_caught: bool, plan, compile_cache) -> Dict[str, Any]:
    """Apply one device-text defect, score its legs, and always restore the source."""
    original = getattr(module, attribute)
    mutated, sites = transform(original)
    if sites == 0 or mutated == original:
        log(f"[src-mut] {key}: NOT ARMED ({sites} sites)")
        return {"armed": False,
                "why": (f"matched {sites} site(s) and changed nothing; the mutation "
                        f"and the kernel have drifted apart")}
    digest = hashlib.sha256(mutated.encode("utf-8")).hexdigest()
    legs: List[Dict[str, Any]] = []
    try:
        setattr(module, attribute, mutated)
        compile_cache.clear_compile_log()
        for spec, side, courant, value_class, amplitude in plan:
            legs.append(one_case("cuda", spec, side, courant, value_class,
                                 amplitude, "fmad_false"))
    finally:
        setattr(module, attribute, original)
    scored = [c for c in legs if not c.get("skipped")]
    caught = sum(1 for c in scored if leg_caught(c))
    from_mutated = sum(1 for entry in compile_cache.compile_log()
                       if entry["source_sha256"] == digest)
    record = {
        "armed": True, "sites": sites, "mutated_source_sha256": digest,
        "specs_scored": sorted({leg[0]["label"] for leg in plan}),
        "kernel_constructions_from_mutated_bytes": from_mutated,
        "ran": len(scored), "caught": caught,
        "caught_at_single_launch": sum(
            1 for c in scored if not c["single_launch"]["bit_identical"]),
        "must_be_caught": must_be_caught,
        "verdict": _verdict(len(scored), caught, must_be_caught),
        "cases": legs,
    }
    if from_mutated == 0 and scored:
        # A leg reporting a pass for a mutation it never applied is worse than no
        # leg: the rewrite matched, the memo is keyed through the source, and yet
        # nothing was built from these bytes -- so this leg measured the SHIPPED
        # kernel and its verdict is about nothing.
        record["verdict"] = "UNACCOUNTED"
        record["why"] = ("no kernel construction used the mutated bytes; this leg "
                         "did not exercise the mutation")
    log(f"[src-mut] {key}: caught {caught}/{len(scored)} "
        f"builds_from_mutated={from_mutated} -> {record['verdict']}")
    return record


def run_source_mutations(results: Dict[str, Any], out_path: str,
                         product: str) -> Dict[str, Any]:
    """Every device-text defect, applied to the SHIPPED strings and recompiled.

    The compile memo is keyed through the SOURCE, so a mutated body is a miss and
    reaches NVRTC. Every leg records how many constructions came from the mutated
    bytes, because a leg reporting a pass for a mutation it never applied is worse
    than no leg -- three measured instances on the sibling track.
    """
    from meep_gpu.cuda_kernels import compile_cache, constitutive_kernels  # noqa: PLC0415

    out: Dict[str, Any] = {}
    e_attribute = "_update_E_pml_real_nonlinear_kernel_code"
    h_attribute = "_update_H_pml_real_kernel_code"
    e_plan = mutation_plan(product, side_filter="E")
    h_plan = mutation_plan(product, side_filter="H")

    for name, (transform, must_be_caught) in E_SOURCE_MUTATIONS.items():
        out[f"E:{name}"] = _score_source_mutation(
            f"E:{name}", nlc, e_attribute, transform, must_be_caught,
            specs_for_source_mutation(name, e_plan), compile_cache)
        results["source_mutations"] = out
        save(results, out_path)
    for name, must_be_caught in SHARED_E_SOURCE_MUTATIONS.items():
        out[f"E:{name}"] = _score_source_mutation(
            f"E:{name}", nlc, e_attribute, probe.SOURCE_MUTATIONS[name],
            must_be_caught, specs_for_source_mutation(name, e_plan), compile_cache)
        results["source_mutations"] = out
        save(results, out_path)
    for name, must_be_caught in H_SOURCE_MUTATIONS.items():
        transform = (_pade_scale_the_H_source if name == "pade_scale_the_H_source"
                     else probe.SOURCE_MUTATIONS[name])
        out[f"H:{name}"] = _score_source_mutation(
            f"H:{name}", constitutive_kernels, h_attribute, transform,
            must_be_caught, specs_for_source_mutation(name, h_plan), compile_cache)
        results["source_mutations"] = out
        save(results, out_path)

    # THE SPELLING PROBE, scored the same way and reported separately. Not a
    # release clause: it is a claim about the compiler, not about this kernel.
    results["division_spelling_probe"] = _score_source_mutation(
        "E:plain_division", nlc, e_attribute, _plain_division, False,
        e_plan, compile_cache)
    save(results, out_path)
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any]) -> Dict[str, Any]:
    """The release decision, with every clause it rests on named."""
    reasons: List[str] = []
    sweep = results.get("sweep", {})
    primary = sweep.get("fmad_false", [])
    scored = [c for c in primary if not c.get("skipped")]
    electric = [c for c in scored if c["side"] == "E"]
    magnetic = [c for c in scored if c["side"] == "H"]
    single_ok = [c for c in scored if c["single_launch"]["bit_identical"]]
    multi_cases = [c for c in scored if "multi_step" in c]
    multi_ok = [c for c in multi_cases if c["multi_step"]["bit_identical"]]

    if not electric:
        reasons.append("no E-side case was scored at all; arm E is the kernel")
    if not magnetic:
        reasons.append("no H-side case was scored at all; arm H is the null")
    if len(single_ok) != len(scored):
        reasons.append(f"single-launch divergence on {len(scored) - len(single_ok)} "
                       f"of {len(scored)} cases")
    if len(multi_ok) != len(multi_cases):
        reasons.append(f"multi-step divergence on {len(multi_cases) - len(multi_ok)} "
                       f"of {len(multi_cases)} cases")

    # THE FAMILY'S OWN NON-VACUITY: at least one scored E case per ghost rule must
    # have had a Pade factor that is not the identity, or the sweep measured the
    # certified linear kernel under a new name.
    live = [c for c in electric if c.get("pade", {}).get("pade_departure", 0.0) > 0.0]
    ghost_rules = {tuple(c["boundary_codes"]) for c in live}
    if not live:
        reasons.append("no scored E case carried a Pade factor away from 1.0")
    if not any(BC_METALLIC in codes for codes in ghost_rules):
        reasons.append("no live E case had a metallic axis; the zero ghost in the "
                       "transverse sums was never exercised")
    if not any(set(codes) == {BC_PERIODIC} for codes in ghost_rules):
        reasons.append("no live E case was all-periodic; the wrap in the transverse "
                       "sums was never exercised")
    if not any(c["chi_is_volume"] for c in live):
        reasons.append("no live E case carried a volume chi; a chi index error "
                       "would be invisible on a uniform one")
    if not any(c["linear_components"] for c in live):
        reasons.append("no live E case carried a linear component; MEEP's "
                       "`else if (u)` branch was never exercised")

    # THE PREDICATES MUST PARTITION on every case, not merely be believed to.
    overlaps = [c["label"] for c in scored if not c["predicate"]["disjoint"]]
    if overlaps:
        reasons.append(f"both predicates admitted the same slot on {sorted(set(overlaps))}")
    unadmitted = [f"{c['label']}/{c['side']}" for c in scored
                  if not c["predicate"]["nonlinear_family"]["covered"]]
    if unadmitted and results.get("backend") == "cuda":
        reasons.append(f"this family's predicate refused a case this gate then "
                       f"measured: {sorted(set(unadmitted))}")

    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] == "NO LEGS":
            reasons.append(f"host mutation {name} was never scored; unasked is not inert")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']})")
    for key, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"source mutation {key} was not armed: {leg.get('why')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            inert = [f"{c['label']}/{c['side']}" for c in leg["cases"]
                     if not c.get("skipped")
                     and c["single_launch"]["bit_identical"]
                     and c.get("multi_step", {}).get("bit_identical", True)]
            reasons.append(f"source mutation {key} is {leg['verdict']} "
                           f"({leg['caught']}/{leg['ran']}); inert on {sorted(inert)}")

    control = [c for c in sweep.get("default_no_options", [])
               if not c.get("skipped") and c["courant"] == INEXACT_COURANT]
    control_diverged = [c for c in control if not c["single_launch"]["bit_identical"]]
    guard_control = {
        "scored_at_inexact_courant": len(control),
        "diverged": len(control_diverged),
        "reading": ("NOT MEASURED on this run: no unguarded leg was scored at the "
                    "inexact courant" if not control else
                    "the contraction guard is load-bearing on this sub-step"
                    if control_diverged else
                    "MEASURED DECORATIVE on these cases: the unguarded leg was "
                    "bit-identical too, so this run carries no evidence that "
                    "--fmad=false changed an answer here"),
    }

    division = results.get("division_spelling_probe", {})
    if not division.get("armed"):
        division_reading = "NOT MEASURED: the plain-division rewrite did not arm"
    elif division.get("verdict") == "NULL CONFIRMED":
        division_reading = ("MEASURED EQUIVALENT on this platform: `num / den` and "
                            "__fdiv_rn(num, den) produced identical bytes on every "
                            "leg, so CUDA's default -prec-div=true really is IEEE "
                            "round-to-nearest division here -- unlike Triton, where "
                            "the plain operator was measured at ~2 ulp")
    else:
        division_reading = ("MEASURED DIFFERENT: the plain operator diverged from "
                            "__fdiv_rn, so the intrinsic spelling is LOAD-BEARING on "
                            "this platform exactly as it is on the Triton track")

    subnormal = [c for c in electric if c["amplitude"] == "subnormal_chi"
                 and not c.get("skipped")]
    reach = {
        "cases": len(subnormal),
        "chi_terms_subnormal": sum(
            c["pade"]["chi_term_census"]["subnormals"] for c in subnormal),
        "quotient_operands_subnormal": sum(
            c["pade"]["pade_operand_census"]["subnormals"] for c in subnormal),
        "reading": None,
    }
    if not subnormal:
        reach["reading"] = "NOT MEASURED: no subnormal_chi case was scored"
    elif reach["chi_terms_subnormal"] == 0:
        reach["reading"] = ("the c2/c3 terms held no subnormal on this run, so this "
                            "class did not put one near the divide; under the flush "
                            "policy that is the expected reading and under keep it "
                            "means the class is vacuous and the chi magnitudes need "
                            "re-choosing")
    elif reach["quotient_operands_subnormal"] == 0:
        reach["reading"] = ("MEASURED INERT: subnormals were present in c2/c3 and NO "
                            "word of num, den or the quotient was subnormal, so "
                            "ptxas's FTZ-carrying div.rn range checks have nothing "
                            "in this expression to reach")
    else:
        reach["reading"] = ("a subnormal reached the quotient itself; the div.rn "
                            "FTZ exception in constitutive_kernels.py's header is "
                            "LIVE on this expression and the identity above is what "
                            "settles it")

    annihilated = sorted({
        f"{c['label']}/{c['value_class']}/{c['amplitude']}"
        for c in primary
        if "trivial on this backend and policy" in (c.get("skipped") or "")})

    return {
        "released": not reasons,
        "reasons": reasons,
        "guard_control": guard_control,
        "chi_annihilated_by_the_policy": {
            "cases": annihilated,
            "reading": ("no case had its chi2/chi3 pair annihilated at install"
                        if not annihilated else
                        "on these cases the installed chi2/chi3 pair was trivial "
                        "after the host policy rounded it, so the run is linear "
                        "and this family's predicate refused it -- the expected "
                        "reading under meep_x86_flush for a float32 chi VOLUME "
                        "drawn in the subnormal band"),
        },
        "division_spelling": {"verdict": division.get("verdict"),
                              "reading": division_reading},
        "subnormal_reach_into_the_divide": reach,
        "scored_cases": len(scored),
        "electric_cases": len(electric),
        "magnetic_cases": len(magnetic),
        "cases_with_a_live_pade_factor": len(live),
        "single_launch_identical": len(single_ok),
        "multi_step_identical": len(multi_ok),
        "multi_step_cases": len(multi_cases),
        "max_pade_departure": max(
            [c.get("pade", {}).get("pade_departure", 0.0) for c in electric] or [0.0]),
        "max_expansion": max(
            [(c.get("margin") or {}).get("expansion", 0.0) for c in electric] or [0.0]),
        "claim": ("meep_gpu/cuda_kernels/nonlinear_constitutive."
                  "update_E_pml_real_nonlinear is byte-identical to "
                  "stepping.update_E under an instantaneous chi2/chi3, and the "
                  "CERTIFIED update_H_pml_real is byte-identical to "
                  "stepping.update_H on the same grids with no change to its bytes"),
        "does_not_claim": [
            "nothing dispatches either kernel; no module in meep_gpu imports "
            "cuda_kernels at all",
            "a fold, a cylindrical grid, complex storage, a Bloch phase, a "
            "polarization and an off-diagonal chi1inv row are REFUSED by this "
            "family's predicate and are not measured here",
            "the CURL's own chi2/chi3 refusal is a different family's clause and "
            "is untouched: step_B and step_D on a nonlinear row stay unserved",
            "the mutation legs run at ONE amplitude class (uniform/near_pole) and "
            "one courant; they show the comparator can see each defect, not that "
            "every defect is visible at every amplitude the family admits -- one "
            "of them was measured invisible on the (1, 1, nz) shape at the "
            "moderate amplitude under the keep policy",
            "no throughput claim is made or possible",
        ],
    }


def save(results: Dict[str, Any], out_path: str) -> None:
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
    parser.add_argument("--import-meep-for-host-policy", action="store_true",
                        help=("import MEEP first so the HOST half of a 'flush' "
                              "policy can be attained; strict install refuses "
                              "otherwise"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    results: Dict[str, Any] = {
        "gate": "cuda_nonlinear_constitutive",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "backend": args.backend,
        "product": args.product,
        "question": ("does the census refusal 'instantaneous chi2/chi3: a Pade "
                     "factor replaces the constitutive product' describe a "
                     "divergence on BOTH sides, or only on update_E?"),
    }

    if args.backend == "cuda":
        if cp is None:
            log("[fatal] --backend cuda but CuPy did not import")
            results["status"] = "refused: no CuPy"
            save(results, args.out)
            return 2
        if args.import_meep_for_host_policy:
            results["meep_host_import"] = probe.import_meep_for_host_policy()
        # THE OBSERVER GOES IN BEFORE THE POLICY, and the order is the design:
        # under 'keep' the policy's strip wraps this, so it records the option
        # tuple NVRTC was really given (post-strip).
        results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
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

    for guard, options in GUARD_SETS:
        if args.backend == "numpy" and guard != "fmad_false":
            continue  # there is no compiler on this leg to guard
        if args.backend == "cuda":
            from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
            nlc._COMPILE_OPTIONS = tuple(options)
            constitutive_kernels._COMPILE_OPTIONS = tuple(options)
            nlc._clear_kernel_cache()
        log(f"[guard] {guard} options={options}")
        run_sweep(results, args.out, args.backend, args.product, guard)

    if args.backend == "cuda":
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
        nlc._COMPILE_OPTIONS = ("--fmad=false",)
        constitutive_kernels._COMPILE_OPTIONS = ("--fmad=false",)
        nlc._clear_kernel_cache()

    if not args.skip_mutations:
        run_host_mutations(results, args.out, args.backend, args.product)
        if args.backend == "cuda":
            run_source_mutations(results, args.out, args.product)

    if args.backend == "cuda":
        results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["summary"] = summarize(results)
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} scored={verdict['scored_cases']} "
        f"E={verdict['electric_cases']} H={verdict['magnetic_cases']} "
        f"live_pade={verdict['cases_with_a_live_pade_factor']} "
        f"single_identical={verdict['single_launch_identical']}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
