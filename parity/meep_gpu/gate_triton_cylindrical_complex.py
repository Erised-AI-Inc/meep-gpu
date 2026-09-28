"""Byte gate for the COMPLEX cylindrical (Dcyl, complex64 storage at every m)
Triton tranche.

m = 0 UNDER COMPLEX STORAGE WAS ADDED 2026-09-04 (``cylindrical_complex.M_ZERO``):
``M_CASES``, ``CONSTITUTIVE_M_VALUES``, ``AXIS_IDENTITY_M``, the mutation configs,
the shape block, the engine and multi-step legs and the reference battery all
carry m = 0 rows, and four needles (``m0_*``) hold the m = 0 axis pair on both the
reference and the kernel layer. The reference in this file already stepped m = 0
(its post-update axis rules branch on ``m == 0``); what changed is that the
kernel now has an arm for it and the sweeps reach it.

GATE-READY, NOT GATED. **No device run has been taken and NO BYTE-IDENTITY CLAIM
IS MADE.** The LAPTOP half runs here and passes; the DEVICE half is written,
wired to :func:`main`, and has never been executed — on a host without CuPy and
Triton every device leg skips with its reason recorded and the artifact's
``byte_identity_claim`` field reads NONE.

THE LAPTOP HALF (``--self-check``), all of it measured on NumPy 2026-08-13:

* ``reference``    — the in-file reference transcription against
  ``stepping.step_B``/``step_D`` on real complex Dcyl ``Grid``/``Fields``/``PML``
  objects, per sub-step, uint32: 544/544 rows identical at 8 sub-steps each, with
  the 24 (m, Courant) pairs above the accurate branch's stability bound RECORDED
  as skips and every ``M_CASES`` entry required to have contributed rows;
* ``constitutive`` — the IDENTITY leg: ``stepping.update_H``/``update_E`` against
  a plain complex elementwise ``dsigw`` reference carrying no cylindrical clause.
  480/480 rows, 0 differing words, 6 sub-steps each, EVERY row asserting its own
  non-vacuity. If this is 0/N the tranche needs one kernel, not two;
* ``constitutive_mutations`` — that reference broken on purpose. It exists
  because the identity leg's first cut had 256 rows that could not fail;
* ``reference_mutations`` — the reference battery, both seedings, with predicted
  nulls recorded WITH reasons and a per-seeding vacuity total;
* ``axis_identity``   — r forced METALLIC against the CYL_AXIS ghost: 0 differing
  words over 80/80 rows, which is what licenses ``BCX = METALLIC``;
* ``stripped_control``— every cylindrical clause switched off: 1,281,955
  differing words, which is what makes this tranche a NEW kernel;
* ``needle_arming``   — every needle in every battery applied to its target on
  the laptop. DISARMED costs milliseconds to find here and a device run there.

THE DEVICE HALF (default legs, or ``--legs``), what it will certify and against
what:

* ``synthetic`` — ``cylindrical_complex.plan_cylindrical_complex_curl_from_arrays``
  against :func:`reference_cyl_complex_step` (this file), uint32 compare of every
  target and every auxiliary, per sub-step, launch-counted;
* ``engine``    — the same through the ENGINE route on a real ``Grid``/``Fields``/
  ``PML``, so the shipped predicate has to admit the run, the shipped prefix has
  to be built from the engine's own ``StepScratch``, and the shipped sub-lattice
  binding has to be right. It passes the probe artifact in and ASSERTS coverage
  and launch count: the first cut passed no probe, so the predicate refused, no
  plan was built, and every row recorded "no divergence" having launched nothing;
* ``mutations`` / ``host_mutations`` — armed mutations of the shipped kernel's
  SOURCE and of the HOST construction, launch-counted, with DISARMED, NO-LAUNCH,
  STALE-BINARY, NEEDLE-MISSED and PREDICTION-BROKEN all failures, and predicted
  nulls recorded WITH their reason plus launch, cache-key and PTX evidence;
* ``guard``     — every synthetic row re-run at ``enable_fp_fusion=True``,
  reported as DATA. It is NOT asserted to bite: platform fact (d) is that fusion
  is not byte-uniform across tranches, and the certified complex family measured
  68/68 fusion-on rows identical. This family states the count it measures and
  certifies under the shipped ``ENABLE_FP_FUSION``;
* ``multi_step``— consecutive sub-steps to a STATED budget, per sub-step, with
  the CLAIMABLE budget computed from the first divergence observed.

CASE DISCIPLINE, clause by clause, and why each one is here:

1. **uint32 compares, never allclose.** Every leg.
2. **A NON-POWER-OF-TWO Courant in every sweep.** At 0.5 the dtdx scaling is
   exact in binary and the FMA/associativity discrepancies vanish; 0.5 ALONE
   CERTIFIES BROKEN KERNELS. :data:`DTDX` carries 0.5 first (so a failure there
   is visibly not a rounding failure) plus FOUR non-power-of-two values. The
   fourth is 1/3.6 and it is not decorative: it is corpus row ``test_pml_cyl``
   idx3's own Courant and the ONLY value below the m = 3 accurate branch's
   stability bound of 1/3.5. Without it the ``(3, True)`` entry of
   :data:`M_CASES` contributed ZERO rows, silently, behind a bare ``continue``.
3. **Both |m| classes and both z terminations**, because the corpus demands both:
   |m| = 1 has the axis-row increments and no near-axis zeroing, |m| >= 2 has the
   zeroing and no increments, and three of the sixteen rows terminate z PERIODIC
   where thirteen terminate METALLIC. Neither is compiled in and both are swept.
   The ``accurate_fields_near_cylorigin`` branch is swept at its own Courant
   bound (``Grid`` refuses it above ``1/(|m| + 0.5)``). EVERY declared case must
   contribute rows: the reference leg counts rows per (m, accurate) and RAISES on
   an empty one, because a table entry that runs nothing advertises coverage.
4. **ZERO-INIT rows with a THIN negative-coefficient absorber and an in-run
   signed-zero census.** MEASURED on this family (NumPy, 2026-08-13): a 2-cell
   absorber gives ``min kms_z = -1.396`` and a peak census of 85 negative-zero
   words; a 3-cell absorber -0.597 and 85; a 5-cell absorber ``min kms_z =
   +0.0415`` and a census of **0**, at which the row is VACUOUS and discriminates
   nothing. **CENSUS 0 IS A FAILURE, NOT A PASS.** The row exists because the
   ``axis_increment_accumulates`` mutation — ``curl[row0] += -inc`` where the
   array path assigns — is NEEDLE-MISSED under random seeding (0 words) and
   CAUGHT under zero-init (16 words): after the ownership mask that row is
   exactly +0.0, and ``+0.0 + x == x`` for every x except -0.0.
5. **Per-sub-step comparison, never end-of-budget.** Same measurement: the
   zero-init divergence is TRANSIENT — a -0.0 that survives one sub-step is
   laundered back by the next ``fu *= kms`` sign flip — and an end-of-run compare
   reported 0 differing words on a state that had genuinely diverged at step 1.
6. **Sources proven non-vacuous against a source-free control.** The sub-step
   legs here are source-free by construction (they compare one curl call), so the
   non-vacuity obligation lands on the SEEDS: every seeded row asserts that the
   two engines started identical and that the reference actually moved the state
   (``moved_words > 0``), and every zero-init row asserts the census. A source
   seam belongs to the composition probe, not to this file.
   AND AT THE SEEDING LEVEL, not only the row level: each mutation battery totals
   the words it moved PER SEEDING and reports a seeding that caught nothing
   across the whole battery as VACUOUS. That check is what a zero-init
   constitutive row failed — 256 of them compared two all-+0.0 states and passed
   against a reference whose coefficients were deliberately swapped.
7. **Armed mutations are launch-counted, and armed ON THE LAPTOP FIRST.** Each
   device row records the launch count of the mutated kernel, that its JIT cache
   key differs from the shipped kernel's, and — when PTX is readable — that its
   PTX equals NO shipped specialization's: Triton's cache can serve a STALE
   binary to a renamed mutant, which silently disarms the leg. ``needle_arming``
   applies every transform in every battery on the laptop, so a rename surfaces
   as DISARMED for the price of a NumPy run rather than a device run.
8. **Predicted nulls are recorded WITH reasons**, plus launch, cache-key and PTX
   evidence, never omitted. This family predicts SIX: the r near-ghost trio
   (``axis_ghost_sign``, ``axis_ghost_phase_dropped``, ``axis_ghost_restored``),
   the subtract-spelling identity (``imr_negation_as_subtract``), and three
   choices measured BYTE-INVISIBLE through this kernel's fold —
   ``imr_planewise_zero_cross_terms`` (with the minuend census as its evidence),
   ``invariant_axis_difference_elided`` and the B side of
   ``axis_increment_operands_swapped``. The last three are held by source-text
   assertions in ``meep_gpu/test_triton_cylindrical_complex.py``, which is the
   layer that can hold a byte-invisible choice; claiming a byte gate holds one is
   the artifact-overclaim inversion the BFAST tranche names.
9. **The step budget is STATED in the artifact.** No budget may be claimed until
   the device leg runs.
10. **``enable_fp_fusion`` is NOT byte-uniform across tranches** — measured per
    tranche: complex 68/68 fusion-on rows identical, special_kz 24/96,
    nonlinear 0/108, offdiag 0/28, bfast 0/52. This family states which
    configuration it certifies under (``ENABLE_FP_FUSION`` as shipped, i.e.
    guard ON) and reports the fusion-on control's identical count as DATA rather
    than asserting a value it has not measured.

No flux or DFT case appears in this family: the sub-steps here are curl and
constitutive only, so the sample-count discipline has nothing to bind. A future
composition probe that adds a monitor MUST assert its sample count.

SUBNORMAL POLICY. This gate runs UNDER THE STRIPPED IEEE-KEEP POLICY, the ship
configuration the five certified families were cut under
(``ieee_keep_ftz_stripped``), installed via ``install_ftz_strip`` before
``guard_kernel_compilation`` so the guard wraps the strip and both apply. A probe
artifact handed in with ``--probe-artifact`` must certify that same policy and is
refused otherwise.

Usage (the GPU host, ONE clear device; the cache dir must carry the policy token).
The device legs REQUIRE a measured expansion probe and refuse without one::

    CUDA_VISIBLE_DEVICES=<free> \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u gate_triton_cylindrical_complex.py \\
        --probe-artifact results/triton_complex_<date>/probe.json \\
        --out results/triton_cylindrical_complex_<date>/gate.json

Laptop (no CUDA, no Triton)::

    python -u gate_triton_cylindrical_complex.py --self-check \\
        --out /tmp/cylcomplex_selfcheck.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import platform
import re
import sys
import tempfile
import textwrap
import time
import types
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
except ImportError:      # the laptop leg; every device leg skips CLEANLY and says so
    cp = None

try:
    import triton  # noqa: F401

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# Shared machinery from the CERTIFIED complex gate (job 2330/2343) — imported,
# never re-implemented: the subnormal policy seam, the launch counter and the
# provenance writer are the same objects the five certified families were cut
# with, so this gate's verdict is about the cylindrical additions and not about
# a second copy of the harness.
import gate_triton_complex as shared  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import IYEE_SHIFTS  # noqa: E402


# ===========================================================================
# The reference — kernel-shaped, every line cited, PROVEN on the laptop
# ===========================================================================
#
# Written the way a Triton kernel computes: explicit per-axis ghosts, explicit
# masks, the split-field recurrence in the array path's operand order. That makes
# the comparison a statement about the KERNEL rather than about NumPy
# broadcasting.
#
# TRANSCRIBED FROM (meep_gpu/stepping.py, this tree):
#   B_CURL_TERMS / D_CURL_TERMS         :214-224
#   fields.IYEE_SHIFTS                  fields.py:214-219
#   step_B cylindrical branch           :295-376
#   step_D cylindrical branch           :410-457
#   cylindrical_rderiv_prefix           :1257-1305
#   _cylindrical_imr_term               :674-724
#   _cylindrical_axis_increment_B / _D  :625-645 / :524-557
#   _cylindrical_axis_zero_B / _D       :648-671 / :560-598
#   _cylindrical_axis_rows              :601-622
#   _mirror_phases (the (-1)^m slot)    :2291-2310
#   _shift_up / _shift_down             :1723-1784 / :1787-1843
#   _curl_from_operands                 :1601-1636
#   _mask_non_owned_cells               :1865-1902
#   _apply_pml_update                   :1905-1935
#   _apply_constitutive_pml             :2065-2096
#   _curl_coefficients                  :2418-2425

B_TERMS = (("Bx", "Ez", 1, "Ey", 2, "y", "z", (0, 1, 1)),
           ("By", "Ex", 2, "Ez", 0, "z", "x", (1, 0, 1)),
           ("Bz", "Ey", 0, "Ex", 1, "x", "y", (1, 1, 0)))
D_TERMS = (("Dx", "Hz", 1, "Hy", 2, "y", "z", (1, 0, 0)),
           ("Dy", "Hx", 2, "Hz", 0, "z", "x", (0, 1, 0)),
           ("Dz", "Hy", 0, "Hx", 1, "x", "y", (0, 0, 1)))

#: The i*m/r call sites — stepping :348-355 (B) / :430-437 (D).
IMR_B = (("Bx", "Ez", +1.0), ("Bz", "Ex", -1.0))
IMR_D = (("Dx", "Hz", -1.0), ("Dz", "Hx", +1.0))

PRIMARIES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
AUXILIARIES = tuple("fu_" + name for name in PRIMARIES)
SOURCES = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
CONSTITUTIVE_VOLUMES = (
    PRIMARIES + SOURCES
    + ("f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez"))
COMPARED = PRIMARIES + AUXILIARIES
NEG_ZERO = np.uint32(0x80000000)


def _face(axis: int, index: int):
    return (slice(None),) * axis + (index,)


def rderiv_prefix(xp, f_p, ir0):
    """stepping.cylindrical_rderiv_prefix (:1257-1287), the unpooled branch.

    THE SCAN IS THE ORACLE, and it is CuPy's summation order, not exact
    arithmetic and not MEEP's sequential loop: ``cupy.cumsum`` in float32 is
    deterministic but is NOT a sequential accumulation, and ``tl.cumsum`` matches
    neither (cylindrical_triton.py:32-56, measured). Complex storage changes
    nothing — the weights take ``f_p.real.dtype`` (:1279).
    """
    real_dtype = f_p.real.dtype
    counts = xp.arange(f_p.shape[0], dtype=xp.float64) + ir0
    weights = counts.reshape(-1, 1, 1).astype(real_dtype)
    divisor = (counts[1:] - 0.5).reshape(-1, 1, 1).astype(real_dtype)
    weighted = f_p * weights
    increment = xp.zeros_like(f_p)
    increment[1:] = (weighted[1:] - weighted[:-1]) / divisor
    return xp.cumsum(increment, axis=0)


def shift_up(xp, field, axis, boundary):
    """stepping._shift_up (:1723-1784). PERIODIC wraps (no phase: this family
    refuses Bloch); METALLIC and CYL_AXIS both serve an exact 0 at the FAR face
    (:1781-1783) — which is why the kernel compiles ``BCX = METALLIC`` for r."""
    shifted = xp.roll(field, -1, axis=axis)
    if boundary == "periodic":
        return shifted
    shifted[_face(axis, -1)] = 0
    return shifted


def shift_down(xp, field, axis, boundary, component, axis_phase):
    """stepping._shift_down (:1787-1843).

    The CYL_AXIS near ghost (:1830-1842) is the ``r_to_minus_r`` image: R- and
    P-direction components sign-flipped, times ``(-1)^m`` (the factor
    ``_mirror_phases`` threads into the mirror-phase slot, :2301-2309). It is
    built here and NOT in the kernel — see the two predicted-null mutations.
    """
    shifted = xp.roll(field, 1, axis=axis)
    if boundary == "periodic":
        return shifted
    if boundary == "metallic":
        shifted[_face(axis, 0)] = 0
        return shifted
    if boundary == "axis":
        sign = -1.0 if component[1] in ("x", "y") else 1.0
        sign *= axis_phase
        shifted[_face(axis, 0)] = sign * field[_face(axis, 0)]
        return shifted
    raise ValueError(boundary)


def curl_from(first, shifted_first, second, shifted_second, dtdx):
    """stepping._curl_from_operands (:1601-1619). DO NOT flatten the parens, and
    DO NOT elide the invariant-phi half: with n = 1 the rolled operand equals the
    original and the difference is an exact +0.0, which is NOT the same bits as
    omitting it when the other difference is -0.0."""
    return dtdx * ((shifted_first - first) + (second - shifted_second))


def curl_from_invariant_elided(first, shifted_first, second, shifted_second,
                               dtdx, a1, a2):
    """THE KERNEL THIS FAMILY MUST NOT BE — the "optimized" invariant-axis curl.

    Referenced by exactly one thing, the ``invariant_axis_difference_elided``
    reference mutation (grouping choice 8), and by nothing else; a laptop test
    asserts that. phi has n = 1, so the rolled operand equals the original and its
    difference is an exact +0.0 — an optimizer that drops the addend returns
    ``dtdx*x`` where the array path returns ``dtdx*(x + 0.0f)``, and the two
    differ exactly when ``x`` is -0.0.

    Keeping the choice pinned by a needle rather than by prose is the point; what
    the needle MEASURES is recorded in the artifact, including when it measures a
    null (see the mutation's entry for the reason it is one on this matrix).
    """
    left = shifted_first - first
    right = second - shifted_second
    if first.shape[a1] == 1:
        return dtdx * right
    if second.shape[a2] == 1:
        return dtdx * left
    return dtdx * (left + right)


def imr_product(factor, values):
    """The i*m/r product, as ONE named multiply so a needle can replace it.

    ``factor`` is the host-built complex64 row vector (real word +0.0) and
    ``values`` the partner volume, so this is a FULL complex product with its
    zero cross terms — ``re = (+0.0)*z_re - c_im*z_im``, NOT the plane-wise
    ``-(c_im*z_im)``. The two differ in the ±0 class only; whether that class is
    observable THROUGH the fold is measured by the
    ``imr_planewise_zero_cross_terms`` mutation and by the minuend census beside
    it, never argued.
    """
    return factor * values


def count_negative_zero_minuend(census: Dict[str, int], target: str, curl) -> None:
    """Accumulate exact ``-0.0`` words of one i*m/r MINUEND into ``census``.

    Read-only on ``curl``, so it can never change what the reference computes —
    which matters because it runs inside the leg that proves the reference equals
    the array path.
    """
    words = np.ascontiguousarray(to_host(curl)).view(np.uint32).ravel()
    census[target] = census.get(target, 0) + int(np.count_nonzero(words == NEG_ZERO))
    census["_words_scanned"] = census.get("_words_scanned", 0) + int(words.size)


def imr_factor(xp, target, sign, m, dtdx, rows, dtype):
    """stepping._cylindrical_imr_term's cached row vector (:713-718), verbatim.

    The numerator is ``(-1j) * (sign*2*m*dtdx)`` with the imaginary unit on the
    LEFT: CPython's complex product then launders the real word to +0.0 for both
    signs of the real factor. The division is float64 and the rounding to the
    storage dtype happens ONCE, at the end.
    """
    iyee_r = IYEE_SHIFTS[target][0]
    r_doubled = 2 * xp.arange(rows, dtype=xp.float64) + iyee_r
    divisor = xp.maximum(r_doubled, 1.0).reshape(-1, 1, 1)
    return (((-1j) * (sign * 2.0 * m * dtdx)) / divisor).astype(dtype)


def mask(curl, iyee, boundaries):
    """stepping._mask_non_owned_cells (:1865-1902), the metallic + is_axis clauses.
    The cylindrical axis masks for MEEP's own reason: little_owned_corner0
    deliberately excludes r = 0 ("which is updated separately", vec.hpp:1100)."""
    for axis in range(3):
        if iyee[axis] != 0:
            continue
        if boundaries[axis] in ("metallic", "axis"):
            curl[_face(axis, 0)] = 0


def recurrence(field, fu, curl, kms, sinv, kms_u, sinv_u):
    """stepping._apply_pml_update (:1905-1935), in the array path's operand order."""
    fu_previous = fu.copy()
    fu *= kms
    fu -= curl
    fu *= sinv
    field *= kms_u
    field += fu
    field -= fu_previous
    field *= sinv_u


def reference_cyl_complex_step(xp, sub_step, fields, coefficients, dtdx,
                               boundaries, m, accurate, minuend_census=None):
    """One COMPLEX cylindrical curl sub-step at |m| >= 0, in place.

    PROVEN byte-identical to ``stepping.step_B``/``step_D`` over the whole
    ``reference`` leg; the artifact states the row count and the per-row sub-step
    budget, and the module docstring quotes them from there rather than from a
    number typed by hand.

    ``minuend_census`` is an optional counter dict. When given, the MINUEND of the
    i*m/r fold — the dtdx-scaled curl, immediately before ``curl + -(factor*g)`` —
    is scanned for exact ``-0.0`` words and the counts are accumulated per target.
    That census is the EVIDENCE behind the ``imr_planewise_zero_cross_terms``
    null: the ±0 disagreement the plane-wise shortcut produces is observable only
    where the minuend is -0.0, so a census of 0 over the sweep is what makes the
    null a measurement rather than an argument.
    """
    backward = sub_step == "step_D"
    terms = D_TERMS if backward else B_TERMS
    sources = ("Hx", "Hy", "Hz") if backward else ("Ex", "Ey", "Ez")
    snap = {name: fields[name] for name in sources}
    dtype = snap[sources[0]].dtype
    rows_r = snap[sources[0]].shape[0]
    axis_phase = (-1.0) ** m                                   # stepping :2308

    # --- the radial prefix (stepping :299-333 B / :414-419 D) -----------------
    if backward:
        prefix = rderiv_prefix(xp, snap["Hy"], 0.5)
    else:
        extended = xp.empty((rows_r + 1,) + snap["Ey"].shape[1:], dtype=dtype)
        extended[:rows_r] = snap["Ey"]
        extended[rows_r] = 0            # THE ZERO WALL ROW; without it the last
        extended_prefix = rderiv_prefix(xp, extended, 0.0)   # forward difference
        #                                becomes minus the whole accumulated sum.

    # --- the |m| = 1 axis-row increments (stepping :625-645 B / :524-557 D) ---
    axis_increment = None
    if abs(m) == 1:
        if backward:
            hr = snap["Hx"]
            hr_below = shift_down(xp, hr, 2, boundaries[2], "Hx", axis_phase)
            axis_increment = ("Dy", dtdx * (hr[_face(0, 0)] - hr_below[_face(0, 0)]
                                            - 2.0 * snap["Hz"][_face(0, 0)]))
        else:
            ep = snap["Ey"]
            ep_above = shift_up(xp, ep, 2, boundaries[2])
            ez_off_axis = xp.take(snap["Ez"], 1, axis=0)   # the FIRST OFF-AXIS row
            axis_increment = ("Bx",
                              (-dtdx) * (ep[_face(0, 0)] - ep_above[_face(0, 0)])
                              - 1j * (m * dtdx) * ez_off_axis)

    imr = {t: (p, s) for t, p, s in (IMR_D if backward else IMR_B)}

    for target, g1, a1, g2, a2, dsig, dsigu, iyee in terms:
        src = dict(snap)
        if backward and target == "Dz":
            src["Hy"] = prefix                                  # stepping :425-426
        if backward:
            sf = shift_down(xp, src[g1], a1, boundaries[a1], g1, axis_phase)
            ss = shift_down(xp, src[g2], a2, boundaries[a2], g2, axis_phase)
        else:
            sf = shift_up(xp, src[g1], a1, boundaries[a1])
            ss = shift_up(xp, src[g2], a2, boundaries[a2])
        curl = curl_from(src[g1], sf, src[g2], ss, dtdx)
        if (not backward) and target == "Bz":                   # stepping :343-347
            # The WHOLE curl is replaced, and the GROUPING differs: one subtract
            # then one multiply, not the four-operand form above.
            curl = dtdx * (extended_prefix[1:] - extended_prefix[:-1])
        if m != 0 and target in imr:                            # stepping :348-355/:430-437
            partner, sign = imr[target]
            factor = imr_factor(xp, target, sign, m, dtdx, rows_r, dtype)
            if minuend_census is not None:
                count_negative_zero_minuend(minuend_census, target, curl)
            curl = curl + -(imr_product(factor, snap[partner]))  # sign convention (:724)
        mask(curl, iyee, boundaries)                            # stepping :369/:450
        if axis_increment is not None and axis_increment[0] == target:
            # REPLACES the row, AFTER the mask (:370-372 / :451-453).
            curl[_face(0, 0)] = -axis_increment[1].astype(curl.dtype)
        recurrence(fields[target], fields["fu_" + target], curl,
                   coefficients["kms_" + dsig], coefficients["sinv_" + dsig],
                   coefficients["kms_" + dsigu], coefficients["sinv_" + dsigu])

    # --- the post-update axis rules (stepping :560-598 D / :648-671 B) --------
    near = slice(0, 1) if accurate else slice(0, abs(m))        # :601-622
    if backward:
        if m == 0:
            fields["Dz"][_face(0, 0)] += (4.0 * dtdx) * fields["Hy"][_face(0, 0)]
            fields["Dy"][_face(0, 0)] = 0
        elif abs(m) == 1:
            fields["Dz"][_face(0, 0)] = 0        # the FIELD only, never fu_Dz
        else:
            for name in ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"):
                fields[name][near] = 0
    else:
        if m == 0:
            fields["Bx"][_face(0, 0)] = 0
        elif abs(m) > 1:
            for name in ("Bx", "By", "Bz", "fu_Bx", "fu_By", "fu_Bz"):
                fields[name][near] = 0
        # |m| = 1 does NOTHING here (:661-663 branches on m == 0 / abs(m) > 1).


def reference_constitutive_step(side, arrays, kps, kms, inverse_epsilon):
    """stepping._apply_constitutive_pml (:2065-2096) with NO cylindrical clause.

    This is the IDENTITY leg's reference: if ``update_H``/``update_E`` on a
    complex Dcyl grid equal THIS, then the certified
    ``complex_fields.bloch_constitutive_step`` already computes them and the
    tranche needs one kernel, not two. ``prev`` is read BEFORE the ``fw`` store
    (:2083-2085); coefficient LEFT (:2086-2087, :2093-2095); ``D * inv_eps`` with
    D LEFT (:982-984).

    Every line here is armed by :data:`CONSTITUTIVE_MUTATIONS`, which is what
    stops the leg being a comparison of two spellings nobody can break.
    """
    targets = ("Hx", "Hy", "Hz") if side == "H" else ("Ex", "Ey", "Ez")
    sources = ("Bx", "By", "Bz") if side == "H" else ("Dx", "Dy", "Dz")
    for index, target in enumerate(targets):
        field = arrays[target]
        fw = arrays["f_w_" + target]
        previous = fw.copy()                     # read BEFORE the store (:2083-2085)
        value = (arrays[sources[index]] * inverse_epsilon[index] if side == "E"
                 else arrays[sources[index]])
        fw[...] = value
        field += kps[index] * fw                 # coefficient LEFT (:2086-2087)
        field -= kms[index] * previous           # coefficient LEFT (:2093-2095)


# ===========================================================================
# The sweep product
# ===========================================================================
#
# Why each axis is in the product rather than trimmed:
#
# * dtdx — 0.5 ALONE CERTIFIES BROKEN KERNELS (exact binary scaling makes the
#   FMA/associativity discrepancies vanish). 0.5 is first so a failure at it is
#   visibly not a rounding failure; the other three are non-power-of-two.
# * m — |m| = 1 and |m| >= 2 are DIFFERENT KERNEL ARMS (increment vs zeroing) and
#   neither implies the other; both signs of m are swept because the i*m/r
#   coefficient and the (-1)^m ghost phase both flip with it, and ``abs(m)`` in
#   the numerator is a measured catch (201,312 words on the reference).
# * accurate_fields_near_cylorigin — selects ZERO_ROWS at |m| >= 2. Swept at its
#   own Courant bound, which ``Grid`` enforces (grid.py:626-641).
# * z termination — METALLIC (13 corpus rows) and PERIODIC (3). They mask
#   DIFFERENT planes and take different ghost arms.
# * shapes — nphi is ALWAYS 1 (Dcyl is 2.5-D), so the sweep varies nr and nz: a
#   non-square (an i<->k index swap cannot pass), a tall-thin and a short-wide
#   (the prefix runs along r, so the two extremes exercise a long scan and a wide
#   one), an odd pair, and one large case.
# * sub_step — B and D shift in opposite directions, read different coefficient
#   sub-lattices, mask DIFFERENT planes, use the prefix DIFFERENTLY (a replaced
#   curl versus a substituted operand), take DIFFERENT i*m/r signs and apply
#   DIFFERENT axis rules. Neither implies the other; both are always run.
# * seeding — random (the bulk) AND zero-init with a thin negative-kms absorber
#   (the signed-zero class random seeds are provably blind to).
# * guard — enable_fp_fusion False (the shipped value) and True (the control).

SHAPES: Tuple[Tuple[int, int, int], ...] = (
    (20, 1, 24),     # the laptop-validated reference shape
    (13, 1, 11),     # odd, non-square
    (64, 1, 12),     # tall and thin: a long radial scan
    (12, 1, 64),     # short and wide: many scan columns, few rows
)
# NOTE: no BENCH_SHAPE. This file carried one and no benchmark leg; a constant
# nothing reads is an advertisement, and this gate makes correctness claims only.

#: Courant numbers. 0.5 FIRST (a failure there is visibly not a rounding
#: failure), then four non-power-of-two values. The last is 1/3.6 — NOT decorative:
#: it is the Courant of corpus row ``test_pml_cyl`` idx3, the family's only
#: ``accurate_fields_near_cylorigin`` demand at m = 3, and it is the ONLY value in
#: this tuple below that row's stability bound of 1/(|m| + 0.5) = 0.2857. Without
#: it the ``(3, True)`` entry of :data:`M_CASES` contributed ZERO rows and the
#: sweep silently advertised a case it never ran.
DTDX: Tuple[float, ...] = (0.5, 0.37, 0.3141592653589793, 0.4472135954999579,
                           0.2777777777777778)

SUB_STEPS: Tuple[str, ...] = ("step_B", "step_D")

#: ``(m, accurate)``. m = 0 (the M_ZERO arm under complex storage: no coupling,
#: the m = 0 axis pair; the corpus row dipole_in_vacuum_cyl_off_axis.py), both
#: signs at |m| = 1 (the coefficient and the ghost phase flip), |m| = 2/3/5 for
#: the zeroing arm, and the accurate branch at |m| = 2, 3.
M_CASES: Tuple[Tuple[int, bool], ...] = (
    (0, False),
    (1, False), (-1, False), (2, False), (-2, False), (3, False), (5, False),
    (2, True), (3, True),
)

#: z termination: True = METALLIC (13 corpus rows), False = PERIODIC (3).
Z_METALLIC: Tuple[bool, ...] = (True, False)

#: PML depth in cells for the SEEDED rows (thick: the ordinary configuration) and
#: for the ZERO-INIT rows (thin: the only one whose deepest kms goes negative and
#: therefore the only one that reaches the signed-zero class).
SEEDED_PML_CELLS = 5
ZERO_INIT_PML_CELLS = 2

SEED = 20260813


# ===========================================================================
# Case construction
# ===========================================================================

def build_engine(xp, shape, m, courant, z_metallic, accurate, pml_cells):
    """A real complex cylindrical Grid/Fields/PML at a chosen shape.

    ``resolution = 1`` with a cell size equal to the cell count lands on the shape
    exactly (``meep_cell_count`` is ``int(size*a + 0.5)``). z is declared
    explicitly because ``Grid``'s own default is periodic and both terminations
    are in the corpus.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    grid = Grid(resolution=1.0,
                cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=int(m),
                boundaries=({"z": "metallic"} if z_metallic else None),
                accurate_fields_near_cylorigin=bool(accurate),
                courant=float(courant), xp=xp)
    if tuple(grid.shape) != tuple(shape):
        raise RuntimeError(f"Grid built {tuple(grid.shape)} for {tuple(shape)}")
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    thickness = {"x": (0, pml_cells), "z": (pml_cells if z_metallic else 0)}
    return grid, fields, PML(grid=grid, thickness=thickness)


def seed_state(xp, fields, shape, zero_init: bool, offset: int = 0) -> int:
    """Fill every compared volume. Returns the number of nonzero words written.

    NONZERO EVERYWHERE on the seeded rows, auxiliaries included: a zero ``fu``
    makes ``fu*kms`` exactly zero on the first sub-step whatever ``kms`` is, so a
    mis-indexed coefficient would only show from step two.
    """
    rng = np.random.default_rng(SEED + offset)
    written = 0
    for name in COMPARED + SOURCES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if zero_init:
            array[...] = 0
        else:
            host = (rng.standard_normal(shape).astype(np.float32)
                    + 1j * rng.standard_normal(shape).astype(np.float32))
            array[...] = xp.asarray(host.astype(np.complex64))
            written += int(np.count_nonzero(host))
    return written


def coefficient_dict(pml, half_integer: bool) -> Dict[str, Any]:
    suffix = "_h" if half_integer else ""
    return {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
            for axis in "xyz" for stem in ("kms", "sinv")}


def flat_coefficients(tables: Dict[str, Any]) -> Dict[str, Any]:
    return {name: array.reshape(-1) for name, array in tables.items()}


def to_host(array) -> np.ndarray:
    getter = getattr(array, "get", None)
    return np.asarray(getter() if callable(getter) else array)


def first_difference(left, right) -> Optional[Dict[str, Any]]:
    """uint32 compare of two complex64 volumes, never allclose."""
    a = np.ascontiguousarray(to_host(left)).view(np.uint32).ravel()
    b = np.ascontiguousarray(to_host(right)).view(np.uint32).ravel()
    bad = np.flatnonzero(a != b)
    if bad.size == 0:
        return None
    index = int(bad[0])
    return {"differing_words": int(bad.size), "first_word": index,
            "left": hex(int(a[index])), "right": hex(int(b[index]))}


def negative_zero_census(fields) -> Dict[str, Any]:
    """Words that are exactly ``0x80000000`` across the compared volumes.

    The zero-init rows' non-vacuity level: a census of 0 means the case cannot
    discriminate the signed-zero class and MUST report itself VACUOUS. Measured
    on this family: 85 at a 2- or 3-cell absorber, **0 at a 5-cell one**.
    """
    total = 0
    where: Dict[str, int] = {}
    for name in COMPARED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        words = np.ascontiguousarray(to_host(array)).view(np.uint32).ravel()
        count = int(np.count_nonzero(words == NEG_ZERO))
        if count:
            where[name] = count
        total += count
    return {"neg_zero_words": total, "by_volume": where}


def nan_census(volumes: Dict[str, Any]) -> Dict[str, int]:
    """Non-finite stored words across the compared volumes.

    PLATFORM FACT (f): a NaN's sign and payload are IEEE-unspecified, so a raw
    uint32 compare over one is not a measurement — two engines that both
    "produced a NaN" can differ in every payload bit and agree physically, or
    agree bit for bit and mean nothing. Every row reports this; on the clean
    routes a nonzero count FAILS the row rather than being compared through.
    On a mutated route it is RECORDED, because a mutant that manufactures a
    non-finite word is a real outcome and aborting the battery on it would
    discard the rows after it.
    """
    nans = 0
    infinities = 0
    for array in volumes.values():
        if array is None:
            continue
        host = to_host(array)
        nans += int(np.count_nonzero(np.isnan(host)))
        infinities += int(np.count_nonzero(np.isinf(host)))
    return {"nan_words": nans, "inf_words": infinities}


def minimum_bound_kms(coefficients: Dict[str, Any]) -> float:
    """The smallest ``kms`` word over the sub-lattice a row actually binds.

    THE PRODUCER OF A STORED ``-0.0`` IN A ZERO-INIT ROW, and the only one: the
    recurrence is ``fu *= kms; fu -= curl; ...``, so from an all-``+0.0`` state a
    negative-zero word can only appear where some ``kms`` is NEGATIVE. Whether
    one is negative is a property of the CONFIGURATION — absorber thickness, and
    which of the two Yee sub-lattices the sub-step reads — not of the harness.
    """
    return min(float(np.min(to_host(array)))
               for name, array in coefficients.items()
               if name.startswith("kms"))


def signed_zero_reach(peak: int, min_kms: float) -> str:
    """Can this zero-init row reach the ±0 class at all, and if not, why not?

    PURE, so the laptop drives every branch of a decision the device leg makes.

    MEASURED (NumPy, shape (20, 1, 24), the gate's own ``build_engine`` at
    ``ZERO_INIT_PML_CELLS`` = 2, Courant 0.5, 8 sub-steps), min bound ``kms``
    against the peak census, for every m the leg carries:

        z METALLIC step_B   min -0.821   peak 66      live
        z METALLIC step_D   min -2.238   peak 19      live
        z PERIODIC step_B   min -0.821   peak 24      live
        z PERIODIC step_D   min +0.190   peak  0      UNREACHABLE

    The last line is not a broken row. A z-periodic grid carries NO z absorber,
    and on the INTEGER sub-lattice the r-high absorber's ``kms`` is positive
    everywhere — so no negative coefficient exists, no ``-0.0`` can be
    manufactured, and the ±0 class is out of reach BY CONSTRUCTION. Calling that
    VACUOUS would fail a quarter of the sweep for a property of the physics; not
    recording it would let a genuinely broken seeding hide among them. So the
    row is classified, the number that decides it is recorded, and the leg
    carries a floor of its own (:func:`assert_leg_reaches_signed_zero`): at
    least one zero-init row must be LIVE, or the leg's zero-init half measured
    nothing.
    """
    if peak > 0:
        return "live"
    if min_kms >= 0.0:
        return "unreachable_no_negative_coefficient"
    return "VACUOUS"


def assert_zero_init_non_vacuous(name: str, census_trail: Sequence[Dict[str, Any]],
                                 min_kms: float) -> str:
    """A zero-init row with a silent census AND a negative coefficient present
    has measured nothing; one with no negative coefficient could not have."""
    peak = max((row["neg_zero_words"] for row in census_trail), default=0)
    reach = signed_zero_reach(peak, min_kms)
    if reach == "VACUOUS":
        raise AssertionError(
            f"{name}: VACUOUS — the reference state never carried a negative-zero "
            f"word although a NEGATIVE bound coefficient is present (min kms = "
            f"{min_kms:.6g}), so the row could have reached the class and did "
            f"not. Use a THIN absorber (measured: 2 or 3 cells give min kms_z = "
            f"-1.396 / -0.597 and a peak census of 85; 5 cells give +0.0415 and "
            f"a census of 0).")
    return reach


def assert_leg_reaches_signed_zero(leg: str, rows: Sequence[Dict[str, Any]]) -> None:
    """THE LEG-LEVEL FLOOR the row-level one cannot carry.

    Individual rows may be out of the ±0 class by construction (see
    :func:`signed_zero_reach`), but a LEG whose every zero-init row is out of it
    has swept the class without measuring it once — and the needle that class
    exists for (``axis_increment_accumulates``) would report NEEDLE-MISSED as a
    property of the sweep rather than of the kernel.
    """
    zero_init = [row for row in rows if row.get("zero_init")]
    if not zero_init:
        return
    if not any(row.get("signed_zero_reach") == "live" for row in zero_init):
        raise AssertionError(
            f"{leg}: VACUOUS zero-init half — none of its {len(zero_init)} "
            f"zero-init rows reached the ±0 class, so the seeding measured "
            f"nothing about the assign-versus-accumulate defect it exists for")


# ===========================================================================
# The legs
# ===========================================================================

def one_curl_case(xp, shape, m, accurate, courant, z_metallic, sub_step,
                  zero_init: bool, guard: Optional[bool], steps: int,
                  expansion: int, kernel=None, route: str = "synthetic",
                  probe: Any = None, host_mutation=None) -> Dict[str, Any]:
    """One (shape, m, Courant, z, sub_step, seeding) row, compared PER SUB-STEP.

    PER SUB-STEP, not at the end of the budget: measured on this family, a
    zero-init divergence is transient and an end-of-run compare reported 0
    differing words on a state that had genuinely diverged at step 1.

    THE ROW CARRIES ITS OWN NON-VACUITY, and the caller must read it: ``covered``,
    ``launches`` and ``moved_words`` are recorded and :func:`verdict_for_curl_row`
    turns them into a status. A row with ``launches == 0`` is NOT a pass — the
    first cut of the engine route returned ``plan = None`` for every case (no
    probe artifact was passed) and recorded ``first_divergence: None``, which
    reads as identical and measured nothing at all.

    ``probe`` is the measured expansion artifact the ENGINE route needs; the
    synthetic route binds ``expansion`` directly because it is the harness route
    and builds the configuration deliberately. ``host_mutation`` is a callable
    ``(cyl_module, context) -> (patches, overrides)`` for the HOST battery.
    """
    from meep_gpu.triton_kernels import cylindrical_complex as cyl  # noqa: PLC0415

    pml_cells = ZERO_INIT_PML_CELLS if zero_init else SEEDED_PML_CELLS
    grid_a, fields_a, pml_a = build_engine(xp, shape, m, courant, z_metallic,
                                           accurate, pml_cells)
    grid_b, fields_b, pml_b = build_engine(xp, shape, m, courant, z_metallic,
                                           accurate, pml_cells)
    moved = seed_state(xp, fields_a, shape, zero_init)
    seed_state(xp, fields_b, shape, zero_init)
    for name in COMPARED:
        if first_difference(getattr(fields_a, name), getattr(fields_b, name)):
            raise AssertionError("the two engines did not start identical")

    dtdx = grid_a.dt / grid_a.dx
    boundaries = stepping._boundary_kinds(grid_a, pml_a)
    coefficients = coefficient_dict(pml_b, half_integer=(sub_step == "step_B"))
    reference_fields = {name: getattr(fields_b, name)
                        for name in COMPARED + SOURCES}

    context = {"sub_step": sub_step, "half_integer": sub_step == "step_B",
               "bcz": 1 if boundaries[2] == "metallic" else 0,
               "expansion": expansion, "m": int(m), "accurate": bool(accurate),
               "dtdx": dtdx}
    patches: List[Tuple[Any, str, Any]] = []
    if host_mutation is not None:
        patches, overrides = host_mutation(cyl, context)
        context.update(overrides)

    restore: List[Tuple[Any, str, Any]] = []
    for obj, attribute, replacement in patches:
        restore.append((obj, attribute, getattr(obj, attribute)))
        setattr(obj, attribute, replacement)
    try:
        if route == "engine":
            # probe= is MANDATORY here. Without it the shipped predicate refuses
            # (no expansion artifact), the plan is None, and the row would report
            # "identical" having launched nothing.
            plan = cyl.plan_cylindrical_complex_curl(fields_a, pml_a, sub_step,
                                                     probe=probe)
            covered = plan is not None
        else:
            arrays = {name: getattr(fields_a, name) for name in COMPARED + SOURCES}
            plan = cyl.plan_cylindrical_complex_curl_from_arrays(
                sub_step, arrays, flat_coefficients(coefficient_dict(
                    pml_a, half_integer=context["half_integer"])),
                context["dtdx"], context["m"], context["accurate"],
                context["bcz"], context["expansion"], xp,
                kernel=kernel)
            covered = True

        divergence = None
        census_trail: List[Dict[str, Any]] = []
        launches = 0
        for step in range(steps):
            if plan is not None:
                plan.run(guard=guard)
                launches += 1
            reference_cyl_complex_step(xp, sub_step, reference_fields,
                                       coefficients, dtdx, boundaries, int(m),
                                       bool(accurate))
            if zero_init:
                census_trail.append(negative_zero_census(fields_b))
            if divergence is None and plan is not None:
                for name in COMPARED:
                    found = first_difference(getattr(fields_a, name),
                                             getattr(fields_b, name))
                    if found is not None:
                        divergence = {"step": step, "volume": name, **found}
                        break
    finally:
        for obj, attribute, original in restore:
            setattr(obj, attribute, original)

    census = nan_census({name: reference_fields[name] for name in COMPARED})
    min_kms = minimum_bound_kms(coefficients)
    row = {"route": route, "shape": list(shape), "m": int(m),
           "accurate": bool(accurate), "courant": float(courant),
           "z": "metallic" if z_metallic else "periodic", "sub_step": sub_step,
           "zero_init": bool(zero_init), "guard": guard, "steps": steps,
           "covered": covered, "launches": launches,
           "moved_words": moved, "first_divergence": divergence,
           "nan_census": census, "min_bound_kms": min_kms,
           "census_peak": (max((c["neg_zero_words"] for c in census_trail),
                               default=0) if zero_init else None)}
    if kernel is None and host_mutation is None and (census["nan_words"]
                                                     or census["inf_words"]):
        raise AssertionError(
            f"m={m} z={'M' if z_metallic else 'P'} {sub_step}: the reference "
            f"state carries {census} non-finite words; platform fact (f) "
            f"forbids reading a raw word through a NaN, so this row's uint32 "
            f"compare is not a measurement")
    if zero_init:
        row["signed_zero_reach"] = assert_zero_init_non_vacuous(
            f"m={m} z={'M' if z_metallic else 'P'} {sub_step}", census_trail,
            min_kms)
    elif moved == 0:
        raise AssertionError("a seeded row wrote nothing; the case is vacuous")
    return row


def verdict_for_curl_row(row: Dict[str, Any],
                         expect: str = "identical") -> str:
    """Turn one curl row into a status. PURE — the laptop tests drive it.

    The classifier is separated from the case so the DEVICE leg's decision logic
    is exercised at the merge bar even though its launches are not. The statuses
    it can return are the ones the discipline names:

    * ``NOT-COVERED``   — the predicate refused; nothing was measured;
    * ``NO-LAUNCH``     — the plan never ran, so "identical" means "untouched";
    * ``SHORT-LAUNCH``  — fewer launches than steps: a partial comparison;
    * ``VACUOUS``       — the row moved no word, so it cannot fail;
    * ``IDENTICAL`` / ``DIVERGED`` — the real answers;
    * ``NEEDLE-MISSED`` — a row that was supposed to diverge and did not.
    """
    if not row.get("covered"):
        return "NOT-COVERED"
    if not row.get("launches"):
        return "NO-LAUNCH"
    if row.get("steps") and row["launches"] != row["steps"]:
        return "SHORT-LAUNCH"
    if row.get("zero_init"):
        # A row out of the ±0 class BY CONSTRUCTION is not vacuous: it measures
        # everything a seeded row measures and simply cannot reach one class.
        # The leg-level floor (assert_leg_reaches_signed_zero) is what keeps a
        # sweep of nothing but such rows from passing.
        if not row.get("census_peak") and \
                row.get("signed_zero_reach") != "unreachable_no_negative_coefficient":
            return "VACUOUS"
    elif not row.get("moved_words"):
        return "VACUOUS"
    diverged = row.get("first_divergence") is not None
    if expect == "diverge":
        return "DIVERGED" if diverged else "NEEDLE-MISSED"
    return "DIVERGED" if diverged else "IDENTICAL"




def seed_constitutive(xp, fields, shape, seeding: str) -> int:
    """Fill every constitutive volume. Returns the nonzero-or-signed-zero words.

    TWO SEEDINGS, and the second one exists because the first CANNOT reach the
    ±0 class here. ``random`` is the ordinary one. ``signed_zero`` writes an
    explicit lattice of ``-0.0`` / ``+0.0`` / normal words, because — unlike the
    curl sub-step, where a thin absorber's negative ``kms`` times a quiet ``+0.0``
    manufactures ``-0.0`` in run — the constitutive sub-step has NO such producer:
    it is ``fw = value; field += kps*fw; field -= kms*prev`` and from an all-zero
    state every word stays ``+0.0`` forever (``+0.0 + (-0.0)`` is ``+0.0``, and so
    is ``+0.0 - (±0.0)``). A ZERO-INIT constitutive row is therefore VACUOUS BY
    CONSTRUCTION — it has no census to assert and no mutation can move it — which
    is why this leg carries a seeded ±0 lattice instead and asserts the census on
    it. Measured: with zero-init, 0 nonzero words after 6 steps and a deliberately
    broken reference still reports IDENTICAL.
    """
    rng = np.random.default_rng(SEED)
    written = 0
    for index, name in enumerate(CONSTITUTIVE_VOLUMES):
        array = getattr(fields, name, None)
        if array is None:
            continue
        if seeding == "random":
            host = (rng.standard_normal(shape).astype(np.float32)
                    + 1j * rng.standard_normal(shape).astype(np.float32))
            host = host.astype(np.complex64)
            written += int(np.count_nonzero(host))
        elif seeding == "signed_zero":
            # A deterministic four-phase lattice per volume: -0.0, +0.0, a normal
            # and its negation. The -0.0 lanes are what carry the class; the
            # normal lanes are what make a swapped coefficient visible at all.
            #
            # THE PHASE IS DRAWN PER VOLUME, not from `index % 4`: with a linear
            # index the lattice repeats every four volumes, and CONSTITUTIVE_
            # VOLUMES puts Bx at 0 and f_w_Hx at 12 — the same phase — so
            # `previous` and the newly stored `fw` were bit-identical and the
            # read-order needle measured nothing on this seeding.
            phase = np.random.default_rng(SEED + 7919 * (index + 1)).integers(
                0, 4, size=shape)
            real = np.where(phase == 0, np.float32(-0.0),
                            np.where(phase == 1, np.float32(0.0),
                                     np.where(phase == 2, np.float32(0.75),
                                              np.float32(-1.25)))).astype(np.float32)
            imag = np.where(phase == 0, np.float32(-0.0),
                            np.where(phase == 3, np.float32(0.0),
                                     np.where(phase == 1, np.float32(-0.5),
                                              np.float32(1.5)))).astype(np.float32)
            host = np.empty(shape, dtype=np.complex64)
            host.real = real
            host.imag = imag
            words = host.view(np.uint32).ravel()
            written += int(np.count_nonzero(words))   # -0.0 counts: its word is nonzero
        else:
            raise ValueError(f"unknown constitutive seeding {seeding!r}")
        array[...] = xp.asarray(host)
    return written


def constitutive_census(fields) -> Dict[str, Any]:
    """Exact ``-0.0`` words across the constitutive volumes (the ±0 level)."""
    total = 0
    where: Dict[str, int] = {}
    for name in CONSTITUTIVE_VOLUMES:
        array = getattr(fields, name, None)
        if array is None:
            continue
        words = np.ascontiguousarray(to_host(array)).view(np.uint32).ravel()
        count = int(np.count_nonzero(words == NEG_ZERO))
        if count:
            where[name] = count
        total += count
    return {"neg_zero_words": total, "by_volume": where}


CONSTITUTIVE_STEPS = 6


def one_constitutive_case(xp, shape, m, courant, z_metallic, side,
                          seeding: str) -> Dict[str, Any]:
    """The IDENTITY leg: update_H/update_E on complex Dcyl == the plain complex
    elementwise dsigw sub-step, compared PER SUB-STEP by uint32 word.

    NON-VACUITY IS ASSERTED IN RUN, at three levels, because the first cut of
    this leg had 256 rows that could not fail: the reference must MOVE the state
    (``moved_words > 0``), the two engines must start identical, and a
    ``signed_zero`` row must carry a nonzero ±0 census. The seeding-level proof
    that the rows discriminate is :func:`run_constitutive_mutations`, which
    breaks this reference on purpose and requires every seeding to catch it.
    """
    pml_cells = SEEDED_PML_CELLS if seeding == "random" else ZERO_INIT_PML_CELLS
    _grid_a, fields_a, pml_a = build_engine(xp, shape, m, courant, z_metallic,
                                            False, pml_cells)
    _grid_b, fields_b, pml_b = build_engine(xp, shape, m, courant, z_metallic,
                                            False, pml_cells)
    seeded = seed_constitutive(xp, fields_a, shape, seeding)
    seed_constitutive(xp, fields_b, shape, seeding)
    for name in CONSTITUTIVE_VOLUMES:
        if getattr(fields_a, name, None) is None:
            continue
        if first_difference(getattr(fields_a, name), getattr(fields_b, name)):
            raise AssertionError("the two engines did not start identical")
    before = {name: np.array(to_host(getattr(fields_b, name)), copy=True)
              for name in CONSTITUTIVE_VOLUMES
              if getattr(fields_b, name, None) is not None}

    half = side == "E"
    suffix = "_h" if half else ""
    kps = [getattr(pml_b, f"kps_{axis}{suffix}") for axis in "xyz"]
    kms = [getattr(pml_b, f"kms_{axis}{suffix}") for axis in "xyz"]
    inverse = [fields_b.inverse_epsilon_for(name) for name in ("Ex", "Ey", "Ez")]
    arrays = {name: getattr(fields_b, name) for name in CONSTITUTIVE_VOLUMES}

    divergence = None
    census_peak = 0
    for step in range(CONSTITUTIVE_STEPS):
        (stepping.update_H if side == "H" else stepping.update_E)(fields_a, pml_a)
        reference_constitutive_step(side, arrays, kps, kms, inverse)
        census_peak = max(census_peak, constitutive_census(fields_b)["neg_zero_words"])
        if divergence is None:
            for name in CONSTITUTIVE_VOLUMES:
                if getattr(fields_a, name, None) is None:
                    continue
                found = first_difference(getattr(fields_a, name),
                                         getattr(fields_b, name))
                if found is not None:
                    divergence = {"step": step, "volume": name, **found}
                    break

    moved = 0
    for name, snapshot in before.items():
        after = np.ascontiguousarray(to_host(getattr(fields_b, name))).view(np.uint32)
        moved += int(np.count_nonzero(
            np.ascontiguousarray(snapshot).view(np.uint32).ravel() != after.ravel()))
    row = {"shape": list(shape), "m": int(m), "courant": float(courant),
           "z": "metallic" if z_metallic else "periodic", "side": side,
           "seeding": seeding, "steps": CONSTITUTIVE_STEPS,
           "seeded_words": seeded, "moved_words": moved,
           "census_peak": census_peak, "first_divergence": divergence}
    if seeded == 0:
        raise AssertionError(f"constitutive row wrote nothing: {row}")
    if moved == 0:
        raise AssertionError(
            f"VACUOUS: the constitutive reference moved no word in "
            f"{CONSTITUTIVE_STEPS} steps, so the row compares two untouched "
            f"states and cannot fail: {row}")
    if seeding == "signed_zero" and census_peak == 0:
        raise AssertionError(
            f"VACUOUS: the signed_zero row never carried a -0.0 word, so it "
            f"cannot discriminate the ±0 class it exists for: {row}")
    return row


# ===========================================================================
# The armed mutations
# ===========================================================================
#
# Every mutation is applied to the SHIPPED kernel's SOURCE, recompiled, and
# launched through the same plan class. Three failure modes are distinguished and
# ALL THREE are failures:
#
#   DISARMED      — the pattern did not match (a rename broke the mutation);
#   STALE-BINARY  — the mutant's PTX equals a shipped specialization's, so
#                   Triton's cache served the wrong binary and nothing was tested;
#   NEEDLE-MISSED — the mutation compiled, launched and changed no byte.
#
# Predicted nulls are recorded WITH their reason and with the same launch and PTX
# evidence, never omitted.

_MUT_IMR = re.compile(
    r"m0_re, m0_im = _mul_general_coefficient_left\(q0_re, q0_im, c_re, c_im, EXPANSION\)")
_MUT_IMR2 = re.compile(
    r"m2_re, m2_im = _mul_general_coefficient_left\(q2_re, q2_im, a_re, a_im, EXPANSION\)")
_MUT_IMR_BLOCK = re.compile(
    r"        m0_re, m0_im = _mul_general_coefficient_left\(q0_re, q0_im, c_re, c_im, EXPANSION\)\n"
    r"        curl0_re = curl0_re - m0_re\n"
    r"        curl0_im = curl0_im - m0_im\n"
    r"        m2_re, m2_im = _mul_general_coefficient_left\(q2_re, q2_im, a_re, a_im, EXPANSION\)\n"
    r"        curl2_re = curl2_re - m2_re\n"
    r"        curl2_im = curl2_im - m2_im\n")
_MUT_M0_BX_ZERO = re.compile(
    r"        else:\n"
    r"            v0_re = tl\.where\(at_r, 0\.0, v0_re\)\n"
    r"            v0_im = tl\.where\(at_r, 0\.0, v0_im\)\n")
_MUT_M0_DZ_ADD = re.compile(
    r"            v2_re = tl\.where\(at_r, v2_re \+ hp_re, v2_re\)\n"
    r"            v2_im = tl\.where\(at_r, v2_im \+ hp_im, v2_im\)\n")
_MUT_M0_DY_ZERO = re.compile(
    r"            v1_re = tl\.where\(at_r, 0\.0, v1_re\)\n"
    r"            v1_im = tl\.where\(at_r, 0\.0, v1_im\)\n")
_MUT_M0_FOUR_DTDX = re.compile(
    r"_mul_coefficient_left\(four_dtdx, b_re, b_im, EXPANSION\)")
_MUT_AXIS_ASSIGN_B = re.compile(
    r"curl0_re = tl\.where\(at_r, inc_re \* -1\.0, curl0_re\)")
_MUT_AXIS_ASSIGN_D = re.compile(
    r"curl1_re = tl\.where\(at_r, inc_re \* -1\.0, curl1_re\)")
_MUT_BZ_PREFIX = re.compile(
    r"curl2_re, curl2_im = _mul_coefficient_left\(dtdx, pu_re - pd_re, pu_im - pd_im,\n"
    r"                                                   EXPANSION\)")
_MUT_ZERO_AUX = re.compile(r"        n0_re = tl\.where\(near, 0\.0, n0_re\)")


def _imr_dropped(source: str) -> Tuple[str, int]:
    """Drop the i*m/r term on target 0. Reference catch: 702,912 words."""
    return _MUT_IMR.subn("m0_re, m0_im = 0.0 * c_re, 0.0 * c_im", source)


def _imr_planewise(source: str) -> Tuple[str, int]:
    """Fold the ZERO CROSS TERMS out of the i*m/r product on BOTH call sites.

    A PREDICTED NULL, and the strongest spelling of one: the product's own words
    genuinely differ (``(+0.0*z_re) - X`` is +0.0 where ``X * -1.0`` is -0.0 at
    X = +0.0), so this is not a no-op mutation — it is a real defect that this
    kernel's FOLD cannot carry to a stored word. ``curl - m`` differs between
    m = +0.0 and m = -0.0 only when the MINUEND is -0.0, and no i*m/r minuend
    here can be: target 0's curl leads with the phi self-difference (+0.0 on a
    one-cell axis, and +0.0 + y is +0.0 even at y = -0.0), and target 2's is
    either a prefix cumsum (which starts at +0.0 and can never produce -0.0) or
    the same +0.0-led sum. The reference battery carries the same name and the
    minuend census in its record is the measurement.

    Mutating BOTH sites — the first cut mutated only target 0 — so that a null
    here is a statement about the family rather than about one register.
    """
    out, n0 = _MUT_IMR.subn(
        "m0_re, m0_im = (q0_im * c_im) * -1.0, q0_im * c_re", source)
    out, n2 = _MUT_IMR2.subn(
        "m2_re, m2_im = (q2_im * a_im) * -1.0, q2_im * a_re", out)
    return out, n0 + n2


def _imr_after_mask(source: str) -> Tuple[str, int]:
    """Apply the i*m/r fold AFTER the ownership mask (grouping choice 6).

    The array path folds it in BEFORE the mask, so the mask zeroes the folded
    curl; applying it after leaves the term alive on rows the array path owns as
    exactly 0. Reference counterpart of the same name.
    """
    # The block sits under `if M_CLASS != M_ZERO:`; the loads stay in place (the
    # guard keeps a body) and the products move past the mask under the same guard.
    out, n1 = _MUT_IMR_BLOCK.subn("        pass\n", source)
    if n1 == 0:
        return source, 0
    out, n2 = re.subn(
        r"(    # --- the \|m\| = 1 axis-row increment)",
        "    if M_CLASS != M_ZERO:\n"
        "        m0_re, m0_im = _mul_general_coefficient_left(q0_re, q0_im, c_re, c_im,\n"
        "                                                    EXPANSION)\n"
        "        curl0_re = curl0_re - m0_re\n"
        "        curl0_im = curl0_im - m0_im\n"
        "        m2_re, m2_im = _mul_general_coefficient_left(q2_re, q2_im, a_re, a_im,\n"
        "                                                    EXPANSION)\n"
        "        curl2_re = curl2_re - m2_re\n"
        "        curl2_im = curl2_im - m2_im\n\n"
        r"\1", out)
    return out, n1 + n2


def _axis_increment_before_mask(source: str) -> Tuple[str, int]:
    """The |m| = 1 axis increment wiped by the ownership mask (choice 6).

    Spelled as the mask RE-APPLIED after the increment, which plants exactly the
    defect the reference's reordering does (the incremented row goes back to
    zero) without moving two blocks past each other in a jitted body.
    """
    return re.subn(
        r"(    # --- split-field recurrence)",
        "    if M_CLASS == M_ONE:\n"
        "        if BACKWARD:\n"
        "            curl1_re = tl.where(at_r, 0.0, curl1_re)\n"
        "            curl1_im = tl.where(at_r, 0.0, curl1_im)\n"
        "        else:\n"
        "            curl0_re = tl.where(at_r, 0.0, curl0_re)\n"
        "            curl0_im = tl.where(at_r, 0.0, curl0_im)\n\n"
        r"\1", source)


def _invariant_axis_elided(source: str) -> Tuple[str, int]:
    """Drop the phi self-difference from every curl that carries one (choice 8).

    phi has n = 1, so the rolled operand IS the original and the addend is an
    exact +0.0 — the "optimization" a reader would make. Predicted null on this
    matrix with the reason recorded; the choice is held by source text.
    """
    hits = 0
    for old, new in (
            ("    t0_re = ((c_p_re - c_re) + (b_re - b_z_re))",
             "    t0_re = (b_re - b_z_re)"),
            ("    t0_im = ((c_p_im - c_im) + (b_im - b_z_im))",
             "    t0_im = (b_im - b_z_im)"),
            ("    t2_re = ((b_r_re - b_re) + (a_re - a_p_re))",
             "    t2_re = (b_r_re - b_re)"),
            ("    t2_im = ((b_r_im - b_im) + (a_im - a_p_im))",
             "    t2_im = (b_r_im - b_im)"),
            ("        t2_re = ((p_down_re - p_here_re) + (a_re - a_p_re))",
             "        t2_re = (p_down_re - p_here_re)"),
            ("        t2_im = ((p_down_im - p_here_im) + (a_im - a_p_im))",
             "        t2_im = (p_down_im - p_here_im)")):
        if old in source:
            source = source.replace(old, new, 1)
            hits += 1
    return source, hits


def _replace_all(source: str, pairs) -> Tuple[str, int]:
    hits = 0
    for old, new in pairs:
        if old in source:
            source = source.replace(old, new, 1)
            hits += 1
    return source, hits


def _axis_increment_operands_swapped(source: str) -> Tuple[str, int]:
    """B side: ``-dtdx * (A - B)`` written as ``dtdx * (B - A)`` (choice 5).

    Identical for every finite pair and different exactly at A == B, where the
    array path's spelling yields -0.0 and the swap yields +0.0 — the state a
    zero-init row is entirely made of. MEASURED NULL on the reference: Bx's
    split-field dsig axis is phi, whose kms is 1.0, so the ``fu*kms`` minuend is
    +0.0 and the difference is laundered before it can be stored. The D-side twin
    below is the one that reaches a stored word.
    """
    return _replace_all(source, (
        ("            d_re = b_re - b_z_re", "            d_re = b_z_re - b_re"),
        ("            d_im = b_im - b_z_im", "            d_im = b_z_im - b_im"),
        ("p_re, p_im = _mul_coefficient_left(minus_dtdx, d_re, d_im, EXPANSION)",
         "p_re, p_im = _mul_coefficient_left(dtdx, d_re, d_im, EXPANSION)")))


def _axis_increment_d_operands_swapped(source: str) -> Tuple[str, int]:
    """D side: the same swap on ``dtdx * (Hr - Hr_below - 2*Hz)``.

    Dy's dsig axis is z, which carries the thin absorber's NEGATIVE kms, so its
    ``fu*kms`` minuend is -0.0 under zero-init and the ±0 difference survives to
    a stored word.
    """
    return _replace_all(source, (
        ("            s_re = (a_re - a_z_re) - two_c_re",
         "            s_re = (a_z_re - a_re) + two_c_re"),
        ("            s_im = (a_im - a_z_im) - two_c_im",
         "            s_im = (a_z_im - a_im) + two_c_im"),
        ("            inc_re, inc_im = _mul_coefficient_left(dtdx, s_re, s_im, EXPANSION)",
         "            inc_re, inc_im = _mul_coefficient_left(minus_dtdx, s_re, s_im, EXPANSION)")))


def _axis_increment_accumulates(source: str) -> Tuple[str, int]:
    """Accumulate the |m| = 1 axis increment where the array path REPLACES.

    THE CLASS THAT ONLY ZERO-INIT REACHES: after the ownership mask the row is
    exactly +0.0 and ``+0.0 + x == x`` for every x except -0.0. Measured on the
    reference: NEEDLE-MISSED across 400 random-seeded rows, CAUGHT (32 words)
    under zero-init with a 2- or 3-cell absorber, and NEEDLE-MISSED again with a
    5-cell one because the census goes silent.
    """
    out, n = _MUT_AXIS_ASSIGN_B.subn(
        "curl0_re = tl.where(at_r, curl0_re + inc_re * -1.0, curl0_re)", source)
    out, n2 = _MUT_AXIS_ASSIGN_D.subn(
        "curl1_re = tl.where(at_r, curl1_re + inc_re * -1.0, curl1_re)", out)
    return out, n + n2


def _bz_prefix_addend_zero(source: str) -> Tuple[str, int]:
    """An exact ``+ 0.0`` added to Bz's prefix difference — a PREDICTED NULL.

    REASON, and it is the same induction the module docstring gives for the
    i*m/r minuend: ``x + 0.0`` differs from ``x`` for exactly one input, ``x =
    -0.0``, and the prefix difference is never -0.0. IEEE round-to-nearest makes
    ``a - b`` an exact ``+0.0`` whenever ``a == b`` and both are finite, so the
    only route to a -0.0 difference is ``pu = -0.0`` with ``pd = +0.0``; and no
    prefix word is ever -0.0, because the prefix is a cumulative scan whose first
    output is +0.0 and ``+0.0 + y`` is +0.0 even at ``y = -0.0``.

    THIS NEEDLE WAS MISCLASSIFIED, and the first device run is what found it: it
    was named ``bz_flat_grouping`` and predicted CAUGHT, borrowing the word count
    of the REFERENCE battery's needle of that name — which plants an entirely
    different defect (it regroups the generic four-operand curl, which Bz's
    prefixed side does not use at all). The device leg reported NEEDLE-MISSED,
    correctly. The choice the reference needle pins — one subtract and one
    complex multiply rather than the four-operand grouping — is STRUCTURAL in the
    kernel: there is no four-operand path for Bz to regroup, so the kernel layer
    cannot carry that needle, and :func:`_bz_prefix_operands_swapped` is what
    holds the prefix difference here instead.
    """
    return _MUT_BZ_PREFIX.subn(
        "curl2_re, curl2_im = _mul_coefficient_left(dtdx, (pu_re - pd_re) + 0.0,\n"
        "                                           (pu_im - pd_im) + 0.0, EXPANSION)",
        source)


def _bz_prefix_operands_swapped(source: str) -> Tuple[str, int]:
    """Bz's prefix difference taken the other way round — ``pd - pu``.

    The needle that actually holds the prefixed Bz curl on the KERNEL layer: the
    array path computes ``prefix[1:] - prefix[:-1]`` (stepping :299-347), and the
    kernel's ``pu`` is the r-UP load. Swapping them negates the whole Bz curl
    wherever the difference is nonzero, so it is a catch on any seeded row —
    unlike the ``+ 0.0`` addend above, which can only reach the ±0 class.
    """
    return _MUT_BZ_PREFIX.subn(
        "curl2_re, curl2_im = _mul_coefficient_left(dtdx, pd_re - pu_re,\n"
        "                                           pd_im - pu_im, EXPANSION)",
        source)


def _zero_rows_field_only(source: str) -> Tuple[str, int]:
    """Zero the |m| >= 2 near-axis FIELDS but not their fu auxiliaries. The
    array path zeroes both (stepping :595-598, :668-671). Reference catch:
    28,512 words."""
    return _MUT_ZERO_AUX.subn("        n0_re = n0_re", source)


def _m0_bx_axis_zero_dropped(source: str) -> Tuple[str, int]:
    """m = 0, B side: the ``Bx[0] = 0`` axis rule dropped (stepping :661-662)."""
    return _MUT_M0_BX_ZERO.subn(
        "        else:\n            v0_re = v0_re\n            v0_im = v0_im\n", source)


def _m0_dz_axis_add_dropped(source: str) -> Tuple[str, int]:
    """m = 0, D side: the on-axis ``Dz[0] += (4*Courant)*Hp[0]`` POST-add dropped
    (stepping :585)."""
    return _MUT_M0_DZ_ADD.subn(
        "            v2_re = v2_re\n            v2_im = v2_im\n", source)


def _m0_dy_axis_zero_dropped(source: str) -> Tuple[str, int]:
    """m = 0, D side: the ``Dy[0] = 0`` axis rule dropped (stepping :586)."""
    return _MUT_M0_DY_ZERO.subn(
        "            v1_re = v1_re\n            v1_im = v1_im\n", source)


def _m0_dz_axis_add_uses_dtdx(source: str) -> Tuple[str, int]:
    """m = 0, D side: the on-axis add scaled by ``dtdx`` instead of the host-rounded
    ``4*dtdx`` — the scalar the array path forms as a Python float and NEP-50
    rounds once (stepping :585)."""
    return _MUT_M0_FOUR_DTDX.subn(
        "_mul_coefficient_left(dtdx, b_re, b_im, EXPANSION)", source)


def _axis_ghost_restored(source: str) -> Tuple[str, int]:
    """PUT THE r_to_minus_r NEAR GHOST BACK — a PREDICTED NULL.

    REASON: the only terms taking a shift-down along r are Dy (partner Hz) and Dz
    (partner Hp), both of r-shift 0, so ``_mask_non_owned_cells`` (:1898-1902)
    zeroes their curl at row 0 — the ONLY row the near ghost writes. At |m| = 1
    that row is then overwritten by the axis increment; at |m| >= 2 the field AND
    its fu are zeroed there. Measured NULL on the reference over the whole matrix
    INCLUDING the zero-init rows, in two independent spellings (invert the
    direction sign; drop the (-1)^m factor).

    If this leg ever reports CAUGHT, the OWNERSHIP MASK has broken, not the ghost.

    The mutation is FAITHFUL AND DELIBERATELY STRONGER than the array path's own
    ghost. On the BACKWARD pass — the only pass with r shift-down operands — the
    r-below loads (``b_r``, ``c_r``, and the prefix's ``p_down``) are served an
    exact +0.0 by the masked load at i = 0. This replaces that zero with the
    stored row-0 value itself, i.e. a GENERALLY NONZERO ghost. If curl row 0 were
    observable at all, substituting a live value for an exact zero must move
    bytes, whichever sign the true ``r_to_minus_r`` image carries — so a null here
    is a stronger statement than reproducing the array path's exact ghost and
    finding it invisible.
    """
    out, n1 = re.subn(
        r"        p_down_re = tl\.load\(pfx \+ 2 \* o_r, mask=vr, other=0\.0\)\n"
        r"        p_down_im = tl\.load\(pfx \+ 2 \* o_r \+ 1, mask=vr, other=0\.0\)\n",
        "        p_down_re = tl.load(pfx + 2 * o_r, mask=vr, other=0.0)\n"
        "        p_down_im = tl.load(pfx + 2 * o_r + 1, mask=vr, other=0.0)\n"
        "        p_c_re = tl.load(pfx + 2 * idx, mask=live, other=0.0)\n"
        "        p_c_im = tl.load(pfx + 2 * idx + 1, mask=live, other=0.0)\n"
        "        p_down_re = tl.where(i == 0, p_c_re * -1.0, p_down_re)\n"
        "        p_down_im = tl.where(i == 0, p_c_im * -1.0, p_down_im)\n",
        source)
    out, n2 = re.subn(
        r"    c_r_re = tl\.load\(g2 \+ 2 \* o_r, mask=vr, other=0\.0\)\n"
        r"    c_r_im = tl\.load\(g2 \+ 2 \* o_r \+ 1, mask=vr, other=0\.0\)\n",
        "    c_r_re = tl.load(g2 + 2 * o_r, mask=vr, other=0.0)\n"
        "    c_r_im = tl.load(g2 + 2 * o_r + 1, mask=vr, other=0.0)\n"
        "    if BACKWARD:\n"
        "        c_r_re = tl.where(i == 0, tl.load(g2 + 2 * idx, mask=live, other=0.0),\n"
        "                          c_r_re)\n"
        "        c_r_im = tl.where(i == 0, tl.load(g2 + 2 * idx + 1, mask=live,\n"
        "                                          other=0.0), c_r_im)\n",
        out)
    return out, n1 + n2


#: ``(name, transform, expectation)``. Every entry's name also appears in
#: :data:`REFERENCE_MUTATIONS`, so the laptop has measured what the device leg
#: expects to see before a GPU is spent — :func:`check_needle_layers` enforces it.
#: ``expectation`` is either ``"caught"``, or ``(kind, reason)`` where kind is
#: ``"null"`` (a PREDICTED null, recorded with its reason and its launch and
#: distinctness evidence) or ``"caught_zero_init_only"``.
MUTATIONS: Tuple[Tuple[str, Callable[[str], Tuple[str, int]], Any], ...] = (
    ("imr_dropped", _imr_dropped, "caught"),
    ("imr_after_mask", _imr_after_mask, "caught"),
    ("axis_increment_before_mask", _axis_increment_before_mask, "caught"),
    ("axis_increment_accumulates", _axis_increment_accumulates,
     ("caught_zero_init_only",
      "after the ownership mask the row is exactly +0.0 and +0.0 + x == x for "
      "every x EXCEPT -0.0, so only a zero-init row with a thin negative-kms "
      "absorber can reach the class")),
    ("axis_increment_operands_swapped", _axis_increment_operands_swapped,
     ("null", "Bx's split-field dsig axis is phi, where kms is 1.0 and never "
              "negative, so the +0.0 `fu*kms` minuend launders the ±0 difference "
              "before it can be stored; measured on the reference leg, and the "
              "D-side twin is what reaches a stored word")),
    ("axis_increment_D_operands_swapped", _axis_increment_d_operands_swapped,
     "caught"),
    ("bz_prefix_operands_swapped", _bz_prefix_operands_swapped, "caught"),
    ("bz_prefix_addend_zero", _bz_prefix_addend_zero,
     ("null",
      "`x + 0.0` differs from `x` only at `x = -0.0`, and the prefix difference "
      "is never -0.0: IEEE round-to-nearest makes `a - b` an exact +0.0 whenever "
      "a == b, and no prefix word is -0.0 because the scan's first output is "
      "+0.0 and `+0.0 + y` is +0.0 even at y = -0.0. Measured on the reference "
      "layer under the same name before any GPU was spent")),
    ("zero_rows_field_only", _zero_rows_field_only, "caught"),
    # The m = 0 axis pair (2026-09-04); scored on the m = 0 rows of MUTATION_CONFIGS.
    ("m0_bx_axis_zero_dropped", _m0_bx_axis_zero_dropped, "caught"),
    ("m0_dz_axis_add_dropped", _m0_dz_axis_add_dropped, "caught"),
    ("m0_dy_axis_zero_dropped", _m0_dy_axis_zero_dropped, "caught"),
    ("m0_dz_axis_add_uses_dtdx", _m0_dz_axis_add_uses_dtdx, "caught"),
    ("imr_planewise_zero_cross_terms", _imr_planewise,
     ("null",
      "the ±0 disagreement the shortcut produces cannot reach a stored word "
      "through `curl - m`: every i*m/r minuend in this kernel is provably never "
      "-0.0 (target 0 leads with the one-cell phi self-difference; target 2 is a "
      "cumsum output or the same +0.0-led sum). Measured on the reference leg "
      "with an in-run minuend census; the choice is pinned by source text")),
    ("invariant_axis_difference_elided", _invariant_axis_elided,
     ("null",
      "the elided addend is an exact +0.0 and the difference only survives at a "
      "-0.0 curl AND a -0.0 recurrence minuend; neither seeding reaches the "
      "pair. Pinned by source text")),
    ("axis_ghost_restored", _axis_ghost_restored,
     ("null", "every shift_down along r feeds only Dy/Dz, whose r-shift is 0, so "
              "the ownership mask zeroes the only row the near ghost writes")),
)


# ---------------------------------------------------------------------------
# The HOST battery — real callables, not prose
# ---------------------------------------------------------------------------
#
# These live on the host side (the coefficient row, the plan's constexpr binding,
# the sub-lattice choice) where a kernel-source regex cannot reach. The first cut
# of this table carried (name, prose, expectation) THREE-TUPLES with no transform,
# so even a driver could not have applied them; each entry now carries a callable
# ``(cyl_module, context) -> (patches, overrides)`` where ``patches`` is a list of
# ``(object, attribute, replacement)`` the case applies and restores, and
# ``overrides`` edits the plan's construction arguments.


def _row_like(xp, iyee_r, sign, m, dtdx, rows, dtype, clamp=1.0):
    """The coefficient row, parameterised so a host needle can bend ONE knob."""
    r_doubled = 2 * xp.arange(rows, dtype=xp.float64) + iyee_r
    divisor = xp.maximum(r_doubled, clamp).reshape(-1, 1, 1)
    return (((-1j) * (sign * 2.0 * int(m) * float(dtdx))) / divisor).astype(dtype)


def _host_row_needle(**bend):
    """Build a host mutation that rebuilds the i*m/r row with one knob bent."""

    def apply(cyl, context):
        from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415

        def patched(xp, target, sign, m, dtdx, rows, dtype):
            iyee_r = 0 if bend.get("iyee_zero") else IYEE_SHIFTS[target][0]
            if bend.get("sign_flipped"):
                sign = -sign
            if bend.get("m_abs"):
                m = abs(int(m))
            if bend.get("dtdx_f32"):
                dtdx = float(np.float32(dtdx))
            return _row_like(xp, iyee_r, sign, m, dtdx, rows, dtype,
                             clamp=bend.get("clamp", 1.0))

        return [(cyl, "imr_coefficient_row", patched)], {}

    return apply


def _host_lattice_swap(cyl, context):
    """kms/sinv taken from the WRONG Yee sub-lattice — a half-cell absorber
    error that no shape or dtype check can see."""
    return [], {"half_integer": not context["half_integer"]}


def _host_bcz_forced_metallic(cyl, context):
    """BCZ compiled in rather than resolved: the periodic-z rows are the needle."""
    return [], {"bcz": 1}


def _host_zero_rows_inverted(cyl, context):
    """ZERO_ROWS takes |m| where the accurate branch wants 1, and 1 where the
    default hack wants |m| (stepping._cylindrical_axis_rows, :601-622)."""

    def patched(m, accurate_fields_near_cylorigin):
        if abs(int(m)) < 2:
            return 0
        return abs(int(m)) if bool(accurate_fields_near_cylorigin) else 1

    return [(cyl, "zero_rows", patched)], {}


def _host_increment_scalars_unrounded(cyl, context):
    """The |m| = 1 B-side scalars passed WITHOUT the complex64 rounding the array
    path applies once at the NEP-50 boundary — a PREDICTED NULL.

    REASON, and it is PLATFORM FACT (a) applied rather than rediscovered: Triton
    types a Python float argument as fp32, so the float64 and the
    float32-rounded spelling of the SAME value arrive at the kernel as the same
    ABI bits, and the double-rounding half of this needle cannot bite at all.
    What is left is the REAL WORD'S SIGN OF ZERO — shipped ``1j * (m*dtdx)``
    gives ``-0.0`` at m < 0 where this spelling gives ``+0.0`` — and that
    difference survives only where the increment's product lands on a signed
    zero, which the whole matrix measures at 0 words: the real plane is
    ``(±0.0)*z_re - inc_im*z_im`` and the second term is nonzero wherever the
    increment is live.

    THE ENTRY WAS PREDICTED "caught" AND THE FIRST DEVICE RUN REPORTED
    NEEDLE-MISSED, correctly. It is recorded as a null with its reason, and
    :func:`_host_increment_scalars_m_abs` is the needle that actually holds these
    scalars — the sign of m, which is a real defect and a large one.
    """

    def patched(m, dtdx):
        return (-float(dtdx), (0.0, float(int(m) * float(dtdx))))

    return [(cyl, "axis_increment_scalars", patched)], {}


def _host_increment_scalars_m_abs(cyl, context):
    """The |m| = 1 B-side scalars built from ``abs(m)``.

    Seven of the sixteen lifted corpus rows carry m = -1, so the SIGN of m in
    ``1j * (m * dtdx)`` is load-bearing and this is the host-side needle for it:
    it flips the imaginary word of the increment's second term, which is a plain
    arithmetic change on every |m| = 1 row rather than a ±0-class one.
    """
    import numpy  # noqa: PLC0415

    def patched(m, dtdx):
        second = numpy.complex64(1j * (abs(int(m)) * float(dtdx)))
        return (float(numpy.float32(-float(dtdx))),
                (float(numpy.float32(second.real)),
                 float(numpy.float32(second.imag))))

    return [(cyl, "axis_increment_scalars", patched)], {}


#: ``(name, description, expectation, apply)``.
HOST_MUTATIONS: Tuple[Tuple[str, str, Any, Callable[..., Any]], ...] = (
    ("imr_iyee_zero", "coefficient row built with iyee_r = 0 for every target",
     "caught", _host_row_needle(iyee_zero=True)),
    ("imr_sign_flipped", "coefficient row numerator sign inverted", "caught",
     _host_row_needle(sign_flipped=True)),
    ("imr_m_abs", "coefficient row numerator uses abs(m)", "caught",
     _host_row_needle(m_abs=True)),
    ("imr_dtdx_prerounded_to_f32",
     "coefficient row built from an f32-rounded dtdx (double rounding)",
     "caught", _host_row_needle(dtdx_f32=True)),
    ("imr_clamp_value_raised", "the divisor clamp raised to 2.0, which moves row "
     "0 of the shift-1 targets (Bz, Dx) — the rows the ownership mask does NOT "
     "zero, and the reason the clamp's VALUE is not free even though DROPPING it "
     "is invisible", "caught", _host_row_needle(clamp=2.0)),
    ("coeff_lattice_swap", "kms/sinv taken from the wrong Yee sub-lattice",
     "caught", _host_lattice_swap),
    ("bcz_forced_metallic", "BCZ compiled in as METALLIC rather than resolved "
     "from the grid; the periodic-z rows are the needle",
     "caught", _host_bcz_forced_metallic),
    ("zero_rows_accurate_inverted", "ZERO_ROWS uses |m| where accurate wants 1",
     "caught", _host_zero_rows_inverted),
    ("axis_increment_scalars_unrounded",
     "the |m| = 1 scalars passed without the complex64 rounding, so the "
     "signed-zero real word never exists",
     ("null",
      "PLATFORM FACT (a): Triton types a Python float argument as fp32, so the "
      "float64 and float32-rounded spellings of one value arrive as the same ABI "
      "bits and the double-rounding half cannot bite. What remains is the real "
      "word's SIGN OF ZERO, and it reaches no stored word on this matrix: the "
      "real plane is (±0.0)*z_re - inc_im*z_im and the second term is nonzero "
      "wherever the increment is live"),
     _host_increment_scalars_unrounded),
    ("axis_increment_scalars_m_abs",
     "the |m| = 1 scalars built from abs(m), so the negative-m rows — seven of "
     "the sixteen lifted ones — take the wrong sign", "caught",
     _host_increment_scalars_m_abs),
)


# ===========================================================================
# Drivers
# ===========================================================================

def run_self_check(out_path: str, steps: int,
                   legs: Sequence[str] = ()) -> Dict[str, Any]:
    """The LAPTOP legs: no CUDA, no Triton, no byte-identity claim.

    Proves (a) the in-file reference equals the shipped array path on real
    complex Dcyl grids, (b) the constitutive identity, non-vacuously, (c) that
    every needle in every battery is armed and live, and (d) the two identity
    controls (r-axis metallic, cylindrical clauses stripped) the module's own
    findings rest on — so no GPU time is spent on a dead needle or an unmeasured
    claim.

    ``legs`` selects a subset; empty means all of :data:`LAPTOP_LEGS`. A leg the
    caller did not ask for must not run, or the summary's ``legs_ran`` lies about
    what produced the record.
    """
    wanted = set(legs) if legs else set(LAPTOP_LEGS)
    results: Dict[str, Any] = {
        "kind": "self_check",
        "claim": "NO BYTE-IDENTITY CLAIM: this leg ran on NumPy with no Triton "
                 "and no device. It certifies the reference transcription and the "
                 "mutation battery only.",
        "environment": environment(),
        "sweep": {"shapes": [list(s) for s in SHAPES], "dtdx": list(DTDX),
                  "m_cases": [list(c) for c in M_CASES],
                  "z": ["metallic", "periodic"], "steps": steps},
    }

    if "reference" not in wanted:
        results["reference"] = {"skipped": "not requested"}
        return _run_remaining_laptop_legs(results, out_path, steps, wanted)

    rows: List[Dict[str, Any]] = []
    skips: List[Dict[str, Any]] = []
    t0 = time.time()
    total = len(M_CASES) * len(SHAPES) * len(DTDX) * len(Z_METALLIC) * len(SUB_STEPS)
    done = 0
    for m, accurate in M_CASES:
        for shape in SHAPES:
            for courant in DTDX:
                if accurate and courant > 1.0 / (abs(m) + 0.5):
                    # RECORDED, never a bare continue: an unstable (m, Courant)
                    # pair Grid itself refuses is not a row this sweep may claim,
                    # and the FIRST cut of this loop dropped the whole (3, True)
                    # case silently because no Courant in DTDX was below 1/3.5.
                    done += len(Z_METALLIC) * len(SUB_STEPS)
                    skips.append({"m": int(m), "accurate": True,
                                  "courant": float(courant),
                                  "bound": 1.0 / (abs(m) + 0.5),
                                  "reason": "above the accurate-branch stability "
                                            "bound Grid enforces"})
                    continue
                for z_metallic in Z_METALLIC:
                    for sub_step in SUB_STEPS:
                        done += 1
                        rows.append(_reference_row(shape, m, accurate, courant,
                                                   z_metallic, sub_step, steps))
                        log(f"[{done}/{total}] reference m={m:+d} acc={int(accurate)} "
                            f"{tuple(shape)} C={courant:.6g} "
                            f"z={'M' if z_metallic else 'P'} {sub_step} -> "
                            f"{'IDENTICAL' if rows[-1]['first_divergence'] is None else 'DIVERGED'}")
                        results["reference"] = {"rows": rows}
                        save(results, out_path)

    # EVERY declared case must have contributed rows. A table entry that runs
    # nothing is worse than an absent one: it advertises coverage.
    per_case: Dict[str, int] = {}
    for row in rows:
        key = f"m={row['m']:+d},accurate={int(row['accurate'])}"
        per_case[key] = per_case.get(key, 0) + 1
    empty = [f"m={m:+d},accurate={int(accurate)}" for m, accurate in M_CASES
             if per_case.get(f"m={m:+d},accurate={int(accurate)}", 0) == 0]
    results["reference"] = {
        "rows": rows,
        "identical_rows": sum(r["first_divergence"] is None for r in rows),
        "total_rows": len(rows), "steps": steps,
        "rows_per_case": per_case, "empty_cases": empty,
        "skipped_unstable": skips,
        "elapsed_s": round(time.time() - t0, 2)}
    save(results, out_path)
    log(f"reference leg: {results['reference']['identical_rows']}"
        f"/{results['reference']['total_rows']} identical, "
        f"{len(skips)} (m, Courant) pairs skipped as unstable, "
        f"empty cases: {empty or 'none'}")
    if empty:
        raise AssertionError(
            f"M_CASES entries {empty} contributed no rows: the sweep advertises "
            f"a case it never ran. Add a Courant below their stability bound.")

    return _run_remaining_laptop_legs(results, out_path, steps, wanted)


def _run_remaining_laptop_legs(results: Dict[str, Any], out_path: Optional[str],
                               steps: int, wanted: set) -> Dict[str, Any]:
    """The laptop legs after ``reference``, each one only if it was asked for."""
    if "constitutive" in wanted:
        run_constitutive_identity(out_path, results)
    if "constitutive_mutations" in wanted:
        run_constitutive_mutations(out_path, results)
    if "reference_mutations" in wanted:
        run_reference_mutations(out_path, results, steps)
    if "axis_identity" in wanted:
        run_axis_identity(out_path, results, steps)
    if "stripped_control" in wanted:
        run_stripped_control(out_path, results, steps)
    if "needle_arming" in wanted:
        run_needle_arming(out_path, results)
    save(results, out_path)
    return results


#: The DISTINCT azimuthal orders the constitutive leg sweeps. Written out rather
#: than derived from :data:`M_CASES` because that table is keyed by
#: ``(m, accurate)`` and ``accurate`` changes nothing in ``update_H``/``update_E``
#: — iterating it duplicated m = 2 and m = 3 and inflated the row count.
CONSTITUTIVE_M_VALUES: Tuple[int, ...] = (0, 1, -1, 2, -2, 3, 5)
CONSTITUTIVE_SHAPES: Tuple[Tuple[int, int, int], ...] = ((20, 1, 24), (13, 1, 11))
CONSTITUTIVE_SEEDINGS: Tuple[str, ...] = ("random", "signed_zero")


def constitutive_identity_row_count() -> int:
    """The leg's row count, from its own axes — never a number typed by hand."""
    return (len(CONSTITUTIVE_M_VALUES) * len(Z_METALLIC) * len(DTDX)
            * len(CONSTITUTIVE_SHAPES) * len(CONSTITUTIVE_SEEDINGS) * 2)


def run_constitutive_identity(out_path: Optional[str],
                              results: Dict[str, Any]) -> Dict[str, Any]:
    """THE FINDING'S EVIDENCE: 32 of the 64 slots need no new kernel.

    ``stepping.update_H``/``update_E`` on complex Dcyl grids against
    :func:`reference_constitutive_step`, which carries NO cylindrical clause. If
    every row is identical, the certified ``complex_fields.bloch_constitutive_step``
    already computes both constitutive sub-steps and this tranche ships ONE kernel.

    THE AXES ARE THE ARTIFACT'S, not the prose's: m (6 distinct orders) x z (2) x
    Courant (:data:`DTDX`, three of them non-power-of-two) x shape (2) x seeding
    (random, signed_zero) x side (H, E). :func:`constitutive_identity_row_count`
    computes the product so the count in the record cannot drift from the loop,
    and every row asserts its own non-vacuity (see :func:`one_constitutive_case`).
    """
    rows: List[Dict[str, Any]] = []
    total = constitutive_identity_row_count()
    for m in CONSTITUTIVE_M_VALUES:
        for z_metallic in Z_METALLIC:
            for courant in DTDX:
                for shape in CONSTITUTIVE_SHAPES:
                    for seeding in CONSTITUTIVE_SEEDINGS:
                        for side in ("H", "E"):
                            rows.append(one_constitutive_case(
                                np, shape, m, courant, z_metallic, side, seeding))
                            if len(rows) % 48 == 0:
                                log(f"[{len(rows)}/{total}] constitutive identity "
                                    f"m={m:+d} C={courant:.6g} {seeding}")
                                results["constitutive_identity"] = {"rows": rows}
                                save(results, out_path)
    if len(rows) != total:
        raise AssertionError(f"constitutive leg ran {len(rows)} rows, "
                             f"its own axes say {total}")
    record = {"rows": rows,
              "identical_rows": sum(r["first_divergence"] is None for r in rows),
              "total_rows": len(rows),
              "steps_per_row": CONSTITUTIVE_STEPS,
              "axes": {"m": list(CONSTITUTIVE_M_VALUES),
                       "z": ["metallic", "periodic"], "dtdx": list(DTDX),
                       "shapes": [list(s) for s in CONSTITUTIVE_SHAPES],
                       "seedings": list(CONSTITUTIVE_SEEDINGS),
                       "sides": ["H", "E"]},
              "min_census_peak_signed_zero": min(
                  [r["census_peak"] for r in rows if r["seeding"] == "signed_zero"],
                  default=0),
              "min_moved_words": min([r["moved_words"] for r in rows], default=0)}
    results["constitutive_identity"] = record
    save(results, out_path)
    log(f"constitutive identity: {record['identical_rows']}/{record['total_rows']} "
        f"identical (min moved={record['min_moved_words']}, min ±0 census="
        f"{record['min_census_peak_signed_zero']}) -> the tranche needs "
        f"{'ONE kernel' if record['identical_rows'] == record['total_rows'] else 'TWO kernels'}")
    return record


#: The constitutive reference's OWN needle battery. Without it the identity leg
#: is a comparison of two spellings nobody can break — and a measured
#: demonstration exists that it was: with zero-init seeding a reference whose
#: kps/kms are swapped still reported IDENTICAL on every row.
CONSTITUTIVE_MUTATIONS: Tuple[Tuple[str, str, str, Any], ...] = (
    ("constitutive_coefficients_swapped",
     "        field += kps[index] * fw                 # coefficient LEFT (:2086-2087)\n"
     "        field -= kms[index] * previous           # coefficient LEFT (:2093-2095)",
     "        field += kms[index] * fw\n"
     "        field -= kps[index] * previous", "caught"),
    ("constitutive_previous_read_after_store",
     "        previous = fw.copy()                     # read BEFORE the store (:2083-2085)\n"
     "        value = (arrays[sources[index]] * inverse_epsilon[index] if side == \"E\"\n"
     "                 else arrays[sources[index]])\n"
     "        fw[...] = value",
     "        value = (arrays[sources[index]] * inverse_epsilon[index] if side == \"E\"\n"
     "                 else arrays[sources[index]])\n"
     "        fw[...] = value\n"
     "        previous = fw.copy()", "caught"),
    ("constitutive_update_sign_flipped",
     "        field -= kms[index] * previous           # coefficient LEFT (:2093-2095)",
     "        field += kms[index] * previous", "caught"),
    ("constitutive_inverse_epsilon_dropped",
     "        value = (arrays[sources[index]] * inverse_epsilon[index] if side == \"E\"\n"
     "                 else arrays[sources[index]])",
     "        value = arrays[sources[index]]", "caught"),
    ("constitutive_half_lattice_swapped",
     '    suffix = "_h" if half else ""',
     '    suffix = "" if half else "_h"', "caught"),
    ("constitutive_inverse_epsilon_right",
     "arrays[sources[index]] * inverse_epsilon[index] if side ==",
     "inverse_epsilon[index] * arrays[sources[index]] if side ==",
     ("null", "IEEE multiplication is commutative bit for bit, so the operand "
              "order around the chi1inv row cannot change a word; the ORDER that "
              "does matter is D-left versus a row product that reads neighbours, "
              "and that configuration is refused by the predicate, not spelled "
              "here")),
)


def run_constitutive_mutations(out_path: Optional[str], results: Dict[str, Any]
                               ) -> Dict[str, Any]:
    """Break the constitutive reference on purpose; every seeding must catch it.

    THE VACUITY LEVEL FOR THE IDENTITY LEG. A seeding that catches nothing across
    the whole battery cannot fail for any reason connected to the arithmetic, and
    is reported VACUOUS rather than passed — that is exactly what a zero-init
    constitutive row was, and this is the check that would have said so.
    """
    source_path = os.path.abspath(__file__)
    source = open(source_path, "r", encoding="utf-8").read()
    report: Dict[str, Any] = {"rows": [], "source": source_path}

    def evaluate(module, seeding: str) -> int:
        total = 0
        for m in (1, -2, 3):
            for z_metallic in Z_METALLIC:
                for side in ("H", "E"):
                    try:
                        row = module.one_constitutive_case(
                            np, (13, 1, 11), m, 0.37, z_metallic, side, seeding)
                    except AssertionError:
                        total += 1          # a raise (including VACUOUS) is a catch
                        continue
                    found = row["first_divergence"]
                    total += 0 if found is None else found["differing_words"]
        return total

    clean = _load_module(source, "cylcomplex_constitutive_clean")
    control = {seeding: evaluate(clean, seeding) for seeding in CONSTITUTIVE_SEEDINGS}
    report["clean_control"] = control
    log(f"constitutive clean control: {control} (every entry must be 0)")
    if any(control.values()):
        report["verdict"] = "VOID: the clean control is not identical"
        results["constitutive_mutations"] = report
        save(results, out_path)
        return report

    per_seeding: Dict[str, int] = {seeding: 0 for seeding in CONSTITUTIVE_SEEDINGS}
    for name, pattern, replacement, predicted in CONSTITUTIVE_MUTATIONS:
        mutated = mutate_region(source, "constitutive", pattern, replacement)
        if mutated is None:
            report["rows"].append(
                {"name": name, "status": "DISARMED",
                 "detail": "pattern not found in the constitutive CODE region"})
            log(f"{name:44s} DISARMED")
            continue
        module = _load_module(mutated, "cylcomplex_constitutive_" + name)
        moved = {seeding: evaluate(module, seeding)
                 for seeding in CONSTITUTIVE_SEEDINGS}
        wants_null = isinstance(predicted, tuple) and predicted[0] == "null"
        if wants_null:
            status = ("NULL-AS-PREDICTED" if not any(moved.values())
                      else "UNEXPECTEDLY-CAUGHT")
        else:
            # ANY seeding, not every one — and the difference is measured, not
            # conceded: `constitutive_inverse_epsilon_dropped` is invisible to the
            # random seeding (these grids are vacuum, so chi1inv is exactly 1.0
            # and the multiply is an identity) and VISIBLE to the ±0 lattice,
            # because `a_im * 0.0` carries the imaginary word's sign into the real
            # part. The per-seeding totals below are what stop "any" from letting
            # a seeding coast.
            status = "CAUGHT" if any(moved.values()) else "NEEDLE-MISSED"
        for seeding in CONSTITUTIVE_SEEDINGS:
            per_seeding[seeding] += moved[seeding]
        report["rows"].append(
            {"name": name, "status": status, "differing_words": moved,
             "predicted": predicted if isinstance(predicted, str) else list(predicted)})
        log(f"{name:44s} {status:20s} {moved}")
        results["constitutive_mutations"] = report
        save(results, out_path)

    report["per_seeding_total_words"] = per_seeding
    vacuous = [seeding for seeding, total in per_seeding.items() if total == 0]
    failures = [row["name"] for row in report["rows"]
                if row["status"] in ("DISARMED", "NEEDLE-MISSED",
                                     "UNEXPECTEDLY-CAUGHT")]
    report["vacuous_seedings"] = vacuous
    report["failed_legs"] = failures
    report["verdict"] = ("all constitutive needles live"
                         if not failures and not vacuous else "FAILED")
    results["constitutive_mutations"] = report
    save(results, out_path)
    log(f"constitutive mutations: {report['verdict']} per_seeding={per_seeding}"
        + (f" vacuous={vacuous}" if vacuous else ""))
    return report


#: The REFERENCE-layer mutation battery. Each entry mutates the in-file reference
#: transcription rather than the Triton kernel, which is what makes it runnable on
#: a Triton-less laptop — the point being to prove every needle is LIVE before a
#: GPU is spent on it.
#:
#: ITS RELATION TO :data:`MUTATIONS` IS A MAP, NOT AN EQUALITY, and
#: :func:`needle_layer_map` is what states it: every KERNEL-SOURCE needle has a
#: reference counterpart of the same name, so a reference measurement licenses
#: what the device leg expects to see; the reference battery additionally carries
#: needles for choices that exist only in the TRANSCRIPTION (the partner tables,
#: the wall row, the ghost phase) or only on the HOST (the coefficient row), which
#: have no kernel-source spelling and are gated by :data:`HOST_MUTATIONS` or by
#: source-text assertions in the laptop test file. An earlier revision of this
#: comment claimed the two batteries mirrored one for one; they were 6 against 20
#: and the mismatch hid a dead needle.
REFERENCE_MUTATIONS: Tuple[Tuple[str, str, str, Any], ...] = (
    ("imr_dropped", "curl = curl + -(imr_product(factor, snap[partner]))",
     "curl = curl + 0.0 * snap[partner]", "caught"),
    ("imr_sign_flipped", "(sign * 2.0 * m * dtdx)", "(-sign * 2.0 * m * dtdx)",
     "caught"),
    ("imr_iyee_zero", "iyee_r = IYEE_SHIFTS[target][0]", "iyee_r = 0", "caught"),
    ("imr_m_abs", "(sign * 2.0 * m * dtdx)", "(sign * 2.0 * abs(m) * dtdx)",
     "caught"),
    ("imr_partner_swapped",
     'IMR_D = (("Dx", "Hz", -1.0), ("Dz", "Hx", +1.0))',
     'IMR_D = (("Dx", "Hx", -1.0), ("Dz", "Hz", +1.0))', "caught"),
    ("axis_increment_B_dropped", 'axis_increment = ("Bx",',
     'axis_increment = ("__none__",', "caught"),
    ("axis_increment_D_dropped", 'axis_increment = ("Dy", dtdx',
     'axis_increment = ("__none__", dtdx', "caught"),
    ("axis_increment_not_negated",
     "curl[_face(0, 0)] = -axis_increment[1].astype(curl.dtype)",
     "curl[_face(0, 0)] = axis_increment[1].astype(curl.dtype)", "caught"),
    ("axis_increment_accumulates",
     "curl[_face(0, 0)] = -axis_increment[1].astype(curl.dtype)",
     "curl[_face(0, 0)] += -axis_increment[1].astype(curl.dtype)",
     ("caught_zero_init_only",
      "after the ownership mask the row is exactly +0.0 and +0.0 + x == x for "
      "every x EXCEPT -0.0; measured NEEDLE-MISSED on 400 random-seeded rows and "
      "CAUGHT (32 words) under zero-init with a 2- or 3-cell absorber")),
    ("ez_off_axis_row_zero", 'xp.take(snap["Ez"], 1, axis=0)',
     'xp.take(snap["Ez"], 0, axis=0)', "caught"),
    ("hz_factor_one", '- 2.0 * snap["Hz"]', '- 1.0 * snap["Hz"]', "caught"),
    ("bz_generic_curl", "curl = dtdx * (extended_prefix[1:] - extended_prefix[:-1])",
     "curl = curl", "caught"),
    ("bz_flat_grouping",
     "return dtdx * ((shifted_first - first) + (second - shifted_second))",
     "return dtdx * (shifted_first - first + second - shifted_second)", "caught"),
    ("bz_prefix_operands_swapped",
     "curl = dtdx * (extended_prefix[1:] - extended_prefix[:-1])",
     "curl = dtdx * (extended_prefix[:-1] - extended_prefix[1:])", "caught"),
    ("bz_prefix_addend_zero",
     "curl = dtdx * (extended_prefix[1:] - extended_prefix[:-1])",
     "curl = dtdx * ((extended_prefix[1:] - extended_prefix[:-1]) + 0.0)",
     ("null",
      "the LAPTOP half of the kernel needle of the same name: `x + 0.0` differs "
      "from `x` only at `x = -0.0`, and a prefix difference is never -0.0 (the "
      "scan is led by +0.0 and `a - b` is +0.0 whenever a == b under RN). If "
      "this is ever CAUGHT the kernel's twin must be re-armed as a catch")),
    ("no_wall_row", "extended[rows_r] = 0", "extended[rows_r] = snap['Ey'][-1]",
     "caught"),
    ("dz_prefix_missing", 'src["Hy"] = prefix ', "pass  #", "caught"),
    # THE m = 0 AXIS PAIR (2026-09-04), reference layer. Each has a kernel twin of
    # the same name; all four are catches on the m = 0 rows of the sweep.
    ("m0_bx_axis_zero_dropped", 'fields["Bx"][_face(0, 0)] = 0',
     "pass  # m0: Bx axis zero dropped", "caught"),
    ("m0_dz_axis_add_dropped",
     'fields["Dz"][_face(0, 0)] += (4.0 * dtdx) * fields["Hy"][_face(0, 0)]',
     "pass  # m0: Dz axis add dropped", "caught"),
    ("m0_dy_axis_zero_dropped", 'fields["Dy"][_face(0, 0)] = 0',
     "pass  # m0: Dy axis zero dropped", "caught"),
    ("m0_dz_axis_add_uses_dtdx",
     '(4.0 * dtdx) * fields["Hy"][_face(0, 0)]',
     '(1.0 * dtdx) * fields["Hy"][_face(0, 0)]', "caught"),
    ("zero_rows_field_only",
     'for name in ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz"):',
     'for name in ("Dx", "Dy", "Dz"):', "caught"),
    ("zero_rows_accurate_always",
     "near = slice(0, 1) if accurate else slice(0, abs(m))", "near = slice(0, 1)",
     "caught"),
    ("imr_negation_as_subtract",
     "curl = curl + -(imr_product(factor, snap[partner]))",
     "curl = curl - (imr_product(factor, snap[partner]))",
     ("null", "IEEE-754 defines subtraction AS addition of the negation, so the "
              "two spellings are the same bits on every input including signed "
              "zeros — this is the IDENTITY leg that licenses the kernel to spell "
              "the fold as a single subtract per plane")),
    ("axis_ghost_sign", 'sign = -1.0 if component[1] in ("x", "y") else 1.0',
     'sign = 1.0 if component[1] in ("x", "y") else -1.0',
     ("null", "every shift_down along r feeds only Dy/Dz, whose r-shift is 0, so "
              "_mask_non_owned_cells zeroes the only row the near ghost writes; at "
              "|m| = 1 that row is then overwritten by the axis increment, at "
              "|m| >= 2 the field AND its fu are zeroed there")),
    ("axis_ghost_phase_dropped", "sign *= axis_phase", "sign *= 1.0",
     ("null", "same ownership-mask argument as axis_ghost_sign")),
    # The name-aligned counterpart of the KERNEL's axis_ghost_restored: the same
    # ghost the mutated kernel plants (row 0 negated, no component sign and no
    # (-1)^m), so a null on both layers is one claim measured twice rather than
    # two claims that happen to agree.
    ("axis_ghost_restored",
     '        sign = -1.0 if component[1] in ("x", "y") else 1.0\n'
     "        sign *= axis_phase\n"
     "        shifted[_face(axis, 0)] = sign * field[_face(axis, 0)]",
     "        shifted[_face(axis, 0)] = -1.0 * field[_face(axis, 0)]",
     ("null", "the near ghost is written only into row 0, and every term that "
              "takes a shift-down along r (Dy's partner Hz, Dz's partner Hp) has "
              "r-shift 0, so the ownership mask zeroes that row before it can be "
              "stored")),

    # ---- the ORDERING choices (grouping choice 6) --------------------------
    # Added 2026-08-13: the module cited a word count for an ``imr_after_mask``
    # mutation that existed in NEITHER battery. The ordering was the one
    # constraint tying the fold, the ownership mask and the axis-row replacement
    # together, and it was pinned by prose alone.
    ("imr_after_mask",
     "            curl = curl + -(imr_product(factor, snap[partner]))  # sign convention (:724)\n"
     "        mask(curl, iyee, boundaries)                            # stepping :369/:450",
     "            pass\n"
     "        mask(curl, iyee, boundaries)                            # stepping :369/:450\n"
     "        if m != 0 and target in imr:\n"
     "            curl = curl + -(imr_product(factor, snap[partner]))",
     "caught"),
    ("axis_increment_before_mask",
     "        mask(curl, iyee, boundaries)                            # stepping :369/:450\n"
     "        if axis_increment is not None and axis_increment[0] == target:\n"
     "            # REPLACES the row, AFTER the mask (:370-372 / :451-453).\n"
     "            curl[_face(0, 0)] = -axis_increment[1].astype(curl.dtype)",
     "        if axis_increment is not None and axis_increment[0] == target:\n"
     "            curl[_face(0, 0)] = -axis_increment[1].astype(curl.dtype)\n"
     "        mask(curl, iyee, boundaries)                            # stepping :369/:450\n"
     "        if False:\n"
     "            curl[_face(0, 0)] = -axis_increment[1].astype(curl.dtype)",
     "caught"),

    # ---- the ±0 choices: measured, and recorded as whatever they measure ----
    ("imr_planewise_zero_cross_terms", "    return factor * values",
     "    real = (factor.imag * values.imag) * -1.0\n"
     "    imag = factor.imag * values.real\n"
     "    out = np.empty(np.broadcast(real, imag).shape, dtype=values.dtype)\n"
     "    out.real, out.imag = real, imag\n"
     "    return out",
     ("null",
      "the plane-wise shortcut and the full product differ ONLY in the ±0 class, "
      "and that class cannot reach a stored word THROUGH this kernel's fold: the "
      "fold is `curl - m`, and `x - (+0.0)` differs from `x - (-0.0)` only at "
      "x = -0.0, while every i*m/r minuend here is provably never -0.0. Target 0 "
      "leads with the phi self-difference `(c_p - c)`, which is +0.0 on a "
      "one-cell axis, and `+0.0 + y` is +0.0 even at y = -0.0; target 2 is either "
      "a cumsum output (B side — a prefix scan that starts at +0.0 can never "
      "produce -0.0) or the same +0.0-led sum (D side). The minuend census in "
      "this leg's record is the measurement, not the argument. The choice is "
      "therefore pinned by SOURCE TEXT in the laptop test file, the layer that "
      "can hold a byte-invisible choice, exactly as BFAST pins its three")),
    ("invariant_axis_difference_elided",
     "        curl = curl_from(src[g1], sf, src[g2], ss, dtdx)",
     "        curl = curl_from_invariant_elided(src[g1], sf, src[g2], ss, dtdx, a1, a2)",
     ("null",
      "eliding the phi addend changes `dtdx*(x + 0.0f)` into `dtdx*x`, which "
      "differs only at x = -0.0 AND only if the difference then survives the "
      "recurrence's own `fu*kms - curl` at a -0.0 minuend. Neither seeding "
      "reaches the pair: the sources are never updated inside a curl-only leg, "
      "so under zero-init every operand word is +0.0 and every difference is "
      "+0.0, and a random seed hits an exact -0.0 difference with probability "
      "zero. Byte-invisible on this matrix and pinned by source text")),
    # BOTH sides, and that is load-bearing rather than thorough: swapping only
    # the B side measured NEEDLE-MISSED on every seeding, because Bx's split-field
    # dsig axis is phi — where kms is 1.0, never negative — so its `fu*kms` minuend
    # is +0.0 and `+0.0 - (±0.0)` launders the difference away before it can be
    # stored. Dy's dsig axis is z, which carries the thin absorber's negative kms,
    # and there the same swap is byte-visible. A needle that can only be reached
    # through ONE of the two sub-steps has to spell both.
    ("axis_increment_operands_swapped",
     "                              (-dtdx) * (ep[_face(0, 0)] - ep_above[_face(0, 0)])",
     "                              dtdx * (ep_above[_face(0, 0)] - ep[_face(0, 0)])",
     ("null",
      "MEASURED null, and the measurement is the point: the swap really does "
      "produce -0.0 where the array path's spelling produces +0.0 in the B-side "
      "increment, but Bx's split-field dsig axis is PHI, where kms is 1.0 and "
      "never negative, so the recurrence's `fu*kms` minuend is +0.0 and "
      "`+0.0 - (±0.0)` is +0.0 either way. The D twin below carries the same "
      "defect through Dy, whose dsig axis is z and does go negative under the "
      "thin absorber — which is why grouping choice 5 needs BOTH spellings and "
      "why the B one alone would have been a dead needle reported as a pass")),
    ("axis_increment_D_operands_swapped",
     '            axis_increment = ("Dy", dtdx * (hr[_face(0, 0)] - hr_below[_face(0, 0)]\n'
     "                                            - 2.0 * snap[\"Hz\"][_face(0, 0)]))",
     '            axis_increment = ("Dy", (-dtdx) * ((hr_below[_face(0, 0)] - hr[_face(0, 0)])\n'
     "                                               + 2.0 * snap[\"Hz\"][_face(0, 0)]))",
     "caught"),

    # ---- the coefficient row's construction (grouping choices 1 and 2) -----
    ("imr_dtdx_prerounded_to_f32", "(((-1j) * (sign * 2.0 * m * dtdx)) / divisor)",
     "(((-1j) * (sign * 2.0 * m * float(np.float32(dtdx)))) / divisor)", "caught"),
    ("imr_clamp_value_raised", "    divisor = xp.maximum(r_doubled, 1.0)",
     "    divisor = xp.maximum(r_doubled, 2.0)", "caught"),
    ("imr_clamp_dropped",
     "    divisor = xp.maximum(r_doubled, 1.0).reshape(-1, 1, 1)\n"
     "    return (((-1j) * (sign * 2.0 * m * dtdx)) / divisor).astype(dtype)",
     "    divisor = r_doubled.reshape(-1, 1, 1)\n"
     "    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n"
     "        return (((-1j) * (sign * 2.0 * m * dtdx)) / divisor).astype(dtype)",
     ("null",
      "the clamp is a DOMAIN GUARD, not arithmetic: `maximum(2*ir + iyee_r, 1.0)` "
      "bites only where the doubled coordinate is 0, i.e. row 0 of a shift-0 "
      "target (Bx, Dz), and that row's curl is zeroed by the ownership mask "
      "before it can be stored. Dropping it divides by zero there and the "
      "resulting non-finite word is masked away — which is also why the clamp's "
      "VALUE is not free: raising it to 2.0 would move row 0 of the shift-1 "
      "targets (Bz, Dx), whose row 0 is NOT masked")),

    # ---- BCZ is a real constexpr, not a compiled-in constant (choice 7) ----
    ("bcz_forced_metallic", '    backward = sub_step == "step_D"',
     '    boundaries = (boundaries[0], boundaries[1], "metallic")\n'
     '    backward = sub_step == "step_D"', "caught"),
)


#: Which LAYER holds each pinned choice. Written as data so "choice N has a
#: needle" is checkable by a laptop test instead of being a claim in prose.
#: ``reference``/``kernel`` name a battery entry; ``host`` names a
#: :data:`HOST_MUTATIONS` entry; ``source_text`` means the choice is byte-INVISIBLE
#: on the measured matrix and is pinned by an assertion in
#: ``meep_gpu/test_triton_cylindrical_complex.py`` — the honest layer for one,
#: and the BFAST tranche's precedent.
GROUPING_CHOICE_NEEDLES: Dict[int, Tuple[str, ...]] = {
    1: ("imr_iyee_zero", "imr_dtdx_prerounded_to_f32", "imr_iyee_zero(host)"),
    2: ("imr_clamp_dropped", "imr_clamp_value_raised"),
    3: ("imr_negation_as_subtract",),
    4: ("axis_increment_accumulates",),
    5: ("axis_increment_not_negated", "axis_increment_operands_swapped",
        "axis_increment_D_operands_swapped"),
    6: ("imr_after_mask", "axis_increment_before_mask"),
    7: ("bcz_forced_metallic", "coeff_lattice_swap(host)"),
    8: ("invariant_axis_difference_elided",),
    9: ("zero_rows_accurate_always", "zero_rows_accurate_inverted(host)",
        "m0_bx_axis_zero_dropped", "m0_dz_axis_add_dropped",
        "m0_dy_axis_zero_dropped", "m0_dz_axis_add_uses_dtdx"),
}


def needle_layer_map() -> Dict[str, Dict[str, bool]]:
    """Where each needle name lives, across the three batteries.

    The gate's own answer to "does the kernel battery mirror the reference one?"
    — the honest form is a MAP: every kernel-source needle must have a reference
    counterpart of the same name, and the reference battery may carry more.
    :func:`check_needle_layers` turns that into a pass/fail.
    """
    reference = {name for name, *_rest in REFERENCE_MUTATIONS}
    kernel = {name for name, *_rest in MUTATIONS}
    host = {name for name, *_rest in HOST_MUTATIONS}
    constitutive = {name for name, *_rest in CONSTITUTIVE_MUTATIONS}
    out: Dict[str, Dict[str, bool]] = {}
    for name in sorted(reference | kernel | host | constitutive):
        out[name] = {"reference": name in reference, "kernel": name in kernel,
                     "host": name in host, "constitutive": name in constitutive}
    return out


def check_needle_layers() -> List[str]:
    """Problems with the needle map, as reasons. Empty is the pass."""
    problems: List[str] = []
    layers = needle_layer_map()
    for name, where in layers.items():
        if where["kernel"] and not where["reference"]:
            problems.append(
                f"{name}: armed on the KERNEL source with no reference "
                f"counterpart, so nothing on the laptop can say whether the "
                f"needle is live before a GPU is spent on it")
    for choice, needles in GROUPING_CHOICE_NEEDLES.items():
        for needle in needles:
            bare = needle.split("(")[0]
            if bare not in layers:
                problems.append(
                    f"grouping choice {choice} names needle {needle!r}, which is "
                    f"in no battery")
    return problems


def run_reference_mutations(out_path: Optional[str], results: Dict[str, Any],
                            steps: int) -> Dict[str, Any]:
    """Arm every needle against the REFERENCE, on the laptop, before a GPU is spent.

    DISARMED (the pattern did not match) and NEEDLE-MISSED (it matched, ran and
    moved nothing) are BOTH failures. Predicted nulls are recorded with their
    reason. The ``caught_zero_init_only`` prediction is checked in BOTH seeding
    modes and must be missed under random seeds and caught under zero-init — a
    row that is caught under both would mean the zero-init case is not measuring
    what it claims to.
    """
    source_path = os.path.abspath(__file__)
    source = open(source_path, "r", encoding="utf-8").read()
    report: Dict[str, Any] = {"rows": [], "source": source_path}

    def evaluate(module, zero_init: bool) -> int:
        total = 0
        for m, accurate in ((0, False), (1, False), (-1, False), (2, False),
                            (-3, False), (3, True)):
            for z_metallic in Z_METALLIC:
                for courant in (0.37, 0.3141592653589793):
                    if accurate and courant > 1.0 / (abs(m) + 0.5):
                        continue
                    for sub_step in SUB_STEPS:
                        pml_cells = (ZERO_INIT_PML_CELLS if zero_init
                                     else SEEDED_PML_CELLS)
                        grid_a, fields_a, pml_a = build_engine(
                            np, (20, 1, 24), m, courant, z_metallic, accurate,
                            pml_cells)
                        _g, fields_b, _p = build_engine(
                            np, (20, 1, 24), m, courant, z_metallic, accurate,
                            pml_cells)
                        seed_state(np, fields_a, (20, 1, 24), zero_init)
                        seed_state(np, fields_b, (20, 1, 24), zero_init)
                        dtdx = grid_a.dt / grid_a.dx
                        boundaries = stepping._boundary_kinds(grid_a, pml_a)
                        coefficients = coefficient_dict(
                            pml_a, half_integer=(sub_step == "step_B"))
                        reference_fields = {name: getattr(fields_b, name)
                                            for name in COMPARED + SOURCES}
                        try:
                            for _ in range(steps):
                                (stepping.step_B if sub_step == "step_B"
                                 else stepping.step_D)(fields_a, pml_a)
                                module.reference_cyl_complex_step(
                                    np, sub_step, reference_fields, coefficients,
                                    dtdx, boundaries, int(m), bool(accurate))
                                # PER SUB-STEP: a signed-zero divergence can cancel
                                # out by the next step, so an end-of-run compare is
                                # blind (measured).
                                for name in COMPARED:
                                    found = first_difference(
                                        getattr(fields_a, name),
                                        getattr(fields_b, name))
                                    total += 0 if found is None else found["differing_words"]
                        except Exception:  # noqa: BLE001 - a raise is a catch
                            total += 1
        return total

    clean = _load_module(source, "cylcomplex_reference_clean")
    control = {"random": evaluate(clean, False), "zero_init": evaluate(clean, True)}
    report["clean_control"] = control
    report["minuend_census"] = run_minuend_census(steps)
    log(f"i*m/r minuend census: {report['minuend_census']}")
    log(f"clean control: random={control['random']} zero_init={control['zero_init']} "
        f"(both must be 0)")
    if control["random"] or control["zero_init"]:
        report["verdict"] = "VOID: the clean control is not identical"
        results["reference_mutations"] = report
        save(results, out_path)
        return report

    for name, pattern, replacement, predicted in REFERENCE_MUTATIONS:
        mutated = mutate_region(source, "reference", pattern, replacement)
        if mutated is None:
            report["rows"].append(
                {"name": name, "status": "DISARMED",
                 "detail": f"pattern not found in the reference CODE region "
                           f"(it may still occur inside the table): {pattern!r}"})
            log(f"{name:32s} DISARMED")
            continue
        module = _load_module(mutated, "cylcomplex_reference_" + name)
        moved = {"random": evaluate(module, False),
                 "zero_init": evaluate(module, True)}
        wants_null = isinstance(predicted, tuple) and predicted[0] == "null"
        zero_only = isinstance(predicted, tuple) and predicted[0] == "caught_zero_init_only"
        if wants_null:
            status = ("NULL-AS-PREDICTED" if not any(moved.values())
                      else "UNEXPECTEDLY-CAUGHT")
        elif zero_only:
            status = ("CAUGHT-ZERO-INIT-ONLY"
                      if moved["zero_init"] > 0 and moved["random"] == 0
                      else "PREDICTION-BROKEN")
        else:
            status = "CAUGHT" if any(moved.values()) else "NEEDLE-MISSED"
        row = {"name": name, "status": status, "differing_words": moved,
               "predicted": predicted if isinstance(predicted, str) else list(predicted)}
        report["rows"].append(row)
        log(f"{name:32s} {status:24s} random={moved['random']} "
            f"zero_init={moved['zero_init']}")
        results["reference_mutations"] = report
        save(results, out_path)

    # PER-SEEDING VACUITY. A seeding that catches nothing across the WHOLE
    # battery cannot fail for any reason connected to the arithmetic, whatever
    # its individual rows say. This is the check that would have caught 256
    # zero-init constitutive rows reporting a pass on an untouched state.
    per_seeding = {"random": 0, "zero_init": 0}
    for row in report["rows"]:
        for seeding in per_seeding:
            per_seeding[seeding] += int(row.get("differing_words", {}).get(seeding, 0))
    report["per_seeding_total_words"] = per_seeding
    vacuous = [seeding for seeding, total in per_seeding.items() if total == 0]
    report["vacuous_seedings"] = vacuous

    failures = [row["name"] for row in report["rows"]
                if row["status"] in ("DISARMED", "NEEDLE-MISSED",
                                     "UNEXPECTEDLY-CAUGHT", "PREDICTION-BROKEN")]
    report["failed_legs"] = failures
    report["verdict"] = ("all needles live" if not failures and not vacuous
                         else "FAILED")
    results["reference_mutations"] = report
    save(results, out_path)
    log(f"reference mutations: {report['verdict']} per_seeding={per_seeding}"
        + (f" vacuous={vacuous}" if vacuous else "")
        + (f" -> {failures}" if failures else ""))
    return report


def run_minuend_census(steps: int, xp=None) -> Dict[str, Any]:
    """How often the i*m/r MINUEND carries an exact ``-0.0`` word.

    THE MEASUREMENT BEHIND TWO PREDICTED NULLS. The plane-wise shortcut and the
    full complex product differ only in the ±0 class, and ``curl - m`` can carry
    that difference to a stored word only where the minuend itself is -0.0. This
    counts them, over both seedings and both sub-steps, rather than arguing it.
    A NONZERO count here would mean the ``imr_planewise_zero_cross_terms`` null
    is no longer predicted and the needle must be re-armed as a catch.

    ``xp`` selects the backend. The laptop leg measures it on NumPy; the device
    leg RE-MEASURES it on CuPy, because the null this census is the evidence for
    is asserted about the device bytes and a class that is unreachable in NumPy
    arithmetic is not thereby unreachable in the array path CuPy runs. Same
    enumeration, same budget, so the two numbers are comparable.
    """
    module = np if xp is None else xp
    census: Dict[str, int] = {}
    for zero_init in (False, True):
        # m = 0 is NOT in this census by construction: the array path forms no
        # i*m/r minuend there (stepping :348/:430) and the kernel compiles the
        # block out, so there is nothing to scan.
        for m in (1, -1, 2, -3):
            for z_metallic in Z_METALLIC:
                for sub_step in SUB_STEPS:
                    pml_cells = (ZERO_INIT_PML_CELLS if zero_init
                                 else SEEDED_PML_CELLS)
                    grid, fields, pml = build_engine(
                        module, (20, 1, 24), m, 0.37, z_metallic, False,
                        pml_cells)
                    seed_state(module, fields, (20, 1, 24), zero_init)
                    dtdx = grid.dt / grid.dx
                    boundaries = stepping._boundary_kinds(grid, pml)
                    coefficients = coefficient_dict(
                        pml, half_integer=(sub_step == "step_B"))
                    volumes = {name: getattr(fields, name)
                               for name in COMPARED + SOURCES}
                    for _ in range(steps):
                        reference_cyl_complex_step(
                            module, sub_step, volumes, coefficients, dtdx,
                            boundaries, int(m), False, minuend_census=census)
    scanned = census.pop("_words_scanned", 0)
    return {"backend": "numpy" if module is np else "cupy",
            "negative_zero_minuend_words": sum(census.values()),
            "by_target": census, "words_scanned": scanned,
            "reading": ("0 means the ±0 class cannot reach a stored word through "
                        "the i*m/r fold, which is what makes "
                        "imr_planewise_zero_cross_terms a PREDICTED null")}


# ===========================================================================
# The two claims the module docstring makes about the KERNEL'S SHAPE
# ===========================================================================
#
# Both were prose with a word count and no harness. They are legs now, because a
# number in a docstring that no command reproduces is indistinguishable from a
# number somebody remembered.

AXIS_IDENTITY_M: Tuple[int, ...] = (0, 1, -1, 2, -3, 5)
AXIS_IDENTITY_DTDX: Tuple[float, ...] = (0.37, 0.3141592653589793)


def _identity_pair(force_r_metallic: bool, strip: Sequence[Tuple[str, str]],
                   shape, m, courant, z_metallic, sub_step, zero_init, steps):
    """Run the reference twice — as shipped, and with ``strip`` applied — and
    count the differing uint32 words per sub-step."""
    source = open(os.path.abspath(__file__), "r", encoding="utf-8").read()
    mutated = source
    if force_r_metallic:
        mutated = mutated.replace(
            '    backward = sub_step == "step_D"',
            '    boundaries = ("metallic", boundaries[1], boundaries[2])\n'
            '    backward = sub_step == "step_D"', 1)
    applied = 0
    for old, new in strip:
        if old in mutated:
            mutated = mutated.replace(old, new)
            applied += 1
    module = _load_module(mutated, f"cylcomplex_variant_{abs(hash(mutated)) % 10**8}")

    pml_cells = ZERO_INIT_PML_CELLS if zero_init else SEEDED_PML_CELLS
    grid_a, fields_a, pml_a = build_engine(np, shape, m, courant, z_metallic,
                                           False, pml_cells)
    _g, fields_b, _p = build_engine(np, shape, m, courant, z_metallic, False,
                                    pml_cells)
    seed_state(np, fields_a, shape, zero_init)
    seed_state(np, fields_b, shape, zero_init)
    dtdx = grid_a.dt / grid_a.dx
    boundaries = stepping._boundary_kinds(grid_a, pml_a)
    coefficients = coefficient_dict(pml_a, half_integer=(sub_step == "step_B"))
    left = {name: getattr(fields_a, name) for name in COMPARED + SOURCES}
    right = {name: getattr(fields_b, name) for name in COMPARED + SOURCES}
    differing = 0
    for _ in range(steps):
        reference_cyl_complex_step(np, sub_step, left, coefficients, dtdx,
                                   boundaries, int(m), False)
        module.reference_cyl_complex_step(np, sub_step, right, coefficients,
                                          dtdx, boundaries, int(m), False)
        for name in COMPARED:
            found = first_difference(left[name], right[name])
            differing += 0 if found is None else found["differing_words"]
    return differing, applied


def run_axis_identity(out_path: Optional[str], results: Dict[str, Any],
                      steps: int) -> Dict[str, Any]:
    """THE r-AXIS IDENTITY: ``BCX = METALLIC`` costs the kernel no branch.

    ``stepping._boundary_kinds`` returns CYL_AXIS on r; ``_shift_up``'s CYL_AXIS
    arm IS the metallic one (a hard zero at the far face, :1781-1783), and
    ``_shift_down``'s writes the ``r_to_minus_r`` image into a near ghost the
    ownership mask makes unobservable. This runs the reference with
    ``boundaries[0]`` forced to ``'metallic'`` — the certified kernel's own
    constexpr arm — against the reference as shipped. 0 differing words licenses
    binding :data:`cylindrical_complex.BCX` to METALLIC; anything else means the
    axis needs a branch after all.
    """
    rows: List[Dict[str, Any]] = []
    for m in AXIS_IDENTITY_M:
        for z_metallic in Z_METALLIC:
            for zero_init in (False, True):
                for courant in AXIS_IDENTITY_DTDX:
                    for sub_step in SUB_STEPS:
                        differing, _applied = _identity_pair(
                            True, (), (20, 1, 24), m, courant, z_metallic,
                            sub_step, zero_init, steps)
                        rows.append({"m": int(m), "z": "metallic" if z_metallic
                                     else "periodic", "zero_init": zero_init,
                                     "courant": courant, "sub_step": sub_step,
                                     "differing_words": differing})
    record = {"rows": rows, "total_rows": len(rows), "steps_per_row": steps,
              "differing_words": sum(r["differing_words"] for r in rows),
              "identical_rows": sum(r["differing_words"] == 0 for r in rows),
              "claim": "0 differing words licenses BCX = METALLIC for the r axis"}
    results["axis_identity"] = record
    save(results, out_path)
    log(f"axis identity (BCX = METALLIC): {record['identical_rows']}/"
        f"{record['total_rows']} rows identical, "
        f"{record['differing_words']} differing words")
    return record


#: Every cylindrical clause, switched off. What remains is the arithmetic the
#: CERTIFIED ``complex_fields.bloch_pml_curl_step`` computes on the same grid.
STRIP_CYLINDRICAL: Tuple[Tuple[str, str], ...] = (
    ("curl = curl + -(imr_product(factor, snap[partner]))", "curl = curl"),
    ("curl = dtdx * (extended_prefix[1:] - extended_prefix[:-1])", "curl = curl"),
    ('src["Hy"] = prefix ', "pass  #"),
    ("curl[_face(0, 0)] = -axis_increment[1].astype(curl.dtype)", "pass"),
    ('fields["Dz"][_face(0, 0)] = 0', "pass"),
    ("near = slice(0, 1) if accurate else slice(0, abs(m))", "near = slice(0, 0)"),
    # The m = 0 axis pair (2026-09-04).
    ('fields["Dz"][_face(0, 0)] += (4.0 * dtdx) * fields["Hy"][_face(0, 0)]', "pass"),
    ('fields["Dy"][_face(0, 0)] = 0', "pass"),
    ('fields["Bx"][_face(0, 0)] = 0', "pass"),
)


def run_stripped_control(out_path: Optional[str], results: Dict[str, Any],
                         steps: int) -> Dict[str, Any]:
    """THE CURL HALF DOES NEED A NEW KERNEL — measured, not assumed.

    The complement of :func:`run_axis_identity`: the certified complex body's
    arithmetic (r forced metallic, EVERY cylindrical clause stripped) against the
    array path over the same matrix. A large word count is the evidence that
    delegating the curl to ``complex_fields`` would be silently wrong, which is
    the claim the whole tranche rests on. A count of ZERO here would mean this
    module should not exist.
    """
    rows: List[Dict[str, Any]] = []
    for m in AXIS_IDENTITY_M:
        for z_metallic in Z_METALLIC:
            for zero_init in (False, True):
                for courant in AXIS_IDENTITY_DTDX:
                    for sub_step in SUB_STEPS:
                        differing, applied = _identity_pair(
                            True, STRIP_CYLINDRICAL, (20, 1, 24), m, courant,
                            z_metallic, sub_step, zero_init, steps)
                        rows.append({"m": int(m), "z": "metallic" if z_metallic
                                     else "periodic", "zero_init": zero_init,
                                     "courant": courant, "sub_step": sub_step,
                                     "clauses_stripped": applied,
                                     "differing_words": differing})
    record = {"rows": rows, "total_rows": len(rows), "steps_per_row": steps,
              "differing_words": sum(r["differing_words"] for r in rows),
              "rows_that_differ": sum(r["differing_words"] > 0 for r in rows),
              "claim": "a nonzero count is what makes this tranche a NEW kernel "
                       "rather than a widened predicate over the certified one"}
    results["stripped_control"] = record
    save(results, out_path)
    log(f"stripped complex control: {record['differing_words']} differing words "
        f"over {record['total_rows']} rows ({record['rows_that_differ']} rows differ)")
    if record["differing_words"] == 0:
        raise AssertionError(
            "the certified complex body reproduces the cylindrical array path "
            "exactly — this module would then be unnecessary, which contradicts "
            "its own reason for existing")
    return record


# ===========================================================================
# Needle arming — on the LAPTOP, before a GPU is spent
# ===========================================================================

def shipped_kernel_source() -> str:
    """The kernel's source text, readable WITHOUT Triton.

    With Triton absent the module's kernel is an ``_UnavailableKernel`` and has no
    ``.fn``, so the text is taken from the module FILE by AST segment. That the
    laptop can read it is the whole point: a rename that disarms a kernel needle
    costs nothing to find here and costs a device run to find there.
    """
    from meep_gpu.triton_kernels import cylindrical_complex as cyl  # noqa: PLC0415

    function = getattr(cyl.cyl_complex_pml_curl_step, "fn", None)
    if function is not None:
        text = textwrap.dedent(inspect.getsource(function))
    else:
        path = cyl.__file__
        source = open(path, "r", encoding="utf-8").read()
        segment = None
        for node in ast.parse(source).body:
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "cyl_complex_pml_curl_step"):
                segment = ast.get_source_segment(source, node)
        if segment is None:
            raise AssertionError(
                "cyl_complex_pml_curl_step is not a module-level function any "
                "more; the kernel mutation battery cannot be armed")
        text = textwrap.dedent(segment)
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("@")) + "\n"


def run_needle_arming(out_path: Optional[str], results: Dict[str, Any]
                      ) -> Dict[str, Any]:
    """Every needle in every battery must MATCH ITS TARGET, checked on NumPy.

    THE MISSING GUARD the DISARMED status exists for. The kernel battery's
    transforms were never applied anywhere on the laptop, so a rename inside
    ``cyl_complex_pml_curl_step`` would have surfaced as DISARMED only after GPU
    time was spent. Applying them here costs milliseconds. The HOST battery is
    armed the same way: its entries carry callables now, and each one must
    actually change either a patched attribute or a plan argument.
    """
    from meep_gpu.triton_kernels import cylindrical_complex as cyl  # noqa: PLC0415

    kernel_source = shipped_kernel_source()
    gate_source = open(os.path.abspath(__file__), "r", encoding="utf-8").read()
    rows: List[Dict[str, Any]] = []

    for name, transform, _predicted in MUTATIONS:
        mutated, hits = transform(kernel_source)
        status = ("DISARMED" if hits == 0 else
                  "NO-OP" if mutated == kernel_source else "ARMED")
        rows.append({"battery": "kernel", "name": name, "hits": hits,
                     "status": status})

    # THROUGH ``mutate_region``, not through ``in source``: a pattern that occurs
    # only inside its own mutation TABLE would pass a substring test and then
    # apply to a string literal, which is how a disarmed needle came to report
    # NEEDLE-MISSED instead of DISARMED. The arming check has to ask the same
    # question the driver asks.
    for battery, table in (("reference", REFERENCE_MUTATIONS),
                           ("constitutive", CONSTITUTIVE_MUTATIONS)):
        for name, pattern, replacement, _predicted in table:
            applied = mutate_region(gate_source, battery, pattern, replacement)
            rows.append({"battery": battery, "name": name,
                         "hits": gate_source.count(pattern),
                         "status": ("DISARMED" if applied is None else
                                    "NO-OP" if applied == gate_source else "ARMED")})

    context = {"sub_step": "step_B", "half_integer": True, "bcz": 0,
               "expansion": 1, "m": -1, "accurate": False, "dtdx": 0.37}
    for name, _description, _predicted, apply in HOST_MUTATIONS:
        patches, overrides = apply(cyl, dict(context))
        changed = bool(overrides) and any(
            context.get(key) != value for key, value in overrides.items())
        for obj, attribute, replacement in patches:
            changed = changed or getattr(obj, attribute) is not replacement
        rows.append({"battery": "host", "name": name,
                     "patches": [attribute for _o, attribute, _r in patches],
                     "overrides": sorted(overrides),
                     "status": "ARMED" if changed else "DISARMED"})

    layer_problems = check_needle_layers()
    disarmed = [row["name"] for row in rows if row["status"] != "ARMED"]
    record = {"rows": rows, "disarmed": disarmed,
              "needle_layers": needle_layer_map(),
              "layer_problems": layer_problems,
              "grouping_choice_needles": {str(k): list(v) for k, v
                                          in GROUPING_CHOICE_NEEDLES.items()},
              "verdict": "all needles armed" if not disarmed and not layer_problems
                         else "FAILED"}
    results["needle_arming"] = record
    save(results, out_path)
    log(f"needle arming: {record['verdict']} "
        f"({len(rows)} needles, disarmed={disarmed or 'none'}, "
        f"layer problems={layer_problems or 'none'})")
    if disarmed or layer_problems:
        raise AssertionError(f"needles disarmed: {disarmed}; "
                             f"layer problems: {layer_problems}")
    return record


#: Where each battery's TABLE begins. Everything before it is the region a
#: mutation of that battery may edit.
TABLE_MARKERS = {"reference": "REFERENCE_MUTATIONS: Tuple[",
                 "constitutive": "CONSTITUTIVE_MUTATIONS: Tuple["}


def mutate_region(source: str, battery: str, pattern: str,
                  replacement: str) -> Optional[str]:
    """Apply one mutation to the CODE, refusing to edit the mutation TABLE.

    A HARNESS-INTEGRITY CHECK, and it is here because it fired: after the i*m/r
    fold was rewritten to call ``imr_product``, two entries still carried the old
    spelling — which still occurred in the file, inside the table's own tuple.
    ``source.replace(pattern, replacement, 1)`` then edited a STRING LITERAL, the
    reference was untouched, and the leg reported NEEDLE-MISSED: a DISARMED
    needle wearing the mask of a measured null. Restricting the edit to the code
    region above the table makes that failure mode report itself as DISARMED,
    which is what it is.

    Returns None when the pattern does not occur in the code region.
    """
    cut = source.index(TABLE_MARKERS[battery])
    head, tail = source[:cut], source[cut:]
    if pattern not in head:
        return None
    return head.replace(pattern, replacement, 1) + tail


def _load_module(source: str, name: str):
    """Compile one mutated copy of THIS file as a throwaway module.

    The whole file is compiled, not a fragment, so a mutation cannot pass by
    landing in a helper the reference does not call.
    """
    module = types.ModuleType(name)
    module.__dict__["__file__"] = os.path.abspath(__file__)
    module.__dict__["__name__"] = name
    exec(compile(source, name, "exec"), module.__dict__)  # noqa: S102
    return module


def _reference_row(shape, m, accurate, courant, z_metallic, sub_step, steps):
    """One reference-vs-array-path row, compared PER SUB-STEP."""
    grid_a, fields_a, pml_a = build_engine(np, shape, m, courant, z_metallic,
                                           accurate, SEEDED_PML_CELLS)
    _grid_b, fields_b, _pml_b = build_engine(np, shape, m, courant, z_metallic,
                                             accurate, SEEDED_PML_CELLS)
    seed_state(np, fields_a, shape, False)
    seed_state(np, fields_b, shape, False)
    dtdx = grid_a.dt / grid_a.dx
    boundaries = stepping._boundary_kinds(grid_a, pml_a)
    coefficients = coefficient_dict(pml_a, half_integer=(sub_step == "step_B"))
    reference_fields = {name: getattr(fields_b, name) for name in COMPARED + SOURCES}
    divergence = None
    for step in range(steps):
        (stepping.step_B if sub_step == "step_B" else stepping.step_D)(fields_a, pml_a)
        reference_cyl_complex_step(np, sub_step, reference_fields, coefficients,
                                   dtdx, boundaries, int(m), bool(accurate))
        if divergence is None:
            for name in COMPARED:
                found = first_difference(getattr(fields_a, name),
                                         getattr(fields_b, name))
                if found is not None:
                    divergence = {"step": step, "volume": name, **found}
                    break
        if divergence is not None:
            break
    return {"shape": list(shape), "m": int(m), "accurate": bool(accurate),
            "courant": float(courant),
            "z": "metallic" if z_metallic else "periodic", "sub_step": sub_step,
            "boundaries": list(boundaries), "steps": steps,
            "first_divergence": divergence}


# ===========================================================================
# THE DEVICE HALF
# ===========================================================================
#
# Written to RUN, not to be described. The first cut of this file had a prose
# list of device legs, a mutation battery with no driver and a ``main`` that
# returned 2 — the legs it advertised did not exist as code at all. Everything
# below is reachable from :func:`main`, and every leg refuses rather than
# reporting a hollow pass:
#
#   * a plan that never launched is NO-LAUNCH, never "identical" (the engine
#     route did exactly that: no probe artifact was passed, ``plan`` was None on
#     every row, and every row recorded ``first_divergence: None``);
#   * a mutant whose PTX equals a shipped specialization's is STALE-BINARY —
#     Triton's cache can serve the wrong binary to a renamed mutant;
#   * a mutation that matched nothing is DISARMED; one that moved no byte where a
#     catch was declared is NEEDLE-MISSED. All three are failures.

#: Sub-steps per device row, and the SEPARATE consecutive budget. Bit-identity is
#: claimable for exactly the number the artifact records and no further.
DEVICE_STEPS = 8
MULTI_STEP_BUDGET = 64

#: Block A varies every arm at one shape; block B varies the SHAPE at one arm of
#: each |m| class. The product of both would be 2,560 rows for no new coverage —
#: an index swap shows on any shape whose extents differ, and the arms are
#: independent of the shape.
SYNTHETIC_SHAPE: Tuple[int, int, int] = (20, 1, 24)
SHAPE_BLOCK_M: Tuple[Tuple[int, bool], ...] = ((0, False), (-1, False), (3, False))
SHAPE_BLOCK_DTDX: Tuple[float, ...] = (0.5, 0.37)

#: The configurations a kernel-source mutation is measured over: both |m|
#: classes, both z terminations, both sub-steps and both seedings, at a
#: non-power-of-two Courant. Small on purpose — each row recompiles nothing, but
#: each mutation is a fresh Triton compile.
MUTATION_CONFIGS: Tuple[Dict[str, Any], ...] = tuple(
    {"m": m, "accurate": False, "courant": 0.37, "z_metallic": z,
     "sub_step": sub_step, "zero_init": zero_init}
    for m in (0, -1, 3) for z in (True, False)
    for sub_step in SUB_STEPS for zero_init in (False, True))


def device_status() -> Tuple[bool, str]:
    """Is a device leg runnable here, and if not, exactly why."""
    if cp is None:
        return False, "no cupy on this host"
    if not _TRITON_AVAILABLE:
        return False, "no triton on this host"
    return True, "cupy and triton present"


def probe_record(path: Optional[str]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """The measured expansion artifact, plus the reasons it may not be used.

    The kernel binds ``EXPANSION`` from a MEASURED platform probe. A device leg
    that guesses it is not a byte gate, and one that runs without it launches
    nothing — which is the shape the engine route failed in.
    """
    from meep_gpu.triton_kernels import complex_fields as cx  # noqa: PLC0415

    record = cx.load_expansion_probe(path)
    if record is None:
        return None, [f"no expansion probe artifact at {path!r}; the EXPANSION "
                      f"constexpr is a measured platform fact and may not be guessed"]
    reasons = list(shared.probe_record_policy_reasons(record)) if hasattr(
        shared, "probe_record_policy_reasons") else []
    if cx.expansion_from_probe(record) is None:
        reasons.append("the probe artifact licenses no single EXPANSION constexpr")
    return (record if not reasons else None), reasons


def _curl_rows(xp, expansion, steps, guard, route, probe=None, kernel=None,
               host_mutation=None, configs=None) -> List[Dict[str, Any]]:
    """Block A + block B, or an explicit config list."""
    rows: List[Dict[str, Any]] = []
    if configs is None:
        configs = []
        for m, accurate in M_CASES:
            for courant in DTDX:
                if accurate and courant > 1.0 / (abs(m) + 0.5):
                    continue
                for z_metallic in Z_METALLIC:
                    for sub_step in SUB_STEPS:
                        for zero_init in (False, True):
                            configs.append({"shape": SYNTHETIC_SHAPE, "m": m,
                                            "accurate": accurate,
                                            "courant": courant,
                                            "z_metallic": z_metallic,
                                            "sub_step": sub_step,
                                            "zero_init": zero_init})
        for shape in SHAPES:
            for m, accurate in SHAPE_BLOCK_M:
                for courant in SHAPE_BLOCK_DTDX:
                    for z_metallic in Z_METALLIC:
                        for sub_step in SUB_STEPS:
                            configs.append({"shape": shape, "m": m,
                                            "accurate": accurate,
                                            "courant": courant,
                                            "z_metallic": z_metallic,
                                            "sub_step": sub_step,
                                            "zero_init": False})
    total = len(configs)
    for index, config in enumerate(configs, 1):
        row = one_curl_case(xp, config.get("shape", SYNTHETIC_SHAPE), config["m"],
                            config["accurate"], config["courant"],
                            config["z_metallic"], config["sub_step"],
                            config["zero_init"], guard, steps, expansion,
                            kernel=kernel, route=route, probe=probe,
                            host_mutation=host_mutation)
        row["verdict"] = verdict_for_curl_row(row)
        rows.append(row)
        log(f"[{index}/{total}] {route} m={config['m']:+d} "
            f"C={config['courant']:.6g} z={'M' if config['z_metallic'] else 'P'} "
            f"{config['sub_step']} zero_init={int(config['zero_init'])} "
            f"guard={guard} -> {row['verdict']} (launches={row['launches']})")
    return rows


def _summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    reach: Dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
        if row.get("zero_init"):
            key = row.get("signed_zero_reach", "unclassified")
            reach[key] = reach.get(key, 0) + 1
    return {"total_rows": len(rows), "by_verdict": counts,
            "zero_init_signed_zero_reach": reach,
            "identical_rows": counts.get("IDENTICAL", 0),
            "total_launches": sum(row["launches"] for row in rows),
            # Reported, never assumed: a leg that compared raw words through a
            # NaN certified nothing (platform fact (f)).
            "nan_census": {
                "nan_words": sum(row.get("nan_census", {}).get("nan_words", 0)
                                 for row in rows),
                "inf_words": sum(row.get("nan_census", {}).get("inf_words", 0)
                                 for row in rows)}}


def run_synthetic(out_path: Optional[str], results: Dict[str, Any],
                  expansion: int, steps: int) -> Dict[str, Any]:
    """The kernel through ``plan_..._from_arrays`` against the in-gate reference."""
    rows = _curl_rows(cp, expansion, steps, None, "synthetic")
    assert_leg_reaches_signed_zero("synthetic", rows)
    record = {"rows": rows, **_summarize_rows(rows), "steps_per_row": steps,
              "guard": "as shipped (ENABLE_FP_FUSION)"}
    # THE MINUEND CENSUS, RE-MEASURED ON THIS HOST'S CuPy ARRAYS. It is the
    # evidence for the imr_planewise_zero_cross_terms predicted null, and that
    # null is asserted about device bytes — so the number the artifact carries
    # beside it may not be a laptop number. Same enumeration and budget as the
    # laptop leg, so the two are comparable rather than merely both present.
    record["minuend_census_device"] = run_minuend_census(steps, xp=cp)
    log(f"[synthetic] device i*m/r minuend census: "
        f"{record['minuend_census_device']}")
    results["synthetic"] = record
    save(results, out_path)
    log(f"[SUMMARY] synthetic {record['identical_rows']}/{record['total_rows']} "
        f"IDENTICAL, {record['total_launches']} launches, "
        f"nan_census={record['nan_census']}")
    return record


def run_guard(out_path: Optional[str], results: Dict[str, Any], expansion: int,
              steps: int) -> Dict[str, Any]:
    """The same rows at ``enable_fp_fusion=True``, reported as DATA.

    ``enable_fp_fusion`` is NOT byte-uniform across tranches — measured: complex
    68/68 fusion-on rows identical, special_kz 24/96, nonlinear 0/108, offdiag
    0/28, bfast 0/52. This family therefore CERTIFIES UNDER THE SHIPPED VALUE and
    records the fusion-on count rather than asserting a number it has not
    measured. An earlier draft of this file demanded the control "must BITE",
    which the complex tranche's own 68/68 shows is not a law.
    """
    rows = _curl_rows(cp, expansion, steps, True, "synthetic")
    record = {"rows": rows, **_summarize_rows(rows), "steps_per_row": steps,
              "guard": True,
              "reading": "DATA, not an assertion: this family certifies under the "
                         "shipped ENABLE_FP_FUSION and states the fusion-on count"}
    results["guard"] = record
    save(results, out_path)
    log(f"[SUMMARY] fusion-on control {record['identical_rows']}/"
        f"{record['total_rows']} identical (data, not a pass/fail)")
    return record


def word_class(word: str) -> str:
    """``zero`` / ``subnormal`` / ``normal`` for one float32 word, from its hex.

    PURE. The multi-step leg's verdict turns on it: a divergence between an
    exact zero and a subnormal is the FLUSH class, which is a property of the
    two compilers' subnormal handling at the bottom of a decaying run, and a
    divergence between two normal words is arithmetic. Conflating them would let
    a real defect hide behind an inherited exposure.
    """
    value = int(word, 16) & 0x7FFFFFFF
    if value == 0:
        return "zero"
    return "subnormal" if (value >> 23) == 0 else "normal"


def divergence_class(divergence: Optional[Dict[str, Any]]) -> Optional[str]:
    """``subnormal_flush`` when one side is an exact zero and the other a
    subnormal; ``arithmetic`` for anything else."""
    if divergence is None:
        return None
    classes = {word_class(divergence["left"]), word_class(divergence["right"])}
    return ("subnormal_flush" if classes == {"zero", "subnormal"}
            else "arithmetic")


def run_multi_step(out_path: Optional[str], results: Dict[str, Any],
                   expansion: int) -> Dict[str, Any]:
    """Consecutive sub-steps to a STATED budget, compared per sub-step.

    THIS LEG MEASURES A BUDGET; it does not assert one. The budget in the
    artifact is the only budget claimable, and the exposure that ends it is
    INHERITED and named in advance: ``cylindrical_triton``'s own consecutive leg
    first diverged at step 23-72 on a SUBNORMAL AGAINST A FLUSHED ZERO, and this
    family shares that module's array-path prefix and the same recurrence.

    So the leg's verdict is not "64 of 64 identical". It is:

    * every first divergence must be of the SUBNORMAL-FLUSH class — one side an
      exact zero, the other a subnormal — because that is the inherited
      exposure, and an ARITHMETIC divergence at any step is a real defect that
      must not be absorbed into a shortened budget;
    * the claimable budget must cover the budget the REST of the gate claims
      (``DEVICE_STEPS``), or the sub-step legs are claiming more consecutive
      launches than the consecutive leg supports.

    MEASURED on the GPU host 2026-08-13: 4 of 8 rows ran the whole 64 sub-steps; the
    other 4 first differ at sub-step 21 (``fu_Dy``, 2 words) and 41 (``fu_Bz``,
    1 word), every one of them a subnormal (``0x0033df30``) against an exact
    ``0x0`` — the predicted class, on the z-METALLIC rows, whose absorber is what
    decays the state into the denormal range. Claimable budget 20, against a
    sub-step claim of 8.
    """
    configs = [{"shape": SYNTHETIC_SHAPE, "m": m, "accurate": False,
                "courant": 0.37, "z_metallic": z, "sub_step": sub_step,
                "zero_init": False}
               for m in (0, -1, 3) for z in (True, False) for sub_step in SUB_STEPS]
    rows = _curl_rows(cp, expansion, MULTI_STEP_BUDGET, None, "synthetic",
                      configs=configs)
    first = [row["first_divergence"]["step"] for row in rows
             if row["first_divergence"] is not None]
    for row in rows:
        row["divergence_class"] = divergence_class(row["first_divergence"])
    arithmetic = [row for row in rows
                  if row["divergence_class"] == "arithmetic"]
    claimable = MULTI_STEP_BUDGET if not first else min(first) - 1
    failed: List[str] = []
    if arithmetic:
        failed.append(
            f"{len(arithmetic)} row(s) diverged ARITHMETICALLY, not on the "
            f"inherited subnormal-flush class: "
            f"{[row['first_divergence'] for row in arithmetic][:3]}")
    if claimable < DEVICE_STEPS:
        failed.append(
            f"claimable budget {claimable} is below the {DEVICE_STEPS} "
            f"consecutive sub-steps the other device legs claim")
    record = {"rows": rows, **_summarize_rows(rows),
              "budget": MULTI_STEP_BUDGET,
              "first_divergence_steps": sorted(first),
              "divergence_classes": sorted(
                  {row["divergence_class"] for row in rows
                   if row["divergence_class"]}),
              "claimable_budget": claimable,
              "failed_legs": failed,
              "verdict": ("budget measured: bit-identity is claimable for "
                          f"{claimable} consecutive sub-steps and no further"
                          if not failed else f"FAILED: {failed}")}
    results["multi_step"] = record
    save(results, out_path)
    log(f"[SUMMARY] multi-step {record['identical_rows']}/{record['total_rows']} "
        f"identical over {MULTI_STEP_BUDGET} sub-steps; classes="
        f"{record['divergence_classes']}; claimable budget "
        f"{record['claimable_budget']}; {record['verdict']}")
    return record


def run_engine(out_path: Optional[str], results: Dict[str, Any],
               probe: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """The ENGINE route: the SHIPPED predicate has to admit the run.

    Everything the synthetic route bypasses is under test here — the predicate's
    verdict, the prefix built from the engine's own ``StepScratch``, and the
    sub-lattice binding taken from the real ``PML``. ``covered`` and ``launches``
    are asserted, not recorded: a refusal here is a FAILURE of the engine leg,
    where in production it would merely mean falling back to the array path.
    """
    configs = [{"shape": SYNTHETIC_SHAPE, "m": m, "accurate": accurate,
                "courant": courant, "z_metallic": z, "sub_step": sub_step,
                "zero_init": zero_init}
               for m, accurate in ((0, False), (-1, False), (1, False), (3, False),
                                   (2, True))
               for courant in (0.37, 0.2777777777777778)
               if not (accurate and courant > 1.0 / (abs(m) + 0.5))
               for z in (True, False) for sub_step in SUB_STEPS
               for zero_init in (False, True)]
    rows = _curl_rows(cp, None, steps, None, "engine", probe=probe,
                      configs=configs)
    assert_leg_reaches_signed_zero("engine", rows)
    record = {"rows": rows, **_summarize_rows(rows), "steps_per_row": steps}
    refused = [row for row in rows if row["verdict"] in ("NOT-COVERED",
                                                         "NO-LAUNCH",
                                                         "SHORT-LAUNCH")]
    record["refused_rows"] = len(refused)
    record["refusal_reasons"] = [row["verdict"] for row in refused][:8]
    results["engine"] = record
    save(results, out_path)
    log(f"[SUMMARY] engine {record['identical_rows']}/{record['total_rows']} "
        f"IDENTICAL, {record['refused_rows']} rows refused or unlaunched")
    return record


# --- mutation compilation, renaming and distinctness -----------------------

_TEMPORARY: List[str] = []


def compile_mutated_kernel(source: str, entry: str):
    """Compile one mutated kernel from a REAL FILE (Triton reads source through
    ``inspect``, so an ``exec``'d body raises at first launch).

    The constexpr codes and the two certified multiply helpers are re-declared in
    the header because a ``@triton.jit`` body may not read a plain module global.
    """
    header = (
        "import triton\n"
        "import triton.language as tl\n"
        "from meep_gpu.triton_kernels.complex_fields import (\n"
        "    _mul_coefficient_left, _mul_field_left)\n"
        "from meep_gpu.triton_kernels.cylindrical_complex import (\n"
        "    _mul_general_coefficient_left)\n"
        "PERIODIC = tl.constexpr(0)\n"
        "METALLIC = tl.constexpr(1)\n"
        "M_ZERO = tl.constexpr(0)\n"
        "M_ONE = tl.constexpr(1)\n"
        "M_MANY = tl.constexpr(2)\n\n"
        "@triton.jit\n")
    handle = tempfile.NamedTemporaryFile("w", suffix="_mutated_cylcomplex.py",
                                         delete=False, encoding="utf-8")
    handle.write(header + source)
    handle.close()
    _TEMPORARY.append(handle.name)
    spec = importlib.util.spec_from_file_location(
        "triton_mutated_cylcomplex_" + str(len(_TEMPORARY)), handle.name)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return getattr(module, entry)


def _renamed(source: str, suffix: str) -> Tuple[str, str]:
    name = f"cyl_complex_pml_curl_step__{suffix}"
    return (source.replace("def cyl_complex_pml_curl_step(", f"def {name}(", 1),
            name)


def _ptx_texts(kernel) -> List[str]:
    """Every compiled specialization's PTX, or an empty list when unavailable.

    Defensive on purpose: the shape of ``JITFunction.cache`` is a Triton
    internal, and a gate that CRASHES on a missing attribute reports nothing at
    all. An empty list is recorded as ``ptx_available: false`` and the
    stale-binary check falls back to the cache key, which is what the certified
    BFAST leg uses.
    """
    texts: List[str] = []
    cache = getattr(kernel, "cache", None)
    if isinstance(cache, dict):
        for per_device in cache.values():
            values = getattr(per_device, "values", None)
            for compiled in (values() if callable(values) else []):
                asm = getattr(compiled, "asm", None)
                if isinstance(asm, dict) and isinstance(asm.get("ptx"), str):
                    texts.append(asm["ptx"])
    return texts


def mutant_distinctness(mutant_kernel) -> Dict[str, Any]:
    """PLATFORM FACT (c): the cache can serve a STALE binary to a renamed mutant.

    Two independent checks, both recorded: the JIT cache keys must differ, and —
    when PTX is readable — the mutant's PTX must equal NO shipped specialization's.
    A mutant that compiled to the shipped bytes tested nothing, and would report a
    NEEDLE-MISSED that is really a harness failure.
    """
    from meep_gpu.triton_kernels import cylindrical_complex as cyl  # noqa: PLC0415

    shipped = cyl.cyl_complex_pml_curl_step
    shipped_ptx = _ptx_texts(shipped)
    mutant_ptx = _ptx_texts(mutant_kernel)
    shipped_key = str(getattr(shipped, "cache_key", "unavailable"))
    mutant_key = str(getattr(mutant_kernel, "cache_key", "unavailable"))
    overlap = [text for text in mutant_ptx if text in shipped_ptx]
    return {"entry_point": getattr(getattr(mutant_kernel, "fn", None),
                                   "__name__", "unknown"),
            "shipped_cache_key": shipped_key[:16],
            "mutant_cache_key": mutant_key[:16],
            "cache_keys_differ": shipped_key != mutant_key,
            "ptx_available": bool(mutant_ptx) and bool(shipped_ptx),
            "shipped_specializations": len(shipped_ptx),
            "mutant_specializations": len(mutant_ptx),
            "ptx_differs_from_every_shipped": not overlap}


def classify_mutation(predicted: Any, moved: Dict[str, int],
                      launches: int, distinctness: Dict[str, Any],
                      hits: int) -> str:
    """The mutation classifier. PURE, so the laptop tests drive every branch.

    Order matters: a needle that never matched, never launched or compiled to the
    shipped bytes has to be reported as a HARNESS failure, not as a measurement.
    """
    if hits == 0:
        return "DISARMED"
    if launches == 0:
        return "NO-LAUNCH"
    if not distinctness.get("cache_keys_differ", False):
        return "STALE-BINARY"
    if distinctness.get("ptx_available") and not distinctness.get(
            "ptx_differs_from_every_shipped"):
        return "STALE-BINARY"
    wants_null = isinstance(predicted, tuple) and predicted[0] == "null"
    zero_only = isinstance(predicted, tuple) and predicted[0] == "caught_zero_init_only"
    if wants_null:
        return "NULL-AS-PREDICTED" if not any(moved.values()) else "UNEXPECTEDLY-CAUGHT"
    if zero_only:
        return ("CAUGHT-ZERO-INIT-ONLY"
                if moved.get("zero_init", 0) > 0 and moved.get("random", 0) == 0
                else "PREDICTION-BROKEN")
    return "CAUGHT" if any(moved.values()) else "NEEDLE-MISSED"


MUTATION_FAILURE_STATUSES = ("DISARMED", "NO-LAUNCH", "STALE-BINARY",
                             "NEEDLE-MISSED", "UNEXPECTEDLY-CAUGHT",
                             "PREDICTION-BROKEN")


def _per_seeding_vacuity(report: Dict[str, Any]
                         ) -> Tuple[Dict[str, int], List[str]]:
    """Words each SEEDING moved across a whole battery, and the vacuous ones.

    PURE, so a laptop test drives it. The obligation is the one the two
    reference-side batteries already carry and the two DEVICE batteries did not:
    a seeding that catches nothing across the whole battery cannot fail for any
    reason connected to the arithmetic, whatever its individual rows say — it is
    the check that caught 256 zero-init constitutive rows reporting a pass on an
    untouched state. Rows whose prediction is a NULL are excluded from the total
    they cannot contribute to; a battery of nothing but predicted nulls has no
    floor to stand on and is reported vacuous on both seedings, which is correct.
    """
    per_seeding = {"random": 0, "zero_init": 0}
    for row in report.get("rows", ()):
        if row.get("status") == "NULL-AS-PREDICTED":
            continue
        moved = row.get("differing_words", {})
        for seeding in per_seeding:
            per_seeding[seeding] += int(moved.get(seeding, 0))
    report["per_seeding_total_words"] = per_seeding
    vacuous = [seeding for seeding, total in per_seeding.items() if total == 0]
    report["vacuous_seedings"] = vacuous
    return per_seeding, vacuous


def run_mutations(out_path: Optional[str], results: Dict[str, Any],
                  expansion: int, steps: int) -> Dict[str, Any]:
    """The KERNEL-SOURCE battery: armed, launch-counted, distinctness-checked."""
    kernel_source = shipped_kernel_source()
    report: Dict[str, Any] = {"rows": []}
    for name, transform, predicted in MUTATIONS:
        mutated, hits = transform(kernel_source)
        if hits == 0 or mutated == kernel_source:
            row = {"name": name, "status": "DISARMED", "hits": hits}
            report["rows"].append(row)
            log(f"[mut] {name:38s} DISARMED")
            results["mutations"] = report
            save(results, out_path)
            continue
        renamed, entry = _renamed(mutated, name)
        counter = shared.CountingKernel(compile_mutated_kernel(renamed, entry))
        moved = {"random": 0, "zero_init": 0}
        for config in MUTATION_CONFIGS:
            row = one_curl_case(cp, SYNTHETIC_SHAPE, config["m"],
                                config["accurate"], config["courant"],
                                config["z_metallic"], config["sub_step"],
                                config["zero_init"], None, steps, expansion,
                                kernel=counter, route="synthetic")
            found = row["first_divergence"]
            key = "zero_init" if config["zero_init"] else "random"
            moved[key] += 0 if found is None else found["differing_words"]
        distinctness = mutant_distinctness(counter.kernel)
        status = classify_mutation(predicted, moved, counter.launches,
                                   distinctness, hits)
        report["rows"].append(
            {"name": name, "status": status, "hits": hits,
             "launches": counter.launches, "differing_words": moved,
             "distinctness": distinctness,
             "predicted": predicted if isinstance(predicted, str) else list(predicted)})
        log(f"[mut] {name:38s} {status:22s} launches={counter.launches} {moved}")
        results["mutations"] = report
        save(results, out_path)

    failures = [row["name"] for row in report["rows"]
                if row["status"] in MUTATION_FAILURE_STATUSES]
    per_seeding, vacuous = _per_seeding_vacuity(report)
    report["failed_legs"] = failures
    report["verdict"] = ("all kernel needles live" if not failures and not vacuous
                         else "FAILED")
    results["mutations"] = report
    save(results, out_path)
    log(f"[SUMMARY] kernel mutations: {report['verdict']} "
        f"per_seeding={per_seeding}"
        + (f" vacuous={vacuous}" if vacuous else "")
        + (f" -> {failures}" if failures else ""))
    return report


def run_host_mutations(out_path: Optional[str], results: Dict[str, Any],
                       expansion: int, steps: int) -> Dict[str, Any]:
    """The HOST battery: the coefficient row, the constexprs, the sub-lattice.

    A source regex cannot reach any of these, and without them the whole
    host-side construction — the part the kernel simply trusts — is ungated.
    """
    report: Dict[str, Any] = {"rows": []}
    for name, description, predicted, apply in HOST_MUTATIONS:
        moved = {"random": 0, "zero_init": 0}
        launches = 0
        for config in MUTATION_CONFIGS:
            row = one_curl_case(cp, SYNTHETIC_SHAPE, config["m"],
                                config["accurate"], config["courant"],
                                config["z_metallic"], config["sub_step"],
                                config["zero_init"], None, steps, expansion,
                                route="synthetic", host_mutation=apply)
            found = row["first_divergence"]
            key = "zero_init" if config["zero_init"] else "random"
            moved[key] += 0 if found is None else found["differing_words"]
            launches += row["launches"]
        status = classify_mutation(predicted, moved, launches,
                                   {"cache_keys_differ": True}, 1)
        report["rows"].append({"name": name, "status": status,
                               "description": description,
                               "launches": launches, "differing_words": moved})
        log(f"[host] {name:38s} {status:22s} launches={launches} {moved}")
        results["host_mutations"] = report
        save(results, out_path)
    failures = [row["name"] for row in report["rows"]
                if row["status"] in MUTATION_FAILURE_STATUSES]
    per_seeding, vacuous = _per_seeding_vacuity(report)
    report["failed_legs"] = failures
    report["verdict"] = ("all host needles live" if not failures and not vacuous
                         else "FAILED")
    results["host_mutations"] = report
    save(results, out_path)
    log(f"[SUMMARY] host mutations: {report['verdict']} per_seeding={per_seeding}"
        + (f" vacuous={vacuous}" if vacuous else "")
        + (f" -> {failures}" if failures else ""))
    return report


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save(results: Dict[str, Any], out_path: Optional[str]) -> None:
    """Atomic rewrite after every case: an interrupted run keeps everything up to
    the failure."""
    if not out_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    temporary = out_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
        _stamp_provenance(results)  # bytes THIS process imported; see gate_provenance
        json.dump(results, handle, indent=1, default=str)
    os.replace(temporary, out_path)


def environment() -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                      time.gmtime()),
    }
    for name in ("cupy", "triton"):
        try:
            record[name] = __import__(name).__version__
        except Exception as exc:  # noqa: BLE001 - absence is the laptop answer
            record[name] = f"absent ({type(exc).__name__})"
    return record


LAPTOP_LEGS: Tuple[str, ...] = ("reference", "constitutive", "constitutive_mutations",
                                "reference_mutations", "axis_identity",
                                "stripped_control", "needle_arming")
DEVICE_LEGS: Tuple[str, ...] = ("synthetic", "guard", "multi_step", "engine",
                                "mutations", "host_mutations")


def _self_check_failures(results: Dict[str, Any]) -> List[str]:
    """Everything the laptop half can fail on, as reasons."""
    failures: List[str] = []
    reference = results.get("reference", {})
    if "total_rows" in reference and \
            reference["identical_rows"] != reference["total_rows"]:
        failures.append(f"reference transcription: {reference['identical_rows']}"
                        f"/{reference['total_rows']} identical")
    constitutive = results.get("constitutive_identity", {})
    if constitutive and constitutive["identical_rows"] != constitutive["total_rows"]:
        failures.append(f"constitutive identity: {constitutive['identical_rows']}"
                        f"/{constitutive['total_rows']} identical")
    for key in ("reference_mutations", "constitutive_mutations", "needle_arming"):
        record = results.get(key, {})
        if record and record.get("verdict", "").startswith("FAILED"):
            failures.append(f"{key}: {record.get('failed_legs')} "
                            f"vacuous={record.get('vacuous_seedings')}")
        if record and record.get("verdict", "").startswith("VOID"):
            failures.append(f"{key}: {record['verdict']}")
    axis = results.get("axis_identity", {})
    if axis and axis["differing_words"] != 0:
        failures.append(f"r-axis identity: {axis['differing_words']} differing "
                        f"words — BCX = METALLIC is NOT licensed")
    stripped = results.get("stripped_control", {})
    if stripped and stripped["differing_words"] == 0:
        failures.append("stripped complex control moved nothing")
    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=None, help="artifact path (JSON)")
    parser.add_argument("--steps", type=int, default=DEVICE_STEPS,
                        help="sub-steps per row; the artifact STATES this budget")
    parser.add_argument("--self-check", action="store_true",
                        help="laptop legs only: reference, constitutive identity, "
                             "both mutation batteries, the two identity controls "
                             "and the needle-arming check. No device.")
    parser.add_argument("--legs", default=None,
                        help="comma-separated leg names; default is every laptop "
                             f"leg plus, when a device is present, {DEVICE_LEGS}")
    parser.add_argument("--probe-artifact", default=None,
                        help="the measured complex-expansion probe (device legs)")
    args = parser.parse_args(argv)

    requested = (tuple(name.strip() for name in args.legs.split(",") if name.strip())
                 if args.legs else
                 LAPTOP_LEGS if args.self_check else LAPTOP_LEGS + DEVICE_LEGS)

    results: Dict[str, Any] = {}
    failures: List[str] = []
    skipped: Dict[str, str] = {}
    ran: List[str] = []

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
        os.makedirs(out_dir, exist_ok=True)
        try:
            shared.write_provenance(out_dir)
        except Exception as exc:  # noqa: BLE001 - provenance is recorded, not fatal
            log(f"[provenance] not written: {exc!r}")

    laptop_legs = [leg for leg in LAPTOP_LEGS if leg in requested]
    if laptop_legs:
        results = run_self_check(args.out, args.steps, laptop_legs)
        ran.extend(laptop_legs)
        failures.extend(_self_check_failures(results))
    results.setdefault("environment", environment())
    results.setdefault("step_budgets", {
        "reference_sub_steps_per_row": args.steps,
        "constitutive_sub_steps_per_row": CONSTITUTIVE_STEPS,
        "device_sub_steps_per_row": args.steps,
        "multi_step_budget": MULTI_STEP_BUDGET,
        "note": "bit-identity may be claimed for exactly these budgets and no "
                "further; the device budgets mean nothing until a device leg runs"})

    available, why = device_status()
    device_legs = [leg for leg in DEVICE_LEGS if leg in requested]
    record: Optional[Dict[str, Any]] = None
    expansion: Optional[int] = None
    if device_legs:
        # The ship subnormal policy, installed BEFORE any CuPy compile. A clean
        # no-op without CuPy; on a device host every licensed byte below this
        # line is an IEEE-keep byte or a refusal.
        shared.install_ftz_strip()
        if not available:
            for leg in device_legs:
                skipped[leg] = (f"{why} — this leg runs on the measurement "
                                f"machine; NOTHING was measured here")
                log(f"[{leg}] SKIPPED cleanly: {skipped[leg]}")
            device_legs = []
        else:
            record, reasons = probe_record(args.probe_artifact)
            if record is None:
                for leg in device_legs:
                    skipped[leg] = "; ".join(reasons)
                    log(f"[{leg}] REFUSED: {skipped[leg]}")
                failures.extend(reasons)
                device_legs = []
            else:
                from meep_gpu.triton_kernels import complex_fields as cx  # noqa: PLC0415

                expansion = cx.expansion_from_probe(record)

    for leg in device_legs:
        if leg == "synthetic":
            summary = run_synthetic(args.out, results, expansion, args.steps)
        elif leg == "guard":
            summary = run_guard(args.out, results, expansion, args.steps)
        elif leg == "multi_step":
            summary = run_multi_step(args.out, results, expansion)
        elif leg == "engine":
            summary = run_engine(args.out, results, record, args.steps)
        elif leg == "mutations":
            summary = run_mutations(args.out, results, expansion, args.steps)
        elif leg == "host_mutations":
            summary = run_host_mutations(args.out, results, expansion, args.steps)
        else:
            failures.append(f"unknown leg {leg!r}")
            continue
        ran.append(leg)
        if leg == "synthetic":
            # The predicted null's own evidence, re-measured here. A nonzero
            # count means imr_planewise_zero_cross_terms is NO LONGER a
            # predicted null on this platform and the needle must be re-armed
            # as a catch; recording it beside a green battery would be the
            # artifact-overclaim inversion.
            device_census = summary.get("minuend_census_device", {})
            if device_census.get("negative_zero_minuend_words"):
                failures.append(
                    f"minuend census on device: "
                    f"{device_census['negative_zero_minuend_words']} negative-zero "
                    f"minuend words of {device_census.get('words_scanned')} "
                    f"scanned — imr_planewise_zero_cross_terms is no longer a "
                    f"PREDICTED null and must be re-armed as a catch")
        if leg == "guard":
            continue                       # DATA, not a pass/fail (platform fact d)
        if "verdict" in summary:
            if summary["verdict"].startswith("FAILED"):
                failures.append(f"{leg}: {summary.get('failed_legs')}")
        elif summary["identical_rows"] != summary["total_rows"]:
            failures.append(f"{leg}: {summary['identical_rows']}"
                            f"/{summary['total_rows']} IDENTICAL "
                            f"({summary['by_verdict']})")

    device_ran = [leg for leg in ran if leg in DEVICE_LEGS]
    if device_ran and cp is not None:
        # The policy the artifact certifies under, with FINAL strip counters.
        # Bytes of unconfirmed policy certify nothing even if every leg passed.
        failures.extend("subnormal policy: " + reason
                        for reason in shared.ftz_strip_license_reasons())
    results["summary"] = {
        # "NOTHING RAN" is its own status. A gate that skipped every leg has not
        # passed anything, and an artifact saying "passed" beside an empty leg
        # list is exactly the kind of record a reader takes for a result.
        "status": ("NOTHING RAN" if not ran else
                   "passed" if not failures else "FAILED"),
        "failures": failures,
        "legs_ran": ran,
        "legs_skipped": skipped,
        "device_legs_ran": device_ran,
        "byte_identity_claim": (
            "NONE: no device leg ran, so nothing is claimed about the compiled "
            "kernel's bytes" if not device_ran else
            f"claimed for the legs {device_ran} at the budgets this artifact "
            f"records, and no further"),
        "subnormal_policy": shared.policy_stamp("cupy" if cp is not None else "numpy"),
    }
    save(results, args.out)
    log(f"[done] status={results['summary']['status']} legs={ran} "
        f"skipped={sorted(skipped)} failures={failures}")
    if not ran:
        log("NOTHING RAN. Pass --self-check for the laptop legs, or run on a "
            "host with cupy + triton and a measured --probe-artifact.")
        return 2
    if not device_ran:
        log("NO BYTE-IDENTITY CLAIM: the device legs did not run here.")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
