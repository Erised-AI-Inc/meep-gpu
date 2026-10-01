"""The INSTANTANEOUS chi2/chi3 constitutive family, in raw CUDA.

THE REFUSAL THIS FILE ANSWERS, quoted from the census
(``parity/meep_gpu/results/cuda_predicate_coverage_2026-08-20_all_families``)::

    instantaneous chi2/chi3: a Pade factor replaces the constitutive product

``coverage.covers_real_pml_constitutive`` states it at coverage.py:1056-1057 and
fires it on BOTH sides, ``update_H`` and ``update_E``, for the two corpus rows
that carry a nonlinearity -- ``3rd-harm-1d.py`` and
``Test3rdHarm1d.test_3rd_harm_1d``. Four slots.

THE REFUSAL IS TWO CLAIMS, AND THEY ARE NOT THE SAME CLAIM. Read off
``stepping.py``:

* **On update_E it is TRUE.** ``stepping.update_E`` branches at :969-971 into
  ``_nonlinear_constitutive`` (:1051), and what that returns is
  ``(gs*us) * calc_nonlinear_u(...)`` (:1078-1080) -- the Pade factor REPLACES the
  product rather than adding to it, and it carries a four-point transverse
  average of the OTHER two D volumes, so the sub-step stops being element-wise.
  That is a different kernel, and this file is it:
  :data:`_update_E_pml_real_nonlinear_kernel_code`.

* **On update_H it is FALSE, and inherited.** ``stepping.update_H``
  (stepping.py:907-923) is six lines: ``_pml_is_active``, a storage requirement,
  and a loop calling ``_apply_constitutive_pml(H, B, kps, kms, f_w_H)``. It does
  not read ``_chi2_components``, ``_chi3_components``, ``has_nonlinearity``,
  ``is_nonlinear``, ``chi2_for``, ``chi3_for``, ``_nonlinear_constitutive``,
  ``calc_nonlinear_u`` or ``_nonlinear_displacement``; none of those names occurs
  between :907 and :923. The module header says so in its own words
  (stepping.py:26-28): "An instantaneous nonlinearity ... lives entirely inside
  ``update_E``". So the H side needs NO NEW KERNEL -- the certified
  ``update_H_pml_real`` is already the whole sub-step -- and what it needs
  is a PREDICATE that says so.

THAT ASYMMETRY IS WHY THIS FILE HAS TWO ARMS AND ONE KERNEL:

* **Arm E (KERNEL).** ``update_E_pml_real_nonlinear``, transcribed below.
* **Arm H (NULL WIDENING).** No kernel. :func:`covers_real_pml_nonlinear_constitutive`
  with ``side="H"`` admits the SHIPPED ``constitutive_kernels.update_H_pml_real``
  on a grid carrying a nonlinearity. An argument from absence ("nothing in
  update_H mentions chi, so it must be fine") is a HYPOTHESIS; the gate
  ``parity/meep_gpu/gate_cuda_nonlinear.py`` measures it, with the refusal's own
  premise armed as a defect (``pade_scale_the_H_source``, which makes the H
  kernel nonlinear and must be CAUGHT) so that the null is a measurement and not
  a blind spot.

DISJOINTNESS IS BY CONSTRUCTION, NOT BY ARRANGEMENT. Every other constitutive
predicate in this package refuses a nonlinearity by name; this one REQUIRES one.
``covers_real_pml_constitutive`` (coverage.py:1056), ``covers_real_pml_offdiag_constitutive``
(:1226), the ADE arm (:1590) and the no-PML null (:1916) all fire on
``_chi2_components or _chi3_components``; :func:`covers_real_pml_nonlinear_constitutive`
refuses when NEITHER is present. The census's disjointness check therefore cannot
report an overlap for this family unless one of those clauses is deleted, and
``test_nonlinear_constitutive.py`` pins the complementarity directly.

=============================================================================
WHAT IS TRANSCRIBED, AND FROM WHERE
=============================================================================

``stepping.update_E`` (stepping.py:926-993) with ``nonlinear`` true, no
polarization, no off-diagonal row, real float32 storage, an active PML::

    volumes = fields.displacement_minus_polarization_volumes()   # fields.py:1107
                                                                 # aliases D with no poles
    gs   = volumes[c]                                            # stepping.py:1092
    us   = fields.inverse_epsilon_for(c)                         # :1093
    g1s  = (g1 + shift_down(g1, a1)) + shift_up(same, a_own)     # :1155-1163
    g2s  = likewise for the second partner
    dsqr = gs*gs + 0.0625*(g1s*g1s + g2s*g2s)                    # :1117-1118
    c2   = (gs   * chi2) * (us*us)                               # :1025
    c3   = (dsqr * chi3) * ((us*us)*us)                          # :1026
    u    = ((1 + c2) + 2*c3) / ((1 + 2*c2) + 3*c3)               # :1027
    src  = (gs * us) * u                                         # :1078-1080
    prev = f_w_c ; f_w_c = src                                   # :2084-2087
    E_c  = (E_c + kps_a_h*src) - kms_a_h*prev                    # :2086-2087, half-integer

A component with no chi installed in a partly nonlinear run takes MEEP's
``else if (u)`` branch (stepping.py:1100-1102) and is the certified plain body,
``src = gs * us``. That is a per-component decision the array path makes with a
Python ``if``; this kernel makes it with a runtime flag, so the NL=0 arm is the
certified expression tree character for character.

=============================================================================
THE SIX THINGS THAT DECIDE BIT-IDENTITY HERE
=============================================================================

The first three are the certified constitutive pair's notes and are unchanged
(``constitutive_kernels.py``): two separate left-to-right accumulations, the
coefficient index is the component's OWN axis, and ``prev`` is read before ``fw``
is written. This family adds three of its own.

4. **THE FOUR-CORNER ASSOCIATION IS THE SHIFTED-PAIR ONE, NOT MEEP C's.**
   ``stepping._nonlinear_transverse_sums`` (:1155-1163) forms
   ``pair = g + shift_down(g, a1)`` and then ``pair + shift_up(pair, a_own)``, so
   the bytes are ``(g[i] + g[i-s1]) + (g[i+s] + g[i+s-s1])``. MEEP C sums left to
   right, ``g[i] + g[i+s] + g[i-s1] + g[i+(s-s1)]`` (step_generic.cpp:646-648).
   The BYTE ARBITER IS ``stepping.py``, because that is what the array path runs
   and what every gate on this track compares against, so the pair association is
   what this kernel carries. The gate's ``four_point_left_to_right`` mutation is
   what makes the difference measured rather than argued.

   The far pair is the near pair moved one cell up, so loading the two far
   corners and adding them is the same bits as shifting the formed pair: a shift
   is data movement. Ghost rules compose per axis -- a periodic index wraps, a
   metallic out-of-range corner contributes ``+0.0f``, and the doubly-shifted
   corner takes BOTH axes' rules -- exactly as shifting an already-shifted array
   applies them in sequence. A METALLIC own-axis top plane therefore contributes
   ``0.0f + 0.0f`` for the WHOLE far pair, which is what
   ``_shift_up``'s ``shifted[face] = 0`` (stepping.py:1830) writes.

5. **THE TWO SHIFTS GO IN OPPOSITE DIRECTIONS.** Half a cell DOWN the partner's
   own axis, half a cell UP this component's axis (stepping.py:1160-1171 states
   the geometry: for Ez at ``(x_i, y_j, z_k+1/2)`` the partner Dx sits at
   ``(x_i+1/2, y_j, z_k)``). Taking both the same way round is a half-cell
   registration error that survives every scalar test; the gate carries
   ``dsqr_same_direction_shifts`` for exactly it.

6. **THE DIVISION IS THE ONLY QUOTIENT ON THIS TRACK, AND IT IS SPELLED
   ``__fdiv_rn``.** ``constitutive_kernels.py``'s header records the one PTX
   exception this package knows of: ptxas expands ``div.rn.f32`` into a
   Newton-Raphson sequence whose range checks carry ``.FTZ`` in SASS regardless
   of the PTX modifier, so a PTX audit does NOT fully enforce a subnormal *keep*
   policy where a division appears. That note ends "if a future constitutive
   variant ever divides ... that exception applies to it". THIS IS THAT VARIANT.

   What the exception can and cannot reach is a question about the OPERANDS, and
   it is measured rather than argued. Both quotient operands are sums anchored at
   ``1.0`` of float32-grid addends, and the driver's pole guard
   (``stepping.nonlinear_margin``, :1315-1357; ``driver._NonlinearityGuard``)
   holds ``|c2| + |c3| < 1/3``, so the numerator, the denominator and the
   quotient all sit in ``[2/3, 2]`` -- no subnormal can reach or leave the divide.
   The gate's ``subnormal_band`` value class puts subnormals in every OPERAND
   ARRAY and the record carries the measured census, so the claim "the .FTZ
   expansion is inert on this expression" is a number in the artifact and not a
   sentence here.

   ``__fdiv_rn`` is written rather than ``/`` for the same reason the sibling
   Triton kernel writes ``tl.math.div_rn``: on THAT platform the plain operator
   was measured to be ``div.full.f32`` (~2 ulp) and every nonlinear sweep case
   diverged. CUDA's default is ``-prec-div=true`` and the two
   spellings are expected to agree HERE -- which is a claim about this platform,
   so the gate probes it (``plain_division``, reported as a spelling probe with a
   reading, exactly as the contraction guard's control is reported).

=============================================================================
PLATFORM FACTS INHERITED FROM THE CERTIFIED PAIR
=============================================================================

* ``--fmad=false`` is CORRECTNESS. Every one of ``gs*gs + 0.0625*(...)``,
  ``(1 + c2) + 2*c3``, ``(1 + 2*c2) + 3*c3`` and both tail accumulations is a
  contraction candidate the array path rounds twice.
* THE DEVICE STRINGS ARE PURE ASCII. ``cupy.cuda.compiler`` writes the source
  through the interpreter's locale encoding, which is ASCII in a non-interactive
  shell on the validation host; two em-dashes in a comment killed a kernel at
  first launch on 2026-08-15. ``test_nonlinear_constitutive.py`` performs the
  encode itself rather than trusting a reading.
* SIGNED ZERO is NOT canonicalized on this path (``-x`` lowers to ``neg.f32``),
  and this kernel is written so that it never depends on the answer: the
  subtraction is spelled ``a - kms*prev``, and no addend is negated.

=============================================================================
WHAT THIS IS NOT
=============================================================================

NOTHING DISPATCHES THIS. No module in ``meep_gpu/`` imports ``cuda_kernels`` at
all (``test_package_boundary.py`` pins the absence in both directions), so a
predicate returning True here licenses a MEASUREMENT and not a production step.

THE OFF-DIAGONAL NONLINEAR CASE IS REFUSED, and by a line rather than by
omission: MEEP's most general branch scales the WHOLE row product, coupling
included, ``fw = (gs*us + OFFDIAG + OFFDIAG) * u`` (stepping.py:1095-1099,
:1075-1080). That is a fused nonlinear+offdiag kernel and it is not this one.
``coverage.covers_real_pml_offdiag_constitutive`` refuses the nonlinearity from
its side (coverage.py:1226) and this refuses the coupling from ours, so the pair
is disjoint in both directions.

COMPLEX STORAGE IS REFUSED. MEEP's ``DOCMP`` loop runs the whole nonlinear
recovery once per Cartesian part (update_eh.cpp:149), so ``u`` is built from real
parts alone and again from imaginary parts alone (stepping.py:1110-1116). That is
two Pade evaluations per component and a different kernel.

A kernel with a gate verdict moves from :data:`UNCERTIFIED_KERNELS` to
:data:`CERTIFIED_KERNELS`; a kernel in neither set fails the slice test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

try:  # The predicate, the plan and the device text must be readable with no GPU.
    import cupy as cp
except ImportError:  # pragma: no cover - the device path
    cp = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from ..fields import Fields
    from ..pml import PML

# The shared clause helpers live in the CuPy-free sibling ``coverage.py`` so this
# predicate stays evaluable and mutable on a machine with no GPU. Imported back
# here because this module is the family's public face; the by-path fallback is
# for the bit-identity probe, which loads these modules outside the package.
try:
    from . import coverage as _coverage
except ImportError:  # pragma: no cover - loaded by path, outside the package
    import importlib.util as _importlib_util
    import os as _os

    _spec = _importlib_util.spec_from_file_location(
        "cuda_kernels_coverage",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "coverage.py"))
    _coverage = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_coverage)

try:
    from . import compile_cache
except ImportError:  # pragma: no cover - loaded by path, outside the package
    import importlib.util as _importlib_util
    import os as _os

    _cache_spec = _importlib_util.spec_from_file_location(
        "cuda_kernels_compile_cache",
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                      "compile_cache.py"))
    compile_cache = _importlib_util.module_from_spec(_cache_spec)
    _cache_spec.loader.exec_module(compile_cache)


#: Byte-identical to the array path with a gate verdict behind it. The evidence
#: is :data:`NONLINEAR_CONSTITUTIVE_CERTIFICATION`; an entry here without a record
#: entry is the failure ``test_nonlinear_constitutive.py`` exists to catch.
CERTIFIED_KERNELS: Tuple[str, ...] = ("update_E_pml_real_nonlinear",)

#: Shipped but not gated. EMPTY, and the partition test requires every shipped
#: kernel to be in exactly one of the two sets -- so a kernel added to this file
#: without a gate verdict fails there rather than shipping unmeasured. Spelled as
#: a dict WITHOUT a type annotation for the reason ``constitutive_kernels.py``
#: records: the partition test reads this name off the syntax tree without
#: importing the module, and an annotated assignment is an ``ast.AnnAssign``,
#: which that reader does not match.
UNCERTIFIED_KERNELS = {}

#: THE GATE VERDICT, and what it does and does not license.
#:
#: Cut 2026-08-20 on the GPU host, ONE verified-empty RTX A6000 (index 5), under BOTH
#: float32 subnormal policies, with the two policies proven to have compiled
#: DISTINCT binaries: the run's NVRTC observer recorded 55 compiles and 49 distinct
#: binaries per policy, with ``all_ftz_true_reached_nvrtc`` False under ``keep``
#: (the strip removed ``-ftz=true`` from every call) and True under ``flush``.
#:
#: ARM E is the kernel; ARM H is the certified ``update_H_pml_real``, run
#: unchanged on grids carrying a live chi -- its byte count is included below and
#: its refusal premise was armed (``pade_scale_the_H_source``, CAUGHT 6/6).
NONLINEAR_CONSTITUTIVE_CERTIFICATION: Dict[str, Any] = {
    "gate": "parity/meep_gpu/gate_cuda_nonlinear.py",
    #: The gate was run by a single-host shell driver that is not part of this
    #: repository; parity/meep_gpu/README.md gives the recipe it followed (one
    #: GPU, a separate CuPy cache per subnormal policy).
    "runner": "a single-host shell driver (not published)",
    "artifacts": "parity/meep_gpu/results/cuda_nonlinear_2026-08-21_rename",
    "recorded_utc": "2026-08-21T23:09:19Z",
    "host": "the GPU host",
    "device": "NVIDIA RTX A6000 (compute capability 8.6, CuPy 13.5.1, NVRTC 11.6)",
    #: The 2026-08-20 run named GPU index 5. The 2026-08-21 re-cut's payload stamps
    #: environment.device_index 0, which is the index INSIDE the run's
    #: CUDA_VISIBLE_DEVICES pin and not the host's numbering, so the host index is
    #: not recoverable from the artifact and is not claimed.
    "gpu_index": None,
    "policies_released": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    "scored_cases": {"keep": 112, "flush": 108},
    "electric_cases": {"keep": 56, "flush": 54},
    "magnetic_cases": {"keep": 56, "flush": 54},
    "single_launch_bit_identical": {"keep": 112, "flush": 108},
    "multi_step_bit_identical": {"keep": 112, "flush": 108},
    "multi_step_launches": 60,
    "words_compared": {"keep": 1113600, "flush": 1067520},
    "differing_words": {"keep": 0, "flush": 0},
    "mutations_caught": 24,
    "mutations_null_confirmed": 5,
    "mutation_legs": 183,
    "max_pade_departure": {"keep": 0.1357, "flush": 0.1357},
    "max_expansion": {"keep": 0.2264, "flush": 0.2264},
    "contraction_guard_control": (
        "55 of 56 unguarded legs DIVERGED at courant 0.35 under keep (52 of 54 "
        "under flush), so --fmad=false is CORRECTNESS on this sub-step and not "
        "tuning -- measured, not asserted"),
    "division_spelling": (
        "NULL CONFIRMED on both policies: rewriting __fdiv_rn(num, den) to "
        "num / den produced identical bytes on every leg, so CUDA's default "
        "-prec-div=true is IEEE round-to-nearest division on this platform. The "
        "sibling Triton track measured the OPPOSITE for its plain operator "
        "(div.full.f32, ~2 ulp), which is why the spelling is explicit here"),
    "subnormal_reach_into_the_divide": (
        "MEASURED INERT under keep: 65760 subnormal words in c2/c3 across 14 "
        "cases, and ZERO subnormal words in num, den or the quotient. ptxas's "
        "FTZ-carrying div.rn range checks (constitutive_kernels.py's header) have "
        "nothing in this expression to reach, because both quotient operands are "
        "sums anchored at 1.0"),
    "flush_annihilates_a_subnormal_chi_volume": (
        "MEASURED: under meep_x86_flush a float32 chi VOLUME drawn in the "
        "subnormal band rounds to zero on the host, so Fields.set_nonlinear_volumes "
        "sees a trivial pair and drops it (fields.py:879-880) and the run is "
        "LINEAR. Those 4 cases are SKIPPED under flush and this predicate refuses "
        "them by name -- which is why the flush leg scores 108 and not 112"),
    "slot_delta": (
        "+4 on the 759-slot census: 557/759 -> 561/759, recomputed by "
        "parity/meep_gpu/results/cuda_nonlinear_slot_recount_2026-08-20/"
        "recount_cuda_nonlinear_slots.py, disjointness clean. ROWS COVERED AT "
        "EVERY SUB-STEP is unchanged at 110: the CURL's own chi2/chi3 clause "
        "still refuses step_B and step_D on both rows, and that is a different "
        "family's refusal"),
    # WHAT THE VERDICT IS PINNED TO, and it is deliberately NARROWER than the file
    # hash. ``certified_module_sha256`` stops matching the moment anyone fixes a
    # typo in a docstring, which makes it useless as a pin on the thing that
    # actually decides the verdict. What NVRTC compiles is the kernel source
    # STRING, and that is what is pinned; the module hash is carried beside it so
    # a post-gate edit has to be DECLARED rather than inherited silently.
    "device_source_sha256": {
        "_NONLINEAR_PRELUDE":
            "408fa79fc5ef45bceeb7a2736a775c392b8639f4aa251c40427031a165977ddd",
        "update_E_pml_real_nonlinear":
            "a729688c2dbd260ca6baa7fac536a50164943164eeb2810307dd0e759223931b",
    },
    "certified_module_sha256":
        "c54175a7f24c8d24427b7d844303820a6517a6c99889591c899b43ce24de570d",
    "post_certification_edits": (
        {"what": ("nonlinear_constitutive.py gained CERTIFIED_KERNELS, "
                  "NONLINEAR_CONSTITUTIVE_CERTIFICATION and this list; no "
                  "device string moved"),
         "touches_device_code": False,
         "why_the_verdict_survives": (
             "MEASURED, not asserted: the sha256 of _NONLINEAR_PRELUDE and of "
             "_update_E_pml_real_nonlinear_kernel_code are byte-identical "
             "between the staged tree the gate compiled on the GPU host and the file "
             "that ships here (408fa79f... and 8cde0fe2...), and only the module "
             "hash moved. test_the_gate_bound_the_device_bytes_that_ship_today "
             "re-performs that comparison on every run"),
         "revision_sha256": None},
        {"what": ("2026-08-22: RE-CUT onto cuda_nonlinear_2026-08-21_rename. The "
                  "CUDA kernel rename (fused_update_E_pml_real_nonlinear -> "
                  "update_E_pml_real_nonlinear) moved the certified device string, "
                  "so the whole record was re-cut against a run that compiled the "
                  "shipped bytes -- both policy legs released and both imported "
                  "nonlinear_constitutive.py at the digest that ships"),
         "touches_device_code": True,
         "why_the_verdict_survives": (
             "It does not survive -- it was RE-EARNED. Every scored number came "
             "back unchanged (112/108 cases, 56/54 electric and magnetic, "
             "1113600/1067520 words, 0 differing, 24 caught, 5 null-confirmed, "
             "183 legs, 60 launches). TWO MEASUREMENTS MOVED and they are recorded "
             "above as measured rather than carried over: max_pade_departure and "
             "max_expansion, which under the 2026-08-20 run differed between the "
             "policies (0.1349/0.1365 and 0.2547/0.2166) and now agree across them "
             "(0.1357 and 0.2264). The cause is in the gate, not the kernel: it "
             "draws its fixtures from np.random.default_rng(SEED + sha256(label)) "
             "as of gate_cuda_nonlinear.py:1122, where it previously salted the "
             "draw with hash() of a tuple containing strings -- a value "
             "PYTHONHASHSEED randomises per process, so the two policy legs "
             "(separate processes by construction) had been drawing DIFFERENT case "
             "sets. Identical maxima across the policies is that reseed working"),
         "revision_sha256": None},
    ),
    "dependency_drift_during_the_run": (
        "coverage.py was edited by a concurrent session BETWEEN the sync and the "
        "end of the run (f005a08d... on the staged tree, 892fea8f... here), so "
        "the shared clause helpers this predicate composes from are worth "
        "checking rather than assuming. MEASURED: all sixteen symbols this module "
        "reads -- PERIODIC, MIRROR, METALLIC, CYL_AXIS, BC_CODES, "
        "CONSTITUTIVE_SIDES, _backend, _grid_facts, _stored_past_owned_from, "
        "_boundary_kinds_from, _array_problem, _coefficient_vector_problem, "
        "real_pml_boundary_kinds, constitutive_sub_lattice, "
        "covers_real_pml_constitutive and covers_real_pml_offdiag_constitutive -- "
        "are BYTE-IDENTICAL between the two versions; the 60 changed lines are "
        "record constants and prose belonging to the off-diagonal and cylindrical "
        "families. stepping.py, the ORACLE, is identical (10f79322...). The slot "
        "recount and the merge-bar suite both ran against the SHIPPING coverage.py "
        "and agree with the device leg on the predicate's verdicts"),
    "limits": (
        "NOTHING DISPATCHES THIS. No module in meep_gpu/ imports cuda_kernels at "
        "all, so the verdict is a statement about two kernels and not about any "
        "run. It says nothing about a fold, a cylindrical grid, complex storage, "
        "a Bloch phase, a registered polarization or an off-diagonal chi1inv row "
        "-- each REFUSED by name -- and nothing about throughput, which was not "
        "measured and cannot be inferred from a correctness gate."),
}

#: The three E components in ``stepping.E_CONSTITUTIVE_TERMS`` order
#: (stepping.py:228), each with its OWN axis -- the axis whose HALF-INTEGER
#: coefficient pair the tail reads (MEEP's ``dsigw``).
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

#: The transverse partners of each component, in MEEP's ``cycle_direction`` order
#: X -> Y -> Z (vec.hpp:586; stepping.py:1183-1185): own axis + 1 first, own axis
#: + 2 second. Ez therefore takes Dx then Dy. Kept literal so the transcription is
#: checkable without the reasoning, and pinned by a test.
TRANSVERSE_PARTNERS: Tuple[Tuple[int, int], ...] = ((1, 2), (2, 0), (0, 1))

#: The boundary kinds whose ghost rule this kernel carries. A fold and the
#: cylindrical axis are refused: both change the ghost the transverse sums read
#: (a parity-weighted mirror image, or the axis row), and neither is transcribed.
NONLINEAR_BOUNDARY_KINDS: Tuple[str, ...] = (_coverage.PERIODIC, _coverage.METALLIC)

#: The two sides this family answers for. ``"E"`` is served by the kernel below;
#: ``"H"`` is the NULL widening -- the arithmetic is the certified
#: ``update_H_pml_real``'s, unchanged, and what is new is only the
#: admission of a grid that carries chi.
NONLINEAR_SIDES: Tuple[str, ...] = ("H", "E")

#: Which side each arm launches. ``None`` means "no kernel of this family": the
#: caller launches ``constitutive_kernels.update_fused_pml_real('H', ...)``.
SIDE_KERNEL: Dict[str, Optional[str]] = {
    "H": None,
    "E": "update_E_pml_real_nonlinear",
}

#: The derived bound ``stepping.nonlinear_margin`` (:1315-1357) records and
#: ``driver._NonlinearityGuard`` enforces: ``1 + 2*c2 + 3*c3 >= 1 - 3*(|c2|+|c3|)``,
#: so an expansion below 1/3 places the denominator strictly above zero. NOT a
#: predicate clause -- the guard is sampled outside the step loop by the driver
#: and this kernel does not change it -- but the gate's amplitude classes are
#: held under it, and the record carries the measured value per case.
POLE_EXPANSION_BOUND = 1.0 / 3.0


# =============================================================================
# THE DEVICE CODE
# =============================================================================

# The tail is the certified constitutive pair's, character for character, so
# that the shared source mutations of ``probe_fused_kernel_bit_identity`` --
# ``regroup_constitutive``, ``drop_fw_store``, ``store_fw_before_reading_prev``,
# ``commute_constitutive_scale`` -- arm against this kernel too. A second
# spelling of the same four lines would silently disarm all four.
_NONLINEAR_PRELUDE = r'''
#define BC_PERIODIC 0
#define BC_METALLIC 1

// stepping._apply_constitutive_pml (stepping.py:2065-2096), MEEP's
// step_update_EDHB with dsigw active. Two SEPARATE accumulations, left to right;
// `prev` loaded before the store.
__device__ __forceinline__ void constitutive_apply(
    float* __restrict__ f, float* __restrict__ fw, int idx, float src,
    float kps, float kms
) {
    float prev = fw[idx];
    fw[idx] = src;
    float a = f[idx] + kps * src;
    f[idx] = a - kms * prev;
}

// stepping._nonlinear_transverse_sums (stepping.py:1155-1163), the SHIFTED-PAIR
// association:  (g[i] + g[i-s1]) + (g[i+s] + g[i+s-s1]).
// `o_c` is this cell, `o_d` the partner-axis neighbour below, `o_u` the own-axis
// neighbour above, `o_ud` the corner that took both shifts. A false validity
// flag is the zero ghost `_shift_up`/`_shift_down` write into the face; the
// index is clamped in bounds by the caller so nothing is dereferenced for it.
__device__ __forceinline__ float four_point_sum(
    const float* __restrict__ g, int o_c, int o_d, int o_u, int o_ud,
    bool v_d, bool v_u, bool v_ud
) {
    float near_pair = g[o_c] + (v_d ? g[o_d] : 0.0f);
    float far_pair = (v_u ? g[o_u] : 0.0f) + (v_ud ? g[o_ud] : 0.0f);
    return near_pair + far_pair;
}

// stepping.calc_nonlinear_u (stepping.py:996-1027), transcribed term for term:
//     c2 = di   * chi2 * (chi1inv * chi1inv)
//     c3 = dsqr * chi3 * (chi1inv * chi1inv * chi1inv)
//     u  = (1 + c2 + 2*c3) / (1 + 2*c2 + 3*c3)
// Python's left-to-right binding makes those `(gs*chi2)*(us*us)`,
// `(dsqr*chi3)*((us*us)*us)`, `((1+c2) + 2*c3) / ((1+2*c2) + 3*c3)`.
// `us_sq` and `us_cu` arrive FORMED so the grouping is decided once, at the one
// place the array path decides it.
// The division is __fdiv_rn: IEEE round-to-nearest, div.rn.f32. See note 6.
__device__ __forceinline__ float pade_u(
    float gs, float dsqr, float chi2, float chi3, float us_sq, float us_cu
) {
    float c2 = (gs * chi2) * us_sq;
    float c3 = (dsqr * chi3) * us_cu;
    float num = (1.0f + c2) + 2.0f * c3;
    float den = (1.0f + 2.0f * c2) + 3.0f * c3;
    return __fdiv_rn(num, den);
}
'''  # stepping.py live lines for the frozen device-text citation(s) in this string: 2065-2096->2112-2143, 1155-1163->1184-1192, 996-1027->1025-1056

# THE E SIDE. Half-integer coefficients (stepping.py:1015), THREE inverse-epsilon
# pointers (never fields.inv_eps, which is the Ez view, fields.py:1259-1260), and
# per-component chi2/chi3 that may each be a volume or a uniform scalar --
# ``Fields._coerce_nonlinear_value`` (fields.py:929-945) accepts both, and MEEP
# stores a zero partner for whichever of the pair the caller did not supply
# (structure.cpp:815-826), so a chi3-only medium arrives here as chi2 == 0.0.
#
# THE THREE SOURCES ARE FORMED BEFORE ANY STORE, and here that is load-bearing
# rather than incidental: the transverse average READS the other two components'
# D volumes, so a store into D would reach a later operand. It does not -- this
# sub-step reads D and inv_eps and writes E and f_w_E, four disjoint sets -- but
# the ordering is written out so a reader does not have to rediscover it.
_update_E_pml_real_nonlinear_kernel_code = _NONLINEAR_PRELUDE + r'''
extern "C" __global__ void update_E_pml_real_nonlinear(
    float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez,
    float* __restrict__ f_w_Ex, float* __restrict__ f_w_Ey,
    float* __restrict__ f_w_Ez,
    const float* __restrict__ Dx, const float* __restrict__ Dy,
    const float* __restrict__ Dz,
    const float* inv_eps_Ex, const float* inv_eps_Ey, const float* inv_eps_Ez,
    const float* chi2_Ex, const float* chi2_Ey, const float* chi2_Ez,
    const float* chi3_Ex, const float* chi3_Ey, const float* chi3_Ez,
    float s2_Ex, float s2_Ey, float s2_Ez,
    float s3_Ex, float s3_Ey, float s3_Ez,
    int nl_x, int nl_y, int nl_z,
    int q2v_x, int q2v_y, int q2v_z,
    int q3v_x, int q3v_y, int q3v_z,
    int nx, int ny, int nz,
    const float* __restrict__ kps_x, const float* __restrict__ kms_x,
    const float* __restrict__ kps_y, const float* __restrict__ kms_y,
    const float* __restrict__ kps_z, const float* __restrict__ kms_z,
    int bc_x, int bc_y, int bc_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
    int nyz = ny * nz;

    // Per-axis neighbour indices in BOTH directions, with that axis's ghost
    // rule (stepping._shift_up :1723, _shift_down :1787, plain PERIODIC and
    // METALLIC branches -- a fold and the cylindrical axis are refused by the
    // predicate). An invariant axis (n == 1, always PERIODIC) needs no case of
    // its own: the wrap returns the same cell, which is what MEEP's stride(d)=0
    // computes for a direction it does not have.
    int di = i - 1, ui = i + 1;
    int dj = j - 1, uj = j + 1;
    int dk = k - 1, uk = k + 1;
    bool dvx = true, uvx = true, dvy = true, uvy = true, dvz = true, uvz = true;
    // An invalid neighbour is CLAMPED IN BOUNDS and gated by its flag, never
    // left negative and guarded by a ternary alone: a predicated load may be
    // issued for a branch whose value is discarded, and a negative offset is an
    // unmapped address rather than a discarded value.
    if (bc_x == BC_METALLIC) {
        dvx = (di >= 0);
        uvx = (ui < nx);
        if (!dvx) { di = 0; }
        if (!uvx) { ui = 0; }
    } else {
        if (di < 0) { di = nx - 1; }
        if (ui == nx) { ui = 0; }
    }
    if (bc_y == BC_METALLIC) {
        dvy = (dj >= 0);
        uvy = (uj < ny);
        if (!dvy) { dj = 0; }
        if (!uvy) { uj = 0; }
    } else {
        if (dj < 0) { dj = ny - 1; }
        if (uj == ny) { uj = 0; }
    }
    if (bc_z == BC_METALLIC) {
        dvz = (dk >= 0);
        uvz = (uk < nz);
        if (!dvz) { dk = 0; }
        if (!uvz) { uk = 0; }
    } else {
        if (dk < 0) { dk = nz - 1; }
        if (uk == nz) { uk = 0; }
    }

    // NOT __restrict__ on the epsilon or the chi pointers, deliberately: an
    // isotropic run hands the same device pointer three times (fields.py:
    // 1321-1326) and a scalar or linear component's chi slot is bound to its own
    // D array by the plan, so restrict would be a promise the caller cannot keep.
    float us_x = inv_eps_Ex[idx];
    float us_y = inv_eps_Ey[idx];
    float us_z = inv_eps_Ez[idx];
    float gs_x = Dx[idx];
    float gs_y = Dy[idx];
    float gs_z = Dz[idx];

    // --- component 0: Ex -- own axis x; partners Dy (down y) then Dz (down z).
    float src_x = gs_x * us_x;
    if (nl_x) {
        float g1s = four_point_sum(
            Dy, idx,
            i * nyz + dj * nz + k,
            ui * nyz + j * nz + k,
            ui * nyz + dj * nz + k,
            dvy, uvx, uvx && dvy);
        float g2s = four_point_sum(
            Dz, idx,
            i * nyz + j * nz + dk,
            ui * nyz + j * nz + k,
            ui * nyz + j * nz + dk,
            dvz, uvx, uvx && dvz);
        float dsqr = gs_x * gs_x + 0.0625f * (g1s * g1s + g2s * g2s);
        float q2 = q2v_x ? chi2_Ex[idx] : s2_Ex;
        float q3 = q3v_x ? chi3_Ex[idx] : s3_Ex;
        float us_sq = us_x * us_x;
        float us_cu = us_sq * us_x;
        src_x = src_x * pade_u(gs_x, dsqr, q2, q3, us_sq, us_cu);
    }

    // --- component 1: Ey -- own axis y; partners Dz (down z) then Dx (down x).
    float src_y = gs_y * us_y;
    if (nl_y) {
        float g1s = four_point_sum(
            Dz, idx,
            i * nyz + j * nz + dk,
            i * nyz + uj * nz + k,
            i * nyz + uj * nz + dk,
            dvz, uvy, uvy && dvz);
        float g2s = four_point_sum(
            Dx, idx,
            di * nyz + j * nz + k,
            i * nyz + uj * nz + k,
            di * nyz + uj * nz + k,
            dvx, uvy, uvy && dvx);
        float dsqr = gs_y * gs_y + 0.0625f * (g1s * g1s + g2s * g2s);
        float q2 = q2v_y ? chi2_Ey[idx] : s2_Ey;
        float q3 = q3v_y ? chi3_Ey[idx] : s3_Ey;
        float us_sq = us_y * us_y;
        float us_cu = us_sq * us_y;
        src_y = src_y * pade_u(gs_y, dsqr, q2, q3, us_sq, us_cu);
    }

    // --- component 2: Ez -- own axis z; partners Dx (down x) then Dy (down y).
    float src_z = gs_z * us_z;
    if (nl_z) {
        float g1s = four_point_sum(
            Dx, idx,
            di * nyz + j * nz + k,
            i * nyz + j * nz + uk,
            di * nyz + j * nz + uk,
            dvx, uvz, uvz && dvx);
        float g2s = four_point_sum(
            Dy, idx,
            i * nyz + dj * nz + k,
            i * nyz + j * nz + uk,
            i * nyz + dj * nz + uk,
            dvy, uvz, uvz && dvy);
        float dsqr = gs_z * gs_z + 0.0625f * (g1s * g1s + g2s * g2s);
        float q2 = q2v_z ? chi2_Ez[idx] : s2_Ez;
        float q3 = q3v_z ? chi3_Ez[idx] : s3_Ez;
        float us_sq = us_z * us_z;
        float us_cu = us_sq * us_z;
        src_z = src_z * pade_u(gs_z, dsqr, q2, q3, us_sq, us_cu);
    }

    // Ex <- dsigw = x.  Ey <- dsigw = y.  Ez <- dsigw = z.
    constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);
    constitutive_apply(Ey, f_w_Ey, idx, src_y, kps_y[j], kms_y[j]);
    constitutive_apply(Ez, f_w_Ez, idx, src_z, kps_z[k], kms_z[k]);
}
'''

#: NVRTC compile options -- CORRECTNESS, not performance. Spelled here rather
#: than imported so that loading this file by path cannot pick up a different
#: tuple than the one the gate compiled; the slice test pins the two spellings
#: equal against ``constitutive_kernels._COMPILE_OPTIONS``.
_COMPILE_OPTIONS = ('--fmad=false',)

#: Lanes per block, the value both certified tracks landed on independently.
_NONLINEAR_THREADS = 256


def _clear_kernel_cache() -> int:
    """Drop every memoized kernel; returns how many entries went."""
    return compile_cache.clear_kernel_cache()


def _get_kernel(name: str):
    """Compile on first use, memoized on (name, options, policy, source).

    THE CODE MAP IS REBUILT PER CALL and that is load-bearing for the same reason
    it is in the sibling modules: the source is part of the memo key, and the gate
    mutates kernels by assigning over the module-level source string. A map
    memoized at first call would hand back the pre-mutation string forever -- a
    leg reporting a pass for a mutation it never applied.
    """
    code_map = {
        'update_E_pml_real_nonlinear': (
            _update_E_pml_real_nonlinear_kernel_code,
            'update_E_pml_real_nonlinear'),
    }
    code, func_name = code_map[name]
    key = compile_cache.kernel_cache_key(name, False, _COMPILE_OPTIONS, code)
    return compile_cache.get_or_compile(
        key, lambda: cp.RawKernel(code, func_name, options=_COMPILE_OPTIONS))


# =============================================================================
# THE PLAN
# =============================================================================

def nonlinear_bindings(fields: 'Fields') -> Dict[str, Any]:
    """Per-component chi2/chi3 operands, split into volumes and fp32 scalars.

    ``Fields._coerce_nonlinear_value`` (fields.py:929-945) stores each entry as a
    PYTHON FLOAT (uniform) or a float32 volume, and a component absent from
    ``_chi2_components`` is linear -- MEEP's ``else if (u)`` branch
    (stepping.py:1100-1102). This resolves the three-way choice ONCE, on the
    host, and the kernel takes it as flags.

    A SCALAR IS ROUNDED ON THE HOST, ONCE. ``np.float32(value)`` is the same
    rounding NumPy performs when the Python float meets the float32 array in
    ``gs * chi2`` (weak promotion: a Python float does not widen a float32
    array). An unused pointer slot is bound to the component's own D array and
    never read -- a null pointer would be an unmapped address the compiler is
    entitled to speculate on.
    """
    out: Dict[str, Any] = {"nonlinear": [], "chi2": [], "chi3": [],
                           "chi2_is_volume": [], "chi3_is_volume": [],
                           "chi2_scalar": [], "chi3_scalar": []}
    for component, source, _axis in E_TERMS:
        displacement = getattr(fields, source)
        live = bool(fields.is_nonlinear(component))
        out["nonlinear"].append(live)
        for label, reader in (("chi2", fields.chi2_for), ("chi3", fields.chi3_for)):
            value = reader(component) if live else 0.0
            is_volume = live and getattr(value, "shape", None) not in (None, ())
            out[f"{label}_is_volume"].append(bool(is_volume))
            out[label].append(value if is_volume else displacement)
            out[f"{label}_scalar"].append(
                np.float32(0.0) if is_volume else np.float32(float(value)))
    return out


def boundary_codes(grid: Any) -> Tuple[Any, Any]:
    """The three integer ghost-rule codes, or a refusal. Exactly one is None.

    Asked through ``coverage.real_pml_boundary_kinds`` -- the same resolution the
    predicate performs -- so the launcher and the predicate cannot answer
    differently about one grid, which is the defect the certified curl pair
    carried for a day in August.
    """
    kinds = _coverage.real_pml_boundary_kinds(grid)
    for axis, kind in enumerate(kinds):
        if kind not in NONLINEAR_BOUNDARY_KINDS:
            return None, (f"axis {axis} resolves to boundary {kind!r}, which this "
                          f"kernel has no ghost rule for")
    return tuple(_coverage.BC_CODES[kind] for kind in kinds), None


def update_E_fused_pml_real_nonlinear(fields: 'Fields', pml: 'PML' = None, *,
                                      tables: Dict[str, Any] = None,
                                      codes: Tuple[int, int, int] = None,
                                      bindings: Dict[str, Any] = None) -> None:
    """``stepping.update_E`` under a PML with chi2/chi3 installed, in one launch.

    ``tables``, ``codes`` and ``bindings`` are THE GATE'S DOOR, keyword-only and
    named for what they are: the gate feeds deliberately mis-paired coefficient
    tables and swapped chi operands, and a launcher that could only derive them
    correctly could not arm those mutations. Supplying ``pml`` derives the tables
    from the SIDE through ``constitutive_sub_lattice`` -- the same function the
    predicate asks -- so the half-cell error is not one argument away on the entry
    point anything real would call.
    """
    if (pml is None) == (tables is None):
        raise ValueError(
            "pass exactly one of pml (the tables are derived from the side) or "
            "tables (the gate supplies its own, including mis-paired ones); got "
            f"pml={'set' if pml is not None else 'None'} and "
            f"tables={'set' if tables is not None else 'None'}")
    if tables is None:
        from . import constitutive_kernels  # noqa: PLC0415 - device-only import
        tables = constitutive_kernels.constitutive_tables_for("E", pml)
    if codes is None:
        codes, refusal = boundary_codes(fields.grid)
        if refusal is not None:
            raise ValueError(refusal)
    if bindings is None:
        bindings = nonlinear_bindings(fields)

    targets = (fields.Ex, fields.Ey, fields.Ez)
    nx, ny, nz = targets[0].shape
    blocks = (nx * ny * nz + _NONLINEAR_THREADS - 1) // _NONLINEAR_THREADS
    arguments = [
        fields.Ex, fields.Ey, fields.Ez,
        fields.f_w_Ex, fields.f_w_Ey, fields.f_w_Ez,
        fields.Dx, fields.Dy, fields.Dz,
        fields.inverse_epsilon_for("Ex"),
        fields.inverse_epsilon_for("Ey"),
        fields.inverse_epsilon_for("Ez"),
        bindings["chi2"][0], bindings["chi2"][1], bindings["chi2"][2],
        bindings["chi3"][0], bindings["chi3"][1], bindings["chi3"][2],
        bindings["chi2_scalar"][0], bindings["chi2_scalar"][1],
        bindings["chi2_scalar"][2],
        bindings["chi3_scalar"][0], bindings["chi3_scalar"][1],
        bindings["chi3_scalar"][2],
    ]
    arguments.extend(np.int32(1 if flag else 0) for flag in bindings["nonlinear"])
    arguments.extend(np.int32(1 if flag else 0) for flag in bindings["chi2_is_volume"])
    arguments.extend(np.int32(1 if flag else 0) for flag in bindings["chi3_is_volume"])
    arguments.extend([
        np.int32(nx), np.int32(ny), np.int32(nz),
        tables["kps_x"], tables["kms_x"],
        tables["kps_y"], tables["kms_y"],
        tables["kps_z"], tables["kms_z"],
        np.int32(codes[0]), np.int32(codes[1]), np.int32(codes[2]),
    ])
    kernel = _get_kernel('update_E_pml_real_nonlinear')
    kernel((blocks,), (_NONLINEAR_THREADS,), tuple(arguments))


# =============================================================================
# THE PREDICATE
# =============================================================================

def _nonlinearity_installed(fields: Any) -> bool:
    """Both maps by name, never ``Fields.has_nonlinearity``.

    That property reads ``_chi2_components`` ALONE (fields.py:966-967). MEEP's
    both-or-neither rule (structure.cpp:815-816, transcribed at fields.py:846-853)
    means a chi3-only install still populates chi2 with an explicit zero, so the
    property is correct on every object ``set_nonlinear_volumes`` built -- but a
    predicate that asked it would answer "linear" for a hand-assembled object
    carrying chi3 alone, and would then hand it to a kernel that multiplies by a
    Pade factor of 1. Reading both maps is the fail-closed spelling and is the
    same one every sibling clause uses.
    """
    return bool(getattr(fields, "_chi2_components", None)
                or getattr(fields, "_chi3_components", None))


def _chi_operand_problem(label: str, value: Any, xp: Any, shape: Tuple) -> Any:
    """Why this chi entry is not a float32 volume of ``shape`` or a finite scalar."""
    if getattr(value, "shape", None) in (None, ()):
        try:
            number = float(value)
        except Exception as exc:  # noqa: BLE001
            return f"{label} is neither a volume nor a number: {type(exc).__name__}: {exc}"
        if not np.isfinite(number):
            return f"{label} is {number!r}, not finite"
        return None
    return _coverage._array_problem(label, value, xp, shape)


def covers_real_pml_nonlinear_constitutive(fields: Any, pml: Any, grid: Any,
                                           side: str) -> tuple:
    """Does this family serve ``update_H``/``update_E`` on a NONLINEAR run?

    Returns ``(covered, reason)``. ``side="E"`` asks about
    ``update_E_pml_real_nonlinear`` in this module; ``side="H"`` asks about
    the CERTIFIED ``constitutive_kernels.update_H_pml_real``, which this
    family admits unchanged because ``stepping.update_H`` (:907-923) contains no
    reference to the nonlinearity -- see the module header, and see the gate,
    which measures it rather than arguing it.

    EVERY CLAUSE OF ``coverage.covers_real_pml_constitutive`` IS RESTATED, with
    exactly three differences, and each is written out so the diff is readable:

    1. **The chi2/chi3 clause is INVERTED.** That predicate refuses a
       nonlinearity; this one requires it. The two are therefore complementary on
       that clause and cannot both admit a slot, which is what keeps the census's
       disjointness check clean without a coordinating edit to either file.
    2. **A FOLD AND THE CYLINDRICAL AXIS ARE REFUSED** where the sibling admits
       them (``CONSTITUTIVE_FOLD_ADMISSION`` / ``CONSTITUTIVE_CYLINDRICAL_ADMISSION``).
       Those admissions were measured for an ELEMENT-WISE sub-step that reads no
       neighbour. This one reads four corners of two partner volumes, so a fold's
       parity-weighted mirror ghost and the axis row are live ghost rules here,
       and neither is transcribed. The refusal is narrower than the sibling's on
       purpose and the reason is the stencil, not the storage.
       (Even the E arm's own array path declines the folded-periodic pairing:
       ``_nonlinear_transverse_sums`` shifts PRODUCTS, which have no single
       component parity, so ``_shift_up`` is called with ``reflect_row`` unset and
       keeps the zero face -- and the driver refuses that combination rather than
       serving the zero as an answer, stepping.py:1785-1791.)
    3. **THE CHI OPERANDS ARE CHECKED**, which that predicate has no reason to:
       each of the six entries must be a float32 C-contiguous volume of the grid's
       shape or a finite scalar.

    THE POLE GUARD IS NOT A CLAUSE. ``stepping.nonlinear_margin`` (:1315-1357) is
    sampled OUTSIDE the step loop by ``driver._NonlinearityGuard`` and is unchanged
    by this kernel; a predicate that re-derived it here would be a second guard
    that can disagree with the first.
    """
    if side not in NONLINEAR_SIDES:
        raise ValueError(f"side must be one of {sorted(NONLINEAR_SIDES)}, got {side!r}")
    spec = _coverage.CONSTITUTIVE_SIDES[side]

    xp, backend = _coverage._backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    if getattr(fields, "force_complex_fields", False):
        return False, ("complex64 storage: MEEP's DOCMP loop builds a separate Pade "
                       "factor per Cartesian part (stepping.py:1110-1116)")
    if not (pml is not None and getattr(pml, "is_active", False)):
        return False, "no active PML layer"
    if not _nonlinearity_installed(fields):
        return False, ("no instantaneous chi2/chi3 is installed; a linear run belongs "
                       "to coverage.covers_real_pml_constitutive and MEEP's "
                       "trivial-pair drop (fields.py:879-880) is what makes the two "
                       "predicates disjoint by construction")
    facts, unreadable = _coverage._grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    if facts["cylindrical"] or any(facts["axis"]):
        return False, ("cylindrical (Dcyl): the transverse average reads the radial "
                       "neighbour and the r = 0 axis row has no ghost rule here")
    if facts["has_symmetry"] != any(facts["mirrored"]):
        return False, ("the grid reports has_symmetry() and no mirrored axis, or the "
                       "reverse; this predicate cannot answer for a grid that "
                       "disagrees with itself")
    if any(facts["mirrored"]):
        return False, ("mirror symmetry: the transverse sums read a neighbour, so the "
                       "fold's parity-weighted mirror ghost is live here -- unlike the "
                       "element-wise constitutive pair, which was measured on a fold")
    for axis, kind in enumerate(_coverage._boundary_kinds_from(facts)):
        if kind not in NONLINEAR_BOUNDARY_KINDS:
            return False, f"axis {axis} resolves to boundary {kind!r}, which has no kernel"
    if facts["has_bloch"]:
        return False, "nonzero Bloch k: the wrapped plane carries a phase real storage cannot hold"
    if facts["bfast_active"]:
        return False, "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"
    if side == "E":
        if getattr(fields, "polarizations", None):
            return False, ("dispersion: update_E's source is (D - sum P) per-component "
                           "scratch, not D, and update_P closes the step")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            return False, ("off-diagonal chi1inv: MEEP's most general branch scales the "
                           "WHOLE row product, coupling included (stepping.py:1095-1099)")
        if not getattr(fields, "stores_E", False):
            return False, "E is recomputed from D rather than stored"

    shape = facts["shape"]
    if len(shape) != 3:
        return False, f"grid shape {shape} is not three-dimensional"
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    if cells >= 2 ** 31:
        return False, f"{cells} cells exceeds the kernel's int32 index range"
    for name in tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"]):
        problem = _coverage._array_problem(name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem
    if side == "E":
        # THE THREE D VOLUMES ARE READ AS NEIGHBOURS, not only at ``idx``, so all
        # three must exist and be this grid's shape even for a run whose
        # nonlinearity lives on one component. ``spec["sources"]`` already checked
        # them; what follows is the epsilon and the chi operands.
        reader = getattr(fields, "inverse_epsilon_for", None)
        if not callable(reader):
            return False, "fields does not expose inverse_epsilon_for"
        for component in ("Ex", "Ey", "Ez"):
            try:
                volume = reader(component)
            except Exception as exc:  # noqa: BLE001 - a raise is not a refusal
                return False, f"inverse_epsilon_for({component!r}) raised {exc!r}"
            if volume is None:
                return False, f"inverse_epsilon_for({component!r}) is None"
            if not getattr(volume, "shape", ()):
                return False, f"inverse_epsilon_for({component!r}) is a scalar, not a volume"
            problem = _coverage._array_problem(f"inverse_epsilon_for({component!r})",
                                               volume, xp, shape)
            if problem is not None:
                return False, problem
        for reader_name, getter in (("chi2_for", getattr(fields, "chi2_for", None)),
                                    ("chi3_for", getattr(fields, "chi3_for", None))):
            if not callable(getter):
                return False, f"fields does not expose {reader_name}"
        checker = getattr(fields, "is_nonlinear", None)
        if not callable(checker):
            return False, "fields does not expose is_nonlinear"
        for component in ("Ex", "Ey", "Ez"):
            if not checker(component):
                continue
            for label, getter in (("chi2", fields.chi2_for), ("chi3", fields.chi3_for)):
                try:
                    value = getter(component)
                except Exception as exc:  # noqa: BLE001
                    return False, f"{label}_for({component!r}) raised {exc!r}"
                problem = _chi_operand_problem(f"{label}_for({component!r})",
                                               value, xp, shape)
                if problem is not None:
                    return False, problem
    suffix = "_h" if spec["half_integer"] else ""
    for axis, name in enumerate(("x", "y", "z")):
        for label in ("kps", "kms"):
            attribute = f"{label}_{name}{suffix}"
            problem = _coverage._coefficient_vector_problem(
                attribute, getattr(pml, attribute, None), xp, axis, shape)
            if problem is not None:
                return False, problem
    return True, None
