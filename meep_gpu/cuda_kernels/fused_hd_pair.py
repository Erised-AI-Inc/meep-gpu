"""The H->D weld: ``update_H`` welded into ``step_D``, one launch, scratch output.

THE FOURTH FUSION SEAM ON THIS BACKEND, and the first one whose two halves sit in
the order constitutive-then-curl. The driver runs

    ``update_H`` (driver.py:3311) -> the electric integrated-source withdraw
    (:3313-3314) -> ``step_D`` (:3315)

and exactly one statement sits between the two consults: ``for source in electric:
getattr(source, "withdraw", _no_withdraw)(self.fields)``. Nothing is INJECTED here --
the electric injection is one seam later (:3317-3322) and the magnetic one is one seam
earlier (:3283-3284) -- so :mod:`..deposit_repair` has nothing to say about this seam
and :mod:`..withdraw_hoist` is the module that does.

=============================================================================
THE SHAPE: SCRATCH OUTPUT AND FOREIGN-CELL RECOMPUTE
=============================================================================

``step_D``'s curl reads ``H`` at the thread's own cell AND at its three BACKWARD
neighbours (``shift_dn``, step_curl_kernels.py's ``_REAL_PML_PRELUDE``), and
``update_H`` writes ``H``. Welded in place, thread ``idx`` would read
``H[idx - stride]`` at a moment decided by the schedule -- the hazard every board on
this track files as structurally unweldable -- and it would read ``f_w_H`` the same
way, which is easy to miss: on this side the newly stored ``f_w_H`` IS ``B`` exactly
(``constitutive_apply``'s ``fw[idx] = src`` with ``src = B[idx]``), so an in-place
write hands a racing neighbour ``B`` where its recurrence needs ``B_prev``.

Both premises are removed the way :mod:`.offdiag_stencil_weld` removes them on the
MIRROR-IMAGE D->E seam, and the way :mod:`..metal_kernels.fused_hd_pair` removes them
on this one:

* **the constitutive half writes nothing in place.** ``H_new`` and ``f_w_H_new`` go to
  LAUNCH-LOCAL SCRATCH, so ``H``, ``f_w_H`` and ``B`` are ``const`` for the whole
  launch and no thread can observe another thread's store;
* **the foreign read is not a read of another thread's output.** It is a RECOMPUTE
  from that same unwritten state, through :func:`raw_update_H_cell_source` -- the
  certified ``update_H_pml_real`` body, whole, evaluated at an arbitrary cell. A pure
  function of unwritten memory has no schedule to depend on.

The launcher then ROTATES the ``H``/``f_w_H`` bindings, which is the choreography
:func:`.offdiag_fused_electric_pair.rotate_into_fields` already ships on the D/fu_D
pair and ``ade_kernels.update_P_fused_pml_real`` ships on ``P``/``P_prev``.

WHY THE RECOMPUTE IS EXACT RATHER THAN CLOSE. ``update_H`` is POINTWISE: per component
it reads ``H``, ``f_w_H`` and ``B`` at ONE cell and writes the same cell
(stepping.py:907-923 through ``_apply_constitutive_pml``, stepping.py:2112-2143; the
kernel is ``constitutive_kernels._update_H_pml_real_kernel_code``, which reads no
neighbour and carries no ownership mask -- that module's note 4). So the stepped value
at any cell is a function of state this launch does not write, and evaluating it twice
gives the same bits by construction rather than by a tolerance. That is CHEAPER than
the D->E weld's recompute, whose recomputed half is itself a curl.

WHY THE COEFFICIENT INDEX NEEDS NO THEOREM. The recompute takes the FLAT INDEX the
certified ``shift_dn`` composed and runs the certified body's OWN decode on it, so
``kps_x[i]`` and ``kms_x[i]`` are indexed at the recomputed cell exactly as the
certified body indexes them at the thread's own. The recompute is the certified
arithmetic at the recomputed cell, not an approximation of it.

=============================================================================
ONE DEVICE STRING, AND THAT IS THIS BACKEND'S OWN FACT
=============================================================================

The boundary kind is a RUNTIME ARGUMENT here (``bc_x``/``bc_y``/``bc_z``, with
``BC_MIRROR_PERIODIC`` alongside ``BC_METALLIC`` and ``BC_PERIODIC`` --
step_curl_kernels.py's ``_REAL_PML_PRELUDE``), where the sibling Metal backend bakes
the triple into the shader text and expresses a fold as a SEPARATE ARM. So this one
product covers what the other two backends need two products for: the CUDA cell
``(update_H cuda_constitutive/ordinary -> step_D cuda_curl/PML)`` is **127 rows**
where Metal's ``(ordinary -> PML)`` is 49 and its ``(folded -> folded)`` a further 78
(``parity/meep_gpu/results/h_to_d_seam_2026-09-04/summary.json``,
``per_backend.cuda.cells`` and ``per_backend.metal.cells``).

One device string also means ONE certification digest, which is this track's standing
rule: no per-boundary variant can end up pinned that nobody ran.

=============================================================================
WHAT SITS IN THE SEAM, AND WHY THIS PRODUCT REFUSES THE WITHDRAW ROWS
=============================================================================

:data:`CARRIES_DEPOSIT_REPAIR` is False and it is a FACT rather than a decision:
nothing is injected between the two consults, so there is no deposit to bracket. The
pass that IS there is the electric withdraw, and :mod:`..withdraw_hoist` is its
contract -- one slot, not two; a hoist, not a bracket.

:data:`HOISTS_THE_WITHDRAW` is False IN THIS ROUND, and the flag and the wiring move
together exactly as ``CARRIES_DEPOSIT_REPAIR``'s do. The only wiring that performs the
hoist is ``fused_pairs._install_fused_pair``'s ``withdraw_hoist.SEAM`` branch, which
wraps the built plan in a :class:`..withdraw_hoist.LeadingWithdrawPlan`; this product
declares :data:`INSTALLABLE` False, so ``fused_pairs._declared_uninstallable`` refuses
it before the seam loop installs anything and that branch is unreachable for it --
NOTHING would perform the withdraw before its launch. A product that declared True
without reachable wiring would let the fused launch read a D still holding the
previous step's standing dipole and report success, the exact failure
``withdraw_hoist`` exists to prevent. So the predicate refuses, BY NAME, every row
whose electric withdraw does work.

WHAT THAT COSTS, MEASURED rather than estimated
(``parity/meep_gpu/results/h_to_d_seam_2026-09-04/h_to_d_seam.jsonl``, joined on the
CUDA arms): the ``(cuda_constitutive/ordinary -> cuda_curl/PML)`` cell is **127 rows**,
of which **122** are ``buildable_not_built`` and **5** are ``withdraw_seam`` --
``examples:absorbed_power_density.py``, ``examples:differential_cross_section.py``,
``examples:finite_grating.py``, ``examples:mie_scattering.py`` and
``tests:TestIntegratedSource.test_integrated_source``. This predicate therefore
reaches **122 of the cell's 127**, and what would raise it to 127 is a package of two
things landed together: :data:`INSTALLABLE` True, and a device gate leg that drives a
complete step on a row with a live INTEGRATED electric source with the hoist in place
and requires both the un-hoisted arrangement and the ``after_step_D`` placement to
diverge.

=============================================================================
IT IS NOT INSTALLED, AND THE REASON IS A MEASURED VERDICT RATHER THAN A MISSING GATE
=============================================================================

:data:`INSTALLABLE` is False, and :data:`INSTALLABLE_REASON` carries the measurement
rather than a policy. Over the driver's ``step_B - update_H - step_D - update_E`` slot
path launches are ``4 - (installed pairs)``, and a product spanning
``update_H``/``step_D`` takes one slot from EACH neighbour, so a span taking one slot
from each neighbour can only tie or lose. Only a four-slot
``step_B -> update_H -> step_D -> update_E`` weld is strictly additive, and it is not
built here.

WHAT KEEPS THE RELEASED B->H PAIR IN ITS SLOT IS NOT THIS FLAG. It is
``fused_pairs._neighbouring_seam_claimant``'s end-edge guard (a B->H span is an END
EDGE of ``FUSED_PAIR_SEAMS`` by the table's own degrees and is never asked whether a
neighbour claims its seam) together with ``_spans_may_absorb`` reading the live
``selected`` after that pair installs first. The flag is kept as belt and braces; the
composition does not rely on it.

What this product IS worth is stated plainly, because no artifact here may imply
otherwise: the priced predicate gap closed under the project's rule *close all fusion gaps
regardless*, and a certified half of the only span that could ever pay. **No timing
exists for this shape and none is licensed by anything in this module.** The fused
route does MORE memory traffic than the two singles for the same step-level launch
count; whether the saved launch and the saved ``H`` write-then-read round trip pay for
up to six extra pointwise constitutive evaluations per thread is a hypothesis.

=============================================================================
THE NONLINEAR CELL IS THE ONE WIDENING, AND IT IS A MODULE IDENTITY
=============================================================================

A chi2/chi3 run selects ``cuda_nonlinear``'s ``nonlinear`` arm at ``update_H`` and the
ordinary ``cuda_curl``/``PML`` arm at ``step_D``
(``h_to_d_seam.jsonl``: ``examples:3rd-harm-1d.py`` and
``tests:Test3rdHarm1d.test_3rd_harm_1d``, 2 rows). On THIS backend that widening is
not a byte comparison between two emitted texts, because there is only one text:
:mod:`.nonlinear_constitutive` ships NO ``update_H`` kernel at all. Its arm H is a
NULL WIDENING -- ``covers_real_pml_nonlinear_constitutive(..., side="H")`` admits the
SHIPPED ``constitutive_kernels.update_H_pml_real`` unchanged, because
``stepping.update_H`` (:907-923) reads no chi (that module's docstring, "Arm H (NULL
WIDENING). No kernel."), and the null is MEASURED by
``parity/meep_gpu/gate_cuda_nonlinear.py`` with its own premise armed as a defect
(``pade_scale_the_H_source``, CAUGHT 6/6). The census agrees from the other side: both
nonlinear rows record ``update_H_kernel: fused_update_H_pml_real``, the ordinary arm's
own label.

So :func:`covers_fused_hd_pair` admits a nonlinear run through those two arms' own
predicates and ``fused_pairs.FUSED_PAIR_EXTRA_ARMS`` carries the matching absorb row.
The two predicates are DISJOINT on ``_chi2_components or _chi3_components`` -- the
ordinary one refuses when either is present and the nonlinear one refuses when neither
is -- so exactly one can answer and the disjunction is a lookup rather than a widening.

=============================================================================
WHAT THIS FAMILY DOES NOT CARRY
=============================================================================

Every other configuration is refused through the two halves' OWN certified predicates,
which this one conjoins without weakening: complex64 storage, a Bloch phase, BFAST,
``grid.beta``, a conductivity on ``step_D``'s target, an inactive absorber, the
cylindrical r = 0 axis, a Dcyl grid (the cylindrical ``step_D`` differences a
column-serial radial scan that is deliberately not a kernel on this track), and any
grid whose two halves resolve different boundary triples.

THE FOLD IS ADMITTED, and by the two halves' own device verdicts rather than by
argument: ``coverage.CONSTITUTIVE_FOLD_ADMISSION`` for the constitutive half and
``BC_MIRROR_PERIODIC`` for the curl, whose mask split was measured by
``gate_cuda_folded_curl.py`` (``results/cuda_folded_curl_2026-08-19/``). The seam
itself is fill-FREE on a folded grid and that is a fact about the driver, not a hope:
``fill_B``/``fill_folded_far_ghosts_B`` close one seam earlier (driver.py:3305-3310)
and ``fill_D`` opens one seam later (:3324-3330), so no ghost plane is rewritten
between the two consults this launch spans.

=============================================================================
NOTHING DISPATCHES THIS
=============================================================================

``meep_gpu.fastpath.plan_fast_path`` builds ``triton_kernels`` plans and never names
this package. This module ships a predicate, an emitter and a launcher;
``fused_pairs`` can plan it opt-in (``arms.plan_step(..., fuse=True)``) and refuses to
INSTALL it on the declaration above, so no shipped code path launches it.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import coverage as _coverage
from . import own_cell_hoist
from .compile_cache import (clear_kernel_cache as _clear_cache,
                            get_or_compile as _get_or_compile,
                            kernel_cache_key as _kernel_cache_key)
from .coverage import (constitutive_sub_lattice, covers_real_pml_constitutive,
                       covers_real_pml_curl)
from .nonlinear_constitutive import covers_real_pml_nonlinear_constitutive

from .. import withdraw_hoist as _withdraw_hoist

# The two certified halves' device text. Taken DEFENSIVELY: both modules import CuPy
# at scope, and the PREDICATE half of this family must answer on a host with no
# device -- that is where the census asks it. The emitter refuses BY NAME where they
# are absent.
try:
    from . import step_curl_kernels
except Exception:  # noqa: BLE001 - no CuPy, or no device toolchain
    step_curl_kernels = None  # type: ignore[assignment]
try:
    from . import constitutive_kernels
except Exception:  # noqa: BLE001
    constitutive_kernels = None  # type: ignore[assignment]

FAMILY = "cuda_fused_hd_pair"

#: The kernel's entry-point symbol, spelled once. Deliberately NOT either half's name:
#: a third kernel wearing a certified one's name would make ``certification.json``'s
#: partition test and any NVRTC binary observation ambiguous about which body it saw.
KERNEL_NAME = "fused_hd_pair_pml_real"

#: The sub-step slot this arm is registered on: the FIRST half of the seam, in the
#: driver's own order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes ONE launch of this kernel performs, in driver order
#: (driver.py:3311, :3315). Declared rather than inferred from the slot names.
#:
#: NOTHING BETWEEN THEM IS CARRIED, and there is only one thing there to carry: the
#: electric withdraw (:3313-3314), which :data:`HOISTS_THE_WITHDRAW` declines in this
#: round. It is not in this tuple because the product does not perform it.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``fused_pairs.FUSED_PAIR_SEAMS`` files this row under, and the module
#: that owns its in-seam pass. Spelled through :mod:`..withdraw_hoist` rather than as a
#: literal so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

#: The volumes the launch writes to SCRATCH and the launcher rotates afterwards, in
#: SIGNATURE ORDER. Named for the field each shadows.
SCRATCH_VOLUMES: Tuple[str, ...] = ("Hx", "Hy", "Hz",
                                    "f_w_Hx", "f_w_Hy", "f_w_Hz")

#: The three magnetic components, in signature and component-tag order -- the certified
#: ``update_H`` body's own spelling.
H_TARGETS: Tuple[str, str, str] = ("Hx", "Hy", "Hz")

#: BYTE-IDENTICAL TO FOUR INDEPENDENT ARRANGEMENTS OF THE SEAM, WITH A DEVICE VERDICT
#: BEHIND IT. ``certification.json``'s ``cuda_fused_hd_pair_2026-09-06`` block names
#: the artifact this line rests on, and ``test_kernel_partition.py`` refuses a name
#: here that no record block claims.
#:
#: THE DECLARATION IS THE FINAL BYTES, and that is this track's rule rather than a
#: claim made ahead of its evidence: a gate certifies the module it IMPORTED, so a
#: module that moved its own name between the two sets AFTER the run would leave the
#: record bound to bytes that no longer ship. The released campaign was cut against
#: THESE bytes, with the name already here; the earlier passes that measured the
#: design are not what the record binds.
CERTIFIED_KERNELS: Tuple[str, ...] = (
    "fused_hd_pair_pml_real",
)

#: EMPTY, and what emptied it was the run above rather than an argument. Spelled as a
#: dict WITHOUT a type annotation for the reason ``constitutive_kernels.py`` records:
#: the partition test reads this name off the syntax tree without importing the
#: module, and an annotated assignment is an ``ast.AnnAssign``, which that reader does
#: not match.
UNCERTIFIED_KERNELS = {}

#: Does this product bracket its launch with the DEPOSIT repair? NO, and it is a fact
#: about the driver rather than a choice: nothing is injected between the ``update_H``
#: consult (driver.py:3311) and the ``step_D`` consult (:3315), so there is no deposit
#: in this seam to carry across the launch. The electric injection is one seam later
#: (:3317-3322) and the magnetic one is one seam earlier (:3283-3284); this predicate
#: therefore does NOT consult ``deposit_repair.seam_source_reasons`` at all, and a
#: source of either polarity is not this seam's business.
CARRIES_DEPOSIT_REPAIR = False

#: Does this product perform the seam's electric withdraw before its launch? NO in this
#: round. See the module docstring: the only wiring that performs it is
#: ``fused_pairs._install_fused_pair``'s :data:`..withdraw_hoist.SEAM` branch, which
#: this product never reaches (:data:`INSTALLABLE` False), so True here would be a
#: claim about wiring that cannot fire. The predicate consequently refuses every row
#: with a standing in-seam withdraw BY NAME -- 5 of the cell's 127.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on every configuration, for a MEASURED
#: VERDICT rather than a policy. ``fused_pairs._declared_uninstallable`` reads this and
#: reports :data:`INSTALLABLE_REASON` on every run.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "slot arbitration, measured rather than preferred. Over the driver's step_B - "
    "update_H - step_D - update_E slot path launches are 4 - (installed pairs), and "
    "this product takes one slot from EACH neighbouring seam: the released B->H pair "
    "holds step_B/update_H and the released D->E pair holds step_D/update_E, so a span "
    "taking one slot from each can only TIE (one neighbour installs: 3 launches / 1 "
    "seam either way) or LOSE (both install: 2 pairs -> 1, one more launch and one "
    "fewer seam served). The only strictly additive shape on this path is a four-slot "
    "step_B -> update_H -> step_D -> update_E weld, which is not built here. The "
    "sibling Metal composer was driven over the 179 buildable rows of the 194-row "
    "basis on 2026-09-04 and reported 133 LOSS, 22 TIE resolved for the incumbent, 22 "
    "refused in a D->E claimant's favour and 2 gain (the nonlinear cell, where neither "
    "neighbour installs) -- 0 rows on which displacing a released pair is a gain. This "
    "is a verdict about the COMPOSITION, not about the arithmetic, which this family's "
    "own device gate measures. What keeps the released B->H pair in its slot is "
    "_neighbouring_seam_claimant's end-edge guard plus _spans_may_absorb reading the "
    "live selected; this flag is belt and braces in front of them")

#: What a released verdict on this family would and would not license. Said once, here,
#: so every artifact that quotes this module quotes the same sentence.
WHAT_A_RELEASE_DOES_NOT_LICENSE = (
    "SERVED on the fusion board is PREDICATE ADMISSION by the board's own definition, "
    "so a released product credits its admitted seam-instances while executing "
    "NOWHERE: nothing under meep_gpu/ imports cuda_kernels outside tests, this family "
    "declares INSTALLABLE = False so the composer refuses to install it on every "
    "configuration, and no timing of any kind has been taken of this shape.")

#: Lanes per block. The two certified halves' own -- ``_REAL_PML_THREADS`` and
#: ``_CONSTITUTIVE_THREADS`` are both 256, and this kernel is one element per lane on
#: a flat 1-D grid like both of them.
_FUSED_THREADS = 256

#: NVRTC options, restated rather than imported for the reason
#: ``offdiag_fused_electric_pair`` restates them: loading this file by path (which the
#: probes do) must not pick up a different tuple than the one the gate compiled.
#: ``--fmad=false`` is CORRECTNESS here and not tuning: ``a - kms * prev``,
#: ``f + kps * src``, ``(fu * kms) - curl`` and ``(f * kms_u) + fu_new`` are all FMA
#: candidates the array path rounds twice.
_COMPILE_OPTIONS: Tuple[str, ...] = ("--fmad=false",)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CERTIFIED_KERNELS", "CONSTITUTIVE_LIFT_EDITS",
    "CURL_LIFT_EDITS", "FAMILY", "HALO_TAPS", "HOISTS_THE_WITHDRAW", "H_TARGETS",
    "INSTALLABLE", "INSTALLABLE_REASON", "KERNEL_NAME", "OWN_LOAD_EDITS", "REPLACES",
    "SCRATCH_VOLUMES", "SEAM", "SLOT", "UNCERTIFIED_KERNELS",
    "WHAT_A_RELEASE_DOES_NOT_LICENSE", "assert_scratch_is_disjoint",
    "constitutive_prelude", "covers_fused_hd_pair", "curl_prelude", "device_sources",
    "fused_hd_pair_codes", "fused_hd_pair_scratch", "fused_hd_pair_tables",
    "kernel_source", "launch_fused_hd_pair", "raw_update_H_cell_source",
    "resolution_source", "rotate_into_fields", "run_fused_hd_pair",
    "shift_dn_recompute_source", "signature", "source_digest", "welded_curl_tail",
    "weld_args_construction", "weld_args_struct",
]


# ---------------------------------------------------------------------------
# The declared edits — DATA, so a gate and the host suite can assert the list
# ---------------------------------------------------------------------------

#: Separates a certified kernel's signature from its body. Every certified string on
#: this track closes its parameter list on its own line.
_BODY_ANCHOR = "\n) {\n"

#: The certified curl/constitutive thread preamble -- the block a ``__device__``
#: function evaluated at an ARBITRARY cell cannot keep.
_THREAD_PREAMBLE = ("    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
                    "    if (idx >= nx * ny * nz) return;\n")

#: The certified curl body's index decode, whose LAST line is where this weld splices
#: its constitutive half in. Matched whole so a reordered decode raises.
_INDEX_DECODE = ("    int k = idx % nz;\n"
                 "    int j = (idx / nz) % ny;\n"
                 "    int i = idx / (ny * nz);\n")

#: ``constitutive_apply``'s three anchors, and what the PURE form makes of each.
_APPLY_SIGNATURE = "__device__ __forceinline__ void constitutive_apply(\n"
_APPLY_SIGNATURE_PURE = "__device__ __forceinline__ float constitutive_apply_pure(\n"
_APPLY_PARAMETERS = (
    "    float* __restrict__ f, float* __restrict__ fw, int idx, float src,\n"
    "    float kps, float kms\n)")
_APPLY_PARAMETERS_PURE = (
    "    const float* __restrict__ f, const float* __restrict__ fw, int idx,\n"
    "    float src,\n"
    "    float kps, float kms, float* fw_out\n)")
_APPLY_FW_STORE = "    fw[idx] = src;\n"
_APPLY_FW_STORE_PURE = (
    "    // THE FIRST OF THE TWO STORES, HANDED BACK INSTEAD OF PERFORMED. The\n"
    "    // caller writes it to the SCRATCH volume, so nothing this launch reads is\n"
    "    // ever a word this launch wrote -- and on this side the stored value IS\n"
    "    // B[idx], so an in-place write would hand a racing neighbour B where its\n"
    "    // recurrence needs B_prev.\n"
    "    *fw_out = src;\n")
_APPLY_F_STORE = "    f[idx] = a - kms * prev;\n"
_APPLY_F_STORE_PURE = (
    "    // THE SECOND STORE, LIKEWISE. Same right-hand side, same tree, same\n"
    "    // operand order, same two SEPARATE accumulations; the value is RETURNED so\n"
    "    // the caller can store it to scratch for its own cell and consume it from a\n"
    "    // register for the curl half.\n"
    "    return a - kms * prev;\n")

#: ``shift_dn``'s declaration and its two leaf loads. The loads' INDEX EXPRESSIONS are
#: never written here: they are parsed out of the certified lines by
#: :func:`shift_dn_recompute_source`, so a change in how the certified helper spells a
#: neighbour's index RAISES instead of silently redirecting a tap.
_SHIFT_DN_SIGNATURE = ("__device__ __forceinline__ float shift_dn(\n"
                       "    const float* __restrict__ g, int idx, int ia, int na, "
                       "int stride, int bc\n)")
_SHIFT_DN_SIGNATURE_RECOMPUTE = (
    "__device__ __forceinline__ float shift_dn_recompute(\n"
    "    int comp, int idx, int ia, int na, int stride, int bc,\n"
    "    const WeldArgs& weld\n)")

#: Every line of certified CONSTITUTIVE text this weld does not lift verbatim, with
#: the reason. DATA, not prose. THERE IS NO ARITHMETIC IN THIS TABLE: every entry is a
#: store that moves to the caller, a pointer qualifier, or the thread binding.
CONSTITUTIVE_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ void constitutive_apply(",
     "became": "__device__ __forceinline__ float constitutive_apply_pure(",
     "why": "the weld writes a SCRATCH volume, so the helper must not store. It "
            "returns the stepped field and hands the split-field value back through "
            "an out-parameter; f and fw become const because nothing in this launch "
            "writes them, which is what makes a foreign recompute a pure function of "
            "unwritten memory."},
    {"line": "    fw[idx] = src;",
     "became": "    *fw_out = src;",
     "why": "the store moves to the caller, which addresses the scratch for its own "
            "cell and stores nothing at all for a foreign recompute. The value and "
            "the expression that produced it are untouched."},
    {"line": "    f[idx] = a - kms * prev;",
     "became": "    return a - kms * prev;",
     "why": "same right-hand side, same tree, same operand order; the result is "
            "returned rather than stored because the curl half consumes the thread's "
            "own cell from a register. The TWO SEPARATE ACCUMULATIONS -- `a = f[idx] "
            "+ kps*src` then `a - kms*prev`, constitutive_kernels.py's note 1 -- are "
            "not touched, and regrouping them is an armed gate mutation."},
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n"
             "    if (idx >= nx * ny * nz) return;",
     "became": "(removed -- idx is a parameter of raw_update_H_cell)",
     "why": "THE WHOLE POINT OF THE LIFT. The certified body becomes a __device__ "
            "function evaluated at an ARBITRARY cell rather than at the thread's own. "
            "The bounds guard leaves with the thread index: every caller is either "
            "the kernel's own guarded thread or an index the certified shift_dn "
            "composed, which is in range by that helper's own branch (ia > 0 puts "
            "idx - stride inside the volume; the periodic arm's idx + (na-1)*stride "
            "is the top plane), and the ghost arm returns an exact 0.0f without "
            "calling this at all."},
    {"line": "    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);",
     "became": "(kept, verbatim, inside raw_update_H_cell)",
     "why": "NOT AN EDIT, and listed because a reader will look for one. The decode "
            "is the certified body's own and is what turns the recomputed cell's flat "
            "index into the per-axis coefficient index kps_x[i] / kps_y[j] / "
            "kps_z[k]. Dropping it and reusing the thread's i/j/k would index the "
            "absorber profile at the WRONG cell -- converged, smooth and wrong."},
    {"line": "    constitutive_apply(H*, f_w_H*, idx, B*[idx], kps_*[*], kms_*[*]);",
     "became": "    h_out[n] = constitutive_apply_pure(H*, f_w_H*, idx, B*[idx], "
               "kps_*[*], kms_*[*], &w_out[n]);",
     "why": "capture both values. The argument list, its order and the coefficient "
            "pairing are untouched; what is added is where the two results go. Three "
            "lines, one per component, each derived from the certified line by prefix "
            "and suffix rather than retyped."},
)

#: Every line of certified CURL text this weld does not lift verbatim, with the reason.
CURL_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "__device__ __forceinline__ float shift_dn(\n"
             "    const float* __restrict__ g, int idx, int ia, int na, int stride, "
             "int bc\n)",
     "became": "__device__ __forceinline__ float shift_dn_recompute(\n"
               "    int comp, int idx, int ia, int na, int stride, int bc,\n"
               "    const WeldArgs& weld\n)",
     "why": "THE FOREIGN TAP. The partner VOLUME becomes a COMPONENT TAG, because the "
            "value the curl needs at a neighbour is not a stored word any more -- it "
            "is update_H's result there, recomputed from pre-launch state. The tag is "
            "a compile-time constant at every call site. The certified shift_dn stays "
            "in the prelude, untouched and unused by this kernel, because the weld "
            "lifts the whole certified curl PRELUDE verbatim."},
    {"line": "    if (ia > 0) return g[idx - stride];",
     "became": "    if (ia > 0) return resolve_H(comp, idx - stride, weld);",
     "why": "the branch, the comparison and the index expression are the certified "
            "helper's own and are PARSED out of that line rather than retyped; only "
            "the leaf load moves from a volume another thread would be writing to a "
            "recompute from state nobody writes."},
    {"line": "    return (bc == BC_PERIODIC) ? g[idx + (na - 1) * stride] : 0.0f;",
     "became": "    return (bc == BC_PERIODIC) ? resolve_H(comp, idx + (na - 1) * "
               "stride, weld) : 0.0f;",
     "why": "the same, for the wrap. THE 0.0f IS UNTOUCHED and that is the point of "
            "lifting the helper rather than rewriting the call sites: a metallic or "
            "mirror-periodic ghost that read an exact 0.0f still reads an exact "
            "0.0f, and the ternary is what keeps an out-of-range recompute from being "
            "evaluated at all."},
    {"line": "        float f1 = H*[idx];   /  float f2 = H*[idx];",
     "became": "        float f1 = own_h[n];  /  float f2 = own_h[n];",
     "why": "the curl's OWN-cell magnetic loads. The thread has already computed its "
            "own H into a register and that register is the post-update_H value the "
            "array path would have loaded. Leaving the load standing would read the "
            "PRE-launch H, which is the armed mutation `own_cell_reads_stale`."},
    {"line": "        float sf = shift_dn(H*, idx, <axis>, <extent>, <stride>, "
             "<bc>);",
     "became": "        float sf = shift_dn_recompute(<tag>, idx, <axis>, <extent>, "
               "<stride>, <bc>, weld);",
     "why": "the curl's six SHIFTED magnetic loads. Only the first argument changes: "
            "the pointer becomes its component tag, read off the certified line "
            "rather than tabulated here, and every other argument -- the axis "
            "coordinate, the extent, the stride and the boundary code -- is carried "
            "across character for character."},
    {"line": "    int idx = blockIdx.x * blockDim.x + threadIdx.x; ... "
             "int i = idx / (ny * nz);",
     "became": "(kept, verbatim, as the fused kernel's own preamble)",
     "why": "NOT AN EDIT. The fused kernel keeps the CURL's preamble and decode as "
            "its own -- one bounds guard, one decode -- and the constitutive half's "
            "copy is dropped instead. Splicing both would redeclare idx, i, j and k "
            "and the kernel would not compile."},
)

#: The curl's six SHIFTED magnetic loads, as ``(register name, ...)`` is NOT written
#: here on purpose: WHICH component and WHICH cell each tap reads is parsed out of the
#: certified body's own ``shift_dn(...)`` call lines by :func:`welded_curl_tail`. This
#: constant records only HOW MANY there are, so a certified curl that grew or lost a
#: tap is a named failure rather than a silently partial redirect.
HALO_TAPS = 6

#: The same for the own-cell loads: two per component, ``f1`` and ``f2``.
OWN_LOAD_EDITS = 6

#: Every member of the pack every ``__device__`` helper takes, in emission order. A
#: struct rather than eighteen parameters threaded through three helpers: the members
#: are built once from the kernel's own arguments, are uniform across the block, and
#: name exactly the state ``update_H`` reads.
WELD_ARGS_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("const float* __restrict__", "Hx"), ("const float* __restrict__", "Hy"),
    ("const float* __restrict__", "Hz"),
    ("const float* __restrict__", "f_w_Hx"), ("const float* __restrict__", "f_w_Hy"),
    ("const float* __restrict__", "f_w_Hz"),
    ("const float* __restrict__", "Bx"), ("const float* __restrict__", "By"),
    ("const float* __restrict__", "Bz"),
    ("const float* __restrict__", "kps_x"), ("const float* __restrict__", "kps_y"),
    ("const float* __restrict__", "kps_z"),
    ("const float* __restrict__", "kms_x"), ("const float* __restrict__", "kms_y"),
    ("const float* __restrict__", "kms_z"),
    ("int", "nx"), ("int", "ny"), ("int", "nz"),
)


# ---------------------------------------------------------------------------
# The lift
# ---------------------------------------------------------------------------

def _certified() -> None:
    if step_curl_kernels is None or constitutive_kernels is None:
        raise RuntimeError(
            "step_curl_kernels / constitutive_kernels are not importable on this host "
            "(they import CuPy at module scope), so there is no certified text to "
            "splice. covers_fused_hd_pair needs neither and still answers")


def _needle(source: str, old: str, new: str, what: str) -> str:
    """Replace ``old`` exactly once, or raise naming what stopped matching.

    A missing anchor is certified text that changed under this family, and splicing
    around it would emit a kernel that compiles and is quietly not the certified
    arithmetic; TWO matches would mean the anchor no longer identifies one statement.
    """
    count = source.count(old)
    if count != 1:
        raise AssertionError(
            f"the certified text carries {count} copies of {what}, not one; this weld "
            f"LIFTS that text rather than retyping it and cannot splice around its "
            f"absence")
    return source.replace(old, new, 1)


def _the_line_starting(body: str, prefix: str, what: str) -> str:
    matches = [line for line in body.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise AssertionError(
            f"the certified body carries {len(matches)} lines starting {prefix!r}; "
            f"this weld lifts exactly one line for {what}")
    return matches[0]


def _split_body(source: str, prelude: str, name: str) -> str:
    """One certified kernel string, reduced to its body."""
    if not source.startswith(prelude):
        raise AssertionError(
            f"{name} no longer begins with the prelude this module lifts separately; "
            f"the splice would emit it twice")
    tail = source[len(prelude):]
    if tail.count(_BODY_ANCHOR) != 1:
        raise AssertionError(
            f"{name} carries {tail.count(_BODY_ANCHOR)} signature terminators, not 1; "
            f"the body anchor no longer identifies the signature")
    body = tail.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(f"{name} does not end with a closing brace")
    return body[: -len("}\n")]


def curl_prelude() -> str:
    """``step_curl_kernels._REAL_PML_PRELUDE``, VERBATIM.

    Nothing in it is edited. ``shift_up`` and ``shift_dn`` read ``H``, which this
    launch does not write, and ``pml_apply`` writes ``D``/``fu_D`` at the thread's OWN
    cell only -- which is safe in place by construction, since no thread reads a
    displacement another thread wrote. The three boundary ``#define``s come from here
    and from nowhere else, which is why the constitutive prelude (which defines none)
    can follow without a duplicate.
    """
    _certified()
    return step_curl_kernels._REAL_PML_PRELUDE  # noqa: SLF001


def constitutive_prelude() -> str:
    """``constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE`` with the helper made PURE.

    Four anchored edits, all of them in :data:`CONSTITUTIVE_LIFT_EDITS`. The
    arithmetic -- ``prev``, the two separate accumulations and their order -- is
    untouched.
    """
    _certified()
    prelude = constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE  # noqa: SLF001
    prelude = _needle(prelude, _APPLY_SIGNATURE, _APPLY_SIGNATURE_PURE,
                      "constitutive_apply's declaration")
    prelude = _needle(prelude, _APPLY_PARAMETERS, _APPLY_PARAMETERS_PURE,
                      "constitutive_apply's parameter list")
    prelude = _needle(prelude, _APPLY_FW_STORE, _APPLY_FW_STORE_PURE,
                      "constitutive_apply's split-field store")
    prelude = _needle(prelude, _APPLY_F_STORE, _APPLY_F_STORE_PURE,
                      "constitutive_apply's field store")
    return prelude


def weld_args_struct() -> str:
    """The argument pack, declared."""
    lines = [
        "",
        "// THE PRE-LAUNCH STATE, packed once. Every member is read-only for the whole",
        "// launch: that is the property the recompute rests on, and binding H, f_w_H",
        "// and B const here is where it is stated in the type system rather than in a",
        "// comment.",
        "struct WeldArgs {",
    ]
    for kind, name in WELD_ARGS_FIELDS:
        lines.append(f"    {kind} {name};")
    lines.append("};")
    return "\n".join(lines) + "\n"


def weld_args_construction(indent: str = "    ") -> str:
    """The struct, built from the kernel's own parameters. One line per member."""
    lines = [f"{indent}WeldArgs weld;"]
    for _kind, name in WELD_ARGS_FIELDS:
        lines.append(f"{indent}weld.{name} = {name};")
    return "\n".join(lines) + "\n"


def raw_update_H_cell_source() -> str:
    """The certified ``update_H_pml_real`` body as a ``__device__`` function of a CELL.

    THE ENTIRE BODY IS LIFTED. Only the thread preamble and the three
    ``constitutive_apply`` call lines are edited, each through an exact anchor, so a
    change in the certified text is a named failure rather than a silently different
    kernel. The decode, the component-to-axis pairing and the ``B*[idx]`` sources
    arrive character for character.

    Both results come back through out-parameters: ``h_out[c]`` is the stepped
    magnetic field and ``w_out[c]`` the split-field history, which the caller stores to
    scratch for its own cell and discards for a foreign recompute.
    """
    _certified()
    body = _split_body(own_cell_hoist.unhoisted_kernel_code(
                           constitutive_kernels._update_H_pml_real_kernel_code,  # noqa: SLF001
                           "update_H_pml_real", "_update_H_pml_real_kernel_code"),
                       constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE,  # noqa: SLF001
                       "_update_H_pml_real_kernel_code")
    body = _needle(
        body, _THREAD_PREAMBLE,
        "    // THE CELL ARRIVES AS A FLAT INDEX. The bounds guard left with the\n"
        "    // thread index: every caller is either the kernel's own guarded thread\n"
        "    // or an index the certified shift_dn composed, which that helper's own\n"
        "    // branch keeps inside the volume. The DECODE BELOW STAYS -- it is what\n"
        "    // indexes the absorber profile at the recomputed cell rather than at\n"
        "    // the thread's own.\n",
        "the constitutive body's thread preamble")
    if body.count(_INDEX_DECODE) != 1:
        raise AssertionError(
            f"the certified update_H body carries {body.count(_INDEX_DECODE)} copies "
            f"of the index decode, not one; this weld keeps that block verbatim and "
            f"cannot vouch for a changed one")
    for component, target in enumerate(H_TARGETS):
        prefix = f"    constitutive_apply({target}, f_w_{target}, idx, "
        line = _the_line_starting(body, prefix,
                                 f"{target}'s certified constitutive_apply call")
        if not line.endswith(");"):
            raise AssertionError(
                f"{target}'s certified constitutive_apply call does not close on its "
                f"own line ({line!r}); the capture has no suffix to take")
        captured = (f"    h_out[{component}] = constitutive_apply_pure("
                    + line[len("    constitutive_apply("):-len(");")]
                    + f", &w_out[{component}]);")
        body = body.replace(line, captured, 1)

    signature = (
        "\n// The certified update_H_pml_real, evaluated at ONE ARBITRARY CELL and\n"
        "// storing nothing. This is the function the curl half calls for every\n"
        "// foreign tap it needs: a pure function of H, f_w_H and B, none of which\n"
        "// this launch writes, so a foreign evaluation cannot depend on which block\n"
        "// ran first.\n"
        "__device__ __forceinline__ void raw_update_H_cell(\n"
        "    int idx, const WeldArgs& weld, float* h_out, float* w_out\n"
        ") {\n")
    unpack = ["    // The certified body's own names, bound to the pack.",
              "    const int nx = weld.nx; const int ny = weld.ny;",
              "    const int nz = weld.nz;",
              "    (void) nx;   // the removed bounds guard's operand"]
    for _kind, name in WELD_ARGS_FIELDS:
        if name.startswith(("H", "f_w_H", "B", "kps_", "kms_")):
            unpack.append(f"    const float* __restrict__ {name} = weld.{name};")
    return signature + "\n".join(unpack) + "\n" + body + "}\n"


def resolution_source() -> str:
    """``resolve_H`` -- ``update_H``'s result for ONE component at any cell.

    A thin selector over :func:`raw_update_H_cell_source`, so the thread's own cell and
    every foreign recompute go through ONE code path and cannot disagree. ``comp`` is a
    compile-time constant at every call site, so the selection folds and the other two
    components' arithmetic is dead-code eliminated.
    """
    return "\n".join([
        "",
        "// update_H's stepped magnetic field for ONE component at ONE cell. The",
        "// component tag is a compile-time constant at every call site.",
        "__device__ __forceinline__ float resolve_H(int comp, int idx,",
        "                                          const WeldArgs& weld) {",
        "    float h[3];",
        "    float w[3];",
        "    raw_update_H_cell(idx, weld, h, w);",
        "    return h[comp];",
        "}",
        "",
    ])


def shift_dn_recompute_source() -> str:
    """The certified ``shift_dn`` with its two leaf loads RESOLVED.

    THE BRANCHES AND THE INDEX EXPRESSIONS ARE THE CERTIFIED HELPER'S OWN, parsed out
    of its text rather than retyped: everything left of ``g[`` is carried across
    verbatim, the expression inside the brackets is taken as written, and the exact
    ``0.0f`` the metallic and mirror-periodic ghosts read is untouched. A change in how
    the certified helper spells a neighbour's index therefore RAISES here instead of
    silently redirecting a tap.
    """
    _certified()
    prelude = step_curl_kernels._REAL_PML_PRELUDE  # noqa: SLF001
    if prelude.count(_SHIFT_DN_SIGNATURE) != 1:
        raise AssertionError(
            f"the certified curl prelude carries {prelude.count(_SHIFT_DN_SIGNATURE)} "
            f"copies of shift_dn's declaration, not one; the recompute has no anchor")
    body = prelude.split(_SHIFT_DN_SIGNATURE, 1)[1]
    if not body.startswith(" {\n"):
        raise AssertionError(
            "the certified shift_dn no longer opens its body on the declaration's own "
            "line; this weld lifts that body and cannot find it")
    body = body[len(" {\n"):]
    end = body.find("}\n")
    if end < 0:
        raise AssertionError("the certified shift_dn does not close")
    # EVERYTHING PAST THE CLOSING BRACE IS DROPPED, and dropping it is load-bearing
    # rather than tidy: the rest of the certified prelude is `pml_apply`, which
    # :func:`curl_prelude` already emits VERBATIM. Carrying it through here a second
    # time is a duplicate definition NVRTC refuses outright -- which is the good
    # failure -- but the anchor below is what keeps a future prelude edit from
    # smuggling a helper in silently.
    body, remainder = body[:end], body[end:]
    if remainder.lstrip("}\n").lstrip().startswith("float"):
        raise AssertionError(
            "the certified shift_dn's body appears to extend past the first closing "
            "brace; this weld lifts a two-statement helper")
    # THE TWO LEAF LOADS. Found by their `g[` spelling, with the index expression
    # taken from inside the brackets and nothing else on the line touched.
    loads = [line for line in body.splitlines(keepends=True) if "g[" in line]
    if len(loads) != 2:
        raise AssertionError(
            f"the certified shift_dn makes {len(loads)} loads of g, not 2; this weld "
            f"resolves exactly the near-neighbour load and the periodic wrap")
    for line in loads:
        head, _, rest = line.partition("g[")
        expression, closing, tail = rest.partition("]")
        if not closing:
            raise AssertionError(
                f"the certified shift_dn line {line!r} opens a load of g and does not "
                f"close it; the index expression this weld carries across cannot be "
                f"read off it")
        body = body.replace(
            line, f"{head}resolve_H(comp, {expression}, weld){tail}", 1)
    if "g[" in body or " g," in body:
        raise AssertionError(
            "the resolved shift_dn still reads the pointer g; in this helper there is "
            "no such parameter and every magnetic read must be a recompute")
    return ("\n// stepping._shift_down's certified device helper, with the leaf loads\n"
            "// RESOLVED: the value the curl needs at a backward neighbour is not a\n"
            "// stored word but update_H's result there, recomputed from pre-launch\n"
            "// state. The branch, the wrap arithmetic and the exact 0.0f ghost are\n"
            "// the certified helper's own.\n"
            + _SHIFT_DN_SIGNATURE_RECOMPUTE + " {\n" + body + "}\n")


def welded_curl_tail() -> Tuple[str, str]:
    """``(head, tail)`` of the certified ``step_D`` body, every H read redirected.

    ``head`` is the certified preamble, the stride constants and the index decode,
    lifted VERBATIM and kept as the fused kernel's own. ``tail`` is everything below
    it with the six own-cell magnetic loads taken from registers and the six shifted
    ones recomputed.
    """
    _certified()
    body = _split_body(step_curl_kernels._step_D_pml_real_kernel_code,  # noqa: SLF001
                       step_curl_kernels._REAL_PML_PRELUDE,  # noqa: SLF001
                       "_step_D_pml_real_kernel_code")
    if body.count(_THREAD_PREAMBLE) != 1 or body.count(_INDEX_DECODE) != 1:
        raise AssertionError(
            "the certified step_D body no longer carries exactly one thread preamble "
            "and one index decode; this weld keeps both verbatim and splices its "
            "constitutive half directly below the decode")
    cut = body.index(_INDEX_DECODE) + len(_INDEX_DECODE)
    head, tail = body[:cut], body[cut:]

    own = 0
    for component, target in enumerate(H_TARGETS):
        for variable in ("f1", "f2"):
            old = f"        float {variable} = {target}[idx];\n"
            if old not in tail:
                continue
            tail = _needle(tail, old,
                           f"        float {variable} = own_h[{component}];\n",
                           f"{target}'s own-cell load into {variable}")
            own += 1
    if own != OWN_LOAD_EDITS:
        raise AssertionError(
            f"the certified step_D body makes {own} own-cell magnetic loads of the "
            f"shape this weld redirects, not {OWN_LOAD_EDITS}; a load left standing "
            f"would read the PRE-launch H")

    taps = 0
    for line in [line + "\n" for line in tail.splitlines() if "shift_dn(" in line]:
        head_text, _, rest = line.partition("shift_dn(")
        pointer, _, arguments = rest.partition(", ")
        if pointer not in H_TARGETS:
            raise AssertionError(
                f"the certified step_D body reads {pointer!r} through shift_dn; this "
                f"weld resolves exactly the three magnetic components {H_TARGETS}")
        if not arguments.rstrip("\n").endswith(");"):
            raise AssertionError(
                f"the certified shift_dn call {line!r} does not close on its own line; "
                f"the weld appends its argument pack to that closing line")
        component = H_TARGETS.index(pointer)
        carried = arguments.rstrip("\n")[: -len(");")]
        tail = _needle(
            tail, line,
            f"{head_text}shift_dn_recompute({component}, {carried}, weld);\n",
            f"the shifted magnetic load {line.strip()!r}")
        taps += 1
    if taps != HALO_TAPS:
        raise AssertionError(
            f"the certified step_D body makes {taps} shifted magnetic loads, not "
            f"{HALO_TAPS}; this weld redirects every one of them and a missed tap "
            f"would read a word another block is writing")
    for target in H_TARGETS:
        if f"{target}[" in tail:
            raise AssertionError(
                f"the welded curl half still reads {target}; in this signature that "
                f"pointer is the PRE-LAUNCH magnetic field and every read must be the "
                f"register or a recompute")
    if "shift_dn(" in tail:
        raise AssertionError(
            "the welded curl half still calls the certified shift_dn, which loads a "
            "stored H; every shifted tap must go through shift_dn_recompute")
    return head, tail


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

def signature() -> str:
    """The kernel's parameter list.

    THIRTY POINTERS AND SEVEN SCALARS, and unlike the sibling Metal family there is no
    binding ceiling to argue from: NVRTC takes its arguments through a 4 KB parameter
    space and this signature is nowhere near it. The ``kms`` group is bound ONCE for
    both halves because both read the INTEGER Yee sub-lattice -- and that is ASSERTED
    at plan time (:func:`fused_hd_pair_tables`) rather than assumed, because binding
    the half-integer set instead compiles and is a half-cell error in the absorber
    profile rather than a failure.
    """
    return f'''
extern "C" __global__ void {KERNEL_NAME}(
    // THE SCRATCH OUTPUTS. A DIFFERENT ALLOCATION from the pre-launch group below --
    // assert_scratch_is_disjoint checks it by base address before every launch -- and
    // that separation is the whole design: nothing this launch reads is ever a word
    // this launch wrote, so the curl half's foreign taps are a pure function of
    // unwritten memory rather than a race with another block.
    float* __restrict__ Hx_out, float* __restrict__ Hy_out,
    float* __restrict__ Hz_out,
    float* __restrict__ f_w_Hx_out, float* __restrict__ f_w_Hy_out,
    float* __restrict__ f_w_Hz_out,
    // THE PRE-LAUNCH STATE, read at ANY cell by ANY thread and written by none.
    const float* __restrict__ Hx, const float* __restrict__ Hy,
    const float* __restrict__ Hz,
    const float* __restrict__ f_w_Hx, const float* __restrict__ f_w_Hy,
    const float* __restrict__ f_w_Hz,
    const float* __restrict__ Bx, const float* __restrict__ By,
    const float* __restrict__ Bz,
    // D and its split-field auxiliary: written IN PLACE, which is safe by
    // construction -- pml_apply reads and writes f[idx] and fu[idx] at the THREAD'S
    // OWN CELL only, so no thread reads a displacement another thread wrote.
    float* __restrict__ Dx, float* __restrict__ Dy, float* __restrict__ Dz,
    float* __restrict__ fu_Dx, float* __restrict__ fu_Dy, float* __restrict__ fu_Dz,
    int nx, int ny, int nz, float dtdx,
    // The D curl's INTEGER split-field coefficients, certified names kept.
    const float* __restrict__ kms_x, const float* __restrict__ sinv_x,
    const float* __restrict__ kms_y, const float* __restrict__ sinv_y,
    const float* __restrict__ kms_z, const float* __restrict__ sinv_z,
    // update_H's kps. Its kms IS the curl's kms above and is bound once: step_D reads
    // the INTEGER sub-lattice (real_pml_curl_tables(pml, False)) and update_H reads
    // the INTEGER one too (constitutive_sub_lattice("H") is False), so the two groups
    // are the same three volumes. There is no kms_half_* here and there must not be.
    const float* __restrict__ kps_x, const float* __restrict__ kps_y,
    const float* __restrict__ kps_z,
    int bc_x, int bc_y, int bc_z
) {{
'''


def kernel_source() -> str:
    """The whole device source. ONE string for this family, every boundary at runtime."""
    head, tail = welded_curl_tail()
    weld = "\n".join([
        "",
        weld_args_construction().rstrip("\n"),
        "",
        "    // --- THE WELD: update_H, computed into registers and stored to SCRATCH",
        "    // Nothing written here is read by this launch. The curl half below takes",
        "    // its OWN cell's magnetic field from these registers and recomputes",
        "    // every foreign tap from PRE-LAUNCH state through the same",
        "    // raw_update_H_cell, so no thread observes another thread's store; the",
        "    // launcher rotates H/f_w_H against the scratch afterwards.",
        "    float own_h[3];",
        "    float own_w[3];",
        "    raw_update_H_cell(idx, weld, own_h, own_w);",
        "    Hx_out[idx] = own_h[0]; Hy_out[idx] = own_h[1]; Hz_out[idx] = own_h[2];",
        "    f_w_Hx_out[idx] = own_w[0]; f_w_Hy_out[idx] = own_w[1];",
        "    f_w_Hz_out[idx] = own_w[2];",
        "",
        "    // --- step_D (stepping.step_D / _apply_pml_update), the certified body",
        "    // from its own stride constants down, with the twelve magnetic reads",
        "    // redirected and NOTHING else touched.",
        "",
    ])
    return (curl_prelude() + constitutive_prelude() + weld_args_struct()
            + raw_update_H_cell_source() + resolution_source()
            + shift_dn_recompute_source() + signature() + head + weld + tail + "}\n")


def device_sources() -> Dict[str, str]:
    """``{kernel name: source}`` -- the shape every family on this track publishes."""
    return {KERNEL_NAME: kernel_source()}


def source_digest() -> str:
    """One sha256 over the family's single device string."""
    import hashlib  # noqa: PLC0415 - stdlib, imported at the one call site

    return hashlib.sha256(kernel_source().encode("utf-8")).hexdigest()


def _get_kernel(source: Optional[str] = None):
    """Compile on first use, memoized on (name, options, policy, source).

    THE SOURCE IS IN THE KEY, so a mutation harness that handed in a rewritten string
    cannot be served the shipped binary and a late policy install cannot be served an
    earlier one. ``cupy`` is imported HERE, never at scope.
    """
    import cupy as cp  # noqa: PLC0415 - device-only, in a laptop-importable module

    code = kernel_source() if source is None else source
    key = _kernel_cache_key(KERNEL_NAME, False, _COMPILE_OPTIONS, code)
    return _get_or_compile(
        key, lambda: cp.RawKernel(code, KERNEL_NAME, options=_COMPILE_OPTIONS))


def _clear_kernel_cache() -> int:
    return _clear_cache()


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def covers_fused_hd_pair(fields: Any, pml: Any, grid: Any,
                         sources: Any = None) -> Tuple[bool, str]:
    """May ONE launch span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal, this
    directory's convention.

    A CONJUNCTION, AND NOTHING IS WEAKENED. A configuration either half's own certified
    predicate refuses is refused here with that half's reason, prefixed so a reader can
    tell which side said it.

    THE ORDER IS THE DRIVER'S. The constitutive half is asked FIRST because it runs
    first (driver.py:3311), so the first refusal a reader sees names the half the
    driver would have reached first -- and on this seam that matters, since the null
    ``update_H`` (an inactive absorber returns before its first statement,
    stepping.py:944) is the single largest non-fusion reason on the board and belongs
    to the constitutive side.

    WHAT THIS ADDS to the two halves: the seam's own withdraw clause, the shared
    boundary-triple check, and the rotation's requirement that the six magnetic volumes
    be allocated attributes this weld may rebind.
    """
    # THE CONSTITUTIVE HALF, admitted by EITHER its ordinary predicate OR the
    # nonlinear-run spine arm's. The two are DISJOINT on chi2/chi3 -- the ordinary one
    # refuses when either map is non-empty (coverage.py's chi clause) and the
    # nonlinear one refuses when both are empty -- so exactly one can answer. A double
    # refusal reports both, prefixed, the way `fused_magnetic_pair`'s nonlinear H
    # widening reports it.
    covered, reason = covers_real_pml_constitutive(fields, pml, grid, "H")
    if not covered:
        widened, widened_reason = covers_real_pml_nonlinear_constitutive(
            fields, pml, grid, "H")
        if not widened:
            return False, (f"constitutive half: {reason}; constitutive half, "
                           f"nonlinear-run widening: {widened_reason}")
    covered, reason = covers_real_pml_curl(fields, pml, grid, "step_D")
    if not covered:
        return False, f"curl half: {reason}"

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all and a source of either polarity is not
    # this seam's business. What IS between them is the electric integrated-source
    # withdraw (driver.py:3313-3314), and `withdraw_hoist` owns it. IGNORANCE IS NEVER
    # AN EMPTY SET: `Fields` does not hold the source list, so a predicate that
    # inferred "no withdraw stands" from not being told would be the over-covering
    # refusal this clause exists to prevent.
    seam_reasons = _withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3313-3314); this product declares "
            f"HOISTS_THE_WITHDRAW = False because it declares INSTALLABLE = False, so "
            f"fused_pairs._install_fused_pair's withdraw-hoist branch is unreachable "
            f"for it and nothing would perform the withdraw before the launch"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES)
    if seam_reasons:
        return False, seam_reasons[0]

    # ONE BOUNDARY TRIPLE FOR BOTH HALVES, CHECKED. The curl resolves a fold's
    # termination into BC_MIRROR_PERIODIC or BC_METALLIC and the constitutive half
    # reads no boundary at all, so the triple this launch binds is the CURL's -- but
    # the resolution must succeed, and a grid it refuses is a grid this launch must
    # not bind. `real_curl_boundary_codes` fails closed and returns the refusal rather
    # than raising.
    codes, refusal = _coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        return False, (f"the curl's own boundary resolution refuses this grid: "
                       f"{refusal}")
    if len(tuple(codes)) != 3:
        return False, (f"the curl's boundary resolution returned {codes!r}, which is "
                       f"not the per-axis triple this launch binds")

    # THE SUB-LATTICE, ASSERTED RATHER THAN ASSUMED. This kernel binds ONE kms group
    # for both halves, which is only correct while step_D's curl and update_H read the
    # SAME Yee sub-lattice. Both read the integer positions today
    # (step_curl_kernels._step_D_fused_pml_real passes half_integer=False;
    # constitutive_sub_lattice("H") is False). If either moved, sharing the group would
    # bind one half's coefficients to the other half's lattice -- a converged, smooth,
    # half-cell-wrong absorber profile rather than a failure.
    if constitutive_sub_lattice("H"):
        return False, ("update_H now reads the HALF-INTEGER PML sub-lattice while "
                       "step_D's curl reads the integer one; this weld binds ONE kms "
                       "group for both halves and may only do so while they agree")

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT. The launcher swaps the ENGINE's
    # references for the six volumes in SCRATCH_VOLUMES after every launch, so those
    # attributes must exist and be allocated.
    for name in SCRATCH_VOLUMES:
        if getattr(fields, name, None) is None:
            return False, (f"{name} is not allocated; this weld rotates it against a "
                           f"launch-local scratch twin after every launch")
    return True, "covered"


# ---------------------------------------------------------------------------
# The launch
# ---------------------------------------------------------------------------

def fused_hd_pair_tables(pml: Any) -> Dict[str, Any]:
    """The ONE coefficient group both halves read, with the sharing checked.

    Returns the curl's own ``kms``/``sinv`` views plus the constitutive's ``kps``, and
    ASSERTS that the constitutive's ``kms`` is the same object as the curl's -- which
    is what the shared binding in :func:`signature` rests on. A drifted sub-lattice
    is a half-cell error in the absorber profile, so it is refused here rather than
    bound.
    """
    _certified()
    curl = step_curl_kernels.real_pml_curl_tables(pml, False)
    constitutive = constitutive_kernels.real_constitutive_tables(
        pml, constitutive_sub_lattice("H"))
    for axis in ("x", "y", "z"):
        left, right = curl[f"kms_{axis}"], constitutive[f"kms_{axis}"]
        if left.data.ptr != right.data.ptr:
            raise ValueError(
                f"step_D's kms_{axis} and update_H's kms_{axis} are different "
                f"allocations; this weld binds ONE kms group for both halves and may "
                f"only do so while both read the INTEGER Yee sub-lattice "
                f"(step_curl_kernels.real_pml_curl_tables(pml, False) and "
                f"constitutive_sub_lattice('H') is False)")
    return {"kms": {axis: curl[f"kms_{axis}"] for axis in "xyz"},
            "sinv": {axis: curl[f"sinv_{axis}"] for axis in "xyz"},
            "kps": {axis: constitutive[f"kps_{axis}"] for axis in "xyz"}}


def fused_hd_pair_codes(grid: Any) -> Tuple[int, ...]:
    """The boundary triple this launch binds, from the CURL's own resolution.

    RAISES where the curl refuses the grid: callers gate on
    :func:`covers_fused_hd_pair` first, and a raise here means they did not.
    """
    codes, refusal = _coverage.real_curl_boundary_codes(grid)
    if refusal is not None:
        raise ValueError(
            f"this grid has no boundary-code triple for the real-field PML curl: "
            f"{refusal}. Ask covers_fused_hd_pair first; it refuses this "
            f"configuration rather than raising")
    return tuple(int(code) for code in codes)


def fused_hd_pair_scratch(fields: Any) -> Dict[str, Any]:
    """The six launch-local volumes, allocated once per frozen configuration.

    NOT ``StepScratch``. That pool is documented for values consumed within the call
    that produced them; these are the opposite -- the rotation hands them to the driver
    as the live ``H``/``f_w_H`` and keeps the retired pair for the next launch.

    UNINITIALIZED IS CORRECT. Every cell of all six is written by every launch, so
    there is no cell whose prior contents a reader could observe.
    """
    import cupy as cp  # noqa: PLC0415 - device-only

    volumes: Dict[str, Any] = {}
    for name in SCRATCH_VOLUMES:
        source = getattr(fields, name, None)
        if source is None:
            raise ValueError(
                f"fields.{name} is not allocated; this weld rotates a scratch shaped "
                f"like it and has nothing to shape one from")
        volumes[name] = cp.empty_like(source)
    return volumes


def assert_scratch_is_disjoint(fields: Any, scratch: Dict[str, Any],
                               tables: Dict[str, Any]) -> int:
    """THE CHECK THE WHOLE DESIGN RESTS ON, plus the ordinary restrict promise.

    Two things, and the first is not a convention:

    1. **no scratch volume is any bound input.** If ``Hx_out`` were ``Hx`` the launch
       would be the IN-PLACE weld the board refused -- every foreign recompute would
       read words other blocks had already overwritten, and the answer would be a
       schedule. This is the gate's ``scratch_aliased_to_storage`` mutation, and it is
       refused here so it cannot happen by accident.
    2. **no two ``__restrict__`` arguments are one allocation.** Two restrict pointers
       to one object is UB whatever the route to it, and NVRTC reorders across it
       without a diagnostic.

    THE SHARED ``kms`` GROUP IS THE ONE DELIBERATE COINCIDENCE and is bound ONCE, so it
    is visited once and is not a collision. Returns the number of distinct allocations
    inspected, so a caller can assert that something was actually examined.
    """
    bound: Dict[int, str] = {}
    collisions: List[str] = []

    def visit(label: str, array: Any) -> None:
        pointer = int(array.data.ptr)
        if pointer in bound:
            collisions.append(f"{label} and {bound[pointer]} are the same allocation")
            return
        bound[pointer] = label

    for name in SCRATCH_VOLUMES:
        visit(f"{name}_out", scratch[name])
    for name in ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                 "Bx", "By", "Bz", "Dx", "Dy", "Dz",
                 "fu_Dx", "fu_Dy", "fu_Dz"):
        visit(name, getattr(fields, name))
    for group in ("kms", "sinv", "kps"):
        for axis in "xyz":
            visit(f"{group}_{axis}", tables[group][axis])
    if collisions:
        raise ValueError(
            "this weld binds every field, scratch and table argument __restrict__, "
            "and these arguments alias -- which is undefined behaviour NVRTC "
            "miscompiles silently rather than diagnosing, and which for the scratch "
            "group would make the launch the in-place weld the board refused: "
            + "; ".join(collisions))
    return len(bound)


def launch_fused_hd_pair(fields: Any, scratch: Dict[str, Any], tables: Dict[str, Any],
                         boundary_codes: Sequence[int], dtdx: float,
                         kernel: Optional[Any] = None,
                         threads: int = _FUSED_THREADS) -> Dict[str, Any]:
    """Both of :data:`REPLACES` in ONE launch. THE CALLER ROTATES AFTERWARDS.

    ``kernel`` IS THE GATE'S DOOR, keyword-optional and named for what it is: a gate
    compiles a deliberately broken copy of the shipped source and hands it here. A
    launcher that could not be handed its own kernel could not arm a single mutation,
    and every mutation leg would silently launch the shipped one and report the defect
    as uncaught.

    ``threads`` is the gate's schedule door. The design's whole claim is that the
    answer does NOT depend on which block ran first, and a block size that never moves
    cannot expose the opposite.

    Returns the launch geometry rather than ``None`` so a gate can assert that
    something was actually launched.
    """
    nx, ny, nz = (int(n) for n in fields.Dx.shape)
    blocks = (nx * ny * nz + threads - 1) // threads
    arguments: List[Any] = [scratch[name] for name in SCRATCH_VOLUMES]
    arguments += [getattr(fields, name) for name in
                  ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz",
                   "Bx", "By", "Bz",
                   "Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")]
    arguments += [np.int32(nx), np.int32(ny), np.int32(nz), np.float32(dtdx)]
    for axis in ("x", "y", "z"):
        arguments += [tables["kms"][axis], tables["sinv"][axis]]
    arguments += [tables["kps"][axis] for axis in ("x", "y", "z")]
    arguments += [np.int32(int(code)) for code in boundary_codes]
    (kernel or _get_kernel())((blocks,), (threads,), tuple(arguments))
    return {"launched": True, "blocks": blocks, "threads": threads,
            "elements": nx * ny * nz, "replaces": REPLACES,
            "boundary_codes": tuple(int(code) for code in boundary_codes)}


def rotate_into_fields(fields: Any, scratch: Dict[str, Any]) -> Dict[str, Any]:
    """THE ROTATION. The scratch becomes the live H/f_w_H; the retired pair becomes scratch.

    THE CERTIFIED CHOREOGRAPHY, one seam earlier than
    :func:`.offdiag_fused_electric_pair.rotate_into_fields` performs it on ``D``/``fu_D``
    and one field type over from ``ade_kernels.update_P_fused_pml_real``'s
    ``P``/``P_prev`` rotation.

    THE CONSUMER AUDIT THIS RESTS ON. Every consumer of the magnetic field in this
    engine resolves it BY NAME at use time -- ``getattr(self.fields, name)`` in
    ``driver.synchronize_magnetic_fields`` (:4339, :4388) and in every energy and flux
    accessor, ``fields.get_component(...)`` in every DFT and flux monitor -- so a
    rebinding between steps is invisible to all of them. That is not asserted from a
    grep: ``test_fused_hd_pair.py`` drives a real ``FdtdDriver`` with sources and
    monitors for several steps with an equivalent rotation shim spliced in at exactly
    this point and requires every stored volume AND every monitor accumulation to be
    byte-identical to the unrotated run.

    Returns the mapping the caller should keep as the NEXT launch's scratch.
    """
    retired: Dict[str, Any] = {}
    for name in SCRATCH_VOLUMES:
        retired[name] = getattr(fields, name)
        setattr(fields, name, scratch[name])
    return retired


def run_fused_hd_pair(fields: Any, grid: Any, pml: Any, dtdx: float, *,
                      scratch: Optional[Dict[str, Any]] = None,
                      sources: Any = None,
                      tables: Optional[Dict[str, Any]] = None,
                      kernel: Optional[Any] = None,
                      threads: int = _FUSED_THREADS,
                      rotate: bool = True) -> Dict[str, Any]:
    """Gate on the predicate, resolve everything from the grid, launch once, rotate.

    THE PREDICATE IS ASKED FIRST AND A REFUSAL IS RETURNED, NOT RAISED, because the
    caller's correct response to a configuration this product does not carry is the
    array path -- never an exception into a stepper that would otherwise have stepped
    correctly.

    ``scratch``, ``tables``, ``kernel``, ``threads`` and ``rotate`` are the gate's
    doors, keyword-only. ``rotate=False`` is how the ``rotation_skipped`` mutation is
    armed: a launcher that always rotated could not measure what the rotation is worth.
    """
    covered, reason = covers_fused_hd_pair(fields, pml, grid, sources)
    if not covered:
        return {"launched": False, "reason": reason}
    if tables is None:
        tables = fused_hd_pair_tables(pml)
    if scratch is None:
        scratch = fused_hd_pair_scratch(fields)
    assert_scratch_is_disjoint(fields, scratch, tables)
    record = launch_fused_hd_pair(fields, scratch, tables, fused_hd_pair_codes(grid),
                                  dtdx, kernel, threads)
    record["rotated"] = bool(rotate)
    record["scratch"] = rotate_into_fields(fields, scratch) if rotate else scratch
    return record
