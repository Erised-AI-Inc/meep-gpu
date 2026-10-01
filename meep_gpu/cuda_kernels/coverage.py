"""Which configurations the hand-CUDA kernels may step — with no CuPy in sight.

NOTHING HERE IMPORTS CUPY, NUMPY OR THE KERNEL MODULE, AND THAT IS THE POINT.
``step_curl_kernels`` imports ``cupy`` at module scope because it holds
``cp.RawKernel`` objects, so for as long as the predicate lived beside them it
could not be imported — let alone exercised — on a machine without a GPU. The
whole 16-test predicate file collapsed to a single sanctioned skip on any laptop,
which meant the one part of this track that is pure decision logic, and the part
whose failure mode is a *silent wrong answer* rather than a crash, was pinned
only on the one host that can launch a kernel.

The sibling track had already settled the shape: ``triton_kernels/coverage.py``
imports ``__future__`` and ``typing`` and nothing else, which is why a 186-row
predicate battery runs there at the merge bar. This module is the same move for
the hand-CUDA track.

ENUMERATED POSITIVELY, never inferred from the absence of a known blocker: a
feature that lands after this is written is refused because it was never
admitted, not admitted because nobody remembered to refuse it. Every refusal
below is a SILENT WRONG ANSWER if it leaks, not a crash — which is why
:func:`covers_real_pml_curl` returns the reason as well as the verdict, and why
its tests mutate it — widen an axis, drop the dtype check, widen the
conductivity loop back over both sub-steps' targets, drop ``stores_E``, drop the
int32 bound — and require every mutation caught. A WIDENING is the same argument
run backwards, and gets the same treatment: the two clauses narrowed on
2026-08-15 carry their pre-widening verdicts as data, asserted ABSENT, because a
fail-closed predicate quietly re-acquiring a refusal costs coverage and looks
exactly like a legitimate "not covered" in a plan log.

WHAT THESE PREDICATES COVER AND WHAT THEY DO NOT. There are two.

:func:`covers_real_pml_curl` answers for exactly two of the fourteen kernels in
``step_curl_kernels`` — ``step_B_pml_real`` and ``step_D_pml_real``,
the certified real-field PML curl pair. The other twelve have no predicate at
all, which is one of the reasons they are marked dead in that file rather than
wired: there is no fail-closed refusal written anywhere for the configurations
they would silently mis-serve.

:func:`covers_real_pml_constitutive` answers for the two kernels in
``constitutive_kernels`` — ``update_H_pml_real`` and
``update_E_pml_real``. Those are authored and NOT yet gated, so the
predicate ships ahead of the certification rather than behind it; that ordering
is deliberate, because the predicate is what decides which configurations the
gate has to cover.

COVERAGE IS A SET PER SUB-STEP, NOT PER RUN — and both functions take the
sub-step by name because of it. A feature that changes one sub-step's semantics
disqualifies THAT sub-step and leaves the others alone: the array path and the
kernels compute the same bits, and the driver separates the sub-steps with
array-path boundary work anyway. Conductivity is the case that forces the
distinction twice over. It is read in ``stepping._apply_curl`` (stepping.py:508)
and NOWHERE else, so it disqualifies a curl and leaves both constitutive
sub-steps untouched — and it is read there PER TERM, so a sigma on Dx disqualifies
``step_D`` and leaves ``step_B`` alone as well.

Both distinctions were once missing, and the second one was measured rather than
argued: until 2026-08-15 ``covers_real_pml_curl`` took no sub-step name, so it
answered for ``step_B_pml_real`` and ``step_D_pml_real`` with one
verdict and refused each for the other's blockers. Against the sibling track's
per-sub-step predicate on the same 186 corpus rows that was **seven sub-steps
refused here and admitted there, and zero the other way** — six of them a
dispersion clause that belongs to ``update_E``, one a D conductivity charged to
``step_B``. Writing one predicate for the whole run would have had to refuse the
conductivity for all four, which is a refusal with no line behind it.

THESE PREDICATES ARE CONSULTED BY DISPATCH, through the hand-written CUDA table's
registry (``registry.py`` names them per slot); a sub-step they refuse is not
served by that table's kernel, and the refusal reason is reported.
"""

from __future__ import annotations

from typing import Any, Tuple

# The boundary kinds this slice serves, as the kernels' integer codes. The
# spellings are ``stepping.PERIODIC`` / ``stepping.METALLIC``; MIRROR and
# CYL_AXIS have no code because they have no kernel.
#
# THIS DICT IS SHARED WITH ``complex_pml_kernels.py``, which refuses any kind
# outside it, and the real-field curl pair's third code is deliberately NOT in
# here: adding it would silently hand a folded axis to a complex kernel that has
# no branch for one. :data:`REAL_CURL_BC_CODES` below is the curl pair's own map,
# and ``test_step_curl_pml_real.py`` pins the two apart.
BC_CODES = {"periodic": 0, "metallic": 1}

#: The codes ``step_B_pml_real`` / ``step_D_pml_real`` take, which is
#: :data:`BC_CODES` plus one. ``"mirror_periodic"`` is NOT a
#: ``stepping._boundary_kinds`` spelling — that function answers ``"mirror"`` at
#: both terminations of a fold (stepping.py:2193) — it is this pair's name for the
#: half of that answer whose stored array carries the slot past MEEP's owned
#: window (``stepping._stored_past_owned``, stepping.py:1503-1518), and which
#: therefore needs the Yee-shift-1 top-plane mask. The other half, a folded
#: METALLIC axis, is handed ``BC_METALLIC`` and has no code of its own because its
#: ghost rule and its masks ARE metallic's — measured 48/48 bit-identical rather
#: than argued (:data:`CURL_FOLD_ADMISSION`).
MIRROR_PERIODIC = "mirror_periodic"
REAL_CURL_BC_CODES = {"periodic": 0, "metallic": 1, MIRROR_PERIODIC: 2}

#: The two curl sub-steps, keyed as :func:`covers_real_pml_curl`'s ``sub_step``,
#: mapping to the targets THAT sub-step writes — the components whose conductivity
#: would route it to the three-history conductive-PML recurrence instead of this
#: pair. ``stepping.step_B`` loops over ``B_CURL_TERMS`` and ``step_D`` over
#: ``D_CURL_TERMS`` (stepping.py:365, :452), and ``_apply_curl`` reads
#: ``fields.condfac_for(term.target)`` — PER TERM, not per run (stepping.py:508).
#: So a sigma on Dx reaches ``step_D`` and leaves ``step_B`` an ordinary curl.
#:
#: Written out here rather than derived from ``stepping.B_CURL_TERMS`` because
#: this module imports nothing — the same discipline that keeps it
#: laptop-evaluable. ``test_step_curl_pml_real.py`` pins the two against each
#: other so the transcription cannot drift.
CURL_SUB_STEPS = {
    "step_B": ("Bx", "By", "Bz"),
    "step_D": ("Dx", "Dy", "Dz"),
}

#: The two constitutive sub-steps, keyed as :func:`covers_real_pml_constitutive`'s
#: ``side``. ``targets`` are written, ``aux`` is this sub-step's ``f_w``, ``sources``
#: is what the constitutive product reads, and ``half_integer`` selects the Yee
#: sub-lattice the ``kps``/``kms`` pair comes from — INTEGER for H
#: (``stepping.update_H``, stepping.py:948) and HALF-INTEGER for E
#: (``stepping.update_E``, :1015). Swapped, it is a half-cell error in the absorber
#: profile: converged, smooth and wrong.
#:
#: Written out here rather than derived from ``stepping.H_CONSTITUTIVE_TERMS``
#: because this module imports nothing — the same discipline that keeps it
#: laptop-evaluable. ``test_constitutive_pml_real.py`` pins the two against each
#: other so the transcription cannot drift.
CONSTITUTIVE_SIDES = {
    "H": {
        "targets": ("Hx", "Hy", "Hz"),
        "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
        "sources": ("Bx", "By", "Bz"),
        "half_integer": False,
    },
    "E": {
        "targets": ("Ex", "Ey", "Ez"),
        "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
        "sources": ("Dx", "Dy", "Dz"),
        "half_integer": True,
    },
}

#: The three E components whose inverse permittivity ``update_E`` multiplies by.
_ELECTRIC_COMPONENTS = ("Ex", "Ey", "Ez")

# ---------------------------------------------------------------------------
# THE OFF-DIAGONAL (TENSOR) ELECTRIC CONSTITUTIVE SUB-STEP
#
# Three tables, transcribed from ``stepping``/``fields`` and pinned against them
# by ``test_offdiag_constitutive_pml_real.py`` — written out here rather than
# imported for the reason the two above are: this module imports nothing, which
# is what lets a 186-row predicate battery run at the merge bar.
# ---------------------------------------------------------------------------

#: The coupling partners of each component, in MEEP's ``cycle_direction`` order
#: X -> Y -> Z (vec.hpp:586; ``stepping._offdiagonal_terms``, stepping.py:1235-1237):
#: own axis + 1 first, own axis + 2 second. Ez therefore takes the Ex-partner
#: volume and then the Ey-partner volume, and the ORDER BINDS COEFFICIENTS TO
#: PARTNERS — mispairing them is a silent half-tensor.
OFFDIAG_TRANSVERSE_PARTNERS = ((1, 2), (2, 0), (0, 1))

#: The six coefficient slots, in plan/kernel argument order: for each row
#: component (``E_CONSTITUTIVE_TERMS`` order, stepping.py:228) the offset-1
#: partner then the offset-2 partner. The keys are
#: ``fields._chi1inv_offdiagonal``'s own (row, partner) pairs (fields.py:547),
#: read back through ``Fields.chi1inv_offdiagonal_for`` (fields.py:1317-1319).
OFFDIAG_ROW_SLOTS = (
    ("Ex", "Ey"), ("Ex", "Ez"),
    ("Ey", "Ez"), ("Ey", "Ex"),
    ("Ez", "Ex"), ("Ez", "Ey"),
)

#: Per component, the axes on which its Yee shift is 0 — the axes whose metallic
#: wall plane ``stepping._mask_metallic_wall_coupling`` zeroes (stepping.py:
#: 1279-1283, looping axes ascending; ``fields.IYEE_SHIFTS`` at fields.py:214-219
#: gives Ex (1,0,0), Ey (0,1,0), Ez (0,0,1)).
OFFDIAG_WALL_MASK_AXES = ((1, 2), (0, 2), (0, 1))


def offdiag_row_volumes(fields):
    """The six coefficient volumes in :data:`OFFDIAG_ROW_SLOTS` order, None where dead.

    The SINGLE place the slot binding is derived from the engine's own rows, so
    the predicate, the launcher and the emitter cannot disagree about which
    coefficient volume pairs with which partner term. Fail-closed on the read
    itself: a ``fields`` that cannot answer yields six ``None``s, which the
    predicate then refuses by name rather than crashing a planner.
    """
    reader = getattr(fields, "chi1inv_offdiagonal_for", None)
    if not callable(reader):
        return (None,) * len(OFFDIAG_ROW_SLOTS)
    out = []
    for row, partner in OFFDIAG_ROW_SLOTS:
        try:
            entries = reader(row) or {}
            out.append(entries.get(partner))
        except Exception:  # noqa: BLE001 - a raise is not a refusal unless caught
            out.append(None)
    return tuple(out)


def offdiag_row_mask(fields):
    """The six liveness flags the emitter specializes the device source on."""
    return tuple(int(volume is not None) for volume in offdiag_row_volumes(fields))


def offdiag_fold_roles(row_mask, mirrored):
    """Which of ``update_E``'s two stencil legs a MIRROR FOLD actually reaches.

    THE FOLD QUESTION ON THIS SUB-STEP IS DECIDED BY THE ROW MASK, not by the
    fold's declared termination — the opposite of the curl pair's split, and
    measured rather than argued (see
    :data:`FOLDED_OFFDIAG_ROW_MASK_ADMISSION`). ``stepping._offdiagonal_terms``
    (stepping.py:1235-1249) builds each term from exactly two shifts:

    * the PARTNER-axis ``_shift_down`` (:1214-1216), which on a folded axis
      writes the mirror ghost ``parity * g[MIRROR_SOURCE_INDEX]`` into stored
      cell 0 (:1826-1828). The emitted ``coord_dn`` has a ``PERIODIC`` branch
      (wrap to ``n-1``) and a ``METALLIC`` branch (the ``-1`` zero ghost), and the
      mirror ghost is NEITHER — so a folded axis in this role always moves bytes,
      whichever code the launcher hands it;
    * the OWN-axis ``_shift_up`` (:1217-1219), called with no ``component`` and no
      ``reflect_row``, so the MIRROR reflect branch (:1779-1786) is not taken and
      the call falls through to ``shifted[_face(axis, -1)] = 0`` (:1787-1789).
      That exact zero is what ``coord_up``'s METALLIC arm writes and is NOT what
      its PERIODIC arm writes, so a folded axis in this role moves bytes under a
      PERIODIC substitution ONLY.

    A component's own axis is never its own partner, so the two roles are disjoint
    per slot and the answer is a pair, not a scalar: ``changes_a_byte`` is keyed by
    the boundary code the launcher would hand the folded axis. ``mirror`` has no
    code today (:data:`BC_CODES` carries ``periodic`` and ``metallic`` only), which
    is why the choice is named here instead of assumed.

    Pure, stdlib-only and total: it reads two sequences and no engine object, so a
    predicate, a gate and a test can all ask it the same question.
    """
    folded = tuple(axis for axis in range(3) if bool(mirrored[axis]))
    live_partner, live_own = set(), set()
    for slot, (row, partner) in enumerate(OFFDIAG_ROW_SLOTS):
        if not row_mask[slot]:
            continue
        live_partner.add("xyz".index(partner[1]))
        live_own.add("xyz".index(row[1]))
    partner_hit = tuple(sorted(set(folded) & live_partner))
    own_hit = tuple(sorted(set(folded) & live_own))
    return {
        "folded_axes": folded,
        "live_partner_axes": tuple(sorted(live_partner)),
        "live_own_axes": tuple(sorted(live_own)),
        "folded_partner_axes": partner_hit,
        "folded_own_axes": own_hit,
        "fold_is_a_live_partner_axis": bool(partner_hit),
        "fold_is_a_live_own_axis": bool(own_hit),
        "changes_a_byte": {
            "metallic": bool(partner_hit),
            "periodic": bool(partner_hit or own_hit),
        },
    }


#: THE FOLDED OFF-DIAGONAL ARM — MEASURED EXACTLY, AND DELIBERATELY NOT INSTALLED.
#:
#: ``covers_real_pml_offdiag_constitutive`` refuses every mirror fold, twice: at
#: ``has_symmetry`` and again per axis. On 2026-08-20 that refusal was measured
#: rather than inherited, and it turned out to be REAL BUT TOO BROAD — there is a
#: sub-arm on which the SHIPPED kernel, unchanged, is byte-identical to
#: ``stepping.update_E`` on a folded grid. This record is that measurement, kept
#: so the clause is not re-litigated from the reading alone, and so the arithmetic
#: behind "do not widen" is auditable instead of remembered.
#:
#: THE ARM, stated exactly, is :func:`offdiag_fold_roles`'s ``changes_a_byte``
#: being False for the code the launcher would hand the folded axis.
#:
#: WHY IT IS NOT INSTALLED — two independent reasons, both measured:
#:
#: 1. **It is worth ZERO corpus slots.** See ``corpus`` below: on the 186-row /
#:    759-slot all-families census the widened predicate admits exactly the same
#:    16 ``update_E`` slots the shipped one does. Every corpus row that a fold
#:    could newly reach drives row mask ``(1, 0, 0, 1, 0, 0)`` — partner axes X
#:    and Y — and every corpus fold is on X and/or Y, which is structural rather
#:    than a sampling accident: a 2-D run with an in-plane mirror and an
#:    anisotropic epsilon installs the in-plane XY tensor block, whose partner
#:    axes are the axes an in-plane fold can sit on.
#: 2. **The launcher cannot serve what the clause would admit.**
#:    ``offdiag_constitutive_kernels.offdiag_boundary_codes`` maps through
#:    :data:`BC_CODES`, which has no ``mirror`` entry, and deliberately lets the
#:    KeyError escape rather than defaulting. A predicate that admitted a fold
#:    would promise a planner a dispatch that raises. Worse, the obvious "fix" —
#:    reading a fold as PERIODIC, which is how a fold is usually described — is
#:    measured WRONG on 32 of the 480 guarded cases per policy: those are the
#:    cases where a folded axis is some live slot's OWN axis, exact under
#:    ``metallic`` and divergent under ``periodic``. The arm therefore depends on
#:    a launcher decision this module does not own.
#:
#: So the clause stays as written, and what changed is that it is now a refusal
#: with a two-sided measurement and a named boundary behind it.
FOLDED_OFFDIAG_ROW_MASK_ADMISSION = {
    "kernel": "update_E_pml_real_offdiag",
    "artifact": "parity/meep_gpu/results/cuda_folded_offdiag_2026-08-21_rename",
    "gate": "parity/meep_gpu/gate_cuda_folded_offdiag.py",
    #: The gate that measures THIS record rather than the kernel: it replays the
    #: rule against every guarded device case (480 at one launch and 44 at 60, per
    #: policy), re-derives the same boundary locally against ``stepping.update_E``
    #: with no artifact in the loop, recomputes the corpus arithmetic, and runs a
    #: mutation battery on the rule. It makes NO device measurement and needs no
    #: GPU.
    "boundary_gate": "parity/meep_gpu/gate_cuda_folded_offdiag_rowmask.py",
    #: Its runs, newest last. A PREFIX rather than one directory, because the gate
    #: refuses to overwrite a manifest and each re-run takes a fresh suffix — and
    #: because a record naming one run would have to be edited to name the run
    #: that measured the edit.
    "boundary_gate_artifact_prefix": ("parity/meep_gpu/results/"
                                      "cuda_folded_offdiag_rowmask_2026-08-22_rename"),
    "host": "the GPU host, NVIDIA RTX A6000, GPU index 7",
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    "oracle": "stepping.update_E on real Grid/Fields/PML, rows installed through "
              "Fields.set_epsilon_volumes; compared as uint32 words",
    "installed": False,
    #: Per policy, over the GUARDED (``--fmad=false``) leg of the sweep. The
    #: unguarded leg is 0/480 identical at the inexact courant on every arm, so
    #: the guard is a precondition of the arm and not a tuning choice.
    "guarded_cases_per_policy": 480,
    "unfolded_controls": 32,
    "folded_bit_identical": 96,
    "folded_divergent": 352,
    "multi_step_budget": 60,
    "multi_step_folded_bit_identical": 13,
    "differing_words_keep": 132514,
    "differing_words_flush": 126261,
    "differing_words_off_the_ghost_delta_planes": 0,
    #: EVERY combination the sweep drove, as
    #: ``(folded axes, row mask, code handed the folded axis, ran, bit-identical)``
    #: at one launch, guarded, per policy. The two policies produced this table
    #: IDENTICALLY, which is why one copy carries both. No combination is mixed:
    #: each is all-identical or all-divergent across both courants (0.5 and 0.35),
    #: both value classes (uniform and subnormal_band) and both plane parities, so
    #: collapsing those axes here is measured rather than assumed.
    "combinations": (
    ((),            (0, 0, 1, 0, 0, 0), 'metallic',     8,   8),
    ((),            (1, 0, 0, 1, 0, 0), 'metallic',     8,   8),
    ((),            (1, 1, 0, 0, 0, 0), 'metallic',     8,   8),
    ((),            (1, 1, 1, 1, 1, 1), 'metallic',     8,   8),
    ((0,),          (0, 0, 1, 0, 0, 0), 'metallic',    20,  20),
    ((0,),          (1, 0, 0, 1, 0, 0), 'metallic',    20,   0),
    ((0,),          (1, 1, 0, 0, 0, 0), 'metallic',    20,  20),
    ((0,),          (1, 1, 1, 1, 1, 1), 'metallic',    20,   0),
    ((1,),          (0, 0, 1, 0, 0, 0), 'metallic',     8,   8),
    ((1,),          (1, 0, 0, 1, 0, 0), 'metallic',     8,   0),
    ((1,),          (1, 1, 0, 0, 0, 0), 'metallic',     8,   0),
    ((1,),          (1, 1, 1, 1, 1, 1), 'metallic',     8,   0),
    ((2,),          (0, 0, 1, 0, 0, 0), 'metallic',    12,   0),
    ((2,),          (1, 0, 0, 1, 0, 0), 'metallic',    12,  12),
    ((2,),          (1, 1, 0, 0, 0, 0), 'metallic',    12,   0),
    ((2,),          (1, 1, 1, 1, 1, 1), 'metallic',    12,   0),
    ((0, 1),        (0, 0, 1, 0, 0, 0), 'metallic',     4,   4),
    ((0, 1),        (1, 0, 0, 1, 0, 0), 'metallic',     4,   0),
    ((0, 1),        (1, 1, 0, 0, 0, 0), 'metallic',     4,   0),
    ((0, 1),        (1, 1, 1, 1, 1, 1), 'metallic',     4,   0),
    ((0, 2),        (0, 0, 1, 0, 0, 0), 'metallic',     4,   0),
    ((0, 2),        (1, 0, 0, 1, 0, 0), 'metallic',     4,   0),
    ((0, 2),        (1, 1, 0, 0, 0, 0), 'metallic',     4,   0),
    ((0, 2),        (1, 1, 1, 1, 1, 1), 'metallic',     4,   0),
    ((0, 1, 2),     (0, 0, 1, 0, 0, 0), 'metallic',     8,   0),
    ((0, 1, 2),     (1, 0, 0, 1, 0, 0), 'metallic',     8,   0),
    ((0, 1, 2),     (1, 1, 0, 0, 0, 0), 'metallic',     8,   0),
    ((0, 1, 2),     (1, 1, 1, 1, 1, 1), 'metallic',     8,   0),
    ((0,),          (0, 0, 1, 0, 0, 0), 'periodic',    20,  20),
    ((0,),          (1, 0, 0, 1, 0, 0), 'periodic',    20,   0),
    ((0,),          (1, 1, 0, 0, 0, 0), 'periodic',    20,   0),
    ((0,),          (1, 1, 1, 1, 1, 1), 'periodic',    20,   0),
    ((1,),          (0, 0, 1, 0, 0, 0), 'periodic',     8,   0),
    ((1,),          (1, 0, 0, 1, 0, 0), 'periodic',     8,   0),
    ((1,),          (1, 1, 0, 0, 0, 0), 'periodic',     8,   0),
    ((1,),          (1, 1, 1, 1, 1, 1), 'periodic',     8,   0),
    ((2,),          (0, 0, 1, 0, 0, 0), 'periodic',    12,   0),
    ((2,),          (1, 0, 0, 1, 0, 0), 'periodic',    12,  12),
    ((2,),          (1, 1, 0, 0, 0, 0), 'periodic',    12,   0),
    ((2,),          (1, 1, 1, 1, 1, 1), 'periodic',    12,   0),
    ((0, 1),        (0, 0, 1, 0, 0, 0), 'periodic',     4,   0),
    ((0, 1),        (1, 0, 0, 1, 0, 0), 'periodic',     4,   0),
    ((0, 1),        (1, 1, 0, 0, 0, 0), 'periodic',     4,   0),
    ((0, 1),        (1, 1, 1, 1, 1, 1), 'periodic',     4,   0),
    ((0, 2),        (0, 0, 1, 0, 0, 0), 'periodic',     4,   0),
    ((0, 2),        (1, 0, 0, 1, 0, 0), 'periodic',     4,   0),
    ((0, 2),        (1, 1, 0, 0, 0, 0), 'periodic',     4,   0),
    ((0, 2),        (1, 1, 1, 1, 1, 1), 'periodic',     4,   0),
    ((0, 1, 2),     (0, 0, 1, 0, 0, 0), 'periodic',     8,   0),
    ((0, 1, 2),     (1, 0, 0, 1, 0, 0), 'periodic',     8,   0),
    ((0, 1, 2),     (1, 1, 0, 0, 0, 0), 'periodic',     8,   0),
    ((0, 1, 2),     (1, 1, 1, 1, 1, 1), 'periodic',     8,   0),
    ),
    #: :func:`offdiag_fold_roles` reproduces ``combinations`` exactly — 52/52
    #: combinations, 480/480 guarded cases per policy, 0 disagreements — and that
    #: is what makes the rule above the MEASURED boundary rather than a reading of
    #: the array path. Re-measured by ``gate_cuda_folded_offdiag_rowmask.py`` and
    #: at the merge bar by ``test_offdiag_constitutive_pml_real.py``.
    "rule_disagreements_with_measurement": 0,
    #: What the arm is worth, recomputed 2026-08-20 against the all-families
    #: census with the union analyzer (``analyze_cuda_coverage.py``, positional
    #: path). The widened predicate admits the SAME 16 update_E slots as the
    #: shipped one, introduces no overlap with another family, and moves the union
    #: not at all.
    "corpus": {
        "census": "parity/meep_gpu/results/"
                  "cuda_predicate_coverage_2026-08-20_all_families",
        "rows": 186,
        "slots": 759,
        "union_admitted_before": 557,
        "union_admitted_after": 557,
        "offdiag_update_E_slots_admitted_before": 16,
        "offdiag_update_E_slots_admitted_after": 16,
        "update_E_slots_no_family_serves": 70,
        "of_those_refused_first_by_the_mirror_clause": 26,
        "of_those_a_fold_is_a_live_partner_axis": 20,
        "of_those_no_off_diagonal_row_survived_at_all": 6,
        "slots_licensed": 0,
        "slots_needing_a_mirror_branch_in_the_kernel": 20,
        "overlaps_introduced": 0,
        #: WHAT A FOLD-BLIND WIDENING WOULD BUY, and it is the sharpest reason
        #: this record exists. Drop the fold clause outright — admit every fold,
        #: which is what "the refusal is too broad" invites — and the union goes
        #: 557 -> 576. All 19 are configurations the device gate measured
        #: DIVERGENT: every one drives row mask (1, 0, 0, 1, 0, 0) with a folded
        #: partner axis. So the 19 slots are reachable and they are wrong, which
        #: is a different fact from the arm being worth nothing, and only the
        #: row-mask rule separates the two. (The twentieth live-partner row,
        #: examples/absorbed_power_density.py, is refused by the dispersion clause
        #: as well.)
        "slots_a_fold_blind_widening_would_gain": 19,
        "and_all_nineteen_were_measured_divergent": True,
    },
    #: An UPPER BOUND on both directions of the corpus arithmetic: the census
    #: record carries no dtypes, strides, contiguity or base addresses, so the
    #: replayed array tail passes by construction. A row whose real arrays would
    #: fail that tail is counted as licensable here and would not be served.
    "corpus_counts_are_an_upper_bound": True,
    "what_would_move_the_twenty": (
        "a MIRROR branch in the emitted coord_dn returning "
        "stepping.MIRROR_SOURCE_INDEX, plus a per-term parity argument the "
        "kernel has no concept of today (a signature change, not a branch), "
        "plus a third boundary code and the launcher mapping for it — a new "
        "kernel variant with its own byte gate, not a line in this module"),
    "not_installed_because": (
        "worth zero corpus slots (see corpus above), and "
        "offdiag_boundary_codes raises KeyError on a folded axis while reading "
        "the fold as periodic is measured wrong on 32/480 guarded cases per "
        "policy"),
}


def _offdiag_wall_flags_from(facts):
    """The wall-mask triple off already-read grid facts — one copy, two callers."""
    return tuple(
        int(bool(facts["metallic"][axis]) and not bool(facts["mirrored"][axis]))
        for axis in range(3)
    )


def offdiag_wall_mask_flags(grid):
    """``wm_x``/``wm_y``/``wm_z``, decided the way the mask itself decides.

    ``stepping._mask_metallic_wall_coupling`` (stepping.py:1282) asks, per axis,
    ``grid.is_metallic(axis) and not grid.is_mirrored(axis)`` — the grid's own
    DECLARATION, deliberately NOT the resolved ghost rule
    :func:`real_pml_boundary_kinds` returns, which is a different question with
    the same answer only while folds are refused. Carried separately for the same
    reason the certified curl keeps its wall mask separate from ``bc_*``: the day
    a fold is admitted, conflating the two zeroes a plane MEEP steps.

    RAISES if the grid cannot answer, exactly as :func:`real_pml_boundary_kinds`
    does; the fail-closed refusal lives in :func:`_grid_facts`, and the launcher
    reaches this only after the predicate has admitted.
    """
    return _offdiag_wall_flags_from({
        "mirrored": tuple(bool(grid.is_mirrored(a)) for a in range(3)),
        "metallic": tuple(bool(grid.is_metallic(a)) for a in range(3)),
    })


def _base_address(array):
    """The device or host base address of a volume, or None when unreadable.

    Used only by the row-alias clause. CuPy exposes ``.data.ptr`` and NumPy
    ``__array_interface__['data'][0]``; neither module is imported to read them.
    """
    try:
        data = getattr(array, "data", None)
        pointer = getattr(data, "ptr", None)
        if pointer is not None:  # CuPy
            return int(pointer)
        interface = getattr(array, "__array_interface__", None)
        if isinstance(interface, dict):  # NumPy
            return int(interface["data"][0])
    except Exception:  # noqa: BLE001
        return None
    return None



def constitutive_sub_lattice(side: str) -> bool:
    """``half_integer`` for one side — the single place the pairing is decided.

    The predicate checks the coefficient vectors on THIS sub-lattice and the
    launch wrapper binds them from the same answer, so the two cannot disagree
    about which table a side reads. Getting them out of step is a plane of wrong
    values in the absorber, not a crash.
    """
    try:
        return bool(CONSTITUTIVE_SIDES[side]["half_integer"])
    except KeyError:
        raise ValueError(
            f"side must be one of {sorted(CONSTITUTIVE_SIDES)}, got {side!r}"
        ) from None


#: The spellings ``stepping`` uses, transcribed. ``CYL_AXIS`` is ``"axis"``, not
#: ``"cyl_axis"``: this module said the latter until 2026-08-15, which made
#: :func:`real_pml_boundary_kinds` disagree with ``stepping._boundary_kinds`` on
#: 19 of the 186 measured corpus rows while its own docstring claimed "same
#: precedence, same spellings". No verdict moved — neither string is in
#: :data:`BC_CODES`, and the ``cylindrical`` and ``is_axis`` clauses fire first —
#: so it was a false-alarm column in the record rather than an admission defect.
#: ``test_step_curl_pml_real.py`` now checks these four against ``stepping``'s.
PERIODIC, MIRROR, METALLIC, CYL_AXIS = "periodic", "mirror", "metallic", "axis"


def real_pml_boundary_kinds(grid: Any) -> Tuple[str, ...]:
    """Resolve each axis's ghost rule the way ``stepping._boundary_kinds`` does.

    Same precedence, same spellings: the cylindrical axis first, then a mirror
    fold, then the declared condition. Neither of the first two has a kernel, so
    :func:`covers_real_pml_curl` refuses them — this reproduces the resolution
    rather than importing it because ``step_curl_kernels`` is also loaded
    standalone, by path, outside the package (the bit-identity probe).

    An axis that merely ABSORBS still resolves to ``periodic``: MEEP's boundary
    condition comes from ``fields::use_bloch`` and knows nothing about PML, and
    the wave leaving one face re-enters at the other where the layer eats it.
    Calling a uniformly-absorbing X metallic was measured at 6.31e-04 against CPU
    MEEP where the wrap gives 4.00e-07 (``stepping._boundary_kinds``).

    RAISES if the grid cannot answer. Callers inside this module go through
    :func:`_grid_facts`, which is where the fail-closed refusal lives; this stays
    the direct, unguarded reading because it is also the function the tests
    compare against ``stepping._boundary_kinds`` on real grids.
    """
    return _boundary_kinds_from({
        "axis": tuple(bool(grid.is_axis(a)) for a in range(3)),
        "mirrored": tuple(bool(grid.is_mirrored(a)) for a in range(3)),
        "metallic": tuple(bool(grid.is_metallic(a)) for a in range(3)),
    })


def _boundary_kinds_from(facts: dict) -> Tuple[str, ...]:
    """The resolution itself, off already-read facts — one copy, two callers."""
    return tuple(
        CYL_AXIS if facts["axis"][axis]
        else MIRROR if facts["mirrored"][axis]
        else (METALLIC if facts["metallic"][axis] else PERIODIC)
        for axis in range(3)
    )


def real_curl_boundary_codes(grid: Any) -> Tuple[Any, Any]:
    """The real-field PML curl pair's three integer codes for this grid, or a refusal.

    Returns ``(codes, refusal)`` with exactly one of them None, so a caller can
    fail closed instead of catching. ``step_curl_kernels.real_curl_boundary_codes``
    is the launch-side face of this and raises where this refuses; the PREDICATE
    consults the same function, which is what stops the launcher and the predicate
    from answering differently about one grid — the defect the shipped pair carried
    between 2026-08-19 and today, when ``covers_real_pml_curl`` began admitting a
    folded METALLIC axis that ``real_pml_boundary_codes(kinds)`` still raised on.
    """
    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return None, unreadable
    return _real_curl_boundary_codes_from(facts)


def _real_curl_boundary_codes_from(facts: dict) -> Tuple[Any, Any]:
    """The code triple off already-read facts — one copy, two callers.

    THE FOLD SPLIT LIVES HERE AND NOT IN :func:`_boundary_kinds_from`. That helper
    is "one copy, two callers" by its own docstring and the CONSTITUTIVE siblings
    are the other caller; putting the split there widened them as a side effect —
    measured 2026-08-19, caught by
    ``test_the_sibling_predicates_did_not_widen_with_this_one`` on both E and H. A
    widening that leaks into a family whose device evidence never covered it is
    exactly the over-claim this suite exists to stop.
    """
    codes = []
    for axis, kind in enumerate(_boundary_kinds_from(facts)):
        if kind == MIRROR:
            problem = _fold_termination_problem(facts, axis)
            if problem is not None:
                return None, problem
            # A folded METALLIC axis takes METALLIC's own code: its far ghost is
            # the same zero, its cell-0 mask is the same mask, and it has no top
            # plane to drop. A folded PERIODIC one gets the third code, whose only
            # difference IS that top-plane mask.
            kind = METALLIC if facts["metallic"][axis] else MIRROR_PERIODIC
        # Stated positively: the kinds the kernels implement, named. A kind added
        # to Grid later is refused here because it is not on this list.
        if kind not in REAL_CURL_BC_CODES:
            return None, (f"axis {axis} resolves to boundary {kind!r}, which has "
                          f"no kernel")
        codes.append(REAL_CURL_BC_CODES[kind])
    return tuple(codes), None


def _fold_termination_problem(facts: dict, axis: int):
    """Why this folded axis's termination cannot be trusted; None when it can.

    TWO INDEPENDENT ROUTES TO ONE FACT, AND THEY MUST AGREE. ``Grid.stored_cells``
    exceeds ``Grid.owned_cells`` exactly when the axis is mirrored and NOT metallic
    (``stepping._stored_past_owned``, stepping.py:1503-1518, which spells out the
    ``update_ntot`` num + 1 slot ``connect_the_chunks`` fills); ``Grid.is_metallic``
    is the declaration the same split is derived from. A disagreement means one of
    them has drifted and NEITHER may be trusted, because the consequence is a
    top-plane mask applied where MEEP steps the plane, or withheld where it does
    not — a wrong answer on one plane of one component, not a crash. Same
    cross-check, and the same reasoning, as ``triton_kernels/symmetry.py``'s
    ``folded_axis_kinds`` (:465-533).
    """
    past_owned = facts["stored_past_owned"][axis]
    metallic = bool(facts["metallic"][axis])
    if past_owned is None:
        return (f"axis {axis} is folded and this grid cannot say whether it stores "
                f"the slot past MEEP's owned window; the top-plane mask cannot be "
                f"decided from is_metallic alone")
    if bool(past_owned) == metallic:
        return (f"axis {axis} is folded with stored_cells > owned_cells "
                f"{bool(past_owned)} and is_metallic {metallic}; the two routes to "
                f"the fold's termination disagree and neither can be trusted")
    return None


# ---------------------------------------------------------------------------
# FAIL-CLOSED READING. A RAISE IS NOT A REFUSAL unless something catches it.
#
# These predicates are read by a planner deciding whether to dispatch, so an
# exception escaping one is a CRASHED RUN where a fail-closed "no" was the correct
# answer. The module already said that in as many words, and applied it at exactly
# one call site (``inverse_epsilon_for``). Constructed and measured on 2026-08-15:
# a grid with no ``is_metallic``, a grid with no ``has_symmetry``, a grid whose
# ``is_axis`` raises, and an array whose ``.flags`` raises each escaped BOTH
# predicates; ``fields`` with no ``condfac_for`` escaped the curl's. All four are
# latent rather than live — a real ``Grid``/``Fields`` always carries them, which
# is why the 186-row battery reports "predicate RAISED on: 0" — but "latent" is a
# statement about today's callers, and the fail-closed rule is the one property
# this module argues for by name.
#
# The reads are gathered rather than guarded one at a time so that there is ONE
# refusal site per object, which keeps the trace test's admitted-list honest: a
# reader can see every question asked in one place, instead of a wrapper around
# each.
# ---------------------------------------------------------------------------

def _grid_facts(grid: Any) -> Tuple[Any, Any]:
    """Every question both predicates ask a grid, asked once and fail-closed.

    Returns ``(facts, refusal)`` with exactly one of them ``None``.
    """
    try:
        return {
            "has_symmetry": bool(grid.has_symmetry()),
            "cylindrical": bool(getattr(grid, "cylindrical", False)),
            "mirrored": tuple(bool(grid.is_mirrored(a)) for a in range(3)),
            "axis": tuple(bool(grid.is_axis(a)) for a in range(3)),
            "metallic": tuple(bool(grid.is_metallic(a)) for a in range(3)),
            "has_bloch": bool(getattr(grid, "has_bloch", False)),
            "bfast_active": bool(getattr(grid, "bfast_active", False)),
            "beta": getattr(grid, "beta", 0.0),
            "shape": tuple(grid.shape),
            "stored_past_owned": _stored_past_owned_from(grid),
        }, None
    except Exception as exc:  # noqa: BLE001 - the failure IS the refusal
        return None, (f"the grid could not answer a question this predicate has to "
                      f"ask: {type(exc).__name__}: {exc}")


def _stored_past_owned_from(grid: Any) -> Tuple[Any, Any, Any]:
    """``stepping._stored_past_owned`` per axis, transcribed — None where unanswerable.

    Transcribed from stepping.py:1503-1518 rather than imported, because this
    module imports nothing (see the header: the predicate has to be evaluable and
    mutable on a machine with no GPU, and ``stepping`` pulls the whole engine in).
    ``test_step_curl_pml_real.py`` pins this against ``stepping._stored_past_owned``
    on every grid the slice builds, so the transcription cannot drift silently.

    ONE DELIBERATE DIVERGENCE FROM THE ORIGINAL, and it is the fail-closed rule.
    ``stepping._stored_past_owned`` answers ``False`` for a grid with no callable
    ``owned_cells`` — a convenience for the stub grids in ``test_stepping.py``,
    and the right answer there because nothing downstream of it masks. Answering
    False HERE would be a silent wrong answer: a folded PERIODIC axis would be
    classified as metallic, dispatched to a kernel with no top-plane mask, and
    step a plane MEEP does not own. So an axis this grid cannot answer for comes
    back ``None``, and the fold clause refuses it by name.
    """
    stored = getattr(grid, "stored_cells", None)
    owned = getattr(grid, "owned_cells", None)
    if not (callable(stored) and callable(owned)):
        return (None, None, None)
    out = []
    for axis in range(3):
        try:
            out.append(bool(grid.is_mirrored(axis))
                       and int(stored(axis)) > int(owned(axis)))
        except Exception:  # noqa: BLE001 - an unanswerable axis is not a False one
            out.append(None)
    return tuple(out)


def _backend(grid: Any) -> Tuple[Any, Any]:
    """``(grid.xp, its __name__)``, or ``(None, None)`` when the grid cannot say.

    Both at once, and fail-closed on the read itself: a separate
    ``getattr(grid, "xp", None)` beside this would raise on a grid whose ``xp`` is
    a property that fails, which is the very case the backend clause is supposed
    to answer "no" to.
    """
    try:
        xp = getattr(grid, "xp", None)
        return (None, None) if xp is None else (xp, xp.__name__)
    except Exception:  # noqa: BLE001
        return None, None


def _array_problem(label: str, array: Any, xp: Any, shape: Tuple) -> Any:
    """Why ``array`` is not a float32 C-contiguous volume of ``shape``; None if it is."""
    try:
        if array is None:
            return f"{label} is not allocated"
        if array.dtype != xp.float32:
            return f"{label} is {array.dtype}, not float32"
        if tuple(array.shape) != shape:
            return f"{label} has shape {tuple(array.shape)}, not the grid's {shape}"
        if not array.flags.c_contiguous:
            return f"{label} is not C-contiguous"
    except Exception as exc:  # noqa: BLE001
        return f"{label} could not be inspected: {type(exc).__name__}: {exc}"
    return None


def _coefficient_vector_problem(label: str, vector: Any, xp: Any, axis: int,
                                shape: Tuple) -> Any:
    """Why ``vector`` is not this axis's float32 profile; None if it is.

    THE SHAPE, NOT JUST THE SIZE. ``PML._reshape_for_broadcast`` stores these as
    ``(n,1,1)`` / ``(1,n,1)`` / ``(1,1,n)``, so comparing ``.size`` alone admits
    another axis's profile whenever the two extents coincide — on a cubic grid,
    always. The broadcast shape says which axis it is FOR.

    AND THE CONTIGUITY, which the field arrays were checked for and these were
    not. The kernel takes ``reshape(-1)`` of this view; measured, a ``(2n,1,1)``
    array strided by 2 reshapes to a NON-contiguous view (the trailing size-1 dims
    let NumPy re-stride rather than copy), so ``real_constitutive_tables`` raises
    ValueError on exactly an input this predicate used to admit — a predicate
    saying yes to a configuration its own launcher then refuses.
    """
    expected = tuple(shape[axis] if a == axis else 1 for a in range(3))
    try:
        if vector is None:
            return f"pml.{label} is missing"
        if vector.dtype != xp.float32:
            return f"pml.{label} is {vector.dtype}, not float32"
        if vector.size != shape[axis]:
            return (f"pml.{label} has {vector.size} entries, not the axis's "
                    f"{shape[axis]}")
        if tuple(vector.shape) not in (expected, (shape[axis],)):
            return (f"pml.{label} has shape {tuple(vector.shape)}, not axis "
                    f"{axis}'s broadcast shape {expected}")
        if not vector.flags.c_contiguous:
            return (f"pml.{label} is not C-contiguous; the kernel indexes "
                    f"reshape(-1) of it as a bare vector")
    except Exception as exc:  # noqa: BLE001
        return f"pml.{label} could not be inspected: {type(exc).__name__}: {exc}"
    return None


#: WHAT LICENSES THE FOLD IN :func:`covers_real_pml_curl`, and what it does not.
#:
#: Until 2026-08-19 this predicate refused every mirror-folded run outright: "the
#: fold changes the ghost rule and the ownership mask, and a curl has both". The
#: refusal was retired in two measured steps, not one, and the steps have DIFFERENT
#: kinds of evidence behind them — which is the reason this record exists rather
#: than a sentence in the clause.
#:
#: STEP 1, 2026-08-19 (``results/cuda_folded_curl_2026-08-19/``): the SHIPPED pair,
#: unchanged, was run on folded grids with the folded axis handed ``BC_METALLIC``.
#: A fold whose every folded axis is METALLIC came back 48/48 bit-identical at one
#: launch and 48/48 at 60, under both float32 subnormal policies. NO DEVICE CODE
#: WAS WRITTEN for that half: a folded METALLIC axis's far ghost is the same exact
#: zero, its cell-0 mask is the same mask, and ``_stored_past_owned`` is false so
#: there is no top plane to drop.
#:
#: STEP 2, the same run's other arm and today's fix: a fold with a PERIODIC
#: termination diverged 0/64 — and the divergence was a SPECIFICATION, not a
#: smear. 19,649 differing words (18,633 under flush), of which 19,649 lay on the
#: mask-delta planes and ZERO anywhere else, always the same entry
#: ``('oracle_only', -1)``: the LAST STORED SLOT, for exactly the components whose
#: Yee shift on the folded axis is 1. The kernels now carry that mask
#: (``BC_MIRROR_PERIODIC``), and ``gate_cuda_folded_curl.py`` re-ran to measure it.
#:
#: THE SPLIT IS NOT PORTABLE ACROSS FAMILIES, and a test enforces exactly that.
#: ``covers_real_pml_constitutive`` admitted its fold with no device code because
#: its sub-step reads its own cell; this one needed a kernel branch. The offdiag
#: constitutive predicate has NO fold evidence at all and still refuses one.
CURL_FOLD_ADMISSION = {
    "gate": "parity/meep_gpu/gate_cuda_folded_curl.py",
    "artifact_round_1_no_device_code": ("parity/meep_gpu/results/"
                                        "cuda_folded_curl_2026-08-19"),
    "artifact_round_2_top_plane_mask": ("parity/meep_gpu/results/"
                                        "cuda_folded_curl_2026-08-20_topmask_r2"),
    "host": "the GPU host, NVIDIA RTX A6000 (GPU 4), CuPy 13.5.1",
    "policies_cut_under": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    #: Simultaneous mirror planes the gate swept. A fold with more planes than
    #: this is refused. RAISED 2 -> 3 ON 2026-08-20 BY RE-RUNNING THE GATE, not by
    #: argument: three FOLD_SPECS were added (``fold_XYZ_metallic``,
    #: ``fold_XYZ_periodic``, ``fold_XYZ_mixed``, stored extents 9/10/11 so a cube
    #: cannot hide an index confusion), two of them were added to
    #: ``MUTATION_SPEC_LABELS`` so the battery is armed on the shape the cap names,
    #: and the gate's own summary now DERIVES this number from the cases that
    #: actually ran (``max_folded_planes_scored``) rather than from its spec tuple
    #: -- which is how the previous 2 came to be a fact about a list. The gate also
    #: refuses to release unless BOTH folded terminations reached the maximum plane
    #: count, so the cap cannot rest on the metallic arm alone. Three is every
    #: plane a 3-D grid has, so this clause is now unreachable rather than merely
    #: unmet. Same cap, same reason, same number as
    #: :data:`CONSTITUTIVE_FOLD_ADMISSION`'s.
    "folded_planes_swept": 3,
    "planes_round_artifacts": ("parity/meep_gpu/results/"
                               "cuda_folded_curl_2026-08-20_threeplane"),
    #: ``results/`` is gitignored, so the digests are how a reader binds this claim
    #: to the bytes that produced it on a checkout that does not carry them.
    "planes_round_artifact_sha256": {
        "keep/gate.json":
            "05743b18bd412a1057905f0a26d1ad805d5e8887a703857101506503f24c91db",
        "flush/gate.json":
            "88f71a6b323eb445564bbdbdb67aec76135df844e534e557ca101d2f9f429b3f",
    },
    #: THE SUBJECT WAS NOT TOUCHED. The gate compiled the SHIPPED device strings;
    #: this is the digest the artifact recorded for them, and it is the digest of
    #: the file in the tree.
    "planes_round_subject_sha256": {
        "meep_gpu/cuda_kernels/step_curl_kernels.py":
            "ba106d813b951bc77627b571e878d90e06747d9c88c479fb5af11dd9d65e0ae8",
    },
    "planes_round_recorded_utc": "2026-08-20T16:00:00Z",
    "planes_round_cases_scored_per_policy": 456,
    "planes_round_max_folded_planes_scored": 3,
    "planes_round_folded_planes_by_arm": {
        "folded_all_metallic": (1, 2, 3), "folded_all_periodic": (1, 2, 3),
        "folded_mixed": (2, 3), "unfolded": (0,)},
    "planes_round_source_mutations_scored": 20,
    "planes_round_all_mutations_as_required": True,
    # RECOMPUTED FROM THE CENSUS with the union analyzer, on scripts copied
    # byte-identical from the 2026-08-20 closeout round so the two are
    # commensurable to the slot. The two slots are ``TestLDOS.test_ldos_3D``'s
    # step_B and step_D; its update_H/update_E are the constitutive round's.
    "planes_round_census": ("parity/meep_gpu/results/"
                            "cuda_predicate_coverage_2026-08-20_w15"),
    "planes_round_union_before": 653,
    "planes_round_union_after": 668,
    "planes_round_slots_this_arm_gained": 2,
    "planes_round_rows_gained": ("TestLDOS.test_ldos_3D",),
    #: THE MISDECLARATION CONTROL STILL FIRES at three planes: the folded-PERIODIC
    #: arm handed BC_METALLIC diverges on 56 of 56 cases with every differing word
    #: on the mask-delta planes and ZERO elsewhere (23018 words under keep, 21854
    #: under flush). An identity there would have meant the licensed arm's
    #: identity came from something other than the branch.
    "planes_round_misdeclaration_still_diverges": "0/56 identical, 0 words off the "
                                                  "mask-delta planes",
    "folded_axes_swept": ("x", "y", "z"),
    "plane_parities_swept": (1, -1),
    "full_count_parities_swept": ("even", "odd"),
    "terminations_swept": ("metallic", "periodic"),
    #: Round 2, per policy, under the SHIPPED code resolution
    #: (``mirror_resolved``): every arm identical at one launch AND at 60.
    "cases_scored_per_policy": 384,
    "identical_folded_all_metallic": "48/48",
    "identical_folded_all_periodic": "48/48",
    "identical_folded_mixed": "16/16",
    "identical_unfolded_controls": "16/16",
    "multi_step_launches_each": 60,
    #: THE MISDECLARATION CONTROL, and the number that makes the branch the cause.
    #: The 64 cases carrying a PERIODIC-terminated folded axis, handed the METALLIC
    #: code instead — which is exactly the pre-branch kernel — still diverge, with
    #: every differing word on the mask-delta plane and ZERO elsewhere.
    "periodic_terminated_fold_cases": 64,
    "identical_under_the_shipped_resolution": "64/64",
    "identical_when_misdeclared_metallic": "0/64",
    "misdeclared_differing_words_off_the_mask_delta_planes": 0,
    #: The kernel delta this admission rests on, in the mutations that arm it. A
    #: mask nothing can break is a mask this record cannot claim.
    "mutations_that_must_be_caught": ("drop_folded_periodic_top_mask",
                                      "drop_folded_periodic_near_mask",
                                      "drop_metallic_mask",
                                      "reverse_folded_axis_coefficients"),
    "top_plane_mask_mutation_caught": "3/3 per sub-step, both policies",
    "near_plane_mask_mutation_caught": "3/3 per sub-step, both policies",
    #: WHAT THE ROUND COST THE CERTIFIED SURFACE: nothing measurable. The branch
    #: is taken only on the third code, and the certification gate re-run on the
    #: EDITED device strings came back 120/120 at both policies, 60/60 multi-step,
    #: 0/120 unguarded.
    #:
    #: THIS IS THIS ROUND'S OWN EVIDENCE and it stays pointed at the run that
    #: produced it. ``certification.json`` no longer pins the pre-edit digests:
    #: the 2026-08-22 kernel rename moved every device string, and the record was
    #: re-cut against ``fused_pml_bit_identity_hand_2026-08-22_rename`` — a later
    #: run of the same probe over the same sweep, which returned the same numbers
    #: this comment reports.
    "recertification_evidence": ("parity/meep_gpu/results/"
                                 "fused_pml_bit_identity_hand_2026-08-20_topmask"),
}


#: WHAT RETIRED THE chi2/chi3 REFUSAL IN :func:`covers_real_pml_curl`, and what
#: it does not license.
#:
#: Until 2026-08-20 this predicate refused both curl sub-steps on any grid
#: carrying an instantaneous nonlinearity, with the census reason "instantaneous
#: chi2/chi3: a Pade factor on the constitutive product". The sentence is true of
#: ``update_E`` and false of this sub-step, and the clause was inherited -- the
#: same shape the 2026-08-20 nonlinear round found on ``update_H``, and the same
#: shape the mirror and dispersion clauses had before them.
#:
#: THE READING WAS PERFORMED RATHER THAN QUOTED. ``gate_cuda_nonlinear_curl_null``
#: parses ``meep_gpu/stepping.py`` with ``ast``, builds the module-level call
#: graph and reports which functions NAME a chi2/chi3/nonlinear symbol in CODE
#: (docstrings excluded, and the exclusion is recorded beside the un-excluded
#: reading, because two functions inside the curl's closure mention the
#: nonlinearity only in prose). Measured on 72 functions: four name one --
#: ``update_E``, ``_nonlinear_constitutive``, ``calc_nonlinear_u`` and
#: ``nonlinear_margin`` -- and the transitive call closures of ``step_B`` and
#: ``step_D`` contain NONE of them. The scan's POSITIVE CONTROL is a release
#: clause: ``update_E``'s closure must contain some, or a scan that finds nothing
#: anywhere reports an empty intersection for the curl and means nothing by it. It
#: found three.
#:
#: THAT IS A FACT ABOUT THE ORACLE. The device leg is the fact about the kernel:
#: the SHIPPED pair, not one byte changed, on grids carrying a chi, compared as
#: raw uint32 against ``stepping.step_B``/``step_D`` from one frozen state.
#:
#: THE PREMISE WAS ARMED. ``chi_pade_scale_the_curl`` rewrites the shipped curl
#: device string so the curl IS scaled by a quotient of the Pade factor's shape,
#: and it was CAUGHT 6/6 on each sub-step under each policy, with the artifact
#: recording that a kernel really was built from the mutated bytes. Without it the
#: 112/112 identity would be consistent with a comparator that sees nothing. The
#: RELEASE VERDICT ITSELF was then shown to flip: a run with that defect planted
#: for the whole sweep came back ``released=False`` with 6 of 6 cases diverging.
#:
#: THE FLOOR THAT MAKES IT A MEASUREMENT. A null gate's characteristic failure is
#: a fixture that never carried the thing it claims does not matter, so every case
#: measures whether the chi CHANGES A WORD: two triples from identical rng
#: streams, one with the chi and one without, one ``stepping.update_E`` each,
#: differing output words counted. Half the sweep (the ``subnormal_band`` value
#: class) came back at ZERO -- with every operand between 1e-45 and 1e-38, ``c2``
#: and ``c3`` underflow and ``calc_nonlinear_u`` returns exactly 1.0 on both paths
#: -- and those cases are scored in a SEPARATE ARM that claims less, rather than
#: being counted as evidence about a live nonlinearity. The live arm is required
#: to be non-empty on each sub-step and to contain the corpus's own 1-D shape.
CURL_NONLINEAR_ADMISSION: dict = {
    "gate": "parity/meep_gpu/gate_cuda_nonlinear_curl_null.py",
    "artifacts": ("parity/meep_gpu/results/"
                  "cuda_nonlinear_curl_null_2026-08-21_rename"),
    #: ``results/`` is gitignored, so the digests are how a reader binds this claim
    #: to the bytes that produced it on a checkout that does not carry them.
    "artifact_sha256": {
        "keep/gate.json":
            "6f0622ca876225d0a04e7fae0b872d197fdadd0ae75f006a1ed6b08bcb97fc62",
        "flush/gate.json":
            "1bcbe9f939c02030fafe3047ed9d8defe1c0800224c395c87c7f816dbbfa2bed",
        "falsify/gate.json":
            "62ca30051d318abcc145bc9e1f2e6f15f79bb1555fea7a7a81b2b696551b466d",
    },
    #: THE SUBJECT: the bytes whose behaviour was measured. Not one of them was
    #: changed by this round -- the gate launches the SHIPPED kernels and mutates
    #: only a copy of the device text.
    "subject_sha256": {
        "meep_gpu/cuda_kernels/step_curl_kernels.py":
            "c6608d5fb52acbb617653407059731bcf8db77ad5d83ef0c26c6bbad9ffd34f6",
    },
    #: BACKFILLED 2026-08-26, which is what the drift clause in
    #: ``test_step_curl_pml_real.py`` says is owed: "this record carries no
    #: device_sha256/code_sha256 today -- so the byte rule below still stands until
    #: those are backfilled". The file hash above moved when the twelve INHERITED
    #: kernels were deleted from the module, and a file hash is the wrong instrument
    #: for that question: what this admission measured is what NVRTC compiles, and the
    #: deletion removed twelve strings the gate never launched.
    #:
    #: MEASURED, so ``weld_survives_edit`` can establish it rather than assume it: the
    #: two certified device strings hash to 362d6420 and cd4edf1d, which are EXACTLY
    #: the values ``certification.json``'s device_source_sha256 recorded when the
    #: 2026-08-22 re-cut compiled and launched them. Pinning them here means any
    #: FUTURE edit that reaches the compiled program is refused outright, which is
    #: strictly stronger than the file-hash rule it replaces -- a docstring edit no
    #: longer trips it, and a kernel edit no longer slips through a re-recorded hash.
    "device_sha256": {
        "meep_gpu/cuda_kernels/step_curl_kernels.py": {
            "kind": 'cuda',
            "digests": {
            "_REAL_PML_PRELUDE": "2cbd6d25c547e0ea0b0dd69d38322169b8064aec056e7da9278a361242801025",
            "_step_B_pml_real_kernel_code": "362d6420b05c20483cc5255b5ccd7a82e989f0e1809a2d6cbf46b51077e8fac7",
            "_step_D_pml_real_kernel_code": "cd4edf1d4f569e2eb4a574ea6cfc4237b5c4076309ab7047c15b9aa1e43e83b8"
            },
        },
    },
    "recorded_utc": "2026-08-22T07:27:48Z",
    "host": "the GPU host, NVIDIA RTX A6000 (sm_86), CuPy 13.5.1",
    "cases_per_policy": 112,
    "identical_single_launch": 112,
    "identical_at_sixty_launches": 112,
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    "policies_really_differed": ("any_ftz_true_reached_nvrtc FALSE under keep, "
                                "TRUE under flush"),
    "courants_swept": (0.5, 0.35),
    "value_classes_swept": ("uniform", "subnormal_band"),
    "chi_amplitudes_swept": ("moderate", "near_pole"),
    "boundary_triples_swept": ("all_periodic", "all_metallic", "mixed_pmp",
                               "mixed_mpm", "one_dimensional",
                               "partly_nonlinear", "volume_chi"),
    # ---- what makes it non-vacuous ----
    "cases_with_a_measurably_live_chi": 56,
    "cases_with_the_chi_underflowed": 56,
    "live_cases_per_sub_step": {"step_B": 28, "step_D": 28},
    "chi_words_moved_in_update_E_on_the_live_arm": "240 to 5760",
    "chi_expansion_range": "4.1e-33 to 0.226 (pole bound 1/3)",
    # ---- the static scan ----
    "stepping_functions_parsed": 72,
    "stepping_functions_naming_chi": ("_nonlinear_constitutive", "calc_nonlinear_u",
                                      "nonlinear_margin", "update_E"),
    "chi_in_step_B_closure": (),
    "chi_in_step_D_closure": (),
    "scan_positive_control": ("update_E's closure names _nonlinear_constitutive, "
                              "calc_nonlinear_u and update_E"),
    # ---- what was armed ----
    "premise_armed_as_a_defect": "chi_pade_scale_the_curl",
    "premise_verdict": "CAUGHT 6/6 on each of step_B and step_D, both policies",
    "source_mutations_scored": 18,
    "all_mutations_as_required": True,
    "verdict_shown_to_flip": ("chi_pade_scale_the_curl planted for the whole run: "
                              "released=False, 6/6 cases diverging"),
    # RECOMPUTED FROM THE CENSUS with the union analyzer, on scripts copied
    # byte-identical from the 2026-08-20 closeout round so the two rounds are
    # commensurable to the slot. FOUR slots: step_B and step_D on each of the two
    # corpus rows that carry an instantaneous nonlinearity. Both rows are now
    # covered at EVERY sub-step, because the nonlinear constitutive family already
    # served their update_H/update_E -- which is why this widening is worth two
    # whole rows rather than four isolated slots.
    "census": ("parity/meep_gpu/results/"
               "cuda_predicate_coverage_2026-08-20_w15"),
    "union_slots_before": 653,
    "union_slots_after": 668,
    "slots_this_admission_gained": 4,
    "rows_gained": ("3rd-harm-1d.py", "Test3rdHarm1d.test_3rd_harm_1d"),
    "rows_now_covered_at_every_sub_step": ("3rd-harm-1d.py",
                                           "Test3rdHarm1d.test_3rd_harm_1d"),
    "what_it_does_not_license": (
        "the nonlinear CONSTITUTIVE sub-step, which is a different family with "
        "its own kernel (nonlinear_constitutive.py), its own predicate and its "
        "own gate. Nothing here strengthens or weakens it.",
        "a chi combined with a fold, a conductivity, BFAST, special_kz, a Bloch "
        "phase or complex storage: each is refused by another clause and none was "
        "swept here.",
        "a MAGNETIC (H-side) nonlinearity: fields.set_nonlinear_volumes refuses "
        "one by name, so there is nothing to measure.",
        "any dispatch. No module in meep_gpu imports cuda_kernels at all, so a "
        "widened predicate licenses a MEASUREMENT and not a production step.",
    ),
}


def covers_real_pml_curl(fields: Any, pml: Any, grid: Any, sub_step: str) -> tuple:
    """Whether ``_step_B_fused_pml_real`` / ``_step_D_fused_pml_real`` may serve this run.

    ``sub_step`` is ``"step_B"`` (``step_B_pml_real``) or ``"step_D"``
    (``step_D_pml_real``). Returns ``(covered, reason)``; ``reason`` names
    the first refusal, so a configuration that unexpectedly stays on the array
    path says why.

    THE ARGUMENT IS REQUIRED, and it is required because it was once absent.
    There are two kernels, one per sub-step, and a predicate with no way to name
    which one it is answering for had to answer for both at once — which means
    answering with the INTERSECTION of two clause sets, and refusing each sub-step
    for the other's blockers. Measured on the 186-row corpus battery, that cost
    seven sub-steps against the sibling track's per-sub-step predicate, every one
    of them a CUDA refusal Triton admits and none the other way. An optional
    ``sub_step=None`` aggregate would have kept the over-refusal reachable by
    omission, which is the shape this argument exists to remove.

    WHAT IS PER-SUB-STEP HERE, and what decided each one by reading ``stepping``:

    * **a conductivity** — ``_apply_curl`` reads ``fields.condfac_for(term.target)``
      (stepping.py:508), per TERM. ``step_B``'s terms target Bx/By/Bz
      (``B_CURL_TERMS``, stepping.py:214) and ``step_D``'s target Dx/Dy/Dz (:219),
      so a D conductivity routes ``step_D`` to the three-history recurrence and
      leaves ``step_B`` an ordinary curl. The corpus row that shows it is
      ``TestAdjointSolver.test_damping``.

    WHAT IS NOT A CURL FACT AT ALL, and was refused here anyway:

    * **a registered polarization.** ``step_B`` reads the stored E arrays and
      ``step_D`` the stored H arrays (``_component_snapshot``, stepping.py:2425);
      neither ``step_B``, ``step_D``, ``_apply_curl``, ``_curl_operands``,
      ``_curl_from_operands`` nor ``_apply_pml_update`` mentions a polarization,
      an inverse epsilon or ``displacement_minus_polarization``. What dispersion
      changes is the VALUES ``update_E`` writes into the array this sub-step
      DIFFERENCES, which is not an operation this kernel performs. The refusal
      cost three corpus rows two sub-steps each. It is replaced below by the
      invariant it was standing in for — ``stores_E`` — rather than deleted:
      admitting dispersion is sound only while the E the curl differences is the
      array ``update_E`` wrote, never a ``D * inv_eps`` recomputed inside the curl
      (``_read_component``, stepping.py:2438-2454, whose E branch reads an inverse
      epsilon ONLY when ``stores_E`` is false).

    * **an instantaneous chi2/chi3**, refused until 2026-08-20 with "a Pade factor
      on the constitutive product". The sentence is true of ``update_E`` --
      ``calc_nonlinear_u`` (stepping.py:1025) is applied where E is recovered from
      D -- and false of this sub-step, which differences the STORED E and H
      arrays. Same inherited shape as the polarization clause above, and retired
      the same way: a static scan of ``stepping`` PERFORMED rather than quoted
      (with a positive control, because a scan that finds nothing anywhere reports
      an empty intersection for everything it is asked about), a device leg on
      grids carrying a live chi, and the premise armed as a defect and caught. The
      refusal cost two corpus rows two sub-steps each. See
      :data:`CURL_NONLINEAR_ADMISSION`.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {sorted(CURL_SUB_STEPS)}, got {sub_step!r}")

    xp, backend = _backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    if getattr(fields, "force_complex_fields", False):
        return False, "complex64 storage: the recurrence is the same but the storage is not"
    if not (pml is not None and getattr(pml, "is_active", False)):
        return False, "no active PML layer"
    # Read AFTER the backend short-circuit, deliberately: the predicate battery
    # factors its coverage number on that clause firing first (PROVENANCE.md), and
    # a grid that cannot answer must not pre-empt it.
    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    # THE FOLD IS ADMITTED AT BOTH TERMINATIONS, and each half was measured on its
    # own (:data:`CURL_FOLD_ADMISSION`). A folded METALLIC axis needed no device
    # code at all — the shipped pair was already exact on it. A folded PERIODIC one
    # diverged on exactly one plane, and the kernels now carry the branch that
    # closes it. The per-axis split, its two-route cross-check and the code triple
    # itself are all in :func:`_real_curl_boundary_codes_from`, below the
    # cylindrical clauses, so the launcher and this predicate cannot disagree.
    #
    # DO NOT GENERALISE A FOLD VERDICT ACROSS FAMILIES. The constitutive pair took
    # the fold for free because its sub-step reads its OWN CELL; a curl reads
    # NEIGHBOURS and has an ownership mask, so the two verdicts had to be measured
    # separately — and they differed, the curl needing device code the constitutive
    # family did not. ``covers_real_pml_offdiag_constitutive`` still refuses a fold
    # outright, with no fold evidence of its own.
    folded = [axis for axis in range(3) if facts["mirrored"][axis]]
    if facts["has_symmetry"] and not folded:
        # Fail closed: has_symmetry with no mirrored axis is a ghost rule this
        # module has never resolved. Unreachable on today's corpus, kept because
        # keying the admission on `mirrored` alone would silently admit it.
        return False, "symmetry with no folded axis: unresolved ghost rule"
    if len(folded) > CURL_FOLD_ADMISSION["folded_planes_swept"]:
        # THE PLANE CAP, which the constitutive sibling has carried since its own
        # widening and this predicate did not: between 2026-08-19 and today it
        # admitted a THREE-plane fold no gate has ever swept, while the slot replay
        # that priced the widening held those rows back. The predicate now agrees
        # with the record it was priced against.
        return False, (f"{len(folded)} mirror planes at once: the fold gate swept "
                       f"{CURL_FOLD_ADMISSION['folded_planes_swept']} simultaneous "
                       f"planes, so a {len(folded)}-plane fold is an extrapolation "
                       f"from the measurement rather than part of it")
    if facts["cylindrical"]:
        return False, "cylindrical (Dcyl): prefix-sum radial derivative and axis-row rules"
    for axis in range(3):
        if facts["axis"][axis]:
            return False, f"axis {axis} is the cylindrical r = 0 axis"
    # THE CODE TRIPLE ITSELF, from the one function the launcher also calls. Its
    # refusals are this predicate's refusals: an axis whose two routes to the
    # fold's termination disagree, an axis the grid cannot answer for, and a kind
    # with no code.
    _codes, no_codes = _real_curl_boundary_codes_from(facts)
    if no_codes is not None:
        return False, no_codes
    if facts["has_bloch"]:
        return False, "nonzero Bloch k: the wrapped plane carries a phase real storage cannot hold"
    if facts["bfast_active"]:
        return False, "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"
    # THIS sub-step's targets only. The opposite curl's sigma reaches the opposite
    # kernel and nothing here — see the docstring for the line that decides it.
    try:
        conductive = [component for component in CURL_SUB_STEPS[sub_step]
                      if fields.condfac_for(component) is not None]
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, (f"fields could not be asked for a conductivity: "
                       f"{type(exc).__name__}: {exc}")
    if conductive:
        return False, (f"{conductive[0]} carries a conductivity: routes to the "
                       f"three-history conductive-PML recurrence")
    # THE INVARIANT DISPERSION IS ADMITTED ON, stated positively. The kernel binds
    # ``fields.Ex``/``Ey``/``Ez`` directly and ``step_B`` differences whatever
    # ``get_E`` returns; with E stored that is the same array, and without it a
    # freshly computed ``D * inv_eps`` — one polarization out of date the moment a
    # pole is live (fields.py:1008-1021). An active PML already forces stored E
    # (``enable_pml_storage`` -> ``enable_field_storage`` -> ``_stored_E``,
    # fields.py:678, :653, :676), so this is belt and braces WITH a reason: an edit
    # that ever makes ``stores_E`` optional under PML has to be caught here, and
    # the dispersion admission above rests on nothing else.
    if not getattr(fields, "stores_E", False):
        return False, "E is recomputed from D rather than stored"
    # AN INSTANTANEOUS chi2/chi3 IS ADMITTED, and until 2026-08-20 it was refused
    # here with "a Pade factor on the constitutive product". THE SENTENCE IS TRUE
    # AND IS NOT ABOUT THIS SUB-STEP: the Pade factor (calc_nonlinear_u,
    # stepping.py:1025) is applied where E is recovered from D, and step_B/step_D
    # difference the STORED E and H arrays. The clause was inherited from the
    # constitutive family, exactly as this predicate's mirror and dispersion
    # clauses once were. See :data:`CURL_NONLINEAR_ADMISSION` for the measurement
    # that retired it -- including the static scan of ``stepping`` that this
    # paragraph would otherwise be asserting, and the device leg that armed the
    # premise as a defect and caught it.
    shape = facts["shape"]
    # THE DIMENSIONALITY CLAUSE, which the constitutive predicate carried from the
    # start and this one did not: a 4-D grid was admitted, and the kernels take
    # (nx, ny, nz), so that is a WRONG-INDEX answer rather than a launch failure.
    # Unreachable from a real Grid (grid.py:583 assigns a 3-tuple), which is why
    # it went unnoticed until the two predicates were censused against each other.
    if len(shape) != 3:
        return False, f"grid shape {shape} is not three-dimensional"
    # THE INT32 INDEX RANGE, which the constitutive predicate carried from the
    # start and this one did not. ``step_B_pml_real`` declares ``int idx``
    # and guards on ``idx >= nx * ny * nz`` in int too
    # (``step_curl_kernels.py:1689-1690``), and its neighbour offsets add ``ny*nz``
    # to that same int. Past 2**31 elements the arithmetic stops being an identity
    # and starts being a wrong answer at a wrapped index — a silently corrupted
    # volume, not a launch failure. Same clause, same reason, same wording as
    # :func:`covers_real_pml_constitutive`'s, because it is the same defect.
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    if cells >= 2 ** 31:
        return False, f"{cells} cells exceeds the kernel's int32 index range"
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
                 "fu_Dx", "fu_Dy", "fu_Dz"):
        problem = _array_problem(name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem
    for axis, name in enumerate(("x", "y", "z")):
        for half in (True, False):
            suffix = "_h" if half else ""
            for label in ("kms", "sinv"):
                attribute = f"{label}_{name}{suffix}"
                problem = _coefficient_vector_problem(
                    attribute, getattr(pml, attribute, None), xp, axis, shape)
                if problem is not None:
                    return False, problem
    return True, "covered"


#: The ghost rules the real-storage constitutive pair serves. MIRROR IS IN IT AND
#: IS NOT IN :data:`BC_CODES`, which is the deliberate divergence from
#: :func:`covers_real_pml_curl` and the one clause of this predicate with a DEVICE
#: verdict rather than a reading behind it -- see
#: :data:`CONSTITUTIVE_FOLD_ADMISSION`. ``CYL_AXIS`` was absent for the same
#: reason and is now present for the same reason it stopped applying: this list
#: read "Dcyl would rest on the argument alone", and
#: :data:`CONSTITUTIVE_CYLINDRICAL_ADMISSION` is the measurement that answers it
#: -- 80 of the 96 scored cases resolve the r axis to ``CYL_AXIS`` (48 with z
#: metallic, 32 with z periodic) and every one was bit-identical on the GPU host, at
#: both launch granularities, with the refusal's own premise armed as a defect
#: and caught. Neither kind is admitted by argument.
CONSTITUTIVE_BOUNDARY_KINDS: Tuple[str, ...] = (PERIODIC, METALLIC, MIRROR, CYL_AXIS)

#: WHAT THE FOLD CLAUSE IS WORTH AND WHAT PAID FOR IT.
#:
#: Until 2026-08-19 this predicate refused every mirror-folded run with "the fold
#: changes the stored extent, and the extent is what turns a cell index into a
#: coefficient index". The PREMISE is true and the CONCLUSION does not follow:
#: ``PML._compute_coefficients`` (pml.py:693-695, :706-708) builds every kps/kms
#: vector from ``grid.nx``/``ny``/``nz`` -- the STORED extent, which is what the
#: fold halves (grid.py:570-575) -- and the field arrays are allocated on the same
#: ``grid.shape``, so both operands move together and the kernel's
#: ``i = idx/(ny*nz)`` indexes a table of exactly the length the fold left.
#:
#: THAT READING WAS NOT TRUSTED. ``parity/meep_gpu/gate_cuda_folded_constitutive.py``
#: ran the SHIPPED pair, unchanged, on folded grids: 112 cases per policy of which
#: 96 folded, 112/112 bit-identical to ``stepping.update_H``/``update_E`` at one
#: launch and at 60, under both float32 subnormal policies. The unguarded control
#: DIVERGED on every one of the 96 folded cases (48 at each courant), which is
#: the gate's summary figure of 56/56 at the inexact courant read the other way:
#: over folded cases alone, and at both courants. The fold-specific mutation
#: -- reversing the folded axis's coefficient vectors, which is the refusal armed
#: as a defect -- was CAUGHT 10/10, and the folded axis's max|coefficient - 1|
#: ranged 0.567 to 3.238, so a coefficient-index error there was visible rather
#: than hidden behind an identity table.
#:
#: WHAT THE 2026-08-19 SWEEP DID NOT REACH, AND WHAT THE 2026-08-20 RE-RUN DID.
#: The first round swept one folded axis (X, Y and Z each), both plane parities,
#: both folded terminations, an odd full count, and TWO planes at once (XY and
#: XZ), so ``folded_planes_swept`` was 2 -- and the corpus's one triply-folded row
#: (``TestLDOS.test_ldos_3D``) was refused for 2 slots here and 2 on the curl.
#: That refusal was a fact about the gate's spec tuple rather than about the
#: kernel, which the clause itself said in as many words.
#:
#: THE RE-RUN IS WHAT RAISED IT. Three FOLD_SPECS were added
#: (``fold_XYZ_metallic``, ``fold_XYZ_periodic``, ``fold_XYZ_mixed``, stored
#: extents 9/10/11 so a cube cannot hide an index confusion), two of them were
#: added to ``MUTATION_SPEC_LABELS`` so the battery is armed on the shape the cap
#: names, and the gate's summary now DERIVES the number from the cases that ran
#: (``max_folded_planes_scored``) and refuses to release unless both folded
#: terminations reached it. 136 cases per policy, 136/136 bit-identical at one
#: launch and at 60, both float32 subnormal policies, every host and source
#: mutation as required. Three is every plane a 3-D grid has, so the clause is now
#: unreachable rather than merely unmet.
#:
#: THE SLOT COUNTS are re-derived from the census record's per-row
#: ``configuration`` blocks by running THIS predicate over stand-ins built from
#: them; the replay is validated first by reproducing every recorded verdict and
#: first-refusal string that is not the retired fold one. All 128 slots gained
#: were previously refused by the fold clause AND BY NOTHING ELSE, on rows that
#: are folded, PML-active and real-storage, across at most the two simultaneous
#: planes the gate swept (57 one-plane, 71 two-plane) -- so this is a widening
#: and not a leak, which the totals alone could not distinguish.
#:
#: THEY ARE AN UPPER BOUND: the array and coefficient-vector tail cannot be
#: replayed from the record, only the decision logic above it. That tail was
#: separately exercised on REAL folded ``Grid``/``Fields``/``PML`` triples for
#: every fold shape the gate swept, where each axis's kps/kms vector was measured
#: to have exactly the folded stored extent's length -- which is the reading's
#: load-bearing fact and the one the retired refusal denied.
CONSTITUTIVE_FOLD_ADMISSION: dict = {
    "gate": "parity/meep_gpu/gate_cuda_folded_constitutive.py",
    # ONE RUN NOW BACKS BOTH SECTIONS. The 2026-08-22 kernel rename moved the
    # bytes this admission is about, so it was re-cut on a run that compiled the
    # SHIPPED module -- and today's gate sweeps one, two and three fold planes in
    # a single sweep, so the base numbers and the third-plane numbers come from
    # the same artifact. The 2026-08-19 two-plane run is history: its sweep is no
    # longer reproducible (FOLD_SPECS now carries the fold_XYZ_* specs and there
    # is no flag to restrict it) and its verdict describes bytes that no longer
    # ship.
    "artifacts": ("parity/meep_gpu/results/"
                  "cuda_folded_constitutive_2026-08-21_rename"),
    # RE-CUT 2026-09-26 (recut_fold_admission.py): recorded_utc and host are those of
    # parity/meep_gpu/results/cuda_regate_2026-09-25_roundb/cuda_folded_constitutive_2026-08-21_rename
    # -- the fold gate re-run on constitutive_kernels.py at 437e65f66fef -- and equal
    # certification.json:folded_constitutive_2026-08-20_threeplane. They were
    # 2026-08-21T23:05:27Z / the GPU host. Every count below reproduced on that run (136
    # cases, 120 folded, 136 and 136 identical per policy, 3 planes); artifacts and
    # the planes_round_* fields are left naming the run they came from.
    "recorded_utc": "2026-09-25T13:51:42Z",
    "host": "the GPU host",
    "cases_per_policy": 136,
    "folded_cases_per_policy": 120,
    "identical_single_launch": 136,
    "identical_at_sixty_launches": 136,
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    "folded_terminations_swept": ("metallic", "periodic"),
    "folded_planes_swept": 3,
    "census": ("parity/meep_gpu/results/"
               "cuda_predicate_coverage_2026-08-16_offdiag"),
    "slots_before": 73,
    "slots_after": 201,
    # WAS 2, AND IS NOW 0: the third plane was swept on 2026-08-20 and the row
    # below is admitted. Kept as a field rather than deleted because it is what
    # the widening is worth on this corpus.
    "slots_refused_for_a_third_fold_plane": 0,
    "row_with_three_fold_planes": "TestLDOS.test_ldos_3D",
    # ------------------------------------------------------------------
    # THE 2026-08-20 THIRD-PLANE RE-RUN
    # ------------------------------------------------------------------
    "planes_round_artifacts": ("parity/meep_gpu/results/"
                               "cuda_folded_constitutive_2026-08-21_rename"),
    #: ``results/`` is gitignored, so the digests are how a reader binds this claim
    #: to the bytes that produced it on a checkout that does not carry them.
    "planes_round_artifact_sha256": {
        "keep/gate.json":
            "582835ed4587ad8c64e7488dc3d6a3178589d5a3f82bde344f70636953e2f6cc",
        "flush/gate.json":
            "ec2bca8d3283ac563bcbfac1515862d2d04c6c37baa5633cc712135040dec525",
    },
    #: THE SUBJECT WAS NOT TOUCHED: the gate compiled the SHIPPED device strings.
    "planes_round_subject_sha256": {
        "meep_gpu/cuda_kernels/constitutive_kernels.py":
            "3a0b46400d10f93690097ff700f147dd185f9702f7a46aa5d80705d62abea4d3",
    },
    # THE SLOT ARITHMETIC OF THE RAISE, over the same 2026-08-16 census the two
    # earlier links use, so the three compose end to end: 73 -> 201 (the fold),
    # 201 -> 207 (Dcyl), 207 -> 209 (this). The two slots are
    # ``TestLDOS.test_ldos_3D``'s update_H and update_E; its two CURL slots are
    # the curl round's, counted separately.
    "planes_round_slots_before": 207,
    "planes_round_slots_after": 209,
    "planes_round_rows_gained": 1,
    # AND THE UNION NUMBER, recomputed with the census's own analyzer over a
    # byte-identical copy of the 2026-08-20 closeout round's scripts, so the two
    # rounds are commensurable to the slot: 653 -> 668 of 759 over all three of
    # the day's widenings, of which TWO slots are this one's (the same row's
    # step_B and step_D are the curl round's). Rows covered at every sub-step
    # 136 -> 139, and TestLDOS.test_ldos_3D is one of the three.
    "planes_round_census": ("parity/meep_gpu/results/"
                            "cuda_predicate_coverage_2026-08-20_w15"),
    "planes_round_union_before": 653,
    "planes_round_union_after": 668,
    "planes_round_recorded_utc": "2026-08-20T15:46:00Z",
    "planes_round_host": "the GPU host, NVIDIA RTX A6000 (sm_86), CuPy 13.5.1",
    "planes_round_cases_per_policy": 136,
    "planes_round_folded_cases_per_policy": 120,
    "planes_round_identical_single_launch": 136,
    "planes_round_identical_at_sixty_launches": 136,
    "planes_round_max_folded_planes_scored": 3,
    "planes_round_folded_planes_by_termination": {
        "metallic": (1, 3), "periodic": (1, 3), "mixed": (2, 3)},
    "planes_round_host_mutations_caught": {
        "reverse_folded_axis_coefficients": "14/14",
        "swap_constitutive_sublattice": "16/16", "swap_kps_kms": "16/16"},
    "planes_round_source_mutations_scored": 15,
    "planes_round_all_mutations_as_required": True,
    "planes_round_policies_really_differed": ("any_ftz_true_reached_nvrtc FALSE "
                                              "under keep, TRUE under flush"),
    #: WHAT THIS CAP DOES NOT LICENSE. ``dispersive_kernels`` used to read this
    #: field for its own fold clause; it now carries
    #: :data:`~.dispersive_kernels.ADE_FOLD_PLANES_SWEPT` instead, pinned at 2,
    #: because the ADE gate has never scored a three-plane case. Raising a cap in
    #: one family must not widen another through a shared constant.
    "does_not_license": (
        "the ADE update_P family, which reads its own cap",
        "the off-diagonal constitutive family, which refuses a fold outright and "
        "has no fold evidence of its own",
    ),
}


CONSTITUTIVE_CYLINDRICAL_ADMISSION: dict = {
    "gate": "parity/meep_gpu/gate_cuda_cylindrical_constitutive.py",
    "artifacts": ("parity/meep_gpu/results/"
                  "cuda_cylindrical_constitutive_2026-08-20"),
    "artifact_sha256": "3be1957e8593d0f78a346707772ed95b6038574381758e680f1aa847ce91f87d",
    "recorded_utc": "2026-08-20T03:12:00Z",
    "host": "the GPU host",
    "cases_scored": 96,
    "dcyl_cases": 80,
    "single_launch_identical": 96,
    "multi_step_identical": 96,
    "m_values_swept": (-1, 0, 1, 2, 3, 5),
    "m_classes_swept": ("m0", "m1", "m2plus"),
    "z_terminations_swept": ("metallic", "periodic"),
    "host_mutations_caught": ("reverse_radial_axis_coefficients",
                             "swap_constitutive_sublattice", "swap_kps_kms"),
    "guard_control": "the contraction guard is load-bearing on this sub-step",
    "census": ("parity/meep_gpu/results/"
               "cuda_predicate_coverage_2026-08-20_all_families"),
    # RECOMPUTED at the merge bar against the 2026-08-16 census, on the same
    # denominator the fold admission uses, so the two widenings compose in one
    # arithmetic: 73 slots before either, 201 after the fold, 207 after this.
    # Every one of the six is a PML-active, real-storage Dcyl row -- the three
    # the corpus carries (TestPMLCylindrical.test_pml_cyl at m = 0 and
    # TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields idx0/idx1), each at
    # both sides. The other thirteen Dcyl rows are complex storage and are NOT
    # admitted: that is a different kernel and this gate did not score it.
    "slots_before": 201,
    "slots_after": 207,
    "rows_gained": 3,
    "what_it_does_not_license": (
        "the Dcyl CURL, which genuinely needs new kernels -- the radial "
        "prefix-sum derivative and the axis-row rules. cylindrical_kernels.py "
        "carries them for m = 0 and nothing carries them for |m| >= 1.",
        "the COMPLEX constitutive pair, which is a different kernel and was not "
        "scored by this gate.",
    ),
}


def covers_real_pml_constitutive(fields: Any, pml: Any, grid: Any,
                                 side: str) -> tuple:
    """Whether ``update_H_pml_real`` / ``update_E_pml_real`` may serve this run.

    ``side`` is ``"H"`` (``stepping.update_H``) or ``"E"`` (``stepping.update_E``).
    Returns ``(covered, reason)``; ``reason`` names the FIRST refusal, so a
    configuration that unexpectedly stays on the array path says why.

    THE SUB-STEP. ``stepping._apply_constitutive_pml`` (stepping.py:2112), MEEP's
    ``step_update_EDHB`` with ``dsigw`` active::

        fwprev = fw[i]; fw[i] = g[i] * u[i];
        f[i] += (kap+sig)*fw[i] - (kap-sig)*fwprev

    It reads NO NEIGHBOUR. That removes the ghost rule and the ownership mask
    from the kernel, and until 2026-08-19 it removed neither from this predicate:
    the fold and cylindrical clauses were kept because they are about the STORED
    EXTENT rather than about any stencil. The fold half of that has since been
    MEASURED and does not survive it — see :data:`CONSTITUTIVE_FOLD_ADMISSION`.
    The cylindrical half has since been MEASURED too and does not survive it --
    see :data:`CONSTITUTIVE_CYLINDRICAL_ADMISSION`. Neither half of the original
    refusal survives contact with a device.

    WHAT THIS ADMITS THAT :func:`covers_real_pml_curl` REFUSES, and why. Two
    things, both measured rather than argued.

    **A MIRROR FOLD**, up to two simultaneous planes. The curl has a real ghost
    rule and a real ownership mask and a fold changes both; this sub-step has
    neither (``_mask_non_owned_cells``, stepping.py:1912, has exactly three call
    sites and no constitutive function is among them), and the shipped pair was
    run unchanged on 96 folded cases per policy at 112/112 bit-identical. Worth
    128 of the corpus's slots — see :data:`CONSTITUTIVE_FOLD_ADMISSION` for the
    verdict, the floors that made it non-vacuous and the one configuration the
    sweep did not reach.

    **A CONDUCTIVITY**.
    ``fields.condfac_for`` is read in ``stepping._apply_curl`` (stepping.py:508)
    and nowhere else in the module — ``update_H``, ``update_E`` and
    ``_apply_constitutive_pml`` never mention it. A conductive run therefore
    routes the curl that WRITES the carrying component to the three-history
    recurrence, and leaves the opposite curl and both constitutive sub-steps as
    ordinary products. Since 2026-08-15 the curl predicate takes a sub-step and
    charges the sigma to that curl alone, so on the corpus's one such row
    (``TestAdjointSolver.test_damping``, a D conductivity) three of the four
    sub-steps are admitted and only ``step_D`` is refused. Refusing it here would
    be a refusal with no line behind it.

    WHAT THE E SIDE REFUSES THAT THE H SIDE DOES NOT. Everything that changes
    what ``source`` is:

    * a registered polarization — ``source`` becomes ``D - sum P``
      (``stepping.py:1010``, ``fields.py:1096-1105``) and ``update_P`` closes the
      step. The kernel binds ``fields.D*`` directly, which is valid ONLY where
      ``displacement_minus_polarization`` aliases the D array (fields.py:1097-1098);
    * an off-diagonal ``chi1inv`` row — the row product reads the OTHER
      components' volumes at neighbouring cells (``stepping.py:1001-1008``,
      ``_offdiagonal_terms`` :1196) and the sub-step stops being element-wise;
    * ``stores_E`` false — ``update_E`` then writes ``field[...] = constitutive``
      with no absorption term at all (``stepping.py:1022``), a different sub-step.
      An active PML already forces stored E, so this is belt and braces WITH a
      reason: an edit that ever makes ``stores_E`` optional under PML has to be
      caught here.

    An instantaneous chi2/chi3 is refused on BOTH sides, because the Pade factor
    REPLACES the constitutive product (``stepping.py:999-1000``) rather than being
    added to it.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(
            f"side must be one of {sorted(CONSTITUTIVE_SIDES)}, got {side!r}")
    spec = CONSTITUTIVE_SIDES[side]

    xp, backend = _backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    if getattr(fields, "force_complex_fields", False):
        return False, "complex64 storage: the recurrence is the same but the storage is not"
    if not (pml is not None and getattr(pml, "is_active", False)):
        # Without an absorber ``update_H`` returns immediately (stepping.py:944)
        # and ``update_E`` takes the plain assignment (:1022). Neither is this
        # kernel, and the array path is the bit-identical one for both.
        return False, "no active PML layer"
    # Read AFTER the backend short-circuit, for the reason the curl predicate
    # states: the battery factors its coverage number on that clause firing first.
    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    # A Dcyl GRID IS ADMITTED, and the refusal it replaces was explicitly
    # provisional: this docstring said "The cylindrical half stands, unmeasured
    # and therefore refused" until :data:`CONSTITUTIVE_CYLINDRICAL_ADMISSION`
    # measured it -- 96/96 cases bit-identical on the GPU host at every |m| class and
    # both z terminations, with the refusal's own premise armed as a defect
    # (``reverse_radial_axis_coefficients``, CAUGHT) so the admission is not an
    # argument from absence. The premise was TRUE and the conclusion did not
    # follow, exactly as it did not for the fold.
    #
    # THE r = 0 AXIS ROW STAYS REFUSED. That is a separate fact from the
    # coordinate system: ``stepping._cylindrical_axis_zero_B`` (:648) and
    # ``_cylindrical_axis_zero_D`` (:560) are called from step_B (:376) and
    # step_D (:457) -- the CURL -- and this predicate has never been asked about
    # a grid whose axis flag is set without the cylindrical flag. It is a
    # fail-closed clause, not an inherited one.
    for axis in range(3):
        if facts["axis"][axis] and not facts["cylindrical"]:
            return False, (f"axis {axis} reports the cylindrical r = 0 rule on a "
                           f"grid that does not report cylindrical coordinates; "
                           f"this predicate cannot answer for a grid that "
                           f"disagrees with itself")
    if facts["has_symmetry"] != any(facts["mirrored"]):
        # FAIL-CLOSED, and unreachable from a real ``Grid``: ``has_symmetry()`` is
        # ``bool(self.symmetry)`` and ``is_mirrored(axis)`` is
        # ``mirrors_by_axis[axis] is not None``, built from the same list
        # (grid.py:1091-1095, :1445-1446). The fold admission below is decided
        # PER AXIS, so a grid that disagrees with itself about whether it is
        # folded is a grid this predicate cannot answer for.
        return False, ("the grid reports has_symmetry() and no mirrored axis, or "
                       "the reverse; the fold admission is decided per axis and "
                       "cannot be read off a grid that disagrees with itself")
    folded = sum(1 for axis in range(3) if facts["mirrored"][axis])
    if folded > CONSTITUTIVE_FOLD_ADMISSION["folded_planes_swept"]:
        return False, (f"{folded} mirror planes at once: the fold gate swept "
                       f"{CONSTITUTIVE_FOLD_ADMISSION['folded_planes_swept']} "
                       f"simultaneous planes, so a {folded}-plane fold would be "
                       f"admitted by argument alone")
    for axis, kind in enumerate(_boundary_kinds_from(facts)):
        # A FOLD IS ON THIS LIST, unlike the curl predicate's, and the clause
        # stays anyway so that a ghost rule added to Grid later is refused
        # because it was never admitted.
        #
        # THIS REVERSES AN ARGUMENT THIS MODULE USED TO MAKE HERE — "coverage is
        # a SET, not a la carte", i.e. a grid whose ghost rule the curl kernel
        # cannot serve must not have two of its four sub-steps quietly taken over
        # by kernels that happen not to look at it. What replaced it is a device
        # verdict on this very pair on folded grids
        # (:data:`CONSTITUTIVE_FOLD_ADMISSION`), and the set-ness the old
        # argument protected is reported directly by the census as ROWS COVERED
        # AT EVERY SUB-STEP — which this widening leaves at 42, because the curl
        # still refuses every fold.
        if kind not in CONSTITUTIVE_BOUNDARY_KINDS:
            return False, f"axis {axis} resolves to boundary {kind!r}, which has no kernel"
    if facts["has_bloch"]:
        return False, "nonzero Bloch k: the wrapped plane carries a phase real storage cannot hold"
    if facts["bfast_active"]:
        return False, "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"
    # Both attributes, by name. ``Fields.has_nonlinearity`` (fields.py:966-967)
    # reads ``_chi2_components`` ONLY, so a configuration carrying chi3 alone
    # would pass a predicate that asked the property instead of the two maps.
    if getattr(fields, "_chi2_components", None) or getattr(fields, "_chi3_components", None):
        return False, "instantaneous chi2/chi3: a Pade factor replaces the constitutive product"
    if side == "E":
        if getattr(fields, "polarizations", None):
            return False, ("dispersion: update_E's source is (D - sum P), not D, "
                           "and update_P closes the step")
        if getattr(fields, "has_offdiagonal_epsilon", False):
            return False, ("off-diagonal chi1inv: the row product reads the other "
                           "components' volumes and this sub-step is element-wise")
        if not getattr(fields, "stores_E", False):
            return False, "E is recomputed from D rather than stored"

    shape = facts["shape"]
    if len(shape) != 3:
        return False, f"grid shape {shape} is not three-dimensional"
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    if cells >= 2 ** 31:
        # The kernel indexes with ``int``. Past 2**31 elements that stops being
        # an arithmetic identity and starts being a wrong answer at a wrapped
        # index — a silently corrupted volume, not a launch failure.
        return False, f"{cells} cells exceeds the kernel's int32 index range"
    for name in tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"]):
        problem = _array_problem(name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem
    if side == "E":
        # THREE volumes, one per component, from ``Fields.inverse_epsilon_for``
        # (fields.py:1337). They may all alias one array
        # (``set_isotropic_epsilon_volume``, fields.py:1321-1326) and the kernel
        # binds three pointers either way. What is NOT covered is a scalar, and
        # what must never happen is binding ``fields.inv_eps`` for all three —
        # that attribute is the Ez view (fields.py:1259-1260), which is what
        # ``update_E_pml_complex`` gets wrong.
        reader = getattr(fields, "inverse_epsilon_for", None)
        if not callable(reader):
            return False, "fields does not expose inverse_epsilon_for"
        for component in _ELECTRIC_COMPONENTS:
            try:
                volume = reader(component)
            except Exception as exc:  # noqa: BLE001
                # A RAISE IS NOT A REFUSAL unless it is caught here. This predicate
                # is read by a planner deciding whether to dispatch; an exception
                # escaping it is a crashed run where a fail-closed "no" was the
                # correct answer. ``inverse_epsilon_for`` raises ValueError on a
                # component it has no volume for (fields.py:1341-1344).
                return False, f"inverse_epsilon_for({component!r}) raised {exc!r}"
            if volume is None:
                return False, f"inverse_epsilon_for({component!r}) is None"
            if not getattr(volume, "shape", ()):
                return False, f"inverse_epsilon_for({component!r}) is a scalar, not a volume"
            problem = _array_problem(f"inverse_epsilon_for({component!r})",
                                     volume, xp, shape)
            if problem is not None:
                return False, problem
    # This side's own sub-lattice only — integer for H, half-integer for E. The
    # curl predicate checks BOTH because the curl kernel plans either sub-step;
    # a constitutive side reads one and binding the other is the half-cell error.
    suffix = "_h" if spec["half_integer"] else ""
    for axis, name in enumerate(("x", "y", "z")):
        for label in ("kps", "kms"):
            attribute = f"{label}_{name}{suffix}"
            problem = _coefficient_vector_problem(
                attribute, getattr(pml, attribute, None), xp, axis, shape)
            if problem is not None:
                return False, problem
    return True, "covered"


def covers_real_pml_offdiag_constitutive(fields: Any, pml: Any, grid: Any) -> tuple:
    """Whether ``update_E_pml_real_offdiag`` may serve this run.

    ONE SUB-STEP AND ONE SIDE, so there is no argument to take: ``stepping``
    installs off-diagonal ``chi1inv`` rows on the ELECTRIC constitutive relation
    alone (``update_E``'s ``elif offdiagonal:`` branch, stepping.py:1001-1008).
    Returns ``(covered, reason)`` with ``reason`` naming the FIRST refusal, the
    convention both predicates above use.

    THE DISJOINTNESS SEAM. :func:`covers_real_pml_constitutive` refuses every
    off-diagonal run at ``side="E"`` (its own clause), and that refusal STAYS —
    it is what keeps the two families from both claiming the sub-step. This
    predicate is its exact complement on that axis: clause (b) below REQUIRES at
    least one surviving row, so a run is admitted by at most one of the two and
    ``test_offdiag_constitutive_pml_real.py`` measures the disjointness rather
    than asserting it.

    WHY THE CLAUSE CHAIN IS WRITTEN OUT AGAIN RATHER THAN DELEGATED. The sibling
    predicate SHORT-CIRCUITS, so "refused for the off-diagonal clause and nothing
    else" is not a question its return value can answer: the clauses after that
    one were never evaluated. Delegating would therefore admit a configuration
    whose ``stores_E``, array-layout or coefficient-table clause was never
    reached. The chain is enumerated positively here for the same reason it is
    there, and a test pins the two chains equal on every axis except the
    off-diagonal one, so the duplication cannot drift silently.

    WHAT IS NEW HERE, and what each would be if it leaked:

    a. **at least one surviving row slot (INVERTED)**, counted over
       :data:`OFFDIAG_ROW_SLOTS` rather than the bare
       ``Fields.has_offdiagonal_epsilon`` flag (fields.py:1313-1315). A row
       planted past ``_validated_offdiagonal_rows`` under a diagonal key sets the
       flag with every slot dead, and the emitter refuses an all-dead row mask by
       raising — so counting the flag would hand a covered verdict to a
       configuration the builder then rejects.
    b. **every surviving row volume real, float32, C-contiguous and grid-shaped.**
       The installer validates the form (fields.py:1262-1310) and this says it
       again, because inferring coverage from another module's guard is what this
       package's rule forbids.
    c. **no surviving row volume aliases an E or f_w output.** Beyond the
       installer's reach: ``set_epsilon_volumes`` keeps the caller's array without
       copying (``astype(..., copy=False)``, fields.py:1296), so a row that IS one
       of this sub-step's outputs arrives legally installed. The coupling re-reads
       the partner volumes at NEIGHBOUR offsets while the outputs are being
       written, so an alias makes the answer depend on block schedule — a wrong
       answer that varies run to run.

    Conductivity is NOT a clause, for the reason the sibling predicate states: it
    is read in ``stepping._apply_curl`` (stepping.py:508) and nowhere else, so it
    disqualifies one curl and no constitutive sub-step.
    """
    spec = CONSTITUTIVE_SIDES["E"]

    xp, backend = _backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    if getattr(fields, "force_complex_fields", False):
        return False, "complex64 storage: the recurrence is the same but the storage is not"
    if not (pml is not None and getattr(pml, "is_active", False)):
        # Without an absorber ``update_E`` takes the plain assignment
        # (stepping.py:1022) — no ``f_w``, no recurrence. A different sub-step.
        return False, "no active PML layer"
    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    if facts["has_symmetry"]:
        # MEASURED, NOT INHERITED, AND MEASURED TOO BROAD — the arm this sentence
        # gives up is real: see :data:`FOLDED_OFFDIAG_ROW_MASK_ADMISSION`, where
        # 96 folded cases per policy are byte-identical to ``stepping.update_E``
        # at one launch and 13 at 60. The refusal STANDS at its present width for
        # two reasons the record carries as numbers rather than as opinions: the
        # arm admits ZERO corpus slots (186 rows, the same 16 update_E slots
        # before and after), and ``offdiag_boundary_codes`` has no code to hand a
        # folded axis, with the natural stand-in — reading the fold as periodic —
        # measured WRONG on 32 of 480 guarded cases per policy. Widening here
        # without the launcher and its own gate would promise a dispatch that
        # raises, in exchange for nothing.
        return False, ("mirror symmetry: the fold changes the stored extent, and "
                       "the extent is what turns a cell index into a coefficient index")
    if facts["cylindrical"]:
        return False, ("cylindrical (Dcyl): the axial extent moves every "
                       "coefficient index")
    for axis in range(3):
        if facts["mirrored"][axis]:
            # AND THE ENGINE DOES STEP THIS, which is why the refusal is stated as
            # this kernel family's rather than as the engine's: the docstring at
            # stepping.py:1219-1220 claims ``set_epsilon_volumes`` refuses a folded
            # axis at install, and ``_validated_offdiagonal_rows`` (fields.py:
            # 1262-1310) demonstrably installs rows unchanged on folded grids with
            # fold-equivalence measured 8.3e-13..4.7e-12. The stale docstring
            # belongs in those files; the refusal here is
            # about the stored extent and the wall mask's mirror abstention.
            #
            # THE SECOND OF THE TWO FOLD REFUSALS, and it is not redundant with
            # the first: they read different accessors (``grid.has_symmetry()``
            # against ``grid.is_mirrored(axis)``), so an object that answers them
            # inconsistently is refused rather than dispatched. Like the first, it
            # is broader than the measured boundary
            # (:data:`FOLDED_OFFDIAG_ROW_MASK_ADMISSION`, whose rule is
            # :func:`offdiag_fold_roles`) — and broader on purpose, for that
            # record's two stated reasons.
            return False, f"axis {axis} is mirrored"
        if facts["axis"][axis]:
            return False, f"axis {axis} is the cylindrical r = 0 axis"
    for axis, kind in enumerate(_boundary_kinds_from(facts)):
        # Stated positively: the two ghost rules the emitter writes, named. Unlike
        # the plain constitutive sub-step this one really does read neighbours, so
        # the clause is load-bearing here rather than belt and braces.
        if kind not in BC_CODES:
            return False, f"axis {axis} resolves to boundary {kind!r}, which has no kernel"
    if facts["has_bloch"]:
        return False, "nonzero Bloch k: the wrapped plane carries a phase real storage cannot hold"
    if facts["bfast_active"]:
        return False, "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"
    # Both maps by name, never ``has_nonlinearity`` — that property reads
    # ``_chi2_components`` alone (fields.py:966-967), so a chi3-only run would pass
    # a predicate that asked it. MEEP's most general case scales the WHOLE row
    # product (including the coupling) by the Pade factor, step_generic.cpp:590-601
    # as ``stepping._nonlinear_constitutive`` (:1066-1073) transcribes it — a fused
    # nonlinear+offdiag kernel is a later leg, not this one.
    if getattr(fields, "_chi2_components", None) or getattr(fields, "_chi3_components", None):
        return False, ("instantaneous chi2/chi3: the Pade factor scales the whole "
                       "row product, coupling included")
    if getattr(fields, "polarizations", None):
        # With poles the sources are per-component ``D - sum P`` scratch buffers
        # (fields.py:1107-1138) and the coupling reads THOSE, not the D primaries
        # the launcher binds. A fused dispersive+offdiag kernel is a later leg.
        return False, ("dispersion: update_E's source is (D - sum P), not D, "
                       "and update_P closes the step")
    # (a) INVERTED. Counted over the six slots, deliberately not the flag.
    rows = offdiag_row_volumes(fields)
    if not any(volume is not None for volume in rows):
        return False, ("no off-diagonal chi1inv row survived installation: that "
                       "configuration is covers_real_pml_constitutive(side='E')'s "
                       "and this predicate must not overlap it")
    if not getattr(fields, "stores_E", False):
        # Forced True by any surviving row (``set_epsilon_volumes`` calls
        # ``enable_field_storage``, fields.py:1254-1255) and by an active PML
        # independently. Belt and braces WITH a reason: an edit that ever makes
        # either implication optional has to be caught here.
        return False, "E is recomputed from D rather than stored"

    shape = facts["shape"]
    if len(shape) != 3:
        return False, f"grid shape {shape} is not three-dimensional"
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    if cells >= 2 ** 31:
        # The kernel indexes with ``int`` and its neighbour offsets add ``ny*nz``
        # to that same int. Past 2**31 cells that stops being an arithmetic
        # identity and starts being a wrong answer at a wrapped index.
        return False, f"{cells} cells exceeds the kernel's int32 index range"
    outputs = tuple(spec["targets"]) + tuple(spec["aux"])
    for name in outputs + tuple(spec["sources"]):
        problem = _array_problem(name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem
    reader = getattr(fields, "inverse_epsilon_for", None)
    if not callable(reader):
        return False, "fields does not expose inverse_epsilon_for"
    for component in _ELECTRIC_COMPONENTS:
        try:
            volume = reader(component)
        except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
            return False, f"inverse_epsilon_for({component!r}) raised {exc!r}"
        if volume is None:
            return False, f"inverse_epsilon_for({component!r}) is None"
        if not getattr(volume, "shape", ()):
            return False, f"inverse_epsilon_for({component!r}) is a scalar, not a volume"
        problem = _array_problem(f"inverse_epsilon_for({component!r})",
                                 volume, xp, shape)
        if problem is not None:
            return False, problem
    # (b) and (c): the surviving rows, readable, well-formed, and not an output.
    output_addresses = {}
    for name in outputs:
        address = _base_address(getattr(fields, name, None))
        if address is not None:
            output_addresses.setdefault(address, name)
    for index, volume in enumerate(rows):
        if volume is None:
            continue
        row, partner = OFFDIAG_ROW_SLOTS[index]
        label = f"chi1inv_offdiagonal[{row!r}][{partner!r}]"
        if not getattr(volume, "shape", ()):
            return False, f"{label} is a scalar, not a volume"
        problem = _array_problem(label, volume, xp, shape)
        if problem is not None:
            return False, problem
        address = _base_address(volume)
        if address is not None and address in output_addresses:
            return False, (f"{label} aliases output {output_addresses[address]}: "
                           f"the coupling re-reads the partner volumes at "
                           f"neighbour offsets while the outputs are written, so "
                           f"the answer would depend on block schedule")
    # HALF-INTEGER ONLY, the E side's own sub-lattice (stepping.py:1015 via
    # ``_constitutive_coefficients(..., half_integer=True)``).
    suffix = "_h" if spec["half_integer"] else ""
    for axis, name in enumerate(("x", "y", "z")):
        for label in ("kps", "kms"):
            attribute = f"{label}_{name}{suffix}"
            problem = _coefficient_vector_problem(
                attribute, getattr(pml, attribute, None), xp, axis, shape)
            if problem is not None:
                return False, problem
    return True, "covered"


# =============================================================================
# THE ADE POLARIZATION SUB-STEP (``update_P``)
# =============================================================================
#
# THE ONLY STRUCTURAL ZERO ON THE BOARD. The 2026-08-16 census measured
# ``update_P  0 / 15 admitted  <- NO CUDA KERNEL, NO PREDICATE``
# (``results/cuda_predicate_coverage_2026-08-16_offdiag/analysis.txt``). The other
# four sub-steps are refused for reasons; this one was refused for absence.
#
# THE SUB-STEP IS ELEMENT-WISE AND THE ARRAY PATH SAYS SO BY NAME.
# ``stepping.update_P`` (stepping.py:1360-1385) is a loop over
# ``PolarizationState.update`` and nothing else, and its docstring (:1398-1403)
# states the two properties this predicate is built on:
#
#     "Needs no boundary pass of any kind. The isotropic update is purely
#      element-wise ... Under mirror symmetry it therefore runs over the WHOLE
#      stored array with no ownership mask ... ``pml`` is accepted and unused
#      ... the absorber reaches P only through W."
#
# That is what separates this predicate from its two siblings above. Theirs index
# a per-axis coefficient VECTOR by cell coordinate, so a fold or a cylindrical
# axis moves every coefficient index and both are refused. This one indexes
# nothing but a flat element: ``P``, ``P_prev``, ``_scratch``, ``sigma`` and the
# drive are all allocated at ``grid.shape`` (dispersion.py:645-650), whatever the
# stored extent happens to be.

#: The two recurrence kinds the ADE body transcribes. Written as a literal rather
#: than imported for the reason every table in this module is: nothing here
#: imports the engine. The sibling track holds the same literal
#: (``triton_kernels/coverage.py:63``) and pins it against
#: ``dispersion.SUSCEPTIBILITY_KINDS``; ``test_ade_update_p.py`` pins this one
#: against both.
COVERED_SUSCEPTIBILITY_KINDS: Tuple[str, ...] = ("lorentzian", "drude")

#: A polarization may drive these and only these. ``PolarizationState`` allocates
#: over E components alone (dispersion.py:640-647) and ``from_meep`` refuses
#: ``H_susceptibilities`` (from_meep.py:1566-1572), so the clause is vacuous
#: TODAY and is written anyway -- this module's rule forbids inferring coverage
#: from another module's guard.
ADE_ELECTRIC_COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: The ghost rules this sub-step admits. MIRROR IS IN IT AND IS NOT IN
#: :data:`BC_CODES`, which is the deliberate divergence from both sibling
#: predicates and the one clause that has a number behind it -- see
#: :data:`ADE_MIRROR_ADMISSION`. ``CYL_AXIS`` is absent: refused below by name.
ADE_BOUNDARY_KINDS: Tuple[str, ...] = (PERIODIC, METALLIC, MIRROR)

#: WHAT THE MIRROR CLAUSE IS WORTH, counted over the 15 corpus rows that have an
#: ``update_P`` sub-step at all, off the shipped record's per-row
#: ``configuration`` and ``polarization`` blocks
#: (``results/cuda_predicate_coverage_2026-08-16_offdiag/per_row_*``). Kept as
#: data so a widening or a narrowing here has to move a number a test reads,
#: rather than only a paragraph.
#:
#: The five rows the fold clause decides are the four ``TestLoadDump`` 2-D rows
#: and ``absorbed_power_density.py``. Refusing them -- the sibling predicates'
#: clause, transplanted -- would leave 3, which is exactly the trio the census's
#: own projection reached by a different route (it required BOTH curls admitted
#: as well, and on this corpus the two conditions coincide).
#:
#: THIS IS A PROJECTION OVER A RECORDED CORPUS, NOT A DEVICE VERDICT. No hand-CUDA
#: ``update_P`` kernel has been gated. The number says what a gate would have to
#: cover, which is the job a predicate does before certification, not after.
ADE_MIRROR_ADMISSION: dict = {
    "record": ("parity/meep_gpu/results/"
               "cuda_predicate_coverage_2026-08-16_offdiag"),
    "rows_with_update_P": 15,
    "admitted_with_mirror": 8,
    "admitted_without_mirror": 3,
    "mirror_rows": (
        "TestLoadDump.test_load_dump_chunk_layout_file_2d",
        "TestLoadDump.test_load_dump_chunk_layout_sim_2d",
        "TestLoadDump.test_load_dump_structure_2d",
        "TestLoadDump.test_load_dump_structure_sharded_2d",
        "absorbed_power_density.py",
    ),
}


def ade_sigma_is_volume(state: Any, component: str) -> bool:
    """The ``SIGMA_IS_VOLUME`` specialization, decided where the clause checks it.

    ONE function so the compile-time choice and the predicate cannot disagree.
    Getting them out of step binds a scalar where the device signature declares a
    pointer, or the reverse -- a wrong answer, not a crash. The sibling track
    carries the same single-decider function (``triton_kernels/coverage.py:605``)
    because its gate's mutation m13 is exactly this disagreement.
    """
    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    return bool(getattr(sigma, "shape", ()))


def _ade_drive_problem(fields: Any, component: str) -> Any:
    """Why ``drive_field(component)`` is not the pointer this family binds.

    THE SINGLE MOST LIKELY SILENT WRONG ANSWER IN DISPERSION, and the only clause
    here that a no-PML test can never catch. ``Fields.drive_field``
    (fields.py:1140-1163) returns ``f_w_<c>`` while the absorber is on and the
    STORED E while it is off, and the two agree EXACTLY outside the layer -- so
    binding E under an active layer passes every no-PML case and is wrong only
    where it matters. The sibling track measured that as a control and it
    DIVERGED (``B_ade_wrong_drive_CONTROL``, 3072/15360 words differing;
    ``triton_kernels/no_pml_ade.py`` carries the record).

    So the check is an IDENTITY check, not an existence check: the array
    ``drive_field`` actually hands back must BE ``f_w_<c>``. A predicate that only
    pinned ``f_w_<c>`` allocated would still admit a ``Fields`` whose storage mode
    disagreed with the layer, which is the state that produces the wrong pointer.
    Transcribed from the Metal predicate's ``_drive_reasons``
    (``metal_kernels/ade_update_p.py:150-181``), which asks the same question the
    same way.
    """
    expected_name = "f_w_" + component
    expected = getattr(fields, expected_name, None)
    if expected is None:
        return f"{expected_name} is not allocated (the drive field under PML)"
    if not getattr(fields, "_pml_active", False):
        return (f"the layer is active but Fields is not in PML storage mode; "
                f"drive_field would hand back the stored {component}, which is "
                f"the constitutive product only OUTSIDE the absorber")
    reader = getattr(fields, "drive_field", None)
    if not callable(reader):
        return "fields does not expose drive_field()"
    try:
        actual = reader(component)
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return f"drive_field({component!r}) raised {exc!r}"
    if actual is not expected:
        return (f"drive_field({component!r}) is not {expected_name}; the kernel "
                f"binds what drive_field returns and this run would drive P from "
                f"the wrong array")
    return None


def _ade_coefficient_problem(state: Any) -> Any:
    """Why the ``(c_now, c_prev, c_drive)`` triple is not three finite floats.

    The kernel takes them as scalar arguments, baked from ``dt`` by
    ``susceptibility.coefficients`` (dispersion.py:643). A NaN or an infinity here
    is a volume of NaN after one step, so the read is checked rather than assumed.
    ``float('nan') != float('nan')`` is the finiteness test, written out because
    this module imports no ``math`` and no ``numpy``.
    """
    coefficients = getattr(state, "_coefficients", None)
    try:
        values = tuple(coefficients)
    except Exception as exc:  # noqa: BLE001 - malformed is a refusal
        return f"the coefficient triple is unreadable ({type(exc).__name__}: {exc})"
    if len(values) != 3:
        return (f"the (c_now, c_prev, c_drive) triple has {len(values)} entries, "
                f"not 3")
    for name, value in zip(("c_now", "c_prev", "c_drive"), values):
        try:
            number = float(value)
        except Exception:  # noqa: BLE001 - a non-numeric coefficient is not coverage
            return f"{name}={value!r} is not a float"
        if number != number:
            return f"{name} is NaN"
        if number in (float("inf"), float("-inf")):
            return f"{name}={number!r} is not finite"
    return None


def covers_real_pml_ade_component(fields: Any, pml: Any, grid: Any,
                                  state: Any, component: str) -> tuple:
    """Whether ``update_P_pml_real*`` may advance ONE (state, component).

    Returns ``(covered, reason)`` naming the FIRST refusal, the convention the
    three predicates above use.

    THE SUB-STEP. ``dispersion.PolarizationState.update`` (dispersion.py:658-691),
    whose array form is three passes over one scratch buffer::

        xp.multiply(p, c_now, out=scratch)      # scratch = P^n * c_now
        scratch += c_prev * p_prev              #         + c_prev * P^(n-1)
        scratch += c_drive * (sigma * w)        #         + c_drive * (sigma * W^n)

    WHAT THIS ADMITS THAT THE SIBLINGS REFUSE, and the line behind each:

    * **a mirror fold.** ``stepping.update_P`` runs the whole stored array with no
      ownership mask (stepping.py:1398-1403) because the fold repair happens
      upstream, on D, before ``update_E`` (``fill_symmetry_bc_D``). Worth 5 of the
      corpus's 15 ``update_P`` rows -- see :data:`ADE_MIRROR_ADMISSION`.
    * **a conductivity**, on the sibling constitutive predicate's own measured
      grounds: ``fields.condfac_for`` is read in ``stepping._apply_curl``
      (stepping.py:508) and nowhere else.
    * **an off-diagonal chi1inv row.** The coupling lives in ``update_E``'s
      ``elif offdiagonal:`` branch (stepping.py:1001-1008) and reaches P only
      through the ``f_w`` that branch wrote. This kernel binds that array; it does
      not form the product.

    WHAT IT REFUSES THAT THE SIBLINGS DO NOT NEED TO NAME:

    * **a susceptibility kind outside** :data:`COVERED_SUSCEPTIBILITY_KINDS`. The
      three-term recurrence IS the Lorentz/Drude one; another kind is a different
      difference equation wearing the same coefficient triple.
    * **a component this state does not drive.** ``PolarizationState`` allocates
      ``P``/``P_prev`` for driven components only (dispersion.py:645-647), so an
      undriven component has no buffer to advance and asking for one is a
      KeyError inside the launcher.

    AND WHAT IS REFUSED FAIL-CLOSED RATHER THAN FOR A MEASURED REASON: a
    cylindrical axis. The arithmetic is element-wise there too, by the same line,
    and this predicate refuses it anyway because no hand-CUDA family has ever been
    measured on a Dcyl run at any sub-step and the corpus drives ZERO such rows
    here -- so the refusal costs nothing measured and the admission would rest on
    an argument alone. That asymmetry with the mirror clause is deliberate: one
    has 5 rows and a transcribed line, the other has 0 rows and a transcribed
    line.
    """
    xp, backend = _backend(grid)
    if backend != "cupy":
        return False, "backend is not CuPy"
    if getattr(fields, "force_complex_fields", False):
        # ``Fields._field_dtype`` (fields.py:571-573) reads this attribute ALONE,
        # so it is the whole storage question -- but every buffer is checked
        # float32 below regardless, because a predicate that inferred the dtype
        # from a flag would follow that flag if it ever stopped deciding.
        return False, ("complex64 storage: the recurrence is the same but the "
                       "storage is not")
    if not (pml is not None and getattr(pml, "is_active", False)):
        # NOT a numerical clause. ``stepping.update_P`` deletes its ``pml``
        # argument (stepping.py:1424) and the body is identical either way; what
        # changes is WHICH ARRAY ``drive_field`` returns. Without a layer that is
        # the stored E, and admitting it here would put two different pointer
        # bindings behind one verdict. The sibling track split exactly this into
        # its own family for the same reason (``triton_kernels/no_pml_ade.py``),
        # and a hand-CUDA no-PML family would be the same kernel with one clause
        # inverted.
        return False, ("no active PML layer: this family binds f_w as the drive "
                       "and drive_field returns the stored E without one")

    if component not in ADE_ELECTRIC_COMPONENTS:
        return False, (f"component {component!r} is outside "
                       f"{ADE_ELECTRIC_COMPONENTS}")
    drives = getattr(state, "drives", None)
    try:
        if not (callable(drives) and drives(component)):
            return False, f"this susceptibility does not drive {component}"
    except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
        return False, f"drives({component!r}) raised {exc!r}"
    kind = getattr(getattr(state, "susceptibility", None), "kind", None)
    if kind not in COVERED_SUSCEPTIBILITY_KINDS:
        return False, (f"kind {kind!r} is outside "
                       f"{COVERED_SUSCEPTIBILITY_KINDS}")
    problem = _ade_coefficient_problem(state)
    if problem is not None:
        return False, problem

    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    if facts["cylindrical"]:
        return False, ("cylindrical (Dcyl): element-wise like every other extent, "
                       "and refused because no hand-CUDA family has been measured "
                       "on one at any sub-step")
    for axis in range(3):
        if facts["axis"][axis]:
            return False, f"axis {axis} is the cylindrical r = 0 axis"
    for axis, boundary in enumerate(_boundary_kinds_from(facts)):
        # A fold IS on this list, unlike the siblings' -- see the docstring. The
        # clause stays anyway so that a ghost rule added to Grid later is refused
        # because it was never admitted.
        if boundary not in ADE_BOUNDARY_KINDS:
            return False, (f"axis {axis} resolves to boundary {boundary!r}, which "
                           f"this sub-step has never been measured on")
    if facts["has_bloch"]:
        # Costs nothing measured: every corpus row with a Bloch k also carries
        # complex storage and is refused above. Named anyway, because
        # ``_field_dtype`` reads ``force_complex_fields`` alone and an edit that
        # ever let a Bloch run store real would land here silently.
        return False, ("nonzero Bloch k: the wrapped plane carries a phase real "
                       "storage cannot hold")
    if facts["bfast_active"]:
        return False, "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return False, "special_kz (grid.beta != 0): extra out-of-plane coupling terms"
    if getattr(fields, "_chi2_components", None) or getattr(fields, "_chi3_components", None):
        # Both maps by name, not ``Fields.has_nonlinearity`` (fields.py:966-967),
        # which reads ``_chi2_components`` alone -- a chi3-only run would pass the
        # property. The Pade factor scales what ``update_E`` stores into f_w, so it
        # reaches P through the drive; unmeasured here, so refused.
        return False, ("instantaneous chi2/chi3: a Pade factor replaces the "
                       "constitutive product this sub-step's drive comes from")

    shape = facts["shape"]
    if len(shape) != 3:
        return False, f"grid shape {shape} is not three-dimensional"
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    if cells >= 2 ** 31:
        return False, f"{cells} cells exceeds the kernel's int32 index range"

    # The three rotating buffers. ``_scratch`` is the OUTPUT of this launch and
    # the shared one: ``PolarizationState.update`` moves it from component to
    # component inside a single call (dispersion.py:689-691), so it is checked
    # once per component rather than once per state.
    buffers = {
        f"P[{component!r}]": (getattr(state, "P", {}) or {}).get(component),
        f"P_prev[{component!r}]": (getattr(state, "P_prev", {}) or {}).get(component),
        "_scratch": getattr(state, "_scratch", None),
    }
    for label, array in buffers.items():
        problem = _array_problem(label, array, xp, shape)
        if problem is not None:
            return False, problem
    addresses = {}
    for label, array in buffers.items():
        address = _base_address(array)
        if address is None:
            return False, (f"{label} exposes no readable base address; the "
                           f"rotation cannot be shown to be alias-free")
        if address in addresses:
            return False, (f"{label} aliases {addresses[address]}; the recurrence "
                           f"reads P and P_prev while writing the scratch, and the "
                           f"rotation that follows would advance one buffer twice")
        addresses[address] = label

    problem = _ade_drive_problem(fields, component)
    if problem is not None:
        return False, problem
    drive = getattr(fields, "f_w_" + component, None)
    problem = _array_problem(f"f_w_{component}", drive, xp, shape)
    if problem is not None:
        return False, problem
    drive_address = _base_address(drive)
    if drive_address is not None and drive_address in addresses:
        return False, (f"f_w_{component} aliases {addresses[drive_address]}; the "
                       f"drive is read while the scratch is written")

    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    if sigma is None:
        return False, f"sigma[{component!r}] is missing"
    if ade_sigma_is_volume(state, component):
        problem = _array_problem(f"sigma[{component!r}]", sigma, xp, shape)
        if problem is not None:
            return False, problem
        # Every pointer the device signature declares carries ``__restrict__``,
        # and this is the clause that pays for the last one. A promise the caller
        # cannot keep is worse than no promise: it licenses the compiler to keep a
        # stale value in a register across the store.
        sigma_address = _base_address(sigma)
        if sigma_address is not None and sigma_address in addresses:
            return False, (f"sigma[{component!r}] aliases {addresses[sigma_address]}; "
                           f"the coefficient volume is read while the scratch is "
                           f"written")
    else:
        try:
            number = float(sigma)
        except Exception:  # noqa: BLE001 - neither a scalar nor a volume
            return False, (f"sigma[{component!r}]={sigma!r} is neither a scalar "
                           f"nor a volume")
        if number != number or number in (float("inf"), float("-inf")):
            return False, f"sigma[{component!r}]={number!r} is not finite"
    return True, "covered"


def covers_real_pml_ade_update_p(fields: Any, pml: Any, grid: Any) -> tuple:
    """Whether the hand-CUDA ADE family may serve the WHOLE ``update_P`` sub-step.

    ALL OR NOTHING OVER EVERY STATE AND EVERY DRIVEN COMPONENT, and that is a
    property of the sub-step rather than a conservatism.
    ``PolarizationState.update`` rotates ONE SHARED SCRATCH buffer from component
    to component inside a single call (dispersion.py:689-691): this step's result
    lands in ``_scratch``, ``_scratch`` takes over the array that held ``P_prev``,
    and the retired history becomes the NEXT component's scratch. Covering two of
    three components and leaving the third to the array path would interleave two
    rotations over one buffer set. Sub-steps compose; halves of one sub-step do
    not. The sibling builder refuses the same way and says the same thing
    (``triton_kernels/launch.py:1841-1858``).

    A run with no polarization at all is refused rather than trivially covered:
    ``stepping.update_P`` returns immediately (stepping.py:1425-1426), so there is
    no sub-step to take over and a "covered" verdict would put a launch where the
    array path does nothing.
    """
    try:
        states = tuple(getattr(fields, "polarizations", ()) or ())
    except Exception as exc:  # noqa: BLE001 - unreadable is not empty
        return False, f"the polarization list is unreadable ({exc!r})"
    if not states:
        return False, "no polarization is registered; update_P is a no-op"
    driven_total = 0
    for index, state in enumerate(states):
        driven = getattr(state, "driven", None)
        try:
            components = tuple(driven()) if callable(driven) else ()
        except Exception as exc:  # noqa: BLE001 - malformed is a refusal
            return False, f"polarization {index}: driven() raised {exc!r}"
        driven_total += len(components)
        for component in components:
            covered, reason = covers_real_pml_ade_component(
                fields, pml, grid, state, component)
            if not covered:
                return False, f"polarization {index} {component}: {reason}"
    if driven_total == 0:
        return False, "no driven component exists; update_P is a no-op"
    return True, "covered"


# ===========================================================================
# THE COMPLEX-STORAGE FAMILY
# ===========================================================================
#
# ``complex_pml_kernels`` / ``complex_emitter``: complex64 storage under a real
# split-field PML, all four sub-steps. The two predicates below are the written
# refusal for what those four kernels must NOT serve.
#
# THE STORAGE CLAUSE IS INVERTED HERE, and that inversion is the whole point.
# :func:`covers_real_pml_curl` and :func:`covers_real_pml_constitutive` refuse
# complex storage by name; these two REQUIRE it. The two sets are a PARTITION of
# ``force_complex_fields or grid.has_bloch`` -- no configuration can be admitted by
# both and none can fall between them -- so the loss mode a second admitter would
# cause (a slot admitted by two families and taken over by neither, or worse by
# both) cannot arise. The Triton sibling makes the same partition on the same
# clause and says so at ``_complex_grid_reasons``' ``require_active_pml``.
#
# WHY REAL STORAGE READ AS COMPLEX IS A WRONG ANSWER RATHER THAN A CRASH: the
# kernels address ``2*idx``/``2*idx+1``, so a float32 volume handed in would be
# stepped as half a volume of made-up pairs, silently, at full speed.

#: The float32 subnormal policy names this family's licence clause compares
#: against. Restated rather than imported (this module imports nothing) and pinned
#: against ``subnormal_policy``'s own spellings by ``test_complex_pml.py``.
COMPLEX_POLICY_NAMES: Tuple[str, ...] = ("keep", "flush")

#: The arms ``complex_emitter`` can emit, by name. A verdict naming anything else
#: is refused rather than defaulted: both arms compile and run, and they differ in
#: the last bits of about a quarter of the words, so a wrong arm is a wrong answer.
COMPLEX_EXPANSION_ARMS: Tuple[str, ...] = ("NAIVE", "FMA_V1")

#: The bases :func:`complex_expansion_refusal` accepts. ``measured`` is a licence
#: this run's probe earned; ``environment_default`` is one taken from
#: ``triton_kernels.complex_fields.ENVIRONMENT_DEFAULTS`` because the policy in
#: force erased the lanes that separate the arms. Both are licences; anything else
#: is not.
COMPLEX_EXPANSION_BASES: Tuple[str, ...] = ("measured", "environment_default")


def complex_expansion_refusal(license: Any, subnormal_policy: Any) -> Any:
    """Why this expansion-licence verdict may not bind an arm here; None if it may.

    THIS IS NOT THE ARBITER AND DELIBERATELY DOES NOT RE-DERIVE ONE. The rule that
    turns a probe artifact into an arm --- which patterns must classify, which
    verdicts refuse, when ``AMBIGUOUS_BOTH`` is an exclusion rather than a veto,
    when the environment table may be consulted --- is
    ``triton_kernels.complex_fields.expansion_license``, and a second spelling of
    it is one too many (that module merged two drifted copies of exactly this
    clause on 2026-08-15 and wrote down what the drift cost). What is checked here
    is that the caller HAS a verdict, that the verdict is refusal-free, that it
    names an arm this family can emit, and that it was cut under the policy this
    run installs.

    THE POLICY COMPARISON IS HERE AND NOT ONLY THERE because this is the last rung
    before the arm is compiled into a binary. ``expansion_license`` computes its
    verdict from the record ALONE -- it cannot know which policy the reading
    process runs under, and says so -- so a flush-cut licence and a keep run agree
    on every field of the verdict. The comparison is one equality between two
    strings, not a licensing rule; the rule it enforces is the one
    ``POLICY_CONDITIONAL_LICENCE`` states: an arm classified under one subnormal
    policy reproduces the platform's bytes under that policy only.

    ``subnormal_policy`` must be a name. ``None`` REFUSES here, unlike the Triton
    ``expansion_policy_reasons`` where ``None`` means "artifact-reading tooling is
    not asking": a caller of this function IS asking, because the only reason to
    ask a coverage predicate is to decide whether to launch.
    """
    if license is None:
        return ("no expansion licence: the complex multiply's arm is a measured "
                "platform fact and may not be guessed")
    if not isinstance(license, dict):
        return (f"the expansion licence is {type(license).__name__}, not the "
                f"verdict dict expansion_license returns")
    refusals = license.get("refusals")
    if refusals:
        first = refusals[0] if isinstance(refusals, (list, tuple)) else refusals
        return f"the expansion licence refuses: {first}"
    arm = license.get("arm")
    if arm not in COMPLEX_EXPANSION_ARMS:
        return (f"the expansion licence names arm {arm!r}, which is not one of "
                f"{list(COMPLEX_EXPANSION_ARMS)}")
    if license.get("expansion") is None:
        return "the expansion licence carries no arm code"
    basis = license.get("basis")
    if basis not in COMPLEX_EXPANSION_BASES:
        return (f"the expansion licence has basis {basis!r}, which is not one of "
                f"{list(COMPLEX_EXPANSION_BASES)}")
    if not isinstance(subnormal_policy, str):
        return ("the float32 subnormal policy this run will use was not given as "
                "a name, so the licence cannot be shown to have been cut under "
                "it; an expansion licence is POLICY-CONDITIONAL")
    resolved = license.get("policy_resolved")
    if resolved is None:
        return ("the expansion licence states no resolved subnormal policy, so it "
                "cannot be shown to have been cut under the policy this run "
                "requires")
    if resolved != subnormal_policy:
        return (f"the expansion licence was cut under the {resolved!r} float32 "
                f"subnormal policy and this run requires {subnormal_policy!r}; a "
                f"licence does not transfer across that boundary")
    return None


#: The ghost rules the COMPLEX constitutive pair serves. ``CYL_AXIS`` is on it and
#: is NOT in :data:`BC_CODES`, which is the deliberate divergence from
#: :func:`covers_real_pml_complex_curl` and the one clause of that predicate with a
#: DEVICE VERDICT rather than a reading behind it --- see
#: :data:`COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION`. The curl keeps
#: ``BC_CODES``: its emitted source takes ``bc_x``/``bc_y``/``bc_z`` as runtime
#: arguments and there is no code for the axis to pass, which is the same fact
#: from the other side.
#: The ghost rules the COMPLEX constitutive pair serves. MIRROR is in it and is
#: NOT in the curl's ``BC_CODES``, which is the same deliberate divergence the
#: real family makes: the constitutive template takes NO boundary argument at all
#: (``complex_pml_kernels.update_fused_pml_complex``, :427-470, passes the three
#: volume triples, the three shapes and the six coefficient vectors and nothing
#: else), so the kind it never reads cannot be the wrong one -- while the emitted
#: CURL takes ``bc_x/bc_y/bc_z`` as runtime arguments and has no code for a
#: mirror. Neither kind is admitted by argument: see
#: :data:`COMPLEX_CONSTITUTIVE_FOLD_ADMISSION` and
#: :data:`COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION`.
COMPLEX_CONSTITUTIVE_BOUNDARY_KINDS: Tuple[str, ...] = (PERIODIC, METALLIC,
                                                        MIRROR, CYL_AXIS)

#: WHAT THE COMPLEX FOLD CLAUSE IS WORTH AND WHAT PAID FOR IT.
#:
#: Until 2026-08-20 ``_complex_grid_refusal`` refused every mirror-folded run for
#: BOTH complex sub-step families with "mirror symmetry: different ghost rule, a
#: parity mask and two fill passes". FOR THE CURL THAT REFUSAL IS REAL AND STAYS:
#: ``complex_emitter``'s ``cshift_up``/``cshift_dn`` are a neighbour stencil, a
#: fold changes the ghost rule and the ownership mask at once, and the emitted
#: curl takes ``bc_x/bc_y/bc_z`` with no code for a mirror. The CONSTITUTIVE pair
#: inherited it through the one function the two predicates share, and inheritance
#: is not evidence -- the same shape the Dcyl clause had before
#: :data:`COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION`.
#:
#: THE READING. ``update_fused_pml_complex`` (complex_pml_kernels.py:427-470)
#: takes NO boundary codes, NO phase pair, NO ``dtdx`` and NO mask -- compare its
#: sibling ``curl_fused_pml_complex`` (:380-425), which passes all of them.
#: ``stepping._apply_constitutive_pml`` (stepping.py:2112-2145) reads no
#: neighbour, and ``_mask_non_owned_cells`` (:1912) has exactly three call sites,
#: none of them constitutive. ``PML._compute_coefficients`` (pml.py:693-695,
#: :706-708) builds every kps/kms vector from the STORED extent the fold halves
#: (grid.py:570-575), and the complex volumes are allocated on the same shape, so
#: both operands move together.
#:
#: THAT READING WAS NOT TRUSTED, AND THE REAL PAIR'S FOLD VERDICT WAS NOT
#: INHERITED EITHER (:data:`CONSTITUTIVE_FOLD_ADMISSION` is different kernel bytes
#: -- complex64 word pairs, the zero cross terms, an expansion arm the real pair
#: does not take). ``parity/meep_gpu/gate_cuda_complex_folded_constitutive.py``
#: ran the SHIPPED complex pair, unchanged, on mirror-folded complex64 grids: 180
#: cases per policy of which 156 folded, 180/180 bit-identical to
#: ``stepping.update_H``/``update_E`` at one launch AND at 60, under BOTH float32
#: subnormal policies on an RTX A6000.
#:
#: THE FLOORS THAT MAKE IT NON-VACUOUS. Every case moved words (92%-100% of the
#: output words changed from the frozen input); the folded axis's coefficient
#: profile was required to differ from the identity, or reversing it -- the
#: refusal's own premise armed -- would be a no-op; the ``signed_zero`` class
#: carried 129666 negative-zero words on each leg and the ``subnormal_band`` class
#: 486068 (keep) / 27992 (flush) subnormal words, so the zero cross terms and the
#: policy both had something to act on.
#:
#: WHAT WAS ARMED. Fourteen legs, every one as required.
#: ``reverse_folded_axis_coefficients`` -- the folded axis's kps/kms read
#: backwards IN BOUNDS -- CAUGHT on every folded leg and SILENT on every unfolded
#: control, where it is a no-op and must be. ``column_major_index`` (the
#: coefficient-index defect planted in the device text, which the unequal stored
#: extents 9/10/11 are what make visible) CAUGHT. The two float2 half-plane store
#: defects CAUGHT. Two NULL legs CONFIRMED. And the RELEASE VERDICT ITSELF was
#: shown to flip: a run with ``half_plane_auxiliary_store`` planted for the whole
#: sweep came out ``released=False`` with 24/24 cases diverging.
#:
#: WHAT THE SWEEP DID NOT REACH. A folded axis carrying a nonzero Bloch k
#: component: MEEP forces a mirror plane's own axis to k = 0 and all eight corpus
#: rows record it, so the sweep put its phases on UNFOLDED axes (24 phased folded
#: cases) and the predicate refuses the other case BY NAME below. It costs 0
#: slots. More than three simultaneous planes is not a configuration a 3-D grid
#: has.
#:
#: WHAT IT IS WORTH, and the arithmetic matters because the fold clause SHORT-
#: CIRCUITS and hides what is behind it. The 2026-08-20 closeout census carries
#: EIGHT complex, PML-active, folded rows, all 32 of their slots unserved. Only
#: the 16 constitutive slots are this pair's; of those, three rows carry
#: ``grid.beta != 0`` and are refused for special_kz on every sub-step (6 slots),
#: and three carry an off-diagonal epsilon whose ``update_E`` is refused by a
#: clause of its own (3 more). 16 - 6 - 3 = 7. NOT ONE of the eight becomes a row
#: the hand-CUDA track can step end to end, because their curls stay refused.
COMPLEX_CONSTITUTIVE_FOLD_ADMISSION: dict = {
    "gate": "parity/meep_gpu/gate_cuda_complex_folded_constitutive.py",
    "artifacts": ("parity/meep_gpu/results/"
                  "cuda_complex_folded_constitutive_2026-08-26"),
    #: ``results/`` is gitignored, so the digests are how a reader binds this claim
    #: to the bytes that produced it on a checkout that does not carry them.
    "artifact_sha256": {
        "keep/gate.json":
            "9978ce0baad7dc9c968b6e9c0fcaf4c6c4988210c16d6fc0ce09f4140256bf6b",
        "flush/gate.json":
            "0bbb4ca004db77714fc3fa3317f5b2de16c748eb4b20ed105c14f00976a496d5",
        "falsify/gate.json":
            "547ff8398ca31448192e62ae50f596a55ffe7d035f7cd874411c9dd9ac2213ae",
    },
    #: THE SUBJECT: the bytes whose behaviour was measured. Not one of them was
    #: changed by this round -- the gate compiles the SHIPPED emitter through the
    #: SHIPPED launcher and mutates only a copy of the text.
    #:
    #: RE-RUN 2026-08-26 AND RE-POINTED. ``complex_pml_kernels.py`` moved when its four
    #: Bloch kernels were RECORDED -- their names left ``UNCERTIFIED_KERNELS`` for
    #: ``CERTIFIED_KERNELS`` and ``certification.json`` gained the
    #: ``complex_pml_2026-08-26`` block. That edit touches two declaration constants
    #: and comments and reaches no device string (the module's twelve ``*_code``
    #: literals are byte-identical across it), but this record carries no
    #: ``device_sha256`` for ``weld_survives_edit`` to establish that from, so the byte
    #: rule stands and the gate was RE-RUN rather than the digest bumped. All three
    #: legs reproduce: keep and flush released, and the planted
    #: ``half_plane_auxiliary_store`` still drives ``released`` FALSE, so the verdict is
    #: still shown to flip. ``complex_emitter.py`` did NOT move and its digest is
    #: unchanged below.
    "subject_sha256": {
        "meep_gpu/cuda_kernels/complex_emitter.py":
            "cbe0e0f3f8cbd7cdfeb6a5e00716c0e6facab73e4804b2ae7f082a2f02759e1f",
        "meep_gpu/cuda_kernels/complex_pml_kernels.py":
            "cfa1db011eebffbc1b5ccbef861b1296dfe31ceba356299b12cefcf08170fdca",
    },
    #: THE CODE IDENTITY OF THOSE SAME BYTES (2026-09-25), for weld_survives_edit's
    #: second tier: neither subject carries a literal device string, so what the gate
    #: measured is everything in them except comments and docstrings. Both digests are
    #: code_identity.code_digest of the certified bytes above (commit 03dc4b7,
    #: sha256 cbe0e0f3… / cfa1db01…), and both equal the digest of the tree after
    #: the 2026-09-25 citation re-point, which moved only line citations in prose. An
    #: edit that reaches code in either file now fails this record by name.
    "code_sha256": {
        "meep_gpu/cuda_kernels/complex_emitter.py":
            "bb60fc538c055f8ecdd04140f101ec04ecf42afc9562976810493f8443fe2999",
        "meep_gpu/cuda_kernels/complex_pml_kernels.py":
            "1a0d3df11a5b730e7b3d0e22efdadf17de9dc55380f9a2cbd23b5bc195fcb184",
    },
    "recorded_utc": "2026-08-27T04:07:11Z",
    "host": "the GPU host, NVIDIA RTX A6000 (sm_86), CuPy 13.5.1",
    "cases_per_policy": 180,
    "folded_cases_per_policy": 156,
    "identical_single_launch": 180,
    "identical_at_sixty_launches": 180,
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    "expansion_arm": "FMA_V1",
    "expansion_arm_basis": "measured",
    "folded_axes_swept": ("x", "y", "z"),
    "folded_terminations_swept": ("metallic", "periodic"),
    "plane_parities_swept": (1, -1),
    "full_count_parities_swept": ("even", "odd"),
    "folded_planes_swept": 3,
    "phased_folded_cases": 24,
    "value_classes_swept": ("uniform", "subnormal_band", "signed_zero"),
    "negative_zero_words_in_signed_zero_cases": 129666,
    "subnormal_words_in_band_cases": {"keep": 486068, "flush": 27992},
    "mutation_legs": 14,
    "mutation_legs_as_required": 14,
    "verdict_shown_to_flip": "half_plane_auxiliary_store, released=False, 24/24 "
                             "diverging",
    "phased_folded_axis_swept": False,
    # RECOMPUTED FROM THE CENSUS with the union analyzer, on scripts copied
    # byte-identical from the 2026-08-20 closeout round so the two rounds are
    # commensurable to the slot. SEVEN slots on FIVE of the eight rows -- five
    # update_H and two update_E -- which is the arithmetic the header derives
    # before the run and the census then confirmed: 16 constitutive slots, minus
    # 6 on the three beta != 0 rows, minus 3 on the three off-diagonal ones.
    #
    # AND NOT ONE ROW BECAME FULLY COVERED. Rows covered at every sub-step went
    # 136 -> 139 over the whole round, and all three are the other two widenings'
    # (TestLDOS.test_ldos_3D and the two nonlinear rows). The eight folded complex
    # rows keep their sixteen refused curl slots, which is the honest half of this
    # record.
    "census": ("parity/meep_gpu/results/"
               "cuda_predicate_coverage_2026-08-20_w15"),
    "census_before": ("parity/meep_gpu/results/"
                      "cuda_predicate_coverage_2026-08-20_closeout"),
    "union_slots_before": 653,
    "union_slots_after": 668,
    "slots_this_admission_gained": 7,
    "rows_touched": 5,
    "rows_gained_end_to_end": 0,
    "rows_covered_at_every_sub_step_before": 136,
    "rows_covered_at_every_sub_step_after": 139,
    "what_it_does_not_license": (
        "the COMPLEX CURL on a fold, which is refused for a real reason: "
        "cshift_up/cshift_dn are a neighbour stencil, a fold changes the ghost "
        "rule and the ownership mask, and the emitted curl takes bc_x/bc_y/bc_z "
        "with no code for a mirror. All sixteen curl slots on the eight corpus "
        "rows stay refused.",
        "a folded axis carrying a nonzero Bloch k component: refused by name, "
        "not swept, and worth 0 slots because MEEP forces a mirror plane's own "
        "axis to k = 0.",
        "the REAL constitutive pair's fold, which is a different kernel and a "
        "different gate.",
        "any dispatch. No module in meep_gpu imports cuda_kernels at all, so a "
        "widened predicate licenses a MEASUREMENT and not a production step.",
    ),
}

#: WHAT THE COMPLEX Dcyl CLAUSE IS WORTH AND WHAT PAID FOR IT.
#:
#: Until 2026-08-20 ``_complex_grid_refusal`` refused every Dcyl run for BOTH
#: complex sub-step families with "cylindrical (Dcyl): prefix-sum radial derivative
#: and axis-row rules". FOR THE CURL THAT REFUSAL IS REAL AND STAYS:
#: ``complex_emitter``'s ``cshift_up``/``cshift_dn`` are a neighbour stencil, and
#: Dcyl replaces the radial derivative with a prefix sum and adds the axis-row
#: rules. The CONSTITUTIVE pair inherited it through the one function the two
#: predicates share, and inheritance is not evidence.
#:
#: THE READING. ``stepping._apply_constitutive_pml`` (stepping.py:2112-2145) reads
#: NO NEIGHBOUR -- ``fw_previous = fw.copy(); fw[...] = source; field += kps*fw;
#: field -= kms*fw_previous``, four whole-array operations with no shift, no ghost
#: and no mask. ``update_H`` (:907-925) and ``update_E`` (:926-995) contain zero
#: occurrences of "cylindrical", "is_axis", "m" or "axis_zero"; the Dcyl repairs
#: ``_cylindrical_axis_zero_B``/``_D`` are called at :376 and :457, inside the CURL
#: and nowhere else; ``_mask_non_owned_cells`` (:1865) has exactly three call sites
#: and no constitutive function is among them. The emitted kernel says the same in
#: its own characters: ``complex_emitter._CONSTITUTIVE_TEMPLATE`` takes no ``bc_*``,
#: no ``ph_*``, no phase pair and no ``dtdx``, so there is no place in it for a
#: ghost rule to be wrong and the only thing Dcyl could still move is the
#: coefficient INDEX.
#:
#: THAT READING WAS NOT TRUSTED, AND THE REAL PAIR'S VERDICT WAS NOT INHERITED
#: EITHER (:data:`CONSTITUTIVE_CYLINDRICAL_ADMISSION` is different kernel bytes --
#: complex64 word pairs, the zero cross terms, an expansion arm the real pair does
#: not take -- and this project does not transfer a verdict across kernels).
#: ``parity/meep_gpu/gate_cuda_complex_cylindrical_constitutive.py`` ran the
#: SHIPPED complex pair, unchanged, on complex64 Dcyl grids: 144 cases per policy
#: of which 120 Dcyl, 144/144 bit-identical to ``stepping.update_H``/``update_E``
#: at one launch AND at 60, under BOTH float32 subnormal policies on an RTX A6000.
#:
#: THE FLOORS THAT MAKE IT NON-VACUOUS. Every case moved words (the array path
#: changed 82%-100% of the output words from the frozen input); the radial axis's
#: max |coefficient - 1| ranged 0.580 to 1.499, so a coefficient-index error on
#: the axis Dcyl moved was visible rather than hidden behind a table of ones; the
#: ``signed_zero`` class carried 34262 (keep) / 34217 (flush) negative-zero words
#: and the ``subnormal_band`` class 127432 (keep) / 7215 (flush) subnormal words,
#: so the zero cross terms and the policy both had something to act on. The two
#: policies were shown to be really different rather than really installed: the
#: NVRTC observer saw 55 compiles on each leg with ``any_ftz_true_reached_nvrtc``
#: FALSE under keep and TRUE under flush.
#:
#: WHAT WAS ARMED. Fourteen legs, every one as required. ``reverse_radial_axis_
#: coefficients`` -- THE REFUSAL'S OWN PREMISE armed as a defect, the r axis's
#: kps/kms read backwards in bounds -- CAUGHT on 30/30 Dcyl legs and silent on all
#: 6 Cartesian ones, where it is a no-op and must be. ``column_major_index``
#: (the coefficient-index defect planted in the device text) CAUGHT 36/36.
#: ``half_plane_field_store`` and ``half_plane_auxiliary_store`` -- this family's
#: own float2 defect class, a store that writes one plane and leaves the other --
#: CAUGHT 36/36 each. ``fold_the_zero_cross_terms`` and ``plane_wise_scaling``
#: CAUGHT on every case in their reach. Two NULL legs (an identity store-reload and
#: a commuted plane sum) CONFIRMED 0/36. And the RELEASE VERDICT ITSELF was shown
#: to flip: a run with ``half_plane_auxiliary_store`` planted for the whole sweep
#: came out ``released=False`` with 24/24 cases diverging on 32256 of 129024 words
#: -- exactly the auxiliary's imaginary plane.
#:
#: WHAT THE SWEEP DID NOT REACH, and is therefore still refused: a PHASED Dcyl
#: grid. Dcyl Bloch is z-only in MEEP (grid.py:654-657), no corpus row carries one
#: (all sixteen record ``has_bloch`` FALSE) and the constitutive kernel takes no
#: phase argument -- but unmeasured is unmeasured, so the pairing is refused by
#: name below rather than admitted by argument. It costs 0 slots.
#:
#: THE SLOT COUNTS are re-derived from the census record's per-row
#: ``configuration`` blocks by running THIS predicate over stand-ins built from
#: them (``parity/meep_gpu/replay_complex_cylindrical_slots.py``); the replay is
#: validated first by reproducing every recorded verdict and first-refusal string
#: that is not the retired Dcyl one, and by leaving every non-Dcyl row's verdict
#: unchanged -- which is what makes this a widening and not a leak.
#:
#: THEY ARE AN UPPER BOUND: the complex-volume and coefficient-vector tail cannot
#: be replayed from the record, only the decision logic above it.
COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION: dict = {
    "gate": "parity/meep_gpu/gate_cuda_complex_cylindrical_constitutive.py",
    # THE ``b`` ROUND IS THE ONE THIS RECORD NAMES, and the reason is provenance
    # rather than a better result: the first round (``..._2026-08-20/``, kept, not
    # overwritten) ran gate bytes that a later docstring cleanup moved, so its
    # artifact records a gate digest the repository no longer holds. The gate was
    # re-run unchanged afterwards and reproduced the verdict case for case, so the
    # record binds to the round whose ``imported_source_sha256`` for the gate IS
    # the shipped file. The two rounds agree on every summary figure.
    "artifacts": ("parity/meep_gpu/results/"
                  "cuda_complex_cylindrical_constitutive_2026-08-20b"),
    "superseded_first_round": ("parity/meep_gpu/results/"
                               "cuda_complex_cylindrical_constitutive_2026-08-20"),
    # ``results/`` is gitignored, so the digests are how a reader binds this claim
    # to the bytes that produced it on a checkout that does not carry them.
    "artifact_sha256": {
        "keep/gate.json":
            "27d883bc9beedd5f1d68834271096ca64886fee58c7020eabf082f6a3fc48c8c",
        "flush/gate.json":
            "f945f0fa64210863755c7b32543a509e7b5ee69a9d858f5b53f27859ace5bf33",
        "falsify/gate.json":
            "843e63fa2f53547c7a204322f0ac9f8aba1358e4a9f9daa7b8b0f42d48fa896b",
    },
    "gate_sha256_that_ran":
        "30a974b896e9df3963627c2c1069704fe8bdd44a4ddcfb488c53c35368c4c4b2",
    # THE SUBJECT: the bytes whose behaviour was measured. Not one of them was
    # changed by this round -- the gate compiles the SHIPPED emitter through the
    # SHIPPED launcher and mutates only a copy of the text.
    "subject_sha256": {
        "meep_gpu/cuda_kernels/complex_emitter.py":
            "657c8ee850602f7a7d12213642a277ad2d22cb3fdf438aba054471d547c6fd98",
        "meep_gpu/cuda_kernels/complex_pml_kernels.py":
            "dd2dbee2b907e3fbcedbbc6f1fa0914cab8737777242b47048afebdca1d6dfd2",
    },
    "recorded_utc": "2026-08-20T13:15:00Z",
    "host": "the GPU host",
    "device": "NVIDIA RTX A6000 (sm_86), CuPy 13.5.1",
    "cases_per_policy": 144,
    "dcyl_cases_per_policy": 120,
    "identical_single_launch": 144,
    "identical_at_sixty_launches": 144,
    "policies": ("ieee_keep_ftz_stripped", "meep_x86_flush"),
    "expansion_arm": "FMA_V1",
    "expansion_arm_basis": "measured",
    "m_values_swept": (-1, 0, 1, 2, 3, 5),
    "m_classes_swept": ("m0", "m1", "m2plus"),
    "z_terminations_swept": ("metallic", "periodic"),
    "value_classes_swept": ("uniform", "subnormal_band", "signed_zero"),
    "mutation_legs": 14,
    "mutation_legs_as_required": 14,
    "verdict_shown_to_flip": "half_plane_auxiliary_store, released=False",
    "phased_dcyl_swept": False,
    "census": ("parity/meep_gpu/results/"
               "cuda_predicate_coverage_2026-08-20_all_families"),
    "replay": "parity/meep_gpu/replay_complex_cylindrical_slots.py",
    # RECOMPUTED with the replay and then RE-MEASURED by running the census's own
    # union analyzer positionally over a patched copy of the record, so the before
    # and after numbers come from one instrument rather than from a tool's own
    # arithmetic. The replay reproduced all 372 recorded complex constitutive
    # verdicts and first-refusal strings, moved no NON-cylindrical row, and moved
    # no curl slot at all. All 32 gained slots were refused by the Dcyl clause AND
    # BY NOTHING ELSE, on the 16 rows that are Dcyl, PML-active and complex
    # storage, and none of them was already served by another family -- so this is
    # a widening and not a leak, and the analyzer still reports the arms disjoint.
    "slots_before": 557,
    "slots_after": 589,
    "slots_gained": 32,
    "rows_gained": 16,
    "denominator": 759,
    # UNCHANGED, and that is the honest half: the complex CURL still refuses all
    # sixteen rows at step_B and step_D, so not one of them becomes a row the
    # hand-CUDA track can step end to end.
    "rows_covered_at_every_sub_step_before": 110,
    "rows_covered_at_every_sub_step_after": 110,
    "patched_census": ("parity/meep_gpu/results/"
                       "cuda_complex_cylindrical_constitutive_2026-08-20/"
                       "census_after"),
    "what_it_does_not_license": (
        "the COMPLEX Dcyl CURL, which is refused for a real reason: cshift_up / "
        "cshift_dn are a neighbour stencil and Dcyl replaces the radial "
        "derivative with a prefix sum and adds the axis-row rules.",
        "a PHASED Dcyl grid on either sub-step: not swept, refused by name.",
        "any dispatch. No module in meep_gpu imports cuda_kernels at all, so a "
        "widened predicate licenses a MEASUREMENT and not a production step.",
    ),
}


def _complex_grid_refusal(fields: Any, pml: Any, grid: Any,
                          admit_cylindrical: bool = False,
                          admit_fold: bool = False) -> Any:
    """The clauses both complex predicates share, or None.

    ONE COPY, TWO CALLERS, for the reason the Triton sibling gives at
    ``_complex_grid_reasons``: a duplicated copy of the phase logic is two things
    that must be kept in step forever.

    ``admit_cylindrical`` AND ``admit_fold`` ARE THE TWO PLACES THE CALLERS
    DIVERGE, and each is a parameter rather than a fork because each divergence is
    exactly one clause -- and the SAME clause, read from the two sides of one
    fact: the curl reads NEIGHBOURS and the constitutive pair reads none. Dcyl
    replaces the radial derivative with a prefix sum; a fold changes the ghost
    rule and the ownership mask. Neither reaches a sub-step whose emitted template
    takes no boundary argument at all.

    PASSING EITHER ONE TRUE IS LICENSED BY A DEVICE VERDICT AND BY NOTHING ELSE
    (:data:`COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION`,
    :data:`COMPLEX_CONSTITUTIVE_FOLD_ADMISSION`). BOTH DEFAULT TO THE REFUSAL, so
    a new caller that has not been measured inherits the "no" -- which is what
    keeps ``covers_real_pml_complex_curl`` refusing both.

    WHY THE PHASE CLAUSE IS SHARED WITH A SUB-STEP THAT CARRIES NO PHASE. The
    constitutive kernels take no phase argument and never see the wrap
    (stepping.py:907-993), so on the face of it the phase consistency check is not
    theirs. It is shared anyway because it is not a statement about the kernel: an
    axis carrying a phase that does not resolve PERIODIC is a configuration
    ``stepping._bloch_phases`` RAISES on (stepping.py:2401-2415), so the ARRAY PATH
    does not step it either. A predicate that admitted it would be admitting a run
    that does not exist.
    """
    xp, backend = _backend(grid)
    if backend != "cupy":
        return "backend is not CuPy"
    # THE INVERTED CLAUSE. Either force_complex_fields or a nonzero k_point puts a
    # run here; the array path itself raises on real storage with a phase
    # (stepping.py:1893-1908), so the two together are the whole complex class.
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        return ("real float32 storage (neither force_complex_fields nor a nonzero "
                "k_point): a real run belongs to the shipped real-field kernels")
    if not (pml is not None and getattr(pml, "is_active", False)):
        return "no active PML layer"
    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return unreadable
    # A FOLD IS REFUSED FOR THE CURL AND ADMITTED FOR THE CONSTITUTIVE PAIR, and
    # the asymmetry is a device verdict rather than an oversight in either
    # direction. The curl's refusal is REAL: cshift_up/cshift_dn are a neighbour
    # stencil, a fold changes the ghost rule and the ownership mask at once, and
    # the emitted curl takes bc_x/bc_y/bc_z with no code for a mirror. The
    # constitutive pair inherited that refusal through THIS function until
    # 2026-08-20, and the inheritance was measured rather than argued away: see
    # :data:`COMPLEX_CONSTITUTIVE_FOLD_ADMISSION` (180/180 per policy on 156
    # folded cases, both terminations, all three axes, up to three simultaneous
    # planes, the refusal's own premise armed as a defect and CAUGHT, and the
    # release verdict itself shown to flip).
    folded = [axis for axis in range(3) if facts["mirrored"][axis]]
    if facts["has_symmetry"] and not folded:
        # FAIL CLOSED: has_symmetry with no mirrored axis is a ghost rule this
        # module has never resolved. Same clause, same reason, same wording as
        # :func:`covers_real_pml_curl`'s.
        return "symmetry with no folded axis: unresolved ghost rule"
    if folded and not admit_fold:
        return "mirror symmetry: different ghost rule, a parity mask and two fill passes"
    if len(folded) > COMPLEX_CONSTITUTIVE_FOLD_ADMISSION["folded_planes_swept"]:
        return (f"{len(folded)} mirror planes at once: the fold gate swept "
                f"{COMPLEX_CONSTITUTIVE_FOLD_ADMISSION['folded_planes_swept']} "
                f"simultaneous planes, so a {len(folded)}-plane fold is an "
                f"extrapolation from the measurement rather than part of it")
    # A Dcyl GRID IS REFUSED FOR THE CURL AND ADMITTED FOR THE CONSTITUTIVE PAIR.
    # The curl's refusal is REAL -- cshift_up/cshift_dn are a neighbour stencil and
    # Dcyl replaces the radial derivative with a prefix sum and adds the axis-row
    # rules. The constitutive pair inherited it through THIS function, and the
    # inheritance was measured rather than argued away: see
    # :data:`COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION` (144/144 per policy on
    # the GPU host, 120 Dcyl cases, both float32 subnormal policies, the refusal's own
    # premise armed as a defect and CAUGHT, and the release verdict itself shown
    # to flip).
    if facts["cylindrical"] and not admit_cylindrical:
        return "cylindrical (Dcyl): prefix-sum radial derivative and axis-row rules"
    if facts["cylindrical"] and facts["has_bloch"]:
        # NOT SWEPT, so refused by name rather than admitted by argument. Dcyl
        # Bloch is z-only in MEEP (grid.py:654-657), no corpus row carries one
        # (all sixteen Dcyl complex rows record has_bloch FALSE) and the
        # constitutive kernel takes no phase argument at all -- but "the kernel
        # cannot see it" is a reading, and this clause costs 0 slots.
        return ("a Bloch phase on a cylindrical grid: the Dcyl gate swept only "
                "unphased grids, which is every Dcyl row the corpus carries")
    for axis in range(3):
        if facts["mirrored"][axis] and not admit_fold:
            return f"axis {axis} is mirrored"
        if facts["axis"][axis] and not facts["cylindrical"]:
            # FAIL-CLOSED, and NOT inherited: a grid whose axis flag is set
            # without the cylindrical flag disagrees with itself, and neither
            # caller can answer for one. Reached on both settings of
            # ``admit_cylindrical`` deliberately.
            return f"axis {axis} is the cylindrical r = 0 axis"
        if facts["axis"][axis] and not admit_cylindrical:
            return f"axis {axis} is the cylindrical r = 0 axis"
    kinds = _boundary_kinds_from(facts)
    # THE CURL KEEPS ``BC_CODES``, which is the same fact from the other side: its
    # emitted source takes bc_x/bc_y/bc_z as runtime arguments and there is no code
    # for the axis to pass. The constitutive template takes no boundary argument at
    # all, so the kind it never reads cannot be the wrong one.
    admitted_kinds = (COMPLEX_CONSTITUTIVE_BOUNDARY_KINDS if admit_cylindrical
                      else tuple(BC_CODES))
    for axis, kind in enumerate(kinds):
        if kind not in admitted_kinds:
            return f"axis {axis} resolves to boundary {kind!r}, which has no kernel"
    # PER-AXIS PHASE CONSISTENCY. A phased axis must resolve PERIODIC, and a
    # metallic axis must carry a k component of exactly 0. ``has_bloch`` FALSE IS
    # ADMITTED: with every phase flag 0 the kernel emits no rotation at all and
    # must reduce to the plain complex path, which is what keeps the complex
    # storage-at-k=0 scripts (wvg-src.py, solve-cw.py) inside this family.
    #
    # THE PHASE IS READ DEFENSIVELY rather than through getattr-with-default: an
    # unreadable phase is not an unphased one, and admitting one would let the
    # launcher's ``bloch_phase_arguments`` raise where a fail-closed "no" was the
    # right answer.
    reader = getattr(grid, "bloch_phase", None)
    if not callable(reader):
        return ("grid.bloch_phase is missing or not callable; an unreadable phase "
                "table is not an unphased one")
    try:
        k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        return f"grid.k_point could not be read: {type(exc).__name__}: {exc}"
    if len(k_point) != 3:
        return f"grid.k_point has {len(k_point)} components, not three"
    for axis, kind in enumerate(kinds):
        try:
            phase = reader(axis)
        except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
            return (f"grid.bloch_phase({axis}) raised {type(exc).__name__}: {exc}; "
                    f"an unreadable phase is not an unphased one")
        if phase is not None and kind != PERIODIC:
            return (f"axis {axis} carries Bloch phase {phase!r} but resolved to "
                    f"{kind!r}; only a periodic wrap can carry a phase")
        if kind == METALLIC and float(k_point[axis]) != 0.0:
            return (f"axis {axis} is metallic with k component {k_point[axis]!r}; "
                    f"a PEC wall gives the axis no lattice vector for the phase")
        if kind == MIRROR and float(k_point[axis]) != 0.0:
            # FAIL-CLOSED AND MEASURED UNREACHABLE FROM A REAL GRID, which is the
            # only honest way to write a clause nothing can drive.
            # ``Grid._resolve_bloch`` (grid.py:975) RAISES on this combination --
            # "mirror symmetry on the Y axis requires k_point component y = 0" --
            # so no ``Grid`` in this engine can present one, all eight folded
            # complex corpus rows carry k = 0 on their folded axis, and the fold
            # gate accordingly put its phases on UNFOLDED axes (24 phased folded
            # cases). The clause costs 0 slots and exists so that a future Grid
            # that stopped raising is refused here rather than admitted by an
            # invariant nobody re-checked.
            return (f"axis {axis} is mirror-folded with k component "
                    f"{k_point[axis]!r}; the fold gate swept phases on unfolded "
                    f"axes only, which is every folded complex row the corpus "
                    f"carries")
    if facts["bfast_active"]:
        return "BFAST: a second additive term on every curl target"
    if facts["beta"] != 0.0:
        return "special_kz (grid.beta != 0): extra out-of-plane coupling terms"
    # DISPERSION IS REFUSED OUTRIGHT ON BOTH SUB-STEP FAMILIES, unlike the real
    # track where the curl admits it. There it is admitted because ``step_B``
    # differences the STORED E that ``update_E`` wrote, and dispersion changes only
    # the values in it. That reasoning holds here too -- but complex-storage ADE is
    # not built on any track, none of the nine demand scripts carries a
    # susceptibility, and a curl admitted into a step whose ``update_E`` and
    # ``update_P`` are both refused buys a sub-step nothing can compose with. The
    # Triton sibling refuses the same way and for the same reason.
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", None) or None):
        return ("a susceptibility is registered: complex-storage ADE is a future "
                "tranche and no corpus script combines dispersion with complex "
                "fields")
    if getattr(fields, "_chi2_components", None) or getattr(fields, "_chi3_components", None):
        return "instantaneous chi2/chi3: a Pade factor replaces the constitutive product"
    if not getattr(fields, "stores_E", False):
        return "E is recomputed from D rather than stored"
    shape = facts["shape"]
    if len(shape) != 3:
        return f"grid shape {shape} is not three-dimensional"
    # THE INT32 BOUND IS HALVED against the real path's. The kernels address
    # ``2*idx`` and ``2*idx+1`` in ``int``, so ``2*ncells`` is what must stay below
    # 2**31, not ``ncells``. Past it the arithmetic stops being an identity and
    # starts being a wrong answer at a wrapped word index -- a scrambled volume,
    # not a launch failure.
    cells = int(shape[0]) * int(shape[1]) * int(shape[2])
    if 2 * cells >= 2 ** 31:
        return (f"{cells} cells is {2 * cells} float32 words, which exceeds the "
                f"kernels' int32 word-offset range (2*ncells must stay below 2**31)")
    return None


def _complex_volume_problem(label: str, array: Any, xp: Any, shape: Tuple) -> Any:
    """Why ``array`` is not a complex64 C-contiguous volume of ``shape``; None if it is.

    The complex twin of :func:`_array_problem`. C-CONTIGUITY IS THE LOAD-BEARING
    CLAUSE: the kernels index the float32 word view flatly, so a strided complex
    view would be read in the wrong order -- a scrambled volume rather than a
    launch failure.
    """
    try:
        if array is None:
            return f"{label} is not allocated"
        if array.dtype != xp.complex64:
            return (f"{label} is {array.dtype}, not complex64 (this family steps "
                    f"complex storage as float32 word pairs)")
        if tuple(array.shape) != shape:
            return f"{label} has shape {tuple(array.shape)}, not the grid's {shape}"
        if not array.flags.c_contiguous:
            return f"{label} is not C-contiguous"
    except Exception as exc:  # noqa: BLE001
        return f"{label} could not be inspected: {type(exc).__name__}: {exc}"
    return None


def covers_real_pml_complex_curl(fields: Any, pml: Any, grid: Any, sub_step: str,
                                 license: Any = None,
                                 subnormal_policy: Any = None) -> tuple:
    """Whether ``step_B_pml_complex_bloch`` / ``..._D_...`` may serve this run.

    ``sub_step`` is ``"step_B"`` or ``"step_D"``. ``license`` is the verdict
    ``triton_kernels.complex_fields.expansion_license`` returns and
    ``subnormal_policy`` the policy name this run installs; both are REQUIRED in
    practice, because the arm is compiled into the binary at this seam and there is
    no later rung to check it at. Returns ``(covered, reason)``.

    THE SUB-STEP ARGUMENT IS REQUIRED, for the reason
    :func:`covers_real_pml_curl`'s is: there are two kernels, and one predicate
    answering for both can only return the INTERSECTION of two clause sets. The
    live per-sub-step clause is the conductivity -- ``stepping._apply_curl`` reads
    ``fields.condfac_for(term.target)`` per TERM (stepping.py:508), so a D
    conductivity routes ``step_D`` to the three-history recurrence and leaves
    ``step_B`` an ordinary curl.

    A CONDUCTIVITY IS REFUSED ON THE SUB-STEP THAT CARRIES IT and admitted on the
    other, exactly as the real pair does. Complex conductive stepping is a separate
    product (stepping.py:2001-2109) with no corpus demand, and this family does not
    implement it.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {sorted(CURL_SUB_STEPS)}, got {sub_step!r}")
    refusal = complex_expansion_refusal(license, subnormal_policy)
    if refusal is not None:
        return False, refusal
    refusal = _complex_grid_refusal(fields, pml, grid)
    if refusal is not None:
        return False, refusal
    xp, _ = _backend(grid)
    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    shape = facts["shape"]
    # THIS sub-step's targets only. A missing or non-callable reader is refused
    # OUTRIGHT rather than falling back to a magnetic flag: inferring "no
    # conductivity" from the ABSENCE of ``condfac_for`` is admission by attribute
    # absence, which is the exact reasoning coverage exists to refuse.
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        return False, ("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
    for component in CURL_SUB_STEPS[sub_step]:
        try:
            conductive = reader(component) is not None
        except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
            return False, (f"condfac_for({component!r}) raised "
                           f"{type(exc).__name__}: {exc}")
        if conductive:
            return False, (f"{component} carries a conductivity: routes to the "
                           f"three-history conductive-PML recurrence")
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
                 "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz",
                 "fu_Dx", "fu_Dy", "fu_Dz"):
        problem = _complex_volume_problem(name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem
    # BOTH sub-lattices, as the real curl predicate checks: the suffix the launcher
    # binds is the sub-step's own, and a swap is the half-cell host mutation.
    for axis, name in enumerate(("x", "y", "z")):
        for half in (True, False):
            suffix = "_h" if half else ""
            for label in ("kms", "sinv"):
                attribute = f"{label}_{name}{suffix}"
                problem = _coefficient_vector_problem(
                    attribute, getattr(pml, attribute, None), xp, axis, shape)
                if problem is not None:
                    return False, problem
    return True, "covered"


def covers_real_pml_complex_constitutive(fields: Any, pml: Any, grid: Any,
                                         side: str, license: Any = None,
                                         subnormal_policy: Any = None) -> tuple:
    """Whether ``update_H_pml_complex_bloch`` / ``..._E_...`` may serve this run.

    ``side`` is ``"H"`` (``stepping.update_H``) or ``"E"`` (``stepping.update_E``).

    WHAT THIS ADMITS THAT :func:`covers_real_pml_complex_curl` REFUSES: a Dcyl
    GRID and a MIRROR FOLD. Both of the curl's refusals are real -- ``cshift_up``/
    ``cshift_dn`` are a neighbour stencil, Dcyl replaces the radial derivative
    with a prefix sum, and a fold changes the ghost rule and the ownership mask --
    and this pair inherited both through the one function the two predicates
    share, because its own emitted template
    (``complex_pml_kernels.update_fused_pml_complex``, :427-470) takes no boundary
    codes, no phase pair and no mask at all. Each inheritance was MEASURED rather
    than argued away: :data:`COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION` (144/144
    per policy on 120 Dcyl cases) and
    :data:`COMPLEX_CONSTITUTIVE_FOLD_ADMISSION` (180/180 per policy on 156 folded
    cases, both terminations, all three axes, up to three simultaneous planes),
    each with the floors that made it non-vacuous, the refusal's own premise armed
    as a defect and caught, and the one configuration the sweep did not reach
    named and refused (a PHASED Dcyl grid; a phase on the FOLDED axis -- each
    worth 0 slots).

    WHAT THE E SIDE REFUSES THAT THE H SIDE DOES NOT: everything that changes what
    ``source`` is. A registered polarization is already refused for the whole
    family, and an off-diagonal ``chi1inv`` row makes the sub-step non-element-wise
    (stepping.py:1001-1008) -- refused HERE and admitted by the curl, which is the
    same per-sub-step split the real track makes.

    A CONDUCTIVITY IS ADMITTED ON BOTH SIDES. ``fields.condfac_for`` is read in
    ``stepping._apply_curl`` (stepping.py:508) and nowhere else in the module;
    ``update_H``, ``update_E`` and ``_apply_constitutive_pml`` never mention it. So
    a conductive complex run routes the curl that WRITES the carrying component to
    a recurrence this family does not implement, and leaves both constitutive
    sub-steps ordinary products. Refusing them here would be a refusal with no line
    behind it.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(
            f"side must be one of {sorted(CONSTITUTIVE_SIDES)}, got {side!r}")
    spec = CONSTITUTIVE_SIDES[side]
    refusal = complex_expansion_refusal(license, subnormal_policy)
    if refusal is not None:
        return False, refusal
    # ``admit_cylindrical=True`` and ``admit_fold=True`` ARE THE TWO ARGUMENTS
    # THAT DIFFER FROM THE CURL'S CALL, and what stands behind each is a device
    # verdict rather than a reading:
    # :data:`COMPLEX_CONSTITUTIVE_CYLINDRICAL_ADMISSION` and
    # :data:`COMPLEX_CONSTITUTIVE_FOLD_ADMISSION`.
    refusal = _complex_grid_refusal(fields, pml, grid, admit_cylindrical=True,
                                    admit_fold=True)
    if refusal is not None:
        return False, refusal
    xp, _ = _backend(grid)
    facts, unreadable = _grid_facts(grid)
    if unreadable is not None:
        return False, unreadable
    shape = facts["shape"]
    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        return False, ("off-diagonal chi1inv: the row product reads the other "
                       "components' volumes and this sub-step is element-wise")
    for name in tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"]):
        problem = _complex_volume_problem(name, getattr(fields, name, None), xp, shape)
        if problem is not None:
            return False, problem
    if side == "E":
        # THREE volumes, one per component, and they stay FLOAT32 under complex
        # storage -- stepping.py:41-50 states it and fields.py:1203-1204 is where
        # it is decided. The kernel indexes them by the COMPLEX CELL index, never
        # word-doubled, which is what makes the float32 pin exactly right rather
        # than a leftover from the real family.
        reader = getattr(fields, "inverse_epsilon_for", None)
        if not callable(reader):
            return False, "fields does not expose inverse_epsilon_for"
        for component in _ELECTRIC_COMPONENTS:
            try:
                volume = reader(component)
            except Exception as exc:  # noqa: BLE001 - a raise is not a refusal unless caught
                return False, f"inverse_epsilon_for({component!r}) raised {exc!r}"
            if volume is None:
                return False, f"inverse_epsilon_for({component!r}) is None"
            if not getattr(volume, "shape", ()):
                return False, f"inverse_epsilon_for({component!r}) is a scalar, not a volume"
            problem = _array_problem(f"inverse_epsilon_for({component!r})",
                                     volume, xp, shape)
            if problem is not None:
                return False, problem
    # This side's own sub-lattice only -- integer for H, half-integer for E.
    suffix = "_h" if spec["half_integer"] else ""
    for axis, name in enumerate(("x", "y", "z")):
        for label in ("kps", "kms"):
            attribute = f"{label}_{name}{suffix}"
            problem = _coefficient_vector_problem(
                attribute, getattr(pml, attribute, None), xp, axis, shape)
            if problem is not None:
                return False, problem
    return True, "covered"
