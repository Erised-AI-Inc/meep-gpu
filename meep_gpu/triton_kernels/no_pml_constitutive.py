"""The NO-PML constitutive family: a predicate over a sub-step that does nothing.

THE FINDING, STATED FIRST, BECAUSE IT IS THE WHOLE POINT OF THE FILE. This family
needs NO NEW KERNEL, and on its load-bearing arm it needs no kernel AT ALL — not a
restated predicate over an already-certified kernel body (which is what
``special_kz.beta_run_constitutive_coverage`` is), but a predicate over a sub-step
whose array-path implementation is a bare ``return``. There is no arithmetic to
transcribe, no grouping to hold, no coefficient to index and no float to round.
The gate leg that certifies it is an IDENTITY leg — every live array byte-unchanged
across the array path's own call — and it runs to completion on a laptop with no
CUDA, no Triton and no CuPy, because nothing here compiles or launches anything.

WHAT ``stepping`` ACTUALLY DOES WITHOUT AN ABSORBER
--------------------------------------------------
``stepping.update_H`` (stepping.py:907-923)::

    if not _pml_is_active(pml):
        return                                  # stepping.py:944-945

Unconditional, for every grid, every boundary, every storage width. H is never
stored without PML — ``Fields.enable_field_storage`` says so in its own docstring
(fields.py:664-666: "H is deliberately NOT allocated: ``update_H`` writes nothing
without PML, so a stored H would sit at zero for the whole run while ``get_H``
returned it") — and ``Fields.get_H`` serves the B array itself, mu = 1.

``stepping.update_E`` (stepping.py:926-993)::

    pml_active = _pml_is_active(pml)            # stepping.py:982
    if not pml_active and not fields.stores_E:
        return                                  # stepping.py:983-984
    ...
    else:
        getattr(fields, component)[...] = constitutive   # stepping.py:1022

so the sub-step has TWO no-PML shapes, and the ``return`` at :954 comes before every
branch that could read a polarization, a chi2/chi3 Pade factor, an off-diagonal row,
a BFAST fold or a beta coupling. Nothing downstream of that line can make the call
non-null; the only question a predicate has to answer is which side of :954 the
configuration falls on.

``_pml_is_active`` (stepping.py:2498-2506) is ``pml is not None and pml.is_active``,
and an all-zero-face layer is INACTIVE — so a ``PML(thickness=0)`` object present in
the driver takes this path, exactly as ``no_pml.plain_curl_coverage`` clause 3 has it
for the curl. That shared test is what makes the two families' coverage sets a
partition rather than an overlap.

THE TWO ARMS
------------
* **Arm N (NULL) — built here.** ``update_H`` on any inactive layer, and ``update_E``
  on an inactive layer with ``fields.stores_E`` False. Zero kernels, zero launches.
  :class:`NullConstitutivePlan` is what the composition puts in the slot.
* **Arm S (STORE) — specified here, BUILT ELSEWHERE as of 2026-08-16.** ``update_E``
  on an inactive layer with ``stores_E`` True: ``E[...] = (D - sum P) * inv_eps``, a
  pointwise store. It lives in ``no_pml_stored_e.py``, which is where
  :data:`STORED_E_ARM`'s ``compose_from`` said it would have to live ("a NEW module
  beside ``dispersive_update_e`` — not a widening of this one. This file is about
  the absence of arithmetic and must not grow any"), and that rule still holds: not
  one line of arithmetic entered this file with it.

**Arm S is NOT ``kernels.constitutive_step`` under a restated predicate, and that was
measured rather than assumed.** The certified body is an ACCUMULATION that also
writes ``f_w``::

    prev = w[i]; w[i] = src; f[i] = (f[i] + kps*src) - kms*prev

Binding ``kps = 1``, ``kms = 0`` and a zeroed ``f`` does not recover ``f[i] = src``:
``0.0 + (-0.0)`` is ``+0.0``, so every negative zero in the source is canonicalised
away (measured on this laptop, NumPy float32: the four-word vector
``[-0.0, 0.0, 1.5, -2.5]`` stores as ``0x80000000 0x00000000 0x3fc00000 0xc0200000``
and accumulates as ``0x00000000 ...`` — 1 word of 4 differing). ``kms * prev`` is
also ``0.0 * prev``, which is NaN for a non-finite ``prev``. And ``f`` is NOT zero in
a real run: it holds the previous step's E. The bodies genuinely differ, so Arm S
would be a new kernel — composed, not written from scratch, from
``dispersive_update_e``'s body with the PML tail replaced by a store and the ``f_w``
write DELETED (without PML ``Fields.drive_field`` hands ``update_P`` the stored E,
not ``f_w``, fields.py:1140-1162, and ``f_w_*`` is not even allocated).

MEASURED DEMAND, AND WHERE IT LANDS
-----------------------------------
``results/predicate_coverage_2026-08-12`` measures 186 corpus rows. 14 carry no PML,
and every one of the 14 loses BOTH constitutive sub-steps to
``coverage.constitutive_coverage``'s clause 3 ("no active PML layer"). Split by
``stores_E``:

* ``stores_E`` False — 5 rows: ``TestAbsorber.test_absorber_2d``,
  ``TestEigenModeSource.test_amp_func_change_sources``,
  ``TestMaterialDispersion.test_material_dispersion_with_user_material``,
  ``TestMedium.test_check_material_frequencies``,
  ``TestSimulation.test_harminv_warnings``. Arm N covers BOTH sub-steps on all five.
  Four of them (all but ``test_absorber_2d``, which carries a conductivity the curl
  predicate refuses) already hold ``plain_curl@step_B`` and ``plain_curl@step_D``,
  so Arm N is exactly what takes them WHOLE-STEP. That is the row signature the
  analysis prints as ``4  no-PML  unadmitted: update_H, update_E``.
* ``stores_E`` True — 9 rows. Arm N still covers ``update_H`` on all nine (H is
  never stored without PML). ``update_E`` on those nine is Arm S, and counted
  honestly it closes NO whole-step row:

  - 6 force complex storage (``TestLoadDump`` x4, ``TestMaterialGrid`` x2) and are
    outside a real-f32 product; 2 of those also carry off-diagonal rows;
  - 3 are real-f32 and reachable in principle — ``TestAbsorber.test_absorber``,
    ``absorber-1d.py``, ``material-dispersion.py``. A conductivity does NOT disturb
    the constitutive sub-step (``coverage._grid_reasons`` says so and keeps it per
    curl), so the first two really are in Arm S's envelope — but their ``step_B``
    and ``step_D`` are refused by the conductive-no-PML gap, so Arm S would add one
    sub-step to a step whose curl is still on the array path;
  - all three additionally lose ``update_P``, because ``ade_update_p_coverage``
    requires ``f_w_<component>`` and no no-PML run allocates it.

  So the kernel would serve three rows, take none of them whole-step, and one of
  the three is a single voxel. That is why Arm S was specified and not built.

  **THE THIRD BULLET STOPPED BEING TRUE ON 2026-08-16 AND WITH IT THE DECISION.**
  ``no_pml_ade`` inverted exactly that ``f_w`` clause and the wiring round gave
  ``update_P`` to exactly those three rows, so the sentence "all three additionally
  lose ``update_P``" is now false about all three. Re-measured against
  ``results/predicate_coverage_2026-08-16_wired_convention/``: Arm S takes
  ``material-dispersion.py`` WHOLE-STEP on its own, and is one of the three arms
  that take ``TestAbsorber.test_absorber`` and ``absorber-1d.py`` whole-step. The
  live count is in :data:`STORED_E_ARM`; this paragraph is kept as the record of
  what the old count was and what invalidated it, because a demand figure that
  quietly changes value is how a costed decision outlives its cost.

WHAT THIS PREDICATE DELIBERATELY DOES **NOT** REFUSE
----------------------------------------------------
Every kernel predicate in this package refuses the fold, the cylindrical axis,
complex storage, a Bloch phase, beta and BFAST. Those clauses exist because a kernel
INDEXES something — ``coverage._grid_reasons`` clause 5 says it plainly, "a folded
axis changes n_a and therefore every cell's coefficient index". A null indexes
nothing. Carrying those clauses here would refuse ``update_H`` on the six complex
no-PML rows and the two off-diagonal ones for a hazard that provably cannot exist,
and this package's rule is that a refusal must be TRUE where it fires
(``no_pml.py`` clause 3b was rewritten for exactly that reason). So they are absent,
by name, and the gate's ``breadth`` leg measures the claim on a folded, a
cylindrical, a complex and a Bloch grid rather than leaving it as an argument.

The refusals that remain are the ones a wrong answer could actually come through,
and they are enumerated positively in :func:`null_constitutive_coverage`.

AND THIS FILE CARRIED THE SAME DEFECT IT CITES, which is why the paragraph above
now has a table behind it. Two things were repaired on 2026-08-13, both measured:

* the chi2/chi3 refusal asserted that the nonlinearity "forces stored E
  (driver.py:2022)". True of ``driver.set_chi2``/``set_chi3``; FALSE of
  ``Fields.set_nonlinear_volumes``, on which ``has_nonlinearity`` is True,
  ``stores_E`` is False, ``stepping.update_E`` returns at :954 and moved 0 of 10080
  words. A right verdict with a false reason is still a defect in a reasons list;
* the file claimed ONE conservative clause. There are FOUR, and
  :data:`CONSERVATIVE_CLAUSES` names each with the words its own needle moved (all
  zero) — while the off-diagonal clause, which the file grouped with them, is
  genuinely vacuous (4758 words moved: the null would have been WRONG there).

The two clauses the family rests on — ``pml.is_active`` and ``fields.stores_E`` —
also stopped defaulting to the ADMITTING answer when the object cannot be read. A
``Fields`` proxy whose ``stores_E`` raised was covered with no reasons while the run
underneath it wrote 4618 words; see :func:`_flag`.

DISPATCH IS NOT WIRED — but the PLANNER now is, and this paragraph said otherwise
for a round. ``fastpath.plan_fast_path`` still returns ``None`` on every branch and
nothing here changes that, so no production step can reach this family.
``launch.plan_step`` DOES consult it: the ``no-PML null`` arm sits last in both
constitutive tables behind the ``absorber_inactive`` gate, and ``__init__`` exports
the two entry points. See ``WIRING`` at the bottom for what that seam is and what
it deliberately still leaves alone.

THE RECORD, and what each artifact is for:

* ``meep_gpu/test_triton_no_pml_constitutive.py`` — the merge bar. Carries BOTH
  halves here (there is no arithmetic a laptop cannot hold): the predicate clause by
  clause, and the identity property against the real ``stepping`` functions over
  multiple full cycles with its failing controls.
* ``parity/meep_gpu/gate_triton_no_pml_constitutive.py`` — the byte gate, and its
  own release contract (``validate_payload``), which the suite mutation-tests.
* ``parity/meep_gpu/probe_triton_no_pml_constitutive_composition.py`` — the
  whole-step question: does this family actually CLOSE a row? Its ``census`` and
  ``constitutive`` legs run anywhere; its ``whole_step`` leg needs a device and
  records itself as skipped, with blockers, when it does not have one.
* ``parity/meep_gpu/run_triton_no_pml_constitutive_direct.sh`` — the device leg, RUN
  2026-08-13 on the GPU host, pinned with ``CUDA_VISIBLE_DEVICES`` to a GPU verified
  physically empty before and after. Slurm is not the mechanism here and that is
  measured rather than preferential: with ``--gres`` the scheduler hands out an index
  already carrying a foreign process, and without it the isolation strips every GPU.
  The ``.slurm`` file beside it is kept as the record of that route.

WHAT THE DEVICE LEG ADDED, AND WHAT IT DID NOT. It re-ran the whole gate under
``--backend cupy`` on real device arrays (6/6 legs, 0 differing words) and ran the
composition probe's ``whole_step`` leg, which is the only leg in the family that needs
a GPU: ``step_B``/``step_D`` on ``no_pml.plan_plain_curl`` beside these two nulls, so
all four heavy slots leave the array path at once. All 5 rows stayed bit-identical
over 24 complete driver steps with all four slots substituted. It did NOT certify a
kernel — there is none — and it did not raise the curl's step budget.

THE DEVICE HOST ALSO EXPOSED TWO DEFECTS THE LAPTOP STRUCTURALLY COULD NOT, both now
closed and pinned:

* CUDA context initialisation calls ``setlocale`` and resets ``LC_CTYPE`` to ASCII
  (measured: preferred encoding ``UTF-8`` before the first device allocation,
  ``ANSI_X3.4-1968`` after). The gate read its own source with a locale-dependent
  ``open``, so the MUTATION BATTERY could not run on any host with a GPU — the CuPy
  gate died at that leg, and the suite failed one test after the first to build a
  device driver. Fixed with an explicit ``encoding="utf-8"``; pinned on a laptop with
  ``PYTHONCOERCECLOCALE=0 LC_ALL=C``, which reproduces the state without a GPU;
* the subnormal-policy stamp was taken BEFORE the first leg, so ``nvrtc_calls`` and
  ``ftz_removed`` were frozen at zero in every artifact — and this file cited that
  zero as confirmation that the family compiles nothing. It could not have read
  anything else. The family's PRODUCT does compile nothing; the gate's harness is
  ordinary CuPy work and left 48 ``.cubin`` files beside the stamp claiming none. The
  stamp is now taken again after the legs, and the certified run reports 44 NVRTC
  compiles with all 44 stripped of ``-ftz=true``.

AND A THIRD, WHICH IS THE ONE THAT MATTERED MOST, because it is this family's
characteristic failure rather than a portability accident. A NULL certifies by
agreeing with a no-op, so "the two drivers are bit-identical" is satisfied just as
well by a run in which NOTHING MOVED. Neither byte-identity leg recorded whether the
step it wrapped had done any work. Adding that measurement — the ARRAY-PATH driver's
own state before the first step against after the last, as a PASS CONDITION — caught a
row that had been passing: ``material_dispersion_user_material_1voxel`` moved 0 of its
6 words across all 24 steps, because on a single periodic voxel every curl difference
is a cell against itself and that row carried no source. Its "all four heavy slots
substituted, bit-identical" was comparing two frozen states. The row now carries a
source (Dz moves, 1 of 6 words — the structural maximum for one voxel) and is
labelled with what it can and cannot show. The rows that carry the real claim move
every word of Bx/By/Bz/Dx/Dy/Dz.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .coverage import (  # READ-ONLY imports; nothing here mutates coverage.py
    ELECTRIC_COMPONENTS,
    MAGNETIC_COMPONENTS,
    Coverage,
    _call,
)

__all__ = [
    "CONSERVATIVE_CLAUSES",
    "NULL_SIDES",
    "STORED_E_ARM",
    "NullConstitutivePlan",
    "NullConstitutiveStepPlan",
    "null_constitutive_coverage",
    "plan_null_constitutive",
    "plan_null_constitutive_step",
    "stored_e_constitutive_reasons",
]

#: The two constitutive slots and the ``stepping`` line whose ``return`` makes each
#: one null. Keyed as ``coverage.CONSTITUTIVE_SIDES`` is, so the two tables can be
#: read side by side and a composer can pass one ``side`` string to either — but the
#: key set is spelled out here rather than imported, because the VALUES are
#: deliberately not array names: this family reads and writes no array, and importing
#: the other table would suggest it does. The gate's ``overlap`` leg pins the two key
#: sets against each other so they cannot drift.
NULL_SIDES: Dict[str, Dict[str, Any]] = {
    "H": {
        "sub_step": "update_H",
        "return_site": "stepping.py:944-945",
        # H is never stored without PML, so there is no second arm on this side.
        "needs_stored_e_clause": False,
    },
    "E": {
        "sub_step": "update_E",
        "return_site": "stepping.py:983-984",
        # With stores_E the sub-step writes E and this family refuses; see Arm S.
        "needs_stored_e_clause": True,
    },
}

#: Arm S — the STORE arm, transcribed, costed, and BUILT as of 2026-08-16.
#:
#: This is a record, not a plan: nothing in this module consumes it. It exists so the
#: next reader does not have to re-derive the product from ``stepping`` to decide
#: whether it is worth a kernel, and so the decision about it is auditable.
#:
#: THE DECISION IT ENCODED WAS REVERSED, AND BY A MEASUREMENT RATHER THAN A
#: PREFERENCE. ``measured_demand`` used to end "ZERO additional whole-step rows. Not
#: built on that evidence", and the evidence stopped holding the day
#: ``no_pml_ade``'s admission was wired: the three rows it counted as losing
#: ``update_P`` no longer lose it. A stale demand figure is worse than none, because
#: it reads as current — so the correction lands in the same change as the build,
#: and ``superseded_demand`` keeps what it said.
STORED_E_ARM: Dict[str, Any] = {
    "sub_step": "update_E",
    "reached_when": "not _pml_is_active(pml) and fields.stores_E",
    "array_path": (
        "for component, _, axis in E_CONSTITUTIVE_TERMS (stepping.py:228, :998):\n"
        "    source       = fields.displacement_minus_polarization(component)  # :1010\n"
        "    constitutive = source * fields.inverse_epsilon_for(component)     # :982-984\n"
        "    fields.<component>[...] = constitutive                            # :993"
    ),
    "displacement_minus_polarization": (
        "fields.py:1079-1105 — s = ((D_c - P_0[c]) - P_1[c]) - ..., LEFT TO RIGHT in "
        "fields.polarizations order, per component, over the contributors that drive "
        "that component only. Pre-accumulating sum(P) is a different float32 number "
        "(dispersive_update_e.py, bit-identity note 1: caught 20/20 at two poles)."
    ),
    "differs_from_certified_body": (
        "kernels.constitutive_step accumulates ((f + kps*src) - kms*prev) and stores "
        "f_w; this arm STORES src and writes no f_w. kps=1/kms=0/f=0 does not recover "
        "it: 0.0 + (-0.0) = +0.0 canonicalises every negative zero, and 0.0*prev is "
        "NaN for non-finite prev. Measured on NumPy float32, 1 differing word of 4."
    ),
    "compose_from": (
        "dispersive_update_e.dispersive_update_e — same pole loop, same left-to-right "
        "subtraction, same D-minus-P-on-the-left operand order; delete the f_w store "
        "and replace the (E + kps*src) - kms*prev tail with tl.store(e + idx, src). "
        "The nonlinear and off-diagonal variants of :993 would need "
        "nonlinear_update_e and offdiag_update_e's row products on the same tail."
    ),
    "drive_field_consequence": (
        "Without PML, Fields.drive_field returns the STORED E, not f_w "
        "(fields.py:1140-1162), and f_w_* is not allocated at all. So this arm must "
        "NOT write f_w, and ade_update_p_coverage — which requires f_w_<component> — "
        "refuses every no-PML row today. Arm S alone does not unblock update_P."
    ),
    "measured_demand": (
        "3 SLOTS ON 3 ROWS, and 3 WHOLE-STEP ROWS when composed. Re-measured "
        "2026-08-16 against results/predicate_coverage_2026-08-16_wired_convention/ "
        "(186 rows, 759 slots): 9 of the 14 no-PML rows reach this arm; 6 force "
        "complex storage (2 of them also off-diagonal) and stay outside a real-f32 "
        "product; the 3 that remain are TestAbsorber.test_absorber, absorber-1d.py "
        "and material-dispersion.py. material-dispersion.py has update_E as its ONLY "
        "unadmitted sub-step — plain_curl already holds both curls, Arm N holds "
        "update_H, no_pml_ade holds update_P — so Arm S takes it whole-step ALONE. "
        "The other two need no_pml_conductive's curls beside this arm, and with all "
        "three they go whole-step too. Built on that evidence."
    ),
    "superseded_demand": (
        "WHAT THIS FIELD SAID UNTIL 2026-08-16, kept because the reason it went "
        "stale is the interesting part: '... would gain update_E and nothing else: "
        "the first two have conductive curls that are still refused, all three lose "
        "update_P for want of f_w_*, and the third is a single voxel. ZERO "
        "additional whole-step rows. Not built on that evidence.' Both blockers it "
        "named were closed by other rounds — no_pml_ade inverted the f_w clause and "
        "the wiring round gave update_P to exactly those three rows — and the note "
        "went on declining the arm for a cost nobody had re-measured. The single "
        "voxel is still a single voxel; it was never the load-bearing half."
    ),
    "built": True,
    "built_as": (
        "triton_kernels/no_pml_stored_e.py — stored_e_constitutive_step, "
        "stored_e_constitutive_coverage, plan_stored_e_constitutive. A NEW module "
        "beside dispersive_update_e, exactly as compose_from required, so this file "
        "still contains no arithmetic. NOTE FOR THE METAL PACKAGE, which imports "
        "this record: 'built' is about the TRITON arm. metal_kernels has no stored-E "
        "arm, and metal_kernels/no_pml_constitutive.py's re-export inherits this "
        "flag rather than stating its own."
    ),
    "device_bytes_owed": (
        "parity/meep_gpu/gate_triton_no_pml_stored_e.py has never run: Triton does "
        "not build on arm64 macOS. The kernel's identity, its mutation battery and "
        "its step budget are OWED, not measured, and no digest for it appears in "
        "fingerprints.json until that gate's artifact exists."
    ),
}


#: The clauses that refuse a configuration on which the null WOULD be byte-exact,
#: each with the measurement that establishes it. Recorded because this package's
#: rule is that a refusal must be TRUE where it fires, and "conservative" is part of
#: the truth: a reader who lands on one of these must be told that the null was not
#: wrong, only unbuilt-for. The ``moved_words`` numbers are what the real
#: ``stepping`` sub-step moved on a seeded ``Fields`` built the way ``needle`` says
#: (uint32 compare, laptop NumPy, 2026-08-13); the suite re-measures every row rather
#: than trusting this table, and a row whose measurement stops agreeing fails there.
#:
#: The one feature clause that is NOT here is the off-diagonal row, and that is a
#: measured distinction rather than an oversight: ``Fields.set_epsilon_volumes``
#: calls ``enable_field_storage()`` on a surviving row (fields.py:1253-1255), so
#: ``stores_E`` is already True and clause 3 has refused before that sentence is
#: reached. Measured on the same fixture: 4758 words moved, i.e. the null would have
#: been WRONG there and the clause is genuinely vacuous rather than conservative.
CONSERVATIVE_CLAUSES: Dict[str, Dict[str, Any]] = {
    "pml_storage_behind_an_inert_layer": {
        "side": "H",
        "clause": "_storage_switch_reasons",
        "needle": "Fields.enable_pml_storage() with PML(thickness=0)",
        "moved_words": 0,
        "why_kept": "no_pml.plain_curl_coverage clause 3b refuses the CURL on this "
                    "configuration, so supplying half a step here would hand the "
                    "package's own unsafe configuration a partial substitution",
    },
    "chi2_chi3_installed_without_stored_e": {
        "side": "E",
        "clause": "_electric_source_reasons",
        "needle": "Fields.set_nonlinear_volumes (NOT driver.set_chi2/set_chi3, which "
                  "calls enable_field_storage at driver.py:2016-2022)",
        "moved_words": 0,
        "why_kept": "update_E's product for those components is nonlinear_update_e's "
                    "Pade factor; the null is byte-exact only because stores_E "
                    "happens to be False, which is a fact about the Fields object "
                    "rather than about the nonlinearity",
    },
    "magnetic_susceptibility_on_the_h_side": {
        "side": "H",
        "clause": "_magnetic_susceptibility_reasons",
        "needle": "a duck-typed polarization whose drives('Bx') is True",
        "moved_words": 0,
        "why_kept": "stepping.py:1406-1407 records the H-side polarization slot as "
                    "deliberately empty; a null must not report coverage of a "
                    "sub-step that has grown a term",
    },
    "a_polarization_that_cannot_answer_driven": {
        "side": "E",
        "clause": "_electric_source_reasons",
        "needle": "a polarization object with no driven() method",
        "moved_words": 0,
        "why_kept": "an unreadable susceptibility is not a covered one — the same "
                    "rule the two load-bearing flags now apply to themselves",
    },
}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

#: Returned by :func:`_flag` when a yes/no attribute could not be read at all.
_UNREADABLE = object()


def _flag(obj: Any, name: str) -> Any:
    """Read a yes/no attribute, or report that the object could not answer.

    ``getattr(obj, name, False)`` is the spelling this file used to carry, and on the
    two clauses the whole family rests on it turns "cannot answer" into ADMIT — the
    opposite of the rule the polarization clause already applied two clauses later
    ("an unreadable susceptibility is not a covered one"). Measured before the fix: a
    ``Fields`` proxy whose ``stores_E`` property raised was covered with NO reasons
    while the underlying run's ``stepping.update_E`` wrote 4618 words.

    Absent, raising, or ``None`` are all the same answer here — *none* — and every
    caller turns that into a refusal. That is conservative in the safe direction: the
    array path reads the same attribute directly and would crash on such an object
    rather than diverge silently, so this converts a wrong verdict on a malformed
    input into a named refusal.
    """
    try:
        value = getattr(obj, name)
    except Exception:  # noqa: BLE001 - an unanswerable question is not coverage
        return _UNREADABLE
    if value is None:
        return _UNREADABLE
    return bool(value)


def _inactive_layer_reasons(pml: Any) -> List[str]:
    """Clause 2, alone, because it is the one clause BOTH sides rest on.

    ``stepping._pml_is_active`` (stepping.py:2498-2506) is the test the array path
    itself branches on at stepping.py:944 and :982. Asking the same question the same
    way is what makes this family's coverage set the exact complement of every
    active-PML constitutive predicate's, rather than an approximation of it — and an
    all-zero-face layer lands HERE, not there, which is the case
    ``PML.is_active``'s own docstring (pml.py:427-434) exists to name.

    A layer that cannot answer ``is_active`` is REFUSED rather than treated as inert:
    this is one of the two load-bearing flags in the file, and defaulting it to the
    admitting answer is the defect :func:`_flag` exists to close.
    """
    if pml is None:
        return []
    active = _flag(pml, "is_active")
    if active is _UNREADABLE:
        return ["the layer does not report is_active: stepping._pml_is_active "
                "(stepping.py:2498-2506) branches on exactly that attribute, and a "
                "layer that cannot answer it is not a demonstrably inert one"]
    if active:
        return ["an active PML layer is installed: update_H/update_E run the dsigw "
                "accumulation (stepping.py:946-951, :1014-1018), which is not a no-op"]
    return []


def _electric_source_reasons(fields: Any) -> List[str]:
    """Clause 3 plus the features that reach the same product, each written out.

    ``stores_E`` is the whole of clause 3 — it is literally what stepping.py:983
    branches on. Of the four features below, THREE force it on and one does not, and
    the difference is measured rather than asserted:

    * a polarization that DRIVES a component  -> driver.py:1538-1542, forces it
    * a surviving off-diagonal chi1inv row    -> fields.py:1253-1255, forces it
      (measured vacuous: on that object clause 3 has already refused, and
      ``stepping.update_E`` moved 4758 words, so the null would have been WRONG)
    * PML storage                             -> fields.py:700, clause 2 already
    * chi2/chi3                               -> forced by ``driver.set_chi2`` /
      ``set_chi3`` (driver.py:2016-2022) but **NOT** by
      ``Fields.set_nonlinear_volumes``. On a ``Fields`` built the second way,
      ``has_nonlinearity`` is True while ``stores_E`` is False, the array path
      returns at stepping.py:983 and the null is byte-exact (measured: 0 of 10080
      words moved, census 10080 / 847 signed zeros / 777 subnormals). The clause is
      therefore CONSERVATIVE on that arm, is labelled so in its own reason text, and
      is listed in :data:`CONSERVATIVE_CLAUSES`.

    This file used to claim all four force storage. That claim was FALSE about the
    object the chi2/chi3 clause fires on, which is exactly the ``no_pml.py`` clause-3b
    defect class this package's rule exists to forbid: a right verdict with a false
    reason is still a defect in a reasons list.

    They are named regardless, because ``coverage.py``'s own rule is that "another
    module already guards it" is the reasoning this package exists to refuse — and
    because each one is a DIFFERENT product (Arm S, nonlinear_update_e,
    offdiag_update_e), so a reader who lands on the refusal is told which.

    The polarization clause asks ``driven()``, not "is a polarization registered".
    A susceptibility whose sigma is identically zero drives nothing and does NOT
    switch storage on (driver.py:1538-1542, "what makes the zero-strength case
    byte-identical rather than merely close"), so refusing on registration alone
    would refuse a configuration where the array path really does return at :954.
    A ``driven()`` that answers with something that is not a collection of component
    names is REFUSED, not raised on: the H side has always routed every question
    through ``coverage._call``, and a predicate that raises where its sibling refuses
    becomes a crash the moment ``launch.plan_step`` consults it.
    """
    reasons: List[str] = []
    stored = _flag(fields, "stores_E")
    if stored is _UNREADABLE:
        reasons.append(
            "fields does not report stores_E: stepping.py:983 branches on exactly "
            "that attribute, and a run that cannot answer it is not a covered one")
    elif stored:
        reasons.append(
            "fields.stores_E is True: update_E writes E[...] = (D - sum P) * inv_eps "
            "(stepping.py:1022) instead of returning at :983 — that is the STORE arm, "
            "no_pml_stored_e.stored_e_constitutive_coverage (STORED_E_ARM)")
    for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        driven = _call(state, "driven", default=None)
        names: Optional[Tuple[Any, ...]]
        if driven is None:
            names = None
        else:
            try:
                names = tuple(driven)
            except TypeError:
                names = None
        if names is None:
            reasons.append(
                f"polarization {index} ({type(state).__name__}) does not report "
                "driven(); an unreadable susceptibility is not a covered one")
            continue
        for name in names:
            if name in ELECTRIC_COMPONENTS:
                reasons.append(
                    f"polarization {index} drives {name}: update_E's source becomes "
                    "(D - sum P) and storage is switched on (driver.py:1542)")
    nonlinear = _flag(fields, "has_nonlinearity")
    if nonlinear is _UNREADABLE:
        reasons.append(
            "fields does not report has_nonlinearity; a run that cannot say whether "
            "it carries a Pade constitutive factor is not a covered one")
    elif nonlinear:
        reasons.append(
            "chi2/chi3 is installed: update_E's product for those components is the "
            "Pade constitutive factor, which belongs to nonlinear_update_e. "
            "CONSERVATIVE where it fires alone — driver.set_chi2/set_chi3 force "
            "stored E (driver.py:2016-2022) so a driver-built run has already been "
            "refused above, but Fields.set_nonlinear_volumes does not, and on that "
            "object update_E returns at stepping.py:983 and the null would be "
            "byte-exact (measured 0 words moved)")
    offdiagonal = _flag(fields, "has_offdiagonal_epsilon")
    if offdiagonal is _UNREADABLE:
        reasons.append(
            "fields does not report has_offdiagonal_epsilon; a run that cannot say "
            "whether update_E forms a row product is not a covered one")
    elif offdiagonal:
        reasons.append(
            "an off-diagonal chi1inv row is installed: surviving rows force stored E "
            "(fields.py:1255) and the row product belongs to offdiag_update_e")
    return reasons


def _magnetic_susceptibility_reasons(fields: Any) -> List[str]:
    """The H side's only feature clause, and it is CONSERVATIVE, not vacuous.

    The distinction matters and is measured: ``update_H`` returns at stepping.py:944
    whatever is registered, so on a duck-typed state that claims ``drives('Bx')`` the
    null is byte-exact — 0 of 10080 words moved on a seeded ``Fields`` — and this
    clause refuses it anyway. "Vacuous" would mean another clause had already
    refused; nothing else does, so this one changes the verdict on its own and
    belongs in :data:`CONSERVATIVE_CLAUSES`.

    What it protects is the CLAIM rather than the arithmetic: a registered magnetic
    susceptibility would mean the H-side slot MEEP fills after ``update_eh(H_stuff)``
    — which stepping.py:1406-1407 records as "deliberately empty ... magnetic
    susceptibilities are refused at the driver" — has become occupied by a sub-step
    nothing in this package has transcribed. A null plan in that world reports full
    coverage of a step that grew a term. Refused by name for that reason and no other.
    """
    reasons: List[str] = []
    for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
        for name in MAGNETIC_COMPONENTS:
            if _call(state, "drives", name, default=False):
                reasons.append(
                    f"polarization {index} drives magnetic component {name}: the "
                    "H-side polarization slot is empty in this engine "
                    "(stepping.py:1406-1407) and a null cannot report coverage of a "
                    "sub-step that grew a term")
    return reasons


def _storage_switch_reasons(fields: Any, pml: Any) -> List[str]:
    """CONSERVATIVE, and labelled so: the null is byte-exact here and still refuses.

    ``Fields.enable_pml_storage`` is a one-way switch that also changes what
    ``get_E``/``get_H`` return (fields.py:678-700). A ``Fields`` whose storage was
    switched on while the layer handed to the stepper is inert reads H from the
    STORED array — which ``update_H`` then declines to write — so the curl
    differences a frozen H. ``no_pml.py`` clause 3b refuses that configuration for the
    curl; this family would be byte-exact on it (doing nothing is exactly what the
    array path does) and refuses anyway, so the package does not supply half a step to
    a configuration whose other half it has declared unsafe.

    It is NOT the only clause here that is unrequired by the arithmetic, and the
    earlier claim that it was is one of the things this round repaired.
    :data:`CONSERVATIVE_CLAUSES` enumerates all four, each with the number of words
    the real ``stepping`` sub-step moved on its own needle — all four zero. This one
    is still written as its own function so that a future reader can delete it
    without having to work out which of the other clauses are load-bearing.
    """
    if pml is not None and _flag(pml, "is_active") is True:
        return []  # Clause 2 has already refused; this sentence would be false here.
    # ``_pml_active`` is PRIVATE, and unlike the four public flags its absence is a
    # real answer rather than an unreadable one: an object carrying no PML-storage
    # concept has no PML storage switched on. Read permissively on purpose, and the
    # clause it feeds is conservative either way.
    if bool(getattr(fields, "_pml_active", False)):
        return ["Fields has PML storage enabled while the layer is inert (get_H would "
                "serve a stored H that update_H never writes) — refused conservatively "
                "to match no_pml.plain_curl_coverage clause 3b, not because the null "
                "would be wrong"]
    return []


def null_constitutive_coverage(fields: Any, pml: Any, side: str) -> Coverage:
    """Is ``update_H`` (side='H') / ``update_E`` (side='E') a no-op for this run?

    Positive clauses only, and there are four of them:

    1. the side is one this family knows and ``fields`` carries a grid;
    2. **the layer is inert** — ``stepping._pml_is_active`` False (stepping.py:944,
       :982). Both sides;
    3. **E is not stored** — ``fields.stores_E`` False (stepping.py:983). E side
       only, with the features that reach the same product named individually;
    4. no magnetic susceptibility (H side), and no PML storage switched on behind an
       inert layer (both sides, conservative).

    Clauses 2 and 3 are the load-bearing pair, and each is read through :func:`_flag`,
    so an object that CANNOT answer is refused by name rather than admitted by
    default. Four clauses refuse configurations on which the null is byte-exact;
    they are enumerated with their measurements in :data:`CONSERVATIVE_CLAUSES`, and
    every reason string that fires on one says so where it fires.

    NOT clauses, deliberately, each of which every other predicate in this package
    carries: the backend (``xp``), the storage width, the ghost rule, the fold, the
    cylindrical axis, the Bloch phase, beta, BFAST, the conductivity, array layout,
    dtype and contiguity. A sub-step that returns before its first statement reads no
    pointer, forms no index and rounds no float, so there is nothing for any of them
    to be wrong about. The gate measures that on a folded, a cylindrical, a complex
    and a Bloch grid rather than asserting it here.

    ``xp`` in particular is absent on purpose. Every kernel predicate opens with
    "array module is not cupy" because a kernel launches against device pointers;
    this product launches nothing, so it is correct — and covered — on NumPy. That
    also makes it the only family in this package whose gate needs no GPU.
    """
    if side not in NULL_SIDES:
        raise ValueError(f"side must be one of {tuple(NULL_SIDES)}, got {side!r}")
    if getattr(fields, "grid", None) is None:
        return Coverage(False, ("fields carries no grid",))

    reasons: List[str] = []
    reasons.extend(_inactive_layer_reasons(pml))
    if NULL_SIDES[side]["needs_stored_e_clause"]:
        reasons.extend(_electric_source_reasons(fields))
    else:
        reasons.extend(_magnetic_susceptibility_reasons(fields))
    reasons.extend(_storage_switch_reasons(fields, pml))
    return Coverage(not reasons, tuple(reasons))


def stored_e_constitutive_reasons(fields: Any, pml: Any) -> Tuple[str, ...]:
    """Why Arm S refuses THIS configuration — now a real question with a real answer.

    **THIS FUNCTION'S CONTRACT CHANGED ON 2026-08-16 AND THE CHANGE IS THE POINT.**
    It used to be a record rather than a predicate, and its docstring said "Always
    non-empty: the arm is not built". The arm IS built — ``no_pml_stored_e`` — so an
    unconditional refusal here would be a false statement about a live product, and
    a composition asking "which product is missing" would be told the wrong thing.

    It now DELEGATES to the built predicate, and returns empty exactly where Arm S
    covers the run. The name is kept because the question is the same one; nothing
    is aliased and no second copy of the clause set exists here. Callers that
    treated a non-empty result as "the arm does not exist" must read
    :data:`STORED_E_ARM`'s ``built`` flag instead, which is what it is for.

    Kept in THIS module rather than moved, for one reason: ``metal_kernels`` imports
    it from here by name and that package is not this round's to edit. Its Metal
    re-export inherits the corrected meaning along with the corrected record.
    """
    from .no_pml_stored_e import stored_e_constitutive_coverage  # noqa: PLC0415

    return stored_e_constitutive_coverage(fields, pml).reasons


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class NullConstitutivePlan:
    """The plan for a sub-step that does nothing: it also does nothing.

    NOT ``launch.NoopPlan``, and the distinction is the whole reason this class
    exists. ``NoopPlan`` is the sentinel a FUSED PAIR leaves in the slot of a sub-step
    another kernel absorbed — it carries ``absorbed_by`` and it means "the work
    happened, over there". This one means "there was no work", which is a different
    fact about the run and must not be reported as the first.

    ``runs`` is a counter, and it is load-bearing for the gate rather than for the
    engine. A leg that certifies "the plan reproduced the array path" by comparing
    bytes is trivially satisfied by a plan that was never invoked, so the gate's
    DISARMED discipline needs something to count. Here that cannot be a kernel launch
    — there is no kernel — so it is this.

    ``block`` is accepted and ignored, so the constructor matches every other plan
    builder in the package and a composer does not have to special-case the call.
    """

    __slots__ = ("side", "sub_step", "runs", "block")

    def __init__(self, side: str, block: Optional[int] = None) -> None:
        if side not in NULL_SIDES:
            raise ValueError(f"side must be one of {tuple(NULL_SIDES)}, got {side!r}")
        self.side = side
        self.sub_step = NULL_SIDES[side]["sub_step"]
        self.block = block
        self.runs = 0

    def run(self, guard: Optional[bool] = None) -> None:
        """Reproduce ``stepping.update_{H,E}``'s no-PML return, exactly.

        ``guard`` is accepted for signature compatibility with the launching plans
        and is meaningless here: ``ENABLE_FP_FUSION`` gates floating-point
        contraction, and this call performs no floating-point operation to contract.
        """
        del guard
        self.runs += 1

    def __repr__(self) -> str:
        return (f"NullConstitutivePlan({self.sub_step}, side={self.side!r}, "
                f"runs={self.runs})")


class NullConstitutiveStepPlan:
    """Which constitutive sub-steps of one inactive-layer run are null, with reasons.

    Composed the way ``no_pml.PlainStepPlan`` is, and reporting the same kind of
    thing: on a run with no absorber and no stored E, ``{update_H, update_E}`` being
    null is FULL coverage of the step's constitutive arithmetic, not a subset of it.
    Together with ``no_pml.PlainStepPlan``'s ``{step_B, step_D}`` that is the whole
    heavy step, which is what takes the four measured rows whole-step.
    """

    __slots__ = ("plans", "refusals")

    def __init__(self, plans: Dict[str, NullConstitutivePlan],
                 refusals: Dict[str, Tuple[str, ...]]) -> None:
        self.plans = plans
        self.refusals = refusals

    @property
    def covered(self) -> Tuple[str, ...]:
        return tuple(name for name in ("update_H", "update_E") if name in self.plans)

    def run(self) -> None:
        """Run both null sub-steps in the driver's order.

        Unlike ``no_pml.PlainStepPlan.run``, this one is implementable: the driver's
        boundary passes sit between ``update_H`` and ``update_E``
        (driver.py:3212-3225) and every one of them is a no-op to a plan that
        performs no operation, so running the two back to back cannot reorder work
        that does not exist. It still counts both, so a caller can prove it ran.
        """
        for name in ("update_H", "update_E"):
            plan = self.plans.get(name)
            if plan is not None:
                plan.run()

    def __repr__(self) -> str:
        covered = ", ".join(self.covered) if self.covered else "nothing"
        return f"NullConstitutiveStepPlan(covered=[{covered}])"


def plan_null_constitutive(fields: Any, pml: Any, side: str,
                           block: Optional[int] = None
                           ) -> Optional[NullConstitutivePlan]:
    """Build the null plan for one constitutive side, or None when refused.

    Allocation-free and import-free: no Triton, no CuPy, no ``kernels`` module, not
    even a lazy import inside the function. Every other builder in this package pulls
    ``DEFAULT_BLOCK`` in below its predicate; there is no block size to choose when
    there is no launch, so this one does not.
    """
    if not null_constitutive_coverage(fields, pml, side).covered:
        return None
    return NullConstitutivePlan(side, block)


def plan_null_constitutive_step(fields: Any, pml: Any,
                                block: Optional[int] = None
                                ) -> NullConstitutiveStepPlan:
    """Ask both sides independently and report the covered subset with its refusals."""
    plans: Dict[str, NullConstitutivePlan] = {}
    refusals: Dict[str, Tuple[str, ...]] = {}
    for side, spec in NULL_SIDES.items():
        name = spec["sub_step"]
        verdict = null_constitutive_coverage(fields, pml, side)
        if verdict.covered:
            plan = plan_null_constitutive(fields, pml, side, block=block)
            if plan is not None:
                plans[name] = plan
                continue
            refusals[name] = ("coverage passed but the plan builder refused",)
        else:
            refusals[name] = verdict.reasons
    return NullConstitutiveStepPlan(plans, refusals)


# ---------------------------------------------------------------------------
# WIRING — items 1 and 3 are DONE; 2, 4 and 5 stand as written
# ---------------------------------------------------------------------------
#
# This block was written as a plan and read for a round as a statement of fact,
# which it had stopped being. What follows is the seam as it now stands, item by
# item, with the two that were performed marked as such.
#
# 1. DONE. ``plan_step`` evaluates ``null_constitutive_coverage`` alongside the
#    ordinary, dispersive, folded and cylindrical constitutive predicates, as the
#    ``no-PML null`` arm — last in both tables, behind the ``absorber_inactive``
#    gate, which is the only gate there that reads the LAYER rather than the
#    fields. Selection is one shared ``_select_slot`` per slot, so two admitters
#    leave the slot unselected naming both: an overlap fails closed here as it
#    does everywhere else, and the null is neither preferred over a kernel nor a
#    kernel over it.
#
# 2. NO overlap is possible by construction, and the gate's ``overlap`` leg measures
#    it rather than asserting it. Every other constitutive predicate reaches
#    ``coverage._grid_reasons`` clause 3 ("no active PML layer" -> refuse) or its
#    per-family restatement, and this one refuses the exact complement at clause 2.
#    The two tests are the same call to ``pml.is_active``.
#
# 3. DONE. ``__init__.py`` carries the two lazy forwarders the other families have
#    (``null_constitutive_coverage``, ``plan_null_constitutive``) and names them in
#    ``__all__``. Unlike every existing one, these need no ``try: import triton``
#    guard at all.
#
# 4. ``fingerprints.json`` does NOT gain an entry. Its census is over kernel source
#    hashes and this family ships no kernel; welding a host predicate to a source
#    digest would record a fact about a file rather than about a numerical product.
#    The gate's identity leg is the record, and it re-runs on any host.
#
# 5. DONE, 2026-08-16, and done the way this item required. Arm S is
#    ``no_pml_stored_e.py`` — a NEW module beside ``dispersive_update_e``, not a
#    widening of this one. This file is still about the absence of arithmetic and
#    still contains none: what changed here is one record (:data:`STORED_E_ARM`)
#    and one function that used to hard-code a refusal
#    (:func:`stored_e_constitutive_reasons`, which now delegates).
#
#    THE TWO ARMS ARE DISJOINT ON ONE ATTRIBUTE and it is the one stepping.py:983
#    branches on: this family refuses ``stores_E`` True by name and Arm S requires
#    it. An overlap is therefore impossible for the same reason item 2's is, and the
#    ``update_E`` slot's inert-layer pair is meant to be read side by side in
#    ``plan_step``'s table.
#
# 6. STILL NOT DONE, and it is not this module's to do: ``fastpath.plan_fast_path``
#    returns ``None`` on every branch, so nothing a production step runs reaches
#    the planner at all. The planner is called by tests, gates and the benchmark.
#
# 7. ONE CONSEQUENCE OF ITEM 1 A CALLER CAN SEE. This is the only family with no
#    backend clause (see :func:`null_constitutive_coverage` — a product that
#    launches nothing is correct on NumPy), so on a host with no CuPy
#    ``plan_step(...).replaces`` now names ``update_H``/``update_E`` where it used
#    to be empty. True, and a meaning change: ``bool(plan.replaces)`` no longer
#    reads as "a Triton kernel covers something here". ``plan.selected`` names the
#    arm and still does.
