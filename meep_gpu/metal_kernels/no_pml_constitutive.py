"""The NO-PML constitutive family on Metal: a predicate over a sub-step that does nothing.

THE FINDING, STATED FIRST, BECAUSE IT DECIDES WHAT THIS FILE IS. This family ships
NO KERNEL — no Metal source, no ``compile_shader`` call, no dispatch, no buffer
binding. Under an inactive absorber ``stepping.update_H`` returns at
stepping.py:944-945 and ``stepping.update_E`` at stepping.py:983-984, before either
one reads an array. There is no arithmetic to transcribe, no grouping to hold, no
coefficient to index and no float to round, so there is nothing for the contraction
directive, the signed-zero rules or the select-instead-of-min rules to be about.

WHAT THAT MAKES THE CLAIM. A planner-level substitution, and it is only meaningful
because the gate ALSO proves the step MOVED STATE. A null agreeing with a no-op is
trivially identical — that pass condition, added late on the Triton track, caught a
live row (``material_dispersion_user_material_1voxel``) that had been certifying two
frozen states. The discipline is what ports here, not just the predicate.

WHY THIS FILE EXISTS AT ALL, WHEN THE PREDICATE IS ALREADY BACKEND-FREE
======================================================================
``triton_kernels.no_pml_constitutive.null_constitutive_coverage`` carries no
backend clause by design — a product that launches nothing is correct on NumPy —
and tranche 1's ``launch.plan_step`` therefore consumed it VERBATIM and recorded
that the family "needed zero Metal work". That was true of the PREDICATE and false
of the COMPOSITION, and the difference was measured on this host rather than
argued:

**THE RESIDENCY VERDICT FALSELY REFUSED THE ONE CONFIGURATION THIS FAMILY EXISTS
FOR.** Measured 2026-08-15, on the exact ``no_pml_no_storage`` case tranche 1's
gate leg 8 builds::

    replaces            ('update_H', 'update_E')
    selected            {'update_H': 'no-PML null', 'update_E': 'no-PML null'}
    residency covered   False
      update_H is planned but ('Hx','Hy','Hz','f_w_Hx','f_w_Hy','f_w_Hz',
                               'Bx','By','Bz') carry no mirror
      update_E is planned but ('Ex','Ey','Ez','f_w_Ex','f_w_Ey','f_w_Ez',
                               'Dx','Dy','Dz') carry no mirror

SIX of those nine names per side are NOT ALLOCATED on such a run at all. This
paragraph said FOUR for a round, which was a count nobody had taken; the number is
now MEASURED through :func:`..coverage.sub_step_volumes` against the fixture the
gate builds (2026-08-15)::

    update_H   9 volumes, 6 unallocated: Hx Hy Hz f_w_Hx f_w_Hy f_w_Hz
    update_E   9 volumes, 6 unallocated: Ex Ey Ez f_w_Ex f_w_Ey f_w_Ez

H is never stored without PML (``Fields.enable_field_storage``, fields.py:664-666);
E is served on demand as ``D * inv_eps`` and is not stored either; ``f_w_*`` is
allocated only by ``enable_pml_storage``. ``Bx/By/Bz`` and ``Dx/Dy/Dz`` are the only
volumes that exist, so the refusal demanded mirrors of arrays that do not exist for
TWO THIRDS of the names it listed, for a sub-step that binds no buffer. It is
recorded in the tranche-1 artifact as ``residency_covered: false`` on that row, and
the gate passed anyway because leg 8 records the field without asserting it. The
gate's ``residency`` leg now COUNTS the unallocated set per side instead of sampling
three names out of it, so this number is re-measured on every cut rather than read
back from here.

The defect is in the residency MODEL, not in the predicate: ``planned`` meant "runs
on the device against the mirror", and a null runs on neither. :func:`..coverage.residency_reasons`
now takes a fourth declared set, ``null``, whose members neither REQUIRE a mirror
(they bind nothing) nor STALE one (they write nothing) — and this module is where
the composition that declares it lives.

THE OTHER THREE METAL-SPECIFIC GROUNDS, each a real difference and none of them
about arithmetic:

1. **The plan protocol.** Every Metal plan is launched as ``run(contract=None)``,
   because the contraction directive is a property of the SOURCE here rather than
   of the launch, so a plan holds one compiled function per mode. The Triton null
   plan's signature is ``run(guard=None)``. A Metal composition that iterated its
   plans calling ``run(contract=mode)`` — which the guard leg does — would get a
   ``TypeError: unexpected keyword argument`` from the Triton object, at the far
   end of a gate leg, reported as a crash rather than as the correct answer. So
   the null plan is re-implemented against the Metal protocol.
2. **The contraction guard is ANSWERED, not skipped.** :meth:`MetalNullConstitutivePlan.run`
   accepts every mode and launches nothing under all of them, and
   :attr:`~MetalNullConstitutivePlan.variants` is empty — which is the honest
   report, because this plan holds no source and therefore no variant of one. A
   family that cannot answer the guard question and a family for which the guard
   question is empty must not look the same in an artifact.
3. **The subnormal precondition is satisfied VACUOUSLY, and on Metal that claim
   has teeth.** The MPS executor flushes float32 subnormals natively and exposes
   no lever (``subnormal.mps_policy_reasons``), so any Metal code that touched a
   field array carrying subnormals would destroy them. This family's gate SEEDS
   DENSE SUBNORMALS deliberately and shows byte-identity anyway — which is a
   direct measurement that nothing on the device touched the arrays, and is a
   stronger statement here than the same leg makes on a keep-policy host.

WHAT IS IMPORTED AND NEVER RE-DERIVED
=====================================
The predicate itself. :func:`null_constitutive_coverage` below IS
``triton_kernels.no_pml_constitutive.null_constitutive_coverage`` — the same
function object, re-exported under this package's namespace with a Metal-shaped
signature wrapper for the arm registry, and
``test_metal_no_pml_constitutive.test_the_predicate_is_the_shared_function_object``
asserts the identity so a copy cannot appear. So are :data:`NULL_SIDES`,
:data:`CONSERVATIVE_CLAUSES`, :data:`STORED_E_ARM` and
:func:`stored_e_constitutive_reasons`. Nothing about which configurations are null
is decided twice.

THE ONE MEASUREMENT EVERY NON-CLAUSE RESTS ON, and it was a READING until
2026-08-15. "A sub-step that returns before its first statement reads no pointer"
was transcribed from stepping.py:944-945 / :983-984 and never executed. It is now
executed: the gate's ``over_coverage`` leg REBINDS every allocated field volume to
an object that raises on ``__getitem__``, ``__setitem__``, ``__array__``, any
arithmetic and any other attribute, and calls the real ``stepping`` sub-steps.
Measured on this host::

    covered no-PML       update_H   7 poisoned  RETURNED, touched nothing
    covered no-PML       update_E   7 poisoned  RETURNED, touched nothing
    covered no layer     update_H   7 poisoned  RETURNED, touched nothing
    covered no layer     update_E   7 poisoned  RETURNED, touched nothing
    CONTROL active PML   update_H  25 poisoned  RAISED at attribute 'xp'
    CONTROL active PML   update_E  25 poisoned  RAISED at __mul__
    CONTROL stores_E     update_E  10 poisoned  RAISED at __mul__

SEVEN is itself the finding restated: on the family's own run only ``Bx/By/Bz``,
``Dx/Dy/Dz`` and ``scratch`` exist to be poisoned at all, which is the same
six-of-nine-per-side count the residency model turns on.

The controls are what make it a measurement rather than a tautology: the same
poison under a configuration this family REFUSES is detected immediately. With that
in hand every entry below is a consequence rather than an argument — a sub-step
that provably touches no field array cannot be wrong about a stride, an index, a
ghost rule or a rounding mode.

THE NON-CLAUSES, NAMED, because every other predicate in this package carries them
and a reader who does not find them here must be told it was deliberate. There are
FIFTEEN and they are enumerated in :data:`METAL_NON_CLAUSES`; this prose said FOUR
while the table held eleven, which is the same false-count defect the Triton file
repaired on its own conservative list, and the table then held ELEVEN while the
siblings fired FIFTEEN distinct clauses on configurations this family ADMITS.
The four HEADINGS below group them as 1 + 1 + 1 + 12, and the TABLE — not this
prose — is what the gate reads:

* **the Metal backend** (torch importable, MPS built and available,
  ``compile_shader`` present, the engine's array module) — a product that compiles
  and launches nothing is correct on a host with no GPU. This is the only family
  in the package whose gate needs no device;
* **the residency declaration** — every kernel predicate refuses a plan built with
  no residency, because two sub-steps that mirrored one volume separately would
  each hold a private copy. This plan mirrors nothing, so there is no copy to
  hold and no volume to share. Refusing here would be a refusal that is FALSE
  where it fires, which is the defect class ``no_pml.py`` clause 3b was rewritten
  to remove;
* **the subnormal policy** — ``coverage._metal_backend_reasons`` clause 4 refuses a
  run resolved to ``keep`` because the device cannot deliver it. No operand,
  result or intermediate exists here, so ``flush`` and ``keep`` are not merely
  indistinguishable, they are both unreached;
* **the remaining TWELVE table entries** — ``storage_width``, ``layout`` (shape,
  dtype and contiguity), ``fold``, ``cylindrical_axis``, ``bloch_phase``,
  ``beta``, ``bfast``, ``conductivity``, ``nonlinearity``, ``volume_allocation``,
  ``susceptibility_kind`` and ``susceptibility_registration``, which with the
  three above make the fifteen. A
  sub-step that returns before its first statement reads no pointer, forms no
  index and rounds no float. The gate's ``breadth`` leg measures that on a folded,
  a cylindrical, a complex, a Bloch, a beta, a BFAST, a conductive, a
  non-contiguous, a chi3 and an unrecognised-pole configuration rather than
  leaving it as an argument, and it measures the SIBLING half too: the Metal
  kernel predicate must refuse each of those grids NAMING THE CLAUSE this family
  drops. Asserting only that the sibling refused would have been vacuous — every
  one of these grids is already refused by ``_grid_reasons`` clause 3 ("no active
  PML layer"), which fires whatever the fold clause does. That is the Metal
  analogue of the Triton suite's ``_residual()`` helper, and it was missing here.

THE LAST FOUR ENTRIES WERE MISSING UNTIL 2026-08-15, and the way they were missing
is the finding rather than the entries. The table's totality assertion ran in ONE
DIRECTION — every table key must have a case — so a clause the siblings really fire
on a configuration this family ADMITS could simply never be written down and
nothing would notice. Measured by sweeping the null-admitted configurations and
collecting every sibling reason: ``chi2/chi3 is installed`` (fires on the H side,
which this family admits; ``stepping.update_H`` moved 0 words), ``<volume> is not
allocated`` (fires on EVERY no-PML configuration, 18 distinct volume names) and
``polarization 0 kind 'sellmeier' is outside ('lorentzian','drude')`` (both sides
admitted, 0 words moved). The gate's ``over_coverage`` leg now runs that sweep and
FAILS on any sibling reason it cannot map to a table key, which is the direction
that was missing.

ONE CANDIDATE WAS REFUTED RATHER THAN ADDED, and it is recorded because a reader
will ask: the GHOST-RULE clause (``axis 1 boundary 'mirror' is outside ('periodic',
'metallic')``) looks like a twelfth omission and is not an independent one.
``COVERED_BOUNDARIES`` is ``('periodic', 'metallic')`` and the only other kinds
``stepping._boundary_kinds`` can return are ``'mirror'`` and ``'axis'``
(stepping.py:2191-2195), which co-fire with ``fold`` and ``cylindrical_axis``
respectively on every configuration that reaches them. Measured over the whole
sweep: the ghost clause never appears without one of those two. It is subsumed,
not omitted, and the ``over_coverage`` leg maps it onto them by name.

ARM S IS NOT BUILT IN THIS NULL-FAMILY MODULE, and the imported record is about the
Triton sibling that originated it. Metal's own stored-E product now lives separately
in :mod:`.no_pml_stored_e`; keeping it there preserves this module's invariant that a
null plan contains no arithmetic. A stored-E plan has a different residency table
(it writes E without ``f_w_E``), so it must remain a distinct family rather than a
flag on this one.

DISPATCH IS NOT WIRED. ``meep_gpu.fastpath.plan_fast_path`` returns ``None`` on
every branch and this module does not touch it, exactly as on the Triton track
through all nine of its families.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

# ONE DEFINITION, IMPORTED. Nothing about which configurations are null is decided
# in this file; see the module docstring's "WHAT IS IMPORTED AND NEVER RE-DERIVED".
from ..triton_kernels.no_pml_constitutive import (  # noqa: F401 - re-exported
    CONSERVATIVE_CLAUSES,
    NULL_SIDES,
    STORED_E_ARM,
    null_constitutive_coverage,
    stored_e_constitutive_reasons,
)
from .coverage import Coverage, residency_coverage

__all__ = [
    "CONSERVATIVE_CLAUSES",
    "METAL_NON_CLAUSES",
    "NULL_SIDES",
    "NULL_SLOTS",
    "STORED_E_ARM",
    "MetalNullConstitutivePlan",
    "MetalNullConstitutiveStepPlan",
    "metal_null_constitutive_coverage",
    "null_constitutive_coverage",
    "plan_metal_null_constitutive",
    "plan_metal_null_constitutive_step",
    "register_arms",
    "slot_side_reasons",
    "stored_e_constitutive_reasons",
]

#: The two slots this family can fill, and the ``NULL_SIDES`` key each asks about.
#: The arm registry keys on SLOTS and ``NULL_SIDES`` keys on SIDES, so one of the
#: two tables has to do the translation somewhere visible.
#:
#: DERIVED, NOT SPELLED, AND THAT CHANGED 2026-08-15 BECAUSE THE LITERAL WAS A
#: MEASURED, BYTE-VISIBLE HOLE. It used to read ``{"update_H": "H", "update_E":
#: "E"}`` — a SECOND literal beside the imported ``NULL_SIDES``, free to drift from
#: it. Measured with the two values swapped: ``launch.plan_step`` on a no-PML run
#: with ``stores_E`` True filled the ``update_E`` SLOT with a null plan whose own
#: ``sub_step`` said ``update_H``, on a run where ``stepping.update_E`` moves 3240
#: words — the composition silently deleting a real sub-step. The 93-test suite and
#: all eight gate legs stayed GREEN. Deriving it from ``NULL_SIDES[side]["sub_step"]``
#: means the slot a side claims and the side a slot asks about are ONE fact, and
#: :func:`slot_side_reasons` refuses the disagreement fail-closed for the case where
#: someone edits the derivation itself (the gate's ``m5`` mutation does exactly that).
NULL_SLOTS: Dict[str, str] = {
    spec["sub_step"]: side for side, spec in NULL_SIDES.items()}

#: The clauses this family deliberately does NOT carry, each with the reason a
#: refusal would be FALSE where it fired. FIFTEEN of them, and the count is here
#: rather than in the prose above because the prose got it wrong.
#:
#: RECORDED AS DATA BECAUSE THE GATE READS IT, and that sentence only became true
#: this round. It previously claimed "the gate's ``breadth`` leg iterates it: an
#: entry with no case in the leg is a claim nothing measured", while the leg
#: iterated a separate six-case literal — measured 2026-08-15: six of the eleven
#: entries then present (``conductivity``, ``cylindrical_axis``, ``layout``,
#: ``metal_backend``, ``residency_declaration``, ``subnormal_policy``) had no case
#: anywhere, so the table's own stated discipline was exactly what it was not doing.
#:
#: BOTH DIRECTIONS ARE CHECKED NOW, and only one of them was:
#:
#: * TABLE -> CASE. ``gate...leg_breadth`` asserts every key names either a grid
#:   case in the sweep or the check that covers it. An entry with neither FAILS.
#:   This direction alone cannot see an OMITTED entry — an entry that is not here
#:   has no key to be orphaned;
#: * CLAUSE -> TABLE. ``gate...leg_over_coverage`` sweeps the configurations this
#:   family ADMITS, collects every reason both sibling predicates give, and FAILS on
#:   any reason it cannot map to a key here. That is what found the last four
#:   entries, and its marker table is JOINED to this one so deleting an entry goes
#:   red rather than quietly un-explaining a clause.
#:
#: Every entry rests on ONE executed measurement rather than on this prose: the
#: covered sub-step is shown to touch no field array at all (see the module
#: docstring's poison table), so there is nothing for a stride, an index, a ghost
#: rule or a rounding mode to be wrong about.
METAL_NON_CLAUSES: Dict[str, str] = {
    "metal_backend": (
        "torch, MPS availability, compile_shader AND the engine's array module "
        "(coverage._metal_backend_reasons' four requirements) are not required: "
        "this product compiles nothing, launches nothing and mirrors nothing, so "
        "it is correct on a host with no GPU and correct whatever xp the engine "
        "holds. It is the only family in this package whose gate needs no device"),
    "residency_declaration": (
        "a plan built with no residency is refused by every kernel predicate here "
        "because two sub-steps mirroring one volume separately would each hold a "
        "private copy. This plan binds no buffer, so there is no private copy to "
        "hold — and a run with no PML does not even allocate f_w_* or the stored "
        "H the refusal would have named"),
    "subnormal_policy": (
        "coverage._metal_backend_reasons clause 4 refuses a run resolved to "
        "'keep' because the MPS executor flushes natively and has no lever. No "
        "operand, result or intermediate exists on this arm, so both policies are "
        "unreached rather than merely indistinguishable"),
    "storage_width": (
        "complex64 storage changes the stride every kernel indexes with; this one "
        "indexes nothing"),
    "fold": "a mirror plane changes n_a and every cell's coefficient index; no index is formed",
    "cylindrical_axis": "an axial extent moves the coefficient index; no index is formed",
    "bloch_phase": "a Bloch phase multiplies one wrapped plane; no plane is read",
    "beta": "special_kz's beta adds an out-of-plane coupling to a curl this arm does not run",
    "bfast": "BFAST adds a second additive curl term to a curl this arm does not run",
    "conductivity": "a conductivity changes the curl, not the constitutive return at :944/:983",
    "layout": (
        "shape, contiguity and dtype are what a flat index assumes; no flat index "
        "is computed"),
    # ADDED 2026-08-15. All three fire on a configuration this family ADMITS and
    # none of them had an entry, because the table's totality assertion only ever
    # ran table -> case and never clause -> table.
    "nonlinearity": (
        "coverage._grid_reasons clause 10 refuses chi2/chi3 on EVERY sub-step "
        "because the Pade factor REPLACES the constitutive product. It replaces "
        "nothing that runs here: update_H returns at stepping.py:944-945 whatever "
        "is installed. MEASURED on a Fields.set_nonlinear_volumes object (which, "
        "unlike driver.set_chi2/set_chi3, does NOT switch storage on): the null "
        "admits the H side, the sibling refuses it naming the clause, and "
        "stepping.update_H moved 0 words. THE ONE SIDED ENTRY IN THIS TABLE — the "
        "E side is refused, conservatively, and is listed in CONSERVATIVE_CLAUSES "
        "as chi2_chi3_installed_without_stored_e"),
    "volume_allocation": (
        "every kernel predicate here refuses a volume that is not allocated "
        "(clause 13, 'f_w_Hx is not allocated'), because a kernel binds one "
        "pointer per volume. This arm binds none — and the refusal would fire on "
        "SIX of the nine volumes per side on the family's OWN configuration, which "
        "is the same measurement the residency `null` set exists for: H is never "
        "stored without PML (fields.py:664-666), E is served on demand as "
        "D*inv_eps, and f_w_* is allocated only by enable_pml_storage"),
    "susceptibility_kind": (
        "coverage._susceptibility_reasons clause 9b refuses a pole whose KIND is "
        "outside ('lorentzian','drude'), because the ADE three-term recurrence is "
        "transcribed for those two only. No recurrence runs here. MEASURED on a "
        "registered pole of kind 'sellmeier' that drives nothing: BOTH sides "
        "admitted and stepping moved 0 words. A pole that DRIVES an E component is "
        "a different question and is refused on the E side by name"),
    "susceptibility_registration": (
        "coverage.constitutive_coverage's E-side clause refuses a run with ANY "
        "polarization REGISTERED ('update_E's source is (D - sum P), not D'). This "
        "family asks driven() instead, and the difference is not pedantry: a "
        "susceptibility whose sigma is identically zero drives nothing and does NOT "
        "switch storage on (driver.py:1542-1547, 'what makes the zero-strength case "
        "byte-identical rather than merely close'), so update_E really does return "
        "at stepping.py:983 with a pole in the register. Refusing on registration "
        "alone would be a refusal that is FALSE where it fires. MEASURED on a "
        "registered pole with driven() == (): both sides admitted, 0 words moved. "
        "FOUND BY THE over_coverage LEG ITSELF, on the cut that introduced it — "
        "which is the leg working, not the leg being wrong"),
}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def slot_side_reasons(slot: str) -> Tuple[str, ...]:
    """Does the slot->side table agree with the side->slot table it is derived from?

    A BOOKKEEPING CLAUSE, NOT A PHYSICS ONE, and it lives here rather than in the
    predicate for that reason: the shared predicate answers "is this SIDE null on
    this run", and it has no way to know which SLOT the composer is about to put the
    answer in. Crossing those two is a silent, byte-visible over-coverage — measured
    2026-08-15 with the two entries swapped, ``launch.plan_step`` put a null in the
    ``update_E`` slot of a run where ``stepping.update_E`` moves 3240 words, and
    nothing in the suite or the gate went red.

    :data:`NULL_SLOTS` is now DERIVED from :data:`NULL_SIDES`, so the disagreement
    is not expressible without editing the derivation. This clause is what catches
    the edit: it is consulted by every entry point that turns a slot into a side —
    :func:`_arm_coverage`, :func:`_arm_plan` and
    :func:`plan_metal_null_constitutive_step` — and it FAILS CLOSED, returning a
    refusal reason rather than raising, so a crossed table costs the slot its
    substitution instead of crashing a composer that would otherwise have stepped
    correctly on the array path.
    """
    side = NULL_SLOTS.get(slot)
    if side is None:
        return (f"{slot!r} is not one of {tuple(NULL_SLOTS)}",)
    claimed = NULL_SIDES[side]["sub_step"]
    if claimed != slot:
        return (f"the slot table maps {slot!r} to side {side!r}, whose sub_step is "
                f"{claimed!r} (NULL_SIDES): a null built for one sub-step and "
                f"placed in the other's slot DELETES a sub-step the array path "
                f"performs — refused rather than composed",)
    return ()


def metal_null_constitutive_coverage(fields: Any, pml: Any, side: str,
                                     residency: Any = None,
                                     contract_variants: Sequence[str] = (),
                                     ) -> Coverage:
    """The shared predicate, under the signature the Metal arm registry calls.

    ``residency`` and ``contract_variants`` are ACCEPTED AND IGNORED, and that is
    the whole Metal content of this function — see :data:`METAL_NON_CLAUSES`
    entries ``residency_declaration`` and ``metal_backend`` for why ignoring each
    of them is CORRECT rather than lax. They are in the signature so this family's
    predicate has the same shape as every other Metal family's, which is what lets
    :func:`_arm_coverage` be a two-line adapter over the registry's
    ``(context, slot)`` call instead of a special case in the composer.

    They are not silently dropped keyword arguments in the the rename-outright rule sense:
    rule 6 forbids keeping a RETIRED name alive as an accepted-and-ignored alias.
    These are live parameters of a shape this arm has a measured reason to ignore,
    and the reason is recorded where a reader will find it.
    """
    del residency, contract_variants  # see METAL_NON_CLAUSES
    return null_constitutive_coverage(fields, pml, side)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalNullConstitutivePlan:
    """The plan for a sub-step that does nothing: it also does nothing.

    IT DOES NOT SUBCLASS :class:`..plans.KernelPlan`, and the reason is that the
    base's ``run`` is exactly wrong here. ``KernelPlan.run`` looks the requested
    contraction mode up in ``_functions`` and RAISES when it is absent — which is
    the right contract for a kernel plan, because a guard selector that silently
    fell back to the pinned source would make the gate's guard leg vacuous. A null
    holds no function under any mode, so inheriting that ``run`` would make every
    launch raise ``KeyError``. It implements the same PROTOCOL instead — ``run``,
    ``variants``, ``volumes``, ``__repr__`` — so a composer never special-cases
    the call, and overrides nothing it would have to fight.

    NOT ``launch.NoopPlan`` either, and the distinction is why this class exists at
    all rather than reusing that sentinel: ``NoopPlan`` is what a FUSED PAIR leaves
    in the slot of a sub-step another kernel absorbed — it means "the work
    happened, over there". This one means "there was no work", which is a different
    fact about the run and must not be reported as the first.

    ``runs`` IS LOAD-BEARING FOR THE GATE rather than for the engine. A leg that
    certifies "the plan reproduced the array path" by comparing bytes is trivially
    satisfied by a plan that was never invoked. On a kernel family the launch
    counter is the evidence; here there is no launch, so this is.
    """

    __slots__ = ("side", "slot", "sub_step", "return_site", "runs", "_modes")

    #: What a plan of this family launches. Named for the artifact and for a
    #: reader; the value is the point.
    family = "no-PML null constitutive"

    #: FALSE, and this is the one attribute the composer acts on. A null plan
    #: dispatches nothing, binds no buffer and writes no host array, so it neither
    #: REQUIRES a mirror nor STALES one — the two halves of the residency
    #: invariant, both satisfied vacuously and for the same reason.
    #: ``launch.plan_step`` reads it to put this slot in ``residency_coverage``'s
    #: ``null`` set instead of its ``planned`` set. Read the module docstring for
    #: the measurement that made this necessary; matching on the arm's label would
    #: have worked equally well until someone renamed the label.
    performs_device_work = False

    def __init__(self, side: str) -> None:
        if side not in NULL_SIDES:
            raise ValueError(f"side must be one of {tuple(NULL_SIDES)}, got {side!r}")
        self.side = side
        self.sub_step = NULL_SIDES[side]["sub_step"]
        self.slot = self.sub_step
        self.return_site = NULL_SIDES[side]["return_site"]
        self.runs = 0
        self._modes: list = []

    @property
    def variants(self) -> Tuple[str, ...]:
        """Which contraction modes this plan can launch: none, because it has no source.

        The EMPTY TUPLE IS THE ANSWER, not a missing one. A kernel plan reports the
        modes it compiled; this plan has no source string, so there is no variant
        of one to report, and an artifact that showed ``('off',)`` here would be
        claiming a compiled guard that does not exist.
        """
        return ()

    @property
    def volumes(self) -> Tuple[str, ...]:
        """Which mirrored volumes this plan binds: none.

        THE RESIDENCY CONSEQUENCE, and the reason this attribute is spelled out on
        a plan that binds nothing. A null neither requires a mirror nor stales one,
        so it belongs in ``residency_reasons``'s ``null`` set rather than its
        ``planned`` set. Reported as a declaration on the plan so a composer reads
        the fact rather than inferring it from an empty binding list.
        """
        return ()

    @property
    def modes_requested(self) -> Tuple[str, ...]:
        """Every contraction mode :meth:`run` was asked for, in call order.

        The guard leg's evidence. ``run(contract='fast')`` on this plan is a no-op
        that must not raise — but "did not raise" is indistinguishable from "was
        never called", so the mode is recorded and the leg asserts both arms were
        actually requested.
        """
        return tuple(self._modes)

    def run(self, contract: Optional[str] = None) -> None:
        """Reproduce ``stepping.update_{H,E}``'s no-PML return, exactly.

        ``contract`` is accepted under every value, INCLUDING modes no kernel plan
        in this package was built with, and it changes nothing. That is not
        laxness: the contraction directive gates floating-point contraction, and
        this call performs no floating-point operation to contract, so there is no
        mode under which its result could differ. A ``KeyError`` here — the kernel
        plans' correct behaviour — would be a refusal to perform a sub-step that
        the array path performs by returning.
        """
        self._modes.append("off" if contract is None else contract)
        self.runs += 1

    def __repr__(self) -> str:
        return (f"MetalNullConstitutivePlan({self.sub_step}, side={self.side!r}, "
                f"runs={self.runs}, variants=())")


class MetalNullConstitutiveStepPlan:
    """Which constitutive sub-steps of one inactive-layer run are null, with reasons.

    Carries its own RESIDENCY VERDICT, which is the whole reason a Metal-side step
    plan exists. On a run this family covers, ``{update_H, update_E}`` being null is
    FULL coverage of the step's constitutive arithmetic — and the mirror set is
    legitimately EMPTY, which the residency model has to be told rather than left to
    read as "nothing is mirrored, so nothing is safe".
    """

    __slots__ = ("plans", "refusals", "residency", "live", "synced")

    def __init__(self, plans: Dict[str, MetalNullConstitutivePlan],
                 refusals: Dict[str, Tuple[str, ...]],
                 residency: Optional[Coverage] = None,
                 live: Sequence[str] = (), synced: Sequence[str] = ()) -> None:
        self.plans = dict(plans)
        self.refusals = dict(refusals)
        self.residency = residency
        self.live = tuple(live)
        self.synced = tuple(synced)

    @property
    def covered(self) -> Tuple[str, ...]:
        return tuple(name for name in ("update_H", "update_E") if name in self.plans)

    def run(self, contract: Optional[str] = None) -> None:
        """Run both null sub-steps in the driver's order.

        Unlike a curl composition's ``run``, this one is implementable: what the
        driver does between ``update_H`` and ``update_E`` is the electric withdraw,
        ``step_D``, the electric source injection, ``fill_symmetry_bc_D``,
        ``zero_metal_D`` and ``fill_folded_far_ghosts_D`` (driver.py:3290-3302,
        between the ``update_H`` call at :3289 and the ``update_E`` call at :3304),
        and
        every one of them is a no-op TO A PLAN THAT PERFORMS NO OPERATION — the
        reordering is of this plan past them, not of them past each other, and a
        call that neither reads nor writes commutes with anything. It still counts
        both, so a caller can prove it ran.

        THE LINE NUMBERS ARE THE MEASURED ONES. This docstring cited
        ``driver.py:3212-3225`` for a round; those lines are inside
        ``FdtdDriver.step``'s own DOCSTRING, not its body, so the citation pointed a
        reader at prose about MEEP's source slots rather than at the calls it named.
        """
        for name in ("update_H", "update_E"):
            plan = self.plans.get(name)
            if plan is not None:
                plan.run(contract)

    def __repr__(self) -> str:
        covered = ", ".join(self.covered) if self.covered else "nothing"
        held = ("unasked" if self.residency is None
                else "held" if self.residency.covered else "REFUSED")
        return (f"MetalNullConstitutiveStepPlan(covered=[{covered}], "
                f"residency={held})")


def plan_metal_null_constitutive(fields: Any, pml: Any, side: str,
                                 residency: Any = None,
                                 contract_variants: Sequence[str] = (),
                                 ) -> Optional[MetalNullConstitutivePlan]:
    """Build the null plan for one constitutive side, or None when refused.

    Allocation-free and import-free: no torch, no ``compile_shader``, no source
    string, not even a lazy import inside the function. Every other builder in this
    package compiles at least one specialisation below its predicate; there is
    nothing to compile when there is no kernel, so this one does not.

    ``residency`` and ``contract_variants`` are accepted for the registry's uniform
    call shape and ignored for the reasons :data:`METAL_NON_CLAUSES` records.
    """
    del residency, contract_variants  # see METAL_NON_CLAUSES
    if not null_constitutive_coverage(fields, pml, side).covered:
        return None
    return MetalNullConstitutivePlan(side)


def plan_metal_null_constitutive_step(fields: Any, pml: Any,
                                      residency: Any = None,
                                      live: Optional[Sequence[str]] = None,
                                      synced: Sequence[str] = (),
                                      ) -> MetalNullConstitutiveStepPlan:
    """Ask both sides independently, and answer the residency question correctly.

    THE RESIDENCY VERDICT IS THE POINT. Both null slots are declared through
    ``residency_coverage``'s ``null`` set rather than its ``planned`` set, because
    a null neither binds a mirror nor writes a host array. Declared as ``planned``
    — which is what tranche 1's composer did, having no other set to put them in —
    the verdict REFUSES the family's own configuration, naming volumes that a
    no-PML run does not allocate. Measured, and quoted in the module docstring.

    ``live`` is DECLARED and ``None`` is a REFUSAL, never an empty set, exactly as
    ``residency_reasons`` requires: which seam work runs depends on the sources,
    the walls and the poles, and none of those is readable from here.
    """
    plans: Dict[str, MetalNullConstitutivePlan] = {}
    refusals: Dict[str, Tuple[str, ...]] = {}
    for slot, side in NULL_SLOTS.items():
        crossed = slot_side_reasons(slot)
        if crossed:
            refusals[slot] = crossed
            continue
        verdict = null_constitutive_coverage(fields, pml, side)
        if not verdict.covered:
            refusals[slot] = verdict.reasons
            continue
        plan = plan_metal_null_constitutive(fields, pml, side)
        if plan is None:
            refusals[slot] = ("coverage passed but the plan builder refused",)
            continue
        if plan.slot != slot:
            # Belt and braces on the same fact, and cheap: the plan names the
            # sub-step it stands for, so the composer can check the placement it
            # is about to make rather than trusting the table it read it from.
            refusals[slot] = (f"the plan for slot {slot!r} reports sub_step "
                              f"{plan.slot!r}",)
            continue
        plans[slot] = plan

    verdict = residency_coverage(
        None if residency is None else residency.names,
        planned=(),
        live=live,
        synced=synced,
        null=tuple(plans),
    )
    return MetalNullConstitutiveStepPlan(plans, refusals, verdict, live or (), synced)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

#: The family name this module registers under, and the one an artifact records.
FAMILY = "no_pml_constitutive"


def _arm_coverage(context: Any, slot: str) -> Coverage:
    """The registry's coverage entry point for one null slot.

    :func:`slot_side_reasons` runs FIRST and its refusal is returned as the
    verdict, so a crossed slot->side table costs the slot its substitution rather
    than admitting a sub-step this family never asked about. See that function for
    the measurement.
    """
    crossed = slot_side_reasons(slot)
    if crossed:
        return Coverage(False, crossed)
    return metal_null_constitutive_coverage(
        context.fields, context.pml, NULL_SLOTS[slot],
        context.residency, context.contract_variants)


def _arm_plan(context: Any, slot: str) -> Optional["MetalNullConstitutivePlan"]:
    """The registry's builder entry point for one null slot.

    Refuses (``None``) on a crossed table for the same reason, and refuses again on
    the plan's own ``slot`` if it somehow disagrees — ``_select_slot`` turns a
    ``None`` builder into an unfilled slot, which is the array path.
    """
    if slot_side_reasons(slot):
        return None
    plan = plan_metal_null_constitutive(
        context.fields, context.pml, NULL_SLOTS[slot],
        context.residency, context.contract_variants)
    if plan is not None and plan.slot != slot:
        return None
    return plan


def _arm_gate(context: Any) -> bool:
    """Consult this arm only when the layer is inert.

    A GATE, NOT A COVERAGE CLAUSE, and the distinction is deliberate: it decides
    whether the arm is CONSULTED, and it reads the LAYER rather than the fields,
    which is the one cheap structural question askable before any predicate runs.
    The predicate then asks ``pml.is_active`` again through
    ``stepping._pml_is_active``'s own spelling, so the gate is an optimisation and
    never the verdict — and if the two ever disagreed the predicate would win,
    which is the safe direction.
    """
    from ..triton_kernels.launch import absorber_inactive  # noqa: PLC0415

    return absorber_inactive(context.pml)


def register_arms() -> Tuple[Any, ...]:
    """Register both null slots with :mod:`..arms`.

    THE ARM GETS NO PREFERENCE OF ANY KIND, in either direction. A null preferred
    over a kernel silently deletes a real sub-step; a kernel preferred over the
    null launches the ``dsigw`` accumulation on a step that allocates no ``f_w_*``
    at all. Two admitters leave the slot unselected naming both, which is the
    registry's contract and not this family's choice — and the gate's
    ``overlap`` leg measures the disjointness rather than asserting it.

    ``wired=True``: this arm is consulted by the composer. It is NOT dispatch —
    ``fastpath.plan_fast_path`` still returns ``None`` on every branch and nothing
    here touches it.
    """
    from . import arms  # noqa: PLC0415

    return tuple(
        arms.register(
            family=FAMILY,
            slot=slot,
            label="no-PML null",
            coverage=_arm_coverage,
            plan=_arm_plan,
            prefix="no-PML null: ",
            noun="no-PML null constitutive",
            gate=_arm_gate,
            wired=True,
        )
        for slot in NULL_SLOTS)
