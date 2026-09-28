"""The hand-CUDA NO-PML constitutive arm: a predicate over a sub-step that does nothing.

THE FINDING FIRST, BECAUSE IT IS WHAT THIS FILE IS. This family needs NO CUDA
KERNEL. Not a smaller one, not an emitted one, not a restated predicate over an
already-certified body -- none at all. Without an absorber ``stepping.update_H``
returns before its first statement and ``stepping.update_E`` returns before its
first statement whenever E is not stored, so the sub-step's whole product is a
``return``. There is nothing to transcribe, no grouping to hold, no coefficient to
index, no float to round, and nothing for ``--fmad=false`` or a subnormal policy to
be wrong about. The thing that certifies it is not a bit-identity sweep against a
kernel but an IDENTITY measurement against the array path's own call: every live
device word unchanged across it.

WHY IT IS STILL A PREDICATE, AND WHY IT MUST BE EXPLICIT. In a fail-closed composer
an UNSELECTED slot and an EMPTY slot must not look alike. A slot nobody claimed
means "the array path runs here"; a slot this arm claims means "the array path would
have run here AND performed no operation". Both leave the same bytes behind, and
only the second is a statement about the configuration. Without it a composer
counting covered sub-steps has to infer "there was no work" from "nobody bid", which
is the same shape of inference this package refuses everywhere else -- and the
inference is WRONG the moment a future ``update_E`` grows a term, because nothing
would have refused it.

WHAT MAKES THE ARM UNSAFE IS THEREFORE NOT ARITHMETIC BUT ADMISSION. A null that
admits one configuration on which the array path DOES move is silent corruption:
the composer would skip a sub-step that had work to do and no byte would ever
disagree with a wrong answer, because the wrong answer is "the previous value". So
every clause below is a refusal of a configuration in which ``stepping`` gets past
its ``return``, and the device leg's whole job is to measure that the admitted set
really moves nothing.

WHAT ``stepping`` DOES WITHOUT AN ABSORBER, transcribed with its lines
---------------------------------------------------------------------
``stepping.update_H`` (stepping.py:934-951)::

    if not _pml_is_active(pml):
        return  # H is served on demand from B; ...      # stepping.py:944-945

Unconditional, for every grid, every boundary, every storage width, every feature.
H is never stored without PML and ``Fields.enable_field_storage`` says so in its own
docstring (fields.py:664-665: "H is deliberately NOT allocated: ``update_H`` writes
nothing without PML"); ``Fields.get_H`` serves the B array itself, mu = 1.

``stepping.update_E`` (stepping.py:954-1022)::

    pml_active = _pml_is_active(pml)                     # stepping.py:982
    if not pml_active and not fields.stores_E:
        return  # E is served on demand from D * inv_eps # stepping.py:983-984
    ...
    else:
        getattr(fields, component)[...] = constitutive   # stepping.py:1022

so the sub-step has TWO no-PML shapes and the ``return`` at :983 comes BEFORE every
branch that could read a polarization, a Pade factor, an off-diagonal row, an
inverse epsilon or a PML coefficient. Nothing downstream of that line can make the
call non-null, so the only question a predicate has to answer is which side of :983
the configuration falls on.

``_pml_is_active`` (stepping.py:2498-2506) is ``pml is not None and pml.is_active``,
and an ALL-ZERO-FACE layer is INACTIVE -- ``PML.is_active`` (pml.py:427-434) is
``any(low > 0 or high > 0 ...)``, and its own docstring is the case: "a
zero-thickness table steps bit-identically to ``pml=None``". A ``PML`` object
present in the driver but inert therefore takes THIS path, not the kernels'. That
shared question -- ``pml.is_active`` -- is what makes this arm's admitted set the
exact COMPLEMENT of :func:`coverage.covers_real_pml_constitutive`'s rather than an
approximation of it, and the suite measures the partition rather than asserting it.

THE TWO ARMS OF THE NO-PML ELECTRIC SUB-STEP
--------------------------------------------
* **Arm N (NULL) -- this file.** ``update_H`` on any inert layer, and ``update_E``
  on an inert layer with ``fields.stores_E`` False. Zero kernels, zero launches,
  zero NVRTC compiles.
* **Arm S (STORE) -- NOT BUILT IN THIS PACKAGE, and refused here by name.**
  ``update_E`` on an inert layer with ``stores_E`` True is ``E[...] = (D - sum P) *
  inv_eps`` (stepping.py:1022), a real pointwise store with a real product. It is a
  DIFFERENT kernel from the certified ``update_E_pml_real``, which accumulates
  and also writes ``f_w``; the sibling track measured that the certified body cannot
  be bound into this one (``triton_kernels/no_pml_constitutive.STORED_E_ARM``:
  ``kps=1``/``kms=0``/``f=0`` does not recover a store, because ``0.0 + (-0.0)`` is
  ``+0.0`` and canonicalises every negative zero, 1 differing word of 4). The Triton
  track built it as its own module; the hand-CUDA track has not. See
  :data:`STORED_E_ARM_STATUS`.

WHAT THIS PREDICATE DELIBERATELY DOES **NOT** REFUSE
----------------------------------------------------
Every other predicate in this package refuses the fold beyond two planes, the
cylindrical axis, complex storage, a Bloch phase, beta, BFAST, a non-float32 dtype,
a non-contiguous array, an int32 overflow and (for the curl) a conductivity. Those
clauses exist because a kernel INDEXES something: ``covers_real_pml_constitutive``
refuses Dcyl because "the axial extent moves every coefficient index", and every
array clause exists because a launcher takes a POINTER. A null indexes nothing,
takes no pointer, and reads no coefficient. Carrying those clauses here would refuse
configurations on a hazard that provably cannot exist, and this package's rule is
that a refusal must be TRUE where it fires.

They are therefore ABSENT BY NAME, and the claim is MEASURED rather than argued: the
gate's ``breadth`` leg admits AND byte-checks a folded, a cylindrical, a complex, a
Bloch, a BFAST, a beta, a conductive and a metallic-walled grid -- eight cases, each
required to move zero words over three complete cycles with a non-vacuous census.
WHERE that has been measured is :data:`NULL_CONSTITUTIVE_CONFIRMATION`: on a host
whose ``device`` field is None the leg has run on NumPy only, and the device half is
OWED rather than done. Nothing in this file may be read as a device verdict while
that field is None -- ``test_confirmation_record_is_all_or_nothing`` is what keeps
the two states from blurring.

**THE BACKEND CLAUSE IS ABSENT TOO, AND THAT IS THE ONE ABSENCE A CALLER CAN SEE.**
Every other predicate here opens with ``backend is not CuPy`` because a kernel
launches against device pointers. This arm launches nothing, so it is correct -- and
covered -- on NumPy, and a refusal naming CuPy would be false where it fired. The
consequence is that ``covered`` and the census's ``covered_modulo_backend`` COINCIDE
for this family where for every other one they differ on a host without CuPy, and
that a caller can no longer read ``covered`` as "a CUDA kernel serves this". Nothing
in this package is dispatched by anything -- ``fastpath.py`` is the TRITON dispatcher
and never names ``cuda_kernels``; the only mention of this package anywhere in
``meep_gpu/`` is a comment (fields.py:196), and ``test_package_boundary.py`` pins
that absence in both directions -- so no run is affected today. The statement is here
so that whoever wires it does not have to rediscover it. (What is NOT true any more,
and was until recently, is that ``plan_fast_path`` returns ``None`` on every branch:
it dispatches certified TRITON kernels. That is a fact about the sibling track and
changes nothing here.)

HOW THIS DIFFERS FROM THE TRITON SIBLING, deliberately, in three places
----------------------------------------------------------------------
1. **One reason, not all of them.** ``triton_kernels.no_pml_constitutive`` returns a
   ``Coverage`` carrying every refusal; this package's convention is
   ``(covered, reason)`` with the FIRST refusal, so a configuration that
   unexpectedly stays on the array path says why in one string. That makes clause
   ORDER part of the contract, so the order is fixed here and pinned per clause by
   its own refusal string in ``test_no_pml_constitutive.py``.
2. **chi2/chi3 is read by NAME, not through the property.**
   ``Fields.has_nonlinearity`` (fields.py:966-967) reads ``_chi2_components`` ALONE,
   so a chi3-only run answers False through it. The sibling reads the property; this
   file reads both maps, exactly as :func:`coverage.covers_real_pml_constitutive`
   already does. Neither answer is WRONG for a null (with ``stores_E`` False the
   array path returns at :983 whatever the nonlinearity is), which is why this is a
   difference in conservatism and not a defect found in the sibling.
3. **It takes ``grid``**, for signature parity with every other predicate here, and
   READS NOTHING OFF IT beyond its presence. A run with no grid is not a run this
   arm can answer for; everything else about the grid is one of the absent clauses
   above. ``test_no_pml_constitutive.py`` pins that by admitting a grid stand-in
   whose every attribute access raises.

THE RECORD
----------
* ``meep_gpu/cuda_kernels/test_no_pml_constitutive.py`` -- the merge bar. Every
  clause by its own refusal string, the transcription pins (including the cited
  ``stepping.py`` lines, read back out of the file), the partition against the three
  shipped constitutive predicates, the identity property against the real
  ``stepping`` functions on real ``Fields``, and the mutation battery. It needs no
  GPU, because this family compiles nothing.
* ``parity/meep_gpu/gate_cuda_no_pml_null_constitutive.py`` -- the device
  confirmation. Identity, controls, breadth, overlap, mutations and a whole-driver
  engine leg, on real CuPy arrays.
* :data:`NULL_CONSTITUTIVE_CONFIRMATION` -- what that run measured, on which device.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, Tuple

__all__ = [
    "NULL_CONSTITUTIVE_SIDES",
    "NULL_ELECTRIC_COMPONENTS",
    "NULL_MAGNETIC_COMPONENTS",
    "CLAUSE_EVIDENCE",
    "STORED_E_ARM_STATUS",
    "NULL_CONSTITUTIVE_REACH",
    "NULL_CONSTITUTIVE_CONFIRMATION",
    "covers_no_pml_null_constitutive",
]

#: The two constitutive slots, keyed as :data:`coverage.CONSTITUTIVE_SIDES` is so the
#: two tables read side by side and a composer can hand one ``side`` string to
#: either. The VALUES are deliberately not array names: this arm reads and writes no
#: array, and copying the sibling table's ``targets``/``aux``/``sources`` would
#: suggest it does. ``return_site`` is the ``stepping`` line whose ``return`` makes
#: the sub-step null, and ``test_no_pml_constitutive.py`` reads those lines back out
#: of ``stepping.py`` so a citation cannot drift into fiction.
NULL_CONSTITUTIVE_SIDES: Dict[str, Dict[str, Any]] = {
    "H": {
        "sub_step": "update_H",
        "return_site": "stepping.py:944-945",
        "return_lines": (944, 945),
        # H is never stored without PML (fields.py:664-665), so this side has no
        # second arm and no ``stores_E`` clause.
        "needs_stored_e_clause": False,
    },
    "E": {
        "sub_step": "update_E",
        "return_site": "stepping.py:983-984",
        "return_lines": (983, 984),
        # With stored E the sub-step writes E and this arm refuses; that is Arm S.
        "needs_stored_e_clause": True,
    },
}

#: The components ``update_E`` writes and ``update_H`` writes. Written out rather
#: than imported for the reason every table in this package is: the module imports
#: ``typing`` and nothing else, which is what lets it be exercised and mutated on a
#: laptop. Pinned against ``coverage.CONSTITUTIVE_SIDES`` by the suite.
NULL_ELECTRIC_COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")
NULL_MAGNETIC_COMPONENTS: Tuple[str, ...] = ("Hx", "Hy", "Hz")

#: EVERY CLAUSE, WITH WHAT THE REAL ``stepping`` SUB-STEP MOVED ON ITS OWN NEEDLE.
#:
#: This table exists because "conservative", "shadowed" and "load-bearing" are three
#: different facts about a refusal and none of them can be read off the source.
#: ``moved_words`` is the number of uint32 words the array path's OWN call changed on
#: a seeded ``Fields`` built the way ``needle`` says -- MEASURED on this laptop
#: (NumPy float32, 8x8x8 periodic grid, uniform + a signed-zero plane + dense
#: subnormals, 2026-08-20) -- and ``role`` is what that number makes the clause.
#: ``total_words`` is the denominator over the array set the suite compares (every
#: allocated field volume plus every polarization array), and it is here so that
#: "moved 0" can be read as a measurement rather than as an empty comparison:
#:
#: * ``load_bearing``  -- the array path MOVES here. The clause is the reason this
#:                        arm is not silent corruption.
#: * ``conservative``  -- the array path moves NOTHING here and the clause refuses
#:                        anyway. The null would have been byte-exact; a reader who
#:                        lands on one of these must be told it was not wrong, only
#:                        unbuilt-for, so every such reason string says so where it
#:                        fires.
#: * ``shadowed``      -- on any object the DRIVER can build, an earlier clause has
#:                        already refused, so this one cannot fire FIRST. Kept
#:                        anyway, and the entry records what the array path does on
#:                        the shadowed configuration: where that number is nonzero
#:                        the shadow is load-bearing, and deleting the clause would
#:                        be safe only for as long as the shadow holds.
#:
#: THE SUITE RE-MEASURES EVERY ROW rather than trusting this table, and a row whose
#: measurement stops agreeing fails there. Recording the number instead of the
#: adjective is what makes that possible.
CLAUSE_EVIDENCE: Dict[str, Dict[str, Any]] = {
    "active_layer": {
        "sides": ("H", "E"),
        "role": "load_bearing",
        "needle": "a real PML(thickness=2) with PML storage enabled",
        "moved_words": {"H": 2734, "E": 2715},
        "total_words": {"H": 12288, "E": 12288},
        "why": "update_H/update_E run the dsigw accumulation "
               "(stepping.py:946-951, :1014-1018), which is not a no-op",
    },
    "unreadable_layer": {
        "sides": ("H", "E"),
        "role": "load_bearing",
        "needle": "a layer object whose is_active property raises",
        "moved_words": {"H": "array path raises", "E": "array path raises"},
        "why": "stepping._pml_is_active branches on exactly that attribute "
               "(stepping.py:2498-2506). Defaulting it to the admitting answer is "
               "how a null covers a run the array path is absorbing in; refusing it "
               "converts a crash into a reason a caller can read",
    },
    "stored_e": {
        "sides": ("E",),
        "role": "load_bearing",
        "needle": "Fields.enable_field_storage() with no layer",
        "moved_words": {"E": 1287},
        "total_words": {"E": 4608},
        "why": "update_E takes the store at stepping.py:1022 instead of returning at "
               ":954 -- that is Arm S, which this package has not built. The H side "
               "of the same object moves 0 of 4608: H is never stored without PML",
    },
    "unreadable_stores_e": {
        "sides": ("E",),
        "role": "load_bearing",
        "needle": "a Fields proxy whose stores_E property raises",
        "moved_words": {"E": "array path raises"},
        "why": "stepping.py:983 branches on exactly that attribute; the sibling "
               "track MEASURED a proxy that could not answer it being covered with "
               "no reasons while the run underneath wrote 4618 words",
    },
    "driven_polarization": {
        "sides": ("E",),
        "role": "conservative",
        "needle": "a hand-built Fields carrying a driven PolarizationState with "
                  "storage NOT switched on -- the driver switches it on at "
                  "driver.py:1543-1549, so no driver-built run reaches this and on "
                  "one the stores_E clause has already refused",
        "moved_words": {"E": 0},
        "total_words": {"E": 6144},
        "why": "update_E's source becomes (D - sum P) and update_P closes the step. "
               "Byte-exact where it fires only because stores_E happens to be "
               "False, which is a fact about the object rather than about the "
               "susceptibility",
    },
    "unreadable_polarization": {
        "sides": ("H", "E"),
        "role": "conservative",
        "needle": "a polarization object with no drives() method",
        "moved_words": {"H": 0, "E": 0},
        "why": "an unreadable susceptibility is not a covered one",
    },
    "nonlinearity": {
        "sides": ("E",),
        "role": "conservative",
        "needle": "Fields.set_nonlinear_volumes -- NOT driver.set_chi2/set_chi3, "
                  "which forces stored E at driver.py:2023-2029",
        "moved_words": {"E": 0},
        "total_words": {"E": 3072},
        "why": "update_E's product for those components is the Pade factor "
               "(stepping.py:999-1000). Byte-exact here, and refused anyway",
    },
    "offdiagonal": {
        "sides": ("E",),
        "role": "shadowed",
        "needle": "Fields.set_epsilon_volumes with a surviving row -- which switches "
                  "storage on at fields.py:1253-1255, so the stores_E clause "
                  "refuses first",
        "moved_words": {"E": 1450},
        "total_words": {"E": 4608},
        "why": "THE SHADOW IS LOAD-BEARING and that is the whole reason the number "
               "is here: on the real route update_E moves 1450 words, so a null "
               "that reached this configuration would be WRONG rather than merely "
               "unbuilt-for. The row product reads the other components' volumes "
               "(stepping.py:1001-1008) and belongs to "
               "covers_real_pml_offdiag_constitutive",
    },
    "magnetic_susceptibility": {
        "sides": ("H",),
        "role": "conservative",
        "needle": "a duck-typed polarization whose drives('Hx') is True",
        "moved_words": {"H": 0},
        "total_words": {"H": 3072},
        "why": "update_H returns at stepping.py:944 whatever is registered, so the "
               "null IS byte-exact here. What the clause protects is the CLAIM: the "
               "H-side polarization slot is deliberately empty in this engine "
               "(stepping.py:1406-1407), and a null must not report coverage of a "
               "sub-step that grew a term",
    },
    "pml_storage_behind_an_inert_layer": {
        "sides": ("H", "E"),
        "role": "conservative on H, shadowed on E",
        "needle": "Fields.enable_pml_storage() while the layer handed to the "
                  "stepper is inert",
        "moved_words": {"H": 0, "E": 1287},
        "total_words": {"H": 12288, "E": 12288},
        "why": "get_H would serve a stored H that update_H never writes "
               "(fields.py:695-702, :1189-1191), and the curl families refuse the "
               "configuration, so supplying half a step here would hand the "
               "package's own unsafe configuration a partial substitution. ON THE E "
               "SIDE IT IS NOT CONSERVATIVE AND NOT REACHED: enable_pml_storage "
               "calls enable_field_storage (fields.py:700), so stores_E refuses "
               "first -- and the array path moves 1287 words there, so the shadow "
               "is load-bearing on that side too",
    },
    "no_grid": {
        "sides": ("H", "E"),
        "role": "conservative",
        "needle": "grid=None",
        "moved_words": {"H": 0, "E": 0},
        "why": "a run with no grid is not a run this arm can answer for; its "
               "presence is the only thing the grid argument is read for at all",
    },
}


#: ARM S, the store arm, as a STATUS rather than a plan: nothing in this module
#: consumes it. It is here so the next reader does not have to re-derive the product
#: from ``stepping`` to find out why ``stores_E`` True is refused by name.
#:
#: PARTLY BUILT FOR THE HAND-CUDA TRACK SINCE 2026-08-26, and the split is the
#: point. Two kernels in this package now STORE E with no layer:
#: ``dispersive_kernels.update_E_no_pml_real_dispersive`` (real storage, and it
#: REFUSES ``stores_E`` without a polarization -- dispersive_kernels.py:1096-1102)
#: and ``complex_no_pml_kernels.update_E_no_pml_complex_stored`` (complex storage).
#: What is still unbuilt here is the arm below: real storage, no layer, no
#: susceptibility, no other feature -- plain ``E = D * inv_eps``. The Triton track
#: built that one (``triton_kernels/no_pml_stored_e.py``); this package has no such
#: kernel, no predicate for one and no gate, and the corpus drives zero such rows.
#: Anyone tempted to bind the certified ``update_E_pml_real`` to it with
#: ``kps=1``/``kms=0`` should read
#: ``triton_kernels.no_pml_constitutive.STORED_E_ARM["differs_from_certified_body"]``
#: first: the certified body ACCUMULATES and writes ``f_w``, and the substitution is
#: measurably not a store.
STORED_E_ARM_STATUS: Dict[str, Any] = {
    "sub_step": "update_E",
    "reached_when": ("not _pml_is_active(pml) and fields.stores_E, with real "
                     "storage and no susceptibility on any component"),
    "array_path": "fields.<component>[...] = (D - sum P) * inv_eps  # stepping.py:1022",
    "built_for_cuda": False,
    "built_for_cuda_with_a_polarization":
        "dispersive_kernels.update_E_no_pml_real_dispersive",
    "built_for_cuda_on_complex_storage":
        "complex_no_pml_kernels.update_E_no_pml_complex_stored",
    "built_for_triton": "triton_kernels/no_pml_stored_e.py",
    "refused_here_as": "fields.stores_E is True",
    "why_not_the_certified_kernel": (
        "kernels update_E_pml_real accumulates ((f + kps*src) - kms*prev) and "
        "stores f_w; the arm STORES src and must write no f_w, because without a "
        "layer Fields.drive_field returns the stored E (fields.py:1160-1162) and "
        "f_w_* is not allocated at all"),
}

#: WHAT THIS ARM IS WORTH, MEASURED, AND WHY THE NUMBER IS AN UPPER BOUND.
#:
#: A DELTA, never a project total: 19 slots this arm admits that NO shipped CUDA
#: predicate admits today, out of 410 constitutive slots in the record. Re-derived by
#: ``parity/meep_gpu/replay_null_constitutive_slots.py`` running BOTH this arm and
#: the three shipped constitutive predicates over stand-ins built from the census
#: record's per-row ``configuration`` blocks, after reproducing 250 of 250 recorded
#: verdicts and 250 of 250 recorded first-refusal strings.
#:
#: THE SPLIT IS THE INTERESTING PART, and it is not 14 rows times two sub-steps.
#: ``update_H`` is admitted on ALL 14 no-PML rows -- including the six that force
#: complex storage, because this arm has no storage clause: it reads no array, so
#: there is nothing for a dtype to be wrong about, and the array path's ``return`` at
#: stepping.py:944 is dtype-blind. ``update_E`` is admitted on the 5 rows with
#: ``stores_E`` False; the other 9 are the STORE arm (:data:`STORED_E_ARM_STATUS`),
#: which this package has not built.
#:
#: IT BUYS SUB-STEPS AND NOT WHOLE STEPS, and the replay measures that rather than
#: leaving it to be inferred: ZERO of the 14 rows is covered at every sub-step,
#: because every CUDA curl predicate requires an active layer. The two distinct curl
#: refusals on these rows are "no active PML layer" and "complex64 storage". The
#: hand-CUDA track has no no-PML curl family; the Triton track does
#: (``triton_kernels/no_pml.py``), which is why the sibling's own composition takes
#: four of these rows whole-step and this one takes none.
#:
#: UPPER BOUND, in three named places, all of them in the STAND-IN and none in the
#: predicate: the record does not carry ``Fields._pml_active`` (taken as
#: ``pml_active``, since ``driver.setup_pml`` at driver.py:2250 is the only engine
#: caller of ``enable_pml_storage``), does not carry whether a polarization DRIVES
#: (taken as ``stores_E``, since a driven state forces storage at
#: driver.py:1543-1549), and records ``has_nonlinearity``, which reads
#: ``_chi2_components`` alone (fields.py:966-967) -- so a chi3-only row would be
#: admitted here and refused by the shipped predicate. There is NO array or
#: coefficient tail to skip, which is the caveat every kernel family's replay carries
#: and this one does not: the arm reads no dtype, stride, contiguity or coefficient.
NULL_CONSTITUTIVE_REACH: Dict[str, Any] = {
    "census": ("parity/meep_gpu/results/"
               "cuda_predicate_coverage_2026-08-16_offdiag"),
    "replay": "parity/meep_gpu/replay_null_constitutive_slots.py",
    "measured_utc": "2026-08-20",
    "constitutive_slots_in_the_record": 410,
    "slots_a_shipped_cuda_predicate_admits_today": 203,
    "slots_this_arm_adds": 19,
    "by_side": {"H": 14, "E": 5},
    "slots_claimed_twice": 0,
    "no_pml_rows": 14,
    "no_pml_rows_with_stored_e": 9,
    "no_pml_rows_with_complex_storage": 6,
    "rows_taken_at_both_constitutive_sub_steps": 5,
    "rows_taken_at_every_sub_step": 0,
    "why_no_whole_step_row": (
        "every CUDA curl predicate requires an active layer; the two curl refusals "
        "measured on these rows are 'no active PML layer' and 'complex64 storage'. "
        "This track has no no-PML curl family."),
    "is_an_upper_bound": True,
    "upper_bound_because": (
        "three facts are derived rather than recorded, all in the stand-in: "
        "Fields._pml_active (driver.py:2250), whether a polarization drives "
        "(driver.py:1543-1549), and chi3-only nonlinearity (fields.py:966-967). "
        "There is no array or coefficient tail to skip."),
    "validation": {"verdicts_reproduced": 250, "refusals_reproduced": 250,
                   "slots_excluded_retired_fold_refusal": 160},
}

#: WHAT THE DEVICE RUN MEASURED. Read out of the artifacts the gate wrote; a claim
#: in this file that is not in one of them is a claim nobody measured.
#:
#: CUT 2026-08-20 on the GPU host, GPU 5, an RTX A6000 verified physically empty before
#: the run and reported again after it. BOTH float32 subnormal policies RELEASED --
#: every leg, both policies, all_ok true.
#:
#: THE HEADLINE NUMBER IS A ZERO, and it is the only zero this arm can be judged on:
#: 0 differing uint32 words out of 127,398 compared per policy, across 15 admitted
#: configurations, each run for three complete ``update_H``+``update_E`` cycles and
#: compared against its ORIGINAL snapshot every cycle. Its control is the other
#: half: on the four refused configurations the array path moved thousands of words
#: (up to 9,231 on an active layer) and the predicate refused every one.
#:
#: WHAT THE FLUSH POLICY MEASURED THAT KEEP COULD NOT, and it is a finding rather
#: than a formality. The gate's first cut demanded ``subnormal_words > 0`` on every
#: case and FAILED 15 of 15 under flush -- with every identity comparison already
#: reporting zero differing words. The cause is not the arm: installing the flush
#: policy drives the HOST FPU, so the seed's ``1e-45`` and ``-3e-44`` literals are
#: flushed in NumPy before anything reaches the device. The three census numbers
#: account for each other exactly (``plain_3d_np2``: keep 10080/847/777, flush
#: 9653/1197/0; 1197-847 = 350 negative subnormals became -0.0, 10080-9653 = 427
#: positive ones became +0.0, and 350+427 = 777). So the flush rule is now the
#: STRICTER one -- ``subnormal_words == 0`` is a positive measurement that the
#: policy reached the seeded data -- and it is the gate's ``CENSUS_CONTRACT``.
#:
#: AND THE COMPILE COUNTERS READ A TRUE NUMBER, which is the other trap the sibling
#: gate documented: under keep the stamp records 37 NVRTC compiles with all 37
#: stripped of ``-ftz=true``. Not one of them belongs to this family -- it has no
#: device source -- they are the harness's own CuPy work. A stamp reading zero here
#: would have been indistinguishable from a stamp taken before the legs ran.
NULL_CONSTITUTIVE_CONFIRMATION: Dict[str, Any] = {
    "gate": "parity/meep_gpu/gate_cuda_no_pml_null_constitutive.py",
    "artifacts": ("parity/meep_gpu/results/"
                  "cuda_no_pml_null_constitutive_2026-08-20_v2"),
    "recorded_utc": "2026-08-20T08:51:12Z",
    "host": "the GPU host",
    "device": "NVIDIA RTX A6000 (compute capability 8.6, CuPy 13.5.1)",
    "gpu_index": 5,
    "policies_released": ("keep", "flush"),
    "identity_cases": 7,
    "identity_bit_identical": 7,
    "breadth_cases": 8,
    "control_cases": 4,
    "controls_that_moved": 4,
    "mutation_legs": 5,
    "engine_steps": 24,
    "cycles_per_identity_case": 3,
    "words_compared_per_policy": 127398,
    "differing_words_per_policy": 0,
    "partition_rows": 114,
    "slots_claimed_twice": 0,
    "engine_driven_vs_source_free_differing_words": 9786,
    "merge_bar_suite_on_device": "293 passed (the suite's CuPy fixture row)",
    "compiles": (
        "none from this family: it has no device source. The stamp records 37 NVRTC "
        "compiles under keep, all 37 stripped of -ftz=true, and every one of them is "
        "the gate harness's own CuPy work -- which is exactly why the stamp is taken "
        "AFTER the legs rather than before them"),
    "limits": (
        "an IDENTITY leg certifies nothing about a kernel, because there is no "
        "kernel: no bit-identity sweep, no PTX evidence, no step budget raised. What "
        "it certifies is that the array path performs NO OPERATION on every "
        "configuration this predicate admits, and DOES move on every configuration "
        "it refuses -- which is the only way a null arm can be wrong. It says "
        "nothing about the STORE arm, which is unbuilt here, and nothing about any "
        "curl: 0 of the 14 rows this arm reaches is covered at every sub-step."),
}


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

#: Returned by :func:`_flag` when a yes/no attribute could not be read at all.
_UNREADABLE = object()


def _flag(obj: Any, name: str) -> Any:
    """Read a yes/no attribute, or report that the object could not answer.

    ``getattr(obj, name, False)`` is the obvious spelling and on the two clauses
    this arm rests on it turns "cannot answer" into ADMIT -- the opposite of the
    rule this package applies everywhere else, and the exact defect the sibling
    track measured: a ``Fields`` proxy whose ``stores_E`` raised was covered with no
    reasons while the underlying run's ``stepping.update_E`` wrote 4618 words.

    Absent, raising and ``None`` are all the same answer here -- *none* -- and every
    caller turns that into a named refusal. That is conservative in the safe
    direction: the array path reads the same attribute directly and would raise on
    such an object rather than diverge quietly, so this converts a wrong verdict on
    a malformed input into a refusal a caller can read.
    """
    try:
        value = getattr(obj, name)
    except Exception:  # noqa: BLE001 - an unanswerable question is not coverage
        return _UNREADABLE
    if value is None:
        return _UNREADABLE
    return bool(value)


def _drives(state: Any, component: str) -> Any:
    """Does this polarization drive ``component``? ``_UNREADABLE`` when it cannot say.

    Routed through here rather than called directly because a predicate that RAISES
    where its siblings refuse becomes a crashed run the moment a planner consults
    it -- ``covers_real_pml_ade_component`` already catches ``drives()`` raising for
    the same reason (``coverage.covers_real_pml_ade_component``, whose ``drives()``
    call is wrapped in a try/except with the comment "A RAISE IS NOT A REFUSAL
    unless it is caught here"). COVERAGE.PY IS CITED BY NAME AND NOT BY LINE
    THROUGHOUT THIS FILE: it is under active edit by other work and its line numbers
    move between rounds, so a line citation there would be a citation that rots. The
    ENGINE's files -- stepping, fields, driver, pml -- are cited by line, and the
    suite reads the two load-bearing ones back out of the file.
    """
    method = getattr(state, "drives", None)
    if not callable(method):
        return _UNREADABLE
    try:
        return bool(method(component))
    except Exception:  # noqa: BLE001 - an unanswerable question is not coverage
        return _UNREADABLE


def covers_no_pml_null_constitutive(fields: Any, pml: Any, grid: Any,
                                    side: str) -> tuple:
    """Whether ``stepping.update_{H,E}`` is a NO-OP for this run, so the arm may claim it.

    ``side`` is ``"H"`` (``stepping.update_H``) or ``"E"`` (``stepping.update_E``).
    Returns ``(covered, reason)``; ``reason`` names the FIRST refusal, the convention
    every predicate in :mod:`coverage` uses.

    Positive clauses only, in this order, and the order is part of the contract
    because only the first one fires:

    1. the side is one this arm knows, and there is a grid;
    2. **the layer is inert** -- ``stepping._pml_is_active`` False (the test at
       stepping.py:944 and :982). Both sides;
    3. **E is not stored** -- ``fields.stores_E`` False (stepping.py:983). E side
       only, followed by the three features that reach a different product;
    4. no magnetic susceptibility (H side);
    5. no PML storage switched on behind an inert layer (both sides, conservative).

    Clauses 2 and 3 are the load-bearing pair and each is read through :func:`_flag`,
    so an object that CANNOT answer is refused by name rather than admitted by
    default. :data:`CLAUSE_EVIDENCE` records, per clause, how many words the real
    ``stepping`` sub-step moved on that clause's own needle, and the suite
    re-measures every row.

    NOT clauses, deliberately, every one of which the other predicates in this
    package carry: the backend, the dtype, the contiguity, the shape, the int32
    bound, the ghost rule, the fold, the cylindrical axis, the Bloch phase, beta,
    BFAST and the conductivity. A sub-step that returns before its first statement
    takes no pointer, forms no index and rounds no float. The module docstring says
    why each is absent; the gate's ``breadth`` leg measures it on device.
    """
    if side not in NULL_CONSTITUTIVE_SIDES:
        raise ValueError(
            f"side must be one of {sorted(NULL_CONSTITUTIVE_SIDES)}, got {side!r}")
    spec = NULL_CONSTITUTIVE_SIDES[side]

    # THE ONLY THING THE GRID IS READ FOR. Not its backend, not its shape, not its
    # boundaries -- its existence. A ``Fields`` with no grid has no arrays and is not
    # a run either sub-step could be called on.
    if grid is None:
        return False, "no grid: this is not a run"

    # Clause 2 -- the one clause BOTH sides rest on, and the one that makes this
    # arm's admitted set the exact complement of the kernel predicates' rather than
    # an approximation of it. ``pml is None`` is inert by definition; an object that
    # cannot answer ``is_active`` is NOT a demonstrably inert one.
    if pml is not None:
        active = _flag(pml, "is_active")
        if active is _UNREADABLE:
            return False, ("the layer does not report is_active: "
                           "stepping._pml_is_active (stepping.py:2498-2506) "
                           "branches on exactly that attribute")
        if active:
            return False, ("an active PML layer is installed: update_H/update_E run "
                           "the dsigw accumulation (stepping.py:946-951, :1014-1018), "
                           "which is not a no-op")

    if spec["needs_stored_e_clause"]:
        # Clause 3 -- literally what stepping.py:983 branches on.
        stored = _flag(fields, "stores_E")
        if stored is _UNREADABLE:
            return False, ("fields does not report stores_E: stepping.py:983 "
                           "branches on exactly that attribute")
        if stored:
            return False, ("fields.stores_E is True: update_E writes "
                           "E[...] = (D - sum P) * inv_eps (stepping.py:1022) "
                           "instead of returning at :983 -- that is the STORE arm, "
                           "which this package has not built (STORED_E_ARM_STATUS)")
        # The three features that reach a DIFFERENT product. Each is named
        # separately because each belongs to a different kernel, so a reader who
        # lands on the refusal is told which one. On any driver-built run
        # ``stores_E`` has already refused above -- these fire on hand-built or
        # duck-typed objects, and :data:`CLAUSE_EVIDENCE` records that as
        # ``shadowed`` rather than pretending they are load-bearing.
        for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
            for component in NULL_ELECTRIC_COMPONENTS:
                driven = _drives(state, component)
                if driven is _UNREADABLE:
                    return False, (
                        f"polarization {index} ({type(state).__name__}) does not "
                        "report drives(); an unreadable susceptibility is not a "
                        "covered one")
                if driven:
                    return False, (
                        f"polarization {index} drives {component}: update_E's "
                        "source becomes (D - sum P) and storage is switched on "
                        "(driver.py:1543-1549)")
        # BOTH maps by name, not ``has_nonlinearity``: that property reads
        # ``_chi2_components`` ALONE (fields.py:966-967), so a chi3-only run answers
        # False through it. Same spelling as covers_real_pml_constitutive.
        if (getattr(fields, "_chi2_components", None)
                or getattr(fields, "_chi3_components", None)):
            return False, ("instantaneous chi2/chi3 is installed: update_E's "
                           "product for those components is the Pade factor "
                           "(stepping.py:999-1000). CONSERVATIVE where it fires "
                           "alone -- driver.set_chi2/set_chi3 force stored E "
                           "(driver.py:2023-2029) so a driver-built run has already "
                           "been refused above, but Fields.set_nonlinear_volumes "
                           "does not, and on that object update_E returns at "
                           "stepping.py:983 and the null is byte-exact")
        offdiagonal = _flag(fields, "has_offdiagonal_epsilon")
        if offdiagonal is _UNREADABLE:
            return False, ("fields does not report has_offdiagonal_epsilon; a run "
                           "that cannot say whether update_E forms a row product is "
                           "not a covered one")
        if offdiagonal:
            return False, ("an off-diagonal chi1inv row is installed: the row "
                           "product reads the other components' volumes "
                           "(stepping.py:1001-1008) and belongs to "
                           "covers_real_pml_offdiag_constitutive")
    else:
        # Clause 4 -- the H side's only feature clause, and it is CONSERVATIVE, not
        # vacuous: ``update_H`` returns at stepping.py:944 whatever is registered, so
        # the null IS byte-exact on a state that claims to drive Hx, and this refuses
        # it anyway. What it protects is the CLAIM rather than the arithmetic. A
        # registered magnetic susceptibility would mean the H-side slot MEEP fills
        # after ``update_eh(H_stuff)`` -- which stepping.py:1406-1407 records as
        # deliberately empty -- has grown a term nothing in this package has
        # transcribed, and a null there would report full coverage of a sub-step that
        # now has work to do.
        for index, state in enumerate(tuple(getattr(fields, "polarizations", ()) or ())):
            for component in NULL_MAGNETIC_COMPONENTS:
                driven = _drives(state, component)
                if driven is _UNREADABLE:
                    return False, (
                        f"polarization {index} ({type(state).__name__}) does not "
                        "report drives(); an unreadable susceptibility is not a "
                        "covered one")
                if driven:
                    return False, (
                        f"polarization {index} drives magnetic component "
                        f"{component}: the H-side polarization slot is empty in "
                        "this engine (stepping.py:1406-1407) and a null cannot "
                        "report coverage of a sub-step that grew a term")

    # Clause 5 -- CONSERVATIVE, and labelled so where it fires. ``_pml_active`` is
    # PRIVATE, and unlike the public flags its ABSENCE is a real answer rather than
    # an unreadable one: an object carrying no PML-storage concept has no PML storage
    # switched on. Read permissively on purpose; the clause it feeds refuses either
    # way, and the null would have been byte-exact here.
    if bool(getattr(fields, "_pml_active", False)):
        return False, ("Fields has PML storage enabled while the layer is inert: "
                       "get_H would serve a stored H that update_H never writes "
                       "(fields.py:695-702, :1189-1191) -- refused conservatively to "
                       "match the curl families, not because the null would be wrong")
    return True, "covered"
