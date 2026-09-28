"""The mirror FOLD composed with the special_kz BETA term — both arms.

Both halves are already byte-certified on this backend alone: :mod:`.special_kz`
(the real and Bloch beta curls, and the real arm's constitutive companion) and
:mod:`.symmetry` / :mod:`.folded_complex` (the fold's curl, fill and constitutive
admission). This module is what happens when they meet.

MODELLED ON :mod:`.folded_complex`, WHICH IS THE TEMPLATE THIS FAMILY FOLLOWS and
the only idiom it uses. That module is the package's other "fold composed with a
certified family" and every structural choice here is its: derive the curl source
from the PARENT'S OWN TEMPLATE by inserting one block at a CHECKED anchor rather
than copying the template; express the two other fold deltas as CALLS into
:mod:`.symmetry`'s emitters rather than as text; re-admit the certified
constitutive body under a restated predicate instead of writing a second one; keep
the wide (fold-optional) verdict for a gate and register a separate routing verdict
that makes the fold mandatory; and state every inverted clause by number.

===========================================================================
WHAT COMPOSES, AND WHAT THE COMPOSITION ACTUALLY MOVES
===========================================================================

* the CURL. The beta increment is added to ``curl`` AFTER ``_curl_from_operands``
  (stepping.py:370 / :458) and BEFORE ``_mask_non_owned_cells`` (:397 / :479). A
  FOLD WIDENS THAT MASK — from "metallic at cell 0" to "non-periodic at cell 0 PLUS
  folded-periodic at the top plane" — so the beta term now lands inside a mask it
  did not land inside on the unfolded arm. That is the whole composition: the same
  two multiply-and-subtract lines, in the same place, under one more mask. Both
  parents say so in their own refusal messages (``symmetry`` clause 12 and
  ``folded_complex`` clause 8 each refuse beta because "the beta increment lands
  inside the fold's widened mask"), and this module is the product they name;

* the CONSTITUTIVE PAIR. ``update_H`` (stepping.py:907-924) and the diagonal
  ``update_E`` (:1012-1020) read nothing beta-dependent AND nothing fold-dependent:
  three ``_apply_constitutive_pml`` calls over integer-position coefficients, no
  shift helper, no ghost, no ownership mask. The fold reaches them through the
  STORED EXTENT alone and beta does not reach them at all. This family therefore
  adds NO DEVICE CODE on those two slots — it re-admits ``launch.ConstitutivePlan``
  (real) and ``complex_fields.ComplexConstitutivePlan`` (complex), and only the
  ADMISSION is new. Exactly the move ``symmetry.plan_folded_constitutive`` and
  ``folded_complex.plan_folded_complex_constitutive`` make;

* the ghost rule, the cell-0 mask, the top-plane mask, the Bloch phase block, the
  boundary-code classifier and the stored-extent clause, all IMPORTED from the two
  parents rather than re-derived.

THE FILL SLOTS ARE DELIBERATELY NOT CLAIMED, and that is a decision rather than an
omission. ``fill_symmetry_bc_*`` and ``fill_folded_far_ghosts_*`` copy
``parity * field[plane]`` and touch no beta term at all, so ``symmetry``'s and
``folded_complex``'s fill predicates carry NO beta clause and already admit a
folded beta run — the composition matrix has pinned exactly that for
``fold_real_2d_beta`` and ``fold_complex_2d_beta`` since before this family
existed. Registering a third fill arm here would make both fill slots AMBIGUOUS and
drop them to the array path, which is a coverage LOSS dressed as completeness.

===========================================================================
WHAT WAS MEASURED ON THIS HOST FOR THIS FAMILY, 2026-08-16
===========================================================================

``parity/meep_gpu/probe_metal_folded_beta_tail.py``, arm64 / macOS 26.2 /
torch 2.10.0 / numpy 2.4.3. 48 cases, every value table BAND-FREE and
OVERFLOW-FREE by construction (no lane can reach the subnormal band this platform
flushes, and none can reach an infinity whose difference would be a NaN with an
unspecified payload), ``moved_words`` at full count on every launch, 0 vacuous
cases and 0 surprises against the predictions recorded before the run.

1. THE FILE-SCOPE ``contract(off)`` DIRECTIVE IS MANDATORY AND THAT IS NOW A DIRECT
   MEASUREMENT rather than an inherited rule. (It is spelled ONCE in the package, in
   :func:`.shaders.contraction_pragma`, and ``test_metal_kernels`` fails the build on
   a second spelling ANYWHERE — including in prose, which is how this paragraph was
   caught quoting it.) Under ``contract(fast)`` the SHIPPED
   real tail ``curl - (c * b)`` diverges from the array path in 44 of 1,728 words —
   the same 44 as the explicitly fused ``fma(-c, b, curl)`` spelling. Under
   ``contract(off)`` both the shipped spelling and the explicit fma are separated:
   shipped 0/1,728, fused 44/1,728. The pragma is what makes the beta tail's
   POSITION between the curl and the two masks survive compilation.

2. NEGATION IS A SIGN-BIT OPERATION HERE, RE-MEASURED AND NOT INHERITED. On the
   exhaustive band-free table: ``curl + (-(c*b))`` 0/1,728 and
   ``curl + ((c*b) * -1.0f)`` 0/1,728, while ``curl + (0.0f - (c*b))``
   MISSES 22/1,728. The complex twin agrees — ``float2(0,0) - z`` misses 78/41,472
   where ``-z`` is exact. Triton spells its negations ``* -1.0`` because *Triton*
   lowers unary minus as ``0.0 - x`` and canonicalizes a zero's sign; that reason
   does not reproduce on Metal and the workaround is not ported. The refuted
   spelling is a gate mutation, not a comment.

3. THE REAL MULTIPLY'S OPERAND ORDER IS A MEASURED NULL (``curl - (b * c)``,
   0/1,728) — float multiplication is commutative — so the shipped order is kept
   for transcription fidelity to stepping.py:811 and not for a bit reason.

4. THE COMPLEX ORIENTATION IS A NULL **ONLY WHILE THE COEFFICIENT'S REAL WORD IS AN
   EXACT ZERO**, and the scope is MEASURED rather than argued. With the shipped
   purely-imaginary coefficient (stepping.py:798-799 multiplies by ``+/-1j``, so the
   real word is a signed zero Python's own complex multiply produced),
   ``c_mul(partner, coefficient)`` differs from ``c_mul(coefficient, partner)`` in
   0/41,472 words; with a GENERAL coefficient the same swap misses 1,080/41,472 at
   up to 8 ulps. Under FMA_V1 the imaginary word is ``fma(z.x, p.y, z.y * p.x)`` and
   the swap fuses a different product — harmless only because ``z.x * p.y`` is exact
   when ``z.x`` is a zero. The shipped orientation is the array path's (coefficient
   LEFT) and stays; the null is recorded WITH its scope so no later edit reads it as
   a general licence.

5. THE ZERO CROSS TERMS STAY SPELLED even though the real word is zero. Folding the
   product to ``float2(-(bpi*b.y), bpi*b.x)`` — the shortcut the orientation null
   invites — misses 78/41,472 under ``contract(off)``. A zero real word does not
   make the cross terms free; it makes them exact, which is a different thing.

6. THE TWO BETA WORDS STAY BOUND UNIFORMS. A ``constant float&`` scalar and the same
   value baked into the source as a decimal literal produced IDENTICAL words —
   0/144 at four coefficients under both contraction modes — so specialising the
   coefficient into the string would buy a source variant per beta value and change
   nothing. See "the arity verdict" below.

7. THE BETA TERM'S POSITION AGAINST **BOTH** MASKS IS REFUTABLE, which is what makes
   the ordering claim above a claim about the code rather than about the comment.
   Against the array path's order (beta, then the cell-0 mask, then the top-plane
   mask): moving the beta term after the cell-0 mask diverges in exactly 1 word,
   after both masks in exactly 2, and omitting the top-plane mask in exactly 1 —
   the structurally correct counts for a one-dimensional stand-in whose only masked
   lanes are the first and the last.

THE ARITY VERDICT, AND WHAT IT DOES AND DOES NOT LICENSE HERE. The round's
transcription settled ``probe_metal_dynamic_loop_arity.py`` as **IDENTICAL** — a
runtime-bounded loop is bit-identical to its unrolled twin on this backend, 48
cases and 0 differing words. THIS FAMILY HAS NO ARITY AXIS: the beta term is two
fixed multiply-and-subtract lines per curl sub-step, not a reduction over a variable
number of terms, so there is no trip count to make runtime and the verdict is not
load-bearing for the kernel body. What the verdict DOES license is the scalar-shaped
version of the same question, and this family makes that choice deliberately: the
two beta coefficients and the six Bloch phase words are bound as ``constant float&``
uniforms and are never specialised into the source string. Leg 3 above is the direct
measurement for the scalar shape (0/144, four coefficients, both contraction modes),
so the choice rests on its own measurement and merely AGREES with the arity round
rather than borrowing from it. Everything that IS specialised here — the boundary
codes, the sub-step direction, the phased flags, the top-plane mask — is a BRANCH
axis with a bounded arm count, which is what source specialisation is for.

THERE IS NO GENERATED-CODE AUDIT ON THIS BACKEND. ``torch.mps.compile_shader``
exposes no AIR, no GPU ISA and no optimisation report (``device.py:46-62``), so this
family cannot refuse a compile whose emitted code violates the contraction policy
and cannot establish that the pragma was obeyed. Every leg is BEHAVIOURAL: it
catches a wrong answer, never a wrong instruction. That is this certification's one
weakness against the Triton twin's and it is stated rather than buried.

===========================================================================
THE INVERTED CLAUSES — six families to stay disjoint from
===========================================================================

``registry.py``'s contract is that every family's grid-reason list carries the same
numbered questions and answers exactly one of them THE OTHER WAY, and that a new
family STATES its inversion rather than inheriting disjointness. This family is a
product of TWO levers and each lever already has families on both sides of it, so
it carries TWO inversions on every slot it claims — and they are the SAME two on
both arms.

CLAUSE 5 (fold) IS INVERTED AGAINST THE TWO BETA FAMILIES:

* ``special_kz._beta_real_grid_reasons`` refuses a mirror plane BY NAME ("the folded
  beta curl is a separate family; this kernel does not carry the fold"), plus one
  line per folded axis;
* ``special_kz._beta_complex_grid_reasons`` refuses it by name too;
* HERE a fold is REQUIRED, through :func:`_requires_a_fold`, which asks TWICE — once
  of the grid (``has_symmetry`` and some ``is_mirrored`` axis) and once of the
  RESOLVED CODES (some axis landed on a mirror code). The two can disagree, and the
  disagreement is the cylindrical hazard: ``_boundary_kinds`` puts ``is_axis``
  AHEAD of ``is_mirrored`` (stepping.py:2191-2196), so a folded r axis reports
  CYL_AXIS and the fold vanishes silently.

CLAUSE 12 (beta) IS INVERTED AGAINST THE TWO FOLD FAMILIES, AND BOTH OF THEM
PRE-REGISTERED THIS INVERSION:

* ``symmetry._folded_curl_grid_reasons`` clause 12 requires ``grid.beta == 0`` and
  names "the folded special_kz product" as the owner of the slots it declines;
* ``folded_complex._folded_complex_grid_reasons`` clause 8 does the same and names
  "the folded beta product";
* HERE ``grid.beta`` must be NONZERO, refused by name above it. A beta = 0 folded
  run belongs to the parents; this family exists only for the intersection.

CLAUSE 2 (storage) SPLITS THE TWO ARMS OF THIS FAMILY FROM EACH OTHER, in the same
direction and by the same predicate every other pair in the package uses: the real
arm refuses ``force_complex_fields``, the complex arm requires
``force_complex_fields OR has_bloch``. NOTE THE TWO SPELLINGS ARE NOT THE SAME
PREDICATE — ``folded_complex`` records the same asymmetry — and a run with
``has_bloch`` true and ``force_complex_fields`` false is separated in BOTH
directions anyway, because the real arm additionally refuses a nonzero ``k_point``
outright (clause 7) and the two layout clauses pin different dtypes. A volume has
one dtype, so one of the two must refuse any concrete run.

CLAUSE 9 (per-axis Bloch legality, INCLUDING the fold clause) IS IMPORTED FROM
``folded_complex`` rather than restated: a folded axis may carry NO phase and NO
nonzero k component, zone edge included, because
``driver._require_bloch_is_representable`` (driver.py:1047-1075) refuses the
configuration outright and a kernel cannot lift what the array path will not run.
Neither ``special_kz``'s complex arm (which never sees a fold) nor ``symmetry``
(which never sees a phase) asks this. The two corpus rows that drive it —
``TestEigCoeffs.test_binary_grating_special_kz_*`` — carry ``kx != 0`` on the
UNFOLDED x axis with the fold on y, which is legal and admitted.

CLAUSE 5b (no k on the INVARIANT axis) IS KEPT FROM ``special_kz``: beta IS that
axis's analytic dependence and ``Grid`` refuses the pairing at construction. Restated
here, never inferred from the constructor guard.

CLAUSE 4 (cylindrical) IS REFUSED TWICE, for the reason both parents give: the grid
flag alone is not the inversion, because ``_mirror_phases`` (stepping.py:2346-2366)
puts ``(-1)**grid.m`` into the SAME SLOT the mirror phase occupies.

SEPARATION FROM THE REMAINING FAMILIES, each by a clause that names them: the
shipped real ``pml_curl``/``constitutive`` and ``bfast_curl`` refuse beta through
``coverage._grid_reasons`` clause 12 and refuse a fold through clause 5;
``complex_fields`` refuses both; ``no_pml_constitutive`` inverts on the ABSORBER (it
requires an INACTIVE one, this family an active one); ``offdiag_update_e`` requires a
LIVE off-diagonal chi1inv row under REAL storage, which this family's real arm
refuses OUTRIGHT (stepping.py:800-810 raises and MEEP fields.cpp:548-549 aborts on
that combination) and whose complex arm refuses on the E side by name;
``folded_offdiag_update_e`` requires the same live row.

===========================================================================
WHAT THIS FAMILY DOES NOT CLAIM, each by name rather than by omission
===========================================================================

An UNFOLDED beta run (both parents' arms own those slots); a folded beta run at
``beta == 0`` (the two fold families); a registered susceptibility on ``update_E``
under either storage (the source becomes ``D - sum P``); an off-diagonal chi1inv row
on ``update_E`` under complex storage (the folded complex off-diagonal beta product,
unbuilt) and under real storage the whole run is refused because the array path
raises; a conductivity; a nonlinearity; BFAST; cylindrical coordinates; ``update_P``,
which this family does not touch at all and whose "no arm is registered" assertion
therefore stands unchanged.

REGISTERED-BUT-CANNOT-WIN IS NOT USED HERE, and the absence is worth stating.
``registry.py`` keeps ``registered`` and ``wins`` separate so a missing capability
gets NAMED at composition time instead of falling silently to the array path, and
this family would have used it if one of its eight slots' products were unported.
All eight ARE ported. The gaps it DOES have are named inside predicates that would
otherwise admit — the folded complex off-diagonal beta ``update_E``, and the folded
dispersive beta ``update_E`` — which is the same naming through a cheaper mechanism:
a registered-but-losing arm would report the same sentence from one slot further out.

THE ONE GAP THIS FAMILY MAKES VISIBLE AND DOES NOT CLOSE. ``special_kz``'s complex
arm registers a constitutive pair that always refuses, because no complex
constitutive had been restated for a beta run. This family restates it FOR THE
FOLDED CASE, which leaves the UNFOLDED complex beta constitutive as the only
remaining beta gap on this backend — two slots, one corpus row
(``TestSpecialKz.test_special_kz``), closable by the same restatement one level out.
``special_kz``'s refusal message is narrowed in this change to say so rather than to
keep claiming that no complex constitutive admits any beta run.

EXPANSION ARM: BOUND FROM ``special_kz``'S PROBE, NOT A NEW ONE. The complex arm's
arithmetic is ``special_kz``'s complex beta curl arithmetic exactly — the same
``c_mul`` with the same purely-imaginary coefficient on the LEFT, plus the same
field-left Bloch rotation — so the pattern set that licenses that family licenses
this one, and cutting a second artifact would record a second choice rather than a
second measurement. The fold contributes no complex multiply of its own: its only
addition is an ownership mask, which assigns a literal zero. (``folded_complex``
needs its OWN artifact because its FILL multiplies by the mirror parity as a complex
coefficient; this family registers no fill.)

===========================================================================
WHAT IS CERTIFIED, AND WHAT IS NOT — measured 2026-08-16, stated as measured
===========================================================================

MEASURED, on this host, on a FROZEN COPY of the tree (the live tree was being
written by three concurrent workflows and its hash moved during the run, so a
live-tree number would not have been a measurement):

* ``meep_gpu/test_metal_folded_beta.py`` — 52 passed. Seven WHOLE-STEP legs walk the
  driver's real pass list (``step_B``, both fills, ``update_H``, ``step_D``, both
  fills, ``update_E``) for six complete steps against ``stepping.py``: four real
  (periodic and metallic terminations, odd parity, an X fold) and three complex
  (periodic, metallic, and a Bloch phase on the unfolded axis). **202,176 uint32
  words compared, 0 differing**, with every slot's launch counter asserted at the
  exact count its plan owes per cycle and a vacuity floor on every leg. THE FILL
  PASSES IN THOSE LEGS ARE THE PARENTS' — this family registers none — so the legs
  also measure that the parents' fills compose with this family's curls, which is
  the claim the "no fill arm" decision rests on;
* four armed must-catch mutations, all CAUGHT (the beta term moved after the cell-0
  mask, after both masks, the top-plane mask dropped, the two sign coefficients
  swapped); one declared EQUIVALENCE measured null (the real multiply's operand
  order); one spelling proved UNOBSERVABLE ON THIS FAMILY'S GRIDS with both halves
  measured — see below. A control launches the UNMUTATED source through the same
  from-arrays route and requires 0, so a null is never the harness;
* the composition sweep over the whole matrix: **4,770 predicate evaluations, 360
  admissions, 2 overlaps — both the PRE-EXISTING PLANTED ones**, no new ambiguity,
  no ``UNCARRIED`` violation, no unregistered-slot breach, and every arm label wins
  at least one row (``folded beta real`` 11, ``folded beta complex`` 8);
* all 36 checked-in SHIPPED kernel-source hashes UNCHANGED. This family's own
  sources carry no ``fingerprints.json`` row, for the reason ``folded_complex``
  states: they are a function of the PROBE-BOUND arm, so a checked-in hash would
  record a choice rather than a measurement.

THE DEDICATED GATE NOW EXISTS AND IS GREEN —
``parity/meep_gpu/gate_metal_folded_beta.py``, artifact
``results/metal_folded_beta_2026-08-16/gate.json``, run on a FROZEN COPY whose hash
was identical before and after. **3,845,376 uint32 words compared, 0 differing**, over
fourteen legs and a fourteen-case matrix. What it adds over the sibling tests, each
because the tests could not reach it:

* the DRIVER'S REAL TEN-PASS LIST, ``zero_metal_B``/``_D`` included, for twelve
  complete steps per case. A LIVE WALL row (``boundaries={"x": "metallic"}``) makes
  the wall pass do real work, and on such a grid BOTH parents' fill predicates refuse
  by name — so three passes per half run on the array path between device launches,
  bracketed by explicit syncs over the volumes ``coverage.SUB_STEP_VOLUMES`` names,
  with ``Residency.verify()`` asserted empty after every step;
* twelve ARMED must-catch mutations, ALL CAUGHT, and four declared nulls all null.
  Three of the twelve are only reachable at whole-step granularity or off the source
  entirely: a STALE MIRROR (the host pass's write never carried back), the WALL-CLEAR
  SEAM (moved after the far fill), and the beta pair built from the OTHER sub-step;
* the contraction guard MEASURED PER ARM: the real arm's ``contract(fast)`` build
  diverges on every case (104-483 words) while the complex arm is a null, backed by a
  NAIVE-arm control that diverges exactly where a GENERAL complex product is live;
* the subnormal WINDOW per case per step, with the scaled control that makes it fire.
  ONE FINDING CORRECTS AN ASSUMPTION THIS MODULE DID NOT MAKE BUT COULD HAVE: at a
  1e-30 state scale the COMPLEX case enters the band at step 6 of 12 while the REAL
  case stays clean for all 12, on the same shape — band entry is a RUN-and-WINDOW
  fact, not a family fact;
* CONTAINMENT against the Triton track, every slot named: 56/56 admitted by BOTH,
  0 Metal-only, 0 Triton-only, 0 neither — modulo two host-fact clauses that are
  named and counted, because those predicates cannot run on this host at all.

ONE MEASUREMENT THE GATE MADE THAT THE SIBLING TESTS' SEEDING HID.
``metal_composition_matrix._fill`` assigns a REAL array into every volume, so a
complex64 volume comes back with an IDENTICALLY ZERO IMAGINARY PART and every complex
product in this family's kernel then carries an exact-zero operand. Measured on
``complex_bloch_x``: with that seeding the WRONG expansion arm (NAIVE) reproduces the
array path word for word and the contraction control cannot fire; with an independent
imaginary part seeded, NAIVE misses 5-8 words per sub-step and the control moves 4-11.
The gate seeds it and asserts a floor; a complex leg run on a fresh matrix build alone
is weaker than it looks.

TWO REFUTED SPELLINGS ARE UNOBSERVABLE HERE, AND THAT IS PROVED RATHER THAN ASSUMED.
``curl + (0.0f - x)`` differs from ``curl - x`` in EXACTLY ONE input pattern —
``curl`` a NEGATIVE zero and ``x`` a positive zero — and the two spellings ARE
different on this backend (measured on that exact pattern: they disagree on that one
lane and agree on the three control lanes). But ``grid.beta`` is legal ONLY on an
effective-2-D grid, whose invariant axis stores ONE cell, so the z-shifted operand IS
the centre operand and ``b - b_z`` is ``+0.0`` for every finite ``b``; a float sum
carrying a ``+0.0`` addend is never ``-0.0``, so ``curl0``/``curl1`` cannot BE a
negative zero on any grid this family is admitted on. Recorded as "unobservable
here, and here is why" rather than as an equivalence, because the two are different
claims and only the first is true.

THE SAME MECHANISM COVERS PARAGRAPH 5'S CROSS-TERM FOLD, and the gate is what
established it. Dropping the zero cross terms misses 78/41,472 words in the PROBE's
isolated tail — but every word it can change is the SIGN OF A ZERO in the beta
increment, and ``curl - (+/-0.0)`` is one word unless ``curl`` is a negative zero.
``gate_metal_folded_beta.py``'s leg ``spellings`` measures both halves: the spellings
really do differ (80 words over four coefficient sign combinations), and the beta-
touched curls carry 0 negative zeros over 46,272 exported words on 28 case/sub-step
pairs with signed zeros PLANTED into the source volumes. Its control is ``curl2``,
which the beta term never touches and which the argument above does not cover: that
one DOES reach ``-0.0`` (60 words across the matrix), so the zeros on ``curl0`` and
``curl1`` are a measurement rather than a blind spot. The cross terms STAY SPELLED
anyway — a spelling kept for transcription fidelity costs nothing and a future grid
shape could make the difference reachable.

WHAT IS NOT DONE, by name rather than by omission:

* ``fingerprints.json`` IS NOT RE-CUT. Only HOST-module hashes moved (measured: ten
  of them, four belonging to concurrently landing families that are not this one's),
  and the re-cut is a deliberate act that would claim their changes too. It is owed
  once the round's families have all landed;
* THE CORPUS SLOT COUNT IS SHAPE-MATCHED, NOT CORPUS-LIFTED. The sixteen slots this
  family targets are the four rows named in
  ``results/metal_coverage_tranche3_2026-08-16/metal_vs_triton.txt`` —
  ``TestSpecialKz.test_eigsrc_kz_1_real_imag`` (real) and
  ``TestSpecialKz.test_eigsrc_kz_0_complex`` plus
  ``TestEigCoeffs.test_binary_grating_special_kz_{0,1}`` (complex, the last two with
  ``kx != 0`` on the unfolded axis) — each a 2-D y fold over a periodic outer
  declaration under an active PML, and each has a matrix row here carrying the same
  flags that composes on all four arithmetic slots. A full corpus re-sweep is the
  authoritative confirmation and it has NOT been run in this change.

Import contract: importable WITHOUT torch. The predicates and the plan builders (to
``None``) must answer on a host with no GPU, which is the merge bar.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..triton_kernels.coverage import (
    CONSTITUTIVE_SIDES,
    CURL_SUB_STEPS,
    CURL_TARGETS,
    Coverage,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _layout_reasons,
    _susceptibility_reasons,
)
from . import complex_fields, shaders, special_kz, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import Residency, compile_source
from .launch import SUB_STEPS
from .plans import KernelPlan

# ONE DEFINITION, IMPORTED FROM THE PARENTS. Every name below is a fact this family
# must not re-derive: the four boundary codes and their MIRROR split, the near
# ghost's source row, the fold classifier, the reduced-code mapping the certified
# emitters take, and the top-plane mask itself. Re-deriving any of them here would
# be a second place to get the fold's single point of failure wrong.
from .symmetry import (  # noqa: F401 - re-exported deliberately
    CODE_METALLIC,
    CODE_MIRROR_METALLIC,
    CODE_MIRROR_PERIODIC,
    CODE_PERIODIC,
    MIRROR_CODES,
    MIRROR_SOURCE_INDEX,
    _has_real_fold,
    folded_axis_kinds,
    folded_top_plane_mask,
)

# Clause 9 (per-axis Bloch legality WITH the fold clause) and the stored-extent
# companion, imported from the family that established them rather than re-asked.
from .folded_complex import _folded_phase_reasons, _stored_extent_reasons

#: The two arm names the registry carries. Two, not one, because clause 2 splits
#: them and the composition sweep must be able to see each refuse the other's runs.
FAMILY_REAL = "folded_beta_real"
FAMILY_COMPLEX = "folded_beta_complex"

#: The labels ``plan_step`` reports and the composition matrix pins.
LABEL_REAL = "folded beta real"
LABEL_COMPLEX = "folded beta complex"

__all__ = [
    "ARMS",
    "FAMILY_COMPLEX",
    "FAMILY_REAL",
    "FoldedBetaBlochPmlCurlPlan",
    "FoldedBetaPmlCurlPlan",
    "LABEL_COMPLEX",
    "LABEL_REAL",
    "enumerate_folded_beta_sources",
    "folded_beta_bloch_curl_source",
    "folded_beta_bloch_curl_template",
    "folded_beta_bloch_pml_curl_coverage",
    "folded_beta_composition_bloch_curl_coverage",
    "folded_beta_composition_curl_coverage",
    "folded_beta_complex_constitutive_coverage",
    "folded_beta_constitutive_coverage",
    "folded_beta_curl_source",
    "folded_beta_curl_template",
    "folded_beta_pml_curl_coverage",
    "plan_folded_beta_bloch_pml_curl",
    "plan_folded_beta_bloch_pml_curl_from_arrays",
    "plan_folded_beta_complex_constitutive",
    "plan_folded_beta_constitutive",
    "plan_folded_beta_pml_curl",
    "plan_folded_beta_pml_curl_from_arrays",
    "register_arms",
]


# ---------------------------------------------------------------------------
# K1/K2 — the two curl sources, DERIVED from special_kz's certified templates
# ---------------------------------------------------------------------------

#: The anchor the top-plane mask is inserted after: the parent beta template's
#: cell-0 ownership mask slot. Checked to appear EXACTLY ONCE in each parent
#: template, so an edit that moves or duplicates it is a build failure rather than a
#: kernel that masks the wrong plane.
_MASK_ANCHOR = "__MASK__\n"

#: The block the fold adds, and the ONLY text in either kernel that neither parent
#: emits. Storage-neutral: the ZERO LITERAL is not here — it rides in through
#: :func:`.symmetry.folded_top_plane_mask`'s ``zero`` parameter, which is ``0.0f``
#: for the real arm and ``float2(0.0f, 0.0f)`` for the complex one (the array path
#: assigns a complex zero to BOTH planes, stepping.py:1943, :1949).
#:
#: THE ``bool last_*`` LINE IS THE SAME LINE ``folded_complex._TOP_MASK_BLOCK``
#: CARRIES, and ``test_metal_folded_beta`` asserts the two are character-identical.
#: Three families now declare those flags; a test is what stops the third from
#: drifting off the first two.
_TOP_MASK_BLOCK = """
    // --- ownership mask, the TOP plane of a folded PERIODIC axis ----------------
    // The COMPLEMENT of the block above: every target whose Yee shift is 1 there.
    // `_mask_non_owned_cells`' `if iyee[axis] != 0` arm (stepping.py:1887-1897)
    // zeroes that slot because it sits half a cell past MEEP's `big_corner`, outside
    // `owns` (vec.cpp:445-462), so the FILL PASS writes it and the curl must not.
    // On a folded METALLIC axis the top plane IS stepped and no line is emitted.
    bool last_x = (i == nxi - 1), last_y = (j == nyi - 1), last_z = (k == nzi - 1);
__TOP_MASK__
"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 1887-1897->1934-1944


def _with_top_mask_slot(template: str, owner: str) -> str:
    """One parent template with the top-plane slot inserted after its cell-0 mask.

    DERIVED RATHER THAN COPIED, and that is the composition made structural. A copy
    would be a second home for twenty-two (real) or thirty (complex) bindings, the
    ghost gather, the curl grouping, the beta insert and the split-field recurrence
    — every one of which is certified in :mod:`.special_kz` and none of which the
    fold changes. Deriving means an edit to the certified beta curl reaches this
    family in the same commit, and means the only text this module can be blamed for
    is :data:`_TOP_MASK_BLOCK`.

    The anchor is checked to occur exactly once. A parent edit that moved,
    duplicated or removed the cell-0 mask slot would otherwise silently produce a
    kernel with the top-plane mask in the wrong place — a plane of wrong values on
    every folded PERIODIC run, and not a crash.
    """
    occurrences = template.count(_MASK_ANCHOR)
    if occurrences != 1:
        raise RuntimeError(
            f"{owner} carries the ownership-mask anchor {_MASK_ANCHOR!r} "
            f"{occurrences} times, not once; the folded beta curl inserts its "
            f"top-plane mask after that anchor and cannot place it unambiguously. "
            f"Re-read the parent template before editing this family")
    return template.replace(_MASK_ANCHOR, _MASK_ANCHOR + _TOP_MASK_BLOCK, 1)


def folded_beta_curl_template() -> str:
    """``special_kz._BETA_CURL_TEMPLATE`` with ONE slot inserted, and nothing else."""
    return _with_top_mask_slot(special_kz._BETA_CURL_TEMPLATE,
                               "special_kz._BETA_CURL_TEMPLATE")


def folded_beta_bloch_curl_template() -> str:
    """``special_kz._BETA_BLOCH_CURL_TEMPLATE`` with ONE slot inserted."""
    return _with_top_mask_slot(special_kz._BETA_BLOCH_CURL_TEMPLATE,
                               "special_kz._BETA_BLOCH_CURL_TEMPLATE")


def _reduced_codes(codes: Sequence[int]) -> Tuple[int, int, int]:
    """:func:`.symmetry._reduced_codes`, re-exported through a checked import.

    Imported as a function call rather than re-implemented: "both mirror codes take
    the METALLIC ghost branch" and "the cell-0 mask widens from ``== METALLIC`` to
    ``!= PERIODIC``" are the SAME statement once every non-periodic code is spelled
    METALLIC on the way in, and that mapping has exactly one home.
    """
    from .symmetry import _reduced_codes as reduce_codes  # noqa: PLC0415

    return reduce_codes(codes)


def folded_beta_curl_source(codes: Sequence[int], backward: bool,
                            has_beta: bool = True,
                            contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised REAL folded ``beta_pml_curl_step`` source.

    ``codes`` is the per-axis PERIODIC / METALLIC / MIRROR_METALLIC /
    MIRROR_PERIODIC quadruple :func:`.symmetry.folded_axis_kinds` resolves — NEVER a
    hand-built triple, because the MIRROR_METALLIC / MIRROR_PERIODIC split is the
    fold's single point of failure and backwards on one axis is a plane of wrong
    values, not a crash.

    THE THREE FOLD DELTAS OVER THE CERTIFIED BETA CURL, all three expressed as calls
    rather than as copied text: (1) both mirror codes take the METALLIC ghost branch
    and (2) the cell-0 mask widens from ``== METALLIC`` to ``!= PERIODIC``, which are
    the same statement through :func:`_reduced_codes`; and (3) a folded PERIODIC axis
    masks its LAST plane for every target whose Yee shift is 1 there, which is
    :func:`.symmetry.folded_top_plane_mask`. Nothing arithmetic changes, and in
    particular the beta insert is the parent's ``_REAL_BETA_INSERT`` verbatim, in the
    parent's position: after the dtdx curl and BEFORE both masks.

    ``has_beta=False`` compiles the term out. Engine-route plans always build 1 (the
    predicate requires beta nonzero); the 0 arm exists so a gate's identity leg can
    pin a beta = 0 build against the CERTIFIED FOLDED curl on the same seeds.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    reduced = _reduced_codes(codes)
    return templates.substitute(folded_beta_curl_template(), {
        "__HEADER__": "#include <metal_stdlib>\nusing namespace metal;",
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", reduced[0], backward),
        "__GHOST_Y__": templates.ghost("y", reduced[1], backward),
        "__GHOST_Z__": templates.ghost("z", reduced[2], backward),
        "__BETA__": (special_kz._REAL_BETA_INSERT if has_beta
                     else special_kz._NO_BETA_INSERT),
        "__MASK__": templates.ownership_mask(reduced, backward),
        "__TOP_MASK__": folded_top_plane_mask(codes, backward),
    })


def folded_beta_bloch_curl_source(codes: Sequence[int], backward: bool,
                                  phased: Sequence[int], expansion: str,
                                  has_beta: bool = True,
                                  contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised COMPLEX folded ``beta_bloch_pml_curl_step`` source.

    ``phased`` is the per-axis Bloch flag and ``expansion`` the PROBE-MEASURED
    complex-multiply arm — never a default, because which arm the reference takes is
    a platform fact.

    A PHASED AXIS MUST BE PLAIN PERIODIC, and a mirror code carrying a Bloch flag is
    refused HERE as well as by the predicate: a mis-baked flag is a plane of wrong
    values rather than a crash, and this is the last place it can be seen.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    if len(codes) != 3 or len(phased) != 3:
        raise ValueError(f"codes and phased must be per-axis triples, got "
                         f"{codes!r} / {phased!r}")
    for axis, (code, flag) in enumerate(zip(codes, phased)):
        if not flag:
            continue
        if code in MIRROR_CODES:
            raise ValueError(
                f"axis {axis} is folded and carries a Bloch phase; a mirror plane "
                f"reflects rather than repeating, so stepping._bloch_phases raises "
                f"(S:2338-2341) and driver._require_bloch_is_representable refuses "
                f"the configuration outright (driver.py:1047-1075)")
        if code != CODE_PERIODIC:
            raise ValueError(
                f"axis {axis} carries a Bloch phase but its ghost rule is METALLIC; "
                f"only a periodic wrap can carry a phase (S:2346-2360)")
    reduced = _reduced_codes(codes)
    return templates.substitute(folded_beta_bloch_curl_template(), {
        "__HEADER__": "#include <metal_stdlib>\nusing namespace metal;",
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__COMPLEX_HELPERS__": templates.complex_helpers(expansion),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", reduced[0], backward),
        "__GHOST_Y__": templates.ghost("y", reduced[1], backward),
        "__GHOST_Z__": templates.ghost("z", reduced[2], backward),
        "__WRAPPED__": special_kz._wrapped_lines(backward),
        "__PHASE__": special_kz._phase_lines(phased),
        "__BETA__": (special_kz._COMPLEX_BETA_INSERT if has_beta else
                     "    // HAS_BETA = 0: the identity arm, beta compiled out"),
        "__MASK__": templates.ownership_mask(reduced, backward,
                                             zero=templates.COMPLEX_ZERO),
        "__TOP_MASK__": folded_top_plane_mask(codes, backward,
                                              zero=templates.COMPLEX_ZERO),
    })


def compile_folded_beta_curl(codes: Sequence[int], backward: bool,
                             has_beta: bool = True,
                             contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised REAL folded beta curl entry point for one configuration."""
    return compile_source(folded_beta_curl_source(
        codes, backward, has_beta, contract)).beta_pml_curl_step


def compile_folded_beta_bloch_curl(codes: Sequence[int], backward: bool,
                                   phased: Sequence[int], expansion: str,
                                   has_beta: bool = True,
                                   contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised COMPLEX folded beta curl entry point."""
    return compile_source(folded_beta_bloch_curl_source(
        codes, backward, phased, expansion, has_beta,
        contract)).beta_bloch_pml_curl_step


#: The four codes, in :mod:`.symmetry`'s vocabulary, so an enumeration can be
#: written without a second literal list.
_ALL_CODES: Tuple[int, ...] = (CODE_PERIODIC, CODE_METALLIC,
                               CODE_MIRROR_METALLIC, CODE_MIRROR_PERIODIC)


def enumerate_folded_beta_sources(expansion: str,
                                  contract: str = shaders.CONTRACT_OFF
                                  ) -> Dict[str, str]:
    """Every specialisation the shipped plans can emit, keyed by a stable label.

    Beta is a 2-D lever (``Grid._resolve_beta`` refuses it anywhere else), so the z
    axis carries no fold and no wall in any constructible configuration — but the
    enumeration sweeps all four codes on all three axes anyway, which is a
    deliberate SUPERSET and the safe direction for a fingerprint sweep. The
    COMPLEX arm additionally sweeps the phased flags only on plain-PERIODIC axes,
    which is what :func:`folded_beta_bloch_curl_source` will accept.
    """
    out: Dict[str, str] = {}
    for backward in (False, True):
        direction = "D" if backward else "B"
        for cx in _ALL_CODES:
            for cy in _ALL_CODES:
                for cz in _ALL_CODES:
                    codes = (cx, cy, cz)
                    key = f"real/{direction}/{cx}{cy}{cz}"
                    out[key] = folded_beta_curl_source(codes, backward, True,
                                                       contract)
                    for px in ((0, 1) if cx == CODE_PERIODIC else (0,)):
                        for py in ((0, 1) if cy == CODE_PERIODIC else (0,)):
                            phased = (px, py, 0)
                            out[f"complex/{direction}/{cx}{cy}{cz}/"
                                f"{px}{py}0"] = folded_beta_bloch_curl_source(
                                    codes, backward, phased, expansion, True,
                                    contract)
    return out


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration, TWO inverted clauses on every slot
# ---------------------------------------------------------------------------
#
# The clause numbering mirrors `special_kz`'s so the two can be diffed, and
# `special_kz`'s mirrors `coverage._grid_reasons`'. Clause 12 (beta) is that
# module's own INVERSION and is KEPT; clause 5 (fold) is INVERTED against it;
# clause 4's boundary whitelist is REPLACED by `folded_axis_kinds` rather than
# widened — widening the shared `COVERED_BOUNDARIES` tuple would silently admit
# folds into every plain arm at once.
#
# Everything else is RESTATED rather than subtracted. The parents' reason lists are
# built inside functions whose clauses cannot be removed from outside, and reaching
# a predicate by subtracting another's output is how a clause silently goes missing.


def _requires_a_fold(grid: Any, codes: Optional[Sequence[int]],
                     what: str) -> List[str]:
    """CLAUSE 5, INVERTED against both beta arms — asked TWICE, deliberately.

    Once of the GRID (``has_symmetry`` and some ``is_mirrored`` axis) and once of the
    RESOLVED CODES (some axis landed on a mirror code). The two can disagree — a grid
    can declare a mirror the boundary resolver reports as something else, which is
    precisely the cylindrical hazard — and a family that asked only the first would
    admit a run whose kernel carries no fold at all.
    """
    reasons: List[str] = []
    if not _has_real_fold(grid):
        reasons.append(f"no mirror plane is active: {what}")
    elif codes is not None and not any(int(code) in MIRROR_CODES for code in codes):
        reasons.append("no axis resolves to a mirror boundary: the grid declares a "
                       "fold that stepping._boundary_kinds does not report as one")
    return reasons


def _requires_beta(grid: Any, what: str) -> List[str]:
    """CLAUSE 12, INVERTED against both fold families — one question, one home."""
    if float(getattr(grid, "beta", 0.0)) == 0.0:
        return [f"grid.beta is zero: this product exists only for the FOLDED "
                f"special_kz intersection; {what}"]
    return []


def _folded_beta_axis_reasons(grid: Any, pml: Any) -> Tuple[
        Optional[Tuple[int, int, int]], List[str]]:
    """Clauses 4 and 5 together, through the engine's own classifier.

    ``folded_axis_kinds`` is IMPORTED, not re-derived: it routes the ghost rule
    through ``stepping._boundary_kinds`` (where a fold outranks the declaration),
    splits MIRROR through ``stepping._stored_past_owned``, and CROSS-CHECKS that
    split against ``grid.is_metallic`` — two independent routes to the same fact,
    with a disagreement a refusal rather than a coin toss.
    """
    codes, reasons = folded_axis_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    return codes, list(reasons)


def _dimension_reasons(grid: Any) -> List[str]:
    """Clause 6b: beta is legal only on an effective-2-D Cartesian grid.

    RESTATED, never inferred from ``Grid._resolve_beta``'s constructor guard. MEEP's
    ``fields`` constructor aborts with "Nonzero beta unsupported in dimensions other
    than 2" (fields.cpp:546-547), and a predicate that trusted the constructor would
    admit a hand-built ``fields`` stand-in the engine could never produce — which is
    exactly what a gate's from-arrays route builds.
    """
    if int(getattr(grid, "dimensions", 0)) != 2:
        return [f"grid dimensions={getattr(grid, 'dimensions', None)!r} is not the "
                f"effective-2-D grid beta requires (MEEP fields.cpp:546-547)"]
    return []


def _cylindrical_reasons(grid: Any) -> List[str]:
    """Clause 4, refused TWICE — the grid flag alone is not the inversion.

    ``_boundary_kinds`` puts ``is_axis`` AHEAD of ``is_mirrored``
    (stepping.py:2191-2196), so a folded r axis reports CYL_AXIS and the fold
    vanishes silently; and ``_mirror_phases`` (stepping.py:2346-2366) puts
    ``(-1)**grid.m`` into the SAME SLOT the mirror phase occupies, so reading a phase
    without first refusing ``is_axis`` reads an m-factor as a parity.
    """
    reasons: List[str] = []
    if getattr(grid, "cylindrical", False):
        reasons.append("cylindrical (Dcyl) coordinates are not carried (and the "
                       "grid itself refuses beta there)")
    for axis in range(3):
        if _call(grid, "is_axis", axis, default=False):
            reasons.append(f"axis {axis} is the cylindrical r = 0 axis")
    return reasons


def _curl_conductivity_reasons(fields: Any, sub_step: Optional[str]) -> List[str]:
    """A conductivity on this curl's targets is refused by name, fail-closed.

    A missing or non-callable ``condfac_for`` is refused OUTRIGHT rather than read as
    "no conductivity": inferring absence from an unreadable table is admission by
    attribute absence, which this clause list refuses by name in three other places.
    """
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        return ["fields does not expose condfac_for; an unreadable conductivity "
                "table is not an absent one"]
    reasons: List[str] = []
    targets = (CURL_SUB_STEPS[sub_step] if sub_step is not None else CURL_TARGETS)
    for target in targets:
        try:
            conductive = reader(target) is not None
        except Exception as exc:  # noqa: BLE001 - unreadable means not covered
            reasons.append(f"condfac_for({target!r}) raised {exc!r}")
            continue
        if conductive:
            reasons.append(
                f"a conductivity is installed on {target}: this curl transcribes "
                f"the plain split-field recurrence only, and neither parent carries "
                f"a conductive arm either")
    return reasons


def _folded_beta_real_grid_reasons(fields: Any, pml: Any, grid: Any,
                                   ) -> Tuple[Optional[Tuple[int, int, int]],
                                              List[str]]:
    """The clauses every REAL folded-beta predicate shares, plus the resolved codes."""
    reasons: List[str] = []

    # 1. The Metal backend, the host array module and the subnormal policy.
    reasons.extend(_metal_backend_reasons(grid))

    # 2. REAL storage required. The complex folded beta run is this module's OTHER
    #    arm, and the split is the same one every real/complex pair in the package
    #    makes.
    if getattr(fields, "force_complex_fields", False):
        reasons.append("force_complex_fields=True: a complex-storage folded beta "
                       "run belongs to this module's complex arm")

    # 3. An absorber that actually absorbs (split-field family only).
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the "
                       "split-field path only)")

    # 4/5. The ghost rules INCLUDING the two folded ones, then the INVERSION.
    #      `COVERED_BOUNDARIES` is NOT consulted and NOT widened.
    codes, fold_reasons = _folded_beta_axis_reasons(grid, pml)
    reasons.extend(fold_reasons)

    # 4b/6b. Cartesian, effective 2-D.
    reasons.extend(_cylindrical_reasons(grid))
    reasons.extend(_dimension_reasons(grid))

    # 7. k = 0 on EVERY axis. Real storage carries no Bloch phase, and on a folded
    #    axis the engine refuses a nonzero k outright anyway.
    if getattr(grid, "has_bloch", False):
        reasons.append(f"nonzero k_point {getattr(grid, 'k_point', None)!r} (real "
                       f"storage carries no Bloch phase; in-plane k + beta is the "
                       f"complex arm's domain)")
    k_point = getattr(grid, "k_point", (0.0, 0.0, 0.0))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {tuple(k_point)!r} is not exactly zero")

    # 9a/9b. A registered susceptibility must be one this package understands.
    #        Dispersion is ADMITTED for the curl: it changes the VALUES update_E
    #        writes into the array the curl differences, not an operation the curl
    #        performs. The E-side constitutive predicate refuses it separately.
    reasons.extend(_susceptibility_reasons(fields))

    # 9c. STORED E — the invariant behind admitting dispersion on the curl.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 10/11. No instantaneous nonlinearity, no BFAST. Both are refused by BOTH
    #        parents for the same reason this family exists: a term added between
    #        the curl and the mask composes with the fold's widened mask rather than
    #        adding to it.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried, and a nonlinear fold has no single parity)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not "
                       "carried; beta and BFAST together on a fold is a third "
                       "composition, not this one)")

    # 12 (INVERTED against both fold families, which pre-registered this owner).
    reasons.extend(_requires_beta(
        grid, "a beta = 0 folded run belongs to symmetry (real) or folded_complex "
              "(complex)"))

    # 12b. Real storage + off-diagonal epsilon + beta: stepping RAISES
    #      (stepping.py:800-810) and MEEP aborts (fields.cpp:548-549) — the
    #      implicit-i trick cancels only while TE and TM stay uncoupled. Refused for
    #      the WHOLE real arm, not only update_E: the run raises inside the first
    #      beta curl.
    if getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append("off-diagonal epsilon with beta in REAL storage: the "
                       "implicit-i trick no longer cancels (stepping.py:800-810 "
                       "raises; MEEP fields.cpp:548-549 aborts) — the run needs "
                       "force_complex_fields=True")

    return codes, reasons


def _folded_beta_complex_grid_reasons(fields: Any, pml: Any, grid: Any,
                                      probe: Any = None,
                                      ) -> Tuple[Optional[Tuple[int, int, int]],
                                                 List[str]]:
    """The clauses every COMPLEX folded-beta predicate shares, plus the codes."""
    reasons: List[str] = []

    reasons.extend(_metal_backend_reasons(grid))

    # 2 (INVERTED against the real arm). Complex64 storage REQUIRED.
    if not (getattr(fields, "force_complex_fields", False)
            or getattr(grid, "has_bloch", False)):
        reasons.append("storage is real float32 (neither force_complex_fields nor "
                       "a nonzero k_point): a real folded beta run belongs to this "
                       "module's real arm")

    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the complex "
                       "split-field path only)")

    codes, fold_reasons = _folded_beta_axis_reasons(grid, pml)
    reasons.extend(fold_reasons)

    reasons.extend(_cylindrical_reasons(grid))
    reasons.extend(_dimension_reasons(grid))

    # 9 (from folded_complex, which established it). Per-axis Bloch legality
    #   INCLUDING the fold clause: a folded axis may carry NO phase and NO nonzero
    #   k component, zone edge included. Neither `special_kz`'s complex arm (which
    #   never sees a fold) nor `symmetry` (which never sees a phase) asks this.
    active_pml = pml if (pml is not None
                         and getattr(pml, "is_active", False)) else None
    reasons.extend(_folded_phase_reasons(grid, _boundary_kinds(grid, active_pml),
                                         codes))

    # 5b (from special_kz). No k on the INVARIANT axis: beta IS that axis's
    #    analytic dependence, and the Grid refuses the pairing at construction.
    #    RESTATED, never inferred.
    k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    if len(k_point) > 2 and float(k_point[2]) != 0.0:
        reasons.append(f"k_point component z = {k_point[2]!r} rides the invariant "
                       f"axis whose dependence beta already carries analytically")

    reasons.extend(_curl_conductivity_reasons(fields, None))

    # 7. No dispersion under complex storage on this backend, folded or not.
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: complex-storage ADE is a "
                       "later tranche, folded or not")
    reasons.extend(_susceptibility_reasons(fields))

    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (a separate family, and a nonlinear "
                       "fold has no single parity)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (a separate family)")

    reasons.extend(_requires_beta(
        grid, "a beta = 0 folded complex run belongs to folded_complex"))

    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 13. EXPANSION binding requires a MEASURED probe artifact carrying
    #     special_kz's extended pattern set — the arithmetic here is that family's,
    #     so its artifact is the one that licenses this arm.
    reasons.extend(special_kz._beta_expansion_reasons(probe))

    return codes, reasons


def folded_beta_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                  residency: Any = None) -> Coverage:
    """May the REAL folded beta curl step this? THE WIDE VERDICT.

    ZERO FOLDED AXES IS ADMITTED HERE, and that is not an oversight: with every axis
    resolving to PERIODIC/METALLIC, :func:`folded_beta_curl_source` reduces to
    ``special_kz.beta_curl_source``'s output CHARACTER FOR CHARACTER except for an
    empty top-plane block, which is a property a gate MEASURES rather than a claim.
    THIS VERDICT MUST NOT BE REGISTERED ON A SLOT — it would make every unfolded beta
    row ambiguous against ``special_kz``.
    :func:`folded_beta_composition_curl_coverage` is the routing verdict.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    codes, reasons = _folded_beta_real_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_curl_conductivity_reasons(fields, sub_step))

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))
    reasons.extend(_stored_extent_reasons(codes, shape))
    return Coverage(not reasons, tuple(reasons))


def folded_beta_composition_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                          residency: Any = None) -> Coverage:
    """THE ROUTING VERDICT — the wide one with a fold MANDATORY.

    The clause that separates this arm from ``special_kz``'s real arm on the two curl
    slots, spelled as its own function so a reader can see the inversion in one
    place. Registering the wide verdict instead would make every unfolded beta row
    ambiguous and would leave the numerical method depending on composer order.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    base = folded_beta_pml_curl_coverage(fields, pml, sub_step, residency)
    codes, _ = _folded_beta_axis_reasons(grid, pml)
    reasons = list(base.reasons)
    reasons.extend(_requires_a_fold(
        grid, codes,
        "the folded beta curl is an equivalence product here, not a composition "
        "candidate; an unfolded beta grid belongs to special_kz's certified curl"))
    return Coverage(not reasons, tuple(reasons))


def folded_beta_bloch_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                        residency: Any = None,
                                        probe: Any = None) -> Coverage:
    """May the COMPLEX folded beta curl step this? THE WIDE VERDICT (see above)."""
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    codes, reasons = _folded_beta_complex_grid_reasons(fields, pml, grid, probe)
    reasons.extend(_residency_declaration_reasons(residency))

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + tuple(spec["sources"]))
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(special_kz._complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))
    reasons.extend(_stored_extent_reasons(codes, shape))
    return Coverage(not reasons, tuple(reasons))


def folded_beta_composition_bloch_curl_coverage(fields: Any, pml: Any,
                                                sub_step: str,
                                                residency: Any = None,
                                                probe: Any = None) -> Coverage:
    """THE ROUTING VERDICT for the complex arm — a fold MANDATORY."""
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    base = folded_beta_bloch_pml_curl_coverage(fields, pml, sub_step, residency,
                                               probe)
    codes, _ = _folded_beta_axis_reasons(grid, pml)
    reasons = list(base.reasons)
    reasons.extend(_requires_a_fold(
        grid, codes,
        "the folded complex beta curl is an equivalence product here, not a "
        "composition candidate; an unfolded complex beta grid belongs to "
        "special_kz's certified Bloch beta curl"))
    return Coverage(not reasons, tuple(reasons))


def folded_beta_constitutive_coverage(fields: Any, pml: Any, side: str,
                                      residency: Any = None) -> Coverage:
    """May the CERTIFIED REAL constitutive kernel step a FOLDED BETA run?

    No new kernel and no new arithmetic: only the ADMISSION is this family's. The
    constitutive sub-steps read nothing beta-dependent AND nothing fold-dependent —
    the fold reaches them through the STORED EXTENT alone, and the PML coefficient
    vectors are already built at that extent — so this delegates to
    ``launch.ConstitutivePlan`` unchanged. The layout clauses below are where "built
    at the folded stored extent" stops being an assumption.

    THE E SIDE REFUSES WHAT CHANGES WHAT ``source`` IS, each naming its owner: a
    registered susceptibility (the source is ``D - sum P``, not ``D``), and
    ``stores_E`` false. An off-diagonal row is already refused module-wide on this
    arm, because the array path RAISES on real storage + off-diagonal + beta.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    codes, reasons = _folded_beta_real_grid_reasons(fields, pml, grid)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_requires_a_fold(
        grid, codes,
        "the folded beta constitutive product has no array-path work to specialize; "
        "an unfolded beta grid belongs to special_kz's constitutive companion"))

    if side == "E":
        if getattr(fields, "has_polarizations", False) or (
                getattr(fields, "polarizations", ()) or ()):
            reasons.append(
                "a susceptibility is registered: update_E's source is (D - sum P), "
                "not D — the folded dispersive beta update_E is not built here")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))
    reasons.extend(_stored_extent_reasons(codes, shape))
    return Coverage(not reasons, tuple(reasons))


def folded_beta_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                              residency: Any = None,
                                              probe: Any = None) -> Coverage:
    """May the CERTIFIED COMPLEX constitutive kernel step a FOLDED BETA run?

    THE ASYMMETRY WITH ``special_kz`` IS OVER, and this function was the first half
    of closing it. ``complex_fields``' clause 8 refuses ``grid.beta`` by name, so
    admitting a complex beta run anywhere takes a restatement; this is that
    restatement FOR THE FOLDED CASE, which is where three of the four corpus rows
    are (six slots). The UNFOLDED case — two slots, one corpus row — was a named
    refusal in ``special_kz`` until 2026-08-19 and is now
    ``special_kz.beta_run_complex_constitutive_coverage``, built the same way one
    level out. The fold clause below is what keeps the two disjoint.

    The arithmetic is ``complex_fields``' certified body; only the ADMISSION is here.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    codes, reasons = _folded_beta_complex_grid_reasons(fields, pml, grid, probe)
    reasons.extend(_residency_declaration_reasons(residency))
    reasons.extend(_requires_a_fold(
        grid, codes,
        "the folded complex beta constitutive product has no array-path work to "
        "specialize; an unfolded complex beta grid's constitutive pair belongs to "
        "special_kz's own complex constitutive companion"))

    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed (the row product reads "
            "neighbours; this sub-step is element-wise). On a fold the partner-axis "
            "down shift is a live interior plane, which is the folded complex "
            "off-diagonal beta product and is not carried on this backend")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(special_kz._complex_layout_reasons(fields, shape, names))
    if side == "E" and len(shape) == 3:
        # inv_eps stays float32 under complex storage (stepping.py:41-50,
        # fields.py:1203-1204), so the shipped float32 pin is exactly right.
        reasons.extend(_inverse_epsilon_reasons(fields, shape))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))
    reasons.extend(_stored_extent_reasons(codes, shape))
    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

class FoldedBetaPmlCurlPlan(KernelPlan):
    """A launchable, allocation-free REAL folded beta curl sub-step.

    ``special_kz.BetaPmlCurlPlan``'s bindings EXACTLY — the fold adds no argument,
    because the STORED EXTENT is what carries it — over a different compiled kernel.
    DELIBERATELY A SIBLING RATHER THAN A SUBCLASS, for ``folded_complex``'s reason:
    the two hold the same argument tuple and launch different functions, and a
    subclass would inherit nothing but the risk that a future edit to the parent's
    ``__init__`` silently rebinds this one.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "beta_plus", "beta_minus", "has_beta", "residency", "volumes")

    family = "folded beta real PML curl"

    REPR_FIELDS = ("sub_step", "shape", "bc", "beta_plus", "beta_minus", "has_beta")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, residency,
                 beta_plus: float, beta_minus: float, targets, auxiliaries,
                 sources, coefficients, functions: Dict[str, Any],
                 volumes: Sequence[str], has_beta: bool = True) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                             f"got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        # Already f32-rounded by the host (special_kz.beta_curl_coefficients);
        # float() keeps the bits and the scalar ABI delivers them. MEASURED on this
        # host: a bound `constant float&` and the same value baked as a literal
        # produce identical words (0/144, four coefficients, both contraction
        # modes), so binding costs nothing and buys one source per configuration
        # instead of one per beta value.
        self.beta_plus = float(beta_plus)
        self.beta_minus = float(beta_minus)
        self.has_beta = bool(has_beta)
        self.residency = residency
        self.volumes = tuple(volumes)
        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem,
               self.dtdx, self.beta_plus, self.beta_minus))


class FoldedBetaBlochPmlCurlPlan(KernelPlan):
    """A launchable, allocation-free COMPLEX folded beta curl sub-step.

    ``special_kz.BetaBlochPmlCurlPlan``'s thirty bindings exactly — ONE below Metal's
    31-buffer ceiling (``device.MAX_BUFFER_BINDINGS``), which is why the complex
    volumes must be ``float2``: a re/im-split build needs nine more and is a compile
    error, not a slower kernel. The fold adds no binding at all.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "phased", "phase_words", "beta_words", "has_beta", "expansion",
                 "residency", "volumes")

    family = "folded beta complex PML curl"

    REPR_FIELDS = ("sub_step", "shape", "bc", "phased", "expansion", "has_beta")

    def __init__(self, sub_step: str, shape, dtdx: float, bc, phased,
                 phase_words, beta_words, expansion: str, residency, targets,
                 auxiliaries, sources, coefficients, functions: Dict[str, Any],
                 volumes: Sequence[str], has_beta: bool = True) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                             f"got {sub_step!r}")
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in bc)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_words = tuple(float(word) for word in phase_words)
        self.beta_words = tuple(tuple(float(word) for word in pair)
                                for pair in beta_words)
        self.has_beta = bool(has_beta)
        self.expansion = str(expansion)
        self.residency = residency
        self.volumes = tuple(volumes)
        (bpr, bpi), (bmr, bmi) = self.beta_words
        super().__init__(
            functions,
            tuple(targets) + tuple(auxiliaries) + tuple(sources)
            + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem,
               self.dtdx) + self.phase_words + (bpr, bpi, bmr, bmi))


def _real_curl_functions(codes, backward: bool, has_beta: bool,
                         contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_folded_beta_curl(codes, backward, has_beta, mode)
            for mode in contract_variants}


def _complex_curl_functions(codes, backward: bool, phased, expansion: str,
                            has_beta: bool,
                            contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_folded_beta_bloch_curl(codes, backward, phased,
                                                 expansion, has_beta, mode)
            for mode in contract_variants}


def plan_folded_beta_pml_curl(fields: Any, pml: Any, sub_step: str,
                              residency: Optional[Residency] = None,
                              contract_variants: Sequence[str] = (
                                  shaders.CONTRACT_OFF,),
                              ) -> Optional[FoldedBetaPmlCurlPlan]:
    """Build a REAL folded beta curl plan from the engine's objects, or None.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.

    The codes come from :func:`.symmetry.folded_axis_kinds` and from nowhere else —
    building them from ``_boundary_kinds`` alone would lose the MIRROR_METALLIC /
    MIRROR_PERIODIC split, which decides the top-plane mask.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not folded_beta_composition_curl_coverage(
            fields, pml, sub_step, residency).covered:
        return None
    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None

    spec = SUB_STEPS[sub_step]
    plus, minus = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(sub_step == "step_B"), complex_storage=False)
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, getattr(fields, "fu_" + n))
                   for n in spec["targets"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return FoldedBetaPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, residency, plus, minus,
        targets, auxiliaries, sources, coefficients,
        _real_curl_functions(codes, spec["backward"], True, contract_variants),
        volumes)


def plan_folded_beta_pml_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any], codes,
        dtdx: float, beta_plus: float, beta_minus: float, residency: Residency,
        functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        has_beta: bool = True) -> FoldedBetaPmlCurlPlan:
    """Build a REAL folded beta curl plan from bare host arrays — the gate's route.

    ``codes`` is the FOUR-valued per-axis quadruple. No predicate runs: the caller is
    a harness that constructed the configuration deliberately, including the
    deliberately wrong ones. ``functions`` is the MUTATION SEAM — dropping it is not
    a slowdown, it is a silent DISARMING, since every mutation leg would then launch
    the shipped kernel and report the defect as uncaught. ``has_beta=False`` is the
    identity leg's arm, which must reduce to the CERTIFIED FOLDED curl.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    targets = [residency.mirror(n, arrays[n]) for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, arrays["fu_" + n])
                   for n in spec["targets"]]
    sources = [residency.mirror(n, arrays[n]) for n in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return FoldedBetaPmlCurlPlan(
        sub_step, shape, dtdx, codes, residency, beta_plus, beta_minus,
        targets, auxiliaries, sources, coefficients,
        functions if functions is not None
        else _real_curl_functions(codes, spec["backward"], has_beta,
                                  contract_variants),
        volumes, has_beta=has_beta)


def plan_folded_beta_bloch_pml_curl(fields: Any, pml: Any, sub_step: str,
                                    residency: Optional[Residency] = None,
                                    contract_variants: Sequence[str] = (
                                        shaders.CONTRACT_OFF,),
                                    probe: Any = None,
                                    ) -> Optional[FoldedBetaBlochPmlCurlPlan]:
    """Build a COMPLEX folded beta curl plan from the engine's objects, or None."""
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not folded_beta_composition_bloch_curl_coverage(
            fields, pml, sub_step, residency, probe).covered:
        return None
    record = probe if probe is not None else special_kz.load_expansion_probe()
    expansion = special_kz.beta_expansion_from_probe(record)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    grid = fields.grid
    codes, _ = folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None

    spec = SUB_STEPS[sub_step]
    # THE PHASE FLAGS COME FROM `_boundary_kinds`, NOT FROM THE FOLDED CODES. A
    # folded axis is refused a phase by clause 9 above, so the two agree on every
    # admitted run; reading the phase off the RESOLVED kinds is what keeps this
    # builder identical to `special_kz`'s and keeps the conjugation in one place.
    kinds = _boundary_kinds(grid, pml)
    phased, phase_words = special_kz.bloch_phase_words(
        grid, kinds, bool(spec["backward"]))
    beta_words = special_kz.beta_curl_coefficients(
        grid.beta, grid.dt, magnetic=(sub_step == "step_B"), complex_storage=True)
    complex64 = special_kz._numpy().complex64
    targets = [residency.mirror(n, getattr(fields, n), dtype=complex64)
               for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, getattr(fields, "fu_" + n),
                                    dtype=complex64) for n in spec["targets"]]
    sources = [residency.mirror(n, getattr(fields, n), dtype=complex64)
               for n in spec["sources"]]
    # The PML coefficient vectors stay REAL on both sides: the array path's
    # `fu *= kms` is a complex-by-real multiply (stepping.py:1976-1982).
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"),
                         constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return FoldedBetaBlochPmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, phased, phase_words,
        beta_words, expansion, residency, targets, auxiliaries, sources,
        coefficients,
        _complex_curl_functions(codes, spec["backward"], phased, expansion, True,
                                contract_variants),
        volumes)


def plan_folded_beta_bloch_pml_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any], codes,
        phased, phase_words, dtdx: float, beta_words, expansion: str,
        residency: Residency, functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        has_beta: bool = True) -> FoldedBetaBlochPmlCurlPlan:
    """Build a COMPLEX folded beta curl plan from bare host arrays — a gate's route.

    ``beta_words`` is ``((bpr, bpi), (bmr, bmi))`` — normally
    :func:`.special_kz.beta_curl_coefficients`'s output, or a deliberately wrong pair
    on a mutation leg. The ``step_D`` phase conjugation is the CALLER's here, exactly
    as the sub-step split makes it the plan builder's on the engine route.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    complex64 = special_kz._numpy().complex64
    targets = [residency.mirror(n, arrays[n], dtype=complex64)
               for n in spec["targets"]]
    auxiliaries = [residency.mirror("fu_" + n, arrays["fu_" + n], dtype=complex64)
                   for n in spec["targets"]]
    sources = [residency.mirror(n, arrays[n], dtype=complex64)
               for n in spec["sources"]]
    coefficients = [residency.mirror(f"pml:{stem}_{axis}:{sub_step}",
                                     flat[f"{stem}_{axis}"], constant=True)
                    for axis in "xyz" for stem in ("kms", "sinv")]
    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    return FoldedBetaBlochPmlCurlPlan(
        sub_step, shape, dtdx, codes, phased, phase_words, beta_words, expansion,
        residency, targets, auxiliaries, sources, coefficients,
        functions if functions is not None
        else _complex_curl_functions(codes, spec["backward"], phased, expansion,
                                     has_beta, contract_variants),
        volumes, has_beta=has_beta)


def plan_folded_beta_constitutive(fields: Any, pml: Any, side: str,
                                  residency: Any = None,
                                  contract_variants: Sequence[str] = (
                                      shaders.CONTRACT_OFF,)) -> Any:
    """A CERTIFIED REAL constitutive plan for a folded beta run, or None.

    The arithmetic, the source and the plan class are the certified ones; only the
    ADMISSION is this family's. Building the plan through the certified builder —
    rather than re-deriving one here — is what makes "same kernel" a fact instead of
    a claim: a divergence would have to come from the predicate, which is the only
    thing this family contributes on these two slots.
    """
    from .launch import (  # noqa: PLC0415 - avoids a circular import at module load
        ConstitutivePlan, _constitutive_functions,
    )

    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not folded_beta_constitutive_coverage(fields, pml, side, residency).covered:
        return None
    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    targets = [residency.mirror(n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [residency.mirror(n, getattr(fields, n)) for n in spec["aux"]]
    sources = [residency.mirror(n, getattr(fields, n)) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, fields.inverse_epsilon_for(n),
                          constant=True) for n in spec["targets"]]
        if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{suffix}",
                         getattr(pml, f"{stem}_{axis}{suffix}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return ConstitutivePlan(
        side, fields.grid.shape, residency, targets, auxiliaries, sources,
        inverse_epsilon, coefficients,
        _constitutive_functions(side, contract_variants), volumes)


def plan_folded_beta_complex_constitutive(fields: Any, pml: Any, side: str,
                                          residency: Any = None,
                                          contract_variants: Sequence[str] = (
                                              shaders.CONTRACT_OFF,),
                                          probe: Any = None) -> Any:
    """A CERTIFIED COMPLEX constitutive plan for a folded beta run, or None.

    ``complex_fields``' own body, its own plan class and its own compiled functions;
    only the ADMISSION is this family's.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not folded_beta_complex_constitutive_coverage(
            fields, pml, side, residency, probe).covered:
        return None
    expansion = special_kz.beta_expansion_from_probe(
        probe if probe is not None else special_kz.load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None

    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    mirror = complex_fields._complex_mirror
    targets = [mirror(residency, n, getattr(fields, n)) for n in spec["targets"]]
    auxiliaries = [mirror(residency, n, getattr(fields, n)) for n in spec["aux"]]
    sources = [mirror(residency, n, getattr(fields, n)) for n in spec["sources"]]
    inverse_epsilon = (
        [residency.mirror("inv_eps_" + n, fields.inverse_epsilon_for(n),
                          constant=True) for n in spec["targets"]]
        if side == "E" else None)
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{suffix}",
                         getattr(pml, f"{stem}_{axis}{suffix}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]
    volumes = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    return complex_fields.ComplexConstitutivePlan(
        side, fields.grid.shape, expansion, residency, targets, auxiliaries,
        sources, inverse_epsilon, coefficients,
        complex_fields._constitutive_functions(side, expansion, contract_variants),
        volumes)


# ---------------------------------------------------------------------------
# WIRING — eight arms over four slots, two arms per slot (real and complex)
# ---------------------------------------------------------------------------
#
# WHAT SEPARATES THEM, per slot, and every one of these is a clause that names the
# other family rather than an omission:
#
#   step_B / step_D    `coverage._grid_reasons` clause 12 refuses beta (pml_curl,
#                      constitutive); `bfast_curl` and `complex_fields` restate it;
#                      `symmetry` clause 12 and `folded_complex` clause 8 refuse beta
#                      BY NAME AND NAME THIS FAMILY; `special_kz`'s two arms refuse
#                      the FOLD by name. Clause 5 HERE is inverted through
#                      `_requires_a_fold` and clause 12 through `_requires_beta`, so
#                      this family is separated from every other by TWO independent
#                      clauses rather than one.
#   update_H/update_E  the same two separations, plus `no_pml_constitutive` inverting
#                      on the ABSORBER (it requires an INACTIVE one and this family
#                      an active one), and `offdiag_update_e` /
#                      `folded_offdiag_update_e` requiring a LIVE off-diagonal row,
#                      which the real arm refuses outright (the array path RAISES on
#                      that combination) and the complex arm refuses on the E side.
#   fill_B / fill_D    NOT CLAIMED. `symmetry`'s and `folded_complex`'s fill
#                      predicates carry no beta clause because `fill_symmetry_bc_*`
#                      copies `parity * field[plane]` and reads no beta term, so they
#                      already admit a folded beta run and the matrix has pinned that
#                      since before this family existed. A third arm here would make
#                      both slots AMBIGUOUS and drop them to the array path.
#
# THE REAL AND COMPLEX ARMS ARE REGISTERED AS TWO FAMILIES, not one with a branch,
# for `special_kz`'s reason: the disjointness sweep reaches every arm through
# `arms.registered()`, and two rows that invert on storage must both be visible to
# it or the pair most likely to collide is the pair it cannot see.

#: Which constitutive side each slot names, so one pair of callables serves both.
_CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}


def _has_folded_beta(context: Any) -> bool:
    """The cheap gate that decides whether this family is CONSULTED at all.

    A gated-out arm contributes NO reason; a consulted-and-refusing arm contributes
    all of its reasons by name. So the gate is deliberately the cheapest possible
    read of the grid and never the predicate itself: on an unfolded run, or a folded
    run at beta = 0, four or five other arms will speak on these slots and a silent
    folded-beta arm costs a reader nothing.

    IT ASKS ABOUT THE FOLD AND BETA BUT NOT ABOUT STORAGE, deliberately. Both arms
    must be consulted on any folded beta run so that exactly one of them says
    "storage is real float32" or "force_complex_fields=True" — that sentence is the
    inversion made visible, and gating it out would hide which arm declined.
    """
    grid = getattr(getattr(context, "fields", None), "grid", None)
    if grid is None:
        return False
    return _has_real_fold(grid) and float(getattr(grid, "beta", 0.0)) != 0.0


def _real_curl_coverage(context: Any, slot: str) -> Coverage:
    return folded_beta_composition_curl_coverage(
        context.fields, context.pml, slot, context.residency)


def _real_curl_plan(context: Any, slot: str) -> Any:
    return plan_folded_beta_pml_curl(context.fields, context.pml, slot,
                                     context.residency, context.contract_variants)


def _complex_curl_coverage(context: Any, slot: str) -> Coverage:
    return folded_beta_composition_bloch_curl_coverage(
        context.fields, context.pml, slot, context.residency,
        context.extra.get("beta_probe"))


def _complex_curl_plan(context: Any, slot: str) -> Any:
    return plan_folded_beta_bloch_pml_curl(
        context.fields, context.pml, slot, context.residency,
        context.contract_variants, context.extra.get("beta_probe"))


def _real_constitutive_coverage(context: Any, slot: str) -> Coverage:
    return folded_beta_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency)


def _real_constitutive_plan(context: Any, slot: str) -> Any:
    return plan_folded_beta_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants)


def _complex_constitutive_coverage(context: Any, slot: str) -> Coverage:
    return folded_beta_complex_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.extra.get("beta_probe"))


def _complex_constitutive_plan(context: Any, slot: str) -> Any:
    return plan_folded_beta_complex_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants,
        context.extra.get("beta_probe"))


def register_arms() -> Tuple[Any, ...]:
    """This family's eight arms: two curl pairs and two constitutive pairs."""
    from . import arms  # noqa: PLC0415 - deferred: `arms` imports nothing of ours

    registered = [
        arms.register(family=FAMILY_REAL, slot=slot, label=LABEL_REAL,
                      coverage=_real_curl_coverage, plan=_real_curl_plan,
                      prefix="folded beta real: ",
                      noun="folded beta real curl",
                      gate=_has_folded_beta, wired=True)
        for slot in ("step_B", "step_D")]
    registered.extend(
        arms.register(family=FAMILY_COMPLEX, slot=slot, label=LABEL_COMPLEX,
                      coverage=_complex_curl_coverage, plan=_complex_curl_plan,
                      prefix="folded beta complex: ",
                      noun="folded beta complex curl",
                      gate=_has_folded_beta, wired=True)
        for slot in ("step_B", "step_D"))
    registered.extend(
        arms.register(family=FAMILY_REAL, slot=slot, label=LABEL_REAL,
                      coverage=_real_constitutive_coverage,
                      plan=_real_constitutive_plan,
                      prefix="folded beta real: ",
                      noun="folded beta real constitutive",
                      gate=_has_folded_beta, wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    registered.extend(
        arms.register(family=FAMILY_COMPLEX, slot=slot, label=LABEL_COMPLEX,
                      coverage=_complex_constitutive_coverage,
                      plan=_complex_constitutive_plan,
                      prefix="folded beta complex: ",
                      noun="folded beta complex constitutive",
                      gate=_has_folded_beta, wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    return tuple(registered)


#: Registered ON IMPORT, once — the registry refuses a duplicate by design.
ARMS: Tuple[Any, ...] = register_arms()
