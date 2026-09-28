"""The one module that imports every family, so the arm table is COMPLETE.

WHY A MODULE AND NOT A LINE IN ``launch.py``. Two families — :mod:`.complex_fields`
and :mod:`.special_kz` — import ``launch.SUB_STEPS`` at their own module scope, so
``launch`` cannot import them back at ITS module scope without a cycle whose
outcome depends on which of the two a caller reached first. (The failure is not
theoretical: with the import at the bottom of ``launch``, ``import complex_fields``
first re-enters ``launch``, which then imports a HALF-INITIALISED
``complex_fields`` and reads an attribute that does not exist yet.) One module,
imported from a function body by :func:`.arms.ensure_registered`, has no cycle to
resolve.

REGISTRATION IS THE IMPORT, NOT A CALL MADE HERE. Every family registers its arms
at its own module scope, so importing it once is what puts its rows in the table
and Python's module cache is what makes "once" true. This module therefore calls
nothing: adding a ``register_arms()`` call beside the import would double-register
and :func:`.arms.register` refuses a duplicate by design.

WHAT IS IN THE TABLE, and what each row is admitted by — measured, not asserted;
see ``parity/meep_gpu/probe_metal_planner_composition.py`` for the sweep and
``meep_gpu/test_metal_planner_composition.py`` for the pinned expectations:

TWO COLUMNS, BECAUSE THEY DIFFER. "registered" is the slots an arm holds a row on;
"wins" is the slots its predicate can actually admit. An arm registered on a slot it
can never win is not dead weight — it is how a refusal gets NAMED instead of being
left unasked, and ``special_kz_complex`` carries two of them deliberately.

===================== ============= ============= ==============================
family                registered    wins          what makes it the ONLY admitter
===================== ============= ============= ==============================
pml_curl              B, D          B, D          real storage, active PML, k = 0
constitutive          H, E          H, E          the same, and no off-diagonal row
no_pml_constitutive   H, E          H, E          absorber INACTIVE (the null plan)
complex_fields        all four      all four      complex64 storage; beta refused
special_kz_real       all four      all four      grid.beta != 0 with real storage
special_kz_complex    all four      B, D          grid.beta != 0 with complex
                                                  storage; the H/E rows exist ONLY
                                                  to name the missing complex-beta
                                                  constitutive rather than fall
                                                  silently to the array path
bfast_curl            all four      all four      grid.bfast_active
offdiag_update_e      E             E             a LIVE off-diagonal chi1inv row
folded                all four      all four      a MIRROR PLANE is active — clause
                      + fill_B/D    + fill_B/D    5 inverted; every other family
                                                  refuses a fold BY NAME, on every
                                                  slot. The two fill rows are the
                                                  ONLY arms on those seam slots and
                                                  are deliberately UNGATED, so an
                                                  unfolded run gets a named refusal
                                                  there rather than silence
folded_complex        all four      all four      a MIRROR PLANE **and** complex64
                      + fill_B/D    + fill_B/D    STORAGE: the intersection
                                                  ``folded`` and ``complex_fields``
                                                  each refuse by name — the first on
                                                  storage, the second on the fold —
                                                  so storage is the whole inversion
                                                  against ``folded`` and the fold is
                                                  the whole inversion against
                                                  ``complex_fields``. Its two fill
                                                  rows sit on the same seam slots as
                                                  the real fold's and are disjoint
                                                  from them by the same clause
cylindrical_          all four      all four      the CYLINDRICAL r = 0 AXIS with
complex                                           complex64 storage at |m| >= 1 —
                                                  clause 6 inverted, which alone
                                                  separates it from every other
                                                  family in this table, since each
                                                  of them refuses ``cylindrical``
                                                  and ``is_axis`` BY NAME. Its NEW
                                                  clause 14 refuses m = 0, naming
                                                  the cylindrical REAL product as
                                                  that row's owner — an inverted
                                                  clause rather than the engine
                                                  coupling clause 2 would have
                                                  left it resting on. The two
                                                  constitutive rows are the
                                                  CERTIFIED complex body under a
                                                  restated predicate: ``update_H``
                                                  and ``update_E`` carry no
                                                  cylindrical branch at all, which
                                                  is measured rather than grepped
cylindrical_real      all four      all four      the CYLINDRICAL r = 0 AXIS with
                                                  REAL float32 storage at m = 0 —
                                                  clause 6 inverted, the same single
                                                  inversion that separates the
                                                  complex cylindrical family from
                                                  every Cartesian one. Against ITS
                                                  cylindrical sibling the separation
                                                  is DOUBLE and deliberately so:
                                                  clause 14 answered ``m == 0`` where
                                                  that family answers ``abs(m) >= 1``,
                                                  and clause 2 kept UN-inverted where
                                                  that family inverts it. Neither
                                                  rests on the engine coupling that
                                                  makes complex storage mandatory at
                                                  |m| >= 1, because a coupling is not
                                                  a clause. Its curl plan is the only
                                                  one in the table that LAUNCHES
                                                  TWICE — a column-serial radial scan
                                                  and then the curl that reads it —
                                                  and its two constitutive rows are
                                                  the CERTIFIED real body under a
                                                  restated predicate
nonlinear             B, D, H, E    B, D, H, E    chi2/chi3 INSTALLED — the shared
_constitutive                                     clause 10 INVERTED. ``update_E``
                                                  uses the dedicated Pade body;
                                                  B/D/H reuse their individually
                                                  certified ordinary PML shaders
                                                  through separate nonlinear-only
                                                  spine arms, because the engine
                                                  does not read chi2/chi3 in those
                                                  three sub-steps. Every other
                                                  family refuses an instantaneous
                                                  nonlinearity by name. The
                                                  inversion is therefore a real
                                                  disjointness boundary, not a
                                                  table-order preference
folded_offdiag        E             E             a MIRROR PLANE **and** a LIVE
_constitutive                                     off-diagonal row: the intersection
                                                  the other two leave uncovered.
                                                  ``offdiag_update_e`` refuses the
                                                  fold through clause 5; ``folded``
                                                  refuses the row because its body is
                                                  element-wise. Three inversions, one
                                                  per neighbour
folded_beta_real      all four      all four      a MIRROR PLANE **and** a nonzero
folded_beta_complex   all four      all four      ``grid.beta``, under real and
                                                  complex storage respectively. TWO
                                                  inverted clauses on every slot,
                                                  not one: clause 5 against
                                                  ``special_kz``'s two arms (which
                                                  refuse the fold by name) and
                                                  clause 12 against ``folded`` and
                                                  ``folded_complex`` (whose beta
                                                  clauses PRE-REGISTERED this family
                                                  as the owner of the slots they
                                                  decline). The two arms split from
                                                  each other on clause 2, and both
                                                  are CONSULTED on every folded beta
                                                  run so that exactly one says which
                                                  storage it needs. Their
                                                  constitutive rows are the
                                                  CERTIFIED bodies under a restated
                                                  predicate — neither the fold nor
                                                  beta reaches ``update_H``/diagonal
                                                  ``update_E`` at all. NO FILL ARM
                                                  IS REGISTERED: the two fold
                                                  families' fill predicates carry no
                                                  beta clause and already admit
                                                  these rows, so a third arm would
                                                  make both seam slots ambiguous
fused_dispersive      step_D        NOTHING       the FIRST FUSED product on this
_pair                 (UNWIRED)     (unwired)     backend, and the one row in this
                                                  table that is deliberately
                                                  UNSELECTABLE. It spans THREE
                                                  passes — ``step_D``,
                                                  ``zero_metal_D`` and ``update_E``
                                                  — and ``plan_step`` assigns at
                                                  most ONE arm per slot, so there is
                                                  no slot it could claim without a
                                                  composition rule nothing has
                                                  measured. It is therefore
                                                  registered ``wired=False``:
                                                  :func:`.arms.arms_for` skips it so
                                                  the composer cannot select it,
                                                  while :func:`.arms.registered`
                                                  still enumerates it so the
                                                  disjointness sweep can see it.
                                                  Its predicate is the CONJUNCTION
                                                  of ``pml_curl``'s step_D verdict
                                                  and ``dispersive_update_e``'s,
                                                  plus the seam clauses (an electric
                                                  source is refused BY NAME because
                                                  the driver injects it inside the
                                                  seam), so it can never admit a
                                                  configuration those two refuse.
                                                  The re-cut whole-step arbiter
                                                  reproduced its pre-registration
                                                  measurement EXACTLY — 60 cases,
                                                  44,243,856 uint32 comparisons,
                                                  zero divergences — which is the
                                                  evidence that this row changed no
                                                  selection
===================== ============= ============= ==============================

THE ROWS ARE CONTRIBUTED BY THE MODULES IN :data:`FAMILY_MODULES` (``launch``
registers two of them, ``special_kz`` two and ``folded_beta`` two). The first
twenty-three were measured 2026-08-16; the real fold's six, the folded complex's six
and the folded off-diagonal's one landed the same round, the cylindrical complex
family's four the round after, ``folded_beta``'s EIGHT — two curl pairs and two
constitutive pairs, and NO fill row — beside them, and ``cylindrical_real``'s FOUR
in the same tranche as the last two.

THE SWEEP ARTIFACT IS THE AUTHORITY AND NO TOTAL IS SPELLED HERE. It records the
table it actually ran against, so a drift between this comment and the package is
visible in ``composition_sweep.json`` rather than only here — which is how this
paragraph was caught understating the table by six, and why the count that replaced
it was removed rather than re-incremented: a hand-maintained total in a tree several
families are landing into is a number that is wrong between commits by
construction, and a reader who trusts it is worse off than one who reads the
artifact.

TWO PAIRS IN THIS TABLE ARE DISJOINT THROUGH AN ENGINE COUPLING RATHER THAN AN
INVERTED CLAUSE, and both are the SAME coupling: ``constitutive`` vs
``offdiag_update_e``, and ``folded`` vs ``folded_offdiag_constitutive``. Each pair
refuses on the FLAG on one side and requires a LIVE ROW SLOT on the other, and the
two agree only because ``has_offdiagonal_epsilon`` is a read-only property over the
very dict the row accessor reads. The matrix therefore carries a PLANTED
disagreement for each pair — unreachable on the engine — whose expected outcome is
that ``update_E`` is left UNSELECTED with both claimants named.

The clause that separates each pair is an INVERTED clause, not an omission: every
family's grid-reason list carries the same numbered questions and answers exactly
one of them the other way. That is why the sweep finds no admitted overlap outside
the two planted cases, and why a new family must state its inversion rather than
inherit disjointness.

NOTHING HERE IS DISPATCH. ``meep_gpu.fastpath.plan_fast_path`` still returns
``None`` on every branch; this module decides only what ``plan_step`` may compose.
"""

from __future__ import annotations

from typing import Tuple

# THE IMPORTS ARE THE REGISTRATION. Alphabetical, and `launch` is in the list
# because it registers the two tranche-1 products (the real PML curl and the
# ordinary constitutive) and calls `no_pml_constitutive.register_arms()`.
from . import (  # noqa: F401 - imported for the registration side effect
    ade_update_p,
    beta_complex_fused_electric_pair,
    beta_complex_fused_magnetic_pair,
    beta_fused_electric_pair,
    beta_fused_magnetic_pair,
    bfast_curl,
    bfast_fused_electric_pair,
    bfast_fused_hd_pair,
    beta_complex_fused_hd_pair,
    beta_real_fused_hd_pair,
    bfast_fused_magnetic_pair,
    complex_conductive_fused_pair,
    complex_conductive_pml,
    complex_fields,
    complex_folded_offdiag_update_e,
    complex_fused_ade_chain,
    complex_fused_electric_pair,
    complex_fused_hd_pair,
    complex_fused_magnetic_pair,
    complex_no_pml_curl,
    complex_no_pml_conductive,
    complex_no_pml_offdiag_fused_electric_pair,
    complex_no_pml_offdiag_update_e,
    complex_dispersive_update_e,
    complex_dispersive_spine,
    complex_no_pml_stored_e,
    conductive_fused_electric_pair,
    conductive_fused_hd_pair,
    folded_complex_fused_hd_pair,
    conductive_pml,
    cylindrical_complex,
    cylindrical_complex_fused_hd_pair,
    cylindrical_fused_electric_pair,
    cylindrical_fused_magnetic_pair,
    cylindrical_real,
    cylindrical_real_fused_electric_pair,
    cylindrical_real_fused_hd_pair,
    cylindrical_real_fused_magnetic_pair,
    dispersive_update_e,
    folded_beta,
    folded_beta_complex_fused_magnetic_pair,
    folded_beta_real_fused_magnetic_pair,
    folded_beta_real_fused_pair,
    folded_beta_complex_fused_pair,
    folded_complex,
    folded_complex_fused_magnetic_pair,
    folded_complex_fused_pair,
    folded_complex_offdiag_fused_electric_pair,
    folded_fused_dispersive_pair,
    folded_fused_hd_pair,
    folded_fused_magnetic_pair,
    folded_fused_pair,
    fused_ade_chain,
    fused_dispersive_pair,
    fused_electric_pair,
    fused_hd_pair,
    fused_magnetic_pair,
    folded_dispersive_update_e,
    folded_offdiag_dispersive_update_e,
    folded_offdiag_fused_electric_pair,
    folded_offdiag_update_e,
    launch,
    no_pml_conductive_fused_electric_pair,
    no_pml_curl,
    no_pml_constitutive,
    no_pml_conductive,
    no_pml_fused_electric_pair,
    no_pml_stored_e,
    nonlinear_fused_magnetic_pair,
    nonlinear_update_e,
    offdiag_fused_electric_pair,
    offdiag_update_e,
    special_kz,
    symmetry,
)

#: Every module whose import contributes rows to the table. Named so a test can
#: assert the package holds no OTHER module that registers an arm — a family added
#: to the tree but not to this tuple is invisible to ``plan_step``, which is a
#: silent coverage loss rather than an error.
FAMILY_MODULES: Tuple[str, ...] = (
    "ade_update_p",
    "beta_complex_fused_electric_pair",
    "beta_complex_fused_magnetic_pair",
    "beta_fused_electric_pair",
    "beta_fused_magnetic_pair",
    "bfast_curl",
    "bfast_fused_electric_pair",
    "bfast_fused_hd_pair",
    "beta_complex_fused_hd_pair",
    "beta_real_fused_hd_pair",
    "bfast_fused_magnetic_pair",
    "complex_conductive_fused_pair",
    "complex_conductive_pml",
    "complex_fields",
    "complex_folded_offdiag_update_e",
    "complex_fused_ade_chain",
    "complex_fused_electric_pair",
    "complex_fused_hd_pair",
    "complex_fused_magnetic_pair",
    "complex_no_pml_curl",
    "complex_no_pml_conductive",
    "complex_no_pml_offdiag_fused_electric_pair",
    "complex_no_pml_offdiag_update_e",
    "complex_dispersive_update_e",
    "complex_dispersive_spine",
    "complex_no_pml_stored_e",
    "conductive_fused_electric_pair",
    "conductive_fused_hd_pair",
    "folded_complex_fused_hd_pair",
    "conductive_pml",
    "cylindrical_complex",
    "cylindrical_complex_fused_hd_pair",
    "cylindrical_fused_electric_pair",
    "cylindrical_fused_magnetic_pair",
    "cylindrical_real",
    "cylindrical_real_fused_electric_pair",
    "cylindrical_real_fused_hd_pair",
    "cylindrical_real_fused_magnetic_pair",
    "dispersive_update_e",
    "folded_beta",
    "folded_beta_complex_fused_magnetic_pair",
    "folded_beta_real_fused_magnetic_pair",
    "folded_beta_real_fused_pair",
    "folded_beta_complex_fused_pair",
    "folded_complex",
    "folded_complex_fused_magnetic_pair",
    "folded_complex_fused_pair",
    "folded_complex_offdiag_fused_electric_pair",
    "folded_fused_dispersive_pair",
    "folded_fused_hd_pair",
    "folded_fused_magnetic_pair",
    "folded_fused_pair",
    "fused_ade_chain",
    "fused_dispersive_pair",
    "fused_electric_pair",
    "fused_hd_pair",
    "fused_magnetic_pair",
    "folded_dispersive_update_e",
    "folded_offdiag_dispersive_update_e",
    "folded_offdiag_fused_electric_pair",
    "folded_offdiag_update_e",
    "launch",
    "no_pml_conductive_fused_electric_pair",
    "no_pml_curl",
    "no_pml_constitutive",
    "no_pml_conductive",
    "no_pml_fused_electric_pair",
    "no_pml_stored_e",
    "nonlinear_fused_magnetic_pair",
    "nonlinear_update_e",
    "offdiag_fused_electric_pair",
    "offdiag_update_e",
    "special_kz",
    "symmetry",
)
