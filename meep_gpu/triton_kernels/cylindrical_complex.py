"""COMPLEX cylindrical (Dcyl, complex64 storage at every m) PML curl tranche — ONE
new kernel, no new constitutive kernel.

THE m = 0 ARM, ADDED 2026-09-04. The family was |m| >= 1 only and refused m = 0
by name as ``cylindrical_triton``'s row; that was right while every m = 0 corpus
row stored real float32, and stopped being right when the extended census priced
``examples:dipole_in_vacuum_cyl_off_axis.py`` — grid (150, 1, 300), m = 0,
``force_complex_fields=True``, r-high and z absorbers — which no arm on any
backend admitted: the real m = 0 kernel refuses complex storage and this one
refused m = 0. ``M_ZERO`` is the third ``M_CLASS`` arm: the i*m/r block is
compiled out (the array path never forms it at m = 0, stepping :348/:430), no
axis-row increment exists, and the post-recurrence axis rules are the m = 0 pair
``cylindrical_triton`` already carries for real storage — ``Bx[0] = 0`` on B;
``Dz[0] += (4*Courant)*Hp[0]`` then ``Dy[0] = 0`` on D (:581-585, :661-662) —
applied to complex word pairs, the add being the coefficient-LEFT zero-imaginary
product NEP-50 makes of a Python float times a complex64 row. The two cylindrical
products are now disjoint BY STORAGE (``force_complex_fields``), not by m; the
predicates say so in both directions and the planner's sweep measures it. The
gate's m = 0 rows, the two fused pairs' m = 0 cases and the composition probe's
m = 0 cases (one at the corpus row's own shape, Courant and termination) are the
measurement; see ``results/triton_regate_2026-09-04_cylm0/``.

DISPATCH IS NOT WIRED, NO BYTE-IDENTITY CLAIM YET. Production dispatch is
untouched: ``fastpath.plan_fast_path`` still returns ``None`` on every branch and
this module does not touch it, so no production step can reach this family.
``launch.plan_step`` composition is NO LONGER DEFERRED — it consults these
predicates through lazy forwarders behind a gate on the cylindrical axis AND
complex storage, and the package ``__init__`` exports them — and what made the
deferral safe is what keeps the wiring safe: ``coverage._grid_reasons`` clauses 2
and 6, ``complex_fields._complex_grid_reasons`` clause 4 and
``cylindrical_triton``'s ``m != 0`` refusal mean no shipped predicate admits a
configuration this one also admits, which the planner's disjointness sweep
measures on real Dcyl triples rather than assuming. ``fingerprints.json`` carries
no entry for this file — the byte gate binds its own provenance record inside its
results directory instead. Callers are the gate
(``parity/meep_gpu/gate_triton_cylindrical_complex.py``), the planner and the
laptop tests. **NO DEVICE RUN HAS BEEN TAKEN**: everything
claimed below as MEASURED was measured on NumPy on the reference laptop, and the
only claim that carries is "the reference leg equals the shipped array path".
Bit-identity of the Triton kernel is UNPROVEN until the gate runs on hardware.

WHAT THIS IS FOR. Sixteen lifted corpus rows demand it — the LARGEST single
remaining coverage gap and 16 of the 34 admitted-nowhere rows, 64 sub-step slots
(``results/predicate_coverage_2026-08-12/analysis.txt``: "16 rows / 64 sub-steps
axis <n> boundary 'axis' is outside ('periodic','metallic')"). Nine examples
(cylinder_cross_section, disc_extraction_efficiency, disc_radiation_pattern,
extraction_eff_ldos, perturbation_theory, planar_cavity_ldos, point_dipole_cyl,
ring-cyl, zone_plate) and seven tests (TestLDOS.test_ldos_cyl,
TestLDOS.test_ldos_ext_eff, TestRingCyl.test_ring_cyl,
TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields idx2,
TestPMLCylindrical.test_pml_cyl idx1/2/3). The three REAL-storage Dcyl rows in
the same lift (test_pml_cyl idx0 at m = 0, test_adjoint_solver_cyl idx0/idx1)
belong to ``cylindrical_triton`` and are refused here by name.

The demand spans BOTH |m| >= 1 classes and BOTH z terminations, which is why
neither is compiled in (the m = 0 class is the 2026-09-04 addition above):

* |m| = 1 — the axis-row increment classes (m = -1 in seven rows, m = +1 in one);
* |m| >= 2 — the near-axis zeroing (m = 2, m = 3, m = 5 in perturbation_theory,
  m = 3 in ring-cyl), including one ``accurate_fields_near_cylorigin=True`` row
  (test_pml_cyl idx3, m = 3 at Courant 1/3.6);
* z METALLIC (13 rows) and z PERIODIC (perturbation_theory, ring-cyl,
  TestRingCyl — 3 rows).

None of the sixteen carries a susceptibility, a conductivity, BFAST, chi2/chi3,
an off-diagonal epsilon or a Bloch phase; every one is complex64 storage under an
active split-field PML with electric sources only. Everything else is refused
outright rather than carried.

======================================================================
THE FINDING: 32 OF THE 64 SLOTS NEED NO NEW KERNEL
======================================================================

``stepping.update_H`` (:907-924) and ``stepping.update_E`` (:926-993) carry NO
cylindrical branch of any kind — grep the functions: the words "cylindrical",
"is_axis" and "m" do not appear. They are element-wise with per-axis coefficient
tables indexed on the component's OWN axis, and Dcyl changes nothing there except
that axis 1 has n = 1. So the CERTIFIED ``complex_fields.bloch_constitutive_step``
body (job 2330/2343) already computes them, and this file adds only a restated
predicate — :func:`cylindrical_complex_constitutive_coverage` — plus a plan
builder that delegates to :class:`complex_fields.ComplexConstitutivePlan`. That is
the same shape ``cylindrical_triton.cylindrical_constitutive_coverage`` took for
real m = 0 over ``kernels.constitutive_step``, and the same shape special_kz took
for its beta runs.

THE IDENTITY LEG THAT PROVES IT, and it is measured rather than argued (the
project's rules refuse an argument from absence): ``stepping.update_H`` /
``update_E`` on a real complex Dcyl ``Fields``/``PML`` against a plain complex
elementwise ``dsigw`` reference carrying NO cylindrical clause —
**480/480 rows, 0 differing uint32 words**, 6 sub-steps each, over
m in {1, -1, 2, -2, 3, 5} x {z metallic, z periodic} x five Courant numbers (four
non-power-of-two) x two shapes x {random, signed-zero} seeding x {H, E}. Cut on
NumPy 2026-08-13. THE HARNESS IS NAMED AND RUNNABLE: ``run_constitutive_identity``
in ``parity/meep_gpu/gate_triton_cylindrical_complex.py``, reached by
``--self-check`` or ``--legs constitutive``, and the row count is computed from
the leg's own axes (``constitutive_identity_row_count``) so it cannot drift from
the loop. An earlier revision of this paragraph claimed 640 rows over an m set
the harness does not carry and cited a ``--identity`` flag that never existed.

EVERY ROW OF IT ASSERTS ITS OWN NON-VACUITY, because half of them used not to
have any: the first cut swept ``zero_init`` as a seeding here, and a zero-init
constitutive row CANNOT FAIL — the sub-step is ``fw = value; field += kps*fw;
field -= kms*prev``, which from an all-zero state leaves every word +0.0 forever,
so a reference with its coefficients deliberately swapped still reported
IDENTICAL on all 256 of them. The seeding is now a ±0 LATTICE, each row asserts
``moved_words > 0`` and a nonzero ±0 census (measured minima over the leg: 1,260
moved words, 952 census words), and ``run_constitutive_mutations`` breaks the
reference on purpose and requires every seeding to catch it. The CuPy re-cut is
the arbiter for the device bytes; this establishes that no cylindrical ARITHMETIC
is missing from the delegation.

======================================================================
THE CURL HALF DOES NEED A NEW KERNEL — MEASURED, NOT ASSUMED
======================================================================

The certified ``complex_fields.bloch_pml_curl_step`` alone, run on a Dcyl
|m| >= 1 grid with every cylindrical clause stripped, is **1,281,955 differing
uint32 words** against the array path — 56 of 80 rows differ, 8 sub-steps each.
HARNESS: ``run_stripped_control`` in the gate (``--self-check``), which switches
off the six cylindrical clauses by name and forces r metallic, so what remains is
exactly the certified body's arithmetic. It FAILS THE GATE if that count is ever
zero, because a zero would mean this module should not exist. The four things the
certified body is missing are enumerated below.

BUT ONE THING IT IS NOT MISSING: **the r axis needs no new ghost code.**
``stepping._boundary_kinds`` returns ``CYL_AXIS`` on r, and
``_shift_up``'s CYL_AXIS branch shares the METALLIC one exactly — a hard zero at
the far r face (:1781-1783) — while ``_shift_down``'s CYL_AXIS branch
(:1830-1842) writes the ``r_to_minus_r`` image into the near ghost, which is
UNOBSERVABLE (see "THE NEAR GHOST" below). Measured: running the reference with
``boundaries[0]`` forced to ``'metallic'`` — i.e. the certified kernel's
``BCX = METALLIC`` constexpr arm — is **0 differing words across 80/80 rows**
(5 values of m x both z terminations x seeded/zero-init x two Courant numbers x
both sub-steps, 8 sub-steps each). HARNESS: ``run_axis_identity`` in the same
gate. So :data:`BCX` is bound to METALLIC and the cylindrical axis costs the
kernel no branch at all.

THE FOUR CYLINDRICAL ADDITIONS, in the order the kernel applies them, each with
its ``stepping.py`` line range:

1. **The radial prefix**, exactly as ``cylindrical_triton`` carries it, and for
   the same measured reason: the scan STAYS ON THE ARRAY PATH.
   * B side (:299-347): Ep = Ey extended by ONE ZERO WALL ROW to (nr+1, 1, nz),
     prefixed at ``ir0 = 0.0``; ``Bz``'s WHOLE curl becomes
     ``dtdx * (prefix_ext[1:] - prefix_ext[:-1])`` — ONE subtract then ONE
     complex multiply, NOT the four-operand grouping.
   * D side (:414-426): Hp = Hy prefixed at ``ir0 = 0.5``, substituted for the
     ``Dz`` term's ``first`` source ONLY; ``Dx``'s Hy operands stay raw.
   ``cupy.cumsum`` in float32 is deterministic but is NOT a sequential
   accumulation, and ``tl.cumsum`` matches neither — the ORACLE is CuPy's scan
   order (cylindrical_triton.py:32-56, measured). :func:`cylindrical_complex_prefix`
   calls ``stepping.cylindrical_rderiv_prefix``, it does not re-derive it.
   The prefix is a COMPLEX volume here, where the m = 0 kernel's was real; the
   scan is dtype-agnostic (``f_p.real.dtype`` picks the weight dtype, :1279).

2. **The i*m/r coupling** (:674-724, call sites :348-355 B / :430-437 D). Per
   real/imaginary part MEEP adds ``the_m / r * g[i]`` with ``g`` the OTHER part of
   the partner, and the part swap IS multiplication by -i, so as one complex
   update ``delta_f = -i * s * 2m * Courant / r2 * g`` with
   ``r2 = 2*ir + iyee_shift_r(target)``. Four call sites and only four:
   ``Bx <- Ez`` at s = +1, ``Bz <- Ex`` at s = -1, ``Dx <- Hz`` at s = -1,
   ``Dz <- Hx`` at s = +1; targets 1 (``By``/``Dy``) get nothing.
   THE PARTNERS ARE CENTER LOADS THE CERTIFIED KERNEL ALREADY MAKES: per
   ``B_CURL_TERMS``/``D_CURL_TERMS`` (:213-223) ``Ez``/``Hz`` is the ``g2``
   center (register ``c``) and ``Ex``/``Hx`` is the ``g0`` center (register
   ``a``), so the term costs no new pointer and no extra traffic — the same
   property special_kz's beta term has. The Dz partner is the RAW ``Hx``, never
   the prefixed operand.

3. **The |m| = 1 axis-row increments** (:625-645 B / :524-557 D), which REPLACE
   curl row 0 rather than adding to it, and do so AFTER the ownership mask:
   * B: ``Bx`` row 0 takes ``(-dtdx)*(Ep[0] - Ep[0, z+1]) - (1j*m*dtdx)*Ez[1]``,
     where ``Ez[1]`` is the FIRST OFF-AXIS r row (MEEP's ``f[Ez][1-cmp] +
     (nz+1)``) and the z shift-up obeys the z ghost rule;
   * D: ``Dy`` row 0 takes ``dtdx*(Hr[0] - Hr[0, z-1] - 2.0*Hz[0])``, the factor
     2 being the doubled-ivec compensation.
   Folded into the curl (not post-added) so the PML recurrence applies MEEP's own
   axis ladder — :524-543 records the measurement that forced it (a plain
   post-add is Ep 4.6e-01 / Hr 2.39 wrong under r+z PML at |m| = 1).

4. **The per-|m| axis rules**, applied to the STORED value after the recurrence:
   * |m| = 1 — B side: NOTHING (``_cylindrical_axis_zero_B`` :648-671 branches on
     ``m == 0`` and ``abs(m) > 1`` only). D side: ``Dz[0] = 0``, the FIELD only,
     never ``fu_Dz``.
   * |m| >= 2 (:560-598, :648-671, rows from :601-622) — ALL THREE components of
     the family AND their ``fu_`` auxiliaries held at zero on rows
     ``[0:|m|]``, or on row 0 alone when ``accurate_fields_near_cylorigin``.
     The component set is the POST-#3164 one (upstream 593a4b42, first released
     in 1.33.0); the 1.29-era code zeroed only Dp/Dz and Br and the two
     references genuinely differ (:573-578).

THE NEAR GHOST IS NOT IMPLEMENTED, AND THAT IS A DECISION WITH A PROOF BEHIND IT
— the same decision ``cylindrical_triton`` took at m = 0, re-established here for
complex storage at every |m| this family carries. The only terms taking a
shift-down along r are ``Dy`` (partner Hz, second operand) and ``Dz`` (partner
Hp, first operand), and BOTH have Yee r-shift 0, so ``_mask_non_owned_cells``
(:1898-1902) zeroes their curl at row 0 — the only row the near ghost writes. At
|m| = 1 that row is then overwritten outright by the axis increment; at |m| >= 2
the field AND its ``fu`` are zeroed there. Measured on the reference leg: two
armed mutations — inverting the ``r_to_minus_r`` direction sign, and dropping the
``(-1)^m`` factor ``_mirror_phases`` (:2301-2309) threads into the same slot —
are **NULL AS PREDICTED, 0 differing words** over the whole matrix INCLUDING the
zero-init rows. The kernel serves an exact 0.0 there (the ``other=`` of a masked
load) and says so HERE, because **no byte gate can certify that choice**: if a
future run ever reports either mutation CAUGHT, the ownership mask has broken,
not the ghost.

======================================================================
WHAT THE COMPLEX TRANCHE CONTRIBUTES UNCHANGED
======================================================================

Everything about complex storage is composed, not rewritten:

* complex64 as float32 WORD PAIRS at ``2*idx`` / ``2*idx + 1``, and the halved
  int32 word bound (``_complex_layout_reasons``, complex_fields.py:763-783);
* :func:`complex_fields._mul_coefficient_left` for the ``dtdx`` scalar-left curl
  multiply (S:1635) and :func:`complex_fields._mul_field_left` for the
  ``fu *= kms`` family (S:1929-1935) — the zero-imaginary products with the
  LITERAL zero cross terms, whose ``* -1.0`` addend spelling is load-bearing
  (never unary minus: Triton lowers ``-x`` as ``0.0 - x`` and canonicalizes ±0);
* the ``EXPANSION`` constexpr and its PROBE ARTIFACT contract, base four patterns
  (:data:`complex_fields.PROBE_PATTERNS`), unchanged and unextended — see below;
* PHX/PHY/PHZ = 0 always. ``Grid`` refuses a k on r or phi (grid.py:654-658) and
  none of the sixteen rows carries a z k_point, so the predicate requires
  ``has_bloch`` false and the rotation compiles away entirely. Admitting a z
  Bloch phase later is a predicate widening plus a gate row, not a kernel change.

**NO NEW PROBE PATTERN IS NEEDED, and that is measured rather than assumed** —
which is the one place this family is CHEAPER than special_kz, whose beta
coefficient forced :data:`special_kz.BETA_PROBE_PATTERN`. Two facts, both cut on
NumPy 2026-08-13:

* the i*m/r coefficient's real word is EXACTLY +0.0 for every (m, sign, r)
  measured (m in {+-1, +-2, +-3, +5}), and the reason is a spelling asymmetry
  worth stating because it is invisible: ``stepping`` writes the numerator
  ``(-1j) * (sign*2*m*dtdx)`` (:717) with the imaginary unit on the LEFT, and
  CPython's complex product then computes ``re = (-0.0)*X - (-1.0)*0.0``, whose
  cross-term subtraction LAUNDERS the sign to +0.0 for both signs of X. The beta
  term writes ``coefficient * (1j)`` (:772) with the unit on the RIGHT, giving
  ``re = X*0.0 - 0.0*1.0``, which PRESERVES the sign (-0.0 for X < 0). Measured
  both ways;
* for a PURE-IMAGINARY coefficient the FMA_V1 and NAIVE arms are DEGENERATE —
  ``fma(+0.0, z_re, -(c_im*z_im))`` and ``(+0.0*z_re) - (c_im*z_im)`` are the
  same single-rounding operation. Measured 0 differing words on the gate's own
  signed-zero and underflow vectors, and the gate's classifier reports
  ``AMBIGUOUS_BOTH`` for the pattern in all three orientations (broadcast row
  vector, materialized contiguous, and field-left).

So the i*m/r product binds whatever ``EXPANSION`` the base four patterns license
and the arm choice cannot change its bytes.

THE ZERO CROSS TERMS ARE KEPT, AND THE CHOICE IS BYTE-INVISIBLE HERE — which is a
correction, not a hedge. The plane-wise shortcut ``{-c_im*z_im, c_im*z_re}``
really does produce different PRODUCT words (``(+0.0*z_re) - X`` is +0.0 where
``X * -1.0`` is -0.0 at X = +0.0), and an earlier revision of this docstring took
that isolated 4/16-word measurement for a statement about the sub-step and had
the gate predict a CATCH. It is not: the fold is ``curl - m``, and ``x - (+0.0)``
differs from ``x - (-0.0)`` only at ``x = -0.0``, while EVERY i*m/r minuend in
this kernel is provably never -0.0 —

* target 0's curl leads with the phi self-difference ``(c_p - c)``, which is an
  exact +0.0 on a one-cell axis, and ``+0.0 + y`` is +0.0 even at ``y = -0.0``;
* target 2's is either a prefix cumsum on the B side (a scan whose first output
  is +0.0 can never produce -0.0, by induction on ``prefix[i] = prefix[i-1] +
  increment[i]``) or the same +0.0-led sum on the D side.

MEASURED, not argued: the gate's ``run_minuend_census`` scans every i*m/r minuend
over both seedings, both sub-steps, four m values and both z terminations and
reports **0 negative-zero words out of 491,520 scanned**, and the mutation
:data:`MUTATION_IMR_PLANEWISE` is recorded as a PREDICTED NULL with that census
as its evidence, on both the reference and the kernel layer (the kernel spelling
mutates BOTH call sites, so the null is a statement about the family). The choice
is therefore pinned where a byte-invisible choice can be pinned — by a SOURCE-TEXT
assertion in ``meep_gpu/test_triton_cylindrical_complex.py`` — which is the BFAST
tranche's precedent for its own three invisible choices. If a future change ever
makes that census nonzero, the null is no longer predicted and the needle must be
re-armed as a catch; the gate says so in the record.

======================================================================
GROUPING CHOICES THE GATE MUST HOLD (stepping.py forces none of these)
======================================================================

EVERY ONE OF THE NINE NAMES ITS NEEDLE, and the needle names are DATA
(:data:`GROUPING_CHOICE_NEEDLES` in the gate, checked against the three mutation
batteries by ``check_needle_layers`` and by a laptop test). Two of them used to
be pinned by this prose alone — one of those cited a word count for a mutation
that existed in NEITHER battery — which is why the mapping is now mechanical.
Where a choice is BYTE-INVISIBLE on the measured matrix it says so with its
measurement and is pinned by a source-text assertion in the test file; claiming a
byte gate holds an invisible choice is the artifact-overclaim inversion the BFAST
tranche names.

1. THE COEFFICIENT ROW VECTOR IS HOST-BUILT AND BOUND AS A DEVICE ARRAY, one
   complex64 entry per r row, per (target, sign). The array path forms it in
   float64 — an ``arange``, a ``maximum(.., 1.0)`` clamp, a complex128 numerator
   and a float64 DIVISION — and rounds ONCE to the storage dtype (:713-718).
   Recomputing it in-kernel from an fp32 ``dtdx`` is a different number
   (``imr_dtdx_prerounded_to_f32``: CAUGHT, 15,264 words), and the division would
   additionally have to be ``tl.math.div_rn`` (f32 ``/`` lowers to
   ``div.full.f32``, ~2 ulp, not correctly rounded). Binding the row makes the
   whole question disappear: THE KERNEL PERFORMS NO FLOATING-POINT DIVISION AT
   ALL. (It does carry integer ``//`` on the lane index at :612-614, which is
   exact and is not what platform fact (e) is about; an earlier revision said
   "there is no division in this kernel" without the qualifier.)
2. The clamp ``maximum(2*ir + iyee_r, 1.0)`` is part of the bound row, not a
   kernel branch, and it is a DOMAIN GUARD rather than arithmetic: it bites only
   where the doubled coordinate is 0, i.e. row 0 of a shift-0 target (``Bx``,
   ``Dz``), whose curl the ownership mask zeroes anyway. MEASURED both ways —
   DROPPING it is a predicted null (``imr_clamp_dropped``: 0 words; the
   divide-by-zero's non-finite word never reaches a compared volume), while
   RAISING it to 2.0 is a catch (``imr_clamp_value_raised``: 12,032 words),
   because that also moves row 0 of the shift-1 targets ``Bz`` and ``Dx``, whose
   row 0 the mask does NOT zero.
3. ``curl + (-(factor * g))`` is spelled as a single subtract per plane.
   IEEE-754 defines subtraction AS addition of the negation, so the two are the
   same bits on every input including signed zeros, and the spelling never leans
   on how Triton lowers unary minus. VERIFIED as a predicted null on the
   reference leg (``imr_negation_as_subtract``: 0 differing words), the same
   grouping choice special_kz makes.
4. The |m| = 1 axis increment REPLACES curl row 0 (``tl.where``), it does not
   accumulate. After the mask that row is exactly +0.0, and ``+0.0 + x == x``
   for every x EXCEPT ``x = -0.0`` — so the two spellings differ only on signed
   zeros. MEASURED: ``axis_increment_accumulates`` is **NEEDLE-MISSED under
   random seeding (0 words)** and **CAUGHT under zero-init with a thin
   negative-``kms`` absorber (16 words)**. See the zero-init note below; this is
   the single most important gate-design fact in the family.
5. The increment's own negation is ``* -1.0`` per word, never unary minus, and
   the increment is computed BEFORE negation exactly as the array path computes
   it: ``-(A - B)`` is NOT ``B - A`` when both are +0.0. THE NEEDLE HAS TO SPELL
   BOTH SIDES, and that is measured: swapping only the B-side operands is a null
   (0 words — ``Bx``'s split-field dsig axis is phi, where ``kms`` is 1.0 and
   never negative, so the ``fu*kms`` minuend is +0.0 and the difference is
   laundered), while the same swap on the D side is CAUGHT (16 words under
   zero-init) because ``Dy``'s dsig axis is z and does go negative under the thin
   absorber. ``axis_increment_not_negated`` is a catch on both (12,288 words).
6. The axis increment is applied AFTER the ownership mask and the i*m/r add, and
   the i*m/r add is applied BEFORE the mask. BOTH orderings are armed on both
   layers now: ``imr_after_mask`` is CAUGHT (12,672 words) and
   ``axis_increment_before_mask`` is CAUGHT (12,288 random / 16 zero-init). An
   earlier revision cited "imr_after_mask: 9,504 words" for a mutation that did
   not exist in any battery.
7. ``BCZ`` is a constexpr, NOT compiled in: three of the sixteen rows terminate z
   PERIODIC and thirteen METALLIC. Forcing it METALLIC is a catch
   (``bcz_forced_metallic``: 24,000 words). ``BCX`` is compiled to METALLIC (the
   r-axis identity above, 0 words over 80/80 rows) and ``BCY`` to PERIODIC (phi
   is the one-cell invariant axis).
8. THE INVARIANT-AXIS DIFFERENCE IS COMPUTED, NEVER ELIDED. phi has n = 1, so the
   rolled operand equals the original and ``(second - shifted_second)`` is
   exactly +0.0; a kernel that "optimizes it away" returns ``dtdx*(a-b)`` where
   the array path returns ``dtdx*((a-b) + 0.0f)``, which differ exactly when
   ``(a-b)`` is -0.0 (cylindrical_triton.py:137-144, and the same trap holds per
   plane in complex storage). MEASURED BYTE-INVISIBLE ON THIS MATRIX:
   ``invariant_axis_difference_elided`` is 0 words over both seedings, because
   the difference survives only at a -0.0 curl AND a -0.0 recurrence minuend, and
   a curl-only leg never updates its sources — under zero-init every operand word
   is +0.0, and a random seed hits an exact -0.0 difference with probability
   zero. Kept because it is what the array path computes, recorded as a predicted
   null with that reason, and pinned by source text.
9. ``M_CLASS`` is a constexpr with three arms — 0 for m = 0 (complex storage;
   since 2026-09-04), 1 for |m| = 1, 2 for |m| >= 2. ``ZERO_ROWS`` carries
   ``|m|`` or 1 (the accurate branch) as a compile-time integer
   (``zero_rows_accurate_always``: CAUGHT, 55,268 words). The m = 0 arm's own
   needles are ``m0_bx_axis_zero_dropped``, ``m0_dz_axis_add_dropped``,
   ``m0_dy_axis_zero_dropped`` and ``m0_dz_axis_add_uses_dtdx``, armed on both
   the reference and the kernel layer.

======================================================================
THE ZERO-INIT CLASS, AND WHY A THICK ABSORBER MAKES IT VACUOUS
======================================================================

Grouping choice 4 is byte-visible only on a stored -0.0, and random seeds are
provably blind to that class. The reachable producer is the complex tranche's:
a negative ``kms = kappa - sigma`` times a quiet +0.0 word. MEASURED on this
family's own grids, 20x1x24 at Courant 0.37:

* absorber 2 cells — ``min kms_z = -1.396`` (integer lattice) / -0.348 (half):
  signed-zero census peaks at **85 words**, the accumulate mutation is **CAUGHT**;
* absorber 3 cells — ``min kms_z = -0.597`` / -0.109: census **85**, **CAUGHT**;
* absorber 5 cells — ``min kms_z = +0.0415`` / +0.224: census **0**, the mutation
  reports NEEDLE-MISSED **for a reason that has nothing to do with the mutation**.
  That row is VACUOUS and must be reported as such, never as a pass.

The gate therefore seeds its zero-init rows with a THIN absorber, asserts a
non-zero census IN RUN, and fails the case as VACUOUS when the census is 0.

THE CONSTITUTIVE SUB-STEP HAS NO SUCH PRODUCER, and that is why its leg seeds a
±0 LATTICE instead of zero-initialising. ``fw = value; field += kps*fw; field -=
kms*prev`` from an all-zero state leaves every word +0.0 forever — ``+0.0 +
(-0.0)`` is +0.0, and so is ``+0.0 - (±0.0)`` — so a zero-init constitutive row
carries no census to assert and no mutation can move it. Measured: with the first
cut's zero-init seeding, a reference whose ``kps``/``kms`` were deliberately
swapped still reported IDENTICAL on every such row. The rule the family follows
is the general one: **a seeding that catches nothing across a whole mutation
battery is VACUOUS, whatever its individual rows say**, and both batteries now
report per-seeding totals and fail on a zero.

A SECOND requirement falls out of the same measurement: **the comparison must be
per sub-step, not at the end of the budget.** The divergence this class produces
is transient — a -0.0 that survives one sub-step is laundered back to +0.0 by the
next ``fu *= kms`` sign flip — and an end-of-run compare reported 0 differing
words on a state that had genuinely diverged at step 1 (measured; it is why the
first cut of the mutation battery called grouping choice 4 a needle-miss).

======================================================================
STEP BUDGET
======================================================================

No budget may be claimed for this family yet: **no device run has been taken.**
When one is, the claimable budget is whatever the gate measures and no more —
``cylindrical_triton``'s own consecutive-step leg first diverged at step 23-72 on
a subnormal-against-flushed-zero, which is the amplifying disagreement of plan
§16 and not a defect; this family inherits that exposure because it shares the
array-path prefix and the same recurrence. The gate's ``multi_step`` leg carries
the budget as a number (``MULTI_STEP_BUDGET``) and computes the CLAIMABLE budget
from the first divergence it observes, so the artifact can never state a budget
larger than what ran.

WHAT EXISTS AND WHAT DOES NOT, stated because the first revision of this file got
it wrong in the direction that flatters: the gate's LAPTOP half (reference
transcription, constitutive identity, both mutation batteries, the r-axis and
stripped-complex identity controls, the needle-arming check) is written, wired to
``main`` and PASSING. Its DEVICE half (``synthetic``, ``guard``, ``multi_step``,
``engine``, ``mutations``, ``host_mutations``) is written and wired and has NEVER
RUN: on a host without CuPy and Triton every one of them skips with its reason
recorded, and the artifact's ``byte_identity_claim`` field says NONE. The earlier
revision described those legs in prose while ``main`` routed only ``--self-check``
and the whole kernel-mutation battery was unreachable code.

Import contract: this module is importable WITHOUT Triton — the predicates and
plan builders (to ``None``) must answer on the laptop that is the merge bar.
Triton is imported at module scope inside a guard, the kernel degrades to
``complex_fields._UnavailableKernel``, and ``ENABLE_FP_FUSION`` is imported from
:mod:`kernels` only inside ``run()`` so the guard keeps its single spelling.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .complex_fields import _UnavailableKernel

try:  # The host predicate and public refusal path must work without Triton.
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - exercised by the absence test
    _TRITON_IMPORT_ERROR = _exc

    class _MissingTriton:
        @staticmethod
        def jit(function):
            return _UnavailableKernel(function.__name__, _TRITON_IMPORT_ERROR)

    class _MissingLanguage:
        @staticmethod
        def constexpr(value):
            return value

    triton = _MissingTriton()  # type: ignore[assignment]
    tl = _MissingLanguage()  # type: ignore[assignment]

from .complex_fields import (  # READ-ONLY: restate, never edit, the certified base
    ComplexConstitutivePlan,
    _complex_layout_reasons,
    _expansion_reasons,
    _mul_coefficient_left,
    _mul_field_left,
    _resolve_expansion,
    _word_view,
)
from .coverage import (  # READ-ONLY imports; nothing here mutates coverage.py
    CONSTITUTIVE_SIDES,
    CURL_SUB_STEPS,
    CURL_TARGETS,
    Coverage,
    _boundary_kinds,
    _call,
    _coefficient_reasons,
    _inverse_epsilon_reasons,
    _susceptibility_reasons,
)
from .launch import SUB_STEPS, CupyPointer, _flat

# Same constexpr codes as ``kernels.PERIODIC``/``kernels.METALLIC``. Restated
# rather than imported for the reason complex_fields.py:192-194 gives: importing
# kernels.py pulls Triton in unconditionally and defeats the host-only route.
PERIODIC = tl.constexpr(0)
METALLIC = tl.constexpr(1)

#: ``M_CLASS`` arms. Three, since 2026-09-04. ``M_ZERO`` is the m = 0 arm UNDER
#: COMPLEX64 STORAGE (``force_complex_fields=True``, which the array path honours
#: at m = 0 exactly as at |m| >= 1 — fields.py:573): the i*m/r coupling is
#: compiled OUT (stepping :348/:430 branch on ``grid.m != 0``), no axis-row
#: increment exists, and the axis rules are the m = 0 pair ``Bx[0] = 0`` /
#: ``Dz[0] += (4*Courant)*Hp[0]; Dy[0] = 0`` (stepping :661-662 / :581-585) —
#: the same rules ``cylindrical_triton`` carries for REAL m = 0 storage, applied
#: here to complex word pairs. The two products stay disjoint BY STORAGE:
#: ``cylindrical_triton`` refuses ``force_complex_fields=True`` by name and this
#: family requires it.
M_ZERO = tl.constexpr(0)      # m = 0, complex storage: no coupling, m = 0 axis pair
M_ONE = tl.constexpr(1)       # |m| = 1: axis-row increments on Bx and Dy
M_MANY = tl.constexpr(2)      # |m| >= 2: near-axis zeroing of all six volumes

#: The boundary triple this family requires on r and phi, and the two it admits
#: on z. r is the cylindrical axis and is compiled as METALLIC — see the module
#: docstring's r-axis identity measurement.
REQUIRED_R_KIND = "axis"
REQUIRED_PHI_KIND = "periodic"
ADMITTED_Z_KINDS: Tuple[str, ...] = ("metallic", "periodic")

#: ``grid.is_axis`` on the three axes. Refused unless it is exactly this.
CYLINDRICAL_AXIS_FLAGS: Tuple[bool, bool, bool] = (True, False, False)

#: The smallest radial extent this product may step. The |m| = 1 axis increment
#: reads the FIRST OFF-AXIS row (``stepping.py:671``'s ``xp.take(Ez, 1, axis=0)``)
#: and the array path itself raises ``IndexError`` below this — see the clause in
#: :func:`_cylindrical_geometry_reasons` for the measurement.
MINIMUM_RADIAL_ROWS = 2

#: The i*m/r call sites — MEEP step_db.cpp:177-294 via stepping.py:376-383 (B)
#: and :459-466 (D). ``(target index, partner register, sign)``; the partner
#: register names which CENTER load of the certified curl body the term reuses
#: ("c" = g2 = Ez/Hz, "a" = g0 = Ex/Hx). Target 1 gets no term on either side.
IMR_TERMS: Dict[str, Tuple[Tuple[int, str, float], ...]] = {
    "step_B": ((0, "c", +1.0), (2, "a", -1.0)),   # Bx <- Ez (+1), Bz <- Ex (-1)
    "step_D": ((0, "c", -1.0), (2, "a", +1.0)),   # Dx <- Hz (-1), Dz <- Hx (+1)
}

#: Which target each sub-step's |m| = 1 axis-row increment REPLACES at r = 0.
AXIS_INCREMENT_TARGET: Dict[str, int] = {"step_B": 0, "step_D": 1}   # Bx / Dy

#: Per-sub-step prefix spec, transcribed from stepping.py:327-375 / :443-455.
PREFIX: Dict[str, Dict[str, Any]] = {
    "step_B": {"component": "Ey", "ir0": 0.0, "extend_wall_row": True},
    "step_D": {"component": "Hy", "ir0": 0.5, "extend_wall_row": False},
}

CURL_SOURCES: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

#: Elements per program — complex CELLS, not words. Restated from
#: ``kernels.DEFAULT_BLOCK`` for the same import reason; no autotune, because the
#: kernel writes its own inputs in place (kernels.py:56-59).
DEFAULT_BLOCK = 256

#: The gate mutation names this module's docstring commits to. Named as data so
#: the gate and the module cannot drift about which choices are pinned.
MUTATION_IMR_PLANEWISE = "imr_planewise_zero_cross_terms"

#: Choices this kernel makes that are BYTE-INVISIBLE on the gate's measured
#: matrix, with the reason each one is. They are pinned by SOURCE-TEXT assertions
#: in ``meep_gpu/test_triton_cylindrical_complex.py``, which is the layer that can
#: hold an invisible choice — a byte gate cannot, and saying it does is the
#: overclaim the BFAST tranche names. Named as data so the module, the gate's
#: predicted-null table and the tests cannot drift about which ones these are.
BYTE_INVISIBLE_CHOICES: Dict[str, str] = {
    MUTATION_IMR_PLANEWISE:
        "the plane-wise shortcut differs from the full product only in the ±0 "
        "class, and `curl - m` carries that class only at a -0.0 minuend; no "
        "i*m/r minuend in this kernel can be one (measured: 0 of 491,520 words)",
    "invariant_axis_difference_elided":
        "the elided phi addend is an exact +0.0, and the difference survives "
        "only at a -0.0 curl AND a -0.0 recurrence minuend; a curl-only leg "
        "never updates its sources, so neither seeding reaches the pair",
    "axis_increment_operands_swapped":
        "on the B side only: Bx's split-field dsig axis is phi, whose kms is "
        "1.0, so the +0.0 `fu*kms` minuend launders the ±0 difference. The D "
        "side of the same choice IS visible and is armed as a catch",
}

__all__ = [
    "ADMITTED_Z_KINDS",
    "AXIS_INCREMENT_TARGET",
    "BYTE_INVISIBLE_CHOICES",
    "CylindricalComplexCurlPlan",
    "IMR_TERMS",
    "MINIMUM_RADIAL_ROWS",
    "MUTATION_IMR_PLANEWISE",
    "PREFIX",
    "cylindrical_complex_constitutive_coverage",
    "cylindrical_complex_curl_coverage",
    "cylindrical_complex_prefix",
    "cyl_complex_pml_curl_step",
    "four_dtdx_scalar",
    "imr_coefficient_row",
    "m_class",
    "plan_cylindrical_complex_constitutive",
    "plan_cylindrical_complex_curl",
    "plan_cylindrical_complex_curl_from_arrays",
    "zero_rows",
]


# ---------------------------------------------------------------------------
# Host-side constants that must be computed the array path's way
# ---------------------------------------------------------------------------

def m_class(m: int) -> int:
    """``M_CLASS`` for one azimuthal order: 0 at m = 0, 1 at |m| = 1, 2 above.

    m = 0 is an arm of THIS family since 2026-09-04 (complex64 storage only; the
    real-storage m = 0 run is still ``cylindrical_triton``'s, and the two are
    split by ``force_complex_fields`` rather than by m).
    """
    if int(m) == 0:
        return 0
    return 1 if abs(int(m)) == 1 else 2


def four_dtdx_scalar(dtdx: float) -> float:
    """The m = 0 on-axis ``Dz`` scalar ``4.0 * (dt/dx)``, rounded as the array
    path rounds it.

    ``stepping._cylindrical_axis_zero_D`` (:585) forms ``(4.0 * (grid.dt /
    grid.dx))`` as a PYTHON FLOAT and multiplies a complex64 row by it, which
    NEP-50 casts WEAKLY to complex64 — so the multiplicand is the float32
    rounding of the float64 product, and the product itself is the
    coefficient-LEFT zero-imaginary form (:func:`complex_fields._mul_coefficient_left`).
    Rounded here on the host, as ``cylindrical_triton``'s ``four_dtdx`` argument
    is, rather than recomputed in-kernel from an fp32 ``dtdx``.
    """
    import numpy  # noqa: PLC0415

    return float(numpy.float32(4.0 * float(dtdx)))


def zero_rows(m: int, accurate_fields_near_cylorigin: bool) -> int:
    """How many near-axis rows the |m| >= 2 rule holds at zero.

    ``stepping._cylindrical_axis_rows`` (:601-622): ``slice(0, 1)`` on the
    ACCURATE branch (an ordinary axis boundary condition and nothing else, stable
    only below Courant ~1/(|m| + 0.5), which ``Grid`` refuses above) and
    ``slice(0, |m|)`` on MEEP's default stability hack. Zero at |m| = 1, where
    neither branch applies.
    """
    if abs(int(m)) < 2:
        return 0
    return 1 if bool(accurate_fields_near_cylorigin) else abs(int(m))


def imr_coefficient_row(xp: Any, target: str, sign: float, m: int, dtdx: float,
                        rows: int, dtype: Any) -> Any:
    """The per-r i*m/r coefficient, built the array path's way and ONLY that way.

    ``stepping._cylindrical_imr_term._build`` (:713-718), transcribed character
    for character. Every step of it is load-bearing:

    * the ``arange`` is FLOAT64 and the doubled coordinate is ``2*ir + iyee_r``
      with ``iyee_r`` the target's OWN radial Yee shift (``fields.IYEE_SHIFTS``);
      using 0 for every target is a measured catch (``imr_iyee_zero``: 439,124
      words on the gate's reference battery, 2026-08-13);
    * the clamp is ``maximum(.., 1.0)``, which bites only on row 0 of a shift-0
      target, whose curl the ownership mask zeroes;
    * the numerator is spelled ``(-1j) * (sign * 2.0 * m * dtdx)`` — multiplied by
      MINUS the imaginary unit, which launders the real word to +0.0 for BOTH
      signs of the real factor X. The SIGN of the unit is what does that, not the
      operand order: measured at X = +-0.37 and +-0.74, all four in float32 words,

          (-1j)*X   ->  (+0.0, -X)      X*(-1j)  ->  (+0.0, -X)   [identical]
          (1j)*X    ->  (-0.0, +X) for X < 0     X*(1j)   ->  same [identical]
          -(1j*X)   ->  (-0.0, -X) for X > 0

      so ``(-1j) * X`` and ``X * (-1j)`` are the SAME BITS and swapping the order
      is a predicted null, while flipping the unit's sign or moving the negation
      outside the product changes the real word. An earlier draft of this comment
      attributed the laundering to the unit being on the LEFT; that is confounded
      — the mirror it named changes the sign AND the order at once — and the
      measurement above is what corrects it;
    * the DIVISION is float64 on the host and the result is rounded ONCE with
      ``.astype(dtype)``. In-kernel this would need ``tl.math.div_rn`` and would
      still be a different number, which is why the row is bound as an array.
    """
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    iyee_r = IYEE_SHIFTS[target][0]
    r_doubled = 2 * xp.arange(rows, dtype=xp.float64) + iyee_r
    divisor = xp.maximum(r_doubled, 1.0).reshape(-1, 1, 1)
    return (((-1j) * (sign * 2.0 * int(m) * float(dtdx))) / divisor).astype(dtype)


def axis_increment_scalars(m: int, dtdx: float):
    """The two |m| = 1 B-side host scalars, rounded as NEP-50 rounds them.

    ``stepping._cylindrical_axis_increment_B`` (:643-644) writes

        (-dtdx) * (ep[0] - ep_above[0]) - 1j * (m * dtdx) * ez_off_axis

    where both scalars meet a complex64 array as WEAK Python scalars, so each is
    converted to complex64 once and the multiply is a FULL complex product with
    the zero cross terms (``np.multiply`` carries only ``FF->F`` complex loops —
    complex_fields.py:51-58).

    Returns ``(minus_dtdx, (re, im))``. The first is a PYTHON FLOAT and stays one
    — that is the ``python_float_left`` probe pattern (S:1635's orientation), and
    :func:`complex_fields._mul_coefficient_left` is the certified helper for it.
    The second is ``1j * (m * dtdx)``, whose REAL WORD IS A SIGNED ZERO: multiplied
    by PLUS the imaginary unit, CPython computes ``re = 0.0*X - 1.0*0.0``, which is
    -0.0 for X < 0 (measured: m = -1 at dtdx = 0.37 gives ``(-0.0, -0.37)``, real
    word ``0x80000000``). That is the OPPOSITE laundering from the i*m/r row's
    ``(-1j) * X``, which yields +0.0 for both signs.

    WHAT MAKES THEM DIFFER IS THE SIGN OF THE UNIT, NOT THE OPERAND ORDER —
    ``1j * X`` and ``X * 1j`` are bit-identical, as are ``(-1j) * X`` and
    ``X * (-1j)`` (measured; see :func:`imr_coefficient_row` for the four-way
    table). So the two spellings in ``stepping`` genuinely produce different bits
    and both are host-rounded and passed through here, never synthesized
    in-kernel — but a reader must not conclude that reordering either product is
    unsafe, and an earlier draft of this comment implied exactly that.
    """
    import numpy  # noqa: PLC0415

    second = numpy.complex64(1j * (int(m) * float(dtdx)))
    return (float(numpy.float32(-float(dtdx))),
            (float(numpy.float32(second.real)), float(numpy.float32(second.imag))))


# ---------------------------------------------------------------------------
# The prefix — the part that stays on the array path
# ---------------------------------------------------------------------------

def cylindrical_complex_prefix(xp: Any, sub_step: str, sources: Dict[str, Any],
                               scratch: Any = None) -> Any:
    """The radial prefix the kernel consumes, from the SHIPPED array-path scan.

    ``stepping.cylindrical_rderiv_prefix`` is CALLED, never re-derived: its
    float32 summation order defines the answer and ``cupy.cumsum``'s order is the
    oracle (cylindrical_triton.py:32-56). The same ``StepScratch`` is threaded
    through so the two cached invariant row vectors are shared with the array
    path rather than rebuilt. Complex storage changes nothing here — the scan
    reads ``f_p.real.dtype`` for its weights (stepping.py:1308) and is otherwise
    dtype-agnostic.

    B side: Ep = Ey EXTENDED BY ONE ZERO WALL ROW (stepping.py:342-361,
    transcribed including the pooled-versus-fresh allocation branch, because the
    two writes are what ``xp.concatenate`` used to do). D side: Hp = Hy,
    unextended — the backward difference reads rows ``i`` and ``i-1``.
    """
    from ..stepping import _face, _span, cylindrical_rderiv_prefix  # noqa: PLC0415

    spec = PREFIX[sub_step]
    source = sources[spec["component"]]
    if not spec["extend_wall_row"]:
        return cylindrical_rderiv_prefix(xp, source, spec["ir0"], scratch=scratch)

    rows = source.shape[0]
    extended_shape = (rows + 1,) + source.shape[1:]
    extended = (xp.empty(extended_shape, dtype=source.dtype) if scratch is None
                else scratch.take("cyl_extended", extended_shape, source.dtype))
    extended[_span(0, 0, rows)] = source
    extended[_face(0, rows)] = 0
    return cylindrical_rderiv_prefix(xp, extended, spec["ir0"], scratch=scratch)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

@triton.jit
def cyl_complex_pml_curl_step(
    f0, f1, f2,                   # targets:     Bx,By,Bz or Dx,Dy,Dz (c8 as words)
    u0, u1, u2,                   # auxiliaries: fu_B*    or fu_D*    (c8 as words)
    g0, g1, g2,                   # sources:     Ex,Ey,Ez or Hx,Hy,Hz (c8 as words)
    pfx,                          # the ARRAY-PATH prefix (c8 as words)
    c0, c2,                       # i*m/r rows for targets 0 and 2 (c8 as words, nr long)
    kmx, sinvx, kmy, sinvy, kmz, sinvz,   # kms/sinv per axis, f32, ONE sub-lattice
    nx, ny, nz, n_elem, dtdx,     # n_elem = COMPLEX cells; dtdx pre-rounded to f32
    minus_dtdx,                   # |m|=1 B: python float -dtdx; unused on D
    inc_b_re, inc_b_im,           # |m|=1 B: complex64(1j*m*dtdx), SIGNED-ZERO real word
    four_dtdx,                    # m=0 D: f32(4*dtdx) for the on-axis Dz add; unused on B
    BACKWARD: tl.constexpr,       # 0 = B (forward differences), 1 = D
    BCZ: tl.constexpr,            # PERIODIC or METALLIC; r is METALLIC, phi PERIODIC
    M_CLASS: tl.constexpr,        # M_ZERO (m = 0), M_ONE (|m| = 1) or M_MANY (|m| >= 2)
    ZERO_ROWS: tl.constexpr,      # |m| or 1 under M_MANY; unused under M_ZERO/M_ONE
    EXPANSION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One COMPLEX cylindrical curl sub-step at |m| >= 1 — the certified complex
    body (``complex_fields.bloch_pml_curl_step``, job 2330) with the four
    cylindrical additions of the module docstring and NOTHING else.

    Differences from the certified body, and only these:

    * ``BCX`` is compiled METALLIC (the r-axis identity: the CYL_AXIS far ghost
      IS the metallic zero, and the near ghost is unobservable) and ``BCY``
      PERIODIC; ``PH*`` are gone entirely (this family refuses a Bloch phase, and
      ``Grid`` refuses one on r or phi outright);
    * target 2's curl is the prefix construction rather than the four-operand
      grouping (B) or takes the prefix as its ``first`` source (D);
    * the i*m/r term on targets 0 and 2, from a bound per-r coefficient row and
      the CENTER registers the body already loaded — compiled OUT under
      ``M_ZERO``, where the array path never forms it (stepping :348/:430);
    * the |m| = 1 axis-row replacement, the |m| >= 2 near-axis zeroing, and the
      m = 0 axis pair (``Bx[0] = 0``; ``Dz[0] += four_dtdx*Hp[0]``, ``Dy[0] = 0``).

    ``pfx`` is (nr+1, ny, nz) on the B side and (nr, ny, nz) on the D side; row
    stride is ``2*ny*nz`` WORDS in both. ``c0``/``c2`` are nr-long complex64 rows
    indexed by ``i`` — word offsets ``2*i`` and ``2*i + 1``.
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = idx < n_elem
    nyz = ny * nz
    k = idx % nz
    plane = idx // nz
    j = plane % ny
    i = plane // ny

    # --- the ghost rule, per axis (stepping._shift_up / _shift_down) -----------
    # r is CYL_AXIS: the far face is the metallic zero (S:1781-1783) and the near
    # ghost (S:1830-1842) is DELIBERATELY NOT IMPLEMENTED — unobservable, see the
    # module docstring. phi is PERIODIC on a length-1 axis: the wrap returns the
    # SAME element, which is what makes the difference an exact +0.0. Computed,
    # never elided (grouping choice 8).
    if BACKWARD:
        si, sj, sk = i - 1, j - 1, k - 1
    else:
        si, sj, sk = i + 1, j + 1, k + 1
    vr = live & (si >= 0) & (si < nx)
    sj = tl.where(sj < 0, ny - 1, tl.where(sj == ny, 0, sj))
    vp = live
    if BCZ == METALLIC:
        vz = live & (sk >= 0) & (sk < nz)
    else:
        sk = tl.where(sk < 0, nz - 1, tl.where(sk == nz, 0, sk))
        vz = live

    o_r = si * nyz + j * nz + k
    o_p = i * nyz + sj * nz + k
    o_z = i * nyz + j * nz + sk

    # --- loads: two words per operand -----------------------------------------
    a_re = tl.load(g0 + 2 * idx, mask=live, other=0.0)
    a_im = tl.load(g0 + 2 * idx + 1, mask=live, other=0.0)
    b_re = tl.load(g1 + 2 * idx, mask=live, other=0.0)
    b_im = tl.load(g1 + 2 * idx + 1, mask=live, other=0.0)
    c_re = tl.load(g2 + 2 * idx, mask=live, other=0.0)
    c_im = tl.load(g2 + 2 * idx + 1, mask=live, other=0.0)
    a_p_re = tl.load(g0 + 2 * o_p, mask=vp, other=0.0)
    a_p_im = tl.load(g0 + 2 * o_p + 1, mask=vp, other=0.0)
    a_z_re = tl.load(g0 + 2 * o_z, mask=vz, other=0.0)
    a_z_im = tl.load(g0 + 2 * o_z + 1, mask=vz, other=0.0)
    b_r_re = tl.load(g1 + 2 * o_r, mask=vr, other=0.0)
    b_r_im = tl.load(g1 + 2 * o_r + 1, mask=vr, other=0.0)
    b_z_re = tl.load(g1 + 2 * o_z, mask=vz, other=0.0)
    b_z_im = tl.load(g1 + 2 * o_z + 1, mask=vz, other=0.0)
    c_r_re = tl.load(g2 + 2 * o_r, mask=vr, other=0.0)
    c_r_im = tl.load(g2 + 2 * o_r + 1, mask=vr, other=0.0)
    c_p_re = tl.load(g2 + 2 * o_p, mask=vp, other=0.0)
    c_p_im = tl.load(g2 + 2 * o_p + 1, mask=vp, other=0.0)

    # --- the curl (stepping._curl_from_operands): DO NOT flatten these parens --
    t0_re = ((c_p_re - c_re) + (b_re - b_z_re))
    t0_im = ((c_p_im - c_im) + (b_im - b_z_im))
    t1_re = ((a_z_re - a_re) + (c_re - c_r_re))
    t1_im = ((a_z_im - a_im) + (c_im - c_r_im))
    t2_re = ((b_r_re - b_re) + (a_re - a_p_re))
    t2_im = ((b_r_im - b_im) + (a_im - a_p_im))
    curl0_re, curl0_im = _mul_coefficient_left(dtdx, t0_re, t0_im, EXPANSION)
    curl1_re, curl1_im = _mul_coefficient_left(dtdx, t1_re, t1_im, EXPANSION)
    curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)

    # --- the cylindrical substitution on target 2 ------------------------------
    if BACKWARD:
        # step_D :425-426 — Dz's `first` source is the prefix; the backward
        # machinery is otherwise unchanged. Its row-0 ghost is unreachable for the
        # same reason every other r near ghost is (Dz's r-shift is 0).
        p_here_re = tl.load(pfx + 2 * idx, mask=live, other=0.0)
        p_here_im = tl.load(pfx + 2 * idx + 1, mask=live, other=0.0)
        p_down_re = tl.load(pfx + 2 * o_r, mask=vr, other=0.0)
        p_down_im = tl.load(pfx + 2 * o_r + 1, mask=vr, other=0.0)
        t2_re = ((p_down_re - p_here_re) + (a_re - a_p_re))
        t2_im = ((p_down_im - p_here_im) + (a_im - a_p_im))
        curl2_re, curl2_im = _mul_coefficient_left(dtdx, t2_re, t2_im, EXPANSION)
    else:
        # step_B :343-347 — Bz's WHOLE curl is the forward difference of the
        # EXTENDED prefix: one subtract, one multiply. The four-operand grouping
        # above is a different float32 number per plane.
        pu_re = tl.load(pfx + 2 * (idx + nyz), mask=live, other=0.0)
        pu_im = tl.load(pfx + 2 * (idx + nyz) + 1, mask=live, other=0.0)
        pd_re = tl.load(pfx + 2 * idx, mask=live, other=0.0)
        pd_im = tl.load(pfx + 2 * idx + 1, mask=live, other=0.0)
        curl2_re, curl2_im = _mul_coefficient_left(dtdx, pu_re - pd_re, pu_im - pd_im,
                                                   EXPANSION)

    # --- the i*m/r coupling (stepping :674-724, sites :348-355 / :430-437) -----
    # AFTER the dtdx curl, BEFORE the ownership mask — the array path's order.
    # The partners are the CENTER registers already loaded: target 0 takes `c`
    # (g2 = Ez/Hz), target 2 takes `a` (g0 = Ex/Hx). The coefficient row is
    # host-built and bound; its real word is +0.0 and MUST survive as a literal
    # cross-term operand (a plane-wise shortcut is byte-wrong on signed zeros).
    # `curl - (c*g)` carries the array path's `curl + (-(c*g))` by IEEE-754.
    # COMPILED OUT at m = 0, exactly as the array path never forms the term there
    # (`if cylindrical and grid.m != 0`). Multiplying by a zero row instead is NOT
    # a no-op at the byte level: `curl - (0*g)` moves a -0.0 curl word to +0.0
    # whenever the zero product's sign lands negative, and a non-finite partner
    # word would propagate into a curl the array path leaves untouched.
    if M_CLASS != M_ZERO:
        q0_re = tl.load(c0 + 2 * i, mask=live, other=0.0)
        q0_im = tl.load(c0 + 2 * i + 1, mask=live, other=0.0)
        q2_re = tl.load(c2 + 2 * i, mask=live, other=0.0)
        q2_im = tl.load(c2 + 2 * i + 1, mask=live, other=0.0)
        m0_re, m0_im = _mul_general_coefficient_left(q0_re, q0_im, c_re, c_im, EXPANSION)
        curl0_re = curl0_re - m0_re
        curl0_im = curl0_im - m0_im
        m2_re, m2_im = _mul_general_coefficient_left(q2_re, q2_im, a_re, a_im, EXPANSION)
        curl2_re = curl2_re - m2_re
        curl2_im = curl2_im - m2_im

    # --- ownership mask (stepping._mask_non_owned_cells, is_axis + metallic) ---
    # Per axis, for every target whose Yee shift there is 0 (fields.IYEE_SHIFTS):
    #   B: Bx(0,1,1) -> r ; By(1,0,1) -> phi only, periodic: nothing ; Bz(1,1,0) -> z
    #   D: Dx(1,0,0) -> phi (nothing) + z ; Dy(0,1,0) -> r + z ; Dz(0,0,1) -> r
    # The z clauses fire only when z is METALLIC; a periodic z masks nothing.
    at_r, at_z = i == 0, k == 0
    if BACKWARD:
        if BCZ == METALLIC:
            curl0_re = tl.where(at_z, 0.0, curl0_re)
            curl0_im = tl.where(at_z, 0.0, curl0_im)
            curl1_re = tl.where(at_z, 0.0, curl1_re)
            curl1_im = tl.where(at_z, 0.0, curl1_im)
        curl1_re = tl.where(at_r, 0.0, curl1_re)
        curl1_im = tl.where(at_r, 0.0, curl1_im)
        curl2_re = tl.where(at_r, 0.0, curl2_re)
        curl2_im = tl.where(at_r, 0.0, curl2_im)
    else:
        curl0_re = tl.where(at_r, 0.0, curl0_re)
        curl0_im = tl.where(at_r, 0.0, curl0_im)
        if BCZ == METALLIC:
            curl2_re = tl.where(at_z, 0.0, curl2_re)
            curl2_im = tl.where(at_z, 0.0, curl2_im)

    # --- the |m| = 1 axis-row increment, AFTER the mask, REPLACING the row -----
    # stepping :370-372 (B) / :451-453 (D). The row is exactly +0.0 here, and
    # `+0.0 + x == x` for every x EXCEPT -0.0 — so REPLACE, never accumulate
    # (grouping choice 4; measured needle-missed on random seeds and caught under
    # zero-init). The negation is `* -1.0`, never unary minus.
    if M_CLASS == M_ONE:
        if BACKWARD:
            # dtdx * (Hr[i] - Hr[i, z-1] - 2.0*Hz[i]) at r = 0, on the whole row.
            # Hr = g0 center `a`, its z-down neighbour, and Hz = g2 center `c`.
            two_c_re, two_c_im = _mul_coefficient_left(2.0, c_re, c_im, EXPANSION)
            s_re = (a_re - a_z_re) - two_c_re
            s_im = (a_im - a_z_im) - two_c_im
            inc_re, inc_im = _mul_coefficient_left(dtdx, s_re, s_im, EXPANSION)
            curl1_re = tl.where(at_r, inc_re * -1.0, curl1_re)
            curl1_im = tl.where(at_r, inc_im * -1.0, curl1_im)
        else:
            # (-dtdx)*(Ep[i] - Ep[i, z+1]) - (1j*m*dtdx)*Ez[r+1] at r = 0.
            # Ep = g1 center `b` and its z-up neighbour; Ez[r+1] is the FIRST
            # OFF-AXIS row of g2, read at the row-1 offset of this lane's column.
            off = nyz + j * nz + k
            e1_re = tl.load(g2 + 2 * off, mask=live, other=0.0)
            e1_im = tl.load(g2 + 2 * off + 1, mask=live, other=0.0)
            d_re = b_re - b_z_re
            d_im = b_im - b_z_im
            # (-dtdx) is a PYTHON FLOAT at the call site (S:643), i.e. the
            # certified zero-imaginary coefficient-LEFT product; the 1j*m*dtdx
            # factor is a general complex scalar whose real word is a signed zero.
            p_re, p_im = _mul_coefficient_left(minus_dtdx, d_re, d_im, EXPANSION)
            q_re, q_im = _mul_general_coefficient_left(inc_b_re, inc_b_im,
                                                       e1_re, e1_im, EXPANSION)
            inc_re = p_re - q_re
            inc_im = p_im - q_im
            curl0_re = tl.where(at_r, inc_re * -1.0, curl0_re)
            curl0_im = tl.where(at_r, inc_im * -1.0, curl0_im)

    # --- split-field recurrence (stepping._apply_pml_update) -------------------
    # dsig/dsigu cycle (vec.hpp cycle_direction): target 0 -> (y, z), 1 -> (z, x),
    # 2 -> (x, y), the same triple on both sides. Every multiply is the
    # zero-imaginary product with the FIELD on the left (S:1929-1935).
    km_x = tl.load(kmx + i, mask=live, other=0.0)
    si_x = tl.load(sinvx + i, mask=live, other=0.0)
    km_y = tl.load(kmy + j, mask=live, other=0.0)
    si_y = tl.load(sinvy + j, mask=live, other=0.0)
    km_z = tl.load(kmz + k, mask=live, other=0.0)
    si_z = tl.load(sinvz + k, mask=live, other=0.0)

    p0_re = tl.load(u0 + 2 * idx, mask=live, other=0.0)
    p0_im = tl.load(u0 + 2 * idx + 1, mask=live, other=0.0)
    x_re, x_im = _mul_field_left(p0_re, p0_im, km_y, EXPANSION)
    x_re = x_re - curl0_re
    x_im = x_im - curl0_im
    n0_re, n0_im = _mul_field_left(x_re, x_im, si_y, EXPANSION)
    e_re = tl.load(f0 + 2 * idx, mask=live, other=0.0)
    e_im = tl.load(f0 + 2 * idx + 1, mask=live, other=0.0)
    r_re, r_im = _mul_field_left(e_re, e_im, km_z, EXPANSION)
    r_re = (r_re + n0_re) - p0_re
    r_im = (r_im + n0_im) - p0_im
    v0_re, v0_im = _mul_field_left(r_re, r_im, si_z, EXPANSION)

    p1_re = tl.load(u1 + 2 * idx, mask=live, other=0.0)
    p1_im = tl.load(u1 + 2 * idx + 1, mask=live, other=0.0)
    x_re, x_im = _mul_field_left(p1_re, p1_im, km_z, EXPANSION)
    x_re = x_re - curl1_re
    x_im = x_im - curl1_im
    n1_re, n1_im = _mul_field_left(x_re, x_im, si_z, EXPANSION)
    e_re = tl.load(f1 + 2 * idx, mask=live, other=0.0)
    e_im = tl.load(f1 + 2 * idx + 1, mask=live, other=0.0)
    r_re, r_im = _mul_field_left(e_re, e_im, km_x, EXPANSION)
    r_re = (r_re + n1_re) - p1_re
    r_im = (r_im + n1_im) - p1_im
    v1_re, v1_im = _mul_field_left(r_re, r_im, si_x, EXPANSION)

    p2_re = tl.load(u2 + 2 * idx, mask=live, other=0.0)
    p2_im = tl.load(u2 + 2 * idx + 1, mask=live, other=0.0)
    x_re, x_im = _mul_field_left(p2_re, p2_im, km_x, EXPANSION)
    x_re = x_re - curl2_re
    x_im = x_im - curl2_im
    n2_re, n2_im = _mul_field_left(x_re, x_im, si_x, EXPANSION)
    e_re = tl.load(f2 + 2 * idx, mask=live, other=0.0)
    e_im = tl.load(f2 + 2 * idx + 1, mask=live, other=0.0)
    r_re, r_im = _mul_field_left(e_re, e_im, km_y, EXPANSION)
    r_re = (r_re + n2_re) - p2_re
    r_im = (r_im + n2_im) - p2_im
    v2_re, v2_im = _mul_field_left(r_re, r_im, si_y, EXPANSION)

    # --- the per-|m| axis rules, folded into the STORED value ------------------
    # m = 0 (stepping :581-585 D / :661-662 B), the rules cylindrical_triton
    #   carries for real storage, applied here to complex word pairs: the B side
    #   zeroes Bx on the axis row; the D side POST-adds (4*Courant)*Hp[0] to the
    #   UPDATED Dz on the axis row — Hp is the g1 CENTER already loaded (`b`),
    #   the same stored Hy the array path re-reads through get_H under an active
    #   PML (fields.py:1181) — and then zeroes Dy there. The add is the
    #   coefficient-LEFT zero-imaginary product the array path's NEP-50 cast
    #   makes of a Python float times a complex64 row; the fold-into-curl
    #   alternative is the one :524-543 measured broken at m = 0 under PML.
    # |m| = 1 (stepping :588-589): the D side zeroes Dz on the axis row — the
    #   FIELD ONLY, never fu_Dz — and the B side does NOTHING (:661-663 branches
    #   on m == 0 and abs(m) > 1).
    # |m| >= 2 (:565-567, :663-671): all three components AND their fu on every
    #   row within ZERO_ROWS of the axis. The component set is POST-#3164.
    if M_CLASS == M_ZERO:
        if BACKWARD:
            hp_re, hp_im = _mul_coefficient_left(four_dtdx, b_re, b_im, EXPANSION)
            v2_re = tl.where(at_r, v2_re + hp_re, v2_re)
            v2_im = tl.where(at_r, v2_im + hp_im, v2_im)
            v1_re = tl.where(at_r, 0.0, v1_re)
            v1_im = tl.where(at_r, 0.0, v1_im)
        else:
            v0_re = tl.where(at_r, 0.0, v0_re)
            v0_im = tl.where(at_r, 0.0, v0_im)
    if M_CLASS == M_ONE:
        if BACKWARD:
            v2_re = tl.where(at_r, 0.0, v2_re)
            v2_im = tl.where(at_r, 0.0, v2_im)
    if M_CLASS == M_MANY:
        near = i < ZERO_ROWS
        v0_re = tl.where(near, 0.0, v0_re)
        v0_im = tl.where(near, 0.0, v0_im)
        v1_re = tl.where(near, 0.0, v1_re)
        v1_im = tl.where(near, 0.0, v1_im)
        v2_re = tl.where(near, 0.0, v2_re)
        v2_im = tl.where(near, 0.0, v2_im)
        n0_re = tl.where(near, 0.0, n0_re)
        n0_im = tl.where(near, 0.0, n0_im)
        n1_re = tl.where(near, 0.0, n1_re)
        n1_im = tl.where(near, 0.0, n1_im)
        n2_re = tl.where(near, 0.0, n2_re)
        n2_im = tl.where(near, 0.0, n2_im)

    # --- stores: u then f (kernels.py:193-198 order), both planes --------------
    tl.store(u0 + 2 * idx, n0_re, mask=live)
    tl.store(u0 + 2 * idx + 1, n0_im, mask=live)
    tl.store(u1 + 2 * idx, n1_re, mask=live)
    tl.store(u1 + 2 * idx + 1, n1_im, mask=live)
    tl.store(u2 + 2 * idx, n2_re, mask=live)
    tl.store(u2 + 2 * idx + 1, n2_im, mask=live)
    tl.store(f0 + 2 * idx, v0_re, mask=live)
    tl.store(f0 + 2 * idx + 1, v0_im, mask=live)
    tl.store(f1 + 2 * idx, v1_re, mask=live)
    tl.store(f1 + 2 * idx + 1, v1_im, mask=live)
    tl.store(f2 + 2 * idx, v2_re, mask=live)
    tl.store(f2 + 2 * idx + 1, v2_im, mask=live)


@triton.jit
def _mul_general_coefficient_left(c_re, c_im, z_re, z_im, EXPANSION: tl.constexpr):
    """(c_re + i*c_im) * z with the COEFFICIENT on the LEFT — the S:717/:724
    orientation ``factor * partner_values`` and the S:644 orientation
    ``scalar * ez_off_axis``.

    Structurally :func:`special_kz._mul_imag_coefficient_left`; restated here
    rather than imported because that module owns a different tranche's contract
    and this one's coefficient is a per-r ROW, not a scalar.

    THE ARM CHOICE IS DEGENERATE AT EVERY ONE OF THIS KERNEL'S CALL SITES, and
    that is measured rather than assumed. All three pass a coefficient whose REAL
    word is a zero: the two i*m/r rows carry exactly +0.0 (the ``(-1j) * X``
    spelling launders the sign — module docstring), and the |m| = 1 B-side scalar
    ``1j*(m*dtdx)`` carries -0.0 for m < 0. With ``c_re = ±0.0`` the product
    ``c_re * z_re`` is exact, so ``fma(c_re, z_re, (c_im*z_im) * -1.0)`` and
    ``(c_re*z_re) - (c_im*z_im)`` are the same single-rounding operation.

    BOTH ARMS ARE WRITTEN ANYWAY because ``EXPANSION`` is a MEASURED platform
    constexpr bound once for the whole family from
    :data:`complex_fields.PROBE_PATTERNS`, and a helper that honoured only one arm
    would silently stop honouring the contract the certified base is bound under
    the moment a call site with a nonzero real coefficient is added. An earlier
    revision justified the second arm by claiming the |m| = 1 B-side ``-dtdx``
    scalar comes through HERE and fuses; it does not — that call site goes through
    :func:`complex_fields._mul_coefficient_left` (:761), and this helper's three
    call sites (:704, :708, :764) all carry a ±0.0 real word.

    THE ZERO CROSS TERMS ARE KEPT because they are what the array path computes.
    With ``c_re = +0.0`` the real output is ``(+0.0*z_re) - (c_im*z_im)``, which
    is +0.0 where a plane-wise ``-(c_im*z_im)`` gives -0.0 — a REAL difference in
    the product's own words, but one that is BYTE-INVISIBLE through this kernel's
    fold, because ``curl - m`` can carry it only at a -0.0 minuend and no i*m/r
    minuend here can be one (module docstring; measured 0 of 491,520 words by the
    gate's minuend census). :data:`MUTATION_IMR_PLANEWISE` is therefore a
    PREDICTED NULL recorded with that evidence, and the choice is held by a
    source-text assertion in the laptop test file.

    NEGATION IS ``* -1.0``, NEVER unary ``-``: Triton lowers ``-x`` as
    ``0.0 - x`` (triton 3.1.0, language/semantic.py:386-391) and canonicalizes
    every +-0 addend to +0, which was measured REACHABLE at driver level by jobs
    2343/2345.
    """
    if EXPANSION == 1:  # FMA_V1
        out_re = tl.math.fma(c_re, z_re, (c_im * z_im) * -1.0)
        out_im = tl.math.fma(c_re, z_im, c_im * z_re)
    else:
        out_re = (c_re * z_re) - (c_im * z_im)
        out_im = (c_re * z_im) + (c_im * z_re)
    return out_re, out_im


# ---------------------------------------------------------------------------
# Coverage — the POSITIVE refusal enumeration
# ---------------------------------------------------------------------------
#
# WRITTEN POSITIVELY AND ENUMERATING. Every requirement is named and checked and
# coverage is never inferred from the absence of a known blocker — the two worst
# defects in this project were silent wrong answers, not crashes.
#
# These predicates do NOT call ``coverage._grid_reasons`` (its clauses 2, 6 and 7
# refuse this whole domain) and do NOT call
# ``complex_fields._complex_grid_reasons`` (its clause 4 refuses cylindrical
# outright). Both are RIGHT for their own products and must stay; their
# non-cylindrical clauses are restated below, individually, with the cylindrical
# ones replaced by positive Dcyl requirements. That duplication is deliberate and
# it is the reason neither shared clause may be widened to admit Dcyl: each is
# now load-bearing in two directions at once.

#: The shared helpers these predicates DO reuse, named as data so a laptop test
#: can assert every one still exists and a rename in a shared file fails at the
#: merge bar instead of silently dropping a clause.
SHARED_CLAUSES: Tuple[str, ...] = (
    "_boundary_kinds", "_call", "_coefficient_reasons", "_inverse_epsilon_reasons",
    "_susceptibility_reasons", "_complex_layout_reasons", "_expansion_reasons")


def _shared_reasons(fields: Any, pml: Any, grid: Any, probe: Any) -> List[str]:
    """The clauses both predicates share, cylindrical geometry excluded."""
    reasons: List[str] = []

    # 1. CuPy backend. The kernel launches against device pointers and — see the
    #    module docstring — the prefix's summation order is CuPy's, not NumPy's.
    xp = getattr(grid, "xp", None)
    if getattr(xp, "__name__", "") != "cupy":
        reasons.append(f"array module is {getattr(xp, '__name__', xp)!r}, not cupy")

    # 2 (INVERTED against coverage.py clause 2). Complex64 storage is REQUIRED.
    #    At |m| >= 1 the array path REFUSES real storage outright
    #    (stepping.py:730-736, driver.py:971-977, MEEP's change_m aborts on the
    #    same combination); at m = 0 both storages exist and the DECLARATION is
    #    what splits the two cylindrical products — a real m = 0 run is
    #    cylindrical_triton's (it refuses force_complex_fields=True by name), a
    #    complex one is this family's M_ZERO arm. The storage dtype itself is
    #    verified by the layout clause; this clause checks the declaration.
    if not getattr(fields, "force_complex_fields", False):
        reasons.append(
            "force_complex_fields is not set: this family steps complex64 word "
            "pairs only. |m| >= 1 has no real-storage form (stepping.py:730-736 "
            "raises, MEEP's change_m aborts), and a REAL m = 0 Dcyl run is "
            "cylindrical_triton's kernel, not this one")

    # 3. An absorber that actually absorbs: this is the split-field product only.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the complex "
                       "cylindrical split-field path only)")

    # 4. No conductivity anywhere on the curl targets. A missing or non-callable
    #    reader is refused OUTRIGHT — inferring "no conductivity" from the
    #    ABSENCE of condfac_for is admission by attribute absence, the exact
    #    reasoning coverage exists to refuse (complex_fields.py:875-884).
    reader = getattr(fields, "condfac_for", None)
    if not callable(reader):
        reasons.append("fields does not expose condfac_for; an unreadable "
                       "conductivity table is not an absent one")
    else:
        for target in CURL_TARGETS:
            try:
                conductive = reader(target) is not None
            except Exception as exc:  # noqa: BLE001 - unreadable means not covered
                reasons.append(f"condfac_for({target!r}) raised {exc!r}")
                continue
            if conductive:
                reasons.append(
                    f"a conductivity is installed on {target}; conductive complex "
                    f"cylindrical stepping is a separate, unbuilt product")
    if getattr(fields, "has_magnetic_conductivity", False):
        reasons.append("a magnetic (B) conductivity is installed")

    # 5. No dispersion, refused OUTRIGHT: none of the sixteen demand rows carries
    #    a susceptibility, and complex ADE belongs to a future tranche. The shape
    #    clauses still run so an unreadable susceptibility is named, not shrugged.
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append(
            "a susceptibility is registered: complex-storage ADE is a future "
            "tranche, and no cylindrical corpus row combines the two")
    reasons.extend(_susceptibility_reasons(fields))

    # 6. No instantaneous nonlinearity, no BFAST, no special_kz beta. Grid already
    #    refuses the last two on a cylindrical cell (grid.py:689-696 beta,
    #    :734-741 BFAST) — the clauses are written because "another module already
    #    guards it" is exactly the reasoning this file exists to refuse.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero")

    # 7. NO BLOCH PHASE. Dcyl Bloch is z-only in MEEP (grid.py:654-658 refuses a k
    #    on r or phi), none of the sixteen rows carries one, and the kernel
    #    compiles no rotation at all. Admitting a z phase later is a predicate
    #    widening plus a gate row — not a silent extension.
    if getattr(grid, "has_bloch", False):
        reasons.append(
            f"nonzero k_point {getattr(grid, 'k_point', None)!r}: this kernel "
            f"compiles no Bloch rotation (a z-only Dcyl phase is a future widening)")
    k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {k_point!r} is not exactly zero")

    # 8. Stored E — the invariant behind differencing E while a pole is live, and
    #    the one an edit that ever makes stores_E optional under PML must be
    #    caught by (coverage.py:296-304's argument).
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 9. EXPANSION binding requires a measured platform probe artifact. The BASE
    #    four patterns, unextended: the i*m/r product's arm is degenerate (module
    #    docstring) and the |m| = 1 real scalar is the covered
    #    ``f4_mul_c8_coefficient_left`` orientation.
    reasons.extend(_expansion_reasons(probe))

    return reasons


def _cylindrical_geometry_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The POSITIVE Dcyl + |m| >= 1 requirements."""
    reasons: List[str] = []

    if not getattr(grid, "cylindrical", False):
        reasons.append("grid is not cylindrical (this product steps Dcyl grids only)")

    # EVERY INTEGER m, since 2026-09-04. m = 0 used to be refused here by name as
    # cylindrical_triton's row; the two products are now split by STORAGE (clause
    # 2 of _shared_reasons requires force_complex_fields, cylindrical_triton
    # refuses it), so a complex m = 0 run — corpus row
    # examples:dipole_in_vacuum_cyl_off_axis.py — is this family's M_ZERO arm and
    # a real m = 0 run is still the other kernel's. An unreadable m is refused: it
    # selects the arm.
    m = getattr(grid, "m", None)
    if m is None:
        reasons.append("grid does not report m")
    else:
        try:
            int(m)
        except Exception:  # noqa: BLE001 - an unreadable m is not coverage
            reasons.append(f"grid.m = {m!r} is not an integer")

    flags = tuple(bool(_call(grid, "is_axis", axis, default=False)) for axis in range(3))
    if flags != CYLINDRICAL_AXIS_FLAGS:
        reasons.append(f"grid.is_axis {flags!r} is not {CYLINDRICAL_AXIS_FLAGS!r} "
                       "(r must be the leading axis)")

    kinds = _boundary_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        if kinds[0] != REQUIRED_R_KIND:
            reasons.append(f"axis 0 boundary {kinds[0]!r} is not {REQUIRED_R_KIND!r}")
        if kinds[1] != REQUIRED_PHI_KIND:
            reasons.append(f"axis 1 boundary {kinds[1]!r} is not {REQUIRED_PHI_KIND!r}")
        if kinds[2] not in ADMITTED_Z_KINDS:
            reasons.append(f"axis 2 boundary {kinds[2]!r} is outside {ADMITTED_Z_KINDS!r}")

    # No fold anywhere. Grid already refuses a mirror on a Dcyl cell
    # (grid.py:648-653); the clause is written for the reason above.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
    else:
        if int(shape[1]) != 1:
            reasons.append(f"phi extent is {shape[1]} cells, not 1 (Dcyl is 2.5-D: "
                           "the exp(i*m*phi) dependence is analytic)")
        # AT LEAST ONE OFF-AXIS RADIAL ROW. Measured 2026-08-13 on a real
        # Grid(cell_size=(1.0, 0, 20), m=1, complex, z metallic) -> shape
        # (1, 1, 20): ``stepping.step_B`` RAISES ``IndexError: index 1 is out of
        # bounds for axis 0 with size 1`` from the |m| = 1 axis increment's
        # ``xp.take(Ez, 1, axis=0)`` (:642), while this predicate admitted the run
        # with no reasons at all. The kernel's matching load — ``off = nyz +
        # j*nz + k`` under ``mask=live`` (:753-755) — is true on every lane of
        # that grid, so it would read a full plane PAST THE END of Ez and store a
        # silently wrong axis row where the array path stops loudly.
        #
        # AND ON THE GPU BACKEND IT DOES NOT STOP LOUDLY AT ALL. Re-measured on
        # the GPU host 2026-08-13 (CuPy 13.5.1, RTX A6000): ``cupy.take`` DOES NOT
        # BOUNDS-CHECK — ``cp.take(a, 1, axis=0)`` on a size-1 axis returns row
        # 0's data instead of raising, so the SAME driver on the SAME grid steps
        # to completion with a finite state where NumPy refuses. The two
        # backends do not agree about whether the configuration is steppable,
        # which is a defect of the ARRAY PATH (``stepping.py``, not this file's
        # to edit) and is recorded here because it changes what this clause is
        # for: it is not a convenience that mirrors a loud array-path failure,
        # it is the only thing between a GPU run and a silently wrong axis row.
        # The evidence is the composition probe's ``nr>=2 refusal`` leg, which
        # measures the per-backend behaviour rather than asserting one of them.
        #
        # WHY THE CLAUSE IS UNCONDITIONAL rather than |m| = 1 only: at |m| >= 2
        # nr = 1 does not fault, but every row is inside ``ZERO_ROWS`` and the
        # whole volume is held at zero, so the configuration is degenerate in
        # both products; and a clause that has to re-derive the m class is a
        # clause that can disagree with :func:`m_class`.
        #
        # ``stepping._require_cylindrical_steppable`` (:513-522) is the hook that
        # would otherwise carry this and is a deliberate ``del grid`` no-op, so
        # the requirement is stated HERE — it may not be inferred from a helper
        # that checks nothing.
        if int(shape[0]) < MINIMUM_RADIAL_ROWS:
            reasons.append(
                f"radial extent is {shape[0]} cell(s), fewer than "
                f"{MINIMUM_RADIAL_ROWS}: the |m| = 1 axis increment reads the "
                f"FIRST OFF-AXIS row (stepping.py:671, where NumPy raises "
                f"IndexError and CuPy silently returns row 0 — both measured) "
                f"and the kernel's matching load would run past the end of the "
                f"source volume")

    # accurate_fields_near_cylorigin is READ, never assumed: it selects
    # ZERO_ROWS at |m| >= 2 (stepping._cylindrical_axis_rows, :601-622) and a
    # grid that cannot answer would compile the wrong constexpr — a plane of
    # wrong values, not a crash.
    if getattr(grid, "accurate_fields_near_cylorigin", None) is None:
        reasons.append("grid does not report accurate_fields_near_cylorigin; it "
                       "selects the |m| >= 2 near-axis row count and may not be assumed")

    # A configuration the array path itself refuses is not one a kernel may step.
    try:
        from ..stepping import _require_cylindrical_steppable  # noqa: PLC0415

        _require_cylindrical_steppable(grid)
    except Exception as exc:  # noqa: BLE001 - an unsteppable grid is refused
        reasons.append(f"stepping refuses this cylindrical grid: {exc!r}")

    return reasons


def cylindrical_complex_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                      probe: Any = None) -> Coverage:
    """May the complex cylindrical curl kernel step this (fields, pml, sub_step)?

    Every clause names its own requirement and appends its own reason; the scan
    continues after a failure so a refusal reports everything that disqualified
    the run rather than the first thing.

    An off-diagonal chi1inv row is ADMITTED here (constitutive-only, its whole
    effect is inside ``update_E``, stepping.py:1001-1008) and refused by the E-side
    constitutive predicate — the same per-sub-step split coverage.py makes.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(
            f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _shared_reasons(fields, pml, grid, probe)
    reasons.extend(_cylindrical_geometry_reasons(fields, pml, grid))

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + CURL_SOURCES)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        # Both sub-lattices: the suffix the plan binds is the sub-step's own, and
        # a swap is a silent half-cell error in the absorber profile.
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def cylindrical_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                              probe: Any = None) -> Coverage:
    """May the CERTIFIED ``complex_fields.bloch_constitutive_step`` run here?

    IT NEEDS NO NEW KERNEL, and that is a MEASURED finding rather than an
    assumption: ``stepping.update_H`` (:907-924) and ``stepping.update_E``
    (:926-993) carry no cylindrical branch at all, and the identity leg in the
    module docstring reports **480/480 rows, 0 differing uint32 words** against a
    plain complex elementwise ``dsigw`` reference on real complex Dcyl grids.
    (An earlier revision of this line said 640, an m set the harness does not
    carry; the count is computed from the leg's own axes by
    ``constitutive_identity_row_count`` and a laptop test pins the two together.)
    What refuses them today is ``complex_fields._complex_grid_reasons`` clause 4,
    the blanket cylindrical refusal — which is shared, is right for that product,
    and must stay.

    The enumeration is :func:`cylindrical_complex_curl_coverage`'s minus the
    curl-specific clauses (no ghost rule, no ownership mask, no prefix, no axis
    rules), plus the E side's own refusals: a registered polarization (already
    refused module-wide), an off-diagonal chi1inv row (the row product reads
    neighbours and the sub-step stops being element-wise), and ``stores_E``.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _shared_reasons(fields, pml, grid, probe)
    reasons.extend(_cylindrical_geometry_reasons(fields, pml, grid))

    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append(
            "an off-diagonal chi1inv row is installed (the row product reads "
            "neighbours; this sub-step is element-wise)")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(_complex_layout_reasons(fields, shape, names))

    if side == "E" and len(shape) == 3:
        # inv_eps stays FLOAT32 under complex storage (stepping.py:37-38,
        # fields.py:1203-1204), so coverage.py's float32 pin is exactly right.
        reasons.extend(_inverse_epsilon_reasons(fields, shape))

    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

class CylindricalComplexCurlPlan:
    """A launchable complex cylindrical curl sub-step: array-path prefix, ONE kernel.

    Built two ways and launched ONE way, exactly as ``launch.PmlCurlPlan`` is:
    :func:`plan_cylindrical_complex_curl` is the engine route (reads a real
    ``Fields``/``PML`` and passes through the predicate),
    :func:`plan_cylindrical_complex_curl_from_arrays` is the gate's route (bare
    device arrays, no predicate, ``kernel=`` for the mutation legs). Both produce
    this object and both go through :meth:`run`, so the bytes the gate certifies
    are the bytes the engine would launch.

    IT IS NOT ALLOCATION-FREE, and cannot be: the prefix is recomputed every
    launch from the current sources. With the engine's ``StepScratch`` threaded
    through it allocates nothing beyond what the array path already pools.

    The two i*m/r coefficient rows are GRID INVARIANTS — Yee shift, radial
    extent, m, Courant — so they are built ONCE at plan time, exactly as
    ``stepping``'s own ``scratch.constant`` cache builds them once per run
    (:713-723), and never rebuilt per launch.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bcz",
                 "m_class", "zero_rows", "expansion", "block", "num_warps",
                 "increment_scalars", "four_dtdx", "xp", "scratch", "_targets",
                 "_aux", "_sources", "_source_map", "_coefficients", "_imr_rows",
                 "_grid", "_kernel")

    def __init__(self, sub_step: str, shape, dtdx: float, m: int,
                 accurate_fields_near_cylorigin: bool, bcz: int, expansion: int,
                 block: int, targets, auxiliaries, sources, coefficients,
                 xp: Any, scratch: Any = None, kernel: Any = None,
                 num_warps: Optional[int] = None) -> None:
        if sub_step not in SUB_STEPS:
            raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
        spec = SUB_STEPS[sub_step]
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path multiplies a complex64 volume by a Python float,
        # which NEP-50 casts to complex64 before the multiply, and Triton types a
        # Python float argument as fp32 — so the two scalars are the same bits.
        self.dtdx = float(dtdx)
        self.backward = int(spec["backward"])
        self.bcz = int(bcz)
        self.m_class = m_class(m)
        self.zero_rows = zero_rows(m, accurate_fields_near_cylorigin)
        self.expansion = int(expansion)
        self.block = int(block)
        self.num_warps = None if num_warps is None else int(num_warps)
        self.xp = xp
        self.scratch = scratch
        self.increment_scalars = axis_increment_scalars(m, dtdx)
        # The m = 0 on-axis Dz scalar; read by the kernel under M_ZERO on the D
        # side only, bound on every launch so the signature is one signature.
        self.four_dtdx = four_dtdx_scalar(dtdx)
        self._targets = tuple(CupyPointer(_word_view(a)) for a in targets)
        self._aux = tuple(CupyPointer(_word_view(a)) for a in auxiliaries)
        self._sources = tuple(CupyPointer(_word_view(a)) for a in sources)
        # The raw arrays, by component name: the prefix is computed from them per
        # launch, so the plan holds the arrays and not only their addresses.
        self._source_map = {name: array
                            for name, array in zip(spec["sources"], sources)}
        self._coefficients = tuple(CupyPointer(_flat(a)) for a in coefficients)
        # At m = 0 the two rows are exactly zero and the kernel's M_ZERO arm never
        # loads them; they are still built and bound so the launch signature does
        # not change shape with the arm.
        dtype = targets[0].dtype
        rows = self.shape[0]
        self._imr_rows = tuple(
            CupyPointer(_word_view(xp.ascontiguousarray(
                imr_coefficient_row(xp, spec["targets"][index], sign, m, dtdx,
                                    rows, dtype).reshape(-1))))
            for index, _register, sign in IMR_TERMS[sub_step])
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        # The override exists for exactly one caller: the gate's mutation legs.
        self._kernel = kernel

    def prefix(self) -> Any:
        """This launch's radial prefix, from the shipped array-path scan."""
        return cylindrical_complex_prefix(self.xp, self.sub_step, self._source_map,
                                          scratch=self.scratch)

    def run(self, guard: Optional[bool] = None) -> None:
        """Launch the sub-step. In place; the driver's references stay valid.

        ``guard`` is not for callers: it exists so the gate can MEASURE the
        contraction guard's effect (identical with, non-identical without).
        """
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = self._kernel if self._kernel is not None else cyl_complex_pml_curl_step
        (minus_dtdx, inc_b) = self.increment_scalars
        nx, ny, nz = self.shape
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *self._targets, *self._aux, *self._sources,
            CupyPointer(_word_view(self.prefix())), *self._imr_rows,
            *self._coefficients,
            nx, ny, nz, self.n_elem, self.dtdx,
            minus_dtdx, inc_b[0], inc_b[1], self.four_dtdx,
            BACKWARD=self.backward,
            BCZ=self.bcz,
            M_CLASS=self.m_class,
            ZERO_ROWS=self.zero_rows,
            EXPANSION=self.expansion,
            BLOCK=self.block,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )

    def __repr__(self) -> str:
        return (f"CylindricalComplexCurlPlan({self.sub_step}, shape={self.shape}, "
                f"m_class={self.m_class}, zero_rows={self.zero_rows}, "
                f"bcz={self.bcz}, expansion={self.expansion}, block={self.block})")


def plan_cylindrical_complex_curl(fields: Any, pml: Any, sub_step: str,
                                  block: Optional[int] = None,
                                  num_warps: Optional[int] = None,
                                  probe: Any = None
                                  ) -> Optional[CylindricalComplexCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the ONLY refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly. The Triton import stays BELOW the predicate, so a
    NumPy host can plan (to ``None``) without the optional dependency at all.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    if not cylindrical_complex_curl_coverage(fields, pml, sub_step, probe=probe).covered:
        return None
    expansion = _resolve_expansion(probe)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None

    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    return CylindricalComplexCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, int(grid.m),
        bool(grid.accurate_fields_near_cylorigin),
        1 if kinds[2] == "metallic" else 0,
        expansion, DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, "fu_" + name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["sources"]],
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")],
        grid.xp, scratch=getattr(fields, "scratch", None), num_warps=num_warps,
    )


def plan_cylindrical_complex_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any], dtdx: float,
        m: int, accurate_fields_near_cylorigin: bool, bcz: int, expansion: int,
        xp: Any, scratch: Any = None, block: Optional[int] = None,
        kernel: Any = None, num_warps: Optional[int] = None
) -> CylindricalComplexCurlPlan:
    """Build a plan from bare device arrays — the gate's and benchmark's route.

    ``arrays`` is keyed by component name (complex64 volumes), ``flat`` by
    ``kms_x``/``sinv_x``/... on the sub-lattice the caller ALREADY SELECTED
    (half-integer for B, integer for D), so the gate can hand over a swapped pair
    and measure that the swap is caught. No predicate runs: the caller is a
    harness that constructed the configuration deliberately, including the
    deliberately wrong ones, and ``kernel=`` carries the mutation override —
    dropping it silently disarms every mutation leg (launch.py:516-522's lesson).
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    return CylindricalComplexCurlPlan(
        sub_step, shape, dtdx, m, accurate_fields_near_cylorigin, bcz, expansion,
        DEFAULT_BLOCK if block is None else block,
        [arrays[name] for name in spec["targets"]],
        [arrays["fu_" + name] for name in spec["targets"]],
        [arrays[name] for name in spec["sources"]],
        [flat[f"{stem}_{axis}"] for axis in "xyz" for stem in ("kms", "sinv")],
        xp, scratch=scratch, kernel=kernel, num_warps=num_warps,
    )


def plan_cylindrical_complex_constitutive(fields: Any, pml: Any, side: str,
                                          block: Optional[int] = None,
                                          num_warps: Optional[int] = None,
                                          probe: Any = None
                                          ) -> Optional[ComplexConstitutivePlan]:
    """Build the CERTIFIED complex constitutive plan for the admitted Dcyl slice.

    The compiled arithmetic is ``complex_fields.bloch_constitutive_step``, which
    is already certified (job 2330/2343); complex cylindrical changes neither its
    pointwise expression nor its coefficient layout, and the identity leg in the
    module docstring measures that. This wrapper owns the distinct predicate, so
    the complex tranche cannot silently broaden its blanket Dcyl refusal.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, got {side!r}")
    if not cylindrical_complex_constitutive_coverage(fields, pml, side,
                                                     probe=probe).covered:
        return None
    expansion = _resolve_expansion(probe)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None

    spec = CONSTITUTIVE_SIDES[side]
    suffix = "_h" if spec["half_integer"] else ""
    return ComplexConstitutivePlan(
        side, fields.grid.shape, expansion,
        DEFAULT_BLOCK if block is None else block,
        [getattr(fields, name) for name in spec["targets"]],
        [getattr(fields, name) for name in spec["aux"]],
        [getattr(fields, name) for name in spec["sources"]],
        ([fields.inverse_epsilon_for(name) for name in spec["targets"]]
         if side == "E" else None),
        [getattr(pml, f"{stem}_{axis}{suffix}")
         for axis in "xyz" for stem in ("kps", "kms")],
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# WIRING — the planner seam is IN; dispatch is not; the byte gate is OWED a run
# ---------------------------------------------------------------------------
#
# This block described the deferral and outlived it, which turned it into a false
# statement about ``launch.py`` and ``__init__.py``. As it now stands: those two
# files carry the forwarders, the ``plan_step`` arms (gated on the cylindrical
# axis AND complex storage) and the ``__all__`` entries; ``fastpath`` is
# untouched and keeps returning None on every branch, so dispatch stays disabled.
#
# WHAT MADE THE DEFERRAL SAFE IS WHAT MAKES THE WIRING SAFE, and it is kept:
# ``coverage._grid_reasons`` clauses 2/6/7 and
# ``complex_fields._complex_grid_reasons`` clause 4 already refuse every complex
# Dcyl run, and ``cylindrical_triton``'s predicate refuses ``force_complex_fields``
# BY NAME (cylindrical_triton.py:555-556) while this one REQUIRES it — the split
# moved from m to STORAGE on 2026-09-04, when this family admitted m = 0 under
# complex64 storage — so no shipped predicate admits a configuration this one
# also admits, and the planner's disjointness sweep measures that on real Dcyl
# triples.
#
# THE OWED RUN IS THE PRECONDITION THIS BLOCK ALREADY NAMED. The coordinated
# change that added the forwarders is the one that re-runs the byte gate, and it
# has not run: ``fingerprints.json``'s ``pending_host_recut`` declares the drift
# for ``launch.py`` and ``__init__.py``, names the two red tests, and is the only
# honest way to carry it until a CUDA host clears it.
