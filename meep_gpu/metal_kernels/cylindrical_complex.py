"""COMPLEX cylindrical (Dcyl, complex64 storage at every m) PML curl on Metal — ONE
new kernel, three compile-time bodies (m = 0, |m| = 1, |m| >= 2).

TEMPLATE FOLLOWED: :mod:`.complex_fields`, end to end and deliberately. Its
``float2`` volume binding, its ``_CURL_TEMPLATE`` shape, its probe-bound
``EXPANSION`` arm, its ``ComplexPmlCurlPlan`` argument layout, its predicate spine
and its ``register_arms`` are the idiom this family extends. Where a fact is that
module's, it is IMPORTED rather than re-spelled — one idiom, not two.

THE LARGEST REMAINING SURFACE ON THIS BACKEND: sixteen lifted corpus rows,
SIXTY-FOUR sub-step slots, and the family carrying the unbounded ``ZERO_ROWS``
axis that made the port look like a week's work. It is not: see the arity note.

======================================================================
THE 64 SLOTS SPLIT 32/32, AND ONLY HALF NEED A KERNEL
======================================================================

``stepping.update_H`` (:907-924) and ``update_E`` (:926-993) carry NO cylindrical
branch of any kind — the words "cylindrical", "is_axis" and "m" do not appear in
either function. They are element-wise with per-axis coefficient tables indexed on
the component's OWN axis, and Dcyl changes nothing there except that axis 1 has
n = 1. So the CERTIFIED :func:`complex_fields.bloch_constitutive_source` body
already computes them, and this module contributes a restated PREDICATE plus a plan
builder that delegates to :class:`complex_fields.ComplexConstitutivePlan`. That is
the same shape :mod:`.folded_complex` took for its constitutive pair.

THE IDENTITY IS ESTABLISHED BY MEASUREMENT, NOT BY GREP. The certified Triton
reference ran ``stepping.update_H``/``update_E`` on a real complex Dcyl
``Fields``/``PML`` against a plain complex elementwise ``dsigw`` reference carrying
no cylindrical clause: 480/480 rows, 0 differing uint32 words, over
m in {1,-1,2,-2,3,5} x {z metallic, z periodic} x five Courant numbers x two shapes
x {random, signed-zero} seeding x {H, E} (harness ``run_constitutive_identity`` in
``parity/meep_gpu/gate_triton_cylindrical_complex.py``). That is a fact about
``stepping.py``, which is backend-free, so it transfers where a fact about a
compiler would not.

======================================================================
WHY ``ZERO_ROWS`` IS A RUNTIME UNIFORM AND NOT A SPECIALISATION
======================================================================

The Triton kernel carries ``ZERO_ROWS`` as a ``tl.constexpr`` whose value is
``abs(m)``, so its variant count is ``8 x UNBOUNDED`` — no author can enumerate it.
Two measurements collapse that here, and neither is inherited:

1. ``parity/meep_gpu/probe_metal_dynamic_loop_arity.py`` measured, ON THIS HOST,
   that a runtime-uniform THRESHOLD (``i < zero_rows`` read from a
   ``constant uint&``) is bit-identical to the literal-threshold twin — 12/12
   cases, 3,145,728 words compared, 0 differing, max ULP 0, in BOTH contraction
   modes, with the off-by-one control CAUGHT at exactly 1/64 of lanes. **The CUDA
   track's favourable PTX result was NOT assumed to transfer**; this backend has no
   disassembly, so the question was settled behaviourally before the family was
   built on top of it.
2. ``ZERO_ROWS`` IS NOT AN ARITY AND NEVER WAS. In the certified Triton body it is
   spent at exactly ONE site (``triton_kernels/cylindrical_complex.py:971``,
   ``near = i < ZERO_ROWS``) feeding twelve ``tl.where`` calls. It is an integer
   comparison on the row index — no accumulation, no trip count, and NO FLOAT
   ROUNDS ANYWHERE ON THAT PATH. So the weaker threshold result is all this family
   needs; the arity result licenses ADE ``update_P``, not this.

The unbounded axis therefore disappears. What remains is
``BACKWARD x BCZ x M_CLASS`` = **8 sources per contraction mode**, finite and
enumerated by :func:`enumerate_cylindrical_sources`.

``M_CLASS`` STAYS A COMPILE-TIME BRANCH and that is not an inconsistency: |m| = 1
and |m| >= 2 compute DIFFERENT ARITHMETIC (an axis-row replacement versus a
near-axis zeroing of six volumes), not the same arithmetic at a different count.
A threshold folds to a uniform; two different bodies do not.

======================================================================
THE r AXIS COSTS THE KERNEL NO BRANCH — one reading, one measurement
======================================================================

``stepping._boundary_kinds`` returns ``CYL_AXIS`` on r (:2145), which is outside
``COVERED_BOUNDARIES``; the kernel compiles ``BCX = METALLIC`` anyway. Two halves,
and they are established differently ON PURPOSE:

* the FAR ghost is a SOURCE IDENTITY. ``_shift_up`` (:1781-1783) is ONE branch over
  ``(MIRROR, METALLIC, CYL_AXIS)`` writing the same hard zero for all three. Nothing
  to measure; ``test_metal_cylindrical_complex`` asserts the branch text so a future
  split of that branch fails a test rather than a field.
* the NEAR ghost is NOT. ``_shift_down``'s ``CYL_AXIS`` branch (:1830-1842) writes
  the ``r_to_minus_r`` image where ``METALLIC`` writes zero. The kernel's claim is
  that ``_mask_non_owned_cells`` (:1898-1902) zeroes the only row that can consume
  it — every array the shift helpers move has Yee shift 1 on r, and the two terms
  taking an r down-shift (``Dy`` partner ``Hz``, ``Dz`` partner ``Hp``) both have
  r-shift 0. That is a claim about ``stepping.py``, so it is MEASURED on this host
  by leg ``axis_identity`` of ``parity/meep_gpu/probe_metal_cylindrical_complex.py``:
  the real sub-step against the same sub-step with ``boundaries[0]`` forced to
  ``metallic``, over m in {1,-1,2,-2,3,5} x {z metallic, z periodic} x {B, D}.

A NONZERO COUNT THERE IS A LEGITIMATE OUTCOME and would mean this family must carry
the near ghost. The probe reports the count; this module does not assume it.

======================================================================
THE FOUR CYLINDRICAL ADDITIONS, in the order the kernel applies them
======================================================================

1. **THE RADIAL PREFIX STAYS ON THE ARRAY PATH.**
   ``stepping.cylindrical_rderiv_prefix`` (:1257) is a SEQUENTIAL prefix scan whose
   float32 summation order DEFINES the answer, and this module CALLS it rather than
   re-deriving it. B side (:299-347): Ep = ``Ey`` extended by ONE ZERO WALL ROW to
   ``(nr+1, ny, nz)``, prefixed at ``ir0 = 0.0``; ``Bz``'s WHOLE curl becomes
   ``dtdx * (prefix_ext[1:] - prefix_ext[:-1])`` — ONE subtract, ONE multiply, NOT
   the four-operand grouping, which is a different float32 number per plane.
   D side (:414-426): Hp = ``Hy`` prefixed at ``ir0 = 0.5``, substituted for the
   ``Dz`` term's ``first`` source only; ``Dx``'s ``Hy`` operands stay raw.

   **AND THIS IS THE ONE PLACE THIS BACKEND IS GENUINELY WORSE THAN THE CuPy ONE.**
   There the scan is a device ``cumsum`` on the engine's own array, zero copy. Here
   the engine holds NumPy (``device.py:21-28``), so the prefix is computed on the
   HOST and the plan must, per launch: sync the prefix SOURCE volume OUT, scan it,
   and sync the prefix IN. That is one component out and one volume in per curl
   sub-step, against a residency layer that exists because a full round trip costs
   12x-64x the kernel (``device.py:167-172``). It is not a correctness problem and
   not a blocker, but **this family's throughput claim must be measured separately
   and may not be inherited from the certified families' benchmark.**
   :meth:`CylindricalComplexCurlPlan.run` performs both syncs so no caller can
   forget one, and counts them so a gate can assert they happened.

2. **THE i*m/r COUPLING** (:674-724; call sites :348-355 B, :430-437 D). As one
   complex update ``delta_f = -i * s * 2m * Courant / r2 * g`` with
   ``r2 = 2*ir + iyee_shift_r(target)``. Four call sites and only four —
   ``Bx <- Ez`` (+1), ``Bz <- Ex`` (-1), ``Dx <- Hz`` (-1), ``Dz <- Hx`` (+1);
   target 1 gets nothing on either side. THE PARTNERS ARE CENTER LOADS THE CERTIFIED
   BODY ALREADY MAKES (``c`` = g2, ``a`` = g0), so the term costs no new pointer.

   **THE COEFFICIENT ROW IS HOST-BUILT AND BOUND, NEVER SYNTHESISED IN-KERNEL**, and
   two independent Metal facts force that rather than suggest it: ``_build`` clamps
   with ``xp.maximum(r_doubled, 1.0)`` and Metal's ``max`` builtin returns the SECOND
   operand when both are zeros and drops NaN, disagreeing with ``numpy.maximum``
   (``shaders.py:41-46``); and the divide is FLOAT64 on the host, rounded ONCE by
   ``.astype(dtype)``, where an in-kernel divide would need a correctly-rounded f32
   divide and ``fast::divide`` is measured-divergent here. :func:`imr_coefficient_row`
   is the transcription; the kernel never divides and never takes a max.

3. **THE |m| = 1 AXIS-ROW INCREMENTS** (:625-645 B, :524-557 D), which REPLACE curl
   row 0 and do so AFTER the ownership mask. A REPLACEMENT, never an accumulation:
   the masked row is exactly ``+0.0`` and ``+0.0 + x == x`` for every x EXCEPT
   ``-0.0``, so an accumulating spelling is a signed-zero defect a random seed
   misses. Folded into the curl rather than post-added because the PML recurrence
   must apply MEEP's own axis ladder — :536-544 records the measurement that forced
   it (a plain post-add is Ep 4.6e-01 / Hr 2.39 wrong under r+z PML at |m| = 1).

4. **THE PER-|m| AXIS RULES**, applied to the STORED value after the recurrence.
   |m| = 1: the D side zeroes ``Dz`` on the axis row — the FIELD ONLY, never
   ``fu_Dz`` — and the B side does NOTHING (:661-663 branches on ``m == 0`` and
   ``abs(m) > 1``). |m| >= 2: all three components AND their ``fu_`` auxiliaries
   held at zero on rows ``[0:|m|]``, or on row 0 alone under
   ``accurate_fields_near_cylorigin``. The component set is POST-#3164 (upstream
   593a4b42, first released in 1.33.0); the 1.29-era code zeroed only Dp/Dz and Br
   and the two references genuinely differ (:571-579).

5. **THE m = 0 ARM** (added 2026-09-04). Complex64 storage at m = 0 is a
   configuration the engine constructs freely — ``force_complex_fields`` is an
   independent switch, and the corpus row ``examples/dipole_in_vacuum_cyl_off_axis.py``
   ((150, 1, 300), m = 0, PML, ``force_complex_fields=True``) IS one — and until this
   round it fell between two partitions: this family refused it by name as "the
   cylindrical REAL product's kernel" and the real family refused it on storage.
   ``M_ZERO`` closes that gap. It is a THIRD COMPILE-TIME BODY, not the |m| >= 1
   arithmetic evaluated at m = 0, and each of its three differences is transcribed
   from the array path's own branches rather than inferred:

   * **the i*m/r block is ABSENT**, not multiplied by a zero row. ``step_B`` (:348)
     and ``step_D`` (:430) branch on ``grid.m != 0`` and never form the term; a zero
     coefficient row through ``c_mul`` would still write ``curl - (±0.0, ±0.0)``,
     which turns a ``-0.0`` curl word into ``+0.0``. The two coefficient rows stay
     BOUND — one signature serves all three arms, and the fused pairs sit on the
     binding ceiling — and are never read;
   * **no axis-row increment on either side** (:545-551 returns ``None`` at m = 0,
     :638 returns ``None`` for |m| != 1);
   * **the m = 0 axis rules, on the STORED value after the recurrence**, in the
     array path's own order and component set: B side ``Bx[r = 0] = 0`` (:661-662,
     the field only); D side ``Dz[r = 0] += (4*Courant) * Hp[r = 0]`` and then
     ``Dy[r = 0] = 0`` (:586-587, the fields only, never ``fu_``). The increment is
     a POST-ADD and NOT a curl fold: the fold was MEASURED wrong for m = 0 under PML
     (stepping.py:575-580 — Er 2.7e-01 / Hp 4.5e-01 against the post-add's
     3.6e-07), the same finding the real m = 0 product transcribes, and the fold is
     an armed mutation here. ``Hp`` is the ``g1`` centre load ``b`` — the stored
     ``Hy`` that ``fields.get_H("Hy")`` returns under an active PML, which
     ``step_D`` does not write. The multiply is
     ``c_mul_coefficient_left(axis_coef, b)``: a Python float on the LEFT of a
     complex64 volume, the ``python_float_left`` orientation the probe classified.
     ``axis_coef`` is HOST-ROUNDED — ``4.0 * (dt/dx)`` formed in float64 and bound
     as one ``constant float&`` — for the real family's reason: the array path
     forms the scalar in float64 and NumPy casts it once, so binding the rounded word
     is the literal transcription. Scaling by four is exact, so ``4.0f * dtdx``
     in-kernel is the same word; the gate carries that spelling as a NULL CONTROL
     rather than leaving the agreement to luck.

   What the arm does NOT change: the prefix substitution on both sides, the r-axis
   METALLIC compile, the ownership mask, the recurrence, and the constitutive pair,
   which has no cylindrical branch at any m. ``accurate_fields_near_cylorigin`` has
   no effect at m = 0 (``_cylindrical_axis_rows`` is consulted for |m| >= 2 only)
   and the gate drives both settings to show it. The arm costs the standalone curl
   ONE binding (26 -> 27, the ``axis_coef`` scalar) and the fused pairs NONE: there
   the scalar rides in the ``Params`` struct, whose itemsize does not move.

======================================================================
NEGATION AND SIGNED ZEROS — RE-MEASURED HERE, NOT INHERITED
======================================================================

The Triton cylindrical kernel spells every negated addend ``x * -1.0`` because
*Triton* lowers unary minus as ``0.0 - x`` and canonicalizes signed zeros. **THAT IS
A FACT ABOUT TRITON AND IT DOES NOT REPRODUCE HERE.** Leg ``negation`` of this
family's probe compiles all three spellings on this host in both contraction modes
and compares them against ``numpy.negative`` on an exhaustive signed-zero table;
this module ships ``-x``, which is what the array path performs (a sign flip on
every word, zeros included). The refuted spelling ``0.0f - x`` is a gate mutation
rather than a comment.

The i*m/r product keeps its LITERAL zero cross terms, through the certified
``c_mul``: the row's real word is exactly ``+0.0`` (the ``(-1j) * X`` spelling
launders the sign for both signs of X), so a plane-wise ``{re*c, im*c}`` shortcut is
byte-wrong on signed zeros in the product's own words. Whether that difference
SURVIVES the ``curl - m`` fold is a separate question and is left to the gate's
minuend census rather than asserted here.

======================================================================
THE INVERTED CLAUSES — six families to stay disjoint from
======================================================================

``registry.py``'s contract: every family's grid-reason list carries the same
numbered questions and answers exactly one of them THE OTHER WAY, and a new family
STATES its inversion rather than inheriting disjointness. Two admitters leave a slot
UNSELECTED and it falls to the array path — a SILENT coverage loss — so a missed
inversion costs slots quietly.

CLAUSE 6 (CARTESIAN) IS INVERTED, and it alone separates this family from ALL NINE
carried families. ``coverage._grid_reasons`` clause 6 (:176-182) refuses
``grid.cylindrical`` and ``is_axis`` BY NAME; ``complex_fields``,
``folded_complex``, ``special_kz``, ``bfast_curl``, ``offdiag_update_e``,
``folded_offdiag_update_e``, ``symmetry`` and ``no_pml_constitutive`` each restate
it. HERE ``grid.cylindrical`` must be TRUE and ``is_axis`` must be exactly
``(True, False, False)``.

CLAUSE 2 (STORAGE) IS INVERTED against the shipped real families and is THE clause
that separates this family from the CYLINDRICAL REAL m = 0 product, in both
directions: complex64 is REQUIRED here and REFUSED there
(``cylindrical_real._cylindrical_real_grid_reasons`` clause 2). Since 2026-09-04
the two families partition Dcyl by STORAGE and by storage alone — a complex64 run at
any integer m is this family's, a float32 run is the real family's (and the engine
admits a float32 run only at m = 0, stepping.py:731-736).

CLAUSE 14 USED TO REFUSE m = 0 BY NAME AND NO LONGER DOES. Until 2026-09-04 this
family carried |m| >= 1 only and named the real product as m = 0's owner; that was
correct about the |m| >= 1 arithmetic and WRONG about the partition, because the
real product refuses complex64 storage, so a complex64 m = 0 run — a constructible
one, and a corpus row — was refused by both and fell to the array path on every
slot. The ``M_ZERO`` arm (addition 5 above) carries it here, where the storage
already is. What survives of clause 14 is the READABILITY requirement: ``grid.m``
must be an integer, because the arm and the near-axis uniform are functions of it
and an unreadable m would bind the wrong body.

CLAUSE 4 (BOUNDARIES) IS WIDENED, PER AXIS, AND NOT GLOBALLY. ``'axis'`` is admitted
on r ONLY; phi must resolve PERIODIC; z may be METALLIC or PERIODIC. A global
widening would silently admit an axis-kind z, which no rule in this kernel serves.

CLAUSE 5 (FOLD) IS REFUSED TWICE, and that is not tidiness. ``_boundary_kinds``
puts ``is_axis`` AHEAD of ``is_mirrored`` (:2144-2148), so a folded r axis reports
``CYL_AXIS`` and THE FOLD VANISHES SILENTLY; and ``_mirror_phases`` (:2291-2308)
puts ``(-1)**grid.m`` into the SAME SLOT the mirror phase occupies. The grid flag
alone is not the inversion. (``Grid`` also refuses a mirror on a Dcyl cell,
grid.py:648-653 — which is exactly the "another module already guards it" reasoning
this file exists to refuse.)

CLAUSE 7 (k = 0) IS KEPT UN-INVERTED. Dcyl Bloch is z-only in MEEP
(grid.py:654-658 refuses a k on r or phi), none of the sixteen demand rows carries
one, and the kernel compiles NO rotation at all. Admitting a z phase later is a
predicate widening plus a gate row, not a silent extension.

CLAUSES 10, 11, 12 ARE KEPT: no chi2/chi3, no BFAST, ``beta == 0``. None of the
sixteen rows carries a susceptibility, a conductivity, BFAST, chi2/chi3, an
off-diagonal epsilon or a Bloch phase.

THE M_CLASS SPLIT IS NOT A CLAUSE AND MUST NOT BECOME THREE FAMILIES. m = 0 versus
|m| = 1 versus |m| >= 2 is a source-substitution branch inside ONE family, exactly
as ``BACKWARD`` and ``BCZ`` are. Two arms registered on one slot is the composer's
"refuses rather than picking by table order" residual, not a win.

======================================================================
WHAT THIS FAMILY DOES NOT CLAIM, each by name
======================================================================

Real float32 storage at any m (the cylindrical REAL product's kernel); a
conductivity on any curl target; a registered susceptibility (complex-storage ADE is
unbuilt on either geometry); chi2/chi3; BFAST; ``grid.beta``; a Bloch phase; a
mirror plane; an off-diagonal chi1inv row on ``update_E``; and a radial extent below
two rows, where the |m| = 1 axis increment reads the FIRST OFF-AXIS row and NumPy
raises where CuPy silently returns row 0 (both measured by the Triton tranche).

REGISTERED-BUT-CANNOT-WIN IS NOT USED HERE, and the absence is stated. Every slot
this family claims IS ported. The capabilities it lacks are refused by name inside
predicates that would otherwise admit, which is the same naming through a cheaper
mechanism.

NO ENTRY IN ``fingerprints.json``'s ``kernel_source_sha256``, deliberately and for
:mod:`.complex_fields`' reason: these sources are a function of the PROBE-BOUND
expansion arm, so a checked-in hash would record a choice rather than a measurement.
This module's own BYTES are hashed under ``host_sha256``, because
``launch._host_modules()`` enumerates the package.

**THERE IS NO GENERATED-CODE AUDIT ON THIS BACKEND.** ``torch.mps.compile_shader``
exposes no AIR, no ISA and no optimisation report, so every leg behind this family is
BEHAVIOURAL: it catches a wrong answer, not a wrong instruction. That is this
backend's standing certification gap against the Triton and CUDA tracks and it stays
stated.

WIRED, WITH NO DEVICE BYTE GATE YET. The arms register ``wired=True`` so
``plan_step`` composes them, and the composition disjointness is what the matrix
rows measure. ``fastpath.plan_fast_path`` still returns ``None`` on every branch;
which track earns the step path is a separate decision this port does not make.

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
    _susceptibility_reasons,
)
from ..triton_kernels.launch import SUB_STEPS
from . import complex_fields, shaders, templates
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .device import MAX_BUFFER_BINDINGS, compile_source
from .plans import KernelPlan

#: The family name the arm table carries. ONE spelling, so a refusal message, a
#: registry row and an artifact column cannot drift apart.
FAMILY = "cylindrical_complex"

#: This family's probe artifact override. SEPARATE from the three that exist
#: (``..._COMPLEX_...`` for the unfolded complex family, ``MEEP_GPU_METAL_EXPANSION_PROBE``
#: for beta, ``..._FOLDED_COMPLEX_...`` for the folded complex fill) because this
#: family requires TWO orientations none of those artifacts classified, and silently
#: reading one of theirs would licence an arm from a record that never measured this
#: call.
PROBE_PATH_ENVIRONMENT = "MEEP_GPU_METAL_CYLINDRICAL_COMPLEX_EXPANSION_PROBE"

#: The array module the probe must have measured. The ENGINE holds NumPy on this
#: host and the kernel must reproduce the ENGINE's bytes.
PROBE_BACKEND = complex_fields.PROBE_BACKEND

#: THE FIRST PATTERN THIS FAMILY ADDS. ``stepping._cylindrical_imr_term`` (:724)
#: spells ``factor * partner_values`` with ``factor`` an ``(nr, 1, 1)`` complex64 ROW
#: and ``partner_values`` an ``(nr, ny, nz)`` complex64 VOLUME — a BROADCAST complex
#: product with the COEFFICIENT ON THE LEFT. None of ``complex_fields``' five base
#: orientations is that call.
IMR_ROW_PROBE_PATTERN = "c8_mul_c8_row_coefficient_left"

#: THE SECOND. ``_cylindrical_axis_increment_B`` (:644) spells
#: ``1j * (m*dtdx) * ez_off_axis`` — a PYTHON COMPLEX on the LEFT of a complex64
#: array, whose REAL WORD IS A SIGNED ZERO (``-0.0`` at m < 0). That operand class is
#: exactly what a plane-wise shortcut gets wrong, so it earns its own pattern rather
#: than inheriting ``c8_mul_c8``'s verdict.
AXIS_SCALAR_PROBE_PATTERN = "c8_mul_c8_scalar_left"

#: What a probe artifact must classify — and classify CONSISTENTLY — before this
#: family may bind an expansion arm. The base five are ``complex_fields``' own call
#: sites, restated THROUGH that module so there is one home for them.
CYLINDRICAL_PROBE_PATTERNS: Tuple[str, ...] = (
    complex_fields.PROBE_PATTERNS
    + (IMR_ROW_PROBE_PATTERN, AXIS_SCALAR_PROBE_PATTERN))

#: The three ``M_CLASS`` arms — three DIFFERENT BODIES, which is why the class stays a
#: compile-time branch where ``zero_rows`` became a runtime uniform. ``M_ZERO`` was
#: added 2026-09-04 (module docstring, addition 5): no i*m/r block, no axis-row
#: increment, and the m = 0 axis rules on the stored value.
M_ZERO = 0      # m = 0: Bx[r=0] = 0 (B); Dz[r=0] += 4*Courant*Hp[r=0], Dy[r=0] = 0 (D)
M_ONE = 1       # |m| = 1: axis-row increments on Bx (B) and Dy (D)
M_MANY = 2      # |m| >= 2: near-axis zeroing of all six volumes
M_ARMS: Tuple[int, ...] = (M_ZERO, M_ONE, M_MANY)
M_ARM_LABELS: Dict[int, str] = {M_ZERO: "m0", M_ONE: "m1", M_MANY: "mN"}

#: The boundary kinds this family requires on r and phi, and the two it admits on z.
#: r is the cylindrical axis and the kernel compiles it METALLIC — see the module
#: docstring's two-halves note and the probe's ``axis_identity`` leg.
REQUIRED_R_KIND = "axis"
REQUIRED_PHI_KIND = "periodic"
ADMITTED_Z_KINDS: Tuple[str, ...] = ("metallic", "periodic")

#: ``grid.is_axis`` on the three axes. Refused unless it is exactly this: the whole
#: kernel assumes r is the LEADING axis.
CYLINDRICAL_AXIS_FLAGS: Tuple[bool, bool, bool] = (True, False, False)

#: The smallest radial extent this product may step. The |m| = 1 axis increment reads
#: the FIRST OFF-AXIS row (``stepping.py:671``'s ``xp.take(Ez, 1, axis=0)``) and the
#: kernel's matching load ``off = nyz + j*nz + k`` is live on every lane, so below
#: this it would read a full plane PAST THE END of the source volume. Measured by the
#: Triton tranche on a real ``(1, 1, 20)`` grid: NumPy RAISES IndexError there and
#: CuPy SILENTLY RETURNS ROW 0 — the two backends disagree about whether the
#: configuration is steppable at all, which is why the clause is unconditional rather
#: than |m| = 1 only.
MINIMUM_RADIAL_ROWS = 2

#: The i*m/r call sites — ``stepping.py:376-383`` (B) and ``:430-437`` (D).
#: ``(target index, partner register, sign)``; the register names which CENTER load
#: of the certified curl body the term reuses ("c" = g2 = Ez/Hz, "a" = g0 = Ex/Hx).
#: Target 1 gets no term on either side.
IMR_TERMS: Dict[str, Tuple[Tuple[int, str, float], ...]] = {
    "step_B": ((0, "c", +1.0), (2, "a", -1.0)),   # Bx <- Ez (+1), Bz <- Ex (-1)
    "step_D": ((0, "c", -1.0), (2, "a", +1.0)),   # Dx <- Hz (-1), Dz <- Hx (+1)
}

#: Which target each sub-step's |m| = 1 axis-row increment REPLACES at r = 0.
AXIS_INCREMENT_TARGET: Dict[str, int] = {"step_B": 0, "step_D": 1}   # Bx / Dy

#: Per-sub-step prefix spec, transcribed from ``stepping.py:327-375`` / ``:414-426``.
PREFIX: Dict[str, Dict[str, Any]] = {
    "step_B": {"component": "Ey", "ir0": 0.0, "extend_wall_row": True},
    "step_D": {"component": "Hy", "ir0": 0.5, "extend_wall_row": False},
}

CURL_SOURCES: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

#: How many buffers the cylindrical complex curl binds: 9 float2 volumes + 1 prefix
#: + 2 i*m/r rows + 6 coefficient vectors + 5 shape/dtdx scalars + 1 zero-rows
#: uniform + 2 axis-increment scalars + 1 m = 0 axis coefficient (27 since
#: 2026-09-04; 26 before the ``M_ZERO`` arm). CHECKED AGAINST THE PLATFORM CEILING
#: AT IMPORT, for :mod:`.complex_fields`' reason — the ``float2`` design rests on the
#: margin and a binding added to the signature must be refused here BY NAME rather
#: than discovered as a compile error a reader has to attribute.
CURL_BINDINGS = 27
if CURL_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the cylindrical complex curl binds {CURL_BINDINGS} buffers and Metal's "
        f"ceiling on this toolchain is {MAX_BUFFER_BINDINGS}; the signature cannot "
        f"be built at all. See complex_fields.split_plane_curl_signature for the "
        f"measurement that established the ceiling")

#: Which constitutive side each slot names, so one pair of callables serves both.
_CONSTITUTIVE_SLOT_SIDES: Dict[str, str] = {"update_H": "H", "update_E": "E"}

__all__ = [
    "ADMITTED_Z_KINDS",
    "ARMS",
    "AXIS_INCREMENT_TARGET",
    "AXIS_SCALAR_PROBE_PATTERN",
    "CURL_BINDINGS",
    "CURL_SOURCES",
    "CYLINDRICAL_AXIS_FLAGS",
    "CYLINDRICAL_PROBE_PATTERNS",
    "CylindricalComplexCurlPlan",
    "FAMILY",
    "IMR_ROW_PROBE_PATTERN",
    "IMR_TERMS",
    "MINIMUM_RADIAL_ROWS",
    "M_ARMS",
    "M_ARM_LABELS",
    "M_MANY",
    "M_ONE",
    "M_ZERO",
    "PREFIX",
    "PROBE_BACKEND",
    "PROBE_PATH_ENVIRONMENT",
    "REQUIRED_PHI_KIND",
    "REQUIRED_R_KIND",
    "SUB_STEPS",
    "axis_coefficient",
    "axis_increment_scalars",
    "cylindrical_complex_constitutive_coverage",
    "cylindrical_complex_pml_curl_coverage",
    "cylindrical_curl_source",
    "cylindrical_prefix",
    "enumerate_cylindrical_sources",
    "expansion_from_probe",
    "imr_coefficient_row",
    "load_expansion_probe",
    "m_class",
    "plan_cylindrical_complex_constitutive",
    "plan_cylindrical_complex_pml_curl",
    "plan_cylindrical_complex_pml_curl_from_arrays",
    "register_arms",
    "zero_rows",
]


# ---------------------------------------------------------------------------
# Host-side constants that must be computed the ARRAY PATH's way
# ---------------------------------------------------------------------------

def m_class(m: int) -> int:
    """``M_CLASS`` for one azimuthal order: ``M_ZERO``, ``M_ONE`` or ``M_MANY``.

    m = 0 has been an arm since 2026-09-04 (it raised here before, naming the real
    product as its owner — see the module docstring's clause 14 note for why that
    was the wrong partition).
    """
    magnitude = abs(int(m))
    if magnitude == 0:
        return M_ZERO
    return M_ONE if magnitude == 1 else M_MANY


def axis_coefficient(dtdx: float) -> float:
    """The m = 0 on-axis Dz coefficient, formed exactly as the array path forms it.

    ``stepping._cylindrical_axis_zero_D`` (:586) writes
    ``fields.Dz[r=0] += (4.0 * (grid.dt / grid.dx)) * fields.get_H("Hy")[r=0]``: a
    PYTHON FLOAT formed in float64, which NumPy casts to complex64 ONCE before the
    multiply. Binding that float into a ``constant float&`` performs the same single
    rounding, so the kernel's word is the array path's word. Returned as a Python
    float rather than pre-rounded to float32 for the same reason ``dtdx`` is: the
    binding rounds correctly and a second rounding would be a second place for the
    two to disagree. (Scaling by four is exact in binary floating point, so
    ``4.0f * dtdx`` in-kernel gives the same bits; the gate pins that agreement as a
    measured null rather than relying on it.)
    """
    return 4.0 * float(dtdx)


def zero_rows(m: int, accurate_fields_near_cylorigin: bool) -> int:
    """How many near-axis rows the |m| >= 2 rule holds at zero.

    ``stepping._cylindrical_axis_rows`` (:601-622): ``slice(0, 1)`` on the ACCURATE
    branch (an ordinary axis boundary condition and nothing else, stable only below
    Courant ~1/(|m| + 0.5), which ``Grid`` refuses above) and ``slice(0, |m|)`` on
    MEEP's default stability hack. Zero at |m| = 1, where neither branch applies.

    THIS IS THE VALUE THE KERNEL TAKES AS A RUNTIME UNIFORM rather than as a
    specialisation — see the module docstring for the two measurements that licence
    it, and note that it is a THRESHOLD on the row index, not a trip count.
    """
    if abs(int(m)) < 2:
        return 0
    return 1 if bool(accurate_fields_near_cylorigin) else abs(int(m))


def imr_coefficient_row(xp: Any, target: str, sign: float, m: int, dtdx: float,
                        rows: int, dtype: Any) -> Any:
    """The per-r i*m/r coefficient, built the array path's way and ONLY that way.

    ``stepping._cylindrical_imr_term._build`` (:713-718), transcribed. Every step is
    load-bearing:

    * the ``arange`` is FLOAT64 and the doubled coordinate is ``2*ir + iyee_r`` with
      ``iyee_r`` the target's OWN radial Yee shift (``fields.IYEE_SHIFTS``);
    * the clamp is ``maximum(.., 1.0)``, which bites only on row 0 of a shift-0
      target, whose curl the ownership mask zeroes. **IT STAYS ON THE HOST**: Metal's
      ``max`` builtin returns the SECOND operand when both are zeros and drops NaN,
      disagreeing with ``numpy.maximum`` (``shaders.py:41-46``);
    * the divide is FLOAT64 and the whole expression is rounded ONCE by
      ``.astype(dtype)``. **IT STAYS ON THE HOST** for the second reason: an in-kernel
      divide would need a correctly-rounded f32 divide and ``fast::divide`` is
      measured-divergent on this platform;
    * the numerator is spelled ``(-1j) * (sign * 2.0 * m * dtdx)`` — multiplied by
      MINUS the imaginary unit, which launders the real word to ``+0.0`` for BOTH
      signs of the real factor. The SIGN of the unit does that, not the operand
      order: ``(-1j)*X`` and ``X*(-1j)`` are the same bits, while flipping the unit's
      sign or moving the negation outside the product changes the real word.
    """
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    iyee_r = IYEE_SHIFTS[target][0]
    r_doubled = 2 * xp.arange(int(rows), dtype=xp.float64) + iyee_r
    divisor = xp.maximum(r_doubled, 1.0).reshape(-1, 1, 1)
    return ((-1j) * (float(sign) * 2.0 * int(m) * float(dtdx))
            / divisor).astype(dtype)


def axis_increment_scalars(m: int, dtdx: float) -> Tuple[float, Tuple[float, float]]:
    """The two |m| = 1 B-side host scalars, rounded as NEP-50 rounds them.

    ``stepping._cylindrical_axis_increment_B`` (:643-644) writes

        (-dtdx) * (ep[0] - ep_above[0]) - 1j * (m * dtdx) * ez_off_axis

    where both scalars meet a complex64 array as WEAK Python scalars, so each is
    converted to complex64 once and the second multiply is a FULL complex product
    with its zero cross terms.

    Returns ``(minus_dtdx, (re, im))``. The first is a PYTHON FLOAT and stays one —
    that is the ``python_float_left`` orientation and
    ``c_mul_coefficient_left`` is the certified helper for it. The second is
    ``1j * (m*dtdx)``, whose REAL WORD IS A SIGNED ZERO: multiplied by PLUS the
    imaginary unit, CPython computes ``re = 0.0*X - 1.0*0.0``, which is ``-0.0`` for
    X < 0. That is the OPPOSITE laundering from the i*m/r row's ``(-1j) * X``, and
    the difference is the SIGN OF THE UNIT rather than the operand order — so both
    spellings are host-rounded and passed through, never synthesised in-kernel.
    """
    import numpy  # noqa: PLC0415

    second = numpy.complex64(1j * (int(m) * float(dtdx)))
    return (float(numpy.float32(-float(dtdx))),
            (float(numpy.float32(second.real)), float(numpy.float32(second.imag))))


def cylindrical_prefix(xp: Any, sub_step: str, sources: Dict[str, Any],
                       scratch: Any = None) -> Any:
    """The radial prefix the kernel consumes, from the SHIPPED array-path scan.

    ``stepping.cylindrical_rderiv_prefix`` is CALLED, never re-derived: its float32
    summation order DEFINES the answer, and a parallel or blocked scan reassociates
    it. (``templates.COLUMN_SERIAL_SCAN`` carries the same licence for a future
    device scan; this family does not use it, because moving the scan to the device
    would change the answer unless the device scan were column-serial AND the
    engine's own array module agreed with it, and the engine here is NumPy.)

    B side: Ep = ``Ey`` EXTENDED BY ONE ZERO WALL ROW (``stepping.py:342-361``,
    including the pooled-versus-fresh allocation branch, because the two writes are
    what ``xp.concatenate`` used to do). D side: Hp = ``Hy``, unextended — the
    backward difference reads rows ``i`` and ``i-1``.
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


def prefix_shape(sub_step: str, shape: Sequence[int]) -> Tuple[int, int, int]:
    """The prefix volume's own shape — ``(nr+1, ny, nz)`` on B, ``(nr, ny, nz)`` on D."""
    nr, ny, nz = (int(n) for n in shape)
    extra = 1 if PREFIX[sub_step]["extend_wall_row"] else 0
    return (nr + extra, ny, nz)


# ---------------------------------------------------------------------------
# The curl kernel
# ---------------------------------------------------------------------------
#
# TWENTY-SIX BINDINGS (see CURL_BINDINGS), five under the platform ceiling. The
# volumes are `float2` for `complex_fields`' measured reason: the re/im-split form
# of the certified complex curl already needs 35 and is a compile error, and this
# kernel binds three MORE buffers than that one.

_CURL_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

kernel void cyl_complex_pml_curl_step(
    device float2*       f0        [[buffer(0)]],
    device float2*       f1        [[buffer(1)]],
    device float2*       f2        [[buffer(2)]],
    device float2*       u0        [[buffer(3)]],
    device float2*       u1        [[buffer(4)]],
    device float2*       u2        [[buffer(5)]],
    device const float2* g0        [[buffer(6)]],
    device const float2* g1        [[buffer(7)]],
    device const float2* g2        [[buffer(8)]],
    device const float2* pfx       [[buffer(9)]],
    device const float2* c0        [[buffer(10)]],
    device const float2* c2        [[buffer(11)]],
    device const float*  kmx       [[buffer(12)]],
    device const float*  sinvx     [[buffer(13)]],
    device const float*  kmy       [[buffer(14)]],
    device const float*  sinvy     [[buffer(15)]],
    device const float*  kmz       [[buffer(16)]],
    device const float*  sinvz     [[buffer(17)]],
    constant uint&       nx        [[buffer(18)]],
    constant uint&       ny        [[buffer(19)]],
    constant uint&       nz        [[buffer(20)]],
    constant uint&       n_elem    [[buffer(21)]],
    constant float&      dtdx      [[buffer(22)]],
    constant uint&       zrows     [[buffer(23)]],
    constant float&      minus_dtdx [[buffer(24)]],
    constant float2&     inc_b     [[buffer(25)]],
    constant float&      axis_coef [[buffer(26)]],
    uint idx [[thread_position_in_grid]])
{
__GUARD__

__DECODE__
    int nxi = int(nx);
    int nyz = nyi * nzi;

    // --- the ghost rule, per axis (stepping._shift_up:1723 / _shift_down:1787) --
    // r is CYL_AXIS and is compiled METALLIC. The FAR ghost is a source identity
    // (:1781-1783 is one branch over MIRROR/METALLIC/CYL_AXIS); the NEAR ghost
    // (:1830-1842) is the r_to_minus_r image and is UNOBSERVABLE because the
    // ownership mask zeroes the only row that could consume it — measured on the
    // array path by the family probe's `axis_identity` leg, never assumed.
    // phi is PERIODIC on a length-1 axis: the wrap returns the SAME element, which
    // is what makes the difference an exact +0.0. Computed, never elided.
    int si = i __SHIFT__, sj = j __SHIFT__, sk = k __SHIFT__;
    bool vx = true, vy = true, vz = true;
__GHOST_X__
__GHOST_Y__
__GHOST_Z__

    int ox = si * nyz + j * nzi + k;
    int oy = i * nyz + sj * nzi + k;
    int oz = i * nyz + j * nzi + sk;

    // METALLIC serves an exact complex 0 past the wall: a (+0.0, +0.0) word pair
    // IS the complex metallic ghost (S:1781-1783, S:1827-1829).
    float2 a   = g0[ii];
    float2 b   = g1[ii];
    float2 c   = g2[ii];
    float2 a_y = vy ? g0[oy] : float2(0.0f, 0.0f);
    float2 a_z = vz ? g0[oz] : float2(0.0f, 0.0f);
    float2 b_x = vx ? g1[ox] : float2(0.0f, 0.0f);
    float2 b_z = vz ? g1[oz] : float2(0.0f, 0.0f);
    float2 c_x = vx ? g2[ox] : float2(0.0f, 0.0f);
    float2 c_y = vy ? g2[oy] : float2(0.0f, 0.0f);

    // NO BLOCH ROTATION. This family refuses a k_point by name and Grid refuses one
    // on r or phi outright, so there is no phase block at all — not a multiply by
    // 1+0j, which would be a different float32 number on a signed zero.

    // --- the curl (stepping._curl_from_operands:1601): DO NOT flatten these parens
    float2 t0 = ((c_y - c) + (b - b_z));
    float2 t1 = ((a_z - a) + (c - c_x));
    float2 t2 = ((b_x - b) + (a - a_y));
    float2 curl0 = c_mul_coefficient_left(dtdx, t0);
    float2 curl1 = c_mul_coefficient_left(dtdx, t1);
    float2 curl2 = c_mul_coefficient_left(dtdx, t2);

    // --- the cylindrical substitution on target 2 (Bz / Dz) --------------------
__PREFIX__

    // --- the i*m/r coupling (stepping :674-724; sites :348-355 B, :430-437 D) ---
    // AFTER the dtdx curl, BEFORE the ownership mask — the array path's order.
__IMR__

    // --- ownership mask (stepping._mask_non_owned_cells:1865) -------------------
    // r masks for MEEP's own reason: little_owned_corner0 deliberately excludes
    // r = 0 ("which is updated separately", vec.hpp:1100-1104) — the axis row
    // belongs to the per-m rules. Emitted by the CERTIFIED emitter over the
    // (METALLIC, PERIODIC, BCZ) triple, which is exactly the r/phi/z enumeration.
    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);
__MASK__

    // --- the |m| = 1 axis-row increment, AFTER the mask, REPLACING the row ------
    // stepping :370-372 (B) / :451-453 (D). The masked row is exactly +0.0 and
    // `+0.0 + x == x` for every x EXCEPT -0.0, so REPLACE and never accumulate.
    // Negation is `-x`: on THIS backend that is a sign-bit operation and
    // `0.0f - x` is a different one, re-measured by this family's own probe rather
    // than inherited from the Triton track's opposite finding.
__AXIS_INCREMENT__

    // --- split-field recurrence (stepping._apply_pml_update:1905) ---------------
    // dsig/dsigu follow vec.hpp's cycle_direction and are the same triple on both
    // sides: target 0 takes (y, z), target 1 (z, x), target 2 (x, y). Every
    // multiply is the zero-imaginary complex product with the FIELD ON THE LEFT
    // (S:1929-1935); the loaded p registers ARE S:1928's fprev copy.
    float km_x = kmx[i], si_x = sinvx[i];
    float km_y = kmy[j], si_y = sinvy[j];
    float km_z = kmz[k], si_z = sinvz[k];

    float2 p0 = u0[ii];
    float2 n0 = c_mul_field_left(c_mul_field_left(p0, km_y) - curl0, si_y);
    float2 v0 = c_mul_field_left((c_mul_field_left(f0[ii], km_z) + n0) - p0, si_z);

    float2 p1 = u1[ii];
    float2 n1 = c_mul_field_left(c_mul_field_left(p1, km_z) - curl1, si_z);
    float2 v1 = c_mul_field_left((c_mul_field_left(f1[ii], km_x) + n1) - p1, si_x);

    float2 p2 = u2[ii];
    float2 n2 = c_mul_field_left(c_mul_field_left(p2, km_x) - curl2, si_x);
    float2 v2 = c_mul_field_left((c_mul_field_left(f2[ii], km_y) + n2) - p2, si_y);

    // --- the per-m axis rules, folded into the STORED value ---------------------
    // m = 0 (stepping :661-662 B, :586-587 D): Bx[r=0] = 0 on the B side; on the D
    //   side Dz[r=0] += (4*Courant)*Hp[r=0] as a POST-ADD and then Dy[r=0] = 0.
    //   FIELDS only, never fu_.
    // |m| = 1 (stepping :588-589): the D side zeroes Dz on the axis row — the FIELD
    //   ONLY, never fu_Dz — and the B side does NOTHING (:661-663 branches on
    //   m == 0 and abs(m) > 1).
    // |m| >= 2 (:591-599, :663-672): all three components AND their fu on every row
    //   within `zrows` of the axis. The component set is POST-#3164.
__AXIS_ZERO__

    u0[ii] = n0; u1[ii] = n1; u2[ii] = n2;
    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;
}
"""

#: The prefix substitution per sub-step. B REPLACES Bz's whole curl with the forward
#: difference of the EXTENDED prefix — ONE subtract and ONE multiply, which is a
#: different float32 number from the four-operand grouping. D swaps the SOURCE of
#: Dz's ``first`` operand and leaves the backward-difference machinery unchanged; its
#: row-0 ghost is unreachable for the same reason every other r near ghost is
#: (``Dz``'s r-shift is 0, so the ownership mask zeroes that row).
_PREFIX_BLOCKS: Dict[bool, str] = {
    False: """    {
        // step_B :343-347 — curl = dtdx * (prefix_ext[1:] - prefix_ext[:-1]).
        // pfx is (nr+1, ny, nz); row stride is nyz CELLS.
        float2 pu = pfx[ii + nyz];
        float2 pd = pfx[ii];
        curl2 = c_mul_coefficient_left(dtdx, pu - pd);
    }""",
    True: """    {
        // step_D :418-426 — Dz's `first` source is the prefix (ir0 = 0.5); the
        // backward machinery is otherwise unchanged. pfx is (nr, ny, nz).
        float2 p_here = pfx[ii];
        float2 p_down = vx ? pfx[ox] : float2(0.0f, 0.0f);
        float2 t2c = ((p_down - p_here) + (a - a_y));
        curl2 = c_mul_coefficient_left(dtdx, t2c);
    }""",
}

#: The i*m/r coupling, emitted under ``M_ONE`` and ``M_MANY`` and ABSENT under
#: ``M_ZERO``. Absent rather than multiplied by a zero row: ``step_B`` (:348) and
#: ``step_D`` (:430) branch on ``grid.m != 0`` and never form the term at m = 0, and
#: ``curl - c_mul((0, 0), g)`` would turn a ``-0.0`` curl word into ``+0.0``. The two
#: coefficient pointers stay in the signature for all three arms and are unread here.
_IMR_BLOCK = """    // The partners are the CENTER registers already loaded: target 0 takes `c`
    // (g2 = Ez/Hz), target 2 takes `a` (g0 = Ex/Hx). The coefficient row is
    // HOST-BUILT and bound; the kernel never divides and never takes a max.
    // `curl - (q*g)` carries the array path's `curl + (-(q*g))` by IEEE-754.
    {
        float2 q0 = c0[i];
        float2 q2 = c2[i];
        float2 m0 = c_mul(q0, c);
        curl0 = curl0 - m0;
        float2 m2 = c_mul(q2, a);
        curl2 = curl2 - m2;
    }"""

_NO_IMR_BLOCK = ("    // m = 0: NO i*m/r coupling. stepping :348 and :430 branch on "
                 "grid.m != 0 and never\n    // form the term; the bound rows c0/c2 "
                 "are not read by this body.")

#: The |m| = 1 axis-row increment per sub-step. Absent under ``M_MANY`` and
#: ``M_ZERO`` — each arm is a different body, not the same body at a different count,
#: which is why ``M_CLASS`` stays a compile-time branch where ``zrows`` became a
#: runtime uniform.
_AXIS_INCREMENT_BLOCKS: Dict[bool, str] = {
    False: """    {
        // B side (:643-645): Bx row 0 takes
        //   (-dtdx)*(Ep[0] - Ep[0, z+1]) - (1j*m*dtdx)*Ez[r+1].
        // Ep = g1 center `b` and its z-UP neighbour; Ez[r+1] is the FIRST OFF-AXIS
        // row of g2, read at the row-1 offset of this lane's own column. The
        // predicate refuses nr < 2, which is what keeps that load in bounds.
        float2 d  = b - b_z;
        float2 p  = c_mul_coefficient_left(minus_dtdx, d);
        float2 e1 = g2[nyz + j * nzi + k];
        float2 q  = c_mul(inc_b, e1);
        float2 inc = p - q;
        curl0 = at_x ? -inc : curl0;
    }""",
    True: """    {
        // D side (:555-556): Dy row 0 takes dtdx*(Hr[0] - Hr[0, z-1] - 2.0*Hz[0]).
        // Hr = g0 center `a` and its z-DOWN neighbour; Hz = g2 center `c`. The
        // factor 2 is the doubled-ivec compensation, spelled as the array path
        // spells it: a PYTHON FLOAT on the left of a complex volume.
        float2 two_c = c_mul_coefficient_left(2.0f, c);
        float2 s = (a - a_z) - two_c;
        float2 inc = c_mul_coefficient_left(dtdx, s);
        curl1 = at_x ? -inc : curl1;
    }""",
}

#: ``M_ONE``'s post-recurrence rule, per sub-step: the D side zeroes ``Dz``'s stored
#: axis row and the B side does nothing at all. The B arm is spelled as a COMMENT
#: rather than omitted so a reader sees that the absence was transcribed.
_M_ONE_ZERO_BLOCKS: Dict[bool, str] = {
    False: ("    // |m| = 1, B side: NOTHING. stepping._cylindrical_axis_zero_B "
            "(:648-671)\n    // branches on m == 0 and abs(m) > 1 only."),
    True: """    v2 = at_x ? float2(0.0f, 0.0f) : v2;   // Dz FIELD only, never fu_Dz.""",
}

#: ``M_ZERO``'s post-recurrence rules, per sub-step, in the array path's own order
#: (``_cylindrical_axis_zero_B`` :661-662, ``_cylindrical_axis_zero_D`` :586-587).
#: The D-side increment is a POST-ADD on the stored value and NOT a curl fold — the
#: fold was measured wrong for m = 0 under PML (stepping.py:575-580) and is an armed
#: mutation in the gate. ``b`` is the ``g1`` centre load, the stored Hp. Two
#: roundings, as ``A += s * B`` performs them: the ``python_float_left`` product,
#: then the complex add.
_M_ZERO_ZERO_BLOCKS: Dict[bool, str] = {
    False: """    v0 = at_x ? float2(0.0f, 0.0f) : v0;   // Bx FIELD only (:661-662), never fu_Bx.""",
    True: """    {
        // Dz[r=0] += (4*Courant) * Hp[r=0] (:586): a POST-ADD, not a curl fold.
        float2 inc0 = c_mul_coefficient_left(axis_coef, b);
        v2 = at_x ? (v2 + inc0) : v2;
        v1 = at_x ? float2(0.0f, 0.0f) : v1;   // Dy FIELD only (:587), never fu_Dy.
    }""",
}

#: ``M_MANY``'s post-recurrence rule. ``zrows`` is the RUNTIME UNIFORM — one integer
#: comparison on the row index, licensed by the threshold measurement in the module
#: docstring. The comparison is spelled ``i < int(zrows)`` so both operands are
#: signed; ``i`` is never negative here, so the two spellings cannot differ in a bit
#: (the whole path is integer and exact).
_M_MANY_ZERO_BLOCK = """    {
        bool near = (i < int(zrows));
        v0 = near ? float2(0.0f, 0.0f) : v0;
        v1 = near ? float2(0.0f, 0.0f) : v1;
        v2 = near ? float2(0.0f, 0.0f) : v2;
        n0 = near ? float2(0.0f, 0.0f) : n0;
        n1 = near ? float2(0.0f, 0.0f) : n1;
        n2 = near ? float2(0.0f, 0.0f) : n2;
    }"""


def cylindrical_curl_source(bcz: int, backward: bool, m_arm: int, expansion: str,
                            contract: str = shaders.CONTRACT_OFF) -> str:
    """The specialised ``cyl_complex_pml_curl_step`` source.

    ``bcz`` is the z axis's PERIODIC/METALLIC code (r is always METALLIC and phi
    always PERIODIC, which is why they are not arguments), ``backward`` selects
    ``step_D``'s negated strides over ``step_B``'s forward ones, ``m_arm`` is
    :data:`M_ZERO`, :data:`M_ONE` or :data:`M_MANY`, and ``expansion`` is the
    PROBE-MEASURED complex-multiply arm. All four are baked into the string, as
    Triton bakes its constexprs — a distinct source per specialisation, and therefore
    a distinct compile memo key and a distinct fingerprint.

    ``zero_rows`` is DELIBERATELY NOT AN ARGUMENT: it is a runtime uniform, which is
    what turns ``12 x UNBOUNDED`` into 12. See the module docstring.
    """
    if bcz not in (templates.PERIODIC, templates.METALLIC):
        raise ValueError(f"bcz must be PERIODIC or METALLIC, got {bcz!r}")
    if m_arm not in M_ARMS:
        raise ValueError(f"m_arm must be M_ZERO ({M_ZERO}), M_ONE ({M_ONE}) or "
                         f"M_MANY ({M_MANY}), got {m_arm!r}")
    backward = bool(backward)
    codes = (templates.METALLIC, templates.PERIODIC, int(bcz))
    if m_arm == M_ONE:
        axis_increment = _AXIS_INCREMENT_BLOCKS[backward]
        axis_zero = _M_ONE_ZERO_BLOCKS[backward]
    elif m_arm == M_MANY:
        axis_increment = ("    // |m| >= 2: no axis-row increment (stepping :559, "
                          ":638 return None).")
        axis_zero = _M_MANY_ZERO_BLOCK
    else:
        axis_increment = ("    // m = 0: no axis-row increment (stepping :545-551 "
                          "and :638 return None).")
        axis_zero = _M_ZERO_ZERO_BLOCKS[backward]
    return templates.substitute(_CURL_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__PREFIX__": _PREFIX_BLOCKS[backward],
        "__IMR__": _NO_IMR_BLOCK if m_arm == M_ZERO else _IMR_BLOCK,
        "__MASK__": templates.ownership_mask(codes, backward,
                                             zero=templates.COMPLEX_ZERO),
        "__AXIS_INCREMENT__": axis_increment,
        "__AXIS_ZERO__": axis_zero,
    })


def compile_cylindrical_curl(bcz: int, backward: bool, m_arm: int, expansion: str,
                             contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(cylindrical_curl_source(
        bcz, backward, m_arm, expansion, contract)).cyl_complex_pml_curl_step


def enumerate_cylindrical_sources(expansion: str,
                                  contract: str = shaders.CONTRACT_OFF
                                  ) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit — TWELVE, keyed by a stable label.

    The whole point of the runtime ``zero_rows`` uniform: this enumeration is CLOSED.
    With ``ZERO_ROWS`` as a constexpr it would be ``12 x UNBOUNDED`` and no author
    could write this function at all. Eight until 2026-09-04; the ``M_ZERO`` arm
    added the four ``/m0/`` labels.
    """
    out: Dict[str, str] = {}
    for backward, name in ((False, "step_B"), (True, "step_D")):
        for bcz in (templates.PERIODIC, templates.METALLIC):
            for m_arm in M_ARMS:
                arm_name = M_ARM_LABELS[m_arm]
                label = (f"cyl_complex_pml_curl_step/{name}/bcz{bcz}/{arm_name}/"
                         f"{expansion}")
                out[label] = cylindrical_curl_source(bcz, backward, m_arm,
                                                     expansion, contract)
    return out


# ---------------------------------------------------------------------------
# The expansion probe — how the arm is bound
# ---------------------------------------------------------------------------

def load_expansion_probe(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Read THIS family's probe artifact JSON, or None when absent or unreadable.

    Unreadable is treated exactly like missing: both are refusals downstream, never
    a silent default arm. The environment variable is this family's own —
    :data:`PROBE_PATH_ENVIRONMENT` — because an artifact that classified five
    orientations may not licence a kernel that performs seven.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    candidate = path if path is not None else os.environ.get(PROBE_PATH_ENVIRONMENT)
    if not candidate:
        return None
    try:
        with open(candidate, "r", encoding="utf-8") as handle:
            record = json.load(handle)
    except Exception:  # noqa: BLE001 - unreadable probe == missing probe
        return None
    return record if isinstance(record, dict) else None


def expansion_from_probe(record: Any) -> Optional[str]:
    """The single expansion arm a probe record licenses, or None.

    ``complex_fields.expansion_from_probe``'s contract over the SUPERSET pattern
    list: the record must name the backend the ENGINE holds, classify EVERY pattern
    of :data:`CYLINDRICAL_PROBE_PATTERNS` — the base five plus this family's two —
    and leave exactly one arm standing once ``AMBIGUOUS_BOTH`` is removed.

    A record carrying only ``complex_fields``' five licenses NOTHING here, and that
    is the point of the separate environment variable: the two new orientations are
    calls this kernel makes and no earlier artifact measured.
    """
    if not isinstance(record, dict) or record.get("backend") != PROBE_BACKEND:
        return None
    patterns = record.get("patterns")
    if not isinstance(patterns, dict):
        return None
    decided: List[str] = []
    for name in CYLINDRICAL_PROBE_PATTERNS:
        value = patterns.get(name)
        if value == complex_fields.AMBIGUOUS_BOTH:
            continue
        if value not in templates.EXPANSIONS:
            return None
        decided.append(value)
    if len(set(decided)) != 1:
        return None
    return decided[0]


def _expansion_reasons(probe: Any = None) -> List[str]:
    """The clause that makes the arm binding a MEASUREMENT rather than a choice."""
    record = probe if probe is not None else load_expansion_probe()
    if record is None:
        return [f"no cylindrical-complex expansion probe artifact is available (set "
                f"{PROBE_PATH_ENVIRONMENT} or pass probe=); which arm the "
                f"{PROBE_BACKEND} reference takes on this family's SEVEN "
                f"orientations is a measured platform fact and may not be guessed"]
    if expansion_from_probe(record) is None:
        return [f"the expansion probe artifact is missing, ambiguous, disagreeing, "
                f"or not for the {PROBE_BACKEND} backend (it needs "
                f"backend={PROBE_BACKEND!r}, every pattern of "
                f"{CYLINDRICAL_PROBE_PATTERNS} classified — the base five PLUS "
                f"{IMR_ROW_PROBE_PATTERN!r} and {AXIS_SCALAR_PROBE_PATTERN!r} — and "
                f"exactly one arm standing among the discriminating ones)"]
    return []


# ---------------------------------------------------------------------------
# Coverage — positive refusal enumeration
# ---------------------------------------------------------------------------
#
# THESE PREDICATES CALL NEITHER SHARED CLAUSE LIST, and that is deliberate rather
# than lazy: `coverage._grid_reasons` clauses 2, 6 and 7 refuse this whole domain,
# and `complex_fields._complex_grid_reasons` clause 4 refuses cylindrical outright.
# Both are RIGHT for their own products and must stay — each is now load-bearing in
# two directions at once. Their non-cylindrical clauses are restated below,
# individually, with the cylindrical ones replaced by POSITIVE Dcyl requirements.

def _shared_reasons(fields: Any, pml: Any, grid: Any, residency: Any,
                    probe: Any) -> List[str]:
    """The clauses both predicates share, cylindrical geometry excluded."""
    reasons: List[str] = []

    # 1. The Metal backend, the host array module the mirrors copy from, and the
    #    resolved subnormal policy. IMPORTED, not re-spelled.
    reasons.extend(_metal_backend_reasons(grid))
    reasons.extend(_residency_declaration_reasons(residency))

    # 2 (INVERTED against coverage.py clause 2, AND THE ONE CLAUSE THAT PARTITIONS
    #    Dcyl BETWEEN THIS FAMILY AND THE REAL ONE). Complex64 storage REQUIRED; the
    #    real family refuses it by name in the other direction. At |m| >= 1 the
    #    array path REFUSES real storage outright (stepping.py:731-736, MEEP's
    #    change_m aborts on the same combination), so a real Dcyl run is m = 0 and
    #    belongs to the cylindrical REAL product; a complex64 run at ANY m belongs
    #    here (m = 0 since 2026-09-04). The DECLARATION is checked here; the dtype is
    #    verified by the layout clause.
    #
    #    THE NARROWER SPELLING IS CORRECT HERE. `complex_fields` admits
    #    `force_complex_fields OR has_bloch`; this family refuses a k_point outright
    #    (clause 7), so `has_bloch` can never be the thing that puts a run here and
    #    including it would admit a configuration clause 7 then refuses — two
    #    clauses disagreeing about the same run.
    if not getattr(fields, "force_complex_fields", False):
        reasons.append(
            "force_complex_fields is not set: this family carries complex64 storage "
            "only. A real-storage Dcyl run is m = 0 (|m| >= 1 has no real-storage "
            "form — stepping.py:731-736 raises, MEEP's change_m aborts on the same "
            "combination) and belongs to the cylindrical REAL product's kernel")

    # 3. An absorber that actually absorbs: this is the split-field product only.
    if pml is None or not getattr(pml, "is_active", False):
        reasons.append("no active PML layer (this product implements the complex "
                       "cylindrical split-field path only)")

    # 4. No conductivity on any curl target. Refused OUTRIGHT on a missing or
    #    non-callable reader — inferring "no conductivity" from the ABSENCE of
    #    condfac_for is admission by attribute absence.
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

    # 5. No dispersion, refused OUTRIGHT: complex-storage ADE is unbuilt on either
    #    geometry. The shape clauses still run so an unreadable susceptibility is
    #    NAMED rather than shrugged at.
    if getattr(fields, "has_polarizations", False) or (
            getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: complex-storage ADE is a "
                       "later tranche, and no cylindrical corpus row combines the two")
    reasons.extend(_susceptibility_reasons(fields))

    # 10/11/12. No instantaneous nonlinearity (the Pade factor REPLACES the
    #    constitutive product, S:970-971), no BFAST, no special_kz beta. `Grid`
    #    already refuses the last two on a cylindrical cell (grid.py:689-696 beta,
    #    :734-741 BFAST) — the clauses are written because "another module already
    #    guards it" is exactly the reasoning this file exists to refuse.
    if getattr(fields, "has_nonlinearity", False):
        reasons.append("chi2/chi3 is installed (the Pade constitutive factor is "
                       "not carried)")
    if getattr(grid, "bfast_active", False):
        reasons.append("BFAST is active (the second additive curl term is not carried)")
    if float(getattr(grid, "beta", 0.0)) != 0.0:
        reasons.append(f"special_kz beta={getattr(grid, 'beta', 0.0)!r} is nonzero "
                       f"(that is grid.beta, not a Bloch phase)")

    # 7 (KEPT UN-INVERTED). NO BLOCH PHASE. Dcyl Bloch is z-only in MEEP
    #    (grid.py:654-658 refuses a k on r or phi), none of the sixteen demand rows
    #    carries one, and the kernel compiles no rotation at all. Admitting a z
    #    phase later is a predicate widening plus a gate row — not a silent
    #    extension, and not something a reader should have to infer from an absent
    #    `__PHASE_Z__` block.
    if getattr(grid, "has_bloch", False):
        reasons.append(
            f"nonzero k_point {getattr(grid, 'k_point', None)!r}: this kernel "
            f"compiles no Bloch rotation (a z-only Dcyl phase is a future widening)")
    k_point = tuple(getattr(grid, "k_point", (0.0, 0.0, 0.0)))
    if any(float(component) != 0.0 for component in k_point):
        reasons.append(f"k_point {k_point!r} is not exactly zero")

    # 9. Stored E. Under an active PML always true (fields.py:678-700), but the
    #    clause is the REASON: an edit that makes stores_E optional under PML must
    #    be caught here.
    if not getattr(fields, "stores_E", False):
        reasons.append("E is recomputed from D rather than stored")

    # 13. The expansion arm must come from a measured artifact carrying THIS
    #     family's seven orientations.
    reasons.extend(_expansion_reasons(probe))

    return reasons


def _cylindrical_geometry_reasons(fields: Any, pml: Any, grid: Any) -> List[str]:
    """The POSITIVE Dcyl requirements — clauses 6, 14 (readability), 4 and 5."""
    reasons: List[str] = []

    # 6 (INVERTED). Cylindrical REQUIRED, where every carried family refuses it.
    if not getattr(grid, "cylindrical", False):
        reasons.append("grid is not cylindrical (this product steps Dcyl grids only)")
    flags = tuple(bool(_call(grid, "is_axis", axis, default=False))
                  for axis in range(3))
    if flags != CYLINDRICAL_AXIS_FLAGS:
        reasons.append(f"grid.is_axis {flags!r} is not {CYLINDRICAL_AXIS_FLAGS!r} "
                       f"(r must be the LEADING axis; the whole kernel indexes it "
                       f"as axis 0)")

    # 14 (READABILITY). `grid.m` must be an INTEGER: the compile-time arm
    #    (`m_class`) and the near-axis uniform (`zero_rows`) are functions of it, and
    #    an unreadable m would bind the wrong body as a smooth wrong field. NO VALUE
    #    OF m IS REFUSED HERE since 2026-09-04: m = 0 is the `M_ZERO` arm (module
    #    docstring, addition 5). Until then this clause refused m = 0 by name as "the
    #    cylindrical REAL product's kernel", which was wrong about the partition —
    #    the real product refuses complex64 storage (its clause 2), so a complex64
    #    m = 0 run was refused by BOTH families and fell to the array path. The
    #    partition between the two cylindrical families is clause 2 (storage), which
    #    is inverted in both directions and needs no second clause.
    m = getattr(grid, "m", None)
    if m is None:
        reasons.append("grid does not report m")
    else:
        try:
            int(m)
        except Exception:  # noqa: BLE001 - an unreadable m is not coverage
            reasons.append(f"grid.m = {m!r} is not an integer")

    # 4 (WIDENED PER AXIS, never globally). 'axis' on r only; phi PERIODIC; z either.
    kinds = _boundary_kinds(
        grid, pml if (pml is not None and getattr(pml, "is_active", False)) else None)
    if kinds is None:
        reasons.append("boundary kinds could not be resolved for this grid")
    else:
        if kinds[0] != REQUIRED_R_KIND:
            reasons.append(f"axis 0 boundary {kinds[0]!r} is not {REQUIRED_R_KIND!r}")
        if kinds[1] != REQUIRED_PHI_KIND:
            reasons.append(f"axis 1 boundary {kinds[1]!r} is not "
                           f"{REQUIRED_PHI_KIND!r}")
        if kinds[2] not in ADMITTED_Z_KINDS:
            reasons.append(f"axis 2 boundary {kinds[2]!r} is outside "
                           f"{ADMITTED_Z_KINDS!r}")

    # 5 (REFUSED TWICE, and the second half is the load-bearing one). The grid FLAG
    #    is not the inversion: `_boundary_kinds` puts is_axis AHEAD of is_mirrored
    #    (stepping.py:2191-2195), so a folded r axis reports CYL_AXIS and THE FOLD
    #    VANISHES SILENTLY; and `_mirror_phases` (:2346-2363) puts (-1)**grid.m into
    #    the SAME SLOT the mirror phase occupies.
    if _call(grid, "has_symmetry", default=False):
        reasons.append("a mirror plane is active (symmetry folding is not carried; "
                       "on a Dcyl grid the r axis reports 'axis' ahead of 'mirror' "
                       "and the fold would vanish silently)")
    for axis in range(3):
        if _call(grid, "is_mirrored", axis, default=False):
            reasons.append(f"axis {axis} is folded by a mirror plane")

    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3:
        reasons.append(f"grid shape {shape!r} is not three-dimensional")
    else:
        if int(shape[1]) != 1:
            reasons.append(f"phi extent is {shape[1]} cells, not 1 (Dcyl is 2.5-D: "
                           f"the exp(i*m*phi) dependence is analytic)")
        if int(shape[0]) < MINIMUM_RADIAL_ROWS:
            reasons.append(
                f"radial extent is {shape[0]} cell(s), fewer than "
                f"{MINIMUM_RADIAL_ROWS}: the |m| = 1 axis increment reads the FIRST "
                f"OFF-AXIS row (stepping.py:671, where NumPy raises IndexError and "
                f"CuPy silently returns row 0 — both measured) and the kernel's "
                f"matching load would run past the end of the source volume")

    # `accurate_fields_near_cylorigin` is READ, never assumed: it selects the
    # |m| >= 2 near-axis row count (stepping._cylindrical_axis_rows, :601-622) and a
    # grid that cannot answer would bind the wrong uniform — a plane of wrong
    # values, not a crash.
    if getattr(grid, "accurate_fields_near_cylorigin", None) is None:
        reasons.append("grid does not report accurate_fields_near_cylorigin; it "
                       "selects the |m| >= 2 near-axis row count and may not be "
                       "assumed")

    # A configuration the ARRAY PATH itself refuses is not one a kernel may step.
    try:
        from ..stepping import _require_cylindrical_steppable  # noqa: PLC0415

        _require_cylindrical_steppable(grid)
    except Exception as exc:  # noqa: BLE001 - an unsteppable grid is refused
        reasons.append(f"stepping refuses this cylindrical grid: {exc!r}")

    return reasons


def cylindrical_complex_pml_curl_coverage(fields: Any, pml: Any, sub_step: str,
                                          residency: Any = None,
                                          probe: Any = None) -> Coverage:
    """May the complex cylindrical curl kernel step this (fields, pml, sub_step)?

    Every clause names its own requirement and appends its own reason; the scan
    continues after a failure so a refusal reports EVERYTHING that disqualified the
    run rather than the first thing.

    An off-diagonal chi1inv row is ADMITTED here — constitutive-only, its whole
    effect is inside ``update_E`` (stepping.py:1001-1008) — and refused by the E-side
    constitutive predicate. The same per-sub-step split ``coverage.py`` makes.
    """
    if sub_step not in CURL_SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(CURL_SUB_STEPS)}, "
                         f"got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    reasons = _shared_reasons(fields, pml, grid, residency, probe)
    reasons.extend(_cylindrical_geometry_reasons(fields, pml, grid))

    spec = SUB_STEPS[sub_step]
    names = (tuple(spec["targets"])
             + tuple("fu_" + name for name in spec["targets"])
             + CURL_SOURCES)
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(complex_fields._complex_layout_reasons(fields, shape, names))
    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        # BOTH sub-lattices: the suffix the plan binds is the sub-step's own, and a
        # swap is a silent half-cell error in the absorber profile.
        reasons.extend(_coefficient_reasons(pml, shape, ("kms", "sinv"), ("", "_h")))

    return Coverage(not reasons, tuple(reasons))


def cylindrical_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                              residency: Any = None,
                                              probe: Any = None) -> Coverage:
    """May the CERTIFIED ``complex_fields.bloch_constitutive_step`` run here?

    NO NEW KERNEL AND NO NEW ARITHMETIC: only the ADMISSION is this family's.
    ``complex_fields.complex_constitutive_coverage`` correctly refuses a cylindrical
    grid, because that product's CURL cannot step one; this verdict proves the Dcyl
    geometry and the coefficient lengths explicitly instead of weakening that family
    globally.

    THE GEOMETRY REACHES THIS SUB-STEP THROUGH NOTHING AT ALL, and that is the
    measured finding this family rests half its slots on: ``stepping.update_H``
    (:907-924) and the diagonal ``update_E`` (:983-991) are three
    ``_apply_constitutive_pml`` calls over per-axis coefficient tables indexed on the
    component's OWN axis — no shift helper, no ghost, no ownership mask, no m. The
    identity leg quoted in the module docstring measured 480/480 rows and 0 differing
    uint32 words against a reference carrying no cylindrical clause.

    The E side refuses what changes what ``source`` is, each naming its owner: a
    registered polarization (already refused module-wide), an off-diagonal chi1inv
    row (the row product reads neighbours, so the sub-step stops being element-wise),
    and ``stores_E`` false.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))

    spec = CONSTITUTIVE_SIDES[side]
    reasons = _shared_reasons(fields, pml, grid, residency, probe)
    reasons.extend(_cylindrical_geometry_reasons(fields, pml, grid))

    if side == "E" and getattr(fields, "has_offdiagonal_epsilon", False):
        reasons.append("an off-diagonal chi1inv row is installed (the row product "
                       "reads neighbours; this sub-step is element-wise). A "
                       "cylindrical off-diagonal update_E is not carried on this "
                       "backend")

    names = tuple(spec["targets"]) + tuple(spec["aux"]) + tuple(spec["sources"])
    for name in names:
        if getattr(fields, name, None) is None:
            reasons.append(f"{name} is not allocated")

    shape = tuple(getattr(grid, "shape", ()))
    reasons.extend(complex_fields._complex_layout_reasons(fields, shape, names))

    if side == "E" and len(shape) == 3:
        # inv_eps stays FLOAT32 under complex storage (stepping.py:41-50,
        # fields.py:1203-1204), so the shipped float32 pin is exactly right.
        reasons.extend(_inverse_epsilon_reasons(fields, shape))

    if pml is not None and getattr(pml, "is_active", False) and len(shape) == 3:
        suffix = ("_h",) if spec["half_integer"] else ("",)
        reasons.extend(_coefficient_reasons(pml, shape, ("kps", "kms"), suffix))

    return Coverage(not reasons, tuple(reasons))


# ---------------------------------------------------------------------------
# The plans
# ---------------------------------------------------------------------------

class CylindricalComplexCurlPlan(KernelPlan):
    """A launchable COMPLEX cylindrical curl sub-step: array-path prefix, ONE kernel.

    THE ONE DIVERGENCE FROM :class:`.plans.KernelPlan`, STATED WHERE IT HAPPENS.
    That base's contract is that ``run`` unpacks one argument tuple and calls one
    function, and this plan HONOURS the launch itself — :meth:`run` calls
    ``super().run``, so the dict lookup, the variant refusal and the launch counter
    are the base's, unchanged. What it ADDS is a pre-pass, because the radial prefix
    is a SEQUENTIAL HOST SCAN whose float32 summation order defines the answer and
    which therefore cannot move to the device (module docstring, addition 1).

    So one launch costs, in this order:

    1. ``sync_out`` of the ONE prefix source volume (``Ey`` on B, ``Hy`` on D) —
       because on this backend the device mirror is authoritative between launches
       and the host copy is stale;
    2. the scan, on the host, through ``stepping.cylindrical_rderiv_prefix``;
    3. ``sync_in`` of the prefix mirror;
    4. the base's launch.

    THE TWO SYNCS ARE PERFORMED HERE AND NOT BY THE CALLER, deliberately. A
    ``prepare()`` a caller had to remember is exactly the stale-mirror class this
    package refuses to comment on: a forgotten call is a smooth, plausible, wrong
    field. :attr:`prefix_syncs` counts them so a gate can assert they happened rather
    than trust that they did — a leg that certified this plan without the counter
    could pass on a prefix computed once and never refreshed.

    THE COST IS REAL AND IS NOT HIDDEN: the certified families' benchmark does not
    describe this one, and this family's throughput must be measured separately.

    The two i*m/r coefficient rows and the axis-increment scalars are GRID
    INVARIANTS — Yee shift, radial extent, m, Courant — so they are built ONCE at
    plan time, exactly as ``stepping``'s own ``scratch.constant`` cache builds them
    once per run (:713-723), and never rebuilt per launch.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bcz", "m_arm",
                 "zero_rows", "axis_coef", "expansion", "residency", "volumes",
                 "scratch", "prefix_syncs", "_prefix_host", "_prefix_name",
                 "_prefix_source_name", "_source_map", "_xp")

    REPR_FIELDS = ("sub_step", "shape", "bcz", "m_arm", "zero_rows", "axis_coef",
                   "expansion")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, sub_step: str, shape, dtdx: float, bcz: int, m_arm: int,
                 zero_row_count: int, expansion: str, residency: Any, targets,
                 auxiliaries, sources, prefix, imr_rows, coefficients,
                 increment_scalars, functions: Dict[str, Any],
                 volumes: Sequence[str], prefix_host: Any, prefix_name: str,
                 prefix_source_name: str, source_map: Dict[str, Any],
                 xp: Any, scratch: Any = None) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64 precision
        # and Metal binds a Python float into `constant float&` correctly rounded.
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bcz = int(bcz)
        self.m_arm = int(m_arm)
        self.zero_rows = int(zero_row_count)
        # The m = 0 on-axis Dz coefficient, HOST-FORMED as the array path forms it
        # (`axis_coefficient`) and bound once; read by the M_ZERO D body only.
        self.axis_coef = axis_coefficient(dtdx)
        self.expansion = str(expansion)
        self.residency = residency
        self.volumes = tuple(volumes)
        self.scratch = scratch
        self.prefix_syncs = 0
        self._prefix_host = prefix_host
        self._prefix_name = str(prefix_name)
        self._prefix_source_name = str(prefix_source_name)
        self._source_map = dict(source_map)
        self._xp = xp
        minus_dtdx, inc_b = increment_scalars
        super().__init__(functions, (
            tuple(targets) + tuple(auxiliaries) + tuple(sources) + (prefix,)
            + tuple(imr_rows) + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem, self.dtdx,
               self.zero_rows, float(minus_dtdx),
               (float(inc_b[0]), float(inc_b[1])), self.axis_coef)))

    def refresh_prefix(self) -> None:
        """Pull the prefix SOURCE off the device, scan it on the host, push it back.

        Separated from :meth:`run` so a gate can call it explicitly and count it, and
        so the residency traffic this family costs has a name. It is NOT an optional
        step: :meth:`run` always calls it.
        """
        self.residency.sync_out((self._prefix_source_name,))
        prefix = cylindrical_prefix(self._xp, self.sub_step, self._source_map,
                                    scratch=self.scratch)
        self._prefix_host[...] = prefix
        self.residency.sync_in((self._prefix_name,))
        self.prefix_syncs += 1

    def run(self, contract: Optional[str] = None) -> None:
        """Refresh the prefix, then take the BASE's launch path unchanged."""
        self.refresh_prefix()
        super().run(contract)


def _curl_functions(bcz: int, backward: bool, m_arm: int, expansion: str,
                    contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_cylindrical_curl(bcz, backward, m_arm, expansion, mode)
            for mode in contract_variants}


def _build_curl_plan(sub_step: str, shape, dtdx: float, m: int, accurate: bool,
                     bcz: int, expansion: str, residency: Any, arrays: Dict[str, Any],
                     flat: Dict[str, Any], coefficient_names: Sequence[str],
                     functions: Optional[Dict[str, Any]],
                     contract_variants: Sequence[str], scratch: Any,
                     ) -> CylindricalComplexCurlPlan:
    """The ONE plan assembly both routes take.

    The engine route and the gate's bare-array route differ ONLY in where the arrays
    come from and how the coefficient mirrors are named. Everything that decides a
    BIT — the m class, the zero-row count, the i*m/r rows, the increment scalars,
    the prefix buffer and the argument order — is assembled here once, so the bytes
    the gate certifies are the bytes the engine would launch.
    """
    import numpy  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in shape)
    arm = m_class(m)
    rows = shape[0]
    mirror = complex_fields._complex_mirror

    targets = [mirror(residency, name, arrays[name]) for name in spec["targets"]]
    auxiliaries = [mirror(residency, "fu_" + name, arrays["fu_" + name])
                   for name in spec["targets"]]
    sources = [mirror(residency, name, arrays[name]) for name in spec["sources"]]

    # THE PREFIX BUFFER IS THE PLAN'S OWN and is allocated once. It must be a stable
    # host array: `Residency.mirror` refuses re-registering a name against a
    # DIFFERENT host array, and `cylindrical_rderiv_prefix` returns a fresh array
    # every call. So the scan's result is copied INTO this buffer rather than the
    # buffer being replaced, which is also what keeps the device allocation stable.
    prefix_host = numpy.zeros(prefix_shape(sub_step, shape), dtype=numpy.complex64)
    prefix_name = f"cyl:pfx:{sub_step}"
    prefix = mirror(residency, prefix_name, prefix_host)

    imr_rows = []
    for index, _register, sign in IMR_TERMS[sub_step]:
        target = spec["targets"][index]
        row = numpy.ascontiguousarray(
            imr_coefficient_row(numpy, target, sign, m, dtdx, rows,
                                numpy.complex64).reshape(-1))
        imr_rows.append(mirror(residency, f"cyl:imr:{sub_step}:{target}", row))

    coefficients = [residency.mirror(name, flat[name], constant=True)
                    for name in coefficient_names]

    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    source_map = {name: arrays[name] for name in spec["sources"]}
    return CylindricalComplexCurlPlan(
        sub_step, shape, dtdx, bcz, arm, zero_rows(m, accurate), expansion,
        residency, targets, auxiliaries, sources, prefix, imr_rows, coefficients,
        axis_increment_scalars(m, dtdx),
        functions if functions is not None
        else _curl_functions(bcz, spec["backward"], arm, expansion,
                             contract_variants),
        volumes, prefix_host, prefix_name, PREFIX[sub_step]["component"],
        source_map, numpy, scratch)


def plan_cylindrical_complex_pml_curl(
        fields: Any, pml: Any, sub_step: str, residency: Any = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        probe: Any = None) -> Optional[CylindricalComplexCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    None is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not cylindrical_complex_pml_curl_coverage(
            fields, pml, sub_step, residency, probe).covered:
        return None
    expansion = expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[sub_step]
    grid = fields.grid
    kinds = resolve(grid, pml)
    bcz = templates.METALLIC if kinds[2] == "metallic" else templates.PERIODIC
    names = (tuple(spec["targets"])
             + tuple("fu_" + n for n in spec["targets"])
             + tuple(spec["sources"]))
    arrays = {name: getattr(fields, name) for name in names}
    coefficient_names = [f"pml:{stem}_{axis}{spec['suffix']}"
                         for axis in "xyz" for stem in ("kms", "sinv")]
    flat = {name: getattr(pml, name.split(":", 1)[1]) for name in coefficient_names}
    return _build_curl_plan(
        sub_step, grid.shape, grid.dt / grid.dx, int(grid.m),
        bool(grid.accurate_fields_near_cylorigin), bcz, expansion, residency,
        arrays, flat, coefficient_names, None, contract_variants,
        getattr(fields, "scratch", None))


def plan_cylindrical_complex_pml_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any], bcz: int,
        m: int, accurate_fields_near_cylorigin: bool, dtdx: float, expansion: str,
        residency: Any, functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        scratch: Any = None) -> CylindricalComplexCurlPlan:
    """Build a plan from bare host arrays — the gate's and benchmark's route.

    No predicate runs: the caller is a harness that constructed the configuration
    deliberately, including the deliberately wrong ones.

    ``functions`` is the MUTATION SEAM. The gate compiles a deliberately broken copy
    of the shipped source and hands it here; dropping the argument is not a silent
    slowdown, it is a silent DISARMING — every mutation leg would then launch the
    shipped kernel and report the defect as uncaught.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    coefficient_names = [f"pml:{stem}_{axis}:{sub_step}"
                         for axis in "xyz" for stem in ("kms", "sinv")]
    resolved = {name: flat[name.split(":")[1]] for name in coefficient_names}
    return _build_curl_plan(
        sub_step, shape, dtdx, m, bool(accurate_fields_near_cylorigin), bcz,
        expansion, residency, arrays, resolved, coefficient_names, functions,
        contract_variants, scratch)


def plan_cylindrical_complex_constitutive(
        fields: Any, pml: Any, side: str, residency: Any = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        probe: Any = None) -> Any:
    """A CERTIFIED complex constitutive plan for a Dcyl run, or None.

    The arithmetic, the source and the plan class are ``complex_fields``' own; only
    the ADMISSION is this family's. Building the plan through the certified pieces —
    rather than re-deriving one here — is what makes "same kernel" a fact instead of
    a claim: a divergence would have to come from the predicate, which is the only
    thing this family contributes on these two slots. The same move
    :func:`folded_complex.plan_folded_complex_constitutive` makes.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not cylindrical_complex_constitutive_coverage(
            fields, pml, side, residency, probe).covered:
        return None
    expansion = expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
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
# WIRING — four arms over four slots
# ---------------------------------------------------------------------------
#
# WHAT SEPARATES THEM, per slot, and every one is a clause that NAMES the other
# family rather than an omission:
#
#   step_B / step_D    every carried family refuses cylindrical through clause 6 —
#                      `coverage._grid_reasons` (:176-182) for `pml_curl`,
#                      `bfast_curl`, `offdiag_update_e` and `special_kz`'s real arm;
#                      `complex_fields._complex_grid_reasons` for the unfolded
#                      complex family; `folded_complex` and `symmetry` restate it
#                      twice each for the is_axis-outranks-is_mirrored reason.
#                      Clause 6 HERE is INVERTED and clause 2 requires complex64,
#                      which is also what separates this family from the cylindrical
#                      REAL one (that family refuses complex64 by name). Clause 14
#                      refused m = 0 until 2026-09-04; the M_ZERO arm carries it now.
#   update_H/update_E  the same separations, plus `no_pml_constitutive` inverting on
#                      the ABSORBER (it requires an INACTIVE one and this family an
#                      active one) and `offdiag_update_e` requiring a LIVE
#                      off-diagonal row under REAL storage, which this family
#                      refuses on the E side and cannot supply on either.
#
# THE ARMS ARE GATED on cylindrical + complex storage, for `folded_complex`'s
# reason: on a Cartesian run five other families speak on these four slots, so a
# silent cylindrical arm costs a reader nothing, while a consulted-and-refusing one
# would add nine reasons to every Cartesian refusal message.


def _is_cylindrical_complex(context: Any) -> bool:
    """The cheap gate that decides whether this family is CONSULTED at all.

    Deliberately the cheapest possible read of the grid and never the predicate
    itself. It asks about the GEOMETRY and the STORAGE and nothing else, so an arm
    that IS consulted stays free to say "no active PML layer" or "a conductivity is
    installed" — the sentences that make the inversions visible.
    """
    fields = getattr(context, "fields", None)
    grid = getattr(fields, "grid", None)
    if grid is None or not getattr(grid, "cylindrical", False):
        return False
    return bool(getattr(fields, "force_complex_fields", False))


def _curl_arm_coverage(context: Any, slot: str) -> Coverage:
    return cylindrical_complex_pml_curl_coverage(
        context.fields, context.pml, slot, context.residency,
        context.extra.get("cylindrical_complex_probe"))


def _curl_arm_plan(context: Any, slot: str) -> Any:
    return plan_cylindrical_complex_pml_curl(
        context.fields, context.pml, slot, context.residency,
        context.contract_variants,
        context.extra.get("cylindrical_complex_probe"))


def _constitutive_arm_coverage(context: Any, slot: str) -> Coverage:
    return cylindrical_complex_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.extra.get("cylindrical_complex_probe"))


def _constitutive_arm_plan(context: Any, slot: str) -> Any:
    return plan_cylindrical_complex_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants,
        context.extra.get("cylindrical_complex_probe"))


def register_arms() -> Tuple[Any, ...]:
    """This family's four arms: two cylindrical curls, the certified complex pair.

    WIRED. The two curl arms carry a NEW kernel; the two constitutive arms carry the
    CERTIFIED ``complex_fields`` body under a restated predicate, which is where 32
    of this family's 64 slots come from at no arithmetic cost.

    REGISTERED-BUT-CANNOT-WIN IS NOT USED, and the absence is stated rather than
    left to a reader. ``registry.py`` keeps ``registered`` and ``wins`` separate so a
    missing capability gets NAMED at composition time instead of falling silently to
    the array path, and this family would have used it if a slot's product were
    unported. Every slot it claims IS ported. The capabilities it LACKS — real
    float32 storage (the cylindrical REAL family's), a cylindrical off-diagonal
    ``update_E``, a cylindrical ADE — are refused by name inside predicates that
    would otherwise admit, which is the same naming through a cheaper mechanism.

    ``update_P`` IS NOT TOUCHED. This family carries no pole recurrence, so
    ``metal_composition_matrix.UNREGISTERED_SLOTS`` and
    ``test_no_arm_is_registered_on_a_slot_no_product_carries`` stand unchanged and
    are NOT retired here.
    """
    from . import arms  # noqa: PLC0415 - deferred: `arms` imports nothing of ours

    registered = [
        arms.register(family=FAMILY, slot=slot, label="cylindrical complex",
                      coverage=_curl_arm_coverage, plan=_curl_arm_plan,
                      prefix="cylindrical complex: ",
                      noun="cylindrical complex PML curl",
                      gate=_is_cylindrical_complex, wired=True)
        for slot in ("step_B", "step_D")]
    registered.extend(
        arms.register(family=FAMILY, slot=slot, label="cylindrical complex",
                      coverage=_constitutive_arm_coverage,
                      plan=_constitutive_arm_plan,
                      prefix="cylindrical complex: ",
                      noun="cylindrical complex constitutive",
                      gate=_is_cylindrical_complex, wired=True)
        for slot in _CONSTITUTIVE_SLOT_SIDES)
    return tuple(registered)


#: Registered ON IMPORT, once — the registry refuses a duplicate by design.
ARMS: Tuple[Any, ...] = register_arms()
