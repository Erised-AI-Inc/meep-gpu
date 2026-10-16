"""COMPLEX cylindrical (Dcyl, complex64 storage at every m) PML curl on Metal — ONE
new kernel, three compile-time bodies (m = 0, |m| = 1, |m| >= 2).

TEMPLATE FOLLOWED: :mod:`.complex_fields`, end to end and deliberately. Its
``float2`` volume binding, its ``_CURL_TEMPLATE`` shape, its part-wise complex
helpers, its ``ComplexPmlCurlPlan`` argument layout, its predicate spine
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

1. **THE RADIAL DIFFERENCE STAYS ON THE ARRAY PATH.** MEEP sums ``f_rderiv_int``
   in every structure chunk from that chunk's first row, ``rinv`` multiplied, and
   the stencil reads the difference of two partial sums of the owning chunk
   (``stepping.cylindrical_rderiv_difference``). Its float32 summation order DEFINES
   the answer, and this module CALLS it (:func:`radial_difference`) with the run's
   chunk starts rather than re-deriving it. B side: Ep = ``Ey`` extended by ONE
   ZERO WALL ROW, ``ir0 = 0.0``, forward; D side: Hp = ``Hy``, ``ir0 = 0.5``,
   backward. Target 2's WHOLE curl is ``dtdx`` times that difference, one product
   per part; the phi partner is absent on a Dcyl grid.

   **AND THIS IS THE ONE PLACE THIS BACKEND PAYS A HOST ROUND TRIP.** The engine
   holds NumPy, so the sum is formed on the HOST and the plan must, per launch, sync
   its SOURCE volume OUT, sum it, and sync the difference IN — one component out and
   one volume in per curl sub-step, against a residency layer that exists because a
   full round trip costs 12x-64x the kernel. **This family's throughput claim must be
   measured separately and may not be inherited from the certified families'
   benchmark.** :meth:`CylindricalComplexCurlPlan.run` performs both syncs so no
   caller can forget one, and counts them (``prefix_syncs``) so a gate can assert
   they happened.

2. **THE i*m/r PASS** (``stepping._cylindrical_imr_increment`` and
   ``_apply_folded_increment``), after the forms and before the stores. The
   increment is ``quarter_turn(rinv, partner)`` — per part ``(-(rinv*g.im),
   rinv*g.re)`` — with ``rinv`` the float32 quotient of the double ``sign*2*m*
   Courant`` word by ``max(2*ir + iyee_r, 1)``; it is masked to +0 on the cells
   MEEP's owned loop skips and SUBTRACTED through MEEP's additive ladder ``df =
   sinv_dsig * inc``: outside the dsigu chunks ``f -= df``, inside ``fu -= df`` and
   ``f -= sinv_dsigu * df``, per part. Four call sites — ``Bx <- Ez`` (+1),
   ``Bz <- Ex`` (-1), ``Dx <- Hz`` (-1), ``Dz <- Hx`` (+1); the partners are centre
   loads the certified body already makes. **THE rinv ROW IS HOST-BUILT AND BOUND**
   (:func:`imr_rinv_row`): Metal's ``max`` disagrees with ``numpy.maximum`` on two
   zeros and ``fast::divide`` is measured-divergent here, so the kernel never divides
   and never takes a max.

3. **THE |m| = 1 AXIS-ROW INCREMENTS**, which REPLACE curl row 0 AFTER the ownership
   mask. B side: ``-(axis_factor * ((Ep[0] - Ep[z+1]) - (m*Ez1.im, -(m*Ez1.re))))``
   on ``Bx``; D side: ``-(axis_factor * ((Hr[0] - Hr[z-1]) - 2*Hz[0]))`` on ``Dy``,
   each part a float32 product with the word of MEEP's double ``-Courant`` /
   ``+Courant`` factor. A REPLACEMENT, never an accumulation: the masked row is
   exactly ``+0.0``, so an accumulating spelling is a signed-zero defect a random
   seed misses.

4. **THE PER-|m| AXIS RULES**, applied to the STORED value after the forms. |m| = 1:
   the D side zeroes ``Dz`` on the axis row — the FIELD ONLY, never ``fu_Dz`` — and
   the B side does NOTHING. |m| >= 2: all three components AND their ``fu_``
   auxiliaries held at zero on rows ``[0:|m|]`` (row 0 alone under
   ``accurate_fields_near_cylorigin``), ``fu_`` stored on those rows unconditionally.

5. **THE m = 0 ARM**, a THIRD COMPILE-TIME BODY: no i*m/r pass (the array path
   branches on ``grid.m != 0``; the rows stay BOUND and are never read), no
   axis-row increment, and the m = 0 axis rules on the stored FIELDS after the
   forms: B side ``Bx[r = 0] = 0``; D side ``Dz[r = 0] += axis_factor * Hp[r = 0]``
   (``Hp`` the stored ``Hy``, the ``g1`` centre ``b``) and then ``Dy[r = 0] = 0``.
   The increment is a POST-ADD, never a curl fold.

   THE AXIS FACTORS ARE MEEP'S DOUBLE PRODUCTS. ``4*Courant``, ``-Courant`` and
   ``+Courant`` multiply a float32 in double and round once
   (``stepping._realnum_double_product``). The kernel forms a float32 product with
   the factor's float32 word, which is that double product exactly when ``Courant``
   is a float32 word (two float32 operands multiply exactly in double); any other
   ``Courant`` is REFUSED by name (:func:`courant_reasons`, applied by
   :func:`needs_courant_product` to the arms that form one), not stepped one
   rounding off. MEEP's default Courant 0.5 is a float32 word.

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

NO ENTRY IN ``fingerprints.json``'s ``kernel_source_sha256``, as for
:mod:`.complex_fields`: the gate hashes the specialisations it launches.
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

import re
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
from . import chunk_bounds as _chunk_bounds
from .coverage import _metal_backend_reasons, _residency_declaration_reasons
from .cylindrical_real import SCAN_TARGETS, courant_reasons
from .device import MAX_BUFFER_BINDINGS, compile_source
from .plans import KernelPlan

#: The family name the arm table carries. ONE spelling, so a refusal message, a
#: registry row and an artifact column cannot drift apart.
FAMILY = "cylindrical_complex"


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
#: the FIRST OFF-AXIS row (``_cylindrical_axis_increment_B``'s ``xp.take(Ez, 1,
#: axis=0)``) and the kernel's matching load ``g2[nyz + j*nzi + k]`` is live on every
#: lane, so below
#: this it would read a full plane PAST THE END of the source volume. Measured by the
#: Triton tranche on a real ``(1, 1, 20)`` grid: NumPy RAISES IndexError there and
#: CuPy SILENTLY RETURNS ROW 0 — the two backends disagree about whether the
#: configuration is steppable at all, which is why the clause is unconditional rather
#: than |m| = 1 only.
MINIMUM_RADIAL_ROWS = 2

#: The i*m/r call sites — ``stepping._CYLINDRICAL_IMR_B`` / ``_CYLINDRICAL_IMR_D``.
#: ``(target index, partner register, sign)``; the register names which CENTER load
#: of the certified curl body the term reuses ("c" = g2 = Ez/Hz, "a" = g0 = Ex/Hx).
#: Target 1 gets no term on either side.
IMR_TERMS: Dict[str, Tuple[Tuple[int, str, float], ...]] = {
    "step_B": ((0, "c", +1.0), (2, "a", -1.0)),   # Bx <- Ez (+1), Bz <- Ex (-1)
    "step_D": ((0, "c", -1.0), (2, "a", +1.0)),   # Dx <- Hz (-1), Dz <- Hx (+1)
}

#: Which target each sub-step's |m| = 1 axis-row increment REPLACES at r = 0.
AXIS_INCREMENT_TARGET: Dict[str, int] = {"step_B": 0, "step_D": 1}   # Bx / Dy

#: Per-sub-step radial-sum spec, transcribed from ``stepping.step_B`` / ``step_D``.
PREFIX: Dict[str, Dict[str, Any]] = {
    "step_B": {"component": "Ey", "ir0": 0.0, "extend_wall_row": True},
    "step_D": {"component": "Hy", "ir0": 0.5, "extend_wall_row": False},
}

CURL_SOURCES: Tuple[str, ...] = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")

#: How many buffers the cylindrical complex curl binds: 9 float2 volumes + 6
#: coefficient vectors + 5 shape/dtdx scalars + the radial difference + 2 i*m/r rows
#: + 1 zero-rows uniform + the axis factor + the m word + the chunk record.
#: CHECKED AGAINST THE PLATFORM CEILING
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
    "CURL_BINDINGS",
    "CURL_SOURCES",
    "CYLINDRICAL_AXIS_FLAGS",
    "CylindricalComplexCurlPlan",
    "FAMILY",
    "IMR_TERMS",
    "MINIMUM_RADIAL_ROWS",
    "M_ARMS",
    "M_ARM_LABELS",
    "M_MANY",
    "M_ONE",
    "M_ZERO",
    "PREFIX",
    "REQUIRED_PHI_KIND",
    "REQUIRED_R_KIND",
    "SUB_STEPS",
    "CURL_RECORD_BUFFER",
    "CYLINDRICAL_CURL2",
    "axis_factor",
    "cylindrical_complex_constitutive_coverage",
    "cylindrical_complex_pml_curl_coverage",
    "cylindrical_curl_source",
    "enumerate_cylindrical_sources",
    "imr_rinv_row",
    "m_class",
    "needs_courant_product",
    "plan_cylindrical_complex_constitutive",
    "plan_cylindrical_complex_pml_curl",
    "plan_cylindrical_complex_pml_curl_from_arrays",
    "pull_radial_source",
    "radial_difference",
    "register_arms",
    "zero_rows",
]


# ---------------------------------------------------------------------------
# Host-side constants that must be computed the ARRAY PATH's way
# ---------------------------------------------------------------------------

def m_class(m: int) -> int:
    """``M_CLASS`` for one azimuthal order: ``M_ZERO``, ``M_ONE`` or ``M_MANY``."""
    magnitude = abs(int(m))
    if magnitude == 0:
        return M_ZERO
    return M_ONE if magnitude == 1 else M_MANY


def axis_factor(sub_step: str, m: int, courant: float) -> float:
    """The double factor of the sub-step's axis rule, bound as its float32 word.

    MEEP multiplies a ``realnum`` by a double factor of ``Courant`` and rounds once
    (step_db.cpp:324, :377; ``stepping._realnum_double_product``): ``4*Courant`` on the
    m = 0 D side (the on-axis Dz increment), ``-Courant`` on the |m| = 1 B side and
    ``+Courant`` on the |m| = 1 D side (the axis-row increments). The kernel's float32
    product with the float32 word of the factor is MEEP's word exactly when ``Courant``
    is a float32 word, which the predicate requires (``courant_reasons``). Zero where
    the arm carries no such rule; the slot is bound for one layout.
    """
    arm = m_class(m)
    courant = float(courant)
    if arm == M_ZERO and sub_step == "step_D":
        return 4.0 * courant
    if arm == M_ONE:
        return -courant if sub_step == "step_B" else courant
    return 0.0


def needs_courant_product(sub_step: str, m: int) -> bool:
    """Does this (sub-step, m) arm form a double product with ``Courant``?"""
    arm = m_class(m)
    return arm == M_ONE or (arm == M_ZERO and sub_step == "step_D")


def zero_rows(m: int, accurate_fields_near_cylorigin: bool) -> int:
    """How many near-axis rows the |m| >= 2 rule holds at zero.

    ``stepping._cylindrical_axis_rows``: ``slice(0, 1)`` on the ACCURATE branch and
    ``slice(0, |m|)`` on MEEP's default; zero below |m| = 2, where neither applies.
    A runtime uniform (a threshold on the row index), not a specialisation.
    """
    if abs(int(m)) < 2:
        return 0
    return 1 if bool(accurate_fields_near_cylorigin) else abs(int(m))


def imr_rinv_row(target: str, sign: float, m: int, courant: float, rows: int) -> Any:
    """The per-r i*m/r factor, built the array path's way: a float32 QUOTIENT.

    ``stepping._cylindrical_imr_increment``: ``the_m = realnum(sign*2*m*Courant)``
    (one rounding of a double), ``here_r = realnum(max(2*ir + iyee_r, 1))``, and
    ``rinv = the_m / here_r`` in float32 — MEEP's ``rinv = the_m / here.r()``
    (step_db.cpp:177-294). Built on the host and bound: the kernel neither divides nor
    takes a max (Metal's ``max`` disagrees with ``numpy.maximum`` on two zeros).
    """
    import numpy  # noqa: PLC0415
    from ..fields import IYEE_SHIFTS  # noqa: PLC0415

    iyee_r = IYEE_SHIFTS[target][0]
    the_m = numpy.float32(float(sign) * 2.0 * int(m) * float(courant))
    r_doubled = 2 * numpy.arange(int(rows), dtype=numpy.int64) + iyee_r
    here_r = numpy.maximum(r_doubled, 1).astype(numpy.float32)
    return numpy.ascontiguousarray(the_m / here_r, dtype=numpy.float32)


def radial_difference(xp: Any, sub_step: str, sources: Dict[str, Any], starts: Any,
                      scratch: Any = None) -> Any:
    """The radial difference the kernel's target 2 reads, from the SHIPPED array path.

    ``stepping.cylindrical_rderiv_difference`` is CALLED, never re-derived: its
    per-chunk float32 summation order DEFINES the answer. B side: Ep = ``Ey`` extended
    by ONE ZERO WALL ROW (``stepping.step_B``, including its pooled-versus-fresh
    allocation), ``ir0 = 0.0``, forward; D side: Hp = ``Hy``, ``ir0 = 0.5``, backward.
    ``starts`` is ``stepping._cylindrical_scan_starts``' answer for Bz or Dz.
    """
    from ..stepping import _face, _span, cylindrical_rderiv_difference  # noqa: PLC0415

    spec = PREFIX[sub_step]
    source = sources[spec["component"]]
    rows = source.shape[0]
    if not spec["extend_wall_row"]:
        return cylindrical_rderiv_difference(xp, source, spec["ir0"], starts, rows,
                                             backward=True, scratch=scratch)
    extended_shape = (rows + 1,) + source.shape[1:]
    extended = (xp.empty(extended_shape, dtype=source.dtype) if scratch is None
                else scratch.take("cyl_extended", extended_shape, source.dtype))
    extended[_span(0, 0, rows)] = source
    extended[_face(0, rows)] = 0
    return cylindrical_rderiv_difference(xp, extended, spec["ir0"], starts, rows,
                                         backward=False, scratch=scratch)


def pull_radial_source(residency: Any, name: str) -> None:
    """Make the HOST copy of the radial sum's source current before the host sum reads it.

    The three plans that form the radial difference on the host (this module's curl
    plan and the two cylindrical complex fused pairs) read the source volume -- ``Ey``
    on B, ``Hy`` on D -- straight from its host array. Two residency modes, two doors:

    * no hold (the shipped bracket): ``sync_out`` copies device -> host
      unconditionally, as it always has;
    * under a hold, ``sync_out`` of a HELD mirror copies NOTHING -- it marks the
      mirror device-owned and seals it, which is that method's contract for a
      launch's own writes. The host array then holds the words of the source's last
      host read, and the sum is formed from them: a smooth, plausible, wrong radial
      difference, and with it wrong ``Bz``/``Dz``. :meth:`device.Residency.acquire_read`
      is the read door: it copies the device's words when the device owns the
      mirror, leaves a host-owned one alone (its words are the newest), and keeps
      the seal.

    MEASURED on the Metal engine route under a hold (64 steps, the whole state against
    the array path): m = 1 and m = 2 runs diverged from step 4 -- 111,415 / 1,376,256
    and 118,204 / 1,146,880 words by step 64 -- and so did a run carrying a flux monitor,
    while the shipped bracket read 0 on every one. The driver-route gate's unfused rows,
    which step these plans' single curls under the same hold, diverged from step 2.
    """
    if name in residency.held:
        residency.acquire_read(name)
    else:
        residency.sync_out((name,))


# ---------------------------------------------------------------------------
# The curl kernel: the certified complex body with the cylindrical additions
# ---------------------------------------------------------------------------

#: The certified complex curl's three phase scalars, which this family does not bind:
#: a Dcyl run carries no Bloch phase (clause 7), so every connection block is the
#: comment the certified emitter writes for an unphased axis.
_PHASE_BINDINGS = (
    "    constant float2&     px      [[buffer(20)]],\n"
    "    constant float2&     py      [[buffer(21)]],\n"
    "    constant float2&     pz      [[buffer(22)]],\n")

#: The bindings this family takes where the phase scalars were, in order.
_EXTRA_BINDINGS = (
    "    device const float2* rad     [[buffer(20)]],\n"
    "    device const float*  q0      [[buffer(21)]],\n"
    "    device const float*  q2      [[buffer(22)]],\n"
    "    constant uint&       zrows   [[buffer(23)]],\n"
    "    constant float&      axis_factor [[buffer(24)]],\n"
    "    constant float&      m_word  [[buffer(25)]],\n")

#: The chunk record's binding on this family's curl, after the six extra bindings.
CURL_RECORD_BUFFER = 26

#: The certified complex curl's target-2 stencil, which the cylindrical curl replaces.
_CERTIFIED_CURL2 = (
    "    float2 curl2 = has_x ? (has_y ? c_mul_coefficient_left(dtdx, ((b_x - b_dx) + a_dy) - a_y)\n"
    "                                  : c_mul_coefficient_left(dtdx, b_x - b_dx))\n"
    "                         : (has_y ? c_mul_coefficient_left(-dtdx, a_y - a_dy) : float2(0.0f, 0.0f));\n")

#: ``part_products.scale(dtdx, radial)``: one product of the radial difference, per part.
CYLINDRICAL_CURL2 = "    float2 curl2 = c_mul_coefficient_left(dtdx, rad[ii]);\n"

#: Where the certified body's ownership mask ends and its per-chunk forms begin.
_FORMS_BEGIN = ("    // --- the two PML forms, per structure chunk "
                "(stepping._apply_curl_regions) --\n")

#: The certified body's first guarded auxiliary store: the forms end above it.
_FORMS_END = "    if (du0) { u0[ii] = n0; }\n"

#: The |m| = 1 axis-row increments, REPLACING curl row 0 after the ownership mask
#: (stepping.step_B / step_D: ``curl[r = 0] = -increment``). MEEP's bracket is formed
#: in realnum left to right, then the double product with the factor, rounded once
#: (``_realnum_double_product``; ``axis_factor`` is the factor's float32 word).
#:
#: B side (``_cylindrical_axis_increment_B``): Br takes ``-Courant * ((Ep[i] -
#: Ep[i + z]) - f_m_mult*f_m)`` with ``f_m_mult*f_m = -quarter_turn(m, Ez[row 1])``,
#: per part ``(m*Ez.im, -(m*Ez.re))``. Ep is the ``g1`` centre ``b`` and its z-up
#: neighbour ``b_z``; Ez's first off-axis row is read at this column's row-1 offset
#: (the predicate refuses nr < 2, which keeps that load in bounds).
#: D side (``_cylindrical_axis_increment_D``): Dp takes ``Courant * ((Hr[i] -
#: Hr[i - z]) - 2*Hz[i])``. Hr is the ``g0`` centre ``a`` and its z-down neighbour
#: ``a_z``; Hz the ``g2`` centre ``c``.
_AXIS_INCREMENT_BLOCKS: Dict[bool, str] = {
    False: """    // --- the |m| = 1 axis-row increment (stepping._cylindrical_axis_increment_B)
    {
        float2 e1 = g2[nyz + j * nzi + k];
        float2 partner = float2(m_word * e1.y, -(m_word * e1.x));
        float2 bracket = (b - b_z) - partner;
        float2 inc = c_mul_coefficient_left(axis_factor, bracket);
        curl0 = at_x ? -inc : curl0;
    }
""",
    True: """    // --- the |m| = 1 axis-row increment (stepping._cylindrical_axis_increment_D)
    {
        float2 bracket = (a - a_z) - c_mul_coefficient_left(2.0f, c);
        float2 inc = c_mul_coefficient_left(axis_factor, bracket);
        curl1 = at_x ? -inc : curl1;
    }
""",
}

_MASK_ROW = re.compile(r"^    curl([012]) = (\w+) \? (.+) : curl\1;$", re.MULTILINE)


def _increment_mask(codes: Sequence[int], backward: bool, targets: Sequence[int]) -> str:
    """The ownership mask rows of the curl, re-emitted on the i*m/r increments.

    ``stepping._mask_non_owned_cells`` masks the increment with the same predicate it
    masks the curl with (the target's own Yee shifts), so the rows are read off the
    certified emitter's own output for the same triple and direction.
    """
    rows = _MASK_ROW.findall(templates.ownership_mask(codes, backward,
                                                      zero=templates.COMPLEX_ZERO))
    lines = [f"        mi{target} = {flag} ? {zero} : mi{target};"
             for target, flag, zero in rows if int(target) in targets]
    return "\n".join(lines) or "        // no unowned cell masks an increment here"


def _imr_pass(sub_step: str, codes: Sequence[int]) -> str:
    """The i*m/r pass after the forms: increment, mask, MEEP's additive ladder.

    ``stepping._cylindrical_imr_increment`` returns ``quarter_turn(rinv, partner)`` —
    per part ``(-(rinv*g.im), rinv*g.re)`` — in curl sign convention; the caller masks
    it to ``+0`` on cells MEEP's owned loop skips and ``_apply_folded_increment``
    SUBTRACTS it through the ladder ``df = sinv_dsig*inc``; outside the dsigu chunks
    ``f -= df``, inside ``fu -= df; f -= sinv_dsigu*df``, per part. ``x - (+0)`` is
    ``x``, so a masked cell keeps its word, ``-0`` included. In register, after the
    forms and before the stores: a float32 stored and reloaded is its own word.
    """
    backward = bool(SUB_STEPS[sub_step]["backward"])
    terms = IMR_TERMS[sub_step]
    lines = ["    // --- the i*m/r pass (stepping._cylindrical_imr_increment, "
             "_apply_folded_increment)",
             "    {"]
    for target, register, _sign in terms:
        row = "q0" if target == 0 else "q2"
        lines.append(f"        float2 mi{target} = float2(-({row}[i] * {register}.y), "
                     f"{row}[i] * {register}.x);")
    lines.append(_increment_mask(codes, backward, [t for t, _r, _s in terms]))
    for target, _register, _sign in terms:
        dsig, dsigu = shaders.CURL_CYCLE[target]
        si, si_u = f"si_{'xyz'[dsig]}", f"si_{'xyz'[dsigu]}"
        lines += [
            f"        float2 mdf{target} = float2({si} * mi{target}.x, {si} * mi{target}.y);",
            f"        n{target} = du{target} ? (n{target} - mdf{target}) : n{target};",
            f"        v{target} = du{target} ? (v{target} - float2({si_u} * mdf{target}.x, "
            f"{si_u} * mdf{target}.y)) : (v{target} - mdf{target});",
        ]
    lines.append("    }")
    return "\n".join(lines) + "\n"


#: The per-m axis rules on the stored values, after the i*m/r pass, in the array
#: path's order (``stepping._cylindrical_axis_zero_B`` / ``_D``).
_M_ZERO_RULES: Dict[bool, str] = {
    False: """    // --- the m = 0 axis rule (stepping._cylindrical_axis_zero_B): Bx[r = 0] = 0
    v0 = at_x ? float2(0.0f, 0.0f) : v0;
""",
    True: """    // --- the m = 0 axis rules (stepping._cylindrical_axis_zero_D): Dz[r = 0] +=
    // 4*Courant*Hp[r = 0] (Hp the stored Hy, the g1 centre), then Dy[r = 0] = 0.
    {
        float2 dfcnd = c_mul_coefficient_left(axis_factor, b);
        v2 = at_x ? (v2 + dfcnd) : v2;
        v1 = at_x ? float2(0.0f, 0.0f) : v1;
    }
""",
}

_M_ONE_RULES: Dict[bool, str] = {
    False: "    // |m| = 1, B side: no axis zero rule (stepping._cylindrical_axis_zero_B).\n",
    True: """    // --- the |m| = 1 axis rule (stepping._cylindrical_axis_zero_D): Dz[r = 0] = 0
    v2 = at_x ? float2(0.0f, 0.0f) : v2;
""",
}

#: |m| >= 2: all three components AND their fu on every row within ``zrows`` of the
#: axis, on every cell (``host_writes.zero_rows`` on each array), so fu is stored on
#: those rows whatever the cell's form.
_M_MANY_RULE = """    // --- the |m| >= 2 near-axis rule (stepping._cylindrical_axis_zero_B / _D)
    {
        bool near = (i < int(zrows));
        v0 = near ? float2(0.0f, 0.0f) : v0;
        v1 = near ? float2(0.0f, 0.0f) : v1;
        v2 = near ? float2(0.0f, 0.0f) : v2;
        n0 = near ? float2(0.0f, 0.0f) : n0;
        n1 = near ? float2(0.0f, 0.0f) : n1;
        n2 = near ? float2(0.0f, 0.0f) : n2;
        if (near) { u0[ii] = n0; u1[ii] = n1; u2[ii] = n2; }
    }
"""


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AssertionError(
            f"the certified complex curl source carries {what} ({old.strip()[:80]!r}) "
            f"{count} times, not once; the cylindrical curl cannot be derived from it")
    return text.replace(old, new, 1)


def cylindrical_curl_source(bcz: int, backward: bool, m_arm: int,
                            contract: str = shaders.CONTRACT_OFF) -> str:
    """The cylindrical complex curl: the certified complex body with checked edits.

    :func:`.complex_fields.bloch_curl_source` over ``(METALLIC, PERIODIC, bcz)`` with no
    phased axis, then: the entry point renamed ``cyl_complex_pml_curl_step``; the three
    unread phase scalars replaced by this family's six bindings, the chunk record after
    them; target 2's stencil replaced by ``dtdx * rad[ii]``; the |m| = 1 axis-row
    increment after the ownership mask; the i*m/r pass (|m| >= 1) and the per-m axis
    rules after the forms. Everything else — the bounds block, the ghost rule, targets
    0 and 1 with their absent phi partner, the ownership mask and the per-chunk forms —
    is the certified text. Each edit must land exactly once.
    """
    if bcz not in (templates.PERIODIC, templates.METALLIC):
        raise ValueError(f"bcz must be PERIODIC or METALLIC, got {bcz!r}")
    if m_arm not in M_ARMS:
        raise ValueError(f"m_arm must be M_ZERO ({M_ZERO}), M_ONE ({M_ONE}) or "
                         f"M_MANY ({M_MANY}), got {m_arm!r}")
    backward = bool(backward)
    sub_step = "step_D" if backward else "step_B"
    codes = (templates.METALLIC, templates.PERIODIC, int(bcz))
    source = complex_fields.bloch_curl_source(codes, backward, (0, 0, 0), contract)
    source = _replace_once(source, "kernel void bloch_pml_curl_step(",
                           "kernel void cyl_complex_pml_curl_step(", "its entry point")
    source = _replace_once(source, _PHASE_BINDINGS, "", "its phase scalars")
    source = _replace_once(
        source,
        f"    constant ChunkBounds& cb [[buffer({complex_fields.CURL_RECORD_BUFFER})]],\n",
        _EXTRA_BINDINGS + f"    constant ChunkBounds& cb [[buffer({CURL_RECORD_BUFFER})]],\n",
        "its chunk-record binding")
    source = _replace_once(source, _CERTIFIED_CURL2, CYLINDRICAL_CURL2,
                           "its target-2 stencil")
    if m_arm == M_ONE:
        source = _replace_once(source, _FORMS_BEGIN,
                               _AXIS_INCREMENT_BLOCKS[backward] + "\n" + _FORMS_BEGIN,
                               "the start of its forms")
    if m_arm == M_ZERO:
        tail = _M_ZERO_RULES[backward]
    elif m_arm == M_ONE:
        tail = _imr_pass(sub_step, codes) + _M_ONE_RULES[backward]
    else:
        tail = _imr_pass(sub_step, codes) + _M_MANY_RULE
    return _replace_once(source, _FORMS_END, tail + "\n" + _FORMS_END,
                         "its first guarded auxiliary store")


def compile_cylindrical_curl(bcz: int, backward: bool, m_arm: int,
                             contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(cylindrical_curl_source(
        bcz, backward, m_arm, contract)).cyl_complex_pml_curl_step


def enumerate_cylindrical_sources(contract: str = shaders.CONTRACT_OFF
                                  ) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit — TWELVE, keyed by a stable label.

    ``zero_rows``, the radial difference, the i*m/r rows, the axis factor and the chunk
    record are bound data, so the enumeration is closed: direction x z rule x m class.
    """
    out: Dict[str, str] = {}
    for backward, name in ((False, "step_B"), (True, "step_D")):
        for bcz in (templates.PERIODIC, templates.METALLIC):
            for m_arm in M_ARMS:
                arm_name = M_ARM_LABELS[m_arm]
                label = f"cyl_complex_pml_curl_step/{name}/bcz{bcz}/{arm_name}"
                out[label] = cylindrical_curl_source(bcz, backward, m_arm, contract)
    return out


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

def _shared_reasons(fields: Any, pml: Any, grid: Any, residency: Any) -> List[str]:
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
                                          residency: Any = None) -> Coverage:
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

    reasons = _shared_reasons(fields, pml, grid, residency)
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

    # 15. The axis rules' products with Courant are MEEP's double products.
    try:
        m = int(getattr(grid, "m"))
    except Exception:  # noqa: BLE001 - clause 14 already names an unreadable m
        m = None
    if m is not None and needs_courant_product(sub_step, m):
        reasons.extend(courant_reasons(grid, "the cylindrical axis rule's product "
                                             "with Courant"))

    return Coverage(not reasons, tuple(reasons))


def cylindrical_complex_constitutive_coverage(fields: Any, pml: Any, side: str,
                                              residency: Any = None) -> Coverage:
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
    reasons = _shared_reasons(fields, pml, grid, residency)
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
    """A launchable COMPLEX cylindrical curl sub-step: array-path radial sum, ONE kernel.

    THE ONE DIVERGENCE FROM :class:`.plans.KernelPlan`, STATED WHERE IT HAPPENS.
    :meth:`run` calls ``super().run``, so the dict lookup, the variant refusal and the
    launch counter are the base's, unchanged. What it ADDS is a pre-pass: the radial
    difference is the array path's per-chunk sequential sum, computed on the host by
    ``stepping.cylindrical_rderiv_difference`` itself (:func:`radial_difference`).

    So one launch costs, in this order: ``sync_out`` of the ONE source volume (``Ey``
    on B, ``Hy`` on D), the sum on the host, ``sync_in`` of the radial-difference
    mirror, and the base's launch. The syncs are performed HERE and not by the caller,
    and :attr:`prefix_syncs` counts the pre-pass so a gate can assert it happened.

    The two i*m/r rows, the axis factor and the chunk starts are GRID INVARIANTS, built
    once at plan time.
    """

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bcz", "m_arm",
                 "zero_rows", "axis_factor", "m_word", "residency",
                 "volumes", "scratch", "starts", "chunk_bounds", "prefix_syncs",
                 "_radial_host", "_radial_name", "_source_name", "_source_map", "_xp")

    REPR_FIELDS = ("sub_step", "shape", "bcz", "m_arm", "zero_rows", "axis_factor")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, sub_step: str, shape, dtdx: float, courant: float, m: int,
                 bcz: int, zero_row_count: int, residency: Any,
                 targets, auxiliaries, sources, radial, imr_rows, coefficients,
                 functions: Dict[str, Any], volumes: Sequence[str], radial_host: Any,
                 radial_name: str, source_name: str, source_map: Dict[str, Any],
                 xp: Any, starts: Any, scratch: Any = None, *,
                 chunk_bounds: Any) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(n) for n in shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path consumes the Python scalar at complex64 precision
        # and Metal binds a Python float into `constant float&` correctly rounded.
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bcz = int(bcz)
        self.m_arm = m_class(m)
        self.zero_rows = int(zero_row_count)
        self.axis_factor = axis_factor(sub_step, m, courant)
        self.m_word = float(int(m))
        self.residency = residency
        self.volumes = tuple(volumes)
        self.scratch = scratch
        self.starts = starts
        self.prefix_syncs = 0
        self._radial_host = radial_host
        self._radial_name = str(radial_name)
        self._source_name = str(source_name)
        self._source_map = dict(source_map)
        self._xp = xp
        self.chunk_bounds = _chunk_bounds.plan_words(
            chunk_bounds, self.shape, f"CylindricalComplexCurlPlan({sub_step})")
        super().__init__(functions, (
            tuple(targets) + tuple(auxiliaries) + tuple(sources) + tuple(coefficients)
            + (self.shape[0], self.shape[1], self.shape[2], self.n_elem, self.dtdx)
            + (radial,) + tuple(imr_rows)
            + (self.zero_rows, float(self.axis_factor), self.m_word)
            + (_chunk_bounds.tensor(self.chunk_bounds,
                                    complex_fields._plan_device(residency)),)))

    def refresh_prefix(self) -> None:
        """Pull the sum's SOURCE off the device, sum it on the host, push it back.

        Separated from :meth:`run` so a gate can call it explicitly and count it. It is
        NOT an optional step: :meth:`run` always calls it. The pull is
        :func:`pull_radial_source`, which reads through the hold when one is armed.
        """
        pull_radial_source(self.residency, self._source_name)
        radial = radial_difference(self._xp, self.sub_step, self._source_map,
                                   self.starts, scratch=self.scratch)
        self._radial_host[...] = radial
        self.residency.sync_in((self._radial_name,))
        self.prefix_syncs += 1

    def run(self, contract: Optional[str] = None) -> None:
        """Refresh the radial difference, then take the BASE's launch path unchanged."""
        self.refresh_prefix()
        super().run(contract)


def _curl_functions(bcz: int, backward: bool, m_arm: int,
                    contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_cylindrical_curl(bcz, backward, m_arm, mode)
            for mode in contract_variants}


def _build_curl_plan(sub_step: str, shape, dtdx: float, courant: float, m: int,
                     accurate: bool, bcz: int, residency: Any,
                     arrays: Dict[str, Any], flat: Dict[str, Any],
                     coefficient_names: Sequence[str],
                     functions: Optional[Dict[str, Any]],
                     contract_variants: Sequence[str], scratch: Any, starts: Any,
                     chunk_bounds: Any) -> CylindricalComplexCurlPlan:
    """The ONE plan assembly both routes take.

    The engine route and the gate's bare-array route differ ONLY in where the arrays
    come from and how the coefficient mirrors are named. Everything that decides a BIT
    — the m class, the zero-row count, the i*m/r rows, the axis factor, the radial
    buffer and the argument order — is assembled here once.
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

    # THE RADIAL BUFFER IS THE PLAN'S OWN and is allocated once: `Residency.mirror`
    # refuses re-registering a name against a DIFFERENT host array, and the array
    # path's sum returns a pooled or fresh array every call, so its result is COPIED
    # into this buffer.
    radial_host = numpy.zeros(shape, dtype=numpy.complex64)
    radial_name = f"cyl:radial:{sub_step}"
    radial = mirror(residency, radial_name, radial_host)

    imr_rows = []
    for index, _register, sign in IMR_TERMS[sub_step]:
        target = spec["targets"][index]
        row = imr_rinv_row(target, sign, m, courant, rows)
        imr_rows.append(residency.mirror(f"cyl:imr:{sub_step}:{target}", row,
                                         constant=True))

    coefficients = [residency.mirror(name, flat[name], constant=True)
                    for name in coefficient_names]

    volumes = (tuple(spec["targets"]) + tuple("fu_" + n for n in spec["targets"])
               + tuple(spec["sources"]))
    source_map = {name: arrays[name] for name in spec["sources"]}
    return CylindricalComplexCurlPlan(
        sub_step, shape, dtdx, courant, m, bcz, zero_rows(m, accurate),
        residency, targets, auxiliaries, sources, radial, imr_rows, coefficients,
        functions if functions is not None
        else _curl_functions(bcz, spec["backward"], arm,
                             contract_variants),
        volumes, radial_host, radial_name, PREFIX[sub_step]["component"],
        source_map, numpy, starts, scratch, chunk_bounds=chunk_bounds)


def plan_cylindrical_complex_pml_curl(
        fields: Any, pml: Any, sub_step: str, residency: Any = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        *, chunk_bounds: Any,
        ) -> Optional[CylindricalComplexCurlPlan]:
    """Build a plan from the engine's own objects, or None when out of coverage.

    ``chunk_bounds`` is the record of the division the run steps with (the composer's
    ``StepContext.chunk_bounds``) and has no default; the radial sum's chunk starts are
    the array path's own (``stepping._cylindrical_scan_starts``). None is the only
    refusal.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, "
                         f"got {sub_step!r}")
    if not cylindrical_complex_pml_curl_coverage(
            fields, pml, sub_step, residency).covered:
        return None
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from ..stepping import _cylindrical_scan_starts  # noqa: PLC0415

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
    starts = _cylindrical_scan_starts(fields, pml, SCAN_TARGETS[sub_step])
    return _build_curl_plan(
        sub_step, grid.shape, grid.dt / grid.dx, grid.courant, int(grid.m),
        bool(grid.accurate_fields_near_cylorigin), bcz, residency,
        arrays, flat, coefficient_names, None, contract_variants,
        getattr(fields, "scratch", None), starts, chunk_bounds)


def plan_cylindrical_complex_pml_curl_from_arrays(
        sub_step: str, arrays: Dict[str, Any], flat: Dict[str, Any], bcz: int,
        m: int, accurate_fields_near_cylorigin: bool, dtdx: float,
        residency: Any, functions: Optional[Dict[str, Any]] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        scratch: Any = None, *, chunk_bounds: Any, courant: Optional[float] = None,
        starts: Any = None) -> CylindricalComplexCurlPlan:
    """Build a plan from bare host arrays — the gate's and benchmark's route.

    No predicate runs. ``chunk_bounds`` is required; ``starts`` is
    ``stepping._cylindrical_scan_starts``' answer for the sub-step's target (``None``:
    one chunk from the axis); ``courant`` defaults to ``dtdx``. ``functions`` is the
    MUTATION SEAM: dropping it is a silent DISARMING, not a slowdown.
    """
    spec = SUB_STEPS[sub_step]
    shape = tuple(int(n) for n in arrays[spec["targets"][0]].shape)
    coefficient_names = [f"pml:{stem}_{axis}:{sub_step}"
                         for axis in "xyz" for stem in ("kms", "sinv")]
    resolved = {name: flat[name.split(":")[1]] for name in coefficient_names}
    return _build_curl_plan(
        sub_step, shape, dtdx, float(dtdx) if courant is None else float(courant), m,
        bool(accurate_fields_near_cylorigin), bcz, residency, arrays,
        resolved, coefficient_names, functions, contract_variants, scratch, starts,
        chunk_bounds)


def plan_cylindrical_complex_constitutive(
        fields: Any, pml: Any, side: str, residency: Any = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        *, chunk_bounds: Any) -> Any:
    """A CERTIFIED complex constitutive plan for a Dcyl run, or None.

    The arithmetic, the source and the plan class are ``complex_fields``' own; only the
    ADMISSION is this family's. ``chunk_bounds`` as the curl takes it.
    """
    if side not in CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(CONSTITUTIVE_SIDES)}, "
                         f"got {side!r}")
    if not cylindrical_complex_constitutive_coverage(
            fields, pml, side, residency).covered:
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
        side, fields.grid.shape, residency, targets, auxiliaries,
        sources, inverse_epsilon, coefficients,
        complex_fields._constitutive_functions(side, contract_variants),
        volumes, chunk_bounds=chunk_bounds)


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
        context.fields, context.pml, slot, context.residency)


def _curl_arm_plan(context: Any, slot: str) -> Any:
    return plan_cylindrical_complex_pml_curl(
        context.fields, context.pml, slot, context.residency,
        context.contract_variants,
        chunk_bounds=context.chunk_bounds)


def _constitutive_arm_coverage(context: Any, slot: str) -> Coverage:
    return cylindrical_complex_constitutive_coverage(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency)


def _constitutive_arm_plan(context: Any, slot: str) -> Any:
    return plan_cylindrical_complex_constitutive(
        context.fields, context.pml, _CONSTITUTIVE_SLOT_SIDES[slot],
        context.residency, context.contract_variants,
        chunk_bounds=context.chunk_bounds)


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
