"""The CYLINDRICAL COMPLEX magnetic seam in one dispatch: Dcyl ``step_B`` welded
into ``update_H``.

The complex-storage Dcyl arm's B->H product — m = 0 and |m| >= 1 since 2026-09-04,
|m| >= 1 before — and the only large cell on the fusion matrix with NO source-seam
attrition at all. Both halves are the certified cylindrical
tranche's own: :func:`.cylindrical_complex.cylindrical_curl_source` with
``backward=False`` on ``step_B``, and — because ``stepping.update_H`` carries no
cylindrical branch whatsoever — :func:`.complex_fields.bloch_constitutive_source`
on the H side, lifted through the shipped
:func:`.complex_fused_magnetic_pair.certified_constitutive_body`. The weld is the
one substitution every shipped B->H pair makes: the constitutive half's three
``float2 srcN = gN[ii];`` reloads become the registers the curl half just produced.

NOTHING IS RE-DERIVED HERE. Every arithmetic line in the emitted kernel comes out
of one of those two certified emitters' own output, and the two edits between them
are the inline wall clear (imported from
:func:`.complex_fused_magnetic_pair.zero_metal_mask`, not re-spelled) and the seam
substitution. A reader can diff :func:`cylindrical_fused_magnetic_pair_source`'s
output against ``cylindrical_curl_source`` + ``bloch_constitutive_source`` and see
that nothing moved; the gate's ``transcription`` leg is what turns that from a
claim into a measurement.

===========================================================================
WHY THIS CELL AND NOT ANOTHER — measured from the fusion matrix
===========================================================================

``parity/meep_gpu/results/fusion_matrix_metal_2026-08-20`` ranks the 24 (curl,
constitutive) cells the 186-row corpus drives that no fused product covers. This
one is RANK 1, and it is the only cell in the ranking whose ceiling equals its row
count:

    ceiling 16  (rows 16)  B->H  cylindrical complex PML -> cylindrical complex

**NO CYLINDRICAL CORPUS ROW DECLARES A MAGNETIC SOURCE.** Re-measured on the
census this file prices against
(``parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19``, the measured rows
of ``examples.jsonl`` + ``tests.jsonl`` + ``tests_param_matched.jsonl``):

    admit the cylindrical complex curl at step_B              16 rows
    admit the cylindrical complex constitutive at update_H    16 rows
    admit BOTH halves                                         16 rows
    ... and declare NO MAGNETIC source                        16 rows

Every one of the sixteen declares electric sources only — ``('D',)`` on fourteen
and ``('D', 'D')`` on ``cylinder_cross_section.py`` and ``zone_plate.py``. So the
clause that costs the real twin 22 of 46 rows and the complex twin 4 of 16 costs
this family NOTHING, and the family's row count IS the seam's row count. That is
the whole reason this cell outranks cells driven by three times as many rows.

The clause is still REFUSED BY NAME and still evaluated per configuration: a
corpus is not a contract, and a Dcyl script that drove a magnetic current would be
admitted by both halves and wrong through this weld.

===========================================================================
THE FIVE PASSES IN THE SEAM, each answered by name
===========================================================================

The driver runs five passes between the two halves (driver.py:3281-3289)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

* **the magnetic sources — REFUSED BY NAME** (driver.py:3283-3284). Costs zero
  rows on the measured corpus and is checked anyway. IGNORANCE IS NEVER AN EMPTY
  SET: ``Fields`` does not hold the source list, so an undeclared ``sources`` is a
  REFUSAL and not an assumed ``()``.
* **``fill_symmetry_bc_B`` — PROVABLY A NO-OP HERE.**
  ``stepping._fill_symmetry_ghost_cells`` returns at its first line unless
  ``grid.has_symmetry()`` (stepping.py:1481-1482), and ``Grid`` refuses a mirror
  plane on a Dcyl cell outright (grid.py:648-653) — which
  :func:`.cylindrical_complex._cylindrical_geometry_reasons` restates rather than
  inherits. RE-CHECKED HERE ANYWAY, for the reason the complex twin re-checks it:
  a weld may not read its own coverage off another module's guard.
* **``zero_metal_B`` — CARRIED INLINE**, through
  :func:`.coverage.zero_metal_axes` and
  :func:`.complex_fused_magnetic_pair.zero_metal_mask`, both IMPORTED.
* **``fill_folded_far_ghosts_B`` — PROVABLY A NO-OP HERE**, by the same
  ``has_symmetry`` guard one level up (stepping.py:1565-1566).

**THE WALL TABLE ON A Dcyl GRID IS ``(False, False, z_metallic)``, AND THAT IS A
POSITIVE REQUIREMENT RATHER THAN AN OBSERVATION.** ``Grid`` builds
``metallic_axes`` as ``pair[1] == METALLIC and pair[0] != AXIS`` (grid.py:512-514)
and the r axis's pair is ``(AXIS, METALLIC)``, so r is NOT a walled axis for
``_zero_metal``'s purposes even though its far face is a metallic wall: the r = 0
row belongs to the per-|m| rules, not to a PEC clear. phi is periodic. So the only
line :func:`.complex_fused_magnetic_pair.zero_metal_mask` can emit here is Bz's
``at_z`` clear, and :func:`cylindrical_fused_magnetic_pair_coverage` REFUSES a
grid whose table says otherwise rather than emitting a clear on the axis row that
the array path does not perform.

That also means the wall mask is emitted on exactly ONE axis on this family, which
is a fact a gate has to design around: a wall mutation armed on the x or y row
would rewrite a line the emitter never emits and report UNCAUGHT while measuring
nothing. The gate arms its wall mutations on the z row and asserts the x and y
rows are ABSENT from the shipped source.

===========================================================================
THE CYLINDRICAL HALF IS THE CURL HALF, AND ONLY THE CURL HALF
===========================================================================

``stepping.update_H`` (:907-925) and ``update_E`` (:926-995) contain ZERO
occurrences of ``cylindrical``, ``is_axis``, ``m`` or ``axis_zero``, and the two
axis-zero passes (``_cylindrical_axis_zero_B`` :376, ``_D`` :457) are called from
the CURL only. The cylindrical tranche measured that rather than argued it —
480/480 rows, 0 differing uint32 words, against a plain complex elementwise
``dsigw`` reference on real Dcyl grids
(:func:`.cylindrical_complex.cylindrical_complex_constitutive_coverage`). So the
constitutive half of this pair is the ORDINARY complex one, and this module lifts
it through the shipped complex pair's own lift rather than writing a second.

CONSEQUENCE FOR THE SEAM: the curl half's ``v0``/``v1``/``v2`` registers already
carry the per-m axis rules (``M_ZERO`` clears ``v0`` — ``Bx[r = 0] = 0``, the field
only; ``M_ONE``'s ``Dz`` clear is a D-side rule and does nothing here; ``M_MANY``'s
near-axis hold zeroes all three ``v`` AND all three ``n``), so the register the
constitutive half consumes is exactly the ``B`` word the array path would have
stored. The wall clear is spliced AFTER the axis rules
and BEFORE the store, which is the driver's order; both write an exact complex
zero, so the two are order-independent in bits and the transcription follows the
driver anyway.

===========================================================================
THE BINDING BUDGET — 31 OF 31, MEASURED, WITH ZERO HEADROOM
===========================================================================

    curl half        18 pointers (9 float2 volumes + prefix + 2 i*m/r rows
                                  + 6 coefficient vectors)
    constitutive     18 pointers (3 H + 3 f_w_H + 3 sources + 3 inv_eps
                                  + 6 coefficient vectors)
    shared / dropped  6          (the 3 sources ARE the curl's 3 targets and
                                  become registers; the 3 inv_eps pointers are
                                  unread on the H side and are dropped)
    fused            30 pointers + ONE packed ``constant Params&`` = 31 bindings

:data:`PACKED_BINDINGS` is 31 and :data:`.device.MAX_BUFFER_BINDINGS` is 31. THERE
IS NO ROOM FOR A THIRTY-SECOND BINDING, and that is measured on this toolchain
rather than asserted: a ``[[buffer(31)]]`` attribute is refused by the front end
with "``'buffer' attribute parameter is out of bounds: must be between 0 and 30``".
The gate's ``binding_ceiling`` leg compiles :func:`refuted_thirty_second_binding`
and requires that failure, then compiles AND LAUNCHES the shipped 31-binding
signature and reads every ``Params`` field back off the device.

**SO THE PACKING IS NOT A STYLE CHOICE HERE, IT IS THE ONLY REACHABLE SIGNATURE.**
The certified curl binds its eight non-pointer arguments SEPARATELY at its own
26-binding size; scaled to this pair's 30 pointers that is 38 bindings, seven over
the ceiling. :func:`refuted_separate_scalar_source` builds it at
:data:`SEPARATE_SCALAR_BINDINGS` and the gate requires that failure too. And a
single additional pointer — an inverse-mu volume, a second prefix, a third i*m/r
row — would put this product over the ceiling outright. That is stated so a later
widening is refused by a number rather than discovered as a compile error.

THE STRUCT FIELD ORDER IS THE MEASURED-SAFE ONE. Metal aligns ``float2`` to 8
bytes, so ``inc_b`` is declared FIRST and the natural NumPy record's offsets
``(0, 8, 12, 16, 20, 24, 28, 32)`` then coincide with Metal's with no internal
padding — the same defect the complex twin measured (a scalars-first record reads
every ``float2`` one word early, and the result is a plausible complex number
rather than garbage). :func:`params_record_dtype` states every offset explicitly
and the gate reads all eight fields back.

===========================================================================
THE PREFIX STAYS ON THE HOST, AND THE PAIR INHERITS THAT COST
===========================================================================

The radial prefix is a SEQUENTIAL float32 scan whose summation order DEFINES the
answer (``stepping.cylindrical_rderiv_prefix``), so it cannot move into this
kernel any more than it could move into the certified curl. One ``run`` therefore
costs, in this order: ``sync_out`` of ``Ey``, the host scan, ``sync_in`` of the
prefix mirror, then ONE dispatch. :attr:`~CylindricalFusedMagneticPairPlan.
prefix_syncs` counts the pre-pass so a gate can assert it happened rather than
trust that it did.

WHAT THE FUSION REMOVES IS THEREFORE ONE DISPATCH AND, ON A WALLED RUN, ONE HOST
ROUND TRIP — not the prefix traffic. The separate composition for an admitted
configuration is: prefix pre-pass, curl dispatch, ``zero_metal_B`` on the host
between two syncs, constitutive dispatch. This is: prefix pre-pass, ONE dispatch.
No throughput claim is made anywhere in this file; the gate measures dispatch and
host-pass counts, which is what a byte-neutral fusion can honestly report.

===========================================================================
THE EXPANSION ARM — bound from THIS family's artifact, never another's
===========================================================================

This kernel performs SEVEN complex-multiply orientations: ``complex_fields``' base
five plus the two the cylindrical tranche added — the i*m/r coefficient as a
complex ROW on the left
(:data:`.cylindrical_complex.IMR_ROW_PROBE_PATTERN`) and the |m| = 1
axis-increment scalar as a complex SCALAR on the left whose real word is a signed
zero (:data:`.cylindrical_complex.AXIS_SCALAR_PROBE_PATTERN`). The constitutive
half adds none — every multiply it performs is ``c_mul_coefficient_left`` with a
real coefficient, which is already in the base five.

So the arm is bound through :func:`.cylindrical_complex.expansion_from_probe` over
:data:`.cylindrical_complex.CYLINDRICAL_PROBE_PATTERNS`, from THAT family's own
environment variable, and this module carries no probe of its own. Reading the
plain complex family's five-pattern artifact here would licence a kernel that
performs seven orientations from a record that measured five — which is exactly
the drift the separate variable exists to prevent.

NOT WIRED, and not an arm ``plan_step`` can select. ``STEP_ORDER`` assigns at most
one arm per slot and this product spans THREE (``step_B``, ``zero_metal_B``,
``update_H``), so it registers ``wired=False``: enumerable by the disjointness
sweep, invisible to ``arms.arms_for``, and unreachable from a default run. The
same deferral every shipped fused pair ships under.

NO ENTRY IN ``fingerprints.json``'s ``kernel_source_sha256``, for
:mod:`.complex_fields`' reason: this module's sources are a function of the
PROBE-BOUND arm, so a checked-in hash would record a choice rather than a
measurement. The gate hashes every specialisation of the arm it actually bound,
into its own results directory, beside the probe artifact that bound it.

DEVICE STATUS: **RELEASED 2026-08-20**, Apple GPU via MPS, torch 2.10.0, under the
    native float32 flush the platform applies and no lever changes. 50/50 gate rows
    passed and the runner welded the release
    (``parity/meep_gpu/results/metal_cylindrical_fused_magnetic_pair_2026-08-20/``).
    Six product cases — both |m| arms, both z terminations, both signs of m and the
    ``accurate_fields_near_cylorigin`` row — were BIT-IDENTICAL over twelve complete
    driver steps to the array path AND to the two separately certified products
    stepping the same seam as separate dispatches. 26 armed defects each diverged;
    3 predicted nulls were confirmed with their measured reason; the disarm control
    and the byte-neutral register-versus-reload control were both clean. The release
    verdict was shown to FLIP: with the wall clear spliced AFTER the store instead
    of before it, the gate reports FAIL and ``released=False``.

    STILL NOT WIRED. A weld licenses the claim, not a dispatch: nothing in a default
    run reaches this plan.

    TWO CHOICES THIS GATE MEASURED AND CANNOT HOLD, both pinned instead by
    source-text assertions in ``meep_gpu/test_metal_cylindrical_fused_magnetic_pair.py``:

    * the i*m/r product's OPERAND ORDER. Every entry of every coefficient row this
      family binds has a real word of exactly ``+0.0`` (measured: 16/16 words
      ``0x00000000`` at m = +-1 and m = 3, on both targets), and with ``c_re = +-0.0``
      the two orientations are the same single-rounding operation. Measured 0
      differing words across twelve rows spanning both value classes and three
      lattice draws;
    * the ``t0`` and ``t2`` curl GROUPING. Both lead with the phi self-difference,
      an exact ``+0.0`` on a one-cell axis, so flattening their parens changes no
      word. Measured 0, against 503 for the identical edit on ``t1`` — which the
      gate arms as a catch.

    AND ONE THE FIRST CUT GOT WRONG. The ``|m| = 1`` increment's REPLACE-versus-
    ACCUMULATE choice differs only at a ``-0.0`` operand. Under random seeding AND
    under zero-init with a thin absorber it is NULL — Bx's split-field dsig axis is
    phi, whose ``kms`` is 1.0, so an all-``+0.0`` state cannot manufacture the class.
    On a **+-0 LATTICE** it is CAUGHT (2 words, on 2 of 3 draws). The first cut of
    the gate swept zero-init only and reported the defect uncaught; leg
    ``signed_zero`` now sweeps both classes and requires the pair.

Import contract: importable WITHOUT torch. The predicates and the plan builder (to
``None``) must answer on a host with no GPU, which is the merge bar.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import (
    CONSTITUTIVE_SIDES, Coverage, MAGNETIC_FIELD_TYPE,
)
from ..triton_kernels.launch import SUB_STEPS
from . import complex_fields, complex_fused_magnetic_pair, cylindrical_complex
from . import shaders, templates
from .coverage import zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, compile_source
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it did when the
#: clause was written out here by hand. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cylindrical_complex_fused_magnetic_pair"

#: The sub-step slot this arm is registered on. It spans three; it holds a row on
#: the first, so a refusal is NAMED on the slot the fusion starts at rather than
#: being invisible to the arm table.
SLOT = "step_B"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"

#: The driver passes ONE dispatch of this plan performs, in driver order
#: (driver.py:3281-3289). Declared, never inferred from the slot name. The two
#: symmetry fills are NOT listed: ``Grid`` refuses a mirror plane on a Dcyl cell
#: outright, so unlike the Cartesian twins there is no configuration on which this
#: family could be handed a live fill to swallow. The predicate re-checks anyway.
REPLACES: Tuple[str, ...] = ("step_B", "zero_metal_B", "update_H")

#: The shipped binding count: 30 pointers + one packed ``constant Params&``.
#: EXACTLY the platform ceiling. See the module docstring.
PACKED_BINDINGS = 31

#: What the same 30 pointers need with the nine non-pointer arguments bound
#: SEPARATELY — the way the certified 27-binding cylindrical curl binds them.
#: Eight over the ceiling; :func:`refuted_separate_scalar_source` builds it and the
#: gate requires the compile failure. (38 before 2026-09-04: the ``M_ZERO`` arm's
#: ``axis_coef`` is the ninth scalar.)
SEPARATE_SCALAR_BINDINGS = 39

#: One more pointer than the shipped signature can hold. Built by
#: :func:`refuted_thirty_second_binding` and REQUIRED to fail, which is what turns
#: "zero headroom" from a comment into a measurement.
OVER_CEILING_BINDINGS = 32

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the cylindrical complex fused magnetic pair binds {PACKED_BINDINGS} "
        f"buffers and Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}; "
        f"the signature cannot be built at all")

#: The host record's total size in bytes. Metal aligns ``float2`` to 8 and with
#: ``inc_b`` FIRST there is no internal padding; the nine members total exactly 40
#: (before 2026-09-04 the eight totalled 36 and the struct carried a trailing pad to
#: the same 40). Spelled as data so the plan and the gate size the record from one
#: number.
PARAMS_ITEMSIZE = 40

#: The wall table this family REQUIRES ``zero_metal_axes`` to report on an admitted
#: grid: never r, never phi, z iff z is declared metallic. See the module
#: docstring — it is a consequence of ``Grid``'s ``metallic_axes`` construction,
#: and it is checked rather than assumed because the emitted mask is a function of
#: it and an r clear would destroy the axis row the per-|m| rules own.
FORBIDDEN_WALL_AXES: Tuple[int, ...] = (0, 1)

__all__ = [
    "ARMS", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAMILY",
    "FORBIDDEN_WALL_AXES", "OVER_CEILING_BINDINGS", "PACKED_BINDINGS",
    "PARAMS_ITEMSIZE", "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "CylindricalFusedMagneticPairPlan",
    "certified_cylindrical_curl_body",
    "compile_cylindrical_fused_magnetic_pair",
    "cylindrical_fused_magnetic_pair_coverage",
    "cylindrical_fused_magnetic_pair_source",
    "enumerate_sources",
    "params_record_dtype",
    "plan_cylindrical_fused_magnetic_pair",
    "refuted_separate_scalar_source",
    "refuted_thirty_second_binding",
    "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------
#
# THIRTY-ONE BINDINGS: 18 curl pointers + 12 constitutive pointers + 1 packed
# Params&. See the module docstring for the two signatures this refutes.

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

// THE float2 MEMBER COMES FIRST. Metal aligns float2 to 8 bytes, so with the
// scalars first this struct needs internal padding and the natural host record
// reads `inc_b` one word early — measured on this toolchain by the complex twin,
// and it reads as a plausible complex number rather than as garbage.
struct Params {
    float2 inc_b;
    uint nx; uint ny; uint nz; uint n_elem; uint zrows;
    float dtdx; float minus_dtdx; float axis_coef;
};

kernel void cylindrical_fused_magnetic_pair_step(
    device float2*       f0      [[buffer(0)]],
    device float2*       f1      [[buffer(1)]],
    device float2*       f2      [[buffer(2)]],
    device float2*       u0      [[buffer(3)]],
    device float2*       u1      [[buffer(4)]],
    device float2*       u2      [[buffer(5)]],
    device const float2* g0      [[buffer(6)]],
    device const float2* g1      [[buffer(7)]],
    device const float2* g2      [[buffer(8)]],
    device const float2* pfx     [[buffer(9)]],
    device const float2* c0      [[buffer(10)]],
    device const float2* c2      [[buffer(11)]],
    device const float*  kmx     [[buffer(12)]],
    device const float*  sinvx   [[buffer(13)]],
    device const float*  kmy     [[buffer(14)]],
    device const float*  sinvy   [[buffer(15)]],
    device const float*  kmz     [[buffer(16)]],
    device const float*  sinvz   [[buffer(17)]],
    device float2*       h0      [[buffer(18)]],
    device float2*       h1      [[buffer(19)]],
    device float2*       h2      [[buffer(20)]],
    device float2*       w0      [[buffer(21)]],
    device float2*       w1      [[buffer(22)]],
    device float2*       w2      [[buffer(23)]],
    device const float*  kp0     [[buffer(24)]],
    device const float*  km0     [[buffer(25)]],
    device const float*  kp1     [[buffer(26)]],
    device const float*  km1     [[buffer(27)]],
    device const float*  kp2     [[buffer(28)]],
    device const float*  km2     [[buffer(29)]],
    constant Params&     prm     [[buffer(30)]],
    uint idx [[thread_position_in_grid]])
{
    // THE NINE PACKED ARGUMENTS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN
    // NAMES, once, before any of the lifted text runs. Everything below this line
    // is then character-for-character what `cylindrical_complex.cylindrical_curl_
    // source` and `complex_fields.bloch_constitutive_source` emit, plus the wall
    // clear and the three seam lines.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    uint zrows = prm.zrows;
    float dtdx = prm.dtdx, minus_dtdx = prm.minus_dtdx, axis_coef = prm.axis_coef;
    float2 inc_b = prm.inc_b;

__BODY__
}
"""

#: The marker that separates a certified shader's signature from its body. Both
#: emitters end their parameter list with this exact line, so ONE anchor lifts
#: either body and a template that stopped carrying it raises here rather than
#: splicing a truncated kernel.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: The certified cylindrical curl body's last statement — the B store. The wall
#: clear goes IMMEDIATELY BEFORE it, because the array path stores B and only then
#: overwrites the wall plane; masking the register first is the same field and one
#: fewer pass. Note the ``u`` store precedes it and is NOT masked: ``zero_metal_B``
#: passes ``B_COMPONENTS`` (stepping.py:2250) and the split-field auxiliary is not
#: in it. The per-|m| axis rules, which DO touch ``n``, have already run.
_CURL_STORE = "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;\n"


def certified_cylindrical_curl_body(bcz: int, m_arm: int, expansion: str,
                                    contract: str = shaders.CONTRACT_OFF) -> str:
    """The Dcyl ``step_B`` curl kernel's BODY, lifted from the certified emitter.

    Not a transcription: this is
    :func:`.cylindrical_complex.cylindrical_curl_source`'s own output with the
    ``#include``/helpers/signature preamble and the closing brace removed.
    ``backward`` is False and never a parameter — this pair is the B seam, and
    ``step_D``'s negated strides, its own prefix substitution and its ``Dz`` axis
    clear are a different product on a seam 157 corpus rows already block.
    """
    source = cylindrical_complex.cylindrical_curl_source(
        bcz, False, m_arm, expansion, contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified cylindrical complex curl source no longer carries the "
            "body anchor; this family lifts that body and would otherwise splice a "
            "truncated kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(
            "the certified cylindrical complex curl source does not end with '}'")
    return body[: -len("}\n")]


def cylindrical_fused_magnetic_pair_source(
        bcz: int, m_arm: int, zero_metal: Sequence[bool], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (z boundary, |m| arm, walls, arm, contraction).

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string
    here, for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes a
    source and nothing else, so specialisation by substitution is what keeps the
    emitted arithmetic identical to the bodies this lifts.

    ``zero_rows`` is deliberately NOT a specialisation — it rides in ``Params`` as
    a runtime uniform, exactly as the certified cylindrical curl carries it, which
    is what keeps this enumeration CLOSED at four sources per arm instead of
    ``4 x UNBOUNDED``.
    """
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    for axis in FORBIDDEN_WALL_AXES:
        if zero_metal[axis]:
            raise ValueError(
                f"axis {axis} is reported walled, but on a Dcyl grid `Grid` builds "
                f"metallic_axes as `pair[1] == METALLIC and pair[0] != AXIS` "
                f"(grid.py:512-514) and r's pair is (AXIS, METALLIC) while phi is "
                f"periodic — so neither can be walled. Emitting a clear on the "
                f"r = 0 row would destroy the row the per-|m| rules own")

    curl = certified_cylindrical_curl_body(bcz, m_arm, expansion, contract)
    if _CURL_STORE not in curl:
        raise AssertionError(
            "the certified cylindrical curl body no longer stores the three targets "
            "on one line; the wall clear has no anchor to sit in front of")
    head, tail = curl.split(_CURL_STORE, 1)
    curl = "".join((
        head,
        "    // --- zero_metal_B, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The DIAGONAL for B: Bx on an x wall, By on y, Bz on z. On a Dcyl\n"
        "    // grid only z can be walled, so this is Bz's row and nothing else.\n"
        "    // A COMPLEX zero: the array path assigns one (S:1896, :1902).\n",
        # IMPORTED, never re-spelled: one home for the B-side wall table, and the
        # complex twin's gate already carries a mutation for the plane-wise clear.
        complex_fused_magnetic_pair.zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    # IMPORTED, never re-spelled: `update_H` has no cylindrical branch at all, so
    # the constitutive half of this pair IS the complex twin's, lift and all.
    magnetic = complex_fused_magnetic_pair.certified_constitutive_body(
        expansion, contract)
    # THE SEAM, and it is exactly three lines. The certified complex H body opens
    # each component with `float2 srcN = gN[ii];` — a RELOAD of the complex flux
    # density the curl just stored. Fused, that becomes the live register.
    for target in range(3):
        old = f"    float2 src{target} = g{target}[ii];\n"
        if old not in magnetic:
            raise AssertionError(
                f"the certified complex H constitutive body no longer reads its "
                f"source as {old.strip()!r}; the seam has no anchor")
        magnetic = magnetic.replace(
            old,
            f"    // THE SEAM: the register step_B just wrote, not a reload of "
            f"B{'xyz'[target]}.\n"
            f"    float2 src{target} = v{target};\n")

    body = "".join((
        curl,
        "\n    // --- update_H (stepping.update_H / _apply_constitutive_pml:2065) --\n"
        "    // NO CYLINDRICAL BRANCH: update_H (:907-925) carries none, and the\n"
        "    // cylindrical tranche MEASURED that (480/480 rows, 0 differing words).\n",
        magnetic,
    ))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__BODY__": body,
    })


def compile_cylindrical_fused_magnetic_pair(
        bcz: int, m_arm: int, zero_metal: Sequence[bool], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (z boundary, |m| arm, walls, arm, mode)."""
    return compile_source(cylindrical_fused_magnetic_pair_source(
        bcz, m_arm, zero_metal, expansion,
        contract)).cylindrical_fused_magnetic_pair_step


def enumerate_sources(expansion: str,
                      contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit — SIX, keyed by a stable label.

    z PERIODIC x z METALLIC, and ``M_ZERO`` x ``M_ONE`` x ``M_MANY`` (four until
    2026-09-04, when the m = 0 arm landed). The wall triple is NOT a free axis:
    ``zero_metal_axes`` cannot report a wall on a periodic axis and cannot report
    one on r or phi at all, so the wall flag is a FUNCTION of ``bcz`` and the
    enumeration stays closed. A label that cannot be built is not fingerprinted as
    if it could.
    """
    out: Dict[str, str] = {}
    for bcz in (templates.PERIODIC, templates.METALLIC):
        walls = (False, False, bcz == templates.METALLIC)
        for m_arm in cylindrical_complex.M_ARMS:
            arm_name = cylindrical_complex.M_ARM_LABELS[m_arm]
            label = (f"cylindrical_fused_magnetic_pair_step/bcz{bcz}/{arm_name}/"
                     f"zm{int(walls[2])}/{expansion}")
            out[label] = cylindrical_fused_magnetic_pair_source(
                bcz, m_arm, walls, expansion, contract)
    return out


# ---------------------------------------------------------------------------
# The two refuted signatures
# ---------------------------------------------------------------------------

def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 39-binding separate-scalar signature this platform REFUSES.

    Not shipped and not launchable: it exists so the gate can compile it and
    require the failure, which is what turns :data:`SEPARATE_SCALAR_BINDINGS` from
    an argument into a measurement. This is the signature the certified 27-binding
    cylindrical curl uses, scaled to this pair's thirty pointers — so what it
    measures is precisely "the packing is FORCED by the twelve pointers the fusion
    brings", not that separate scalars are bad style.

    The body touches every buffer: what is being measured is the SIGNATURE, and a
    body the compiler could drop would let dead-code elimination decide the answer.
    """
    written = ["f0", "f1", "f2", "u0", "u1", "u2", "h0", "h1", "h2",
               "w0", "w1", "w2"]
    read_complex = ["g0", "g1", "g2", "pfx", "c0", "c2"]
    read_real = ["kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                 "kp0", "km0", "kp1", "km1", "kp2", "km2"]
    lines: List[str] = []
    slot = 0
    for name in written:
        lines.append(f"    device float2*       {name:<10}[[buffer({slot})]],")
        slot += 1
    for name in read_complex:
        lines.append(f"    device const float2* {name:<10}[[buffer({slot})]],")
        slot += 1
    for name in read_real:
        lines.append(f"    device const float*  {name:<10}[[buffer({slot})]],")
        slot += 1
    for name in ("nx", "ny", "nz", "n_elem", "zrows"):
        lines.append(f"    constant uint&       {name:<10}[[buffer({slot})]],")
        slot += 1
    for name in ("dtdx", "minus_dtdx", "axis_coef"):
        lines.append(f"    constant float&      {name:<10}[[buffer({slot})]],")
        slot += 1
    lines.append(f"    constant float2&     inc_b     [[buffer({slot})]],")
    slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - a construction invariant
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    body = "\n".join(f"    {name}[idx] = {name}[idx] * dtdx;" for name in written)
    touch_real = " + ".join(f"{name}[0]" for name in read_real)
    touch_complex = " + ".join(f"{name}[0].x" for name in read_complex)
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "kernel void refuted_separate_scalar(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float touch = ({touch_real}) + ({touch_complex});",
        "    touch = touch + inc_b.x + minus_dtdx + axis_coef;",
        "    f0[idx] = f0[idx] + touch * float(nx + ny + nz + zrows);",
        body,
        "}",
        "",
    ))


def refuted_thirty_second_binding(contract: str = shaders.CONTRACT_OFF) -> str:
    """The shipped signature PLUS one pointer: 32 bindings, which must NOT compile.

    THIS IS THE MEASUREMENT BEHIND "ZERO HEADROOM". :data:`PACKED_BINDINGS` is 31
    and the platform's highest legal buffer index is 30, so this family sits
    exactly at the ceiling and one more binding of any kind — an inverse-mu volume,
    a second prefix, a third i*m/r row — puts it over. The gate compiles this and
    requires the failure, so a later widening is refused by a compiler on this
    toolchain rather than by a sentence in this docstring.

    The extra pointer is named ``over_ceiling`` and stands for whatever a future
    widening would add; nothing about the refusal depends on what it holds.
    """
    pointers = [f"    device float2* p{n:<2} [[buffer({n})]]," for n in range(31)]
    if len(pointers) + 1 != OVER_CEILING_BINDINGS:  # pragma: no cover - invariant
        raise AssertionError((len(pointers) + 1, OVER_CEILING_BINDINGS))
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "struct Params { float2 inc_b; uint n_elem; float dtdx; };",
        "kernel void over_ceiling(",
        *pointers,
        f"    constant Params& prm [[buffer({OVER_CEILING_BINDINGS - 1})]],",
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= prm.n_elem) { return; }",
        "    p0[idx] = p1[idx] * prm.dtdx + prm.inc_b;",
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def cylindrical_fused_magnetic_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span Dcyl ``step_B`` -> wall -> ``update_H``?

    A conjunction of the two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said
    it. That is the construction every shipped pair uses, and it is what makes the
    expansion-probe clause, the complex-storage clause (the partition against the
    real m = 0 product), the nr >= 2 clause and the Dcyl geometry clauses INHERITED
    rather than restated.
    """
    reasons: List[str] = []

    curl = cylindrical_complex.cylindrical_complex_pml_curl_coverage(
        fields, pml, CURL_SUB_STEP, residency, probe)
    if not curl.covered:
        reasons.extend(f"cylindrical curl half: {reason}" for reason in curl.reasons)
    magnetic = cylindrical_complex.cylindrical_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency, probe)
    if not magnetic.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in magnetic.reasons)

    # THE SOURCE SEAM. A magnetic source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B.
    # MEASURED COST ON THE CORPUS: ZERO — all sixteen cylindrical rows declare
    # electric sources only, which is what makes this cell the only large one with
    # no source attrition. It is checked anyway: a corpus is not a contract.
    # IGNORANCE IS NEVER AN EMPTY SET — `Fields` does not hold the source list, so
    # an undeclared `sources` is a refusal and not an assumed ().
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the "
            f"driver injects it BETWEEN step_B and update_H "
            f"(driver.py:3283-3284), which is work inside the seam this "
            f"dispatch closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO SYMMETRY FILLS. `Grid` refuses a mirror plane on a Dcyl cell outright
    # (grid.py:648-653) and both halves restate that refusal, so on this family the
    # fills are dead by construction rather than by clause. RE-CHECKED HERE ANYWAY,
    # not inferred: this module's coverage may not be read off another module's
    # guard, and the cost is one attribute read.
    mirrored = getattr(grid, "is_mirrored", None)
    if callable(mirrored):
        for axis in range(3):
            if bool(mirrored(axis)):
                reasons.append(
                    f"axis {axis} is folded: stepping.fill_symmetry_bc_B and "
                    f"stepping.fill_folded_far_ghosts_B both run inside this seam "
                    f"(driver.py:3285-3287) and neither is carried by this family")

    # zero_metal_B is CARRIED INLINE, so the grid must be able to answer which axes
    # are walled. A grid that cannot answer would silently be treated as unwalled,
    # which is a plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")

    # THE WALL TABLE MUST BE THE Dcyl ONE. On a cylindrical grid `Grid` cannot
    # report r or phi as walled (module docstring), and the emitted mask is a
    # FUNCTION of this table — a clear on the r = 0 row would overwrite the row the
    # per-|m| rules own with a zero the array path never writes. Refused by name
    # rather than emitted, because `zero_metal_mask` would happily emit it.
    if all(getattr(grid, name, None) is not None
           for name in ("has_metallic", "is_metallic", "is_mirrored")):
        walls = zero_metal_axes(grid)
        for axis in FORBIDDEN_WALL_AXES:
            if walls[axis]:
                reasons.append(
                    f"zero_metal_axes reports axis {axis} walled on a grid this "
                    f"family admits; on a Dcyl cell neither r nor phi can be "
                    f"(grid.py:512-514), and the r = 0 clear this would emit is a "
                    f"row the per-|m| axis rules own")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class CylindricalFusedMagneticPairPlan(KernelPlan):
    """ONE dispatch that performs three driver passes, complex B never leaving a
    register — plus the host prefix pre-pass the cylindrical family cannot avoid.

    THE ONE DIVERGENCE FROM :class:`.plans.KernelPlan`, and it is the certified
    cylindrical curl plan's, unchanged: the radial prefix is a SEQUENTIAL float32
    host scan whose summation order defines the answer, so :meth:`run` performs a
    pre-pass — ``sync_out`` of ``Ey``, the scan, ``sync_in`` of the prefix mirror —
    and only then calls ``super().run``. The dict lookup, the variant refusal and
    the launch counter stay the base's.

    ``launches_per_run`` is the base's 1: the pre-pass is host work and residency
    traffic, not a kernel launch, so the whole-step arbiter's per-cycle launch
    assertion needs no special case. :attr:`prefix_syncs` counts the pre-pass
    separately, so a gate can assert it HAPPENED rather than trust that it did — a
    leg that certified this plan without the counter could pass on a prefix
    computed once and never refreshed.

    ``runs`` is an alias of ``launches`` rather than a second counter, because with
    one dispatch per run the two ARE the same number.
    """

    __slots__ = ("residency", "volumes", "bcz", "m_arm", "zero_rows", "zero_metal",
                 "shape", "n_elem", "dtdx", "expansion", "params", "scratch",
                 "prefix_syncs", "_prefix_host", "_prefix_name",
                 "_prefix_source_name", "_source_map", "_xp")

    family = "cylindrical complex fused PML B-curl/stored-H pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "bcz", "m_arm", "zero_rows", "zero_metal", "expansion")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Any, volumes: Sequence[str], bcz: int,
                 m_arm: int, zero_row_count: int, zero_metal: Sequence[bool],
                 shape: Sequence[int], dtdx: float, expansion: str, params: Any,
                 pointers: Sequence[Any], functions: Mapping[str, Any],
                 prefix_host: Any, prefix_name: str, prefix_source_name: str,
                 source_map: Mapping[str, Any], xp: Any,
                 scratch: Any = None) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.bcz = int(bcz)
        self.m_arm = int(m_arm)
        self.zero_rows = int(zero_row_count)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        # The COMPLEX CELL count, which is also what the dispatch is sized from: a
        # complex64 tensor's element count is its cell count. Binding a float32
        # word view instead would size the grid at twice the cells.
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        # float(): the array path multiplies a complex64 volume by a Python float
        # and the packed struct holds numpy.float32 of the same value, so the two
        # scalars are the same bits.
        self.dtdx = float(dtdx)
        self.expansion = str(expansion)
        self.params = params
        self.scratch = scratch
        self.prefix_syncs = 0
        self._prefix_host = prefix_host
        self._prefix_name = str(prefix_name)
        self._prefix_source_name = str(prefix_source_name)
        self._source_map = dict(source_map)
        self._xp = xp
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches

    def refresh_prefix(self) -> None:
        """Pull the prefix SOURCE off the device, scan it on the host, push it back.

        Separated from :meth:`run` so a gate can call it explicitly and count it,
        and so the residency traffic this family costs has a name. It is NOT an
        optional step: :meth:`run` always calls it. A ``prepare()`` a caller had to
        remember is exactly the stale-mirror class this package refuses to comment
        on — a forgotten call is a smooth, plausible, wrong field.
        """
        self.residency.sync_out((self._prefix_source_name,))
        prefix = cylindrical_complex.cylindrical_prefix(
            self._xp, CURL_SUB_STEP, self._source_map, scratch=self.scratch)
        self._prefix_host[...] = prefix
        self.residency.sync_in((self._prefix_name,))
        self.prefix_syncs += 1

    def run(self, contract: Optional[str] = None) -> None:
        """Refresh the prefix, then take the BASE's launch path unchanged."""
        self.refresh_prefix()
        super().run(contract)


def params_record_dtype() -> Any:
    """The host record's dtype — the ``float2`` member FIRST, itemsize 40.

    ONE HOME FOR THE LAYOUT, because getting it wrong is a silent wrong answer
    rather than a crash: see the module docstring for the complex twin's
    measurement. The offsets are stated EXPLICITLY even though the float2-first
    order makes them the natural ones, so the record cannot drift into agreeing
    with Metal only by accident; ``itemsize`` is stated so the record and the
    struct are sized from one number. ``axis_coef`` (the m = 0 on-axis Dz
    coefficient, unread on this B/H seam but carried because the lifted curl body
    names it) is the ninth member, at offset 36.
    """
    import numpy as np  # noqa: PLC0415

    return np.dtype({
        "names": ["inc_b", "nx", "ny", "nz", "n_elem", "zrows", "dtdx",
                  "minus_dtdx", "axis_coef"],
        "formats": [("<f4", 2), "<u4", "<u4", "<u4", "<u4", "<u4", "<f4", "<f4",
                    "<f4"],
        "offsets": [0, 8, 12, 16, 20, 24, 28, 32, 36],
        "itemsize": PARAMS_ITEMSIZE,
    })


def _params_tensor(shape: Sequence[int], dtdx: float, zero_row_count: int,
                   minus_dtdx: float, inc_b: Sequence[float],
                   device: str, axis_coef: float) -> Any:
    """The nine non-pointer arguments as one 40-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason the complex
    twin's ``_params_tensor`` gives: :meth:`.Residency.mirror` binds float32 and
    complex64 volumes and refuses anything else BY NAME, because a wider element
    silently reinterprets. This record is neither — it is a packed struct of one
    float2, five uints and three floats that nothing on the host reads back and
    nothing on the device writes — so it is built here, once, at plan time, and
    held by the plan. Nothing on the launch path allocates.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["zrows"] = int(zero_row_count)
    record["dtdx"] = np.float32(dtdx)
    record["minus_dtdx"] = np.float32(minus_dtdx)
    record["axis_coef"] = np.float32(axis_coef)
    record["inc_b"] = (np.float32(inc_b[0]), np.float32(inc_b[1]))
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_cylindrical_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None,
        ) -> Optional[CylindricalFusedMagneticPairPlan]:
    """Build the Dcyl fused B/H plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl`
    gives: a configuration this kernel does not carry must fall back to the array
    path, never raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken
    copy of the shipped source and hands it here; dropping the argument is not a
    silent slowdown, it is a silent DISARMING — every mutation leg would then
    launch the shipped kernel and report the defect as uncaught.
    """
    if not cylindrical_fused_magnetic_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    expansion = cylindrical_complex.expansion_from_probe(
        probe if probe is not None
        else cylindrical_complex.load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None

    import numpy  # noqa: PLC0415

    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    spec = SUB_STEPS[CURL_SUB_STEP]
    side = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    kinds = _boundary_kinds(grid, pml)
    bcz = (templates.METALLIC if kinds[2] == "metallic" else templates.PERIODIC)
    m = int(grid.m)
    arm = cylindrical_complex.m_class(m)
    zero_row_count = cylindrical_complex.zero_rows(
        m, bool(grid.accurate_fields_near_cylorigin))
    walls = zero_metal_axes(grid)
    dtdx = grid.dt / grid.dx
    minus_dtdx, inc_b = cylindrical_complex.axis_increment_scalars(m, dtdx)
    axis_coef = cylindrical_complex.axis_coefficient(dtdx)

    volumes: List[str] = []

    def bind_complex(name: str, host: Any) -> Any:
        volumes.append(name)
        return complex_fields._complex_mirror(residency, name, host)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step,
    # so the groups are built in the order the kernel declares them.
    flux = [bind_complex(name, getattr(fields, name)) for name in spec["targets"]]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in spec["targets"]]
    electric = [bind_complex(name, getattr(fields, name))
                for name in spec["sources"]]

    # THE PREFIX BUFFER IS THE PLAN'S OWN and is allocated once. It must be a
    # stable host array: `Residency.mirror` refuses re-registering a name against a
    # DIFFERENT host array, and `cylindrical_rderiv_prefix` returns a fresh array
    # every call. So the scan's result is copied INTO this buffer rather than the
    # buffer being replaced, which is also what keeps the device allocation stable.
    prefix_host = numpy.zeros(
        cylindrical_complex.prefix_shape(CURL_SUB_STEP, grid.shape),
        dtype=numpy.complex64)
    prefix_name = f"cylfused:pfx:{CURL_SUB_STEP}"
    prefix = bind_complex(prefix_name, prefix_host)

    # The two i*m/r rows are GRID INVARIANTS — Yee shift, radial extent, m, Courant
    # — so they are built ONCE here, exactly as `stepping`'s own scratch.constant
    # cache builds them once per run (:713-723).
    imr_rows = []
    for index, _register, sign in cylindrical_complex.IMR_TERMS[CURL_SUB_STEP]:
        target = spec["targets"][index]
        row = numpy.ascontiguousarray(
            cylindrical_complex.imr_coefficient_row(
                numpy, target, sign, m, dtdx, int(grid.shape[0]),
                numpy.complex64).reshape(-1))
        imr_rows.append(bind_complex(f"cylfused:imr:{CURL_SUB_STEP}:{target}", row))

    # THE B CURL TAKES THE HALF-INTEGER LATTICE and the H constitutive the INTEGER
    # one (SUB_STEPS['step_B']['suffix'] == '_h'; CONSTITUTIVE_SIDES['H']
    # ['half_integer'] is False). That is the OPPOSITE pairing to the D/E pair, and
    # the kernel cannot tell — the gate carries a host mutation for each group.
    curl_coefficients = [
        bind_real(f"pml:{stem}_{axis}{spec['suffix']}",
                  getattr(pml, f"{stem}_{axis}{spec['suffix']}"))
        for axis in "xyz" for stem in ("kms", "sinv")]

    magnetic = [bind_complex(name, getattr(fields, name))
                for name in side["targets"]]
    workspace = [bind_complex(name, getattr(fields, name)) for name in side["aux"]]
    constitutive_coefficients = [
        bind_real(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"))
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + electric + [prefix] + imr_rows
                + curl_coefficients + magnetic + workspace
                + constitutive_coefficients)
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_cylindrical_fused_magnetic_pair(
                bcz, arm, walls, expansion, mode)

    # The residency layer DECLARES its device; sniffing it off a bound tensor would
    # read "mps:0" where the mirrors were built with "mps" and put the struct on a
    # nominally different device from the volumes it describes.
    return CylindricalFusedMagneticPairPlan(
        residency, volumes, bcz, arm, zero_row_count, walls, grid.shape, dtdx,
        expansion,
        _params_tensor(grid.shape, dtdx, zero_row_count, minus_dtdx, inc_b,
                       residency.device, axis_coef),
        pointers, selected, prefix_host, prefix_name,
        cylindrical_complex.PREFIX[CURL_SUB_STEP]["component"],
        {name: getattr(fields, name) for name in spec["sources"]}, numpy,
        getattr(fields, "scratch", None))


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"cylindrical fused magnetic pair cannot fill {slot}",))
    return cylindrical_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[CylindricalFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_cylindrical_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY. ``plan_step`` assigns at most one arm
    per slot and this product spans three, so there is no slot it could claim
    without a composition rule nothing has measured — the same open question every
    fused pair records. Registering unwired keeps it ENUMERABLE for the
    disjointness sweep (``arms.registered``) while ``arms.arms_for`` skips it, so
    ``plan_step`` cannot select it and no existing arm's selection changes.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "cylindrical complex fused magnetic B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="cylindrical complex fused magnetic B/H pair: ",
                          noun="cylindrical complex fused PML B-curl/stored-H pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
