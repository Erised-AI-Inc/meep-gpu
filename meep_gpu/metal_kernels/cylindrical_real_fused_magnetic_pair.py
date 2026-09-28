"""The Dcyl m = 0 magnetic seam in one dispatch: ``step_B`` welded into ``update_H``.

THE METAL TWIN of :mod:`meep_gpu.triton_kernels.cylindrical_real_fused_magnetic_pair`.
Same seam, same three carried driver passes, same two provably-inert ones; the source
is SPLICED FROM THIS BACKEND'S OWN CERTIFIED EMITTERS rather than ported from the
Triton text, so the two products are siblings and not a translation.

Its sibling :mod:`.cylindrical_fused_magnetic_pair` takes the |m| >= 1 complex Dcyl
arm; this file takes the m = 0 real one, and the two row sets are disjoint —
measured on the Metal census, 3 rows against 16, intersection empty.

EVERYTHING HERE IS LIFTED, not retyped:

* the cylindrical B curl — :func:`.cylindrical_real.cylindrical_curl_source` with
  ``sub_step='step_B'``, whose body :func:`certified_cyl_curl_body` takes verbatim.
  Its ghost gather is :func:`.templates.ghost`, its cell-0 mask
  :func:`.templates.ownership_mask`, its ``Bz`` substitution the extended radial
  prefix's forward difference and its tail the m = 0 axis rule ``Bx[r=0] = 0``;
* the wall clear — ``stepping._zero_metal`` restricted to ``B_COMPONENTS``, through
  :func:`.fused_magnetic_pair.zero_metal_mask` and its ``_ZERO_METAL_ROWS``, which
  are IMPORTED rather than re-spelled. The B table is the Yee DIAGONAL (Bx on an x
  wall, By on y, Bz on z) and the D table is its complement; a fused pair that
  reused the wrong one would clear the wrong two components on every walled run;
* the constitutive half — :func:`.fused_magnetic_pair.certified_constitutive_body`,
  itself :func:`.shaders.constitutive_source` on side ``"H"`` with its decode
  prologue dropped and its three targets renamed ``f`` -> ``h``. IMPORTED for the
  same reason: one lift, one place to check.

Because every half is LIFTED, "the fused arithmetic is the certified arithmetic" is a
property of the construction a reader can check by running the two body functions —
and the gate's ``transcription`` leg checks exactly that, because a construction is a
hypothesis until something compares the strings.

===========================================================================
WHAT IT IS WORTH — measured before it was built, on BOTH backends' censuses
===========================================================================

``parity/meep_gpu/results/fusion_matrix_metal_2026-08-20`` ranks the cells the
186-row corpus drives that no fused product covers, by REACHABLE demand::

    reach 3  (of 3 rows)  B_to_H  (cylindrical m=0, cylindrical m=0)
        pointers 16 + 18 -> 28..28 (ceiling 30)   FITS -- NOT BUILT

Re-measured against the census this file prices against
(``parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19``)::

    admit cylindrical_real_curl@step_B                          3 rows
    admit cylindrical_real_constitutive@update_H                3 rows
    admit BOTH halves                                           3 rows
    ... and declare NO MAGNETIC source                          3 rows

3 -> 3 -> 3 -> 3. NO CLAUSE IN THE LADDER CUTS ANYTHING, which is rare on this board:
the three rows (``tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_0_0`` and
``_1_0``, ``tests:TestPMLCylindrical.test_pml_cyl_0_0_0``) all declare an ELECTRIC
source only, so the magnetic seam is empty on every one. The TRITON census funnels to
the SAME three rows through its own independently written predicates.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

The driver runs five passes between ``step_B`` and ``update_H``
(driver.py:3281-3289, mirrored in :data:`.coverage.RESIDENCY_ORDER`)::

    step_B -> magnetic sources -> fill_symmetry_bc_B -> zero_metal_B
           -> fill_folded_far_ghosts_B -> update_H

* **the magnetic sources — REFUSED BY NAME.** ``driver.step`` injects them between
  the halves (driver.py:3283-3284), so a fused pair would consume a pre-injection
  ``B``. Ignorance is never an empty set: ``Fields`` does not hold the source list,
  so an undeclared ``sources`` is a REFUSAL and not an assumed ``()``. On this cell
  the clause costs nothing — all three corpus rows are electric-only — and it is
  still checked, because "the corpus happens not to" is not a kernel property.
* **``fill_symmetry_bc_B`` — PROVABLY INERT, not carried.**
  ``stepping._fill_symmetry_ghost_cells`` (:1437-1439) returns before touching a
  single cell unless ``grid.has_symmetry()``, and both halves' predicate refuses a
  mirror plane. A pass that cannot execute is not a pass this dispatch has to
  absorb — and that is a different statement from "refused", which is what the
  folded families do because there the fill does real work.
* **``zero_metal_B`` — CARRIED INLINE**, and NOT vacuous on the rows this product
  is priced against: all three declare a metallic z, so the ``Bz`` row fires on
  every one of them.

  IT IS NOT UNIVERSAL ON THIS BACKEND, AND THAT IS A MEASURED DIVERGENCE FROM THE
  TRITON TWIN. ``triton_kernels/cylindrical_triton.py`` pins the Dcyl declaration
  to ``('axis', 'periodic', 'metallic')`` and refuses anything else by name;
  :mod:`.cylindrical_real` PARAMETERISES the z ghost rule and admits a PERIODIC z
  as well, with its own certified gate covering two such rows. Measured here:
  ``matrix.cylindrical(m=0, complex_storage=False, z_kind='periodic')`` is admitted
  by this predicate and plans, with ``zero_metal_axes`` all false — on which the
  wall clear emits a comment and nothing else, and ``zero_metal_B`` is not even in
  the composer's live pass set. So this family is BROADER than its Triton twin, the
  extra breadth is worth zero seam-instances on the measured corpus, and the gate
  carries a z-periodic case so the empty-clear path is executed rather than
  assumed.
* **``fill_folded_far_ghosts_B`` — PROVABLY INERT, not carried**, at the same line
  and for the same reason (:1516-1517), plus it needs an axis whose stored extent
  exceeds its owned one, which only a folded periodic axis has.

So :data:`REPLACES` names THREE passes and not five, and the predicate is what
guarantees the other two cannot run.

===========================================================================
THE SIGNATURE — 28 pointers plus one packed struct
===========================================================================

Counting what an all-three-component fused cylindrical B/H dispatch needs::

    3 B + 3 fu_B + 3 E + 1 prefix + 6 curl coefficients (kms/sinv per axis)  = 16
  + 3 H + 3 f_w_H + 6 constitutive coefficients (kps/kms per axis)           = 12
                                                                            ---- 28
  + 6 scalars (nx, ny, nz, n_elem, dtdx, axis_coef)                          = 34

and :data:`.device.MAX_BUFFER_BINDINGS` is 31. There is NO inverse-mu volume: the H
product is ``source = B`` with mu = 1 baked into the array path too
(``stepping.update_H`` passes ``fields.Bx``), which is why the constitutive half
brings 12 pointers and not the certified kernel's 18.

34 separate bindings is a COMPILE ERROR — buffer attribute indices must be 0..30.
Packing the six scalars into one ``constant Params&`` buys back five slots and lands
at :data:`PACKED_BINDINGS` = 29. Both halves of that are MEASURED by the gate's
``binding_ceiling`` leg: the 34-binding signature must FAIL and the packed one must
COMPILE, LAUNCH and read every field back.

``axis_coef`` RIDES IN THE STRUCT AND IS ZERO ON THIS SIDE. The certified curl binds
it for both sub-steps so the layout is one layout; the B-side body never reads it
(the m = 0 D increment is the D side's). It is carried rather than dropped because
dropping it would fork the lifted body, and the lift is the whole point.

===========================================================================
THE PREFIX IS UPSTREAM OF THE SEAM, and the fusion is the SECOND launch
===========================================================================

The cylindrical curl reads a radial prefix scan, and on THIS backend that scan runs
ON THE DEVICE — :mod:`.cylindrical_real` inverts the Triton module's refusal with a
measurement (this engine's host module is NumPy, whose ``cumsum`` IS byte-equal to a
sequential accumulation, so a column-serial device scan reproduces the oracle by
construction). That decision is unchanged here.

What matters for FUSION is only that the scan runs BEFORE the curl, so no dispatch
straddles it. This plan is therefore TWO dispatches per run — the scan, then the
fused curl+wall+constitutive — where the separate composition is THREE. ``launches``
counts kernel launches and not ``run`` calls, and :attr:`prefix_launches` and
:attr:`fused_launches` are carried apart so "the scan ran and the fused kernel did
not" is distinguishable from "the plan ran once".

NOT WIRED, for the reason the other fused pairs are not. ``STEP_ORDER`` assigns at
most one arm per slot and this product spans three, so there is no slot it can claim
without a composition rule nothing has measured; it registers ``wired=False``. The
composer cannot select it, the disjointness sweep can still enumerate it, and no
existing arm's behaviour changes. ``meep_gpu.fastpath.plan_fast_path`` still returns
``None`` on every branch.

DEVICE STATUS: **RELEASED 2026-08-20** on this host (Apple GPU via MPS, torch 2.10.0,
    metalfe-32023.850.10) under the ``flush`` subnormal policy the backend can honour,
    with the subnormal-free precondition checked on every step —
    ``parity/meep_gpu/results/metal_cylindrical_real_fused_magnetic_pair_2026-08-20b/``:

    * **6/6 product cases, 12 complete steps each, bit-identical** on the uint32 view
      of all 24 stored volumes to the array path, with the dispatch counts asserted
      per cycle (one scan, one fused dispatch);
    * **2/2 separate controls**: the array path, the two ALREADY CERTIFIED Metal
      products, and this one, all three word-for-word equal at every step. THE
      FUSION, MEASURED: three dispatches per step become two, and on a walled run the
      separate side leaves ``zero_metal_B`` on the HOST where this side carries it;
    * the BINDING CEILING measured both ways: the 34-binding separate-scalar
      signature FAILS to compile ("'buffer' attribute parameter is out of bounds"),
      and the packed 29-binding one compiles, launches and reads every struct field
      back;
    * the two omitted driver passes EXECUTED and measured inert on every admitted
      case, with ``zero_metal_B`` as the control that must move something;
    * **21 armed defects, 20 caught and 1 a CONFIRMED null**, plus a byte-neutral
      control that must NOT diverge and a disarm row on the shipped bytes.

    THE ONE NULL IS A MEASUREMENT. Flattening ``dtdx * ((c_y - c) + (b - b_z))`` is a
    different float32 number on a Cartesian grid; here ``c_y`` IS ``c`` — phi is the
    one-cell invariant axis — so the flattened form differs only where ``(b - b_z)``
    is ``-0.0``, and it came back 0 differing words. Its SIBLING on target 1, whose
    pair is (z, r) and both live, is armed as must-catch and diverges by 312 words.
    That pairing is what stops the null reading as "the parenthesisation does not
    matter here".

    An earlier artifact of the same gate against the pre-amendment bytes of this file
    is kept at ``..._2026-08-20/``; the ``b`` directory is the one whose digests match
    what shipped.

    STILL NOT WIRED. A weld licenses the claim, not a dispatch: nothing in a default
    run reaches this plan.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage, MAGNETIC_FIELD_TYPE
from . import shaders, templates
from .coverage import zero_metal_axes
from .cylindrical_real import (
    METALLIC,
    boundary_codes,
    compile_cylindrical_prefix,
    cylindrical_curl_source,
    cylindrical_real_constitutive_coverage,
    cylindrical_real_curl_coverage,
    prefix_row_vectors,
    scan_rows,
    scratch_mirror,
)
from .device import Residency, compile_source
from .fused_magnetic_pair import (
    _CURL_STORE,
    _ZERO_METAL_ROWS,
    certified_constitutive_body,
    zero_metal_mask,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? While False the
#: source-presence clause below refuses every in-seam deposit, exactly as it did when the
#: clause was written out here by hand. Flipping it is a claim about the PLAN this module
#: builds -- that the leading slot saves and the trailing slot restores -- and is only
#: ever changed in the same edit as that wiring. See ``deposit_repair`` for why.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "cylindrical_real_fused_magnetic_pair"

#: The sub-step slot this arm holds a row on. It spans three; a refusal is NAMED on
#: the slot the fusion starts at rather than being invisible to the table.
SLOT = "step_B"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_B"
CONSTITUTIVE_SIDE = "H"

#: The driver passes ONE fused dispatch performs, in driver order
#: (driver.py:3281-3289). THREE, not five: see :data:`INERT_PASSES`.
REPLACES: Tuple[str, ...] = ("step_B", "zero_metal_B", "update_H")

#: The two driver passes this product does NOT carry, with the reason each cannot
#: execute on a configuration the predicate admits. Named as data so the gate can
#: assert them inert rather than assume it.
INERT_PASSES: Dict[str, str] = {
    "fill_symmetry_bc_B": ("stepping._fill_symmetry_ghost_cells:1437-1439 returns "
                           "unless grid.has_symmetry(); both halves' predicate "
                           "refuses a mirror plane"),
    "fill_folded_far_ghosts_B": ("stepping._fill_folded_far_ghosts:1516-1517 returns "
                                 "unless grid.has_symmetry(), and additionally needs "
                                 "an axis whose stored extent exceeds its owned one"),
}

#: How many bindings the all-three-component shape needs with the six scalars bound
#: SEPARATELY, and how many with them packed. Spelled as data so the gate compiles
#: the refuted signature from the same number this docstring argues from.
SEPARATE_SCALAR_BINDINGS = 34
PACKED_BINDINGS = 29

__all__ = [
    "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAMILY", "INERT_PASSES",
    "PACKED_BINDINGS", "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "MetalCylindricalRealFusedMagneticPairPlan", "certified_cyl_curl_body",
    "compile_cylindrical_real_fused_magnetic_pair",
    "cylindrical_real_fused_magnetic_pair_source",
    "metal_cylindrical_real_fused_magnetic_pair_coverage",
    "plan_metal_cylindrical_real_fused_magnetic_pair",
    "refuted_separate_scalar_source",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params {
    uint nx; uint ny; uint nz; uint n_elem; float dtdx; float axis_coef;
};

kernel void cyl_real_fused_magnetic_pair_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device const float* pfx     [[buffer(9)]],
    device const float* kmx     [[buffer(10)]],
    device const float* sinvx   [[buffer(11)]],
    device const float* kmy     [[buffer(12)]],
    device const float* sinvy   [[buffer(13)]],
    device const float* kmz     [[buffer(14)]],
    device const float* sinvz   [[buffer(15)]],
    device float*       h0      [[buffer(16)]],
    device float*       h1      [[buffer(17)]],
    device float*       h2      [[buffer(18)]],
    device float*       w0      [[buffer(19)]],
    device float*       w1      [[buffer(20)]],
    device float*       w2      [[buffer(21)]],
    device const float* kp0     [[buffer(22)]],
    device const float* km0     [[buffer(23)]],
    device const float* kp1     [[buffer(24)]],
    device const float* km1     [[buffer(25)]],
    device const float* kp2     [[buffer(26)]],
    device const float* km2     [[buffer(27)]],
    constant Params&    prm     [[buffer(28)]],
    uint idx [[thread_position_in_grid]])
{
    // THE SIX SCALARS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES, once,
    // before any of the lifted text runs. Everything below this line is then
    // character-for-character what `cylindrical_real.cylindrical_curl_source` and
    // `shaders.constitutive_source` emit, plus the wall clear and the one seam
    // substitution.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float axis_coef = prm.axis_coef;
    (void)axis_coef;   // bound for one layout; the B-side body does not read it.

__BODY__
}
"""

#: The marker that separates a certified shader's signature from its body. The
#: cylindrical curl template and the certified constitutive template both end their
#: parameter list with this exact line, so ONE anchor lifts either body and a
#: template that stopped carrying it raises here rather than splicing a truncated
#: kernel.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"


def certified_cyl_curl_body(codes: Sequence[int],
                            contract: str = shaders.CONTRACT_OFF) -> str:
    """The Dcyl ``step_B`` curl kernel's BODY, lifted from the certified emitter.

    Not a transcription: this is :func:`.cylindrical_real.cylindrical_curl_source`'s
    own output with the ``#include``/signature preamble and the closing brace
    removed. ``sub_step`` is pinned to ``step_B`` and is never a parameter — this
    pair is the magnetic seam, and ``step_D``'s backward differences, its prefix
    substitution on ``Dz`` and its two-rule axis tail are a different product whose
    seam carries an electric source on 157 of the corpus's 186 rows.
    """
    source = cylindrical_curl_source(CURL_SUB_STEP, codes, contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified cylindrical curl source no longer carries the body "
            "anchor; this family lifts that body and would otherwise splice a "
            "truncated kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError(
            "the certified cylindrical curl source does not end with '}'")
    return body[: -len("}\n")]


def cylindrical_real_fused_magnetic_pair_source(
        codes: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundary triple, walls, contraction mode).

    ``codes`` is the per-axis PERIODIC/METALLIC pair triple
    :func:`.cylindrical_real.boundary_codes` resolves, with the r axis pinned
    METALLIC — the certified curl emitter refuses anything else by name, and this
    function refuses it again before the splice so a caller cannot reach a kernel
    whose far r ghost wraps around the cylinder.

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string
    here, for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes a
    source and nothing else, so specialisation by substitution is what keeps the
    emitted arithmetic identical to the bodies this lifts.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if codes[0] != METALLIC:
        raise ValueError(
            f"the r axis must compile as METALLIC (code {METALLIC}), got {codes[0]!r}: "
            f"CYL_AXIS shares _shift_up's metallic zero ghost exactly "
            f"(stepping.py:1828-1830), and a PERIODIC r axis would wrap the far face "
            f"onto the axis row")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")

    curl = certified_cyl_curl_body(codes, contract)
    if _CURL_STORE not in curl:
        raise AssertionError(
            "the certified cylindrical curl body no longer stores the three targets "
            "on one line; the wall clear has no anchor to sit in front of")
    head, tail = curl.split(_CURL_STORE, 1)
    curl = "".join((
        head,
        "    // --- zero_metal_B, carried inline (stepping._zero_metal:2232) ------\n"
        "    // The DIAGONAL for B: Bx on an x wall, By on y, Bz on z. AFTER the\n"
        "    // m = 0 axis rule above and BEFORE the store, which is the driver's\n"
        "    // own order (step_B ends with the axis rules, then zero_metal_B).\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    magnetic = certified_constitutive_body(contract)
    # THE SEAM, and it is exactly three lines. The certified H body opens each
    # component with `float srcN = gN[ii];` — a RELOAD of the flux density the curl
    # just stored. Fused, that becomes the live register. `gN` in THIS kernel is the
    # curl's electric source, so leaving the reload standing would read the wrong
    # volume rather than merely re-reading the right one.
    for target in range(3):
        old = f"    float src{target} = g{target}[ii];\n"
        if old not in magnetic:
            raise AssertionError(
                f"the certified H constitutive body no longer reads its source as "
                f"{old.strip()!r}; the seam has no anchor")
        magnetic = magnetic.replace(
            old,
            f"    // THE SEAM: the register step_B just wrote, not a reload of "
            f"B{'xyz'[target]}.\n"
            f"    float src{target} = v{target};\n")

    body = "".join((
        curl,
        "\n    // --- update_H (stepping.update_H / _apply_constitutive_pml:2083) --\n",
        magnetic,
    ))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__BODY__": body,
    })


def compile_cylindrical_real_fused_magnetic_pair(
        codes: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, walls, mode)."""
    return compile_source(cylindrical_real_fused_magnetic_pair_source(
        codes, zero_metal, contract)).cyl_real_fused_magnetic_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 34-binding separate-scalar signature this platform REFUSES.

    Not shipped and not launchable: it exists so the gate can compile it and require
    the failure, which is what turns :data:`SEPARATE_SCALAR_BINDINGS` from an
    argument into a measurement. The body touches every buffer, because a body the
    compiler could drop would let dead-code elimination decide the answer.
    """
    written = ["f0", "f1", "f2", "u0", "u1", "u2", "h0", "h1", "h2",
               "w0", "w1", "w2"]
    read = ["g0", "g1", "g2", "pfx",
            "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
            "kp0", "km0", "kp1", "km1", "kp2", "km2"]
    lines: List[str] = []
    slot = 0
    for name in written:
        lines.append(f"    device float*       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in read:
        lines.append(f"    device const float* {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in ("nx", "ny", "nz", "n_elem"):
        lines.append(f"    constant uint&      {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in ("dtdx", "axis_coef"):
        lines.append(f"    constant float&     {name:<8}[[buffer({slot})]],")
        slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    touch = " + ".join(f"{name}[0]" for name in read)
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        templates.contraction_pragma(contract),
        "",
        "kernel void refuted_separate_scalars(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float touch = {touch};",
        "    f0[idx] = f0[idx] + touch * float(nx + ny + nz) * axis_coef;",
        *(f"    {name}[idx] = {name}[idx] * dtdx;" for name in written),
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_cylindrical_real_fused_magnetic_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_B`` -> wall -> ``update_H`` on a Dcyl m = 0 grid?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it. That is
    the construction :func:`.fused_magnetic_pair.metal_fused_magnetic_pair_coverage`
    uses.
    """
    reasons: List[str] = []

    curl = cylindrical_real_curl_coverage(fields, pml, CURL_SUB_STEP, residency)
    if not curl.covered:
        reasons.extend(f"cylindrical curl half: {reason}" for reason in curl.reasons)
    magnetic = cylindrical_real_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency)
    if not magnetic.covered:
        reasons.extend(f"cylindrical constitutive half: {reason}"
                       for reason in magnetic.reasons)

    # THE SOURCE SEAM. A MAGNETIC source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B. An
    # ELECTRIC source is injected in the D/E half and does NOT disqualify this pair —
    # and it is what all three corpus rows carry. IGNORANCE IS NEVER AN EMPTY SET:
    # `Fields` does not hold the source list, so a predicate that inferred "no
    # magnetic source" from not being told would be the over-covering refusal this
    # clause exists to prevent.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'B',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "magnetic source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is magnetic: the "
            f"driver injects it BETWEEN step_B and update_H "
            f"(driver.py:3283-3284)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILL PASSES MUST BE INERT, not merely absent from this kernel. Both
    # halves already refuse a mirror plane; RESTATED here because this product's
    # REPLACES tuple names three passes and not five, and a reader must not have to
    # chase another predicate to learn why the other two are missing.
    reader = getattr(grid, "has_symmetry", None)
    if callable(reader) and bool(reader()):
        reasons.append(
            "a mirror plane is active: fill_symmetry_bc_B and "
            "fill_folded_far_ghosts_B then do real work inside this seam "
            "(driver.py:3285, :3287) and this kernel carries neither")
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded by a mirror plane: the two fill passes in "
                f"this seam stop being inert")

    # zero_metal_B is CARRIED, so the grid must be able to answer which axes are
    # walled. A grid that cannot would silently be treated as unwalled, which is a
    # plane of wrong values rather than a crash. NOT VACUOUS on the rows this
    # product is priced against: all three declare a metallic z. A z-PERIODIC Dcyl
    # grid is admitted too on this backend (see the module docstring) and clears
    # nothing, which is correct and is exercised by the gate rather than assumed.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalCylindricalRealFusedMagneticPairPlan:
    """The device radial scan, then ONE dispatch for three of the driver's passes.

    DELIBERATELY A SIBLING OF :class:`.plans.KernelPlan` RATHER THAN A SUBCLASS, for
    the reason :class:`.cylindrical_real.CylindricalRealCurlPlan` is one: that base's
    whole contract is that ``run`` unpacks ONE argument tuple and calls ONE function.
    This plan cannot honour it — the fused kernel READS the scan's output, so the two
    are ordered and cannot be one dispatch — and the divergence is stated here rather
    than hidden behind an overridden ``run``.

    THE FUSION IS THE SECOND LAUNCH. Separate, this configuration is THREE dispatches
    (scan, curl, constitutive); here it is TWO, and the flux density never leaves a
    register between the curl and the constitutive.

    ``launches`` COUNTS KERNEL LAUNCHES, NOT ``run`` CALLS, and it is two per cycle.
    :attr:`prefix_launches` and :attr:`fused_launches` are carried apart so "the scan
    ran and the fused kernel did not" is distinguishable from "the plan ran once".
    """

    __slots__ = ("shape", "n_elem", "scan_shape", "n_cols", "dtdx", "axis_coef",
                 "bc", "zero_metal", "residency", "volumes", "prefix_host",
                 "launches", "prefix_launches", "fused_launches", "runs",
                 "_prefix_functions", "_prefix_args",
                 "_fused_functions", "_fused_args")

    family = "cylindrical m = 0 fused PML B-curl/stored-H pair"

    replaces_sub_steps = REPLACES

    #: This plan launches kernels; the composer's planned/null split reads it.
    performs_device_work = True

    #: TWO: the radial scan and the fused pair. Declared with the same name
    #: :class:`.plans.KernelPlan` declares it under — this class is deliberately not
    #: a subclass of that base, so the attribute is restated rather than inherited.
    launches_per_run = 2

    REPR_FIELDS = ("shape", "bc", "zero_metal", "scan_shape")

    def __init__(self, shape, scan_shape, dtdx: float, axis_coef: float, bc,
                 zero_metal, residency: Residency, prefix_host: Any,
                 prefix_args: Sequence[Any], fused_args: Sequence[Any],
                 prefix_functions: Mapping[str, Any],
                 fused_functions: Mapping[str, Any],
                 volumes: Sequence[str]) -> None:
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or int(self.shape[1]) != 1:
            raise ValueError(
                f"a Dcyl grid stores one phi cell; shape {self.shape} does not "
                f"(the exp(i*m*phi) dependence is analytic)")
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.scan_shape = tuple(int(n) for n in scan_shape)
        self.n_cols = self.shape[1] * self.shape[2]
        # float(): the array path multiplies a float32 volume by a Python float,
        # which NumPy casts to float32 before the multiply; the packed struct holds
        # numpy.float32 of the same value, so the two scalars carry the same bits.
        self.dtdx = float(dtdx)
        self.axis_coef = float(axis_coef)
        self.bc = tuple(int(code) for code in bc)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.prefix_host = prefix_host
        self._prefix_functions = dict(prefix_functions)
        self._fused_functions = dict(fused_functions)
        self._prefix_args = tuple(prefix_args)
        self._fused_args = tuple(fused_args)
        self.launches = 0
        self.prefix_launches = 0
        self.fused_launches = 0
        self.runs = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        """The contraction modes BOTH kernels were built with.

        The intersection rather than either alone: a plan holding a ``fast`` scan and
        no ``fast`` fused kernel would report a variant it cannot launch.
        """
        return tuple(sorted(set(self._prefix_functions) & set(self._fused_functions)))

    def _function(self, table: Mapping[str, Any], mode: str, which: str) -> Any:
        function = table.get(mode)
        if function is None:
            raise KeyError(
                f"this plan holds no {mode!r} {which} variant (it was built with "
                f"{self.variants}); build it with contract_variants={(mode,)} rather "
                f"than launching the pinned one")
        return function

    def run_prefix(self, contract: Optional[str] = None) -> None:
        """The radial scan alone — ``cylindrical_rderiv_prefix`` on the device."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._function(self._prefix_functions, mode, "scan")
        self.launches += 1
        self.prefix_launches += 1
        function(*self._prefix_args)

    def run_fused(self, contract: Optional[str] = None) -> None:
        """The fused pair alone. Reads the scan's output, so it is never run first."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._function(self._fused_functions, mode, "fused pair")
        self.launches += 1
        self.fused_launches += 1
        function(*self._fused_args)

    def run(self, contract: Optional[str] = None) -> None:
        """The seam: scan, then the fused pair. In place.

        THE ORDER IS THE CONTRACT. The fused kernel reads the prefix the scan writes,
        and the prefix is a function of THIS sub-step's source volume — so a scan
        hoisted out of the loop, or run after the pair, feeds it the PREVIOUS
        timestep's radial derivative. That is a smooth, plausible, entirely wrong
        field rather than an error.
        """
        self.runs += 1
        self.run_prefix(contract)
        self.run_fused(contract)

    def describe(self) -> str:
        parts = ", ".join(f"{name}={getattr(self, name)!r}"
                          for name in self.REPR_FIELDS)
        return (f"MetalCylindricalRealFusedMagneticPairPlan({parts}, "
                f"variants={self.variants})")

    def __repr__(self) -> str:
        return self.describe()


def _params_tensor(shape: Sequence[int], dtdx: float, axis_coef: float,
                   device: str) -> Any:
    """The six scalars as one 24-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason
    :func:`.fused_magnetic_pair._params_tensor` gives: :meth:`.Residency.mirror`
    binds float32 and complex64 volumes and refuses anything else BY NAME, because a
    wider element silently reinterprets. This record is neither — a packed struct of
    four uints and two floats that nothing on the host reads back and nothing on the
    device writes — so it is built here, once, at plan time. Nothing on the launch
    path allocates.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=np.dtype([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"),
                                         ("n_elem", "<u4"), ("dtdx", "<f4"),
                                         ("axis_coef", "<f4")]))
    nx, ny, nz = (int(n) for n in shape)
    record[0] = (nx, ny, nz, nx * ny * nz, np.float32(dtdx), np.float32(axis_coef))
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def _fused_functions(codes, zero_metal,
                     contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_cylindrical_real_fused_magnetic_pair(codes, zero_metal, mode)
            for mode in contract_variants}


def plan_metal_cylindrical_real_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        prefix_function: Optional[Mapping[str, Any]] = None,
        fused_function: Optional[Mapping[str, Any]] = None,
        ) -> Optional[MetalCylindricalRealFusedMagneticPairPlan]:
    """Build the fused Dcyl B/H plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise have
    stepped correctly.

    ``prefix_function`` and ``fused_function`` are the mutation seams, and they are
    SEPARATE so a leg can plant a defect in one kernel while the other stays shipped.
    Dropping them is not a silent slowdown but a silent DISARMING: every mutation leg
    would then launch the shipped kernels and report the defect as uncaught.
    """
    if not metal_cylindrical_real_fused_magnetic_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    import numpy as np  # noqa: PLC0415
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415
    from ..triton_kernels.coverage import CONSTITUTIVE_SIDES  # noqa: PLC0415
    from ..triton_kernels.launch import SUB_STEPS  # noqa: PLC0415

    spec = SUB_STEPS[CURL_SUB_STEP]
    side = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    codes = boundary_codes(resolve(grid, pml))
    walls = zero_metal_axes(grid)
    shape = tuple(int(n) for n in grid.shape)
    scan_shape = (scan_rows(CURL_SUB_STEP, shape[0]), shape[1], shape[2])

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step.
    flux = [bind(name, getattr(fields, name)) for name in spec["targets"]]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in spec["targets"]]
    electric = [bind(name, getattr(fields, name)) for name in spec["sources"]]
    # THE B CURL TAKES THE HALF-INTEGER LATTICE and the H constitutive the INTEGER
    # one (SUB_STEPS['step_B']['suffix'] == '_h'; CONSTITUTIVE_SIDES['H']
    # ['half_integer'] is False). The kernel takes both and never asks which is
    # which, so a swap here is a silent half-cell error in the absorber profile.
    curl_coefficients = [
        bind(f"pml:{stem}_{axis}{spec['suffix']}",
             getattr(pml, f"{stem}_{axis}{spec['suffix']}"), constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    magnetic = [bind(name, getattr(fields, name)) for name in side["targets"]]
    workspace = [bind(name, getattr(fields, name)) for name in side["aux"]]
    constitutive_coefficients = [
        bind(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    # The scan reads the sub-step's PHI source — Ep on the B side, which is
    # `sources[1]`. Named through the term table rather than by index so a table
    # change cannot silently repoint it.
    scan_source_name = spec["sources"][1]
    scan_source = residency.mirror(scan_source_name,
                                   getattr(fields, scan_source_name))
    prefix_host, prefix = scratch_mirror(
        residency, f"cyl_pfx:{CURL_SUB_STEP}",
        lambda: np.zeros(scan_shape, dtype=np.float32))
    _, weight_vector = scratch_mirror(
        residency, f"cyl_weights:{CURL_SUB_STEP}",
        lambda: prefix_row_vectors(CURL_SUB_STEP, shape[0])[0], constant=True)
    _, divisor_vector = scratch_mirror(
        residency, f"cyl_divisor:{CURL_SUB_STEP}",
        lambda: prefix_row_vectors(CURL_SUB_STEP, shape[0])[1], constant=True)

    dtdx = grid.dt / grid.dx
    # ZERO ON THIS SIDE: the m = 0 on-axis increment is the D side's rule. The slot
    # is bound so the lifted body needs no fork.
    axis_coef = 0.0
    pointers = (flux + auxiliary + electric + [prefix] + curl_coefficients
                + magnetic + workspace + constitutive_coefficients)
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    return MetalCylindricalRealFusedMagneticPairPlan(
        shape, scan_shape, dtdx, axis_coef, codes, walls, residency, prefix_host,
        (prefix, scan_source, weight_vector, divisor_vector,
         scan_shape[0], shape[1], shape[2], shape[1] * shape[2]),
        tuple(pointers) + (_params_tensor(shape, dtdx, axis_coef,
                                          residency.device),),
        dict(prefix_function) if prefix_function is not None else {
            mode: compile_cylindrical_prefix(CURL_SUB_STEP, mode)
            for mode in contract_variants},
        dict(fused_function) if fused_function is not None
        else _fused_functions(codes, walls, contract_variants),
        volumes)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"cylindrical fused magnetic pair cannot fill {slot}",))
    return metal_cylindrical_real_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalCylindricalRealFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_cylindrical_real_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY. ``plan_step`` assigns at most one arm
    per slot and this product spans three, so there is no slot it could claim without
    a composition rule nothing has measured. Registering unwired keeps it ENUMERABLE
    for the disjointness sweep (``arms.registered``) while ``arms.arms_for`` skips
    it, so ``plan_step`` cannot select it and no existing arm's selection changes —
    including the two the cylindrical family already wins.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "cylindrical m=0 fused magnetic B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="cylindrical fused magnetic B/H pair: ",
                          noun="cylindrical m = 0 fused PML B-curl/stored-H pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
