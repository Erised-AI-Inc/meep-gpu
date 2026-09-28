"""The Dcyl m = 0 ELECTRIC seam in one dispatch: ``step_D`` welded into ``update_E``.

THE ELECTRIC TWIN of :mod:`.cylindrical_real_fused_magnetic_pair`, over the SAME two
certified halves on the other seam, and the Metal sibling of
:mod:`..triton_kernels.cylindrical_real_fused_electric_pair`. The source is SPLICED
FROM THIS BACKEND'S OWN CERTIFIED EMITTERS rather than ported from the Triton text,
so the two are siblings and not a translation.

===========================================================================
THE CELL, AND WHY IT WAS NOT BUILT UNTIL NOW
===========================================================================

``parity/meep_gpu/results/fusion_matrix_metal_2026-08-30_lastcells`` ranks this cell
SEVENTH among the unbuilt ones and gives it the only verdict on the board that is a
PLATFORM refusal rather than a structural one::

    reach 3  (of 3 rows)  D_to_E  (cylindrical m=0, cylindrical m=0)
        cylindrical_real.curl 16p + shaders.constitutive_E 18p, sharing 3
        -> 31 pointers, ceiling 30      UNFUSABLE ON METAL

OVER BY EXACTLY ONE POINTER. Its magnetic twin fits with two to spare (28 pointers)
because the H constitutive brings no inverse-permeability volume; on the E side the
three inverse-epsilon volumes are real and the same cylindrical curl half — whose
sixteenth pointer is the radial prefix ``pfx`` — puts the pair one over.

WHAT MOVED IS NOT THE CEILING. :data:`.device.MAX_BUFFER_BINDINGS` is 31 and stays
31; this file measures it again rather than citing it. What changed is the SIGNATURE:
the curl half's six per-axis PML coefficient vectors are now ONE buffer with six
element offsets carried in ``Params`` — see :mod:`.coefficient_pack` for why an array
pack is a NEW step on this backend and why it costs nothing here. The pair binds 26
pointers where the unpacked shape needs 31, and both numbers are COMPILED rather than
counted (:data:`UNPACKED_POINTER_BINDINGS`, :data:`PACKED_BINDINGS`).

THE PREFIX IS DELIBERATELY NOT IN THE PACK, and that is the one design decision here
worth stating twice. ``pfx`` is the only member of the curl half's read-only set that
the DEVICE writes — the radial scan fills it in the launch immediately before this
one. Folding it into the same allocation as the coefficient vectors would have saved
a seventh pointer and created a hazard nothing in this package could catch: a wrong
offset in the SCAN would overwrite a PML coefficient vector, permanently (a constant
mirror is uploaded once and never refreshed) and silently (an absorber that leaks a
little is a plausible field). The pack is read-only by construction; the prefix keeps
its own binding.

===========================================================================
ALL THREE ROWS DECLARE AN ELECTRIC SOURCE
===========================================================================

``tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_{0_0,1_0}`` and
``tests:TestPMLCylindrical.test_pml_cyl_0_0_0`` — the same three rows the magnetic
twin serves, and on THIS seam their sources are inside the seam rather than outside
it. With :data:`CARRIES_DEPOSIT_REPAIR` at False this arm would admit NOTHING AT ALL,
which is the exact reverse of the twin's situation. The flag is declared in the same
change as the plan wiring that licenses it.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

The driver runs five passes between ``step_D`` and ``update_E``
(driver.py:3292-3304, mirrored in :data:`.coverage.RESIDENCY_ORDER`)::

    step_D -> electric sources -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

* **the electric sources — CARRIED, through the SHIPPED deposit repair.** The
  injection lands between the halves, so the fused launch consumes a pre-injection
  ``D``; :class:`..deposit_repair.LeadingRepairPlan` saves the stateful arrays at the
  deposit points before the launch and :class:`..deposit_repair.TrailingRepairPlan`
  recomputes them after, in the driver's own order. IGNORANCE IS STILL NEVER AN EMPTY
  SET: ``Fields`` does not hold the source list, so an undeclared ``sources`` is a
  REFUSAL and a source that cannot publish the index it writes is refused by name.
* **``fill_symmetry_bc_D`` — PROVABLY INERT, not carried.**
  ``stepping._fill_symmetry_ghost_cells`` (:1437-1439) returns before touching a cell
  unless ``grid.has_symmetry()``, and both halves' predicate refuses a mirror plane.
* **``zero_metal_D`` — CARRIED INLINE**, and NOT vacuous on the rows this product is
  priced against: all three declare a metallic z, so the z row fires on every one.
  The table is the D OFF-DIAGONAL (Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z) and it
  is IMPORTED from :func:`.fused_electric_pair.zero_metal_mask` rather than
  re-spelled — the magnetic twin's table is its complement, and a family that reused
  the wrong one would clear the wrong two components on every walled run.
* **``fill_folded_far_ghosts_D`` — PROVABLY INERT**, at the same line and for the
  same reason (:1516-1517), plus it needs an axis whose stored extent exceeds its
  owned one, which only a folded periodic axis has.

So :data:`REPLACES` names THREE passes and not five, and the predicate is what
guarantees the other two cannot run. THE INJECTION IS NOT ONE OF THE THREE: it is
carried by the repair bracket around the launch, not by the launch.

===========================================================================
WHAT IS CYLINDRICAL HERE, AND WHAT IS NOT
===========================================================================

**The curl half is deeply cylindrical, and more so than the magnetic twin's.** Three
things absent at ``step_B`` are live at ``step_D``: ``Dz``'s curl takes the BACKWARD
difference of the radial prefix (the four-operand grouping, not ``Bz``'s two-operand
forward difference); the ownership mask is the D family's; and the m = 0 axis tail is
``_cylindrical_axis_zero_D`` — ``Dz[0] += (4*Courant)*Hp[0]`` as a POST-ADD, then
``Dp[0] = 0``. All three are the certified emitter's own text, lifted.

**The constitutive half is not cylindrical at all**, and that is a FINDING this
product inherits rather than an assumption it makes:
:func:`.cylindrical_real.cylindrical_real_constitutive_coverage` records that
``update_E`` carries no cylindrical branch — it is element-wise with per-axis
coefficient tables indexed on the component's own axis.

===========================================================================
THE PREFIX IS UPSTREAM OF THE SEAM, and the fusion is the SECOND launch
===========================================================================

The radial scan runs ON THE DEVICE on this backend and BEFORE the curl, so no
dispatch straddles it. This plan is TWO dispatches per run — the scan, then the fused
curl+wall+constitutive — where the separate composition is THREE. ``launches`` counts
kernel launches and not ``run`` calls, and :attr:`prefix_launches` and
:attr:`fused_launches` are carried apart so "the scan ran and the fused kernel did
not" is distinguishable from "the plan ran once".

NOT SELECTED AS AN ARM, but ROUTED BY THE ABSORB TABLE. ``STEP_ORDER`` assigns at
most one arm per slot and this product spans three, so it registers ``wired=False``
and ``_select_slot`` cannot pick it. What DOES reach it is
``launch._install_fused_pairs``, through the ``FUSED_PAIR_ARMS`` row this product
landed with — ``("cylindrical m=0", "cylindrical m=0")``, read off ``plan_step(...,
fuse=False)`` on the Dcyl m = 0 matrix row. That row is not decoration: it is what
lets the installer bracket this launch with the deposit repair, and a Metal family
that flipped :data:`CARRIES_DEPOSIT_REPAIR` without it would be claiming a bracket the
seam loop refuses to build. ``meep_gpu.fastpath.plan_fast_path`` still returns ``None``
on every branch, so no default run reaches this plan.

DEVICE STATUS: **RELEASED 2026-08-31** on this host (Apple GPU via MPS, torch 2.10.0)
    under the ``flush`` subnormal policy the backend can honour, with the
    subnormal-free precondition checked on every step —
    ``parity/meep_gpu/results/metal_cylindrical_real_fused_electric_pair_2026-08-31_pack/``:

    * **6/6 product cases, 12 complete steps each, bit-identical** on the uint32 view
      of all 24 stored volumes to the array path, with the dispatch counts asserted
      per cycle (one scan, one fused dispatch); 6/6 again on the +-0 lattice;
    * **THE BINDING CEILING MEASURED FOUR WAYS**, which is what licenses the pack:
      the 37-binding separate-scalar signature FAILS; the 32-binding UNPACKED-pointer
      signature FAILS with the platform's own ``'buffer' attribute parameter is out of
      bounds``; the shipped 27-binding one compiles, launches and reads every struct
      field back including the six offsets; and the ceiling is BISECTED on this host —
      30 pointers plus one struct COMPILES, 31 is REFUSED. So ``over_the_ceiling_by``
      is 1 and ``pack_saves_pointers`` is 5, both measured rather than argued;
    * **the pack read as six vectors**: an 82-element pack against six separate
      pointers, 0 differing words per vector on the host and 0/20 on the device;
    * **a REAL electric deposit carried**: the shipped composer installs
      ``LeadingRepairPlan``/``TrailingRepairPlan``, 6 deposit points repaired, the
      injection moves 24 words, and the walk is bit-identical for 12 complete steps.
      Its NULL CONTROL — the same launch with the trailing slot left as the
      ``NoopPlan`` the False branch would leave there — DIVERGES at step 1 by 4 words
      (Ez 2, f_w_Ez 2);
    * **2/2 separate controls**: the array path, the two ALREADY CERTIFIED Metal
      products, and this one, all three word-for-word equal at every step. THE FUSION,
      MEASURED: three dispatches per step become two, and on a walled run the separate
      side leaves ``zero_metal_D`` on the HOST where this side carries it;
    * the two omitted driver passes EXECUTED and measured inert on every admitted
      case, with ``zero_metal_D`` as the control that must move something;
    * **37 armed defects, 33 caught and 4 CONFIRMED nulls**, plus a byte-neutral
      control that must NOT diverge and a disarm row on the shipped bytes.

    THE FOUR NULLS ARE MEASUREMENTS AND EACH HAS A LIVE SIBLING. Flattening
    ``dtdx * ((c_y - c) + (b - b_z))`` is a different float32 number on a Cartesian
    grid; here ``c_y`` IS ``c`` — phi is the one-cell invariant axis — so it came back
    0 differing words, while the same edit on target 1, whose pair is (z, r) and both
    live, diverges. Dropping the r ownership mask on ``Dz`` moves nothing, because
    that curl's radial pair is (prefix[0], the metallic zero ghost) and prefix row 0
    is an exact ``+0.0``; the same edit on ``Dy`` is caught. And the two PHI OFFSET
    swaps are null because ``kms_y`` and ``sinv_y`` are each ONE element holding
    exactly ``1.0f`` — the phi axis carries no PML — while the SAME two offsets moved
    the other way, into a neighbouring vector, are caught. Each pairing is what stops
    its null reading as "this does not matter here".
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import coefficient_pack, shaders, templates
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
from .fused_electric_pair import (
    _CURL_STORE,
    certified_constitutive_body,
    zero_metal_mask,
)
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: THIS cell it is the difference between three seam-instances and zero: every one of
#: the three corpus rows declares an ELECTRIC source, which the driver injects between
#: the two halves (driver.py:3294-3299). The flag is a claim about the PLAN this
#: module builds -- that the leading slot saves and the trailing slot restores -- and
#: is only ever changed in the same edit as that wiring. Its magnetic twin holds the
#: flag at False and the same three rows cost it nothing, which is the whole
#: difference between the two seams on this configuration.
#:
#: THE FLAG IS NOT THE WHOLE CLAIM. It only reaches ``deposit_repair.repairable``,
#: which goes on refusing BY NAME every seam the repair cannot invert -- an
#: off-diagonal constitutive, a nonlinear one, a FOLD whose fill map it cannot read,
#: and an absorber whose split-field recurrence never ran. The rows this admits carry
#: no fold at all, so that clause is quiet here rather than absent.
#:
#: THE PRECONDITION THIS PRODUCT MEETS is that its constitutive half is POINTWISE:
#: ``E += kps*(D*inv_eps) - kms*f_w_E`` cell by cell, reading the register the curl
#: half wrote at the SAME cell. So a whole-grid launch against an uninjected D is
#: wrong ONLY at the deposit points, which is exactly what a point repair puts back.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cylindrical_real_fused_electric_pair"

#: The sub-step slot this arm holds a row on. It spans three; a refusal is NAMED on
#: the slot the fusion starts at rather than being invisible to the table.
SLOT = "step_D"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: The driver passes ONE fused dispatch performs, in driver order
#: (driver.py:3292-3304). THREE, not five: see :data:`INERT_PASSES`.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The two driver passes this product does NOT carry, with the reason each cannot
#: execute on a configuration the predicate admits. Named as data so the gate can
#: assert them inert rather than assume it.
INERT_PASSES: Dict[str, str] = {
    "fill_symmetry_bc_D": ("stepping._fill_symmetry_ghost_cells:1437-1439 returns "
                           "unless grid.has_symmetry(); both halves' predicate "
                           "refuses a mirror plane"),
    "fill_folded_far_ghosts_D": ("stepping._fill_folded_far_ghosts:1516-1517 returns "
                                 "unless grid.has_symmetry(), and additionally needs "
                                 "an axis whose stored extent exceeds its owned one"),
}

#: The curl half's per-axis PML coefficient vectors, IN PACK ORDER. One buffer holds
#: them end to end and ``Params`` carries six element offsets. The order is the order
#: the certified curl body declares them in, so a reader comparing the packed
#: signature to the unpacked one is comparing one line to six and nothing else.
PACKED_VECTORS: Tuple[str, ...] = ("kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz")

#: THREE BINDING COUNTS, ALL THREE COMPILED BY THE GATE rather than argued from here.
#:
#: * :data:`SEPARATE_SCALAR_BINDINGS` — 31 pointers with the six scalars bound
#:   SEPARATELY. Refused, and by a wide margin; it is the shape the scalar packing
#:   the shipped pairs already do exists to avoid.
#: * :data:`UNPACKED_POINTER_BINDINGS` — 31 pointers plus ONE packed ``Params&``.
#:   THIS IS THE NUMBER THAT MATTERS: it is the fused pair as every other D/E product
#:   on this backend builds it, and it is REFUSED — 32 bindings against a ceiling of
#:   31, over by exactly one. That refusal is what the board's ``UNFUSABLE ON METAL``
#:   verdict recorded, and it is re-measured here rather than cited.
#: * :data:`PACKED_BINDINGS` — the shipped shape, 26 pointers plus ``Params``.
SEPARATE_SCALAR_BINDINGS = 37
UNPACKED_POINTER_BINDINGS = 32
PACKED_BINDINGS = 27

#: The pointer count of the shipped signature, and of the unpacked one it replaces.
#: Spelled apart from the binding counts because the ceiling is a BINDING ceiling and
#: the sharing calibration the fusion matrix runs is a POINTER count.
PACKED_POINTERS = 26
UNPACKED_POINTERS = 31

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAMILY",
    "INERT_PASSES", "PACKED_BINDINGS", "PACKED_POINTERS", "PACKED_VECTORS",
    "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT", "UNPACKED_POINTERS",
    "UNPACKED_POINTER_BINDINGS",
    "MetalCylindricalRealFusedElectricPairPlan", "certified_cyl_curl_body",
    "compile_cylindrical_real_fused_electric_pair",
    "cylindrical_real_fused_electric_pair_source",
    "metal_cylindrical_real_fused_electric_pair_coverage",
    "plan_metal_cylindrical_real_fused_electric_pair",
    "refuted_separate_scalar_source", "refuted_unpacked_pointer_source",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

#: THE BUFFER ORDER IS ``fused_electric_pair``'s, GROUP FOR GROUP, with two
#: differences and no others: the radial prefix sits immediately after the curl's
#: three sources (where the magnetic twin puts it), and the six curl coefficient
#: vectors have become the ONE ``cpml`` pack in the slot the first of them held.
_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params {
    uint nx; uint ny; uint nz; uint n_elem; float dtdx; float axis_coef;
__PACK_FIELDS__
};

kernel void cyl_real_fused_electric_pair_step(
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
    device float*       e0      [[buffer(10)]],
    device float*       e1      [[buffer(11)]],
    device float*       e2      [[buffer(12)]],
    device float*       w0      [[buffer(13)]],
    device float*       w1      [[buffer(14)]],
    device float*       w2      [[buffer(15)]],
    device const float* ie0     [[buffer(16)]],
    device const float* ie1     [[buffer(17)]],
    device const float* ie2     [[buffer(18)]],
    device const float* cpml    [[buffer(19)]],
    device const float* kp0     [[buffer(20)]],
    device const float* km0     [[buffer(21)]],
    device const float* kp1     [[buffer(22)]],
    device const float* km1     [[buffer(23)]],
    device const float* kp2     [[buffer(24)]],
    device const float* km2     [[buffer(25)]],
    constant Params&    prm     [[buffer(26)]],
    uint idx [[thread_position_in_grid]])
{
    // THE SIX SCALARS AND THE SIX PACKED VECTORS ARE UNPACKED INTO THE CERTIFIED
    // BODIES' OWN NAMES, once, before any of the lifted text runs. Everything below
    // this line is then character-for-character what
    // `cylindrical_real.cylindrical_curl_source` and `shaders.constitutive_source`
    // emit, plus the wall clear and the one seam substitution.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float axis_coef = prm.axis_coef;
    (void)axis_coef;   // read by the D-side axis tail; bound for one layout.

    // --- the curl half's PML coefficient vectors, one allocation, six offsets ---
    // READ-ONLY: nothing on the device writes `cpml`. The radial prefix `pfx` is the
    // one member of this half's read-only set the device DOES write (the scan fills
    // it in the launch before this one) and it keeps its own binding for that reason.
__PACK_PROLOGUE__

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
    """The Dcyl ``step_D`` curl kernel's BODY, lifted from the certified emitter.

    Not a transcription: this is :func:`.cylindrical_real.cylindrical_curl_source`'s
    own output with the ``#include``/signature preamble and the closing brace removed.
    ``sub_step`` is pinned to ``step_D`` and is never a parameter — this pair is the
    electric seam, and ``step_B``'s forward differences, its two-operand ``Bz`` prefix
    substitution and its one-rule axis tail are the magnetic twin's product.
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


def cylindrical_real_fused_electric_pair_source(
        codes: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundary triple, walls, contraction mode).

    ``codes`` is the per-axis PERIODIC/METALLIC pair triple
    :func:`.cylindrical_real.boundary_codes` resolves, with the r axis pinned
    METALLIC — the certified curl emitter refuses anything else by name, and this
    function refuses it again before the splice so a caller cannot reach a kernel
    whose far r ghost wraps around the cylinder.

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string here,
    for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes a source
    and nothing else, so specialisation by substitution is what keeps the emitted
    arithmetic identical to the bodies this lifts.
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
        "    // --- zero_metal_D, carried inline (stepping._zero_metal:2232) ------\n"
        "    // The OFF-DIAGONAL for D: Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z.\n"
        "    // AFTER the m = 0 axis rules above and BEFORE the store, which is the\n"
        "    // driver's own order (step_D ends with the axis rules, then\n"
        "    // zero_metal_D).\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    electric = certified_constitutive_body(contract)
    # THE SEAM, and it is exactly three lines. The certified E body opens each
    # component with `float srcN = gN[ii] * ieN[ii];` — a RELOAD of the displacement
    # the curl just stored, scaled by the inverse permittivity. Fused, the reload
    # becomes the live register and NOTHING ELSE MOVES: the inverse-epsilon factor
    # stays on the right and D stays on the LEFT, which is the order the array path
    # writes (stepping.py:1011-1013) and not a commutation this transcription may make.
    for target in range(3):
        old = f"    float src{target} = g{target}[ii] * ie{target}[ii];\n"
        if old not in electric:
            raise AssertionError(
                f"the certified E constitutive body no longer reads its source as "
                f"{old.strip()!r}; the seam has no anchor")
        electric = electric.replace(
            old,
            f"    // THE SEAM: the register step_D just wrote, not a reload of "
            f"D{'xyz'[target]}.\n"
            f"    float src{target} = v{target} * ie{target}[ii];\n")
    # THE CURL'S `g` IS H AND THE CONSTITUTIVE'S `g` WAS D. After the seam there must
    # be no `g` left in the electric half at all — one surviving read would take the
    # magnetic field as a displacement, which is a smooth, plausible, entirely wrong
    # answer rather than a crash.
    for target in range(3):
        if f"g{target}[" in electric:
            raise AssertionError(
                f"the lifted E constitutive half still reads g{target}, which in the "
                f"fused signature is H{'xyz'[target]} and not D{'xyz'[target]}")

    body = "".join((
        curl,
        "\n    // --- update_E (stepping.update_E / _apply_constitutive_pml:2083) --\n",
        electric,
    ))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__PACK_FIELDS__": coefficient_pack.params_fields(PACKED_VECTORS),
        "__PACK_PROLOGUE__": coefficient_pack.prologue("cpml", PACKED_VECTORS),
        "__BODY__": body,
    })


def compile_cylindrical_real_fused_electric_pair(
        codes: Sequence[int], zero_metal: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, walls, mode)."""
    return compile_source(cylindrical_real_fused_electric_pair_source(
        codes, zero_metal, contract)).cyl_real_fused_electric_pair_step


#: The pointer names of the UNPACKED fused signature, in the order the two certified
#: halves declare them. Written once and consumed by both refuted emitters, so the
#: number this family says it was over by is the number both refusals are built from.
_UNPACKED_WRITTEN: Tuple[str, ...] = ("f0", "f1", "f2", "u0", "u1", "u2",
                                      "e0", "e1", "e2", "w0", "w1", "w2")
_UNPACKED_READ: Tuple[str, ...] = ("g0", "g1", "g2", "pfx",
                                   "ie0", "ie1", "ie2",
                                   "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
                                   "kp0", "km0", "kp1", "km1", "kp2", "km2")


def _touching_body(written: Sequence[str], read: Sequence[str], guard: str,
                   scale: str, extents: str) -> List[str]:
    """A body that READS every input and WRITES every output.

    A body the compiler could drop would let dead-code elimination decide the answer,
    and the answer being measured is about the SIGNATURE.
    """
    touch = " + ".join(f"{name}[0]" for name in read)
    return [f"    {guard}",
            f"    float touch = {touch};",
            f"    f0[idx] = f0[idx] + touch * {extents};",
            *(f"    {name}[idx] = {name}[idx] * {scale};" for name in written)]


def refuted_unpacked_pointer_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 32-binding signature this platform REFUSES — 31 pointers plus ``Params``.

    THIS IS THE MEASUREMENT THE PACK EXISTS FOR. It is the fused pair built the way
    every other D/E product on this backend is built — six scalars packed into one
    ``constant Params&``, every vector its own pointer — and it is over the ceiling by
    exactly one binding. Not shipped and not launchable: it exists so the gate can
    compile it and require the failure, which is what turns
    :data:`UNPACKED_POINTER_BINDINGS` from an argument into a measurement.
    """
    lines: List[str] = []
    slot = 0
    for name in _UNPACKED_WRITTEN:
        lines.append(f"    device float*       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in _UNPACKED_READ:
        lines.append(f"    device const float* {name:<8}[[buffer({slot})]],")
        slot += 1
    if slot != UNPACKED_POINTERS:  # pragma: no cover - arithmetic guard
        raise AssertionError((slot, UNPACKED_POINTERS))
    lines.append(f"    constant Params&    prm     [[buffer({slot})]],")
    slot += 1
    if slot != UNPACKED_POINTER_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((slot, UNPACKED_POINTER_BINDINGS))
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        templates.contraction_pragma(contract),
        "",
        "struct Params {",
        "    uint nx; uint ny; uint nz; uint n_elem; float dtdx; float axis_coef;",
        "};",
        "",
        "kernel void refuted_unpacked_pointers(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        *_touching_body(_UNPACKED_WRITTEN, _UNPACKED_READ,
                        "if (idx >= prm.n_elem) { return; }",
                        "prm.dtdx",
                        "float(prm.nx + prm.ny + prm.nz) * prm.axis_coef"),
        "}",
        "",
    ))


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 37-binding separate-scalar signature this platform REFUSES.

    The shape the SCALAR packing every shipped pair already does exists to avoid,
    kept here because a family that reported only the pointer refusal would leave a
    reader to wonder whether the scalars were the problem. They are not: even with
    them packed the pointers alone are one over.
    """
    lines: List[str] = []
    slot = 0
    for name in _UNPACKED_WRITTEN:
        lines.append(f"    device float*       {name:<8}[[buffer({slot})]],")
        slot += 1
    for name in _UNPACKED_READ:
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
        *_touching_body(_UNPACKED_WRITTEN, _UNPACKED_READ,
                        "if (idx >= n_elem) { return; }",
                        "dtdx", "float(nx + ny + nz) * axis_coef"),
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_cylindrical_real_fused_electric_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_D`` -> wall -> ``update_E`` on a Dcyl m = 0 grid?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. That is the
    construction
    :func:`.cylindrical_real_fused_magnetic_pair.metal_cylindrical_real_fused_magnetic_pair_coverage`
    uses on the other seam.
    """
    reasons: List[str] = []

    curl = cylindrical_real_curl_coverage(fields, pml, CURL_SUB_STEP, residency)
    if not curl.covered:
        reasons.extend(f"cylindrical curl half: {reason}" for reason in curl.reasons)
    electric = cylindrical_real_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency)
    if not electric.covered:
        reasons.extend(f"cylindrical constitutive half: {reason}"
                       for reason in electric.reasons)

    # THE SOURCE SEAM. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D unless
    # the deposit repair brackets the launch — which for this family it does, and
    # which is the only reason the cell is worth anything at all. A MAGNETIC source
    # is injected in the B/H half and does NOT disqualify this pair. IGNORANCE IS
    # NEVER AN EMPTY SET: `Fields` does not hold the source list, so a predicate that
    # inferred "no electric source" from not being told would be the over-covering
    # refusal this clause exists to prevent.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294-3299)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILL PASSES MUST BE INERT, not merely absent from this kernel. Both
    # halves already refuse a mirror plane; RESTATED here because this product's
    # REPLACES tuple names three passes and not five, and a reader must not have to
    # chase another predicate to learn why the other two are missing.
    reader = getattr(grid, "has_symmetry", None)
    if callable(reader) and bool(reader()):
        reasons.append(
            "a mirror plane is active: fill_symmetry_bc_D and "
            "fill_folded_far_ghosts_D then do real work inside this seam "
            "(driver.py:3300-3302) and this kernel carries neither")
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded by a mirror plane: the two fill passes in "
                f"this seam stop being inert")

    # zero_metal_D is CARRIED, so the grid must be able to answer which axes are
    # walled. A grid that cannot would silently be treated as unwalled, which is a
    # plane of wrong values rather than a crash. NOT VACUOUS on the rows this product
    # is priced against: all three declare a metallic z.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # A polarization would make the constitutive source (D - sum P) rather than D.
    # The cylindrical constitutive predicate already refuses it; restated because
    # this family's kernel bakes the plain product, the same restatement
    # `fused_electric_pair` carries.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalCylindricalRealFusedElectricPairPlan:
    """The device radial scan, then ONE dispatch for three of the driver's passes.

    DELIBERATELY A SIBLING OF :class:`.plans.KernelPlan` RATHER THAN A SUBCLASS, for
    the reason :class:`.cylindrical_real.CylindricalRealCurlPlan` is one: that base's
    whole contract is that ``run`` unpacks ONE argument tuple and calls ONE function.
    This plan cannot honour it — the fused kernel READS the scan's output, so the two
    are ordered and cannot be one dispatch — and the divergence is stated here rather
    than hidden behind an overridden ``run``.

    THE FUSION IS THE SECOND LAUNCH. Separate, this configuration is THREE dispatches
    (scan, curl, constitutive); here it is TWO, and the displacement never leaves a
    register between the curl and the constitutive.
    """

    __slots__ = ("shape", "n_elem", "scan_shape", "n_cols", "dtdx", "axis_coef",
                 "bc", "zero_metal", "residency", "volumes", "prefix_host",
                 "pack_layout", "launches", "prefix_launches", "fused_launches",
                 "runs", "_prefix_functions", "_prefix_args",
                 "_fused_functions", "_fused_args")

    family = "cylindrical m = 0 fused PML D-curl/stored-E pair"

    replaces_sub_steps = REPLACES

    #: This plan launches kernels; the composer's planned/null split reads it.
    performs_device_work = True

    #: TWO: the radial scan and the fused pair. Declared with the same name
    #: :class:`.plans.KernelPlan` declares it under — this class is deliberately not a
    #: subclass of that base, so the attribute is restated rather than inherited.
    launches_per_run = 2

    REPR_FIELDS = ("shape", "bc", "zero_metal", "scan_shape")

    def __init__(self, shape, scan_shape, dtdx: float, axis_coef: float, bc,
                 zero_metal, residency: Residency, prefix_host: Any,
                 pack_layout: Any, prefix_args: Sequence[Any],
                 fused_args: Sequence[Any],
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
        self.pack_layout = pack_layout
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
        return (f"MetalCylindricalRealFusedElectricPairPlan({parts}, "
                f"variants={self.variants})")

    def __repr__(self) -> str:
        return self.describe()


def _params_tensor(shape: Sequence[int], dtdx: float, axis_coef: float,
                   offsets: Sequence[int], device: str) -> Any:
    """The six scalars AND the six pack offsets as one 48-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason
    :func:`.fused_electric_pair._params_tensor` gives: :meth:`.Residency.mirror` binds
    float32 and complex64 volumes and refuses anything else BY NAME, because a wider
    element silently reinterprets. This record is neither — a packed struct of ten
    uints and two floats that nothing on the host reads back and nothing on the device
    writes — so it is built here, once, at plan time. Nothing on the launch path
    allocates.

    THE OFFSET FIELDS ARE THE PACK'S WHOLE COST. Six uints, in the pack's own order,
    and they are what buys back five buffer bindings. The field order here and the
    struct order in :data:`_TEMPLATE` are BOTH generated from
    :data:`PACKED_VECTORS`, so the two cannot drift into a reinterpretation.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    fields = ([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4"),
               ("dtdx", "<f4"), ("axis_coef", "<f4")]
              + coefficient_pack.record_dtype_fields(PACKED_VECTORS))
    record = np.zeros(1, dtype=np.dtype(fields))
    nx, ny, nz = (int(n) for n in shape)
    offsets = tuple(int(offset) for offset in offsets)
    if len(offsets) != len(PACKED_VECTORS):  # pragma: no cover - arithmetic guard
        raise AssertionError((len(offsets), len(PACKED_VECTORS)))
    record[0] = (nx, ny, nz, nx * ny * nz, np.float32(dtdx),
                 np.float32(axis_coef)) + offsets
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def _fused_functions(codes, zero_metal,
                     contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_cylindrical_real_fused_electric_pair(codes, zero_metal, mode)
            for mode in contract_variants}


def plan_metal_cylindrical_real_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        prefix_function: Optional[Mapping[str, Any]] = None,
        fused_function: Optional[Mapping[str, Any]] = None,
        params: Optional[Any] = None,
        ) -> Optional[MetalCylindricalRealFusedElectricPairPlan]:
    """Build the fused Dcyl D/E plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.

    ``prefix_function`` and ``fused_function`` are the KERNEL mutation seams, and they
    are SEPARATE so a leg can plant a defect in one kernel while the other stays
    shipped. ``params`` is the HOST mutation seam this family adds and its magnetic
    twin has no use for: the packed offsets are host arithmetic, so a leg that corrupts
    one has to be able to hand this builder a struct the shipped code would not have
    written. Dropping any of the three is not a silent slowdown but a silent
    DISARMING, since every mutation leg would then launch the shipped bytes and report
    the defect as uncaught.
    """
    if not metal_cylindrical_real_fused_electric_pair_coverage(
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
    magnetic = [bind(name, getattr(fields, name)) for name in spec["sources"]]

    # The scan reads the sub-step's PHI source — Hp on the D side, which is
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
    volumes.append(f"cyl_pfx:{CURL_SUB_STEP}")

    stored = [bind(name, getattr(fields, name)) for name in side["targets"]]
    workspace = [bind(name, getattr(fields, name)) for name in side["aux"]]
    inverse_epsilon = [
        bind("inv_eps_" + name, fields.inverse_epsilon_for(name), constant=True)
        for name in side["targets"]]

    # THE D CURL TAKES THE INTEGER LATTICE and the E constitutive the HALF-INTEGER
    # one (SUB_STEPS['step_D']['suffix'] is '' and CONSTITUTIVE_SIDES['E']
    # ['half_integer'] is True). That is the OPPOSITE pairing to the B/H twin, and
    # the kernel cannot tell — the gate carries a host mutation for each group.
    #
    # THE PACK IS THE CURL GROUP AND NOTHING ELSE. Six read-only vectors, one buffer,
    # six element offsets in Params. The constitutive group stays six separate
    # pointers because it does not have to move: packing it too would buy margin
    # nothing needs and put two packs in one signature for a reader to keep apart.
    curl_pack, pack_layout = coefficient_pack.packed_mirror(
        residency, f"pml_pack:cyl_real_curl:{CURL_SUB_STEP}", np, PACKED_VECTORS,
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")])
    volumes.append(f"pml_pack:cyl_real_curl:{CURL_SUB_STEP}")
    constitutive_coefficients = [
        bind(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    dtdx = grid.dt / grid.dx
    # stepping.py:586 forms `4.0 * (grid.dt / grid.dx)` in float64 and NumPy rounds
    # that ONE scalar to float32 before the multiply. LIVE on this side — the m = 0
    # on-axis Dz increment is the D side's rule, where the magnetic twin binds the
    # slot at zero without reading it.
    axis_coef = 4.0 * dtdx

    pointers = (flux + auxiliary + magnetic + [prefix] + stored + workspace
                + inverse_epsilon + [curl_pack] + constitutive_coefficients)
    if len(pointers) != PACKED_POINTERS:  # pragma: no cover - arithmetic guard
        raise AssertionError((len(pointers), PACKED_POINTERS))
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    record = (_params_tensor(shape, dtdx, axis_coef, pack_layout.offsets,
                             residency.device) if params is None else params)
    return MetalCylindricalRealFusedElectricPairPlan(
        shape, scan_shape, dtdx, axis_coef, codes, walls, residency, prefix_host,
        pack_layout,
        (prefix, scan_source, weight_vector, divisor_vector,
         scan_shape[0], shape[1], shape[2], shape[1] * shape[2]),
        tuple(pointers) + (record,),
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
            False, (f"cylindrical fused electric pair cannot fill {slot}",))
    return metal_cylindrical_real_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalCylindricalRealFusedElectricPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_cylindrical_real_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY, and it says what it says on every other
    pair: ``plan_step`` assigns at most one arm per slot and this product spans three,
    so there is no slot it could claim without a composition rule nothing has
    measured. Registering unwired keeps it ENUMERABLE for the disjointness sweep
    (``arms.registered``) while ``arms.arms_for`` skips it, so ``plan_step`` cannot
    select it and no existing arm's selection changes — including the two the
    cylindrical family already wins.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "cylindrical m=0 fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="cylindrical fused electric D/E pair: ",
                          noun="cylindrical m = 0 fused PML D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
