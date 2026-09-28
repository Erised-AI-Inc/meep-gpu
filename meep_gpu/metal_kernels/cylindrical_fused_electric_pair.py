"""The Dcyl complex-storage ELECTRIC seam in one dispatch: ``step_D`` welded into
``update_E`` — m = 0 and |m| >= 1 since 2026-09-04, |m| >= 1 before.

THE ELECTRIC TWIN of :mod:`.cylindrical_fused_magnetic_pair`, over the SAME two
certified halves on the other seam, and the Metal sibling of
:mod:`..triton_kernels.cylindrical_fused_electric_pair`. The source is SPLICED FROM
THIS BACKEND'S OWN CERTIFIED EMITTERS rather than ported from the Triton text.

===========================================================================
THE LARGEST CELL ON THE BOARD THAT NO PRODUCT COULD BE BUILT FOR
===========================================================================

``parity/meep_gpu/results/fusion_matrix_metal_2026-08-30_lastcells`` ranks this cell
FIRST among the unbuilt ones — 16 reachable D->E seam-instances, more than any other
gap — and gives it a verdict no other large cell has::

    reach 16 (of 16 rows)  D_to_E  (cylindrical complex, cylindrical complex)
        cylindrical_complex.curl 18p + complex_fields.bloch_constitutive_E 18p,
        sharing 3  ->  33 pointers, ceiling 30      UNFUSABLE ON METAL

OVER BY THREE POINTERS AND BY NOTHING ELSE. Its magnetic twin sits at EXACTLY the
ceiling — :data:`.cylindrical_fused_magnetic_pair.PACKED_BINDINGS` is 31 with zero
headroom — because the H constitutive brings no inverse-permeability volume. On the E
side the three inverse-epsilon volumes are real, and 30 + 3 = 33.

WHAT MOVED IS THE SIGNATURE AND NOT THE CEILING. :data:`.device.MAX_BUFFER_BINDINGS`
is 31 and stays 31; the gate re-measures it by bisection rather than citing it. The
curl half's six per-axis PML coefficient vectors ride in ONE buffer with six element
offsets carried in the ``Params`` struct the eight non-pointer arguments already ride
in — see :mod:`.coefficient_pack`. Six pointers become one, and the pair binds 28
where the unpacked shape needs 33.

THE SAME PACK CLEARS BOTH DCYL CELLS AND THAT IS THE POINT. Its m = 0 real sibling
(:mod:`.cylindrical_real_fused_electric_pair`) was over by ONE and this one by THREE;
one technique, applied to the same group in both, clears both with margin. Packing is
therefore a TOOL on this backend rather than a rescue for one cell.

THE PREFIX IS DELIBERATELY NOT IN THE PACK. ``pfx`` is the only member of the curl
half's read-only set the plan REFRESHES every step — the radial scan runs on the HOST
here and its result is pushed back into that mirror — so folding it into the same
allocation as the coefficients would put a per-step write into a buffer that also
holds absorber constants nothing ever re-uploads. A wrong offset there is a permanent,
silent absorber corruption. The pack is read-only by construction; the prefix keeps
its own binding.

===========================================================================
ALL SIXTEEN ROWS DECLARE AN ELECTRIC SOURCE
===========================================================================

That is the fact that makes this cell the largest one on the board and the reason its
MAGNETIC twin ships with :data:`CARRIES_DEPOSIT_REPAIR` at False: on the B/H seam
those sixteen rows are empty, so the twin's clause costs it nothing. On THIS seam the
same sixteen sources land BETWEEN the halves, so with the flag at False this product
would compile, gate green on every other leg, and serve NOTHING AT ALL. The flag is
declared in the same change as the wiring that licenses it — the ``FUSED_PAIR_ARMS``
row in ``launch.py`` that lets ``_install_fused_pairs`` bracket the launch.

===========================================================================
THE FIVE PASSES IN THE SEAM
===========================================================================

The driver runs five passes between ``step_D`` and ``update_E``
(driver.py:3292-3304)::

    step_D -> electric sources -> fill_symmetry_bc_D -> zero_metal_D
           -> fill_folded_far_ghosts_D -> update_E

* **the electric sources — CARRIED, through the SHIPPED deposit repair**, with the
  leading slot saving the stateful arrays at the deposit points and the trailing slot
  recomputing them after the driver has injected, filled and cleared;
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` — DEAD BY CONSTRUCTION.**
  ``Grid`` refuses a mirror plane on a Dcyl cell outright (grid.py:648-653) and both
  halves restate that refusal, so unlike the Cartesian twins there is no configuration
  on which this family could be handed a live fill. The predicate re-checks anyway;
* **``zero_metal_D`` — CARRIED INLINE**, and on a Dcyl grid only z can be walled
  (:data:`FORBIDDEN_WALL_AXES`), so the emitted clear is the two TANGENTIAL D
  components at z = 0 and nothing else. The table is the D OFF-DIAGONAL, IMPORTED
  from :func:`.complex_fused_electric_pair.zero_metal_mask`; the magnetic twin's is
  its complement, and a family that reused the wrong one would clear the wrong two
  components on every walled run.

===========================================================================
THE PREFIX IS A HOST SCAN, AND THE PAIR IS STILL ONE LAUNCH
===========================================================================

The cylindrical COMPLEX curl reads a radial prefix the HOST computes — a sequential
float32 scan whose summation order defines the answer — so this plan performs a
pre-pass (``sync_out`` of Hy, the scan, ``sync_in`` of the prefix mirror) and then ONE
dispatch, exactly as :class:`.cylindrical_fused_magnetic_pair.CylindricalFusedMagneticPairPlan`
does. ``launches_per_run`` stays 1 because the pre-pass is host work and residency
traffic rather than a kernel launch; :attr:`prefix_syncs` counts it apart so a gate can
assert it HAPPENED rather than trust that it did.

NOT SELECTED AS AN ARM, but ROUTED BY THE ABSORB TABLE, exactly as its m = 0 sibling
is: ``wired=False`` keeps it out of ``_select_slot`` while its ``FUSED_PAIR_ARMS`` row
— ``("cylindrical complex", "cylindrical complex")``, read off ``plan_step(...,
fuse=False)`` on the Dcyl |m| = 1 matrix row — lets ``_install_fused_pairs`` reach and
bracket it. ``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every
branch.

DEVICE STATUS: **RELEASED 2026-08-31** on this host (Apple GPU via MPS, torch 2.10.0)
    under the ``flush`` subnormal policy the backend can honour, with the
    subnormal-free precondition checked on every step, and with the complex expansion
    arm bound from this family's own probe artifact (``FMA_V1``) rather than defaulted
    — ``parity/meep_gpu/results/metal_cylindrical_fused_electric_pair_2026-08-31_pack/``:

    * **6/6 product cases, 12 complete steps each, bit-identical** on the uint32 view
      of all 24 stored volumes to the array path, across ``m = 1``, ``m = -1``,
      ``m = 2``, ``m = 3`` and the ``accurate_fields_near_cylorigin`` row, on both z
      declarations, with the launch AND prefix-sync counters asserted per cycle; 6/6
      again on the complex +-0 lattice;
    * **THE BINDING CEILING MEASURED FOUR WAYS**, which is what licenses the pack:
      the 41-binding separate-scalar signature FAILS; the 34-binding UNPACKED-pointer
      signature FAILS with the platform's own ``'buffer' attribute parameter is out of
      bounds``; a synthetic 32-binding kernel FAILS; and the shipped 29-binding one
      compiles, launches and reads every struct field back — the six offsets and the
      ``float2 inc_b`` included, from the REAL packer, at ``itemsize`` 64. The ceiling
      is BISECTED on this host: 30 pointers plus one struct COMPILES, 31 is REFUSED.
      So ``over_the_ceiling_by`` is 3 and ``pack_saves_pointers`` is 5, both measured;
    * **the pack read as six vectors**: a 74-element pack against six separate
      pointers, 0 differing words per vector on the host and 0/20 on the device;
    * **a REAL electric deposit carried**: the shipped composer installs
      ``LeadingRepairPlan``/``TrailingRepairPlan``, 6 deposit points repaired, the
      injection moves 48 words, and the walk is bit-identical for 12 complete steps.
      Its NULL CONTROL — the same launch with the trailing slot left as the
      ``NoopPlan`` the False branch would leave there — DIVERGES at step 1 by 8 words
      (Ez 4, f_w_Ez 4);
    * **2/2 separate controls**: the array path, the two ALREADY CERTIFIED Metal
      products, and this one, all three word-for-word equal at every step. THE FUSION,
      MEASURED: 24 dispatches become 12 over twelve steps, the prefix pre-pass is paid
      12 times on BOTH sides (so no reader can mistake the dispatch saving for a scan
      saving), and on a walled run the separate side leaves ``zero_metal_D`` on the
      HOST where this side carries it;
    * **25 armed defects, 23 caught and 2 CONFIRMED nulls**, plus a byte-neutral
      control that must NOT diverge and a disarm row on the shipped bytes.

    THE TWO NULLS ARE ONE MEASUREMENT WITH A LIVE SIBLING EACH. ``kms_y`` and
    ``sinv_y`` are each ONE element on a Dcyl grid and both hold exactly ``1.0f`` —
    the phi axis carries no PML — so swapping their two pack offsets reads the same
    word. The SAME two offsets moved the other way, into a neighbouring vector, are
    caught (4,264 words each). That pairing is what stops the nulls reading as "the
    offsets do not matter".
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from ..triton_kernels.launch import SUB_STEPS
from . import (
    coefficient_pack, complex_fields, complex_fused_electric_pair,
    cylindrical_complex, shaders, templates,
)
from .coverage import zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, compile_source
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: THIS cell it is the difference between sixteen seam-instances and zero: every one
#: of the sixteen corpus rows declares an ELECTRIC source, which the driver injects
#: between the two halves (driver.py:3294-3299). The flag is a claim about the PLAN
#: this module builds -- that the leading slot saves and the trailing slot restores --
#: and is only ever changed in the same edit as that wiring.
#:
#: ITS MAGNETIC TWIN HOLDS THE FLAG AT FALSE and needs neither the flag nor a row:
#: the same sixteen rows leave the B/H seam empty, so its clause costs it nothing.
#: That asymmetry is the whole difference between the two seams on this cell.
#:
#: THE FLAG IS NOT THE WHOLE CLAIM. It only reaches ``deposit_repair.repairable``,
#: which goes on refusing BY NAME every seam the repair cannot invert -- an
#: off-diagonal constitutive, a nonlinear one, a FOLD whose fill map it cannot read,
#: and an absorber whose split-field recurrence never ran. A Dcyl grid carries no
#: fold at all, so that clause is quiet here rather than absent.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "cylindrical_complex_fused_electric_pair"

#: The sub-step slot this arm holds a row on. It spans three; a refusal is NAMED on
#: the slot the fusion starts at rather than being invisible to the arm table.
SLOT = "step_D"

#: The curl sub-step this product starts at, and the constitutive side it ends at.
CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "E"

#: The driver passes ONE dispatch of this plan performs, in driver order
#: (driver.py:3292-3304). THE INJECTION IS NOT ONE OF THEM: it is carried by the
#: repair bracket around the launch, not by the launch.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The curl half's per-axis PML coefficient vectors, IN PACK ORDER — the order the
#: certified curl emitter declares them in, so a reader comparing the packed
#: signature to the unpacked one is comparing one line to six and nothing else.
PACKED_VECTORS: Tuple[str, ...] = ("kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz")

#: FOUR BINDING COUNTS, ALL FOUR COMPILED BY THE GATE rather than argued from here.
#:
#: * :data:`SEPARATE_SCALAR_BINDINGS` — 33 pointers with the nine non-pointer
#:   arguments bound SEPARATELY, the way the certified 27-binding cylindrical curl
#:   binds them. Eleven over (ten before 2026-09-04; the ``M_ZERO`` arm's
#:   ``axis_coef`` is the ninth scalar).
#: * :data:`UNPACKED_POINTER_BINDINGS` — 33 pointers plus ONE packed ``Params&``.
#:   THIS IS THE NUMBER THAT MATTERS: it is the fused pair as every other D/E product
#:   on this backend builds it, and it is REFUSED — 34 bindings against a ceiling of
#:   31, over by THREE. That refusal is what the board's ``UNFUSABLE ON METAL``
#:   verdict recorded for this cell, and it is re-measured rather than cited.
#: * :data:`PACKED_BINDINGS` — the shipped shape, 28 pointers plus ``Params``.
#: * :data:`OVER_CEILING_BINDINGS` — 32, the smallest refused count, so the ceiling
#:   itself is located rather than assumed.
SEPARATE_SCALAR_BINDINGS = 42
UNPACKED_POINTER_BINDINGS = 34
PACKED_BINDINGS = 29
OVER_CEILING_BINDINGS = 32

#: The pointer counts apart from the binding counts: the ceiling is a BINDING ceiling
#: and the fusion matrix's sharing calibration is a POINTER count.
PACKED_POINTERS = 28
UNPACKED_POINTERS = 33

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the cylindrical complex fused electric pair binds {PACKED_BINDINGS} "
        f"buffers and Metal's ceiling on this toolchain is {MAX_BUFFER_BINDINGS}; "
        f"the signature cannot be built at all")

#: The host record's total size in bytes. ``float2`` FIRST so there is no internal
#: padding, then five uints, three floats and the six pack offsets: exactly 64 bytes
#: of members (60 rounded up to the 8-byte alignment before 2026-09-04, when the
#: ``M_ZERO`` arm's ``axis_coef`` became the ninth scalar).
PARAMS_ITEMSIZE = 64

#: The wall table this family REQUIRES ``zero_metal_axes`` to report on an admitted
#: grid: never r, never phi, z iff z is declared metallic. A consequence of ``Grid``'s
#: ``metallic_axes`` construction, checked rather than assumed because the emitted
#: mask is a function of it and an r clear would destroy the row the per-|m| rules own.
FORBIDDEN_WALL_AXES: Tuple[int, ...] = (0, 1)

__all__ = [
    "ARMS", "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_SIDE", "CURL_SUB_STEP", "FAMILY",
    "FORBIDDEN_WALL_AXES", "OVER_CEILING_BINDINGS", "PACKED_BINDINGS",
    "PACKED_POINTERS", "PACKED_VECTORS", "PARAMS_ITEMSIZE", "REPLACES",
    "SEPARATE_SCALAR_BINDINGS", "SLOT", "UNPACKED_POINTERS",
    "UNPACKED_POINTER_BINDINGS",
    "CylindricalFusedElectricPairPlan",
    "certified_cylindrical_curl_body",
    "compile_cylindrical_fused_electric_pair",
    "cylindrical_fused_electric_pair_coverage",
    "cylindrical_fused_electric_pair_source",
    "enumerate_sources",
    "params_record_dtype",
    "plan_cylindrical_fused_electric_pair",
    "refuted_separate_scalar_source",
    "refuted_thirty_second_binding",
    "refuted_unpacked_pointer_source",
    "register_arms",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------
#
# TWENTY-NINE BINDINGS: 18 curl pointers with six of them PACKED into one (so 13)
# + 15 constitutive pointers + 1 packed Params&. See the module docstring for the
# three signatures this refutes.

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

// THE float2 MEMBER COMES FIRST. Metal aligns float2 to 8 bytes, so with the
// scalars first this struct needs internal padding and the natural host record
// reads `inc_b` one word early — measured on this toolchain by the complex twin,
// and it reads as a plausible complex number rather than as garbage. The six pack
// offsets are uints and go LAST, where they add no padding at all.
struct Params {
    float2 inc_b;
    uint nx; uint ny; uint nz; uint n_elem; uint zrows;
    float dtdx; float minus_dtdx; float axis_coef;
__PACK_FIELDS__
};

kernel void cylindrical_fused_electric_pair_step(
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
    device const float*  cpml    [[buffer(12)]],
    device float2*       h0      [[buffer(13)]],
    device float2*       h1      [[buffer(14)]],
    device float2*       h2      [[buffer(15)]],
    device float2*       w0      [[buffer(16)]],
    device float2*       w1      [[buffer(17)]],
    device float2*       w2      [[buffer(18)]],
    device const float*  e0      [[buffer(19)]],
    device const float*  e1      [[buffer(20)]],
    device const float*  e2      [[buffer(21)]],
    device const float*  kp0     [[buffer(22)]],
    device const float*  km0     [[buffer(23)]],
    device const float*  kp1     [[buffer(24)]],
    device const float*  km1     [[buffer(25)]],
    device const float*  kp2     [[buffer(26)]],
    device const float*  km2     [[buffer(27)]],
    constant Params&     prm     [[buffer(28)]],
    uint idx [[thread_position_in_grid]])
{
    // THE NINE PACKED ARGUMENTS AND THE SIX PACKED VECTORS ARE UNPACKED INTO THE
    // CERTIFIED BODIES' OWN NAMES, once, before any of the lifted text runs.
    // Everything below this line is then character-for-character what
    // `cylindrical_complex.cylindrical_curl_source` and
    // `complex_fields.bloch_constitutive_source` emit, plus the wall clear and the
    // three seam lines.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    uint zrows = prm.zrows;
    float dtdx = prm.dtdx, minus_dtdx = prm.minus_dtdx, axis_coef = prm.axis_coef;
    float2 inc_b = prm.inc_b;

    // --- the curl half's PML coefficient vectors, one allocation, six offsets ---
    // READ-ONLY: nothing writes `cpml`. The radial prefix `pfx` is the one member of
    // this half's read-only set the plan REFRESHES every step, and it keeps its own
    // binding for that reason.
__PACK_PROLOGUE__

__BODY__
}
"""

#: The marker that separates a certified shader's signature from its body. Both
#: emitters end their parameter list with this exact line, so ONE anchor lifts either
#: body and a template that stopped carrying it raises here rather than splicing a
#: truncated kernel.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: The certified cylindrical curl body's last statement — the D store. The wall clear
#: goes IMMEDIATELY BEFORE it, because the array path stores D and only then
#: overwrites the wall plane; masking the register first is the same field and one
#: fewer pass. The ``u`` store precedes it and is NOT masked: ``zero_metal_D`` passes
#: ``D_COMPONENTS`` and the split-field auxiliary is not in it.
_CURL_STORE = "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;\n"


def certified_cylindrical_curl_body(bcz: int, m_arm: int, expansion: str,
                                    contract: str = shaders.CONTRACT_OFF) -> str:
    """The Dcyl ``step_D`` curl kernel's BODY, lifted from the certified emitter.

    Not a transcription: this is
    :func:`.cylindrical_complex.cylindrical_curl_source`'s own output with the
    ``#include``/helpers/signature preamble and the closing brace removed.
    ``backward`` is True and never a parameter — this pair is the D seam, and
    ``step_B``'s forward strides, its own prefix substitution and its ``Br`` axis
    clear are the magnetic twin's product.
    """
    source = cylindrical_complex.cylindrical_curl_source(
        bcz, True, m_arm, expansion, contract)
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


def cylindrical_fused_electric_pair_source(
        bcz: int, m_arm: int, zero_metal: Sequence[bool], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (z boundary, |m| arm, walls, arm, contraction).

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string here,
    for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes a source
    and nothing else, so specialisation by substitution is what keeps the emitted
    arithmetic identical to the bodies this lifts.

    ``zero_rows`` is deliberately NOT a specialisation — it rides in ``Params`` as a
    runtime uniform, exactly as the certified cylindrical curl carries it, which keeps
    this enumeration CLOSED at four sources per arm.
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
        "    // --- zero_metal_D, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The OFF-DIAGONAL for D: Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z.\n"
        "    // On a Dcyl grid only z can be walled, so this is the two TANGENTIAL\n"
        "    // components at z = 0 and nothing else. A COMPLEX zero: the array path\n"
        "    // assigns one (S:1896, :1902).\n",
        # IMPORTED, never re-spelled: one home for the D-side wall table.
        complex_fused_electric_pair.zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    # IMPORTED, never re-spelled: `update_E` has no cylindrical branch at all, so the
    # constitutive half of this pair IS the complex twin's, lift and all.
    electric = complex_fused_electric_pair.certified_constitutive_body(
        expansion, contract)
    # THE SEAM, and it is exactly three lines. The certified complex E body opens each
    # component with `float2 srcN = c_mul_field_left(gN[ii], eN[ii]);` — a RELOAD of
    # the complex displacement the curl just stored, scaled by the inverse
    # permittivity. Fused, the reload becomes the live register and NOTHING ELSE
    # MOVES: the inverse-epsilon factor stays on the right and D stays on the LEFT,
    # which is what `c_mul_field_left` means and the order the array path writes
    # (stepping.py:1011-1013) — not a commutation this transcription may make.
    for target in range(3):
        old = (f"    float2 src{target} = "
               f"c_mul_field_left(g{target}[ii], e{target}[ii]);\n")
        if old not in electric:
            raise AssertionError(
                f"the certified complex E constitutive body no longer reads its "
                f"source as {old.strip()!r}; the seam has no anchor")
        electric = electric.replace(
            old,
            f"    // THE SEAM: the register step_D just wrote, not a reload of "
            f"D{'xyz'[target]}.\n"
            f"    float2 src{target} = c_mul_field_left(v{target}, e{target}[ii]);\n")
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
        "\n    // --- update_E (stepping.update_E / _apply_constitutive_pml:2083) --\n"
        "    // NO CYLINDRICAL BRANCH: update_E (:926-945) carries none, and the\n"
        "    // cylindrical tranche MEASURED that (480/480 rows, 0 differing words).\n",
        electric,
    ))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__PACK_FIELDS__": coefficient_pack.params_fields(PACKED_VECTORS),
        "__PACK_PROLOGUE__": coefficient_pack.prologue("cpml", PACKED_VECTORS),
        "__BODY__": body,
    })


def compile_cylindrical_fused_electric_pair(
        bcz: int, m_arm: int, zero_metal: Sequence[bool], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (z boundary, |m| arm, walls, arm, mode)."""
    return compile_source(cylindrical_fused_electric_pair_source(
        bcz, m_arm, zero_metal, expansion,
        contract)).cylindrical_fused_electric_pair_step


def enumerate_sources(expansion: str,
                      contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a shipped plan can emit — SIX, keyed by a stable label.

    z PERIODIC x z METALLIC, and ``M_ZERO`` x ``M_ONE`` x ``M_MANY`` (four until
    2026-09-04, when the m = 0 arm landed). The wall triple is NOT a free axis:
    ``zero_metal_axes`` cannot report a wall on a periodic axis and cannot report one
    on r or phi at all, so the wall flag is a FUNCTION of ``bcz`` and the
    enumeration stays closed. A label that cannot be built is not fingerprinted as if
    it could.
    """
    out: Dict[str, str] = {}
    for bcz in (templates.PERIODIC, templates.METALLIC):
        walls = (False, False, bcz == templates.METALLIC)
        for m_arm, arm_name in ((cylindrical_complex.M_ZERO, "m0"),
                                (cylindrical_complex.M_ONE, "m1"),
                                (cylindrical_complex.M_MANY, "mmany")):
            label = ("metallic_z" if bcz == templates.METALLIC
                     else "periodic_z") + f":{arm_name}"
            out[label] = cylindrical_fused_electric_pair_source(
                bcz, m_arm, walls, expansion, contract)
    return out


# ---------------------------------------------------------------------------
# The three refuted signatures
# ---------------------------------------------------------------------------

_UNPACKED_COMPLEX: Tuple[str, ...] = ("f0", "f1", "f2", "u0", "u1", "u2",
                                      "h0", "h1", "h2", "w0", "w1", "w2")
_UNPACKED_COMPLEX_READ: Tuple[str, ...] = ("g0", "g1", "g2", "pfx", "c0", "c2")
_UNPACKED_REAL_READ: Tuple[str, ...] = (
    "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
    "e0", "e1", "e2", "kp0", "km0", "kp1", "km1", "kp2", "km2")


def _signature_lines(with_struct: bool) -> Tuple[List[str], int]:
    lines: List[str] = []
    slot = 0
    for name in _UNPACKED_COMPLEX:
        lines.append(f"    device float2*       {name:<10}[[buffer({slot})]],")
        slot += 1
    for name in _UNPACKED_COMPLEX_READ:
        lines.append(f"    device const float2* {name:<10}[[buffer({slot})]],")
        slot += 1
    for name in _UNPACKED_REAL_READ:
        lines.append(f"    device const float*  {name:<10}[[buffer({slot})]],")
        slot += 1
    if slot != UNPACKED_POINTERS:  # pragma: no cover - arithmetic guard
        raise AssertionError((slot, UNPACKED_POINTERS))
    if with_struct:
        lines.append(f"    constant Params&     prm       [[buffer({slot})]],")
        slot += 1
    return lines, slot


def _touching_body(guard: str, scale: str) -> List[str]:
    """A body that READS every input and WRITES every output.

    A body the compiler could drop would let dead-code elimination decide the answer,
    and the answer being measured is about the SIGNATURE.
    """
    reads = list(_UNPACKED_COMPLEX_READ)
    real_reads = list(_UNPACKED_REAL_READ)
    complex_touch = " + ".join(f"{name}[0]" for name in reads)
    real_touch = " + ".join(f"{name}[0]" for name in real_reads)
    return [f"    {guard}",
            f"    float2 touch = {complex_touch};",
            f"    float weight = {real_touch};",
            *(f"    {name}[idx] = {name}[idx] + touch * weight * {scale};"
              for name in _UNPACKED_COMPLEX)]


def refuted_unpacked_pointer_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 34-binding signature this platform REFUSES — 33 pointers plus ``Params``.

    THIS IS THE MEASUREMENT THE PACK EXISTS FOR. It is the fused pair built the way
    every other D/E product on this backend is built — the non-pointer arguments
    packed into one ``constant Params&``, every vector its own pointer — and it is
    over the ceiling by THREE bindings. Not shipped and not launchable: it exists so
    the gate can compile it and require the failure.
    """
    lines, slot = _signature_lines(with_struct=True)
    if slot != UNPACKED_POINTER_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((slot, UNPACKED_POINTER_BINDINGS))
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "struct Params {",
        "    float2 inc_b;",
        "    uint nx; uint ny; uint nz; uint n_elem; uint zrows;",
        "    float dtdx; float minus_dtdx; float axis_coef;",
        "};",
        "",
        "kernel void refuted_unpacked_pointers(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        *_touching_body("if (idx >= prm.n_elem) { return; }", "prm.dtdx"),
        "}",
        "",
    ))


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 42-binding separate-scalar signature this platform REFUSES.

    The shape the SCALAR packing every shipped pair already does exists to avoid, kept
    because a family that reported only the pointer refusal would leave a reader to
    wonder whether the scalars were the problem. They are not: even with them packed
    the pointers alone are three over.
    """
    lines, slot = _signature_lines(with_struct=False)
    for name in ("nx", "ny", "nz", "n_elem", "zrows"):
        lines.append(f"    constant uint&       {name:<10}[[buffer({slot})]],")
        slot += 1
    for name in ("dtdx", "minus_dtdx", "axis_coef"):
        lines.append(f"    constant float&      {name:<10}[[buffer({slot})]],")
        slot += 1
    lines.append(f"    constant float2&     inc_b     [[buffer({slot})]],")
    slot += 1
    if slot != SEPARATE_SCALAR_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((slot, SEPARATE_SCALAR_BINDINGS))
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "kernel void refuted_separate_scalars(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        *_touching_body("if (idx >= n_elem + zrows) { return; }", "dtdx"),
        "    f0[idx] = f0[idx] + inc_b * (minus_dtdx + axis_coef) * float(nx + ny + nz);",
        "}",
        "",
    ))


def refuted_thirty_second_binding(contract: str = shaders.CONTRACT_OFF) -> str:
    """A 32-binding kernel, REQUIRED to fail — the ceiling itself, located.

    The shipped signature sits at 29 with two to spare, so "one more buffer is
    refused" is not a statement about THIS kernel; it is a statement about the
    PLATFORM, and it is what turns :data:`MAX_BUFFER_BINDINGS` from a constant this
    module cites into a number this host returned.
    """
    pointers = [f"    device float2* p{n:<2} [[buffer({n})]],"
                for n in range(OVER_CEILING_BINDINGS - 1)]
    touch = " + ".join(f"p{n}[0]"
                       for n in range(OVER_CEILING_BINDINGS - 1))
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        shaders.contraction_pragma(contract),
        "",
        "struct Params { uint n_elem; float dtdx; };",
        "",
        "kernel void refuted_thirty_second_binding(",
        *pointers,
        f"    constant Params& prm [[buffer({OVER_CEILING_BINDINGS - 1})]],",
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= prm.n_elem) { return; }",
        f"    float2 touch = {touch};",
        "    p0[idx] = p0[idx] + touch * prm.dtdx;",
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def cylindrical_fused_electric_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span Dcyl ``step_D`` -> wall -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with that
    half's reasons, prefixed so a reader can tell which side said it. That is what
    makes the expansion-probe clause, the complex-storage clause (the partition
    against the real m = 0 product), the nr >= 2 clause and the Dcyl geometry
    clauses INHERITED rather than restated.
    """
    reasons: List[str] = []

    curl = cylindrical_complex.cylindrical_complex_pml_curl_coverage(
        fields, pml, CURL_SUB_STEP, residency, probe)
    if not curl.covered:
        reasons.extend(f"cylindrical curl half: {reason}" for reason in curl.reasons)
    electric = cylindrical_complex.cylindrical_complex_constitutive_coverage(
        fields, pml, CONSTITUTIVE_SIDE, residency, probe)
    if not electric.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D unless
    # the deposit repair brackets the launch — which for this family it does, and
    # which is the only reason the cell is worth anything at all: all sixteen corpus
    # rows carry one. A MAGNETIC source is injected in the B/H half and does NOT
    # disqualify this pair. IGNORANCE IS NEVER AN EMPTY SET — `Fields` does not hold
    # the source list, so an undeclared `sources` is a refusal and not an assumed ().
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the "
            f"driver injects it BETWEEN step_D and update_E "
            f"(driver.py:3294-3299), which is work inside the seam this "
            f"dispatch closes"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO SYMMETRY FILLS. `Grid` refuses a mirror plane on a Dcyl cell outright
    # (grid.py:648-653) and both halves restate that refusal, so on this family the
    # fills are dead by construction rather than by clause. RE-CHECKED HERE ANYWAY,
    # not inferred: this module's coverage may not be read off another module's guard.
    mirrored = getattr(grid, "is_mirrored", None)
    if callable(mirrored):
        for axis in range(3):
            if bool(mirrored(axis)):
                reasons.append(
                    f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                    f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                    f"(driver.py:3300-3302) and neither is carried by this family")

    # zero_metal_D is CARRIED INLINE, so the grid must be able to answer which axes
    # are walled. A grid that cannot answer would silently be treated as unwalled,
    # which is a plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # THE WALL TABLE MUST BE THE Dcyl ONE. On a cylindrical grid `Grid` cannot report
    # r or phi as walled, and the emitted mask is a FUNCTION of this table — a clear
    # on the r = 0 row would overwrite the row the per-|m| rules own with a zero the
    # array path never writes. Refused by name rather than emitted.
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

class CylindricalFusedElectricPairPlan(KernelPlan):
    """ONE dispatch that performs three driver passes, complex D never leaving a
    register — plus the host prefix pre-pass the cylindrical family cannot avoid.

    THE ONE DIVERGENCE FROM :class:`.plans.KernelPlan` is the certified cylindrical
    curl plan's, unchanged: the radial prefix is a SEQUENTIAL float32 host scan whose
    summation order defines the answer, so :meth:`run` performs a pre-pass —
    ``sync_out`` of Hy, the scan, ``sync_in`` of the prefix mirror — and only then
    calls ``super().run``.

    ``launches_per_run`` is the base's 1: the pre-pass is host work and residency
    traffic, not a kernel launch. :attr:`prefix_syncs` counts it separately, so a gate
    can assert it HAPPENED rather than trust that it did.
    """

    __slots__ = ("residency", "volumes", "bcz", "m_arm", "zero_rows", "zero_metal",
                 "shape", "n_elem", "dtdx", "expansion", "params", "scratch",
                 "pack_layout", "prefix_syncs", "_prefix_host", "_prefix_name",
                 "_prefix_source_name", "_source_map", "_xp")

    family = "cylindrical complex fused PML D-curl/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "bcz", "m_arm", "zero_rows", "zero_metal", "expansion")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Any, volumes: Sequence[str], bcz: int,
                 m_arm: int, zero_row_count: int, zero_metal: Sequence[bool],
                 shape: Sequence[int], dtdx: float, expansion: str, params: Any,
                 pointers: Sequence[Any], functions: Mapping[str, Any],
                 prefix_host: Any, prefix_name: str, prefix_source_name: str,
                 source_map: Mapping[str, Any], xp: Any, pack_layout: Any,
                 scratch: Any = None) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.bcz = int(bcz)
        self.m_arm = int(m_arm)
        self.zero_rows = int(zero_row_count)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        # The COMPLEX CELL count, which is also what the dispatch is sized from.
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.expansion = str(expansion)
        self.params = params
        self.scratch = scratch
        self.pack_layout = pack_layout
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

        Separated from :meth:`run` so a gate can call it explicitly and count it. It
        is NOT an optional step: :meth:`run` always calls it. A ``prepare()`` a caller
        had to remember is exactly the stale-mirror class this package refuses to
        comment on — a forgotten call is a smooth, plausible, wrong field.
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
    """The host record's dtype — the ``float2`` member FIRST, itemsize 64.

    ONE HOME FOR THE LAYOUT, because getting it wrong is a silent wrong answer rather
    than a crash. The offsets are stated EXPLICITLY even though the float2-first order
    makes them the natural ones, so the record cannot drift into agreeing with Metal
    only by accident; ``itemsize`` is stated because the members total 60 bytes and
    Metal's struct is 8-byte aligned.
    """
    import numpy as np  # noqa: PLC0415

    names = ["inc_b", "nx", "ny", "nz", "n_elem", "zrows", "dtdx", "minus_dtdx",
             "axis_coef"]
    formats: List[Any] = [("<f4", 2), "<u4", "<u4", "<u4", "<u4", "<u4", "<f4", "<f4",
                          "<f4"]
    offsets = [0, 8, 12, 16, 20, 24, 28, 32, 36]
    cursor = 40
    for name, kind in coefficient_pack.record_dtype_fields(PACKED_VECTORS):
        names.append(name)
        formats.append(kind)
        offsets.append(cursor)
        cursor += 4
    return np.dtype({"names": names, "formats": formats, "offsets": offsets,
                     "itemsize": PARAMS_ITEMSIZE})


def _params_tensor(shape: Sequence[int], dtdx: float, zero_row_count: int,
                   minus_dtdx: float, inc_b: Sequence[float],
                   offsets: Sequence[int], device: str, axis_coef: float) -> Any:
    """The nine non-pointer arguments AND the six pack offsets as one 64-byte record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason the twin's
    ``_params_tensor`` gives: :meth:`.Residency.mirror` binds float32 and complex64
    volumes and refuses anything else BY NAME. This record is neither, so it is built
    here, once, at plan time. Nothing on the launch path allocates.

    THE OFFSET FIELDS ARE THE PACK'S WHOLE COST. Six uints, in the pack's own order,
    and they are what buys back five buffer bindings. The field order here and the
    struct order in :data:`_TEMPLATE` are BOTH generated from :data:`PACKED_VECTORS`,
    so the two cannot drift into a reinterpretation.
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
    offsets = tuple(int(offset) for offset in offsets)
    if len(offsets) != len(PACKED_VECTORS):  # pragma: no cover - arithmetic guard
        raise AssertionError((len(offsets), len(PACKED_VECTORS)))
    for name, offset in zip(PACKED_VECTORS, offsets):
        record[f"off_{name}"] = offset
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_cylindrical_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None, params: Optional[Any] = None,
        ) -> Optional[CylindricalFusedElectricPairPlan]:
    """Build the Dcyl fused D/E plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have
    stepped correctly.

    ``functions`` is the KERNEL mutation seam and ``params`` the HOST one this family
    adds — the packed offsets are host arithmetic, so a leg that corrupts one has to be
    able to hand this builder a struct the shipped code would not have written.
    Dropping either is not a silent slowdown but a silent DISARMING.
    """
    if not cylindrical_fused_electric_pair_coverage(
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

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step.
    flux = [bind_complex(name, getattr(fields, name)) for name in spec["targets"]]
    auxiliary = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                 for name in spec["targets"]]
    magnetic = [bind_complex(name, getattr(fields, name))
                for name in spec["sources"]]

    # THE PREFIX BUFFER IS THE PLAN'S OWN and is allocated once. It must be a stable
    # host array: `Residency.mirror` refuses re-registering a name against a DIFFERENT
    # host array, and `cylindrical_rderiv_prefix` returns a fresh array every call.
    prefix_host = numpy.zeros(
        cylindrical_complex.prefix_shape(CURL_SUB_STEP, grid.shape),
        dtype=numpy.complex64)
    prefix_name = f"cylfusedE:pfx:{CURL_SUB_STEP}"
    prefix = bind_complex(prefix_name, prefix_host)

    # The two i*m/r rows are GRID INVARIANTS — Yee shift, radial extent, m, Courant —
    # so they are built ONCE here.
    imr_rows = []
    for index, _register, sign in cylindrical_complex.IMR_TERMS[CURL_SUB_STEP]:
        target = spec["targets"][index]
        row = numpy.ascontiguousarray(
            cylindrical_complex.imr_coefficient_row(
                numpy, target, sign, m, dtdx, int(grid.shape[0]),
                numpy.complex64).reshape(-1))
        imr_rows.append(bind_complex(f"cylfusedE:imr:{CURL_SUB_STEP}:{target}", row))

    # THE D CURL TAKES THE INTEGER LATTICE and the E constitutive the HALF-INTEGER one
    # (SUB_STEPS['step_D']['suffix'] is ''; CONSTITUTIVE_SIDES['E']['half_integer'] is
    # True). That is the OPPOSITE pairing to the B/H twin, and the kernel cannot tell.
    #
    # THE PACK IS THE CURL GROUP AND NOTHING ELSE. Six read-only vectors, one buffer,
    # six element offsets in Params. The constitutive group stays six separate
    # pointers because it does not have to move: packing it too would buy margin
    # nothing needs and put two packs in one signature for a reader to keep apart.
    curl_pack, pack_layout = coefficient_pack.packed_mirror(
        residency, f"pml_pack:cyl_complex_curl:{CURL_SUB_STEP}", numpy,
        PACKED_VECTORS,
        [getattr(pml, f"{stem}_{axis}{spec['suffix']}")
         for axis in "xyz" for stem in ("kms", "sinv")])
    volumes.append(f"pml_pack:cyl_complex_curl:{CURL_SUB_STEP}")

    stored = [bind_complex(name, getattr(fields, name)) for name in side["targets"]]
    workspace = [bind_complex(name, getattr(fields, name)) for name in side["aux"]]
    inverse_epsilon = [
        bind_real("inv_eps_" + name, fields.inverse_epsilon_for(name))
        for name in side["targets"]]
    constitutive_coefficients = [
        bind_real(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"))
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + magnetic + [prefix] + imr_rows + [curl_pack]
                + stored + workspace + inverse_epsilon
                + constitutive_coefficients)
    if len(pointers) != PACKED_POINTERS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_POINTERS))
    if len(pointers) + 1 != PACKED_BINDINGS:  # pragma: no cover - an invariant
        raise AssertionError((len(pointers), PACKED_BINDINGS))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_cylindrical_fused_electric_pair(
                bcz, arm, walls, expansion, mode)

    record = (_params_tensor(grid.shape, dtdx, zero_row_count, minus_dtdx, inc_b,
                             pack_layout.offsets, residency.device, axis_coef)
              if params is None else params)
    return CylindricalFusedElectricPairPlan(
        residency, volumes, bcz, arm, zero_row_count, walls, grid.shape, dtdx,
        expansion, record, pointers, selected, prefix_host, prefix_name,
        cylindrical_complex.PREFIX[CURL_SUB_STEP]["component"],
        {name: getattr(fields, name) for name in spec["sources"]}, numpy,
        pack_layout, getattr(fields, "scratch", None))


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"cylindrical complex fused electric pair cannot fill {slot}",))
    return cylindrical_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[CylindricalFusedElectricPairPlan]:
    if slot != SLOT:
        return None
    return plan_cylindrical_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    ``plan_step`` assigns at most one arm per slot and this product spans three, so
    there is no slot it could claim through the arm table. Registering unwired keeps
    it ENUMERABLE for the disjointness sweep (``arms.registered``) while
    ``arms.arms_for`` skips it. THE OTHER HALF IS THE ABSORB TABLE:
    ``launch._install_fused_pairs`` reaches this row precisely BECAUSE it is
    registered, and ``launch.FUSED_PAIR_ARMS[FAMILY]`` declares which arm each of the
    two slots implements — which is what lets the deposit repair bracket it.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "cylindrical complex fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="cylindrical complex fused electric D/E pair: ",
                          noun="cylindrical complex fused PML D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
