"""``step_D`` welded into the ORDINARY ``update_E``, all three components, one launch.

THE ELECTRIC TWIN OF :mod:`.fused_magnetic_pair`, and the D/E seam's plain pair —
the one shape the D side has been missing while the two ends of it shipped
separately: :mod:`.fused_dispersive_pair` fuses this curl to the POLE-AWARE E half
one component at a time, and :mod:`.folded_fused_pair` fuses it to this same
ordinary E half but only on a FOLDED grid. Neither serves an unfolded, pole-free
run, which is what the corpus mostly is.

ONE DISPATCH FOR ALL THREE COMPONENTS. The separate composition for this
configuration is TWO device dispatches (the certified curl, then the certified
constitutive) plus one host wall pass on a walled run; this is ONE, and the
displacement never leaves a register.

EVERYTHING HERE IS TRANSCRIBED, and the two halves are not re-spelled at all —
they are SPLICED FROM THE CERTIFIED EMITTERS' OWN OUTPUT, exactly as the magnetic
twin splices its two:

* the D curl half — :func:`.shaders.curl_source` with ``backward=True``
  (shaders.py:242-260), whose body is lifted verbatim by
  :func:`certified_curl_body`. Its ghost gather is :func:`.templates.ghost` and its
  cell-0 mask :func:`.templates.ownership_mask`, reached through that emitter
  rather than through a second copy;
* the wall clear — ``stepping._zero_metal`` (stepping.py:2206-2247) restricted to
  ``D_COMPONENTS`` (stepping.py:184), carried inline through
  :data:`.fused_dispersive_pair._ZERO_METAL_ROWS`, IMPORTED rather than re-derived
  for the reason :func:`.folded_fused_pair.zero_metal_lines` gives: that table is
  already the D family's off-diagonal and is already pinned against the array path;
* the constitutive half — :func:`.shaders.constitutive_source` on side ``"E"``
  (shaders.py:363-385), whose body is lifted verbatim by
  :func:`certified_constitutive_body`, itself the transcription of
  ``stepping.update_E`` and ``stepping._apply_constitutive_pml``
  (stepping.py:2130-2143).

Because both halves are LIFTED rather than retyped, "the fused arithmetic is the
certified arithmetic" is a property of the construction that a reader can check by
running :func:`certified_curl_body` — and the gate's ``transcription`` leg checks
exactly that, because a construction is a hypothesis until something compares the
strings.

===========================================================================
THE SIGNATURE — 30 pointers plus one packed struct, EXACTLY ON THE CEILING
===========================================================================

Counting what an all-three-component fused D/E kernel needs:

    3 D  +  3 fu_D  +  3 H  +  6 curl coefficients (kms/sinv per axis)
  + 3 E  +  3 f_w_E  +  3 inverse epsilon  +  6 constitutive coefficients  =  30

THAT COUNT IS NOT NEW HERE. :mod:`.fused_dispersive_pair`'s docstring spells the
same thirty and records the three compilations that priced them on this host
(torch 2.10.0, metalfe-32023.850.10) on 2026-08-19::

    35 separate bindings                       -> FAILED, 'buffer' attribute
                                                  parameter is out of bounds
    30 pointers + 5 scalars in one Params&     -> COMPILES (31 bindings)
    30 pointers                                -> COMPILES

and it says in as many words that the reason THAT family dispatches per component
"is the pole budget (up to eight ``q`` buffers per component) plus that packing
work, not a platform refusal". This family has no pole budget: the ordinary E
constitutive reads ``D * inv_eps`` and nothing else, so the count stops at 30 and
the packed shape is the whole seam in one launch.

THE THREE INVERSE-EPSILON POINTERS ARE WHY THIS IS 30 AND THE MAGNETIC TWIN IS 27,
and the difference is physics rather than layout: ``update_H`` is ``H = B`` with
mu = 1 baked into the array path (shaders.py:351-355 — ``stepping.update_H`` passes
``fields.Bx``), so the magnetic pair DROPS the three volumes the certified
constitutive keeps in its signature. ``update_E`` cannot: its product is
``source * fields.inverse_epsilon_for(component)`` (stepping.py:1011-1013) with D on
the LEFT, and all three volumes are read.

THIS FAMILY SITS EXACTLY ON :data:`.device.MAX_BUFFER_BINDINGS` — 30 pointers plus
one ``constant Params&`` is 31, and 31 is the count, not the headroom. One more
pointer of any kind is a COMPILE ERROR, which is why the five scalars ride packed
and why :data:`SEPARATE_SCALAR_BINDINGS` is spelled as data: the gate compiles that
35-binding signature and REQUIRES the failure, so the packing is a measurement
rather than a precaution. :mod:`.folded_fused_pair` lands on the same 31 with three
extra ints inside its own ``Params`` for the far-carry reflect rows; the two
families are the same buffer layout in the same order, which is deliberate.

``dtdx`` rides in the struct as float32, for the reason :mod:`.folded_fused_pair`
gives: the array path multiplies a float32 volume by a Python float and NumPy's
weak promotion casts that double to float32, ``constant float&`` does the same, and
``numpy.float32(dtdx)`` in the struct is the same bits again.

===========================================================================
THE TWO YEE SUB-LATTICES, AND WHY THEY ARE CROSSED HERE
===========================================================================

The D curl reads the INTEGER split-field coefficients ``kms_a``/``sinv_a`` and the
E constitutive reads the HALF-INTEGER ``kps_a_h``/``kms_a_h``
(launch.py ``SUB_STEPS['step_D']['suffix'] == ''``;
``CONSTITUTIVE_SIDES['E']['half_integer'] is True``, coverage.py:107-120). That is
the OPPOSITE pairing to :mod:`.fused_magnetic_pair`, where the curl takes the
half-integer lattice and the constitutive the integer one. The kernel takes twelve
coefficient pointers and never asks which lattice they came from, so a swap on
either group is a silent half-cell error in the absorber profile — converged,
smooth and wrong — and the gate carries one host mutation for each group.

===========================================================================
THE SEAM, AND WHAT SITS INSIDE IT — measured from the driver, not assumed
===========================================================================

The driver runs FIVE passes between ``step_D`` and ``update_E``
(driver.py:3292-3302, mirrored in :data:`.coverage.RESIDENCY_ORDER`):
``step_D`` -> ELECTRIC SOURCES -> ``fill_symmetry_bc_D`` -> ``zero_metal_D`` ->
``fill_folded_far_ghosts_D`` -> ``update_E``.

* **the electric source slot — CARRIED, through the deposit repair.** See
  :data:`CARRIES_DEPOSIT_REPAIR`. Until that machinery existed this seam was the
  reason the plain D/E pair was worth nothing: :mod:`.fused_magnetic_pair`'s own
  docstring records the measurement — "the same census gives the D/E plain seam 26
  rows admitting both halves and 0 surviving its electric-source clause". Those 26
  rows are this family's whole demand, and every one of them carries an electric
  source. The repair is not an optimisation here; it is the product.
* **``zero_metal_D`` — CARRIED INLINE**, on the register, through
  :func:`zero_metal_mask`. ``_zero_metal`` writes zero into stored cell 0 of every
  component whose Yee shift on a walled axis is 0 (stepping.py:2244-2247), which for
  D is Dy/Dz on an x wall, Dx/Dz on a y wall and Dx/Dy on a z wall — the
  OFF-DIAGONAL, the exact complement of the magnetic twin's diagonal. Getting that
  table from the wrong side would clear the wrong two components on every walled
  run, so it is imported from the family that already ships it rather than typed
  again here. ``fu_D`` is deliberately NOT masked: the array-path pass touches
  ``D_COMPONENTS`` only (stepping.py:2250).
* **``fill_symmetry_bc_D`` and ``fill_folded_far_ghosts_D`` — INERT, and refused
  anyway.** Both return before touching a cell unless ``grid.has_symmetry()``
  (stepping.py:1481-1483, :1565-1566), and both halves' clause 5 already refuses
  every folded grid (coverage.py ``_grid_reasons``), so no configuration this
  family admits can reach a live one. That is the same standing
  :mod:`.cylindrical_real_fused_magnetic_pair` records for its two fills, and it is
  why :data:`REPLACES` names THREE driver passes and not five. A folded D/E seam is
  a different product and it already exists: :mod:`.folded_fused_pair`.

  IT IS RESTATED BY NAME IN THE PREDICATE ANYWAY. A reader should not have to chase
  a shared grid clause to learn that two driver passes sit between the halves on a
  folded grid, and a coverage claim read off another module's guard is exactly the
  inference this package refuses to make.

NOT WIRED, for the reason every Metal fused pair is not. ``STEP_ORDER`` assigns at
most one arm per slot and this product spans THREE of them (``step_D``,
``zero_metal_D``, ``update_E``); there is no slot it can claim through the arm
table, so it registers ``wired=False``. The composer cannot select it, the
disjointness sweep can still enumerate it, and no existing arm's behaviour changes.
``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every branch. What
DOES compose it is the absorb table — ``launch.FUSED_PAIR_ARMS`` gains the row
``fused_electric_pair: ("PML", "ordinary")``, read the way every row there is read:
this family's predicate is literally ``pml_curl_coverage(fields, pml, "step_D",
...)`` AND ``constitutive_coverage(fields, pml, "E", ...)``, which are the ``PML``
curl arm's body (launch.py:422-427) and the ``ordinary`` constitutive arm's, and
the ``cart_pml_real`` row of ``parity/meep_gpu/metal_composition_matrix.py`` records
``step_D="PML"`` and ``update_E="ordinary"`` winning those two slots.

WHAT THE CORPUS SAYS THIS IS WORTH, measured rather than argued, from the 186-row
census (``parity/meep_gpu/results/metal_coverage_tranche6_2026-08-19``): the
``(PML, ordinary)`` cell at D->E is 26 seam-instances of 26 rows — THE LARGEST
UNSERVED CELL ON THE METAL BOARD. All 26 declare ``has_symmetry`` False and
``pml_active`` True, 23 of them carry a metallic wall, and all 26 carry an electric
source inside the seam.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from . import shaders
from .coverage import constitutive_coverage, pml_curl_coverage, zero_metal_axes
from .device import Residency, compile_source
from .fused_dispersive_pair import _ZERO_METAL_ROWS
from .launch import SUB_STEPS
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? YES, and on
#: this family it is load-bearing rather than incidental: every one of the 26 corpus
#: rows it reaches carries an electric source inside the seam, so with this False the
#: product would be built and serve nothing.
#:
#: The wiring this claims is ``launch._install_fused_pair`` (``launch.py``), reached
#: because ``launch.FUSED_PAIR_SEAMS`` maps ``step_D`` to ``update_E`` and
#: ``launch.FUSED_PAIR_ARMS`` carries the row ``fused_electric_pair: ("PML",
#: "ordinary")``. When ``deposit_repair.in_seam_sources`` is non-empty that installer
#: puts a ``LeadingRepairPlan`` in this pair's FIRST slot (``step_D``), which saves
#: ``Ex/Ey/Ez`` and ``f_w_E*`` at every deposit point immediately before the launch,
#: and a ``TrailingRepairPlan`` in the SECOND (``update_E``), which recomputes them
#: after the driver has injected (``driver.py:3294-3299``), filled symmetry and
#: cleared walls.
#:
#: THE PRECONDITION THIS PRODUCT MEETS is that its constitutive half is POINTWISE.
#: ``update_E`` here is ``E += kps*(D*inv_eps) - kms*f_w_E`` cell by cell — one
#: ``_apply_constitutive_pml`` per component and nothing else (stepping.py:1012-1016) —
#: and the fused kernel's own half reads the register ``step_D`` just wrote at the
#: SAME cell rather than a neighbour. So a whole-grid launch against an uninjected D
#: is wrong ONLY at the deposit points, which is exactly what a point repair can put
#: back.
#:
#: THE TWO NON-POINTWISE CONFIGURATIONS ``deposit_repair.repairable`` refuses by name
#: -- an off-diagonal chi1inv row and an instantaneous chi2/chi3 -- ARE electric
#: properties of ``update_E`` and DO reach this seam, unlike on the magnetic twin
#: where they cannot. Both are already refused by ``constitutive_coverage("E")`` and
#: by ``_grid_reasons`` clause 10, and the clause below consults
#: ``deposit_repair.seam_source_reasons`` per run anyway rather than assuming it: a
#: predicate that read its coverage off another predicate's guard is the inference
#: this package does not make.
#:
#: THE WALL CLEAR IS WHY THE REPAIR NEEDS NO WALL HANDLING. This kernel carries
#: ``zero_metal_D`` inline and therefore clears BEFORE the injection, where the driver
#: clears after it (``driver.py:3301``) -- but that host pass is unconditional rather
#: than behind a ``dispatch`` consult, so by the second consult D is already injected,
#: filled and cleared and the repair reads a final field. A deposit that lands on a
#: walled cell is therefore recomputed from the zero the host clear left, which is
#: what the array path computes too.
#:
#: IT IS NOT A HINT. A product that passes True without those wrappers computes the
#: constitutive half against a pre-injection field and reports success, which is the
#: exact failure ``deposit_repair`` exists to prevent. Flag and wiring move together,
#: and ``gate_metal_fused_electric_pair.py`` leg ``deposit`` measures the pair: it
#: byte-compares complete driver steps with a real electric ``VolumeSource`` in the
#: seam, and its null control -- the same launch with the trailing repair replaced by
#: the ``NoopPlan`` this flag's False branch would leave there -- MUST diverge.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "fused_electric_pair"

#: The sub-step slot this arm is registered on. It spans three; it holds a row on
#: the first, so a refusal is NAMED on the slot the fusion starts at rather than
#: being invisible to the table.
SLOT = "step_D"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3292-3302). Declared rather than inferred from the slot name. The two
#: fills are absent because they are INERT on every grid this family admits, not
#: because they were forgotten -- see the module docstring's seam section.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: How many bindings the all-three-component shape needs with the five scalars
#: bound SEPARATELY, and how many with them packed. Spelled as data so the gate
#: can compile the refuted signature from the same number this docstring argues
#: from. 31 IS THE CEILING ITSELF (device.MAX_BUFFER_BINDINGS), not a margin.
SEPARATE_SCALAR_BINDINGS = 35
PACKED_BINDINGS = 31

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "FAMILY", "PACKED_BINDINGS", "REPLACES",
    "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "MetalFusedElectricPairPlan", "certified_constitutive_body",
    "certified_curl_body", "compile_fused_electric_pair",
    "fused_electric_pair_source", "metal_fused_electric_pair_coverage",
    "plan_metal_fused_electric_pair", "refuted_separate_scalar_source",
    "zero_metal_mask",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

# THE BUFFER ORDER IS `folded_fused_pair`'s, POINTER FOR POINTER. That family is the
# other D/E pair at this same 31-binding count, and keeping one layout across the two
# means the host binding groups below are the same groups in the same order -- so a
# reader comparing the two plans is comparing arithmetic, not bookkeeping.
_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx; };

kernel void fused_electric_pair_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device float*       e0      [[buffer(9)]],
    device float*       e1      [[buffer(10)]],
    device float*       e2      [[buffer(11)]],
    device float*       w0      [[buffer(12)]],
    device float*       w1      [[buffer(13)]],
    device float*       w2      [[buffer(14)]],
    device const float* ie0     [[buffer(15)]],
    device const float* ie1     [[buffer(16)]],
    device const float* ie2     [[buffer(17)]],
    device const float* kmx     [[buffer(18)]],
    device const float* sinvx   [[buffer(19)]],
    device const float* kmy     [[buffer(20)]],
    device const float* sinvy   [[buffer(21)]],
    device const float* kmz     [[buffer(22)]],
    device const float* sinvz   [[buffer(23)]],
    device const float* kp0     [[buffer(24)]],
    device const float* km0     [[buffer(25)]],
    device const float* kp1     [[buffer(26)]],
    device const float* km1     [[buffer(27)]],
    device const float* kp2     [[buffer(28)]],
    device const float* km2     [[buffer(29)]],
    constant Params&    prm     [[buffer(30)]],
    uint idx [[thread_position_in_grid]])
{
    // THE FIVE SCALARS ARE UNPACKED INTO THE CERTIFIED BODIES' OWN NAMES, once,
    // before any of the lifted text runs. Everything below this line is then
    // character-for-character what `shaders.curl_source` and
    // `shaders.constitutive_source` emit, plus the wall clear.
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;

__BODY__
}
"""

#: The marker that separates a certified shader's signature from its body. Both
#: templates end their parameter list with this exact line, so ONE anchor lifts
#: either body and a template that stopped carrying it raises here rather than
#: splicing a truncated kernel.
_BODY_ANCHOR = "uint idx [[thread_position_in_grid]])\n{\n"

#: The certified curl body's last statement. The wall clear goes IMMEDIATELY
#: BEFORE it, because the array path stores D and only then overwrites the wall
#: plane; masking the register first is the same field and one fewer pass.
_CURL_STORE = "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;\n"

#: The certified constitutive body's decode prologue, which the curl body has
#: already emitted. Splicing both would redeclare ``ii``/``i``/``j``/``k`` and the
#: kernel would not compile — the lift drops exactly this much and no more.
_CONSTITUTIVE_PROLOGUE_END = "    int i   = plane / nyi;\n"


def certified_curl_body(codes: Sequence[int],
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The ``step_D`` curl kernel's BODY, lifted from the certified emitter.

    Not a transcription: this is :func:`.shaders.curl_source`'s own output with the
    ``#include``/signature preamble and the closing brace removed. ``backward`` is
    True and never a parameter — this pair is the D seam, and ``step_B``'s forward
    strides are a different product entirely (:mod:`.fused_magnetic_pair`). The
    value is read from :data:`.launch.SUB_STEPS` rather than written as a literal,
    so the two cannot drift.
    """
    source = shaders.curl_source(codes, bool(SUB_STEPS["step_D"]["backward"]),
                                 contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified curl source no longer carries the body anchor; this "
            "family lifts that body and would otherwise splice a truncated kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified curl source does not end with '}'")
    return body[: -len("}\n")]


def certified_constitutive_body(contract: str = shaders.CONTRACT_OFF) -> str:
    """The ``update_E`` constitutive kernel's BODY, lifted and renamed.

    Three edits, and every one of them is POINTER SPELLING rather than arithmetic —
    the same kind of rename :mod:`.fused_magnetic_pair` applies to its H targets and
    :mod:`.fused_dispersive_pair` to its pole buffers, and for the same reason:
    fused, ``f0`` would mean D in the curl half and E in the constitutive half in
    one scope, and ``e0`` would mean the inverse-epsilon volume in one half and the
    E target in the other.

    * the decode prologue is dropped (the curl half already emitted it);
    * the three INVERSE-EPSILON reads ``e0``/``e1``/``e2`` become ``ie0``/``ie1``/
      ``ie2``;
    * the three targets ``f0``/``f1``/``f2`` become ``e0``/``e1``/``e2``.

    THE TWO RENAMES CROSS, so they are done through a placeholder rather than in
    sequence: renaming the target first would then rename it again on the second
    pass, and renaming the inverse epsilon first without a placeholder would collide
    with the target's new name. Each step asserts its own anchor, because a rename
    that silently found nothing would emit a kernel reading the wrong volume.

    THE SOURCE LINES ARE NOT EDITED HERE. ``float src0 = g0[ii] * ie0[ii];`` is left
    standing and :func:`fused_electric_pair_source` is what turns ``g0[ii]`` into
    the register — so the seam is one visible substitution in one place, not a
    rename that quietly also moved it.
    """
    source = shaders.constitutive_source("E", contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified constitutive source no longer carries the body anchor")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified constitutive source does not end with '}'")
    body = body[: -len("}\n")]
    if _CONSTITUTIVE_PROLOGUE_END not in body:
        raise AssertionError(
            "the certified constitutive body no longer carries its decode "
            "prologue; the lift would redeclare the curl half's indices")
    body = body.split(_CONSTITUTIVE_PROLOGUE_END, 1)[1]
    for target in range(3):
        old, placeholder = f"e{target}[ii]", f"__INVERSE_EPSILON_{target}__"
        if old not in body:
            raise AssertionError(
                f"the certified E constitutive body has no {old}; the inverse "
                f"epsilon this family renames is not where the lift expects it")
        body = body.replace(old, placeholder)
    for target in range(3):
        old, new = f"f{target}[ii]", f"e{target}[ii]"
        if old not in body:
            raise AssertionError(f"the certified constitutive body has no {old}")
        body = body.replace(old, new)
    for target in range(3):
        body = body.replace(f"__INVERSE_EPSILON_{target}__", f"ie{target}[ii]")
    if "__" in body:
        raise AssertionError(
            "a rename placeholder survived into the lifted constitutive body")
    return body


def zero_metal_mask(zero_metal: Sequence[bool], zero: str = "0.0f") -> str:
    """``stepping.zero_metal_D`` for all three targets, carried inline on the register.

    The array path stores D and then overwrites the wall plane
    (stepping.py:2247); this masks the value before it is stored and before the E
    half consumes it, which is the same field and one fewer pass. ``fu_D`` is not
    masked: ``zero_metal_D`` passes ``D_COMPONENTS`` (stepping.py:2250) and the
    split-field auxiliary is not in it.

    The rows are :data:`.fused_dispersive_pair._ZERO_METAL_ROWS` — the D
    OFF-DIAGONAL (Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z), imported rather than
    re-derived. The magnetic twin's table is the complement and a family that reused
    it here would clear the wrong two components on every walled run, which is why
    neither family spells the other's.

    The select spelling is :func:`.templates.ownership_mask`'s — ``flag ? 0.0f : v``
    — because that is the form already measured to deliver an exact ``0.0`` on
    this platform.
    """
    lines = [f"    v{target} = {flag} ? {zero} : v{target};"
             for target, axis, flag in _ZERO_METAL_ROWS if bool(zero_metal[axis])]
    return "\n".join(lines) or "    // no walled axis clears a D component"


def fused_electric_pair_source(codes: Sequence[int], zero_metal: Sequence[bool],
                               contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundary triple, walls, contraction mode).

    Every constant Triton bakes into a ``tl.constexpr`` is baked into the string
    here, for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader`` takes
    a source and nothing else, so specialisation by substitution is what keeps the
    emitted arithmetic identical to the bodies this lifts.
    """
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")

    curl = certified_curl_body(codes, contract)
    if _CURL_STORE not in curl:
        raise AssertionError(
            "the certified curl body no longer stores the three targets on one "
            "line; the wall clear has no anchor to sit in front of")
    head, tail = curl.split(_CURL_STORE, 1)
    curl = "".join((
        head,
        "    // --- zero_metal_D, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The OFF-DIAGONAL for D: Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z.\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    electric = certified_constitutive_body(contract)
    # THE SEAM, and it is exactly three lines. The certified E body opens each
    # component with `float srcN = gN[ii] * ieN[ii];` — a RELOAD of the displacement
    # the curl just stored, scaled by the inverse permittivity. Fused, the reload
    # becomes the live register and NOTHING ELSE MOVES: the inverse-epsilon factor
    # stays on the right and D stays on the LEFT, which is the order the array path
    # writes (stepping.py:1011-1013) and not a commutation this transcription is
    # entitled to make.
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
    # be no `g` left in the electric half at all -- one surviving read would take the
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
    return shaders.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__BODY__": body,
    })


def compile_fused_electric_pair(codes: Sequence[int], zero_metal: Sequence[bool],
                                contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, walls, mode)."""
    return compile_source(fused_electric_pair_source(
        codes, zero_metal, contract)).fused_electric_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 35-binding separate-scalar signature this platform REFUSES.

    Not shipped and not launchable: it exists so the gate can compile it and
    require the failure, which is what turns :data:`SEPARATE_SCALAR_BINDINGS` from
    an argument into a measurement. The body is a trivial touch of every buffer —
    what is being measured is the SIGNATURE, and a body the compiler could drop
    would let dead-code elimination decide the answer.
    """
    written = ["f0", "f1", "f2", "u0", "u1", "u2", "e0", "e1", "e2",
               "w0", "w1", "w2"]
    read = ["g0", "g1", "g2", "ie0", "ie1", "ie2",
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
    lines.append(f"    constant float&     dtdx    [[buffer({slot})]],")
    slot += 1
    assert slot == SEPARATE_SCALAR_BINDINGS, (slot, SEPARATE_SCALAR_BINDINGS)
    body = "\n".join(f"    {name}[idx] = {name}[idx] * dtdx;" for name in written)
    touch = " + ".join(f"{name}[0]" for name in read)
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
        f"    float touch = {touch};",
        "    f0[idx] = f0[idx] + touch * float(nx + ny + nz);",
        body,
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_fused_electric_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                       residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_D`` -> wall -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said
    it. That is the construction
    :func:`.fused_magnetic_pair.metal_fused_magnetic_pair_coverage` uses and the one
    :func:`.fused_dispersive_pair.metal_fused_dispersive_pair_coverage` uses before
    it.
    """
    reasons: List[str] = []

    curl = pml_curl_coverage(fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = constitutive_coverage(fields, pml, "E", residency)
    if not electric.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM. An ELECTRIC source is injected BETWEEN the two halves
    # (driver.py:3294-3299), so a fused pair would consume a pre-injection D unless
    # the deposit repair brackets the launch — which for this family it does, and
    # which is the only reason the cell is worth anything at all.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent. A MAGNETIC source is
    # injected in the B/H half and does NOT disqualify the pair.
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
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILLS. Both halves already refuse a mirror plane (coverage.py clause
    # 5), so this restates by NAME what that refusal buys on THIS seam — a reader
    # should not have to chase a shared grid clause to learn that two driver passes
    # sit between the halves on a folded grid.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3300-3302) and neither is carried by this family")

    # zero_metal_D is CARRIED, so the grid must be able to answer which axes are
    # walled. `zero_metal_axes` asks `has_metallic`/`is_metallic`/`is_mirrored`; a
    # grid that cannot answer would silently be treated as unwalled, which is a
    # plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    # A polarization would make the constitutive source (D - sum P) rather than D.
    # `constitutive_coverage("E")` already refuses it; restated because this family's
    # kernel bakes the plain product and a reader should not have to chase the other
    # predicate to learn that — the same restatement `folded_fused_pair` carries.
    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the plain "
                       "constitutive product, whose source is D and not (D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFusedElectricPairPlan(KernelPlan):
    """ONE dispatch that performs three driver passes, D never leaving a register.

    ``launches_per_run`` is the base's 1 and is left there deliberately: this plan
    unpacks one argument tuple and calls one function, which is the base's whole
    contract, so the whole-step arbiter's per-cycle launch assertion needs no
    special case for it.

    ``runs`` is an alias of ``launches`` rather than a second counter, because with
    one dispatch per run the two ARE the same number and a second counter would be
    a second thing to get out of step.
    """

    __slots__ = ("residency", "volumes", "codes", "zero_metal", "shape", "dtdx",
                 "params")

    family = "fused PML D-curl/stored-E pair"

    replaces_sub_steps = REPLACES

    REPR_FIELDS = ("shape", "codes", "zero_metal")

    #: This plan launches a kernel; the composer's planned/null split reads it.
    performs_device_work = True

    def __init__(self, residency: Residency, volumes: Sequence[str],
                 codes: Sequence[int], zero_metal: Sequence[bool],
                 shape: Sequence[int], dtdx: float, params: Any,
                 pointers: Sequence[Any], functions: Mapping[str, Any]) -> None:
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        # float(): the array path multiplies a float32 volume by a Python float and
        # the packed struct holds numpy.float32 of the same value, so the two
        # scalars are the same bits.
        self.dtdx = float(dtdx)
        self.params = params
        super().__init__(dict(functions), tuple(pointers) + (params,))

    @property
    def runs(self) -> int:
        return self.launches


def _params_tensor(shape: Sequence[int], dtdx: float, device: str) -> Any:
    """The five scalars as one 20-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason
    :func:`.folded_fused_pair._params_tensor` gives: :meth:`.Residency.mirror` binds
    float32 and complex64 volumes and refuses anything else BY NAME, because a
    wider element silently reinterprets. This record is neither — it is a packed
    struct of four uints and one float that nothing on the host reads back and
    nothing on the device writes — so it is built here, once, at plan time, and
    held by the plan. Nothing on the launch path allocates.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=np.dtype([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"),
                                         ("n_elem", "<u4"), ("dtdx", "<f4")]))
    nx, ny, nz = (int(n) for n in shape)
    record[0] = (nx, ny, nz, nx * ny * nz, np.float32(dtdx))
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[MetalFusedElectricPairPlan]:
    """Build the fused D/E plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl`
    gives: a configuration this kernel does not carry must fall back to the array
    path, never raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken
    copy of the shipped source and hands it here; dropping the argument is not a
    silent slowdown, it is a silent DISARMING — every mutation leg would then
    launch the shipped kernel and report the defect as uncaught.
    """
    if not metal_fused_electric_pair_coverage(fields, pml, sources, residency).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, pml))
    walls = zero_metal_axes(grid)

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    # THE ORDER HERE IS THE SIGNATURE'S ORDER and nothing else keeps them in step,
    # so the groups are built in the order the kernel declares them. The names come
    # from `SUB_STEPS` and `CONSTITUTIVE_SIDES` rather than from six string literals,
    # which is what stops this plan and the shipped tables drifting apart.
    curl_spec = SUB_STEPS["step_D"]
    electric_spec = CONSTITUTIVE_SIDES["E"]
    flux = [bind(name, getattr(fields, name)) for name in curl_spec["targets"]]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in curl_spec["targets"]]
    magnetic = [bind(name, getattr(fields, name)) for name in curl_spec["sources"]]
    stored = [bind(name, getattr(fields, name)) for name in electric_spec["targets"]]
    workspace = [bind(name, getattr(fields, name)) for name in electric_spec["aux"]]
    inverse_epsilon = [
        bind("inv_eps_" + name, fields.inverse_epsilon_for(name), constant=True)
        for name in electric_spec["targets"]]
    # THE D CURL TAKES THE INTEGER LATTICE and the E constitutive the HALF-INTEGER
    # one (stepping.py:948 vs :1015; SUB_STEPS['step_D']['suffix'] is '' and
    # CONSTITUTIVE_SIDES['E']['half_integer'] is True). That is the OPPOSITE pairing
    # to the B/H pair, and the kernel cannot tell — the gate carries a host mutation
    # for each group.
    curl_coefficients = [
        bind(f"pml:{stem}_{axis}{curl_spec['suffix']}",
             getattr(pml, f"{stem}_{axis}{curl_spec['suffix']}"), constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + magnetic + stored + workspace + inverse_epsilon
                + curl_coefficients + constitutive_coefficients)
    assert len(pointers) + 1 == PACKED_BINDINGS, (len(pointers), PACKED_BINDINGS)

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_fused_electric_pair(codes, walls, mode)

    # The residency layer DECLARES its device; sniffing it off a bound tensor
    # would read "mps:0" where the mirrors were built with "mps" and put the
    # struct on a nominally different device from the volumes it describes.
    dtdx = grid.dt / grid.dx
    return MetalFusedElectricPairPlan(
        residency, volumes, codes, walls, grid.shape, dtdx,
        _params_tensor(grid.shape, dtdx, residency.device), pointers, selected)


# ---------------------------------------------------------------------------
# Registration — wired=False (no arm selection), routed by the absorb table
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"fused electric pair cannot fill {slot}",))
    return metal_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[MetalFusedElectricPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``.

    THE FLAG SAYS WHAT IT SAYS ON EVERY OTHER PAIR: ``plan_step`` assigns at most
    one ARM per slot and this product spans three, so there is no slot it could
    claim through the arm table. Registering unwired keeps it ENUMERABLE for the
    disjointness sweep (``arms.registered``) while ``arms.arms_for`` skips it, so
    ``_select_slot`` cannot select it and no existing arm's selection changes —
    including the ``PML`` and ``ordinary`` arms this pair's own predicate is built
    out of.

    THE OTHER HALF IS THE ABSORB TABLE. ``launch._install_fused_pairs`` reaches this
    row through ``arms.registered`` precisely BECAUSE it is registered — ``is_weld``
    with ``replaces`` covering ``update_E`` — and
    ``launch.FUSED_PAIR_ARMS["fused_electric_pair"]`` declares which arm each of the
    two slots implements.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="fused electric D/E pair: ",
                          noun="fused PML D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
