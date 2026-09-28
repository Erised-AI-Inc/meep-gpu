"""The THIRD fused Metal kernel: ``step_B`` welded into ``update_H``.

THE MAGNETIC HALF of :mod:`.fused_dispersive_pair`, and the first fused product on
this backend whose seam is the B/H one. ONE DISPATCH FOR ALL THREE COMPONENTS: the
plain magnetic constitutive carries no pole buffers and no inverse-mu volume, so
the whole B half fits in a single launch — 27 pointers plus one packed
``constant Params&``, 28 of the 31 bindings the platform allows (see "THE
SIGNATURE" below). The separate composition for this configuration is TWO device
dispatches (the certified curl, then the certified constitutive) plus one host
wall pass on a walled run; this is ONE, and the flux density never leaves a
register.

EVERYTHING HERE IS TRANSCRIBED, and the two halves are not re-spelled at all —
they are SPLICED FROM THE CERTIFIED EMITTERS' OWN OUTPUT:

* the B curl half — :func:`.shaders.curl_source` with ``backward=False``
  (shaders.py:242-260), whose body is lifted verbatim by
  :func:`certified_curl_body`. Its ghost gather is :func:`.templates.ghost` and its
  cell-0 mask :func:`.templates.ownership_mask`, reached through that emitter
  rather than through a second copy;
* the wall clear — ``stepping._zero_metal`` (stepping.py:2206-2247) restricted to
  ``B_COMPONENTS`` (stepping.py:183), carried inline through
  :data:`_ZERO_METAL_ROWS`;
* the constitutive half — :func:`.shaders.constitutive_source` on side ``"H"``
  (shaders.py:363-390), whose body is lifted verbatim by
  :func:`certified_constitutive_body`, itself the transcription of
  ``stepping.update_H`` and ``stepping._apply_constitutive_pml``
  (stepping.py:2130-2143).

Because both halves are LIFTED rather than retyped, "the fused arithmetic is the
certified arithmetic" is a property of the construction that a reader can check by
running :func:`certified_curl_body` — and the gate's ``transcription`` leg checks
exactly that, because a construction is a hypothesis until something compares the
strings.

===========================================================================
THE SEAM, AND WHAT SITS INSIDE IT — measured from the driver, not assumed
===========================================================================

The driver runs FOUR passes between ``step_B`` and ``update_H``
(driver.py:3279-3289, mirrored in :data:`.coverage.RESIDENCY_ORDER`):
``step_B`` -> MAGNETIC SOURCES -> ``fill_symmetry_bc_B`` -> ``zero_metal_B`` ->
``fill_folded_far_ghosts_B`` -> ``update_H``.

**THE MAGNETIC SOURCE SLOT IS REAL, AND IT IS THE BINDING CLAUSE OF THIS FAMILY.**
This is worth stating plainly because the opposite was believed when this module
was scoped: the B/H seam was expected to be empty, on the grounds that the D/E
seam's source injection is what caps the first two fused pairs. It is not empty.
``driver.step`` injects magnetic currents at driver.py:3283-3284, in the exact
mirror of the electric slot at :3295-3299, and MEEP's own order is what puts it
there — ``step_db(B_stuff)``, ``step_source(B_stuff)``, ``step_boundaries(B_stuff)``
(step.cpp:64-72). A fused pair that spanned the slot would consume a
pre-injection B, which on a run with a magnetic current is a phase error of the
same class the driver's docstring measures at 1.6e-01 against CPU MEEP.

So the clause is REFUSED BY NAME, exactly as the electric clause is on the D side,
and with the polarity flipped: an ELECTRIC source is injected in the D/E half and
does NOT disqualify this pair.

* **``zero_metal_B`` — CARRIED INLINE.** The one pass whose arithmetic is a
  transcribable select on the value already in the register. ``_zero_metal`` writes
  zero into stored cell 0 of every component whose Yee shift on a walled axis is 0
  (stepping.py:2244-2247), and for B that is the DIAGONAL — ``IYEE_SHIFTS``
  (fields.py:214-219) gives Bx (0,1,1), By (1,0,1), Bz (1,1,0), so Bx clears on an
  x wall, By on a y wall, Bz on a z wall. That is the EXACT COMPLEMENT of the D
  side's off-diagonal table (Dy/Dz on x, Dx/Dz on y, Dx/Dy on z), and the
  difference is not cosmetic: a fused pair that reused the D table here would clear
  the wrong two components on every walled run. :data:`_ZERO_METAL_ROWS` is
  therefore spelled HERE rather than imported from
  :mod:`.fused_dispersive_pair`, and the gate carries a mutation that swaps one
  row for its D-side partner.
  ``fu_B`` is deliberately NOT masked: the array-path pass touches ``B_COMPONENTS``
  only (stepping.py:2250).
* **``fill_symmetry_bc_B`` and ``fill_folded_far_ghosts_B`` — REFUSED** through
  both halves' own "a mirror plane is active" clause (coverage.py clause 5), which
  is what makes both fill slots dead for every configuration this family admits.
  A folded B/H seam is the natural sequel to this product and is NOT built here.

===========================================================================
THE SIGNATURE — 27 pointers plus one packed struct
===========================================================================

Counting what an all-three-component fused B/H kernel needs:

    3 B  +  3 fu_B  +  3 E  +  6 curl coefficients (kms/sinv per axis)
  + 3 H  +  3 f_w_H  +  6 constitutive coefficients (kps/kms per axis)   =  27

and there is NO inverse-mu volume: the H product is ``source = B`` with mu = 1
baked into the array path too (shaders.py:355-360 — ``stepping.update_H`` passes
``fields.Bx``). The certified constitutive kernel keeps three inverse-epsilon
pointers on BOTH sides so its buffer layout is one layout; this family serves the
H side only, has no second side to share a layout with, and DROPS them. That is
the whole reason 27 rather than 30.

27 pointers plus five separate scalars is 32 bindings, which is over
:data:`.device.MAX_BUFFER_BINDINGS` — buffer attribute indices must be 0..30 and a
32nd binding is a COMPILE ERROR. Packing the five scalars into one
``constant Params&`` buys back four slots and lands at 28. Both halves of that are
MEASURED by the gate's ``binding_ceiling`` leg: the 32-binding signature must FAIL
and the packed 28-binding one must COMPILE, LAUNCH and read every field back.

``dtdx`` rides in the struct as float32, for the reason
:mod:`.folded_fused_pair` gives: the array path multiplies a float32 volume by a
Python float and NumPy's weak promotion casts that double to float32,
``constant float&`` does the same, and ``numpy.float32(dtdx)`` in the struct is
the same bits again.

===========================================================================
THE TWO YEE SUB-LATTICES, AND WHY THEY ARE CROSSED HERE
===========================================================================

The B curl reads the HALF-INTEGER split-field coefficients ``kms_a_h``/``sinv_a_h``
and the H constitutive reads the INTEGER ``kps_a``/``kms_a``
(launch.py:109-124 ``SUB_STEPS['step_B']['suffix'] == '_h'``; ``CONSTITUTIVE_SIDES['H']
['half_integer'] is False``, coverage.py:79-93). That is the OPPOSITE pairing to
the D/E fused pair, where the curl takes the integer lattice and the constitutive
the half-integer one. The kernel takes twelve coefficient pointers and never asks
which lattice they came from, so a swap on either group is a silent half-cell
error in the absorber profile — converged, smooth and wrong — and the gate carries
one host mutation for each group.

NOT WIRED, for the reason the first two fused pairs are not. ``STEP_ORDER`` assigns
at most one arm per slot and this product spans THREE of them (``step_B``,
``zero_metal_B``, ``update_H``); there is no slot it can claim without a
composition rule nothing has measured, so it registers ``wired=False``. The
composer cannot select it, the disjointness sweep can still enumerate it, and no
existing arm's behaviour changes. ``meep_gpu.fastpath.plan_fast_path`` still
returns ``None`` on every branch.

WHAT THE CORPUS SAYS THIS IS WORTH, measured rather than argued, from the 186-row
census (``parity/meep_gpu/results/metal_coverage_recut_2026-08-16``):

    admit BOTH halves (pml_curl@step_B and constitutive@update_H)   46 rows
    ... and declare NO MAGNETIC source                              24 rows
    ... and are unfolded (all 24 already are)                       24 rows

so the magnetic-source clause costs 22 of the 46, and 24 is what this family can
reach. Beside it, the same census gives the D/E plain seam 26 rows admitting both
halves and 0 surviving its electric-source clause. THE B SIDE IS WORTH 24 ROWS
AGAINST THE D SIDE'S 0 — the seam being cheaper is real and is the reason this
product exists — but it is 24 and not the 45 that the count of
``pml_curl@step_D`` admissions suggested before the source slot was read out of
the driver. That number is reported beside the gate rather than left to be
discovered.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage, MAGNETIC_FIELD_TYPE
from . import shaders
from .coverage import constitutive_coverage, pml_curl_coverage, zero_metal_axes
from .device import Residency, compile_source
from .plans import KernelPlan
from .. import deposit_repair as _deposit_repair

#: Does this product bracket its fused launch with the deposit repair? YES, and the
#: wiring this claims is ``launch._install_fused_pair`` (``launch.py:645``), which is
#: reached for this family and no other on this track: ``launch.FUSED_PAIR_SEAMS``
#: maps ``step_B`` to ``update_H``, and ``launch.FUSED_PAIR_ARMS`` carries exactly one
#: row -- ``fused_magnetic_pair: ("PML", "ordinary")`` -- so every other Metal weld is
#: refused at the absorb table before its predicate is asked. When
#: ``deposit_repair.in_seam_sources`` is non-empty that installer puts a
#: ``LeadingRepairPlan`` in this pair's FIRST slot (``step_B``), which saves ``Hx/Hy/Hz``
#: and ``f_w_H*`` at every deposit point immediately before the launch, and a
#: ``TrailingRepairPlan`` in the SECOND (``update_H``), which recomputes them after the
#: driver has injected (``driver.py:3283-3284``), filled symmetry and cleared walls.
#:
#: THE PRECONDITION THIS PRODUCT MEETS is that its constitutive half is POINTWISE.
#: ``update_H`` is ``H = B/mu`` with mu = 1, cell by cell: one
#: ``_apply_constitutive_pml(H, B, kps, kms, f_w_H)`` per component and nothing else
#: (``stepping.py:947-951``), and the fused kernel's own half reads the register
#: ``step_B`` just wrote at the SAME cell rather than a neighbour. So a whole-grid
#: launch against an uninjected B is wrong ONLY at the deposit points.
#: ``deposit_repair.apply``'s B branch recomputes exactly that expression --
#: ``constitutive = fields.B*`` with the same ``_constitutive_coefficients(pml, axis,
#: half_integer=False)`` and the same operand order -- from the final injected B.
#:
#: THE TWO NON-POINTWISE CONFIGURATIONS deposit_repair refuses by name -- an
#: off-diagonal chi1inv row and an instantaneous chi2/chi3 -- are ELECTRIC properties
#: of ``update_E`` and cannot reach this seam; the clause below consults
#: ``deposit_repair.repairable`` per run rather than assuming that.
#:
#: THE WALL CLEAR IS WHY THE REPAIR NEEDS NO WALL HANDLING. This kernel carries
#: ``zero_metal_B`` inline and therefore clears BEFORE the injection, where the driver
#: clears after it (``driver.py:3295``) -- but that host pass is unconditional rather
#: than behind a ``dispatch`` consult, so by the second consult B is already injected,
#: filled and cleared and the repair reads a final field. A deposit that lands on a
#: walled cell is therefore recomputed from the zero the host clear left, which is what
#: the array path computes too.
#:
#: IT IS NOT A HINT. A product that passes True without those wrappers computes the
#: constitutive half against a pre-injection field and reports success, which is the
#: exact failure ``deposit_repair`` exists to prevent. Flag and wiring move together,
#: and ``gate_metal_fused_magnetic_pair.py`` leg ``deposit`` measures the pair: it
#: byte-compares complete driver steps with a real magnetic ``VolumeSource`` in the
#: seam, and its null control -- the same launch with the trailing repair replaced by
#: the ``NoopPlan`` this flag's False branch would leave there -- MUST diverge.
CARRIES_DEPOSIT_REPAIR = True

FAMILY = "fused_magnetic_pair"

#: The sub-step slot this arm is registered on. It spans three; it holds a row on
#: the first, so a refusal is NAMED on the slot the fusion starts at rather than
#: being invisible to the table.
SLOT = "step_B"

#: The driver passes one launch of this plan performs, in driver order
#: (driver.py:3281-3289). Declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("step_B", "zero_metal_B", "update_H")

#: How many bindings the all-three-component shape needs with the five scalars
#: bound SEPARATELY, and how many with them packed. Spelled as data so the gate
#: can compile the refuted signature from the same number this docstring argues
#: from.
SEPARATE_SCALAR_BINDINGS = 32
PACKED_BINDINGS = 28

__all__ = [
    "FAMILY", "PACKED_BINDINGS", "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "MetalFusedMagneticPairPlan", "certified_constitutive_body",
    "certified_curl_body", "compile_fused_magnetic_pair",
    "fused_magnetic_pair_source", "metal_fused_magnetic_pair_coverage",
    "plan_metal_fused_magnetic_pair", "refuted_separate_scalar_source",
    "zero_metal_mask",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params { uint nx; uint ny; uint nz; uint n_elem; float dtdx; };

kernel void fused_magnetic_pair_step(
    device float*       f0      [[buffer(0)]],
    device float*       f1      [[buffer(1)]],
    device float*       f2      [[buffer(2)]],
    device float*       u0      [[buffer(3)]],
    device float*       u1      [[buffer(4)]],
    device float*       u2      [[buffer(5)]],
    device const float* g0      [[buffer(6)]],
    device const float* g1      [[buffer(7)]],
    device const float* g2      [[buffer(8)]],
    device float*       h0      [[buffer(9)]],
    device float*       h1      [[buffer(10)]],
    device float*       h2      [[buffer(11)]],
    device float*       w0      [[buffer(12)]],
    device float*       w1      [[buffer(13)]],
    device float*       w2      [[buffer(14)]],
    device const float* kmx     [[buffer(15)]],
    device const float* sinvx   [[buffer(16)]],
    device const float* kmy     [[buffer(17)]],
    device const float* sinvy   [[buffer(18)]],
    device const float* kmz     [[buffer(19)]],
    device const float* sinvz   [[buffer(20)]],
    device const float* kp0     [[buffer(21)]],
    device const float* km0     [[buffer(22)]],
    device const float* kp1     [[buffer(23)]],
    device const float* km1     [[buffer(24)]],
    device const float* kp2     [[buffer(25)]],
    device const float* km2     [[buffer(26)]],
    constant Params&    prm     [[buffer(27)]],
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
#: BEFORE it, because the array path stores B and only then overwrites the wall
#: plane; masking the register first is the same field and one fewer pass.
_CURL_STORE = "    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;\n"

#: The certified constitutive body's decode prologue, which the curl body has
#: already emitted. Splicing both would redeclare ``ii``/``i``/``j``/``k`` and the
#: kernel would not compile — the lift drops exactly this much and no more.
_CONSTITUTIVE_PROLOGUE_END = "    int i   = plane / nyi;\n"

#: ``_zero_metal`` for the B components, as (target, walled axis, flag) rows.
#:
#: A component is cleared on the wall of axis ``d`` exactly when its Yee shift
#: there is 0 (stepping.py:2256-2260, :2300). ``IYEE_SHIFTS`` (fields.py:216-217)
#: gives Bx (0,1,1), By (1,0,1), Bz (1,1,0), so these rows are THE DIAGONAL — the
#: exact complement of the D side's :data:`.fused_dispersive_pair._ZERO_METAL_ROWS`.
#: Spelled here rather than imported for that reason.
_ZERO_METAL_ROWS: Tuple[Tuple[int, int, str], ...] = (
    (0, 0, "at_x"),
    (1, 1, "at_y"),
    (2, 2, "at_z"),
)


def certified_curl_body(codes: Sequence[int],
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The ``step_B`` curl kernel's BODY, lifted from the certified emitter.

    Not a transcription: this is :func:`.shaders.curl_source`'s own output with the
    ``#include``/signature preamble and the closing brace removed. ``backward`` is
    False and never a parameter — this pair is the B seam, and ``step_D``'s negated
    strides are a different product entirely.
    """
    source = shaders.curl_source(codes, False, contract)
    if _BODY_ANCHOR not in source:
        raise AssertionError(
            "the certified curl source no longer carries the body anchor; this "
            "family lifts that body and would otherwise splice a truncated kernel")
    body = source.split(_BODY_ANCHOR, 1)[1]
    if not body.endswith("}\n"):
        raise AssertionError("the certified curl source does not end with '}'")
    return body[: -len("}\n")]


def certified_constitutive_body(contract: str = shaders.CONTRACT_OFF) -> str:
    """The ``update_H`` constitutive kernel's BODY, lifted and renamed.

    Two edits, and both are POINTER SPELLING rather than arithmetic — the same
    kind of rename :mod:`.fused_dispersive_pair` applies to its pole buffers, and
    for the same reason: fused, ``f0`` would mean B in the curl half and H in the
    constitutive half in one scope.

    * the decode prologue is dropped (the curl half already emitted it);
    * the three targets ``f0``/``f1``/``f2`` become ``h0``/``h1``/``h2``.

    THE SOURCE LINES ARE NOT EDITED HERE. ``float src0 = g0[ii];`` is left standing
    and :func:`fused_magnetic_pair_source` is what turns it into the register — so
    the seam is one visible substitution in one place, not a rename that quietly
    also moved it.
    """
    source = shaders.constitutive_source("H", contract)
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
        old, new = f"f{target}[ii]", f"h{target}[ii]"
        if old not in body:
            raise AssertionError(f"the certified constitutive body has no {old}")
        body = body.replace(old, new)
    return body


def zero_metal_mask(zero_metal: Sequence[bool], zero: str = "0.0f") -> str:
    """``stepping.zero_metal_B`` for all three targets, carried inline on the register.

    The array path stores B and then overwrites the wall plane
    (stepping.py:2247); this masks the value before it is stored and before the H
    half consumes it, which is the same field and one fewer pass. ``fu_B`` is not
    masked: ``zero_metal_B`` passes ``B_COMPONENTS`` (stepping.py:2250) and the
    split-field auxiliary is not in it.

    The select spelling is :func:`.templates.ownership_mask`'s — ``flag ? 0.0f : v``
    — because that is the form already measured to deliver an exact ``0.0`` on
    this platform.
    """
    lines = [f"    v{target} = {flag} ? {zero} : v{target};"
             for target, axis, flag in _ZERO_METAL_ROWS if bool(zero_metal[axis])]
    return "\n".join(lines) or "    // no walled axis clears a B component"


def fused_magnetic_pair_source(codes: Sequence[int], zero_metal: Sequence[bool],
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
        "    // --- zero_metal_B, carried inline (stepping._zero_metal:2232) ------\n",
        "    // The DIAGONAL for B: Bx on an x wall, By on y, Bz on z.\n",
        zero_metal_mask(zero_metal), "\n",
        _CURL_STORE, tail,
    ))

    magnetic = certified_constitutive_body(contract)
    # THE SEAM, and it is exactly three lines. The certified H body opens each
    # component with `float srcN = gN[ii];` — a RELOAD of the flux density the
    # curl just stored. Fused, that becomes the live register.
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
    return shaders.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__BODY__": body,
    })


def compile_fused_magnetic_pair(codes: Sequence[int], zero_metal: Sequence[bool],
                                contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (boundaries, walls, mode)."""
    return compile_source(fused_magnetic_pair_source(
        codes, zero_metal, contract)).fused_magnetic_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 32-binding separate-scalar signature this platform REFUSES.

    Not shipped and not launchable: it exists so the gate can compile it and
    require the failure, which is what turns :data:`SEPARATE_SCALAR_BINDINGS` from
    an argument into a measurement. The body is a trivial touch of every buffer —
    what is being measured is the SIGNATURE, and a body the compiler could drop
    would let dead-code elimination decide the answer.
    """
    written = (["f0", "f1", "f2", "u0", "u1", "u2", "h0", "h1", "h2",
                "w0", "w1", "w2"])
    read = (["g0", "g1", "g2",
             "kmx", "sinvx", "kmy", "sinvy", "kmz", "sinvz",
             "kp0", "km0", "kp1", "km1", "kp2", "km2"])
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

def metal_fused_magnetic_pair_coverage(fields: Any, pml: Any, sources: Any = None,
                                       residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_B`` -> wall -> ``update_H``?

    A conjunction of the two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said
    it. That is the construction
    :func:`.fused_dispersive_pair.metal_fused_dispersive_pair_coverage` uses.

    WHAT A BORROWER OF THIS PREDICATE INHERITS, and it is not the seam verdict.
    :mod:`.nonlinear_fused_magnetic_pair` delegates every non-nonlinear question
    here. While :data:`CARRIES_DEPOSIT_REPAIR` was False that was harmless: this
    predicate refused every in-seam source and so did the borrower. It is now True,
    so what this returns for a magnetic source is an ADMISSION -- and the borrower
    holds no row in ``launch.FUSED_PAIR_ARMS``, so nothing would bracket ITS launch.
    That module therefore asks the seam clause AGAIN with its own (False)
    declaration rather than inheriting this one, which is the same correction
    ``triton_kernels/nonlinear_fused_magnetic_pair.py`` made when the Triton pairs
    flipped. The clause below stays this family's own, spelled once.
    """
    reasons: List[str] = []

    curl = pml_curl_coverage(fields, pml, "step_B", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    magnetic = constitutive_coverage(fields, pml, "H", residency)
    if not magnetic.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in magnetic.reasons)

    # THE SOURCE SEAM, and for this family it is the BINDING clause on the measured
    # corpus. A MAGNETIC source is injected BETWEEN the two halves
    # (driver.py:3283-3284), so a fused pair would consume a pre-injection B.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no sources" from not being told would be the
    # over-covering refusal this clause exists to prevent. An ELECTRIC source is
    # injected in the D/E half and does NOT disqualify the pair.
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

    # THE TWO FILLS. Both halves already refuse a mirror plane (coverage.py clause
    # 5), so this restates by NAME what that refusal buys on THIS seam — a reader
    # should not have to chase a shared grid clause to learn that two driver passes
    # sit between the halves on a folded grid.
    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_B and "
                f"stepping.fill_folded_far_ghosts_B both run inside this seam "
                f"(driver.py:3285-3287) and neither is carried by this family")

    # zero_metal_B is CARRIED, so the grid must be able to answer which axes are
    # walled. `zero_metal_axes` asks `has_metallic`/`is_metallic`/`is_mirrored`; a
    # grid that cannot answer would silently be treated as unwalled, which is a
    # plane of wrong values rather than a crash.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_B cannot be carried inline")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalFusedMagneticPairPlan(KernelPlan):
    """ONE dispatch that performs three driver passes, B never leaving a register.

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

    family = "fused PML B-curl/stored-H pair"

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


def plan_metal_fused_magnetic_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[MetalFusedMagneticPairPlan]:
    """Build the fused B/H plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal, for the reason :func:`.launch.plan_pml_curl`
    gives: a configuration this kernel does not carry must fall back to the array
    path, never raise into a caller that would otherwise have stepped correctly.

    ``functions`` is the mutation seam. The gate compiles a deliberately broken
    copy of the shipped source and hands it here; dropping the argument is not a
    silent slowdown, it is a silent DISARMING — every mutation leg would then
    launch the shipped kernel and report the defect as uncaught.
    """
    if not metal_fused_magnetic_pair_coverage(fields, pml, sources, residency).covered:
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
    # so the groups are built in the order the kernel declares them.
    flux = [bind(name, getattr(fields, name)) for name in ("Bx", "By", "Bz")]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in ("Bx", "By", "Bz")]
    electric = [bind(name, getattr(fields, name)) for name in ("Ex", "Ey", "Ez")]
    magnetic = [bind(name, getattr(fields, name)) for name in ("Hx", "Hy", "Hz")]
    workspace = [bind("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in ("Hx", "Hy", "Hz")]
    # THE B CURL TAKES THE HALF-INTEGER LATTICE and the H constitutive the INTEGER
    # one (launch.py:109-124; CONSTITUTIVE_SIDES['H']['half_integer'] is False).
    # That is the OPPOSITE pairing to the D/E pair, and the kernel cannot tell —
    # the gate carries a host mutation for each group.
    curl_coefficients = [
        bind(f"pml:{stem}_{axis}_h", getattr(pml, f"{stem}_{axis}_h"), constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    constitutive_coefficients = [
        bind(f"pml:{stem}_{axis}", getattr(pml, f"{stem}_{axis}"), constant=True)
        for axis in "xyz" for stem in ("kps", "kms")]

    pointers = (flux + auxiliary + electric + magnetic + workspace
                + curl_coefficients + constitutive_coefficients)
    assert len(pointers) + 1 == PACKED_BINDINGS, (len(pointers), PACKED_BINDINGS)

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_fused_magnetic_pair(codes, walls, mode)

    # The residency layer DECLARES its device; sniffing it off a bound tensor
    # would read "mps:0" where the mirrors were built with "mps" and put the
    # struct on a nominally different device from the volumes it describes.
    dtdx = grid.dt / grid.dx
    return MetalFusedMagneticPairPlan(
        residency, volumes, codes, walls, grid.shape, dtdx,
        _params_tensor(grid.shape, dtdx, residency.device), pointers, selected)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"fused magnetic pair cannot fill {slot}",))
    return metal_fused_magnetic_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[MetalFusedMagneticPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_fused_magnetic_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_B``, ``wired=False``.

    THE FLAG IS THE WHOLE COMPOSITION STORY. ``plan_step`` assigns at most one arm
    per slot and this product spans three, so there is no slot it could claim
    without a composition rule nothing has measured — the same open question the
    two D-side fused pairs record. Registering unwired keeps it ENUMERABLE for the
    disjointness sweep (``arms.registered``) while ``arms.arms_for`` skips it, so
    ``plan_step`` cannot select it and no existing arm's selection changes.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "fused magnetic B/H pair",
                          _arm_coverage, _arm_plan,
                          prefix="fused magnetic B/H pair: ",
                          noun="fused PML B-curl/stored-H pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
