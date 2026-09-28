"""The Dcyl COMPLEX H->D seam in TWO launches: ``update_H`` folded with the device
radial scan, then the certified complex curl — no host round trip.

THE CELL. Sixteen ``cylindrical complex`` corpus rows (``h_to_d_seam.instances``,
identical on the Metal, Triton and CUDA boards), every one ``pml_active`` with ``ny = 1``,
radial extents 80 to 1050 rows and 1 to 497 columns. Today each of them pays, on every
``step_D``, a device->host->device round trip so that ``numpy`` can scan the radial
prefix (:meth:`.cylindrical_complex.CylindricalComplexCurlPlan.refresh_prefix`,
``cylindrical_complex.py:1468-1485``) — the one place that module says this backend is
worse than the CuPy one. This product removes the round trip and owns both seam slots.

=========================================================================
THE SHAPE, AND WHY IT IS TWO LAUNCHES AND NOT ONE
=========================================================================

``stepping.step_D`` on a Dcyl grid differences ``cylindrical_rderiv_prefix(Hy)`` for the
``Dz`` term (stepping.py:447, :461-462), and the prefix is a strictly serial scan down
each radial column of the POST-``update_H`` ``Hy`` — a non-local dependence that a
single pointwise dispatch cannot satisfy (a local halo is REFUSED: the prefix cannot be
algebraically collapsed, 103,175 of 124,096 words). The recompute unit is a whole
radial column, so:

* **LAUNCH 1** (``cyl_complex_hd_lead_step``, the ``update_H`` slot): every thread
  computes its OWN cell's ``H`` and split-field ``f_w_H`` through ``h_cell`` — the
  certified complex ``bloch_constitutive_step`` body lifted as a function of an
  arbitrary cell, reading ONLY pre-launch buffers — and stores them to WRITE-ONLY
  scratch. The ``r = 0`` thread of each (phi, z) column additionally walks its column,
  recomputing ``Hy`` at every row through the SAME ``h_cell`` and accumulating the
  D-side prefix (``ir0 = 0.5``, no wall row) into a prefix scratch — the
  :data:`.cylindrical_complex_scan.COLUMN_SCAN_LOOP` with the row source being the
  recompute rather than a load. NOTHING WRITTEN BY THIS LAUNCH IS READ BY IT, so no
  thread observes another thread's store; the plan rotates ``H``/``f_w_H`` to the
  scratch afterwards, exactly as :class:`.offdiag_weld_common.ScratchWeldPairPlan` does
  for the D->E welds and :mod:`.fused_hd_pair` for the Cartesian H->D one.
* **LAUNCH 2** (the ``step_D`` slot): the CERTIFIED complex cylindrical curl,
  :func:`.cylindrical_complex.cylindrical_curl_source` UNCHANGED, bound to the rotated
  ``H`` and to the prefix scratch launch 1 wrote. No new pointer, no new arithmetic.

Bit identity is the composition of two pieces each measured on its own: the pointwise
constitutive recompute (the lift is the certified body with pointer renames and the
stores moved, no arithmetic touched — :data:`CONSTITUTIVE_LIFT_EDITS`) and the
column-serial complex scan (:mod:`.cylindrical_complex_scan`, gated against
``stepping.cylindrical_rderiv_prefix`` itself). What is NEW to measure is the seam
between them — that the leader's recomputed ``Hy`` at row r is bit-identical to what
the thread at row r stored — and the gate measures it as a complete driver step.

LAUNCHES: two per seam, where the array path's complex family pays two launches AND a
host round trip of one component out and one volume in. The extra arithmetic is one
redundant ``Hy`` per cell (the leader's recompute), 1.0x the cell count — not the
233.6x a per-thread whole-column recompute would cost.

=========================================================================
WHAT THE DRIVER RUNS INSIDE THIS SEAM
=========================================================================

``driver.py:3313`` consults ``update_H``; ``:3315-3316`` is the electric
integrated-source WITHDRAW; ``:3317`` consults ``step_D``. Nothing is INJECTED here (the
magnetic deposit is one seam earlier, the electric one one seam later), so
:data:`CARRIES_DEPOSIT_REPAIR` is False and ``deposit_repair`` is not consulted. The
withdraw IS in the seam, and this product does not perform it
(:data:`HOISTS_THE_WITHDRAW` False), so the predicate refuses BY NAME every row with a
standing electric withdraw — the two ``withdraw_seam`` rows the boards already file
(``cylinder_cross_section.py``, ``zone_plate.py``) — through
:func:`..withdraw_hoist.seam_withdraw_reasons`, the same clause :mod:`.fused_hd_pair`
spells.

=========================================================================
THE BINDING CEILING IS MEASURED, NOT COUNTED
=========================================================================

Launch 2 is the certified curl at its own :data:`.cylindrical_complex.CURL_BINDINGS`
(27). Launch 1 binds :data:`LEAD_BINDINGS` = 25 (24 pointers plus one packed
``Params``): six scratch, six pre-launch, three ``B``, the prefix scratch, six
constitutive coefficient vectors, the two row vectors. That count is read off the
emitted text (:func:`lead_signature_bindings`), and the gate COMPILES it on the
platform and BISECTS the ceiling around it rather than trusting the arithmetic — a
one-launch weld of this cell would be over the ceiling before it was written (the curl
alone is 27), which is the other reason the shape is two launches.

=========================================================================
COMPOSITION: WHAT THIS PRODUCT DOES NOT CLAIM
=========================================================================

:data:`INSTALLABLE` is False, for the reason and by the precedent :mod:`.fused_hd_pair`
gives. On every one of the sixteen rows the released ``cylindrical complex fused
electric D/E pair`` claims ``step_D``; the composer's launch arithmetic counts a tie
(three launches either way) and resolves it for the incumbent. What the composer's
arithmetic does NOT price is the host round trip the D->E pair still pays and this
product removes — that is a composition decision recorded in
:data:`INSTALLABLE_REASON` for the owner, not one this module takes. The natural end
state for these rows — one product owning ``update_H`` through ``update_E``, launch 1
as here and launch 2 the released D->E fused kernel reading the device prefix — is
named there too and is not built here.

REGISTERED UNWIRED on ``update_H`` (``wired=False``, ``replaces=REPLACES``), so the
seam loop can ask it and ``plan_step`` cannot select it; ``launch.FUSED_PAIR_ARMS``
carries its absorb row and ``registry.FAMILY_MODULES`` imports it.

NO TORCH AT MODULE LEVEL, as everywhere in this package.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .. import withdraw_hoist as _withdraw_hoist
from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from ..triton_kernels.launch import SUB_STEPS
from . import complex_fields, cylindrical_complex, cylindrical_complex_scan as scan
from . import offdiag_weld_common as _weld
from . import shaders, templates
from .cylindrical_real import scratch_mirror
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source

FAMILY = "cylindrical_complex_fused_hd_pair"

#: The slot this product would hold a row on: the FIRST half of the seam, in the
#: driver's own order.
SLOT = "update_H"

#: The two driver passes one ``run`` performs, in driver order (driver.py:3313, :3317).
#: The electric withdraw between them is NOT carried (:data:`HOISTS_THE_WITHDRAW`).
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.FUSED_PAIR_SEAMS`` files this row under.
SEAM: str = _withdraw_hoist.SEAM

#: Nothing is injected between the two consults; no deposit to carry.
CARRIES_DEPOSIT_REPAIR = False

#: This product does not perform the seam's electric withdraw; rows with a standing
#: withdraw are refused by name.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on the :mod:`.fused_hd_pair` precedent,
#: for the composition reason in :data:`INSTALLABLE_REASON`. The arithmetic is what the
#: gate certifies; the composition is a decision recorded for the owner.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "arbitration, not arithmetic. On every one of the sixteen cylindrical complex rows "
    "the released cylindrical complex fused electric D/E pair (CARRIES_DEPOSIT_REPAIR "
    "True; all sixteen rows declare an electric source inside the D->E seam) claims "
    "step_D, and launch.FUSED_PAIR_ARMS carries no row for the complex B->H pair, so "
    "over step_B..update_E the composer counts THREE launches whether the D->E pair or "
    "this product installs (step_B + update_H + [D/E fused] versus step_B + [this, two "
    "launches] + update_E) and _neighbouring_seam_claimant resolves the tie for the "
    "incumbent. What that count does not price: the D->E pair still pays the complex "
    "family's per-step host round trip for the radial prefix (cylindrical_complex.py:"
    "1468-1485, sync_out of Hy, the numpy scan, sync_in of the prefix), which this "
    "product removes, and a round trip was measured at 12x-64x the kernel it "
    "surrounds (device.py). Whether a removed round trip outranks a tied launch count "
    "is a composition decision for the owner, and the natural end state — one product "
    "owning update_H through update_E, launch 1 as here and launch 2 the released D->E "
    "fused kernel reading the device prefix — is not built here. Until that decision "
    "this product installs on zero rows, executes nowhere outside its own gate and "
    "tests, and licenses no timing claim and no dispatch claim")

#: WELDED. Empty means ``fingerprints.json`` carries
#: ``metal_cylindrical_complex_fused_hd_pair_device_gate`` (minted from the released
#: artifact by ``parity/meep_gpu/mint_metal_weld.py``); see
#: ``test_metal_weld_contract.test_every_metal_family_is_welded``.
WELD_OWED: str = ""

#: The rotating volumes, in SIGNATURE ORDER: the three stored magnetic fields then the
#: three split-field histories. :meth:`MetalCylindricalComplexFusedHdPairPlan.run_lead`
#: binds every SCRATCH first and every PRE-LAUNCH buffer second, in this order, then the
#: static arguments — so this order IS the signature's.
ROTATED_NAMES: Tuple[str, ...] = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")

#: Launch 1's bindings: 6 scratch + 6 pre-launch + 3 B + 1 prefix + 6 coefficient
#: vectors + 2 row vectors = 24 pointers, plus one ``Params``. Compared against the
#: emitted text by :func:`lead_signature_bindings` and against the platform by the gate.
LEAD_POINTERS = 24
LEAD_BINDINGS = 25
if LEAD_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the complex H->D lead launch binds {LEAD_BINDINGS} buffers and Metal's "
        f"ceiling on this toolchain is {MAX_BUFFER_BINDINGS}")

#: Launch 2's bindings are the certified curl's own.
CURL_BINDINGS = cylindrical_complex.CURL_BINDINGS

#: The kernel entry points.
LEAD_KERNEL = "cyl_complex_hd_lead_step"
CURL_KERNEL = "cyl_complex_pml_curl_step"

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_LIFT_EDITS", "CURL_BINDINGS", "CURL_KERNEL",
    "FAMILY", "HOISTS_THE_WITHDRAW", "H_CELL_POINTERS", "INSTALLABLE",
    "INSTALLABLE_REASON", "LEAD_BINDINGS", "LEAD_KERNEL", "LEAD_POINTERS", "REPLACES",
    "ROTATED_NAMES", "SEAM", "SLOT", "WELD_OWED",
    "MetalCylindricalComplexFusedHdPairPlan", "certified_complex_constitutive_tail",
    "compile_lead", "curl_source", "enumerate_sources", "h_cell_function",
    "lead_signature_bindings", "lead_source",
    "metal_cylindrical_complex_fused_hd_pair_coverage",
    "plan_metal_cylindrical_complex_fused_hd_pair", "refuted_lead_plus_source",
    "register_arms",
]


# ---------------------------------------------------------------------------
# The lift: the certified complex H constitutive as a function of an arbitrary cell
# ---------------------------------------------------------------------------

#: Where the certified complex constitutive body ends its index decode —
#: ``templates.DECODE_IJK``'s last line, which :data:`.offdiag_weld_common.DECODE_END`
#: names for every family that lifts a body.
DECODE_END = _weld.DECODE_END

#: ``h_cell``'s pointer parameters, under the CERTIFIED BODY'S OWN NAMES for the
#: coefficient vectors (``kp0``..``km2``, which the lifted text indexes) and under the
#: fused kernel's names for the three volume groups the lift renames.
H_CELL_POINTERS: Tuple[Tuple[str, str], ...] = tuple(
    [("device const float2*", name) for name in
     ("hi0", "hi1", "hi2", "wi0", "wi1", "wi2", "b0", "b1", "b2")]
    + [("device const float*", name) for name in
       ("kp0", "km0", "kp1", "km1", "kp2", "km2")])

#: What one call site forwards after the three coordinates.
H_CELL_TAIL_ARGS = ", ".join(("nxi", "nyi", "nzi")
                             + tuple(name for _kind, name in H_CELL_POINTERS))

#: Every line of certified constitutive text this weld does not lift verbatim, with the
#: reason. DATA, so the suite and the gate can assert the list. THERE IS NO ARITHMETIC
#: IN THIS TABLE: every entry is a pointer spelling or a store that moves to the caller.
CONSTITUTIVE_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    float2 prevN = wN[ii];",
     "became": "    float2 prevN = wiN[ii];",
     "why": "the split-field history is read from the PRE-LAUNCH buffer, which nothing "
            "in this launch writes; the certified in-place read would hand a racing "
            "column leader the value another thread had just stored"},
    {"line": "    float2 srcN = gN[ii];",
     "became": "    float2 srcN = bN[ii];",
     "why": "pointer spelling only: the flux density B under its own name, because "
            "`g` is the curl's magnetic source in the other launch"},
    {"line": "    wN[ii] = srcN;",
     "became": "(removed -- the caller stores own.srcN to the SCRATCH volume)",
     "why": "the weld writes scratch; the value and the expression that produced it "
            "are untouched, only the destination moves, and only for the thread's OWN "
            "cell (the leader's recompute stores nothing)"},
    {"line": "    float2 aN = fN[ii];",
     "became": "    float2 aN = hiN[ii];",
     "why": "the accumulator is read from the PRE-LAUNCH H, const for the dispatch: "
            "this is what makes the leader's recompute a pure function of unwritten "
            "memory"},
    {"line": "    fN[ii] = aN;",
     "became": "(removed -- the value is RETURNED in the struct)",
     "why": "the destination moves to the caller; the two accumulations, their order "
            "and their operands are untouched"},
    {"line": "    int k   = ii % nzi; ... int i   = plane / nyi;",
     "became": "(removed -- i, j and k are parameters; ii is composed from them)",
     "why": "the certified body becomes a function evaluated at an ARBITRARY cell, so "
            "the column leader can evaluate it at every row of its column with the "
            "SAME text the thread at that row evaluates for its own cell"},
)


def certified_complex_constitutive_tail(expansion: str,
                                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The complex ``update_H`` body BELOW the decode prologue, stores removed.

    :func:`.complex_fields.bloch_constitutive_source`'s own output for side ``H`` with
    the preamble, the decode prologue and the closing brace cut, and with exactly the
    substitutions :data:`CONSTITUTIVE_LIFT_EDITS` names, each applied through
    :func:`.offdiag_weld_common.needle` so a spelling that drifted raises here.
    """
    source = complex_fields.bloch_constitutive_source("H", expansion, contract)
    edits: List[Tuple[str, str]] = []
    for target in range(3):
        edits.extend((
            (f"    float2 prev{target} = w{target}[ii];\n",
             f"    float2 prev{target} = wi{target}[ii];\n"),
            (f"    float2 src{target} = g{target}[ii];\n",
             f"    float2 src{target} = b{target}[ii];\n"),
            (f"    w{target}[ii] = src{target};\n", ""),
            (f"    float2 a{target} = f{target}[ii];\n",
             f"    float2 a{target} = hi{target}[ii];\n"),
            (f"    f{target}[ii] = a{target};\n", ""),
        ))
    tail = _weld.lift_curl_tail(source, store_edits=tuple(edits), decode_end=DECODE_END)
    # In the lead kernel `f*`, `w*`, `g*` and `e*` do not exist; a surviving read
    # would be a compile error, but the check is kept so the reason is stated.
    for stem in ("f", "w", "g", "e"):
        for target in range(3):
            if f"{stem}{target}[" in tail:
                raise AssertionError(
                    f"the lifted complex update_H body still indexes {stem}{target}")
    return tail


def h_cell_function(expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """Complex ``update_H`` at an arbitrary cell, as one inline function.

    ONE code path serves the thread's own cell and the column leader's recompute at
    every row, so the two cannot disagree — the whole identity of this product rests
    on that sentence, and the gate measures it.
    """
    return _weld.step_cell_function(
        "h_cell", certified_complex_constitutive_tail(expansion, contract),
        value_type="float2", pointer_parameters=H_CELL_POINTERS, scalar_parameters=(),
        returns=("a0", "a1", "a2", "src0", "src1", "src2"))


# ---------------------------------------------------------------------------
# Launch 1: the lead kernel
# ---------------------------------------------------------------------------

_LEAD_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

__DIVIDE__

struct Params { uint nx; uint ny; uint nz; uint n_elem; };

__H_CELL__

kernel void cyl_complex_hd_lead_step(
    device float2*       ho0     [[buffer(0)]],
    device float2*       ho1     [[buffer(1)]],
    device float2*       ho2     [[buffer(2)]],
    device float2*       wo0     [[buffer(3)]],
    device float2*       wo1     [[buffer(4)]],
    device float2*       wo2     [[buffer(5)]],
    device const float2* hi0     [[buffer(6)]],
    device const float2* hi1     [[buffer(7)]],
    device const float2* hi2     [[buffer(8)]],
    device const float2* wi0     [[buffer(9)]],
    device const float2* wi1     [[buffer(10)]],
    device const float2* wi2     [[buffer(11)]],
    device const float2* b0      [[buffer(12)]],
    device const float2* b1      [[buffer(13)]],
    device const float2* b2      [[buffer(14)]],
    device float2*       out     [[buffer(15)]],
    device const float*  kp0     [[buffer(16)]],
    device const float*  km0     [[buffer(17)]],
    device const float*  kp1     [[buffer(18)]],
    device const float*  km1     [[buffer(19)]],
    device const float*  kp2     [[buffer(20)]],
    device const float*  km2     [[buffer(21)]],
    device const float*  weights [[buffer(22)]],
    device const float*  divisor [[buffer(23)]],
    constant Params&     prm     [[buffer(24)]],
    uint idx [[thread_position_in_grid]])
{
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
__GUARD__

__DECODE__
    int nxi = int(nx);
    int nyz = nyi * nzi;

    // --- THE WELD: update_H at the thread's OWN cell, stored to SCRATCH -----------
    // Nothing written in this launch is read by it: `own` is a function of the
    // pre-launch buffers alone, and the launcher rotates H/f_w_H afterwards.
    h_cell_result own = h_cell(i, j, k, __H_CELL_ARGS__);
    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;
    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;

    // --- THE COLUMN LEADER: the r = 0 thread scans its (phi, z) column -------------
    // stepping.py:418 — prefixed["Hy"] = cylindrical_rderiv_prefix(Hy, ir0 = 0.5) over
    // the POST-update_H Hy. The leader recomputes Hy at every row through the SAME
    // h_cell the thread at that row used for its own store, from the same pre-launch
    // bytes, so the value it scans IS the value that thread stored. Row 0 is its own.
    if (i == 0) {
        int base = j * nzi + k;
__LOOP__
    }
}
"""  # stepping.py live lines for the frozen device-text citation(s) in this string: 418->447


def lead_source(expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """Launch 1's source for one expansion arm and contraction mode.

    The column loop is :data:`.cylindrical_complex_scan.COLUMN_SCAN_LOOP` with the row
    sources substituted: row 0 is the thread's own ``own.a1`` and row ``i`` is the
    recompute ``h_cell(i, j, k, ...).a1`` — the D side scans ``Hy`` raw (no wall row),
    exactly as :func:`.cylindrical_complex_scan.source_reads` spells ``step_D``. The
    divide helper is the scan module's own, so the two kernels cannot spell the
    complex-by-real divide differently.
    """
    loop = templates.substitute(scan.COLUMN_SCAN_LOOP, {
        "__SRC0__": "own.a1",
        "__SRCI__": f"h_cell(i, j, k, {H_CELL_TAIL_ARGS}).a1",
    })
    loop = "\n".join("    " + line if line.strip() else line
                     for line in loop.splitlines())
    return templates.substitute(_LEAD_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__DIVIDE__": scan._DIVIDE_HELPER,
        "__H_CELL__": h_cell_function(expansion, contract),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__H_CELL_ARGS__": H_CELL_TAIL_ARGS,
        "__LOOP__": loop,
    })


def compile_lead(expansion: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    return getattr(compile_source(lead_source(expansion, contract)), LEAD_KERNEL)


def curl_source(bcz: int, m_arm: int, expansion: str,
                contract: str = shaders.CONTRACT_OFF) -> str:
    """Launch 2's source: the CERTIFIED complex ``step_D`` curl, character for character."""
    return cylindrical_complex.cylindrical_curl_source(bcz, True, m_arm, expansion,
                                                       contract)


def compile_curl(bcz: int, m_arm: int, expansion: str,
                 contract: str = shaders.CONTRACT_OFF) -> Any:
    return cylindrical_complex.compile_cylindrical_curl(bcz, True, m_arm, expansion,
                                                        contract)


def lead_signature_bindings(expansion: str = "FMA_V1") -> int:
    """How many buffers the emitted lead signature binds — read from the TEXT."""
    source = lead_source(expansion)
    signature = source.split("kernel void", 1)[1].split("{", 1)[0]
    return signature.count("[[buffer(")


def refuted_lead_plus_source(extra: int, expansion: str = "FMA_V1",
                             contract: str = shaders.CONTRACT_OFF) -> str:
    """The lead signature with ``extra`` more pointers, for the gate's ceiling bisect.

    The extra pointers are declared and TOUCHED (a body the compiler could drop would
    let dead-code elimination decide where the ceiling is); the arithmetic is the
    shipped kernel's. ``extra = MAX_BUFFER_BINDINGS - LEAD_BINDINGS`` must compile and
    one more must fail, which is what turns :data:`LEAD_BINDINGS` and its headroom from
    a count into a measurement.
    """
    source = lead_source(expansion, contract)
    anchor = "    constant Params&     prm     [[buffer(24)]],\n"
    if anchor not in source:
        raise AssertionError("the lead signature no longer carries its Params anchor")
    pointers = "".join(f"    device const float*  x{n:<6}[[buffer({LEAD_BINDINGS + n})]],\n"
                       for n in range(extra))
    source = source.replace(
        anchor, "    constant Params&     prm     [[buffer(24)]],\n" + pointers)
    if extra:
        touch = " + ".join(f"x{n}[0]" for n in range(extra))
        body_anchor = "    ho0[ii] = own.a0;"
        if body_anchor not in source:
            raise AssertionError("the lead body no longer carries its store anchor")
        source = source.replace(
            body_anchor,
            f"    float touch = {touch};\n"
            f"    ho0[ii] = own.a0 + float2(touch * 0.0f, 0.0f);", 1)
    return source


def enumerate_sources(expansion: str,
                      contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a plan can emit: ONE lead per arm, the six curls."""
    out = {f"{LEAD_KERNEL}/{expansion}": lead_source(expansion, contract)}
    for bcz in (templates.PERIODIC, templates.METALLIC):
        for m_arm in cylindrical_complex.M_ARMS:
            label = (f"{CURL_KERNEL}/step_D/bcz{bcz}/"
                     f"{cylindrical_complex.M_ARM_LABELS[m_arm]}/{expansion}")
            out[label] = curl_source(bcz, m_arm, expansion, contract)
    return out


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_cylindrical_complex_fused_hd_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May two launches span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    A conjunction of the two halves' OWN certified predicates —
    :func:`.cylindrical_complex.cylindrical_complex_constitutive_coverage` on ``H`` and
    :func:`.cylindrical_complex.cylindrical_complex_pml_curl_coverage` on ``step_D`` —
    in the driver's order, plus the seam clause. Nothing is weakened.
    """
    reasons: List[str] = []
    constitutive = cylindrical_complex.cylindrical_complex_constitutive_coverage(
        fields, pml, "H", residency, probe)
    if not constitutive.covered:
        reasons.extend(f"complex constitutive half: {reason}"
                       for reason in constitutive.reasons)
    curl = cylindrical_complex.cylindrical_complex_pml_curl_coverage(
        fields, pml, "step_D", residency, probe)
    if not curl.covered:
        reasons.extend(f"complex cylindrical curl half: {reason}"
                       for reason in curl.reasons)

    # THE SEAM CLAUSE. The only pass between the two consults is the electric
    # integrated-source withdraw (driver.py:3315-3316). IGNORANCE IS NEVER AN EMPTY
    # SET: `Fields` does not hold the source list.
    reasons.extend(_withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) holds a standing "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3315-3316); this product declares "
            f"HOISTS_THE_WITHDRAW = False and would launch against a D still "
            f"holding the previous step's dipole"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))
    shape = tuple(getattr(grid, "shape", ()))
    if len(shape) != 3 or int(shape[1]) != 1:
        reasons.append(f"a Dcyl grid stores one phi cell; shape {shape} does not")
    if residency is None:
        reasons.append("no residency was declared: this product rotates H/f_w_H "
                       "between plan-owned twins and needs the shared registry")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalCylindricalComplexFusedHdPairPlan:
    """Launch 1 (constitutive + column scan, then the H/f_w_H rotation), launch 2 (the
    certified curl against the rotated H and the device prefix).

    A SIBLING of :class:`.plans.KernelPlan`, not a subclass, for the reason the other
    two-launch plans are: the base freezes ONE argument tuple and calls ONE function,
    and here the magnetic arguments ALTERNATE between two physical buffers and the two
    launches are ordered. :attr:`lead_launches` and :attr:`curl_launches` are carried
    apart so "the lead ran and the curl did not" is distinguishable from "ran once".

    THE ORDER IS THE CONTRACT. The curl reads the prefix and the H the lead wrote; a
    curl run first, or a lead hoisted out of the loop, feeds the curl the PREVIOUS
    step's magnetic field — a smooth, plausible, entirely wrong field.
    """

    __slots__ = ("residency", "fields", "shape", "n_elem", "rotated", "volumes",
                 "prefix_host", "static_lead_args", "curl_static", "curl_tail",
                 "launches", "lead_launches", "curl_launches", "runs",
                 "_lead_functions", "_curl_functions", "m_arm", "bcz", "expansion")

    family = "cylindrical complex fused H/D pair"
    replaces_sub_steps = REPLACES
    performs_device_work = True
    launches_per_run = 2
    repair_paths: Tuple[str, ...] = ()
    REPR_FIELDS = ("shape", "bcz", "m_arm", "expansion")

    def __init__(self, residency: Residency, fields: Any, shape: Sequence[int],
                 rotated: Mapping[str, Any], prefix_host: Any,
                 static_lead_args: Sequence[Any], curl_static: Sequence[Any],
                 curl_tail: Sequence[Any], lead_functions: Mapping[str, Any],
                 curl_functions: Mapping[str, Any], volumes: Sequence[str],
                 bcz: int, m_arm: int, expansion: str) -> None:
        self.residency = residency
        self.fields = fields
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or self.shape[1] != 1:
            raise ValueError(f"a Dcyl grid stores one phi cell; shape {self.shape} does not")
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.rotated = dict(rotated)
        if set(self.rotated) != set(ROTATED_NAMES):
            raise ValueError(f"the twin table {sorted(self.rotated)} does not name "
                             f"{ROTATED_NAMES}")
        self.volumes = tuple(dict.fromkeys(volumes))
        self.prefix_host = prefix_host
        self.static_lead_args = tuple(static_lead_args)
        self.curl_static = tuple(curl_static)
        self.curl_tail = tuple(curl_tail)
        self._lead_functions = dict(lead_functions)
        self._curl_functions = dict(curl_functions)
        self.bcz = int(bcz)
        self.m_arm = int(m_arm)
        self.expansion = str(expansion)
        self.launches = 0
        self.lead_launches = 0
        self.curl_launches = 0
        self.runs = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self._lead_functions) & set(self._curl_functions)))

    def _function(self, table: Mapping[str, Any], mode: str, which: str) -> Any:
        function = table.get(mode)
        if function is None:
            raise KeyError(f"this plan holds no {mode!r} {which} variant (it was built "
                           f"with {self.variants})")
        return function

    def _resolve(self) -> Tuple[List[Any], List[Any]]:
        """(scratch write tensors, pre-launch read tensors), by CURRENT role."""
        writes: List[Any] = []
        reads: List[Any] = []
        for name in ROTATED_NAMES:
            current = getattr(self.fields, name)
            read = self.residency.tensor_for_host(current)
            write = self.residency.tensor_for_host(self.rotated[name])
            if read is None or write is None:
                raise RuntimeError(f"the {FAMILY} weld lost the device mirror of {name}")
            if read is write:
                raise RuntimeError(f"the {FAMILY} weld resolved {name} and its scratch "
                                   f"to ONE tensor; nothing written may be read")
            reads.append(read)
            writes.append(write)
        return writes, reads

    def run_lead(self, contract: Optional[str] = None) -> None:
        """Launch 1, then the rotation. Never run twice without a curl between."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._function(self._lead_functions, mode, "lead")
        writes, reads = self._resolve()
        self.launches += 1
        self.lead_launches += 1
        function(*writes, *reads, *self.static_lead_args)
        # THE ROTATION — only after the launch returned. The engine's references move
        # to the freshly written buffers; the pre-launch buffers become next launch's
        # scratch.
        for name in ROTATED_NAMES:
            current = getattr(self.fields, name)
            setattr(self.fields, name, self.rotated[name])
            self.rotated[name] = current

    def run_curl(self, contract: Optional[str] = None) -> None:
        """Launch 2: the certified curl against the ROTATED H and the device prefix."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._function(self._curl_functions, mode, "curl")
        sources = []
        for name in SUB_STEPS["step_D"]["sources"]:
            tensor = self.residency.tensor_for_host(getattr(self.fields, name))
            if tensor is None:
                raise RuntimeError(f"the {FAMILY} weld lost the device mirror of {name}")
            sources.append(tensor)
        self.launches += 1
        self.curl_launches += 1
        function(*self.curl_static, *sources, *self.curl_tail)

    def run(self, contract: Optional[str] = None) -> None:
        self.runs += 1
        self.run_lead(contract)
        self.run_curl(contract)

    def describe(self) -> str:
        parts = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.REPR_FIELDS)
        return f"MetalCylindricalComplexFusedHdPairPlan({parts}, variants={self.variants})"

    def __repr__(self) -> str:
        return self.describe()


def _params_tensor(shape: Sequence[int], device: str) -> Any:
    """The four extents as one 16-byte device record, plan-owned."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=np.dtype([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"),
                                         ("n_elem", "<u4")]))
    nx, ny, nz = (int(n) for n in shape)
    record[0] = (nx, ny, nz, nx * ny * nz)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_cylindrical_complex_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        lead_functions: Optional[Mapping[str, Any]] = None,
        curl_functions: Optional[Mapping[str, Any]] = None,
        probe: Any = None) -> Optional[MetalCylindricalComplexFusedHdPairPlan]:
    """Build the two-launch complex H/D plan, or ``None`` when the seam is refused.

    ``lead_functions`` and ``curl_functions`` are the KERNEL mutation seams, SEPARATE so
    a gate can plant a defect in one launch while the other stays shipped. Dropping
    either is a silent DISARMING.
    """
    if not metal_cylindrical_complex_fused_hd_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    expansion = cylindrical_complex.expansion_from_probe(
        probe if probe is not None else cylindrical_complex.load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    import numpy  # noqa: PLC0415

    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    shape = tuple(int(n) for n in grid.shape)
    kinds = _boundary_kinds(grid, pml)
    bcz = templates.METALLIC if kinds[2] == "metallic" else templates.PERIODIC
    m = int(grid.m)
    m_arm = cylindrical_complex.m_class(m)
    dtdx = grid.dt / grid.dx
    spec = SUB_STEPS["step_D"]
    side = CONSTITUTIVE_SIDES["H"]
    volumes: List[str] = []
    mirror = complex_fields._complex_mirror

    def bind_complex(name: str, host: Any) -> Any:
        volumes.append(name)
        return mirror(residency, name, host)

    def bind_real(name: str, host: Any) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=True)

    # THE ROTATING GROUP: every stored magnetic field and split-field history is
    # mirrored under the engine's name AND given a complex64 twin.
    twins: Dict[str, Any] = {}
    for name in ROTATED_NAMES:
        host = getattr(fields, name)
        bind_complex(name, host)
        twin, _mirror = _weld.scratch_twin(residency, name, host, dtype=numpy.complex64)
        twins[name] = twin

    flux = [bind_complex(name, getattr(fields, name)) for name in side["sources"]]

    # THE PREFIX SCRATCH: (nr, ny, nz) complex64, fully overwritten by every lead launch
    # before the curl reads it, shared between plans built against one residency.
    # NOT through `scratch_mirror`: that helper binds float32 by default, which on a
    # complex64 array is a silent cast that discards the imaginary plane (the
    # `scratch_twin` docstring records the same trap). The reuse rule is the same —
    # a second plan built against one residency shares the array — spelled here for
    # the complex dtype.
    prefix_name = "cylhd:pfx:step_D"
    prefix_host = residency.host(prefix_name)
    if prefix_host is None:
        prefix_host = numpy.zeros(cylindrical_complex.prefix_shape("step_D", shape),
                                  dtype=numpy.complex64)
    prefix = mirror(residency, prefix_name, prefix_host)
    volumes.append(prefix_name)

    # THE CONSTITUTIVE COEFFICIENTS on H's sub-lattice (stepping.py:948): kps/kms per
    # axis, integer lattice, in the certified plan's own order.
    suffix = "_h" if side["half_integer"] else ""
    coefficients = [bind_real(f"pml:{stem}_{axis}{suffix}",
                              getattr(pml, f"{stem}_{axis}{suffix}"))
                    for axis in "xyz" for stem in ("kps", "kms")]

    # THE TWO ROW VECTORS, from stepping's own builder, grid invariants.
    rows = shape[0]
    weights, divisor = scan.prefix_row_vectors("step_D", rows)
    weights_host, weights_t = scratch_mirror(
        residency, "cylhd:weights:step_D", lambda: weights, constant=True)
    divisor_host, divisor_t = scratch_mirror(
        residency, "cylhd:divisor:step_D", lambda: divisor, constant=True)
    volumes.extend(("cylhd:weights:step_D", "cylhd:divisor:step_D"))

    static_lead = (flux + [prefix] + coefficients + [weights_t, divisor_t]
                   + [_params_tensor(shape, residency.device)])
    if 2 * len(ROTATED_NAMES) + len(static_lead) != LEAD_BINDINGS:  # pragma: no cover
        raise AssertionError((2 * len(ROTATED_NAMES) + len(static_lead), LEAD_BINDINGS))

    # LAUNCH 2's arguments, assembled EXACTLY as `cylindrical_complex._build_curl_plan`
    # assembles the certified plan's, with the magnetic sources resolved at run time
    # (they rotate) and the prefix being the device scratch.
    targets = [bind_complex(name, getattr(fields, name)) for name in spec["targets"]]
    auxiliaries = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                   for name in spec["targets"]]
    imr_rows = []
    for index, _register, sign in cylindrical_complex.IMR_TERMS["step_D"]:
        target = spec["targets"][index]
        row = numpy.ascontiguousarray(cylindrical_complex.imr_coefficient_row(
            numpy, target, sign, m, dtdx, rows, numpy.complex64).reshape(-1))
        imr_rows.append(bind_complex(f"cylhd:imr:step_D:{target}", row))
    curl_coefficients = [bind_real(f"pml:{stem}_{axis}{spec['suffix']}",
                                   getattr(pml, f"{stem}_{axis}{spec['suffix']}"))
                         for axis in "xyz" for stem in ("kms", "sinv")]
    minus_dtdx, inc_b = cylindrical_complex.axis_increment_scalars(m, dtdx)
    zero_rows = cylindrical_complex.zero_rows(m, bool(grid.accurate_fields_near_cylorigin))
    axis_coef = cylindrical_complex.axis_coefficient(dtdx)
    curl_static = tuple(targets) + tuple(auxiliaries)
    curl_tail = ((prefix,) + tuple(imr_rows) + tuple(curl_coefficients)
                 + (shape[0], shape[1], shape[2], shape[0] * shape[1] * shape[2],
                    float(dtdx), int(zero_rows), float(minus_dtdx),
                    (float(inc_b[0]), float(inc_b[1])), float(axis_coef)))

    leads = dict(lead_functions or {})
    curls = dict(curl_functions or {})
    for mode in contract_variants:
        leads.setdefault(mode, compile_lead(expansion, mode))
        curls.setdefault(mode, compile_curl(bcz, m_arm, expansion, mode))
    return MetalCylindricalComplexFusedHdPairPlan(
        residency, fields, shape, twins, prefix_host, static_lead, curl_static,
        curl_tail, leads, curls, volumes, bcz, m_arm, expansion)


# ---------------------------------------------------------------------------
# Registration -- wired=False, refused by the composer on INSTALLABLE
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"cylindrical complex fused H/D pair cannot fill {slot}",))
    return metal_cylindrical_complex_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency,
        context.extra.get("cylindrical_complex_probe"))


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalCylindricalComplexFusedHdPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_cylindrical_complex_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, probe=context.extra.get("cylindrical_complex_probe"))


def register_arms() -> Tuple[Any, ...]:
    """One row on ``update_H``, ``wired=False``, gated on a complex Dcyl run.

    Registered so it is ENUMERABLE (the disjointness sweep and
    ``launch._neighbouring_seam_claimant`` read ``arms.registered``) and so the seam
    loop can ask it; ``arms_for`` skips an unwired row so ``plan_step`` cannot select
    it, and the loop refuses it on :data:`INSTALLABLE` before its predicate is asked.
    Gated exactly as the complex family gates its four arms, so a Cartesian refusal
    message gains no reasons from this row.
    """
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "cylindrical complex fused H/D pair",
                          _arm_coverage, _arm_plan,
                          prefix="cylindrical complex fused H/D pair: ",
                          noun="cylindrical complex fused stored-H/PML D-curl pair",
                          gate=cylindrical_complex._is_cylindrical_complex,
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
