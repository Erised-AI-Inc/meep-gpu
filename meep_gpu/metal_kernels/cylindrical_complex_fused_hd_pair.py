"""The Dcyl COMPLEX H->D seam in TWO launches: ``update_H`` folded with the device
radial scan, then the certified complex curl — no host round trip.

THE CELL. Sixteen ``cylindrical complex`` corpus rows (``h_to_d_seam.instances``,
identical on the Metal, Triton and CUDA boards), every one ``pml_active`` with ``ny = 1``,
radial extents 80 to 1050 rows and 1 to 497 columns. Today each of them pays, on every
``step_D``, a device->host->device round trip so that ``numpy`` can form the radial
difference (:meth:`.cylindrical_complex.CylindricalComplexCurlPlan.refresh_prefix`).
This product removes the round trip and owns both seam slots.

=========================================================================
THE SHAPE, AND WHY IT IS TWO LAUNCHES AND NOT ONE
=========================================================================

``stepping.step_D`` on a Dcyl grid takes the ``Dz`` term's radial derivative as the
backward difference of MEEP's ``f_rderiv_int`` (``stepping.cylindrical_rderiv_difference``):
a sum down each radial column of the POST-``update_H`` ``Hy``, restarted at +0 on every
structure chunk's first row, with ``rinv`` multiplied. The sum is strictly serial within
a chunk — a non-local dependence that a single pointwise dispatch cannot satisfy, since
the difference of two rounded partial sums is not the rounded increment. The recompute
unit is a whole radial column, so:

* **LAUNCH 1** (``cyl_complex_hd_lead_step``, the ``update_H`` slot): every thread
  computes its OWN cell's ``H`` and split-field ``f_w_H`` through ``h_cell`` — the
  certified complex ``bloch_constitutive_step`` body lifted as a function of an
  arbitrary cell, reading ONLY pre-launch buffers — and stores them to WRITE-ONLY
  scratch. The ``r = 0`` thread of each (phi, z) column additionally walks its column,
  recomputing ``Hy`` at every row through the SAME ``h_cell`` and writing the D-side
  radial difference (``ir0 = 0.5``, no wall row, per-chunk restart) into a scratch —
  :func:`.cylindrical_complex_scan.column_scan_loop` for ``step_D`` with the row source
  being the recompute rather than a load. NOTHING WRITTEN BY THIS LAUNCH IS READ BY IT, so no
  thread observes another thread's store; the plan rotates ``H``/``f_w_H`` to the
  scratch afterwards, exactly as :class:`.offdiag_weld_common.ScratchWeldPairPlan` does
  for the D->E welds and :mod:`.fused_hd_pair` for the Cartesian H->D one.
* **LAUNCH 2** (the ``step_D`` slot): the CERTIFIED complex cylindrical curl,
  :func:`.cylindrical_complex.cylindrical_curl_source` UNCHANGED, bound to the rotated
  ``H`` and to the radial scratch launch 1 wrote. No new pointer, no new arithmetic.

Bit identity is the composition of two pieces each measured on its own: the pointwise
constitutive recompute (``h_cell``, the complex H->D weld's lift of the certified body,
imported from :mod:`.complex_fused_hd_pair`) and the column-serial complex radial sum
(:mod:`.cylindrical_complex_scan`, against ``stepping.cylindrical_rderiv_difference``
itself). What is NEW to measure is the seam between them — that the leader's recomputed
``Hy`` at row r is bit-identical to what the thread at row r stored — and the gate
measures it as a complete driver step.

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
(27). Launch 1 binds :data:`LEAD_BINDINGS` = 26 (25 pointers plus one packed
``Params`` that carries the chunk record): six scratch, six pre-launch, three ``B``, the
radial scratch, six constitutive coefficient vectors, the two row vectors and the chunk
starts. That count is read off the
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
as here and launch 2 the released D->E fused kernel reading the device difference — is
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
from . import chunk_bounds as _chunk_bounds
from .complex_fused_hd_pair import H_CELL_POINTERS, H_CELL_TAIL_ARGS, h_cell_function
from .cylindrical_real import int32_tensor, scratch_mirror, scan_starts
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
    "family's per-step host round trip for the radial sum (cylindrical_complex."
    "CylindricalComplexCurlPlan.refresh_prefix: sync_out of Hy, the numpy sum, sync_in "
    "of the radial difference), which this "
    "product removes, and a round trip was measured at 12x-64x the kernel it "
    "surrounds (device.py). Whether a removed round trip outranks a tied launch count "
    "is a composition decision for the owner, and the natural end state — one product "
    "owning update_H through update_E, launch 1 as here and launch 2 the released D->E "
    "fused kernel reading the device radial difference — is not built here. Until "
    "that decision this product installs on zero rows, executes nowhere outside its "
    "own gate and tests, and licenses no timing claim and no dispatch claim")

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

#: Launch 1's bindings: 6 scratch + 6 pre-launch + 3 B + 1 radial difference + 6
#: coefficient vectors + 2 row vectors + the chunk starts = 25 pointers, plus one
#: ``Params``. Compared against the emitted text by :func:`lead_signature_bindings`
#: and against the platform by the gate.
LEAD_POINTERS = 25
LEAD_BINDINGS = 26
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
    "CARRIES_DEPOSIT_REPAIR", "CURL_BINDINGS", "CURL_KERNEL",
    "FAMILY", "HOISTS_THE_WITHDRAW", "H_CELL_POINTERS", "INSTALLABLE",
    "INSTALLABLE_REASON", "LEAD_BINDINGS", "LEAD_KERNEL", "LEAD_POINTERS", "REPLACES",
    "ROTATED_NAMES", "SCAN_LIFT_EDITS", "SEAM", "SLOT", "WELD_OWED",
    "MetalCylindricalComplexFusedHdPairPlan",
    "compile_lead", "curl_source", "enumerate_sources", "h_cell_function",
    "lead_signature_bindings", "lead_source",
    "metal_cylindrical_complex_fused_hd_pair_coverage",
    "plan_metal_cylindrical_complex_fused_hd_pair", "refuted_lead_plus_source",
    "register_arms",
]


# ---------------------------------------------------------------------------
# Launch 1: the lead kernel
# ---------------------------------------------------------------------------
#
# ``h_cell`` is the complex H->D weld's (:func:`.complex_fused_hd_pair.h_cell_function`):
# the CERTIFIED complex ``update_H`` body as a function of an arbitrary cell over the
# pre-launch buffers, with the chunk record as its last parameter so the dsigw form is
# the recomputed cell's. IMPORTED, not re-lifted: one lift of that body in the package.

#: The two source reads of the certified complex D-side scan loop this product does
#: not lift verbatim, with what each became. DATA, so the suite and the gate can assert
#: the list. THERE IS NO ARITHMETIC IN THIS TABLE: each replaces a memory read with the
#: recompute of the same value.
SCAN_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": scan.source_reads("step_D")[0],
     "became": "own.a1",
     "why": "row 0 of the column's Hy is the leader's own cell, which it has just "
            "computed into `own` from the pre-launch state"},
    {"line": scan.source_reads("step_D")[1],
     "became": f"h_cell(g, j, k, {H_CELL_TAIL_ARGS}).a1",
     "why": "row g of the column's Hy, RECOMPUTED from the pre-launch state through "
            "the same h_cell the thread at that row stores its own cell from, instead "
            "of read from a volume this launch writes"},
)

_LEAD_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

__RECORD_STRUCT__

struct Params { uint nx; uint ny; uint nz; uint n_elem;
__RECORD_MEMBER__ };

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
    device const float*  kp1     [[buffer(17)]],
    device const float*  kp2     [[buffer(18)]],
    device const float*  kmx     [[buffer(19)]],
    device const float*  kmy     [[buffer(20)]],
    device const float*  kmz     [[buffer(21)]],
    device const float*  weights [[buffer(22)]],
    device const float*  rinv    [[buffer(23)]],
    device const int*    starts  [[buffer(24)]],
    constant Params&     prm     [[buffer(25)]],
    uint idx [[thread_position_in_grid]])
{
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    // THE CHUNK RECORD rides in the struct; `h_cell` evaluates membership from it at
    // the cell it steps.
__RECORD_ALIAS__
__GUARD__

__DECODE__
    int nxi = int(nx);
    int nyz = nyi * nzi;

    // --- THE WELD: update_H at the thread's OWN cell, stored to SCRATCH -----------
    // Nothing written in this launch is read by it: `own` is a function of the
    // pre-launch buffers alone, and the launcher rotates H/f_w_H afterwards.
    h_cell_result own = h_cell(i, j, k, __H_CELL_ARGS__);
    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;
    wo0[ii] = own.wn0; wo1[ii] = own.wn1; wo2[ii] = own.wn2;

    // --- THE COLUMN LEADER: the r = 0 thread walks its (phi, z) column -------------
    // stepping.step_D: Dz's radial difference of the per-chunk sum over the
    // POST-update_H Hy (ir0 = 0.5, backward, no wall row). The leader recomputes Hy
    // at every row through the SAME h_cell the thread at that row used for its own
    // store, from the same pre-launch bytes, so the value it sums IS the value that
    // thread stored. Row 0 is its own.
    if (i == 0) {
        int base = j * nzi + k;
__LOOP__
    }
}
"""


def lead_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """Launch 1's source for one contraction mode.

    The column loop is :func:`.cylindrical_complex_scan.column_scan_loop` for
    ``step_D`` with exactly the two source reads :data:`SCAN_LIFT_EDITS` names
    replaced, each required to land once.
    """
    loop = scan.column_scan_loop("step_D")
    for edit in SCAN_LIFT_EDITS:
        if loop.count(edit["line"]) != 1:
            raise AssertionError(
                f"the certified complex scan loop reads {edit['line']!r} "
                f"{loop.count(edit['line'])} times, not once; the lead cannot lift it")
        loop = loop.replace(edit["line"], edit["became"])
    if "src[" in loop:
        raise AssertionError("the lifted scan loop still reads `src`, which this kernel "
                             "does not bind")
    loop = "\n".join("    " + line if line.strip() else line
                     for line in loop.splitlines())
    return templates.substitute(_LEAD_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(),
        "__RECORD_STRUCT__": _chunk_bounds.struct_source(),
        "__RECORD_MEMBER__": _chunk_bounds.params_member(),
        "__RECORD_ALIAS__": _chunk_bounds.params_alias(),
        "__H_CELL__": h_cell_function(contract),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__H_CELL_ARGS__": H_CELL_TAIL_ARGS,
        "__LOOP__": loop,
    })


def compile_lead(contract: str = shaders.CONTRACT_OFF) -> Any:
    return getattr(compile_source(lead_source(contract)), LEAD_KERNEL)


def curl_source(bcz: int, m_arm: int,
                contract: str = shaders.CONTRACT_OFF) -> str:
    """Launch 2's source: the CERTIFIED complex ``step_D`` curl, character for character."""
    return cylindrical_complex.cylindrical_curl_source(bcz, True, m_arm,
                                                       contract)


def compile_curl(bcz: int, m_arm: int,
                 contract: str = shaders.CONTRACT_OFF) -> Any:
    return cylindrical_complex.compile_cylindrical_curl(bcz, True, m_arm,
                                                        contract)


def lead_signature_bindings() -> int:
    """How many buffers the emitted lead signature binds — read from the TEXT."""
    source = lead_source()
    signature = source.split("kernel void", 1)[1].split("{", 1)[0]
    return signature.count("[[buffer(")


def refuted_lead_plus_source(extra: int,
                             contract: str = shaders.CONTRACT_OFF) -> str:
    """The lead signature with ``extra`` more pointers, for the gate's ceiling bisect.

    The extra pointers are declared and TOUCHED (a body the compiler could drop would
    let dead-code elimination decide where the ceiling is); the arithmetic is the
    shipped kernel's. ``extra = MAX_BUFFER_BINDINGS - LEAD_BINDINGS`` must compile and
    one more must fail, which is what turns :data:`LEAD_BINDINGS` and its headroom from
    a count into a measurement.
    """
    source = lead_source(contract)
    anchor = f"    constant Params&     prm     [[buffer({LEAD_POINTERS})]],\n"
    if anchor not in source:
        raise AssertionError("the lead signature no longer carries its Params anchor")
    pointers = "".join(f"    device const float*  x{n:<6}[[buffer({LEAD_BINDINGS + n})]],\n"
                       for n in range(extra))
    source = source.replace(anchor, anchor + pointers)
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


def enumerate_sources(contract: str = shaders.CONTRACT_OFF) -> Dict[str, str]:
    """Every specialisation a plan can emit: ONE lead, the six curls."""
    out = {f"{LEAD_KERNEL}": lead_source(contract)}
    for bcz in (templates.PERIODIC, templates.METALLIC):
        for m_arm in cylindrical_complex.M_ARMS:
            label = (f"{CURL_KERNEL}/step_D/bcz{bcz}/"
                     f"{cylindrical_complex.M_ARM_LABELS[m_arm]}")
            out[label] = curl_source(bcz, m_arm, contract)
    return out


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_cylindrical_complex_fused_hd_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None) -> Coverage:
    """May two launches span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    A conjunction of the two halves' OWN certified predicates —
    :func:`.cylindrical_complex.cylindrical_complex_constitutive_coverage` on ``H`` and
    :func:`.cylindrical_complex.cylindrical_complex_pml_curl_coverage` on ``step_D`` —
    in the driver's order, plus the seam clause. Nothing is weakened.
    """
    reasons: List[str] = []
    constitutive = cylindrical_complex.cylindrical_complex_constitutive_coverage(
        fields, pml, "H", residency)
    if not constitutive.covered:
        reasons.extend(f"complex constitutive half: {reason}"
                       for reason in constitutive.reasons)
    curl = cylindrical_complex.cylindrical_complex_pml_curl_coverage(
        fields, pml, "step_D", residency)
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

    # THE RUN'S FIRST update_H. The array path starts H and f_w from B on the dsigw
    # cells at that update (stepping.separate_h_from_b) and h_cell does not, so this
    # product takes that update only where the start cannot change a word: the base
    # single's own clause (launch.h_start_reasons), in its own words.
    from .launch import h_start_reasons  # noqa: PLC0415 - launch imports this family's peers
    reasons.extend(f"complex constitutive half: {reason}"
                   for reason in h_start_reasons(fields, sources))

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
    certified curl against the rotated H and the device radial difference).

    A SIBLING of :class:`.plans.KernelPlan`, not a subclass, for the reason the other
    two-launch plans are: the base freezes ONE argument tuple and calls ONE function,
    and here the magnetic arguments ALTERNATE between two physical buffers and the two
    launches are ordered. :attr:`lead_launches` and :attr:`curl_launches` are carried
    apart so "the lead ran and the curl did not" is distinguishable from "ran once".

    THE ORDER IS THE CONTRACT. The curl reads the radial difference and the H the lead
    wrote; a
    curl run first, or a lead hoisted out of the loop, feeds the curl the PREVIOUS
    step's magnetic field — a smooth, plausible, entirely wrong field.
    """

    __slots__ = ("residency", "fields", "shape", "n_elem", "rotated", "volumes",
                 "radial_host", "starts", "chunk_bounds", "static_lead_args",
                 "curl_static", "curl_tail",
                 "launches", "lead_launches", "curl_launches", "runs",
                 "_lead_functions", "_curl_functions", "m_arm", "bcz")

    family = "cylindrical complex fused H/D pair"
    replaces_sub_steps = REPLACES
    performs_device_work = True
    launches_per_run = 2
    repair_paths: Tuple[str, ...] = ()
    REPR_FIELDS = ("shape", "bcz", "m_arm")

    def __init__(self, residency: Residency, fields: Any, shape: Sequence[int],
                 rotated: Mapping[str, Any], radial_host: Any, starts: Any,
                 chunk_bounds: Any,
                 static_lead_args: Sequence[Any], curl_static: Sequence[Any],
                 curl_tail: Sequence[Any], lead_functions: Mapping[str, Any],
                 curl_functions: Mapping[str, Any], volumes: Sequence[str],
                 bcz: int, m_arm: int) -> None:
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
        self.radial_host = radial_host
        self.starts = starts
        self.chunk_bounds = chunk_bounds
        self.static_lead_args = tuple(static_lead_args)
        self.curl_static = tuple(curl_static)
        self.curl_tail = tuple(curl_tail)
        self._lead_functions = dict(lead_functions)
        self._curl_functions = dict(curl_functions)
        self.bcz = int(bcz)
        self.m_arm = int(m_arm)
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
        """Launch 2: the certified curl against the ROTATED H and the device difference."""
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


def params_record_dtype() -> Any:
    """The host dtype of the lead's ``Params``: the four extents, then the chunk record."""
    import numpy as np  # noqa: PLC0415

    return np.dtype([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"), ("n_elem", "<u4")]
                    + _chunk_bounds.record_dtype_fields())


def _params_tensor(shape: Sequence[int], device: str, *, chunk_bounds: Any) -> Any:
    """The four extents and the chunk record as one device record, plan-owned."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    for name, value in (("nx", nx), ("ny", ny), ("nz", nz), ("n_elem", nx * ny * nz)):
        record[name] = value
    record["cb"] = _chunk_bounds.pack(chunk_bounds)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def plan_metal_cylindrical_complex_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        lead_functions: Optional[Mapping[str, Any]] = None,
        curl_functions: Optional[Mapping[str, Any]] = None,
        *, chunk_bounds: Any,
        ) -> Optional[MetalCylindricalComplexFusedHdPairPlan]:
    """Build the two-launch complex H/D plan, or ``None`` when the seam is refused.

    ``chunk_bounds`` is the record of the division the run steps with (the composer's
    ``StepContext.chunk_bounds``) and has no default; the radial sum's chunk starts are
    the array path's own (``stepping._cylindrical_scan_starts`` for Dz).

    ``lead_functions`` and ``curl_functions`` are the KERNEL mutation seams, SEPARATE so
    a gate can plant a defect in one launch while the other stays shipped. Dropping
    either is a silent DISARMING.
    """
    if not metal_cylindrical_complex_fused_hd_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    import numpy  # noqa: PLC0415

    from ..stepping import _boundary_kinds, _cylindrical_scan_starts  # noqa: PLC0415

    grid = fields.grid
    shape = tuple(int(n) for n in grid.shape)
    words = _chunk_bounds.plan_words(chunk_bounds, shape,
                                     "plan_metal_cylindrical_complex_fused_hd_pair")
    start_line = scan_starts("step_D", _cylindrical_scan_starts(fields, pml, "Dz"),
                             shape[0])
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

    # THE RADIAL SCRATCH: (nr, ny, nz) complex64, fully overwritten by every lead
    # launch before the curl reads it, shared between plans built against one
    # residency. NOT through `scratch_mirror`: that helper binds float32 by default,
    # which on a complex64 array is a silent cast that discards the imaginary plane
    # (the `scratch_twin` docstring records the same trap). The reuse rule is the
    # same, spelled here for the complex dtype.
    radial_name = "cylhd:radial:step_D"
    radial_host = residency.host(radial_name)
    if radial_host is None:
        radial_host = numpy.zeros(shape, dtype=numpy.complex64)
    radial = mirror(residency, radial_name, radial_host)
    volumes.append(radial_name)

    # THE CONSTITUTIVE COEFFICIENTS on H's sub-lattice: kps per axis then kms per axis,
    # integer lattice, in `h_cell`'s parameter order (complex_fused_hd_pair).
    suffix = "_h" if side["half_integer"] else ""
    coefficients = ([bind_real(f"pml:kps_{axis}{suffix}",
                               getattr(pml, f"kps_{axis}{suffix}")) for axis in "xyz"]
                    + [bind_real(f"pml:kms_{axis}{suffix}",
                                 getattr(pml, f"kms_{axis}{suffix}")) for axis in "xyz"])

    # THE TWO ROW VECTORS, from stepping's own builder, grid invariants.
    rows = shape[0]
    weights, rinv = scan.prefix_row_vectors("step_D", rows)
    weights_host, weights_t = scratch_mirror(
        residency, "cylhd:weights:step_D", lambda: weights, constant=True)
    rinv_host, rinv_t = scratch_mirror(
        residency, "cylhd:rinv:step_D", lambda: rinv, constant=True)
    volumes.extend(("cylhd:weights:step_D", "cylhd:rinv:step_D"))

    static_lead = (flux + [radial] + coefficients + [weights_t, rinv_t]
                   + [int32_tensor(start_line, residency.device)]
                   + [_params_tensor(shape, residency.device, chunk_bounds=words)])
    if 2 * len(ROTATED_NAMES) + len(static_lead) != LEAD_BINDINGS:  # pragma: no cover
        raise AssertionError((2 * len(ROTATED_NAMES) + len(static_lead), LEAD_BINDINGS))

    # LAUNCH 2's arguments, assembled EXACTLY as `cylindrical_complex._build_curl_plan`
    # assembles the certified plan's, with the magnetic sources resolved at run time
    # (they rotate) and the radial difference being the device scratch.
    targets = [bind_complex(name, getattr(fields, name)) for name in spec["targets"]]
    auxiliaries = [bind_complex("fu_" + name, getattr(fields, "fu_" + name))
                   for name in spec["targets"]]
    imr_rows = []
    for index, _register, sign in cylindrical_complex.IMR_TERMS["step_D"]:
        target = spec["targets"][index]
        row = cylindrical_complex.imr_rinv_row(target, sign, m, grid.courant, rows)
        imr_rows.append(bind_real(f"cylhd:imr:step_D:{target}", row))
    curl_coefficients = [bind_real(f"pml:{stem}_{axis}{spec['suffix']}",
                                   getattr(pml, f"{stem}_{axis}{spec['suffix']}"))
                         for axis in "xyz" for stem in ("kms", "sinv")]
    zero_rows = cylindrical_complex.zero_rows(m, bool(grid.accurate_fields_near_cylorigin))
    curl_static = tuple(targets) + tuple(auxiliaries)
    # THE CURL'S ARGUMENT ORDER IS THE CERTIFIED EMITTER'S: D, fu_D, H (resolved at run
    # time, they rotate), the six coefficient vectors, the five scalars, the radial
    # difference, the two i*m/r rows, the near-axis uniform, the axis factor, the m
    # word and the chunk record.
    curl_tail = (tuple(curl_coefficients)
                 + (shape[0], shape[1], shape[2], shape[0] * shape[1] * shape[2],
                    float(dtdx))
                 + (radial,) + tuple(imr_rows)
                 + (int(zero_rows),
                    float(cylindrical_complex.axis_factor("step_D", m, grid.courant)),
                    float(m),
                    _chunk_bounds.tensor(words, complex_fields._plan_device(residency))))

    leads = dict(lead_functions or {})
    curls = dict(curl_functions or {})
    for mode in contract_variants:
        leads.setdefault(mode, compile_lead(mode))
        curls.setdefault(mode, compile_curl(bcz, m_arm, mode))
    return MetalCylindricalComplexFusedHdPairPlan(
        residency, fields, shape, twins, radial_host, start_line, words, static_lead,
        curl_static, curl_tail, leads, curls, volumes, bcz, m_arm)


# ---------------------------------------------------------------------------
# Registration -- wired=False, refused by the composer on INSTALLABLE
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"cylindrical complex fused H/D pair cannot fill {slot}",))
    return metal_cylindrical_complex_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any,
              slot: str) -> Optional[MetalCylindricalComplexFusedHdPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_cylindrical_complex_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants,
        chunk_bounds=context.chunk_bounds)


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
