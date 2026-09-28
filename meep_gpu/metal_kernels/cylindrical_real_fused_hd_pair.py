"""The Dcyl m = 0 H->D seam in TWO launches: ``update_H`` and the radial prefix in one,
the certified ``step_D`` curl in the other, one product owning both slots.

THE CELL. ``h_to_d_seam.instances`` files 3 ``cylindrical m=0 -> cylindrical m=0`` rows
(``tests:TestAdjointSolver.test_adjoint_solver_cyl_n2f_fields_0_0``, ``..._1_0``,
``tests:TestPMLCylindrical.test_pml_cyl_0_0_0``) as ``buildable_not_built`` on all three
boards, identically. Every one is real float32 storage, ``ny = 1``, an active split-field
PML, and an electric ``GaussianPulsedSource`` whose withdraw is the no-op -- so nothing
runs between the two consults on any of them (driver.py:3311-3315).

===========================================================================
WHY THIS SEAM WAS FILED AS PERMANENTLY REFUSED, AND WHY THAT WAS WRONG HERE
===========================================================================

:mod:`.fused_hd_pair` refuses cylindrical by name: the cylindrical ``step_D`` differences
``cylindrical_rderiv_prefix(Hy)``, a column-serial radial scan, and a scan "is deliberately
NOT a kernel on this track (a hand-written scan is a different float32 number)". That is
the CuPy premise (``cupy.cumsum`` is not a sequential accumulation, device-measured) and
this backend's engine is NumPy (``coverage.MIRRORED_HOST_MODULE``), where ``numpy.cumsum``
IS a strictly-serial accumulation -- :mod:`.cylindrical_real` states the inversion at
:80-95 and ships the scan as a gated device kernel (``cyl_rderiv_prefix``, 0 differing
words of 257,712 against ``stepping.cylindrical_rderiv_prefix`` itself). So the premise
does not hold on this backend, and the seam is reachable with the shape below.

WHAT IS GENUINELY HARD ABOUT THIS SEAM, and it is one thing: the prefix's SOURCE is what
the seam's first half WRITES. ``stepping.py:447`` prefixes ``magnetic["Hy"]`` at
``ir0 = 0.5``, and under an active PML ``get_H`` returns the STORED H -- ``update_H``'s own
output (fields.py:1181-1182). A single launch would have every thread's Dz depend on
every ``Hy[r' <= r]`` of its column as written by other threads in the same launch. The
prefix cannot be collapsed to a local halo either: ``prefix[i+1] - prefix[i] ==
increment[i+1]`` fails in float32 on 103,175 of 124,096 words (the adjudication's direct
probe), so a halo weld would be bit-right on most cells and bit-wrong on a sparse,
data-dependent set. THE UNIT OF RECOMPUTE IS A WHOLE RADIAL COLUMN.

===========================================================================
THE SHAPE: TWO LAUNCHES, ZERO FOREIGN READS OF ANYTHING THIS PRODUCT WRITES
===========================================================================

**Launch 1** occupies ``update_H``. Every thread computes its own cell's H pointwise
through :func:`.fused_hd_pair.h_cell_function` -- the CERTIFIED ``update_H`` body as a
function of ``(i, j, k)`` over const ``B``, pre-launch ``H``/``f_w_H`` and the per-axis
coefficients -- and stores it to WRITE-ONLY SCRATCH (``H_out``/``f_w_H_out``), exactly as
:mod:`.fused_hd_pair` does. The COLUMN-LEADER threads (``i == 0``, one per ``(phi, z)``
column) additionally walk their own radial column, RECOMPUTING ``Hy`` at every row through
the SAME ``h_cell`` from the SAME const inputs -- never reading ``H_out`` -- forming the
weighted term ``f_p * weights`` (field LEFT), the difference of two already-weighted rows
over the row-constant divisor spelled as ``/`` (NOT a reciprocal multiply: the real family
measures that mutation at 6,373 words, and the adjudication's ``adj_divide`` at
102,004 of 400,000), and accumulating the prefix STRICTLY SERIALLY in the exact order of
``stepping.cylindrical_rderiv_prefix`` at ``ir0 = 0.5``. The scan text is LIFTED from
:func:`.cylindrical_real.cylindrical_prefix_source` with exactly two declared edits -- the
two ``src[...]`` reads become the recompute (:data:`SCAN_LIFT_EDITS`). Nothing written by
this launch is read by it, so no thread observes another thread's store and the launch has
no schedule to depend on.

**Launch 2** occupies ``step_D``: the CERTIFIED cylindrical curl, UNCHANGED --
:func:`.cylindrical_real.compile_cylindrical_curl` for ``step_D`` -- reading the ROTATED
``H`` (the scratch launch 1 just wrote, now under the engine's own names) and the prefix
as const. ``Dx`` reads the RAW ``Hy``; only ``Dz`` takes the prefixed term
(stepping.py:461-462); the m = 0 axis post-add re-reads the raw ``Hy`` at r = 0.

The launcher ROTATES the six ``H``/``f_w_H`` references between the two launches, which
is :class:`.offdiag_weld_common.ScratchWeldPairPlan`'s certified choreography one seam
over.

WHY THE RECOMPUTED ``Hy`` IS THE POINTWISE ``Hy`` TO THE BIT. ``update_H`` is pointwise
(stepping.py:907-923 through ``_apply_constitutive_pml``, :2112-2143): the stepped value
at a cell is a pure function of state this launch does not write, evaluated twice by one
inline function with the contraction pragma on. The gate does not take that on
construction: leg ``recompute_identity`` runs launch 1 and then the SHIPPED
``cyl_rderiv_prefix`` over the rotated ``H_out`` into a second scratch, and compares the
two prefixes word for word, beside ``stepping.cylindrical_rderiv_prefix`` itself over the
synced host ``Hy``; the null control runs the shipped scan over the PRE-launch ``Hy`` and
must differ.

===========================================================================
LAUNCHES, COST, BINDINGS
===========================================================================

* **Launches at the seam: 3 -> 2.** The separate certified composition is ``update_H``
  (1) + ``step_D`` (the scan, then the curl: 2). This product is 2. Its launch counters
  are carried APART (:attr:`constitutive_launches`, :attr:`curl_launches`) so "launch 1
  ran and launch 2 did not" is distinguishable from "the plan ran once".
* **Extra work: one redundant pointwise ``Hy`` per cell** -- ``n_elem`` recomputes over
  the leaders' walks, 1.0x the cell count, against 233.6x for a per-thread whole-column
  recompute. Not a throughput claim; a count.
* **Bindings: 25 of the 31 the platform allows** for launch 1 (:data:`CONSTITUTIVE_BINDINGS`
  = 24 pointers + one packed ``Params``), six of headroom; launch 2 is the certified
  curl's 22. Both are COMPILED by the gate rather than counted from a signature, and the
  ceiling itself is bisected on this host (leg ``binding_ceiling``): 30 pointers plus one
  struct compiles, 31 is refused, the launch-1 signature with six more pointers compiles
  and with seven is refused with the platform's own ``'buffer' attribute parameter is
  out of bounds``.

===========================================================================
WHAT SITS IN THE SEAM
===========================================================================

Nothing is INJECTED between the two consults (the electric injection is one seam later,
the magnetic one seam earlier), so :data:`CARRIES_DEPOSIT_REPAIR` is False as a fact about
the driver and :mod:`..deposit_repair` is not consulted. The one pass there is the electric
integrated-source withdraw, owned by :mod:`..withdraw_hoist`; :data:`HOISTS_THE_WITHDRAW`
is False in this round, so a row with a standing withdraw is refused BY NAME. None of the
cell's three rows has one.

===========================================================================
NOT INSTALLED, AND THE REASON IS MEASURED ARBITRATION -- A TIE, NOT A LOSS
===========================================================================

:data:`INSTALLABLE` is False. What the composer installs on these rows TODAY, measured
through ``plan_step(fuse=True)`` on the gate's fixture (leg ``arbitration``): the
``cylindrical m=0`` curl arm at ``step_B``, the ``cylindrical m=0`` constitutive arm at
``update_H``, and the RELEASED :mod:`.cylindrical_real_fused_electric_pair` across
``step_D``/``update_E`` (it carries the rows' electric deposit through the repair
bracket) -- 2 + 1 + 2 = 5 launches over the slot path. The B->H cylindrical twin holds
no absorb row and does not install. This product would replace ``update_H`` +
``step_D`` (2) and leave ``update_E`` to the constitutive arm (1): 2 + 2 + 1 = 5. A
TIE at the step level, one seam served either way, and the released precedent gives a
tie to the incumbent: the D->E pair installs first (``FUSED_PAIR_SEAMS`` order) and
holds ``step_D``, so ``_pair_may_absorb`` refuses this product by name. Under the
owner's standing rule the priced predicate gap is closed regardless, as
:mod:`.fused_hd_pair` closed its 47.

===========================================================================
STANDALONE IN THIS ROUND: registered UNWIRED, no absorb row, no board row
===========================================================================

This module registers ``wired=False`` on ``update_H`` and is reached by no composer
table yet. The rows that reach it -- ``registry.FAMILY_MODULES``,
``launch.FUSED_PAIR_ARMS`` (``("cylindrical m=0", "cylindrical m=0")``),
``build_fusion_matrix.PRODUCTS`` / ``RELEASE_BINDING`` and the ``fingerprints.json`` weld
entry ``mint_metal_weld.py`` derives from the gate artifact -- are edits to existing
modules that three in-flight gate campaigns pin, and they are written as
``lanes/cyl_round/metal_m0/wiring.patch`` rather than applied. :data:`WELD_OWED` says so
in the words ``test_metal_weld_contract`` reads.

WHAT THIS MODULE DOES NOT CARRY: complex64 cylindrical storage (the |m| >= 1 rows and the
m = 0 complex arm belong to the complex family, whose scan must spell the divide as a
RECIPROCAL MULTIPLY -- the spelling INVERTS between the two families, measured), any
|m| >= 1 coupling, a fold, beta, BFAST, conductivity, an inactive absorber, a standing
withdraw, or the E->B cylindrical seam (the ``ir0 = 0.0`` prefix over ``Ey`` inside
``step_B``, which has the same structure and is unpriced).
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import CONSTITUTIVE_SIDES, Coverage
from ..triton_kernels.launch import SUB_STEPS
from . import offdiag_weld_common as _weld
from . import shaders, templates
from .cylindrical_real import (
    METALLIC,
    PREFIX_IR0,
    boundary_codes,
    compile_cylindrical_curl,
    cylindrical_curl_source,
    cylindrical_prefix_source,
    cylindrical_real_constitutive_coverage,
    cylindrical_real_curl_coverage,
    prefix_row_vectors,
    scan_rows,
    scratch_mirror,
)
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .fused_hd_pair import H_CELL_POINTERS, H_CELL_TAIL_ARGS, h_cell_function
from .. import withdraw_hoist as _withdraw_hoist

FAMILY = "cylindrical_real_fused_hd_pair"

#: The sub-step slot this arm is registered on: the FIRST half of the seam in the
#: driver's order, so a refusal is named on the slot the fusion starts at.
SLOT = "update_H"

#: The driver passes this product performs, in driver order (driver.py:3311, :3315).
#: Nothing between them is carried: the only statement there is the electric withdraw,
#: which :data:`HOISTS_THE_WITHDRAW` declines.
REPLACES: Tuple[str, ...] = ("update_H", "step_D")

#: The seam name ``launch.FUSED_PAIR_SEAMS`` files this row under; spelled through the
#: module that owns the in-seam pass so the two cannot drift.
SEAM: str = _withdraw_hoist.SEAM

CURL_SUB_STEP = "step_D"
CONSTITUTIVE_SIDE = "H"

#: ``cylindrical_rderiv_prefix``'s ``ir0`` on this side: half of Hp's r-Yee shift.
#: Read off the certified family's table rather than written as 0.5 here.
IR0: float = PREFIX_IR0[CURL_SUB_STEP]

#: Does this product bracket its launch with the DEPOSIT repair? NO, and it is a fact
#: about the driver: nothing is injected between the ``update_H`` and ``step_D``
#: consults, so there is no deposit in this seam to carry.
CARRIES_DEPOSIT_REPAIR = False

#: Does this product perform the seam's electric withdraw before its launch? NO in this
#: round. The only wiring that performs it is ``launch._install_fused_pair``'s
#: ``withdraw_hoist.SEAM`` branch, which this product does not reach (no absorb row,
#: :data:`INSTALLABLE` False), so True here would claim a wrapper that cannot fire. The
#: predicate refuses every row with a standing withdraw by name; the cell has none.
HOISTS_THE_WITHDRAW = False

#: May the composer install this product? NO, on a MEASURED arbitration verdict: on
#: every row this predicate reaches, BOTH neighbouring seams are now held by released
#: fused pairs, and this product would take one slot from each.
INSTALLABLE = False

INSTALLABLE_REASON = (
    "measured 2026-09-06 through the Metal composer on the Dcyl m = 0 fixture "
    "(parity/meep_gpu/results/metal_cylindrical_real_fused_hd_pair_2026-09-06_cyl, leg "
    "arbitration): with no absorb row the seam loop refuses this product before asking "
    "anything about the run; with the absorb row ('cylindrical m=0', 'cylindrical m=0') "
    "applied IN-PROCESS and INSTALLABLE held True, plan_step(fuse=True) still refuses "
    "it by name -- step_D was selected by the released 'cylindrical m=0 fused electric "
    "D/E pair' arm, which installs first and holds the slot, so _pair_may_absorb names "
    "that arm -- and the installed composition does not move by one slot. Today's "
    "composition on these rows is the cylindrical m=0 curl at step_B (2 launches: scan, "
    "curl), the cylindrical m=0 constitutive at update_H (1) and the released D->E pair "
    "(2): 5 launches over step_B..update_E; this product in its place would be 2 + 2 + "
    "1 = 5. A TIE at the step level and one seam served either way, resolved for the "
    "incumbent as the released precedent resolves every tie.\n\n"
    "RE-MEASURED 2026-09-17, AND IT IS NO LONGER A TIE. The sentence that used to "
    "close this reason -- 'the B->H cylindrical twin holds no absorb row and does not "
    "install here, so no row is a loss and none is a gain' -- was a statement about "
    "the composer's tables rather than about this product, and the all-paths batch "
    "changed it: cylindrical_real_fused_magnetic_pair gained a "
    "launch.FUSED_PAIR_ARMS row, and the 2026-09-17 arbitration leg measures BOTH "
    "seams fused (step_B/update_H on 'cylindrical m=0 fused magnetic B/H pair', "
    "step_D/update_E on 'cylindrical m=0 fused electric D/E pair'). Installing this "
    "product in that composition takes one slot from EACH released pair -- two pairs "
    "become one -- which is a LOSS, not a tie, and the verdict it supports is the "
    "same one it always was. The composer's first refusal moved with it, from step_D "
    "to update_H: 'update_H was selected by the cylindrical m=0 fused magnetic B/H "
    "pair arm; this fused product implements the cylindrical m=0 one and may not "
    "substitute it'. This is slot arbitration, not a verdict about the arithmetic, "
    "which the family's own gate certifies on its fixture and over the cell's three "
    "lifted rows")

#: THE WELD IS OWED TO THE WIRING ROUND, and this constant is how the fleet partition
#: (``test_metal_weld_contract.test_every_metal_family_is_welded``) is told so. The
#: gate ran and RELEASED on this host --
#: parity/meep_gpu/results/metal_cylindrical_real_fused_hd_pair_2026-09-06_cyl/gate.json
#: -- but a weld entry lives in ``metal_kernels/fingerprints.json``, an existing record
#: this round may not edit while three gate campaigns pin its bytes. What a release
#: still owes: ``mint_metal_weld.py --family cylindrical_real_fused_hd_pair --stamp
#: 2026-09-06_cyl`` against that artifact, the ``registry.FAMILY_MODULES`` row, the
#: ``launch.FUSED_PAIR_ARMS`` row, the board's ``PRODUCTS``/``RELEASE_BINDING`` rows --
#: all written in lanes/cyl_round/metal_m0/wiring.patch -- and then emptying this
#: string in the same edit as the mint, which drifts the artifact's raw digest of this
#: file and so is followed by a re-gate into a fresh directory.
WELD_OWED: str = ""

#: The rotating volumes, in SIGNATURE ORDER: the three stored magnetic fields then the
#: three split-field histories. Launch 1 binds every SCRATCH first and every PRE-LAUNCH
#: buffer second, in this order, then the static arguments.
ROTATED_NAMES: Tuple[str, ...] = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")

#: Launch 1's signature: 24 pointers plus one packed ``Params``. Six under the ceiling.
CONSTITUTIVE_POINTERS = 24
CONSTITUTIVE_BINDINGS = 25

#: Launch 2's signature is the certified cylindrical curl's, unchanged: 22 bindings
#: (16 pointers + 6 scalars), read off :mod:`.cylindrical_real`'s own text by the test.
CURL_BINDINGS = 22

#: How many pointers launch 1 could still take: the gate compiles the shipped signature
#: plus this many and requires success, plus one more and requires the refusal.
HEADROOM = MAX_BUFFER_BINDINGS - CONSTITUTIVE_BINDINGS

#: Every line of the certified SCAN text this product does not lift verbatim, with the
#: reason. DATA, so the gate's transcription leg can assert the list and a needle that
#: stopped matching RAISES rather than leaving a stale read standing. THERE IS NO
#: ARITHMETIC IN THIS TABLE: the two edits replace a memory read with the recompute of
#: the same value; the weight multiply, the difference, the divide and the serial
#: accumulation are untouched.
SCAN_LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "    float prev = src[base] * weights[0];",
     "became": f"    float prev = h_cell(0, j, k, {H_CELL_TAIL_ARGS}).a1 * weights[0];",
     "why": "row 0 of the column's Hy, RECOMPUTED from the pre-launch state through "
            "the same h_cell every thread stores its own cell from, instead of read "
            "from a volume this launch writes. Field LEFT of the weight, as "
            "stepping.py:1313/:1327 spells it."},
    {"line": "        float w = src[i * nyz + base] * weights[i];",
     "became": f"        float w = h_cell(i, j, k, {H_CELL_TAIL_ARGS}).a1 * weights[i];",
     "why": "row i of the column's Hy, recomputed the same way. `i` here is the scan "
            "loop's own variable (the certified loop's spelling, kept), shadowing the "
            "leader's i == 0 inside the loop body only."},
)

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CONSTITUTIVE_BINDINGS", "CONSTITUTIVE_POINTERS",
    "CONSTITUTIVE_SIDE", "CURL_BINDINGS", "CURL_SUB_STEP", "FAMILY", "HEADROOM",
    "HOISTS_THE_WITHDRAW", "INSTALLABLE", "INSTALLABLE_REASON", "IR0", "REPLACES",
    "ROTATED_NAMES", "SCAN_LIFT_EDITS", "SEAM", "SLOT", "WELD_OWED",
    "MetalCylindricalRealFusedHdPairPlan", "certified_scan_tail",
    "compile_cylindrical_real_fused_hd_pair_constitutive",
    "corpus_digest", "cylindrical_real_fused_hd_pair_constitutive_source",
    "cylindrical_real_fused_hd_pair_curl_source",
    "largest_fitting_source", "metal_cylindrical_real_fused_hd_pair_coverage",
    "plan_metal_cylindrical_real_fused_hd_pair", "refuted_over_the_ceiling_source",
    "register_arms", "shipped_signature_bindings",
]


# ---------------------------------------------------------------------------
# The lift: the certified scan as a column walk over the recomputed Hy
# ---------------------------------------------------------------------------

#: Where the certified scan's per-thread prologue ends. Everything after this line --
#: the ``+0.0`` first row, the row-0 weighted term, the serial loop -- is lifted.
_SCAN_PROLOGUE_END = "    int base = j * nzi + k;\n"


def certified_scan_tail(contract: str = shaders.CONTRACT_OFF) -> str:
    """The ``step_D`` radial scan BELOW its prologue, its two source reads redirected.

    Not a transcription: this is :func:`.cylindrical_real.cylindrical_prefix_source`'s
    own output for ``step_D`` with the preamble, the signature and the per-thread
    index prologue cut, and exactly the edits :data:`SCAN_LIFT_EDITS` names applied
    through :func:`.offdiag_weld_common.needle`, so a spelling that drifted raises
    rather than emitting a scan that reads a stale volume. The ``out`` pointer name and
    every arithmetic line are the certified text's.
    """
    source = cylindrical_prefix_source(CURL_SUB_STEP, contract)
    if _SCAN_PROLOGUE_END not in source:
        raise AssertionError(
            "the certified radial scan no longer carries its index prologue anchor; "
            "this family lifts the scan below it and has nowhere to cut")
    tail = source.split(_SCAN_PROLOGUE_END, 1)[1]
    if not tail.endswith("}\n"):
        raise AssertionError("the certified radial scan source does not end with '}'")
    tail = tail[: -len("}\n")]
    for edit in SCAN_LIFT_EDITS:
        tail = _weld.needle(tail, edit["line"] + "\n", edit["became"] + "\n")
    if "src[" in tail:
        raise AssertionError(
            "the lifted scan still reads `src`; in this kernel there is no such "
            "pointer and every Hy the scan consumes must be the recompute")
    return tail


# ---------------------------------------------------------------------------
# The source of launch 1
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params { uint nx; uint ny; uint nz; uint n_elem; };

__H_CELL__

kernel void cyl_real_hd_constitutive_prefix(
    device float*       ho0     [[buffer(0)]],
    device float*       ho1     [[buffer(1)]],
    device float*       ho2     [[buffer(2)]],
    device float*       wo0     [[buffer(3)]],
    device float*       wo1     [[buffer(4)]],
    device float*       wo2     [[buffer(5)]],
    device const float* hi0     [[buffer(6)]],
    device const float* hi1     [[buffer(7)]],
    device const float* hi2     [[buffer(8)]],
    device const float* wi0     [[buffer(9)]],
    device const float* wi1     [[buffer(10)]],
    device const float* wi2     [[buffer(11)]],
    device const float* b0      [[buffer(12)]],
    device const float* b1      [[buffer(13)]],
    device const float* b2      [[buffer(14)]],
    device const float* kp0     [[buffer(15)]],
    device const float* kp1     [[buffer(16)]],
    device const float* kp2     [[buffer(17)]],
    device const float* kmx     [[buffer(18)]],
    device const float* kmy     [[buffer(19)]],
    device const float* kmz     [[buffer(20)]],
    device float*       out     [[buffer(21)]],
    device const float* weights [[buffer(22)]],
    device const float* divisor [[buffer(23)]],
    constant Params&    prm     [[buffer(24)]],
    uint idx [[thread_position_in_grid]])
{
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    // The dispatch is sized from the first tensor argument's element count, so a
    // volume wider than n_elem would step cells the array path does not own.
    if (idx >= n_elem) { return; }

    int nxi = int(nx), nyi = int(ny), nzi = int(nz);
    int nyz = nyi * nzi;
    int ii  = int(idx);
    int k   = ii % nzi;
    int plane = ii / nzi;
    int j   = plane % nyi;
    int i   = plane / nyi;

    // --- update_H, computed into registers and stored to SCRATCH ---------------
    // Nothing written here is read by this launch: every thread's own cell comes
    // from `h_cell` over the PRE-LAUNCH state, and the column leader below
    // RECOMPUTES the Hy it scans from that same state rather than reading ho1.
    h_cell_result own = h_cell(i, j, k, __H_CELL_ARGS__);
    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;
    wo0[ii] = own.src0; wo1[ii] = own.src1; wo2[ii] = own.src2;

    // --- the radial prefix, by the COLUMN LEADER (r = 0) -------------------------
    // One thread per (phi, z) column walks its whole column from r = 0 upward,
    // strictly serially, exactly as `cylindrical_real.cyl_rderiv_prefix` does and
    // as numpy.cumsum does (the oracle: stepping.cylindrical_rderiv_prefix at
    // ir0 = 0.5). The text below is that kernel's, lifted, with the two source reads
    // replaced by the recompute (SCAN_LIFT_EDITS).
    if (i != 0) { return; }
    int base = j * nzi + k;
__SCAN__
}
"""


def cylindrical_real_fused_hd_pair_constitutive_source(
        contract: str = shaders.CONTRACT_OFF) -> str:
    """Launch 1's source for one contraction mode.

    No boundary specialisation: ``update_H`` is pointwise and the scan reads its own
    column, so nothing here depends on the ghost rule. That belongs to launch 2, which
    is the certified curl and takes the boundary triple there.
    """
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__H_CELL__": h_cell_function(contract),
        "__H_CELL_ARGS__": H_CELL_TAIL_ARGS,
        "__SCAN__": certified_scan_tail(contract),
    })


def cylindrical_real_fused_hd_pair_curl_source(
        codes: Sequence[int], contract: str = shaders.CONTRACT_OFF) -> str:
    """Launch 2's source: the certified cylindrical ``step_D`` curl, byte for byte.

    Spelled as a function of this module so the gate can assert that the text this
    product launches at ``step_D`` IS :func:`.cylindrical_real.cylindrical_curl_source`'s
    output and nothing else. The r axis is pinned METALLIC by that emitter and refused
    again here before the call.
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
    return cylindrical_curl_source(CURL_SUB_STEP, codes, contract)


def compile_cylindrical_real_fused_hd_pair_constitutive(
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """Launch 1's entry point for one contraction mode."""
    return compile_source(cylindrical_real_fused_hd_pair_constitutive_source(
        contract)).cyl_real_hd_constitutive_prefix


def shipped_signature_bindings() -> int:
    """How many bindings launch 1 declares, counted off its own emitted text."""
    source = cylindrical_real_fused_hd_pair_constitutive_source()
    signature = source.split("kernel void cyl_real_hd_constitutive_prefix(", 1)[1]
    signature = signature.split("uint idx [[thread_position_in_grid]])", 1)[0]
    return signature.count("[[buffer(")


def _padded_signature_source(extra: int, contract: str = shaders.CONTRACT_OFF) -> str:
    """Launch 1's signature with ``extra`` more pointers, its body touching them all.

    What is measured is the SIGNATURE; a body the compiler could drop would let
    dead-code elimination decide where the ceiling is, so every added pointer is read
    and the first output is written from the sum.
    """
    source = cylindrical_real_fused_hd_pair_constitutive_source(contract)
    anchor = "    constant Params&    prm     [[buffer(24)]],\n"
    if anchor not in source:
        raise AssertionError("launch 1's signature no longer binds Params at 24")
    lines = "".join(f"    device const float* extra{n} [[buffer({24 + n})]],\n"
                    for n in range(extra))
    source = source.replace(
        anchor, lines + f"    constant Params&    prm     [[buffer({24 + extra})]],\n")
    touch = " + ".join(f"extra{n}[0]" for n in range(extra))
    marker = "    ho0[ii] = own.a0; ho1[ii] = own.a1; ho2[ii] = own.a2;\n"
    if marker not in source:
        raise AssertionError("launch 1's own-cell store line moved; the padded "
                             "signature has no anchor for its touch")
    return source.replace(
        marker, f"    ho0[ii] = own.a0 + ({touch}); ho1[ii] = own.a1; ho2[ii] = own.a2;\n")


def largest_fitting_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """Launch 1 plus :data:`HEADROOM` pointers -- 31 bindings, which must COMPILE."""
    return _padded_signature_source(HEADROOM, contract)


def refuted_over_the_ceiling_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """Launch 1 plus :data:`HEADROOM` + 1 pointers -- 32 bindings, which must FAIL.

    Not shipped and not launchable: it exists so the gate can compile it and require the
    platform's own refusal, which is what turns :data:`HEADROOM` from an argument into a
    measurement.
    """
    return _padded_signature_source(HEADROOM + 1, contract)


def corpus_digest(contract: str = shaders.CONTRACT_OFF) -> Dict[str, Any]:
    """One sha256 over launch 1's emitted text, so a weld can pin the DEVICE source.

    Launch 2 is :mod:`.cylindrical_real`'s certified emitter and is pinned by that
    family's own corpus; this digest is the text THIS module is answerable for.
    """
    from . import corpus as _corpus  # noqa: PLC0415

    return _corpus.corpus_digest_for(cylindrical_real_fused_hd_pair_constitutive_source,
                                     contract=(contract,))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_cylindrical_real_fused_hd_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Coverage:
    """May this product span ``update_H`` -> the electric withdraw -> ``step_D`` here?

    A conjunction of the two halves' OWN certified predicates -- the ``cylindrical m=0``
    constitutive arm on ``update_H`` and the ``cylindrical m=0`` curl arm on ``step_D``,
    asked in the driver's order -- plus the seam clauses. Nothing is weakened: a
    configuration either half refuses is refused here with that half's reasons,
    prefixed so a reader can tell which side said it.
    """
    reasons: List[str] = []

    magnetic = cylindrical_real_constitutive_coverage(fields, pml, CONSTITUTIVE_SIDE,
                                                     residency)
    if not magnetic.covered:
        reasons.extend(f"cylindrical constitutive half: {reason}"
                       for reason in magnetic.reasons)
    curl = cylindrical_real_curl_coverage(fields, pml, CURL_SUB_STEP, residency)
    if not curl.covered:
        reasons.extend(f"cylindrical curl half: {reason}" for reason in curl.reasons)

    # THE SEAM'S ONE PASS. Nothing is INJECTED between the two consults, so
    # `deposit_repair` is not consulted at all; what IS between them is the electric
    # integrated-source withdraw (driver.py:3313-3314), and `withdraw_hoist` owns it.
    # IGNORANCE IS NEVER AN EMPTY SET: `Fields` does not hold the source list, so a
    # predicate that inferred "no withdraw stands" from not being told would be the
    # over-covering refusal this clause exists to prevent.
    reasons.extend(_withdraw_hoist.seam_withdraw_reasons(
        fields, sources,
        undeclared=(
            "the source set was not declared: this predicate cannot infer from Fields "
            "that no electric withdraw stands between update_H and step_D"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) has a standing integrated "
            f"electric withdraw, which the driver runs BETWEEN the update_H and "
            f"step_D consults (driver.py:3313-3314); this product declares "
            f"HOISTS_THE_WITHDRAW = False, so nothing would perform the withdraw "
            f"before the launch"),
        hoists_the_withdraw=HOISTS_THE_WITHDRAW,
        span=REPLACES))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE ROTATION IS THE PRODUCT'S OWN INVARIANT. The plan swaps the ENGINE's
    # references for the six volumes in ROTATED_NAMES between its two launches, so
    # those attributes must be settable and must be the arrays the residency mirrored.
    for name in ROTATED_NAMES:
        if getattr(fields, name, None) is None:
            reasons.append(
                f"{name} is not allocated; this weld rotates it against a plan-owned "
                f"scratch twin between its two launches")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class MetalCylindricalRealFusedHdPairPlan:
    """TWO launches spanning the H->D seam, with the rotation between them.

    NOT a :class:`.plans.KernelPlan` and NOT a :class:`.offdiag_weld_common.ScratchWeldPairPlan`,
    though it borrows the latter's choreography: that class's whole contract is ONE
    launch then the rotation, and this plan's is launch 1, the rotation, then launch 2
    reading what launch 1 wrote. The divergence is stated here rather than hidden
    behind an overridden ``run``.

    THE ORDER IS THE CONTRACT. Launch 2 reads the prefix and the H launch 1 writes, and
    both are functions of THIS step's B -- so launch 2 hoisted ahead of launch 1, or
    launch 2 bound to the pre-launch H, feeds the curl the PREVIOUS half-step's magnetic
    field: a smooth, plausible, entirely wrong field rather than an error, and a gate
    mutation rather than a comment.

    ``launches`` COUNTS KERNEL LAUNCHES, NOT ``run`` CALLS, and it is two per cycle.
    :attr:`constitutive_launches` and :attr:`curl_launches` are carried apart so a gate
    can assert that BOTH advanced and no slot passed by not executing.
    """

    __slots__ = ("shape", "n_elem", "scan_shape", "n_cols", "dtdx", "axis_coef", "bc",
                 "residency", "fields", "volumes", "rotated", "rotated_names",
                 "prefix_host", "launches", "constitutive_launches", "curl_launches",
                 "runs", "_constitutive_functions", "_curl_functions",
                 "_constitutive_static", "_curl_static")

    family = "cylindrical m = 0 fused stored-H/PML D-curl pair"

    replaces_sub_steps = REPLACES

    #: No deposit repair is installed around this plan; the seam carries none.
    repair_paths: Tuple[str, ...] = ()

    #: This plan launches kernels; the composer's planned/null split reads it.
    performs_device_work = True

    #: TWO: the constitutive-plus-prefix launch and the certified curl.
    launches_per_run = 2

    REPR_FIELDS = ("shape", "bc", "scan_shape")

    def __init__(self, shape, scan_shape, dtdx: float, axis_coef: float, bc,
                 residency: Residency, fields: Any, rotated: Mapping[str, Any],
                 prefix_host: Any, constitutive_static: Sequence[Any],
                 curl_static: Sequence[Any],
                 constitutive_functions: Mapping[str, Any],
                 curl_functions: Mapping[str, Any],
                 volumes: Sequence[str]) -> None:
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or int(self.shape[1]) != 1:
            raise ValueError(
                f"a Dcyl grid stores one phi cell; shape {self.shape} does not "
                f"(the exp(i*m*phi) dependence is analytic)")
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.scan_shape = tuple(int(n) for n in scan_shape)
        self.n_cols = self.shape[1] * self.shape[2]
        self.dtdx = float(dtdx)
        self.axis_coef = float(axis_coef)
        self.bc = tuple(int(code) for code in bc)
        self.residency = residency
        self.fields = fields
        self.rotated = dict(rotated)
        self.rotated_names = ROTATED_NAMES
        if set(self.rotated_names) != set(self.rotated):
            raise ValueError(
                f"the rotation order {self.rotated_names} does not name the same "
                f"volumes as the twin table {tuple(sorted(self.rotated))}")
        self.prefix_host = prefix_host
        self.volumes = tuple(dict.fromkeys(volumes))
        self._constitutive_static = tuple(constitutive_static)
        self._curl_static = tuple(curl_static)
        self._constitutive_functions = dict(constitutive_functions)
        self._curl_functions = dict(curl_functions)
        self.launches = 0
        self.constitutive_launches = 0
        self.curl_launches = 0
        self.runs = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        """The contraction modes BOTH launches were built with -- the intersection."""
        return tuple(sorted(set(self._constitutive_functions) & set(self._curl_functions)))

    def _function(self, table: Mapping[str, Any], mode: str, which: str) -> Any:
        function = table.get(mode)
        if function is None:
            raise KeyError(
                f"this plan holds no {mode!r} {which} variant (it was built with "
                f"{self.variants}); build it with contract_variants={(mode,)} rather "
                f"than launching the pinned one")
        return function

    def resolve(self) -> Tuple[List[Any], List[Any]]:
        """(scratch write tensors, pre-launch read tensors), by CURRENT role.

        Resolved by HOST IDENTITY immediately before every launch
        (:meth:`.Residency.tensor_for_host`), never cached: the two roles alternate
        between two physical buffers every run.
        """
        writes: List[Any] = []
        reads: List[Any] = []
        for name in self.rotated_names:
            current = getattr(self.fields, name)
            read = self.residency.tensor_for_host(current)
            write = self.residency.tensor_for_host(self.rotated[name])
            if read is None or write is None:
                raise RuntimeError(
                    f"the {FAMILY} weld lost the device mirror of {name}; a launch "
                    f"against an unresolved buffer would step garbage")
            if read is write:
                raise RuntimeError(
                    f"the {FAMILY} weld resolved {name} and its scratch to ONE tensor; "
                    f"the whole design is that nothing written is read, so an aliased "
                    f"pair is refused rather than launched")
            reads.append(read)
            writes.append(write)
        return writes, reads

    def launch_constitutive(self, writes: Sequence[Any], reads: Sequence[Any],
                            contract: Optional[str] = None) -> None:
        """Launch 1 alone, on the given scratch/pre-launch bindings. Counted."""
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._function(self._constitutive_functions, mode, "constitutive")
        self.launches += 1
        self.constitutive_launches += 1
        function(*(list(writes) + list(reads) + list(self._constitutive_static)))

    def launch_curl(self, magnetic: Sequence[Any], contract: Optional[str] = None) -> None:
        """Launch 2 alone, reading ``magnetic`` as ``g0..g2``. Counted.

        Never run first: it reads the prefix launch 1 writes.
        """
        mode = shaders.CONTRACT_OFF if contract is None else contract
        function = self._function(self._curl_functions, mode, "curl")
        static = self._curl_static
        self.launches += 1
        self.curl_launches += 1
        # THE CURL'S ARGUMENT ORDER IS THE CERTIFIED EMITTER'S: D, fu_D, H, pfx, the
        # six coefficient vectors, then the six scalars. `magnetic` is spliced into
        # the H slot; everything else is static for the life of the plan.
        function(*(list(static[:6]) + list(magnetic) + list(static[6:])))

    def rotate(self) -> None:
        """Swap the ENGINE's references for the six volumes with the plan's twins."""
        for name in self.rotated_names:
            current = getattr(self.fields, name)
            setattr(self.fields, name, self.rotated[name])
            self.rotated[name] = current

    def run(self, contract: Optional[str] = None) -> None:
        """The seam: launch 1, the rotation, launch 2. In place for D and fu_D."""
        self.runs += 1
        writes, reads = self.resolve()
        self.launch_constitutive(writes, reads, contract)
        # THE ROTATION -- only after launch 1 returned. The engine's references move
        # to the freshly written buffers; the pre-launch buffers become the plan's
        # twins and will be next run's scratch.
        self.rotate()
        # LAUNCH 2 READS WHAT LAUNCH 1 WROTE: the scratch tensors, now the engine's H.
        self.launch_curl(writes[:3], contract)

    def describe(self) -> str:
        parts = ", ".join(f"{name}={getattr(self, name)!r}" for name in self.REPR_FIELDS)
        return f"MetalCylindricalRealFusedHdPairPlan({parts}, variants={self.variants})"

    def __repr__(self) -> str:
        return self.describe()


def _params_tensor(shape: Sequence[int], device: str) -> Any:
    """The four extents as one 16-byte device record.

    PLAN-OWNED AND OUTSIDE THE RESIDENCY REGISTRY, for the reason every fused pair's
    ``_params_tensor`` gives: :meth:`.Residency.mirror` binds float32 and complex64
    volumes and refuses anything else BY NAME, because a wider element silently
    reinterprets. This record is neither.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=np.dtype([("nx", "<u4"), ("ny", "<u4"), ("nz", "<u4"),
                                         ("n_elem", "<u4")]))
    nx, ny, nz = (int(n) for n in shape)
    record[0] = (nx, ny, nz, nx * ny * nz)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


def _constitutive_functions(contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_cylindrical_real_fused_hd_pair_constitutive(mode)
            for mode in contract_variants}


def _curl_functions(codes, contract_variants: Sequence[str]) -> Dict[str, Any]:
    return {mode: compile_cylindrical_curl(CURL_SUB_STEP, codes, mode)
            for mode in contract_variants}


def plan_metal_cylindrical_real_fused_hd_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        constitutive_function: Optional[Mapping[str, Any]] = None,
        curl_function: Optional[Mapping[str, Any]] = None,
        scan_vectors: Optional[Tuple[Any, Any]] = None,
        ) -> Optional[MetalCylindricalRealFusedHdPairPlan]:
    """Build the fused Dcyl H/D plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must fall
    back to the array path, never raise into a caller that would otherwise have stepped
    correctly.

    ``constitutive_function`` and ``curl_function`` are the KERNEL mutation seams, and
    they are SEPARATE so a leg can plant a defect in one launch while the other stays
    shipped. ``scan_vectors`` is the HOST mutation seam this family adds: the
    ``(weights, divisor)`` ladder the scan binds, so a leg can hand it the ``ir0 = 0.0``
    ladder the B side uses. Dropping any of the three is not a silent slowdown but a
    silent DISARMING, since every mutation leg would then launch the shipped bytes and
    report the defect as uncaught.
    """
    if not metal_cylindrical_real_fused_hd_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    import numpy as np  # noqa: PLC0415
    from ..stepping import _boundary_kinds as resolve  # noqa: PLC0415

    spec = SUB_STEPS[CURL_SUB_STEP]
    side = CONSTITUTIVE_SIDES[CONSTITUTIVE_SIDE]
    grid = fields.grid
    codes = boundary_codes(resolve(grid, pml))
    shape = tuple(int(n) for n in grid.shape)
    scan_shape = (scan_rows(CURL_SUB_STEP, shape[0]), shape[1], shape[2])

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    # THE ROTATING GROUP FIRST, in ROTATED_NAMES order: every scratch, then every
    # pre-launch buffer, resolved per launch by `resolve`. The twins are registered
    # here and never bound by position.
    twins: Dict[str, Any] = {}
    for name in ROTATED_NAMES:
        host = getattr(fields, name)
        bind(name, host)
        twin, _mirror = _weld.scratch_twin(residency, name, host)
        twins[name] = twin

    # ONE SUB-LATTICE, TWO HALVES: `SUB_STEPS['step_D']['suffix']` is '' and
    # `CONSTITUTIVE_SIDES['H']['half_integer']` is False, so the curl's kms and the
    # constitutive's kms are the SAME three volumes. Both launches bind them; the
    # equality is ASSERTED because binding the half-integer set instead is a smooth,
    # half-cell-wrong absorber rather than a failure.
    suffix = spec["suffix"]
    constitutive_suffix = "_h" if side["half_integer"] else ""
    if suffix != constitutive_suffix:
        raise AssertionError(
            f"step_D reads the {suffix or 'integer'!r} PML sub-lattice and update_H "
            f"the {constitutive_suffix or 'integer'!r} one; this product binds one kms "
            f"group to both launches and that is only correct while they agree")

    flux = [bind(name, getattr(fields, name)) for name in side["sources"]]
    kps = [bind(f"pml:kps_{axis}{constitutive_suffix}",
                getattr(pml, f"kps_{axis}{constitutive_suffix}"), constant=True)
           for axis in "xyz"]
    kms = [bind(f"pml:kms_{axis}{suffix}", getattr(pml, f"kms_{axis}{suffix}"),
                constant=True) for axis in "xyz"]

    # THE PREFIX MIRROR IS PLAN-OWNED SCRATCH, registered NON-CONSTANT (the device
    # writes it) through `scratch_mirror`, which REUSES the array already bound under
    # the name so this plan and the certified step_D curl plan may share one.
    prefix_host, prefix = scratch_mirror(
        residency, f"cyl_pfx:{CURL_SUB_STEP}",
        lambda: np.zeros(scan_shape, dtype=np.float32))
    volumes.append(f"cyl_pfx:{CURL_SUB_STEP}")
    if scan_vectors is None:
        _, weight_vector = scratch_mirror(
            residency, f"cyl_weights:{CURL_SUB_STEP}",
            lambda: prefix_row_vectors(CURL_SUB_STEP, shape[0])[0], constant=True)
        _, divisor_vector = scratch_mirror(
            residency, f"cyl_divisor:{CURL_SUB_STEP}",
            lambda: prefix_row_vectors(CURL_SUB_STEP, shape[0])[1], constant=True)
    else:
        weight_vector = residency.mirror(f"mutation:cyl_weights:{CURL_SUB_STEP}",
                                         np.ascontiguousarray(scan_vectors[0]),
                                         constant=True)
        divisor_vector = residency.mirror(f"mutation:cyl_divisor:{CURL_SUB_STEP}",
                                          np.ascontiguousarray(scan_vectors[1]),
                                          constant=True)

    displacement = [bind(name, getattr(fields, name)) for name in spec["targets"]]
    auxiliary = [bind("fu_" + name, getattr(fields, "fu_" + name))
                 for name in spec["targets"]]
    curl_coefficients = [
        bind(f"pml:{stem}_{axis}{suffix}", getattr(pml, f"{stem}_{axis}{suffix}"),
             constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]

    dtdx = grid.dt / grid.dx
    # stepping.py:586 forms `4.0 * (grid.dt / grid.dx)` in float64 and NumPy rounds
    # that ONE scalar to float32 before the multiply -- the D side's axis post-add.
    axis_coef = 4.0 * dtdx
    n_elem = shape[0] * shape[1] * shape[2]

    constitutive_static = (flux + kps + kms + [prefix, weight_vector, divisor_vector]
                           + [_params_tensor(shape, residency.device)])
    if 2 * len(ROTATED_NAMES) + len(constitutive_static) != CONSTITUTIVE_BINDINGS:
        raise AssertionError((2 * len(ROTATED_NAMES) + len(constitutive_static),
                              CONSTITUTIVE_BINDINGS))  # pragma: no cover - arithmetic
    curl_static = (displacement + auxiliary + [prefix] + curl_coefficients
                   + [shape[0], shape[1], shape[2], n_elem, float(dtdx),
                      float(axis_coef)])
    if len(curl_static) + 3 != CURL_BINDINGS:  # pragma: no cover - arithmetic guard
        raise AssertionError((len(curl_static) + 3, CURL_BINDINGS))

    return MetalCylindricalRealFusedHdPairPlan(
        shape, scan_shape, dtdx, axis_coef, codes, residency, fields, twins,
        prefix_host, constitutive_static, curl_static,
        dict(constitutive_function) if constitutive_function is not None
        else _constitutive_functions(contract_variants),
        dict(curl_function) if curl_function is not None
        else _curl_functions(codes, contract_variants),
        volumes)


# ---------------------------------------------------------------------------
# Registration -- wired=False, no absorb row, not in registry.FAMILY_MODULES yet
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"cylindrical fused H/D pair cannot fill {slot}",))
    return metal_cylindrical_real_fused_hd_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalCylindricalRealFusedHdPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_cylindrical_real_fused_hd_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``update_H``, ``wired=False``.

    REGISTERED SO IT IS ENUMERABLE, NOT SO IT IS SELECTABLE: ``arms.arms_for`` skips an
    unwired row so ``plan_step`` cannot select it, while ``arms.registered`` still
    returns it for the disjointness sweep. The seam loop reaches it only through a
    ``launch.FUSED_PAIR_ARMS`` row it does not hold yet (lanes/cyl_round/metal_m0/
    wiring.patch), and would then refuse it on :data:`INSTALLABLE` and on arbitration.
    """
    from . import arms  # noqa: PLC0415

    return (arms.register(FAMILY, SLOT, "cylindrical m=0 fused H/D pair",
                          _arm_coverage, _arm_plan,
                          prefix="cylindrical fused H/D pair: ",
                          noun="cylindrical m = 0 fused stored-H/PML D-curl pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
