"""Folded off-diagonal dispersive ``update_E`` welded to ``update_P`` — the
Triton E->P product for the fold + tensor-epsilon + dispersion intersection.

THE CELL, BY NAME. One corpus row reaches the E->P seam on the folded
off-diagonal dispersive ``update_E`` body: ``examples:absorbed_power_density.py``
(a mirror plane, one surviving ``chi1inv`` row, one Lorentz pole, an active
absorber, boundary kinds ``[periodic, mirror, periodic]``). Its E half is
:mod:`.folded_offdiag_dispersive_update_e` (the census label ``"folded
off-diagonal dispersive"``) and its ADE half is ``kernels.ade_update_p`` (the
census label ``"ADE update_P"``); :mod:`.fused_ade_chain` refuses that row on all
three of its arms — the folded arm by the row ("an off-diagonal chi1inv row is
installed"), the dispersive arm by the fold, the no-PML arm by the absorber — and
:mod:`.complex_fused_ade_chain` refuses it by storage. This module is the one
product that spans that cell, and it is the LAST buildable, unbuilt seam-instance
in the Triton column of the fusion board.

===========================================================================
THE SEAM CARRIES NOTHING THE RECURRENCE READS, AND THAT IS WHY THIS PRODUCT EXISTS
===========================================================================

``FdtdDriver.step`` runs::

    if fast is None or not fast.dispatch("update_E", self.fields):
        update_E(self.fields, self.pml)
    f_minus_p_end(self.fields, f_minus_p_D)
    if fast is None or not fast.dispatch("update_P", self.fields):
        update_P(self.fields, self.pml)

The one statement between them is the electric ``f_minus_p`` restore
(``sources.f_minus_p_end``): it puts back the D words an integrated electric source
replaced and re-runs D's boundary passes, so it writes D (and D's ghost and wall cells)
ALONE. ``update_P`` reads the polarizations, sigma and MEEP's drive ``w`` (the stored E
or ``f_w``, ``Fields.drive_field``) and never D, so a launch that forms E and then P
before the restore leaves every word the driver's order leaves: no injection, no mirror
fill of E, no wall clear of E, no far-ghost pass of E sits in the seam. Every other seam
on this board has a driver pass inside it that a fused kernel must carry inline or refuse; this one has none, which is why the
fusion matrices record ``E->P: 15 instances, 0 blocked by the source seam,
CEILING 15``.

===========================================================================
THE SHAPE: ONE LAUNCH, THE CERTIFIED BODY LIFTED CONTIGUOUSLY, THE ARMS APPENDED
===========================================================================

The kernel is :func:`folded_offdiag_dispersive_update_e.folded_offdiag_dispersive_
constitutive_step`'s body — every statement of it, in order, as one contiguous
span — followed by the ADE recurrence in :mod:`.fused_ade_chain`'s shape: one
arm per (component, pole slot), each writing that susceptibility's scratch FOR
THAT COMPONENT. The certified body is not paraphrased and not re-derived: the
laptop test parses both files and requires the certified function's statement
list to be an AST-identical PREFIX of this kernel's, and the two halves of the
signature (the runtime names, the ``constexpr`` names) each to be a prefix of the
fused lists. What this module adds to the certified text is declared in
:data:`LIFT_EDITS`, six entries, and nothing else.

WHY THE E HALF IS NOT SPLIT PER COMPONENT. Every certified Triton ``update_E``
writes all three components in one launch, and this body more than any: each
partner term gathers ``D - sum(P)`` at FOUR neighbour addresses through
``_folded_dispersive_term``, on a folded axis including interior stored row 2.
Splitting the body per component to interleave with the recurrence would put the
arithmetic outside what was gated, and would also be the exact hazard the
hand-CUDA polarization-pair weld (released 2026-09-02) refused for ITS
off-diagonal dispersive body: an off-diagonal row product must never read a
rotated P slot. Here no rotation happens inside the launch at all — the ADE
outputs are scratches, every P is read through the pointer the E half bound, and
the host rotates AFTER the launch returns.

WHY ONE SCRATCH PER DRIVEN COMPONENT. The orbit enumeration that licenses the
shape (``parity/meep_gpu/probe_triton_ade_rotation_shape.py``, LEG 2) is about
the buffer permutation ``dispersion.PolarizationState.update`` performs and never
about the body that computes the values, so it transfers here unchanged: the
per-component-scratch shape is alias-free at every position of the orbit for
every ``d``, with orbit 3. The E half's extra reads — the six row volumes and
every P at neighbour offsets — are inputs this launch never writes, so the write
set ``{E, f_w, one scratch per (state, component)}`` stays disjoint from the
read set BY CONSTRUCTION, and :meth:`FoldedOffdiagFusedAdeChainPlan.run` asserts
that disjointness before every launch, with the row volumes counted among the
reads (the addition over the chain's inventory).

WHAT IT COSTS: ``K * (d - 1)`` extra float32 field volumes, exactly as the chain
pays — on the corpus row (K = 1, d = 3) two volumes. The one thing that does NOT
match the reference after a step is the retired scratch: the reference's single
``_scratch`` holds the retired ``P_prev`` of the LAST driven component, the
per-component scratches hold the retired ``P_prev`` of EACH. Dead state — the
next launch overwrites every one before reading it — and the gate compares ``P``
and ``P_prev`` per (susceptibility, component) plus the last component's
scratch, and says so.

===========================================================================
WHAT IS FUSED, AND WHY THE POLE IS RE-READ RATHER THAN REGISTER-REUSED
===========================================================================

* **the drive.** ``update_P``'s drive is ``Fields.drive_field(component)``, which
  under an active layer is ``f_w_<c>`` (fields.py:1160-1162) — the very word the
  E half of this launch stored from its ``src{c}`` register
  (``tl.store(w{c} + idx, src{c}, mask=live)``). A float32 word stored and
  reloaded from the same address is the identity on the bits, so ``w =
  tl.load(drive + idx, ...)`` becomes ``w = src{c}``: the register. The chain's
  gate measured the reload-vs-register form as a NULL, with PTX evidence;
* **the pole.** ``ade_update_p`` loads ``p_now``, and in the chain that load is
  the register the subtraction chain already named. Here the E half's own-cell
  pole loads happen INSIDE the certified JIT helper ``_load_d_minus_p`` — no
  ``pa0`` register exists in the caller — and naming them would edit certified
  helper text. So the pole is RE-READ through the once-bound pointer: ``p =
  tl.load(a0 + idx, mask=live, other=0.0)``. Nothing in this launch writes
  ``a0`` (the ADE outputs are scratches), so the re-read sees the pre-step P,
  which is what ``p_now`` is; the chain's gate measured reload-vs-register on the
  pole as a null too (its m3). ``p_now`` is therefore NOT a parameter: each pole
  pointer is bound exactly once and both halves read through it.

EVERY OTHER LINE IS TRANSCRIBED FROM THE BODY IT REPLACES, and the transcription
is CHECKED BY PARSING both sources rather than restated as data here.
``test_triton_folded_offdiag_fused_ade_chain`` reads the certified kernels' own
text with ``ast`` and requires the E body to be a statement-prefix and each ADE
arm to be ``ade_update_p``'s lines modulo the declared renames; the device gate
re-runs that check beside the bytes.

===========================================================================
SCOPE, AND WHY EACH BOUND IS A REFUSAL BY NAME
===========================================================================

* **the E half's own admission, whole.** Real float32 storage, an active PML, a
  real fold, at least one surviving off-diagonal row, at least one registered
  supported susceptibility, stored E, ``P``/``P_prev`` per driven component, at
  most :data:`MAX_POLES` poles per component — every one of them is
  :func:`folded_offdiag_dispersive_update_e.folded_offdiag_dispersive_constitutive_
  coverage`'s clause, taken in full and prefixed ``"E half: "``, never weakened;
* **the ADE half's own admission, per (state, component).** The PML arm only
  (``coverage.ade_update_p_coverage``), because the E half requires an active
  absorber and the drive under one is ``f_w``;
* **at most :data:`CHAIN_MAX_POLES` poles on one component.** TWO: the cell's
  corpus maximum is one, and a second live slot is what keeps the slot-1 kernel
  mutations armed on a case that can see them. A third pole is refused BY NAME
  rather than truncated — the E chain would subtract it and the ADE half would
  never advance it, a frozen P, smooth and wrong — and the plan raises on it too;
* **the seam's own clauses**, named rather than inherited: something is driven;
  ``drive_field(c)`` IS ``f_w_<c>`` for every component (the register hand-off
  rests on it); every recurrence volume carries exactly the extent the E half's
  one bounds guard walks (the ADE guard is dropped); each state's ``driven()``
  agrees with the components whose ``D - sum P`` chain subtracts it; and each
  subtraction slot's array is the state's own ``P``.

NOT WIRED, and dispatch is unchanged. ``launch.plan_step`` does not import this
module and no arm table names it, exactly as the three other E->P chain products
are held: :data:`INSTALLABLE` is ``False`` and :data:`INSTALLABLE_REASON` names
the slot protocol the composer would have to grow. The product exists to be
MEASURED — ``parity/meep_gpu/gate_triton_folded_offdiag_fused_ade_chain.py`` is
its byte gate — and a default run still steps this seam on the array path.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import folded_offdiag_dispersive_update_e as _fod
from . import folded_offdiag_update_e as _folded
from . import offdiag_update_e as _offdiag
from . import symmetry as _symmetry
from .coverage import Coverage
from .dispersive_update_e import E_TERMS, LivePoleBinding, poles_per_component

__all__ = [
    "CARRIES_DEPOSIT_REPAIR",
    "CELL",
    "CHAIN_MAX_POLES",
    "E_TERMS",
    "FAMILY",
    "INSTALLABLE",
    "INSTALLABLE_REASON",
    "KERNEL_NAME",
    "LETTERS",
    "LIFT_EDITS",
    "MAX_POLES",
    "POLICY",
    "REPLACES",
    "RUNTIME_ARGUMENTS",
    "FoldedOffdiagFusedAdeChainPlan",
    "explain_folded_offdiag_fused_ade_chain",
    "folded_offdiag_fused_ade_chain_coverage",
    "folded_offdiag_fused_ade_chain_kernel",
    "folded_offdiag_fused_ade_chain_step",
    "plan_folded_offdiag_fused_ade_chain",
]

#: The family name — the board's product key and the ledger key's stem.
FAMILY = "folded_offdiag_fused_ade_chain"

#: The kernel's entry-point name. ITS OWN, not the certified body's: a second
#: kernel wearing the certified one's name would make the JIT cache and the
#: fingerprint partition ambiguous about which body they saw (LIFT_EDITS E1).
KERNEL_NAME = "folded_offdiag_fused_ade_chain_step"

#: The driver passes one run of this plan performs, in driver order
#: (FdtdDriver.step). Declared rather than inferred from a slot name.
REPLACES: Tuple[str, ...] = ("update_E", "update_P")

#: The census cell this product spans: the E-half arm label the planner writes
#: at ``update_E`` (launch.py, the ``"folded off-diagonal dispersive"`` arm) and
#: the ADE label it writes at ``update_P``. Data, so the board and the battery
#: cannot disagree with this module about which cell is claimed.
CELL: Tuple[str, str] = ("folded off-diagonal dispersive", "ADE update_P")

#: The kernel's per-component argument prefix, in signature order.
LETTERS: Tuple[str, str, str] = ("a", "b", "c")

#: The E half's compiled pole slots. EIGHT, because that is what the certified
#: body compiles (``folded_offdiag_dispersive_update_e.MAX_POLES``, which is
#: ``dispersive_update_e.MAX_POLES``) and the subtraction chain its helper
#: unrolls is that chain — pinned equal to both by the laptop test rather than
#: imported, so that module's edit cannot silently truncate this kernel's chain.
MAX_POLES = 8

#: The ADE half's live slots per component, and the bound the predicate refuses
#: past. TWO: the cell's corpus maximum is ONE (``absorbed_power_density.py``
#: registers a single Lorentz pole), and a second slot is kept live so the
#: slot-1 kernel mutations are armed on a case that can see them — the same
#: arity envelope the hand-CUDA polarization-pair weld's digests walk. Every
#: extra slot costs SIX more kernel arguments per component. A third pole is
#: refused BY NAME rather than truncated: the E chain would subtract it and the
#: ADE half would never advance it.
CHAIN_MAX_POLES = 2

#: The settled launch configuration. ONE WARP is the shipped policy for every
#: cross-sub-step fused product (``fused_dispersive_chain.POLICY``, measured
#: 1.0739x median over 12 shape/boundary cases). ``enable_fp_fusion`` is off
#: engine-wide (``kernels.ENABLE_FP_FUSION``) and is what keeps this seam from
#: contracting a multiply and an add into an FMA the reference did not take.
#: This product does not tune.
POLICY: Dict[str, Any] = {
    "num_warps": 1,
    "block": "kernels.DEFAULT_BLOCK",
    "enable_fp_fusion": "kernels.ENABLE_FP_FUSION (False)",
    "status": "keep",
}

#: Nothing is INJECTED between the two driver passes this product spans, so
#: there is no deposit to repair and no repair path to consult.
CARRIES_DEPOSIT_REPAIR = False

#: THE COMPOSITION IS REFUSED, ON EVERY ROW, FOR A MEASURED REASON OF ITS OWN.
#: ``launch._declared_uninstallable`` reads this flag off the module if the
#: product is ever tabled; until then it is a declaration the laptop test pins.
INSTALLABLE = False
INSTALLABLE_REASON = (
    "an E->P product: launch.py's pair seam loop (CERTIFIED_FUSED_PAIR_SEAMS) "
    "has no update_E -> update_P row, the update_P slot holds a LIST of "
    "per-susceptibility plans and update_E is a constitutive slot rather than a "
    "curl, a label plan_step could write must also be declared in the "
    "dispatcher's own tables outside this package, and no composition gate has "
    "driven this product end to end through plan_step")

#: Runtime (non-constexpr) parameters :func:`folded_offdiag_fused_ade_chain_step`
#: declares: the certified body's fifty-seven (three targets, three auxiliaries,
#: three sources, three inverse epsilons, twenty-four pole slots, six row
#: slots, six PML coefficient columns, three ghost weights, the chunk record, the
#: row presence, four geometry scalars) followed by seven ADE arguments per slot
#: per component (output, history, sigma and MEEP's four update_P words). Asserted
#: against the kernel's OWN signature by the laptop test and against the
#: assembled argument list at EVERY launch, so a signature edit that the
#: launcher does not follow fails at the call rather than binding the wrong
#: pointer to the wrong slot.
RUNTIME_ARGUMENTS = (3 + 3 + 3 + 3 + 3 * MAX_POLES + 6 + 6 + 3 + 2 + 4
                     + 3 * 7 * CHAIN_MAX_POLES)

#: THE SIX EDITS THIS MODULE MAKES TO THE TEXT IT LIFTS, and nothing else. Each
#: ``line`` is a needle in the certified source it came from (the folded
#: off-diagonal dispersive ``update_E`` body or ``kernels.ade_update_p``) and
#: each ``became`` a needle in this kernel's text, or a parenthesised note where
#: the edit is an elision. The laptop test checks every needle; the device gate
#: re-checks them beside the bytes.
LIFT_EDITS: Tuple[Dict[str, str], ...] = (
    {"line": "def folded_offdiag_dispersive_constitutive_step(",
     "became": "def folded_offdiag_fused_ade_chain_step(",
     "why": "THE ENTRY POINT'S NAME, and nothing else on that line. The JIT "
            "cache and the fingerprint partition must never see two kernels "
            "under one name, so the lifted body is compiled under its own."},
    {"line": "nx, ny, nz, n_elem,\n        NP0: tl.constexpr,",
     "became": "nx, ny, nz, n_elem,\n        p_out_a0,",
     "why": "THE SIGNATURE SPLICE. The ADE runtime block (p_out, p_prev, sigma "
            "and MEEP's four update_P words per slot per component) is spliced AFTER "
            "n_elem and the SV_* constexprs AFTER BLOCK, so the certified "
            "runtime names and the certified constexpr names are each a PREFIX "
            "of the fused lists. No arithmetic lives in a signature, and p_now "
            "is ABSENT: each pole pointer a{k}/b{k}/c{k} is bound exactly once "
            "and both halves read through it."},
    {"line": "idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
             "    live = idx < n_elem",
     "became": "(dropped: the certified E body's first two statements are the "
               "identical text; n_elem is the same number and the mask the same "
               "live, and the ADE arms are appended AFTER component 2's store "
               "so the certified body stays one contiguous verbatim span)",
     "why": "ade_update_p's own index and bounds guard would redeclare idx and "
            "live. The predicate requires every P/P_prev/scratch/sigma volume "
            "to carry exactly the stored extent the E half's guard walks, so "
            "the dropped bound is the same number the certified kernel used."},
    {"line": "w = tl.load(drive + idx, mask=live, other=0.0)",
     "became": "w = src0",
     "why": "THE SEAM. update_P's drive is fields.drive_field(c), MEEP's w per "
            "chunk: f_w_<c> on the layer's chunks, the stored E elsewhere -- the "
            "words this launch just stored from this register (into f_w on the "
            "accumulation's cells, as the field everywhere else), and a float32 "
            "word stored and reloaded is the identity on the bits. One register "
            "per component: w = src0, src1, src2."},
    {"line": "p = tl.load(p_now + idx, mask=live, other=0.0)",
     "became": "p = tl.load(a0 + idx, mask=live, other=0.0)",
     "why": "A RE-READ THROUGH THE ONCE-BOUND POINTER, not the chain's register "
            "reuse: the E half's own-cell pole loads live inside the certified "
            "JIT helper _load_d_minus_p and exposing them would edit certified "
            "helper text. Nothing in this launch writes a{k}/b{k}/c{k} (the ADE "
            "outputs are scratches), so the re-read sees the pre-step P that "
            "p_now is; the chain gate measured reload-vs-register as a null."},
    {"line": "tl.store(p_out + idx,\n"
             "             gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + "
             "(omega0dtsqr * (s * w))),\n             mask=live)",
     "became": "tl.store(p_out_a0 + idx,\n"
               "                     gamma1inv * (((p * two_minus_od) - (gamma1 * q)) "
               "+ (omega0dtsqr * (s * w))),\n                     mask=live)",
     "why": "THE PER-SLOT RENAMES. p_out/p_prev/sigma become p_out_{L}{k}/"
            "p_prev_{L}{k}/sigma_{L}{k}, MEEP's four words are rebound as "
            "two_minus_od = two_minus_od_{L}{k} (and gamma1, omega0dtsqr, "
            "gamma1inv) so the arithmetic line is character for character "
            "ade_update_p's, the SIGMA_IS_VOLUME arm becomes if SV_{L}{k}:, and "
            "each arm sits under if NP{c} > k: -- one arm per (component, slot), "
            "the chain's spelling."},
)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# Defined at MODULE scope behind a conditional import, not inside a builder:
# `@triton.jit` resolves a body's names — including the `tl.constexpr` annotations
# and every called JIT helper — through the defining module's `__globals__`, so a
# kernel defined inside a function compiles to `NameError('tl is not defined')`
# at first launch (dispersive_update_e.py:138-143 records the run that showed
# it, 320 of 320). The six names the lifted body calls or compares against are
# therefore bound HERE, as direct globals, to the certified modules' own objects.

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the byte gate
    PERIODIC = tl.constexpr(_fod.CODE_PERIODIC)
    METALLIC = tl.constexpr(_fod.CODE_METALLIC)
    MIRROR_ROW = tl.constexpr(_fod.MIRROR_SOURCE_INDEX)
    PRESENCE_LINES = tl.constexpr(_offdiag.PRESENCE_MASK_WORDS)
    # Direct global bindings, the mechanism the E half itself uses: a called JIT
    # function is resolved from the defining module's globals, and a module-
    # attribute call is not a frontend contract this package asks Triton to carry.
    _load_d_minus_p = _fod._load_d_minus_p
    _folded_dispersive_term = _fod._folded_dispersive_term
    from .forms import constitutive_form, record_in  # noqa: E402
    from .kernels import RECORD_HALF_INTEGER  # noqa: E402
    from .offdiag_update_e import _row_add  # noqa: E402

    @triton.jit
    def folded_offdiag_fused_ade_chain_step(
        f0, f1, f2,                     # targets:      Ex, Ey, Ez             (in/out)
        w0, w1, w2,                     # auxiliaries:  f_w_Ex, f_w_Ey, f_w_Ez (in/out)
        g0, g1, g2,                     # sources:      Dx, Dy, Dz             (read-only)
        e0, e1, e2,                     # inverse-epsilon VOLUMES (three distinct or aliased)
        a0, a1, a2, a3, a4, a5, a6, a7,   # component 0's poles, IN ORDER
        b0, b1, b2, b3, b4, b5, b6, b7,   # component 1's poles, IN ORDER
        c0, c1, c2, c3, c4, c5, c6, c7,   # component 2's poles, IN ORDER
        u01, u02,                       # row Ex: partner Ey (down y), partner Ez (down z)
        u11, u12,                       # row Ey: partner Ez (down z), partner Ex (down x)
        u21, u22,                       # row Ez: partner Ex (down x), partner Ey (down y)
        kp0, km0, kp1, km1, kp2, km2,   # kps/kms on each component's OWN axis, half-integer
        gwx, gwy, gwz,                  # RUNTIME mirror ghost weights, exactly +1.0 / -1.0
        cb,                             # chunk record: each component's dsigw membership
        op,                             # off-diagonal presence: slot masks, segment lines
        nx, ny, nz, n_elem,
        p_out_a0, p_out_a1, p_prev_a0, p_prev_a1, sigma_a0, sigma_a1,
        two_minus_od_a0, two_minus_od_a1, gamma1_a0, gamma1_a1,
        omega0dtsqr_a0, omega0dtsqr_a1, gamma1inv_a0, gamma1inv_a1,
        p_out_b0, p_out_b1, p_prev_b0, p_prev_b1, sigma_b0, sigma_b1,
        two_minus_od_b0, two_minus_od_b1, gamma1_b0, gamma1_b1,
        omega0dtsqr_b0, omega0dtsqr_b1, gamma1inv_b0, gamma1inv_b1,
        p_out_c0, p_out_c1, p_prev_c0, p_prev_c1, sigma_c0, sigma_c1,
        two_minus_od_c0, two_minus_od_c1, gamma1_c0, gamma1_c1,
        omega0dtsqr_c0, omega0dtsqr_c1, gamma1inv_c0, gamma1inv_c1,
        NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
        R01: tl.constexpr, R02: tl.constexpr,
        R11: tl.constexpr, R12: tl.constexpr,
        R21: tl.constexpr, R22: tl.constexpr,
        P01: tl.constexpr, P02: tl.constexpr,
        P11: tl.constexpr, P12: tl.constexpr,
        P21: tl.constexpr, P22: tl.constexpr,
        PRESENCE: tl.constexpr,
        BCX: tl.constexpr, BCY: tl.constexpr, BCZ: tl.constexpr,
        MG_X: tl.constexpr, MG_Y: tl.constexpr, MG_Z: tl.constexpr,
        WM_X: tl.constexpr, WM_Y: tl.constexpr, WM_Z: tl.constexpr,
        BLOCK: tl.constexpr,
        SV_a0: tl.constexpr, SV_a1: tl.constexpr,
        SV_b0: tl.constexpr, SV_b1: tl.constexpr,
        SV_c0: tl.constexpr, SV_c1: tl.constexpr,
    ):
        """Folded off-diagonal dispersive ``update_E`` welded to ``update_P``:
        both driver passes, ONE grid.

        THE E HALF IS THE CERTIFIED BODY, CONTIGUOUS AND VERBATIM:
        ``folded_offdiag_dispersive_update_e.folded_offdiag_dispersive_
        constitutive_step``, every statement in order, calling the same JIT helpers
        (``_load_d_minus_p``, ``_folded_dispersive_term``, ``offdiag_update_e.
        _row_add``, ``forms.record_in``, ``forms.constitutive_form``) and comparing
        against the same constexprs (``PERIODIC``, ``METALLIC``, ``MIRROR_ROW``,
        ``PRESENCE_LINES``, ``RECORD_HALF_INTEGER``), all bound at this module's scope
        to the certified modules' own objects: MEEP's row in double with per-chunk
        presence, and the per-chunk constitutive form keyed on the record ``cb``. The
        ordered ``D - P0 - P1 - ...`` subtraction happens INSIDE ``_load_d_minus_p``
        at every gathered address, which is why no pole register is visible here.

        THE ADE HALF IS ``kernels.ade_update_p``, MEEP's expression on the four
        realnum words, one arm per (component, pole slot), appended AFTER component
        2's store so the certified span is unbroken. Exactly two of its loads are
        replaced:

        * ``w = tl.load(drive + idx, ...)`` becomes ``w = src{c}`` — MEEP's ``w`` on
          every cell: the E half stored it into ``f_w_<c>`` on the layer's chunks and
          as the field everywhere else, which is what ``Fields.drive_field`` composes;
        * ``p = tl.load(p_now + idx, ...)`` becomes ``p = tl.load({L}{k} + idx,
          ...)`` — the SAME volume through the pointer the E half bound, re-read
          rather than register-reused because the E half's own-cell pole load is
          inside the helper. Nothing in this launch writes that pointer.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        nyz = ny * nz
        k = idx % nz
        plane = idx // nz
        j = plane % ny
        i = plane // ny

        at_x, at_y, at_z = i == 0, j == 0, k == 0

        # --- per-axis neighbour indices, both directions, with the ghost rule.
        # stepping._shift_down (:1787) / _shift_up (:1723): PERIODIC wraps;
        # METALLIC masks both faces; MIRROR redirects the DOWN ghost to stored
        # row MIRROR_ROW with the plane's weight and masks the UP ghost.
        di, dj, dk = i - 1, j - 1, k - 1
        ui, uj, uk = i + 1, j + 1, k + 1
        dvx, dvy, dvz = live, live, live
        uvx, uvy, uvz = live, live, live
        wx, wy, wz = 1.0, 1.0, 1.0
        if BCX == PERIODIC:
            di = tl.where(di < 0, nx - 1, di)
            ui = tl.where(ui == nx, 0, ui)
        elif BCX == METALLIC:
            dvx = live & (di >= 0)
            uvx = live & (ui < nx)
        else:  # MIRROR_METALLIC or MIRROR_PERIODIC
            di = tl.where(at_x, MIRROR_ROW, di)
            wx = tl.where(at_x, gwx, 1.0)
            uvx = live & (ui < nx)
        if BCY == PERIODIC:
            dj = tl.where(dj < 0, ny - 1, dj)
            uj = tl.where(uj == ny, 0, uj)
        elif BCY == METALLIC:
            dvy = live & (dj >= 0)
            uvy = live & (uj < ny)
        else:
            dj = tl.where(at_y, MIRROR_ROW, dj)
            wy = tl.where(at_y, gwy, 1.0)
            uvy = live & (uj < ny)
        if BCZ == PERIODIC:
            dk = tl.where(dk < 0, nz - 1, dk)
            uk = tl.where(uk == nz, 0, uk)
        elif BCZ == METALLIC:
            dvz = live & (dk >= 0)
            uvz = live & (uk < nz)
        else:
            dk = tl.where(at_z, MIRROR_ROW, dk)
            wz = tl.where(at_z, gwz, 1.0)
            uvz = live & (uk < nz)

        # Which chunk of the {low, interior, high} product owns the cell, per
        # component lattice (presence_record's segment lines: integer x, y, z,
        # then half-integer x, y, z). Read only when some slot's presence varies.
        cell0, cell1, cell2 = i * 0, i * 0, i * 0
        if PRESENCE:
            sxi = tl.load(op + PRESENCE_LINES + i, mask=live, other=0)
            syi = tl.load(op + PRESENCE_LINES + nx + j, mask=live, other=0)
            szi = tl.load(op + PRESENCE_LINES + nx + ny + k, mask=live, other=0)
            half = PRESENCE_LINES + nx + ny + nz
            sxh = tl.load(op + half + i, mask=live, other=0)
            syh = tl.load(op + half + nx + j, mask=live, other=0)
            szh = tl.load(op + half + nx + ny + k, mask=live, other=0)
            cell0 = (sxh * 9 + syi * 3) + szi     # Ex: half-integer on x
            cell1 = (sxi * 9 + syh * 3) + szi     # Ey: half-integer on y
            cell2 = (sxi * 9 + syi * 3) + szh     # Ez: half-integer on z

        # Which cells take the accumulation: each component's own (dsigw) layer on
        # the half-integer sub-lattice (chunk_map.record_index(True, axis, 0)).
        dw0 = record_in(cb, RECORD_HALF_INTEGER + 0, i)
        dw1 = record_in(cb, RECORD_HALF_INTEGER + 4, j)
        dw2 = record_in(cb, RECORD_HALF_INTEGER + 8, k)

        # Component 0 takes its coefficient from axis x, 1 from y, 2 from z.
        kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
        km_0 = tl.load(km0 + i, mask=live, other=0.0)
        kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
        km_1 = tl.load(km1 + j, mask=live, other=0.0)
        kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
        km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0: Ex — own axis x; partners Dy (down y) then Dz (down z)
        prev0 = tl.load(w0 + idx, mask=live & dw0, other=0.0)   # BEFORE the store.
        gs0 = _load_d_minus_p(
            g0, a0, a1, a2, a3, a4, a5, a6, a7, idx, live, NP0)
        us0 = tl.load(e0 + idx, mask=live, other=0.0)
        if R01:
            row0 = tl.cast(gs0 * us0, tl.float64)
            row0 = _row_add(row0, _folded_dispersive_term(
                g1, b0, b1, b2, b3, b4, b5, b6, b7, u01, idx,
                i * nyz + dj * nz + k,
                ui * nyz + j * nz + k,
                ui * nyz + dj * nz + k,
                live, dvy, uvx, uvx & dvy, wy, NP1, MG_Y),
                at_y, at_z, op, cell0, 0, WM_Y, WM_Z, P01)
            if R02:
                row0 = _row_add(row0, _folded_dispersive_term(
                    g2, c0, c1, c2, c3, c4, c5, c6, c7, u02, idx,
                    i * nyz + j * nz + dk,
                    ui * nyz + j * nz + k,
                    ui * nyz + j * nz + dk,
                    live, dvz, uvx, uvx & dvz, wz, NP2, MG_Z),
                    at_y, at_z, op, cell0, 1, WM_Y, WM_Z, P02)
            src0 = tl.cast(row0, tl.float32)
        else:
            if R02:
                row0 = tl.cast(gs0 * us0, tl.float64)
                row0 = _row_add(row0, _folded_dispersive_term(
                    g2, c0, c1, c2, c3, c4, c5, c6, c7, u02, idx,
                    i * nyz + j * nz + dk,
                    ui * nyz + j * nz + k,
                    ui * nyz + j * nz + dk,
                    live, dvz, uvx, uvx & dvz, wz, NP2, MG_Z),
                    at_y, at_z, op, cell0, 1, WM_Y, WM_Z, P02)
                src0 = tl.cast(row0, tl.float32)
            else:
                src0 = gs0 * us0
        tl.store(w0 + idx, src0, mask=live & dw0)
        value0 = tl.load(f0 + idx, mask=live & dw0, other=0.0)
        value0 = constitutive_form(value0, prev0, src0, kp_0, km_0, dw0)
        tl.store(f0 + idx, value0, mask=live)

        # --- component 1: Ey — own axis y; partners Dz (down z) then Dx (down x)
        prev1 = tl.load(w1 + idx, mask=live & dw1, other=0.0)
        gs1 = _load_d_minus_p(
            g1, b0, b1, b2, b3, b4, b5, b6, b7, idx, live, NP1)
        us1 = tl.load(e1 + idx, mask=live, other=0.0)
        if R11:
            row1 = tl.cast(gs1 * us1, tl.float64)
            row1 = _row_add(row1, _folded_dispersive_term(
                g2, c0, c1, c2, c3, c4, c5, c6, c7, u11, idx,
                i * nyz + j * nz + dk,
                i * nyz + uj * nz + k,
                i * nyz + uj * nz + dk,
                live, dvz, uvy, uvy & dvz, wz, NP2, MG_Z),
                at_x, at_z, op, cell1, 2, WM_X, WM_Z, P11)
            if R12:
                row1 = _row_add(row1, _folded_dispersive_term(
                    g0, a0, a1, a2, a3, a4, a5, a6, a7, u12, idx,
                    di * nyz + j * nz + k,
                    i * nyz + uj * nz + k,
                    di * nyz + uj * nz + k,
                    live, dvx, uvy, uvy & dvx, wx, NP0, MG_X),
                    at_x, at_z, op, cell1, 3, WM_X, WM_Z, P12)
            src1 = tl.cast(row1, tl.float32)
        else:
            if R12:
                row1 = tl.cast(gs1 * us1, tl.float64)
                row1 = _row_add(row1, _folded_dispersive_term(
                    g0, a0, a1, a2, a3, a4, a5, a6, a7, u12, idx,
                    di * nyz + j * nz + k,
                    i * nyz + uj * nz + k,
                    di * nyz + uj * nz + k,
                    live, dvx, uvy, uvy & dvx, wx, NP0, MG_X),
                    at_x, at_z, op, cell1, 3, WM_X, WM_Z, P12)
                src1 = tl.cast(row1, tl.float32)
            else:
                src1 = gs1 * us1
        tl.store(w1 + idx, src1, mask=live & dw1)
        value1 = tl.load(f1 + idx, mask=live & dw1, other=0.0)
        value1 = constitutive_form(value1, prev1, src1, kp_1, km_1, dw1)
        tl.store(f1 + idx, value1, mask=live)

        # --- component 2: Ez — own axis z; partners Dx (down x) then Dy (down y)
        prev2 = tl.load(w2 + idx, mask=live & dw2, other=0.0)
        gs2 = _load_d_minus_p(
            g2, c0, c1, c2, c3, c4, c5, c6, c7, idx, live, NP2)
        us2 = tl.load(e2 + idx, mask=live, other=0.0)
        if R21:
            row2 = tl.cast(gs2 * us2, tl.float64)
            row2 = _row_add(row2, _folded_dispersive_term(
                g0, a0, a1, a2, a3, a4, a5, a6, a7, u21, idx,
                di * nyz + j * nz + k,
                i * nyz + j * nz + uk,
                di * nyz + j * nz + uk,
                live, dvx, uvz, uvz & dvx, wx, NP0, MG_X),
                at_x, at_y, op, cell2, 4, WM_X, WM_Y, P21)
            if R22:
                row2 = _row_add(row2, _folded_dispersive_term(
                    g1, b0, b1, b2, b3, b4, b5, b6, b7, u22, idx,
                    i * nyz + dj * nz + k,
                    i * nyz + j * nz + uk,
                    i * nyz + dj * nz + uk,
                    live, dvy, uvz, uvz & dvy, wy, NP1, MG_Y),
                    at_x, at_y, op, cell2, 5, WM_X, WM_Y, P22)
            src2 = tl.cast(row2, tl.float32)
        else:
            if R22:
                row2 = tl.cast(gs2 * us2, tl.float64)
                row2 = _row_add(row2, _folded_dispersive_term(
                    g1, b0, b1, b2, b3, b4, b5, b6, b7, u22, idx,
                    i * nyz + dj * nz + k,
                    i * nyz + j * nz + uk,
                    i * nyz + dj * nz + uk,
                    live, dvy, uvz, uvz & dvy, wy, NP1, MG_Y),
                    at_x, at_y, op, cell2, 5, WM_X, WM_Y, P22)
                src2 = tl.cast(row2, tl.float32)
            else:
                src2 = gs2 * us2
        tl.store(w2 + idx, src2, mask=live & dw2)
        value2 = tl.load(f2 + idx, mask=live & dw2, other=0.0)
        value2 = constitutive_form(value2, prev2, src2, kp_2, km_2, dw2)
        tl.store(f2 + idx, value2, mask=live)

        # --- component 0: update_P, one arm per pole that drives it ----
        if NP0 > 0:
            two_minus_od = two_minus_od_a0
            gamma1 = gamma1_a0
            omega0dtsqr = omega0dtsqr_a0
            gamma1inv = gamma1inv_a0
            p = tl.load(a0 + idx, mask=live, other=0.0)
            q = tl.load(p_prev_a0 + idx, mask=live, other=0.0)
            w = src0
            if SV_a0:
                s = tl.load(sigma_a0 + idx, mask=live, other=0.0)
            else:
                s = sigma_a0
            tl.store(p_out_a0 + idx,
                     gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s * w))),
                     mask=live)
        if NP0 > 1:
            two_minus_od = two_minus_od_a1
            gamma1 = gamma1_a1
            omega0dtsqr = omega0dtsqr_a1
            gamma1inv = gamma1inv_a1
            p = tl.load(a1 + idx, mask=live, other=0.0)
            q = tl.load(p_prev_a1 + idx, mask=live, other=0.0)
            w = src0
            if SV_a1:
                s = tl.load(sigma_a1 + idx, mask=live, other=0.0)
            else:
                s = sigma_a1
            tl.store(p_out_a1 + idx,
                     gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s * w))),
                     mask=live)

        # --- component 1: update_P, one arm per pole that drives it ----
        if NP1 > 0:
            two_minus_od = two_minus_od_b0
            gamma1 = gamma1_b0
            omega0dtsqr = omega0dtsqr_b0
            gamma1inv = gamma1inv_b0
            p = tl.load(b0 + idx, mask=live, other=0.0)
            q = tl.load(p_prev_b0 + idx, mask=live, other=0.0)
            w = src1
            if SV_b0:
                s = tl.load(sigma_b0 + idx, mask=live, other=0.0)
            else:
                s = sigma_b0
            tl.store(p_out_b0 + idx,
                     gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s * w))),
                     mask=live)
        if NP1 > 1:
            two_minus_od = two_minus_od_b1
            gamma1 = gamma1_b1
            omega0dtsqr = omega0dtsqr_b1
            gamma1inv = gamma1inv_b1
            p = tl.load(b1 + idx, mask=live, other=0.0)
            q = tl.load(p_prev_b1 + idx, mask=live, other=0.0)
            w = src1
            if SV_b1:
                s = tl.load(sigma_b1 + idx, mask=live, other=0.0)
            else:
                s = sigma_b1
            tl.store(p_out_b1 + idx,
                     gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s * w))),
                     mask=live)

        # --- component 2: update_P, one arm per pole that drives it ----
        if NP2 > 0:
            two_minus_od = two_minus_od_c0
            gamma1 = gamma1_c0
            omega0dtsqr = omega0dtsqr_c0
            gamma1inv = gamma1inv_c0
            p = tl.load(c0 + idx, mask=live, other=0.0)
            q = tl.load(p_prev_c0 + idx, mask=live, other=0.0)
            w = src2
            if SV_c0:
                s = tl.load(sigma_c0 + idx, mask=live, other=0.0)
            else:
                s = sigma_c0
            tl.store(p_out_c0 + idx,
                     gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s * w))),
                     mask=live)
        if NP2 > 1:
            two_minus_od = two_minus_od_c1
            gamma1 = gamma1_c1
            omega0dtsqr = omega0dtsqr_c1
            gamma1inv = gamma1inv_c1
            p = tl.load(c1 + idx, mask=live, other=0.0)
            q = tl.load(p_prev_c1 + idx, mask=live, other=0.0)
            w = src2
            if SV_c1:
                s = tl.load(sigma_c1 + idx, mask=live, other=0.0)
            else:
                s = sigma_c1
            tl.store(p_out_c1 + idx,
                     gamma1inv * (((p * two_minus_od) - (gamma1 * q)) + (omega0dtsqr * (s * w))),
                     mask=live)

else:  # pragma: no cover - the laptop path
    PERIODIC = None  # type: ignore[assignment]
    METALLIC = None  # type: ignore[assignment]
    MIRROR_ROW = None  # type: ignore[assignment]
    _load_d_minus_p = None  # type: ignore[assignment]
    _folded_dispersive_term = None  # type: ignore[assignment]
    folded_offdiag_fused_ade_chain_step = None  # type: ignore[assignment]


def folded_offdiag_fused_ade_chain_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError."""
    if folded_offdiag_fused_ade_chain_step is None:
        raise ImportError(
            "the folded off-diagonal fused E->P chain kernel needs the optional "
            "`triton` package (pip install triton). The engine runs without it; "
            f"only this fast path is unavailable. Original error: "
            f"{_TRITON_IMPORT_ERROR}")
    return folded_offdiag_fused_ade_chain_step


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
#
# A conjunction of the two halves' OWN certified predicates plus the seam clauses.
# Nothing is weakened and nothing is re-decided: a configuration either half
# refuses is refused here with that half's reasons, prefixed so a reader can tell
# which side said it. There is ONE arm — the E half requires an active absorber,
# so the ADE half is the PML predicate and the drive is f_w — and no `arm`
# parameter, because there is nothing to select between.


def folded_offdiag_fused_ade_chain_coverage(fields: Any, pml: Any) -> Coverage:
    """May ONE launch span ``update_E`` -> ``update_P`` for this configuration?

    THERE IS NO SOURCE CLAUSE, AND THE ABSENCE IS DELIBERATE. Every other fused
    product on this backend must be told the source inventory, because the driver
    injects INSIDE its seam and ignorance is not an empty set. This seam has no
    injection in it at all — ``driver.step`` runs ``update_E``, then the electric
    ``f_minus_p`` restore, which writes D alone and which ``update_P`` does not read,
    then ``update_P`` — so requiring a declaration would refuse configurations the
    driver cannot break. The fusion matrices measure
    the same fact from the other end: ``E->P: 15 instances, 0 blocked by the
    source seam``.
    """
    reasons: List[str] = []

    electric = _fod.folded_offdiag_dispersive_constitutive_coverage(fields, pml)
    if not electric.covered:
        reasons.extend(f"E half: {reason}" for reason in electric.reasons)

    states = tuple(getattr(fields, "polarizations", ()) or ())
    try:
        order = poles_per_component(fields)
    except Exception as exc:  # noqa: BLE001 - an unreadable partition is a refusal
        return Coverage(False, tuple(dict.fromkeys(
            reasons + [f"the pole partition is unreadable ({exc!r})"])))

    electric_names = {term[0] for term in E_TERMS}
    driven_pairs: List[Tuple[int, Any, str]] = []
    for index, state in enumerate(states):
        driven = _coverage._call(state, "driven", default=None)
        if driven is None:
            reasons.append(
                f"polarization {index} ({type(state).__name__}) does not report "
                f"driven(); an unreadable susceptibility is not a covered one")
            continue
        for component in tuple(driven):
            verdict = _coverage.ade_update_p_coverage(fields, state, component)
            if not verdict.covered:
                reasons.extend(f"ADE half {index}.{component}: {reason}"
                               for reason in verdict.reasons)
            if component in electric_names:
                driven_pairs.append((index, state, component))
        # THE TWO HALVES MUST AGREE ABOUT WHICH COMPONENTS THIS STATE DRIVES.
        # The E chain subtracts wherever the pole partition lists this state and
        # the ADE half advances wherever `driven()` names it; the two are one
        # attribute in the reference (dispersion.py:652-656) and a disagreement
        # would subtract a pole this launch never advances — a frozen P, smooth
        # and wrong.
        subtracted = tuple(component for component, _displacement, _axis in E_TERMS
                           if any(state is other
                                  for other in order.get(component, ())))
        named = tuple(name for name in tuple(driven) if name in electric_names)
        if named != subtracted:
            reasons.append(
                f"polarization {index}: driven()={tuple(driven)!r} disagrees with "
                f"the components whose D - sum P chain subtracts it ({subtracted!r})")

    # SOMETHING MUST BE DRIVEN. With no recurrence to weld this product would
    # span one sub-step, not two, and that configuration is the E half's own.
    if not driven_pairs:
        reasons.append(
            "no polarization drives any electric component, so there is no "
            "recurrence to weld: that configuration is the certified folded "
            "off-diagonal dispersive update_E alone")

    for component, _displacement, _axis in E_TERMS:
        count = len(order.get(component, ()))
        if count > CHAIN_MAX_POLES:
            reasons.append(
                f"{component} is driven by {count} poles; this product binds "
                f"{CHAIN_MAX_POLES} ADE arms per component, and a pole the E chain "
                f"subtracts but the ADE half never advances is a frozen P")
        for slot, state in enumerate(order.get(component, ())):
            pole = (getattr(state, "P", {}) or {}).get(component)
            if pole is None:
                continue  # already reported by the ADE half
            if (getattr(state, "P", {}) or {}).get(component) is not pole:
                reasons.append(
                    f"{component} slot {slot}: the subtracted pole volume is not "
                    f"the state's own P; the pointer the ADE arm re-reads would "
                    f"be a different array")

    # THE SEAM'S OWN CONDITION, named rather than inherited: the register this
    # kernel hands the recurrence is ``src``, which is MEEP's ``w`` on every cell
    # (update_pols.cpp:44, ``w = f_w ? f_w : f`` per chunk): the E half stores it
    # into f_w on the layer's chunks and as the field everywhere else. That is what
    # ``Fields.drive_field`` composes, so the reader must be that method and no
    # other; a Fields whose reader is replaced is refused BY NAME.
    if not bool(getattr(fields, "_pml_active", False)):
        reasons.append(
            "Fields is not in PML storage mode: drive_field would return the "
            "stored E everywhere (fields.py drive_field), and this kernel's PML "
            "tail writes f_w on the layer's chunks")
    reader = getattr(fields, "drive_field", None)
    own = getattr(type(fields), "drive_field", None)
    if not callable(reader):
        reasons.append("fields does not expose drive_field(); an unreadable "
                       "drive is not a covered one")
    elif own is None or getattr(reader, "__func__", None) is not own:
        reasons.append(
            "drive_field is not Fields.drive_field, MEEP's per-chunk w; this "
            "kernel's register is the w that method composes (f_w on the "
            "layer's chunks, the stored E elsewhere), so another reader would "
            "drive the polarization from a different field")

    # THE EXTENT THE DROPPED ADE GUARD ASSUMED. This kernel emits ONE bounds
    # guard — the E half's, over the stored extent — so every recurrence volume
    # must carry exactly that many cells, or the P half would run over a range
    # its own certified kernel would not have.
    for index, state, component in driven_pairs:
        target = getattr(fields, component, None)
        try:
            walked = int(target.size)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"fields.{component} could not state its size: "
                           f"{type(exc).__name__}: {exc}")
            continue
        for label, volume in (("P", (getattr(state, "P", {}) or {}).get(component)),
                              ("P_prev", (getattr(state, "P_prev", {}) or {})
                               .get(component)),
                              ("_scratch", getattr(state, "_scratch", None))):
            if volume is None:
                continue  # the ADE half already refused the missing buffer
            try:
                size = int(volume.size)
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"polarization {index} {label}[{component}] could "
                               f"not state its size: {type(exc).__name__}: {exc}")
                continue
            if size != walked:
                reasons.append(
                    f"polarization {index} {label}[{component}] holds {size} cells "
                    f"and the launch walks {walked}; this weld drops the certified "
                    f"update_P guard and may only do so where the two are one "
                    f"number")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_folded_offdiag_fused_ade_chain(fields: Any, pml: Any) -> Coverage:
    """The verdict with its reasons, for reports. Needs no Triton."""
    return folded_offdiag_fused_ade_chain_coverage(fields, pml)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


class FoldedOffdiagFusedAdeChainPlan:
    """ONE launch that performs two driver passes, plus the reference rotation.

    COMPOSITION OVER THE CERTIFIED E PLAN, NOT A SUBCLASS. The E-half bindings —
    the boundary-code / wall-axis / ghost-weight validation, the output-alias
    refusal, the ``MAX_POLES`` check, the dead-slot padding and the live pole
    binding's per-launch resolution with its live-P-versus-output refusal — are
    :class:`folded_offdiag_dispersive_update_e.FoldedOffdiagDispersiveConstitutive
    Plan`'s and are not restated here; this plan holds one, reads its bindings
    at launch, and launches a DIFFERENT kernel. A subclass that inherited ``run``
    would launch the certified one — the exact trap the folded off-diagonal
    base plan documents against (folded_offdiag_update_e.py:1066-1072).

    ``launches_per_run`` is ONE and is declared rather than inherited: a whole-step
    arbiter asserts the exact per-cycle launch count on every filled slot, and that
    assertion is what stops a slot passing by not executing.

    THE ROTATION IS PERFORMED HERE, AFTER THE LAUNCH, and it is
    ``dispersion.PolarizationState.update``'s (dispersion.py:687-691) over a
    DIFFERENT buffer set: one scratch per driven component rather than one shared
    per susceptibility. The three-cycle each (susceptibility, component) walks is
    the reference's; what changes is that no two of them share a buffer, which is
    what makes the launch's write set disjoint from its read set. See the module
    docstring for why the chain's orbit enumeration transfers to this body.

    THE POINTERS ARE RESOLVED PER LAUNCH, DELIBERATELY. The rotation moves which
    allocation holds ``P``, ``P_prev`` and each scratch on every step, so a plan
    that cached them would be stale after the FIRST step — and stale in a way that
    still computes, giving a smooth wrong field. The ORDER is cached (it is fixed
    for the run) and re-checked every call by the E half's own ``LivePoleBinding``.

    THIS PLAN OWNS THE ROTATION FOR THE RUN. ``PolarizationState.update`` rotates a
    single shared ``_scratch``; interleaving it with this plan would hand one
    component's launch a buffer another slot already owns. ``_scratch`` is kept
    pointing at the FIRST driven component's scratch so the engine object is never
    left holding an allocation this plan has also given to ``P``, but the array
    path and this plan must not step the same ``Fields``.
    """

    performs_device_work = True
    replaces_sub_steps = REPLACES
    launches_per_run = 1

    __slots__ = ("fields", "shape", "n_elem", "block", "num_warps", "counts",
                 "extra_scratch_volumes", "launches", "runs", "alias_checks",
                 "_e", "_pointer", "_order", "_states", "_index_of", "_scratch",
                 "_first_driven", "_sigma", "_coeff", "_sv", "_kernel",
                 "_read_ids")

    def __init__(self, fields: Any, pml: Any, block: int,
                 num_warps: Optional[int] = 1, kernel: Any = None,
                 pointer: Any = None, chunk_layout: Any = None) -> None:
        from .launch import CupyPointer, chunk_bounds_for  # noqa: PLC0415

        if pointer is None:
            pointer = CupyPointer
        self.fields = fields
        self.shape = tuple(int(n) for n in fields.grid.shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self.num_warps = num_warps
        self._pointer = pointer
        self._kernel = kernel
        self.launches = 0
        self.runs = 0
        self.alias_checks = 0

        codes, fold_reasons = _symmetry.folded_axis_kinds(fields.grid, pml)
        if codes is None:
            raise ValueError(
                "the folded axis kinds could not be classified: "
                + "; ".join(fold_reasons))

        components = tuple(term[0] for term in E_TERMS)
        targets = [getattr(fields, name) for name in components]
        auxiliaries = [getattr(fields, "f_w_" + name) for name in components]
        sources = [getattr(fields, term[1]) for term in E_TERMS]
        inverse = [fields.inverse_epsilon_for(name) for name in components]
        rows = _offdiag.row_volumes_for(fields)
        columns = [getattr(pml, f"{stem}_{axis}_h")
                   for axis in "xyz" for stem in ("kps", "kms")]

        partition = poles_per_component(fields)
        self._order = {name: tuple(states) for name, states in partition.items()}
        # THE CERTIFIED E PLAN, built with its own argument list. Its pole order
        # is this plan's order — one derivation, so the subtraction slots and the
        # ADE arms cannot disagree about which state sits in which slot.
        self._e = _fod.FoldedOffdiagDispersiveConstitutivePlan(
            fields.grid.shape, self.block, targets, auxiliaries, sources,
            inverse, LivePoleBinding(fields, self._order), rows, columns, codes,
            _offdiag.wall_mask_axes(fields.grid),
            _folded.mirror_ghost_weights(fields.grid),
            chunk_bounds=chunk_bounds_for(fields, pml, chunk_layout),
            presence=_offdiag.presence_for(fields, pml))
        self.counts = tuple(len(self._order[name]) for name in components)
        for name, count in zip(components, self.counts):
            if count > CHAIN_MAX_POLES:
                raise ValueError(
                    f"{name} is driven by {count} poles; this product binds "
                    f"{CHAIN_MAX_POLES} ADE arms per component and the predicate "
                    f"refuses more")

        # THE READ INVENTORY the alias check holds the ADE outputs against. The
        # chain's set plus THE SIX ROW VOLUMES: the E half gathers them at
        # neighbour offsets while the outputs are being written, so a scratch
        # that IS a row volume would make the answer depend on block schedule.
        read_ids = {id(array) for array in targets + auxiliaries + sources
                    + inverse + columns}
        read_ids.update(id(value) for value in rows if value is not None)

        self._states = tuple(getattr(fields, "polarizations", ()) or ())
        self._index_of = {id(state): index
                          for index, state in enumerate(self._states)}

        # ONE SCRATCH PER DRIVEN COMPONENT — the shape, allocated. The FIRST driven
        # component takes the susceptibility's own ``_scratch`` and the rest are
        # new, so the extra is ``K * (d - 1)`` volumes and not ``K * d``; that is
        # the figure the shape probe's LEG 3 costed and this counter reports what
        # the run actually paid.
        xp = fields.grid.xp
        self._scratch: Dict[Tuple[int, str], Any] = {}
        self._first_driven: Dict[int, str] = {}
        extra = 0
        for state_index, state in enumerate(self._states):
            driven = [name for name in components
                      if any(state is other for other in self._order[name])]
            if not driven:
                continue
            self._first_driven[state_index] = driven[0]
            reference = state.P[driven[0]]
            for position, name in enumerate(driven):
                if position == 0:
                    scratch = getattr(state, "_scratch", None)
                    if scratch is None:
                        raise ValueError(
                            f"polarization {state_index} carries no _scratch; the "
                            f"first driven component reuses it and the predicate "
                            f"checks it")
                else:
                    scratch = xp.zeros(self.shape, dtype=reference.dtype)
                    extra += 1
                self._scratch[(state_index, name)] = scratch
        self.extra_scratch_volumes = extra

        self._sigma: Dict[Tuple[str, int], Any] = {}
        self._coeff: Dict[Tuple[str, int], Tuple[float, float, float, float]] = {}
        self._sv: Dict[str, int] = {}
        for index, name in enumerate(components):
            letter = LETTERS[index]
            for slot in range(CHAIN_MAX_POLES):
                if slot >= len(self._order[name]):
                    self._sv[f"SV_{letter}{slot}"] = 0
                    continue
                state = self._order[name][slot]
                # Decided by the same function the predicate checked with, so the
                # clause and the constexpr cannot disagree; getting them out of
                # step reads a scalar as a pointer, which is a wrong answer and
                # not a crash (coverage.sigma_is_volume).
                volume = bool(_coverage.sigma_is_volume(state, name))
                sigma = state.sigma[name]
                self._sigma[(name, slot)] = (pointer(sigma) if volume
                                             else float(sigma))
                if volume:
                    read_ids.add(id(sigma))
                # MEEP's four realnum words (PolarizationState.constants), each a
                # float32 word, so the Python float passed is that word exactly.
                self._coeff[(name, slot)] = tuple(float(word)
                                                  for word in state.constants)
                self._sv[f"SV_{letter}{slot}"] = 1 if volume else 0
        self._read_ids = frozenset(read_ids)

    # -- one launch ---------------------------------------------------------

    def _resolve(self, groups: Sequence[Sequence[Any]]) -> List[List[Tuple[Any, ...]]]:
        """The destination chain for every component, resolved BEFORE the launch.

        ``(key, state, out, pole, prev)`` per live slot. The pole is taken from the
        E half's own live binding and CHECKED to be the state's ``P`` — the
        pointer the ADE arm re-reads is that slot's, so a disagreement would
        advance one pole's recurrence from another pole's history.
        """
        rows: List[List[Tuple[Any, ...]]] = []
        for index, (component, _displacement, _axis) in enumerate(E_TERMS):
            entries: List[Tuple[Any, ...]] = []
            for slot, state in enumerate(self._order[component]):
                pole = state.P[component]
                if pole is not groups[index][slot]:
                    raise RuntimeError(
                        f"{component} slot {slot}: the ADE arm's pole is not the "
                        f"array the subtraction slot resolved; the re-read pointer "
                        f"would be a different volume")
                key = (self._index_of[id(state)], component)
                entries.append((key, state, self._scratch[key], pole,
                                state.P_prev[component]))
            rows.append(entries)
        return rows

    def _check_aliasing(self, chain: Sequence[Sequence[Tuple[Any, ...]]]) -> None:
        """No ADE output is an array this launch reads, and no two share one.

        THE ALIAS CHECK IS NOT DECORATION. The whole licence for this shape is that
        the launch never writes a buffer it reads, and the rotation moves those
        identities every step. The check is O(poles) integer comparisons over lists
        resolved for the launch anyway, and it is what turns the orbit enumeration
        into an invariant this plan cannot violate silently. The read inventory
        includes the six row volumes the E half gathers at neighbour offsets.
        """
        written = [row[2] for entries in chain for row in entries]
        identifiers = {id(value) for value in written}
        if len(identifiers) != len(written):
            raise RuntimeError(
                "two ADE arms resolved the SAME output buffer, so one recurrence "
                "would overwrite the other's result in the same launch; each "
                "(susceptibility, component) owns its own scratch and this cannot "
                "happen on a plan-built chain, which is why it is checked rather "
                "than assumed")
        read = set(self._read_ids)
        for entries in chain:
            for _key, _state, _out, pole, prev in entries:
                read.add(id(pole))
                read.add(id(prev))
        collision = identifiers & read
        if collision:
            raise RuntimeError(
                f"{len(collision)} ADE output buffer(s) are also read by this "
                f"launch; this shape's whole licence is that they cannot be (see "
                f"the module docstring's orbit enumeration)")
        self.alias_checks += 1

    def _arguments(self, chain: Sequence[Sequence[Tuple[Any, ...]]]) -> List[Any]:
        """Every runtime argument, in the order the signature declares them.

        THE CERTIFIED RUN'S ORDER FIRST — targets, auxiliaries, sources, inverse
        epsilons, the live pole slots (padded by the certified plan), the six row
        slots, the six coefficient columns, the three ghost weights, the chunk
        record and the row presence, the geometry — read off the composed E plan
        rather than re-derived, THEN the ADE block laid out as the chain lays it:
        p_out, p_prev, sigma and MEEP's four words per component, dead slots padded
        with the component's own D pointer and ``0.0``.

        SEPARATE FROM :meth:`run` so the ORDER is testable on a host with no
        Triton: a launcher that assembles the right pointers in the wrong order
        binds each one to a neighbouring slot, which is a wrong answer and not a
        crash. The count is asserted against :data:`RUNTIME_ARGUMENTS` here and
        that constant is asserted against the kernel's own signature by the test.
        """
        pointer = self._pointer
        base = self._e._base
        nx, ny, nz = base.shape
        ade: List[Any] = []
        for index, entries in enumerate(chain):
            component = E_TERMS[index][0]
            padding = base._sources[index]
            live = len(entries)
            ade.extend([pointer(row[2]) for row in entries])
            ade.extend([padding] * (CHAIN_MAX_POLES - live))
            ade.extend([pointer(row[4]) for row in entries])
            ade.extend([padding] * (CHAIN_MAX_POLES - live))
            ade.extend([self._sigma[(component, slot)] for slot in range(live)])
            ade.extend([0.0] * (CHAIN_MAX_POLES - live))
            for position in range(4):
                ade.extend([self._coeff[(component, slot)][position]
                            for slot in range(live)])
                ade.extend([0.0] * (CHAIN_MAX_POLES - live))
        arguments = [*base._targets, *base._aux, *base._sources, *base._inv_eps,
                     *self._e._live_pole_slots(), *base._rows,
                     *base._coefficients, *base.ghost_weights, base._cb, base._op,
                     nx, ny, nz, base.n_elem, *ade]
        if len(arguments) != RUNTIME_ARGUMENTS:
            raise RuntimeError(
                f"the launcher assembled {len(arguments)} runtime arguments and "
                f"the kernel declares {RUNTIME_ARGUMENTS}; a signature edit the "
                f"launcher did not follow binds every later pointer to the wrong "
                f"slot, which is a wrong answer and not a crash")
        return arguments

    def _rotate(self, chain: Sequence[Sequence[Tuple[Any, ...]]]) -> None:
        """``dispersion.py:687-691``, per (susceptibility, component).

        The reference's three-cycle, walked independently for every driven
        component instead of once over a shared scratch. ``_scratch`` on the
        susceptibility is left pointing at the FIRST driven component's slot so the
        engine object never holds an allocation this plan has also handed to ``P``.
        """
        for index, entries in enumerate(chain):
            component = E_TERMS[index][0]
            for key, state, out, pole, prev in entries:
                state.P[component] = out
                state.P_prev[component] = pole
                self._scratch[key] = prev
        for state_index, component in self._first_driven.items():
            self._states[state_index]._scratch = self._scratch[
                (state_index, component)]

    def run(self, guard: Optional[bool] = None) -> None:
        """One complete E->P seam, in place. Rotate only after the launch succeeds.

        The launch comes FIRST and the Python-side rotation second, the ordering
        the chain and the certified ADE plan both use and for the same reason: a
        plan that rotated first and then raised would leave the engine holding a P
        history that never happened. The constexprs are the composed E plan's own
        bindings — row mask, boundary codes, ghost axes, wall axes, block — so
        the E half is specialised exactly as the certified launch specialises it.
        """
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else folded_offdiag_fused_ade_chain_kernel())
        e = self._e
        base = e._base
        groups = e._poles.arrays()
        chain = self._resolve(groups)
        self._check_aliasing(chain)
        arguments = self._arguments(chain)
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        varies = base.presence_varies
        kernel[base.launch_grid](
            *arguments,
            NP0=e.counts[0], NP1=e.counts[1], NP2=e.counts[2],
            R01=base.row_mask[0], R02=base.row_mask[1],
            R11=base.row_mask[2], R12=base.row_mask[3],
            R21=base.row_mask[4], R22=base.row_mask[5],
            P01=varies[0], P02=varies[1], P11=varies[2], P12=varies[3],
            P21=varies[4], P22=varies[5], PRESENCE=int(any(varies)),
            BCX=base.boundary_codes[0], BCY=base.boundary_codes[1],
            BCZ=base.boundary_codes[2],
            MG_X=base.ghost_axes[0], MG_Y=base.ghost_axes[1],
            MG_Z=base.ghost_axes[2],
            WM_X=base.wall_axes[0], WM_Y=base.wall_axes[1],
            WM_Z=base.wall_axes[2],
            BLOCK=base.block, **self._sv,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        self.launches += 1
        self._rotate(chain)
        self.runs += 1

    def __repr__(self) -> str:
        return (f"FoldedOffdiagFusedAdeChainPlan(shape={self.shape}, "
                f"rows={self._e.row_mask}, poles={self.counts}, "
                f"extra_scratch={self.extra_scratch_volumes}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_folded_offdiag_fused_ade_chain(fields: Any, pml: Any,
                                        block: Optional[int] = None,
                                        num_warps: Optional[int] = 1,
                                        kernel: Any = None,
                                        chunk_layout: Any = None
                                        ) -> Optional[FoldedOffdiagFusedAdeChainPlan]:
    """Build the fused E->P plan for this cell, or return ``None`` on any refusal.

    ``None`` means REFUSED and the reasons are available from
    :func:`folded_offdiag_fused_ade_chain_coverage`. The Triton import stays
    BELOW the predicate, as in every builder in this package: a NumPy host must
    be able to plan (to ``None``) without the optional dependency being
    importable at all.

    ``kernel=`` IS LOAD-BEARING: the gate's mutation legs route a mutated kernel
    through it, and a builder that dropped it would launch the SHIPPED kernel and
    report a pass for a defect it never introduced.
    """
    if not folded_offdiag_fused_ade_chain_coverage(fields, pml).covered:
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    return FoldedOffdiagFusedAdeChainPlan(
        fields, pml, DEFAULT_BLOCK if block is None else block,
        num_warps=num_warps, kernel=kernel, chunk_layout=chunk_layout)


# ---------------------------------------------------------------------------
# WIRING — deferred, exactly as the other cross-sub-step products are held
# ---------------------------------------------------------------------------
#
# ``launch.plan_step`` does not import this module, ``launch.FAMILY_MODULES``
# does not name it and ``launch.CERTIFIED_FUSED_PRODUCTS`` does not table it, so
# nothing about any existing arm's behaviour changes and a default run still
# steps this seam on the array path. The product exists to be MEASURED:
# ``parity/meep_gpu/gate_triton_folded_offdiag_fused_ade_chain.py`` is its byte
# gate. Composition is a separate change with its own disjointness argument,
# because this product spans TWO slots and the planner's arm table assigns at
# most one arm per slot; :data:`INSTALLABLE_REASON` names what would have to grow.
