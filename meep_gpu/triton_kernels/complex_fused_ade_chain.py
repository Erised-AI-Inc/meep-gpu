"""Complex ``update_E`` welded to complex ``update_P`` — the Triton E->P product
for complex64 storage.

THE COMPLEX64 SIBLING of :mod:`.fused_ade_chain`, in exactly the relation
:mod:`.complex_no_pml_stored_e` has to :mod:`.no_pml_stored_e`: a separate family,
a separate predicate, the same seam, the same launch shape. It claims the ONE
E->P cell that module refuses BY NAME — "**real float32 only.** The four
``TestLoadDump.*_3d`` rows are complex64; their E half is
``complex_no_pml_stored_e`` and their ADE half ``complex_ade``, a different
signature and a different arm. Refused, exactly as the Metal family declines
them" (fused_ade_chain.py, "SCOPE, AND WHY EACH BOUND IS A REFUSAL BY NAME").

The Metal twin (``metal_kernels/complex_fused_ade_chain.py``) claims the same
cell for the same four rows, and the two backends' reach at this seam is then
equal.

===========================================================================
THE SEAM IS EMPTY, AND THAT IS THE WHOLE REASON THIS PRODUCT EXISTS
===========================================================================

``FdtdDriver.step`` runs::

    if fast is None or not fast.dispatch("update_E", self.fields):
        update_E(self.fields, self.pml)                       # driver.py:3303-3304
    if fast is None or not fast.dispatch("update_P", self.fields):
        update_P(self.fields, self.pml)                       # driver.py:3305-3306

Nothing sits between them: no source injection, no mirror fill, no wall clear, no
far-ghost pass. Every other seam on this board has a driver pass inside it that a
fused kernel must carry inline or refuse; this one has none, which is why the
fusion matrices record ``E->P: 15 instances, 0 blocked by the source seam,
CEILING 15``.

===========================================================================
WHAT IT IS WORTH, AND THE FOUR ROWS BY NAME
===========================================================================

FOUR SEAM-INSTANCES OF THE 387. The rows are
``tests:TestLoadDump.test_load_dump_{structure,structure_sharded,chunk_layout_file,
chunk_layout_sim}_3d`` and their configuration is READ FROM THE CENSUS rather
than assumed here (``results/predicate_coverage_2026-08-21_eop_chain/``, the same
walk that measured :mod:`.fused_ade_chain`'s ten): ``force_complex_fields=True``,
``pml_active=False``, ONE registered lorentzian susceptibility driving all three
of ``Ex``/``Ey``/``Ez``, no fold, no off-diagonal ``chi1inv``, no ``chi2``/
``chi3``, no cylindrical axis, one MAGNETIC source. They are the same four rows
:mod:`.complex_no_pml_stored_e` was built for and the same four
:mod:`.complex_ade` was built for; this product is their seam and nothing wider.

===========================================================================
WHAT IS FUSED, AND IT IS EXACTLY TWO LOADS PER COMPONENT PER CELL
===========================================================================

* **the drive.** ``complex_ade_update_p``'s drive is
  ``Fields.drive_field(component)``, and this family's ADE half already REQUIRES
  that to be the stored ``E`` and not ``f_w`` (complex_ade.py:184-215, ``_inert_drive_reasons``). That is the value this kernel's E half computed and stored one
  launch earlier, which the certified complex E body already NAMES before storing
  it (``o0_re, o0_im = _mul_field_left(...)``,
  complex_no_pml_stored_e.py:152-155). So ``w_re = tl.load(drive + word)``
  becomes ``w_re = o0_re`` — the register;
* **the pole.** ``complex_ade_update_p`` loads ``p_now``; the pole-aware ``D - P``
  subtraction in the E half has already loaded exactly that volume, so
  ``p_re = tl.load(p_now + word)`` becomes ``p_re = pa0_re``. Naming that load is
  the ONE edit this module makes to ``_subtract_complex_poles``'s body, and it is
  the same edit :mod:`.fused_ade_chain` makes to its own subtraction chain.

Neither elimination changes a rounded value: a float32 word stored and re-loaded
from the same address is the identity, and the register reused is the one the
eliminated load would have produced. That is an argument for why byte identity is
EXPECTED, not evidence that it holds; the device gate is what decides it, per
COMPLETE seam step, against both the array path and the two certified sub-step
products dispatched beside it.

EVERY OTHER LINE IS TRANSCRIBED FROM THE BODY IT REPLACES, and the transcription
is CHECKED BY PARSING both sources rather than restated as data here — a second
copy of "the load-bearing lines" is one more place to drift.
``test_triton_complex_fused_ade_chain`` reads the certified kernels' own text with
``ast`` and requires each arithmetic line to appear in this kernel modulo the
documented renames, and the device gate re-runs that check beside the bytes.

===========================================================================
THE ONE FACT THIS FAMILY MUST NOT INHERIT: WHICH COMPLEX EXPANSION ARM
===========================================================================

Both halves multiply complex numbers, and WHICH EXPANSION the platform's reference
takes is a MEASURED fact rather than a choice
(:func:`complex_fields.expansion_from_probe`). The two certified halves ask the
SAME probe record for DIFFERENT pattern sets:

* ``complex_no_pml_stored_e`` needs the shared four
  (:data:`complex_fields.PROBE_PATTERNS`) — its only complex multiply is
  ``c8_mul_f4_field_left`` (complex_no_pml_stored_e.py:27-30);
* ``complex_ade`` needs those four PLUS ``c8_mul_python_float_field_left``, the
  fifth orientation ``P * c_now`` introduces (complex_ade.py:11-24).

ONE FUSED SOURCE CARRIES ONE ``EXPANSION`` CONSTEXPR. A platform whose probe
licensed different arms for the two pattern sets would need the two halves to
disagree inside a single scope, which is unspellable. On the certified host both
resolve to ``FMA_V1`` — but agreeing today is a coincidence and not an invariant,
so this module does not assume it: :func:`complex_fused_ade_chain_coverage` calls
BOTH resolvers and REFUSES BY NAME when they disagree, and
:func:`resolve_chain_expansion` is the single place the launched constexpr comes
from. A platform that split the two arms earns a new product rather than a silent
downgrade of one half.

===========================================================================
THE ROTATION, AND WHY THE SHAPE IS ONE SCRATCH PER DRIVEN COMPONENT
===========================================================================

``complex_ade_update_p`` launches once per DRIVEN COMPONENT and the host rotates
``P`` / ``P_prev`` / ``_scratch`` after each launch (complex_ade.py:342-377,
transcribing dispersion.py:687-691). A fused kernel must either take one
component, bake the rotation, or change the buffer set.

**The per-component-launch shape is not available here, for
:mod:`.fused_ade_chain`'s measured reason and one more.** Every certified Triton
``update_E`` writes ALL THREE components in one launch, and
``complex_stored_e_step`` is one of the three the shape probe parsed and found to
do so (``probe_triton_ade_rotation_shape.py`` names
``complex_no_pml_stored_e``/``complex_stored_e_step`` in its subject table at
:116-117, LEG 1). Splitting that certified body to get Metal's shape would put
this product's arithmetic outside what has been gated.

**The shared-scratch shape aliases inside the launch, and the four rows this
product serves are exactly the case where it does.** ``dispersion.py`` allocates
ONE ``_scratch`` per susceptibility, so with ``d`` driven components the Ey arm's
OUTPUT buffer IS the Ex arm's ``p_prev`` INPUT and the Ez arm's is Ey's. That is
:mod:`.fused_ade_state`'s recorded hazard, whose own note says the safety of it is
"an OBSERVATION about one toolchain rather than a language guarantee" and that
"Fusing further REQUIRES breaking the chain first, not inheriting it." The four
rows here drive ``d = 3``, which is the worst position of that orbit and not the
benign ``d = 1`` one.

**So this product takes the third shape, one scratch per DRIVEN COMPONENT, and
the licence for it transfers to complex64 EXACTLY because the walk that
established it never looked at a dtype.**
``probe_triton_ade_rotation_shape.rotate`` (:277-304) walks buffer NAMES through
``dispersion.PolarizationState.update``'s three assignments, and ``orbit``
(:313-334) closes the permutation. ``PolarizationState.update`` is one body for
every storage kind — it assigns buffer references and never reads an element — so
the permutation a complex64 susceptibility performs IS the permutation that walk
enumerated::

    d   per-component launch   shared scratch          one scratch per component
    1   orbit 3, 0 conflicts   orbit 3, 0 conflicts    orbit 3, 0 conflicts
    2   orbit 5, 0 conflicts   orbit 5, 5 conflicts    orbit 3, 0 conflicts
    3   orbit 7, 0 conflicts   orbit 7, 7 conflicts    orbit 3, 0 conflicts

Transferring an argument is not the same as re-establishing it, so the device
gate RE-WALKS that orbit for this product's own buffer set rather than citing the
table, and :meth:`ComplexFusedAdeChainPlan.run` ASSERTS the read/write
disjointness before EVERY launch rather than trusting either.

WHAT IT COSTS: ``K * (d - 1)`` extra complex64 field volumes — the reference
allocates one scratch per susceptibility and this allocates one per driven
component. Over the four rows this serves that is ``+2`` complex volumes each
(K = 1, d = 3), i.e. four float32 field volumes' worth. The cost is MEMORY ONLY:
no extra launch, no extra arithmetic, and the values left in ``P`` and ``P_prev``
after the rotation are the reference's, which is what the byte gate compares.
``extra_scratch_volumes`` on the plan reports the number the run actually paid.

THE RETIRED SCRATCH IS THE ONE THING THAT DOES *NOT* MATCH THE REFERENCE, and it
is named here rather than discovered by a comparator. After a step the reference's
single ``_scratch`` holds the retired ``P_prev`` of the LAST driven component;
this plan's per-component scratches hold the retired ``P_prev`` of EACH. They are
dead state — the next step's launch overwrites every one of them before reading
anything from it. The gate compares ``P`` and ``P_prev`` per (susceptibility,
component) plus the last component's scratch, and says so.

===========================================================================
SCOPE, AND WHY EACH BOUND IS A REFUSAL BY NAME
===========================================================================

* **ONE ARM, and it is not a choice.** The complex ADE half REQUIRES the inert
  stored-E drive (complex_ade.py:184-215) and the complex E half REQUIRES no
  active PML (complex_no_pml_stored_e.py:199-207). The two refusals meet at
  exactly one configuration, so unlike :mod:`.fused_ade_chain` there is no second
  or third arm to select between. A complex row under an ACTIVE absorber has no
  certified complex E body that writes ``f_w`` beside a complex ADE half that
  reads it, and is refused by both halves rather than by a clause here;
* **at most :data:`CHAIN_MAX_POLES` pole on one component.** ONE, because one is
  the corpus maximum at this cell (K = 1 on all four rows) and every extra slot
  costs six more kernel arguments per component. A second pole is refused BY NAME
  rather than truncated: the E half compiles :data:`MAX_POLES` = 8 slots and would
  SUBTRACT it while the ADE half never ADVANCED it — a frozen P, smooth and wrong.
  :mod:`.fused_dispersive_chain` bounds itself the same way and for the same
  reason (``CHAIN_MAX_POLES = 1``, fused_dispersive_chain.py:84);
* **no off-diagonal ``chi1inv``, no ``chi2``/``chi3``, no fold, no cylindrical
  axis, no BFAST, no beta** — every one of them inherited from
  ``complex_stored_e_coverage`` unchanged and not re-decided here. A Bloch phase
  and a conductivity ARE admitted, also inherited: this sub-step is pointwise and
  a conductivity enters the B/D curl recurrences rather than ``update_E``.

NOT WIRED, and dispatch is unchanged. ``launch.plan_step`` does not import this
module and no arm table names it, exactly as :mod:`.fused_ade_chain` and
:mod:`.fused_dispersive_chain` are held: the product is built, gated and
measurable, and a default run still steps this seam on the array path.

DEVICE RESULT
-------------
``parity/meep_gpu/gate_triton_complex_fused_ade_chain.py`` ran on the GPU host's RTX
A6000 (GPU pinned by UUID and verified physically empty immediately before the
run) on 2026-08-21, Triton 3.1.0 / CuPy 13.5.1, under the ``ieee_keep_ftz_stripped``
subnormal policy. EIGHT product rows — one to three driven components, one and two
susceptibilities, scalar and volume sigmas, Lorentz and Drude, periodic and
metallic walls, zero and nonzero Bloch vector — were byte-identical to the CuPy
array path over SIXTY complete E->P seam steps, at ONE launch per step, and
identical to the two separately certified sub-step products dispatched beside them
on identical state (FOUR dispatches per step collapsing to ONE). Both halves of
every complex volume carried live values on every row. Four value-class rows — a
+-0 lattice and the subnormal band — were identical too, the band measured LIVE
rather than assumed. Ten kernel-source mutations and four host mutations were
CAUGHT; four kernel nulls and one host null were CONFIRMED with their PTX
evidence, and one edit was declared DEAD and not scored. The release verdict was
shown to FLIP against a planted defect. Readable entries:
``triton_complex_fused_ade_chain_device_gate`` in ``fingerprints.json`` and
``results/triton_complex_fused_ade_chain_2026-08-21/``. This is a correctness
result, not a throughput claim.

TWO MEASURED NULLS ARE THIS PRODUCT'S OWN CLAIM, and both were scored on a case
that COULD have seen them. Reversing the operand orientation
(``_mul_field_left`` -> ``_mul_coefficient_left``) and pinning ``EXPANSION`` to the
other licensable arm each compiled to different PTX and changed NO byte, on the
physical band AND on the subnormal band with the census live (11 and 7 subnormal
words in the reference). The reason is structural: every complex multiply in this
kernel has a REAL operand, and the two arms separate only where the fused
product's extra precision changes a rounded word. **The ``EXPANSION`` clause is
therefore a LICENCE requirement rather than something these cases can falsify**,
and the predicate refuses without a probe for that reason and not because a
divergence was observed.

ONE SUBNORMAL POLICY, AND THE BOUND IS INHERITED. The complex ADE half needs a
seventh probe orientation (``c8_mul_python_float_field_left``) and the only writer
of a record carrying it, ``gate_triton_unified_expansion.py``, calls
``install_ftz_strip()`` unconditionally — so no FLUSH-cut artifact that licenses
this family exists and the flush leg is a NAMED refusal rather than a missing row.
Every complex Triton family on this board is certified keep-only, including BOTH
of this product's halves (``triton_complex_ade_device_gate``: "keep";
``triton_complex_no_pml_stored_e_device_gate``: "keep"). A fused product cannot be
certified under a policy its halves were not.

CORPUS REACH: four seam-instances of the 387, the four
``tests:TestLoadDump.*_3d`` rows, measured by the shipped predicate in
``results/predicate_coverage_2026-08-21_eop_chain/``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import complex_ade as _complex_ade
from . import complex_no_pml_stored_e as _complex_e
from . import coverage as _coverage
from .complex_fields import (
    DEFAULT_BLOCK,
    _UnavailableKernel,
    _mul_coefficient_left,
    _mul_field_left,
    _word_view,
)
from .coverage import Coverage

__all__ = [
    "ARM",
    "CHAIN_MAX_POLES",
    "E_TERMS",
    "LETTERS",
    "MAX_POLES",
    "POLICY",
    "REPLACES",
    "RUNTIME_ARGUMENTS",
    "ComplexFusedAdeChainPlan",
    "complex_fused_ade_chain_coverage",
    "complex_fused_ade_chain_kernel",
    "complex_fused_ade_chain_step",
    "explain_complex_fused_ade_chain",
    "plan_complex_fused_ade_chain",
    "resolve_chain_expansion",
]

#: This family has ONE arm and the name is data rather than a literal so the
#: predicate, the plan and the fusion matrix cannot disagree about it. See the
#: docstring's SCOPE section for why there is no second: the two certified halves'
#: own refusals meet at exactly one configuration.
ARM = "complex_no_pml"

#: The driver passes one run of this plan performs, in driver order
#: (driver.py:3303-3306). Declared rather than inferred from a slot name — a
#: module whose REPLACES disagrees with what its kernel carries is a defect class
#: this package has already paid for once.
REPLACES: Tuple[str, ...] = ("update_E", "update_P")

#: The E half's compiled pole slots. EIGHT, because that is what the certified
#: complex body compiles (``complex_no_pml_stored_e.MAX_POLES``) and the
#: subtraction chain below is that body's chain — pinned equal to it by
#: ``test_triton_complex_fused_ade_chain``, which is the honest way to keep two
#: independently-written bodies in step. NOT imported from it: an import lets that
#: module's edit silently truncate this kernel's chain.
MAX_POLES = 8

#: The ADE half's live slots per component, and the bound the predicate refuses
#: past. ONE — the corpus maximum at this cell, measured (all four rows carry
#: K = 1). A second pole is refused BY NAME rather than truncated; see SCOPE.
CHAIN_MAX_POLES = 1

#: ``(component, displacement)`` — ``complex_no_pml_stored_e.E_TERMS``. Pinned
#: equal to it by test rather than imported, for :data:`MAX_POLES`'s reason.
E_TERMS: Tuple[Tuple[str, str], ...] = (("Ex", "Dx"), ("Ey", "Dy"), ("Ez", "Dz"))

#: The per-component argument-name letters, matching the kernel's signature.
LETTERS: Tuple[str, str, str] = ("a", "b", "c")

#: The settled launch configuration. ONE WARP is the shipped policy for every
#: cross-sub-step fused product (``fused_dispersive_chain.POLICY``, measured
#: 1.0739x median over 12 shape/boundary cases). ``enable_fp_fusion`` is off
#: engine-wide (``kernels.ENABLE_FP_FUSION``) and is what keeps this seam from
#: contracting a multiply and an add into an FMA the reference did not take.
#: This product does not tune.
POLICY: Dict[str, Any] = {
    "num_warps": 1,
    "block": "complex_fields.DEFAULT_BLOCK",
    "enable_fp_fusion": "kernels.ENABLE_FP_FUSION (False)",
    "status": "keep",
}

#: Runtime (non-constexpr) parameters :func:`complex_fused_ade_chain_step`
#: declares: three targets, three sources, three inverse epsilons, the E half's
#: pole slots, six ADE arguments per live slot per component, and ``n_elem``.
#: Asserted against the kernel's OWN signature by the laptop test and against the
#: assembled argument list at EVERY launch, so a signature edit the launcher does
#: not follow fails at the call rather than binding the wrong pointer to the wrong
#: slot.
RUNTIME_ARGUMENTS = (3 + 3 + 3 + 3 * MAX_POLES + 3 * 6 * CHAIN_MAX_POLES + 1)


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------
#
# Defined at MODULE scope behind a conditional import, not inside a builder:
# `@triton.jit` resolves a body's names — including the `tl.constexpr` annotations
# — through the defining module's `__globals__`, so a kernel defined inside a
# function compiles to `NameError('tl is not defined')` at first launch
# (dispersive_update_e.py:138-143 records the run that showed it, 320 of 320).

try:  # pragma: no cover - the absent branch is exercised by the absence test
    import triton
    import triton.language as tl
    _TRITON_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _exc:  # pragma: no cover - the laptop path
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = _exc


if triton is not None:  # pragma: no cover - device code, certified by the byte gate

    @triton.jit
    def complex_fused_ade_chain_step(
        f0, f1, f2,                     # targets:  Ex, Ey, Ez word pointers (out)
        g0, g1, g2,                     # sources:  Dx, Dy, Dz word pointers (in)
        e0, e1, e2,                     # float32 inverse epsilon, per component
        a0, a1, a2, a3, a4, a5, a6, a7,   # component 0's poles, IN ORDER
        b0, b1, b2, b3, b4, b5, b6, b7,   # component 1's poles, IN ORDER
        c0, c1, c2, c3, c4, c5, c6, c7,   # component 2's poles, IN ORDER
        # one ADE arm per component: p_out / p_prev / sigma / the coefficient triple
        p_out_a0, p_prev_a0, sigma_a0, cnow_a0, cprev_a0, cdrive_a0,
        p_out_b0, p_prev_b0, sigma_b0, cnow_b0, cprev_b0, cdrive_b0,
        p_out_c0, p_prev_c0, sigma_c0, cnow_c0, cprev_c0, cdrive_c0,
        n_elem,
        NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
        SV_a0: tl.constexpr, SV_b0: tl.constexpr, SV_c0: tl.constexpr,
        EXPANSION: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Complex ``update_E`` welded to complex ``update_P``: both driver passes,
        ONE grid, complex64 addressed as interleaved float32 word pairs.

        THE E HALF IS ``complex_no_pml_stored_e.complex_stored_e_step``
        (complex_no_pml_stored_e.py:132-170), with ``_subtract_complex_poles``
        (:96-129) INLINED. Inlining is not a rewrite: the eight ``if NP > n``
        guards, the load order and the subtraction order are that helper's, and the
        ONE edit is that each loaded pole is NAMED (``pa0_re``) before it is
        subtracted instead of being subtracted anonymously. Naming it is the whole
        point — that register IS ``complex_ade_update_p``'s ``p_now``.

        THE ADE HALF IS ``complex_ade.complex_ade_update_p`` (complex_ade.py:
        103-139), its body verbatim, with exactly two loads replaced by registers
        this kernel already holds:

        * ``w_re/w_im`` — the drive. The certified body loads it from
          ``drive + word``; this family's predicate REQUIRES ``drive_field`` to be
          the stored ``E`` (complex_ade.py:184-215), which is what the E half
          stored one line earlier and already named ``o{n}_re/o{n}_im``;
        * ``p_re/p_im`` — the pole. The certified body loads it from
          ``p_now + word``; the subtraction chain above already loaded exactly
          that volume into ``pa0_re/pa0_im``.

        A float32 word stored and re-loaded from the same address is bit-identical
        to the register, so both eliminations are byte-neutral BY CONSTRUCTION —
        which is a hypothesis until a comparator agrees, and the device gate
        measures it per complete seam step against both the array path and the two
        separate certified products.

        THE ARMS ARE ORDER-INDEPENDENT AND NOTHING IS HOISTED. Every ADE arm writes
        ``p_out_{L}0``, which is that susceptibility's scratch FOR THAT COMPONENT —
        one per driven component rather than one shared per susceptibility. The
        launch's write set is therefore disjoint from its read set at every
        position of the rotation's orbit. ``fused_dispersive_chain`` must hoist
        every history load above every store because it shares one scratch and
        therefore DOES alias inside the launch (fused_dispersive_chain.py:326-329);
        this kernel does not.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        word = 2 * idx

        # --- component 0: update_E ------------------------------------------
        s0_re = tl.load(g0 + word, mask=live, other=0.0)
        s0_im = tl.load(g0 + word + 1, mask=live, other=0.0)
        if NP0 > 0:
            pa0_re = tl.load(a0 + word, mask=live, other=0.0)
            s0_re = s0_re - pa0_re
            pa0_im = tl.load(a0 + word + 1, mask=live, other=0.0)
            s0_im = s0_im - pa0_im
        if NP0 > 1:
            s0_re = s0_re - tl.load(a1 + word, mask=live, other=0.0)
            s0_im = s0_im - tl.load(a1 + word + 1, mask=live, other=0.0)
        if NP0 > 2:
            s0_re = s0_re - tl.load(a2 + word, mask=live, other=0.0)
            s0_im = s0_im - tl.load(a2 + word + 1, mask=live, other=0.0)
        if NP0 > 3:
            s0_re = s0_re - tl.load(a3 + word, mask=live, other=0.0)
            s0_im = s0_im - tl.load(a3 + word + 1, mask=live, other=0.0)
        if NP0 > 4:
            s0_re = s0_re - tl.load(a4 + word, mask=live, other=0.0)
            s0_im = s0_im - tl.load(a4 + word + 1, mask=live, other=0.0)
        if NP0 > 5:
            s0_re = s0_re - tl.load(a5 + word, mask=live, other=0.0)
            s0_im = s0_im - tl.load(a5 + word + 1, mask=live, other=0.0)
        if NP0 > 6:
            s0_re = s0_re - tl.load(a6 + word, mask=live, other=0.0)
            s0_im = s0_im - tl.load(a6 + word + 1, mask=live, other=0.0)
        if NP0 > 7:
            s0_re = s0_re - tl.load(a7 + word, mask=live, other=0.0)
            s0_im = s0_im - tl.load(a7 + word + 1, mask=live, other=0.0)
        o0_re, o0_im = _mul_field_left(
            s0_re, s0_im, tl.load(e0 + idx, mask=live, other=0.0), EXPANSION)
        tl.store(f0 + word, o0_re, mask=live)
        tl.store(f0 + word + 1, o0_im, mask=live)

        # --- component 0: update_P ------------------------------------------
        if NP0 > 0:
            p_re = pa0_re
            p_im = pa0_im
            q_re = tl.load(p_prev_a0 + word, mask=live, other=0.0)
            q_im = tl.load(p_prev_a0 + word + 1, mask=live, other=0.0)
            w_re = o0_re
            w_im = o0_im

            # dispersion.py:686-688, including operand orientation and grouping.
            a_re, a_im = _mul_field_left(p_re, p_im, cnow_a0, EXPANSION)
            b_re, b_im = _mul_coefficient_left(cprev_a0, q_re, q_im, EXPANSION)
            out_re = a_re + b_re
            out_im = a_im + b_im

            if SV_a0:
                s = tl.load(sigma_a0 + idx, mask=live, other=0.0)
            else:
                s = sigma_a0
            sw_re, sw_im = _mul_coefficient_left(s, w_re, w_im, EXPANSION)
            d_re, d_im = _mul_coefficient_left(cdrive_a0, sw_re, sw_im, EXPANSION)
            out_re = out_re + d_re
            out_im = out_im + d_im

            tl.store(p_out_a0 + word, out_re, mask=live)
            tl.store(p_out_a0 + word + 1, out_im, mask=live)

        # --- component 1: update_E ------------------------------------------
        s1_re = tl.load(g1 + word, mask=live, other=0.0)
        s1_im = tl.load(g1 + word + 1, mask=live, other=0.0)
        if NP1 > 0:
            pb0_re = tl.load(b0 + word, mask=live, other=0.0)
            s1_re = s1_re - pb0_re
            pb0_im = tl.load(b0 + word + 1, mask=live, other=0.0)
            s1_im = s1_im - pb0_im
        if NP1 > 1:
            s1_re = s1_re - tl.load(b1 + word, mask=live, other=0.0)
            s1_im = s1_im - tl.load(b1 + word + 1, mask=live, other=0.0)
        if NP1 > 2:
            s1_re = s1_re - tl.load(b2 + word, mask=live, other=0.0)
            s1_im = s1_im - tl.load(b2 + word + 1, mask=live, other=0.0)
        if NP1 > 3:
            s1_re = s1_re - tl.load(b3 + word, mask=live, other=0.0)
            s1_im = s1_im - tl.load(b3 + word + 1, mask=live, other=0.0)
        if NP1 > 4:
            s1_re = s1_re - tl.load(b4 + word, mask=live, other=0.0)
            s1_im = s1_im - tl.load(b4 + word + 1, mask=live, other=0.0)
        if NP1 > 5:
            s1_re = s1_re - tl.load(b5 + word, mask=live, other=0.0)
            s1_im = s1_im - tl.load(b5 + word + 1, mask=live, other=0.0)
        if NP1 > 6:
            s1_re = s1_re - tl.load(b6 + word, mask=live, other=0.0)
            s1_im = s1_im - tl.load(b6 + word + 1, mask=live, other=0.0)
        if NP1 > 7:
            s1_re = s1_re - tl.load(b7 + word, mask=live, other=0.0)
            s1_im = s1_im - tl.load(b7 + word + 1, mask=live, other=0.0)
        o1_re, o1_im = _mul_field_left(
            s1_re, s1_im, tl.load(e1 + idx, mask=live, other=0.0), EXPANSION)
        tl.store(f1 + word, o1_re, mask=live)
        tl.store(f1 + word + 1, o1_im, mask=live)

        # --- component 1: update_P ------------------------------------------
        if NP1 > 0:
            p_re = pb0_re
            p_im = pb0_im
            q_re = tl.load(p_prev_b0 + word, mask=live, other=0.0)
            q_im = tl.load(p_prev_b0 + word + 1, mask=live, other=0.0)
            w_re = o1_re
            w_im = o1_im

            # dispersion.py:686-688, including operand orientation and grouping.
            a_re, a_im = _mul_field_left(p_re, p_im, cnow_b0, EXPANSION)
            b_re, b_im = _mul_coefficient_left(cprev_b0, q_re, q_im, EXPANSION)
            out_re = a_re + b_re
            out_im = a_im + b_im

            if SV_b0:
                s = tl.load(sigma_b0 + idx, mask=live, other=0.0)
            else:
                s = sigma_b0
            sw_re, sw_im = _mul_coefficient_left(s, w_re, w_im, EXPANSION)
            d_re, d_im = _mul_coefficient_left(cdrive_b0, sw_re, sw_im, EXPANSION)
            out_re = out_re + d_re
            out_im = out_im + d_im

            tl.store(p_out_b0 + word, out_re, mask=live)
            tl.store(p_out_b0 + word + 1, out_im, mask=live)

        # --- component 2: update_E ------------------------------------------
        s2_re = tl.load(g2 + word, mask=live, other=0.0)
        s2_im = tl.load(g2 + word + 1, mask=live, other=0.0)
        if NP2 > 0:
            pc0_re = tl.load(c0 + word, mask=live, other=0.0)
            s2_re = s2_re - pc0_re
            pc0_im = tl.load(c0 + word + 1, mask=live, other=0.0)
            s2_im = s2_im - pc0_im
        if NP2 > 1:
            s2_re = s2_re - tl.load(c1 + word, mask=live, other=0.0)
            s2_im = s2_im - tl.load(c1 + word + 1, mask=live, other=0.0)
        if NP2 > 2:
            s2_re = s2_re - tl.load(c2 + word, mask=live, other=0.0)
            s2_im = s2_im - tl.load(c2 + word + 1, mask=live, other=0.0)
        if NP2 > 3:
            s2_re = s2_re - tl.load(c3 + word, mask=live, other=0.0)
            s2_im = s2_im - tl.load(c3 + word + 1, mask=live, other=0.0)
        if NP2 > 4:
            s2_re = s2_re - tl.load(c4 + word, mask=live, other=0.0)
            s2_im = s2_im - tl.load(c4 + word + 1, mask=live, other=0.0)
        if NP2 > 5:
            s2_re = s2_re - tl.load(c5 + word, mask=live, other=0.0)
            s2_im = s2_im - tl.load(c5 + word + 1, mask=live, other=0.0)
        if NP2 > 6:
            s2_re = s2_re - tl.load(c6 + word, mask=live, other=0.0)
            s2_im = s2_im - tl.load(c6 + word + 1, mask=live, other=0.0)
        if NP2 > 7:
            s2_re = s2_re - tl.load(c7 + word, mask=live, other=0.0)
            s2_im = s2_im - tl.load(c7 + word + 1, mask=live, other=0.0)
        o2_re, o2_im = _mul_field_left(
            s2_re, s2_im, tl.load(e2 + idx, mask=live, other=0.0), EXPANSION)
        tl.store(f2 + word, o2_re, mask=live)
        tl.store(f2 + word + 1, o2_im, mask=live)

        # --- component 2: update_P ------------------------------------------
        if NP2 > 0:
            p_re = pc0_re
            p_im = pc0_im
            q_re = tl.load(p_prev_c0 + word, mask=live, other=0.0)
            q_im = tl.load(p_prev_c0 + word + 1, mask=live, other=0.0)
            w_re = o2_re
            w_im = o2_im

            # dispersion.py:686-688, including operand orientation and grouping.
            a_re, a_im = _mul_field_left(p_re, p_im, cnow_c0, EXPANSION)
            b_re, b_im = _mul_coefficient_left(cprev_c0, q_re, q_im, EXPANSION)
            out_re = a_re + b_re
            out_im = a_im + b_im

            if SV_c0:
                s = tl.load(sigma_c0 + idx, mask=live, other=0.0)
            else:
                s = sigma_c0
            sw_re, sw_im = _mul_coefficient_left(s, w_re, w_im, EXPANSION)
            d_re, d_im = _mul_coefficient_left(cdrive_c0, sw_re, sw_im, EXPANSION)
            out_re = out_re + d_re
            out_im = out_im + d_im

            tl.store(p_out_c0 + word, out_re, mask=live)
            tl.store(p_out_c0 + word + 1, out_im, mask=live)

else:  # pragma: no cover - the laptop path
    complex_fused_ade_chain_step = _UnavailableKernel(  # type: ignore[assignment]
        "complex_fused_ade_chain_step", _TRITON_IMPORT_ERROR)


def complex_fused_ade_chain_kernel() -> Any:
    """The shipped kernel, or a raise naming the missing dependency."""
    if triton is None:
        raise RuntimeError(
            "Triton is not importable; the complex fused E->P chain has no host "
            f"fallback ({_TRITON_IMPORT_ERROR!r})")
    return complex_fused_ade_chain_step


# ---------------------------------------------------------------------------
# The expansion, resolved ONCE for both halves
# ---------------------------------------------------------------------------


def resolve_chain_expansion(probe: Any = None) -> Tuple[Optional[int], List[str]]:
    """The ONE ``EXPANSION`` constexpr both halves must accept, or the refusals.

    THE TWO CERTIFIED HALVES ASK THE SAME PROBE RECORD FOR DIFFERENT PATTERN SETS
    (see the module docstring). This is the single place the fused constexpr comes
    from, and it takes the arm only when BOTH resolvers return one AND they agree.
    A disagreement is a refusal by name, never a silent choice of one half's arm.
    """
    reasons: List[str] = []
    reasons.extend(f"E half: {reason}"
                   for reason in _complex_e._expansion_reasons(probe))
    reasons.extend(f"ADE half: {reason}"
                   for reason in _complex_ade._complex_ade_expansion_reasons(probe))
    if reasons:
        return None, reasons

    electric = _complex_e._resolve_expansion(probe)
    ade = _complex_ade._resolve_complex_ade_expansion(probe)
    if electric is None or ade is None:  # pragma: no cover - the reasons above
        return None, ["the expansion probe licensed no arm for one of the halves"]
    if int(electric) != int(ade):
        return None, [
            f"the probe licenses EXPANSION={electric} for the complex E half "
            f"(complex_fields.PROBE_PATTERNS) and EXPANSION={ade} for the complex "
            f"ADE half (complex_ade.COMPLEX_ADE_PROBE_PATTERNS); one fused source "
            f"carries ONE arm, so a platform that splits them earns a new product "
            f"rather than a silent downgrade of one half"]
    return int(electric), []


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


def complex_fused_ade_chain_coverage(fields: Any, pml: Any,
                                     probe: Any = None) -> Coverage:
    """May ONE launch span complex ``update_E`` -> complex ``update_P`` here?

    THERE IS NO SOURCE CLAUSE, AND THE ABSENCE IS DELIBERATE. Every other fused
    product on this backend must be told the source inventory, because the driver
    injects INSIDE its seam and ignorance is not an empty set. This seam has no
    injection in it at all — ``driver.step`` runs ``update_E`` at :3304 and
    ``update_P`` at :3306 with nothing between — so requiring a declaration would
    refuse configurations the driver cannot break. The fusion matrices measure the
    same fact from the other end: ``E->P: 15 instances, 0 blocked by the source
    seam``.
    """
    reasons: List[str] = []

    electric = _complex_e.complex_stored_e_coverage(fields, pml, probe=probe)
    if not electric.covered:
        reasons.extend(f"E half: {reason}" for reason in electric.reasons)

    states = tuple(getattr(fields, "polarizations", ()) or ())
    if not states:
        reasons.append(
            "no susceptibility is registered: with nothing to advance this product "
            "spans one sub-step, not two, and that configuration is the E arm's own")

    try:
        order = _complex_e.poles_per_component(fields)
    except Exception as exc:  # noqa: BLE001 - an unreadable partition is a refusal
        return Coverage(False, tuple(dict.fromkeys(
            reasons + [f"the pole partition is unreadable ({exc!r})"])))

    for index, state in enumerate(states):
        driven = _coverage._call(state, "driven", default=None)
        if driven is None:
            reasons.append(
                f"polarization {index} ({type(state).__name__}) does not report "
                f"driven(); an unreadable susceptibility is not a covered one")
            continue
        for component in tuple(driven):
            verdict = _complex_ade.complex_ade_update_p_coverage(
                fields, pml, state, component, probe=probe)
            if not verdict.covered:
                reasons.extend(f"ADE half {index}.{component}: {reason}"
                               for reason in verdict.reasons)
        # THE TWO HALVES MUST AGREE ABOUT WHICH COMPONENTS THIS STATE DRIVES.
        # The E chain subtracts wherever the pole partition lists this state and
        # the ADE half advances wherever `driven()` names it; the two are one
        # attribute in the reference (dispersion.py:652-656) and a disagreement
        # would subtract a pole this launch never advances — a frozen P, smooth
        # and wrong.
        subtracted = tuple(component for component, _displacement in E_TERMS
                           if any(state is other
                                  for other in order.get(component, ())))
        named = tuple(name for name in tuple(driven)
                      if name in {term[0] for term in E_TERMS})
        if named != subtracted:
            reasons.append(
                f"polarization {index}: driven()={tuple(driven)!r} disagrees with "
                f"the components whose D - sum P chain subtracts it ({subtracted!r})")

    for component, _displacement in E_TERMS:
        count = len(order.get(component, ()))
        if count > CHAIN_MAX_POLES:
            reasons.append(
                f"{component} is driven by {count} poles; this product binds "
                f"{CHAIN_MAX_POLES} ADE arm per component, and a pole the E chain "
                f"subtracts but the ADE half never advances is a frozen P")
        for slot, state in enumerate(order.get(component, ())):
            pole = (getattr(state, "P", {}) or {}).get(component)
            if pole is None:
                continue  # already reported by the ADE half
            if (getattr(state, "P", {}) or {}).get(component) is not pole:
                reasons.append(
                    f"{component} slot {slot}: the subtracted pole volume is not "
                    f"the state's own P; the register the ADE arm reuses would be "
                    f"a different array")

    # THE SEAM'S OWN CONDITION, named rather than inherited: the register this
    # kernel hands the recurrence is what its E half stored, so `drive_field` must
    # return THAT array. The complex ADE half already asserts the stored-E identity
    # from its own side (complex_ade.py:184-215); this restates the requirement
    # from the E side so a `Fields` in PML storage mode is refused HERE too rather
    # than only through a per-component clause.
    if bool(getattr(fields, "_pml_active", False)):
        reasons.append(
            "Fields is in PML storage mode: drive_field would return f_w rather "
            "than the stored E (fields.py:1160-1162), and the register this kernel "
            "passes the recurrence is the value it stored in E")

    # ONE EXPANSION ARM FOR ONE FUSED SOURCE. This is the clause that is genuinely
    # this product's own and not either half's; see the module docstring.
    _arm, expansion_reasons = resolve_chain_expansion(probe)
    reasons.extend(expansion_reasons)

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_complex_fused_ade_chain(fields: Any, pml: Any,
                                    probe: Any = None) -> Coverage:
    """The verdict with its reasons, for reports. Needs no Triton."""
    return complex_fused_ade_chain_coverage(fields, pml, probe=probe)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


class ComplexFusedAdeChainPlan:
    """ONE launch that performs two driver passes, plus the reference rotation.

    ``launches_per_run`` is ONE and is declared rather than inherited: a whole-step
    arbiter asserts the exact per-cycle launch count on every filled slot, and that
    assertion is what stops a slot passing by not executing.

    THE ROTATION IS PERFORMED HERE, AFTER THE LAUNCH, and it is
    ``dispersion.PolarizationState.update``'s (dispersion.py:687-691) over a
    DIFFERENT buffer set: one scratch per driven component rather than one shared
    per susceptibility. The three-cycle each (susceptibility, component) walks is
    the reference's; what changes is that no two of them share a buffer, which is
    what makes the launch's write set disjoint from its read set. See the module
    docstring for the orbit enumeration that licenses the shape and for why that
    enumeration transfers to complex64 unchanged.

    THE POINTERS ARE RESOLVED PER LAUNCH, DELIBERATELY. The rotation moves which
    allocation holds ``P``, ``P_prev`` and each scratch on every step, so a plan
    that cached them would be stale after the FIRST step — and stale in a way that
    still computes, giving a smooth wrong field. The ORDER is cached (it is fixed
    for the run) and re-checked every call by the E arm's own
    ``LiveComplexPoleBinding``.

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

    __slots__ = ("fields", "arm", "shape", "n_elem", "block", "num_warps", "counts",
                 "expansion", "extra_scratch_volumes", "launches", "runs",
                 "alias_checks", "_pointer", "_targets", "_sources", "_inv_eps",
                 "_poles", "_order", "_states", "_index_of", "_scratch",
                 "_first_driven", "_sigma", "_coeff", "_sv", "_kernel", "_grid",
                 "_read_ids")

    def __init__(self, fields: Any, pml: Any, block: int,
                 expansion: int, num_warps: Optional[int] = 1,
                 kernel: Any = None, pointer: Any = None) -> None:
        from .launch import CupyPointer  # noqa: PLC0415

        if pointer is None:
            pointer = CupyPointer
        self.fields = fields
        self.arm = ARM
        self.shape = tuple(int(n) for n in fields.grid.shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self.num_warps = num_warps
        self.expansion = int(expansion)
        self._pointer = pointer
        self._kernel = kernel
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self.launches = 0
        self.runs = 0
        self.alias_checks = 0

        components = tuple(term[0] for term in E_TERMS)
        targets = [getattr(fields, name) for name in components]
        sources = [getattr(fields, term[1]) for term in E_TERMS]
        inverse = [fields.inverse_epsilon_for(name) for name in components]
        # COMPLEX VOLUMES GO IN AS THEIR FLOAT32 WORD VIEW and the inverse epsilon
        # does NOT — that asymmetry is the certified E plan's own
        # (complex_no_pml_stored_e.py:359-362) and the kernel indexes the two
        # differently (`word` versus `idx`).
        self._targets = tuple(pointer(_word_view(array)) for array in targets)
        self._sources = tuple(pointer(_word_view(array)) for array in sources)
        self._inv_eps = tuple(pointer(array) for array in inverse)
        read_ids = {id(array) for array in targets + sources + inverse}

        partition = _complex_e.poles_per_component(fields)
        self._order = {name: tuple(states) for name, states in partition.items()}
        self._poles = _complex_e.LiveComplexPoleBinding(fields, self._order)
        self.counts = tuple(len(self._order[name]) for name in components)
        for name, count in zip(components, self.counts):
            if count > CHAIN_MAX_POLES:
                raise ValueError(
                    f"{name} is driven by {count} poles; this product binds "
                    f"{CHAIN_MAX_POLES} ADE arm per component and the predicate "
                    f"refuses more")

        self._states = tuple(getattr(fields, "polarizations", ()) or ())
        self._index_of = {id(state): index
                          for index, state in enumerate(self._states)}

        # ONE SCRATCH PER DRIVEN COMPONENT — the shape, allocated. The FIRST driven
        # component takes the susceptibility's own ``_scratch`` and the rest are
        # new, so the extra is ``K * (d - 1)`` volumes and not ``K * d``.
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
        self._coeff: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
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
                # step reads a scalar as a pointer, which is a wrong answer and not
                # a crash (coverage.sigma_is_volume).
                volume = bool(_coverage.sigma_is_volume(state, name))
                sigma = state.sigma[name]
                self._sigma[(name, slot)] = (pointer(sigma) if volume
                                             else float(sigma))
                if volume:
                    read_ids.add(id(sigma))
                c_now, c_prev, c_drive = state._coefficients
                self._coeff[(name, slot)] = (float(c_now), float(c_prev),
                                             float(c_drive))
                self._sv[f"SV_{letter}{slot}"] = 1 if volume else 0
        self._read_ids = frozenset(read_ids)

    # -- one launch ---------------------------------------------------------

    def _resolve(self, groups: Sequence[Sequence[Any]]) -> List[List[Tuple[Any, ...]]]:
        """The destination chain for every component, resolved BEFORE the launch.

        ``(key, state, out, pole, prev)`` per live slot. The pole is taken from the
        E arm's own live binding and CHECKED to be the state's ``P`` — the register
        the ADE arm reuses is that load, so a disagreement would advance one pole's
        recurrence from another pole's history.
        """
        rows: List[List[Tuple[Any, ...]]] = []
        for index, (component, _displacement) in enumerate(E_TERMS):
            entries: List[Tuple[Any, ...]] = []
            for slot, state in enumerate(self._order[component]):
                pole = state.P[component]
                if pole is not groups[index][slot]:
                    raise RuntimeError(
                        f"{component} slot {slot}: the ADE arm's pole is not the "
                        f"array the subtraction slot resolved; the reused register "
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
        into an invariant this plan cannot violate silently.
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

    def _pole_slots(self, groups: Sequence[Sequence[Any]]) -> List[Any]:
        """The E half's pole pointers, padded to :data:`MAX_POLES` per component."""
        pointer = self._pointer
        slots: List[Any] = []
        for index, group in enumerate(groups):
            if len(group) != self.counts[index]:
                raise RuntimeError(
                    f"component {index} resolved {len(group)} poles, not the "
                    f"compiled count {self.counts[index]}")
            slots.extend(pointer(_word_view(array)) for array in group)
            # Unused slots take this component's own D pointer, exactly as the
            # certified complex E plan pads them
            # (complex_no_pml_stored_e.py:381-383).
            slots.extend([self._sources[index]] * (MAX_POLES - len(group)))
        return slots

    def _arguments(self, slots: Sequence[Any],
                   chain: Sequence[Sequence[Tuple[Any, ...]]]) -> List[Any]:
        """Every runtime argument, in the order the signature declares them.

        SEPARATE FROM :meth:`run` so the ORDER is testable on a host with no
        Triton: a launcher that assembles the right pointers in the wrong order
        binds each one to a neighbouring slot, which is a wrong answer and not a
        crash. The count is asserted against :data:`RUNTIME_ARGUMENTS` here and
        that constant is asserted against the kernel's own signature by the test.
        """
        pointer = self._pointer
        ade: List[Any] = []
        for index, entries in enumerate(chain):
            component = E_TERMS[index][0]
            padding = self._sources[index]
            live = len(entries)
            for slot in range(CHAIN_MAX_POLES):
                if slot < live:
                    row = entries[slot]
                    ade.append(pointer(_word_view(row[2])))
                    ade.append(pointer(_word_view(row[4])))
                    ade.append(self._sigma[(component, slot)])
                    ade.extend(self._coeff[(component, slot)])
                else:
                    ade.extend([padding, padding, 0.0, 0.0, 0.0, 0.0])
        arguments = [*self._targets, *self._sources, *self._inv_eps,
                     *slots, *ade, self.n_elem]
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
        :class:`complex_ade.ComplexAdeUpdatePPlan` uses and for the same reason: a
        plan that rotated first and then raised would leave the engine holding a P
        history that never happened.

        THE FUSION GUARD IS THE SHARED SPELLING, not the deferred one the two
        certified halves use. Their exemption exists because their own laptop tests
        call ``run(guard=False)`` on a host with no Triton, so importing
        ``kernels`` there would raise; nothing in this family's suite calls ``run``
        at all, so it takes the invariant's plain form
        (``test_triton_kernels.test_every_launch_site_passes_the_shared_guard_
        constant``, which counts EVERY occurrence in the package and would fail on
        a third module quietly adopting the exemption).
        """
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else complex_fused_ade_chain_kernel())
        groups = self._poles.arrays()
        chain = self._resolve(groups)
        self._check_aliasing(chain)
        arguments = self._arguments(self._pole_slots(groups), chain)
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *arguments,
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            EXPANSION=self.expansion, BLOCK=self.block, **self._sv,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        self.launches += 1
        self._rotate(chain)
        self.runs += 1

    def __repr__(self) -> str:
        return (f"ComplexFusedAdeChainPlan(shape={self.shape}, "
                f"poles={self.counts}, expansion={self.expansion}, "
                f"extra_scratch={self.extra_scratch_volumes}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_complex_fused_ade_chain(fields: Any, pml: Any,
                                 block: Optional[int] = None,
                                 num_warps: Optional[int] = 1,
                                 kernel: Any = None,
                                 probe: Any = None
                                 ) -> Optional[ComplexFusedAdeChainPlan]:
    """Build the complex fused E->P plan, or return ``None`` on any refusal.

    ``None`` means REFUSED and the reasons are available from
    :func:`complex_fused_ade_chain_coverage`. The Triton import stays BELOW the
    predicate, as in every builder in this package: a NumPy host must be able to
    plan (to ``None``) without the optional dependency being importable at all.

    ``kernel=`` IS LOAD-BEARING: the gate's mutation legs route a mutated kernel
    through it, and a builder that dropped it would launch the SHIPPED kernel and
    report a pass for a defect it never introduced.
    """
    if not complex_fused_ade_chain_coverage(fields, pml, probe=probe).covered:
        return None
    expansion, _reasons = resolve_chain_expansion(probe)
    if expansion is None:  # pragma: no cover - coverage refused the same record
        return None
    return ComplexFusedAdeChainPlan(
        fields, pml, DEFAULT_BLOCK if block is None else block, expansion,
        num_warps=num_warps, kernel=kernel)


# ---------------------------------------------------------------------------
# WIRING — deferred, exactly as the other cross-sub-step products are held
# ---------------------------------------------------------------------------
#
# ``launch.plan_step`` does not import this module and ``launch.FAMILY_MODULES``
# does not name it, so nothing about any existing arm's behaviour changes and a
# default run still steps this seam on the array path. The product exists to be
# MEASURED: ``parity/meep_gpu/gate_triton_complex_fused_ade_chain.py`` is its byte
# gate. Composition is a separate change with its own disjointness argument,
# because this product spans TWO slots and the planner's arm table assigns at most
# one arm per slot.
