"""``update_E`` welded to ``update_P`` — the Triton product at the E->P seam.

THE SEAM IS EMPTY, AND THAT IS THE WHOLE REASON THIS PRODUCT EXISTS.
``driver.step`` runs::

    if fast is None or not fast.dispatch("update_E", self.fields):
        update_E(self.fields, self.pml)                       # driver.py:3303-3304
    if fast is None or not fast.dispatch("update_P", self.fields):
        update_P(self.fields, self.pml)                       # driver.py:3305-3306

Nothing sits between them: no source injection, no mirror fill, no wall clear, no
far-ghost pass. Every other seam on this board has a driver pass inside it that a
fused kernel must either carry inline or refuse; this one has none, which is why
``results/fusion_matrix_triton_2026-08-20_eop/`` records ``E->P: 15 instances,
0 blocked by the source seam, CEILING 15``.

===========================================================================
THE ROTATION QUESTION, AND WHY TRITON'S ANSWER IS NOT METAL'S
===========================================================================

``ade_update_p`` launches once per DRIVEN COMPONENT and the host rotates
``P`` / ``P_prev`` / ``_scratch`` after each launch (launch.py:1830-1834,
transcribing dispersion.py:687-691). A fused kernel must either take one
component, bake the rotation, or change the buffer set.

The Metal twin (``metal_kernels/fused_ade_chain.py``) takes the FIRST: it emits
one kernel per component and interleaves ``update_E(c)`` with ``update_P(*, c)``,
which is bit-identical to the driver's order and alias-free at every orbit
position. **That shape is not available here, and the reason was measured rather
than assumed** (``parity/meep_gpu/probe_triton_ade_rotation_shape.py`` /
``results/triton_ade_rotation_shape_2026-08-20/``, LEG 1): every certified Triton
``update_E`` writes ALL THREE components in one launch — ``stored_e_constitutive_
step`` stores to ``f0/f1/f2``, ``constitutive_step_dispersive`` to ``f0..f2`` and
``w0..w2``, ``complex_stored_e_step`` to ``f0/f1/f2``. Three parsed, none writing
one component per launch. Metal's E products are emitted per axis and dispatched
three times; Triton's are one launch, and splitting a certified body to get the
other shape would put this product's arithmetic outside what has been gated.

``fused_ade_state`` takes the SECOND — all components in one launch over one
SHARED scratch — and its own note records the cost: arm 1's OUTPUT buffer IS arm
0's ``p_prev`` INPUT, the safety of that alias is an OBSERVATION about one
toolchain rather than a language guarantee, and "Fusing further REQUIRES breaking
the chain first, not inheriting it."

THIS PRODUCT TAKES THE THIRD, WHICH IS THE ONE THAT NOTE ASKS FOR: **one scratch
per DRIVEN COMPONENT** instead of one shared per susceptibility. The same orbit
enumeration walked to closure (LEG 2, d = 1, 2, 3) finds:

    d   per-component launch   shared scratch (a)      one scratch per component
    1   orbit 3, 0 conflicts   orbit 3, 0 conflicts    orbit 3, 0 conflicts
    2   orbit 5, 0 conflicts   orbit 5, 5 conflicts    orbit 3, 0 conflicts
    3   orbit 7, 0 conflicts   orbit 7, 7 conflicts    orbit 3, 0 conflicts

The shared-scratch shape conflicts at EVERY position of the orbit for d > 1 —
that is ``fused_ade_state``'s alias, reproduced from the reference's own rotation.
The per-component-scratch shape is ALIAS-FREE AT EVERY POSITION, FOR EVERY d, and
its orbit is 3 rather than ``2d + 1`` because the permutation decomposes into d
independent 3-cycles. So this launch's write set is disjoint from its read set BY
CONSTRUCTION rather than by an observation about one compiler's reordering, and
:meth:`FusedAdeChainPlan.run` ASSERTS that disjointness before every launch rather
than trusting the argument.

WHAT IT COSTS, MEASURED (LEG 3): ``K * (d - 1)`` extra float32 field volumes — the
reference allocates one scratch per susceptibility (dispersion.py:646-650) and
this allocates one per driven component. Over the ten E->P rows this serves that
is +4 at K=2 (``material-dispersion.py``), +10 at K=5 (``absorber-1d.py``,
``TestAbsorber``) and **+12 at K=6** (``stochastic_emitter{,_line,_reciprocity}
.py``, the worst row on the board). The cost is MEMORY ONLY: no extra launch, no
extra arithmetic, and the values left in ``P`` and ``P_prev`` after the rotation
are the reference's, which is what the byte gate compares. ``extra_scratch_
volumes`` on the plan reports the number the run actually paid.

THE RETIRED SCRATCH IS THE ONE THING THAT DOES *NOT* MATCH THE REFERENCE, and it
is named here rather than discovered by a comparator. After a step the reference's
single ``_scratch`` holds the retired ``P_prev`` of the LAST driven component;
this plan's per-component scratches hold the retired ``P_prev`` of EACH. The two
agree on the last component's and cannot agree on the others — the reference
overwrote those buffers with the next component's result. They are dead state: the
next step's launch overwrites every one of them before reading anything from it.
The gate compares ``P`` and ``P_prev`` per (susceptibility, component) plus the
last component's scratch, and says so.

===========================================================================
WHAT IS FUSED, AND IT IS EXACTLY TWO LOADS PER POLE PER CELL
===========================================================================

* **the drive.** ``update_P``'s drive is ``Fields.drive_field(component)``: the
  stored E without an absorber, ``f_w_E*`` under one (fields.py:1140-1163). In
  both cases that is the value this kernel's E half just computed and stored one
  launch earlier, so ``w = drive[idx]`` becomes ``w = src{c}`` — the register;
* **the pole.** ``ade_update_p`` loads ``p_now``; the pole-aware ``D - P``
  subtraction in the E half has already loaded exactly that volume, so
  ``p = p_now[idx]`` becomes ``p = p{L}{k}``.

Neither elimination changes a rounded value: a float32 store followed by a float32
load of the same address is the identity, and the register reused is the one the
eliminated load would have produced. That is an argument for why byte identity is
EXPECTED, not evidence that it holds; the device gate is what decides it.

EVERY OTHER LINE IS TRANSCRIBED FROM THE BODY IT REPLACES, and the transcription
is CHECKED BY PARSING both sources rather than restated as data here — a second
copy of "the load-bearing lines" is one more place to drift.
``test_triton_fused_ade_chain`` reads the certified kernels' own text with ``ast``
and requires each arithmetic line to appear in this kernel modulo the documented
renames, and the device gate re-runs that check beside the bytes.

===========================================================================
SCOPE, AND WHY EACH BOUND IS A REFUSAL BY NAME
===========================================================================

* **real float32 only.** The four ``TestLoadDump.*_3d`` rows are complex64; their
  E half is ``complex_no_pml_stored_e`` and their ADE half ``complex_ade``, a
  different signature and a different arm. Refused, exactly as the Metal family
  declines them;
* **no off-diagonal chi1inv row and no chi2/chi3.** Both make ``update_E`` read
  the OTHER components' volumes (stepping.py:991-1008) and stop it being
  element-wise; both are already refused by both E-half predicates. That is what
  keeps ``absorbed_power_density.py`` out;
* **at most :data:`CHAIN_MAX_POLES` poles on one component.** The E half compiles
  :data:`MAX_POLES` = 8 slots and this product binds 6 ADE arms per component, so
  a seventh pole would be SUBTRACTED by the E chain and never ADVANCED by the ADE
  half — a smooth, plausible, wrong field. Refused by name in the predicate and
  raised by the plan. Six is the corpus maximum at this seam (K = 6 on the three
  ``stochastic_emitter`` rows) and it is where the Metal twin's binding ceiling
  lands too (31 bindings at six poles, 34 at seven), so the two backends' reach is
  bounded at the same place for two unrelated reasons;
* **a cylindrical axis, a Bloch phase, beta, BFAST, a conductivity** —
  inherited from whichever E-half predicate the arm names, unchanged and not
  re-decided here. A FOLD is the one of these that is not simply refused: the
  ``folded`` arm REQUIRES one and the other two refuse one, which is
  ``folded_dispersive_update_e``'s own disjointness and not a new rule. What no
  arm here carries is a fold on the NO-PML side — no corpus row is in that state
  (``predicate_coverage_2026-08-16_wired_convention/remaining.txt`` lists no
  fold-plus-dispersion-plus-no-PML row) and it would be a fourth admission.

NOT WIRED, and dispatch is unchanged. ``launch.plan_step`` does not import this
module and no arm table names it, exactly as ``fused_dispersive_chain`` is held:
the product is built, gated and measurable, and a default run still steps this
seam on the array path.

DEVICE RESULT
-------------
``parity/meep_gpu/gate_triton_fused_ade_chain.py`` ran on the GPU host's RTX A6000
(GPU verified physically empty by UUID) on 2026-08-21, under BOTH float32
subnormal policies, one process and one cache pair each. Nine product rows —
three arms, one to six poles, scalar and volume sigmas — were byte-identical to
the CuPy array path over SIXTY complete E->P seam steps, at one launch per step,
and identical to the separately certified sub-step products dispatched beside
them on identical state (16, 19, 7 and 16 dispatches per step collapsing to 1).
Nineteen kernel-source mutations and four host mutations were CAUGHT; five nulls
were CONFIRMED with their PTX evidence, and three of those five are this
product's own claim — re-loading the drive from ``f_w``, re-loading it from the
stored E, and re-loading the pole instead of reusing the register all compiled to
different PTX and changed no byte. The release verdict was shown to FLIP against
a planted defect. The readable entries are ``triton_fused_ade_chain_device_gate``
in ``fingerprints.json`` and ``results/triton_fused_ade_chain_2026-08-21/``. This
is a correctness result, not a throughput claim.

CORPUS REACH: ten seam-instances of the 387,
``results/fusion_matrix_triton_2026-08-21_ade_chain/`` (86 -> 96) — the same ten
rows the Metal twin serves, row for row.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import coverage as _coverage
from . import dispersive_update_e as _dispersive
from . import folded_dispersive_update_e as _folded
from . import no_pml_ade as _no_pml_ade
from . import no_pml_stored_e as _stored_e
from .coverage import Coverage

__all__ = [
    "ARMS",
    "CHAIN_MAX_POLES",
    "PML_ARMS",
    "E_TERMS",
    "MAX_POLES",
    "POLICY",
    "REPLACES",
    "RUNTIME_ARGUMENTS",
    "FusedAdeChainPlan",
    "explain_fused_ade_chain",
    "fused_ade_chain_coverage",
    "fused_ade_chain_kernel",
    "fused_ade_chain_step",
    "plan_fused_ade_chain",
]

#: The three E-side constitutive arms this family welds to the ADE recurrence.
#: ``no_pml`` is ``no_pml_stored_e``'s body and ``dispersive`` is
#: ``dispersive_update_e``'s; the kernel carries both behind the ``PML``
#: constexpr, so one specialisation per arm and nothing shared at runtime.
#:
#: ``folded`` IS NOT A THIRD BODY. It is the dispersive body under
#: ``folded_dispersive_update_e``'s ADMISSION — a real fold REQUIRED where the
#: unfolded predicate refuses one, and nothing else different. That module is
#: itself "no kernel and no arithmetic" over ``dispersive_update_e``
#: (folded_dispersive_update_e.py:64-72), and this arm inherits that relation
#: whole: same kernel, same plan class, same live pole binding, a different
#: predicate. The Metal twin splits the same three ways and for the same reason.
ARMS: Tuple[str, str, str] = ("no_pml", "dispersive", "folded")

#: Which arms drive the kernel's ``PML`` constexpr, and therefore bind ``f_w``
#: and the split-field coefficient columns. Declared as data so the plan, the
#: predicate and the binding branch cannot disagree about it.
PML_ARMS: Tuple[str, str] = ("dispersive", "folded")

#: The driver passes one run of this plan performs, in driver order
#: (driver.py:3303-3306). Declared rather than inferred from a slot name.
REPLACES: Tuple[str, ...] = ("update_E", "update_P")

#: The E half's compiled pole slots. EIGHT, because that is what both certified
#: bodies compile (``no_pml_stored_e.MAX_POLES``, ``dispersive_update_e.MAX_POLES``)
#: and the subtraction chain below is their chain — pinned equal to both by
#: ``test_triton_fused_ade_chain``, which is the honest way to keep three
#: independently-written bodies in step. NOT imported from either: an import lets
#: that module's edit silently truncate this kernel's chain.
MAX_POLES = 8

#: The ADE half's live slots per component, and the bound the predicate refuses
#: past. SIX is the corpus maximum at this seam — Ag, the only real multi-pole
#: material, carries 1 Drude + 5 Lorentz on the three ``stochastic_emitter`` rows —
#: and every extra slot costs SIX more kernel arguments per component. A seventh
#: pole is refused BY NAME rather than truncated: the E chain would subtract it and
#: the ADE half would never advance it.
CHAIN_MAX_POLES = 6

#: ``(component, displacement, axis)`` — ``dispersive_update_e.E_TERMS``, whose
#: axis column the PML tail indexes. Pinned equal to it by test rather than
#: imported, for :data:`MAX_POLES`'s reason.
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

#: The settled launch configuration. ONE WARP is the shipped policy for every
#: cross-sub-step fused product (``fused_dispersive_chain.POLICY``, measured
#: 1.0739x median over 12 shape/boundary cases). ``enable_fp_fusion`` is off
#: engine-wide (``kernels.ENABLE_FP_FUSION``) and is what keeps this new seam from
#: contracting a multiply and an add into an FMA. This product does not tune.
POLICY: Dict[str, Any] = {
    "num_warps": 1,
    "block": "kernels.DEFAULT_BLOCK",
    "enable_fp_fusion": "kernels.ENABLE_FP_FUSION (False)",
    "status": "keep",
}

#: Runtime (non-constexpr) parameters :func:`fused_ade_chain_step` declares:
#: three targets, three auxiliaries, three sources, three inverse epsilons, the
#: E half's pole slots, six PML coefficient columns, six ADE arguments per live
#: slot per component, and the four geometry scalars. Asserted against the
#: kernel's OWN signature by the laptop test and against the assembled argument
#: list at EVERY launch, so a signature edit that the launcher does not follow
#: fails at the call rather than binding the wrong pointer to the wrong slot.
RUNTIME_ARGUMENTS = (3 + 3 + 3 + 3 + 3 * MAX_POLES + 6
                     + 3 * 6 * CHAIN_MAX_POLES + 4)


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
    def fused_ade_chain_step(
        f0, f1, f2,                     # targets:      Ex, Ey, Ez          (in/out)
        w0, w1, w2,                     # auxiliaries:  f_w_Ex/y/z          (in/out)
        g0, g1, g2,                     # sources:      Dx, Dy, Dz          (in)
        e0, e1, e2,                     # inverse epsilon, per component    (in)
        a0, a1, a2, a3, a4, a5, a6, a7,   # component 0's poles, IN ORDER
        b0, b1, b2, b3, b4, b5, b6, b7,   # component 1's poles, IN ORDER
        c0, c1, c2, c3, c4, c5, c6, c7,   # component 2's poles, IN ORDER
        kp0, km0, kp1, km1, kp2, km2,   # kps/kms on each component's OWN axis
        # component 0's ADE arms: p_out / p_prev / sigma / the coefficient triple
        p_out_a0, p_out_a1, p_out_a2, p_out_a3, p_out_a4, p_out_a5,
        p_prev_a0, p_prev_a1, p_prev_a2, p_prev_a3, p_prev_a4, p_prev_a5,
        sigma_a0, sigma_a1, sigma_a2, sigma_a3, sigma_a4, sigma_a5,
        cnow_a0, cnow_a1, cnow_a2, cnow_a3, cnow_a4, cnow_a5,
        cprev_a0, cprev_a1, cprev_a2, cprev_a3, cprev_a4, cprev_a5,
        cdrive_a0, cdrive_a1, cdrive_a2, cdrive_a3, cdrive_a4, cdrive_a5,
        # component 1's ADE arms: p_out / p_prev / sigma / the coefficient triple
        p_out_b0, p_out_b1, p_out_b2, p_out_b3, p_out_b4, p_out_b5,
        p_prev_b0, p_prev_b1, p_prev_b2, p_prev_b3, p_prev_b4, p_prev_b5,
        sigma_b0, sigma_b1, sigma_b2, sigma_b3, sigma_b4, sigma_b5,
        cnow_b0, cnow_b1, cnow_b2, cnow_b3, cnow_b4, cnow_b5,
        cprev_b0, cprev_b1, cprev_b2, cprev_b3, cprev_b4, cprev_b5,
        cdrive_b0, cdrive_b1, cdrive_b2, cdrive_b3, cdrive_b4, cdrive_b5,
        # component 2's ADE arms: p_out / p_prev / sigma / the coefficient triple
        p_out_c0, p_out_c1, p_out_c2, p_out_c3, p_out_c4, p_out_c5,
        p_prev_c0, p_prev_c1, p_prev_c2, p_prev_c3, p_prev_c4, p_prev_c5,
        sigma_c0, sigma_c1, sigma_c2, sigma_c3, sigma_c4, sigma_c5,
        cnow_c0, cnow_c1, cnow_c2, cnow_c3, cnow_c4, cnow_c5,
        cprev_c0, cprev_c1, cprev_c2, cprev_c3, cprev_c4, cprev_c5,
        cdrive_c0, cdrive_c1, cdrive_c2, cdrive_c3, cdrive_c4, cdrive_c5,
        nx, ny, nz, n_elem,
        NP0: tl.constexpr, NP1: tl.constexpr, NP2: tl.constexpr,
        PML: tl.constexpr,
        SV_a0: tl.constexpr, SV_a1: tl.constexpr, SV_a2: tl.constexpr,
        SV_a3: tl.constexpr, SV_a4: tl.constexpr, SV_a5: tl.constexpr,
        SV_b0: tl.constexpr, SV_b1: tl.constexpr, SV_b2: tl.constexpr,
        SV_b3: tl.constexpr, SV_b4: tl.constexpr, SV_b5: tl.constexpr,
        SV_c0: tl.constexpr, SV_c1: tl.constexpr, SV_c2: tl.constexpr,
        SV_c3: tl.constexpr, SV_c4: tl.constexpr, SV_c5: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``update_E`` welded to ``update_P``: both driver passes, ONE grid.

        THE E HALF IS THE CERTIFIED BODY. ``PML`` selects which one: with it set
        the lines are ``dispersive_update_e.constitutive_step_dispersive``'s
        (dispersive_update_e.py:207-230 per component), with it clear they are
        ``no_pml_stored_e.stored_e_constitutive_step``'s (no_pml_stored_e.py:
        233-250). Both chains are unrolled by the SAME ``if NP > n`` constexpr
        chain those kernels carry, over the same eight slots, so a component with
        no pole compiles to exactly its certified non-dispersive body.

        TWO REGISTERS ARE NAMED THAT THE CERTIFIED NO-PML BODY DID NOT NAME, and
        they are the whole fusion:

        * ``src{c}`` — the constitutive product. The dispersive body already names
          it (:225) one line before storing it into ``f_w``; the no-PML body stores
          it unnamed (no_pml_stored_e.py:250). Named, it IS the drive
          ``ade_update_p`` would have re-loaded, on BOTH arms, because
          ``Fields.drive_field`` returns ``f_w`` under an absorber and the stored E
          without one (fields.py:1140-1163) and this kernel wrote whichever one
          applies one line earlier;
        * ``p{L}{k}`` — the pole the subtraction chain already loaded, which is
          ``ade_update_p``'s ``p_now``. ``fused_dispersive_chain`` names slot 0's
          load for the same reason (fused_dispersive_chain.py:248); here every
          live slot is named, because every one of them drives an ADE arm.

        A float32 value stored and re-loaded is bit-identical to the register, so
        both eliminations are byte-neutral BY CONSTRUCTION — which is a hypothesis
        until a comparator agrees, and the device gate measures it per complete
        seam step against both the array path and the separate certified products.

        THE ARMS ARE ORDER-INDEPENDENT AND NOTHING IS HOISTED. Every ADE arm
        writes ``p_out_{L}{k}``, which is that susceptibility's scratch FOR THAT
        COMPONENT — one per driven component rather than one shared per
        susceptibility. The launch's write set is therefore disjoint from its read
        set at every position of the rotation's orbit, for every pole count, which
        is what ``parity/meep_gpu/probe_triton_ade_rotation_shape.py`` walked to
        closure. ``fused_dispersive_chain`` must hoist every history load above
        every store because it shares one scratch and therefore DOES alias inside
        the launch (fused_dispersive_chain.py:326-329); this kernel does not, and
        the arms are emitted per component to bound live registers instead.

        NO ``nx``. It is taken and not used, exactly as the certified dispersive
        body takes and does not use it (dispersive_update_e.py:167): ``i`` comes
        from ``plane // ny``. The parameter is kept so the geometry block below
        is that body's own text.
        """
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = idx < n_elem
        if PML:
            k = idx % nz
            plane = idx // nz
            j = plane % ny
            i = plane // ny

            # Component 0 takes its coefficient from axis x, 1 from y, 2 from z.
            kp_0 = tl.load(kp0 + i, mask=live, other=0.0)
            km_0 = tl.load(km0 + i, mask=live, other=0.0)
            kp_1 = tl.load(kp1 + j, mask=live, other=0.0)
            km_1 = tl.load(km1 + j, mask=live, other=0.0)
            kp_2 = tl.load(kp2 + k, mask=live, other=0.0)
            km_2 = tl.load(km2 + k, mask=live, other=0.0)

        # --- component 0: update_E --------------------------------------------
        if PML:
            prev0 = tl.load(w0 + idx, mask=live, other=0.0)   # BEFORE the store.
        s0 = tl.load(g0 + idx, mask=live, other=0.0)
        if NP0 > 0:
            pa0 = tl.load(a0 + idx, mask=live, other=0.0)
            s0 = s0 - pa0
        if NP0 > 1:
            pa1 = tl.load(a1 + idx, mask=live, other=0.0)
            s0 = s0 - pa1
        if NP0 > 2:
            pa2 = tl.load(a2 + idx, mask=live, other=0.0)
            s0 = s0 - pa2
        if NP0 > 3:
            pa3 = tl.load(a3 + idx, mask=live, other=0.0)
            s0 = s0 - pa3
        if NP0 > 4:
            pa4 = tl.load(a4 + idx, mask=live, other=0.0)
            s0 = s0 - pa4
        if NP0 > 5:
            pa5 = tl.load(a5 + idx, mask=live, other=0.0)
            s0 = s0 - pa5
        if NP0 > 6:
            pa6 = tl.load(a6 + idx, mask=live, other=0.0)
            s0 = s0 - pa6
        if NP0 > 7:
            pa7 = tl.load(a7 + idx, mask=live, other=0.0)
            s0 = s0 - pa7
        src0 = s0 * tl.load(e0 + idx, mask=live, other=0.0)
        if PML:
            tl.store(w0 + idx, src0, mask=live)
            v0 = tl.load(f0 + idx, mask=live, other=0.0)
            v0 = v0 + kp_0 * src0
            v0 = v0 - km_0 * prev0
            tl.store(f0 + idx, v0, mask=live)
        else:
            tl.store(f0 + idx, src0, mask=live)

        # --- component 0: update_P, one arm per pole that drives it ----
        if NP0 > 0:
            c_now = cnow_a0
            c_prev = cprev_a0
            c_drive = cdrive_a0
            p = pa0
            q = tl.load(p_prev_a0 + idx, mask=live, other=0.0)
            w = src0
            if SV_a0:
                s = tl.load(sigma_a0 + idx, mask=live, other=0.0)
            else:
                s = sigma_a0
            tl.store(p_out_a0 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP0 > 1:
            c_now = cnow_a1
            c_prev = cprev_a1
            c_drive = cdrive_a1
            p = pa1
            q = tl.load(p_prev_a1 + idx, mask=live, other=0.0)
            w = src0
            if SV_a1:
                s = tl.load(sigma_a1 + idx, mask=live, other=0.0)
            else:
                s = sigma_a1
            tl.store(p_out_a1 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP0 > 2:
            c_now = cnow_a2
            c_prev = cprev_a2
            c_drive = cdrive_a2
            p = pa2
            q = tl.load(p_prev_a2 + idx, mask=live, other=0.0)
            w = src0
            if SV_a2:
                s = tl.load(sigma_a2 + idx, mask=live, other=0.0)
            else:
                s = sigma_a2
            tl.store(p_out_a2 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP0 > 3:
            c_now = cnow_a3
            c_prev = cprev_a3
            c_drive = cdrive_a3
            p = pa3
            q = tl.load(p_prev_a3 + idx, mask=live, other=0.0)
            w = src0
            if SV_a3:
                s = tl.load(sigma_a3 + idx, mask=live, other=0.0)
            else:
                s = sigma_a3
            tl.store(p_out_a3 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP0 > 4:
            c_now = cnow_a4
            c_prev = cprev_a4
            c_drive = cdrive_a4
            p = pa4
            q = tl.load(p_prev_a4 + idx, mask=live, other=0.0)
            w = src0
            if SV_a4:
                s = tl.load(sigma_a4 + idx, mask=live, other=0.0)
            else:
                s = sigma_a4
            tl.store(p_out_a4 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP0 > 5:
            c_now = cnow_a5
            c_prev = cprev_a5
            c_drive = cdrive_a5
            p = pa5
            q = tl.load(p_prev_a5 + idx, mask=live, other=0.0)
            w = src0
            if SV_a5:
                s = tl.load(sigma_a5 + idx, mask=live, other=0.0)
            else:
                s = sigma_a5
            tl.store(p_out_a5 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)

        # --- component 1: update_E --------------------------------------------
        if PML:
            prev1 = tl.load(w1 + idx, mask=live, other=0.0)   # BEFORE the store.
        s1 = tl.load(g1 + idx, mask=live, other=0.0)
        if NP1 > 0:
            pb0 = tl.load(b0 + idx, mask=live, other=0.0)
            s1 = s1 - pb0
        if NP1 > 1:
            pb1 = tl.load(b1 + idx, mask=live, other=0.0)
            s1 = s1 - pb1
        if NP1 > 2:
            pb2 = tl.load(b2 + idx, mask=live, other=0.0)
            s1 = s1 - pb2
        if NP1 > 3:
            pb3 = tl.load(b3 + idx, mask=live, other=0.0)
            s1 = s1 - pb3
        if NP1 > 4:
            pb4 = tl.load(b4 + idx, mask=live, other=0.0)
            s1 = s1 - pb4
        if NP1 > 5:
            pb5 = tl.load(b5 + idx, mask=live, other=0.0)
            s1 = s1 - pb5
        if NP1 > 6:
            pb6 = tl.load(b6 + idx, mask=live, other=0.0)
            s1 = s1 - pb6
        if NP1 > 7:
            pb7 = tl.load(b7 + idx, mask=live, other=0.0)
            s1 = s1 - pb7
        src1 = s1 * tl.load(e1 + idx, mask=live, other=0.0)
        if PML:
            tl.store(w1 + idx, src1, mask=live)
            v1 = tl.load(f1 + idx, mask=live, other=0.0)
            v1 = v1 + kp_1 * src1
            v1 = v1 - km_1 * prev1
            tl.store(f1 + idx, v1, mask=live)
        else:
            tl.store(f1 + idx, src1, mask=live)

        # --- component 1: update_P, one arm per pole that drives it ----
        if NP1 > 0:
            c_now = cnow_b0
            c_prev = cprev_b0
            c_drive = cdrive_b0
            p = pb0
            q = tl.load(p_prev_b0 + idx, mask=live, other=0.0)
            w = src1
            if SV_b0:
                s = tl.load(sigma_b0 + idx, mask=live, other=0.0)
            else:
                s = sigma_b0
            tl.store(p_out_b0 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP1 > 1:
            c_now = cnow_b1
            c_prev = cprev_b1
            c_drive = cdrive_b1
            p = pb1
            q = tl.load(p_prev_b1 + idx, mask=live, other=0.0)
            w = src1
            if SV_b1:
                s = tl.load(sigma_b1 + idx, mask=live, other=0.0)
            else:
                s = sigma_b1
            tl.store(p_out_b1 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP1 > 2:
            c_now = cnow_b2
            c_prev = cprev_b2
            c_drive = cdrive_b2
            p = pb2
            q = tl.load(p_prev_b2 + idx, mask=live, other=0.0)
            w = src1
            if SV_b2:
                s = tl.load(sigma_b2 + idx, mask=live, other=0.0)
            else:
                s = sigma_b2
            tl.store(p_out_b2 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP1 > 3:
            c_now = cnow_b3
            c_prev = cprev_b3
            c_drive = cdrive_b3
            p = pb3
            q = tl.load(p_prev_b3 + idx, mask=live, other=0.0)
            w = src1
            if SV_b3:
                s = tl.load(sigma_b3 + idx, mask=live, other=0.0)
            else:
                s = sigma_b3
            tl.store(p_out_b3 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP1 > 4:
            c_now = cnow_b4
            c_prev = cprev_b4
            c_drive = cdrive_b4
            p = pb4
            q = tl.load(p_prev_b4 + idx, mask=live, other=0.0)
            w = src1
            if SV_b4:
                s = tl.load(sigma_b4 + idx, mask=live, other=0.0)
            else:
                s = sigma_b4
            tl.store(p_out_b4 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP1 > 5:
            c_now = cnow_b5
            c_prev = cprev_b5
            c_drive = cdrive_b5
            p = pb5
            q = tl.load(p_prev_b5 + idx, mask=live, other=0.0)
            w = src1
            if SV_b5:
                s = tl.load(sigma_b5 + idx, mask=live, other=0.0)
            else:
                s = sigma_b5
            tl.store(p_out_b5 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)

        # --- component 2: update_E --------------------------------------------
        if PML:
            prev2 = tl.load(w2 + idx, mask=live, other=0.0)   # BEFORE the store.
        s2 = tl.load(g2 + idx, mask=live, other=0.0)
        if NP2 > 0:
            pc0 = tl.load(c0 + idx, mask=live, other=0.0)
            s2 = s2 - pc0
        if NP2 > 1:
            pc1 = tl.load(c1 + idx, mask=live, other=0.0)
            s2 = s2 - pc1
        if NP2 > 2:
            pc2 = tl.load(c2 + idx, mask=live, other=0.0)
            s2 = s2 - pc2
        if NP2 > 3:
            pc3 = tl.load(c3 + idx, mask=live, other=0.0)
            s2 = s2 - pc3
        if NP2 > 4:
            pc4 = tl.load(c4 + idx, mask=live, other=0.0)
            s2 = s2 - pc4
        if NP2 > 5:
            pc5 = tl.load(c5 + idx, mask=live, other=0.0)
            s2 = s2 - pc5
        if NP2 > 6:
            pc6 = tl.load(c6 + idx, mask=live, other=0.0)
            s2 = s2 - pc6
        if NP2 > 7:
            pc7 = tl.load(c7 + idx, mask=live, other=0.0)
            s2 = s2 - pc7
        src2 = s2 * tl.load(e2 + idx, mask=live, other=0.0)
        if PML:
            tl.store(w2 + idx, src2, mask=live)
            v2 = tl.load(f2 + idx, mask=live, other=0.0)
            v2 = v2 + kp_2 * src2
            v2 = v2 - km_2 * prev2
            tl.store(f2 + idx, v2, mask=live)
        else:
            tl.store(f2 + idx, src2, mask=live)

        # --- component 2: update_P, one arm per pole that drives it ----
        if NP2 > 0:
            c_now = cnow_c0
            c_prev = cprev_c0
            c_drive = cdrive_c0
            p = pc0
            q = tl.load(p_prev_c0 + idx, mask=live, other=0.0)
            w = src2
            if SV_c0:
                s = tl.load(sigma_c0 + idx, mask=live, other=0.0)
            else:
                s = sigma_c0
            tl.store(p_out_c0 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP2 > 1:
            c_now = cnow_c1
            c_prev = cprev_c1
            c_drive = cdrive_c1
            p = pc1
            q = tl.load(p_prev_c1 + idx, mask=live, other=0.0)
            w = src2
            if SV_c1:
                s = tl.load(sigma_c1 + idx, mask=live, other=0.0)
            else:
                s = sigma_c1
            tl.store(p_out_c1 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP2 > 2:
            c_now = cnow_c2
            c_prev = cprev_c2
            c_drive = cdrive_c2
            p = pc2
            q = tl.load(p_prev_c2 + idx, mask=live, other=0.0)
            w = src2
            if SV_c2:
                s = tl.load(sigma_c2 + idx, mask=live, other=0.0)
            else:
                s = sigma_c2
            tl.store(p_out_c2 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP2 > 3:
            c_now = cnow_c3
            c_prev = cprev_c3
            c_drive = cdrive_c3
            p = pc3
            q = tl.load(p_prev_c3 + idx, mask=live, other=0.0)
            w = src2
            if SV_c3:
                s = tl.load(sigma_c3 + idx, mask=live, other=0.0)
            else:
                s = sigma_c3
            tl.store(p_out_c3 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP2 > 4:
            c_now = cnow_c4
            c_prev = cprev_c4
            c_drive = cdrive_c4
            p = pc4
            q = tl.load(p_prev_c4 + idx, mask=live, other=0.0)
            w = src2
            if SV_c4:
                s = tl.load(sigma_c4 + idx, mask=live, other=0.0)
            else:
                s = sigma_c4
            tl.store(p_out_c4 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)
        if NP2 > 5:
            c_now = cnow_c5
            c_prev = cprev_c5
            c_drive = cdrive_c5
            p = pc5
            q = tl.load(p_prev_c5 + idx, mask=live, other=0.0)
            w = src2
            if SV_c5:
                s = tl.load(sigma_c5 + idx, mask=live, other=0.0)
            else:
                s = sigma_c5
            tl.store(p_out_c5 + idx,
                     ((p * c_now) + (c_prev * q)) + (c_drive * (s * w)),
                     mask=live)

else:  # pragma: no cover - the laptop path
    fused_ade_chain_step = None  # type: ignore[assignment]


def fused_ade_chain_kernel() -> Any:
    """The shipped ``@triton.jit`` kernel, or a diagnosable ImportError."""
    if fused_ade_chain_step is None:
        raise ImportError(
            "the fused E->P chain kernel needs the optional `triton` package "
            "(pip install triton). The engine runs without it; only this fast "
            f"path is unavailable. Original error: {_TRITON_IMPORT_ERROR}")
    return fused_ade_chain_step


#: The kernel's per-component argument prefix, in signature order.
LETTERS: Tuple[str, str, str] = ("a", "b", "c")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
#
# A conjunction of the two halves' OWN certified predicates plus the seam clauses.
# Nothing is weakened and nothing is re-decided: a configuration either half
# refuses is refused here with that half's reasons, prefixed so a reader can tell
# which side said it. The arm decides WHICH pair of predicates is asked, and the
# pairing is not free — the E arm and the ADE arm must agree about the absorber,
# because the drive this kernel keeps in a register is whichever field that arm's
# E half wrote.

#: arm -> (the E-half predicate, the module whose pole ORDER and live binding the
#: E half itself uses). Taking the order from the arm's OWN module rather than
#: from a copy here is what stops the fused chain subtracting the poles in one
#: order while the certified body it transcribes uses another.
_E_HALF = {
    "no_pml": (_stored_e.stored_e_constitutive_coverage, _stored_e),
    "dispersive": (_dispersive.dispersive_constitutive_coverage, _dispersive),
    # The folded arm takes ITS OWN admission and the UNFOLDED arm's pole order —
    # which is right, because folded_dispersive_update_e's own builder takes
    # dispersive_update_e's poles_per_component and LivePoleBinding unchanged
    # (folded_dispersive_update_e.py:275-288). The two predicates are disjoint by
    # construction: one requires a real fold, the other refuses one.
    "folded": (_folded.folded_dispersive_constitutive_coverage, _dispersive),
}


def _ade_coverage(fields: Any, pml: Any, arm: str, state: Any,
                  component: str) -> Coverage:
    """The ADE-half predicate this arm's absorber state selects.

    THE TWO ARE NOT INTERCHANGEABLE AND THE DIFFERENCE IS THE DRIVE.
    ``coverage.ade_update_p_coverage`` requires ``f_w_<component>`` to exist,
    because under an absorber that is what ``Fields.drive_field`` returns;
    ``no_pml_ade.no_pml_ade_update_p_coverage`` inverts exactly that clause and
    requires ``drive_field`` to hand back the stored E BY IDENTITY. Asking the
    wrong one of a run would admit a configuration whose register is the wrong
    field — "the single most likely silent wrong answer in dispersion"
    (fields.py:1149-1156), which agrees everywhere except inside the absorber.
    """
    if arm in PML_ARMS:
        return _coverage.ade_update_p_coverage(fields, state, component)
    return _no_pml_ade.no_pml_ade_update_p_coverage(fields, pml, state, component)


def fused_ade_chain_coverage(fields: Any, pml: Any, arm: str) -> Coverage:
    """May ONE launch span ``update_E`` -> ``update_P`` for this configuration?

    THERE IS NO SOURCE CLAUSE, AND THE ABSENCE IS DELIBERATE. Every other fused
    product on this backend must be told the source inventory, because the driver
    injects INSIDE its seam and ignorance is not an empty set. This seam has no
    injection in it at all — ``driver.step`` runs ``update_E`` at :3304 and
    ``update_P`` at :3306 with nothing between — so requiring a declaration would
    refuse configurations the driver cannot break. The fusion matrix measures the
    same fact from the other end: ``E->P: 15 instances, 0 blocked by the source
    seam``.
    """
    if arm not in ARMS:
        return Coverage(False, (f"arm must be one of {list(ARMS)}, got {arm!r}",))
    electric_coverage, order_module = _E_HALF[arm]

    reasons: List[str] = []
    electric = electric_coverage(fields, pml)
    if not electric.covered:
        reasons.extend(f"E half: {reason}" for reason in electric.reasons)

    states = tuple(getattr(fields, "polarizations", ()) or ())
    if not states:
        reasons.append(
            "no susceptibility is registered: with nothing to advance this product "
            "spans one sub-step, not two, and that configuration is the E arm's own")

    try:
        order = order_module.poles_per_component(fields)
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
            verdict = _ade_coverage(fields, pml, arm, state, component)
            if not verdict.covered:
                reasons.extend(f"ADE half {index}.{component}: {reason}"
                               for reason in verdict.reasons)
        # THE TWO HALVES MUST AGREE ABOUT WHICH COMPONENTS THIS STATE DRIVES.
        # The E chain subtracts wherever `drives(component)` is True and the ADE
        # half advances wherever `driven()` lists it; the two are one attribute in
        # the reference (dispersion.py:652-656) and a disagreement would subtract a
        # pole this launch never advances — a frozen P, smooth and wrong.
        subtracted = tuple(component for component, _displacement, _axis in E_TERMS
                           if any(state is other
                                  for other in order.get(component, ())))
        if tuple(name for name in tuple(driven) if name in dict(
                (term[0], term) for term in E_TERMS)) != subtracted:
            reasons.append(
                f"polarization {index}: driven()={tuple(driven)!r} disagrees with "
                f"the components whose D - sum P chain subtracts it ({subtracted!r})")

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
                    f"the state's own P; the register the ADE arm reuses would be "
                    f"a different array")

    # THE SEAM'S OWN CONDITION, named rather than inherited: the register this
    # kernel hands the recurrence is what its E half stored, so `drive_field` must
    # return THAT array. Under an absorber that is `f_w_<c>`; the no-PML ADE
    # predicate already asserts the stored-E identity from the other side.
    if arm in PML_ARMS:
        if not bool(getattr(fields, "_pml_active", False)):
            reasons.append(
                "Fields is not in PML storage mode: drive_field would return the "
                "stored E rather than f_w (fields.py:1160-1162), and the register "
                "this kernel passes the recurrence is the split-field drive")
        reader = getattr(fields, "drive_field", None)
        if not callable(reader):
            reasons.append("fields does not expose drive_field(); an unreadable "
                           "drive is not a covered one")
        else:
            for component, _displacement, _axis in E_TERMS:
                auxiliary = getattr(fields, "f_w_" + component, None)
                try:
                    drive = reader(component)
                except Exception as exc:  # noqa: BLE001
                    reasons.append(f"drive_field({component!r}) raised {exc!r}")
                    continue
                if auxiliary is None or drive is not auxiliary:
                    reasons.append(
                        f"drive_field({component!r}) is not f_w_{component}; this "
                        f"kernel's register is the value it stores THERE")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def explain_fused_ade_chain(fields: Any, pml: Any, arm: str) -> Coverage:
    """The verdict with its reasons, for reports. Needs no Triton."""
    return fused_ade_chain_coverage(fields, pml, arm)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

class FusedAdeChainPlan:
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
    docstring for the orbit enumeration that licenses the shape.

    THE POINTERS ARE RESOLVED PER LAUNCH, DELIBERATELY. The rotation moves which
    allocation holds ``P``, ``P_prev`` and each scratch on every step, so a plan
    that cached them would be stale after the FIRST step — and stale in a way that
    still computes, giving a smooth wrong field. The ORDER is cached (it is fixed
    for the run) and re-checked every call by the E arm's own ``LivePoleBinding``.

    THIS PLAN OWNS THE ROTATION FOR THE RUN. ``PolarizationState.update`` rotates
    a single shared ``_scratch``; interleaving it with this plan would hand one
    component's launch a buffer another slot already owns. ``_scratch`` is kept
    pointing at the FIRST driven component's scratch so the engine object is never
    left holding an allocation this plan has also given to ``P``, but the array
    path and this plan must not step the same ``Fields``.
    """

    performs_device_work = True
    replaces_sub_steps = REPLACES
    launches_per_run = 1

    __slots__ = ("fields", "arm", "shape", "n_elem", "block", "num_warps", "counts",
                 "extra_scratch_volumes", "launches", "runs", "alias_checks",
                 "_pointer", "_targets", "_aux", "_sources", "_inv_eps",
                 "_coefficients", "_poles", "_order", "_states", "_index_of",
                 "_scratch", "_first_driven", "_sigma", "_coeff", "_sv", "_kernel",
                 "_grid", "_pml_flag", "_read_ids")

    def __init__(self, fields: Any, pml: Any, arm: str, block: int,
                 num_warps: Optional[int] = 1, kernel: Any = None,
                 pointer: Any = None) -> None:
        from .launch import CupyPointer, _flat  # noqa: PLC0415

        if arm not in ARMS:
            raise ValueError(f"arm must be one of {list(ARMS)}, got {arm!r}")
        if pointer is None:
            pointer = CupyPointer
        _electric, order_module = _E_HALF[arm]
        self.fields = fields
        self.arm = str(arm)
        self.shape = tuple(int(n) for n in fields.grid.shape)
        self.n_elem = self.shape[0] * self.shape[1] * self.shape[2]
        self.block = int(block)
        self.num_warps = num_warps
        self._pointer = pointer
        self._kernel = kernel
        self._pml_flag = 1 if arm in PML_ARMS else 0
        self._grid = ((self.n_elem + self.block - 1) // self.block,)
        self.launches = 0
        self.runs = 0
        self.alias_checks = 0

        components = tuple(term[0] for term in E_TERMS)
        targets = [getattr(fields, name) for name in components]
        sources = [getattr(fields, term[1]) for term in E_TERMS]
        inverse = [fields.inverse_epsilon_for(name) for name in components]
        self._targets = tuple(pointer(array) for array in targets)
        self._sources = tuple(pointer(array) for array in sources)
        self._inv_eps = tuple(pointer(array) for array in inverse)
        read_ids = {id(array) for array in targets + sources + inverse}

        if arm in PML_ARMS:
            auxiliaries = [getattr(fields, "f_w_" + name) for name in components]
            columns = [getattr(pml, f"{stem}_{axis}_h")
                       for axis in "xyz" for stem in ("kps", "kms")]
            self._aux = tuple(pointer(array) for array in auxiliaries)
            self._coefficients = tuple(pointer(_flat(array)) for array in columns)
            read_ids.update(id(array) for array in auxiliaries + columns)
        else:
            # NO AUXILIARY AND NO COEFFICIENT COLUMNS EXIST on this path, and the
            # ``PML`` constexpr is what stops the kernel reading them — but a
            # pointer argument still has to type, and a null makes the launcher's
            # failure a TypeError far from its cause. Dx, for the reason the pole
            # pad takes it (no_pml_stored_e.py:667-670).
            self._aux = tuple([self._sources[0]] * 3)
            self._coefficients = tuple([self._sources[0]] * 6)

        partition = order_module.poles_per_component(fields)
        self._order = {name: tuple(states) for name, states in partition.items()}
        self._poles = order_module.LivePoleBinding(fields, self._order)
        self.counts = tuple(len(self._order[name]) for name in components)
        for name, count in zip(components, self.counts):
            if count > CHAIN_MAX_POLES:
                raise ValueError(
                    f"{name} is driven by {count} poles; this product binds "
                    f"{CHAIN_MAX_POLES} ADE arms per component and the predicate "
                    f"refuses more")

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
                # step reads a scalar as a pointer, which is a wrong answer and
                # not a crash (coverage.sigma_is_volume).
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
        for index, (component, _displacement, _axis) in enumerate(E_TERMS):
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
            slots.extend(pointer(array) for array in group)
            # Unused slots take this component's own D pointer, exactly as both
            # certified E bodies' plans pad them (no_pml_stored_e.py:667-670).
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
        nx, ny, nz = self.shape
        ade: List[Any] = []
        for index, entries in enumerate(chain):
            component = E_TERMS[index][0]
            padding = self._sources[index]
            live = len(entries)
            ade.extend([pointer(row[2]) for row in entries])
            ade.extend([padding] * (CHAIN_MAX_POLES - live))
            ade.extend([pointer(row[4]) for row in entries])
            ade.extend([padding] * (CHAIN_MAX_POLES - live))
            ade.extend([self._sigma[(component, slot)] for slot in range(live)])
            ade.extend([0.0] * (CHAIN_MAX_POLES - live))
            for position in range(3):
                ade.extend([self._coeff[(component, slot)][position]
                            for slot in range(live)])
                ade.extend([0.0] * (CHAIN_MAX_POLES - live))
        arguments = [*self._targets, *self._aux, *self._sources, *self._inv_eps,
                     *slots, *self._coefficients, *ade,
                     nx, ny, nz, self.n_elem]
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
        :class:`fused_ade_state.FusedAdeStatePlan` uses and for the same reason: a
        plan that rotated first and then raised would leave the engine holding a P
        history that never happened.
        """
        from .kernels import ENABLE_FP_FUSION  # noqa: PLC0415

        kernel = (self._kernel if self._kernel is not None
                  else fused_ade_chain_kernel())
        groups = self._poles.arrays()
        chain = self._resolve(groups)
        self._check_aliasing(chain)
        arguments = self._arguments(self._pole_slots(groups), chain)
        extra = {} if self.num_warps is None else {"num_warps": self.num_warps}
        kernel[self._grid](
            *arguments,
            NP0=self.counts[0], NP1=self.counts[1], NP2=self.counts[2],
            PML=self._pml_flag, BLOCK=self.block, **self._sv,
            enable_fp_fusion=ENABLE_FP_FUSION if guard is None else bool(guard),
            **extra,
        )
        self.launches += 1
        self._rotate(chain)
        self.runs += 1

    def __repr__(self) -> str:
        return (f"FusedAdeChainPlan(arm={self.arm!r}, shape={self.shape}, "
                f"poles={self.counts}, extra_scratch={self.extra_scratch_volumes}, "
                f"block={self.block}, num_warps={self.num_warps})")


def plan_fused_ade_chain(fields: Any, pml: Any, arm: str,
                         block: Optional[int] = None,
                         num_warps: Optional[int] = 1,
                         kernel: Any = None) -> Optional[FusedAdeChainPlan]:
    """Build the fused E->P plan for one arm, or return ``None`` on any refusal.

    ``None`` means REFUSED and the reasons are available from
    :func:`fused_ade_chain_coverage`. The Triton import stays BELOW the predicate,
    as in every builder in this package: a NumPy host must be able to plan (to
    ``None``) without the optional dependency being importable at all.

    ``kernel=`` IS LOAD-BEARING: the gate's mutation legs route a mutated kernel
    through it, and a builder that dropped it would launch the SHIPPED kernel and
    report a pass for a defect it never introduced.
    """
    if not fused_ade_chain_coverage(fields, pml, arm).covered:
        return None
    from .kernels import DEFAULT_BLOCK  # noqa: PLC0415

    return FusedAdeChainPlan(fields, pml, arm,
                             DEFAULT_BLOCK if block is None else block,
                             num_warps=num_warps, kernel=kernel)


# ---------------------------------------------------------------------------
# WIRING — deferred, exactly as the other cross-sub-step products are held
# ---------------------------------------------------------------------------
#
# ``launch.plan_step`` does not import this module and ``launch.FAMILY_MODULES``
# does not name it, so nothing about any existing arm's behaviour changes and a
# default run still steps this seam on the array path. The product exists to be
# MEASURED: ``parity/meep_gpu/gate_triton_fused_ade_chain.py`` is its byte gate,
# ``results/triton_fused_ade_chain_2026-08-21/`` is the run that released it, and
# ``results/fusion_matrix_triton_2026-08-21_ade_chain/`` is where its corpus reach
# is scored. Composition is a separate change with its own disjointness argument,
# because this product spans TWO slots and the planner's arm table assigns at most
# one arm per slot.
