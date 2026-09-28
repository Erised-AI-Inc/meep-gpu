"""``update_E`` welded to ``update_P`` — the first fused product at the E->P seam.

THE SEAM IS EMPTY, AND THAT IS THE WHOLE REASON THIS PRODUCT EXISTS.
``driver.step`` runs::

    if fast is None or not fast.dispatch("update_E", self.fields):
        update_E(self.fields, self.pml)                       # driver.py:3304
    if fast is None or not fast.dispatch("update_P", self.fields):
        update_P(self.fields, self.pml)                       # driver.py:3306

Nothing sits between them: no source injection, no mirror fill, no wall clear, no
far-ghost pass. Every other seam on this board has a driver pass inside it that a
fused kernel must either carry inline or refuse; this one has none, which is why
the Metal fusion matrix records ``E_to_P: 15 instances, 0 blocked by the source
seam, CEILING 15``.

===========================================================================
THE ROTATION QUESTION, AND THE THIRD ANSWER
===========================================================================

``ade_update_p`` launches once per DRIVEN COMPONENT and the host rotates
``P`` / ``P_prev`` / ``_scratch`` after each launch (ade_update_p.py:13-18, and
the three assignments at :353-355 transcribing dispersion.py:689-691). The fusion
census named that rotation as the blocking unknown and offered two answers — take
one component, or bake the rotation into the launch — and could not decide from
records which was required.

``triton_kernels.fused_ade_state`` takes the second. Its note records what that
costs: arm 1's OUTPUT buffer IS arm 0's ``p_prev`` INPUT, the safety of that alias
is an OBSERVATION on one toolchain rather than a language guarantee, and — the
sentence this family is built around — "Fusing further REQUIRES breaking the chain
first, not inheriting it."

THERE IS A THIRD ANSWER AND IT IS MEASURED, not argued:
``parity/meep_gpu/probe_metal_ade_rotation_seam.py`` /
``results/metal_ade_rotation_seam_2026-08-20/``.

* **The rotation does not have to move at all.** A PER-COMPONENT interleave —
  ``update_E(c)`` immediately followed by every ``update_P(state, c)``, for c in
  Ex, Ey, Ez — is BIT-IDENTICAL to the driver's order (all of ``update_E``, then
  all of ``update_P``) over 12 complete steps on five configurations at 1, 2, 5
  and 6 poles with and without PML, compared as uint32 words over all 44 live
  arrays, using the ALREADY CERTIFIED products entry-sliced so the only variable
  is dispatch order. The equivalence holds for an arithmetic reason the two
  certified predicates already enforce: this ``update_E`` reads only its OWN
  component's ``D``, ``inv_eps`` and ``P`` volumes (no off-diagonal row, no
  ``chi3`` magnitude — both refused by the E half), and ``update_P(state, c)``
  advances only ``P[c]``, so no component's E-side read is disturbed by another
  component's P-side write.
* **Per component, the rotation is ALIAS-FREE BY CONSTRUCTION.** The rotation is a
  permutation of ``2d + 1`` buffers for ``d`` driven components, so its orbit is
  FINITE and the probe walks it to closure rather than sampling: at EVERY
  configuration in the orbit, for d = 1, 2, 3, a per-component launch's write set
  ``{out_k[c]}`` is disjoint from its read set ``{P_k[c], P_prev_k[c],
  sigma_k[c]}``. ``out_k[c]`` is either that susceptibility's ``_scratch`` or the
  ``P_prev`` of an EARLIER component, and neither is read by component ``c``'s
  launch. The same enumeration run over the ALL-COMPONENT shape finds a conflict
  at every orbit position for d > 1 — that is ``fused_ade_state``'s alias, and it
  is the thing this shape does not have.

So the answer to the census's open question is: THE HOST ROUND TRIP STAYS, AND IT
WAS NEVER THE BLOCKER. One launch per component, the rotation left exactly where
the certified plan already performs it, and the fused kernel inherits no alias.

ONE ALIAS DOES SURVIVE, ACROSS LAUNCHES, AND IT IS INHERITED RATHER THAN NEW.
Component Ey's launch WRITES the buffer component Ex's launch READ as ``p_prev``
— that is what the rotation is. It is a write-after-read across two dispatches on
one MPS stream, which in-order execution settles, and it is EXACTLY the relation
``MetalAdeUpdatePPlan`` already has between its own consecutive launches
(ade_update_p.py:353-355). The gate's ``separate_control`` leg runs that certified
plan beside this one on identical state and both agree with the array path word
for word, so the inherited relation is measured on both sides rather than assumed
on one. What this shape removes is the INTRA-launch alias, which is a different
animal: no compiler ordering can repair a cross-program-instance race, and one
inside a single launch is what ``fused_ade_state``'s note is about.

===========================================================================
WHAT IS FUSED, AND IT IS EXACTLY ONE VALUE
===========================================================================

``update_P``'s drive is ``Fields.drive_field(component)``: the stored E without
PML, and ``f_w_E*`` under one. In both cases that is the value ``update_E`` just
computed and stored one launch earlier. Fused, the reload becomes a register:

* **dispersive PML E** — the certified body ALREADY names it. ``float src =
  source * inv_e[idx]; fw[idx] = src;`` (dispersive_update_e.py:79-80), and
  ``src`` IS ``f_w_E[c][idx]``. The ADE half's ``float w = drive[idx];`` becomes
  ``float w = src;`` and NOTHING ELSE MOVES.
* **no-PML stored E** — the certified body stores the product without naming it:
  ``e_out[idx] = source * inv_e[idx];`` (no_pml_stored_e.py:87). Fused, that ONE
  line becomes two, ``float e = source * inv_e[idx]; e_out[idx] = e;``, and the
  ADE half's ``float w = drive[idx];`` becomes ``float w = e;``. A float32 value
  stored to a ``device float*`` and reloaded is bit-identical to the register, so
  the split is byte-neutral by construction — and the gate MEASURES it anyway,
  per complete step, because "by construction" is a hypothesis until a comparator
  agrees.

EVERY OTHER LINE IS TRANSCRIBED CHARACTER FOR CHARACTER from the body it replaces.
The lines are NOT restated anywhere in this module as data: a second declaration
of "here are the load-bearing lines" is one more place to drift. Instead
``test_metal_fused_ade_chain`` CALLS each certified emitter, searches its output
for the line, and searches this family's output for the same line with the
documented rename applied — so the check parses both sources and mirrors neither.

===========================================================================
WHY THE POINTER NAMES MOVE, AND WHY THE SCALARS MOVE INTO A STRUCT
===========================================================================

**POINTER RENAMES.** The E half names its pole buffers ``p0..p7``
(no_pml_stored_e.py:59-66, dispersive_update_e.py:44-51) and the ADE half names
its three volumes ``p_now`` / ``p_prev`` / ``p_out`` (ade_update_p.py:64-66).
Fused, ``p0`` would mean two things in one scope, so the ADE volumes become
``o{k}`` (out), ``q{k}`` (prev) and ``s{k}`` (sigma) while ``p{k}`` stays the
E half's spelling — AND IS THE SHARED POINTER. ``p_now`` is not bound a second
time: pole ``k``'s ``P[c]`` is the same array the E chain subtracts, so the fused
signature binds it once. That shared binding is the fusion's whole footprint.

**THE SCALARS.** :data:`.device.MAX_BUFFER_BINDINGS` is 31 and the corpus's worst
E->P row (``stochastic_emitter*.py``) carries SIX poles on every component. Six
poles costs ``3 x 6 = 18`` ADE pointers plus up to six sigma volumes on top of the
E half's own six, and the coefficients are ``3 x 6 = 18`` scalars besides. Bound
one per index that is 52 bindings against a ceiling of 31. Packing every scalar
into ONE ``constant Params&`` — the shape ``fused_dispersive_pair``'s docstring
records as measured to compile — brings the worst corpus configuration to exactly
31 bindings, and 32 is a compile error. Measured on this host 2026-08-20
(``results/metal_ade_rotation_seam_2026-08-20``)::

    dispersive PML E, 6 poles, 6 sigma volumes  -> 30 pointers + Params = 31  COMPILES
    dispersive PML E, 6 poles, scalar sigma     -> 24 pointers + Params = 25  COMPILES
    no-PML stored E,  5 poles, 5 sigma volumes  -> 23 pointers + Params = 24  COMPILES
    one pole more (7/7)                         -> 34 bindings            REFUSED

so :func:`binding_count` is evaluated by the predicate and a configuration over
the ceiling is a REFUSAL BY NAME rather than a compile error at plan time. The
three-line prologue that copies ``prm.ny`` / ``prm.nz`` / ``prm.n_elem`` back into
locals exists so :data:`.templates.GUARD` and :data:`.templates.DECODE_IJK` are
emitted from the CERTIFIED emitters unchanged.

THE POLE SLOTS ARE NOT PADDED TO EIGHT. Both pointwise families declare
``p0..p7`` always and bind the displacement into the unused slots
(no_pml_stored_e.py:213, dispersive_update_e.py:213). Here the signature declares
exactly ``pole_count`` slots, because the pad is what puts this over the ceiling:
eight declared poles is 8 unused pointers the fused signature cannot afford. The
body is already specialised per pole count in both certified families, so nothing
about the ARITHMETIC changes with the declaration.

===========================================================================
WHAT THIS FAMILY REFUSES
===========================================================================

* **an off-diagonal chi1inv row or a chi2/chi3** — refused by BOTH E-half
  predicates already, and named again here as a SEAM clause because the reason is
  different: the interleave's equivalence rests on ``update_E(c)`` reading only
  component ``c``'s volumes, and both of those features read the OTHER components'
  ``D - sum P`` volumes (stepping.py:991-997). That is why the fifth E->P cell,
  ``(folded off-diagonal dispersive PML E, ADE update_P)`` — one corpus row,
  ``absorbed_power_density.py`` — is NOT claimed by this family.
* **complex64 storage** — a separate signature and a separate arm; the four
  ``TestLoadDump.*_3d`` rows are not claimed here.
* **a pole set the two halves disagree on** — the E half's pole order for a
  component (``_poles``) and the ADE plan's entry order for that component are
  both ``fields.polarizations`` order filtered by ``drives(component)``. The plan
  builder CHECKS that identity per component rather than inheriting it; a
  disagreement would silently pair pole ``k``'s displacement subtraction with pole
  ``j``'s recurrence.

NOT WIRED. ``STEP_ORDER`` assigns at most one arm per slot and this product spans
TWO (``update_E`` and ``update_P``), so it registers ``wired=False``: the composer
cannot select it, and the disjointness sweep can still enumerate it. Nothing about
any existing arm's behaviour changes, and ``meep_gpu.fastpath.plan_fast_path``
still returns ``None`` on every branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.coverage import Coverage
from . import shaders, templates
from .ade_update_p import (
    _physical_name,
    metal_ade_update_p_coverage,
)
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .dispersive_update_e import (
    _poles,
    metal_dispersive_e_coverage,
)
from .folded_dispersive_update_e import folded_dispersive_e_coverage
from .no_pml_stored_e import metal_stored_e_coverage

#: THREE FAMILY NAMES, ONE MODULE, and the split is forced rather than stylistic.
#: ``arms`` holds one row per (family, label) and the table's invariant is that NO
#: SLOT CARRIES TWO ARMS FROM ONE FAMILY (test_metal_planner_composition.py:
#: ``test_no_slot_carries_two_arms_from_one_family``) — two rows from one family on
#: one slot would leave that slot permanently UNSELECTED, a family silently
#: disabling itself. The three E->P cells are three products sharing one builder,
#: which is exactly the relation ``folded_dispersive_update_e`` already has to
#: ``dispersive_update_e``: a separate FAMILY, a separate predicate, the same
#: binding builder. ``FAMILY`` is the module's own name and the no-PML arm's.
FAMILY = "fused_ade_chain"
DISPERSIVE_FAMILY = "fused_ade_chain_dispersive"
FOLDED_FAMILY = "folded_fused_ade_chain"

#: The slot every one of them is registered on. The product spans two; it holds a
#: row on the FIRST, so a refusal is named on the slot the fusion starts at rather
#: than being invisible to the table.
SLOT = "update_E"

#: The driver passes one run of this plan performs, in driver order
#: (driver.py:3304-3306). Declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("update_E", "update_P")

#: The two E-side constitutive arms this family welds to the ADE recurrence, and
#: the fixed (non-pole) POINTER count each contributes to the fused signature.
#:
#: * ``no_pml`` — ``e_out, d_in, inv_e`` (no_pml_stored_e.py:56-70 minus the
#:   eight-slot pole pad).
#: * ``dispersive`` — ``e_out, fw, d_in, inv_e, kps, kms``
#:   (dispersive_update_e.py:41-59 minus the same pad).
ARMS_FIXED_POINTERS: Dict[str, int] = {"no_pml": 3, "dispersive": 6}

#: ``(component, displacement, axis, axis name)`` — the same table both certified
#: E bodies iterate (no_pml_stored_e.E_TERMS, dispersive_update_e.E_TERMS), joined
#: so one loop serves both arms.
E_TERMS: Tuple[Tuple[str, str, int, str], ...] = (
    ("Ex", "Dx", 0, "x"), ("Ey", "Dy", 1, "y"), ("Ez", "Dz", 2, "z"))

__all__ = [
    "ARMS_FIXED_POINTERS", "DISPERSIVE_FAMILY", "E_TERMS", "FAMILY",
    "FOLDED_FAMILY", "REPLACES", "SLOT",
    "MetalFusedAdeChainPlan", "binding_count",
    "compile_fused_ade_chain", "fused_ade_chain_coverage",
    "fused_ade_chain_source", "pack_params", "params_words",
    "plan_metal_fused_ade_chain",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params {
    float c_now[__NP__];
    float c_prev[__NP__];
    float c_drive[__NP__];
    float sigma[__NP__];
    uint  ny;
    uint  nz;
    uint  n_elem;
};

kernel void fused_ade_chain_component(
__SIGNATURE__
    uint idx [[thread_position_in_grid]])
{
    // The scalars live in ONE binding (see the module docstring's binding
    // arithmetic); copying them back into the names the certified emitters use
    // is what lets GUARD and DECODE_IJK below be those emitters' own text.
    uint n_elem = prm.n_elem;
    uint ny = prm.ny;
    uint nz = prm.nz;
__BODY__
}
"""


def binding_count(arm: str, pole_count: int, volume_sigmas: int) -> int:
    """Buffer attribute indices the fused signature would occupy.

    ``pointers = fixed + 3 * poles + volume_sigmas`` — the E half's own fixed
    volumes, then per pole ``p`` (SHARED between the halves), ``o`` and ``q``,
    then a ``s`` only for the poles whose sigma is a volume — plus ONE for the
    packed ``Params&``.
    """
    if arm not in ARMS_FIXED_POINTERS:
        raise ValueError(f"arm must be one of {sorted(ARMS_FIXED_POINTERS)}, got {arm!r}")
    if not 0 <= int(volume_sigmas) <= int(pole_count):
        raise ValueError(
            f"volume_sigmas={volume_sigmas!r} is not in [0, {pole_count!r}]")
    return (ARMS_FIXED_POINTERS[arm] + 3 * int(pole_count) + int(volume_sigmas)) + 1


def params_words(pole_count: int) -> int:
    """Length of the ``Params`` blob in 4-byte words: ``4 * max(N, 1) + 3``.

    Every member is a 4-byte scalar or an array of them, so the struct's layout is
    a flat word sequence with no padding to reason about. ``max(N, 1)`` because
    MSL has no zero-length array; a zero-pole specialisation never reads them.
    """
    return 4 * max(int(pole_count), 1) + 3


def pack_params(coefficients: Sequence[Tuple[float, float, float]],
                scalar_sigmas: Sequence[float], ny: int, nz: int,
                n_elem: int) -> np.ndarray:
    """The ``Params`` blob as float32 words, uints carried as their bit patterns.

    ``float()`` on every coefficient for the reason ``fused_dispersive_pair``
    gives its ``dtdx``: the array path multiplies a float32 volume by a Python
    float and Metal binds a Python float into a ``float`` member the same way, so
    the two scalars are the same bits.
    """
    pole_count = len(coefficients)
    span = max(pole_count, 1)
    blob = np.zeros(params_words(pole_count), dtype=np.float32)
    for index, (c_now, c_prev, c_drive) in enumerate(coefficients):
        blob[index] = np.float32(c_now)
        blob[span + index] = np.float32(c_prev)
        blob[2 * span + index] = np.float32(c_drive)
    for index, sigma in enumerate(scalar_sigmas):
        blob[3 * span + index] = np.float32(sigma)
    blob[4 * span:4 * span + 3] = np.frombuffer(
        np.array([int(ny), int(nz), int(n_elem)], dtype=np.uint32).tobytes(),
        dtype=np.float32)
    return blob


def _signature(arm: str, pole_count: int, sigma_is_volume: Sequence[bool]) -> str:
    """Every binding, in the order :meth:`MetalFusedAdeChainPlan.run` passes them."""
    lines: List[str] = []
    slot = 0

    def bind(declaration: str) -> None:
        nonlocal slot
        lines.append(f"    {declaration:<32}[[buffer({slot})]],")
        slot += 1

    bind("device float*       e_out")
    if arm == "dispersive":
        bind("device float*       fw")
    bind("device const float* d_in")
    bind("device const float* inv_e")
    if arm == "dispersive":
        bind("device const float* kps")
        bind("device const float* kms")
    for index in range(pole_count):        # SHARED: the E half's pole AND p_now.
        bind(f"device const float* p{index}")
    for index in range(pole_count):
        bind(f"device float*       o{index}")
    for index in range(pole_count):
        bind(f"device const float* q{index}")
    for index in range(pole_count):
        if sigma_is_volume[index]:
            bind(f"device const float* s{index}")
    bind("constant Params&    prm")
    assert slot == binding_count(
        arm, pole_count, sum(1 for flag in sigma_is_volume if flag)), (
        slot, arm, pole_count, tuple(sigma_is_volume))
    return "\n".join(lines)


def _ade_arm(index: int, sigma_is_volume: bool, drive_register: str) -> List[str]:
    """One pole's ``update_P``, transcribed from ``ade_update_p.ade_source``.

    ``ade_update_p.py:126-130`` is the float32 body::

        float p = p_now[idx];
        float q = p_prev[idx];
        float w = drive[idx];
        float s = __SIGMA_VALUE__;
        p_out[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));

    Reproduced here with the pointer renames the module docstring names and ONE
    substitution — ``drive[idx]`` becomes the register ``update_E`` just wrote.
    The braces give each arm its own ``p``/``q``/``w``/``s``/``c_now``/``c_prev``/
    ``c_drive`` scope, so the final line is character for character that body's.
    """
    sigma = f"s{index}[idx]" if sigma_is_volume else f"prm.sigma[{index}]"
    return [
        f"    {{  // pole {index}: ade_update_p.ade_source, float32 body (:126-130)",
        f"        float c_now = prm.c_now[{index}];",
        f"        float c_prev = prm.c_prev[{index}];",
        f"        float c_drive = prm.c_drive[{index}];",
        f"        float p = p{index}[idx];",
        f"        float q = q{index}[idx];",
        f"        float w = {drive_register};   // THE SEAM: the register update_E just wrote.",
        f"        float s = {sigma};",
        f"        o{index}[idx] = ((p * c_now) + (c_prev * q)) + (c_drive * (s * w));",
        "    }",
    ]


def fused_ade_chain_source(arm: str, axis: int, pole_count: int,
                           sigma_is_volume: Sequence[bool],
                           contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (E arm, component, poles, sigma kinds, mode).

    Every constant Triton would bake into a ``tl.constexpr`` is baked into the
    string, for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader``
    takes a source and nothing else.
    """
    if arm not in ARMS_FIXED_POINTERS:
        raise ValueError(f"arm must be one of {sorted(ARMS_FIXED_POINTERS)}, got {arm!r}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    pole_count = int(pole_count)
    if pole_count < 0:
        raise ValueError(f"pole_count must be non-negative, got {pole_count!r}")
    sigma_is_volume = tuple(bool(flag) for flag in sigma_is_volume)
    if len(sigma_is_volume) != pole_count:
        raise ValueError(
            f"sigma_is_volume must carry one flag per pole, got "
            f"{len(sigma_is_volume)} for {pole_count} poles")
    bindings = binding_count(arm, pole_count,
                             sum(1 for flag in sigma_is_volume if flag))
    if bindings > MAX_BUFFER_BINDINGS:
        raise ValueError(
            f"the fused {arm!r} signature at {pole_count} poles needs {bindings} "
            f"bindings and this platform allows {MAX_BUFFER_BINDINGS} "
            f"(device.py:72); the predicate refuses this configuration by name "
            f"rather than letting it reach the compiler")

    coordinate = ("i", "j", "k")[axis]
    chain = [f"    source = source - p{index}[idx];" for index in range(pole_count)]

    if arm == "no_pml":
        # --- no_pml_stored_e.stored_e_source (no_pml_stored_e.py:83-87) --------
        # ONE line differs and it is THE SEAM: `e_out[idx] = source * inv_e[idx];`
        # is split so the product has a name the ADE half can read from a
        # register. float32 store-then-reload is exact, so the split is
        # byte-neutral; the gate measures it per complete step regardless.
        electric = [
            templates.GUARD,
            "",
            "    // --- pole-aware update_E (stepping.update_E:981-984) ----------",
            "    float source = d_in[idx];",
            *chain,
            "    // THE SEAM: name the constitutive product so update_P reads a register.",
            "    float e = source * inv_e[idx];",
            "    e_out[idx] = e;",
        ]
        drive = "e"
    else:
        # --- dispersive_update_e.dispersive_e_source (:76-83) ------------------
        # NOTHING is rewritten here: the certified body already names the drive
        # `src` at :79, one line before it stores it into f_w.
        electric = [
            templates.GUARD,
            templates.DECODE_IJK,
            "",
            "    // --- pole-aware update_E under PML (stepping.update_E:981-987,",
            "    //     stepping._apply_constitutive_pml:2083-2096) --------------",
            "    float prev = fw[idx];",
            "    float source = d_in[idx];",
            *chain,
            "    float src = source * inv_e[idx];",
            "    fw[idx] = src;",
            "    float value = e_out[idx];",
            f"    value = value + kps[{coordinate}] * src;",
            f"    value = value - kms[{coordinate}] * prev;",
            "    e_out[idx] = value;",
        ]
        drive = "src"

    polarization: List[str] = [
        "",
        "    // --- update_P for every pole that drives this component ----------",
        "    // (stepping.update_P:1383-1384 -> dispersion.PolarizationState.update:679-691)",
        "    // The rotation is NOT here: the host performs it between launches,",
        "    // exactly as ade_update_p.py:353-355 already does. See the module",
        "    // docstring for the orbit measurement that licenses this shape.",
    ]
    for index in range(pole_count):
        polarization.extend(_ade_arm(index, sigma_is_volume[index], drive))

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__NP__": str(max(pole_count, 1)),
        "__SIGNATURE__": _signature(arm, pole_count, sigma_is_volume),
        "__BODY__": "\n".join(electric + polarization),
    })


def compile_fused_ade_chain(arm: str, axis: int, pole_count: int,
                            sigma_is_volume: Sequence[bool],
                            contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (arm, component, poles, sigmas, mode)."""
    return compile_source(fused_ade_chain_source(
        arm, axis, pole_count, sigma_is_volume, contract)).fused_ade_chain_component


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

#: The E-half predicate each arm conjoins, keyed by ``(arm, folded)``.
_ELECTRIC_COVERAGE = {
    ("no_pml", False): metal_stored_e_coverage,
    ("dispersive", False): metal_dispersive_e_coverage,
    ("dispersive", True): folded_dispersive_e_coverage,
}


def _sigma_flags(state: Any, component: str) -> bool:
    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    return bool(getattr(sigma, "shape", ()))


def fused_ade_chain_coverage(fields: Any, pml: Any, arm: str,
                             folded: bool = False,
                             residency: Any = None) -> Coverage:
    """May ONE dispatch per component span ``update_E`` -> ``update_P``?

    A conjunction of the two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said
    it — the construction ``metal_fused_dispersive_pair_coverage`` uses.

    THERE IS NO SOURCE CLAUSE, AND THE ABSENCE IS DELIBERATE. Every other fused
    product on this backend must be told the source inventory, because the driver
    injects INSIDE its seam and ignorance is not an empty set. This seam has no
    injection in it at all — ``driver.step`` runs ``update_E`` at :3304 and
    ``update_P`` at :3306 with nothing between — so requiring a declaration would
    refuse configurations the driver cannot break. The fusion matrix measures the
    same fact from the other end: ``E_to_P: 15 instances, 0 blocked by the source
    seam``.
    """
    reasons: List[str] = []
    electric_coverage = _ELECTRIC_COVERAGE.get((arm, bool(folded)))
    if electric_coverage is None:
        return Coverage(False, (
            f"no E-half predicate is registered for arm={arm!r} folded={folded!r}",))

    electric = electric_coverage(fields, pml, residency)
    if not electric.covered:
        reasons.extend(f"E half: {reason}" for reason in electric.reasons)
    polarization = metal_ade_update_p_coverage(fields, pml, residency)
    if not polarization.covered:
        reasons.extend(f"ADE half: {reason}" for reason in polarization.reasons)

    # THE SEAM CLAUSE. The per-component interleave is equivalent to the driver's
    # order only while update_E(c) reads NOTHING but component c's own volumes.
    # Both features that break that are already refused by the E half; they are
    # named again because the FACT being asserted is different — this one is about
    # the interleave, not about the arm's arithmetic.
    if bool(getattr(fields, "has_offdiagonal_epsilon", False)):
        reasons.append(
            "an off-diagonal chi1inv row makes update_E(c) read the other "
            "components' D - sum P volumes (stepping.py:991-997), so the "
            "per-component interleave is not the driver's order")
    if bool(getattr(fields, "has_nonlinearity", False)):
        reasons.append(
            "chi2/chi3 makes update_E(c) read the other components' D - sum P "
            "volumes (stepping.py:991-997), so the per-component interleave is "
            "not the driver's order")

    # THE POLE SETS MUST AGREE, PER COMPONENT. Both are fields.polarizations order
    # filtered by drives(component); a disagreement would pair pole k's
    # displacement subtraction with pole j's recurrence and produce a smooth,
    # plausible, wrong field.
    states = tuple(getattr(fields, "polarizations", ()) or ())
    try:
        order = _poles(fields)
    except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
        return Coverage(False, tuple(dict.fromkeys(
            reasons + [f"the pole partition is unreadable ({exc!r})"])))
    for component, _displacement, _axis, _name in E_TERMS:
        chain = tuple(order.get(component, ()))
        try:
            ade = tuple(state for state in states
                        if component in tuple(state.driven()))
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"{component}: driven() raised {exc!r}")
            continue
        if tuple(id(state) for state in chain) != tuple(id(state) for state in ade):
            reasons.append(
                f"{component}: the E half's pole order and the ADE half's entry "
                f"order disagree; the fused arms would be mispaired")
        volume = sum(1 for state in chain if _sigma_flags(state, component))
        bindings = binding_count(arm, len(chain), volume)
        if bindings > MAX_BUFFER_BINDINGS:
            reasons.append(
                f"{component}: {len(chain)} poles ({volume} with a volume sigma) "
                f"needs {bindings} bindings and this platform allows "
                f"{MAX_BUFFER_BINDINGS} (device.py:72)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Entry:
    """One component's whole fused binding set, resolved once at plan time."""

    component: str
    axis: int
    arm: str
    pole_count: int
    sigma_is_volume: Tuple[bool, ...]
    target: Any
    auxiliary: Any            # f_w_E<c> under PML; None otherwise
    displacement: Any
    inverse_epsilon: Any
    kps: Any
    kms: Any
    states: Tuple[Any, ...]
    sigmas: Tuple[Any, ...]   # the volume sigmas, in pole order
    params: Any               # the packed Params blob (a host array)


class MetalFusedAdeChainPlan:
    """Three dispatches that perform two sub-step passes, the drive never stored-and-reloaded.

    ``launches_per_run`` is THREE and is declared rather than inherited: the
    whole-step arbiter asserts the exact per-cycle launch count on every filled
    slot, and that assertion is what stops a slot passing by not executing.

    THE ROTATION IS PERFORMED HERE, AFTER EACH LAUNCH, and it is the SAME rotation
    ``MetalAdeUpdatePPlan.run`` performs (ade_update_p.py:353-355), which is
    ``dispersion.PolarizationState.update``'s (dispersion.py:689-691). It is not
    baked into the kernel and this plan does not want it to be: see the module
    docstring for the orbit measurement.
    """

    performs_device_work = True
    replaces_sub_steps = REPLACES
    launches_per_run = len(E_TERMS)

    __slots__ = ("entries", "residency", "volumes", "arm", "folded", "shape",
                 "_functions", "_tensors", "launches", "runs", "alias_checks")

    def __init__(self, entries: Sequence[_Entry], residency: Residency,
                 volumes: Sequence[str], arm: str, folded: bool,
                 shape: Sequence[int],
                 functions: Mapping[Tuple[str, int], Any],
                 tensors: Mapping[int, Any]) -> None:
        self.entries = tuple(entries)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.arm = str(arm)
        self.folded = bool(folded)
        self.shape = tuple(int(n) for n in shape)
        self._functions = dict(functions)
        self._tensors = dict(tensors)
        self.launches = 0
        self.runs = 0
        self.alias_checks = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._functions}))

    def run(self, contract: Optional[str] = None) -> None:
        """One complete E->P seam, in place, against the mirrors.

        THE DESTINATION CHAIN IS RESOLVED AHEAD OF EACH LAUNCH and the reference
        rotation applied only AFTER it succeeds — the discipline
        ``MetalAdeUpdatePPlan.run`` states (ade_update_p.py:19-23): caching
        semantic pointers would advance the wrong buffer from launch two onward.

        THE ALIAS CHECK IS NOT DECORATION. The whole licence for this shape is
        that a per-component launch never writes a buffer it reads, and the
        rotation moves those identities every step. The check is O(poles) integer
        comparisons over lists resolved for the launch anyway, and it is what turns
        the orbit measurement into an invariant this plan cannot violate silently.
        """
        mode = shaders.CONTRACT_OFF if contract is None else contract
        for entry in self.entries:
            function = self._functions.get((mode, entry.axis))
            if function is None:
                raise KeyError(
                    f"this plan holds no {mode!r} variant (it was built with "
                    f"{self.variants}); build it with contract_variants={(mode,)} "
                    f"rather than launching the pinned one")
            component = entry.component
            poles = [state.P[component] for state in entry.states]
            prevs = [state.P_prev[component] for state in entry.states]
            outs: List[Any] = []
            scratch_of: Dict[int, Any] = {}
            for position, state in enumerate(entry.states):
                outs.append(state._scratch)
                scratch_of[position] = state._scratch
            if len(poles) != entry.pole_count:
                raise RuntimeError(
                    f"{component} resolved {len(poles)} poles, not the compiled "
                    f"count {entry.pole_count}")
            written = {id(value) for value in outs}
            read = {id(value) for value in poles} | {id(value) for value in prevs}
            if written & read:
                raise RuntimeError(
                    f"{component}: the rotation put a launch OUTPUT on one of its "
                    f"own INPUTS; this shape's whole licence is that it cannot "
                    f"(see the module docstring's orbit measurement)")
            if len(written) != len(outs):
                raise RuntimeError(
                    f"{component}: two poles resolved the SAME output buffer, so "
                    f"one recurrence would overwrite the other's result in the "
                    f"same launch; each susceptibility owns its own scratch "
                    f"(dispersion.py:646-650) and this cannot happen on an engine "
                    f"object, which is why it is checked rather than assumed")
            self.alias_checks += 1

            arguments: List[Any] = [self._tensors[id(entry.target)]]
            if entry.arm == "dispersive":
                arguments.append(self._tensors[id(entry.auxiliary)])
            arguments.append(self._tensors[id(entry.displacement)])
            arguments.append(self._tensors[id(entry.inverse_epsilon)])
            if entry.arm == "dispersive":
                arguments.append(self._tensors[id(entry.kps)])
                arguments.append(self._tensors[id(entry.kms)])
            arguments.extend(self._tensors[id(value)] for value in poles)
            arguments.extend(self._tensors[id(value)] for value in outs)
            arguments.extend(self._tensors[id(value)] for value in prevs)
            arguments.extend(self._tensors[id(value)] for value in entry.sigmas)
            arguments.append(self._tensors[id(entry.params)])
            function(*arguments)
            self.launches += 1

            # dispersion.py:689-691, via ade_update_p.py:353-355. AFTER the launch.
            for position, state in enumerate(entry.states):
                state.P[component] = scratch_of[position]
                state.P_prev[component] = poles[position]
                state._scratch = prevs[position]
        self.runs += 1

    def __repr__(self) -> str:
        return (f"MetalFusedAdeChainPlan(arm={self.arm!r}, folded={self.folded}, "
                f"entries={len(self.entries)}, variants={self.variants!r})")


def plan_metal_fused_ade_chain(
        fields: Any, pml: Any, arm: str, folded: bool = False,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[Tuple[str, int], Any]] = None,
        ) -> Optional[MetalFusedAdeChainPlan]:
    """Build the fused E->P plan for one arm, or return ``None`` on any refusal."""
    if not fused_ade_chain_coverage(fields, pml, arm, folded, residency).covered:
        return None
    assert residency is not None
    tensors: Dict[int, Any] = {}
    volumes: List[str] = []
    entries: List[_Entry] = []
    required: Dict[Tuple[str, int], Tuple[str, int, int, Tuple[bool, ...]]] = {}
    order = _poles(fields)
    states = tuple(getattr(fields, "polarizations", ()) or ())
    index_of = {id(state): position for position, state in enumerate(states)}
    shape = tuple(int(n) for n in fields.grid.shape)

    for component, displacement_name, axis, axis_name in E_TERMS:
        target = getattr(fields, component)
        displacement = getattr(fields, displacement_name)
        inverse = fields.inverse_epsilon_for(component)
        auxiliary = kps = kms = None
        bindings: List[Tuple[str, Any, bool]] = [
            (component, target, False), (displacement_name, displacement, False)]
        if arm == "dispersive":
            auxiliary = getattr(fields, "f_w_" + component)
            kps = getattr(pml, f"kps_{axis_name}_h")
            kms = getattr(pml, f"kms_{axis_name}_h")
            bindings.append(("f_w_" + component, auxiliary, False))
            # The SAME residency names the certified pointwise plan registers, so
            # a composition that holds both binds one tensor rather than two.
            bindings.append((f"inv_eps_{component}", inverse, True))
            bindings.append((f"pml:kps_{axis_name}_h", kps, True))
            bindings.append((f"pml:kms_{axis_name}_h", kms, True))
        else:
            bindings.append((f"inv_eps_{component}:no_pml", inverse, True))
        for name, host, constant in bindings:
            tensors[id(host)] = residency.mirror(name, host, constant=constant)
            volumes.append(name)

        chain = tuple(order.get(component, ()))
        sigma_flags = tuple(_sigma_flags(state, component) for state in chain)
        sigmas: List[Any] = []
        coefficients: List[Tuple[float, float, float]] = []
        scalar_sigmas: List[float] = []
        for position, state in enumerate(chain):
            state_index = index_of[id(state)]
            for label, host in ((f"P:{component}", state.P[component]),
                                (f"P_prev:{component}", state.P_prev[component])):
                name = _physical_name(state_index, label, host)
                tensors[id(host)] = residency.mirror(name, host)
                volumes.append(name)
            scratch = state._scratch
            name = _physical_name(state_index, "scratch", scratch)
            tensors[id(scratch)] = residency.mirror(name, scratch)
            volumes.append(name)
            sigma = state.sigma[component]
            if sigma_flags[position]:
                sigma_name = _physical_name(
                    state_index, f"sigma:{component}", sigma)
                tensors[id(sigma)] = residency.mirror(
                    sigma_name, sigma, constant=True)
                volumes.append(sigma_name)
                sigmas.append(sigma)
                scalar_sigmas.append(0.0)
            else:
                scalar_sigmas.append(float(sigma))
            c_now, c_prev, c_drive = state._coefficients
            coefficients.append((float(c_now), float(c_prev), float(c_drive)))

        params = pack_params(coefficients, scalar_sigmas, shape[1], shape[2],
                             int(target.size))
        params_name = f"fused_ade_chain:params:{component}"
        tensors[id(params)] = residency.mirror(params_name, params, constant=True)
        volumes.append(params_name)

        entries.append(_Entry(
            component, axis, arm, len(chain), sigma_flags, target, auxiliary,
            displacement, inverse, kps, kms, chain, tuple(sigmas), params))
        for mode in contract_variants:
            required[(mode, axis)] = (arm, axis, len(chain), sigma_flags)

    selected = dict(functions or {})
    for key, (arm_name, axis, pole_count, flags) in sorted(required.items()):
        if key not in selected:
            selected[key] = compile_fused_ade_chain(
                arm_name, axis, pole_count, flags, key[0])
    return MetalFusedAdeChainPlan(entries, residency, volumes, arm, folded,
                                  shape, selected, tensors)


# ---------------------------------------------------------------------------
# Registration -- three arms, none wired
# ---------------------------------------------------------------------------

#: label -> (family, arm, folded). Three registered arms under THREE family names,
#: one per E->P cell this module claims. The two cells it does NOT claim — complex
#: no-PML stored E and folded off-diagonal dispersive PML E — are absent by name,
#: not by omission; see the module docstring.
REGISTERED_ARMS: Dict[str, Tuple[str, str, bool]] = {
    "fused no-PML stored E -> ADE": (FAMILY, "no_pml", False),
    "fused dispersive PML E -> ADE": (DISPERSIVE_FAMILY, "dispersive", False),
    "fused folded dispersive PML E -> ADE": (FOLDED_FAMILY, "dispersive", True),
}


def _arm_coverage(context: Any, slot: str, arm: str, folded: bool) -> Coverage:
    if slot != SLOT:
        return Coverage(False, (f"the fused ADE chain cannot fill slot {slot!r}",))
    return fused_ade_chain_coverage(context.fields, context.pml, arm, folded,
                                    context.residency)


def _arm_plan(context: Any, slot: str, arm: str, folded: bool
              ) -> Optional[MetalFusedAdeChainPlan]:
    if slot != SLOT:
        return None
    return plan_metal_fused_ade_chain(
        context.fields, context.pml, arm, folded, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    specs: List[Any] = []
    for label, (family, arm, folded) in REGISTERED_ARMS.items():
        specs.append(arms.register(
            family=family,
            slot=SLOT,
            label=label,
            coverage=(lambda context, slot, arm=arm, folded=folded:
                      _arm_coverage(context, slot, arm, folded)),
            plan=(lambda context, slot, arm=arm, folded=folded:
                  _arm_plan(context, slot, arm, folded)),
            prefix=f"{label}: ",
            noun="fused E->P chain",
            wired=False,
            replaces=REPLACES,
        ))
    return tuple(specs)


ARMS = register_arms()
