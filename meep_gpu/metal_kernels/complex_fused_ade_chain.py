"""Complex ``update_E`` welded to complex ``update_P`` — the fourth E->P product.

THE COMPLEX64 SIBLING of :mod:`.fused_ade_chain`, in exactly the relation
:mod:`.complex_no_pml_stored_e` has to :mod:`.no_pml_stored_e`: a separate family,
a separate predicate, the same seam and the same shape. It claims the ONE E->P
cell that module names as unclaimed — "**complex64 storage** — a separate signature
and a separate arm; the four ``TestLoadDump.*_3d`` rows are not claimed here"
(fused_ade_chain.py, "WHAT THIS FAMILY REFUSES").

===========================================================================
WHAT IT IS WORTH: +4 SEAM-INSTANCES OF 387, RECOMPUTED
===========================================================================

MEASURED, not inferred from the ranked-gap table's reachable ceiling.
``results/fusion_matrix_metal_2026-08-20_unbuilt_cells/`` re-runs the standing
CLOSED cut (``results/fusion_matrix_metal_2026-08-20_closed``, 118/387) with this
product and one other added and nothing else changed; E->P moves 10 -> 14 and the
total 118 -> 126. The ladder is short because the cell has no attrition at all —
this seam carries no source injection, so all four rows that admit both halves
survive::

    complex_stored_e@update_E     4
    ade_update_p@update_P         4
    no_offdiagonal_epsilon        4
    no_chi2_chi3                  4
    fits_the_binding_ceiling      4

The four rows are ``TestLoadDump.test_load_dump_{structure,structure_sharded,
chunk_layout_file,chunk_layout_sim}_3d``, and they are the SAME four
:mod:`.complex_conductive_fused_pair` serves at D->E.

THE SEAM IS EMPTY, and that is again the whole reason the product exists.
``FdtdDriver.step`` runs::

    if fast is None or not fast.dispatch("update_E", self.fields):
        update_E(self.fields, self.pml)                       # driver.py:3304
    if fast is None or not fast.dispatch("update_P", self.fields):
        update_P(self.fields, self.pml)                       # driver.py:3306

Nothing sits between them, which is why the Metal fusion matrix records
``E_to_P: 15 instances, 0 blocked by the source seam, CEILING 15``.

===========================================================================
WHAT IS FUSED, AND IT IS EXACTLY ONE VALUE
===========================================================================

``update_P``'s drive is ``Fields.drive_field(component)``: without PML that is the
STORED E, the value ``update_E`` computed one launch earlier. The certified
complex E body stores the product without naming it::

    e_out[idx] = c_mul_field_left(source, inv_e[idx]);   # complex_no_pml_stored_e.py:80

Fused, that ONE line becomes two — ``float2 e = c_mul_field_left(source,
inv_e[idx]); e_out[idx] = e;`` — and the ADE half's ``float2 w = drive[idx];``
becomes ``float2 w = e;``. A ``float2`` stored to a ``device float2*`` and
reloaded is bit-identical to the register, so the split is byte-neutral by
construction; the gate's ``byte_neutral_control`` leg MEASURES it anyway, per
complete seam step, because "by construction" is a hypothesis until a comparator
agrees.

EVERY OTHER LINE IS TRANSCRIBED CHARACTER FOR CHARACTER from the body it replaces,
with the pointer renames named below and nothing else. The lines are not restated
anywhere in this module as data; ``test_metal_complex_fused_ade_chain`` CALLS each
certified emitter, searches its output for the line, and searches this family's
output for the same line with the documented rename applied — so the check parses
both sources and mirrors neither.

===========================================================================
THE ONE FACT THIS FAMILY MUST NOT INHERIT: WHICH COMPLEX ARM
===========================================================================

Both halves multiply complex numbers, and WHICH EXPANSION a platform's reference
takes is a MEASURED fact, not a choice (:func:`.complex_fields.expansion_from_probe`).
The two certified halves disagree about where that fact comes from:

* ``complex_no_pml_stored_e`` takes the arm the PROBE licenses, threaded through
  its plan (complex_no_pml_stored_e.py:77, :208);
* ``ade_update_p`` BAKES ``FMA_V1`` into its complex body (ade_update_p.py:105).

One fused source carries ONE set of helpers under one set of names, so a platform
whose probe licensed ``NAIVE`` would need the two halves to disagree inside a
single scope, which is unspellable. On this host the probe licenses ``FMA_V1`` and
the two agree — but agreeing today is a coincidence, not an invariant, so this
module does not assume it:

* :func:`ade_expansion_arm` PARSES the certified ADE emitter's own output and
  reports which arm's helper block it carries. It does not restate ``FMA_V1``
  anywhere, so a change to ade_update_p.py:105 moves this module's answer with it;
* :func:`complex_fused_ade_chain_coverage` REFUSES BY NAME when the probe-licensed
  arm is not that one. A configuration on such a platform earns a new arm rather
  than a silent downgrade of one half.

===========================================================================
THE ROTATION, AND WHAT IS MEASURED RATHER THAN INHERITED
===========================================================================

``ade_update_p`` launches once per DRIVEN COMPONENT and the host rotates
``P`` / ``P_prev`` / ``_scratch`` after each launch (ade_update_p.py:13-18, the
three assignments at :353-355 transcribing dispersion.py:689-691). This family
takes the shape :mod:`.fused_ade_chain` established: ONE LAUNCH PER COMPONENT,
the rotation left exactly where the certified plan already performs it, so the
fused kernel inherits no intra-launch alias.

WHAT IS **NOT** INHERITED IS THE LICENCE. ``probe_metal_ade_rotation_seam`` — the
orbit walk and the per-component-interleave equivalence that
:mod:`.fused_ade_chain` rests on — SKIPPED this cell by name: "this leg prices the
two REAL diagonal E bodies; the complex and off-diagonal arms carry different
signatures and are not claimed here" (probe_metal_ade_rotation_seam.py:579-584).
So the two facts are re-established here on complex storage:

* **the interleave is the driver's order.** The equivalence holds because this
  ``update_E`` reads only its OWN component's ``d_in``, ``inv_e`` and ``p`` volumes
  (no off-diagonal row, no ``chi3`` magnitude — both REFUSED by the E half) and
  ``update_P(state, c)`` advances only ``P[c]``. The gate's ``separate_control``
  leg runs the two ALREADY CERTIFIED complex products in the DRIVER's order beside
  this one on identical state and requires three-way word agreement, over sixty
  complete seam steps, which is what turns that paragraph into a measurement;
* **the rotation is alias-free per component**, and :meth:`MetalComplexFusedAde
  ChainPlan.run` CHECKS it before every launch rather than trusting the orbit: the
  write set ``{out_k[c]}`` must be disjoint from the read set ``{P_k[c],
  P_prev_k[c]}``. It is O(poles) integer comparisons over lists resolved for the
  launch anyway.

===========================================================================
THE POINTER RENAMES, AND WHY THE SCALARS MOVE INTO A STRUCT
===========================================================================

**POINTER RENAMES.** The E half names its pole buffers ``p0..p7``
(complex_no_pml_stored_e.py:53-60) and the ADE half names its three volumes
``p_now`` / ``p_prev`` / ``p_out`` (ade_update_p.py:69-71). Fused, ``p0`` would
mean two things in one scope, so the ADE volumes become ``o{k}`` (out), ``q{k}``
(prev) and ``s{k}`` (sigma) while ``p{k}`` stays the E half's spelling — AND IS
THE SHARED POINTER. ``p_now`` is not bound a second time: pole ``k``'s ``P[c]`` is
the same array the E chain subtracts, so the fused signature binds it once. That
shared binding is the fusion's whole footprint.

**THE SCALARS.** :data:`.device.MAX_BUFFER_BINDINGS` is 31. Bound one per index,
a component at ``N`` poles would need ``3 + 3N + V`` pointers PLUS ``3N``
coefficient scalars, the scalar sigmas and ``n_elem`` — 46 bindings at six poles,
against a ceiling of 31. Packing every scalar into ONE ``constant Params&`` — the
shape :mod:`.fused_dispersive_pair` measured to compile, and the one
:mod:`.fused_ade_chain` already uses at this seam — brings the same configuration
to 28. :func:`binding_count` is evaluated BY THE PREDICATE, so a configuration
over the ceiling is a REFUSAL BY NAME rather than a compile error at plan time.

**THE PACKER IS IMPORTED, NOT RESPELLED.** :func:`.fused_ade_chain.pack_params`
and :func:`.fused_ade_chain.params_words` lay out exactly the blob this family
needs, and a second implementation of a measured layout is one more thing to
drift. ``ny`` and ``nz`` ride along unread: this E body is POINTWISE and has no
``DECODE_IJK``, so only ``n_elem`` is copied back into a local, and the prologue
that does it exists so :data:`.templates.GUARD` below is the certified emitters'
own text.

THE POLE SLOTS ARE NOT PADDED TO EIGHT. ``complex_no_pml_stored_e`` declares
``p0..p7`` always and binds the displacement into the unused slots
(complex_no_pml_stored_e.py:196). Here the signature declares exactly
``pole_count`` slots, because the pad is what puts this over the ceiling. The body
is already specialised per pole count in the certified family, so nothing about
the ARITHMETIC changes with the declaration.

===========================================================================
WHAT THIS FAMILY REFUSES
===========================================================================

* **real (float32) storage** — that is :mod:`.fused_ade_chain`'s ``no_pml`` arm,
  and the E half refuses it already;
* **an active PML layer** — the drive would be ``f_w_E*`` rather than the stored E,
  which is a different pointer and a different cell inside the absorber; refused
  by both halves;
* **an off-diagonal chi1inv row or a chi2/chi3** — refused by BOTH halves already,
  and named again here as a SEAM clause because the reason is different: the
  interleave's equivalence rests on ``update_E(c)`` reading only component ``c``'s
  volumes, and both features read the OTHER components' ``D - sum P``
  (stepping.py:991-997);
* **a pole set the two halves disagree on** — the E half's pole order for a
  component and the ADE plan's entry order for that component are both
  ``fields.polarizations`` order filtered by ``drives(component)``. The plan
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
from . import ade_update_p, shaders, templates
from .ade_update_p import _physical_name, metal_ade_update_p_coverage
from .complex_fields import expansion_from_probe, load_expansion_probe
from .complex_no_pml_stored_e import (
    _poles,
    metal_complex_stored_e_coverage,
)
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .fused_ade_chain import pack_params, params_words

#: The family's own name. ONE cell, ONE arm — unlike :mod:`.fused_ade_chain`,
#: which carries three cells under three family names in one module.
FAMILY = "complex_fused_ade_chain"

#: The slot the arm is registered on. The product spans two; it holds a row on the
#: FIRST, so a refusal is named on the slot the fusion starts at rather than being
#: invisible to the table.
SLOT = "update_E"

#: The driver passes one run of this plan performs, in driver order
#: (driver.py:3304-3306). Declared rather than inferred from the slot name.
REPLACES: Tuple[str, ...] = ("update_E", "update_P")

#: The E half's fixed (non-pole) POINTER contribution: ``e_out, d_in, inv_e``
#: (complex_no_pml_stored_e.py:50-52), minus the eight-slot pole pad.
FIXED_POINTERS = 3

#: ``(component, displacement, axis)`` — the table the certified complex E body
#: iterates (complex_no_pml_stored_e.E_TERMS), with the axis joined on so one loop
#: serves the plan.
E_TERMS: Tuple[Tuple[str, str, int], ...] = (
    ("Ex", "Dx", 0), ("Ey", "Dy", 1), ("Ez", "Dz", 2))

__all__ = [
    "E_TERMS", "FAMILY", "FIXED_POINTERS", "REPLACES", "SLOT",
    "MetalComplexFusedAdeChainPlan", "ade_expansion_arm", "binding_count",
    "compile_complex_fused_ade_chain", "complex_fused_ade_chain_coverage",
    "complex_fused_ade_chain_source", "plan_metal_complex_fused_ade_chain",
]


# ---------------------------------------------------------------------------
# Which complex arm the certified ADE half carries — PARSED, never restated
# ---------------------------------------------------------------------------

def ade_expansion_arm() -> str:
    """The expansion arm ``ade_update_p``'s complex body is emitted with.

    MEASURED FROM THE CERTIFIED EMITTER'S OWN OUTPUT, not copied from
    ade_update_p.py:105. The helper block for each arm is a distinct literal
    (:data:`.templates._COMPLEX_HELPERS`), so asking which one appears in the
    emitted source is a parse, not a mirror — and a change on that line moves this
    answer with it rather than leaving a stale constant behind.

    Raises when the emitted source carries no arm's helpers, or more than one:
    either would mean this module cannot say which arithmetic the ADE half does,
    and guessing is exactly what the probe machinery exists to stop.
    """
    emitted = ade_update_p.ade_source("complex64", False)
    found = [arm for arm in templates.EXPANSION_ARMS
             if templates.complex_helpers(arm) in emitted]
    if len(found) != 1:
        raise RuntimeError(
            f"the certified complex ADE body carries {found!r} of the known "
            f"expansion arms {templates.EXPANSION_ARMS!r}; this family cannot "
            f"weld a half whose complex arithmetic it cannot name")
    return found[0]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

struct Params {
    float c_now[__NP__];
    float c_prev[__NP__];
    float c_drive[__NP__];
    float sigma[__NP__];
    uint  ny;
    uint  nz;
    uint  n_elem;
};

kernel void complex_fused_ade_chain_component(
__SIGNATURE__
    uint idx [[thread_position_in_grid]])
{
    // The scalars live in ONE binding (see the module docstring's binding
    // arithmetic); copying n_elem back into the name the certified emitters use is
    // what lets GUARD below be those emitters' own text. ny and nz are packed by
    // the SHARED packer and unread here: this E body is pointwise.
    uint n_elem = prm.n_elem;
__BODY__
}
"""


def binding_count(pole_count: int, volume_sigmas: int) -> int:
    """Buffer attribute indices the fused signature would occupy.

    ``pointers = 3 + 3 * poles + volume_sigmas`` — the E half's own fixed volumes
    (``e_out, d_in, inv_e``), then per pole ``p`` (SHARED between the halves), ``o``
    and ``q``, then an ``s`` only for the poles whose sigma is a volume — plus ONE
    for the packed ``Params&``.
    """
    pole_count = int(pole_count)
    if pole_count < 0:
        raise ValueError(f"pole_count must be non-negative, got {pole_count!r}")
    if not 0 <= int(volume_sigmas) <= pole_count:
        raise ValueError(
            f"volume_sigmas={volume_sigmas!r} is not in [0, {pole_count!r}]")
    return (FIXED_POINTERS + 3 * pole_count + int(volume_sigmas)) + 1


def _signature(pole_count: int, sigma_is_volume: Sequence[bool]) -> str:
    """Every binding, in the order :meth:`MetalComplexFusedAdeChainPlan.run` passes them."""
    lines: List[str] = []
    slot = 0

    def bind(declaration: str) -> None:
        nonlocal slot
        lines.append(f"    {declaration:<33}[[buffer({slot})]],")
        slot += 1

    bind("device float2*       e_out")
    bind("device const float2* d_in")
    bind("device const float*  inv_e")
    for index in range(pole_count):        # SHARED: the E half's pole AND p_now.
        bind(f"device const float2* p{index}")
    for index in range(pole_count):
        bind(f"device float2*       o{index}")
    for index in range(pole_count):
        bind(f"device const float2* q{index}")
    for index in range(pole_count):
        if sigma_is_volume[index]:
            bind(f"device const float*  s{index}")
    bind("constant Params&     prm")
    assert slot == binding_count(
        pole_count, sum(1 for flag in sigma_is_volume if flag)), (
        slot, pole_count, tuple(sigma_is_volume))
    return "\n".join(lines)


def _ade_arm(index: int, sigma_is_volume: bool, drive_register: str) -> List[str]:
    """One pole's ``update_P``, transcribed from ``ade_update_p.ade_source``.

    ``ade_update_p.py:106-114`` is the complex64 body::

        float2 p = p_now[idx];
        float2 q = p_prev[idx];
        float2 w = drive[idx];
        float s = __SIGMA_VALUE__;
        float2 a = c_mul_field_left(p, c_now);
        float2 b = c_mul_coefficient_left(c_prev, q);
        float2 sw = c_mul_coefficient_left(s, w);
        float2 d = c_mul_coefficient_left(c_drive, sw);
        p_out[idx] = (a + b) + d;

    Reproduced here with the pointer renames the module docstring names and ONE
    substitution — ``drive[idx]`` becomes the register ``update_E`` just wrote.
    The braces give each arm its own ``p``/``q``/``w``/``s``/``a``/``b``/``sw``/
    ``d``/``c_now``/``c_prev``/``c_drive`` scope, so the final five lines are
    character for character that body's.
    """
    sigma = f"s{index}[idx]" if sigma_is_volume else f"prm.sigma[{index}]"
    return [
        f"    {{  // pole {index}: ade_update_p.ade_source, complex64 body (:106-114)",
        f"        float c_now = prm.c_now[{index}];",
        f"        float c_prev = prm.c_prev[{index}];",
        f"        float c_drive = prm.c_drive[{index}];",
        f"        float2 p = p{index}[idx];",
        f"        float2 q = q{index}[idx];",
        f"        float2 w = {drive_register};   // THE SEAM: the register update_E just wrote.",
        f"        float s = {sigma};",
        "        float2 a = c_mul_field_left(p, c_now);",
        "        float2 b = c_mul_coefficient_left(c_prev, q);",
        "        float2 sw = c_mul_coefficient_left(s, w);",
        "        float2 d = c_mul_coefficient_left(c_drive, sw);",
        f"        o{index}[idx] = (a + b) + d;",
        "    }",
    ]


def complex_fused_ade_chain_source(pole_count: int,
                                   sigma_is_volume: Sequence[bool],
                                   expansion: str,
                                   contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (poles, sigma kinds, expansion arm, mode).

    Every constant Triton would bake into a ``tl.constexpr`` is baked into the
    string, for the reason :mod:`.shaders` gives: ``torch.mps.compile_shader``
    takes a source and nothing else.
    """
    pole_count = int(pole_count)
    if pole_count < 0:
        raise ValueError(f"pole_count must be non-negative, got {pole_count!r}")
    sigma_is_volume = tuple(bool(flag) for flag in sigma_is_volume)
    if len(sigma_is_volume) != pole_count:
        raise ValueError(
            f"sigma_is_volume must carry one flag per pole, got "
            f"{len(sigma_is_volume)} for {pole_count} poles")
    bindings = binding_count(pole_count,
                             sum(1 for flag in sigma_is_volume if flag))
    if bindings > MAX_BUFFER_BINDINGS:
        raise ValueError(
            f"the complex fused signature at {pole_count} poles needs {bindings} "
            f"bindings and this platform allows {MAX_BUFFER_BINDINGS} "
            f"(device.py:72); the predicate refuses this configuration by name "
            f"rather than letting it reach the compiler")

    # --- complex_no_pml_stored_e.complex_stored_e_source (:78-80) -------------
    # ONE line differs and it is THE SEAM: the certified body's
    # `e_out[idx] = c_mul_field_left(source, inv_e[idx]);` is split so the product
    # has a name the ADE half can read from a register. A float2 stored to a
    # device float2* and reloaded is exact, so the split is byte-neutral; the gate
    # measures it per complete seam step regardless.
    electric = [
        templates.GUARD,
        "",
        "    // --- pole-aware complex update_E (stepping.update_E:981-984) -----",
        "    float2 source = d_in[idx];",
        *[f"    source = source - p{index}[idx];" for index in range(pole_count)],
        "    // THE SEAM: name the constitutive product so update_P reads a register.",
        "    float2 e = c_mul_field_left(source, inv_e[idx]);",
        "    e_out[idx] = e;",
    ]

    polarization: List[str] = [
        "",
        "    // --- update_P for every pole that drives this component ----------",
        "    // (stepping.update_P:1383-1384 -> dispersion.PolarizationState.update:679-691)",
        "    // The rotation is NOT here: the host performs it between launches,",
        "    // exactly as ade_update_p.py:353-355 already does. See the module",
        "    // docstring for what this family measures rather than inherits.",
    ]
    for index in range(pole_count):
        polarization.extend(_ade_arm(index, sigma_is_volume[index], "e"))

    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__NP__": str(max(pole_count, 1)),
        "__SIGNATURE__": _signature(pole_count, sigma_is_volume),
        "__BODY__": "\n".join(electric + polarization),
    })


def compile_complex_fused_ade_chain(pole_count: int,
                                    sigma_is_volume: Sequence[bool],
                                    expansion: str,
                                    contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one (poles, sigmas, expansion, mode)."""
    return compile_source(complex_fused_ade_chain_source(
        pole_count, sigma_is_volume, expansion,
        contract)).complex_fused_ade_chain_component


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _sigma_is_volume(state: Any, component: str) -> bool:
    sigma = (getattr(state, "sigma", {}) or {}).get(component)
    return bool(getattr(sigma, "shape", ()))


def complex_fused_ade_chain_coverage(fields: Any, pml: Any,
                                     residency: Any = None,
                                     probe: Any = None) -> Coverage:
    """May ONE dispatch per component span complex ``update_E`` -> ``update_P``?

    A conjunction of the two halves' OWN certified predicates plus the seam
    clauses. Nothing is weakened: a configuration either half refuses is refused
    here with that half's reasons, prefixed so a reader can tell which side said
    it — the construction :func:`.fused_ade_chain.fused_ade_chain_coverage` uses.

    THERE IS NO SOURCE CLAUSE, AND THE ABSENCE IS DELIBERATE. Every fused product
    at the two curl seams must be told the source inventory, because the driver
    injects INSIDE those seams and ignorance is not an empty set. This seam has no
    injection in it at all — ``FdtdDriver.step`` runs ``update_E`` at :3304 and
    ``update_P`` at :3306 with nothing between — so requiring a declaration would
    refuse configurations the driver cannot break.
    """
    reasons: List[str] = []
    electric = metal_complex_stored_e_coverage(fields, pml, residency, probe)
    if not electric.covered:
        reasons.extend(f"E half: {reason}" for reason in electric.reasons)
    polarization = metal_ade_update_p_coverage(fields, pml, residency)
    if not polarization.covered:
        reasons.extend(f"ADE half: {reason}" for reason in polarization.reasons)

    # THE ARM CLAUSE. One fused source carries one set of helpers under one set of
    # names; a platform whose probe licenses an arm the certified ADE half does not
    # bake in cannot be served by this signature at all. See the module docstring.
    record = probe if probe is not None else load_expansion_probe()
    licensed = expansion_from_probe(record)
    try:
        baked = ade_expansion_arm()
    except RuntimeError as exc:  # noqa: BLE001 - unnameable is a refusal
        reasons.append(str(exc))
        baked = None
    if licensed is not None and baked is not None and licensed != baked:
        reasons.append(
            f"the probe licenses the {licensed!r} complex arm but the certified "
            f"ADE half bakes in {baked!r} (ade_update_p.py:105); one fused source "
            f"cannot carry two arms under one set of helper names, so this "
            f"platform earns a new arm rather than a silent downgrade of a half")

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
    for component, _displacement, _axis in E_TERMS:
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
        volume = sum(1 for state in chain if _sigma_is_volume(state, component))
        bindings = binding_count(len(chain), volume)
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
    pole_count: int
    sigma_is_volume: Tuple[bool, ...]
    target: Any
    displacement: Any
    inverse_epsilon: Any
    states: Tuple[Any, ...]
    sigmas: Tuple[Any, ...]   # the volume sigmas, in pole order
    params: Any               # the packed Params blob (a host array)


class MetalComplexFusedAdeChainPlan:
    """Three dispatches that perform two sub-step passes, the drive never stored-and-reloaded.

    ``launches_per_run`` is THREE and is declared rather than inherited: the
    whole-step arbiter asserts the exact per-cycle launch count on every filled
    slot, and that assertion is what stops a slot passing by not executing.

    THE ROTATION IS PERFORMED HERE, AFTER EACH LAUNCH, and it is the SAME rotation
    ``MetalAdeUpdatePPlan.run`` performs (ade_update_p.py:353-355), which is
    ``dispersion.PolarizationState.update``'s (dispersion.py:689-691).
    """

    performs_device_work = True
    replaces_sub_steps = REPLACES
    launches_per_run = len(E_TERMS)

    __slots__ = ("entries", "residency", "volumes", "expansion", "shape",
                 "_functions", "_tensors", "launches", "runs", "alias_checks")

    def __init__(self, entries: Sequence[_Entry], residency: Residency,
                 volumes: Sequence[str], expansion: str, shape: Sequence[int],
                 functions: Mapping[Tuple[str, int], Any],
                 tensors: Mapping[int, Any]) -> None:
        self.entries = tuple(entries)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.expansion = str(expansion)
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
        rotation moves those identities every step.
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
                    f"(see the module docstring's rotation section)")
            if len(written) != len(outs):
                raise RuntimeError(
                    f"{component}: two poles resolved the SAME output buffer, so "
                    f"one recurrence would overwrite the other's result in the "
                    f"same launch; each susceptibility owns its own scratch "
                    f"(dispersion.py:646-650) and this cannot happen on an engine "
                    f"object, which is why it is checked rather than assumed")
            self.alias_checks += 1

            arguments: List[Any] = [
                self._tensors[id(entry.target)],
                self._tensors[id(entry.displacement)],
                self._tensors[id(entry.inverse_epsilon)],
            ]
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
        return (f"MetalComplexFusedAdeChainPlan(entries={len(self.entries)}, "
                f"expansion={self.expansion!r}, variants={self.variants!r})")


def plan_metal_complex_fused_ade_chain(
        fields: Any, pml: Any, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        probe: Any = None,
        functions: Optional[Mapping[Tuple[str, int], Any]] = None,
        ) -> Optional[MetalComplexFusedAdeChainPlan]:
    """Build the complex fused E->P plan, or return ``None`` on any refusal."""
    if not complex_fused_ade_chain_coverage(fields, pml, residency, probe).covered:
        return None
    assert residency is not None
    expansion = expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
    if expansion is None:
        return None
    tensors: Dict[int, Any] = {}
    volumes: List[str] = []
    entries: List[_Entry] = []
    required: Dict[Tuple[str, int], Tuple[int, Tuple[bool, ...]]] = {}
    order = _poles(fields)
    states = tuple(getattr(fields, "polarizations", ()) or ())
    index_of = {id(state): position for position, state in enumerate(states)}
    shape = tuple(int(n) for n in fields.grid.shape)

    for component, displacement_name, axis in E_TERMS:
        target = getattr(fields, component)
        displacement = getattr(fields, displacement_name)
        inverse = fields.inverse_epsilon_for(component)
        # The SAME residency names the certified pointwise plan registers, so a
        # composition that holds both binds one tensor rather than two.
        for name, host, constant, dtype in (
                (component, target, False, np.complex64),
                (displacement_name, displacement, False, np.complex64),
                (f"inv_eps_{component}:no_pml", inverse, True, np.float32)):
            tensors[id(host)] = residency.mirror(
                name, host, constant=constant, dtype=dtype)
            volumes.append(name)

        chain = tuple(order.get(component, ()))
        sigma_flags = tuple(_sigma_is_volume(state, component) for state in chain)
        sigmas: List[Any] = []
        coefficients: List[Tuple[float, float, float]] = []
        scalar_sigmas: List[float] = []
        for position, state in enumerate(chain):
            state_index = index_of[id(state)]
            for label, host in ((f"P:{component}", state.P[component]),
                                (f"P_prev:{component}", state.P_prev[component])):
                name = _physical_name(state_index, label, host)
                tensors[id(host)] = residency.mirror(name, host, dtype=np.complex64)
                volumes.append(name)
            scratch = state._scratch
            name = _physical_name(state_index, "scratch", scratch)
            tensors[id(scratch)] = residency.mirror(
                name, scratch, dtype=np.complex64)
            volumes.append(name)
            sigma = state.sigma[component]
            if sigma_flags[position]:
                sigma_name = _physical_name(
                    state_index, f"sigma:{component}", sigma)
                tensors[id(sigma)] = residency.mirror(
                    sigma_name, sigma, constant=True, dtype=np.float32)
                volumes.append(sigma_name)
                sigmas.append(sigma)
                scalar_sigmas.append(0.0)
            else:
                scalar_sigmas.append(float(sigma))
            c_now, c_prev, c_drive = state._coefficients
            coefficients.append((float(c_now), float(c_prev), float(c_drive)))

        # THE PACKER IS THE FLOAT32 FAMILY'S OWN — see the module docstring.
        params = pack_params(coefficients, scalar_sigmas, shape[1], shape[2],
                             int(target.size))
        assert params.size == params_words(len(chain)), (
            params.size, params_words(len(chain)))
        params_name = f"complex_fused_ade_chain:params:{component}"
        tensors[id(params)] = residency.mirror(params_name, params, constant=True)
        volumes.append(params_name)

        entries.append(_Entry(
            component, axis, len(chain), sigma_flags, target, displacement,
            inverse, chain, tuple(sigmas), params))
        for mode in contract_variants:
            required[(mode, axis)] = (len(chain), sigma_flags)

    selected = dict(functions or {})
    for key, (pole_count, flags) in sorted(required.items()):
        if key not in selected:
            selected[key] = compile_complex_fused_ade_chain(
                pole_count, flags, expansion, key[0])
    return MetalComplexFusedAdeChainPlan(entries, residency, volumes, expansion,
                                         shape, selected, tensors)


# ---------------------------------------------------------------------------
# Registration -- one arm, not wired
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False, (f"the complex fused ADE chain cannot fill slot {slot!r}",))
    return complex_fused_ade_chain_coverage(
        context.fields, context.pml, context.residency,
        context.extra.get("complex_probe"))


def _arm_plan(context: Any, slot: str) -> Optional[MetalComplexFusedAdeChainPlan]:
    if slot != SLOT:
        return None
    return plan_metal_complex_fused_ade_chain(
        context.fields, context.pml, context.residency,
        context.contract_variants, context.extra.get("complex_probe"))


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    return (arms.register(
        family=FAMILY,
        slot=SLOT,
        label="fused complex no-PML stored E -> ADE",
        coverage=_arm_coverage,
        plan=_arm_plan,
        prefix="fused complex no-PML stored E -> ADE: ",
        noun="complex fused E->P chain",
        wired=False,
        replaces=REPLACES,
    ),)


ARMS = register_arms()
