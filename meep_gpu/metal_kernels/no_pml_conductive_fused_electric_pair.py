"""The conductive no-PML ``step_D`` welded into stored-E ``update_E``, deposit carried.

THE FIRST METAL PRODUCT ON THE SECOND DEPOSIT REPAIR. Every other fused Metal pair
that carries a deposit inverts the SPLIT-FIELD constitutive recurrence when its seam
holds a source; this configuration runs ``update_E``'s PLAIN overwrite instead
(``stepping.py:1019-1022``) — no ``f_w``, no coefficients, no previous value — so
:data:`meep_gpu.deposit_repair.PLAIN_PATH` inverts it and this module declares that
path rather than the default. Its lossless sibling
:mod:`.no_pml_fused_electric_pair` is the same construction over the plain curl arm.

THE CELL, from ``results/fusion_matrix_metal_2026-08-31_plainrepair``::

    D->E   conductive no-PML curl x no-PML stored E    2 seam-instances
           examples:absorber-1d.py, tests:TestAbsorber.test_absorber
           verdict: FITS -- 21..24 pointers against a 30 ceiling; NOT BUILT

Both rows are the same configuration twice: an ``mp.Absorber`` (a graded
conductivity, not a PML), ``meep.materials.Al`` (5 poles per component), METALLIC on
z, and ONE non-integrated ``mp.Ex`` source injected INSIDE this seam.

AND IT NOW SERVES BOTH, where it shipped serving 0 of 2. The blocker was never the
deposit — that is what the repair carries — but a SECOND pass in the same seam:
``FdtdDriver._inject_electric_through_conductivity`` used to rescale the WHOLE
target D volume for a NON-INTEGRATED source on a conductive run, canonicalising
every ``-0.0`` at cells no deposit closure can name, and
:func:`_scaled_conductive_injection_reasons` refused that configuration BY NAME on
a measured 1231-word step-1 divergence. THE DRIVER'S PASS IS NOW SPARSE — it
replays the rescale only at the deposit cells the sources publish, in the retired
passes' exact operand order, leaving every other word untouched (and matching
stock MEEP, step.cpp:294-317) — and the lift was taken ON A MEASUREMENT, not on
the edit: the same signed-zero protocol re-run with the sparse driver reads
byte-identical at every step while the retired whole-volume passes replayed as a
control still diverge (the ``lifted_refusal`` leg). What the clause still refuses
is the driver's own fallback: a scaled source publishing NO deposit table, which
no in-tree electric source class is. Both corpus rows
(``mp.Source(GaussianSource, component=mp.Ex)``, non-integrated, publishing
tables like every in-tree source) are therefore admitted and bracketed.

===========================================================================
THE SEAM, PASS BY PASS
===========================================================================

The driver runs four passes between ``step_D`` and ``update_E``
(``driver.py:3292-3306``):

* **the electric injection — CARRIED, through the deposit repair.**
  ``launch._install_fused_pair`` puts a
  :class:`~meep_gpu.deposit_repair.LeadingRepairPlan` in ``step_D`` and a
  :class:`~meep_gpu.deposit_repair.TrailingRepairPlan` in ``update_E``, and — new
  with this family — hands it this plan's own :attr:`repair_paths`, so the bracket
  saves and restores through the PLAIN repair. On this branch the save is the CELL
  SET alone (a pure overwrite has no previous value to capture) and the apply is one
  recomputation per component at the deposit cells. IGNORANCE IS STILL A REFUSAL: an
  undeclared source list is refused, never read as an empty seam;
* **the scaled conductive injection — CARRIED**, now that the driver replays the
  condinv rescale sparsely at the published deposit cells (identity everywhere
  else); only a table-less source still hits the whole-volume fallback and is
  refused by the clause above;
* **``zero_metal_D`` — CARRIED INLINE**, on the register, because two of this cell's
  two rows are METALLIC on z. The rows are the D OFF-DIAGONAL
  (:data:`.fused_dispersive_pair._ZERO_METAL_ROWS`), imported rather than re-derived;
* **``fill_symmetry_bc_D`` / ``fill_folded_far_ghosts_D`` — INERT**: both halves
  refuse a mirror plane already, restated here by name.

===========================================================================
EVERYTHING IS TRANSCRIBED, and from where
===========================================================================

* the curl half — :data:`.no_pml_conductive._CONDUCTIVE_PLAIN_CURL_TEMPLATE`'s own
  decode, ghost, load, curl, mask and per-target conductive tail text
  (``no_pml_conductive.py:59-176``), specialised to ONE target per dispatch the way
  :mod:`.fused_dispersive_pair` specialises the PML curl. Every emitted line is
  asserted to be a line of the certified all-component emitter's own output;
* the wall — :func:`zero_metal_mask` over the imported D off-diagonal table;
* the constitutive half — :func:`.no_pml_stored_e.stored_e_source`'s own bounded,
  left-associated subtraction chain and ``source * inv_e`` store
  (``no_pml_stored_e.py:76-88``).

WHAT MOVES IS ONE LINE, declared in :data:`LIFT_EDITS` as data: the certified E body
opens ``float source = d_in[idx];`` — a reload of the displacement the curl just
stored — and fused that reload becomes the register the conductive tail already
names (``value{axis}``). The D store is KEPT: the next timestep's curl and
``update_P`` read it.

WHY PER COMPONENT AND NOT ONE LAUNCH, measured against the ceiling: this cell's own
rows carry FIVE poles per component, so an all-three-component signature needs
3 D + 3 B + 3 E + 3 inv_eps + 3 condfac + 3 condinv + 15 poles = 33 pointers — over
:data:`.device.MAX_BUFFER_BINDINGS` even with every scalar packed. Per component the
count is :data:`BINDINGS_PER_COMPONENT` = 21 with all eight pole slots declared and
the five scalars bound separately, the same resolution
:mod:`.fused_dispersive_pair` records for the same pole-budget reason. One dispatch
per component IS the fusion: the stepped displacement never leaves a register.

NOT WIRED, like every Metal fused pair: it spans three passes and ``plan_step``
assigns at most one arm per slot, so it registers ``wired=False`` and is composed
only by the absorb table (``launch.FUSED_PAIR_ARMS``). MEASURED through ``plan_step``
with ``fuse=False`` under ``MEEP_GPU_SUBNORMAL_POLICY=flush`` on
``metal_composition_matrix.conductive(real_stored_e_no_pml(with_polarization=True))``:
``step_D`` comes back ``'conductive no-PML curl'`` and ``update_E``
``'no-PML stored E'`` — the two labels this family's row declares. Dispatch is not
wired either: ``meep_gpu.fastpath.plan_fast_path`` still returns ``None`` on every
branch.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from ..triton_kernels.no_pml import SUB_STEPS as _NO_PML_SUB_STEPS
from ..triton_kernels.no_pml_conductive import conductive_no_pml_targets
from . import shaders, templates
from .ade_update_p import _physical_name
from .coverage import zero_metal_axes
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .fused_dispersive_pair import _TARGETS, _ZERO_METAL_ROWS
from .no_pml_conductive import (
    _tail as certified_conductive_tail,
    conductive_plain_curl_source,
    metal_conductive_plain_curl_coverage,
)
from .no_pml_stored_e import (
    E_TERMS,
    MAX_POLES,
    _poles,
    metal_stored_e_coverage,
    stored_e_source,
)
from .. import deposit_repair as _deposit_repair

FAMILY = "no_pml_conductive_fused_electric_pair"

#: Does this product bracket its fused launch with the deposit repair? TRUE, and on
#: this family the flag IS the product: both corpus rows of its cell declare an
#: ELECTRIC source inside the seam, so the same module with this at False would
#: compile, gate green on every arithmetic leg, and serve nothing. The wiring this
#: claims is ``launch._install_fused_pair``, which reads :attr:`repair_paths` off the
#: built plan — flag, path and wiring moved in the same edit.
CARRIES_DEPOSIT_REPAIR = True

#: WHICH repair the two slots install: the PLAIN one, alone. Both halves refuse an
#: active absorber layer, so the recurrence this seam inverts is always ``update_E``'s
#: plain overwrite (``stepping.py:1019-1022``) and never the split-field one.
#: ``deposit_repair.repairable`` refuses by name any configuration whose recurrence is
#: not among the declared paths, so a wrong declaration is a refusal rather than a
#: silent mis-repair.
REPAIR_PATHS: Tuple[str, ...] = (_deposit_repair.PLAIN_PATH,)

#: The slot the arm is registered on: the product spans three, it holds a row on the
#: first, so a refusal is named on the slot the fusion starts at.
SLOT = "step_D"

#: The driver passes one run of this plan performs, in driver order
#: (``driver.py:3292-3306``). The two fills are absent because they are INERT on
#: every grid this family admits, not because they were forgotten.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The curl tables, READ from the no-absorber arm rather than spelled. The two
#: ``SUB_STEPS`` tables in this package disagree about ``step_D``'s sources:
#: ``launch.SUB_STEPS`` says ``('Hx','Hy','Hz')``, right under an absorber where H is
#: stored; the no-absorber arms say ``('Bx','By','Bz')`` because
#: ``Fields.enable_field_storage`` deliberately does not allocate H at all and
#: ``get_H`` returns the B array itself (``no_pml.py:178-181``).
_CURL_SPEC = _NO_PML_SUB_STEPS[SLOT]
CURL_TARGETS: Tuple[str, ...] = tuple(_CURL_SPEC["targets"])
CURL_SOURCES: Tuple[str, ...] = tuple(_CURL_SPEC["sources"])

#: The per-component signature, as data the gate compiles rather than arithmetic in
#: prose: 16 pointers (the D target, three B sources, the E target, the inverse
#: epsilon, condfac, condinv, and eight pole slots) plus five separate scalars.
POINTERS_PER_COMPONENT = 16
BINDINGS_PER_COMPONENT = POINTERS_PER_COMPONENT + 5

#: Why one launch is NOT the shape here: the all-three-component signature at this
#: cell's own five poles per component. Spelled as data so the gate can refute the
#: padded form from the same number this docstring argues from.
ALL_COMPONENT_POINTERS_AT_FIVE_POLES = 33

#: The one edit the lift applies to the certified constitutive body, as DATA: the
#: certified line, what replaces it (``{axis}`` formatted per target), and why. The
#: emitter asserts the certified line is present before editing and absent after.
LIFT_EDITS: Tuple[Tuple[str, str, str], ...] = (
    ("    float source = d_in[idx];",
     "    float source = value{axis};",
     "THE SEAM: the register the conductive tail already names, not a reload of "
     "the displacement the curl just stored"),
)

__all__ = [
    "ALL_COMPONENT_POINTERS_AT_FIVE_POLES", "BINDINGS_PER_COMPONENT",
    "CARRIES_DEPOSIT_REPAIR", "CURL_SOURCES", "CURL_TARGETS", "FAMILY",
    "LIFT_EDITS", "POINTERS_PER_COMPONENT", "REPAIR_PATHS", "REPLACES", "SLOT",
    "MetalNoPmlConductiveFusedElectricPairPlan",
    "compile_no_pml_conductive_fused_electric_pair",
    "no_pml_conductive_fused_electric_pair_coverage",
    "no_pml_conductive_fused_electric_pair_source",
    "plan_metal_no_pml_conductive_fused_electric_pair",
    "refuted_over_ceiling_source", "zero_metal_mask",
]


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

kernel void no_pml_conductive_fused_electric_pair_component(
    device float*       f__T__  [[buffer(0)]],
    device const float* g0      [[buffer(1)]],
    device const float* g1      [[buffer(2)]],
    device const float* g2      [[buffer(3)]],
    device float*       e_out   [[buffer(4)]],
    device const float* inv_e   [[buffer(5)]],
    device const float* cf__T__ [[buffer(6)]],
    device const float* ci__T__ [[buffer(7)]],
    device const float* p0      [[buffer(8)]],
    device const float* p1      [[buffer(9)]],
    device const float* p2      [[buffer(10)]],
    device const float* p3      [[buffer(11)]],
    device const float* p4      [[buffer(12)]],
    device const float* p5      [[buffer(13)]],
    device const float* p6      [[buffer(14)]],
    device const float* p7      [[buffer(15)]],
    constant uint&      nx      [[buffer(16)]],
    constant uint&      ny      [[buffer(17)]],
    constant uint&      nz      [[buffer(18)]],
    constant uint&      n_elem  [[buffer(19)]],
    constant float&     dtdx    [[buffer(20)]],
    uint idx [[thread_position_in_grid]])
{
__BODY__
}
"""


def zero_metal_mask(axis: int, zero_metal: Sequence[bool]) -> str:
    """``stepping.zero_metal_D`` for ONE target, on the register, before the store.

    The rows are :data:`.fused_dispersive_pair._ZERO_METAL_ROWS` — the D
    OFF-DIAGONAL (Dy/Dz on an x wall, Dx/Dz on y, Dx/Dy on z), imported rather than
    re-derived; the register spelling is the certified conductive tail's own
    ``value{axis}``.
    """
    lines = [f"    value{axis} = {flag} ? 0.0f : value{axis};"
             for row_target, wall_axis, flag in _ZERO_METAL_ROWS
             if row_target == axis and bool(zero_metal[wall_axis])]
    return "\n".join(lines) or "    // no walled axis clears this target"


def _certified_curl_reference(codes: Sequence[int], conductive: Sequence[bool]) -> str:
    """The certified ALL-COMPONENT conductive curl source, the anchor authority."""
    return conductive_plain_curl_source(codes, True, False, conductive)


def _per_component_curl(codes: Sequence[int], axis: int, conductive: bool,
                        zero_metal: Sequence[bool]) -> List[str]:
    """The certified curl text, restricted to ONE target, every line anchored.

    The loads, the curl expression, the decode block, the ghost rules and the
    ownership-mask rows are the certified emitter's own lines: each one is asserted
    to appear verbatim in :func:`.no_pml_conductive.conductive_plain_curl_source`'s
    output for the same specialisation, so a drift in the certified template reaches
    this family as an AssertionError rather than as a silent re-derivation.
    """
    spec = _TARGETS[axis]
    flags = tuple(conductive if index == axis else False for index in range(3))
    certified = _certified_curl_reference(codes, flags)
    lines = [
        templates.GUARD,
        "",
        "    int nxi = int(nx), nyi = int(ny), nzi = int(nz);",
        "    int nyz = nyi * nzi;",
        "    int ii = int(idx);",
        "    int k = ii % nzi;",
        "    int plane = ii / nzi;",
        "    int j = plane % nyi;",
        "    int i = plane / nyi;",
        "",
        "    int si = i - 1, sj = j - 1, sk = k - 1;",
        "    bool vx = true, vy = true, vz = true;",
        templates.ghost("x", codes[0], True),
        templates.ghost("y", codes[1], True),
        templates.ghost("z", codes[2], True),
        "",
        "    int ox = si * nyz + j * nzi + k;",
        "    int oy = i * nyz + sj * nzi + k;",
        "    int oz = i * nyz + j * nzi + sk;",
        "",
        *spec["loads"],
        "",
        spec["curl"],
        "",
        "    bool at_x = (i == 0), at_y = (j == 0), at_z = (k == 0);",
    ]
    mask = [line for line in templates.ownership_mask(codes, True).splitlines()
            if line.strip().startswith(f"curl{axis} ")]
    lines.extend(mask or [f"    // no metallic axis masks curl{axis}"])
    lines.append("")
    tail = certified_conductive_tail(axis, conductive)
    if conductive:
        # The certified tail already names the register; the wall clear goes
        # between its condinv multiply and its store, on the register.
        head, store = tail.rsplit("\n", 1)
        assert store == f"    f{axis}[ii] = value{axis};", store
        lines.extend([head, zero_metal_mask(axis, zero_metal), store])
    else:
        # The certified lossless tail is one line, `f{a}[ii] = f{a}[ii] - curl{a};`.
        # The fusion needs the stepped value in a register, so the line is split in
        # two -- the same store-then-name split the complex conductive pair records
        # as byte-neutral by construction, and the gate's equivalence leg measures.
        assert tail == f"    f{axis}[ii] = f{axis}[ii] - curl{axis};", tail
        lines.extend([
            f"    float value{axis} = f{axis}[ii] - curl{axis};",
            zero_metal_mask(axis, zero_metal),
            f"    f{axis}[ii] = value{axis};",
        ])
    # EVERY CURL LINE IS A LINE OF THE CERTIFIED EMITTER'S OWN OUTPUT. The decode,
    # ghost, offset, load, curl, mask and (conductive) tail text may not drift from
    # the certified module without failing here.
    for line in lines:
        text = line.strip()
        if (not text or text.startswith("//") or "value" in line
                or line == templates.GUARD):
            continue
        assert line in certified, (
            f"lifted curl line is not the certified conductive emitter's own: "
            f"{line!r}")
    return lines


def _constitutive_chain(axis: int, pole_count: int) -> List[str]:
    """The certified stored-E body with :data:`LIFT_EDITS` applied, every line anchored."""
    certified = stored_e_source(pole_count)
    lines: List[str] = []
    for old, new, why in LIFT_EDITS:
        assert old in certified, (
            f"the certified stored-E body no longer carries {old!r}; the seam has "
            f"no anchor")
        lines.append(f"    // {why}.")
        lines.append(new.format(axis=axis))
    for index in range(int(pole_count)):
        line = f"    source = source - p{index}[idx];"
        assert line in certified, line
        lines.append(line)
    store = "    e_out[idx] = source * inv_e[idx];"
    assert store in certified
    lines.append(store)
    return lines


def no_pml_conductive_fused_electric_pair_source(
        codes: Sequence[int], axis: int, pole_count: int, conductive: bool,
        zero_metal: Sequence[bool], contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source: (boundaries, target, poles, sigma flag, walls)."""
    codes = tuple(int(code) for code in codes)
    if len(codes) != 3:
        raise ValueError(f"codes must be a per-axis triple, got {codes!r}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis!r}")
    if not 0 <= int(pole_count) <= MAX_POLES:
        raise ValueError(f"pole_count must be in [0, {MAX_POLES}], got {pole_count!r}")
    zero_metal = tuple(bool(value) for value in zero_metal)
    if len(zero_metal) != 3:
        raise ValueError(f"zero_metal must be a per-axis triple, got {zero_metal!r}")
    body = _per_component_curl(codes, axis, bool(conductive), zero_metal)
    body.append("")
    body.append("    // --- stored-E update_E (stepping.py:981-984, :993) ---------")  # stepping.py live lines for the frozen device-text citation(s) in this string: 981-984->1010-1013, 993->1022
    body.extend(_constitutive_chain(axis, int(pole_count)))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__T__": str(axis),
        "__BODY__": "\n".join(body),
    })


def compile_no_pml_conductive_fused_electric_pair(
        codes: Sequence[int], axis: int, pole_count: int, conductive: bool,
        zero_metal: Sequence[bool], contract: str = shaders.CONTRACT_OFF) -> Any:
    return compile_source(no_pml_conductive_fused_electric_pair_source(
        codes, axis, pole_count, conductive, zero_metal, contract)
    ).no_pml_conductive_fused_electric_pair_component


def refuted_over_ceiling_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The shipped per-component signature padded one binding PAST the ceiling.

    Not shipped and not launchable: the gate compiles it and requires the failure,
    which is what turns "21 bindings, ten under the ceiling" into a measurement of
    where the ceiling actually is. The pad buffers are read so dead-code elimination
    cannot decide the answer.
    """
    source = no_pml_conductive_fused_electric_pair_source(
        (0, 0, 0), 0, MAX_POLES, True, (False, False, False), contract)
    pad = "".join(
        f"    device const float* pad{slot:<4}[[buffer({slot})]],\n"
        for slot in range(BINDINGS_PER_COMPONENT, MAX_BUFFER_BINDINGS + 1))
    anchor = "    uint idx [[thread_position_in_grid]])"
    assert anchor in source
    source = source.replace(anchor, pad + anchor)
    touch = " + ".join(
        f"pad{slot}[0]"
        for slot in range(BINDINGS_PER_COMPONENT, MAX_BUFFER_BINDINGS + 1))
    marker = "    e_out[idx] = source * inv_e[idx];"
    assert marker in source
    return source.replace(
        marker, f"    e_out[idx] = source * inv_e[idx] + ({touch}) * 0.0f;")


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

def _scaled_conductive_injection_reasons(fields: Any, sources: Any) -> Tuple[str, ...]:
    """Why a scaled electric source with NO deposit table is still refused BY NAME.

    THE CLAUSE THAT USED TO PRICE THIS CELL AT 0 OF 2, LIFTED ON A MEASUREMENT.
    ``FdtdDriver._inject_electric_through_conductivity`` injects an INTEGRATED
    source point-wise; for a SCALED (non-integrated) one it used to rescale the
    WHOLE target volume by difference (``array -= before; array *= condinv;
    array += before``), which canonicalises every ``-0.0`` in the component at
    cells no deposit closure can name — measured 2026-08-31 as 2417 Dz words moved
    and 1231 Ez words divergent on step 1 of a signed-zero seed, identical from
    step 2 on (the wash-out that made per-step comparison load-bearing).

    THE DRIVER NO LONGER DOES THAT. It snapshots the target at the deposit cells
    the sources publish (``deposit_repair._deposit_index`` reads the same tables),
    injects, and replays the rescale per deposit cell in the whole-volume passes'
    exact operand order — exact at every deposit cell, UNTOUCHED everywhere else,
    and closer to stock MEEP (step.cpp:294-317 scales only the injected current).
    RE-MEASURED with the sparse replay in place, same protocol, same
    +-0-lattice seed class: the bracketed fused walk is byte-identical to the
    array path at EVERY step, and the old whole-volume passes replayed as a
    control still diverge — see the ``lifted_refusal`` leg of
    ``gate_metal_no_pml_fused_electric_pairs.py``.

    WHAT SURVIVES is the driver's own fallback: a scaled source that publishes NO
    deposit table at all (no ``_point_ix`` attribute) still takes the whole-volume
    difference passes, because an unnameable deposit must still be scaled. No
    in-tree electric source class is one — every source publishes
    ``_point_ix/_point_iy/_point_iz`` at setup — so on this corpus the clause
    refuses nothing; it stays because a duck-typed source without the table would
    otherwise reintroduce the divergence silently.

    FAIL CLOSED: a source whose ``is_integrated`` cannot be read, and a ``Fields``
    that cannot answer ``condinv_for``, are both refusals.
    """
    if sources is None:
        return ()  # the undeclared-source refusal is the seam clause's, not this one
    if not bool(getattr(fields, "has_conductivity", False)):
        # The driver's own guard, restated: `if electric and fields.has_conductivity`
        # (driver.py:3305). Without a conductivity every electric source takes the
        # plain per-source inject loop (:3308).
        return ()
    reader = getattr(fields, "condinv_for", None)
    if not callable(reader):
        return ("fields does not expose condinv_for, so whether "
                "FdtdDriver._inject_electric_through_conductivity would rescale "
                "inside this seam cannot be established",)
    reasons: List[str] = []
    for index, source in _deposit_repair._in_seam_indexed(sources, "D"):
        try:
            integrated = bool(source.is_integrated)
        except Exception as error:  # noqa: BLE001 - unreadable is a refusal
            reasons.append(
                f"source {index} ({type(source).__name__}) could not answer "
                f"is_integrated ({error!r}), so whether the driver would rescale "
                f"inside this seam cannot be established")
            continue
        if integrated:
            continue
        component = getattr(source, "component", None)
        try:
            target = "D" + str(component)[1]
            scaled = reader(target) is not None
        except Exception as error:  # noqa: BLE001 - unreadable is a refusal
            reasons.append(
                f"source {index} drives {component!r}, whose conductivity could "
                f"not be read ({error!r})")
            continue
        if not scaled:
            continue
        # THE DRIVER'S OWN PARTITION: a published deposit table takes the sparse
        # per-cell replay (identity away from the deposit); only the table-less
        # source falls back to the whole-volume passes.
        if not hasattr(source, "_point_ix"):
            reasons.append(
                f"source {index} ({type(source).__name__}) is electric, NOT "
                f"integrated, on a run whose {target} carries a conductivity, and "
                f"publishes NO deposit table (no _point_ix): the driver's "
                f"fallback rescales the WHOLE {target} volume inside this seam "
                f"(_inject_electric_through_conductivity's dense branch), which "
                f"canonicalises every -0.0 at cells the deposit closure cannot "
                f"name; the fused launch has already computed E from the "
                f"pre-rescale values there")
    return tuple(reasons)


def no_pml_conductive_fused_electric_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Coverage:
    """May one dispatch per component span conductive ``step_D`` -> wall -> ``update_E``?

    A conjunction of the two halves' OWN certified predicates plus the seam clauses.
    Nothing is weakened: a configuration either half refuses is refused here with
    that half's reasons, prefixed so a reader can tell which side said it — the
    construction every Metal fused pair uses.
    """
    reasons: List[str] = []

    curl = metal_conductive_plain_curl_coverage(fields, pml, SLOT, residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = metal_stored_e_coverage(fields, pml, residency)
    if not electric.covered:
        reasons.extend(f"stored-E half: {reason}" for reason in electric.reasons)

    # THE SOURCE SEAM, CARRIED. `pml` IS FORWARDED, unlike the predicates written
    # before the second repair: deposit_repair.repairable refuses a plain-only
    # declaration with no layer BY NAME, since which recurrence ran cannot be
    # established without one.
    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an empty "
            "electric source seam from Fields"
        ),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3294-3299), and on "
            f"a conductive row through _inject_electric_through_conductivity "
            f"(driver.py:3305)"),
        carries_repair=CARRIES_DEPOSIT_REPAIR,
        pml=pml,
        repair_paths=REPAIR_PATHS))

    # THE CONDUCTIVE WHOLE-VOLUME RESCALE — the clause that prices this cell.
    reasons.extend(_scaled_conductive_injection_reasons(fields, sources))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False,
                        tuple(dict.fromkeys(reasons + ["fields carries no grid"])))

    # THE TWO FILLS, restated by name: both halves refuse a mirror plane already,
    # but a reader should not have to chase a shared grid clause to learn which two
    # driver passes that refusal empties on THIS seam.
    mirrored = getattr(grid, "is_mirrored", None)
    for axis in range(3):
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam "
                f"(driver.py:3300-3302) and neither is carried by this family")

    # zero_metal_D is CARRIED, so the grid must be able to answer which axes are
    # walled; an unreadable answer would compile the wall away on a run that needs it.
    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried inline")

    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Entry:
    """One E component's whole fused binding set, resolved once at plan time."""

    component: str
    displacement_name: str
    axis: int
    pole_count: int
    conductive: bool
    displacement: Any
    target: Any
    inverse_epsilon: Any
    condfac: Any
    condinv: Any
    states: Tuple[Any, ...]


class MetalNoPmlConductiveFusedElectricPairPlan:
    """Three dispatches that perform three driver passes, D never leaving a register."""

    performs_device_work = True
    replaces_sub_steps = REPLACES
    launches_per_run = len(E_TERMS)

    #: The repairs this product's two slots install, read by
    #: ``launch._install_fused_pair`` so the bracket it builds is the bracket the
    #: predicate declared. The class attribute mirrors the module constant; the two
    #: may never diverge and the host suite pins them equal.
    repair_paths = REPAIR_PATHS

    __slots__ = ("entries", "residency", "volumes", "codes", "zero_metal", "shape",
                 "dtdx", "_functions", "_tensors", "_magnetic", "launches", "runs")

    def __init__(self, entries: Sequence[_Entry], residency: Residency,
                 volumes: Sequence[str], codes: Sequence[int],
                 zero_metal: Sequence[bool], shape: Sequence[int], dtdx: float,
                 magnetic: Sequence[Any],
                 functions: Mapping[Tuple[str, int], Any],
                 tensors: Mapping[int, Any]) -> None:
        self.entries = tuple(entries)
        self.residency = residency
        self.volumes = tuple(dict.fromkeys(volumes))
        self.codes = tuple(int(code) for code in codes)
        self.zero_metal = tuple(bool(value) for value in zero_metal)
        self.shape = tuple(int(n) for n in shape)
        # float(): the array path multiplies a float32 volume by a Python float and
        # Metal binds a Python float into `constant float&` the same way.
        self.dtdx = float(dtdx)
        self._magnetic = tuple(magnetic)
        self._functions = dict(functions)
        self._tensors = dict(tensors)
        self.launches = 0
        self.runs = 0

    @property
    def variants(self) -> Tuple[str, ...]:
        return tuple(sorted({key[0] for key in self._functions}))

    def run(self, contract: Optional[str] = None) -> None:
        """One complete D->E seam, in place, against the mirrors.

        THE POLE POINTERS ARE RESOLVED AT LAUNCH, NOT AT PLAN TIME: ``update_P``
        rotates ``P``/``P_prev``/scratch after every recurrence, so the physical
        array holding a component's current P changes between steps while the
        mirror registry does not — the same discipline
        :class:`.no_pml_stored_e.MetalStoredEPlan` uses.
        """
        mode = shaders.CONTRACT_OFF if contract is None else contract
        for entry in self.entries:
            key = (mode, entry.axis)
            function = self._functions.get(key)
            if function is None:
                raise KeyError(
                    f"this plan holds no {mode!r} variant (it was built with "
                    f"{self.variants})")
            poles = tuple(state.P[entry.component] for state in entry.states)
            if len(poles) != entry.pole_count:
                raise RuntimeError(
                    f"{entry.component} resolved {len(poles)} poles, not the "
                    f"compiled count {entry.pole_count}")
            # THE UNUSED POLE SLOTS BIND THE INVERSE EPSILON, NOT THE DISPLACEMENT:
            # this kernel WRITES the D volume, and padding with it would alias a
            # written buffer into a `device const float*` in the same dispatch —
            # the hazard `fused_dispersive_pair.run` names. The dead condfac /
            # condinv slots of a lossless component bind it for the same reason.
            slots = poles + (entry.inverse_epsilon,) * (MAX_POLES - len(poles))
            function(
                self._tensors[id(entry.displacement)],
                *(self._tensors[id(value)] for value in self._magnetic),
                self._tensors[id(entry.target)],
                self._tensors[id(entry.inverse_epsilon)],
                self._tensors[id(entry.condfac)],
                self._tensors[id(entry.condinv)],
                *(self._tensors[id(value)] for value in slots),
                self.shape[0], self.shape[1], self.shape[2],
                int(entry.displacement.size), self.dtdx,
            )
            self.launches += 1
        self.runs += 1

    def __repr__(self) -> str:
        return (f"MetalNoPmlConductiveFusedElectricPairPlan(shape={self.shape}, "
                f"poles={tuple(e.pole_count for e in self.entries)}, "
                f"cond={tuple(e.conductive for e in self.entries)}, "
                f"bc={self.codes}, zero_metal={self.zero_metal})")


def plan_metal_no_pml_conductive_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[Tuple[str, int], Any]] = None,
        ) -> Optional[MetalNoPmlConductiveFusedElectricPairPlan]:
    """Build the fused plan, or ``None`` when the seam is refused.

    ``None`` is the only refusal: a configuration this kernel does not carry must
    fall back to the array path, never raise into a caller that would otherwise
    have stepped correctly. ``functions`` is the mutation seam — dropping it is not
    a slowdown, it is a DISARMING: every mutation leg would then launch the shipped
    kernel and report the defect as uncaught.
    """
    if not no_pml_conductive_fused_electric_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    # `None` for the layer: this family's own predicate has already refused an
    # active absorber, and `_boundary_kinds` consulted WITH one would report the
    # absorber's softening rather than the grid's declaration.
    codes = tuple(1 if kind == "metallic" else 0
                  for kind in _boundary_kinds(grid, None))
    walls = zero_metal_axes(grid)
    flags = conductive_no_pml_targets(fields, SLOT)
    order = _poles(fields)

    tensors: Dict[int, Any] = {}
    volumes: List[str] = []
    entries: List[_Entry] = []
    required: set = set()

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        tensors[id(host)] = residency.mirror(name, host, constant=constant)
        volumes.append(name)
        return host

    magnetic = tuple(bind(name, getattr(fields, name)) for name in CURL_SOURCES)
    for (component, displacement_name), axis in zip(E_TERMS, range(3)):
        displacement = bind(displacement_name, getattr(fields, displacement_name))
        target = bind(component, getattr(fields, component))
        inverse = bind(f"inv_eps_{component}:no_pml",
                       fields.inverse_epsilon_for(component), constant=True)
        if flags[axis]:
            condfac = bind(f"condfac_{displacement_name}:no_pml_conductive",
                           fields.condfac_for(displacement_name), constant=True)
            condinv = bind(f"condinv_{displacement_name}:no_pml_conductive",
                           fields.condinv_for(displacement_name), constant=True)
        else:
            condfac = condinv = inverse
        states = order[component]
        entries.append(_Entry(component, displacement_name, axis, len(states),
                              bool(flags[axis]), displacement, target, inverse,
                              condfac, condinv, states))
        required.update((mode, axis) for mode in contract_variants)

    for state_index, state in enumerate(tuple(getattr(fields, "polarizations", ())
                                              or ())):
        for component in tuple(state.driven()):
            for label, host in ((f"P:{component}", state.P[component]),
                                (f"P_prev:{component}", state.P_prev[component])):
                bind(_physical_name(state_index, label, host), host)
        scratch = state._scratch
        bind(_physical_name(state_index, "scratch", scratch), scratch)

    selected = dict(functions or {})
    by_axis = {entry.axis: entry for entry in entries}
    for key in required:
        if key not in selected:
            entry = by_axis[key[1]]
            selected[key] = compile_no_pml_conductive_fused_electric_pair(
                codes, key[1], entry.pole_count, entry.conductive, walls, key[0])

    return MetalNoPmlConductiveFusedElectricPairPlan(
        entries, residency, volumes, codes, walls, grid.shape,
        grid.dt / grid.dx, magnetic, selected, tensors)


# ---------------------------------------------------------------------------
# Registration — NOT WIRED
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(False,
                        (f"conductive no-PML fused electric pair cannot fill {slot}",))
    return no_pml_conductive_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str
              ) -> Optional[MetalNoPmlConductiveFusedElectricPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_no_pml_conductive_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False`` — enumerable, never selectable."""
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT, "conductive no-PML fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="conductive no-PML fused electric D/E pair: ",
                          noun="conductive no-PML fused D-curl/stored-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
