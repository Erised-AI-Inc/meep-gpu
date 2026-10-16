"""Complex/Bloch conductive split-field PML curls for PyTorch/MPS.

This is the complex counterpart of :mod:`.conductive_pml`.  Its arithmetic is
the array path's conductive branch inside ``stepping._apply_curl``
(``_apply_conductive_pml_update``), each product one per part: a target
first advances the conductivity recurrence, then one or both split-field PML
recurrences consume that result.  The two histories are independent: ``f_cond``
is read and written only on the cells whose chunk carries the target's dsig
layer, and ``fu`` only on those whose chunk carries its dsigu layer.

WHICH OF MEEP'S FOUR CASES A CELL TAKES IS ITS CHUNK'S, read from the bound chunk
record at the cell's own stored index -- the same two memberships the array path
passes (``dsig_active`` = ``RegionPlan.axis_mask`` on the dsig axis, ``dsigu_active``
= the curl split), on the target's own Yee lattice. A target without a
conductivity takes the ordinary complex forms (:mod:`.complex_fields`). A lossy
target takes MEEP's four conductive forms (step_generic.cpp:84-210), every product a
real coefficient times one part, every operation rounded to float32 in this order,
with ``X`` the curl, ``cf``/``ci`` the conductivity coefficients, ``kms``/``sinv``
the dsig axis's and ``kms_u``/``sinv_u`` the dsigu axis's:

* both layers: ``fcnd' = ((fcnd*cf) - X)*ci``; ``fu' = ((fu*kms) + (fcnd' - fcnd))*sinv``;
  ``f' = (((f*kms_u) + fu') - fu)*sinv_u``;
* dsig only: ``fcnd'`` as above, then ``f' = ((f*kms) + (fcnd' - fcnd))*sinv``;
* dsigu only: ``fu' = ((fu*cf) - X)*ci``, then ``f'`` as in the first form;
* neither: ``f' = ((f*cf) - X)*ci``.

The increment ``(fcnd' - fcnd)`` is formed FIRST and then added, MEEP's grouping
(step_generic.cpp:167, :209), as :mod:`.conductive_pml` forms it on real storage.

PER-CHUNK CONDUCTIVITY PRESENCE IS NOT IN THE RECORD. A chunk whose conductivity
samples are all zero keeps no conductivity array and steps the lossless forms
(structure.cpp:883-900, ``ChunkLayout.conductive_chunks``), so this family serves a
curl sub-step only where every chunk keeps the conductivity of each lossy target and
refuses the rest by name to the array path
(:func:`.conductive_pml.conductive_presence_reasons`, read on the composer's
``chunk_layout``).

The ordinary complex PML curl needs 24 bindings.  Adding three ``f_cond``
histories and six conductivity volumes would make 33 if its three phase values
remained separate constant arguments, exceeding Metal's measured 31-binding
limit.  This family instead binds a three-``float2`` phase table at one buffer
position and the chunk record last: 31 bindings, the ceiling exactly.  The
table's contents are exactly those produced by ``complex_fields.phase_arguments``
for the sub-step's side.

The direct MPS gate establishes the stencil, recurrence, phase-table ABI and
per-target histories.  The whole-step gate separately proves the two supported
B/H/D/E compositions.  Its constitutive companion deliberately masks only
conductivity before reusing the established complex H/E body: conductivity is
read by the curl recurrence, not by ``update_H`` or ``update_E``.  Dispersion,
nonlinearity, off-diagonal E rows, and every other complex-constitutive clause
remain live and refuse by name.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..triton_kernels.conductivity import conductive_targets
from ..triton_kernels.coverage import (
    CURL_TARGETS,
    Coverage,
    _boundary_kinds,
    _susceptibility_reasons,
    _volume_reasons,
)
from . import chunk_bounds as _chunk_bounds
from . import complex_fields as base
from . import shaders, templates
from .conductive_pml import conductive_presence_reasons
from .device import Residency, compile_source
from .plans import KernelPlan

FAMILY = "complex_conductive_pml"
SUB_STEPS = base.SUB_STEPS

__all__ = [
    "FAMILY",
    "SUB_STEPS",
    "MetalComplexConductivePmlCurlPlan",
    "complex_conductive_pml_curl_source",
    "compile_complex_conductive_pml_curl",
    "metal_complex_conductive_pml_constitutive_coverage",
    "metal_complex_conductive_pml_curl_coverage",
    "plan_metal_complex_conductive_pml_constitutive",
    "plan_metal_complex_conductive_pml_curl",
]


# The existing complex PML body is the exact source of the stencil, ghost,
# phase, grouping and mask.  Its unique tail is replaced below.  Keeping this
# contract checked at import prevents a future edit to the certified stencil
# from silently giving this family a stale near-copy.
_BASE_SIGNATURE = r"""    device float2*       f0      [[buffer(0)]],
    device float2*       f1      [[buffer(1)]],
    device float2*       f2      [[buffer(2)]],
    device float2*       u0      [[buffer(3)]],
    device float2*       u1      [[buffer(4)]],
    device float2*       u2      [[buffer(5)]],
    device const float2* g0      [[buffer(6)]],
    device const float2* g1      [[buffer(7)]],
    device const float2* g2      [[buffer(8)]],
    device const float*  kmx     [[buffer(9)]],
    device const float*  sinvx   [[buffer(10)]],
    device const float*  kmy     [[buffer(11)]],
    device const float*  sinvy   [[buffer(12)]],
    device const float*  kmz     [[buffer(13)]],
    device const float*  sinvz   [[buffer(14)]],
    constant uint&       nx      [[buffer(15)]],
    constant uint&       ny      [[buffer(16)]],
    constant uint&       nz      [[buffer(17)]],
    constant uint&       n_elem  [[buffer(18)]],
    constant float&      dtdx    [[buffer(19)]],
    constant float2&     px      [[buffer(20)]],
    constant float2&     py      [[buffer(21)]],
    constant float2&     pz      [[buffer(22)]],
    __RECORD_BINDING__,
    uint idx [[thread_position_in_grid]])"""

_CONDUCTIVE_SIGNATURE = r"""    device float2*       f0      [[buffer(0)]],
    device float2*       f1      [[buffer(1)]],
    device float2*       f2      [[buffer(2)]],
    device float2*       u0      [[buffer(3)]],
    device float2*       u1      [[buffer(4)]],
    device float2*       u2      [[buffer(5)]],
    device float2*       h0      [[buffer(6)]],
    device float2*       h1      [[buffer(7)]],
    device float2*       h2      [[buffer(8)]],
    device const float*  cf0     [[buffer(9)]],
    device const float*  cf1     [[buffer(10)]],
    device const float*  cf2     [[buffer(11)]],
    device const float*  ci0     [[buffer(12)]],
    device const float*  ci1     [[buffer(13)]],
    device const float*  ci2     [[buffer(14)]],
    device const float2* g0      [[buffer(15)]],
    device const float2* g1      [[buffer(16)]],
    device const float2* g2      [[buffer(17)]],
    device const float*  kmx     [[buffer(18)]],
    device const float*  sinvx   [[buffer(19)]],
    device const float*  kmy     [[buffer(20)]],
    device const float*  sinvy   [[buffer(21)]],
    device const float*  kmz     [[buffer(22)]],
    device const float*  sinvz   [[buffer(23)]],
    constant uint&       nx      [[buffer(24)]],
    constant uint&       ny      [[buffer(25)]],
    constant uint&       nz      [[buffer(26)]],
    constant uint&       n_elem  [[buffer(27)]],
    constant float&      dtdx    [[buffer(28)]],
    device const float2* phases  [[buffer(29)]],
    __RECORD_BINDING__,
    uint idx [[thread_position_in_grid]])"""

#: The chunk record's binding index: one past the phase table, the 31st binding.
RECORD_BUFFER = 30

_LOSSLESS_TAIL = r"""    float2 p0 = du0 ? u0[ii] : f0[ii];
    float2 n0 = c_mul_field_left(c_mul_field_left(p0, km_y) - curl0, si_y);
    float2 v0 = du0 ? c_mul_field_left((c_mul_field_left(f0[ii], km_z) + n0) - p0, si_z) : n0;

    float2 p1 = du1 ? u1[ii] : f1[ii];
    float2 n1 = c_mul_field_left(c_mul_field_left(p1, km_z) - curl1, si_z);
    float2 v1 = du1 ? c_mul_field_left((c_mul_field_left(f1[ii], km_x) + n1) - p1, si_x) : n1;

    float2 p2 = du2 ? u2[ii] : f2[ii];
    float2 n2 = c_mul_field_left(c_mul_field_left(p2, km_x) - curl2, si_x);
    float2 v2 = du2 ? c_mul_field_left((c_mul_field_left(f2[ii], km_y) + n2) - p2, si_y) : n2;

    if (du0) { u0[ii] = n0; }
    if (du1) { u1[ii] = n1; }
    if (du2) { u2[ii] = n2; }
    f0[ii] = v0; f1[ii] = v1; f2[ii] = v2;"""


def _tail(index: int, conductive: bool, km1: str, si1: str,
          km2: str, si2: str, dsig: str) -> str:
    """One target's curl form, specialized per target.

    ``km1``/``si1`` are the dsig axis's coefficients and ``km2``/``si2`` the dsigu
    axis's; ``dsig`` is the dsig axis's membership local and ``du{index}`` the
    dsigu one the base template already declares. A target without a conductivity
    takes the ordinary complex forms (``stepping._apply_curl_regions``); a conductive
    one the four cases of ``stepping._apply_conductive_pml_update``, each selected
    on its chunk's two memberships, with ``f_cond`` touched only on dsig cells and
    ``fu`` only on dsigu cells. The ``f_cond`` increment ``hs{index}`` is formed
    before either form that adds it.
    """
    if not conductive:
        return f"""    float2 p{index} = du{index} ? u{index}[ii] : f{index}[ii];
    float2 n{index} = c_mul_field_left(c_mul_field_left(p{index}, {km1}) - curl{index}, {si1});
    float2 v{index} = du{index} ? c_mul_field_left((c_mul_field_left(f{index}[ii], {km2}) + n{index}) - p{index}, {si2}) : n{index};
    if (du{index}) {{ u{index}[ii] = n{index}; }}
    f{index}[ii] = v{index};"""
    return f"""    float2 fp{index} = f{index}[ii];
    const bool dsig{index} = {dsig};
    const bool dsigu{index} = du{index};
    float2 up{index} = dsigu{index} ? u{index}[ii] : float2(0.0f, 0.0f);
    float2 hp{index} = dsig{index} ? h{index}[ii] : float2(0.0f, 0.0f);
    if (dsig{index} && dsigu{index}) {{
        float2 hn{index} = c_mul_field_left(c_mul_field_left(hp{index}, cf{index}[ii]) - curl{index}, ci{index}[ii]);
        float2 hs{index} = hn{index} - hp{index};
        float2 un{index} = c_mul_field_left(c_mul_field_left(up{index}, {km1}) + hs{index}, {si1});
        float2 fn{index} = c_mul_field_left((c_mul_field_left(fp{index}, {km2}) + un{index}) - up{index}, {si2});
        h{index}[ii] = hn{index}; u{index}[ii] = un{index}; f{index}[ii] = fn{index};
    }} else if (dsigu{index}) {{
        float2 un{index} = c_mul_field_left(c_mul_field_left(up{index}, cf{index}[ii]) - curl{index}, ci{index}[ii]);
        float2 fn{index} = c_mul_field_left((c_mul_field_left(fp{index}, {km2}) + un{index}) - up{index}, {si2});
        u{index}[ii] = un{index}; f{index}[ii] = fn{index};
    }} else if (dsig{index}) {{
        float2 hn{index} = c_mul_field_left(c_mul_field_left(hp{index}, cf{index}[ii]) - curl{index}, ci{index}[ii]);
        float2 hs{index} = hn{index} - hp{index};
        float2 fn{index} = c_mul_field_left(c_mul_field_left(fp{index}, {km1}) + hs{index}, {si1});
        h{index}[ii] = hn{index}; f{index}[ii] = fn{index};
    }} else {{
        f{index}[ii] = c_mul_field_left(c_mul_field_left(fp{index}, cf{index}[ii]) - curl{index}, ci{index}[ii]);
    }}"""


def _conductive_template() -> str:
    """Derive only the new signature/tail from the certified complex stencil."""
    template = base._CURL_TEMPLATE  # noqa: SLF001 - checked source contract below
    if template.count(_BASE_SIGNATURE) != 1:
        raise RuntimeError("the complex PML curl signature changed; recut this ABI")
    if template.count(_LOSSLESS_TAIL) != 1:
        raise RuntimeError("the complex PML curl tail changed; recut this recurrence")
    if template.count("{\n__GUARD__") != 1:
        raise RuntimeError("the complex PML curl body opening changed; recut phase binding")
    return (template.replace(_BASE_SIGNATURE, _CONDUCTIVE_SIGNATURE)
            .replace(_LOSSLESS_TAIL, "__CONDUCTIVE_TAIL__")
            .replace("{\n__GUARD__", "{\n    float2 px = phases[0];\n"
                     "    float2 py = phases[1];\n"
                     "    float2 pz = phases[2];\n__GUARD__"))


_TEMPLATE = _conductive_template()


def complex_conductive_pml_curl_source(
        codes: Sequence[int], backward: bool, phased: Sequence[int],
        conductive: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """Emit the specialized complex PML/conductivity curl source.

    ``phased`` is the per-axis connection kind of
    :func:`.complex_fields.phase_arguments`; the chunk division is the bound record.
    """
    codes = tuple(int(code) for code in codes)
    phased = tuple(int(flag) for flag in phased)
    flags = tuple(bool(flag) for flag in conductive)
    if len(codes) != 3 or len(phased) != 3 or len(flags) != 3:
        raise ValueError("codes, phased, and conductive must each be triples")
    for axis, (code, flag) in enumerate(zip(codes, phased)):
        if flag not in base.CONNECT_KINDS:
            raise ValueError(f"axis {axis} connection kind {flag!r} is not one of "
                             f"{base.CONNECT_KINDS}")
        if flag and code != templates.PERIODIC:
            raise ValueError(f"axis {axis} carries a phase but is not periodic")
    du = shaders.curl_membership_flags(backward)
    half = shaders.CURL_HALF_INTEGER[bool(backward)]
    dsig = tuple(_chunk_bounds.membership_name(half, dsig_axis)
                 for dsig_axis, _dsigu_axis in shaders.CURL_CYCLE)
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(),
        "__RECORD_STRUCT__": _chunk_bounds.struct_source(),
        "__RECORD_BINDING__": _chunk_bounds.binding(RECORD_BUFFER),
        "__GUARD__": templates.GUARD,
        "__DECODE__": templates.DECODE_IJK,
        "__BOUNDS__": shaders.curl_bounds_block(backward),
        "__SHIFT__": "- 1" if backward else "+ 1",
        "__GHOST_X__": templates.ghost("x", codes[0], backward),
        "__GHOST_Y__": templates.ghost("y", codes[1], backward),
        "__GHOST_Z__": templates.ghost("z", codes[2], backward),
        "__PHASE_X__": base._phase_block("x", phased[0]),  # noqa: SLF001
        "__PHASE_Y__": base._phase_block("y", phased[1]),  # noqa: SLF001
        "__PHASE_Z__": base._phase_block("z", phased[2]),  # noqa: SLF001
        "__MASK__": templates.ownership_mask(
            codes, backward, zero=templates.COMPLEX_ZERO),
        "__DU0__": du[0],
        "__DU1__": du[1],
        "__DU2__": du[2],
        "__CONDUCTIVE_TAIL__": "\n\n".join((
            _tail(0, flags[0], "km_y", "si_y", "km_z", "si_z", dsig[0]),
            _tail(1, flags[1], "km_z", "si_z", "km_x", "si_x", dsig[1]),
            _tail(2, flags[2], "km_x", "si_x", "km_y", "si_y", dsig[2]),
        )),
    })


def compile_complex_conductive_pml_curl(
        codes: Sequence[int], backward: bool, phased: Sequence[int],
        conductive: Sequence[bool],
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """Compile one complex conductive PML specialization."""
    return compile_source(complex_conductive_pml_curl_source(
        codes, backward, phased, conductive,
        contract)).bloch_pml_curl_step


class _CurlScopeView:
    """Mask whole-product facts irrelevant to curl arithmetic before reusing it."""

    __slots__ = ("_fields",)

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> Any:
        if name == "condfac_for":
            return lambda _target: None
        if name == "has_magnetic_conductivity":
            return False
        if name == "polarizations":
            return ()
        if name == "has_polarizations":
            return False
        return getattr(object.__getattribute__(self, "_fields"), name)


class _ConstitutiveScopeView:
    """Expose the established complex H/E body without curl conductivity.

    ``complex_fields._complex_grid_reasons`` is intentionally shared by its
    original curl and constitutive predicates, and therefore normally rejects a
    conductivity on every caller.  That is correct for its curl but over-broad for
    H/E: ``stepping.update_H`` and ``stepping.update_E`` never read
    ``condfac_for`` or ``condinv_for``.  This view masks those two readers only.
    It does *not* hide polarizations, nonlinearity, tensor rows, beta, storage, or
    any geometry fact, so the reused constitutive predicate stays fail-closed for
    every change that actually alters H/E arithmetic.
    """

    __slots__ = ("_fields",)

    def __init__(self, fields: Any) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> Any:
        if name in {"condfac_for", "condinv_for"}:
            return lambda _target: None
        return getattr(object.__getattribute__(self, "_fields"), name)


def metal_complex_conductive_pml_curl_coverage(
        fields: Any, pml: Any, sub_step: str, residency: Any = None, *,
        chunk_layout: Any = None) -> Coverage:
    """Whether one complex conductive PML curl slot is fully described.

    Beside the complex PML clauses and the conductivity volumes, the per-chunk
    conductivity presence of the sub-step's lossy targets
    (:func:`.conductive_pml.conductive_presence_reasons`, read on ``chunk_layout``, the
    division the run steps with; ``None`` is the one-rank division of the grid and
    layer): a sub-step whose lossy target is not kept by every chunk is refused.
    """
    if sub_step not in SUB_STEPS:
        raise ValueError(f"sub_step must be one of {tuple(SUB_STEPS)}, got {sub_step!r}")
    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, ("fields carries no grid",))
    shared = base.complex_pml_curl_coverage(
        _CurlScopeView(fields), pml, sub_step, residency)
    reasons: List[str] = list(shared.reasons)
    # Curl arithmetic is compatible with the supported electric ADE states, but
    # malformed/magnetic states are still named rather than silently masked.
    reasons.extend(_susceptibility_reasons(fields))
    reader, inverse_reader = (getattr(fields, "condfac_for", None),
                              getattr(fields, "condinv_for", None))
    if not callable(reader) or not callable(inverse_reader):
        reasons.append("fields does not expose callable condfac_for/condinv_for")
        return Coverage(False, tuple(dict.fromkeys(reasons)))
    any_conductive = False
    shape = tuple(getattr(grid, "shape", ()))
    for target in CURL_TARGETS:
        try:
            condfac, condinv = reader(target), inverse_reader(target)
        except Exception as exc:  # noqa: BLE001 - unreadable is a refusal
            reasons.append(f"condfac_for/condinv_for({target!r}) raised {exc!r}")
            continue
        if (condfac is None) != (condinv is None):
            reasons.append(f"{target} has only one of condfac/condinv")
            continue
        if condfac is None:
            continue
        any_conductive = True
        reasons.extend(_volume_reasons(f"condfac[{target}]", condfac, shape))
        reasons.extend(_volume_reasons(f"condinv[{target}]", condinv, shape))
        history = getattr(fields, "f_cond_" + target, None)
        if history is None:
            reasons.append(f"f_cond_{target} is not allocated")
        else:
            reasons.extend(base._complex_volume_reasons(  # noqa: SLF001
                f"f_cond_{target}", history, shape))
    if not any_conductive:
        reasons.append("no curl target carries a conductivity")
    if not reasons:
        reasons.extend(conductive_presence_reasons(fields, pml, sub_step, chunk_layout))
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


def metal_complex_conductive_pml_constitutive_coverage(
        fields: Any, pml: Any, side: str, residency: Any = None, *,
        chunk_layout: Any = None) -> Coverage:
    """Whether a complex H/E plan can join a conductive-PML curl composition.

    The curl predicate is intentionally included as a companion condition.  It
    establishes that a live, well-formed conductivity history exists on the same
    PML configuration, while the scoped H/E predicate establishes that the
    constitutive source and PML half-lattice are still the ordinary pointwise
    ones.  The conjunction keeps this family a complete four-slot product rather
    than letting its H/E arms make a partial configuration look fully covered, so it
    also carries the D side's per-chunk conductivity presence: a configuration whose
    ``step_D`` this family refuses for it gets none of the family's H/E arms either.
    """
    if side not in base.CONSTITUTIVE_SIDES:
        raise ValueError(f"side must be one of {tuple(base.CONSTITUTIVE_SIDES)}, got {side!r}")
    scoped = base.complex_constitutive_coverage(
        _ConstitutiveScopeView(fields), pml, side, residency)
    companion = metal_complex_conductive_pml_curl_coverage(
        fields, pml, "step_B", residency, chunk_layout=chunk_layout)
    presence = (conductive_presence_reasons(fields, pml, "step_D", chunk_layout)
                if companion.covered else [])
    return Coverage(scoped.covered and companion.covered and not presence,
                    tuple(dict.fromkeys(scoped.reasons + companion.reasons
                                        + tuple(presence))))


class MetalComplexConductivePmlCurlPlan(KernelPlan):
    """Persistent complex PML mirrors, including only live conductivity history."""

    __slots__ = ("sub_step", "shape", "n_elem", "dtdx", "backward", "bc",
                 "phased", "phase_table", "conductive", "residency",
                 "volumes", "chunk_bounds")
    REPR_FIELDS = ("sub_step", "shape", "bc", "phased", "conductive")

    def __init__(self, sub_step: str, shape: Sequence[int], dtdx: float,
                 codes: Sequence[int], phased: Sequence[int], phase_table: Any,
                 conductive: Sequence[bool], residency: Residency,
                 targets, auxiliaries, history, condfac, condinv, sources,
                 coefficients, functions: Dict[str, Any], volumes: Sequence[str], *,
                 chunk_bounds: Any) -> None:
        self.sub_step = sub_step
        self.shape = tuple(int(value) for value in shape)
        self.n_elem = int(np.prod(self.shape))
        self.dtdx = float(dtdx)
        self.backward = bool(SUB_STEPS[sub_step]["backward"])
        self.bc = tuple(int(code) for code in codes)
        self.phased = tuple(int(flag) for flag in phased)
        self.phase_table = phase_table
        self.conductive = tuple(bool(flag) for flag in conductive)
        self.residency = residency
        self.volumes = tuple(volumes)
        self.chunk_bounds = _chunk_bounds.plan_words(
            chunk_bounds, self.shape, f"MetalComplexConductivePmlCurlPlan({sub_step})")
        super().__init__(functions, tuple(targets) + tuple(auxiliaries) + tuple(history)
                         + tuple(condfac) + tuple(condinv) + tuple(sources)
                         + tuple(coefficients) + (self.shape[0], self.shape[1],
                                                  self.shape[2], self.n_elem, self.dtdx,
                                                  phase_table,
                                                  _chunk_bounds.tensor(
                                                      self.chunk_bounds,
                                                      str(getattr(residency, "device",
                                                                  "mps")))))


def _functions(codes, backward, phased, conductive, variants):
    return {mode: compile_complex_conductive_pml_curl(
        codes, backward, phased, conductive, mode) for mode in variants}


def _phase_table(residency: Residency, sub_step: str,
                 values: Sequence[Tuple[float, float]]) -> Any:
    """Mirror the three connection phase pairs of the sub-step's side."""
    name = f"bloch_phase:{FAMILY}:{sub_step}"
    table = np.asarray(values, dtype=np.float32)
    existing_reader = getattr(residency, "host", None)
    existing = existing_reader(name) if callable(existing_reader) else None
    if existing is not None:
        if not np.array_equal(existing, table):
            raise ValueError(f"{name} was already bound with different phase values")
        table = existing
    return residency.mirror(name, table, constant=True, dtype=np.float32)


def plan_metal_complex_conductive_pml_curl(
        fields: Any, pml: Any, sub_step: str, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Dict[str, Any]] = None, *, chunk_bounds: Any,
        chunk_layout: Any = None,
        ) -> Optional[MetalComplexConductivePmlCurlPlan]:
    """Build one covered complex conductive PML curl plan, otherwise ``None``.

    ``chunk_bounds`` is the record of the division the run steps with (the composer's
    ``StepContext.chunk_bounds``) and has no default; ``chunk_layout`` is that
    division, read for the per-chunk conductivity presence (``None``: the one-rank
    division of the grid and layer).
    """
    if not metal_complex_conductive_pml_curl_coverage(
            fields, pml, sub_step, residency, chunk_layout=chunk_layout).covered:
        return None
    assert residency is not None
    grid, spec = fields.grid, SUB_STEPS[sub_step]
    kinds = _boundary_kinds(grid, pml)
    codes = tuple(1 if kind == "metallic" else 0 for kind in kinds)
    phased, values = base.phase_arguments(
        base.bloch_phase_table(grid, kinds), backward=bool(spec["backward"]))
    targets = [base._complex_mirror(residency, name, getattr(fields, name))  # noqa: SLF001
               for name in spec["targets"]]
    auxiliaries = [base._complex_mirror(  # noqa: SLF001
        residency, "fu_" + name, getattr(fields, "fu_" + name))
        for name in spec["targets"]]
    conductive = conductive_targets(fields, sub_step)
    history = [
        (base._complex_mirror(residency, "f_cond_" + name,  # noqa: SLF001
                              getattr(fields, "f_cond_" + name))
         if conductive[index] else targets[index])
        for index, name in enumerate(spec["targets"])]
    condfac = [
        (residency.mirror(f"condfac_{name}:{FAMILY}", fields.condfac_for(name),
                          constant=True)
         if conductive[index] else targets[index])
        for index, name in enumerate(spec["targets"])]
    condinv = [
        (residency.mirror(f"condinv_{name}:{FAMILY}", fields.condinv_for(name),
                          constant=True)
         if conductive[index] else targets[index])
        for index, name in enumerate(spec["targets"])]
    sources = [base._complex_mirror(residency, name, getattr(fields, name))  # noqa: SLF001
               for name in spec["sources"]]
    coefficients = [
        residency.mirror(f"pml:{stem}_{axis}{spec['suffix']}",
                         getattr(pml, f"{stem}_{axis}{spec['suffix']}"), constant=True)
        for axis in "xyz" for stem in ("kms", "sinv")]
    phase_table = _phase_table(residency, sub_step, values)
    selected = functions if functions is not None else _functions(
        codes, bool(spec["backward"]), phased, conductive,
        contract_variants)
    volumes = (tuple(spec["targets"]) + tuple("fu_" + name for name in spec["targets"])
               + tuple("f_cond_" + name for index, name in enumerate(spec["targets"])
                       if conductive[index]) + tuple(spec["sources"]))
    return MetalComplexConductivePmlCurlPlan(
        sub_step, grid.shape, grid.dt / grid.dx, codes, phased, phase_table,
        conductive, residency, targets, auxiliaries, history, condfac,
        condinv, sources, coefficients, selected, volumes, chunk_bounds=chunk_bounds)


def plan_metal_complex_conductive_pml_constitutive(
        fields: Any, pml: Any, side: str, residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        *, chunk_bounds: Any, chunk_layout: Any = None) -> Optional[Any]:
    """Build the conductivity-neutral complex H/E companion plan, or refuse."""
    if not metal_complex_conductive_pml_constitutive_coverage(
            fields, pml, side, residency, chunk_layout=chunk_layout).covered:
        return None
    return base.plan_complex_constitutive(
        _ConstitutiveScopeView(fields), pml, side, residency,
        contract_variants, chunk_bounds=chunk_bounds)


def _arm_coverage(context: Any, slot: str) -> Coverage:
    return metal_complex_conductive_pml_curl_coverage(
        context.fields, context.pml, slot, context.residency, chunk_layout=context.chunk_layout)


def _arm_plan(context: Any, slot: str) -> Optional[MetalComplexConductivePmlCurlPlan]:
    return plan_metal_complex_conductive_pml_curl(
        context.fields, context.pml, slot, context.residency,
        context.contract_variants,
        chunk_bounds=context.chunk_bounds, chunk_layout=context.chunk_layout)


def _constitutive_arm_coverage(context: Any, slot: str) -> Coverage:
    side = "H" if slot == "update_H" else "E"
    return metal_complex_conductive_pml_constitutive_coverage(
        context.fields, context.pml, side, context.residency, chunk_layout=context.chunk_layout)


def _constitutive_arm_plan(context: Any, slot: str) -> Optional[Any]:
    side = "H" if slot == "update_H" else "E"
    return plan_metal_complex_conductive_pml_constitutive(
        context.fields, context.pml, side, context.residency,
        context.contract_variants,
        chunk_bounds=context.chunk_bounds, chunk_layout=context.chunk_layout)


def _arm_gate(context: Any) -> bool:
    return bool(getattr(context.pml, "is_active", False))


def register_arms() -> Tuple[Any, ...]:
    from . import arms  # noqa: PLC0415

    registered = [arms.register(
        family=FAMILY, slot=slot, label="complex conductive PML curl",
        coverage=_arm_coverage, plan=_arm_plan,
        prefix="complex conductive PML curl: ",
        noun="complex conductive PML curl", gate=_arm_gate, wired=True,
    ) for slot in SUB_STEPS]
    registered.extend(
        arms.register(
            family=FAMILY, slot=slot,
            label="complex conductive PML constitutive",
            coverage=_constitutive_arm_coverage, plan=_constitutive_arm_plan,
            prefix="complex conductive PML constitutive: ",
            noun="complex conductive PML constitutive", gate=_arm_gate, wired=True)
        for slot in ("update_H", "update_E"))
    return tuple(registered)


ARMS = register_arms()
