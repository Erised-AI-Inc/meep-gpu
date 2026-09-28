"""The complex no-PML ``step_D`` welded into the complex OFF-DIAGONAL ``update_E``.

THE ONE STENCIL WELD THAT NEEDS NO PACK, and the cheapest evidence that the
scratch design is the answer to the boards' ``STRUCTURALLY UNFUSABLE`` verdict
rather than the binding ceiling being it. The 2026-09-01 Metal board scores this
cell::

    D->E  (complex no-PML curl, complex no-PML off-diagonal)   2 rows, 1 clearing
        complex_no_pml_curl.curl 12p
      + complex_no_pml_offdiag_update_e.offdiag 15p, sharing 3
        -> 24 pointers, ceiling 30
        binding verdict had it been buildable: FITS — 24..24, NOT BUILT
        verdict: STRUCTURALLY UNFUSABLE ON ANY BACKEND (the stencil)

THIS CELL ALWAYS FITTED. Nothing about the ceiling ever refused it: the board
priced it two under and refused it purely on the stencil, so it is the clean
separation of the two obstructions. The scratch weld costs three more pointers
than the shared-D count the board used (pre-launch D and scratch D are different
buffers) and the shipped signature is 21 pointers plus ``Params`` —
:data:`PACKED_BINDINGS` = 22, nine under the ceiling, WITH NO COEFFICIENT PACK
AT ALL. Its two siblings need one pack each for the PML vectors and a second for
the material volumes; this family has no PML at all, so there is nothing to pack
and packing anyway would duplicate nine volumes on the device for no binding
need — the rule :mod:`.conductive_fused_electric_pair` states and this family is
the one that gets to follow it.

WHAT THE NO-PML SHAPE REMOVES, and it is more than the vectors:

* **no split field.** ``step_D`` here is ``D = D - curl`` (no
  ``_apply_pml_update`` recurrence), so there is no ``fu_D`` to write and no
  ``fu_D`` to re-derive. The rotation is three volumes, not six, and
  :func:`.offdiag_weld_common.seam_lines` emits no auxiliary store.
* **no ``f_w_E``.** The constitutive is ``E = (D * inv_eps) + row terms`` with no
  absorber accumulation, so the three workspace pointers are gone too.
* **the store IS the computation.** The certified curl's last three lines are
  ``f0[ii] = f0[ii] - curl0;`` — the value exists nowhere else — so the lift
  REWRITES them to ``float2 v0 = f0[ii] - curl0;`` rather than deleting them.
  The right-hand side is untouched: what moves is the destination, from a volume
  the fused kernel must not write to a register the struct returns. That is the
  second form :func:`.offdiag_weld_common.lift_curl_tail` documents.

THE DECODE ANCHOR IS THE COMPLEX ONE. Both complex emitters put ``int nxi =
int(nx);`` and ``int nyz = nyi * nzi;`` BELOW the ``int i = plane / nyi;`` line
that ends the real emitters' prologue, so cutting at the real anchor would make
the helper redeclare ``nyz`` and read an ``nx`` it has no parameter for. The
curl is cut after ``int nyz = nyi * nzi;`` and the constitutive's seam splices
after ``int nxi = int(nx), nyz = nyi * nzi;``.

THE BLOCH PHASES ARE SIX SEPARATE ``float2`` SCALARS and stay separate: the curl
reads its own wrapped-lane phases (``px``/``py``/``pz``, backward) and the
constitutive reads a DOWN set and an UP set (``dpx``.. / ``upx``..), because the
row product's two shifts go in opposite directions. All nine ride in ``Params``
as ``float2`` members FIRST, which is
:func:`.complex_fused_electric_pair.params_record_dtype`'s measured layout rule.

CARRIES_DEPOSIT_REPAIR IS FALSE and is forced by the off-diagonal constitutive,
exactly as on the three siblings. That is the distance between this cell's 2 rows
and its 1.

NOT WIRED AS AN ARM; reached through ``launch.FUSED_PAIR_ARMS``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import (complex_no_pml_curl as _curl,
               complex_no_pml_offdiag_update_e as _offdiag_c,
               offdiag_update_e as _offdiag, shaders, templates)
from .complex_no_pml_conductive import _source_binding
from . import offdiag_weld_common as _weld
from .complex_fields import (
    _complex_mirror, bloch_phase_table, expansion_from_probe,
    load_expansion_probe, phase_arguments,
)
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .launch import SUB_STEPS
from .. import deposit_repair as _deposit_repair

#: Forced False: ``deposit_repair.repairable`` refuses an off-diagonal chi1inv.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "complex_no_pml_offdiag_fused_electric_pair"

SLOT = "step_D"

#: TWO driver passes. ``zero_metal_D`` is still carried — the closed form's clear
#: arm is emitted whenever an axis is walled — but it is named here only when it
#: does work, and a no-PML complex row may be walled or not. It stays in the list
#: because the kernel carries it unconditionally: the specialisation emits no
#: guard on an unwalled grid, which is the same statement the certified emitters
#: make about their own masks.
REPLACES: Tuple[str, ...] = ("step_D", "zero_metal_D", "update_E")

#: The curl's three store lines. Here the store IS the computation, so the edit
#: REWRITES the destination and leaves the right-hand side alone.
CURL_STORE_EDITS: Tuple[Tuple[str, str], ...] = tuple(
    (f"    f{target}[ii] = f{target}[ii] - curl{target};\n",
     f"    float2 v{target} = f{target}[ii] - curl{target};\n")
    for target in range(3))

#: The complex emitters' decode anchors — see the module docstring.
CURL_DECODE_END = "    int nyz = nyi * nzi;\n"
CONSTITUTIVE_DECODE_END = "    int nxi = int(nx), nyz = nyi * nzi;\n"

#: ``step_cell``'s pointer parameters, under the certified curl body's own names.
#: The six CONDUCTIVITY pointers of the shared complex stencil are NOT here: the
#: all-lossless specialisation emits no read of them, which the builder asserts
#: rather than assumes, so binding them would be six dead pointers.
STEP_POINTERS: Tuple[Tuple[str, str], ...] = tuple(
    ("device const float2*", name) for name in
    ("f0", "f1", "f2", "g0", "g1", "g2"))

_ACTUALS: Dict[str, str] = {"f0": "pf0", "f1": "pf1", "f2": "pf2"}

STEP_SCALARS: Tuple[Tuple[str, str], ...] = (
    ("float", "dtdx"), ("float2", "px"), ("float2", "py"), ("float2", "pz"))

STEP_ARGUMENTS = ", " + ", ".join(
    [_ACTUALS.get(name, name) for _kind, name in STEP_POINTERS])
CALL_ARGUMENTS = (", " + ", ".join(f"{kind} {name}"
                                   for kind, name in STEP_POINTERS)
                  + ", " + ", ".join(f"{kind} {name}"
                                     for kind, name in STEP_SCALARS))
FORWARD = (", ".join(_weld.COORDS) + ", nxi, nyi, nzi" + STEP_ARGUMENTS
           + ", dtdx, px, py, pz")

#: The per-axis ``(down, up)`` neighbour-index variable names, NARROWED from this
#: emitter's own EIGHT-member table (which also carries the validity flags, the
#: home coordinate, the extent and the two Bloch phases). The real emitters' table
#: has five members and is keyed by axis letter; narrowing both to this pair is
#: what lets one needle builder serve every family.
NEIGHBOUR_NAMES: Dict[int, Tuple[str, str]] = {
    axis: (_offdiag_c._NAMES[axis][0], _offdiag_c._NAMES[axis][1])
    for axis in range(3)}

#: The counts. SEPARATE_SCALAR_BINDINGS is 21 pointers + 4 uints + dtdx + nine
#: float2 phases = 35, over the ceiling; PACKED_BINDINGS is the shipped 22.
SEPARATE_SCALAR_BINDINGS = 35
PACKED_BINDINGS = 22
PACKED_POINTERS = 21

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the complex no-PML off-diagonal fused pair binds {PACKED_BINDINGS} "
        f"buffers and Metal's ceiling is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "FAMILY", "PACKED_BINDINGS", "PACKED_POINTERS",
    "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT", "certified_curl_body",
    "compile_complex_no_pml_offdiag_fused_electric_pair",
    "complex_no_pml_offdiag_fused_electric_pair_source",
    "metal_complex_no_pml_offdiag_fused_electric_pair_coverage",
    "params_record_dtype",
    "plan_metal_complex_no_pml_offdiag_fused_electric_pair",
    "refuted_separate_scalar_source", "register_arms",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

struct Params {
    float2 px; float2 py; float2 pz;
    float2 dpx; float2 dpy; float2 dpz;
    float2 upx; float2 upy; float2 upz;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
};

__STEP_CELL__

__D_FINAL__

kernel void complex_no_pml_offdiag_fused_electric_pair_step(
    device float2*       d0s     [[buffer(0)]],
    device float2*       d1s     [[buffer(1)]],
    device float2*       d2s     [[buffer(2)]],
    device const float2* pf0     [[buffer(3)]],
    device const float2* pf1     [[buffer(4)]],
    device const float2* pf2     [[buffer(5)]],
    device float2*       f0      [[buffer(6)]],
    device float2*       f1      [[buffer(7)]],
    device float2*       f2      [[buffer(8)]],
    device const float2* g0      [[buffer(9)]],
    device const float2* g1      [[buffer(10)]],
    device const float2* g2      [[buffer(11)]],
    device const float*  e0      [[buffer(12)]],
    device const float*  e1      [[buffer(13)]],
    device const float*  e2      [[buffer(14)]],
    device const float*  u01     [[buffer(15)]],
    device const float*  u02     [[buffer(16)]],
    device const float*  u11     [[buffer(17)]],
    device const float*  u12     [[buffer(18)]],
    device const float*  u21     [[buffer(19)]],
    device const float*  u22     [[buffer(20)]],
    constant Params&     prm     [[buffer(21)]],
    uint idx [[thread_position_in_grid]])
{
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float2 px = prm.px, py = prm.py, pz = prm.pz;
    float2 dpx = prm.dpx, dpy = prm.dpy, dpz = prm.dpz;
    float2 upx = prm.upx, upy = prm.upy, upz = prm.upz;
__BODY__
}
"""


def certified_curl_body(codes: Sequence[int], phased: Sequence[int],
                        expansion: str,
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The lossless complex ``step_D`` curl's body as an inline function."""
    source = _curl.complex_no_pml_curl_source(
        codes, bool(SUB_STEPS["step_D"]["backward"]), phased, expansion, contract)
    tail = _weld.lift_curl_tail(source, store_edits=CURL_STORE_EDITS,
                                decode_end=CURL_DECODE_END)
    # THE LOSSLESS ARM READS NO CONDUCTIVITY, and this asserts it rather than
    # assuming it: the shared complex stencil DECLARES cf0..ci2 at every
    # specialisation and this family omits them from the helper's parameters, so
    # a specialisation that started reading one would be a compile error at best
    # and a wrong pointer at worst.
    for label in ("cf0", "cf1", "cf2", "ci0", "ci1", "ci2"):
        if label in tail:
            raise AssertionError(
                f"the all-lossless complex curl now reads {label}; this weld omits "
                f"the six conductivity pointers from step_cell's signature")
    return _weld.step_cell_function(
        "step_cell", tail, value_type="float2",
        pointer_parameters=STEP_POINTERS, scalar_parameters=STEP_SCALARS,
        returns=("v0", "v1", "v2"))


def complex_no_pml_offdiag_fused_electric_pair_source(
        row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
        phased: Sequence[int], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source."""
    row_mask = tuple(int(flag) for flag in row_mask)
    walls = tuple(int(flag) for flag in walls)
    if len(row_mask) != len(_offdiag.ROW_SLOTS):
        raise ValueError(f"row_mask must fill the {len(_offdiag.ROW_SLOTS)} slots "
                         f"of ROW_SLOTS, got {row_mask!r}")

    needles = _weld.displacement_needles(
        row_mask, e_terms=_offdiag_c.E_TERMS,
        transverse_partners=_offdiag_c.TRANSVERSE_PARTNERS,
        neighbour_names=NEIGHBOUR_NAMES, index=_offdiag_c._index,
        register="dfin{0}",
        call="d_final_{0}({1}, nxi, nyi, nzi" + STEP_ARGUMENTS
             + ", dtdx, px, py, pz)")
    body = _weld.weld_body(
        _offdiag_c.complex_no_pml_offdiag_source(
            row_mask, codes, walls, phased, expansion, contract),
        decode_end=CONSTITUTIVE_DECODE_END,
        seam=_weld.seam_lines(value_type="float2", forward=FORWARD,
                              split_field=False),
        needles=needles)
    finals = "\n\n".join(
        _weld.d_final_function(
            component, step_name="step_cell", call_arguments=CALL_ARGUMENTS,
            value_type="float2", folded_axes={}, far_axes={},
            zero_metal=walls, complex_storage=True)
        for component in range(3))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__STEP_CELL__": certified_curl_body(codes, phased, expansion, contract),
        "__D_FINAL__": finals,
        "__BODY__": body,
    })


def compile_complex_no_pml_offdiag_fused_electric_pair(
        row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
        phased: Sequence[int], expansion: str,
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one configuration."""
    return compile_source(complex_no_pml_offdiag_fused_electric_pair_source(
        row_mask, codes, walls, phased, expansion, contract)
    ).complex_no_pml_offdiag_fused_electric_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 35-binding all-separate signature this platform REFUSES.

    Twenty-one pointers plus four uints, ``dtdx`` and the NINE ``float2`` phases
    bound one at a time. The body touches every buffer, because what is under
    test is the signature and a droppable body would let dead-code elimination
    answer instead of the compiler's argument table.
    """
    written = ("d0s", "d1s", "d2s", "f0", "f1", "f2")
    read = ("pf0", "pf1", "pf2", "g0", "g1", "g2", "e0", "e1", "e2",
            "u01", "u02", "u11", "u12", "u21", "u22")
    lines: List[str] = []
    slot = 0
    for label in written:
        lines.append(f"    device float2*       {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in read:
        lines.append(f"    device const float2* {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in ("nx", "ny", "nz", "n_elem"):
        lines.append(f"    constant uint&       {label:<8}[[buffer({slot})]],")
        slot += 1
    lines.append(f"    constant float&      dtdx    [[buffer({slot})]],")
    slot += 1
    for label in ("px", "py", "pz", "dpx", "dpy", "dpz", "upx", "upy", "upz"):
        lines.append(f"    constant float2&     {label:<8}[[buffer({slot})]],")
        slot += 1
    assert slot == SEPARATE_SCALAR_BINDINGS, (slot, SEPARATE_SCALAR_BINDINGS)
    touch = " + ".join(f"{label}[0]" for label in read)
    return "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        templates.contraction_pragma(contract),
        "",
        "kernel void refuted_separate_scalar(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float2 touch = {touch};",
        "    d0s[idx] = d0s[idx] + touch * float(nx + ny + nz) * dtdx;",
        *[f"    {label}[idx] = {label}[idx] + px + py + pz;"
          for label in written[1:]],
        "}",
        "",
    ))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_complex_no_pml_offdiag_fused_electric_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span the complex no-PML off-diagonal D/E seam?"""
    reasons: List[str] = []

    curl = _curl.metal_complex_no_pml_curl_coverage(
        fields, pml, "step_D", residency, probe)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = _offdiag_c.metal_complex_no_pml_offdiag_coverage(
        fields, pml, residency, probe)
    if not electric.covered:
        reasons.extend(f"constitutive half: {reason}" for reason in electric.reasons)

    reasons.extend(_deposit_repair.seam_source_reasons(
        fields, sources, 'D',
        undeclared=(
            "the source set was not declared: this predicate cannot infer an "
            "empty electric source seam from Fields"),
        refusal=lambda index, source: (
            f"source {index} ({type(source).__name__}) is electric: the driver "
            f"injects it BETWEEN step_D and update_E (driver.py:3294-3299) and "
            f"deposit_repair.repairable REFUSES an off-diagonal chi1inv "
            f"constitutive, so no repair can carry it"),
        carries_repair=CARRIES_DEPOSIT_REPAIR))

    grid = getattr(fields, "grid", None)
    if grid is None:
        return Coverage(False, tuple(dict.fromkeys(
            reasons + ["fields carries no grid"])))

    for axis in range(3):
        mirrored = getattr(grid, "is_mirrored", None)
        if callable(mirrored) and bool(mirrored(axis)):
            reasons.append(
                f"axis {axis} is folded: stepping.fill_symmetry_bc_D and "
                f"stepping.fill_folded_far_ghosts_D both run inside this seam; "
                f"folded_complex_offdiag_fused_electric_pair carries them")

    for name in ("has_metallic", "is_metallic", "is_mirrored"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; zero_metal_D cannot be carried "
                f"inline")

    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the "
                       "plain constitutive product, whose source is D and not "
                       "(D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

PARAMS_ITEMSIZE = 96


def params_record_dtype() -> Any:
    """The host record — THE NINE ``float2`` MEMBERS FIRST, itemsize 96.

    ONE HOME FOR THE LAYOUT, for the reason
    :func:`.complex_fused_electric_pair.params_record_dtype` gives: getting it
    wrong is a silent wrong answer rather than a crash. Nine ``float2`` at 8-byte
    alignment fill 0..71, the four uints and ``dtdx`` fill 72..91, and Metal
    rounds the struct to the 8-byte alignment of its widest member, so the
    itemsize is 96 where the natural record is 92.
    """
    import numpy as np  # noqa: PLC0415

    phases = ["px", "py", "pz", "dpx", "dpy", "dpz", "upx", "upy", "upz"]
    names = phases + ["nx", "ny", "nz", "n_elem", "dtdx"]
    formats = [("<f4", 2)] * len(phases) + ["<u4", "<u4", "<u4", "<u4", "<f4"]
    offsets = [8 * index for index in range(len(phases))]
    offsets += [72, 76, 80, 84, 88]
    return np.dtype({"names": names, "formats": formats, "offsets": offsets,
                     "itemsize": PARAMS_ITEMSIZE})


def _params_tensor(shape: Sequence[int], dtdx: float,
                   curl_phases: Sequence[Tuple[float, float]],
                   down_phases: Sequence[Tuple[float, float]],
                   up_phases: Sequence[Tuple[float, float]], device: str) -> Any:
    """The nine phases and the five scalars as one 96-byte device record."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(dtdx)
    groups = (("px", "py", "pz"), ("dpx", "dpy", "dpz"), ("upx", "upy", "upz"))
    for labels, values in zip(groups, (curl_phases, down_phases, up_phases)):
        for label, (real, imag) in zip(labels, tuple(values)):
            record[label] = (np.float32(real), np.float32(imag))
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


#: Three rotating volumes: no PML means no split-field auxiliary to rotate.
ROTATED_NAMES: Tuple[str, ...] = ("Dx", "Dy", "Dz")


def plan_metal_complex_no_pml_offdiag_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        probe: Any = None,
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[_weld.ScratchWeldPairPlan]:
    """Build the complex no-PML fused off-diagonal plan, or ``None``."""
    if not metal_complex_no_pml_offdiag_fused_electric_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    expansion = expansion_from_probe(
        probe if probe is not None else load_expansion_probe())
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    import numpy as _np  # noqa: PLC0415

    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    kinds = _boundary_kinds(grid, None)
    codes = tuple(templates.METALLIC if kind == "metallic" else templates.PERIODIC
                  for kind in kinds)
    walls = _offdiag_c.wall_mask_axes(grid)
    table = bloch_phase_table(grid, kinds)
    phased_down, phase_down = phase_arguments(table, backward=True)
    phased_up, phase_up = phase_arguments(table, backward=False)
    if phased_down != phased_up:
        raise ValueError("forward and backward phase flags disagree")
    curl_phased, curl_phase = phase_arguments(
        table, backward=bool(SUB_STEPS["step_D"]["backward"]))
    rows = _offdiag_c.row_volumes_for(fields)
    row_mask = tuple(int(value is not None) for value in rows)

    volumes: List[str] = []
    twins: Dict[str, Any] = {}
    for name in ROTATED_NAMES:
        host = getattr(fields, name)
        volumes.append(name)
        _complex_mirror(residency, name, host)
        twin, _mirror = _weld.scratch_twin(residency, name, host,
                                           dtype=_np.complex64)
        twins[name] = twin

    electric = tuple(term[0] for term in _offdiag_c.E_TERMS)
    stored = [_complex_mirror(residency, name, getattr(fields, name))
              for name in electric]
    volumes.extend(electric)
    # THE MAGNETIC SOURCE IS CANONICALISED, not read off `fields.Hx`. In a no-PML
    # step `Fields.get_H("Hx")` returns the live `Bx` allocation, so binding the
    # logical name would either mirror a `None` or make a SECOND device copy of a
    # volume the B curl already wrote — `_source_binding` is the seam the certified
    # complex curl plan already uses for exactly this.
    source_bindings = [_source_binding(fields, "step_D", name)
                       for name in SUB_STEPS["step_D"]["sources"]]
    magnetic = [_complex_mirror(residency, name, source)
                for name, source in source_bindings]
    volumes.extend(name for name, _source in source_bindings)
    inverse = [residency.mirror(f"inv_eps_{name}:{FAMILY}",
                                fields.inverse_epsilon_for(name), constant=True)
               for name in electric]
    # A DEAD ROW SLOT STILL BINDS: the specialisation is what stops the read, the
    # buffer argument still has to bind, and the component's own inverse epsilon
    # is a resident read-only volume of the right shape.
    component_of_slot = tuple(
        next(index for index, term in enumerate(_offdiag_c.E_TERMS)
             if term[0] == row)
        for row, _partner in _offdiag_c.ROW_SLOTS)
    row_mirrors = [
        residency.mirror(f"chi1inv_offdiag:{row}:{partner}:{FAMILY}", value,
                         constant=True)
        if value is not None else inverse[component_of_slot[slot]]
        for slot, ((row, partner), value) in enumerate(
            zip(_offdiag_c.ROW_SLOTS, rows))]

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_complex_no_pml_offdiag_fused_electric_pair(
                row_mask, codes, walls, phased_down, expansion, mode)

    dtdx = grid.dt / grid.dx
    static = (list(stored) + list(magnetic) + list(inverse) + list(row_mirrors)
              + [_params_tensor(grid.shape, dtdx, curl_phase, phase_down,
                                phase_up, residency.device)])
    assert (2 * len(ROTATED_NAMES) + len(static)) == PACKED_BINDINGS, (
        2 * len(ROTATED_NAMES) + len(static), PACKED_BINDINGS)
    return _weld.plan_scratch_weld(
        FAMILY, residency, fields, rotated_names=ROTATED_NAMES,
        static_args=static, functions=selected, volumes=volumes,
        shape=grid.shape, codes=codes, zero_metal=walls, row_mask=row_mask,
        replaces=REPLACES, twins=twins)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _arm_coverage(context: Any, slot: str) -> Coverage:
    if slot != SLOT:
        return Coverage(
            False,
            (f"complex no-PML offdiag fused electric pair cannot fill {slot}",))
    return metal_complex_no_pml_offdiag_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency,
        context.extra.get("probe"))


def _arm_plan(context: Any, slot: str) -> Optional[_weld.ScratchWeldPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_complex_no_pml_offdiag_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, context.extra.get("probe"))


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False``."""
    from . import arms  # noqa: PLC0415
    return (arms.register(
        FAMILY, SLOT, "complex no-PML off-diagonal fused electric D/E pair",
        _arm_coverage, _arm_plan,
        prefix="complex no-PML off-diagonal fused electric D/E pair: ",
        noun="fused complex no-PML curl/off-diagonal-E pair",
        wired=False, replaces=REPLACES),)


ARMS = register_arms()
