"""The folded-complex ``step_D`` welded into the complex folded OFF-DIAGONAL E.

THE WIDEST OF THE FOUR STENCIL WELDS — the only cell either board priced over the
ceiling by SEVEN rather than six, and the one that carries a fold, complex
storage, an absorber and a Bloch phase table at once. The 2026-09-01 Metal board
scores it::

    D->E  (folded complex, complex folded off-diagonal PML E)   3 rows, 1 clearing
        folded_complex.bloch_curl 15p
      + complex_folded_offdiag_update_e.offdiag 25p, sharing 3
        -> 37 pointers, ceiling 30                     UNFUSABLE ON METAL
        verdict: STRUCTURALLY UNFUSABLE ON ANY BACKEND (the stencil)

THE DESIGN IS :mod:`.offdiag_fused_electric_pair`'s, and its docstring is the one
place that argument lives: scratch outputs, re-derivation from pre-launch state
through one inline ``step_cell``, the three in-seam passes carried by the closed
form, and the launcher's post-launch rotation. THE SEAM MACHINERY IS
:mod:`.folded_offdiag_fused_electric_pair`'s: both ghost fills are live, so
:data:`REPLACES` names FIVE driver passes and the three runtime reflect rows ride
in ``Params`` as ``d_final_c``'s own parameters.

WHAT IS THIS FAMILY'S ALONE:

* **the parity is a COMPLEX MULTIPLY at BOTH signs, never a copy and never a
  negation.** ``c_mul((+/-1.0f, +0.0f), z)`` is the folded complex fill's own
  arithmetic (``folded_complex.mirror_parity_coefficients`` rounds both words
  through ``numpy.complex64`` on the host and the fill launches the product at
  +1 too), and the distinction is MEASURED rather than inherited: over a hundred
  engineered (real, imag) pairs of signed zeros, subnormals and normals, the
  array path's ``phase * plane`` and this ``c_mul`` agree on every word at both
  signs, while a plain copy at +1 and a bare ``-z`` at -1 — the spelling the real
  families use, correctly, on float32 — differ in TEN of the hundred at each
  sign. One of those ten is reachable on a real fixture: a folded complex grid
  with a metallic wall, where the far fill images a cell the wall clear has
  already zeroed and then multiplies it (``results/metal_scratch_weld_closed_\
form_2026-09-01T2``, case ``D_fold_complex_walled``, the
  ``parity_spelled_as_bare_sign`` null, one differing word).
* **a multi-axis corner is NESTED, not one product.** Same artifact, word-level
  leg: nested and single-product complex parities differ in 39 of 400 engineered
  combinations. No fixture on this corpus arms it (a corner source is stored cell
  2, which the wall clear never reaches), so it is measured at word level and the
  null is recorded unreachable WITH that derivation rather than with a hope.
* **two phase carriers, and they are different shapes.** The curl reads three
  ``float2`` SCALARS (its own wrapped-lane phases) and the constitutive reads a
  six-element ``float2`` VECTOR ``ph`` (the down set then the up set, packed on
  the host exactly as :func:`.complex_folded_offdiag_update_e.plan_metal_complex_\
folded_offdiag` packs it). The scalars ride in ``Params``; ``ph`` stays its own
  pointer, because it is a device VECTOR and the coefficient pack takes
  ``float`` members only.

THE BINDING COUNT. Unpacked and unshared the signature needs 43 pointers. The
twelve per-axis ``float`` coefficient vectors ride ``cpack`` (-11) and the nine
read-only material volumes ``mpack`` (-8), leaving 24 pointers plus ``Params`` —
:data:`PACKED_BINDINGS` = 25, the widest shipped stencil weld and six under the
ceiling. The extra pointer over its three siblings is ``ph``, the constitutive's
six-element complex phase VECTOR, which cannot join ``cpack``: that pack binds
``device const float*`` members and a ``float2`` one would be read at half stride
by every certified line that indexes it. With ``cpack`` alone the signature is 33
bindings against a ceiling of 31, which the gate compiles and REQUIRES to fail.

CARRIES_DEPOSIT_REPAIR IS FALSE and is forced by the off-diagonal constitutive,
as on all four. That is the distance between this cell's 3 rows and its 1.

NOT WIRED AS AN ARM; reached through ``launch.FUSED_PAIR_ARMS``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import (coefficient_pack, complex_fields as _complex,
               complex_folded_offdiag_update_e as _offdiag_cf,
               folded_complex as _folded_complex,
               folded_offdiag_update_e as _folded_real,
               offdiag_update_e as _offdiag, shaders, templates)
from . import offdiag_weld_common as _weld
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .launch import SUB_STEPS
from .offdiag_fused_electric_pair import (
    CURL_STORE_EDITS, NEIGHBOUR_NAMES, PACKED_VECTORS, PACKED_VOLUMES,
)
from .folded_offdiag_fused_electric_pair import (
    REFLECT_ACTUALS, REFLECT_FIELDS, REFLECT_PARAMETERS, folded_axis_tables,
)
from .. import deposit_repair as _deposit_repair

#: Forced False: ``deposit_repair.repairable`` refuses an off-diagonal chi1inv.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "folded_complex_offdiag_fused_electric_pair"

SLOT = "step_D"

#: FIVE driver passes: on a folded grid both ghost fills are LIVE.
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The complex emitters' decode anchors.
CURL_DECODE_END = "    int nyz = nyi * nzi;\n"
CONSTITUTIVE_DECODE_END = "    int nxi = int(nx), nyz = nyi * nzi;\n"

#: ``step_cell``'s pointer parameters, under the certified curl body's own names.
STEP_POINTERS: Tuple[Tuple[str, str], ...] = (
    tuple(("device const float2*", name) for name in
          ("f0", "f1", "f2", "u0", "u1", "u2", "g0", "g1", "g2"))
    + tuple(("device const float*", name) for name in PACKED_VECTORS[:6]))

_ACTUALS: Dict[str, str] = {"f0": "pf0", "f1": "pf1", "f2": "pf2",
                            "u0": "pu0", "u1": "pu1", "u2": "pu2"}

STEP_SCALARS: Tuple[Tuple[str, str], ...] = (
    ("float", "dtdx"), ("float2", "px"), ("float2", "py"), ("float2", "pz"))

STEP_ARGUMENTS = ", " + ", ".join(_ACTUALS.get(name, name)
                                  for _kind, name in STEP_POINTERS)
CALL_ARGUMENTS = (", " + ", ".join(f"{kind} {name}"
                                   for kind, name in STEP_POINTERS)
                  + ", " + ", ".join(f"{kind} {name}"
                                     for kind, name in STEP_SCALARS))
FORWARD = (", ".join(_weld.COORDS) + ", nxi, nyi, nzi" + STEP_ARGUMENTS
           + ", dtdx, px, py, pz")

#: The counts. The three ``float2`` phases and the three reflect ints are
#: ``Params`` members and cost NO binding.
SEPARATE_SCALAR_BINDINGS = 54
UNPACKED_POINTER_BINDINGS = 44
CPACK_ONLY_BINDINGS = 33
PACKED_BINDINGS = 25
UNPACKED_POINTERS = 43
PACKED_POINTERS = 24

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the folded complex off-diagonal fused pair binds {PACKED_BINDINGS} "
        f"buffers and Metal's ceiling is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CPACK_ONLY_BINDINGS", "FAMILY", "PACKED_BINDINGS",
    "PACKED_POINTERS", "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "UNPACKED_POINTERS", "UNPACKED_POINTER_BINDINGS", "certified_curl_body",
    "compile_folded_complex_offdiag_fused_electric_pair",
    "folded_complex_offdiag_fused_electric_pair_source",
    "metal_folded_complex_offdiag_fused_electric_pair_coverage",
    "params_record_dtype",
    "plan_metal_folded_complex_offdiag_fused_electric_pair",
    "refuted_separate_scalar_source", "register_arms",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

__HELPERS__

struct Params {
    float2 px; float2 py; float2 pz;
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
__PACK_FIELDS__
    int reflect_x; int reflect_y; int reflect_z;
};

__STEP_CELL__

__D_FINAL__

kernel void folded_complex_offdiag_fused_electric_pair_step(
    device float2*       d0s     [[buffer(0)]],
    device float2*       d1s     [[buffer(1)]],
    device float2*       d2s     [[buffer(2)]],
    device float2*       n0s     [[buffer(3)]],
    device float2*       n1s     [[buffer(4)]],
    device float2*       n2s     [[buffer(5)]],
    device const float2* pf0     [[buffer(6)]],
    device const float2* pf1     [[buffer(7)]],
    device const float2* pf2     [[buffer(8)]],
    device const float2* pu0     [[buffer(9)]],
    device const float2* pu1     [[buffer(10)]],
    device const float2* pu2     [[buffer(11)]],
    device float2*       f0      [[buffer(12)]],
    device float2*       f1      [[buffer(13)]],
    device float2*       f2      [[buffer(14)]],
    device float2*       w0      [[buffer(15)]],
    device float2*       w1      [[buffer(16)]],
    device float2*       w2      [[buffer(17)]],
    device const float2* g0      [[buffer(18)]],
    device const float2* g1      [[buffer(19)]],
    device const float2* g2      [[buffer(20)]],
    device const float2* ph      [[buffer(21)]],
    device const float*  cpack   [[buffer(22)]],
    device const float*  mpack   [[buffer(23)]],
    constant Params&     prm     [[buffer(24)]],
    uint idx [[thread_position_in_grid]])
{
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    float2 px = prm.px, py = prm.py, pz = prm.pz;
    int reflect_x = prm.reflect_x;
    int reflect_y = prm.reflect_y;
    int reflect_z = prm.reflect_z;
__CPACK_PROLOGUE__
__MPACK_PROLOGUE__
__BODY__
}
"""


def certified_curl_body(codes: Sequence[int], phased: Sequence[int],
                        expansion: str,
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The folded complex ``step_D`` curl's body as an inline function."""
    source = _folded_complex.folded_bloch_curl_source(
        codes, bool(SUB_STEPS["step_D"]["backward"]), phased, expansion, contract)
    tail = _weld.lift_curl_tail(source, store_edits=CURL_STORE_EDITS,
                                decode_end=CURL_DECODE_END)
    return _weld.step_cell_function(
        "step_cell", tail, value_type="float2",
        pointer_parameters=STEP_POINTERS, scalar_parameters=STEP_SCALARS,
        returns=("v0", "v1", "v2", "n0", "n1", "n2"))


def folded_complex_offdiag_fused_electric_pair_source(
        row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
        phased: Sequence[int], negate: Sequence[int],
        folded_axes: Mapping[int, int], far_axes: Mapping[int, int],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source."""
    row_mask = tuple(int(flag) for flag in row_mask)
    walls = tuple(int(flag) for flag in walls)
    if len(row_mask) != len(_offdiag.ROW_SLOTS):
        raise ValueError(f"row_mask must fill the {len(_offdiag.ROW_SLOTS)} slots "
                         f"of ROW_SLOTS, got {row_mask!r}")
    if not folded_axes:
        raise ValueError(
            "no folded axis: that configuration is "
            "complex_no_pml_offdiag_fused_electric_pair's or the plain complex "
            "product's, and emitting this source for it would overlap them")

    needles = _weld.displacement_needles(
        row_mask, e_terms=_offdiag_cf.E_TERMS,
        transverse_partners=_offdiag_cf.TRANSVERSE_PARTNERS
        if hasattr(_offdiag_cf, "TRANSVERSE_PARTNERS")
        else _offdiag.TRANSVERSE_PARTNERS,
        neighbour_names=NEIGHBOUR_NAMES, index=_offdiag_cf._index,
        register="dfin{0}",
        call="d_final_{0}({1}, nxi, nyi, nzi" + STEP_ARGUMENTS
             + ", dtdx, px, py, pz" + REFLECT_ACTUALS + ")")
    body = _weld.weld_body(
        _offdiag_cf.complex_folded_offdiag_source(
            row_mask, codes, walls, phased, negate, expansion, contract),
        decode_end=CONSTITUTIVE_DECODE_END,
        seam=_weld.seam_lines(value_type="float2", forward=FORWARD,
                              split_field=True,
                              d_final_arguments=REFLECT_ACTUALS),
        needles=needles)
    finals = "\n\n".join(
        _weld.d_final_function(
            component, step_name="step_cell", call_arguments=CALL_ARGUMENTS,
            value_type="float2", folded_axes=folded_axes, far_axes=far_axes,
            zero_metal=walls, complex_storage=True,
            own_parameters=REFLECT_PARAMETERS)
        for component in range(3))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": templates.contraction_pragma(contract),
        "__HELPERS__": templates.complex_helpers(expansion),
        "__PACK_FIELDS__": (coefficient_pack.params_fields(PACKED_VECTORS)
                            + "\n" + coefficient_pack.params_fields(PACKED_VOLUMES)),
        "__STEP_CELL__": certified_curl_body(codes, phased, expansion, contract),
        "__D_FINAL__": finals,
        "__CPACK_PROLOGUE__": coefficient_pack.prologue("cpack", PACKED_VECTORS),
        "__MPACK_PROLOGUE__": coefficient_pack.prologue("mpack", PACKED_VOLUMES),
        "__BODY__": body,
    })


def compile_folded_complex_offdiag_fused_electric_pair(
        row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
        phased: Sequence[int], negate: Sequence[int],
        folded_axes: Mapping[int, int], far_axes: Mapping[int, int],
        expansion: str, contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one configuration."""
    return compile_source(folded_complex_offdiag_fused_electric_pair_source(
        row_mask, codes, walls, phased, negate, folded_axes, far_axes,
        expansion, contract)
    ).folded_complex_offdiag_fused_electric_pair_step


_WRITTEN: Tuple[str, ...] = ("d0s", "d1s", "d2s", "n0s", "n1s", "n2s",
                             "f0", "f1", "f2", "w0", "w1", "w2")
_COMPLEX_READ: Tuple[str, ...] = ("pf0", "pf1", "pf2", "pu0", "pu1", "pu2",
                                  "g0", "g1", "g2", "ph")


def _refuted(name: str, float_read: Sequence[str],
             scalars: Sequence[Tuple[str, str]], contract: str) -> Tuple[int, str]:
    """One over-ceiling signature with a body that touches every buffer."""
    written, complex_read = _WRITTEN, _COMPLEX_READ
    lines: List[str] = []
    slot = 0
    for label in written:
        lines.append(f"    device float2*       {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in complex_read:
        lines.append(f"    device const float2* {label:<8}[[buffer({slot})]],")
        slot += 1
    for label in float_read:
        lines.append(f"    device const float*  {label:<8}[[buffer({slot})]],")
        slot += 1
    for kind, label in scalars:
        lines.append(f"    {kind:<20} {label:<10}[[buffer({slot})]],")
        slot += 1
    touch = " + ".join(f"{label}[0]" for label in complex_read)
    scale = " + ".join(f"{label}[0]" for label in float_read)
    return slot, "\n".join((
        "#include <metal_stdlib>",
        "using namespace metal;",
        "",
        templates.contraction_pragma(contract),
        "",
        f"kernel void {name}(",
        *lines,
        "    uint idx [[thread_position_in_grid]])",
        "{",
        "    if (idx >= n_elem) { return; }",
        f"    float2 touch = {touch};",
        f"    float scale = {scale};",
        "    d0s[idx] = d0s[idx] + touch * scale * dtdx"
        " * float(nx + ny + nz);",
        *[f"    {label}[idx] = {label}[idx] + touch;"
          for label in written[1:]],
        "}",
        "",
    ))


_UINTS: Tuple[Tuple[str, str], ...] = tuple(
    ("constant uint&", label) for label in ("nx", "ny", "nz", "n_elem"))


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 54-binding all-separate signature this platform REFUSES."""
    slots, source = _refuted(
        "refuted_separate_scalar", PACKED_VECTORS + PACKED_VOLUMES,
        _UINTS + (("constant float&", "dtdx"),)
        + tuple(("constant float2&", label) for label in ("px", "py", "pz"))
        + tuple(("constant int&", label) for label in REFLECT_FIELDS), contract)
    assert slots == SEPARATE_SCALAR_BINDINGS, (slots, SEPARATE_SCALAR_BINDINGS)
    return source


def refuted_cpack_only_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 33-binding ONE-PACK signature — over the ceiling by TWO.

    The measurement that makes ``mpack`` a necessity here: with only the twelve
    per-axis vectors packed, the nine material volumes are still nine pointers,
    and this family carries ``ph`` on top of what its siblings carry.
    """
    float_read = PACKED_VOLUMES + ("cpack",)
    _slots, source = _refuted(
        "refuted_cpack_only", float_read,
        _UINTS + (("constant float&", "dtdx"),), contract)
    # The count under test is the POINTER count plus one packed `Params`; the
    # compiled shape binds its five scalars separately, which is over the ceiling
    # by even more and refuses for the same reason.
    pointers = len(_WRITTEN) + len(_COMPLEX_READ) + len(float_read)
    assert pointers + 1 == CPACK_ONLY_BINDINGS, (pointers, CPACK_ONLY_BINDINGS)
    return source


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_folded_complex_offdiag_fused_electric_pair_coverage(
        fields: Any, pml: Any, sources: Any = None, residency: Any = None,
        probe: Any = None) -> Coverage:
    """May ONE dispatch span the folded complex off-diagonal D/E seam?"""
    reasons: List[str] = []

    curl = _folded_complex.folded_complex_composition_curl_coverage(
        fields, pml, "step_D", residency, probe)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = _offdiag_cf.metal_complex_folded_offdiag_coverage(
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

    mirrored = getattr(grid, "is_mirrored", None)
    if not callable(mirrored) or not any(bool(mirrored(axis)) for axis in range(3)):
        reasons.append(
            "no axis is folded: complex_no_pml_offdiag_fused_electric_pair is the "
            "unfolded complex off-diagonal product")

    for name in ("has_metallic", "is_metallic", "is_mirrored", "mirror_phase",
                 "stored_cells", "owned_cells"):
        if getattr(grid, name, None) is None:
            reasons.append(
                f"grid does not expose {name}; the in-seam fills and the wall "
                f"clear cannot be carried inline without it")

    if tuple(getattr(fields, "polarizations", ()) or ()):
        reasons.append("a susceptibility is registered: this kernel bakes the "
                       "plain constitutive product, whose source is D and not "
                       "(D - sum P)")
    return Coverage(not reasons, tuple(dict.fromkeys(reasons)))


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

#: Three ``float2`` at 0/8/16, five scalars at 24..43, twenty-one pack offsets at
#: 44..127, three reflect ints at 128..139, rounded to the struct's own 8-byte
#: alignment.
PARAMS_ITEMSIZE = 144


def params_record_dtype() -> Any:
    """The host record — THREE ``float2`` FIRST, then scalars, offsets, reflects."""
    import numpy as np  # noqa: PLC0415

    names = ["px", "py", "pz", "nx", "ny", "nz", "n_elem", "dtdx"]
    formats = [("<f4", 2), ("<f4", 2), ("<f4", 2),
               "<u4", "<u4", "<u4", "<u4", "<f4"]
    offsets = [0, 8, 16, 24, 28, 32, 36, 40]
    running = 44
    for label in PACKED_VECTORS + PACKED_VOLUMES:
        names.append(f"off_{label}")
        formats.append("<u4")
        offsets.append(running)
        running += 4
    for label in REFLECT_FIELDS:
        names.append(label)
        formats.append("<i4")
        offsets.append(running)
        running += 4
    if running != 140:  # pragma: no cover - a layout invariant
        raise RuntimeError(f"the folded complex weld's Params record ends at "
                           f"{running} bytes, not the 140 the itemsize assumes")
    return np.dtype({"names": names, "formats": formats, "offsets": offsets,
                     "itemsize": PARAMS_ITEMSIZE})


def _params_tensor(shape: Sequence[int], dtdx: float,
                   curl_phases: Sequence[Tuple[float, float]],
                   offsets: Mapping[str, int],
                   reflect: Sequence[Optional[int]], device: str) -> Any:
    """The curl phases, the scalars, the pack offsets and the reflect rows."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(dtdx)
    for label, (real, imag) in zip(("px", "py", "pz"), tuple(curl_phases)):
        record[label] = (np.float32(real), np.float32(imag))
    for label, offset in offsets.items():
        record[f"off_{label}"] = int(offset)
    for label, row in zip(REFLECT_FIELDS, tuple(reflect)):
        record[label] = -1 if row is None else int(row)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


ROTATED_NAMES: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")


def plan_metal_folded_complex_offdiag_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        probe: Any = None,
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[_weld.ScratchWeldPairPlan]:
    """Build the folded complex fused off-diagonal plan, or ``None``."""
    if not metal_folded_complex_offdiag_fused_electric_pair_coverage(
            fields, pml, sources, residency, probe).covered:
        return None
    import numpy as np  # noqa: PLC0415

    record = probe if probe is not None else _folded_complex.load_expansion_probe()
    expansion = _folded_complex.expansion_from_probe(record)
    if expansion is None:  # pragma: no cover - the predicate already refused
        return None
    from ..stepping import _boundary_kinds  # noqa: PLC0415

    grid = fields.grid
    codes, _reasons = _folded_complex.folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    kinds = _boundary_kinds(grid, pml)
    table = _complex.bloch_phase_table(grid, kinds)
    phased_down, down = _complex.phase_arguments(table, backward=True)
    phased_up, up = _complex.phase_arguments(table, backward=False)
    if phased_down != phased_up:
        raise ValueError("forward and backward phase flags disagree")
    curl_phased, curl_phase = _complex.phase_arguments(
        table, backward=bool(SUB_STEPS["step_D"]["backward"]))
    walls = _offdiag.wall_mask_axes(grid)
    negate = _folded_real.negated_axes(
        codes, _folded_real.mirror_ghost_weights(grid))
    folded_axes, far_axes, reflect = folded_axis_tables(grid)
    rows = _offdiag.row_volumes_for(fields)
    row_mask = tuple(int(value is not None) for value in rows)

    volumes: List[str] = []
    twins: Dict[str, Any] = {}
    for name in ROTATED_NAMES:
        host = getattr(fields, name)
        volumes.append(name)
        _complex._complex_mirror(residency, name, host)
        twin, _mirror = _weld.scratch_twin(residency, name, host,
                                           dtype=np.complex64)
        twins[name] = twin

    electric = tuple(term[0] for term in _offdiag_cf.E_TERMS)
    stored = [_complex._complex_mirror(residency, name, getattr(fields, name))
              for name in electric]
    workspace = [_complex._complex_mirror(residency, "f_w_" + name,
                                          getattr(fields, "f_w_" + name))
                 for name in electric]
    volumes.extend(electric)
    volumes.extend("f_w_" + name for name in electric)
    magnetic = [_complex._complex_mirror(residency, name, getattr(fields, name))
                for name in SUB_STEPS["step_D"]["sources"]]
    volumes.extend(SUB_STEPS["step_D"]["sources"])

    # THE CONSTITUTIVE'S PHASE VECTOR, packed on the host exactly as the certified
    # plan packs it: the DOWN triple then the UP triple, one complex64 each.
    phase_pack = np.asarray([complex(real, imag) for real, imag in (*down, *up)],
                            dtype=np.complex64)
    phases = residency.mirror(f"{FAMILY}:phases", phase_pack, constant=True,
                              dtype=np.complex64)

    suffix = SUB_STEPS["step_D"]["suffix"]
    vectors = ([getattr(pml, f"{stem}_{axis}{suffix}")
                for axis in "xyz" for stem in ("kms", "sinv")]
               + [getattr(pml, f"{stem}_{axis}_h")
                  for axis in "xyz" for stem in ("kps", "kms")])
    cpack, cpack_layout = coefficient_pack.packed_mirror(
        residency, f"{FAMILY}:cpack", np, PACKED_VECTORS, vectors)

    inverse = [fields.inverse_epsilon_for(name) for name in electric]
    component_of_slot = tuple(
        next(index for index, term in enumerate(_offdiag_cf.E_TERMS)
             if term[0] == row)
        for row, _partner in _offdiag_cf.ROW_SLOTS)
    row_volumes = [value if value is not None else inverse[component_of_slot[slot]]
                   for slot, value in enumerate(rows)]
    mpack, mpack_layout = coefficient_pack.packed_mirror(
        residency, f"{FAMILY}:mpack", np, PACKED_VOLUMES, inverse + row_volumes)

    offsets = dict(zip(cpack_layout.names, cpack_layout.offsets))
    offsets.update(zip(mpack_layout.names, mpack_layout.offsets))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_complex_offdiag_fused_electric_pair(
                row_mask, codes, walls, phased_down, negate, folded_axes,
                far_axes, expansion, mode)

    dtdx = grid.dt / grid.dx
    static = (list(stored) + list(workspace) + list(magnetic)
              + [phases, cpack, mpack,
                 _params_tensor(grid.shape, dtdx, curl_phase, offsets, reflect,
                                residency.device)])
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
            (f"folded complex offdiag fused electric pair cannot fill {slot}",))
    return metal_folded_complex_offdiag_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency,
        context.extra.get("probe"))


def _arm_plan(context: Any, slot: str) -> Optional[_weld.ScratchWeldPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_complex_offdiag_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants, context.extra.get("probe"))


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False`` — five absorbed passes declared."""
    from . import arms  # noqa: PLC0415
    return (arms.register(
        FAMILY, SLOT, "folded complex off-diagonal fused electric D/E pair",
        _arm_coverage, _arm_plan,
        prefix="folded complex off-diagonal fused electric D/E pair: ",
        noun="fused folded-complex D-curl/off-diagonal-E pair",
        wired=False, replaces=REPLACES),)


ARMS = register_arms()
