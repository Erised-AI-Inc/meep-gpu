"""The FOLDED ``step_D`` welded into the FOLDED OFF-DIAGONAL ``update_E``.

THE LARGEST OF THE FOUR STENCIL WELDS — 19 corpus rows, 9 of them clearing the
source seam, the biggest single cell either board has carried as ``STRUCTURALLY
UNFUSABLE ON ANY BACKEND``. The 2026-09-01 Metal board scores it::

    D->E  (folded curl, folded offdiag)                19 rows, 9 clearing
        symmetry.folded_curl 15p
      + folded_offdiag_update_e.offdiag 24p, sharing 3
        -> 36 pointers, ceiling 30                     UNFUSABLE ON METAL
        verdict: STRUCTURALLY UNFUSABLE ON ANY BACKEND (the stencil)

THE STENCIL AND THE CEILING ARE ANSWERED EXACTLY AS THE UNFOLDED TWIN ANSWERS
THEM — scratch outputs, re-derivation from pre-launch state, two coefficient
packs — and :mod:`.offdiag_fused_electric_pair`'s docstring is the one place
that argument lives. WHAT IS NEW HERE IS THE SEAM ITSELF: this family runs on a
folded grid, so the two ghost fills are not inert, and :data:`REPLACES` names
FIVE driver passes rather than three.

===========================================================================
THE THREE IN-SEAM PASSES, ALL CARRIED BY THE CLOSED FORM
===========================================================================

The driver runs, between ``step_D`` and ``update_E`` (driver.py:3292-3302):
``step_D`` -> electric sources -> ``fill_symmetry_bc_D`` -> ``zero_metal_D`` ->
``fill_folded_far_ghosts_D`` -> ``update_E``. The last three are carried per
cell by :func:`.offdiag_weld_common.d_final_function`, whose composition is

    far?  ->  parity_far x [ clear ? +0 : near(y) ]      (the mask INSIDE)
    else  ->  clear ? +0 : near(x)                        (the mask OUTSIDE)

with ``near`` the multi-axis redirect carrying ONE PARITY MULTIPLY PER AXIS, in
ascending axis order — because the array path writes plane after plane in X, Y,
Z order (stepping.py:1441-1447), so a doubly-unowned corner carries ``p_y *
(p_x * raw)`` NESTED and not the single product of the two. On real storage the
two spellings are the same bits; the distinction is measured and is why the
nesting is emitted rather than folded.

THE FOLDED AND WALLED AXES ARE DISJOINT BY CONSTRUCTION — ``_zero_metal`` skips
a folded axis by name (stepping.py:2282-2284), because its stored cell 0 holds
the parity ghost rather than a wall — and that disjointness is what makes the
clear guard readable off the UNMUTATED coordinates after the near redirect has
moved the folded ones.

THE FAR ARM IS THE ONE THAT NEEDS A RUNTIME INPUT. Its image row is
``stepping._far_reflect_rows``' ``n_full - stored + 2``, which is ``stored - 2``
at an even full count and ``stored - 3`` at an odd one, so it cannot be baked:
the three rows ride in ``Params`` and reach ``d_final_c`` as parameters of its
own — never forwarded to ``step_cell``, which has no use for them. That is why
:func:`.offdiag_weld_common.d_final_function` takes ``own_parameters``
separately from ``call_arguments``.

THE CLOSED FORM IS MEASURED against the array path's own three passes over
seventeen host fixtures including a two-axis MIXED-PHASE fold, an odd full
count, a walled fold and a folded-metallic termination — zero differing uint32
words over two complete steps, every reachable null diverging
(``parity/meep_gpu/results/metal_scratch_weld_closed_form_2026-09-01T2``) — and
again on device by this family's gate.

===========================================================================
WHAT IS LIFTED
===========================================================================

* the curl — :func:`.symmetry.folded_curl_source` with ``backward=True``, body
  cut at the decode anchor and re-headed as an inline function. Its TOP-PLANE
  MASK comes with it, which matters: that block is the only text the certified
  emitters do not produce, it fires only on a folded PERIODIC axis, and a weld
  that carried the far fill but dropped the mask would leave a plane of
  unmasked cells behind;
* the constitutive — :func:`.folded_offdiag_update_e.folded_offdiag_source`'s
  own output, whose MIRROR GHOST LANE (index redirected to stored 2, sign
  applied as a select on the value) is left exactly as that emitter writes it.
  Only the displacement READS are redirected, by needles built from the
  certified emitter's own index tables;
* ``negate`` is :func:`.folded_offdiag_update_e.negated_axes` of the codes and
  the ghost weights, taken from that module rather than re-derived, because a
  redirect and a sign that disagreed would read stored row 2 without the parity.

CARRIES_DEPOSIT_REPAIR IS FALSE and is forced: ``deposit_repair.repairable``
refuses an off-diagonal chi1inv by name, so the ten rows of this cell that carry
an electric source in the seam are out of reach of any fused product on any
backend. That is the whole distance between the cell's 19 rows and its 9.

NOT WIRED AS AN ARM; reached through ``launch.FUSED_PAIR_ARMS``.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..triton_kernels.coverage import Coverage
from . import (coefficient_pack, folded_offdiag_update_e as _folded,
               offdiag_update_e as _offdiag, shaders, symmetry as _symmetry,
               templates)
from . import offdiag_weld_common as _weld
from .device import MAX_BUFFER_BINDINGS, Residency, compile_source
from .launch import SUB_STEPS
from .offdiag_fused_electric_pair import (
    CALL_ARGUMENTS,
    NEIGHBOUR_NAMES,
    CURL_STORE_EDITS,
    FORWARD,
    PACKED_VECTORS,
    PACKED_VOLUMES,
    STEP_ARGUMENTS,
    STEP_POINTERS,
    _touch_kernel,
    _READ_VOLUMES,
    _WRITTEN,
)
from .. import deposit_repair as _deposit_repair

#: Forced False for the reason the unfolded twin's flag records: an off-diagonal
#: chi1inv constitutive is one of the two shapes ``deposit_repair.repairable``
#: refuses BY NAME, so no repair can bracket this seam.
CARRIES_DEPOSIT_REPAIR = False

FAMILY = "folded_offdiag_fused_electric_pair"

SLOT = "step_D"

#: FIVE driver passes, because on a folded grid the two ghost fills are LIVE.
REPLACES: Tuple[str, ...] = ("step_D", "fill_D", "zero_metal_D",
                             "fill_folded_far_ghosts_D", "update_E")

#: The three runtime reflect rows the far arm needs, in axis order. They are
#: ``d_final_c``'s OWN parameters and are never forwarded to ``step_cell``.
REFLECT_PARAMETERS = ", int reflect_x, int reflect_y, int reflect_z"
REFLECT_FIELDS = ("reflect_x", "reflect_y", "reflect_z")

#: The counts, in the same four shapes the unfolded twin measures, plus the three
#: reflect ints — which are ``Params`` members and cost NO binding.
SEPARATE_SCALAR_BINDINGS = 50
UNPACKED_POINTER_BINDINGS = 43
CPACK_ONLY_BINDINGS = 32
PACKED_BINDINGS = 24
UNPACKED_POINTERS = 42
PACKED_POINTERS = 23

if PACKED_BINDINGS > MAX_BUFFER_BINDINGS:  # pragma: no cover - a design invariant
    raise RuntimeError(
        f"the folded off-diagonal fused electric pair binds {PACKED_BINDINGS} "
        f"buffers and Metal's ceiling is {MAX_BUFFER_BINDINGS}")

__all__ = [
    "CARRIES_DEPOSIT_REPAIR", "CPACK_ONLY_BINDINGS", "FAMILY", "PACKED_BINDINGS",
    "PACKED_POINTERS", "REPLACES", "SEPARATE_SCALAR_BINDINGS", "SLOT",
    "UNPACKED_POINTERS", "UNPACKED_POINTER_BINDINGS", "certified_curl_body",
    "compile_folded_offdiag_fused_electric_pair", "folded_axis_tables",
    "folded_offdiag_fused_electric_pair_source",
    "metal_folded_offdiag_fused_electric_pair_coverage", "params_record_dtype",
    "plan_metal_folded_offdiag_fused_electric_pair",
    "refuted_cpack_only_source", "refuted_separate_scalar_source", "register_arms",
]


_TEMPLATE = r"""
#include <metal_stdlib>
using namespace metal;

__CONTRACT__

struct Params {
    uint nx; uint ny; uint nz; uint n_elem; float dtdx;
__PACK_FIELDS__
    int reflect_x; int reflect_y; int reflect_z;
};

__STEP_CELL__

__D_FINAL__

kernel void folded_offdiag_fused_electric_pair_step(
    device float*       d0s     [[buffer(0)]],
    device float*       d1s     [[buffer(1)]],
    device float*       d2s     [[buffer(2)]],
    device float*       n0s     [[buffer(3)]],
    device float*       n1s     [[buffer(4)]],
    device float*       n2s     [[buffer(5)]],
    device const float* pf0     [[buffer(6)]],
    device const float* pf1     [[buffer(7)]],
    device const float* pf2     [[buffer(8)]],
    device const float* pu0     [[buffer(9)]],
    device const float* pu1     [[buffer(10)]],
    device const float* pu2     [[buffer(11)]],
    device float*       f0      [[buffer(12)]],
    device float*       f1      [[buffer(13)]],
    device float*       f2      [[buffer(14)]],
    device float*       w0      [[buffer(15)]],
    device float*       w1      [[buffer(16)]],
    device float*       w2      [[buffer(17)]],
    device const float* g0      [[buffer(18)]],
    device const float* g1      [[buffer(19)]],
    device const float* g2      [[buffer(20)]],
    device const float* cpack   [[buffer(21)]],
    device const float* mpack   [[buffer(22)]],
    constant Params&    prm     [[buffer(23)]],
    uint idx [[thread_position_in_grid]])
{
    uint nx = prm.nx, ny = prm.ny, nz = prm.nz, n_elem = prm.n_elem;
    float dtdx = prm.dtdx;
    int reflect_x = prm.reflect_x;
    int reflect_y = prm.reflect_y;
    int reflect_z = prm.reflect_z;
__CPACK_PROLOGUE__
__MPACK_PROLOGUE__
__BODY__
}
"""

CONSTITUTIVE_DECODE_END = "    int i   = plane / nyi;\n"

#: What the seam and every redirected read pass to ``d_final_c`` beyond the curl's
#: own arguments.
REFLECT_ACTUALS = ", reflect_x, reflect_y, reflect_z"


def certified_curl_body(codes: Sequence[int],
                        contract: str = shaders.CONTRACT_OFF) -> str:
    """The FOLDED ``step_D`` curl's body as an inline function of ``(i, j, k)``."""
    source = _symmetry.folded_curl_source(
        codes, bool(SUB_STEPS["step_D"]["backward"]), contract)
    tail = _weld.lift_curl_tail(source, store_edits=CURL_STORE_EDITS)
    return _weld.step_cell_function(
        "step_cell", tail, value_type="float",
        pointer_parameters=STEP_POINTERS,
        scalar_parameters=(("float", "dtdx"),),
        returns=("v0", "v1", "v2", "n0", "n1", "n2"))


def folded_axis_tables(grid: Any) -> Tuple[Dict[int, int], Dict[int, int],
                                           Tuple[Optional[int], ...]]:
    """``(folded axis -> phase, far axis -> phase, reflect rows)`` for one grid.

    ONE DERIVATION, from ``stepping``'s own helpers, so the three cannot
    disagree: a far axis is a folded axis with ``_stored_past_owned`` true, and
    its reflect row is ``_far_reflect_rows``'. Building the far set from anything
    else — the boundary kinds alone, say — loses the full-count parity that
    decides the row.
    """
    from ..stepping import _far_reflect_rows, _stored_past_owned  # noqa: PLC0415

    folded = {axis: int(grid.mirror_phase(axis)) for axis in range(3)
              if grid.is_mirrored(axis)}
    far = {axis: phase for axis, phase in folded.items()
           if _stored_past_owned(grid, axis)}
    return folded, far, tuple(_far_reflect_rows(grid))


def folded_offdiag_fused_electric_pair_source(
        row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
        negate: Sequence[int], folded_axes: Mapping[int, int],
        far_axes: Mapping[int, int],
        contract: str = shaders.CONTRACT_OFF) -> str:
    """One specialised fused source.

    ``codes`` is the FOUR-VALUE folded quadruple :func:`.symmetry.folded_axis_kinds`
    resolves and never a hand-built triple, for the reason that emitter gives: the
    MIRROR_METALLIC / MIRROR_PERIODIC split is the folded family's single point of
    failure and backwards on one axis is a plane of wrong values.
    """
    row_mask = tuple(int(flag) for flag in row_mask)
    walls = tuple(int(flag) for flag in walls)
    if len(row_mask) != len(_offdiag.ROW_SLOTS):
        raise ValueError(f"row_mask must fill the {len(_offdiag.ROW_SLOTS)} slots "
                         f"of ROW_SLOTS, got {row_mask!r}")
    if not folded_axes:
        raise ValueError(
            "no folded axis: that configuration is offdiag_fused_electric_pair's, "
            "and emitting this source for it would overlap the two families")

    needles = _weld.displacement_needles(
        row_mask, e_terms=_offdiag.E_TERMS,
        transverse_partners=_offdiag.TRANSVERSE_PARTNERS,
        neighbour_names=NEIGHBOUR_NAMES, index=_offdiag._index,
        register="dfin{0}",
        call="d_final_{0}({1}, nxi, nyi, nzi" + STEP_ARGUMENTS + ", dtdx"
             + REFLECT_ACTUALS + ")")
    body = _weld.weld_body(
        _folded.folded_offdiag_source(row_mask, codes, walls, negate, contract),
        decode_end=CONSTITUTIVE_DECODE_END,
        seam=_weld.seam_lines(value_type="float", forward=FORWARD,
                              split_field=True,
                              d_final_arguments=REFLECT_ACTUALS),
        needles=needles)
    finals = "\n\n".join(
        _weld.d_final_function(
            component, step_name="step_cell", call_arguments=CALL_ARGUMENTS,
            value_type="float", folded_axes=folded_axes, far_axes=far_axes,
            zero_metal=walls, complex_storage=False,
            own_parameters=REFLECT_PARAMETERS)
        for component in range(3))
    return templates.substitute(_TEMPLATE, {
        "__CONTRACT__": shaders.contraction_pragma(contract),
        "__PACK_FIELDS__": (coefficient_pack.params_fields(PACKED_VECTORS)
                            + "\n" + coefficient_pack.params_fields(PACKED_VOLUMES)),
        "__STEP_CELL__": certified_curl_body(codes, contract),
        "__D_FINAL__": finals,
        "__CPACK_PROLOGUE__": coefficient_pack.prologue("cpack", PACKED_VECTORS),
        "__MPACK_PROLOGUE__": coefficient_pack.prologue("mpack", PACKED_VOLUMES),
        "__BODY__": body,
    })


def compile_folded_offdiag_fused_electric_pair(
        row_mask: Sequence[int], codes: Sequence[int], walls: Sequence[int],
        negate: Sequence[int], folded_axes: Mapping[int, int],
        far_axes: Mapping[int, int],
        contract: str = shaders.CONTRACT_OFF) -> Any:
    """The specialised entry point for one configuration."""
    return compile_source(folded_offdiag_fused_electric_pair_source(
        row_mask, codes, walls, negate, folded_axes, far_axes, contract)
    ).folded_offdiag_fused_electric_pair_step


def refuted_separate_scalar_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 50-binding all-separate signature this platform REFUSES."""
    read = _READ_VOLUMES + PACKED_VECTORS + PACKED_VOLUMES
    slots, source = _touch_kernel(
        "refuted_separate_scalar", _WRITTEN, read,
        (("constant uint&", "nx"), ("constant uint&", "ny"),
         ("constant uint&", "nz"), ("constant uint&", "n_elem"),
         ("constant float&", "dtdx"), ("constant int&", "reflect_x"),
         ("constant int&", "reflect_y"), ("constant int&", "reflect_z")),
        contract)
    assert slots == SEPARATE_SCALAR_BINDINGS, (slots, SEPARATE_SCALAR_BINDINGS)
    return source


def refuted_cpack_only_source(contract: str = shaders.CONTRACT_OFF) -> str:
    """The 32-binding ONE-PACK signature — over the ceiling BY ONE."""
    read = _READ_VOLUMES + PACKED_VOLUMES + ("cpack",)
    _slots, source = _touch_kernel(
        "refuted_cpack_only", _WRITTEN, read,
        (("constant uint&", "nx"), ("constant uint&", "ny"),
         ("constant uint&", "nz"), ("constant uint&", "n_elem"),
         ("constant float&", "dtdx")), contract)
    pointers = len(_WRITTEN) + len(read)
    assert pointers + 1 == CPACK_ONLY_BINDINGS, (pointers, CPACK_ONLY_BINDINGS)
    return source


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def metal_folded_offdiag_fused_electric_pair_coverage(
        fields: Any, pml: Any, sources: Any = None,
        residency: Any = None) -> Coverage:
    """May ONE dispatch span ``step_D`` -> both fills -> wall -> ``update_E``?"""
    reasons: List[str] = []

    curl = _symmetry.folded_composition_curl_coverage(
        fields, pml, "step_D", residency)
    if not curl.covered:
        reasons.extend(f"curl half: {reason}" for reason in curl.reasons)
    electric = _folded.folded_offdiag_constitutive_coverage(fields, pml, residency)
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

    # THIS FAMILY EXISTS FOR THE FOLD. An unfolded grid is the unfolded twin's
    # product and must be refused here rather than served by a degenerate form,
    # or the two families overlap and the disjointness sweep is right to complain.
    mirrored = getattr(grid, "is_mirrored", None)
    if not callable(mirrored) or not any(bool(mirrored(axis)) for axis in range(3)):
        reasons.append("no axis is folded: offdiag_fused_electric_pair is the "
                       "product for an unfolded off-diagonal D/E seam")

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

PARAMS_ITEMSIZE = 4 * (5 + len(PACKED_VECTORS) + len(PACKED_VOLUMES) + 3)


def params_record_dtype() -> Any:
    """The host record's dtype — scalars, pack offsets, then three reflect rows."""
    import numpy as np  # noqa: PLC0415

    names = ["nx", "ny", "nz", "n_elem", "dtdx"]
    formats = ["<u4", "<u4", "<u4", "<u4", "<f4"]
    for label in PACKED_VECTORS + PACKED_VOLUMES:
        names.append(f"off_{label}")
        formats.append("<u4")
    names.extend(REFLECT_FIELDS)
    formats.extend(["<i4", "<i4", "<i4"])
    return np.dtype({
        "names": names, "formats": formats,
        "offsets": [4 * index for index in range(len(names))],
        "itemsize": PARAMS_ITEMSIZE,
    })


def _params_tensor(shape: Sequence[int], dtdx: float, offsets: Mapping[str, int],
                   reflect: Sequence[Optional[int]], device: str) -> Any:
    """The scalars, the two packs' offsets and the three reflect rows.

    AN ABSENT REFLECT ROW IS WRITTEN ``-1`` AND IS NEVER READ: the far arm is
    emitted only for an axis in ``far_axes``, which is exactly the set with a row.
    ``-1`` rather than 0 so that a build which emitted the arm anyway indexes off
    the volume and is caught, instead of silently imaging plane zero.
    """
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    record = np.zeros(1, dtype=params_record_dtype())
    nx, ny, nz = (int(n) for n in shape)
    record["nx"], record["ny"], record["nz"] = nx, ny, nz
    record["n_elem"] = nx * ny * nz
    record["dtdx"] = np.float32(dtdx)
    for label, offset in offsets.items():
        record[f"off_{label}"] = int(offset)
    for name, row in zip(REFLECT_FIELDS, tuple(reflect)):
        record[name] = -1 if row is None else int(row)
    words = np.frombuffer(record.tobytes(), dtype=np.int32).copy()
    return torch.from_numpy(words).to(torch.device(device))


ROTATED_NAMES: Tuple[str, ...] = ("Dx", "Dy", "Dz", "fu_Dx", "fu_Dy", "fu_Dz")


def plan_metal_folded_offdiag_fused_electric_pair(
        fields: Any, pml: Any, sources: Any = None,
        residency: Optional[Residency] = None,
        contract_variants: Sequence[str] = (shaders.CONTRACT_OFF,),
        functions: Optional[Mapping[str, Any]] = None,
        ) -> Optional[_weld.ScratchWeldPairPlan]:
    """Build the folded fused off-diagonal D/E plan, or ``None`` when refused."""
    if not metal_folded_offdiag_fused_electric_pair_coverage(
            fields, pml, sources, residency).covered:
        return None
    import numpy as np  # noqa: PLC0415

    grid = fields.grid
    codes, _reasons = _symmetry.folded_axis_kinds(grid, pml)
    if codes is None:  # pragma: no cover - the predicate already refused
        return None
    weights = _folded.mirror_ghost_weights(grid)
    walls = _offdiag.wall_mask_axes(grid)
    negate = _folded.negated_axes(codes, weights)
    folded_axes, far_axes, reflect = folded_axis_tables(grid)
    rows = _offdiag.row_volumes_for(fields)
    row_mask = tuple(int(value is not None) for value in rows)

    volumes: List[str] = []

    def bind(name: str, host: Any, constant: bool = False) -> Any:
        volumes.append(name)
        return residency.mirror(name, host, constant=constant)

    twins: Dict[str, Any] = {}
    for name in ROTATED_NAMES:
        host = getattr(fields, name)
        bind(name, host)
        twin, _mirror = _weld.scratch_twin(residency, name, host)
        twins[name] = twin

    electric = ("Ex", "Ey", "Ez")
    stored = [bind(name, getattr(fields, name)) for name in electric]
    workspace = [bind("f_w_" + name, getattr(fields, "f_w_" + name))
                 for name in electric]
    magnetic = [bind(name, getattr(fields, name))
                for name in SUB_STEPS["step_D"]["sources"]]

    suffix = SUB_STEPS["step_D"]["suffix"]
    vectors = ([getattr(pml, f"{stem}_{axis}{suffix}")
                for axis in "xyz" for stem in ("kms", "sinv")]
               + [getattr(pml, f"{stem}_{axis}_h")
                  for axis in "xyz" for stem in ("kps", "kms")])
    cpack, cpack_layout = coefficient_pack.packed_mirror(
        residency, f"{FAMILY}:cpack", np, PACKED_VECTORS, vectors)

    inverse = [fields.inverse_epsilon_for(name) for name in electric]
    component_of_slot = tuple(
        next(index for index, term in enumerate(_offdiag.E_TERMS)
             if term[0] == row)
        for row, _partner in _offdiag.ROW_SLOTS)
    row_volumes = [value if value is not None else inverse[component_of_slot[slot]]
                   for slot, value in enumerate(rows)]
    mpack, mpack_layout = coefficient_pack.packed_mirror(
        residency, f"{FAMILY}:mpack", np, PACKED_VOLUMES, inverse + row_volumes)

    offsets = dict(zip(cpack_layout.names, cpack_layout.offsets))
    offsets.update(zip(mpack_layout.names, mpack_layout.offsets))

    selected = dict(functions or {})
    for mode in contract_variants:
        if mode not in selected:
            selected[mode] = compile_folded_offdiag_fused_electric_pair(
                row_mask, codes, walls, negate, folded_axes, far_axes, mode)

    dtdx = grid.dt / grid.dx
    static = (list(stored) + list(workspace) + list(magnetic)
              + [cpack, mpack,
                 _params_tensor(grid.shape, dtdx, offsets, reflect,
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
            False, (f"folded offdiag fused electric pair cannot fill {slot}",))
    return metal_folded_offdiag_fused_electric_pair_coverage(
        context.fields, context.pml, context.sources, context.residency)


def _arm_plan(context: Any, slot: str) -> Optional[_weld.ScratchWeldPairPlan]:
    if slot != SLOT:
        return None
    return plan_metal_folded_offdiag_fused_electric_pair(
        context.fields, context.pml, context.sources, context.residency,
        context.contract_variants)


def register_arms() -> Tuple[Any, ...]:
    """One row on ``step_D``, ``wired=False`` — five absorbed passes declared."""
    from . import arms  # noqa: PLC0415
    return (arms.register(FAMILY, SLOT,
                          "folded off-diagonal fused electric D/E pair",
                          _arm_coverage, _arm_plan,
                          prefix="folded off-diagonal fused electric D/E pair: ",
                          noun="fused folded D-curl/off-diagonal-E pair",
                          wired=False, replaces=REPLACES),)


ARMS = register_arms()
