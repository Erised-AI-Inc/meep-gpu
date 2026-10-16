#!/usr/bin/env python3
"""Native MPS byte gate for folded tensor-row dispersive ``update_E``.

RULING R7: SOFTWARE BINARY64. The kernel forms the off-diagonal single's double row
over the ordered ``D - P`` gathers (``offdiag_update_e._row_assembly``, through
``metal_kernels.softfp64``) with the six per-chunk presence words in ``Params``, and
its arm is wired. Every plan is the ENGINE builder's
(``plan_folded_offdiag_dispersive``) with the record of the division the oracle steps
with. The gate runs:

* ``rows`` -- every product case on a GENERIC state (where a float32 row would
  differ from the array path's double row) against ``stepping.update_E`` over
  ``--cycles`` launches, ``f_w`` and every pole compared;
* ``b_select`` -- the per-chunk constitutive tail and the fold on EXACT-ROW states
  (dyadic words on which every float32 operation of the row, the ``D - P`` chain
  included, is exact, so a differing word is the tail's or the fold's; the tail's
  ``kps*src`` and ``kms*prev`` are not exact, so its grouping and form stay
  byte-visible). Each case records the cells of each constitutive form per component;
* ``mutations`` -- on the exact states: the previous accumulation grouping planted
  (must-catch, KERNELS.md §7.1), the select removed, the auxiliary stored on a plain
  cell, the integer lattice read, and the pole and fold defects; on a generic state
  the float32 row (ruling R7's must-catch, the pre-R7 spelling); on a planted
  cancelling-terms state the terms summed before the diagonal (must-catch, a
  different double rounding visible only there);
* ``presence`` -- rows zero in the PML chunks of the metallic axes with
  ``Dx - P = -0``: the words are not all-kept and expand back to the array path's
  presence, the shipped kernel is 0 words from the oracle, which keeps ``-0`` rows,
  and the presence ignored (source), cleared and forced all-kept (``Params``) are
  caught;
* ``engine`` -- every product case: the arm is wired and offered, the predicate
  admits with no ruling clause, and the builder returns a plan whose words are
  ``offdiag_update_e.presence_for``'s.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import DRUDE, LORENTZIAN, PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.metal_kernels import folded_offdiag_dispersive_update_e as family  # noqa: E402
from meep_gpu.metal_kernels import offdiag_update_e as offdiag  # noqa: E402
from meep_gpu.metal_kernels import arms  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal, symmetry  # noqa: E402
from meep_gpu.metal_kernels.device import Residency, compile_source, metal_frontend_version  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

from gate_chunk_records import region_plan_record  # noqa: E402

#: The states a case can be built on (:func:`_build`).
STATES = ("exact", "generic", "cancelling", "presence")


def _words(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value).reshape(-1).view(np.uint32)


def _different(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


def _build(*, cell: Sequence[float], axes: str, phases: Sequence[int], boundaries: Any,
           rows: Mapping[str, Sequence[str]], counts: Sequence[int], kind: str,
           seed: int, state: str = "exact") -> Tuple[Fields, PML]:
    dimensions = 2 if float(cell[2]) == 0.0 else 3
    grid = Grid(resolution=8.0, cell_size=tuple(cell), dimensions=dimensions,
                courant=0.35, boundaries=boundaries,
                symmetry=tuple(Mirror(axis, int(phase))
                               for axis, phase in zip(axes, phases)), xp=np)
    fields = Fields(grid=grid)
    shape = grid.shape
    rng = np.random.default_rng(seed)
    epsilon = {name: np.full(shape, value, np.float32)
               for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))}
    inverse = {name: (np.float32(1.0) / value).astype(np.float32)
               for name, value in epsilon.items()}
    if state == "presence":
        tensor_rows = _presence_rows(shape, boundaries, rows, rng)
    else:
        tensor_rows = {row: {partner: rng.uniform(-0.18, 0.18, shape).astype(np.float32)
                             for partner in partners}
                       for row, partners in rows.items()}
    if state == "cancelling":
        tensor_rows.update(_cancelling_rows(shape))
    fields.set_epsilon_volumes(epsilon, inverse, chi1inv_offdiagonal=tensor_rows)
    for index in range(max(counts)):
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.8 + index * 0.1, 0.06 + index * 0.01, kind),
            {name: 0.18 + index * 0.025 if counts[axis] > index else 0.0
             for axis, name in enumerate(("Ex", "Ey", "Ez"))}, grid, np.float32))
    fields.enable_pml_storage()
    folded = {"XYZ".index(axis) for axis in axes}
    thickness = []
    for axis, size in enumerate(shape):
        if size < 6:
            thickness.append((0, 0))
        elif axis in folded:
            thickness.append((0, 2))
        else:
            thickness.append((2, 2))
    pml = PML(grid=grid, thickness=tuple(thickness))
    # The region plan a driver installs with the layer: the one-rank division.
    stepping._region_plan(fields, pml)
    if state == "exact":
        _exact_state(fields, rng)
    else:
        _generic_state(fields, rng)
    if state == "cancelling":
        _plant_cancelling_terms(fields, seed)
    elif state == "presence":
        _plant_negative_zero_rows(fields, rows, rng)
    return fields, pml


def _generic_state(fields: Fields, rng: Any) -> None:
    """Generic float32 words (24-bit mantissas): a float32 row differs from the double
    one on about one word in sixty here."""
    shape = fields.grid.shape
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = rng.uniform(-0.4, 0.4, shape).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            for table in (state.P, state.P_prev):
                table[component][...] = rng.uniform(-0.2, 0.2, shape).astype(np.float32)


#: The term magnitude of :func:`_plant_cancelling_terms`: an exact power of two, so
#: each term's float32 pair sum ``(1 + 1)*r + (1 + 1)*r`` and its double quarter are
#: exact and the two terms cancel to an exact zero (gate_metal_offdiag's value).
CANCELLING_TERM = 2.0 ** -5


def _cancelling_rows(shape: Sequence[int]) -> Dict[str, Dict[str, Any]]:
    """Ex's two row slots, ``+r`` and ``-r`` on every cell (the case's other rows
    stay as built, so the row mask is the case's own)."""
    value = np.float32(CANCELLING_TERM)
    return {"Ex": {"Ey": np.full(shape, value, dtype=np.float32),
                   "Ez": np.full(shape, -value, dtype=np.float32)}}


def _plant_cancelling_terms(fields: Fields, seed: int) -> None:
    """The state on which the row's association is byte-visible.

    Every pole word zero, so ``D - P`` is ``D`` exactly; ``Dy = Dz = 1``, so off the
    fold plane Ex's term1 is ``+r`` and term2 ``-r`` exactly; ``Dx`` near ``2**-40``
    (normal, a random 24-bit mantissa), so ``gs*us`` sits 35 binary exponents below
    the terms. ``(gs*us + term1) + term2`` rounds ``gs*us`` to the terms' binary64
    grid and ``gs*us + (term1 + term2)`` keeps it whole.
    """
    rng = np.random.default_rng(seed)
    shape = fields.Dx.shape
    for state in fields.polarizations:
        for component in state.driven():
            for table in (state.P, state.P_prev):
                table[component][...] = np.float32(0.0)
    fields.Dy[...] = np.float32(1.0)
    fields.Dz[...] = np.float32(1.0)
    mantissa = rng.uniform(1.0, 2.0, size=shape).astype(np.float32)
    fields.Dx[...] = (mantissa * np.float32(2.0 ** -40)).astype(np.float32)


def _presence_rows(shape: Sequence[int], boundaries: Any,
                   rows: Mapping[str, Sequence[str]], rng: Any) -> Dict[str, Dict[str, Any]]:
    """Rows zero outside the interior of every metallic axis: MEEP deletes them in the
    PML chunks there (the record's labels name those chunks one to one)."""
    kinds = boundaries if isinstance(boundaries, dict) else {}
    keep = np.zeros(shape, dtype=bool)
    keep[tuple(slice(3, -3) if kinds.get("xyz"[axis]) == "metallic" and n > 8
               else slice(None) for axis, n in enumerate(shape))] = True
    return {row: {partner: np.where(keep, rng.uniform(0.05, 0.2, shape), 0.0
                                    ).astype(np.float32) for partner in partners}
            for row, partners in rows.items()}


def _plant_negative_zero_rows(fields: Fields, rows: Mapping[str, Sequence[str]],
                              rng: Any) -> None:
    """``D - P = -0`` on every row component (its P words +0) and positive partner
    gathers, so a term the chunk dropped keeps a ``-0`` row where adding its ``+0``
    would give ``+0``."""
    shape = fields.grid.shape
    row_components = set(rows)
    for state in fields.polarizations:
        for component in state.driven():
            if component in row_components:
                state.P[component][...] = np.float32(0.0)
            else:
                state.P[component][...] = rng.uniform(-0.005, 0.0, shape).astype(np.float32)
    for component in ("Ex", "Ey", "Ez"):
        array = getattr(fields, "D" + component[1])
        if component in row_components:
            array[...] = np.float32(-0.0)
        else:
            array[...] = (np.abs(array) + np.float32(0.01)).astype(np.float32)


def _exact_state(fields: Fields, rng: Any) -> None:
    """Dyadic words on which every float32 operation of the row is exact.

    D, E, f_w and every P and P_prev at most seven significant bits (k/64, |k| <= 64),
    inverse epsilon four (k/8, 4 <= k <= 8), every row coefficient four and never zero
    (sign * k/64, 1 <= k <= 8; no chunk drops a term): ``D - P0 - ...``, each product
    and both sums stay below 24 bits, so the double row's one rounding is the identity
    and the float32 row is the same word (gate_metal_offdiag.exact_state, with the
    poles added).
    """
    shape = fields.grid.shape
    for name in ("Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"):
        getattr(fields, name)[...] = (rng.integers(-64, 65, size=shape) / 64.0
                                      ).astype(np.float32)
    seen = set()
    for component in ("Ex", "Ey", "Ez"):
        inverse = fields.inverse_epsilon_for(component)
        if id(inverse) in seen:
            continue
        seen.add(id(inverse))
        inverse[...] = (rng.integers(4, 9, size=shape) / 8.0).astype(np.float32)
    for row, partner in family.ROW_SLOTS:
        value = (fields.chi1inv_offdiagonal_for(row) or {}).get(partner)
        if value is None:
            continue
        sign = np.where(rng.integers(0, 2, size=shape) == 0, 1.0, -1.0)
        value[...] = (sign * rng.integers(1, 9, size=shape) / 64.0).astype(np.float32)
    for state in fields.polarizations:
        for component in state.driven():
            for table in (state.P, state.P_prev):
                table[component][...] = (rng.integers(-16, 17, size=shape) / 64.0
                                         ).astype(np.float32)


def _names(fields: Fields) -> Tuple[str, ...]:
    names = ["Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"]
    for index, state in enumerate(fields.polarizations):
        for component in state.driven():
            names.extend((f"P[{index}].{component}", f"P_prev[{index}].{component}"))
    return tuple(names)


def _array(fields: Fields, name: str) -> Any:
    if name in {"Ex", "Ey", "Ez", "f_w_Ex", "f_w_Ey", "f_w_Ez"}:
        return getattr(fields, name)
    head, component = name.split(".", 1)
    state = fields.polarizations[int(head[head.index("[") + 1:head.index("]")])]
    return state.P_prev[component] if head.startswith("P_prev") else state.P[component]


def _advance_poles(fields: Fields) -> None:
    for state in fields.polarizations:
        state.update(fields.drive_field, fields.grid.dt)


def _forms(fields: Fields, pml: PML) -> Dict[str, Any]:
    """Per component: (accumulating cells, plain cells) of the region plan stepped."""
    plan = stepping._region_plan(fields, pml)
    out: Dict[str, Any] = {}
    for component in ("Ex", "Ey", "Ez"):
        mask = np.broadcast_to(plan.mask("constitutive", component, np), fields.grid.shape)
        inside = int(np.count_nonzero(mask))
        out[component] = {"mode": plan.spec("constitutive", component).mode,
                          "accumulate": inside, "plain": int(mask.size) - inside}
    return out


def _rebind_presence(plan: Any, words: Sequence[int]) -> None:
    """The plan's ``Params`` re-packed with other presence words (a harness-side
    mutant: the words the host binds, not the source)."""
    import torch  # noqa: PLC0415

    packed = family.params_words(plan.shape, plan.chunk_bounds, words)
    plan._params.copy_(torch.from_numpy(packed.copy()).to(plan._params.device))


def _run(*, cell: Sequence[float], axes: str, phases: Sequence[int], boundaries: Any,
         rows: Mapping[str, Sequence[str]], counts: Sequence[int], kind: str,
         seed: int, functions: Mapping[str, Any] | None = None, state: str = "exact",
         cycles: int = 1, presence: Sequence[int] | None = None) -> Dict[str, Any]:
    """``cycles`` update_E launches on one state, the kernel against
    ``stepping.update_E``; ``presence`` replaces the words the plan binds."""
    reference, reference_pml = _build(cell=cell, axes=axes, phases=phases,
                                      boundaries=boundaries, rows=rows, counts=counts,
                                      kind=kind, seed=seed, state=state)
    actual, actual_pml = _build(cell=cell, axes=axes, phases=phases,
                                boundaries=boundaries, rows=rows, counts=counts,
                                kind=kind, seed=seed, state=state)
    names = _names(actual)
    before = {name: np.array(_array(actual, name), copy=True) for name in names}
    residency = Residency()
    plan = family.plan_folded_offdiag_dispersive(
        actual, actual_pml, residency, functions=functions,
        chunk_bounds=region_plan_record(actual))
    if plan is None:
        return {"passed": False, "reason": "the declared product was refused"}
    words = tuple(plan.presence)
    if presence is not None:
        _rebind_presence(plan, presence)
    residency.sync_in()
    per_cycle = []
    for cycle in range(1, cycles + 1):
        stepping.update_E(reference, reference_pml)
        plan.run()
        residency.sync_out()
        difference = sum(_different(_array(reference, name), _array(actual, name))
                         for name in names)
        census = sum(subnormal.census(_array(reference, name)) for name in names)
        per_cycle.append({"cycle": cycle, "differing_words": difference,
                          "reference_subnormals": census})
        if difference or census:
            break
    difference = per_cycle[-1]["differing_words"]
    census = per_cycle[-1]["reference_subnormals"]
    moved = sum(_different(before[name], _array(actual, name)) for name in names)
    fw_moved = [_different(before[name], _array(actual, name))
                for name in ("f_w_Ex", "f_w_Ey", "f_w_Ez")]
    forms = _forms(actual, actual_pml)
    split_ok = all(row["accumulate"] and row["plain"] for row in forms.values()
                   if row["mode"] not in ("outside_only", "inside_only"))
    compared = sum(int(np.asarray(_array(actual, name)).size) for name in names)
    negative_zero = sum(int(np.count_nonzero(_words(_array(reference, name))
                                             == np.uint32(0x80000000)))
                        for name in ("Ex", "Ey", "Ez"))
    return {"passed": (len(per_cycle) == cycles
                       and all(row["differing_words"] == row["reference_subnormals"] == 0
                               for row in per_cycle)
                       and moved > 0 and plan.runs == cycles and split_ok),
            "per_cycle": per_cycle,
            "differing_words": difference, "compared_words": compared,
            "reference_subnormals": census, "moved_words": moved,
            "f_w_moved": fw_moved, "forms": forms, "presence_words": list(words),
            "oracle_negative_zero_e_words": negative_zero,
            "launches": plan.launches, "counts": plan.counts,
            "stored_shape": plan.shape, "mirrors": len(residency.names)}


def _planted(source: str, needle: str, replacement: str) -> str:
    if source.count(needle) != 1:
        raise RuntimeError(f"mutation needle {needle!r} is not unique in the source")
    return source.replace(needle, replacement, 1)


_ACCUMULATE0 = "    if (dw0) { a0 = f0[ii] + ((kp_0 * src0) - (km_0 * prev0)); }"


def _drop_ex_second_pole(source: str) -> str:
    """Ex's second pole dropped inside its own ``D - P`` helper (``dmp0``) only."""
    first, rest = source.split("static inline float dmp1(", 1)
    return (_planted(first, "value = value - p[1u * n_elem + uint(index)];",
                     "// dropped pole") + "static inline float dmp1(" + rest)


#: Component 0's two-term double row, as the shipped source spells it.
_ROW0 = ("    ulong od_row0 = sd_widen_f(gs0 * us0, true);\n"
         "    if (od_kept_00) { od_row0 = sd_add(od_row0, sd_scale_pow2(sd_widen_f("
         "sum_00, true), -2, true), true); }\n"
         "    if (od_kept_01) { od_row0 = sd_add(od_row0, sd_scale_pow2(sd_widen_f("
         "sum_01, true), -2, true), true); }\n")


def _float32_row(source: str) -> str:
    """RULING R7'S MUTATION: component 0's row in float32, the pre-R7 spelling --
    ``0.25f`` times each pair sum, the terms summed and added to ``gs*us`` in float32."""
    return _planted(source, _ROW0 + "    float src0 = sd_narrow_f(od_row0, true);",
                    "    float total0 = (0.25f * sum_00) + (0.25f * sum_01);\n"
                    "    float src0 = (gs0 * us0) + total0;")


def _terms_grouped_first(source: str) -> str:
    """Component 0's double row with its two terms summed before the diagonal."""
    return _planted(source, _ROW0,
                    "    ulong od_t0 = sd_add(sd_scale_pow2(sd_widen_f(sum_00, true), -2, "
                    "true), sd_scale_pow2(sd_widen_f(sum_01, true), -2, true), true);\n"
                    "    ulong od_row0 = sd_add(sd_widen_f(gs0 * us0, true), od_t0, true);\n")


def _presence_ignored(source: str, row_mask: Sequence[int]) -> str:
    """Every live slot's presence flag forced true in the source."""
    for component in range(3):
        for offset in (0, 1):
            slot = 2 * component + offset
            if row_mask[slot]:
                source = _planted(
                    source,
                    f"    const bool od_kept_{component}{offset} = "
                    f"((kw.{offdiag.PRESENCE_MEMBERS[slot]} >> od_chunk{component})"
                    f" & 1u) != 0u;",
                    f"    const bool od_kept_{component}{offset} = true;")
    return source


def _mutant_sources(counts: Sequence[int]) -> Dict[str, Tuple[str, str, str]]:
    """label -> (expected, state, source)."""
    kwargs = dict(row_mask=(1, 1, 1, 0, 0, 0), codes=(0, 3, 0), walls=(0, 0, 0),
                  negate=(0, 1, 0), counts=counts)
    base = family.folded_offdiag_dispersive_source(**kwargs)
    head, tail = base.split("// chunk bounds: begin", 1)
    exact = {
        "second_pole_dropped": ("caught", _drop_ex_second_pole(base)),
        "partner_pole_dropped": ("caught", _planted(
            base, "dmp1(g1, p1, ii, n_elem) + (at_y", "g1[ii] + (at_y")),
        "mirror_ghost_sign_dropped": ("caught", _planted(
            base, "at_y ? -dn_00 : dn_00", "dn_00")),
        # §7.1, inverted: the package's previous grouping, (f + kps*src) - kms*prev.
        "previous_accumulation_grouping": ("caught", _planted(
            base, _ACCUMULATE0,
            "    if (dw0) { a0 = (f0[ii] + (kp_0 * src0)) - (km_0 * prev0); }")),
        "select_removed": ("caught", _planted(
            _planted(_planted(base, "    float prev0 = dw0 ? w0[ii] : 0.0f;",
                              "    float prev0 = w0[ii];"),
                     "    if (dw0) { w0[ii] = src0; }", "    w0[ii] = src0;"),
            _ACCUMULATE0, "    a0 = f0[ii] + ((kp_0 * src0) - (km_0 * prev0));")),
        "auxiliary_stored_on_a_plain_cell": ("caught", _planted(
            base, "    if (dw0) { w0[ii] = src0; }", "    w0[ii] = src0;")),
        "half_integer_lines_read_on_the_integer_lattice": ("caught", head
            + "// chunk bounds: begin" + tail.replace("cbh_", "cbi_").replace("in_h_", "in_i_")),
        "auxiliary_load_unconditional": ("null", _planted(
            base, "    float prev0 = dw0 ? w0[ii] : 0.0f;", "    float prev0 = w0[ii];")),
    }
    out = {label: (expected, "exact", source)
           for label, (expected, source) in exact.items()}
    out["float32_row"] = ("caught", "generic", _float32_row(base))
    out["terms_grouped_first"] = ("caught", "cancelling", _terms_grouped_first(base))
    return out


def _mutants(counts: Sequence[int]) -> Dict[str, Tuple[str, str, Dict[str, Any]]]:
    return {label: (expected, state, {shaders.CONTRACT_OFF: compile_source(source)
                                      .folded_offdiag_dispersive_step})
            for label, (expected, state, source) in _mutant_sources(counts).items()}


def _engine_leg(products: Sequence[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
    """The engine route admits every case with no ruling clause, the arm is wired and
    offered, and the builder's words are ``offdiag_update_e.presence_for``'s."""
    arm = next(spec for spec in arms.registered("update_E") if spec.family == family.FAMILY)
    rows_out = []
    for label, cell, axes, phases, boundaries, rows, counts, kind in products:
        fields, pml = _build(cell=cell, axes=axes, phases=phases, boundaries=boundaries,
                             rows=rows, counts=counts, kind=kind, seed=35300,
                             state="generic")
        residency = Residency()
        record = region_plan_record(fields)
        verdict = family.folded_offdiag_dispersive_coverage(fields, pml, residency,
                                                            record)
        plan = family.plan_folded_offdiag_dispersive(
            fields, pml, residency, chunk_bounds=record)
        codes, _notes = symmetry.folded_axis_kinds(fields.grid, pml)
        words = offdiag.presence_for(fields, pml, codes, record)
        context = arms.StepContext(fields, pml, residency, (shaders.CONTRACT_OFF,),
                                   chunk_bounds=record)
        offered = any(bound.label == family.LABEL and bound.gate
                      for bound in arms.arms_for("update_E", context))
        rows_out.append({"case": label, "covered": verdict.covered,
                         "reasons": list(verdict.reasons),
                         "plan_built": plan is not None, "arm_wired": arm.wired,
                         "arm_offered": offered,
                         "plan_words_are_presence_for": (
                             plan is not None and tuple(plan.presence) == tuple(words)),
                         "passed": (verdict.covered and plan is not None and arm.wired
                                    and offered and tuple(plan.presence) == tuple(words))})
    return rows_out


#: The presence cases: rows zero in the PML chunks of the metallic axes,
#: ``Dx - P = -0``.
PRESENCE_CASES = (
    ("metallic_xz_fold_y_ex_two_terms", (1.6, 1.2, 1.6), "Y", (1,),
     {"x": "metallic", "y": "periodic", "z": "metallic"}, {"Ex": ("Ey", "Ez")},
     (1, 1, 1), LORENTZIAN),
    ("metallic_xz_fold_y_odd_ez_ey", (1.6, 1.2, 1.6), "Y", (-1,),
     {"x": "metallic", "y": "periodic", "z": "metallic"}, {"Ez": ("Ey",)},
     (2, 1, 2), DRUDE),
)


def _presence_leg() -> List[Dict[str, Any]]:
    out = []
    for index, (label, cell, axes, phases, boundaries, rows, counts, kind) in enumerate(
            PRESENCE_CASES):
        spec = dict(cell=cell, axes=axes, phases=phases, boundaries=boundaries,
                    rows=rows, counts=counts, kind=kind, seed=35400 + index,
                    state="presence")
        shipped = _run(**spec)
        words = shipped.get("presence_words", [])
        fields, pml = _build(**spec)
        codes, _notes = symmetry.folded_axis_kinds(fields.grid, pml)
        record = region_plan_record(fields)
        from meep_gpu.fields import IYEE_SHIFTS  # noqa: PLC0415
        expands = []
        slots = offdiag.row_volumes_for(fields)
        layout = stepping._chunk_layout(fields, pml)
        for slot, (value, word) in enumerate(zip(slots, words)):
            if value is None:
                continue
            row, partner = offdiag.ROW_SLOTS[slot]
            present = fields.offdiag_presence(row, partner, layout)
            chunk_codes = offdiag.cell_chunk_codes(record, IYEE_SHIFTS[row],
                                                   tuple(fields.grid.shape), codes)
            expands.append(bool(present is not None and np.array_equal(
                offdiag.expand_presence_word(word, chunk_codes),
                np.broadcast_to(np.asarray(present, dtype=bool), chunk_codes.shape))))
        row_mask = tuple(int(value is not None) for value in slots)
        live = [word for value, word in zip(slots, words) if value is not None]
        source = _case_source(fields, pml)
        ignored = _run(**spec, functions={shaders.CONTRACT_OFF: compile_source(
            _presence_ignored(source, row_mask)).folded_offdiag_dispersive_step})
        cleared = _run(**spec, presence=(0,) * 6)
        all_kept = _run(**spec, presence=offdiag.ALL_KEPT)
        out.append({
            "case": label, "words": [int(w) for w in words],
            "live_words_not_all_kept": bool(live) and all(
                w != offdiag.ALL_CHUNKS_KEPT for w in live),
            "words_expand_to_the_array_paths_presence": expands,
            "differing_words": shipped.get("differing_words"),
            "oracle_negative_zero_e_words": shipped.get("oracle_negative_zero_e_words"),
            "presence_ignored_differing": ignored.get("differing_words"),
            "presence_cleared_differing": cleared.get("differing_words"),
            "presence_all_kept_differing": all_kept.get("differing_words"),
            "launches": [shipped.get("launches"), ignored.get("launches"),
                         cleared.get("launches"), all_kept.get("launches")],
            "passed": bool(shipped.get("passed") and live and all(
                w != offdiag.ALL_CHUNKS_KEPT for w in live) and expands and all(expands)
                and shipped.get("oracle_negative_zero_e_words", 0) > 0
                and ignored.get("differing_words", 0) > 0
                and cleared.get("differing_words", 0) > 0
                and all_kept.get("differing_words", 0) > 0
                and ignored.get("launches") == cleared.get("launches")
                == all_kept.get("launches") == 1),
        })
    return out


def _case_source(fields: Fields, pml: PML) -> str:
    """The SHIPPED source of one case's own specialisation, read off the engine's plan
    (row mask, codes, walls, parity, pole counts), compiled nothing."""
    plan = family.plan_folded_offdiag_dispersive(
        fields, pml, Residency(), contract_variants=(),
        chunk_bounds=region_plan_record(fields))
    assert plan is not None, "a declared folded tensor dispersive product was refused"
    return family.folded_offdiag_dispersive_source(
        plan.row_mask, plan.codes, plan.walls, plan.negate, plan.counts)


def _emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(f"{row['index']}/{row['total']} {row['label']}: "
          f"diff={row.get('differing_words', '-')}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    if args.cycles < 1:
        raise SystemExit("--cycles must be positive")
    import torch
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")

    products = (
        ("periodic_even_all_rows", (2.0, 1.5, 1.25), "Y", (1,), "periodic",
         {"Ex": ("Ey", "Ez"), "Ey": ("Ez",)}, (2, 1, 2), LORENTZIAN),
        ("metallic_odd", (2.0, 1.5, 1.25), "Y", (-1,), {"y": "metallic"},
         {"Ex": ("Ey", "Ez"), "Ez": ("Ex",)}, (2, 2, 1), DRUDE),
        ("two_axis_fold", (2.0, 1.5, 1.25), "XY", (1, -1), "periodic",
         {"Ex": ("Ey",), "Ey": ("Ex", "Ez")}, (2, 1, 2), LORENTZIAN),
        ("sparse_row_control", (2.0, 1.5, 1.25), "Y", (1,), "periodic",
         {"Ex": ("Ey",)}, (2, 1, 2), LORENTZIAN),
        # The signature's reason for existing: eight poles on EACH component still
        # fit only because each component is packed into one persistent buffer.
        ("max_eight_poles_per_component", (2.0, 1.5, 1.25), "Y", (1,), "periodic",
         {"Ex": ("Ey", "Ez"), "Ey": ("Ez", "Ex"), "Ez": ("Ex", "Ey")},
         (8, 8, 8), LORENTZIAN),
    )
    mutation_rows = {"Ex": ("Ey", "Ez"), "Ey": ("Ez",)}
    mutation_counts = (2, 1, 2)
    mutations = _mutants(mutation_counts)
    total = 2 * len(products) + len(mutations) + 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()
    report = []
    with jsonl.open("w", encoding="utf-8") as handle:
        index = 0
        for label, cell, axes, phases, boundaries, rows, counts, kind in products:
            index += 1
            row = {"index": index, "total": total, "leg": "rows", "label": label,
                   "cell": cell, "axes": axes, "phases": phases, "boundaries": boundaries,
                   "rows": rows, "counts": counts, "kind": kind, "cycles": args.cycles,
                   "generic_state": True,
                   **_run(cell=cell, axes=axes, phases=phases, boundaries=boundaries,
                          rows=rows, counts=counts, kind=kind, seed=35000 + index,
                          state="generic", cycles=args.cycles)}
            report.append(row)
            _emit(handle, row)
        for label, cell, axes, phases, boundaries, rows, counts, kind in products:
            index += 1
            row = {"index": index, "total": total, "leg": "b_select", "label": label,
                   "cell": cell, "axes": axes, "phases": phases, "boundaries": boundaries,
                   "rows": rows, "counts": counts, "kind": kind,
                   **_run(cell=cell, axes=axes, phases=phases, boundaries=boundaries,
                          rows=rows, counts=counts, kind=kind, seed=35100 + index)}
            report.append(row)
            _emit(handle, row)
        for label, (expected, state, functions) in mutations.items():
            index += 1
            spec = dict(cell=(2.0, 1.5, 1.25), axes="Y", phases=(1,),
                        boundaries="periodic", rows=mutation_rows,
                        counts=mutation_counts, kind=LORENTZIAN, seed=35200 + index,
                        state=state)
            control = _run(**spec)
            result = _run(**spec, functions=functions)
            difference = int(result.get("differing_words", 0))
            caught = difference > 0
            row = {"index": index, "total": total, "leg": "mutation", "label": label,
                   "state": state, "expected": expected, "differing_words": difference,
                   "control_differing_words": control.get("differing_words"),
                   "launches": result.get("launches"), "caught": caught,
                   "passed": (result.get("launches") == 1 and bool(control.get("passed"))
                              and caught == (expected == "caught"))}
            report.append(row)
            _emit(handle, row)
        index += 1
        presence_rows = _presence_leg()
        row = {"index": index, "total": total, "leg": "presence", "label": "presence",
               "cases": presence_rows,
               "passed": all(item["passed"] for item in presence_rows)}
        report.append(row)
        _emit(handle, row)
        index += 1
        engine_rows = _engine_leg(products)
        row = {"index": index, "total": total, "leg": "engine", "label": "engine",
               "cases": engine_rows, "passed": all(item["passed"] for item in engine_rows)}
        report.append(row)
        _emit(handle, row)
    result = {"verdict": "PASS" if all(row["passed"] for row in report) else "FAIL",
              "elapsed_seconds": time.perf_counter() - started,
              "rows_rows": len(products), "b_select_rows": len(products),
              "mutation_rows": len(mutations), "presence_cases": len(PRESENCE_CASES),
              "rows": report, "torch_version": torch.__version__,
              "metal_frontend": metal_frontend_version(), "jsonl": str(jsonl),
              "source_sha256": {"family": hashlib.sha256(Path(family.__file__).read_bytes()).hexdigest(),
                                "gate": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; artifact {args.out}",
          flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
