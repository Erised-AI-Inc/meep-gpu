#!/usr/bin/env python3
"""Native MPS byte gate for complex/Bloch conductive PML curls.

This gate runs the engine's NumPy ``stepping.step_B``/``step_D`` oracle beside
the new 30-buffer Metal plans.  It compares float32 words, including both planes
of every complex64 field and both PML histories.  Each case is appended and
flushed to JSONL before the following MPS launch, so a stopped run retains its
completed evidence.

The gate is intentionally narrow: it certifies this one family, not a complete
driver step or production dispatch.  It also runs only under the explicit MPS
``flush`` subnormal policy; every product records a zero subnormal census.

THE PER-CHUNK FORMS (bit_parity_0_10, KERNELS.md §7, steps M4/M5). Each plan binds
the chunk record of the division the reference steps with (``gate_chunk_records.
layer_record``), and a target's case is its chunk's: MEEP's four conductive forms on a
lossy target, ``f_cond`` only on dsig cells and ``fu`` only on dsigu cells, and a
non-conductive target in the ordinary complex forms. Each product records its form
census (every conductive form and both lossless forms, on the record it binds), and a
form no product reaches fails the gate, because a mutation of that form would be
vacuous.

The mutations follow. The grouping mutations are INVERTED (§7.1): the shipped stencil
and the shipped ``f_cond`` increment ``(fcnd' - fcnd)``, formed first, are MEEP's
groupings, so ``curl_previous_grouping`` and ``increment_previous_grouping`` plant the
package's previous groupings and must be caught. The §7.2 battery is armed on the
conductive and lossless tails (``b*``); the record mutations (``b3``, ``b4``) hand the
shipped kernel a wrong record while the reference steps the true division; the one
declared null (``b7``, an ``f_cond`` load hoisted above the select with every store
still guarded) must change no word. No planted signed-zero leg: every form here is
arithmetic on its own inputs, and the plain output selects are the complex base gate's
(``gate_metal_complex.py``). The refusal leg holds a chunk that keeps no conductivity
(MEEP steps the lossless forms there, and the record carries no per-chunk presence) on
each side, refused by name to the array path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.chunk_map import (  # noqa: E402
    ChunkLayout,
    membership_from_record,
    record_index,
)
from meep_gpu.fields import IYEE_SHIFTS, Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import complex_conductive_pml as family  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency,
    compile_source,
    metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402

from gate_chunk_records import layer_record  # noqa: E402


def words(value: Any) -> np.ndarray:
    """All uint32 words of a float32 or complex64 volume."""
    array = np.ascontiguousarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.complex64)):
        raise TypeError(f"gate expected float32/complex64, got {array.dtype}")
    return array.reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def build(*, sides: Sequence[str], mixed: bool, seed: int,
          phase: Tuple[float, float, float] = (0.2, -0.125, 0.0),
          boundaries: Sequence[str] = ("periodic", "periodic", "periodic"),
          ) -> Tuple[Fields, PML]:
    """A live complex PML engine state with nontrivial curl/history values."""
    grid = Grid(resolution=8.0, cell_size=(1.0, 0.875, 0.75), courant=0.3,
                boundaries=tuple(boundaries), k_point=phase, xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.set_background_eps(2.25)
    pml = PML(grid=grid, thickness=1)
    fields.enable_pml_storage()
    sigma = np.linspace(np.float32(0.06), np.float32(0.22), grid.shape[0],
                        dtype=np.float32)[:, None, None]
    sigma = np.broadcast_to(sigma, grid.shape).copy()
    if "B" in sides:
        fields.set_b_conductivity(
            {"Bx": sigma if mixed else sigma,
             "By": None if mixed else sigma,
             "Bz": None if mixed else sigma})
    if "D" in sides:
        fields.set_d_conductivity(
            {"Dx": None if mixed else sigma,
             "Dy": None if mixed else sigma,
             "Dz": sigma})
    rng = np.random.default_rng(seed)
    names = (
        "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
        "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
        "f_cond_Bx", "f_cond_By", "f_cond_Bz", "f_cond_Dx", "f_cond_Dy", "f_cond_Dz",
    )
    for name in names:
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = (rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32)
                          + 1j * rng.uniform(-0.3, 0.3, grid.shape).astype(np.float32))
    return fields, pml


def compared_names(fields: Fields) -> Tuple[str, ...]:
    names = [
        "Bx", "By", "Bz", "Dx", "Dy", "Dz",
        "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
    ]
    names.extend(
        "f_cond_" + target for target in ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
        if getattr(fields, "f_cond_" + target) is not None)
    return tuple(names)


def subnormal_words(fields: Fields, names: Iterable[str]) -> int:
    return sum(subnormal.census(getattr(fields, name)) for name in names)


def execute_case(*, sides: Sequence[str], mixed: bool, cycles: int, seed: int,
                 b_functions: Optional[Mapping[str, Any]] = None,
                 phase: Tuple[float, float, float] = (0.2, -0.125, 0.0),
                 boundaries: Sequence[str] = ("periodic", "periodic", "periodic"),
                 record: Any = None,
                 ) -> Dict[str, Any]:
    """Run the complete B/D curl pair against the engine oracle.

    ``record`` is the chunk record both plans bind; ``None`` is the true one-rank
    record of the grid (the division the reference steps with). A record mutation
    hands in a wrong one and the reference stays on the truth.
    """
    reference, reference_pml = build(
        sides=sides, mixed=mixed, seed=seed, phase=phase, boundaries=boundaries)
    actual, actual_pml = build(
        sides=sides, mixed=mixed, seed=seed, phase=phase, boundaries=boundaries)
    names = compared_names(actual)
    before = {name: np.array(getattr(actual, name), copy=True) for name in names}
    residency = Residency()
    if record is None:
        record = layer_record(actual.grid, actual_pml)
    b_plan = family.plan_metal_complex_conductive_pml_curl(
        actual, actual_pml, "step_B", residency, functions=b_functions,
        chunk_bounds=record)
    d_plan = family.plan_metal_complex_conductive_pml_curl(
        actual, actual_pml, "step_D", residency, chunk_bounds=record)
    if b_plan is None or d_plan is None:
        return {"passed": False, "reason": "the native plan builder refused a declared product"}
    residency.sync_in()
    per_cycle = []
    for cycle in range(1, cycles + 1):
        b_plan.run()
        d_plan.run()
        import torch  # noqa: PLC0415

        torch.mps.synchronize()
        stepping.step_B(reference, reference_pml)
        stepping.step_D(reference, reference_pml)
        residency.sync_out(names)
        diff = sum(differing(getattr(reference, name), getattr(actual, name))
                   for name in names)
        census = subnormal_words(reference, names)
        per_cycle.append({"cycle": cycle, "differing_words": diff,
                          "reference_subnormals": census})
        if diff or census:
            break
    moved = sum(differing(before[name], getattr(actual, name)) for name in names)
    return {
        "passed": (len(per_cycle) == cycles
                   and all(row["differing_words"] == 0
                           and row["reference_subnormals"] == 0 for row in per_cycle)
                   and moved > 0 and b_plan.launches == len(per_cycle)
                   and d_plan.launches == len(per_cycle)),
        "per_cycle": per_cycle,
        "words_compared": int(sum(words(getattr(actual, name)).size for name in names))
        * len(per_cycle),
        "differing_words": per_cycle[-1]["differing_words"],
        "reference_subnormals": per_cycle[-1]["reference_subnormals"],
        "moved_words": moved,
        "b_flags": list(b_plan.conductive),
        "d_flags": list(d_plan.conductive),
        "b_launches": b_plan.launches,
        "d_launches": d_plan.launches,
        "mirrors": len(residency.names),
    }


def _needle(source: str, old: str, new: str) -> str:
    """One planted edit; a needle that matches nothing is refused by name."""
    if old not in source:
        raise LookupError(f"mutation needle not found: {old!r}")
    return source.replace(old, new, 1)


def _needles(source: str, pairs: Sequence[Tuple[str, str]]) -> str:
    """Several edits in turn; each must match exactly once."""
    for old, new in pairs:
        if source.count(old) != 1:
            raise LookupError(f"{old!r} matches {source.count(old)} times, expected 1")
        source = source.replace(old, new)
    return source


def _mutation_kinds() -> Tuple[int, int, int]:
    """The B side's connection kinds on the mutation case's own grid (the plan's)."""
    fields, pml = build(sides=("B", "D"), mixed=False, seed=0)
    phases = family.base.bloch_phase_table(
        fields.grid, stepping._boundary_kinds(fields.grid, pml))  # noqa: SLF001
    kinds, _ = family.base.phase_arguments(phases, backward=False)
    return kinds


#: The B side's dsig / dsigu membership locals per target (half-integer lattice,
#: vec.hpp's cycle_direction) and the coefficient locals each form reads:
#: ``(km1, si1)`` on the dsig axis, ``(km2, si2)`` on the dsigu axis.
_B_DSIG = ("in_h_y", "in_h_z", "in_h_x")
_B_DSIGU = ("in_h_z", "in_h_x", "in_h_y")
_B_COEFFICIENTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("km_y", "si_y", "km_z", "si_z"), ("km_z", "si_z", "km_x", "si_x"),
    ("km_x", "si_x", "km_y", "si_y"))
_DU_LINE = "    const bool du0 = in_h_z, du1 = in_h_x, du2 = in_h_y;\n"


def _flags(source: str, dsig: Sequence[str], dsigu: Sequence[str]) -> str:
    """Every target's (dsig, dsigu) membership replaced by C expressions."""
    pairs = [(_DU_LINE, f"    const bool du0 = {dsigu[0]}, du1 = {dsigu[1]}, "
                        f"du2 = {dsigu[2]};\n")]
    for index in range(3):
        pairs.append((f"    const bool dsig{index} = {_B_DSIG[index]};\n",
                      f"    const bool dsig{index} = {dsig[index]};\n"))
    return _needles(source, pairs)


def _identity(km: str, si: str) -> str:
    """The package's previous predicate: coefficient identity, ``!= 1.0f``."""
    return f"(({km} != 1.0f) || ({si} != 1.0f))"


def mutation_table() -> Dict[str, Dict[str, Any]]:
    """Every source mutation: its kernel text, whether it must be caught, and why.

    ``must_catch``: True (must diverge) or False (a declared null: must change no
    word). ``mixed``: run on the per-target product (Bx lossy, By and Bz lossless),
    where the lossless tails execute; the source is then that product's
    specialisation. The base is the specialisation the case's B plan compiles (its
    connection kinds), so each mutant differs from the shipped kernel by its planted
    edit alone.
    """
    kinds = _mutation_kinds()
    full = family.complex_conductive_pml_curl_source(
        (0, 0, 0), False, kinds, (True, True, True))
    mixed = family.complex_conductive_pml_curl_source(
        (0, 0, 0), False, kinds, (True, False, False))
    swapped = full.replace("cf0[ii]", "TEMP_CF0", 1).replace("ci0[ii]", "cf0[ii]", 1)
    swapped = swapped.replace("TEMP_CF0", "ci0[ii]", 1)
    every = ("true", "true", "true")
    table: Dict[str, Dict[str, Any]] = {
        "condfac_condinv_swapped": {
            "source": swapped, "must_catch": True, "mixed": False,
            "why": "the conductivity factor and its inverse exchanged in the both-layer "
                   "f_cond stage"},
        "f_cond_store_dropped": {
            "source": _needle(full, "h0[ii] = hn0; u0[ii] = un0; f0[ii] = fn0;",
                              "u0[ii] = un0; f0[ii] = fn0;"),
            "must_catch": True, "mixed": False,
            "why": "the f_cond history never advances on the both-layer cells"},
        "bloch_x_connection_dropped": {
            "source": _needle(full, "        b_dx = c_mul(b, px);\n", ""),
            "must_catch": True, "mixed": False,
            "why": "the x connection of target 2's unshifted operand dropped"},
        "select_removed": {
            "source": _needle(full, _DU_LINE,
                              "    const bool du0 = true, du1 = true, du2 = true;\n"),
            "must_catch": True, "mixed": False,
            "why": "every cell takes a dsigu form: fu read and written everywhere"},
        "f_cond_advanced_on_every_cell": {
            "source": _needle(full, "    const bool dsig0 = in_h_y;\n",
                              "    const bool dsig0 = true;\n"),
            "must_catch": True, "mixed": False,
            "why": "target 0's f_cond stage on every cell"},
        "dsig_on_coefficient_identity": {
            "source": _needle(full, "    const bool dsig0 = in_h_y;\n",
                              "    const bool dsig0 = (km_y != 1.0f) || (si_y != 1.0f);\n"),
            "must_catch": True, "mixed": False,
            "why": "target 0's dsig case keyed on coefficient identity"},
        "curl_previous_grouping": {
            "source": _needle(full, "c_mul_coefficient_left(dtdx, ((c_y - c_dy) + b_dz) - b_z)",
                              "c_mul_coefficient_left(dtdx, (c_y - c_dy) + (b_dz - b_z))"),
            "must_catch": True, "mixed": False,
            "why": "INVERTED (KERNELS.md §7.1): the package's previous pairwise grouping "
                   "(g1s - g1) + (g2 - g2s) planted in place of MEEP's left-to-right "
                   "((g1s - g1) + g2) - g2s, per part"},
        "increment_previous_grouping": {
            "source": _needles(full, (
                ("float2 un0 = c_mul_field_left(c_mul_field_left(up0, km_y) + hs0, si_y);",
                 "float2 un0 = c_mul_field_left((c_mul_field_left(up0, km_y) + hn0) - hp0, "
                 "si_y);"),
                ("float2 fn0 = c_mul_field_left(c_mul_field_left(fp0, km_y) + hs0, si_y);",
                 "float2 fn0 = c_mul_field_left((c_mul_field_left(fp0, km_y) + hn0) - hp0, "
                 "si_y);"))),
            "must_catch": True, "mixed": False,
            "why": "INVERTED (KERNELS.md §7.1): the package's previous grouping "
                   "((x*kms) + fcnd') - fcnd planted in place of MEEP's increment "
                   "(fcnd' - fcnd) formed first (step_generic.cpp:167, :209), on the "
                   "both-layer and dsig-only forms"},
        "b1_select_removed": {
            "source": _flags(full, every, every),
            "must_catch": True, "mixed": False,
            "why": "every cell steps the both-layer form (the whole-grid recurrence): "
                   "f_cond and fu are written everywhere"},
        "b2_predicate_on_coefficient_identity": {
            "source": _flags(full, [_identity(c[0], c[1]) for c in _B_COEFFICIENTS],
                             [_identity(c[2], c[3]) for c in _B_COEFFICIENTS]),
            "must_catch": True, "mixed": False,
            "why": "the forms keyed on kms/sinv != 1 (the previous package predicate) "
                   "instead of chunk membership: a sigma = 0 row inside a layer chunk "
                   "takes another form"},
        "b5_dsig_dsigu_swapped": {
            "source": _flags(full, _B_DSIGU, _B_DSIG),
            "must_catch": True, "mixed": False,
            "why": "each target's dsig and dsigu memberships exchanged: dsig-only and "
                   "dsigu-only cells trade forms"},
        "b6_f_cond_stored_off_the_dsig_cells": {
            "source": _needle(full, "        u0[ii] = un0; f0[ii] = fn0;\n",
                              "        h0[ii] = c_mul_field_left(c_mul_field_left(h0[ii], "
                              "cf0[ii]) - curl0, ci0[ii]); u0[ii] = un0; f0[ii] = fn0;\n"),
            "must_catch": True, "mixed": False,
            "why": "f_cond advanced on the dsigu-only cells, where the array path leaves "
                   "it untouched (the auxiliary contract) and the gate compares it"},
        "b6_fu_stored_off_the_dsigu_cells": {
            "source": _needle(full, "        h0[ii] = hn0; f0[ii] = fn0;\n",
                              "        h0[ii] = hn0; u0[ii] = c_mul_field_left("
                              "c_mul_field_left(u0[ii], km_y) + hs0, si_y); "
                              "f0[ii] = fn0;\n"),
            "must_catch": True, "mixed": False,
            "why": "fu advanced on the dsig-only cells, where the array path leaves it "
                   "untouched"},
        "b6_lossless_auxiliary_stored_on_a_plain_cell": {
            "source": _needle(mixed, "    if (du1) { u1[ii] = n1; }\n",
                              "    u1[ii] = n1;\n"),
            "must_catch": True, "mixed": True,
            "why": "a lossless target's fu written outside the recurrence's cells"},
        "b7_history_load_unconditional": {
            "source": _needle(full,
                              "    float2 hp0 = dsig0 ? h0[ii] : float2(0.0f, 0.0f);\n",
                              "    float2 h0_hoisted = h0[ii];\n"
                              "    float2 hp0 = dsig0 ? h0_hoisted : float2(0.0f, 0.0f);\n"),
            "must_catch": False, "mixed": False,
            "why": "DECLARED NULL: an f_cond load hoisted above the select with every "
                   "store still guarded changes no word; the guard is traffic, pinned "
                   "statically by test_metal_complex_conductive_pml"},
        "b8_lossless_plain_curl_through_the_recurrence": {
            "source": _needles(mixed, (
                ("    float2 p1 = du1 ? u1[ii] : f1[ii];\n",
                 "    float2 p1 = du1 ? u1[ii] : float2(0.0f, 0.0f);\n"),
                ("    float2 v1 = du1 ? c_mul_field_left((c_mul_field_left(f1[ii], km_x) "
                 "+ n1) - p1, si_x) : n1;\n",
                 "    float2 v1 = c_mul_field_left((c_mul_field_left(f1[ii], km_x) + n1) "
                 "- p1, si_x);\n"))),
            "must_catch": True, "mixed": True,
            "why": "a lossless target's dsig-only form spelled as the recurrence with "
                   "p := 0: differs wherever its dsig coefficients are not 1"},
    }
    return table


def mutation_sources() -> Dict[str, str]:
    """Each source mutation's kernel text (the table's ``source`` column)."""
    return {name: entry["source"] for name, entry in mutation_table().items()}


def _bound_off_by_one(record: Any) -> Any:
    """Every non-empty low run one row longer: the boundary plane changes form."""
    out = np.array(record, copy=True)
    for half in (False, True):
        for axis in range(3):
            a0, a1, b0 = (int(out[record_index(half, axis, w)]) for w in range(3))
            if a1 > a0 and (b0 == 0 or a1 + 1 < b0):
                out[record_index(half, axis, 1)] = a1 + 1
    return out


def _lattices_swapped(record: Any) -> Any:
    """The integer and half-integer entries exchanged."""
    out = np.array(record, copy=True)
    for axis in range(3):
        for word in range(4):
            i, h = record_index(False, axis, word), record_index(True, axis, word)
            out[i], out[h] = record[h], record[i]
    return out


#: The record mutations (KERNELS.md §7.2): the kernel is the shipped one and the record
#: it is handed is not; the reference steps the true division.
RECORD_MUTATIONS: Dict[str, Tuple[Any, str]] = {
    "b3_bound_off_by_one": (_bound_off_by_one, "a run end one row past the chunk"),
    "b4_lattices_swapped": (_lattices_swapped,
                            "the integer and half-integer records swapped"),
}


def form_counts(fields: Fields, pml: PML, sub_step: str) -> Dict[str, int]:
    """Cells per (target, form) on the true record: the forms this product reaches."""
    record = layer_record(fields.grid, pml)
    shape = tuple(fields.grid.shape)
    half = shaders.CURL_HALF_INTEGER[sub_step == "step_D"]

    def line(axis: int) -> Any:
        view = [1, 1, 1]
        view[axis] = shape[axis]
        return np.broadcast_to(
            membership_from_record(record, half, axis, shape[axis]).reshape(view), shape)

    counts: Dict[str, int] = {}
    for index, target in enumerate(family.SUB_STEPS[sub_step]["targets"]):
        dsig_axis, dsigu_axis = shaders.CURL_CYCLE[index]
        ds, du = line(dsig_axis), line(dsigu_axis)
        if fields.condfac_for(target) is not None:
            for name, mask in (("both", ds & du), ("dsigu_only", ~ds & du),
                               ("dsig_only", ds & ~du), ("neither", ~ds & ~du)):
                counts[f"{target}/{name}"] = int(np.count_nonzero(mask))
        else:
            for name, mask in (("recurrence", du), ("dsig_only", ~du)):
                counts[f"{target}/lossless_{name}"] = int(np.count_nonzero(mask))
    return counts


#: Every form a product must reach somewhere, or a mutation of it is vacuous.
FORMS = ("both", "dsigu_only", "dsig_only", "neither", "lossless_recurrence",
         "lossless_dsig_only")


#: The refusal leg's cases: (side, curl sub-step, the one lossy target).
PRESENCE_CASES = (("B", "step_B", "Bx"), ("D", "step_D", "Dz"))


def partial_presence(side: str, slot: str, target: str) -> Tuple[Fields, PML, Any]:
    """One lossy target whose conductivity is nonzero on two x rows only.

    On the one-rank division some chunks keep it and others do not
    (``ChunkLayout.conductive_chunks``); returns the fields, the absorber and the
    per-chunk presence.
    """
    fields, pml = build(sides=(), mixed=False, seed=19500)
    sigma = np.zeros(fields.grid.shape, dtype=np.float32)
    sigma[3:5] = np.float32(0.15)
    setter = fields.set_b_conductivity if side == "B" else fields.set_d_conductivity
    setter({name: (sigma if name == target else None)
            for name in family.SUB_STEPS[slot]["targets"]})
    kept = ChunkLayout.one_rank(fields.grid, pml).conductive_chunks(
        sigma, IYEE_SHIFTS[target])
    return fields, pml, kept


def refusal_row() -> Dict[str, Any]:
    """A chunk that keeps no conductivity, on each side, is refused by name.

    The B case refuses the B curl; the D case refuses the D curl; each refuses the
    family's H/E arms through the companion. A control with the same conductivity on
    every row is admitted.
    """
    out: Dict[str, Any] = {}
    for side, slot, target in PRESENCE_CASES:
        fields, pml, kept = partial_presence(side, slot, target)
        curl = family.metal_complex_conductive_pml_curl_coverage(
            fields, pml, slot, Residency())
        constitutive = family.metal_complex_conductive_pml_constitutive_coverage(
            fields, pml, "E", Residency())
        plan = family.plan_metal_complex_conductive_pml_curl(
            fields, pml, slot, Residency(),
            chunk_bounds=layer_record(fields.grid, pml))
        full, full_pml = build(sides=(side,), mixed=False, seed=19501)
        control = family.metal_complex_conductive_pml_curl_coverage(
            full, full_pml, slot, Residency())
        out[side] = {
            "kept": list(kept),
            "curl_covered": curl.covered, "curl_reasons": list(curl.reasons),
            "constitutive_covered": constitutive.covered,
            "constitutive_reasons": list(constitutive.reasons),
            "plan_refused": plan is None,
            "control_covered": control.covered,
            "control_reasons": list(control.reasons),
        }
        needle = f"structure chunks keep the conductivity of {target}"
        out[side]["passed"] = bool(
            any(kept) and not all(kept) and not curl.covered and plan is None
            and any(needle in reason for reason in curl.reasons)
            and not constitutive.covered
            and any(needle in reason for reason in constitutive.reasons)
            and control.covered)
    out["passed"] = all(out[side]["passed"] for side in ("B", "D"))
    return out


def source_provenance() -> Dict[str, Any]:
    specializations = {}
    for codes, phased, flags in (
            ((0, 0, 0), (1, 1, 0), (True, True, True)),
            ((0, 0, 0), (1, 1, 0), (False, False, False)),
            ((1, 0, 1), (0, 1, 0), (True, False, True))):
        for sub_step in ("step_B", "step_D"):
            source = family.complex_conductive_pml_curl_source(
                codes, sub_step == "step_D", phased, flags)
            specializations[f"{sub_step}/{codes}/{phased}/{flags}"] = sha256_bytes(
                source.encode("utf-8"))
    return {
        "files": {
            "family": sha256_file(Path(family.__file__).resolve()),
            "complex_stencil": sha256_file(Path(family.base.__file__).resolve()),
            "gate": sha256_file(Path(__file__).resolve()),
        },
        "specialized_metal_sources": specializations,
        "mutation_metal_sources": {
            name: sha256_bytes(source.encode("utf-8"))
            for name, source in mutation_sources().items()
        },
    }


def emit(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    must = row.get("must_catch")
    print(f"{row['index']}/{row['total']} {row['leg']} {row['label']}: "
          f"diff={row.get('differing_words', '-')}"
          f"{'' if must is None else f' must_catch={must}'} passed={row.get('passed')}",
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="Final JSON artifact; the sibling JSONL is incremental")
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()
    if args.cycles < 1:
        raise SystemExit("--cycles must be positive")
    import torch

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")
    products = (
        ("two_sided_phased", ("B", "D"), False, (0.2, -0.125, 0.0),
         ("periodic", "periodic", "periodic")),
        ("D_only_phased", ("D",), False, (0.2, -0.125, 0.0),
         ("periodic", "periodic", "periodic")),
        ("B_only_x_phase", ("B",), False, (0.2, 0.0, 0.0),
         ("periodic", "periodic", "periodic")),
        ("mixed_components", ("B", "D"), True, (0.2, -0.125, 0.0),
         ("periodic", "periodic", "periodic")),
        ("metallic_xz", ("B", "D"), False, (0.0, 0.125, 0.0),
         ("metallic", "periodic", "metallic")),
    )
    table = mutation_table()
    total = len(products) + len(table) + len(RECORD_MUTATIONS) + 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    rows = []
    reached: Dict[str, int] = {}
    started = time.perf_counter()
    with jsonl.open("w", encoding="utf-8") as handle:
        for index, (label, sides, mixed, phase, boundaries) in enumerate(products, 1):
            result = execute_case(sides=sides, mixed=mixed, cycles=args.cycles,
                                  seed=18000 + index, phase=phase,
                                  boundaries=boundaries)
            fields, pml = build(sides=sides, mixed=mixed, seed=18000 + index,
                                phase=phase, boundaries=boundaries)
            forms: Dict[str, int] = {}
            for slot in ("step_B", "step_D"):
                forms.update(form_counts(fields, pml, slot))
            for key, count in forms.items():
                form = key.split("/", 1)[1]
                reached[form] = reached.get(form, 0) + count
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "sides": sides, "mixed": mixed, "phase": phase,
                   "boundaries": boundaries, "cycles": args.cycles, "forms": forms,
                   **result}
            rows.append(row)
            emit(handle, row)
        census_passed = all(reached.get(form, 0) > 0 for form in FORMS)
        print(f"form census {reached} passed={census_passed}", flush=True)
        offset = len(products)
        for number, (label, entry) in enumerate(table.items(), 1):
            function = compile_source(entry["source"]).bloch_pml_curl_step
            result = execute_case(sides=("B", "D"), mixed=entry["mixed"], cycles=1,
                                  seed=19000 + number,
                                  b_functions={shaders.CONTRACT_OFF: function})
            diff = int(result.get("differing_words", 0))
            caught = diff > 0
            row = {"index": offset + number, "total": total, "leg": "mutation",
                   "label": label, "differing_words": diff,
                   "b_launches": result.get("b_launches"),
                   "must_catch": entry["must_catch"], "mixed": entry["mixed"],
                   "why": entry["why"], "caught": caught,
                   "passed": bool(result.get("b_launches") == 1
                                  and caught == entry["must_catch"])}
            rows.append(row)
            emit(handle, row)
        offset += len(table)
        for number, (label, (mutate, why)) in enumerate(RECORD_MUTATIONS.items(), 1):
            fields, pml = build(sides=("B", "D"), mixed=False, seed=19300 + number)
            truth = layer_record(fields.grid, pml)
            wrong = mutate(truth)
            changed = int(np.count_nonzero(np.asarray(wrong) != np.asarray(truth)))
            result = execute_case(sides=("B", "D"), mixed=False, cycles=1,
                                  seed=19300 + number, record=wrong)
            diff = int(result.get("differing_words", 0))
            row = {"index": offset + number, "total": total, "leg": "record_mutation",
                   "label": label, "differing_words": diff, "record_words_changed": changed,
                   "b_launches": result.get("b_launches"), "must_catch": True,
                   "why": why, "caught": diff > 0,
                   "passed": bool(changed > 0 and result.get("b_launches") == 1
                                  and diff > 0)}
            rows.append(row)
            emit(handle, row)
        refusal = refusal_row()
        row = {"index": total, "total": total, "leg": "refusal",
               "label": "partial_conductivity_presence", **refusal}
        rows.append(row)
        emit(handle, row)
    product_rows = [row for row in rows if row["leg"] == "product"]
    result = {
        "verdict": ("PASS" if census_passed and all(row["passed"] for row in rows)
                    else "FAIL"),
        "elapsed_seconds": time.perf_counter() - started,
        "product_rows": len(products),
        "mutation_rows": len(table) + len(RECORD_MUTATIONS),
        "form_census": reached,
        "form_census_passed": census_passed,
        "words_compared": sum(int(row.get("words_compared", 0)) for row in product_rows),
        "words_differing": sum(int(row.get("differing_words", 0)) for row in product_rows),
        "rows": rows,
        "torch_version": torch.__version__,
        "metal_frontend": metal_frontend_version(),
        "subnormal_policy": subnormal.mps_policy_report("flush"),
        "source_provenance": source_provenance(),
        "jsonl": str(jsonl),
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(result)  # bytes THIS process imported; see gate_provenance
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(f"VERDICT {result['verdict']} in {result['elapsed_seconds']:.2f}s; "
          f"artifact {args.out}", flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
