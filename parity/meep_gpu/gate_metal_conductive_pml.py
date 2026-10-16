#!/usr/bin/env python3
"""Native MPS byte gate for real conductive active-PML B/D curl sub-steps.

This is the active-absorber counterpart of the no-PML conductive gate.  It tests
the branch that owns both the split-field ``fu`` recurrence and MEEP's per-target
``f_cond`` recurrence.  The gate is deliberately narrower than a complete driver
step: its claim is the B/D arithmetic and persistent mirror continuity, not
production dispatch or cross-sub-step fusion.

Every product compares float32 *words*, never tolerances, against ``stepping``.
The mutations are compiled from altered Metal source and must visibly diverge;
the refusal leg protects the inverse predicate boundary so this family cannot
silently overlap a lossless ordinary-PML plan.

THE PER-CHUNK FORMS (KERNELS.md §7). The kernel selects MEEP's four conductive forms
per structure chunk from the bound chunk record, and each plan here binds the one-rank
record of its grid (``gate_chunk_records.layer_record``), the division ``stepping``
steps a bare ``Fields`` with. Each product records its form census (every conductive
form and both lossless forms on in-run and out-of-run cells). The grouping mutations
are INVERTED (§7.1): the shipped stencil and the shipped ``f_cond`` increment are
MEEP's groupings, so each mutation plants the package's PREVIOUS grouping and stays
must-catch. The §7.2 battery is armed on the conductive and lossless tails (``b*``),
the record mutations hand the shipped kernel a wrong record, and the one declared null
(``b7``, an unconditional ``f_cond`` load with every store still guarded) must change
no word.

THE PER-CHUNK CONDUCTIVITY PRESENCE (KERNELS.md §2's presence extension). Two products
keep the conductivity in the interior chunk along x only (``partial`` in
:func:`build`), so a lossy target steps the lossless forms in the other chunks and the
plan binds non-trivial presence words; the form census counts those cells, two
must-catch rows invert the kernel's presence flag and bind the all-kept words on that
state, and the refusal leg requires a record that cannot carry the presence to be
refused by name. No planted signed-zero leg: every form here is arithmetic on its own inputs
(the dsig-only form with unit coefficients equals MEEP's plain one on every special
pair, DESIGN.md D1); the plain output selects are the base gate's
(``gate_metal_pml.py``, ``planted_zero``).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

import metal_gate_kit as kit  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.chunk_map import (  # noqa: E402
    ChunkLayout,
    membership_from_record,
    record_index,
)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import conductive_pml as family  # noqa: E402
from meep_gpu.metal_kernels import launch, shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import Residency, compile_source  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from gate_chunk_records import layer_record  # noqa: E402


def _words(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value)
    if array.dtype != np.float32:
        raise TypeError(f"expected float32, got {array.dtype}")
    return array.reshape(-1).view(np.uint32)


def _differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(_words(left) != _words(right)))


class _Tally:
    """The gate headline is exact words, not a misleading row count."""

    def __init__(self) -> None:
        self.compared = 0
        self.differing = 0

    def compare(self, actual: Any, reference: Any) -> int:
        self.compared += int(_words(actual).size)
        differing = _differing(actual, reference)
        self.differing += differing
        return differing


TALLY = _Tally()


def _sigma(shape: Sequence[int]) -> np.ndarray:
    """A nonconstant f32 loss volume: all four PML cases are reachable."""
    line = np.linspace(np.float32(0.06), np.float32(0.22), int(shape[0]),
                       dtype=np.float32)[:, None, None]
    return np.broadcast_to(line, tuple(shape)).copy()


def _partial_sigma(shape: Sequence[int]) -> np.ndarray:
    """The loss of :func:`_sigma`, kept on the interior chunk along x only.

    Rows 4..5 of the ten x rows: every chunk whose x run is the interior one keeps the
    conductivity and every other chunk deletes it (structure.cpp:883-900), on either
    sub-lattice. The refusal leg asserts the split is real (some, not all, chunks keep it).
    """
    sigma = np.zeros(tuple(shape), dtype=np.float32)
    sigma[4:6] = _sigma(shape)[4:6]
    return sigma


def build(*, sides: Sequence[str], mixed: bool, boundaries: Sequence[str],
          seed: int, partial: bool = False) -> Tuple[Fields, PML]:
    """One real active-PML state with selected B and/or D conductivity.

    ``partial`` keeps the conductivity in some structure chunks only
    (:func:`_partial_sigma`).
    """
    grid = Grid(resolution=8.0, cell_size=(1.25, 1.0, 0.75), courant=0.35,
                boundaries=tuple(boundaries), xp=np)
    fields = Fields(grid=grid)
    fields.set_background_eps(2.25)
    fields.enable_pml_storage()
    rng = np.random.default_rng(seed)
    for name in (
            "Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez",
            "Hx", "Hy", "Hz", "fu_Bx", "fu_By", "fu_Bz", "fu_Dx",
            "fu_Dy", "fu_Dz"):
        getattr(fields, name)[...] = rng.uniform(
            -0.3, 0.3, grid.shape).astype(np.float32)

    sigma = _partial_sigma(grid.shape) if partial else _sigma(grid.shape)
    if "B" in sides:
        fields.set_b_conductivity({
            "Bx": sigma if not mixed else sigma,
            "By": sigma if not mixed else None,
            "Bz": sigma if not mixed else None,
        })
    if "D" in sides:
        fields.set_d_conductivity({
            "Dx": sigma if not mixed else None,
            "Dy": sigma if not mixed else None,
            "Dz": sigma if not mixed else sigma,
        })
    for target in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        history = getattr(fields, "f_cond_" + target, None)
        if history is not None:
            history[...] = rng.uniform(-0.2, 0.2, grid.shape).astype(np.float32)
    return fields, PML(grid=grid, thickness=1)


def _watched(fields: Fields, sub_step: str) -> Tuple[str, ...]:
    names: List[str] = []
    for target in family.CONDUCTIVE_SUB_STEPS[sub_step]:
        names.extend((target, "fu_" + target))
        if getattr(fields, "f_cond_" + target, None) is not None:
            names.append("f_cond_" + target)
    return tuple(names)


def _sub_steps(sides: Sequence[str]) -> Tuple[str, ...]:
    return tuple(slot for side, slot in (("B", "step_B"), ("D", "step_D"))
                 if side in sides)


def execute(*, sides: Sequence[str], mixed: bool, boundaries: Sequence[str],
            seed: int, cycles: int,
            functions: Mapping[str, Mapping[str, Any]] | None = None,
            record_tally: bool = True, record: Any = None, partial: bool = False,
            presence: Any = None,
            ) -> Dict[str, Any]:
    """Run the selected conductive sub-steps over one shared MPS residency.

    ``record`` is the chunk record the plans bind; ``None`` is the true one-rank record
    of the grid (the division the reference steps with). A record mutation hands in a
    wrong one and the reference stays on the truth. ``presence`` is the words the plans
    bind; ``None`` builds them from the array path's own presence (a presence mutation
    hands in wrong ones).
    """
    reference, reference_pml = build(sides=sides, mixed=mixed,
                                     boundaries=boundaries, seed=seed, partial=partial)
    actual, actual_pml = build(sides=sides, mixed=mixed,
                                boundaries=boundaries, seed=seed, partial=partial)
    slots = _sub_steps(sides)
    residency = Residency()
    bound = layer_record(actual.grid, actual_pml) if record is None else record
    plans = {
        slot: family.plan_metal_conductive_pml_curl(
            actual, actual_pml, slot, residency,
            functions=None if functions is None else functions.get(slot),
            chunk_bounds=bound, presence=presence)
        for slot in slots
    }
    assert all(plan is not None for plan in plans.values()), (
        "declared conductive product was refused", slots, plans)
    watched = {slot: _watched(actual, slot) for slot in slots}
    before = {name: np.array(getattr(actual, name), copy=True)
              for names in watched.values() for name in names}
    rows: List[Dict[str, Any]] = []
    residency.sync_in()
    for cycle in range(1, cycles + 1):
        for slot in slots:
            plan = plans[slot]
            assert plan is not None
            plan.run()
            import torch  # noqa: PLC0415

            torch.mps.synchronize()
            getattr(stepping, slot)(reference, reference_pml)
            residency.sync_out(watched[slot])
            compare = TALLY.compare if record_tally else _differing
            differing = sum(compare(getattr(actual, name), getattr(reference, name))
                            for name in watched[slot])
            moved = sum(_differing(before[name], getattr(reference, name))
                        for name in watched[slot])
            kit.assert_moved(moved, f"{sides}/{slot}/cycle-{cycle}")
            row = {
                "cycle": cycle,
                "slot": slot,
                "differing_words": differing,
                "moved_words": moved,
                "words_compared": int(sum(_words(getattr(actual, name)).size
                                          for name in watched[slot])),
                "reference_subnormal_words": int(sum(
                    subnormal.census(getattr(reference, name))
                    for name in watched[slot])),
                "launches": plan.launches,
                "conductive": list(plan.conductive),
            }
            rows.append(row)
            if differing or row["reference_subnormal_words"]:
                return {"passed": False, "rows": rows,
                        "reason": "different or subnormal reference words"}
    residency.verify()
    launches = {slot: plans[slot].launches for slot in slots if plans[slot] is not None}
    return {
        "passed": (all(row["differing_words"] == 0 for row in rows)
                   and all(count == cycles for count in launches.values())),
        "rows": rows,
        "launches": launches,
        "residency_mirrors": len(residency.names),
        "presence": {slot: list(plans[slot].presence) for slot in slots},
    }


def _needles(source: str, pairs: Sequence[Tuple[str, str]]) -> str:
    """Apply several one-occurrence needles in turn; each must land."""
    for old, new in pairs:
        if source.count(old) != 1:
            raise LookupError(f"{old!r} matches {source.count(old)} times, expected 1")
        source = source.replace(old, new)
    return source


#: The B side's membership locals per target (half-integer lattice), as
#: :func:`.conductive_pml.membership_flags` names them.
_B_FLAGS = family.membership_flags(False)
_B_COEFFICIENTS: Tuple[Tuple[str, str, str, str], ...] = (
    ("km_y", "si_y", "km_z", "si_z"), ("km_z", "si_z", "km_x", "si_x"),
    ("km_x", "si_x", "km_y", "si_y"))


def _flags_line(index: int, conductive: bool) -> str:
    dsig, dsigu = _B_FLAGS[index]
    if conductive:
        return f"    const bool ds{index} = {dsig}, du{index} = {dsigu};\n"
    return f"    const bool du{index} = {dsigu};\n"


def _set_flags(source: str, conductive: Sequence[bool],
               values: Sequence[Tuple[str, str]]) -> str:
    """Every target's membership flags replaced by C expressions ``(dsig, dsigu)``."""
    pairs = []
    for index, flag in enumerate(conductive):
        dsig, dsigu = values[index]
        new = (f"    const bool ds{index} = {dsig}, du{index} = {dsigu};\n" if flag
               else f"    const bool du{index} = {dsigu};\n")
        pairs.append((_flags_line(index, flag), new))
    return _needles(source, pairs)


def _identity_predicate(index: int) -> Tuple[str, str]:
    """The package's previous predicate: coefficient identity, ``!= 1.0f``."""
    km1, si1, km2, si2 = _B_COEFFICIENTS[index]
    return (f"(({km1} != 1.0f) || ({si1} != 1.0f))",
            f"(({km2} != 1.0f) || ({si2} != 1.0f))")


#: ``must_catch``: True (must be caught) | False (a declared null: must change no word).
#: ``mixed``: run on the per-target product (Bx lossy, By and Bz lossless), which is
#: where the lossless tails execute. Every row runs on the B side under a metallic x/z
#: and periodic y boundary, where every form is reached (the products leg's census).
def _mutation_table() -> Dict[str, Dict[str, Any]]:
    full = family.conductive_pml_curl_source((1, 0, 1), False, (True, True, True))
    mixed = family.conductive_pml_curl_source((1, 0, 1), False, (True, False, False))
    if "cf0[ii]" not in full or "ci0[ii]" not in full:
        raise LookupError("conductive coefficient operands")
    # ``cf0`` / ``ci0`` occur in all four PML branches.  Swap every occurrence:
    # a one-occurrence mutation would leave three branches compiling against an
    # undefined temporary and would test Metal's diagnostics, not the recurrence.
    swapped = full.replace("cf0[ii]", "TEMP_CF0").replace(
        "ci0[ii]", "cf0[ii]").replace("TEMP_CF0", "ci0[ii]")
    lossy, lossless = (True, True, True), (True, False, False)
    return {
        "condfac_condinv_swapped": {
            "source": swapped, "must_catch": True, "mixed": False,
            "why": "the conductivity factor and its inverse exchanged in every branch"},
        "conductivity_history_store_dropped": {
            "source": _needles(full, (
                ("c0[ii] = c0_new; u0[ii] = u0_new; f0[ii] = f0_new;",
                 "u0[ii] = u0_new; f0[ii] = f0_new;"),)),
            "must_catch": True, "mixed": False,
            "why": "the f_cond history never advances on the both-layer cells"},
        "curl_previous_grouping": {
            "source": _needles(full, (("dtdx * (((c_y - c) + b) - b_z)",
                                       "dtdx * ((c_y - c) + (b - b_z))"),)),
            "must_catch": True, "mixed": False,
            "why": "INVERTED (KERNELS.md §7.1): the package's previous pairwise "
                   "grouping (g1s - g1) + (g2 - g2s) planted in place of MEEP's "
                   "left-to-right ((g1s - g1) + g2) - g2s"},
        "increment_previous_grouping": {
            "source": _needles(full, (
                ("float u0_new = ((u0_previous * km_y) + c0_step) * si_y;",
                 "float u0_new = (((u0_previous * km_y) + c0_new) - c0_previous) * si_y;"),
                ("float f0_new = ((f0_previous * km_y) + c0_step) * si_y;",
                 "float f0_new = (((f0_previous * km_y) + c0_new) - c0_previous) * si_y;"))),
            "must_catch": True, "mixed": False,
            "why": "INVERTED (KERNELS.md §7.1): the package's previous grouping "
                   "((x*kms) + fcnd') - fcnd planted in place of MEEP's increment "
                   "(fcnd' - fcnd) formed first (step_generic.cpp:167, :209)"},
        "dsig_membership_disabled": {
            "source": _needles(full, ((_flags_line(0, True),
                                       f"    const bool ds0 = false, du0 = {_B_FLAGS[0][1]};\n"),)),
            "must_catch": True, "mixed": False,
            "why": "target 0's dsig layer never selected: both-layer cells step the "
                   "dsigu-only form and dsig-only cells the neither form"},
        "b1_select_removed": {
            "source": _set_flags(full, lossy, [("true", "true")] * 3),
            "must_catch": True, "mixed": False,
            "why": "every cell steps the both-layer form (the whole-grid recurrence): "
                   "f_cond and fu are written everywhere"},
        "b2_predicate_on_coefficient_identity": {
            "source": _set_flags(full, lossy, [_identity_predicate(t) for t in range(3)]),
            "must_catch": True, "mixed": False,
            "why": "the forms keyed on kms/sinv != 1 (the previous package predicate) "
                   "instead of chunk membership: a sigma = 0 row inside a layer chunk "
                   "takes another form"},
        "b5_dsig_dsigu_swapped": {
            "source": _set_flags(full, lossy, [(dsigu, dsig) for dsig, dsigu in _B_FLAGS]),
            "must_catch": True, "mixed": False,
            "why": "each target's dsig and dsigu memberships exchanged: dsig-only and "
                   "dsigu-only cells trade forms"},
        "b6_f_cond_stored_off_the_dsig_cells": {
            "source": _needles(full, (
                ("        u0[ii] = u0_new; f0[ii] = f0_new;\n",
                 "        c0[ii] = ((c0[ii] * cf0[ii]) - curl0) * ci0[ii]; "
                 "u0[ii] = u0_new; f0[ii] = f0_new;\n"),)),
            "must_catch": True, "mixed": False,
            "why": "f_cond advanced on the dsigu-only cells, where the array path "
                   "leaves it untouched (the auxiliary contract) and the gate compares it"},
        "b6_fu_stored_off_the_dsigu_cells": {
            "source": _needles(full, (
                ("        c0[ii] = c0_new; f0[ii] = f0_new;\n",
                 "        c0[ii] = c0_new; u0[ii] = ((u0[ii] * km_y) + c0_step) * si_y; "
                 "f0[ii] = f0_new;\n"),)),
            "must_catch": True, "mixed": False,
            "why": "fu advanced on the dsig-only cells, where the array path leaves it "
                   "untouched"},
        "b6_lossless_auxiliary_stored_on_a_plain_cell": {
            "source": _needles(mixed, (("    if (du1) { u1[ii] = n1; }\n",
                                        "    u1[ii] = n1;\n"),)),
            "must_catch": True, "mixed": True,
            "why": "a lossless target's fu written outside the recurrence's cells"},
        "b7_history_load_unconditional": {
            "source": _needles(full, (
                ("    if (!kc0) {\n",
                 "    float c0_hoisted = c0[ii];\n    if (!kc0) {\n"),
                ("    } else if (ds0 && du0) {\n        float c0_previous = c0[ii];\n",
                 "    } else if (ds0 && du0) {\n        float c0_previous = c0_hoisted;\n"),
                )),
            "must_catch": False, "mixed": False,
            "why": "DECLARED NULL: an f_cond load hoisted above the select with every "
                   "store still guarded changes no word; the guard is traffic, pinned "
                   "statically by test_metal_conductive_pml"},
        "b9_presence_flag_inverted": {
            "source": _needles(full, (("    if (!kc0) {\n", "    if (kc0) {\n"),)),
            "must_catch": True, "mixed": False, "partial": True,
            "why": "target 0's presence flag inverted: the chunks that keep the "
                   "conductivity step the lossless forms and the chunks that deleted "
                   "it the conductive ones, advancing f_cond there"},
        "b8_lossless_plain_curl_through_the_recurrence": {
            "source": _needles(mixed, (
                ("    float p1 = du1 ? u1[ii] : f1[ii];\n",
                 "    float p1 = du1 ? u1[ii] : 0.0f;\n"),
                ("    float f1_new = du1 ? ((((f1[ii] * km_x) + n1) - p1) * si_x) : n1;\n",
                 "    float f1_new = (((f1[ii] * km_x) + n1) - p1) * si_x;\n"))),
            "must_catch": True, "mixed": True,
            "why": "a lossless target's dsig-only form spelled as the recurrence with "
                   "p := 0: differs wherever its dsig coefficients are not 1"},
    }


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

#: The presence mutations: the kernel is the shipped one and the words it is handed
#: are not, on the partial state; the reference steps the array path's presence.
PRESENCE_MUTATIONS: Dict[str, Tuple[Any, str]] = {
    "b10_presence_words_all_kept": (
        family.ALL_KEPT,
        "every chunk declared to keep the conductivity: the chunks that deleted it "
        "step the conductive forms with unit coefficients, which differ from the "
        "lossless ones on their dsig cells, and advance f_cond there"),
}


def _form_counts(fields: Fields, pml: PML, sub_step: str) -> Dict[str, int]:
    """Cells per (target, form) on the true record: the forms this product reaches."""
    record = layer_record(fields.grid, pml)
    shape = tuple(fields.grid.shape)
    backward = sub_step == "step_D"
    half = shaders.CURL_HALF_INTEGER[backward]

    def line(axis: int) -> Any:
        view = [1, 1, 1]
        view[axis] = shape[axis]
        return membership_from_record(record, half, axis, shape[axis]).reshape(view)

    counts: Dict[str, int] = {}
    region = stepping._region_plan(fields, pml)  # noqa: SLF001
    for index, target in enumerate(family.CONDUCTIVE_SUB_STEPS[sub_step]):
        dsig_axis, dsigu_axis = shaders.CURL_CYCLE[index]
        ds = np.broadcast_to(line(dsig_axis), shape)
        du = np.broadcast_to(line(dsigu_axis), shape)
        if fields.condfac_for(target) is not None:
            # The cells whose chunk keeps the conductivity take the four forms; the
            # others (a partial presence) the lossless two, counted apart.
            kept = np.broadcast_to(region.conductive_mask(
                target, fields.conductivity_for(target), np), shape)
            for name, mask in (("both", ds & du), ("dsigu_only", ~ds & du),
                               ("dsig_only", ds & ~du), ("neither", ~ds & ~du)):
                counts[f"{target}/{name}"] = int(np.count_nonzero(mask & kept))
            for name, mask in (("recurrence", du), ("dsig_only", ~du)):
                counts[f"{target}/unkept_{name}"] = int(np.count_nonzero(mask & ~kept))
        else:
            for name, mask in (("recurrence", du), ("dsig_only", ~du)):
                counts[f"{target}/lossless_{name}"] = int(np.count_nonzero(mask))
    return counts


def leg_products(payload: Dict[str, Any], out: str) -> None:
    """Full, B-only, D-only, and per-target split conductivity products."""
    products: Tuple[Tuple[str, Tuple[str, ...], bool, Tuple[str, ...], bool], ...] = (
        ("two_sided_periodic", ("B", "D"), False,
         ("periodic", "periodic", "periodic"), False),
        ("b_only_metallic", ("B",), False,
         ("metallic", "periodic", "metallic"), False),
        ("d_only_mixed", ("D",), False,
         ("periodic", "metallic", "periodic"), False),
        ("per_target_mixed", ("B", "D"), True,
         ("metallic", "periodic", "metallic"), False),
        # The per-chunk presence: the conductivity kept by the interior chunk along x
        # only, so every lossy target also steps the lossless forms elsewhere.
        ("partial_presence_two_sided", ("B", "D"), False,
         ("metallic", "periodic", "metallic"), True),
        ("partial_presence_per_target", ("B", "D"), True,
         ("periodic", "metallic", "periodic"), True),
    )
    rows: List[Dict[str, Any]] = []
    payload["legs"]["products"] = rows
    for index, (label, sides, mixed, boundaries, partial) in enumerate(products, 1):
        result = execute(sides=sides, mixed=mixed, boundaries=boundaries,
                         seed=30100 + index, cycles=4, partial=partial)
        fields, pml = build(sides=sides, mixed=mixed, boundaries=boundaries,
                            seed=30100 + index, partial=partial)
        forms = {}
        for slot in _sub_steps(sides):
            forms.update(_form_counts(fields, pml, slot))
        row = {"label": label, "sides": list(sides), "mixed": mixed,
               "boundaries": list(boundaries), "partial": partial, "forms": forms,
               **result}
        rows.append(row)
        payload["legs"]["products"] = rows
        kit.save(payload, out)
        differences = sum(item["differing_words"] for item in result["rows"])
        moved = sum(item["moved_words"] for item in result["rows"])
        compared = sum(item["words_compared"] for item in result["rows"])
        kit.log(f"[product] {index}/{len(products)} {label:<22} "
                f"moved={moved:<7} differing={differences:<4} words={compared} "
                f"forms_zero={sorted(k for k, v in forms.items() if not v)}")
        assert result["passed"], row
    # THE FORM CENSUS, over the products: every conductive form and both lossless
    # forms are reached on some product, or a mutation of that form is vacuous.
    reached: Dict[str, int] = {}
    for row in rows:
        for key, count in row["forms"].items():
            form = key.split("/", 1)[1]
            reached[form] = reached.get(form, 0) + count
    payload["legs"]["form_census"] = reached
    kit.save(payload, out)
    kit.log(f"[product] form census {reached}")
    for form in ("both", "dsigu_only", "dsig_only", "neither", "lossless_recurrence",
                 "lossless_dsig_only", "unkept_recurrence", "unkept_dsig_only"):
        kit.assert_census_floor(reached.get(form, 0), f"form {form}")


def leg_mutations(payload: Dict[str, Any], out: str) -> None:
    """Faults in the active-PML conductive recurrence must be byte-visible.

    Every source row is compiled and launched on the B side; a must-catch row must
    diverge and the declared null must not. The record rows launch the SHIPPED kernel
    bound to a wrong record, against the reference stepped on the true one.
    """
    rows: List[Dict[str, Any]] = []
    payload["legs"]["mutations"] = rows
    boundaries = ("metallic", "periodic", "metallic")
    table = _mutation_table()
    total = len(table) + len(RECORD_MUTATIONS) + len(PRESENCE_MUTATIONS)

    def score(label: str, result: Dict[str, Any], must_catch: bool, why: str,
              index: int) -> None:
        differing = sum(item["differing_words"] for item in result["rows"])
        # ``execute`` stops at the *first* mismatch so it preserves the smallest
        # discriminating counterexample.  The successful path has ``launches``;
        # the caught path records it in its final per-cycle row instead.
        launches = result.get("launches", {}).get(
            "step_B", result["rows"][-1]["launches"])
        row = {"label": label, "differing_words": differing, "launches": launches,
               "caught": differing > 0, "must_catch": must_catch, "why": why,
               "result": result}
        rows.append(row)
        payload["legs"]["mutations"] = rows
        kit.save(payload, out)
        kit.log(f"[mutation] {index}/{total} {label:<46} launches={launches} "
                f"differing={differing} must_catch={must_catch}")
        assert launches >= 1, row
        if must_catch:
            assert differing > 0, row
        else:
            assert differing == 0, row

    for index, (label, entry) in enumerate(table.items(), 1):
        function = compile_source(entry["source"]).conductive_pml_curl_step
        result = execute(
            sides=("B",), mixed=entry["mixed"], boundaries=boundaries,
            seed=30200 + index, cycles=2,
            functions={"step_B": {shaders.CONTRACT_OFF: function}},
            record_tally=False, partial=bool(entry.get("partial")),
        )
        score(label, result, entry["must_catch"], entry["why"], index)

    for offset, (label, (mutate, why)) in enumerate(RECORD_MUTATIONS.items(), 1):
        fields, pml = build(sides=("B",), mixed=False, boundaries=boundaries,
                            seed=30300 + offset)
        truth = layer_record(fields.grid, pml)
        wrong = mutate(truth)
        assert not np.array_equal(wrong, truth), f"{label} changed no record word"
        result = execute(sides=("B",), mixed=False, boundaries=boundaries,
                         seed=30300 + offset, cycles=2, record_tally=False,
                         record=wrong)
        score(label, result, True, why, len(table) + offset)

    for offset, (label, (words, why)) in enumerate(PRESENCE_MUTATIONS.items(), 1):
        fields, pml = build(sides=("B",), mixed=False, boundaries=boundaries,
                            seed=30400 + offset, partial=True)
        truth = family.presence_for(fields, pml, "step_B",
                                    layer_record(fields.grid, pml))
        assert tuple(words) != tuple(truth), f"{label} changed no presence word"
        result = execute(sides=("B",), mixed=False, boundaries=boundaries,
                         seed=30400 + offset, cycles=2, record_tally=False,
                         partial=True, presence=words)
        score(label, result, True, why,
              len(table) + len(RECORD_MUTATIONS) + offset)


def leg_refusals(payload: Dict[str, Any], out: str) -> None:
    """The conductive arm owns only its target side and rejects the incumbent."""
    fields, pml = build(sides=("B",), mixed=False,
                        boundaries=("periodic", "periodic", "periodic"), seed=30301)
    b = family.metal_conductive_pml_curl_coverage(fields, pml, "step_B", Residency())
    d = family.metal_conductive_pml_curl_coverage(fields, pml, "step_D", Residency())
    ordinary = launch.plan_pml_curl(fields, pml, "step_B", Residency(),
                                    chunk_bounds=layer_record(fields.grid, pml))
    inactive = PML(grid=fields.grid, thickness=0)
    inactive_verdict = family.metal_conductive_pml_curl_coverage(
        fields, inactive, "step_B", Residency())
    # A chunk that keeps no conductivity: MEEP steps the lossless forms there, and the
    # presence words say which chunks those are, so the sub-step is SERVED; a record
    # whose chunk codes cannot carry the array path's presence is refused by name.
    partial, partial_pml = build(sides=(), mixed=False,
                                 boundaries=("periodic", "periodic", "periodic"),
                                 seed=30302)
    sigma = np.zeros(partial.grid.shape, dtype=np.float32)
    sigma[4:6] = np.float32(0.15)
    partial.set_b_conductivity({"Bx": sigma, "By": None, "Bz": None})
    kept = ChunkLayout.one_rank(partial.grid, partial_pml).conductive_chunks(
        sigma, (0, 1, 1))
    presence = family.metal_conductive_pml_curl_coverage(
        partial, partial_pml, "step_B", Residency())
    wrong = np.array(layer_record(partial.grid, partial_pml), copy=True)
    wrong[record_index(False, 0, 1)] += 2
    uncarried = family.presence_reasons(partial, partial_pml, "step_B",
                                        chunk_bounds=wrong)
    row = {
        "b_covered": b.covered,
        "d_reasons": list(d.reasons),
        "ordinary_plan": ordinary is not None,
        "inactive_reasons": list(inactive_verdict.reasons),
        "presence_kept": list(kept),
        "presence_reasons": list(presence.reasons),
        "uncarried_presence_reasons": uncarried,
    }
    payload["legs"]["refusals"] = row
    kit.save(payload, out)
    kit.log("[refusal] B and a partial presence admitted; D-only, ordinary-PML "
            "overlap, inactive-PML and an uncarried-presence record refused")
    assert any(kept) and not all(kept), kept
    assert presence.covered, presence.reasons
    assert uncarried and "conductivity presence is refused" in uncarried[0], uncarried
    assert b.covered, b.reasons
    assert not d.covered and any("no step_D target" in reason for reason in d.reasons)
    assert ordinary is None
    assert not inactive_verdict.covered
    assert any("active PML" in reason for reason in inactive_verdict.reasons)


def main(argv: Sequence[str]) -> int:
    parser = kit.argument_parser(__doc__ or "")
    args = parser.parse_args(list(argv))
    out = os.path.abspath(args.out)
    started = time.time()
    payload: Dict[str, Any] = {
        "gate": "metal_conductive_pml",
        "family": family.FAMILY,
        "environment": kit.environment_stamp(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "legs": {},
    }
    kit.save(payload, out)
    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device available")
    reasons.extend(subnormal.mps_policy_reasons())
    if reasons:
        return kit.cannot_certify(payload, out, reasons)

    kit.provenance(
        os.path.dirname(out),
        {
            "conductive_pml.py": str(Path(family.__file__).resolve()),
            "stepping.py": str((API_ROOT / "meep_gpu/stepping.py").resolve()),
            "gate": str(Path(__file__).resolve()),
        },
        kernel_sources={
            f"{codes}/{backward}/{conductive}": family.conductive_pml_curl_source(
                codes, backward, conductive)
            for codes, backward, conductive in (
                ((0, 0, 0), False, (True, True, True)),
                ((1, 0, 1), False, (True, False, False)),
                ((0, 1, 0), True, (False, False, True)),
            )
        },
        name="provenance_conductive_pml.json",
    )
    ran = kit.run_legs(
        (("products", leg_products), ("mutations", leg_mutations),
         ("refusals", leg_refusals)),
        payload, out, kit.wanted_legs(args.legs))
    payload["totals"] = {
        "uint32_words_compared": TALLY.compared,
        "uint32_words_differing": TALLY.differing,
    }
    kit.save(payload, out)
    return kit.summarize(
        payload, out,
        claim=("the real conductive active-PML B/D curl kernels reproduce the "
               "array path word for word across side-specific and per-target "
               "conductivity products, with the PML and conductivity histories "
               "preserved across launches"),
        scope=("real float32 Cartesian fields, active PML, periodic/metallic "
               "boundaries, no fold, no Bloch phase, no poles, no nonlinear "
               "material, and no driver dispatch claim"),
        stated_weakness=("behavioural MPS evidence only: compile_shader exposes "
                         "no generated AIR or ISA; this is a B/D sub-step gate, "
                         "not a complete driver or throughput measurement"),
        started=started, legs_run=ran, compared=TALLY.compared,
        certified=TALLY.differing == 0,
        extra={"products": len(payload["legs"].get("products", [])),
               "mutations": len(payload["legs"].get("mutations", [])),
               "form_census": payload["legs"].get("form_census")},
    )


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
