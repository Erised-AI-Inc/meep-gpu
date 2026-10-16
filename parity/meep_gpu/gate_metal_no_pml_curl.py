#!/usr/bin/env python3
"""Byte gate for the real no-absorber Metal curl family.

The oracle is assembled exclusively from ``meep_gpu.stepping``'s term table,
stencil gather, grouped curl (``_curl_from_operands`` with ``curl_presence``) and
ownership mask.  Each product row is written to JSONL and flushed before the next
row begins, so an interrupted MPS run retains its completed evidence.

Every case binds the chunk record of a run with no absorber (:func:`case_record`:
no membership run, and the partner-presence bit of every axis that is not
invariant), and the oracle reads its presence from the same invariance. The cases
cover each of MEEP's stencil forms: a 3-D grid (both partners), a 3-D grid with a
one-cell axis that is NOT invariant (the full bracket over a one-cell wrap), a 2-D
grid (z invariant: one partner absent on Bx/Dx and on By/Dy, the second and the
first respectively) and a 1-D grid (x and y invariant: one partner on Bx/By, none
on Bz).

A planted-zero leg steps each grid from a state whose field words are all +0 or -0
(``build_arrays(..., planted=True)``): there every difference the stencil takes is a
signed zero, which is where MEEP's ``(-dtdx) * (g2s - g2)`` and the reversed
difference ``dtdx * (g2 - g2s)`` part, and random states never reach it.

The mutations are scored per row: a must-catch row passes when a word differs, a
declared null when none does. ``previous_curl_grouping`` plants the package's
previous pairwise bracket ``(g1s - g1) + (g2 - g2s)``, which the shipped stencil
replaced; the absent-partner rows plant the full bracket where a partner is absent,
and ``presence_word_ignored`` binds a record that marks every axis present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
API_ROOT = Path(__file__).resolve().parents[2]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.chunk_map import (  # noqa: E402
    RECORD_PRESENCE_WORD,
    RECORD_WORDS,
    ChunkLayout,
    chunk_bounds_record,
)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import no_pml_curl as family  # noqa: E402
from meep_gpu.metal_kernels import launch as metal_launch  # noqa: E402
from meep_gpu.metal_kernels import shaders, subnormal  # noqa: E402
from meep_gpu.metal_kernels.device import (  # noqa: E402
    Residency,
    compile_source,
    metal_frontend_version,
)
from meep_gpu.pml import PML  # noqa: E402


class OracleGrid:
    """The boundary queries the stepping oracle's ownership mask and curl presence read."""

    def __init__(self, shape: Sequence[int], codes: Sequence[int],
                 invariant: Sequence[bool] = (False, False, False)) -> None:
        self.shape = tuple(int(value) for value in shape)
        self._codes = tuple(int(value) for value in codes)
        self._invariant = tuple(bool(value) for value in invariant)

    def is_metallic(self, axis: int) -> bool:
        return self._codes[axis] == shaders.METALLIC

    def is_mirrored(self, axis: int) -> bool:  # noqa: ARG002
        return False

    def is_axis(self, axis: int) -> bool:  # noqa: ARG002
        return False

    def is_invariant(self, axis: int) -> bool:
        return self._invariant[axis]


def case_record(invariant: Sequence[bool]) -> np.ndarray:
    """The chunk record of a run with no absorber on a grid with this invariance.

    No membership run on either lattice (no chunk carries a layer) and the
    partner-presence bit of every axis that is not invariant: what
    ``chunk_map.chunk_bounds_record(ChunkLayout.one_rank(grid, None))`` returns for
    an engine grid, which :func:`run_engine_composition` checks.
    """
    record = np.zeros(RECORD_WORDS, dtype=np.int32)
    record[RECORD_PRESENCE_WORD] = sum(1 << axis for axis in range(3)
                                       if not bool(invariant[axis]))
    return record


def words(array: Any) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32).reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_provenance() -> Dict[str, Any]:
    """Bind the result to every source specialization compiled by this gate."""
    specializations: Dict[str, str] = {}
    for codes in ((0, 0, 0), (1, 0, 1), (1, 1, 1)):
        for sub_step, derive in (("step_B", True), ("step_B", False),
                                 ("step_D", False)):
            backward = bool(family.SUB_STEPS[sub_step]["backward"])
            label = f"{sub_step}/derive-{int(derive)}/{codes}"
            source = family.plain_curl_source(codes, backward, derive)
            specializations[label] = sha256_bytes(source.encode("utf-8"))
    mutation_hashes = {
        name: sha256_bytes(source.encode("utf-8"))
        for name, source, _derive in mutation_sources()
    }
    return {
        "files": {
            "family": sha256_file(Path(family.__file__).resolve()),
            "launch": sha256_file(Path(metal_launch.__file__).resolve()),
            "residency_coverage": sha256_file(
                API_ROOT / "meep_gpu" / "metal_kernels" / "coverage.py"),
            "gate": sha256_file(Path(__file__).resolve()),
        },
        "specialized_metal_sources": specializations,
        "mutation_metal_sources": mutation_hashes,
    }


def build_arrays(shape: Sequence[int], seed: int,
                 planted: bool = False) -> Dict[str, np.ndarray]:
    """Random field words, or with ``planted`` a +0/-0 lattice (inverse epsilon stays random)."""
    rng = np.random.default_rng(seed)
    if planted:
        arrays = {
            name: np.where(rng.random(shape) < 0.5, np.float32(0.0),
                           np.float32(-0.0)).astype(np.float32)
            for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez")
        }
    else:
        arrays = {
            name: rng.uniform(-0.45, 0.45, shape).astype(np.float32)
            for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez")
        }
    for name in ("Ex", "Ey", "Ez"):
        arrays["inv_eps_" + name] = rng.uniform(0.25, 0.85, shape).astype(np.float32)
    return arrays


def oracle_once(arrays: Dict[str, np.ndarray], sub_step: str,
                codes: Sequence[int], dtdx: float, derive: bool,
                invariant: Sequence[bool] = (False, False, False)) -> None:
    spec = family.SUB_STEPS[sub_step]
    if derive:
        snapshot: Dict[str, np.ndarray] = {}
        for e_name, d_name in zip(("Ex", "Ey", "Ez"), spec["displacement"]):
            value = np.empty_like(arrays[d_name])
            np.multiply(arrays[d_name], arrays["inv_eps_" + e_name], out=value)
            snapshot[e_name] = value
    elif sub_step == "step_D":
        # ``Fields.get_H`` serves B directly when no absorber is active, but the
        # D term table names the logical H components. Preserve both facts rather
        # than renaming the term table or allocating stored H arrays.
        snapshot = {h_name: arrays[b_name]
                    for h_name, b_name in zip(("Hx", "Hy", "Hz"),
                                              spec["sources"])}
    else:
        snapshot = {name: arrays[name] for name in spec["sources"]}
    boundaries = tuple("metallic" if code else "periodic" for code in codes)
    terms = stepping.B_CURL_TERMS if sub_step == "step_B" else stepping.D_CURL_TERMS
    grid = OracleGrid(arrays[spec["targets"][0]].shape, codes, invariant)
    for term in terms:
        operands = stepping._curl_operands(  # noqa: SLF001 - this is the oracle
            np, snapshot, term, boundaries,
            (None, None, None), (None, None, None),
            backward=bool(spec["backward"]),
        )
        curl = stepping._curl_from_operands(  # noqa: SLF001
            operands, dtdx, stepping.curl_presence(grid, term))
        stepping._mask_non_owned_cells(curl, grid, term.iyee)  # noqa: SLF001
        arrays[term.target] -= curl


def run_plan(arrays: Dict[str, np.ndarray], sub_step: str,
             codes: Sequence[int], dtdx: float, derive: bool, cycles: int,
             record: np.ndarray,
             functions: Dict[str, Any] | None = None) -> Tuple[Any, Residency]:
    residency = Residency()
    plan = family.plan_metal_plain_curl_from_arrays(
        sub_step, arrays, codes, dtdx, residency, derive=derive,
        functions=functions, chunk_bounds=record,
    )
    residency.sync_in()
    for _ in range(cycles):
        plan.run()
    residency.sync_out(plan.volumes)
    return plan, residency


def emit(handle, payload: Dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    label = payload.get("label", payload.get("leg", "row"))
    print(f"{payload['index']}/{payload['total']} {label}: "
          f"diff={payload.get('differing_words', '-')}", flush=True)


#: The case grids: (shape, invariant axes, boundary triples). An invariant axis is
#: periodic (``Grid`` refuses a metallic one), so its triples keep it periodic.
CASE_GRIDS: Tuple[Tuple[Tuple[int, int, int], Tuple[bool, bool, bool],
                        Tuple[Tuple[int, int, int], ...]], ...] = (
    # 3-D: every partner present.
    ((17, 13, 9), (False, False, False), ((0, 0, 0), (1, 0, 1), (1, 1, 1))),
    # 3-D with a one-cell y that is NOT invariant (cell_size.y = 0 at dimensions=3):
    # the full bracket, over a one-cell wrap or a metallic one-cell axis.
    ((19, 1, 23), (False, False, False), ((0, 0, 0), (1, 0, 1), (1, 1, 1))),
    # 2-D: z invariant, so Bx/Dx lose g2 and By/Dy lose g1.
    ((17, 13, 1), (False, False, True), ((0, 0, 0), (1, 1, 0))),
    # 1-D: x and y invariant, so Bx/Dx lose g1, By/Dy lose g2 and Bz/Dz lose both.
    ((1, 1, 29), (True, True, False), ((0, 0, 0), (0, 0, 1))),
)

#: The mutation legs' case per grid kind: (shape, codes, invariant, planted state).
MUTATION_GRIDS: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int],
                                Tuple[bool, bool, bool], bool]] = {
    "3d": ((17, 13, 9), (1, 0, 1), (False, False, False), False),
    "2d": ((17, 13, 1), (1, 1, 0), (False, False, True), False),
    "2d_planted": ((17, 13, 1), (1, 1, 0), (False, False, True), True),
}


def planted_cases():
    """One planted-zero row per case grid and sub-step: (shape, codes, sub_step, derive, invariant)."""
    cases = []
    for shape, invariant, boundaries in CASE_GRIDS:
        codes = boundaries[-1]
        cases.append((shape, codes, "step_B", True, invariant))
        cases.append((shape, codes, "step_D", False, invariant))
    return tuple(cases)


def product_cases():
    courants = (0.35, 0.5)
    cases = []
    for shape, invariant, boundaries in CASE_GRIDS:
        for codes in boundaries:
            for dtdx in courants:
                cases.append((shape, codes, dtdx, "step_B", True, invariant))
                cases.append((shape, codes, dtdx, "step_B", False, invariant))
                cases.append((shape, codes, dtdx, "step_D", False, invariant))
    return tuple(cases)


def needle(source: str, old: str, new: str, count: int = 1) -> str:
    """Replace ``old`` exactly ``count`` times, or refuse: an unarmed mutation would
    report its defect as uncaught (or a declared null as held) for no reason."""
    hits = source.count(old)
    if hits != count:
        raise AssertionError(
            f"the mutation needle {old!r} matches {hits} times, expected {count}")
    return source.replace(old, new)


#: The both-partner brackets, shipped and the package's previous pairwise grouping.
_BRACKETS = (
    ("(dtdx * (((c_y - c) + b) - b_z))", "(dtdx * ((c_y - c) + (b - b_z)))"),
    ("(dtdx * (((a_z - a) + c) - c_x))", "(dtdx * ((a_z - a) + (c - c_x)))"),
    ("(dtdx * (((b_x - b) + a) - a_y))", "(dtdx * ((b_x - b) + (a - a_y)))"),
)


def mutation_sources() -> Tuple[Tuple[str, str, bool], ...]:
    """(name, source, derive) per source mutation, on the (1, 0, 1) forward specialisation."""
    base = family.plain_curl_source((1, 0, 1), False, True)
    add = base
    for index in range(3):
        add = needle(add,
                     f"f{index}[ii] = f{index}[ii] - curl{index};",
                     f"f{index}[ii] = f{index}[ii] + curl{index};")
    previous = base
    for shipped, old in _BRACKETS:
        previous = needle(previous, shipped, old)
    no_derive = family.plain_curl_source((1, 0, 1), False, False)
    # g2 absent on target 0 (Bx without its z partner): the full bracket instead.
    full_bracket = needle(base, ": (dtdx * (c_y - c)))",
                          ": (dtdx * (((c_y - c) + b) - b_z)))")
    # g1 absent on target 1 (By without its z partner): dtdx instead of -dtdx.
    negation = needle(base, "(has_x ? ((-dtdx) * (c_x - c)) : 0.0f)",
                      "(has_x ? (dtdx * (c_x - c)) : 0.0f)")
    # The same form as the reversed difference. In IEEE arithmetic it equals MEEP's on
    # every word but the signed zero of equal neighbours, where (-dtdx) * (+0) is -0 and
    # dtdx * (+0) is +0; compiled for this platform it also differs on random words
    # (MEASURED: 254 words on the random 2-D state), so it is scored on the planted
    # state only, where either reason catches it.
    reversed_difference = needle(base, "(has_x ? ((-dtdx) * (c_x - c)) : 0.0f)",
                                 "(has_x ? (dtdx * (c - c_x)) : 0.0f)")
    return (("add_instead_of_subtract", add, True),
            ("previous_curl_grouping", previous, True),
            ("drop_derived_epsilon", no_derive, True),
            ("full_bracket_over_an_absent_partner", full_bracket, True),
            ("absent_first_partner_keeps_the_sign_of_dtdx", negation, True),
            ("absent_first_partner_as_the_reversed_difference", reversed_difference,
             True))


#: The mutation rows: (row, source mutation or None, grid kind, must_catch,
#: record override). A must-catch row passes when a word differs and a declared
#: null when none does. The null rows are the reason the absent-partner forms need
#: the 2-D rows at all: on a grid with every partner present the absent arms never
#: run, so the same planted text is invisible there.
MUTATION_ROWS: Tuple[Tuple[str, Any, str, bool, Any], ...] = (
    ("add_instead_of_subtract", "add_instead_of_subtract", "3d", True, None),
    ("previous_curl_grouping", "previous_curl_grouping", "3d", True, None),
    ("drop_derived_epsilon", "drop_derived_epsilon", "3d", True, None),
    ("full_bracket_over_an_absent_partner", "full_bracket_over_an_absent_partner",
     "2d", True, None),
    ("full_bracket_over_an_absent_partner_on_a_resolved_grid",
     "full_bracket_over_an_absent_partner", "3d", False, None),
    ("absent_first_partner_keeps_the_sign_of_dtdx",
     "absent_first_partner_keeps_the_sign_of_dtdx", "2d", True, None),
    ("absent_first_partner_keeps_the_sign_of_dtdx_on_a_resolved_grid",
     "absent_first_partner_keeps_the_sign_of_dtdx", "3d", False, None),
    # On the planted state, whose equal neighbours give the signed-zero case.
    ("absent_first_partner_as_the_reversed_difference",
     "absent_first_partner_as_the_reversed_difference", "2d_planted", True, None),
    # The kernel is shipped; the RECORD lies: every axis marked present on the 2-D
    # grid, so the kernel takes the full bracket where MEEP drops a partner.
    ("presence_word_ignored", None, "2d", True, (False, False, False)),
)


#: The engine legs' grids: (label, cell_size, dimensions).
ENGINE_GRIDS: Tuple[Tuple[str, Tuple[float, float, float], int], ...] = (
    ("3d", (1.25, 1.0, 0.75), 3),
    ("2d", (1.25, 1.0, 0.0), 2),
)


def build_engine(seed: int, cell_size: Sequence[float] = (1.25, 1.0, 0.75),
                 dimensions: int = 3) -> Tuple[Fields, PML]:
    grid = Grid(resolution=8.0, cell_size=tuple(cell_size),
                courant=0.35, xp=np, dimensions=int(dimensions))
    fields = Fields(grid=grid)
    fields.set_epsilon_volumes(
        {name: np.full(grid.shape, value, np.float32)
         for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))},
        {name: np.full(grid.shape, np.float32(1.0 / value), np.float32)
         for name, value in zip(("Ex", "Ey", "Ez"), (2.0, 2.5, 3.0))},
    )
    rng = np.random.default_rng(seed)
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        getattr(fields, name)[...] = rng.uniform(
            -0.4, 0.4, grid.shape).astype(np.float32)
    return fields, PML(grid=grid, thickness=0)


def run_engine_composition(cycles: int, cell_size: Sequence[float] = (1.25, 1.0, 0.75),
                           dimensions: int = 3) -> Dict[str, Any]:
    reference, reference_pml = build_engine(10401, cell_size, dimensions)
    actual, actual_pml = build_engine(10401, cell_size, dimensions)
    residency = Residency()
    plan = metal_launch.plan_step(
        actual, actual_pml, residency=residency, sources=())
    expected = {"step_B": "no-PML curl", "update_H": "no-PML null",
                "step_D": "no-PML curl", "update_E": "no-PML null"}
    if plan.selected != expected or not plan.residency.covered:
        return {"differing_words": -1, "moved_words": 0,
                "selected": plan.selected,
                "residency_reasons": list(plan.residency.reasons),
                "passed": False}
    # The composer's record is the layout's own, and equals the gate's construction
    # for this grid's invariance: the bare-array legs bind what the engine binds.
    grid = actual.grid
    invariant = tuple(bool(grid.is_invariant(axis)) for axis in range(3))
    engine_record = chunk_bounds_record(ChunkLayout.one_rank(grid, actual_pml))
    bound = {slot: np.asarray(plan.plans[slot].chunk_bounds).tolist()
             for slot in ("step_B", "step_D")}
    record_agrees = (np.array_equal(engine_record, case_record(invariant))
                     and all(words == case_record(invariant).tolist()
                             for words in bound.values()))
    before = {name: np.array(getattr(actual, name), copy=True)
              for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz")}
    residency.sync_in()
    for _ in range(cycles):
        stepping.step_B(reference, reference_pml)
        stepping.update_H(reference, reference_pml)
        stepping.step_D(reference, reference_pml)
        stepping.update_E(reference, reference_pml)
        for slot in ("step_B", "update_H", "step_D", "update_E"):
            plan.plans[slot].run()
    residency.sync_out()
    names = tuple(before)
    diff = sum(differing(getattr(reference, name), getattr(actual, name))
               for name in names)
    moved = sum(differing(before[name], getattr(actual, name)) for name in names)
    census = sum(subnormal.census(getattr(reference, name)) for name in names)
    launches = {slot: getattr(plan.plans[slot], "launches",
                              getattr(plan.plans[slot], "runs", None))
                for slot in expected}
    return {"differing_words": diff, "moved_words": moved,
            "reference_subnormals": census, "selected": plan.selected,
            "residency_reasons": list(plan.residency.reasons),
            "launches": launches, "shape": list(grid.shape),
            "invariant": list(invariant), "bound_records": bound,
            "record_agrees": bool(record_agrees),
            "passed": diff == 0 and moved > 0 and census == 0 and record_agrees
            and all(value == cycles for value in launches.values())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="Final JSON artifact; a sibling .jsonl is incremental")
    parser.add_argument("--cycles", type=int, default=4)
    args = parser.parse_args()

    import torch

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available; this gate must run on an Apple GPU")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    products = product_cases()
    mutations = {name: (source, derive) for name, source, derive in mutation_sources()}
    planted = planted_cases()
    total = len(products) + len(planted) + len(MUTATION_ROWS) + len(ENGINE_GRIDS)
    rows = []
    started = time.perf_counter()
    with jsonl.open("w", encoding="utf-8") as stream:
        for index, (shape, codes, dtdx, sub_step, derive, invariant) in enumerate(
                products, 1):
            label = (f"product/{sub_step}/derive-{int(derive)}/{shape}/{codes}/{dtdx}"
                     f"/invariant-{''.join(str(int(v)) for v in invariant)}")
            reference = build_arrays(shape, 8100 + index)
            actual = {name: np.array(value, copy=True)
                      for name, value in reference.items()}
            before = {name: np.array(actual[name], copy=True)
                      for name in family.SUB_STEPS[sub_step]["targets"]}
            for _ in range(args.cycles):
                oracle_once(reference, sub_step, codes, dtdx, derive, invariant)
            plan, _ = run_plan(actual, sub_step, codes, dtdx, derive, args.cycles,
                               case_record(invariant))
            targets = family.SUB_STEPS[sub_step]["targets"]
            diff = sum(differing(reference[name], actual[name]) for name in targets)
            moved = sum(differing(before[name], actual[name]) for name in targets)
            census = sum(subnormal.census(reference[name]) for name in targets)
            row = {"index": index, "total": total, "leg": "product", "label": label,
                   "shape": shape, "codes": codes, "dtdx": dtdx,
                   "sub_step": sub_step, "derive": derive,
                   "invariant": list(invariant),
                   "presence": int(case_record(invariant)[RECORD_PRESENCE_WORD]),
                   "cycles": args.cycles, "launches": plan.launches,
                   "differing_words": diff, "moved_words": moved,
                   "reference_subnormals": census,
                   "passed": diff == 0 and moved > 0 and census == 0
                   and plan.launches == args.cycles}
            rows.append(row)
            emit(stream, row)

        for number, (shape, codes, sub_step, derive, invariant) in enumerate(planted, 1):
            index = len(products) + number
            label = (f"planted_zero/{sub_step}/derive-{int(derive)}/{shape}/{codes}"
                     f"/invariant-{''.join(str(int(v)) for v in invariant)}")
            reference = build_arrays(shape, 8700 + number, planted=True)
            actual = {name: np.array(value, copy=True)
                      for name, value in reference.items()}
            dtdx = 0.35
            targets = family.SUB_STEPS[sub_step]["targets"]
            before = {name: np.array(actual[name], copy=True) for name in targets}
            for _ in range(args.cycles):
                oracle_once(reference, sub_step, codes, dtdx, derive, invariant)
            plan, _ = run_plan(actual, sub_step, codes, dtdx, derive, args.cycles,
                               case_record(invariant))
            diff = sum(differing(reference[name], actual[name]) for name in targets)
            moved = sum(differing(before[name], actual[name]) for name in targets)
            negative_zero = sum(int(np.count_nonzero(words(reference[name])
                                                     == np.uint32(0x80000000)))
                                for name in targets)
            row = {"index": index, "total": total, "leg": "planted_zero",
                   "label": label, "shape": shape, "codes": codes,
                   "sub_step": sub_step, "derive": derive,
                   "invariant": list(invariant), "cycles": args.cycles,
                   "launches": plan.launches, "differing_words": diff,
                   "moved_words": moved, "reference_negative_zeros": negative_zero,
                   "passed": diff == 0 and plan.launches == args.cycles}
            rows.append(row)
            emit(stream, row)

        for number, (grid_label, cell_size, dimensions) in enumerate(ENGINE_GRIDS, 1):
            engine = run_engine_composition(args.cycles, cell_size, dimensions)
            engine.update({"index": len(products) + len(planted) + number,
                           "total": total,
                           "leg": "engine",
                           "label": f"complete_no_pml_arithmetic/{grid_label}",
                           "cycles": args.cycles})
            rows.append(engine)
            emit(stream, engine)

        offset = len(products) + len(planted) + len(ENGINE_GRIDS)
        for number, (name, mutation, grid_kind, must_catch, record_invariant) in \
                enumerate(MUTATION_ROWS, 1):
            index = offset + number
            shape, codes, invariant, planted_state = MUTATION_GRIDS[grid_kind]
            dtdx, sub_step = 0.35, "step_B"
            source, derive = (mutations[mutation] if mutation is not None
                              else (family.plain_curl_source(codes, False, True), True))
            reference = build_arrays(shape, 9200 + number, planted=planted_state)
            actual = {key: np.array(value, copy=True)
                      for key, value in reference.items()}
            oracle_once(reference, sub_step, codes, dtdx, derive, invariant)
            function = compile_source(source).no_pml_curl_step
            record = case_record(invariant if record_invariant is None
                                 else record_invariant)
            plan, _ = run_plan(actual, sub_step, codes, dtdx, derive, 1, record,
                               {shaders.CONTRACT_OFF: function})
            targets = family.SUB_STEPS[sub_step]["targets"]
            diff = sum(differing(reference[target], actual[target])
                       for target in targets)
            caught = diff > 0
            row = {"index": index, "total": total, "leg": "mutation",
                   "label": name, "grid": grid_kind, "shape": shape,
                   "planted": planted_state,
                   "codes": codes, "invariant": list(invariant),
                   "bound_presence": int(record[RECORD_PRESENCE_WORD]),
                   "must_catch": must_catch, "differing_words": diff,
                   "launches": plan.launches, "caught": caught,
                   "passed": caught == must_catch and plan.launches == 1}
            rows.append(row)
            emit(stream, row)

    result = {
        "verdict": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "product_rows": len(products),
        "planted_zero_rows": len(planted),
        "mutation_rows": len(MUTATION_ROWS),
        "engine_rows": len(ENGINE_GRIDS),
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
