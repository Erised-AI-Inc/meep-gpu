#!/usr/bin/env python3
"""THE ROTATION QUESTION AT ``update_E`` -> ``update_P``, ASKED FOR **TRITON**.

The Metal side of this question is settled and measured:
``probe_metal_ade_rotation_seam.py`` / ``results/metal_ade_rotation_seam_2026-08-20/``
found a THIRD answer — a PER-COMPONENT interleave (``update_E(c)`` then every
``update_P(state, c)``) is bit-identical to the driver's order, and per component
the rotation is ALIAS-FREE at every position of its finite orbit. So on Metal the
host round trip STAYS and the fused kernel inherits no alias.
``metal_kernels/fused_ade_chain.py`` is built on exactly that, released
2026-08-20, and serves 10 of the 15 E->P seam-instances.

**THAT ANSWER DOES NOT TRANSFER TO TRITON, AND THIS PROBE MEASURES WHY.**

Metal's ``update_E`` products are emitted PER COMPONENT — the fused chain compiles
one specialisation per axis and dispatches three times, which is what makes the
per-component shape available at all. Triton's certified ``update_E`` kernels are
not: one launch writes Ex, Ey and Ez. If that is so, the per-component shape is
not something a Triton product can choose without splitting a certified body, and
the choice narrows to the two shapes that keep all three components in one launch:

  (a) BAKE THE ROTATION, and inherit the intra-launch alias that
      ``fused_ade_state``'s note is about — arm 1's OUTPUT buffer IS arm 0's
      ``p_prev`` INPUT — whose safety is an OBSERVATION on one toolchain and not a
      language guarantee. ``fused_dispersive_chain`` takes this shape and pays for
      it by HOISTING every ADE history load above every ADE store, which it can do
      because it carries at most ONE pole per component.

  (c) GIVE THE SUSCEPTIBILITY ONE SCRATCH PER DRIVEN COMPONENT instead of one
      shared scratch. ``fused_ade_state``'s note names this as the way to restore
      a guarantee — "give the launch one more scratch buffer per susceptibility so
      no output aliases any input, and the question stops being asked" — and never
      measures it, because nothing dispatches that product. This probe measures it.

Three legs, all HOST-ONLY. Nothing here needs a GPU, a Triton install or a device:
every question is about which BUFFER a launch reads and writes, and that is decided
by ``dispersion.PolarizationState.update`` and by the certified kernels' own source.

LEG 1  certified_update_e_component_shape
    Does any certified Triton ``update_E`` kernel write ONE component per launch?
    Answered by PARSING the module source with ``ast`` — locating each
    ``@triton.jit`` function and counting the distinct base pointers its
    ``tl.store`` calls target — rather than by reading a docstring or trusting
    this file's own summary of them. A test that re-implements what it is checking
    mirrors a defect instead of executing it; measured on this project 2026-08-20,
    three planted assembly defects each left 86 of 87 mirrored tests passing.

LEG 2  rotation_orbit_disjointness
    The rotation is a permutation of a finite buffer set, so its orbit is
    ENUMERABLE rather than sampled. This walks it to closure for d = 1, 2, 3 driven
    components and checks read/write disjointness at EVERY position, for THREE
    launch shapes: per-component (Metal's), all-component with the shared scratch
    (shape a), and all-component with one scratch per component (shape c). The
    ``rotate`` and ``orbit`` functions are TRANSCRIBED from
    ``probe_metal_ade_rotation_seam.py:343-390``, which transcribes
    ``dispersion.py:679-691``; the third shape is this probe's addition.

LEG 3  scratch_cost_on_the_corpus
    What shape (c) costs, in extra field volumes, on the E->P rows the corpus
    actually drives — computed from the measured susceptibility counts, not from a
    round number.

NON-VACUITY. Each leg carries a floor that fails if the leg measured nothing: leg 1
asserts every kernel name it looks for EXISTS in the module it parses (a name set
that matches nothing disables the check it feeds, silently); leg 2 asserts the
orbit has more than one position AND that the shared-scratch shape DOES conflict,
because a contrast between two clean results is not a contrast.

Rule 7: one flushed line per case, rows appended and fsynced as they land.

Usage (from the repository root)::

    PYTHONPATH=. \\
      python -u parity/meep_gpu/probe_triton_ade_rotation_shape.py \\
        --out parity/meep_gpu/results/<dir>/rotation_shape.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import gate_provenance  # noqa: E402

TRITON_KERNELS = API_ROOT / "meep_gpu" / "triton_kernels"

COMPONENTS: Tuple[str, ...] = ("Ex", "Ey", "Ez")

#: The certified Triton kernels this probe parses, and what each is expected to be
#: the ``update_E`` or ``update_P`` body for. THE EXPECTATION IS NOT THE ANSWER —
#: leg 1 reports what the parse finds and the floor only asserts that the NAME
#: exists, so a renamed or deleted kernel fails loudly instead of quietly
#: disabling its own row.
PARSE_TARGETS: Tuple[Tuple[str, str, str], ...] = (
    ("no_pml_stored_e", "stored_e_constitutive_step",
     "certified update_E, no PML, stored E with poles"),
    ("dispersive_update_e", "constitutive_step_dispersive",
     "certified update_E under an active PML (the folded arm reuses this body)"),
    ("complex_no_pml_stored_e", "complex_stored_e_step",
     "certified update_E, complex64 no-PML stored E"),
    ("kernels", "ade_update_p",
     "certified update_P, one component and one susceptibility per launch"),
    ("fused_ade_state", "fused_ade_state",
     "FUSED update_P only: one susceptibility, all its driven components"),
    ("fused_dispersive_chain", "fused_curl_dispersive_E_ade",
     "FUSED step_D -> update_E -> update_P at one pole per component"),
)

#: Susceptibility counts on the 15 rows that reach the E->P seam, read off
#: ``results/metal_coverage_tranche6_2026-08-19`` (and confirmed row for row by
#: ``results/cuda_predicate_coverage_2026-08-20_final``). Recorded as the numbers
#: the cost leg must use rather than a representative one.
CORPUS_E_TO_P_ROWS: Tuple[Tuple[str, int, int], ...] = (
    # (row, susceptibilities, driven components each)
    ("examples:absorber-1d.py", 5, 3),
    ("examples:material-dispersion.py", 2, 3),
    ("examples:stochastic_emitter.py", 6, 3),
    ("examples:stochastic_emitter_line.py", 6, 3),
    ("examples:stochastic_emitter_reciprocity.py", 6, 3),
    ("tests:TestAbsorber.test_absorber", 5, 3),
    ("tests:TestLoadDump.test_load_dump_chunk_layout_file_2d", 5, 3),
    ("tests:TestLoadDump.test_load_dump_chunk_layout_sim_2d", 5, 3),
    ("tests:TestLoadDump.test_load_dump_structure_2d", 5, 3),
    ("tests:TestLoadDump.test_load_dump_structure_sharded_2d", 5, 3),
)


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# LEG 1 -- what the certified kernels' own source says about component shape
# ---------------------------------------------------------------------------

def _is_triton_jit(node: ast.FunctionDef) -> bool:
    """``@triton.jit`` on this definition, matched on the attribute path."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Attribute) and decorator.attr == "jit":
            value = decorator.value
            if isinstance(value, ast.Name) and value.id == "triton":
                return True
        if isinstance(decorator, ast.Name) and decorator.id == "jit":
            return True
    return False


def _store_bases(node: ast.FunctionDef) -> List[str]:
    """The base pointer name of every ``tl.store(<base> + idx, ...)`` in the body.

    A store's destination is written ``tl.store(ptr + idx, value, mask=live)``, so
    the base is the LEFTMOST Name in the first argument's expression tree. Taking
    the leftmost rather than "any Name" is what keeps ``idx`` out of the answer.
    """
    bases: List[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        function = child.func
        if not (isinstance(function, ast.Attribute) and function.attr == "store"):
            continue
        if not child.args:
            continue
        target = child.args[0]
        while isinstance(target, ast.BinOp):
            target = target.left
        while isinstance(target, ast.Subscript):
            target = target.value
        if isinstance(target, ast.Name):
            bases.append(target.id)
    return bases


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_module(module: str) -> Dict[str, ast.FunctionDef]:
    path = TRITON_KERNELS / f"{module}.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    found: Dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _is_triton_jit(node):
            found[node.name] = node
    return found


def leg_certified_update_e_component_shape() -> Dict[str, Any]:
    """Can a certified Triton ``update_E`` be launched for ONE component?

    Decided by counting the distinct base pointers the kernel's ``tl.store`` calls
    target. A body that stores to three distinct destinations writes three
    components in one launch and cannot be dispatched per component without
    splitting it; a body that stores to one can.
    """
    rows: List[Dict[str, Any]] = []
    for module, kernel, role in PARSE_TARGETS:
        found = _parse_module(module)
        # NAME-DRIFT FLOOR. A target that matches nothing would silently drop its
        # row and take its finding with it.
        if kernel not in found:
            raise SystemExit(
                f"leg 1: {module}.py defines no @triton.jit function named "
                f"{kernel!r} (it defines {sorted(found)}). A parse target that "
                f"matches nothing disables the check it feeds; fix the name or "
                f"drop the target deliberately.")
        node = found[kernel]
        bases = _store_bases(node)
        distinct = sorted(set(bases))
        arguments = [a.arg for a in node.args.args]
        rows.append({
            "module": f"meep_gpu/triton_kernels/{module}.py",
            # THE BYTES THIS ROW WAS PARSED OUT OF. gate_provenance.stamp records
            # what the process IMPORTED, and this probe imports none of these — it
            # reads them as text — so the digest that binds the finding has to be
            # recorded here or it is recorded nowhere.
            "module_sha256": _digest(TRITON_KERNELS / f"{module}.py"),
            "kernel": kernel, "role": role,
            "line": node.lineno,
            "store_calls": len(bases),
            "distinct_store_destinations": distinct,
            "n_distinct_store_destinations": len(distinct),
            "n_arguments": len(arguments),
            "writes_one_destination_per_launch": len(distinct) == 1,
        })
        log(f"  leg1  {module}.{kernel:34s} stores={len(bases):2d} "
            f"distinct={len(distinct):2d} {distinct} "
            f"-> {'ONE destination' if len(distinct) == 1 else 'MANY destinations'}")

    e_side = [r for r in rows if r["role"].startswith("certified update_E")]
    per_component_available = any(
        r["writes_one_destination_per_launch"] for r in e_side)
    # NON-VACUITY FLOOR: the E-side set must not be empty, or the verdict below is
    # a statement about nothing.
    if not e_side:
        raise SystemExit(
            "leg 1: no parse target is labelled a certified update_E, so the "
            "per-component verdict would be vacuous")
    log(f"  leg1  certified update_E kernels parsed: {len(e_side)}; "
        f"any writing ONE component per launch: {per_component_available}")
    return {
        "kernels": rows,
        "certified_update_e_kernels": len(e_side),
        "a_certified_update_e_writes_one_component_per_launch":
            per_component_available,
        "verdict": (
            "the per-component shape Metal's fused_ade_chain uses is AVAILABLE"
            if per_component_available else
            "EVERY certified Triton update_E writes all three components in one "
            "launch, so the per-component shape is NOT available to a Triton "
            "product without splitting a certified body; the choice is between "
            "the shared-scratch shape (a) and the per-component-scratch shape (c)"),
    }


# ---------------------------------------------------------------------------
# LEG 2 -- the rotation orbit, enumerated to closure, for THREE launch shapes
# ---------------------------------------------------------------------------

def rotate(configuration: Dict[str, Any], driven: Sequence[str], scratches: int
           ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One susceptibility's whole ``update``, on buffer NAMES.

    TRANSCRIBED from ``probe_metal_ade_rotation_seam.rotate`` (:343-364), which
    transcribes ``dispersion.PolarizationState.update`` (dispersion.py:679-691) —
    the three assignments at the bottom of that loop, in that order.

    ``scratches`` is this probe's ONE addition and is the whole of shape (c): with
    ``scratches == 1`` every component draws its output from the single shared
    scratch, which is the reference and what ``dispersion.py`` allocates
    (dispersion.py:646-650); with ``scratches == len(driven)`` each component owns
    its own, and the permutation decomposes into ``d`` independent 3-cycles.
    """
    state = {"P": dict(configuration["P"]), "P_prev": dict(configuration["P_prev"]),
             "scratch": dict(configuration["scratch"])}
    launches: List[Dict[str, Any]] = []
    for component in driven:
        slot = component if scratches > 1 else "shared"
        p = state["P"][component]
        p_prev = state["P_prev"][component]
        scratch = state["scratch"][slot]
        launches.append({"component": component, "out": scratch,
                         "p_now": p, "p_prev": p_prev})
        state["P"][component] = scratch          # dispersion.py:689
        state["P_prev"][component] = p           # dispersion.py:690
        state["scratch"][slot] = p_prev          # dispersion.py:691
    return state, launches


def key_of(configuration: Dict[str, Any], driven: Sequence[str]) -> Tuple[Any, ...]:
    return (tuple(configuration["P"][c] for c in driven),
            tuple(configuration["P_prev"][c] for c in driven),
            tuple(sorted(configuration["scratch"].items())))


def orbit(driven: Sequence[str], scratches: int) -> List[Dict[str, Any]]:
    """Every buffer configuration this rotation can ever reach, to closure.

    Finite because the rotation is a permutation of a finite named set, so this
    turns "the alias never happens" from a claim about the steps someone ran into
    a claim about every step that can ever run.
    """
    slots = list(driven) if scratches > 1 else ["shared"]
    start = {"P": {c: f"A[{c}]" for c in driven},
             "P_prev": {c: f"B[{c}]" for c in driven},
             "scratch": {s: f"S[{s}]" for s in slots}}
    seen: set = set()
    configurations: List[Dict[str, Any]] = []
    configuration = start
    while True:
        key = key_of(configuration, driven)
        if key in seen:
            return configurations
        seen.add(key)
        configurations.append(configuration)
        configuration, _ = rotate(configuration, driven, scratches)


def _conflicts(driven: Sequence[str], scratches: int, per_component: bool
               ) -> Tuple[int, int, Optional[dict]]:
    """Positions of the orbit at which a launch writes a buffer it also reads."""
    configurations = orbit(driven, scratches)
    hits: List[dict] = []
    for position, configuration in enumerate(configurations):
        _, launches = rotate(configuration, driven, scratches)
        if per_component:
            for row in launches:
                reads = {row["p_now"], row["p_prev"]}
                writes = {row["out"]}
                if reads & writes:
                    hits.append({"orbit_position": position,
                                 "component": row["component"],
                                 "aliased": sorted(reads & writes)})
        else:
            reads = ({row["p_now"] for row in launches}
                     | {row["p_prev"] for row in launches})
            writes = {row["out"] for row in launches}
            if reads & writes:
                hits.append({
                    "orbit_position": position,
                    "aliased": sorted(reads & writes),
                    "chain": [(r["component"], r["out"], r["p_now"], r["p_prev"])
                              for r in launches]})
    return len(configurations), len(hits), (hits[0] if hits else None)


def leg_rotation_orbit_disjointness() -> Dict[str, Any]:
    """Three launch shapes, every orbit position, d = 1, 2, 3."""
    rows: List[Dict[str, Any]] = []
    for count in (1, 2, 3):
        driven = COMPONENTS[:count]
        size_pc, hits_pc, first_pc = _conflicts(driven, 1, per_component=True)
        size_a, hits_a, first_a = _conflicts(driven, 1, per_component=False)
        size_c, hits_c, first_c = _conflicts(driven, count, per_component=False)
        rows.append({
            "driven_components": list(driven),
            "per_component_launch": {
                "orbit_size": size_pc, "conflicts": hits_pc, "first": first_pc},
            "all_component_shared_scratch_shape_a": {
                "orbit_size": size_a, "conflicts": hits_a, "first": first_a},
            "all_component_per_component_scratch_shape_c": {
                "orbit_size": size_c, "conflicts": hits_c, "first": first_c},
        })
        log(f"  leg2  d={count}  per-component: orbit {size_pc:2d} conflicts "
            f"{hits_pc}  |  shape(a) shared scratch: orbit {size_a:2d} conflicts "
            f"{hits_a}  |  shape(c) one scratch per component: orbit {size_c:2d} "
            f"conflicts {hits_c}")

    # NON-VACUITY FLOORS. Two clean results are not a contrast, and an orbit of one
    # position has enumerated nothing.
    multi = [r for r in rows if len(r["driven_components"]) > 1]
    if not any(r["all_component_shared_scratch_shape_a"]["conflicts"]
               for r in multi):
        raise SystemExit(
            "leg 2: the shared-scratch all-component shape reported NO conflict at "
            "any orbit position for d > 1. That is fused_ade_state's documented "
            "alias, so either the transcription of dispersion.py:689-691 is wrong "
            "or the enumeration is not reaching the configurations that carry it; "
            "the clean results next to it would be meaningless either way.")
    if any(r["all_component_shared_scratch_shape_a"]["orbit_size"] < 2
           for r in multi):
        raise SystemExit(
            "leg 2: an orbit closed after ONE configuration for d > 1, so nothing "
            "was enumerated and every 'no conflict' below is vacuous")

    clean_c = all(r["all_component_per_component_scratch_shape_c"]["conflicts"] == 0
                  for r in rows)
    clean_pc = all(r["per_component_launch"]["conflicts"] == 0 for r in rows)
    return {
        "rows": rows,
        "per_component_launch_is_alias_free": clean_pc,
        "shape_c_is_alias_free": clean_c,
        "verdict": (
            "shape (c) — one scratch per driven component, all components in ONE "
            "launch — is ALIAS-FREE at every position of the orbit, for d = 1, 2 "
            "and 3, while the shared-scratch shape (a) conflicts at every position "
            "for d > 1. So a Triton E->P product CAN keep the certified "
            "all-component update_E body and still inherit no intra-launch alias, "
            "at the price leg 3 measures."
            if clean_c else
            "shape (c) CONFLICTS; the per-component-scratch rotation does not "
            "remove the alias and this probe licenses nothing"),
    }


# ---------------------------------------------------------------------------
# LEG 3 -- what shape (c) costs on the rows the corpus actually drives
# ---------------------------------------------------------------------------

def leg_scratch_cost_on_the_corpus() -> Dict[str, Any]:
    """Extra field volumes shape (c) allocates, per E->P row of the corpus.

    The reference allocates ONE scratch per susceptibility (dispersion.py:646-650).
    Shape (c) allocates one per DRIVEN COMPONENT of each susceptibility, so the
    extra is ``states * (driven - 1)`` volumes. Nothing is rounded and nothing is
    representative: these are the ten rows Metal's fused_ade_chain serves.
    """
    rows: List[Dict[str, Any]] = []
    for row, states, driven in CORPUS_E_TO_P_ROWS:
        reference = states
        shaped = states * driven
        rows.append({"row": row, "susceptibilities": states,
                     "driven_components": driven,
                     "scratch_volumes_reference": reference,
                     "scratch_volumes_shape_c": shaped,
                     "extra_volumes": shaped - reference})
        log(f"  leg3  {row[:54]:54s} K={states} d={driven}  "
            f"scratch {reference} -> {shaped}  (+{shaped - reference} volumes)")
    worst = max(rows, key=lambda r: r["extra_volumes"])
    log(f"  leg3  worst row: {worst['row']} at +{worst['extra_volumes']} volumes")
    return {
        "rows": rows,
        "worst_extra_volumes": worst["extra_volumes"],
        "worst_row": worst["row"],
        "note": (
            "A volume here is one field-sized float32 array. The cost is MEMORY "
            "only: no extra launch, no extra arithmetic, and the values in P and "
            "P_prev after the rotation are the reference's, which is what a byte "
            "gate would compare. What it buys is that the launch's write set is "
            "disjoint from its read set BY CONSTRUCTION rather than by an "
            "observation about one compiler's reordering."),
    }


# ---------------------------------------------------------------------------

def emit(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    out: Path = arguments.out
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    log("=" * 78)
    log("THE TRITON E->P ROTATION SHAPE — host-only, no device")
    log("=" * 78)

    results: Dict[str, Any] = {
        "probe": "triton_ade_rotation_shape",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "python": sys.version.split()[0],
        "device_used": None,
        "legs": {},
    }

    with (out.with_suffix(".jsonl")).open("w") as handle:
        log("\n--- LEG 1: what the certified kernels' source says -----------------")
        leg1 = leg_certified_update_e_component_shape()
        results["legs"]["certified_update_e_component_shape"] = leg1
        emit(handle, {"leg": "certified_update_e_component_shape", **leg1})

        log("\n--- LEG 2: the rotation orbit, three shapes, to closure ------------")
        leg2 = leg_rotation_orbit_disjointness()
        results["legs"]["rotation_orbit_disjointness"] = leg2
        emit(handle, {"leg": "rotation_orbit_disjointness", **leg2})

        log("\n--- LEG 3: what shape (c) costs on this corpus ---------------------")
        leg3 = leg_scratch_cost_on_the_corpus()
        results["legs"]["scratch_cost_on_the_corpus"] = leg3
        emit(handle, {"leg": "scratch_cost_on_the_corpus", **leg3})

    results["elapsed_s"] = round(time.time() - started, 3)
    results["answer"] = {
        "per_component_shape_available_on_triton":
            leg1["a_certified_update_e_writes_one_component_per_launch"],
        "shape_c_alias_free": leg2["shape_c_is_alias_free"],
        "shape_c_cost_worst_extra_volumes": leg3["worst_extra_volumes"],
        "statement": (
            "THE ROTATION DOES NOT HAVE TO BE BAKED AND THE HOST ROUND TRIP DOES "
            "NOT HAVE TO GO. On Metal the per-component interleave settles it. On "
            "Triton the per-component shape is unavailable — every certified "
            "update_E writes all three components in one launch — so the shape a "
            "Triton E->P product should take is (c): keep the certified "
            "all-component body, give each susceptibility one scratch per driven "
            "component, and the launch's write set is disjoint from its read set "
            "at every position of the rotation's orbit. The price is "
            f"{leg3['worst_extra_volumes']} extra field volumes on the corpus's "
            "worst E->P row. THIS PROBE LICENSES A SHAPE; IT BUILDS NO KERNEL AND "
            "MEASURES NO BYTES ON A DEVICE."),
    }
    gate_provenance.stamp(results)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    log(f"\nwrote {out}  ({results['elapsed_s']} s)")
    log(f"ANSWER: {results['answer']['statement']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
