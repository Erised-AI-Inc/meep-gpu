"""MEASURE the composed ghost value the D seam leaves, before any kernel carries it.

WHY THIS RUNS FIRST. The Triton ``folded_fused_pair`` refuses
``fill_folded_far_ghosts_D`` by name. Carrying it needs a rule for what ONE lane must
write at every cell the driver's three post-injection D passes touch::

    fill_symmetry_bc_D (driver.py:3300)  ->  D[..0..]  = +phase * D[..2..]   on a != m
    zero_metal_D       (driver.py:3301)  ->  D[wall]   = 0
    fill_folded_far_ghosts_D (:3302)     ->  D[..-1..] = -phase * D[..r_m..]  on a == m

THE D GEOMETRY INVERTS THE B GEOMETRY and the inversion is not only in which axes
each fill touches. On the B half ``_zero_metal`` and the NEAR fill act on the same
cell set (both ``iyee == 0``, i.e. axis ``m``), and a folded axis is never walled, so
the wall clear can never land on a near destination and the ORDER of parity and clear
is unobservable. On the D half the near fill touches the two axes that are NOT the
component's own, while the wall clear touches those same two — so a component folded
on ``y`` can still be walled on ``z``, the clear DOES land on a near destination, and
because the driver clears AFTER the near fill and BEFORE the far one the two ghosts
take OPPOSITE orders:

* a NEAR ghost is ``clear(parity * v)``   — parity first, clear second;
* a FAR  ghost is ``parity * clear(v)``   — clear first, parity second, and NOTHING
  clears it afterwards, because :3302 is the last pass before ``update_E``.

Those differ in exactly the sign bit of a zero, which is a byte difference on the
uint32 view every gate on this track compares. So a far carry that reuses the near
carry's composed value — or a composite that multiplies the raw parities together
instead of chaining through the already-clamped near ghost — is wrong in a way no
tolerance would show. THIS SCRIPT SETTLES WHICH BY MEASUREMENT rather than by the
paragraph above.

WHAT IT COMPARES. For every case it runs the array path's OWN three passes over a
random ``D`` and compares, as uint32 words over the whole volume, against a
CANDIDATE rule evaluated per destination cell. Two candidates are scored, and the
second exists to prove the first is not vacuous:

* ``chained``    — the rule this module proposes: the far ghost is the far parity
  times the ALREADY-CLAMPED near ghost at the far source row;
* ``flat_parity`` — the same cells written with the product of the raw parities times
  the clamped source value, which is what mirroring the B side's construction would
  give. A case where the two AGREE cannot discriminate and is reported as such.

Progress is one flushed line per case and every case's verdict is appended to the
JSON as it lands, so an interrupted run keeps everything up to the failure.
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_API = os.path.dirname(os.path.dirname(_HERE))
if _API not in sys.path:
    sys.path.insert(0, _API)

import numpy  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import IYEE_SHIFTS, Fields, mirror_parity  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402

#: The D family, in the order ``stepping.D_CURL_TERMS`` names them.
D_TARGETS = ("Dx", "Dy", "Dz")

#: ``stepping._write_mirror_ghost`` reads this stored row (:1451, via
#: ``_mirror_source``'s ``MIRROR_SOURCE_INDEX``).
NEAR_SOURCE_ROW = 2


def _seed_for(case_name: str) -> int:
    """A per-case seed from a sha256 DIGEST, never ``hash()``.

    ``hash()`` on a str is salted per interpreter process (PYTHONHASHSEED), so a
    case named the same way would draw different values on a re-run and a failure
    could not be reproduced from the record.
    """
    digest = hashlib.sha256(case_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _words(array: Any) -> Any:
    """The uint32 view of a float32 array — signed zeros and NaNs included."""
    return numpy.asarray(array, dtype=numpy.float32).view(numpy.uint32)


def _face(axis: int, index: int) -> Tuple[Any, ...]:
    """``stepping._face`` (:2484), restated so this script does not import a private."""
    return tuple(index if a == axis else slice(None) for a in range(3))


def _fill_values(shape: Tuple[int, int, int], rng: Any, value_class: str) -> Any:
    """One volume of float32 draws in the named value class.

    ``uniform`` is the ordinary case. ``subnormal_band`` draws inside the float32
    subnormal range, where a policy that flushes to zero and one that keeps would
    disagree — this script never launches a kernel, but the composition law it
    measures has to hold on those words too, because the carry multiplies them by
    +-1.0. ``signed_zero_lattice`` plants +0.0 and -0.0 deliberately, which is the
    only class in which ``chained`` and ``flat_parity`` can differ.
    """
    n = int(numpy.prod(shape))
    if value_class == "uniform":
        out = rng.uniform(-1.0, 1.0, size=n).astype(numpy.float32)
    elif value_class == "subnormal_band":
        out = (rng.uniform(1.0, 8.0, size=n) * 1e-42).astype(numpy.float32)
        out *= rng.choice([-1.0, 1.0], size=n).astype(numpy.float32)
    elif value_class == "signed_zero_lattice":
        # Every third word an exact zero, alternating sign, the rest ordinary.
        out = rng.uniform(-1.0, 1.0, size=n).astype(numpy.float32)
        out[0::3] = numpy.float32(0.0)
        out[1::6] = numpy.float32(-0.0)
    else:  # pragma: no cover - the caller's table is closed
        raise ValueError(f"unknown value class {value_class!r}")
    return out.reshape(shape)


class _Case:
    """One grid declaration, its fold codes and its wall set."""

    def __init__(self, name: str, cell: Tuple[float, float, float],
                 symmetry: Tuple[Any, ...], boundary: Optional[Dict[str, str]],
                 value_class: str) -> None:
        self.name = name
        self.cell = cell
        self.symmetry = symmetry
        self.boundary = boundary
        self.value_class = value_class


def _build(case: _Case) -> Tuple[Any, Any]:
    """The grid and a Fields carrying only what the three passes read."""
    kwargs: Dict[str, Any] = dict(resolution=10.0, cell_size=case.cell,
                                  symmetry=case.symmetry)
    if case.boundary is not None:
        kwargs["boundaries"] = case.boundary
    grid = Grid(**kwargs)
    fields = Fields(grid=grid)
    return grid, fields


def _array_path_oracle(fields: Any, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Run the driver's three post-injection D passes, in driver order, on ``raw``.

    driver.py:3300-3302 — ``fill_symmetry_bc_D``, then ``zero_metal_D``, then
    ``fill_folded_far_ghosts_D``. Called through the SHIPPED functions rather than
    re-implemented; a second implementation of the oracle is a mirrored evaluator.
    """
    for name in D_TARGETS:
        getattr(fields, name)[...] = raw[name]
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    return {name: numpy.array(getattr(fields, name)) for name in D_TARGETS}


def _wall_axes(grid: Any) -> Tuple[bool, bool, bool]:
    """``stepping._zero_metal``'s ``walled`` list (:2235-2237), as a per-axis flag."""
    if not getattr(grid, "has_metallic", False):
        return (False, False, False)
    return tuple(bool(grid.is_metallic(a) and not grid.is_mirrored(a))
                 for a in range(3))


#: Every rule scored. ``chained`` is the one the kernel will carry; each other name
#: is ONE transcription error a writer of that kernel could plausibly commit, kept
#: so the run reports how large each hazard is rather than asserting it exists.
RULES: Tuple[str, ...] = (
    "chained",
    "frozen_near_source",
    "clear_before_parity",
    "far_reclears",
    "far_flat_product",
)

#: What each alternative gets wrong, in one line, carried into the JSON so the
#: record does not need this file to be read alongside it.
RULE_HAZARDS: Dict[str, str] = {
    "chained":
        "the proposed rule: near ghost = clear(parity * v), far ghost = far parity "
        "* the ALREADY-CLAMPED near-composed value at the reflect row, no clear after",
    "frozen_near_source":
        "the near fills read a FROZEN snapshot instead of imaging in place, so a "
        "cell unowned on two near axes loses the second parity "
        "(stepping._fill_symmetry_ghost_cells applies X, then Y, then Z, :1441-1447)",
    "clear_before_parity":
        "the near ghost is parity * clear(v) instead of clear(parity * v); the "
        "driver clears at :3301 AFTER the near fill at :3300, so on an odd plane "
        "the two differ in the sign bit of a zero",
    "far_reclears":
        "the far ghost is re-cleared after its parity; nothing clears it in the "
        "array path, because fill_folded_far_ghosts_D (:3302) is the LAST pass "
        "before update_E",
    "far_flat_product":
        "a far-over-near composite takes the product of the raw parities against "
        "the owned value instead of chaining through the clamped near ghost",
}


def _candidate(grid: Any, raw: Dict[str, Any], rule: str) -> Dict[str, Any]:
    """Evaluate one composition rule cell by cell, from ``raw`` alone.

    ONE LANE OWNS EVERY GHOST. For component ``m`` the source lane sits at stored
    row ``2`` on each near axis it writes and at ``r_m`` on the far axis, and it
    writes every cell that back-substitutes to it. This evaluates that from the
    destination's side, which is the same set read the other way round.
    """
    if rule not in RULES:  # pragma: no cover - closed table
        raise ValueError(f"unknown rule {rule!r}")
    near_axes = [a for a in range(3) if grid.is_mirrored(a)]
    far_axes = [a for a in range(3) if stepping._stored_past_owned(grid, a)]
    rows = stepping._far_reflect_rows(grid)
    phases = tuple(grid.mirror_phase(a) for a in range(3))
    walls = _wall_axes(grid)

    out: Dict[str, Any] = {}
    for name in D_TARGETS:
        shifts = IYEE_SHIFTS[name]

        def clear(volume: Any, _shifts: Any = shifts) -> Any:
            """``zero_metal_D`` on one component (:2244-2247)."""
            for axis in range(3):
                if walls[axis] and _shifts[axis] == 0:
                    volume[_face(axis, 0)] = 0
            return volume

        # --- the near fill (driver.py:3300) --------------------------------------
        near = numpy.array(raw[name])
        frozen = numpy.array(raw[name])
        if rule == "clear_before_parity":
            # The clear moved AHEAD of the parity: image an already-cleared source
            # and never clear the destination.
            near = clear(near)
            frozen = numpy.array(near)
        for axis in near_axes:
            if shifts[axis] != 0:
                continue
            source = (frozen if rule == "frozen_near_source" else near)
            near[_face(axis, 0)] = (mirror_parity(name, axis, phases[axis])
                                    * source[_face(axis, NEAR_SOURCE_ROW)])

        # --- the wall clear (driver.py:3301) -------------------------------------
        if rule != "clear_before_parity":
            near = clear(near)

        # --- the far fill (driver.py:3302) ---------------------------------------
        far = numpy.array(near)
        for axis in far_axes:
            if shifts[axis] != 1:
                continue
            parity = mirror_parity(name, axis, phases[axis])
            if rule == "far_flat_product":
                # The composite takes the raw owned value times the product of the
                # parities, rather than the clamped near ghost. Reproduced by
                # imaging the PRE-NEAR volume and re-applying each near parity.
                base = numpy.array(raw[name])
                for other in near_axes:
                    if shifts[other] != 0:
                        continue
                    base[_face(other, 0)] = (
                        mirror_parity(name, other, phases[other])
                        * base[_face(other, NEAR_SOURCE_ROW)])
                far[_face(axis, -1)] = parity * base[_face(axis, int(rows[axis]))]
            else:
                far[_face(axis, -1)] = parity * near[_face(axis, int(rows[axis]))]
            if rule == "far_reclears":
                far = clear(far)
        out[name] = far
    return out


def _diff_words(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, int]:
    """Per-component count of differing uint32 words."""
    return {name: int(numpy.count_nonzero(_words(a[name]) != _words(b[name])))
            for name in D_TARGETS}


def _ghost_cell_count(grid: Any) -> int:
    """How many words the three passes actually rewrite — the NON-VACUITY floor.

    A case whose fold set writes nothing would make every rule agree trivially. The
    count is derived from the same two predicates the passes use, not from the
    comparison.
    """
    shape = tuple(int(n) for n in grid.shape)
    total = 0
    for name in D_TARGETS:
        shifts = IYEE_SHIFTS[name]
        touched = numpy.zeros(shape, dtype=bool)
        for axis in range(3):
            if grid.is_mirrored(axis) and shifts[axis] == 0:
                touched[_face(axis, 0)] = True
            if stepping._stored_past_owned(grid, axis) and shifts[axis] == 1:
                touched[_face(axis, -1)] = True
        total += int(numpy.count_nonzero(touched))
    return total


def _cases() -> List[_Case]:
    """The case table: fold sets x outer declarations x value classes.

    Every combination of one, two and three folded axes appears, at both mirror
    phases, with and without a metallic wall on an axis that is NOT folded — the
    last is the configuration in which ``chained`` and ``flat_parity`` can differ at
    all, and a table without it would report the two rules equal and settle nothing.
    Two full counts are carried per fold: an EVEN one (reflect row ``stored - 2``)
    and an ODD one (``stored - 3``), because the reflect row is where a fixed
    ``n - 2`` would be a whole cell wrong.
    """
    cases: List[_Case] = []
    axis_names = "XYZ"
    # Cell extents chosen so the folded axis's FULL count is even (1.2 -> 12) or
    # odd (1.3 -> 13) at resolution 10.
    even = 1.2
    odd = 1.3
    for count in (1, 2, 3):
        for axes in itertools.combinations(range(3), count):
            for parity_tag, full in (("even", even), ("odd", odd)):
                for phase_tag, phase in (("+", 1), ("-", -1)):
                    cell = [0.8, 0.8, 0.8]
                    for a in axes:
                        cell[a] = full
                    symmetry = tuple(Mirror(axis_names[a], phase) for a in axes)
                    unfolded = [a for a in range(3) if a not in axes]
                    walls: List[Optional[Dict[str, str]]] = [None]
                    if unfolded:
                        walls.append({"xyz"[unfolded[0]]: "metallic"})
                    for wall in walls:
                        wall_tag = ("wall_" + "".join(sorted(wall))) if wall else "nowall"
                        for value_class in ("uniform", "signed_zero_lattice",
                                            "subnormal_band"):
                            name = (f"fold{''.join('xyz'[a] for a in axes)}"
                                    f"_{parity_tag}_ph{phase_tag}_{wall_tag}"
                                    f"_{value_class}")
                            cases.append(_Case(name, tuple(cell), symmetry, wall,
                                               value_class))
    return cases


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None,
                        help="directory for composition_law.json (default: beside "
                             "this script under results/)")
    args = parser.parse_args(argv)

    out_dir = args.out or os.path.join(
        _HERE, "results", "d_side_composition_law_" + time.strftime("%Y-%m-%d"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "composition_law.json")

    cases = _cases()
    record: Dict[str, Any] = {
        "what": "the composed value the driver's three post-injection D passes leave",
        "driver_passes": ["fill_symmetry_bc_D (driver.py:3300)",
                          "zero_metal_D (driver.py:3301)",
                          "fill_folded_far_ghosts_D (driver.py:3302)"],
        "rules_scored": list(RULES),
        "rule_hazards": RULE_HAZARDS,
        "cases": [],
    }
    print(f"[composition-law] {len(cases)} cases x {len(RULES)} rules -> {path}",
          flush=True)

    started = time.time()
    matches = {rule: 0 for rule in RULES}
    caught_by_some_case = {rule: 0 for rule in RULES if rule != "chained"}
    vacuous = 0
    for index, case in enumerate(cases, start=1):
        t0 = time.time()
        grid, fields = _build(case)
        shape = tuple(int(n) for n in grid.shape)
        rng = numpy.random.default_rng(_seed_for(case.name))
        raw = {name: _fill_values(shape, rng, case.value_class) for name in D_TARGETS}

        oracle = _array_path_oracle(fields, raw)
        floor = _ghost_cell_count(grid)
        vacuous += int(floor == 0)

        per_rule: Dict[str, Dict[str, int]] = {}
        for rule in RULES:
            per_rule[rule] = _diff_words(oracle, _candidate(grid, raw, rule))
            if sum(per_rule[rule].values()) == 0:
                matches[rule] += 1
            elif rule != "chained":
                caught_by_some_case[rule] += 1

        chained_clean = sum(per_rule["chained"].values()) == 0
        entry = {
            "case": case.name,
            "shape": list(shape),
            "seed": _seed_for(case.name),
            "value_class": case.value_class,
            "folded_axes": [a for a in range(3) if grid.is_mirrored(a)],
            "far_axes": [a for a in range(3)
                         if stepping._stored_past_owned(grid, a)],
            "reflect_rows": [None if r is None else int(r)
                             for r in stepping._far_reflect_rows(grid)],
            "wall_axes": list(_wall_axes(grid)),
            "ghost_words_rewritten": floor,
            "differing_words_per_rule": {
                rule: sum(per_rule[rule].values()) for rule in RULES},
            "chained_differing_words_per_component": per_rule["chained"],
            "verdict": ("CHAINED MATCHES" if chained_clean else "CHAINED FAILS"),
        }
        record["cases"].append(entry)
        # Partial results as they land: the file is rewritten every case, so an
        # interrupted run keeps everything up to the failure.
        with open(path, "w") as handle:
            json.dump(record, handle, indent=1)

        others = " ".join(f"{rule.split('_')[0]}={sum(per_rule[rule].values())}"
                          for rule in RULES if rule != "chained")
        print(f"[{index}/{len(cases)}] {case.name} shape={shape} "
              f"ghost_words={floor} chained={sum(per_rule['chained'].values())} "
              f"| {others} ({time.time() - t0:.2f} s)", flush=True)

    record["summary"] = {
        "cases": len(cases),
        "matches_the_array_path_per_rule": matches,
        "cases_that_CAUGHT_each_alternative": caught_by_some_case,
        "vacuous_cases_no_ghost_written": vacuous,
        "elapsed_s": round(time.time() - started, 2),
    }
    with open(path, "w") as handle:
        json.dump(record, handle, indent=1)

    print("", flush=True)
    print(f"cases                                : {len(cases)}", flush=True)
    for rule in RULES:
        print(f"  {rule:<20} matches {matches[rule]:>4}/{len(cases)}"
              + ("" if rule == "chained"
                 else f"   CAUGHT on {caught_by_some_case[rule]} cases"), flush=True)
    print(f"vacuous cases (no ghost written)     : {vacuous}", flush=True)
    print(f"-> {path}", flush=True)

    if vacuous:
        print("REFUSED: a case wrote no ghost at all; it cannot discriminate any "
              "rule and must not be counted as evidence.", flush=True)
        return 2
    unexercised = [rule for rule, count in caught_by_some_case.items() if count == 0]
    if unexercised:
        print(f"REFUSED: no case separates {unexercised} from the proposed rule, so "
              f"this run measures nothing about the hazard(s) they name.", flush=True)
        return 2
    if matches["chained"] != len(cases):
        print("REFUSED: the proposed rule does not reproduce the array path on every "
              "case.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
