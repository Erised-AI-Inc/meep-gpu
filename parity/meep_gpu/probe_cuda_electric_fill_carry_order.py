"""Does the D-side fill carry's CLOSED FORM reproduce the driver's three in-seam passes?

WHY THIS PROBE EXISTS, AND WHY IT IS NOT THE DEVICE GATE. ``fused_electric_pair`` gained
the ownership inversion on 2026-08-31, and the arithmetic it now carries is a COMPOSITION
of three driver passes that run in a fixed order:

    fill_symmetry_bc_D (driver.py:3309) -> zero_metal_D (:3310) -> fill_folded_far_ghosts_D (:3311)

The kernel performs all three from ONE thread per destination, so the order stops being a
sequence of launches and becomes a sequence of three lines. Whether those three lines are
the same arithmetic is a question about the closed form, and it is answerable on a laptop
against the ENGINE'S OWN passes -- which is what this probe answers, before a device is
booked. ``gate_cuda_fused_electric_pair.py`` is what answers it about the compiled kernel.

WHAT IT MEASURES, AND THE CONTROL THAT MAKES IT NON-VACUOUS. Two orderings are run
against the array path on every configuration:

  * SHIPPED   -- the near product is formed from the PRE-clear displacement and the wall
                 clear is applied to the RESULT, which is the driver's order;
  * NAIVE     -- the near product multiplies the POST-clear register, which is what a
                 carry ported from ``fused_magnetic_pair`` without the pre-clear copy
                 would do, and which is bit-identical on the B seam.

A configuration where the naive ordering does NOT diverge measures nothing about the
ordering, and this probe SAYS SO per row rather than reporting a pass. The rows where it
does diverge are the ones the gate's spec table has to carry.

THE GEOMETRY THAT MAKES THE TWO FAMILIES DIFFERENT, in one paragraph. A B component's
Yee shift is 0 on its OWN axis and 1 on the other two, so its ONE near axis is its own
and is also the only axis ``zero_metal_B`` clears it on -- and a folded axis is never
walled (``stepping._zero_metal``:2237-2239), so a B near ghost can never land in a cleared
plane. A D component's shifts are the exact complement: its TWO near axes are the other
two, which are exactly the two axes ``zero_metal_D`` clears it on, so it can be folded on
one and walled on the other at once. Its near ghost then lands in a plane the clear owns,
where the array path leaves ``+0.0f`` and the naive ordering leaves ``-0.0f`` whenever the
plane's declared phase is ODD.

Progress is one flushed line per configuration (the progress-reporting rule) and the artifact is
written incrementally, so an interrupted run keeps every row that landed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_API = _HERE.parents[1]
if str(_REPO_API) not in sys.path:
    sys.path.insert(0, str(_REPO_API))

from meep_gpu import stepping  # noqa: E402
from meep_gpu.cuda_kernels import fused_electric_pair as family  # noqa: E402
from meep_gpu.cuda_kernels.in_seam_coverage import (  # noqa: E402
    folded_far_rows, mirror_fill_phases, zero_metal_axes)
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

#: Every stored volume the fixture seeds. The auxiliaries start NONZERO for the reason
#: every leg on this track gives: a zero ``fu`` makes ``fu * kms`` exactly zero on the
#: first pass whatever ``kms`` is, which would hide a mis-indexed coefficient.
_SEEDED: Tuple[str, ...] = (
    "Dx", "Dy", "Dz", "Hx", "Hy", "Hz", "Ex", "Ey", "Ez",
    "fu_Dx", "fu_Dy", "fu_Dz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

_TARGETS: Tuple[str, ...] = ("Dx", "Dy", "Dz")
_AXIS_OF = {"x": 0, "y": 1, "z": 2}
#: ``in_seam_passes._zero_metal_D_kernel_code``'s OFF-DIAGONAL, as the axes each
#: component is cleared on. Read from the shipped module's own table rather than typed.
_CLEAR_AXES: Dict[str, Tuple[int, ...]] = {
    axis: tuple(_AXIS_OF[coordinate_axis]
                for _flag, coordinate, axes in family._ZERO_METAL_ROWS
                for coordinate_axis in ("x", "y", "z")
                if axis in axes and coordinate == {"x": "i", "y": "j", "z": "k"}[
                    coordinate_axis])
    for axis in ("x", "y", "z")
}

SEED = 20260831

#: Every configuration, and what each is here to exercise. THE LAST TWO ARE THE POINT:
#: an ODD plane on one near axis with a wall on the other is the only shape in which the
#: driver's order between :3309 and :3310 is observable at all.
CASES: Tuple[Dict[str, Any], ...] = (
    {"label": "all_periodic", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (), "pml": 2,
     "why": "neither fill runs; the carry must be inert"},
    {"label": "wall_xyz", "cell": (9., 10., 11.),
     "boundaries": ("metallic",) * 3, "symmetry": (), "pml": 2,
     "why": "every wall row live, no fold"},
    {"label": "fold_y_periodic", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (("Y", 1),), "pml": 2,
     "why": "both fills live on one axis"},
    {"label": "fold_y_periodic_odd", "cell": (9., 11., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (("Y", 1),), "pml": 2,
     "why": "the ODD full count, where the far reflect row is stored - 3"},
    {"label": "fold_y_odd_phase", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (("Y", -1),), "pml": 2,
     "why": "an ODD plane: a baked +1 near weight is exact at phase +1 and wrong here"},
    {"label": "fold_y_metallic", "cell": (9., 10., 11.),
     "boundaries": ("periodic", "metallic", "periodic"), "symmetry": (("Y", 1),),
     "pml": 2, "why": "the other termination: near fill only, no far fill"},
    {"label": "fold_y_wall_z", "cell": (9., 10., 11.),
     "boundaries": ("periodic", "periodic", "metallic"), "symmetry": (("Y", 1),),
     "pml": 2, "why": "a fold and a wall at once, EVEN phase"},
    {"label": "fold_y_odd_wall_z", "cell": (9., 10., 11.),
     "boundaries": ("periodic", "periodic", "metallic"), "symmetry": (("Y", -1),),
     "pml": 2,
     "why": "THE CROSS TERM: Dx folded on y at an ODD phase and walled on z, so its "
            "near ghost lands in the cleared plane"},
    {"label": "fold_y_odd_wall_x", "cell": (9., 10., 11.),
     "boundaries": ("metallic", "periodic", "periodic"), "symmetry": (("Y", -1),),
     "pml": 2,
     "why": "THE CROSS TERM on the other component: Dz folded on y at an ODD phase "
            "and walled on x"},
    {"label": "fold_xy_mixed_phase", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (("X", 1), ("Y", -1)), "pml": 2,
     "why": "two folded axes at mixed phases: the composition, and the row where a "
            "source thread is itself a fill destination"},
    {"label": "fold_xyz", "cell": (9., 10., 11.), "boundaries": ("periodic",) * 3,
     "symmetry": (("X", 1), ("Y", -1), ("Z", 1)), "pml": 2,
     "why": "three folded axes: seven ghosts per component, the deepest the closed "
            "form goes"},
    {"label": "fold_x_deep_pml", "cell": (9., 10., 11.),
     "boundaries": ("periodic",) * 3, "symmetry": (("X", 1),),
     "pml": ((0, 5), (2, 2), (2, 2)),
     "why": "an absorber reaching the far ghost's coefficient row, so the "
            "destination's kps entry differs from the source's"},
)


def build(case: Dict[str, Any]):
    """One seeded NumPy engine: ``(fields, grid, pml)``."""
    rng = np.random.default_rng(SEED)
    grid = Grid(resolution=1.0, cell_size=tuple(case["cell"]),
                boundaries=tuple(case["boundaries"]),
                symmetry=tuple(Mirror(axis, int(phase))
                               for axis, phase in case["symmetry"]),
                xp=np, courant=0.5)
    fields = Fields(grid=grid)
    fields.enable_pml_storage()
    # A FOLDED AXIS TAKES ITS ABSORBER ON THE HIGH FACE ONLY: the mirror plane is the
    # low face and ``stepping._require_consistent_pml`` admits nothing there.
    folded = {"XYZ".index(axis) for axis, _phase in case["symmetry"]}
    if isinstance(case["pml"], int):
        thickness = tuple((0, case["pml"]) if axis in folded
                          else (case["pml"], case["pml"]) for axis in range(3))
    else:
        thickness = tuple(tuple(int(v) for v in pair) for pair in case["pml"])
    pml = PML(grid=grid, thickness=thickness)
    shape = tuple(int(n) for n in grid.shape)
    for name in _SEEDED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        array[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    return fields, grid, pml


def words(array: Any) -> np.ndarray:
    """Raw uint32 WORDS. Byte compares, never ``allclose``: ``-0.0 == 0.0`` lies, and
    the sign of zero is the entire finding this probe exists to record."""
    return np.ascontiguousarray(array, dtype=np.float32).ravel().view(np.uint32)


def closed_form(pre: Dict[str, np.ndarray], shape: Tuple[int, int, int],
                near: Sequence[int], reflect: Sequence[int],
                phase: Sequence[np.float32], walls: Sequence[bool],
                naive: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """The kernel's own composition, evaluated as arrays.

    ONE ARRAY OP PER EMITTED LINE, in the order :func:`family.fill_carry_blocks` emits
    them, and the ownership guard is the mask the emitted blocks nest inside. The write
    COUNT per cell is returned beside the volume: the carry's whole legality argument is
    that every destination word is written by exactly one thread, and a probe that only
    compared values would not notice two threads racing on one word.
    """
    out: Dict[str, np.ndarray] = {}
    writes: Dict[str, int] = {}
    index = np.indices(shape)
    for target, name in enumerate(_TARGETS):
        axis_letter = name[-1]
        near_axes = family.near_fill_axes(target)
        far_axes = family.far_fill_axes(target)
        owned = np.ones(shape, dtype=bool)
        for axis in near_axes:
            if near[axis]:
                owned &= index[axis] != 0
        for axis in far_axes:
            if reflect[axis] >= 0:
                owned &= index[axis] != shape[axis] - 1
        cleared = np.zeros(shape, dtype=bool)
        for axis in _CLEAR_AXES[axis_letter]:
            if walls[axis]:
                cleared |= index[axis] == 0
        volume = np.array(pre[name], copy=True)
        volume[owned & cleared] = np.float32(0.0)
        written = np.zeros(shape, dtype=np.int32)
        written[owned] += 1
        for far_subset, near_subset in family.carried_destinations(near_axes, far_axes):
            source: List[Any] = [slice(None)] * 3
            destination: List[Any] = [slice(None)] * 3
            live = True
            for axis in far_subset:
                if reflect[axis] < 0:
                    live = False
                    break
                source[axis] = int(reflect[axis])
                destination[axis] = shape[axis] - 1
            for axis in near_subset:
                if not live or not near[axis]:
                    live = False
                    break
                source[axis] = family.NEAR_SOURCE_INDEX
                destination[axis] = 0
            if not live:
                continue
            own_source = owned[tuple(source)]
            cleared_source = cleared[tuple(source)]
            value = np.array(pre[name][tuple(source)], copy=True)
            if near_subset:
                product = np.float32(1.0)
                for axis in near_subset:
                    product = np.float32(product * phase[axis])
                if naive:
                    value = np.float32(product) * np.where(
                        cleared_source, np.float32(0.0), value)
                else:
                    value = np.float32(product) * value
                    value = np.where(cleared_source, np.float32(0.0), value)
            else:
                value = np.where(cleared_source, np.float32(0.0), value)
            for axis in far_subset:
                value = np.float32(-phase[axis]) * value
            out_slice = volume[tuple(destination)]
            volume[tuple(destination)] = np.where(
                own_source, value.astype(np.float32), out_slice).astype(np.float32)
            written[tuple(destination)] = (written[tuple(destination)]
                                           + own_source.astype(np.int32))
        out[name] = volume
        writes[name] = int(np.count_nonzero(written != 1))
    return out, writes


def measure(case: Dict[str, Any]) -> Dict[str, Any]:
    """One configuration: the array path, then both orderings against it."""
    started = time.time()
    fields, grid, pml = build(case)
    stepping.step_D(fields, pml)
    pre = {name: np.array(getattr(fields, name), copy=True) for name in _TARGETS}
    stepping.fill_symmetry_bc_D(fields)
    stepping.zero_metal_D(fields)
    stepping.fill_folded_far_ghosts_D(fields)
    final = {name: np.array(getattr(fields, name), copy=True) for name in _TARGETS}

    phases = mirror_fill_phases(grid)
    rows = folded_far_rows(grid)
    walls = zero_metal_axes(grid)
    plan = family.fused_electric_pair_fills(grid)
    near = plan["near"]
    reflect = plan["reflect"]
    phase = tuple(np.float32(value) for value in plan["phase"])
    shape = tuple(int(n) for n in grid.shape)

    result: Dict[str, Any] = {
        "label": case["label"], "why": case["why"], "shape": list(shape),
        "near": list(near), "reflect": list(reflect),
        "phase": [float(v) for v in plan["phase"]],
        "walls": [bool(v) for v in walls],
        "fills_live": bool(any(near) or any(row >= 0 for row in reflect)),
    }
    for ordering, naive in (("shipped", False), ("naive", True)):
        model, writes = closed_form(pre, shape, near, reflect, phase, walls, naive)
        differing = {name: int(np.count_nonzero(words(model[name])
                                                != words(final[name])))
                     for name in _TARGETS}
        result[ordering] = {
            "differing_words": sum(differing.values()),
            "per_component": differing,
            "cells_not_written_exactly_once": sum(writes.values()),
        }
    result["control_is_vacuous"] = result["naive"]["differing_words"] == 0
    result["elapsed_s"] = round(time.time() - started, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="DIRECTORY to write cuda_electric_fill_carry_order.json")
    arguments = parser.parse_args()
    out_dir = arguments.out
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "cuda_electric_fill_carry_order.json"

    print("the D-side fill carry's closed form against the driver's own three passes",
          flush=True)
    print(f"  probe   : {Path(__file__).name}", flush=True)
    print(f"  artifact: {artifact}", flush=True)
    print(f"  cases   : {len(CASES)}", flush=True)

    rows: List[Dict[str, Any]] = []
    failed = 0
    discriminating = 0
    for number, case in enumerate(CASES, start=1):
        row = measure(case)
        rows.append(row)
        shipped = row["shipped"]["differing_words"]
        naive = row["naive"]["differing_words"]
        races = row["shipped"]["cells_not_written_exactly_once"]
        if shipped or races:
            failed += 1
        if naive:
            discriminating += 1
        verdict = "OK" if not shipped and not races else "DIFFERS"
        control = ("DISCRIMINATING" if naive else
                   ("vacuous control" if row["fills_live"] else "no fill runs here"))
        print(f"  case {number}/{len(CASES)} {case['label']}: shipped={shipped} words "
              f"naive={naive} words races={races} -> {verdict}, {control} "
              f"({row['elapsed_s']} s)", flush=True)
        # WRITTEN PER CASE, not at the end: an interrupted run keeps every row that
        # landed, which is what makes a long sweep resumable (the progress-reporting rule).
        artifact.write_text(json.dumps({
            "probe": "cuda_electric_fill_carry_order",
            "subject": "meep_gpu/cuda_kernels/fused_electric_pair.py",
            "passes": ["fill_symmetry_bc_D (driver.py:3309)",
                       "zero_metal_D (driver.py:3310)",
                       "fill_folded_far_ghosts_D (driver.py:3311)"],
            "orderings": {
                "shipped": "the near product is formed from the PRE-clear "
                           "displacement and the wall clear applies to the result",
                "naive": "the near product multiplies the POST-clear register -- what "
                         "a port of fused_magnetic_pair's carry would do, and "
                         "bit-identical on the B seam",
            },
            "complete": len(rows) == len(CASES),
            "cases": rows,
            "cases_measured": len(rows),
            "cases_differing": failed,
            "cases_where_the_control_diverges": discriminating,
            "passed": failed == 0 and discriminating > 0,
        }, indent=1) + "\n", encoding="utf-8")

    print("", flush=True)
    print(f"  {len(rows)} configurations, {failed} differing from the array path, "
          f"{discriminating} where the naive ordering diverges", flush=True)
    if not discriminating:
        print("  THE WHOLE SWEEP IS VACUOUS: no configuration separates the two "
              "orderings, so this probe measured nothing about the order", flush=True)
    print(f"  wrote {artifact}", flush=True)
    return 0 if (failed == 0 and discriminating > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
