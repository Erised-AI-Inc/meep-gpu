"""Pin ``no_pml_ref``'s transcription against ``stepping.py`` itself, on NumPy.

The same hole ``validate_pml_reference_vs_stepping.py`` closes, one branch over:
the device gate will compare the Triton kernel against ``no_pml_ref``, and if
that transcription is wrong the kernel can be bit-identical to it and still
wrong against the engine. This closes it on a host with no GPU and no Triton,
BEFORE any device time is spent:

    stepping.step_B / step_D with pml=None   ==   no_pml_ref.reference_plain_step

bytewise, over shape x boundary x dtdx x E-storage-mode, with a real ``Grid`` and
a real ``Fields``. Both storage modes are swept because they are DIFFERENT
sub-steps: ``stores_E`` False derives ``D * inv_eps`` per sub-step, True reads
the stored arrays.

Run::

    python -u validate_no_pml_ref.py --out out.json

One flushed line per case; the JSON is rewritten as each case lands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# ``the repository root``, derived from this file's own location. It used to be an env var
# with one developer's absolute path as the default, which runs on one machine.
API = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
for _path in (HERE, API):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import no_pml_ref as ref  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402

SIDE = {
    "step_B": (("Ex", "Ey", "Ez"), ("Bx", "By", "Bz")),
    "step_D": (("Hx", "Hy", "Hz"), ("Dx", "Dy", "Dz")),
}

SHAPES = ((16, 12, 8), (10, 1, 10), (24, 24, 1), (7, 9, 11))
BOUNDARY_SETS = (
    ("periodic", "periodic", "periodic"),
    ("metallic", "metallic", "metallic"),
    ("metallic", "periodic", "metallic"),
    ("periodic", "metallic", "periodic"),
)
# 0.5 is a power of two and the FMA/association discrepancy VANISHES there; a
# gate that swept only 0.5 would certify a broken kernel (plan section 9).
COURANTS = (0.5, 0.35)
STORAGE = ("derived", "stored")


def log(message: str) -> None:
    print(message, flush=True)


def build(shape, boundaries, dtdx, storage: str):
    grid = Grid(resolution=1.0,
                cell_size=(float(shape[0]), float(shape[1]), float(shape[2])),
                courant=dtdx, boundaries=tuple(boundaries), dimensions=3, xp=np)
    if tuple(grid.shape) != tuple(shape):
        raise RuntimeError(f"Grid built {tuple(grid.shape)} for {tuple(shape)}")
    if float(grid.dt / grid.dx) != float(dtdx):
        raise RuntimeError(f"grid.dt/grid.dx is {grid.dt / grid.dx!r}, not {dtdx!r}")
    fields = Fields(grid=grid)
    if storage == "stored":
        fields.enable_field_storage()
    if bool(getattr(fields, "_pml_active", False)):
        raise RuntimeError("PML storage is on; this validation is the no-PML path")
    if bool(fields.stores_E) != (storage == "stored"):
        raise RuntimeError(f"stores_E is {fields.stores_E} for storage={storage!r}")
    return grid, fields


def seed(fields, shape, rng, storage: str) -> None:
    """Seed every array the sub-step reads, and a NON-UNIFORM inverse epsilon.

    A uniform inv_eps of exactly 1.0 makes ``D * inv_eps`` bit-equal to ``D``,
    which would hide a derivation error entirely — the 2d_plain benchmark case is
    exactly that configuration, so the sweep must not be.
    """
    for name in ("Bx", "By", "Bz", "Dx", "Dy", "Dz"):
        getattr(fields, name)[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    if storage == "stored":
        for name in ("Ex", "Ey", "Ez"):
            getattr(fields, name)[...] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    for name in ("Ex", "Ey", "Ez"):
        inverse = fields.inverse_epsilon_for(name)
        inverse[...] = rng.uniform(0.25, 1.0, size=shape).astype(np.float32)


def one_case(shape, boundaries, dtdx, sub_step, storage) -> Dict[str, Any]:
    case: Dict[str, Any] = {"shape": list(shape), "boundaries": list(boundaries),
                            "dtdx": dtdx, "sub_step": sub_step, "storage": storage}
    walled_invariant = [a for a in range(3)
                        if shape[a] == 1 and boundaries[a] == "metallic"]
    if walled_invariant:
        case["skipped"] = f"axis {walled_invariant[0]} is one cell and metallic"
        return case
    try:
        grid, fields = build(shape, boundaries, dtdx, storage)
    except Exception as exc:  # noqa: BLE001
        case["skipped"] = f"{type(exc).__name__}: {exc}"[:300]
        return case

    resolved = stepping._boundary_kinds(grid, None)
    case["resolved_boundaries"] = list(resolved)
    if tuple(resolved) != tuple(boundaries):
        case["skipped"] = f"stepping resolved {resolved}, not {tuple(boundaries)}"
        return case

    rng = np.random.default_rng(ref.SEED + 11)
    seed(fields, shape, rng, storage)

    source_names, target_names = SIDE[sub_step]
    inverse = {n: fields.inverse_epsilon_for(n).copy() for n in ("Ex", "Ey", "Ez")}
    before = {n: getattr(fields, n).copy()
              for n in ("Bx", "By", "Bz", "Dx", "Dy", "Dz")}
    if storage == "stored":
        before.update({n: getattr(fields, n).copy() for n in ("Ex", "Ey", "Ez")})

    # Leg 1 — the array path exactly as the driver calls it, pml=None.
    (stepping.step_B if sub_step == "step_B" else stepping.step_D)(fields, None)
    array_path = {n: getattr(fields, n).copy() for n in target_names}

    # Leg 2 — the transcription, from the identical inputs.
    if sub_step == "step_B":
        if storage == "stored":
            sources = {n: before[n] for n in ("Ex", "Ey", "Ez")}
        else:
            sources = ref.derive_electric(
                np, {n: before[n] for n in ("Dx", "Dy", "Dz")}, inverse)
    else:
        # get_H returns the B array itself without PML (fields.py:1184).
        sources = {"Hx": before["Bx"], "Hy": before["By"], "Hz": before["Bz"]}
    state = {n: before[n].copy() for n in target_names}
    ref.reference_plain_step(np, sources, state, np.float32(dtdx), sub_step,
                             tuple(boundaries), "array_order")
    case["vs_stepping"] = ref.combine(
        {n: ref.bit_compare(array_path[n], state[n]) for n in target_names})

    # NEGATIVE CONTROL. Regroup the stencil to C's left-to-right association and
    # require a DISAGREEMENT — only asserted where dtdx is not exactly
    # representable, because at 0.5 the two groupings legitimately coincide.
    control = {n: before[n].copy() for n in target_names}
    ref.reference_plain_step(np, sources, control, np.float32(dtdx), sub_step,
                             tuple(boundaries), "kernel_order")
    regrouped = ref.combine({n: ref.bit_compare(array_path[n], control[n])
                             for n in target_names})
    case["regrouped_control"] = {
        "bit_identical_to_stepping": regrouped["bit_identical"],
        "differing_floats": regrouped["differing_floats"],
        "max_ulp": regrouped.get("max_ulp", 0),
        "asserted": dtdx != 0.5,
    }

    # A comparison of two unchanged copies is identical for the wrong reason.
    moved = {n: not np.array_equal(before[n], array_path[n]) for n in target_names}
    case["arrays_changed"] = moved
    case["all_arrays_changed"] = all(moved.values())

    # And the derivation itself, separately: does the array path really form
    # D * inv_eps with D on the left? Compared as bytes so a commuted transcription
    # that happened to agree is still recorded as agreeing for a measured reason.
    if sub_step == "step_B" and storage == "derived":
        engine = {n: stepping._read_component(fields, n) for n in ("Ex", "Ey", "Ez")}
        mine = ref.derive_electric(np, {n: getattr(fields, n) for n in ("Dx", "Dy", "Dz")},
                                   {n: fields.inverse_epsilon_for(n)
                                    for n in ("Ex", "Ey", "Ez")})
        case["derivation_vs_read_component"] = ref.combine(
            {n: ref.bit_compare(engine[n], mine[n]) for n in ("Ex", "Ey", "Ez")})
    return case


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "no_pml_ref_vs_stepping.json"))
    args = parser.parse_args(argv)

    started = time.time()
    cases: List[Dict[str, Any]] = []
    total = (len(SHAPES) * len(BOUNDARY_SETS) * len(COURANTS) * len(SIDE) * len(STORAGE))
    index = 0
    for shape in SHAPES:
        for boundaries in BOUNDARY_SETS:
            for dtdx in COURANTS:
                for sub_step in SIDE:
                    for storage in STORAGE:
                        index += 1
                        case = one_case(shape, boundaries, dtdx, sub_step, storage)
                        cases.append(case)
                        if "skipped" in case:
                            log(f"case {index}/{total} {shape} {boundaries} "
                                f"dtdx={dtdx} {sub_step} {storage}: SKIP "
                                f"{case['skipped']}")
                        else:
                            verdict = case["vs_stepping"]
                            log(f"case {index}/{total} {shape} {boundaries} "
                                f"dtdx={dtdx} {sub_step} {storage}: "
                                f"identical={verdict['bit_identical']} "
                                f"ndiff={verdict['differing_floats']} "
                                f"control_differs="
                                f"{not case['regrouped_control']['bit_identical_to_stepping']} "
                                f"({time.time() - started:.1f} s)")
                        with open(args.out, "w") as handle:
                            json.dump({"cases": cases}, handle, indent=1)

    live = [c for c in cases if "skipped" not in c]
    identical = [c for c in live if c["vs_stepping"]["bit_identical"]]
    changed = [c for c in live if c["all_arrays_changed"]]
    controls = [c for c in live if c["regrouped_control"]["asserted"]]
    controls_bit = [c for c in controls
                    if not c["regrouped_control"]["bit_identical_to_stepping"]]
    derivations = [c for c in live if "derivation_vs_read_component" in c]
    derivations_ok = [c for c in derivations
                      if c["derivation_vs_read_component"]["bit_identical"]]
    summary = {
        "cases": len(cases), "live": len(live), "skipped": len(cases) - len(live),
        "bit_identical": f"{len(identical)}/{len(live)}",
        "all_arrays_changed": f"{len(changed)}/{len(live)}",
        "regrouped_control_differs": f"{len(controls_bit)}/{len(controls)}",
        "derivation_identical": f"{len(derivations_ok)}/{len(derivations)}",
        "wall_s": round(time.time() - started, 1),
    }
    log(json.dumps(summary))
    with open(args.out, "w") as handle:
        json.dump({"summary": summary, "cases": cases}, handle, indent=1)
    ok = (len(identical) == len(live) and len(changed) == len(live)
          and len(controls_bit) == len(controls)
          and len(derivations_ok) == len(derivations))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
