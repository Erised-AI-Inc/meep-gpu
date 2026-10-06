#!/usr/bin/env python
"""The geometry digest of the simulation the GPU harness lifts, beside the MEEP bench's.

``bench_fused_products.run_case`` lifts each leg through
``gate_dispatch_end_to_end.run_leg(case, builder, label, env, counter, gpu, res)``. This
script makes that same call with the builder ``timing_cases.resolve`` returns (the route
gate's own for a built-in case, the injected one for a new case), then computes
``bench_meep_identical_case.geometry_digest`` on the ``mp.Simulation`` the lift itself
initialised, and reads the lifted driver's own grid, PML, source and monitor counts.

``--compare`` names MEEP bench row files; every ``geometry`` row in them at the same
case and resolution must carry the same digest. ``--digest-file`` makes this lift the
DIGEST OF RECORD for its (case, resolution): written when the file is absent (with a
``.source`` beside it reading ``harness-lift:...``), compared when it is present. It is
written ONLY when the lifted driver agrees with the MEEP object on every fact checked
(grid shape, cells, ``dt``, PML thickness, source and monitor counts): a lift whose
driver disagrees is not the simulation the GPU rows time, so its digest is never a
digest of record, and the script exits non-zero.

The lift is the NumPy reference (``prefer_gpu=False``) unless ``--prefer-gpu`` is given:
the digest is a fact about the MEEP object, which is built and initialised before any
device is chosen.

One process, one rank. Writes ``<out>/harness_lift_digest_<case>_res<res>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bench_meep_identical_case as bench  # noqa: E402
import timing_cases  # noqa: E402


def driver_facts(driver: Any) -> Dict[str, Any]:
    shape = [int(v) for v in driver.shape]
    cells = 1
    for count in shape:
        cells *= count
    facts: Dict[str, Any] = {"shape": shape, "cells": cells,
                             "dt": float(driver.dt),
                             "array_module": getattr(driver.xp, "__name__", None),
                             "gpu": getattr(driver, "gpu", None)}
    grid = getattr(driver, "grid", None)
    resolution = getattr(grid, "resolution", None)
    facts["resolution"] = float(resolution) if resolution is not None else None
    pml = getattr(driver, "pml", None)
    thickness = getattr(pml, "thickness", None)
    facts["pml_thickness_cells"] = float(thickness) if thickness is not None else None
    facts["pml_thickness_length"] = (float(thickness) / float(resolution)
                                     if thickness is not None and resolution else None)
    facts["pml"] = repr(pml)
    facts["sources"] = len(getattr(driver, "_sources", []) or [])
    facts["flux_monitors"] = len(getattr(driver, "_flux_monitors", []) or [])
    facts["dft_monitors"] = len(getattr(driver, "_dft_monitors", []) or [])
    return facts


def output_name(case: str, res: int) -> str:
    return f"harness_lift_digest_{case}_res{res}.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--harness-root", default=None, dest="harness_root")
    parser.add_argument("--case", default="pml_3d")
    parser.add_argument("--table", default=None, choices=("triton", "cuda", "metal"),
                        help="the kernel table whose GPU rows this case must match")
    parser.add_argument("--res", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefer-gpu", action="store_true", dest="prefer_gpu")
    parser.add_argument("--compare", nargs="*", default=[],
                        help="MEEP bench rows_r<N>.jsonl files to compare against")
    parser.add_argument("--digest-file", default=None, dest="digest_file",
                        help="the digest of record for this case and resolution: "
                             "written when absent, compared when present")
    args = parser.parse_args(argv)

    harness_root = args.harness_root or timing_cases.find_harness_root(HERE)
    builder, builder_facts = timing_cases.resolve(args.case, harness_root, args.table)
    timing_cases.put_on_path(os.path.abspath(harness_root))

    import meep as mp  # noqa: PLC0415
    import gate_dispatch_end_to_end as e2e  # noqa: PLC0415
    import gate_dispatch_fused_route as route  # noqa: PLC0415

    e2e.PREFER_GPU = bool(args.prefer_gpu)
    print(f"[{time.strftime('%H:%M:%S')}] lifting {args.case} res {args.res} "
          f"(prefer_gpu={e2e.PREFER_GPU})", flush=True)
    started = time.time()
    leg = e2e.run_leg(args.case, builder, "array", dict(route.ARRAY_ENV), None, 0,
                      args.res)
    lift_seconds = time.time() - started
    try:
        digest = bench.geometry_digest(mp, leg["sim"])
        lifted = driver_facts(leg["driver"])
    finally:
        try:
            leg["driver"].close()
        except Exception:  # noqa: BLE001
            pass
    facts = digest["facts"]
    sources = facts["sources"]
    monitors = facts["monitors"]
    pml = [layer["thickness"] for layer in facts["boundary_layers"]]
    agreement = {
        "driver_cells_equal_meep_cells": lifted["cells"] == facts["cells"],
        "driver_shape_equals_meep_grid": lifted["shape"] == [
            facts["grid"]["nx"], facts["grid"]["ny"], facts["grid"]["nz"]],
        "driver_dt_equals_meep_dt": abs(lifted["dt"] - float(facts["dt"])) < 1e-12,
        "driver_pml_thickness_equals_meep": (
            lifted["pml_thickness_length"] is not None and bool(pml)
            and abs(lifted["pml_thickness_length"] - float(pml[0])) < 1e-9),
        "driver_source_count_equals_meep": lifted["sources"] == len(sources),
        "driver_monitor_count_equals_meep": (
            lifted["flux_monitors"] + lifted["dft_monitors"] == len(monitors)),
    }

    compared: List[Dict[str, Any]] = []
    for path in args.compare:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if (row.get("row") != "geometry" or row.get("case") != args.case
                        or int(row.get("resolution", -1)) != args.res):
                    continue
                theirs = row["geometry"]
                compared.append({
                    "file": os.path.abspath(path), "row_id": row.get("row_id"),
                    "ranks": row.get("ranks"),
                    "digest": theirs["digest"],
                    "digest_equal": theirs["digest"] == digest["digest"],
                    "epsilon_exact_equal": (theirs.get("epsilon_sha256_exact")
                                            == digest["epsilon_sha256_exact"]),
                    "cells": row.get("cells"),
                })
    of_record: Dict[str, Any] = {"file": args.digest_file, "action": None}
    driver_agrees = all(agreement.values())
    if args.digest_file:
        if not driver_agrees:
            of_record.update({"action": "refused", "equal": False,
                              "why": "the lifted driver disagrees with the MEEP object on "
                                     + ", ".join(k for k, v in agreement.items() if not v)})
        elif os.path.isfile(args.digest_file):
            with open(args.digest_file, "r", encoding="utf-8") as handle:
                held = handle.read().strip()
            of_record.update({"action": "compared", "held": held,
                              "equal": held == digest["digest"]})
        else:
            import timing_records  # noqa: PLC0415 - standard library only
            timing_records.write_lift_digest(args.digest_file, digest["digest"],
                                             f"{args.case}_res{args.res}")
            of_record.update({"action": "written", "equal": True})
    record = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case": args.case, "resolution": args.res,
        "lift": {"call": "gate_dispatch_end_to_end.run_leg", "label": "array",
                 "prefer_gpu": bool(args.prefer_gpu), "seconds": round(lift_seconds, 2)},
        "builder": builder_facts,
        "geometry": digest,
        "driver": lifted,
        "agreement": agreement,
        "compared_with": compared,
        "digest_of_record": of_record,
        "meep": mp.__version__, "single_precision": bool(mp.is_single_precision()),
    }
    equal = sum(1 for c in compared if c["digest_equal"])
    identical = bool(all(agreement.values()) and equal == len(compared)
                     and of_record.get("equal", True) is not False)
    if not compared and not args.digest_file:
        identical = False
    record["verdict"] = {
        "driver_agrees": all(agreement.values()),
        "meep_bench_rows_compared": len(compared),
        "meep_bench_rows_equal": equal,
        "digest_of_record": of_record.get("action"),
        "identical": identical,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, output_name(args.case, args.res))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, default=str)
    print(f"harness lift: digest {digest['digest']} cells {facts['cells']:,} "
          f"driver shape {lifted['shape']} pml {lifted['pml_thickness_length']} "
          f"sources {lifted['sources']} flux monitors {lifted['flux_monitors']} "
          f"({lift_seconds:.1f} s)", flush=True)
    for name, value in agreement.items():
        print(f"  {name}: {value}", flush=True)
    for entry in compared:
        print(f"  MEEP bench ranks {entry['ranks']} row {entry['row_id']}: digest "
              f"equal {entry['digest_equal']}, epsilon bit-equal "
              f"{entry['epsilon_exact_equal']}", flush=True)
    if args.digest_file:
        print(f"  digest of record {of_record['action']}: {args.digest_file}"
              + ("" if of_record.get("equal", True) else
                 f" ({of_record['why']})" if of_record.get("why") else
                 f" HOLDS {of_record.get('held')}"), flush=True)
    print(f"VERDICT identical={identical} ({equal} of {len(compared)} MEEP bench "
          f"geometry rows equal; driver agrees {record['verdict']['driver_agrees']}; "
          f"digest of record {of_record.get('action')}) -> {path}", flush=True)
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
