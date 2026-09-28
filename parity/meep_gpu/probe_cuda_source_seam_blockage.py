#!/usr/bin/env python3
"""WHY EACH CUDA SEAM-INSTANCE IS BLOCKED BY THE SOURCE SEAM — the grouping, measured.

THE QUESTION. ``build_cuda_fusion_matrix.py`` prints one number for this: *"186
instances, 82 blocked by the source seam"*. That number is a verdict, not a diagnosis.
The driver injects between the curl and the constitutive half, so a fused pair spanning
the seam consumes a pre-injection field unless the PRODUCT ON THAT CELL carries
``meep_gpu/deposit_repair.py``'s bracket. Three completely different things produce the
same "blocked":

  A. **A shipped product occupies the cell and does not carry the repair.** One flag and
     two slots close it.
  B. **No product occupies the cell at all.** The board prices such a cell on the driver
     fact alone, so every row with a source in that seam is blocked whether or not the
     repair would apply. A product has to be BUILT.
  C. **The repair genuinely does not apply.** ``deposit_repair.repairable`` refuses BY
     NAME — an off-diagonal chi1inv row (``update_E`` is a stencil over the partner
     volumes), an instantaneous chi2/chi3 (not the linear accumulation the repair
     inverts), the cylindrical r = 0 axis of a folded grid, and a plain (no-PML)
     constitutive path whose kps/kms recurrence never ran.

Only (A) and (B) are headroom. Reporting them together as "82 blocked" says nothing
about which. THIS SCRIPT IS THAT SPLIT, and it is the deliverable of the measure phase.

HOW IT STAYS HONEST
===================

**IT IMPORTS THE BOARD RATHER THAN RESTATING IT.** The clause that decides blocked
(:func:`build_cuda_fusion_matrix.clears_source_seam`), the cell keys
(:data:`~build_cuda_fusion_matrix.PRODUCT_ON_CELL`), the seam-to-slot map and the census
default are all read from the board module. A second spelling of the source clause is
precisely how the sibling Metal board came to print a refusal a shipped product had
stopped making (``build_cuda_fusion_matrix.py:1390-1396``), and this file would be the
second spelling.

**THE (C) TEST IS WIDER THAN THE BOARD'S.** The board's ``_deposit_is_repairable``
checks two of ``repairable``'s clauses and says of itself that every count it credits is
an UPPER BOUND. This script adds the two further refusals that ARE readable from the
census configuration — a folded grid with a cylindrical r = 0 axis, and a row whose
constitutive half takes the plain no-PML path — so a cell counted as headroom here is
headroom under a strictly stronger test than the one that priced the ceiling. The
remaining clause (``f_w_<component>`` unallocated) needs a live engine object and is
NOT readable from a census row; it is reported as such rather than assumed away.

**A CELL IS NOT HEADROOM IF THE WELD ITSELF IS REFUSED.** A STENCIL-BLOCKED cell or one
whose constitutive half launches nothing can never carry a product, so a source-blocked
instance sitting on one is reported in its own group: closing its injection would move
nothing. The fitness verdict is read from the board JSON the run is pointed at, which is
the same measurement the headline was cut from.

Progress is a flushed line per group (the progress-reporting rule); the run is seconds, and the
artifact is the JSON beside the log.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parent.parent
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


def _load_board():
    """Import ``build_cuda_fusion_matrix`` as a module even though it is a script.

    It lives beside this file with no package ``__init__`` reachable as
    ``parity.meep_gpu.build_cuda_fusion_matrix`` on every host, so it is loaded by path.
    """
    path = HERE / "build_cuda_fusion_matrix.py"
    spec = importlib.util.spec_from_file_location("_cuda_board", path)
    if spec is None or spec.loader is None:  # pragma: no cover - a moved board
        raise SystemExit(f"cannot load the board from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cuda_board"] = module
    spec.loader.exec_module(module)
    return module


BOARD = _load_board()


#: The refusals of :func:`meep_gpu.deposit_repair.repairable` that a CENSUS ROW can
#: answer, each keyed to the configuration fact that decides it. The reasons are the
#: module's own, shortened to the clause; the file:line is where the clause lives.
#: ``f_w_<component>`` unallocated is deliberately ABSENT: it needs an engine object,
#: and inventing a proxy for it would be the over-covering direction.
def _repair_refusals(seam: str, configuration: dict) -> List[str]:
    """Every ``repairable`` clause THIS row trips, read from the census configuration."""
    reasons: List[str] = []
    # deposit_repair._folded_seam_reasons: the cylindrical r = 0 axis of a FOLDED grid.
    if configuration.get("has_symmetry"):
        is_axis = configuration.get("is_axis") or ()
        for axis, on_axis in enumerate(is_axis):
            if on_axis:
                reasons.append(
                    f"deposit_repair.py:307-317 — axis {axis} is the cylindrical r = 0 "
                    f"axis of a folded grid; its below-axis ghost is the r_to_minus_r "
                    f"image and its _mirror_phases slot carries (-1)^m, so the fill "
                    f"closure this repair inverts has never been measured on it")
    # deposit_repair._absorber_reasons: the plain path never ran the kps/kms recurrence.
    if not configuration.get("pml_active"):
        reasons.append(
            "deposit_repair.py:388-392 — no active PML layer: the plain path's "
            "update_H returns without touching H and update_E writes E with no f_w "
            "accumulation, so the recurrence this repair inverts is not the one that ran")
    if seam == "D_to_E":
        if configuration.get("has_offdiagonal_epsilon"):
            reasons.append(
                "deposit_repair.py:468-473 — an off-diagonal chi1inv row: update_E "
                "reads the PARTNER components' volumes at shifted indices, so the "
                "constitutive half is a stencil over what the curl wrote in place")
        if configuration.get("has_nonlinearity"):
            reasons.append(
                "deposit_repair.py:474-477 — an instantaneous chi2/chi3: the "
                "constitutive step is not the linear accumulation this repair inverts")
    return reasons


#: Fitness verdicts under which NO product can ever occupy the cell, so a source-blocked
#: instance there is not headroom. Read from the board's own vocabulary.
UNBUILDABLE_VERDICTS = ("STENCIL-BLOCKED", "NOT A FUSION CANDIDATE", "UNDETERMINED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, required=True,
                        help="a fusion_matrix_cuda.json (or its directory) whose "
                             "fitness verdicts this run reads")
    parser.add_argument("--out", type=Path, required=True,
                        help="DIRECTORY to write the grouping into")
    args = parser.parse_args()
    started = time.time()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    board_path = args.board
    if board_path.is_dir():
        board_path = board_path / "fusion_matrix_cuda.json"
    board_json = json.loads(board_path.read_text())

    print("=" * 78, flush=True)
    print("WHY EACH CUDA SEAM-INSTANCE IS BLOCKED BY THE SOURCE SEAM", flush=True)
    print("=" * 78, flush=True)
    print(f"census : {BOARD.CENSUS.name}", flush=True)
    print(f"board  : {board_path}", flush=True)

    record, engine_facts = BOARD.rows()
    configuration_of = {f"{r['leg']}:{r['row']}": (r.get("configuration") or {})
                        for r in record}
    print(f"rows   : {len(record)}", flush=True)

    # ---------------------------------------------------------------- the cells
    # The composer's selection per slot, asked exactly the way the board asks it.
    slot_pick: Dict[str, Dict[str, Optional[dict]]] = {}
    for row in record:
        label = f"{row['leg']}:{row['row']}"
        picked: Dict[str, Optional[dict]] = {}
        for slot in BOARD.SLOTS:
            plan = row.get("plan_step") or {}
            family = (plan.get("selected_family") or {}).get(slot)
            arm = (plan.get("selected") or {}).get(slot)
            kernel = (plan.get("selected_kernel") or {}).get(slot)
            picked[slot] = (None if family is None or arm is None
                            else {"family": family, "arm": arm, "kernel": kernel})
        slot_pick[label] = picked

    has_polarization = {f"{r['leg']}:{r['row']}":
                        bool((r.get("configuration") or {}).get("n_polarizations"))
                        for r in record}

    #: The board's own fitness verdict per cell key, so this run never re-derives one.
    verdict_of: Dict[Tuple[str, str, str], str] = {}
    for cell in board_json["cells_ranked_by_demand"]:
        verdict_of[(cell["seam"], cell["curl_arm"], cell["constitutive_arm"])] = \
            cell["verdict"]

    groups: Dict[str, List[dict]] = collections.defaultdict(list)
    per_instance: List[dict] = []
    for row in record:
        label = f"{row['leg']}:{row['row']}"
        configuration = configuration_of[label]
        for seam, (curl_slot, const_slot) in BOARD.SEAM_SLOTS.items():
            if seam == "E_to_P" and not has_polarization[label]:
                continue
            curl_arm, const_arm = slot_pick[label][curl_slot], slot_pick[label][const_slot]
            curl_key = f"{curl_arm['family']}/{curl_arm['arm']}" if curl_arm else None
            const_key = f"{const_arm['family']}/{const_arm['arm']}" if const_arm else None
            product = BOARD.product_on_cell(seam, curl_key, const_key)
            clears = BOARD.clears_source_seam(
                seam, engine_facts[label], configuration, product)
            instance = {
                "row": label, "seam": seam,
                "cell": f"({curl_key}, {const_key})",
                "curl_arm": curl_key, "constitutive_arm": const_key,
                "product_on_cell": product,
                "source_field_types": list(engine_facts[label]["source_field_types"]),
                "clears_source_seam": clears,
            }
            per_instance.append(instance)
            if clears:
                continue

            # ---------------------------------------------------- the three groups
            verdict = verdict_of.get((seam, curl_key, const_key), "UNKNOWN")
            refusals = _repair_refusals(seam, configuration)
            instance["fitness_verdict"] = verdict
            instance["repair_refusals"] = refusals
            if product is not None and not BOARD._carries_deposit_repair(product):
                group = "needs CARRIES_DEPOSIT_REPAIR on an existing product"
            elif refusals:
                group = "the repair genuinely does not apply"
            elif verdict in UNBUILDABLE_VERDICTS:
                group = f"the weld itself is refused ({verdict})"
            else:
                group = "needs a fused product that does not exist yet"
            instance["group"] = group
            groups[group].append(instance)

    blocked = sum(len(v) for v in groups.values())
    print(f"\nseam-instances : {len(per_instance)}", flush=True)
    print(f"blocked        : {blocked}", flush=True)

    print("\n--- THE GROUPING ------------------------------------------------",
          flush=True)
    summary: Dict[str, Any] = {}
    for group in sorted(groups, key=lambda g: -len(groups[g])):
        members = groups[group]
        by_cell = collections.Counter(
            f"{m['seam']}  {m['cell']}" for m in members)
        print(f"\n  {len(members):3d}  {group}", flush=True)
        for cell, count in by_cell.most_common():
            print(f"        {count:3d}  {cell}", flush=True)
        summary[group] = {
            "count": len(members),
            "by_cell": dict(by_cell),
            "by_seam": dict(collections.Counter(m["seam"] for m in members)),
        }

    # A cell in "needs a fused product" is the actionable list, ranked.
    build_list = collections.Counter(
        f"{m['seam']}  {m['cell']}"
        for m in groups.get("needs a fused product that does not exist yet", []))
    print("\n--- WHAT BUILDING WOULD UNBLOCK, RANKED --------------------------",
          flush=True)
    for rank, (cell, count) in enumerate(build_list.most_common(), start=1):
        print(f"  {rank}. {count:3d} blocked seam-instances   {cell}", flush=True)

    artifact = {
        "census": str(BOARD.CENSUS),
        "board": str(board_path),
        "seam_instances": len(per_instance),
        "blocked_by_the_source_seam": blocked,
        "groups": summary,
        "ranked_build_list": dict(build_list.most_common()),
        "instances": per_instance,
        "what_this_cannot_read": (
            "deposit_repair.repairable's f_w_<component> clause needs a live engine "
            "object; no census row answers it, so no instance is credited or refused "
            "on it here"),
    }
    (out_dir / "cuda_source_seam_blockage.json").write_text(
        json.dumps(artifact, indent=1, sort_keys=True))
    print(f"\nwrote {out_dir / 'cuda_source_seam_blockage.json'} "
          f"({time.time() - started:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
