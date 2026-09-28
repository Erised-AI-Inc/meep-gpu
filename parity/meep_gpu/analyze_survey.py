"""Read the example survey's JSONL and answer the coverage questions it was run for.

Three tables: what the corpus is made of, what blocks each script, and — the one
that decides where work goes — how many scripts would become liftable if a given
blocker were removed, alone and in combination. Run after
``survey_meep_examples.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from parity.meep_gpu.survey_meep_examples import tag_reason


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", default="parity/meep_gpu/results/survey/survey.jsonl")
    args = parser.parse_args()
    records = load(Path(args.survey))
    with_sim = [r for r in records if r.get("has_simulation")]
    without = [r for r in records if not r.get("has_simulation")]

    print(f"scripts in corpus            : {len(records)}")
    print(f"built an mp.Simulation       : {len(with_sim)}")
    print(f"no mp.Simulation to survey   : {len(without)}")
    for record in sorted(without, key=lambda r: r["script"]):
        print(f"    {record['script']:<42} {record.get('outcome')}  {record.get('error','')[:90]}")

    print("\n--- corpus shape ---")
    dims = Counter(str((r["facts"] or {}).get("effective_dims")) for r in with_sim)
    for key, count in dims.most_common():
        print(f"    {key + 'D':<14} {count}")
    print(f"    with DFT/flux monitors attached at run(): "
          f"{sum(1 for r in with_sim if (r['facts'] or {}).get('n_dft_objects'))}")
    print(f"    k_point declared                        : "
          f"{sum(1 for r in with_sim if (r['facts'] or {}).get('k_point'))}")
    print(f"    homogeneous (no geometry objects)       : "
          f"{sum(1 for r in with_sim if not (r['facts'] or {}).get('n_geometry'))}")

    tagged = {}
    for record in with_sim:
        tagged[record["script"]] = {tag_reason(x) for x in record.get("reasons", [])}

    print("\n--- blockers ranked ---")
    counts = Counter()
    for tags in tagged.values():
        counts.update(tags)
    for tag, count in counts.most_common():
        print(f"    {count:>3}  {tag}")

    print("\n--- unlock analysis: scripts that become liftable if a blocker is removed ---")
    singles = sorted(counts, key=lambda t: -counts[t])
    for tag in singles:
        freed = [name for name, tags in tagged.items() if tags == {tag}]
        if freed:
            print(f"    remove {tag:<24} -> +{len(freed)}  {sorted(freed)}")
    print("    (pairs and triples)")
    for size in (2, 3, 4):
        best: list[tuple[int, tuple[str, ...], list[str]]] = []
        for combo in combinations(singles, size):
            freed = [name for name, tags in tagged.items() if tags and tags <= set(combo)]
            if freed:
                best.append((len(freed), combo, sorted(freed)))
        best.sort(reverse=True)
        for count, combo, freed in best[:3]:
            print(f"    remove {' + '.join(combo):<58} -> {count} liftable")

    print("\n--- already liftable ---")
    for record in with_sim:
        if record.get("supported"):
            facts = record["facts"]
            print(f"    {record['script']}  dims={facts['effective_dims']} "
                  f"cell={facts['cell_size']} k={facts['k_point']} "
                  f"geom={facts['n_geometry']} dft={facts['n_dft_objects']}")

    print("\n--- per script ---")
    for record in sorted(with_sim, key=lambda r: r["script"]):
        tags = sorted(tagged[record["script"]])
        facts = record["facts"]
        print(f"    {record['script']:<42} {str(facts['effective_dims']):>11}D  "
              f"{'LIFTABLE' if record.get('supported') else ','.join(tags)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
