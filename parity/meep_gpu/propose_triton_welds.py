#!/usr/bin/env python3
"""Turn a gate campaign's artifacts into PROPOSED weld entries — and verify them.

WHAT A WELD IS AND WHY IT NEEDS THIS. An entry in ``triton_kernels/fingerprints.json``
says "this device gate released, and here are the exact bytes it ran against".
``test_dispatch_contract.test_the_no_absorber_gate_records_are_welded_to_the_live_sources``
recomputes every named digest from the live tree, so a weld that names a path
whose bytes have moved fails the merge bar — and a weld that names a path the
GATE NEVER IMPORTED is worse than that: it passes the test while asserting a
device result about bytes no device executed. Two attempts this week were
rejected for exactly that, both naming a test file no gate imports.

So this script never types a path. For each gate it takes the intersection of

* :data:`BINDS` — the modules that family's arm actually launches through, plus
  the gate script itself, stated once here with a reason; and
* the run's own ``imported_source_sha256`` — every repo module the gate PROCESS
  imported, enumerated from ``sys.modules`` by :mod:`gate_provenance`.

A path in ``BINDS`` that the run did not import is DROPPED and reported, never
proposed. A path the run imported that ``BINDS`` does not name is not proposed
either — the imported set is 30-40 modules wide and includes the whole package's
import graph; a weld is a statement about what the family binds, not a snapshot
of the interpreter.

Every proposed digest is then recomputed FROM THE LIVE LOCAL TREE and compared
to the artifact's. A mismatch means the staged tree the gate ran on and the tree
the weld would describe are different programs, and the entry is refused rather
than proposed.

NOTHING HERE WRITES ``fingerprints.json``. Output is a proposal packet.

Usage::

    python parity/meep_gpu/propose_triton_welds.py \\
        --results parity/meep_gpu/results/triton_welds_2026-08-19 \\
        --out /tmp/proposals.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]

K = "meep_gpu/triton_kernels/"
G = "parity/meep_gpu/"

#: What each family's weld would BIND, and why that set and not another.
#:
#: The shape is the one the ten existing welds use: the kernel module(s) whose
#: bytes the measured launches executed, the host module that binds the pointers
#: and the constexprs (``launch.py``), the predicate that decides whether the
#: kernel runs at all (``coverage.py``) where the gate exercises it, and the gate
#: script — which is ``__main__`` in the run, so it is always in the imported set
#: and is the one file that decides what was measured.
BINDS: Dict[str, Sequence[str]] = {
    "unified_expansion": (
        K + "complex_fields.py", K + "folded_complex.py", K + "complex_ade.py",
        G + "gate_triton_complex.py", G + "gate_triton_folded_complex.py",
        G + "gate_triton_complex_ade.py", G + "gate_triton_unified_expansion.py"),
    "complex": (K + "complex_fields.py", K + "kernels.py", K + "launch.py",
                G + "gate_triton_complex.py"),
    "fused_electric": (K + "kernels.py", K + "launch.py", K + "coverage.py",
                       G + "gate_triton_fused_electric.py"),
    "no_pml": (K + "no_pml.py", K + "launch.py", K + "coverage.py",
               G + "gate_triton_no_pml.py"),
    "no_pml_constitutive": (K + "no_pml_constitutive.py", K + "coverage.py",
                            K + "launch.py",
                            G + "gate_triton_no_pml_constitutive.py"),
    "conductivity": (K + "conductivity.py", K + "launch.py",
                     G + "gate_triton_conductivity.py"),
    "symmetry": (K + "symmetry.py", K + "launch.py",
                 G + "gate_triton_symmetry.py"),
    "cylindrical": (K + "cylindrical_triton.py", K + "kernels.py",
                    K + "launch.py", G + "gate_triton_cylindrical.py"),
    "offdiag": (K + "offdiag_update_e.py", K + "kernels.py", K + "launch.py",
                G + "gate_triton_offdiag.py"),
    "nonlinear": (K + "nonlinear_update_e.py", K + "kernels.py", K + "launch.py",
                  G + "gate_triton_nonlinear.py"),
    "bfast": (K + "bfast_curl.py", K + "kernels.py", K + "launch.py",
              G + "gate_triton_bfast.py"),
    "special_kz": (K + "special_kz.py", K + "complex_fields.py", K + "launch.py",
                   G + "gate_triton_special_kz.py"),
    "folded_complex": (K + "folded_complex.py", K + "complex_fields.py",
                       K + "special_kz.py", K + "symmetry.py", K + "launch.py",
                       G + "gate_triton_folded_complex.py"),
    "cylindrical_complex": (K + "cylindrical_complex.py",
                            K + "complex_fields.py", K + "launch.py",
                            G + "gate_triton_cylindrical_complex.py"),
    # THE COMPLEX D->E FUSED PAIR, 2026-08-30. The launched bytes are the product
    # module's own kernel; ``complex_fields.py`` because both certified halves it
    # transcribes AND the three multiply helpers its body calls live there;
    # ``coverage.py`` because the seam predicate reads ``zero_metal_axes`` and
    # ``CONSTITUTIVE_SIDES`` from it and the gate exercises that predicate in both
    # directions; ``launch.py`` for ``SUB_STEPS``, ``CupyPointer`` and ``_flat``,
    # which decide every pointer and constexpr the launch binds; and
    # ``deposit_repair.py``, which is NOT decoration on this family — the carry
    # cases run through its shipped ``LeadingRepairPlan``/``TrailingRepairPlan`` and
    # three of the eight device legs measure nothing without it.
    "complex_fused_electric_pair": (
        K + "complex_fused_electric_pair.py", K + "complex_fields.py",
        K + "kernels.py", K + "coverage.py", K + "launch.py",
        "meep_gpu/deposit_repair.py",
        G + "probe_triton_complex_fused_electric_pair.py"),
    # THE FOLDED OFF-DIAGONAL E->P CHAIN, 2026-09-10. The launched bytes are this
    # module's own kernel PLUS the JIT helpers it calls: ``_load_d_minus_p`` and
    # ``_folded_dispersive_term`` execute out of
    # ``folded_offdiag_dispersive_update_e.py`` and ``_masked_row_sum`` out of
    # ``folded_offdiag_update_e.py``, so neither is decoration — an edit to
    # either changes the arithmetic this gate measured. ``offdiag_update_e.py``
    # decides the six row slots and the wall mask, ``dispersive_update_e.py``
    # the pole partition and the live binding, ``symmetry.py`` the boundary
    # codes and the ghost weights, ``coverage.py`` the ADE half's admission and
    # ``sigma_is_volume``, ``kernels.py`` the recurrence line this kernel
    # transcribes, and ``launch.py`` the pointer and constexpr machinery every
    # binding goes through.
    "folded_offdiag_fused_ade_chain": (
        K + "folded_offdiag_fused_ade_chain.py",
        K + "folded_offdiag_dispersive_update_e.py",
        K + "folded_offdiag_update_e.py", K + "offdiag_update_e.py",
        K + "dispersive_update_e.py", K + "symmetry.py", K + "coverage.py",
        K + "kernels.py", K + "launch.py",
        G + "gate_triton_folded_offdiag_fused_ade_chain.py"),
}

#: The ``fingerprints.json`` key each family's entry would take, in the naming
#: the ten existing welds already use.
ENTRY_KEY = {name: f"triton_{name}_device_gate" for name in BINDS}


def sha256_of(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def one_gate(name: str, artifact: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {"gate": name, "artifact": str(artifact)}
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        row["error"] = f"artifact unreadable: {exc!r}"
        return row

    canonical = payload.get("canonical_verdict") or {}
    imported: Dict[str, str] = payload.get("imported_source_sha256") or {}
    row.update({
        "released": canonical.get("released"),
        "read_from": canonical.get("read_from"),
        "reasons": canonical.get("reasons") or [],
        "unreadable": canonical.get("unreadable"),
        "imported_count": len(imported),
        "artifact_sha256": sha256_of(artifact),
    })
    stamp = payload.get("subnormal_policy")
    row["subnormal_policy"] = (stamp.get("policy") if isinstance(stamp, dict)
                               else stamp)

    # THE GATE'S OWN WORDS, beside the canonical reading. ``released: null``
    # means UNREADABLE, not refused, and the difference is decidable only from
    # the keys the gate actually wrote: measured 2026-08-19,
    # gate_triton_no_pml_constitutive finished every leg green and exited 0
    # while reading as null, because it spells its verdict ``all_ok`` — an
    # ELEVENTH spelling on top of the ten read_verdict knows.
    native: Dict[str, Any] = {}
    for key in ("all_ok", "status", "passed", "certifies", "seconds"):
        if key in payload:
            native[key] = payload[key]
    for key in ("summary", "verdict", "validation"):
        value = payload.get(key)
        if isinstance(value, (bool, str, int)):
            native[key] = value
        elif isinstance(value, dict):
            native[key] = {k: v for k, v in value.items()
                           if isinstance(v, (bool, str, int, float, dict, list))}
    row["native_verdict"] = native

    # EVERY imported digest against the live tree, not only the ones proposed.
    # A family whose weld would name four files but whose run imported a fifth
    # that has since moved is still a run against a tree that no longer exists,
    # and the campaign's reader should see that before the proposal.
    drifted, missing = {}, []
    for path, digest in sorted(imported.items()):
        live = sha256_of(API_ROOT / path)
        if live is None:
            missing.append(path)
        elif live != digest:
            drifted[path] = {"artifact": digest, "live": live}
    row["imported_drifted"] = drifted
    row["imported_missing"] = missing

    binds = BINDS.get(name, ())
    proposed, not_imported, bad = {}, [], {}
    for path in binds:
        if path not in imported:
            not_imported.append(path)
            continue
        live = sha256_of(API_ROOT / path)
        if live != imported[path]:
            bad[path] = {"artifact": imported[path], "live": live}
            continue
        proposed[path] = imported[path]
    row["source_sha256"] = proposed
    row["dropped_not_imported"] = not_imported
    row["dropped_drifted"] = bad
    row["proposable"] = bool(
        row["released"] is True and proposed and not bad and not not_imported)
    row["entry_key"] = ENTRY_KEY.get(name)
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", required=True,
                        help="campaign directory holding <gate>/gate.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    root = Path(args.results).resolve()
    rows = []
    for name in BINDS:
        artifact = root / name / "gate.json"
        if not artifact.exists():
            print(f"{name:<22} NO ARTIFACT at {artifact}", flush=True)
            continue
        row = one_gate(name, artifact)
        rows.append(row)
        print(f"{name:<22} released={str(row.get('released')):<6} "
              f"imported={row.get('imported_count', 0):<4} "
              f"proposed={len(row.get('source_sha256', {})):<3} "
              f"drifted={len(row.get('imported_drifted', {})):<3} "
              f"dropped={len(row.get('dropped_not_imported', []))} "
              f"proposable={row.get('proposable')}", flush=True)
        for path in row.get("dropped_not_imported", ()):
            print(f"    NOT IMPORTED, so not proposed: {path}", flush=True)
        for path, pair in (row.get("dropped_drifted") or {}).items():
            print(f"    DRIFTED from the live tree: {path} "
                  f"{pair['artifact'][:12]}... -> {str(pair['live'])[:12]}...",
                  flush=True)

    packet = {"results": str(root), "api_root": str(API_ROOT), "gates": rows}
    text = json.dumps(packet, indent=1, sort_keys=True, default=str) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
