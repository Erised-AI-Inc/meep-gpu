#!/usr/bin/env python3
"""Does ``update_E(c)`` read another component's ``P``? — the E->P interleave's licence.

WHY THIS EXISTS. Every fused E->P product on this backend takes the same shape:
ONE LAUNCH PER COMPONENT, ``update_E(c)`` immediately followed by every
``update_P(state, c)``, with the host rotation left where the certified plan
already performs it. That shape is equivalent to the DRIVER's order — all of
``update_E`` at driver.py:3304, then all of ``update_P`` at :3306 — for exactly one
reason, which :mod:`.fused_ade_chain` states and this probe MEASURES:

    ``update_E(c)`` reads only component ``c``'s own ``D``, ``inv_eps`` and ``P``
    volumes, and ``update_P(state, c)`` advances only ``P[c]``.

If that is false the interleave consumes a polarization the driver had not yet
advanced, and every byte claim resting on the shape is void. The fusion matrix
records the consequence: the fifth E->P cell,
``(folded off-diagonal dispersive PML E, ADE update_P)`` — one corpus row,
``examples:absorbed_power_density.py`` — reads ``FITS — 23..23 pointers against a
30 ceiling; NOT BUILT``. This probe asks whether "not built" is the whole story.

HOW IT IS MEASURED, and it is not by reading ``stepping.py``. The dependency is
established by PERTURBATION on the array path: seed two identical engines, change
ONE component's ``P`` in one of them, run ``update_E``, and count the differing
uint32 words in each E component. A component whose E moves is a component whose
``update_E`` READ the perturbed ``P``.

  * **the DIAGONAL control** — with an ordinary ``chi1inv``, perturbing ``P[Ex]``
    must move ``Ex`` and NOTHING else. That is the licence the three shipped chains
    and ``complex_fused_ade_chain`` stand on, measured here on both storage widths
    rather than inherited;
  * **the OFF-DIAGONAL leg** — with a ``chi1inv`` row installed, perturbing
    ``P[Ex]`` must move a PARTNER component's E. That is the refusal: the fused
    kernel would compute ``Ey`` from an ``Ex`` polarization ``update_P(Ex)`` had
    already advanced, and no ordering inside one launch repairs it.

THE NON-VACUITY FLOOR. A perturbation that moved nothing anywhere would report
"no dependency" for every case, so each leg asserts its OWN component moved before
it is allowed to say anything about the partners.

Rule 7: one flushed line per case, every row appended and fsynced as it lands.

Usage (from the repository root)::

    PYTHONPATH=. python -u \\
      parity/meep_gpu/probe_metal_eop_interleave_dependency.py --out <dir>/interleave.json
"""

# Derived from MEEP (https://github.com/NanoComp/meep).
# Copyright (C) 2005-2025 Massachusetts Institute of Technology and MEEP contributors.
# SPDX-License-Identifier: GPL-2.0-or-later

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")
HERE = Path(__file__).resolve().parent
API_ROOT = HERE.parents[1]
for _path in (str(API_ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

from meep_gpu import stepping  # noqa: E402
from meep_gpu.dispersion import LORENTZIAN, PolarizationState, Susceptibility  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

SEED = 41_000
SHAPE = (7, 6, 8)
COMPONENTS = ("Ex", "Ey", "Ez")


def log(message: str) -> None:
    print(message, flush=True)


def case_seed(label: str) -> int:
    """A replayable per-case seed: ``SEED + sha256(label)``, never ``hash()``."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()[:4]
    return SEED + int.from_bytes(digest, "big")


def words(array: Any) -> np.ndarray:
    value = np.ascontiguousarray(array)
    return value.reshape(-1).view(np.uint32)


def differing(left: Any, right: Any) -> int:
    return int(np.count_nonzero(words(left) != words(right)))


def build(label: str, complex_storage: bool, offdiagonal: bool, poles: int = 1
          ) -> Tuple[Fields, Optional[PML]]:
    """One seeded engine, diagonal or with a ``chi1inv`` row installed."""
    seed = case_seed(label)
    grid = Grid(resolution=10.0, cell_size=tuple(n / 10.0 for n in SHAPE),
                boundaries="periodic", dimensions=3, courant=0.35,
                k_point=(0.0, 0.0, 0.0), xp=np)
    fields = Fields(grid=grid, force_complex_fields=complex_storage)
    rng = np.random.default_rng(seed)
    dtype = np.complex64 if complex_storage else np.float32
    epsilon = {name: (1.45 + 0.30 * rng.random(SHAPE)).astype(np.float32)
               for name in COMPONENTS}
    inverse = {name: (np.float32(1.0) / value).astype(np.float32)
               for name, value in epsilon.items()}
    row: Optional[Dict[str, Dict[str, Any]]] = None
    if offdiagonal:
        # MEEP's chi1inv[c][d] for d not the row component's own direction: the
        # outer key is the ROW component, the inner the PARTNER whose D it couples
        # in (Fields.set_epsilon_volumes). A nonzero entry is exactly what makes
        # update_E a stencil across components.
        # EVERY COMPONENT APPEARS AS A PARTNER, and that is a property of this
        # table rather than of off-diagonal rows in general: MEEP's chi1inv rows
        # are per-row and need not be symmetric, so a table missing a column would
        # leave one perturbation with no partner to move and make the criterion
        # below pass or fail for a reason about the FIXTURE. The assertion after
        # the table is what stops that being an accident.
        row = {"Ex": {"Ey": (0.11 + 0.03 * rng.random(SHAPE)).astype(np.float32),
                      "Ez": (0.05 + 0.02 * rng.random(SHAPE)).astype(np.float32)},
               "Ey": {"Ex": (0.09 + 0.02 * rng.random(SHAPE)).astype(np.float32),
                      "Ez": (0.06 + 0.02 * rng.random(SHAPE)).astype(np.float32)},
               "Ez": {"Ex": (0.07 + 0.02 * rng.random(SHAPE)).astype(np.float32),
                      "Ey": (0.08 + 0.02 * rng.random(SHAPE)).astype(np.float32)}}
        partners = {name for entries in row.values() for name in entries}
        assert partners == set(COMPONENTS), (
            f"the chi1inv row table names {sorted(partners)} as partners; a "
            f"component that is nobody's partner cannot couple, and the "
            f"off-diagonal criterion would then be a fact about this fixture")
    fields.set_epsilon_volumes(epsilon, inverse, row)
    for order in range(poles):
        base = {"Ex": 0.31 + 0.05 * order, "Ey": 0.24 + 0.04 * order,
                "Ez": 0.19 + 0.06 * order}
        fields.polarizations.append(PolarizationState(
            Susceptibility(0.62 + 0.19 * order, 0.04 + 0.012 * order, LORENTZIAN),
            base, grid, dtype))
    fields.enable_field_storage()

    def noise(scale: float) -> Any:
        real = (rng.standard_normal(SHAPE) * scale).astype(np.float32)
        if not complex_storage:
            return real
        return (real + 1j * (rng.standard_normal(SHAPE) * scale)).astype(np.complex64)

    for stem in ("B", "D", "E", "H"):
        for axis in "xyz":
            array = getattr(fields, stem + axis, None)
            if array is not None:
                array[...] = noise(0.37)
    for state in fields.polarizations:
        for component in state.driven():
            state.P[component][...] = noise(0.21)
            state.P_prev[component][...] = noise(0.19)
    return fields, PML(grid=grid, thickness=0)


def run(label: str, complex_storage: bool, offdiagonal: bool,
        perturbed: str) -> Dict[str, Any]:
    """Perturb ONE component's ``P`` and count which components' E moved."""
    reference, reference_pml = build(label, complex_storage, offdiagonal)
    actual, actual_pml = build(label, complex_storage, offdiagonal)
    drift = {name: differing(getattr(reference, name), getattr(actual, name))
             for name in COMPONENTS}
    assert not any(drift.values()), f"{label}: the two builds differ: {drift}"

    # ONE WORD IS ENOUGH and one word is what is changed: a whole-array
    # perturbation could not distinguish "Ey read P[Ex]" from "the two engines
    # were never identical". The cell is interior so no boundary rule masks it.
    state = actual.polarizations[0]
    index = (SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2)
    before = state.P[perturbed][index]
    state.P[perturbed][index] = before + type(before)(0.5)
    assert differing(reference.polarizations[0].P[perturbed],
                     state.P[perturbed]) > 0, "the perturbation changed nothing"

    stepping.update_E(reference, reference_pml)
    stepping.update_E(actual, actual_pml)
    moved = {name: differing(getattr(reference, name), getattr(actual, name))
             for name in COMPONENTS}
    own = perturbed
    partners = [name for name in COMPONENTS if name != own]
    return {
        "label": label,
        "complex_storage": complex_storage,
        "offdiagonal": offdiagonal,
        "perturbed": perturbed,
        "differing_words_per_component": moved,
        "own_component_moved": moved[own] > 0,
        "partner_components_moved": sorted(n for n in partners if moved[n] > 0),
        "reads_only_its_own_polarization": (moved[own] > 0
                                            and not any(moved[n] for n in partners)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = args.out.with_suffix(".jsonl")
    started = time.perf_counter()

    rows: List[Dict[str, Any]] = []
    with jsonl.open("w", encoding="utf-8") as handle:
        for complex_storage in (False, True):
            for offdiagonal in (False, True):
                for perturbed in COMPONENTS:
                    label = (f"{'complex' if complex_storage else 'real'}_"
                             f"{'offdiag' if offdiagonal else 'diagonal'}_"
                             f"perturb_{perturbed}")
                    row = run(label, complex_storage, offdiagonal, perturbed)
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    log(f"  {label:40s} own_moved={row['own_component_moved']} "
                        f"partners_moved={row['partner_components_moved']} "
                        f"words={row['differing_words_per_component']}")

    diagonal = [row for row in rows if not row["offdiagonal"]]
    offdiag = [row for row in rows if row["offdiagonal"]]
    # THE FLOOR: every leg's own component must have moved, or the probe measured
    # nothing and its verdict about the partners is empty.
    floor = all(row["own_component_moved"] for row in rows)
    diagonal_clean = all(row["reads_only_its_own_polarization"] for row in diagonal)
    offdiag_couples = all(row["partner_components_moved"] for row in offdiag)

    verdict = {
        "probe": "does update_E(c) read another component's P?",
        # THIS IS A HOST MEASUREMENT AND MAKES NO DEVICE CLAIM. It runs the ARRAY
        # PATH only — no Metal kernel is compiled or launched anywhere in this file
        # — and the runner's `release.released` here means "the bytes were welded
        # and the measurement exited zero", NOT "a device gate passed". Stated in
        # the artifact's own words because a host leg's `released` is copied into
        # `canonical_verdict`, and weld tooling that read this as a passed device
        # gate would be reading a laptop run as a certification.
        "measurement_kind": "host array path (numpy); no Metal kernel is launched",
        "device_claim": False,
        "certifies_a_kernel": False,
        "shape": list(SHAPE), "seed_base": SEED,
        "rows": rows,
        "non_vacuity_floor_every_own_component_moved": floor,
        "diagonal_reads_only_its_own_polarization": diagonal_clean,
        "offdiagonal_couples_every_partner": offdiag_couples,
        "determination": {
            "diagonal": (
                "the per-component interleave IS the driver's order: with an "
                "ordinary chi1inv, update_E(c) reads only component c's own P, so "
                "advancing P[c] immediately after cannot disturb any other "
                "component's constitutive read. This is the licence "
                "fused_ade_chain, fused_ade_chain_dispersive, "
                "folded_fused_ade_chain and complex_fused_ade_chain stand on, and "
                "it is measured here on BOTH storage widths."
                if diagonal_clean else "NOT ESTABLISHED"),
            "offdiagonal": (
                "REFUSED, STRUCTURALLY, AND NOT ON BINDINGS. With a chi1inv row "
                "installed, update_E on one component reads the PARTNER "
                "components' polarizations, so a per-component interleave feeds "
                "the partner's constitutive a P that update_P has ALREADY "
                "advanced — a value the driver's order (all update_E at "
                "driver.py:3304, then all update_P at :3306) never produces. The "
                "fusion matrix's `FITS - 23..23 pointers against a 30 ceiling; "
                "NOT BUILT` for (folded off-diagonal dispersive PML E, ADE "
                "update_P) is a BINDING verdict and it is not the operative one: "
                "the cell is refused at the interleave whatever the signature "
                "costs. The only remaining fused shape would take ALL THREE "
                "components in one launch, which reintroduces exactly the "
                "intra-launch alias triton_kernels.fused_ade_state carries and "
                "fused_ade_chain was built to avoid."
                if offdiag_couples else "NOT ESTABLISHED"),
        },
        "verdict": "PASS" if (floor and diagonal_clean and offdiag_couples) else "FAIL",
        "elapsed_seconds": time.perf_counter() - started,
        "numpy_version": np.__version__,
        "jsonl": str(jsonl),
        "source_sha256": {
            "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    from gate_provenance import stamp as _stamp_provenance  # noqa: PLC0415
    _stamp_provenance(verdict)
    args.out.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str)
                        + "\n", encoding="utf-8")
    log(f"VERDICT {verdict['verdict']} in {verdict['elapsed_seconds']:.2f}s; "
        f"artifact {args.out}")
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from metal_gate_runner import run_current_measurement

    raise SystemExit(run_current_measurement(__file__, sys.argv[1:]))
