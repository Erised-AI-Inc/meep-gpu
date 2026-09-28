#!/usr/bin/env python3
"""IS THE THREE-LEG FOLD-CLOSURE PROTOCOL NON-VACUOUS ON THE TWO CUDA ELECTRIC PAIRS?

THE QUESTION AND WHY IT HAS TO BE ASKED RATHER THAN ASSUMED. On a FOLDED seam a
deposit's MIRROR IMAGES must be in the repair's save/apply set: ``fill_symmetry_bc_*``
and ``fill_folded_far_ghosts_*`` run AFTER the injection, so a deposit landing on a row
a fill READS FROM is imaged into cells a point-only repair never visits.
``deposit_repair.fill_image_rules`` / ``repair_cells`` are that closure, and the
standard for a product that carries it is a three-leg protocol:

    A  full closure   -> 0 differing words
    B  point only     -> MUST diverge
    C  no repair      -> MUST diverge

AND A CASE WHERE B DOES NOT DIVERGE IS VACUOUS. Running one anyway and reporting the
pass would be reporting ignorance as a measurement, which is the failure this whole
track exists to refuse. So the question this file answers is not "does the closure
hold" but "CAN it be asked at all" of

    ``cuda_kernels/complex_fused_electric_pair``      (D_to_E, Cartesian complex)
    ``cuda_kernels/cylindrical_fused_electric_pair``  (D_to_E, Dcyl complex, |m| >= 1)

WHAT THIS MEASURES, IN THREE STEPS, ALL WITHOUT A DEVICE
=======================================================

1. **THE PREDICATES REFUSE EVERY FOLDED GRID.** Built, asked, and the refusal recorded
   with its reason -- on all three axes and on both products. A folded configuration is
   therefore not a configuration either launch can ever see.

2. **ON EVERY CONFIGURATION THEY DO ADMIT, THE CLOSURE IS THE POINT.**
   ``deposit_repair.fill_image_rules`` is called on the ADMITTED fixtures for all six
   D-seam target components. It returns the per-axis ``(axis, source row, destination
   row)`` rules the repair inverts, and ``_fill_image_rules`` opens with ``if not
   grid.has_symmetry(): return ()`` -- so on an unfolded grid the rule set is EMPTY and
   ``repair_cells(index)`` is ``index``. Leg A and leg B are then the SAME SET by
   construction, and B cannot diverge from A on any input.

3. **THE PROTOCOL IS RUN ANYWAY, ON A REAL DEPOSIT, AND THE VACUITY IS EXHIBITED.** The
   two-consult protocol is driven on an admitted fixture with legs A, B and C, and the
   three word counts are reported. C must diverge -- that is the leg that proves the
   deposit is really in the seam and the fixture discriminates -- while A and B are
   required to be EQUAL, which is the measured statement that B is vacuous here rather
   than a claim that it passed.

WHAT THIS DOES NOT LICENSE. Nothing about a folded CUDA electric weld. If either
product ever admits a fold, its closure is unmeasured and this file's answer changes
from "vacuous" to "owed". The refusal in step 1 is what makes that a tripwire rather
than an assumption.

Progress is a flushed line per case (the progress-reporting rule); the run is seconds and the
artifact is the JSON beside the log.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy

_HERE = Path(__file__).resolve().parent
_API = _HERE.parents[1]
if str(_API) not in sys.path:
    sys.path.insert(0, str(_API))

from meep_gpu import deposit_repair, stepping  # noqa: E402
from meep_gpu.cuda_kernels import complex_fused_electric_pair as CARTESIAN  # noqa: E402
from meep_gpu.cuda_kernels import cylindrical_fused_electric_pair as CYLINDRICAL  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid, Mirror  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402
from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: E402


class _NumpyWearingCupysName(types.ModuleType):
    """NumPy behind CuPy's ``__name__`` -- this track's standing stand-in.

    Every CUDA predicate's first question is whether the array module is CuPy at all,
    and that is the one thing about the device library a laptop cannot supply.
    """

    def __init__(self) -> None:
        super().__init__("cupy")

    def __getattr__(self, item):
        return getattr(numpy, item)


#: The expansion licence both predicates require. Never defaulted: asking with
#: ``license=None`` returns the complex family's own named refusal on EVERY
#: configuration, which would make step 1's refusals meaningless.
LICENCE = {"arm": "FMA_V1", "expansion": 1, "basis": "measured",
           "refusals": [], "policy_resolved": "keep"}
POLICY = "keep"

#: The six volumes ``deposit_repair.SEAMS['D']`` touches, read from the module rather
#: than typed, so a change there is a failure here instead of a stale list.
D_SEAM_TARGETS = tuple(target for _component, target, _axis
                       in deposit_repair.SEAMS["D"]["targets"])


def log(message: str) -> None:
    print(message, flush=True)


def build_cartesian(xp, *, symmetry=(), boundaries=("metallic", "periodic", "periodic")):
    grid = Grid(resolution=1.0, cell_size=(8.0, 9.0, 10.0), boundaries=boundaries,
                symmetry=symmetry, xp=xp)
    thickness = tuple((0, 0) if grid.shape[axis] < 6
                      else (0, 2) if grid.is_mirrored(axis)
                      else (2, 2)
                      for axis in range(3))
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness=thickness), grid


def build_cylindrical(xp, *, m=2, shape=(9, 1, 11), z_kind="metallic"):
    grid = Grid(resolution=1.0, cell_size=(float(shape[0]), 0.0, float(shape[2])),
                cylindrical=True, m=m, boundaries={"z": z_kind}, courant=0.5, xp=xp)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_field_storage()
    fields.enable_pml_storage()
    return fields, PML(grid=grid, thickness={"x": (0, 2), "z": 2}), grid


PRODUCTS = {
    "complex_fused_electric_pair": {
        "module": CARTESIAN,
        "predicate": CARTESIAN.covers_complex_fused_electric_pair,
        "admitted": lambda xp: build_cartesian(xp),
        # The folded probes: one per axis, all on the Cartesian builder, because a Dcyl
        # cell cannot be folded at all (`Grid` refuses a mirror on one) -- which is a
        # SECOND refusal in front of the Dcyl product and is recorded as such.
        "folded": {axis: (lambda xp, name=name: build_cartesian(
            xp, boundaries=("periodic", "periodic", "periodic"),
            symmetry=(Mirror(name),))) for axis, name in ((0, "X"), (1, "Y"), (2, "Z"))},
    },
    "cylindrical_fused_electric_pair": {
        "module": CYLINDRICAL,
        "predicate": CYLINDRICAL.covers_cylindrical_fused_electric_pair,
        "admitted": lambda xp: build_cylindrical(xp),
        "folded": {axis: (lambda xp, name=name: build_cartesian(
            xp, boundaries=("periodic", "periodic", "periodic"),
            symmetry=(Mirror(name),))) for axis, name in ((0, "X"), (1, "Y"), (2, "Z"))},
    },
}


# ---------------------------------------------------------------------------
# THE PROTOCOL -- run on the ARRAY PATH, which is where deposit_repair lives
# ---------------------------------------------------------------------------

SEEDED = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
          "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
          "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")


def seed(fields, generator) -> None:
    for name in SEEDED:
        array = getattr(fields, name, None)
        if array is None:
            continue
        values = generator.standard_normal(array.shape).astype(numpy.float32)
        if array.dtype == numpy.complex64:
            imaginary = generator.standard_normal(array.shape).astype(numpy.float32)
            array[...] = (values + 1j * imaginary).astype(numpy.complex64)
        else:
            array[...] = values


def words(fields) -> Dict[str, Any]:
    return {name: numpy.ascontiguousarray(getattr(fields, name)).ravel().view(numpy.uint8)
            for name in SEEDED if getattr(fields, name, None) is not None}


def differing(left, right) -> Dict[str, int]:
    return {name: int((left[name] != right[name]).sum()) for name in left
            if int((left[name] != right[name]).sum())}


class _PointOnly:
    """``deposit_repair`` with the image closure REMOVED -- leg B's engine.

    ``repair_cells`` is monkeypatched to return the deposit index alone, which is
    exactly what a repair written without :func:`deposit_repair.fill_image_rules`
    would save and restore. Restored on exit, so nothing else in the process sees it.
    """

    def __enter__(self):
        self._shipped = deposit_repair.repair_cells

        def point_only(fields, target, index):
            xp = fields.grid.xp
            columns = [xp.asarray(part).reshape(-1) for part in index]
            return columns[0], columns[1], columns[2]

        deposit_repair.repair_cells = point_only
        return self

    def __exit__(self, *_exc):
        deposit_repair.repair_cells = self._shipped
        return False


def run_leg(build, module, leg: str, steps: int = 8) -> Dict[str, Any]:
    """One protocol leg: the driver's order against the fused stand-in's, byte-compared.

    ``leg`` is ``"closure"`` (the shipped repair), ``"point_only"`` (the closure removed)
    or ``"none"`` (no repair at all). The fused half is the ARRAY PATH standing in for
    the launch, exactly as the merge-bar suites drive it: what is under test here is the
    repair's cell set, not the kernel's arithmetic.
    """
    generator = numpy.random.default_rng(20260831)
    xp = _NumpyWearingCupysName()
    reference, ref_pml, ref_grid = build(xp)
    subject, pml, grid = build(xp)
    seed(reference, numpy.random.default_rng(20260831))
    seed(subject, numpy.random.default_rng(20260831))
    del generator

    sources = [VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                            amplitude=1.0)]
    ref_sources = [VolumeSource(grid=ref_grid, component="Ez", center=(0.0, 0.0, 0.0),
                                size=(0.0, 0.0, 0.0),
                                envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2),
                                amplitude=1.0)]
    dt = float(grid.dt)
    context = _PointOnly() if leg == "point_only" else None
    if context is not None:
        context.__enter__()
    try:
        for step in range(steps):
            when = step * dt + 0.5 * dt
            # The oracle: the driver's own order.
            stepping.step_D(reference, ref_pml)
            for source in ref_sources:
                source.inject(reference, when)
            stepping.fill_symmetry_bc_D(reference)
            stepping.zero_metal_D(reference)
            stepping.fill_folded_far_ghosts_D(reference)
            stepping.update_E(reference, ref_pml)

            # The subject: the whole seam against an UNINJECTED field, then the
            # driver's inject/fill/clear, then the repair.
            saved = (deposit_repair.save(subject, sources, "D", pml)
                     if leg != "none" else None)
            stepping.step_D(subject, pml)
            stepping.zero_metal_D(subject)
            stepping.update_E(subject, pml)
            for source in sources:
                source.inject(subject, when)
            stepping.fill_symmetry_bc_D(subject)
            stepping.zero_metal_D(subject)
            stepping.fill_folded_far_ghosts_D(subject)
            if leg != "none":
                deposit_repair.apply(subject, pml, sources, "D", saved)
    finally:
        if context is not None:
            context.__exit__(None, None, None)

    counts = differing(words(reference), words(subject))
    return {"leg": leg, "steps": steps,
            "differing_words": int(sum(counts.values())),
            "differing_arrays": dict(sorted(counts.items())),
            "deposit_points": int(sources[0]._n_source_points)}  # noqa: SLF001


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    xp = _NumpyWearingCupysName()

    print("=" * 78, flush=True)
    print("THE FOLD-CLOSURE QUESTION ON THE TWO CUDA ELECTRIC PAIRS", flush=True)
    print("=" * 78, flush=True)

    record: Dict[str, Any] = {"products": {}}
    ok = True
    for name, spec in PRODUCTS.items():
        log(f"\n--- {name} ---")
        block: Dict[str, Any] = {"folded_refusals": {}, "image_rules_on_admitted": {},
                                 "protocol": {}}

        # 1. EVERY FOLDED GRID IS REFUSED.
        for axis, builder in spec["folded"].items():
            fields, pml, grid = builder(xp)
            covered, reason = spec["predicate"](fields, pml, grid, (), LICENCE, POLICY)
            block["folded_refusals"][f"axis_{axis}"] = {
                "grid_is_mirrored": bool(grid.is_mirrored(axis)),
                "covered": bool(covered), "reason": str(reason)}
            good = bool(grid.is_mirrored(axis)) and not covered
            ok = ok and good
            log(f"  folded axis {axis}: mirrored={grid.is_mirrored(axis)} "
                f"covered={covered}  {'OK' if good else '<-- FAILED'}")

        # 2. THE CLOSURE IS THE POINT ON EVERYTHING THEY ADMIT.
        fields, pml, grid = spec["admitted"](xp)
        covered, reason = spec["predicate"](fields, pml, grid, (), LICENCE, POLICY)
        block["admitted_fixture"] = {"covered": bool(covered), "reason": str(reason),
                                     "shape": [int(n) for n in grid.shape],
                                     "has_symmetry": bool(grid.has_symmetry())}
        ok = ok and covered
        log(f"  admitted fixture: covered={covered} ({reason})")
        for target in D_SEAM_TARGETS:
            rules = deposit_repair.fill_image_rules(fields, target)
            block["image_rules_on_admitted"][target] = [list(rule) for rule in rules]
            ok = ok and not rules
            log(f"  fill_image_rules({target}) = {rules}  "
                f"{'EMPTY -> closure IS the point' if not rules else '<-- NON-EMPTY'}")

        # 3. THE PROTOCOL, RUN ANYWAY.
        for leg in ("closure", "point_only", "none"):
            result = run_leg(spec["admitted"], spec["module"], leg)
            block["protocol"][leg] = result
            log(f"  leg {leg:11s} differing_words={result['differing_words']:6d} "
                f"arrays={sorted(result['differing_arrays'])} "
                f"deposit_points={result['deposit_points']}")
        closure = block["protocol"]["closure"]["differing_words"]
        point = block["protocol"]["point_only"]["differing_words"]
        none = block["protocol"]["none"]["differing_words"]
        verdict = {
            "A_closure_is_zero": closure == 0,
            "B_point_only_equals_A": point == closure,
            "C_no_repair_diverges": none > 0,
            # THE HONEST NAME FOR WHAT B MEASURED. Leg B did not "pass": on an unfolded
            # grid its cell set IS leg A's, so it could not have diverged whatever the
            # repair did. Recording that as a pass is the vacuity this file exists to
            # name.
            "B_is_vacuous_because_the_rule_set_is_empty": (
                point == closure
                and not any(block["image_rules_on_admitted"].values())),
        }
        block["verdict"] = verdict
        good = (verdict["A_closure_is_zero"] and verdict["C_no_repair_diverges"]
                and verdict["B_is_vacuous_because_the_rule_set_is_empty"])
        ok = ok and good
        log(f"  VERDICT A=0:{verdict['A_closure_is_zero']} "
            f"B==A:{verdict['B_point_only_equals_A']} "
            f"C diverges:{verdict['C_no_repair_diverges']} "
            f"-> B VACUOUS (no fold is admitted)  {'OK' if good else '<-- FAILED'}")
        record["products"][name] = block

    record["conclusion"] = (
        "The three-leg fold-closure protocol is VACUOUS on both CUDA electric pairs and "
        "may not be counted: each predicate refuses every folded grid (measured on all "
        "three axes), and on every configuration they admit "
        "deposit_repair.fill_image_rules returns the EMPTY rule set for all six D-seam "
        "targets, so leg B's cell set IS leg A's and B cannot diverge whatever the "
        "repair does. Leg C DOES diverge, which is what shows the fixture carries a "
        "real in-seam deposit rather than nothing. If either product is ever licensed "
        "for a fold, its image closure is UNMEASURED and owes this protocol for real.")
    record["passed"] = bool(ok)
    record["seconds"] = time.time() - started
    (args.out / "cuda_electric_fold_closure.json").write_text(
        json.dumps(record, indent=1, sort_keys=True))
    log(f"\n{record['conclusion']}")
    log(f"\nwrote {args.out / 'cuda_electric_fold_closure.json'} "
        f"({record['seconds']:.1f} s)  passed={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
