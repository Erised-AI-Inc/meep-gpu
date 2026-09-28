"""Composition probe for the Metal complex-field / Bloch family.

WHAT A GATE CANNOT ANSWER AND THIS DOES. ``gate_metal_complex.py`` certifies one
sub-step at a time against ``stepping.py``. A COMPOSITION is a different claim: four
device sub-steps in a row, sharing ONE residency, with the driver's seam work
(source fills, wall clears) happening between them on the host. Three things can go
wrong there that no single-sub-step gate can see:

1. **a stale mirror.** ``step_B`` writes ``Bx`` and ``update_H`` reads it. If the
   two plans held private copies the second launch would read the first one's stale
   bytes and produce a smooth, plausible, entirely wrong field. Measured here by
   holding one ``Residency`` across a whole cycle and asserting
   ``Residency.verify()`` is empty — over a NON-EMPTY registry, because an empty
   dict from an empty registry proves nothing;
2. **the seam.** A source fill deposits into B or D between ``step_B`` and
   ``update_H``; on the host, into an array the device is mirroring. The residency
   predicate REFUSES that composition unless the caller declares the sync, and this
   probe measures that the refusal is load-bearing by running the seam BOTH ways —
   declared-and-synced (must match the array path) and undeclared (must diverge);
3. **the zero-init census.** From an all-zero state the array path stores ``-0.0``
   wherever a thin absorber's deepest ``kms = kappa - sigma`` goes negative. That is
   the class the Triton complex tranche found REACHABLE at stored bytes, and it is
   why the negation spelling was load-bearing there. This probe censuses it here so
   the Metal claim rests on the same measured reachability rather than on Triton's.

NOT A GATE. It writes an artifact and asserts, but its subject is the COMPOSITION
rather than the arithmetic; the bytes were certified next door. It shares the gate's
kit and its expansion probe artifact.

Progress is one flushed line per case, the artifact is rewritten atomically after
every case, and the exit code is 0 / 1 / 75 exactly as the gate's.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
API_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")

from meep_gpu import stepping  # noqa: E402
from meep_gpu.fields import Fields  # noqa: E402
from meep_gpu.grid import Grid  # noqa: E402
from meep_gpu.metal_kernels import complex_fields as cx  # noqa: E402
from meep_gpu.metal_kernels import coverage as metal_coverage  # noqa: E402
from meep_gpu.metal_kernels import device, preconditions, shaders, subnormal  # noqa: E402
from meep_gpu.pml import PML  # noqa: E402

import metal_gate_kit as kit  # noqa: E402

log, save, differing = kit.log, kit.save, kit.differing

STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz",
         "fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
         "f_w_Hx", "f_w_Hy", "f_w_Hz", "f_w_Ex", "f_w_Ey", "f_w_Ez")

CURL = {"step_B": {"targets": ("Bx", "By", "Bz"), "sources": ("Ex", "Ey", "Ez"),
                   "backward": False, "suffix": "_h"},
        "step_D": {"targets": ("Dx", "Dy", "Dz"), "sources": ("Hx", "Hy", "Hz"),
                   "backward": True, "suffix": ""}}
SIDES = {"H": {"targets": ("Hx", "Hy", "Hz"),
               "aux": ("f_w_Hx", "f_w_Hy", "f_w_Hz"),
               "sources": ("Bx", "By", "Bz"), "suffix": "", "step": "update_H"},
         "E": {"targets": ("Ex", "Ey", "Ez"),
               "aux": ("f_w_Ex", "f_w_Ey", "f_w_Ez"),
               "sources": ("Dx", "Dy", "Dz"), "suffix": "_h", "step": "update_E"}}

#: (name, boundaries, k_point). The metallic case is the one with LIVE WALL CLEARS,
#: which is the seam a residency model that enumerated only the four STEP_ORDER
#: slots would hold a mirror straight across.
CASES: Tuple[Tuple[str, Any, Tuple[float, float, float]], ...] = (
    ("periodic_kx", "periodic", (0.3, 0.0, 0.0)),
    ("periodic_edge", "periodic", (0.5, 0.0, 0.0)),
    ("metallic_xy_kz", ("metallic", "metallic", "periodic"), (0.0, 0.0, 0.4)),
)

EXPANSION: Optional[str] = None


def complex_from_planes(real: Any, imag: Any) -> Any:
    """``real + 1j*imag`` destroys the sign of zeros in ``imag``. Word view instead."""
    out = np.empty(np.shape(real), dtype=np.complex64)
    view = out.view(np.float32)
    view[..., 0::2] = np.asarray(real, dtype=np.float32)
    view[..., 1::2] = np.asarray(imag, dtype=np.float32)
    return out


def build(boundaries, k_point, seed: int, zero_init: bool = False,
          thin_absorber: bool = False):
    grid = Grid(resolution=10.0, cell_size=(1.2, 1.0, 0.9), boundaries=boundaries,
                dimensions=3, courant=0.35, k_point=k_point, xp=np)
    fields = Fields(grid=grid, force_complex_fields=True)
    fields.enable_pml_storage()
    count = int(np.prod(grid.shape))
    index = np.arange(count, dtype=np.float32).reshape(grid.shape)
    epsilon = (1.45 + 0.30 * np.sin(index * np.float32(0.037))).astype(np.float32)
    fields.set_isotropic_epsilon_volume(
        epsilon, (np.float32(1.0) / epsilon).astype(np.float32))
    # A THIN absorber is what drives `kms = kappa - sigma` NEGATIVE at its deepest
    # cell, and a negative coefficient times a quiet +0.0 is what stores a -0.0.
    # Without it the zero-init census is vacuous.
    thickness = (1, 1) if thin_absorber else (2, 2)
    pml = PML(grid=grid,
              thickness=tuple(thickness if grid.shape[a] >= 6 else (0, 0)
                              for a in range(3)))
    rng = np.random.default_rng(seed)
    for name in STATE:
        array = getattr(fields, name, None)
        if array is None:
            continue
        if zero_init:
            array[...] = np.complex64(0)
            continue
        real = (rng.standard_normal(grid.shape) * 0.37).astype(np.float32)
        imag = (rng.standard_normal(grid.shape) * 0.29).astype(np.float32)
        real.reshape(-1)[::17] = np.float32(-0.0)
        imag.reshape(-1)[3::19] = np.float32(-0.0)
        array[...] = complex_from_planes(real, imag)
    return grid, fields, pml


def snapshot(fields) -> Dict[str, Any]:
    return {n: np.array(getattr(fields, n), copy=True) for n in STATE
            if getattr(fields, n, None) is not None}


def restore(fields, state: Dict[str, Any]) -> None:
    for name, value in state.items():
        getattr(fields, name)[...] = value


def build_plans(fields, pml, residency, probe):
    """One residency, four plans — the sharing this probe exists to measure."""
    plans: Dict[str, Any] = {}
    for sub_step in CURL:
        plan = cx.plan_complex_pml_curl(fields, pml, sub_step, residency,
                                        (shaders.CONTRACT_OFF,), probe=probe)
        assert plan is not None, cx.complex_pml_curl_coverage(
            fields, pml, sub_step, residency, probe).reasons
        plans[sub_step] = plan
    for side in SIDES:
        plan = cx.plan_complex_constitutive(fields, pml, side, residency,
                                            (shaders.CONTRACT_OFF,), probe=probe)
        assert plan is not None, cx.complex_constitutive_coverage(
            fields, pml, side, residency, probe).reasons
        plans["update_" + side] = plan
    return plans


# ---------------------------------------------------------------------------
# CASE 1 — a whole cycle on ONE residency
# ---------------------------------------------------------------------------

def case_cycle(payload: Dict[str, Any], out: str, probe: Any, cycles: int = 4) -> None:
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, boundaries, k_point in CASES:
        grid, fields, pml = build(boundaries, k_point, 20260815)
        base = snapshot(fields)
        for _ in range(cycles):
            stepping.step_B(fields, pml)
            stepping.update_H(fields, pml)
            stepping.step_D(fields, pml)
            stepping.update_E(fields, pml)
        oracle = snapshot(fields)

        restore(fields, base)
        residency = device.Residency()
        plans = build_plans(fields, pml, residency, probe)
        assert residency.names, "the mirror registry is EMPTY; verify() would pass"
        for _ in range(cycles):
            for slot in ("step_B", "update_H", "step_D", "update_E"):
                plans[slot].run()
        residency.sync_out()
        drift = residency.verify()
        got = snapshot(fields)
        bad = {n: differing(got[n], oracle[n]) for n in oracle}
        moved = sum(differing(oracle[n], base[n]) for n in oracle)
        row = {"case": name, "cycles": cycles, "mirrors": len(residency.names),
               "moved_words": moved, "residency_drift": drift,
               "launches": {slot: plans[slot].launches for slot in plans},
               "differing": {k: v for k, v in bad.items() if v}}
        rows.append(row)
        log(f"[cycle] {name:<16} cycles={cycles} mirrors={len(residency.names)} "
            f"moved={moved} drift={len(drift)} "
            f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
            f"({time.time() - started:.1f}s)")
        payload["cases"]["cycle"] = rows
        save(payload, out)
        kit.assert_moved(moved, f"{name} {cycles} cycles")
        assert not row["differing"], row
        assert not drift, (
            f"{name}: the host arrays and their device mirrors DISAGREE after a "
            f"complete composition ({drift}); a mirror that drifts is a smooth, "
            f"plausible, wrong field rather than an error")
        assert all(count == cycles for count in row["launches"].values()), row


# ---------------------------------------------------------------------------
# CASE 2 — the seam, measured BOTH ways
# ---------------------------------------------------------------------------

def case_seam(payload: Dict[str, Any], out: str, probe: Any) -> None:
    """A host write between two device sub-steps: declared-and-synced vs not.

    THIS IS WHERE THE RESIDENCY PREDICATE EARNS ITS REFUSAL. The undeclared run must
    DIVERGE from the array path — if it did not, the mirror would not be
    load-bearing and the clause would be certifying nothing.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    name, boundaries, k_point = CASES[0]
    grid, fields, pml = build(boundaries, k_point, 20260815)
    base = snapshot(fields)
    deposit = (complex_from_planes(
        np.full(grid.shape, np.float32(0.013)),
        np.full(grid.shape, np.float32(-0.007))))

    # The array path WITH the seam write: the answer the driver would produce.
    restore(fields, base)
    stepping.step_B(fields, pml)
    for component in ("Bx", "By", "Bz"):
        getattr(fields, component)[...] += deposit
    stepping.update_H(fields, pml)
    oracle = snapshot(fields)

    for declared in (True, False):
        restore(fields, base)
        residency = device.Residency()
        plans = build_plans(fields, pml, residency, probe)
        plans["step_B"].run()
        if declared:
            # The caller BRACKETS the seam: device -> host, seam, host -> device.
            residency.sync_out()
        for component in ("Bx", "By", "Bz"):
            getattr(fields, component)[...] += deposit
        if declared:
            residency.sync_in()
        plans["update_H"].run()
        residency.sync_out()
        got = snapshot(fields)
        names = ("Hx", "Hy", "Hz", "f_w_Hx", "f_w_Hy", "f_w_Hz")
        bad = sum(differing(got[n], oracle[n]) for n in names)
        # All four sub-steps are PLANNED (the plans were built and share the
        # residency); only two are launched in this measurement, which is the seam
        # being isolated. The declaration under test is `fill_B`.
        verdict = metal_coverage.residency_coverage(
            residency.names,
            ("step_B", "update_H", "step_D", "update_E"),
            ("step_B", "fill_B", "update_H", "step_D", "update_E"),
            ("fill_B",) if declared else ())
        row = {"case": name, "seam": "fill_B", "declared": declared,
               "differing_words": bad, "residency_covered": bool(verdict.covered),
               "residency_reasons": list(verdict.reasons)}
        rows.append(row)
        log(f"[seam] declared={declared} differing={bad} "
            f"residency_covered={verdict.covered} ({time.time() - started:.1f}s)")
        payload["cases"]["seam"] = rows
        save(payload, out)
        if declared:
            assert bad == 0, row
            assert verdict.covered, row
        else:
            assert bad > 0, (
                "the UNDECLARED seam produced the array path's bytes anyway, so the "
                "host write was not load-bearing and the residency clause certifies "
                "nothing")
            assert not verdict.covered, row
            assert any("stale" in reason for reason in verdict.reasons), row


# ---------------------------------------------------------------------------
# CASE 3 — the zero-init census, and the subnormal window over the composition
# ---------------------------------------------------------------------------

def case_zero_init(payload: Dict[str, Any], out: str, probe: Any,
                   cycles: int = 3) -> None:
    """From an all-zero state, with a THIN absorber whose deepest kms is negative.

    THE CLASS THIS MEASURES: a negative coefficient times a quiet ``+0.0`` stores a
    ``-0.0``. On the Triton track that class was what made the negation spelling
    load-bearing; here the negation spelling was already refuted directly on the
    exhaustive table, and this case measures whether the class is REACHED IN A
    COMPOSITION at stored bytes — which is a different question and the one that
    says whether it matters in a run.

    The subnormal window runs over the same composition, so the artifact carries the
    precondition for the COMPOSED claim and not only for a single sub-step.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, boundaries, k_point in CASES:
        grid, fields, pml = build(boundaries, k_point, 20260815, zero_init=True,
                                  thin_absorber=True)
        base = snapshot(fields)
        window = preconditions.SubnormalWindow(0, cycles, per_array_words=64)
        for volume, array in sorted(base.items()):
            window.observe(f"in:{volume}", array, step=0)
        for _ in range(cycles):
            stepping.step_B(fields, pml)
            stepping.update_H(fields, pml)
            stepping.step_D(fields, pml)
            stepping.update_E(fields, pml)
        oracle = snapshot(fields)
        for volume, array in sorted(oracle.items()):
            window.observe(f"out:{volume}", array, step=cycles)

        restore(fields, base)
        residency = device.Residency()
        plans = build_plans(fields, pml, residency, probe)
        for _ in range(cycles):
            for slot in ("step_B", "update_H", "step_D", "update_E"):
                plans[slot].run()
        residency.sync_out()
        got = snapshot(fields)
        bad = {n: differing(got[n], oracle[n]) for n in oracle}
        census = {n: subnormal.signed_zero_census(oracle[n])["negative_zero"]
                  for n in oracle}
        negative_zeros = sum(census.values())
        report = preconditions.assert_clean_or_refuse(
            window, f"{name} zero-init composition")
        row = {"case": name, "cycles": cycles,
               "negative_zero_words": negative_zeros,
               "negative_zero_by_volume": {k: v for k, v in census.items() if v},
               "subnormal": report["subnormal_words"],
               "observed_words": report["observed_words"],
               "differing": {k: v for k, v in bad.items() if v}}
        rows.append(row)
        log(f"[zero_init] {name:<16} neg_zero_words={negative_zeros} "
            f"subnormal={report['subnormal_words']} "
            f"{'IDENTICAL' if not row['differing'] else 'DIFFERS'} "
            f"({time.time() - started:.1f}s)")
        payload["cases"]["zero_init"] = rows
        save(payload, out)
        assert not row["differing"], row
        kit.assert_census_floor(negative_zeros, f"{name} zero-init composition")


# ---------------------------------------------------------------------------
# CASE 4 — the live set, and the residency verdict this family earns
# ---------------------------------------------------------------------------

def case_residency_model(payload: Dict[str, Any], out: str, probe: Any) -> None:
    """Which sub-steps are LIVE, and what the verdict is for each declaration.

    ``None`` for the live set is a REFUSAL and not an empty set: which seam work runs
    depends on the sources, the walls and the poles, and none of those is readable
    from a plan. Measured here rather than asserted, including the walled case whose
    ``zero_metal_*`` passes a four-slot residency model would silently hold a mirror
    across.
    """
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for name, boundaries, k_point in CASES:
        grid, fields, pml = build(boundaries, k_point, 20260815)
        residency = device.Residency()
        build_plans(fields, pml, residency, probe)
        walls = any(metal_coverage.zero_metal_axes(grid))
        planned = ("step_B", "update_H", "step_D", "update_E")
        wall_passes = ("zero_metal_B", "zero_metal_D") if walls else ()
        live_no_sources = planned + wall_passes
        verdicts = {
            "undeclared_live": metal_coverage.residency_coverage(
                residency.names, planned, None, ()),
            # THE WALLED CASE IS THE POINT of enumerating seam work at all: a
            # residency model that listed only the four STEP_ORDER slots would call
            # this covered, and it is not — `zero_metal_B`/`_D` clear stored cell 0
            # on the HOST every sub-step (stepping.py:2211/:2238).
            "no_seam_work_undeclared": metal_coverage.residency_coverage(
                residency.names, planned, live_no_sources, ()),
            "walls_declared": metal_coverage.residency_coverage(
                residency.names, planned, live_no_sources, wall_passes),
            "with_an_undeclared_source_fill": metal_coverage.residency_coverage(
                residency.names, planned, live_no_sources + ("fill_D",),
                wall_passes),
            "fully_declared": metal_coverage.residency_coverage(
                residency.names, planned, live_no_sources + ("fill_D",),
                wall_passes + ("fill_D",)),
        }
        row = {"case": name, "metallic_walls": walls,
               "mirrors": len(residency.names),
               "verdicts": {k: {"covered": bool(v.covered),
                                "reasons": list(v.reasons)}
                            for k, v in verdicts.items()}}
        rows.append(row)
        log(f"[residency] {name:<16} walls={walls} " +
            " ".join(f"{k}={int(v.covered)}" for k, v in verdicts.items()) +
            f" ({time.time() - started:.1f}s)")
        payload["cases"]["residency_model"] = rows
        save(payload, out)
        assert not verdicts["undeclared_live"].covered, row
        assert any("live sub-step set was not declared" in r
                   for r in verdicts["undeclared_live"].reasons), row
        assert not verdicts["with_an_undeclared_source_fill"].covered, row
        assert verdicts["fully_declared"].covered, row
        assert verdicts["walls_declared"].covered, row
        # A four-slot residency model would call the WALLED case covered here. It is
        # not, and the difference between the two rows is the whole reason the model
        # enumerates seam work.
        assert verdicts["no_seam_work_undeclared"].covered != walls, row
        if walls:
            assert any("zero_metal" in r
                       for r in verdicts["no_seam_work_undeclared"].reasons), row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = kit.argument_parser(__doc__)
    parser.add_argument("--probe", default="",
                        help="expansion probe artifact (defaults to the environment)")
    arguments = parser.parse_args()
    out_dir = os.path.dirname(os.path.abspath(arguments.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    payload: Dict[str, Any] = {
        "environment": kit.environment_stamp(),
        "subnormal_policy": subnormal.mps_policy_report(),
        "cases": {},
        "counts": {},
    }
    save(payload, arguments.out)

    reasons: List[str] = []
    if not payload["environment"].get("mps_available"):
        reasons.append("no MPS device is available on this host")
    if not payload["subnormal_policy"]["admitted"]:
        reasons.extend(payload["subnormal_policy"]["reasons"])
    probe = cx.load_expansion_probe(arguments.probe or None)
    if probe is None or cx.expansion_from_probe(probe) is None:
        reasons.append(
            f"no usable complex-expansion probe artifact (pass --probe or set "
            f"{cx.PROBE_PATH_ENVIRONMENT}); the arm is a MEASURED platform fact and "
            f"this probe will not guess it. Run gate_metal_complex.py --legs "
            f"expansion to write one")
    if reasons:
        return kit.cannot_certify(payload, arguments.out, reasons)

    global EXPANSION
    EXPANSION = cx.expansion_from_probe(probe)
    payload["expansion_arm"] = EXPANSION
    payload["provenance"] = kit.provenance(out_dir, {
        "stepping.py": os.path.join(API_ROOT, "meep_gpu", "stepping.py"),
        "metal_kernels/complex_fields.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "complex_fields.py"),
        "metal_kernels/device.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "device.py"),
        "metal_kernels/coverage.py": os.path.join(
            API_ROOT, "meep_gpu", "metal_kernels", "coverage.py"),
        "probe_metal_complex_composition.py": os.path.abspath(__file__),
        # A DISTINCT NAME. This probe and `gate_metal_complex.py` write into ONE
        # results directory and both called `kit.provenance` with the default, so
        # whichever ran second replaced the other's record wholesale — measured on
        # results/metal_complex_2026-08-15/, where the surviving provenance.json
        # holds these five sources and not the gate's ten. That is the exact loss
        # the kit's `name` parameter was added for.
    }, kernel_sources=cx.enumerate_complex_sources(EXPANSION),
        name="provenance_composition.json")
    save(payload, arguments.out)
    log(f"[env] arm={EXPANSION} torch={payload['environment'].get('torch')} "
        f"policy={payload['subnormal_policy']['resolved']}")

    cases = (("cycle", case_cycle), ("seam", case_seam),
             ("zero_init", case_zero_init),
             ("residency_model", case_residency_model))
    wanted = kit.wanted_legs(arguments.legs)
    ran: List[str] = []
    for name, function in cases:
        if wanted and name not in wanted:
            log(f"[skip] case {name} (not in --legs)")
            continue
        log(f"=== CASE {name} ===")
        function(payload, arguments.out, probe)
        ran.append(name)
        payload["counts"][name] = len(payload["cases"].get(name, []))
    compared = sum(int(v) for v in payload["counts"].values())
    return kit.summarize(
        payload, arguments.out,
        claim=("a four-sub-step complex/Bloch composition on ONE residency "
               "reproduces stepping.py byte for byte, the mirrors do not drift, the "
               "seam refusal is load-bearing, and the zero-init signed-zero class is "
               "REACHED at stored bytes"),
        scope=("complex64 storage with per-axis Bloch phases; periodic and metallic "
               "ghost rules; the fill_B seam measured both ways; residency verdicts "
               "for the undeclared, unsynced and synced declarations"),
        stated_weakness=("no PTX-equivalent audit exists on this executor, and the "
                         "subnormal window censuses STORED state plus operands, not "
                         "the kernel's own registers"),
        started=started, legs_run=ran, compared=compared, certified=True)


if __name__ == "__main__":
    raise SystemExit(main())
