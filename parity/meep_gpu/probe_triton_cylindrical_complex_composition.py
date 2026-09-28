"""Complete-driver composition probe for the COMPLEX cylindrical (Dcyl, complex64
storage at every m) tranche.

m = 0 UNDER COMPLEX STORAGE (2026-09-04, ``cylindrical_complex.M_ZERO``): three
cases below step it through complete driver steps — one at the fixture shape with
a Gaussian source, one at the corpus row's OWN shape, Courant, m and z termination
(examples:dipole_in_vacuum_cyl_off_axis.py: (150, 1, 300) at resolution 50,
Courant 0.5, z metallic), and one on the +-0 lattice — and a fourth armed mutation
negates the m = 0 on-axis ``Dz`` scalar both curl plans carry.

WHAT THE SUB-STEP GATE CANNOT ANSWER. ``gate_triton_cylindrical_complex.py``
certifies ONE curl call and ONE constitutive call against an in-file reference.
It says nothing about the four of them composing inside a real
``FdtdDriver.step`` — where the sources land between ``step_B`` and
``update_H``, where the axis rules of one sub-step feed the prefix of the next,
and where the state that reaches ``update_E`` is the state THIS FAMILY'S curl
kernel wrote rather than one the harness seeded. This probe measures that, per
COMPLETE driver step, in uint32 words.

THE FOUR SLOTS, and why there are four and not six. The driver's order is
``step_B -> [magnetic sources] -> fill_symmetry_bc_B -> zero_metal_B ->
fill_folded_far_ghosts_B -> update_H -> [withdraw] -> step_D -> [electric
sources] -> fill_symmetry_bc_D -> zero_metal_D -> fill_folded_far_ghosts_D ->
update_E -> update_P`` (driver.py:3205-3228). No case here carries a mirror, a
metallic wall on a folded axis or a polarization, so the fill and ``zero_metal``
calls are the driver's own no-ops and stay on the array path; the CORE FOUR are

    step_B   -> CylindricalComplexCurlPlan     (this family's NEW kernel)
    update_H -> ComplexConstitutivePlan        (the CERTIFIED complex body)
    step_D   -> CylindricalComplexCurlPlan     (this family's NEW kernel)
    update_E -> ComplexConstitutivePlan        (the CERTIFIED complex body)

and the artifact records the launch count of each one against the step budget. A
bit-identical case whose plans never launched measures nothing, so a slot whose
count differs from the steps taken FAILS the case.

CASE DISCIPLINE, the clauses this family's composition can actually bind:

* **uint32 compares, never allclose**, over every stored volume — the twelve
  field arrays and the twelve auxiliaries — after EVERY complete step.
* **Both |m| classes and both z terminations.** |m| = 1 has the axis-row
  increments and no near-axis zeroing; |m| >= 2 has the zeroing and no
  increments; thirteen of the sixteen lifted corpus rows terminate z METALLIC
  and three PERIODIC. One case carries ``accurate_fields_near_cylorigin`` at its
  own Courant bound, where the |m| >= 2 zeroing collapses to row 0 alone.
* **A NON-POWER-OF-TWO Courant in every case.** At 0.5 the ``dt/dx`` scaling is
  exact in binary and an associativity discrepancy vanishes; 0.5 alone certifies
  broken kernels.
* **Sources proven non-vacuous against a source-free control.** Every sourced
  case steps a THIRD driver with the source declarations removed and asserts the
  two separated; a case whose source never moved a word is a case whose source
  seam was never exercised. Sources are ELECTRIC only — all sixteen lifted rows
  are — and placed OFF the axis, because a point source on r = 0 sits on the row
  the |m| = 1 increment overwrites and the |m| >= 2 rule holds at zero.
* **A +-0 LATTICE seeding with its own vacuity floor, per seeding.** A plain
  zero-init is a FIXED POINT of the constitutive sub-step (``fw = value;
  field += kps*fw; field -= kms*prev`` leaves every word +0.0 forever from an
  all-zero state), so two of these cases seed an explicit lattice of -0.0 / +0.0
  / normal words instead. Census 0 is VACUOUS, not passed: the floor is asserted
  at seed AND after the run.
* **The AXIS RULE is asserted to have been the path.** The +-0 cases additionally
  assert the family's own observable: at |m| >= 2 the near-axis rows are exactly
  zero after the run while the first live row is not (so the zeroing ran and did
  not merely zero everything), and at |m| = 1 the axis row of ``Bx``/``Dy``
  carries amplitude (so the axis increment reached a stored word). A case that
  passes with a dead axis rule certifies the rest of the volume only.
* **NaN census every step.** Platform fact (f): a NaN's sign and payload are
  IEEE-unspecified, so a raw uint32 compare over one is not a measurement. Every
  step reports the count and a nonzero count FAILS the case.
* **Armed mutations, caught at step 1**, three of them, each read from the built
  plan so the arming cannot be a no-op on a case whose value happens to match.
* **The nr >= 2 REFUSAL, exercised rather than asserted in prose.** A one-radial-
  row grid is built, the shipped predicate is required to refuse it BY THAT
  CLAUSE, the plan builder is required to return None, and the ARRAY PATH is
  required to raise ``IndexError`` on the same grid at |m| = 1 — which is what
  makes the clause a refusal of an over-coverage rather than a taste.

NO FLUX OR DFT MONITOR is attached: the sub-steps composed here are curl and
constitutive, and a monitor would add a sample-count obligation this probe does
not need. A future case that adds one MUST assert its sample count.

DISPATCH STAYS DISABLED. The plans are installed by monkeypatching the driver
module's sub-step functions FOR ONE ``Fields`` OBJECT, exactly as the certified
composition probes do. ``fastpath.plan_fast_path`` still returns None on every
branch and ``launch.plan_step`` does not know this module exists.

Unbuffered per-step progress; atomic JSON rewrite per case; own provenance
record; run under the stripped IEEE-keep subnormal policy with the in-run strip
counters stamped after the cases run. Correctness only: no throughput or timing
claim is made or admissible from this box.

Usage (the GPU host, one clear device)::

    CUDA_VISIBLE_DEVICES=N \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u probe_triton_cylindrical_complex_composition.py \\
            --probe-artifact results/.../probe.json \\
            --out results/triton_cylindrical_complex_<date>/composition.json

Laptop preflight (NumPy drivers, no launches, no byte claim)::

    python -u probe_triton_cylindrical_complex_composition.py --dry-run \\
        --out /tmp/cylcomplex_composition_preflight.json
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:                                   # the laptop preflight
    cp = None

from meep_gpu.triton_kernels import complex_fields as cxmod          # noqa: E402
from meep_gpu.triton_kernels import cylindrical_complex as cylmod    # noqa: E402
from meep_gpu.triton_kernels import cylindrical_triton as cylreal    # noqa: E402

SEED = 20260813
E_NAMES = ("Ex", "Ey", "Ez")
PRIMARY_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
FIELD_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
               "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
AUX_STATE = ("fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
             "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")

#: The four driver slots this probe serves, in the order the driver calls them.
SLOTS: Tuple[str, ...] = ("step_B", "update_H", "step_D", "update_E")


def log(message: str) -> None:
    print(message, flush=True)


def save(payload: Any, path: str) -> None:
    """Atomic rewrite after every case (the progress-reporting rule)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def write_provenance(results_dir: str) -> str:
    import meep_gpu                                              # noqa: PLC0415

    package = os.path.dirname(os.path.abspath(meep_gpu.__file__))
    kernels = os.path.join(package, "triton_kernels")
    tracked = {
        "meep_gpu/driver.py": os.path.join(package, "driver.py"),
        "meep_gpu/stepping.py": os.path.join(package, "stepping.py"),
        "triton_kernels/complex_fields.py": os.path.join(
            kernels, "complex_fields.py"),
        "triton_kernels/cylindrical_complex.py": os.path.join(
            kernels, "cylindrical_complex.py"),
        "parity/probe_triton_cylindrical_complex_composition.py":
            os.path.abspath(__file__),
    }
    record = {name: _digest(path) for name, path in tracked.items()
              if os.path.exists(path)}
    path = os.path.join(results_dir, "composition_provenance.json")
    save(record, path)
    return path


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------
#
# Every Courant below is non-power-of-two. ``accurate`` is swept at its own
# stability bound: ``Grid`` refuses accurate_fields_near_cylorigin above
# 1/(|m| + 0.5), so the m = 2 accurate case runs at 0.37 < 1/2.5 = 0.4.

CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "m3_z_metallic_cw", "m": 3, "accurate": False,
     "cell": (2.4, 0.0, 3.2), "resolution": 10.0, "courant": 0.37,
     "z_boundary": "metallic", "pml": {"x": {"high": 3}, "z": 3},
     "steps": 10, "seeding": "random", "sources": "cw_ez",
     "why": "|m| >= 2 near-axis zeroing, z METALLIC, the majority corpus shape"},
    {"name": "m_minus1_z_periodic_gaussian", "m": -1, "accurate": False,
     "cell": (2.4, 0.0, 3.2), "resolution": 10.0, "courant": 0.31,
     "z_boundary": "periodic", "pml": {"x": {"high": 3}},
     "steps": 10, "seeding": "random", "sources": "gaussian_ez",
     "why": "|m| = 1 axis-row increments, z PERIODIC (ring-cyl's termination), "
            "and the NEGATIVE m of seven of the sixteen rows"},
    {"name": "m2_accurate_z_metallic", "m": 2, "accurate": True,
     "cell": (2.0, 0.0, 2.8), "resolution": 10.0, "courant": 0.37,
     "z_boundary": "metallic", "pml": {"x": {"high": 3}, "z": 3},
     "steps": 8, "seeding": "random", "sources": "cw_er",
     "why": "accurate_fields_near_cylorigin: the |m| >= 2 zeroing collapses to "
            "row 0 alone, at that branch's own Courant bound"},
    {"name": "m5_z_metallic_cw", "m": 5, "accurate": False,
     "cell": (2.8, 0.0, 2.4), "resolution": 10.0, "courant": 0.2777777777777778,
     "z_boundary": "metallic", "pml": {"x": {"high": 4}, "z": 3},
     "steps": 8, "seeding": "random", "sources": "cw_ez",
     "why": "the largest |m| in the lift (perturbation_theory): five zeroed "
            "rows, so ZERO_ROWS is not a one-or-two constexpr in disguise"},
    {"name": "m1_signed_zero_lattice", "m": 1, "accurate": False,
     "cell": (2.4, 0.0, 3.2), "resolution": 10.0, "courant": 0.37,
     "z_boundary": "metallic", "pml": {"x": {"high": 2}, "z": 2},
     "steps": 6, "seeding": "signed_zero_lattice", "sources": "none",
     "why": "the +-0 LATTICE at |m| = 1, where the axis increment REPLACES row "
            "0 and +0.0 + x == x for every x except -0.0"},
    {"name": "m3_signed_zero_lattice_periodic", "m": 3, "accurate": False,
     "cell": (2.4, 0.0, 3.2), "resolution": 10.0, "courant": 0.31,
     "z_boundary": "periodic", "pml": {"x": {"high": 2}},
     "steps": 6, "seeding": "signed_zero_lattice", "sources": "none",
     "why": "the +-0 LATTICE at |m| >= 2 under the OTHER z termination: the "
            "near-axis zeroing writing over a signed-zero background"},
    # ---- m = 0 under COMPLEX storage, 2026-09-04 (the M_ZERO arm) -------------
    {"name": "m0_z_metallic_gaussian", "m": 0, "accurate": False,
     "cell": (2.4, 0.0, 3.2), "resolution": 10.0, "courant": 0.37,
     "z_boundary": "metallic", "pml": {"x": {"high": 3}, "z": 3},
     "steps": 10, "seeding": "random", "sources": "gaussian_ez",
     "why": "m = 0 with complex64 storage: no i*m/r block, no increment, the "
            "m = 0 axis pair (Bx[0] = 0; Dz[0] += 4*Courant*Hp[0], Dy[0] = 0) on "
            "complex word pairs, at a non-power-of-two Courant"},
    {"name": "m0_corpus_dipole_off_axis", "m": 0, "accurate": False,
     "cell": (3.0, 0.0, 6.0), "resolution": 50.0, "courant": 0.5,
     "z_boundary": "metallic", "pml": {"x": {"high": 50}, "z": 50},
     "steps": 6, "seeding": "random", "sources": "gaussian_ez",
     # Courant 0.5 IS the corpus row's own, and is exact in binary; every other
     # case here carries a non-power-of-two one, which the laptop test pins.
     "courant_is_the_corpus_rows_own": True,
     "why": "the corpus row examples:dipole_in_vacuum_cyl_off_axis.py at its OWN "
            "shape (150, 1, 300), resolution, Courant, m, storage and z "
            "termination; the lift record carries no absorber depth, so 50 cells "
            "(1.0 length unit at resolution 50) is the fixture's choice"},
    {"name": "m0_signed_zero_lattice", "m": 0, "accurate": False,
     "cell": (2.4, 0.0, 3.2), "resolution": 10.0, "courant": 0.37,
     "z_boundary": "metallic", "pml": {"x": {"high": 2}, "z": 2},
     "steps": 6, "seeding": "signed_zero_lattice", "sources": "none",
     "why": "the +-0 LATTICE at m = 0: the axis pair writes complex zero over "
            "Bx/Dy row 0 and the Dz add lands on a signed-zero background"},
)

# THE ABSORBER ON A +-0 CASE IS THIN, AND THE THICKNESS IS THE MEASUREMENT.
# The only in-run producer of a stored -0.0 here is a NEGATIVE split-field
# coefficient multiplying a quiet +0.0, and whether one exists is a property of
# the absorber's profile, not of the seeding. Measured on
# ``m3_signed_zero_lattice_periodic`` (NumPy, 2026-08-13), r-high absorber only:
#
#     2 cells -> min kms_x_h = -0.1293, census trail [64, 0, 32, 0, 32, 0]
#     3 cells -> min kms_x_h = +0.0706, census trail [0, 0, 0, 0, 0, 0]  VACUOUS
#     4 cells -> min kms_x_h = +0.2315, census trail [0, 0, 0, 0, 0, 0]  VACUOUS
#
# The 3-cell arrangement is what this case was first written with, and it passed
# every byte compare while measuring nothing about the +-0 class. The floor
# below is what caught it.

#: Sources are ELECTRIC only (every one of the sixteen lifted rows is) and OFF
#: THE AXIS. A point source at r = 0 lands on the row the |m| = 1 increment
#: overwrites outright and the |m| >= 2 rule holds at zero, so an on-axis
#: declaration would deposit a current the very next sub-step discards — a
#: source that cannot separate its own control.
SOURCE_CENTER = (0.7, 0.0, 0.25)


def source_declarations(kind: str) -> List[Dict[str, Any]]:
    if kind == "none":
        return []
    if kind == "cw_ez":
        return [{"component": "Ez", "frequency": 0.75, "center": SOURCE_CENTER,
                 "size": (0.0, 0.0, 0.0), "amplitude": 0.65}]
    if kind == "cw_er":
        # Er (stored as Ex) exercises the OTHER i*m/r call site's partner class.
        return [{"component": "Ex", "frequency": 0.8, "center": SOURCE_CENTER,
                 "size": (0.0, 0.0, 0.0), "amplitude": 0.55}]
    if kind == "gaussian_ez":
        return [{"component": "Ez", "source_type": "gaussian",
                 "frequency": 0.8, "fwidth": 0.4, "center": SOURCE_CENTER,
                 "size": (0.0, 0.0, 0.0), "amplitude": 1.0}]
    raise ValueError(f"unknown source kind {kind!r}")


def build_driver(case: Dict[str, Any], xp, prefer_gpu: bool,
                 with_source: bool = True):
    from meep_gpu.driver import FdtdDriver                     # noqa: PLC0415

    driver = FdtdDriver(
        cell_size=case["cell"], resolution=case["resolution"],
        cylindrical=True, m=case["m"],
        accurate_fields_near_cylorigin=case["accurate"],
        force_complex_fields=True, courant=case["courant"],
        boundaries={"z": case["z_boundary"]},
        prefer_gpu=prefer_gpu, gpu_id=0)
    # r carries an absorber on the HIGH face only: the low face is the cylinder
    # axis, where a layer would eat the axis rows this family exists for.
    driver.setup_pml(case["pml"])
    if with_source:
        for declaration in source_declarations(case["sources"]):
            driver.add_source(dict(declaration))
    return driver


# ---------------------------------------------------------------------------
# Seeding, and the +-0 lattice's own vacuity floor
# ---------------------------------------------------------------------------

def _row(index: int) -> Tuple[Any, ...]:
    return (index,)


def seed_state(driver, xp, case: Dict[str, Any]) -> Dict[str, Any]:
    """Seed the primaries and return the seeding's own record."""
    shape = tuple(int(n) for n in driver.shape)
    kind = case["seeding"]
    if kind == "random":
        rng = np.random.default_rng(SEED + 4000 + len(case["name"]))
        for name in PRIMARY_NAMES:
            host = (0.25 * rng.uniform(-1.0, 1.0, size=shape)
                    + 0.25j * rng.uniform(-1.0, 1.0, size=shape))
            driver.set_field(name, xp.asarray(
                np.ascontiguousarray(host.astype(np.complex64))))
        return {"kind": kind}
    if kind != "signed_zero_lattice":
        raise ValueError(f"unknown seeding {kind!r}")

    # THE +-0 LATTICE, and it exists because a plain zero-init is a FIXED POINT
    # of the constitutive sub-step here (see the module docstring). The sign of
    # the zero alternates by index parity so the background carries negative
    # zeros for a sign-bit defect to show against; the FIRST LIVE RADIAL ROW —
    # row |m| at |m| >= 2, row 1 at |m| = 1 — carries real amplitude, so the
    # axis rule and the i*m/r coupling are the only paths from there to the
    # axis rows this case watches.
    live = max(1, abs(int(case["m"])))
    index = np.indices(shape).sum(axis=0)
    zeros = np.where((index % 2) == 0, np.float32(0.0), np.float32(-0.0))
    rng = np.random.default_rng(SEED + 4100)
    plane = tuple(n for i, n in enumerate(shape) if i != 0)
    for name in PRIMARY_NAMES:
        host = np.empty(shape, dtype=np.complex64)
        host.real = zeros
        host.imag = np.where((index % 2) == 0, np.float32(-0.0), np.float32(0.0))
        amplitude = rng.uniform(0.2, 0.6, plane).astype(np.float32)
        host.real[_row(live)] = amplitude
        host.imag[_row(live)] = -amplitude
        driver.set_field(name, xp.asarray(np.ascontiguousarray(host)))
    return {"kind": kind, "amplitude_row": int(live),
            "note": "row |m| (or row 1 at |m| = 1) is the first live radial row"}


def to_host(xp, array) -> np.ndarray:
    return np.asarray(array) if xp is np else np.asarray(xp.asnumpy(array))


def negative_zero_census(xp, driver, names=FIELD_STATE + AUX_STATE) -> int:
    """Stored words that are NEGATIVE ZERO — not merely negative.

    Counted on the uint32 word, over both planes of the complex storage:
    ``(x == 0) & signbit(x)`` on the viewed float32 pairs. Never a sign-bit
    tally, which on a continuous seed counts about half the stored words and
    would relabel an ordinary negative count as this class's witness. Census 0
    means the case cannot see a defect that lives in the sign of a zero —
    VACUOUS, not passed.

    OVER THE WHOLE STATE, and PER STEP rather than at the end — both measured
    on this family (NumPy, ``m1_signed_zero_lattice``, 2026-08-13):

    * primaries alone report 37 words after step 1 where the whole state reports
      177: the class lives mostly in the ``fu_`` auxiliaries and ``f_w_``, which
      is exactly where the recurrence reads it;
    * the class is TRANSIENT WITH STEP PARITY — 177, 0, 99, 0, 95, 0 over six
      complete steps — because a ``-0.0`` that survives one sub-step is
      laundered back by the next ``fu *= kms`` sign flip. An END-OF-RUN census
      therefore reports 0 on a case where the class was live at every odd step,
      and the vacuity floor is the PEAK over the trail, not the final value.
    """
    total = 0
    for name in names:
        array = getattr(driver.fields, name, None)
        if array is None:
            continue
        host = np.ascontiguousarray(to_host(xp, array))
        parts = host.view(np.float32) if host.dtype.kind == "c" else host
        total += int(np.count_nonzero((parts == 0) & np.signbit(parts)))
    return total


def nan_census(xp, driver) -> Dict[str, int]:
    """Non-finite stored words, by class. Platform fact (f): a NaN's sign and
    payload are IEEE-unspecified, so a raw uint32 compare over one is not a
    measurement — every case reports this, and a nonzero count fails."""
    nans = 0
    infinities = 0
    for name in FIELD_STATE + AUX_STATE:
        array = getattr(driver.fields, name, None)
        if array is None:
            continue
        host = to_host(xp, array)
        nans += int(np.count_nonzero(np.isnan(host)))
        infinities += int(np.count_nonzero(np.isinf(host)))
    return {"nan_words": nans, "inf_words": infinities}


def axis_rule_witness(xp, driver, case: Dict[str, Any]) -> Dict[str, Any]:
    """Did the family's own axis rule reach a stored word?

    The +-0 cases assert this. At |m| >= 2 the near-axis rows must be EXACTLY
    zero after the run while the first live row must not be (the zeroing ran,
    and it did not merely zero the volume); at |m| = 1 the axis row of the
    components the increment writes must carry amplitude (the increment reached
    a stored word). A case that passes with a dead axis rule has certified the
    rest of the volume and nothing this family adds.
    """
    m = int(case["m"])
    rows = cylmod.zero_rows(m, bool(case["accurate"]))
    record: Dict[str, Any] = {"m": m, "m_class": cylmod.m_class(m),
                              "zero_rows": int(rows)}
    if m == 0:
        # m = 0 (2026-09-04): Bx and Dy row 0 must be EXACTLY zero after the run
        # (the axis pair ran through the composition) and Dz row 0 must not be
        # (the on-axis add reached a stored word rather than the row being dead).
        record["max_abs_axis_row_Bx"] = float(np.max(np.abs(
            to_host(xp, driver.fields.Bx)[0])))
        record["max_abs_axis_row_Dy"] = float(np.max(np.abs(
            to_host(xp, driver.fields.Dy)[0])))
        record["max_abs_axis_row_Dz"] = float(np.max(np.abs(
            to_host(xp, driver.fields.Dz)[0])))
        record["witness"] = "m = 0 axis pair"
    elif rows:
        held = 0.0
        for name in FIELD_STATE:
            array = getattr(driver.fields, name, None)
            if array is None:
                continue
            host = to_host(xp, array)
            held = max(held, float(np.max(np.abs(host[:rows]))))
        first_live = max(
            float(np.max(np.abs(to_host(xp, getattr(driver.fields, name))[rows])))
            for name in FIELD_STATE
            if getattr(driver.fields, name, None) is not None)
        record["max_abs_on_zeroed_rows"] = held
        record["max_abs_on_first_live_row"] = first_live
        record["witness"] = "near-axis zeroing"
    else:
        # |m| = 1: the B-side increment writes Bx row 0, the D-side writes Dy.
        record["max_abs_axis_row_Bx"] = float(np.max(np.abs(
            to_host(xp, driver.fields.Bx)[0])))
        record["max_abs_axis_row_Dy"] = float(np.max(np.abs(
            to_host(xp, driver.fields.Dy)[0])))
        record["witness"] = "|m| = 1 axis-row increment"
    return record


def assert_axis_rule_live(record: Dict[str, Any], name: str) -> None:
    if record["witness"] == "m = 0 axis pair":
        if record["max_abs_axis_row_Bx"] != 0.0 or record["max_abs_axis_row_Dy"] != 0.0:
            raise AssertionError(
                f"{name}: Bx/Dy row 0 is not zero after the run "
                f"({record['max_abs_axis_row_Bx']}, {record['max_abs_axis_row_Dy']}); "
                f"the m = 0 axis pair did not run through the composition")
        if not record["max_abs_axis_row_Dz"]:
            raise AssertionError(
                f"{name}: VACUOUS — Dz row 0 is zero too, so the m = 0 on-axis add "
                f"never reached a stored word")
        return
    if record["witness"] == "near-axis zeroing":
        if record["max_abs_on_zeroed_rows"] != 0.0:
            raise AssertionError(
                f"{name}: the near-axis rows are not zero after the run "
                f"({record['max_abs_on_zeroed_rows']}); the |m| >= 2 rule did "
                f"not run through the composition")
        if not record["max_abs_on_first_live_row"]:
            raise AssertionError(
                f"{name}: VACUOUS — the first LIVE radial row is zero too, so "
                f"'the near-axis rows are zero' says nothing about the rule")
    elif not (record["max_abs_axis_row_Bx"] or record["max_abs_axis_row_Dy"]):
        raise AssertionError(
            f"{name}: VACUOUS — the |m| = 1 axis increment left row 0 of both "
            f"Bx and Dy at zero, so the increment never reached a stored word")


# ---------------------------------------------------------------------------
# The seams — asserted by name before any step is taken
# ---------------------------------------------------------------------------

def seam_report(fields, pml, probe, drop_backend: bool) -> Dict[str, Any]:
    def strip(reasons: Sequence[str]) -> List[str]:
        out = list(reasons)
        if drop_backend:
            out = [reason for reason in out if "array module" not in reason]
        return out

    report: Dict[str, Any] = {}
    for sub_step in ("step_B", "step_D"):
        reasons = strip(cylmod.cylindrical_complex_curl_coverage(
            fields, pml, sub_step, probe=probe).reasons)
        report[f"curl_{sub_step}"] = {"covered": not reasons, "reasons": reasons}
    for side in ("H", "E"):
        reasons = strip(cylmod.cylindrical_complex_constitutive_coverage(
            fields, pml, side, probe=probe).reasons)
        report[f"constitutive_{side}"] = {"covered": not reasons,
                                          "reasons": reasons}
    # THE TWO DISJOINTNESS SEAMS. Neither sibling predicate may admit this run:
    # cylindrical_triton refuses m != 0 BY NAME, and the certified complex
    # family refuses every cylindrical grid. If either narrowed underfoot, two
    # predicates could admit one configuration and the install route below would
    # be choosing between them silently.
    real_curl = strip(cylreal.cylindrical_curl_coverage(fields, pml).reasons)
    complex_curl = strip(cxmod.complex_pml_curl_coverage(
        fields, pml, "step_B", probe=probe).reasons)
    report["sibling_real_cylindrical"] = {
        "m_clause": [r for r in real_curl if "m " in r or "m=" in r or "m!" in r
                     or "azimuthal" in r],
        "all": real_curl}
    report["sibling_certified_complex"] = {
        "cylindrical_clause": [r for r in complex_curl if "cylind" in r.lower()],
        "all": complex_curl}
    return report


def assert_seams(report: Dict[str, Any], case_name: str) -> None:
    for slot in ("curl_step_B", "curl_step_D", "constitutive_H",
                 "constitutive_E"):
        if not report[slot]["covered"]:
            raise AssertionError(
                f"{case_name}: the {slot} predicate refused a run this probe "
                f"is built to compose: {report[slot]['reasons']}")
    if not report["sibling_real_cylindrical"]["all"]:
        raise AssertionError(
            f"{case_name}: the REAL-storage cylindrical predicate no longer "
            f"refuses this complex run; the disjointness seam (its "
            f"force_complex_fields clause, cylindrical_triton.py:555-556) was "
            f"narrowed underfoot")
    if not report["sibling_certified_complex"]["cylindrical_clause"]:
        raise AssertionError(
            f"{case_name}: the CERTIFIED complex predicate no longer refuses "
            f"the cylindrical grid; the other disjointness seam "
            f"(complex_fields._complex_grid_reasons clause 4) was narrowed")


# ---------------------------------------------------------------------------
# Plans, launch counting and installation
# ---------------------------------------------------------------------------

class PlanCounter:
    """Launch counting for EVERY installed slot. A slot whose count differs
    from the step budget fails the case: a bit-identical row whose plans never
    ran measures nothing."""

    def __init__(self, plan: Any) -> None:
        self.plan = plan
        self.launches = 0

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.launches += 1
        self.plan.run(*args, **kwargs)


def build_plans(driver, probe) -> Dict[str, Any]:
    """All four sub-step plans from the ENGINE's own objects.

    A builder that returns None here is a regression, not an expected refusal:
    :func:`assert_seams` has already measured that every one of these predicates
    admits this run.
    """
    fields, pml = driver.fields, driver.pml
    plans = {
        "step_B": cylmod.plan_cylindrical_complex_curl(fields, pml, "step_B",
                                                       probe=probe),
        "update_H": cylmod.plan_cylindrical_complex_constitutive(fields, pml, "H",
                                                                 probe=probe),
        "step_D": cylmod.plan_cylindrical_complex_curl(fields, pml, "step_D",
                                                       probe=probe),
        "update_E": cylmod.plan_cylindrical_complex_constitutive(fields, pml, "E",
                                                                 probe=probe),
    }
    for slot, plan in plans.items():
        if plan is None:
            raise AssertionError(
                f"the {slot} engine builder refused a run its predicate was "
                f"asserted to admit")
    return plans


def install(driver_module, plans: Dict[str, Any], owner):
    """Serve the four sub-steps from the plans, for the OWNER fields only.

    Sources, the fill calls and the guards stay the driver's own; production
    dispatch is untouched (this is a monkeypatch on the driver module, and
    ``plan_fast_path`` still returns None everywhere).
    """
    names = ("step_B", "update_H", "step_D", "update_E")
    originals = {name: getattr(driver_module, name) for name in names}

    def wrapper_for(name):
        def wrapper(fields, pml=None):
            entry = plans.get(name) if fields is owner else None
            if entry is None:
                return originals[name](fields, pml)
            entry.run()
            return None
        return wrapper

    for name in names:
        setattr(driver_module, name, wrapper_for(name))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


# ---------------------------------------------------------------------------
# The armed mutations
# ---------------------------------------------------------------------------
#
# Each one is read FROM THE BUILT PLAN and negated/bent relative to what is
# actually there, so it can never be a no-op on a case whose value happens to
# match a hard-coded mutant. All three must be caught at STEP 1: the i*m/r
# coupling, the near-axis zeroing and the axis increment are all live at the
# FIRST step_B, so a later catch means the needle is not the one declared.

def arm_imr_row_negated(driver, plans: Dict[str, Any], case) -> str:
    """Negate the i*m/r coefficient rows the CURL plans point at.

    The plan holds the device pointer, so the kernel reads the negated bytes
    while the array-path reference recomputes the true row — an in-place edit
    AFTER the plans are built, which makes it a mutation of the kernel's input
    and not of the physics both sides see. ``* -1.0`` per word, never unary
    minus (platform fact (b)).
    """
    touched = 0
    for slot in ("step_B", "step_D"):
        plan = plans[slot].plan if isinstance(plans[slot], PlanCounter) \
            else plans[slot]
        for pointer in plan._imr_rows:                       # noqa: SLF001
            pointer.array[...] = pointer.array * -1.0
            touched += 1
    if not touched:
        raise AssertionError("no i*m/r coefficient row to negate")
    return f"{touched} i*m/r coefficient rows negated in place"


def arm_zero_rows_reduced(driver, plans: Dict[str, Any], case) -> str:
    """Rebuild both curl plans with ZERO_ROWS one lower than the shipped value.

    The true count is READ FROM THE PLAN first, so on a case where it is already
    1 the mutant is 0 and the near-axis rule vanishes entirely; on a case where
    it is 0 (|m| = 1) this mutation does not apply and the leg says so rather
    than reporting a null it never armed.
    """
    plan = plans["step_B"].plan if isinstance(plans["step_B"], PlanCounter) \
        else plans["step_B"]
    if not plan.zero_rows:
        raise AssertionError(
            "this case has no near-axis zeroing to reduce (|m| = 1); the "
            "mutation table must not schedule it here")
    before = int(plan.zero_rows)
    for slot in ("step_B", "step_D"):
        entry = plans[slot]
        target = entry.plan if isinstance(entry, PlanCounter) else entry
        target.zero_rows = before - 1
    return f"ZERO_ROWS {before} -> {before - 1} on both curl plans"


def arm_axis_increment_negated(driver, plans: Dict[str, Any], case) -> str:
    """Negate the |m| = 1 axis-increment scalars the curl plans carry.

    Read from the plan and negated, so the arming is relative. The scalars are
    ``(-dtdx, (re, im))``; all three words are flipped, which moves both the B
    replacement row and the D one.
    """
    plan = plans["step_B"].plan if isinstance(plans["step_B"], PlanCounter) \
        else plans["step_B"]
    if plan.m_class != cylmod.m_class(1):
        raise AssertionError(
            "this case is not |m| = 1; the mutation table must not schedule "
            "the axis-increment needle here")
    changed = []
    for slot in ("step_B", "step_D"):
        entry = plans[slot]
        target = entry.plan if isinstance(entry, PlanCounter) else entry
        minus_dtdx, inc = target.increment_scalars
        target.increment_scalars = (minus_dtdx * -1.0,
                                    (inc[0] * -1.0, inc[1] * -1.0))
        changed.append((slot, minus_dtdx, list(inc)))
    return f"axis increment scalars negated on {changed}"


def arm_m0_axis_add_scalar_negated(driver, plans: Dict[str, Any], case) -> str:
    """Negate the m = 0 on-axis ``Dz`` scalar (``4*Courant``) both curl plans carry.

    Read from the plan and negated, so the arming is relative. Only the D-side
    plan reads it (the B side's M_ZERO arm has no scalar), so the B plan's copy
    is flipped for symmetry and measures nothing; the catch is on ``Dz`` row 0
    at the FIRST step_D, whose ``Hp`` operand is the ``update_H`` of the seeded
    ``By`` and is therefore live from step 1.
    """
    plan = plans["step_D"].plan if isinstance(plans["step_D"], PlanCounter) \
        else plans["step_D"]
    if plan.m_class != cylmod.m_class(0):
        raise AssertionError(
            "this case is not m = 0; the mutation table must not schedule the "
            "m = 0 axis-scalar needle here")
    changed = []
    for slot in ("step_B", "step_D"):
        entry = plans[slot]
        target = entry.plan if isinstance(entry, PlanCounter) else entry
        before = float(target.four_dtdx)
        target.four_dtdx = before * -1.0
        changed.append((slot, before, target.four_dtdx))
    return f"m = 0 on-axis Dz scalar negated on {changed}"


#: ``(name, case_name, callable, expected_step, why)``. Scheduled per case,
#: because two of the three only exist in one |m| class, and each declares the
#: COMPLETE DRIVER STEP at which it must be caught — asserted exactly, not as an
#: upper bound. "Caught eventually" is the weaker claim, and a needle caught two
#: steps late is usually a different needle.
MUTATIONS: Tuple[Tuple[str, str, Callable[..., str], int, str], ...] = (
    ("imr_row_negated", "m3_z_metallic_cw", arm_imr_row_negated, 1,
     "the i*m/r coupling, this family's addition that touches every step: the "
     "bound coefficient row negated where only the kernel reads it. Live at "
     "the FIRST step_B, whose partners are the seeded primaries' own volumes"),
    ("zero_rows_reduced", "m3_z_metallic_cw", arm_zero_rows_reduced, 1,
     "the |m| >= 2 near-axis rule, off by one row. Live at the first step_B: "
     "the seeded state carries amplitude on the rows the rule zeroes"),
    ("axis_increment_negated", "m_minus1_z_periodic_gaussian",
     arm_axis_increment_negated, 2,
     "the |m| = 1 axis-row increment, negated — the other m class's whole "
     "addition, and the one the array path applies BEFORE the recurrence. "
     "STEP 2 IS STRUCTURAL, not a weaker claim: these two scalars are the "
     "B-SIDE pair (cylindrical_complex.py:715 — 'unused on D'), the B-side "
     "increment reads Ep and Ez, and the driver runs update_E at the END of a "
     "step, so E is exactly +0.0 at every run's FIRST step_B whatever the "
     "seeding. Measured: caught at step 2 with the needle armed from step 1"),
    ("m0_axis_add_scalar_negated", "m0_z_metallic_gaussian",
     arm_m0_axis_add_scalar_negated, 1,
     "the m = 0 arm's own addition (2026-09-04): the on-axis Dz post-add "
     "4*Courant*Hp[0], its host-rounded scalar negated where only the kernel "
     "reads it. Live at the FIRST step_D: Hp there is update_H of the seeded By"),
)


# ---------------------------------------------------------------------------
# One composition case
# ---------------------------------------------------------------------------

def compare(xp, left, right) -> Dict[str, Any]:
    a = np.ascontiguousarray(to_host(xp, left)).view(np.uint32).ravel()
    b = np.ascontiguousarray(to_host(xp, right)).view(np.uint32).ravel()
    return {"bit_identical": bool(np.array_equal(a, b)),
            "differing_words": int(np.count_nonzero(a != b)),
            "total_words": int(a.size)}


def driver_state(driver) -> Dict[str, Any]:
    return {name: getattr(driver.fields, name)
            for name in FIELD_STATE + AUX_STATE
            if getattr(driver.fields, name, None) is not None}


def run_case(xp, case: Dict[str, Any], probe,
             mutation: Optional[Tuple[str, str, Callable[..., str], int, str]] = None
             ) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module                # noqa: PLC0415

    name = case["name"] + ("" if mutation is None else f"::{mutation[0]}")
    steps = case["steps"] if mutation is None else mutation[3] + 1
    reference = build_driver(case, xp, prefer_gpu=True)
    candidate = build_driver(case, xp, prefer_gpu=True)
    control = (build_driver(case, xp, prefer_gpu=True, with_source=False)
               if case["sources"] != "none" else None)
    undo: Callable[[], Any] = lambda: None                      # noqa: E731
    scaled = round(case["courant"] * 2 ** 8)
    row: Dict[str, Any] = {
        "case": name, "base_case": case["name"], "why": case["why"],
        "steps": steps, "mutation": None if mutation is None else mutation[0],
        "mutation_why": None if mutation is None else mutation[4],
        "mutation_expected_step": None if mutation is None else mutation[3],
        "seeding": case["seeding"], "sources": case["sources"],
        "m": case["m"], "accurate_fields_near_cylorigin": case["accurate"],
        "z_boundary": case["z_boundary"], "courant": case["courant"],
        "courant_is_power_of_two": bool(
            abs(case["courant"] * 2 ** 8 - scaled) < 1e-12
            and scaled and (scaled & (scaled - 1)) == 0),
        "per_step": [],
    }
    try:
        row["shape"] = [int(n) for n in candidate.shape]
        row["dtype"] = str(candidate.fields.Bx.dtype)
        for target in (reference, candidate) + ((control,) if control else ()):
            row["seeding_record"] = seed_state(target, xp, case)
        row["seams"] = seam_report(candidate.fields, candidate.pml, probe,
                                   drop_backend=False)
        assert_seams(row["seams"], name)

        # THE PER-SEEDING VACUITY FLOOR, at seed time.
        row["negative_zero_census_at_seed"] = negative_zero_census(xp, reference)
        if case["seeding"] == "signed_zero_lattice" and \
                row["negative_zero_census_at_seed"] == 0:
            raise AssertionError(
                f"{name}: VACUOUS seeding — the +-0 lattice produced no stored "
                f"negative zero, so this case cannot see a sign-bit defect "
                f"(census 0 is vacuous, not passed)")

        plans = build_plans(candidate, probe)
        row["plan_types"] = {slot: type(plan).__name__
                             for slot, plan in plans.items()}
        curl_b = plans["step_B"]
        row["plan_constexprs"] = {
            "m_class": int(curl_b.m_class), "zero_rows": int(curl_b.zero_rows),
            "bcz": int(curl_b.bcz), "expansion": int(curl_b.expansion),
            "block": int(curl_b.block)}
        counters = {slot: PlanCounter(plan) for slot, plan in plans.items()}
        if mutation is not None:
            row["mutation_detail"] = mutation[2](candidate, counters, case)
        undo = install(driver_module, counters, candidate.fields)

        source_effect_seen = control is None
        started = time.time()
        first_divergence = None
        for step in range(1, steps + 1):
            reference.step()
            candidate.step()
            if control is not None:
                control.step()
            if xp is not np:
                xp.cuda.runtime.deviceSynchronize()
            left = driver_state(reference)
            right = driver_state(candidate)
            if set(left) != set(right):
                raise AssertionError(
                    f"{name} step {step}: state inventory differs: "
                    f"{sorted(left)} vs {sorted(right)}")
            parts = {key: compare(xp, right[key], left[key])
                     for key in sorted(left)}
            if control is not None:
                source_effect_seen = source_effect_seen or any(
                    not compare(xp, left[key],
                                getattr(control.fields, key))["bit_identical"]
                    for key in PRIMARY_NAMES)
            census = nan_census(xp, reference)
            signed_zero = negative_zero_census(xp, reference)
            point = {
                "step": step,
                "negative_zero_census": signed_zero,
                "bit_identical": all(part["bit_identical"]
                                     for part in parts.values()),
                "differing_words": sum(part["differing_words"]
                                       for part in parts.values()),
                "total_words": sum(part["total_words"]
                                   for part in parts.values()),
                "differing_arrays": sorted(
                    key for key, part in parts.items()
                    if not part["bit_identical"]),
                "source_effect_seen": source_effect_seen,
                "nan_census": census,
            }
            row["per_step"].append(point)
            log(f"[composition] {name} step {step}/{steps} identical="
                f"{point['bit_identical']} ndiff={point['differing_words']} "
                f"nan={census['nan_words']} neg0={signed_zero} "
                f"({time.time() - started:.1f} s)")
            if census["nan_words"] or census["inf_words"]:
                raise AssertionError(
                    f"{name} step {step}: the reference state carries "
                    f"{census} non-finite words; platform fact (f) forbids "
                    f"reading raw words through a NaN")
            if not point["bit_identical"] and first_divergence is None:
                first_divergence = point
                # BYTE DIVERGENCE: first case/step/array, STOP, report.
                break

        row["launches"] = {slot: counter.launches
                           for slot, counter in counters.items()}
        row["first_divergence"] = first_divergence
        row["bit_identical"] = first_divergence is None
        row["source_effect_seen"] = source_effect_seen

        row["axis_rule"] = axis_rule_witness(xp, reference, case)
        row["negative_zero_census_trail"] = [point["negative_zero_census"]
                                             for point in row["per_step"]]
        row["negative_zero_census_peak"] = max(
            row["negative_zero_census_trail"], default=0)
        row["negative_zero_census_after"] = (
            row["negative_zero_census_trail"][-1]
            if row["negative_zero_census_trail"] else 0)
        # THE PER-SEEDING VACUITY FLOOR, on the PEAK of the trail. The final
        # value is recorded beside it and is NOT the floor: the class alternates
        # with step parity here (see :func:`negative_zero_census`), so an
        # end-of-run test would report VACUOUS on a live case.
        if case["seeding"] == "signed_zero_lattice":
            if row["negative_zero_census_peak"] == 0:
                raise AssertionError(
                    f"{name}: VACUOUS — the +-0 lattice never reached the "
                    f"state the sub-steps read at ANY step (peak census 0)")
            assert_axis_rule_live(row["axis_rule"], name)

        expected = steps if first_divergence is None else first_divergence["step"]
        row["expected_launches_per_slot"] = expected
        row["launch_counts_match_budget"] = all(
            count == expected for count in row["launches"].values())
        if not row["launch_counts_match_budget"]:
            raise AssertionError(
                f"{name}: a slot's launch count does not match the steps taken "
                f"({row['launches']} vs {expected}): a bit-identical row "
                f"without launch evidence certifies nothing")
        if mutation is None and not source_effect_seen:
            raise AssertionError(
                f"{name}: the source never separated the reference from its "
                f"source-free control")
        if mutation is not None:
            row["caught"] = first_divergence is not None
            row["caught_at_step"] = (None if first_divergence is None
                                     else first_divergence["step"])
            if not row["caught"]:
                raise AssertionError(
                    f"{name}: ARMED MUTATION NOT CAUGHT — the composition "
                    f"reproduced the array path with the defect installed")
            if first_divergence["step"] != mutation[3]:
                raise AssertionError(
                    f"{name}: the armed mutation was caught at step "
                    f"{first_divergence['step']}, where this leg DECLARES step "
                    f"{mutation[3]}. The step is asserted exactly, not as an "
                    f"upper bound: a needle caught late is usually a different "
                    f"needle. The declaration's reason: {mutation[4]}")
        elif not row["bit_identical"]:
            raise AssertionError(
                f"{name}: BYTE DIVERGENCE at step {first_divergence['step']}, "
                f"arrays {first_divergence['differing_arrays']}")
        return row
    finally:
        undo()
        for target in (reference, candidate, control):
            if target is not None:
                try:
                    target.close()
                except Exception:                   # noqa: BLE001
                    pass
        if xp is not np:
            xp.get_default_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# The nr >= 2 REFUSAL, exercised
# ---------------------------------------------------------------------------
#
# ``cylindrical_complex._cylindrical_geometry_reasons`` refuses a grid with
# fewer than MINIMUM_RADIAL_ROWS radial cells. The clause is an OVER-COVERAGE
# refusal, not a taste: the |m| = 1 axis increment reads the FIRST OFF-AXIS row
# through ``xp.take(Ez, 1, axis=0)`` (stepping.py:671), and the kernel's
# matching masked load would read a full plane past the end of the source
# volume and store a silently wrong axis row.
#
# THE ARRAY PATH'S BEHAVIOUR IS NOT THE SAME ON BOTH BACKENDS, and that is a
# MEASURED FINDING this leg exists to pin rather than a detail:
#
#   numpy  -> IndexError out of take(); the run stops loudly
#   cupy   -> NO BOUNDS CHECK. cp.take(a, 1, axis=0) on a size-1 axis returns
#             row 0's data, so the SAME driver on the SAME grid steps to
#             completion with a finite state
#
# (the GPU host, CuPy 13.5.1, RTX A6000, 2026-08-13). The two backends therefore do
# not agree about whether the configuration is steppable at all, which is a
# defect of the ARRAY PATH — ``stepping.py`` is not this family's file and the
# finding is ATTRIBUTED, not fixed here — and it changes what the clause is for:
# on the GPU backend the refusal is the only thing between a run and a silently
# wrong axis row. The leg asserts the predicate's refusal BY THAT CLAUSE and
# that the plan builder returns None on both backends, and it asserts the array
# path's behaviour against a DECLARED per-backend expectation, so a backend that
# starts or stops raising fails here instead of quietly changing the story.

#: What the ARRAY PATH does on a one-radial-row |m| = 1 grid, per backend.
#: Measured, declared, and asserted — never inferred from whichever host runs.
ARRAY_PATH_ON_NR1: Dict[str, str] = {
    "numpy": "raises IndexError",
    "cupy": "steps without raising (cupy.take does not bounds-check)",
}

#: ONE radial cell, and a z axis tall enough to carry an absorber with an
#: interior left over — the predicate's PML clauses must not be what refuses,
#: or the leg would measure the wrong clause.
NR1_CELL = (1.0, 0.0, 8.0)
NR1_RESOLUTION = 1.0


def run_refusal(xp, probe) -> Dict[str, Any]:
    from meep_gpu.driver import FdtdDriver                      # noqa: PLC0415

    record: Dict[str, Any] = {"leg": "nr>=2 refusal",
                              "minimum_radial_rows": int(cylmod.MINIMUM_RADIAL_ROWS)}
    driver = FdtdDriver(cell_size=NR1_CELL, resolution=NR1_RESOLUTION,
                        cylindrical=True, m=1, force_complex_fields=True,
                        courant=0.37, boundaries={"z": "metallic"},
                        prefer_gpu=xp is not np, gpu_id=0)
    try:
        driver.setup_pml({"z": 1})
        record["shape"] = [int(n) for n in driver.shape]
        if int(driver.shape[0]) != 1:
            raise AssertionError(
                f"the refusal leg wanted ONE radial row and built "
                f"{driver.shape[0]}; the clause would not be the difference")
        reasons: Dict[str, List[str]] = {}
        for sub_step in ("step_B", "step_D"):
            verdict = cylmod.cylindrical_complex_curl_coverage(
                driver.fields, driver.pml, sub_step, probe=probe)
            reasons[sub_step] = list(verdict.reasons)
            if verdict.covered:
                raise AssertionError(
                    f"the predicate ADMITTED a one-radial-row grid on "
                    f"{sub_step}; the over-coverage clause is gone")
            if not any("radial extent" in reason for reason in verdict.reasons):
                raise AssertionError(
                    f"{sub_step} was refused, but NOT by the radial-extent "
                    f"clause: {verdict.reasons}")
            if cylmod.plan_cylindrical_complex_curl(
                    driver.fields, driver.pml, sub_step, probe=probe) is not None:
                raise AssertionError(
                    f"the {sub_step} plan builder returned a plan for a grid "
                    f"its own predicate refuses")
        record["refusal_reasons"] = reasons
        # THE OTHER HALF: what the ARRAY PATH does here, measured against the
        # DECLARED per-backend expectation. The grid is SEEDED first, so a
        # backend that does not raise is shown to produce a live state rather
        # than an empty one — "it stepped" is a much weaker statement than "it
        # stepped to a finite, nonzero, entirely plausible wrong answer".
        backend = "numpy" if xp is np else "cupy"
        rng = np.random.default_rng(SEED + 9100)
        shape = tuple(int(n) for n in driver.shape)
        for name in PRIMARY_NAMES:
            host = (0.25 * rng.uniform(-1.0, 1.0, size=shape)
                    + 0.25j * rng.uniform(-1.0, 1.0, size=shape))
            driver.set_field(name, xp.asarray(
                np.ascontiguousarray(host.astype(np.complex64))))
        observed: str
        try:
            driver.step()
        except IndexError as exc:
            observed = "raises IndexError"
            record["array_path_exception"] = f"IndexError: {exc}"
        else:
            observed = ("steps without raising (cupy.take does not "
                        "bounds-check)" if backend == "cupy"
                        else "steps without raising")
            state = {name: to_host(xp, getattr(driver.fields, name))
                     for name in PRIMARY_NAMES}
            record["array_path_state_after"] = {
                "all_finite": bool(all(np.all(np.isfinite(v))
                                       for v in state.values())),
                "max_abs": {name: float(np.max(np.abs(value)))
                            for name, value in state.items()}}
        record["backend"] = backend
        record["array_path_observed"] = observed
        record["array_path_expected"] = ARRAY_PATH_ON_NR1[backend]
        if observed != ARRAY_PATH_ON_NR1[backend]:
            raise AssertionError(
                f"the ARRAY PATH on {backend} {observed!r} where this leg "
                f"declares {ARRAY_PATH_ON_NR1[backend]!r}. The clause's stated "
                f"reason is per-backend and MEASURED; a backend that changed "
                f"its behaviour must be re-measured and the declaration "
                f"updated, not silently accepted")
        record["ok"] = True
        return record
    finally:
        try:
            driver.close()
        except Exception:                            # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

#: The laptop stand-in for a measured expansion record. It licenses NOTHING
#: about device bytes; it exists so the preflight can drive the predicates and
#: the plan builders on NumPy. Since 2026-08-17 a record must state the policy
#: it was cut under and the run must DECLARE one (``declaring_run_policy``), so
#: the stub carries a ``keep`` stamp and the preflight declares ``keep`` — the
#: policy every weld in fingerprints.json records. A real record passed as
#: ``--probe-artifact`` to the dry run takes precedence over this stub.
LAPTOP_PROBE = {"backend": "cupy",
                "patterns": {name: "FMA_V1"
                             for name in cxmod.PROBE_PATTERNS},
                "subnormal_policy": {"policy": "ieee_keep_ftz_stripped",
                                     "resolved": "keep",
                                     "note": "LAPTOP STUB: licenses no device byte"},
                "candidates": {"policy": "keep"}}
LAPTOP_RUN_POLICY = "keep"


def _plans_from_arrays(driver, case) -> Dict[str, Any]:
    """Curl plans built WITHOUT the predicate, for the arming check only.

    The engine route refuses on a NumPy host by construction (the array-module
    clause), so the laptop cannot reach a plan that way — and an arming callable
    that has never run is a needle that is DISARMED for the price of a device
    slot rather than of a laptop second. This is the gate's own
    ``plan_..._from_arrays`` route: no predicate, no launch, and the object it
    returns is the same class the engine route builds.
    """
    fields, pml, grid = driver.fields, driver.pml, driver.grid
    from meep_gpu.triton_kernels.launch import SUB_STEPS          # noqa: PLC0415

    plans: Dict[str, Any] = {}
    for sub_step in ("step_B", "step_D"):
        spec = SUB_STEPS[sub_step]
        suffix = spec["suffix"]
        arrays = {name: getattr(fields, name)
                  for name in FIELD_STATE + AUX_STATE
                  if getattr(fields, name, None) is not None}
        flat = {f"{stem}_{axis}": getattr(pml, f"{stem}_{axis}{suffix}")
                for axis in "xyz" for stem in ("kms", "sinv")}
        plans[sub_step] = cylmod.plan_cylindrical_complex_curl_from_arrays(
            sub_step, arrays, flat, grid.dt / grid.dx, int(grid.m),
            bool(grid.accurate_fields_near_cylorigin),
            1 if driver.grid.is_metallic(2) else 0, 1, np)
    return plans


def run_arming(out_path: str) -> Dict[str, Any]:
    """Apply every armed mutation on the laptop and require it to BITE the plan.

    Not a byte measurement — no kernel runs here. It answers the one question a
    device slot must not be spent on: does each callable actually change the
    thing it names, on the case the table schedules it for?
    """
    by_name = {case["name"]: case for case in CASES}
    rows: List[Dict[str, Any]] = []
    for name, case_name, apply, expected_step, why in MUTATIONS:
        case = by_name[case_name]
        record: Dict[str, Any] = {"mutation": name, "case": case_name,
                                  "why": why}
        driver = build_driver(case, np, prefer_gpu=False)
        try:
            seed_state(driver, np, case)
            plans = _plans_from_arrays(driver, case)
            before = {
                "zero_rows": int(plans["step_B"].zero_rows),
                "increment": [float(plans["step_B"].increment_scalars[0]),
                              complex(plans["step_B"].increment_scalars[1][0]).real,
                              complex(plans["step_B"].increment_scalars[1][1]).real],
                "imr": [np.array(pointer.array, copy=True)
                        for pointer in plans["step_B"]._imr_rows],  # noqa: SLF001
                "four_dtdx": float(plans["step_D"].four_dtdx),
            }
            record["detail"] = apply(driver, plans, case)
            after_imr = [np.asarray(pointer.array)
                         for pointer in plans["step_B"]._imr_rows]  # noqa: SLF001
            bitten = False
            if name == "imr_row_negated":
                moved = sum(int(np.count_nonzero(a != b))
                            for a, b in zip(before["imr"], after_imr))
                record["moved_coefficient_words"] = moved
                bitten = moved > 0
            elif name == "zero_rows_reduced":
                record["zero_rows"] = [before["zero_rows"],
                                       int(plans["step_B"].zero_rows)]
                bitten = plans["step_B"].zero_rows == before["zero_rows"] - 1
            elif name == "m0_axis_add_scalar_negated":
                record["four_dtdx"] = [before["four_dtdx"],
                                       float(plans["step_D"].four_dtdx)]
                bitten = (np.float32(before["four_dtdx"]).tobytes()
                          != np.float32(plans["step_D"].four_dtdx).tobytes())
            else:
                after = [float(plans["step_B"].increment_scalars[0]),
                         complex(plans["step_B"].increment_scalars[1][0]).real,
                         complex(plans["step_B"].increment_scalars[1][1]).real]
                record["increment_scalars"] = [before["increment"], after]
                # WORD comparison, not ``!=``: the increment's real part is an
                # exact -0.0 here and ``-0.0 != 0.0`` is False in Python, so a
                # value comparison would call a real sign-bit flip a no-op.
                bitten = any(
                    np.float32(a).tobytes() != np.float32(b).tobytes()
                    for a, b in zip(before["increment"], after))
            record["armed"] = bool(bitten)
            if not bitten:
                record["error"] = "DISARMED: the callable changed nothing"
        except Exception as exc:                    # noqa: BLE001
            record["armed"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                driver.close()
            except Exception:                       # noqa: BLE001
                pass
        rows.append(record)
        log(f"[arming] {name} on {case_name}: armed={record['armed']} "
            f"{record.get('error', record.get('detail', ''))}")
    return {"rows": rows,
            "disarmed": [row["mutation"] for row in rows if not row["armed"]]}


def run_preflight(out_path: str, probe: Optional[Dict[str, Any]] = None) -> int:
    """Build every case on the ARRAY path and assert the seams with the
    array-module clause stripped. No launch, no byte claim: this exists so a
    broken enumeration is found on a laptop rather than on a device slot.

    ``probe`` is a measured expansion record when the caller has one; otherwise
    :data:`LAPTOP_PROBE`. Either way the run policy is DECLARED as ``keep`` for
    the duration, which is what the licence clause needs on a host that has
    installed nothing (``complex_fields._policy_in_force``).
    """
    from meep_gpu.expansion_refusal import declaring_run_policy  # noqa: PLC0415

    with declaring_run_policy(LAPTOP_RUN_POLICY):
        return _run_preflight(out_path, LAPTOP_PROBE if probe is None else probe)


def _run_preflight(out_path: str, probe: Dict[str, Any]) -> int:
    payload: Dict[str, Any] = {
        "probe": "triton_cylindrical_complex_composition",
        "mode": "preflight (NumPy drivers; NO byte-identity claim)",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "declared_run_policy": LAPTOP_RUN_POLICY,
        "probe_is_the_laptop_stub": probe is LAPTOP_PROBE,
        "step_budgets": {case["name"]: case["steps"] for case in CASES},
        "cases": [],
    }
    save(payload, out_path)
    failures: List[str] = []
    for case in CASES:
        started = time.time()
        record: Dict[str, Any] = {"case": case["name"], "steps": case["steps"]}
        try:
            driver = build_driver(case, np, prefer_gpu=False)
            record["shape"] = [int(n) for n in driver.shape]
            record["dtype"] = str(driver.fields.Bx.dtype)
            record["zero_rows"] = int(cylmod.zero_rows(case["m"],
                                                       case["accurate"]))
            record["m_class"] = int(cylmod.m_class(case["m"]))
            record["seeding_record"] = seed_state(driver, np, case)
            record["negative_zero_census_at_seed"] = negative_zero_census(
                np, driver)
            record["seams"] = seam_report(driver.fields, driver.pml,
                                          probe, drop_backend=True)
            assert_seams(record["seams"], case["name"])
            if case["seeding"] == "signed_zero_lattice" and \
                    record["negative_zero_census_at_seed"] == 0:
                raise AssertionError("VACUOUS seeding: census 0 at seed")
            # The composition really runs on the array path too, so a case that
            # cannot be stepped at all is found here rather than on the device,
            # and the +-0 trail is measured over the SAME budget the device leg
            # will use — the floor is the peak, and the peak needs the trail.
            trail: List[int] = []
            for _ in range(case["steps"]):
                driver.step()
                trail.append(negative_zero_census(np, driver))
            record["negative_zero_census_trail"] = trail
            record["negative_zero_census_peak"] = max(trail, default=0)
            record["axis_rule"] = axis_rule_witness(np, driver, case)
            if case["seeding"] == "signed_zero_lattice":
                # The floor the device leg asserts, measured here too: a lattice
                # that never reaches the state the sub-steps read is as vacuous
                # as one that was never seeded, and finding that out here costs
                # a laptop second rather than a device slot.
                if record["negative_zero_census_peak"] == 0:
                    raise AssertionError(
                        "VACUOUS: the +-0 lattice never reached the stepped "
                        "state at any step of the budget")
                assert_axis_rule_live(record["axis_rule"], case["name"])
            record["nan_census"] = nan_census(np, driver)
            record["ok"] = not (record["nan_census"]["nan_words"]
                                or record["nan_census"]["inf_words"])
            if not record["ok"]:
                record["error"] = f"non-finite words: {record['nan_census']}"
                failures.append(case["name"])
            driver.close()
        except Exception as exc:                    # noqa: BLE001
            record["ok"] = False
            record["error"] = f"{type(exc).__name__}: {exc}"
            failures.append(case["name"])
        record["seconds"] = round(time.time() - started, 3)
        payload["cases"].append(record)
        log(f"[preflight] {case['name']}: ok={record['ok']} "
            f"{record.get('error', '')}")
        save(payload, out_path)

    payload["arming"] = run_arming(out_path)
    if payload["arming"]["disarmed"]:
        failures.append(f"DISARMED mutations: {payload['arming']['disarmed']}")
    save(payload, out_path)

    try:
        payload["refusal"] = run_refusal(np, probe)
    except Exception as exc:                        # noqa: BLE001
        payload["refusal"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        failures.append("nr>=2 refusal")
    log(f"[preflight] nr>=2 refusal: ok={payload['refusal'].get('ok')} "
        f"{payload['refusal'].get('error', '')}")

    payload["failures"] = failures
    payload["status"] = "preflight passed" if not failures else "FAILED"
    payload["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(payload, out_path)
    log(f"[preflight] {payload['status']}; artifact {out_path}")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def environment() -> Dict[str, Any]:
    import triton                                               # noqa: PLC0415

    properties = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "cupy": cp.__version__, "triton": triton.__version__,
        "device": properties["name"].decode(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUPY_CACHE_DIR": os.environ.get("CUPY_CACHE_DIR"),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--probe-artifact", default=None,
                        help="the measured complex-expansion probe (device run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="NumPy preflight: enumeration, seams and refusal")
    args = parser.parse_args(argv)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if args.dry_run:
        record = (cxmod.load_expansion_probe(args.probe_artifact)
                  if args.probe_artifact else None)
        return run_preflight(out_path, record)

    if cp is None:
        log("[composition] cupy is not importable; NO byte-identity claim")
        return 75

    import gate_triton_complex as gate                           # noqa: PLC0415
    from meep_gpu import backends                                # noqa: PLC0415

    # Strip FIRST, guard second: the guard wraps whatever sits at the seam.
    gate.install_ftz_strip()
    backends.guard_kernel_compilation(cp)

    record = cxmod.load_expansion_probe(args.probe_artifact)
    reasons = list(gate.probe_record_policy_reasons(record)) if record else [
        f"no expansion probe artifact at {args.probe_artifact!r}; the EXPANSION "
        f"constexpr is a measured platform fact and may not be guessed"]
    if record is not None and cxmod.expansion_from_probe(record) is None:
        reasons.append("the probe artifact licenses no single EXPANSION constexpr")
    if reasons:
        payload = {"probe": "triton_cylindrical_complex_composition",
                   "status": "REFUSED", "refusal_reasons": reasons,
                   "byte_identity_claim": "NONE"}
        save(payload, out_path)
        log(f"[composition] REFUSED: {reasons}")
        return 1

    payload: Dict[str, Any] = {
        "probe": "triton_cylindrical_complex_composition",
        "module": "meep_gpu/triton_kernels/cylindrical_complex.py",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment(),
        "dispatch": "DISABLED (the engine's fast-path hook returns None)",
        "installed_slots": list(SLOTS),
        "expansion": int(cxmod.expansion_from_probe(record)),
        "step_budgets": {
            **{case["name"]: case["steps"] for case in CASES},
            "mutation_cases": 2,
            "note": "bit-identity is claimed for exactly these COMPLETE driver "
                    "steps and no further"},
        "subnormal_policy_at_start": gate.policy_stamp("cupy"),
        "timing_claim": ("NONE. Correctness only; this box carries foreign "
                         "load and nothing here is timed"),
        "cases": [], "mutations": [], "refusal": None,
    }
    save(payload, out_path)
    payload["provenance"] = write_provenance(os.path.dirname(out_path))
    save(payload, out_path)

    failures: List[str] = []
    for case in CASES:
        log(f"[composition] starting {case['name']}: {case['why']}")
        try:
            payload["cases"].append(run_case(cp, case, record))
        except Exception as exc:                    # noqa: BLE001
            payload["cases"].append({"case": case["name"], "ok": False,
                                     "error": f"{type(exc).__name__}: {exc}"})
            failures.append(case["name"])
            log(f"[composition] {case['name']} FAILED: {exc}")
        save(payload, out_path)

    by_name = {case["name"]: case for case in CASES}
    for mutation in MUTATIONS:
        case = by_name[mutation[1]]
        log(f"[composition] arming {mutation[0]} on {case['name']}")
        try:
            payload["mutations"].append(run_case(cp, case, record,
                                                 mutation=mutation))
        except Exception as exc:                    # noqa: BLE001
            payload["mutations"].append(
                {"case": f"{case['name']}::{mutation[0]}", "caught": False,
                 "error": f"{type(exc).__name__}: {exc}"})
            failures.append(f"{case['name']}::{mutation[0]}")
            log(f"[composition] mutation {mutation[0]} FAILED: {exc}")
        save(payload, out_path)

    try:
        payload["refusal"] = run_refusal(cp, record)
    except Exception as exc:                        # noqa: BLE001
        payload["refusal"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        failures.append("nr>=2 refusal")
    log(f"[composition] nr>=2 refusal: ok={payload['refusal'].get('ok')} "
        f"{payload['refusal'].get('error', '')}")
    save(payload, out_path)

    policy_problems = gate.ftz_strip_license_reasons()
    failures.extend("subnormal policy: " + reason for reason in policy_problems)
    payload["subnormal_policy"] = gate.policy_stamp("cupy")
    payload["nan_census_total"] = {
        "nan_words": sum(point["nan_census"]["nan_words"]
                         for row in payload["cases"] + payload["mutations"]
                         for point in row.get("per_step", ())),
        "inf_words": sum(point["nan_census"]["inf_words"]
                         for row in payload["cases"] + payload["mutations"]
                         for point in row.get("per_step", ()))}
    payload["summary"] = {
        "status": "passed" if not failures else "FAILED",
        "failures": failures,
        "cases_bit_identical": sum(1 for row in payload["cases"]
                                   if row.get("bit_identical")),
        "cases_total": len(CASES),
        "complete_driver_steps": sum(len(row.get("per_step", ()))
                                     for row in payload["cases"]),
        "compared_words": sum(point["total_words"] for row in payload["cases"]
                              for point in row.get("per_step", ())),
        "differing_words": sum(point["differing_words"]
                               for row in payload["cases"]
                               for point in row.get("per_step", ())),
        "mutations_caught": sum(1 for row in payload["mutations"]
                                if row.get("caught")),
        "mutations_total": len(MUTATIONS),
        "byte_identity_claim": (
            "claimed for the COMPLETE driver steps this artifact records, at "
            "the budgets it states, and no further"
            if not failures else "NONE: this run failed"),
    }
    payload["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(payload, out_path)
    log(f"[composition] {payload['summary']['status']}: "
        f"{json.dumps(payload['summary'], sort_keys=True)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
