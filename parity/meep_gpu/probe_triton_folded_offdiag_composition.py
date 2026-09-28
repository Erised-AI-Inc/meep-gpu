"""Complete-driver CUDA gate for the FOLDED off-diagonal epsilon composition.

The sub-step gate (``gate_triton_folded_offdiag.py``) certifies the kernel
against an in-file transcription pinned to ``stepping.update_E``. This probe
asks the composition question, which is a different one: with ``step_B`` and
``step_D`` served by the CERTIFIED FOLDED curl kernel, ``fill_B``/``fill_D`` by
the CERTIFIED MIRROR GHOST FILL, ``update_H`` by the CERTIFIED constitutive
kernel on a validated folded extent, and ``update_E`` by the NEW folded
off-diagonal kernel — installed by monkeypatch, production dispatch untouched,
``plan_fast_path`` still returning None — does the complete driver state
reproduce the array-path driver EXACTLY after EVERY COMPLETE STEP, for a STATED
budget, sources, PML, ghost fills and metal zeroing included?

THIS IS THE PARTIAL-UNLOCK MEASUREMENT FOR THE FOLDED FAMILY, and the unlock it
completes is narrow and stated. On a folded grid carrying off-diagonal rows:

* ``symmetry.folded_pml_curl_coverage``      ADMITS   -> step_B, step_D
* ``symmetry.mirror_ghost_fill_coverage``    ADMITS   -> fill_B, fill_D
* ``symmetry.folded_constitutive_coverage(H)`` ADMITS -> update_H
* ``symmetry.folded_constitutive_coverage(E)`` REFUSES, for the OFF-DIAGONAL
  clause (symmetry.py:814-817) and nothing else — the disjointness seam
* ``offdiag_update_e.offdiag_constitutive_coverage`` REFUSES, for the FOLD
  clause (coverage.py:171-175) and nothing else — the other seam
* ``folded_offdiag_update_e.folded_offdiag_constitutive_coverage`` ADMITS

so ``update_E`` is the LAST uncovered sub-step of such a run, not the first.
Both seams are ASSERTED per case, by name, before any step is taken: a seam
that narrowed underfoot would make this probe's install route wrong rather than
its numbers wrong, and that has to fail loudly.

WHAT EACH CASE IS THE ONLY WITNESS TO. The demand the family's own gate spans is
carried here at DRIVER level, where the fold's ghost fills and the curls run
between successive ``update_E`` calls:

1. ``foldY_periodic_even_varying``   folded PERIODIC (Gamma point), plane phase
   +1, EVEN stored count, spatially varying rows, three CW point sources.
2. ``foldY_metallic_odd``            folded METALLIC, plane phase -1, ODD stored
   count. A sweep carrying one termination proves nothing about the other, and
   the parity sign is the family's whole delta.
3. ``foldXY_two_planes``             TWO folded axes at once, mixed phases: every
   row slot then takes a ghosted down shift under weights that are not all
   equal.
4. ``foldY_partial_row``             a single surviving slot ``Ez <- Ex`` on a
   folded Y: two components compile the plain arm while one takes the coupled
   arm, over whole steps.
5. ``foldY_signed_zero_lattice``     THE +-0 LATTICE SEEDING (below), no source.
6. ``foldY_reduced_2d``              the 2-D reduced run: the invariant axis's
   partner pair is g+g (MEEP's stride(d)=0 double read), composed over whole
   steps.

THE +-0 LATTICE SEEDING, AND WHY A PLAIN ZERO-INIT IS NOT ONE. The cylindrical
round measured that ZERO-INIT IS A FIXED POINT of the constitutive sub-step: with
every array zero the sub-step writes zeros, the mirror ghost's parity sign is
annihilated by the ``values + ghost`` add (``+0.0 + -0.0 == +0.0``), the
negative-zero census is 0 and a sign-flipped ghost is INVISIBLE. A leg seeded
that way measures nothing and must be recorded VACUOUS, never as coverage.

What is seeded instead is a LATTICE of signed zeros — ``+0.0`` and ``-0.0`` by
index parity across every primary — with stored row ``MIRROR_SOURCE_INDEX`` of
the folded axis carrying real amplitude, so the mirror ghost is the only path to
a nonzero fold-plane E while the signed-zero background stays in place for the
sign bit to be visible against.

PER-SEEDING VACUITY FLOOR, checked at the seed AND after the first complete step,
each failing the case by name rather than passing quietly:

* the negative-zero census over the stored primaries must be > 0 at seed time
  (census 0 = VACUOUS, not passed);
* it must still be > 0 after the first complete step, or the seeding did not
  survive into the state the sub-step reads;
* the fold plane's max |E| must be > 0 after the first step, or the ghost never
  reached the plane the case exists to watch;
* every word must stay FINITE (platform fact (f)): a NaN's sign and payload are
  IEEE-unspecified, so a raw-word compare over one is not a measurement. The NaN
  census is reported for every case, seeding or not.

LAUNCH COUNTING. All six installed plans are wrapped in :class:`PlanCounter`;
the artifact records per-slot launches and a slot whose count differs from the
case's step budget FAILS the case loudly. Bit-identical rows without launch
evidence certify nothing — the harness-disarm failure the certified families
document three times over.

ARMED MUTATIONS, both launch-counted, both required to be caught AT STEP 1
(the coupling and the ghost are live at the first ``update_E``):

* ``coefficient_doubled``    — the installed row volume the candidate's plan
  points at is doubled IN PLACE after the plans are built, so only the Triton
  kernel reads the doubled bytes while the array-path reference keeps the true
  material. The certified family's own composition needle, re-run folded.
* ``ghost_parity_flipped``   — THIS family's delta at composition level: the
  update_E plan is rebuilt with the folded axis's ghost weight NEGATED. The
  true weight is read first and the mutant is its negation, so the arming can
  never be a no-op on a case whose parity happens to be +1.

A SOURCE-FREE CONTROL driver runs beside every sourced case and the source must
separate the reference from it, so a case cannot pass with the source dark.

Requirements carried from prior defects: per-COMPLETE-STEP uint32 compare of
every allocated stored volume, never allclose; the step budget STATED in the
artifact; unbuffered per-step progress; atomic JSON rewrite per case; own
provenance record; SUBNORMAL POLICY under the stripped IEEE-keep policy with the
in-run strip counters stamped AFTER the cases run. Correctness only: no
throughput or timing claim is made or admissible.

Usage (the GPU host, one clear device)::

    CUDA_VISIBLE_DEVICES=N \\
    CUPY_CACHE_DIR=$RUN_ROOT/results/cupy_cache_ftz_stripped_$JOB \\
        python -u probe_triton_folded_offdiag_composition.py \\
            --out results/triton_folded_offdiag_<date>/composition.json

Laptop preflight (NumPy drivers, no launches, no byte claim)::

    python -u probe_triton_folded_offdiag_composition.py --dry-run \\
        --out /tmp/folded_offdiag_composition_preflight.json
"""

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

from meep_gpu.triton_kernels import folded_offdiag_update_e as fomod  # noqa: E402
from meep_gpu.triton_kernels import offdiag_update_e as odmod         # noqa: E402
from meep_gpu.triton_kernels import symmetry as symmod                # noqa: E402

SEED = 20260813
E_NAMES = ("Ex", "Ey", "Ez")
PRIMARY_NAMES = ("Bx", "By", "Bz", "Dx", "Dy", "Dz")
FIELD_STATE = ("Bx", "By", "Bz", "Dx", "Dy", "Dz",
               "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
AUX_STATE = ("fu_Bx", "fu_By", "fu_Bz", "fu_Dx", "fu_Dy", "fu_Dz",
             "f_w_Ex", "f_w_Ey", "f_w_Ez", "f_w_Hx", "f_w_Hy", "f_w_Hz")

#: test_tensor_epsilon._EPS_TENSOR's inverse: its off-diagonals are NEGATIVE, so
#: the uniform row form already carries negative coefficients.
EPS_TENSOR = np.array([[2.0, 0.35, 0.20],
                       [0.35, 2.5, 0.15],
                       [0.20, 0.15, 3.0]], dtype=np.float64)
INV_TENSOR = np.linalg.inv(EPS_TENSOR)

#: The six driver slots this probe serves, in the order the driver calls them.
SLOTS = ("step_B", "fill_B", "update_H", "step_D", "fill_D", "update_E")


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
        "triton_kernels/symmetry.py": os.path.join(kernels, "symmetry.py"),
        "triton_kernels/offdiag_update_e.py": os.path.join(
            kernels, "offdiag_update_e.py"),
        "triton_kernels/folded_offdiag_update_e.py": os.path.join(
            kernels, "folded_offdiag_update_e.py"),
        "parity/probe_triton_folded_offdiag_composition.py":
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

CASES: Tuple[Dict[str, Any], ...] = (
    {"name": "foldY_periodic_even_varying", "mirrors": (("Y", 1),),
     "cell": (1.6, 2.0, 0.8), "resolution": 15.0, "dimensions": 3,
     "boundaries": ("periodic", "periodic", "periodic"), "courant": 0.35,
     "pml": {"x": 4, "z": 3}, "rows": "varying_full", "steps": 10,
     "seeding": "random", "sources": "three_cw",
     "why": "folded PERIODIC at the Gamma point, phase +1, even stored count"},
    {"name": "foldY_metallic_odd", "mirrors": (("Y", -1),),
     "cell": (1.6, 2.0666666666666667, 0.8), "resolution": 15.0,
     "dimensions": 3, "boundaries": ("metallic", "metallic", "periodic"),
     "courant": 0.35, "pml": {"x": 4}, "rows": "uniform_full", "steps": 10,
     "seeding": "random", "sources": "gaussian_ez",
     "source_center": (0.0, 0.30, 0.0),
     "why": "folded METALLIC, plane phase -1, ODD stored count"},
    {"name": "foldXY_two_planes", "mirrors": (("X", 1), ("Y", -1)),
     "cell": (2.0, 2.0, 0.8), "resolution": 12.0, "dimensions": 3,
     "boundaries": ("periodic", "periodic", "periodic"), "courant": 0.3125,
     "pml": {"z": 3}, "rows": "varying_full", "steps": 8,
     "seeding": "random", "sources": "gaussian_ez",
     "source_center": (0.25, 0.30, 0.0),
     "why": "TWO folded axes at once under ghost weights that differ"},
    {"name": "foldY_partial_row", "mirrors": (("Y", 1),),
     "cell": (1.6, 2.0, 0.8), "resolution": 15.0, "dimensions": 3,
     "boundaries": ("metallic", "periodic", "periodic"), "courant": 0.35,
     "pml": {"x": 4}, "rows": "single_ez_ex", "steps": 8,
     "seeding": "random", "sources": "gaussian_ez",
     "why": "one surviving slot: two components take the plain arm"},
    {"name": "foldY_signed_zero_lattice", "mirrors": (("Y", 1),),
     "cell": (1.6, 2.0, 0.8), "resolution": 15.0, "dimensions": 3,
     "boundaries": ("periodic", "periodic", "periodic"), "courant": 0.35,
     "pml": {"x": 4, "z": 3}, "rows": "varying_full", "steps": 6,
     "seeding": "signed_zero_lattice", "sources": "none",
     "why": "the +-0 LATTICE seeding; a plain zero-init is a FIXED POINT here"},
    {"name": "foldY_reduced_2d", "mirrors": (("Y", 1),),
     "cell": (4.8, 4.8, 0.0), "resolution": 20.0, "dimensions": 2,
     "boundaries": ("periodic", "periodic", "periodic"), "courant": 0.35,
     "pml": {"x": 5}, "rows": "uniform_full", "steps": 8,
     "seeding": "random", "sources": "gaussian_ez",
     "why": "the 2-D reduced run: the invariant axis's partner pair is g+g"},
)

# SOURCE PLACEMENT IS CONSTRAINED BY THE FOLD, and the engine says so rather
# than depositing a plausible wrong answer: a point source ON the mirror plane
# whose component the plane makes ODD carries only zero current, and the engine
# REFUSES it (MEEP does not — it deposits with use_symmetry=false and returns a
# field measured 57%-283% away from the full-domain run). Every source below is
# therefore either OFF the plane or on a component the plane leaves EVEN, and
# the preflight is what caught the three declarations that were neither.
THREE_CW = (("Ex", (0.15, 0.30, -0.15), 0.7),
            ("Ey", (-0.15, 0.25, 0.05), 0.5),
            ("Ez", (0.05, 0.35, -0.25), 1.0))


def source_declarations(kind: str,
                        center: Sequence[float] = (0.0, 0.0, 0.0)
                        ) -> List[Dict[str, Any]]:
    if kind == "none":
        return []
    if kind == "three_cw":
        return [{"component": component, "frequency": 0.7, "center": at,
                 "size": (0.0, 0.0, 0.0), "amplitude": amplitude}
                for component, at, amplitude in THREE_CW]
    if kind == "gaussian_ez":
        return [{"component": "Ez", "source_type": "gaussian",
                 "frequency": 0.8, "fwidth": 0.4, "center": tuple(center),
                 "size": (0.0, 0.0, 0.0), "amplitude": 1.0}]
    raise ValueError(f"unknown source kind {kind!r}")


def _uniform(shape, value) -> np.ndarray:
    return np.full(shape, value, dtype=np.float32)


def install_tensor(driver, rows_kind: str) -> None:
    """Install the oracle tensor through ``set_epsilon_components`` — the same
    route ``from_meep``'s epsilon_offdiag ingest takes."""
    shape = tuple(int(n) for n in driver.shape)
    volumes = {name: _uniform(shape, 1.0 / INV_TENSOR[i, i])
               for i, name in enumerate(E_NAMES)}
    if rows_kind == "uniform_full":
        rows = {row: {partner: _uniform(shape, INV_TENSOR[i, j])
                      for j, partner in enumerate(E_NAMES) if j != i}
                for i, row in enumerate(E_NAMES)}
    elif rows_kind == "varying_full":
        rng = np.random.default_rng(SEED + 101)
        rows = {row: {partner: (_uniform(shape, INV_TENSOR[i, j])
                                * rng.uniform(0.5, 1.5, size=shape)
                                .astype(np.float32)).astype(np.float32)
                      for j, partner in enumerate(E_NAMES) if j != i}
                for i, row in enumerate(E_NAMES)}
    elif rows_kind == "single_ez_ex":
        rows = {"Ez": {"Ex": _uniform(shape, INV_TENSOR[2, 0])}}
    else:
        raise ValueError(f"unknown row form {rows_kind!r}")
    driver.set_epsilon_components(volumes, chi1inv_offdiagonal=rows)


def build_driver(case: Dict[str, Any], xp, prefer_gpu: bool,
                 with_source: bool = True, with_rows: bool = True):
    from meep_gpu.driver import FdtdDriver                     # noqa: PLC0415
    from meep_gpu.grid import Mirror                           # noqa: PLC0415

    mirrors = tuple(Mirror(axis, phase) for axis, phase in case["mirrors"])
    driver = FdtdDriver(
        cell_size=case["cell"], resolution=case["resolution"],
        dimensions=case["dimensions"], courant=case["courant"],
        boundaries=case["boundaries"], symmetry=mirrors,
        force_complex_fields=False, prefer_gpu=prefer_gpu, gpu_id=0)
    if with_rows:
        install_tensor(driver, case["rows"])
    else:
        shape = tuple(int(n) for n in driver.shape)
        driver.set_epsilon_components(
            {name: _uniform(shape, 1.0 / INV_TENSOR[i, i])
             for i, name in enumerate(E_NAMES)})
    # A folded axis's PLANE face carries no absorber (a layer there would eat
    # the mirrored half); the driver's own arrangement.
    folded = {letter.lower() for letter, _phase in case["mirrors"]}
    spec = {axis: ({"high": cells} if axis in folded else cells)
            for axis, cells in case["pml"].items()}
    driver.setup_pml(spec)
    if with_source:
        for declaration in source_declarations(
                case["sources"], case.get("source_center", (0.0, 0.0, 0.0))):
            driver.add_source(dict(declaration))
    return driver


# ---------------------------------------------------------------------------
# Seeding, and the +-0 lattice's own vacuity floor
# ---------------------------------------------------------------------------

def folded_axes(driver) -> List[int]:
    return [axis for axis in range(3) if driver.grid.is_mirrored(axis)]


def _face(axis: int, index: Any) -> Tuple[Any, ...]:
    return (slice(None),) * axis + (index,)


def seed_state(driver, xp, case: Dict[str, Any]) -> Dict[str, Any]:
    """Seed the primaries and return the seeding's own record."""
    shape = tuple(int(n) for n in driver.shape)
    kind = case["seeding"]
    if kind == "random":
        rng = np.random.default_rng(SEED + 3000 + len(case["name"]))
        for name in PRIMARY_NAMES:
            host = np.ascontiguousarray(
                (0.25 * rng.uniform(-1.0, 1.0, size=shape)).astype(np.float32))
            driver.set_field(name, xp.asarray(host))
        return {"kind": kind}
    if kind != "signed_zero_lattice":
        raise ValueError(f"unknown seeding {kind!r}")

    # THE +-0 LATTICE. Zero everywhere, but with the SIGN alternating by index
    # parity, so the background carries negative zeros for a sign-bit defect to
    # be visible against; stored row MIRROR_SOURCE_INDEX of the folded axis then
    # carries real amplitude on the D primaries, which makes the mirror ghost
    # the only path to a nonzero fold-plane E.
    axis = folded_axes(driver)[0]
    index = np.indices(shape).sum(axis=0)
    lattice = np.where((index % 2) == 0, np.float32(0.0), np.float32(-0.0))
    lattice = np.ascontiguousarray(lattice.astype(np.float32))
    rng = np.random.default_rng(SEED + 3100)
    plane_shape = tuple(n for i, n in enumerate(shape) if i != axis)
    for name in PRIMARY_NAMES:
        host = lattice.copy()
        if name.startswith("D"):
            host[_face(axis, fomod.MIRROR_SOURCE_INDEX)] = rng.uniform(
                0.2, 0.6, plane_shape).astype(np.float32)
        driver.set_field(name, xp.asarray(np.ascontiguousarray(host)))
    return {"kind": kind, "needle_row": int(fomod.MIRROR_SOURCE_INDEX),
            "folded_axis": axis}


def to_host(xp, array) -> np.ndarray:
    return np.asarray(array) if xp is np else np.asarray(xp.asnumpy(array))


def negative_zero_census(xp, driver, names=PRIMARY_NAMES) -> int:
    """Stored words that are NEGATIVE ZERO — not merely negative.

    ``(x == 0) & signbit(x)``, never a sign-bit tally: on a continuous seed the
    latter counts about half the stored words and would relabel an ordinary
    negative count as this class's witness. Census 0 means the case cannot see a
    defect that lives in the sign of a zero — VACUOUS, not passed.
    """
    total = 0
    for name in names:
        array = getattr(driver.fields, name, None)
        if array is None:
            continue
        host = to_host(xp, array)
        total += int(np.count_nonzero((host == 0) & np.signbit(host)))
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


def fold_plane_max_abs_e(xp, driver, axis: int) -> float:
    return max(float(np.max(np.abs(to_host(xp, getattr(driver.fields, name))
                                   [_face(axis, 0)])))
               for name in E_NAMES)


# ---------------------------------------------------------------------------
# The seams — asserted by name before any step is taken
# ---------------------------------------------------------------------------

def seam_report(fields, pml, drop_backend: bool) -> Dict[str, Any]:
    def strip(reasons: Sequence[str]) -> List[str]:
        out = list(reasons)
        if drop_backend:
            out = [reason for reason in out if "array module" not in reason]
        return out

    certified_e = strip(symmod.folded_constitutive_coverage(
        fields, pml, "E").reasons)
    unfolded = strip(odmod.offdiag_constitutive_coverage(fields, pml).reasons)
    new = strip(fomod.folded_offdiag_constitutive_coverage(fields, pml).reasons)
    return {
        "folded_curl": strip(symmod.folded_pml_curl_coverage(fields, pml).reasons),
        "ghost_fill_B": strip(symmod.mirror_ghost_fill_coverage(fields, "B").reasons),
        "ghost_fill_D": strip(symmod.mirror_ghost_fill_coverage(fields, "D").reasons),
        "folded_constitutive_H": strip(
            symmod.folded_constitutive_coverage(fields, pml, "H").reasons),
        "folded_constitutive_E": {
            "offdiag_clause": [r for r in certified_e if "off-diagonal" in r],
            "other": [r for r in certified_e if "off-diagonal" not in r]},
        "certified_offdiag_E": {
            "fold_clause": [r for r in unfolded
                            if "mirror" in r or "fold" in r or "symmetr" in r],
            "other": [r for r in unfolded
                      if not ("mirror" in r or "fold" in r or "symmetr" in r)]},
        "folded_offdiag_E": {"covered": not new, "reasons": new},
    }


def assert_seams(report: Dict[str, Any], case_name: str) -> None:
    for slot in ("folded_curl", "ghost_fill_B", "ghost_fill_D",
                 "folded_constitutive_H"):
        if report[slot]:
            raise AssertionError(
                f"{case_name}: the certified {slot} predicate refuses this "
                f"folded off-diagonal run — the partial-unlock premise does "
                f"not hold here: {report[slot]}")
    if not report["folded_constitutive_E"]["offdiag_clause"]:
        raise AssertionError(
            f"{case_name}: symmetry's E-side predicate no longer refuses the "
            f"off-diagonal clause; the disjointness seam (symmetry.py:814-817) "
            f"was narrowed underfoot and this install route must be rebuilt")
    if report["folded_constitutive_E"]["other"]:
        raise AssertionError(
            f"{case_name}: symmetry's E-side predicate refuses this run for "
            f"MORE than the off-diagonal clause: "
            f"{report['folded_constitutive_E']['other']}")
    if not report["certified_offdiag_E"]["fold_clause"]:
        raise AssertionError(
            f"{case_name}: the CERTIFIED off-diagonal predicate no longer "
            f"refuses the fold; the other disjointness seam "
            f"(coverage.py:171-175) was narrowed underfoot")
    if not report["folded_offdiag_E"]["covered"]:
        raise AssertionError(
            f"{case_name}: the folded off-diagonal update_E predicate refused: "
            f"{report['folded_offdiag_E']['reasons']}")


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


def ghost_fill_install_is_sound(driver) -> Tuple[bool, str]:
    """Can the COMBINED ghost-fill plan be installed at the near slot here?

    MEASURED FINDING, and it belongs to the composition INSTALL CONVENTION, not
    to any kernel. The driver's order is

        step_B -> sources -> fill_symmetry_bc_B -> zero_metal_B
               -> fill_folded_far_ghosts_B -> update_H          (driver.py:3205-3211)

    with ``zero_metal_*`` BETWEEN the two fill calls, deliberately (MEEP's
    ``step_boundaries`` runs after ``step_source``, so a current deposited on a
    perfect conductor is injected first and wiped second — driver.py:3175-3183).
    ``symmetry.MirrorGhostFillPlan`` combines the near and far passes into ONE
    launch, so installing it at the near slot and suppressing the far call moves
    the far ghosts to BEFORE the metal zeroing.

    That reordering is invisible unless BOTH are live at once — a folded axis
    with a far reflect row (a folded PERIODIC EVEN axis) AND a metallic axis —
    and the certified symmetry composition's own case list carries no such
    configuration: its metallic case folds a METALLIC axis, which has no reflect
    row. This probe's ``foldY_partial_row`` is the first, and it diverged at
    step 1 in 48 words across ``Bx``, ``Dy``, ``f_w_Ey`` and ``f_w_Hx`` — a
    pattern that starts at the B-side fill, three sub-steps upstream of this
    family's ``update_E``. The ATTRIBUTION leg measures that directly rather
    than arguing it.

    Where the reordering is live, the fills stay on the ARRAY PATH and the four
    CORE slots — including this family's ``update_E`` — are still served by
    kernels. The artifact records the decision and this reason per case.
    """
    from meep_gpu import stepping                                # noqa: PLC0415

    grid = driver.grid
    reflect = stepping._far_reflect_rows(grid)
    far_live = [axis for axis in range(3) if reflect[axis] is not None]
    metallic = [axis for axis in range(3) if grid.is_metallic(axis)]
    if far_live and metallic:
        return False, (
            f"zero_metal_* runs BETWEEN the near and far fill calls "
            f"(driver.py:3208-3210, :3222-3224) and MirrorGhostFillPlan "
            f"combines the two passes, so installing it at the near slot "
            f"reorders the far ghosts across the metal zeroing. Both are live "
            f"here: far reflect rows on axes {far_live}, metallic axes "
            f"{metallic}. The fills run on the ARRAY PATH for this case; the "
            f"four core slots, update_E included, are still kernels.")
    return True, ""


CORE_SLOTS = ("step_B", "update_H", "step_D", "update_E")


def build_plans(driver) -> Dict[str, Any]:
    """All six sub-step plans from the ENGINE's own objects.

    A builder that returns None here is a regression in the family that owns it,
    not an expected refusal: :func:`assert_seams` has already measured that every
    one of these predicates admits this run.
    """
    fields, pml = driver.fields, driver.pml
    plans = {
        "step_B": symmod.plan_folded_pml_curl(fields, pml, "step_B"),
        "fill_B": symmod.plan_mirror_ghost_fill(fields, "B"),
        "update_H": symmod.plan_folded_constitutive(fields, pml, "H"),
        "step_D": symmod.plan_folded_pml_curl(fields, pml, "step_D"),
        "fill_D": symmod.plan_mirror_ghost_fill(fields, "D"),
        "update_E": fomod.plan_folded_offdiagonal_constitutive(fields, pml),
    }
    for slot, plan in plans.items():
        if plan is None:
            raise AssertionError(
                f"the {slot} engine builder refused a run its predicate was "
                f"asserted to admit")
    return plans


def install(driver_module, plans: Dict[str, Any], owner):
    """Serve the six sub-steps from the plans, for the OWNER fields only.

    Sources, ``zero_metal_*`` and the guards stay the driver's own; production
    dispatch is untouched (this is a monkeypatch on the driver module, and
    ``plan_fast_path`` still returns None everywhere).

    ``MirrorGhostFillPlan`` combines the near and far passes, so it runs in the
    driver's FIRST fill slot and the later far-fill call is suppressed for this
    owner only — the arrangement the certified symmetry composition uses.
    """
    core = ("step_B", "update_H", "step_D", "update_E", "update_P")
    near = {"fill_symmetry_bc_B": "fill_B", "fill_symmetry_bc_D": "fill_D"}
    far = {"fill_folded_far_ghosts_B": "fill_B",
           "fill_folded_far_ghosts_D": "fill_D"}
    names = core + tuple(near) + tuple(far)
    originals = {name: getattr(driver_module, name) for name in names}

    def core_wrapper(name):
        def wrapper(fields, pml=None):
            entry = plans.get(name) if fields is owner else None
            if entry is None:
                return originals[name](fields, pml)
            entry.run()
            return None
        return wrapper

    def near_wrapper(name, slot):
        def wrapper(fields):
            entry = plans.get(slot) if fields is owner else None
            if entry is None:
                return originals[name](fields)
            entry.run()
            return None
        return wrapper

    def far_wrapper(name, slot):
        def wrapper(fields):
            if fields is owner and slot in plans:
                return None
            return originals[name](fields)
        return wrapper

    for name in core:
        setattr(driver_module, name, core_wrapper(name))
    for name, slot in near.items():
        setattr(driver_module, name, near_wrapper(name, slot))
    for name, slot in far.items():
        setattr(driver_module, name, far_wrapper(name, slot))
    return lambda: [setattr(driver_module, name, function)
                    for name, function in originals.items()]


# ---------------------------------------------------------------------------
# The armed mutations
# ---------------------------------------------------------------------------

def arm_coefficient_doubled(driver, plans: Dict[str, Any]) -> str:
    """Double the installed row volume the CANDIDATE's plan points at.

    The plan holds the pointer, so the kernel reads the doubled bytes while the
    array-path reference keeps the true material — an in-place edit after the
    plans are built, which is what makes it a mutation of the kernel's input and
    not of the physics both sides see.
    """
    rows = odmod.row_volumes_for(driver.fields)
    for index, volume in enumerate(rows):
        if volume is not None:
            volume *= 2.0
            return f"row slot {index} ({odmod.ROW_SLOTS[index]}) doubled in place"
    raise AssertionError("no surviving row slot to double")


def arm_ghost_parity_flipped(driver, plans: Dict[str, Any]) -> str:
    """Rebuild ``update_E`` with the folded axis's ghost weight NEGATED.

    The true weight is READ FIRST and the mutant is its negation, so this can
    never be a no-op on a case whose parity happens to be +1 — the arming
    failure a hard-coded ``+1.0`` would have.
    """
    fields, pml = driver.fields, driver.pml
    plan = plans["update_E"]
    weights = list(plan.ghost_weights)
    axes = [axis for axis in range(3) if plan.ghost_axes[axis]]
    if not axes:
        raise AssertionError("no ghosted axis to flip")
    for axis in axes:
        weights[axis] = -weights[axis]
    codes, _ = symmod.folded_axis_kinds(fields.grid, pml)
    targets = tuple(term[0] for term in fomod.E_TERMS)
    volumes = fields.displacement_minus_polarization_volumes()
    plans["update_E"] = fomod.FoldedOffdiagConstitutivePlan(
        fields.grid.shape, fomod.DEFAULT_BLOCK,
        [getattr(fields, name) for name in targets],
        [getattr(fields, "f_w_" + name) for name in targets],
        [volumes[name] for name in targets],
        [fields.inverse_epsilon_for(name) for name in targets],
        odmod.row_volumes_for(fields),
        [getattr(pml, f"{stem}_{axis}_h")
         for axis in "xyz" for stem in ("kps", "kms")],
        codes, odmod.wall_mask_axes(fields.grid), weights)
    return (f"ghost weights {list(plan.ghost_weights)} -> {weights} on axes "
            f"{axes}")


MUTATIONS: Tuple[Tuple[str, Callable[..., str], str], ...] = (
    ("coefficient_doubled", arm_coefficient_doubled,
     "the certified family's composition needle, re-run folded: only the "
     "kernel reads the doubled row volume"),
    ("ghost_parity_flipped", arm_ghost_parity_flipped,
     "THIS family's delta at composition level: the mirror ghost's parity "
     "weight negated, read from the plan so the arming cannot be a no-op"),
)


# ---------------------------------------------------------------------------
# One composition case
# ---------------------------------------------------------------------------

def compare(xp, left, right) -> Dict[str, Any]:
    a = np.ascontiguousarray(to_host(xp, left)).view(np.uint32).ravel()
    b = np.ascontiguousarray(to_host(xp, right)).view(np.uint32).ravel()
    return {"bit_identical": bool(np.array_equal(a, b)),
            "differing_floats": int(np.count_nonzero(a != b)),
            "total_floats": int(a.size)}


def driver_state(driver) -> Dict[str, Any]:
    return {name: getattr(driver.fields, name)
            for name in FIELD_STATE + AUX_STATE
            if getattr(driver.fields, name, None) is not None}


def run_case(xp, case: Dict[str, Any],
             mutation: Optional[Tuple[str, Callable[..., str], str]] = None,
             slots: Optional[Sequence[str]] = None,
             steps_override: Optional[int] = None,
             tolerate_divergence: bool = False) -> Dict[str, Any]:
    from meep_gpu import driver as driver_module                # noqa: PLC0415

    name = case["name"] + ("" if mutation is None else f"::{mutation[0]}")
    steps = case["steps"] if steps_override is None else steps_override
    reference = build_driver(case, xp, prefer_gpu=True)
    candidate = build_driver(case, xp, prefer_gpu=True)
    control = (build_driver(case, xp, prefer_gpu=True, with_source=False)
               if case["sources"] != "none" else None)
    undo: Callable[[], Any] = lambda: None                      # noqa: E731
    row: Dict[str, Any] = {
        "case": name, "base_case": case["name"], "why": case["why"],
        "steps": steps, "mutation": None if mutation is None else mutation[0],
        "mutation_why": None if mutation is None else mutation[2],
        "seeding": case["seeding"], "rows": case["rows"],
        "boundaries": list(case["boundaries"]),
        "mirrors": [list(spec) for spec in case["mirrors"]],
        "courant": case["courant"],
        "courant_is_power_of_two": bool(
            abs(case["courant"] * 2 ** 8 - round(case["courant"] * 2 ** 8)) < 1e-12
            and (round(case["courant"] * 2 ** 8)
                 & (round(case["courant"] * 2 ** 8) - 1)) == 0),
        "per_step": [],
    }
    try:
        row["shape"] = [int(n) for n in candidate.shape]
        row["stored_cells"] = [int(candidate.grid.stored_cells(a))
                               for a in range(3)]
        row["stored_count_parity"] = [
            "even" if int(candidate.grid.stored_cells(a)) % 2 == 0 else "odd"
            for a in range(3)]
        row["folded_axes"] = folded_axes(candidate)
        seeds = {}
        for target in (reference, candidate) + ((control,) if control else ()):
            seeds = seed_state(target, xp, case)
        row["seeding_record"] = seeds
        row["seams"] = seam_report(candidate.fields, candidate.pml,
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

        plans = build_plans(candidate)
        sound, why_not = ghost_fill_install_is_sound(candidate)
        row["ghost_fills_installable"] = sound
        row["ghost_fill_install_note"] = why_not
        if slots is None:
            slots = SLOTS if sound else CORE_SLOTS
        row["installed_slots"] = list(slots)
        plans = {slot: plan for slot, plan in plans.items() if slot in slots}
        row["plan_types"] = {slot: type(plan).__name__
                             for slot, plan in plans.items()}
        if "update_E" in plans:
            row["update_E_ghost_weights"] = list(plans["update_E"].ghost_weights)
            row["update_E_boundary_codes"] = list(plans["update_E"].boundary_codes)
            row["update_E_row_mask"] = list(plans["update_E"].row_mask)
        if mutation is not None:
            row["mutation_detail"] = mutation[1](candidate, plans)
        counters = {slot: PlanCounter(plan) for slot, plan in plans.items()}
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
            point = {
                "step": step,
                "bit_identical": all(part["bit_identical"]
                                     for part in parts.values()),
                "differing_words": sum(part["differing_floats"]
                                       for part in parts.values()),
                "total_words": sum(part["total_floats"]
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
                f"nan={census['nan_words']} ({time.time() - started:.1f} s)")
            if census["nan_words"] or census["inf_words"]:
                raise AssertionError(
                    f"{name} step {step}: the reference state carries "
                    f"{census} non-finite words; platform fact (f) forbids "
                    f"reading raw words through a NaN")
            if not point["bit_identical"] and first_divergence is None:
                first_divergence = point
                if mutation is None:
                    # BYTE DIVERGENCE: first case/step/array, STOP, report.
                    break
            if mutation is not None and first_divergence is not None:
                break

        row["launches"] = {slot: counter.launches
                           for slot, counter in counters.items()}
        row["first_divergence"] = first_divergence
        row["bit_identical"] = first_divergence is None
        row["source_effect_seen"] = source_effect_seen

        # THE PER-SEEDING VACUITY FLOOR, after the run.
        if case["seeding"] == "signed_zero_lattice":
            axis = row["folded_axes"][0]
            row["negative_zero_census_after"] = negative_zero_census(xp, reference)
            row["fold_plane_max_abs_E_after"] = fold_plane_max_abs_e(
                xp, reference, axis)
            if row["negative_zero_census_after"] == 0:
                raise AssertionError(
                    f"{name}: VACUOUS — the +-0 lattice did not survive into "
                    f"the state the sub-step reads (census 0 after the run)")
            if not row["fold_plane_max_abs_E_after"]:
                raise AssertionError(
                    f"{name}: VACUOUS — the fold plane stayed at zero, so the "
                    f"mirror ghost never reached the plane this case watches")

        expected = steps if first_divergence is None else first_divergence["step"]
        row["expected_launches_per_slot"] = expected
        row["launch_counts_match_budget"] = all(
            count == expected for count in row["launches"].values())
        if not row["launch_counts_match_budget"]:
            raise AssertionError(
                f"{name}: a slot's launch count does not match the steps taken "
                f"({row['launches']} vs {expected}): a bit-identical row "
                f"without launch evidence certifies nothing")
        if tolerate_divergence:
            # THE ATTRIBUTION LEG. Divergence is the measurement here, not a
            # failure: the point is WHICH slot set first produces one.
            return row
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
            if first_divergence["step"] != 1:
                raise AssertionError(
                    f"{name}: the armed mutation was caught at step "
                    f"{first_divergence['step']}, not step 1; the coupling and "
                    f"the ghost are both live at the FIRST update_E, so a later "
                    f"catch means the needle is not the one this leg declares")
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
# ATTRIBUTION — which slot set produces the divergence
# ---------------------------------------------------------------------------
#
# A composition probe that simply EXCLUDED the awkward configuration would be
# choosing its own evidence. This leg keeps it and answers the only question
# that matters about it: is the divergence THIS FAMILY'S, or the install
# convention's? Four installs on the same case, two steps each:
#
#   all_six_slots            everything                       -> expected DIVERGE
#   certified_five_slots     the fills and curls, update_E on
#                            the ARRAY PATH — this family's
#                            kernel is not in the run at all   -> expected DIVERGE
#   four_core_slots          curls + constitutives + update_E,
#                            fills on the array path           -> expected IDENTICAL
#   update_E_only            this family's kernel alone        -> expected IDENTICAL
#
# The second row is the discriminating one. If a run with NO folded off-diagonal
# kernel in it diverges the same way, the finding cannot be about that kernel.

ATTRIBUTION_SUBSETS: Tuple[Tuple[str, Tuple[str, ...], bool], ...] = (
    ("all_six_slots", SLOTS, False),
    ("certified_five_slots_array_path_update_E",
     ("step_B", "fill_B", "update_H", "step_D", "fill_D"), False),
    ("four_core_slots_array_path_fills", CORE_SLOTS, True),
    ("update_E_only", ("update_E",), True),
)

ATTRIBUTION_STEPS = 2


def run_attribution(xp, case: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for label, slots, expect_identical in ATTRIBUTION_SUBSETS:
        row = run_case(xp, case, slots=slots, steps_override=ATTRIBUTION_STEPS,
                       tolerate_divergence=True)
        rows.append({
            "install": label, "slots": list(slots),
            "expect_identical": expect_identical,
            "bit_identical": row.get("bit_identical"),
            "first_divergence": row.get("first_divergence"),
            "launches": row.get("launches"),
            "agrees": bool(row.get("bit_identical")) == expect_identical})
        log(f"[attribution] {case['name']} {label}: identical="
            f"{row.get('bit_identical')} (expected {expect_identical})")
    verdict = {
        "case": case["name"], "steps": ATTRIBUTION_STEPS, "rows": rows,
        "attributed": all(row["agrees"] for row in rows),
        "why": ("the divergence follows the GHOST-FILL INSTALL, not this "
                "family's kernel: a run with no folded off-diagonal kernel in "
                "it at all diverges the same way, and every install that "
                "carries that kernel WITHOUT the combined fill is "
                "byte-identical"),
        "belongs_to": ("symmetry.MirrorGhostFillPlan and "
                       "the composition install convention; recorded here "
                       "because this probe's case list is the first to carry a "
                       "folded PERIODIC EVEN axis beside a metallic one"),
    }
    return verdict


# ---------------------------------------------------------------------------
# Preflight — the whole enumeration on NumPy drivers, no launches
# ---------------------------------------------------------------------------

def run_preflight(out_path: str) -> int:
    """Build every case on the ARRAY path and assert the seams with the
    array-module clause stripped. No launch, no byte claim: this exists so a
    broken enumeration is found on a laptop rather than on a device slot."""
    payload: Dict[str, Any] = {
        "probe": "triton_folded_offdiag_composition",
        "mode": "preflight (NumPy drivers; NO byte-identity claim)",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
            record["stored_cells"] = [int(driver.grid.stored_cells(a))
                                      for a in range(3)]
            record["stored_count_parity"] = [
                "even" if int(driver.grid.stored_cells(a)) % 2 == 0 else "odd"
                for a in range(3)]
            record["folded_axes"] = folded_axes(driver)
            record["ghost_weights"] = list(
                fomod.mirror_ghost_weights(driver.grid))
            seeding = seed_state(driver, np, case)
            record["seeding_record"] = seeding
            record["negative_zero_census_at_seed"] = negative_zero_census(
                np, driver)
            record["seams"] = seam_report(driver.fields, driver.pml,
                                          drop_backend=True)
            assert_seams(record["seams"], case["name"])
            sound, why_not = ghost_fill_install_is_sound(driver)
            record["ghost_fills_installable"] = sound
            record["ghost_fill_install_note"] = why_not
            record["installed_slots"] = list(SLOTS if sound else CORE_SLOTS)
            if case["seeding"] == "signed_zero_lattice" and \
                    record["negative_zero_census_at_seed"] == 0:
                raise AssertionError("VACUOUS seeding: census 0 at seed")
            record["ok"] = True
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
    parser.add_argument("--dry-run", action="store_true",
                        help="NumPy preflight: enumeration and seams only")
    args = parser.parse_args(argv)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if args.dry_run:
        return run_preflight(out_path)

    if cp is None:
        log("[composition] cupy is not importable; NO byte-identity claim")
        return 75

    import gate_triton_complex as gate                           # noqa: PLC0415
    from meep_gpu import backends                                # noqa: PLC0415

    # Strip FIRST, guard second: the guard wraps whatever sits at the seam.
    gate.install_ftz_strip()
    backends.guard_kernel_compilation(cp)

    payload: Dict[str, Any] = {
        "probe": "triton_folded_offdiag_composition",
        "module": "meep_gpu/triton_kernels/folded_offdiag_update_e.py",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": environment(),
        "dispatch": "DISABLED (the engine's fast-path hook returns None)",
        "installed_slots": list(SLOTS),
        "subnormal_policy_at_start": gate.policy_stamp("cupy"),
        "timing_claim": ("NONE. Correctness only; this box carries foreign "
                         "load and nothing here is timed"),
        "cases": [], "mutations": [], "attribution": [],
    }
    save(payload, out_path)
    payload["provenance"] = write_provenance(os.path.dirname(out_path))
    save(payload, out_path)

    failures: List[str] = []
    for case in CASES:
        log(f"[composition] starting {case['name']}: {case['why']}")
        try:
            payload["cases"].append(run_case(cp, case))
        except Exception as exc:                    # noqa: BLE001
            payload["cases"].append({"case": case["name"], "ok": False,
                                     "error": f"{type(exc).__name__}: {exc}"})
            failures.append(case["name"])
            log(f"[composition] {case['name']} FAILED: {exc}")
        save(payload, out_path)
        if failures:
            break                                   # divergence: STOP, report

    # THE ATTRIBUTION LEG runs on every case whose fills could not be installed
    # — the configurations that produced the finding. It is evidence, not a
    # footnote, and a run that could not attribute the divergence FAILS.
    if not failures:
        for case in CASES:
            row = next((r for r in payload["cases"]
                        if r.get("base_case") == case["name"]), None)
            if row is None or row.get("ghost_fills_installable", True):
                continue
            log(f"[composition] attributing the ghost-fill finding on "
                f"{case['name']}")
            try:
                verdict = run_attribution(cp, case)
            except Exception as exc:                # noqa: BLE001
                verdict = {"case": case["name"], "attributed": False,
                           "error": f"{type(exc).__name__}: {exc}"}
            payload["attribution"].append(verdict)
            if not verdict.get("attributed"):
                failures.append(f"attribution::{case['name']}")
            save(payload, out_path)

    if not failures:
        mutation_case = CASES[0]
        for mutation in MUTATIONS:
            log(f"[composition] arming {mutation[0]} on "
                f"{mutation_case['name']}")
            try:
                payload["mutations"].append(
                    run_case(cp, mutation_case, mutation=mutation))
            except Exception as exc:                # noqa: BLE001
                payload["mutations"].append(
                    {"case": f"{mutation_case['name']}::{mutation[0]}",
                     "ok": False, "error": f"{type(exc).__name__}: {exc}"})
                failures.append(mutation[0])
                log(f"[composition] mutation {mutation[0]} FAILED: {exc}")
            save(payload, out_path)

    residual = gate.ftz_strip_license_reasons()
    payload["subnormal_policy"] = gate.policy_stamp("cupy")
    payload["subnormal_policy_residual"] = residual
    payload["failures"] = failures
    complete_steps = sum(len(row.get("per_step", []))
                         for row in payload["cases"])
    payload["summary"] = {
        "cases": len(payload["cases"]),
        "cases_exact": sum(1 for row in payload["cases"]
                           if row.get("bit_identical")),
        "complete_steps_compared": complete_steps,
        "armed_mutations": len(payload["mutations"]),
        "armed_mutations_caught": sum(1 for row in payload["mutations"]
                                      if row.get("caught")),
        "fold_terminations": sorted({
            "metallic" if "metallic" in row.get("boundaries", []) else "periodic"
            for row in payload["cases"] if "boundaries" in row}),
        "mirror_phases": sorted({phase for row in payload["cases"]
                                 for _axis, phase in row.get("mirrors", [])}),
        "stored_count_parities": sorted({
            parity for row in payload["cases"]
            for axis, parity in zip(row.get("folded_axes", []),
                                    [row.get("stored_count_parity", [])[a]
                                     for a in row.get("folded_axes", [])])
            for parity in (parity,)}),
        "fold_counts": sorted({len(row.get("folded_axes", []))
                               for row in payload["cases"]}),
        "six_slot_cases": sum(1 for row in payload["cases"]
                              if row.get("ghost_fills_installable")),
        "four_slot_cases": sum(1 for row in payload["cases"]
                               if row.get("ghost_fills_installable") is False),
        "attributions": len(payload["attribution"]),
        "attributions_resolved": sum(1 for row in payload["attribution"]
                                     if row.get("attributed")),
    }
    passed = (not failures and not residual
              and payload["summary"]["cases_exact"] == len(CASES)
              and payload["summary"]["armed_mutations_caught"] == len(MUTATIONS)
              and (payload["summary"]["attributions"]
                   == payload["summary"]["attributions_resolved"]))
    payload["passed"] = bool(passed)
    payload["status"] = "passed" if passed else f"FAILED: {failures or residual}"
    payload["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(payload, out_path)
    log(f"[composition] {payload['status']}: {payload['summary']}")
    return 0 if passed else 1


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(main())
