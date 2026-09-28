"""Byte-identity gate for the FIRST fused hand-CUDA product: ``step_B`` -> ``update_H``.

WHAT IS UNDER TEST, AND WHY IT CANNOT BE MEASURED PER SUB-STEP.
``meep_gpu/cuda_kernels/fused_magnetic_pair.py`` ships ONE kernel that performs
THREE driver passes in one launch::

    step_B (driver.py:3291) -> zero_metal_B (:3295) -> update_H (:3298)

The claim is about the SEAM between them -- the flux density never leaving a
register -- so a per-sub-step comparison could not see it at all. The comparison
here is therefore per COMPLETE DRIVER STEP, over a stated launch budget, against
``stepping``'s own eleven-pass sequence for the same configuration, with the D/E
half running on the array path on BOTH sides so a corrupted B reaches E on the
very next step. The first divergent step is reported rather than a final
pass/fail.

=============================================================================
THE NINE THINGS THIS GATE REFUSES TO LET PASS SILENTLY
=============================================================================

1. **A step this gate invented.** ``leg_driver_order`` reads ``FdtdDriver.step``'s
   own source and extracts the ordered list of ``stepping`` calls it makes. The
   oracle walks THAT list. A gate carrying its own private model of what a
   timestep is would certify the kernel against the model.

2. **A silent fallback.** Bytes alone cannot prove the fused kernel ran: an
   engine that never launched it is byte-identical to the oracle BY
   CONSTRUCTION, because the oracle is the array path the same engine would
   otherwise take. So every case asserts EXACT launch counts from TWO
   INDEPENDENT COUNTERS -- a proxy installed over the shipped compile memo
   (``compile_cache._compiled_kernels``, the seam every ``_get_kernel`` route
   passes through) and the launcher's own returned report -- and requires them
   to agree, to equal the step budget, and to name ONLY the fused kernel. A run
   that launched ``step_B_pml_real`` beside the fused kernel is a run where the
   fusion added a launch instead of removing two, and the counter says so.

3. **A vacuous comparison.** Two floors, per volume. WHAT MUST MOVE: on the
   uniform class every one of the twenty-four stored volumes must differ from
   its seeded value, or the comparison is a no-op agreeing with a no-op. WHAT
   MUST NOT MOVE: the nineteen read-only volumes -- eighteen PML coefficient
   vectors and ``inv_eps`` -- must be bit-unchanged from before the run, on the
   device side. Every argument in this signature is ``__restrict__``, and a
   kernel that wrote through a ``const`` binding is UB that NVRTC does not
   diagnose.

   THE BAND CLASS ANSWERS TO A DIFFERENT MOVEMENT FLOOR, AND IT IS DECLARED
   HERE RATHER THAN DISCOVERED AFTERWARDS. Under ``flush`` a state seeded
   entirely in the subnormal band drives ``constitutive_apply`` to
   ``f[idx] = (f[idx] + 0) - 0``, which is the identity on H: ``kps * src`` and
   ``kms * prev`` both flush. So the band class is held to the three families
   the fused kernel provably writes regardless (:data:`BAND_MUST_MOVE`) and the
   rest is RECORDED. See :data:`MOVEMENT_FLOOR_NOTE`.

4. **An unexercised policy.** ``operand_census`` counts the subnormals and
   signed zeros the operands REALLY hold, off the bits. The band class must
   contain subnormals and the uniform class must contain none -- a precondition
   that fires everywhere discriminates nothing, and one that never fires is a
   detector never shown to work. Both readings are required, in
   ``summarize``. The reference state's census is additionally taken after
   EVERY step, so "the band survived into the arithmetic" is a measurement.

5. **A comparison that could not fail.** ``leg_guard_control`` compiles the
   SHIPPED source with the contraction guard REMOVED and requires it to
   DIVERGE. Both halves reach bit-identity only under ``--fmad=false`` (the
   certified pair measured 120/120 with it and 0/120 without), and both
   ``(fprev * kms) - curl`` and ``(f[idx] * kms_u) + fu_new`` are FMA
   candidates the array path rounds twice. A guard control that came back
   identical would mean this comparison cannot see a rounding change at all,
   and it is scored, not merely reported.

6. **A hollow pass.** SIXTEEN source defects and FOUR host defects are armed and
   MUST be caught; THREE source defects and ONE host defect are declared NULL in
   advance, with the arithmetic reason, and must come back UNCAUGHT. Leg
   ``disarm`` reruns the identical harness with the shipped bytes and requires
   zero divergence, so a "caught" cannot be a harness that diverges anyway. A
   needle that matches nothing is reported NOT ARMED rather than passing
   silently, and a mutation is scored only on the specs it is LIVE on -- a wall
   defect scored on an all-periodic row would report UNCAUGHT for a reason about
   the case table.

7. **A seam that is not really a seam.** ``seam_reloads_from_global`` replaces
   the three register reads with reloads of the very words the curl half (and
   the wall clear) just stored. That mutant must NOT diverge -- it is the same
   float32 word by construction -- which is what makes "the fusion removes a
   round trip and changes no arithmetic" a measurement rather than a claim. It
   is the single most important leg in this file after the product legs.

8. **A fusion that is only a name.** ``leg_separate_control`` steps THREE
   engines from one seed: the array path, the two ALREADY CERTIFIED sub-step
   products with the certified in-seam wall pass between them as three separate
   launches, and the fused product. All three must agree word for word at every
   complete step, and both launch counts are recorded -- which is the only place
   the difference between the compositions shows at all, since a correct fusion
   is byte-neutral by construction. 3 launches per step become 1 on a walled
   grid; 2 become 1 unwalled.

9. **A refusal that is really an omission.** ``leg_refusal`` measures the clauses
   this predicate ADDS over its two halves: an undeclared source list refused
   (ignorance is not an empty set); a real MAGNETIC ``VolumeSource`` now ADMITTED
   *and* the two repair plans actually installed by the shipped composer; a
   magnetic source that publishes no deposit index still refused BY NAME; a real
   ELECTRIC one NOT refused (it is injected in the D/E half); and a FOLDED grid
   refused naming BOTH uncarried fills without mentioning the deposit.

10. **An admission nothing measures.** ``run_deposit_case`` and
   ``leg_deposit_null_control`` are what the flipped
   ``CARRIES_DEPOSIT_REPAIR`` is worth. Complete driver steps with a real
   magnetic source in the seam -- a point deposit and an extended one, on a
   walled and an unwalled grid -- with the fused launch consuming an UNINJECTED
   field, the driver's own inject/fill/clear between the two consults, and the
   repair recomputing the deposit points at the second. Byte compared per step.
   The null control asks the shipped composer for the SAME plan with an empty
   source list, injects anyway, and MUST diverge: without it a green row would
   be consistent with a deposit too small to see.

=============================================================================
WHAT THIS GATE DOES NOT CLAIM
=============================================================================

No throughput claim of any kind. No dispatch claim: nothing in ``meep_gpu``
imports ``cuda_kernels``, and a verdict here does not wire this product to
anything -- ``fused_pairs`` is a COMPOSER, and composing is not dispatching.
No claim about a folded, complex, cylindrical, conductive, BFAST or
special-kz run -- every one is refused BY NAME and none was swept. A
MAGNETICALLY-DRIVEN run is no longer in that list: it is carried, and legs 9-10
are what carry it. And no verdict about the D/E seam, which is a different
product with the opposite sub-lattice pairing and no CUDA product at all.

=============================================================================
RUNNING IT
=============================================================================

ONE verified-empty GPU, pinned by UUID. ``CUPY_ACCELERATORS`` must be EMPTY IN
THE ENVIRONMENT before CuPy imports, and the CuPy disk cache must be private per
policy because CuPy's cache key is computed ABOVE the strip seam::

    CUDA_VISIBLE_DEVICES=<uuid> CUPY_ACCELERATORS= \\
      CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
      python -u gate_cuda_fused_magnetic_pair.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

    CUDA_VISIBLE_DEVICES=<uuid> CUPY_ACCELERATORS= \\
      CUPY_CACHE_DIR=$OUT/cupy_cache/native \\
      python -u gate_cuda_fused_magnetic_pair.py \\
        --subnormal-policy flush --import-meep-for-host-policy \\
        --out $OUT/flush/gate.json

``--out`` is a FILE. One flushed line per case (the progress-reporting rule); the artifact
is rewritten atomically after every case, so an interrupted run keeps everything
up to the failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_API = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_API not in sys.path:
    sys.path.insert(0, _REPO_API)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import cupy as cp
except ImportError:  # laptop: only the source legs can run
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import deposit_repair, stepping  # noqa: E402

log = probe.log
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

#: Per-case RNG seed base. Seeded from a sha256 of the case's OWN label, never
#: from ``hash()``: ``PYTHONHASHSEED`` salts the hash of a string, so a
#: hash-seeded case could not be replayed from the record that names it.
SEED = 20260827

#: Complete DRIVER STEPS every product leg runs. Sixty, the budget every
#: hand-CUDA record on this track is cut at -- "identical for N steps" is a claim
#: about N. The budget matters more here than for a sub-step: ``fu_B`` and
#: ``f_w_H`` are state carried between steps, the D/E half feeds E back into the
#: curl, and a coefficient index off by one axis needs several steps to reach the
#: low bits of the interior.
STEPS = 60

#: Every stored volume a complete step can touch, and every one is compared. The
#: D/E half is in this list even though the fused product does not touch it,
#: because a pair that corrupted B reaches E through ``step_D`` on the very next
#: step and a comparison blind to that would be reporting on half the engine.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: The volumes that must be BIT-UNCHANGED by a run. Eighteen PML coefficient
#: vectors on both Yee sub-lattices, plus the inverse permittivity. Every
#: argument in the fused signature is ``__restrict__`` and six of them are
#: ``const``; a kernel that wrote through one is undefined behaviour NVRTC does
#: not diagnose, and a wrong answer at full speed rather than a launch failure.
IMMUTABLE_PML: Tuple[str, ...] = tuple(
    f"{name}_{axis}{suffix}"
    for axis in "xyz" for name in ("kms", "sinv", "kps") for suffix in ("", "_h"))

#: The families the SUBNORMAL BAND class is held to. See
#: :data:`MOVEMENT_FLOOR_NOTE` -- declared before the first run, not after it.
BAND_MUST_MOVE: Tuple[str, ...] = tuple(
    [f"B{axis}" for axis in "xyz"] + [f"fu_B{axis}" for axis in "xyz"]
    + [f"f_w_H{axis}" for axis in "xyz"])

MOVEMENT_FLOOR_NOTE = (
    "The uniform class is held to the full movement floor: all twenty-four "
    "stored volumes must differ from their seeded values. The subnormal-band "
    "class is held to BAND_MUST_MOVE only, and the reason is arithmetic and was "
    "written down before the first run: under 'flush' a state seeded entirely "
    "in the band drives constitutive_apply to f[idx] = (f[idx] + 0) - 0, since "
    "kps*src and kms*prev both flush, so H is the identity and need not move. "
    "B, fu_B and f_w_H are written unconditionally by the curl half and by the "
    "constitutive store, so they must move under either policy. Arrays that did "
    "not move are RECORDED for both classes either way.")

#: The two operand classes this track sweeps, and the only two. They are the
#: curl's and the constitutive's own (``probe.PML_VALUE_CLASSES``,
#: ``CONSTITUTIVE_VALUE_CLASSES``), so this product's verdict is commensurable
#: with the verdicts on the halves it welds.
VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: NVRTC options the shipped module compiles with, restated so the guard control
#: can drop them. NOT imported from the module: a control that read the module's
#: own tuple would compile the same thing twice if that tuple were ever emptied.
GUARD_OPTIONS: Tuple[str, ...] = ("--fmad=false",)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------
#
# THE EXTENTS ARE DELIBERATELY UNEQUAL (9, 10, 11). A cube lets an index
# decomposition swap i and k and stay bit-identical, and this kernel decomposes
# a flat thread index into i/j/k and indexes THREE coefficient vectors of three
# different lengths with them.
#
# EVERY AXIS IS WALLED SOMEWHERE and one row is walled NOWHERE. The wall table
# this product carries is the B DIAGONAL -- Bx on x, By on y, Bz on z -- so a row
# that walled only one axis cannot tell the diagonal from the D-side
# off-diagonal complement, and the all-periodic row proves the family is not
# wall-dependent.
#
# NO ROW IS FOLDED, COMPLEX, CONDUCTIVE OR SOURCE-DRIVEN. Every one of those is
# refused BY NAME by the predicate, and leg_refusal measures the refusals rather
# than the sweep pretending to cover them.

CELL: Tuple[float, float, float] = (9.0, 10.0, 11.0)

SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "all_periodic", "boundaries": ("periodic", "periodic", "periodic"),
     "pml": 2, "cell": CELL},
    {"label": "wall_x", "boundaries": ("metallic", "periodic", "periodic"),
     "pml": 2, "cell": CELL},
    {"label": "wall_y", "boundaries": ("periodic", "metallic", "periodic"),
     "pml": 2, "cell": CELL},
    {"label": "wall_z", "boundaries": ("periodic", "periodic", "metallic"),
     "pml": 2, "cell": CELL},
    {"label": "wall_xz", "boundaries": ("metallic", "periodic", "metallic"),
     "pml": 2, "cell": CELL},
    {"label": "wall_xyz", "boundaries": ("metallic", "metallic", "metallic"),
     "pml": 2, "cell": CELL},
    # A THINNER LAYER on a different shape: the absorber profile changes, so a
    # coefficient index that happened to land inside a flat region of a 2-cell
    # layer has a different table to be wrong about.
    {"label": "wall_xy_thin_pml",
     "boundaries": ("metallic", "metallic", "periodic"),
     "pml": 1, "cell": (12.0, 9.0, 10.0)},

    # ----- THE FOLDED ROWS, NEW ON 2026-08-28 -------------------------------
    #
    # Every one of these was REFUSED BY NAME until this round, and each carries a
    # property no other row has. They are the whole point of the fill carry: on an
    # unfolded grid neither fill runs at all, so a sweep of the rows above would
    # certify the ownership inversion without ever executing one line of it.
    #
    # A folded PERIODIC axis runs BOTH fills; a folded METALLIC one runs only the
    # NEAR fill and keeps its unstored zero ghost. A mutation caught on one is not
    # caught on the other.
    {"label": "fold_y_periodic", "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "pml": 2, "cell": CELL},
    # The ODD full count, where _far_reflect_rows is `stored - 3` rather than
    # `stored - 2`. A sweep carrying only the even count cannot see a baked n - 2.
    {"label": "fold_y_periodic_odd", "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "pml": 2, "cell": (9.0, 11.0, 11.0)},
    # The ODD PARITY plane: mirror_parity is `phase * (1 - 2*iyee)`, so a kernel
    # that baked +1 for the near fill and -1 for the far one is exact at phase +1
    # and wrong everywhere here.
    {"label": "fold_y_periodic_odd_phase",
     "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("Y", -1),), "pml": 2, "cell": CELL},
    # The other TERMINATION: no far fill at all, and the curl takes BC_METALLIC.
    {"label": "fold_y_metallic", "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "pml": 2, "cell": CELL},
    # A FOLD AND A WALL AT ONCE, on different axes -- the one row where the far
    # image of a CLEARED plane is exercised, which is where the driver's order
    # between zero_metal_B (:3295) and fill_folded_far_ghosts_B (:3296) shows and
    # where the carried register must be read AFTER the clear.
    {"label": "fold_y_wall_z", "boundaries": ("periodic", "periodic", "metallic"),
     "symmetry": (("Y", 1),), "pml": 2, "cell": CELL},
    # TWO FOLDED AXES AT MIXED PHASES. The only shape in which the composition
    # question exists at all: a corner at the top of both far planes carries the
    # PRODUCT of two parities, and a cell that is a near destination on one axis
    # and a far one on another carries the product of theirs.
    {"label": "fold_xy_mixed_phase",
     "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("X", 1), ("Y", -1)), "pml": 2, "cell": CELL},
    # THREE FOLDED AXES: one source thread owns SEVEN ghost cells per component,
    # which is the deepest the closed form ever goes.
    {"label": "fold_xyz", "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("X", 1), ("Y", -1), ("Z", 1)), "pml": 2, "cell": CELL},
    # A FOLDED AXIS WHOSE ABSORBER REACHES STORED ROW 2, and it exists for ONE
    # defect. The near ghost sits at stored 0 of the component's OWN axis -- the
    # axis update_H indexes on -- so it must take the DESTINATION's coefficient
    # pair and not the source thread's at stored 2. A folded axis carries no
    # absorber at the mirror plane (the layer is on the HIGH face alone), so on
    # every other folded row here kps[0] and kps[2] are the same word to the bit and
    # a kernel that reused the source's pair would be BYTE-IDENTICAL. The layer has
    # to be deep enough to reach stored cell 2 from the far side before the
    # distinction is observable at all -- which is the same measured finding the
    # Metal folded pair records, and :func:`_near_ghost_pair_is_observable` reads
    # the coefficient vectors rather than trusting this comment.
    {"label": "fold_x_deep_pml", "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("X", 1),), "pml": ((0, 5), (2, 2), (2, 2)), "cell": CELL},

    # ----- THE NONLINEAR ROWS, NEW ON 2026-09-01 ----------------------------
    #
    # The board's ``B_to_H (cuda_curl/PML, cuda_nonlinear/nonlinear)`` cell: 2
    # seam-instances (``3rd-harm-1d.py``, ``Test3rdHarm1d.test_3rd_harm_1d``),
    # both with ``zero_metal_B`` live in the seam. The chi lives entirely inside
    # ``update_E`` (stepping.py:26-28), so the fused kernel's bytes are
    # unchanged; what these rows measure is the widened predicate, the
    # (PML, nonlinear) absorb wiring, and the weld inside a step whose D/E half
    # runs MEEP's Pade branch on both engines. One walled row for the cell the
    # corpus drives, one unwalled so the admission is shown not to be
    # wall-dependent.
    {"label": "wall_z_nonlinear",
     "boundaries": ("periodic", "periodic", "metallic"),
     "pml": 2, "cell": CELL, "nonlinear": True},
    {"label": "all_periodic_nonlinear",
     "boundaries": ("periodic", "periodic", "periodic"),
     "pml": 2, "cell": CELL, "nonlinear": True},
)

#: The row the mutation legs are ARMED on, plus the two they are additionally
#: SCORED on where live. ``wall_xyz`` emits every line the shipped kernel can
#: emit on this predicate: all three wall-clear rows and all three metallic curl
#: masks. ``wall_x`` is carried so a wall defect is caught by a WRONG PLANE
#: rather than only by the difference between three walls and none, and
#: ``all_periodic`` so the wall-free arithmetic is scored too.
#: Three folded rows join on 2026-08-28: a single folded PERIODIC axis (both fills
#: live, one destination plane per component), the mixed-phase two-axis fold (the
#: composition), and the fold-plus-wall row (the far image of a cleared plane). A
#: fill defect scored only on unfolded rows would report UNCAUGHT for a reason
#: about the case table.
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("wall_xyz", "wall_x", "all_periodic",
                                         "fold_y_periodic", "fold_y_periodic_odd",
                                         "fold_y_metallic", "fold_xy_mixed_phase",
                                         "fold_y_wall_z", "fold_x_deep_pml")

#: The specs the separate-composition control runs on: one walled (3 launches
#: per step become 1) and one not (2 become 1).
#: A folded row joins: there the separate composition is FIVE launches (curl, two
#: near-fill planes... one per folded axis, the wall pass where live, the far fill,
#: constitutive) against the fused one, which is the largest reduction this product
#: makes and the only one that exercises the certified in-seam fill kernels beside it.
SEPARATE_CONTROL_LABELS: Tuple[str, ...] = ("wall_xyz", "all_periodic",
                                            "fold_y_periodic", "fold_xy_mixed_phase")

#: The rows ``--product reduced`` keeps. NOT ``SPECS[:3]``, which is what it was
#: until 2026-08-28 and which is now the wrong smoke test: the three rows it took
#: are all UNFOLDED, so neither mirror fill runs and the whole of this round's
#: carry would go unexercised in the mode a run is smoke-tested in.
REDUCED_LABELS: Tuple[str, ...] = ("wall_xyz", "fold_y_periodic",
                                   "fold_xy_mixed_phase", "wall_z_nonlinear")

#: The specs the DEPOSIT legs run on. BOTH TERMINATIONS, because the one asymmetry
#: between the fused route and the driver's is where a deposit lands on a cell
#: ``zero_metal_B`` clears -- the kernel clears BEFORE the injection and the driver
#: AFTER it -- and a sweep with no wall could not see it at all.
#: ``deposits_on_a_cleared_cell`` counts that overlap per case and reports it.
#: A FOLDED row joins, and it is the one that makes CARRIES_DEPOSIT_REPAIR mean
#: something on this seam: the two fills run AFTER the injection, so a repair that
#: wrote only the deposit INDEX would leave every mirror image of it describing the
#: pre-injection field. ``deposit_repair.repair_cells`` hands save/apply the CLOSURE
#: of those images, and this row is where that is measured rather than inherited.
DEPOSIT_SPEC_LABELS: Tuple[str, ...] = ("wall_xyz", "all_periodic",
                                        "fold_y_periodic")


def case_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


def build(spec: Dict[str, Any], value_class: str, rng):
    """One seeded engine: ``(fields, grid, pml)``. Called two or three times per
    case, identically, from the same label-derived draw.

    ``enable_pml_storage`` runs BEFORE the seeding: it allocates E, H and both
    auxiliary families, and a seed written before the allocation would be
    overwritten by it.

    THE AUXILIARIES START NONZERO in both classes, for the reason the curl and
    constitutive legs give: a zero ``fu`` makes ``fu * kms`` exactly zero on the
    first launch whatever ``kms`` is, which would hide a mis-indexed coefficient
    until step two.
    """
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415

    from meep_gpu.grid import Mirror  # noqa: PLC0415

    grid = Grid(resolution=1.0, cell_size=tuple(spec["cell"]),
                boundaries=tuple(spec["boundaries"]),
                symmetry=tuple(Mirror(axis, int(phase))
                               for axis, phase in spec.get("symmetry", ())),
                xp=cp, courant=0.5)
    fields = Fields(grid=grid)
    fields.enable_pml_storage()
    # A FOLDED AXIS TAKES ITS ABSORBER ON THE HIGH FACE ONLY. The mirror plane is
    # the low face and ``stepping._require_consistent_pml`` admits nothing there;
    # a scalar thickness on a folded axis is a layer over the fold.
    folded = {"XYZ".index(axis) for axis, _phase in spec.get("symmetry", ())}
    if not isinstance(spec["pml"], int):
        thickness = tuple(tuple(int(v) for v in pair) for pair in spec["pml"])
    elif folded:
        thickness = tuple((0, spec["pml"]) if axis in folded
                          else (spec["pml"], spec["pml"]) for axis in range(3))
    else:
        thickness = spec["pml"]
    pml = PML(grid=grid, thickness=thickness)
    shape = tuple(int(n) for n in grid.shape)
    if value_class == "uniform":
        host = {name: rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                for name in STATE_NAMES}
    elif value_class == "subnormal_band":
        host = subnormal_band_hosts(STATE_NAMES, shape, rng)
    else:
        raise ValueError(f"value class {value_class!r} is not one of {VALUE_CLASSES}")
    for name, values in host.items():
        getattr(fields, name)[...] = cp.asarray(np.ascontiguousarray(values))
    if spec.get("nonlinear"):
        # THE NONLINEAR ROWS, 2026-09-01. An instantaneous chi2/chi3 on every E
        # component, drawn as VOLUMES so the array path's per-cell chi lookup is
        # exercised rather than a broadcast scalar (gate_cuda_nonlinear.py's own
        # fixture rule), and drawn AFTER the state so every linear label's
        # sequence is untouched. The chi feeds ``stepping.update_E`` ONLY -- the
        # kernel under test binds neither chi nor epsilon -- so what these rows
        # measure is the widened predicate's admission, the (PML, nonlinear)
        # wiring, and the weld's bit-identity inside a step whose D/E half runs
        # MEEP's Pade branch on both engines. ``leg_nonlinear_widening`` holds
        # the liveness floor that keeps the chi from being decorative.
        shape = tuple(int(n) for n in grid.shape)
        chi2 = {component: cp.asarray(
                    (np.float32(0.02) * rng.uniform(0.4, 1.6, size=shape)
                     ).astype(np.float32)) for component in ("Ex", "Ey", "Ez")}
        chi3 = {component: cp.asarray(
                    (np.float32(0.05) * rng.uniform(0.4, 1.6, size=shape)
                     ).astype(np.float32)) for component in ("Ex", "Ey", "Ez")}
        fields.set_nonlinear_volumes(chi2, chi3)
    return fields, grid, pml, host


def words(array: Any) -> np.ndarray:
    """One array as raw uint32 WORDS. Byte compares, never ``allclose``:
    ``-0.0 == 0.0`` and ``NaN != NaN`` both lie, and the band class puts signed
    zeros in the operands deliberately."""
    return np.ascontiguousarray(to_host(array), dtype=np.float32).ravel().view(np.uint32)


def differing(left: Any, right: Any) -> int:
    a, b = words(left), words(right)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a != b))


def state_of(fields) -> Dict[str, Any]:
    return {name: getattr(fields, name) for name in STATE_NAMES}


def frozen(fields) -> Dict[str, np.ndarray]:
    return {name: words(value).copy() for name, value in state_of(fields).items()}


def compare(left, right) -> Dict[str, int]:
    a, b = state_of(left), state_of(right)
    return {name: n for name in STATE_NAMES if (n := differing(a[name], b[name]))}


def read_only_snapshot(fields, pml) -> Dict[str, np.ndarray]:
    """Every volume the kernel binds ``const``, plus ``inv_eps``."""
    out: Dict[str, np.ndarray] = {}
    for name in IMMUTABLE_PML:
        value = getattr(pml, name, None)
        if value is not None:
            out[f"pml.{name}"] = words(value).copy()
    inv_eps = getattr(fields, "inv_eps", None)
    if inv_eps is not None:
        out["fields.inv_eps"] = words(inv_eps).copy()
    return out


def read_only_drift(before: Dict[str, np.ndarray], fields, pml) -> List[str]:
    after = read_only_snapshot(fields, pml)
    return sorted(name for name in before
                  if name not in after
                  or before[name].shape != after[name].shape
                  or not np.array_equal(before[name], after[name]))


# ---------------------------------------------------------------------------
# WHAT A COMPLETE STEP IS -- read off the driver, never modelled here
# ---------------------------------------------------------------------------

#: The eleven ``stepping`` calls ``FdtdDriver.step`` makes, in driver order.
#: ASSERTED against the driver's own source by :func:`leg_driver_order`, not
#: trusted: a gate that carried a private model of a timestep would certify the
#: kernel against the model.
DRIVER_ORDER: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E", "update_P",
)

#: The FIVE the fused kernel replaces, declared by the module itself and asserted
#: equal to ``fused_magnetic_pair.REPLACES`` by :func:`leg_driver_order`. Three
#: until 2026-08-28: the two mirror fills joined when the ownership inversion
#: landed, and a gate that kept the old tuple would run them on the array path
#: BESIDE the kernel that also performs them -- writing each ghost twice and
#: reporting a pass, because the fill is idempotent on its own output only when
#: the parity is +1.
FUSED_PASSES: Tuple[str, ...] = ("step_B", "fill_symmetry_bc_B", "zero_metal_B",
                                 "fill_folded_far_ghosts_B", "update_H")

#: Which passes take the PML layer. Read off ``stepping``'s signatures rather
#: than guessed, so a pass that gained or lost the argument is a TypeError here
#: rather than a silently skipped absorber.
_TAKES_PML = {"step_B", "update_H", "step_D", "update_E", "update_P"}


def run_pass(name: str, fields, pml) -> None:
    function = getattr(stepping, name)
    if name in _TAKES_PML:
        function(fields, pml)
    else:
        function(fields)


def array_step(fields, pml) -> None:
    """One COMPLETE driver step on the array path. The oracle."""
    for name in DRIVER_ORDER:
        run_pass(name, fields, pml)


def leg_driver_order() -> Dict[str, Any]:
    """``DRIVER_ORDER`` must be ``FdtdDriver.step``'s own call sequence.

    Read out of ``driver.py``'s text rather than by importing the driver: the
    question is what the SOURCE says, and a gate that imported the module to ask
    would be asking a different object than the one a reader checks.
    """
    path = os.path.join(_REPO_API, "meep_gpu", "driver.py")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"\n    def step\(self.*?(?=\n    def )", text, re.S)
    if match is None:
        return {"passed": False,
                "why": "FdtdDriver.step could not be located in driver.py"}
    body = match.group(0)
    names = "|".join(sorted(set(DRIVER_ORDER), key=len, reverse=True))
    found = tuple(re.findall(rf"\b({names})\(self\.fields", body))
    return {
        "passed": found == DRIVER_ORDER,
        "driver_file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "driver_call_sequence": list(found),
        "gate_declared_order": list(DRIVER_ORDER),
        "fused_replaces": list(FUSED_PASSES),
    }


# ---------------------------------------------------------------------------
# Launch counting: two independent counters
# ---------------------------------------------------------------------------

class _CountingKernel:
    """A ``cp.RawKernel`` that counts its launches and delegates everything else."""

    __slots__ = ("_kernel", "_counter", "_name")

    def __init__(self, kernel: Any, counter: Dict[str, int], name: str) -> None:
        self._kernel, self._counter, self._name = kernel, counter, name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._counter[self._name] = self._counter.get(self._name, 0) + 1
        self._counter["_total"] = self._counter.get("_total", 0) + 1
        return self._kernel(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._kernel, item)


class MemoLaunchCounter:
    """Wrap every memoized device kernel; restore on exit.

    THE MEMO IS THE RIGHT SEAM and it is the shipped one. Every launcher in this
    package reaches its kernel through ``compile_cache.get_or_compile``, which
    returns ``_compiled_kernels[key]`` on a hit, so replacing the stored object
    counts every launch through every shipped route without editing one byte
    under ``meep_gpu/``. NOTHING IS INSTALLED UNLESS THE MEMO IS ALREADY WARM,
    which is why every case warms it first: a cold memo would have the factory
    store a raw kernel and the count would read zero for a kernel that ran.
    """

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self._saved: Dict[Any, Any] = {}

    def __enter__(self) -> "MemoLaunchCounter":
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
        store = compile_cache._compiled_kernels  # noqa: SLF001 - the shipped memo
        self._saved = dict(store)
        for key, kernel in list(store.items()):
            store[key] = _CountingKernel(kernel, self.counts, str(key[0]))
        return self

    def __exit__(self, *exc: Any) -> None:
        from meep_gpu.cuda_kernels import compile_cache  # noqa: PLC0415
        store = compile_cache._compiled_kernels  # noqa: SLF001
        for key, kernel in list(store.items()):
            if isinstance(kernel, _CountingKernel) and key in self._saved:
                store[key] = self._saved[key]

    @property
    def total(self) -> int:
        return int(self.counts.get("_total", 0))

    def named(self) -> Dict[str, int]:
        return {k: v for k, v in sorted(self.counts.items()) if k != "_total"}


# ---------------------------------------------------------------------------
# One product case
# ---------------------------------------------------------------------------

def _warm_memo(specs: Sequence[Dict[str, Any]]) -> None:
    """Compile and launch every kernel this file drives, once, off the counted path.

    The memo must be warm before :class:`MemoLaunchCounter` installs. IT TAKES THE
    WHOLE SPEC LIST since 2026-08-28: this kernel's source is one string for every
    fold orientation, so a single warm-up still compiles it once -- but the
    CERTIFIED in-seam fill kernels the separate-composition control launches are
    only reached on a folded row, and warming on an unfolded spec alone would leave
    their first compile inside the counted region.

    A REFUSED WARM-UP IS FATAL RATHER THAN SILENT: a spec the predicate does not
    admit would otherwise reach the product sweep and fail there with a reason
    about coverage, one leg later than the table that was wrong.
    """
    from meep_gpu.cuda_kernels import (constitutive_kernels, fused_magnetic_pair,  # noqa: PLC0415
                                       in_seam_coverage, in_seam_passes,
                                       step_curl_kernels)
    for warm in specs:
        fields, grid, pml, _ = build(warm, "uniform", np.random.default_rng(1))
        dtdx = float(grid.dt / grid.dx)
        report = fused_magnetic_pair.run_fused_magnetic_pair(
            fields, grid, pml, dtdx, sources=())
        if not report.get("launched"):
            raise SystemExit(f"the memo warm-up could not launch on {warm['label']}: "
                             f"{report.get('reason')}")
        step_curl_kernels._step_B_fused_pml_real(  # noqa: SLF001
            fields, step_curl_kernels.real_pml_curl_tables(pml, True),
            step_curl_kernels.real_curl_boundary_codes(grid), dtdx)
        # ALL THREE IN-SEAM PASSES, because the separate-composition control
        # launches all three and a cold memo there would be counted as a launch of
        # this kernel by the proxy that watches the shared compile cache.
        if any(in_seam_coverage.zero_metal_axes(grid)):
            in_seam_passes.run_pass("zero_metal", fields, "B", grid=grid)
        if in_seam_coverage.plan("fill_symmetry", grid):
            in_seam_passes.run_pass("fill_symmetry", fields, "B", grid=grid)
        if in_seam_coverage.plan("fill_folded_far", grid):
            in_seam_passes.run_pass("fill_folded_far", fields, "B", grid=grid)
        constitutive_kernels.update_fused_pml_real("H", fields, pml=pml)
    cp.cuda.runtime.deviceSynchronize()


def run_case(spec: Dict[str, Any], value_class: str, steps: int,
             kernel: Optional[Any] = None,
             tables_patch: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None,
             codes_patch: Optional[Callable[[Sequence[Any]], Sequence[Any]]] = None,
             walls_patch: Optional[Callable[[Sequence[int], Any], Sequence[int]]] = None,
             fills_patch: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None,
             label_suffix: str = "") -> Dict[str, Any]:
    """Step two engines side by side and compare per COMPLETE DRIVER STEP.

    ``kernel`` and the four ``*_patch`` hooks are THE GATE'S DOORS -- the
    shipped launcher takes a kernel, a table set and the three grid readings
    precisely so a gate can arm a defect. ``fills_patch`` is the newest and is what
    lets a HOST mutation confuse the two mirror fills' plan with the wall plan or
    the boundary codes, which are three different readings of one grid. A harness that could not hand in its own kernel could not arm a
    single mutation and every mutation leg would launch the shipped kernel and
    report the defect as uncaught.
    """
    from meep_gpu.cuda_kernels import fused_magnetic_pair as family  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage, step_curl_kernels  # noqa: PLC0415

    started = time.time()
    label = f"{spec['label']}|{value_class}{label_suffix}"
    case: Dict[str, Any] = {"label": spec["label"], "value_class": value_class,
                            "case_seed_label": label, "steps_requested": steps,
                            "boundaries": list(spec["boundaries"]),
                            "pml_cells": spec["pml"]}

    reference, ref_grid, ref_pml, host = build(spec, value_class, case_rng(label))
    actual, grid, pml, _ = build(spec, value_class, case_rng(label))
    case["shape"] = [int(n) for n in grid.shape]
    case["dtdx"] = float(grid.dt / grid.dx)
    case["operand_census"] = operand_census(host)

    drift = compare(reference, actual)
    if drift:
        case["passed"] = False
        case["why"] = f"the two builds are not identical: {drift}"
        return case

    covered, reason = family.covers_fused_magnetic_pair(actual, pml, grid, ())
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    if not covered:
        case["passed"] = False
        case["why"] = f"the predicate refused this fixture: {reason}"
        return case

    tables = family.fused_magnetic_pair_tables(pml)
    case["distinct_allocations_checked"] = family.assert_disjoint_bindings(actual, tables)
    if tables_patch is not None:
        tables = tables_patch(tables, pml)
    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    walls = in_seam_coverage.zero_metal_axes(grid)
    fills = family.fused_magnetic_pair_fills(grid)
    case["boundary_codes"] = [int(c) for c in codes]
    case["zero_metal_axes"] = [bool(w) for w in walls]
    case["fill_plan"] = {key: list(value) for key, value in fills.items()}
    if codes_patch is not None:
        codes = codes_patch(codes)
    if walls_patch is not None:
        walls = walls_patch(walls, grid)
    if fills_patch is not None:
        fills = fills_patch(fills, grid)
    case["boundary_codes_used"] = [int(c) for c in codes]
    case["zero_metal_axes_used"] = [bool(w) for w in walls]
    case["fill_plan_used"] = {key: list(value) for key, value in fills.items()}

    before = frozen(actual)
    read_only = read_only_snapshot(actual, pml)
    launched = 0
    counter = MemoLaunchCounter()
    per_step: List[Dict[str, Any]] = []
    with counter:
        for step in range(1, steps + 1):
            array_step(reference, ref_pml)
            report = family.launch_fused_magnetic_pair(
                actual, tables, codes, walls, fills, case["dtdx"], kernel)
            launched += int(bool(report.get("launched")))
            for name in DRIVER_ORDER:
                if name not in FUSED_PASSES:
                    run_pass(name, actual, pml)
            cp.cuda.runtime.deviceSynchronize()
            difference = compare(reference, actual)
            census = operand_census({n: to_host(v)
                                     for n, v in state_of(reference).items()})
            per_step.append({"step": step,
                             "differing_words": sum(difference.values()),
                             "differing_arrays": dict(sorted(difference.items())),
                             "reference_subnormals": census["subnormals"],
                             "reference_negative_zeros": census["negative_zeros"]})
            if difference:
                break

    after = state_of(actual)
    moved = {name: int(np.count_nonzero(before[name] != words(after[name])))
             for name in STATE_NAMES}
    still = sorted(name for name, count in moved.items() if count == 0)
    required = STATE_NAMES if value_class == "uniform" else BAND_MUST_MOVE
    unmoved_required = sorted(name for name in required if moved[name] == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    drifted = read_only_drift(read_only, actual, pml)
    launch_ok = (launched == len(per_step)
                 and counter.named().get(family.KERNEL_NAME, 0)
                 == (len(per_step) if kernel is None else 0))
    case.update({
        "passed": bool(identical and not unmoved_required and not drifted
                       and launch_ok),
        "bit_identical": identical,
        "steps_run": len(per_step),
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"] if per_step else -1,
        "differing_arrays": per_step[-1]["differing_arrays"] if per_step else {},
        "arrays_compared": len(STATE_NAMES),
        "arrays_that_never_moved": still,
        "movement_floor": {"class": value_class, "required": list(required),
                           "unmet": unmoved_required, "note": MOVEMENT_FLOOR_NOTE},
        "read_only_volumes_checked": sorted(read_only),
        "read_only_volumes_that_moved": drifted,
        "launch_counts": {
            "launcher_reports": launched,
            "memo_proxy": counter.named(),
            "memo_proxy_total": counter.total,
            "kernel_supplied_by_gate": kernel is not None,
            "agree": bool(launch_ok),
            "expected_per_step": 1,
        },
        "seconds": time.time() - started,
    })
    return case


# ---------------------------------------------------------------------------
# The separate composition: what the fusion actually removes, measured
# ---------------------------------------------------------------------------

def leg_separate_control(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """THREE engines from one seed: array path, separate certified products, fused.

    THIS IS THE LEG THAT MAKES "FUSION" A MEASUREMENT RATHER THAN A NAME. The
    separate side is not a strawman: it is ``step_curl_kernels``' certified
    ``step_B_pml_real``, ``in_seam_passes``' certified ``fill_symmetry_B``,
    ``zero_metal_B`` and ``fill_folded_far_B``, and ``constitutive_kernels``'
    certified ``update_H_pml_real``, launched exactly as the driver would compose
    them, with the three in-seam passes BETWEEN the halves and in the driver's own
    order. All three engines must agree word for word at every complete step -- a
    correct fusion is byte-neutral by construction, so the launch counts are the
    only place the difference shows at all.

    ON A FOLDED ROW THE SEPARATE SIDE IS THE CERTIFIED FILL KERNELS THEMSELVES, one
    launch per folded axis per fill, which is what makes this leg the strongest
    statement the gate makes about the carry: the fused kernel is compared not to
    the array path alone but to the very device kernels whose arithmetic it lifted.
    """
    from meep_gpu.cuda_kernels import (constitutive_kernels, fused_magnetic_pair as family,  # noqa: PLC0415
                                       in_seam_coverage, in_seam_passes,
                                       step_curl_kernels)

    started = time.time()
    label = f"separate|{spec['label']}"
    reference, _rg, ref_pml, _ = build(spec, "uniform", case_rng(label))
    separate, sep_grid, sep_pml, _ = build(spec, "uniform", case_rng(label))
    fused, fused_grid, fused_pml, _ = build(spec, "uniform", case_rng(label))

    dtdx = float(fused_grid.dt / fused_grid.dx)
    curl_tables = step_curl_kernels.real_pml_curl_tables(sep_pml, True)
    codes = step_curl_kernels.real_curl_boundary_codes(sep_grid)
    walled = bool(any(in_seam_coverage.zero_metal_axes(sep_grid)))
    # ONE LAUNCH PER FOLDED AXIS PER FILL, derived by the same in_seam_coverage.plan
    # the certified launcher uses, so the separate side cannot disagree with the
    # fused one about which axes each pass visits.
    near_plan = in_seam_coverage.plan("fill_symmetry", sep_grid)
    far_plan = in_seam_coverage.plan("fill_folded_far", sep_grid)

    separate_calls = 0
    fused_calls = 0
    separate_counter = MemoLaunchCounter()
    fused_counter = MemoLaunchCounter()
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(reference, ref_pml)

        with separate_counter:
            step_curl_kernels._step_B_fused_pml_real(  # noqa: SLF001
                separate, curl_tables, codes, dtdx)
            separate_calls += 1
            # THE DRIVER'S OWN ORDER (driver.py:3294, :3295, :3296). The near fill
            # runs BEFORE the wall clear and the far fill AFTER it, and on a row
            # that is both folded and walled that order is what decides the ghost.
            if near_plan:
                in_seam_passes.run_pass("fill_symmetry", separate, "B", grid=sep_grid)
                separate_calls += len(near_plan)
            if walled:
                in_seam_passes.run_pass("zero_metal", separate, "B", grid=sep_grid)
                separate_calls += 1
            if far_plan:
                in_seam_passes.run_pass("fill_folded_far", separate, "B",
                                        grid=sep_grid)
                separate_calls += len(far_plan)
            constitutive_kernels.update_fused_pml_real("H", separate, pml=sep_pml)
            separate_calls += 1
        for name in DRIVER_ORDER:
            if name not in FUSED_PASSES:
                run_pass(name, separate, sep_pml)

        with fused_counter:
            report = family.run_fused_magnetic_pair(fused, fused_grid, fused_pml,
                                                    dtdx, sources=())
            fused_calls += int(bool(report.get("launched")))
        for name in DRIVER_ORDER:
            if name not in FUSED_PASSES:
                run_pass(name, fused, fused_pml)

        cp.cuda.runtime.deviceSynchronize()
        row = {"step": step,
               "separate_vs_array": sum(compare(reference, separate).values()),
               "fused_vs_array": sum(compare(reference, fused).values()),
               "fused_vs_separate": sum(compare(separate, fused).values())}
        per_step.append(row)
        if any(value for key, value in row.items() if key != "step"):
            break

    identical = (len(per_step) == steps
                 and all(row["separate_vs_array"] == row["fused_vs_array"]
                         == row["fused_vs_separate"] == 0 for row in per_step))
    expected_separate = 2 + int(walled) + len(near_plan) + len(far_plan)
    counts_ok = (separate_counter.total == separate_calls
                 and fused_counter.total == fused_calls
                 and fused_calls == len(per_step)
                 and separate_calls == expected_separate * len(per_step))
    return {
        "passed": bool(identical and counts_ok and fused_counter.total),
        "label": spec["label"],
        "walled": walled,
        "steps_compared": len(per_step),
        "per_step": per_step,
        "three_way_identical": identical,
        "separate_launches_per_step": expected_separate,
        "fused_launches_per_step": 1,
        "separate_launches_total": separate_counter.total,
        "fused_launches_total": fused_counter.total,
        "separate_host_calls": separate_calls,
        "fused_host_calls": fused_calls,
        "separate_kernels": separate_counter.named(),
        "fused_kernels": fused_counter.named(),
        "launch_counts_agree": counts_ok,
        "launches_removed_per_step": expected_separate - 1,
        "seam_host_passes_for_separate": (
            (["fill_symmetry_bc_B"] if near_plan else [])
            + (["zero_metal_B"] if walled else [])
            + (["fill_folded_far_ghosts_B"] if far_plan else [])),
        "seam_host_passes_for_fused": [],
        "near_fill_launches": len(near_plan),
        "far_fill_launches": len(far_plan),
        "seconds": time.time() - started,
    }


# ---------------------------------------------------------------------------
# The magnetic half-step: the second consult site, never driven on this backend
# ---------------------------------------------------------------------------

def sync_order() -> Tuple[str, ...]:
    """The ``stepping`` passes ``synchronize_magnetic_fields`` runs, off ITS source.

    The same reading :func:`leg_driver_order` takes of ``step``, on the OTHER site
    the driver consults from. A gate that modelled the half-step itself would
    certify the kernel against the model.
    """
    path = os.path.join(_REPO_API, "meep_gpu", "driver.py")
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"\n    def synchronize_magnetic_fields\(self.*?(?=\n    def )",
                      text, re.S)
    assert match is not None, "synchronize_magnetic_fields is not in driver.py"
    names = "|".join(sorted(set(DRIVER_ORDER), key=len, reverse=True))
    return tuple(re.findall(rf"\b({names})\(self\.fields", match.group(0)))


def _sync_backup(fields) -> Dict[str, Any]:
    """The driver's own backup list, from the driver's own class attributes."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    names = FdtdDriver._SYNC_FIELDS + FdtdDriver._SYNC_AUXILIARY
    return {name: getattr(fields, name).copy()
            for name in names if getattr(fields, name, None) is not None}


def _sync_average(fields, backup: Mapping[str, Any]) -> None:
    """``average_with_backup`` — ``_SYNC_FIELDS`` only, exactly as the driver does."""
    from meep_gpu.driver import FdtdDriver  # noqa: PLC0415

    for name in FdtdDriver._SYNC_FIELDS:
        array = getattr(fields, name, None)
        if array is not None and name in backup:
            array *= 0.5
            array += 0.5 * backup[name]


def _sync_restore(fields, backup: Mapping[str, Any]) -> None:
    for name, saved in backup.items():
        array = getattr(fields, name, None)
        if array is not None:
            array[...] = saved


def leg_synchronize(spec: Dict[str, Any], value_class: str, steps: int
                    ) -> Dict[str, Any]:
    """THE SECOND CONSULT SITE. No leg on this backend had ever driven it.

    ``flux_in_box`` and ``field_energy_in_box`` run ``step``'s magnetic half a
    second time and then UNDO it, backing up ``_SYNC_FIELDS`` + ``_SYNC_AUXILIARY``
    and restoring exactly those. This product replaces every one of the five passes
    that half-step runs, so at that site it is ONE launch and no host pass at all —
    a composition every other leg here misses, because every other leg drives
    ``step``.

    The claim is this file's usual one on a new site: after the half-step, and again
    after the restore, the fused engine is byte-identical to the array engine over
    every stored volume as uint32 words.

    TWO ARMED CONTROLS, both required to diverge:

    * ``advanced_D_inside_the_window`` — the half-step with ``step_D`` appended,
      which is what an ``update_H``/``step_D`` weld installed at ``update_H`` would
      do here. ``D`` and ``fu_D`` are in NEITHER backup list, so the restore cannot
      reach them. This is the hazard ``fastpath.SYNC_PASS_OWNERS`` exists for,
      executed on the device rather than argued;
    * ``restore_without_f_w_H`` — the constitutive workspace dropped from the
      backup, which must also diverge, and which is what says this comparison sees
      the auxiliaries and not only the primaries.
    """
    from meep_gpu.cuda_kernels import fused_magnetic_pair as family  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage, step_curl_kernels  # noqa: PLC0415

    order = sync_order()
    label = f"{spec['label']}|{value_class}|synchronize"
    row: Dict[str, Any] = {"label": spec["label"], "value_class": value_class,
                           "sync_order": list(order),
                           "steps_before_the_half_step": steps}
    reference, ref_grid, ref_pml, _ = build(spec, value_class, case_rng(label))
    actual, grid, pml, _ = build(spec, value_class, case_rng(label))
    if compare(reference, actual):
        row["passed"] = False
        row["why"] = "the two builds are not identical"
        return row
    covered, reason = family.covers_fused_magnetic_pair(actual, pml, grid, ())
    if not covered:
        row["passed"] = False
        row["why"] = f"the predicate refused this fixture: {reason}"
        return row

    tables = family.fused_magnetic_pair_tables(pml)
    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    walls = in_seam_coverage.zero_metal_axes(grid)
    fills = family.fused_magnetic_pair_fills(grid)
    dtdx = float(grid.dt / grid.dx)
    carried = tuple(name for name in order if name in FUSED_PASSES)
    on_the_host = tuple(name for name in order if name not in FUSED_PASSES)

    def fused_step(fields, layer) -> int:
        report = family.launch_fused_magnetic_pair(
            fields, tables, codes, walls, fills, dtdx, None)
        for name in DRIVER_ORDER:
            if name not in FUSED_PASSES:
                run_pass(name, fields, layer)
        cp.cuda.runtime.deviceSynchronize()
        return int(bool(report.get("launched")))

    launched = 0
    for _ in range(steps):
        array_step(reference, ref_pml)
        launched += fused_step(actual, pml)
    row["launches_over_the_steps"] = launched
    drift = compare(reference, actual)
    row["drift_after_the_steps"] = drift
    if drift:
        row["passed"] = False
        return row

    # --- the half-step, on both engines --------------------------------------
    reference_backup = _sync_backup(reference)
    actual_backup = _sync_backup(actual)
    for name in order:
        run_pass(name, reference, ref_pml)
    _sync_average(reference, reference_backup)
    report = family.launch_fused_magnetic_pair(
        actual, tables, codes, walls, fills, dtdx, None)
    for name in on_the_host:
        run_pass(name, actual, pml)
    cp.cuda.runtime.deviceSynchronize()
    _sync_average(actual, actual_backup)
    row["launches_for_the_half_step"] = int(bool(report.get("launched")))
    row["passes_the_product_carried_in_launch"] = list(carried)
    row["passes_left_on_the_host_inside_the_window"] = list(on_the_host)
    row["differing_after_the_half_step"] = compare(reference, actual)
    row["words_the_half_step_moved"] = int(sum(
        int(np.count_nonzero(reference_backup[name] != getattr(reference, name)))
        for name in reference_backup))

    _sync_restore(reference, reference_backup)
    _sync_restore(actual, actual_backup)
    row["differing_after_the_restore"] = compare(reference, actual)
    row["arrays_the_restore_did_not_return"] = {
        name: n for name in actual_backup
        if (n := differing(actual_backup[name], getattr(actual, name)))}

    # --- the armed controls --------------------------------------------------
    controls: Dict[str, Dict[str, int]] = {}
    spoiled, _sg, spoiled_pml, _sh = build(spec, value_class, case_rng(label))
    for _ in range(steps):
        array_step(spoiled, spoiled_pml)
    backup = _sync_backup(spoiled)
    for name in order:
        run_pass(name, spoiled, spoiled_pml)
    run_pass("step_D", spoiled, spoiled_pml)   # the H->D weld's second half
    _sync_average(spoiled, backup)
    _sync_restore(spoiled, backup)
    controls["advanced_D_inside_the_window"] = compare(reference, spoiled)

    thin, _tg, thin_pml, _th = build(spec, value_class, case_rng(label))
    for _ in range(steps):
        array_step(thin, thin_pml)
    partial = {name: value for name, value in _sync_backup(thin).items()
               if not name.startswith("f_w_H")}
    for name in order:
        run_pass(name, thin, thin_pml)
    _sync_average(thin, partial)
    _sync_restore(thin, partial)
    controls["restore_without_f_w_H"] = compare(reference, thin)
    row["controls_that_must_diverge"] = {
        name: {"arrays": len(difference), "words": int(sum(difference.values())),
               "diverged": bool(difference)}
        for name, difference in controls.items()}

    row["passed"] = bool(
        carried
        and launched == steps
        and row["launches_for_the_half_step"] == 1
        and not row["differing_after_the_half_step"]
        and not row["differing_after_the_restore"]
        and not row["arrays_the_restore_did_not_return"]
        and row["words_the_half_step_moved"] > 0
        and all(difference for difference in controls.values()))
    return row


# ---------------------------------------------------------------------------
# The lift: both halves must be the certified emitters' own bytes
# ---------------------------------------------------------------------------

def leg_lift() -> Dict[str, Any]:
    """The module claims its arithmetic is LIFTED rather than retyped. Measured.

    Four questions, each a property of the construction rather than of prose:

    1. the certified curl body must survive into the fused source differing in
       EXACTLY the three ``pml_apply`` captures plus the hoisted register block;
    2. the certified constitutive body must differ in EXACTLY the three
       ``constitutive_apply`` lines (the seam plus the sub-lattice rename);
    3. the certified curl prelude must differ in EXACTLY the ``pml_apply``
       signature and its store, and the certified constitutive prelude must be
       carried with ZERO edits;
    4. ``LIFT_EDITS`` must have exactly five entries -- it is data so that a
       silent sixth edit is a visible diff, and a gate that did not count them
       would leave that promise unkept.

    And one correctness question that is not about the lift at all: ``Bx``,
    ``By`` and ``Bz`` must be bound EXACTLY ONCE each in the signature, and
    never as a ``const __restrict__`` source. Two ``__restrict__`` pointers to
    one allocation is UB that NVRTC miscompiles without a diagnostic.
    """
    from meep_gpu.cuda_kernels import (constitutive_kernels, fused_magnetic_pair as family,  # noqa: PLC0415
                                       own_cell_hoist, step_curl_kernels)

    source = family.fused_magnetic_pair_source()
    raw_curl = family._split_body(  # noqa: SLF001
        step_curl_kernels._step_B_pml_real_kernel_code,  # noqa: SLF001
        step_curl_kernels._REAL_PML_PRELUDE, "curl")  # noqa: SLF001
    lifted_curl = family.certified_curl_body()
    # THE CONSTITUTIVE COMPARAND IS THE RAW BODY'S TAIL, not the whole body. The
    # lift deliberately DROPS the decode prologue -- the curl half already
    # emitted it and a second copy would redeclare i/j/k -- so comparing against
    # the whole body would report six dropped lines that are not edits to the
    # arithmetic at all. The tail is taken at the SAME anchor the module splits
    # on, so the two cannot disagree about where the prologue ends.
    # THE SINGLE IS HOISTED (register view); the pair lifts its statement form, which
    # own_cell_hoist returns byte for byte (and refuses by name if it cannot).
    raw_const_body = family._split_body(  # noqa: SLF001
        own_cell_hoist.unhoisted_kernel_code(
            constitutive_kernels._update_H_pml_real_kernel_code,  # noqa: SLF001
            "update_H_pml_real", "_update_H_pml_real_kernel_code"),
        constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE, "constitutive")  # noqa: SLF001
    decode_end = family._DECODE_END  # noqa: SLF001
    raw_const = raw_const_body.split(decode_end, 1)[1]
    lifted_const = family.certified_constitutive_body()
    hoisted_const = family.hoisted_constitutive_body()

    def changed(before: str, after: str) -> List[List[str]]:
        a = [line for line in before.splitlines() if line.strip()]
        b = [line for line in after.splitlines() if line.strip()]
        common = set(a) & set(b)
        return [[line for line in a if line not in common],
                [line for line in b if line not in common]]

    curl_dropped, curl_added = changed(raw_curl, lifted_curl)
    const_dropped, const_added = changed(raw_const, lifted_const)
    hoist_dropped, hoist_added = changed(lifted_const, hoisted_const)
    prelude = family.fused_magnetic_pair_prelude()
    prelude_dropped, prelude_added = changed(
        step_curl_kernels._REAL_PML_PRELUDE  # noqa: SLF001
        + constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE, prelude)  # noqa: SLF001

    # THE OWN-CELL HOIST (LIFT_EDITS 8, 11): twelve pre-loads, each behind the
    # certified load's own guard, and the three captures now take them.
    hoisted_loads = [line.strip() for line in curl_added
                     if line.strip().startswith("float pre_")]
    expected_loads = sorted(
        [f"float pre_w_{a} = own_{a} ? f_w_H{a}[idx] : 0.0f;" for a in "xyz"]
        + [f"float pre_h_{a} = own_{a} ? H{a}[idx] : 0.0f;" for a in "xyz"]
        + [f"float pre_fu_{a} = fu_B{a}[idx];" for a in "xyz"]
        + [f"float pre_b_{a} = own_{a} ? B{a}[idx] : 0.0f;" for a in "xyz"])
    curl_ok = (len(curl_dropped) == 3
               and all("pml_apply(B" in line for line in curl_dropped)
               and all(f"b_{a} = pml_apply_reg_pre(B{a}" in "".join(curl_added)
                       and f"own_{a}, pre_fu_{a}, pre_b_{a});" in "".join(curl_added)
                       for a in "xyz")
               and sorted(hoisted_loads) == expected_loads)
    # THE CONSTITUTIVE SIDE GAINED A SECOND KIND OF EDIT ON 2026-08-28 and the
    # question changed with it. Three lines still leave (the certified reloads of
    # B) and three still arrive in their place (the seam plus the sub-lattice
    # rename) -- but each now sits inside its component's ownership guard, followed
    # by the carry blocks. So what is asked is that NOTHING ELSE arrived: every
    # added line must be one of the guard, the rewritten certified call, a brace, a
    # comment, or a line of a carry block, and the carry blocks must be exactly the
    # 21 destinations carried_destinations names (7 per component).
    carry_shapes = ("int g", "float g", "constitutive_apply(H", "if (", "}",
                    "B", "//")
    const_unexpected = [line for line in const_added
                        if not line.strip().startswith(carry_shapes)]
    ghost_blocks = [line for line in const_added if "_i = idx " in line]
    seam_calls = [line for line in const_added
                  if line.strip().startswith("constitutive_apply(")
                  and ", idx, b_" in line]
    const_ok = (len(const_dropped) == 3
                and all("constitutive_apply(" in line for line in const_dropped)
                and len(seam_calls) == 3
                and all("kms_int_" in line for line in seam_calls)
                and len(ghost_blocks) == 21
                and not const_unexpected
                and const_added.count("    if (own_x) {") == 1
                and const_added.count("    if (own_y) {") == 1
                and const_added.count("    if (own_z) {") == 1)
    # THE HOISTED CONSTITUTIVE STATEMENTS (LIFT_EDIT 12), measured on the text the
    # composer emits: exactly the three owned calls move, to the _pre helper with
    # their two preloaded words, and nothing else in the body does.
    hoist_ok = (len(hoist_dropped) == 3 and len(hoist_added) == 3
                and all(line.strip().startswith(f"constitutive_apply(H{a}, f_w_H{a}, idx, b_{a}, ")
                        for line, a in zip(sorted(hoist_dropped), "xyz"))
                and all(line.strip().startswith(
                            f"constitutive_apply_pre(H{a}, f_w_H{a}, idx, b_{a}, "
                            f"pre_w_{a}, pre_h_{a}, ")
                        for line, a in zip(sorted(hoist_added), "xyz"))
                and hoisted_const.count("constitutive_apply(H") == 21)
    const_ok = const_ok and hoist_ok
    prelude_ok = (len(prelude_dropped) == 4
                  and any("void pml_apply(" in line for line in prelude_dropped)
                  and any(line.strip() == "float fprev = fu[idx];" for line in prelude_dropped)
                  and "pml_apply_reg(" not in prelude
                  and prelude.count("float pml_apply_reg_pre(") == 1
                  and prelude.count("void constitutive_apply_pre(") == 1
                  and any("float kms, float sinv, float kms_u, float sinv_u" in line
                          for line in prelude_dropped)
                  and any("if (!owned) return 0.0f;" in line
                          for line in prelude_added)
                  and constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE in prelude)  # noqa: SLF001
    stencil = source.count("dtdx * ((sf - f1) + (f2 - ss))")
    binding_ok = (source.count("float* __restrict__ Bx") == 1
                  and "const float* __restrict__ Bx" not in source
                  and all(source.count(f"float* __restrict__ B{a}") == 1
                          for a in "xyz"))
    diagonal_ok = all(
        f"if (own_{a} && wall_{a} && {c} == 0) "
        f"{{ b_{a} = 0.0f; B{a}[idx] = 0.0f; }}" in source
        for a, c in zip("xyz", "ijk"))
    # THE OWNERSHIP INVERSION, as a property of the emitted text rather than of
    # prose: three flags, 21 carried destinations, and the fu store ABOVE the guard
    # while the flux store is below it.
    guard = prelude.index("if (!owned) return 0.0f;")
    ownership_ok = (source.count("int own_") == 3
                    and prelude.count("if (!owned) return 0.0f;") == 1
                    and source.count("_i = idx ") == 21
                    and prelude.index("fu[idx] = fu_new;") < guard
                    < prelude.index("f[idx] = value;"))
    return {
        "passed": bool(curl_ok and const_ok and prelude_ok and binding_ok
                       and diagonal_ok and ownership_ok and stencil == 3
                       and len(family.LIFT_EDITS) == 12),
        "curl_lines_replaced": curl_dropped,
        "curl_lines_introduced": curl_added,
        "curl_lift_ok": curl_ok,
        "constitutive_lines_replaced": const_dropped,
        "constitutive_lines_introduced": const_added,
        "constitutive_lift_ok": const_ok,
        "prelude_lines_replaced": prelude_dropped,
        "prelude_lift_ok": prelude_ok,
        "parenthesised_stencils_verbatim": stencil,
        "shared_volume_bound_once": binding_ok,
        "zero_metal_table_is_the_b_diagonal": diagonal_ok,
        "constitutive_lines_unexpected": const_unexpected,
        "carried_destinations_emitted": len(ghost_blocks),
        "ownership_flags_declared": source.count("int own_"),
        "ownership_inversion_ok": ownership_ok,
        "declared_lift_edits": len(family.LIFT_EDITS),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_lines": len(source.splitlines()),
    }


# ---------------------------------------------------------------------------
# The refusals this predicate ADDS over its two halves
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The deposit: an in-seam magnetic source CARRIED across the fused launch
# ---------------------------------------------------------------------------
#
# THIS IS THE LEG THE FLAG IS WORTH. `fused_magnetic_pair.CARRIES_DEPOSIT_REPAIR`
# became True on 2026-08-28 with `cuda_kernels/fused_pairs.py`, so this predicate
# ADMITS rows it refused before -- every row whose magnetic source lands between
# `step_B` and `update_H` (driver.py:3293), which is 58 of the 186 B_to_H
# seam-instances the corpus drives. AN ADMITTED ROW THAT NOTHING MEASURES IS STRICTLY
# WORSE THAN A REFUSED ONE: the refusal at least computed the right numbers on the
# array path.
#
# WHY THIS WALK IS NOT `run_case`, and the difference is the whole mechanism:
#
#   * the INJECTION is performed, between the fused launch and the repair, because
#     that is where the driver performs it;
#   * `zero_metal_B` runs ON THE HOST after the injection even though the kernel
#     carries it inline, because driver.py:3295 runs it UNCONDITIONALLY rather than
#     behind a `dispatch` consult. That is what lets the repair read a final B and
#     own no wall handling of its own (`deposit_repair`, "WHERE IT BELONGS IN THE
#     STEP"). A deposit that lands on a walled cell is recomputed from the zero that
#     clear left, which is what the array path computes too;
#   * the two slots are run as the two separate consults they are: `LeadingRepairPlan`
#     at `step_B` (save, then launch) and `TrailingRepairPlan` at `update_H` (apply).
#
# AND THE NULL CONTROL IS THE POINT. `repair=False` asks the SHIPPED composer for a
# plan with an EMPTY source list -- the bare pair plus the `NoopPlan` the False branch
# of this flag would leave in the trailing slot -- and then injects anyway. That is
# exactly "a product that passes True without those wrappers", built out of shipped
# code rather than a hand-written mutant, and it MUST diverge. Without it a green
# `repair=True` row would be consistent with a deposit too small to see.

#: The extent of the EXTENDED deposit row. A point source deposits into four cells;
#: this one deposits into a volume, so the repair's per-point loop is exercised at
#: many points rather than at four and a repair that handled only its first point
#: would show.
EXTENDED_SIZE: Tuple[float, float, float] = (2.0, 2.1, 1.2)

#: Which H component each declared mirror plane leaves EVEN, and it is the ENGINE'S
#: OWN rule rather than a table kept here: ``sources._validate_symmetry_parity``
#: refuses a source centred on a plane that gives it parity -1, because MEEP would
#: not refuse it and would instead return a smooth, plausible field 57%-283% away
#: from the full-domain run. ``fields.mirror_parity(c, axis, phase)`` is
#: ``phase * (1 - 2*iyee[c][axis])``, so on a folded axis ``a`` the even H
#: components are exactly those with ``iyee == 0`` at ``phase +1`` and those with
#: ``iyee == 1`` at ``phase -1``.
def deposit_rows(spec: Dict[str, Any]
                 ) -> Tuple[Tuple[str, str, Tuple[float, float, float]], ...]:
    """``(tag, component, size)`` for this spec's two deposit rows: POINT and EXTENT.

    THE TWO ROWS DIFFER IN EXTENT, WHICH IS THE PROPERTY THEY EXIST FOR: a point
    source deposits into four cells and an extended one into a volume, so the
    repair's per-point loop is exercised at many points rather than at four. The
    COMPONENT is a second axis of variation and is DERIVED rather than declared.

    A FOLDED PLANE ADMITS ONLY ITS EVEN COMPONENTS, and that is the engine's own
    refusal, not a choice made here: ``sources._validate_symmetry_parity`` rejects a
    source centred on a plane that gives it parity -1, because MEEP does not and
    instead returns a smooth, plausible field 57%-283% away from the full-domain
    run. ``fields.mirror_parity(c, axis, phase)`` is ``phase * (1 - 2*iyee)``, and a
    SINGLE mirror plane leaves exactly ONE H component even -- so a folded row runs
    both extents on that one component rather than dropping the extended row, which
    would quietly take the repair's loop back to four points.
    """
    from meep_gpu.fields import mirror_parity  # noqa: PLC0415

    folded = {"XYZ".index(axis): int(phase) for axis, phase in spec.get("symmetry", ())}
    admissible = [name for name in ("Hy", "Hz", "Hx")
                  if all(mirror_parity(name, axis, phase) == 1
                         for axis, phase in folded.items())]
    if not admissible:
        raise SystemExit(
            f"spec {spec['label']} leaves NO H component even about its mirror "
            f"planes, so no in-seam magnetic deposit can be built on it at all and "
            f"the row would report a refusal as an absence")
    point, extent = admissible[0], admissible[min(1, len(admissible) - 1)]
    return (("point", point, (0.0, 0.0, 0.0)), ("extent", extent, EXTENDED_SIZE))


#: THE CARRIER FREQUENCY, AND IT IS A MEASUREMENT RATHER THAN A DEFAULT.
#: ``continuous_src_time::dipole`` is ``amp_factor * exp(-i*omega*t)`` with
#: ``amp_factor`` imaginary (MEEP sources.cpp:104), so the REAL part a real-storage
#: run deposits is proportional to ``sin(omega t)``. These fixtures run at
#: ``courant 0.5 / resolution 1.0``, i.e. ``dt = 0.5`` -- and at ``frequency 1.0``,
#: which is what this file used until 2026-08-28, ``omega * t = 2*pi * k * 0.5`` is
#: an exact multiple of ``pi`` at EVERY sampled step, so the deposit is IDENTICALLY
#: ZERO and every deposit leg agreed with a no-op. ``leg_deposit_null_control``
#: caught it on the first device run of these legs: the UNBRACKETED launch did not
#: diverge, at 4, 12, 30 and 60 steps. 0.37 samples off the zeros -- measured
#: max|dB| 9.87e-02, 7.84e-02, 3.64e-02, 1.07e-01 over the first four steps -- and
#: :func:`run_deposit_case` now measures the deposit rather than assuming it.
DEPOSIT_FREQUENCY = 0.37


def _magnetic_source(fields, component: str = "Hy",
                     size: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """One REAL engine magnetic source, built against THIS engine's own grid.

    Never a stub with a hand-set ``field_type``: a stub would let this leg pass while
    the engine classified the same component the other way. ``VolumeSource`` resolves
    the slot itself through ``_field_type_for``.
    """
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=(0.0, 0.0, 0.0), size=size,
                        envelope=ContinuousEnvelope(frequency=DEPOSIT_FREQUENCY))


def deposit_magnitude(fields, source, steps: int, dt: float) -> float:
    """The largest standing deposit this source leaves over the step budget.

    THE PRECONDITION THE DEPOSIT LEGS WERE MISSING. A source whose real part is zero
    at every sampled time deposits nothing, and then "the repair reproduces the
    driver" and "there was nothing to reproduce" are the same green row. Measured on
    a THROWAWAY copy of the field state, restored before the leg runs.
    """
    component = "B" + str(source.component)[-1]
    array = getattr(fields, component)
    base = cp.array(array, copy=True)
    try:
        largest = 0.0
        for step in range(1, steps + 1):
            _withdraw((source,), fields)
            source.inject(fields, (step - 1) * dt)
            largest = max(largest, float(cp.abs(array - base).max()))
        return largest
    finally:
        array[...] = base


def _withdraw(sources: Sequence[Any], fields) -> None:
    """``driver.py:3289``'s withdraw pass, on both sides identically."""
    for source in sources:
        hook = getattr(source, "withdraw", None)
        if callable(hook):
            hook(fields)


def deposits_on_a_cleared_cell(fields, source) -> int:
    """How many of this source's deposit points ``zero_metal_B`` also zeroes.

    THE ONE ASYMMETRY BETWEEN THE TWO PATHS, MEASURED RATHER THAN ARGUED. The fused
    kernel carries ``zero_metal_B`` INLINE and therefore clears before the injection;
    the driver clears AFTER it (driver.py:3295). The two can only differ where a
    deposit lands on a cell the clear zeroes, and both paths would then read the same
    final zero -- but "would" is a claim, so the count is measured per row and
    reported. A zero this leg never looked at would be a silence.
    """
    index = deposit_repair._deposit_index(source)  # noqa: SLF001
    if index is None:
        return 0
    probe_arrays = {name: cp.array(getattr(fields, name), copy=True)
                    for name in ("Bx", "By", "Bz")}
    try:
        for name in ("Bx", "By", "Bz"):
            getattr(fields, name)[...] = 1.0
        stepping.zero_metal_B(fields)
        component = "B" + str(source.component)[-1]
        return int(cp.count_nonzero(getattr(fields, component)[index] == 0.0))
    finally:
        for name, saved in probe_arrays.items():
            getattr(fields, name)[...] = saved


def run_deposit_case(spec: Dict[str, Any], value_class: str, steps: int,
                     repair: bool = True, component: str = "Hy",
                     size: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     ) -> Dict[str, Any]:
    """Complete driver steps with a magnetic source IN the seam, byte compared."""
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    started = time.time()
    label = f"{spec['label']}|{value_class}|deposit|{component}|{'repair' if repair else 'null'}"
    case: Dict[str, Any] = {"label": spec["label"], "value_class": value_class,
                            "case_seed_label": label, "steps_requested": steps,
                            "repair_wired": repair, "source_component": component,
                            "source_size": list(size)}

    reference, ref_grid, ref_pml, _ = build(spec, value_class, case_rng(label))
    actual, grid, pml, _ = build(spec, value_class, case_rng(label))
    drift = compare(reference, actual)
    if drift:
        case["passed"] = False
        case["why"] = f"the two builds are not identical: {drift}"
        return case

    reference_sources = (_magnetic_source(reference, component, size),)
    actual_sources = (_magnetic_source(actual, component, size),)
    case["source_field_type"] = str(actual_sources[0].field_type)
    case["deposit_points_on_a_cleared_cell"] = deposits_on_a_cleared_cell(
        actual, actual_sources[0])
    index = deposit_repair._deposit_index(actual_sources[0])  # noqa: SLF001
    if index is None or len(deposit_repair.in_seam_sources(actual_sources, "B")) != 1:
        case["passed"] = False
        case["why"] = ("the source this leg built is not an in-seam B deposit; the "
                       "walk would inject nothing and compare a no-op with a no-op")
        return case
    case["deposit_points"] = int(cp.asarray(index[0]).size)
    # NON-VACUITY, MEASURED. A deposit of exactly zero makes every comparison below
    # agree with a no-op -- which is what frequency 1.0 did at dt = 0.5 until
    # 2026-08-28 (see DEPOSIT_FREQUENCY). Measured on a throwaway copy, before the
    # walk, so a leg that would have measured nothing says so instead of passing.
    case["deposit_magnitude"] = deposit_magnitude(
        actual, _magnetic_source(actual, component, size), steps, float(grid.dt))
    if case["deposit_magnitude"] <= 0.0:
        case["passed"] = False
        case["why"] = ("this source deposits exactly zero over the whole budget, so "
                       "the repaired and unrepaired walks would agree with each "
                       "other and with a no-op")
        return case

    # THE SHIPPED COMPOSER BUILDS THE PLAN, not this file. Reaching for
    # `run_fused_magnetic_pair` directly would test the kernel and skip the wiring,
    # and the wiring is what this leg exists for.
    composed = arms.plan_step(actual, pml, grid,
                              sources=(actual_sources if repair else ()), fuse=True)
    leading = composed.plans.get("step_B")
    trailing = composed.plans.get("update_H")
    installed = (type(leading).__name__, type(trailing).__name__)
    case["installed_plans"] = list(installed)
    case["selected"] = {slot: composed.selected.get(slot)
                        for slot in ("step_B", "update_H")}
    if repair:
        installed_ok = installed == ("LeadingRepairPlan", "TrailingRepairPlan")
    else:
        # The null control's leading slot is the builder's own class and is asserted
        # only to NOT be the repair wrapper -- naming it would pin this leg to a class
        # name the builder is free to change.
        installed_ok = (installed[0] != "LeadingRepairPlan"
                        and installed[1] == "NoopPlan")
    case["installed_as_expected"] = installed_ok
    if leading is None or trailing is None:
        case["passed"] = False
        case["why"] = "the composer did not fuse the seam"
        case["refusals"] = [reason for key, value in composed.reasons.items()
                            if key.startswith("fused_pair") for reason in value]
        return case
    inner = getattr(leading, "absorbed_by", leading)

    before = frozen(actual)
    dt = float(grid.dt)
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        when = (step - 1) * dt
        _withdraw(reference_sources, reference)
        for name in DRIVER_ORDER:
            run_pass(name, reference, ref_pml)
            if name == "step_B":
                for source in reference_sources:
                    source.inject(reference, when)

        _withdraw(actual_sources, actual)
        for name in DRIVER_ORDER:
            if name == "step_B":
                leading.run()
                for source in actual_sources:
                    source.inject(actual, when)
            elif name == "update_H":
                trailing.run()
            elif name == "zero_metal_B":
                run_pass(name, actual, pml)
            else:
                run_pass(name, actual, pml)
        cp.cuda.runtime.deviceSynchronize()

        difference = compare(reference, actual)
        census = operand_census({n: to_host(v)
                                 for n, v in state_of(reference).items()})
        per_step.append({"step": step,
                         "differing_words": sum(difference.values()),
                         "differing_arrays": dict(sorted(difference.items())),
                         "reference_subnormals": census["subnormals"]})
        if difference:
            break

    after = state_of(actual)
    moved = {name: int(np.count_nonzero(before[name] != words(after[name])))
             for name in STATE_NAMES}
    still = sorted(name for name, count in moved.items() if count == 0)
    identical = (len(per_step) == steps
                 and all(row["differing_words"] == 0 for row in per_step))
    repairs = int(getattr(leading, "repairs", 0))
    case.update({
        "passed": bool(identical and installed_ok and not still
                       and (repairs > 0 if repair else repairs == 0)),
        "bit_identical": identical,
        "deposit_points_repaired": repairs,
        "steps_run": len(per_step),
        "first_divergence": next((row["step"] for row in per_step
                                  if row["differing_words"]), None),
        "per_step": per_step,
        "differing_words": per_step[-1]["differing_words"] if per_step else -1,
        "differing_arrays": per_step[-1]["differing_arrays"] if per_step else {},
        "arrays_that_never_moved": still,
        "launches": int(getattr(inner, "launches", 0)),
        "seconds": time.time() - started,
    })
    return case


def leg_deposit_null_control(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """The unbracketed launch MUST diverge. Scored on diverging, not on agreeing.

    RUN ON A FOLDED ROW TOO SINCE 2026-08-28, and that row is the one that carries
    the argument: on a folded seam the two fills run AFTER the injection, so a
    repair that wrote only the deposit INDEX would leave every mirror image of it
    describing the pre-injection field. The component is the one the fold leaves
    even (:func:`deposit_rows`), because the engine refuses the others outright.
    """
    result = run_deposit_case(spec, "uniform", steps, repair=False,
                              component=deposit_rows(spec)[0][1])
    diverged = not result.get("bit_identical", False)
    return {
        "passed": bool(diverged and result.get("installed_as_expected")),
        "diverged": diverged,
        "installed_plans": result.get("installed_plans"),
        "first_divergence": result.get("first_divergence"),
        "differing_words": result.get("differing_words"),
        "differing_arrays": result.get("differing_arrays"),
        "deposit_points_repaired": result.get("deposit_points_repaired"),
        "deposit_points": result.get("deposit_points"),
    }


def leg_refusal() -> Dict[str, Any]:
    """The source seam and the fold, each measured BY NAME.

    A clause nothing tests is a claim. TWO CLAUSES HAVE INVERTED. CLAUSE 2, the
    magnetic-source slot used to CAP this family on the measured corpus; it is now
    CARRIED, because ``fused_magnetic_pair.CARRIES_DEPOSIT_REPAIR`` is True and
    ``fused_pairs._install_fused_pair`` brackets the launch. So this leg no longer
    asks that a magnetic source be refused -- the ``deposit`` legs measure what that
    admission is worth -- and asks instead the five questions the flip does NOT
    license:

    1. an UNDECLARED source list must still be refused: ignorance is not an empty
       set, and no repair can carry a deposit nobody declared;
    2. a real MAGNETIC ``VolumeSource`` must now be ADMITTED, and the admission must
       come with THE WIRING -- ``arms.plan_step(..., fuse=True)`` must put a
       ``LeadingRepairPlan`` in ``step_B`` and a ``TrailingRepairPlan`` in
       ``update_H``. Admission without those two wrappers is the exact defect
       ``deposit_repair`` exists to prevent, so the two are asserted together;
    2b. a magnetic source that does NOT publish the index it writes must still be
       refused BY NAME -- the flip carries deposits this module can reconstruct, not
       every deposit;
    3. a real ELECTRIC ``VolumeSource`` is still not this seam's business;
    4. CLAUSE 4 HAS INVERTED TOO, 2026-08-28. A FOLDED grid used to be refused
       naming both uncarried fills; the ownership inversion carries them, so the
       fold is now ADMITTED and what is asked instead is what the carry does NOT
       reach, each BY NAME:

       4a. a mirrored axis whose declared phase is not +/-1 -- delegated to
           ``in_seam_coverage.covers_fill_symmetry`` and refused with that pass
           named, so a clause added there reaches this seam without a second edit;
       4b. a folded axis too short to hold the near fill's source row, which
           ``stepping._mirror_source`` raises on and this kernel would index
           outside;
       4c. an axis reported both folded and walled -- unreachable from a real
           ``Grid`` (``zero_metal_axes`` is ``is_metallic and not is_mirrored``)
           and the fact the near carry rests on, so it is armed at
           ``fused_magnetic_pair_fills`` where it IS reachable.

    The sources are real ``VolumeSource`` objects, never stubs with a hand-set
    ``field_type``: a stub would let this leg pass while the engine classified the
    same component the other way.
    """
    from meep_gpu.cuda_kernels import arms, fused_magnetic_pair as family  # noqa: PLC0415
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    spec = dict(SPECS[1])
    fields, grid, pml, _ = build(spec, "uniform", case_rng("refusal"))

    def source_on(component: str):
        return VolumeSource(grid=grid, component=component,
                            center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                            envelope=ContinuousEnvelope(frequency=1.0))

    undeclared = family.covers_fused_magnetic_pair(fields, pml, grid, None)
    magnetic = source_on("Hy")
    magnetic_verdict = family.covers_fused_magnetic_pair(fields, pml, grid, (magnetic,))

    # THE WIRING, ASKED OF THE SHIPPED COMPOSER. A fresh build, because plan_step
    # composes against the fields it is handed.
    wired_fields, wired_grid, wired_pml, _ = build(spec, "uniform", case_rng("wired"))
    wired_source = VolumeSource(grid=wired_grid, component="Hy",
                                center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
                                envelope=ContinuousEnvelope(frequency=1.0))
    composed = arms.plan_step(wired_fields, wired_pml, wired_grid,
                              sources=(wired_source,), fuse=True)
    installed = [type(composed.plans.get(slot)).__name__
                 for slot in ("step_B", "update_H")]
    wiring_ok = installed == ["LeadingRepairPlan", "TrailingRepairPlan"]

    # 2b. THE DEPOSIT THIS MODULE CANNOT SAVE. Built by BLANKING the real source's
    # resolved index rather than by stubbing a fake source, so `field_type` stays the
    # engine's own answer.
    indexless = source_on("Hy")
    indexless._point_ix = indexless._point_iy = indexless._point_iz = None  # noqa: SLF001
    indexless_verdict = family.covers_fused_magnetic_pair(
        fields, pml, grid, (indexless,))
    indexless_run = family.run_fused_magnetic_pair(
        fields, grid, pml, float(grid.dt / grid.dx), sources=(indexless,))

    electric = source_on("Ez")
    electric_verdict = family.covers_fused_magnetic_pair(fields, pml, grid, (electric,))

    folded_spec = dict(next(row for row in SPECS
                            if row["label"] == "fold_y_periodic"))
    folded_fields, folded_grid, folded_pml, _ = build(
        folded_spec, "uniform", case_rng("refusal_folded"))
    folded_verdict = family.covers_fused_magnetic_pair(
        folded_fields, folded_pml, folded_grid, ())
    folded_plan = family.fused_magnetic_pair_fills(folded_grid)

    # 4a. A MIRRORED AXIS WITH NO DECLARED +/-1 PHASE. Delegated: the refusal must
    # name the FILL, so a reader can tell this seam asked in_seam_coverage rather
    # than carrying its own weaker copy of the question.
    class _OddPhase:
        def __getattr__(self, item):
            return getattr(folded_grid, item)

        def mirror_phase(self, axis):
            return 0 if axis == 1 else folded_grid.mirror_phase(axis)

    phase_verdict = family.covers_fused_magnetic_pair(
        folded_fields, folded_pml, _OddPhase(), ())

    # 4b. A FOLDED AXIS TOO SHORT FOR STORED ROW 2.
    class _TooShort:
        def __getattr__(self, item):
            return getattr(folded_grid, item)

        def stored_cells(self, axis):
            return 2 if axis == 1 else folded_grid.stored_cells(axis)

    short_verdict = family.covers_fused_magnetic_pair(
        folded_fields, folded_pml, _TooShort(), ())

    # 4c. FOLDED AND WALLED AT ONCE, armed where it is reachable: the plan builder.
    # `zero_metal_axes` can never report it, so a clause tested only through the
    # predicate would be a clause nothing can reach -- indistinguishable from one
    # that returns True.
    held = family.zero_metal_axes
    try:
        family.zero_metal_axes = lambda _grid: (False, True, False)
        try:
            family.fused_magnetic_pair_fills(folded_grid)
            walled_fold_refused, walled_fold_reason = False, "the plan was built"
        except ValueError as error:
            walled_fold_refused, walled_fold_reason = True, str(error)
        walled_fold_verdict = family.covers_fused_magnetic_pair(
            folded_fields, folded_pml, folded_grid, ())
    finally:
        family.zero_metal_axes = held

    undeclared_named = "was not declared" in str(undeclared[1])
    indexless_named = ("does not publish the index it writes"
                       in str(indexless_verdict[1]))
    phase_named = ("fill_symmetry_bc_B" in str(phase_verdict[1])
                   and "mirror phase" in str(phase_verdict[1]))
    short_named = ("does not exist" in str(short_verdict[1])
                   or "stored_cells" in str(short_verdict[1]))
    walled_fold_named = ("both folded and walled" in walled_fold_reason
                         and "both folded and walled" in str(walled_fold_verdict[1]))
    return {
        "passed": bool(not undeclared[0] and undeclared_named
                       and magnetic_verdict[0] and wiring_ok
                       and not indexless_verdict[0] and indexless_named
                       and not indexless_run.get("launched")
                       and electric_verdict[0]
                       and folded_verdict[0]
                       and not phase_verdict[0] and phase_named
                       and not short_verdict[0] and short_named
                       and walled_fold_refused and walled_fold_named
                       and not walled_fold_verdict[0]),
        "carries_deposit_repair": bool(family.CARRIES_DEPOSIT_REPAIR),
        "undeclared_refused": not undeclared[0],
        "undeclared_reason": undeclared[1],
        "magnetic_field_type": str(getattr(magnetic, "field_type", "?")),
        "magnetic_admitted": bool(magnetic_verdict[0]),
        "magnetic_reason": magnetic_verdict[1],
        "magnetic_wiring_installed": installed,
        "magnetic_wiring_ok": wiring_ok,
        "indexless_refused": not indexless_verdict[0],
        "indexless_reason": indexless_verdict[1],
        "indexless_named_the_index": indexless_named,
        "indexless_launch_refused": not indexless_run.get("launched"),
        "electric_field_type": str(getattr(electric, "field_type", "?")),
        "electric_admitted": bool(electric_verdict[0]),
        "electric_reason": electric_verdict[1],
        "folded_admitted": bool(folded_verdict[0]),
        "folded_reason": folded_verdict[1],
        "folded_fill_plan": {key: list(value) for key, value in folded_plan.items()},
        "odd_phase_refused": not phase_verdict[0],
        "odd_phase_reason": phase_verdict[1],
        "odd_phase_named_the_fill": phase_named,
        "short_fold_refused": not short_verdict[0],
        "short_fold_reason": short_verdict[1],
        "short_fold_named_the_row": short_named,
        "walled_fold_plan_refused": walled_fold_refused,
        "walled_fold_plan_reason": walled_fold_reason,
        "walled_fold_predicate_refused": not walled_fold_verdict[0],
        "walled_fold_predicate_reason": walled_fold_verdict[1],
        "walled_fold_named": walled_fold_named,
    }


def leg_nonlinear_widening() -> Dict[str, Any]:
    """The 2026-09-01 nonlinear widening, measured on all four of its edges.

    ``covers_fused_magnetic_pair`` now admits a run carrying an instantaneous
    chi2/chi3 through ``covers_real_pml_nonlinear_constitutive(side="H")`` -- the
    engine's chi lives entirely inside ``update_E`` (stepping.py:26-28), so the
    weld's bytes are untouched -- and ``fused_pairs.FUSED_PAIR_EXTRA_ARMS``
    declares the matching (``PML``, ``nonlinear``) absorption. Four measurements,
    each of which has a way to be silently wrong without it:

    1. **THE CHI IS LIVE, not decorative.** Two engines from one seed, one with
       the chi cleared, stepped once on the ARRAY path: the E family must
       diverge. A fixture whose chi moved nothing would score every nonlinear
       case as a linear case wearing the label.
    2. **THE COMPOSER'S OWN SELECTION** is (``PML``, ``nonlinear``) at
       ``fuse=False`` -- read from ``plan_step``, never assumed from the census.
    3. **THE ABSORPTION**: at ``fuse=True`` the pair takes both slots through the
       extras row, and the trailing slot is the ``NoopPlan`` (no source declared,
       nothing in the seam to repair).
    4. **THE LINEAR TWIN CONTROL**: the same spec with no chi selects
       (``PML``, ``ordinary``) and absorbs through the PRIMARY row, so the extras
       table is shown to widen the wiring rather than replace it.

    The nonlinear rows in :data:`SPECS` carry the bit-identity, movement-floor
    and launch-count halves of this claim through the ordinary product sweep;
    the mutation battery is deliberately NOT re-run on them -- the kernel text is
    unchanged, and a mutation leg answers "can this gate see this defect", which
    does not depend on the chi the kernel never reads.
    """
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    spec = next(s for s in SPECS if s["label"] == "wall_z_nonlinear")

    # 1. The liveness floor.
    with_chi, grid_a, pml_a, _ = build(spec, "uniform", case_rng("nl-live"))
    without_chi, grid_b, pml_b, _ = build(spec, "uniform", case_rng("nl-live"))
    without_chi.set_nonlinear_volumes({}, {})
    drift = compare(with_chi, without_chi)
    seed_identical = not drift
    array_step(with_chi, pml_a)
    array_step(without_chi, pml_b)
    moved = compare(with_chi, without_chi)
    chi_moved_words = sum(moved.values())

    # 2 + 3. Selection, then absorption, on a fresh build.
    fields, grid, pml, _ = build(spec, "uniform", case_rng("nl-wiring"))
    unfused = arms.plan_step(fields, pml, grid, sources=(), fuse=False)
    selected = {slot: unfused.selected.get(slot)
                for slot in ("step_B", "update_H")}
    fused = arms.plan_step(fields, pml, grid, sources=(), fuse=True)
    installed = [type(fused.plans.get(slot)).__name__
                 for slot in ("step_B", "update_H")]
    absorbed = (installed[0] == "CudaFusedPairPlan"
                and installed[1] == "NoopPlan")

    # 4. The linear twin.
    linear_spec = {key: value for key, value in spec.items()
                   if key != "nonlinear"}
    linear_spec["label"] = spec["label"] + "|linear_twin"
    l_fields, l_grid, l_pml, _ = build(linear_spec, "uniform",
                                       case_rng("nl-linear-twin"))
    l_unfused = arms.plan_step(l_fields, l_pml, l_grid, sources=(), fuse=False)
    l_selected = {slot: l_unfused.selected.get(slot)
                  for slot in ("step_B", "update_H")}
    l_fused = arms.plan_step(l_fields, l_pml, l_grid, sources=(), fuse=True)
    l_installed = [type(l_fused.plans.get(slot)).__name__
                   for slot in ("step_B", "update_H")]
    l_absorbed = (l_installed[0] == "CudaFusedPairPlan"
                  and l_installed[1] == "NoopPlan")

    return {
        "passed": bool(seed_identical and chi_moved_words > 0
                       and selected == {"step_B": "PML", "update_H": "nonlinear"}
                       and absorbed
                       and l_selected == {"step_B": "PML",
                                          "update_H": "ordinary"}
                       and l_absorbed),
        "seed_identical_before_chi_clear": seed_identical,
        "chi_moved_words_in_one_array_step": chi_moved_words,
        "chi_moved_arrays": dict(sorted(moved.items())),
        "selected_unfused": selected,
        "installed_fused": installed,
        "absorbed_through_extras": absorbed,
        "linear_twin_selected": l_selected,
        "linear_twin_installed": l_installed,
        "linear_twin_absorbed_through_primary": l_absorbed,
    }


# ---------------------------------------------------------------------------
# The armed defects
# ---------------------------------------------------------------------------

def needle(source: str, old: str, new: str, count: int = 1) -> Tuple[str, int]:
    """Replace and REPORT THE SITE COUNT. A rewrite that matched nothing comes
    back NOT ARMED rather than launching the shipped kernel and reporting the
    defect as uncaught -- the one failure mode a mutation leg cannot see from its
    own result."""
    sites = source.count(old)
    if sites == 0:
        return source, 0
    return source.replace(old, new, count if count > 0 else -1), sites


#: ``(name, edit, must_be_caught, applies)``. ``applies`` takes the spec and says
#: whether the mutated line is REACHABLE there: a wall defect on an all-periodic
#: row is inert for a reason about the case table, not about the kernel, and
#: collapsing "unasked" into "uncaught" is how a gate acquires a silent hole.
#:
#: EVERY NULL CARRIES ITS REASON IN THE COMMENT ABOVE IT, and it is an arithmetic
#: reason rather than an expectation.

def _folded_axes(spec: Dict[str, Any]) -> set:
    return {axis.lower() for axis, _phase in spec.get("symmetry", ())}


def _wall(axis: str) -> Callable[[Dict[str, Any]], bool]:
    """Is ``axis`` a live WALL here -- ``is_metallic and not is_mirrored``?

    THE FOLD EXCLUSION IS ``zero_metal_axes``' OWN (stepping.py:2284-2286) and it
    has to be restated here now that folded rows are in the sweep: a folded
    METALLIC axis declares ``metallic`` and carries NO wall, so a wall mutation
    scored there would report UNCAUGHT for a reason about the case table.
    """
    index = "xyz".index(axis)
    return lambda spec: (spec["boundaries"][index] == "metallic"
                         and axis not in _folded_axes(spec))


def _any_wall(spec: Dict[str, Any]) -> bool:
    return any(kind == "metallic" and "xyz"[index] not in _folded_axes(spec)
               for index, kind in enumerate(spec["boundaries"]))


def _folded(spec: Dict[str, Any]) -> bool:
    """Is EITHER fill live here? Both need a declared mirror plane."""
    return bool(spec.get("symmetry"))


def _two_folds(spec: Dict[str, Any]) -> bool:
    """Is the COMPOSITION reachable? One folded axis makes every order trivial."""
    return len(spec.get("symmetry", ())) >= 2


def _folded_axis(axis: str) -> Callable[[Dict[str, Any]], bool]:
    """Is THIS axis folded here?

    THE FILL CARRIES ARE PER COMPONENT AND PER AXIS, so a needle that names ``gx_n``
    (Bx's NEAR ghost, which images along x because ``iyee[Bx][x] == 0``) is a dead
    line on a row that folds only Y -- and a mutation scored there reports UNCAUGHT
    for a reason about the case table rather than about the kernel. Measured: before
    this predicate existed, four fill mutations came back PARTIAL 1/3 on exactly the
    rows whose fold was on another axis.
    """
    return lambda spec: axis in _folded_axes(spec)


def _folded_periodic_axis(axis: str) -> Callable[[Dict[str, Any]], bool]:
    """Is this axis folded AND periodic-terminated -- i.e. does the FAR fill run?"""
    index = "xyz".index(axis)
    return lambda spec: (axis in _folded_axes(spec)
                         and spec["boundaries"][index] != "metallic")


def _bx_has_a_destination(spec: Dict[str, Any]) -> bool:
    """Does either fill image a cell of Bx here -- i.e. is ``own_x`` ever 0?

    ``iyee[Bx] == (0, 1, 1)``: the NEAR fill images Bx on X and the FAR fill on Y and
    Z. So a needle that names ``own_x`` is a DEAD LINE on a grid whose only fold is a
    METALLIC Y -- there the near fill touches By alone and the far fill does not run
    at all, ``own_x`` is identically 1, and the mutation is the identity. Measured:
    ``ownership_guard_dropped_on_the_constitutive`` came back 5/6 on 2026-08-28 with
    ``fold_y_metallic`` the single miss, for exactly this reason.
    """
    folded = _folded_axes(spec)
    walled = {"xyz"[index] for index, kind in enumerate(spec["boundaries"])
              if kind == "metallic"}
    return "x" in folded or any(axis in folded and axis not in walled
                                for axis in ("y", "z"))


def _folded_metallic(spec: Dict[str, Any]) -> bool:
    """Is some axis folded AND declared metallic?

    The ONE shape on which ``zero_metal_axes`` (``is_metallic and not
    is_mirrored``) and the grid's bare metallic declaration disagree. Every other
    row gives the two the same answer, which is why the mutation that confuses them
    was a declared NULL until the fold was admitted.
    """
    return any("xyz"[index] in _folded_axes(spec)
               for index, kind in enumerate(spec["boundaries"]) if kind == "metallic")


def _odd_full_count(axis: str) -> Callable[[Dict[str, Any]], bool]:
    """Is the far reflect row ``stored - 3`` rather than ``stored - 2`` here?

    ``_far_reflect_rows`` is ``n_full - stored + 2``. At an EVEN full count that
    equals ``stored - 2``, so a kernel that baked ``n - 2`` is exact and the defect
    is invisible; only the odd count separates them. Read off the SPEC's own extent
    rather than off a built grid, so the table can be scored without a device.
    """
    index = "xyz".index(axis)
    return lambda spec: (axis in _folded_axes(spec)
                         and spec["boundaries"][index] != "metallic"
                         and int(round(spec["cell"][index])) % 2 == 1)


def _near_ghost_pair_is_observable(axis: str) -> Callable[[Dict[str, Any]], bool]:
    """Do the constitutive coefficients differ between stored rows 0 and 2 on ``axis``?

    MEASURED FROM THE SPEC'S OWN ABSORBER, not declared. The near ghost takes the
    DESTINATION's pair at stored 0 while the source thread holds stored 2's, and a
    folded axis carries no absorber at the mirror plane -- so on a shallow layer the
    two entries are the same word to the bit and a kernel that reused the wrong one
    is BYTE-IDENTICAL. Reporting that as UNCAUGHT would leave a reader unable to
    tell a blind fixture from a dead line; this predicate is what makes it a fact
    about the fixture.
    """
    def observable(spec: Dict[str, Any]) -> bool:
        if axis not in _folded_axes(spec):
            return False
        from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415

        _fields, _grid, pml, _host = build(spec, "uniform", np.random.default_rng(7))
        tables = constitutive_kernels.real_constitutive_tables(pml, False)
        for key in (f"kps_{axis}", f"kms_{axis}"):
            column = to_host(tables[key]).ravel()
            if column.size > 2 and column[0] != column[2]:
                return True
        return False

    return observable


def _always(spec: Dict[str, Any]) -> bool:
    return True


SOURCE_MUTATIONS: Tuple[Tuple[str, Tuple[str, str, int], Optional[bool],
                              Callable[[Dict[str, Any]], bool]], ...] = (
    # --- the seam itself ----------------------------------------------------
    # The wall clear must reach the REGISTER as well as global memory, or
    # update_H consumes the un-wiped flux density -- the same error one sub-step
    # earlier than leaving global B dirty.
    ("wall_clear_misses_the_register",
     ("if (own_x && wall_x && i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }",
      "if (own_x && wall_x && i == 0) { Bx[idx] = 0.0f; }", 1), True, _wall("x")),
    # ... and it must reach GLOBAL MEMORY as well as the register, or the next
    # timestep's recurrence reads a plane of un-wiped values. Caught from step 2,
    # which is why the budget is complete steps and not one launch.
    ("wall_clear_misses_global",
     ("if (own_z && wall_z && k == 0) { b_z = 0.0f; Bz[idx] = 0.0f; }",
      "if (own_z && wall_z && k == 0) { b_z = 0.0f; }", 1), True, _wall("z")),
    # THE B TABLE IS THE DIAGONAL; the D family's is the off-diagonal complement.
    # Reusing it clears Bx on the wrong plane -- the single most likely slip in
    # porting this family from its D-side sibling.
    ("wall_clear_uses_the_wrong_axis",
     ("if (own_x && wall_x && i == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }",
      "if (own_x && wall_x && j == 0) { b_x = 0.0f; Bx[idx] = 0.0f; }", 1),
     True, _wall("x")),
    # The seam, wrong component: Hx's source is b_x, not b_y.
    ("seam_takes_the_wrong_component",
     ("constitutive_apply_pre(Hx, f_w_Hx, idx, b_x,",
      "constitutive_apply_pre(Hx, f_w_Hx, idx, b_y,", 1), True, _always),
    # THE SUB-LATTICE SHADOW, and this defect exists only because the fusion put
    # both halves in one scope. The curl's kms is HALF-INTEGER and the
    # constitutive's INTEGER; letting the certified name shadow is a half-cell
    # error in the absorber profile -- converged, smooth and wrong.
    ("constitutive_sublattice_shadow",
     ("kms_int_x[i]);", "kms_x[i]);", 1), True, _always),
    # The dsigw index is the component's OWN axis (stepping.py:947-948).
    ("constitutive_coefficient_index_moved",
     ("constitutive_apply_pre(Hy, f_w_Hy, idx, b_y, pre_w_y, pre_h_y, kps_y[j], kms_int_y[j]);",
      "constitutive_apply_pre(Hy, f_w_Hy, idx, b_y, pre_w_y, pre_h_y, kps_y[i], kms_int_y[i]);", 1),
     True, _always),
    # The two accumulations are LEFT-TO-RIGHT and SEPARATE; reassociating them is
    # a different float32 number.
    ("constitutive_accumulations_reversed",
     (" + kps * src;\n    f[idx] = a - kms * prev;",
      " - kms * prev;\n    f[idx] = a + kps * src;", -1),
     True, _always),
    # --- the two mirror fills, carried by the ownership inversion -----------
    #
    # EVERY ONE OF THESE IS NEW ON 2026-08-28 and every one is scored only where
    # the pass it breaks is LIVE: an unfolded row runs neither fill, so a fill
    # defect scored there would report UNCAUGHT for a reason about the case table.
    #
    # THE OWNERSHIP GUARD ITSELF. Dropping it puts the destination thread back on
    # its own cell: it forms v from a B another block is writing, stores B there,
    # and runs update_H there -- so the ghost is written twice and the value that
    # survives is undefined. On the array path that cell holds the fill's image.
    ("ownership_guard_dropped_on_the_constitutive",
     ("    if (own_x) {\n", "    if (1) {\n", 1), True, _bx_has_a_destination),
    # THE SAME GUARD, INVERTED, AND THE DETERMINISTIC ONE. Dropping a guard makes
    # two threads write one word, and which wins is undefined -- so the entry above
    # is a race that HAPPENS to be caught (3/3 on this device, 2026-08-28) rather
    # than a divergence the hardware owes anyone. Inverting it instead leaves every
    # word with exactly one writer: the OWNER runs nothing, so H and f_w_H at every
    # owned cell keep the previous step's value, and the ghosts are written by
    # nobody. No race, and caught on an unfolded row too, where `!own_x` is
    # identically false and the constitutive half simply does not run.
    ("ownership_guard_inverted_on_the_constitutive",
     ("    if (own_x) {\n", "    if (!own_x) {\n", 1), True, _always),
    # ... and inside pml_apply_reg, where it stops the destination thread READING
    # a word another block writes. REPORTED RATHER THAN REQUIRED, and the reason is
    # the defect's own nature: without the guard the destination thread ALSO stores
    # B at its own cell, so that word has two writers and WHICH ONE WINS IS
    # UNDEFINED. A data race has no deterministic verdict -- measured UNCAUGHT 0/3
    # on this device on 2026-08-28, which is a race resolving in the shipped
    # kernel's favour, not evidence the guard is dead. What pins this guard
    # deterministically is the write-set enumeration over the EMITTED TEXT in
    # `meep_gpu/cuda_kernels/test_fused_magnetic_pair_fill_carry.py`
    # (`test_the_write_enumeration_would_catch_a_dropped_ownership_guard`), which
    # reports the contended words directly instead of hoping the hardware picks the
    # wrong one; and `ownership_guard_inverted` below, which is deterministic.
    ("ownership_guard_dropped_in_pml_apply",
     ("    if (!owned) return 0.0f;\n", "", 1), None, _folded),
    # THE SAME GUARD, INVERTED, AND DETERMINISTIC: the OWNER returns before storing
    # B and the destination stores instead, so every owned cell keeps its previous
    # step's flux density. No race -- the owner simply stops writing.
    ("ownership_guard_inverted",
     ("    if (!owned) return 0.0f;", "    if (owned) return 0.0f;", 1),
     True, _always),
    # THE NEAR FILL'S SOURCE ROW is stepping.MIRROR_SOURCE_INDEX = 2, MEEP's
    # halved-grid io = -2. Imaging row 1 is a whole cell wrong.
    ("near_fill_images_the_wrong_row",
     ("int gx_n_i = idx - 2 * sx;", "int gx_n_i = idx - 1 * sx;", 1),
     True, _folded_axis("x")),
    ("near_fill_source_guard_moved",
     ("near_x && i == 2) {", "near_x && i == 1) {", -1), True, _folded_axis("x")),
    # THE NEAR PARITY is +phase (the imaged components have Yee shift 0 there) and
    # the FAR parity is -phase (theirs is 1). Swapping either is the doubly
    # mirrored value with the wrong sign -- smooth, converged and wrong.
    ("near_fill_parity_negated",
     ("float gx_n_p = phase_x;", "float gx_n_p = -phase_x;", 1),
     True, _folded_axis("x")),
    ("far_fill_parity_unnegated",
     ("float gx_y_p = (-phase_y);", "float gx_y_p = (phase_y);", 1), True,
     _folded_periodic_axis("y")),
    # THE REFLECT ROW is n_full - stored + 2, which is stored - 2 at an even full
    # count and stored - 3 at an odd one. Baking n - 2 reflects about the window
    # top instead of about the mirror -- exact on the even rows, a whole cell
    # wrong on fold_y_periodic_odd, which is why that spec exists.
    # AND THE ROW IT IMAGES FROM. `reflect_y` -> `ny - 2` is the fixed formula, and
    # it is EXACT at an even full count (`_far_reflect_rows` IS `stored - 2` there),
    # so this is scored on the ODD row alone -- which is why fold_y_periodic_odd is
    # in MUTATION_SPEC_LABELS. An earlier form of this needle wrote `idx + 1 * sy`
    # and was measured UNCAUGHT 0/3 for an arithmetic reason: on those fixtures
    # `ny - 1 - reflect_y` IS 1, so the mutation was the identity.
    ("far_fill_bakes_n_minus_two",
     ("int gx_y_i = idx + (ny - 1 - reflect_y) * sy;",
      "int gx_y_i = idx + (ny - 1 - (ny - 2)) * sy;", 1), True,
     _odd_full_count("y")),
    # THE NEAR GHOST TAKES THE DESTINATION'S COEFFICIENT PAIR, at stored index 0,
    # because the near fill images along the very axis update_H indexes on.
    # Reusing the source thread's applies stored cell 2's absorber profile to
    # stored cell 0.
    ("near_ghost_reuses_the_source_coefficient_pair",
     ("gx_n_i, gx_n_v, kps_x[0], kms_int_x[0]);",
      "gx_n_i, gx_n_v, kps_x[i], kms_int_x[i]);", 1), True,
     _near_ghost_pair_is_observable("x")),
    # THE COMPOSITION. A corner at the top of two far planes carries the PRODUCT
    # of both parities; dropping one factor is exact wherever that phase is +1,
    # which is why the mixed-phase spec is in MUTATION_SPEC_LABELS.
    # DROPPING THE X FACTOR AND NOT THE Y ONE, and that is arithmetic rather than
    # taste: the mixed-phase row runs phase_x +1 and phase_y -1, so the product is
    # (-1)*(+1) = -1 and dropping the Y factor leaves -1 -- the identity. Measured
    # UNCAUGHT 0/1 in that form on 2026-08-28. Dropping the X factor leaves +1.
    ("composed_parity_drops_a_factor",
     ("float gz_xy_p = (-phase_x) * (-phase_y);",
      "float gz_xy_p = (-phase_y);", 1), True, _two_folds),
    # THE CONSTITUTIVE AT THE GHOST. Without it H and f_w_H at every destination
    # plane keep the previous step's value while B carries the new image.
    ("ghost_constitutive_dropped",
     ("            constitutive_apply(Hx, f_w_Hx, gx_n_i, gx_n_v, "
      "kps_x[0], kms_int_x[0]);\n", "", 1), True, _folded_axis("x")),

    # --- the curl half ------------------------------------------------------
    ("curl_parens_flattened",
     ("dtdx * ((sf - f1) + (f2 - ss))", "dtdx * (sf - f1 + f2 - ss)", -1),
     True, _always),
    ("curl_gathers_the_lower_neighbour",
     ("float sf = shift_up(Ez, idx, j, ny, sy, bc_y);",
      "float sf = shift_dn(Ez, idx, j, ny, sy, bc_y);", 1), True, _always),
    # The recurrence pairs are vec.hpp's cycle_direction: Bx takes (y, z).
    ("recurrence_axis_pair_swapped",
     ("pml_apply_reg_pre(Bx, fu_Bx, idx, curl, kms_y[j], sinv_y[j], "
      "kms_z[k], sinv_z[k], own_x, pre_fu_x, pre_b_x);",
      "pml_apply_reg_pre(Bx, fu_Bx, idx, curl, kms_z[k], sinv_z[k], "
      "kms_y[j], sinv_y[j], own_x, pre_fu_x, pre_b_x);", 1), True, _always),
    ("fu_store_dropped", ("    fu[idx] = fu_new;\n", "", 1), True, _always),
    ("flux_store_dropped", ("    f[idx] = value;\n", "", 1), True, _always),
    ("fw_store_dropped", ("    fw[idx] = src;\n", "", -1), True, _always),
    ("metallic_curl_mask_dropped",
     ("        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;\n", "", 1),
     True, _wall("x")),
    # A metallic ghost serves an exact 0.0 past the face; wrapping it is the
    # periodic rule on a walled axis.
    ("metallic_ghost_wraps",
     ("    return (bc == BC_PERIODIC) ? g[idx - ia * stride] : 0.0f;",
      "    return g[idx - ia * stride];", 1), True, _wall("x")),
    ("curl_drops_the_courant_factor",
     ("float curl = dtdx * ((sf - f1) + (f2 - ss));",
      "float curl = ((sf - f1) + (f2 - ss));", -1), True, _always),

    # --- THE NULLS. Each must come back UNCAUGHT, and each has an arithmetic
    # --- reason, not an expectation.
    #
    # THE FUSION'S CENTRAL CLAIM, WRITTEN AS A PROGRAM. `Bx[idx] = value`
    # executes inside pml_apply_reg and the wall clear stores the same 0.0f it
    # put in the register, so `Bx[idx]` and `b_x` hold the same float32 word at
    # this point. A float32 stored to global and reloaded is the identity on the
    # bits. If this DIVERGES, the fusion changed arithmetic somewhere.
    ("seam_reloads_from_global",
     ("idx, b_x, pre_w_x", "idx, Bx[idx], pre_w_x", 1), False, _always),
    # float32 multiply is bitwise commutative.
    ("constitutive_product_commuted",
     (" + kps * src;", " + src * kps;", -1),
     False, _always),
    # The certified track's own null: fu_new was stored to fu[idx] one line above
    # and a 32-bit float in a 32-bit slot round-trips exactly.
    ("fu_reloaded_from_memory",
     ("+ fu_new) - fprev", "+ fu[idx]) - fprev", 1), False, _always),
    # THE COMPOSED PARITY IS A PRODUCT OF EXACT +/-1 WORDS, so its two factors
    # commute bitwise -- and so does the order the array path applies its two far
    # axes in, which is what carried_destinations' closed form asserts. A DECLARED
    # NULL: if this DIVERGES, the composition is order-dependent and the closed
    # form is not the array path's answer.
    ("composed_parity_factors_commuted",
     ("float gz_xy_p = (-phase_x) * (-phase_y);",
      "float gz_xy_p = (-phase_y) * (-phase_x);", 1), False, _two_folds),
    # The far ghost sits at the SAME coordinate on the indexed axis as the thread
    # that owns it, so re-reading the pair there reads the identical word.
    ("far_ghost_reloads_its_own_coefficient_pair",
     ("gx_y_i, gx_y_v, kps_x[i], kms_int_x[i]);",
      "gx_y_i, gx_y_v, kps_x[i + 0], kms_int_x[i + 0]);", 1), False,
     _folded_periodic_axis("y")),
)


def _integer_curl_tables(tables: Dict[str, Any], pml: Any) -> Dict[str, Any]:
    from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415
    return {"curl": step_curl_kernels.real_pml_curl_tables(pml, False),
            "constitutive": tables["constitutive"]}


def _half_integer_constitutive_tables(tables: Dict[str, Any], pml: Any) -> Dict[str, Any]:
    from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415
    return {"curl": tables["curl"],
            "constitutive": constitutive_kernels.real_constitutive_tables(pml, True)}


def _codes_all_periodic(codes: Sequence[Any]) -> Sequence[Any]:
    return tuple(np.int32(0) for _ in codes)


def _walls_dropped(walls: Sequence[int], grid: Any) -> Sequence[int]:
    return (0, 0, 0)


def _fills_dropped(fills: Dict[str, Any], grid: Any) -> Dict[str, Any]:
    """The two fills, told they do not run. What the product did until 2026-08-28."""
    return {"near": (0, 0, 0), "reflect": (-1, -1, -1), "phase": (0.0, 0.0, 0.0)}


def _fills_from_walls(fills: Dict[str, Any], grid: Any) -> Dict[str, Any]:
    """The NEAR flag read off the wall table instead of the mirror phases.

    THREE READINGS OF ONE GRID, and this is the confusion the launcher's docstring
    warns about: ``zero_metal_axes`` EXCLUDES a folded axis, so a plan built from it
    reports no fill on exactly the axes that have one.
    """
    from meep_gpu.cuda_kernels import in_seam_coverage  # noqa: PLC0415
    walls = in_seam_coverage.zero_metal_axes(grid)
    return {"near": tuple(int(bool(w)) for w in walls),
            "reflect": fills["reflect"], "phase": fills["phase"]}


def _fill_phases_flipped(fills: Dict[str, Any], grid: Any) -> Dict[str, Any]:
    """Every declared mirror plane's parity, negated on the HOST."""
    return {"near": fills["near"], "reflect": fills["reflect"],
            "phase": tuple(-value for value in fills["phase"])}


def _walls_from_boundary_codes(walls: Sequence[int], grid: Any) -> Sequence[int]:
    """The grid's metallic DECLARATION, which is what ``_boundary_kinds``
    resolves -- as opposed to ``zero_metal_axes``, which additionally excludes a
    FOLDED metallic axis.

    STILL A NULL AFTER THE FOLD WAS ADMITTED, FOR A DIFFERENT AND BETTER REASON,
    and the round measured its way to it rather than assuming it.

    The OLD reason: every folded axis was refused by name, so the two questions had
    the same answer on every grid this product could serve. That reason expired on
    2026-08-28.

    The reason NOW: on a folded METALLIC axis the two readings really do differ --
    this patch reports a wall there and ``zero_metal_axes`` does not -- and the
    difference REACHES NO WORD. ``_ZERO_METAL_ROWS`` is the B DIAGONAL, so a wall on
    axis ``a`` clears component ``a`` at stored 0 of ``a``; the NEAR fill images that
    same component at that same cell when ``a`` is folded (``iyee[Ba][a] == 0`` in
    both tables). That cell is therefore a fill DESTINATION, the emitted clear is
    guarded on ``own_a``, and the line does not fire. The two passes are mutually
    exclusive per component, which is the CUDA restatement of the Metal folded
    pair's "THE WALL CLEAR AND THE NEAR FILL CANNOT MEET".

    SCORED WHERE THE TWO READINGS DIFFER (``_folded_metallic``) rather than
    everywhere: a null confirmed on rows where the patch is the identity confirms
    nothing. What pins the GUARD this null rests on is host-side and deterministic:
    ``test_fused_pairs.test_the_one_re_spelled_pass_still_emits_the_diagonal_and_a_plus_zero``
    requires ``if (own_a && wall_a && ...)`` and refuses ``own_b && wall_a``."""
    return tuple(int(bool(grid.is_metallic(axis))) for axis in range(3))


#: ``(name, kwargs for run_case, must_be_caught, applies)``.
HOST_MUTATIONS: Tuple[Tuple[str, Dict[str, Any], bool,
                            Callable[[Dict[str, Any]], bool]], ...] = (
    ("curl_takes_the_integer_lattice",
     {"tables_patch": _integer_curl_tables}, True, _always),
    ("constitutive_takes_the_half_integer_lattice",
     {"tables_patch": _half_integer_constitutive_tables}, True, _always),
    ("boundary_codes_all_periodic",
     {"codes_patch": _codes_all_periodic}, True, _wall("x")),
    ("walls_dropped", {"walls_patch": _walls_dropped}, True, _any_wall),
    ("walls_from_boundary_codes",
     {"walls_patch": _walls_from_boundary_codes}, False, _folded_metallic),
    ("fills_dropped", {"fills_patch": _fills_dropped}, True, _folded),
    ("fills_from_walls", {"fills_patch": _fills_from_walls}, True, _folded),
    ("fill_phases_flipped", {"fills_patch": _fill_phases_flipped}, True, _folded),
)


def compile_source(source: str, options: Sequence[str]):
    from meep_gpu.cuda_kernels import fused_magnetic_pair as family  # noqa: PLC0415
    return cp.RawKernel(source, family.KERNEL_NAME, options=tuple(options))


def _verdict(caught: int, scored: int, must_be_caught: Optional[bool]) -> str:
    if not scored:
        return "NO LEGS"
    if must_be_caught is None:
        return "REPORTED"
    if must_be_caught:
        return "CAUGHT" if caught == scored else ("PARTIAL" if caught else "UNCAUGHT")
    return "NULL CONFIRMED" if caught == 0 else "NULL VIOLATED"


# ---------------------------------------------------------------------------
# The guard control
# ---------------------------------------------------------------------------

def leg_guard_control(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """The SHIPPED source with the contraction guard removed must DIVERGE.

    Both ``(fprev * kms) - curl`` and ``(f[idx] * kms_u) + fu_new`` are FMA
    candidates, and the array path rounds them twice. The certified pair reaches
    bit-identity ONLY under ``--fmad=false`` -- 120/120 with it, 0/120 without --
    so a guard control that came back IDENTICAL would mean this comparison
    cannot see a rounding change at all, and every "identical" above would be
    worth much less. It is scored on diverging, not merely reported.
    """
    from meep_gpu.cuda_kernels import fused_magnetic_pair as family  # noqa: PLC0415
    source = family.fused_magnetic_pair_source()
    kernel = compile_source(source, ())
    result = run_case(spec, "uniform", steps, kernel=kernel,
                      label_suffix="|unguarded")
    diverged = not result.get("bit_identical", True)
    return {
        "passed": bool(diverged and result.get("launch_counts", {}).get("agree")),
        "guard_removed": list(GUARD_OPTIONS),
        "diverged": diverged,
        "first_divergence": result.get("first_divergence"),
        "differing_words": result.get("differing_words"),
        "differing_arrays": result.get("differing_arrays"),
        "reading": ("the comparison IS sensitive to float32 contraction: the same "
                    "source without --fmad=false diverges from the array path"
                    if diverged else
                    "the unguarded source did NOT diverge; either NVRTC contracted "
                    "nothing here or this comparison cannot see a rounding change"),
        "case": result,
    }


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def save(results: Dict[str, Any], out_path: str) -> None:
    gate_provenance.stamp(results)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=False, default=str)
    os.replace(tmp, out_path)


def run_product(results: Dict[str, Any], out_path: str, specs: Sequence[Dict[str, Any]],
                classes: Sequence[str], steps: int) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    total = len(specs) * len(classes)
    for index, spec in enumerate(specs):
        for value_class in classes:
            case = run_case(spec, value_class, steps)
            cases.append(case)
            results["product"] = cases
            save(results, out_path)
            log(f"[product] {len(cases)}/{total} {spec['label']} {value_class} "
                f"shape={case.get('shape')} steps={case.get('steps_run')} "
                f"identical={case.get('bit_identical')} "
                f"diff={case.get('differing_words')} "
                f"launches={case.get('launch_counts', {}).get('launcher_reports')} "
                f"unmoved={case.get('movement_floor', {}).get('unmet')} "
                f"subnormal_operands={case.get('operand_census', {}).get('subnormals')} "
                f"({case.get('seconds', 0):.1f} s)")
    return cases


def run_mutations(results: Dict[str, Any], out_path: str, steps: int,
                  specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from meep_gpu.cuda_kernels import fused_magnetic_pair as family  # noqa: PLC0415

    base = family.fused_magnetic_pair_source()
    out: Dict[str, Any] = {}
    for name, (old, new, count), must_be_caught, applies in SOURCE_MUTATIONS:
        mutated, sites = needle(base, old, new, count)
        if not sites or mutated == base:
            out[name] = {"armed": False, "sites": sites,
                         "why": "the needle matched nothing; the mutation and the "
                                "kernel have drifted apart"}
            log(f"[src-mut] {name}: NOT ARMED")
            results["source_mutations"] = out
            save(results, out_path)
            continue
        try:
            kernel = compile_source(mutated, GUARD_OPTIONS)
        except Exception as exc:  # noqa: BLE001 - a compile failure is a result
            out[name] = {"armed": False, "sites": sites,
                         "why": f"the mutated source did not compile: "
                                f"{type(exc).__name__}: {exc}"[:400]}
            log(f"[src-mut] {name}: DID NOT COMPILE")
            results["source_mutations"] = out
            save(results, out_path)
            continue
        legs, inert_on = [], []
        for spec in specs:
            if not applies(spec):
                inert_on.append(spec["label"])
                continue
            legs.append(run_case(spec, "uniform", steps, kernel=kernel,
                                 label_suffix=f"|{name}"))
        caught = sum(1 for leg in legs if not leg.get("bit_identical", True))
        launched = all(leg.get("launch_counts", {}).get("launcher_reports")
                       for leg in legs)
        out[name] = {
            "armed": True, "sites": sites,
            "mutated_source_sha256": hashlib.sha256(mutated.encode()).hexdigest(),
            "ran": len(legs), "caught": caught,
            "must_be_caught": must_be_caught,
            "not_applicable_on": inert_on,
            "every_leg_launched": bool(launched),
            "verdict": _verdict(caught, len(legs), must_be_caught),
            "first_divergences": [leg.get("first_divergence") for leg in legs],
            "cases": legs,
        }
        if not launched:
            out[name]["verdict"] = "UNACCOUNTED"
            out[name]["why"] = ("a leg reported no launch; that leg did not "
                                "exercise the mutation")
        log(f"[src-mut] {name}: caught {caught}/{len(legs)} -> {out[name]['verdict']}")
        results["source_mutations"] = out
        save(results, out_path)
    return out


def run_host_mutations(results: Dict[str, Any], out_path: str, steps: int,
                       specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, kwargs, must_be_caught, applies in HOST_MUTATIONS:
        legs, inert_on = [], []
        for spec in specs:
            if not applies(spec):
                inert_on.append(spec["label"])
                continue
            legs.append(run_case(spec, "uniform", steps,
                                 label_suffix=f"|{name}", **kwargs))
        caught = sum(1 for leg in legs if not leg.get("bit_identical", True))
        out[name] = {
            "ran": len(legs), "caught": caught,
            "must_be_caught": must_be_caught,
            "not_applicable_on": inert_on,
            "verdict": _verdict(caught, len(legs), must_be_caught),
            "first_divergences": [leg.get("first_divergence") for leg in legs],
            "cases": legs,
        }
        log(f"[host-mut] {name}: caught {caught}/{len(legs)} -> {out[name]['verdict']}")
        results["host_mutations"] = out
        save(results, out_path)
    return out


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def summarize(results: Dict[str, Any], steps: int) -> Dict[str, Any]:
    reasons: List[str] = []
    product = results.get("product", [])
    if not product:
        reasons.append("no product case was scored at all")
    for case in product:
        if not case.get("passed"):
            reasons.append(
                f"product case {case.get('case_seed_label')} failed: "
                f"identical={case.get('bit_identical')} "
                f"first_divergence={case.get('first_divergence')} "
                f"unmet_movement={case.get('movement_floor', {}).get('unmet')} "
                f"read_only_moved={case.get('read_only_volumes_that_moved')} "
                f"launches={case.get('launch_counts')}")

    # BOTH VALUE CLASSES, and the precondition must DISCRIMINATE. A census that
    # fires everywhere is a detector that cannot tell the classes apart, and one
    # that never fires is a detector never shown to work.
    classes = {case["value_class"] for case in product}
    if classes != set(VALUE_CLASSES):
        reasons.append(f"value classes scored were {sorted(classes)}, not "
                       f"{sorted(VALUE_CLASSES)}")
    band = [c for c in product if c["value_class"] == "subnormal_band"]
    uniform = [c for c in product if c["value_class"] == "uniform"]
    band_fired = [c for c in band if c.get("operand_census", {}).get("subnormals", 0)]
    uniform_clean = [c for c in uniform
                     if c.get("operand_census", {}).get("subnormals", 0) == 0]
    if band and len(band_fired) != len(band):
        reasons.append("a subnormal-band case held no subnormal operand; the "
                       "policy legs measured the fixture, not the policy")
    if uniform and len(uniform_clean) != len(uniform):
        reasons.append("a uniform case held a subnormal operand; the precondition "
                       "cannot discriminate between the two classes")

    # EVERY WALL AXIS MUST HAVE BEEN LIVE somewhere. The B table is the diagonal,
    # and a sweep that only ever walls x cannot tell it from the D-side one.
    walls = set()
    for case in product:
        walls.update(a for a, w in enumerate(case.get("zero_metal_axes", []))
                     if w)
    if walls != {0, 1, 2}:
        reasons.append(f"only wall axes {sorted(walls)} were ever live; the "
                       f"diagonal wall table is untested on the rest")

    # THE FOLD FLOOR, NEW ON 2026-08-28 AND THE SAME ARGUMENT AS THE WALL ONE.
    # This round's whole content is that the two mirror fills are now CARRIED, and
    # on an unfolded grid neither of them runs at all -- so a sweep of the original
    # rows would release the carry without executing one line of it. Four
    # properties, each of which no other row supplies:
    #   * both fills live on every axis (the near fill's plane is the component's
    #     own axis and the far fill's the other two, so a sweep folding only Y
    #     leaves the X and Z destination planes unwritten);
    #   * BOTH fold terminations (a folded METALLIC axis runs the near fill and not
    #     the far one);
    #   * BOTH declared parities (mirror_parity is phase * (1 - 2*iyee), so a
    #     kernel that baked the signs is exact at phase +1);
    #   * TWO folded axes at once, which is the only shape in which the composed
    #     parity exists to be got wrong.
    near_axes, far_axes, phases, fold_counts = set(), set(), set(), set()
    for case in product:
        plan = case.get("fill_plan") or {}
        near_axes.update(a for a, live in enumerate(plan.get("near", ())) if live)
        far_axes.update(a for a, row in enumerate(plan.get("reflect", ())) if row >= 0)
        phases.update(value for value in plan.get("phase", ()) if value)
        fold_counts.add(sum(1 for live in plan.get("near", ()) if live))
    if near_axes != {0, 1, 2}:
        reasons.append(f"fill_symmetry_bc_B was only ever live on axes "
                       f"{sorted(near_axes)}; the near carry is untested on the rest")
    if far_axes != {0, 1, 2}:
        reasons.append(f"fill_folded_far_ghosts_B was only ever live on axes "
                       f"{sorted(far_axes)}; the far carry is untested on the rest")
    if phases != {1.0, -1.0}:
        reasons.append(f"mirror phases exercised were {sorted(phases)}; a kernel "
                       f"that baked the parity signs is exact at +1 alone")
    if not any(count >= 2 for count in fold_counts):
        reasons.append("no case folded two axes at once, so the composed parity "
                       "a corner carries was never formed")
    if not any(sum(1 for live in (case.get("fill_plan") or {}).get("near", ())
                   if live)
               > sum(1 for row in (case.get("fill_plan") or {}).get("reflect", ())
                     if row >= 0)
               for case in product):
        reasons.append("no case carried a folded METALLIC axis (near fill live, "
                       "far fill absent); one termination was never exercised")

    for leg_name in ("driver_order", "lift", "refusal", "nonlinear_widening"):
        leg = results.get(leg_name, {})
        if not leg.get("passed"):
            reasons.append(f"leg {leg_name} failed: "
                           f"{json.dumps(leg, default=str)[:400]}")

    # THE NONLINEAR ROWS MUST HAVE BEEN SWEPT. The widening is released only with
    # cases that ran MEEP's Pade branch beside the weld; a product list without
    # one would release an admission the sweep never exercised.
    if not any(str(case.get("case_seed_label", "")).startswith(
            ("wall_z_nonlinear|", "all_periodic_nonlinear|")) for case in product):
        reasons.append("no nonlinear product case was scored; the 2026-09-01 "
                       "widening would be released unexercised")

    # THE DEPOSIT IS PART OF THE RELEASE, not a side observation: the predicate now
    # ADMITS every in-seam magnetic deposit, so a run that did not measure one would
    # be releasing an admission it never exercised.
    deposit = results.get("deposit", {})
    if not deposit:
        reasons.append("no deposit case ran; the predicate admits an in-seam "
                       "magnetic source and nothing here measured one")
    for name, leg in deposit.items():
        if not float(leg.get("deposit_magnitude") or 0.0) > 0.0:
            reasons.append(f"deposit case {name} deposited exactly zero; the walk "
                           f"compared a no-op with a no-op")
        if not leg.get("passed"):
            reasons.append(f"deposit case {name} failed: "
                           f"identical={leg.get('bit_identical')} "
                           f"repaired={leg.get('deposit_points_repaired')} "
                           f"plans={leg.get('installed_plans')} "
                           f"words={leg.get('differing_words')}")
    nulls = results.get("deposit_null_control", {})
    if not nulls:
        reasons.append("no unbracketed null control ran; every deposit case above "
                       "is consistent with a deposit too small to see")
    for label, null in nulls.items():
        if not null.get("passed"):
            reasons.append(
                f"the UNBRACKETED launch on {label} did not diverge: every deposit "
                f"case there is consistent with a deposit too small to see "
                f"({json.dumps(null, default=str)[:300]})")

    # THE SECOND CONSULT SITE IS PART OF THE RELEASE. A product certified only
    # through `step` is certified on one of the two sites the driver dispatches it
    # from, and the other one restores magnetic names only.
    synchronize = results.get("synchronize", {})
    if not synchronize:
        reasons.append("no synchronize case ran; the driver consults this product "
                       "from synchronize_magnetic_fields too and nothing measured it")
    for name, leg in synchronize.items():
        if not leg.get("passed"):
            reasons.append(
                f"the magnetic half-step on {name} failed: "
                f"after_half_step={leg.get('differing_after_the_half_step')} "
                f"after_restore={leg.get('differing_after_the_restore')} "
                f"not_returned={leg.get('arrays_the_restore_did_not_return')} "
                f"controls={leg.get('controls_that_must_diverge')}")

    guard = results.get("guard_control", {})
    if not guard.get("passed"):
        reasons.append(f"the contraction-guard control did not diverge: "
                       f"{guard.get('reading')}")

    for name, leg in results.get("separate_control", {}).items():
        if not leg.get("passed"):
            reasons.append(f"separate-composition control {name} failed: "
                           f"identical={leg.get('three_way_identical')} "
                           f"counts_agree={leg.get('launch_counts_agree')} "
                           f"separate={leg.get('separate_launches_total')} "
                           f"fused={leg.get('fused_launches_total')}")

    for name, leg in results.get("source_mutations", {}).items():
        if not leg.get("armed"):
            reasons.append(f"source mutation {name} was not armed: {leg.get('why')}")
        elif leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED", "REPORTED"):
            reasons.append(f"source mutation {name} is {leg['verdict']} "
                           f"({leg.get('caught')}/{leg.get('ran')})")
    for name, leg in results.get("host_mutations", {}).items():
        if leg["verdict"] not in ("CAUGHT", "NULL CONFIRMED"):
            reasons.append(f"host mutation {name} is {leg['verdict']} "
                           f"({leg.get('caught')}/{leg.get('ran')})")

    disarm = results.get("disarm", {})
    if not disarm.get("passed"):
        reasons.append("the disarm leg diverged with the SHIPPED bytes; every "
                       "'caught' above is a harness that diverges on its own")

    armed = results.get("source_mutations", {})
    return {
        "released": not reasons,
        "reasons": reasons,
        "steps_per_case": steps,
        "product_cases": len(product),
        "product_identical": sum(1 for c in product if c.get("bit_identical")),
        "value_classes": sorted(classes),
        "wall_axes_exercised": sorted(walls),
        "synchronize_cases": len(synchronize),
        "synchronize_cases_identical": sum(
            1 for leg in synchronize.values()
            if not leg.get("differing_after_the_restore")),
        "deposit_cases": len(deposit),
        "deposit_cases_identical": sum(1 for leg in deposit.values()
                                       if leg.get("bit_identical")),
        "deposit_points_repaired": sum(int(leg.get("deposit_points_repaired") or 0)
                                       for leg in deposit.values()),
        "deposit_null_controls_diverged": {
            label: bool(leg.get("diverged")) for label, leg in nulls.items()},
        "near_fill_axes_exercised": sorted(near_axes),
        "far_fill_axes_exercised": sorted(far_axes),
        "mirror_phases_exercised": sorted(phases),
        "mutations_armed": sum(1 for leg in armed.values() if leg.get("armed")),
        "mutations_must_be_caught": sum(
            1 for leg in armed.values()
            if leg.get("armed") and leg.get("must_be_caught")),
        "nulls_declared": sum(1 for leg in armed.values()
                              if leg.get("armed") and leg.get("must_be_caught") is False)
        + sum(1 for leg in results.get("host_mutations", {}).values()
              if leg.get("must_be_caught") is False),
        "host_mutations": len(results.get("host_mutations", {})),
        "claim": (f"the shipped CUDA fused magnetic pair "
                  f"(step_B -> zero_metal_B -> update_H in ONE launch) leaves "
                  f"every stored volume BYTE-IDENTICAL to stepping's own eleven-"
                  f"pass driver step, per complete step over {steps} consecutive "
                  f"steps, on every configuration this sweep scored, under the "
                  f"float32 subnormal policy this run installed"),
        "does_not_claim": [
            "NO THROUGHPUT. This gate times nothing and licenses no speed claim.",
            "NO DISPATCH. Nothing in meep_gpu imports cuda_kernels; this product "
            "is not registered on any arm and a verdict here does not wire it.",
            "NOTHING ABOUT A FOLDED, COMPLEX, CYLINDRICAL, CONDUCTIVE, BFAST or "
            "SPECIAL-KZ run: each is refused BY NAME and none was swept.",
            "THE IN-SEAM MAGNETIC DEPOSIT IS NOW CARRIED RATHER THAN REFUSED, and "
            "what the `deposit` legs settle is the TWO-CONSULT PROTOCOL around the "
            "launch -- the fused kernel against an uninjected field, the driver's "
            "own inject/fill/clear, then the repair -- byte-identical to the array "
            "path over complete driver steps, with the unbracketed launch diverging. "
            "It says nothing about a deposit whose index the engine never resolved, "
            "which is refused by name, and nothing about the D/E seam's deposit, "
            "which has no CUDA product.",
            "NOTHING ABOUT THE D/E SEAM, which is a different product with the "
            "opposite Yee sub-lattice pairing.",
            "NO VERDICT ABOUT THE OTHER TWO IN-SEAM PASSES. fill_symmetry_bc_B "
            "and fill_folded_far_ghosts_B are refused by this predicate, not "
            "carried, and this run says nothing about carrying them.",
            "THE ORACLE IS stepping ON CuPy, NOT CPU MEEP. This is a "
            "byte-identity claim against the package's own array path -- the "
            "same oracle both halves were certified against -- and it is not a "
            "physics claim against stock MEEP. What it settles is that the weld "
            "changed nothing; what MEEP-parity the array path already has, it "
            "keeps, and nothing here extends it.",
            "THE MUTATION LEGS RUN ON THE UNIFORM CLASS ONLY. A mutation leg "
            "answers 'can this gate see this defect at all'; multiplying it by "
            "the value classes buys repetitions of that answer rather than a "
            "second question, and the product legs carry both classes.",
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="artifact FILE path")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--product", choices=("full", "reduced"), default="full")
    parser.add_argument("--subnormal-policy", choices=("keep", "flush"), default=None)
    parser.add_argument("--import-meep-for-host-policy", action="store_true")
    parser.add_argument("--skip-mutations", action="store_true")
    args = parser.parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be positive")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    started = time.perf_counter()
    results: Dict[str, Any] = {
        "gate": "cuda_fused_magnetic_pair",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "question": ("does ONE launch of fused_magnetic_pair_pml_real leave every "
                     "stored volume byte-identical to stepping's step_B -> "
                     "zero_metal_B -> update_H inside a complete driver step?"),
        "budget_steps": args.steps,
        "movement_floor_note": MOVEMENT_FLOOR_NOTE,
    }
    if cp is None:
        log("[fatal] CuPy did not import; this gate needs a device")
        results["status"] = "refused: no CuPy"
        save(results, args.out)
        return 2

    if args.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    # THE OBSERVER GOES IN BEFORE THE POLICY: under 'keep' the policy's strip
    # wraps it, so it records the option tuple NVRTC was really given.
    results["nvrtc_observer"] = probe.install_nvrtc_binary_observer()
    if args.subnormal_policy:
        results["subnormal_policy_install"] = probe.install_subnormal_policy_for_run(
            args.subnormal_policy, _REPO_API)
    results["environment"] = probe.device_info()
    results["subnormal_policy_stamp"] = probe.subnormal_policy_stamp(_REPO_API)
    save(results, args.out)

    specs = SPECS if args.product == "full" else [
        spec for spec in SPECS if spec["label"] in REDUCED_LABELS]
    classes = VALUE_CLASSES
    by_label = {spec["label"]: spec for spec in SPECS}
    mutation_specs = [by_label[label] for label in MUTATION_SPEC_LABELS]

    results["driver_order"] = leg_driver_order()
    log(f"[driver] order matches driver.py: {results['driver_order']['passed']}")
    results["lift"] = leg_lift()
    log(f"[lift] both halves are the certified bytes: {results['lift']['passed']}")
    results["refusal"] = leg_refusal()
    log(f"[refusal] the seam clauses hold: {results['refusal']['passed']}")
    results["nonlinear_widening"] = leg_nonlinear_widening()
    log(f"[nonlinear] widening measured: "
        f"{results['nonlinear_widening']['passed']} "
        f"chi_moved={results['nonlinear_widening']['chi_moved_words_in_one_array_step']} "
        f"selected={results['nonlinear_widening']['selected_unfused']}")
    save(results, args.out)

    _warm_memo(specs)
    run_product(results, args.out, specs, classes, args.steps)

    results["separate_control"] = {}
    for label in SEPARATE_CONTROL_LABELS:
        leg = leg_separate_control(by_label[label], args.steps)
        results["separate_control"][label] = leg
        save(results, args.out)
        log(f"[separate] {label}: three-way identical="
            f"{leg['three_way_identical']} separate={leg['separate_launches_total']} "
            f"fused={leg['fused_launches_total']} "
            f"removed_per_step={leg['launches_removed_per_step']}")

    # THE MAGNETIC HALF-STEP -- the driver's SECOND consult site, which no leg on
    # this backend had driven. Run on the separate-control rows, which span a walled
    # and an unwalled specialisation, because the wall clear is carried in-launch
    # here and the half-step is the one site with no host pass on top of it.
    results["synchronize"] = {}
    for label in SEPARATE_CONTROL_LABELS:
        leg = leg_synchronize(by_label[label], "uniform", args.steps)
        results["synchronize"][label] = leg
        save(results, args.out)
        log(f"[synchronize] {label}: passed={leg.get('passed')} "
            f"after_half_step={leg.get('differing_after_the_half_step')} "
            f"after_restore={leg.get('differing_after_the_restore')} "
            f"controls={leg.get('controls_that_must_diverge')}")

    # THE DEPOSIT LEGS. Two carried rows -- a POINT source and an EXTENDED one, so
    # the repair's per-point loop runs at many points and not only at four -- and the
    # unbracketed null control, which must diverge.
    results["deposit"] = {}
    for label in DEPOSIT_SPEC_LABELS:
        for tag, component, size in deposit_rows(by_label[label]):
            leg = run_deposit_case(by_label[label], "uniform", args.steps,
                                   repair=True, component=component, size=size)
            results["deposit"][f"{label}|{component}|{tag}"] = leg
            save(results, args.out)
            log(f"[deposit] {label}|{component}|{tag}: "
                f"identical={leg.get('bit_identical')} "
                f"points={leg.get('deposit_points')} "
                f"repaired={leg.get('deposit_points_repaired')} "
                f"plans={leg.get('installed_plans')}")
    # ONE NULL CONTROL PER DEPOSIT ROW, and the folded one is the point: without it
    # a green folded deposit case would be consistent with a fill closure the repair
    # never needed.
    results["deposit_null_control"] = {
        label: leg_deposit_null_control(by_label[label], args.steps)
        for label in DEPOSIT_SPEC_LABELS}
    save(results, args.out)
    for label, leg in results["deposit_null_control"].items():
        log(f"[deposit] null control {label} diverged={leg['diverged']} at step "
            f"{leg['first_divergence']}")

    results["guard_control"] = leg_guard_control(by_label["wall_xyz"], args.steps)
    save(results, args.out)
    log(f"[guard] unguarded source diverged="
        f"{results['guard_control']['diverged']} at step "
        f"{results['guard_control']['first_divergence']}")

    if not args.skip_mutations:
        run_mutations(results, args.out, args.steps, mutation_specs)
        run_host_mutations(results, args.out, args.steps, mutation_specs)

    # THE DISARM CHECK. The identical harness, the identical cases, the SHIPPED
    # bytes and no patch. A divergence here would mean every "caught" above is a
    # harness that diverges on its own.
    disarm_cases = [run_case(spec, "uniform", args.steps, label_suffix="|disarm")
                    for spec in mutation_specs]
    results["disarm"] = {
        "passed": all(case.get("passed") for case in disarm_cases),
        "cases": disarm_cases,
        "differing_words": [case.get("differing_words") for case in disarm_cases],
    }
    log(f"[disarm] shipped bytes on the mutation specs: "
        f"{results['disarm']['passed']}")

    results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["source_sha256"] = {
        "gate": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "family": hashlib.sha256(
            open(os.path.join(_REPO_API, "meep_gpu", "cuda_kernels",
                              "fused_magnetic_pair.py"), "rb").read()).hexdigest(),
    }
    results["summary"] = summarize(results, args.steps)
    results["elapsed_seconds"] = time.perf_counter() - started
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(results, args.out)

    verdict = results["summary"]
    log(f"[verdict] released={verdict['released']} "
        f"product={verdict['product_identical']}/{verdict['product_cases']} "
        f"policy={results.get('subnormal_policy_stamp', {}).get('policy')} "
        f"in {results['elapsed_seconds']:.1f}s -> {args.out}")
    for reason in verdict["reasons"]:
        log(f"[verdict]   - {reason}")
    return 0 if verdict["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
