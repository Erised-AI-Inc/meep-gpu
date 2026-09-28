"""Byte-identity gate for the hand-CUDA fused pair on the ELECTRIC seam.

WHAT IS UNDER TEST. ``meep_gpu/cuda_kernels/fused_electric_pair.py``: ONE kernel
performing three driver passes in one launch::

    step_D (driver.py:3302) -> zero_metal_D (:3310) -> update_E (:3313)

It is the FIRST product this track has put on ``D_to_E``, and what it is worth is
not the launch count. The board prices that seam at 29 reachable seam-instances of
186 because the driver deposits the electric sources INSIDE it and no D-side product
brackets its launch; this one declares ``CARRIES_DEPOSIT_REPAIR`` and is wired
through ``fused_pairs``' two slots, so 75 of its cell's 79 rows stop being blocked by
the injection. THE DEPOSIT LEGS ARE THEREFORE THE POINT OF THIS GATE, not a side
observation: an admission nothing exercised is an admission nobody measured.

=============================================================================
WHY THE COMPARISON IS PER COMPLETE DRIVER STEP
=============================================================================

The claim is about the SEAM between the halves -- the flux density never leaving a
register -- so a per-sub-step comparison could not see it at all. Two engines are
built from ONE seed and stepped side by side: the reference runs ``stepping``'s own
eleven-pass sequence, the subject runs the fused launch plus the eight passes it does
not replace, and the two are compared as raw uint32 WORDS over all twenty-four stored
volumes after EVERY step. The B/H half runs on the array path on both sides, so a
corrupted E reaches B on the very next step; the first divergent step is reported
rather than a final pass/fail.

=============================================================================
THE NINE THINGS THIS GATE REFUSES TO LET PASS SILENTLY
=============================================================================

1. **A step this gate invented.** :func:`leg_driver_order` reads ``FdtdDriver.step``'s
   own source and extracts the ordered list of ``stepping`` calls it makes. The
   oracle walks THAT list, and the product's ``REPLACES`` is asserted to be a subset
   of it in that order.

2. **A silent fallback.** Bytes alone cannot prove the fused kernel ran: an engine
   that never launched it is byte-identical to the oracle BY CONSTRUCTION, because
   the oracle is the array path the same engine would otherwise take. So every case
   asserts EXACT launch counts from TWO INDEPENDENT COUNTERS -- a proxy installed
   over the shipped compile memo (``compile_cache._compiled_kernels``, the seam every
   ``_get_kernel`` route passes through) and the launcher's own returned report --
   and requires them to agree, to equal the step budget, and to name ONLY the fused
   kernel.

3. **A vacuous comparison.** Two floors per volume. WHAT MUST MOVE: on the uniform
   class every one of the twenty-four stored volumes must differ from its seeded
   value. WHAT MUST NOT MOVE: the read-only volumes -- the PML coefficient vectors
   and the three inverse-permittivity volumes -- must be bit-unchanged. Every field
   and table argument in this signature is ``__restrict__``, and a kernel that wrote
   through a ``const`` binding is UB NVRTC does not diagnose.

4. **An unexercised policy.** ``operand_census`` counts the subnormals and signed
   zeros the operands REALLY hold. The band class must contain subnormals and the
   uniform class must contain none -- a precondition that fires everywhere
   discriminates nothing, and one that never fires is a detector never shown to work.

5. **A wall table taken from the wrong family.** ``zero_metal_D`` is the OFF-DIAGONAL
   complement of ``zero_metal_B``'s diagonal -- TWO components per wall, and never
   the component whose own axis is walled. Reusing the B table is the single most
   likely slip in porting this product from its magnetic sibling, so it is armed as a
   mutation AND asserted in the emitted text, and the sweep walls every axis
   somewhere so a one-wall sweep cannot mistake one table for the other.

6. **A sub-lattice shadow.** The D curl reads the INTEGER coefficient vectors and
   ``update_E`` the HALF-INTEGER ones -- the MIRROR IMAGE of the B/H seam's pairing.
   Both compile, both run, and they differ by half a cell in the absorber profile.
   Armed from both sides as host mutations, and the emitted rename is asserted.

7. **An isotropic fixture hiding a per-component binding.** ``update_E`` binds THREE
   inverse-permittivity pointers, and on an isotropic run all three are the SAME
   allocation -- so a kernel that read ``inv_eps_Ez`` for all three would be
   byte-identical on every isotropic row. Half the sweep is DIAGONALLY ANISOTROPIC
   for that reason, and the component-swap mutation is scored only there.

8. **A predicate that admits what the launch cannot serve.** :func:`leg_refusal`
   builds the configurations the module refuses BY NAME -- a folded grid naming both
   uncarried fills, a conductivity naming the driver's other injection route, an
   electric source with no publishable index, an undeclared source set, a complex
   run -- and requires the refusal with the reason recorded. This is the one failure
   mode no bit-comparison catches.

9. **A deposit computed against a pre-injection field.** :func:`run_deposit_case` and
   :func:`leg_deposit_null_control` are what the declared repair is worth. Complete
   driver steps with a real ELECTRIC source in the seam -- a point deposit and an
   extended one, on a walled and an unwalled grid -- with the fused launch consuming
   an UNINJECTED field, the driver's own inject/fill/clear between the two consults,
   and the repair recomputing the deposit points at the second. The null control asks
   the shipped composer for the SAME plan with an empty source list, injects anyway,
   and MUST diverge.

=============================================================================
WHAT THIS GATE DOES NOT CLAIM
=============================================================================

No throughput claim of any kind. No dispatch claim: nothing in ``meep_gpu`` imports
``cuda_kernels``, and a verdict here does not wire this product to anything. No claim
about a FOLDED, complex, cylindrical, conductive, BFAST, special-kz, dispersive or
off-diagonal run -- every one is refused BY NAME and none was swept. And no verdict
about the B/H seam, which is a different product with the opposite sub-lattice
pairing and its own gate.

=============================================================================
RUNNING IT
=============================================================================

ONE verified-empty GPU, pinned by UUID. ``CUPY_ACCELERATORS`` must be EMPTY IN THE
ENVIRONMENT before CuPy imports, and the CuPy disk cache must be private per policy
because CuPy's cache key is computed ABOVE the strip seam::

    CUDA_VISIBLE_DEVICES=<uuid> CUPY_ACCELERATORS= \\
      CUPY_CACHE_DIR=$OUT/cupy_cache/ftz_stripped \\
      python -u gate_cuda_fused_electric_pair.py \\
        --subnormal-policy keep --out $OUT/keep/gate.json

    CUDA_VISIBLE_DEVICES=<uuid> CUPY_ACCELERATORS= \\
      CUPY_CACHE_DIR=$OUT/cupy_cache/native \\
      python -u gate_cuda_fused_electric_pair.py \\
        --subnormal-policy flush --import-meep-for-host-policy \\
        --out $OUT/flush/gate.json

``--out`` is a FILE. One flushed line per case (the progress-reporting rule); the artifact is
rewritten atomically after every case, so an interrupted run keeps everything up to
the failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
except ImportError:  # laptop: only the source legs can run
    cp = None

import probe_fused_kernel_bit_identity as probe  # noqa: E402
import gate_provenance  # noqa: E402

from meep_gpu import deposit_repair, stepping  # noqa: E402

log = probe.log
to_host = probe.to_host
operand_census = probe.operand_census
subnormal_band_hosts = probe.subnormal_band_hosts

#: Per-case RNG seed base. Seeded from a sha256 of the case's OWN label, never from
#: ``hash()``: ``PYTHONHASHSEED`` salts the hash of a string, so a hash-seeded case
#: could not be replayed from the record that names it.
SEED = 20260830

#: Complete DRIVER STEPS every product leg runs. Sixty, the budget every hand-CUDA
#: record on this track is cut at -- "identical for N steps" is a claim about N.
STEPS = 60

#: Every stored volume a complete step can touch, and every one is compared. The B/H
#: half is here even though this product does not touch it, because a pair that
#: corrupted E reaches B through ``step_B`` on the very next step.
STATE_NAMES: Tuple[str, ...] = tuple(
    [f"{stem}{axis}" for stem in ("B", "D", "E", "H") for axis in "xyz"]
    + [f"fu_{stem}{axis}" for stem in ("B", "D") for axis in "xyz"]
    + [f"f_w_{stem}{axis}" for stem in ("E", "H") for axis in "xyz"])

#: The volumes that must be BIT-UNCHANGED by a run: the PML coefficient vectors on
#: both Yee sub-lattices. The three inverse-permittivity volumes are added per case by
#: :func:`read_only_snapshot`, because which arrays those are depends on whether the
#: row is isotropic (one allocation bound three times) or anisotropic (three).
IMMUTABLE_PML: Tuple[str, ...] = tuple(
    f"{name}_{axis}{suffix}"
    for axis in "xyz" for name in ("kms", "sinv", "kps") for suffix in ("", "_h"))

#: The families the SUBNORMAL BAND class is held to. Declared before the first run.
BAND_MUST_MOVE: Tuple[str, ...] = tuple(
    [f"D{axis}" for axis in "xyz"] + [f"fu_D{axis}" for axis in "xyz"]
    + [f"f_w_E{axis}" for axis in "xyz"])

MOVEMENT_FLOOR_NOTE = (
    "The uniform class is held to the full movement floor: all twenty-four stored "
    "volumes must differ from their seeded values. The subnormal-band class is held "
    "to BAND_MUST_MOVE only, and the reason is arithmetic and was written down "
    "before the first run: under 'flush' a state seeded entirely in the band drives "
    "constitutive_apply to f[idx] = (f[idx] + 0) - 0, since kps*src and kms*prev "
    "both flush, so E is the identity and need not move. D, fu_D and f_w_E are "
    "written unconditionally by the curl half and by the constitutive store, so they "
    "must move under either policy. Arrays that did not move are RECORDED for both "
    "classes either way.")

VALUE_CLASSES: Tuple[str, ...] = ("uniform", "subnormal_band")

#: NVRTC options the shipped module compiles with, restated so the guard control can
#: drop them. NOT imported from the module: a control that read the module's own tuple
#: would compile the same thing twice if that tuple were ever emptied.
GUARD_OPTIONS: Tuple[str, ...] = ("--fmad=false",)


# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------
#
# THE EXTENTS ARE DELIBERATELY UNEQUAL (9, 10, 11). A cube lets an index
# decomposition swap i and k and stay bit-identical, and this kernel decomposes a
# flat thread index into i/j/k and indexes THREE coefficient vectors of three
# different lengths with them.
#
# EVERY AXIS IS WALLED SOMEWHERE and one row is walled NOWHERE. The wall table this
# product carries is the D OFF-DIAGONAL -- Dy and Dz on x, Dx and Dz on y, Dx and Dy
# on z -- so a row that walled only one axis cannot tell it from the B-side diagonal,
# and the all-periodic row proves the family is not wall-dependent.
#
# HALF THE ROWS ARE DIAGONALLY ANISOTROPIC. update_E binds THREE inverse-permittivity
# pointers, and an isotropic run hands the same allocation three times -- so a kernel
# that read one of them for all three components would be BYTE-IDENTICAL on every
# isotropic row. Both shapes are swept because both ship: the signature deliberately
# does NOT promise those three do not alias (see the module's own note), so the
# isotropic rows are the ones that exercise the aliasing the certified kernel allows.
#
# NO ROW IS FOLDED, COMPLEX, CONDUCTIVE, DISPERSIVE, OFF-DIAGONAL, NONLINEAR OR
# SOURCE-DRIVEN in the product sweep. Every one of those is refused BY NAME by the
# predicate, and leg_refusal measures the refusals rather than the sweep pretending to
# cover them; the deposit legs are where a source-driven run is measured.

CELL: Tuple[float, float, float] = (9.0, 10.0, 11.0)

SPECS: Tuple[Dict[str, Any], ...] = (
    {"label": "all_periodic", "boundaries": ("periodic", "periodic", "periodic"),
     "pml": 2, "cell": CELL, "epsilon": "isotropic"},
    {"label": "wall_x", "boundaries": ("metallic", "periodic", "periodic"),
     "pml": 2, "cell": CELL, "epsilon": "isotropic"},
    {"label": "wall_y", "boundaries": ("periodic", "metallic", "periodic"),
     "pml": 2, "cell": CELL, "epsilon": "anisotropic"},
    {"label": "wall_z", "boundaries": ("periodic", "periodic", "metallic"),
     "pml": 2, "cell": CELL, "epsilon": "anisotropic"},
    {"label": "wall_xz", "boundaries": ("metallic", "periodic", "metallic"),
     "pml": 2, "cell": CELL, "epsilon": "isotropic"},
    {"label": "wall_xyz", "boundaries": ("metallic", "metallic", "metallic"),
     "pml": 2, "cell": CELL, "epsilon": "anisotropic"},
    # A THINNER LAYER on a different shape: the absorber profile changes, so a
    # coefficient index that happened to land inside a flat region of a 2-cell layer
    # has a different table to be wrong about.
    {"label": "wall_xy_thin_pml",
     "boundaries": ("metallic", "metallic", "periodic"),
     "pml": 1, "cell": (12.0, 9.0, 10.0), "epsilon": "anisotropic"},
    # ALL THREE WALLED AND ISOTROPIC: the row where every wall line is live AND the
    # three inv_eps pointers are one allocation, which is the configuration the
    # signature's one non-restrict group exists for.
    {"label": "wall_xyz_isotropic", "boundaries": ("metallic", "metallic", "metallic"),
     "pml": 2, "cell": (10.0, 11.0, 9.0), "epsilon": "isotropic"},

    # ----- THE FOLDED ROWS, NEW ON 2026-08-31 -----------------------------------
    #
    # Every one of these was REFUSED BY NAME until this round, and each carries a
    # property no other row has. They are the whole point of the fill carry: on an
    # unfolded grid neither fill runs at all, so a sweep of the rows above would
    # certify the ownership inversion without ever executing one line of it. 53 of
    # this product's cell's 79 corpus rows are folded, measured on the board.
    #
    # A folded PERIODIC axis runs BOTH fills; a folded METALLIC one runs only the NEAR
    # fill and keeps its unstored zero ghost. A mutation caught on one is not caught on
    # the other.
    {"label": "fold_y_periodic", "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "pml": 2, "cell": CELL, "epsilon": "anisotropic"},
    # The ODD full count, where _far_reflect_rows is `stored - 3` rather than
    # `stored - 2`. A sweep carrying only the even count cannot see a baked n - 2.
    {"label": "fold_y_periodic_odd",
     "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("Y", 1),), "pml": 2, "cell": (9.0, 11.0, 11.0),
     "epsilon": "isotropic"},
    # The ODD PARITY plane: mirror_parity is `phase * (1 - 2*iyee)`, so a kernel that
    # baked +1 for the near fill and -1 for the far one is exact at phase +1 and wrong
    # everywhere here.
    {"label": "fold_y_odd_phase", "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("Y", -1),), "pml": 2, "cell": CELL, "epsilon": "anisotropic"},
    # The other TERMINATION: no far fill at all, and the curl takes BC_METALLIC.
    {"label": "fold_y_metallic", "boundaries": ("periodic", "metallic", "periodic"),
     "symmetry": (("Y", 1),), "pml": 2, "cell": CELL, "epsilon": "isotropic"},
    # THE TWO CROSS-TERM ROWS, AND THEY ARE THIS FAMILY'S ALONE. A D component's near
    # axes ARE the two axes zero_metal_D clears it on, so it can be folded on one and
    # walled on the other -- and then its near ghost lands in the cleared plane. The
    # array path leaves +0.0f there (fill at :3309, clear at :3310); a carry that
    # multiplied the POST-clear register by an ODD plane's parity would leave -0.0f.
    # The B family has no such row: its one near axis is the only axis its wall clears
    # it on, and folded excludes walled. AN EVEN PHASE MAKES THIS VACUOUS -- +1 * 0.0f
    # is 0.0f -- so both rows are declared ODD, and
    # `probe_cuda_electric_fill_carry_order.py` is the measurement: 9 and 11 differing
    # words against 0 on every other configuration.
    {"label": "fold_y_odd_wall_z", "boundaries": ("periodic", "periodic", "metallic"),
     "symmetry": (("Y", -1),), "pml": 2, "cell": CELL, "epsilon": "anisotropic"},
    {"label": "fold_y_odd_wall_x", "boundaries": ("metallic", "periodic", "periodic"),
     "symmetry": (("Y", -1),), "pml": 2, "cell": CELL, "epsilon": "isotropic"},
    # TWO FOLDED AXES AT MIXED PHASES. The only shape in which the composition question
    # exists at all: a cell at stored 0 of both near axes carries the PRODUCT of two
    # parities, and a source thread can itself be a fill destination -- which is what
    # the ownership guard around every carry block is for.
    {"label": "fold_xy_mixed_phase",
     "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("X", 1), ("Y", -1)), "pml": 2, "cell": CELL,
     "epsilon": "anisotropic"},
    # THREE FOLDED AXES: one source thread owns SEVEN ghost cells per component, which
    # is the deepest the closed form ever goes.
    {"label": "fold_xyz", "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("X", 1), ("Y", -1), ("Z", 1)), "pml": 2, "cell": CELL,
     "epsilon": "isotropic"},
    # A FOLDED AXIS WHOSE ABSORBER REACHES THE FAR GHOST'S COEFFICIENT ROW, and it
    # exists for ONE defect. The FAR ghost sits at the top plane of the component's OWN
    # axis -- the axis update_E indexes on -- so it must take the DESTINATION's
    # coefficient pair at `n - 1` and not the source thread's at `reflect`. On a
    # shallow layer those two entries are the same word to the bit and a kernel that
    # reused the source's pair would be BYTE-IDENTICAL. THE MIRROR IMAGE of the B
    # family, where it was the NEAR ghost that moved the indexed axis.
    {"label": "fold_x_deep_pml", "boundaries": ("periodic", "periodic", "periodic"),
     "symmetry": (("X", 1),), "pml": ((0, 5), (2, 2), (2, 2)), "cell": CELL,
     "epsilon": "anisotropic"},
)

#: The row the mutation legs are ARMED on, plus the ones they are additionally SCORED
#: on where live. ``wall_xyz`` emits every line the shipped kernel can emit: all three
#: wall-clear rows and all three metallic curl masks, and it is ANISOTROPIC so the
#: per-component inv_eps binding is live. ``wall_x`` is carried so a wall defect is
#: caught by a WRONG PLANE rather than only by the difference between three walls and
#: none, ``all_periodic`` so the wall-free arithmetic is scored, and
#: ``wall_xyz_isotropic`` so a defect that only shows where the three inv_eps pointers
#: ALIAS is scored too.
#: Five folded rows join on 2026-08-31: a single folded PERIODIC axis (both fills live),
#: the folded METALLIC termination (near fill only), the mixed-phase two-axis fold (the
#: composition, and the row where a carry source is itself a fill destination), the
#: odd-phase fold-plus-wall row (the cross term the B family cannot have), and the deep
#: layer (where the far ghost's coefficient pair differs from its source's). A fill
#: defect scored only on unfolded rows would report UNCAUGHT for a reason about the case
#: table.
#: ``fold_y_odd_wall_x`` joins as the SECOND cross-term row, and for a reason the
#: device measured: the cross term belongs to a COMPONENT, not to a spec. On
#: ``fold_y_odd_wall_z`` it is Dx that is folded on y and walled on z; on
#: ``fold_y_odd_wall_x`` it is Dz that is folded on y and walled on x. A leg anchored
#: on one component's ghost is inert on the other row, and the two together are what
#: keeps every ``applies`` predicate here honest about which component it is asking
#: about.
MUTATION_SPEC_LABELS: Tuple[str, ...] = ("wall_xyz", "wall_x", "all_periodic",
                                         "wall_xyz_isotropic", "fold_y_periodic",
                                         "fold_y_metallic", "fold_xy_mixed_phase",
                                         "fold_y_odd_wall_z", "fold_y_odd_wall_x",
                                         "fold_x_deep_pml")

#: The specs the separate-composition control runs on: one walled (2 launches per step
#: become 1) and one not (2 become 1... the wall pass is absent there, so the reduction
#: is smaller and the row is kept precisely so the two numbers differ).
#: Two folded rows join: there the separate composition is FIVE launches (curl, the near
#: fill, the wall pass where live, the far fill, constitutive) against the fused one,
#: which is the largest reduction this product makes and the only one that runs the
#: certified in-seam fill kernels beside it.
SEPARATE_CONTROL_LABELS: Tuple[str, ...] = ("wall_xyz", "all_periodic",
                                            "fold_y_periodic", "fold_xy_mixed_phase")

#: The rows ``--product reduced`` keeps. NOT the three unfolded rows it kept until
#: 2026-08-31, which is now the wrong smoke test: all three are unfolded, so neither
#: mirror fill runs and the whole of this round's carry would go unexercised in the mode
#: a run is smoke-tested in.
REDUCED_LABELS: Tuple[str, ...] = ("wall_xyz", "wall_xyz_isotropic", "all_periodic",
                                   "fold_y_periodic", "fold_y_odd_wall_z")

#: The specs the DEPOSIT legs run on. BOTH TERMINATIONS, because the one asymmetry
#: between the fused route and the driver's is where a deposit lands on a cell
#: ``zero_metal_D`` clears -- the kernel clears BEFORE the injection and the driver
#: AFTER it -- and a sweep with no wall could not see it at all.
#: A FOLDED row joins, and it is the one that makes CARRIES_DEPOSIT_REPAIR mean anything
#: on a folded electric seam: the two fills run AFTER the injection, so a repair that
#: wrote only the deposit INDEX would leave every mirror image of it describing the
#: pre-injection field. ``deposit_repair.repair_cells`` hands save/apply the CLOSURE of
#: those images, and :func:`leg_deposit_image_closure` is where that is measured against
#: a point-only control rather than inherited from the B seam.
DEPOSIT_SPEC_LABELS: Tuple[str, ...] = ("wall_xyz", "all_periodic", "fold_y_periodic")

#: The rows the THREE-LEG IMAGE-CLOSURE proof runs on. Folded only: on an unfolded grid
#: ``deposit_repair.fill_image_rules`` returns the empty rule set, so the point-only
#: control is the full closure and cannot diverge -- a vacuous pass, which is exactly
#: what this product's closure probe recorded while it still refused every fold.
IMAGE_CLOSURE_SPEC_LABELS: Tuple[str, ...] = ("fold_y_periodic", "fold_xy_mixed_phase")

#: The extent of the EXTENDED deposit row, in grid units.
EXTENDED_SIZE: Tuple[float, float, float] = (2.0, 2.0, 2.0)


def case_rng(label: str) -> np.random.Generator:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return np.random.default_rng(SEED + int.from_bytes(digest[:4], "big"))


def _install_epsilon(fields, spec: Dict[str, Any], rng) -> str:
    """Give this engine its permittivity, isotropic or diagonally anisotropic.

    THE ANISOTROPIC ROWS ARE THE ONLY PLACE A PER-COMPONENT BINDING DEFECT IS
    OBSERVABLE. ``set_epsilon_volumes`` with three DISTINCT arrays is what
    ``fields.py`` calls diagonal anisotropy; ``set_isotropic_epsilon_volume`` aliases
    one array three times, which is the shape the certified kernel's non-restrict
    inv_eps group exists to allow. Both ship, so both are swept.

    THE VALUES ARE DRAWN, NOT CONSTANT. A uniform permittivity makes
    ``D * inv_eps`` a single scaling, and a component swap between two constant
    volumes of the same value is byte-identical.
    """
    shape = tuple(int(n) for n in fields.grid.shape)
    if spec.get("epsilon", "isotropic") == "isotropic":
        epsilon = cp.asarray(rng.uniform(1.0, 4.0, size=shape).astype(np.float32))
        fields.set_isotropic_epsilon_volume(epsilon, 1.0 / epsilon)
        return "isotropic"
    eps_by, inv_by = {}, {}
    for component in ("Ex", "Ey", "Ez"):
        epsilon = cp.asarray(rng.uniform(1.0, 4.0, size=shape).astype(np.float32))
        eps_by[component] = epsilon
        inv_by[component] = 1.0 / epsilon
    fields.set_epsilon_volumes(eps_by, inv_by)
    return "anisotropic"


def build(spec: Dict[str, Any], value_class: str, rng):
    """One seeded engine: ``(fields, grid, pml, host)``. Called two or three times per
    case, identically, from the same label-derived draw.

    ``enable_pml_storage`` runs BEFORE the seeding: it allocates E, H and both
    auxiliary families, and a seed written before the allocation would be overwritten
    by it.

    THE AUXILIARIES START NONZERO in both classes, for the reason the curl and
    constitutive legs give: a zero ``fu`` makes ``fu * kms`` exactly zero on the first
    launch whatever ``kms`` is, which would hide a mis-indexed coefficient until step
    two.
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
    epsilon_shape = _install_epsilon(fields, spec, np.random.default_rng(
        SEED + int.from_bytes(hashlib.sha256(
            (spec["label"] + "|eps").encode()).digest()[:4], "big")))
    # A FOLDED AXIS TAKES ITS ABSORBER ON THE HIGH FACE ONLY. The mirror plane is the
    # low face and ``stepping._require_consistent_pml`` admits nothing there; a scalar
    # thickness on a folded axis is a layer written over the fold.
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
    fields._gate_epsilon_shape = epsilon_shape  # noqa: SLF001 - recorded, never read by the engine
    return fields, grid, pml, host


def words(array: Any) -> np.ndarray:
    """One array as raw uint32 WORDS. Byte compares, never ``allclose``: ``-0.0 ==
    0.0`` and ``NaN != NaN`` both lie, and the band class puts signed zeros in the
    operands deliberately."""
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
    """Every volume the kernel binds read-only: the PML tables and the three inv_eps.

    THE THREE INVERSE-PERMITTIVITY VOLUMES ARE IN HERE and are the reason this
    snapshot is not the magnetic gate's. They are the only ``const`` group in the
    signature that is NOT ``__restrict__``, so a write through one would be a plain
    aliasing bug rather than UB -- and a plain bug is exactly the kind a byte
    comparison of the STATE arrays cannot see.
    """
    out: Dict[str, np.ndarray] = {}
    for name in IMMUTABLE_PML:
        value = getattr(pml, name, None)
        if value is not None:
            out[f"pml.{name}"] = words(value).copy()
    for component in ("Ex", "Ey", "Ez"):
        out[f"inv_eps.{component}"] = words(
            fields.inverse_epsilon_for(component)).copy()
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

#: The eleven ``stepping`` calls ``FdtdDriver.step`` makes, in driver order. ASSERTED
#: against the driver's own source by :func:`leg_driver_order`, not trusted: a gate
#: that carried a private model of a timestep would certify the kernel against the
#: model.
DRIVER_ORDER: Tuple[str, ...] = (
    "step_B", "fill_symmetry_bc_B", "zero_metal_B", "fill_folded_far_ghosts_B",
    "update_H",
    "step_D", "fill_symmetry_bc_D", "zero_metal_D", "fill_folded_far_ghosts_D",
    "update_E", "update_P",
)

#: The FIVE the fused kernel replaces, declared by the module itself and asserted equal
#: to ``fused_electric_pair.REPLACES`` by :func:`leg_driver_order`. THE TWO MIRROR FILLS
#: JOINED ON 2026-08-31 with the ownership inversion; until then they were refused and
#: kept running on the array path beside the launch, and the run named here was a
#: subsequence of the driver's rather than a contiguous span of it.
FUSED_PASSES: Tuple[str, ...] = ("step_D", "fill_symmetry_bc_D", "zero_metal_D",
                                 "fill_folded_far_ghosts_D", "update_E")

#: Which passes take the PML layer. Read off ``stepping``'s signatures rather than
#: guessed, so a pass that gained or lost the argument is a TypeError here rather than
#: a silently skipped absorber.
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

    Read out of ``driver.py``'s text rather than by importing the driver: the question
    is what the SOURCE says, and a gate that imported the module to ask would be
    asking a different object than the one a reader checks.

    ``REPLACES`` is additionally required to be an ordered SUBSEQUENCE of that list,
    and SINCE 2026-08-31 a CONTIGUOUS run of it -- the two mirror fills are carried
    now, so a gap would be a driver pass the launch skipped while the array path
    performed it. Both are recorded; the contiguity is what the verdict requires.
    """
    from meep_gpu.cuda_kernels import fused_electric_pair as family  # noqa: PLC0415

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
    declared = tuple(family.REPLACES)
    positions = [found.index(name) for name in declared if name in found]
    subsequence = (len(positions) == len(declared)
                   and positions == sorted(positions))
    contiguous = bool(positions) and positions == list(
        range(positions[0], positions[0] + len(positions)))
    return {
        "passed": bool(found == DRIVER_ORDER and subsequence and contiguous
                       and declared == FUSED_PASSES),
        "driver_file_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "driver_call_sequence": list(found),
        "gate_declared_order": list(DRIVER_ORDER),
        "fused_replaces": list(declared),
        "replaces_is_an_ordered_subsequence": subsequence,
        "replaces_is_a_contiguous_run": contiguous,
        "passes_left_on_the_array_path": [name for name in DRIVER_ORDER
                                          if name not in declared],
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
    package reaches its kernel through ``compile_cache.get_or_compile``, which returns
    ``_compiled_kernels[key]`` on a hit, so replacing the stored object counts every
    launch through every shipped route without editing one byte under ``meep_gpu/``.
    NOTHING IS INSTALLED UNLESS THE MEMO IS ALREADY WARM, which is why every case
    warms it first: a cold memo would have the factory store a raw kernel and the
    count would read zero for a kernel that ran.
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

    The memo must be warm before :class:`MemoLaunchCounter` installs.

    A REFUSED WARM-UP IS FATAL RATHER THAN SILENT: a spec the predicate does not admit
    would otherwise reach the product sweep and fail there with a reason about
    coverage, one leg later than the table that was wrong.
    """
    from meep_gpu.cuda_kernels import (constitutive_kernels,  # noqa: PLC0415
                                       fused_electric_pair, in_seam_coverage,
                                       in_seam_passes, step_curl_kernels)
    for warm in specs:
        fields, grid, pml, _ = build(warm, "uniform", np.random.default_rng(1))
        dtdx = float(grid.dt / grid.dx)
        report = fused_electric_pair.run_fused_electric_pair(
            fields, grid, pml, dtdx, sources=())
        if not report.get("launched"):
            raise SystemExit(f"the memo warm-up could not launch on {warm['label']}: "
                             f"{report.get('reason')}")
        step_curl_kernels._step_D_fused_pml_real(  # noqa: SLF001
            fields, step_curl_kernels.real_pml_curl_tables(pml, False),
            step_curl_kernels.real_curl_boundary_codes(grid), dtdx)
        # The certified in-seam passes the separate-composition control launches
        # beside the two halves. A COLD MEMO IS NOT A MISSING LAUNCH BUT IT IS A
        # MISSING COUNT: MemoLaunchCounter wraps what is already stored, so a kernel
        # compiled inside the counted region is launched and not counted, and the
        # control's launch total comes back short by exactly the number of kernels it
        # compiled. MEASURED on the GPU host 2026-08-31: fold_y_periodic reported 22
        # separate launches against the 24 its plan names, short by the two fills'
        # first launches, while fold_xy_mixed_phase (which ran second, on a warm memo)
        # reported all 36.
        for pass_name in ("zero_metal", "fill_symmetry", "fill_folded_far"):
            if in_seam_coverage.plan(pass_name, grid):
                in_seam_passes.run_pass(pass_name, fields, "D", grid=grid)
        constitutive_kernels.update_fused_pml_real("E", fields, pml=pml)
    cp.cuda.runtime.deviceSynchronize()


def run_case(spec: Dict[str, Any], value_class: str, steps: int,
             kernel: Optional[Any] = None,
             tables_patch: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None,
             codes_patch: Optional[Callable[[Sequence[Any]], Sequence[Any]]] = None,
             walls_patch: Optional[Callable[[Sequence[int], Any], Sequence[int]]] = None,
             fills_patch: Optional[Callable[[Dict[str, Any], Any], Dict[str, Any]]] = None,
             label_suffix: str = "") -> Dict[str, Any]:
    """Step two engines side by side and compare per COMPLETE DRIVER STEP.

    ``kernel`` and the three ``*_patch`` hooks are THE GATE'S DOORS -- the shipped
    launcher takes a kernel, a table set and the two grid readings precisely so a gate
    can arm a defect. A harness that could not hand in its own kernel could not arm a
    single mutation and every mutation leg would launch the shipped kernel and report
    the defect as uncaught.
    """
    from meep_gpu.cuda_kernels import fused_electric_pair as family  # noqa: PLC0415
    from meep_gpu.cuda_kernels import in_seam_coverage, step_curl_kernels  # noqa: PLC0415

    started = time.time()
    label = f"{spec['label']}|{value_class}{label_suffix}"
    case: Dict[str, Any] = {"label": spec["label"], "value_class": value_class,
                            "case_seed_label": label, "steps_requested": steps,
                            "boundaries": list(spec["boundaries"]),
                            "epsilon": spec.get("epsilon", "isotropic"),
                            "pml_cells": spec["pml"]}

    reference, ref_grid, ref_pml, host = build(spec, value_class, case_rng(label))
    actual, grid, pml, _ = build(spec, value_class, case_rng(label))
    case["shape"] = [int(n) for n in grid.shape]
    case["dtdx"] = float(grid.dt / grid.dx)
    case["operand_census"] = operand_census(host)
    case["distinct_inverse_epsilon_allocations"] = len({
        int(actual.inverse_epsilon_for(c).data.ptr) for c in ("Ex", "Ey", "Ez")})

    drift = compare(reference, actual)
    if drift:
        case["passed"] = False
        case["why"] = f"the two builds are not identical: {drift}"
        return case
    eps_drift = {c: differing(reference.inverse_epsilon_for(c),
                              actual.inverse_epsilon_for(c))
                 for c in ("Ex", "Ey", "Ez")}
    if any(eps_drift.values()):
        case["passed"] = False
        case["why"] = (f"the two builds disagree on the permittivity: {eps_drift}; "
                       f"the comparison would be between two different materials")
        return case

    covered, reason = family.covers_fused_electric_pair(actual, pml, grid, ())
    case["predicate"] = {"covered": bool(covered), "reason": reason}
    if not covered:
        case["passed"] = False
        case["why"] = f"the predicate refused this fixture: {reason}"
        return case

    tables = family.fused_electric_pair_tables(pml)
    case["distinct_allocations_checked"] = family.assert_disjoint_bindings(actual,
                                                                          tables)
    if tables_patch is not None:
        tables = tables_patch(tables, pml)
    codes = step_curl_kernels.real_curl_boundary_codes(grid)
    walls = in_seam_coverage.zero_metal_axes(grid)
    # THE THIRD READING OF THE GRID, and it is not either of the other two: a folded
    # METALLIC axis carries the near fill and no wall, and a folded PERIODIC one
    # carries both fills. A host mutation that confused the three is armed through
    # ``fills_patch``.
    fills = family.fused_electric_pair_fills(grid)
    case["boundary_codes"] = [int(c) for c in codes]
    case["zero_metal_axes"] = [bool(w) for w in walls]
    case["fills"] = {key: list(fills[key]) for key in ("near", "reflect", "phase")}
    case["fills_live"] = bool(any(fills["near"])
                              or any(row >= 0 for row in fills["reflect"]))
    if codes_patch is not None:
        codes = codes_patch(codes)
    if walls_patch is not None:
        walls = walls_patch(walls, grid)
    if fills_patch is not None:
        fills = fills_patch(fills, grid)
    case["boundary_codes_used"] = [int(c) for c in codes]
    case["zero_metal_axes_used"] = [bool(w) for w in walls]
    case["fills_used"] = {key: list(fills[key])
                          for key in ("near", "reflect", "phase")}

    before = frozen(actual)
    read_only = read_only_snapshot(actual, pml)
    launched = 0
    counter = MemoLaunchCounter()
    per_step: List[Dict[str, Any]] = []
    with counter:
        for step in range(1, steps + 1):
            array_step(reference, ref_pml)
            for name in DRIVER_ORDER:
                if name == "step_D":
                    report = family.launch_fused_electric_pair(
                        actual, tables, codes, walls, fills, case["dtdx"], kernel)
                    launched += int(bool(report.get("launched")))
                elif name not in FUSED_PASSES:
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

    WHAT THE FUSION REMOVES IS A NUMBER, and it is measured rather than described: the
    separate composition launches the certified ``step_D_pml_real``, the certified
    ``zero_metal_D`` where a wall is live, and the certified ``update_E_pml_real``;
    the fused product launches once. Both must be byte-identical to the array path AND
    to each other, which is the only place this gate compares the fused kernel against
    the certified halves directly rather than against ``stepping``.
    """
    from meep_gpu.cuda_kernels import (constitutive_kernels,  # noqa: PLC0415
                                       fused_electric_pair as family,
                                       in_seam_coverage, in_seam_passes,
                                       step_curl_kernels)

    started = time.time()
    label = f"{spec['label']}|separate"
    oracle, _og, oracle_pml, _ = build(spec, "uniform", case_rng(label))
    separate, sep_grid, sep_pml, _ = build(spec, "uniform", case_rng(label))
    fused, grid, pml, _ = build(spec, "uniform", case_rng(label))

    dtdx = float(grid.dt / grid.dx)
    curl_tables = step_curl_kernels.real_pml_curl_tables(sep_pml, False)
    codes = step_curl_kernels.real_curl_boundary_codes(sep_grid)
    walls = in_seam_coverage.zero_metal_axes(sep_grid)
    fused_tables = family.fused_electric_pair_tables(pml)

    separate_counter = MemoLaunchCounter()
    fused_counter = MemoLaunchCounter()
    per_step: List[Dict[str, Any]] = []
    for step in range(1, steps + 1):
        array_step(oracle, oracle_pml)
        with separate_counter:
            for name in DRIVER_ORDER:
                if name == "step_D":
                    step_curl_kernels._step_D_fused_pml_real(  # noqa: SLF001
                        separate, curl_tables, codes, dtdx)
                elif name == "zero_metal_D":
                    if any(walls):
                        in_seam_passes.run_pass("zero_metal", separate, "D",
                                                grid=sep_grid)
                # THE TWO CERTIFIED IN-SEAM FILLS, launched beside the two halves --
                # the only place this gate runs the stand-alone passes the carry
                # lifts, and the reason a folded row belongs in this control.
                elif name == "fill_symmetry_bc_D":
                    if sep_grid.has_symmetry():
                        in_seam_passes.run_pass("fill_symmetry", separate, "D",
                                                grid=sep_grid)
                elif name == "fill_folded_far_ghosts_D":
                    if sep_grid.has_symmetry() and any(
                            row is not None
                            for row in in_seam_coverage.folded_far_rows(sep_grid)):
                        in_seam_passes.run_pass("fill_folded_far", separate, "D",
                                                grid=sep_grid)
                elif name == "update_E":
                    constitutive_kernels.update_fused_pml_real("E", separate,
                                                               pml=sep_pml)
                else:
                    run_pass(name, separate, sep_pml)
        with fused_counter:
            for name in DRIVER_ORDER:
                if name == "step_D":
                    family.launch_fused_electric_pair(
                        fused, fused_tables,
                        step_curl_kernels.real_curl_boundary_codes(grid),
                        in_seam_coverage.zero_metal_axes(grid),
                        family.fused_electric_pair_fills(grid), dtdx)
                elif name not in FUSED_PASSES:
                    run_pass(name, fused, pml)
        cp.cuda.runtime.deviceSynchronize()
        against_oracle = compare(oracle, fused)
        against_separate = compare(separate, fused)
        per_step.append({"step": step,
                         "fused_vs_array": sum(against_oracle.values()),
                         "fused_vs_separate": sum(against_separate.values())})
        if against_oracle or against_separate:
            break

    identical = (len(per_step) == steps
                 and all(row["fused_vs_array"] == 0 and row["fused_vs_separate"] == 0
                         for row in per_step))
    separate_total = separate_counter.total
    fused_total = fused_counter.total
    # THE EXPECTED COUNT IS THE CERTIFIED PASSES' OWN PLAN, not a formula. Each fill
    # is ONE LAUNCH PER FOLDED AXIS in X, Y, Z order (in_seam_coverage.plan's own
    # docstring: a component unowned on two planes has its corner written twice and
    # carries the product of both parities, which two axes in one launch cannot
    # promise), while zero_metal is one launch for all three. A hand-written
    # "+1 if folded" was wrong on the two-axis row by exactly two launches.
    per_pass = {name: len(in_seam_coverage.plan(name, sep_grid))
                for name in ("zero_metal", "fill_symmetry", "fill_folded_far")}
    expected_separate = steps * (2 + sum(per_pass.values()))
    return {
        "passed": bool(identical and separate_total == expected_separate
                       and fused_total == steps),
        "three_way_identical": identical,
        "steps_run": len(per_step),
        "per_step": per_step,
        "walls": [bool(w) for w in walls],
        "separate_launches_per_pass": per_pass,
        "separate_launches_total": separate_total,
        "separate_launches_named": separate_counter.named(),
        "separate_launches_expected": expected_separate,
        "fused_launches_total": fused_total,
        "fused_launches_named": fused_counter.named(),
        "launch_counts_agree": bool(separate_total == expected_separate
                                    and fused_total == steps),
        "launches_removed_per_step": (separate_total - fused_total) / max(steps, 1),
        "seconds": time.time() - started,
    }


# ---------------------------------------------------------------------------
# The lift: is the emitted text the certified text?
# ---------------------------------------------------------------------------

def leg_lift() -> Dict[str, Any]:
    """The module claims its arithmetic is LIFTED rather than retyped. Measured.

    Five questions, each a property of the construction rather than of prose:

    1. the certified curl body must survive into the fused source differing in EXACTLY
       the three ``pml_apply`` captures plus the hoisted register block;
    2. the certified constitutive body must differ in EXACTLY the three flux-density
       reads and the three ``constitutive_apply`` lines (the seam and the sub-lattice
       rename), and in nothing else;
    3. the certified curl prelude must differ in EXACTLY the ``pml_apply`` signature
       and its store, and the certified constitutive prelude must be carried with ZERO
       edits;
    4. ``LIFT_EDITS`` must have exactly NINE entries -- six until 2026-08-31, plus the
       three the fill carry added -- and it is data so that a silent tenth edit is a
       visible diff;
    5. the carried ``zero_metal_D`` must be the OFF-DIAGONAL table: two components per
       wall, never the component whose own axis is walled, and ``+0.0f``;
    6. the two carried fills must image each component on the axes ``IYEE_SHIFTS``
       says and no others, every carry block must sit inside its component's ownership
       guard, and the near carry must read the PRE-clear register while the far carry
       reads the post-clear one -- the driver's order between :3309 and :3310.

    And one correctness question that is not about the lift at all: ``Dx``, ``Dy`` and
    ``Dz`` must be bound EXACTLY ONCE each in the signature and never as a ``const
    __restrict__`` source, while the three ``inv_eps`` pointers must NOT be
    ``__restrict__`` -- the certified kernel's own decision, because an isotropic run
    hands one allocation three times.
    """
    from meep_gpu.cuda_kernels import (constitutive_kernels,  # noqa: PLC0415
                                       fused_electric_pair as family,
                                       own_cell_hoist, step_curl_kernels)

    source = family.fused_electric_pair_source()
    raw_curl = family._split_body(  # noqa: SLF001
        step_curl_kernels._step_D_pml_real_kernel_code,  # noqa: SLF001
        step_curl_kernels._REAL_PML_PRELUDE, "curl")  # noqa: SLF001
    lifted_curl = family.certified_curl_body()
    # THE CONSTITUTIVE COMPARAND IS THE RAW BODY'S TAIL, not the whole body. The lift
    # deliberately DROPS the decode prologue -- the curl half already emitted it and a
    # second copy would redeclare i/j/k -- so comparing against the whole body would
    # report six dropped lines that are not edits to the arithmetic at all.
    raw_const_body = family._split_body(  # noqa: SLF001
        own_cell_hoist.unhoisted_kernel_code(
            constitutive_kernels._update_E_pml_real_kernel_code,  # noqa: SLF001
            "update_E_pml_real", "_update_E_pml_real_kernel_code"),
        constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE, "constitutive")  # noqa: SLF001
    raw_const = raw_const_body.split(family._DECODE_END, 1)[1]  # noqa: SLF001
    lifted_const = family.certified_constitutive_body()

    def changed(before: str, after: str) -> List[List[str]]:
        a = [line for line in before.splitlines() if line.strip()]
        b = [line for line in after.splitlines() if line.strip()]
        common = set(a) & set(b)
        return [[line for line in a if line not in common],
                [line for line in b if line not in common]]

    curl_dropped, curl_added = changed(raw_curl, lifted_curl)
    const_dropped, const_added = changed(raw_const, lifted_const)
    prelude = family.fused_electric_pair_prelude()
    prelude_dropped, prelude_added = changed(
        step_curl_kernels._REAL_PML_PRELUDE  # noqa: SLF001
        + constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE, prelude)  # noqa: SLF001

    # THE OWNERSHIP DECLARATIONS ARE ADDED, NOT SUBSTITUTED, so the curl's dropped set
    # is still the three pml_apply calls and the added set carries the three flags.
    curl_ok = (len(curl_dropped) == 3
               and all("pml_apply(D" in line for line in curl_dropped)
               and all(f"d_{a} = pml_apply_reg(D{a}" in "".join(curl_added)
                       for a in "xyz")
               and all(f", own_{a});" in "".join(curl_added) for a in "xyz")
               and all(f"int own_{a} = !(" in "".join(curl_added) for a in "xyz"))
    seam_lines = [line for line in const_added
                  if line.strip().startswith("float src_")]
    call_lines = [line for line in const_added
                  if line.strip().startswith("constitutive_apply(E")
                  and ", idx, src_" in line]
    # THE CARRY'S OWN LINES ARE SUBTRACTED EXACTLY, not by prefix. A prefix allow-list
    # would admit any line that happened to start the same way -- including a store to
    # the wrong volume -- which is precisely the class of edit this leg exists to
    # notice. The expected set is built by asking the SHIPPED emitter what it emits, so
    # a carry line that appeared without the emitter emitting it is unexpected.
    carry_lines = {line.strip()
                   for target in range(3)
                   for line in family.fill_carry_blocks(target)
                   if line.strip()}
    carry_lines |= {f"if (own_{axis}) {{" for axis in "xyz"}
    carry_lines.add("}")
    const_unexpected = [
        line for line in const_added
        if line.strip() not in carry_lines
        and not line.strip().startswith(("float src_", "constitutive_apply(", "//"))]
    const_ok = (len(const_dropped) == 6
                and sum(1 for line in const_dropped
                        if line.strip().startswith("float src_")) == 3
                and sum(1 for line in const_dropped
                        if line.strip().startswith("constitutive_apply(")) == 3
                and len(seam_lines) == 3
                and all(f"d_{a} *" in "".join(seam_lines) for a in "xyz")
                and len(call_lines) == 3
                and all("kms_half_" in line for line in call_lines)
                and not const_unexpected)
    prelude_ok = (len(prelude_dropped) == 3
                  and any("void pml_apply(" in line for line in prelude_dropped)
                  and any("float kms, float sinv, float kms_u, float sinv_u"
                          in line for line in prelude_dropped)
                  and any("f[idx] = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;"
                          in line for line in prelude_dropped)
                  and any("if (!owned) return 0.0f;" in line
                          for line in prelude_added)
                  and any("return value;" in line for line in prelude_added)
                  and constitutive_kernels._REAL_CONSTITUTIVE_PRELUDE in prelude)  # noqa: SLF001

    # ----- THE FILL CARRY, as a property of the emitted text --------------------
    #
    # The AXIS SETS are read from the shipped module (which reads IYEE_SHIFTS) and
    # checked against the text, so a carry that imaged a component on the B family's
    # faces would be caught here rather than only on a folded device row.
    carry_axis_sets = {
        family._TARGETS[target]: {  # noqa: SLF001
            "near": [int(a) for a in family.near_fill_axes(target)],
            "far": [int(a) for a in family.far_fill_axes(target)],
            "destinations": len(family.carried_destinations(
                family.near_fill_axes(target), family.far_fill_axes(target))),
        }
        for target in range(3)
    }
    carry_geometry_ok = all(
        entry["far"] == [target] and sorted(entry["near"]) == sorted(
            {0, 1, 2} - {target}) and entry["destinations"] == 7
        for target, entry in enumerate(carry_axis_sets.values()))
    # EVERY BLOCK NESTED, and no constitutive_apply on E outside a guard.
    guards_ok = True
    for target, axis in enumerate("xyz"):
        if f"    if (own_{axis}) {{" not in lifted_const:
            guards_ok = False
            continue
        block = lifted_const.split(f"    if (own_{axis}) {{", 1)[1].split(
            "\n    }", 1)[0]
        guards_ok = guards_ok and block.count(f"constitutive_apply(E{axis},") == 8
    guards_ok = guards_ok and lifted_const.count("constitutive_apply(E") == 24
    # THE DRIVER'S ORDER, in the emitted text: the near product from pre_*, the clear,
    # then the far weight. See probe_cuda_electric_fill_carry_order.py, which measures
    # that the two orderings are NOT the same arithmetic (9 and 11 differing words).
    order_ok = ("float pre_x = d_x;" in source
                and "float gx_ny_v = phase_y * pre_x;" in source
                and "if (clr_x) gx_ny_v = 0.0f;" in source
                and "float gx_x_v = d_x;" in source
                and "gx_x_v = (-phase_x) * gx_x_v;" in source)
    composed = source.split(
        "if (reflect_x >= 0 && i == reflect_x && near_y && j == 2) {", 1)
    if len(composed) == 2:
        tail = composed[1]
        marks = [tail.find("float gx_xny_v = phase_y * pre_x;"),
                 tail.find("if (clr_x) gx_xny_v = 0.0f;"),
                 tail.find("gx_xny_v = (-phase_x) * gx_xny_v;")]
        order_ok = order_ok and all(m >= 0 for m in marks) and marks == sorted(marks)
    else:
        order_ok = False
    # THE FAR GHOST TAKES THE DESTINATION'S COEFFICIENT PAIR and the near ghost this
    # thread's -- the mirror image of the B family, where the near fill moved the
    # indexed axis.
    coefficient_ok = all(
        f"kps_{a}[n{a} - 1], kms_half_{a}[n{a} - 1]);" in source for a in "xyz")
    stencil = source.count("dtdx * ((sf - f1) + (f2 - ss))")
    binding_ok = (all(source.count(f"float* __restrict__ D{a}") == 1 for a in "xyz")
                  and "const float* __restrict__ Dx" not in source
                  and all(f"const float* inv_eps_E{a}" in source for a in "xyz")
                  and "__restrict__ inv_eps" not in source)
    # THE OFF-DIAGONAL TABLE, as a property of the emitted text. Two components per
    # wall, and NEVER the component whose own axis carries the wall. The flags are
    # NAMED since 2026-08-31 because the near carry reads them; the store is guarded on
    # ownership because a cell a fill images is written by its source thread.
    off_diagonal_ok = all(
        f"if (wall_{a} && {c} == 0) {{ " in source
        and f"clr_{a} = 1;" not in
            source.split(f"if (wall_{a} && {c} == 0) {{ ", 1)[1].split("\n", 1)[0]
        for a, c in zip("xyz", "ijk"))
    pairs_ok = all(
        source.count(f"if (wall_{a} && {c} == 0) {{ clr_{p} = 1; clr_{q} = 1; }}") == 1
        for a, c, p, q in (("x", "i", "y", "z"), ("y", "j", "x", "z"),
                           ("z", "k", "x", "y")))
    pairs_ok = pairs_ok and all(
        source.count(f"if (own_{a} && clr_{a}) "
                     f"{{ d_{a} = 0.0f; D{a}[idx] = d_{a}; }}") == 1 for a in "xyz")
    return {
        "passed": bool(curl_ok and const_ok and prelude_ok and binding_ok
                       and off_diagonal_ok and pairs_ok and stencil == 3
                       and carry_geometry_ok and guards_ok and order_ok
                       and coefficient_ok
                       and "-0.0f" not in source
                       and len(family.LIFT_EDITS) == 9),
        "curl_lines_replaced": curl_dropped,
        "curl_lines_introduced": curl_added,
        "curl_lift_ok": curl_ok,
        "constitutive_lines_replaced": const_dropped,
        "constitutive_lines_introduced": const_added,
        "constitutive_lines_unexpected": const_unexpected,
        "constitutive_lift_ok": const_ok,
        "prelude_lines_replaced": prelude_dropped,
        "prelude_lift_ok": prelude_ok,
        "parenthesised_stencils_verbatim": stencil,
        "shared_volume_bound_once": binding_ok,
        "zero_metal_table_is_the_d_off_diagonal": bool(off_diagonal_ok and pairs_ok),
        "fill_carry_axis_sets": carry_axis_sets,
        "fill_carry_geometry_ok": bool(carry_geometry_ok),
        "every_carry_block_inside_its_ownership_guard": bool(guards_ok),
        "near_reads_pre_clear_far_reads_post_clear": bool(order_ok),
        "far_ghost_takes_the_destinations_coefficient_pair": bool(coefficient_ok),
        "declared_lift_edits": len(family.LIFT_EDITS),
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "source_lines": len(source.splitlines()),
    }


# ---------------------------------------------------------------------------
# The refusals this predicate ADDS over its two halves
# ---------------------------------------------------------------------------

def leg_refusal() -> Dict[str, Any]:
    """Rule 3 of this campaign, measured: does the predicate refuse BY NAME?

    A predicate that admits a row the launch cannot serve is a SILENT wrong answer --
    the one failure mode a byte comparison is blind to, because the row it would be
    wrong on is a row this sweep never builds. Every configuration below is
    constructed and asked, and each must come back refused with a reason naming the
    thing that made it unservable.

    TWO OF THESE ARE THIS PRODUCT'S OWN CLAUSES rather than a half's, and they are the
    ones that matter: BOTH certified halves ADMIT a folded grid, so the fold refusal
    exists only in ``fused_electric_pair.py``; and the conductivity refusal is about
    the DRIVER'S INJECTION ROUTE rather than about the sub-step, so no half makes it
    either. The leg asserts that fact directly -- it asks both halves on the folded
    configuration and requires them to ADMIT -- so a future edit that moved the clause
    into a half would be caught rather than silently making this leg vacuous.
    """
    from meep_gpu.cuda_kernels import fused_electric_pair as family  # noqa: PLC0415
    from meep_gpu.cuda_kernels.coverage import (  # noqa: PLC0415
        covers_real_pml_constitutive, covers_real_pml_curl)
    from meep_gpu.fields import Fields  # noqa: PLC0415
    from meep_gpu.grid import Grid, Mirror  # noqa: PLC0415
    from meep_gpu.pml import PML  # noqa: PLC0415
    from meep_gpu.sources import GaussianEnvelope, VolumeSource  # noqa: PLC0415

    checks: Dict[str, Any] = {}

    def plain(**kwargs):
        grid = Grid(resolution=1.0, cell_size=CELL,
                    boundaries=("periodic", "periodic", "periodic"), xp=cp,
                    courant=0.5, **kwargs)
        fields = Fields(grid=grid)
        fields.enable_pml_storage()
        return fields, PML(grid=grid, thickness=2), grid

    # 0. THE POSITIVE CONTROL. A refusal leg whose every case refuses proves nothing
    #    about the predicate: it would look identical if the predicate refused
    #    everything.
    fields, pml, grid = plain()
    covered, reason = family.covers_fused_electric_pair(fields, pml, grid, ())
    checks["admits_a_plain_run"] = {"covered": bool(covered), "reason": reason,
                                    "expected": True}

    # 1. THE FOLD, ADMITTED SINCE 2026-08-31, and the two clauses that replaced the
    #    blanket refusal. The widening is the point of this round -- 53 of the cell's
    #    79 corpus rows -- so what has to be measured is that it is a widening TO WHAT
    #    THE CARRY IMPLEMENTS and not past it.
    folded_fields, folded_pml, folded_grid = plain(symmetry=(Mirror("Y", 1),))
    folded_pml = PML(grid=folded_grid, thickness=((2, 2), (0, 2), (2, 2)))
    covered, reason = family.covers_fused_electric_pair(
        folded_fields, folded_pml, folded_grid, ())
    curl_half = covers_real_pml_curl(folded_fields, folded_pml, folded_grid, "step_D")
    const_half = covers_real_pml_constitutive(folded_fields, folded_pml, folded_grid,
                                              "E")
    checks["admits_a_folded_grid"] = {
        "covered": bool(covered), "reason": reason, "expected": True,
        "curl_half_admits_it": bool(curl_half[0]),
        "constitutive_half_admits_it": bool(const_half[0]),
        "the_carry_is_what_admits_it": bool(curl_half[0] and const_half[0]),
    }

    # 1a. A FOLDED AXIS TOO SHORT FOR THE NEAR SOURCE ROW. ``stepping._mirror_source``
    #     raises below three stored cells; this kernel would index outside the volume.
    class _ShortAxis:
        def __getattr__(self, item):
            return getattr(folded_grid, item)

        def stored_cells(self, axis):
            return 2 if axis == 1 else folded_grid.stored_cells(axis)

    class _ShortFields:
        def __getattr__(self, item):
            return getattr(folded_fields, item)

    short = _ShortFields()
    short.__dict__["Dx"] = type("_Extent", (), {
        "shape": tuple(2 if axis == 1 else int(folded_fields.Dx.shape[axis])
                       for axis in range(3))})()
    covered, reason = family.covers_fused_electric_pair(
        short, folded_pml, _ShortAxis(), ())
    checks["refuses_a_folded_axis_below_the_near_source_row"] = {
        "covered": bool(covered), "reason": reason, "expected": False}

    # 1b. AN AXIS REPORTED BOTH FOLDED AND WALLED. Unreachable from a real Grid
    #     (``zero_metal_axes`` is ``is_metallic and not is_mirrored``) and refused
    #     anyway: a D component's near axes ARE the axes zero_metal_D clears it on, so
    #     the two sets being disjoint is what makes the source thread's own clear flag
    #     the DESTINATION's test in the carry.
    saved_walls = family.zero_metal_axes
    try:
        family.zero_metal_axes = lambda _grid: (False, True, False)  # noqa: SLF001
        covered, reason = family.covers_fused_electric_pair(
            folded_fields, folded_pml, folded_grid, ())
    finally:
        family.zero_metal_axes = saved_walls
    checks["refuses_an_axis_both_folded_and_walled"] = {
        "covered": bool(covered), "reason": reason, "expected": False,
        "names_the_conflict": "both folded and walled" in reason}

    # 2. THE CONDUCTIVITY. Not a clause about the sub-step: the driver deposits
    #    through _inject_electric_through_conductivity (driver.py:3305) whenever the
    #    ENGINE carries one, and the repair has no verdict on that route.
    class _Conductive:
        def __getattr__(self, item):
            return getattr(fields, item)

        has_conductivity = True

    covered, reason = family.covers_fused_electric_pair(_Conductive(), pml, grid, ())
    checks["refuses_a_conductive_engine"] = {
        "covered": bool(covered), "reason": reason, "expected": False,
        "names_the_route": "driver.py:3305" in reason and "condinv" in reason}

    # 3. THE SOURCE SEAM. An electric source with no publishable index cannot be
    #    saved and restored; a MAGNETIC one is injected in the other half entirely.
    class _Opaque:
        field_type = VolumeSource(
            grid=grid, component="Ez", center=(0.0, 0.0, 0.0), size=(0.0, 0.0, 0.0),
            envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2)).field_type

    covered, reason = family.covers_fused_electric_pair(fields, pml, grid, [_Opaque()])
    checks["refuses_a_source_with_no_index"] = {
        "covered": bool(covered), "reason": reason, "expected": False,
        "names_the_index": "does not publish the index it writes" in reason}

    magnetic = VolumeSource(grid=grid, component="Hz", center=(0.0, 0.0, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2))
    covered, reason = family.covers_fused_electric_pair(fields, pml, grid, [magnetic])
    checks["admits_a_magnetic_source"] = {"covered": bool(covered), "reason": reason,
                                          "expected": True}

    electric = VolumeSource(grid=grid, component="Ez", center=(0.0, 0.0, 0.0),
                            size=(0.0, 0.0, 0.0),
                            envelope=GaussianEnvelope(frequency=1.0, fwidth=0.2))
    covered, reason = family.covers_fused_electric_pair(fields, pml, grid, [electric])
    checks["admits_a_repairable_electric_source"] = {
        "covered": bool(covered), "reason": reason, "expected": True}

    covered, reason = family.covers_fused_electric_pair(fields, pml, grid, None)
    checks["refuses_an_undeclared_source_set"] = {
        "covered": bool(covered), "reason": reason, "expected": False,
        "names_the_ignorance": "was not declared" in reason}

    # 4. COMPLEX STORAGE, which is a half's clause and is asked anyway: the fused
    #    signature is float32 and would read the wrong stride.
    complex_grid = Grid(resolution=1.0, cell_size=CELL,
                        boundaries=("periodic", "periodic", "periodic"), xp=cp,
                        courant=0.5)
    complex_fields = Fields(grid=complex_grid, force_complex_fields=True)
    complex_fields.enable_pml_storage()
    covered, reason = family.covers_fused_electric_pair(
        complex_fields, PML(grid=complex_grid, thickness=2), complex_grid, ())
    checks["refuses_complex_storage"] = {"covered": bool(covered), "reason": reason,
                                         "expected": False}

    # 5. NO ABSORBER: without one update_E takes the plain assignment (stepping.py:1022)
    #    and this is the wrong sub-step entirely.
    covered, reason = family.covers_fused_electric_pair(fields, None, grid, ())
    checks["refuses_a_run_with_no_pml"] = {"covered": bool(covered), "reason": reason,
                                           "expected": False}

    passed = all(bool(entry["covered"]) == bool(entry["expected"])
                 for entry in checks.values())
    passed = passed and checks["admits_a_folded_grid"]["the_carry_is_what_admits_it"]
    passed = passed and checks[
        "refuses_an_axis_both_folded_and_walled"]["names_the_conflict"]
    passed = passed and checks["refuses_a_conductive_engine"]["names_the_route"]
    passed = passed and checks["refuses_a_source_with_no_index"]["names_the_index"]
    passed = passed and checks["refuses_an_undeclared_source_set"]["names_the_ignorance"]
    return {"passed": bool(passed), "checks": checks}


# ---------------------------------------------------------------------------
# The in-seam deposit
# ---------------------------------------------------------------------------

#: THE CARRIER FREQUENCY, AND IT IS A MEASUREMENT RATHER THAN A DEFAULT.
#: ``continuous_src_time::dipole`` is ``amp_factor * exp(-i*omega*t)`` with
#: ``amp_factor`` imaginary (MEEP sources.cpp:104), so the REAL part a real-storage
#: run deposits is proportional to ``sin(omega t)``. These fixtures run at
#: ``courant 0.5 / resolution 1.0``, i.e. ``dt = 0.5`` -- and at ``frequency 1.0``
#: ``omega * t`` is an exact multiple of ``pi`` at every sampled step, so the deposit
#: is IDENTICALLY ZERO and every deposit leg would agree with a no-op. The magnetic
#: gate's null control caught exactly that on its first device run; this file inherits
#: the frequency AND :func:`deposit_magnitude`, which measures rather than assumes.
#:
#: THE ELECTRIC SEAM SAMPLES AT ``t + dt/2`` (driver.py:3303), not at ``t``, so the
#: zeros are at different instants than the magnetic seam's -- which is one more
#: reason the magnitude is measured on this side rather than inherited as a fact.
DEPOSIT_FREQUENCY = 0.37


def deposit_rows(spec: Dict[str, Any]
                 ) -> Tuple[Tuple[str, str, Tuple[float, float, float]], ...]:
    """``(tag, component, size)`` for this spec's two deposit rows: POINT and EXTENT.

    THE TWO ROWS DIFFER IN EXTENT, WHICH IS THE PROPERTY THEY EXIST FOR: a point
    source deposits into a handful of cells and an extended one into a volume, so the
    repair's per-point loop is exercised at many points rather than at four. The
    COMPONENT differs too, so no single component's Yee table is the only one asked.

    A FOLDED PLANE ADMITS ONLY ITS EVEN COMPONENTS SINCE 2026-08-31, and that is the
    engine's own refusal rather than a choice made here:
    ``sources._validate_symmetry_parity`` rejects a source centred on a plane that
    gives it parity -1, because MEEP does not and instead returns a smooth, plausible
    field 57%-283% away from the full-domain run. ``fields.mirror_parity(c, axis,
    phase)`` is ``phase * (1 - 2*iyee)``, and for the E family a SINGLE even plane on
    axis ``a`` leaves the two components whose Yee shift there is 0 -- i.e. every
    component except the one along ``a`` -- so a folded row still has two to choose
    from and does not have to collapse the extended row onto the point one.
    """
    from meep_gpu.fields import mirror_parity  # noqa: PLC0415

    folded = {"XYZ".index(axis): int(phase)
              for axis, phase in spec.get("symmetry", ())}
    admissible = [name for name in ("Ez", "Ey", "Ex")
                  if all(mirror_parity(name, axis, phase) == 1
                         for axis, phase in folded.items())]
    if not admissible:
        raise SystemExit(
            f"spec {spec['label']} leaves NO E component even about its mirror "
            f"planes, so no in-seam electric deposit can be built on it at all and "
            f"the row would report a refusal as an absence")
    point, extent = admissible[0], admissible[min(1, len(admissible) - 1)]
    return (("point", point, (0.0, 0.0, 0.0)), ("extent", extent, EXTENDED_SIZE))


def _electric_source(fields, component: str = "Ez",
                     size: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     centre: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """One REAL engine electric source, built against THIS engine's own grid.

    Never a stub with a hand-set ``field_type``: a stub would let this leg pass while
    the engine classified the same component the other way. ``VolumeSource`` resolves
    the slot itself through ``_field_type_for``.

    ``centre`` is an argument since 2026-08-31 so the image-closure leg can PLACE a
    deposit on a row a post-injection fill reads from, rather than depending on where a
    centred source happens to land -- a deposit that missed every imaged row would make
    that leg's point-only control vacuous for a reason about the fixture.
    """
    from meep_gpu.sources import ContinuousEnvelope, VolumeSource  # noqa: PLC0415

    return VolumeSource(grid=fields.grid, component=component,
                        center=tuple(centre), size=size,
                        envelope=ContinuousEnvelope(frequency=DEPOSIT_FREQUENCY))


def _withdraw(sources: Sequence[Any], fields) -> None:
    """``driver.py:3299``'s withdraw pass, on both sides identically."""
    for source in sources:
        hook = getattr(source, "withdraw", None)
        if callable(hook):
            hook(fields)


def deposit_magnitude(fields, source, steps: int, dt: float) -> float:
    """The largest standing deposit this source leaves over the step budget.

    THE PRECONDITION THE DEPOSIT LEGS CANNOT DO WITHOUT. A source whose real part is
    zero at every sampled time deposits nothing, and then "the repair reproduces the
    driver" and "there was nothing to reproduce" are the same green row. Measured on a
    THROWAWAY copy of the field state, restored before the leg runs, and SAMPLED AT
    THE ELECTRIC SEAM'S OWN INSTANT (``t + dt/2``).
    """
    component = "D" + str(source.component)[-1]
    array = getattr(fields, component)
    base = cp.array(array, copy=True)
    try:
        largest = 0.0
        for step in range(1, steps + 1):
            _withdraw((source,), fields)
            source.inject(fields, (step - 1) * dt + 0.5 * dt)
            largest = max(largest, float(cp.abs(array - base).max()))
        return largest
    finally:
        array[...] = base


def deposits_on_a_cleared_cell(fields, source) -> int:
    """How many of this source's deposit points ``zero_metal_D`` also zeroes.

    THE ONE ASYMMETRY BETWEEN THE TWO PATHS, MEASURED RATHER THAN ARGUED. The fused
    kernel carries ``zero_metal_D`` INLINE and therefore clears before the injection;
    the driver clears AFTER it (driver.py:3310). The two can only differ where a
    deposit lands on a cell the clear zeroes, and both paths would then read the same
    final zero -- but "would" is a claim, so the count is measured per row.
    """
    index = deposit_repair._deposit_index(source)  # noqa: SLF001
    if index is None:
        return 0
    saved = {name: cp.array(getattr(fields, name), copy=True)
             for name in ("Dx", "Dy", "Dz")}
    try:
        for name in ("Dx", "Dy", "Dz"):
            getattr(fields, name)[...] = 1.0
        stepping.zero_metal_D(fields)
        component = "D" + str(source.component)[-1]
        return int(cp.count_nonzero(getattr(fields, component)[index] == 0.0))
    finally:
        for name, value in saved.items():
            getattr(fields, name)[...] = value


def run_deposit_case(spec: Dict[str, Any], value_class: str, steps: int,
                     repair: bool = True, component: str = "Ez",
                     size: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     centre: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     label_suffix: str = "",
                     ) -> Dict[str, Any]:
    """Complete driver steps with an electric source IN the seam, byte compared."""
    from meep_gpu.cuda_kernels import arms  # noqa: PLC0415

    started = time.time()
    label = (f"{spec['label']}|{value_class}|deposit|{component}|"
             f"{'repair' if repair else 'null'}{label_suffix}")
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

    reference_sources = (_electric_source(reference, component, size, centre),)
    actual_sources = (_electric_source(actual, component, size, centre),)
    case["source_centre"] = list(centre)
    case["source_field_type"] = str(actual_sources[0].field_type)
    case["deposit_points_on_a_cleared_cell"] = deposits_on_a_cleared_cell(
        actual, actual_sources[0])
    index = deposit_repair._deposit_index(actual_sources[0])  # noqa: SLF001
    if index is None or len(deposit_repair.in_seam_sources(actual_sources, "D")) != 1:
        case["passed"] = False
        case["why"] = ("the source this leg built is not an in-seam D deposit; the "
                       "walk would inject nothing and compare a no-op with a no-op")
        return case
    case["deposit_points"] = int(cp.asarray(index[0]).size)
    case["deposit_magnitude"] = deposit_magnitude(
        actual, _electric_source(actual, component, size, centre), steps,
        float(grid.dt))
    if case["deposit_magnitude"] <= 0.0:
        case["passed"] = False
        case["why"] = ("this source deposits exactly zero over the whole budget, so "
                       "the repaired and unrepaired walks would agree with each "
                       "other and with a no-op")
        return case

    # THE SHIPPED COMPOSER BUILDS THE PLAN, not this file. Reaching for
    # `run_fused_electric_pair` directly would test the kernel and skip the wiring,
    # and the wiring is what this leg exists for.
    composed = arms.plan_step(actual, pml, grid,
                              sources=(actual_sources if repair else ()), fuse=True)
    leading = composed.plans.get("step_D")
    trailing = composed.plans.get("update_E")
    installed = (type(leading).__name__, type(trailing).__name__)
    case["installed_plans"] = list(installed)
    case["selected"] = {slot: composed.selected.get(slot)
                        for slot in ("step_D", "update_E")}
    if repair:
        installed_ok = installed == ("LeadingRepairPlan", "TrailingRepairPlan")
    else:
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
            if name == "step_D":
                for source in reference_sources:
                    source.inject(reference, when + 0.5 * dt)

        _withdraw(actual_sources, actual)
        for name in DRIVER_ORDER:
            if name == "step_D":
                leading.run()
                for source in actual_sources:
                    source.inject(actual, when + 0.5 * dt)
            elif name == "update_E":
                trailing.run()
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


class _PointOnly:
    """``deposit_repair`` with the image closure REMOVED -- the middle leg's engine.

    ``repair_cells`` is monkeypatched to return the deposit index alone, which is
    exactly what a repair written without :func:`deposit_repair.fill_image_rules` would
    save and restore. Restored on exit, so nothing else in the process sees it.
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


def leg_deposit_image_closure(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """THREE LEGS ON A FOLDED SEAM: full closure, point-only, no repair.

    WHAT THIS ANSWERS THAT THE DEPOSIT LEGS DO NOT. The two fills run AFTER the
    injection (driver.py:3309, :3311), so a repair that wrote only the deposit INDEX
    would leave every MIRROR IMAGE of that index describing the pre-injection field.
    ``deposit_repair.repair_cells`` hands ``save``/``apply`` the CLOSURE of those
    images; this leg is where that is measured on the electric seam rather than
    inherited from the magnetic one.

      A  full closure -- MUST be bit-identical;
      B  point-only   -- MUST diverge;
      C  no repair    -- MUST diverge.

    A CASE WHERE B DOES NOT DIVERGE IS VACUOUS AND IS REPORTED AS A FAILURE, not as a
    pass. That is not hypothetical on this product: while it refused every fold,
    ``fill_image_rules`` returned the EMPTY rule set on every configuration it
    admitted, so ``repair_cells(index)`` WAS ``index`` and leg B was leg A by
    construction (``probe_cuda_electric_fold_closure.py``, 2026-08-31). The fold is
    admitted now, so the protocol is owed for real.

    THE DEPOSIT IS PLACED ON AN IMAGED ROW rather than left wherever a centred source
    happens to land: ``fill_image_rules`` names the row each fill READS FROM, and a
    deposit that missed every one of them would make B vacuous for a reason about the
    fixture. The rule set and the two cell counts are recorded, so the reader can see
    the closure is strictly larger than the point set before reading the verdict.
    """
    started = time.time()
    # THE COMPONENT IS DERIVED, not declared: a folded plane admits only the E
    # components it leaves EVEN (``sources._validate_symmetry_parity``), and on
    # ``fold_xy_mixed_phase`` that is exactly one. Asking for a fixed ``Ez`` there
    # would raise inside the engine's own refusal and report a fixture bug as a gate
    # failure.
    component = deposit_rows(spec)[0][1]
    probe_fields, probe_grid, probe_pml, _ = build(
        spec, "uniform", case_rng(f"{spec['label']}|clos"))
    rules = deposit_repair.fill_image_rules(probe_fields, f"D{component[-1]}")
    # PLACE THE DEPOSIT ON A ROW A FILL READS FROM, BY MEASUREMENT RATHER THAN BY
    # ARITHMETIC. `fill_image_rules` gives (axis, source row, destination row) in
    # STORED cells, and converting a stored row back to a `center` on a FOLDED axis is
    # exactly the kind of index arithmetic this campaign does not do by hand: the
    # halved grid's origin sits at MEEP's `io = -2`, the stored half is the positive
    # side, and a centre computed the obvious way lands in the half the fold discards
    # (which the engine refuses by name -- measured on the first device run). So the
    # candidates are SWEPT and the engine is ASKED which row each one deposits on.
    size = (0.0, 0.0, 0.0)
    centre = [0.0, 0.0, 0.0]
    placed_on = None
    source = None
    index = None
    for axis, source_row, _destination in rules:
        span = float(spec["cell"][axis]) * 0.5
        for candidate in np.linspace(0.0, span,
                                     4 * int(probe_grid.stored_cells(axis)) + 1):
            trial = list(centre)
            trial[axis] = float(candidate)
            try:
                built = _electric_source(probe_fields, component, size,
                                         centre=tuple(trial))
            except ValueError:
                continue  # the engine's own parity / discarded-half refusals
            rows = deposit_repair._deposit_index(built)  # noqa: SLF001
            if rows is None:
                continue
            if int(source_row) in set(np.asarray(to_host(rows[axis])).ravel().tolist()):
                centre, source, index = trial, built, rows
                placed_on = {"axis": int(axis), "source_row": int(source_row),
                             "centre": float(candidate)}
                break
        if source is not None:
            break
    point_cells = int(cp.asarray(index[0]).size) if index is not None else 0
    closure = deposit_repair.repair_cells(probe_fields, f"D{component[-1]}", index) \
        if index is not None else None
    closure_cells = int(cp.asarray(closure[0]).size) if closure is not None else 0
    if source is None:
        # NEVER A SILENT PASS. A leg that could not place the deposit on an imaged row
        # has measured nothing about the closure, and says so.
        return {
            "passed": False, "spec": spec["label"], "component": component,
            "fill_image_rules": [list(map(int, rule)) for rule in rules],
            "deposit_placed_on": None, "point_cells": 0, "closure_cells": 0,
            "control_is_vacuous": True,
            "why": ("no candidate centre deposited on a row either fill READS FROM, so "
                    "the point-only control could not differ from the closure and this "
                    "leg would have measured nothing"),
            "seconds": time.time() - started,
        }

    legs: Dict[str, Any] = {}
    legs["A_full_closure"] = run_deposit_case(
        spec, "uniform", steps, repair=True, component=component, size=size,
        centre=tuple(centre), label_suffix="|closure")
    with _PointOnly():
        legs["B_point_only"] = run_deposit_case(
            spec, "uniform", steps, repair=True, component=component, size=size,
            centre=tuple(centre), label_suffix="|point_only")
    legs["C_no_repair"] = run_deposit_case(
        spec, "uniform", steps, repair=False, component=component, size=size,
        centre=tuple(centre), label_suffix="|none")

    a_identical = bool(legs["A_full_closure"].get("bit_identical"))
    b_diverged = not legs["B_point_only"].get("bit_identical", False)
    c_diverged = not legs["C_no_repair"].get("bit_identical", False)
    vacuous = closure_cells <= point_cells
    return {
        "passed": bool(a_identical and b_diverged and c_diverged and not vacuous),
        "spec": spec["label"],
        "component": component,
        "fill_image_rules": [list(map(int, rule)) for rule in rules],
        "deposit_placed_on": placed_on,
        "deposit_centre": list(centre),
        "point_cells": point_cells,
        "closure_cells": closure_cells,
        "closure_is_strictly_larger": bool(not vacuous),
        "control_is_vacuous": bool(vacuous),
        "A_full_closure_identical": a_identical,
        "B_point_only_diverged": b_diverged,
        "C_no_repair_diverged": c_diverged,
        "A_differing_words": legs["A_full_closure"].get("differing_words"),
        "B_differing_words": legs["B_point_only"].get("differing_words"),
        "C_differing_words": legs["C_no_repair"].get("differing_words"),
        "B_differing_arrays": legs["B_point_only"].get("differing_arrays"),
        "deposit_magnitude": legs["A_full_closure"].get("deposit_magnitude"),
        "deposit_points_repaired": legs["A_full_closure"].get(
            "deposit_points_repaired"),
        "seconds": time.time() - started,
    }


def leg_deposit_null_control(spec: Dict[str, Any], steps: int) -> Dict[str, Any]:
    """The unbracketed launch MUST diverge. Scored on diverging, not on agreeing.

    Without it every green deposit case above is consistent with a deposit too small
    to see -- which is not hypothetical: it is what the magnetic gate's first device
    run found, and the reason :data:`DEPOSIT_FREQUENCY` is what it is.
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
        "deposit_magnitude": result.get("deposit_magnitude"),
    }


# ---------------------------------------------------------------------------
# The mutations
# ---------------------------------------------------------------------------

def needle(source: str, old: str, new: str, count: int = 1) -> Tuple[str, int]:
    """Replace and REPORT THE SITE COUNT. A rewrite that matched nothing comes back
    NOT ARMED rather than launching the shipped kernel and reporting the defect as
    uncaught -- the one failure mode a mutation leg cannot see from its own result."""
    sites = source.count(old)
    if sites == 0:
        return source, 0
    return source.replace(old, new, count if count > 0 else -1), sites


def _folded_axes(spec: Dict[str, Any]) -> set:
    return {"XYZ".index(axis) for axis, _phase in spec.get("symmetry", ())}


def _wall(axis: str) -> Callable[[Dict[str, Any]], bool]:
    """Is ``axis`` a live WALL here -- ``is_metallic and not is_mirrored``?

    THE FOLD CLAUSE IS NOT DECORATION SINCE 2026-08-31. ``stepping._zero_metal``
    excludes a folded axis (:2237-2239), so ``fold_y_metallic`` declares a metallic y
    and carries NO wall there; a predicate that read the declaration alone would score
    a wall mutation on a row where the shipped launch binds ``wall_y = 0`` and the
    mutation is not present at all.
    """
    index = "xyz".index(axis)

    def applies(spec: Dict[str, Any]) -> bool:
        return (spec["boundaries"][index] == "metallic"
                and index not in _folded_axes(spec))

    return applies


def _any_wall(spec: Dict[str, Any]) -> bool:
    return any(kind == "metallic" and index not in _folded_axes(spec)
               for index, kind in enumerate(spec["boundaries"]))


def _near_fill(spec: Dict[str, Any]) -> bool:
    """Does ``fill_symmetry_bc_D`` run here? Its condition is a declared mirror phase,
    which a folded METALLIC axis carries just as a folded PERIODIC one does."""
    return bool(spec.get("symmetry"))


def _far_fill(spec: Dict[str, Any]) -> bool:
    """Does ``fill_folded_far_ghosts_D`` run here? Only on a folded PERIODIC axis:
    ``stepping._far_reflect_rows`` returns None on a folded METALLIC one, where the
    unstored zero ghost IS the boundary condition."""
    return any(spec["boundaries"]["XYZ".index(axis)] != "metallic"
               for axis, _phase in spec.get("symmetry", ()))


def _two_folded_axes(spec: Dict[str, Any]) -> bool:
    """The composition: a cell at stored 0 of two near axes carries the PRODUCT of two
    parities, and a carry source can itself be a fill destination."""
    return len(spec.get("symmetry", ())) >= 2


def _folded(axis: str) -> Callable[[Dict[str, Any]], bool]:
    """Is this axis MIRROR-folded here (either termination)?"""
    def applies(spec: Dict[str, Any]) -> bool:
        return any(name == axis for name, _phase in spec.get("symmetry", ()))
    return applies


def _clears(component: str) -> Callable[[Dict[str, Any]], bool]:
    """Is ``clr_<component>`` EVER 1 on this spec?

    THE OFF-DIAGONAL, AND IT IS NOT THE SAME QUESTION AS "IS ANY AXIS WALLED".
    ``zero_metal_D`` clears a D component on the two axes that are NOT its own, so a
    row that walls ONLY x never sets ``clr_x`` and every mutation to Dx's clear is
    inert there BY ARITHMETIC. MEASURED on the GPU host 2026-08-31 before this predicate
    existed: ``wall_clear_misses_global`` (which edits Dx's clear) came back 3/4, and
    the one leg that did not catch it was ``wall_x``.
    """
    others = tuple(a for a in "xyz" if a != component)

    def applies(spec: Dict[str, Any]) -> bool:
        return any(_wall(axis)(spec) for axis in others)

    return applies


def _has_ghosts(component: str) -> Callable[[Dict[str, Any]], bool]:
    """Does ``own_<component>`` ever read 0 on this spec -- has it ANY ghost cell?

    A D component is imaged on its two NEAR axes (any fold) and on its OWN axis (a
    folded PERIODIC one only). MEASURED before this predicate existed:
    ``ownership_guard_dropped_on_the_constitutive`` edits Ey's guard and came back
    4/5, and the leg that did not catch it was ``fold_y_metallic`` -- where y is the
    only folded axis, y IS Dy's own axis, and a folded METALLIC axis runs no far fill,
    so ``own_y`` is identically 1 and ``if (own_y)`` -> ``if (1)`` is the shipped
    kernel.
    """
    own = "XYZ"["xyz".index(component)]
    others = tuple("XYZ"["xyz".index(a)] for a in "xyz" if a != component)

    def applies(spec: Dict[str, Any]) -> bool:
        near = any(_folded(axis)(spec) for axis in others)
        far = any(name == own and spec["boundaries"]["XYZ".index(name)] != "metallic"
                  for name, _phase in spec.get("symmetry", ()))
        return bool(near or far)

    return applies


def _cross_term(component: str) -> Callable[[Dict[str, Any]], bool]:
    """THE CROSS TERM, FOR ONE COMPONENT, and it belongs to a component rather than to
    a spec -- which is what the device measured.

    The order between ``fill_symmetry_bc_D`` (:3309) and ``zero_metal_D`` (:3310) is
    observable only where ONE component is folded on one of ITS near axes at an ODD
    phase AND walled on ANOTHER of its near axes: the near ghost then lands in the
    cleared plane, where the array path leaves ``+0.0f`` and multiplying the
    post-clear register by ``-1`` leaves ``-0.0f``. An EVEN phase makes it vacuous
    (``+1 * 0.0f`` is ``0.0f``), and so does a wall on an axis this component is not
    cleared on.

    MEASURED on the GPU host 2026-08-31 before this predicate was component-specific: a
    leg anchored on Dx's near-y ghost scored 1/2, and the row it did not catch was
    ``fold_y_odd_wall_x`` -- where the walled axis is x, which is Dz's clear axis and
    not Dx's. The two rows carry the cross term on DIFFERENT components, so the two
    are scored separately and each is armed on its own component's ghost.
    """
    others = tuple("XYZ"["xyz".index(a)] for a in "xyz" if a != component)

    def applies(spec: Dict[str, Any]) -> bool:
        odd = any(name in others and int(phase) == -1
                  for name, phase in spec.get("symmetry", ()))
        return bool(odd and _clears(component)(spec))

    return applies


def _some_axis_unwalled(spec: Dict[str, Any]) -> bool:
    """Is there an axis this spec does NOT wall?

    THE APPLICABILITY OF ``walls_asserted_everywhere``, and the reason is arithmetic
    rather than an expectation. That mutation hands the launcher ``(1, 1, 1)``; on a
    row whose three axes are already metallic the shipped reading IS ``(1, 1, 1)``, so
    the patched launch is the shipped launch and there is no defect present to catch.
    MEASURED on the GPU host 2026-08-30 before this clause existed: caught 2/2 on
    ``all_periodic`` and ``wall_x``, uncaught 0/2 on ``wall_xyz`` and
    ``wall_xyz_isotropic`` -- exactly the two fully-walled rows. Scoring those two
    would collapse "the mutation was not present" into "the gate could not see it",
    which is the bookkeeping every ``applies`` predicate in this table exists to keep
    apart.
    """
    return any(kind != "metallic" for kind in spec["boundaries"])


def _anisotropic(spec: Dict[str, Any]) -> bool:
    return spec.get("epsilon", "isotropic") == "anisotropic"


def _always(spec: Dict[str, Any]) -> bool:
    return True


SOURCE_MUTATIONS: Tuple[Tuple[str, Tuple[str, str, int], Optional[bool],
                              Callable[[Dict[str, Any]], bool]], ...] = (
    # --- the seam itself ----------------------------------------------------
    # The wall clear must reach the REGISTER as well as global memory, or update_E
    # consumes the un-wiped flux density -- the same error one sub-step earlier than
    # leaving global D dirty.
    ("wall_clear_misses_the_register",
     ("if (own_y && clr_y) { d_y = 0.0f; Dy[idx] = d_y; }",
      "if (own_y && clr_y) { Dy[idx] = 0.0f; }", 1),
     True, _clears("y")),
    # ... and it must reach GLOBAL MEMORY as well as the register, or the next
    # timestep's recurrence reads a plane of un-wiped values. Caught from step 2,
    # which is why the budget is complete steps and not one launch.
    ("wall_clear_misses_global",
     ("if (own_x && clr_x) { d_x = 0.0f; Dx[idx] = d_x; }",
      "if (own_x && clr_x) { d_x = 0.0f; }", 1), True, _clears("x")),
    # THE D TABLE IS THE OFF-DIAGONAL; the B family's is the diagonal. Reusing it
    # clears Dx on the x wall and leaves Dy and Dz standing -- the single most likely
    # slip in porting this family from its B-side sibling, and the one this gate was
    # shaped around.
    ("wall_clear_uses_the_b_diagonal",
     ("if (wall_x && i == 0) { clr_y = 1; clr_z = 1; }",
      "if (wall_x && i == 0) { clr_x = 1; }", 1), True, _wall("x")),
    # ONE OF THE PAIR DROPPED. A weaker defect than the whole table being wrong, and
    # the one a hand edit is most likely to leave behind.
    ("wall_clear_drops_one_of_the_pair",
     ("if (wall_y && j == 0) { clr_x = 1; clr_z = 1; }",
      "if (wall_y && j == 0) { clr_x = 1; }", 1), True, _wall("y")),
    # The wall plane is stored cell 0 of the WALLED axis, not of another one.
    ("wall_clear_uses_the_wrong_coordinate",
     ("if (wall_x && i == 0) {", "if (wall_x && j == 0) {", 1), True, _wall("x")),
    # THE WALL CLEAR IS GUARDED ON OWNERSHIP, AND THIS LEG IS A MEASUREMENT RATHER
    # THAN A REQUIRED CATCH -- ``must_be_caught`` is None and REPORTED is the expected
    # verdict. The guard exists so that no cell is written by two threads; it is NOT
    # there to change a value, and on this family it CANNOT: a near ghost only moves a
    # FOLDED axis, a folded axis is never walled, so the destination's clear test and
    # the source thread's are the same test on the same coordinates (which is the
    # identity the whole carry rests on). Both writers therefore write the same word
    # and the race is benign. Scored on a row where the edited line is REACHABLE
    # (``clr_z`` live and Dz imaged) so the reading is about the arithmetic rather
    # than about an unexecuted line.
    ("wall_clear_unguarded_on_ownership",
     ("if (own_z && clr_z) { d_z = 0.0f; Dz[idx] = d_z; }",
      "if (clr_z) { d_z = 0.0f; Dz[idx] = d_z; }", 1),
     None, lambda spec: _clears("z")(spec) and _has_ghosts("z")(spec)),
    # The seam, wrong component: Ex's source is d_x, not d_y.
    ("seam_takes_the_wrong_component",
     ("float src_x = d_x * inv_eps_Ex[idx];",
      "float src_x = d_y * inv_eps_Ex[idx];", 1), True, _always),
    # THE PER-COMPONENT INVERSE PERMITTIVITY. On an isotropic row all three pointers
    # are ONE allocation and this is inert BY ARITHMETIC -- which is exactly why half
    # the sweep is anisotropic and why this leg is scored only there.
    ("inverse_permittivity_component_swapped",
     ("float src_x = d_x * inv_eps_Ex[idx];",
      "float src_x = d_x * inv_eps_Ez[idx];", 1), True, _anisotropic),
    # THE SUB-LATTICE SHADOW, and this defect exists only because the fusion put both
    # halves in one scope. The curl's kms is INTEGER and the constitutive's
    # HALF-INTEGER; letting the certified name shadow is a half-cell error in the
    # absorber profile -- converged, smooth and wrong.
    ("constitutive_sublattice_shadow",
     ("constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_half_x[i]);",
      "constitutive_apply(Ex, f_w_Ex, idx, src_x, kps_x[i], kms_x[i]);", 1),
     True, _always),
    # The dsigw index is the component's OWN axis (stepping.py:1015).
    ("constitutive_coefficient_index_moved",
     ("constitutive_apply(Ey, f_w_Ey, idx, src_y, kps_y[j], kms_half_y[j]);",
      "constitutive_apply(Ey, f_w_Ey, idx, src_y, kps_y[i], kms_half_y[i]);", 1),
     True, _always),
    # The two accumulations are LEFT-TO-RIGHT and SEPARATE; reassociating them is a
    # different float32 number.
    ("constitutive_accumulations_reversed",
     ("    float a = f[idx] + kps * src;\n    f[idx] = a - kms * prev;",
      "    float a = f[idx] - kms * prev;\n    f[idx] = a + kps * src;", 1),
     True, _always),
    # THE D CURL'S BACKWARD SHIFT. step_D differences DOWN and step_B UP; a pair
    # ported from the magnetic sibling without changing this is a different engine.
    ("curl_shift_direction_flipped",
     ("float sf = shift_dn(Hz, idx, j, ny, sy, bc_y);",
      "float sf = shift_up(Hz, idx, j, ny, sy, bc_y);", 1), True, _always),
    # The metallic cell-0 mask on the curl. Distinct from the wall CLEAR: this one
    # zeroes the curl term, the other zeroes the stored flux density. ANCHORED ON THE
    # PAIR, not on the line: `bc_y == BC_METALLIC && j == 0` appears in the Dx block
    # too, and a needle that matched twice would arm whichever the replace happened to
    # reach first -- a mutation leg reporting a verdict about a line nobody chose.
    ("curl_metallic_mask_dropped",
     ("        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;\n"
      "        if (bc_y == BC_METALLIC && j == 0) curl = 0.0f;\n",
      "        if (bc_x == BC_METALLIC && i == 0) curl = 0.0f;\n", 1),
     True, _wall("y")),
    # The recurrence's own order: fu is stored BEFORE the flux density reads fprev.
    ("pml_recurrence_reads_the_fresh_auxiliary",
     ("    float value = (((f[idx] * kms_u) + fu_new) - fprev) * sinv_u;",
      "    float value = (((f[idx] * kms_u) + fu_new) - fu_new) * sinv_u;", 1),
     True, _always),
    # THE CAPTURE ITSELF. A weld that stored the flux density and let the
    # constitutive half read a stale register is the defect the whole product is
    # about; here the register is left at its declared zero.
    ("seam_reads_a_stale_register",
     ("    d_x = pml_apply_reg(Dx, fu_Dx, ",
      "    pml_apply_reg(Dx, fu_Dx, ", 1), True, _always),

    # --- THE FILL CARRY, NEW 2026-08-31 -------------------------------------
    #
    # Every one of these is inert on an unfolded row BY ARITHMETIC -- neither fill
    # runs, so the guarded blocks are never entered -- which is why each carries an
    # ``applies`` predicate naming the fold it needs. Scoring them on the unfolded
    # rows would collapse "the mutation was not present" into "the gate could not see
    # it".
    #
    # THE OWNERSHIP INVERSION. A thread standing on a cell a fill images must stop
    # after fu: forming the displacement reads a D word another block writes in this
    # same launch. Inverting the flag makes every owner stop and every ghost run.
    ("ownership_guard_inverted",
     ("    int own_x = !(", "    int own_x = (", 1), True, _has_ghosts("x")),
    # DROPPED ON THE CONSTITUTIVE ONLY: the curl still skips the ghost's store, but
    # the thread standing on the ghost now runs update_E there as well as its source
    # thread -- two threads on one word.
    ("ownership_guard_dropped_on_the_constitutive",
     ("    if (own_y) {\n        constitutive_apply(Ey,",
      "    if (1) {\n        constitutive_apply(Ey,", 1), True, _has_ghosts("y")),
    # THE NEAR FILL'S SOURCE ROW is MEEP's io = -2 halved origin: stored 2, never 3.
    # TWO SITES ON THIS FAMILY and the count says so: near_y is read by Dx AND Dz
    # (every component whose Yee shift on y is 0), where on the B family each near
    # guard belonged to one component. Both are edited, and a change in that number is
    # NOT ARMED rather than an arbitrary one of them.
    ("near_fill_source_guard_moved",
     ("if (near_y && j == 2) {", "if (near_y && j == 3) {", 2), True,
     lambda spec: _near_fill(spec) and any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ()))),
    # ...and the DESTINATION row, which the offset carries. Anchored on the tagged
    # index line, so it names one component's one ghost.
    ("near_fill_writes_the_wrong_destination_row",
     ("int gx_ny_i = idx - 2 * sy;", "int gx_ny_i = idx - 1 * sy;", 1), True,
     lambda spec: _near_fill(spec) and any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ()))),
    # THE NEAR WEIGHT is the plane's bare phase (Yee shift 0 there); negating it is
    # the far fill's weight applied to the near ghost.
    ("near_fill_parity_negated",
     ("float gx_ny_v = phase_y * pre_x;", "float gx_ny_v = -phase_y * pre_x;", 1),
     True, lambda spec: _near_fill(spec) and any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ()))),
    # THE FAR WEIGHT is -phase (Yee shift 1 there). Un-negating it is exact at no
    # phase and wrong at both.
    ("far_fill_parity_unnegated",
     ("gx_x_v = (-phase_x) * gx_x_v;", "gx_x_v = (phase_x) * gx_x_v;", 1), True,
     lambda spec: _far_fill(spec) and any(
         axis == "X" for axis, _phase in spec.get("symmetry", ()))),
    # THE REFLECT ROW IS A RUNTIME ARGUMENT, not `n - 2`. Baking it is exact at an
    # even full count and a whole cell wrong at an odd one.
    ("far_fill_bakes_n_minus_two",
     ("int gx_x_i = idx + (nx - 1 - reflect_x) * sx;",
      "int gx_x_i = idx + (nx - 1 - (nx - 2)) * sx;", 1), True,
     lambda spec: _far_fill(spec) and any(
         axis == "X" for axis, _phase in spec.get("symmetry", ()))),
    # THE DRIVER'S ORDER BETWEEN :3309 AND :3310, and this is the defect a port of
    # the B family's carry would ship: the near product formed from the POST-clear
    # register instead of the pre-clear one. Bit-identical on the B seam and on every
    # EVEN-phase row here; -0.0f against +0.0f on an odd plane whose component is
    # walled on its other near axis.
    # BOTH LINES, AND THE COUNT IS THE POINT. Editing only the register leaves the
    # clear standing one line below, and the clear then rewrites the sign-flipped zero
    # back to +0.0f -- the mutation is MASKED and the leg reports UNCAUGHT for a reason
    # about the harness. MEASURED on the GPU host 2026-08-31: the single-line form scored
    # 0/1. A naive port of fused_magnetic_pair's carry has NEITHER the pre-clear
    # register NOR the separate clear (the B family needs neither), so the mutation
    # that reproduces it removes both at once.
    ("near_fill_reads_the_post_clear_register",
     ("            float gx_ny_v = phase_y * pre_x;\n"
      "            // zero_metal_D (:3310) at the DESTINATION. Its walled coordinates are\n"
      "            // this thread's -- a near ghost only moves a FOLDED axis, and a folded\n"
      "            // axis is never walled -- so the source thread's own flag IS the test.\n"
      "            if (clr_x) gx_ny_v = 0.0f;\n",
      "            float gx_ny_v = phase_y * d_x;\n", 1),
     True, _cross_term("x")),
    # THE SAME DEFECT ON THE OTHER COMPONENT, and it is a separate leg rather than a
    # wider predicate: on ``fold_y_odd_wall_z`` the cross term belongs to Dx (folded on
    # y, walled on z) and on ``fold_y_odd_wall_x`` it belongs to Dz (folded on y,
    # walled on x). One anchor cannot reach both, and a predicate that scored one
    # anchor on both rows reports the row where it is absent as a gate that cannot see
    # it.
    ("near_fill_reads_the_post_clear_register_on_dz",
     ("            float gz_ny_v = phase_y * pre_z;\n"
      "            // zero_metal_D (:3310) at the DESTINATION. Its walled coordinates are\n"
      "            // this thread's -- a near ghost only moves a FOLDED axis, and a folded\n"
      "            // axis is never walled -- so the source thread's own flag IS the test.\n"
      "            if (clr_z) gz_ny_v = 0.0f;\n",
      "            float gz_ny_v = phase_y * d_z;\n", 1),
     True, _cross_term("z")),
    # ...and the clear at the near ghost dropped entirely.
    ("near_ghost_clear_dropped",
     ("if (clr_x) gx_ny_v = 0.0f;", "if (0) gx_ny_v = 0.0f;", 1), True,
     _cross_term("x")),
    ("near_ghost_clear_dropped_on_dz",
     ("if (clr_z) gz_ny_v = 0.0f;", "if (0) gz_ny_v = 0.0f;", 1), True,
     _cross_term("z")),
    # THE COMPOSED PARITY. On two folded axes a corner carries the PRODUCT; dropping
    # a factor is exact on every single-axis row and wrong only here.
    # ANCHORED ON Dz, whose two NEAR axes are x and y -- the pair the sweep's
    # two-axis row actually folds. Dx's near pair is y and z and no spec folds both,
    # so the Dx anchor this started as scored NO LEGS on the device.
    ("composed_parity_drops_a_factor",
     ("float gz_nxy_v = phase_x * phase_y * pre_z;",
      "float gz_nxy_v = phase_x * pre_z;", 1), True,
     lambda spec: {"X", "Y"} <= {axis for axis, _p in spec.get("symmetry", ())}),
    # THE FAR GHOST'S COEFFICIENT PAIR is the DESTINATION's, at n - 1, not this
    # thread's at the reflect row. Byte-identical wherever the absorber is flat across
    # those two rows, which is why fold_x_deep_pml exists.
    # FOUR SITES: Dx has four far destinations (the far axis alone, and with each of
    # the two near subsets and both together), and every one of them takes the top
    # plane's pair. The count is stated so a change in the destination table is NOT
    # ARMED rather than a partial edit.
    ("far_ghost_reuses_the_source_coefficient_pair",
     ("kps_x[nx - 1], kms_half_x[nx - 1]);", "kps_x[i], kms_half_x[i]);", 4), True,
     lambda spec: _far_fill(spec) and any(
         axis == "X" for axis, _phase in spec.get("symmetry", ()))),
    # THE GHOST'S FLUX-DENSITY STORE DROPPED, and this one is not hypothetical: it is
    # what the FIRST device run of this carry found, at 297 differing words on a 9x7x11
    # single-fold row -- exactly three ghost planes of 99. D at a ghost is what the next
    # timestep's curl differences, so a carry that ran update_E there and never wrote
    # the imaged displacement back is a plane of wrong operands one step later.
    ("ghost_flux_density_store_dropped",
     ("            Dx[gx_ny_i] = gx_ny_v;\n", "", 1), True,
     lambda spec: _near_fill(spec) and any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ()))),
    # THE GHOST'S CONSTITUTIVE STEP DROPPED. The fills write D; E and f_w_E at a ghost
    # are written by nothing but update_E, so a carry that imaged the displacement and
    # stopped leaves the constitutive state at the ghost describing the previous step.
    ("ghost_constitutive_dropped",
     ("constitutive_apply(Ey, f_w_Ey, gy_nx_i, gy_nx_s, ",
      "constitutive_apply(Ey, f_w_Ey, idx, gy_nx_s, ", 1), True,
     lambda spec: _near_fill(spec) and any(
         axis == "X" for axis, _phase in spec.get("symmetry", ()))),
    # THE GHOST'S INVERSE PERMITTIVITY is read at the GHOST index, not at this
    # thread's -- inert on an isotropic row only if the volume happens to be uniform,
    # which it is not (the values are drawn).
    ("ghost_inverse_permittivity_read_at_the_source",
     ("float gx_ny_s = gx_ny_v * inv_eps_Ex[gx_ny_i];",
      "float gx_ny_s = gx_ny_v * inv_eps_Ex[idx];", 1), True,
     lambda spec: _near_fill(spec) and any(
         axis == "Y" for axis, _phase in spec.get("symmetry", ()))),
    # A NULL, AND IT MUST NOT BE CAUGHT: the two near weights are constants and their
    # product is formed before the register multiply, so commuting them is the same
    # float32 word.
    ("composed_parity_factors_commuted",
     ("float gz_nxy_v = phase_x * phase_y * pre_z;",
      "float gz_nxy_v = phase_y * phase_x * pre_z;", 1), False,
     lambda spec: {"X", "Y"} <= {axis for axis, _p in spec.get("symmetry", ())}),
)


def _integer_curl_tables(tables: Dict[str, Any], pml: Any) -> Dict[str, Any]:
    """THE CURL'S OWN LATTICE, handed to it -- a NULL, and it must NOT be caught."""
    from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415

    return {"curl": step_curl_kernels.real_pml_curl_tables(pml, False),
            "constitutive": tables["constitutive"]}


def _half_integer_curl_tables(tables: Dict[str, Any], pml: Any) -> Dict[str, Any]:
    """The B curl's lattice on the D curl. Half a cell wrong, and it must be caught."""
    from meep_gpu.cuda_kernels import step_curl_kernels  # noqa: PLC0415

    return {"curl": step_curl_kernels.real_pml_curl_tables(pml, True),
            "constitutive": tables["constitutive"]}


def _integer_constitutive_tables(tables: Dict[str, Any], pml: Any) -> Dict[str, Any]:
    """update_H's lattice on update_E. The mirror of the mutation above."""
    from meep_gpu.cuda_kernels import constitutive_kernels  # noqa: PLC0415

    return {"curl": tables["curl"],
            "constitutive": constitutive_kernels.real_constitutive_tables(pml, False)}


def _codes_all_periodic(codes: Sequence[Any]) -> Sequence[Any]:
    return tuple(type(code)(0) for code in codes)


def _walls_dropped(walls: Sequence[int], grid: Any) -> Sequence[int]:
    return (0, 0, 0)


def _walls_all_on(walls: Sequence[int], grid: Any) -> Sequence[int]:
    return (1, 1, 1)


def _fills_dropped(fills: Dict[str, Any], grid: Any) -> Dict[str, Any]:
    """Both fills declared absent. THE WHOLE CARRY, unperformed -- which is exactly
    what this product did on a folded grid until 2026-08-31, and it must be caught on
    every folded row."""
    return {"near": (0, 0, 0), "reflect": (-1, -1, -1),
            "phase": tuple(fills["phase"])}


def _fills_from_walls(fills: Dict[str, Any], grid: Any) -> Dict[str, Any]:
    """The near flag read off the WALL table instead of the mirror phases.

    THREE READINGS OF ONE GRID and this confuses two of them. ``zero_metal_axes`` is
    ``is_metallic and not is_mirrored``; the near fill's condition is a declared mirror
    phase. On ``fold_y_metallic`` the two are exact complements on the folded axis.
    """
    from meep_gpu.cuda_kernels.in_seam_coverage import zero_metal_axes  # noqa: PLC0415

    walls = zero_metal_axes(grid)
    return {"near": tuple(int(bool(walls[axis])) for axis in range(3)),
            "reflect": tuple(fills["reflect"]), "phase": tuple(fills["phase"])}


def _fill_phases_flipped(fills: Dict[str, Any], grid: Any) -> Dict[str, Any]:
    """Every declared plane's parity inverted. An ODD plane read as EVEN is a run wrong
    by twice the field wherever the parity mattered, and it is smooth."""
    return {"near": tuple(fills["near"]), "reflect": tuple(fills["reflect"]),
            "phase": tuple(-float(value) for value in fills["phase"])}


def _far_reflect_row_shifted(fills: Dict[str, Any], grid: Any) -> Dict[str, Any]:
    """The far fill's image row moved one cell. The row is MEEP's lattice-translation
    image (stepping._far_reflect_rows); one off is a whole plane taken from the wrong
    place, and it is the defect a baked ``n - 2`` produces at an odd full count."""
    return {"near": tuple(fills["near"]),
            "reflect": tuple(row - 1 if row > 0 else row
                             for row in fills["reflect"]),
            "phase": tuple(fills["phase"])}


#: ``(name, kwargs for run_case, must_be_caught, applies)``.
HOST_MUTATIONS: Tuple[Tuple[str, Dict[str, Any], Optional[bool],
                            Callable[[Dict[str, Any]], bool]], ...] = (
    # THE PAIRING, FROM BOTH SIDES. This seam's curl reads the INTEGER lattice and its
    # constitutive the HALF-INTEGER one; the B/H seam is the other way round, so a
    # port that carried the pairing over is wrong on both groups at once.
    ("curl_takes_the_half_integer_lattice",
     {"tables_patch": _half_integer_curl_tables}, True, _always),
    ("constitutive_takes_the_integer_lattice",
     {"tables_patch": _integer_constitutive_tables}, True, _always),
    # A NULL: the curl's OWN lattice handed to it explicitly must change nothing.
    # Without it "caught" above could mean the harness diverges on any table object.
    ("curl_takes_its_own_integer_lattice",
     {"tables_patch": _integer_curl_tables}, False, _always),
    ("boundary_codes_all_periodic",
     {"codes_patch": _codes_all_periodic}, True, _wall("x")),
    ("walls_dropped", {"walls_patch": _walls_dropped}, True, _any_wall),
    # WALLS DECLARED WHERE THE GRID HAS NONE: the complement of the one above, and the
    # leg that shows the flags are read rather than ignored. SCORED ONLY WHERE AN AXIS
    # IS ACTUALLY UNWALLED -- on a fully metallic row the patched flags equal the
    # shipped ones and there is no defect present (:func:`_some_axis_unwalled`).
    ("walls_asserted_everywhere", {"walls_patch": _walls_all_on}, True,
     _some_axis_unwalled),
    # --- THE FILL PLAN, NEW 2026-08-31 --------------------------------------
    # The third reading of the grid, and the one the carry is driven by. Each is
    # scored only where the fill it corrupts is LIVE: on an unfolded row the shipped
    # plan already says "no fill" and the mutation is not present.
    ("fills_dropped", {"fills_patch": _fills_dropped}, True, _near_fill),
    ("fills_from_walls", {"fills_patch": _fills_from_walls}, True, _near_fill),
    ("fill_phases_flipped", {"fills_patch": _fill_phases_flipped}, True, _near_fill),
    ("far_reflect_row_shifted", {"fills_patch": _far_reflect_row_shifted}, True,
     _far_fill),
)


def compile_source(source: str, options: Sequence[str]):
    from meep_gpu.cuda_kernels import fused_electric_pair as family  # noqa: PLC0415
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

    Both ``(fprev * kms) - curl`` and ``(f[idx] * kms_u) + fu_new`` are FMA candidates,
    and the array path rounds them twice. The certified halves reach bit-identity ONLY
    under ``--fmad=false``, so a guard control that came back IDENTICAL would mean this
    comparison cannot see a rounding change at all, and every "identical" above would
    be worth much less. Scored on diverging, not merely reported.
    """
    from meep_gpu.cuda_kernels import fused_electric_pair as family  # noqa: PLC0415

    source = family.fused_electric_pair_source()
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


def run_product(results: Dict[str, Any], out_path: str,
                specs: Sequence[Dict[str, Any]], classes: Sequence[str],
                steps: int) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    total = len(specs) * len(classes)
    for spec in specs:
        for value_class in classes:
            case = run_case(spec, value_class, steps)
            cases.append(case)
            results["product"] = cases
            save(results, out_path)
            log(f"[product] {len(cases)}/{total} {spec['label']} {value_class} "
                f"eps={case.get('epsilon')} shape={case.get('shape')} "
                f"steps={case.get('steps_run')} "
                f"identical={case.get('bit_identical')} "
                f"diff={case.get('differing_words')} "
                f"launches={case.get('launch_counts', {}).get('launcher_reports')} "
                f"unmoved={case.get('movement_floor', {}).get('unmet')} "
                f"subnormal_operands={case.get('operand_census', {}).get('subnormals')} "
                f"({case.get('seconds', 0):.1f} s)")
    return cases


def run_mutations(results: Dict[str, Any], out_path: str, steps: int,
                  specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from meep_gpu.cuda_kernels import fused_electric_pair as family  # noqa: PLC0415

    base = family.fused_electric_pair_source()
    out: Dict[str, Any] = {}
    for name, (old, new, count), must_be_caught, applies in SOURCE_MUTATIONS:
        mutated, sites = needle(base, old, new, count)
        # AN AMBIGUOUS NEEDLE IS NOT ARMED EITHER, and that is stricter than the
        # magnetic sibling. A needle that matches twice with ``count = 1`` edits
        # whichever line ``str.replace`` reaches first, so the leg would report a
        # verdict about a statement nobody chose -- and if the two sites ever swapped
        # order the same leg would silently start measuring the other one.
        if not sites or mutated == base or (count > 0 and sites != count):
            out[name] = {"armed": False, "sites": sites,
                         "why": ("the needle matched nothing; the mutation and the "
                                 "kernel have drifted apart" if sites <= 1 else
                                 f"the needle matched {sites} sites but edits "
                                 f"{count}; it does not identify one statement")}
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
            out[name]["why"] = ("a leg reported no launch; that leg did not exercise "
                                "the mutation")
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

    # BOTH VALUE CLASSES, and the precondition must DISCRIMINATE.
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
        reasons.append("a subnormal-band case held no subnormal operand; the policy "
                       "legs measured the fixture, not the policy")
    if uniform and len(uniform_clean) != len(uniform):
        reasons.append("a uniform case held a subnormal operand; the precondition "
                       "cannot discriminate between the two classes")

    # EVERY WALL AXIS MUST HAVE BEEN LIVE somewhere. The D table is the off-diagonal,
    # and a sweep that only ever walls x cannot tell it from the B-side diagonal.
    walls = set()
    for case in product:
        walls.update(a for a, w in enumerate(case.get("zero_metal_axes", [])) if w)
    if walls != {0, 1, 2}:
        reasons.append(f"only wall axes {sorted(walls)} were ever live; the "
                       f"off-diagonal wall table is untested on the rest")

    # EVERY FILL MUST HAVE BEEN LIVE somewhere, on every axis. THE CARRY IS THIS
    # ROUND'S WHOLE CONTENT and on an unfolded row not one line of it executes, so a
    # sweep that admitted the fold and then never folded anything would release the
    # widening against evidence about a different configuration entirely.
    near_axes, far_axes = set(), set()
    for case in product:
        plan = case.get("fills", {})
        near_axes.update(a for a, live in enumerate(plan.get("near", ())) if live)
        far_axes.update(a for a, row in enumerate(plan.get("reflect", ())) if row >= 0)
    if near_axes != {0, 1, 2}:
        reasons.append(f"fill_symmetry_bc_D was live only on axes {sorted(near_axes)}; "
                       f"the near carry is untested on the rest")
    if not far_axes:
        reasons.append("fill_folded_far_ghosts_D was never live; the far carry, its "
                       "reflect row and its -phase weight went unexercised")
    phases = {float(p) for case in product for p in case.get("fills", {}).get(
        "phase", ()) if p}
    if phases != {1.0, -1.0}:
        reasons.append(f"mirror phases exercised were {sorted(phases)}; a kernel that "
                       f"baked +1 for the near weight is exact at phase +1 and wrong "
                       f"at -1")
    # ...and the ONE configuration where the driver's order between :3309 and :3310 is
    # observable at all. Without a row that is folded at an ODD phase on one near axis
    # and walled on the other, the pre-clear register is unmeasured: every other row
    # agrees with the naive ordering by arithmetic
    # (probe_cuda_electric_fill_carry_order.py, 9 and 11 differing words).
    cross_term = [case for case in product
                  if any(p == -1.0 for p in case.get("fills", {}).get("phase", ()))
                  and any(case.get("zero_metal_axes", []))]
    if not cross_term:
        reasons.append("no case was folded at an ODD phase AND walled; the near "
                       "carry's pre-clear register is what separates this family from "
                       "its B-side sibling and nothing here exercised it")

    # BOTH EPSILON SHAPES. An isotropic-only sweep cannot see a per-component
    # inverse-permittivity binding defect at all; an anisotropic-only one never
    # exercises the aliasing the certified signature deliberately permits.
    shapes = {case.get("epsilon") for case in product}
    if shapes != {"isotropic", "anisotropic"}:
        reasons.append(f"epsilon shapes scored were {sorted(shapes)}; a "
                       f"per-component inv_eps defect is invisible on an isotropic "
                       f"row and the aliased binding is unexercised without one")
    aliased = [c for c in product
               if c.get("distinct_inverse_epsilon_allocations") == 1]
    distinct = [c for c in product
                if c.get("distinct_inverse_epsilon_allocations") == 3]
    if not aliased:
        reasons.append("no case bound one inv_eps allocation three times; the "
                       "non-restrict group in the signature was never exercised")
    if not distinct:
        reasons.append("no case bound three distinct inv_eps allocations; the "
                       "per-component binding was never exercised")

    for leg_name in ("driver_order", "lift", "refusal"):
        leg = results.get(leg_name, {})
        if not leg.get("passed"):
            reasons.append(f"leg {leg_name} failed: "
                           f"{json.dumps(leg, default=str)[:400]}")

    # THE DEPOSIT IS THE POINT OF THIS PRODUCT, not a side observation: the predicate
    # ADMITS every repairable in-seam electric deposit, and 75 of its cell's 79 corpus
    # rows carry one. A run that did not measure one would be releasing an admission
    # it never exercised.
    deposit = results.get("deposit", {})
    if not deposit:
        reasons.append("no deposit case ran; the predicate admits an in-seam electric "
                       "source and nothing here measured one")
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
        reasons.append("no unbracketed null control ran; every deposit case above is "
                       "consistent with a deposit too small to see")
    for label, null in nulls.items():
        if not null.get("passed"):
            reasons.append(
                f"the UNBRACKETED launch on {label} did not diverge: every deposit "
                f"case there is consistent with a deposit too small to see "
                f"({json.dumps(null, default=str)[:300]})")

    # THE THREE-LEG IMAGE CLOSURE, and it is required rather than optional now that a
    # fold is admitted: the two fills run AFTER the injection, so a repair that wrote
    # only the deposit INDEX would leave every mirror image of it describing the
    # pre-injection field. B MUST DIVERGE -- a leg where it does not measured nothing.
    closure = results.get("deposit_image_closure", {})
    if not closure:
        reasons.append("no image-closure leg ran; the fold is admitted and a "
                       "point-only repair is untested on this seam")
    for label, leg in closure.items():
        if leg.get("control_is_vacuous"):
            reasons.append(
                f"image closure on {label} is VACUOUS: the closure holds "
                f"{leg.get('closure_cells')} cells against the deposit's "
                f"{leg.get('point_cells')}, so the point-only control is the shipped "
                f"repair and could not have diverged")
        if not leg.get("passed"):
            reasons.append(
                f"image closure on {label} failed: A={leg.get('A_differing_words')} "
                f"B={leg.get('B_differing_words')} C={leg.get('C_differing_words')} "
                f"(A must be 0, B and C must not be)")

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
        "epsilon_shapes": sorted(s for s in shapes if s),
        "wall_axes_exercised": sorted(walls),
        "deposit_cases": len(deposit),
        "deposit_cases_identical": sum(1 for leg in deposit.values()
                                       if leg.get("bit_identical")),
        "deposit_points_repaired": sum(int(leg.get("deposit_points_repaired") or 0)
                                       for leg in deposit.values()),
        "deposit_null_controls_diverged": {
            label: bool(leg.get("diverged")) for label, leg in nulls.items()},
        "mutations_armed": sum(1 for leg in armed.values() if leg.get("armed")),
        "mutations_must_be_caught": sum(
            1 for leg in armed.values()
            if leg.get("armed") and leg.get("must_be_caught")),
        "nulls_declared": sum(1 for leg in armed.values()
                              if leg.get("armed") and leg.get("must_be_caught") is False)
        + sum(1 for leg in results.get("host_mutations", {}).values()
              if leg.get("must_be_caught") is False),
        "host_mutations": len(results.get("host_mutations", {})),
        "claim": (f"the shipped CUDA fused electric pair "
                  f"(step_D -> fill_symmetry_bc_D -> zero_metal_D -> "
                  f"fill_folded_far_ghosts_D -> update_E in ONE launch) leaves every "
                  f"stored volume BYTE-IDENTICAL to stepping's own eleven-pass driver "
                  f"step, per complete step over {steps} consecutive steps, on every "
                  f"configuration this sweep scored, under the float32 subnormal "
                  f"policy this run installed -- including complete steps carrying a "
                  f"real electric source inside the seam, through the shipped "
                  f"composer's two repair plans, and including FOLDED rows at both "
                  f"terminations, both plane parities, one two three folded axes, and "
                  f"the odd-phase fold-plus-wall row where the driver's order between "
                  f"fill_symmetry_bc_D and zero_metal_D is observable"),
        "does_not_claim": [
            "NO THROUGHPUT. This gate times nothing and licenses no speed claim.",
            "NO DISPATCH. Nothing in meep_gpu imports cuda_kernels; this product is "
            "not registered on any arm and a verdict here does not wire it.",
            "NOTHING ABOUT A COMPLEX, CYLINDRICAL, CONDUCTIVE, BFAST, SPECIAL-KZ, "
            "DISPERSIVE, NONLINEAR or OFF-DIAGONAL run: each is refused BY NAME and "
            "none was swept.",
            "NOTHING ABOUT A CYLINDRICAL FOLD. The r = 0 axis is a different ghost "
            "rule and deposit_repair refuses it by name; every folded row here is "
            "Cartesian.",
            "NOTHING ABOUT THE B/H SEAM, which is a different product with the "
            "opposite Yee sub-lattice pairing and its own gate.",
            "THE ORACLE IS stepping ON CuPy, NOT CPU MEEP. This is a byte-identity "
            "claim against the package's own array path -- the same oracle both "
            "halves were certified against -- and it is not a physics claim against "
            "stock MEEP.",
            "THE MUTATION LEGS RUN ON THE UNIFORM CLASS ONLY. A mutation leg answers "
            "'can this gate see this defect at all'; multiplying it by the value "
            "classes buys repetitions of that answer rather than a second question.",
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
        "gate": "cuda_fused_electric_pair",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "question": ("does ONE launch of fused_electric_pair_pml_real leave every "
                     "stored volume byte-identical to stepping's step_D -> "
                     "zero_metal_D -> update_E inside a complete driver step, "
                     "including with an electric source deposited in the seam?"),
        "budget_steps": args.steps,
        "movement_floor_note": MOVEMENT_FLOOR_NOTE,
    }
    # THE MODULE THIS GATE CERTIFIES, STAMPED OFF THE MODULE ITSELF. Every sibling
    # gate writes this block and ``record_fused_electric_pair.py`` requires it by
    # name -- it derives a block's ``certified_kernels`` from
    # ``module.kernel_symbols`` and refuses to default one, which is the right
    # stance: a record assembled from defaults is a record about nothing. This gate
    # never wrote it, so its verdict could be cut and never transcribed, and the
    # certification row naming this weld resolved to an absent ledger entry.
    from meep_gpu.cuda_kernels import fused_electric_pair as _family  # noqa: PLC0415
    results["module"] = {
        "family": _family.FAMILY,
        "kernel": _family.KERNEL_NAME,
        "kernel_symbols": [_family.KERNEL_NAME],
        "replaces": list(_family.REPLACES),
        "carries_deposit_repair": bool(_family.CARRIES_DEPOSIT_REPAIR),
        "certified_kernels": list(_family.CERTIFIED_KERNELS),
        "uncertified_kernels": dict(_family.UNCERTIFIED_KERNELS),
        "lift_edits": len(_family.LIFT_EDITS),
    }
    if cp is None:
        log("[fatal] CuPy did not import; this gate needs a device")
        results["status"] = "refused: no CuPy"
        save(results, args.out)
        return 2

    if args.import_meep_for_host_policy:
        results["meep_host_import"] = probe.import_meep_for_host_policy()
    # THE OBSERVER GOES IN BEFORE THE POLICY: under 'keep' the policy's strip wraps
    # it, so it records the option tuple NVRTC was really given.
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
    save(results, args.out)

    # WARM THE UNION, not just the product sweep. Every leg below drives kernels
    # through the same shared memo, and a kernel first compiled inside a counted
    # region is launched and NOT counted -- so a spec that only the separate control
    # or a deposit leg visits has to be warmed here too or that leg's launch total
    # comes back short.
    warm_labels = list(dict.fromkeys(
        [spec["label"] for spec in specs]
        + list(SEPARATE_CONTROL_LABELS) + list(DEPOSIT_SPEC_LABELS)
        + list(IMAGE_CLOSURE_SPEC_LABELS) + list(MUTATION_SPEC_LABELS)))
    _warm_memo([by_label[label] for label in warm_labels])
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

    # THE DEPOSIT LEGS. Two carried rows per spec -- a POINT source and an EXTENDED
    # one, so the repair's per-point loop runs at many points and not only at four --
    # and the unbracketed null control, which must diverge.
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
                f"magnitude={leg.get('deposit_magnitude')} "
                f"repaired={leg.get('deposit_points_repaired')} "
                f"plans={leg.get('installed_plans')}")
    results["deposit_null_control"] = {
        label: leg_deposit_null_control(by_label[label], args.steps)
        for label in DEPOSIT_SPEC_LABELS}
    save(results, args.out)
    for label, leg in results["deposit_null_control"].items():
        log(f"[deposit] null control {label} diverged={leg['diverged']} at step "
            f"{leg['first_divergence']}")

    # THE THREE-LEG IMAGE CLOSURE. Folded rows only: on an unfolded grid
    # ``fill_image_rules`` is empty and the point-only control IS the shipped repair.
    results["deposit_image_closure"] = {}
    for label in IMAGE_CLOSURE_SPEC_LABELS:
        leg = leg_deposit_image_closure(by_label[label], args.steps)
        results["deposit_image_closure"][label] = leg
        save(results, args.out)
        log(f"[closure] {label}: rules={leg['fill_image_rules']} "
            f"point={leg['point_cells']} closure={leg['closure_cells']} "
            f"A={leg['A_differing_words']} B={leg['B_differing_words']} "
            f"C={leg['C_differing_words']} "
            f"vacuous={leg['control_is_vacuous']} passed={leg['passed']}")

    results["guard_control"] = leg_guard_control(by_label["wall_xyz"], args.steps)
    save(results, args.out)
    log(f"[guard] unguarded source diverged="
        f"{results['guard_control']['diverged']} at step "
        f"{results['guard_control']['first_divergence']}")

    if not args.skip_mutations:
        run_mutations(results, args.out, args.steps, mutation_specs)
        run_host_mutations(results, args.out, args.steps, mutation_specs)

    # THE DISARM CHECK. The identical harness, the identical cases, the SHIPPED bytes
    # and no patch. A divergence here would mean every "caught" above is a harness
    # that diverges on its own.
    disarm_cases = [run_case(spec, "uniform", args.steps, label_suffix="|disarm")
                    for spec in mutation_specs]
    results["disarm"] = {
        "passed": all(case.get("passed") for case in disarm_cases),
        "cases": disarm_cases,
        "differing_words": [case.get("differing_words") for case in disarm_cases],
    }
    log(f"[disarm] shipped bytes on the mutation specs: {results['disarm']['passed']}")

    results["nvrtc_binary_report"] = probe.nvrtc_binary_report()
    results["source_sha256"] = {
        "gate": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "family": hashlib.sha256(
            open(os.path.join(_REPO_API, "meep_gpu", "cuda_kernels",
                              "fused_electric_pair.py"), "rb").read()).hexdigest(),
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
